"""Leverage: the rule that sets it, what it costs to carry, and the live checks.

The rule is the project's expert's (see ``risk/leverage.py``): a hard cap, an
optional tail-risk target, convex de-leveraging in drawdown, cut-fast /
rebuild-slow, and the margin cushion overriding everything. Each test pins one
of those clauses, and the backtest tests pin that borrowing is paid for.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from contracts.errors import ContractViolation
from contracts.execution import Charge, ChargeKind, Fill, Side
from contracts.identifiers import InstrumentId, PortfolioId, TenantId
from contracts.risk import LeverageState
from contracts.targets import TargetIntent
from engine.accounting import Book
from engine.decide import leverage_scale
from engine.financing import FinancingModel, gross_leverage
from risk.leverage import LeverageSchedule, expected_shortfall

T0 = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
AAA, BBB = InstrumentId("AAA"), InstrumentId("BBB")
BOOK = PortfolioId(TenantId("user"), "lev")


def state(equity=(100.0,), base=(), previous=1.0, cushion=None, bars_per_week=1.0):
    return LeverageState(equity=equity, base_returns=base, previous=previous,
                         cushion=cushion, bars_per_week=bars_per_week)


# -- the rule ------------------------------------------------------------------------


def test_a_fixed_target_is_reached_step_by_step_and_never_exceeds_the_cap():
    rule = LeverageSchedule(target=1.3, maximum=1.3)
    level, equity = 1.0, [100.0]
    for _ in range(10):
        equity.append(equity[-1] * 1.01)
        level = rule(state(equity=equity, previous=level))
        assert level <= 1.3
    assert level == pytest.approx(1.3), "rebuilt at 0.05x a week"
    assert rule(state(equity=[100.0, 101.0], previous=1.0)) == pytest.approx(1.05)


def test_the_step_up_is_per_calendar_week_whatever_the_bar_size():
    rule = LeverageSchedule(target=1.3, maximum=1.3)
    daily = rule(state(equity=[100.0, 101.0], previous=1.0, bars_per_week=5.0))
    assert daily == pytest.approx(1.01), "0.05x a week is 0.01x a trading day"


def test_leverage_does_not_rise_while_equity_is_not_improving():
    rule = LeverageSchedule(target=1.5, maximum=1.5)
    equity = [100.0, 99.0, 98.0, 97.0, 96.0]
    assert rule(state(equity=equity, previous=1.2)) == pytest.approx(1.2)


def test_drawdown_cuts_leverage_at_once_and_convexly():
    rule = LeverageSchedule(target=1.5, maximum=1.5)
    # 10% drawdown: nothing yet. 22.5% (halfway to 35%): severity 0.5, cut by 0.25.
    assert rule(state(equity=[100.0, 90.0], previous=1.5)) == pytest.approx(1.5)
    halfway = rule(state(equity=[100.0, 77.5], previous=1.5))
    assert halfway == pytest.approx(1.0 + 0.5 * (1 - 0.5 ** 2))
    assert rule(state(equity=[100.0, 60.0], previous=1.5)) == pytest.approx(1.0), (
        "at 35% drawdown the borrowing is gone; the unlevered strategy is left"
    )


def test_the_floor_below_one_restores_the_experts_whole_book_version():
    rule = LeverageSchedule(target=1.5, maximum=1.5, floor=0.1)
    assert rule(state(equity=[100.0, 60.0], previous=1.5)) == pytest.approx(0.1)


def test_a_thin_margin_cushion_overrides_everything():
    rule = LeverageSchedule(target=1.5, maximum=1.5)
    assert rule(state(equity=[100.0, 101.0], previous=1.4, cushion=0.20)) == 1.0, (
        "below the critical cushion the borrowing goes"
    )
    assert rule(state(equity=[100.0, 101.0], previous=1.2, cushion=0.30)) == pytest.approx(1.2), (
        "below the warning line nothing is added"
    )


def test_tail_risk_targeting_sets_leverage_from_the_strategys_shortfall():
    rng = np.random.default_rng(3)
    calm = rng.normal(0.002, 0.01, 200)
    wild = rng.normal(0.002, 0.04, 200)
    rule = LeverageSchedule(target=1.0, maximum=2.0, cvar_target=0.03)
    assert rule.base_level(calm) > 1.0 > rule.base_level(wild)
    assert rule.base_level(calm) == pytest.approx(0.03 / expected_shortfall(calm[-104:]))
    assert rule.base_level(calm[:10]) == 1.0, "too little history: the fixed target"


def test_an_impossible_schedule_is_refused():
    with pytest.raises(ContractViolation, match="target"):
        LeverageSchedule(target=1.6, maximum=1.5)
    with pytest.raises(ContractViolation, match="drawdown"):
        LeverageSchedule(target=1.2, maximum=1.5, drawdown_start=0.4, drawdown_full=0.3)


# -- how the engine applies it ------------------------------------------------------------


def target(weights, rotated=True):
    return TargetIntent(weights=weights, horizon_bars=1, as_of=T0,
                        diagnostics={"rotated": 1.0 if rotated else 0.0})


def test_a_rotation_is_scaled_to_the_leverage():
    assert leverage_scale(target({AAA: 0.5, BBB: 0.5}), 1.3) == 1.3


def test_a_hold_is_left_alone_unless_the_leverage_was_just_cut():
    held = target({AAA: 0.65, BBB: 0.65}, rotated=False)  # a 1.3x book, restated
    assert leverage_scale(held, 1.3, previous=1.3) == 1.0, "not levered twice"
    assert leverage_scale(held, 1.5, previous=1.3) == 1.0, "increases wait for the rotation"
    assert leverage_scale(held, 1.1, previous=1.3) == pytest.approx(1.1 / 1.3), "cut at once"


def test_an_unlevered_hold_that_drifted_is_not_trimmed():
    drifted = target({AAA: 0.51, BBB: 0.51}, rotated=False)
    assert leverage_scale(drifted, 1.0, previous=1.0) == 1.0


# -- what carrying it costs -----------------------------------------------------------------


def book_with(quantity_a, cash):
    fill = Fill(client_order_id="x", instrument=AAA, side=Side.BUY if quantity_a > 0 else Side.SELL,
                quantity=abs(quantity_a), price=100.0, at=T0)
    opening = Book(portfolio=BOOK, cash=cash + quantity_a * 100.0, as_of=T0)
    return opening.apply(fill)


def test_interest_is_charged_on_the_debit_for_the_calendar_days_held():
    book = book_with(1300, -30_000.0)  # 130k of stock on 100k of equity
    charges = FinancingModel(margin_rate=0.05).charges(book, {AAA: 100.0}, T0, T0 + timedelta(days=7))
    (interest,) = charges
    assert interest.kind is ChargeKind.MARGIN_INTEREST
    assert interest.amount == pytest.approx(30_000 * 0.05 * 7 / 365)
    after = book.charge(interest)
    assert after.cash == pytest.approx(-30_000 - interest.amount)
    assert after.financing_paid == pytest.approx(interest.amount)
    assert after.positions == book.positions, "a carrying cost moves no shares"


def test_a_short_pays_the_lender_and_a_paid_up_book_pays_nothing():
    short = book_with(-500, 100_000.0)
    (fee,) = FinancingModel(borrow_fee=0.02).charges(short, {AAA: 100.0}, T0, T0 + timedelta(days=365))
    assert fee.kind is ChargeKind.BORROW_FEE and fee.amount == pytest.approx(1_000.0)
    paid_up = book_with(900, 10_000.0)
    assert FinancingModel().charges(paid_up, {AAA: 100.0}, T0, T0 + timedelta(days=30)) == ()


def test_the_modelled_cushion_follows_reg_t_maintenance():
    levered = book_with(1500, -50_000.0)  # 1.5x gross
    model = FinancingModel()
    assert gross_leverage(levered, {AAA: 100.0}) == pytest.approx(1.5)
    assert model.cushion(levered, {AAA: 100.0}) == pytest.approx(1 - 0.25 * 1.5)


def test_a_charge_cannot_be_negative_or_backdated():
    with pytest.raises(ContractViolation, match="negative"):
        Charge(at=T0, amount=-1.0, kind=ChargeKind.MARGIN_INTEREST)
    book = book_with(100, 0.0)
    with pytest.raises(ContractViolation, match="before the book"):
        book.charge(Charge(at=T0 - timedelta(days=1), amount=1.0, kind=ChargeKind.MARGIN_INTEREST))


# -- a levered backtest, end to end ----------------------------------------------------------


def test_a_levered_backtest_holds_more_than_its_equity_and_pays_for_it(tmp_path):
    from contracts.identifiers import RunId
    from engine.decide import SizingPolicy
    from execution.simulated import CostModel
    from risk.rules import GrossExposureLimit, NetExposureLimit, RiskSupervisor
    from runtime.research import run_once
    from runtime.strategies import build_strategy
    from runtime.wiring import load_market
    from tests.live_fixtures import build_market

    build_market(tmp_path, weeks=120)
    market = load_market(tmp_path)
    strategy = build_strategy(
        "weekly-momentum", {"rebalance_weeks": 1, "top_n": 2, "lookback_weeks": 13}, market
    )
    common = dict(
        market=market, strategy=strategy, schedule=list(market.schedule)[30:],
        costs=CostModel(0.0, 0.0), policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )
    plain = run_once(run=RunId("plain"), **common, financing=FinancingModel())
    levered = run_once(
        run=RunId("lev"), **common,
        supervisor=RiskSupervisor(rules=(GrossExposureLimit(1.3), NetExposureLimit(-1, 1.3))),
        leverage=LeverageSchedule(target=1.3, maximum=1.3), financing=FinancingModel(),
    )
    assert plain.financing_paid < 1.0, "a paid-up book pays at most for a gap's cents"
    assert max(s.gross_after for s in levered.steps) == pytest.approx(1.3, abs=0.05)
    assert max(s.gross_after for s in levered.steps) <= 1.3 + 0.02, "the cap holds"
    assert levered.financing_paid > 0, "borrowed cash is not free"
    assert levered.min_cushion is not None and levered.min_cushion < plain.min_cushion
    assert len(levered.base_returns()) > 20


# -- configuration ----------------------------------------------------------------------------


def _config(**kw):
    from contracts.live import TradingMode
    from runtime.config import LiveConfig

    return LiveConfig(strategy_id="m", mode=TradingMode.PAPER, account="DU1234567",
                      sleeve_capital=100_000.0, **kw)


def test_leverage_above_the_gross_cap_is_refused():
    from runtime.config import LeverageSettings, RiskSettings

    with pytest.raises(ContractViolation, match="above risk.max_gross"):
        _config(leverage=LeverageSettings(target=1.3)).validate()
    with pytest.raises(ContractViolation, match="max_net"):
        _config(leverage=LeverageSettings(target=1.3),
                risk=RiskSettings(max_gross=1.3, max_net=1.0)).validate()
    ok = _config(leverage=LeverageSettings(target=1.3), risk=RiskSettings(max_gross=1.3)).validate()
    assert ok.risk.net_cap == 1.3, "an empty max_net follows the gross cap"


# -- live -------------------------------------------------------------------------------------


@pytest.fixture
def levered(tmp_path):
    pytest.importorskip("ib_async")
    from contracts.live import TradingMode
    from execution.ibkr import IBKRBroker
    from runtime.config import LeverageSettings, RiskSettings, StrategySettings
    from runtime.live import LiveSession
    from tests.fake_gateway import FakeGateway
    from tests.live_fixtures import Clock, build_market

    frames = build_market(tmp_path / "store")
    gateway = FakeGateway(cash=100_000.0)
    gateway.prices = {s: float(f["close"].iloc[-1]) for s, f in frames.items()}
    config = _config(
        strategy=StrategySettings(params={"rebalance_weeks": 1, "top_n": 2, "lookback_weeks": 13}),
        risk=RiskSettings(max_gross=1.3, stop_distance=0.12, max_order_fraction=0.7),
        leverage=LeverageSettings(target=1.3), state_dir=tmp_path / "state",
    )
    from dataclasses import replace

    config = replace(config, strategy_id="lev").validate()
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0,
                        order_prefix="ql-lev.")
    clock = Clock(datetime(2026, 9, 19, 10, tzinfo=timezone.utc))
    session = LiveSession(config, broker, tmp_path / "store", clock=clock)
    return session, gateway, clock, frames


def test_a_live_proposal_is_sized_to_the_schedules_leverage(levered):
    session, gateway, clock, frames = levered
    session.init()
    session.sync()
    proposal = session.propose()
    assert proposal.leverage == pytest.approx(1.05), "rebuilt from 1.0 at 0.05x a week"
    bought = sum(o.estimated_value for o in proposal.orders if o.intent.side is Side.BUY)
    assert bought > proposal.equity, "more than the sleeve's own cash: borrowing"
    assert bought == pytest.approx(1.05 * proposal.equity, rel=0.03)
    from contracts.live import EventKind

    event = session.journal.last(EventKind.PROPOSAL)
    assert event.payload["leverage"] == pytest.approx(1.05)


def test_borrowing_the_account_cannot_margin_is_refused(levered):
    session, gateway, *_ = levered
    gateway.long_margin = 1.0  # a cash account: every dollar bought needs a dollar
    session.init()
    session.sync()
    with pytest.raises(ContractViolation, match="margin check failed"):
        session.propose()


def test_the_sleeve_pays_interest_on_what_it_borrowed(levered):
    from contracts.live import EventKind

    session, gateway, clock, frames = levered
    session.init()
    session.sync()
    proposal = session.propose()
    session.approve(proposal.proposal_id, proposal.proposal_id)
    clock.advance(days=2)
    gateway.opening_auction({s: float(f["close"].iloc[-1]) for s, f in frames.items()})
    session.sync()
    assert session.book().cash < 0
    clock.advance(days=7)
    session.sync()
    charges = session.journal.events(EventKind.FINANCING)
    assert charges and charges[-1].payload["kind"] == "margin_interest"
    assert session.book().financing_paid == pytest.approx(
        sum(e.payload["amount"] for e in charges)
    ), "the journal replays the charges into the sleeve"


def test_a_thin_margin_cushion_stops_new_exposure(levered):
    from contracts.live import DegradationState

    session, gateway, *_ = levered
    gateway.cushion = 0.18
    session.init()
    report = session.sync()
    assert session.state() is DegradationState.REDUCE_ONLY
    assert any("cushion" in n for n in report.notes)
    assert session.broker.account_snapshot().cushion == pytest.approx(0.18)
