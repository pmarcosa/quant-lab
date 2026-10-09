"""The live weekly cycle, end to end, against a stand-in gateway.

Everything here runs the same code a real session runs; only the gateway and the
clock are stand-ins. The failure-path tests matter more than the happy path:
each one is a way real money could be spent on the wrong trade, and each one
must end in a refusal that says why.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("ib_async")

from contracts.errors import ContractViolation  # noqa: E402
from contracts.live import DegradationState, EventKind, TradingMode  # noqa: E402
from execution.ibkr import IBKRBroker  # noqa: E402
from runtime.config import LiveConfig, RiskSettings, StrategySettings  # noqa: E402
from runtime.live import LiveSession  # noqa: E402
from tests.fake_gateway import FakeGateway  # noqa: E402
from tests.live_fixtures import STEADY, Clock, append_week, build_market  # noqa: E402

SATURDAY = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


def config(tmp_path, **risk):
    return LiveConfig(
        strategy_id="momentum",
        mode=TradingMode.PAPER, account="DU1234567", sleeve_capital=100_000.0,
        strategy=StrategySettings(params={"rebalance_weeks": 1, "top_n": 2, "lookback_weeks": 13, **STEADY}),
        risk=RiskSettings(**{"stop_distance": 0.12, "max_order_fraction": 0.6, **risk}),
        state_dir=tmp_path / "state",
    )


@pytest.fixture
def world(tmp_path):
    frames = build_market(tmp_path / "store")
    gateway = FakeGateway(cash=150_000.0)
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    clock = Clock(SATURDAY)
    session = LiveSession(config(tmp_path), broker, tmp_path / "store", clock=clock)
    return session, gateway, clock, frames, tmp_path


def opens_for(frames, session):
    """The opening prices the auction fills at: the latest close, nudged."""
    return {s: float(f["close"].iloc[-1]) * 1.002 for s, f in frames.items()}


def run_first_rotation(world):
    session, gateway, clock, frames, _ = world
    session.init()
    session.sync()
    proposal = session.propose()
    session.approve(proposal.proposal_id, proposal.proposal_id)
    clock.advance(days=2)  # Monday
    gateway.opening_auction(opens_for(frames, session))
    report = session.sync()
    return proposal, report


# -- the happy path ----------------------------------------------------------


def test_a_full_week_from_opening_to_protected_positions(world):
    session, gateway, clock, frames, _ = world
    proposal, report = run_first_rotation(world)

    assert proposal.rotation
    assert {o.intent.side.value for o in proposal.orders} == {"buy"}
    assert len(proposal.orders) == 2, "top two names"
    assert all(o.intent.time_in_force.value == "opg" for o in proposal.orders), (
        "rotation orders go to the opening auction, as the backtest assumes"
    )

    assert report.new_fills == 2
    assert report.reconciliation.status == "ok", report.reconciliation.findings
    assert report.stops_placed == 2
    assert report.state is DegradationState.NORMAL

    book = session.book()
    held = {str(p.instrument): p.quantity for p in gateway_positions(session)}
    assert {str(i): p.quantity for i, p in book.positions.items()} == held
    for working in session.broker.working_orders():
        fill = book.positions[working.instrument].average_cost
        assert working.stop_price == pytest.approx(fill * 0.88, abs=0.011), (
            "the stop is anchored at the rotation's fill price"
        )


def gateway_positions(session):
    return session.broker.positions(session.portfolio)


def test_nothing_is_sent_without_the_typed_code(world):
    session, gateway, *_ = world
    session.init()
    session.sync()
    proposal = session.propose()
    with pytest.raises(ContractViolation, match="type exactly"):
        session.approve(proposal.proposal_id, "yes")
    assert gateway.placed == 0


def test_live_mode_needs_a_longer_confirmation(tmp_path):
    cfg = LiveConfig(
        strategy_id="momentum",
        mode=TradingMode.LIVE, account="U7654321", sleeve_capital=1.0,
        state_dir=tmp_path,
    )
    session = LiveSession(cfg, None, tmp_path)
    assert session.confirmation_phrase("PABC123") == "LIVE PABC123"


def test_the_journal_rebuilds_the_same_book_after_a_restart(world):
    session, gateway, clock, frames, tmp_path = world
    run_first_rotation(world)
    restarted = LiveSession(session.config, session.broker, tmp_path / "store", clock=clock)
    assert restarted.book() == session.book()


# -- the failure paths -------------------------------------------------------


def test_a_proposal_cannot_be_approved_after_the_sleeve_changed(world):
    """A stop firing between proposal and approval changes every quantity."""
    session, gateway, clock, frames, tmp_path = world
    run_first_rotation(world)
    frames = append_week(tmp_path / "store", frames, clock.advance(days=5))
    session.sync()
    proposal = session.propose()
    held = next(iter(session.book().positions))
    gateway.trigger_stops({str(held): 0.01})
    session.sync()
    with pytest.raises(ContractViolation, match="has changed"):
        session.approve(proposal.proposal_id, proposal.proposal_id)


def test_an_expired_proposal_is_refused(world):
    session, gateway, clock, *_ = world
    session.init()
    session.sync()
    proposal = session.propose()
    clock.advance(hours=session.config.proposal_ttl_hours + 1)
    with pytest.raises(ContractViolation, match="expired"):
        session.approve(proposal.proposal_id, proposal.proposal_id)


def test_only_the_latest_proposal_can_be_approved(world):
    session, *_ = world
    session.init()
    session.sync()
    first = session.propose()
    session.propose()
    with pytest.raises(ContractViolation, match="not the pending proposal"):
        session.approve(first.proposal_id, first.proposal_id)


def test_a_reconciliation_mismatch_halts_the_system(world):
    """The broker holding less than the sleeve believes is not a warning."""
    session, gateway, clock, frames, _ = world
    run_first_rotation(world)
    held = str(next(iter(session.book().positions)))
    gateway._positions[held] -= 5
    report = session.sync()
    assert report.reconciliation.status == "mismatch"
    assert session.state() is DegradationState.HALTED
    with pytest.raises(ContractViolation, match="halted"):
        session.propose()


def test_a_halt_is_cleared_only_by_a_person_after_the_cause_is_fixed(world):
    session, gateway, clock, frames, _ = world
    run_first_rotation(world)
    held = str(next(iter(session.book().positions)))
    gateway._positions[held] -= 5
    session.sync()
    with pytest.raises(ContractViolation, match="mismatch"):
        session.clear(DegradationState.NORMAL, "tried to clear without fixing anything")
    broker_quantity = gateway._positions[held]
    session.adjust(held, broker_quantity, "five shares sold by hand in TWS on Monday")
    with pytest.raises(ContractViolation, match="real reason"):
        session.clear(DegradationState.NORMAL, "ok")
    session.clear(DegradationState.NORMAL, "sleeve adjusted to the broker after a manual sale")
    assert session.state() is DegradationState.NORMAL
    last = session.journal.last(EventKind.STATE_CHANGE)
    assert last.payload["by"] == "person"


def test_the_system_never_moves_itself_up_the_ladder(world):
    session, *_ = world
    session.init()
    session.degrade(DegradationState.REDUCE_ONLY, "test")
    assert session.degrade(DegradationState.NORMAL, "test") is DegradationState.REDUCE_ONLY


def test_reduce_only_withholds_every_buy(world):
    session, gateway, *_ = world
    session.init()
    session.sync()
    session.degrade(DegradationState.REDUCE_ONLY, "drawdown in the tail of the backtest")
    proposal = session.propose()
    assert proposal.orders == ()
    assert any("withheld" in f for f in proposal.findings)


def test_a_halted_system_can_still_propose_an_orderly_exit(world):
    session, gateway, clock, frames, _ = world
    run_first_rotation(world)
    session.degrade(DegradationState.HALTED, "changepoint probability above one half")
    session.sync()
    exit_plan = session.propose(liquidate=True)
    assert exit_plan.liquidation
    assert {o.intent.side.value for o in exit_plan.orders} == {"sell"}
    assert {str(o.intent.instrument) for o in exit_plan.orders} == {
        str(i) for i in session.book().positions
    }


def test_an_order_too_large_for_the_sleeve_is_refused(tmp_path):
    frames = build_market(tmp_path / "store")
    gateway = FakeGateway(cash=150_000.0)
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    cfg = config(tmp_path, max_order_fraction=0.3)
    session = LiveSession(cfg, broker, tmp_path / "store", clock=Clock(SATURDAY))
    session.init()
    session.sync()
    with pytest.raises(ContractViolation, match="max_order_fraction"):
        session.propose()
    del frames


def test_a_name_a_stop_closed_is_not_bought_back_before_the_next_rotation(world):
    session, gateway, clock, frames, tmp_path = world
    run_first_rotation(world)
    stopped = str(next(iter(session.book().positions)))
    gateway.trigger_stops({stopped: 0.01})
    report = session.sync()
    assert any("stop filled" in n for n in report.notes)
    frames = append_week(tmp_path / "store", frames, clock.advance(days=5))
    session.sync()
    proposal = session.propose()
    bought = {str(o.intent.instrument) for o in proposal.orders if o.intent.side.value == "buy"}
    assert stopped not in bought


@pytest.mark.parametrize("change", ["tighter", "off_then_on"])
def test_a_replaced_stop_is_really_placed(world, change):
    """A stop cancelled and placed again for the same rotation must rest at the broker.

    Its id is derived from the decision, which has not changed. Reusing it made
    the broker return the cancelled order instead of placing the new one:
    ``place_stops`` reported success, the journal said "placed", and the
    positions were unprotected.
    """
    session, gateway, clock, frames, tmp_path = world
    run_first_rotation(world)
    broker = session.broker
    before = {str(w.instrument) for w in broker.working_orders()}
    assert len(before) == 2

    def reopened(distance):
        return LiveSession(config(tmp_path, stop_distance=distance), broker,
                           tmp_path / "store", clock=clock)

    if change == "tighter":
        placed, cancelled = reopened(0.10).place_stops()
        assert (placed, cancelled) == (2, 2)
        expected = 0.90
    else:
        assert reopened(0).place_stops() == (0, 2)
        assert not broker.working_orders()
        reopened(0.12).place_stops()
        expected = 0.88

    working = broker.working_orders()
    assert {str(w.instrument) for w in working} == before, "every position protected again"
    book = session.book()
    for w in working:
        assert w.stop_price == pytest.approx(
            book.positions[w.instrument].average_cost * expected, abs=0.011
        )


def _hand_trade(gateway, symbol="DDD", reference=""):
    import ib_async

    manual = gateway.placeOrder(
        ib_async.Stock(symbol, "SMART", "USD"),
        ib_async.Order(action="BUY", totalQuantity=7, orderType="MKT", orderRef=reference),
    )
    gateway.execute(manual, 50.0)


def test_a_trade_made_by_hand_in_a_dedicated_account_halts(world):
    """The account is the strategy's. A position nobody recorded is an incident."""
    session, gateway, *_ = world
    session.init()
    _hand_trade(gateway)
    report = session.sync()
    assert report.new_fills == 0
    assert "DDD" not in {str(i) for i in session.book().positions}
    assert any(f.instrument == "DDD" and f.level == "mismatch" for f in report.reconciliation.findings)
    assert session.state() is DegradationState.HALTED


def test_a_trade_made_by_hand_in_a_shared_account_is_reported(tmp_path):
    from dataclasses import replace

    build_market(tmp_path / "store")
    gateway = FakeGateway(cash=150_000.0)
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    session = LiveSession(replace(config(tmp_path), account_scope="shared"), broker,
                          tmp_path / "store", clock=Clock(SATURDAY))
    session.init()
    _hand_trade(gateway)
    report = session.sync()
    assert report.new_fills == 0
    assert any(f.instrument == "DDD" and f.level == "warn" for f in report.reconciliation.findings)
    assert session.state() is DegradationState.NORMAL


def test_another_strategys_fills_are_never_claimed(world):
    """Two strategies' order ids start differently; a fill is claimed by prefix."""
    session, gateway, *_ = world
    session.init()
    _hand_trade(gateway, reference="ql-other.0123456789abcdef0123")
    report = session.sync()
    assert report.new_fills == 0


def test_the_sleeve_cannot_open_twice(world):
    session, *_ = world
    session.init()
    with pytest.raises(ContractViolation, match="opens once"):
        session.init()


def test_adopting_a_position_the_account_does_not_hold_is_refused(world):
    session, *_ = world
    with pytest.raises(ContractViolation, match="does not hold"):
        session.init(adopt=["AAA"])


def test_adopted_positions_take_their_place_in_the_sleeve(world):
    session, gateway, *_ = world
    gateway.hold("CCC", 100, 45.0)
    from contracts.identifiers import InstrumentId

    book = session.init(adopt=["CCC"])
    assert book.quantity(InstrumentId("CCC")) == 100
    assert book.positions[InstrumentId("CCC")].average_cost == 45.0, "the broker's basis"
    assert book.cash < 100_000.0, "the adopted value is part of the sleeve's capital"
    report = session.sync()
    assert report.stops_placed == 1, "an adopted position is protected like any other"


def test_proposing_on_stale_data_is_refused(world):
    session, gateway, clock, *_ = world
    session.init()
    clock.advance(days=12)
    session.sync()
    with pytest.raises(ContractViolation, match="ql data refresh"):
        session.propose()


def test_a_rejection_needs_a_reason_and_is_recorded(world):
    session, *_ = world
    session.init()
    session.sync()
    proposal = session.propose()
    with pytest.raises(ContractViolation, match="real reason"):
        session.reject(proposal.proposal_id, "no")
    session.reject(proposal.proposal_id, "earnings on Tuesday; waiting a week by choice")
    assert session.pending_proposal() is None
