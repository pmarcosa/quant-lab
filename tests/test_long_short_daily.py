"""A daily long/short strategy through the live cycle, against the stand-in gateway.

The deployed strategy is weekly and long-only. Nothing in the machinery may
assume either, so these tests drive the same live session with a toy strategy
(``tests/toy_strategies.py``) that decides on daily bars and holds one long and
one short. What they pin down is what changes when a book can be short:

* a short is only sold when the broker says the shares can be borrowed and the
  account can margin it;
* its protective stop is a *buy* stop above the anchor;
* reduce-only and liquidation cover shorts instead of adding to them;
* reconciliation compares signed positions, so a long where a short should be
  is a mismatch, not a rounding difference.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

pytest.importorskip("ib_async")

from contracts.errors import ContractViolation  # noqa: E402
from contracts.identifiers import InstrumentId  # noqa: E402
from contracts.live import DegradationState, EventKind, TradingMode  # noqa: E402
from contracts.temporal import BarInterval  # noqa: E402
from execution.ibkr import IBKRBroker  # noqa: E402
from runtime.config import LiveConfig, RiskSettings  # noqa: E402
from runtime.live import LiveSession  # noqa: E402
from tests.fake_gateway import FakeGateway  # noqa: E402
from tests.live_fixtures import Clock, append_day, build_daily_market  # noqa: E402
from tests.toy_strategies import DailyLongShort  # noqa: E402

#: The morning after Friday's session: the latest complete daily bar is Friday's.
SATURDAY = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
LONG, SHORT = "AAA", "EEE"  # the best and the worst trailing performers


def config(tmp_path, **risk):
    return LiveConfig(
        strategy_id="toyls",
        mode=TradingMode.PAPER, account="DU1234567", sleeve_capital=100_000.0,
        risk=RiskSettings(**{
            "allow_short": True, "stop_distance": 0.10, "max_order_fraction": 0.6, **risk,
        }),
        state_dir=tmp_path / "state",
    )


def make_session(tmp_path, cfg=None, clock=None, gateway=None):
    gateway = gateway or FakeGateway(cash=150_000.0)
    cfg = cfg or config(tmp_path)
    broker = IBKRBroker(
        gateway, cfg.account, cfg.mode, settle_seconds=0,
        order_prefix=f"ql-{cfg.strategy_id}.",
    )
    return LiveSession(
        cfg, broker, tmp_path / "store", clock=clock or Clock(SATURDAY),
        strategy_factory=lambda _: DailyLongShort(),
    )


@pytest.fixture
def world(tmp_path):
    frames = build_daily_market(tmp_path / "store")
    gateway = FakeGateway(cash=150_000.0)
    gateway.prices = {s: float(f["close"].iloc[-1]) for s, f in frames.items()}
    gateway.shortable = {s: 1e6 for s in frames}
    clock = Clock(SATURDAY)
    session = make_session(tmp_path, clock=clock, gateway=gateway)
    return session, gateway, clock, frames, tmp_path


def opens(frames):
    return {s: float(f["close"].iloc[-1]) * 1.002 for s, f in frames.items()}


def first_session(world):
    """Open the sleeve, approve the first decision, fill it at Monday's open."""
    session, gateway, clock, frames, _ = world
    session.init()
    session.sync()
    proposal = session.propose()
    session.approve(proposal.proposal_id, proposal.proposal_id)
    clock.advance(days=2)  # Monday morning
    gateway.opening_auction(opens(frames))
    report = session.sync()
    return proposal, report


def quantities(session):
    return {str(i): p.quantity for i, p in session.book().positions.items()}


def next_day(world, drift=None):
    """Monday's session closes; the decision moves to Tuesday morning."""
    session, gateway, clock, frames, tmp_path = world
    clock.now = datetime(2026, 9, 22, 6, 0, tzinfo=timezone.utc)
    frames = append_day(tmp_path / "store", frames, clock.now, drift=drift)
    gateway.prices = {s: float(f["close"].iloc[-1]) for s, f in frames.items()}
    return frames


# -- the happy path ----------------------------------------------------------------


def test_a_long_and_a_short_are_opened_and_each_is_protected(world):
    session, gateway, *_ = world
    proposal, report = first_session(world)

    sides = {str(o.intent.instrument): o.intent.side.value for o in proposal.orders}
    assert sides == {LONG: "buy", SHORT: "sell"}
    assert all(o.intent.time_in_force.value == "opg" for o in proposal.orders), (
        "a daily strategy still trades at the opening auction"
    )
    assert all(o.intent.client_order_id.startswith("ql-toyls.") for o in proposal.orders)

    assert report.new_fills == 2
    assert report.reconciliation.status == "ok", report.reconciliation.findings
    held = quantities(session)
    assert held[LONG] > 0 and held[SHORT] < 0
    assert held == {str(p.instrument): p.quantity for p in session.broker.positions(session.portfolio)}

    stops = {str(w.instrument): w for w in session.broker.working_orders()}
    assert report.stops_placed == 2
    assert stops[LONG].side.value == "sell"
    assert stops[SHORT].side.value == "buy", "a short is protected by a buy stop"
    book = session.book()
    short_fill = book.positions[InstrumentId(SHORT)].average_cost
    long_fill = book.positions[InstrumentId(LONG)].average_cost
    assert stops[SHORT].stop_price == pytest.approx(short_fill * 1.10, abs=0.011)
    assert stops[LONG].stop_price == pytest.approx(long_fill * 0.90, abs=0.011)
    assert stops[SHORT].quantity == abs(held[SHORT])
    assert all(w.client_order_id.startswith("ql-toyls.") for w in stops.values())


def test_a_short_stop_fires_on_the_high_and_the_short_is_closed(world):
    session, gateway, *_ = world
    first_session(world)
    gateway.trigger_stops({}, highs={SHORT: 1e6})
    report = session.sync()
    assert report.new_fills == 1
    assert any("stop filled" in n for n in report.notes)
    assert SHORT not in quantities(session)
    assert report.reconciliation.status == "ok", report.reconciliation.findings


def test_a_reversal_covers_the_short_and_goes_long(world):
    """The worst name rallies to the top: cover and buy in one order."""
    session, gateway, clock, frames, tmp_path = world
    first_session(world)
    was_short = quantities(session)[SHORT]
    next_day(world, drift={SHORT: 0.60})
    session.sync()
    proposal = session.propose()

    by_name = {str(o.intent.instrument): o.intent for o in proposal.orders}
    assert by_name[SHORT].side.value == "buy"
    assert by_name[SHORT].quantity > abs(was_short), "covers, then buys the long"
    assert by_name[SHORT].reason == "reverse"
    assert by_name[LONG].side.value == "sell" and by_name[LONG].reason == "close"
    new_short = next(n for n, i in by_name.items() if i.reason == "open")
    assert by_name[new_short].side.value == "sell"
    reasons = [o.intent.reason for o in proposal.orders]
    assert reasons.index("open") == len(reasons) - 1, "exposure-reducing orders go first"

    session.approve(proposal.proposal_id, proposal.proposal_id)
    cancelled = {
        str(e.payload["instrument"]) for e in session.journal.events(EventKind.STOP_CANCELLED)
    }
    assert {SHORT, LONG} <= cancelled, (
        "the resting stops on the traded names are cancelled first, or they would "
        "close the same shares twice"
    )
    clock.advance(hours=4)
    gateway.opening_auction(opens(frames))
    report = session.sync()
    assert report.reconciliation.status == "ok", report.reconciliation.findings
    held = quantities(session)
    assert held[SHORT] > 0 and held[new_short] < 0 and LONG not in held
    stops = {str(w.instrument): w.side.value for w in session.broker.working_orders()}
    assert stops == {SHORT: "sell", new_short: "buy"}


# -- the pre-trade checks on a short ----------------------------------------------


def test_a_short_target_fails_loudly_unless_the_strategy_may_short(tmp_path):
    """A strategy that emits a short by accident must stop, not borrow stock."""
    build_daily_market(tmp_path / "store")
    gateway = FakeGateway(cash=150_000.0)
    gateway.shortable = {SHORT: 1e6}
    session = make_session(tmp_path, cfg=config(tmp_path, allow_short=False), gateway=gateway)
    session.init()
    session.sync()
    with pytest.raises(ContractViolation, match="short targets are not permitted"):
        session.propose()
    assert gateway.placed == 0


def test_a_short_the_broker_cannot_borrow_is_cut(world):
    session, gateway, *_ = world
    gateway.shortable = {}  # the broker reports nothing: treated as not borrowable
    session.init()
    session.sync()
    proposal = session.propose()
    assert SHORT not in {str(o.intent.instrument) for o in proposal.orders}
    assert any("no borrow availability" in f for f in proposal.findings)


def test_a_short_is_cut_to_the_shares_available(world):
    session, gateway, *_ = world
    gateway.shortable = {SHORT: 120.0}
    session.init()
    session.sync()
    proposal = session.propose()
    short = next(o.intent for o in proposal.orders if str(o.intent.instrument) == SHORT)
    assert short.quantity == 120
    assert any("cut to 120" in f for f in proposal.findings)


def test_a_short_the_account_cannot_margin_is_refused(world):
    session, gateway, *_ = world
    gateway.short_margin = 10.0  # every dollar short needs ten of margin
    session.init()
    session.sync()
    with pytest.raises(ContractViolation, match="margin check failed"):
        session.propose()


def test_a_broker_that_cannot_report_margin_refuses_shorts(world):
    session, gateway, *_ = world

    class NoMargin:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "margin_check":
                raise AttributeError(name)
            return getattr(self._inner, name)

    session.broker = NoMargin(session.broker)
    session.init()
    session.sync()
    with pytest.raises(ContractViolation, match="cannot report margin"):
        session.propose()


# -- degraded states ---------------------------------------------------------------


def test_reduce_only_covers_a_short_but_never_reverses_or_opens(world):
    session, gateway, clock, frames, _ = world
    first_session(world)
    was_short = quantities(session)[SHORT]
    next_day(world, drift={SHORT: 0.60})
    session.sync()
    session.degrade(DegradationState.REDUCE_ONLY, "drawdown past the 80th percentile")
    proposal = session.propose()
    by_name = {str(o.intent.instrument): o.intent for o in proposal.orders}
    assert by_name[SHORT].side.value == "buy"
    assert by_name[SHORT].quantity == abs(was_short), "the reversal stops at flat"
    assert all(i.side.value == "buy" or str(i.instrument) == LONG for i in by_name.values()), (
        "no new short is opened"
    )
    assert set(by_name) == {LONG, SHORT}


def test_a_liquidation_sells_the_longs_and_covers_the_shorts(world):
    session, *_ = world
    first_session(world)
    held = quantities(session)
    session.degrade(DegradationState.HALTED, "changepoint probability above one half")
    session.sync()
    plan = session.propose(liquidate=True)
    orders = {str(o.intent.instrument): o.intent for o in plan.orders}
    assert orders[LONG].side.value == "sell" and orders[LONG].quantity == held[LONG]
    assert orders[SHORT].side.value == "buy" and orders[SHORT].quantity == -held[SHORT]


def test_a_long_where_the_sleeve_is_short_halts(world):
    session, gateway, *_ = world
    first_session(world)
    gateway._positions[SHORT] = -gateway._positions[SHORT]
    report = session.sync()
    assert report.reconciliation.status == "mismatch"
    assert any(f.instrument == SHORT and f.level == "mismatch" for f in report.reconciliation.findings)
    assert session.state() is DegradationState.HALTED


def test_a_manual_adjustment_can_record_a_short_only_if_shorts_are_allowed(tmp_path):
    build_daily_market(tmp_path / "store")
    gateway = FakeGateway(cash=150_000.0)
    gateway.hold(SHORT, -50, 30.0)
    session = make_session(
        tmp_path, cfg=replace(config(tmp_path, allow_short=False), account_scope="shared"),
        gateway=gateway,
    )
    session.init()
    with pytest.raises(ContractViolation, match="short"):
        session.adjust(SHORT, -50, "short opened by hand, now handed to the sleeve")


# -- the clock of a daily strategy -------------------------------------------------


def test_a_proposal_is_dead_once_a_newer_bar_has_closed(tmp_path):
    frames = build_daily_market(tmp_path / "store")
    gateway = FakeGateway(cash=150_000.0)
    gateway.shortable = {s: 1e6 for s in frames}
    clock = Clock(SATURDAY)
    session = make_session(
        tmp_path, cfg=replace(config(tmp_path), proposal_ttl_hours=500), clock=clock,
        gateway=gateway,
    )
    session.init()
    session.sync()
    proposal = session.propose()
    clock.now = datetime(2026, 9, 22, 6, 0, tzinfo=timezone.utc)
    append_day(tmp_path / "store", frames, clock.now)
    with pytest.raises(ContractViolation, match="newer one has closed"):
        session.approve(proposal.proposal_id, proposal.proposal_id)


def test_daily_data_older_than_its_limit_is_refused(world):
    session, gateway, clock, *_ = world
    session.init()
    clock.advance(days=4)  # Wednesday, and still Friday's bar
    session.sync()
    with pytest.raises(ContractViolation, match="latest complete day"):
        session.propose()


def test_the_opened_event_names_the_strategy_and_its_bar_size(world):
    session, *_ = world
    session.init()
    opened = session.journal.last(EventKind.OPENED).payload
    assert opened["strategy_id"] == "toyls"
    assert opened["account"] == "DU1234567"
    assert opened["interval"] == BarInterval.DAY.value
    assert opened["strategy_version"] == str(DailyLongShort().version)
    assert session.interval is BarInterval.DAY
