"""The risk layer.

The contract tests come first, because the invariant they protect — risk may
only reduce exposure — is the one a new rule could break by accident.
"""

from __future__ import annotations

import pytest

from contracts.errors import ContractViolation
from contracts.execution import (
    InstrumentConstraints,
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
)
from contracts.identifiers import InstrumentId, RunId, StrategyVersion
from contracts.risk import PositionRisk, RiskFinding, RiskReview, Severity
from engine.accounting import Book
from execution.simulated import CostModel, SimulatedBroker
from risk.rules import (
    CorrelatedClusterWarning,
    GrossExposureLimit,
    ProtectiveStop,
    RiskSupervisor,
)
from tests.conftest import at

AAA, BBB, CCC = (InstrumentId(s) for s in ("aaa", "bbb", "ccc"))
RUN = RunId("t")
VERSION = StrategyVersion.of("s", {"a": 1})


def order(instrument, side, quantity, order_type=OrderType.MARKET, stop=None, tag=""):
    return OrderIntent(
        client_order_id=f"{instrument}-{side.value}-{quantity:.0f}{tag}",
        run=RUN,
        portfolio=None.__class__ and _PORTFOLIO,
        instrument=instrument,
        strategy_version=VERSION,
        side=side,
        quantity=quantity,
        order_type=order_type,
        decision_time=at(2020, 1, 3),
        stop_price=stop,
        time_in_force=TimeInForce.GTC if order_type is OrderType.STOP else TimeInForce.DAY,
    )


def position(instrument, quantity=100.0, mark=100.0, weight=0.25, anchor=None, cost=100.0):
    return PositionRisk(
        instrument=instrument,
        quantity=quantity,
        average_cost=cost,
        mark=mark,
        weight=weight,
        anchor=anchor,
    )


@pytest.fixture(autouse=True)
def _portfolio(portfolio):
    global _PORTFOLIO
    _PORTFOLIO = portfolio
    return portfolio


# -- the invariant -----------------------------------------------------------


def test_risk_may_not_enlarge_a_buy(portfolio):
    proposed = (order(AAA, Side.BUY, 100),)
    with pytest.raises(ContractViolation, match="only reduce exposure"):
        RiskReview(proposed=proposed, approved=(order(AAA, Side.BUY, 150),))


def test_risk_may_not_add_a_buy_that_was_not_proposed(portfolio):
    with pytest.raises(ContractViolation, match="only reduce exposure"):
        RiskReview(proposed=(), approved=(order(BBB, Side.BUY, 10),))


def test_risk_may_not_shrink_a_sell(portfolio):
    """A sell is an exit; watering one down is an increase wearing a limit's clothes."""
    proposed = (order(AAA, Side.SELL, 100),)
    with pytest.raises(ContractViolation, match="exit"):
        RiskReview(proposed=proposed, approved=(order(AAA, Side.SELL, 40),))


def test_risk_may_not_drop_a_sell(portfolio):
    with pytest.raises(ContractViolation, match="exit"):
        RiskReview(proposed=(order(AAA, Side.SELL, 100),), approved=())


def test_risk_may_shrink_a_buy_and_add_a_sell(portfolio):
    review = RiskReview(
        proposed=(order(AAA, Side.BUY, 100),),
        approved=(order(AAA, Side.BUY, 60), order(BBB, Side.SELL, 25)),
    )
    assert review.changed


def test_findings_separate_a_warning_from_a_limit():
    review = RiskReview(
        proposed=(),
        approved=(),
        findings=(
            RiskFinding("cluster", Severity.WARN, "28% of the book"),
            RiskFinding("gross", Severity.LIMIT, "trimmed"),
        ),
    )
    assert len(review.warnings) == 1
    assert len(review.limits_applied) == 1


# -- the rules ---------------------------------------------------------------


def test_gross_exposure_trims_a_buy_that_would_overshoot(portfolio):
    rule = GrossExposureLimit(maximum=1.0)
    positions = {AAA: position(AAA, weight=0.90, mark=100.0)}
    intents = (order(AAA, Side.BUY, 50),)  # 5,000 against 1,000 of room
    approved, findings = rule.apply(intents, positions, equity=10_000.0)
    assert approved[0].quantity == 10
    assert findings[0].severity is Severity.LIMIT
    RiskReview(proposed=intents, approved=approved)  # the invariant still holds


def test_gross_exposure_never_touches_a_sell(portfolio):
    rule = GrossExposureLimit(maximum=0.1)
    positions = {AAA: position(AAA, weight=0.99, mark=100.0)}
    intents = (order(AAA, Side.SELL, 100),)
    approved, _ = rule.apply(intents, positions, equity=10_000.0)
    assert approved == intents


def test_a_correlated_cluster_warns_and_does_not_block(portfolio):
    rule = CorrelatedClusterWarning(clusters={"crypto": frozenset({AAA, BBB})}, threshold=0.25)
    positions = {AAA: position(AAA, weight=0.20), BBB: position(BBB, weight=0.15)}
    intents = (order(CCC, Side.BUY, 10),)
    approved, findings = rule.apply(intents, positions, equity=10_000.0)
    assert approved == intents, "warned, not blocked"
    assert findings[0].severity is Severity.WARN
    assert "35%" in findings[0].message


def test_a_cluster_below_the_threshold_says_nothing(portfolio):
    rule = CorrelatedClusterWarning(clusters={"crypto": frozenset({AAA})}, threshold=0.25)
    _, findings = rule.apply((), {AAA: position(AAA, weight=0.2)}, equity=1.0)
    assert findings == ()


# -- the stop ----------------------------------------------------------------


def test_the_stop_is_anchored_to_the_rotation_price_not_the_cost_basis():
    """The detail the project flags as impossible to get wrong safely.

    A stop from a long-held winner's average cost sits far below the market and
    protects nothing; from a loser's it sits above the market and sells at once.
    """
    stop = ProtectiveStop(distance=0.12)
    winner = position(AAA, mark=250.0, anchor=250.0, cost=176.11)
    loser = position(BBB, mark=95.0, anchor=95.0, cost=164.62)

    orders = stop.orders_for(
        {AAA: winner, BBB: loser},
        run=RUN, portfolio=_PORTFOLIO, strategy_version=VERSION,
        moment=at(2020, 1, 3), constraints_for=lambda i: InstrumentConstraints(i, "USD"),
    )
    levels = {o.instrument: o.stop_price for o in orders}
    assert levels[AAA] == pytest.approx(220.0)
    assert levels[BBB] == pytest.approx(83.6)
    # From the cost basis these would have been 155.0 and 144.8: the first 38%
    # below the market, the second above it.
    assert levels[AAA] > winner.average_cost * (1 - 0.12)
    assert levels[BBB] < loser.average_cost * (1 - 0.12)
    assert levels[BBB] < loser.mark, "a stop above the market would sell instantly"


def test_the_stop_distance_is_the_same_for_calm_and_volatile_names():
    """Volatility enters through position size, not through stop distance."""
    stop = ProtectiveStop(distance=0.12)
    assert stop.level(100.0) == pytest.approx(88.0)
    assert stop.level(1000.0) == pytest.approx(880.0)


def test_an_impossible_stop_distance_is_refused():
    with pytest.raises(ContractViolation, match="fraction"):
        ProtectiveStop(distance=1.5)
    with pytest.raises(ContractViolation, match="fraction"):
        ProtectiveStop(distance=0.0)


def test_stops_rest_as_gtc_orders():
    stop = ProtectiveStop()
    (placed,) = stop.orders_for(
        {AAA: position(AAA, anchor=100.0)},
        run=RUN, portfolio=_PORTFOLIO, strategy_version=VERSION,
        moment=at(2020, 1, 3), constraints_for=lambda i: InstrumentConstraints(i, "USD"),
    )
    assert placed.order_type is OrderType.STOP
    assert placed.time_in_force is TimeInForce.GTC
    assert placed.side is Side.SELL


# -- the broker's side of it -------------------------------------------------


def test_a_stop_fills_at_its_level_when_the_bar_touches_it(portfolio):
    broker = SimulatedBroker(costs=CostModel(0.0, 0.0))
    broker.submit(order(AAA, Side.SELL, 100, OrderType.STOP, stop=88.0))
    fills = broker.advance(at(2020, 1, 10), {AAA: 95.0}, lows={AAA: 85.0})
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(88.0)


def test_a_stop_does_not_fill_on_a_bar_that_never_reaches_it(portfolio):
    broker = SimulatedBroker(costs=CostModel(0.0, 0.0))
    broker.submit(order(AAA, Side.SELL, 100, OrderType.STOP, stop=88.0))
    assert broker.advance(at(2020, 1, 10), {AAA: 95.0}, lows={AAA: 91.0}) == ()
    assert len(broker.working) == 1, "it is still resting"


def test_a_gapped_stop_fills_at_the_open_not_at_its_level(portfolio):
    """The error here is largest exactly in the crashes the stop exists for.

    If the bar opens below the level, the market was already past it when
    trading started. Filling at the level credits an exit nobody could have got.
    """
    broker = SimulatedBroker(costs=CostModel(0.0, 0.0))
    broker.submit(order(AAA, Side.SELL, 100, OrderType.STOP, stop=88.0))
    fills = broker.advance(at(2020, 1, 10), {AAA: 70.0}, lows={AAA: 65.0})
    assert fills[0].price == pytest.approx(70.0)
    assert fills[0].price < 88.0


def test_a_resting_stop_survives_a_week_with_no_print(portfolio):
    """Cancelling it would remove protection from the instrument that halted."""
    broker = SimulatedBroker(costs=CostModel(0.0, 0.0))
    broker.submit(order(AAA, Side.SELL, 100, OrderType.STOP, stop=88.0))
    assert broker.advance(at(2020, 1, 10), {}, lows={}) == ()
    assert broker.unfilled == 0
    assert len(broker.working) == 1
    fills = broker.advance(at(2020, 1, 17), {AAA: 90.0}, lows={AAA: 80.0})
    assert len(fills) == 1


def test_a_day_order_with_no_print_still_expires(portfolio):
    broker = SimulatedBroker(costs=CostModel(0.0, 0.0))
    broker.submit(order(AAA, Side.BUY, 100))
    assert broker.advance(at(2020, 1, 10), {}, lows={}) == ()
    assert broker.unfilled == 1
    assert broker.working == ()


# -- the supervisor ----------------------------------------------------------


def test_the_supervisor_runs_every_rule_and_reports_once(portfolio):
    supervisor = RiskSupervisor(
        rules=(
            GrossExposureLimit(maximum=1.0),
            CorrelatedClusterWarning(clusters={"all": frozenset({AAA})}, threshold=0.1),
        ),
        stop=ProtectiveStop(),
    )
    positions = {AAA: position(AAA, weight=0.95, mark=100.0)}
    review = supervisor.review((order(AAA, Side.BUY, 100),), positions, equity=10_000.0)
    assert len(review.limits_applied) == 1
    assert len(review.warnings) == 1
    assert review.approved[0].quantity < 100


def test_a_supervisor_without_a_stop_places_none(portfolio):
    supervisor = RiskSupervisor(rules=(), stop=None)
    assert supervisor.protective_orders({}, RUN, portfolio, None, lambda i: None) == ()


def test_a_book_position_with_no_mark_stops_the_risk_assessment(portfolio):
    from engine.run import _position_risk

    book = Book(portfolio=portfolio, cash=1000.0, as_of=at(2020))
    assert _position_risk(book, {}, {}) == {}

    from contracts.execution import Fill

    book = book.apply(
        Fill(
            client_order_id="x", instrument=AAA, side=Side.BUY,
            quantity=5, price=100.0, at=at(2020, 1, 2),
        )
    )
    with pytest.raises(ContractViolation, match="without marks"):
        _position_risk(book, {}, {})


# -- the loop, end to end ----------------------------------------------------


def _weekly(n):
    from datetime import timedelta

    return [at(2020, 1, 3) + timedelta(weeks=i) for i in range(n)]


class _Holds:
    """A strategy that wants one name, always, at full weight."""

    def __init__(self, instrument, rotate_every=1):
        self.instrument = instrument
        self.rotate_every = rotate_every
        self._n = 0

    @property
    def version(self):
        return VERSION

    @property
    def filtration_spec(self):
        from contracts.temporal import BarInterval, FiltrationSpec

        return FiltrationSpec(interval=BarInterval.WEEK, observation_lag_bars=0)

    def universe(self, moment):
        return ()

    def target(self, filtration, held):
        from contracts.targets import TargetIntent

        self._n += 1
        rotated = (self._n - 1) % self.rotate_every == 0
        weights = {self.instrument: 1.0} if rotated else dict(held)
        return TargetIntent(
            weights=weights,
            horizon_bars=1,
            as_of=filtration.decision_time,
            diagnostics={"rotated": 1.0 if rotated else 0.0},
        )

    def state(self):
        return {}


class _NoFiltration:
    def __init__(self, moment):
        self._m = moment

    @property
    def decision_time(self):
        return self._m

    def history(self, *a, **k):
        import pandas as pd

        return pd.Series(dtype="float64")

    def frame(self, *a, **k):
        import pandas as pd

        return pd.DataFrame()

    def is_available(self, *a, **k):
        return True

    def metadata(self, *a, **k):
        return {}


def _prices(weeks, values):
    return dict(zip(weeks, [{AAA: v} for v in values], strict=True))


def test_a_stop_closes_the_position_and_it_is_not_re_entered(portfolio):
    """No auto re-entry: a stop followed by a re-buy protects nothing."""
    from engine.decide import SizingPolicy
    from engine.run import run_backtest

    weeks = _weekly(6)
    closes = _prices(weeks, [100.0, 100.0, 100.0, 80.0, 82.0, 84.0])
    opens = _prices(weeks, [100.0, 100.0, 100.0, 95.0, 82.0, 84.0])
    lows = _prices(weeks, [99.0, 99.0, 99.0, 80.0, 81.0, 83.0])

    result = run_backtest(
        run=RUN,
        opening=Book.opening(portfolio, 100_000.0, weeks[0]),
        strategy=_Holds(AAA, rotate_every=4),
        schedule=weeks,
        filtration_at=_NoFiltration,
        marks_at=closes.__getitem__,
        execution_at=opens.__getitem__,
        broker=SimulatedBroker(costs=CostModel(0.0, 0.0)),
        tradable_at=lambda m: {AAA},
        lows_at=lows.__getitem__,
        supervisor=RiskSupervisor(rules=(), stop=ProtectiveStop(distance=0.12)),
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )

    assert result.stops_fired() == 1, "the stop should have fired once"
    stopped_at = next(i for i, s in enumerate(result.steps) if s.stopped_out)
    # After the stop and before the next rotation, nothing is bought back.
    for step in result.steps[stopped_at + 1 :]:
        assert not any(o.side is Side.BUY for o in step.intents), step.marked_at


def test_a_rotation_sell_and_a_resting_stop_never_both_fill(portfolio):
    """Two orders to sell the same shares would take a long-only book short."""
    from engine.decide import SizingPolicy
    from engine.run import run_backtest

    weeks = _weekly(5)
    closes = _prices(weeks, [100.0, 100.0, 70.0, 70.0, 70.0])
    opens = _prices(weeks, [100.0, 100.0, 75.0, 70.0, 70.0])
    lows = _prices(weeks, [99.0, 99.0, 70.0, 69.0, 69.0])

    class _ThenCash(_Holds):
        def target(self, filtration, held):
            from contracts.targets import TargetIntent

            self._n += 1
            weights = {self.instrument: 1.0} if self._n <= 2 else {}
            return TargetIntent(
                weights=weights, horizon_bars=1,
                as_of=filtration.decision_time, diagnostics={"rotated": 1.0},
            )

    result = run_backtest(
        run=RUN,
        opening=Book.opening(portfolio, 100_000.0, weeks[0]),
        strategy=_ThenCash(AAA),
        schedule=weeks,
        filtration_at=_NoFiltration,
        marks_at=closes.__getitem__,
        execution_at=opens.__getitem__,
        broker=SimulatedBroker(costs=CostModel(0.0, 0.0)),
        tradable_at=lambda m: {AAA},
        lows_at=lows.__getitem__,
        supervisor=RiskSupervisor(rules=(), stop=ProtectiveStop(distance=0.12)),
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )

    for step in result.steps:
        assert step.book_after.quantity(AAA) >= 0, (
            f"went short at {step.marked_at}: a stop and an exit both filled"
        )
    assert result.steps[-1].book_after.quantity(AAA) == 0
