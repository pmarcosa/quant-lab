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
    NetExposureLimit,
    ProtectiveStop,
    ReduceOnly,
    RiskSupervisor,
    ShortSales,
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

LONG = {AAA: 100.0}
SHORT = {AAA: -100.0}


def test_risk_may_not_enlarge_a_buy(portfolio):
    proposed = (order(AAA, Side.BUY, 100),)
    with pytest.raises(ContractViolation, match="only reduce exposure"):
        RiskReview(proposed=proposed, approved=(order(AAA, Side.BUY, 150),), held={})


def test_risk_may_not_add_a_buy_that_was_not_proposed(portfolio):
    with pytest.raises(ContractViolation, match="only reduce exposure"):
        RiskReview(proposed=(), approved=(order(BBB, Side.BUY, 10),), held={})


def test_risk_may_not_shrink_a_sell(portfolio):
    """A sell out of a long is an exit; watering one down keeps exposure on."""
    proposed = (order(AAA, Side.SELL, 100),)
    with pytest.raises(ContractViolation, match="exit"):
        RiskReview(proposed=proposed, approved=(order(AAA, Side.SELL, 40),), held=LONG)


def test_risk_may_not_drop_a_sell(portfolio):
    with pytest.raises(ContractViolation, match="exit"):
        RiskReview(proposed=(order(AAA, Side.SELL, 100),), approved=(), held=LONG)


def test_risk_may_shrink_a_buy_and_add_a_sell(portfolio):
    review = RiskReview(
        proposed=(order(AAA, Side.BUY, 100),),
        approved=(order(AAA, Side.BUY, 60), order(BBB, Side.SELL, 25)),
        held={BBB: 50.0},
    )
    assert review.changed


def test_findings_separate_a_warning_from_a_limit():
    review = RiskReview(
        proposed=(),
        approved=(),
        held={},
        findings=(
            RiskFinding("cluster", Severity.WARN, "28% of the book"),
            RiskFinding("gross", Severity.LIMIT, "trimmed"),
        ),
    )
    assert len(review.warnings) == 1
    assert len(review.limits_applied) == 1


# -- the invariant, on a book that can be short -------------------------------


def test_risk_may_not_shrink_a_buy_that_covers_a_short(portfolio):
    """For a short, the exit is a buy. The same rule protects it."""
    with pytest.raises(ContractViolation, match="exit"):
        RiskReview(
            proposed=(order(AAA, Side.BUY, 100),),
            approved=(order(AAA, Side.BUY, 40),),
            held=SHORT,
        )


def test_risk_may_trim_a_short_sale_that_opens_exposure(portfolio):
    review = RiskReview(
        proposed=(order(AAA, Side.SELL, 100),), approved=(order(AAA, Side.SELL, 30),), held={}
    )
    assert review.changed


def test_risk_may_not_enlarge_a_short_sale(portfolio):
    with pytest.raises(ContractViolation, match="only reduce exposure"):
        RiskReview(
            proposed=(order(AAA, Side.SELL, 100),), approved=(order(AAA, Side.SELL, 150),), held={}
        )


def test_risk_may_stop_a_reversal_at_flat_but_not_before_it(portfolio):
    """Long 100, asked to go short 50: flat is allowed, still long is not."""
    proposed = (order(AAA, Side.SELL, 150),)
    RiskReview(proposed=proposed, approved=(order(AAA, Side.SELL, 100),), held=LONG)
    with pytest.raises(ContractViolation, match="exit"):
        RiskReview(proposed=proposed, approved=(order(AAA, Side.SELL, 60),), held=LONG)


def test_risk_may_add_a_cover_the_strategy_did_not_ask_for(portfolio):
    RiskReview(proposed=(), approved=(order(AAA, Side.BUY, 100),), held=SHORT)
    with pytest.raises(ContractViolation, match="only reduce exposure"):
        RiskReview(proposed=(), approved=(order(AAA, Side.BUY, 150),), held=SHORT)


# -- the rules ---------------------------------------------------------------

MARKS = {AAA: 100.0, BBB: 100.0, CCC: 100.0}


def test_gross_exposure_trims_a_buy_that_would_overshoot(portfolio):
    rule = GrossExposureLimit(maximum=1.0)
    positions = {AAA: position(AAA, quantity=90, weight=0.90, mark=100.0)}
    intents = (order(AAA, Side.BUY, 50),)  # 5,000 against 1,000 of room
    approved, findings = rule.apply(intents, positions, equity=10_000.0, marks=MARKS)
    assert approved[0].quantity == 10
    assert findings[0].severity is Severity.LIMIT
    RiskReview(proposed=intents, approved=approved, held={AAA: 90.0})


def test_gross_exposure_never_touches_a_sell(portfolio):
    rule = GrossExposureLimit(maximum=0.1)
    positions = {AAA: position(AAA, quantity=99, weight=0.99, mark=100.0)}
    intents = (order(AAA, Side.SELL, 99),)
    approved, _ = rule.apply(intents, positions, equity=10_000.0, marks=MARKS)
    assert approved == intents


def test_gross_exposure_counts_what_a_rotation_releases(portfolio):
    """Fully invested, selling A to buy B: not over the limit.

    The earlier rule ignored the room a sale frees and let any name the book did
    not hold through unpriced, so it only ever limited additions to positions.
    """
    rule = GrossExposureLimit(maximum=1.0)
    positions = {AAA: position(AAA, quantity=99, weight=0.99, mark=100.0)}
    intents = (order(AAA, Side.SELL, 99), order(BBB, Side.BUY, 120))
    approved, findings = rule.apply(intents, positions, equity=10_000.0, marks=MARKS)
    assert [o.quantity for o in approved] == [99, 100]
    assert findings[0].instrument == BBB


def test_gross_exposure_counts_shorts_as_exposure(portfolio):
    rule = GrossExposureLimit(maximum=1.0)
    positions = {AAA: position(AAA, quantity=60, weight=0.6, mark=100.0)}
    intents = (order(BBB, Side.SELL, 60),)  # a 60% short on top of a 60% long
    approved, _ = rule.apply(intents, positions, equity=10_000.0, marks=MARKS)
    assert approved[0].quantity == 40


def test_gross_exposure_trims_only_the_opening_leg_of_a_reversal(portfolio):
    rule = GrossExposureLimit(maximum=0.5)
    positions = {AAA: position(AAA, quantity=40, weight=0.4, mark=100.0)}
    intents = (order(AAA, Side.SELL, 140),)  # close 40 long, open 100 short
    approved, _ = rule.apply(intents, positions, equity=10_000.0, marks=MARKS)
    assert approved[0].quantity == 90  # 40 closed, 50 short allowed
    RiskReview(proposed=intents, approved=approved, held={AAA: 40.0})


def test_a_new_position_without_a_mark_fails_closed(portfolio):
    rule = GrossExposureLimit(maximum=1.0)
    with pytest.raises(ContractViolation, match="cannot value"):
        rule.apply((order(BBB, Side.BUY, 10),), {}, equity=10_000.0, marks={})


def test_net_exposure_limits_both_sides(portfolio):
    rule = NetExposureLimit(minimum=-0.2, maximum=0.2)
    longs, _ = rule.apply((order(AAA, Side.BUY, 50),), {}, equity=10_000.0, marks=MARKS)
    shorts, _ = rule.apply((order(AAA, Side.SELL, 50),), {}, equity=10_000.0, marks=MARKS)
    assert longs[0].quantity == 20 and shorts[0].quantity == 20


def test_a_hedged_book_has_net_room_but_uses_gross(portfolio):
    positions = {AAA: position(AAA, quantity=50, weight=0.5, mark=100.0)}
    net, _ = NetExposureLimit(-0.1, 0.1).apply(
        (order(BBB, Side.SELL, 50),), positions, equity=10_000.0, marks=MARKS
    )
    assert net[0].quantity == 50, "a short against a long brings net back to zero"


def test_short_sales_are_off_unless_enabled(portfolio):
    intents = (order(AAA, Side.SELL, 30),)
    approved, findings = ShortSales().apply(intents, {}, 10_000.0, MARKS)
    assert approved == () and "not enabled" in findings[0].message
    # Selling a long is not a short sale.
    positions = {AAA: position(AAA, quantity=30)}
    approved, findings = ShortSales().apply(intents, positions, 10_000.0, MARKS)
    assert approved == intents and findings == ()


def test_a_short_is_cut_to_what_can_be_borrowed(portfolio):
    rule = ShortSales(allowed=True, availability={AAA: 12.0, BBB: None})
    approved, findings = rule.apply(
        (order(AAA, Side.SELL, 30), order(BBB, Side.SELL, 5)), {}, 10_000.0, MARKS
    )
    assert [(o.instrument, o.quantity) for o in approved] == [(AAA, 12.0)]
    assert "no borrow availability" in findings[1].message


def test_an_expensive_borrow_is_refused(portfolio):
    rule = ShortSales(allowed=True, availability={AAA: 1e6}, borrow_fees={AAA: 0.30},
                      max_borrow_fee=0.05)
    approved, _ = rule.apply((order(AAA, Side.SELL, 30),), {}, 10_000.0, MARKS)
    assert approved == ()


def test_reduce_only_lets_a_short_be_covered_and_nothing_opened(portfolio):
    rule = ReduceOnly()
    positions = {AAA: position(AAA, quantity=-100), BBB: position(BBB, quantity=50)}
    intents = (
        order(AAA, Side.BUY, 150),   # cover 100, would open a 50 long
        order(BBB, Side.BUY, 10),    # adds to a long
        order(CCC, Side.SELL, 10),   # opens a short
    )
    approved, findings = rule.apply(intents, positions, 10_000.0, MARKS)
    assert [(o.instrument, o.side, o.quantity) for o in approved] == [(AAA, Side.BUY, 100.0)]
    assert len(findings) == 3


def test_a_correlated_cluster_warns_and_does_not_block(portfolio):
    rule = CorrelatedClusterWarning(clusters={"crypto": frozenset({AAA, BBB})}, threshold=0.25)
    positions = {AAA: position(AAA, weight=0.20), BBB: position(BBB, weight=0.15)}
    intents = (order(CCC, Side.BUY, 10),)
    approved, findings = rule.apply(intents, positions, equity=10_000.0, marks=MARKS)
    assert approved == intents, "warned, not blocked"
    assert findings[0].severity is Severity.WARN
    assert "35%" in findings[0].message


def test_a_cluster_below_the_threshold_says_nothing(portfolio):
    rule = CorrelatedClusterWarning(clusters={"crypto": frozenset({AAA})}, threshold=0.25)
    _, findings = rule.apply((), {AAA: position(AAA, weight=0.2)}, equity=1.0, marks=MARKS)
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


def test_a_short_is_protected_by_a_buy_stop_above_the_anchor():
    stop = ProtectiveStop(distance=0.12)
    (placed,) = stop.orders_for(
        {AAA: position(AAA, quantity=-40, anchor=100.0)},
        run=RUN, portfolio=_PORTFOLIO, strategy_version=VERSION,
        moment=at(2020, 1, 3), constraints_for=lambda i: InstrumentConstraints(i, "USD"),
    )
    assert placed.side is Side.BUY
    assert placed.quantity == 40
    assert placed.stop_price == pytest.approx(112.0)


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


def test_a_buy_stop_fills_when_the_bar_trades_up_to_it(portfolio):
    broker = SimulatedBroker(costs=CostModel(0.0, 0.0))
    broker.submit(order(AAA, Side.BUY, 40, OrderType.STOP, stop=112.0))
    assert broker.advance(at(2020, 1, 10), {AAA: 105.0}, lows={AAA: 100.0}, highs={AAA: 110.0}) == ()
    fills = broker.advance(at(2020, 1, 17), {AAA: 105.0}, lows={AAA: 100.0}, highs={AAA: 115.0})
    assert fills[0].price == pytest.approx(112.0)


def test_a_buy_stop_gapped_through_fills_at_the_open(portfolio):
    """A squeeze that opens above the stop: the fill is the open, however far."""
    broker = SimulatedBroker(costs=CostModel(0.0, 0.0))
    broker.submit(order(AAA, Side.BUY, 40, OrderType.STOP, stop=112.0))
    fills = broker.advance(at(2020, 1, 10), {AAA: 140.0}, lows={AAA: 135.0}, highs={AAA: 150.0})
    assert fills[0].price == pytest.approx(140.0)


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
    positions = {AAA: position(AAA, quantity=95, weight=0.95, mark=100.0)}
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


class _HoldsShort(_Holds):
    """Short one name at half the equity, rebalanced on the rotation schedule."""

    def target(self, filtration, held):
        from contracts.targets import TargetIntent

        self._n += 1
        rotated = (self._n - 1) % self.rotate_every == 0
        weights = {self.instrument: -0.5} if rotated else dict(held)
        return TargetIntent(
            weights=weights, horizon_bars=1, as_of=filtration.decision_time,
            diagnostics={"rotated": 1.0 if rotated else 0.0},
        )


def test_a_short_in_a_backtest_is_stopped_on_the_high_and_never_flips_long(portfolio):
    """The engine's side of shorting: a buy stop above the entry, triggered by
    the bar's high, filled at its level, and no re-entry before the rotation."""
    from engine.decide import SizingPolicy
    from engine.run import run_backtest
    from risk.rules import ShortSales

    weeks = _weekly(6)
    closes = _prices(weeks, [100.0, 100.0, 100.0, 120.0, 118.0, 116.0])
    opens = _prices(weeks, [100.0, 100.0, 100.0, 105.0, 118.0, 116.0])
    lows = _prices(weeks, [99.0, 99.0, 99.0, 104.0, 117.0, 115.0])
    highs = _prices(weeks, [101.0, 101.0, 101.0, 125.0, 119.0, 117.0])

    result = run_backtest(
        run=RUN,
        opening=Book.opening(portfolio, 100_000.0, weeks[0]),
        strategy=_HoldsShort(AAA, rotate_every=4),
        schedule=weeks,
        filtration_at=_NoFiltration,
        marks_at=closes.__getitem__,
        execution_at=opens.__getitem__,
        broker=SimulatedBroker(costs=CostModel(0.0, 0.0)),
        tradable_at=lambda m: {AAA},
        lows_at=lows.__getitem__,
        highs_at=highs.__getitem__,
        supervisor=RiskSupervisor(
            rules=(ShortSales(allowed=True),), stop=ProtectiveStop(distance=0.12)
        ),
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0, allow_short=True),
    )

    assert result.steps[1].book_after.quantity(AAA) < 0, "the short was opened"
    assert result.stops_fired() == 1
    stopped_at = next(i for i, s in enumerate(result.steps) if s.stopped_out)
    assert result.steps[stopped_at].book_after.quantity(AAA) == 0
    for step in result.steps:
        assert step.book_after.quantity(AAA) <= 0, f"went long at {step.marked_at}"
    for step in result.steps[stopped_at + 1 :]:
        assert not any(o.side is Side.SELL for o in step.intents), "no re-entry before rotation"


def test_a_short_backtest_without_permission_fails_loudly(portfolio):
    from engine.decide import SizingPolicy
    from engine.run import run_backtest

    weeks = _weekly(3)
    prices = _prices(weeks, [100.0, 100.0, 100.0])
    with pytest.raises(ContractViolation, match="short targets are not permitted"):
        run_backtest(
            run=RUN,
            opening=Book.opening(portfolio, 100_000.0, weeks[0]),
            strategy=_HoldsShort(AAA),
            schedule=weeks,
            filtration_at=_NoFiltration,
            marks_at=prices.__getitem__,
            execution_at=prices.__getitem__,
            broker=SimulatedBroker(costs=CostModel(0.0, 0.0)),
            tradable_at=lambda m: {AAA},
            policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
        )
