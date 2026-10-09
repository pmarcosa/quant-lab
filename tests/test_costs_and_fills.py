"""Commission plans in dollars, a minimum order size, and fills at the decision.

Three things a small account's backtest gets wrong when they are missing. A flat
number of basis points hides that a broker charges a minimum per order; without
a minimum order size the engine sends orders that fee makes pointless; and the
choice between filling at the next open and at the decision's close has to be
one switch, so that two runs can differ in it and in nothing else.
"""

from __future__ import annotations

import pytest

from contracts.errors import ContractViolation
from contracts.execution import Side
from engine.accounting import Book
from engine.decide import SizingPolicy, decide
from engine.run import run_backtest
from execution.simulated import (
    COMMISSION_PLANS,
    IBKR_FIXED,
    IBKR_TIERED,
    CommissionSchedule,
    CostModel,
    SimulatedBroker,
)
from tests.conftest import at
from tests.test_engine import (
    AAA,
    BBB,
    RUN,
    FixedFiltration,
    ScriptedStrategy,
    _fill,
    _weeks,
    opening,
    whole_shares,
)

# -- commission plans ------------------------------------------------------------


def test_a_small_order_pays_the_minimum_not_the_per_share_rate():
    """Ten shares at 0.35 cents each would be 3.5 cents. The plan charges 35."""
    assert IBKR_TIERED.broker_commission(10, 300.0) == pytest.approx(0.35)
    assert IBKR_FIXED.broker_commission(10, 300.0) == pytest.approx(1.00)


def test_a_large_order_pays_per_share():
    assert IBKR_TIERED.broker_commission(1_000, 50.0) == pytest.approx(3.50)
    assert IBKR_FIXED.broker_commission(1_000, 50.0) == pytest.approx(5.00)


def test_the_cap_wins_over_the_minimum():
    """One share at 20 dollars: 1% of the value is 20 cents, below either minimum."""
    assert IBKR_TIERED.broker_commission(1, 20.0) == pytest.approx(0.20)
    assert IBKR_FIXED.broker_commission(1, 20.0) == pytest.approx(0.20)


def test_the_tiered_plan_adds_the_fees_the_fixed_plan_includes():
    # 0.35 commission + 10 x 0.003 exchange + 10 x 0.0002 clearing
    # + 10 x 0.000003 audit trail + 0.35 x 0.000738 pass-through.
    bought = IBKR_TIERED.fee(10, 300.0, Side.BUY)
    assert bought == pytest.approx(0.35 + 0.03 + 0.002 + 0.00003 + 0.35 * 0.000738)
    # The fixed plan is all-in apart from the regulators.
    assert IBKR_FIXED.fee(10, 300.0, Side.BUY) == pytest.approx(1.00 + 0.00003)


def test_a_sell_also_pays_the_regulators():
    bought = IBKR_TIERED.fee(10, 300.0, Side.BUY)
    sold = IBKR_TIERED.fee(10, 300.0, Side.SELL)
    assert sold - bought == pytest.approx(3_000.0 * 0.0000206 + 10 * 0.000166)


def test_the_trading_activity_fee_has_a_ceiling():
    plan = CommissionSchedule(name="t", per_share=0.0, minimum=0.0, taf_per_share=0.01,
                              taf_maximum=5.0)
    assert plan.fee(1_000_000, 1.0, Side.SELL) == pytest.approx(5.0)


def test_a_typical_share_price_undoes_the_split_adjustment():
    """4,000 dollars of a stock whose adjusted price is one dollar.

    Charged on the stored quantity that is 4,000 shares and 14 dollars. The
    stock really traded near 100 dollars before its splits, so the order was 40
    shares and paid the minimum.
    """
    assert IBKR_TIERED.broker_commission(4_000, 1.0) == pytest.approx(14.0)
    typical = IBKR_TIERED.with_reference_price(100.0)
    assert typical.shares_charged(4_000, 1.0) == pytest.approx(40.0)
    assert typical.broker_commission(4_000, 1.0) == pytest.approx(0.35)
    assert IBKR_TIERED.reference_share_price is None, "the published plan is unchanged"


def test_a_plan_replaces_the_basis_points_in_the_cost_model():
    flat = CostModel(commission_bps=10.0, slippage_bps=0.0)
    planned = CostModel(commission_bps=10.0, slippage_bps=0.0, schedule=IBKR_TIERED)
    assert flat.commission(10, 300.0) == pytest.approx(3.0)
    assert planned.commission(10, 300.0, Side.BUY) == pytest.approx(
        IBKR_TIERED.fee(10, 300.0, Side.BUY)
    )
    assert flat.describe() == "10bp"
    assert planned.describe() == "ibkr-tiered"
    assert CostModel(schedule=IBKR_TIERED.with_reference_price(100.0)).describe() == (
        "ibkr-tiered@100"
    )


def test_the_plans_are_found_by_name():
    assert COMMISSION_PLANS["ibkr-tiered"] is IBKR_TIERED
    assert COMMISSION_PLANS["ibkr-fixed"] is IBKR_FIXED


def test_a_plan_refuses_a_negative_fee():
    with pytest.raises(ContractViolation):
        CommissionSchedule(name="bad", per_share=-0.001, minimum=0.0)
    with pytest.raises(ContractViolation):
        CommissionSchedule(name="bad", per_share=0.0, minimum=0.0, reference_share_price=0.0)


def test_the_simulator_charges_a_sell_as_a_sell(portfolio):
    """The fill carries the plan's fee for its own side, not the buy's."""
    weeks = _weeks(3)
    prices = {week: {AAA: 100.0} for week in weeks}

    class InThenOut(ScriptedStrategy):
        def target(self, filtration, held):
            self.weights = {AAA: 1.0} if filtration.decision_time == weeks[0] else {}
            return super().target(filtration, held)

    broker = SimulatedBroker(costs=CostModel(slippage_bps=0.0, schedule=IBKR_TIERED))
    result = run_backtest(
        run=RUN, opening=opening(portfolio), strategy=InThenOut({}), schedule=weeks,
        filtration_at=FixedFiltration, marks_at=prices.__getitem__,
        execution_at=prices.__getitem__, broker=broker,
        policy=SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.0),
    )
    bought, sold = result.all_fills()
    assert bought.commission == pytest.approx(IBKR_TIERED.fee(990, 100.0, Side.BUY))
    assert sold.commission == pytest.approx(IBKR_TIERED.fee(990, 100.0, Side.SELL))
    assert sold.commission > bought.commission


# -- the minimum order size ------------------------------------------------------


def _decide(book, weights, marks, minimum):
    return decide(
        run=RUN, book=book, strategy=ScriptedStrategy(weights),
        filtration=FixedFiltration(at(2020, 1, 3)), marks=marks,
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0, min_order_value=minimum),
    )


def test_an_adjustment_below_the_minimum_order_is_not_sent(portfolio):
    # 95 shares held, 100 wanted: a 500-dollar top-up against a 1,000 minimum.
    book = Book(portfolio=portfolio, cash=10_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 95, 100.0, at(2020, 1, 2))
    )
    decision = _decide(book, {AAA: 1.0}, {AAA: 100.0}, minimum=1_000.0)
    assert decision.is_flat
    assert decision.skipped[AAA] == "below the minimum order size"
    # Without the rule the same decision sends the five shares.
    assert _decide(book, {AAA: 1.0}, {AAA: 100.0}, minimum=0.0).intents[0].quantity == 5


def test_a_new_position_below_the_minimum_order_is_not_opened(portfolio):
    decision = _decide(opening(portfolio, cash=10_000.0), {AAA: 0.92, BBB: 0.08},
                       {AAA: 100.0, BBB: 100.0}, minimum=1_000.0)
    assert [o.instrument for o in decision.intents] == [AAA]
    assert decision.skipped[BBB] == "below the minimum order size"


def test_an_order_exactly_at_the_minimum_is_sent(portfolio):
    decision = _decide(opening(portfolio, cash=10_000.0), {AAA: 0.1}, {AAA: 100.0},
                       minimum=1_000.0)
    assert decision.intents[0].quantity == 10


def test_an_exit_is_never_held_back_for_being_small(portfolio):
    """Three shares, 300 dollars, a 1,000 minimum: the position still closes."""
    book = Book(portfolio=portfolio, cash=10_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 3, 100.0, at(2020, 1, 2))
    )
    decision = _decide(book, {}, {AAA: 100.0}, minimum=1_000.0)
    (order,) = decision.intents
    assert (order.side, order.quantity, order.reason) == (Side.SELL, 3, "close")


def test_a_negative_minimum_order_is_refused():
    with pytest.raises(ContractViolation):
        SizingPolicy(min_order_value=-1.0)
    with pytest.raises(ContractViolation):
        SizingPolicy(min_order_fraction=1.0)


def test_the_minimum_order_can_be_a_fraction_of_equity(portfolio):
    """The rule for an account that stays its size: 6% of 10,000 is 600 dollars."""
    book = Book(portfolio=portfolio, cash=10_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 95, 100.0, at(2020, 1, 2))
    )

    def decided(fraction, dollars=0.0):
        return decide(
            run=RUN, book=book, strategy=ScriptedStrategy({AAA: 1.0}),
            filtration=FixedFiltration(at(2020, 1, 3)), marks={AAA: 100.0},
            constraints_for=whole_shares,
            policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0,
                                min_order_value=dollars, min_order_fraction=fraction),
        )

    # The 500-dollar top-up is 5% of the 10,000 book.
    assert decided(0.06).skipped[AAA] == "below the minimum order size"
    assert decided(0.04).intents[0].quantity == 5
    # The larger threshold applies.
    assert decided(0.04, dollars=1_000.0).is_flat


# -- filling at the decision -----------------------------------------------------


def _run(portfolio, at_decision, closes, opens, weeks, **more):
    broker = SimulatedBroker(costs=CostModel(commission_bps=0.0, slippage_bps=0.0))
    return run_backtest(
        run=RUN, opening=opening(portfolio), strategy=ScriptedStrategy({AAA: 1.0}),
        schedule=weeks, filtration_at=FixedFiltration, marks_at=closes.__getitem__,
        execution_at=opens.__getitem__, broker=broker,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
        fill_at_decision=at_decision, **more,
    )


def test_an_order_filled_at_the_decision_trades_at_the_mark_it_was_sized_on(portfolio):
    """The mirror of ``test_orders_fill_at_the_next_bar_not_the_decision_bar``.

    Decided on a close of 100 with the next open at 200. Filled at the decision
    the 1,000 shares cost exactly the equity; filled at the next open they cost
    twice that. Both are sized on 100.
    """
    weeks = _weeks(2)
    closes = {weeks[0]: {AAA: 100.0}, weeks[1]: {AAA: 250.0}}
    opens = {weeks[0]: {AAA: 100.0}, weeks[1]: {AAA: 200.0}}

    now = _run(portfolio, True, closes, opens, weeks)
    (fill,) = now.all_fills()
    assert (fill.price, fill.quantity, fill.at) == (100.0, 1_000, weeks[0])
    assert now.steps[0].book_after.cash == pytest.approx(0.0)
    # Marked where the default convention marks it: at the next bar's open.
    assert now.steps[0].equity_after == pytest.approx(200_000.0)

    later = _run(portfolio, False, closes, opens, weeks)
    assert later.all_fills()[0].price == 200.0
    assert later.steps[0].equity_after == pytest.approx(100_000.0)


def test_the_last_decision_stays_unfilled_when_filling_at_the_decision(portfolio):
    weeks = _weeks(1)
    prices = {weeks[0]: {AAA: 100.0}}
    result = _run(portfolio, True, prices, prices, weeks)
    assert result.all_fills() == ()
    assert len(result.steps[0].intents) == 1


def test_filling_at_the_decision_changes_fill_prices_and_nothing_else(portfolio):
    """With every open equal to the close before it, the two conventions agree."""
    weeks = _weeks(4)
    level = [100.0, 110.0, 105.0, 120.0]
    closes = {week: {AAA: price} for week, price in zip(weeks, level, strict=True)}
    opens = {weeks[0]: {AAA: 100.0}}
    opens.update({week: {AAA: price} for week, price in zip(weeks[1:], level, strict=False)})
    now = _run(portfolio, True, closes, opens, weeks)
    later = _run(portfolio, False, closes, opens, weeks)
    assert [e for _, e in now.equity_curve()] == pytest.approx(
        [e for _, e in later.equity_curve()]
    )


# -- limit orders from the sizing rules, and the config that sets them all ---------


def test_a_limit_band_prices_orders_through_the_mark(portfolio):
    from contracts.execution import OrderType

    book = Book(portfolio=portfolio, cash=10_000.0, as_of=at(2020)).apply(
        _fill(BBB, Side.BUY, 50, 100.0, at(2020, 1, 2))
    )
    decision = decide(
        run=RUN, book=book, strategy=ScriptedStrategy({AAA: 0.5}),
        filtration=FixedFiltration(at(2020, 1, 3)), marks={AAA: 100.0, BBB: 100.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0, limit_band=0.02),
    )
    by_side = {o.side: o for o in decision.intents}
    assert all(o.order_type is OrderType.LIMIT for o in decision.intents)
    assert by_side[Side.BUY].limit_price == pytest.approx(102.0), "pays up to 2% more"
    assert by_side[Side.SELL].limit_price == pytest.approx(98.0), "accepts down to 2% less"


def test_without_a_band_orders_are_market_orders(portfolio):
    from contracts.execution import OrderType

    decision = decide(
        run=RUN, book=opening(portfolio), strategy=ScriptedStrategy({AAA: 0.5}),
        filtration=FixedFiltration(at(2020, 1, 3)), marks={AAA: 100.0},
        constraints_for=whole_shares, policy=SizingPolicy(),
    )
    (only,) = decision.intents
    assert only.order_type is OrderType.MARKET and only.limit_price is None


def test_the_config_says_how_orders_are_sized_and_what_they_cost():
    """One place, read by the backtest, the baseline and the live proposal."""
    from contracts.temporal import BarInterval
    from runtime.config import ExecutionSettings, RiskSettings

    plain = ExecutionSettings()
    policy = plain.sizing(BarInterval.WEEK)
    assert (policy.cash_buffer, policy.min_trade_fraction, policy.limit_band) == (0.01, 0.005, None)
    assert plain.costs().describe() == "10bp"

    weekly_task = ExecutionSettings(cash_buffer=0.0, no_trade_band=0.0, commission="ibkr-tiered")
    policy = weekly_task.sizing(BarInterval.WEEK, allow_short=False, min_order_value=500.0)
    assert (policy.cash_buffer, policy.min_trade_fraction) == (0.0, 0.0)
    assert policy.min_order_value == 500.0, "a caller may add to the config's rules"
    assert weekly_task.costs().describe() == "ibkr-tiered@100"
    assert ExecutionSettings(commission="ibkr-fixed", share_price=0).costs().describe() == (
        "ibkr-fixed"
    )

    assert RiskSettings(stop_distance=0.0).stop() is None
    assert RiskSettings().stop().limit_offset is None
    assert RiskSettings(stop_limit_offset=0.005).stop().limit_offset == 0.005
