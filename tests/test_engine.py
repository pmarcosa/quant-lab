"""The engine: sizing, ordering, execution, and the backtest/live equivalence.

The headline test is ``test_a_live_proposal_matches_the_backtest_decision``. Every
other guarantee in this file protects one detail; that one protects the reason
the architecture exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from contracts.errors import ContractViolation
from contracts.execution import (
    InstrumentConstraints,
    OrderStatus,
    OrderType,
    Side,
)
from contracts.identifiers import InstrumentId, PortfolioId, RunId, StrategyVersion
from contracts.targets import TargetIntent
from contracts.temporal import BarInterval, FiltrationSpec
from engine.accounting import Book
from engine.decide import SizingPolicy, decide
from engine.run import propose, run_backtest
from execution.simulated import CostModel, SimulatedBroker
from tests.conftest import at

AAA = InstrumentId("aaa")
BBB = InstrumentId("bbb")
CCC = InstrumentId("ccc")
RUN = RunId("t")


class FixedFiltration:
    """A filtration pinned at a moment that answers nothing. Enough for sizing."""

    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    @property
    def decision_time(self) -> datetime:
        return self._moment

    def history(self, instrument, field, count):
        return pd.Series(dtype="float64")

    def frame(self, instruments, field, count):
        return pd.DataFrame()

    def is_available(self, instrument, min_bars=1):
        return True

    def metadata(self, instrument):
        return {}


@dataclass
class ScriptedStrategy:
    """A strategy that returns what it was told to, and records what it saw."""

    weights: Mapping[InstrumentId, float]
    seen: list[Mapping[InstrumentId, float]] = field(default_factory=list)

    @property
    def version(self) -> StrategyVersion:
        return StrategyVersion.of(
            "scripted", {str(i): w for i, w in self.weights.items()}
        )

    @property
    def filtration_spec(self) -> FiltrationSpec:
        return FiltrationSpec(interval=BarInterval.WEEK, observation_lag_bars=0)

    def universe(self, moment):
        return ()

    def target(self, filtration, held) -> TargetIntent:
        self.seen.append(dict(held))
        return TargetIntent(
            weights=dict(self.weights), horizon_bars=1, as_of=filtration.decision_time
        )

    def state(self) -> Mapping[str, Any]:
        return {"name": "scripted"}


def whole_shares(instrument: InstrumentId) -> InstrumentConstraints:
    return InstrumentConstraints(instrument=instrument, currency="USD")


def opening(portfolio: PortfolioId, cash: float = 100_000.0) -> Book:
    return Book.opening(portfolio, cash, at(2020))


# -- sizing ------------------------------------------------------------------


def test_weights_become_whole_shares_against_equity(portfolio):
    book = opening(portfolio)
    decision = decide(
        run=RUN,
        book=book,
        strategy=ScriptedStrategy({AAA: 0.5, BBB: 0.5}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0, BBB: 300.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )
    by_instrument = {o.instrument: o for o in decision.intents}
    assert by_instrument[AAA].quantity == 500  # 50,000 / 100
    assert by_instrument[BBB].quantity == 166  # 50,000 / 300, rounded down
    assert all(o.side is Side.BUY for o in decision.intents)
    assert decision.equity == 100_000.0


def test_the_cash_buffer_is_held_back_from_sizing(portfolio):
    decision = decide(
        run=RUN,
        book=opening(portfolio),
        strategy=ScriptedStrategy({AAA: 1.0}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.05, min_trade_fraction=0.0),
    )
    assert decision.intents[0].quantity == 950  # 95,000 / 100


def test_a_small_adjustment_is_not_worth_its_commission(portfolio):
    """The no-trade band is measured on value, not on share count."""
    # 995 shares at 100 plus 500 in cash. The target of 100% asks for 1,000, so
    # the adjustment is 5 shares -- 0.5% of the book, against a 5% band.
    book = Book(portfolio=portfolio, cash=100_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 995, 100.0, at(2020, 1, 2))
    )
    decision = decide(
        run=RUN,
        book=book,
        strategy=ScriptedStrategy({AAA: 1.0}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.05),
    )
    assert decision.is_flat
    assert decision.skipped[AAA] == "inside the no-trade band"


def test_an_exit_closes_the_whole_position_regardless_of_the_band(portfolio):
    book = Book(portfolio=portfolio, cash=101_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 10, 100.0, at(2020, 1, 2))
    )
    decision = decide(
        run=RUN,
        book=book,
        strategy=ScriptedStrategy({}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.5),
    )
    assert len(decision.intents) == 1
    assert decision.intents[0].side is Side.SELL
    assert decision.intents[0].quantity == 10
    assert decision.intents[0].reason == "close"


def test_sells_are_ordered_before_buys(portfolio):
    book = Book(portfolio=portfolio, cash=100_000.0, as_of=at(2020)).apply(
        _fill(CCC, Side.BUY, 500, 100.0, at(2020, 1, 2))
    )
    decision = decide(
        run=RUN,
        book=book,
        strategy=ScriptedStrategy({AAA: 0.5, BBB: 0.5}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0, BBB: 100.0, CCC: 100.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )
    sides = [o.side for o in decision.intents]
    assert sides[0] is Side.SELL
    assert set(sides[1:]) == {Side.BUY}, "the proceeds have to exist before they are spent"


def test_order_ids_are_derived_from_the_decision_not_a_counter(portfolio):
    """Two runs of the same decision produce the same ids, so a retry is safe."""
    kwargs = dict(
        run=RUN,
        book=opening(portfolio),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )
    first = decide(strategy=ScriptedStrategy({AAA: 1.0}), **kwargs)
    second = decide(strategy=ScriptedStrategy({AAA: 1.0}), **kwargs)
    assert [o.client_order_id for o in first.intents] == [
        o.client_order_id for o in second.intents
    ]
    later = decide(
        run=RUN,
        book=opening(portfolio),
        strategy=ScriptedStrategy({AAA: 1.0}),
        filtration=FixedFiltration(at(2020, 1, 10)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )
    assert later.intents[0].client_order_id != first.intents[0].client_order_id


# -- what the strategy may and may not see -----------------------------------


def test_the_strategy_is_given_the_shape_of_the_book_never_its_size(portfolio):
    # 750 shares at 100 leaves 25,000 in cash: a book that is 75% invested.
    book = Book(portfolio=portfolio, cash=100_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 750, 100.0, at(2020, 1, 2))
    )
    strategy = ScriptedStrategy({AAA: 1.0})
    decide(
        run=RUN,
        book=book,
        strategy=strategy,
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
    )
    (held,) = strategy.seen
    assert held == {AAA: pytest.approx(0.75)}, "weights, which reveal no equity"
    assert all(abs(v) <= 1.0 for v in held.values())


def test_a_short_target_is_refused_unless_it_was_asked_for(portfolio):
    with pytest.raises(ContractViolation, match="short"):
        decide(
            run=RUN,
            book=opening(portfolio),
            strategy=ScriptedStrategy({AAA: -0.5}),
            filtration=FixedFiltration(at(2020, 1, 3)),
            marks={AAA: 100.0},
            constraints_for=whole_shares,
        )


def test_a_held_position_without_a_mark_stops_the_decision(portfolio):
    book = Book(portfolio=portfolio, cash=1_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 10, 100.0, at(2020, 1, 2))
    )
    with pytest.raises(ContractViolation, match="held instrument"):
        decide(
            run=RUN,
            book=book,
            strategy=ScriptedStrategy({}),
            filtration=FixedFiltration(at(2020, 1, 3)),
            marks={},
            constraints_for=whole_shares,
        )


def test_an_untradable_instrument_is_recorded_not_silently_dropped(portfolio):
    """Priced and untradable is a real state: value it, do not trade it."""
    book = Book(portfolio=portfolio, cash=10_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 100, 100.0, at(2020, 1, 2))
    )
    decision = decide(
        run=RUN,
        book=book,
        strategy=ScriptedStrategy({}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 90.0},
        constraints_for=whole_shares,
        tradable=frozenset(),
    )
    assert decision.is_flat
    assert "not trading" in decision.skipped[AAA]
    assert decision.equity == pytest.approx(9_000.0), "still valued at its last mark"


# -- the loop ----------------------------------------------------------------


def _fill(instrument, side, quantity, price, moment):
    from contracts.execution import Fill

    return Fill(
        client_order_id=f"seed-{instrument}-{moment.isoformat()}",
        instrument=instrument,
        side=side,
        quantity=quantity,
        price=price,
        at=moment,
    )


def _weeks(n: int) -> list[datetime]:
    return [at(2020, 1, 3) + timedelta(weeks=i) for i in range(n)]


def test_orders_fill_at_the_next_bar_not_the_decision_bar(portfolio):
    """The single most common way a backtest reports money that was never there.

    The decision is taken on a bar whose close is 100 and the next bar opens at
    200. A fill at 100 would be a trade at a price the decision was still
    watching; the honest fill is at 200, so the same weight buys half as much.
    """
    weeks = _weeks(2)
    closes = {weeks[0]: {AAA: 100.0}, weeks[1]: {AAA: 250.0}}
    opens = {weeks[0]: {AAA: 100.0}, weeks[1]: {AAA: 200.0}}

    broker = SimulatedBroker(costs=CostModel(commission_bps=0.0, slippage_bps=0.0))
    result = run_backtest(
        run=RUN,
        opening=opening(portfolio),
        strategy=ScriptedStrategy({AAA: 1.0}),
        schedule=weeks,
        filtration_at=FixedFiltration,
        marks_at=closes.__getitem__,
        execution_at=opens.__getitem__,
        broker=broker,
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )
    (fill,) = result.all_fills()
    assert fill.price == 200.0, "filled at the next open"
    assert fill.quantity == 1_000, "sized on the decision mark of 100"
    # 1,000 shares bought at 200 costs 200,000 against 100,000 of equity, so the
    # book is levered -- which is exactly why the cash buffer exists, and why
    # this test uses none: it makes the gap visible rather than absorbing it.
    assert result.steps[0].book_after.cash == pytest.approx(-100_000.0)


def test_the_schedule_must_ascend(portfolio):
    weeks = _weeks(3)
    with pytest.raises(ContractViolation, match="ascend"):
        run_backtest(
            run=RUN,
            opening=opening(portfolio),
            strategy=ScriptedStrategy({}),
            schedule=[weeks[0], weeks[2], weeks[1]],
            filtration_at=FixedFiltration,
            marks_at=lambda m: {},
            execution_at=lambda m: {},
            broker=SimulatedBroker(),
        )


def test_the_last_decision_is_recorded_but_not_filled(portfolio):
    """There is no bar after the last one, so its orders have nowhere to fill."""
    weeks = _weeks(2)
    prices = dict.fromkeys(weeks, {AAA: 100.0})
    result = run_backtest(
        run=RUN,
        opening=opening(portfolio),
        strategy=ScriptedStrategy({AAA: 1.0}),
        schedule=weeks,
        filtration_at=FixedFiltration,
        marks_at=prices.__getitem__,
        execution_at=prices.__getitem__,
        broker=SimulatedBroker(costs=CostModel(0.0, 0.0)),
        policy=SizingPolicy(cash_buffer=0.0, min_trade_fraction=0.0),
    )
    assert result.steps[-1].intents == () or result.steps[-1].fills == ()


# -- the equivalence ---------------------------------------------------------


def test_a_live_proposal_matches_the_backtest_decision(portfolio):
    """Backtest and live are one code path, and this is what says so.

    The backtest runs to a bar. A live proposal is then made standing at that
    same bar, from the book the backtest had reached. The orders must be
    identical down to the client order id — which is derived from the decision,
    so equal ids mean equal decisions, not merely similar ones.
    """
    weeks = _weeks(5)
    closes = {w: {AAA: 100.0 + 5 * i, BBB: 50.0} for i, w in enumerate(weeks)}
    opens = {w: {AAA: 99.0 + 5 * i, BBB: 50.0} for i, w in enumerate(weeks)}
    policy = SizingPolicy(cash_buffer=0.02, min_trade_fraction=0.01)

    broker = SimulatedBroker(costs=CostModel(commission_bps=5.0, slippage_bps=5.0))
    backtest = run_backtest(
        run=RUN,
        opening=opening(portfolio),
        strategy=ScriptedStrategy({AAA: 0.6, BBB: 0.4}),
        schedule=weeks,
        filtration_at=FixedFiltration,
        marks_at=closes.__getitem__,
        execution_at=opens.__getitem__,
        broker=broker,
        policy=policy,
    )

    step = backtest.steps[3]
    live = propose(
        run=RUN,
        book=step.book_before,
        strategy=ScriptedStrategy({AAA: 0.6, BBB: 0.4}),
        moment=weeks[3],
        filtration_at=FixedFiltration,
        marks_at=closes.__getitem__,
        constraints_for=broker.constraints,
        policy=policy,
    )

    assert live.intents == step.decision.intents
    assert [o.client_order_id for o in live.intents] == [
        o.client_order_id for o in step.decision.intents
    ]
    assert live.equity == step.decision.equity
    assert live.target.weights == step.decision.target.weights


def test_the_live_path_cannot_send_anything(portfolio):
    """``propose`` takes no broker. Execution is a person, not a parameter."""
    import inspect

    from engine.run import propose as live

    parameters = set(inspect.signature(live).parameters)
    assert "broker" not in parameters
    assert "execution_at" not in parameters


# -- the broker --------------------------------------------------------------


def test_submitting_the_same_order_twice_creates_one_order(portfolio):
    broker = SimulatedBroker()
    decision = decide(
        run=RUN,
        book=opening(portfolio),
        strategy=ScriptedStrategy({AAA: 1.0}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
    )
    intent = decision.intents[0]
    first = broker.submit(intent)
    second = broker.submit(intent)
    assert first == second
    assert len(broker.working) == 1
    fills = broker.advance(at(2020, 1, 10), {AAA: 100.0})
    assert len(fills) == 1


def test_an_order_with_no_price_expires_rather_than_filling_stale(portfolio):
    broker = SimulatedBroker()
    decision = decide(
        run=RUN,
        book=opening(portfolio),
        strategy=ScriptedStrategy({AAA: 1.0}),
        filtration=FixedFiltration(at(2020, 1, 3)),
        marks={AAA: 100.0},
        constraints_for=whole_shares,
    )
    broker.submit(decision.intents[0])
    fills = broker.advance(at(2020, 1, 10), {})
    assert fills == ()
    assert broker.unfilled == 1
    (state,) = broker.poll([decision.intents[0].client_order_id])
    assert state.status is OrderStatus.CANCELLED
    assert "no price" in state.message


def test_the_market_cannot_move_backwards():
    broker = SimulatedBroker()
    broker.advance(at(2020, 6, 1), {AAA: 10.0})
    with pytest.raises(ContractViolation, match="cannot move"):
        broker.advance(at(2020, 1, 1), {AAA: 10.0})


def test_an_unknown_order_polls_as_unknown_not_as_missing():
    """UNKNOWN is the honest answer after a timeout, and it is not terminal."""
    broker = SimulatedBroker()
    (state,) = broker.poll(["never-sent"])
    assert state.status is OrderStatus.UNKNOWN
    assert not state.status.is_terminal


def test_costs_are_split_because_they_behave_differently():
    costs = CostModel(commission_bps=10.0, slippage_bps=20.0)
    assert costs.fill_price(100.0, Side.BUY) == pytest.approx(100.2)
    assert costs.fill_price(100.0, Side.SELL) == pytest.approx(99.8)
    assert costs.commission(10, 100.0) == pytest.approx(1.0)


def test_the_broker_refuses_an_order_type_it_does_not_support(portfolio):
    from dataclasses import replace

    from execution.simulated import SIMULATED_CAPABILITIES

    market_only = replace(SIMULATED_CAPABILITIES, order_types=frozenset({OrderType.MARKET}))
    broker = SimulatedBroker(supports=market_only)
    with pytest.raises(ContractViolation, match="stop_limit"):
        broker.capabilities().require(OrderType.STOP_LIMIT)
    # The simulator itself models all four, the stop-limit's missed fills included.
    SimulatedBroker().capabilities().require(OrderType.STOP_LIMIT)


def test_a_target_that_restates_a_position_asks_for_no_trade(portfolio):
    """Keeping a position must not shave it.

    Sized afresh, 995 shares at their own weight come back as 985 once the cash
    buffer is held out, and the engine would sell ten shares of a position the
    strategy said to leave alone. No band is set here to hide it.
    """
    book = Book(portfolio=portfolio, cash=100_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 995, 100.0, at(2020, 1, 2))
    )
    weight = book.weights({AAA: 110.0})[AAA]
    decision = decide(
        run=RUN, book=book, strategy=ScriptedStrategy({AAA: weight}),
        filtration=FixedFiltration(at(2020, 1, 3)), marks={AAA: 110.0},
        constraints_for=whole_shares,
        policy=SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.0),
    )
    assert decision.is_flat


def test_the_strategy_is_told_each_positions_gain_on_its_cost(portfolio):
    """Dimensionless, like the weights: how a position has done, not how big it is."""
    book = Book(portfolio=portfolio, cash=100_000.0, as_of=at(2020)).apply(
        _fill(AAA, Side.BUY, 100, 100.0, at(2020, 1, 2))
    )
    seen = {}

    class Watching(ScriptedStrategy):
        def target(self, filtration, held):
            seen.update(held.gains)
            return super().target(filtration, held)

    decide(
        run=RUN, book=book, strategy=Watching({AAA: 0.1}),
        filtration=FixedFiltration(at(2020, 1, 3)), marks={AAA: 75.0},
        constraints_for=whole_shares, policy=SizingPolicy(),
    )
    assert seen == {AAA: pytest.approx(-0.25)}
