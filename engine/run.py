"""The loop. Deliberately thin, because everything it does is done elsewhere.

A backtest here is: for each decision time, build the filtration, call
:func:`~engine.decide.decide`, hand the orders to a broker, apply the fills the
broker reports. That is the same sequence a live session runs. The loop holds no
sizing logic, no accounting and no strategy knowledge, so there is nothing in it
that can disagree with the live path — which is the property the whole design
exists to buy.

Two ordering rules are load-bearing:

1. **Decide on a close, fill on the next open.** A decision made from a bar's
   close cannot transact at that close; the first price it could actually reach
   is the next bar's open. Filling at the decision price is the most common way a
   backtest quietly reports profit that was never available.
2. **Mark the book before deciding, not after.** Equity used for sizing is the
   equity knowable at the decision time.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from contracts.errors import ContractViolation
from contracts.execution import Fill, OrderIntent, Side
from contracts.identifiers import InstrumentId, PortfolioId, RunId
from contracts.risk import PositionRisk, RiskReview
from contracts.strategy import Strategy
from contracts.temporal import Filtration, utc
from engine.accounting import Book
from engine.decide import DEFAULT_SIZING, Decision, SizingPolicy, decide

#: Builds the view of the world pinned at one decision time.
FiltrationAt = Callable[[datetime], Filtration]

#: Prices at a moment: the marks used for sizing, or the prices a fill happens
#: at. Which one it is depends on the moment it is asked for, not on the type.
PricesAt = Callable[[datetime], Mapping[InstrumentId, float]]

#: Which instruments may be traded at a moment. Separate from having a price.
TradableAt = Callable[[datetime], Collection[InstrumentId]]


class RiskSupervision(Protocol):
    """What the engine needs from a risk layer, and nothing more.

    Declared here as a structural type rather than imported from ``risk/`` so the
    engine keeps depending on contracts alone: a second risk implementation is a
    new class, not a change to the loop.
    """

    def review(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
    ) -> RiskReview:
        ...

    def protective_orders(
        self,
        positions: Mapping[InstrumentId, PositionRisk],
        run: RunId,
        portfolio: PortfolioId,
        decision: Decision,
        constraints_for,
    ) -> Sequence[OrderIntent]:
        """Resting orders to place after a rotation. May be empty."""
        ...


@dataclass(frozen=True, slots=True)
class Step:
    """One decision and everything that followed from it."""

    decision: Decision
    fills: tuple[Fill, ...]
    book_before: Book
    book_after: Book
    equity_after: float
    marked_at: datetime
    stopped_out: tuple[str, ...] = ()
    risk_findings: tuple[str, ...] = ()

    @property
    def intents(self) -> tuple[OrderIntent, ...]:
        return self.decision.intents


@dataclass(frozen=True, slots=True)
class RunResult:
    """A completed run: every step, and the equity curve it produced."""

    run: RunId
    portfolio: PortfolioId
    steps: tuple[Step, ...]
    opening_equity: float

    @property
    def final_equity(self) -> float:
        return self.steps[-1].equity_after if self.steps else self.opening_equity

    @property
    def book(self) -> Book | None:
        return self.steps[-1].book_after if self.steps else None

    def equity_curve(self) -> tuple[tuple[datetime, float], ...]:
        """Equity after each step, marked at the step's fill time."""
        return tuple((s.marked_at, s.equity_after) for s in self.steps)

    def all_fills(self) -> tuple[Fill, ...]:
        return tuple(f for step in self.steps for f in step.fills)

    def stops_fired(self) -> int:
        """How many positions a protective stop closed."""
        return sum(len(step.stopped_out) for step in self.steps)

    def decisions(self) -> tuple[Decision, ...]:
        return tuple(step.decision for step in self.steps)


def run_backtest(
    *,
    run: RunId,
    opening: Book,
    strategy: Strategy,
    schedule: Sequence[datetime],
    filtration_at: FiltrationAt,
    marks_at: PricesAt,
    execution_at: PricesAt,
    broker,
    tradable_at: TradableAt | None = None,
    lows_at: PricesAt | None = None,
    supervisor: RiskSupervision | None = None,
    policy: SizingPolicy = DEFAULT_SIZING,
) -> RunResult:
    """Run ``strategy`` over ``schedule``, one decision per entry.

    Args:
        run: Identifies this run; part of every order id it produces.
        opening: Starting cash and positions.
        strategy: What to ask at each decision time.
        schedule: Decision times, ascending. Typically each bar's close.
        filtration_at: Builds the knowable view at a decision time.
        marks_at: Prices for sizing and valuation at a decision time.
        execution_at: Prices orders fill at, asked for at the *execution* time —
            the moment after the decision.
        broker: An ``ExecutionPort``. The simulator and a real adapter are
            interchangeable here, which is the point.
        lows_at: The bar's lows, so resting stops can be triggered on a price the
            market touched rather than only on one it closed at.
        supervisor: Optional risk layer. It sees each decision before the orders
            are sent and may only reduce exposure; the contract enforces that.
        policy: Sizing rules.

    Returns:
        Every step in order, with the books before and after.

    Raises:
        ContractViolation: If the schedule is not strictly ascending. An
            out-of-order schedule would let a later decision see earlier state.
    """
    moments = [utc(m) for m in schedule]
    # strict=False: the tail is shorter by one by construction.
    for earlier, later in zip(moments, moments[1:], strict=False):
        if later <= earlier:
            raise ContractViolation(
                f"schedule must ascend strictly; {later.isoformat()} follows {earlier.isoformat()}"
            )

    book = opening
    steps: list[Step] = []
    # The price each position's stop is measured from. Set at the rotation that
    # opened or renewed it, and never from the average cost -- see
    # risk.rules.ProtectiveStop for why that distinction is load-bearing.
    anchors: dict[InstrumentId, float] = {}
    # Closed by a stop since the last rotation. These may not be re-entered
    # before the next one: a stop followed by an immediate re-entry is a round
    # trip that pays costs and protects nothing.
    stopped_since_rotation: set[InstrumentId] = set()
    # Stop order id to instrument. The engine tracks this itself rather than
    # asking the broker, so the execution port stays as narrow as it is.
    resting_stops: dict[str, InstrumentId] = {}
    opening_equity = opening.equity(
        {i: marks_at(opening.as_of)[i] for i in opening.positions}
    )

    for index, moment in enumerate(moments):
        filtration = filtration_at(moment)
        marks = marks_at(moment)
        decision = decide(
            run=run,
            book=book.at(moment),
            strategy=strategy,
            filtration=filtration,
            marks=marks,
            constraints_for=broker.constraints,
            tradable=_tradable(tradable_at, moment, marks, stopped_since_rotation),
            policy=policy,
        )

        # Orders decided on this close reach the market at the next one. The last
        # decision in a schedule has no bar after it, so it is recorded and left
        # unfilled rather than silently filled at its own price.
        has_next = index + 1 < len(moments)
        execution_time = moments[index + 1] if has_next else moment
        before = book

        # The risk layer sees the decision before anything is sent. It may only
        # reduce exposure; RiskReview refuses anything else.
        findings: tuple[str, ...] = ()
        intents = decision.intents
        rotated = bool(decision.target.diagnostics.get("rotated", 1.0))
        if supervisor is not None:
            exposure = _position_risk(book.at(moment), marks, anchors)
            review = supervisor.review(intents, exposure, decision.equity)
            intents = review.approved
            findings = tuple(f.line() for f in review.findings)

        # Cancel-replace, in that order. A resting stop and a rotation's own sell
        # are two orders to sell the same shares: if both reach the market the
        # position goes short, which is how a long-only system acquires a short
        # book without anyone deciding to.
        selling = {i.instrument for i in intents if i.side is Side.SELL}
        for oid, instrument in list(resting_stops.items()):
            if instrument in selling:
                broker.cancel(oid)
                resting_stops.pop(oid, None)

        fills: tuple[Fill, ...] = ()
        for intent in intents:
            broker.submit(intent)
        if has_next:
            lows = None if lows_at is None else lows_at(execution_time)
            fills = broker.advance(execution_time, execution_at(execution_time), lows)
            book = book.at(execution_time).apply_all(fills)

        stopped = tuple(
            str(f.instrument) for f in fills if f.client_order_id.endswith("-stop")
        )
        for oid in [f.client_order_id for f in fills if f.client_order_id in resting_stops]:
            resting_stops.pop(oid, None)
        for name in stopped:
            stopped_since_rotation.add(InstrumentId(name))
            # A position closed by a risk mechanism stays closed. Re-entering it
            # on the next bar because the strategy still likes it turns a stop
            # into a round trip with costs and no protection. It becomes a
            # candidate again at the next rotation, like anything else.
            anchors.pop(InstrumentId(name), None)

        # Stops rest at the broker between rotations and are replaced at each
        # one, at the new anchor. Leaving an old stop in place would be
        # protecting a price that is no longer relevant.
        if rotated:
            stopped_since_rotation.clear()
        if supervisor is not None and rotated and has_next:
            for oid in list(resting_stops):
                broker.cancel(oid)
            resting_stops.clear()
            fill_prices = execution_at(execution_time)
            for instrument, position in book.positions.items():
                anchors[instrument] = _anchor_price(
                    instrument, fill_prices, position.average_cost
                )
            marks_now = {**fill_prices, **{i: anchors[i] for i in book.positions}}
            for intent in supervisor.protective_orders(
                _position_risk(book, marks_now, anchors),
                run,
                book.portfolio,
                decision,
                broker.constraints,
            ):
                broker.submit(intent)
                resting_stops[intent.client_order_id] = intent.instrument

        marked_at = execution_time if has_next else moment
        prices = execution_at(marked_at) if has_next else marks
        steps.append(
            Step(
                decision=decision,
                fills=fills,
                book_before=before,
                book_after=book,
                equity_after=book.equity(
                    {i: prices.get(i, marks[i]) for i in book.positions}
                ),
                marked_at=marked_at,
                stopped_out=stopped,
                risk_findings=findings,
            )
        )

    return RunResult(
        run=run, portfolio=opening.portfolio, steps=tuple(steps), opening_equity=opening_equity
    )


def _tradable(
    tradable_at: TradableAt | None,
    moment: datetime,
    marks: Mapping[InstrumentId, float],
    blocked: Collection[InstrumentId],
) -> Collection[InstrumentId] | None:
    """What may be traded now, minus anything a stop just closed."""
    base = set(marks) if tradable_at is None else set(tradable_at(moment))
    return base - set(blocked)


def _position_risk(
    book: Book,
    marks: Mapping[InstrumentId, float],
    anchors: Mapping[InstrumentId, float],
) -> dict[InstrumentId, PositionRisk]:
    """The view of the book a risk rule gets: quantities, marks, weights, anchors."""
    missing = sorted(str(i) for i in book.positions if i not in marks)
    if missing:
        # Guessing a mark here is exactly the silent approximation this layer
        # exists to prevent.
        raise ContractViolation(f"cannot assess risk without marks for {missing}")
    priced = {i: marks[i] for i in book.positions}
    equity = book.equity(priced)
    return {
        instrument: PositionRisk(
            instrument=instrument,
            quantity=position.quantity,
            average_cost=position.average_cost,
            mark=priced[instrument],
            weight=(
                position.market_value(priced[instrument]) / equity if equity > 0 else 0.0
            ),
            anchor=anchors.get(instrument),
        )
        for instrument, position in book.positions.items()
    }


def _anchor_price(
    instrument: InstrumentId, prices: Mapping[InstrumentId, float], fallback: float
) -> float:
    """The rotation price; the cost basis only when this bar has no print."""
    price = prices.get(instrument)
    return float(price) if price and price > 0 else float(fallback)


def propose(
    *,
    run: RunId,
    book: Book,
    strategy: Strategy,
    moment: datetime,
    filtration_at: FiltrationAt,
    marks_at: PricesAt,
    constraints_for,
    tradable: Collection[InstrumentId] | None = None,
    policy: SizingPolicy = DEFAULT_SIZING,
) -> Decision:
    """One decision, for a human to approve. The live path.

    This is the whole live decision path, and it is four lines because it is the
    same four lines the backtest runs. It submits nothing: the system proposes
    and a person approves. Execution is not wired to this function, and giving it
    a broker would be a design change, not a configuration one.
    """
    return decide(
        run=run,
        book=book.at(moment),
        strategy=strategy,
        filtration=filtration_at(moment),
        marks=marks_at(moment),
        constraints_for=constraints_for,
        tradable=tradable,
        policy=policy,
    )
