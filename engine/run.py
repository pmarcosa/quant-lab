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

from contracts.errors import ContractViolation
from contracts.execution import Fill, OrderIntent
from contracts.identifiers import InstrumentId, PortfolioId, RunId
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


@dataclass(frozen=True, slots=True)
class Step:
    """One decision and everything that followed from it."""

    decision: Decision
    fills: tuple[Fill, ...]
    book_before: Book
    book_after: Book
    equity_after: float
    marked_at: datetime

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
            tradable=None if tradable_at is None else tradable_at(moment),
            policy=policy,
        )

        # Orders decided on this close reach the market at the next one. The last
        # decision in a schedule has no bar after it, so it is recorded and left
        # unfilled rather than silently filled at its own price.
        has_next = index + 1 < len(moments)
        execution_time = moments[index + 1] if has_next else moment
        before = book

        fills: tuple[Fill, ...] = ()
        if decision.intents and has_next:
            for intent in decision.intents:
                broker.submit(intent)
            fills = broker.advance(execution_time, execution_at(execution_time))
            book = book.at(execution_time).apply_all(fills)
        elif decision.intents:
            for intent in decision.intents:
                broker.submit(intent)

        marked_at = execution_time if has_next else moment
        prices = execution_at(marked_at) if has_next else marks
        steps.append(
            Step(
                decision=decision,
                fills=fills,
                book_before=before,
                book_after=book,
                equity_after=book.equity({i: prices[i] for i in book.positions}),
                marked_at=marked_at,
            )
        )

    return RunResult(
        run=run, portfolio=opening.portfolio, steps=tuple(steps), opening_equity=opening_equity
    )


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
