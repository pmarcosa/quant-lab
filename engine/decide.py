"""One decision: from a filtration and a book to a set of orders.

This function is the whole point of the architecture. A backtest and a live
session are not two implementations that must be kept in agreement — they are two
callers of :func:`decide`, differing only in where the filtration, the book and
the prices come from. There is no second sizing path to drift, no "live
adjustment" that the backtest never exercised, and no way to add one without
this module changing.

The division of labour is fixed:

- the **strategy** says *what fraction of the book* it wants in each instrument,
  and is never told the equity (see ``contracts.targets``),
- **this module** turns fractions into whole tradable quantities, given the
  equity, the marks and the broker's lot and tick rules,
- the **broker** decides what the fill actually costs.

Sizing uses decision-time marks. The fill will happen at a different price, so
the realised weights will not match the target exactly. That gap is real, it
exists in live trading too, and the engine does not paper over it by back-solving
quantities from the fill — doing so is a lookahead.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from contracts.errors import ContractViolation
from contracts.execution import (
    InstrumentConstraints,
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
    client_order_id,
)
from contracts.identifiers import InstrumentId, PortfolioId, RunId, StrategyVersion
from contracts.strategy import Strategy
from contracts.targets import TargetIntent
from contracts.temporal import Filtration, utc
from engine.accounting import Book

#: Resolves an instrument to the broker's lot step, tick size and currency.
#: A callable rather than a port, so the engine never imports an adapter.
ConstraintsFor = Callable[[InstrumentId], InstrumentConstraints]


@dataclass(frozen=True, slots=True)
class SizingPolicy:
    """How intended weights become orders.

    Attributes:
        cash_buffer: Fraction of equity held back from sizing. Covers commission
            and the gap between the decision mark and the fill, so a fully
            invested target does not overdraw the account on a gap up.
        min_trade_fraction: Orders worth less than this fraction of equity are
            not sent. Without a no-trade band a rotation emits a handful of
            single-share orders every period, each paying full commission and
            spread to correct a rounding difference.
        allow_short: Whether a negative target weight may open a short. Off by
            default: a strategy that emits one by accident should fail, not
            silently borrow stock.
    """

    cash_buffer: float = 0.01
    min_trade_fraction: float = 0.005
    allow_short: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.cash_buffer < 1.0:
            raise ContractViolation(f"cash_buffer must be in [0, 1); got {self.cash_buffer}")
        if self.min_trade_fraction < 0.0:
            raise ContractViolation(
                f"min_trade_fraction cannot be negative; got {self.min_trade_fraction}"
            )


#: The default sizing rules. Frozen, so sharing one instance is safe.
DEFAULT_SIZING = SizingPolicy()


@dataclass(frozen=True, slots=True)
class Decision:
    """Everything one decision produced, kept together so it can be audited.

    A decision is worth storing whole. Months later the question is never "what
    did it buy" alone — it is "what did it believe, what could it see, and what
    was the book worth at the time". Splitting those across three places is how
    a rotation becomes unexplainable.
    """

    run: RunId
    portfolio: PortfolioId
    strategy_version: StrategyVersion
    decision_time: datetime
    target: TargetIntent
    equity: float
    marks: Mapping[InstrumentId, float]
    intents: tuple[OrderIntent, ...]
    strategy_state: Mapping[str, Any] = field(default_factory=dict)
    skipped: Mapping[InstrumentId, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", utc(self.decision_time))
        object.__setattr__(self, "marks", dict(self.marks))
        object.__setattr__(self, "strategy_state", dict(self.strategy_state))
        object.__setattr__(self, "skipped", dict(self.skipped))

    @property
    def is_flat(self) -> bool:
        """Whether the decision was to do nothing."""
        return not self.intents


def decide(
    *,
    run: RunId,
    book: Book,
    strategy: Strategy,
    filtration: Filtration,
    marks: Mapping[InstrumentId, float],
    constraints_for: ConstraintsFor,
    tradable: Collection[InstrumentId] | None = None,
    policy: SizingPolicy = DEFAULT_SIZING,
) -> Decision:
    """Turn a strategy's intended book into orders against the current one.

    Args:
        run: The run these orders belong to. Part of every order's identity.
        book: Cash and positions as the ledger holds them now.
        strategy: What to ask. Called exactly once.
        filtration: What is knowable at the decision time. The strategy sees only
            this; it never receives the book, the equity or the marks.
        marks: Decision-time prices for sizing and valuation. Must cover every
            held instrument and every instrument with a non-zero target.
        constraints_for: The broker's rules per instrument.
        tradable: Instruments that may be traded right now. ``None`` means all of
            them. An instrument can be priced and still untradable — halted,
            suspended, or simply not printing this period — and the two must stay
            separate: a held position needs a mark to be *valued* every period,
            which is not permission to transact in it. Collapsing them either
            stops the book being valued or lets the backtest trade through a
            halt, and both look like working code.
        policy: How to turn weights into quantities.

    Returns:
        The decision, including the orders in the sequence they must be sent:
        sells first, so the cash they release is available to the buys.

    Raises:
        ContractViolation: If a required mark is missing, if the target is
            inadmissible, or if the strategy's decision time disagrees with the
            filtration's.
    """
    moment = filtration.decision_time
    unpriced = [str(i) for i in book.positions if i not in marks or not _usable(marks[i])]
    if unpriced:
        raise ContractViolation(
            f"no usable mark at {moment.isoformat()} for held instrument(s) {sorted(unpriced)}"
        )

    # The strategy sees the shape of the book, never its size. Computing this
    # before the target keeps the equity out of the strategy's reach entirely.
    held_now = (
        book.weights({i: marks[i] for i in book.positions}) if book.positions else {}
    )
    target = strategy.target(filtration, held_now)

    if target.as_of != moment:
        raise ContractViolation(
            f"strategy decided at {target.as_of.isoformat()} from a filtration pinned at "
            f"{moment.isoformat()}; a target may only describe its own decision time"
        )
    if not policy.allow_short and any(w < 0 for w in target.weights.values()):
        shorts = sorted(str(i) for i, w in target.weights.items() if w < 0)
        raise ContractViolation(f"short targets are not permitted: {shorts}")

    wanted = {i: w for i, w in target.weights.items() if w != 0.0}
    touched = sorted(set(book.positions) | set(wanted), key=str)

    missing = [str(i) for i in touched if i not in marks or not _usable(marks[i])]
    if missing:
        raise ContractViolation(f"no usable mark at {moment.isoformat()} for {sorted(missing)}")

    equity = book.equity({i: marks[i] for i in book.positions})
    if equity <= 0:
        raise ContractViolation(f"cannot size against equity of {equity}")

    investable = equity * (1.0 - policy.cash_buffer)
    floor_value = equity * policy.min_trade_fraction

    intents: list[OrderIntent] = []
    skipped: dict[InstrumentId, str] = {}
    can_trade = None if tradable is None else frozenset(tradable)

    for instrument in touched:
        if can_trade is not None and instrument not in can_trade:
            # Recorded, not silently dropped. A position stuck open because its
            # instrument stopped printing is a risk fact, and the run should be
            # able to show how often it happened.
            skipped[instrument] = (
                "not trading this period"
                if instrument in book.positions
                else "not trading this period; not entered"
            )
            continue
        price = marks[instrument]
        rules = constraints_for(instrument)
        held = book.quantity(instrument)
        weight = wanted.get(instrument, 0.0)

        if weight == 0.0:
            # Exiting. Close the whole position: never leave a stub behind
            # because the rounding rule happened to produce one.
            desired = 0.0
        else:
            desired = rules.round_quantity(weight * investable / price)
            if desired == 0.0:
                skipped[instrument] = "below the broker's minimum size"
                continue

        delta = desired - held
        if delta == 0.0:
            continue

        # The no-trade band is measured on the *value* of the adjustment, not on
        # the share count: one share of a 900-dollar stock is not the same trade
        # as one share of a 3-dollar stock.
        if held != 0.0 and desired != 0.0 and abs(delta) * price < floor_value:
            skipped[instrument] = "inside the no-trade band"
            continue

        side = Side.BUY if delta > 0 else Side.SELL
        quantity = rules.round_quantity(abs(delta))
        if quantity == 0.0:
            skipped[instrument] = "below the broker's minimum size"
            continue
        # Closing must not be defeated by rounding: if the remainder is not a
        # tradable lot the position cannot be closed, and that is worth knowing.
        if desired == 0.0 and quantity != abs(held):
            quantity = abs(held)

        intents.append(
            OrderIntent(
                client_order_id=client_order_id(
                    run, book.portfolio, instrument, moment, side, quantity
                ),
                run=run,
                portfolio=book.portfolio,
                instrument=instrument,
                strategy_version=strategy.version,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                decision_time=moment,
                time_in_force=TimeInForce.DAY,
                reason=_reason(held, desired),
            )
        )

    # Sells first so their proceeds fund the buys; within a side, by instrument,
    # so the sequence is deterministic and two runs are comparable line by line.
    intents.sort(key=lambda o: (o.side is Side.BUY, str(o.instrument)))

    return Decision(
        run=run,
        portfolio=book.portfolio,
        strategy_version=strategy.version,
        decision_time=moment,
        target=target,
        equity=equity,
        marks={i: marks[i] for i in touched},
        intents=tuple(intents),
        strategy_state=strategy.state(),
        skipped=skipped,
    )


def _usable(price: float) -> bool:
    return isinstance(price, (int, float)) and price > 0 and price == price


def _reason(held: float, desired: float) -> str:
    if held == 0.0:
        return "open"
    if desired == 0.0:
        return "close"
    return "increase" if desired > held else "reduce"
