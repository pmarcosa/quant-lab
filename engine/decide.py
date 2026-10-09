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

import math
import numbers
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
from contracts.targets import Holdings, TargetIntent
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
        min_order_value: Orders worth less than this many dollars are not sent,
            unless they close a position. A broker's minimum fee is a fixed
            number of dollars, so below some size an order pays more in
            commission than the adjustment is worth; unlike the no-trade band
            this is an absolute size, and it also applies to opening a new
            position. Exits are exempt: an order the strategy or a stop needs
            to get out is never held back for being small. Zero disables it.
        min_order_fraction: The same rule as a fraction of equity. A number of
            dollars stops binding as a backtest's equity compounds, so a rule
            that matters at 17,000 dollars is invisible at 500,000; the fraction
            asks what the rule does to an account that stays the size it is
            today (1,000 dollars at 17,000 is about 0.06). The larger of the two
            thresholds applies.
        limit_band: Send the rotation's orders as limit orders this far through
            the decision mark -- a buy up to ``mark * (1 + band)``, a sell down
            to ``mark * (1 - band)`` -- instead of as market orders. The order
            then cannot fill at any price, and an instrument that gaps past the
            band is not traded at all. ``None`` sends market orders.
        allow_short: Whether a negative target weight may open a short. Off by
            default: a strategy that emits one by accident should fail, not
            silently borrow stock.
        time_in_force: For the rotation's market orders. ``OPG`` -- the opening
            auction -- is the default because it is the live counterpart of the
            backtest's fill: a decision on Friday's close fills at Monday's open
            in both, so live and simulated results are measured at one price.
        leverage: Gross exposure the strategy's weights are scaled to, set per
            decision by a ``LeveragePolicy``. The strategy states its intent
            unlevered; how much of it to hold is the framework's decision, never
            the strategy's. 1.0 is no borrowing. See :func:`leverage_scale` for
            how a decision that only restates the book is treated.
        previous_leverage: The leverage set at the previous decision, so a
            decision that lowers it can cut a book it would otherwise hold.
    """

    cash_buffer: float = 0.01
    min_trade_fraction: float = 0.005
    min_order_value: float = 0.0
    min_order_fraction: float = 0.0
    limit_band: float | None = None
    allow_short: bool = False
    time_in_force: TimeInForce = TimeInForce.OPG
    leverage: float = 1.0
    previous_leverage: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.cash_buffer < 1.0:
            raise ContractViolation(f"cash_buffer must be in [0, 1); got {self.cash_buffer}")
        if self.min_trade_fraction < 0.0:
            raise ContractViolation(
                f"min_trade_fraction cannot be negative; got {self.min_trade_fraction}"
            )
        if self.min_order_value < 0.0:
            raise ContractViolation(
                f"min_order_value cannot be negative; got {self.min_order_value}"
            )
        if not 0.0 <= self.min_order_fraction < 1.0:
            raise ContractViolation(
                f"min_order_fraction must be in [0, 1); got {self.min_order_fraction}"
            )
        if self.limit_band is not None and not 0.0 <= self.limit_band < 1.0:
            raise ContractViolation(f"limit_band must be in [0, 1); got {self.limit_band}")
        if not 0.0 < self.leverage <= 10.0:
            raise ContractViolation(f"leverage must be in (0, 10]; got {self.leverage}")


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
    #: The factor the strategy's weights were scaled by (see ``leverage_scale``).
    leverage_scale: float = 1.0

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
    held_now = Holdings(
        book.weights({i: marks[i] for i in book.positions}) if book.positions else {},
        gains={
            i: marks[i] / p.average_cost - 1.0
            for i, p in book.positions.items() if p.average_cost > 0
        },
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

    scale = leverage_scale(target, policy.leverage, policy.previous_leverage)
    wanted = {i: w * scale for i, w in target.weights.items() if w != 0.0}
    touched = sorted(set(book.positions) | set(wanted), key=str)

    missing = [str(i) for i in touched if i not in marks or not _usable(marks[i])]
    if missing:
        raise ContractViolation(f"no usable mark at {moment.isoformat()} for {sorted(missing)}")

    equity = book.equity({i: marks[i] for i in book.positions})
    if equity <= 0:
        raise ContractViolation(f"cannot size against equity of {equity}")

    investable = equity * (1.0 - policy.cash_buffer)
    floor_value = equity * policy.min_trade_fraction
    smallest_order = max(policy.min_order_value, equity * policy.min_order_fraction)

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

        # A target that restates a position's present weight asks for nothing.
        # Sizing it again would not return the shares held: the cash buffer
        # shaves it and rounding takes a share, so every week a strategy said
        # "keep this" the engine would sell a little of it, and only the
        # no-trade band stood in the way.
        if held != 0.0 and weight != 0.0 and math.isclose(
            weight, held_now.get(instrument, 0.0), rel_tol=1e-12, abs_tol=0.0
        ):
            continue

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
        # A fixed minimum fee makes a small order expensive whatever it is for,
        # so this applies to opening a position as well as adjusting one. Never
        # to closing: a small exit is still an exit.
        if desired != 0.0 and quantity * price < smallest_order:
            skipped[instrument] = "below the minimum order size"
            continue

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
                order_type=OrderType.MARKET if policy.limit_band is None else OrderType.LIMIT,
                limit_price=_limit(price, side, policy.limit_band, rules),
                decision_time=moment,
                time_in_force=policy.time_in_force,
                reason=_reason(held, desired),
            )
        )

    # Exits before entries, so the cash and margin they release is there for
    # what they fund; then sells before buys, by instrument, so the sequence is
    # deterministic and two runs are comparable line by line. For a long-only
    # book this is the old "sells first" rule; with shorts, covering a short (a
    # buy) is an exit and goes early, and opening one (a sell) is an entry.
    intents.sort(key=lambda o: (o.reason in ("open", "increase"), o.side is Side.BUY, str(o.instrument)))

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
        leverage_scale=scale,
    )


def leverage_scale(
    target: TargetIntent, leverage: float, previous: float | None = None
) -> float:
    """The factor a target's weights are multiplied by, given the leverage.

    A rotation states the strategy's intent afresh and unlevered, so it is
    scaled to the leverage in full. A decision that is not a rotation restates
    the book as it stands -- already levered by the last rotation -- so scaling
    it again would compound the leverage every bar. Such a decision is only
    ever *cut*, and only when the leverage was just lowered (a drawdown or a
    thin margin cushion): the book is then scaled down to the new level if its
    gross is above it. Otherwise it is left alone -- prices drift a book's
    gross between rotations, and trimming that drift would add trades the
    validated strategy never made -- and any increase waits for the rotation.
    """
    if bool(target.diagnostics.get("rotated", 1.0)):
        return leverage
    lowered = previous is not None and leverage < previous - 1e-9
    gross = target.gross
    if lowered and gross > leverage > 0.0:
        return leverage / gross
    return 1.0


def _limit(
    mark: float, side: Side, band: float | None, rules: InstrumentConstraints
) -> float | None:
    """The limit for a rotation order: the mark, moved ``band`` against the trader."""
    if band is None:
        return None
    return rules.round_price(mark * (1.0 + band) if side is Side.BUY else mark * (1.0 - band))


def _usable(price: float) -> bool:
    """A price that can size an order: a finite, positive real number.

    ``numbers.Real`` rather than ``(int, float)``, so a NumPy integer mark is
    accepted; ``isfinite`` rather than ``price == price``, which caught NaN but
    let infinity through to size every order at zero shares without a word.
    """
    return isinstance(price, numbers.Real) and math.isfinite(price) and price > 0


def _reason(held: float, desired: float) -> str:
    """What an order does to exposure, on either side of zero.

    ``open``, ``increase``, ``reduce``, ``close`` and ``reverse``. Monitoring
    counts entries (open, increase) against exits (reduce, close, reverse), so
    this must describe exposure, not direction: covering a short is a buy and
    an exit.
    """
    if held == 0.0:
        return "open"
    if desired == 0.0:
        return "close"
    if (desired > 0) != (held > 0):
        return "reverse"
    return "increase" if abs(desired) > abs(held) else "reduce"
