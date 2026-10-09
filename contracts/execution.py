"""Orders, broker capabilities, and the three objects that must never be confused.

The single most expensive class of bug in an automated trading system is not a
bad signal. It is the system believing something about its own positions that is
not true. Two shapes of it:

*The strategy thinks it is flat* because it emitted a close order, while the
order sits unfilled in the book. It then opens a new position on top of the old
one.

*The order is sent twice* because the network timed out after the broker accepted
it and the client retried.

Both are prevented structurally here. Three separate types — what we wanted, what
the broker says, what the ledger reconciled — so that conflating them is a type
error rather than an assumption. And every order carries a deterministic
client-generated id, so a retry of the same decision is the same order.

On numbers: quantities and prices are floats, rounded explicitly to the
instrument's lot step and price tick before an order is built. At this scale that
is safe and keeps one numeric type across pandas, the engine and the ledger.
Rounding is never left implicit.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId, PortfolioId, RunId, StrategyVersion
from contracts.temporal import utc

#: Relative slack when counting whole lots, for float division's last bit.
_LOT_TOLERANCE = 1e-9


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> float:
        """+1 for a buy, -1 for a sell: the direction the order moves a position."""
        return 1.0 if self is Side.BUY else -1.0

    @classmethod
    def closing(cls, position: float) -> Side:
        """The side that moves a signed position towards zero.

        A sell for a long, a buy for a short. Every exit, stop and liquidation
        in the system asks this rather than assuming "sell", because in a book
        that can be short, "sell" is also how exposure is *added*.
        """
        if position == 0:
            raise ContractViolation("a flat position has no closing side")
        return cls.SELL if position > 0 else cls.BUY


def split_legs(held: float, side: Side, quantity: float) -> tuple[float, float]:
    """An order against a signed position, as (closing, opening) quantities.

    Buying 150 against a short of 100 closes 100 and opens a 50 long; selling 30
    of a 100 long closes 30 and opens nothing. Exposure is reduced only by the
    closing leg, and increased only by the opening one -- which is the
    distinction every risk rule needs, and which "buy" and "sell" do not make
    once a book can be short.
    """
    if quantity < 0:
        raise ContractViolation(f"order quantities are unsigned; got {quantity}")
    delta = side.sign * quantity
    if held == 0 or (held > 0) == (delta > 0):
        return 0.0, quantity
    closing = min(quantity, abs(held))
    return closing, quantity - closing


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """How long an order stays working.

    ``OPG`` is "at the opening": a market order that participates in the opening
    auction and nothing else. It exists because it is the live counterpart of the
    backtest's fill convention -- a decision taken on Friday's close is filled at
    Monday's open in both -- so the live system and the simulation are measured
    against the same price rather than two different ones.
    """

    DAY = "day"
    GTC = "gtc"
    OPG = "opg"


class OrderStatus(str, Enum):
    """Lifecycle of an order as the broker reports it.

    ``UNKNOWN`` is a real and important state, not a placeholder: it is what the
    system knows after a timeout, and the only safe response is to reconcile
    against the broker rather than assume either outcome.
    """

    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """Whether no further change is expected."""
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass(frozen=True, slots=True)
class InstrumentConstraints:
    """What the broker will accept for one instrument.

    The sizer needs all of it. A weight of 18% means nothing until it is a whole
    number of lots at a valid price increment.
    """

    instrument: InstrumentId
    currency: str
    lot_step: float = 1.0
    min_quantity: float = 1.0
    tick_size: float = 0.01
    shortable: bool = False

    def round_quantity(self, quantity: float) -> float:
        """Round down to a tradable quantity, or zero if below the minimum.

        Down, but not past a whole lot that float division merely obscured:
        ``0.3 / 0.1`` is ``2.9999999999999996``, and truncating that sells two
        lots of a three-lot position. The tolerance is far below any real lot,
        so it only ever rescues an exact multiple, never rounds a fraction up.
        """
        if self.lot_step <= 0:
            raise ContractViolation(f"lot_step must be positive; got {self.lot_step}")
        lots = math.floor(abs(quantity) / self.lot_step + _LOT_TOLERANCE)
        # Rounded so a fractional step does not leave 0.30000000000000004 behind.
        rounded = round(lots * self.lot_step, 10)
        if rounded < self.min_quantity:
            return 0.0
        return rounded if quantity >= 0 else -rounded

    def round_price(self, price: float) -> float:
        """Round to a valid price increment."""
        if self.tick_size <= 0:
            raise ContractViolation(f"tick_size must be positive; got {self.tick_size}")
        return round(round(price / self.tick_size) * self.tick_size, 10)


@dataclass(frozen=True, slots=True)
class BrokerCapabilities:
    """What a broker adapter can do, declared rather than assumed.

    This exists because the risk layer and the sizer need broker facts, and
    reaching into the adapter for them is what fused the layers last time: the
    IBKR executor ended up importing the risk manager directly. Declaring
    capabilities keeps the dependency pointing the right way, and means a second
    adapter is a new implementation rather than an architecture change.
    """

    broker: str
    order_types: frozenset[OrderType]
    currencies: frozenset[str]
    supports_fractional: bool = False
    supports_native_trailing: bool = False
    max_leverage: float = 1.0
    calendar: str = "NYSE"

    def require(self, order_type: OrderType) -> None:
        """Raise unless the broker supports this order type."""
        if order_type not in self.order_types:
            raise ContractViolation(
                f"{self.broker} does not support {order_type.value} orders; "
                f"it supports {sorted(t.value for t in self.order_types)}"
            )


#: Every order this system sends starts with this. Anything without it was
#: placed by someone else -- by hand in TWS, or by another program.
ORDER_PREFIX = "ql-"
#: Between the strategy id and the hash. Not a character an id may contain.
ORDER_ID_SEPARATOR = "."
#: Every protective stop's id ends with this, so a fill or a working order says
#: whether a stop or a rotation produced it without a lookup.
STOP_SUFFIX = "-stop"

#: A strategy tag: lowercase letters, digits and dashes, starting with a letter.
_TAG_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
MAX_TAG_LENGTH = 16


def strategy_tag(portfolio: PortfolioId) -> str:
    """The short, readable strategy id carried on every order of this book.

    The book's name *is* the strategy id in live trading (``LiveConfig``
    enforces the format), so the id travels with each order to the broker and
    back: in IBKR it appears in the Order Ref column, in executions, and in
    statements. A book name that is not a valid tag -- research runs use free
    text -- is reduced to one deterministically.
    """
    raw = portfolio.name.lower()
    cleaned = "".join(c if c in _TAG_CHARS else "-" for c in raw).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return (cleaned or "book")[:MAX_TAG_LENGTH].rstrip("-")


def order_prefix(portfolio: PortfolioId) -> str:
    """What every order of this book starts with: ``ql-<strategy>.``.

    The dot cannot occur in a strategy id, so no strategy's prefix is the start
    of another's: ``ql-trend.`` does not match ``ql-trend-fx.…``. With a dash
    there, strategy ``trend`` would recognise ``trend-fx``'s orders as its own.
    """
    return f"{ORDER_PREFIX}{strategy_tag(portfolio)}{ORDER_ID_SEPARATOR}"


def client_order_id(
    run: RunId,
    portfolio: PortfolioId,
    instrument: InstrumentId,
    decision_time: datetime,
    side: Side,
    quantity: float,
) -> str:
    """A deterministic idempotency key for one intended order.

    Derived from the decision, not from a clock or a counter, so retrying the same
    decision after a timeout produces the same id and the broker rejects the
    duplicate instead of filling it twice.

    Shaped ``ql-<strategy>.<hash>``: the strategy id is readable, so an order
    found at the broker says which strategy sent it without looking anything
    up, and two strategies can never claim each other's fills.

    Args:
        run: The run the order belongs to.
        portfolio: The book being traded. Its name is the strategy id.
        instrument: What is being traded.
        decision_time: The decision this order implements.
        side: Buy or sell.
        quantity: Rounded quantity.
    """
    return order_prefix(portfolio) + _decision_hash(
        run, portfolio, instrument, decision_time, side, quantity
    )


def stop_order_id(
    run: RunId,
    portfolio: PortfolioId,
    instrument: InstrumentId,
    decision_time: datetime,
    side: Side,
    quantity: float,
    placement: int = 0,
) -> str:
    """The idempotency key for a protective stop: ``ql-<strategy>.<hash>-stop``.

    ``placement`` counts the stops already placed for this position since the
    decision. It exists because a stop, unlike a rotation order, is cancelled
    and placed again for the *same* decision -- when its level moves, or when
    stops are switched off and on. Reusing the first id for the replacement
    would make an idempotent broker return the cancelled order instead of
    placing the new one: the call succeeds, the journal says "placed", and the
    position is unprotected. The first placement keeps the plain decision hash,
    so ids made before this argument existed are unchanged.
    """
    extra = (f"placement {placement}",) if placement else ()
    return (
        order_prefix(portfolio)
        + _decision_hash(run, portfolio, instrument, decision_time, side, quantity, *extra)
        + STOP_SUFFIX
    )


def is_stop_order(order_id: str) -> bool:
    """Whether an order id belongs to a protective stop rather than a rotation."""
    return order_id.endswith(STOP_SUFFIX)


def _decision_hash(
    run: RunId,
    portfolio: PortfolioId,
    instrument: InstrumentId,
    decision_time: datetime,
    side: Side,
    quantity: float,
    *extra: str,
) -> str:
    material = "|".join(
        [
            str(run),
            str(portfolio),
            str(instrument),
            utc(decision_time).isoformat(),
            side.value,
            f"{quantity:.8f}",
            *extra,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def is_valid_strategy_id(value: str) -> bool:
    """Whether ``value`` can be used verbatim as a strategy tag."""
    return (
        0 < len(value) <= MAX_TAG_LENGTH
        and value[0].isalpha()
        and set(value) <= _TAG_CHARS
        and value == value.lower()
        and not value.endswith("-")
        and "--" not in value
    )


def _positive(value: float) -> bool:
    """Finite and above zero. False for NaN and infinity, which ``<= 0`` misses."""
    return math.isfinite(value) and value > 0


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """What the system decided to do. One of the three, and only the first."""

    client_order_id: str
    run: RunId
    portfolio: PortfolioId
    instrument: InstrumentId
    strategy_version: StrategyVersion
    side: Side
    quantity: float
    order_type: OrderType
    decision_time: datetime
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", utc(self.decision_time))
        # Not `quantity <= 0`: NaN fails every comparison, so that form waves
        # it through, and it surfaces much later as a NaN book.
        if not _positive(self.quantity):
            raise ContractViolation(
                f"quantity must be positive; direction is carried by side, got {self.quantity}"
            )
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ContractViolation(f"{self.order_type.value} order requires a limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ContractViolation(f"{self.order_type.value} order requires a stop_price")


@dataclass(frozen=True, slots=True)
class BrokerOrderState:
    """What the broker says about an order. Never inferred, only observed."""

    client_order_id: str
    status: OrderStatus
    observed_at: datetime
    broker_order_id: str | None = None
    filled_quantity: float = 0.0
    average_fill_price: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", utc(self.observed_at))
        if self.filled_quantity < 0:
            raise ContractViolation(f"filled_quantity cannot be negative; got {self.filled_quantity}")


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution: shares changed hands at a price, and it cost something.

    A fill is an *event*, not a state. It is the only thing that may move the
    ledger, which is what makes the book replayable: the same fills in the same
    order always produce the same positions and the same cash.

    Commission is carried separately rather than folded into the price because
    the two behave differently. Price affects the cost basis and therefore future
    profit; commission is spent immediately and never recovered. Folding it in
    understates the basis and quietly flatters every subsequent return.
    """

    client_order_id: str
    instrument: InstrumentId
    side: Side
    quantity: float
    price: float
    at: datetime
    commission: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", utc(self.at))
        # A fill is the only thing that moves the book, so a NaN here would
        # poison cash and equity for the rest of the run without an error.
        if not _positive(self.quantity):
            raise ContractViolation(
                f"fill quantity must be positive; direction is carried by side, "
                f"got {self.quantity}"
            )
        if not _positive(self.price):
            raise ContractViolation(f"fill price must be positive and finite; got {self.price}")
        if not (math.isfinite(self.commission) and self.commission >= 0):
            raise ContractViolation(f"commission cannot be negative; got {self.commission}")

    @property
    def signed_quantity(self) -> float:
        """Shares added to the position: positive on a buy, negative on a sell."""
        return self.quantity if self.side is Side.BUY else -self.quantity

    @property
    def cash_flow(self) -> float:
        """Change in cash, commission included. Negative on a buy."""
        gross = -self.signed_quantity * self.price
        return gross - self.commission


class ChargeKind(str, Enum):
    """Why cash left the account without shares changing hands."""

    MARGIN_INTEREST = "margin_interest"  # interest on a debit balance (borrowed cash)
    BORROW_FEE = "borrow_fee"            # the lender's fee on stock borrowed for a short


@dataclass(frozen=True, slots=True)
class Charge:
    """A cost of carrying the book: interest on borrowed cash, a borrow fee.

    Like a fill, a charge is an event, and applying the same charges in the same
    order always gives the same cash. It exists because leverage and shorts cost
    money every day they are held, not only when they trade: a backtest that
    charges only at the fill makes a levered or short book look cheaper than it
    is, by exactly the amount the broker bills at month end.

    Attributes:
        at: When the charge accrued up to.
        amount: What it cost, positive. Refunds are not modelled.
        kind: What it is for.
        detail: The basis, for the record (balance, rate, days).
    """

    at: datetime
    amount: float
    kind: ChargeKind
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", utc(self.at))
        if not self.amount >= 0:
            raise ContractViolation(f"a charge is a cost and cannot be negative; got {self.amount}")


@dataclass(frozen=True, slots=True)
class PositionLedgerEntry:
    """What the ledger holds after reconciling fills. The only source of truth.

    Accounting is in shares and cash, never in weights. Weights are a derived
    view. The previous system renormalised weights after a stop fired, which
    deleted the stopped position's value from the portfolio total and produced a
    50% CAGR — an accounting error does not look like an error, it looks like a
    brilliant strategy.
    """

    portfolio: PortfolioId
    instrument: InstrumentId
    quantity: float
    average_cost: float
    as_of: datetime
    realised_pnl: float = 0.0
    from_orders: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", utc(self.as_of))
        if self.quantity != 0 and self.average_cost <= 0:
            raise ContractViolation(
                f"a non-zero position in {self.instrument} needs a positive average cost; "
                f"got {self.average_cost}"
            )

    def market_value(self, price: float) -> float:
        """Mark-to-market value of the position."""
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        """Unrealised profit at ``price``."""
        return self.quantity * (price - self.average_cost)


@runtime_checkable
class ExecutionPort(Protocol):
    """The seam every broker adapter implements.

    Adding a broker is a new implementation of this port plus passing the
    conformance suite. It is never a change to the engine or the risk layer.
    """

    def capabilities(self) -> BrokerCapabilities:
        """What this broker supports."""
        ...

    def constraints(self, instrument: InstrumentId) -> InstrumentConstraints:
        """Lot step, tick size and currency for one instrument."""
        ...

    def submit(self, intent: OrderIntent) -> BrokerOrderState:
        """Send an order. Must be idempotent on ``intent.client_order_id``.

        Submitting the same id twice must not produce two orders; the second call
        returns the existing order's state.
        """
        ...

    def poll(self, client_order_ids: Sequence[str]) -> Sequence[BrokerOrderState]:
        """Current broker state for the given orders."""
        ...

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        """Request cancellation. Cancelling an unknown or finished order is not an error."""
        ...

    def positions(self, portfolio: PortfolioId) -> Sequence[PositionLedgerEntry]:
        """Positions as the broker holds them, for reconciliation against the ledger."""
        ...
