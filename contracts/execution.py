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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId, PortfolioId, RunId, StrategyVersion
from contracts.temporal import utc


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"


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
        """Round down to a tradable quantity, or zero if below the minimum."""
        if self.lot_step <= 0:
            raise ContractViolation(f"lot_step must be positive; got {self.lot_step}")
        lots = int(abs(quantity) / self.lot_step)
        rounded = lots * self.lot_step
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

    Args:
        run: The run the order belongs to.
        portfolio: The book being traded.
        instrument: What is being traded.
        decision_time: The decision this order implements.
        side: Buy or sell.
        quantity: Rounded quantity.
    """
    material = "|".join(
        [
            str(run),
            str(portfolio),
            str(instrument),
            utc(decision_time).isoformat(),
            side.value,
            f"{quantity:.8f}",
        ]
    )
    return "ql-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


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
        if self.quantity <= 0:
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
