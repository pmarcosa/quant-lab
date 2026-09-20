"""A broker that fills from bar data, behind the same port as a real one.

The simulator implements :class:`~contracts.execution.ExecutionPort` in full,
including the parts that are inconvenient. That is deliberate. A simulator that
fills instantly and always is not a cheaper broker, it is a different one, and
every behaviour it omits is a code path the live system will take for the first
time with real money in it.

So orders here are **accepted, then filled later**, exactly as they are at a real
broker. ``submit`` returns ``ACCEPTED``, never ``FILLED``. Nothing fills until
:meth:`SimulatedBroker.advance` is given a later market, which is what forces the
engine to be written against "I have asked, I do not yet know" rather than
against the fiction that a decision and its fill are one event.

What is modelled: idempotency by client order id, a separate commission and
slippage, per-instrument lot and tick rules, and the refusal to fill an
instrument that has no price. What is not modelled: partial fills, queue
position, market impact and borrowing. Those are honest gaps, not oversights —
they are listed here so the next person does not have to discover them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from contracts.errors import ContractViolation
from contracts.execution import (
    BrokerCapabilities,
    BrokerOrderState,
    Fill,
    InstrumentConstraints,
    OrderIntent,
    OrderStatus,
    OrderType,
    PositionLedgerEntry,
    Side,
    TimeInForce,
)
from contracts.identifiers import InstrumentId, PortfolioId
from contracts.temporal import utc


@dataclass(frozen=True, slots=True)
class CostModel:
    """What a trade costs, split by cause so each can be argued with.

    Commission is a fee: it leaves the account and never comes back. Slippage is
    a worse price: it raises the cost basis on a buy and lowers the proceeds on a
    sell. Merging them into one "cost" number makes the basis wrong and hides
    which assumption a result is sensitive to.

    Both are in basis points of notional, applied per side.
    """

    commission_bps: float = 10.0
    slippage_bps: float = 10.0
    minimum_commission: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ContractViolation("costs cannot be negative")

    def fill_price(self, reference: float, side: Side) -> float:
        """The reference price moved against the trader."""
        drift = reference * self.slippage_bps / 10_000.0
        return reference + drift if side is Side.BUY else reference - drift

    def commission(self, quantity: float, price: float) -> float:
        """Fee on one execution."""
        return max(quantity * price * self.commission_bps / 10_000.0, self.minimum_commission)


#: Stands in for "the market clock has not started yet" when reporting the state
#: of an order at a broker that has not seen a market.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

SIMULATED_CAPABILITIES = BrokerCapabilities(
    broker="simulated",
    order_types=frozenset({OrderType.MARKET, OrderType.LIMIT, OrderType.STOP}),
    currencies=frozenset({"USD"}),
    supports_fractional=False,
    supports_native_trailing=False,
    max_leverage=1.0,
)


@dataclass(eq=False)
class SimulatedBroker:
    """An in-memory broker for backtests and dry runs.

    The field names avoid ``capabilities`` and ``constraints`` because those are
    the port's method names: a field shadowing one silently stops the adapter
    satisfying the port it claims to implement.

    Attributes:
        costs: Commission and slippage.
        overrides: Per-instrument lot and tick rules; anything absent trades in
            whole shares at a cent tick.
        supports: What this broker claims to support.
    """

    costs: CostModel = field(default_factory=CostModel)
    overrides: Mapping[InstrumentId, InstrumentConstraints] = field(default_factory=dict)
    supports: BrokerCapabilities = SIMULATED_CAPABILITIES

    def __post_init__(self) -> None:
        self._orders: dict[str, OrderIntent] = {}
        self._states: dict[str, BrokerOrderState] = {}
        self._positions: dict[tuple[PortfolioId, InstrumentId], PositionLedgerEntry] = {}
        self._now: datetime | None = None
        self._market: dict[InstrumentId, float] = {}
        self.unfilled = 0

    # -- the port ----------------------------------------------------------

    def capabilities(self) -> BrokerCapabilities:
        return self.supports

    def constraints(self, instrument: InstrumentId) -> InstrumentConstraints:
        """The lot and tick rules for one instrument. Whole shares by default."""
        override = self.overrides.get(instrument)
        if override is not None:
            return override
        return InstrumentConstraints(instrument=instrument, currency="USD")

    def submit(self, intent: OrderIntent) -> BrokerOrderState:
        """Accept an order. Idempotent on ``client_order_id``.

        A resubmission of the same id returns the existing state rather than
        creating a second order. This is the behaviour the live path depends on
        after a timeout, so the simulator has to have it or the retry logic is
        never exercised before it matters.
        """
        self.supports.require(intent.order_type)
        existing = self._states.get(intent.client_order_id)
        if existing is not None:
            if self._orders[intent.client_order_id] != intent:
                raise ContractViolation(
                    f"client_order_id {intent.client_order_id} was already used for a "
                    f"different order; ids must be derived from the decision"
                )
            return existing

        state = BrokerOrderState(
            client_order_id=intent.client_order_id,
            status=OrderStatus.ACCEPTED,
            observed_at=intent.decision_time,
            broker_order_id=f"sim-{len(self._orders) + 1:06d}",
        )
        self._orders[intent.client_order_id] = intent
        self._states[intent.client_order_id] = state
        return state

    def poll(self, client_order_ids: Sequence[str]) -> Sequence[BrokerOrderState]:
        """Current state of the given orders. Unknown ids come back as UNKNOWN."""
        observed = self._now or _EPOCH
        return [
            self._states.get(
                oid,
                BrokerOrderState(
                    client_order_id=oid,
                    status=OrderStatus.UNKNOWN,
                    observed_at=observed,
                    message="no such order at this broker",
                ),
            )
            for oid in client_order_ids
        ]

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        """Cancel a working order. Cancelling a finished or unknown one is fine."""
        state = self._states.get(client_order_id)
        observed = self._now or _EPOCH
        if state is None:
            return BrokerOrderState(
                client_order_id=client_order_id,
                status=OrderStatus.UNKNOWN,
                observed_at=observed,
                message="no such order at this broker",
            )
        if state.status.is_terminal:
            return state
        cancelled = BrokerOrderState(
            client_order_id=client_order_id,
            status=OrderStatus.CANCELLED,
            observed_at=observed,
            broker_order_id=state.broker_order_id,
        )
        self._states[client_order_id] = cancelled
        return cancelled

    def positions(self, portfolio: PortfolioId) -> Sequence[PositionLedgerEntry]:
        """Positions as this broker believes them, for reconciliation."""
        return [p for (book, _), p in sorted(self._positions.items(), key=lambda kv: str(kv[0][1]))
                if book == portfolio]

    # -- the simulation ----------------------------------------------------

    def advance(
        self,
        moment: datetime,
        prices: Mapping[InstrumentId, float],
        lows: Mapping[InstrumentId, float] | None = None,
    ) -> tuple[Fill, ...]:
        """Move the market forward and fill everything working.

        Args:
            moment: The new market time. Must not move backwards.
            prices: Reference prices — in a bar backtest, the open of the bar
                *after* the decision, because that is the first price a decision
                made on the close could actually have traded at.

        Returns:
            The fills produced, in the order the orders were submitted.

        Raises:
            ContractViolation: If time moves backwards.
        """
        moment = utc(moment)
        if self._now is not None and moment < self._now:
            raise ContractViolation(
                f"market cannot move from {self._now.isoformat()} back to {moment.isoformat()}"
            )
        self._now = moment
        self._market = dict(prices)

        fills: list[Fill] = []
        for oid, intent in self._orders.items():
            if self._states[oid].status.is_terminal:
                continue
            if intent.decision_time > moment:
                continue  # not yet in the market
            reference = prices.get(intent.instrument)
            if reference is None or not (reference > 0):
                if intent.time_in_force is TimeInForce.GTC:
                    # A resting order outlives a bar with no print. Cancelling it
                    # would quietly remove a protective stop on exactly the
                    # instrument that has stopped trading.
                    continue
                # No print, no fill. The third option -- transacting at the last
                # known price -- is the one that must never happen: it invents
                # liquidity exactly where there was none, and it does so in the
                # weeks most likely to matter. A day order that did not trade
                # expires; the engine sees it in the next decision's position
                # diff and re-decides, which is what happens in reality.
                self._states[oid] = BrokerOrderState(
                    client_order_id=oid,
                    status=OrderStatus.CANCELLED,
                    observed_at=moment,
                    broker_order_id=self._states[oid].broker_order_id,
                    message="expired unfilled: no price at this bar",
                )
                self.unfilled += 1
                continue
            rules = self.constraints(intent.instrument)

            if intent.order_type is OrderType.STOP:
                touched = self._stop_touched(intent, reference, lows)
                if touched is None:
                    continue  # rests until it is hit or cancelled
                reference = touched

            price = rules.round_price(self.costs.fill_price(reference, intent.side))
            commission = self.costs.commission(intent.quantity, price)
            fill = Fill(
                client_order_id=oid,
                instrument=intent.instrument,
                side=intent.side,
                quantity=intent.quantity,
                price=price,
                at=moment,
                commission=commission,
            )
            fills.append(fill)
            self._states[oid] = BrokerOrderState(
                client_order_id=oid,
                status=OrderStatus.FILLED,
                observed_at=moment,
                broker_order_id=self._states[oid].broker_order_id,
                filled_quantity=intent.quantity,
                average_fill_price=price,
            )
            self._record(intent.portfolio, fill)
        return tuple(fills)

    def _stop_touched(
        self,
        intent: OrderIntent,
        open_price: float,
        lows: Mapping[InstrumentId, float] | None,
    ) -> float | None:
        """The price a resting stop would fill at this bar, or None if untouched.

        Gap handling is the part that matters. If the bar opens *below* a sell
        stop, the stop did not fill at its level — the market was already past it
        when trading started, and the honest fill is the open. Filling a gapped
        stop at its trigger price credits the backtest with an exit that nobody
        could have got, and it does so precisely in the crashes the stop exists
        for, which is where the error is largest.
        """
        level = intent.stop_price
        if level is None:  # pragma: no cover - OrderIntent refuses this
            return None
        low = None if lows is None else lows.get(intent.instrument)

        if intent.side is Side.SELL:
            if open_price <= level:
                return open_price
            if low is not None and low <= level:
                return level
            return None
        if open_price >= level:
            return open_price
        return None

    def _record(self, portfolio: PortfolioId, fill: Fill) -> None:
        """Keep the broker's own view of positions, for reconciliation."""
        key = (portfolio, fill.instrument)
        existing = self._positions.get(key)
        held = existing.quantity if existing else 0.0
        basis = existing.average_cost if existing else 0.0
        resulting = held + fill.signed_quantity
        if resulting == 0.0:
            self._positions.pop(key, None)
            return
        if held == 0.0 or (held > 0) == (fill.signed_quantity > 0):
            new_basis = (
                fill.price
                if held == 0.0
                else (abs(held) * basis + fill.quantity * fill.price)
                / (abs(held) + fill.quantity)
            )
        else:
            new_basis = fill.price if abs(fill.signed_quantity) > abs(held) else basis
        self._positions[key] = PositionLedgerEntry(
            portfolio=portfolio,
            instrument=fill.instrument,
            quantity=resulting,
            average_cost=new_basis,
            as_of=fill.at,
        )

    @property
    def working(self) -> tuple[str, ...]:
        """Ids of orders that are accepted but not yet finished."""
        return tuple(
            oid for oid, state in self._states.items() if not state.status.is_terminal
        )
