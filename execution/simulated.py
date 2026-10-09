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
slippage, per-instrument lot and tick rules, limit prices (an order whose limit
the bar never reaches does not fill), stop-limit orders (which can trigger and
still not fill), and the refusal to fill an instrument that has no price. What is not modelled: partial fills, queue
position, market impact and borrowing. Those are honest gaps, not oversights —
they are listed here so the next person does not have to discover them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
class CommissionSchedule:
    """A broker's published commission plan, in dollars per order.

    A flat number of basis points is the right model for a large account and the
    wrong one for a small one. A real plan charges per share with a minimum per
    order, so a 300-dollar order and a 30,000-dollar order can cost the same
    fee, and the small one pays a hundred times more of its value. Which orders
    are worth sending depends on that, and basis points cannot show it.

    The broker's own commission is ``per_share`` times the shares, raised to
    ``minimum`` and then capped at ``max_fraction`` of the trade's value (the cap
    wins over the minimum: a 20-dollar order is not charged 35 cents). Fees the
    broker passes on are added after that: the exchange's and the clearing
    house's, per share; the regulators' on sells; and a pass-through charged as
    a fraction of the commission itself.

    Attributes:
        name: The plan's name, written into a run's ledger note.
        per_share: The broker's commission per share.
        minimum: The least the broker charges for one order.
        max_fraction: The most it charges, as a fraction of the trade's value.
        exchange_per_share: Exchange fee per share. Depends on the venue and on
            whether the order added or removed liquidity; one number here.
        clearing_per_share: Clearing fee per share.
        clearing_max_fraction: Cap on the clearing fee, as a fraction of value.
        sec_fee_rate: Regulator's transaction fee on sells, times value.
        taf_per_share: Regulator's trading-activity fee on sells, per share.
        taf_maximum: The most that fee is for one order.
        cat_per_share: Audit-trail fee per share, both sides.
        pass_through_rate: Fees charged as a fraction of the commission.
        reference_share_price: When set, per-share fees are charged on
            ``value / reference_share_price`` shares rather than on the filled
            quantity. Stored history is split-adjusted: a 2009 bar of a stock
            that has since split 40 for 1 shows a price forty times lower than
            the one traded, so the same dollars buy forty times more "shares"
            and a per-share fee is overcharged forty times, on exactly the names
            that went up most. A typical share price removes that error at the
            cost of ignoring real differences between cheap and dear stocks.
    """

    name: str
    per_share: float
    minimum: float
    max_fraction: float = 0.01
    exchange_per_share: float = 0.0
    clearing_per_share: float = 0.0
    clearing_max_fraction: float = 0.005
    sec_fee_rate: float = 0.0
    taf_per_share: float = 0.0
    taf_maximum: float = 0.0
    cat_per_share: float = 0.0
    pass_through_rate: float = 0.0
    reference_share_price: float | None = None

    def __post_init__(self) -> None:
        amounts = (
            self.per_share, self.minimum, self.max_fraction, self.exchange_per_share,
            self.clearing_per_share, self.clearing_max_fraction, self.sec_fee_rate,
            self.taf_per_share, self.taf_maximum, self.cat_per_share, self.pass_through_rate,
        )
        if any(a < 0 for a in amounts):
            raise ContractViolation(f"commission plan {self.name}: no fee may be negative")
        if self.reference_share_price is not None and not self.reference_share_price > 0:
            raise ContractViolation(
                f"commission plan {self.name}: reference_share_price must be positive"
            )

    def shares_charged(self, quantity: float, price: float) -> float:
        """The share count per-share fees are charged on."""
        if self.reference_share_price is None:
            return quantity
        return quantity * price / self.reference_share_price

    def broker_commission(self, quantity: float, price: float) -> float:
        """The broker's own commission: per share, with the minimum and the cap."""
        value = quantity * price
        raised = max(self.shares_charged(quantity, price) * self.per_share, self.minimum)
        return min(raised, value * self.max_fraction)

    def fee(self, quantity: float, price: float, side: Side) -> float:
        """Everything one execution pays: commission plus passed-on fees."""
        value = quantity * price
        shares = self.shares_charged(quantity, price)
        commission = self.broker_commission(quantity, price)
        passed_on = (
            shares * self.exchange_per_share
            + min(shares * self.clearing_per_share, value * self.clearing_max_fraction)
            + shares * self.cat_per_share
            + commission * self.pass_through_rate
        )
        if side is Side.SELL:
            taf = shares * self.taf_per_share
            if self.taf_maximum > 0:
                taf = min(taf, self.taf_maximum)
            passed_on += value * self.sec_fee_rate + taf
        return commission + passed_on

    def with_reference_price(self, price: float | None) -> CommissionSchedule:
        """The same plan, charging per-share fees at a typical share price."""
        return replace(self, reference_share_price=price)


# Interactive Brokers' two plans for US stocks, as published on its commission
# page (read 2026-10-08). Regulatory rates change every year or so; the trading
# activity fee is waived until 2026-12-31 and is kept here at its earlier rate.
# The exchange fee of the tiered plan is the usual rate for an order that takes
# liquidity; auctions and orders that add liquidity cost less.

#: IBKR Fixed: one all-in rate per share, at least a dollar an order.
IBKR_FIXED = CommissionSchedule(
    name="ibkr-fixed",
    per_share=0.005,
    minimum=1.00,
    max_fraction=0.01,
    sec_fee_rate=0.0000206,
    taf_per_share=0.000166,
    taf_maximum=8.30,
    cat_per_share=0.000003,
)

#: IBKR Tiered (first tier, up to 300,000 shares a month): a lower rate and
#: minimum, with exchange and clearing fees charged on top.
IBKR_TIERED = CommissionSchedule(
    name="ibkr-tiered",
    per_share=0.0035,
    minimum=0.35,
    max_fraction=0.01,
    exchange_per_share=0.003,
    clearing_per_share=0.0002,
    clearing_max_fraction=0.005,
    sec_fee_rate=0.0000206,
    taf_per_share=0.000166,
    taf_maximum=8.30,
    cat_per_share=0.000003,
    pass_through_rate=0.000175 + 0.000563,
)

#: The plans a backtest can be asked for by name.
COMMISSION_PLANS: Mapping[str, CommissionSchedule] = {
    IBKR_FIXED.name: IBKR_FIXED,
    IBKR_TIERED.name: IBKR_TIERED,
}


@dataclass(frozen=True, slots=True)
class CostModel:
    """What a trade costs, split by cause so each can be argued with.

    Commission is a fee: it leaves the account and never comes back. Slippage is
    a worse price: it raises the cost basis on a buy and lowers the proceeds on a
    sell. Merging them into one "cost" number makes the basis wrong and hides
    which assumption a result is sensitive to.

    Slippage is in basis points of notional, applied per side. Commission is
    either the same -- ``commission_bps``, with an optional minimum -- or a
    broker's plan in dollars (``schedule``), which then replaces it.
    """

    commission_bps: float = 10.0
    slippage_bps: float = 10.0
    minimum_commission: float = 0.0
    schedule: CommissionSchedule | None = None

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0 or self.minimum_commission < 0:
            raise ContractViolation("costs cannot be negative")

    def fill_price(self, reference: float, side: Side) -> float:
        """The reference price moved against the trader."""
        drift = reference * self.slippage_bps / 10_000.0
        return reference + drift if side is Side.BUY else reference - drift

    def commission(self, quantity: float, price: float, side: Side = Side.BUY) -> float:
        """Fee on one execution. ``side`` matters only to a plan: sells pay regulators."""
        if self.schedule is not None:
            return self.schedule.fee(quantity, price, side)
        return max(quantity * price * self.commission_bps / 10_000.0, self.minimum_commission)

    def describe(self) -> str:
        """The commission assumption in a few words, for a run's note."""
        if self.schedule is None:
            return f"{self.commission_bps:g}bp"
        reference = self.schedule.reference_share_price
        return self.schedule.name + (f"@{reference:g}" if reference else "")


#: Stands in for "the market clock has not started yet" when reporting the state
#: of an order at a broker that has not seen a market.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

SIMULATED_CAPABILITIES = BrokerCapabilities(
    broker="simulated",
    order_types=frozenset(
        {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
    ),
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
        # Orders not yet filled, cancelled or expired, in submission order. A
        # backtest submits thousands of orders and almost all finish within a
        # bar; scanning only these keeps each bar's cost at the open orders,
        # not at every order the run has ever sent.
        self._working: dict[str, OrderIntent] = {}
        self._positions: dict[tuple[PortfolioId, InstrumentId], PositionLedgerEntry] = {}
        self._now: datetime | None = None
        self._market: dict[InstrumentId, float] = {}
        # Stop-limit orders whose stop has been touched and whose limit has not
        # filled: from then on they are plain limit orders, and stay so.
        self._triggered: set[str] = set()
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
        self._working[intent.client_order_id] = intent
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
        self._working.pop(client_order_id, None)
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
        highs: Mapping[InstrumentId, float] | None = None,
    ) -> tuple[Fill, ...]:
        """Move the market forward and fill everything working.

        Args:
            moment: The new market time. Must not move backwards.
            prices: Reference prices — in a bar backtest, the open of the bar
                *after* the decision, because that is the first price a decision
                made on the close could actually have traded at.
            lows: The bar's lows, which trigger sell stops.
            highs: The bar's highs, which trigger buy stops (a short's
                protection). Without them a buy stop fires only on a gap.

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
        for oid, intent in list(self._working.items()):
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
                del self._working[oid]
                self.unfilled += 1
                continue
            rules = self.constraints(intent.instrument)

            bound = None
            if intent.order_type is OrderType.STOP:
                touched = self._stop_touched(intent, reference, lows, highs)
                if touched is None:
                    continue  # rests until it is hit or cancelled
                reference = touched
            elif intent.order_type is OrderType.STOP_LIMIT:
                touched = self._stop_limit_touched(oid, intent, reference, lows, highs)
                if touched is None:
                    continue  # untouched, or triggered and waiting for its limit
                reference, bound = touched, intent.limit_price
            elif intent.order_type is OrderType.LIMIT:
                touched = self._limit_reached(intent, reference, lows, highs)
                if touched is None:
                    if intent.time_in_force is TimeInForce.GTC:
                        continue
                    # A day order whose limit the bar never reached. It expires,
                    # like one with no print: the engine sees the position it
                    # did not get and decides again.
                    self._states[oid] = BrokerOrderState(
                        client_order_id=oid,
                        status=OrderStatus.CANCELLED,
                        observed_at=moment,
                        broker_order_id=self._states[oid].broker_order_id,
                        message="expired unfilled: the limit was not reached",
                    )
                    del self._working[oid]
                    self.unfilled += 1
                    continue
                reference, bound = touched, intent.limit_price

            price = self.costs.fill_price(reference, intent.side)
            if bound is not None:
                # Slippage never carries a fill through the order's own limit.
                price = min(price, bound) if intent.side is Side.BUY else max(price, bound)
            price = rules.round_price(price)
            commission = self.costs.commission(intent.quantity, price, intent.side)
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
            del self._working[oid]
            self._record(intent.portfolio, fill)
        return tuple(fills)

    def _stop_touched(
        self,
        intent: OrderIntent,
        open_price: float,
        lows: Mapping[InstrumentId, float] | None,
        highs: Mapping[InstrumentId, float] | None = None,
    ) -> float | None:
        """The price a resting stop would fill at this bar, or None if untouched.

        Gap handling is the part that matters. If the bar opens *past* the stop
        -- below a sell stop, above a buy stop -- the stop did not fill at its
        level: the market was already beyond it when trading started, and the
        honest fill is the open. Filling a gapped stop at its trigger price
        credits the backtest with an exit that nobody could have got, and it
        does so precisely in the crashes (and, for shorts, the squeezes) the
        stop exists for, which is where the error is largest.
        """
        level = intent.stop_price
        if level is None:  # pragma: no cover - OrderIntent refuses this
            return None

        if intent.side is Side.SELL:
            low = None if lows is None else lows.get(intent.instrument)
            if open_price <= level:
                return open_price
            if low is not None and low <= level:
                return level
            return None
        high = None if highs is None else highs.get(intent.instrument)
        if open_price >= level:
            return open_price
        if high is not None and high >= level:
            return level
        return None

    @staticmethod
    def _limit_reached(
        intent: OrderIntent,
        open_price: float,
        lows: Mapping[InstrumentId, float] | None,
        highs: Mapping[InstrumentId, float] | None,
    ) -> float | None:
        """The price a limit order fills at this bar, or None if out of reach.

        A buy fills at the open when the bar opens at or under its limit, and at
        the limit when the bar only trades down to it later; a sell is the
        mirror. Without the bar's range only the open can be judged.
        """
        limit = intent.limit_price
        if limit is None:  # pragma: no cover - OrderIntent refuses this
            return None
        if intent.side is Side.BUY:
            if open_price <= limit:
                return open_price
            low = None if lows is None else lows.get(intent.instrument)
            return limit if low is not None and low <= limit else None
        if open_price >= limit:
            return open_price
        high = None if highs is None else highs.get(intent.instrument)
        return limit if high is not None and high >= limit else None

    def _stop_limit_touched(
        self,
        oid: str,
        intent: OrderIntent,
        open_price: float,
        lows: Mapping[InstrumentId, float] | None,
        highs: Mapping[InstrumentId, float] | None,
    ) -> float | None:
        """The price a stop-limit fills at this bar, or None.

        Touching the stop turns the order into a limit order; it does not fill
        it. When the bar trades down through a sell stop, the limit just below
        is reached on the way and the fill is at the stop. When the bar *opens*
        below the stop, the order becomes a limit above the market: it fills at
        the open if that is still at or over the limit, at the limit if the bar
        later trades back up to it, and otherwise not at all -- the position
        stays open through the fall the stop was there for. That is the price
        of a stop-limit, and a simulation that fills it anyway hides it.
        """
        level = intent.stop_price
        if level is None:  # pragma: no cover - OrderIntent refuses this
            return None
        if oid in self._triggered:
            return self._limit_reached(intent, open_price, lows, highs)
        selling = intent.side is Side.SELL
        gapped = open_price <= level if selling else open_price >= level
        if gapped:
            self._triggered.add(oid)
            return self._limit_reached(intent, open_price, lows, highs)
        extreme = (lows if selling else highs) or {}
        reached = extreme.get(intent.instrument)
        if reached is not None and (reached <= level if selling else reached >= level):
            self._triggered.add(oid)
            return level
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
        return tuple(self._working)
