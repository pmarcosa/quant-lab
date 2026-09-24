"""Interactive Brokers, behind the same port as the simulator.

Everything the engine knows about a broker is ``ExecutionPort``. This adapter
implements it against IB Gateway or TWS through ``ib_async``, which is why going
live is a change of adapter rather than a change to the engine, the risk layer or
any strategy.

Four properties this adapter guarantees, because each one's absence has a known
and expensive failure mode:

**It refuses the wrong account.** On connection it checks that the configured
account is one the gateway manages, and that its prefix matches the declared
mode — ``DU`` for paper, ``U`` for live. A configuration pointed at the wrong
port stops here, not at the first order.

**Submission is idempotent.** Every order carries the engine's client order id in
IBKR's ``orderRef`` field. Before placing an order the adapter looks for one
with the same reference — open, filled or cancelled — and returns that instead.
Retrying after a timeout therefore cannot send the same order twice.

**Status is observed, never assumed.** ``submit`` returns what the broker
reported, and an order that cannot be found is ``UNKNOWN``, not "probably
filled". The live cycle reconciles; it does not guess.

**Fills carry the broker's execution id**, so journaling the same execution
twice — after a reconnect, say — is detectable and skipped.

**Every order carries its strategy.** The ``orderRef`` is ``ql-<strategy>.…``
(``contracts.execution.client_order_id``), and an adapter built for one
strategy only claims fills, and only manages stops, under its own prefix. With
one IB Gateway login showing several linked accounts, orders are also scoped to
the configured account.

``ib_async`` is imported lazily. Nothing else in the system needs it, and a
backtest must never acquire a dependency on whether a gateway is running.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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
from contracts.live import TradingMode

IBKR_CAPABILITIES = BrokerCapabilities(
    broker="ibkr",
    order_types=frozenset(
        {OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
    ),
    currencies=frozenset({"USD"}),
    supports_fractional=False,
    supports_native_trailing=True,
    max_leverage=1.0,
)

_ORDER_TYPES = {
    OrderType.MARKET: "MKT",
    OrderType.LIMIT: "LMT",
    OrderType.STOP: "STP",
    OrderType.STOP_LIMIT: "STP LMT",
}
_TIF = {TimeInForce.DAY: "DAY", TimeInForce.GTC: "GTC", TimeInForce.OPG: "OPG"}

#: IBKR's order states, mapped onto the engine's. ``PreSubmitted`` is what a
#: resting stop or an opening-auction order shows until it triggers, so it is a
#: working order, not a pending one.
_STATUS = {
    "PendingSubmit": OrderStatus.PENDING,
    "ApiPending": OrderStatus.PENDING,
    "PreSubmitted": OrderStatus.ACCEPTED,
    "Submitted": OrderStatus.ACCEPTED,
    "ApiUpdate": OrderStatus.ACCEPTED,
    "PendingCancel": OrderStatus.ACCEPTED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "ApiCancelled": OrderStatus.CANCELLED,
    "Inactive": OrderStatus.REJECTED,
    "ValidationError": OrderStatus.REJECTED,
}


@dataclass(frozen=True, slots=True)
class BrokerFill:
    """A fill plus the broker's execution id, for de-duplication."""

    execution_id: str
    fill: Fill


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """What the broker says the whole account is worth, in USD."""

    account: str
    net_liquidation: float
    cash: float
    observed_at: datetime
    #: The maintenance requirement and IBKR's cushion, ``1 - maintenance / net
    #: liquidation``. IBKR liquidates without a call when the cushion reaches
    #: zero; ``None`` when the gateway did not report them.
    maintenance_margin: float | None = None
    cushion: float | None = None


@dataclass(frozen=True, slots=True)
class WorkingOrder:
    """An order resting at the broker, as the broker holds it."""

    client_order_id: str
    instrument: InstrumentId
    side: Side
    quantity: float
    order_type: str
    stop_price: float | None
    status: OrderStatus


@dataclass(frozen=True, slots=True)
class ShortInfo:
    """What the broker says about borrowing an instrument to sell short.

    ``shares`` is IBKR's shortable-shares figure; ``fee_rate`` the annual
    borrow fee as a fraction, when known. The TWS API reports availability but
    not the fee (IBKR publishes fees separately), so ``fee_rate`` is usually
    ``None`` and the fee check only applies where a fee is reported.
    """

    shares: float | None
    fee_rate: float | None = None


@dataclass(frozen=True, slots=True)
class MarginVerdict:
    ok: bool
    message: str
    initial_margin_after: float | None = None
    equity_with_loan: float | None = None


def _ib_types():
    try:
        import ib_async
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ContractViolation(
            'the IBKR adapter needs ib_async: pip install -e ".[ibkr]"'
        ) from error
    return ib_async


class IBKRBroker:
    """An ``ExecutionPort`` backed by IB Gateway or TWS.

    Construct with :meth:`connect` in normal use. The constructor takes an
    already-connected client so that tests can supply a stand-in gateway.
    """

    def __init__(
        self,
        ib: Any,
        account: str,
        mode: TradingMode,
        settle_seconds: float = 2.0,
        order_prefix: str = "ql-",
    ) -> None:
        self._ib = ib
        self.account = account.strip().upper()
        self.mode = mode
        self._settle = settle_seconds
        self.order_prefix = order_prefix
        self._constraints: dict[InstrumentId, InstrumentConstraints] = {}
        self._verify_account()

    # -- connection ----------------------------------------------------------

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        client_id: int,
        account: str,
        mode: TradingMode,
        timeout: float = 20.0,
        order_prefix: str = "ql-",
    ) -> IBKRBroker:  # pragma: no cover - needs a live gateway
        """Connect to a running gateway and verify the account before returning."""
        ib_async = _ib_types()
        ib = ib_async.IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=timeout, account=account)
        except Exception as error:
            raise ContractViolation(
                f"could not reach IB Gateway/TWS at {host}:{port} ({error}). Is it "
                f"running and logged in, with API socket clients enabled?"
            ) from error
        try:
            return cls(ib, account, mode, order_prefix=order_prefix)
        except ContractViolation:
            ib.disconnect()
            raise

    def _verify_account(self) -> None:
        if not self.mode.admits(self.account):
            raise ContractViolation(
                f"account {self.account} is not a {self.mode.value} account "
                f"(paper accounts start with DU, live with U)"
            )
        managed = {a.strip().upper() for a in self._ib.managedAccounts()}
        if self.account not in managed:
            raise ContractViolation(
                f"the gateway manages {sorted(managed) or 'no accounts'}, not "
                f"{self.account}. Check the config and which gateway you started."
            )

    def disconnect(self) -> None:
        self._ib.disconnect()

    @property
    def connected(self) -> bool:
        return bool(self._ib.isConnected())

    # -- the port ------------------------------------------------------------

    def capabilities(self) -> BrokerCapabilities:
        return IBKR_CAPABILITIES

    def constraints(self, instrument: InstrumentId) -> InstrumentConstraints:
        """Tick size from IBKR's contract details; whole shares only.

        Cached per session: the minimum tick does not change between the
        proposal and the order, and asking twice doubles the round trips.
        """
        cached = self._constraints.get(instrument)
        if cached is not None:
            return cached
        details = self._ib.reqContractDetails(self._stock(instrument))
        tick = float(details[0].minTick) if details and details[0].minTick else 0.01
        found = InstrumentConstraints(
            instrument=instrument, currency="USD", lot_step=1.0,
            min_quantity=1.0, tick_size=tick,
        )
        self._constraints[instrument] = found
        return found

    def submit(self, intent: OrderIntent) -> BrokerOrderState:
        """Place an order, or return the one already placed under this id."""
        IBKR_CAPABILITIES.require(intent.order_type)
        existing = self._trade(intent.client_order_id)
        if existing is not None:
            return self._state(intent.client_order_id, existing)

        ib_async = _ib_types()
        order = ib_async.Order(
            action="BUY" if intent.side is Side.BUY else "SELL",
            totalQuantity=float(intent.quantity),
            orderType=_ORDER_TYPES[intent.order_type],
            tif=_TIF[intent.time_in_force],
            orderRef=intent.client_order_id,
            account=self.account,
            outsideRth=False,
            transmit=True,
        )
        if intent.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            order.lmtPrice = float(intent.limit_price)
        if intent.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            order.auxPrice = float(intent.stop_price)

        trade = self._ib.placeOrder(self._stock(intent.instrument), order)
        self._ib.sleep(self._settle)
        return self._state(intent.client_order_id, trade)

    def poll(self, client_order_ids: Sequence[str]) -> Sequence[BrokerOrderState]:
        return [self._state(oid, self._trade(oid)) for oid in client_order_ids]

    def cancel(self, client_order_id: str) -> BrokerOrderState:
        trade = self._trade(client_order_id)
        if trade is None:
            return self._state(client_order_id, None)
        status = _STATUS.get(trade.orderStatus.status, OrderStatus.UNKNOWN)
        if status.is_terminal:
            return self._state(client_order_id, trade)
        self._ib.cancelOrder(trade.order)
        self._ib.sleep(self._settle)
        return self._state(client_order_id, self._trade(client_order_id) or trade)

    def positions(self, portfolio: PortfolioId) -> Sequence[PositionLedgerEntry]:
        """US stock positions in the configured account, as the broker holds them."""
        now = datetime.now(timezone.utc)
        found: list[PositionLedgerEntry] = []
        for position in self._ib.positions(self.account):
            contract = position.contract
            if getattr(contract, "secType", "STK") != "STK" or position.position == 0:
                continue
            found.append(
                PositionLedgerEntry(
                    portfolio=portfolio,
                    instrument=InstrumentId(contract.symbol.upper()),
                    quantity=float(position.position),
                    average_cost=float(position.avgCost) if position.avgCost else 0.0,
                    as_of=now,
                )
            )
        return sorted(found, key=lambda p: str(p.instrument))

    # -- beyond the port: what the live cycle needs --------------------------

    def account_snapshot(self) -> AccountSnapshot:
        """Net liquidation and cash for the whole account, in USD."""
        values = {
            (v.tag, v.currency): v.value
            for v in self._ib.accountValues(self.account)
            if v.currency in ("USD", "BASE", "")
        }

        def pick(tag: str, required: bool = True, currencies=("USD", "BASE")) -> float | None:
            for currency in currencies:
                raw = values.get((tag, currency))
                if raw not in (None, ""):
                    return float(raw)
            if required:
                raise ContractViolation(f"the broker did not report {tag} for {self.account}")
            return None

        net = pick("NetLiquidation")
        maintenance = pick("MaintMarginReq", required=False)
        cushion = pick("Cushion", required=False, currencies=("", "USD", "BASE"))
        if cushion is None and maintenance is not None and net and net > 0:
            cushion = 1.0 - maintenance / net
        return AccountSnapshot(
            account=self.account,
            net_liquidation=net,
            cash=pick("TotalCashValue"),
            observed_at=datetime.now(timezone.utc),
            maintenance_margin=maintenance,
            cushion=cushion,
        )

    def fills(self) -> Sequence[BrokerFill]:
        """Every execution the gateway reports that carries one of our order ids.

        Executions without this strategy's ``orderRef`` prefix were not placed
        by it — trades made by hand in TWS, or another strategy's — and are
        deliberately excluded: the sleeve must not absorb trades it did not
        make. Reconciliation will still notice the position they create.
        """
        found: list[BrokerFill] = []
        for item in self._ib.fills():
            execution = item.execution
            reference = getattr(execution, "orderRef", "") or ""
            if not reference.startswith(self.order_prefix):
                continue
            if getattr(execution, "acctNumber", self.account).upper() != self.account:
                continue
            commission = 0.0
            report = getattr(item, "commissionReport", None)
            if report is not None and report.commission not in (None, ""):
                raw = float(report.commission)
                # IBKR reports an unset commission as a huge sentinel value.
                commission = raw if raw < 1e9 else 0.0
            when = execution.time
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            found.append(
                BrokerFill(
                    execution_id=str(execution.execId),
                    fill=Fill(
                        client_order_id=reference,
                        instrument=InstrumentId(item.contract.symbol.upper()),
                        side=Side.BUY if execution.side.upper() in ("BOT", "BUY") else Side.SELL,
                        quantity=float(execution.shares),
                        price=float(execution.price),
                        at=when,
                        commission=abs(commission),
                    ),
                )
            )
        return found

    def working_orders(self) -> Sequence[WorkingOrder]:
        """Orders resting at the broker for this account, ours or not."""
        found: list[WorkingOrder] = []
        for trade in self._open_trades():
            order = trade.order
            if (getattr(order, "account", "") or self.account).upper() != self.account:
                continue
            status = _STATUS.get(trade.orderStatus.status, OrderStatus.UNKNOWN)
            if status.is_terminal:
                continue
            aux = getattr(order, "auxPrice", None)
            found.append(
                WorkingOrder(
                    client_order_id=order.orderRef or f"external-{order.orderId}",
                    instrument=InstrumentId(trade.contract.symbol.upper()),
                    side=Side.BUY if order.action.upper() == "BUY" else Side.SELL,
                    quantity=float(order.totalQuantity),
                    order_type=order.orderType,
                    stop_price=float(aux) if aux and aux < 1e9 else None,
                    status=status,
                )
            )
        return found

    def short_availability(
        self, instruments: Sequence[InstrumentId]
    ) -> dict[InstrumentId, ShortInfo]:
        """Shortable shares per instrument (IBKR generic tick 236).

        Needs a market-data subscription for the instrument. Where IBKR sends
        nothing, the answer is ``ShortInfo(None)`` -- unknown -- and the risk
        layer treats unknown as not borrowable.
        """
        found: dict[InstrumentId, ShortInfo] = {}
        tickers = {}
        for instrument in instruments:
            tickers[instrument] = self._ib.reqMktData(
                self._stock(instrument), genericTickList="236", snapshot=False
            )
        self._ib.sleep(self._settle)
        for instrument, ticker in tickers.items():
            shares = getattr(ticker, "shortableShares", None)
            usable = shares is not None and shares == shares and shares >= 0
            found[instrument] = ShortInfo(float(shares) if usable else None)
            self._ib.cancelMktData(ticker.contract)
        return found

    def margin_check(self, intents: Sequence[OrderIntent]) -> MarginVerdict:
        """Whether the account can margin these orders, by IBKR's own what-if.

        Each order is priced with a what-if request, which returns the initial
        margin it would add without placing it. The changes are summed onto the
        current requirement and compared with equity with loan value. Summing
        is conservative: it ignores offsets between the orders themselves.
        """
        ib_async = _ib_types()
        before = equity = None
        change = 0.0
        for intent in intents:
            order = ib_async.Order(
                action="BUY" if intent.side is Side.BUY else "SELL",
                totalQuantity=float(intent.quantity), orderType="MKT",
                account=self.account, whatIf=True,
            )
            state = self._ib.whatIfOrder(self._stock(intent.instrument), order)
            if state is None or not getattr(state, "initMarginChange", None):
                return MarginVerdict(False, f"IBKR gave no margin estimate for {intent.instrument}")
            change += _number(state.initMarginChange)
            if before is None:
                before = _number(state.initMarginBefore)
                equity = _number(state.equityWithLoanBefore)
        if before is None or equity is None:
            return MarginVerdict(True, "no orders to check")
        after = before + change
        if after > equity:
            return MarginVerdict(
                False,
                f"initial margin would be {after:,.0f} against equity with loan of {equity:,.0f}",
                after, equity,
            )
        return MarginVerdict(True, f"initial margin {after:,.0f} of {equity:,.0f}", after, equity)

    def historical_bars(
        self,
        instrument: InstrumentId,
        bar_size: str,
        duration: str,
        end: datetime | None = None,
        what: str = "TRADES",
    ):
        """Price history through the same connection, as an ib_async bar list.

        ``end``: the last moment to fetch up to, for paging backwards through
        long histories; now when omitted. Regular trading hours only.

        ``what``: ``TRADES`` (adjusted for splits, not dividends: the primary
        series, see ``data.adjustments``) or ``ADJUSTED_LAST`` (also dividends).
        IBKR accepts ADJUSTED_LAST only without an end date and for bars of a day
        or less; anything else it rejects and then lets time out, so it is
        refused here instead.
        """
        if what == "ADJUSTED_LAST" and (end is not None or not _at_most_a_day(bar_size)):
            raise ContractViolation(
                f"IBKR serves ADJUSTED_LAST only up to now and for bars of a day or less; "
                f"asked for {bar_size!r}{' with an end date' if end is not None else ''}"
            )
        return self._ib.reqHistoricalData(
            self._stock(instrument),
            endDateTime="" if end is None else end.astimezone(timezone.utc),
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what,
            useRTH=True,
            formatDate=2,
        )

    # -- internals -------------------------------------------------------------

    def _stock(self, instrument: InstrumentId):
        ib_async = _ib_types()
        return ib_async.Stock(str(instrument), "SMART", "USD")

    def _open_trades(self):
        # reqAllOpenOrders covers orders placed by earlier sessions and other
        # client ids, which openTrades alone would miss after a restart.
        seen: dict[int, Any] = {}
        for trade in (*self._ib.reqAllOpenOrders(), *self._ib.openTrades()):
            seen[id(trade.order) if trade.order.permId == 0 else trade.order.permId] = trade
        return list(seen.values())

    def _trade(self, client_order_id: str):
        """The most recent trade with this reference, open or finished."""
        match = None
        for trade in (*self._ib.trades(), *self._open_trades()):
            if getattr(trade.order, "orderRef", "") == client_order_id:
                match = trade
        return match

    def _state(self, client_order_id: str, trade) -> BrokerOrderState:
        now = datetime.now(timezone.utc)
        if trade is None:
            return BrokerOrderState(
                client_order_id=client_order_id,
                status=OrderStatus.UNKNOWN,
                observed_at=now,
                message="no order with this reference at the broker",
            )
        report = trade.orderStatus
        status = _STATUS.get(report.status, OrderStatus.UNKNOWN)
        filled = float(report.filled or 0.0)
        if status is OrderStatus.ACCEPTED and filled > 0:
            status = OrderStatus.PARTIALLY_FILLED
        message = ""
        if getattr(trade, "log", None):
            message = trade.log[-1].message or ""
        return BrokerOrderState(
            client_order_id=client_order_id,
            status=status,
            observed_at=now,
            broker_order_id=str(trade.order.permId or trade.order.orderId),
            filled_quantity=filled,
            average_fill_price=float(report.avgFillPrice) if filled else None,
            message=message,
        )


def _number(value) -> float:
    """IBKR reports margin figures as strings, with a huge sentinel for 'unset'."""
    number = float(value)
    return 0.0 if number > 1e300 else number


def _at_most_a_day(bar_size: str) -> bool:
    """Whether an IBKR bar size is one day or shorter ("1 day", "1 hour", "5 mins"...)."""
    unit = bar_size.split()[-1].lower()
    if unit.startswith(("sec", "min", "hour")):
        return True
    return unit.startswith("day") and bar_size.split()[0] == "1"
