"""A stand-in for IB Gateway, built from real ib_async types.

The adapter is tested against this rather than against a mock of itself: the
objects it receives are genuine ``ib_async`` Orders, Trades, Fills and
Positions, so a field-name mistake in the adapter fails here the same way it
would against the real gateway. What is simulated is only the exchange —
orders rest until the test says the market moved.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import ib_async


class FakeGateway:
    def __init__(self, accounts=("DU1234567",), cash=100_000.0, prices=None):
        self.accounts = list(accounts)
        self.cash = cash
        self.prices = dict(prices or {})
        self.connected = True
        self._trades: list[ib_async.Trade] = []
        self._fills: list[ib_async.Fill] = []
        self._positions: dict[str, float] = {}
        self._costs: dict[str, float] = {}
        self._next_id = 1
        self._exec = 1
        self.placed = 0
        self.history: dict[str, list] = {}
        #: Shortable shares per symbol; a symbol not listed reports nothing.
        self.shortable: dict[str, float] = {}
        #: Initial margin per unit of short notional, for what-if orders.
        self.short_margin = 0.5

    # -- the API the adapter uses ------------------------------------------

    def managedAccounts(self):
        return list(self.accounts)

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False

    def sleep(self, secs=0.0):
        return True

    def reqContractDetails(self, contract):
        details = ib_async.ContractDetails()
        details.contract = contract
        details.minTick = 0.01
        return [details]

    def placeOrder(self, contract, order):
        self.placed += 1
        order.orderId = self._next_id
        order.permId = 1000 + self._next_id
        self._next_id += 1
        trade = ib_async.Trade(
            contract=contract,
            order=order,
            orderStatus=ib_async.OrderStatus(
                orderId=order.orderId, status="PreSubmitted",
                remaining=order.totalQuantity,
            ),
            fills=[],
            log=[ib_async.TradeLogEntry(time=datetime.now(timezone.utc), status="PreSubmitted")],
        )
        self._trades.append(trade)
        return trade

    def cancelOrder(self, order, manualCancelOrderTime=""):
        for trade in self._trades:
            if trade.order is order and trade.orderStatus.status not in ("Filled", "Cancelled"):
                trade.orderStatus.status = "Cancelled"
        return

    def trades(self):
        return list(self._trades)

    def openTrades(self):
        return [t for t in self._trades if t.orderStatus.status not in ("Filled", "Cancelled", "Inactive")]

    def reqAllOpenOrders(self):
        return self.openTrades()

    def fills(self):
        return list(self._fills)

    def positions(self, account=""):
        return [
            ib_async.Position(
                account=self.accounts[0],
                contract=ib_async.Stock(symbol, "SMART", "USD"),
                position=quantity,
                avgCost=self._costs.get(symbol, 0.0),
            )
            for symbol, quantity in self._positions.items()
            if quantity
        ]

    def accountValues(self, account=""):
        value = self.cash + sum(q * self.prices.get(s, 0.0) for s, q in self._positions.items())
        return [
            ib_async.AccountValue(self.accounts[0], "NetLiquidation", f"{value:.2f}", "USD", ""),
            ib_async.AccountValue(self.accounts[0], "TotalCashValue", f"{self.cash:.2f}", "USD", ""),
        ]

    def reqHistoricalData(self, contract, **kwargs):
        return list(self.history.get(contract.symbol, []))

    def serve_history(self, symbol, frame):
        """Make ``reqHistoricalData`` return this OHLCV frame as ib_async bars."""
        self.history[symbol] = [
            ib_async.BarData(
                date=label.date(), open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]), volume=float(r["volume"]),
            )
            for label, r in frame.iterrows()
        ]

    # -- test controls: the exchange ---------------------------------------

    def reqMktData(self, contract, genericTickList="", snapshot=False, regulatorySnapshot=False,
                   mktDataOptions=None):
        ticker = ib_async.Ticker(contract=contract)
        if contract.symbol in self.shortable:
            ticker.shortableShares = self.shortable[contract.symbol]
        return ticker

    def cancelMktData(self, contract):
        return None

    def whatIfOrder(self, contract, order):
        price = self.prices.get(contract.symbol, 100.0)
        held = self._positions.get(contract.symbol, 0.0)
        signed = order.totalQuantity if order.action == "BUY" else -order.totalQuantity
        short_before = max(-held, 0.0) * price
        short_after = max(-(held + signed), 0.0) * price
        equity = self.cash + sum(q * self.prices.get(s, 0.0) for s, q in self._positions.items())
        before = self._short_margin_total()
        change = self.short_margin * (short_after - short_before)
        return ib_async.OrderState(
            initMarginBefore=str(before), initMarginChange=str(change),
            initMarginAfter=str(before + change), equityWithLoanBefore=str(equity),
            equityWithLoanAfter=str(equity),
        )

    def _short_margin_total(self):
        return sum(
            self.short_margin * -q * self.prices.get(s, 100.0)
            for s, q in self._positions.items() if q < 0
        )

    def hold(self, symbol, quantity, cost):
        """Seed a position that exists before the system starts."""
        self._positions[symbol] = quantity
        self._costs[symbol] = cost

    def execute(self, trade, price, quantity=None, when=None, commission=1.0):
        """Fill some or all of a working order at a price."""
        order = trade.order
        remaining = trade.orderStatus.remaining or order.totalQuantity
        shares = remaining if quantity is None else quantity
        when = when or datetime.now(timezone.utc)
        execution = ib_async.Execution(
            execId=f"0001.{self._exec:04d}",
            time=when,
            acctNumber=self.accounts[0],
            side="BOT" if order.action == "BUY" else "SLD",
            shares=shares,
            price=price,
            orderId=order.orderId,
            permId=order.permId,
            orderRef=order.orderRef,
        )
        self._exec += 1
        report = ib_async.CommissionReport(execId=execution.execId, commission=commission, currency="USD")
        fill = ib_async.Fill(trade.contract, execution, report, when)
        trade.fills.append(fill)
        self._fills.append(fill)

        signed = shares if order.action == "BUY" else -shares
        symbol = trade.contract.symbol
        held = self._positions.get(symbol, 0.0)
        if held == 0 or (held > 0) == (signed > 0):  # opening or adding, either side
            total = abs(held) + abs(signed)
            self._costs[symbol] = (
                (abs(held) * self._costs.get(symbol, 0.0) + abs(signed) * price) / total
            )
        self._positions[symbol] = held + signed
        self.cash -= signed * price + commission

        status = trade.orderStatus
        status.filled = (status.filled or 0.0) + shares
        status.remaining = order.totalQuantity - status.filled
        status.avgFillPrice = price
        status.status = "Filled" if status.remaining <= 0 else "Submitted"
        return fill

    def opening_auction(self, opens, when=None):
        """Fill every working opening-auction order at the given opens."""
        when = when or datetime.now(timezone.utc) + timedelta(days=2)
        for trade in list(self.openTrades()):
            if trade.order.tif == "OPG" and trade.contract.symbol in opens:
                self.execute(trade, opens[trade.contract.symbol], when=when)

    def trigger_stops(self, lows, when=None, highs=None):
        """Fill resting stops whose level the market touched: sell stops on the
        low, buy stops (a short's) on the high."""
        when = when or datetime.now(timezone.utc)
        highs = highs or {}
        for trade in list(self.openTrades()):
            order = trade.order
            if order.orderType != "STP":
                continue
            symbol = trade.contract.symbol
            if order.action == "SELL":
                low = lows.get(symbol)
                if low is not None and low <= order.auxPrice:
                    self.execute(trade, order.auxPrice, when=when)
            else:
                high = highs.get(symbol)
                if high is not None and high >= order.auxPrice:
                    self.execute(trade, order.auxPrice, when=when)

    def trade_for(self, reference):
        return next(t for t in self._trades if t.order.orderRef == reference)
