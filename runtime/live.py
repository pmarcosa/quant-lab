"""The live operating cycle: sync, propose, approve, submit, protect, reconcile.

This is the only module that sends orders to a real broker, so it is written
around one rule: **when anything is uncertain, refuse and say why.** Each step
below checks the conditions it depends on and stops if they do not hold, rather
than proceeding on an assumption.

The weekly cycle, in the order a person runs it:

1. ``sync`` — pull fills and order statuses from the broker into the journal,
   place or refresh protective stops, snapshot the sleeve, reconcile.
2. ``propose`` — decide on the latest complete week, run the risk layer, record
   the proposal. Nothing is sent.
3. ``approve`` — a person types the proposal's code. Only then are orders sent,
   and only if the book is exactly the one the proposal was computed against.
4. ``sync`` again after the fills, which places the new stops.

**Nothing here decides to trade on its own.** ``approve`` requires a typed
confirmation that no code path supplies. The automatic mode is a different
module (``runtime.automation``), not a flag on this one: it passes the same
:meth:`LiveSession.approvable` checks, then its own gates, and sends through the
same :meth:`LiveSession.send` -- only after a person has armed it with a typed
phrase, and never after a halt.

**The sleeve is a sub-account.** The strategy manages ``sleeve_capital``, not the
whole IBKR account. Its cash and positions are rebuilt from the journal; the
broker is reconciled against them, never silently copied into them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from contracts.errors import ContractViolation, StateIntegrityError
from contracts.execution import (
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    client_order_id,
    is_stop_order,
    order_prefix,
    split_legs,
    stop_order_id,
)
from contracts.identifiers import InstrumentId, PortfolioId, RunId, TenantId
from contracts.live import DegradationState, EventKind, TradingMode
from contracts.strategy import Strategy
from contracts.temporal import BarInterval
from engine.accounting import Book
from engine.run import _position_risk, propose
from risk.rules import (
    GrossExposureLimit,
    NetExposureLimit,
    ProtectiveStop,
    ReduceOnly,
    RiskSupervisor,
    ShortSales,
)
from runtime.config import LiveConfig
from runtime.journal import (
    Journal,
    book_fingerprint,
    fill_to_dict,
    intent_from_dict,
    intent_to_dict,
    sleeve_book,
)
from runtime.strategies import build_strategy
from runtime.wiring import Market, load_market

TENANT = TenantId("user")

#: The least intraday history an intraday strategy may trade on. The expert asks
#: for three to five years: enough regimes, rate decisions and earnings seasons
#: to judge a strategy on, however many bars one year holds.
INTRADAY_MIN_YEARS = 3.0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# -- what the cycle reports --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing reconciliation or a check noticed."""

    level: str  # "ok" | "warn" | "mismatch"
    message: str
    instrument: str = ""

    def line(self) -> str:
        where = f" [{self.instrument}]" if self.instrument else ""
        return f"{self.level.upper():<9}{self.message}{where}"


@dataclass(frozen=True, slots=True)
class Reconciliation:
    at: datetime
    findings: tuple[Finding, ...]

    @property
    def status(self) -> str:
        levels = {f.level for f in self.findings}
        if "mismatch" in levels:
            return "mismatch"
        return "warn" if "warn" in levels else "ok"


@dataclass(frozen=True, slots=True)
class ProposedOrder:
    intent: OrderIntent
    estimated_price: float
    estimated_value: float
    share_of_equity: float


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_id: str
    created_at: datetime
    expires_at: datetime
    decision_time: datetime
    data_as_of: datetime
    rotation: bool
    liquidation: bool
    state: DegradationState
    equity: float
    cash: float
    orders: tuple[ProposedOrder, ...]
    current_weights: Mapping[str, float]
    target_weights: Mapping[str, float]
    findings: tuple[str, ...]
    fingerprint: str
    #: Gross leverage the decision was sized to (1.0: no borrowing).
    leverage: float = 1.0

    @property
    def is_empty(self) -> bool:
        return not self.orders


@dataclass(frozen=True, slots=True)
class SyncReport:
    new_fills: int
    status_changes: int
    stops_placed: int
    stops_cancelled: int
    snapshot: Mapping[str, Any]
    reconciliation: Reconciliation
    state: DegradationState
    notes: tuple[str, ...] = ()


# -- the session -------------------------------------------------------------


@dataclass
class LiveSession:
    """Everything the live cycle needs, wired once.

    Attributes:
        config: Validated live configuration.
        broker: The IBKR adapter, or anything with the same methods.
        store_root: The ``var/store`` directory.
        clock: Returns the current UTC time. Injected so that tests — and a
            rehearsal — control time instead of the wall clock.
    """

    config: LiveConfig
    broker: Any
    store_root: Path
    clock: Callable[[], datetime] = now_utc
    market_loader: Callable[[Path], Market] | None = None
    #: Builds the strategy. Defaults to the registry in ``runtime.strategies``;
    #: tests inject strategies that are not deployable.
    strategy_factory: Callable[[LiveConfig], Strategy] | None = None
    journal: Journal = field(init=False)

    def __post_init__(self) -> None:
        self.journal = Journal(self.config.journal_path)
        if self.broker is not None and getattr(self.broker, "mode", self.config.mode) is not (
            self.config.mode
        ):
            raise ContractViolation(
                f"the broker is connected in {self.broker.mode.value} mode but the "
                f"config says {self.config.mode.value}"
            )
        if self.broker is not None and getattr(self.broker, "account", self.config.account) != (
            self.config.account
        ):
            raise ContractViolation(
                f"the broker is connected to {self.broker.account} but strategy "
                f"{self.config.strategy_id!r} is configured for {self.config.account}"
            )
        self._check_identity()

    def _check_identity(self) -> None:
        """Refuse a journal written by another strategy or for another account.

        The journal's folder is named after the strategy, so this only fires when
        files have been moved or a config edited under a running sleeve. Either
        way, trading one strategy's book with another's settings is exactly the
        silent error this guards against.
        """
        opened = self.journal.last(EventKind.OPENED)
        if opened is None:
            return
        recorded = opened.payload.get("strategy_id")
        if recorded is not None and recorded != self.config.strategy_id:
            raise StateIntegrityError(
                f"the journal at {self.config.journal_path} belongs to strategy "
                f"{recorded!r}, not {self.config.strategy_id!r}"
            )
        if "universe" in opened.payload:  # journals opened before universes lack the key
            recorded_universe = opened.payload["universe"]
            current = _universe_record(self.universe())
            if (recorded_universe or {}).get("fingerprint") != (current or {}).get("fingerprint"):
                raise StateIntegrityError(
                    f"the {self.config.strategy_id!r} sleeve was opened on universe "
                    f"{_universe_label(recorded_universe)}, but the config now gives "
                    f"{_universe_label(current)}. Another universe is another strategy: "
                    f"open a new strategy id for it."
                )
        account = opened.payload.get("account")
        if account is not None and account != self.config.account:
            raise StateIntegrityError(
                f"the {self.config.strategy_id!r} sleeve was opened in account {account}, "
                f"but the config now says {self.config.account}. A sleeve does not move "
                f"between accounts; open a new strategy id for the new account."
            )

    # -- identity and wiring -------------------------------------------------

    @property
    def portfolio(self) -> PortfolioId:
        """The book, named by the strategy id -- which every order then carries."""
        return PortfolioId(TENANT, self.config.strategy_id)

    @property
    def run(self) -> RunId:
        return RunId(f"{self.config.strategy_id}-{self.config.mode.value}")

    @property
    def order_prefix(self) -> str:
        """``ql-<strategy_id>.``: how this strategy's orders are recognised."""
        return order_prefix(self.portfolio)

    def strategy(self) -> Strategy:
        if self.strategy_factory is not None:
            return self.strategy_factory(self.config)
        s = self.config.strategy
        return build_strategy(s.name, s.params)

    @property
    def interval(self) -> BarInterval:
        """The strategy's bar size. Everything periodic follows it."""
        return self.strategy().filtration_spec.interval

    def universe(self):
        """The strategy's universe list, or None for every instrument in the store."""
        from runtime.wiring import universe_list

        return universe_list(self.config.strategy.universe)

    def market(self) -> Market:
        if self.market_loader is not None:
            return self.market_loader(self.store_root)
        universe = self.universe()
        return load_market(self.store_root, interval=self.interval,
                           symbols=universe.symbols if universe else None)

    def stop_rule(self) -> ProtectiveStop | None:
        return self.config.risk.stop()

    def supervisor(
        self,
        state: DegradationState,
        availability: Mapping[InstrumentId, float | None] | None = None,
        borrow_fees: Mapping[InstrumentId, float | None] | None = None,
    ) -> RiskSupervisor:
        """The risk rules for this state, in order.

        Short sales are checked against what the broker says can be borrowed
        when ``availability`` is given; without it (no short orders in the
        decision) only the on/off switch applies.
        """
        r = self.config.risk
        rules: list = [
            ShortSales(
                allowed=r.allow_short, availability=availability,
                borrow_fees=borrow_fees or {}, max_borrow_fee=r.max_borrow_fee,
            ),
            GrossExposureLimit(maximum=r.max_gross),
            NetExposureLimit(minimum=r.min_net, maximum=r.net_cap),
        ]
        if not state.permits_entries:
            rules.append(ReduceOnly(reason=f"the system is {state.value.replace('_', '-')}"))
        return RiskSupervisor(rules=tuple(rules), stop=self.stop_rule())

    def book(self) -> Book:
        return sleeve_book(self.journal, self.portfolio)

    # -- the degradation ladder ---------------------------------------------

    def state(self) -> DegradationState:
        last = self.journal.last(EventKind.STATE_CHANGE)
        return DegradationState(last.payload["to"]) if last else DegradationState.NORMAL

    def degrade(self, to: DegradationState, reason: str, by: str = "system") -> DegradationState:
        """Move down the ladder. Never up: see :meth:`clear`.

        A halt also disarms automation: after a halt nothing may resume on its
        own, and that includes sending orders without a person.
        """
        current = self.state()
        target = current.worst(to)
        if target is not current:
            self.journal.append(
                EventKind.STATE_CHANGE, self.clock(),
                {"from": current.value, "to": target.value, "reason": reason, "by": by},
            )
            if target is DegradationState.HALTED and self.automation_scope() is not None:
                self.journal.append(EventKind.AUTOMATION, self.clock(), {
                    "action": "disarm", "reason": f"halted: {reason}", "by": "system",
                })
        return target

    def automation_scope(self) -> str | None:
        """What automation is armed for now -- ``exits`` or ``full`` -- or ``None``."""
        for event in reversed(self.journal.events(EventKind.AUTOMATION)):
            action = event.payload.get("action")
            if action == "arm":
                return str(event.payload["scope"])
            if action == "disarm":
                return None
        return None

    def restrict(self, to: DegradationState, reason: str) -> DegradationState:
        """A person moves the system *down* the ladder: a pause or a halt.

        A reduce-only set this way is the person's, and monitoring will not lift
        it -- going on holiday is a reason monitoring cannot see.
        """
        if len(reason.strip()) < 10:
            raise ContractViolation("a pause or halt needs a real reason (at least ten characters)")
        current = self.state()
        if to.rank <= current.rank:
            raise ContractViolation(
                f"the system is already {current.value}; use `ql live clear` to move up"
            )
        self.journal.append(
            EventKind.STATE_CHANGE, self.clock(),
            {"from": current.value, "to": to.value, "reason": reason.strip(), "by": "person"},
        )
        return to

    def apply_assessment(self, recommended: DegradationState, reasons: Sequence[str]) -> DegradationState:
        """Let monitoring move the state, within the rules of the ladder.

        A halt is applied and sticks. A reduce-only that monitoring imposed is
        lifted by monitoring when the evidence clears; one a person imposed is
        not touched. Nothing here ever lifts a halt.
        """
        current = self.state()
        last = self.journal.last(EventKind.STATE_CHANGE)
        set_by = last.payload.get("by") if last else "system"
        reason = "; ".join(reasons) or "monitoring found nothing outside its bands"
        if recommended is DegradationState.HALTED:
            return self.degrade(DegradationState.HALTED, "monitoring: " + reason)
        if current is DegradationState.HALTED:
            return current
        if current is DegradationState.REDUCE_ONLY and set_by != "monitor":
            # A person's pause, or the margin guard's: not monitoring's to lift.
            return current
        if recommended is not current:
            self.journal.append(
                EventKind.STATE_CHANGE, self.clock(),
                {"from": current.value, "to": recommended.value,
                 "reason": "monitoring: " + reason, "by": "monitor"},
            )
        return recommended

    def clear(self, to: DegradationState, reason: str) -> DegradationState:
        """A person moves the system back up the ladder, with a written reason."""
        if len(reason.strip()) < 10:
            raise ContractViolation(
                "clearing a degraded state needs a real reason (at least ten characters); "
                "it is the record of why it was safe to resume"
            )
        current = self.state()
        if to.rank >= current.rank:
            raise ContractViolation(
                f"clear moves up the ladder; {to.value} is not above {current.value}"
            )
        last = self.reconcile(record=False)
        if last.status == "mismatch":
            raise ContractViolation(
                "reconciliation still shows a mismatch; resolve it with `ql live adjust` "
                "before clearing"
            )
        self.journal.append(
            EventKind.STATE_CHANGE, self.clock(),
            {"from": current.value, "to": to.value, "reason": reason.strip(), "by": "person"},
        )
        return to

    # -- opening the sleeve --------------------------------------------------

    def init(self, adopt: Sequence[str] = ()) -> Book:
        """Record the sleeve's opening balance.

        Adopted positions are taken at the quantity and average cost the broker
        reports, marked at the latest close. The sleeve's cash is whatever of
        ``sleeve_capital`` the adopted positions do not already account for.

        Raises:
            ContractViolation: If the sleeve is already open, an adopted ticker
                is not held, is unmanaged, or the adopted positions are worth
                more than the sleeve's capital.
        """
        if self.journal.is_open:
            raise ContractViolation(
                f"the {self.config.mode.value} sleeve is already open; it opens once"
            )
        held = {str(p.instrument): p for p in self.broker.positions(self.portfolio)}
        wanted = [a.upper() for a in adopt]
        for ticker in wanted:
            if ticker not in held:
                raise ContractViolation(f"cannot adopt {ticker}: the account does not hold it")
            if ticker in self.config.unmanaged:
                raise ContractViolation(f"cannot adopt {ticker}: it is listed as unmanaged")

        market = self.market()
        latest = market.schedule[-1]
        marks = market.window.marks_at(latest)
        positions = []
        value = 0.0
        for ticker in wanted:
            instrument = InstrumentId(ticker)
            if instrument not in marks:
                raise ContractViolation(
                    f"cannot adopt {ticker}: it is not in the price data, so the strategy "
                    f"could not value or manage it. Add it to the universe first."
                )
            position = held[ticker]
            value += position.quantity * marks[instrument]
            positions.append({
                "instrument": ticker,
                "quantity": position.quantity,
                "average_cost": position.average_cost,
                "mark": marks[instrument],
            })
        cash = self.config.sleeve_capital - value
        if cash < 0:
            raise ContractViolation(
                f"the adopted positions are worth {value:,.0f}, more than the sleeve's "
                f"capital of {self.config.sleeve_capital:,.0f}"
            )
        account = self.broker.account_snapshot()
        self.journal.append(
            EventKind.OPENED, self.clock(),
            {
                "cash": cash, "positions": positions,
                "sleeve_capital": self.config.sleeve_capital,
                "strategy_id": self.config.strategy_id,
                "strategy": {"name": self.config.strategy.name,
                             "params": dict(self.config.strategy.params)},
                "strategy_version": str(self.strategy().version),
                "universe": _universe_record(self.universe()),
                "interval": self.interval.value,
                "account": self.config.account, "mode": self.config.mode.value,
                "account_net_liquidation": account.net_liquidation,
                "data_as_of": latest.isoformat(),
            },
        )
        return self.book()

    # -- synchronising with the broker ---------------------------------------

    def _our_order_ids(self) -> set[str]:
        ids = {e.payload["client_order_id"] for e in self.journal.events(EventKind.SUBMISSION)}
        ids |= {e.payload["client_order_id"] for e in self.journal.events(EventKind.STOP_PLACED)}
        return ids

    def sync(self) -> SyncReport:
        """Bring the journal up to date with the broker, then protect and check."""
        if not self.journal.is_open:
            raise StateIntegrityError("the sleeve is not open; run `ql live init` first")
        now = self.clock()
        notes: list[str] = []

        # 1. Fills. De-duplicated on the broker's execution id, and limited to
        # orders this journal sent: a trade made by hand is not the sleeve's.
        journaled = {
            e.payload.get("execution_id") for e in self.journal.events(EventKind.FILL)
        }
        ours = self._our_order_ids()
        new_fills = 0
        for found in sorted(self.broker.fills(), key=lambda f: f.fill.at):
            if found.execution_id in journaled or found.fill.client_order_id not in ours:
                continue
            payload = fill_to_dict(found.fill)
            payload["execution_id"] = found.execution_id
            payload["stop"] = is_stop_order(found.fill.client_order_id)
            self.journal.append(EventKind.FILL, now, payload)
            new_fills += 1
            if payload["stop"]:
                notes.append(f"protective stop filled: {found.fill.instrument}")

        # 2. Order statuses, recorded only when they change.
        last_status: dict[str, str] = {}
        for event in self.journal.events(EventKind.SUBMISSION, EventKind.ORDER_STATUS):
            oid = event.payload["client_order_id"]
            last_status[oid] = event.payload.get("status", "submitted")
        pending = [oid for oid, st in last_status.items() if st not in ("filled", "cancelled", "rejected")]
        changes = 0
        for state in self.broker.poll(pending):
            if state.status.value != last_status.get(state.client_order_id):
                self.journal.append(
                    EventKind.ORDER_STATUS, now,
                    {
                        "client_order_id": state.client_order_id,
                        "status": state.status.value,
                        "filled": state.filled_quantity,
                        "average_fill_price": state.average_fill_price,
                        "message": state.message,
                    },
                )
                changes += 1

        # 3. Stops, then 4. a snapshot, then 5. reconciliation.
        placed, cancelled = self.place_stops()
        snapshot = self.snapshot()
        notes.extend(self._margin_guard(snapshot.get("account_cushion")))
        reconciliation = self.reconcile()
        if reconciliation.status == "mismatch":
            self.degrade(
                DegradationState.HALTED,
                "reconciliation mismatch: " + "; ".join(
                    f.message for f in reconciliation.findings if f.level == "mismatch"
                ),
            )
        return SyncReport(
            new_fills=new_fills, status_changes=changes,
            stops_placed=placed, stops_cancelled=cancelled,
            snapshot=snapshot, reconciliation=reconciliation,
            state=self.state(), notes=tuple(notes),
        )

    def snapshot(self) -> dict[str, Any]:
        """Accrue carrying costs, mark the sleeve at the latest closes, record it."""
        market = self.market()
        latest = market.schedule[-1]
        marks = market.window.marks_at(latest)
        book = self.book()
        missing = [str(i) for i in book.positions if i not in marks]
        if missing:
            raise ContractViolation(f"no price data to mark {missing}; refresh the data")
        if self._accrue_financing(book, marks):
            book = self.book()
        equity = book.equity(marks)
        account = self.broker.account_snapshot()
        payload = {
            "marks_as_of": latest.isoformat(),
            "sleeve_equity": equity,
            "cash": book.cash,
            "positions": [
                {"instrument": str(i), "quantity": p.quantity, "mark": marks[i],
                 "average_cost": p.average_cost}
                for i, p in sorted(book.positions.items(), key=lambda kv: str(kv[0]))
            ],
            "account_net_liquidation": account.net_liquidation,
            "account_cash": account.cash,
            "account_maintenance_margin": getattr(account, "maintenance_margin", None),
            "account_cushion": getattr(account, "cushion", None),
        }
        self.journal.append(EventKind.SNAPSHOT, self.clock(), payload)
        return payload

    def readiness(self, market: Market | None = None) -> tuple[str, ...]:
        """What an intraday strategy still lacks before it may trade; empty when ready.

        The project's expert's prerequisites that the system can check:
        the exchange's session calendar, and three to five years of intraday
        history (the market's regimes are measured in calendar time, not in
        bars). Daily and weekly strategies need neither and are always ready.
        """
        interval = self.interval
        if not interval.is_intraday:
            return ()
        from data import calendar

        problems = []
        if not calendar.is_available():
            problems.append(
                "the exchange calendar is not installed (pip install -e \".[intraday]\")"
            )
        market = market or self.market()
        span = (market.schedule[-1] - market.schedule[0]).days / 365.25 if market.schedule else 0
        if span < INTRADAY_MIN_YEARS:
            problems.append(
                f"{span:.1f} years of {interval.frequency} history in the store; at least "
                f"{INTRADAY_MIN_YEARS:g} are needed (`ql data backfill`)"
            )
        return tuple(problems)

    def _margin_guard(self, cushion: float | None) -> list[str]:
        """Stop new exposure while IBKR's margin cushion is thin; lift it once it is not.

        IBKR liquidates without a call when the cushion reaches zero. Well
        before that -- below ``leverage.cushion_critical`` -- nothing new is
        opened, and the leverage rule sizes the next decision back to its
        floor. The guard lifts its own reduce-only once the cushion is back
        above ``leverage.cushion_warning``; it never lifts anyone else's.
        """
        if cushion is None:
            return []
        lv = self.config.leverage
        state = self.state()
        last = self.journal.last(EventKind.STATE_CHANGE)
        if cushion < lv.cushion_critical and state.permits_entries:
            self.degrade(
                DegradationState.REDUCE_ONLY,
                f"margin cushion {cushion:.0%} is below {lv.cushion_critical:.0%}: no new exposure",
                by="margin",
            )
            return [f"margin cushion {cushion:.0%}: reduce-only"]
        if (
            state is DegradationState.REDUCE_ONLY and last is not None
            and last.payload.get("by") == "margin" and cushion >= lv.cushion_warning
        ):
            self.journal.append(EventKind.STATE_CHANGE, self.clock(), {
                "from": state.value, "to": DegradationState.NORMAL.value, "by": "margin",
                "reason": f"margin cushion back to {cushion:.0%}",
            })
            return [f"margin cushion {cushion:.0%}: reduce-only lifted"]
        return []

    def _accrue_financing(self, book: Book, marks: Mapping[InstrumentId, float]) -> bool:
        """Charge the sleeve for carrying borrowed cash or stock since the last snapshot.

        The same model the backtest charges, so live and simulated returns pay
        for leverage alike. It is an estimate of what IBKR bills: the sleeve's
        debit and short value at the rates in ``financing``. The actual monthly
        interest can be trued up with ``ql live adjust --cash-delta``.
        """
        if book.cash >= 0 and all(p.quantity > 0 for p in book.positions.values()):
            return False
        last = self.journal.last(EventKind.SNAPSHOT) or self.journal.last(EventKind.OPENED)
        if last is None:
            return False
        now = self.clock()
        charges = self.config.financing.model().charges(
            book, {i: marks[i] for i in book.positions}, last.at, now
        )
        for charge in charges:
            self.journal.append(EventKind.FINANCING, now, {
                "amount": charge.amount, "kind": charge.kind.value, "detail": charge.detail,
                "from": last.at.isoformat(), "through": now.isoformat(),
            })
        return bool(charges)

    # -- reconciliation ------------------------------------------------------

    def reconcile(self, record: bool = True) -> Reconciliation:
        """The journal's sleeve against what the broker holds, on signed positions.

        A **mismatch** halts the system. It is any difference that means the
        sleeve believes in exposure the account does not have, or has it on the
        wrong side:

        - the sign differs (the sleeve is short, the account long, or the
          reverse): a cover sent for a short that is really a long *adds*
          exposure instead of removing it -- the expert's first kill condition;
        - the account holds less, in the sleeve's direction, than the sleeve
          believes: the system would size and stop shares that are not there;
        - in a ``dedicated`` account, anything else as well: shares beyond the
          sleeve's, or a position the sleeve does not hold at all (unless listed
          as unmanaged). The account is the strategy's; an orphan position means
          a fill nobody recorded or a trade made by hand.

        In a ``shared`` account those last differences are warnings: other
        holdings are expected.
        """
        book = self.book()
        findings: list[Finding] = []
        broker_positions = {
            str(p.instrument): p.quantity for p in self.broker.positions(self.portfolio)
        }
        sleeve = {str(i): p.quantity for i, p in book.positions.items()}
        dedicated = self.config.account_scope == "dedicated"
        extra_level = "mismatch" if dedicated else "warn"

        for ticker, quantity in sorted(sleeve.items()):
            held = broker_positions.get(ticker, 0.0)
            if held != 0 and (held > 0) != (quantity > 0):
                findings.append(Finding(
                    "mismatch",
                    f"the sleeve is {'long' if quantity > 0 else 'short'} {abs(quantity):g} but "
                    f"the account is {'long' if held > 0 else 'short'} {abs(held):g}",
                    ticker,
                ))
            elif abs(held) + 1e-9 < abs(quantity):
                findings.append(Finding(
                    "mismatch",
                    f"the sleeve holds {quantity:g} but the broker holds {held:g}",
                    ticker,
                ))
            elif abs(held) > abs(quantity) + 1e-9:
                findings.append(Finding(
                    extra_level,
                    f"the broker holds {held:g}, {abs(held) - abs(quantity):g} more than the "
                    f"sleeve; the extra shares are outside the strategy",
                    ticker,
                ))
        for ticker, quantity in sorted(broker_positions.items()):
            if ticker in sleeve or ticker in self.config.unmanaged or not quantity:
                continue
            findings.append(Finding(
                extra_level,
                f"held at the broker ({quantity:g}) but not by the sleeve and not listed as "
                f"unmanaged",
                ticker,
            ))

        # Orders we sent that the broker no longer knows about.
        working = {w.client_order_id for w in self.broker.working_orders()}
        last_status: dict[str, str] = {}
        for event in self.journal.events(EventKind.SUBMISSION, EventKind.ORDER_STATUS):
            last_status[event.payload["client_order_id"]] = event.payload.get("status", "submitted")
        for oid, status in last_status.items():
            if status in ("submitted", "accepted", "pending", "partially_filled") and (
                oid not in working
            ):
                states = self.broker.poll([oid])
                if states and states[0].status is OrderStatus.UNKNOWN:
                    findings.append(Finding(
                        "mismatch", f"order {oid} was sent but the broker has no record of it"
                    ))

        # Every sleeve position should have a resting stop on its closing side.
        if self.stop_rule() is not None:
            protected = {
                (str(w.instrument), w.side) for w in self.broker.working_orders()
                if w.client_order_id.startswith(self.order_prefix)
                and is_stop_order(w.client_order_id)
            }
            for ticker, quantity in sorted(sleeve.items()):
                if (ticker, Side.closing(quantity)) not in protected:
                    findings.append(Finding("warn", "no protective stop resting at the broker", ticker))

        # A short whose shares can no longer be borrowed can be bought in by the
        # broker whatever the strategy decides. The system cannot prevent a
        # recall; it can say so while there is still time to act.
        shorts = [InstrumentId(t) for t, q in sleeve.items() if q < 0]
        report = getattr(self.broker, "short_availability", None)
        if shorts and report is not None:
            found = report(shorts)
            for instrument in shorts:
                info = found.get(instrument)
                if info is None or not info.shares:
                    findings.append(Finding(
                        "warn",
                        "short position with no borrow availability reported: recall or "
                        "buy-in risk",
                        str(instrument),
                    ))

        account = self.broker.account_snapshot()
        if account.cash + 1e-6 < book.cash:
            findings.append(Finding(
                "warn",
                f"the account holds {account.cash:,.2f} in cash, less than the sleeve's "
                f"{book.cash:,.2f}",
            ))

        result = Reconciliation(at=self.clock(), findings=tuple(findings))
        if record:
            self.journal.append(
                EventKind.RECONCILIATION, result.at,
                {"status": result.status, "findings": [f.line() for f in findings]},
            )
        return result

    def adjust(
        self, instrument: str | None, quantity: float | None, reason: str,
        average_cost: float | None = None, cash_delta: float = 0.0,
    ) -> Book:
        """Record a correction to the sleeve. Always with a reason."""
        if len(reason.strip()) < 10:
            raise ContractViolation("an adjustment needs a real reason (at least ten characters)")
        payload: dict[str, Any] = {"reason": reason.strip(), "cash_delta": float(cash_delta)}
        if instrument is not None:
            if quantity is None:
                raise ContractViolation(
                    "an adjustment to a position needs the correct signed quantity "
                    "(negative for a short, 0 to remove it)"
                )
            if quantity < 0 and not self.config.risk.allow_short:
                raise ContractViolation(
                    f"{instrument}: a negative quantity records a short, and this strategy "
                    f"does not allow shorts. If the account really is short, that is an "
                    f"incident; if not, the sign is wrong."
                )
            payload["instrument"] = instrument.upper()
            payload["quantity"] = float(quantity)
            if average_cost is not None:
                payload["average_cost"] = float(average_cost)
        self.journal.append(EventKind.ADJUSTMENT, self.clock(), payload)
        return self.book()

    # -- proposing -----------------------------------------------------------

    def propose(self, liquidate: bool = False) -> Proposal:
        """Decide on the latest complete bar and record the proposal. Sends nothing."""
        if not self.journal.is_open:
            raise StateIntegrityError("the sleeve is not open; run `ql live init` first")
        state = self.state()
        if not state.permits_proposals and not liquidate:
            raise ContractViolation(
                "the system is halted. Either resolve the cause and run `ql live clear`, "
                "or propose an orderly exit with `ql live propose --liquidate`."
            )
        last_reconciliation = self.journal.last(EventKind.RECONCILIATION)
        if last_reconciliation is None:
            raise ContractViolation("no reconciliation yet; run `ql live sync` first")
        if last_reconciliation.payload["status"] == "mismatch":
            raise ContractViolation(
                "the last reconciliation found a mismatch; resolve it before proposing"
            )
        now = self.clock()
        if now - last_reconciliation.at > timedelta(hours=24):
            raise ContractViolation(
                "the last reconciliation is more than a day old; run `ql live sync` first"
            )
        if self.rotation_orders_working():
            raise ContractViolation(
                "orders from the last approval are still working; wait for them to finish "
                "and run `ql live sync`"
            )

        market = self.market()
        interval = market.interval
        moment = market.schedule[-1]
        age = now - moment
        limit = self.config.monitoring.data_age_hours(interval)
        if age > timedelta(hours=limit):
            raise ContractViolation(
                f"the latest complete {interval.noun} closed {age.total_seconds() / 3600:.0f} "
                f"hours ago (limit {limit:.0f}); run `ql data refresh`"
            )
        if interval.is_intraday:
            missing = self.readiness(market)
            if missing:
                raise ContractViolation(
                    "this intraday strategy is not ready to trade: " + "; ".join(missing)
                )
            from data import calendar

            due = calendar.last_closed_bar(now, interval.duration)
            if due is not None and moment < due:
                raise ContractViolation(
                    f"the {interval.noun} that closed at {due:%Y-%m-%d %H:%M} UTC is not in the "
                    f"data yet (latest {moment:%Y-%m-%d %H:%M}); run `ql data refresh`"
                )

        book = self.book()
        # The decision is taken on the latest complete week, but the sleeve may
        # have been touched since (a fill, a snapshot's adjustment). The book is
        # the current book; it is considered at the decision moment, which is a
        # label for the data it is judged against, not a claim about its past.
        if book.as_of > moment:
            book = replace(book, as_of=moment)
        marks = dict(market.window.marks_at(moment))
        strategy = self.strategy()
        blocked = self._stopped_since_last_rotation() | {
            InstrumentId(t) for t in self.config.unmanaged
        }
        tradable = set(market.window.fresh_at(moment)) - blocked

        leverage, previous_leverage, _ = (1.0, None, None) if liquidate else self.leverage_now()
        if liquidate:
            decision_intents = self._liquidation_intents(book, moment)
            target: dict[str, float] = {}
            rotation = True
        else:
            decision = propose(
                run=self.run, book=book, strategy=strategy, moment=moment,
                filtration_at=market.filtration_at, marks_at=lambda _: marks,
                constraints_for=self.broker.constraints, tradable=tradable,
                policy=self.config.execution.sizing(
                    interval, allow_short=self.config.risk.allow_short,
                    leverage=leverage, previous_leverage=previous_leverage,
                ),
            )
            decision_intents = decision.intents
            target = {str(i): w for i, w in decision.target.weights.items()}
            rotation = bool(decision.target.diagnostics.get("rotated", 0.0))

        missing = [str(i) for i in book.positions if i not in marks]
        if missing:
            raise ContractViolation(f"no price data for held {missing}; refresh the data")
        equity = book.equity({i: marks[i] for i in book.positions})
        availability, fees = self._borrow_data(book, decision_intents)
        review = self.supervisor(state, availability, fees).review(
            decision_intents, _position_risk(book, marks, {}), equity, marks
        )
        orders = []
        for intent in review.approved:
            price = marks.get(intent.instrument, 0.0)
            value = intent.quantity * price
            share = value / equity if equity > 0 else 0.0
            # The sanity bound applies to new exposure, on either side: a large
            # exit is the book getting smaller, a large entry is a unit error.
            _, opening = split_legs(book.quantity(intent.instrument), intent.side, intent.quantity)
            opening_share = opening * price / equity if equity > 0 else 0.0
            if opening_share > self.config.risk.max_order_fraction:
                raise ContractViolation(
                    f"{intent.instrument}: an order opening {opening_share:.0%} of the sleeve "
                    f"exceeds max_order_fraction ({self.config.risk.max_order_fraction:.0%}). "
                    f"This is a sanity bound; something upstream is wrong."
                )
            orders.append(ProposedOrder(intent, price, value, share))
        self._check_margin(book, [o.intent for o in orders], marks)

        current = book.weights({i: marks[i] for i in book.positions}) if book.positions else {}
        fingerprint = book_fingerprint(book)
        # The count of earlier proposals is part of the code, so two proposals
        # made in the same second on the same book still get different codes --
        # otherwise approving "the old one" could silently approve the new one.
        ordinal = len(self.journal.events(EventKind.PROPOSAL)) + 1
        proposal_id = "P" + hashlib.sha256(
            f"{ordinal}|{moment.isoformat()}|{fingerprint}|"
            f"{[o.intent.client_order_id for o in orders]}|{now.isoformat()}".encode()
        ).hexdigest()[:6].upper()
        proposal = Proposal(
            proposal_id=proposal_id,
            created_at=now,
            expires_at=now + timedelta(hours=self.config.proposal_ttl_hours),
            decision_time=moment,
            data_as_of=moment,
            rotation=rotation,
            liquidation=liquidate,
            state=state,
            equity=equity,
            cash=book.cash,
            orders=tuple(orders),
            current_weights={str(i): w for i, w in current.items()},
            target_weights=target,
            findings=tuple(f.line() for f in review.findings),
            fingerprint=fingerprint,
            leverage=leverage,
        )
        self.journal.append(EventKind.PROPOSAL, now, {
            "proposal_id": proposal.proposal_id,
            "expires_at": proposal.expires_at.isoformat(),
            "decision_time": moment.isoformat(),
            "rotation": rotation,
            "liquidation": liquidate,
            "state": state.value,
            "equity": equity,
            "cash": book.cash,
            "fingerprint": fingerprint,
            "intents": [intent_to_dict(o.intent) for o in orders],
            "marks": {str(i): p for i, p in marks.items()},
            "current_weights": proposal.current_weights,
            "target_weights": target,
            "findings": list(proposal.findings),
            "leverage": leverage,
        })
        return proposal

    def _borrow_data(self, book: Book, intents: Sequence[OrderIntent]):
        """What the broker says can be borrowed, for orders that open a short.

        ``(None, None)`` when nothing in the decision is a short sale, so the
        common case never asks. A broker that cannot report availability gives
        an empty answer, which the short-sale rule treats as "not borrowable".
        """
        shorting = [
            i.instrument for i in intents
            if i.side is Side.SELL and split_legs(book.quantity(i.instrument), i.side, i.quantity)[1] > 0
        ]
        if not shorting:
            return None, None
        report = getattr(self.broker, "short_availability", None)
        if report is None:
            return {i: None for i in shorting}, {}
        found = report(shorting)
        return (
            {i: found.get(i).shares if found.get(i) else None for i in shorting},
            {i: found.get(i).fee_rate if found.get(i) else None for i in shorting},
        )

    def _check_margin(
        self, book: Book, intents: Sequence[OrderIntent],
        marks: Mapping[InstrumentId, float] | None = None,
    ) -> None:
        """Refuse a proposal that borrows more than the account can margin.

        Two things borrow: a short sale (stock) and a purchase the sleeve's cash
        does not cover (cash). Either way the orders that add exposure are
        priced by the broker with IBKR's what-if order, not computed here:
        margin rules differ by account type, instrument and house policy, and a
        local approximation that is wrong in the permissive direction is worse
        than none. A proposal that borrows nothing is not asked about.
        """
        opening = [
            i for i in intents
            if split_legs(book.quantity(i.instrument), i.side, i.quantity)[1] > 0
        ]
        shorts = [i for i in opening if i.side is Side.SELL]
        prices = marks or {}
        cash_after = book.cash - sum(
            i.side.sign * i.quantity * prices.get(i.instrument, 0.0) for i in intents
        )
        borrowing_cash = cash_after < 0 and any(i.side is Side.BUY for i in opening)
        needs = opening if borrowing_cash else shorts
        if not needs:
            return
        check = getattr(self.broker, "margin_check", None)
        if check is None:
            raise ContractViolation(
                "this broker cannot report margin, so orders that borrow (short sales, or "
                "buying beyond the sleeve's cash) cannot be checked; refused"
            )
        verdict = check(needs)
        if not verdict.ok:
            raise ContractViolation(f"margin check failed: {verdict.message}")

    def _last_leverage(self) -> float:
        """The leverage of the last approved proposal: what the book was built at."""
        approved = {e.payload["proposal_id"] for e in self.journal.events(EventKind.APPROVAL)}
        for event in reversed(self.journal.events(EventKind.PROPOSAL)):
            if event.payload["proposal_id"] in approved and "leverage" in event.payload:
                return float(event.payload["leverage"])
        return 1.0

    def leverage_now(self) -> tuple[float, float | None, float | None]:
        """The leverage the next decision is sized to, the previous one, and the cushion.

        The same rule the backtest applies (``risk.leverage.LeverageSchedule``),
        fed the sleeve's own history: equity per bar from snapshots, and the
        strategy's per-unit returns -- the baseline backtest's, then the live
        ones -- for the tail-risk estimate. The cushion is IBKR's.
        """
        rule = self.config.leverage.schedule(self.config.risk.max_gross)
        if rule.is_static_unlevered:
            return 1.0, None, None
        from contracts.risk import LeverageState
        from runtime.monitor import Baseline, live_returns

        previous = self._last_leverage()
        _, equity, returns = live_returns(self)
        base: list[float] = []
        if self.config.baseline_path.exists():
            try:
                base = list(Baseline.load(self.config.baseline_path).base_returns)
            except ContractViolation:
                base = []
        base += [r / max(previous, 1e-9) for r in returns]
        account = self.broker.account_snapshot()
        cushion = getattr(account, "cushion", None)
        level = rule(LeverageState(
            equity=equity or [self.config.sleeve_capital], base_returns=base,
            previous=previous, cushion=cushion, bars_per_week=self.interval.bars_per_week,
        ))
        return level, previous, cushion

    def _liquidation_intents(self, book: Book, moment: datetime) -> tuple[OrderIntent, ...]:
        """Close everything the sleeve holds at the next open: sell longs, cover shorts."""
        tif = self.config.execution.resolve(self.interval)
        intents = []
        for instrument, position in sorted(book.positions.items(), key=lambda kv: str(kv[0])):
            quantity = abs(position.quantity)
            side = Side.closing(position.quantity)
            intents.append(OrderIntent(
                client_order_id=client_order_id(
                    self.run, self.portfolio, instrument, moment, side, quantity
                ),
                run=self.run, portfolio=self.portfolio, instrument=instrument,
                strategy_version=self.strategy().version, side=side,
                quantity=quantity, order_type=OrderType.MARKET, decision_time=moment,
                time_in_force=tif, reason="liquidation",
            ))
        return tuple(intents)

    def rotation_orders_working(self) -> bool:
        last_status: dict[str, str] = {}
        for event in self.journal.events(EventKind.SUBMISSION, EventKind.ORDER_STATUS):
            last_status[event.payload["client_order_id"]] = event.payload.get("status", "submitted")
        return any(
            status not in ("filled", "cancelled", "rejected") and not is_stop_order(oid)
            for oid, status in last_status.items()
        )

    def _last_approved_rotation(self):
        approved = {e.payload["proposal_id"] for e in self.journal.events(EventKind.APPROVAL)}
        found = None
        for event in self.journal.events(EventKind.PROPOSAL):
            if event.payload["proposal_id"] in approved and event.payload.get("rotation"):
                found = event
        return found

    def _stopped_since_last_rotation(self) -> set[InstrumentId]:
        """Names a stop closed since the last approved rotation. Not re-bought yet."""
        last = self._last_approved_rotation()
        since = last.sequence if last else 0
        return {
            InstrumentId(e.payload["instrument"])
            for e in self.journal.events(EventKind.FILL)
            if e.payload.get("stop") and e.sequence > since
        }

    # -- approving -----------------------------------------------------------

    def confirmation_phrase(self, proposal_id: str) -> str:
        """What a person must type to approve. Longer in live mode, on purpose."""
        if self.config.mode is TradingMode.LIVE:
            return f"LIVE {proposal_id}"
        return proposal_id

    def pending_proposal(self):
        """The latest proposal, if it is still undecided.

        A proposal is decided by a rejection, by a person's approval, or once
        every one of its orders has been sent. An automatic approval that sent
        only the exits leaves it pending: its entries wait for a person.
        """
        latest = self.journal.last(EventKind.PROPOSAL)
        if latest is None:
            return None
        pid = latest.payload["proposal_id"]
        if any(e.payload["proposal_id"] == pid for e in self.journal.events(EventKind.REJECTION)):
            return None
        approvals = [
            e for e in self.journal.events(EventKind.APPROVAL) if e.payload["proposal_id"] == pid
        ]
        if any(not e.payload.get("partial") for e in approvals):
            return None
        sent = self.sent_orders(pid)
        if approvals and all(row["client_order_id"] in sent for row in latest.payload["intents"]):
            return None
        return latest

    def sent_orders(self, proposal_id: str) -> set[str]:
        """Order ids already sent for a proposal, by any approval."""
        return {
            oid
            for e in self.journal.events(EventKind.APPROVAL)
            if e.payload["proposal_id"] == proposal_id
            for oid in e.payload.get("orders", ())
        }

    def approvable(self, proposal_id: str):
        """The proposal's event, the book and the state, if it may still be sent.

        Refuses unless: the proposal is the latest and still undecided; it has
        not expired; no newer bar has arrived since it was decided; the sleeve
        is exactly the one it was computed against; and the system has not been
        halted since. Shared by a person's approval and an automatic one.
        """
        event = self.pending_proposal()
        if event is None or event.payload["proposal_id"] != proposal_id:
            raise ContractViolation(
                f"{proposal_id} is not the pending proposal. Only the latest undecided "
                f"proposal can be approved; run `ql live propose` for a fresh one."
            )
        now = self.clock()
        if now >= datetime.fromisoformat(event.payload["expires_at"]):
            raise ContractViolation(f"{proposal_id} has expired; propose again")
        # A proposal is a decision on one bar. Once a newer bar exists it is a
        # decision on stale data, however young the proposal is -- the check
        # that makes a fixed time-to-live safe for strategies of any frequency.
        decided = datetime.fromisoformat(event.payload["decision_time"])
        latest = self.market().schedule[-1]
        if latest > decided:
            raise ContractViolation(
                f"{proposal_id} was decided on the {self.interval.noun} closing "
                f"{decided.isoformat(timespec='minutes')}, and a newer one has closed since "
                f"({latest.isoformat(timespec='minutes')}); propose again"
            )
        book = self.book()
        if book_fingerprint(book) != event.payload["fingerprint"]:
            raise ContractViolation(
                f"the sleeve has changed since {proposal_id} was made (a fill, a stop or an "
                f"adjustment). Its quantities no longer apply; propose again."
            )
        state = self.state()
        liquidation = bool(event.payload.get("liquidation"))
        if not state.permits_proposals and not liquidation:
            raise ContractViolation("the system was halted after this proposal was made")
        return event, book, state

    def approve(self, proposal_id: str, typed: str) -> list[tuple[OrderIntent, Any]]:
        """Send an approved proposal's orders -- those not already sent.

        The typed text must be exactly the confirmation phrase; then every check
        in :meth:`approvable` applies.
        """
        expected = self.confirmation_phrase(proposal_id)
        if typed.strip() != expected:
            raise ContractViolation(
                f"confirmation did not match; type exactly: {expected}"
            )
        event, book, state = self.approvable(proposal_id)
        sent = self.sent_orders(proposal_id)
        intents = [
            intent_from_dict(row) for row in event.payload["intents"]
            if row["client_order_id"] not in sent
        ]
        if not state.permits_entries:
            # Degraded after the proposal was made: keep only what closes
            # exposure, on either side of zero.
            intents = [
                kept for kept in (
                    closing_part(i, book.quantity(i.instrument)) for i in intents
                ) if kept is not None
            ]
        return self.send(event, intents, confirmation="typed", by="person", partial=False)

    def send(
        self, event, intents: Sequence[OrderIntent], confirmation: str, by: str, partial: bool
    ) -> list[tuple[OrderIntent, Any]]:
        """Record the approval, cancel stops on the same side, submit. No checks here.

        Called only after :meth:`approvable` has passed, by :meth:`approve` or
        by automation (``runtime.automation``). ``partial`` marks an approval
        that leaves orders for a person.
        """
        proposal_id = event.payload["proposal_id"]
        now = self.clock()
        self.journal.append(EventKind.APPROVAL, now, {
            "proposal_id": proposal_id,
            "orders": [i.client_order_id for i in intents],
            "confirmation": confirmation,
            "by": by,
            "partial": partial,
        })

        # Cancel-replace: a resting stop and an approved order on the same side
        # are two orders to close the same shares. Both reaching the market
        # would push the position through zero -- a long into a short, or the
        # reverse -- without anyone deciding to.
        trading = {(str(i.instrument), i.side) for i in intents}
        for working in self.broker.working_orders():
            if (
                working.client_order_id.startswith(self.order_prefix)
                and is_stop_order(working.client_order_id)
                and (str(working.instrument), working.side) in trading
            ):
                self.broker.cancel(working.client_order_id)
                self.journal.append(EventKind.STOP_CANCELLED, now, {
                    "client_order_id": working.client_order_id,
                    "instrument": str(working.instrument),
                    "reason": f"replaced by an approved {working.side.value}",
                })

        sent = []
        for intent in intents:  # exits first; the engine already ordered them
            state_reported = self.broker.submit(intent)
            self.journal.append(EventKind.SUBMISSION, self.clock(), {
                **intent_to_dict(intent),
                "proposal_id": proposal_id,
                "status": state_reported.status.value,
                "broker_order_id": state_reported.broker_order_id,
                "message": state_reported.message,
            })
            sent.append((intent, state_reported))
        return sent

    def reject(self, proposal_id: str, reason: str) -> None:
        """Decline a proposal. Recorded as an override, which monitoring prices."""
        if len(reason.strip()) < 10:
            raise ContractViolation(
                "a rejection needs a real reason (at least ten characters). Overriding the "
                "system is legitimate; doing it without a record is not."
            )
        event = self.pending_proposal()
        if event is None or event.payload["proposal_id"] != proposal_id:
            raise ContractViolation(f"{proposal_id} is not the pending proposal")
        self.journal.append(EventKind.REJECTION, self.clock(), {
            "proposal_id": proposal_id, "reason": reason.strip(),
        })

    # -- protective stops ----------------------------------------------------

    def place_stops(self) -> tuple[int, int]:
        """Make the stops resting at the broker match the sleeve.

        A stop is re-anchored only when a rotation has finished: at the average
        fill price for names the rotation traded, and at the decision close for
        names it held unchanged. Between rotations the anchor stays put and only
        the quantity follows the position. A long gets a sell stop below its
        anchor, a short a buy stop above it.
        """
        rule = self.stop_rule()
        now = self.clock()
        book = self.book()
        ours = [
            w for w in self.broker.working_orders()
            if w.client_order_id.startswith(self.order_prefix) and is_stop_order(w.client_order_id)
        ]
        if rule is None:
            for w in ours:
                self.broker.cancel(w.client_order_id)
                self.journal.append(EventKind.STOP_CANCELLED, now, {
                    "client_order_id": w.client_order_id, "instrument": str(w.instrument),
                    "reason": "stops disabled in the config",
                })
            return 0, len(ours)
        if self.rotation_orders_working():
            return 0, 0  # wait until the rotation has filled before re-anchoring

        anchors = self._current_anchors()
        rotation = self._last_approved_rotation()
        rotation_id = rotation.payload["proposal_id"] if rotation else "opening"
        rotation_fills = self._rotation_fill_prices(rotation)
        rotation_marks = rotation.payload.get("marks", {}) if rotation else {}
        opening = self.journal.last(EventKind.OPENED)
        opening_marks = {
            p["instrument"]: p.get("mark") for p in opening.payload.get("positions", ())
        } if opening else {}

        # quantity, level, rotation, decision time, anchor
        desired: dict[str, tuple[float, float, str, datetime, float]] = {}
        for instrument, position in book.positions.items():
            ticker = str(instrument)
            anchor_rotation = anchors.get(ticker, {}).get("rotation")
            if anchor_rotation == rotation_id and ticker in anchors:
                anchor = anchors[ticker]["anchor"]
            elif ticker in rotation_fills:
                anchor = rotation_fills[ticker]
            elif ticker in rotation_marks:
                anchor = float(rotation_marks[ticker])
            elif ticker in anchors:
                anchor = anchors[ticker]["anchor"]
            else:
                anchor = float(opening_marks.get(ticker) or position.average_cost)
            level = self.broker.constraints(instrument).round_price(
                rule.level(anchor, position.quantity)
            )
            decision = (
                datetime.fromisoformat(rotation.payload["decision_time"]) if rotation
                else opening.at
            )
            desired[ticker] = (position.quantity, level, rotation_id, decision, anchor)

        placed = cancelled = 0
        for w in ours:
            want = desired.get(str(w.instrument))
            if (
                want is None
                or abs(w.quantity - abs(want[0])) > 1e-9
                or w.side is not Side.closing(want[0])
                or w.stop_price is None
                or abs(w.stop_price - want[1]) > 1e-6
            ):
                self.broker.cancel(w.client_order_id)
                self.journal.append(EventKind.STOP_CANCELLED, now, {
                    "client_order_id": w.client_order_id, "instrument": str(w.instrument),
                    "reason": "position or anchor changed" if want else "position closed",
                })
                cancelled += 1
            else:
                desired.pop(str(w.instrument))
        # A replacement stop is the same decision as the one it replaces, so
        # it needs a fresh id or the broker hands back the cancelled order
        # (``stop_order_id``). Numbered from the journal, which keeps the id
        # deterministic; ids just cancelled are skipped too, in case one was
        # placed but never journaled.
        placements: dict[tuple[str, str | None], int] = {}
        for event in self.journal.events(EventKind.STOP_PLACED):
            key = (event.payload["instrument"], event.payload.get("rotation"))
            placements[key] = placements.get(key, 0) + 1
        taken = {w.client_order_id for w in ours}
        version = self.strategy().version
        for ticker, (position, level, rotation_key, decision, anchor) in sorted(desired.items()):
            instrument = InstrumentId(ticker)
            side = Side.closing(position)
            quantity = abs(position)
            placement = placements.get((ticker, rotation_key), 0)
            order_id = stop_order_id(
                self.run, self.portfolio, instrument, decision, side, quantity, placement
            )
            while order_id in taken:
                placement += 1
                order_id = stop_order_id(
                    self.run, self.portfolio, instrument, decision, side, quantity, placement
                )
            intent = OrderIntent(
                client_order_id=order_id,
                run=self.run, portfolio=self.portfolio, instrument=instrument,
                strategy_version=version, side=side,
                quantity=quantity, order_type=OrderType.STOP, decision_time=decision,
                stop_price=level, time_in_force=TimeInForce.GTC, reason="protective stop",
            )
            state = self.broker.submit(intent)
            self.journal.append(EventKind.STOP_PLACED, now, {
                "client_order_id": intent.client_order_id, "instrument": ticker,
                "side": side.value, "quantity": quantity, "anchor": anchor, "level": level,
                "rotation": rotation_key, "status": state.status.value,
            })
            placed += 1
        return placed, cancelled

    def _current_anchors(self) -> dict[str, dict[str, Any]]:
        anchors: dict[str, dict[str, Any]] = {}
        for event in self.journal.events(EventKind.STOP_PLACED):
            anchors[event.payload["instrument"]] = {
                "anchor": float(event.payload["anchor"]),
                "rotation": event.payload.get("rotation"),
            }
        return anchors

    def _rotation_fill_prices(self, rotation) -> dict[str, float]:
        """Average entry fill price per instrument for the rotation's orders.

        Entry means the side the position now points: buys for a long, sales
        for a short. That is the price the new exposure was taken at, and so
        the one a stop is measured from.
        """
        if rotation is None:
            return {}
        approval = next(
            (e for e in self.journal.events(EventKind.APPROVAL)
             if e.payload["proposal_id"] == rotation.payload["proposal_id"]),
            None,
        )
        if approval is None:
            return {}
        order_ids = set(approval.payload.get("orders", ()))
        book = self.book()
        entry_side = {
            str(i): ("buy" if p.quantity > 0 else "sell") for i, p in book.positions.items()
        }
        totals: dict[str, tuple[float, float]] = {}
        for event in self.journal.events(EventKind.FILL):
            p = event.payload
            if p["client_order_id"] in order_ids and p["side"] == entry_side.get(p["instrument"]):
                q, v = totals.get(p["instrument"], (0.0, 0.0))
                totals[p["instrument"]] = (q + p["quantity"], v + p["quantity"] * p["price"])
        return {t: v / q for t, (q, v) in totals.items() if q > 0}

    # -- status --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Everything a person needs to see before deciding what to do next."""
        book = self.book()
        snapshot = self.journal.last(EventKind.SNAPSHOT)
        reconciliation = self.journal.last(EventKind.RECONCILIATION)
        pending = self.pending_proposal()
        now = self.clock()
        marks = {}
        if snapshot:
            marks = {p["instrument"]: p["mark"] for p in snapshot.payload["positions"]}
        equity = book.cash + sum(
            p.quantity * marks.get(str(i), p.average_cost) for i, p in book.positions.items()
        )
        if self.broker is not None:
            stop_by_name = {
                str(w.instrument): w.stop_price for w in self.broker.working_orders()
                if is_stop_order(w.client_order_id)
            }
            stop_source = "broker"
        else:
            stop_by_name = self._journal_stops()
            stop_source = "journal"
        return {
            "mode": self.config.mode.value,
            "account": self.config.account,
            "state": self.state().value,
            "automation": self.automation_scope(),
            "sleeve_equity": equity,
            "cash": book.cash,
            "positions": [
                {
                    "instrument": str(i), "quantity": p.quantity,
                    "average_cost": p.average_cost, "mark": marks.get(str(i)),
                    "stop": stop_by_name.get(str(i)),
                }
                for i, p in sorted(book.positions.items(), key=lambda kv: str(kv[0]))
            ],
            "stop_source": stop_source,
            "pending_proposal": pending.payload["proposal_id"] if pending else None,
            "last_snapshot_hours": (now - snapshot.at).total_seconds() / 3600 if snapshot else None,
            "last_reconciliation": reconciliation.payload["status"] if reconciliation else None,
            "last_reconciliation_hours": (
                (now - reconciliation.at).total_seconds() / 3600 if reconciliation else None
            ),
        }

    def _journal_stops(self) -> dict[str, float]:
        """Stops the journal says are working: placed, and not since cancelled or filled.

        What the sleeve *believes*; the broker is the authority, and ``sync``
        reconciles the two. Used when no gateway is connected.
        """
        live: dict[str, tuple[str, float]] = {}
        for event in self.journal.events(
            EventKind.STOP_PLACED, EventKind.STOP_CANCELLED, EventKind.FILL
        ):
            p = event.payload
            if event.kind is EventKind.STOP_PLACED and p.get("status") not in ("rejected", "cancelled"):
                live[p["instrument"]] = (p["client_order_id"], float(p["level"]))
            elif event.kind is EventKind.STOP_CANCELLED:
                if live.get(p["instrument"], ("",))[0] == p["client_order_id"]:
                    live.pop(p["instrument"])
            elif event.kind is EventKind.FILL and is_stop_order(p.get("client_order_id", "")):
                live.pop(p.get("instrument", ""), None)
        return {name: level for name, (_, level) in live.items()}


def closing_part(intent: OrderIntent, held: float) -> OrderIntent | None:
    """The part of an order that reduces exposure, or ``None`` if it only adds."""
    closing, opening = split_legs(held, intent.side, intent.quantity)
    if closing <= 0:
        return None
    return intent if opening == 0 else replace(intent, quantity=float(closing))


def _universe_record(universe) -> dict | None:
    if universe is None:
        return None
    return {"name": universe.name, "symbols": list(universe.symbols),
            "fingerprint": universe.fingerprint}


def _universe_label(record) -> str:
    if not record:
        return "every instrument in the store"
    return f"{record['name']} ({len(record['symbols'])} symbols, {record['fingerprint']})"
