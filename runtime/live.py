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
confirmation that no code path supplies. That is the propose-and-approve
discretion mode; an automatic mode would be a different function, not a flag on
this one.

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
)
from contracts.identifiers import InstrumentId, PortfolioId, RunId, TenantId
from contracts.live import DegradationState, EventKind, TradingMode
from contracts.temporal import BarInterval
from engine.accounting import Book
from engine.decide import SizingPolicy
from engine.run import _position_risk, propose
from risk.rules import GrossExposureLimit, ProtectiveStop, ReduceOnly, RiskSupervisor
from runtime.config import LiveConfig
from runtime.journal import (
    Journal,
    book_fingerprint,
    fill_to_dict,
    intent_from_dict,
    intent_to_dict,
    sleeve_book,
)
from runtime.wiring import Market, load_market
from strategies.momentum import MomentumParams, WeeklyMomentum

TENANT = TenantId("user")


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

    # -- identity and wiring -------------------------------------------------

    @property
    def portfolio(self) -> PortfolioId:
        return PortfolioId(TENANT, f"{self.config.mode.value}-sleeve")

    @property
    def run(self) -> RunId:
        return RunId(f"{self.config.mode.value}-live")

    def strategy(self) -> WeeklyMomentum:
        s = self.config.strategy
        return WeeklyMomentum(
            MomentumParams(
                rebalance_weeks=s.rebalance_weeks,
                top_n=s.top_n,
                lookback_weeks=s.lookback_weeks,
            )
        )

    def market(self) -> Market:
        if self.market_loader is not None:
            return self.market_loader(self.store_root)
        return load_market(self.store_root, interval=BarInterval.WEEK)

    def stop_rule(self) -> ProtectiveStop | None:
        d = self.config.risk.stop_distance
        return ProtectiveStop(distance=d) if d > 0 else None

    def supervisor(self, state: DegradationState) -> RiskSupervisor:
        rules: list = [GrossExposureLimit(maximum=self.config.risk.max_gross)]
        if not state.permits_buys:
            rules.append(ReduceOnly(reason=f"the system is {state.value.replace('_', '-')}"))
        return RiskSupervisor(rules=tuple(rules), stop=self.stop_rule())

    def book(self) -> Book:
        return sleeve_book(self.journal, self.portfolio)

    # -- the degradation ladder ---------------------------------------------

    def state(self) -> DegradationState:
        last = self.journal.last(EventKind.STATE_CHANGE)
        return DegradationState(last.payload["to"]) if last else DegradationState.NORMAL

    def degrade(self, to: DegradationState, reason: str) -> DegradationState:
        """Move down the ladder. Never up: see :meth:`clear`."""
        current = self.state()
        target = current.worst(to)
        if target is not current:
            self.journal.append(
                EventKind.STATE_CHANGE, self.clock(),
                {"from": current.value, "to": target.value, "reason": reason, "by": "system"},
            )
        return target

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
        if current is DegradationState.REDUCE_ONLY and set_by == "person":
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
            payload["stop"] = found.fill.client_order_id.endswith("-stop")
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
        """Mark the sleeve at the latest closes and record it."""
        market = self.market()
        latest = market.schedule[-1]
        marks = market.window.marks_at(latest)
        book = self.book()
        missing = [str(i) for i in book.positions if i not in marks]
        if missing:
            raise ContractViolation(f"no price data to mark {missing}; refresh the data")
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
        }
        self.journal.append(EventKind.SNAPSHOT, self.clock(), payload)
        return payload

    # -- reconciliation ------------------------------------------------------

    def reconcile(self, record: bool = True) -> Reconciliation:
        """The journal's sleeve against what the broker holds.

        The sleeve is a sub-account, so the broker may hold *more* than the
        sleeve — other positions, extra shares — and that is a warning, not an
        error. The broker holding *less* than the sleeve believes is a mismatch:
        the system would be sizing and stopping positions that are not there.
        """
        book = self.book()
        findings: list[Finding] = []
        broker_positions = {
            str(p.instrument): p.quantity for p in self.broker.positions(self.portfolio)
        }
        sleeve = {str(i): p.quantity for i, p in book.positions.items()}

        for ticker, quantity in sorted(sleeve.items()):
            held = broker_positions.get(ticker, 0.0)
            if held + 1e-9 < quantity:
                findings.append(Finding(
                    "mismatch",
                    f"the sleeve holds {quantity:g} but the broker holds {held:g}",
                    ticker,
                ))
            elif held > quantity + 1e-9:
                findings.append(Finding(
                    "warn",
                    f"the broker holds {held:g}, {held - quantity:g} more than the sleeve; "
                    f"the extra shares are outside the strategy",
                    ticker,
                ))
        for ticker, quantity in sorted(broker_positions.items()):
            if ticker in sleeve or ticker in self.config.unmanaged or not quantity:
                continue
            findings.append(Finding(
                "warn",
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

        # Every sleeve position should have a resting stop.
        if self.stop_rule() is not None:
            protected = {
                str(w.instrument) for w in self.broker.working_orders()
                if w.client_order_id.endswith("-stop") and w.side is Side.SELL
            }
            for ticker in sorted(sleeve):
                if ticker not in protected:
                    findings.append(Finding("warn", "no protective stop resting at the broker", ticker))

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
            if quantity is None or quantity < 0:
                raise ContractViolation("an adjustment to a position needs a quantity of 0 or more")
            payload["instrument"] = instrument.upper()
            payload["quantity"] = float(quantity)
            if average_cost is not None:
                payload["average_cost"] = float(average_cost)
        self.journal.append(EventKind.ADJUSTMENT, self.clock(), payload)
        return self.book()

    # -- proposing -----------------------------------------------------------

    def propose(self, liquidate: bool = False) -> Proposal:
        """Decide on the latest complete week and record the proposal. Sends nothing."""
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
        if self._rotation_orders_working():
            raise ContractViolation(
                "orders from the last approval are still working; wait for them to finish "
                "and run `ql live sync`"
            )

        market = self.market()
        moment = market.schedule[-1]
        age = now - moment
        if age > timedelta(days=self.config.monitoring.max_data_age_days):
            raise ContractViolation(
                f"the latest complete week is {age.days} days old; run `ql data refresh`"
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

        if liquidate:
            decision_intents = self._liquidation_intents(book, moment)
            target: dict[str, float] = {}
            rotation = True
        else:
            decision = propose(
                run=self.run, book=book, strategy=strategy, moment=moment,
                filtration_at=market.filtration_at, marks_at=lambda _: marks,
                constraints_for=self.broker.constraints, tradable=tradable,
                policy=SizingPolicy(time_in_force=TimeInForce.OPG),
            )
            decision_intents = decision.intents
            target = {str(i): w for i, w in decision.target.weights.items()}
            rotation = bool(decision.target.diagnostics.get("rotated", 0.0))

        missing = [str(i) for i in book.positions if i not in marks]
        if missing:
            raise ContractViolation(f"no price data for held {missing}; refresh the data")
        equity = book.equity({i: marks[i] for i in book.positions})
        review = self.supervisor(state).review(
            decision_intents, _position_risk(book, marks, {}), equity
        )
        orders = []
        for intent in review.approved:
            price = marks.get(intent.instrument, 0.0)
            value = intent.quantity * price
            share = value / equity if equity > 0 else 0.0
            if intent.side is Side.BUY and share > self.config.risk.max_order_fraction:
                raise ContractViolation(
                    f"{intent.instrument}: an order worth {share:.0%} of the sleeve exceeds "
                    f"max_order_fraction ({self.config.risk.max_order_fraction:.0%}). This is a "
                    f"sanity bound; something upstream is wrong."
                )
            orders.append(ProposedOrder(intent, price, value, share))

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
        })
        return proposal

    def _liquidation_intents(self, book: Book, moment: datetime) -> tuple[OrderIntent, ...]:
        """Sell everything the sleeve holds at the next open."""
        intents = []
        for instrument, position in sorted(book.positions.items(), key=lambda kv: str(kv[0])):
            quantity = abs(position.quantity)
            intents.append(OrderIntent(
                client_order_id=client_order_id(
                    self.run, self.portfolio, instrument, moment, Side.SELL, quantity
                ),
                run=self.run, portfolio=self.portfolio, instrument=instrument,
                strategy_version=self.strategy().version, side=Side.SELL,
                quantity=quantity, order_type=OrderType.MARKET, decision_time=moment,
                time_in_force=TimeInForce.OPG, reason="liquidation",
            ))
        return tuple(intents)

    def _rotation_orders_working(self) -> bool:
        last_status: dict[str, str] = {}
        for event in self.journal.events(EventKind.SUBMISSION, EventKind.ORDER_STATUS):
            last_status[event.payload["client_order_id"]] = event.payload.get("status", "submitted")
        return any(
            status not in ("filled", "cancelled", "rejected") and not oid.endswith("-stop")
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
        """The latest proposal, if it has not been approved or rejected."""
        decided = {
            e.payload["proposal_id"]
            for e in self.journal.events(EventKind.APPROVAL, EventKind.REJECTION)
        }
        latest = self.journal.last(EventKind.PROPOSAL)
        if latest is None or latest.payload["proposal_id"] in decided:
            return None
        return latest

    def approve(self, proposal_id: str, typed: str) -> list[tuple[OrderIntent, Any]]:
        """Send an approved proposal's orders. The only path to the market.

        Refuses unless: the typed text is exactly the confirmation phrase; the
        proposal is the latest and still undecided; it has not expired; the
        sleeve is exactly the one it was computed against; and the system has
        not been halted since.
        """
        expected = self.confirmation_phrase(proposal_id)
        if typed.strip() != expected:
            raise ContractViolation(
                f"confirmation did not match; type exactly: {expected}"
            )
        event = self.pending_proposal()
        if event is None or event.payload["proposal_id"] != proposal_id:
            raise ContractViolation(
                f"{proposal_id} is not the pending proposal. Only the latest undecided "
                f"proposal can be approved; run `ql live propose` for a fresh one."
            )
        now = self.clock()
        if now >= datetime.fromisoformat(event.payload["expires_at"]):
            raise ContractViolation(f"{proposal_id} has expired; propose again")
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

        intents = [intent_from_dict(row) for row in event.payload["intents"]]
        if not state.permits_buys:
            intents = [i for i in intents if i.side is Side.SELL]

        self.journal.append(EventKind.APPROVAL, now, {
            "proposal_id": proposal_id,
            "orders": [i.client_order_id for i in intents],
            "confirmation": "typed",
        })

        # Cancel-replace: a resting stop and this approval's sell are two orders
        # for the same shares. Both reaching the market would take the book short.
        selling = {str(i.instrument) for i in intents if i.side is Side.SELL}
        for working in self.broker.working_orders():
            if working.client_order_id.endswith("-stop") and str(working.instrument) in selling:
                self.broker.cancel(working.client_order_id)
                self.journal.append(EventKind.STOP_CANCELLED, now, {
                    "client_order_id": working.client_order_id,
                    "instrument": str(working.instrument),
                    "reason": "replaced by an approved sell",
                })

        sent = []
        for intent in intents:  # sells first; the engine already ordered them
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
        the quantity follows the position.
        """
        rule = self.stop_rule()
        now = self.clock()
        book = self.book()
        ours = [w for w in self.broker.working_orders() if w.client_order_id.endswith("-stop")]
        if rule is None:
            for w in ours:
                self.broker.cancel(w.client_order_id)
                self.journal.append(EventKind.STOP_CANCELLED, now, {
                    "client_order_id": w.client_order_id, "instrument": str(w.instrument),
                    "reason": "stops disabled in the config",
                })
            return 0, len(ours)
        if self._rotation_orders_working():
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

        desired: dict[str, tuple[float, float, str, datetime]] = {}
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
            level = self.broker.constraints(instrument).round_price(rule.level(anchor))
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
                or abs(w.quantity - want[0]) > 1e-9
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
        for ticker, (quantity, level, rotation_key, decision, anchor) in sorted(desired.items()):
            instrument = InstrumentId(ticker)
            intent = OrderIntent(
                client_order_id=client_order_id(
                    self.run, self.portfolio, instrument, decision, Side.SELL, quantity
                ) + "-stop",
                run=self.run, portfolio=self.portfolio, instrument=instrument,
                strategy_version=self.strategy().version, side=Side.SELL,
                quantity=quantity, order_type=OrderType.STOP, decision_time=decision,
                stop_price=level, time_in_force=TimeInForce.GTC, reason="protective stop",
            )
            state = self.broker.submit(intent)
            self.journal.append(EventKind.STOP_PLACED, now, {
                "client_order_id": intent.client_order_id, "instrument": ticker,
                "quantity": quantity, "anchor": anchor, "level": level,
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
        """Average buy fill price per instrument for the rotation's orders."""
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
        totals: dict[str, tuple[float, float]] = {}
        for event in self.journal.events(EventKind.FILL):
            p = event.payload
            if p["client_order_id"] in order_ids and p["side"] == "buy":
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
                if w.client_order_id.endswith("-stop")
            }
            stop_source = "broker"
        else:
            stop_by_name = self._journal_stops()
            stop_source = "journal"
        return {
            "mode": self.config.mode.value,
            "account": self.config.account,
            "state": self.state().value,
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
            elif event.kind is EventKind.FILL and p.get("client_order_id", "").endswith("-stop"):
                live.pop(p.get("instrument", ""), None)
        return {name: level for name, (_, level) in live.items()}
