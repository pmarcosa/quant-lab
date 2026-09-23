"""The live journal: an append-only record from which the sleeve is rebuilt.

Everything the live system does is written here before it is acted on, or as
soon as it is known. The strategy's book is then a pure function of the
journal — :func:`sleeve_book` — which makes three things possible that a mutable
position table does not:

- **Recovery.** After a crash, replaying the journal gives exactly the book the
  system held, with no question about which writes landed.
- **Reconciliation.** The broker is compared against what the journal says the
  sleeve should hold, and a difference is a finding rather than something the
  next write quietly overwrites.
- **Explanation.** Any position can be traced to the fills that produced it and
  the approval that authorised them.

The file is JSON Lines, one event per line, flushed and fsynced on every write.
A truncated final line is an integrity error, not a skipped row: a journal that
cannot be read in full cannot be trusted to describe the book.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts.errors import ContractViolation, StateIntegrityError
from contracts.execution import (
    Charge,
    ChargeKind,
    Fill,
    OrderIntent,
    OrderType,
    PositionLedgerEntry,
    Side,
    TimeInForce,
)
from contracts.identifiers import (
    InstrumentId,
    PortfolioId,
    RunId,
    StrategyId,
    StrategyVersion,
    TenantId,
)
from contracts.live import EventKind, JournalEvent
from contracts.temporal import utc
from engine.accounting import Book

JOURNAL_FORMAT = 1


class Journal:
    """Append-only event log for one live sleeve."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._next = self._last_sequence() + 1

    def _last_sequence(self) -> int:
        last = 0
        for event in self:
            last = event.sequence
        return last

    def append(self, kind: EventKind, at: datetime, payload: Mapping[str, Any]) -> JournalEvent:
        """Write one event and return it with its sequence number."""
        event = JournalEvent(kind=kind, at=at, payload=payload, sequence=self._next)
        row = {
            "format": JOURNAL_FORMAT,
            "sequence": event.sequence,
            "kind": event.kind.value,
            "at": event.at.isoformat(),
            "payload": event.payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, separators=(",", ":"), sort_keys=True, default=_encode)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next += 1
        return event

    def __iter__(self) -> Iterator[JournalEvent]:
        if not self.path.exists():
            return
        previous = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StateIntegrityError(
                        f"{self.path}:{number} is not valid JSON. The journal cannot be "
                        f"trusted to describe the book while this line is unreadable."
                    ) from error
                if row.get("format") != JOURNAL_FORMAT:
                    raise StateIntegrityError(
                        f"{self.path}:{number} is format {row.get('format')!r}; "
                        f"this code reads {JOURNAL_FORMAT}"
                    )
                event = JournalEvent(
                    kind=EventKind(row["kind"]),
                    at=datetime.fromisoformat(row["at"]),
                    payload=row["payload"],
                    sequence=int(row["sequence"]),
                )
                if event.sequence <= previous:
                    raise StateIntegrityError(
                        f"{self.path}:{number} has sequence {event.sequence} after "
                        f"{previous}; the journal has been edited or merged"
                    )
                previous = event.sequence
                yield event

    def events(self, *kinds: EventKind) -> tuple[JournalEvent, ...]:
        """Every event, or only those of the given kinds, oldest first."""
        wanted = set(kinds)
        return tuple(e for e in self if not wanted or e.kind in wanted)

    def last(self, kind: EventKind) -> JournalEvent | None:
        found = None
        for event in self:
            if event.kind is kind:
                found = event
        return found

    @property
    def is_open(self) -> bool:
        """Whether an opening balance has been recorded."""
        return self.last(EventKind.OPENED) is not None


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc(value).isoformat()
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    raise TypeError(f"{type(value).__name__} is not journal-serialisable")


# -- codecs ------------------------------------------------------------------


def intent_to_dict(intent: OrderIntent) -> dict[str, Any]:
    """An order intent as plain values, reversible by :func:`intent_from_dict`."""
    return {
        "client_order_id": intent.client_order_id,
        "run": str(intent.run),
        "tenant": str(intent.portfolio.tenant),
        "portfolio": intent.portfolio.name,
        "instrument": str(intent.instrument),
        "strategy": str(intent.strategy_version.strategy),
        "params_hash": intent.strategy_version.params_hash,
        "code_version": intent.strategy_version.code_version,
        "side": intent.side.value,
        "quantity": intent.quantity,
        "order_type": intent.order_type.value,
        "decision_time": intent.decision_time.isoformat(),
        "limit_price": intent.limit_price,
        "stop_price": intent.stop_price,
        "time_in_force": intent.time_in_force.value,
        "reason": intent.reason,
    }


def intent_from_dict(row: Mapping[str, Any]) -> OrderIntent:
    return OrderIntent(
        client_order_id=row["client_order_id"],
        run=RunId(row["run"]),
        portfolio=PortfolioId(TenantId(row["tenant"]), row["portfolio"]),
        instrument=InstrumentId(row["instrument"]),
        strategy_version=StrategyVersion(
            strategy=StrategyId(row["strategy"]),
            params_hash=row["params_hash"],
            code_version=row["code_version"],
        ),
        side=Side(row["side"]),
        quantity=float(row["quantity"]),
        order_type=OrderType(row["order_type"]),
        decision_time=datetime.fromisoformat(row["decision_time"]),
        limit_price=row.get("limit_price"),
        stop_price=row.get("stop_price"),
        time_in_force=TimeInForce(row["time_in_force"]),
        reason=row.get("reason", ""),
    )


def fill_to_dict(fill: Fill) -> dict[str, Any]:
    return {
        "client_order_id": fill.client_order_id,
        "instrument": str(fill.instrument),
        "side": fill.side.value,
        "quantity": fill.quantity,
        "price": fill.price,
        "at": fill.at.isoformat(),
        "commission": fill.commission,
    }


def fill_from_dict(row: Mapping[str, Any]) -> Fill:
    return Fill(
        client_order_id=row["client_order_id"],
        instrument=InstrumentId(row["instrument"]),
        side=Side(row["side"]),
        quantity=float(row["quantity"]),
        price=float(row["price"]),
        at=datetime.fromisoformat(row["at"]),
        commission=float(row.get("commission", 0.0)),
    )


# -- the book, replayed ------------------------------------------------------


def sleeve_book(journal: Journal, portfolio: PortfolioId) -> Book:
    """The sleeve's cash and positions, rebuilt from the journal alone.

    Only four kinds of event move it: the opening balance, fills, recorded
    adjustments, and financing charges (interest on borrowed cash, borrow fees).
    Everything else — proposals, approvals, statuses — is context. That is what
    makes the book explainable: every share and every dollar can be traced to
    one of those four.

    Raises:
        StateIntegrityError: If there is no opening balance, or more than one.
    """
    openings = journal.events(EventKind.OPENED)
    if not openings:
        raise StateIntegrityError(
            "the journal has no opening balance; run `ql live init` first"
        )
    if len(openings) > 1:
        raise StateIntegrityError(
            f"the journal has {len(openings)} opening balances; a sleeve opens once"
        )

    book: Book | None = None
    for event in journal:
        if event.kind is EventKind.OPENED:
            book = _opening_book(event, portfolio)
        elif book is None:
            continue
        elif event.kind is EventKind.FILL:
            fill = fill_from_dict(event.payload)
            # A fill reported late can carry an execution time earlier than an
            # event journaled before it. The journal's order is authoritative,
            # so the fill is applied in that order; its quantity, price and
            # commission -- the economic content -- are untouched.
            if fill.at < book.as_of:
                fill = replace(fill, at=book.as_of)
            book = book.apply(fill)
        elif event.kind is EventKind.ADJUSTMENT:
            book = _adjusted(book, event)
        elif event.kind is EventKind.FINANCING:
            at = max(event.at, book.as_of)
            book = book.charge(Charge(
                at=at, amount=float(event.payload["amount"]),
                kind=ChargeKind(event.payload["kind"]), detail=event.payload.get("detail", ""),
            ))
    assert book is not None
    return book


def _opening_book(event: JournalEvent, portfolio: PortfolioId) -> Book:
    positions: dict[InstrumentId, PositionLedgerEntry] = {}
    for row in event.payload.get("positions", ()):
        instrument = InstrumentId(row["instrument"])
        quantity = float(row["quantity"])
        if quantity == 0:
            continue
        positions[instrument] = PositionLedgerEntry(
            portfolio=portfolio,
            instrument=instrument,
            quantity=quantity,
            average_cost=float(row["average_cost"]),
            as_of=event.at,
            from_orders=("opening",),
        )
    cash = float(event.payload["cash"])
    if cash < 0:
        raise ContractViolation(f"opening sleeve cash cannot be negative; got {cash}")
    return Book(portfolio=portfolio, cash=cash, as_of=event.at, positions=positions)


def _adjusted(book: Book, event: JournalEvent) -> Book:
    """Apply a recorded correction: set a position outright and move cash.

    Corrections set the absolute quantity rather than applying a delta, because
    the usual reason for one is "accept what the broker says", and a delta
    computed from a book that was already wrong compounds the error.
    """
    payload = event.payload
    if not payload.get("reason"):
        raise StateIntegrityError(f"adjustment #{event.sequence} has no reason")
    positions = dict(book.positions)
    if "instrument" in payload:
        instrument = InstrumentId(payload["instrument"])
        quantity = float(payload["quantity"])
        if quantity == 0:
            positions.pop(instrument, None)
        else:
            existing = positions.get(instrument)
            positions[instrument] = PositionLedgerEntry(
                portfolio=book.portfolio,
                instrument=instrument,
                quantity=quantity,
                average_cost=float(
                    payload.get("average_cost")
                    or (existing.average_cost if existing else 0.0)
                ),
                as_of=event.at,
                realised_pnl=existing.realised_pnl if existing else 0.0,
                from_orders=(*(existing.from_orders if existing else ()), f"adj-{event.sequence}"),
            )
    return replace(
        book,
        cash=book.cash + float(payload.get("cash_delta", 0.0)),
        positions=positions,
        as_of=max(book.as_of, event.at),
    )


def book_fingerprint(book: Book) -> str:
    """A short stable digest of what the sleeve holds.

    Stored with every proposal and checked again at approval. If the book
    changed in between — a stop fired, a fill arrived, someone traded by hand —
    the proposal's quantities were computed against a book that no longer
    exists, and approving it would be approving the wrong trades.
    """
    import hashlib

    material = "|".join(
        f"{i}:{p.quantity:.6f}" for i, p in sorted(book.positions.items(), key=lambda kv: str(kv[0]))
    )
    material += f"|cash:{round(book.cash, 2):.2f}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def pending_intents(journal: Journal) -> Sequence[str]:
    """Client order ids that were submitted and have not reached a final state."""
    submitted: dict[str, str] = {}
    for event in journal.events(EventKind.SUBMISSION, EventKind.ORDER_STATUS):
        oid = event.payload["client_order_id"]
        if event.kind is EventKind.SUBMISSION:
            submitted[oid] = "submitted"
        else:
            submitted[oid] = event.payload["status"]
    final = {"filled", "cancelled", "rejected"}
    return tuple(oid for oid, status in submitted.items() if status not in final)
