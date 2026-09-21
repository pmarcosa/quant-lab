"""The vocabulary of a live system: modes, degradation states, journal events.

Three ideas, each chosen because the alternative fails silently.

**Paper and live are different accounts, not a flag.** IBKR paper accounts are
prefixed ``DU`` and live ones ``U``. :meth:`TradingMode.admits` checks the
account the gateway reports against the mode the configuration declares, so a
configuration pointed at the wrong port fails at connection time instead of
placing a "test" order with real money.

**Degradation is a ladder, and every rung only removes permissions.** Normal
operation proposes rotations. Reduce-only permits exits and protective stops but
no new buys. Halted permits nothing but protective stops and an orderly exit,
until a person clears it with a written reason.

Who may move it, and which way, is the safety property. Monitoring may put the
system into reduce-only and lift it again when the evidence clears — that rung
exists precisely to be used on uncertain evidence, and it cannot make anything
worse. **Nothing automatic ever lifts a halt**, and a reduce-only a person set is
theirs to lift. A system that cannot move itself off the bottom rung cannot talk
itself back into risk.

**The journal is an event log, not a status table.** Every proposal, approval,
order, fill, correction and snapshot is appended with a sequence number and
never rewritten. The strategy's book is a pure replay of those events, so what
the system believes can always be explained by showing how it came to believe
it — and compared against what the broker says.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from contracts.errors import ContractViolation
from contracts.temporal import utc


class TradingMode(str, Enum):
    """Which kind of account the configuration intends to trade."""

    PAPER = "paper"
    LIVE = "live"

    def admits(self, account: str) -> bool:
        """Whether an IBKR account id belongs to this mode.

        IBKR prefixes paper accounts with ``DU``; live individual accounts start
        with ``U``. Anything else is refused rather than guessed at.
        """
        account = account.strip().upper()
        if self is TradingMode.PAPER:
            return account.startswith("DU")
        return account.startswith("U") and not account.startswith("DU")


class DegradationState(str, Enum):
    """How much the live system is currently allowed to do.

    Ordered from most to least permissive. Automatic transitions may only move
    down; moving up requires a person and a reason in the journal.
    """

    NORMAL = "normal"
    REDUCE_ONLY = "reduce_only"
    HALTED = "halted"

    @property
    def rank(self) -> int:
        return {"normal": 0, "reduce_only": 1, "halted": 2}[self.value]

    @property
    def permits_buys(self) -> bool:
        return self is DegradationState.NORMAL

    @property
    def permits_proposals(self) -> bool:
        return self is not DegradationState.HALTED

    def worst(self, other: DegradationState) -> DegradationState:
        """The more restrictive of the two."""
        return self if self.rank >= other.rank else other


class EventKind(str, Enum):
    """Everything that can happen to a live sleeve."""

    OPENED = "opened"                  # opening balance: sleeve cash and adopted positions
    PROPOSAL = "proposal"              # what the system proposed, and on what basis
    APPROVAL = "approval"              # a person approved a proposal, by typed confirmation
    REJECTION = "rejection"            # a person declined one, with a reason
    SUBMISSION = "submission"          # an order reached the broker
    ORDER_STATUS = "order_status"      # what the broker reported about it
    FILL = "fill"                      # shares changed hands; the only thing that moves the book
    STOP_PLACED = "stop_placed"
    STOP_CANCELLED = "stop_cancelled"
    SNAPSHOT = "snapshot"              # sleeve equity and broker net liquidation at a moment
    RECONCILIATION = "reconciliation"  # journal against broker, and what differed
    ADJUSTMENT = "adjustment"          # a recorded correction, always with a reason
    STATE_CHANGE = "state_change"      # a move on the degradation ladder
    NOTE = "note"                      # free text for the human record


@dataclass(frozen=True, slots=True)
class JournalEvent:
    """One entry in the live journal.

    Attributes:
        kind: What happened.
        at: When it happened, in UTC.
        payload: The details, as plain JSON-compatible values.
        sequence: Position in the journal, assigned on append. Strictly
            increasing, so ordering never depends on clocks agreeing.
    """

    kind: EventKind
    at: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", utc(self.at))
        object.__setattr__(self, "payload", dict(self.payload))
        if not isinstance(self.kind, EventKind):
            raise ContractViolation(f"unknown journal event kind {self.kind!r}")
        if self.sequence < 0:
            raise ContractViolation(f"sequence cannot be negative; got {self.sequence}")
