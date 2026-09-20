"""Point-in-time membership: which instruments existed and qualified, and when.

Selecting from today's list of names is the most expensive mistake a backtest can
make, because it is invisible. Every instrument in that list survived to today by
construction, so the backtest never buys the ones that were delisted, acquired or
went to zero, and the result looks like skill.

A membership record has an open interval. Asking the universe for a date
reconstructs the list as it stood, including the names that later disappeared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from contracts.temporal import utc


@dataclass(frozen=True, slots=True)
class Membership:
    """An instrument's window of eligibility.

    Attributes:
        instrument: The durable identifier.
        joined: First date the instrument was tradable in this universe.
        delisted: Last date it was tradable, or None if it still is. A name that
            disappeared keeps its record; that is the entire point.
        reason: Why it left — delisted, acquired, failed a liquidity screen.
    """

    instrument: InstrumentId
    joined: datetime
    delisted: datetime | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "joined", utc(self.joined))
        if self.delisted is not None:
            delisted = utc(self.delisted)
            if delisted < self.joined:
                raise ContractViolation(
                    f"{self.instrument} delisted {delisted.isoformat()} before it joined "
                    f"{self.joined.isoformat()}"
                )
            object.__setattr__(self, "delisted", delisted)

    def covers(self, moment: datetime) -> bool:
        """Whether the instrument was a member at ``moment``."""
        point = utc(moment)
        if point < self.joined:
            return False
        return self.delisted is None or point <= self.delisted


@runtime_checkable
class UniverseSource(Protocol):
    """Reconstructs the tradable list as it stood on any date."""

    def members_at(self, moment: datetime) -> Sequence[InstrumentId]:
        """Instruments that were members at ``moment``, in a stable order."""
        ...

    def memberships(self) -> Sequence[Membership]:
        """Every membership record, including those that have ended."""
        ...
