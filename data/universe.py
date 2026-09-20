"""Reconstructing the tradable list as it stood on a past date.

Selecting from today's list of names is the most expensive mistake a backtest can
make, and the hardest to notice: every name on that list survived to today by
construction, so the simulation never holds the ones that were delisted, acquired
or went to zero, and the result looks like skill.

**What this fixes and what it does not.** The engine stops a backtest of 2011
from holding an instrument that listed in 2019 — a real and common error it
removes completely. It cannot invent the names that were eligible in 2011 and are
absent from the data, because a universe reconstructed from files you downloaded
in 2026 contains only instruments you thought to download in 2026. Fixing that
needs a vendor history of index constituents. Until then the honest statement is:
the *timing* bias is gone, the *selection* bias is not, and a report built on this
universe must say so.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from contracts.temporal import utc
from contracts.universe import Membership, UniverseSource

MEMBERSHIP_COLUMNS = ("instrument", "joined", "delisted", "reason", "derived")


@dataclass(frozen=True, slots=True)
class PointInTimeUniverse(UniverseSource):
    """Membership intervals, queried by date.

    Args:
        records: Membership windows, including ones that have ended.
    """

    records: tuple[Membership, ...]

    def members_at(self, moment: datetime) -> Sequence[InstrumentId]:
        """Instruments that were members at ``moment``, ordered for determinism."""
        point = utc(moment)
        return tuple(
            sorted((r.instrument for r in self.records if r.covers(point)), key=str)
        )

    def memberships(self) -> Sequence[Membership]:
        return self.records

    def survivors_only(self) -> tuple[InstrumentId, ...]:
        """Instruments that never left. Useful only to quantify the bias avoided."""
        return tuple(sorted((r.instrument for r in self.records if r.delisted is None), key=str))

    @classmethod
    def from_csv(cls, path: Path) -> PointInTimeUniverse:
        """Load membership records from a CSV."""
        if not path.exists():
            raise ContractViolation(f"no universe file at {path}")
        frame = pd.read_csv(path)
        missing = [c for c in ("instrument", "joined") if c not in frame.columns]
        if missing:
            raise ContractViolation(f"universe file is missing {missing}")
        records = []
        for row in frame.itertuples(index=False):
            delisted = getattr(row, "delisted", None)
            records.append(
                Membership(
                    instrument=InstrumentId(str(row.instrument)),
                    joined=pd.Timestamp(row.joined, tz="UTC").to_pydatetime(),
                    delisted=(
                        None
                        if delisted is None or pd.isna(delisted)
                        else pd.Timestamp(delisted, tz="UTC").to_pydatetime()
                    ),
                    reason=str(getattr(row, "reason", "") or ""),
                )
            )
        return cls(records=tuple(records))

    def to_csv(self, path: Path, derived: bool = False) -> None:
        """Write membership records, flagging whether they were inferred."""
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "instrument": str(r.instrument),
                    "joined": r.joined.isoformat(),
                    "delisted": r.delisted.isoformat() if r.delisted else "",
                    "reason": r.reason,
                    "derived": derived,
                }
                for r in sorted(self.records, key=lambda r: str(r.instrument))
            ],
            columns=list(MEMBERSHIP_COLUMNS),
        ).to_csv(path, index=False)


def derive_memberships(
    first_and_last: Iterable[tuple[InstrumentId, datetime, datetime]],
    still_trading_after: datetime,
) -> PointInTimeUniverse:
    """Infer membership windows from the span of data actually held.

    A stopgap, and labelled as one. ``joined`` becomes the first observation and
    ``delisted`` the last, unless the last is recent enough that the instrument is
    presumed still trading.

    This captures when an instrument *started* appearing, which is what stops a
    2011 backtest holding a 2019 listing. It cannot capture instruments that never
    entered the dataset at all.

    Args:
        first_and_last: Triples of instrument, first observation, last observation.
        still_trading_after: A last observation at or after this is treated as
            ongoing rather than as a delisting.
    """
    cutoff = utc(still_trading_after)
    records = tuple(
        Membership(
            instrument=instrument,
            joined=first,
            delisted=None if utc(last) >= cutoff else last,
            reason="" if utc(last) >= cutoff else "no further data",
        )
        for instrument, first, last in first_and_last
    )
    return PointInTimeUniverse(records=records)
