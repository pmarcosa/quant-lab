"""An append-only store where a query cannot return the future.

Two ideas, both structural rather than advisory.

**Append-only.** There is no update. A correction is a new row carrying a later
``available_time``. Nothing is ever overwritten, so the state of the world as it
appeared on any past date can always be reconstructed. The practical case here is
not accounting restatements: it is splits and dividends. IBKR adjusts history
backwards, so a backtest run today sees prices that nobody could have seen at the
time. Keeping both versions makes that visible instead of silent.

**The query demands a decision time.** :meth:`BitemporalStore.as_of` has no
optional end date and no "give me everything" mode. Every read is filtered by
``event_time <= decision_time`` and ``available_time <= decision_time``, so a
look-ahead requires deliberately passing a future timestamp rather than merely
forgetting to pass anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from contracts.temporal import utc

#: The two columns every record carries, ahead of its payload.
TIME_COLUMNS = ("event_time", "available_time")


@dataclass(frozen=True, slots=True)
class BitemporalStore:
    """Observations for one dataset, on disk, one file per instrument.

    CSV rather than a database because at this size the whole dataset is a few
    megabytes, and a format a human can open and a diff can show is worth more
    than query speed. The interface is what matters; the backing store can change
    without callers noticing.

    Args:
        root: Directory the dataset lives in.
        dataset: Name of the dataset, e.g. "bars_1week".
    """

    root: Path
    dataset: str

    @property
    def path(self) -> Path:
        """Directory holding this dataset's files."""
        return self.root / self.dataset

    def _file(self, instrument: InstrumentId) -> Path:
        return self.path / f"{instrument}.csv"

    def instruments(self) -> tuple[InstrumentId, ...]:
        """Every instrument with data in this dataset."""
        if not self.path.is_dir():
            return ()
        return tuple(InstrumentId(p.stem) for p in sorted(self.path.glob("*.csv")))

    def append(self, instrument: InstrumentId, records: pd.DataFrame) -> int:
        """Add observations. Never rewrites an existing row.

        Args:
            instrument: What the records describe.
            records: Must carry ``event_time`` and ``available_time`` columns,
                both timezone-aware, plus any payload columns.

        Returns:
            How many rows were written.

        Raises:
            ContractViolation: If the time columns are missing, naive, or any
                record claims to have been available before it happened.
        """
        missing = [c for c in TIME_COLUMNS if c not in records.columns]
        if missing:
            raise ContractViolation(f"records are missing {missing}; every row needs {TIME_COLUMNS}")
        if records.empty:
            return 0

        frame = records.copy()
        for column in TIME_COLUMNS:
            converted = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if converted.isna().any():
                raise ContractViolation(f"{column} contains unparseable values")
            frame[column] = converted

        early = frame["available_time"] < frame["event_time"]
        if early.any():
            raise ContractViolation(
                f"{int(early.sum())} row(s) for {instrument} claim availability before the event; "
                f"the system cannot have known them at the time."
            )

        frame = frame.sort_values(list(TIME_COLUMNS))
        target = self._file(instrument)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, mode="a", header=not target.exists(), index=False)
        return len(frame)

    def revise(self, instrument: InstrumentId, records: pd.DataFrame) -> int:
        """Record a correction to values already stored.

        Identical to :meth:`append` — which is the point. A revision is a new
        observation of the same event, carrying a later ``available_time``. The
        old values stay, so a query pinned before the revision still returns what
        was known then.
        """
        return self.append(instrument, records)

    def as_of(
        self,
        instrument: InstrumentId,
        decision_time: datetime,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Everything knowable about ``instrument`` at ``decision_time``.

        For each ``event_time``, the most recently published version that was
        already available. Events after the decision time are excluded entirely.

        Args:
            instrument: What to read.
            decision_time: The instant the caller is deciding at.
            fields: Payload columns to return; all of them when omitted.

        Returns:
            Frame indexed by ``event_time``, oldest first, with an
            ``available_time`` column alongside the payload. Empty when nothing
            was knowable.
        """
        latest = latest_revisions(self.knowable(instrument, decision_time))
        if latest.empty:
            return pd.DataFrame()
        if fields is not None:
            unknown = [f for f in fields if f not in latest.columns]
            if unknown:
                raise ContractViolation(f"unknown field(s) {unknown} in dataset {self.dataset!r}")
            latest = latest[["available_time", *fields]]
        return latest

    def knowable(self, instrument: InstrumentId, decision_time: datetime) -> pd.DataFrame:
        """Every version of every event knowable at ``decision_time``, revisions included.

        :meth:`as_of` is this with only the latest version of each event kept.
        The difference matters to anything that caches one read and slices it
        for earlier decision times: a revision published between the decision
        and the read replaces the version the decision could actually see, so
        a cache must hold every version and choose per decision, never once.

        Returns:
            Frame indexed by ``event_time`` (repeated once per revision), with
            each event's versions in publication order. Empty when nothing was
            knowable.
        """
        moment = utc(decision_time)
        target = self._file(instrument)
        if not target.exists():
            return pd.DataFrame()

        frame = pd.read_csv(target)
        for column in TIME_COLUMNS:
            # An explicit format keeps pandas off the per-element dateutil path,
            # which is roughly two orders of magnitude slower on a long file.
            frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")

        knowable = frame[(frame["event_time"] <= moment) & (frame["available_time"] <= moment)]
        if knowable.empty:
            return pd.DataFrame()
        # Stable, so two rows with identical times keep file order and the one
        # appended later is the later revision -- deterministically.
        return knowable.sort_values(list(TIME_COLUMNS), kind="stable").set_index("event_time")

    def first_known(self, instrument: InstrumentId, horizon: datetime) -> pd.Series:
        """For each event, the moment it was *first* knowable, up to ``horizon``.

        Distinct from :meth:`as_of`, which returns the latest revision. When a
        split restates a year of history, every restated row arrives with a new
        ``available_time``; asking when a week became knowable must still answer
        with the original publication, not the restatement, or every past
        decision would appear to have been taken today.

        Returns:
            Series of first ``available_time`` indexed by ``event_time``, oldest
            first. Empty when nothing was knowable.
        """
        moment = utc(horizon)
        target = self._file(instrument)
        if not target.exists():
            return pd.Series(dtype="datetime64[ns, UTC]")
        frame = pd.read_csv(target, usecols=list(TIME_COLUMNS))
        for column in TIME_COLUMNS:
            frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
        known = frame[(frame["event_time"] <= moment) & (frame["available_time"] <= moment)]
        if known.empty:
            return pd.Series(dtype="datetime64[ns, UTC]")
        return known.groupby("event_time")["available_time"].min().sort_index()

    def first_event_at(
        self, instrument: InstrumentId, decision_time: datetime
    ) -> datetime | None:
        """The earliest event knowable at ``decision_time``, or None.

        Used for point-in-time availability: how much history an instrument
        actually had on a given date, rather than how much it has today.
        """
        known = self.as_of(instrument, decision_time)
        if known.empty:
            return None
        return known.index[0].to_pydatetime()

    def bars_known_at(self, instrument: InstrumentId, decision_time: datetime) -> int:
        """How many observations were knowable at ``decision_time``."""
        return len(self.as_of(instrument, decision_time))


def latest_revisions(versions: pd.DataFrame) -> pd.DataFrame:
    """The latest version of each event, from :meth:`BitemporalStore.knowable` rows.

    The whole latest row, not the last non-null value per column (which is what
    ``groupby().last()`` returns): a revision that blanks one field must not be
    stitched together with a field from the version it replaced.
    """
    if versions.empty or versions.index.is_unique:
        return versions
    return versions[~versions.index.duplicated(keep="last")]


def observations_from_bars(
    bars: pd.DataFrame, publication_lag: pd.Timedelta, available_time: datetime | None = None
) -> pd.DataFrame:
    """Turn a plain OHLCV frame into bitemporal records.

    A price file carries one timestamp. Turning it into two requires a decision
    about when each bar became knowable, and this function makes that decision
    explicit rather than assuming zero.

    Args:
        bars: Frame indexed by bar timestamp, with payload columns.
        publication_lag: How long after a bar closes before it can be acted on.
            For an end-of-session bar consumed the next morning this is hours, not
            zero.
        available_time: Override for every row — use it when loading a file whose
            contents all became known at one moment, such as a vendor download.

    Returns:
        Frame with ``event_time`` and ``available_time`` columns plus the payload.
    """
    frame = bars.copy()
    index = pd.to_datetime(frame.index, utc=True)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "event_time", index)
    if available_time is not None:
        frame.insert(1, "available_time", utc(available_time))
    else:
        frame.insert(1, "available_time", index + publication_lag)
    return frame
