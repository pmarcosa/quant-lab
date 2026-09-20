"""The concrete filtration a strategy is handed.

This is where the point-in-time rule stops being a policy and becomes the only
thing available. A strategy receives one of these, pinned to a decision time at
construction. Its methods take a count of bars, never a date range, so there is
no argument through which a future date could be passed.

Reads are memoised per instrument because a strategy typically asks for several
fields of the same instrument while ranking, and every read otherwise re-filters
the whole file.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from contracts.errors import CausalityViolation
from contracts.identifiers import InstrumentId
from contracts.temporal import Filtration, utc
from contracts.universe import UniverseSource
from data.bitemporal import BitemporalStore


class StoreFiltration(Filtration):
    """A filtration backed by a bitemporal store and a point-in-time universe.

    Args:
        store: Where the observations live.
        universe: Which instruments existed at the decision time.
        decision_time: The instant this view is pinned to.
        min_bars: Default history an instrument needs to count as available.
    """

    def __init__(
        self,
        store: BitemporalStore,
        universe: UniverseSource,
        decision_time: datetime,
        min_bars: int = 1,
    ) -> None:
        self._store = store
        self._universe = universe
        self._decision_time = utc(decision_time)
        self._min_bars = min_bars
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def decision_time(self) -> datetime:
        return self._decision_time

    def _known(self, instrument: InstrumentId) -> pd.DataFrame:
        key = str(instrument)
        if key not in self._cache:
            self._cache[key] = self._store.as_of(instrument, self._decision_time)
        return self._cache[key]

    def _eligible(self, instrument: InstrumentId) -> bool:
        return instrument in self._universe.members_at(self._decision_time)

    def history(self, instrument: InstrumentId, field: str, count: int) -> pd.Series:
        """The last ``count`` values of ``field``, oldest first.

        Returns an empty series when the instrument was not a member at the
        decision time — not a partial one, because a name that had not listed has
        no history rather than a short one.

        Raises:
            CausalityViolation: If ``count`` is not positive.
        """
        if count <= 0:
            raise CausalityViolation(f"count must be positive; got {count}")
        if not self._eligible(instrument):
            return pd.Series(dtype="float64")
        known = self._known(instrument)
        if known.empty or field not in known.columns:
            return pd.Series(dtype="float64")
        return known[field].tail(count).astype(float)

    def frame(
        self, instruments: Sequence[InstrumentId], field: str, count: int
    ) -> pd.DataFrame:
        """``history`` for several instruments, aligned on a common index.

        Alignment is a plain outer join with no forward fill: a gap stays a gap,
        because filling one invents a price that never traded.
        """
        columns = {
            str(instrument): self.history(instrument, field, count)
            for instrument in instruments
        }
        populated = {name: series for name, series in columns.items() if not series.empty}
        if not populated:
            return pd.DataFrame()
        return pd.DataFrame(populated).sort_index()

    def is_available(self, instrument: InstrumentId, min_bars: int | None = None) -> bool:
        """Whether the instrument had listed and has enough history at this instant."""
        required = self._min_bars if min_bars is None else min_bars
        if not self._eligible(instrument):
            return False
        return len(self._known(instrument)) >= required

    def universe(self, min_bars: int | None = None) -> tuple[InstrumentId, ...]:
        """Every instrument tradable and sufficiently seasoned at this instant."""
        return tuple(
            instrument
            for instrument in self._universe.members_at(self._decision_time)
            if self.is_available(instrument, min_bars)
        )

    def metadata(self, instrument: InstrumentId) -> dict[str, Any]:
        """What is known about the instrument at this instant."""
        known = self._known(instrument)
        return {
            "instrument": str(instrument),
            "eligible": self._eligible(instrument),
            "bars_known": len(known),
            "first_event": known.index[0].to_pydatetime() if len(known) else None,
            "last_event": known.index[-1].to_pydatetime() if len(known) else None,
            "decision_time": self._decision_time,
        }


class ReplayFiltrations:
    """Filtrations for many decision times, reading each instrument once.

    A backtest asks for a new filtration at every bar. Building each one straight
    from the store re-reads and re-filters every CSV, which for a 900-week run
    over 39 instruments is 35,000 file reads and turns a two-second backtest into
    a two-minute one. This holds each instrument's full history in memory and
    slices it per decision time.

    The important thing is what this does **not** change. Every view is still
    produced by the same two conditions — ``event_time <= t`` and
    ``available_time <= t`` — applied to the same rows; only the source of the
    rows differs. Nothing here can widen what a decision can see, and
    ``test_the_cache_cannot_widen_what_is_visible`` pins that against the
    uncached path on real data rather than trusting the argument.

    The loaded frame is itself point-in-time: it is read once at a *horizon*, and
    a decision after that horizon is refused rather than served stale data.
    """

    def __init__(
        self,
        store: BitemporalStore,
        universe: UniverseSource,
        horizon: datetime,
        min_bars: int = 1,
    ) -> None:
        self._store = store
        self._universe = universe
        self._horizon = utc(horizon)
        self._min_bars = min_bars
        self._frames: dict[str, pd.DataFrame] = {}

    def _frame(self, instrument: InstrumentId) -> pd.DataFrame:
        key = str(instrument)
        if key not in self._frames:
            self._frames[key] = self._store.as_of(instrument, self._horizon)
        return self._frames[key]

    def at(self, decision_time: datetime) -> _SlicedFiltration:
        """The view pinned at ``decision_time``.

        Raises:
            CausalityViolation: If the moment is after the horizon this was
                loaded at, because the answer would be missing rows that existed
                by then rather than merely filtered.
        """
        moment = utc(decision_time)
        if moment > self._horizon:
            raise CausalityViolation(
                f"decision time {moment.isoformat()} is past the load horizon "
                f"{self._horizon.isoformat()}; reload rather than serve stale history"
            )
        return _SlicedFiltration(self, moment)


class _SlicedFiltration(Filtration):
    """One decision time's view over :class:`ReplayFiltrations`' loaded frames."""

    def __init__(self, source: ReplayFiltrations, decision_time: datetime) -> None:
        self._source = source
        self._decision_time = decision_time
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def decision_time(self) -> datetime:
        return self._decision_time

    def _known(self, instrument: InstrumentId) -> pd.DataFrame:
        key = str(instrument)
        if key not in self._cache:
            frame = self._source._frame(instrument)
            if frame.empty:
                self._cache[key] = frame
            else:
                visible = frame[frame["available_time"] <= self._decision_time]
                self._cache[key] = visible[visible.index <= self._decision_time]
        return self._cache[key]

    def _eligible(self, instrument: InstrumentId) -> bool:
        members = self._source._universe.members_at(self._decision_time)
        return instrument in set(members)

    def history(self, instrument: InstrumentId, field: str, count: int) -> pd.Series:
        if count <= 0:
            raise CausalityViolation(f"count must be positive; got {count}")
        if not self._eligible(instrument):
            return pd.Series(dtype="float64")
        known = self._known(instrument)
        if known.empty or field not in known.columns:
            return pd.Series(dtype="float64")
        return known[field].tail(count).astype(float)

    def frame(
        self, instruments: Sequence[InstrumentId], field: str, count: int
    ) -> pd.DataFrame:
        columns = {
            str(instrument): self.history(instrument, field, count)
            for instrument in instruments
        }
        populated = {name: series for name, series in columns.items() if not series.empty}
        if not populated:
            return pd.DataFrame()
        return pd.DataFrame(populated).sort_index()

    def is_available(self, instrument: InstrumentId, min_bars: int | None = None) -> bool:
        required = self._source._min_bars if min_bars is None else min_bars
        if not self._eligible(instrument):
            return False
        return len(self._known(instrument)) >= required

    def universe(self, min_bars: int | None = None) -> tuple[InstrumentId, ...]:
        return tuple(
            instrument
            for instrument in self._source._universe.members_at(self._decision_time)
            if self.is_available(instrument, min_bars)
        )

    def metadata(self, instrument: InstrumentId) -> dict[str, Any]:
        known = self._known(instrument)
        return {
            "instrument": str(instrument),
            "eligible": self._eligible(instrument),
            "bars_known": len(known),
            "first_event": known.index[0].to_pydatetime() if len(known) else None,
            "last_event": known.index[-1].to_pydatetime() if len(known) else None,
            "decision_time": self._decision_time,
        }
