"""The composition root: where the pieces are connected, and the only such place.

Everything above this module depends on contracts. This module depends on
everything, and nothing depends on it — the architecture test enforces both
halves. That is what makes a second broker, a second data source or a second
strategy a change *here* rather than a change everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from contracts.temporal import BarInterval, utc
from data.bitemporal import BitemporalStore
from data.filtration import ReplayFiltrations
from data.universe import PointInTimeUniverse

#: Far enough ahead that every stored row is knowable: used only to find the
#: last bar, never to answer a question about a decision.
_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)

#: Bars are stamped available a quarter of an hour after the session close by the
#: ingest, so a decision taken on a close happens at the close plus this.
DECISION_OFFSET = timedelta(minutes=15)


@dataclass(frozen=True, slots=True)
class MarketWindow:
    """Prices by week, the decision schedule, and which prices are actually fresh.

    Instruments do not share a calendar. IBKR stamps a weekly bar on the first
    trading day of the week, so a holiday shifts one instrument's stamp and not
    another's, and a few weeks are split into two bars. Across this universe that
    is 1,547 distinct timestamps covering 1,437 weeks.

    The previous system reindexed everything onto SPY's calendar and forward
    filled. That keeps the panel rectangular, and it means some backtested
    **fills happened at a price that was never quoted that week**. Forward filling
    an indicator is a modelling choice one can argue about; forward filling an
    execution price is inventing liquidity.

    Here, bars are matched by ISO week rather than by exact timestamp, and
    nothing is filled. A week where an instrument did not print is a week it
    cannot be traded — which is what a halt is — and :meth:`fresh_at` says so.
    Valuation still needs a number for a held position, so :meth:`marks_at`
    carries the last known close forward; the two are kept separate precisely so
    that a carried price can value a position without ever pricing a trade.
    """

    opens: pd.DataFrame
    closes: pd.DataFrame
    lows: pd.DataFrame
    fresh: pd.DataFrame
    index: tuple[datetime, ...]
    schedule: tuple[datetime, ...]

    @classmethod
    def from_store(
        cls,
        store: BitemporalStore,
        universe: PointInTimeUniverse,
        horizon: datetime,
        start: datetime | None = None,
    ) -> MarketWindow:
        """Load every member's bars once and align them by ISO week.

        Args:
            store: Where the bars live.
            universe: Which instruments to load.
            horizon: The as-of time to read the store at.
            start: Earliest decision moment to schedule, if not the beginning.
        """
        horizon = utc(horizon)
        opens: dict[str, pd.Series] = {}
        closes: dict[str, pd.Series] = {}
        lows: dict[str, pd.Series] = {}
        stamps: dict[tuple[int, int], list[pd.Timestamp]] = {}

        for instrument in universe.survivors_only():
            known = store.as_of(instrument, horizon, fields=["open", "low", "close"])
            if known.empty:
                continue
            iso = known.index.isocalendar()
            weeks = pd.MultiIndex.from_arrays([iso.year, iso.week])
            # Two bars in one week happens where a session was split. The later
            # one carries the week's close, so it wins.
            frame = known.assign(_week=weeks)
            frame = frame[~frame["_week"].duplicated(keep="last")]
            index = pd.MultiIndex.from_tuples(list(frame["_week"]), names=("year", "week"))
            opens[str(instrument)] = pd.Series(frame["open"].astype(float).values, index=index)
            lows[str(instrument)] = pd.Series(frame["low"].astype(float).values, index=index)
            closes[str(instrument)] = pd.Series(frame["close"].astype(float).values, index=index)
            for week, stamp in zip(index, frame.index, strict=True):
                stamps.setdefault(week, []).append(stamp)

        if not closes:
            raise ContractViolation("no instruments had any bars at this horizon")

        close_frame = pd.DataFrame(closes).sort_index()
        open_frame = pd.DataFrame(opens).reindex(close_frame.index)
        low_frame = pd.DataFrame(lows).reindex(close_frame.index)
        fresh = close_frame.notna() & (close_frame > 0)

        # One decision moment per week, at the last close that week plus the
        # publication lag: the earliest instant every instrument's bar for that
        # week was knowable.
        # One entry per frame row, always. ``schedule`` is a subset of these;
        # the two are kept separate because indexing the frames by a position in
        # a filtered schedule reads a different week and does it silently.
        index = tuple(
            max(stamps[tuple(week)]).to_pydatetime() + DECISION_OFFSET
            for week in close_frame.index
        )
        schedule = tuple(m for m in index if start is None or m >= utc(start))

        return cls(
            opens=open_frame,
            closes=close_frame.ffill(),
            lows=low_frame,
            fresh=fresh,
            index=index,
            schedule=schedule,
        )

    def _position(self, moment: datetime) -> int:
        """The frame row for a moment. Against ``index``, never ``schedule``."""
        try:
            return self.index.index(utc(moment))
        except ValueError as error:
            raise ContractViolation(f"no week at {utc(moment).isoformat()}") from error

    def marks_at(self, moment: datetime) -> Mapping[InstrumentId, float]:
        """Closing prices for valuation and sizing, carried forward if stale.

        Carried rather than dropped so that a held position always has a value.
        Use :meth:`fresh_at` before trading on one of these.
        """
        row = self.closes.iloc[self._position(moment)]
        return {
            InstrumentId(name): float(value)
            for name, value in row.items()
            if pd.notna(value) and value > 0
        }

    def fresh_at(self, moment: datetime) -> frozenset[InstrumentId]:
        """Instruments that actually printed a bar this week, and so may trade."""
        row = self.fresh.iloc[self._position(moment)]
        return frozenset(InstrumentId(name) for name, ok in row.items() if bool(ok))

    def lows_at(self, moment: datetime) -> Mapping[InstrumentId, float]:
        """The bar's lows: what a resting stop is triggered against.

        A stop is a price the market has to *touch*, not one it has to close at.
        Checking only opens and closes misses most of the weeks a stop would
        actually have fired, which makes a backtest with stops look more like one
        without them.
        """
        row = self.lows.iloc[self._position(moment)]
        return {
            InstrumentId(name): float(value)
            for name, value in row.items()
            if pd.notna(value) and value > 0
        }

    def opens_at(self, moment: datetime) -> Mapping[InstrumentId, float]:
        """Opening prices of this week's bar: where orders fill.

        A decision taken on one week's close is executed at the next week's open,
        the first price it could actually have reached. Only genuine opens appear
        here: an instrument that did not print has no fill price, and its order
        goes unfilled rather than transacting at a stale one.
        """
        row = self.opens.iloc[self._position(moment)]
        return {
            InstrumentId(name): float(value)
            for name, value in row.items()
            if pd.notna(value) and value > 0
        }


@dataclass(frozen=True, slots=True)
class Market:
    """A loaded dataset: the store, its universe, filtrations and prices."""

    store: BitemporalStore
    universe: PointInTimeUniverse
    filtrations: ReplayFiltrations
    window: MarketWindow
    interval: BarInterval

    @property
    def schedule(self) -> Sequence[datetime]:
        return self.window.schedule

    @property
    def periods_per_year(self) -> float:
        return self.interval.periods_per_year

    def filtration_at(self, moment: datetime):
        """The knowable view at a decision moment."""
        return self.filtrations.at(moment)


def load_market(
    root: Path,
    interval: BarInterval = BarInterval.WEEK,
    horizon: datetime | None = None,
    start: datetime | None = None,
    min_bars: int = 1,
) -> Market:
    """Assemble everything a run needs from a store on disk.

    Args:
        root: The ``var/store`` directory.
        interval: Bar size; picks the dataset and the universe file.
        horizon: As-of time for reading the store. Defaults to the last bar, so a
            run is reproducible rather than dependent on the wall clock.
        start: Earliest decision moment.
        min_bars: Default history an instrument needs to count as available.
    """
    frequency = "weekly" if interval is BarInterval.WEEK else "daily"
    dataset = f"bars_{interval.value.lower()}"
    store = BitemporalStore(root, dataset)
    universe = PointInTimeUniverse.from_csv(root / f"universe_{frequency}.csv")

    if horizon is None:
        latest = max(
            (
                store.as_of(instrument, _FAR_FUTURE).index.max()
                for instrument in store.instruments()
            ),
            default=None,
        )
        if latest is None:
            raise ContractViolation(f"dataset {dataset} is empty")
        horizon = latest.to_pydatetime() + timedelta(days=1)

    return Market(
        store=store,
        universe=universe,
        filtrations=ReplayFiltrations(store, universe, horizon, min_bars=min_bars),
        window=MarketWindow.from_store(store, universe, horizon, start=start),
        interval=interval,
    )
