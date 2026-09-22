"""Bringing the price data up to date from the broker, as a recorded fact.

Two destinations, updated together:

- the **cache CSV**, the committed reproducible input, which gains the new weeks
  and takes the vendor's latest view of any restated ones;
- the **bitemporal store**, which gains only rows that are new or changed, each
  stamped with the moment of the fetch.

That second stamp is the point. A week fetched on Saturday is recorded as known
on Saturday, so the decision that uses it is scheduled on Saturday — after the
data existed, never before. And when a split restates a year of prices, the
restated rows arrive as revisions with today's date: a query pinned before today
still sees what was known then.

Only complete bars are written. A bar still forming has a high, low and close
that are not final, and deciding on one is a discrepancy between live and
backtest that nothing downstream can detect.

The bar size is the strategy's: weekly, daily or intraday. Weekly and daily
bars are labelled by date; intraday ones keep their UTC start time, because
their close is one bar length later (``data.ingest.bar_close``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.identifiers import InstrumentId
from contracts.temporal import BarInterval
from data.bitemporal import BitemporalStore
from data.ingest import complete_bars, merge_split_weeks, to_observations
from data.vendor import CACHE_COLUMNS, cache_inventory, write_cache_csv

PRICE_COLUMNS = ("open", "high", "low", "close")


#: IBKR's bar-size strings and how far back a routine refresh re-fetches.
#: Weekly and daily go back far enough to pick up a restatement from a recent
#: split; intraday history is expensive to request and rarely restated.
IBKR_BAR_SIZES = {
    BarInterval.WEEK: ("1 week", "2 Y"),
    BarInterval.DAY: ("1 day", "1 Y"),
    BarInterval.HOUR: ("1 hour", "10 D"),
    BarInterval.MINUTE: ("1 min", "2 D"),
}


@dataclass(frozen=True, slots=True)
class RefreshResult:
    instrument: str
    new_bars: int
    revised_bars: int
    last_bar: str
    error: str = ""

    # The weekly names, for callers written before intervals were general.
    @property
    def new_weeks(self) -> int:
        return self.new_bars

    @property
    def revised_weeks(self) -> int:
        return self.revised_bars

    @property
    def last_week(self) -> str:
        return self.last_bar


def bars_from_broker(bars, interval: BarInterval = BarInterval.WEEK) -> pd.DataFrame:
    """Turn an ib_async bar list into an OHLCV frame indexed by bar label.

    Daily and weekly labels are dates. Intraday labels keep their time, in UTC.
    """
    rows = []
    for b in bars:
        stamp = pd.Timestamp(b.date)
        if interval.is_intraday:
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        elif stamp.tzinfo is not None:
            stamp = stamp.tz_localize(None)
        rows.append({
            "timestamp": stamp,
            "open": float(b.open), "high": float(b.high), "low": float(b.low),
            "close": float(b.close), "volume": float(b.volume or 0),
        })
    if not rows:
        return pd.DataFrame(columns=list(CACHE_COLUMNS))
    frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
    if not interval.is_intraday:
        frame.index = frame.index.normalize()
    return frame


def _keys(index: pd.Index, interval: BarInterval) -> list:
    if interval is BarInterval.WEEK:
        iso = pd.DatetimeIndex(index).isocalendar()
        return list(zip(iso.year, iso.week, strict=True))
    return list(pd.DatetimeIndex(index))


def merge_into_cache(
    existing: pd.DataFrame, fresh: pd.DataFrame, interval: BarInterval = BarInterval.WEEK
) -> pd.DataFrame:
    """Existing history, with the fresh bars winning wherever periods overlap."""
    if existing.empty:
        return fresh
    new_keys = set(_keys(fresh.index, interval))
    kept = existing[[key not in new_keys for key in _keys(existing.index, interval)]]
    return pd.concat([kept, fresh]).sort_index()


def changed_rows(
    store: BitemporalStore, instrument: InstrumentId, rows: pd.DataFrame, horizon: datetime
) -> tuple[pd.DataFrame, int, int]:
    """Rows that are new to the store, or differ from what it last recorded."""
    known = store.as_of(instrument, horizon)
    if known.empty:
        return rows, len(rows), 0
    new_mask, revised_mask = [], []
    for _, row in rows.iterrows():
        event = row["event_time"]
        if event not in known.index:
            new_mask.append(True)
            revised_mask.append(False)
            continue
        previous = known.loc[event]
        differs = any(
            not np.isclose(float(row[c]), float(previous[c]), rtol=1e-9, atol=1e-9)
            for c in PRICE_COLUMNS
            if c in row and c in previous
        )
        new_mask.append(False)
        revised_mask.append(differs)
    keep = [a or b for a, b in zip(new_mask, revised_mask, strict=True)]
    return rows.loc[keep], sum(new_mask), sum(revised_mask)


def refresh(
    broker,
    cache_root: Path,
    store: BitemporalStore,
    now: datetime,
    interval: BarInterval = BarInterval.WEEK,
    symbols: Sequence[str] | None = None,
    duration: str | None = None,
) -> tuple[RefreshResult, ...]:
    """Fetch recent bars for every cached instrument and record them.

    Args:
        broker: Anything with ``historical_bars(instrument, bar_size, duration)``.
        cache_root: The ``data/ibkr_cache`` directory.
        store: The bitemporal store for ``interval``.
        now: The moment of the fetch; becomes the available time of new rows.
        interval: The bar size -- the strategy's.
        symbols: Restrict to these; all cached instruments when omitted.
        duration: How far back to re-fetch, in IBKR's syntax. Defaults by
            interval (``IBKR_BAR_SIZES``).
    """
    bar_size, default_duration = IBKR_BAR_SIZES[interval]
    frequency = interval.frequency
    inventory = cache_inventory(cache_root, frequency)
    wanted = [s.upper() for s in symbols] if symbols else list(inventory["symbol"])
    results: list[RefreshResult] = []
    for symbol in wanted:
        instrument = InstrumentId(symbol)
        try:
            fresh = bars_from_broker(
                broker.historical_bars(instrument, bar_size, duration or default_duration),
                interval,
            )
        except Exception as error:  # a failure for one symbol must not stop the rest
            results.append(RefreshResult(symbol, 0, 0, "", error=str(error)))
            continue
        if interval is BarInterval.WEEK:
            fresh = merge_split_weeks(fresh)
        fresh = complete_bars(fresh, now, interval=interval)
        if fresh.empty:
            results.append(RefreshResult(symbol, 0, 0, "", error="no complete bars returned"))
            continue

        path = cache_root / frequency / f"{symbol}.csv"
        existing = pd.DataFrame(columns=list(CACHE_COLUMNS))
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        write_cache_csv(symbol, merge_into_cache(existing, fresh, interval), cache_root, frequency)

        rows = to_observations(fresh, available_at=now, interval=interval)
        delta, new, revised = changed_rows(store, instrument, rows, now)
        if not delta.empty:
            store.append(instrument, delta)
        last = fresh.index[-1]
        label = last.isoformat() if interval.is_intraday else last.date().isoformat()
        results.append(RefreshResult(symbol, new, revised, label))
    return tuple(results)


def refresh_weekly(
    broker,
    cache_root: Path,
    store: BitemporalStore,
    now: datetime,
    symbols: Sequence[str] | None = None,
    duration: str = "2 Y",
) -> tuple[RefreshResult, ...]:
    """:func:`refresh` for weekly bars. Kept for callers written before intervals."""
    return refresh(broker, cache_root, store, now, BarInterval.WEEK, symbols, duration)
