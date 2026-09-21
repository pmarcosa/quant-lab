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

Only complete weeks are written. A bar for the week in progress has a high, low
and close that are not final, and deciding on one is a discrepancy between live
and backtest that nothing downstream can detect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.identifiers import InstrumentId
from data.bitemporal import BitemporalStore
from data.ingest import complete_bars, merge_split_weeks, to_observations
from data.vendor import CACHE_COLUMNS, cache_inventory, write_cache_csv

PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class RefreshResult:
    instrument: str
    new_weeks: int
    revised_weeks: int
    last_week: str
    error: str = ""


def bars_from_broker(bars) -> pd.DataFrame:
    """Turn an ib_async bar list into an OHLCV frame indexed by bar label."""
    rows = [
        {
            "timestamp": pd.Timestamp(b.date).tz_localize(None)
            if getattr(pd.Timestamp(b.date), "tzinfo", None)
            else pd.Timestamp(b.date),
            "open": float(b.open), "high": float(b.high), "low": float(b.low),
            "close": float(b.close), "volume": float(b.volume or 0),
        }
        for b in bars
    ]
    if not rows:
        return pd.DataFrame(columns=list(CACHE_COLUMNS))
    frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
    frame.index = frame.index.normalize()
    return frame


def merge_into_cache(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Existing history, with the fresh bars winning wherever weeks overlap."""
    if existing.empty:
        return fresh
    iso_old = existing.index.isocalendar()
    iso_new = fresh.index.isocalendar()
    old_keys = list(zip(iso_old.year, iso_old.week, strict=True))
    new_keys = set(zip(iso_new.year, iso_new.week, strict=True))
    kept = existing[[key not in new_keys for key in old_keys]]
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


def refresh_weekly(
    broker,
    cache_root: Path,
    store: BitemporalStore,
    now: datetime,
    symbols: Sequence[str] | None = None,
    duration: str = "2 Y",
) -> tuple[RefreshResult, ...]:
    """Fetch recent weekly bars for every cached instrument and record them.

    Args:
        broker: Anything with ``historical_bars(instrument, bar_size, duration)``.
        cache_root: The ``data/ibkr_cache`` directory.
        store: The weekly bitemporal store.
        now: The moment of the fetch; becomes the available time of new rows.
        symbols: Restrict to these; all cached instruments when omitted.
        duration: How far back to re-fetch. Two years is enough to pick up a
            restatement from a recent split without re-fetching all history.
    """
    inventory = cache_inventory(cache_root, "weekly")
    wanted = [s.upper() for s in symbols] if symbols else list(inventory["symbol"])
    results: list[RefreshResult] = []
    for symbol in wanted:
        instrument = InstrumentId(symbol)
        try:
            fresh = bars_from_broker(broker.historical_bars(instrument, "1 week", duration))
        except Exception as error:  # a failure for one symbol must not stop the rest
            results.append(RefreshResult(symbol, 0, 0, "", error=str(error)))
            continue
        fresh = complete_bars(merge_split_weeks(fresh), now, week_ending=True)
        if fresh.empty:
            results.append(RefreshResult(symbol, 0, 0, "", error="no complete bars returned"))
            continue

        path = cache_root / "weekly" / f"{symbol}.csv"
        existing = pd.DataFrame(columns=list(CACHE_COLUMNS))
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        write_cache_csv(symbol, merge_into_cache(existing, fresh), cache_root, "weekly")

        rows = to_observations(fresh, week_ending=True, available_at=now)
        delta, new, revised = changed_rows(store, instrument, rows, now)
        if not delta.empty:
            store.append(instrument, delta)
        results.append(
            RefreshResult(symbol, new, revised, fresh.index[-1].date().isoformat())
        )
    return tuple(results)
