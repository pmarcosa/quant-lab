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

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.identifiers import InstrumentId
from contracts.temporal import BarInterval
from data.adjustments import (
    adjust,
    factors_from,
    is_rescaled,
    merge_factors,
    read_factors,
    rescaling,
    weeks_from_trades,
    write_factors,
)
from data.bitemporal import BitemporalStore
from data.ingest import complete_bars, to_observations
from data.vendor import CACHE_COLUMNS, cache_inventory, write_cache_csv

PRICE_COLUMNS = ("open", "high", "low", "close")


#: IBKR's bar-size strings and how far back a routine refresh re-fetches.
#: Weekly and daily go back far enough to pick up a restatement from a recent
#: split; intraday history is expensive to request and rarely restated.
IBKR_BAR_SIZES = {
    # Weekly asks for daily bars, grouped into ISO weeks by merge_split_weeks:
    # IBKR refuses ADJUSTED_LAST for any bar longer than a day (error 321).
    BarInterval.WEEK: ("1 day", "2 Y"),
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
    note: str = ""

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


#: How far back an intraday refresh re-reads the dividend factors. It has to
#: overlap the previous download (data.adjustments.merge_factors).
FACTOR_WINDOW = "2 Y"
#: A symbol's history when nothing is cached: what ``ql data fetch`` asks for.
FULL_HISTORY = "22 Y"


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

    The cache takes IBKR's TRADES bars and the factor table the dividend factors;
    the store takes TRADES × factor (``data.adjustments``). Two events rewrite
    history, and both are recorded as revisions rather than silently merged:

    - a dividend that went ex since the last refresh rescales every earlier
      factor by one constant, so every stored bar is revised;
    - a split since the last download shows as a constant ratio between the new
      and the cached closes; the symbol's whole history is downloaded again
      (the expert's rule: re-download, never patch).

    Args:
        broker: Anything with ``historical_bars(instrument, bar_size, duration,
            end=None, what="TRADES")``.
        cache_root: The ``data/ibkr_cache`` directory.
        store: The bitemporal store for ``interval``.
        now: The moment of the fetch; becomes the available time of new rows.
        interval: The bar size -- the strategy's.
        symbols: Restrict to these; all cached instruments when omitted.
        duration: How far back to re-fetch, in IBKR's syntax. Defaults by
            interval (``IBKR_BAR_SIZES``).
    """
    _, default_duration = IBKR_BAR_SIZES[interval]
    frequency = interval.frequency
    inventory = cache_inventory(cache_root, frequency)
    wanted = [s.upper() for s in symbols] if symbols else list(inventory["symbol"])
    results: list[RefreshResult] = []
    for symbol in wanted:
        instrument = InstrumentId(symbol)
        path = cache_root / frequency / f"{symbol}.csv"
        existing = _read_cache(path, interval)
        stored = read_factors(cache_root, symbol)
        note = ""
        # No factors yet (a cache from before them, or from the connector): the
        # whole history once, so every bar gets its factor. Cached bars older
        # than what IBKR still serves are kept, at the same (split-adjusted) scale.
        full = stored is None and not interval.is_intraday
        try:
            fresh, factors = _download(
                broker, instrument, interval,
                _span(existing, now) if full else duration or default_duration,
                factor_duration=_span(existing, now) if stored is None else None,
            )
            fresh = complete_bars(fresh, now, interval=interval)
            if fresh.empty or factors is None:
                results.append(RefreshResult(symbol, 0, 0, "", error="no complete bars returned"))
                continue
            if full:
                note = "no dividend factors yet: full history downloaded"
            k = rescaling(existing, fresh, interval)
            if is_rescaled(k):
                note = f"split since the last download (x{k:.4g}): "
                if interval.is_intraday:
                    note += "cache restarted; run `ql data backfill` for its history"
                else:
                    if not full:
                        fresh, factors = _download(
                            broker, instrument, interval, _span(existing, now))
                        fresh = complete_bars(fresh, now, interval=interval)
                    stored = None  # the new download is whole; factors ignore splits anyway
                    note += "full history downloaded again"
                existing = _empty()
            merged, c = merge_factors(stored, factors)
        except Exception as error:  # a failure for one symbol must not stop the rest
            results.append(RefreshResult(symbol, 0, 0, "", error=str(error)))
            continue

        cache = merge_into_cache(existing, fresh, interval)
        write_cache_csv(symbol, cache, cache_root, frequency)
        write_factors(cache_root, symbol, merged)
        if is_rescaled(c) and not note:
            note = f"dividend since the last refresh: history rescaled x{c:.4f}"
        # A rescaled history revises every stored bar, not only the window.
        source = complete_bars(cache, now, interval=interval) if note else fresh
        rows = to_observations(adjust(source, merged, interval), available_at=now,
                               interval=interval)
        delta, new, revised = changed_rows(store, instrument, rows, now)
        if not delta.empty:
            store.append(instrument, delta)
        last = fresh.index[-1]
        label = last.isoformat() if interval.is_intraday else last.date().isoformat()
        results.append(RefreshResult(symbol, new, revised, label, note=note))
    return tuple(results)


def _download(
    broker,
    instrument: InstrumentId,
    interval: BarInterval,
    duration: str,
    factor_duration: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """TRADES bars and dividend factors for one instrument (``data.adjustments``).

    Weekly bars are built from daily ones; the factors come from daily TRADES and
    ADJUSTED_LAST over the same window (for intraday, over ``factor_duration``).
    """
    bar_size, _ = IBKR_BAR_SIZES[interval]
    raw = bars_from_broker(
        broker.historical_bars(instrument, bar_size, duration, what="TRADES"), interval
    )
    if interval.is_intraday:
        raw = _regular(raw)
        factor_span = factor_duration or FACTOR_WINDOW
        daily = bars_from_broker(
            broker.historical_bars(instrument, "1 day", factor_span, what="TRADES"),
            BarInterval.DAY,
        )
    else:
        factor_span, daily = duration, raw
    if raw.empty:
        return raw, None
    adjusted = bars_from_broker(
        broker.historical_bars(instrument, "1 day", factor_span, what="ADJUSTED_LAST"),
        BarInterval.DAY,
    )
    factors = factors_from(daily, adjusted)
    bars = weeks_from_trades(raw, factors) if interval is BarInterval.WEEK else raw
    return bars, factors


def _span(existing: pd.DataFrame, now: datetime) -> str:
    """A duration reaching back past the first cached bar (the full default if none)."""
    if existing.empty:
        return FULL_HISTORY
    first = _as_utc(existing.index.min())
    years = (now - first).days / 365.25
    return f"{min(30, int(np.ceil(years)) + 1)} Y"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CACHE_COLUMNS))


def _read_cache(path: Path, interval: BarInterval) -> pd.DataFrame:
    if not path.exists():
        return _empty()
    frame = pd.read_csv(path)
    stamps = pd.to_datetime(frame.pop("timestamp"), utc=interval.is_intraday)
    frame.index = pd.DatetimeIndex(stamps, name="timestamp")
    return frame.sort_index()


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


# -- backfilling long intraday history --------------------------------------------------------

#: How much each request asks for when paging back through history. IBKR's
#: documented maximum per request is larger; these keep each response to a few
#: thousand bars, which returns promptly.
BACKFILL_CHUNKS = {
    BarInterval.WEEK: "20 Y", BarInterval.DAY: "10 Y",
    BarInterval.HOUR: "1 Y", BarInterval.MINUTE: "1 M",
}
#: IBKR allows at most 60 historical-data requests in any ten minutes. Ten and a
#: half seconds apart never trips it, whatever else is running.
PACING_SECONDS = 10.5


@dataclass(frozen=True, slots=True)
class BackfillResult:
    instrument: str
    bars_added: int
    first_bar: str
    requests: int
    exhausted: bool
    error: str = ""
    note: str = ""


def backfill(
    broker,
    cache_root: Path,
    interval: BarInterval,
    now: datetime,
    years: float,
    symbols: Sequence[str],
    chunk: str | None = None,
    pace: float = PACING_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[BackfillResult, ...]:
    """Page backwards through the broker's history until ``years`` are cached.

    Pages are TRADES bars, the only series IBKR serves with an end date. The
    dividend factors for the whole span come first, from two daily requests
    (``data.adjustments``), and the store applies them when it is built.

    Resumable: it starts from the earliest bar already in each symbol's cache
    and writes every page as it arrives, so an interruption loses one request.
    The first page reaches one day into what is cached; if the shared bars show
    a split since the cache was written, the cache is started again from now
    (the expert's rule: re-download, never patch). Stops early, and says so,
    when the broker has nothing older. Writes the cache only; ``ql data ingest
    --rebuild`` then loads it into the store, as for any other cache change.

    For intraday bars only regular-session bars are kept (``data.calendar``).
    """
    bar_size, _ = IBKR_BAR_SIZES[interval]
    step = chunk or BACKFILL_CHUNKS[interval]
    target = now - timedelta(days=365.25 * years)
    factor_span = f"{int(np.ceil(years)) + 1} Y"
    frequency = interval.frequency
    results: list[BackfillResult] = []
    first_request = True

    def ask(*args, **kwargs):
        nonlocal first_request
        if not first_request:
            sleep(pace)
        first_request = False
        return broker.historical_bars(*args, **kwargs)

    for symbol in [s.upper() for s in symbols]:
        instrument = InstrumentId(symbol)
        path = cache_root / frequency / f"{symbol}.csv"
        existing = _read_cache(path, interval)
        requests, added, exhausted, error, note = 0, 0, False, "", ""
        try:
            daily = bars_from_broker(
                ask(instrument, "1 day", factor_span, what="TRADES"), BarInterval.DAY)
            requests += 1
            adjusted = bars_from_broker(
                ask(instrument, "1 day", factor_span, what="ADJUSTED_LAST"), BarInterval.DAY)
            requests += 1
            merged, _ = merge_factors(read_factors(cache_root, symbol),
                                      factors_from(daily, adjusted))
            write_factors(cache_root, symbol, merged)
        except Exception as failure:  # one symbol's failure must not stop the rest
            results.append(BackfillResult(symbol, 0, _first(existing, interval), requests,
                                          False, f"dividend factors: {failure}"))
            continue

        end = _as_utc(existing.index.min()) if not existing.empty else now
        check = not existing.empty
        while end > target:
            asked = min(end + timedelta(days=1), now) if check else end
            try:
                page = bars_from_broker(
                    ask(instrument, bar_size, step, end=asked, what="TRADES"), interval
                )
            except Exception as failure:  # one symbol's failure must not stop the rest
                error = str(failure)
                break
            requests += 1
            if interval.is_intraday:
                page = _regular(page)
            if check:
                check = False
                shared = page[[_as_utc(t) >= end for t in page.index]] if not page.empty else page
                try:
                    k = rescaling(existing, shared, interval)
                except Exception as failure:
                    error = str(failure)
                    break
                if is_rescaled(k):
                    note = f"split since the cache was written (x{k:.4g}): started again from now"
                    existing = _empty()
                    path.unlink(missing_ok=True)
                    end = now
                    continue
            page = page[[_as_utc(t) < end for t in page.index]] if not page.empty else page
            if page.empty:
                exhausted = True
                break
            existing = merge_into_cache(existing, page, interval)
            write_cache_csv(symbol, existing, cache_root, frequency)
            added += len(page)
            end = _as_utc(page.index.min())
        results.append(BackfillResult(symbol, added, _first(existing, interval), requests,
                                      exhausted, error, note))
    return tuple(results)


def _first(existing: pd.DataFrame, interval: BarInterval) -> str:
    return "" if existing.empty else _label(existing.index.min(), interval)


def _as_utc(stamp) -> datetime:
    value = pd.Timestamp(stamp)
    value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return value.to_pydatetime()


def _label(stamp, interval: BarInterval) -> str:
    value = pd.Timestamp(stamp)
    return value.isoformat() if interval.is_intraday else value.date().isoformat()


def _regular(frame: pd.DataFrame) -> pd.DataFrame:
    """Regular-session bars only, when the calendar is installed."""
    from data import calendar

    return calendar.regular_bars(frame) if calendar.is_available() else frame
