"""Turning a vendor's price payload into the cache format, and nothing more.

Two things live here and they are deliberately separate:

- :func:`bars_from_connector` converts what IBKR returns into a frame. It is
  pure, so it is tested against real payload shapes rather than against a live
  connection.
- :func:`write_cache_csv` writes that frame where the ingest expects it.

Fetching itself is *not* here. A network call inside the data layer is how a
backtest quietly acquires a dependency on whether the market is open, so the
fetch lives in a script that a person runs on purpose.

The cache format is the contract between whoever fetched the data and the
ingest: ``timestamp,open,high,low,close,volume``, one row per bar, oldest first,
dates only. It is deliberately dull and diffable — these files are committed, so
a change in the data shows up in a review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts.errors import ContractViolation

CACHE_COLUMNS = ("open", "high", "low", "close", "volume")

#: What the ingest reads. Anything else is a different dataset, not a variant.
FREQUENCIES = ("weekly", "daily", "hourly", "minute")
INTRADAY_FREQUENCIES = ("hourly", "minute")


def bars_from_connector(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert an IBKR price-history payload into an OHLCV frame.

    The payload is column-oriented — parallel arrays for ``time``, ``open``,
    ``high``, ``low``, ``close`` and ``volume`` — so the first thing to check is
    that they are the same length. A short array would otherwise pair a price
    with the wrong date and produce a file that looks perfectly ordinary.

    Args:
        payload: The connector's response, already parsed from JSON.

    Returns:
        Frame indexed by bar date, oldest first, with the cache's columns.

    Raises:
        ContractViolation: If a field is missing, the arrays disagree in length,
            or a price is not positive.
    """
    missing = [f for f in ("time", *CACHE_COLUMNS) if f not in payload]
    if missing:
        raise ContractViolation(f"payload is missing field(s) {missing}")

    lengths = {field: len(payload[field]) for field in ("time", *CACHE_COLUMNS)}
    if len(set(lengths.values())) != 1:
        raise ContractViolation(
            f"payload arrays disagree in length: {lengths}. Pairing a price with "
            f"the wrong date produces a file that looks entirely normal."
        )
    if lengths["time"] == 0:
        raise ContractViolation("payload contains no bars")

    frame = pd.DataFrame(
        {field: payload[field] for field in CACHE_COLUMNS},
        index=pd.to_datetime(payload["time"], utc=True, format="ISO8601"),
    ).sort_index()
    frame.index.name = "timestamp"

    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            bad = frame.index[values.isna() | (values <= 0)][0]
            raise ContractViolation(f"{column} is not positive at {bad.date()}")
        frame[column] = values.astype(float)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype("int64")

    inconsistent = frame[(frame["high"] < frame["low"]) | (frame["high"] < frame["open"])]
    if not inconsistent.empty:
        raise ContractViolation(
            f"high is below low or open at {inconsistent.index[0].date()}; "
            f"the payload is not a consistent set of bars"
        )
    if frame.index.duplicated().any():
        raise ContractViolation("payload contains two bars for the same timestamp")
    return frame


#: Corporate actions that change the number of shares: their ``value`` is new
#: shares per old one (a 10-for-1 split is 10.0, a 1-for-10 reverse split 0.1,
#: NVIDIA's 2021 four-for-one, listed as a stock dividend, 4.0).
SHARE_ACTIONS = ("Splits", "StockDividends")


def connector_dividends(payload: Mapping[str, Any]) -> tuple[tuple[date, float, str], ...]:
    """The cash dividends a price-history answer lists, per share as it trades today.

    The answer lists each dividend per share *as it was paid*, while its bars are
    adjusted for every split since. A dividend paid before a split is therefore
    divided by the shares each old share became -- the product of the
    ``SHARE_ACTIONS`` dated after its ex-date. Left as listed, NVIDIA's 0.16 of
    2020 would be set against a split-adjusted price of about 0.60 and take a
    quarter off every earlier bar.

    Returns:
        ``(ex_date, amount, currency)``, oldest first.

    Raises:
        ContractViolation: If a share action's ratio is not a positive number.
    """
    actions = payload.get("corp_actions") or ()
    listed = sorted(
        (_day(a["date"]), float(a["value"]), str(a.get("currency", "USD")))
        for a in actions if a.get("type") == "CashDividends"
    )
    changes = []
    for action in actions:
        if action.get("type") in SHARE_ACTIONS:
            ratio = float(action["value"])
            if not np.isfinite(ratio) or ratio <= 0:
                raise ContractViolation(
                    f"{action.get('type')} on {action.get('date')} has ratio {ratio}")
            day = _day(action["date"])
            if ratio < 1.0 and not _reverse_split(day, ratio, listed):
                continue
            changes.append((day, ratio))
    found = []
    for ex, amount, currency in listed:
        later = float(np.prod([ratio for day, ratio in changes if day > ex])) if changes else 1.0
        found.append((ex, amount / later, currency))
    return tuple(found)


def _reverse_split(day: date, ratio: float, listed) -> bool:
    """Whether a share action below 1 behaves like a reverse split in the dividends.

    IBKR lists a spin-off's price adjustment as a split too (Honeywell, 0.5 on
    2026-06-29), and a spin-off leaves each share a share. A reverse split of
    1-for-n multiplies the dividend per share by about n; a spin-off cuts it. So
    the last dividend before, scaled as if the shares had been consolidated, is
    compared with the first one after: within a factor of 2.5 it was a reverse
    split (Duke Energy 2012: 0.25 / 0.333 = 0.75 against 0.765 after); further
    apart, or with no dividend on one side to compare, it is not counted.
    """
    before = [amount for ex, amount, _ in listed if ex < day]
    after = [amount for ex, amount, _ in listed if ex >= day]
    if not before or not after:
        return not before and not after   # nothing to scale either way
    scaled, first = before[-1] / ratio, after[0]
    return max(scaled, first) / max(min(scaled, first), 1e-12) <= 2.5


def _day(stamp: Any) -> date:
    text = str(stamp).replace("-", "")
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def clip_bad_prints(
    bars: pd.DataFrame, low_ratio: float = 0.5, high_ratio: float = 1.5
) -> tuple[pd.DataFrame, list[str]]:
    """Bars with isolated bad prints in their high or low brought back to the body.

    IBKR's long weekly histories carry the odd bad tick in a bar's extremes -- a
    low of 0.01 under a week that opened at 7.65 and closed at 7.87, a high of
    10,000 above one that never left 25. The open and the close are fine; only
    the extreme is not. A stop checked against such a low fires, and fills at a
    price that never traded. A low below ``low_ratio`` times the lower of the
    open and close, or a high above ``high_ratio`` times the higher, is set to
    that body edge. Genuine crash weeks stay: the worst large-cap weeks of 2008
    and March 2020 had lows above 0.55 of the body.

    Returns:
        The repaired bars and one line per bar changed, for the record.
    """
    if not 0.0 < low_ratio < 1.0 < high_ratio:
        raise ContractViolation(
            f"low_ratio must be in (0, 1) and high_ratio above 1; got {low_ratio}, {high_ratio}")
    out = bars.copy()
    body_low = out[["open", "close"]].min(axis=1)
    body_high = out[["open", "close"]].max(axis=1)
    low_bad = out["low"] < low_ratio * body_low
    high_bad = out["high"] > high_ratio * body_high
    changes = [
        f"{pd.Timestamp(stamp).date()}: low {out.at[stamp, 'low']:g} -> {body_low[stamp]:g}"
        for stamp in out.index[low_bad]
    ] + [
        f"{pd.Timestamp(stamp).date()}: high {out.at[stamp, 'high']:g} -> {body_high[stamp]:g}"
        for stamp in out.index[high_bad]
    ]
    out.loc[low_bad, "low"] = body_low[low_bad]
    out.loc[high_bad, "high"] = body_high[high_bad]
    return out, changes


def write_cache_csv(
    symbol: str, bars: pd.DataFrame, root: Path, frequency: str = "weekly"
) -> Path:
    """Write one instrument's bars into the local cache.

    Overwrites. The cache is a snapshot of what the vendor currently says, and
    the bitemporal store — not this file — is where the history of what was said
    when is kept.

    Args:
        symbol: Ticker, used as the filename.
        bars: Frame from :func:`bars_from_connector`.
        root: The ``data/ibkr_cache`` directory.
        frequency: ``weekly`` or ``daily``.

    Returns:
        The path written.
    """
    if frequency not in FREQUENCIES:
        raise ContractViolation(f"frequency must be one of {FREQUENCIES}; got {frequency!r}")
    if not symbol.isupper() or not symbol.isalnum():
        raise ContractViolation(
            f"symbol must be an uppercase alphanumeric ticker; got {symbol!r}"
        )
    target = root / frequency / f"{symbol}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out = bars.copy()
    stamps = pd.DatetimeIndex(out.index)
    if frequency in INTRADAY_FREQUENCIES:
        # Intraday labels keep their time, in UTC: a date alone would give every
        # bar of a session the same label.
        stamps = stamps.tz_localize("UTC") if stamps.tz is None else stamps.tz_convert("UTC")
        out.index = [t.isoformat() for t in stamps]
    else:
        out.index = stamps.strftime("%Y-%m-%d")
    out.index.name = "timestamp"
    out[list(CACHE_COLUMNS)].to_csv(target)
    return target


def cache_inventory(root: Path, frequency: str = "weekly") -> pd.DataFrame:
    """One row per cached instrument: bars and date range.

    Useful before and after expanding the universe, because the most common
    failure is a file that wrote but holds far less history than the others.
    """
    rows: list[dict[str, Any]] = []
    directory = root / frequency
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        rows.append(
            {
                "symbol": path.stem,
                "bars": len(frame),
                "first": frame["timestamp"].min().date(),
                "last": frame["timestamp"].max().date(),
            }
        )
    return pd.DataFrame(rows)


def check_coverage(inventory: pd.DataFrame, minimum_bars: int = 27) -> Sequence[str]:
    """Instruments with too little history to be selectable.

    They are not an error — a recent listing genuinely has little history, and
    the point-in-time universe handles that correctly. They are worth naming so
    that a symbol which silently fetched only a few bars is noticed.
    """
    if inventory.empty:
        return ()
    return tuple(inventory.loc[inventory["bars"] < minimum_bars, "symbol"])
