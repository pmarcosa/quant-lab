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
from pathlib import Path
from typing import Any

import pandas as pd

from contracts.errors import ContractViolation

CACHE_COLUMNS = ("open", "high", "low", "close", "volume")

#: What the ingest reads. Anything else is a different dataset, not a variant.
FREQUENCIES = ("weekly", "daily")


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


def write_cache_csv(
    symbol: str, bars: pd.DataFrame, root: Path, frequency: str = "weekly"
) -> Path:
    """Write one instrument's bars into the committed cache.

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
    out.index = out.index.strftime("%Y-%m-%d")
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
