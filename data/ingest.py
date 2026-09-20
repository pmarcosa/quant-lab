"""Loading price files into the bitemporal store.

A vendor CSV carries one timestamp per bar. Turning it into two — when the bar
happened, and when it could first have been acted on — is a judgement that has to
be made somewhere, and making it here, once and explicitly, is better than each
caller assuming zero.

For US equity sessions the bar's ``event_time`` is the session close rather than
the file's midnight date, because a strategy deciding on Friday evening and one
deciding at Friday midnight see different worlds. ``available_time`` adds a short
publication lag for the print to settle. How long a strategy must then wait before
acting is a separate matter, carried by ``FiltrationSpec.observation_lag_bars``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from data.bitemporal import BitemporalStore
from data.universe import PointInTimeUniverse, derive_memberships

#: US equity regular session close, in UTC. Ignores the DST shift to 21:00; at
#: weekly and daily resolution that hour never changes which bars are knowable.
US_SESSION_CLOSE = time(20, 0, tzinfo=timezone.utc)

#: How long after the close a bar is treated as final.
PUBLICATION_LAG = timedelta(minutes=15)


def load_price_csv(path: Path) -> pd.DataFrame:
    """Read a price CSV indexed by timestamp.

    Args:
        path: File with a ``timestamp`` column plus OHLCV payload.

    Raises:
        ContractViolation: If the file has no ``timestamp`` column.
    """
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ContractViolation(f"{path.name} has no 'timestamp' column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.set_index("timestamp").sort_index()


def to_observations(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach event and availability timestamps to session bars."""
    dates = pd.to_datetime(bars.index)
    closes = pd.to_datetime(
        [
            datetime.combine(d.date(), US_SESSION_CLOSE.replace(tzinfo=None), tzinfo=timezone.utc)
            for d in dates
        ],
        utc=True,
    )
    frame = bars.reset_index(drop=True)
    frame.insert(0, "event_time", closes)
    frame.insert(1, "available_time", closes + PUBLICATION_LAG)
    return frame


def ingest_directory(
    source: Path,
    store: BitemporalStore,
    drop_last_bar: bool = False,
    symbols: Sequence[str] | None = None,
) -> dict[str, int]:
    """Load every price CSV in a directory into the store.

    Args:
        source: Directory of ``<SYMBOL>.csv`` files.
        store: Destination.
        drop_last_bar: Drop the final row of each file. True for weekly data,
            where the last bar is the week still in progress and its high, low and
            close are not final — a partial bar is a live-versus-backtest
            discrepancy waiting to happen.
        symbols: Restrict to these symbols; all of them when omitted.

    Returns:
        Symbol to number of observations written.
    """
    if not source.is_dir():
        raise ContractViolation(f"no such directory: {source}")

    written: dict[str, int] = {}
    wanted = set(symbols) if symbols else None
    for path in sorted(source.glob("*.csv")):
        symbol = path.stem
        if wanted is not None and symbol not in wanted:
            continue
        bars = load_price_csv(path)
        if drop_last_bar and len(bars) > 1:
            bars = bars.iloc[:-1]
        if bars.empty:
            continue
        written[symbol] = store.append(InstrumentId(symbol), to_observations(bars))
    return written


def universe_from_store(
    store: BitemporalStore, still_trading_after: datetime
) -> PointInTimeUniverse:
    """Derive membership windows from the data the store actually holds.

    Approximate by construction — see ``data/universe.py`` for exactly which bias
    this removes and which it does not.
    """
    now = datetime.now(timezone.utc)
    spans = []
    for instrument in store.instruments():
        known = store.as_of(instrument, now)
        if known.empty:
            continue
        spans.append(
            (
                instrument,
                known.index[0].to_pydatetime(),
                known.index[-1].to_pydatetime(),
            )
        )
    return derive_memberships(spans, still_trading_after)
