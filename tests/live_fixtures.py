"""A small synthetic market and a controllable clock, for the live cycle tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.identifiers import InstrumentId
from data.bitemporal import BitemporalStore
from data.ingest import to_observations, universe_from_store

#: Weekly growth rates: four names trending up at different speeds, one falling.
GROWTH = {"AAA": 0.020, "BBB": 0.015, "CCC": 0.008, "DDD": 0.004, "EEE": -0.01}
LAST_LABEL = datetime(2026, 9, 14)  # the Monday label of the latest complete week


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> datetime:
        self.now = self.now + timedelta(**kwargs)
        return self.now


def weekly_frame(growth: float, weeks: int, last_label: datetime, start: float = 50.0):
    labels = pd.date_range(end=last_label, periods=weeks, freq="W-MON")
    rng = np.random.default_rng(abs(hash(growth)) % 2**32)
    closes = start * np.cumprod(1 + growth + rng.normal(0, 0.002, weeks))
    opens = np.concatenate([[closes[0]], closes[:-1]]) * 1.001
    return pd.DataFrame(
        {
            "open": opens, "high": np.maximum(opens, closes) * 1.01,
            "low": np.minimum(opens, closes) * 0.99, "close": closes,
            "volume": np.full(weeks, 1e6),
        },
        index=labels,
    )


def build_market(root: Path, weeks: int = 80) -> dict[str, pd.DataFrame]:
    """Write a weekly store and its universe under ``root``."""
    store = BitemporalStore(root, "bars_1week")
    frames = {}
    for symbol, growth in GROWTH.items():
        frame = weekly_frame(growth, weeks, LAST_LABEL)
        frames[symbol] = frame
        store.append(InstrumentId(symbol), to_observations(frame, week_ending=True))
    universe = universe_from_store(store, still_trading_after=datetime(2026, 1, 1, tzinfo=timezone.utc))
    universe.to_csv(root / "universe_weekly.csv", derived=True)
    return frames


def append_week(root: Path, frames: dict[str, pd.DataFrame], known_at: datetime, drift=None):
    """Add the next week to every instrument, recorded as known at ``known_at``."""
    store = BitemporalStore(root, "bars_1week")
    for symbol, frame in frames.items():
        last = frame.iloc[-1]
        growth = (drift or {}).get(symbol, GROWTH[symbol])
        close = float(last["close"]) * (1 + growth)
        label = frame.index[-1] + pd.Timedelta(weeks=1)
        row = pd.DataFrame(
            {"open": [float(last["close"]) * 1.001], "high": [max(close, last["close"]) * 1.01],
             "low": [min(close, last["close"]) * 0.99], "close": [close], "volume": [1e6]},
            index=[label],
        )
        frames[symbol] = pd.concat([frame, row])
        store.append(InstrumentId(symbol), to_observations(row, week_ending=True, available_at=known_at))
    return frames
