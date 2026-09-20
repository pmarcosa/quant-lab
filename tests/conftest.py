"""Shared fixtures."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.identifiers import InstrumentId, PortfolioId, TenantId  # noqa: E402
from data.bitemporal import BitemporalStore  # noqa: E402
from data.ingest import to_observations  # noqa: E402


@pytest.fixture
def tenant() -> TenantId:
    return TenantId("user")


@pytest.fixture
def portfolio(tenant: TenantId) -> PortfolioId:
    return PortfolioId(tenant, "ibkr-main")


@pytest.fixture
def store(tmp_path: Path) -> BitemporalStore:
    return BitemporalStore(tmp_path, "bars_1week")


def weekly_bars(start: str, periods: int, first_close: float = 100.0) -> pd.DataFrame:
    """A simple rising weekly series, indexed by session date."""
    index = pd.date_range(start, periods=periods, freq="W-FRI")
    closes = [first_close + i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1e6] * periods,
        },
        index=index,
    )


@pytest.fixture
def loaded_store(store: BitemporalStore) -> BitemporalStore:
    """Two instruments with different listing dates.

    ``old`` has traded since 2009. ``new`` only since 2019 — the shape that
    survivorship and listing-date bias hide.
    """
    store.append(InstrumentId("old"), to_observations(weekly_bars("2009-01-02", 900)))
    store.append(InstrumentId("new"), to_observations(weekly_bars("2019-04-05", 380)))
    return store


def at(year: int, month: int = 1, day: int = 1, hour: int = 22) -> datetime:
    """A decision time, after the US close by default."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc)
