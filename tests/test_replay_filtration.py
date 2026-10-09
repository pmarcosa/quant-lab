"""The cached filtration, pinned against the uncached one.

``ReplayFiltrations`` exists only to make a 900-step backtest fast. Speed work on
the component that enforces causality is exactly the place a guarantee gets lost,
so this checks the fast path against the slow one rather than against an
argument about why it must be equivalent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from contracts.errors import CausalityViolation
from contracts.identifiers import InstrumentId
from data.bitemporal import BitemporalStore
from data.filtration import ReplayFiltrations, StoreFiltration
from data.ingest import to_observations, universe_from_store
from tests.conftest import weekly_bars

REAL_STORE = Path(__file__).resolve().parent.parent / "var" / "store"


@pytest.fixture
def pair(tmp_path):
    """A store with two instruments of different ages, and its universe."""
    store = BitemporalStore(tmp_path, "bars_1week")
    store.append(InstrumentId("old"), to_observations(weekly_bars("2009-01-02", 600)))
    store.append(InstrumentId("new"), to_observations(weekly_bars("2019-04-05", 300)))
    horizon = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return store, universe_from_store(store, still_trading_after=horizon - timedelta(days=400))


def test_the_cache_cannot_widen_what_is_visible(pair):
    """Every instrument, every field, every moment: the two must agree exactly."""
    store, universe = pair
    horizon = datetime(2026, 1, 1, tzinfo=timezone.utc)
    replay = ReplayFiltrations(store, universe, horizon, min_bars=1)
    instruments = [InstrumentId("old"), InstrumentId("new")]

    moments = [
        datetime(year, month, 5, 22, tzinfo=timezone.utc)
        for year in (2010, 2015, 2019, 2022, 2025)
        for month in (3, 9)
    ]
    for moment in moments:
        slow = StoreFiltration(store, universe, moment, min_bars=1)
        fast = replay.at(moment)
        assert fast.decision_time == slow.decision_time
        assert fast.universe() == slow.universe()
        for instrument in instruments:
            for field in ("open", "high", "low", "close"):
                pd.testing.assert_series_equal(
                    fast.history(instrument, field, 10_000),
                    slow.history(instrument, field, 10_000),
                    check_names=False,
                )
            assert fast.is_available(instrument) == slow.is_available(instrument)
            assert fast.metadata(instrument) == slow.metadata(instrument)


def test_a_revision_after_the_decision_does_not_hide_the_original(tmp_path):
    """A dividend refresh re-appends every stored bar with a later available time.

    A decision before that refresh could see the original rows, and must still
    see them through the cache. Holding only the latest version made the whole
    history vanish from every earlier decision -- the instrument silently
    dropped out of the backtest -- while the uncached path saw it all.
    """
    store = BitemporalStore(tmp_path, "bars_1week")
    spy = InstrumentId("spy")
    bars = weekly_bars("2015-01-02", 300)
    store.append(spy, to_observations(bars))
    restated_at = datetime(2021, 1, 4, tzinfo=timezone.utc)
    restated = to_observations(bars.loc[bars.index < "2020-12-25"] * 0.98)
    restated["available_time"] = restated_at
    store.revise(spy, restated)
    universe = universe_from_store(store, still_trading_after=datetime(2020, 1, 1, tzinfo=timezone.utc))
    replay = ReplayFiltrations(store, universe, datetime(2021, 6, 1, tzinfo=timezone.utc))

    for moment in (
        datetime(2018, 3, 5, 22, tzinfo=timezone.utc),   # before the restatement
        datetime(2021, 3, 1, 22, tzinfo=timezone.utc),   # after it
    ):
        slow = StoreFiltration(store, universe, moment).history(spy, "close", 10_000)
        fast = replay.at(moment).history(spy, "close", 10_000)
        assert len(slow) > 100, "the check is vacuous if nothing is visible"
        pd.testing.assert_series_equal(fast, slow, check_names=False)

    before = replay.at(datetime(2018, 3, 5, 22, tzinfo=timezone.utc))
    assert before.history(spy, "close", 1).iloc[-1] == pytest.approx(
        float(bars.loc[:"2018-03-05", "close"].iloc[-1])
    ), "before the restatement, the original price"


def test_a_decision_past_the_load_horizon_is_refused(pair):
    """Serving a pinned view from a stale load would silently hide later rows."""
    store, universe = pair
    horizon = datetime(2020, 1, 1, tzinfo=timezone.utc)
    replay = ReplayFiltrations(store, universe, horizon)
    replay.at(horizon - timedelta(days=1))
    with pytest.raises(CausalityViolation, match="past the load horizon"):
        replay.at(horizon + timedelta(days=1))


@pytest.mark.skipif(
    not (REAL_STORE / "universe_weekly.csv").exists(),
    reason="needs the ingested IBKR store; run scripts/ingest_ibkr_cache.py",
)
def test_the_cache_matches_the_store_on_the_real_data():
    """The same check against 17 years of real bars, where the gaps are real."""
    from data.universe import PointInTimeUniverse

    store = BitemporalStore(REAL_STORE, "bars_1week")
    universe = PointInTimeUniverse.from_csv(REAL_STORE / "universe_weekly.csv")
    horizon = datetime(2026, 9, 30, tzinfo=timezone.utc)
    replay = ReplayFiltrations(store, universe, horizon, min_bars=27)

    for moment in (
        datetime(2011, 1, 3, 20, 15, tzinfo=timezone.utc),
        datetime(2016, 6, 6, 20, 15, tzinfo=timezone.utc),
        datetime(2021, 3, 1, 20, 15, tzinfo=timezone.utc),
    ):
        slow = StoreFiltration(store, universe, moment, min_bars=27)
        fast = replay.at(moment)
        assert fast.universe() == slow.universe()
        assert fast.universe(), "the check is vacuous if nothing is available"
        for instrument in fast.universe():
            pd.testing.assert_series_equal(
                fast.history(instrument, "close", 10_000),
                slow.history(instrument, "close", 10_000),
                check_names=False,
            )
