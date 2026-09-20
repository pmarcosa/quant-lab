"""The precomputed-indicator path, pinned against the honest one.

Precomputing indicators over all of history and slicing at the decision time is
a fifteen-fold speedup and a plausible way to introduce look-ahead. It is sound
only because every indicator at a bar uses that bar and earlier, which is tested
separately; this file checks that the two paths actually agree, on the real data,
rather than trusting the argument.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from contracts.identifiers import InstrumentId
from contracts.temporal import BarInterval
from strategies.momentum import MomentumParams, WeeklyMomentum, indicators

REAL_STORE = Path(__file__).resolve().parent.parent / "var" / "store"
needs_store = pytest.mark.skipif(
    not (REAL_STORE / "universe_weekly.csv").exists(),
    reason="needs the ingested IBKR store; run scripts/ingest_ibkr_cache.py",
)


@pytest.fixture(scope="module")
def market():
    from runtime.wiring import load_market

    return load_market(
        REAL_STORE,
        interval=BarInterval.WEEK,
        start=datetime(2010, 1, 1, tzinfo=timezone.utc),
    )


@needs_store
def test_the_precomputed_path_agrees_with_the_filtration_path(market):
    """Same targets, week by week, across seventeen years of real bars."""
    from runtime.research import precompute_indicators

    params = MomentumParams(rebalance_weeks=4)
    honest = WeeklyMomentum(params)
    fast = WeeklyMomentum(params, precomputed=precompute_indicators(market, params))

    schedule = list(market.schedule)
    # Every 37th week, so the sample spans the whole period and lands on both
    # rotation and non-rotation weeks.
    held: dict[InstrumentId, float] = {}
    checked = 0
    for moment in schedule[::37]:
        view = market.filtration_at(moment)
        slow_target = honest.target(view, held)
        fast_target = fast.target(view, held)
        assert slow_target.weights.keys() == fast_target.weights.keys(), moment
        for instrument, weight in slow_target.weights.items():
            assert fast_target.weights[instrument] == pytest.approx(weight, rel=1e-9), (
                f"{instrument} at {moment}"
            )
        held = dict(slow_target.weights)
        checked += 1
    assert checked > 20, "the comparison has to cover enough weeks to mean something"


@needs_store
def test_the_precomputed_frame_is_never_read_past_the_decision_time(market):
    """The frame holds the future; the strategy must not reach into it."""
    from runtime.research import precompute_indicators

    params = MomentumParams(rebalance_weeks=4)
    frames = precompute_indicators(market, params)
    instrument = next(iter(frames))
    frame = frames[instrument]
    assert frame.index[-1] > market.schedule[10], "the frame really does extend past"

    early = market.schedule[10]
    fast = WeeklyMomentum(params, precomputed=frames)
    visible = fast._indicators_at(market.filtration_at(early), instrument, 0)
    assert visible is not None
    assert visible.index.max() <= early


def test_slicing_a_full_frame_equals_computing_up_to_that_point():
    """The property the optimisation rests on, stated directly."""
    params = MomentumParams()
    index = pd.date_range("2010-01-04", periods=300, freq="W-MON", tz=timezone.utc)
    closes = pd.Series(range(100, 400), index=index, dtype=float)
    bars = pd.DataFrame(
        {"high": closes * 1.02, "low": closes * 0.98, "close": closes}, index=index
    )
    whole = indicators(bars, params)
    for cut in (120, 200, 299):
        upto = indicators(bars.iloc[: cut + 1], params)
        pd.testing.assert_series_equal(whole.iloc[cut], upto.iloc[-1], check_names=False)
