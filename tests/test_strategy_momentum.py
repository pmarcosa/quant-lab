"""The momentum strategy and its sizing.

The sizing tests carry the regression for a defect that was live until
2026-09-20: a declared weight cap that did not hold.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from strategies.momentum import (
    MomentumParams,
    WeeklyMomentum,
    entry_ok,
    exit_reason,
    indicators,
)
from strategies.sizing import capped_proportional

A, B, C, D = (InstrumentId(s) for s in ("aaa", "bbb", "ccc", "ddd"))


def series(closes, highs=None, lows=None):
    index = pd.date_range("2018-01-01", periods=len(closes), freq="W-MON", tz=timezone.utc)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "high": closes * 1.01 if highs is None else highs,
            "low": closes * 0.99 if lows is None else lows,
            "close": closes,
        },
        index=index,
    )


# -- sizing: the regression --------------------------------------------------


def clamp_then_renormalise(scores, floor, cap):
    """The construction that shipped, kept so the defect stays reproducible."""
    total = sum(scores.values())
    w = {k: v / total for k, v in scores.items()}
    w = {k: min(max(v, floor), cap) for k, v in w.items()}
    scale = sum(w.values())
    return {k: v / scale for k, v in w.items()}


def test_the_cap_holds_when_dispersion_binds():
    """The defect. Fixed 2026-09-20; live in every backtest before it.

    Clamping the largest weight down leaves the book short of one, and the final
    division scales everything back up -- including the weight just clamped. The
    result still sums to one, which is why it passed review for months.
    """
    scores = {A: 5.0, B: 1.0, C: 1.0, D: 1.0}
    equal = 0.25
    floor, cap = 0.5 * equal, 2.0 * equal

    broken = clamp_then_renormalise(scores, floor, cap)
    assert broken[A] == pytest.approx(0.5714, abs=1e-4)
    assert broken[A] / equal == pytest.approx(2.29, abs=0.01), "2.29x a declared 2.0x cap"
    assert sum(broken.values()) == pytest.approx(1.0)

    fixed = capped_proportional(scores, floor=floor, cap=cap)
    assert fixed[A] == pytest.approx(cap)
    assert sum(fixed.values()) == pytest.approx(1.0)
    assert max(fixed.values()) <= cap + 1e-12


def test_the_floor_holds_when_it_binds():
    scores = {A: 1.0, B: 30.0, C: 30.0, D: 30.0}
    floor = 0.5 * 0.25
    weights = capped_proportional(scores, floor=floor, cap=2.0 * 0.25)
    assert weights[A] == pytest.approx(floor)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_names_away_from_a_bound_keep_their_exact_proportions():
    weights = capped_proportional({A: 40.0, B: 6.0, C: 5.0, D: 4.0}, floor=0.125, cap=0.5)
    assert weights[B] / weights[C] == pytest.approx(6.0 / 5.0)
    assert weights[C] / weights[D] == pytest.approx(5.0 / 4.0)


def test_bounds_hold_across_extreme_dispersion():
    rng = np.random.default_rng(20260920)
    for _ in range(300):
        n = int(rng.integers(2, 9))
        equal = 1.0 / n
        floor, cap = 0.5 * equal, 2.0 * equal
        # Log-normal over several orders of magnitude: far wider than real ATR
        # dispersion, so the bounds are tested where they actually bind.
        scores = {InstrumentId(f"s{i}"): float(np.exp(rng.normal(0, 3))) for i in range(n)}
        weights = capped_proportional(scores, floor=floor, cap=cap)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert min(weights.values()) >= floor - 1e-9
        assert max(weights.values()) <= cap + 1e-9


def test_impossible_bounds_are_refused_not_quietly_rescaled():
    with pytest.raises(ContractViolation, match="cap"):
        capped_proportional({A: 1.0, B: 1.0}, floor=0.0, cap=0.4)
    with pytest.raises(ContractViolation, match="floor"):
        capped_proportional({A: 1.0, B: 1.0, C: 1.0, D: 1.0}, floor=0.3, cap=1.0)
    with pytest.raises(ContractViolation, match="positive"):
        capped_proportional({A: 0.0, B: 1.0}, floor=0.0, cap=1.0)


def test_equal_scores_give_equal_weights():
    weights = capped_proportional({A: 7.0, B: 7.0, C: 7.0, D: 7.0}, floor=0.125, cap=0.5)
    assert all(w == pytest.approx(0.25) for w in weights.values())


# -- indicators --------------------------------------------------------------


def test_every_indicator_uses_only_its_own_bar_and_earlier():
    """Change the future and nothing earlier may move."""
    params = MomentumParams()
    base = series(np.linspace(100, 200, 120))
    altered = base.copy()
    altered.iloc[-5:] *= 3.0

    first = indicators(base, params).iloc[:-5]
    second = indicators(altered, params).iloc[:-5]
    pd.testing.assert_frame_equal(first, second)


def test_bars_missing_a_required_column_are_refused():
    with pytest.raises(ContractViolation, match="missing"):
        indicators(pd.DataFrame({"close": [1.0, 2.0]}), MomentumParams())


def test_a_clean_uptrend_passes_the_entry_gate():
    params = MomentumParams()
    table = indicators(series(np.linspace(100, 220, 120)), params)
    assert entry_ok(table.iloc[-1], params)
    assert exit_reason(table.iloc[-1], params) is None


def test_a_downtrend_fails_the_entry_gate():
    params = MomentumParams()
    table = indicators(series(np.linspace(220, 100, 120)), params)
    assert not entry_ok(table.iloc[-1], params)


def test_a_decelerating_advance_is_rejected_even_though_it_is_up():
    """The ranking cannot see that a move has already finished; the pace can."""
    params = MomentumParams()
    # A hundred weeks from 100 to 200, then six weeks that crawl to 201.
    closes = np.concatenate([np.linspace(100, 200, 100), np.linspace(200, 201, 6)])
    row = indicators(series(closes), params).iloc[-1]

    assert row["ret_long"] > 0, "still up over the lookback"
    assert row["close"] > row["sma"] > row["sma_slope_ref"], "still in an uptrend"
    # Everything the ranking and the trend filter look at says buy. Only the
    # pace filter sees that the advance has already been made.
    assert row["ret_pace"] / params.pace_weeks < (
        params.pace_ratio_min * row["ret_long"] / params.lookback_weeks
    )
    assert not entry_ok(row, params)


def test_a_drawdown_from_the_rolling_high_forces_an_exit():
    params = MomentumParams()
    closes = np.concatenate([np.linspace(100, 200, 100), np.linspace(200, 150, 20)])
    table = indicators(series(closes), params)
    assert exit_reason(table.iloc[-1], params) in ("trend_break", "below_high")


# -- parameters are identity -------------------------------------------------


def test_changing_a_parameter_changes_the_version():
    first = WeeklyMomentum(MomentumParams(lookback_weeks=13)).version
    second = WeeklyMomentum(MomentumParams(lookback_weeks=26)).version
    assert first.strategy == second.strategy
    assert first.params_hash != second.params_hash, (
        "two lookbacks are two strategies; evidence earned by one is not the other's"
    )
    assert WeeklyMomentum(MomentumParams(lookback_weeks=13)).version == first


def test_impossible_weight_bounds_are_refused_at_construction():
    with pytest.raises(ContractViolation, match="floor <= 1 <= cap"):
        MomentumParams(weight_cap_mult=0.5)
    with pytest.raises(ContractViolation, match="at least 1"):
        MomentumParams(top_n=0)


def test_the_strategy_carries_no_universe_of_its_own():
    """A list of instruments held by the strategy would be a list from today."""
    strategy = WeeklyMomentum()
    assert strategy.universe(datetime(2011, 1, 1, tzinfo=timezone.utc)) == ()


def test_state_reports_identity_not_history():
    strategy = WeeklyMomentum(MomentumParams(top_n=5))
    state = strategy.state()
    assert state["name"] == "weekly-momentum"
    assert state["top_n"] == 5
    assert strategy.state() == state, "the same instance reports the same thing twice"
