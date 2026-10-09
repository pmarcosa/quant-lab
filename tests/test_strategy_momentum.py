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


@pytest.mark.parametrize("n", [10, 49])
def test_a_cap_of_exactly_equal_weight_terminates(n):
    """``weight_cap_mult=1.0`` is allowed, and makes the bounds only just feasible.

    n weights of 1/n then sum to a hair under one in floating point (ten 0.1s
    on Python 3.10, forty-nine 1/49s on every version), and the bracketing loop
    compared that sum with exactly one: it doubled the scale into infinity and
    never returned. A regression shows as a hang here, not a failure.
    """
    scores = {InstrumentId(f"n{i}"): 1.0 + i for i in range(n)}
    weights = capped_proportional(scores, floor=0.5 / n, cap=1.0 / n)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(w <= 1.0 / n + 1e-12 for w in weights.values())


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


def test_an_uptrend_that_is_still_gathering_pace_passes_the_entry_gate():
    params = MomentumParams()
    # A slow climb, then four weeks that carry more than 40% of the quarter.
    closes = np.concatenate([np.linspace(100, 190, 116), np.linspace(192, 200, 4)])
    table = indicators(series(closes), params)
    assert entry_ok(table.iloc[-1], params)
    assert exit_reason(table.iloc[-1], params) is None


def test_a_steady_climb_does_not_pass_the_pace_filter():
    """The filter compares total returns, so it asks for more than "not slowing".

    At a constant pace the last 4 weeks of 13 carry about 31% of the move. The
    filter wants 40%: a name has to be speeding up to be bought. Compared per
    week, as this was until 2026-10-09, the same climb passed with room to spare.
    """
    params = MomentumParams()
    row = indicators(series(np.linspace(100, 220, 120)), params).iloc[-1]
    assert row["ret_pace"] / row["ret_long"] == pytest.approx(4 / 13, abs=0.02)
    assert not entry_ok(row, params)
    assert entry_ok(row, MomentumParams(pace_ratio_min=0.25))
    assert exit_reason(row, params) is None, "failing to qualify is not a reason to sell"


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
    with pytest.raises(ContractViolation, match="0 for every name that passes"):
        MomentumParams(top_n=-1)
    assert MomentumParams(top_n=0).top_n == 0, "zero is 'no limit', not an error"


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


# -- what the book is split between: every name that passes, and the frozen ---------

RISING, FASTER, PLODDING, FALLING = (InstrumentId(s) for s in ("rise", "fast", "plod", "fall"))


def _gathering_pace(top: float) -> np.ndarray:
    """A climb whose last four weeks carry well over 40% of the quarter."""
    return np.concatenate([np.linspace(100, top * 0.94, 116), np.linspace(top * 0.955, top, 4)])


class _Bars:
    """A filtration over four made-up names, pinned at their last bar."""

    frames = {
        RISING: series(_gathering_pace(200)),
        FASTER: series(_gathering_pace(300)),
        PLODDING: series(np.linspace(100, 220, 120)),   # up, steady: fails the pace filter
        FALLING: series(np.linspace(220, 100, 120)),    # breaks its trend: an exit
    }

    @property
    def decision_time(self):
        return self.frames[RISING].index[-1].to_pydatetime()

    def universe(self, min_bars=1):
        return tuple(self.frames)

    def history(self, instrument, field, count):
        return self.frames[instrument][field]


def _target(held=None, **params):
    from contracts.targets import Holdings

    strategy = WeeklyMomentum(MomentumParams(rebalance_weeks=1, **params))
    return strategy.target(_Bars(), held if held is not None else Holdings())


def test_with_no_limit_every_name_that_passes_is_held():
    target = _target(top_n=0)
    assert set(target.weights) == {RISING, FASTER}
    assert sum(target.weights.values()) == pytest.approx(1.0)
    assert target.diagnostics["eligible"] == 2.0


def test_a_limit_keeps_the_best_trailing_return():
    target = _target(top_n=1)
    assert set(target.weights) == {FASTER}
    assert target.weights[FASTER] == pytest.approx(1.0)


def test_a_held_name_that_no_longer_qualifies_is_frozen_not_sold():
    """It fails the pace filter and has triggered no exit, so it stays as it is.

    The names that pass share what it leaves: 70% of the book, not all of it.
    """
    target = _target({PLODDING: 0.30}, top_n=0)
    assert target.weights[PLODDING] == 0.30, "exactly the weight it had"
    assert target.weights[RISING] + target.weights[FASTER] == pytest.approx(0.70)
    assert target.diagnostics["frozen"] == 1.0


def test_without_freezing_a_name_that_no_longer_qualifies_is_sold():
    target = _target({PLODDING: 0.30}, top_n=0, hold_unqualified=False)
    assert PLODDING not in target.weights
    assert sum(target.weights.values()) == pytest.approx(1.0)


def test_a_held_name_that_triggers_an_exit_is_sold_frozen_or_not():
    target = _target({FALLING: 0.30}, top_n=0)
    assert FALLING not in target.weights
    assert target.diagnostics["exits"] == 1.0
    assert sum(target.weights.values()) == pytest.approx(1.0), "its share is redeployed"


def test_a_held_name_that_passes_but_ranks_below_the_limit_is_frozen():
    target = _target({RISING: 0.40}, top_n=1)
    assert target.weights[RISING] == 0.40
    assert target.weights[FASTER] == pytest.approx(0.60)


def test_a_held_name_with_no_bar_this_week_is_left_alone():
    ghost = InstrumentId("ghost")
    target = _target({ghost: 0.25}, top_n=0)
    assert target.weights[ghost] == 0.25


def test_a_loser_that_is_still_losing_is_sold_on_its_cost():
    """Down on the quarter, but above its average and near its high: no other exit."""
    params = MomentumParams()
    closes = np.concatenate([np.full(100, 100.0), np.full(9, 95.0), [96.0, 97.0, 98.0, 99.0]])
    row = indicators(series(closes), params).iloc[-1]
    assert row["ret_long"] < 0
    assert exit_reason(row, params) is None, "price alone gives no reason to sell"
    assert exit_reason(row, params, gain=-0.25) == "cost_stop"
    assert exit_reason(row, params, gain=-0.19) is None, "not 20% under its cost"
    assert exit_reason(row, MomentumParams(cost_stop_loss=0.0), gain=-0.25) is None


def test_a_loss_on_cost_alone_does_not_sell_a_name_that_is_rising():
    """The cost is the holder's past. A name up on the quarter is kept."""
    params = MomentumParams()
    row = indicators(series(_gathering_pace(200)), params).iloc[-1]
    assert row["ret_long"] > 0
    assert exit_reason(row, params, gain=-0.40) is None


def test_the_four_week_calendar_is_the_live_accounts():
    """Rebalances on Monday 2026-09-21, decided on the close of Friday the 18th."""
    strategy = WeeklyMomentum(MomentumParams(rebalance_weeks=4))

    def decided(month, day):
        return strategy.rotates_at(datetime(2026, month, day, 21, 15, tzinfo=timezone.utc))

    assert decided(9, 18) and decided(10, 16) and decided(11, 13)
    assert not any(decided(m, d) for m, d in ((9, 11), (9, 25), (10, 2), (10, 9)))
    # A bar fetched on the Saturday belongs to the same week.
    assert strategy.rotates_at(datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc))


def test_a_frozen_name_can_be_made_to_fade(portfolio=None):
    """Half its weight at each rotation: capital leaves without one sale."""
    target = _target({PLODDING: 0.30}, top_n=0, freeze_fade=0.5)
    assert target.weights[PLODDING] == pytest.approx(0.15)
    assert target.weights[RISING] + target.weights[FASTER] == pytest.approx(0.85)


def test_a_limit_on_freezing_sells_a_name_that_has_not_qualified_lately():
    """The plodder never passed the filter, so one rotation of grace is not enough."""
    assert PLODDING in _target({PLODDING: 0.30}, top_n=0).weights
    limited = _target({PLODDING: 0.30}, top_n=0, freeze_rotations=1)
    assert PLODDING not in limited.weights
    assert sum(limited.weights.values()) == pytest.approx(1.0)


def test_a_limit_on_freezing_counts_earlier_rotations_not_today():
    """It passes now but ranks below the limit, and did not pass a rotation ago: sold.

    Four weeks ago the slower climber had not yet gathered pace, and passing
    today earns a place only among the best ``top_n``.
    """
    from contracts.targets import Holdings

    params = MomentumParams(rebalance_weeks=4, top_n=1, freeze_rotations=1)
    strategy = WeeklyMomentum(params)
    table = indicators(_Bars.frames[RISING], params)
    assert entry_ok(table.iloc[-1], params) and not entry_ok(table.iloc[-5], params)

    def moved(weeks: int) -> type:
        """The same bars, later by some weeks, so the last one can be a rotation."""
        later = {}
        for name, frame in _Bars.frames.items():
            later[name] = frame.set_axis(frame.index + pd.Timedelta(weeks=weeks))
        return type("Moved", (_Bars,), {"frames": later})

    bars = next(m() for m in map(moved, range(4)) if strategy.rotates_at(m().decision_time))
    target = strategy.target(bars, Holdings({RISING: 0.20}))
    assert RISING not in target.weights
    assert target.weights[FASTER] == pytest.approx(1.0)


def test_a_limit_on_freezing_keeps_a_name_that_qualified_at_the_last_rotation():
    """Qualified a week ago (the cadence here is weekly), stalled this week: kept."""
    from contracts.targets import Holdings

    stalled = InstrumentId("stall")
    closes = np.append(_gathering_pace(200), 196.0)

    class Stalled(_Bars):
        frames = {**_Bars.frames, stalled: series(closes)}

        @property
        def decision_time(self):
            return self.frames[stalled].index[-1].to_pydatetime()

    params = MomentumParams(rebalance_weeks=1, top_n=0, freeze_rotations=1)
    row_now = indicators(series(closes), params).iloc[-1]
    row_before = indicators(series(closes), params).iloc[-2]
    assert entry_ok(row_before, params) and not entry_ok(row_now, params)
    assert exit_reason(row_now, params) is None
    target = WeeklyMomentum(params).target(Stalled(), Holdings({stalled: 0.25}))
    assert target.weights[stalled] == 0.25


def test_impossible_freezing_settings_are_refused():
    with pytest.raises(ContractViolation):
        MomentumParams(freeze_fade=0.0)
    with pytest.raises(ContractViolation):
        MomentumParams(freeze_rotations=-1)
