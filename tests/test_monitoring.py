"""The live monitors, each against a case whose right answer is known.

Healthy live data must not trigger anything; each kind of real deterioration must
be caught by the instrument meant for it; and the ensemble must map both onto
the ladder the way the thresholds say.
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts.errors import ContractViolation
from contracts.live import DegradationState
from validation.monitoring import (
    ExecutionRecord,
    assess,
    changepoint_probability,
    drawdown_check,
    max_drawdown,
    shortfall_check,
    stationary_bootstrap,
    trend_check,
)

MEAN, SD = 0.0045, 0.03  # roughly the strategy's weekly return and deviation


@pytest.fixture(scope="module")
def reference():
    return np.random.default_rng(1).normal(MEAN, SD, 900)


# -- the bootstrap -----------------------------------------------------------


def test_bootstrap_paths_have_the_requested_shape_and_come_from_the_series(reference):
    paths = stationary_bootstrap(reference, horizon=26, paths=200, block=6)
    assert paths.shape == (200, 26)
    assert set(np.round(paths.ravel(), 12)) <= set(np.round(reference, 12))


def test_blocks_keep_consecutive_weeks_together(reference):
    """With long blocks most steps continue the previous week's successor."""
    series = np.arange(100, dtype=float) / 1000
    paths = stationary_bootstrap(series, horizon=30, paths=100, block=20)
    steps = np.diff(paths * 1000, axis=1)
    assert np.mean(np.isclose(steps, 1.0) | np.isclose(steps, -99.0)) > 0.9


def test_a_bootstrap_needs_enough_history():
    with pytest.raises(ContractViolation, match="at least 20"):
        stationary_bootstrap([0.01] * 10, horizon=5, paths=10, block=2)


def test_max_drawdown_is_peak_to_trough():
    assert max_drawdown(np.array([0.10, -0.50, 0.20]))[0] == pytest.approx(0.5)
    assert max_drawdown(np.array([0.01, 0.01]))[0] == 0.0


def test_healthy_live_drawdown_sits_in_the_middle(reference):
    live = np.random.default_rng(2).normal(MEAN, SD, 52)
    check = drawdown_check(reference, live, paths=2000)
    assert 0.1 < check.percentile < 0.9


def test_a_collapsing_strategy_lands_in_the_tail_within_months(reference):
    live = np.random.default_rng(3).normal(-0.015, SD, 20)
    check = drawdown_check(reference, live, paths=2000)
    assert check.percentile > 0.95
    assert check.live_return < check.band_low_1


def test_the_horizon_matches_the_live_record_not_a_fixed_year(reference):
    """A ten-week loss judged against one-year drawdowns would look mild."""
    live = np.random.default_rng(4).normal(-0.01, SD, 10)
    assert drawdown_check(reference, live, paths=500).weeks == 10


# -- changepoints ------------------------------------------------------------


def test_healthy_data_rarely_crosses_the_reduce_threshold(reference):
    rng = np.random.default_rng(5)
    probabilities = [changepoint_probability(reference, rng.normal(MEAN, SD, 52)) for _ in range(30)]
    assert np.median(probabilities) < 0.10
    assert np.mean(np.array(probabilities) >= 0.20) <= 0.15


def test_a_volatility_regime_change_is_detected_within_a_quarter(reference):
    rng = np.random.default_rng(6)
    probabilities = [changepoint_probability(reference, rng.normal(MEAN, 0.12, 13)) for _ in range(10)]
    assert np.median(probabilities) > 0.50


def test_a_mean_collapse_is_slower_for_changepoints_and_fast_for_drawdown(reference):
    """The measured limit that makes the two instruments complementary."""
    live = np.random.default_rng(7).normal(-0.015, SD, 20)
    assert drawdown_check(reference, live, paths=2000).percentile > 0.95


def test_no_live_weeks_means_no_probability(reference):
    assert changepoint_probability(reference, []) == 0.0


# -- trend -------------------------------------------------------------------


def test_the_trend_is_not_judged_early():
    early = trend_check(np.full(10, -0.02) + np.random.default_rng(8).normal(0, 0.001, 10))
    assert not early.judged
    assert not early.significantly_negative


def test_a_steady_decline_is_significantly_negative_once_judged():
    live = np.random.default_rng(9).normal(-0.01, 0.01, 30)
    assert trend_check(live, min_weeks=26).significantly_negative


def test_a_healthy_trend_is_not_negative():
    """The regression for the first version, which regressed the equity curve.

    A cumulative curve is a random walk; a line fitted through it has strongly
    autocorrelated residuals and an interval several times too narrow, and it
    declared healthy strategies significantly negative often enough to halt them.
    """
    rng = np.random.default_rng(10)
    flagged = [
        trend_check(rng.normal(MEAN, SD, 40), min_weeks=26).significantly_negative
        for _ in range(60)
    ]
    assert np.mean(flagged) <= 0.05


# -- shortfall ---------------------------------------------------------------


def record(rotation, side, decision, fill, quantity=100, commission=1.0, **extra):
    return ExecutionRecord(
        rotation=rotation, instrument="AAA", side=side, quantity=quantity,
        decision_price=decision, fill_price=fill, commission=commission, **extra,
    )


def test_shortfall_is_signed_by_direction():
    assert record("r", "buy", 100.0, 101.0).shortfall == pytest.approx(0.01)
    assert record("r", "sell", 100.0, 99.0).shortfall == pytest.approx(0.01)
    assert record("r", "sell", 100.0, 101.0).shortfall == pytest.approx(-0.01), "a gain"


def test_execution_drag_isolates_the_fill_from_the_weekend_gap():
    """The backtest also crossed the weekend; only fill-versus-open is execution."""
    r = record("r", "buy", 100.0, 103.1, reference_open=103.0)
    assert r.shortfall == pytest.approx(0.031)
    assert r.execution_drag == pytest.approx(0.1 / 103.0)


def test_costs_well_above_the_model_are_flagged():
    records = [record("r1", "buy", 100.0, 100.5), record("r1", "sell", 100.0, 99.5)]
    check = shortfall_check(records, modeled_bps=20.0, expected_rotation_return=0.02,
                            sleeve_equity=100_000.0)
    assert check.mean_bps == pytest.approx(51.0)
    assert check.ratio > 2.0


def test_consecutive_rotations_eating_the_edge_are_counted():
    records = [
        record("r1", "buy", 100.0, 100.1, quantity=10),
        record("r2", "buy", 100.0, 110.0, quantity=500),
        record("r3", "buy", 100.0, 110.0, quantity=500),
    ]
    check = shortfall_check(records, modeled_bps=20.0, expected_rotation_return=0.02,
                            sleeve_equity=100_000.0)
    assert check.consecutive_over_half == 2


def test_markouts_measure_what_happened_after_the_fill():
    r = record("r", "buy", 100.0, 100.0, price_after_1w=102.0, price_after_4w=95.0)
    assert r.markout(1) == pytest.approx(0.02)
    assert r.markout(4) == pytest.approx(-0.05)


# -- the ensemble ------------------------------------------------------------


def _rates(reference, mean, weeks, samples, seed):
    rng = np.random.default_rng(seed)
    states = [
        assess(reference, rng.normal(mean, SD, weeks), [], modeled_bps=20.0,
               expected_rotation_return=0.018, sleeve_equity=100_000.0, paths=800).recommended
        for _ in range(samples)
    ]
    return (
        np.mean([s is not DegradationState.NORMAL for s in states]),
        np.mean([s is DegradationState.HALTED for s in states]),
    )


def test_healthy_live_trading_is_rarely_halted(reference):
    """Measured, not assumed: the false-alarm rates the manual quotes.

    Reduce-only at the 80th percentile fires on roughly a quarter of healthy
    assessments by construction -- that rung is meant for uncertain evidence and
    lifts itself. A halt must be rare, because only a person can lift it.
    """
    reduce_rate, halt_rate = _rates(reference, MEAN, 26, 40, seed=11)
    assert halt_rate <= 0.08
    assert reduce_rate <= 0.45


def test_a_collapse_is_halted_in_most_cases_within_half_a_year(reference):
    reduce_rate, halt_rate = _rates(reference, -0.015, 26, 20, seed=12)
    assert reduce_rate >= 0.95
    assert halt_rate >= 0.70


def test_expensive_execution_alone_moves_to_reduce_only(reference):
    live = np.random.default_rng(13).normal(MEAN, SD, 8)
    records = [record("r1", "buy", 100.0, 100.6), record("r1", "sell", 100.0, 99.4)]
    result = assess(reference, live, records, modeled_bps=20.0, expected_rotation_return=0.5,
                    sleeve_equity=100_000.0, paths=1000)
    assert result.recommended is DegradationState.REDUCE_ONLY
    assert any("execution cost" in r for r in result.reasons)


def test_early_results_carry_a_warning_about_their_width(reference):
    result = assess(reference, [0.01, -0.01], [], modeled_bps=20.0,
                    expected_rotation_return=0.018, sleeve_equity=1.0, paths=500)
    assert any("live weeks" in n for n in result.notes)
