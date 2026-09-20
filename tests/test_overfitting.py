"""CPCV, the DSR and the PBO.

Each statistic is checked against a case where the right answer is known by
construction: pure noise must fail, a genuinely strong signal must pass, and the
leakage controls must actually remove the leaked observations.
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts.errors import ContractViolation
from validation.cpcv import (
    block_sharpes,
    contiguous_runs,
    purged_splits,
    split_count,
)
from validation.overfitting import (
    deflated_sharpe,
    effective_trials,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    trial_sharpe_dispersion,
)

# -- CPCV --------------------------------------------------------------------


def test_the_expert_configuration_gives_the_expected_shape():
    """S=10 over ~900 weekly bars: 252 combinations of 90-bar blocks."""
    assert split_count(blocks=10) == 252
    splits = list(purged_splits(900, holding_bars=4, blocks=10, embargo_bars=9))
    assert len(splits) == 252
    assert all(len(s.test) == 450 for s in splits)


def test_train_and_test_never_share_an_observation():
    for split in purged_splits(200, holding_bars=4, blocks=6, embargo_bars=5):
        assert split.is_clean()
        assert np.intersect1d(split.train, split.test).size == 0


def test_purging_removes_exactly_the_overlapping_observations():
    """A position opened within the holding period is still open in the test."""
    holding = 4
    (split,) = list(
        purged_splits(100, holding_bars=holding, blocks=4, test_blocks=1, embargo_bars=0)
    )[1:2]
    start = int(split.test[0])
    assert set(split.purged.tolist()) == set(range(start - holding, start))
    assert all(index not in split.train for index in range(start - holding, start))


def test_the_embargo_quarantines_the_bars_after_a_test_block():
    embargo = 7
    splits = list(
        purged_splits(200, holding_bars=1, blocks=4, test_blocks=1, embargo_bars=embargo)
    )
    split = splits[0]
    end = int(split.test[-1])
    assert set(split.embargoed.tolist()) == set(range(end + 1, end + 1 + embargo))


def test_adjacent_test_blocks_are_one_run_not_two():
    """Purge and embargo apply at the edges of a run, not of every block."""
    runs = contiguous_runs(np.array([0, 1, 2, 7, 8, 20]))
    assert [r.tolist() for r in runs] == [[0, 1, 2], [7, 8], [20]]


def test_a_bigger_embargo_costs_more_training_data():
    small = list(purged_splits(300, holding_bars=4, blocks=6, embargo_bars=2))
    large = list(purged_splits(300, holding_bars=4, blocks=6, embargo_bars=20))
    assert sum(s.dropped for s in large) > sum(s.dropped for s in small)
    assert all(
        len(wide.train) <= len(narrow.train)
        for narrow, wide in zip(small, large, strict=True)
    )


def test_impossible_configurations_are_refused():
    with pytest.raises(ContractViolation, match="even"):
        list(purged_splits(200, holding_bars=1, blocks=5))
    with pytest.raises(ContractViolation, match="combinatorial"):
        list(purged_splits(200, holding_bars=1, blocks=2))
    with pytest.raises(ContractViolation, match="cannot make"):
        list(purged_splits(10, holding_bars=1, blocks=10))


def test_block_sharpes_returns_a_distribution_not_a_number():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.02, 400)
    splits = list(purged_splits(400, holding_bars=4, blocks=6, embargo_bars=4))
    values = block_sharpes(returns, splits)
    assert values.shape == (len(splits),)
    assert values.std() > 0, "the spread is the output"


# -- effective trials --------------------------------------------------------


def test_identical_trials_count_as_one():
    base = np.random.default_rng(2).normal(0, 0.02, 300)
    matrix = np.column_stack([base] * 8)
    assert effective_trials(matrix) == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_trials_count_as_all_of_them():
    rng = np.random.default_rng(3)
    matrix = rng.normal(0, 0.02, (4000, 6))
    assert effective_trials(matrix) == pytest.approx(6.0, rel=0.05)


def test_a_parameter_sweep_counts_as_far_fewer_than_its_trials():
    """Twenty lookbacks around one idea are not twenty independent attempts."""
    rng = np.random.default_rng(4)
    base = rng.normal(0.001, 0.02, 500)
    matrix = np.column_stack([base + rng.normal(0, 0.002, 500) for _ in range(20)])
    effective = effective_trials(matrix)
    assert 1.0 < effective < 6.0, effective


def test_both_methods_agree_on_the_extremes():
    rng = np.random.default_rng(5)
    matrix = rng.normal(0, 1, (3000, 5))
    entropy = effective_trials(matrix, method="entropy")
    herfindahl = effective_trials(matrix, method="herfindahl")
    assert entropy == pytest.approx(herfindahl, rel=0.1)
    with pytest.raises(ContractViolation, match="unknown method"):
        effective_trials(matrix, method="vibes")


# -- the deflated Sharpe -----------------------------------------------------


def test_the_luck_threshold_rises_with_the_number_of_trials():
    dispersion = 0.05
    assert expected_max_sharpe(1, dispersion) == 0.0
    thresholds = [expected_max_sharpe(n, dispersion) for n in (2, 10, 100, 1000)]
    assert thresholds == sorted(thresholds)
    # sqrt(2 ln N) growth: a hundredfold more trials is roughly double the bar.
    assert expected_max_sharpe(1000, dispersion) / expected_max_sharpe(
        10, dispersion
    ) == pytest.approx(2.0, abs=0.4)


def test_the_threshold_scales_with_how_scattered_the_trials_were():
    """Bailey and Lopez de Prado eq. (1): the bracket multiplies std(SR_i).

    Without this factor the threshold is a bare normal quantile -- about 2.3 for
    fifty trials -- which no weekly per-period Sharpe reaches, so every result
    would be rejected while the statistic looked merely strict.
    """
    tight = expected_max_sharpe(50, 0.01)
    scattered = expected_max_sharpe(50, 0.10)
    assert scattered == pytest.approx(10 * tight)
    assert tight < 0.1, "a tight cluster of trials is weak evidence of luck"
    assert expected_max_sharpe(50, 0.0) == 0.0


def test_dispersion_is_measured_from_the_trials_that_were_run():
    rng = np.random.default_rng(20)
    base = rng.normal(0.001, 0.02, 400)
    alike = np.column_stack([base + rng.normal(0, 0.0005, 400) for _ in range(10)])
    varied = np.column_stack([rng.normal(m, 0.02, 400) for m in np.linspace(-0.004, 0.004, 10)])
    assert trial_sharpe_dispersion(alike) < trial_sharpe_dispersion(varied)
    # One trial has no dispersion; the fallback is the standard error of a
    # single Sharpe estimate.
    single = trial_sharpe_dispersion(base.reshape(-1, 1))
    assert single == pytest.approx(1 / np.sqrt(400))


def test_pure_noise_does_not_survive_being_deflated():
    """The case the statistic exists for: nothing there, many things tried."""
    rng = np.random.default_rng(6)
    trials = [rng.normal(0, 0.02, 400) for _ in range(50)]
    matrix = np.column_stack(trials)
    best = max(trials, key=lambda r: r.mean() / r.std(ddof=1))
    result = deflated_sharpe(
        best, n_effective=effective_trials(matrix),
        trial_sharpe_std=trial_sharpe_dispersion(matrix),
    )
    assert result.dsr < 0.95, result


def test_a_strong_signal_survives_a_large_trial_count():
    rng = np.random.default_rng(7)
    strong = rng.normal(0.004, 0.01, 900)  # per-period Sharpe around 0.4
    others = np.column_stack([rng.normal(0, 0.02, 900) for _ in range(49)] + [strong])
    result = deflated_sharpe(
        strong, n_effective=50, trial_sharpe_std=trial_sharpe_dispersion(others)
    )
    assert result.dsr > 0.95, result
    assert result.clears_the_luck_threshold


def test_more_trials_deflate_the_same_result_further():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.002, 0.02, 600)
    few = deflated_sharpe(returns, n_effective=2, trial_sharpe_std=0.04)
    many = deflated_sharpe(returns, n_effective=500, trial_sharpe_std=0.04)
    assert few.dsr > many.dsr
    assert few.observed_sharpe == many.observed_sharpe, "the Sharpe did not change"
    assert many.expected_max > few.expected_max, "the bar did"


def test_negative_skew_is_penalised():
    """Small steady gains with occasional large losses: the option-seller shape."""
    rng = np.random.default_rng(9)
    n = 800
    skewed = np.full(n, 0.0045)
    skewed[::40] = -0.09  # rare, large losses
    # A symmetric series built to have exactly the same mean and deviation, so
    # the only thing that differs between them is the shape of the tail.
    symmetric = rng.normal(0, 1, n)
    symmetric = (symmetric - symmetric.mean()) / symmetric.std(ddof=1)
    symmetric = symmetric * skewed.std(ddof=1) + skewed.mean()

    balanced = deflated_sharpe(symmetric, n_effective=10, trial_sharpe_std=0.04)
    lopsided = deflated_sharpe(skewed, n_effective=10, trial_sharpe_std=0.04)

    assert lopsided.skew < -1.0
    assert abs(balanced.skew) < 0.3
    assert lopsided.observed_sharpe == pytest.approx(balanced.observed_sharpe, rel=1e-6), (
        "identical Sharpe by construction"
    )
    assert lopsided.z < balanced.z, "the same Sharpe is weaker evidence with a fat left tail"


def test_the_dsr_is_computed_in_the_return_frequency():
    """Annualising the Sharpe but not the threshold would not be the DSR.

    The expected-maximum term and the non-normality correction are both in the
    return's own period, so the observed Sharpe must be too. Scaling the whole
    series leaves the Sharpe -- and therefore the statistic -- unchanged, which
    an annualising implementation would not do.
    """
    rng = np.random.default_rng(10)
    returns = rng.normal(0.002, 0.02, 500)
    result = deflated_sharpe(returns, n_effective=20, trial_sharpe_std=0.04)
    assert result.observed_sharpe == pytest.approx(
        returns.mean() / returns.std(ddof=1), rel=1e-9
    )
    assert abs(result.observed_sharpe) < 1.0, "per-period, not annualised"


def test_a_series_with_no_variance_has_no_sharpe_to_deflate():
    with pytest.raises(ContractViolation, match="constant"):
        deflated_sharpe([0.01] * 100, n_effective=2, trial_sharpe_std=0.04)
    with pytest.raises(ContractViolation, match="three returns"):
        deflated_sharpe([0.01, 0.02], n_effective=2, trial_sharpe_std=0.04)


# -- the PBO -----------------------------------------------------------------


def test_selecting_among_pure_noise_is_a_coin_toss_at_best():
    """With no signal anywhere, the in-sample winner is random out-of-sample."""
    rng = np.random.default_rng(11)
    matrix = rng.normal(0, 0.02, (600, 12))
    result = probability_of_backtest_overfitting(matrix, blocks=8)
    assert result.pbo > 0.35, result.pbo
    assert result.combinations == 70


def test_selecting_a_genuinely_better_strategy_is_reliable():
    rng = np.random.default_rng(12)
    matrix = rng.normal(0.0, 0.02, (600, 10))
    matrix[:, 3] = rng.normal(0.006, 0.02, 600)  # one real edge
    result = probability_of_backtest_overfitting(matrix, blocks=8)
    assert result.pbo < 0.10, result.pbo
    assert result.verdict == "pass"
    assert result.median_relative_rank > 0.5


def test_the_verdict_names_the_three_regimes():
    rng = np.random.default_rng(13)
    matrix = rng.normal(0, 0.02, (400, 8))
    result = probability_of_backtest_overfitting(matrix, blocks=6)
    assert result.verdict in (
        "pass",
        "inconclusive: selection is not reliably better than chance",
        "discard: selection is worse than chance",
    )


def test_one_trial_is_not_a_selection():
    matrix = np.random.default_rng(14).normal(0, 0.02, (400, 1))
    with pytest.raises(ContractViolation, match="no selection"):
        probability_of_backtest_overfitting(matrix)
