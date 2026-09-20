"""The funnel gates.

Each gate is fed a case whose correct verdict is known by construction. The
discard cases matter most: a gate that can only pass or fail cannot tell "not
proven yet" from "demonstrably does not work".
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts.errors import ContractViolation
from validation.funnel import (
    FunnelReport,
    GateResult,
    Verdict,
    cpcv_gate,
    deflated_sharpe_gate,
    monkey_test,
    pbo_gate,
    robustness_gate,
    walk_forward_gate,
)

RNG = np.random.default_rng(2026)


# -- gate 1 ------------------------------------------------------------------


def test_beating_the_controls_passes():
    controls = [RNG.normal(0.0005, 0.02, 400) for _ in range(60)]
    strong = RNG.normal(0.004, 0.02, 400)
    result = monkey_test(strong, controls)
    assert result.verdict is Verdict.PASS
    assert result.measured["percentile"] >= 0.95


def test_being_indistinguishable_from_the_controls_fails():
    controls = [RNG.normal(0.001, 0.02, 400) for _ in range(60)]
    ordinary = RNG.normal(0.001, 0.02, 400)
    assert monkey_test(ordinary, controls).verdict in (Verdict.FAIL, Verdict.DISCARD)


def test_losing_to_the_median_control_is_a_discard():
    """Worse than picking at random is not 'needs more work'."""
    controls = [RNG.normal(0.003, 0.02, 400) for _ in range(60)]
    weak = RNG.normal(-0.001, 0.02, 400)
    assert monkey_test(weak, controls).verdict is Verdict.DISCARD


def test_a_percentile_from_too_few_controls_is_refused():
    with pytest.raises(ContractViolation, match="at least 20"):
        monkey_test(RNG.normal(0, 1, 100), [RNG.normal(0, 1, 100) for _ in range(5)])


# -- gate 2 ------------------------------------------------------------------


def test_mostly_positive_folds_pass():
    folds = RNG.normal(0.25, 0.08, 252)
    assert cpcv_gate(folds).verdict is Verdict.PASS


def test_too_many_negative_folds_fail():
    folds = RNG.normal(0.05, 0.20, 252)
    result = cpcv_gate(folds)
    assert result.verdict is Verdict.FAIL
    assert result.measured["negative_fraction"] > 0.15


def test_a_bimodal_fold_distribution_is_a_discard_whatever_its_mean():
    """Excellent in one regime and useless in the other is not an average."""
    folds = np.concatenate([RNG.normal(-0.4, 0.05, 126), RNG.normal(1.2, 0.05, 126)])
    result = cpcv_gate(folds)
    assert result.verdict is Verdict.DISCARD
    assert "BIMODAL" in result.note
    assert np.median(folds) > 0, "its median looks fine, which is the point"


def test_a_distribution_needs_enough_folds_to_be_one():
    with pytest.raises(ContractViolation, match="ten folds"):
        cpcv_gate([0.1, 0.2, 0.3])


# -- gate 3 ------------------------------------------------------------------


def test_holding_up_out_of_sample_passes():
    inside = np.full(12, 0.20)
    outside = np.full(12, 0.16)
    result = walk_forward_gate(inside, outside)
    assert result.verdict is Verdict.PASS
    assert result.measured["wfe"] == pytest.approx(0.8)


def test_collapsing_out_of_sample_is_a_discard():
    inside = np.full(12, 0.30)
    outside = np.full(12, 0.03)
    assert walk_forward_gate(inside, outside).verdict is Verdict.DISCARD


def test_middling_efficiency_fails_without_discarding():
    result = walk_forward_gate(np.full(12, 0.30), np.full(12, 0.12))
    assert result.verdict is Verdict.FAIL
    assert result.measured["wfe"] == pytest.approx(0.4)


def test_nothing_profitable_in_sample_is_a_discard():
    result = walk_forward_gate(np.full(8, -0.05), np.full(8, 0.02))
    assert result.verdict is Verdict.DISCARD
    assert "nothing to carry forward" in result.note


# -- gate 4 ------------------------------------------------------------------


def test_a_high_dsr_passes_and_a_low_one_fails():
    assert deflated_sharpe_gate(0.99, 2.4, 0.30, 0.10, 40).verdict is Verdict.PASS
    assert deflated_sharpe_gate(0.60, 0.3, 0.12, 0.10, 40).verdict is Verdict.FAIL


def test_not_even_clearing_the_luck_threshold_is_a_discard():
    """The observed Sharpe is below what N worthless trials produce by chance."""
    result = deflated_sharpe_gate(0.02, -1.9, 0.05, 0.22, 200)
    assert result.verdict is Verdict.DISCARD


def test_pbo_above_a_half_is_a_discard():
    assert pbo_gate(0.62, 252).verdict is Verdict.DISCARD
    assert pbo_gate(0.28, 252).verdict is Verdict.FAIL
    assert pbo_gate(0.04, 252).verdict is Verdict.PASS


# -- gate 5 ------------------------------------------------------------------


def test_a_plateau_passes_and_a_needle_fails():
    plateau = RNG.normal(0.9, 0.1, 40)
    assert robustness_gate(1.0, plateau).verdict is Verdict.PASS

    needle = np.concatenate([np.full(3, 0.9), RNG.normal(0.1, 0.05, 37)])
    assert robustness_gate(1.0, needle).verdict is Verdict.FAIL


def test_perturbations_that_mostly_turn_negative_are_a_discard():
    fragile = RNG.normal(-0.2, 0.1, 40)
    assert robustness_gate(1.0, fragile).verdict is Verdict.DISCARD


def test_a_baseline_that_is_not_positive_is_a_discard():
    assert robustness_gate(-0.1, np.full(10, 0.2)).verdict is Verdict.DISCARD


def test_too_few_perturbations_are_refused():
    with pytest.raises(ContractViolation, match="five perturbations"):
        robustness_gate(1.0, [0.9, 0.8])


# -- the report --------------------------------------------------------------


def test_one_discard_discards_the_whole_funnel():
    report = FunnelReport((
        GateResult("a", Verdict.PASS, "x", {}),
        GateResult("b", Verdict.DISCARD, "x", {}),
        GateResult("c", Verdict.PASS, "x", {}),
    ))
    assert report.verdict is Verdict.DISCARD


def test_a_failure_outranks_a_pass_but_not_a_discard():
    passes_and_fail = FunnelReport((
        GateResult("a", Verdict.PASS, "x", {}),
        GateResult("b", Verdict.FAIL, "x", {}),
    ))
    assert passes_and_fail.verdict is Verdict.FAIL


def test_an_unrun_gate_keeps_the_funnel_from_passing():
    """Four gates out of five is not a pass, it is an unfinished funnel."""
    report = FunnelReport((
        GateResult("a", Verdict.PASS, "x", {}),
        GateResult("b", Verdict.NOT_RUN, "x", {}),
    ))
    assert report.verdict is Verdict.NOT_RUN


def test_the_report_renders_every_gate():
    report = FunnelReport((
        GateResult("1. monkey test", Verdict.PASS, "p >= 95%", {}, "100% of 60"),
        GateResult("4b. PBO", Verdict.FAIL, "PBO < 10%", {}, "PBO 22%"),
    ))
    rendered = report.render()
    assert "monkey test" in rendered
    assert "FAIL" in rendered
    assert "overall: FAIL" in rendered


# -- the bimodality detector, tested against its own failure mode ------------


def test_the_bimodality_detector_does_not_fire_on_ordinary_samples():
    """The first version did, on every one of them.

    Splitting a sample at its median always yields two groups whose means differ
    by more than their spread, so that detector discarded plain normal folds
    while looking like it worked. These cases are the regression.
    """
    from validation.funnel import BIMODALITY_THRESHOLD, bimodality_coefficient

    for seed in range(8):
        normal = np.random.default_rng(seed).normal(0.2, 0.1, 252)
        assert bimodality_coefficient(normal) < BIMODALITY_THRESHOLD, seed
        assert cpcv_gate(normal).verdict is Verdict.PASS

    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        split = np.concatenate([rng.normal(-0.3, 0.05, 126), rng.normal(0.9, 0.05, 126)])
        assert bimodality_coefficient(split) > BIMODALITY_THRESHOLD, seed
        assert cpcv_gate(split).verdict is Verdict.DISCARD


def test_the_coefficient_lands_where_the_theory_says():
    from validation.funnel import bimodality_coefficient

    rng = np.random.default_rng(7)
    assert bimodality_coefficient(rng.normal(0, 1, 20_000)) == pytest.approx(1 / 3, abs=0.03)
    assert bimodality_coefficient(rng.uniform(0, 1, 20_000)) == pytest.approx(5 / 9, abs=0.03)


# -- the walk-forward helper, and the scale error it hides -------------------


def test_walk_forward_windows_compare_like_with_like():
    """The bug this helper exists to prevent.

    Summing returns over a 156-week training window and a 52-week test window
    gives a ratio near 52/156 = 0.33 for *any* strategy, because one window is
    three times longer. That number sits just above the 0.30 discard line and
    reads exactly like a real walk-forward failure. It was the first result this
    funnel produced.
    """
    from validation.funnel import walk_forward_windows

    steady = np.full(1000, 0.002)  # identical every period, so WFE must be 1.0
    inside, outside = walk_forward_windows(steady, train=156, test=52)
    assert inside.size == outside.size > 5
    assert np.allclose(inside, 0.002)
    assert np.allclose(outside, 0.002)
    assert walk_forward_gate(inside, outside).measured["wfe"] == pytest.approx(1.0)

    # The same data summed rather than averaged gives the artefact.
    naive_in = inside * 156
    naive_out = outside * 52
    assert walk_forward_gate(naive_in, naive_out).measured["wfe"] == pytest.approx(
        52 / 156, rel=1e-9
    )


def test_walk_forward_windows_detect_a_real_collapse():
    """An edge that decays: every test window is a fraction of its training one.

    A decaying exponential makes the ratio the same for every window, so the
    result is a property of the decay rather than of where the windows happened
    to land -- which is the failure mode of a naive periodic construction.
    """
    from validation.funnel import walk_forward_windows

    periods = np.arange(1200)
    returns = 0.01 * np.exp(-periods / 70.0)
    inside, outside = walk_forward_windows(returns, train=156, test=52)
    result = walk_forward_gate(inside, outside)
    assert result.measured["wfe"] < 0.30
    assert result.verdict is Verdict.DISCARD


def test_walk_forward_windows_refuse_a_degenerate_window():
    from validation.funnel import walk_forward_windows

    with pytest.raises(ContractViolation, match="at least two periods"):
        walk_forward_windows(np.full(100, 0.01), train=1, test=52)
