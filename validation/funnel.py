"""The falsification funnel: five gates a strategy must survive to be believed.

The funnel is not a scoring system. Each gate asks a different way for the
strategy to be *wrong*, and the order is deliberate — the cheap tests that catch
obvious nonsense run first, and the expensive ones that catch subtle
self-deception run last.

1. **Sign-flip and monkey test.** Does the strategy beat a random selection from
   the same universe, on the same dates, with the same sizing and costs? And does
   the signal carry any mutual information with the outcome? A strategy that
   fails here is not overfitted, it is a description of the universe's drift.
2. **CPCV with purging and embargo.** Out-of-sample Sharpe across every
   combinatorial partition. The output is a distribution: the threshold is on
   ``P(Sharpe < 0)``, and a *bimodal* distribution is a discard regardless of its
   mean, because it means the result depends on which period you happened to get.
3. **Walk-forward efficiency.** Out-of-sample return divided by in-sample return.
   Below 0.30 the in-sample result was fiction.
4. **Deflated Sharpe and PBO.** Given how many configurations were tried, how
   surprising is the winner, and does the selection procedure beat a coin toss?
   This is the gate that bites hardest and it needs the research ledger.
5. **Jitter, noise and slippage.** Perturb the inputs and the costs. A result
   that survives only at one parameter setting is a needle, not a plateau.

The thresholds are the project's, fixed in the design before any of them were
run, which is the only time it is honest to fix them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from contracts.errors import ContractViolation


class Verdict(str, Enum):
    """What a gate concluded.

    ``DISCARD`` is not a worse ``FAIL``. A failure means the evidence is not
    there yet; a discard means the evidence is that the strategy does not work,
    and further tuning of it is wasted effort.
    """

    PASS = "pass"
    FAIL = "fail"
    DISCARD = "discard"
    NOT_RUN = "not run"


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's conclusion, with the numbers behind it."""

    gate: str
    verdict: Verdict
    threshold: str
    measured: Mapping[str, float]
    note: str = ""

    def line(self) -> str:
        marks = {
            Verdict.PASS: "pass",
            Verdict.FAIL: "FAIL",
            Verdict.DISCARD: "DISCARD",
            Verdict.NOT_RUN: "-",
        }
        return f"{self.gate:<34}{marks[self.verdict]:<9}{self.threshold:<26}{self.note}"


@dataclass(frozen=True, slots=True)
class FunnelReport:
    """Every gate, and what the set of them means together."""

    results: tuple[GateResult, ...] = field(default_factory=tuple)

    @property
    def verdict(self) -> Verdict:
        """One discard discards. Otherwise the weakest gate decides."""
        verdicts = [r.verdict for r in self.results]
        if Verdict.DISCARD in verdicts:
            return Verdict.DISCARD
        if Verdict.FAIL in verdicts:
            return Verdict.FAIL
        if Verdict.NOT_RUN in verdicts:
            return Verdict.NOT_RUN
        return Verdict.PASS

    def render(self) -> str:
        header = f"{'gate':<34}{'verdict':<9}{'threshold':<26}detail"
        rule = "-" * 96
        lines = [header, rule, *(r.line() for r in self.results), rule]
        lines.append(f"overall: {self.verdict.value.upper()}")
        return "\n".join(lines)


# -- gate 1: is there anything there at all ----------------------------------


def monkey_test(
    strategy_returns: Sequence[float],
    random_returns: Sequence[Sequence[float]],
    quantile: float = 0.95,
) -> GateResult:
    """Does the strategy beat random selection from the same universe?

    The control must share everything except the selection rule: the same dates,
    the same number of positions, the same sizing, the same costs, the same
    point-in-time universe. Otherwise it measures the difference between two
    experiments rather than the value of the signal.

    What this can and cannot establish: it neutralises selection bias *within*
    the universe. It says nothing about a universe that was itself assembled with
    hindsight, which is a separate and larger problem.
    """
    series = np.asarray(strategy_returns, dtype=float)
    controls = [np.asarray(r, dtype=float) for r in random_returns]
    if len(controls) < 20:
        raise ContractViolation(
            f"a percentile from {len(controls)} controls is noise; use at least 20"
        )

    strategy_sharpe = _sharpe(series)
    control_sharpes = np.array([_sharpe(r) for r in controls])
    percentile = float((control_sharpes < strategy_sharpe).mean())
    p_value = float((control_sharpes >= strategy_sharpe).mean())

    verdict = Verdict.PASS if percentile >= quantile else Verdict.FAIL
    if percentile < 0.5:
        verdict = Verdict.DISCARD
    return GateResult(
        gate="1. monkey test",
        verdict=verdict,
        threshold=f"percentile >= {quantile:.0%}",
        measured={
            "sharpe": strategy_sharpe,
            "percentile": percentile,
            "p_value": p_value,
            "controls": float(len(controls)),
            "control_median_sharpe": float(np.median(control_sharpes)),
        },
        note=f"{percentile:.0%} of {len(controls)} controls, p={p_value:.3f}",
    )


# -- gate 2: does it survive honest resampling -------------------------------


def cpcv_gate(
    fold_sharpes: Sequence[float], max_negative: float = 0.15
) -> GateResult:
    """Out-of-sample Sharpe across combinatorial partitions.

    Two ways to fail. The obvious one is too many negative folds. The subtle one
    is **bimodality**: a strategy that is excellent in one half of history and
    useless in the other has a respectable average and no future, because nothing
    says which regime comes next. That is a discard, not a fail.
    """
    values = np.asarray(fold_sharpes, dtype=float)
    if values.size < 10:
        raise ContractViolation(f"need at least ten folds to judge a distribution; got {values.size}")

    negative = float((values < 0).mean())
    bimodal = _is_bimodal(values)
    coefficient = bimodality_coefficient(values)

    if bimodal:
        verdict = Verdict.DISCARD
    elif negative <= max_negative:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.FAIL

    note = f"{negative:.0%} negative of {values.size} folds, median {np.median(values):+.3f}"
    if bimodal:
        note += " -- BIMODAL"
    return GateResult(
        gate="2. CPCV purged + embargoed",
        verdict=verdict,
        threshold=f"P(Sharpe<0) <= {max_negative:.0%}",
        measured={
            "negative_fraction": negative,
            "median": float(np.median(values)),
            "folds": float(values.size),
            "worst": float(values.min()),
            "best": float(values.max()),
            "bimodality": coefficient,
        },
        note=note,
    )


#: Sarle's bimodality coefficient exceeds this for a uniform distribution and
#: anything flatter or more split; a normal sits near 1/3.
BIMODALITY_THRESHOLD = 5.0 / 9.0


def bimodality_coefficient(values: np.ndarray) -> float:
    """Sarle's coefficient: ``(skew^2 + 1) / corrected excess kurtosis``.

    Used instead of the obvious "split at the median and compare the halves",
    which was tried first and is worthless: splitting *any* sample at its median
    produces two groups whose means differ by more than their own spread, so that
    test fires on a plain normal sample and would have discarded every real
    result while looking like a working detector.

    This one is closed-form, needs no fitting, and separates the cases by
    construction: a normal gives about 0.33, a uniform exactly 5/9, and two
    well-separated clusters approach 1.
    """
    n = values.size
    if n < 4:
        return 0.0
    deviation = values.std(ddof=1)
    if deviation == 0:
        return 0.0
    centred = (values - values.mean()) / deviation
    skewness = float(np.mean(centred**3) * n**2 / ((n - 1) * (n - 2)))
    excess = float(
        (np.sum(centred**4) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)))
        - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    )
    denominator = excess + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denominator <= 0:
        return 1.0
    return float((skewness**2 + 1.0) / denominator)


def _is_bimodal(values: np.ndarray) -> bool:
    """Whether the folds fall into two groups rather than one."""
    if values.size < 20:
        return False
    return bimodality_coefficient(values) > BIMODALITY_THRESHOLD


# -- gate 3: does it hold up rolled forward ----------------------------------


def walk_forward_windows(
    returns: Sequence[float], train: int, test: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling in-sample and out-of-sample performance, on a comparable scale.

    Returns **mean return per period** for each window, not the sum. The
    distinction is the whole correctness of the gate: with a 156-week training
    window and a 52-week test window, comparing sums gives a ratio of about 0.33
    no matter what the strategy does, because the in-sample window is three times
    longer. That artefact is indistinguishable from a real walk-forward failure
    and sits just above the 0.30 discard line, which is where it was first found.

    Args:
        returns: The periodic return series.
        train: Periods in each fitting window.
        test: Periods in each evaluation window, and the step between windows.

    Returns:
        ``(in_sample, out_of_sample)`` per-period means, one pair per window.
    """
    series = np.asarray(returns, dtype=float)
    if train < 2 or test < 2:
        raise ContractViolation(f"windows must hold at least two periods; got {train}/{test}")
    inside, outside = [], []
    start = 0
    while start + train + test <= series.size:
        inside.append(float(series[start : start + train].mean()))
        outside.append(float(series[start + train : start + train + test].mean()))
        start += test
    return np.array(inside), np.array(outside)


def walk_forward_gate(
    in_sample: Sequence[float], out_of_sample: Sequence[float], minimum: float = 0.50
) -> GateResult:
    """Walk-forward efficiency: out-of-sample performance over in-sample.

    One means the strategy performed out of sample exactly as it did in sample.
    Below 0.30 the in-sample figure described the fitting, not the market.

    Both arguments must be on the **same scale** — per-period means, or
    annualised figures, but not a three-year sum against a one-year sum. Use
    :func:`walk_forward_windows` to build them.
    """
    inside = np.asarray(in_sample, dtype=float)
    outside = np.asarray(out_of_sample, dtype=float)
    if inside.size != outside.size:
        raise ContractViolation("each in-sample window needs its out-of-sample counterpart")
    if inside.size < 3:
        raise ContractViolation(f"need at least three windows; got {inside.size}")

    usable = inside > 0
    if not usable.any():
        return GateResult(
            gate="3. walk-forward efficiency",
            verdict=Verdict.DISCARD,
            threshold=f"WFE >= {minimum:.2f}",
            measured={"windows": float(inside.size)},
            note="no window was profitable in sample; there is nothing to carry forward",
        )
    efficiency = float(np.median(outside[usable] / inside[usable]))

    if efficiency < 0.30:
        verdict = Verdict.DISCARD
    elif efficiency >= minimum:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.FAIL
    return GateResult(
        gate="3. walk-forward efficiency",
        verdict=verdict,
        threshold=f"WFE >= {minimum:.2f}",
        measured={"wfe": efficiency, "windows": float(int(usable.sum()))},
        note=f"WFE {efficiency:.2f} over {int(usable.sum())} windows",
    )


# -- gate 4: how many things were tried --------------------------------------


def deflated_sharpe_gate(
    dsr: float, z: float, observed: float, expected_max: float, n_effective: float,
    minimum: float = 0.95,
) -> GateResult:
    """The Deflated Sharpe Ratio against the trial count."""
    if z < 0 and observed <= expected_max:
        verdict = Verdict.DISCARD
    elif dsr >= minimum:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.FAIL
    return GateResult(
        gate="4a. deflated Sharpe",
        verdict=verdict,
        threshold=f"DSR >= {minimum:.2f}",
        measured={
            "dsr": dsr, "z": z, "observed_sharpe": observed,
            "expected_max_sharpe": expected_max, "n_effective": n_effective,
        },
        note=(
            f"DSR {dsr:.3f}; SR {observed:+.3f} vs luck {expected_max:+.3f} "
            f"at N_eff {n_effective:.1f}"
        ),
    )


def pbo_gate(pbo: float, combinations: int, maximum: float = 0.10) -> GateResult:
    """Probability of backtest overfitting: does the *selection* work?"""
    if pbo > 0.50:
        verdict = Verdict.DISCARD
    elif pbo < maximum:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.FAIL
    return GateResult(
        gate="4b. probability of overfitting",
        verdict=verdict,
        threshold=f"PBO < {maximum:.0%}",
        measured={"pbo": pbo, "combinations": float(combinations)},
        note=f"PBO {pbo:.1%} over {combinations} partitions",
    )


# -- gate 5: is it a plateau or a needle -------------------------------------


def robustness_gate(
    baseline: float, perturbed: Sequence[float], minimum_retained: float = 0.50
) -> GateResult:
    """Perturb the inputs and see what survives.

    A result that holds only at one parameter value, one cost assumption and one
    random seed is a needle in the search space — it was found by looking, not by
    being there. The measure is the fraction of perturbations that keep at least
    half the baseline performance.
    """
    values = np.asarray(perturbed, dtype=float)
    if values.size < 5:
        raise ContractViolation(f"need at least five perturbations; got {values.size}")
    if baseline <= 0:
        return GateResult(
            gate="5. jitter, noise, slippage",
            verdict=Verdict.DISCARD,
            threshold=f"retained >= {minimum_retained:.0%}",
            measured={"baseline": baseline},
            note="the unperturbed result is not positive",
        )

    retained = float((values >= baseline * 0.5).mean())
    still_positive = float((values > 0).mean())
    verdict = Verdict.PASS if retained >= minimum_retained else Verdict.FAIL
    if still_positive < 0.5:
        verdict = Verdict.DISCARD
    return GateResult(
        gate="5. jitter, noise, slippage",
        verdict=verdict,
        threshold=f"retained >= {minimum_retained:.0%}",
        measured={
            "baseline": baseline, "retained": retained,
            "still_positive": still_positive, "perturbations": float(values.size),
            "median": float(np.median(values)),
        },
        note=(
            f"{retained:.0%} kept half the baseline, {still_positive:.0%} stayed positive "
            f"over {values.size} runs"
        ),
    )


def _sharpe(returns: np.ndarray) -> float:
    deviation = returns.std(ddof=1)
    return float(returns.mean() / deviation) if deviation > 0 else 0.0
