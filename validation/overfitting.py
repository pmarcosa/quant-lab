"""How much of a result is skill, given how many things were tried.

Two statistics, both from Bailey and López de Prado, and both answering a
question a Sharpe ratio cannot.

**Deflated Sharpe Ratio.** If you try enough strategies, one of them will look
good. The DSR asks: given N trials, how surprised should we be by the best
Sharpe we found? It also corrects for non-normal returns, because a strategy
that makes small gains and occasional catastrophic losses has a flattering
Sharpe precisely where it is most dangerous.

**Probability of Backtest Overfitting.** The DSR judges a result. The PBO judges
the *procedure*: across many train/test partitions, how often does the
configuration that looked best in-sample fall below the median out-of-sample? If
that happens more than half the time, the selection process is worse than
choosing at random, and the answer is not a better configuration.

Both need an honest trial count, which is what ``validation.ledger`` exists to
keep.

A note on frequency. The Sharpe entering the DSR is in the **same period as the
returns**, not annualised. The expected-maximum term is also in that scale, and
the non-normality correction is built from periodic moments, so annualising one
part and not the others produces a number that is not the DSR. The reference
implementation supplied by the project's expert annualises ``sr_hat`` while its
own definition of the variable says "in the same time frequency as the return
series"; the definition is what is followed here, and the test
``test_the_dsr_is_computed_in_the_return_frequency`` pins it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, skew

from contracts.errors import ContractViolation

#: Euler-Mascheroni, in the expected-maximum-Sharpe approximation.
EULER_MASCHERONI = 0.5772156649015329


def effective_trials(returns_matrix: np.ndarray, method: str = "entropy") -> float:
    """How many *distinct* things were tried, given that trials are correlated.

    Sweeping a lookback from 12 to 14 weeks is three trials but barely more than
    one idea: the return series are nearly identical. Counting them as three
    independent attempts over-penalises the DSR; counting a genuinely diverse
    set as one under-penalises it. Both are answered by the eigenvalue spectrum
    of the trials' correlation matrix.

    Args:
        returns_matrix: ``T x N`` array, one column per trial.
        method: ``"entropy"`` for ``exp(H)`` of the normalised spectrum (the
            expert's recommendation), or ``"herfindahl"`` for Roy's effective
            rank, ``(sum l)^2 / sum l^2``.

    Returns:
        A value in ``[1, N]``. One when every trial is the same idea; N when they
        are orthogonal.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ContractViolation("returns_matrix must be a non-empty T x N array")
    n_trials = matrix.shape[1]
    if n_trials == 1:
        return 1.0

    correlation = np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(correlation), 1e-12)

    if method == "entropy":
        weights = eigenvalues / eigenvalues.sum()
        entropy = float(-np.sum(weights * np.log(weights)))
        effective = math.exp(entropy)
    elif method == "herfindahl":
        effective = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
    else:
        raise ContractViolation(f"unknown method {method!r}; use entropy or herfindahl")
    return float(min(max(effective, 1.0), n_trials))


def expected_max_sharpe(n_trials: float, trial_sharpe_std: float) -> float:
    """The Sharpe the best of ``n_trials`` worthless strategies would show.

    Bailey and Lopez de Prado, equation (1)::

        E[max SR_N] = mean(SR_i) + std(SR_i) * [(1-g) Z(1 - 1/N) + g Z(1 - 1/(Ne))]

    Under the null the mean is zero, so the threshold is the **cross-trial
    standard deviation of the Sharpe estimates** times a bracket that grows like
    sqrt(2 ln N).

    That standard deviation is not optional, and dropping it is not a small
    simplification: the bracket alone is a standard-normal quantile, around 2.3
    for fifty trials, which on a weekly per-period Sharpe scale is a threshold no
    real strategy approaches. Every result would be rejected, and the statistic
    would look conservative rather than broken. The project's expert supplied the
    bracket without this factor — see the module note in the build log — and the
    published paper is what is implemented here.

    What the factor means: if fifty configurations all produce nearly the same
    Sharpe, their dispersion is small and the best of them is barely above the
    rest, so luck explains little. If they are scattered widely, the best one is
    much more likely to be the top of a noisy draw.

    Args:
        n_trials: Effective independent trials.
        trial_sharpe_std: Standard deviation of the per-period Sharpe estimates
            across those trials.

    Returns:
        The per-period Sharpe that luck alone would produce.
    """
    if n_trials < 1:
        raise ContractViolation(f"n_trials must be at least 1; got {n_trials}")
    if trial_sharpe_std < 0:
        raise ContractViolation(
            f"trial_sharpe_std cannot be negative; got {trial_sharpe_std}"
        )
    if n_trials == 1 or trial_sharpe_std == 0:
        return 0.0
    first = norm.ppf(1.0 - 1.0 / n_trials)
    second = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    bracket = (1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second
    return float(trial_sharpe_std * bracket)


def trial_sharpe_dispersion(returns_matrix: np.ndarray) -> float:
    """Standard deviation of the per-period Sharpe estimates across trials.

    The ``std(SR_i)`` that :func:`expected_max_sharpe` needs. Taken from the
    trials actually run, which is the reason the ledger keeps their return
    series rather than only their summary numbers.

    With a single trial there is no dispersion to measure; the fallback is the
    asymptotic standard error of one Sharpe estimate, ``1/sqrt(T)``, which is
    what the dispersion tends to when the trials are independent draws under the
    null. It is a substitute, and a study that reports one should say so.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ContractViolation("returns_matrix must be a non-empty T x N array")
    periods, n_trials = matrix.shape
    if n_trials < 2:
        return float(1.0 / math.sqrt(periods)) if periods > 0 else 0.0
    deviation = matrix.std(axis=0, ddof=1)
    sharpes = np.divide(
        matrix.mean(axis=0), deviation, out=np.zeros(n_trials), where=deviation > 0
    )
    return float(sharpes.std(ddof=1))


@dataclass(frozen=True, slots=True)
class DeflatedSharpe:
    """The DSR and everything needed to argue with it.

    Attributes:
        dsr: Probability the true Sharpe exceeds the luck threshold. The project
            requires 0.95.
        z: The underlying statistic.
        observed_sharpe: Per-period Sharpe of the selected strategy.
        expected_max: Per-period Sharpe luck alone would produce over N trials.
        n_effective: Distinct trials behind the selection.
        trial_sharpe_std: Dispersion of the trials' Sharpe estimates.
        skew: Sample skewness of the returns.
        kurtosis: Sample Pearson kurtosis (3 is normal).
        periods: Observations in the series.
    """

    dsr: float
    z: float
    observed_sharpe: float
    expected_max: float
    n_effective: float
    trial_sharpe_std: float
    skew: float
    kurtosis: float
    periods: int

    @property
    def clears_the_luck_threshold(self) -> bool:
        """Whether the observed Sharpe even beats what N trials produce by chance."""
        return self.observed_sharpe > self.expected_max


def deflated_sharpe(
    returns: Sequence[float], n_effective: float, trial_sharpe_std: float
) -> DeflatedSharpe:
    """Deflate a Sharpe ratio by the number of trials and the shape of the returns.

    Args:
        returns: Periodic returns of the selected strategy.
        n_effective: Effective independent trials, from :func:`effective_trials`.
        trial_sharpe_std: Dispersion of the trials' Sharpe estimates, from
            :func:`trial_sharpe_dispersion`.

    Returns:
        The statistic and its inputs.

    Raises:
        ContractViolation: If the series is too short or has no variance.
    """
    series = np.asarray([r for r in returns if r == r], dtype=float)
    periods = series.size
    if periods < 3:
        raise ContractViolation(f"need at least three returns; got {periods}")
    deviation = float(series.std(ddof=1))
    # Not `== 0`: the sample deviation of a constant series comes out as a tiny
    # non-zero float, which sails past an equality check and then produces
    # nonsense moments with a precision-loss warning rather than an error.
    scale = float(np.abs(series).max())
    if deviation <= max(scale, 1.0) * 1e-12:
        raise ContractViolation(
            f"a constant return series has no Sharpe to deflate (deviation {deviation:.3g})"
        )

    # Per-period, not annualised: the expected-maximum term and the
    # non-normality correction are both in this scale.
    observed = float(series.mean()) / deviation
    sample_skew = float(skew(series, bias=False))
    sample_kurtosis = float(kurtosis(series, fisher=False, bias=False))
    threshold = expected_max_sharpe(n_effective, trial_sharpe_std)

    # The standard error of a Sharpe estimate under non-normal returns. Negative
    # skew and fat tails both inflate it, which is the point: the same Sharpe is
    # weaker evidence when the return distribution hides tail risk.
    variance = 1.0 - sample_skew * observed + ((sample_kurtosis - 1.0) / 4.0) * observed**2
    if variance <= 0:
        raise ContractViolation(
            f"the non-normality correction is non-positive ({variance:.4g}); the sample "
            f"moments and the Sharpe are mutually inconsistent, so the DSR is undefined"
        )

    z = (observed - threshold) * math.sqrt(periods - 1) / math.sqrt(variance)
    return DeflatedSharpe(
        dsr=float(norm.cdf(z)),
        z=float(z),
        observed_sharpe=observed,
        expected_max=threshold,
        n_effective=float(n_effective),
        trial_sharpe_std=float(trial_sharpe_std),
        skew=sample_skew,
        kurtosis=sample_kurtosis,
        periods=periods,
    )


@dataclass(frozen=True, slots=True)
class Overfitting:
    """The PBO and the distribution behind it.

    Attributes:
        pbo: Fraction of partitions where the in-sample winner landed below the
            out-of-sample median. Above 0.5 means the selection procedure is
            worse than a coin toss.
        combinations: Partitions evaluated.
        logits: Per-partition log-odds of the winner's out-of-sample rank.
        median_relative_rank: Median out-of-sample rank of the winner, in (0, 1).
    """

    pbo: float
    combinations: int
    logits: np.ndarray
    median_relative_rank: float

    @property
    def verdict(self) -> str:
        if self.pbo > 0.50:
            return "discard: selection is worse than chance"
        if self.pbo >= 0.10:
            return "inconclusive: selection is not reliably better than chance"
        return "pass"


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, blocks: int = 10
) -> Overfitting:
    """PBO by combinatorially symmetric cross-validation.

    Cut the sample into S contiguous blocks; for each way of splitting them into
    equal in-sample and out-of-sample halves, find the trial with the best
    in-sample Sharpe and see where it ranks out-of-sample. The logit of that
    relative rank, accumulated over every partition, gives a distribution; the
    PBO is the mass at or below zero.

    Args:
        returns_matrix: ``T x N`` of trial returns, all on the same clock.
        blocks: Contiguous blocks, even.

    Returns:
        The statistic and its distribution.

    Raises:
        ContractViolation: With fewer than two trials — there is no selection to
            judge — or if the blocks do not fit.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ContractViolation("returns_matrix must be T x N")
    periods, n_trials = matrix.shape
    if n_trials < 2:
        raise ContractViolation(
            f"PBO judges a choice between trials; got {n_trials}. With one trial "
            f"there was no selection, and therefore no selection bias to measure."
        )
    if blocks % 2 != 0:
        raise ContractViolation(f"blocks must be even; got {blocks}")
    if periods < blocks * 2:
        raise ContractViolation(f"{periods} periods cannot make {blocks} usable blocks")

    size = periods // blocks
    pieces = [matrix[i * size : (i + 1) * size, :] for i in range(blocks)]

    logits: list[float] = []
    ranks: list[float] = []
    for in_sample in itertools.combinations(range(blocks), blocks // 2):
        out_of_sample = tuple(sorted(set(range(blocks)) - set(in_sample)))
        train = np.vstack([pieces[i] for i in in_sample])
        test = np.vstack([pieces[i] for i in out_of_sample])

        winner = int(np.argmax(_sharpes(train)))
        test_sharpes = _sharpes(test)
        # Rank 1 is worst, N is best, so a high relative rank is a good outcome.
        order = np.argsort(np.argsort(test_sharpes)) + 1
        relative = float(order[winner]) / (n_trials + 1.0)
        ranks.append(relative)
        logits.append(math.log(relative / (1.0 - relative)))

    values = np.asarray(logits, dtype=float)
    return Overfitting(
        pbo=float(np.mean(values <= 0.0)),
        combinations=values.size,
        logits=values,
        median_relative_rank=float(np.median(ranks)),
    )


def _sharpes(matrix: np.ndarray) -> np.ndarray:
    deviation = matrix.std(axis=0, ddof=1)
    return np.divide(
        matrix.mean(axis=0), deviation, out=np.zeros(matrix.shape[1]), where=deviation > 0
    )
