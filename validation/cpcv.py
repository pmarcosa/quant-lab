"""Combinatorial purged cross-validation: splits that do not leak.

Ordinary k-fold on a return series leaks, twice over.

**Overlap.** A position opened in week *t* and held for four weeks is still open
in week *t+3*. If week *t* is in training and week *t+3* is in test, the training
set already contains the outcome the test is supposed to be predicting. The fix
is **purging**: drop the training observations whose holding period reaches into
the test block.

**Serial memory.** Even with no overlap, the bar immediately after a test block
is not independent of it — volatility clusters, and a rolling indicator computed
just after the block still carries values from inside it. The fix is an
**embargo**: a quarantine after each test block that no training observation may
occupy.

The combinatorial part is what makes the result a distribution rather than a
number. With S blocks and S/2 of them held out, there are C(S, S/2) different
train/test partitions, each giving its own estimate. One walk-forward split gives
one number and no sense of how much it could have differed.

Parameters here follow the project's expert for a 4-week holding period over
roughly 900 weekly bars: S = 10 blocks of about 90 weeks, 252 combinations,
purge of 4 bars (the holding period), embargo of 9 bars (1% of the sample).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from contracts.errors import ContractViolation

#: Blocks the sample is cut into. Must be even so S/2 is a whole number.
DEFAULT_BLOCKS = 10

#: Bars of quarantine after each test block, as a fraction of the sample. The
#: literature puts it between 1% and 3%; 1% of 900 weekly bars is 9.
DEFAULT_EMBARGO_FRACTION = 0.01


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition, after purging and embargo.

    Attributes:
        train: Indices usable for fitting.
        test: Indices held out.
        purged: Indices dropped because their holding period reached into a test
            block.
        embargoed: Indices dropped because they sat in the quarantine after one.
    """

    train: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    embargoed: np.ndarray

    @property
    def dropped(self) -> int:
        """How many observations the leakage controls cost."""
        return len(self.purged) + len(self.embargoed)

    def is_clean(self) -> bool:
        """No index appears on both sides. The property the whole module exists for."""
        return not (set(self.train.tolist()) & set(self.test.tolist()))


def contiguous_runs(indices: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into runs of consecutive values.

    A combinatorial test set is several blocks, which may or may not be adjacent.
    Purge and embargo apply at the edge of each *run*, not of each block: two
    adjacent blocks have one leading edge and one trailing edge between them, not
    two of each.
    """
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    return np.split(indices, breaks)


def purged_splits(
    n_samples: int,
    holding_bars: int,
    blocks: int = DEFAULT_BLOCKS,
    test_blocks: int | None = None,
    embargo_bars: int | None = None,
) -> Iterator[Split]:
    """Every combinatorial split of ``n_samples`` with purging and embargo.

    Args:
        n_samples: Length of the return series.
        holding_bars: How long a position stays open. Sets the purge: an
            observation opened within ``holding_bars`` of a test block is still
            open inside it.
        blocks: Number of contiguous blocks. Must be even and at least 4.
        test_blocks: Blocks held out per split. Defaults to ``blocks // 2``,
            which makes the train and test halves equally precise.
        embargo_bars: Quarantine after each test run. Defaults to 1% of the
            sample, rounded up.

    Yields:
        One :class:`Split` per combination, in a deterministic order.

    Raises:
        ContractViolation: If the parameters cannot produce usable splits.
    """
    if blocks % 2 != 0:
        raise ContractViolation(f"blocks must be even; got {blocks}")
    if blocks < 4:
        raise ContractViolation(f"blocks must be at least 4 to be combinatorial; got {blocks}")
    if n_samples < blocks * 2:
        raise ContractViolation(
            f"{n_samples} samples cannot make {blocks} blocks of a usable size"
        )
    if holding_bars < 1:
        raise ContractViolation(f"holding_bars must be at least 1; got {holding_bars}")

    held_out = blocks // 2 if test_blocks is None else test_blocks
    if not 1 <= held_out < blocks:
        raise ContractViolation(f"test_blocks must be in [1, {blocks}); got {held_out}")
    embargo = (
        int(np.ceil(n_samples * DEFAULT_EMBARGO_FRACTION))
        if embargo_bars is None
        else embargo_bars
    )
    if embargo < 0:
        raise ContractViolation(f"embargo_bars cannot be negative; got {embargo}")

    everything = np.arange(n_samples)
    groups = np.array_split(everything, blocks)

    for chosen in itertools.combinations(range(blocks), held_out):
        test = np.sort(np.concatenate([groups[g] for g in chosen]))
        outside = np.setdiff1d(everything, test, assume_unique=False)

        purge_mask = np.zeros(n_samples, dtype=bool)
        embargo_mask = np.zeros(n_samples, dtype=bool)
        for run in contiguous_runs(test):
            start, end = int(run[0]), int(run[-1])
            # Purge before: anything opened within the holding period is still
            # open once the test block starts.
            purge_mask[max(0, start - holding_bars) : start] = True
            # Embargo after: serial dependence does not stop at the boundary.
            embargo_mask[end + 1 : min(n_samples, end + 1 + embargo)] = True

        excluded = purge_mask | embargo_mask
        purged = outside[purge_mask[outside]]
        embargoed = outside[embargo_mask[outside] & ~purge_mask[outside]]
        train = outside[~excluded[outside]]
        yield Split(train=train, test=test, purged=purged, embargoed=embargoed)


def split_count(blocks: int = DEFAULT_BLOCKS, test_blocks: int | None = None) -> int:
    """How many combinations a configuration produces, without building them."""
    held_out = blocks // 2 if test_blocks is None else test_blocks
    return len(list(itertools.combinations(range(blocks), held_out)))


def block_sharpes(returns: np.ndarray, splits: Sequence[Split]) -> np.ndarray:
    """Out-of-sample Sharpe for one return series across every split.

    The distribution, not the average, is the output. A strategy whose
    out-of-sample Sharpe is sometimes 1.4 and sometimes -0.3 has a mean that
    describes none of its behaviour, and the bimodality is itself a discard
    criterion.
    """
    values = np.asarray(returns, dtype=float)
    out = np.empty(len(splits), dtype=float)
    for i, split in enumerate(splits):
        sample = values[split.test]
        deviation = sample.std(ddof=1)
        out[i] = sample.mean() / deviation if deviation > 0 else 0.0
    return out
