"""Turning scores into weights, with bounds that actually hold.

Lives in ``strategies/`` because it is a strategy's business how it splits its
book between the names it picked. The engine's job starts afterwards, at whole
shares.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId


def capped_proportional(
    scores: Mapping[InstrumentId, float], floor: float, cap: float, tol: float = 1e-12
) -> dict[InstrumentId, float]:
    """Weights proportional to ``scores``, inside ``[floor, cap]``, summing to one.

    The obvious construction is wrong, and was live in the previous system until
    2026-09-20. Normalising, clamping to the bounds and then dividing by the new
    sum does not respect the bounds: clamping the largest weight down leaves the
    total below one, and the final division scales every weight back up —
    including the one just capped. The declared limit buys nothing, and it fails
    silently, because the weights still sum to one.

    Solve for the scale instead of applying one afterwards. Let
    ``f(s) = sum(clip(s * score_i, floor, cap))``. It is continuous and
    non-decreasing, runs from ``n * floor`` to ``n * cap``, so while
    ``n * floor <= 1 <= n * cap`` there is an ``s`` where ``f(s) = 1``. Bisection
    finds it. Names away from a bound keep their exact proportions; constrained
    ones sit *on* their bound rather than past it.

    Args:
        scores: Positive, finite, unnormalised scores.
        floor: Minimum weight for any one name.
        cap: Maximum weight for any one name.
        tol: Tolerance on the weight sum.

    Returns:
        Weights summing to one, each within the bounds.

    Raises:
        ContractViolation: If the bounds admit no book summing to one, or a score
            is not positive and finite.
    """
    names = list(scores)
    n = len(names)
    if n == 0:
        return {}
    for name in names:
        value = scores[name]
        if not math.isfinite(value) or value <= 0:
            raise ContractViolation(f"score for {name} must be finite and positive; got {value!r}")
    if floor > cap:
        raise ContractViolation(f"floor {floor} is above cap {cap}")
    if n * floor > 1.0 + tol:
        raise ContractViolation(
            f"{n} names at a {floor:.4f} floor require {n * floor:.4f}, which exceeds one"
        )
    if n * cap < 1.0 - tol:
        raise ContractViolation(
            f"{n} names at a {cap:.4f} cap reach only {n * cap:.4f}, which is short of one"
        )

    def total(scale: float) -> float:
        return sum(min(max(scale * scores[k], floor), cap) for k in names)

    # Start from the unconstrained scale, so the common case where no bound
    # binds converges in a couple of steps.
    lo = hi = 1.0 / sum(scores[k] for k in names)
    while total(lo) > 1.0:
        lo /= 2.0
    while total(hi) < 1.0:
        hi *= 2.0

    scale = 0.5 * (lo + hi)
    for _ in range(200):
        scale = 0.5 * (lo + hi)
        reached = total(scale)
        if abs(reached - 1.0) <= tol:
            break
        if reached < 1.0:
            lo = scale
        else:
            hi = scale

    weights = {k: min(max(scale * scores[k], floor), cap) for k in names}

    # Bisection lands within tol of one, not on it. Push the residue onto the
    # names with room, so the book sums to one without anything crossing a bound.
    residue = 1.0 - sum(weights.values())
    if residue:
        room = {k: (cap - weights[k]) if residue > 0 else (weights[k] - floor) for k in names}
        spare = sum(room.values())
        if spare > 0:
            for k in names:
                weights[k] += residue * room[k] / spare
    return weights
