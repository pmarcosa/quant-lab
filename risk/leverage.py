"""How much leverage to use, decided by rule rather than by mood.

From the project's expert, for a concentrated momentum book with a 30-40%
drawdown tolerance:

1. **A hard cap, 1.3x-1.5x gross.** Kelly-style sizing assumes a known return
   distribution; with fat tails even a quarter-Kelly exceeds that tolerance.
   The cap is ``risk.max_gross``, enforced by the gross-exposure rule; this
   module never asks for more.
2. **Tail-risk scaling (optional).** Leverage is set so the book's expected
   shortfall at 95% matches a target: ``L = CVaR_target / CVaR_base``, where
   ``CVaR_base`` is measured on the strategy's returns per unit of exposure.
   Without a target, the leverage is a fixed ``target``.
3. **Convex de-leveraging in drawdown.** Past a 10% drawdown the leverage falls
   with the square of how far the drawdown has gone towards 35%:
   ``L* = L_floor + (L - L_floor) * (1 - S**2)``, ``S = clip((D - 0.10) / 0.25)``.
4. **Cut at once, rebuild slowly.** A lower number applies at the next
   decision; a higher one rises at most 0.05x per calendar week, and only while
   equity is above its recent low.
5. **Margin cushion.** Below 35% (IBKR's cushion, ``1 - maintenance/equity``)
   nothing is added; below 25% the leverage drops to the floor.

**One adaptation, deliberate.** The expert's formula scales the *whole* book
towards zero, with a floor of 0.1x. Here it scales only the borrowed part, down
to ``floor`` (1.0 by default): below 1x the strategy is no longer the one that
was validated, and the unlevered strategy's own drawdowns are already governed
by monitoring's reduce-only and halt. Setting ``floor`` below 1 restores the
expert's version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contracts.errors import ContractViolation
from contracts.risk import LeverageState


@dataclass(frozen=True, slots=True)
class LeverageSchedule:
    """The expert's adaptive leverage, as one pure function of the book's history.

    Attributes:
        target: Gross leverage wanted in normal conditions, when no tail-risk
            target is set. 1.0 means no borrowing.
        maximum: The hard cap. Mirrors ``risk.max_gross``.
        cvar_target: Expected shortfall per bar the levered book should carry,
            as a positive fraction (0.04 = 4% per bar). ``None``: fixed target.
        cvar_alpha: Tail level for the shortfall.
        cvar_window: Bars of history the shortfall is measured over.
        min_history: Fewer base returns than this and the shortfall is not
            trusted; the fixed target applies.
        drawdown_start: Drawdown where de-leveraging begins.
        drawdown_full: Drawdown where the leverage has reached the floor.
        convexity: The exponent on the drawdown severity (the expert's gamma).
        step_up_per_week: Most the leverage may rise per calendar week.
        floor: The least leverage the schedule will set. See the module note.
        cushion_warning: Below this cushion, leverage may not rise.
        cushion_critical: Below this cushion, leverage drops to the floor.
        recovery_lookback: Bars equity must beat the low of before rising.
    """

    target: float = 1.0
    maximum: float = 1.0
    cvar_target: float | None = None
    cvar_alpha: float = 0.95
    cvar_window: int = 104
    min_history: int = 26
    drawdown_start: float = 0.10
    drawdown_full: float = 0.35
    convexity: float = 2.0
    step_up_per_week: float = 0.05
    floor: float = 1.0
    cushion_warning: float = 0.35
    cushion_critical: float = 0.25
    recovery_lookback: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.floor <= self.maximum:
            raise ContractViolation("leverage: floor must be in (0, maximum]")
        if not self.floor <= self.target <= self.maximum:
            raise ContractViolation(
                f"leverage: target {self.target} must lie between the floor "
                f"{self.floor} and the maximum {self.maximum} (risk.max_gross)"
            )
        if self.cvar_target is not None and self.cvar_target <= 0:
            raise ContractViolation("leverage: cvar_target must be positive")
        if not 0 <= self.drawdown_start < self.drawdown_full < 1:
            raise ContractViolation("leverage: need 0 <= drawdown_start < drawdown_full < 1")
        if self.convexity < 1:
            raise ContractViolation("leverage: convexity below 1 de-levers faster early, not later")
        if not 0 < self.cushion_critical <= self.cushion_warning < 1:
            raise ContractViolation("leverage: need 0 < cushion_critical <= cushion_warning < 1")

    @property
    def is_static_unlevered(self) -> bool:
        """True when this schedule can only ever answer 1.0."""
        return self.maximum <= 1.0 and self.floor == 1.0 and self.target == 1.0

    def base_level(self, base_returns) -> float:
        """Leverage before drawdown and cushion: fixed, or set by tail risk."""
        if self.cvar_target is None:
            return self.target
        window = np.asarray(list(base_returns)[-self.cvar_window:], dtype=float)
        window = window[np.isfinite(window)]
        if window.size < self.min_history:
            return self.target
        shortfall = expected_shortfall(window, self.cvar_alpha)
        if shortfall <= 1e-9:
            return self.maximum
        return self.cvar_target / shortfall

    def __call__(self, state: LeverageState) -> float:
        equity = [e for e in state.equity if math.isfinite(e) and e > 0]
        wanted = min(max(self.base_level(state.base_returns), self.floor), self.maximum)

        # Convex de-leveraging on the drawdown from the running peak.
        if equity:
            peak = max(equity)
            drawdown = 1.0 - equity[-1] / peak
            severity = min(max(
                (drawdown - self.drawdown_start) / (self.drawdown_full - self.drawdown_start), 0.0
            ), 1.0)
            wanted = self.floor + (wanted - self.floor) * (1.0 - severity ** self.convexity)

        # The margin cushion overrides everything above it.
        if state.cushion is not None and state.cushion < self.cushion_critical:
            return self.floor

        previous = min(max(state.previous, self.floor), self.maximum)
        if wanted <= previous:
            return wanted  # cut at once
        if state.cushion is not None and state.cushion < self.cushion_warning:
            return previous
        recent = equity[-(self.recovery_lookback + 1):-1]
        if equity and recent and equity[-1] <= min(recent):
            return previous  # equity has not improved: hold
        step = self.step_up_per_week / max(state.bars_per_week, 1e-9)
        return min(wanted, previous + step)


def expected_shortfall(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Mean loss beyond the (1 - alpha) quantile, as a positive number."""
    if returns.size == 0:
        return 0.0
    cutoff = np.percentile(returns, (1.0 - alpha) * 100.0)
    tail = returns[returns <= cutoff]
    return float(-tail.mean()) if tail.size else float(-cutoff)
