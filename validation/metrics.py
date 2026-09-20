"""Performance measured from an equity curve, with the definitions written down.

Every number here has more than one defensible definition, and the differences
are large enough to change a decision. The previous system reported a Sharpe of
1.80 where the conventional definition gives 1.61 — on the identical backtest —
because it divided CAGR by volatility instead of annualising the mean excess
return. Neither is wrong; quoting one while comparing against the other is.

So: the definition used is stated in the docstring of the function that computes
it, the periodicity is always passed in rather than guessed from the data, and
the alternative is computed alongside rather than left for someone to rediscover.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from contracts.errors import ContractViolation

#: Calendar days in a year, for converting a span to years.
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class Performance:
    """What a run did, by the conventional definitions.

    Attributes:
        years: Span of the curve, in calendar years.
        cagr: Geometric annual growth of equity.
        volatility: Annualised standard deviation of periodic returns.
        sharpe: Annualised mean excess return over its standard deviation.
        sharpe_geometric: ``(cagr - rf) / volatility``. Reported because the
            previous system used it; systematically higher, and not comparable
            with published Sharpe figures.
        sortino: As Sharpe, but against downside deviation only.
        max_drawdown: Deepest peak-to-trough fall in equity, as a fraction.
        final_equity: Where the curve ended.
        periods: Number of return observations.
    """

    years: float
    cagr: float
    volatility: float
    sharpe: float
    sharpe_geometric: float
    sortino: float
    max_drawdown: float
    final_equity: float
    periods: int

    def as_row(self) -> dict[str, float]:
        """A flat mapping, for a table or a ledger row."""
        return {
            "years": self.years,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sharpe_geometric": self.sharpe_geometric,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "final_equity": self.final_equity,
            "periods": float(self.periods),
        }


def summarise(
    curve: Sequence[tuple[datetime, float]],
    periods_per_year: float,
    risk_free: float = 0.0,
) -> Performance:
    """Summarise an equity curve.

    Args:
        curve: (moment, equity) pairs, ascending. At least two points.
        periods_per_year: 52 for weekly, 252 for daily. Passed rather than
            inferred: inferring it from timestamps gets holidays and gaps wrong,
            and a wrong annualisation factor moves Sharpe by tens of percent
            without looking like an error.
        risk_free: Annual risk-free rate, as a fraction.

    Returns:
        The summary.

    Raises:
        ContractViolation: If the curve is too short, not ascending, or contains
            a non-positive equity (a return series through zero is meaningless,
            and a blown-up account should be read as such, not annualised).
    """
    if len(curve) < 2:
        raise ContractViolation(f"need at least two points to measure a curve; got {len(curve)}")

    moments = [m for m, _ in curve]
    values = np.asarray([v for _, v in curve], dtype=float)
    for earlier, later in zip(moments, moments[1:], strict=False):
        if later < earlier:
            raise ContractViolation("curve must be ascending in time")
    if np.any(values <= 0):
        raise ContractViolation("equity must stay positive to be measured as a return series")

    span_days = (moments[-1] - moments[0]).total_seconds() / 86_400.0
    years = span_days / DAYS_PER_YEAR
    if years <= 0:
        raise ContractViolation("curve must span a positive amount of time")

    returns = values[1:] / values[:-1] - 1.0
    cagr = (values[-1] / values[0]) ** (1.0 / years) - 1.0

    # Sample standard deviation: with n periods the mean costs one degree of
    # freedom, and on a short run the difference is not negligible.
    volatility = float(np.std(returns, ddof=1)) * math.sqrt(periods_per_year) if len(returns) > 1 else 0.0

    periodic_rf = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess = returns - periodic_rf
    deviation = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    sharpe = (
        float(np.mean(excess)) / deviation * math.sqrt(periods_per_year) if deviation > 0 else 0.0
    )

    downside = excess[excess < 0.0]
    downside_deviation = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
    sortino = (
        float(np.mean(excess)) / downside_deviation * math.sqrt(periods_per_year)
        if downside_deviation > 0
        else 0.0
    )

    peak = np.maximum.accumulate(values)
    max_drawdown = float(np.min(values / peak - 1.0))

    return Performance(
        years=years,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sharpe_geometric=(cagr - risk_free) / volatility if volatility > 0 else 0.0,
        sortino=sortino,
        max_drawdown=max_drawdown,
        final_equity=float(values[-1]),
        periods=len(returns),
    )
