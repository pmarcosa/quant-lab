"""Performance measurement, including the definition that was quietly different."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from contracts.errors import ContractViolation
from validation.metrics import summarise


def curve(values, weeks=1):
    start = datetime(2020, 1, 3, tzinfo=timezone.utc)
    return [(start + timedelta(weeks=weeks * i), v) for i, v in enumerate(values)]


def test_a_doubling_over_one_year_is_a_hundred_percent():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    points = [(start, 100.0), (start + timedelta(days=365.25), 200.0)]
    stats = summarise(points, periods_per_year=1)
    assert stats.years == pytest.approx(1.0)
    assert stats.cagr == pytest.approx(1.0)
    assert stats.final_equity == 200.0


def test_a_flat_curve_has_no_return_and_no_risk():
    stats = summarise(curve([100.0] * 53), periods_per_year=52)
    assert stats.cagr == pytest.approx(0.0, abs=1e-9)
    assert stats.volatility == pytest.approx(0.0)
    assert stats.sharpe == 0.0
    assert stats.max_drawdown == pytest.approx(0.0)


def test_the_drawdown_is_peak_to_trough_not_start_to_trough():
    stats = summarise(curve([100.0, 200.0, 120.0, 260.0]), periods_per_year=52)
    assert stats.max_drawdown == pytest.approx(-0.4), "200 down to 120"


def test_the_two_sharpe_definitions_are_reported_side_by_side():
    """The previous system's 1.80 and the conventional 1.61 on one backtest.

    Both are computable from the same curve; quoting one while comparing against
    the other is the error, so both are returned and named.
    """
    values = [100.0]
    for i in range(104):
        values.append(values[-1] * (1.01 if i % 3 else 0.99))
    stats = summarise(curve(values), periods_per_year=52)

    assert stats.sharpe != stats.sharpe_geometric
    assert stats.sharpe_geometric == pytest.approx(stats.cagr / stats.volatility)


def test_annualisation_uses_the_periodicity_it_is_given():
    """A wrong factor moves Sharpe by tens of percent without looking wrong."""
    values = [100.0 * (1.002 ** i) if i % 2 else 100.0 * (1.001 ** i) for i in range(105)]
    weekly = summarise(curve(values), periods_per_year=52)
    daily = summarise(curve(values), periods_per_year=252)
    assert daily.sharpe / weekly.sharpe == pytest.approx(math.sqrt(252 / 52), rel=1e-6)


def test_a_curve_that_reaches_zero_is_refused_rather_than_annualised():
    with pytest.raises(ContractViolation, match="positive"):
        summarise(curve([100.0, 50.0, 0.0]), periods_per_year=52)


def test_a_curve_needs_two_points():
    with pytest.raises(ContractViolation, match="two points"):
        summarise(curve([100.0]), periods_per_year=52)


def test_a_curve_must_ascend_in_time():
    points = curve([100.0, 110.0, 120.0])
    with pytest.raises(ContractViolation, match="ascending"):
        summarise([points[0], points[2], points[1]], periods_per_year=52)


def test_sortino_ignores_upside_deviation():
    """Two curves with identical downside and different upside."""
    calm = [100.0]
    wild = [100.0]
    for i in range(104):
        down = i % 4 == 0
        calm.append(calm[-1] * (0.98 if down else 1.01))
        wild.append(wild[-1] * (0.98 if down else 1.05))
    calm_stats = summarise(curve(calm), periods_per_year=52)
    wild_stats = summarise(curve(wild), periods_per_year=52)
    assert wild_stats.sortino > calm_stats.sortino
