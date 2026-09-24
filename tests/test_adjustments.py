"""data.adjustments: TRADES in the cache, dividend factors beside it, the product in the store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts.errors import ContractViolation
from contracts.temporal import BarInterval
from data.adjustments import (
    adjust,
    factor_for,
    factors_from,
    is_rescaled,
    merge_factors,
    rescaling,
    weeks_from_trades,
)
from data.ingest import weeks_from_days


def trades(days=300, seed=1, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=days)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    spread = np.abs(rng.normal(0, 0.01, days))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.005, days)),
        "high": close * (1 + spread + 0.006),
        "low": close * (1 - spread - 0.006),
        "close": close,
        "volume": rng.integers(1_000, 5_000, days).astype(float),
    }, index=index)


def with_dividends(bars, ex_dates, yields):
    """ADJUSTED_LAST for dividends going ex on ``ex_dates``: 1 - D/P off every
    earlier day, P being the close before the ex-date."""
    factor = pd.Series(1.0, index=bars.index)
    for ex, y in zip(ex_dates, yields, strict=True):
        factor[bars.index < pd.Timestamp(ex)] *= 1 - y
    out = bars.copy()
    for column in ("open", "high", "low", "close"):
        out[column] = out[column] * factor
    return out, factor


EX = ["2024-02-14", "2024-05-15", "2024-08-14", "2024-11-13"]  # Wednesdays
YIELDS = [0.008, 0.009, 0.007, 0.01]


def test_the_factor_is_adjusted_over_traded_and_one_today():
    raw = trades()
    adjusted, expected = with_dividends(raw, EX, YIELDS)
    factors = factors_from(raw, adjusted)
    assert np.allclose(factors.to_numpy(), expected.to_numpy())
    assert factors.iloc[-1] == pytest.approx(1.0)
    assert (factors.diff().dropna() >= -1e-12).all(), "never lower on a later day"


@pytest.mark.parametrize("broken, message", [
    (lambda f: f * 1.02, "above TRADES"),
    (lambda f: f.where(f.index < f.index[-1], 0.9), "latest factor"),
    (lambda f: f.where(f.index != f.index[100], f.iloc[100] * 0.9), "falls"),
])
def test_factors_that_no_dividend_could_produce_are_refused(broken, message):
    raw = trades()
    _, factor = with_dividends(raw, EX, YIELDS)
    bad = raw.copy()
    bad["close"] = raw["close"] * broken(factor)
    with pytest.raises(ContractViolation, match=message):
        factors_from(raw, bad)


def test_the_store_gets_exactly_the_weeks_of_the_adjusted_series():
    """The weekly cache holds TRADES; cache x factor must equal grouping the
    fully adjusted daily series, including weeks with an ex-date inside."""
    raw = trades()
    adjusted, _ = with_dividends(raw, EX, YIELDS)
    factors = factors_from(raw, adjusted)
    stored = adjust(weeks_from_trades(raw, factors), factors, BarInterval.WEEK)
    truth = weeks_from_days(adjusted)
    pd.testing.assert_frame_equal(stored, truth, check_exact=False, rtol=1e-12)


def test_a_later_dividend_leaves_the_weekly_cache_unchanged():
    """Within-week ratios are fixed once the week is over: a dividend going ex
    later rescales every earlier day by the same constant."""
    raw = trades()
    early, _ = with_dividends(raw, EX[:2], YIELDS[:2])
    late, _ = with_dividends(raw, EX, YIELDS)
    cutoff = pd.Timestamp("2024-06-01")
    before = weeks_from_trades(raw[raw.index < cutoff], factors_from(raw, early))
    after = weeks_from_trades(raw[raw.index < cutoff], factors_from(raw, late))
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-12)


def test_a_new_dividend_rescales_the_stored_factors_by_one_constant():
    raw = trades()
    old_window = raw[raw.index < pd.Timestamp("2024-07-31")]  # downloaded before EX[2]
    old, _ = with_dividends(old_window, EX[:2], YIELDS[:2])
    stored = factors_from(old_window, old)
    new, _ = with_dividends(raw, EX, YIELDS)
    incoming = factors_from(raw.iloc[100:], new.iloc[100:])  # a one-window refresh
    merged, c = merge_factors(stored, incoming)
    assert is_rescaled(c) and c == pytest.approx((1 - YIELDS[2]) * (1 - YIELDS[3]))
    assert np.allclose(merged.to_numpy(), factors_from(raw, new).to_numpy())


def test_factor_downloads_that_disagree_are_refused():
    raw = trades()
    adjusted, _ = with_dividends(raw, EX, YIELDS)
    stored = factors_from(raw, adjusted)
    incoming = stored.iloc[150:].copy()
    incoming.iloc[:20] *= 0.99  # half the overlap moved, half not: no single constant
    with pytest.raises(ContractViolation, match="more than one constant"):
        merge_factors(stored, incoming)
    with pytest.raises(ContractViolation, match="overlaps"):
        merge_factors(stored.iloc[:100], stored.iloc[150:])


def test_a_split_shows_as_one_constant_between_cached_and_new_closes():
    raw = trades()
    halved = raw.copy()
    for column in ("open", "high", "low", "close"):
        halved[column] = raw[column] / 2
    k = rescaling(raw.iloc[:200], halved.iloc[150:], BarInterval.DAY)
    assert is_rescaled(k) and k == pytest.approx(0.5)
    assert not is_rescaled(rescaling(raw.iloc[:200], raw.iloc[150:], BarInterval.DAY))
    noisy = raw.copy()
    noisy.loc[noisy.index[160:170], "close"] *= 1.05
    with pytest.raises(ContractViolation):
        rescaling(raw.iloc[:200], noisy.iloc[150:], BarInterval.DAY)


def test_the_last_cached_bar_is_not_compared():
    """It may have been written mid-period by an earlier fetch."""
    raw = trades()
    cached = raw.iloc[:200].copy()
    cached.iloc[-1, cached.columns.get_loc("close")] *= 1.03
    assert not is_rescaled(rescaling(cached, raw.iloc[150:], BarInterval.DAY))


def test_bars_outside_the_factors_take_the_nearest_sensible_value():
    raw = trades()
    adjusted, _ = with_dividends(raw, EX, YIELDS)
    factors = factors_from(raw, adjusted).iloc[50:250]
    days = raw.index
    f = factor_for(days, factors, BarInterval.DAY)
    assert (f[:50] == factors.iloc[0]).all(), "before the factors: the first (price-only)"
    assert (f[250:] == 1.0).all(), "after the download: nothing ahead of them yet"
    hours = pd.DatetimeIndex([days[120] + pd.Timedelta(hours=h) for h in (13.5, 17.5, 19.5)],
                             tz="UTC")
    assert np.allclose(factor_for(hours, factors, BarInterval.HOUR), factors[days[120]]), \
        "every bar of a session takes that session's factor"


def test_volume_is_not_adjusted():
    raw = trades()
    adjusted, _ = with_dividends(raw, EX, YIELDS)
    factors = factors_from(raw, adjusted)
    assert (adjust(raw, factors, BarInterval.DAY)["volume"] == raw["volume"]).all()
