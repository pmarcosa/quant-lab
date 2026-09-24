"""Dividend adjustment: the series as traded, plus a table of daily factors.

IBKR offers two price series for stocks:

* ``TRADES``: adjusted for splits but not for dividends ("TRADES data is adjusted
  for splits, but not dividends", TWS API documentation). It can be requested for
  any bar size and paged backwards with an end date.
* ``ADJUSTED_LAST``: adjusted for splits *and* dividends, but only up to now (the
  end date must be empty) and only for bars of a day or less (error 321, "Multi
  day bar size not supported with adjusted last").

The expert's criterion (NotebookLM, 2026-09-24) is that the primary layer keeps
the series as traded, immutable, and the dividend adjustment lives in a separate
table that is applied when the data is read. A pre-adjusted series is never the
primary copy, because every new dividend rewrites all of its history. Here:

* the cache (``data/ibkr_cache/<frequency>/``) holds TRADES bars;
* ``data/ibkr_cache/factors/<SYMBOL>.csv`` holds one factor per session, taken as
  ``ADJUSTED_LAST / TRADES`` on that day's close. The factor is the product of
  ``1 - D/P`` over every ex-dividend date after that day, and 1.0 on the day of
  the download;
* the store (derived, rebuilt by ``ql data ingest``) holds TRADES × factor.
  Volume is left alone: splits are already in TRADES, and a cash dividend does
  not change the number of shares.

Two stitching rules keep downloads made on different days consistent:

* **Factors.** A dividend that went ex between two downloads multiplies every
  earlier factor by the same constant. Over the days both downloads cover, the
  newer factors are the older ones times a constant ``c``, and the older days
  are rescaled by ``c``. A ratio that is not constant is refused.
* **Bars.** A split between two downloads rescales all of IBKR's earlier TRADES,
  so the ratio ``k`` of new to cached closes over the shared bars is a constant
  other than 1. The symbol's history is then downloaded again in full. The
  expert: locating the split date and patching the cache is where rounding,
  volume and edge-case errors come from. A ratio that is not constant is
  refused.

Weekly bars are built from daily TRADES. In a week with an ex-dividend date,
the sessions before it are expressed at the scale of the week's last session
before grouping. That within-week ratio is known by the end of the week and
never changes afterwards (a later dividend rescales the whole week by the same
constant), so the weekly cache stays a fixed record. The store then applies the
factor of the week's last session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from contracts.errors import ContractViolation
from contracts.temporal import BarInterval

FACTOR_DIR = "factors"
PRICE_COLUMNS = ("open", "high", "low", "close")

#: Two series that should agree up to a constant are compared point by point.
#: IBKR rounds prices to a few decimals, which moves the ratio of two low-priced
#: closes by a few parts in ten thousand. 95% of the points must sit within
#: TYPICAL of the constant, and none further than WORST.
TYPICAL = 1e-3
WORST = 5e-3
#: A ratio further than this from 1 is a rescaling: a split for bars, a new
#: dividend for factors.
RESCALED = 1e-4


# -- the factors ---------------------------------------------------------------------------


def session_dates(index) -> pd.DatetimeIndex:
    """Each label's session date, timezone-free. Regular US sessions fall on one
    UTC date, so an intraday bar's UTC date is its session's date."""
    stamps = pd.DatetimeIndex(pd.to_datetime(index))
    if stamps.tz is not None:
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    return stamps.normalize()


def daily_closes(bars: pd.DataFrame) -> pd.Series:
    """The last close of each session in ``bars``."""
    closes = bars["close"].astype(float).copy()
    closes.index = session_dates(bars.index)
    return closes.groupby(level=0).last()


def factors_from(trades: pd.DataFrame, adjusted: pd.DataFrame) -> pd.Series:
    """Daily dividend factors from IBKR's two daily series.

    Args:
        trades: Daily TRADES bars.
        adjusted: Daily ADJUSTED_LAST bars over the same window.

    Returns:
        ``ADJUSTED_LAST / TRADES`` on each session both hold, indexed by date.

    Raises:
        ContractViolation: If the two share no session, or the factors break what
            a dividend adjustment must satisfy: at most 1, 1 on the latest day,
            and never lower on a later day than on an earlier one (a later day has
            fewer dividends ahead of it).
    """
    t, a = daily_closes(trades), daily_closes(adjusted)
    common = t.index.intersection(a.index)
    if len(common) == 0:
        raise ContractViolation("TRADES and ADJUSTED_LAST share no session to compare")
    factors = (a[common] / t[common]).rename("factor")
    factors.index.name = "date"
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ContractViolation("a dividend factor is not a positive number")
    high = factors[factors > 1 + WORST]
    if not high.empty:
        raise ContractViolation(
            f"ADJUSTED_LAST is above TRADES on {high.index[0].date()} "
            f"(factor {high.iloc[0]:.4f}); a dividend adjustment only lowers past prices"
        )
    if abs(factors.iloc[-1] - 1) > WORST:
        raise ContractViolation(
            f"the latest factor is {factors.iloc[-1]:.4f}, not 1: the two series end "
            f"at different reference days"
        )
    fall = factors / factors.cummax() - 1
    if fall.min() < -WORST:
        day = fall.idxmin()
        raise ContractViolation(
            f"the dividend factor falls on {day.date()} ({fall.min():.2%}); a later "
            f"session can never have more dividends ahead of it"
        )
    return factors


def merge_factors(stored: pd.Series | None, incoming: pd.Series) -> tuple[pd.Series, float]:
    """Stitch a newer factor download onto the stored table.

    Returns:
        The merged factors and ``c``, the constant the stored days were rescaled
        by (1.0 when no dividend went ex between the two downloads).

    Raises:
        ContractViolation: If the two do not overlap, or disagree by more than a
            constant.
    """
    incoming = incoming.sort_index()
    if stored is None or stored.empty:
        return incoming, 1.0
    common = stored.index.intersection(incoming.index)
    if len(common) == 0:
        raise ContractViolation(
            f"the new dividend factors start on {incoming.index[0].date()}, after the "
            f"stored ones end ({stored.index[-1].date()}); fetch a window that overlaps"
        )
    c = _constant(incoming[common] / stored[common], "dividend factors")
    older = stored[stored.index < incoming.index[0]] * c
    return pd.concat([older, incoming]).rename("factor"), c


def factor_for(index, factors: pd.Series | None, interval: BarInterval) -> np.ndarray:
    """The dividend factor of each bar labelled by ``index``.

    A daily or intraday bar takes its session's factor, and a weekly bar takes
    the factor of its week's last session. A bar after the last factor gets 1.0:
    it is newer than the download the factors are relative to. A bar before the
    first factor gets the first factor: its returns are then price-only, but the
    series stays continuous. :func:`uncovered` reports how many there are.
    """
    n = len(index)
    if factors is None or factors.empty:
        return np.ones(n)
    days = session_dates(index)
    if interval is BarInterval.WEEK:
        days = days + pd.to_timedelta(6 - days.dayofweek, unit="D")  # the week's Sunday
    table = factors.sort_index()
    lookup = table.reindex(table.index.union(days.unique())).ffill()
    values = lookup.reindex(days).to_numpy(float)
    values = np.where(np.isnan(values), float(table.iloc[0]), values)
    return np.where(days > table.index[-1], 1.0, values)


def uncovered(index, factors: pd.Series | None) -> int:
    """How many bars predate the first factor (so carry no dividend adjustment)."""
    if factors is None or factors.empty:
        return len(index)
    return int((session_dates(index) < factors.index.min()).sum())


def adjust(bars: pd.DataFrame, factors: pd.Series | None, interval: BarInterval) -> pd.DataFrame:
    """TRADES bars × their dividend factors. Volume is unchanged."""
    if bars.empty:
        return bars
    scale = factor_for(bars.index, factors, interval)
    out = bars.copy()
    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].astype(float) * scale
    return out


def weeks_from_trades(daily: pd.DataFrame, factors: pd.Series | None) -> pd.DataFrame:
    """Weekly TRADES bars from daily ones, each week at its last session's scale.

    See the module docstring. Without factors this is plain grouping.
    """
    from data.ingest import weeks_from_days

    if daily.empty:
        return daily
    scaled = daily.copy()
    if factors is not None and not factors.empty:
        f = pd.Series(factor_for(daily.index, factors, BarInterval.DAY), index=daily.index)
        iso = session_dates(daily.index).isocalendar()
        week = pd.MultiIndex.from_arrays([iso.year.to_numpy(), iso.week.to_numpy()])
        last = f.groupby(week.to_numpy()).transform("last")
        ratio = (f / last).to_numpy()
        for column in PRICE_COLUMNS:
            scaled[column] = scaled[column].astype(float) * ratio
    return weeks_from_days(scaled)


# -- splits --------------------------------------------------------------------------------


def rescaling(stored: pd.DataFrame, incoming: pd.DataFrame, interval: BarInterval) -> float:
    """``k``, new over cached closes on the bars both hold (1.0 if they share none).

    The last cached bar is left out: it may have been written while its period
    was still open.

    Raises:
        ContractViolation: If the ratio is not one constant: the two downloads
            disagree about more than a split.
    """
    if stored.empty or incoming.empty or len(stored) < 2:
        return 1.0
    stored = stored.iloc[:-1]
    s, i = _by_period(stored, interval), _by_period(incoming, interval)
    common = s.index.intersection(i.index)
    if len(common) == 0:
        return 1.0
    return _constant(i[common] / s[common], "cached and downloaded closes")


def is_rescaled(ratio: float) -> bool:
    return abs(ratio - 1.0) > RESCALED


def _by_period(bars: pd.DataFrame, interval: BarInterval) -> pd.Series:
    closes = bars["close"].astype(float)
    if interval is BarInterval.WEEK:
        iso = session_dates(bars.index).isocalendar()
        closes = pd.Series(closes.to_numpy(), index=pd.MultiIndex.from_arrays(
            [iso.year.to_numpy(), iso.week.to_numpy()]))
    elif interval is BarInterval.DAY:
        closes = pd.Series(closes.to_numpy(), index=session_dates(bars.index))
    else:
        stamps = pd.DatetimeIndex(pd.to_datetime(bars.index))
        if stamps.tz is None:
            stamps = stamps.tz_localize("UTC")
        closes = pd.Series(closes.to_numpy(), index=stamps.tz_convert("UTC"))
    return closes[~closes.index.duplicated(keep="last")]


def _constant(ratio: pd.Series, what: str) -> float:
    """The constant ``ratio`` is, within rounding; refuses one that is not."""
    ratio = ratio.astype(float)
    c = float(ratio.median())
    deviation = (ratio / c - 1).abs()
    if deviation.quantile(0.95) > TYPICAL or deviation.max() > WORST:
        worst = deviation.idxmax()
        raise ContractViolation(
            f"{what} differ by more than one constant factor: ×{c:.4f} typically, but "
            f"{deviation.max():.2%} off it at {worst}. Refused rather than stitched."
        )
    return c


# -- files ---------------------------------------------------------------------------------


def factors_path(cache_root: Path, symbol: str) -> Path:
    return cache_root / FACTOR_DIR / f"{symbol}.csv"


def read_factors(cache_root: Path, symbol: str) -> pd.Series | None:
    path = factors_path(cache_root, symbol)
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame["factor"].astype(float).sort_index()


def write_factors(cache_root: Path, symbol: str, factors: pd.Series) -> Path:
    path = factors_path(cache_root, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = factors.sort_index().rename("factor").to_frame()
    out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m-%d")
    out.index.name = "date"
    out.to_csv(path, float_format="%.10g")
    return path
