"""Weekly cross-sectional momentum: rank, gate, rotate.

The strategy the user runs, expressed against the framework's contracts. It imports
``contracts`` and nothing else — not the engine, not the data layer, not another
strategy — which is what the architecture test enforces and what makes it
replaceable.

The rules, unchanged from the version validated over 17 years:

1. **Rank** the eligible universe by trailing return over ``lookback_weeks``.
2. **Gate** on trend and pace. A name must be above a rising SMA, and its recent
   pace must be a real fraction of its long-horizon pace, which rejects names
   that have already finished their move — something the ranking alone cannot
   see.
3. **Exit** held names on trend break, RSI reversion, or a drawdown from the
   rolling high. Exits are evaluated before selection, so a name that triggered
   one cannot be re-bought the same week.
4. **Size** inverse to volatility, bounded (see :mod:`strategies.sizing`).

An empty target is a position, not a failure: it means nothing qualified and the
book should be in cash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId, StrategyVersion
from contracts.targets import TargetIntent
from contracts.temporal import BarInterval, Filtration, FiltrationSpec
from strategies.sizing import capped_proportional

STRATEGY_NAME = "weekly-momentum"

#: A fixed Monday, so "which week is this" is a function of the decision time
#: alone. Counting bars from the start of a run would make the rotation phase
#: depend on where the run began, and two runs over overlapping windows would
#: rotate on different weeks -- which is a silent difference, not a visible one.
CADENCE_EPOCH = datetime(1999, 1, 4, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class MomentumParams:
    """Every number the rules read. All horizons in weeks.

    These are part of the strategy's identity, not settings applied to it: a
    13-week lookback and a 26-week lookback are two strategies, and validation
    evidence earned by one does not transfer. ``StrategyVersion`` fingerprints
    them for exactly that reason.

    Attributes:
        lookback_weeks: Ranking horizon; the score is the trailing return.
        pace_weeks: Short horizon for the deceleration filter.
        sma_weeks: Trend SMA period.
        sma_slope_weeks: Bars back the SMA must have risen over.
        pace_ratio_min: Short pace must reach this fraction of the long pace.
        top_n: Positions held.
        min_history_weeks: Bars a name needs before it may be selected. Point-in-
            time availability, not a data-quality filter.
        rsi_period: RSI lookback.
        atr_period: ATR lookback, used for inverse-volatility sizing.
        use_rsi_entry_filter: Retired 2026-09-14 — it cost more than it saved.
            Kept switchable so the removal stays falsifiable.
        rsi_entry_max: RSI ceiling when that retired filter is switched on.
        rsi_exit_high: RSI level that arms the reversion exit.
        rsi_exit_low: RSI level that fires it once armed.
        rsi_exit_window: Bars the arming level is looked for in.
        high_drawdown_exit: Fraction below the rolling high that forces an exit.
        weight_floor_mult: Lower weight bound, as a multiple of equal weight.
        weight_cap_mult: Upper weight bound, as a multiple of equal weight.
        rebalance_weeks: Weeks between rotations. Between them the strategy holds
            what it holds; it does not go to cash and it does not re-rank.
    """

    lookback_weeks: int = 13
    pace_weeks: int = 4
    sma_weeks: int = 10
    sma_slope_weeks: int = 4
    pace_ratio_min: float = 0.40
    top_n: int = 4
    min_history_weeks: int = 27
    rsi_period: int = 14
    atr_period: int = 14
    use_rsi_entry_filter: bool = False
    rsi_entry_max: float = 70.0
    rsi_exit_high: float = 70.0
    rsi_exit_low: float = 50.0
    rsi_exit_window: int = 4
    high_drawdown_exit: float = 0.12
    weight_floor_mult: float = 0.5
    weight_cap_mult: float = 2.0
    rebalance_weeks: int = 1

    def __post_init__(self) -> None:
        if self.rebalance_weeks < 1:
            raise ContractViolation(
                f"rebalance_weeks must be at least 1; got {self.rebalance_weeks}"
            )
        if self.top_n < 1:
            raise ContractViolation(f"top_n must be at least 1; got {self.top_n}")
        if not 0.0 <= self.weight_floor_mult <= 1.0 <= self.weight_cap_mult:
            raise ContractViolation(
                f"weight bounds must satisfy floor <= 1 <= cap as multiples of equal weight; "
                f"got floor {self.weight_floor_mult}, cap {self.weight_cap_mult}"
            )
        if self.weight_cap_mult * self.top_n < 1.0:
            raise ContractViolation(
                f"{self.top_n} names capped at {self.weight_cap_mult}x equal weight cannot "
                f"fill the book"
            )

    @property
    def warmup_weeks(self) -> int:
        """Bars needed before every series the rules read has a value."""
        return max(
            self.lookback_weeks,
            self.sma_weeks + self.sma_slope_weeks,
            self.rsi_period + self.rsi_exit_window,
            self.atr_period,
        ) + 1


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, the recursive form behind RSI and ATR.

    Written out rather than taken from a library on purpose. Implementations seed
    the first value differently, and these rules are threshold comparisons where
    a different seed silently changes which weeks qualify. This matches the
    engine the rules were validated on, which is the only version whose results
    mean anything.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def indicators(bars: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
    """Every series the rules read, from one instrument's bars.

    Each column at bar ``i`` uses only bars up to and including ``i``. That is a
    property of the operations used (rolling, shift forward, recursive smoothing)
    and is tested rather than asserted.

    Args:
        bars: Frame with high, low and close, oldest first.
        params: Horizons.

    Returns:
        Frame on the same index with the columns the rules consume.
    """
    missing = [c for c in ("high", "low", "close") if c not in bars.columns]
    if missing:
        raise ContractViolation(f"bars are missing column(s): {missing}")

    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)

    change = close.diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = wilder(change.clip(lower=0.0), params.rsi_period) / wilder(
            (-change).clip(lower=0.0), params.rsi_period
        )

    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["sma"] = close.rolling(params.sma_weeks).mean()
    out["sma_prev"] = out["sma"].shift(1)
    out["sma_slope_ref"] = out["sma"].shift(params.sma_slope_weeks)
    out["ret_long"] = close / close.shift(params.lookback_weeks) - 1.0
    out["ret_pace"] = close / close.shift(params.pace_weeks) - 1.0
    out["rsi"] = 100.0 - 100.0 / (1.0 + strength)
    out["rsi_recent_max"] = out["rsi"].rolling(params.rsi_exit_window, min_periods=1).max()
    out["atr"] = wilder(true_range, params.atr_period)
    out["atr_pct"] = out["atr"] / close.where(close > 0)
    out["high_long"] = high.rolling(params.lookback_weeks).max()
    return out


def entry_ok(row: Mapping[str, float], params: MomentumParams) -> bool:
    """Whether a name passes the ranking gate and both entry filters."""
    ret_long = row["ret_long"]
    if not np.isfinite(ret_long) or ret_long <= 0:
        return False

    if params.use_rsi_entry_filter:
        rsi_now = row["rsi"]
        if not (np.isfinite(rsi_now) and rsi_now < params.rsi_entry_max):
            return False

    sma, sma_ref, close = row["sma"], row["sma_slope_ref"], row["close"]
    if not (np.isfinite(sma) and np.isfinite(sma_ref)):
        return False
    if not (close > sma and sma > sma_ref):
        return False

    ret_pace = row["ret_pace"]
    if not np.isfinite(ret_pace):
        return False
    pace_long = ret_long / params.lookback_weeks
    pace_short = ret_pace / params.pace_weeks
    return not (pace_long > 0 and (ret_pace < 0 or pace_short < params.pace_ratio_min * pace_long))


def exit_reason(row: Mapping[str, float], params: MomentumParams) -> str | None:
    """Why a held name should be sold, or None to keep holding.

    Returns one of ``trend_break``, ``rsi_reversion``, ``below_high``, or None.
    """
    close, sma, sma_prev = row["close"], row["sma"], row["sma_prev"]
    if np.isfinite(sma) and close < sma and not (sma > sma_prev):
        return "trend_break"

    rsi_now, rsi_max = row["rsi"], row["rsi_recent_max"]
    if np.isfinite(rsi_max) and rsi_max >= params.rsi_exit_high and rsi_now < params.rsi_exit_low:
        return "rsi_reversion"

    high_long = row["high_long"]
    if np.isfinite(high_long) and close < (1.0 - params.high_drawdown_exit) * high_long:
        return "below_high"
    return None


#: Ask the filtration for everything it knows. Wilder smoothing is recursive, so
#: an indicator computed over a short trailing window differs from the same
#: indicator computed over the full history -- by little, but these rules are
#: threshold comparisons, and a small difference changes which weeks qualify.
FULL_HISTORY = 1_000_000


@dataclass(frozen=True, slots=True)
class WeeklyMomentum:
    """The strategy. Deterministic: same filtration and holdings, same target.

    ``precomputed`` is a per-run optimisation and nothing more. Recomputing every
    instrument's indicators at every decision time is 35,000 passes over a
    900-bar series for one 17-year run, which makes a funnel of seventy-five
    runs take half an hour. When a caller supplies frames covering the whole run,
    the strategy slices them at the decision time instead.

    This is safe for exactly one reason, and it is tested rather than asserted:
    every indicator at bar *i* uses only bars up to *i*
    (``test_every_indicator_uses_only_its_own_bar_and_earlier``), so the row at
    time *t* of a frame built over all history is identical to the last row of a
    frame built over history up to *t*. The strategy still reads only
    ``frame.loc[:t]``, and
    ``test_the_precomputed_path_agrees_with_the_filtration_path`` pins the two
    paths against each other on the real data.
    """

    params: MomentumParams = MomentumParams()
    code_version: str = "2.0"
    precomputed: Mapping[InstrumentId, pd.DataFrame] | None = None

    @property
    def version(self) -> StrategyVersion:
        return StrategyVersion.of(
            STRATEGY_NAME, asdict(self.params), code_version=self.code_version
        )

    @property
    def filtration_spec(self) -> FiltrationSpec:
        # Zero lag: the ingest already stamps each weekly bar as available a
        # quarter of an hour after its close, so the lag lives in the data where
        # it is measurable, not in a setting that has to be remembered.
        return FiltrationSpec(interval=BarInterval.WEEK, observation_lag_bars=0)

    def universe(self, moment: datetime) -> Sequence[InstrumentId]:
        """Deliberately empty: the point-in-time universe is the filtration's.

        A strategy that carried its own list of instruments would be carrying a
        list assembled today, which is how a backtest of 2011 ends up trading
        names that listed in 2019.
        """
        return ()

    def rank(self, filtration: Filtration, held: Mapping[InstrumentId, float]) -> pd.DataFrame:
        """Score and gate the whole eligible universe at the decision time.

        Returns one row per available instrument, sorted by score descending.
        """
        params = self.params
        depth = FULL_HISTORY
        rows: list[dict[str, Any]] = []

        for instrument in filtration.universe(min_bars=params.min_history_weeks):
            table = self._indicators_at(filtration, instrument, depth)
            if table is None or len(table) < params.min_history_weeks:
                continue
            row = table.iloc[-1]
            if not np.isfinite(row["close"]):
                continue
            is_held = instrument in held
            reason = exit_reason(row, params) if is_held else None
            rows.append(
                {
                    "instrument": instrument,
                    "score": float(row["ret_long"]),
                    "pace": float(row["ret_pace"]),
                    "atr_pct": float(row["atr_pct"]),
                    "rsi": float(row["rsi"]),
                    "held": is_held,
                    "exit_reason": reason,
                    "eligible": bool(entry_ok(row, params)) and reason is None,
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("score", ascending=False, na_position="last").reset_index(
            drop=True
        )

    def rotates_at(self, moment: datetime) -> bool:
        """Whether this decision time is a rotation week.

        Derived from the calendar, not from a counter, so a replay of any window
        rotates on exactly the weeks the original run did.
        """
        weeks = (moment - CADENCE_EPOCH).days // 7
        return weeks % self.params.rebalance_weeks == 0

    def target(
        self, filtration: Filtration, held: Mapping[InstrumentId, float]
    ) -> TargetIntent:
        """The book to hold from this decision time until the next.

        Between rotations the strategy re-states what it already holds. That is
        not a no-op: it is the difference between a cadence and a gap. Returning
        an empty target on a non-rotation week would mean "go to cash", and
        skipping the decision entirely would leave the book unmarked -- which is
        how a drawdown between rotations goes unmeasured. The previous system
        marked only at rebalance closes, and doing it properly moved its reported
        maximum drawdown from -23.7% to -28.8%.
        """
        params = self.params
        if not self.rotates_at(filtration.decision_time):
            return TargetIntent(
                weights=dict(held),
                horizon_bars=params.rebalance_weeks,
                as_of=filtration.decision_time,
                diagnostics={"rotated": 0.0},
            )

        table = self.rank(filtration, held)

        weights: dict[InstrumentId, float] = {}
        diagnostics: dict[str, float] = {"candidates": float(len(table))}

        if not table.empty:
            picks = table.loc[table["eligible"]].head(params.top_n)
            scored = {
                row.instrument: 1.0 / row.atr_pct
                for row in picks.itertuples()
                if np.isfinite(row.atr_pct) and row.atr_pct > 0
            }
            if len(scored) == len(picks) and scored:
                equal = 1.0 / len(scored)
                weights = capped_proportional(
                    scored,
                    floor=params.weight_floor_mult * equal,
                    cap=params.weight_cap_mult * equal,
                )
            elif len(picks):
                # An unusable ATR would let one name dominate the book through a
                # division by something near zero. Equal weight is the honest
                # fallback; silently dropping the name is not.
                equal = 1.0 / len(picks)
                weights = {row.instrument: equal for row in picks.itertuples()}
            diagnostics["selected"] = float(len(weights))
            diagnostics["eligible"] = float(int(table["eligible"].sum()))

        diagnostics["rotated"] = 1.0
        return TargetIntent(
            weights=weights,
            horizon_bars=params.rebalance_weeks,
            as_of=filtration.decision_time,
            diagnostics=diagnostics,
        )

    def _indicators_at(
        self, filtration: Filtration, instrument: InstrumentId, depth: int
    ) -> pd.DataFrame | None:
        """Indicators knowable at the decision time, precomputed or not."""
        if self.precomputed is not None:
            frame = self.precomputed.get(instrument)
            if frame is None or frame.empty:
                return None
            visible = frame.loc[frame.index <= filtration.decision_time]
            return None if visible.empty else visible
        bars = _bars(filtration, instrument, FULL_HISTORY)
        if bars.empty:
            return None
        return indicators(bars, self.params)

    def state(self) -> Mapping[str, Any]:
        """The parameters. Per-decision values ride on the target's diagnostics.

        Keeping this free of per-decision state is what lets one instance serve a
        backtest, a replay and a live proposal at once without the three
        interfering. A strategy that remembered its last decision here would be
        reporting its history rather than its identity.
        """
        return {"name": STRATEGY_NAME, "code_version": self.code_version, **asdict(self.params)}


def _bars(filtration: Filtration, instrument: InstrumentId, count: int) -> pd.DataFrame:
    """High, low and close for one instrument, aligned, oldest first."""
    fields = {field: filtration.history(instrument, field, count) for field in ("high", "low", "close")}
    if any(series.empty for series in fields.values()):
        return pd.DataFrame()
    return pd.DataFrame(fields)
