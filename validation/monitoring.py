"""Live monitoring: is the system still the system that was validated?

With a four-week rotation marked weekly, a year of live trading is about fifty-two
observations. Means and t-statistics on fifty-two points cannot tell a real loss
of edge from a bad run, and acting on them produces both errors: switching off a
healthy strategy after bad luck, and keeping one whose edge has gone. So nothing
here compares live results with a single backtest number. Every live statistic is
placed inside a *distribution* built from the backtest, for a window the same
length as the live record.

Four instruments, from the project's expert, each answering a different question:

1. **Drawdown against the bootstrap.** A stationary bootstrap (Politis and
   Romano) of the backtest's weekly returns generates thousands of synthetic
   paths as long as the live record. The live drawdown's percentile in that
   distribution says how unusual it is *for this strategy*, not for strategies in
   general. The same paths give a prediction band for cumulative return.
2. **Online changepoint detection.** Bayesian online changepoint detection (Adams
   and MacKay) runs over the backtest and then the live weeks, and reports the
   posterior probability that the return process changed during live trading.
3. **Robust trend.** The median weekly return, with a bootstrap interval.
   Judged only after enough weeks, and only when the whole interval is below
   zero: a point estimate on a few dozen weeks is noise.
4. **Implementation shortfall.** What execution cost against the decision price,
   compared with what the backtest assumed. It separates "the signal stopped
   working" from "the fills got worse", which call for different responses.

Each maps onto the degradation ladder with the expert's thresholds. These
functions only *recommend* a state. The live session applies it: monitoring may
lift a reduce-only it imposed itself, but only a person lifts a halt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln, logsumexp

from contracts.errors import ContractViolation
from contracts.live import DegradationState

# -- 1. the bootstrap --------------------------------------------------------


def stationary_bootstrap(
    returns: Sequence[float], horizon: int, paths: int, block: float, seed: int = 7
) -> np.ndarray:
    """Synthetic return paths that keep the series' short-range dependence.

    Each path starts at a random week and continues through consecutive weeks,
    jumping to a new random start with probability ``1/block`` at each step, and
    wrapping at the end. Blocks of random length keep the volatility clustering
    and momentum persistence an independent resample would destroy, which is what
    makes the resulting drawdowns realistic rather than optimistic.

    Returns:
        Array of shape ``(paths, horizon)``.
    """
    series = np.asarray(returns, dtype=float)
    n = series.size
    if n < 20:
        raise ContractViolation(f"a bootstrap reference needs at least 20 returns; got {n}")
    if horizon < 1:
        raise ContractViolation(f"horizon must be at least one week; got {horizon}")
    if block < 1:
        raise ContractViolation(f"mean block length must be at least 1; got {block}")
    rng = np.random.default_rng(seed)
    index = np.empty((paths, horizon), dtype=np.int64)
    index[:, 0] = rng.integers(0, n, size=paths)
    restart = rng.random((paths, horizon)) < 1.0 / block
    fresh = rng.integers(0, n, size=(paths, horizon))
    for t in range(1, horizon):
        index[:, t] = np.where(restart[:, t], fresh[:, t], (index[:, t - 1] + 1) % n)
    return series[index]


def max_drawdown(returns: np.ndarray) -> np.ndarray:
    """Deepest peak-to-trough fall along the last axis, as a positive fraction."""
    values = np.cumprod(1.0 + np.atleast_2d(returns), axis=-1)
    values = np.concatenate([np.ones((values.shape[0], 1)), values], axis=-1)
    peaks = np.maximum.accumulate(values, axis=-1)
    return np.max(1.0 - values / peaks, axis=-1)


@dataclass(frozen=True, slots=True)
class DrawdownCheck:
    live_drawdown: float
    percentile: float  # share of bootstrap paths with a shallower drawdown
    live_return: float
    band_low_10: float
    band_low_1: float
    weeks: int


def drawdown_check(
    backtest_returns: Sequence[float], live_returns: Sequence[float],
    paths: int = 5000, block: float = 6.0,
) -> DrawdownCheck | None:
    """The live drawdown and cumulative return, placed in the bootstrap's distribution.

    The horizon is the live record's own length, capped at a year. Comparing a
    ten-week live drawdown with the distribution of one-year drawdowns — as a
    fixed 52-week window would — understates how unusual an early loss is.
    """
    live = np.asarray(live_returns, dtype=float)
    if live.size == 0:
        return None
    window = live[-52:]
    simulated = stationary_bootstrap(backtest_returns, window.size, paths, block)
    drawdowns = max_drawdown(simulated)
    observed = float(max_drawdown(window)[0])
    cumulative = np.prod(1.0 + simulated, axis=1) - 1.0
    return DrawdownCheck(
        live_drawdown=observed,
        percentile=float(np.mean(drawdowns < observed)),
        live_return=float(np.prod(1.0 + window) - 1.0),
        band_low_10=float(np.quantile(cumulative, 0.10)),
        band_low_1=float(np.quantile(cumulative, 0.01)),
        weeks=int(window.size),
    )


# -- 2. changepoint detection ------------------------------------------------


def changepoint_probability(
    reference: Sequence[float], live: Sequence[float], hazard_weeks: float = 250.0
) -> float:
    """Posterior probability that the return process changed during live trading.

    Bayesian online changepoint detection with a Normal-Gamma model and a
    constant hazard. The run length is the number of weeks since the last
    change; after the backtest and then the live weeks have been processed, the
    answer is the posterior mass on run lengths shorter than the live record.

    The prior for a *new* regime is deliberately vague — centred on zero with a
    wide spread — rather than copied from the backtest. If a fresh regime were
    given the backtest's own parameters, "nothing changed" and "it changed into
    something identical" would fit equally well, and the probability would just
    echo the hazard rate instead of responding to the data.

    **What it can and cannot see, measured** (``tests/test_monitoring.py``): on
    weekly returns with a 3% weekly deviation it stays below 0.20 on healthy data
    in all but a few percent of 52-week samples, and it picks up a tripling of
    volatility within a quarter. A collapse of the *mean* is harder: a fall from
    +0.45% to -1.5% a week — two thirds of a standard deviation — takes most of a
    year to reach one half. That is not a defect to be tuned away; it is the
    power fifty-two observations a year actually have. The drawdown bootstrap
    catches that case within months, which is why the two run together.
    """
    history = np.asarray(reference, dtype=float)
    recent = np.asarray(live, dtype=float)
    if recent.size == 0:
        return 0.0
    data = np.concatenate([history, recent])
    scale = float(np.std(history, ddof=1)) if history.size > 1 else float(np.std(data))
    scale = scale if scale > 0 else 1e-3
    log_h = math.log(1.0 / hazard_weeks)
    log_1mh = math.log(1.0 - 1.0 / hazard_weeks)

    # Vague prior: mean 0, weak confidence, variance of the right order.
    mu0, kappa0, alpha0 = 0.0, 1.0, 1.0
    beta0 = alpha0 * scale**2

    log_r = np.array([0.0])
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    for x in data:
        df = 2.0 * alpha
        var = beta * (kappa + 1.0) / (alpha * kappa)
        log_pred = (
            gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * np.log(df * math.pi * var)
            - (df + 1) / 2 * np.log1p((x - mu) ** 2 / (df * var))
        )
        growth = log_r + log_pred + log_1mh
        change = logsumexp(log_r + log_pred + log_h)
        log_r = np.concatenate([[change], growth])
        log_r -= logsumexp(log_r)

        mu_new = (kappa * mu + x) / (kappa + 1.0)
        beta_new = beta + kappa * (x - mu) ** 2 / (2.0 * (kappa + 1.0))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa + 1.0])
        alpha = np.concatenate([[alpha0], alpha + 0.5])
        beta = np.concatenate([[beta0], beta_new])

    # Run length r counts the observations absorbed since the last change. A
    # change just before the first live week leaves r equal to the live length
    # exactly, so the sum runs to *and including* it -- that is the single most
    # likely changepoint when live trading is what changed things, and an
    # earlier version of this line left it out.
    probabilities = np.exp(log_r)
    return float(np.clip(probabilities[: recent.size + 1].sum(), 0.0, 1.0))


# -- 3. robust trend ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrendCheck:
    weekly_slope: float
    low: float
    high: float
    weeks: int
    judged: bool

    @property
    def significantly_negative(self) -> bool:
        return self.judged and self.high < 0


def trend_check(
    live_returns: Sequence[float], min_weeks: int = 26, paths: int = 2000, block: float = 4.0
) -> TrendCheck | None:
    """The live equity trend, with an interval that means what it says.

    The slope of log equity against time *is* the mean weekly log return, so it
    is estimated from the returns — robustly, as their median — and its interval
    comes from a stationary bootstrap of those returns.

    The obvious alternative, and the expert's suggestion, is a robust regression
    line through the equity curve itself. It was implemented first and it fails
    on healthy data: a cumulative curve is a random walk, its residuals around
    any line are strongly autocorrelated, and a regression that assumes they are
    not reports an interval several times too narrow. On a healthy synthetic
    strategy it declared a significant negative trend often enough to halt a
    working system. ``test_a_healthy_trend_is_not_negative`` is the regression.
    """
    live = np.log1p(np.asarray(live_returns, dtype=float))
    if live.size < 3:
        return None
    slope = float(np.median(live))
    if live.size >= 20:
        resampled = stationary_bootstrap(live, live.size, paths, block, seed=11)
        medians = np.median(resampled, axis=1)
        low, high = (float(q) for q in np.quantile(medians, [0.025, 0.975]))
    else:
        low, high = float(live.min()), float(live.max())
    return TrendCheck(
        weekly_slope=slope, low=low, high=high,
        weeks=int(live.size), judged=live.size >= min_weeks,
    )


# -- 4. implementation shortfall ---------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One fill against the price the decision was made at."""

    rotation: str
    instrument: str
    side: str  # "buy" | "sell"
    quantity: float
    decision_price: float
    fill_price: float
    commission: float
    reference_open: float | None = None  # the bar open the backtest would have used
    price_after_1w: float | None = None
    price_after_4w: float | None = None

    @property
    def direction(self) -> float:
        return 1.0 if self.side == "buy" else -1.0

    @property
    def notional(self) -> float:
        return self.quantity * self.decision_price

    @property
    def shortfall(self) -> float:
        """Cost against the decision price, as a fraction. Positive is worse."""
        return self.direction * (self.fill_price - self.decision_price) / self.decision_price

    @property
    def execution_drag(self) -> float | None:
        """Cost against the open the backtest assumed. Isolates execution quality."""
        if not self.reference_open:
            return None
        return self.direction * (self.fill_price - self.reference_open) / self.reference_open

    def markout(self, weeks: int) -> float | None:
        later = self.price_after_1w if weeks == 1 else self.price_after_4w
        if not later:
            return None
        return self.direction * (later - self.fill_price) / self.fill_price


@dataclass(frozen=True, slots=True)
class ShortfallCheck:
    per_rotation_bps: Mapping[str, float]
    mean_bps: float
    modeled_bps: float
    ratio: float
    execution_drag_bps: float | None
    alpha_share_by_rotation: Mapping[str, float]
    consecutive_over_half: int
    markout_1w_bps: float | None
    markout_4w_bps: float | None
    fills: int


def shortfall_check(
    records: Sequence[ExecutionRecord],
    modeled_bps: float,
    expected_rotation_return: float,
    sleeve_equity: float,
) -> ShortfallCheck | None:
    """Execution cost per rotation, against the model and against the edge."""
    if not records:
        return None
    rotations: dict[str, list[ExecutionRecord]] = {}
    for record in records:
        rotations.setdefault(record.rotation, []).append(record)

    per_rotation: dict[str, float] = {}
    alpha_share: dict[str, float] = {}
    for rotation, items in rotations.items():
        notional = sum(r.notional for r in items)
        cost = sum(r.notional * r.shortfall + r.commission for r in items)
        per_rotation[rotation] = 10_000.0 * cost / notional if notional else 0.0
        if expected_rotation_return > 0 and sleeve_equity > 0:
            alpha_share[rotation] = (cost / sleeve_equity) / expected_rotation_return
    streak = 0
    for rotation in reversed(list(rotations)):
        if alpha_share.get(rotation, 0.0) > 0.5:
            streak += 1
        else:
            break

    def weighted(values: list[tuple[float, float]]) -> float | None:
        total = sum(w for _, w in values)
        return 10_000.0 * sum(v * w for v, w in values) / total if total else None

    total_notional = sum(r.notional for r in records)
    mean = (
        10_000.0 * sum(r.notional * r.shortfall + r.commission for r in records) / total_notional
        if total_notional else 0.0
    )
    return ShortfallCheck(
        per_rotation_bps=per_rotation,
        mean_bps=mean,
        modeled_bps=modeled_bps,
        ratio=mean / modeled_bps if modeled_bps > 0 else float("inf"),
        execution_drag_bps=weighted(
            [(r.execution_drag, r.notional) for r in records if r.execution_drag is not None]
        ),
        alpha_share_by_rotation=alpha_share,
        consecutive_over_half=streak,
        markout_1w_bps=weighted(
            [(r.markout(1), r.notional) for r in records if r.markout(1) is not None]
        ),
        markout_4w_bps=weighted(
            [(r.markout(4), r.notional) for r in records if r.markout(4) is not None]
        ),
        fills=len(records),
    )


# -- process metrics ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProcessMetrics:
    """How the person and the system are working together.

    Asymmetry is the one to watch: buy compliance minus sell compliance. A
    sustained positive number means buys are executed and sells are not — the
    system proposes, and a person quietly overrules it on the way out, which is
    the most expensive habit a discretionary override can have.
    """

    proposals: int
    approved: int
    rejected: int
    expired: int
    buy_compliance: float | None
    sell_compliance: float | None
    median_latency_hours: float | None
    override_cost: float | None
    stop_coverage: float | None

    @property
    def asymmetry(self) -> float | None:
        if self.buy_compliance is None or self.sell_compliance is None:
            return None
        return self.buy_compliance - self.sell_compliance


# -- the verdict -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Thresholds:
    reduce_percentile: float = 0.80
    halt_percentile: float = 0.99
    reduce_break_probability: float = 0.20
    halt_break_probability: float = 0.50
    reduce_shortfall_multiple: float = 1.5
    halt_shortfall_alpha_share: float = 0.5
    halt_shortfall_cycles: int = 2


@dataclass(frozen=True, slots=True)
class Assessment:
    """Every check, and the state they recommend together."""

    recommended: DegradationState
    reasons: tuple[str, ...]
    drawdown: DrawdownCheck | None
    break_probability: float | None
    trend: TrendCheck | None
    shortfall: ShortfallCheck | None
    weeks: int
    notes: tuple[str, ...] = field(default_factory=tuple)


#: The expert's defaults. Frozen, so one shared instance is safe.
DEFAULT_THRESHOLDS = Thresholds()


def assess(
    backtest_returns: Sequence[float],
    live_returns: Sequence[float],
    records: Sequence[ExecutionRecord],
    modeled_bps: float,
    expected_rotation_return: float,
    sleeve_equity: float,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    paths: int = 5000,
    block: float = 6.0,
    hazard_weeks: float = 250.0,
    trend_min_weeks: int = 26,
) -> Assessment:
    """Run every check and map the results onto the degradation ladder."""
    t = thresholds
    state = DegradationState.NORMAL
    reasons: list[str] = []
    notes: list[str] = []

    def at_least(level: DegradationState, reason: str) -> None:
        nonlocal state
        state = state.worst(level)
        reasons.append(f"{level.value}: {reason}")

    drawdown = drawdown_check(backtest_returns, live_returns, paths, block)
    if drawdown is not None:
        if drawdown.percentile > t.halt_percentile:
            at_least(DegradationState.HALTED,
                     f"drawdown {drawdown.live_drawdown:.1%} is deeper than "
                     f"{drawdown.percentile:.0%} of {drawdown.weeks}-week backtest paths")
        elif drawdown.percentile > t.reduce_percentile:
            at_least(DegradationState.REDUCE_ONLY,
                     f"drawdown {drawdown.live_drawdown:.1%} is deeper than "
                     f"{drawdown.percentile:.0%} of {drawdown.weeks}-week backtest paths")
        if drawdown.live_return < drawdown.band_low_1:
            at_least(DegradationState.HALTED,
                     f"return {drawdown.live_return:+.1%} is below the 1% prediction band "
                     f"({drawdown.band_low_1:+.1%})")
        elif drawdown.live_return < drawdown.band_low_10:
            at_least(DegradationState.REDUCE_ONLY,
                     f"return {drawdown.live_return:+.1%} is below the 10% prediction band "
                     f"({drawdown.band_low_10:+.1%})")

    live = list(live_returns)
    probability = changepoint_probability(backtest_returns, live, hazard_weeks) if live else None
    if probability is not None:
        if probability > t.halt_break_probability:
            at_least(DegradationState.HALTED,
                     f"changepoint probability {probability:.0%} since going live")
        elif probability >= t.reduce_break_probability:
            at_least(DegradationState.REDUCE_ONLY,
                     f"changepoint probability {probability:.0%} since going live")

    trend = trend_check(live, trend_min_weeks)
    if trend is not None:
        if trend.significantly_negative:
            at_least(DegradationState.HALTED,
                     f"equity trend is significantly negative over {trend.weeks} weeks")
        elif not trend.judged:
            notes.append(f"trend not judged before {trend_min_weeks} weeks ({trend.weeks} so far)")

    shortfall = shortfall_check(records, modeled_bps, expected_rotation_return, sleeve_equity)
    if shortfall is not None:
        if shortfall.consecutive_over_half >= t.halt_shortfall_cycles:
            at_least(DegradationState.HALTED,
                     f"execution cost exceeded half the expected return for "
                     f"{shortfall.consecutive_over_half} rotations in a row")
        elif shortfall.ratio >= t.reduce_shortfall_multiple:
            at_least(DegradationState.REDUCE_ONLY,
                     f"execution cost {shortfall.mean_bps:.0f} bps is {shortfall.ratio:.1f}x "
                     f"the {shortfall.modeled_bps:.0f} bps the backtest assumed")

    if len(live) < 13:
        notes.append(
            f"{len(live)} live weeks: every statistic here is wide. The ladder uses "
            f"distributions for exactly this reason, but read it as an early warning."
        )
    return Assessment(
        recommended=state, reasons=tuple(reasons), drawdown=drawdown,
        break_probability=probability, trend=trend, shortfall=shortfall,
        weeks=len(live), notes=tuple(notes),
    )
