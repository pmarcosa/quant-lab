"""Turn results into report documents.

The only place that knows both what a run or a monitoring pass *is* and what a
report *contains*. It reads; it never decides. Every number here was computed
by ``validation/`` or by the engine, and is only arranged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from contracts.identifiers import InstrumentId
from contracts.live import DegradationState
from engine.run import RunResult
from reports.document import Chart, Metric, Report, Series, Status, Table, cumulative, drawdown
from runtime.monitor import MonitorReport
from runtime.wiring import Market
from validation.metrics import summarise

_STATE_LEVEL = {
    DegradationState.NORMAL: "good",
    DegradationState.REDUCE_ONLY: "warning",
    DegradationState.HALTED: "critical",
}

_STATE_LABEL = {
    DegradationState.NORMAL: "NORMAL — rotations allowed",
    DegradationState.REDUCE_ONLY: "REDUCE-ONLY — sells and stops only, no new buys",
    DegradationState.HALTED: "HALTED — no rotations; a person must investigate and clear",
}


# -- backtest --------------------------------------------------------------------


def backtest_report(
    result: RunResult,
    market: Market,
    title: str,
    settings: Mapping[str, Any],
    generated_at: datetime,
    benchmark: str = "SPY",
    benchmark_market: Market | None = None,
) -> Report:
    """A backtest: what it earned, how it fell, against what, and what it held.

    ``benchmark_market``: where to price the benchmark when the strategy's own
    market is restricted to a universe that does not include it.
    """
    curve = result.equity_curve()
    stats = summarise(curve, periods_per_year=market.periods_per_year)
    x = tuple(m.date().isoformat() for m, _ in curve)
    equity = tuple(float(v) for _, v in curve)

    priced = benchmark_market or market
    bench_prices = [priced.window.marks_at(m).get(InstrumentId(benchmark)) for m, _ in curve]
    # The benchmark starts where the strategy's equity is when the benchmark's
    # own history begins, so the two lines share a starting point even when the
    # benchmark listed (or was added to the data) later.
    start = next((i for i, p in enumerate(bench_prices) if p), None)
    bench_equity: tuple[float | None, ...] = tuple(
        (p / bench_prices[start] * equity[start]) if (p and start is not None and i >= start)
        else None
        for i, p in enumerate(bench_prices)
    )
    bench_curve = [(m, v) for (m, _), v in zip(curve, bench_equity, strict=True) if v]
    bench_stats = (
        summarise(bench_curve, periods_per_year=market.periods_per_year)
        if len(bench_curve) > 2 else None
    )

    rotations = [s for s in result.steps if s.decision.target.diagnostics.get("rotated")]
    commission = sum(f.commission for f in result.all_fills())
    metrics = (
        Metric("CAGR", stats.cagr, "pct",
               f"{benchmark} {bench_stats.cagr:.1%}" + (f" (from {x[start]})" if start else "")
               if bench_stats else ""),
        Metric("Max drawdown", stats.max_drawdown, "pct",
               f"{benchmark} {bench_stats.max_drawdown:.1%}" if bench_stats else ""),
        Metric("Sharpe", stats.sharpe, "num", "annualised mean excess / st.dev."),
        Metric("Sortino", stats.sortino, "num"),
        Metric("Volatility", stats.volatility, "pct", "annualised"),
        Metric("Final equity", stats.final_equity, "money", f"from {equity[0]:,.0f}"),
        Metric("Years", stats.years, "num", f"{stats.periods:,} weekly marks"),
        Metric("Rotations", len(rotations), "int"),
        Metric("Stops fired", result.stops_fired(), "int"),
        Metric("Commission paid", commission, "money"),
    )

    charts = (
        Chart("equity", "Equity, strategy vs benchmark (same starting capital)", x,
              (Series("Strategy", equity), Series(benchmark, bench_equity)), format="money",
              note=("Log scale is not used: read the drawdown chart for the size of falls."
                    + (f" {benchmark} data begins {x[start]}; it starts from the strategy's "
                       f"equity on that date." if start else ""))),
        Chart("drawdown", "Drawdown from the running peak", x,
              (Series("Strategy", drawdown(equity)), Series(benchmark, drawdown(bench_equity))),
              format="pct", baseline=0.0, area=True),
    )

    yearly = _yearly(x, equity, bench_equity)
    last = rotations[-25:]
    holdings = Table(
        "Last rotations: what the strategy chose to hold",
        ("decision", "equity", "holdings"),
        tuple(
            (s.decision.decision_time.date().isoformat(), s.decision.equity,
             ", ".join(f"{i} {w:.0%}" for i, w in sorted(
                 s.decision.target.weights.items(), key=lambda kv: -kv[1])))
            for s in reversed(last)
        ),
        ("text", "money", "text"),
    )
    settings_table = Table("Settings", ("setting", "value"),
                           tuple((k, str(v)) for k, v in settings.items()), ("text", "text"))
    return Report(
        kind="backtest", title=title,
        subtitle=f"{x[0]} to {x[-1]} · {result.run}",
        generated_at=generated_at.isoformat(timespec="seconds"),
        metrics=metrics, charts=charts,
        tables=(yearly, holdings, settings_table),
        notes=(
            "A backtest is one path. Its drawdown is a sample, not a bound: the "
            "monitoring bootstrap is what says how deep a normal fall can go.",
            "Sharpe uses the conventional definition (mean excess return over its "
            f"standard deviation, annualised). The geometric variant is {stats.sharpe_geometric:.2f}.",
        ),
        context={"strategy": str(result.run), "benchmark": benchmark, **{
            k: str(v) for k, v in settings.items()}},
    ).validate()


def _yearly(x, equity, bench) -> Table:
    by_year: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(x):
        by_year[label[:4]].append(i)
    rows = []
    previous_end: int | None = None
    for year in sorted(by_year):
        idx = by_year[year]
        start = previous_end if previous_end is not None else idx[0]
        end = idx[-1]
        strat = equity[end] / equity[start] - 1.0
        b = (bench[end] / bench[start] - 1.0) if bench[end] and bench[start] else None
        rows.append((year, strat, b, strat - b if b is not None else None))
        previous_end = end
    return Table("Calendar-year returns", ("year", "strategy", "benchmark", "difference"),
                 tuple(rows), ("text", "pct", "pct", "pct_signed"),
                 note="The first and last years are partial.")


# -- live monitoring ---------------------------------------------------------------


def live_report(
    report: MonitorReport,
    status: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
    benchmark: str = "SPY",
) -> Report:
    """The live dashboard: state first, then every check that produced it.

    Args:
        report: One monitoring pass.
        status: ``LiveSession.status()`` — positions, stops, cash.
        thresholds: The monitoring settings in force (``asdict(config.monitoring)``),
            so the page shows the lines the checks were judged against.
    """
    a = report.assessment
    unit = a.unit
    reasons = list(a.reasons)
    problems = list(report.health.problems)
    level = _STATE_LEVEL[report.state_after]
    label = _STATE_LABEL[report.state_after]
    if report.state_after != report.state_before:
        label += f" (was {report.state_before.value})"
    if problems:
        reasons += [f"health: {p}" for p in problems]
        if level == "good":
            level, label = "warning", label + " — but the health checks need attention"

    th = dict(thresholds or {})
    reduce_p = float(th.get("reduce_percentile", 0.80))
    halt_p = float(th.get("halt_percentile", 0.99))
    reduce_b = float(th.get("reduce_break_probability", 0.20))
    halt_b = float(th.get("halt_break_probability", 0.50))

    def by(value, warn, crit):
        if value is None:
            return "neutral"
        return "critical" if value >= crit else "warning" if value >= warn else "good"

    dd = a.drawdown
    tr = a.trend
    sf = a.shortfall
    pm = report.process
    metrics = [
        Metric("Sleeve equity", status.get("sleeve_equity"), "money",
               f"cash {status.get('cash', 0):,.0f}"),
        Metric(f"Live {unit}s", a.periods, "int",
               f"since {report.periods[0]}" if report.periods else ""),
        Metric("Live return", dd.live_return if dd else None, "pct_signed",
               f"over the last {dd.periods} {unit}s" if dd else f"no live {unit}s yet"),
        Metric("Live drawdown", -dd.live_drawdown if dd else None, "pct",
               f"deeper than {dd.percentile:.0%} of backtest paths" if dd else "",
               by(dd.percentile if dd else None, reduce_p, halt_p)),
        Metric("Break probability", a.break_probability, "pct",
               f"reduce ≥ {reduce_b:.0%}, halt ≥ {halt_b:.0%}",
               by(a.break_probability, reduce_b, halt_b)),
        Metric(f"Trend (per {unit}, median)", tr.slope if tr else None, "pct_signed",
               (f"CI {tr.low:+.2%} … {tr.high:+.2%}" + ("" if tr.judged else " · not judged yet"))
               if tr else "", "critical" if tr and tr.significantly_negative else "neutral"),
        Metric("Shortfall per rotation", sf.mean_bps if sf else None, "bps",
               f"model {sf.modeled_bps:.1f} bp · ratio {sf.ratio:.2f}" if sf else "no fills yet",
               by(sf.ratio if sf else None, float(th.get("reduce_shortfall_multiple", 1.5)), 1e9)),
        Metric("Execution drag vs open", sf.execution_drag_bps if sf else None, "bps",
               "fill against the bar's open"),
        Metric("Markout 1w / 4w", _pair(sf.markout_1w_bps, sf.markout_4w_bps) if sf else None,
               "text", "price move after the fill"),
        Metric("Approved / proposed", f"{pm.approved} / {pm.proposals}", "text",
               f"{pm.rejected} rejected · {pm.expired} expired"),
        Metric("Compliance asymmetry", pm.asymmetry, "pct_signed",
               "entries executed minus exits executed",
               by(pm.asymmetry, 0.10, 0.25) if pm.asymmetry is not None else "neutral"),
        Metric("Stop coverage", pm.stop_coverage, "pct", "positions with a working stop",
               "good" if pm.stop_coverage == 1.0 else "serious" if pm.stop_coverage is not None
               else "neutral"),
        Metric("Data age", report.health.data_age_days, "days", "last weekly bar",
               "warning" if report.health.data_age_days is None
               or report.health.data_age_days > float(th.get("max_data_age_days", 10)) else "good"),
        Metric("Last sync", report.health.last_sync_hours, "hours",
               f"reconciliation: {report.health.last_reconciliation or 'never'}",
               "critical" if report.health.last_reconciliation == "mismatch" else "neutral"),
    ]

    charts = []
    if report.equity:
        base = report.equity[0]
        series = [Series("Sleeve", tuple(report.equity))]
        if report.benchmark and len(report.benchmark) == len(report.equity) - 1:
            series.append(Series(benchmark, tuple(v * base for v in cumulative(report.benchmark))))
        charts.append(Chart("live-equity", "Sleeve equity vs benchmark", tuple(report.periods),
                            tuple(series), format="money"))
    if report.returns:
        cum = tuple(v - 1.0 for v in cumulative(report.returns))
        bands = ()
        if dd:
            bands = (("backtest P10", dd.band_low_10), ("backtest P1", dd.band_low_1))
        charts.append(Chart(
            "live-return", "Cumulative live return against the backtest's range",
            tuple(report.periods), (Series("Live", cum),), format="pct", baseline=0.0, bands=bands,
            note="Dashed lines: the 10th and 1st percentile of cumulative return that "
                 f"bootstrapped backtest paths reach over the same number of {unit}s.",
        ))
        charts.append(Chart("live-drawdown", "Live drawdown", tuple(report.periods),
                            (Series("Sleeve", drawdown(report.equity)),),
                            format="pct", baseline=0.0, area=True))
    if sf and sf.per_rotation_bps:
        keys = tuple(sf.per_rotation_bps)
        charts.append(Chart(
            "shortfall", "Implementation shortfall per rotation", keys,
            (Series("Measured", tuple(sf.per_rotation_bps[k] for k in keys)),),
            format="bps", baseline=0.0, bands=(("model", sf.modeled_bps),),
        ))

    positions = Table(
        "Positions and protective stops",
        ("instrument", "quantity", "avg cost", "mark", "stop", "stop distance"),
        tuple(
            (p["instrument"], p["quantity"], p["average_cost"], p.get("mark"), p.get("stop"),
             (p["stop"] / p["mark"] - 1.0) if p.get("stop") and p.get("mark") else None)
            for p in status.get("positions", ())
        ),
        ("text", "num", "num", "num", "num", "pct_signed"),
        note=("Stops as recorded in the journal (the monitor runs without the gateway; "
              "`ql live sync` confirms them against the broker). "
              if status.get("stop_source") == "journal" else "Stops as the broker reports them. ")
        + "A position without a stop shows an empty stop column: run `ql live stops`.",
    )
    shortfall_table = Table(
        "Rotations: cost and edge",
        ("rotation", "shortfall", "alpha share"),
        tuple((k, v, sf.alpha_share_by_rotation.get(k)) for k, v in sf.per_rotation_bps.items())
        if sf else (),
        ("text", "bps", "pct"),
        note="Alpha share: shortfall as a fraction of the return a rotation is expected to earn. "
             "Two consecutive rotations above one half halt the system.",
    )
    rules = Table(
        "Thresholds in force",
        ("check", "reduce-only at", "halt at"),
        (
            ("drawdown percentile", reduce_p, halt_p),
            ("break probability", reduce_b, halt_b),
            ("shortfall vs model", f"{th.get('reduce_shortfall_multiple', 1.5)}×",
             f"alpha share > {th.get('halt_shortfall_alpha_share', 0.5)} for "
             f"{th.get('halt_shortfall_cycles', 2)} rotations"),
        ),
        ("text", "pct", "pct"),
    )
    return Report(
        kind="live", title=f"Live monitor — {report.strategy_id or 'strategy'} ({report.mode})",
        subtitle=f"account {status.get('account', '?')} · baseline "
                 f"{report.baseline.strategy_version} ({report.baseline.first_bar} to "
                 f"{report.baseline.last_bar}) · "
                 f"{ {'week': 'weekly', 'day': 'daily', 'hour': 'hourly'}.get(unit, unit)} bars",
        generated_at=report.generated_at.isoformat(timespec="seconds"),
        status=Status(level, label, tuple(reasons)),
        metrics=tuple(metrics), charts=tuple(charts),
        tables=(positions, shortfall_table, rules),
        notes=tuple(a.notes) + (
            "Monitoring can impose and lift REDUCE-ONLY. It can impose HALTED but never lift "
            "it: `ql live clear` with a written reason is the only way out.",
        ),
        context={"strategy_id": report.strategy_id, "interval": report.interval,
                 "mode": report.mode, "state": report.state_after.value,
                 "pending_proposal": status.get("pending_proposal")},
    ).validate()


def _pair(a, b) -> str:
    f = lambda v: "—" if v is None else f"{v:+.0f} bp"  # noqa: E731
    return f"{f(a)} / {f(b)}"
