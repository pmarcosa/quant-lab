"""Gathering what monitoring needs, running it, and applying the result.

Monitoring compares the live sleeve against a **baseline**: the backtest of the
exact strategy and risk settings being traded, over the data in the store. The
baseline is built once — ``ql monitor baseline`` — and saved, because comparing
against a reference that silently changes every time the data grows would make
yesterday's alarm and today's incomparable. Rebuild it deliberately, after a
change of strategy settings or a widening of the universe, and the file records
which settings it was built with so a mismatch is refused rather than used.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

from contracts.errors import ContractViolation
from contracts.identifiers import RunId
from contracts.live import DegradationState, EventKind
from contracts.temporal import BarInterval
from engine.decide import SizingPolicy
from execution.simulated import CostModel
from risk.rules import GrossExposureLimit, ProtectiveStop, RiskSupervisor
from runtime.live import LiveSession
from runtime.research import precompute_indicators, run_once
from runtime.wiring import Market, load_market
from strategies.momentum import MomentumParams, WeeklyMomentum
from validation.monitoring import (
    Assessment,
    ExecutionRecord,
    ProcessMetrics,
    Thresholds,
    assess,
)

BASELINE_FORMAT = 1


@dataclass(frozen=True, slots=True)
class Baseline:
    """The backtest that live results are measured against."""

    strategy_version: str
    settings: dict[str, Any]
    weekly_returns: list[float]
    modeled_bps: float
    expected_rotation_return: float
    first_week: str
    last_week: str
    built_at: str

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": BASELINE_FORMAT, **asdict(self)}, indent=1))

    @classmethod
    def load(cls, path) -> Baseline:
        if not path.exists():
            raise ContractViolation(
                "no monitoring baseline yet; build it with `ql monitor baseline`"
            )
        raw = json.loads(path.read_text())
        if raw.pop("format", None) != BASELINE_FORMAT:
            raise ContractViolation("the baseline file is from another version; rebuild it")
        return cls(**raw)


def baseline_path(session: LiveSession):
    return session.config.state_dir / "live" / f"{session.config.mode.value}-baseline.json"


def _settings(session: LiveSession) -> dict[str, Any]:
    c = session.config
    return {
        "rebalance_weeks": c.strategy.rebalance_weeks, "top_n": c.strategy.top_n,
        "lookback_weeks": c.strategy.lookback_weeks, "stop_distance": c.risk.stop_distance,
        "max_gross": c.risk.max_gross,
    }


def build_baseline(session: LiveSession, now: datetime) -> Baseline:
    """Backtest the configured strategy over the store and keep what monitoring needs."""
    c = session.config
    market = load_market(session.store_root, interval=BarInterval.WEEK)
    params = MomentumParams(
        rebalance_weeks=c.strategy.rebalance_weeks, top_n=c.strategy.top_n,
        lookback_weeks=c.strategy.lookback_weeks,
    )
    strategy = WeeklyMomentum(params, precomputed=precompute_indicators(market, params))
    stop = ProtectiveStop(c.risk.stop_distance) if c.risk.stop_distance > 0 else None
    supervisor = RiskSupervisor(rules=(GrossExposureLimit(c.risk.max_gross),), stop=stop)
    schedule = list(market.schedule)[params.warmup_weeks + params.min_history_weeks:]
    result = run_once(
        market, strategy, schedule, RunId("baseline"),
        CostModel(commission_bps=10.0, slippage_bps=10.0),
        SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.005), supervisor=supervisor,
    )
    equity = np.array([e for _, e in result.equity_curve()], dtype=float)
    returns = equity[1:] / equity[:-1] - 1.0

    # What the backtest paid to execute, against the decision price: the number
    # live shortfall is compared with.
    costs, notional = 0.0, 0.0
    for step in result.steps:
        marks = step.decision.marks
        for fill in step.fills:
            mark = marks.get(fill.instrument)
            if not mark or fill.client_order_id.endswith("-stop"):
                continue
            direction = 1.0 if fill.side.value == "buy" else -1.0
            costs += fill.quantity * mark * direction * (fill.price - mark) / mark + fill.commission
            notional += fill.quantity * mark
    modeled = 10_000.0 * costs / notional if notional else 20.0

    k = c.strategy.rebalance_weeks
    per_rotation = [
        float(np.prod(1.0 + returns[i : i + k]) - 1.0) for i in range(0, len(returns) - k + 1, k)
    ]
    moments = [m for m, _ in result.equity_curve()]
    return Baseline(
        strategy_version=str(strategy.version),
        settings=_settings(session),
        weekly_returns=[float(r) for r in returns],
        modeled_bps=float(modeled),
        expected_rotation_return=float(np.mean(per_rotation)) if per_rotation else 0.0,
        first_week=moments[0].date().isoformat(),
        last_week=moments[-1].date().isoformat(),
        built_at=now.isoformat(),
    )


def load_baseline(session: LiveSession) -> Baseline:
    baseline = Baseline.load(baseline_path(session))
    if baseline.settings != _settings(session):
        raise ContractViolation(
            "the monitoring baseline was built with different strategy or risk settings "
            f"({baseline.settings}); rebuild it with `ql monitor baseline`"
        )
    return baseline


# -- reading the journal -----------------------------------------------------


def live_weekly_returns(session: LiveSession) -> tuple[list[str], list[float], list[float]]:
    """Weekly sleeve returns from snapshots: the last snapshot of each data week.

    External cash — an adjustment's ``cash_delta`` — is removed from the week it
    arrived in, so a correction is not reported as performance.
    """
    by_week: dict[str, tuple[datetime, float]] = {}
    for event in session.journal.events(EventKind.SNAPSHOT):
        week = datetime.fromisoformat(event.payload["marks_as_of"]).date().isoformat()
        by_week[week] = (event.at, float(event.payload["sleeve_equity"]))
    weeks = sorted(by_week)
    flows = [
        (e.at, float(e.payload.get("cash_delta", 0.0)))
        for e in session.journal.events(EventKind.ADJUSTMENT)
    ]
    equity = [by_week[w][1] for w in weeks]
    returns = []
    for i in range(1, len(weeks)):
        start, end = by_week[weeks[i - 1]][0], by_week[weeks[i]][0]
        flow = sum(amount for at, amount in flows if start < at <= end)
        returns.append((equity[i] - flow) / equity[i - 1] - 1.0 if equity[i - 1] > 0 else 0.0)
    return weeks, equity, returns


def execution_records(session: LiveSession, market: Market) -> list[ExecutionRecord]:
    """Every rotation fill, against its decision price and the open the backtest used."""
    proposals = {
        e.payload["proposal_id"]: e for e in session.journal.events(EventKind.PROPOSAL)
    }
    order_to_rotation: dict[str, str] = {}
    for event in session.journal.events(EventKind.APPROVAL):
        for oid in event.payload.get("orders", ()):
            order_to_rotation[oid] = event.payload["proposal_id"]
    schedule = list(market.schedule)
    records = []
    for event in session.journal.events(EventKind.FILL):
        p = event.payload
        rotation = order_to_rotation.get(p["client_order_id"])
        if rotation is None or p.get("stop"):
            continue
        proposal = proposals[rotation].payload
        decision = datetime.fromisoformat(proposal["decision_time"])
        mark = proposal["marks"].get(p["instrument"])
        if not mark:
            continue
        later = [m for m in schedule if m > decision]
        reference_open = after_1w = after_4w = None
        if later:
            reference_open = market.window.opens_at(later[0]).get(_id(p["instrument"]))
        if len(later) >= 1:
            after_1w = market.window.marks_at(later[0]).get(_id(p["instrument"]))
        if len(later) >= 4:
            after_4w = market.window.marks_at(later[3]).get(_id(p["instrument"]))
        records.append(ExecutionRecord(
            rotation=rotation, instrument=p["instrument"], side=p["side"],
            quantity=float(p["quantity"]), decision_price=float(mark),
            fill_price=float(p["price"]), commission=float(p.get("commission", 0.0)),
            reference_open=reference_open, price_after_1w=after_1w, price_after_4w=after_4w,
        ))
    return records


def _id(ticker: str):
    from contracts.identifiers import InstrumentId

    return InstrumentId(ticker)


def process_metrics(session: LiveSession, market: Market, now: datetime) -> ProcessMetrics:
    """How proposals, approvals and fills have lined up so far."""
    journal = session.journal
    proposals = journal.events(EventKind.PROPOSAL)
    approvals = {e.payload["proposal_id"]: e for e in journal.events(EventKind.APPROVAL)}
    rejections = {e.payload["proposal_id"]: e for e in journal.events(EventKind.REJECTION)}
    filled = {e.payload["client_order_id"] for e in journal.events(EventKind.FILL)}

    # One proposal per decision week counts: the last one made for it.
    final_by_week: dict[str, Any] = {}
    for event in proposals:
        final_by_week[event.payload["decision_time"]] = event
    expired = 0
    proposed = {"buy": 0, "sell": 0}
    executed = {"buy": 0, "sell": 0}
    latencies = []
    for event in final_by_week.values():
        pid = event.payload["proposal_id"]
        if pid not in approvals and pid not in rejections:
            if now >= datetime.fromisoformat(event.payload["expires_at"]):
                expired += 1
            else:
                continue  # still open: not yet a decision either way
        for row in event.payload["intents"]:
            proposed[row["side"]] += 1
            if row["client_order_id"] in filled:
                executed[row["side"]] += 1
        if pid in approvals:
            latencies.append((approvals[pid].at - event.at).total_seconds() / 3600)

    override_cost = None
    schedule = list(market.schedule)
    for pid in rejections:
        proposal = next((e for e in proposals if e.payload["proposal_id"] == pid), None)
        if proposal is None:
            continue
        decision = datetime.fromisoformat(proposal.payload["decision_time"])
        later = [m for m in schedule if m > decision]
        if not later:
            continue
        prices = market.window.marks_at(later[0])
        cost = 0.0
        for row in proposal.payload["intents"]:
            mark = proposal.payload["marks"].get(row["instrument"])
            after = prices.get(_id(row["instrument"]))
            if mark and after:
                sign = 1.0 if row["side"] == "buy" else -1.0
                cost += sign * float(row["quantity"]) * (after - mark)
        override_cost = (override_cost or 0.0) + cost

    last_reconciliation = journal.last(EventKind.RECONCILIATION)
    coverage = None
    if last_reconciliation is not None:
        book = session.book()
        unprotected = sum(
            1 for f in last_reconciliation.payload["findings"] if "no protective stop" in f
        )
        coverage = 1.0 - unprotected / len(book.positions) if book.positions else 1.0

    def ratio(side: str) -> float | None:
        return executed[side] / proposed[side] if proposed[side] else None

    return ProcessMetrics(
        proposals=len(proposals), approved=len(approvals), rejected=len(rejections),
        expired=expired, buy_compliance=ratio("buy"), sell_compliance=ratio("sell"),
        median_latency_hours=statistics.median(latencies) if latencies else None,
        override_cost=override_cost, stop_coverage=coverage,
    )


@dataclass(frozen=True, slots=True)
class Health:
    data_age_days: float | None
    last_sync_hours: float | None
    last_reconciliation: str | None
    pending_proposal: str | None
    open_findings: tuple[str, ...]

    @property
    def problems(self) -> tuple[str, ...]:
        found = []
        if self.data_age_days is None or self.data_age_days > 10:
            found.append("price data is stale; run `ql data refresh`")
        if self.last_sync_hours is None or self.last_sync_hours > 24 * 8:
            found.append("no sync for over a week; run `ql live sync`")
        if self.last_reconciliation == "mismatch":
            found.append("reconciliation mismatch outstanding")
        return tuple(found)


def health(session: LiveSession, market: Market, now: datetime) -> Health:
    journal = session.journal
    snapshot = journal.last(EventKind.SNAPSHOT)
    reconciliation = journal.last(EventKind.RECONCILIATION)
    pending = session.pending_proposal()
    return Health(
        data_age_days=(now - market.schedule[-1]).total_seconds() / 86400 if market.schedule else None,
        last_sync_hours=(now - snapshot.at).total_seconds() / 3600 if snapshot else None,
        last_reconciliation=reconciliation.payload["status"] if reconciliation else None,
        pending_proposal=pending.payload["proposal_id"] if pending else None,
        open_findings=tuple(reconciliation.payload["findings"]) if reconciliation else (),
    )


# -- the run -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MonitorReport:
    generated_at: datetime
    mode: str
    state_before: DegradationState
    state_after: DegradationState
    assessment: Assessment
    process: ProcessMetrics
    health: Health
    weeks: tuple[str, ...]
    equity: tuple[float, ...]
    returns: tuple[float, ...]
    baseline: Baseline
    benchmark: tuple[float, ...] = ()


def run_monitor(session: LiveSession, apply: bool = True, benchmark: str = "SPY") -> MonitorReport:
    """Every check, the resulting state, and everything a report needs."""
    now = session.clock()
    baseline = load_baseline(session)
    market = session.market()
    weeks, equity, returns = live_weekly_returns(session)
    records = execution_records(session, market)
    m = session.config.monitoring
    thresholds = Thresholds(
        reduce_percentile=m.reduce_percentile, halt_percentile=m.halt_percentile,
        reduce_break_probability=m.reduce_break_probability,
        halt_break_probability=m.halt_break_probability,
        reduce_shortfall_multiple=m.reduce_shortfall_multiple,
        halt_shortfall_alpha_share=m.halt_shortfall_alpha_share,
        halt_shortfall_cycles=m.halt_shortfall_cycles,
    )
    sleeve_equity = equity[-1] if equity else session.config.sleeve_capital
    assessment = assess(
        baseline.weekly_returns, returns, records,
        modeled_bps=baseline.modeled_bps,
        expected_rotation_return=baseline.expected_rotation_return,
        sleeve_equity=sleeve_equity, thresholds=thresholds,
        paths=m.bootstrap_paths, block=m.bootstrap_block_weeks,
        hazard_weeks=m.changepoint_hazard_weeks, trend_min_weeks=m.trend_min_weeks,
    )
    before = session.state()
    after = (
        session.apply_assessment(assessment.recommended, assessment.reasons)
        if apply and session.journal.is_open else before
    )
    bench = _benchmark_returns(market, weeks, benchmark)
    return MonitorReport(
        generated_at=now, mode=session.config.mode.value,
        state_before=before, state_after=after, assessment=assessment,
        process=process_metrics(session, market, now), health=health(session, market, now),
        weeks=tuple(weeks), equity=tuple(equity), returns=tuple(returns),
        baseline=baseline, benchmark=bench,
    )


def _benchmark_returns(market: Market, weeks: Sequence[str], ticker: str) -> tuple[float, ...]:
    """The benchmark's weekly returns over the same weeks, where the data has it."""
    index = {m.date().isoformat(): m for m in market.schedule}
    closes = []
    for week in weeks:
        moment = index.get(week)
        price = market.window.marks_at(moment).get(_id(ticker)) if moment else None
        closes.append(price)
    out = []
    for a, b in zip(closes, closes[1:], strict=False):
        out.append(b / a - 1.0 if a and b else 0.0)
    return tuple(out)

