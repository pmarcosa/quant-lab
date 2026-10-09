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
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from contracts.errors import ContractViolation
from contracts.execution import is_stop_order
from contracts.identifiers import RunId
from contracts.live import DegradationState, EventKind
from risk.rules import (
    GrossExposureLimit,
    NetExposureLimit,
    RiskSupervisor,
    ShortSales,
)
from runtime.live import LiveSession
from runtime.research import run_once
from runtime.strategies import build_strategy
from runtime.wiring import Market
from validation.monitoring import (
    Assessment,
    ExecutionRecord,
    ProcessMetrics,
    Thresholds,
    assess,
)

BASELINE_FORMAT = 3


@dataclass(frozen=True, slots=True)
class Baseline:
    """The backtest that live results are measured against.

    ``returns`` are per bar of the strategy's own interval, from the first bar
    the backtest held anything: the warm-up before a strategy can decide is
    cash, and a run of zero returns in the reference would make every live
    drawdown look unusual.
    """

    strategy_version: str
    settings: dict[str, Any]
    interval: str
    returns: list[float]
    modeled_bps: float
    expected_rotation_return: float
    first_bar: str
    last_bar: str
    built_at: str
    #: Returns per unit of gross exposure, for the leverage rule's tail-risk
    #: estimate until the live record is long enough to carry it.
    base_returns: list[float] = field(default_factory=list)
    #: Value traded per trading step over equity: what "a normal rotation"
    #: looks like, for automation's turnover gate.
    rotation_turnover: list[float] = field(default_factory=list)
    #: Each order's value over equity: the largest order the backtest sent.
    order_shares: list[float] = field(default_factory=list)

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"format": BASELINE_FORMAT, **asdict(self)}, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path) -> Baseline:
        if not path.exists():
            raise ContractViolation(
                "no monitoring baseline yet; build it with `ql monitor baseline`"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.pop("format", None) != BASELINE_FORMAT:
            raise ContractViolation("the baseline file is from another version; rebuild it")
        return cls(**raw)


def baseline_path(session: LiveSession):
    return session.config.baseline_path


def _settings(session: LiveSession) -> dict[str, Any]:
    """Everything that changes what the baseline backtest would produce."""
    c = session.config
    return {
        "strategy": c.strategy.name,
        "params": dict(sorted(c.strategy.params.items())),
        "interval": session.interval.value,
        "stop_distance": c.risk.stop_distance,
        "stop_limit_offset": c.risk.stop_limit_offset,
        "execution": asdict(c.execution),
        "capital": c.sleeve_capital,
        "universe": c.strategy.universe,
        "max_gross": c.risk.max_gross,
        "max_net": c.risk.net_cap,
        "min_net": c.risk.min_net,
        "allow_short": c.risk.allow_short,
        "leverage": asdict(c.leverage),
        "financing": asdict(c.financing),
    }


def _strategy_for_backtest(session: LiveSession, market: Market):
    if session.strategy_factory is not None:
        return session.strategy_factory(session.config)
    s = session.config.strategy
    return build_strategy(s.name, s.params, market)


def build_baseline(session: LiveSession, now: datetime) -> Baseline:
    """Backtest the configured strategy over the store and keep what monitoring needs."""
    c = session.config
    market = session.market()
    strategy = _strategy_for_backtest(session, market)
    stop = c.risk.stop()
    schedule = c.leverage.schedule(c.risk.max_gross)
    supervisor = RiskSupervisor(
        rules=(
            ShortSales(allowed=c.risk.allow_short),
            GrossExposureLimit(c.risk.max_gross),
            NetExposureLimit(c.risk.min_net, c.risk.net_cap),
        ),
        stop=stop,
    )
    result = run_once(
        market, strategy, list(market.schedule), RunId("baseline"),
        # The costs and sizing the live proposal uses, at the sleeve's own
        # capital: with a commission in dollars and whole shares, what a
        # rotation costs depends on how much money it moves.
        c.execution.costs(),
        c.execution.sizing(market.interval, allow_short=c.risk.allow_short),
        supervisor=supervisor,
        leverage=None if schedule.is_static_unlevered else schedule,
        financing=c.financing.model(),
        capital=c.sleeve_capital,
    )
    curve = result.equity_curve()
    equity = np.array([e for _, e in curve], dtype=float)
    returns = equity[1:] / equity[:-1] - 1.0

    # Trading steps: where orders other than stops filled. The reference starts
    # at the first, and rotations are measured between them.
    trading = [
        k for k, step in enumerate(result.steps)
        if any(not is_stop_order(f.client_order_id) for f in step.fills)
    ]
    if not trading:
        raise ContractViolation("the baseline backtest never traded; nothing to compare with")
    first = trading[0]
    returns = returns[first:]

    # What the backtest paid to execute, against the decision price: the number
    # live shortfall is compared with.
    costs, notional = 0.0, 0.0
    for step in result.steps:
        marks = step.decision.marks
        for fill in step.fills:
            mark = marks.get(fill.instrument)
            if not mark or is_stop_order(fill.client_order_id):
                continue
            direction = 1.0 if fill.side.value == "buy" else -1.0
            costs += fill.quantity * mark * direction * (fill.price - mark) / mark + fill.commission
            notional += fill.quantity * mark
    modeled = 10_000.0 * costs / notional if notional else 20.0

    # Expected return of one rotation: the return between consecutive trading
    # steps. Strategy-agnostic -- it reads when the strategy traded, not what
    # its parameters say about when it should.
    #
    # Until 2026-10-09 this compounded ``1 + equity ratio`` instead of the
    # ratio, so every bar counted as a doubling and a four-week rotation
    # "expected" about 1,500%. The halt on execution cost divides by this
    # number, so it could never fire.
    per_rotation = [
        float(equity[e] / equity[b] - 1.0)
        for b, e in zip(trading, trading[1:], strict=False)
        if e > b
    ]
    moments = [m for m, _ in curve]
    return Baseline(
        strategy_version=str(strategy.version),
        settings=_settings(session),
        interval=market.interval.value,
        returns=[float(r) for r in returns],
        modeled_bps=float(modeled),
        expected_rotation_return=float(np.mean(per_rotation)) if per_rotation else 0.0,
        first_bar=moments[first].date().isoformat(),
        last_bar=moments[-1].date().isoformat(),
        built_at=now.isoformat(),
        base_returns=[float(r) for r in result.base_returns()],
        rotation_turnover=[float(t) for t in result.turnover()],
        order_shares=[float(x) for x in result.order_shares()],
    )


def load_baseline(session: LiveSession) -> Baseline:
    baseline = Baseline.load(baseline_path(session))
    if baseline.settings != _settings(session):
        raise ContractViolation(
            "the baseline was built for other strategy or risk settings; rebuild it with "
            "`ql monitor baseline`"
        )
    return baseline


def live_returns(session: LiveSession) -> tuple[list[str], list[float], list[float]]:
    """Sleeve returns per bar, from snapshots: the last snapshot of each data bar.

    External cash -- an adjustment's ``cash_delta`` -- is removed from the bar it
    arrived in, so a correction is not reported as performance.
    """
    by_bar: dict[str, tuple[datetime, float]] = {}
    for event in session.journal.events(EventKind.SNAPSHOT):
        label = datetime.fromisoformat(event.payload["marks_as_of"])
        key = label.date().isoformat() if not session.interval.is_intraday else label.isoformat()
        by_bar[key] = (event.at, float(event.payload["sleeve_equity"]))
    labels = sorted(by_bar)
    flows = [
        (e.at, float(e.payload.get("cash_delta", 0.0)))
        for e in session.journal.events(EventKind.ADJUSTMENT)
    ]
    equity = [by_bar[k][1] for k in labels]
    returns = []
    for i in range(1, len(labels)):
        start, end = by_bar[labels[i - 1]][0], by_bar[labels[i]][0]
        flow = sum(amount for at, amount in flows if start < at <= end)
        returns.append((equity[i] - flow) / equity[i - 1] - 1.0 if equity[i - 1] > 0 else 0.0)
    return labels, equity, returns


#: The weekly name, kept for callers written before intervals were general.
live_weekly_returns = live_returns


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
        # Markouts one and four calendar weeks on, in bars of this interval.
        one = max(int(round(market.interval.bars_per_week)), 1)
        if later:
            reference_open = market.window.opens_at(later[0]).get(_id(p["instrument"]))
        if len(later) >= one:
            after_1w = market.window.marks_at(later[one - 1]).get(_id(p["instrument"]))
        if len(later) >= 4 * one:
            after_4w = market.window.marks_at(later[4 * one - 1]).get(_id(p["instrument"]))
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


def _kind(row: dict[str, Any]) -> str:
    """Whether a proposed order was an entry (new exposure) or an exit.

    Read from the order's reason, which describes exposure; the side does not
    (covering a short is a buy and an exit). Orders journaled before reasons
    were recorded fall back to the long-only reading.
    """
    reason = row.get("reason") or ""
    if reason in ("open", "increase"):
        return "entry"
    if reason:
        return "exit"
    return "entry" if row["side"] == "buy" else "exit"


def process_metrics(session: LiveSession, market: Market, now: datetime) -> ProcessMetrics:
    """How proposals, approvals and fills have lined up so far."""
    journal = session.journal
    proposals = journal.events(EventKind.PROPOSAL)
    approvals = {e.payload["proposal_id"]: e for e in journal.events(EventKind.APPROVAL)}
    rejections = {e.payload["proposal_id"]: e for e in journal.events(EventKind.REJECTION)}
    filled = {e.payload["client_order_id"] for e in journal.events(EventKind.FILL)}

    # One proposal per decision bar counts: the last one made for it.
    final_by_bar: dict[str, Any] = {}
    for event in proposals:
        final_by_bar[event.payload["decision_time"]] = event
    expired = 0
    proposed = {"entry": 0, "exit": 0}
    executed = {"entry": 0, "exit": 0}
    latencies = []
    for event in final_by_bar.values():
        pid = event.payload["proposal_id"]
        if pid not in approvals and pid not in rejections:
            if now >= datetime.fromisoformat(event.payload["expires_at"]):
                expired += 1
            else:
                continue  # still open: not yet a decision either way
        for row in event.payload["intents"]:
            kind = _kind(row)
            proposed[kind] += 1
            if row["client_order_id"] in filled:
                executed[kind] += 1
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

    def ratio(kind: str) -> float | None:
        return executed[kind] / proposed[kind] if proposed[kind] else None

    return ProcessMetrics(
        proposals=len(proposals), approved=len(approvals), rejected=len(rejections),
        expired=expired, entry_compliance=ratio("entry"), exit_compliance=ratio("exit"),
        median_latency_hours=statistics.median(latencies) if latencies else None,
        override_cost=override_cost, stop_coverage=coverage,
    )


@dataclass(frozen=True, slots=True)
class Health:
    data_age_hours: float | None
    last_sync_hours: float | None
    last_reconciliation: str | None
    pending_proposal: str | None
    open_findings: tuple[str, ...]
    data_age_limit_hours: float = 240.0
    sync_limit_hours: float = 240.0

    @property
    def data_age_days(self) -> float | None:
        return None if self.data_age_hours is None else self.data_age_hours / 24

    @property
    def problems(self) -> tuple[str, ...]:
        found = []
        if self.data_age_hours is None or self.data_age_hours > self.data_age_limit_hours:
            found.append("price data is stale; run `ql data refresh`")
        if self.last_sync_hours is None or self.last_sync_hours > self.sync_limit_hours:
            found.append(
                f"no sync for over {self.sync_limit_hours / 24:.0f} days; run `ql live sync`"
            )
        if self.last_reconciliation == "mismatch":
            found.append("reconciliation mismatch outstanding")
        return tuple(found)


def health(session: LiveSession, market: Market, now: datetime) -> Health:
    journal = session.journal
    snapshot = journal.last(EventKind.SNAPSHOT)
    reconciliation = journal.last(EventKind.RECONCILIATION)
    pending = session.pending_proposal()
    interval = market.interval
    return Health(
        data_age_hours=(now - market.schedule[-1]).total_seconds() / 3600 if market.schedule else None,
        last_sync_hours=(now - snapshot.at).total_seconds() / 3600 if snapshot else None,
        last_reconciliation=reconciliation.payload["status"] if reconciliation else None,
        pending_proposal=pending.payload["proposal_id"] if pending else None,
        open_findings=tuple(reconciliation.payload["findings"]) if reconciliation else (),
        data_age_limit_hours=session.config.monitoring.data_age_hours(interval),
        # One bar plus three days: a weekly strategy synced every Saturday, a
        # daily one every session, both allowed a long weekend.
        sync_limit_hours=interval.duration.total_seconds() / 3600 + 72,
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
    periods: tuple[str, ...]
    equity: tuple[float, ...]
    returns: tuple[float, ...]
    baseline: Baseline
    benchmark: tuple[float, ...] = ()
    strategy_id: str = ""
    interval: str = "1Week"

    @property
    def weeks(self) -> tuple[str, ...]:
        """The older name for :attr:`periods`."""
        return self.periods


def bars(weeks: float, interval) -> float:
    """A calendar duration in weeks, as a number of bars of ``interval``."""
    return float(weeks) * interval.bars_per_week


def run_monitor(session: LiveSession, apply: bool = True, benchmark: str = "SPY") -> MonitorReport:
    """Every check, the resulting state, and everything a report needs.

    The monitoring settings are calendar durations; they are converted into
    bars of the strategy's interval here, following the expert's rule that the
    market's memory and the expected time between regimes are fixed in
    calendar time, not in observations.
    """
    now = session.clock()
    baseline = load_baseline(session)
    market = session.market()
    interval = market.interval
    labels, equity, returns = live_returns(session)
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
        baseline.returns, returns, records,
        modeled_bps=baseline.modeled_bps,
        expected_rotation_return=baseline.expected_rotation_return,
        sleeve_equity=sleeve_equity, thresholds=thresholds,
        paths=m.bootstrap_paths,
        block=max(bars(m.bootstrap_block_weeks, interval), 1.0),
        hazard_bars=max(bars(m.changepoint_hazard_weeks, interval), 2.0),
        trend_min_bars=max(int(math.ceil(bars(m.trend_min_weeks, interval))), 3),
        horizon=max(int(round(bars(m.horizon_weeks, interval))), 1),
        unit=interval.noun,
    )
    before = session.state()
    after = (
        session.apply_assessment(assessment.recommended, assessment.reasons)
        if apply and session.journal.is_open else before
    )
    bench = _benchmark_returns(market, labels, benchmark)
    return MonitorReport(
        generated_at=now, mode=session.config.mode.value,
        state_before=before, state_after=after, assessment=assessment,
        process=process_metrics(session, market, now), health=health(session, market, now),
        periods=tuple(labels), equity=tuple(equity), returns=tuple(returns),
        baseline=baseline, benchmark=bench,
        strategy_id=session.config.strategy_id, interval=interval.value,
    )


def _benchmark_returns(market: Market, labels: Sequence[str], ticker: str) -> tuple[float, ...]:
    """The benchmark's returns over the same bars, where the data has it."""
    intraday = market.interval.is_intraday
    index = {(m.isoformat() if intraday else m.date().isoformat()): m for m in market.schedule}
    closes = []
    for label in labels:
        moment = index.get(label)
        price = market.window.marks_at(moment).get(_id(ticker)) if moment else None
        closes.append(price)
    out = []
    for a, b in zip(closes, closes[1:], strict=False):
        out.append(b / a - 1.0 if a and b else 0.0)
    return tuple(out)
