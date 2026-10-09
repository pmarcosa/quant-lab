"""The monitor over several simulated live weeks, end to end."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("ib_async")

from contracts.errors import ContractViolation  # noqa: E402
from contracts.live import DegradationState, EventKind, TradingMode  # noqa: E402
from execution.ibkr import IBKRBroker  # noqa: E402
from runtime.config import LiveConfig, RiskSettings, StrategySettings  # noqa: E402
from runtime.live import LiveSession  # noqa: E402
from runtime.monitor import (  # noqa: E402
    Baseline,
    baseline_path,
    build_baseline,
    live_weekly_returns,
    load_baseline,
    run_monitor,
)
from tests.fake_gateway import FakeGateway  # noqa: E402
from tests.live_fixtures import STEADY, Clock, append_week, build_market  # noqa: E402

SATURDAY = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


def make_session(tmp_path, weeks=120, **strategy):
    frames = build_market(tmp_path / "store", weeks=weeks)
    gateway = FakeGateway(cash=150_000.0)
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    clock = Clock(SATURDAY)
    config = LiveConfig(
        strategy_id="momentum",
        mode=TradingMode.PAPER, account="DU1234567", sleeve_capital=100_000.0,
        strategy=StrategySettings(
            params={"rebalance_weeks": 1, "top_n": 2, "lookback_weeks": 13, **STEADY, **strategy}
        ),
        risk=RiskSettings(stop_distance=0.12),
        state_dir=tmp_path / "state",
    )
    return LiveSession(config, broker, tmp_path / "store", clock=clock), gateway, clock, frames


def live_weeks(session, gateway, clock, frames, root, n):
    """Run the weekly cycle ``n`` times: refresh, sync, propose, approve, fill."""
    session.init()
    session.sync()
    for _ in range(n):
        proposal = session.propose()
        if proposal.orders:
            session.approve(proposal.proposal_id, proposal.proposal_id)
            clock.advance(days=2)
            gateway.opening_auction({s: float(f["close"].iloc[-1]) for s, f in frames.items()})
            session.sync()
            clock.advance(days=5)
        else:
            clock.advance(days=7)
        frames = append_week(root, frames, clock.now)
        session.sync()
    return frames


def test_monitoring_refuses_to_run_without_a_baseline(tmp_path):
    session, *_ = make_session(tmp_path)
    session.init()
    with pytest.raises(ContractViolation, match="ql monitor baseline"):
        run_monitor(session)


def test_a_baseline_is_built_saved_and_reloaded(tmp_path):
    session, gateway, clock, frames = make_session(tmp_path)
    baseline = build_baseline(session, clock.now)
    baseline.save(baseline_path(session))
    reloaded = load_baseline(session)
    assert reloaded == baseline
    assert len(baseline.returns) > 20
    assert baseline.modeled_bps > 0


def test_the_expected_return_of_a_rotation_is_a_return_not_a_count_of_bars(tmp_path):
    """One rotation of a strategy that makes a few percent a month earns a few percent.

    It was computed by compounding ``1 + equity ratio`` -- every bar a doubling --
    and came out near 1 for a weekly rotation and near 15 for a four-weekly one.
    The halt on execution cost divides the cost by this number, so with the
    doublings in it the halt could never fire.
    """
    session, _, clock, _ = make_session(tmp_path)
    baseline = build_baseline(session, clock.now)
    assert 0.0 < baseline.expected_rotation_return < 0.5


def test_the_baseline_trades_the_way_the_config_says(tmp_path):
    """The config's costs and sizing reach the baseline; changing them makes it stale."""
    from dataclasses import replace

    from runtime.config import ExecutionSettings

    session, _, clock, _ = make_session(tmp_path)
    flat = build_baseline(session, clock.now)
    tiered = ExecutionSettings(commission="ibkr-tiered", cash_buffer=0.0, no_trade_band=0.0)
    session.config = replace(session.config, execution=tiered)
    planned = build_baseline(session, clock.now)
    assert planned.settings["execution"]["commission"] == "ibkr-tiered"
    assert planned.settings != flat.settings, "so the old baseline is refused as stale"
    assert planned.modeled_bps < flat.modeled_bps, "a plan in dollars costs less than 10 bp here"


def test_a_baseline_built_for_other_settings_is_refused(tmp_path):
    session, gateway, clock, frames = make_session(tmp_path)
    build_baseline(session, clock.now).save(baseline_path(session))
    changed, *_ = make_session(tmp_path / "other", top_n=3)
    changed.config.state_dir.mkdir(parents=True, exist_ok=True)
    stale = Baseline.load(baseline_path(session))
    stale.save(baseline_path(changed))
    with pytest.raises(ContractViolation, match="other strategy or risk settings"):
        load_baseline(changed)


def test_several_live_weeks_produce_a_full_report(tmp_path):
    session, gateway, clock, frames = make_session(tmp_path)
    build_baseline(session, clock.now).save(baseline_path(session))
    live_weeks(session, gateway, clock, frames, tmp_path / "store", 4)

    weeks, equity, returns = live_weekly_returns(session)
    assert len(weeks) >= 4
    assert len(returns) == len(weeks) - 1

    report = run_monitor(session)
    assert report.assessment.periods == len(returns)
    assert report.assessment.drawdown is not None
    assert report.assessment.shortfall is not None and report.assessment.shortfall.fills > 0
    assert report.process.entry_compliance == 1.0, "every proposed entry was approved and filled"
    assert report.process.stop_coverage == 1.0
    assert report.health.last_reconciliation == "ok"
    assert report.state_after in tuple(DegradationState)


def test_a_rejected_proposal_is_counted_and_priced(tmp_path):
    session, gateway, clock, frames = make_session(tmp_path)
    build_baseline(session, clock.now).save(baseline_path(session))
    session.init()
    session.sync()
    proposal = session.propose()
    session.reject(proposal.proposal_id, "sitting this week out on purpose")
    clock.advance(days=7)
    append_week(tmp_path / "store", frames, clock.now)
    session.sync()
    report = run_monitor(session)
    assert report.process.rejected == 1
    assert report.process.entry_compliance == 0.0
    assert report.process.override_cost is not None, "what not following the system cost"


def test_monitoring_can_lift_its_own_reduce_only_but_never_a_halt(tmp_path):
    session, *_ = make_session(tmp_path)
    session.init()
    session.apply_assessment(DegradationState.REDUCE_ONLY, ["drawdown in the tail"])
    assert session.state() is DegradationState.REDUCE_ONLY
    session.apply_assessment(DegradationState.NORMAL, [])
    assert session.state() is DegradationState.NORMAL, "the monitor lifted what it imposed"

    session.apply_assessment(DegradationState.HALTED, ["changepoint"])
    session.apply_assessment(DegradationState.NORMAL, [])
    assert session.state() is DegradationState.HALTED, "nothing automatic lifts a halt"


def test_a_reduce_only_a_person_set_is_not_lifted_by_monitoring(tmp_path):
    session, *_ = make_session(tmp_path)
    session.init()
    session.journal.append(EventKind.STATE_CHANGE, session.clock(), {
        "from": "normal", "to": "reduce_only", "reason": "going on holiday", "by": "person",
    })
    session.apply_assessment(DegradationState.NORMAL, [])
    assert session.state() is DegradationState.REDUCE_ONLY
