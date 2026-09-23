"""Automatic sending: off until a person arms it, and fenced by gates when on.

Each test is one of the project expert's requirements for an automatic mode
(see ``runtime/automation.py``), made to fail. The weekly momentum strategy is
driven through the stand-in gateway, as in ``test_live_cycle``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

pytest.importorskip("ib_async")

from contracts.errors import ContractViolation  # noqa: E402
from contracts.live import DegradationState, EventKind, TradingMode  # noqa: E402
from execution.ibkr import IBKRBroker  # noqa: E402
from runtime import automation  # noqa: E402
from runtime.config import (  # noqa: E402
    AutomationSettings,
    LiveConfig,
    RiskSettings,
    StrategySettings,
)
from runtime.live import LiveSession  # noqa: E402
from runtime.monitor import build_baseline  # noqa: E402
from tests.fake_gateway import FakeGateway  # noqa: E402
from tests.live_fixtures import Clock, append_week, build_market  # noqa: E402

SATURDAY = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


def config(tmp_path, mode="full", **auto):
    return LiveConfig(
        strategy_id="momentum",
        mode=TradingMode.PAPER, account="DU1234567", sleeve_capital=100_000.0,
        strategy=StrategySettings(params={"rebalance_weeks": 1, "top_n": 2, "lookback_weeks": 13}),
        risk=RiskSettings(stop_distance=0.12, max_order_fraction=0.6),
        automation=AutomationSettings(mode=mode, **auto),
        state_dir=tmp_path / "state",
    )


def make(tmp_path, cfg=None, baseline=True):
    frames = build_market(tmp_path / "store", weeks=120)
    gateway = FakeGateway(cash=150_000.0)
    gateway.prices = {s: float(f["close"].iloc[-1]) for s, f in frames.items()}
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0,
                        order_prefix="ql-momentum.")
    clock = Clock(SATURDAY)
    session = LiveSession(cfg or config(tmp_path), broker, tmp_path / "store", clock=clock)
    if baseline:
        build_baseline(session, clock.now).save(session.config.baseline_path)
    session.init()
    return session, gateway, clock, frames


def armed(session, scope="full"):
    automation.arm(session, scope, automation.arm_phrase(session, scope))
    return session


def opens(frames):
    return {s: float(f["close"].iloc[-1]) * 1.002 for s, f in frames.items()}


# -- off until armed ------------------------------------------------------------------------


def test_without_arming_the_cycle_proposes_and_sends_nothing(tmp_path):
    session, gateway, *_ = make(tmp_path)
    report = automation.run_cycle(session)
    assert report.ok and report.proposal_id and report.proposed_orders
    assert report.decision is None
    assert gateway.placed == 0
    assert session.pending_proposal() is not None, "left for a person to approve"


def test_the_config_is_a_ceiling_arming_cannot_exceed(tmp_path):
    session, *_ = make(tmp_path, config(tmp_path, mode="manual"), baseline=False)
    with pytest.raises(ContractViolation, match="allows automation up to 'manual'"):
        automation.arm(session, "exits", automation.arm_phrase(session, "exits"))
    session2, *_ = make(tmp_path / "b", config(tmp_path / "b", mode="exits"), baseline=False)
    with pytest.raises(ContractViolation, match="up to 'exits'"):
        automation.arm(session2, "full", automation.arm_phrase(session2, "full"))


def test_arming_needs_the_exact_phrase(tmp_path):
    session, *_ = make(tmp_path, baseline=False)
    with pytest.raises(ContractViolation, match="type exactly: AUTO FULL momentum"):
        automation.arm(session, "full", "yes")
    assert session.automation_scope() is None


# -- armed ----------------------------------------------------------------------------------


def test_armed_for_everything_the_cycle_sends_the_rotation_once(tmp_path):
    session, gateway, clock, frames = make(tmp_path)
    armed(session)
    report = automation.run_cycle(session)
    assert report.ok, report.errors
    assert report.decision and len(report.decision.sent) == report.proposed_orders > 0
    assert not report.decision.held
    assert gateway.placed == report.proposed_orders
    approval = session.journal.last(EventKind.APPROVAL)
    assert approval.payload["by"] == "automation" and approval.payload["partial"] is False
    assert session.pending_proposal() is None

    again = automation.run_cycle(session)
    assert again.proposal_id is None, "one proposal per bar"
    assert gateway.placed == report.proposed_orders, "nothing is sent twice"

    clock.advance(days=2)
    gateway.opening_auction(opens(frames))
    after = automation.run_cycle(session)
    assert after.sync.new_fills == report.proposed_orders
    assert after.sync.stops_placed == report.proposed_orders


def test_armed_for_exits_only_the_exits_leave_and_the_entries_wait(tmp_path):
    session, gateway, clock, frames = make(tmp_path)
    session.sync()
    first = session.propose()
    session.approve(first.proposal_id, first.proposal_id)
    clock.advance(days=2)
    gateway.opening_auction(opens(frames))
    session.sync()
    held = {str(i) for i in session.book().positions}

    armed(session, "exits")
    clock.advance(days=5)
    newcomer = next(s for s in frames if s not in held)
    frames = append_week(tmp_path / "store", frames, clock.now, drift={newcomer: 0.6})
    report = automation.run_cycle(session)
    assert report.ok, report.errors
    d = report.decision
    assert d.sent and d.held, "exits sent, entries held"
    sent = {session.journal.events(EventKind.SUBMISSION)[-1].payload["side"]}
    assert sent == {"sell"}
    event = session.pending_proposal()
    assert event is not None, "the entries wait for a person"
    pid = event.payload["proposal_id"]
    before = gateway.placed
    session.approve(pid, pid)
    assert gateway.placed - before == len(d.held), "a person's approval sends only what is left"
    assert session.pending_proposal() is None


def test_a_halt_disarms_and_arming_waits_for_a_clear(tmp_path):
    session, *_ = make(tmp_path, baseline=False)
    armed(session)
    session.degrade(DegradationState.HALTED, "changepoint probability above one half")
    assert session.automation_scope() is None
    with pytest.raises(ContractViolation, match="halted"):
        automation.arm(session, "full", automation.arm_phrase(session, "full"))


def test_live_money_needs_the_exits_stage_before_everything(tmp_path):
    frames = build_market(tmp_path / "store", weeks=120)
    gateway = FakeGateway(accounts=("U7654321",), cash=150_000.0)
    cfg = replace(config(tmp_path), mode=TradingMode.LIVE, account="U7654321",
                  gateway=replace(config(tmp_path).gateway, port=4001))
    broker = IBKRBroker(gateway, "U7654321", TradingMode.LIVE, settle_seconds=0,
                        order_prefix="ql-momentum.")
    session = LiveSession(cfg, broker, tmp_path / "store", clock=Clock(SATURDAY))
    session.init()
    phrase = automation.arm_phrase(session, "full")
    assert phrase == "LIVE AUTO FULL momentum"
    with pytest.raises(ContractViolation, match="exits stage first"):
        automation.arm(session, "full", phrase)
    automation.arm(session, "full", phrase,
                   override="paper ran eight clean weeks of exits; accepting the risk")
    assert session.journal.last(EventKind.AUTOMATION).payload["override"].startswith("paper")
    del frames


# -- the gates --------------------------------------------------------------------------------


def test_without_a_baseline_entries_are_held_because_turnover_cannot_be_judged(tmp_path):
    session, gateway, *_ = make(tmp_path, baseline=False)
    armed(session)
    report = automation.run_cycle(session)
    d = report.decision
    assert not d.sent, "a first rotation is all entries"
    assert any(g.name == "turnover" and not g.passed for g in d.gates)
    assert gateway.placed == 0


def test_an_order_larger_than_the_automatic_cap_is_held(tmp_path):
    session, gateway, *_ = make(tmp_path, config(tmp_path, max_order_fraction=0.2))
    armed(session)
    d = automation.run_cycle(session).decision
    assert any(g.name == "order size" and not g.passed for g in d.gates)
    assert gateway.placed == 0


def test_a_stale_reconciliation_blocks_everything(tmp_path):
    session, gateway, clock, _ = make(tmp_path)
    session.sync()
    session.propose()
    armed(session)
    clock.advance(hours=2)
    d = automation.auto_approve(session)
    assert any(g.name == "reconciliation" and not g.passed for g in d.gates)
    assert gateway.placed == 0


def test_a_bar_below_the_backtests_tail_halts_once(tmp_path):
    session, gateway, clock, frames = make(tmp_path)
    armed(session)
    automation.run_cycle(session)
    clock.advance(days=2)
    gateway.opening_auction(opens(frames))
    automation.run_cycle(session)
    clock.advance(days=5)
    crash = {s: -0.45 for s in frames}
    frames = append_week(tmp_path / "store", frames, clock.now, drift=crash)
    report = automation.run_cycle(session)
    assert report.breaker and "loss breaker" in report.breaker
    assert session.state() is DegradationState.HALTED
    assert session.automation_scope() is None, "a halt disarms"
    session.clear(DegradationState.NORMAL, "looked at it: a synthetic crash in a test")
    assert automation.loss_breaker(session) is None, "the same bar is not judged twice"


# -- scheduling -------------------------------------------------------------------------------


def test_a_launch_agent_is_written_for_the_bar_size(tmp_path):
    from contracts.temporal import BarInterval

    times = automation.schedule_times(BarInterval.WEEK)
    assert (6, 10, 0) in times, "Saturday morning: refresh, propose, send"
    plist = automation.launchd_plist("com.quantlab.m.cycle", ["/usr/bin/python3", "-m",
                                     "runtime.cli", "live", "cycle"], tmp_path, tmp_path / "log",
                                     times)
    assert "<key>Weekday</key><integer>6</integer>" in plist
    assert "<string>runtime.cli</string>" in plist
    assert len(automation.schedule_times(BarInterval.HOUR)) == 45, "hourly through the session"
    with pytest.raises(ContractViolation, match="minute-bar"):
        automation.schedule_times(BarInterval.MINUTE)


# -- the command line -------------------------------------------------------------------------


def test_the_cycle_and_arming_from_the_command_line(tmp_path):
    from tests.test_cli import Harness

    h = Harness(tmp_path)
    h.config.write_text(h.config.read_text() + "automation: {mode: full}\n")
    assert h.run("live", "init")[0] == 0
    assert h.run("monitor", "baseline")[0] == 0
    code, out = h.run("live", "cycle", "--no-refresh", "--no-monitor")
    assert code == 0 and "needs `ql live approve" in out and not h.gateway.trades()
    code, out = h.run("live", "auto", "arm", "--scope", "full", "--confirm", "nope")
    assert code == 2 and "AUTO FULL momentum" in out
    code, out = h.run("live", "auto", "arm", "--scope", "full", "--confirm", "AUTO FULL momentum")
    assert code == 0 and "armed for full" in out
    code, out = h.run("live", "cycle", "--no-refresh", "--no-monitor")
    assert code == 0 and "automatic  scope full" in out and h.gateway.trades()
    code, out = h.run("live", "auto", "status")
    assert "armed      full" in out
    code, out = h.run("live", "auto", "disarm", "--reason", "going on holiday for two weeks")
    assert code == 0
    code, out = h.run("live", "auto", "schedule")
    assert code == 0 and "launchctl bootstrap" in out and "Sat 10:00" in out
