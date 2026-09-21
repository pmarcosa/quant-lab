"""The ``ql`` command, end to end against the fake gateway.

Each test drives the CLI exactly as a person would, with the gateway replaced by
the fake one and the clock controlled. The weekly cycle — sync, propose,
approve, fill, sync, monitor — is exercised through the command line only.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("ib_async")

from execution.ibkr import IBKRBroker  # noqa: E402
from runtime.cli import Context, main  # noqa: E402
from tests.fake_gateway import FakeGateway  # noqa: E402
from tests.live_fixtures import Clock, append_week, build_market  # noqa: E402
from tests.test_monitor_wiring import SATURDAY  # noqa: E402

CONFIG = """
mode: paper
account: DU1234567
sleeve_capital: 100000
gateway: {{host: 127.0.0.1, port: 4002, client_id: 5}}
strategy: {{rebalance_weeks: 1, top_n: 2, lookback_weeks: 13}}
risk: {{stop_distance: 0.12}}
monitoring: {{bootstrap_paths: 1000}}
state_dir: {state}
"""


class Harness:
    def __init__(self, tmp_path, answers=()):
        self.root = tmp_path
        self.frames = build_market(tmp_path / "store", weeks=120)
        self.gateway = FakeGateway(cash=150_000.0)
        self.clock = Clock(SATURDAY)
        self.config = tmp_path / "live.yaml"
        self.config.write_text(CONFIG.format(state=tmp_path / "state"))
        self.lines: list[str] = []
        self.answers = list(answers)
        self.connections = 0

    def broker(self, config):
        self.connections += 1
        return IBKRBroker(self.gateway, config.account, config.mode, settle_seconds=0)

    def run(self, *argv: str) -> tuple[int, str]:
        start = len(self.lines)
        context = Context(
            config_path=self.config, store=self.root / "store", cache=self.root / "cache",
            broker_factory=self.broker, clock=lambda: self.clock.now,
            input=lambda prompt: self.answers.pop(0), out=self.lines.append,
        )
        code = main(list(argv), context)
        return code, "\n".join(self.lines[start:])

    def fill_open(self):
        self.gateway.opening_auction({s: float(f["close"].iloc[-1]) for s, f in self.frames.items()})

    def next_week(self):
        self.frames = append_week(self.root / "store", self.frames, self.clock.now)


def proposal_id(text: str) -> str:
    return next(line.split()[1] for line in text.splitlines() if line.startswith("proposal "))


def test_the_weekly_cycle_runs_from_the_command_line(tmp_path):
    h = Harness(tmp_path)
    code, out = h.run("live", "status")
    assert code == 0 and "connected   DU1234567" in out and "not open yet" in out
    code, out = h.run("live", "init")
    assert code == 0 and "opened the paper sleeve" in out

    code, out = h.run("live", "propose")
    assert code == 0 and "ROTATION" in out and "BUY" in out
    pid = proposal_id(out)

    code, out = h.run("live", "approve", pid, "--confirm", pid)
    assert code == 0 and "orders sent" in out

    h.clock.advance(days=2)
    h.fill_open()
    code, out = h.run("live", "sync")
    assert code == 0 and "RECONCILE   OK" in out.upper()

    code, out = h.run("live", "status")
    assert "NORMAL" in out
    rows = [line for line in out.splitlines() if line.startswith(("AAA", "BBB"))]
    assert len(rows) == 2 and all("—" not in r.split()[-1] for r in rows), "stops are shown"

    connections = h.connections
    code, out = h.run("live", "status", "--offline")
    assert code == 0 and "offline" in out
    assert h.connections == connections, "offline status never touches the gateway"


def test_data_status_reports_the_store(tmp_path):
    h = Harness(tmp_path)
    code, out = h.run("data", "status")
    assert code == 0 and "store      120 weeks" in out


def test_approval_needs_the_typed_phrase(tmp_path):
    h = Harness(tmp_path, answers=["yes"])
    h.run("live", "init")
    _, out = h.run("live", "propose")
    pid = proposal_id(out)
    code, out = h.run("live", "approve")  # prompts; the answer is "yes"
    assert code == 1 and "nothing was sent" in out
    assert not h.gateway.trades(), "no order reached the gateway"
    # The right phrase at the prompt sends.
    h.answers.append(pid)
    code, out = h.run("live", "approve")
    assert code == 0 and "orders sent" in out


def test_errors_are_reported_not_raised(tmp_path):
    h = Harness(tmp_path)
    code, out = h.run("live", "propose")
    assert code == 2 and out.startswith("error:")
    code, out = h.run("live", "reject", "nope", "--reason", "short")
    assert code == 2


def test_a_wrong_account_type_is_refused_before_connecting(tmp_path):
    h = Harness(tmp_path)
    h.config.write_text(h.config.read_text().replace("DU1234567", "U1234567"))
    code, out = h.run("live", "status")
    assert code == 2 and "paper" in out
    assert h.connections == 0


def test_pause_halt_and_clear(tmp_path):
    h = Harness(tmp_path)
    h.run("live", "init")
    code, out = h.run("live", "pause", "--reason", "earnings week, holding off")
    assert "REDUCE_ONLY" in out
    code, out = h.run("live", "halt", "--reason", "investigating a data issue")
    assert "HALTED" in out
    code, out = h.run("live", "propose")
    assert code == 2 and "halted" in out.lower()
    code, out = h.run("live", "clear", "--reason", "data issue fixed and verified")
    assert code == 0 and "NORMAL" in out
    code, out = h.run("live", "journal", "--kind", "state_change")
    assert out.count("state_change") >= 3


def test_monitoring_and_the_dashboard_from_the_command_line(tmp_path):
    h = Harness(tmp_path)
    code, out = h.run("monitor", "baseline")
    assert code == 0 and "saved" in out
    h.run("live", "init")
    for _ in range(3):
        _, out = h.run("live", "propose")
        if "BUY" in out or "SELL" in out:
            pid = proposal_id(out)
            h.run("live", "approve", pid, "--confirm", pid)
            h.clock.advance(days=2)
            h.fill_open()
            h.run("live", "sync")
            h.clock.advance(days=5)
        else:
            h.clock.advance(days=7)
        h.next_week()
        h.run("live", "sync")
    code, out = h.run("monitor", "run", "--benchmark", "AAA")
    assert code == 0, out
    assert "state      NORMAL" in out
    page = next(line.split(maxsplit=1)[1] for line in out.splitlines() if line.startswith("report"))
    from pathlib import Path

    html = Path(page).read_text(encoding="utf-8")
    assert "Live monitor" in html and "Positions and protective stops" in html
    data = json.loads(Path(page.replace(".html", ".json")).read_text(encoding="utf-8"))
    positions = next(t for t in data["tables"] if t["title"].startswith("Positions"))
    assert positions["rows"] and all(r[4] is not None for r in positions["rows"])

    code, out = h.run("report", "render", page.replace(".html", ".json"),
                      "--out", str(tmp_path / "again.html"))
    assert code == 0 and (tmp_path / "again.html").exists()
    code, out = h.run("report", "list")
    assert "live-paper" in out


def test_a_backtest_with_a_report_is_recorded_as_a_trial(tmp_path):
    h = Harness(tmp_path)
    ledger = tmp_path / "research.jsonl"
    code, out = h.run("backtest", "--report", "--benchmark", "AAA", "--ledger", str(ledger))
    assert code == 0, out
    assert "CAGR" in out and "report" in out
    assert len(ledger.read_text().splitlines()) == 1
    h.run("backtest", "--ledger", str(ledger))
    assert len(ledger.read_text().splitlines()) == 1, "the same evaluation is counted once"
    h.run("backtest", "--stop", "0", "--ledger", str(ledger))
    assert len(ledger.read_text().splitlines()) == 2


def test_the_incident_playbook_works_as_the_manual_describes(tmp_path):
    """Manual 11.1 and 11.2: a manual sale, the halt, the correction, the exit."""
    h = Harness(tmp_path)
    h.run("live", "init")
    _, out = h.run("live", "propose")
    pid = proposal_id(out)
    h.run("live", "approve", pid, "--confirm", pid)
    h.clock.advance(days=2)
    h.fill_open()
    h.run("live", "sync")

    # You sell AAA yourself, outside the system.
    held = {p.contract.symbol: p.position for p in h.gateway.positions()}
    price = float(h.frames["AAA"]["close"].iloc[-1])
    h.gateway.hold("AAA", 0, 0.0)
    h.gateway.cash += held["AAA"] * price

    code, out = h.run("live", "sync")
    assert code == 1 and "MISMATCH" in out and "HALTED" in out
    code, out = h.run("live", "clear", "--reason", "trying to clear without fixing it")
    assert code == 2 and "mismatch" in out

    # Record the truth, then clear.
    code, out = h.run("live", "adjust", "--instrument", "AAA", "--quantity", "0",
                      "--cash-delta", str(held["AAA"] * price),
                      "--reason", "sold AAA manually; proceeds stay in the sleeve")
    assert code == 0
    code, out = h.run("live", "reconcile")
    assert code == 0 and "MISMATCH" not in out
    code, out = h.run("live", "clear", "--reason", "manual AAA sale recorded; reconciliation ok")
    assert code == 0 and "NORMAL" in out

    # Halt and exit in an orderly way: liquidation still needs the typed code.
    h.run("live", "halt", "--reason", "decided to stop the strategy for now")
    h.run("live", "sync")
    code, out = h.run("live", "propose", "--liquidate")
    assert code == 0 and "LIQUIDATION" in out and "SELL" in out and "BUY" not in out
    pid = proposal_id(out)
    code, out = h.run("live", "approve", pid, "--confirm", pid)
    assert code == 0
    h.clock.advance(days=1)
    h.fill_open()
    h.run("live", "sync")
    _, out = h.run("live", "status")
    assert "HALTED" in out
    assert not [line for line in out.splitlines() if line.startswith("BBB")], "sleeve is flat"
