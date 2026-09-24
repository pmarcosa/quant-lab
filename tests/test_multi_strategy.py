"""Several strategies, one account each, every order carrying its strategy's id.

The design (manual, section 7): each deployed strategy has its own config file,
its own IBKR account, its own journal, and an id written on every order it
sends. These tests pin down the parts of that which stop one strategy's state
or orders from being mistaken for another's.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from contracts.errors import ContractViolation, StateIntegrityError
from contracts.execution import (
    Side,
    client_order_id,
    is_valid_strategy_id,
    order_prefix,
    split_legs,
    strategy_tag,
)
from contracts.identifiers import InstrumentId, PortfolioId, RunId, TenantId
from contracts.live import TradingMode
from runtime.config import (
    LiveConfig,
    check_accounts,
    config_paths,
    interval_of,
    load_config,
)

AT = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
TENANT = TenantId("user")

YAML = """
strategy_id: {sid}
mode: paper
account: {account}
sleeve_capital: 50000
strategy:
  name: weekly-momentum
  params: {{rebalance_weeks: 4, top_n: 4, lookback_weeks: 13}}
state_dir: {state}
"""


def write(root: Path, sid: str, account: str, name: str | None = None) -> Path:
    folder = root / "configs" / "strategies"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name or sid}.yaml"
    path.write_text(YAML.format(sid=sid, account=account, state=root / "state"))
    return path


# -- the id on every order ------------------------------------------------------------


@pytest.mark.parametrize("value", ["momentum", "mom-2", "a", "x" * 16, "trend-fx-daily"])
def test_valid_strategy_ids(value):
    assert is_valid_strategy_id(value)


@pytest.mark.parametrize(
    "value", ["", "Momentum", "2mom", "mom--x", "mom-", "mom_x", "mom x", "x" * 17, "-mom"]
)
def test_invalid_strategy_ids(value):
    assert not is_valid_strategy_id(value)


def test_every_order_id_starts_with_its_strategy():
    book = PortfolioId(TENANT, "momentum")
    oid = client_order_id(RunId("momentum-paper"), book, InstrumentId("AAA"), AT, Side.BUY, 10)
    assert order_prefix(book) == "ql-momentum."
    assert oid.startswith("ql-momentum.")
    assert len(oid) == len("ql-momentum.") + 20


def test_two_strategies_never_share_an_order_id_prefix():
    a = order_prefix(PortfolioId(TENANT, "trend"))
    b = order_prefix(PortfolioId(TENANT, "trend-fx"))
    assert not b.startswith(a) and not a.startswith(b), (
        "the dot keeps 'trend' from claiming 'trend-fx' orders"
    )


def test_a_free_text_book_name_is_reduced_to_a_tag():
    assert strategy_tag(PortfolioId(TENANT, "momentum_backtest.2020")) == "momentum-backtes"
    assert strategy_tag(PortfolioId(TENANT, "trend__fx")) == "trend-fx"


def test_the_same_decision_gives_the_same_id_and_another_strategy_a_different_one():
    run = RunId("r")
    args = (InstrumentId("AAA"), AT, Side.BUY, 10)
    one = client_order_id(run, PortfolioId(TENANT, "momentum"), *args)
    assert one == client_order_id(run, PortfolioId(TENANT, "momentum"), *args)
    assert one != client_order_id(run, PortfolioId(TENANT, "other"), *args)


@pytest.mark.parametrize(
    ("held", "side", "qty", "legs"),
    [
        (0, Side.BUY, 10, (0, 10)),
        (100, Side.SELL, 30, (30, 0)),
        (100, Side.SELL, 130, (100, 30)),  # a long reversed into a short
        (-50, Side.BUY, 20, (20, 0)),  # a partial cover
        (-50, Side.BUY, 80, (50, 30)),  # a short reversed into a long
        (-50, Side.SELL, 10, (0, 10)),  # a larger short
    ],
)
def test_an_order_splits_into_the_part_that_closes_and_the_part_that_opens(held, side, qty, legs):
    assert split_legs(held, side, qty) == legs


# -- one config per strategy ----------------------------------------------------------


def test_a_config_without_a_strategy_id_is_refused(tmp_path):
    path = tmp_path / "live.yaml"
    path.write_text("mode: paper\naccount: DU1\nsleeve_capital: 1000\n")
    with pytest.raises(ContractViolation, match="strategy_id is required"):
        load_config(path)


def test_a_malformed_strategy_id_is_refused(tmp_path):
    path = write(tmp_path, "Momentum_1", "DU1234567", name="bad")
    with pytest.raises(ContractViolation, match="strategy_id 'Momentum_1'"):
        load_config(path)


def test_an_unknown_strategy_name_is_refused(tmp_path):
    path = write(tmp_path, "momentum", "DU1234567")
    path.write_text(path.read_text().replace("weekly-momentum", "no-such-strategy"))
    with pytest.raises(ContractViolation, match="no-such-strategy"):
        load_config(path)


def test_the_old_flat_momentum_settings_still_load(tmp_path):
    path = tmp_path / "live.yaml"
    path.write_text(
        "strategy_id: momentum\nmode: paper\naccount: DU1234567\nsleeve_capital: 1000\n"
        "strategy: {rebalance_weeks: 2, top_n: 3, lookback_weeks: 13}\n"
        "monitoring: {max_data_age_days: 3}\n"
    )
    config = load_config(path)
    assert config.strategy.name == "weekly-momentum"
    assert config.strategy.params["top_n"] == 3
    assert config.monitoring.max_data_age_hours == 72


def test_each_strategy_keeps_its_state_in_its_own_folder(tmp_path):
    config = LiveConfig(strategy_id="momentum", mode=TradingMode.PAPER, account="DU1",
                        sleeve_capital=1.0, state_dir=tmp_path)
    assert config.journal_path == tmp_path / "live" / "momentum" / "paper-journal.jsonl"
    assert config.baseline_path == tmp_path / "live" / "momentum" / "paper-baseline.json"
    assert config.reports_dir == tmp_path / "reports" / "momentum"
    assert interval_of(config).value == "1Week"


def test_configs_are_found_in_the_strategies_folder(tmp_path):
    write(tmp_path, "beta", "DU2222222")
    write(tmp_path, "alpha", "DU1111111")
    names = [p.stem for p in config_paths(tmp_path / "configs")]
    assert names == ["alpha", "beta"]


def test_two_strategies_may_not_share_an_id(tmp_path):
    a = load_config(write(tmp_path, "alpha", "DU1111111"))
    b = load_config(write(tmp_path, "alpha", "DU2222222", name="copy"))
    with pytest.raises(ContractViolation, match="used by two paper configs"):
        check_accounts([a, b])


def test_two_strategies_may_not_share_an_account(tmp_path):
    a = load_config(write(tmp_path, "alpha", "DU1111111"))
    b = load_config(write(tmp_path, "beta", "DU1111111"))
    with pytest.raises(ContractViolation, match="own IBKR account"):
        check_accounts([a, b])


def test_a_strategy_may_have_a_paper_and_a_live_config(tmp_path):
    """Same id, two modes: journals, accounts and orders are already apart."""
    paper = load_config(write(tmp_path, "alpha", "DU1111111", name="alpha-paper"))
    live = write(tmp_path, "alpha", "U1111111")
    live.write_text(live.read_text().replace("mode: paper", "mode: live")
                    .replace("state_dir", "gateway: {port: 4001}\nstate_dir"))
    live = load_config(live)
    check_accounts([paper, live])
    assert paper.journal_path.parent == live.journal_path.parent
    assert paper.journal_path != live.journal_path


def test_the_strategy_flag_names_the_config_file(tmp_path):
    cli = _Cli(tmp_path)
    write(tmp_path, "alpha", "DU1111111", name="alpha-paper")
    live = write(tmp_path, "alpha", "U1111111")
    live.write_text(live.read_text().replace("mode: paper", "mode: live")
                    .replace("state_dir", "gateway: {port: 4001}\nstate_dir"))
    code, out = cli.run("--strategy", "alpha-paper", "live", "status", "--offline")
    assert code == 0 and "alpha" in out and "paper" in out


def test_two_strategies_in_two_accounts_are_fine(tmp_path):
    a = load_config(write(tmp_path, "alpha", "DU1111111"))
    b = load_config(write(tmp_path, "beta", "DU2222222"))
    check_accounts([a, b])


# -- a journal belongs to one strategy, in one account -----------------------------


def _opened_session(tmp_path, strategy_id="momentum", account="DU1234567"):
    pytest.importorskip("ib_async")
    from execution.ibkr import IBKRBroker
    from runtime.live import LiveSession
    from tests.fake_gateway import FakeGateway
    from tests.live_fixtures import Clock, build_market

    if not (tmp_path / "store").exists():
        build_market(tmp_path / "store")
    config = LiveConfig(strategy_id=strategy_id, mode=TradingMode.PAPER, account=account,
                        sleeve_capital=100_000.0, state_dir=tmp_path / "state")
    broker = IBKRBroker(FakeGateway(accounts=(account,)), account, TradingMode.PAPER,
                        settle_seconds=0, order_prefix=f"ql-{strategy_id}.")
    session = LiveSession(config, broker, tmp_path / "store",
                          clock=Clock(datetime(2026, 9, 19, 10, tzinfo=timezone.utc)))
    session.init()
    return session


def test_a_journal_moved_under_another_strategy_is_refused(tmp_path):
    from runtime.live import LiveSession

    session = _opened_session(tmp_path)
    other = LiveConfig(strategy_id="other", mode=TradingMode.PAPER, account="DU1234567",
                       sleeve_capital=1.0, state_dir=tmp_path / "state")
    other.live_dir.mkdir(parents=True)
    shutil.copy(session.config.journal_path, other.journal_path)
    with pytest.raises(StateIntegrityError, match="belongs to strategy 'momentum'"):
        LiveSession(other, None, tmp_path / "store")


def test_a_sleeve_does_not_move_to_another_account(tmp_path):
    from dataclasses import replace

    from runtime.live import LiveSession

    session = _opened_session(tmp_path)
    moved = replace(session.config, account="DU7654321")
    with pytest.raises(StateIntegrityError, match="does not move"):
        LiveSession(moved, None, tmp_path / "store")


def test_a_broker_on_the_wrong_account_is_refused(tmp_path):
    pytest.importorskip("ib_async")
    from execution.ibkr import IBKRBroker
    from runtime.live import LiveSession
    from tests.fake_gateway import FakeGateway

    config = LiveConfig(strategy_id="momentum", mode=TradingMode.PAPER, account="DU1234567",
                        sleeve_capital=1.0, state_dir=tmp_path)
    broker = IBKRBroker(FakeGateway(accounts=("DU1111111",)), "DU1111111", TradingMode.PAPER,
                        settle_seconds=0)
    with pytest.raises(ContractViolation, match="configured for DU1234567"):
        LiveSession(config, broker, tmp_path)


def test_two_strategies_in_two_accounts_keep_separate_books(tmp_path):
    first = _opened_session(tmp_path, "alpha", "DU1111111")
    second = _opened_session(tmp_path, "beta", "DU2222222")
    assert first.config.journal_path != second.config.journal_path
    assert first.order_prefix == "ql-alpha." and second.order_prefix == "ql-beta."
    assert first.run != second.run


# -- the command line with several strategies ----------------------------------------


class _Cli:
    def __init__(self, tmp_path):
        pytest.importorskip("ib_async")
        from tests.live_fixtures import build_market

        self.root = tmp_path
        build_market(tmp_path / "store")
        self.lines: list[str] = []

    def run(self, *argv):
        from runtime.cli import Context, main

        start = len(self.lines)
        context = Context(
            store=self.root / "store", cache=self.root / "cache",
            configs_root=self.root / "configs", out=self.lines.append,
            broker_factory=lambda c: pytest.fail("no command here should connect"),
        )
        code = main(list(argv), context)
        return code, "\n".join(self.lines[start:])


def test_the_cli_asks_which_strategy_when_there_are_several(tmp_path):
    cli = _Cli(tmp_path)
    write(tmp_path, "alpha", "DU1111111")
    write(tmp_path, "beta", "DU2222222")
    code, out = cli.run("live", "status", "--offline")
    assert code == 2 and "--strategy" in out and "alpha" in out and "beta" in out
    code, out = cli.run("--strategy", "beta", "live", "status", "--offline")
    assert code == 0 and "beta" in out


def test_the_cli_names_the_configured_strategies_when_one_is_missing(tmp_path):
    cli = _Cli(tmp_path)
    write(tmp_path, "alpha", "DU1111111")
    code, out = cli.run("--strategy", "gamma", "live", "status", "--offline")
    assert code == 2 and "gamma" in out and "alpha" in out


def test_a_single_strategy_needs_no_flag(tmp_path):
    cli = _Cli(tmp_path)
    write(tmp_path, "alpha", "DU1111111")
    code, out = cli.run("live", "status", "--offline")
    assert code == 0 and "alpha" in out


def test_the_cli_refuses_to_run_one_strategy_while_two_share_an_account(tmp_path):
    cli = _Cli(tmp_path)
    write(tmp_path, "alpha", "DU1111111")
    write(tmp_path, "beta", "DU1111111")
    code, out = cli.run("--strategy", "alpha", "live", "status", "--offline")
    assert code == 2 and "own IBKR account" in out


def test_ql_strategies_lists_every_configured_strategy(tmp_path):
    cli = _Cli(tmp_path)
    write(tmp_path, "alpha", "DU1111111")
    write(tmp_path, "beta", "DU2222222")
    code, out = cli.run("strategies")
    assert code == 0
    rows = {line.split()[0]: line.split() for line in out.splitlines()[2:]}
    assert set(rows) == {"alpha", "beta"}
    assert rows["alpha"][1:6] == ["weekly-momentum", "weekly", "whole-store", "paper", "DU1111111"]
    assert rows["beta"][-2:] == ["not", "opened"]
