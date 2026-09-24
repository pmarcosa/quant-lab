"""Universes as part of a strategy, the whole research line in the trial count, the N/K grid."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from contracts.errors import ContractViolation
from data.universe import UniverseList
from runtime.wiring import load_market, universe_list
from tests.live_fixtures import build_market


def write_universe(tmp_path, name, text):
    path = tmp_path / f"{name}.txt"
    path.write_text(text)
    return path


def test_a_universe_file_is_a_sorted_set_of_tickers(tmp_path):
    path = write_universe(tmp_path, "mine", "# a comment\nbbb  # trailing\nAAA\n\nAAA\n")
    universe = UniverseList.read(path)
    assert (universe.name, universe.symbols) == ("mine", ("AAA", "BBB"))
    other = UniverseList.read(write_universe(tmp_path, "other", "AAA\nBBB\nCCC\n"))
    assert universe.fingerprint != other.fingerprint
    with pytest.raises(ContractViolation, match="not tickers"):
        UniverseList.read(write_universe(tmp_path, "bad", "AAA\nnot a ticker\n"))
    with pytest.raises(ContractViolation, match="no symbols"):
        UniverseList.read(write_universe(tmp_path, "empty", "# nothing\n"))


def test_universes_are_found_by_name_or_path(tmp_path):
    write_universe(tmp_path, "etfs", "XLK\nXLE\n")
    assert universe_list("etfs", directory=tmp_path).symbols == ("XLE", "XLK")
    assert universe_list(str(tmp_path / "etfs.txt")).name == "etfs"
    assert universe_list(None) is None
    with pytest.raises(ContractViolation, match="known: etfs"):
        universe_list("nope", directory=tmp_path)


def test_the_repository_ships_the_sector_etfs():
    etfs = universe_list("sector-etfs")
    assert len(etfs.symbols) == 11 and {"XLK", "XLRE", "XLC"} <= set(etfs.symbols)


def test_a_universe_restricts_the_market_and_names_what_is_missing(tmp_path):
    build_market(tmp_path / "store")
    market = load_market(tmp_path / "store", symbols=("AAA", "BBB", "ZZZ"))
    assert {str(m.instrument) for m in market.universe.memberships()} == {"AAA", "BBB"}
    assert market.missing == ("ZZZ",)
    assert set(market.window.closes.columns) <= {"AAA", "BBB"}
    whole = load_market(tmp_path / "store")
    assert whole.fingerprint() != market.fingerprint(), "the data a run saw is in its label"
    with pytest.raises(ContractViolation, match="none of the universe"):
        load_market(tmp_path / "store", symbols=("ZZZ",))


# -- the CLI -----------------------------------------------------------------------------


def run_cli(tmp_path, *argv):
    from runtime.cli import Context, main

    lines: list[str] = []
    context = Context(store=tmp_path / "store",
                      cache=tmp_path / "cache", configs_root=tmp_path / "configs",
                      out=lines.append)
    code = main(list(argv), context)
    return code, "\n".join(lines)


def test_backtest_takes_a_universe_and_a_capital_and_records_both(tmp_path):
    build_market(tmp_path / "store", weeks=120)
    path = write_universe(tmp_path, "three", "AAA\nBBB\nCCC\n")
    ledger = tmp_path / "ledger.jsonl"
    code, out = run_cli(tmp_path, "backtest", "--universe", str(path), "--capital", "25000",
                        "--top", "2", "--ledger", str(ledger))
    assert code == 0, out
    assert "universe     three (3 symbols" in out and "from 25,000" in out
    row = json.loads(ledger.read_text().splitlines()[-1])
    assert "universe=three:" in row["note"] and "capital=25000" in row["note"]
    assert "data=" in row["window"], "the data seen is part of the trial's label"
    code, whole = run_cli(tmp_path, "backtest", "--top", "2", "--ledger", str(ledger))
    assert len(ledger.read_text().splitlines()) == 2, "another universe is another trial"


def test_a_report_on_a_universe_still_shows_the_benchmark(tmp_path, monkeypatch):
    import runtime.cli as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)  # reports land in tmp_path/state/reports
    build_market(tmp_path / "store", weeks=120)
    path = write_universe(tmp_path, "two", "BBB\nCCC\n")
    code, out = run_cli(tmp_path, "backtest", "--universe", str(path), "--top", "1",
                        "--benchmark", "AAA", "--report", "--ledger", str(tmp_path / "l.jsonl"))
    assert code == 0, out
    report = next((tmp_path / "state" / "reports").glob("backtest-*.json"))
    data = json.loads(report.read_text())
    series = [s for c in data["charts"] if c["id"] == "equity" for s in c["series"]]
    bench = next(s for s in series if s["name"] == "AAA")
    assert any(v is not None for v in bench["values"]), "the benchmark is priced"


def test_a_sleeve_refuses_a_universe_other_than_the_one_it_opened_on(tmp_path):
    pytest.importorskip("ib_async")
    from tests.test_cli import Harness

    h = Harness(tmp_path)
    universes = tmp_path / "universes"
    universes.mkdir()
    first = write_universe(universes, "first", "AAA\nBBB\nCCC\nDDD\n")
    second = write_universe(universes, "second", "AAA\nBBB\nCCC\nDDD\nEEE\n")
    config = h.config.read_text()
    h.config.write_text(config.replace(
        "strategy: {rebalance_weeks: 1, top_n: 2, lookback_weeks: 13}",
        f"strategy: {{params: {{rebalance_weeks: 1, top_n: 2, lookback_weeks: 13}}, "
        f"universe: '{first}'}}"))
    code, out = h.run("live", "init")
    assert code == 0, out
    code, out = h.run("live", "propose")
    assert code == 0 and "EEE" not in out
    h.config.write_text(h.config.read_text().replace(str(first), str(second)))
    code, out = h.run("live", "status")
    assert code != 0 and "Another universe is another strategy" in out


# -- the research line and the grid ------------------------------------------------------


def test_the_research_line_is_every_study_but_not_the_controls(tmp_path):
    from contracts.identifiers import StrategyVersion
    from validation.ledger import ResearchLedger, Study

    ledger = ResearchLedger(tmp_path / "ledger.jsonl")
    rng = np.random.default_rng(0)
    momentum = [StrategyVersion.of("weekly-momentum", {"top_n": n}) for n in (2, 3, 4)]
    with Study("manual", ledger) as study:
        study.evaluate(momentum[0], "a", {"sharpe": 0.1}, returns=rng.normal(0, 1, 100))
        study.evaluate(momentum[1], "b", {"sharpe": 0.1}, returns=rng.normal(0, 1, 60))
    with Study("grid", ledger) as study:
        study.evaluate(momentum[2], "a", {"sharpe": 0.1}, returns=rng.normal(0, 1, 100))
    with Study("funnel", ledger) as study:
        study.evaluate(StrategyVersion.of("random-control", {"seed": 1}), "a",
                       {"sharpe": 0.0}, returns=rng.normal(0, 1, 100))
    matrix, kept, others = ledger.research_line("weekly-momentum")
    assert matrix.shape == (100, 2) and len(kept) == 2
    assert len(others) == 1, "a different window is counted, at face value"


def random_walk_store(root, weeks=420):
    from data.bitemporal import BitemporalStore
    from data.ingest import to_observations, universe_from_store
    from tests.live_fixtures import LAST_LABEL, weekly_frame

    store = BitemporalStore(root, "bars_1week")
    rng = np.random.default_rng(7)
    from contracts.identifiers import InstrumentId

    for i, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
        frame = weekly_frame(0.0, weeks, LAST_LABEL)
        closes = 50 * np.cumprod(1 + rng.normal(0.002 * (i - 2), 0.03, weeks))
        frame["close"] = closes
        frame["open"] = np.concatenate([[closes[0]], closes[:-1]])
        frame["high"] = np.maximum(frame["open"], closes) * 1.01
        frame["low"] = np.minimum(frame["open"], closes) * 0.99
        store.append(InstrumentId(symbol), to_observations(frame, week_ending=True))
    universe = universe_from_store(store, still_trading_after=datetime(2026, 1, 1,
                                                                        tzinfo=timezone.utc))
    universe.to_csv(root / "universe_weekly.csv", derived=True)


def load_funnel():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "run_funnel.py"
    spec = importlib.util.spec_from_file_location("run_funnel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_grid_scores_every_combination_and_finds_a_plateau(tmp_path, capsys):
    random_walk_store(tmp_path / "store")
    funnel = load_funnel()
    ledger = tmp_path / "ledger.jsonl"
    code = funnel.main(["--store", str(tmp_path / "store"), "--ledger", str(ledger),
                        "--start", "2019-01-01", "--grid-top", "1,2,3",
                        "--grid-rebalance", "1,2,4"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert out.count("OOS Sharpe P10") >= 9 and "plateau centre  N=" in out
    assert "grid PBO" in out and "ql funnel --top" in out
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 9 and {r["study"] for r in rows} == {"grid"}

    again = funnel.main(["--store", str(tmp_path / "store"), "--ledger", str(ledger),
                         "--start", "2019-01-01", "--grid-top", "1,2,3",
                         "--grid-rebalance", "1,2,4"])
    assert again == 0 and len(ledger.read_text().splitlines()) == 9, \
        "the same trials on the same data are reused, not counted twice"


def test_the_plateau_is_not_the_peak():
    """A lone high point with poor neighbours loses to a flat good region."""
    import numpy as np

    p10 = np.array([[0.1, 0.1, 0.1, 0.1],
                    [0.1, 0.9, 0.1, 0.1],
                    [0.1, 0.1, 0.5, 0.5],
                    [0.1, 0.1, 0.5, 0.5]])
    plateau = np.array([[np.min(p10[max(0, i - 1):i + 2, max(0, j - 1):j + 2])
                         for j in range(4)] for i in range(4)])
    assert np.unravel_index(np.argmax(p10), p10.shape) == (1, 1)
    assert np.unravel_index(np.argmax(plateau), plateau.shape) == (3, 3)


def test_the_funnel_counts_the_grid_and_varies_the_capital(tmp_path, capsys):
    random_walk_store(tmp_path / "store")
    funnel = load_funnel()
    ledger = tmp_path / "ledger.jsonl"
    common = ["--store", str(tmp_path / "store"), "--ledger", str(ledger),
              "--start", "2019-06-01"]
    assert funnel.main([*common, "--grid-top", "1,2", "--grid-rebalance", "2,4"]) == 0
    capsys.readouterr()
    funnel.main([*common, "--top", "2", "--rebalance-weeks", "4", "--controls", "20",
                 "--capital", "30000"])
    out = capsys.readouterr().out
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    notes = [r["note"] for r in rows if r["study"] == "funnel"]
    assert any("capital x0.5" in n for n in notes) and any("capital x2" in n for n in notes)
    assert all("capital=30000" in n for n in notes)
    momentum = [r for r in rows if r["strategy"] == "weekly-momentum"]
    assert f"research line           {len(momentum)} trials" in out, \
        "the grid's trials are in the funnel's count"
    assert "(4 backtests" not in out


def test_a_setting_the_loader_does_not_read_is_refused(tmp_path):
    from runtime.config import load_config

    example = (__import__("pathlib").Path(__file__).resolve().parent.parent
               / "configs" / "live.example.yaml").read_text()
    misplaced = tmp_path / "momentum.yaml"
    misplaced.write_text(example.replace("strategy_id: momentum",
                                         "strategy_id: momentum\nuniverse: sector-etfs", 1))
    with pytest.raises(ContractViolation, match="'universe' \\(did you mean it under strategy:"):
        load_config(misplaced)
    typo = tmp_path / "typo.yaml"
    typo.write_text(example.replace("mode: paper", "mode: paper\nsleve_capital: 1", 1))
    with pytest.raises(ContractViolation, match="'sleve_capital'"):
        load_config(typo)
    placed = tmp_path / "placed.yaml"
    placed.write_text(example.replace("  # universe: sector-etfs", "  universe: sector-etfs", 1))
    assert load_config(placed).strategy.universe == "sector-etfs"
