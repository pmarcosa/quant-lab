"""Commands that hand their arguments to a script must receive leading options."""

from __future__ import annotations

import pytest

import runtime.cli as cli


@pytest.fixture
def calls(monkeypatch, tmp_path):
    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(cli, "_run_script", lambda script, argv: seen.append((script, argv)) or 0)
    monkeypatch.chdir(tmp_path)
    return seen


def test_fetch_receives_leading_options(calls):
    cli.main(["--config", "/nonexistent.yaml", "data", "fetch",
              "--symbols", "NFLX,ORCL", "--freq", "weekly", "--port", "4002"])
    assert calls == [("fetch_ibkr", ["--symbols", "NFLX,ORCL", "--freq", "weekly", "--port", "4002"])]


def test_funnel_receives_leading_options(calls):
    cli.main(["funnel", "--help"])
    assert calls == [("run_funnel", ["--help"])]


def test_other_commands_still_refuse_unknown_options(calls):
    with pytest.raises(SystemExit):
        cli.main(["data", "status", "--symbols", "A"])
    assert calls == []
