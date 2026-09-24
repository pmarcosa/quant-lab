"""scripts/fetch_ibkr.py against a stand-in for ib_async's IB."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import pytest

ib_async = pytest.importorskip("ib_async")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_ibkr.py"


def load_script():
    spec = importlib.util.spec_from_file_location("fetch_ibkr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weekly_bars(n: int = 60):
    start = date(2020, 1, 6)
    return [
        ib_async.BarData(date=start + timedelta(weeks=i), open=10 + i, high=11 + i,
                         low=9 + i, close=10.5 + i, volume=1000)
        for i in range(n)
    ]


class FakeIB:
    """Replies to history requests from a script of (messages, bars) per call."""

    replies: list = []
    requests: list = []

    def __init__(self):
        self.errorEvent = ib_async.Event("errorEvent")

    def connect(self, *args, **kwargs):
        self.errorEvent.emit(-1, 2106, "HMDS data farm connection is OK:ushmds", None)

    def qualifyContracts(self, contract):
        return [contract]

    def reqHistoricalData(self, contract, **kwargs):
        FakeIB.requests.append(kwargs)
        messages, bars = FakeIB.replies.pop(0)
        for code, text in messages:
            self.errorEvent.emit(4, code, text, contract)
        return bars

    def disconnect(self):
        pass


@pytest.fixture
def script(monkeypatch, tmp_path):
    module = load_script()
    monkeypatch.setattr(module, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(ib_async, "IB", FakeIB)
    FakeIB.requests = []
    return module


def test_retries_and_passes_the_options(script, capsys):
    FakeIB.replies = [([(366, "No historical data query found")], []), ([], weekly_bars())]
    written = script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1,
                           duration="5 Y", timeout=30, attempts=2)
    assert written == 1
    assert (script.CACHE / "weekly" / "NFLX.csv").exists()
    first = FakeIB.requests[0]
    assert (first["durationStr"], first["timeout"], first["endDateTime"]) == ("5 Y", 30, "")
    out = capsys.readouterr()
    assert "HMDS data farm connection is OK" in out.out
    assert "IBKR 366" in out.err and "cancelled" in out.err


def test_explains_a_permissions_refusal(script, capsys):
    text = "Historical Market Data Service error message:No market data permissions"
    FakeIB.replies = [([(162, text)], []), ([(162, text)], [])]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1) == 0
    err = capsys.readouterr().err
    assert err.count("IBKR 162") == 2
    assert "permissions" in err and "SKIPPED" in err


def test_silence_is_reported_as_a_timeout(script, capsys):
    FakeIB.replies = [([], [])]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1, attempts=1, timeout=7) == 0
    assert "no reply from IBKR within 7s" in capsys.readouterr().err


def daily_bars(days: int = 400):
    start = date(2024, 1, 1)  # a Monday
    out = []
    for i in range(days):
        day = start + timedelta(days=i)
        if day.weekday() < 5:
            out.append(ib_async.BarData(date=day, open=100 + i, high=101 + i, low=99 + i,
                                        close=100.5 + i, volume=10))
    return out


def test_weekly_asks_for_daily_bars_and_groups_them_into_weeks(script):
    FakeIB.replies = [([], daily_bars())]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1) == 1
    assert FakeIB.requests[0]["barSizeSetting"] == "1 day"
    assert FakeIB.requests[0]["whatToShow"] == "ADJUSTED_LAST"
    import pandas as pd

    frame = pd.read_csv(script.CACHE / "weekly" / "NFLX.csv", parse_dates=["timestamp"])
    first = frame.iloc[0]
    assert str(first["timestamp"].date()) == "2024-01-01"
    assert (first["open"], first["close"], first["volume"]) == (100, 104.5, 50)
    assert (first["high"], first["low"]) == (105, 99)
    iso = frame["timestamp"].dt.isocalendar()
    assert not iso.duplicated(subset=["year", "week"]).any()


def test_a_rejected_request_is_not_retried(script, capsys):
    text = "Multi day bar size not supported with adjusted last"
    FakeIB.replies = [([(321, text)], [])]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1, attempts=3) == 0
    assert len(FakeIB.requests) == 1
    assert "IBKR 321" in capsys.readouterr().err
