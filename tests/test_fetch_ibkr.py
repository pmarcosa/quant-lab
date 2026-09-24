"""scripts/fetch_ibkr.py against a stand-in for ib_async's IB."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ib_async = pytest.importorskip("ib_async")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_ibkr.py"


def load_script():
    spec = importlib.util.spec_from_file_location("fetch_ibkr", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def daily_bars(days: int = 400, dividend_on: date | None = None, factor: float = 1.0):
    """Weekday bars from Monday 1 January 2024. Before ``dividend_on`` the prices
    are multiplied by ``factor``: that is ADJUSTED_LAST for a dividend going ex
    that day."""
    start = date(2024, 1, 1)
    out = []
    for i in range(days):
        day = start + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        f = factor if dividend_on is not None and day < dividend_on else 1.0
        out.append(ib_async.BarData(date=day, open=(100 + i) * f, high=(101 + i) * f,
                                    low=(99 + i) * f, close=(100.5 + i) * f, volume=10))
    return out


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


def test_asks_for_daily_trades_then_adjusted_last_and_retries(script, capsys):
    FakeIB.replies = [
        ([(366, "No historical data query found")], []),  # TRADES, first try
        ([], daily_bars()),                                # TRADES, second try
        ([], daily_bars()),                                # ADJUSTED_LAST
    ]
    written = script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1,
                           duration="5 Y", timeout=30, attempts=2)
    assert written == 1
    kinds = [(r["whatToShow"], r["barSizeSetting"], r["endDateTime"]) for r in FakeIB.requests]
    assert kinds == [("TRADES", "1 day", "")] * 2 + [("ADJUSTED_LAST", "1 day", "")]
    assert (FakeIB.requests[0]["durationStr"], FakeIB.requests[0]["timeout"]) == ("5 Y", 30)
    assert (script.CACHE / "weekly" / "NFLX.csv").exists()
    assert (script.CACHE / "factors" / "NFLX.csv").exists()
    out = capsys.readouterr()
    assert "HMDS data farm connection is OK" in out.out
    assert "IBKR 366" in out.err and "cancelled" in out.err


def test_the_cache_keeps_trades_and_the_factors_keep_the_dividend(script):
    ex = date(2024, 3, 13)  # a Wednesday: the dividend goes ex inside a week
    FakeIB.replies = [([], daily_bars()), ([], daily_bars(dividend_on=ex, factor=0.98))]
    assert script.fetch(["XOM"], "weekly", "127.0.0.1", 4002, 1) == 1
    factors = pd.read_csv(script.CACHE / "factors" / "XOM.csv", parse_dates=["date"])
    before = factors[factors["date"] < pd.Timestamp(ex)]["factor"]
    after = factors[factors["date"] >= pd.Timestamp(ex)]["factor"]
    assert before.round(10).eq(0.98).all() and after.round(10).eq(1.0).all()
    weeks = pd.read_csv(script.CACHE / "weekly" / "XOM.csv", parse_dates=["timestamp"])
    week = weeks[weeks["timestamp"] == pd.Timestamp("2024-03-11")].iloc[0]
    # Close: Friday's TRADES close, as traded. Open: Monday's, at the week-end
    # scale (x0.98), fixed once the week is over.
    assert week["close"] == pytest.approx(100.5 + 74)
    assert week["open"] == pytest.approx((100 + 70) * 0.98)
    plain = weeks[weeks["timestamp"] == pd.Timestamp("2024-03-04")].iloc[0]
    assert plain["open"] == pytest.approx(100 + 63), "weeks without an ex-date are as traded"


def test_explains_a_permissions_refusal(script, capsys):
    text = "Historical Market Data Service error message:No market data permissions"
    FakeIB.replies = [([(162, text)], []), ([(162, text)], [])]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1) == 0
    err = capsys.readouterr().err
    assert err.count("IBKR 162") == 2
    assert "permissions" in err and "SKIPPED" in err
    assert len(FakeIB.requests) == 2, "no ADJUSTED_LAST request once TRADES has failed"


def test_silence_is_reported_as_a_timeout(script, capsys):
    FakeIB.replies = [([], [])]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1, attempts=1, timeout=7) == 0
    assert "no reply from IBKR within 7s" in capsys.readouterr().err


def test_weekly_bars_are_grouped_from_daily_ones(script):
    FakeIB.replies = [([], daily_bars()), ([], daily_bars())]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1) == 1
    frame = pd.read_csv(script.CACHE / "weekly" / "NFLX.csv", parse_dates=["timestamp"])
    first = frame.iloc[0]
    # The window's first week is dropped (a window almost always opens mid-week);
    # the first bar kept is the complete week of 8 January.
    assert str(first["timestamp"].date()) == "2024-01-08"
    assert (first["open"], first["close"], first["volume"]) == (107, 111.5, 50)
    assert (first["high"], first["low"]) == (112, 106)
    iso = frame["timestamp"].dt.isocalendar()
    assert not iso.duplicated(subset=["year", "week"]).any()


def test_a_rejected_request_is_not_retried(script, capsys):
    text = "Multi day bar size not supported with adjusted last"
    FakeIB.replies = [([(321, text)], [])]
    assert script.fetch(["NFLX"], "weekly", "127.0.0.1", 4002, 1, attempts=3) == 0
    assert len(FakeIB.requests) == 1
    assert "IBKR 321" in capsys.readouterr().err


def test_a_window_opening_mid_week_does_not_leave_a_short_first_week():
    from data.ingest import weeks_from_days

    days = pd.bdate_range("2024-01-03", "2024-01-26")  # opens on a Wednesday
    daily = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0},
                         index=days)
    weeks = weeks_from_days(daily)
    assert [d.date().isoformat() for d in weeks.index] == ["2024-01-08", "2024-01-15", "2024-01-22"]
    assert (weeks["volume"] == 5).all()
