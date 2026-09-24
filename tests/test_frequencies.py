"""Bar sizes other than weekly: timing, alignment, refresh and monitoring.

The deployed strategy is weekly, and for a long time "a bar" silently meant "a
week" in several places. These tests fix the behaviour for daily and intraday
bars, so a strategy of another frequency does not inherit a weekly assumption
nobody wrote down.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pandas as pd
import pytest

from contracts.errors import ContractViolation
from contracts.execution import TimeInForce
from contracts.temporal import BarInterval
from data.ingest import bar_close
from runtime.config import ExecutionSettings, MonitoringSettings
from runtime.wiring import load_market
from tests.live_fixtures import LAST_DAY, build_daily_market

UTC = timezone.utc


# -- what a bar is ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "interval"),
    [("weekly", BarInterval.WEEK), ("1Day", BarInterval.DAY), ("hour", BarInterval.HOUR),
     ("MINUTE", BarInterval.MINUTE), ("daily", BarInterval.DAY)],
)
def test_a_bar_size_is_parsed_from_a_plain_word(word, interval):
    assert BarInterval.parse(word) is interval


def test_an_unknown_bar_size_is_refused():
    with pytest.raises(ContractViolation, match="unknown bar interval"):
        BarInterval.parse("fortnightly")


def test_calendar_weeks_become_bars():
    assert BarInterval.WEEK.bars_per_week == 1
    assert BarInterval.DAY.bars_per_week == pytest.approx(252 / 52)
    assert BarInterval.HOUR.bars_per_week == pytest.approx(7 * 252 / 52)
    assert BarInterval.HOUR.is_intraday and not BarInterval.DAY.is_intraday


def test_a_daily_bar_closes_at_that_days_session_close():
    assert bar_close(datetime(2026, 9, 16), interval=BarInterval.DAY) == datetime(
        2026, 9, 16, 21, 0, tzinfo=UTC
    )


def test_a_weekly_bar_closes_on_friday_whatever_its_label():
    assert bar_close(datetime(2026, 9, 14), interval=BarInterval.WEEK) == datetime(
        2026, 9, 18, 21, 0, tzinfo=UTC
    )


def test_an_intraday_bar_closes_one_bar_after_its_label():
    """IBKR labels intraday bars with their start. Stamped there, a decision
    would see a bar that had not finished forming: one bar of lookahead."""
    label = datetime(2026, 9, 16, 14, 30, tzinfo=UTC)
    assert bar_close(label, interval=BarInterval.HOUR) == label + timedelta(hours=1)
    assert bar_close(label, interval=BarInterval.MINUTE) == label + timedelta(minutes=1)


def test_an_intraday_bar_without_a_timezone_is_refused():
    with pytest.raises(ContractViolation, match="no timezone"):
        bar_close(datetime(2026, 9, 16, 14, 30), interval=BarInterval.HOUR)


# -- execution and data age by bar size ----------------------------------------------


def test_orders_go_to_the_opening_auction_unless_the_strategy_is_intraday():
    auto = ExecutionSettings()
    assert auto.resolve(BarInterval.WEEK) is TimeInForce.OPG
    assert auto.resolve(BarInterval.DAY) is TimeInForce.OPG
    assert auto.resolve(BarInterval.HOUR) is TimeInForce.DAY
    assert ExecutionSettings("day").resolve(BarInterval.WEEK) is TimeInForce.DAY


def test_the_data_age_limit_follows_the_bar_size_unless_set():
    m = MonitoringSettings()
    assert m.data_age_hours(BarInterval.WEEK) > m.data_age_hours(BarInterval.DAY)
    assert m.data_age_hours(BarInterval.DAY) > 72, "Friday's bar must still be usable on Monday"
    assert MonitoringSettings(max_data_age_hours=5).data_age_hours(BarInterval.WEEK) == 5


# -- a daily market is daily ---------------------------------------------------------


def test_a_daily_store_is_not_collapsed_into_weeks(tmp_path):
    build_daily_market(tmp_path, days=60)
    market = load_market(tmp_path, interval=BarInterval.DAY)
    assert market.interval is BarInterval.DAY
    assert len(market.schedule) == 60, "one decision per session, not per week"
    assert len({m.date() for m in market.schedule}) == 60
    latest = market.schedule[-1]
    assert latest.date() == LAST_DAY.date() and latest.time() >= time(21, 0), (
        "decided once Friday's session has closed"
    )
    # Consecutive bars carry their own prices, not a week's last one repeated.
    closes = [market.window.marks_at(m)[next(iter(market.window.marks_at(m)))]
              for m in market.schedule[-5:]]
    assert len(set(closes)) == 5


def test_a_daily_market_knows_each_bars_high(tmp_path):
    frames = build_daily_market(tmp_path, days=30)
    market = load_market(tmp_path, interval=BarInterval.DAY)
    from contracts.identifiers import InstrumentId

    last = market.schedule[-1]
    assert market.window.highs_at(last)[InstrumentId("AAA")] == pytest.approx(
        float(frames["AAA"]["high"].iloc[-1])
    )


# -- refreshing daily bars -----------------------------------------------------------


def test_a_daily_refresh_records_only_sessions_that_have_closed(tmp_path):
    pytest.importorskip("ib_async")
    from contracts.identifiers import InstrumentId
    from contracts.live import TradingMode
    from data.bitemporal import BitemporalStore
    from data.vendor import write_cache_csv
    from execution.ibkr import IBKRBroker
    from runtime.refresh import refresh
    from tests.fake_gateway import FakeGateway

    frames = build_daily_market(tmp_path / "store", days=40)
    for symbol, frame in frames.items():
        write_cache_csv(symbol, frame, tmp_path / "cache", "daily")
    frame = frames["AAA"]
    extra = []
    for k in (1, 2):  # Monday, which has closed, and Tuesday, which has not
        close = float(frame["close"].iloc[-1]) * (1.01 ** k)
        extra.append(pd.DataFrame(
            {"open": [close], "high": [close * 1.01], "low": [close * 0.99],
             "close": [close], "volume": [1e6]},
            index=[frame.index[-1] + pd.offsets.BDay(k)],
        ))
    gateway = FakeGateway()
    gateway.serve_history("AAA", pd.concat([frame, *extra]))
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    store = BitemporalStore(tmp_path / "store", "bars_1day")
    now = datetime(2026, 9, 22, 15, 0, tzinfo=UTC)  # Tuesday afternoon, session open

    (result,) = refresh(broker, tmp_path / "cache", store, now, BarInterval.DAY, symbols=["AAA"])

    assert (result.new_bars, result.revised_bars) == (1, 0)
    assert result.last_bar.startswith("2026-09-21")
    known = store.as_of(InstrumentId("AAA"), now)
    assert known["available_time"].iloc[-1] == pd.Timestamp(now)


# -- monitoring a daily strategy -----------------------------------------------------


def test_a_daily_long_short_strategy_is_monitored_in_days(tmp_path):
    pytest.importorskip("ib_async")
    from runtime.monitor import build_baseline, run_monitor
    from tests.fake_gateway import FakeGateway
    from tests.live_fixtures import Clock, append_day
    from tests.test_long_short_daily import SATURDAY, make_session

    root = tmp_path / "store"
    frames = build_daily_market(root)
    gateway = FakeGateway(cash=150_000.0)
    gateway.shortable = {s: 1e6 for s in frames}
    clock = Clock(SATURDAY)
    session = make_session(tmp_path, clock=clock, gateway=gateway)

    baseline = build_baseline(session, clock.now)
    assert baseline.interval == BarInterval.DAY.value
    assert len(baseline.returns) > 150, "a year of daily bars, less the warm-up"
    baseline.save(session.config.baseline_path)

    session.init()
    session.sync()
    for _ in range(4):
        gateway.prices = {s: float(f["close"].iloc[-1]) for s, f in frames.items()}
        proposal = session.propose()
        day = (frames["AAA"].index[-1] + pd.offsets.BDay(1)).date()
        if proposal.orders:
            session.approve(proposal.proposal_id, proposal.proposal_id)
            clock.now = datetime.combine(day, time(13, 30), tzinfo=UTC)
            gateway.opening_auction(
                {s: float(f["close"].iloc[-1]) * 1.001 for s, f in frames.items()}, when=clock.now
            )
            session.sync()
        clock.now = datetime.combine(day, time(22, 0), tzinfo=UTC)
        frames = append_day(root, frames, clock.now)
        session.sync()

    report = run_monitor(session)
    assert report.interval == BarInterval.DAY.value
    assert report.assessment.unit == "day"
    assert report.assessment.periods == len(report.returns) >= 3
    assert report.process.stop_coverage == 1.0, "longs and shorts are both protected"
    assert report.health.last_reconciliation == "ok"


# -- the session calendar and intraday readiness ----------------------------------------------

calendar = pytest.importorskip("data.calendar")
needs_calendar = pytest.mark.skipif(not calendar.is_available(), reason="exchange_calendars")


@needs_calendar
def test_the_calendar_knows_holidays_and_half_days():
    from datetime import date

    assert calendar.session_bounds(date(2026, 11, 26)) is None, "Thanksgiving"
    opened, closed = calendar.session_bounds(date(2026, 11, 27))
    assert calendar.is_early_close(date(2026, 11, 27)) and closed.hour == 18, "13:00 New York"
    assert calendar.in_regular_session(datetime(2026, 9, 22, 14, 30, tzinfo=UTC))
    assert not calendar.in_regular_session(datetime(2026, 9, 22, 12, 0, tzinfo=UTC)), "pre-market"


@needs_calendar
def test_the_last_closed_bar_skips_nights_weekends_and_holidays():
    hour = timedelta(hours=1)
    assert calendar.last_closed_bar(datetime(2026, 9, 22, 16, 10, tzinfo=UTC), hour) == datetime(
        2026, 9, 22, 15, 30, tzinfo=UTC)
    assert calendar.last_closed_bar(datetime(2026, 11, 26, 15, 0, tzinfo=UTC), hour) == datetime(
        2026, 11, 25, 21, 0, tzinfo=UTC), "Thanksgiving: the previous session's close"


@needs_calendar
def test_an_hourly_bar_ends_at_the_session_close_if_that_comes_first():
    last = datetime(2026, 9, 22, 19, 30, tzinfo=UTC)  # 15:30 New York
    assert bar_close(last, interval=BarInterval.HOUR) == datetime(2026, 9, 22, 20, 0, tzinfo=UTC)
    half_day = datetime(2026, 11, 27, 17, 30, tzinfo=UTC)  # 12:30 on a 13:00 close
    assert bar_close(half_day, interval=BarInterval.HOUR) == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)


def hourly_frame(start: str, end: str, price: float = 100.0) -> pd.DataFrame:
    """Hourly regular-session bars between two dates, plus one pre-market bar a day."""
    rows = {}
    for day in pd.bdate_range(start, end):
        bounds = calendar.session_bounds(day.date())
        if bounds is None:
            continue
        opened, closed = bounds
        rows[pd.Timestamp(opened) - pd.Timedelta(hours=2)] = price  # pre-market
        t = pd.Timestamp(opened)
        while t < pd.Timestamp(closed):
            rows[t] = price
            t += pd.Timedelta(hours=1)
            price *= 1.0001
    frame = pd.DataFrame({"close": pd.Series(rows)})
    frame["open"] = frame["close"]
    frame["high"] = frame["close"] * 1.001
    frame["low"] = frame["close"] * 0.999
    frame["volume"] = 1e5
    return frame[["open", "high", "low", "close", "volume"]]


@needs_calendar
def test_backfill_pages_back_paced_resumable_and_regular_hours_only(tmp_path):
    pytest.importorskip("ib_async")
    from contracts.live import TradingMode
    from execution.ibkr import IBKRBroker
    from runtime.refresh import backfill
    from tests.fake_gateway import FakeGateway

    gateway = FakeGateway()
    gateway.serve_intraday("AAA", hourly_frame("2024-06-03", "2026-09-18"))
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    now = datetime(2026, 9, 19, 12, tzinfo=UTC)
    waits = []
    (result,) = backfill(broker, tmp_path, BarInterval.HOUR, now, years=3, symbols=["AAA"],
                         chunk="1 Y", sleep=waits.append)
    assert result.exhausted, "the broker ran out of history before three years"
    assert result.first_bar.startswith("2024-06-03T13:30"), "the first regular bar, not pre-market"
    # Two daily requests for the dividend factors, then four pages of TRADES.
    assert result.requests == 6 and len(waits) == 5 and all(w >= 10 for w in waits)
    assert gateway.history_kinds[:2] == ["TRADES", "ADJUSTED_LAST"]
    assert set(gateway.history_kinds[2:]) == {"TRADES"}, "pages are TRADES: IBKR pages nothing else"
    assert not gateway.rejected
    cached = pd.read_csv(tmp_path / "hourly" / "AAA.csv", parse_dates=["timestamp"])
    assert len(cached) == result.bars_added
    assert cached["timestamp"].is_unique, "intraday labels keep their time"
    assert str(cached["timestamp"].iloc[0]).startswith("2024-06-03 13:30:00+00:00")
    assert (tmp_path / "factors" / "AAA.csv").exists()
    again = backfill(broker, tmp_path, BarInterval.HOUR, now, years=3, symbols=["AAA"],
                     chunk="1 Y", sleep=waits.append)[0]
    assert again.bars_added == 0, "resumes from the earliest cached bar"


@needs_calendar
def test_backfill_starts_again_when_a_split_rescaled_the_history(tmp_path):
    pytest.importorskip("ib_async")
    from contracts.live import TradingMode
    from execution.ibkr import IBKRBroker
    from runtime.refresh import backfill
    from tests.fake_gateway import FakeGateway

    history = hourly_frame("2025-06-02", "2026-09-18")
    gateway = FakeGateway()
    gateway.serve_intraday("AAA", history)
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    now = datetime(2026, 9, 19, 12, tzinfo=UTC)
    backfill(broker, tmp_path, BarInterval.HOUR, datetime(2026, 3, 2, tzinfo=UTC), years=0.5,
             symbols=["AAA"], chunk="1 M", sleep=lambda _: None)  # a cache written in March
    split = history.copy()
    for column in ("open", "high", "low", "close"):
        split[column] = split[column] / 4
    gateway.serve_intraday("AAA", split)
    (result,) = backfill(broker, tmp_path, BarInterval.HOUR, now, years=2, symbols=["AAA"],
                         chunk="1 Y", sleep=lambda _: None)  # reaches past the cache
    assert "split" in result.note and "x0.25" in result.note
    cached = pd.read_csv(tmp_path / "hourly" / "AAA.csv", parse_dates=["timestamp"])
    assert cached["close"].iloc[0] == pytest.approx(split["close"].iloc[0]), \
        "the whole cache is at the new scale"
    assert cached["timestamp"].is_unique


def test_an_intraday_strategy_is_not_ready_without_years_of_history(tmp_path):
    from types import SimpleNamespace

    from contracts.live import TradingMode
    from runtime.config import LiveConfig
    from runtime.live import LiveSession
    from tests.toy_strategies import DailyLongShort

    class Hourly(DailyLongShort):
        @property
        def filtration_spec(self):
            from contracts.temporal import FiltrationSpec

            return FiltrationSpec(interval=BarInterval.HOUR, observation_lag_bars=0)

    config = LiveConfig(strategy_id="hourly", mode=TradingMode.PAPER, account="DU1234567",
                        sleeve_capital=1.0, state_dir=tmp_path)
    session = LiveSession(config, None, tmp_path, strategy_factory=lambda _: Hourly())
    one_year = SimpleNamespace(schedule=[datetime(2025, 9, 19, tzinfo=UTC),
                                         datetime(2026, 9, 18, tzinfo=UTC)])
    problems = session.readiness(one_year)
    assert any("1.0 years of hourly history" in p for p in problems)
    five = SimpleNamespace(schedule=[datetime(2021, 9, 17, tzinfo=UTC),
                                     datetime(2026, 9, 18, tzinfo=UTC)])
    assert session.readiness(five) == () or not calendar.is_available()
