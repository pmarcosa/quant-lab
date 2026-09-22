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
