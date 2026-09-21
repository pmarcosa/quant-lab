"""The live data refresh: new weeks in, restatements as revisions, nothing partial."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

pytest.importorskip("ib_async")

from contracts.identifiers import InstrumentId  # noqa: E402
from contracts.live import TradingMode  # noqa: E402
from data.bitemporal import BitemporalStore  # noqa: E402
from data.vendor import write_cache_csv  # noqa: E402
from execution.ibkr import IBKRBroker  # noqa: E402
from runtime.refresh import refresh_weekly  # noqa: E402
from tests.fake_gateway import FakeGateway  # noqa: E402
from tests.live_fixtures import build_market  # noqa: E402

AAA = InstrumentId("AAA")
SATURDAY = datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def setup(tmp_path):
    frames = build_market(tmp_path / "store")
    cache = tmp_path / "cache"
    for symbol, frame in frames.items():
        write_cache_csv(symbol, frame, cache, "weekly")
    gateway = FakeGateway()
    broker = IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)
    store = BitemporalStore(tmp_path / "store", "bars_1week")
    return frames, cache, gateway, broker, store


def served(frame, extra_weeks=1, drift=0.02):
    """The frame plus new weeks, as the broker would return it."""
    rows = [frame]
    last = frame.iloc[-1]
    for k in range(1, extra_weeks + 1):
        close = float(last["close"]) * (1 + drift) ** k
        rows.append(pd.DataFrame(
            {"open": [close * 0.99], "high": [close * 1.01], "low": [close * 0.98],
             "close": [close], "volume": [1e6]},
            index=[frame.index[-1] + pd.Timedelta(weeks=k)],
        ))
    return pd.concat(rows)


def test_a_new_complete_week_is_recorded_as_known_at_the_fetch(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"]))
    (result,) = refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    assert (result.new_weeks, result.revised_weeks) == (1, 0)
    known = store.as_of(AAA, SATURDAY)
    assert known["available_time"].iloc[-1] == pd.Timestamp(SATURDAY)
    first = store.first_known(AAA, SATURDAY)
    assert first.iloc[-1] == pd.Timestamp(SATURDAY), "decided on after the fetch, not before"


def test_the_week_in_progress_is_not_recorded(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"], extra_weeks=2))
    (result,) = refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    assert result.new_weeks == 1, "the second new week has not closed yet on this Saturday"


def test_a_restatement_arrives_as_a_revision_without_rewriting_the_past(setup):
    """A 2:1 split halves every past price. The old view must still be there."""
    frames, cache, gateway, broker, store = setup
    split = frames["AAA"].copy()
    for column in ("open", "high", "low", "close"):
        split[column] = split[column] / 2
    gateway.serve_history("AAA", split)
    (result,) = refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    assert result.revised_weeks == len(split)
    before = datetime(2026, 9, 20, tzinfo=timezone.utc)
    assert store.as_of(AAA, before)["close"].iloc[-1] == pytest.approx(
        frames["AAA"]["close"].iloc[-1]
    ), "a query pinned before the fetch sees the unsplit price"
    assert store.as_of(AAA, SATURDAY)["close"].iloc[-1] == pytest.approx(
        split["close"].iloc[-1]
    )


def test_an_unchanged_history_writes_nothing(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", frames["AAA"])
    (result,) = refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    assert (result.new_weeks, result.revised_weeks) == (0, 0)


def test_the_cache_keeps_older_history_and_gains_the_new_week(setup):
    frames, cache, gateway, broker, store = setup
    recent = served(frames["AAA"]).iloc[-10:]
    gateway.serve_history("AAA", recent)
    refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    written = pd.read_csv(cache / "weekly" / "AAA.csv")
    assert len(written) == len(frames["AAA"]) + 1


def test_one_failing_symbol_does_not_stop_the_rest(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"]))
    results = {r.instrument: r for r in refresh_weekly(broker, cache, store, SATURDAY)}
    assert results["AAA"].new_weeks == 1
    assert results["BBB"].error, "BBB returned nothing and says so"


def test_the_live_market_schedules_the_decision_after_the_fetch(setup, tmp_path):
    frames, cache, gateway, broker, store = setup
    from contracts.temporal import BarInterval
    from runtime.wiring import load_market

    gateway.serve_history("AAA", served(frames["AAA"]))
    for symbol in ("BBB", "CCC", "DDD", "EEE"):
        gateway.serve_history(symbol, served(frames[symbol]))
    refresh_weekly(broker, cache, store, SATURDAY)
    market = load_market(tmp_path / "store", interval=BarInterval.WEEK)
    assert market.schedule[-1] == SATURDAY
    assert market.schedule[-2].weekday() == 4, "historical weeks are decided on Friday evening"
