"""The live data refresh: new weeks in, restatements as revisions, nothing partial."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ib_async")

from contracts.identifiers import InstrumentId  # noqa: E402
from contracts.live import TradingMode  # noqa: E402
from contracts.temporal import BarInterval  # noqa: E402
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
    # Every served week but the first: a refresh window's first week is
    # dropped as possibly partial (data.ingest.weeks_from_days).
    assert result.revised_weeks == len(split) - 1
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


# -- TRADES in the cache, dividend factors beside it (data.adjustments) ------------------

NEXT_SATURDAY = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)


def with_dividend(frame, before, factor):
    """What ADJUSTED_LAST returns when a dividend goes ex on ``before``."""
    out = frame.copy()
    for column in ("open", "high", "low", "close"):
        out.loc[out.index < before, column] *= factor
    return out


def test_every_request_is_one_ibkr_accepts(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"]))
    refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    assert not gateway.rejected
    assert set(gateway.history_kinds) == {"TRADES", "ADJUSTED_LAST"}
    from contracts.errors import ContractViolation

    with pytest.raises(ContractViolation):
        broker.historical_bars(AAA, "1 week", "2 Y", what="ADJUSTED_LAST")
    with pytest.raises(ContractViolation):
        broker.historical_bars(AAA, "1 hour", "1 M", end=SATURDAY, what="ADJUSTED_LAST")


def test_the_first_refresh_without_factors_takes_the_whole_history(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"]))
    (result,) = refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    assert "full history" in result.note
    assert (cache / "factors" / "AAA.csv").exists()
    assert gateway.history_requests[0][1] != "2 Y", "a span covering the cache, not the window"


def test_a_dividend_since_the_last_refresh_revises_the_whole_history(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"]))
    refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])

    history = served(frames["AAA"], extra_weeks=2)
    ex = history.index[-1]  # went ex in the newest week
    gateway.serve_history("AAA", history, adjusted=with_dividend(history, ex, 0.99))
    (result,) = refresh_weekly(broker, cache, store, NEXT_SATURDAY, symbols=["AAA"])

    assert "dividend" in result.note and result.new_weeks == 1
    assert result.revised_weeks >= len(frames["AAA"]) - 2, "every earlier week, not the window"
    cached = pd.read_csv(cache / "weekly" / "AAA.csv", parse_dates=["timestamp"])
    assert cached["close"].iloc[-3] == pytest.approx(history["close"].iloc[-3]), \
        "the cache keeps the price as traded"
    known = store.as_of(AAA, NEXT_SATURDAY)
    assert known["close"].iloc[-3] == pytest.approx(history["close"].iloc[-3] * 0.99)
    assert known["close"].iloc[-1] == pytest.approx(history["close"].iloc[-1])
    pinned = store.as_of(AAA, datetime(2026, 9, 30, tzinfo=timezone.utc))
    assert pinned["close"].iloc[-1] == pytest.approx(history["close"].iloc[-2]), \
        "a query pinned before the refresh still sees the old scale"
    assert known["close"].iloc[-2] == pytest.approx(history["close"].iloc[-2] * 0.99)


def test_a_split_since_the_last_download_downloads_the_history_again(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"]))
    refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])

    split = served(frames["AAA"], extra_weeks=2)
    for column in ("open", "high", "low", "close"):
        split[column] = split[column] / 2
    gateway.serve_history("AAA", split)
    before = len(gateway.history_requests)
    (result,) = refresh_weekly(broker, cache, store, NEXT_SATURDAY, symbols=["AAA"])

    assert "split" in result.note and "x0.5" in result.note
    spans = [r[1] for r in gateway.history_requests[before:]]
    assert spans[0] == "2 Y" and spans[-1] != "2 Y", "the window first, then the whole history"
    cached = pd.read_csv(cache / "weekly" / "AAA.csv")
    assert cached["close"].iloc[-3] == pytest.approx(split["close"].iloc[-3])
    assert store.as_of(AAA, NEXT_SATURDAY)["close"].iloc[-3] == pytest.approx(
        split["close"].iloc[-3])


def test_a_download_that_disagrees_with_the_cache_is_refused(setup):
    frames, cache, gateway, broker, store = setup
    gateway.serve_history("AAA", served(frames["AAA"]))
    refresh_weekly(broker, cache, store, SATURDAY, symbols=["AAA"])
    before = (cache / "weekly" / "AAA.csv").read_text()

    noisy = served(frames["AAA"], extra_weeks=2)
    noisy.iloc[-30:-20, noisy.columns.get_loc("close")] *= 1.04
    gateway.serve_history("AAA", noisy)
    (result,) = refresh_weekly(broker, cache, store, NEXT_SATURDAY, symbols=["AAA"])

    assert "more than one constant" in result.error
    assert (cache / "weekly" / "AAA.csv").read_text() == before, "the cache is left untouched"


def test_ingest_multiplies_the_cache_by_its_factors(tmp_path):
    from data.adjustments import write_factors
    from data.ingest import ingest_directory

    frames = build_market(tmp_path / "unused")
    cache = tmp_path / "cache"
    for symbol in ("AAA", "BBB"):
        write_cache_csv(symbol, frames[symbol], cache, "weekly")
    days = pd.bdate_range(frames["AAA"].index[0], frames["AAA"].index[-1] + pd.Timedelta(days=4))
    factors = pd.Series(np.where(days < frames["AAA"].index[40], 0.97, 1.0), index=days)
    write_factors(cache, "AAA", factors)
    store = BitemporalStore(tmp_path / "store", "bars_1week")
    ingest_directory(cache / "weekly", store, interval=BarInterval.WEEK,
                     factors=cache / "factors", now=SATURDAY)
    aaa = store.as_of(AAA, SATURDAY)["close"].to_numpy()
    assert np.allclose(aaa[:40], frames["AAA"]["close"].to_numpy()[:40] * 0.97)
    assert np.allclose(aaa[40:], frames["AAA"]["close"].to_numpy()[40:])
    bbb = store.as_of(InstrumentId("BBB"), SATURDAY)["close"].to_numpy()
    assert np.allclose(bbb, frames["BBB"]["close"].to_numpy()), "no factors: price-only"
