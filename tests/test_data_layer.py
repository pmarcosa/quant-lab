"""The data layer: append-only storage, point-in-time universes, and filtrations.

The tests that matter here are the negative ones. A store that returns the right
answer is unremarkable; a store that *cannot* return the future is the point.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from contracts.errors import CausalityViolation, ContractViolation
from contracts.identifiers import InstrumentId
from data.bitemporal import BitemporalStore
from data.filtration import StoreFiltration
from data.ingest import to_observations, universe_from_store
from data.universe import PointInTimeUniverse, derive_memberships
from tests.conftest import at, weekly_bars

OLD = InstrumentId("old")
NEW = InstrumentId("new")


# -- the store ---------------------------------------------------------------


def test_records_available_before_they_happened_are_refused(store: BitemporalStore) -> None:
    bad = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2026-09-18", tz="UTC")],
            "available_time": [pd.Timestamp("2026-09-17", tz="UTC")],
            "close": [100.0],
        }
    )
    with pytest.raises(ContractViolation, match="availability before the event"):
        store.append(OLD, bad)


def test_records_without_both_timestamps_are_refused(store: BitemporalStore) -> None:
    with pytest.raises(ContractViolation, match="missing"):
        store.append(OLD, pd.DataFrame({"event_time": [], "close": []}))


def test_a_query_cannot_see_past_its_decision_time(loaded_store: BitemporalStore) -> None:
    early = loaded_store.as_of(OLD, at(2012))
    late = loaded_store.as_of(OLD, at(2020))
    assert len(early) < len(late)
    assert early.index.max() <= pd.Timestamp(at(2012))


def test_a_bar_is_not_knowable_until_it_is_published(store: BitemporalStore) -> None:
    """The publication lag is real: at the close itself the print is not final."""
    store.append(OLD, to_observations(weekly_bars("2026-09-04", 3)))
    # 21:00 UTC: the close taken at its later seasonal value, so no bar is ever
    # stamped before its session ended.
    close = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
    assert len(store.as_of(OLD, close)) == 2
    assert len(store.as_of(OLD, close + timedelta(minutes=20))) == 3


def test_a_revision_does_not_rewrite_what_was_known_before(store: BitemporalStore) -> None:
    """A split restates history backwards. A query pinned earlier must not see it.

    This is the concrete case for this system: IBKR adjusts prices for splits and
    dividends across the whole history, so a backtest run today would otherwise
    use prices nobody could have seen at the time.
    """
    bars = weekly_bars("2026-01-02", 4, first_close=200.0)
    store.append(OLD, to_observations(bars))
    before_split = store.as_of(OLD, at(2026, 3, 1))["close"].tolist()

    halved = bars.assign(close=bars["close"] / 2, open=bars["open"] / 2)
    restated = to_observations(halved)
    restated["available_time"] = pd.Timestamp("2026-06-01", tz="UTC")
    store.revise(OLD, restated)

    assert store.as_of(OLD, at(2026, 3, 1))["close"].tolist() == before_split
    assert store.as_of(OLD, at(2026, 7, 1))["close"].tolist() == [c / 2 for c in before_split]


def test_the_store_has_no_update_method(store: BitemporalStore) -> None:
    """Append-only is enforced by absence, not by discipline."""
    assert not hasattr(store, "update")
    assert not hasattr(store, "delete")


def test_unknown_fields_are_refused(loaded_store: BitemporalStore) -> None:
    with pytest.raises(ContractViolation, match="unknown field"):
        loaded_store.as_of(OLD, at(2020), fields=["nonexistent"])


# -- the universe ------------------------------------------------------------


def test_an_instrument_is_not_a_member_before_it_listed() -> None:
    universe = derive_memberships(
        [(NEW, at(2019, 4, 5), at(2026, 9, 1))], still_trading_after=at(2026, 6, 1)
    )
    assert NEW not in universe.members_at(at(2011))
    assert NEW in universe.members_at(at(2020))


def test_a_delisted_instrument_keeps_its_record() -> None:
    """The names that disappeared are exactly the ones survivorship bias hides."""
    universe = derive_memberships(
        [(OLD, at(2009), at(2015, 6, 1))], still_trading_after=at(2026, 6, 1)
    )
    assert OLD in universe.members_at(at(2012))
    assert OLD not in universe.members_at(at(2020))
    assert universe.survivors_only() == ()


def test_derived_universe_from_a_store_respects_listing_dates(
    loaded_store: BitemporalStore,
) -> None:
    universe = universe_from_store(loaded_store, still_trading_after=at(2026, 1, 1))
    assert set(universe.members_at(at(2011))) == {OLD}
    assert set(universe.members_at(at(2021))) == {OLD, NEW}


def test_universe_survives_a_round_trip_through_csv(tmp_path) -> None:
    original = derive_memberships(
        [(OLD, at(2009), at(2015)), (NEW, at(2019), at(2026, 9, 1))],
        still_trading_after=at(2026, 6, 1),
    )
    path = tmp_path / "universe.csv"
    original.to_csv(path, derived=True)
    restored = PointInTimeUniverse.from_csv(path)
    assert restored.members_at(at(2012)) == original.members_at(at(2012))
    assert restored.members_at(at(2021)) == original.members_at(at(2021))


# -- the filtration ----------------------------------------------------------


def filtration_at(store: BitemporalStore, moment: datetime) -> StoreFiltration:
    universe = universe_from_store(store, still_trading_after=at(2026, 1, 1))
    return StoreFiltration(store, universe, moment)


def test_a_strategy_cannot_see_an_instrument_that_had_not_listed(
    loaded_store: BitemporalStore,
) -> None:
    """The headline guarantee: a 2011 decision cannot touch a 2019 listing."""
    view = filtration_at(loaded_store, at(2011, 6, 1))
    assert not view.is_available(NEW)
    assert view.history(NEW, "close", 10).empty
    assert NEW not in view.universe()
    assert OLD in view.universe()


def test_history_is_truncated_at_the_decision_time(loaded_store: BitemporalStore) -> None:
    early = filtration_at(loaded_store, at(2012, 1, 1)).history(OLD, "close", 10_000)
    late = filtration_at(loaded_store, at(2020, 1, 1)).history(OLD, "close", 10_000)
    assert len(early) < len(late)
    assert early.tolist() == late.tolist()[: len(early)]


def test_appending_future_data_does_not_change_a_past_view(
    loaded_store: BitemporalStore,
) -> None:
    """Causality, stated as the test that catches every leak.

    Whatever arrives later must leave an earlier decision bit-identical. A ranking
    computed over the full sample, a universe filtered on survivors, or a
    normalisation fitted on everything all show up as a failure on this line.
    """
    before = filtration_at(loaded_store, at(2015, 6, 1)).history(OLD, "close", 500).tolist()
    loaded_store.append(OLD, to_observations(weekly_bars("2026-01-02", 30, first_close=999.0)))
    after = filtration_at(loaded_store, at(2015, 6, 1)).history(OLD, "close", 500).tolist()
    assert before == after


def test_availability_requires_enough_history(loaded_store: BitemporalStore) -> None:
    """Point-in-time seasoning, not a data-quality check."""
    just_listed = filtration_at(loaded_store, at(2019, 5, 1))
    assert just_listed.is_available(NEW, min_bars=1)
    assert not just_listed.is_available(NEW, min_bars=27)
    assert NEW not in just_listed.universe(min_bars=27)


def test_frames_align_without_inventing_prices(loaded_store: BitemporalStore) -> None:
    """A gap stays a gap: forward filling would invent a price that never traded."""
    view = filtration_at(loaded_store, at(2021, 1, 1))
    frame = view.frame([OLD, NEW], "close", 20)
    assert set(frame.columns) == {"old", "new"}
    assert not frame.empty


def test_a_non_positive_count_is_refused(loaded_store: BitemporalStore) -> None:
    with pytest.raises(CausalityViolation):
        filtration_at(loaded_store, at(2020)).history(OLD, "close", 0)


def test_the_filtration_exposes_no_date_range_parameter() -> None:
    """There is no argument through which a future date could be passed."""
    import inspect

    for name in ("history", "frame", "is_available"):
        parameters = set(inspect.signature(getattr(StoreFiltration, name)).parameters)
        assert not parameters & {"start", "end", "until", "as_of", "date_range"}


# -- weekly labelling and completeness ---------------------------------------


def test_a_weekly_bar_is_stamped_at_the_end_of_its_week() -> None:
    """IBKR labels a weekly bar with its first session; it closes on Friday."""
    from data.ingest import bar_close

    monday_label = datetime(2026, 9, 14)
    tuesday_label = datetime(2026, 9, 8)  # a week that began on a holiday Monday
    assert bar_close(monday_label, week_ending=True) == datetime(
        2026, 9, 18, 21, 0, tzinfo=timezone.utc
    )
    assert bar_close(tuesday_label, week_ending=True).date().isoformat() == "2026-09-11"
    assert bar_close(monday_label).date().isoformat() == "2026-09-14", "daily keeps its day"


def test_only_bars_that_have_closed_are_kept() -> None:
    """A finished week fetched on Saturday is kept; the week in progress is not."""
    from data.ingest import complete_bars

    index = pd.to_datetime(["2026-09-07", "2026-09-14", "2026-09-21"])
    bars = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)
    saturday = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
    kept = complete_bars(bars, saturday, week_ending=True)
    assert list(kept["close"]) == [1.0, 2.0]
    friday_before_close = datetime(2026, 9, 18, 20, 30, tzinfo=timezone.utc)
    assert list(complete_bars(bars, friday_before_close, week_ending=True)["close"]) == [1.0]


def test_rows_fetched_late_are_stamped_with_the_fetch() -> None:
    """A live refresh records when the system actually learned the bar."""
    from data.ingest import to_observations

    bars = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-09-14"]))
    fetched = datetime(2026, 9, 19, 9, 30, tzinfo=timezone.utc)
    rows = to_observations(bars, week_ending=True, available_at=fetched)
    assert rows["available_time"].iloc[0] == pd.Timestamp(fetched)
    assert rows["event_time"].iloc[0] == pd.Timestamp("2026-09-18 21:00", tz="UTC")


def test_first_known_ignores_later_restatements(store: BitemporalStore) -> None:
    """A split restates history; when a week became knowable does not change."""
    original = to_observations(weekly_bars("2026-01-02", 10))
    store.append(OLD, original)
    restated = original.copy()
    restated["close"] = restated["close"] / 2
    restated["available_time"] = pd.Timestamp("2026-09-19", tz="UTC")
    store.revise(OLD, restated)
    first = store.first_known(OLD, datetime(2026, 12, 31, tzinfo=timezone.utc))
    assert (first.values == original["available_time"].values).all()
    latest = store.as_of(OLD, datetime(2026, 12, 31, tzinfo=timezone.utc))
    assert latest["close"].iloc[0] == original["close"].iloc[0] / 2, "prices are the latest"


def test_a_split_week_becomes_one_weekly_bar() -> None:
    """Two bars in one ISO week are one week, not a phantom extra one."""
    from data.ingest import merge_split_weeks

    index = pd.to_datetime(["2026-06-29", "2026-07-01", "2026-07-06"])
    bars = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0], "high": [11.0, 13.0, 12.5],
            "low": [9.5, 10.5, 11.5], "close": [10.8, 12.2, 12.1],
            "volume": [100, 50, 80],
        },
        index=index,
    )
    merged = merge_split_weeks(bars)
    assert len(merged) == 2
    week = merged.iloc[0]
    assert (week["open"], week["high"], week["low"], week["close"], week["volume"]) == (
        10.0, 13.0, 9.5, 12.2, 150
    )
    assert merged.index[0] == pd.Timestamp("2026-06-29"), "the vendor's first label is kept"


def test_relabelling_within_a_week_does_not_change_what_is_stored(tmp_path) -> None:
    """The vendor's choice of label is irrelevant once bars are stamped at week-end.

    The same four weeks, labelled once on Mondays and once on assorted weekdays,
    must produce identical observations. This is the half of the 2026-09-21
    change that is supposed to be result-neutral, pinned on its own.
    """
    from data.ingest import to_observations

    values = {"open": [1.0, 2.0, 3.0, 4.0], "close": [1.5, 2.5, 3.5, 4.5]}
    mondays = pd.DataFrame(values, index=pd.to_datetime(
        ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"]))
    assorted = pd.DataFrame(values, index=pd.to_datetime(
        ["2026-08-04", "2026-08-12", "2026-08-17", "2026-08-28"]))
    pd.testing.assert_frame_equal(
        to_observations(mondays, week_ending=True),
        to_observations(assorted, week_ending=True),
    )
