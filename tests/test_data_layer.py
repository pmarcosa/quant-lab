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
    close = datetime(2026, 9, 18, 20, 0, tzinfo=timezone.utc)
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
