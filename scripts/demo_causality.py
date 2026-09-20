#!/usr/bin/env python3
"""Demonstrate the two phase-1 guarantees against the real IBKR data.

    python scripts/demo_causality.py

Guarantee 1 -- point-in-time universe. A backtest standing at 2011 sees the
instruments that were listed in 2011, not the ones we happen to hold in 2026.

Guarantee 2 -- causal admissibility. A filtration pinned at a decision time
cannot return an observation the decision could not have known, whether because
the event had not happened yet or because it had not been published yet. There
is no date-range parameter to get this wrong with: the decision time is the
only knob, and it is set when the view is constructed.

This script asserts. A silent run is a passing run.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.identifiers import InstrumentId  # noqa: E402
from data.bitemporal import BitemporalStore  # noqa: E402
from data.filtration import StoreFiltration  # noqa: E402
from data.universe import PointInTimeUniverse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "var" / "store"


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    if not (STORE / "universe_weekly.csv").exists():
        print("No store. Run: python scripts/ingest_ibkr_cache.py", file=sys.stderr)
        return 1

    store = BitemporalStore(STORE, "bars_1week")
    universe = PointInTimeUniverse.from_csv(STORE / "universe_weekly.csv")

    # ------------------------------------------------------------------
    rule("1. The universe is a function of the decision time")

    moments = ["2011-01-03", "2013-01-07", "2021-01-04", "2026-09-07"]
    sizes = {}
    for moment in moments:
        members = universe.members_at(utc(moment))
        sizes[moment] = len(members)
        print(f"  {moment}: {len(members):>2} instruments")

    assert sizes["2011-01-03"] < sizes["2013-01-07"] < sizes["2021-01-04"], sizes
    assert sizes["2026-09-07"] == len(universe.survivors_only())

    # Instruments that had not listed in 2011 must be absent from the 2011 view
    # even though they are in the file we loaded.
    later = {
        InstrumentId("META"): "2012",
        InstrumentId("GOOGL"): "2015",
        InstrumentId("ZM"): "2019",
        InstrumentId("PLTR"): "2020",
        InstrumentId("CRCL"): "2025",
        InstrumentId("SPCX"): "2026",
    }
    in_2011 = set(universe.members_at(utc("2011-01-03")))
    everything = set(universe.survivors_only())
    for instrument, listed in later.items():
        assert instrument in everything, f"{instrument} missing from the file"
        assert instrument not in in_2011, f"{instrument} (listed {listed}) leaked into 2011"
    print(f"  excluded from 2011: {', '.join(f'{k} ({v})' for k, v in later.items())}")

    # ------------------------------------------------------------------
    rule("2. A pinned filtration cannot see past its decision time")

    decision = utc("2011-01-03")
    view = StoreFiltration(store, universe, decision, min_bars=52)
    aapl = InstrumentId("AAPL")

    closes = view.history(aapl, "close", 100_000)
    print(f"  AAPL closes requested: 100,000   returned: {len(closes):,}")
    print(f"  last bar returned: {closes.index[-1].date()}  (decision: {decision.date()})")
    assert closes.index.max() < decision, "a future bar was returned"

    # The store holds far more AAPL than the view returns. The difference is
    # exactly the part of history that had not happened yet.
    today = utc("2026-09-07")
    all_closes = StoreFiltration(store, universe, today).history(aapl, "close", 100_000)
    print(f"  the same query at {today.date()}: {len(all_closes):,} bars")
    assert len(all_closes) > len(closes) * 2

    # An instrument that is in the store but not yet listed is not merely
    # filtered -- it is not available at all.
    pltr = InstrumentId("PLTR")
    assert store.bars_known_at(pltr, today) > 0, "PLTR should exist in the store"
    assert view.history(pltr, "close", 10).empty, "PLTR returned data before it listed"
    assert not view.is_available(pltr)
    assert pltr not in view.universe()
    print(f"  PLTR: {store.bars_known_at(pltr, today):,} bars in the store, 0 visible in 2011")

    # ------------------------------------------------------------------
    rule("3. Publication lag, not just event time")

    # Find a real bar and step the decision time across its availability.
    # as_of is indexed by event_time and carries available_time alongside, which
    # is the whole point: the two clocks stay separable after the read.
    known = store.as_of(aapl, today)
    position = len(known) // 2
    event_time = known.index[position].to_pydatetime()
    available_time = known["available_time"].iloc[position].to_pydatetime()
    print(f"  a real AAPL bar closed {event_time:%Y-%m-%d %H:%M} UTC")
    print(f"                published {available_time:%Y-%m-%d %H:%M} UTC")
    assert available_time > event_time, "ingest recorded no publication lag"

    between = event_time + (available_time - event_time) / 2
    seen_between = StoreFiltration(store, universe, between).history(aapl, "close", 100_000)
    seen_after = StoreFiltration(
        store, universe, available_time + timedelta(seconds=1)
    ).history(aapl, "close", 100_000)
    print(f"  visible mid-lag: {len(seen_between):,} bars   after publication: {len(seen_after):,}")
    assert event_time not in seen_between.index, "an unpublished bar was visible"
    assert event_time in seen_after.index, "a published bar was hidden"
    assert len(seen_after) == len(seen_between) + 1

    print("\nAll guarantees hold.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
