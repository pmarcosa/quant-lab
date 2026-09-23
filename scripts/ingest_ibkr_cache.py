#!/usr/bin/env python3
"""Load the cached IBKR price files into the bitemporal store.

    python scripts/ingest_ibkr_cache.py

Raw CSVs live under ``data/ibkr_cache``: the input, fetched from IBKR by each
user and never committed (the market-data licence forbids redistribution). The
bitemporal store under ``var/store`` is derived: anyone with the CSVs can
rebuild it by running this.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.temporal import BarInterval  # noqa: E402
from data.bitemporal import BitemporalStore  # noqa: E402
from data.ingest import ingest_directory, universe_from_store  # noqa: E402
from data.vendor import FREQUENCIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "ibkr_cache"
STORE = ROOT / "var" / "store"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freq", choices=("weekly", "daily", "hourly", "minute", "both", "all"), default="all",
        help="Which cache folders to ingest. 'all' (default): every one that exists.",
    )
    parser.add_argument(
        "--still-trading-after",
        default="2026-06-01",
        help="A last bar at or after this date means the instrument is presumed live.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete the derived store and rebuild it. Touches var/store only; "
             "the research ledger and the live journal live in state/ and are kept.",
    )
    args = parser.parse_args(argv)

    if args.rebuild and STORE.exists():
        import shutil

        shutil.rmtree(STORE)
        print(f"removed {STORE.relative_to(ROOT)}")

    cutoff = datetime.fromisoformat(args.still_trading_after).replace(tzinfo=timezone.utc)
    if args.freq == "all":
        frequencies = tuple(f for f in FREQUENCIES if (CACHE / f).is_dir())
    elif args.freq == "both":
        frequencies = ("weekly", "daily")
    else:
        frequencies = (args.freq,)

    for frequency in frequencies:
        interval = BarInterval.parse(frequency)
        dataset = f"bars_{interval.value.lower()}"
        store = BitemporalStore(STORE, dataset)
        if store.path.exists():
            print(f"{dataset}: already present, skipping (use --rebuild to rebuild)")
            continue

        # Weekly bars are stamped at their week's close; any bar whose session
        # has not closed yet is left out rather than stored half-finished.
        written = ingest_directory(
            CACHE / frequency, store, interval=interval
        )
        print(f"{dataset}: {len(written)} instruments, {sum(written.values()):,} observations")

        universe = universe_from_store(store, still_trading_after=cutoff)
        universe.to_csv(STORE / f"universe_{frequency}.csv", derived=True)
        ended = [m for m in universe.memberships() if m.delisted is not None]
        print(f"  universe: {len(universe.memberships())} members, {len(ended)} no longer trading")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
