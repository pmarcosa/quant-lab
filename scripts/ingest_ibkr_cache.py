#!/usr/bin/env python3
"""Load the cached IBKR price files into the bitemporal store.

    python scripts/ingest_ibkr_cache.py

Raw CSVs are committed under ``data/ibkr_cache`` because they are the
reproducible input. The bitemporal store under ``var/store`` is derived and is
not committed: anyone can rebuild it by running this.
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

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "ibkr_cache"
STORE = ROOT / "var" / "store"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", choices=("weekly", "daily", "both"), default="both")
    parser.add_argument(
        "--still-trading-after",
        default="2026-06-01",
        help="A last bar at or after this date means the instrument is presumed live.",
    )
    args = parser.parse_args(argv)

    cutoff = datetime.fromisoformat(args.still_trading_after).replace(tzinfo=timezone.utc)
    frequencies = ("weekly", "daily") if args.freq == "both" else (args.freq,)

    for frequency in frequencies:
        interval = BarInterval.WEEK if frequency == "weekly" else BarInterval.DAY
        dataset = f"bars_{interval.value.lower()}"
        store = BitemporalStore(STORE, dataset)
        if store.path.exists():
            print(f"{dataset}: already present, skipping (delete var/store to rebuild)")
            continue

        # The final weekly bar is the week in progress: its high, low and close are
        # not final, and a partial bar is a live-versus-backtest discrepancy.
        written = ingest_directory(
            CACHE / frequency, store, drop_last_bar=(frequency == "weekly")
        )
        print(f"{dataset}: {len(written)} instruments, {sum(written.values()):,} observations")

        universe = universe_from_store(store, still_trading_after=cutoff)
        universe.to_csv(STORE / f"universe_{frequency}.csv", derived=True)
        ended = [m for m in universe.memberships() if m.delisted is not None]
        print(f"  universe: {len(universe.memberships())} members, {len(ended)} no longer trading")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
