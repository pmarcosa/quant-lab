#!/usr/bin/env python3
"""Add instruments to the cache from IBKR price-history payloads.

    python scripts/import_ibkr_json.py AAPL=aapl.json MSFT=msft.json
    python scripts/import_ibkr_json.py --from-dir incoming/ --freq weekly

Each file is one instrument's response from IBKR's price-history endpoint, saved
as JSON. This is the path that does not need IB Gateway running: the payload can
come from anywhere that speaks to IBKR, including a connector, and the
conversion and validation happen here rather than wherever it was fetched.

After importing, rebuild the store:

    rm -rf var && python scripts/ingest_ibkr_cache.py

The universe is *derived* from the cache, so a new file is a new universe member
with the listing date its own data implies. There is no separate list to edit —
which is the point: a hand-maintained universe list is how a backtest ends up
trading names that had not listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.errors import ContractViolation  # noqa: E402
from data.vendor import (  # noqa: E402
    bars_from_connector,
    cache_inventory,
    check_coverage,
    write_cache_csv,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "ibkr_cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs", nargs="*", metavar="SYMBOL=FILE",
        help="One instrument per argument, e.g. NFLX=netflix.json",
    )
    parser.add_argument(
        "--from-dir", default=None,
        help="Import every FILE.json in this directory, using the filename as the ticker.",
    )
    parser.add_argument("--freq", choices=("weekly", "daily"), default="weekly")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate without writing anything."
    )
    args = parser.parse_args(argv)

    jobs: list[tuple[str, Path]] = []
    for pair in args.pairs:
        if "=" not in pair:
            print(f"expected SYMBOL=FILE, got {pair!r}", file=sys.stderr)
            return 2
        symbol, _, path = pair.partition("=")
        jobs.append((symbol.upper(), Path(path)))
    if args.from_dir:
        for path in sorted(Path(args.from_dir).glob("*.json")):
            jobs.append((path.stem.upper(), path))

    if not jobs:
        parser.print_help()
        return 2

    written = 0
    for symbol, path in jobs:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            bars = bars_from_connector(payload)
        except (OSError, json.JSONDecodeError, ContractViolation) as error:
            # One bad file does not stop the rest, but it is never silent.
            print(f"{symbol:<8} SKIPPED  {error}", file=sys.stderr)
            continue
        span = f"{bars.index[0].date()} to {bars.index[-1].date()}"
        if args.dry_run:
            print(f"{symbol:<8} {len(bars):>5} bars  {span}  (dry run)")
            continue
        write_cache_csv(symbol, bars, CACHE, frequency=args.freq)
        written += 1
        print(f"{symbol:<8} {len(bars):>5} bars  {span}")

    if args.dry_run:
        return 0

    inventory = cache_inventory(CACHE, args.freq)
    thin = check_coverage(inventory)
    print(f"\n{written} written; cache now holds {len(inventory)} {args.freq} instruments")
    if thin:
        print(f"too little history to be selectable yet: {', '.join(thin)}")
    print("\nRebuild the store:  rm -rf var && python scripts/ingest_ibkr_cache.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
