#!/usr/bin/env python3
"""Add instruments to the cache from IBKR price-history payloads.

    python scripts/import_ibkr_json.py AAPL=aapl.json MSFT=msft.json
    python scripts/import_ibkr_json.py --from-dir incoming/ --freq weekly

Each file is one instrument's response from IBKR's price-history endpoint, saved
as JSON. This is the path that does not need IB Gateway running: the payload can
come from anywhere that speaks to IBKR, including a connector, and the
conversion and validation happen here rather than wherever it was fetched.

With ``--dividends`` (weekly only), a payload that lists its cash dividends
(``include_corporate_actions``) also gets a dividend factor file built from them
(``data.adjustments.factors_from_dividends``), so the store holds total-return
bars for it. A factor file that already exists is left alone: one made by
``fetch_ibkr.py`` from IBKR's own ADJUSTED_LAST series is the better record.

After importing, rebuild the store:

    python scripts/ingest_ibkr_cache.py --rebuild

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
from data.adjustments import (  # noqa: E402
    factors_from_dividends,
    factors_path,
    write_factors,
)
from data.vendor import (  # noqa: E402
    bars_from_connector,
    cache_inventory,
    check_coverage,
    clip_bad_prints,
    connector_dividends,
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
    parser.add_argument(
        "--clip-bad-prints", action="store_true",
        help="Bring isolated bad ticks in a bar's high or low back to its open/close "
             "(data.vendor.clip_bad_prints); every bar changed is listed.",
    )
    parser.add_argument(
        "--dividends", action="store_true",
        help="Weekly only: build dividend factors from the payload's cash dividends, "
             "for symbols that have no factor file yet.",
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
    if args.dividends and args.freq != "weekly":
        print("--dividends builds weekly factors; use it with --freq weekly", file=sys.stderr)
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
        if args.clip_bad_prints:
            bars, changes = clip_bad_prints(bars)
            for change in changes:
                print(f"{symbol:<8} bad print {change}")
        span = f"{bars.index[0].date()} to {bars.index[-1].date()}"
        if args.dry_run:
            print(f"{symbol:<8} {len(bars):>5} bars  {span}  (dry run)")
            continue
        write_cache_csv(symbol, bars, CACHE, frequency=args.freq)
        written += 1
        note = ""
        if args.dividends:
            note = _dividend_factors(symbol, payload, bars)
        print(f"{symbol:<8} {len(bars):>5} bars  {span}{note}")

    if args.dry_run:
        return 0

    inventory = cache_inventory(CACHE, args.freq)
    thin = check_coverage(inventory)
    print(f"\n{written} written; cache now holds {len(inventory)} {args.freq} instruments")
    if thin:
        print(f"too little history to be selectable yet: {', '.join(thin)}")
    print("\nRebuild the store:  python scripts/ingest_ibkr_cache.py --rebuild")
    return 0


def _dividend_factors(symbol: str, payload: dict, bars) -> str:
    """Write the symbol's factor file from the payload's dividends; say what was done."""
    if factors_path(CACHE, symbol).exists():
        return "  (factor file kept)"
    paid = [(ex, amount) for ex, amount, _ in connector_dividends(payload)]
    if not paid:
        return "  (no dividends listed)"
    write_factors(CACHE, symbol, factors_from_dividends(bars, paid))
    return f"  ({len(paid)} dividends -> factors)"


if __name__ == "__main__":
    raise SystemExit(main())
