#!/usr/bin/env python3
"""Fetch bars from a live IBKR connection into the cache.

    python scripts/fetch_ibkr.py --symbols NFLX,ORCL,ADBE --freq weekly
    python scripts/fetch_ibkr.py --refresh            # everything already cached

Needs IB Gateway or TWS running and logged in on this machine, with API socket
clients enabled, and ``pip install -e ".[ibkr]"``.

This is a separate, deliberate step. Nothing in a backtest reaches for the
network: a run that silently re-fetched would give different answers depending
on whether the market was open, and the point of committing the raw CSVs is that
a result can be reproduced years later from the repository alone.

If IB Gateway is not available, use ``scripts/import_ibkr_json.py`` instead —
it takes the same payloads from any source that can talk to IBKR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.errors import ContractViolation  # noqa: E402
from data.vendor import cache_inventory, check_coverage, write_cache_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "ibkr_cache"

#: IBKR caps a single request at roughly a thousand bars. Weekly reaches back
#: about twenty-two years at that limit; daily only about four.
MAX_BARS = 1000

BAR_SIZE = {"weekly": "1 week", "daily": "1 day"}
DURATION = {"weekly": "22 Y", "daily": "4 Y"}


def fetch(symbols: list[str], frequency: str, host: str, port: int, client_id: int) -> int:
    """Connect, fetch each symbol, write the cache. Returns how many were written."""
    try:
        from ib_async import IB, Stock, util
    except ImportError:  # pragma: no cover - depends on the optional extra
        print(
            'ib_async is not installed. Run: pip install -e ".[ibkr]"',
            file=sys.stderr,
        )
        return 0

    import pandas as pd

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
    except Exception as error:  # pragma: no cover - needs a live gateway
        print(
            f"could not reach IB Gateway at {host}:{port} ({error}).\n"
            f"Is TWS or IB Gateway running and logged in, with API clients enabled?",
            file=sys.stderr,
        )
        return 0

    written = 0
    try:
        for symbol in symbols:
            contract = Stock(symbol, "SMART", "USD")
            try:
                ib.qualifyContracts(contract)
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=DURATION[frequency],
                    barSizeSetting=BAR_SIZE[frequency],
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,
                    formatDate=1,
                )
            except Exception as error:  # pragma: no cover - needs a live gateway
                print(f"{symbol:<8} SKIPPED  {error}", file=sys.stderr)
                continue
            if not bars:
                print(f"{symbol:<8} SKIPPED  no bars returned", file=sys.stderr)
                continue

            frame = util.df(bars)
            frame = frame.rename(columns={"date": "timestamp"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp").sort_index()
            try:
                write_cache_csv(symbol, frame, CACHE, frequency=frequency)
            except ContractViolation as error:
                print(f"{symbol:<8} SKIPPED  {error}", file=sys.stderr)
                continue
            written += 1
            print(
                f"{symbol:<8} {len(frame):>5} bars  "
                f"{frame.index[0].date()} to {frame.index[-1].date()}"
            )
    finally:
        ib.disconnect()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers to fetch")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-fetch every instrument already cached"
    )
    parser.add_argument("--freq", choices=("weekly", "daily"), default="weekly")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001, help="4001 gateway, 7496 TWS")
    parser.add_argument("--client-id", type=int, default=17)
    args = parser.parse_args(argv)

    inventory = cache_inventory(CACHE, args.freq)
    if args.refresh:
        symbols = list(inventory["symbol"]) if not inventory.empty else []
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        parser.print_help()
        return 2
    if not symbols:
        print("nothing to fetch", file=sys.stderr)
        return 2

    print(f"fetching {len(symbols)} {args.freq} series from {args.host}:{args.port}\n")
    written = fetch(symbols, args.freq, args.host, args.port, args.client_id)
    if not written:
        return 1

    inventory = cache_inventory(CACHE, args.freq)
    thin = check_coverage(inventory)
    print(f"\n{written} written; cache now holds {len(inventory)} {args.freq} instruments")
    if thin:
        print(f"too little history to be selectable yet: {', '.join(thin)}")
    print("\nRebuild the store:  rm -rf var && python scripts/ingest_ibkr_cache.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
