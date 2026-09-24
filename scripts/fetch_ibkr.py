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

Each symbol takes two daily requests: TRADES (adjusted for splits, not
dividends), which is what the cache keeps, and ADJUSTED_LAST, whose ratio to
TRADES is the day's dividend factor, kept in ``data/ibkr_cache/factors/``. The
store applies the factors when it is built (``data.adjustments``, the expert's
rule: the primary copy is the series as traded, never a pre-adjusted one).
Weekly bars are grouped from the daily ones.

If IB Gateway is not available, use ``scripts/import_ibkr_json.py`` instead —
it takes the same payloads from any source that can talk to IBKR (without
dividend factors: those symbols stay price-only until fetched here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.errors import ContractViolation  # noqa: E402
from data.adjustments import (  # noqa: E402
    factors_from,
    merge_factors,
    read_factors,
    weeks_from_trades,
    write_factors,
)
from data.vendor import cache_inventory, check_coverage, write_cache_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "ibkr_cache"

#: Both frequencies ask for daily bars: IBKR serves ADJUSTED_LAST only for bars
#: of a day or less (321, "Multi day bar size not supported with adjusted
#: last"), and the factors are daily. Weekly bars are grouped here.
BAR_SIZE = "1 day"
DURATION = {"weekly": "22 Y", "daily": "4 Y"}


#: What IBKR's own messages mean for a history request, in plain words.
HINTS = {
    162: "IBKR's historical data service refused the request (the message says why: "
         "most often missing market-data permissions for this login, or pacing)",
    200: "IBKR does not recognise this contract",
    321: "IBKR rejected the request's parameters",
    354: "this login has no market-data subscription for the instrument",
    366: "IBKR sent nothing before the timeout, so the request was cancelled",
    10090: "part of the requested data needs a subscription this login lacks",
    10167: "only delayed data is available to this login",
    10168: "this login has no market data; paper accounts must share the live account's",
}

#: Refusals that a retry of the identical request cannot fix.
REJECTIONS = {200, 321, 354, 10090, 10168}  # not 162: it also reports pacing

#: Connection notices about the data farms. The historical one is "HMDS".
FARM_CODES = {2103, 2104, 2105, 2106, 2107, 2108, 2157, 2158}


def fetch(
    symbols: list[str],
    frequency: str,
    host: str,
    port: int,
    client_id: int,
    duration: str | None = None,
    timeout: float = 120.0,
    attempts: int = 2,
) -> int:
    """Connect, fetch each symbol, write the cache and its factors.

    Returns how many symbols were written. Every message IBKR sends about a
    request is printed with the symbol, so a failure says why rather than only
    that nothing came back.
    """
    try:
        from ib_async import IB, Stock
    except ImportError:  # pragma: no cover - depends on the optional extra
        print(
            'ib_async is not installed. Run: pip install -e ".[ibkr]"',
            file=sys.stderr,
        )
        return 0

    ib = IB()
    messages: list[tuple[int, str]] = []

    def on_error(*event) -> None:  # (reqId, code, message, ...) across ib_async versions
        code, text = int(event[1]), str(event[2])
        messages.append((code, text))

    ib.errorEvent += on_error
    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
    except Exception as error:  # pragma: no cover - needs a live gateway
        print(
            f"could not reach IB Gateway at {host}:{port} ({error}).\n"
            f"Is TWS or IB Gateway running and logged in, with API clients enabled?",
            file=sys.stderr,
        )
        return 0

    for code, text in messages:
        if code in FARM_CODES:
            print(f"gateway  {code}  {text}")
    if any(code in FARM_CODES for code, _ in messages):
        print()

    span = duration or DURATION[frequency]
    written = 0
    try:
        for symbol in symbols:
            contract = Stock(symbol, "SMART", "USD")
            try:
                if not ib.qualifyContracts(contract):
                    print(f"{symbol:<8} SKIPPED  IBKR does not recognise it as a US stock",
                          file=sys.stderr)
                    continue
                request = (ib, contract, symbol, span, messages, timeout, attempts)
                trades = _history(*request, what="TRADES")
                adjusted = _history(*request, what="ADJUSTED_LAST") if trades is not None else None
            except Exception as error:  # pragma: no cover - needs a live gateway
                print(f"{symbol:<8} SKIPPED  {error}", file=sys.stderr)
                continue
            if trades is None or adjusted is None:
                print(f"{symbol:<8} SKIPPED  no bars returned", file=sys.stderr)
                continue
            try:
                factors = factors_from(trades, adjusted)
                merged, _ = merge_factors(read_factors(CACHE, symbol), factors)
                bars = weeks_from_trades(trades, factors) if frequency == "weekly" else trades
                write_cache_csv(symbol, bars, CACHE, frequency=frequency)
                write_factors(CACHE, symbol, merged)
            except ContractViolation as error:
                print(f"{symbol:<8} SKIPPED  {error}", file=sys.stderr)
                continue
            written += 1
            print(
                f"{symbol:<8} {len(bars):>5} bars  "
                f"{bars.index[0].date()} to {bars.index[-1].date()}  "
                f"dividend factor {float(factors.iloc[0]):.3f} at the start"
            )
    finally:
        ib.disconnect()
    return written


def _history(ib, contract, symbol, span, messages, timeout, attempts, what):
    """One daily series, retried; the frame, or None after explaining why not."""
    import pandas as pd
    from ib_async import util

    bars = []
    for attempt in range(1, attempts + 1):
        del messages[:]
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",  # must be empty for ADJUSTED_LAST
            durationStr=span,
            barSizeSetting=BAR_SIZE,
            whatToShow=what,
            useRTH=True,
            formatDate=1,
            timeout=timeout,
        )
        if bars:
            break
        _explain(f"{symbol} {what}", attempt, attempts, timeout, messages)
        if any(code in REJECTIONS for code, _ in messages):
            break  # the same request would be refused again
    frame = util.df(bars) if bars else None
    if frame is None or frame.empty:
        return None
    frame = frame.rename(columns={"date": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame[["open", "high", "low", "close", "volume"]]


def _explain(
    symbol: str, attempt: int, attempts: int, timeout: float, messages: list[tuple[int, str]]
) -> None:
    """Print why a request came back empty, in IBKR's words and in plain ones."""
    print(f"{symbol:<8} attempt {attempt}/{attempts}: no bars", file=sys.stderr)
    relevant = [(c, t) for c, t in messages if c not in FARM_CODES]
    if not relevant:
        print(f"         no reply from IBKR within {timeout:.0f}s", file=sys.stderr)
    for code, text in relevant:
        print(f"         IBKR {code}: {text}", file=sys.stderr)
        if code in HINTS:
            print(f"         -> {HINTS[code]}", file=sys.stderr)
    for code, text in messages:
        if code in FARM_CODES and code in (2103, 2105, 2107, 2157):
            print(f"         data farm: {text}", file=sys.stderr)


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
    parser.add_argument(
        "--duration", default=None,
        help='How far back, IBKR syntax (default: "22 Y" weekly, "4 Y" daily). '
             'A shorter span answers faster.',
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Seconds to wait for each request before giving up (default 120)",
    )
    parser.add_argument("--attempts", type=int, default=2, help="Tries per symbol")
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
    written = fetch(
        symbols, args.freq, args.host, args.port, args.client_id,
        duration=args.duration, timeout=args.timeout, attempts=args.attempts,
    )
    if not written:
        return 1

    inventory = cache_inventory(CACHE, args.freq)
    thin = check_coverage(inventory)
    print(f"\n{written} written; cache now holds {len(inventory)} {args.freq} instruments")
    if thin:
        print(f"too little history to be selectable yet: {', '.join(thin)}")
    print("\nRebuild the store:  python scripts/ingest_ibkr_cache.py --rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
