#!/usr/bin/env python3
"""A wider universe, and a sector gate before the stocks.

    python scripts/compare_sectors.py

Two questions, asked of the strategy as it is configured (top 5, freeze limit,
pace per week, 12% stop, IBKR tiered costs):

1. **A wider universe.** The 53 stocks of ``momentum`` are mostly technology.
   ``momentum-wide`` adds 97 large companies from the other ten sectors. Same
   rules, more to choose from.
2. **A sector gate.** ``sector-momentum`` lets a stock enter only while its
   sector is trending, read from the sector's Select Sector SPDR: the fund
   passes the stock entry filters (``trend``), or ranks among the four best
   sectors by 13-week return (``top 4``). Tried on the wide universe and, to
   tell the gate from the widening, on the 53.

Every variant runs on the four rotation calendars from the same first decision,
late 2007, when the sector funds have the history the gate needs. A variant is
compared with another on the weekly difference of returns, averaged over the
four calendars (they overlap in time, so they are one series, not four
samples), with Newey-West errors. Every run goes to the research ledger: these
are five more trials on top of the ones already counted.

Both universes are lists chosen today, so both carry survivorship bias; the
comparison between them is the reliable part, the levels are optimistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.execution import is_stop_order  # noqa: E402
from contracts.identifiers import RunId, StrategyVersion  # noqa: E402
from risk.rules import (  # noqa: E402
    GrossExposureLimit,
    NetExposureLimit,
    RiskSupervisor,
    ShortSales,
)
from runtime.config import STRATEGY_CONFIGS, interval_of, load_config  # noqa: E402
from runtime.research import (  # noqa: E402
    periodic_dates,
    periodic_returns,
    precompute_indicators,
    run_once,
    trial_label,
)
from runtime.strategies import sector_map  # noqa: E402
from runtime.wiring import load_market, universe_list  # noqa: E402
from scripts.compare_execution import ShiftedMomentum, newey_west_t  # noqa: E402
from scripts.compare_positions import book_shape  # noqa: E402
from strategies.momentum import CADENCE_EPOCH, MomentumParams  # noqa: E402
from strategies.sector_momentum import (  # noqa: E402
    STRATEGY_NAME as SECTOR_NAME,
)
from strategies.sector_momentum import (  # noqa: E402
    SectorMomentum,
    SectorMomentumParams,
)
from validation.ledger import ResearchLedger, Study  # noqa: E402
from validation.metrics import summarise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDY = "sector-gate"
BASE = "momentum · 53 (as traded)"


@dataclass(frozen=True)
class ShiftedSector(SectorMomentum):
    """The sector-gated rules on another of the four rotation calendars."""

    shift: int = 0

    @property
    def version(self) -> StrategyVersion:
        from dataclasses import asdict

        code = self.code_version if self.shift == 0 else f"{self.code_version}+shift{self.shift}"
        return StrategyVersion.of(SECTOR_NAME, asdict(self.params), code_version=code)

    def rotates_at(self, moment: datetime) -> bool:
        weeks = (moment - CADENCE_EPOCH).days // 7
        return (weeks - self.shift) % self.params.rebalance_weeks == 0


def sector_exposure(result, sector_of: dict[str, str]) -> dict:
    """How much of the book sat in each sector, on average over the weeks."""
    totals: dict[str, float] = {}
    for step in result.steps:
        weights = step.decision.target.weights
        for instrument, weight in weights.items():
            key = sector_of.get(str(instrument), "other")
            totals[key] = totals.get(key, 0.0) + abs(weight)
    weeks = max(1, len(result.steps))
    return {k: v / weeks for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(STRATEGY_CONFIGS / "momentum.yaml"))
    parser.add_argument("--store", default=str(ROOT / "var" / "store"))
    parser.add_argument("--ledger", default=str(ROOT / "state" / "research.jsonl"))
    parser.add_argument("--capital", type=float, default=15_000.0)
    parser.add_argument("--start", default="2007-10-05",
                        help="First decision, ISO. The sector funds start in March 2007 and "
                             "the gate needs 27 weeks of them.")
    parser.add_argument("--top-k", type=int, default=4, help="Sectors kept by the top-k gate")
    parser.add_argument("--budget", type=float, default=0.0,
                        help="Seconds after which no new run starts (0: no limit)")
    parser.add_argument("--cache", default=str(ROOT / "state" / "scratch"))
    parser.add_argument("--out", default=str(ROOT / "state" / "scratch" / "compare_sectors.json"))
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    interval = interval_of(config)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    narrow = universe_list(config.strategy.universe)
    wide = universe_list("momentum-wide")
    pairs = sector_map("momentum-wide")
    sector_of = dict(pairs)
    funds = sorted(set(sector_of.values()))
    narrow_pairs = tuple(p for p in pairs if p[0] in set(narrow.symbols))

    def market_of(symbols):
        return load_market(Path(args.store), interval=interval, start=start, symbols=symbols)

    markets = {
        "narrow": market_of(narrow.symbols),
        "narrow+funds": market_of([*narrow.symbols, *funds]),
        "wide": market_of(wide.symbols),
        "wide+funds": market_of([*wide.symbols, *funds]),
    }
    configured = MomentumParams(**dict(config.strategy.params))
    gate = dict(sector_trend=True, sector_top_k=0)
    top = dict(sector_trend=False, sector_top_k=args.top_k)

    def gated(sectors, how):
        return SectorMomentumParams(**{**_fields(configured), "sectors": sectors, **how})

    # name: (market, parameters)
    variants = {
        BASE: ("narrow", configured),
        "momentum · 53 · sector trend gate": ("narrow+funds", gated(narrow_pairs, gate)),
        "wide · 150": ("wide", configured),
        "wide · 150 · sector trend gate": ("wide+funds", gated(pairs, gate)),
        f"wide · 150 · top {args.top_k} sectors": ("wide+funds", gated(pairs, top)),
    }
    for name, market in markets.items():
        if market.missing:
            print(f"{name}: no data for {', '.join(market.missing)}", file=sys.stderr)

    # The same decisions for every variant: the narrowest market's schedule from
    # the first decision on, so the paired differences compare like with like.
    first = max(min(m.schedule) for m in markets.values())
    last = min(max(m.schedule) for m in markets.values())
    schedules = {k: [d for d in m.schedule if first <= d <= last] for k, m in markets.items()}
    indicators = {k: precompute_indicators(m, configured) for k, m in markets.items()}
    execution, risk = config.execution, config.risk
    costs = execution.costs()

    ledger = ResearchLedger(Path(args.ledger))
    series: dict[tuple[str, int], pd.Series] = {}
    measured: dict[tuple[str, int], dict] = {}
    label = " | ".join(f"{k}: {trial_label(markets[k], schedules[k])}" for k in sorted(markets))
    kept_at = Path(args.cache) / f"sectors-{hashlib.sha256(label.encode()).hexdigest()[:12]}.pkl"
    kept: dict = pickle.loads(kept_at.read_bytes()) if kept_at.exists() else {}
    began = time.monotonic()

    def run(name: str, shift: int) -> bool:
        where, params = variants[name]
        note = (f"{name} shift={shift} stop={risk.stop_distance} commission={costs.describe()} "
                f"slip={execution.slippage_bps} capital={args.capital:g}")
        if note in kept:
            series[(name, shift)], measured[(name, shift)] = kept[note]
            return True
        if args.budget and time.monotonic() - began > args.budget:
            return False
        if isinstance(params, SectorMomentumParams):
            strategy = ShiftedSector(params, precomputed=indicators[where], shift=shift)
        else:
            strategy = ShiftedMomentum(params, precomputed=indicators[where], shift=shift)
        supervisor = RiskSupervisor(
            rules=(ShortSales(allowed=risk.allow_short), GrossExposureLimit(risk.max_gross),
                   NetExposureLimit(risk.min_net, risk.net_cap)),
            stop=risk.stop(),
        )
        result = run_once(markets[where], strategy, schedules[where], RunId(f"sg-{shift}"),
                          costs, execution.sizing(interval, allow_short=risk.allow_short),
                          supervisor=supervisor, financing=config.financing.model(),
                          capital=args.capital)
        returns, dates = periodic_returns(result), periodic_dates(result)
        trial = trial_label(markets[where], schedules[where])
        with Study(STUDY, ledger) as study:
            if study.existing(strategy.version, trial, note) is None:
                sd = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
                study.evaluate(strategy.version, trial, {
                    "sharpe": float(returns.mean() / sd) if sd > 0 else 0.0,
                    "final_equity": result.final_equity, "periods": float(returns.size),
                }, returns=returns, note=note, dates=dates)
        stats = summarise(result.equity_curve(), periods_per_year=markets[where].periods_per_year)
        years = (result.steps[-1].marked_at - result.steps[0].decision.decision_time).days / 365.25
        fills = [f for f in result.all_fills() if not is_stop_order(f.client_order_id)]
        series[(name, shift)] = pd.Series(returns, index=pd.to_datetime(list(dates)))
        measured[(name, shift)] = {
            "shift": shift, "cagr": stats.cagr, "volatility": stats.volatility,
            "sharpe": stats.sharpe, "max_drawdown": stats.max_drawdown,
            "final_equity": stats.final_equity, "orders_per_year": len(fills) / years,
            "stops_fired": result.stops_fired(),
            "average_invested": float(np.mean([s.gross_after for s in result.steps])),
            "sectors": sector_exposure(result, sector_of),
            **book_shape(result, args.capital),
        }
        kept[note] = (series[(name, shift)], measured[(name, shift)])
        kept_at.parent.mkdir(parents=True, exist_ok=True)
        kept_at.write_bytes(pickle.dumps(kept))
        print(f"  ran {name} (calendar {shift})", file=sys.stderr, flush=True)
        return True

    calendars = range(configured.rebalance_weeks)
    for name in variants:
        for shift in calendars:
            if not run(name, shift):
                print(f"out of time with {len(kept)} of {len(variants) * len(calendars)} runs "
                      f"kept; run again to go on")
                return 3

    def apart(first_name: str, second_name: str) -> dict:
        gaps = [series[(first_name, s)] - series[(second_name, s)] for s in calendars]
        gap = pd.concat(gaps, axis=1, join="inner").dropna().mean(axis=1)
        halves = {}
        middle = gap.index[len(gap) // 2]
        for part, piece in (("first half", gap[gap.index < middle]),
                            ("second half", gap[gap.index >= middle])):
            halves[part] = {"per_year": float(piece.mean() * 52.0),
                            "t": newey_west_t(piece.to_numpy())}
        by_year = gap.groupby(gap.index.year).sum()
        return {
            "first": first_name, "second": second_name,
            "per_year": float(gap.mean() * 52.0), "t": newey_west_t(gap.to_numpy()),
            "calendars_ahead": int(sum(float(g.dropna().mean()) > 0 for g in gaps)),
            "years_ahead": f"{int((by_year > 0).sum())}/{len(by_year)}",
            "halves": halves,
            "worst_year": {"year": int(by_year.idxmin()), "gap": float(by_year.min())},
            "best_year": {"year": int(by_year.idxmax()), "gap": float(by_year.max())},
        }

    summary = {}
    for name in variants:
        rows = [measured[(name, s)] for s in calendars]
        summary[name] = {
            "rows": rows,
            **{f"mean_{k}": float(np.mean([r[k] for r in rows]))
               for k in ("cagr", "sharpe", "max_drawdown", "volatility", "positions_mean",
                         "average_invested", "orders_per_year", "stops_fired",
                         "rotations_in_cash")},
            "worst_max_drawdown": float(np.min([r["max_drawdown"] for r in rows])),
            "sectors": rows[0]["sectors"],
        }
    pairs_tested = [
        ("wide · 150", BASE),
        ("momentum · 53 · sector trend gate", BASE),
        ("wide · 150 · sector trend gate", BASE),
        (f"wide · 150 · top {args.top_k} sectors", BASE),
        ("wide · 150 · sector trend gate", "wide · 150"),
        (f"wide · 150 · top {args.top_k} sectors", "wide · 150"),
    ]
    output = {
        "settings": {
            "config": Path(args.config).name, "capital": args.capital,
            "window": f"{schedules['narrow'][0].date()}..{schedules['narrow'][-1].date()}",
            "params": {k: v for k, v in sorted(dict(config.strategy.params).items())},
            "commission": costs.describe(), "slippage_bps": execution.slippage_bps,
            "stop": risk.stop_distance, "top_k": args.top_k,
            "universes": {k: len(m.universe.memberships()) for k, m in markets.items()},
        },
        "variants": summary,
        "pairs": [apart(a, b) for a, b in pairs_tested],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=1, default=str))

    s = output["settings"]
    print(f"window {s['window']} · capital {args.capital:,.0f} · {s['commission']} · "
          f"stop {s['stop']:.0%}")
    print(f"{'mean of the four calendars':40s} {'CAGR':>6s} {'Sharpe':>6s} {'max DD':>7s} "
          f"{'worst':>7s} {'names':>5s} {'invested':>8s} {'orders/yr':>9s}")
    for name, block in summary.items():
        print(f"{name:40s} {block['mean_cagr']:6.1%} {block['mean_sharpe']:6.2f} "
              f"{block['mean_max_drawdown']:7.1%} {block['worst_max_drawdown']:7.1%} "
              f"{block['mean_positions_mean']:5.1f} {block['mean_average_invested']:8.0%} "
              f"{block['mean_orders_per_year']:9.0f}")
    print("\npaired, weekly difference averaged over the calendars (Newey-West t)")
    for row in output["pairs"]:
        print(f"{row['first']:34s} − {row['second']:28s} {row['per_year']:+6.2%}/yr "
              f"(t {row['t']:+.2f}) calendars {row['calendars_ahead']}/4, years {row['years_ahead']}"
              f" · halves {row['halves']['first half']['per_year']:+.2%} / "
              f"{row['halves']['second half']['per_year']:+.2%}")
    print(f"\nledger: study '{STUDY}' holds {ledger.count(STUDY)} trials · details in {args.out}")
    return 0


def _fields(params: MomentumParams) -> dict:
    from dataclasses import fields

    return {f.name: getattr(params, f.name) for f in fields(MomentumParams)}


if __name__ == "__main__":
    raise SystemExit(main())
