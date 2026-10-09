#!/usr/bin/env python3
"""How many names the momentum book should hold, and which pace filter.

    python scripts/compare_positions.py

The strategy holds every name that passes its filters and keeps -- freezes -- a
held name that no longer passes but has triggered no exit. With a small account
that can mean many small positions. Two things limit it: ranking the names that
pass and splitting the book between the best N, and limiting how long a name
may stay frozen.

This runs the configured strategy as it is; with N from 4 to 10; and, for a few
values of N, with each way of handling a frozen name (kept, kept for at most one
or two rotations, halved at each rotation, sold). Every combination runs on the
four rotation calendars and is compared with the strategy as configured on the
weekly difference of returns.

It also runs the pace filter both ways -- the last four weeks' return against
40% of the quarter's, and the same comparison per week -- with everything else
equal, under the present rules and under the ones in force until 2026-10-09.

Three more switches run on the live calendar only: the stop-loss on cost, a
plain stop instead of a stop-limit, and limit orders from the decision price.

Everything else -- universe, costs, sizing, stop -- comes from the strategy's
config. Every run is recorded in the research ledger: this is a search over
some thirty variants, and the best of thirty is partly luck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.execution import is_stop_order  # noqa: E402
from contracts.identifiers import RunId  # noqa: E402
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
from runtime.wiring import load_market, universe_list  # noqa: E402
from scripts.compare_execution import (  # noqa: E402
    ShiftedMomentum,
    fees_at_todays_size,
    newey_west_t,
)
from strategies.momentum import MomentumParams  # noqa: E402
from validation.ledger import ResearchLedger, Study  # noqa: E402
from validation.metrics import summarise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDY = "position-count"
ALL = "all · frozen kept"


class OutOfTime(Exception):
    """The time allowed for one invocation ran out before every run was done."""


def book_shape(result, account: float) -> dict:
    """How many positions the book held, and how small the smallest target was."""
    held = [len(step.book_after.positions) for step in result.steps]
    rotations = [s for s in result.steps if s.decision.target.diagnostics.get("rotated")]
    passing = [int(s.decision.target.diagnostics.get("eligible", 0)) for s in rotations]
    frozen = [int(s.decision.target.diagnostics.get("frozen", 0)) for s in rotations]
    frozen_weight = [s.decision.target.diagnostics.get("frozen_weight", 0.0) for s in rotations]
    smallest = [
        min(s.decision.target.weights.values()) for s in rotations if s.decision.target.weights
    ]
    dollars = np.asarray(smallest) * account
    empty = sum(1 for s in rotations if not s.decision.target.weights)
    return {
        "positions_mean": float(np.mean(held)), "positions_median": float(np.median(held)),
        "positions_max": int(np.max(held)),
        "weeks_with_10_or_more": float(np.mean(np.asarray(held) >= 10)),
        "weeks_with_2_or_fewer": float(np.mean(np.asarray(held) <= 2)),
        "passing_median": float(np.median(passing)), "passing_max": int(np.max(passing)),
        "passing_p90": float(np.percentile(passing, 90)),
        "frozen_mean": float(np.mean(frozen)),
        "frozen_weight_mean": float(np.mean(frozen_weight)),
        "frozen_weight_p90": float(np.percentile(frozen_weight, 90)),
        "rotations": len(rotations), "rotations_in_cash": empty,
        "smallest_target_median": float(np.median(dollars)) if dollars.size else 0.0,
        "smallest_target_p10": float(np.percentile(dollars, 10)) if dollars.size else 0.0,
        "rotations_with_a_target_under_1000": (
            float(np.mean(dollars < 1_000.0)) if dollars.size else 0.0),
        "rotations_with_a_target_under_500": (
            float(np.mean(dollars < 500.0)) if dollars.size else 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(STRATEGY_CONFIGS / "momentum.yaml"))
    parser.add_argument("--store", default=str(ROOT / "var" / "store"))
    parser.add_argument("--ledger", default=str(ROOT / "state" / "research.jsonl"))
    parser.add_argument("--capital", type=float, default=16_729.0,
                        help="Opening equity: the account's size today")
    parser.add_argument("--tops", default="4,5,6,7,8,9,10",
                        help="Limits tried with frozen names kept")
    parser.add_argument("--crossed", default="0,5,6,8",
                        help="Limits tried with every way of handling frozen names (0: all)")
    parser.add_argument("--budget", type=float, default=0.0,
                        help="Seconds after which no new run is started (0: no limit). "
                             "Finished runs are kept, so running again goes on.")
    parser.add_argument("--cache", default=str(ROOT / "state" / "scratch"))
    parser.add_argument("--pace-ratios", default="",
                        help="Also run these pace ratios (on total returns), comma-separated, "
                             "to see whether the result is a plateau or a peak")
    parser.add_argument("--start", default=None)
    parser.add_argument("--out", default=str(ROOT / "state" / "scratch" / "compare_positions.json"))
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    interval = interval_of(config)
    universe = universe_list(config.strategy.universe)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    market = load_market(Path(args.store), interval=interval, start=start,
                         symbols=universe.symbols if universe else None)
    base = MomentumParams(**dict(config.strategy.params))
    base = replace(base, top_n=0)
    schedule = list(market.schedule)[base.warmup_weeks + base.min_history_weeks:]
    indicators = precompute_indicators(market, base)
    execution, risk = config.execution, config.risk
    tops = [int(t) for t in args.tops.split(",") if t]

    # What happens to a held name that no longer qualifies and has triggered no
    # exit. The two limits are the project expert's ways of keeping frozen
    # names from piling up.
    freezing = {
        "kept": {},
        "at most 2 rotations": {"freeze_rotations": 2},
        "at most 1 rotation": {"freeze_rotations": 1},
        "halved each rotation": {"freeze_fade": 0.5},
        "sold": {"hold_unqualified": False},
    }
    crossed = [int(t) for t in args.crossed.split(",") if t != ""]

    def named(n: int, mode: str) -> str:
        return f"{'all' if n == 0 else f'top {n}'} · frozen {mode}"

    # name: (strategy parameters, execution settings, risk settings)
    variants = {ALL: (base, execution, risk)}
    for n in tops:
        variants[named(n, "kept")] = (replace(base, top_n=n), execution, risk)
    for mode, change in freezing.items():
        for n in crossed:
            variants.setdefault(named(n, mode), (replace(base, top_n=n, **change), execution, risk))
    # The pace filter both ways. Compared per week it is the same test as on
    # totals with the ratio scaled by the two horizons, so one parameter moves.
    per_week = base.pace_ratio_min * base.pace_weeks / base.lookback_weeks
    before = replace(base, top_n=4, pace_ratio_min=per_week, hold_unqualified=False,
                     cost_stop_loss=0.0)
    pace = {
        "pace per week · all · frozen kept":
            (replace(base, pace_ratio_min=per_week), execution, risk),
        "pace per week · top 4 · frozen sold (rules before 9 Oct)": (before, execution, risk),
        named(4, "sold"): (replace(base, top_n=4, hold_unqualified=False), execution, risk),
    }
    variants.update(pace)
    ratios = [float(r) for r in args.pace_ratios.split(",") if r != ""]
    for ratio in ratios:
        variants[f"pace at least {ratio:.0%} of the quarter · all · frozen kept"] = (
            replace(base, pace_ratio_min=ratio), execution, risk)
    singles = {
        "no stop-loss on cost": (replace(base, cost_stop_loss=0.0), execution, risk),
        "plain stop instead of stop-limit":
            (base, execution, replace(risk, stop_limit_offset=None)),
        "limit orders within 2% of the decision price":
            (base, replace(execution, limit_band=0.02), risk),
    }
    grid = list(variants)
    variants.update(singles)
    #: Pairs that differ in the pace filter and in nothing else.
    pace_pairs = [
        ("pace per week · all · frozen kept", ALL),
        ("pace per week · top 4 · frozen sold (rules before 9 Oct)", named(4, "sold")),
    ]

    ledger = ResearchLedger(Path(args.ledger))
    series: dict[tuple[str, int], pd.Series] = {}
    measured: dict[tuple[str, int], dict] = {}

    how_price = execution.share_price or 100.0
    label = trial_label(market, schedule)
    # Finished runs are kept between invocations, keyed by what they saw (dates,
    # data and code), so a long comparison can be run in several sittings.
    kept_at = Path(args.cache) / f"positions-{hashlib.sha256(label.encode()).hexdigest()[:12]}.pkl"
    kept: dict = pickle.loads(kept_at.read_bytes()) if kept_at.exists() else {}
    began = time.monotonic()

    def run(name: str, shift: int) -> None:
        params, how, limits = variants[name]
        costs = how.costs()
        note = (f"{name} shift={shift} stop={limits.stop_distance} "
                f"stop_limit={limits.stop_limit_offset} commission={costs.describe()} "
                f"slip={how.slippage_bps} buffer={how.cash_buffer:g} band={how.no_trade_band:g} "
                f"limit_band={how.limit_band} capital={args.capital:g} "
                f"universe={universe.name}:{universe.fingerprint}")
        if note in kept:
            series[(name, shift)], measured[(name, shift)] = kept[note]
            return
        if args.budget and time.monotonic() - began > args.budget:
            raise OutOfTime
        strategy = ShiftedMomentum(params, precomputed=indicators, shift=shift)
        supervisor = RiskSupervisor(
            rules=(ShortSales(allowed=limits.allow_short), GrossExposureLimit(limits.max_gross),
                   NetExposureLimit(limits.min_net, limits.net_cap)),
            stop=limits.stop(),
        )
        result = run_once(market, strategy, schedule, RunId(f"pc-{shift}"), costs,
                          how.sizing(interval, allow_short=limits.allow_short),
                          supervisor=supervisor, financing=config.financing.model(),
                          capital=args.capital)
        returns, dates = periodic_returns(result), periodic_dates(result)
        with Study(STUDY, ledger) as study:
            if study.existing(strategy.version, label, note) is None:
                sd = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
                study.evaluate(strategy.version, label, {
                    "sharpe": float(returns.mean() / sd) if sd > 0 else 0.0,
                    "final_equity": result.final_equity, "periods": float(returns.size),
                }, returns=returns, note=note, dates=dates)
        stats = summarise(result.equity_curve(), periods_per_year=market.periods_per_year)
        years = (result.steps[-1].marked_at - result.steps[0].decision.decision_time).days / 365.25
        fills = [f for f in result.all_fills() if not is_stop_order(f.client_order_id)]
        fees = fees_at_todays_size(result, args.capital, how_price)
        series[(name, shift)] = pd.Series(returns, index=pd.to_datetime(list(dates)))
        measured[(name, shift)] = {
            "shift": shift, "cagr": stats.cagr, "volatility": stats.volatility,
            "sharpe": stats.sharpe, "max_drawdown": stats.max_drawdown,
            "final_equity": stats.final_equity, "orders_per_year": len(fills) / years,
            "stops_fired": result.stops_fired(),
            "average_invested": float(np.mean([s.gross_after for s in result.steps])),
            "turnover_per_year": sum(result.turnover()) / years,
            "fees_today_percent": fees["percent_of_account_per_year"]["ibkr-tiered"],
            "orders_today_per_year": fees["orders_per_year"],
            "median_order_today": fees["median_order"],
            "orders_today_below_1000": fees["orders_below_1000"],
            **book_shape(result, args.capital),
        }
        kept[note] = (series[(name, shift)], measured[(name, shift)])
        kept_at.parent.mkdir(parents=True, exist_ok=True)
        kept_at.write_bytes(pickle.dumps(kept))
        print(f"  ran {name} (calendar {shift})", file=sys.stderr, flush=True)

    def describe(name: str, shift: int) -> dict:
        both = pd.concat([series[(name, shift)], series[(ALL, shift)]], axis=1,
                         join="inner").dropna()
        difference = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
        return {**measured[(name, shift)],
                "vs_all_per_year": float(difference.mean() * 52.0),
                "t": newey_west_t(difference)}

    calendars = range(base.rebalance_weeks)
    output: dict = {"settings": {
        "config": Path(args.config).name,
        "universe": universe.describe() if universe else "store",
        "with_data": len(market.universe.memberships()), "missing": list(market.missing),
        "window": f"{schedule[0].date()}..{schedule[-1].date()}", "capital": args.capital,
        "params": {k: v for k, v in sorted(dict(config.strategy.params).items())},
        "execution": {"commission": execution.costs().describe(),
                      "cash_buffer": execution.cash_buffer, "no_trade_band": execution.no_trade_band,
                      "limit_band": execution.limit_band, "slippage_bps": execution.slippage_bps},
        "stop": risk.stop_distance, "stop_limit_offset": risk.stop_limit_offset,
    }, "live": {}, "calendars": {}}

    try:
        for shift in calendars:
            run(ALL, shift)
        for name in variants:
            run(name, 0)
        for name in grid:
            for shift in calendars:
                run(name, shift)
    except OutOfTime:
        todo = len(grid) * len(calendars) + len(singles)
        print(f"out of time with {len(kept)} of about {todo} runs kept; run again to go on")
        return 3
    for name in variants:
        output["live"][name] = describe(name, 0)
    for name in grid:
        rows = []
        for shift in calendars:
            rows.append(describe(name, shift))
        # The four calendars overlap in time, so they are not four samples. Their
        # average each week is one series, and that is what is tested.
        apart = pd.concat(
            [series[(name, shift)] - series[(ALL, shift)] for shift in calendars],
            axis=1, join="inner",
        ).dropna().mean(axis=1).to_numpy()
        output["calendars"][name] = {
            "rows": rows,
            "ensemble_vs_all_per_year": float(apart.mean() * 52.0) if apart.size else 0.0,
            "ensemble_t": newey_west_t(apart),
            **{f"mean_{k}": float(np.mean([r[k] for r in rows]))
               for k in ("cagr", "sharpe", "max_drawdown", "volatility", "positions_mean",
                         "vs_all_per_year", "average_invested", "frozen_weight_mean",
                         "smallest_target_median", "rotations_with_a_target_under_500",
                         "turnover_per_year", "orders_today_per_year")},
            "max_positions": int(max(r["positions_max"] for r in rows)),
            "worst_max_drawdown": float(np.min([r["max_drawdown"] for r in rows])),
            "calendars_ahead": int(sum(r["vs_all_per_year"] > 0 for r in rows)),
        }

    def apart(first: str, second: str) -> dict:
        """One variant against another: the four calendars' weekly average."""
        gap = pd.concat(
            [series[(first, shift)] - series[(second, shift)] for shift in calendars],
            axis=1, join="inner",
        ).dropna().mean(axis=1).to_numpy()
        ahead = sum(
            float((series[(first, shift)] - series[(second, shift)]).dropna().mean()) > 0
            for shift in calendars
        )
        return {"first": first, "second": second, "per_year": float(gap.mean() * 52.0),
                "t": newey_west_t(gap), "calendars_ahead": int(ahead)}

    output["pace"] = [apart(a, b) for a, b in pace_pairs]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=1, default=str))

    s = output["settings"]
    print(f"window {s['window']} · universe {s['universe']} ({s['with_data']} with data) · "
          f"capital {args.capital:,.0f}")
    print(f"{'on the live calendar':46s} {'CAGR':>6s} {'Sharpe':>6s} {'max DD':>7s} "
          f"{'names':>6s} {'max':>4s} {'vs all':>7s} {'t':>6s}")
    for name, row in output["live"].items():
        print(f"{name:46s} {row['cagr']:6.1%} {row['sharpe']:6.2f} {row['max_drawdown']:7.1%} "
              f"{row['positions_mean']:6.1f} {row['positions_max']:4d} "
              f"{row['vs_all_per_year']:+7.2%} {row['t']:6.2f}")
    print("\nmean of the four rotation calendars")
    for name, block in output["calendars"].items():
        print(f"{name:46s} {block['mean_cagr']:6.1%} {block['mean_sharpe']:6.2f} "
              f"{block['mean_max_drawdown']:7.1%} (worst {block['worst_max_drawdown']:6.1%}) "
              f"names {block['mean_positions_mean']:4.1f} vs all "
              f"{block['mean_vs_all_per_year']:+6.2%} (t {block['ensemble_t']:+.2f}) "
              f"ahead on {block['calendars_ahead']}/4")
    print("\nthe pace filter, everything else equal (per week minus total returns)")
    for row in output["pace"]:
        print(f"{row['first'][:58]:58s} {row['per_year']:+6.2%} a year (t {row['t']:+.2f}) "
              f"ahead on {row['calendars_ahead']}/4")
    print(f"\nledger: study '{STUDY}' holds {ledger.count(STUDY)} trials · details in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
