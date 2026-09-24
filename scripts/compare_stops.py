#!/usr/bin/env python3
"""What the protective stop actually buys, measured on seventeen years.

    python scripts/compare_stops.py

A stop is not free. It converts some drawdown into realised loss, and in a
momentum system — where winners advance through deep retracements — it can cut
exactly the positions that were about to work. The only way to know whether that
trade is worth making is to run both and look.

Every run here is recorded in the research ledger, because a sweep of stop
distances is a sweep of trials and the Deflated Sharpe Ratio needs to know.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.identifiers import RunId  # noqa: E402
from contracts.temporal import BarInterval  # noqa: E402
from engine.decide import SizingPolicy  # noqa: E402
from execution.simulated import CostModel  # noqa: E402
from risk.rules import GrossExposureLimit, ProtectiveStop, RiskSupervisor  # noqa: E402
from runtime.research import precompute_indicators, run_once, trial_label  # noqa: E402
from runtime.wiring import load_market  # noqa: E402
from strategies.momentum import MomentumParams, WeeklyMomentum  # noqa: E402
from validation.ledger import ResearchLedger, Study  # noqa: E402
from validation.metrics import summarise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "var" / "store"
START = datetime(2009, 2, 24, tzinfo=timezone.utc)
DISTANCES = (0.0, 0.08, 0.10, 0.12, 0.15, 0.20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--ledger", default=str(ROOT / "state" / "research.jsonl"))
    args = parser.parse_args(argv)
    store = Path(args.store)
    if not (store / "universe_weekly.csv").exists():
        print("No store. Run: python scripts/ingest_ibkr_cache.py", file=sys.stderr)
        return 1

    market = load_market(store, interval=BarInterval.WEEK, start=START)
    schedule = list(market.schedule)
    params = MomentumParams(rebalance_weeks=4)
    strategy = WeeklyMomentum(params, precomputed=precompute_indicators(market, params))
    costs = CostModel(commission_bps=10.0, slippage_bps=10.0)
    policy = SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.005)
    ledger = ResearchLedger(Path(args.ledger))
    trial = trial_label(market)

    print(f"{len(schedule)} weekly marks, {schedule[0].date()} to {schedule[-1].date()}")
    print(f"{'stop':<10}{'CAGR':>8}{'vol':>8}{'Sharpe':>8}{'Sortino':>9}"
          f"{'maxDD':>9}{'stops':>7}{'final':>13}")
    print("-" * 72)

    rows = {}
    with Study("stop-distance", ledger) as study:
        for distance in DISTANCES:
            supervisor = RiskSupervisor(
                rules=(GrossExposureLimit(maximum=1.0),),
                stop=ProtectiveStop(distance=distance) if distance > 0 else None,
            )
            result = run_once(
                market, strategy, schedule, RunId(f"stop{int(distance * 100)}"),
                costs, policy, supervisor=supervisor,
            )
            stats = summarise(result.equity_curve(), periods_per_year=52)
            import numpy as np

            values = np.array([e for _, e in result.equity_curve()], dtype=float)
            returns = values[1:] / values[:-1] - 1.0
            note = f"protective stop {distance:.0%}"
            # Recorded once: an identical re-run (same label and note) is the
            # same trial, not another one.
            if study.existing(strategy.version, trial, note) is None:
                deviation = returns.std(ddof=1)
                study.evaluate(
                    strategy.version,
                    window=trial,
                    metrics={"sharpe": float(returns.mean() / deviation) if deviation else 0.0,
                             "sharpe_annual": stats.sharpe, "cagr": stats.cagr,
                             "max_drawdown": stats.max_drawdown},
                    returns=returns,
                    note=note,
                )
            label = "none" if distance == 0 else f"fixed {distance:.0%}"
            rows[label] = stats
            print(
                f"{label:<10}{stats.cagr:>8.1%}{stats.volatility:>8.1%}{stats.sharpe:>8.2f}"
                f"{stats.sortino:>9.2f}{stats.max_drawdown:>9.1%}"
                f"{result.stops_fired():>7}{stats.final_equity:>13,.0f}"
            )

    best = max(rows.items(), key=lambda kv: kv[1].sharpe)
    shallowest = max(rows.items(), key=lambda kv: kv[1].max_drawdown)
    print()
    print(f"best Sharpe:          {best[0]} ({best[1].sharpe:.2f})")
    print(f"shallowest drawdown:  {shallowest[0]} ({shallowest[1].max_drawdown:.1%})")
    none = rows["none"]
    print()
    print("A stop is a purchase, not an improvement. Against no stop at all:")
    for label, stats in rows.items():
        if label == "none":
            continue
        print(
            f"  {label:<10} {stats.cagr - none.cagr:+6.1%} CAGR  "
            f"{stats.sharpe - none.sharpe:+5.2f} Sharpe  "
            f"{stats.max_drawdown - none.max_drawdown:+6.1%} drawdown"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
