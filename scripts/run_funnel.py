#!/usr/bin/env python3
"""Put the momentum strategy through all five gates, and report what happens.

    python scripts/run_funnel.py

Every backtest this runs is written to the research ledger before its result is
used, so the trial count behind the Deflated Sharpe Ratio is the real one rather
than a remembered one. That is the difference between a funnel and a formality.

The verdict is printed whatever it is.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.temporal import BarInterval  # noqa: E402
from engine.decide import SizingPolicy  # noqa: E402
from execution.simulated import CostModel  # noqa: E402
from runtime.research import RandomSelection, evaluate  # noqa: E402
from runtime.wiring import load_market  # noqa: E402
from strategies.momentum import MomentumParams, WeeklyMomentum  # noqa: E402
from validation.cpcv import block_sharpes, purged_splits  # noqa: E402
from validation.funnel import (  # noqa: E402
    FunnelReport,
    Verdict,
    cpcv_gate,
    deflated_sharpe_gate,
    monkey_test,
    pbo_gate,
    robustness_gate,
    walk_forward_gate,
    walk_forward_windows,
)
from validation.ledger import ResearchLedger, Study  # noqa: E402
from validation.overfitting import (  # noqa: E402
    deflated_sharpe,
    effective_trials,
    probability_of_backtest_overfitting,
    trial_sharpe_dispersion,
)

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "var" / "store"
START = datetime(2009, 2, 24, tzinfo=timezone.utc)

#: The configurations already tried on this strategy before quant-lab existed:
#: 6 RSI thresholds, 4 cadences, 11 stops, 3 redeployments, 4 sizes, 3 costs.
#: They were not recorded as they ran, so they cannot be reconstructed -- which
#: is exactly the problem the ledger exists to prevent from recurring. They are
#: counted here as a floor on the trial count, not as trials with return series.
PRIOR_TRIALS = 40


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=int, default=60)
    parser.add_argument("--rebalance-weeks", type=int, default=4)
    # The ledger lives in state/, not var/: var/ is derived and safe to delete,
    # and the trial count behind the DSR cannot be rebuilt once it is gone.
    parser.add_argument("--ledger", default=str(ROOT / "state" / "research.jsonl"))
    args = parser.parse_args(argv)

    if not (STORE / "universe_weekly.csv").exists():
        print("No store. Run: python scripts/ingest_ibkr_cache.py", file=sys.stderr)
        return 1

    market = load_market(STORE, interval=BarInterval.WEEK, start=START)
    schedule = list(market.schedule)
    params = MomentumParams(rebalance_weeks=args.rebalance_weeks)
    strategy = WeeklyMomentum(params)
    costs = CostModel(commission_bps=10.0, slippage_bps=10.0)
    policy = SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.005)
    ledger = ResearchLedger(Path(args.ledger))

    print(f"{len(schedule)} weekly marks, {schedule[0].date()} to {schedule[-1].date()}")
    print(f"rotating every {params.rebalance_weeks} weeks, top {params.top_n}\n")

    with Study("funnel", ledger) as study:
        _, returns = evaluate(
            study, market, strategy, schedule, "full", costs, policy, note="baseline"
        )

        done = ledger.count("funnel")
        print(f"running {args.controls} random controls (resuming from {done}) ...", flush=True)
        control_returns = []
        for seed in range(args.controls):
            _, control = evaluate(
                study, market, RandomSelection(params, seed), schedule,
                "full", costs, policy, note=f"random control {seed}",
            )
            control_returns.append(control)

        print("running cost and parameter perturbations ...", flush=True)
        perturbed: list[float] = []
        perturbation_returns = []
        for bps in (5.0, 15.0, 20.0, 30.0, 40.0):
            _, series = evaluate(
                study, market, strategy, schedule, "full",
                CostModel(commission_bps=bps, slippage_bps=bps), policy,
                note=f"cost {bps:.0f}bps per side",
            )
            perturbed.append(_sharpe(series))
            perturbation_returns.append(series)
        for lookback in (10, 11, 12, 14, 15, 16):
            variant = WeeklyMomentum(
                MomentumParams(lookback_weeks=lookback, rebalance_weeks=params.rebalance_weeks)
            )
            _, series = evaluate(
                study, market, variant, schedule, "full", costs, policy,
                note=f"lookback {lookback}w",
            )
            perturbed.append(_sharpe(series))
            perturbation_returns.append(series)
        for top_n in (3, 5, 6):
            variant = WeeklyMomentum(
                MomentumParams(top_n=top_n, rebalance_weeks=params.rebalance_weeks)
            )
            _, series = evaluate(
                study, market, variant, schedule, "full", costs, policy,
                note=f"top {top_n}",
            )
            perturbed.append(_sharpe(series))
            perturbation_returns.append(series)

    print(f"ledger: {ledger.count('funnel')} trials recorded\n")

    # -- gate 1 -------------------------------------------------------------
    gate1 = monkey_test(returns, control_returns)

    # -- gate 2 -------------------------------------------------------------
    splits = list(
        purged_splits(
            len(returns), holding_bars=params.rebalance_weeks, blocks=10, embargo_bars=9
        )
    )
    gate2 = cpcv_gate(block_sharpes(returns, splits))

    # -- gate 3 -------------------------------------------------------------
    inside, outside = walk_forward_windows(returns, train=156, test=52)
    gate3 = walk_forward_gate(inside, outside)

    # -- gate 4 -------------------------------------------------------------
    matrix, kept = ledger.returns_matrix("funnel")
    n_effective = max(effective_trials(matrix), 1.0)
    # The forty configurations tried before the ledger existed have no return
    # series, so they cannot be de-correlated. Counting them at face value is the
    # conservative choice and the honest one.
    n_total = n_effective + PRIOR_TRIALS
    dispersion = trial_sharpe_dispersion(matrix)
    dsr = deflated_sharpe(returns, n_effective=n_total, trial_sharpe_std=dispersion)
    gate4a = deflated_sharpe_gate(
        dsr.dsr, dsr.z, dsr.observed_sharpe, dsr.expected_max, n_total
    )

    variants = np.column_stack([returns, *perturbation_returns])
    overfitting = probability_of_backtest_overfitting(variants, blocks=10)
    gate4b = pbo_gate(overfitting.pbo, overfitting.combinations)

    # -- gate 5 -------------------------------------------------------------
    gate5 = robustness_gate(_sharpe(returns), perturbed)

    report = FunnelReport((gate1, gate2, gate3, gate4a, gate4b, gate5))
    print(report.render())
    print()
    print(f"trials in ledger        {len(kept)} with return series, {ledger.count('funnel')} total")
    print(f"effective (de-correlated) {n_effective:.1f}")
    print(f"prior undocumented      {PRIOR_TRIALS} (counted at face value)")
    print(f"trial Sharpe dispersion {dispersion:.4f}")
    print(f"skew {dsr.skew:+.2f}  kurtosis {dsr.kurtosis:.1f}  over {dsr.periods} weeks")
    return 0 if report.verdict is Verdict.PASS else 2


def _sharpe(returns: np.ndarray) -> float:
    deviation = returns.std(ddof=1)
    return float(returns.mean() / deviation) if deviation > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
