#!/usr/bin/env python3
"""Put the momentum strategy through all five gates, and report what happens.

    python scripts/run_funnel.py                          # top 4, every 4 weeks, whole store
    python scripts/run_funnel.py --universe sector-etfs --top 3 --rebalance-weeks 2
    python scripts/run_funnel.py --universe sector-etfs \\
        --grid-top 1,2,3,4,5 --grid-rebalance 1,2,4,6,8   # choose N and K first

Every backtest this runs is written to the research ledger before its result is
used, so the trial count behind the Deflated Sharpe Ratio is the real one rather
than a remembered one. That is the difference between a funnel and a formality.

The verdict is printed whatever it is.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.temporal import BarInterval  # noqa: E402
from engine.decide import SizingPolicy  # noqa: E402
from execution.simulated import CostModel  # noqa: E402
from runtime.research import STARTING_CAPITAL, RandomSelection, evaluate  # noqa: E402
from runtime.wiring import load_market, universe_list  # noqa: E402
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
    parser.add_argument("--top", type=int, default=4, help="Positions held (N)")
    parser.add_argument("--rebalance-weeks", type=int, default=4, help="Weeks between rotations (K)")
    parser.add_argument("--lookback", type=int, default=13, help="Momentum lookback, weeks")
    parser.add_argument("--universe", default=None,
                        help="data/universes/<name>.txt, or a file (default: the whole store)")
    parser.add_argument("--start", default=START.date().isoformat(), help="First decision, ISO")
    parser.add_argument("--capital", type=float, default=STARTING_CAPITAL,
                        help="Fixed before validating; the funnel also runs half and double")
    parser.add_argument("--grid-top", default=None,
                        help="Grid mode: comma-separated N values, e.g. 1,2,3,4,5")
    parser.add_argument("--grid-rebalance", default=None,
                        help="Grid mode: comma-separated K values, e.g. 1,2,4,6,8")
    # The ledger lives in state/, not var/: var/ is derived and safe to delete,
    # and the trial count behind the DSR cannot be rebuilt once it is gone.
    parser.add_argument("--ledger", default=str(ROOT / "state" / "research.jsonl"))
    parser.add_argument("--store", default=str(STORE))
    args = parser.parse_args(argv)

    store = Path(args.store)
    if not (store / "universe_weekly.csv").exists():
        print("No store. Run: python scripts/ingest_ibkr_cache.py", file=sys.stderr)
        return 1

    universe = universe_list(args.universe)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    market = load_market(store, interval=BarInterval.WEEK, start=start,
                         symbols=universe.symbols if universe else None)
    schedule = list(market.schedule)
    costs = CostModel(commission_bps=10.0, slippage_bps=10.0)
    policy = SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.005)
    ledger = ResearchLedger(Path(args.ledger))
    # What the data was: dates, prices (a re-fetch or another universe changes
    # it) and the capital. A result recorded under another label is never reused.
    label = f"{schedule[0].date()}..{schedule[-1].date()} data={market.fingerprint()}"
    context = (f"universe={universe.name}:{universe.fingerprint}" if universe else "universe=store")
    if args.capital != STARTING_CAPITAL:
        context += f" capital={args.capital:g}"

    members = len(market.universe.memberships())
    print(f"universe: {universe.describe() if universe else 'every instrument in the store'}, "
          f"{members} with data")
    if market.missing:
        print(f"  not in the store, left out: {', '.join(market.missing)}")
    print(f"{len(schedule)} weekly marks, {schedule[0].date()} to {schedule[-1].date()}")

    if args.grid_top or args.grid_rebalance:
        return run_grid(args, market, schedule, costs, policy, ledger, label, context, members)

    params = MomentumParams(top_n=args.top, rebalance_weeks=args.rebalance_weeks,
                            lookback_weeks=args.lookback)
    strategy = WeeklyMomentum(params)
    print(f"rotating every {params.rebalance_weeks} weeks, top {params.top_n}, "
          f"lookback {params.lookback_weeks} weeks\n")

    def run(study, candidate, note, cost=costs, capital=args.capital):
        return evaluate(study, market, candidate, schedule, label, cost, policy,
                        note=f"{note} {context}", capital=capital)

    with Study("funnel", ledger) as study:
        _, returns = run(study, strategy, "baseline")

        print(f"running {args.controls} random controls ...", flush=True)
        control_returns = [
            run(study, RandomSelection(params, seed), f"random control {seed}")[1]
            for seed in range(args.controls)
        ]

        print("running cost, parameter and capital perturbations ...", flush=True)
        perturbed: list[float] = []
        perturbation_returns = []
        for bps in (5.0, 15.0, 20.0, 30.0, 40.0):
            _, series = run(study, strategy, f"cost {bps:.0f}bps per side",
                            cost=CostModel(commission_bps=bps, slippage_bps=bps))
            perturbed.append(_sharpe(series))
            perturbation_returns.append(series)
        for lookback in _around(params.lookback_weeks, (-3, -2, -1, 1, 2, 3), low=2):
            variant = WeeklyMomentum(replace(params, lookback_weeks=lookback))
            _, series = run(study, variant, f"lookback {lookback}w")
            perturbed.append(_sharpe(series))
            perturbation_returns.append(series)
        for top_n in _around(params.top_n, (-1, 1, 2), low=1, high=max(1, members - 1)):
            variant = WeeklyMomentum(replace(params, top_n=top_n))
            _, series = run(study, variant, f"top {top_n}")
            perturbed.append(_sharpe(series))
            perturbation_returns.append(series)
        # The expert: capital is fixed before validating, and varying it is a
        # test of friction (whole shares), not a parameter to choose.
        for factor in (0.5, 2.0):
            _, series = run(study, strategy, f"capital x{factor:g}",
                            capital=args.capital * factor)
            perturbed.append(_sharpe(series))
            perturbation_returns.append(series)

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
    # Every trial of this research line counts, whatever study recorded it: the
    # manual backtests, earlier funnels, grids, other universes (the expert).
    matrix, kept, others = ledger.research_line(str(strategy.version.strategy))
    n_effective = max(effective_trials(matrix), 1.0)
    # Trials whose series cannot be aligned, and the forty configurations tried
    # before the ledger existed, cannot be de-correlated. Counting them at face
    # value is the conservative choice and the honest one.
    n_total = n_effective + len(others) + PRIOR_TRIALS
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
    print(f"research line           {len(kept) + len(others)} trials of "
          f"{strategy.version.strategy} in the ledger, every study")
    print(f"  aligned, de-correlated {len(kept)} -> N_eff {n_effective:.1f}")
    print(f"  other windows         {len(others)} (counted at face value)")
    print(f"  prior undocumented    {PRIOR_TRIALS} (counted at face value)")
    print(f"trial Sharpe dispersion {dispersion:.4f}")
    print(f"skew {dsr.skew:+.2f}  kurtosis {dsr.kurtosis:.1f}  over {dsr.periods} weeks")
    return 0 if report.verdict is Verdict.PASS else 2


def run_grid(args, market, schedule, costs, policy, ledger, label, context, members) -> int:
    """Every (N, K) on the grid, judged by its worst out-of-sample splits.

    The expert's selection rule: a plateau, not a peak. For each combination
    the 252 purged CPCV splits give a distribution of out-of-sample Sharpe; its
    10th percentile is the combination's score. A combination is then judged by
    the worst score in its neighbourhood (itself and the adjacent N and K), and
    the best of those is the plateau's centre. A combination that only scores
    well where its neighbours do not is a peak fitted to noise.

    Adaptation: the expert clusters the best combinations with k-means and takes
    the winning centroid. On a grid of a few dozen points k-means depends on its
    seed and on how many clusters one asks for; the worst-neighbour rule asks the
    same question (is the neighbourhood good, not just the point) deterministically.

    Every combination is a trial in the ledger (study "grid"), and the funnel's
    Deflated Sharpe for whichever one is chosen counts them all.
    """
    tops = _values(args.grid_top, [args.top])
    rebalances = _values(args.grid_rebalance, [args.rebalance_weeks])
    too_many = [n for n in tops if n >= members]
    if too_many:
        print(f"N must be below the {members} instruments with data; dropping {too_many}",
              file=sys.stderr)
        tops = [n for n in tops if n < members]
    if not tops:
        return 1
    print(f"grid: N in {tops}, K in {rebalances}, lookback {args.lookback} weeks "
          f"({len(tops) * len(rebalances)} backtests, each one a trial)\n", flush=True)

    p10 = np.full((len(tops), len(rebalances)), np.nan)
    median = np.full_like(p10, np.nan)
    series = []
    with Study("grid", ledger) as study:
        for i, n in enumerate(tops):
            for j, k in enumerate(rebalances):
                strategy = WeeklyMomentum(MomentumParams(
                    top_n=n, rebalance_weeks=k, lookback_weeks=args.lookback))
                _, returns = evaluate(study, market, strategy, schedule, label, costs, policy,
                                      note=f"grid N={n} K={k} {context}", capital=args.capital)
                splits = list(purged_splits(len(returns), holding_bars=k, blocks=10,
                                            embargo_bars=9))
                oos = block_sharpes(returns, splits) * np.sqrt(52)
                p10[i, j] = np.percentile(oos, 10)
                median[i, j] = np.median(oos)
                series.append(returns)
                print(f"  N={n:<2} K={k:<2} OOS Sharpe P10 {p10[i, j]:+.2f}  "
                      f"median {median[i, j]:+.2f}", flush=True)

    plateau = np.full_like(p10, np.nan)
    for i in range(len(tops)):
        for j in range(len(rebalances)):
            plateau[i, j] = np.min(p10[max(0, i - 1):i + 2, max(0, j - 1):j + 2])
    ci, cj = np.unravel_index(np.nanargmax(plateau), plateau.shape)
    pi, pj = np.unravel_index(np.nanargmax(p10), p10.shape)

    print("\nOOS Sharpe P10 (annualised), rows N, columns K; [x] plateau centre, (x) peak")
    print("      " + "".join(f"K={k:<7}" for k in rebalances))
    for i, n in enumerate(tops):
        cells = []
        for j in range(len(rebalances)):
            text = f"{p10[i, j]:+.2f}"
            if (i, j) == (ci, cj):
                text = f"[{text}]"
            elif (i, j) == (pi, pj):
                text = f"({text})"
            cells.append(f"{text:<9}")
        print(f"N={n:<3} " + "".join(cells))

    matrix = np.column_stack(series)
    overfitting = probability_of_backtest_overfitting(matrix, blocks=10)
    print(f"\nplateau centre  N={tops[ci]} K={rebalances[cj]}: P10 {p10[ci, cj]:+.2f}, "
          f"worst neighbour {plateau[ci, cj]:+.2f}, median {median[ci, cj]:+.2f}")
    if (pi, pj) != (ci, cj):
        print(f"peak            N={tops[pi]} K={rebalances[pj]}: P10 {p10[pi, pj]:+.2f}, but "
              f"its worst neighbour is {plateau[pi, pj]:+.2f}: not a plateau")
    print(f"grid PBO        {overfitting.pbo:.0%} over {overfitting.combinations} partitions "
          f"(chance the in-sample best of this grid is below median out of sample)")
    universe = f" --universe {args.universe}" if args.universe else ""
    print(f"\nnext: validate the centre, with every grid trial counted:\n"
          f"  ql funnel{universe} --top {tops[ci]} --rebalance-weeks {rebalances[cj]}"
          f" --start {args.start}")
    return 0


def _values(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return default
    return sorted({int(v) for v in text.split(",") if v.strip()})


def _around(centre: int, offsets, low: int = 1, high: int | None = None) -> list[int]:
    values = [centre + d for d in offsets]
    return [v for v in values if v >= low and (high is None or v <= high)]


def _sharpe(returns: np.ndarray) -> float:
    deviation = returns.std(ddof=1)
    return float(returns.mean() / deviation) if deviation > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
