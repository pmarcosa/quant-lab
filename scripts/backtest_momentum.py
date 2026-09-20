#!/usr/bin/env python3
"""Run the weekly momentum strategy over the stored history.

    python scripts/backtest_momentum.py --freq weekly

The script is thin on purpose: it parses arguments and prints. Every decision it
appears to make is made by the engine, so nothing here can make a backtest
disagree with a live proposal.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.identifiers import PortfolioId, RunId, TenantId  # noqa: E402
from contracts.temporal import BarInterval  # noqa: E402
from engine.accounting import Book  # noqa: E402
from engine.decide import SizingPolicy  # noqa: E402
from engine.run import run_backtest  # noqa: E402
from execution.simulated import CostModel, SimulatedBroker  # noqa: E402
from runtime.wiring import load_market  # noqa: E402
from strategies.momentum import MomentumParams, WeeklyMomentum  # noqa: E402
from validation.metrics import summarise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "var" / "store"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", choices=("weekly", "daily"), default="weekly")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=13)
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Commission per side")
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--cash-buffer", type=float, default=0.01)
    parser.add_argument("--min-trade", type=float, default=0.005)
    parser.add_argument("--start", default=None, help="Earliest decision date, ISO")
    parser.add_argument(
        "--rebalance-weeks", type=int, default=1,
        help="Weeks between rotations. The book is marked every week regardless.",
    )
    args = parser.parse_args(argv)

    interval = BarInterval.WEEK if args.freq == "weekly" else BarInterval.DAY
    start = (
        datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    )
    market = load_market(STORE, interval=interval, start=start)

    # Every week is a decision; only some are rotations. Marking the book only
    # on rotation weeks hides whatever happened between them.
    schedule = list(market.schedule)
    strategy = WeeklyMomentum(
        MomentumParams(
            top_n=args.top,
            lookback_weeks=args.lookback,
            rebalance_weeks=args.rebalance_weeks,
        )
    )
    portfolio = PortfolioId(TenantId("user"), "backtest")
    broker = SimulatedBroker(
        costs=CostModel(commission_bps=args.cost_bps, slippage_bps=args.slippage_bps)
    )

    result = run_backtest(
        run=RunId(f"bt-{args.freq}-{args.rebalance_weeks}"),
        opening=Book.opening(portfolio, args.capital, schedule[0]),
        strategy=strategy,
        schedule=schedule,
        filtration_at=market.filtration_at,
        marks_at=market.window.marks_at,
        execution_at=market.window.opens_at,
        tradable_at=market.window.fresh_at,
        broker=broker,
        policy=SizingPolicy(
            cash_buffer=args.cash_buffer, min_trade_fraction=args.min_trade
        ),
    )

    curve = result.equity_curve()
    stats = summarise(curve, periods_per_year=market.periods_per_year)

    print(f"strategy   {strategy.version.strategy} {strategy.version.params_hash}")
    print(f"decisions  {len(result.steps):,} from {schedule[0].date()} to {schedule[-1].date()}")
    rotations = sum(1 for s in result.steps if s.decision.target.diagnostics.get("rotated"))
    print(f"rotations  {rotations:,} of {len(result.steps):,} weekly marks")
    print(f"orders     {sum(len(s.intents) for s in result.steps):,}")
    print(f"fills      {len(result.all_fills()):,}")
    print(f"unfilled   {broker.unfilled:,} (no print at the execution bar)")
    print()
    print(f"{'years':<14}{stats.years:>12.2f}")
    print(f"{'CAGR':<14}{stats.cagr:>11.1%}")
    print(f"{'volatility':<14}{stats.volatility:>11.1%}")
    print(f"{'Sharpe':<14}{stats.sharpe:>12.2f}")
    print(f"{'  geometric':<14}{stats.sharpe_geometric:>12.2f}")
    print(f"{'Sortino':<14}{stats.sortino:>12.2f}")
    print(f"{'max drawdown':<14}{stats.max_drawdown:>11.1%}")
    print(f"{'final equity':<14}{stats.final_equity:>12,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
