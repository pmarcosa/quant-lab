#!/usr/bin/env python3
"""Price the execution conventions, on the real data, one convention at a time.

    python scripts/reconcile_conventions.py

The previous system (``regime-trader``) fills a weekly rotation at the close of
the very bar the decision was made on. quant-lab fills at the next bar's open,
because a decision taken on a close cannot transact at that close. Both
conventions are run through the *same* engine here, so the difference between
them is the convention and nothing else — not a different universe, a different
cost model or a different sizing rule.

This asserts the direction of the effect, so it is a regression rather than an
anecdote. The magnitude is printed, not asserted: it is a property of this
dataset, and pinning it would make the script fail every time the cache grows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
from validation.metrics import Performance, summarise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "var" / "store"
START = datetime(2009, 2, 24, tzinfo=timezone.utc)
ROTATE_WEEKS = 4


@dataclass(frozen=True, slots=True)
class Variant:
    label: str
    same_bar_fill: bool
    commission_bps: float
    slippage_bps: float
    cash_buffer: float
    no_trade_band: float


VARIANTS = (
    Variant("regime-trader convention (same-bar close)", True, 20.0, 0.0, 0.0, 0.0),
    Variant("+ next-open fill (the honest one)", False, 20.0, 0.0, 0.0, 0.0),
    Variant("+ 1% cash buffer", False, 20.0, 0.0, 0.01, 0.0),
    Variant("+ 0.5% no-trade band", False, 20.0, 0.0, 0.01, 0.005),
    Variant("+ 10bps slippage (quant-lab default)", False, 10.0, 10.0, 0.01, 0.005),
)


def main() -> int:
    if not (STORE / "universe_weekly.csv").exists():
        print("No store. Run: python scripts/ingest_ibkr_cache.py", file=sys.stderr)
        return 1

    market = load_market(STORE, interval=BarInterval.WEEK, start=START)
    # Every week is a decision and a mark; the strategy rotates every fourth.
    # Subsampling the schedule instead would push each fill four weeks past its
    # decision, which is a different (and much worse) convention than either of
    # the two being compared here.
    schedule = list(market.schedule)
    position_of = {moment: i for i, moment in enumerate(schedule)}

    def decision_bar_close(moment: datetime):
        """The lookahead, reproduced: an order at week k fills at week k's close."""
        index = position_of[moment]
        return market.window.marks_at(schedule[index - 1] if index else moment)

    def measure(variant: Variant) -> Performance:
        broker = SimulatedBroker(
            costs=CostModel(
                commission_bps=variant.commission_bps, slippage_bps=variant.slippage_bps
            )
        )
        result = run_backtest(
            run=RunId("reconcile"),
            opening=Book.opening(
                PortfolioId(TenantId("user"), "reconcile"), 100_000.0, schedule[0]
            ),
            strategy=WeeklyMomentum(MomentumParams(rebalance_weeks=ROTATE_WEEKS)),
            schedule=schedule,
            filtration_at=market.filtration_at,
            marks_at=market.window.marks_at,
            execution_at=(
                decision_bar_close if variant.same_bar_fill else market.window.opens_at
            ),
            broker=broker,
            tradable_at=market.window.fresh_at,
            policy=SizingPolicy(
                cash_buffer=variant.cash_buffer, min_trade_fraction=variant.no_trade_band
            ),
        )
        return summarise(result.equity_curve(), periods_per_year=52)

    print(
        f"{len(schedule)} weekly marks, rotating every {ROTATE_WEEKS}, "
        f"{schedule[0].date()} to {schedule[-1].date()}"
    )
    print(f"{'':<44}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}{'final':>13}")
    print("-" * 82)
    results = {}
    for variant in VARIANTS:
        stats = measure(variant)
        results[variant.label] = stats
        print(
            f"{variant.label:<44}{stats.cagr:>8.1%}{stats.sharpe:>8.2f}"
            f"{stats.max_drawdown:>9.1%}{stats.final_equity:>13,.0f}"
        )

    lookahead = results[VARIANTS[0].label]
    honest = results[VARIANTS[1].label]

    # The direction is the finding; the size is data.
    assert lookahead.cagr > honest.cagr, "the lookahead should flatter returns"
    assert lookahead.sharpe > honest.sharpe, "and flatter risk-adjusted returns"
    assert lookahead.max_drawdown > honest.max_drawdown, "and understate drawdown"

    print()
    print("Filling at the decision bar's own close is worth:")
    print(f"  {lookahead.cagr - honest.cagr:+.1%} of CAGR")
    print(f"  {lookahead.sharpe - honest.sharpe:+.2f} of Sharpe")
    print(f"  {lookahead.max_drawdown - honest.max_drawdown:+.1%} of reported drawdown")
    print()
    print("None of it was ever available to trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
