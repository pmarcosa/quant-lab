"""Running a strategy many times, with every run recorded.

The composition root for research. Everything here writes to the ledger, because
the whole point of the funnel is that the trial count is honest — and a trial
count is only honest if it was written down as the trials happened.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from contracts.identifiers import InstrumentId, PortfolioId, RunId, TenantId
from contracts.targets import TargetIntent
from contracts.temporal import Filtration
from engine.accounting import Book
from engine.decide import SizingPolicy
from engine.run import RunResult, run_backtest
from execution.simulated import CostModel, SimulatedBroker
from runtime.wiring import Market
from strategies.momentum import MomentumParams, WeeklyMomentum, indicators
from validation.ledger import Study

STARTING_CAPITAL = 100_000.0


def precompute_indicators(
    market: Market, params: MomentumParams
) -> dict[InstrumentId, pd.DataFrame]:
    """Indicator frames for the whole run, one pass per instrument.

    A research study is dozens or thousands of backtests over one dataset, and
    recomputing every instrument's indicators at every decision time dominates
    the cost: 35,000 passes over a 900-bar series for a single 17-year run.

    The frames cover all of history. That is safe because every indicator at a
    bar uses only that bar and earlier, so slicing at a decision time gives the
    same values as computing up to it -- and the strategy only ever slices. The
    equivalence is tested against the uncached path on the real data rather than
    argued for here.
    """
    frames: dict[InstrumentId, pd.DataFrame] = {}
    horizon = market.schedule[-1]
    view = market.filtration_at(horizon)
    for instrument in market.universe.survivors_only():
        fields = {
            field: view.history(instrument, field, 1_000_000)
            for field in ("high", "low", "close")
        }
        if any(series.empty for series in fields.values()):
            continue
        frames[instrument] = indicators(pd.DataFrame(fields), params)
    return frames


@dataclass(frozen=True, slots=True)
class RandomSelection:
    """The control: the same machinery, picking at random.

    Shares the point-in-time universe, the dates, the position count, the sizing
    bounds, the costs and the rotation cadence with the strategy under test. The
    only difference is how the names are chosen. Anything else that differed
    would make the comparison a comparison of two experiments rather than a
    measure of the signal.
    """

    params: MomentumParams
    seed: int

    @property
    def version(self):
        from contracts.identifiers import StrategyVersion

        return StrategyVersion.of("random-control", {"seed": self.seed, "top_n": self.params.top_n})

    @property
    def filtration_spec(self):
        return WeeklyMomentum(self.params).filtration_spec

    def universe(self, moment: datetime) -> Sequence[InstrumentId]:
        return ()

    def target(
        self, filtration: Filtration, held: Mapping[InstrumentId, float]
    ) -> TargetIntent:
        momentum = WeeklyMomentum(self.params)
        if not momentum.rotates_at(filtration.decision_time):
            return TargetIntent(
                weights=dict(held),
                horizon_bars=self.params.rebalance_weeks,
                as_of=filtration.decision_time,
            )
        available = filtration.universe(min_bars=self.params.min_history_weeks)
        if not available:
            return TargetIntent(weights={}, horizon_bars=1, as_of=filtration.decision_time)

        # Seeded on the decision time as well as the run, so the control is
        # reproducible and two controls do not accidentally pick alike.
        rng = np.random.default_rng(
            (self.seed, int(filtration.decision_time.timestamp()))
        )
        count = min(self.params.top_n, len(available))
        chosen = rng.choice(len(available), size=count, replace=False)
        weight = 1.0 / count
        return TargetIntent(
            weights={available[int(i)]: weight for i in chosen},
            horizon_bars=self.params.rebalance_weeks,
            as_of=filtration.decision_time,
        )

    def state(self) -> Mapping[str, object]:
        return {"name": "random-control", "seed": self.seed}


def run_once(
    market: Market,
    strategy,
    schedule: Sequence[datetime],
    run: RunId,
    costs: CostModel,
    policy: SizingPolicy,
) -> RunResult:
    """One backtest. The only way research runs anything."""
    portfolio = PortfolioId(TenantId("user"), "research")
    broker = SimulatedBroker(costs=costs)
    return run_backtest(
        run=run,
        opening=Book.opening(portfolio, STARTING_CAPITAL, schedule[0]),
        strategy=strategy,
        schedule=list(schedule),
        filtration_at=market.filtration_at,
        marks_at=market.window.marks_at,
        execution_at=market.window.opens_at,
        broker=broker,
        tradable_at=market.window.fresh_at,
        policy=policy,
    )


def periodic_returns(result: RunResult) -> np.ndarray:
    """The equity curve as a return series."""
    values = np.array([equity for _, equity in result.equity_curve()], dtype=float)
    if values.size < 2:
        return np.array([])
    return values[1:] / values[:-1] - 1.0


def evaluate(
    study: Study,
    market: Market,
    strategy,
    schedule: Sequence[datetime],
    label: str,
    costs: CostModel,
    policy: SizingPolicy,
    note: str = "",
) -> tuple[RunResult | None, np.ndarray]:
    """Run a backtest and record it. There is no variant that skips the record.

    An evaluation already in the ledger is reused rather than repeated: the row
    exists, so the trial is already counted, and re-running it would both waste
    the time and double-count the trial. The result object is ``None`` in that
    case — the returns are what research consumes.
    """
    already = study.existing(strategy.version, window=label, note=note)
    if already is not None:
        return None, np.asarray(already.returns, dtype=float)

    result = run_once(
        market, strategy, schedule, RunId(f"r{abs(hash(label)) % 10**9}"), costs, policy
    )
    returns = periodic_returns(result)
    deviation = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    study.evaluate(
        strategy.version,
        window=label,
        metrics={
            "sharpe": float(returns.mean() / deviation) if deviation > 0 else 0.0,
            "final_equity": result.final_equity,
            "periods": float(returns.size),
        },
        returns=returns,
        note=note,
    )
    return result, returns
