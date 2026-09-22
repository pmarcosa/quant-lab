"""Strategies that exist only to exercise the machinery.

The deployable strategies are long-only and weekly. The system must not assume
either, so the live cycle, the risk layer and monitoring are also driven with
this one: daily bars, one long and one short, chosen by trailing return.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from contracts.identifiers import InstrumentId, StrategyVersion
from contracts.targets import TargetIntent
from contracts.temporal import BarInterval, Filtration, FiltrationSpec


@dataclass(frozen=True, slots=True)
class LongShortParams:
    long_weight: float = 0.4
    short_weight: float = 0.3
    lookback: int = 20


class DailyLongShort:
    """Long the best trailing performer, short the worst, every session."""

    def __init__(self, params: LongShortParams | None = None) -> None:
        self.params = params or LongShortParams()

    @property
    def version(self) -> StrategyVersion:
        return StrategyVersion.of("toy-long-short", asdict(self.params))

    @property
    def filtration_spec(self) -> FiltrationSpec:
        return FiltrationSpec(interval=BarInterval.DAY, observation_lag_bars=0)

    def universe(self, moment: datetime) -> Sequence[InstrumentId]:
        return ()

    def target(self, filtration: Filtration, held: Mapping[InstrumentId, float]) -> TargetIntent:
        p = self.params
        scores = {}
        for instrument in filtration.universe(min_bars=p.lookback + 1):
            closes = filtration.history(instrument, "close", p.lookback + 1)
            if len(closes) > p.lookback:
                scores[instrument] = float(closes.iloc[-1] / closes.iloc[0] - 1.0)
        weights: dict[InstrumentId, float] = {}
        if len(scores) >= 2:
            ranked = sorted(scores, key=lambda i: scores[i])
            weights[ranked[-1]] = p.long_weight
            weights[ranked[0]] = -p.short_weight
        return TargetIntent(
            weights=weights, horizon_bars=1, as_of=filtration.decision_time,
            diagnostics={"rotated": 1.0},
        )

    def state(self) -> Mapping[str, Any]:
        return {"name": "toy-long-short", **asdict(self.params)}
