"""Weekly momentum with a sector gate: pick the sectors in trend, then the stocks.

The weekly momentum rules (:mod:`strategies.momentum`) choose stocks from the
whole universe on their own trailing returns. Much of a stock's momentum is its
industry's (Moskowitz and Grinblatt, "Do Industries Explain Momentum?", 1999),
so this strategy asks the sector first: a stock may *enter* only when its sector
is trending. Everything else -- ranking, the pace and trend filters, the exits,
the freeze limit, the sizing -- is the weekly momentum strategy's, unchanged.

The sector is read from one reference instrument per sector, a sector ETF, given
in ``sectors`` as ``(stock, sector instrument)`` pairs. Reference instruments are
read and never traded: they are left out of the ranking unless held. A sector is
open when:

* ``sector_trend``: its instrument passes the same entry filters a stock must
  pass -- a positive trailing return, above a rising average, pace not fading;
* ``sector_top_k`` (when not 0): it is among the ``k`` sectors with the best
  trailing return, and that return is positive.

Both may be set; a sector must then meet both. A sector whose instrument has no
history yet at the decision time (an ETF that had not listed) cannot be judged,
so it is left open: a gate that closed on missing data would shut a sector for
the years before its fund existed. A stock with no sector in the map is not
gated either.

The gate acts on entries only. A held stock whose sector closes is judged by its
exits and the freeze limit exactly as before: a position is judged by its own
rules, not by whether it would be bought again today.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId, StrategyVersion
from contracts.temporal import Filtration
from strategies.momentum import FULL_HISTORY, MomentumParams, WeeklyMomentum, entry_ok

STRATEGY_NAME = "sector-momentum"


@dataclass(frozen=True, slots=True)
class SectorMomentumParams(MomentumParams):
    """The weekly momentum parameters, and the sector gate.

    Attributes:
        sectors: ``(stock, sector instrument)`` pairs. The sector instruments are
            reference series: read, never traded.
        sector_trend: A sector is open only if its instrument passes the entry
            filters.
        sector_top_k: A sector is open only if it is among the ``k`` best by
            trailing return, and that return is positive. 0: no such limit.
    """

    sectors: tuple[tuple[str, str], ...] = ()
    sector_trend: bool = True
    sector_top_k: int = 0

    def __post_init__(self) -> None:
        MomentumParams.__post_init__(self)
        if self.sector_top_k < 0:
            raise ContractViolation(
                f"sector_top_k must be a number of sectors, or 0 for no limit; "
                f"got {self.sector_top_k}"
            )
        if not self.sector_trend and not self.sector_top_k:
            raise ContractViolation(
                "a sector gate needs sector_trend, sector_top_k or both; without either it "
                "is the weekly momentum strategy (weekly-momentum)"
            )
        stocks = [stock for stock, _ in self.sectors]
        if len(stocks) != len(set(stocks)):
            raise ContractViolation("a stock appears twice in sectors")
        if set(stocks) & {sector for _, sector in self.sectors}:
            raise ContractViolation("a sector instrument is also listed as a stock")

    @property
    def sector_of(self) -> dict[InstrumentId, InstrumentId]:
        return {InstrumentId(s): InstrumentId(e) for s, e in self.sectors}

    @property
    def references(self) -> frozenset[InstrumentId]:
        return frozenset(InstrumentId(e) for _, e in self.sectors)


@dataclass(frozen=True, slots=True)
class SectorMomentum(WeeklyMomentum):
    """:class:`WeeklyMomentum` whose entries are limited to sectors in trend."""

    params: SectorMomentumParams = SectorMomentumParams(sectors=())
    code_version: str = "1.0"

    @property
    def version(self) -> StrategyVersion:
        return StrategyVersion.of(STRATEGY_NAME, asdict(self.params),
                                  code_version=self.code_version)

    def sector_table(self, filtration: Filtration) -> pd.DataFrame:
        """One row per sector instrument with history: its return, filters and gate."""
        params = self.params
        rows: list[dict[str, Any]] = []
        for sector in sorted(params.references, key=str):
            table = self._indicators_at(filtration, sector, FULL_HISTORY)
            if table is None or len(table) < params.min_history_weeks:
                continue
            row = table.iloc[-1]
            if not np.isfinite(row["close"]):
                continue
            rows.append({"sector": sector, "score": float(row["ret_long"]),
                         "trend": bool(entry_ok(row, params))})
        frame = pd.DataFrame(rows, columns=["sector", "score", "trend"])
        if frame.empty:
            return frame.assign(open=pd.Series(dtype=bool))
        frame = frame.sort_values("score", ascending=False, na_position="last")
        is_open = pd.Series(True, index=frame.index)
        if params.sector_trend:
            is_open &= frame["trend"]
        if params.sector_top_k:
            best = frame.index[: params.sector_top_k]
            is_open &= frame.index.isin(best) & (frame["score"] > 0)
        return frame.assign(open=is_open.astype(bool)).reset_index(drop=True)

    def rank(
        self, filtration: Filtration, held: Mapping[InstrumentId, float], *,
        weeks_to_rotation: int = 0,
    ) -> pd.DataFrame:
        """The weekly momentum ranking, with entries closed in sectors not in trend.

        Adds ``sector`` and ``sector_open`` columns. Sector instruments are left
        out unless held (a held one is then judged like any position).
        """
        table = WeeklyMomentum.rank(self, filtration, held,
                                    weeks_to_rotation=weeks_to_rotation)
        if table.empty:
            return table
        params = self.params
        sectors = self.sector_table(filtration)
        judged = {row.sector: bool(row.open) for row in sectors.itertuples()}
        sector_of = params.sector_of
        table = table.loc[
            ~table["instrument"].isin(params.references) | table["held"]
        ].reset_index(drop=True)
        table["sector"] = [str(sector_of[i]) if i in sector_of else None
                           for i in table["instrument"]]
        # Unknown -- no map entry, or a sector instrument with no history yet --
        # is open: the gate closes on evidence, not on its absence.
        table["sector_open"] = [judged.get(sector_of.get(i), True) for i in table["instrument"]]
        table["eligible"] = table["eligible"] & table["sector_open"]
        return table

    def state(self) -> Mapping[str, Any]:
        return {"name": STRATEGY_NAME, "code_version": self.code_version, **asdict(self.params)}
