"""Strategies by name: what a live configuration can deploy.

A live config names a strategy and its parameters; this module turns that into
an instance. Nothing in the live cycle, the monitor or the CLI imports a
strategy class directly any more, so adding a second strategy -- long/short,
daily, anything the ``Strategy`` contract admits -- is one entry here and a
config file, not a change to the machinery.

The bar interval is the strategy's own (``filtration_spec.interval``). The
live cycle, the data refresh and monitoring all ask the strategy rather than
assuming a week.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from contracts.errors import ContractViolation
from contracts.strategy import Strategy
from runtime.research import precompute_indicators
from runtime.wiring import Market
from strategies.momentum import MomentumParams, WeeklyMomentum
from strategies.sector_momentum import SectorMomentum, SectorMomentumParams

#: Builds a strategy from its parameters and, optionally, the market it will
#: run on (for strategies that can precompute indicators over a whole run).
Factory = Callable[[Mapping[str, Any], "Market | None"], Strategy]


def _weekly_momentum(params: Mapping[str, Any], market: Market | None = None) -> Strategy:
    try:
        settings = MomentumParams(**dict(params))
    except TypeError as error:
        raise ContractViolation(f"weekly-momentum: {error}") from error
    precomputed = precompute_indicators(market, settings) if market is not None else None
    return WeeklyMomentum(settings, precomputed=precomputed)


def sector_map(spec: Any) -> tuple[tuple[str, str], ...]:
    """``(stock, sector instrument)`` pairs from a config value.

    A config names a map file -- ``data/universes/<name>.sectors.csv`` or a path,
    with columns ``symbol`` and ``sector`` -- or gives the pairs inline. The pairs,
    not the file name, go into the parameters, so editing the file changes the
    strategy's version as any other parameter would.
    """
    from runtime.wiring import UNIVERSE_DIR

    if isinstance(spec, str):
        path = Path(spec)
        if not path.suffix:
            path = UNIVERSE_DIR / f"{spec}.sectors.csv"
        if not path.exists():
            raise ContractViolation(f"sector-momentum: no sector map at {path}")
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.split("#", 1)[0].strip()
            if not line or line.lower().startswith("symbol"):
                continue
            parts = [part.strip().upper() for part in line.split(",")]
            if len(parts) != 2 or not all(parts):
                raise ContractViolation(f"{path.name}:{number}: expected 'SYMBOL,SECTOR'")
            rows.append((parts[0], parts[1]))
        return tuple(sorted(rows))
    if isinstance(spec, Mapping):
        return tuple(sorted((str(k).upper(), str(v).upper()) for k, v in spec.items()))
    return tuple(sorted((str(a).upper(), str(b).upper()) for a, b in spec or ()))


def _sector_momentum(params: Mapping[str, Any], market: Market | None = None) -> Strategy:
    values = dict(params)
    values["sectors"] = sector_map(values.get("sectors"))
    try:
        settings = SectorMomentumParams(**values)
    except TypeError as error:
        raise ContractViolation(f"sector-momentum: {error}") from error
    precomputed = precompute_indicators(market, settings) if market is not None else None
    return SectorMomentum(settings, precomputed=precomputed)


#: Every deployable strategy. A frozen table: registration happens here, in
#: code review, not at run time.
STRATEGIES: Mapping[str, Factory] = {
    "weekly-momentum": _weekly_momentum,
    "sector-momentum": _sector_momentum,
}


def build_strategy(
    name: str, params: Mapping[str, Any], market: Market | None = None
) -> Strategy:
    """An instance of the named strategy.

    Raises:
        ContractViolation: If the name is unknown or the parameters are not the
            strategy's.
    """
    factory = STRATEGIES.get(name)
    if factory is None:
        raise ContractViolation(
            f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}"
        )
    return factory(params, market)
