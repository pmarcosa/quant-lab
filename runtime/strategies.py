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
from typing import Any

from contracts.errors import ContractViolation
from contracts.strategy import Strategy
from runtime.research import precompute_indicators
from runtime.wiring import Market
from strategies.momentum import MomentumParams, WeeklyMomentum

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


#: Every deployable strategy. A frozen table: registration happens here, in
#: code review, not at run time.
STRATEGIES: Mapping[str, Factory] = {
    "weekly-momentum": _weekly_momentum,
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
