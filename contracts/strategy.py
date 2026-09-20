"""The contract every strategy implements.

Four things, and nothing else:

1. the universe it may consider on a date,
2. the book it intends to hold, given a filtration,
3. how it samples time and how late its inputs arrive,
4. its internal state, so a decision can be replayed and explained.

Everything else — sizing to shares, risk limits, accounting, friction, the
falsification funnel — belongs to the framework and is identical for every
strategy. That is what makes the modules shared rather than merely reused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from contracts.identifiers import InstrumentId, StrategyVersion
from contracts.targets import TargetIntent
from contracts.temporal import Filtration, FiltrationSpec


@runtime_checkable
class Strategy(Protocol):
    """A pure function from a filtration to an intended book.

    Implementations must be deterministic: the same filtration and the same
    parameters produce the same intent, always. No wall clock, no reading files
    at decision time, no unseeded randomness. Without that, a backtest cannot be
    reproduced and a live decision cannot be audited.
    """

    @property
    def version(self) -> StrategyVersion:
        """Family plus parameter fingerprint. Parameters are part of identity."""
        ...

    @property
    def filtration_spec(self) -> FiltrationSpec:
        """The bar size and observation lag this strategy needs."""
        ...

    def universe(self, moment: datetime) -> Sequence[InstrumentId]:
        """Instruments this strategy may consider at ``moment``.

        For a single-asset allocator this is one instrument; for a cross-sectional
        rotation it is the point-in-time eligible list. The engine uses it to load
        only what is needed.
        """
        ...

    def target(self, filtration: Filtration) -> TargetIntent:
        """The book the strategy intends to hold, given what is knowable."""
        ...

    def state(self) -> Mapping[str, Any]:
        """Parameters and internal values behind the most recent decision.

        Recorded with every decision so that a rotation can be replayed and
        explained months later without rerunning the model.
        """
        ...
