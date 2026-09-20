"""What a strategy emits: intended weights, never quantities.

A strategy says "31% of the book in this instrument". It does not say "12
shares", and it has no way to, because it is never told the account's equity.
Translating a weight into an integer number of shares needs the capital, the
price, the minimum lot and the margin rules, and all four live in the engine.

That separation is what lets one backtest engine, one risk layer and one
execution path serve strategy families as different as a single-asset exposure
allocator and a cross-sectional rotation: the allocator is the dimension-1 case
of the same object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from contracts.temporal import utc

#: A weight this large almost certainly means someone returned currency or a
#: share count. Weights are fractions of equity, so anything past this is a bug
#: worth failing on rather than sizing on.
MAX_PLAUSIBLE_WEIGHT = 10.0


@dataclass(frozen=True, slots=True)
class TargetIntent:
    """A strategy's intended book at one instant.

    Attributes:
        weights: Instrument to fraction of equity. Positive is long. Instruments
            absent from the mapping are intended flat. An empty mapping is a
            position: it means cash.
        horizon_bars: How long the strategy expects to hold. The engine uses it to
            choose a friction model and the validation layer to size purge and
            embargo windows, so an honest number matters.
        confidence: How strongly the strategy believes this, in [0, 1]. A scalar
            applies to the whole book. Used for sizing under uncertainty and for
            checking, later, whether high-confidence calls actually did better.
        as_of: The decision time this intent belongs to.
    """

    weights: Mapping[InstrumentId, float]
    horizon_bars: int
    as_of: datetime
    confidence: float = 1.0
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", utc(self.as_of))
        object.__setattr__(self, "weights", dict(self.weights))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

        if self.horizon_bars < 1:
            raise ContractViolation(f"horizon_bars must be at least 1; got {self.horizon_bars}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractViolation(f"confidence must be in [0, 1]; got {self.confidence}")

        for instrument, weight in self.weights.items():
            if not isinstance(instrument, InstrumentId):
                raise ContractViolation(
                    f"weights must be keyed by InstrumentId; got {type(instrument).__name__}"
                )
            if not math.isfinite(weight):
                raise ContractViolation(f"weight for {instrument} is not finite: {weight!r}")
            if abs(weight) > MAX_PLAUSIBLE_WEIGHT:
                raise ContractViolation(
                    f"weight {weight} for {instrument} exceeds {MAX_PLAUSIBLE_WEIGHT}. "
                    f"Weights are fractions of equity; this looks like currency or a share count."
                )

    @property
    def gross(self) -> float:
        """Gross exposure: the sum of absolute weights."""
        return sum(abs(w) for w in self.weights.values())

    @property
    def net(self) -> float:
        """Net exposure: the signed sum of weights."""
        return sum(self.weights.values())

    @property
    def is_cash(self) -> bool:
        """Whether the intent is to hold nothing."""
        return not self.weights

    def instruments(self) -> tuple[InstrumentId, ...]:
        """The instruments carrying a non-zero weight, in a stable order."""
        return tuple(sorted((i for i, w in self.weights.items() if w != 0.0), key=str))
