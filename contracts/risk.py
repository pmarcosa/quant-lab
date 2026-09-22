"""The risk supervisor's vocabulary, and the one rule it must never break.

Risk sits above the strategy, not beside it. It sees what the strategy decided
and may veto or shrink it, but it has no opinion about what is attractive — that
separation is what keeps a risk limit from quietly becoming a signal.

**Risk may only reduce exposure.** Stated per instrument, on signed positions,
because a book that can be short makes "buy" and "sell" useless for this: a buy
is how a short is closed, and a sell is how one is opened.

Let ``h`` be the position held, ``p`` the position the strategy's orders would
leave, and ``a`` the position the approved orders would leave. Then ``a`` must
lie between zero and ``p``, inclusive. Equivalently:

- risk may trim anything that adds exposure, down to nothing,
- it may add orders that move a position towards zero,
- it may never leave a position larger than the strategy asked for, never on
  the other side of zero from where the strategy asked for it, and **never
  further from zero than the strategy's exit would have left it**.

That last clause is the one that looks like an exception and is not. An exit
reduced is exposure kept on, which is an increase dressed as a limit. A risk
layer that can water down an exit can talk itself into holding a losing
position, which is the failure it exists to prevent. For a long-only book the
rule reduces to the familiar one: never add or enlarge a buy, never shrink or
drop a sell.

The invariant is checked in :class:`RiskReview`, not left to each rule, so a new
rule cannot violate it by being written carelessly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from contracts.errors import ContractViolation
from contracts.execution import OrderIntent
from contracts.identifiers import InstrumentId


class Severity(str, Enum):
    """How much authority a finding carries.

    ``WARN`` is deliberately distinct from ``LIMIT``. Some conditions — a
    correlated cluster growing past a quarter of the book — are worth saying out
    loud without blocking a trade, and collapsing the two would mean either
    losing the warning or acquiring a limit nobody agreed to.
    """

    WARN = "warn"
    LIMIT = "limit"


@dataclass(frozen=True, slots=True)
class RiskFinding:
    """Something a rule noticed, whether or not it changed anything."""

    rule: str
    severity: Severity
    message: str
    instrument: InstrumentId | None = None

    def line(self) -> str:
        where = f" [{self.instrument}]" if self.instrument else ""
        return f"{self.severity.value.upper():<6}{self.rule}{where}: {self.message}"


@dataclass(frozen=True, slots=True)
class RiskReview:
    """A decision after the supervisor has seen it.

    Attributes:
        proposed: What the strategy asked for.
        approved: What may actually be sent.
        held: Signed position per instrument before any of these orders. It is
            required, not defaulted: without it a sell from a long cannot be
            told apart from a sell that opens a short, and the invariant would
            check the wrong thing.
        findings: What the rules noticed.
    """

    proposed: tuple[OrderIntent, ...]
    approved: tuple[OrderIntent, ...]
    held: Mapping[InstrumentId, float]
    findings: tuple[RiskFinding, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        proposed = _net(self.proposed)
        approved = _net(self.approved)
        for instrument in set(proposed) | set(approved):
            held = float(self.held.get(instrument, 0.0))
            asked = held + proposed.get(instrument, 0.0)
            kept = held + approved.get(instrument, 0.0)
            if not _between_zero_and(kept, asked):
                raise ContractViolation(
                    f"risk would leave {instrument} at {kept:g} where the strategy asked "
                    f"for {asked:g} (held {held:g}). The supervisor may only reduce exposure: "
                    f"the result must lie between zero and what was asked, so an exit can "
                    f"never be watered down and an entry never enlarged."
                )

    @property
    def changed(self) -> bool:
        return self.approved != self.proposed

    @property
    def warnings(self) -> tuple[RiskFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARN)

    @property
    def limits_applied(self) -> tuple[RiskFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.LIMIT)


def _net(intents: Sequence[OrderIntent]) -> dict[InstrumentId, float]:
    """Signed quantity per instrument across a set of orders."""
    totals: dict[InstrumentId, float] = {}
    for intent in intents:
        totals[intent.instrument] = (
            totals.get(intent.instrument, 0.0) + intent.side.sign * intent.quantity
        )
    return totals


def _between_zero_and(value: float, bound: float, eps: float = 1e-9) -> bool:
    if abs(bound) <= eps:
        return abs(value) <= eps
    ratio = value / bound
    return -eps <= ratio <= 1.0 + eps


@dataclass(frozen=True, slots=True)
class PositionRisk:
    """What a rule needs to know about one open position."""

    instrument: InstrumentId
    quantity: float
    average_cost: float
    mark: float
    weight: float
    anchor: float | None = None
    opened_at: str = ""


@runtime_checkable
class RiskRule(Protocol):
    """One check. Rules compose; each sees the output of the last.

    Defence in depth: a rule is not told whether another rule already caught
    something, because a rule that relies on its neighbours is a rule that fails
    silently when the neighbour is removed.
    """

    @property
    def name(self) -> str:
        ...

    def apply(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
        marks: Mapping[InstrumentId, float],
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        """Return the intents this rule permits, plus what it noticed.

        ``positions`` are the signed holdings; ``marks`` price every instrument
        an intent touches, held or not, so a limit can value a new position
        instead of waving it through for want of a price.
        """
        ...
