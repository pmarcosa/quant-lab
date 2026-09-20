"""The risk supervisor's vocabulary, and the one rule it must never break.

Risk sits above the strategy, not beside it. It sees what the strategy decided
and may veto or shrink it, but it has no opinion about what is attractive — that
separation is what keeps a risk limit from quietly becoming a signal.

**Risk may only reduce exposure.** Stated precisely, because the loose version is
ambiguous and the ambiguity is where the damage happens:

- it may drop a buy, or shrink one,
- it may *add* a sell that closes or reduces a position,
- it may never add a buy, never enlarge one, and **never shrink or drop a sell**.

That last clause is the one that looks like an exception and is not. A sell in
this system is an exit: reducing it leaves more exposure on, which is an increase
dressed as a limit. A risk layer that can water down an exit can talk itself into
holding a losing position, which is the failure it exists to prevent.

The invariant is checked in :class:`RiskReview`, not left to each rule, so a new
rule cannot violate it by being written carelessly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from contracts.errors import ContractViolation
from contracts.execution import OrderIntent, Side
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
        approved: What may actually be sent, including any protective orders the
            supervisor added.
        findings: What the rules noticed.
    """

    proposed: tuple[OrderIntent, ...]
    approved: tuple[OrderIntent, ...]
    findings: tuple[RiskFinding, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        proposed_buys = _by_instrument(self.proposed, Side.BUY)
        approved_buys = _by_instrument(self.approved, Side.BUY)
        proposed_sells = _by_instrument(self.proposed, Side.SELL)
        approved_sells = _by_instrument(self.approved, Side.SELL)

        for instrument, quantity in approved_buys.items():
            allowed = proposed_buys.get(instrument, 0.0)
            if quantity > allowed + 1e-9:
                raise ContractViolation(
                    f"risk increased the buy in {instrument} from {allowed} to {quantity}. "
                    f"The supervisor may only reduce exposure."
                )
        for instrument, quantity in proposed_sells.items():
            kept = approved_sells.get(instrument, 0.0)
            if kept < quantity - 1e-9:
                raise ContractViolation(
                    f"risk reduced the sell in {instrument} from {quantity} to {kept}. "
                    f"A sell is an exit; shrinking one leaves exposure on, which is an "
                    f"increase wearing a limit's clothes."
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


def _by_instrument(
    intents: Sequence[OrderIntent], side: Side
) -> dict[InstrumentId, float]:
    totals: dict[InstrumentId, float] = {}
    for intent in intents:
        if intent.side is side:
            totals[intent.instrument] = totals.get(intent.instrument, 0.0) + intent.quantity
    return totals


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
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        """Return the intents this rule permits, plus what it noticed."""
        ...
