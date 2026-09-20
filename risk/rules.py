"""The risk rules. Each one may only reduce exposure; the contract enforces it.

Ordered from cheapest to most consequential, and composed so that each sees the
previous one's output. Defence in depth: no rule assumes another ran.

What is deliberately *not* here, because the user decided against it and a risk
layer that quietly acquires limits nobody agreed to is worse than none:

- no per-position cap (the strategy's weight bounds already do that job),
- no sector cap,
- the correlated-cluster check **warns and does not block**.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from contracts.errors import ContractViolation
from contracts.execution import (
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
    client_order_id,
)
from contracts.identifiers import InstrumentId
from contracts.risk import PositionRisk, RiskFinding, RiskReview, Severity


@dataclass(frozen=True, slots=True)
class GrossExposureLimit:
    """Cap total exposure as a fraction of equity, by trimming buys.

    Trims rather than rejects: a decision that would take the book to 105% of
    equity is not wrong, it is one order too large, and refusing the whole
    rotation over it would be a bigger intervention than the problem.
    """

    maximum: float = 1.0

    @property
    def name(self) -> str:
        return "gross exposure"

    def apply(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        if equity <= 0:
            raise ContractViolation(f"cannot measure exposure against equity of {equity}")
        held = sum(abs(p.weight) for p in positions.values())
        findings: list[RiskFinding] = []
        approved: list[OrderIntent] = []
        room = max(self.maximum - held, 0.0) * equity

        for intent in intents:
            if intent.side is Side.SELL:
                approved.append(intent)  # exits are never trimmed
                continue
            mark = positions[intent.instrument].mark if intent.instrument in positions else None
            if mark is None or mark <= 0:
                approved.append(intent)
                continue
            value = intent.quantity * mark
            if value <= room:
                approved.append(intent)
                room -= value
                continue
            # The epsilon absorbs binary representation error, not the limit:
            # 1.0 - 0.90 is 0.09999999999999998, so ten whole shares of room
            # measures as 9.999999999999998 and would be shaved to nine.
            allowed = int(room / mark + 1e-9)
            findings.append(
                RiskFinding(
                    rule=self.name,
                    severity=Severity.LIMIT,
                    instrument=intent.instrument,
                    message=(
                        f"trimmed {intent.quantity:.0f} to {allowed:.0f} shares; "
                        f"gross would have passed {self.maximum:.0%} of equity"
                    ),
                )
            )
            if allowed >= 1:
                approved.append(replace(intent, quantity=float(allowed)))
                room = 0.0
        return tuple(approved), tuple(findings)


@dataclass(frozen=True, slots=True)
class CorrelatedClusterWarning:
    """Warn when correlated names together pass a share of the book.

    Warns and does not block, by explicit decision. The reason is that the
    clusters are defined by hand and correlations are unstable — the project
    already measured XOM's correlation with SPY moving from -0.03 to -0.62
    between two halves of one year. Blocking on an unstable measurement would
    stop trades for a reason that will not hold next quarter; saying it out loud
    costs nothing and is read by someone who can judge.
    """

    clusters: Mapping[str, frozenset[InstrumentId]]
    threshold: float = 0.25

    @property
    def name(self) -> str:
        return "correlated cluster"

    def apply(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        findings: list[RiskFinding] = []
        for label, members in self.clusters.items():
            weight = sum(p.weight for i, p in positions.items() if i in members)
            if weight > self.threshold:
                findings.append(
                    RiskFinding(
                        rule=self.name,
                        severity=Severity.WARN,
                        message=(
                            f"{label} is {weight:.0%} of the book, past {self.threshold:.0%}. "
                            f"Not blocked."
                        ),
                    )
                )
        return tuple(intents), tuple(findings)


@dataclass(frozen=True, slots=True)
class ProtectiveStop:
    """A resting stop a fixed distance below the anchor, replaced each rotation.

    Two decisions that are easy to confuse, both settled by measurement over
    seventeen years:

    **Distance: a fixed 12%,** the same for a calm instrument and a volatile one.
    Volatility enters the system through position *size* (inverse-ATR weighting),
    not through stop distance.

    **Anchor: the rotation price, never the average cost.** This is the detail the
    project flags as impossible to get wrong safely. Measured on the live book: a
    stop set from AAPL's average cost sat 47% below the market and protected
    nothing, while one set from DIS's average cost sat above the market and would
    have sold immediately. Average cost is information about the past of the
    holder, not about the future of the instrument.

    The stop is a **broker-resting order**, not a process check. If the machine
    running this is off, the position is still protected.
    """

    distance: float = 0.12

    def __post_init__(self) -> None:
        if not 0.0 < self.distance < 1.0:
            raise ContractViolation(
                f"stop distance must be a fraction in (0, 1); got {self.distance}"
            )

    @property
    def name(self) -> str:
        return "protective stop"

    def level(self, anchor: float) -> float:
        """Where the stop sits for a position anchored at ``anchor``."""
        return anchor * (1.0 - self.distance)

    def orders_for(
        self,
        positions: Mapping[InstrumentId, PositionRisk],
        run,
        portfolio,
        strategy_version,
        moment,
        constraints_for,
    ) -> tuple[OrderIntent, ...]:
        """One resting stop per open position, anchored at this rotation."""
        orders: list[OrderIntent] = []
        for instrument, position in sorted(positions.items(), key=lambda kv: str(kv[0])):
            anchor = position.anchor if position.anchor is not None else position.mark
            if anchor is None or anchor <= 0 or position.quantity <= 0:
                continue
            rules = constraints_for(instrument)
            quantity = rules.round_quantity(position.quantity)
            if quantity <= 0:
                continue
            level = rules.round_price(self.level(anchor))
            orders.append(
                OrderIntent(
                    client_order_id=client_order_id(
                        run, portfolio, instrument, moment, Side.SELL, quantity
                    )
                    + "-stop",
                    run=run,
                    portfolio=portfolio,
                    instrument=instrument,
                    strategy_version=strategy_version,
                    side=Side.SELL,
                    quantity=quantity,
                    order_type=OrderType.STOP,
                    decision_time=moment,
                    stop_price=level,
                    time_in_force=TimeInForce.GTC,
                    reason="protective stop",
                )
            )
        return tuple(orders)


@dataclass(frozen=True, slots=True)
class RiskSupervisor:
    """Runs the rules in order and reports one review.

    Fail-closed: a rule that raises stops the decision rather than being skipped.
    A risk layer that swallows its own errors is decoration.
    """

    rules: tuple = ()
    stop: ProtectiveStop | None = None

    def protective_orders(
        self, positions, run, portfolio, decision, constraints_for
    ) -> tuple[OrderIntent, ...]:
        """Resting stops for the current book, or none if no stop is configured."""
        if self.stop is None:
            return ()
        return self.stop.orders_for(
            positions,
            run=run,
            portfolio=portfolio,
            strategy_version=decision.strategy_version,
            moment=decision.decision_time,
            constraints_for=constraints_for,
        )

    def review(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
    ) -> RiskReview:
        current = tuple(intents)
        findings: list[RiskFinding] = []
        for rule in self.rules:
            current, produced = rule.apply(current, positions, equity)
            findings.extend(produced)
        return RiskReview(
            proposed=tuple(intents), approved=current, findings=tuple(findings)
        )
