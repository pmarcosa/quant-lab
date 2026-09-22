"""The risk rules. Each one may only reduce exposure; the contract enforces it.

Ordered from cheapest to most consequential, and composed so that each sees the
previous one's output. Defence in depth: no rule assumes another ran.

Every rule reasons about **signed positions**: a long is positive, a short
negative, and exposure is the distance from zero. An order is split into the
leg that closes existing exposure and the leg that opens new exposure
(``contracts.execution.split_legs``); limits trim only opening legs, and
closing legs always pass. Nothing here assumes the book is long-only.

What is deliberately *not* here, because the user decided against it and a risk
layer that quietly acquires limits nobody agreed to is worse than none:

- no per-position cap (the strategy's weight bounds already do that job),
- no sector cap,
- the correlated-cluster check **warns and does not block**.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from contracts.errors import ContractViolation
from contracts.execution import (
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
    client_order_id,
    split_legs,
)
from contracts.identifiers import InstrumentId
from contracts.risk import PositionRisk, RiskFinding, RiskReview, Severity

# -- shared arithmetic ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Change:
    """What the proposed orders would do to one instrument's position."""

    instrument: InstrumentId
    held: float
    final: float
    base: float  # the position after only the exposure-reducing part
    mark: float

    @property
    def increase(self) -> float:
        """Shares of new exposure: how far past ``base`` the orders take it."""
        return abs(self.final) - abs(self.base)

    @property
    def direction(self) -> float:
        return 1.0 if self.final > 0 else -1.0


def _changes(
    intents: Sequence[OrderIntent],
    positions: Mapping[InstrumentId, PositionRisk],
    marks: Mapping[InstrumentId, float],
) -> dict[InstrumentId, _Change]:
    net: dict[InstrumentId, float] = {}
    for intent in intents:
        net[intent.instrument] = net.get(intent.instrument, 0.0) + intent.side.sign * intent.quantity
    changes: dict[InstrumentId, _Change] = {}
    for instrument, delta in net.items():
        held = positions[instrument].quantity if instrument in positions else 0.0
        final = held + delta
        if held != 0 and (final == 0 or (final > 0) == (held > 0)) and abs(final) <= abs(held):
            base = final  # a pure reduction
        elif held != 0 and final != 0 and (final > 0) == (held > 0):
            base = held  # adding on the same side
        else:
            base = 0.0  # opening from flat, or reversing through zero
        changes[instrument] = _Change(instrument, held, final, base, _mark(instrument, positions, marks))
    return changes


def _mark(
    instrument: InstrumentId,
    positions: Mapping[InstrumentId, PositionRisk],
    marks: Mapping[InstrumentId, float],
) -> float:
    price = marks.get(instrument)
    if price is None and instrument in positions:
        price = positions[instrument].mark
    if price is None or not price > 0:
        # Fail closed. Waving an order through because it could not be valued
        # is how a limit stops applying to exactly the names it has not seen.
        raise ContractViolation(f"risk cannot value {instrument}: no usable mark")
    return float(price)


def _resize(intent: OrderIntent, change: _Change, allowed_increase: float) -> OrderIntent | None:
    """The intent with its opening leg cut to ``allowed_increase`` shares."""
    closing = abs(change.held - change.base)
    quantity = closing + max(allowed_increase, 0.0)
    if quantity <= 0:
        return None
    return replace(intent, quantity=float(quantity))


def _trim_to_budget(
    intents: Sequence[OrderIntent],
    changes: Mapping[InstrumentId, _Change],
    room_for,  # (change, shares) -> shares allowed
    rule: str,
    describe,
) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
    counts: dict[InstrumentId, int] = {}
    for intent in intents:
        counts[intent.instrument] = counts.get(intent.instrument, 0) + 1
    approved: list[OrderIntent] = []
    findings: list[RiskFinding] = []
    for intent in intents:
        change = changes[intent.instrument]
        if change.increase <= 1e-12:
            approved.append(intent)  # reductions are never trimmed
            continue
        if counts[intent.instrument] > 1:
            raise ContractViolation(
                f"{intent.instrument} has several orders adding exposure; a limit cannot "
                f"decide which to trim"
            )
        allowed = room_for(change, change.increase)
        if allowed >= change.increase - 1e-9:
            approved.append(intent)
            continue
        allowed = float(math.floor(max(allowed, 0.0) + 1e-9))
        findings.append(RiskFinding(
            rule=rule, severity=Severity.LIMIT, instrument=intent.instrument,
            message=f"new exposure trimmed from {change.increase:g} to {allowed:g} shares; {describe}",
        ))
        resized = _resize(intent, change, allowed)
        if resized is not None:
            approved.append(resized)
    return tuple(approved), tuple(findings)


# -- the rules -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GrossExposureLimit:
    """Cap gross exposure (longs plus shorts, as a fraction of equity).

    Trims rather than rejects: a decision that would take the book to 105% of
    equity is not wrong, it is one order too large, and refusing the whole
    rotation over it would be a bigger intervention than the problem.

    Exposure the same decision releases is counted before exposure it adds, in
    whatever order the orders arrive: a rotation that sells A to buy B is not
    over its limit just because it is fully invested before it starts. And a
    new position is valued at its mark like any other -- an earlier version let
    any name the book did not yet hold through unpriced, so the limit only ever
    applied to additions to existing positions.
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
        marks: Mapping[InstrumentId, float],
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        if equity <= 0:
            raise ContractViolation(f"cannot measure exposure against equity of {equity}")
        changes = _changes(intents, positions, marks)
        base = sum(
            abs(p.quantity) * _mark(i, positions, marks)
            for i, p in positions.items() if i not in changes
        ) + sum(abs(c.base) * c.mark for c in changes.values())
        room = [self.maximum * equity - base]

        def room_for(change: _Change, shares: float) -> float:
            allowed = min(shares, max(room[0], 0.0) / change.mark)
            room[0] -= allowed * change.mark
            return allowed

        return _trim_to_budget(
            intents, changes, room_for, self.name,
            f"gross would have passed {self.maximum:.0%} of equity",
        )


@dataclass(frozen=True, slots=True)
class NetExposureLimit:
    """Keep net exposure (longs minus shorts, over equity) inside a band.

    For a long-only book the band's top is the same limit as gross and the
    bottom never binds. It exists for books that can be short, where "fully
    hedged" and "doubly long" have the same gross.
    """

    minimum: float = -1.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ContractViolation("net exposure band is empty: minimum above maximum")

    @property
    def name(self) -> str:
        return "net exposure"

    def apply(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
        marks: Mapping[InstrumentId, float],
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        if equity <= 0:
            raise ContractViolation(f"cannot measure exposure against equity of {equity}")
        changes = _changes(intents, positions, marks)
        net = [(sum(
            p.quantity * _mark(i, positions, marks)
            for i, p in positions.items() if i not in changes
        ) + sum(c.base * c.mark for c in changes.values())) / equity]

        def room_for(change: _Change, shares: float) -> float:
            per_share = change.direction * change.mark / equity
            limit = self.maximum if per_share > 0 else self.minimum
            headroom = (limit - net[0]) / per_share
            allowed = min(shares, max(headroom, 0.0))
            net[0] += allowed * per_share
            return allowed

        return _trim_to_budget(
            intents, changes, room_for, self.name,
            f"net would have left [{self.minimum:.0%}, {self.maximum:.0%}] of equity",
        )


@dataclass(frozen=True, slots=True)
class ShortSales:
    """Whether new shorts may open, and how many shares can be borrowed.

    Off by default: a strategy that emits a short by accident must fail, not
    silently borrow stock. When allowed, a live session passes what the broker
    reports as borrowable, and an opening short leg is cut to it. A name with no
    availability figure is treated as not borrowable -- the pre-trade check the
    project's expert lists first, and one where guessing "probably fine" is how
    an order meets a forced buy-in.

    ``availability=None`` means no data source at all (a backtest), and only the
    on/off switch applies.
    """

    allowed: bool = False
    availability: Mapping[InstrumentId, float | None] | None = None
    borrow_fees: Mapping[InstrumentId, float | None] = field(default_factory=dict)
    max_borrow_fee: float | None = None

    @property
    def name(self) -> str:
        return "short sales"

    def apply(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
        marks: Mapping[InstrumentId, float],
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        approved: list[OrderIntent] = []
        findings: list[RiskFinding] = []
        held = {i: p.quantity for i, p in positions.items()}
        for intent in intents:
            position = held.get(intent.instrument, 0.0)
            closing, opening = split_legs(position, intent.side, intent.quantity)
            shorting = opening > 0 and intent.side is Side.SELL
            keep = intent.quantity
            if shorting:
                limit, why = self._shortable(intent.instrument, opening)
                if limit < opening:
                    keep = closing + limit
                    findings.append(RiskFinding(
                        rule=self.name, severity=Severity.LIMIT, instrument=intent.instrument,
                        message=f"short of {opening:g} cut to {limit:g}: {why}",
                    ))
            if keep > 0:
                approved.append(intent if keep == intent.quantity else replace(intent, quantity=float(keep)))
                held[intent.instrument] = position + intent.side.sign * keep
        return tuple(approved), tuple(findings)

    def _shortable(self, instrument: InstrumentId, wanted: float) -> tuple[float, str]:
        if not self.allowed:
            return 0.0, "short selling is not enabled for this strategy"
        if self.availability is None:
            return wanted, ""
        available = self.availability.get(instrument)
        if available is None:
            return 0.0, "the broker reports no borrow availability for it"
        fee = self.borrow_fees.get(instrument)
        if self.max_borrow_fee is not None and fee is not None and fee > self.max_borrow_fee:
            return 0.0, f"borrow fee {fee:.2%} is above the {self.max_borrow_fee:.2%} limit"
        return float(math.floor(max(available, 0.0))), "only that many shares can be borrowed"


@dataclass(frozen=True, slots=True)
class ReduceOnly:
    """No new exposure: exits and trims pass, entries do not, on either side.

    For a long that means sells pass and buys do not; for a short, buying to
    cover passes and selling more does not. An order that would reverse a
    position through zero is cut to the part that closes it -- the book goes
    flat, not onto the other side. That is the expert's definition: the target
    may not be larger than the current position, nor of the opposite sign.

    What the live system runs under when monitoring has seen something it cannot
    yet explain. It cannot make anything worse, which is the only property a rule
    that fires on uncertain evidence may have.
    """

    reason: str = "reduce-only"

    @property
    def name(self) -> str:
        return "reduce only"

    def apply(
        self,
        intents: Sequence[OrderIntent],
        positions: Mapping[InstrumentId, PositionRisk],
        equity: float,
        marks: Mapping[InstrumentId, float],
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        held = {i: p.quantity for i, p in positions.items()}
        kept: list[OrderIntent] = []
        findings: list[RiskFinding] = []
        for intent in intents:
            position = held.get(intent.instrument, 0.0)
            closing, opening = split_legs(position, intent.side, intent.quantity)
            if opening > 0:
                findings.append(RiskFinding(
                    rule=self.name, severity=Severity.LIMIT, instrument=intent.instrument,
                    message=f"{intent.side.value} of {opening:g} new shares withheld: {self.reason}",
                ))
            if closing > 0:
                kept.append(intent if opening == 0 else replace(intent, quantity=float(closing)))
                held[intent.instrument] = position + intent.side.sign * closing
        return tuple(kept), tuple(findings)


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
        marks: Mapping[InstrumentId, float],
    ) -> tuple[tuple[OrderIntent, ...], tuple[RiskFinding, ...]]:
        findings: list[RiskFinding] = []
        for label, members in self.clusters.items():
            # Gross, not net: a long and a short in the same cluster are two
            # bets on it, not none.
            weight = sum(abs(p.weight) for i, p in positions.items() if i in members)
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
    """A resting stop a fixed distance from the anchor, replaced each rotation.

    For a long it is a sell stop *below* the anchor; for a short, a buy stop
    *above* it. The short side is not a mirror image in its risks, and the
    project's expert is explicit about it: a short's loss has no ceiling, a gap
    up through the level fills at the open however far away that is, and a
    squeeze or a recall can force a buy-in regardless of any stop. The stop
    bounds the ordinary case; it does not make a short as safe as a long.

    Two decisions that are easy to confuse, both settled by measurement over
    seventeen years:

    **Distance: a fixed fraction,** the same for a calm instrument and a
    volatile one. Volatility enters the system through position *size*, not
    through stop distance.

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

    def level(self, anchor: float, position: float = 1.0) -> float:
        """Where the stop sits for a position anchored at ``anchor``.

        Below the anchor for a long (``position > 0``), above it for a short.
        """
        if position < 0:
            return anchor * (1.0 + self.distance)
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
        """One resting stop per open position, on its closing side."""
        orders: list[OrderIntent] = []
        for instrument, position in sorted(positions.items(), key=lambda kv: str(kv[0])):
            anchor = position.anchor if position.anchor is not None else position.mark
            if anchor is None or anchor <= 0 or position.quantity == 0:
                continue
            rules = constraints_for(instrument)
            quantity = rules.round_quantity(abs(position.quantity))
            if quantity <= 0:
                continue
            side = Side.closing(position.quantity)
            level = rules.round_price(self.level(anchor, position.quantity))
            orders.append(
                OrderIntent(
                    client_order_id=client_order_id(
                        run, portfolio, instrument, moment, side, quantity
                    )
                    + "-stop",
                    run=run,
                    portfolio=portfolio,
                    instrument=instrument,
                    strategy_version=strategy_version,
                    side=side,
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
        marks: Mapping[InstrumentId, float] | None = None,
    ) -> RiskReview:
        """Every rule in order, then the invariant check on the result.

        ``marks`` must price every instrument an intent touches. When omitted,
        only held positions' marks are known, and a rule that needs to value a
        new position fails closed rather than guessing.
        """
        prices = {i: p.mark for i, p in positions.items()} if marks is None else dict(marks)
        current = tuple(intents)
        findings: list[RiskFinding] = []
        for rule in self.rules:
            current, produced = rule.apply(current, positions, equity, prices)
            findings.extend(produced)
        return RiskReview(
            proposed=tuple(intents), approved=current,
            held={i: p.quantity for i, p in positions.items()},
            findings=tuple(findings),
        )
