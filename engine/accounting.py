"""The book: cash, positions, and the only place either changes.

Accounting is in **shares and cash**. Weights are a derived view, computed from
the book when someone asks, and never stored. This is the single most important
constraint in the file, and it is here because of a measured failure: the
previous system tracked weights, and when a stop fired it renormalised the
remaining weights to sum to one. That silently deleted the stopped position's
value from the portfolio total, so a loss was recorded as "the rest of the book
is now bigger". It produced a 58% CAGR. An accounting error does not look like an
error — it looks like a brilliant strategy, which is why it survives review.

Two rules make that class of bug unrepresentable here:

1. **Equity is always ``cash + sum(quantity * price)``.** There is no other
   formula anywhere. A position that closes converts to cash; nothing is
   redistributed, because nothing was ever expressed as a share of a total.
2. **Only a fill moves the book.** :meth:`Book.apply` is the sole mutator and it
   is pure: it returns a new book. So the same fills in the same order always
   produce the same state, a run can be replayed from its fills alone, and there
   is no path by which a report, a risk check or a strategy can adjust the
   accounts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime

from contracts.errors import ContractViolation
from contracts.execution import Fill, PositionLedgerEntry
from contracts.identifiers import InstrumentId, PortfolioId
from contracts.temporal import utc


@dataclass(frozen=True, slots=True)
class Book:
    """Cash and positions for one portfolio at one instant.

    Attributes:
        portfolio: Whose book this is.
        cash: Settled cash. May go negative only if the caller allows margin;
            the engine checks before sizing, not here.
        positions: Open positions by instrument. A closed position is removed
            rather than kept at zero, so ``instrument in book.positions`` means
            exactly "we hold some".
        as_of: The moment this state is true at.
    """

    portfolio: PortfolioId
    cash: float
    as_of: datetime
    positions: Mapping[InstrumentId, PositionLedgerEntry] = field(default_factory=dict)
    realised_pnl: float = 0.0
    commission_paid: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", utc(self.as_of))
        object.__setattr__(self, "positions", dict(self.positions))
        for instrument, position in self.positions.items():
            if position.instrument != instrument:
                raise ContractViolation(
                    f"position filed under {instrument} reports {position.instrument}"
                )
            if position.quantity == 0:
                raise ContractViolation(
                    f"{instrument} is held at zero; close it by removing the entry"
                )

    @classmethod
    def opening(
        cls, portfolio: PortfolioId, cash: float, as_of: datetime
    ) -> Book:
        """An empty book with a starting balance."""
        if cash <= 0:
            raise ContractViolation(f"opening cash must be positive; got {cash}")
        return cls(portfolio=portfolio, cash=cash, as_of=as_of)

    # -- valuation ---------------------------------------------------------

    def market_value(self, prices: Mapping[InstrumentId, float]) -> float:
        """Value of the open positions. Every held instrument must have a price.

        A missing price is an error rather than a zero. Treating an unpriced
        holding as worthless marks the book down by its full value, and treating
        it as unchanged hides a halt or a delisting — both are silent.
        """
        missing = [str(i) for i in self.positions if i not in prices]
        if missing:
            raise ContractViolation(f"no price for held instrument(s): {sorted(missing)}")
        return sum(p.market_value(prices[i]) for i, p in self.positions.items())

    def equity(self, prices: Mapping[InstrumentId, float]) -> float:
        """Total value: cash plus positions. The only definition in the system."""
        return self.cash + self.market_value(prices)

    def weights(self, prices: Mapping[InstrumentId, float]) -> dict[InstrumentId, float]:
        """The book as fractions of equity. A derived view, never stored.

        Note what this does *not* do: it does not normalise the position values to
        sum to one. They sum to one minus the cash fraction, because cash is part
        of the book. That is the whole point.
        """
        total = self.equity(prices)
        if total <= 0:
            raise ContractViolation(f"cannot express weights against equity of {total}")
        return {i: p.market_value(prices[i]) / total for i, p in self.positions.items()}

    def cash_weight(self, prices: Mapping[InstrumentId, float]) -> float:
        """The fraction held in cash. Position weights plus this equal one."""
        return self.cash / self.equity(prices)

    def quantity(self, instrument: InstrumentId) -> float:
        """Shares held, zero if flat."""
        position = self.positions.get(instrument)
        return position.quantity if position else 0.0

    # -- the only mutator --------------------------------------------------

    def apply(self, fill: Fill) -> Book:
        """The book after ``fill``. Pure: returns a new book, changes nothing.

        Average cost is a weighted average over additions only. A partial sale
        does not change the cost basis of what remains — it realises profit on
        what left. Reducing the basis on a sale would move realised profit into
        unrealised and flatter whatever comes next.
        """
        if fill.at < self.as_of:
            raise ContractViolation(
                f"fill at {fill.at.isoformat()} is before the book at {self.as_of.isoformat()}; "
                f"fills must be applied in order"
            )

        existing = self.positions.get(fill.instrument)
        held = existing.quantity if existing else 0.0
        basis = existing.average_cost if existing else 0.0
        realised_before = existing.realised_pnl if existing else 0.0
        orders = existing.from_orders if existing else ()

        delta = fill.signed_quantity
        resulting = held + delta
        realised = 0.0

        if held == 0.0 or (held > 0) == (delta > 0):
            # Opening or adding: the basis is the weighted average of what was
            # paid, which is only defined over purchases.
            new_basis = (
                fill.price
                if held == 0.0
                else (abs(held) * basis + abs(delta) * fill.price) / (abs(held) + abs(delta))
            )
        else:
            # Reducing or reversing. Profit is realised on the overlap only.
            closed = min(abs(delta), abs(held))
            direction = 1.0 if held > 0 else -1.0
            realised = closed * (fill.price - basis) * direction
            # A reversal flips through zero onto a fresh basis; a partial sale
            # leaves the basis of the remainder alone.
            new_basis = fill.price if abs(delta) > abs(held) else basis

        positions = dict(self.positions)
        if resulting == 0.0:
            positions.pop(fill.instrument, None)
        else:
            positions[fill.instrument] = PositionLedgerEntry(
                portfolio=self.portfolio,
                instrument=fill.instrument,
                quantity=resulting,
                average_cost=new_basis,
                as_of=fill.at,
                realised_pnl=realised_before + realised,
                from_orders=(*orders, fill.client_order_id),
            )

        return replace(
            self,
            cash=self.cash + fill.cash_flow,
            positions=positions,
            as_of=fill.at,
            realised_pnl=self.realised_pnl + realised,
            commission_paid=self.commission_paid + fill.commission,
        )

    def apply_all(self, fills: Iterable[Fill]) -> Book:
        """Apply fills in order. Convenience only; the semantics are unchanged."""
        book = self
        for fill in fills:
            book = book.apply(fill)
        return book

    def at(self, moment: datetime) -> Book:
        """The same book, carried forward to a later moment without trading."""
        moment = utc(moment)
        if moment < self.as_of:
            raise ContractViolation(
                f"cannot carry the book backwards to {moment.isoformat()} "
                f"from {self.as_of.isoformat()}"
            )
        return replace(self, as_of=moment)


def replay(opening: Book, fills: Iterable[Fill]) -> Book:
    """Rebuild a book from its opening balance and its fills.

    This is the recovery path and the audit path at once. If a replay of the
    recorded fills does not reproduce the stored book, one of them is wrong, and
    the discrepancy is the finding.
    """
    return opening.apply_all(fills)
