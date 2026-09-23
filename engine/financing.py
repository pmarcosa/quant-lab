"""What it costs to carry a levered or short book, and how close it is to a margin call.

A long-only book fully paid for costs nothing to hold. Two things change that:

* **Borrowed cash.** Buying more than the equity leaves a debit balance, and the
  broker charges interest on it every day. IBKR Pro charges its benchmark rate
  plus a spread by balance tier (benchmark + 1.5% on the first $100k).
* **Borrowed stock.** A short pays the lender a fee on the shares' value for as
  long as it is open.

Both accrue with calendar time, whatever the bar size, so a weekly backtest
charges seven days per bar and a daily one one to three. Leaving them out is the
project's expert's first listed way a levered backtest "becomes an illusion".

The maintenance-margin estimate follows Reg T's house minimums (25% of long
value, 30% of short value). IBKR sets higher requirements for volatile names, so
the live system asks the broker for the real figure; this one exists so a
backtest can report how close it came.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from contracts.errors import ContractViolation
from contracts.execution import Charge, ChargeKind
from contracts.identifiers import InstrumentId
from engine.accounting import Book

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class FinancingModel:
    """Annual rates for carrying costs, and the maintenance-margin estimate.

    Attributes:
        margin_rate: Annual interest on a debit balance. The default is IBKR
            Pro's first-tier USD rate in September 2026 (5.38%), rounded up. A
            constant rate overstates the cost in the zero-rate years and
            understates it in 2023; it is a stated assumption, not a history.
        borrow_fee: Annual fee on the value of shorted stock. 0.5% is a
            general-collateral assumption; hard-to-borrow names cost far more,
            and the live system checks availability and, when reported, the fee.
        maintenance_long: Maintenance requirement per unit of long value.
        maintenance_short: Maintenance requirement per unit of short value.
    """

    margin_rate: float = 0.055
    borrow_fee: float = 0.005
    maintenance_long: float = 0.25
    maintenance_short: float = 0.30

    def __post_init__(self) -> None:
        for name in ("margin_rate", "borrow_fee", "maintenance_long", "maintenance_short"):
            if getattr(self, name) < 0:
                raise ContractViolation(f"{name} cannot be negative")

    def charges(
        self, book: Book, prices: Mapping[InstrumentId, float], start: datetime, end: datetime
    ) -> tuple[Charge, ...]:
        """What carrying ``book`` from ``start`` to ``end`` cost, as charges at ``end``.

        The balance is taken at ``start`` -- the book as it stood through the
        period -- and priced at ``prices``.
        """
        days = (end - start).total_seconds() / 86_400.0
        if days <= 0:
            return ()
        out: list[Charge] = []
        debit = max(-book.cash, 0.0)
        if debit > 0 and self.margin_rate > 0:
            out.append(Charge(
                at=end, amount=debit * self.margin_rate * days / DAYS_PER_YEAR,
                kind=ChargeKind.MARGIN_INTEREST,
                detail=f"debit {debit:,.2f} at {self.margin_rate:.2%} for {days:.1f} days",
            ))
        short_value = short_market_value(book, prices)
        if short_value > 0 and self.borrow_fee > 0:
            out.append(Charge(
                at=end, amount=short_value * self.borrow_fee * days / DAYS_PER_YEAR,
                kind=ChargeKind.BORROW_FEE,
                detail=f"short value {short_value:,.2f} at {self.borrow_fee:.2%} for {days:.1f} days",
            ))
        return tuple(out)

    def cushion(self, book: Book, prices: Mapping[InstrumentId, float]) -> float:
        """IBKR's "cushion": excess liquidity over net liquidation, 1 - MM/NLV.

        One for a book with no positions; zero or below at the point the broker
        starts liquidating without notice.
        """
        equity = book.equity({i: prices[i] for i in book.positions})
        if equity <= 0:
            return 0.0
        long_value = sum(
            p.quantity * prices[i] for i, p in book.positions.items() if p.quantity > 0
        )
        requirement = (
            self.maintenance_long * long_value
            + self.maintenance_short * short_market_value(book, prices)
        )
        return 1.0 - requirement / equity


def short_market_value(book: Book, prices: Mapping[InstrumentId, float]) -> float:
    """The value of the shares owed, positive."""
    return sum(-p.quantity * prices[i] for i, p in book.positions.items() if p.quantity < 0)


def gross_leverage(book: Book, prices: Mapping[InstrumentId, float]) -> float:
    """Longs plus shorts, over equity."""
    equity = book.equity({i: prices[i] for i in book.positions})
    if equity <= 0:
        return float("inf")
    return sum(abs(p.quantity) * prices[i] for i, p in book.positions.items()) / equity
