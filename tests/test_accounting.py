"""The ledger, and the specific accounting error that produced a 58% CAGR.

Most of these are ordinary double-entry checks. The one that matters is
``test_a_stop_does_not_inflate_the_rest_of_the_book``: it is the regression for
the failure that made the previous system look brilliant.
"""

from __future__ import annotations

import pytest

from contracts.errors import ContractViolation
from contracts.execution import Fill, Side
from contracts.identifiers import InstrumentId, PortfolioId
from engine.accounting import Book, replay
from tests.conftest import at

AAA = InstrumentId("AAA")
BBB = InstrumentId("BBB")
CCC = InstrumentId("CCC")
DDD = InstrumentId("DDD")


def buy(instrument, quantity, price, moment, commission=0.0, tag="o"):
    return Fill(
        client_order_id=f"{tag}-{instrument}-buy",
        instrument=instrument,
        side=Side.BUY,
        quantity=quantity,
        price=price,
        at=moment,
        commission=commission,
    )


def sell(instrument, quantity, price, moment, commission=0.0, tag="o"):
    return Fill(
        client_order_id=f"{tag}-{instrument}-sell",
        instrument=instrument,
        side=Side.SELL,
        quantity=quantity,
        price=price,
        at=moment,
        commission=commission,
    )


@pytest.fixture
def opening(portfolio: PortfolioId) -> Book:
    return Book.opening(portfolio, cash=100_000.0, as_of=at(2020))


# -- the essentials ----------------------------------------------------------


def test_an_empty_book_is_worth_its_cash(opening):
    assert opening.equity({}) == 100_000.0
    assert opening.cash_weight({}) == 1.0
    assert opening.weights({}) == {}


def test_a_purchase_moves_cash_into_a_position(opening):
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3), commission=1.0))
    assert book.cash == pytest.approx(100_000 - 5_000 - 1.0)
    assert book.quantity(AAA) == 100
    assert book.positions[AAA].average_cost == 50.0
    # Equity drops by exactly the commission. Nothing else changed hands.
    assert book.equity({AAA: 50.0}) == pytest.approx(99_999.0)


def test_commission_is_spent_not_capitalised(opening):
    """Folding commission into the basis would hide it inside future profit."""
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3), commission=25.0))
    assert book.positions[AAA].average_cost == 50.0
    assert book.commission_paid == 25.0
    # Selling straight back at the same price realises nothing, and the round
    # trip costs exactly the two commissions.
    book = book.apply(sell(AAA, 100, 50.0, at(2020, 1, 4), commission=25.0))
    assert book.realised_pnl == pytest.approx(0.0)
    assert book.equity({}) == pytest.approx(99_950.0)


def test_adding_averages_the_basis_over_purchases_only(opening):
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3)))
    book = book.apply(buy(AAA, 100, 70.0, at(2020, 1, 10), tag="p"))
    assert book.quantity(AAA) == 200
    assert book.positions[AAA].average_cost == pytest.approx(60.0)


def test_a_partial_sale_leaves_the_remaining_basis_alone(opening):
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3)))
    book = book.apply(sell(AAA, 40, 80.0, at(2020, 1, 10)))
    assert book.quantity(AAA) == 60
    assert book.positions[AAA].average_cost == 50.0, "the basis of what remains is unchanged"
    assert book.realised_pnl == pytest.approx(40 * 30.0)


def test_closing_removes_the_position_rather_than_holding_it_at_zero(opening):
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3)))
    book = book.apply(sell(AAA, 100, 50.0, at(2020, 1, 10)))
    assert AAA not in book.positions
    assert book.quantity(AAA) == 0.0


def test_a_reversal_starts_a_fresh_basis(opening):
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3)))
    book = book.apply(sell(AAA, 150, 60.0, at(2020, 1, 10)))
    assert book.quantity(AAA) == -50
    assert book.realised_pnl == pytest.approx(100 * 10.0), "profit on the 100 that closed"
    assert book.positions[AAA].average_cost == 60.0, "the new short is based at 60"


# -- the regression ----------------------------------------------------------


def test_a_stop_does_not_inflate_the_rest_of_the_book(opening):
    """The 58% CAGR bug, made unrepresentable.

    Four equal positions; one is stopped out 20% down. The old system tracked
    weights and renormalised the survivors to sum to one, which deleted the loss
    from the total and read as "everything else just got bigger". Here a close
    converts to cash and equity falls by exactly the loss.
    """
    book = opening
    for instrument in (AAA, BBB, CCC, DDD):
        book = book.apply(buy(instrument, 250, 100.0, at(2020, 1, 3)))
    assert book.cash == pytest.approx(0.0)
    assert book.equity(dict.fromkeys((AAA, BBB, CCC, DDD), 100.0)) == pytest.approx(100_000.0)

    # AAA falls 20% and the stop fires. Everything else is untouched.
    book = book.apply(sell(AAA, 250, 80.0, at(2020, 2, 7)))
    prices = {BBB: 100.0, CCC: 100.0, DDD: 100.0}

    assert book.realised_pnl == pytest.approx(-5_000.0)
    assert book.cash == pytest.approx(20_000.0)
    assert book.equity(prices) == pytest.approx(95_000.0), "equity falls by the loss, exactly"

    # And the survivors did not grow. Each is still 25,000, now 26.3% of a
    # smaller book -- not 33.3% of a book that pretends the loss never happened.
    weights = book.weights(prices)
    assert all(w == pytest.approx(25_000 / 95_000) for w in weights.values())
    assert sum(weights.values()) + book.cash_weight(prices) == pytest.approx(1.0)
    assert sum(weights.values()) == pytest.approx(0.7895, abs=1e-4), "not 1.0"


def test_weights_never_normalise_away_cash(opening):
    book = opening.apply(buy(AAA, 100, 100.0, at(2020, 1, 3)))
    weights = book.weights({AAA: 100.0})
    assert weights[AAA] == pytest.approx(0.10)
    assert book.cash_weight({AAA: 100.0}) == pytest.approx(0.90)


# -- structural guarantees ---------------------------------------------------


def test_a_held_instrument_without_a_price_is_an_error_not_a_zero(opening):
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3)))
    with pytest.raises(ContractViolation, match="no price"):
        book.equity({})
    with pytest.raises(ContractViolation, match="AAA"):
        book.market_value({BBB: 10.0})


def test_fills_must_be_applied_in_order(opening):
    book = opening.apply(buy(AAA, 100, 50.0, at(2020, 6, 1)))
    with pytest.raises(ContractViolation, match="before the book"):
        book.apply(buy(BBB, 10, 10.0, at(2020, 1, 3)))


def test_applying_a_fill_does_not_change_the_original(opening):
    after = opening.apply(buy(AAA, 100, 50.0, at(2020, 1, 3)))
    assert opening.cash == 100_000.0
    assert opening.positions == {}
    assert after is not opening


def test_replaying_the_fills_reproduces_the_book(opening):
    fills = [
        buy(AAA, 100, 50.0, at(2020, 1, 3), commission=1.0),
        buy(BBB, 200, 25.0, at(2020, 1, 10), commission=1.0),
        sell(AAA, 40, 80.0, at(2020, 2, 7), commission=1.0),
        buy(AAA, 10, 90.0, at(2020, 3, 6), commission=1.0, tag="p"),
    ]
    direct = opening.apply_all(fills)
    rebuilt = replay(opening, fills)
    assert rebuilt == direct
    assert rebuilt.cash == pytest.approx(direct.cash)
    assert rebuilt.positions[AAA] == direct.positions[AAA]


def test_the_book_can_be_carried_forward_but_not_backward(opening):
    later = opening.at(at(2021))
    assert later.as_of.year == 2021
    assert later.cash == opening.cash
    with pytest.raises(ContractViolation, match="backwards"):
        later.at(at(2019))


def test_a_position_held_at_zero_is_refused(portfolio):
    from contracts.execution import PositionLedgerEntry

    with pytest.raises(ContractViolation, match="zero"):
        Book(
            portfolio=portfolio,
            cash=1.0,
            as_of=at(2020),
            positions={
                AAA: PositionLedgerEntry(
                    portfolio=portfolio,
                    instrument=AAA,
                    quantity=0.0,
                    average_cost=0.0,
                    as_of=at(2020),
                )
            },
        )
