"""The IBKR adapter, against a stand-in gateway built from real ib_async types."""

from __future__ import annotations

import pytest

pytest.importorskip("ib_async")

from contracts.errors import ContractViolation  # noqa: E402
from contracts.execution import (  # noqa: E402
    ExecutionPort,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from contracts.identifiers import InstrumentId, RunId, StrategyVersion  # noqa: E402
from contracts.live import TradingMode  # noqa: E402
from execution.ibkr import IBKRBroker  # noqa: E402
from tests.conftest import at  # noqa: E402
from tests.fake_gateway import FakeGateway  # noqa: E402

AAA = InstrumentId("AAA")
VERSION = StrategyVersion.of("s", {"a": 1})


@pytest.fixture
def gateway():
    return FakeGateway(prices={"AAA": 100.0, "BBB": 50.0})


@pytest.fixture
def broker(gateway):
    return IBKRBroker(gateway, "DU1234567", TradingMode.PAPER, settle_seconds=0)


def intent(portfolio, side=Side.BUY, quantity=10, order_type=OrderType.MARKET,
           tif=TimeInForce.OPG, stop=None, oid="ql-0000000000000000001"):
    return OrderIntent(
        client_order_id=oid, run=RunId("live"), portfolio=portfolio, instrument=AAA,
        strategy_version=VERSION, side=side, quantity=quantity, order_type=order_type,
        decision_time=at(2026, 9, 18), stop_price=stop, time_in_force=tif,
    )


# -- the account guard -------------------------------------------------------


def test_the_adapter_satisfies_the_port(broker):
    assert isinstance(broker, ExecutionPort)


def test_a_live_account_is_refused_in_paper_mode():
    gateway = FakeGateway(accounts=("U7654321",))
    with pytest.raises(ContractViolation, match="not a paper account"):
        IBKRBroker(gateway, "U7654321", TradingMode.PAPER)


def test_a_paper_account_is_refused_in_live_mode():
    with pytest.raises(ContractViolation, match="not a live account"):
        IBKRBroker(FakeGateway(), "DU1234567", TradingMode.LIVE)


def test_an_account_the_gateway_does_not_manage_is_refused():
    with pytest.raises(ContractViolation, match="Check the config"):
        IBKRBroker(FakeGateway(accounts=("DU9999999",)), "DU1234567", TradingMode.PAPER)


# -- orders ------------------------------------------------------------------


def test_a_rotation_order_goes_to_the_opening_auction(broker, gateway, portfolio):
    state = broker.submit(intent(portfolio))
    assert state.status is OrderStatus.ACCEPTED
    (trade,) = gateway.trades()
    assert trade.order.orderType == "MKT"
    assert trade.order.tif == "OPG"
    assert trade.order.orderRef == "ql-0000000000000000001"
    assert trade.order.account == "DU1234567"
    assert trade.order.action == "BUY"


def test_submitting_the_same_order_twice_places_it_once(broker, gateway, portfolio):
    """A retry after a timeout must not become a second order."""
    first = broker.submit(intent(portfolio))
    second = broker.submit(intent(portfolio))
    assert gateway.placed == 1
    assert first.broker_order_id == second.broker_order_id


def test_a_filled_order_is_not_resubmitted_either(broker, gateway, portfolio):
    broker.submit(intent(portfolio))
    gateway.opening_auction({"AAA": 101.0})
    again = broker.submit(intent(portfolio))
    assert gateway.placed == 1
    assert again.status is OrderStatus.FILLED


def test_a_stop_rests_as_a_gtc_stop_order(broker, gateway, portfolio):
    broker.submit(intent(portfolio, side=Side.SELL, order_type=OrderType.STOP,
                         tif=TimeInForce.GTC, stop=88.0, oid="ql-stop-1"))
    (trade,) = gateway.trades()
    assert trade.order.orderType == "STP"
    assert trade.order.auxPrice == 88.0
    assert trade.order.tif == "GTC"
    (working,) = broker.working_orders()
    assert working.stop_price == 88.0
    assert working.client_order_id == "ql-stop-1"


def test_status_is_observed_and_an_unknown_order_says_so(broker, gateway, portfolio):
    broker.submit(intent(portfolio))
    gateway.opening_auction({"AAA": 99.5})
    (filled,) = broker.poll(["ql-0000000000000000001"])
    assert filled.status is OrderStatus.FILLED
    assert filled.average_fill_price == 99.5
    (missing,) = broker.poll(["ql-never-sent"])
    assert missing.status is OrderStatus.UNKNOWN
    assert not missing.status.is_terminal


def test_a_partial_fill_is_reported_as_one(broker, gateway, portfolio):
    broker.submit(intent(portfolio, quantity=10))
    gateway.execute(gateway.trade_for("ql-0000000000000000001"), 100.0, quantity=4)
    (state,) = broker.poll(["ql-0000000000000000001"])
    assert state.status is OrderStatus.PARTIALLY_FILLED
    assert state.filled_quantity == 4


def test_cancelling_works_and_cancelling_twice_is_harmless(broker, portfolio):
    broker.submit(intent(portfolio, tif=TimeInForce.DAY))
    assert broker.cancel("ql-0000000000000000001").status is OrderStatus.CANCELLED
    assert broker.cancel("ql-0000000000000000001").status is OrderStatus.CANCELLED
    assert broker.cancel("ql-unknown").status is OrderStatus.UNKNOWN


# -- fills and positions -----------------------------------------------------


def test_fills_carry_the_execution_id_and_our_order_id(broker, gateway, portfolio):
    broker.submit(intent(portfolio))
    gateway.opening_auction({"AAA": 101.0})
    (found,) = broker.fills()
    assert found.execution_id.startswith("0001.")
    assert found.fill.client_order_id == "ql-0000000000000000001"
    assert found.fill.price == 101.0
    assert found.fill.side is Side.BUY
    assert found.fill.commission == 1.0


def test_trades_made_by_hand_are_not_absorbed_into_the_sleeve(broker, gateway):
    """An execution without our order reference was not placed by this system."""
    import ib_async

    manual = gateway.placeOrder(
        ib_async.Stock("AAA", "SMART", "USD"),
        ib_async.Order(action="BUY", totalQuantity=5, orderType="MKT", orderRef=""),
    )
    gateway.execute(manual, 100.0)
    assert broker.fills() == []
    assert {str(p.instrument) for p in broker.positions(None)} == {"AAA"}, (
        "but the position is still visible, so reconciliation will notice it"
    )


def test_positions_come_from_the_broker(broker, gateway, portfolio):
    gateway.hold("AAA", 20, 95.0)
    (position,) = broker.positions(portfolio)
    assert position.instrument == AAA
    assert position.quantity == 20
    assert position.average_cost == 95.0


def test_the_account_snapshot_reads_net_liquidation_and_cash(broker, gateway):
    gateway.hold("AAA", 10, 100.0)
    snapshot = broker.account_snapshot()
    assert snapshot.cash == pytest.approx(100_000.0)
    assert snapshot.net_liquidation == pytest.approx(101_000.0)


def test_tick_size_comes_from_contract_details(broker):
    rules = broker.constraints(AAA)
    assert rules.tick_size == 0.01
    assert rules.lot_step == 1.0
