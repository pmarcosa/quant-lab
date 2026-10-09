"""Contract invariants: the promises the types make."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contracts.errors import ContractViolation
from contracts.execution import (
    BrokerCapabilities,
    Fill,
    InstrumentConstraints,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    client_order_id,
    is_stop_order,
    stop_order_id,
)
from contracts.identifiers import (
    InstrumentId,
    PortfolioId,
    RunId,
    StrategyId,
    StrategyVersion,
    TenantId,
)
from contracts.targets import TargetIntent
from contracts.temporal import BarInterval, FiltrationSpec, Observation, utc
from contracts.universe import Membership

NOW = datetime(2026, 9, 18, 20, 0, tzinfo=timezone.utc)


# -- identifiers -------------------------------------------------------------


def test_parameters_are_part_of_strategy_identity() -> None:
    """Changing a parameter produces a different version, not the same one tuned.

    This is what stops validation evidence earned by one parameterisation being
    silently claimed by another.
    """
    family = StrategyId("xs_momentum")
    thirteen = StrategyVersion.of(family, {"lookback_weeks": 13, "top_n": 4})
    twenty_six = StrategyVersion.of(family, {"lookback_weeks": 26, "top_n": 4})
    assert thirteen != twenty_six
    assert thirteen.strategy == twenty_six.strategy


def test_parameter_order_does_not_change_identity() -> None:
    family = StrategyId("xs_momentum")
    one = StrategyVersion.of(family, {"a": 1, "b": 2})
    other = StrategyVersion.of(family, {"b": 2, "a": 1})
    assert one == other


def test_unserialisable_parameters_are_refused() -> None:
    with pytest.raises(ContractViolation, match="JSON-serialisable"):
        StrategyVersion.of(StrategyId("x"), {"callback": object()})


def test_identifiers_reject_unsafe_values() -> None:
    for bad in ("", "../escape", "Has Spaces", "UPPER"):
        with pytest.raises(ContractViolation):
            TenantId(bad)


# -- temporal ----------------------------------------------------------------


def test_naive_timestamps_are_refused() -> None:
    with pytest.raises(ContractViolation, match="timezone-naive"):
        utc(datetime(2026, 1, 1))


def test_availability_cannot_precede_the_event() -> None:
    with pytest.raises(ContractViolation, match="precedes event_time"):
        Observation(event_time=NOW, available_time=NOW - timedelta(hours=1), payload=1.0)


def test_observation_is_not_knowable_before_it_is_published() -> None:
    observation = Observation(NOW, NOW + timedelta(hours=2), 1.0)
    assert not observation.knowable_at(NOW + timedelta(hours=1))
    assert observation.knowable_at(NOW + timedelta(hours=3))


def test_negative_observation_lag_is_refused() -> None:
    with pytest.raises(ContractViolation, match="look-ahead"):
        FiltrationSpec(BarInterval.WEEK, observation_lag_bars=-1)


def test_annualisation_factors_live_in_one_place() -> None:
    assert BarInterval.DAY.periods_per_year == 252
    assert BarInterval.WEEK.periods_per_year == 52


# -- targets -----------------------------------------------------------------


def test_weights_that_look_like_currency_are_refused() -> None:
    """A weight of 4500 is a bug worth failing on, not a position worth sizing."""
    with pytest.raises(ContractViolation, match="fractions of equity"):
        TargetIntent(weights={InstrumentId("x"): 4500.0}, horizon_bars=1, as_of=NOW)


def test_non_finite_weights_are_refused() -> None:
    with pytest.raises(ContractViolation, match="not finite"):
        TargetIntent(weights={InstrumentId("x"): float("nan")}, horizon_bars=1, as_of=NOW)


def test_empty_intent_is_a_position() -> None:
    """Holding nothing is a decision, and the type says so."""
    intent = TargetIntent(weights={}, horizon_bars=4, as_of=NOW)
    assert intent.is_cash
    assert intent.gross == 0.0


def test_gross_and_net_differ_for_a_long_short_book() -> None:
    intent = TargetIntent(
        weights={InstrumentId("a"): 0.5, InstrumentId("b"): -0.3}, horizon_bars=4, as_of=NOW
    )
    assert intent.gross == pytest.approx(0.8)
    assert intent.net == pytest.approx(0.2)


# -- execution ---------------------------------------------------------------


def test_the_same_decision_produces_the_same_order_id() -> None:
    """Idempotency: a retry after a timeout must not become a second order."""
    args = (
        RunId("r1"),
        PortfolioId(TenantId("user"), "ibkr-main"),
        InstrumentId("265598"),
        NOW,
        Side.BUY,
        12.0,
    )
    assert client_order_id(*args) == client_order_id(*args)


def test_a_different_quantity_produces_a_different_order_id() -> None:
    base = (RunId("r1"), PortfolioId(TenantId("user"), "m"), InstrumentId("i"), NOW, Side.BUY)
    assert client_order_id(*base, 12.0) != client_order_id(*base, 13.0)


def test_a_replaced_stop_gets_a_new_id_and_the_first_keeps_the_old_one() -> None:
    """A stop placed again for the same decision must not reuse the cancelled one's id."""
    base = (RunId("r1"), PortfolioId(TenantId("user"), "m"), InstrumentId("i"), NOW, Side.SELL, 12.0)
    first = stop_order_id(*base)
    assert first == client_order_id(*base) + "-stop", "ids made before placements existed"
    assert stop_order_id(*base, placement=1) not in (first, stop_order_id(*base, placement=2))
    assert is_stop_order(first) and not is_stop_order(client_order_id(*base))


def test_quantity_rounds_down_and_respects_the_minimum() -> None:
    constraints = InstrumentConstraints(InstrumentId("i"), "USD", lot_step=1.0, min_quantity=1.0)
    assert constraints.round_quantity(12.7) == 12.0
    assert constraints.round_quantity(0.4) == 0.0


def test_an_exact_multiple_of_a_fractional_lot_is_not_rounded_down() -> None:
    """0.3 / 0.1 is 2.9999999999999996 in floating point; three lots are still three."""
    constraints = InstrumentConstraints(InstrumentId("i"), "USD", lot_step=0.1, min_quantity=0.1)
    assert constraints.round_quantity(0.3) == 0.3
    assert constraints.round_quantity(0.7 - 0.3) == 0.4
    assert constraints.round_quantity(0.39) == 0.3, "a fraction still rounds down"


def test_price_rounds_to_the_tick() -> None:
    constraints = InstrumentConstraints(InstrumentId("i"), "USD", tick_size=0.01)
    assert constraints.round_price(291.4873) == pytest.approx(291.49)


def test_unsupported_order_types_are_refused_by_the_broker_contract() -> None:
    capabilities = BrokerCapabilities(
        "test", frozenset({OrderType.MARKET}), frozenset({"USD"})
    )
    capabilities.require(OrderType.MARKET)
    with pytest.raises(ContractViolation, match="does not support"):
        capabilities.require(OrderType.STOP)


def test_direction_lives_in_side_not_in_the_sign_of_quantity() -> None:
    with pytest.raises(ContractViolation, match="direction is carried by side"):
        OrderIntent(
            client_order_id="x",
            run=RunId("r"),
            portfolio=PortfolioId(TenantId("user"), "m"),
            instrument=InstrumentId("i"),
            strategy_version=StrategyVersion.of(StrategyId("s"), {}),
            side=Side.SELL,
            quantity=-5.0,
            order_type=OrderType.MARKET,
            decision_time=NOW,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_fill_or_an_order_without_a_finite_number_is_refused(bad) -> None:
    """NaN fails every comparison, so ``x <= 0`` let it through to poison the book."""
    with pytest.raises(ContractViolation):
        Fill("x", InstrumentId("i"), Side.BUY, quantity=1.0, price=bad, at=NOW)
    with pytest.raises(ContractViolation):
        Fill("x", InstrumentId("i"), Side.BUY, quantity=bad, price=10.0, at=NOW)
    with pytest.raises(ContractViolation):
        Fill("x", InstrumentId("i"), Side.BUY, quantity=1.0, price=10.0, at=NOW, commission=bad)
    with pytest.raises(ContractViolation):
        OrderIntent(
            client_order_id="x", run=RunId("r"), portfolio=PortfolioId(TenantId("user"), "m"),
            instrument=InstrumentId("i"), strategy_version=StrategyVersion.of(StrategyId("s"), {}),
            side=Side.BUY, quantity=bad, order_type=OrderType.MARKET, decision_time=NOW,
        )


def test_unknown_is_not_a_terminal_status() -> None:
    """After a timeout the system knows nothing, and must reconcile rather than assume."""
    assert not OrderStatus.UNKNOWN.is_terminal
    assert OrderStatus.FILLED.is_terminal


# -- universe ----------------------------------------------------------------


def test_membership_covers_only_its_window() -> None:
    window = Membership(
        InstrumentId("zm"),
        datetime(2019, 4, 18, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert not window.covers(datetime(2018, 6, 1, tzinfo=timezone.utc))
    assert window.covers(datetime(2020, 6, 1, tzinfo=timezone.utc))
    assert not window.covers(datetime(2026, 6, 1, tzinfo=timezone.utc))


def test_delisting_before_listing_is_refused() -> None:
    with pytest.raises(ContractViolation, match="before it joined"):
        Membership(
            InstrumentId("x"),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2019, 1, 1, tzinfo=timezone.utc),
        )
