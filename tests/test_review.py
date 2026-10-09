"""The weekly review: the connector's answers in, the week's decision out.

Every payload here is made up in the connector's own shapes. What the tests
hold the review to is that it adds nothing of its own to the decision: the
names come from the strategy, the shares from the engine, the stops from the
risk rule, the ladder from monitoring.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from runtime import connector
from runtime import review as weekly
from runtime.config import (
    AutomationSettings,
    Definition,
    ExecutionSettings,
    MonitoringSettings,
    ReviewSettings,
    RiskSettings,
    StrategySettings,
    load_config,
    load_definition,
)
from runtime.connector import Payloads
from runtime.monitor import Baseline
from runtime.strategies import build_strategy
from strategies.momentum import indicators

WEEKS = 120
#: The Friday the made-up histories end on, and a Monday review after it.
LAST_CLOSE = datetime(2026, 10, 2, 21, tzinfo=timezone.utc)
MONDAY = datetime(2026, 10, 5, 6, tzinfo=timezone.utc)


def _climb(top: float) -> np.ndarray:
    """A climb whose last four weeks carry well over 40% of the quarter."""
    return np.concatenate([np.linspace(100, top * 0.94, WEEKS - 4), np.linspace(top * 0.955, top, 4)])


SHAPES = {
    "RISE": _climb(200.0),
    "FAST": _climb(300.0),
    "PLOD": np.linspace(100, 220, WEEKS),       # up and steady: fails a 40% pace filter
    "FALL": np.linspace(220, 100, WEEKS),       # breaks its trend: an exit when held
    "SPY": np.linspace(400, 500, WEEKS),
}


def history(closes, dividends_on=(), partial_week=True, last_close=LAST_CLOSE) -> dict:
    """A ``get_price_history`` answer: weekly bars labelled by their Monday.

    Like the real thing it opens with part of a week and, when ``partial_week``
    is set, closes with a bar for a week that has not ended.
    """
    closes = [float(c) for c in closes]
    last_monday = last_close.date() - timedelta(days=4)
    mondays = [last_monday - timedelta(days=7 * k) for k in range(len(closes) - 1, -1, -1)]
    times = [(mondays[0] - timedelta(days=4)).isoformat() + "T00:00:00Z"]   # the opening stub
    prices = [closes[0]]
    for monday, close in zip(mondays, closes, strict=True):
        times.append(monday.isoformat() + "T00:00:00Z")
        prices.append(close)
    if partial_week:
        times.append((last_monday + timedelta(days=7)).isoformat() + "T00:00:00Z")
        prices.append(closes[-1] * 1.5)   # a wild move that must not be seen
    payload = {
        "chart_step": 604800, "time": times, "open": list(prices),
        "high": [p * 1.01 for p in prices], "low": [p * 0.99 for p in prices],
        "close": list(prices), "volume": [1000.0] * len(prices),
    }
    if dividends_on:
        payload["corp_actions"] = [
            {"type": "CashDividends", "date": day.strftime("%Y%m%d"), "value": str(amount),
             "currency": "USD"}
            for day, amount in dividends_on
        ]
    return payload


def account(cash: float = 1_000.0, net: float | None = None) -> dict:
    return {"currency": "USD", "net_liquidation": net if net is not None else cash,
            "total_cash_value": cash}


def positions(**held: tuple[float, float]) -> dict:
    return {"positions": [
        {"contract_id": 100 + k, "contract_description": symbol, "position": quantity,
         "market_price": price, "average_price": price, "currency": "USD", "asset_class": "STK"}
        for k, (symbol, (quantity, price)) in enumerate(held.items())
    ]}


def stop_order(symbol: str, quantity: float, stop: float, order_id: int = 1) -> dict:
    return {"order_id": order_id, "order_status": "NEW", "order_type": "STOP_LIMIT",
            "side": "SELL", "limit_price": str(round(stop * 0.995, 2)),
            "total_shares_qty": str(quantity), "remaining_shares_qty": str(quantity),
            "primary_description": f"Sell {quantity:g} {symbol}",
            "secondary_description": f"STP {stop:.2f} LMT {stop * 0.995:.2f}, GTC",
            "order_time": "2026-09-21T08:00:00Z"}


def performance(weekly_returns: list[float], last_friday: date | None = None) -> dict:
    """A time-weighted series, one point per Friday, ending on ``last_friday``."""
    last_friday = last_friday or LAST_CLOSE.date()
    fridays = [last_friday - timedelta(days=7 * k) for k in range(len(weekly_returns), -1, -1)]
    growth = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(weekly_returns))])
    return {"portfolio_measure": "TWR", "accounts": {"account": {"periods": {"1Y": {
        "dates": [d.strftime("%Y%m%d") for d in fridays],
        "cps": [float(g - 1.0) for g in growth], "nav": [float(10_000 * g) for g in growth],
    }}}}}


@pytest.fixture
def universe(tmp_path: Path) -> str:
    path = tmp_path / "made-up.txt"
    path.write_text("RISE\nFAST\nPLOD\nFALL\n")
    return str(path)


def definition(universe: str, monitor_from: str | None = None, **params) -> Definition:
    settings = {"rebalance_weeks": 1, "top_n": 0, "pace_ratio_min": 0.40, "cost_stop_loss": 0.0,
                **params}
    return Definition(
        strategy_id="momentum",
        strategy=StrategySettings(name="weekly-momentum", params=settings, universe=universe),
        execution=ExecutionSettings(cash_buffer=0.0, no_trade_band=0.0, commission="ibkr-tiered"),
        risk=RiskSettings(stop_distance=0.12, stop_limit_offset=0.005),
        monitoring=MonitoringSettings(bootstrap_paths=500),
        automation=AutomationSettings(),
        review=ReviewSettings(monitor_from=monitor_from),
    )


def payloads(held: dict | None = None, cash: float = 10_000.0, orders=(), **changes) -> Payloads:
    found = Payloads(
        history={symbol: history(closes) for symbol, closes in SHAPES.items()},
        account=account(cash), positions=positions(**(held or {})),
        orders={"orders": list(orders)}, trades={"trades": []},
    )
    for name, value in changes.items():
        setattr(found, name, value)
    return found


def a_baseline(version, seed: int = 3) -> Baseline:
    rng = np.random.default_rng(seed)
    return Baseline(
        strategy_version=str(version), settings={}, interval="1Week",
        returns=[float(x) for x in rng.normal(0.003, 0.03, 600)], modeled_bps=10.0,
        expected_rotation_return=0.01, first_bar="2010-01-01", last_bar="2026-10-02",
        built_at="2026-10-09T00:00:00+00:00",
    )


def version_of(d: Definition):
    return build_strategy(d.strategy.name, d.strategy.params).version


# -- the connector's bars --------------------------------------------------------------------


def test_bars_are_whole_weeks_that_have_ended():
    bars = connector.weekly_bars(history(SHAPES["RISE"]), MONDAY)
    assert len(bars) == WEEKS, "the opening stub and the unfinished week are not weeks"
    assert bars.index[-1] == LAST_CLOSE
    assert bars["close"].iloc[-1] == pytest.approx(200.0)
    assert (bars.index.weekday == 4).all() and (bars.index.hour == 21).all()


def test_a_week_is_not_seen_before_it_ends():
    """On the Friday afternoon the week's bar exists and is not yet a week."""
    friday_afternoon = LAST_CLOSE - timedelta(hours=5)
    bars = connector.weekly_bars(history(SHAPES["RISE"]), friday_afternoon)
    assert bars.index[-1] == LAST_CLOSE - timedelta(days=7)


def test_a_bar_for_today_belongs_to_its_week():
    """The connector closes its answer with a bar for the day of the request."""
    payload = history(SHAPES["RISE"], partial_week=False)
    for name, value in (("time", "2026-10-02T00:00:00Z"), ("open", 199.0), ("high", 207.0),
                        ("low", 198.0), ("close", 205.0), ("volume", 10.0)):
        payload[name].append(value)
    last = connector.weekly_bars(payload, MONDAY).iloc[-1]
    assert last["close"] == 205.0, "the week's close is the last trade of the week"
    assert last["high"] == 207.0 and last["low"] == pytest.approx(198.0)
    assert last["volume"] == 1010.0


def test_a_holiday_monday_does_not_move_the_week():
    payload = history(SHAPES["RISE"])
    payload["time"][-3] = "2026-09-22T00:00:00Z"   # that week's bar starts on a Tuesday
    bars = connector.weekly_bars(payload, MONDAY)
    assert len(bars) == WEEKS and bars.index.is_unique
    assert bars.index[-2] == LAST_CLOSE - timedelta(days=7)


def test_a_daily_history_is_refused():
    payload = {**history(SHAPES["RISE"]), "chart_step": 86400}
    with pytest.raises(ContractViolation, match="weekly"):
        connector.weekly_bars(payload, MONDAY)


def test_a_dividend_is_taken_out_of_the_weeks_before_it():
    ex_day = date(2026, 9, 9)                     # in the week that closes 11 September
    raw = connector.weekly_bars(history(SHAPES["PLOD"], dividends_on=[(ex_day, 2.0)]), MONDAY)
    adjusted = connector.adjusted(raw, connector.dividends(
        history(SHAPES["PLOD"], dividends_on=[(ex_day, 2.0)])))
    ex_week = connector.week_close(ex_day)
    before = raw.index < ex_week
    factor = 1.0 - 2.0 / raw.loc[before, "close"].iloc[-1]
    assert np.allclose(adjusted.loc[before, "close"], raw.loc[before, "close"] * factor)
    assert np.allclose(adjusted.loc[before, "high"], raw.loc[before, "high"] * factor)
    assert np.allclose(adjusted.loc[~before, "close"], raw.loc[~before, "close"])
    assert adjusted["close"].iloc[-1] == raw["close"].iloc[-1], "the last bar is the price as traded"
    assert (adjusted["volume"] == raw["volume"]).all()


def test_a_dividend_in_a_week_not_yet_closed_changes_nothing():
    raw = connector.weekly_bars(history(SHAPES["PLOD"]), MONDAY)
    upcoming = (connector.Dividend(date(2026, 10, 7), 2.0),)
    assert connector.adjusted(raw, upcoming).equals(raw)


def test_an_ex_dividend_drop_is_not_a_loss_to_the_strategy():
    """The same business, with and without a payout: the signals must agree."""
    params = build_strategy("weekly-momentum", {}).params
    flat = np.full(WEEKS, 100.0)
    paying = flat.copy()
    paying[-6:] -= 5.0                              # 5 paid six weeks ago: the price steps down
    ex_day = (LAST_CLOSE - timedelta(days=7 * 5 + 2)).date()
    raw = connector.weekly_bars(history(paying, dividends_on=[(ex_day, 5.0)]), MONDAY)
    adjusted = connector.adjusted(raw, (connector.Dividend(ex_day, 5.0),))
    as_traded = indicators(raw, params).iloc[-1]
    total_return = indicators(adjusted, params).iloc[-1]
    assert as_traded["ret_long"] == pytest.approx(-0.05), "read as traded, it lost 5%"
    assert total_return["ret_long"] == pytest.approx(0.0, abs=1e-12), "it lost nothing"


# -- the connector's account -----------------------------------------------------------------


def test_a_stop_is_read_from_its_description():
    (order,) = connector.open_orders({"orders": [stop_order("PANW", 2, 356.91)]})
    assert (order.symbol, order.quantity, order.stop_price) == ("PANW", 2.0, 356.91)
    assert order.is_stop and order.good_till_cancelled and order.side == "SELL"


def test_finished_orders_are_not_working_orders():
    done = {**stop_order("PANW", 2, 356.91), "order_status": "CANCELLED"}
    assert connector.open_orders({"orders": [done]}) == ()


def test_a_money_weighted_series_is_refused():
    payload = performance([0.01, 0.02])
    payload["portfolio_measure"] = "MWR"
    with pytest.raises(ContractViolation, match="time-weighted"):
        connector.performance(payload)


def test_a_quote_is_the_last_trade_else_the_midpoint():
    assert connector.quote({"last": {"price": 10.5}, "bid-ask": {"bid": 10, "ask": 12}}) == 10.5
    assert connector.quote({"last": {}, "bid-ask": {"bid": 10, "ask": 12}}) == 11.0
    assert connector.quote({"last": {}, "bid-ask": {}}) is None


# -- a session's log -------------------------------------------------------------------------


def _log(path: Path, exchanges: list[tuple[str, str, dict, object]]) -> Path:
    """A session log: one tool call and its answer per exchange, dated as given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for number, (at, tool, arguments, result) in enumerate(exchanges):
        call_id = f"{path.stem}-{number}"
        lines.append({"message": {"content": [
            {"type": "tool_use", "id": call_id, "name": tool, "input": arguments}]}})
        content = result if isinstance(result, str) else json.dumps(result)
        lines.append({"timestamp": at, "message": {"content": [
            {"type": "tool_result", "tool_use_id": call_id,
             "content": [{"type": "text", "text": content}]}]}})
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


IBKR = "mcp__Interactive_Brokers_IBKR__"


def test_answers_are_found_in_the_session_and_in_its_helpers(tmp_path: Path):
    watch = {"instruments": [{"contract_id_ex": "11", "contract_description": "RISE"},
                             {"contract_id_ex": "12", "contract_description": "FAST"}]}
    weekly_call = {"security_type": "STK", "step": "ONE_WEEK"}
    main = _log(tmp_path / "session.jsonl", [
        ("2026-10-05T06:00:00Z", IBKR + "get_watchlist", {"id": "104"}, watch),
        ("2026-10-05T06:00:10Z", IBKR + "get_account_summary", {}, account(500.0)),
        ("2026-10-05T06:00:20Z", IBKR + "get_price_history",
         {"contract_id": 11, **weekly_call}, history(SHAPES["PLOD"])),
        ("2026-10-05T06:00:30Z", IBKR + "get_price_history",
         {"contract_id": 99, **weekly_call}, history(SHAPES["FALL"])),
        ("2026-10-05T06:00:40Z", IBKR + "get_price_history",
         {"contract_id": 12, "security_type": "STK", "step": "ONE_DAY"}, history(SHAPES["FALL"])),
        ("2026-10-05T06:00:50Z", IBKR + "get_price_snapshot", {"contract_id": 12},
         {"last": {"price": 301.0}}),
        ("2026-10-05T06:00:55Z", IBKR + "get_account_orders", {}, "not json at all"),
    ])
    _log(tmp_path / "session" / "subagents" / "agent-1.jsonl", [
        ("2026-10-05T06:05:00Z", IBKR + "get_price_history",
         {"contract_id": 11, **weekly_call}, history(SHAPES["RISE"])),
        ("2026-10-05T06:05:10Z", IBKR + "get_price_history",
         {"contract_id": 12, **weekly_call}, history(SHAPES["FAST"])),
    ])
    found = connector.gather(main)
    assert sorted(found.history) == ["FAST", "RISE"]
    assert found.history["RISE"]["close"][-2] == 200.0, "the helper's later answer replaces the first"
    assert found.unnamed == [99], "a history that cannot be named is reported, not guessed"
    assert connector.quote(found.quotes["FAST"]) == 301.0
    assert connector.cash(found.account) == 500.0
    assert found.orders is None, "an answer that is not JSON is not an answer"


def test_saved_answers_load_back_the_same(tmp_path: Path):
    before = payloads(held={"RISE": (10, 200.0)}, performance=performance([0.01]))
    before.quotes["RISE"] = {"last": {"price": 201.0}}
    before.save(tmp_path / "answers")
    after = Payloads.load(tmp_path / "answers")
    assert after.history == before.history and after.quotes == before.quotes
    assert after.positions == before.positions and after.performance == before.performance


# -- the decision ----------------------------------------------------------------------------


def test_the_review_buys_what_the_strategy_and_the_engine_say(universe: str):
    d = definition(universe)
    result = weekly.review(payloads(), d, MONDAY)
    assert result["week"] == "rebalance" and result["decision_bar"] == "2026-10-02"
    assert result["selection"]["best"] == ["FAST", "RISE"], "by 13-week return"
    orders = {o["symbol"]: o for o in result["orders"]}
    assert set(orders) == {"FAST", "RISE"}
    target = result["target"]
    assert sum(target.values()) == pytest.approx(1.0)
    for symbol, close in (("FAST", 300.0), ("RISE", 200.0)):
        order = orders[symbol]
        assert order["side"] == "BUY" and order["type"] == "LIMIT" and order["time_in_force"] == "DAY"
        assert order["quantity"] == np.floor(target[symbol] * 10_000.0 / close), "whole shares, down"
        assert order["limit"] == pytest.approx(round(close * 1.005, 2)), "half a percent through"
        assert order["decision_price"] == close
    assert result["cash_after"]["cash"] >= -0.005 * 10_000.0


def test_the_first_pass_of_a_rotation_asks_for_quotes(universe: str):
    d = definition(universe)
    first = weekly.review(payloads(), d, MONDAY)
    assert first["final"] is False and first["quotes_needed"] == ["FAST", "RISE"]
    quoted = payloads()
    quoted.quotes = {"FAST": {"last": {"price": 306.0}}, "RISE": {"last": {"price": 198.0}}}
    second = weekly.review(quoted, d, MONDAY)
    assert second["final"] is True and second["quotes_needed"] == []
    orders = {o["symbol"]: o for o in second["orders"]}
    assert orders["FAST"]["limit"] == pytest.approx(round(306.0 * 1.005, 2))
    assert orders["FAST"]["decision_price"] == 300.0, "the decision was made at the close"
    stops = {s["symbol"]: s for s in second["stops"]["place"]}
    assert stops["FAST"]["stop"] == pytest.approx(round(306.0 * 0.88, 2)), "12% under today's price"
    assert stops["FAST"]["limit"] == pytest.approx(round(stops["FAST"]["stop"] * 0.995, 2))
    assert stops["FAST"]["quantity"] == orders["FAST"]["quantity"]


def test_a_rotation_chosen_from_part_of_the_universe_is_not_final(universe: str):
    partial = payloads()
    del partial.history["FAST"]
    result = weekly.review(partial, definition(universe), MONDAY)
    assert result["history_needed"] == ["FAST"] and result["final"] is False


def test_a_symbol_whose_bars_stop_early_is_left_out(universe: str):
    late = payloads()
    late.history["FAST"] = history(SHAPES["FAST"], last_close=LAST_CLOSE - timedelta(days=7),
                                   partial_week=False)
    result = weekly.review(late, definition(universe), MONDAY)
    assert result["data"]["stale"] == ["FAST"]
    assert "FAST" not in [row["symbol"] for row in result["scan"]]


def test_a_week_between_rotations_proposes_nothing(universe: str):
    d = definition(universe, rebalance_weeks=4)
    strategy = build_strategy(d.strategy.name, d.strategy.params)
    quiet = next(MONDAY - timedelta(days=7 * k) for k in range(4)
                 if not strategy.rotates_at(LAST_CLOSE - timedelta(days=7 * k) + timedelta(minutes=15)))
    held = {"RISE": (10, 200.0)}
    result = weekly.review(payloads(held=held), d, quiet)
    assert result["week"] == "monitoring"
    assert result["orders"] == [] and result["stops"]["replace"] is False
    assert result["final"] is True and result["quotes_needed"] == []
    upcoming = date.fromisoformat(result["next_rebalance"])
    assert upcoming.weekday() == 0 and 0 < (upcoming - quiet.date()).days <= 28
    assert "RISE" in result["selection"]["best"], "the scan is shown all the same"


def test_frozen_positions_and_exits_are_the_strategys(universe: str):
    held = {"PLOD": (15, 220.0), "FALL": (20, 100.0)}
    result = weekly.review(payloads(held=held, cash=4_700.0), definition(universe), MONDAY)
    status = {row["symbol"]: row["status"] for row in result["scan"]}
    assert status["PLOD"] == "frozen" and status["FALL"].startswith("exit: trend_break")
    orders = {o["symbol"]: o for o in result["orders"]}
    assert "PLOD" not in orders, "a frozen position is not traded"
    assert orders["FALL"]["side"] == "SELL" and orders["FALL"]["quantity"] == 20
    assert result["target"]["PLOD"] == pytest.approx(15 * 220.0 / 10_000.0, abs=1e-6)
    placed = {s["symbol"]: s["quantity"] for s in result["stops"]["place"]}
    assert placed["PLOD"] == 15 and "FALL" not in placed, "stops follow the book after the trades"


def test_a_position_outside_the_universe_is_reported_and_left_alone(universe: str):
    held = {"RISE": (10, 200.0), "GLD": (3, 300.0)}
    result = weekly.review(payloads(held=held, cash=8_000.0), definition(universe), MONDAY)
    assert [p["symbol"] for p in result["account"]["outside_the_strategy"]] == ["GLD"]
    assert result["account"]["managed_equity"] == pytest.approx(10_000.0)
    assert "GLD" not in {o["symbol"] for o in result["orders"]}


def test_the_account_and_the_positions_are_required(universe: str):
    blind = payloads()
    blind.positions = None
    with pytest.raises(ContractViolation, match="positions"):
        weekly.review(blind, definition(universe), MONDAY)


# -- stops -----------------------------------------------------------------------------------


def test_stop_checks_name_what_is_wrong(universe: str):
    held = {"RISE": (10, 200.0), "FAST": (5, 300.0), "PLOD": (4, 220.0)}
    orders = [stop_order("RISE", 8, 176.0, 1), stop_order("PLOD", 4, 190.0, 2),
              stop_order("PLOD", 4, 185.0, 3), stop_order("GONE", 7, 50.0, 4)]
    result = weekly.review(payloads(held=held, cash=5_620.0, orders=orders),
                           definition(universe), MONDAY)
    problems = "\n".join(result["stop_checks"]["problems"])
    assert "RISE: the stop is for 8 shares, the position is 10" in problems
    assert "FAST: no stop for its 5 shares" in problems
    assert "PLOD: 2 stops resting" in problems
    assert "GONE: 1 stop(s) resting with no position" in problems
    assert result["stop_checks"]["ok"] is False
    assert len(result["stops"]["cancel"]) == 4, "a rotation replaces every stop, strays included"


# -- monitoring ------------------------------------------------------------------------------


def test_weekly_returns_come_from_the_brokers_series():
    growth = connector.performance(performance([0.01, -0.02, 0.03]))
    returns = weekly.weekly_returns(growth, LAST_CLOSE)
    assert list(returns) == ["2026-09-18", "2026-09-25", "2026-10-02"]
    assert list(returns.values()) == pytest.approx([0.01, -0.02, 0.03])
    earlier = weekly.weekly_returns(growth, LAST_CLOSE - timedelta(days=7))
    assert list(earlier) == ["2026-09-18", "2026-09-25"], "a week is counted once it has ended"


def test_the_live_record_starts_with_the_rules(universe: str):
    d = definition(universe, monitor_from="2026-09-21")   # the Monday of the first rotation
    series = performance([0.05, 0.04, 0.01, -0.02])        # weeks to 11, 18, 25 Sep and 2 Oct
    result = weekly.review(payloads(performance=series), d, MONDAY, baseline=a_baseline(version_of(d)))
    watch = result["monitoring"]
    assert list(watch["returns"]) == ["2026-09-25", "2026-10-02"], "earlier weeks were other rules"
    assert watch["weeks"] == 2 and watch["judged"] is True and watch["state"] == "normal"


def test_without_a_baseline_the_ladder_is_not_judged(universe: str):
    d = definition(universe, monitor_from="2026-09-21")
    result = weekly.review(payloads(performance=performance([0.01, -0.5])), d, MONDAY)
    assert result["monitoring"]["judged"] is False and result["monitoring"]["state"] == "normal"
    assert any("no published baseline" in note for note in result["notes"])


def test_a_week_the_backtest_almost_never_had_halts(universe: str):
    d = definition(universe, monitor_from="2026-09-21")
    series = performance([0.01, 0.01, 0.0, -0.30])
    held = {"RISE": (10, 200.0)}
    result = weekly.review(payloads(held=held, cash=8_000.0, performance=series), d, MONDAY,
                           baseline=a_baseline(version_of(d)))
    watch = result["monitoring"]
    assert watch["state"] == "halted" and watch["breaker"]["fired"] is True
    assert result["orders"] == [], "a halted system proposes nothing"
    assert len(result["orders_held_back"]) > 0
    assert result["stops"]["replace"] is False, "and leaves the resting stops where they are"
    assert result["state"]["ladder"]["state"] == "halted"

    # The halt is kept, and a better week does not lift it.
    calmer = performance([0.01, 0.01, 0.0, -0.30, 0.04], last_friday=date(2026, 10, 9))
    later = payloads(held=held, cash=8_000.0, performance=calmer)
    for symbol, closes in SHAPES.items():
        later.history[symbol] = history(np.append(closes, closes[-1]),
                                        last_close=LAST_CLOSE + timedelta(days=7))
    again = weekly.review(later, d, MONDAY + timedelta(days=7), state=result["state"],
                          baseline=a_baseline(version_of(d)))
    assert again["monitoring"]["state"] == "halted" and again["orders"] == []

    cleared = weekly.clear_ladder(again["state"], "normal", "looked at it: one bad week, no bug",
                                  MONDAY + timedelta(days=8))
    resumed = weekly.review(later, d, MONDAY + timedelta(days=8), state=cleared,
                            baseline=a_baseline(version_of(d)))
    assert resumed["monitoring"]["state"] == "normal", "the same bar is judged once"
    assert resumed["orders"] != []


def test_clearing_needs_a_reason_and_goes_up(universe: str):
    halted = {**weekly.empty_state("v"), "ladder": {"state": "halted", "by": "monitor",
                                                    "reason": "x", "since": None}}
    with pytest.raises(ContractViolation, match="real reason"):
        weekly.clear_ladder(halted, "normal", "ok", MONDAY)
    normal = weekly.empty_state("v")
    with pytest.raises(ContractViolation, match="up the ladder"):
        weekly.clear_ladder(normal, "reduce_only", "a long enough reason", MONDAY)


def test_reduce_only_lets_exits_through_and_nothing_else(universe: str):
    d = definition(universe)
    paused = {**weekly.empty_state(version_of(d)),
              "ladder": {"state": "reduce_only", "by": "person", "reason": "on holiday",
                         "since": None}}
    held = {"FALL": (20, 100.0)}
    result = weekly.review(payloads(held=held, cash=8_000.0), d, MONDAY, state=paused,
                           baseline=a_baseline(version_of(d)))
    assert [(o["symbol"], o["side"]) for o in result["orders"]] == [("FALL", "SELL")]
    assert {o["symbol"] for o in result["orders_held_back"]} == {"FAST", "RISE"}
    assert result["monitoring"]["state"] == "reduce_only", "a person's pause is not monitoring's to lift"


def test_early_weeks_show_the_return_checks_without_acting_on_them(universe: str):
    """A collapse over a few weeks, none of them bad enough alone for the breaker."""
    d = definition(universe, monitor_from="2026-07-06")
    baseline = a_baseline(version_of(d))
    floor = float(np.quantile(baseline.returns, 0.005))
    slide = [max(-0.06, floor + 0.005)] * 8
    result = weekly.review(payloads(performance=performance([0.0, *slide])), d, MONDAY,
                           baseline=baseline)
    watch = result["monitoring"]
    assert watch["burn_in"] is True and watch["weeks"] < d.review.burn_in_weeks
    assert watch["shown_but_not_acted_on"], "the drawdown is seen"
    assert watch["state"] == "normal" and watch["breaker"]["fired"] is False

    patient = Definition(**{**{f: getattr(d, f) for f in d.__dataclass_fields__},
                            "review": ReviewSettings(monitor_from="2026-07-06", burn_in_weeks=4)})
    acted = weekly.review(payloads(performance=performance([0.0, *slide])), patient, MONDAY,
                          baseline=a_baseline(version_of(patient)))
    assert acted["monitoring"]["burn_in"] is False
    assert acted["monitoring"]["state"] in ("reduce_only", "halted")


def test_fills_are_matched_to_the_rotation_that_proposed_them(universe: str):
    d = definition(universe)
    quoted = payloads()
    quoted.quotes = {"FAST": {"last": {"price": 300.0}}, "RISE": {"last": {"price": 200.0}}}
    proposed = weekly.review(quoted, d, MONDAY, baseline=a_baseline(version_of(d)))
    (rotation,) = proposed["state"]["rotations"]
    assert rotation["id"] == "2026-10-02" and rotation["final"] is True
    quantities = {p["symbol"]: p["quantity"] for p in rotation["proposals"]}

    def trade(number: int, symbol: str, price: float, when: str, kind: str = "LIMIT") -> dict:
        return {"trade_id": f"t{number}", "order_id": number, "symbol": symbol, "side": "BUY",
                "size": quantities.get(symbol, 1), "price": price, "commission": 1.0,
                "trade_time": when, "order_type": kind, "sec_type": "STK"}

    week_later = payloads(held={"FAST": (quantities["FAST"], 303.0),
                                "RISE": (quantities["RISE"], 200.0)}, cash=100.0)
    for symbol, closes in SHAPES.items():
        week_later.history[symbol] = history(np.append(closes, closes[-1] * 1.01),
                                             last_close=LAST_CLOSE + timedelta(days=7))
    week_later.trades = {"trades": [
        trade(1, "FAST", 303.0, "2026-10-05T13:35:00Z"),
        trade(2, "RISE", 200.0, "2026-10-05T13:36:00Z"),
        trade(3, "RISE", 150.0, "2026-09-28T13:36:00Z"),            # before the decision
        trade(4, "PLOD", 220.0, "2026-10-05T13:37:00Z"),            # never proposed
        trade(5, "FAST", 280.0, "2026-10-06T15:00:00Z", "STOP_LIMIT"),   # a stop, not a rotation
    ]}
    result = weekly.review(week_later, d, MONDAY + timedelta(days=7), state=proposed["state"],
                           baseline=a_baseline(version_of(d)))
    (kept,) = [r for r in result["state"]["rotations"] if r["id"] == "2026-10-02"]
    assert sorted(f["trade_id"] for f in kept["fills"]) == ["t1", "t2"]
    assert kept["opens"]["FAST"] == pytest.approx(300.0 * 1.01), "the open the backtest would use"
    (line,) = [c for c in result["monitoring"]["compliance"] if c["rotation"] == "2026-10-02"]
    assert (line["buys_done"], line["buys"]) == (2, 2)
    shortfall = result["monitoring"]["shortfall"]
    assert shortfall["fills"] == 2
    assert shortfall["per_rotation_bps"]["2026-10-02"] > 0, "FAST filled 1% above its decision price"

    # Reading the same trades again adds nothing.
    twice = weekly.review(week_later, d, MONDAY + timedelta(days=7), state=result["state"],
                          baseline=a_baseline(version_of(d)))
    (same,) = [r for r in twice["state"]["rotations"] if r["id"] == "2026-10-02"]
    assert len(same["fills"]) == 2


def test_the_state_survives_a_round_trip_and_a_change_of_rules(universe: str, tmp_path: Path):
    d = definition(universe)
    result = weekly.review(payloads(), d, MONDAY)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(result["state"]))
    assert weekly.load_state(path, version_of(d)) == result["state"]

    halted = {**result["state"], "ladder": {"state": "halted", "by": "monitor", "reason": "r",
                                            "since": None}}
    other = definition(universe, top_n=1)
    carried = weekly.load_state(halted, version_of(other))
    assert carried["ladder"]["state"] == "halted", "a new version does not lift a halt"
    assert carried["rotations"] == [] and carried["weekly_returns"] == {}


def test_the_summary_says_what_the_review_decided(universe: str):
    held = {"PLOD": (15, 220.0)}
    text = weekly.summary(weekly.review(payloads(held=held, cash=6_700.0), definition(universe),
                                        MONDAY))
    assert "REBALANCE WEEK" in text and "NOT FINAL" in text
    assert "frozen: PLOD" in text and "PLOD: no stop for its 15 shares" in text
    assert "BUY" in text and "STP" in text


# -- one definition, several users -----------------------------------------------------------


def test_a_config_that_extends_a_definition_restates_nothing(tmp_path: Path, universe: str):
    shared = tmp_path / "definitions" / "momentum.yaml"
    shared.parent.mkdir()
    shared.write_text(f"""
strategy_id: momentum
strategy:
  name: weekly-momentum
  params: {{rebalance_weeks: 4, top_n: 5, pace_ratio_min: 0.1231}}
  universe: {universe}
execution: {{cash_buffer: 0.0, commission: ibkr-tiered}}
risk: {{stop_limit_offset: 0.005}}
review: {{monitor_from: 2026-10-19}}
""")
    private = tmp_path / "strategies" / "momentum.yaml"
    private.parent.mkdir()
    private.write_text("""
extends: ../definitions/momentum.yaml
mode: paper
account: DU1234567
sleeve_capital: 15000
strategy:
  params: {top_n: 3}
""")
    config = load_config(private)
    d = load_definition(shared)
    assert config.execution == d.execution and config.risk == d.risk and config.review == d.review
    assert config.review.monitor_from == "2026-10-19"
    assert dict(config.strategy.params) == {**dict(d.strategy.params), "top_n": 3}, \
        "an override is one visible line, and the rest comes from the definition"
    assert config.strategy.universe == d.strategy.universe and config.strategy_id == "momentum"


def test_a_definition_names_no_account(tmp_path: Path):
    path = tmp_path / "leaky.yaml"
    path.write_text("strategy_id: momentum\naccount: U7654321\nmode: live\n")
    with pytest.raises(ContractViolation, match="names no account"):
        load_definition(path)
    private = tmp_path / "private.yaml"
    private.write_text("extends: leaky.yaml\nmode: paper\naccount: DU1234567\nsleeve_capital: 1\n")
    with pytest.raises(ContractViolation, match="names no account"):
        load_config(private)


def test_the_published_definition_loads_and_has_its_baseline():
    """What the weekly task runs from, as committed."""
    from runtime.config import DEFINITIONS

    d = load_definition(DEFINITIONS / "momentum.yaml")
    version = build_strategy(d.strategy.name, d.strategy.params).version
    baseline = weekly.load_baseline(version)
    assert baseline is not None, f"publish it: ql review baseline (no baselines/{version}.json)"
    assert baseline.strategy_version == str(version)
    assert dict(baseline.settings["params"]) == dict(d.strategy.params)
    assert InstrumentId("x") == InstrumentId("x")


# -- from the command line -------------------------------------------------------------------


def test_the_review_runs_from_saved_answers(tmp_path: Path, universe: str):
    from runtime.cli import Context, main

    shared = tmp_path / "made-up.yaml"
    shared.write_text(f"""
strategy_id: momentum
strategy:
  name: weekly-momentum
  params: {{rebalance_weeks: 1, top_n: 0, pace_ratio_min: 0.4}}
  universe: {universe}
execution: {{cash_buffer: 0.0, no_trade_band: 0.0}}
""")
    answers = payloads()
    answers.save(tmp_path / "answers")
    lines: list[str] = []
    context = Context(out=lines.append, clock=lambda: MONDAY)
    common = ["review", "run", "--definition", str(shared), "--inputs", str(tmp_path / "answers"),
              "--out", str(tmp_path / "review.json"), "--state-out", str(tmp_path / "state.json")]

    assert main(common, context=context) == 3, "a rotation without quotes is not final"
    assert not (tmp_path / "state.json").exists(), "and nothing is kept from a draft"
    assert any("NOT FINAL" in line for line in "\n".join(lines).splitlines())

    answers.quotes = {s: {"last": {"price": float(SHAPES[s][-1])}} for s in ("FAST", "RISE")}
    answers.save(tmp_path / "answers")
    assert main(common, context=Context(out=lines.append, clock=lambda: MONDAY)) == 0
    result = json.loads((tmp_path / "review.json").read_text())
    kept = json.loads((tmp_path / "state.json").read_text())
    assert result["final"] is True and kept == result["state"]
    assert kept["rotations"][0]["proposals"], "the proposals are kept for next week's check"

    kept["ladder"] = {"state": "halted", "by": "monitor", "reason": "a test", "since": None}
    (tmp_path / "state.json").write_text(json.dumps(kept))
    assert main(["review", "clear", "--state", str(tmp_path / "state.json"), "--to", "normal",
                 "--reason", "looked at it and it was a test"],
                context=Context(out=lines.append, clock=lambda: MONDAY)) == 0
    assert json.loads((tmp_path / "state.json").read_text())["ladder"]["state"] == "normal"
