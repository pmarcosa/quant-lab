"""The weekly review: the strategy's decision from the broker connector's data.

``ql live`` talks to a gateway on the machine it runs on. This is the same
decision made where there is no gateway: a scheduled session reaches the broker
through the IBKR connector, hands this module the connector's answers, and gets
back what to propose. The session does the fetching and the telling; every
rule is here, which means every rule is the one the backtest ran:

- signals come from :class:`strategies.momentum.WeeklyMomentum` -- the filters,
  the exits, the ranking, the frozen positions and the weights;
- whole shares come from :func:`engine.decide.decide`;
- stop levels come from :class:`risk.rules.ProtectiveStop`;
- the degradation ladder comes from :func:`validation.monitoring.assess`.

Nothing in this module restates one of those. What it adds is the plumbing
between them and a session with no memory:

**Prices.** The connector's bars are rebuilt on the total-return scale before
the strategy reads them (``runtime.connector``). Orders, stops and the value
of the book are priced as traded.

**A review proposes; a person approves.** The orders are day limit orders a
little through a reference price -- the latest quote when the session supplies
one, else the last weekly close -- because that is what the broker's order
instructions can carry and what a person can sanity-check. On a rotation the
review first says which quotes it needs and is then run again with them.

**State.** A scheduled session starts from nothing, so what monitoring needs
from the past travels in one small JSON document the session keeps between
runs: the ladder's state, each rotation's proposals and the fills that
followed, and the weekly returns once they are older than the broker's window.
The live return itself is not kept: it is read each week from the broker's
time-weighted performance series, which a deposit does not move.

**The live record starts where the rules did.** ``review.monitor_from`` is the
first rotation traded on this strategy version. Weeks before it were another
strategy's and are never compared with this one's backtest. Until
``review.burn_in_weeks`` have passed, the drawdown and changepoint checks are
shown but only the execution-cost check and the loss breaker may move the
ladder: the project's expert asks for both, and both follow from how little a
dozen weekly returns can say.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts.errors import ContractViolation
from contracts.execution import InstrumentConstraints, PositionLedgerEntry, Side
from contracts.identifiers import InstrumentId, PortfolioId, RunId, TenantId
from contracts.live import DegradationState
from contracts.risk import PositionRisk
from contracts.targets import Holdings
from engine.accounting import Book
from engine.decide import decide
from runtime import connector
from runtime.config import ROOT, Definition
from runtime.connector import Payloads
from runtime.strategies import build_strategy
from validation.monitoring import ExecutionRecord, Thresholds, assess

STATE_FORMAT = 1
BASELINES = ROOT / "baselines"
TENANT = TenantId("user")

#: A rotation's fills are looked for this long after its decision: the orders
#: are day orders placed on the Monday, re-placed at most for a few sessions.
FILL_WINDOW = timedelta(days=7)
#: How much of the past the state document carries.
KEEP_ROTATIONS = 60
#: A week in which the positions' own move and the account's reported return
#: differ by more than this, with no trade to explain it, is flagged.
RETURN_TOLERANCE = 0.0075


# -- what the strategy sees ------------------------------------------------------------------


class MemoryFiltration:
    """A filtration over frames already cut at the decision time.

    The frames hold closed bars only and end at the decision bar, so there is
    no future here to ask for.
    """

    def __init__(self, decision_time: datetime, frames: Mapping[InstrumentId, pd.DataFrame]):
        self._decision_time = decision_time
        self._frames = dict(frames)

    @property
    def decision_time(self) -> datetime:
        return self._decision_time

    def history(self, instrument: InstrumentId, field: str, count: int) -> pd.Series:
        frame = self._frames.get(instrument)
        if frame is None or field not in frame.columns:
            return pd.Series(dtype="float64")
        return frame[field].tail(count).astype(float)

    def frame(self, instruments: Sequence[InstrumentId], field: str, count: int) -> pd.DataFrame:
        columns = {str(i): self.history(i, field, count) for i in instruments}
        populated = {name: series for name, series in columns.items() if not series.empty}
        return pd.DataFrame(populated).sort_index() if populated else pd.DataFrame()

    def universe(self, min_bars: int | None = None) -> tuple[InstrumentId, ...]:
        needed = 1 if min_bars is None else min_bars
        return tuple(sorted(
            (i for i, frame in self._frames.items() if len(frame) >= needed), key=str
        ))


# -- the baseline ----------------------------------------------------------------------------


def baseline_file(version: object, folder: Path = BASELINES) -> Path:
    """Where a strategy version's monitoring baseline is published."""
    return folder / f"{version}.json"


def load_baseline(version: object, folder: Path = BASELINES):
    """The published baseline for exactly this version, or None.

    Named by the version, so a change of rules cannot be judged against the
    old rules' backtest by leaving a file in place.
    """
    from runtime.monitor import Baseline

    path = baseline_file(version, folder)
    if not path.exists():
        return None
    baseline = Baseline.load(path)
    if baseline.strategy_version != str(version):
        raise ContractViolation(
            f"{path.name} holds the baseline of {baseline.strategy_version}, not {version}"
        )
    return baseline


# -- state -----------------------------------------------------------------------------------


def empty_state(version: object) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT, "strategy_version": str(version),
        "ladder": {"state": DegradationState.NORMAL.value, "by": "system", "reason": "",
                   "since": None},
        "weekly_returns": {}, "rotations": [], "breaker_bars": [], "last_review": None,
    }


def load_state(source: str | Path | Mapping[str, Any] | None, version: object) -> dict[str, Any]:
    """The state a session kept -- a file, its text, or the mapping -- or a fresh one.

    A state written for another strategy version keeps its ladder -- a halt is
    a person's to lift, not a version bump's -- and starts the record again.
    """
    if source is None:
        return empty_state(version)
    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        text = source.read_text(encoding="utf-8") if isinstance(source, Path) else str(source)
        raw = json.loads(text) if text.strip() else {}
    if not raw:
        return empty_state(version)
    if raw.get("format") != STATE_FORMAT:
        raise ContractViolation("the review state is from another version of the format")
    fresh = empty_state(version)
    if raw.get("strategy_version") != str(version):
        fresh["ladder"] = raw.get("ladder") or fresh["ladder"]
        fresh["previous_version"] = raw.get("strategy_version")
        return fresh
    return {**fresh, **raw}


def clear_ladder(state: Mapping[str, Any], to: str, reason: str, at: datetime) -> dict[str, Any]:
    """A person moves the ladder back up, with a written reason."""
    if len(reason.strip()) < 10:
        raise ContractViolation(
            "clearing a degraded state needs a real reason (at least ten characters); "
            "it is the record of why it was safe to resume"
        )
    target = DegradationState(to)
    current = DegradationState((state.get("ladder") or {}).get("state", "normal"))
    if target.rank >= current.rank:
        raise ContractViolation(
            f"clear moves up the ladder; {target.value} is not above {current.value}"
        )
    out = dict(state)
    out["ladder"] = {"state": target.value, "by": "person", "reason": reason.strip(),
                     "since": at.isoformat()}
    return out


# -- the review ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prices:
    """One symbol's bars, both ways."""

    traded: pd.DataFrame
    total_return: pd.DataFrame
    payouts: int


def _prices(payloads: Payloads, symbols: Sequence[str], as_of: datetime,
            notes: list[str], actions: list[str]) -> dict[str, Prices]:
    found: dict[str, Prices] = {}
    for symbol in symbols:
        payload = payloads.history.get(symbol)
        if payload is None:
            continue
        try:
            traded = connector.weekly_bars(payload, as_of)
        except ContractViolation as error:
            notes.append(f"{symbol}: {error}")
            continue
        if traded.empty:
            continue
        payouts = connector.dividends(payload)
        actions.extend(f"{symbol}: {action}" for action in connector.other_actions(payload))
        found[symbol] = Prices(traded, connector.adjusted(traded, payouts), len(payouts))
    return found


def _next_rotation(strategy, after: datetime) -> date | None:
    """The Monday that follows the next decision the strategy rotates at."""
    moment = after
    for _ in range(60):
        moment += timedelta(days=7)
        if strategy.rotates_at(moment):
            return (moment + timedelta(days=3)).date()
    return None


def _round(value: float | None, places: int = 6) -> float | None:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), places)


def review(
    payloads: Payloads,
    definition: Definition,
    as_of: datetime,
    state: Mapping[str, Any] | None = None,
    baseline=None,
) -> dict[str, Any]:
    """Everything one weekly review decides, as a plain document.

    Args:
        payloads: The connector's answers (``runtime.connector``).
        definition: The strategy (``configs/definitions``).
        as_of: When the review is made. Bars of weeks that have not ended by
            then are not seen.
        state: What the previous reviews kept (:func:`load_state`).
        baseline: The monitoring baseline for this strategy version, or None.

    Returns:
        A JSON-ready mapping. ``state`` inside it is what to keep for the next
        review; ``quotes_needed`` lists symbols to quote before a rotation's
        orders are final.

    Raises:
        ContractViolation: If the account, the positions or every price are
            missing: a review that cannot see the book decides nothing.
    """
    from runtime.wiring import universe_list

    if payloads.account is None or payloads.positions is None:
        raise ContractViolation(
            "the review needs the account summary and the positions; neither a rule nor a "
            "price can stand in for what is held"
        )
    notes: list[str] = []
    strategy = build_strategy(definition.strategy.name, definition.strategy.params)
    interval = strategy.filtration_spec.interval
    version = strategy.version
    kept = load_state(state, version)
    if kept.get("previous_version"):
        notes.append(
            f"the kept state was for {kept['previous_version']}; the record starts again "
            f"for {version}"
        )

    universe = universe_list(definition.strategy.universe)
    if universe is None:
        raise ContractViolation("the review needs a named universe (strategy.universe)")
    benchmark = definition.review.benchmark
    actions: list[str] = []
    prices = _prices(payloads, [*universe.symbols, benchmark], as_of, notes, actions)
    tradable = {s: p for s, p in prices.items() if s in universe.symbols}
    if not tradable:
        raise ContractViolation("no price history for any symbol of the universe")

    # The decision bar is the last week that has ended. A symbol whose history
    # stops earlier is left out rather than read as if its last bar were now.
    expected = connector.last_closed_week(as_of)
    ends = pd.Series({s: p.traded.index[-1] for s, p in tradable.items()})
    decision_bar = ends.max()
    # No history reaching the week that has ended is a failed fetch, not a quiet
    # week: deciding on last week's bar would call a rotation a monitoring week.
    behind = bool(decision_bar < expected)
    if behind:
        notes.append(
            f"the newest bar closes {decision_bar.date()}, but the week to "
            f"{expected.date()} has ended: the data is behind"
        )
    stale = sorted(ends.index[ends < decision_bar])
    for symbol in stale:
        notes.append(f"{symbol}: last bar {ends[symbol].date()}, not used this week")
    current = {s: p for s, p in tradable.items() if s not in stale}
    missing = sorted(set(universe.symbols) - set(tradable))
    decision_time = decision_bar.to_pydatetime() + timedelta(minutes=15)
    # A recent listing is short by nature and the strategy leaves it out by its
    # own rule. Many short histories at once mean the session asked for too little.
    short = {s: len(p.traded) for s, p in sorted(current.items()) if len(p.traded) < 60}
    if len(short) > 5:
        notes.append(
            f"{len(short)} stocks have under 60 weekly bars: RSI and ATR are not settled on so "
            f"little; ask for two years of history"
        )

    filtration = MemoryFiltration(
        decision_time, {InstrumentId(s): p.total_return for s, p in current.items()}
    )

    # -- the book, as traded ---------------------------------------------------
    quotes = {
        s: q for s, q in ((s, connector.quote(p)) for s, p in payloads.quotes.items())
        if q is not None and q > 0
    }
    closes = {s: float(p.traded["close"].iloc[-1]) for s, p in current.items()}
    held = connector.positions(payloads.positions)
    managed = [p for p in held if p.symbol in universe.symbols
               and p.symbol not in definition.unmanaged and p.asset_class == "STK"]
    outside = [p for p in held if p not in managed]

    def reference(symbol: str) -> float | None:
        return quotes.get(symbol) or closes.get(symbol)

    marks: dict[InstrumentId, float] = {InstrumentId(s): reference(s) for s in closes}
    for position in managed:
        if InstrumentId(position.symbol) not in marks:
            marks[InstrumentId(position.symbol)] = position.market_price
            notes.append(
                f"{position.symbol} is held but has no bar this week; valued at the broker's "
                f"price and left as it is"
            )
    portfolio = PortfolioId(TENANT, definition.strategy_id)
    book = Book(
        portfolio=portfolio, cash=connector.cash(payloads.account), as_of=decision_time,
        positions={
            InstrumentId(p.symbol): PositionLedgerEntry(
                portfolio=portfolio, instrument=InstrumentId(p.symbol), quantity=p.quantity,
                average_cost=p.average_cost if p.average_cost > 0 else marks[InstrumentId(p.symbol)],
                as_of=decision_time,
            )
            for p in managed
        },
    )
    held_marks = {i: marks[i] for i in book.positions}
    equity = book.equity(held_marks)
    if equity <= 0:
        raise ContractViolation(f"the managed book is worth {equity:.2f}; nothing to size against")
    holdings = Holdings(
        book.weights(held_marks) if book.positions else {},
        gains={i: marks[i] / p.average_cost - 1.0 for i, p in book.positions.items()
               if p.average_cost > 0},
    )

    # -- what the strategy says --------------------------------------------------
    rotation = bool(strategy.rotates_at(decision_time))
    table = strategy.rank(filtration, holdings)
    limit = int(dict(definition.strategy.params).get("top_n", strategy.params.top_n))
    passing = table.loc[table["eligible"]] if not table.empty else table
    best = [str(i) for i in (passing.head(limit) if limit else passing)["instrument"]] \
        if not table.empty else []

    def constraints_for(instrument: InstrumentId) -> InstrumentConstraints:
        return InstrumentConstraints(instrument=instrument, currency="USD")

    decision = decide(
        run=RunId(f"review-{decision_bar.date().isoformat()}"), book=book, strategy=strategy,
        filtration=filtration, marks=marks, constraints_for=constraints_for,
        policy=definition.execution.sizing(interval, allow_short=definition.risk.allow_short),
    )
    target = {str(i): w for i, w in decision.target.weights.items()}

    # -- monitoring, before any order: the ladder decides what may be proposed ----
    working = connector.open_orders(payloads.orders or {})
    fills = connector.trades(payloads.trades or {})
    watch = _monitor(
        definition=definition, kept=kept, baseline=baseline, payloads=payloads,
        prices=prices, decision_bar=decision_bar, equity=equity, fills=fills,
        managed={p.symbol: p.quantity for p in managed}, closes=closes,
        cash=book.cash, as_of=as_of, notes=notes, unit=interval.noun,
    )
    ladder = DegradationState(watch["state"]["ladder"]["state"])

    # -- orders ------------------------------------------------------------------
    offset = definition.review.limit_offset
    orders: list[dict[str, Any]] = []
    held_back: list[dict[str, Any]] = []
    for intent in decision.intents:
        symbol = str(intent.instrument)
        price = float(marks[intent.instrument])
        through = 1.0 + offset if intent.side is Side.BUY else 1.0 - offset
        order = {
            "symbol": symbol, "side": intent.side.value.upper(), "quantity": intent.quantity,
            "type": "LIMIT", "time_in_force": "DAY",
            "limit": constraints_for(intent.instrument).round_price(price * through),
            "reference": _round(price, 4), "value": _round(intent.quantity * price, 2),
            "reason": intent.reason, "decision_price": _round(closes.get(symbol), 4),
        }
        adds = intent.reason in ("open", "increase")
        if ladder is DegradationState.HALTED or (ladder is DegradationState.REDUCE_ONLY and adds):
            held_back.append({**order, "held_back": f"the system is {ladder.value}"})
        else:
            orders.append(order)

    after = {str(i): p.quantity for i, p in book.positions.items()}
    for order in orders:
        sign = 1.0 if order["side"] == "BUY" else -1.0
        after[order["symbol"]] = after.get(order["symbol"], 0.0) + sign * order["quantity"]
    after = {s: q for s, q in after.items() if abs(q) > 1e-9}

    buys = sum(o["quantity"] * o["limit"] for o in orders if o["side"] == "BUY")
    sells = sum(o["quantity"] * o["limit"] for o in orders if o["side"] == "SELL")

    # -- stops -------------------------------------------------------------------
    checks = _stop_checks({p.symbol: p.quantity for p in managed}, working)
    stops: dict[str, Any] = {"replace": False, "cancel": [], "place": []}
    rule = definition.risk.stop()
    if rotation and rule is not None and ladder is not DegradationState.HALTED:
        stops["replace"] = True
        stops["cancel"] = [
            {"order_id": o.order_id, "symbol": o.symbol, "quantity": o.quantity,
             "stop": o.stop_price, "limit": o.limit_price}
            for o in working if o.is_stop and o.side == "SELL"
        ]
        risks = {
            InstrumentId(s): PositionRisk(
                instrument=InstrumentId(s), quantity=q, average_cost=0.0,
                mark=float(marks[InstrumentId(s)]), weight=0.0,
                anchor=float(marks[InstrumentId(s)]),
            )
            for s, q in after.items() if InstrumentId(s) in marks
        }
        for intent in rule.orders_for(
            risks, decision.run, portfolio, version, decision_time, constraints_for
        ):
            stops["place"].append({
                "symbol": str(intent.instrument), "side": intent.side.value.upper(),
                "quantity": intent.quantity, "type": "STP LMT" if intent.limit_price else "STP",
                "stop": intent.stop_price, "limit": intent.limit_price, "time_in_force": "GTC",
                "anchor": _round(float(marks[intent.instrument]), 4),
            })

    wanted = sorted({o["symbol"] for o in orders} | set(after)) if rotation else []
    quotes_needed = [s for s in wanted if s not in quotes] if (orders or stops["place"]) else []
    # A rotation chosen from part of the universe is a different rotation: the
    # stock that was not fetched may be the one that belonged among the best.
    history_needed = sorted({*missing, *stale}) if rotation else []
    final = not quotes_needed and not history_needed and not behind

    # -- the record of this rotation's proposals, for next week's execution check --
    new_state = watch["state"]
    if rotation:
        _record_rotation(new_state, decision_bar, as_of, orders, final=final)
    new_state["last_review"] = {
        "as_of": as_of.isoformat(), "decision_bar": decision_bar.date().isoformat(),
        "positions": {p.symbol: p.quantity for p in managed},
        "closes": {p.symbol: closes[p.symbol] for p in managed if p.symbol in closes},
        "cash": _round(book.cash, 2),
    }

    scan = _scan(table, best, target, holdings, rotation)
    upcoming = _next_rotation(strategy, decision_time)
    return {
        "format": 1,
        "as_of": as_of.isoformat(),
        "strategy_version": str(version),
        "definition": definition.source.name if definition.source else None,
        "week": "rebalance" if rotation else "monitoring",
        "decision_bar": decision_bar.date().isoformat(),
        "decision_time": decision_time.isoformat(),
        "next_rebalance": upcoming.isoformat() if upcoming else None,
        "this_rebalance": (decision_time + timedelta(days=3)).date().isoformat() if rotation else None,
        "final": final,
        "quotes_needed": quotes_needed,
        "history_needed": history_needed,
        "data_behind": expected.date().isoformat() if behind else None,
        # What to ask the connector for, for the symbols above: it takes contract ids.
        "contracts": {symbol: contract for contract, symbol in sorted(payloads.names.items())
                      if symbol in {*quotes_needed, *history_needed}},
        "data": {
            "universe": universe.describe(), "with_bars": len(current), "missing": missing,
            "stale": stale, "bars": int(min(len(p.traded) for p in current.values())),
            "dividends_applied": int(sum(p.payouts for p in current.values())),
            "quotes": sorted(quotes), "corporate_actions": actions, "short_history": short,
        },
        "account": {
            "net_liquidation": _round(connector.net_liquidation(payloads.account), 2),
            "cash": _round(book.cash, 2), "managed_equity": _round(equity, 2),
            "positions": [
                {"symbol": p.symbol, "quantity": p.quantity,
                 "price": _round(float(marks[InstrumentId(p.symbol)]), 4),
                 "value": _round(p.quantity * float(marks[InstrumentId(p.symbol)]), 2),
                 "weight": _round(holdings.get(InstrumentId(p.symbol), 0.0), 4),
                 "average_cost": _round(p.average_cost, 4)}
                for p in sorted(managed, key=lambda p: p.symbol)
            ],
            "outside_the_strategy": [
                {"symbol": p.symbol, "quantity": p.quantity, "asset_class": p.asset_class}
                for p in outside
            ],
        },
        "selection": {
            "passing": int(len(passing)), "limit": limit, "best": best,
            "frozen": sorted(r["symbol"] for r in scan if r["status"] == "frozen"),
            "exits": sorted(r["symbol"] for r in scan if r["status"].startswith("exit")),
            "out_of_time": sorted(r["symbol"] for r in scan if r["status"] == "frozen too long"),
        },
        "scan": scan,
        "target": {s: _round(w, 6) for s, w in sorted(target.items())} if rotation else None,
        "orders": orders,
        "orders_held_back": held_back,
        "cash_after": {
            "buys": _round(buys, 2), "sells": _round(sells, 2),
            "cash": _round(book.cash + sells - buys, 2),
        } if orders else None,
        "skipped": {str(i): why for i, why in decision.skipped.items()},
        "stops": stops,
        "stop_checks": checks,
        "monitoring": watch["report"],
        "performance": _performance(payloads, prices, definition, decision_bar),
        "notes": notes,
        "state": new_state,
    }


def _scan(table: pd.DataFrame, best: Sequence[str], target: Mapping[str, float],
          holdings: Holdings, rotation: bool) -> list[dict[str, Any]]:
    """Every stock the strategy looked at, with what it made of it."""
    rows: list[dict[str, Any]] = []
    if table.empty:
        return rows
    for position, row in enumerate(table.itertuples(), start=1):
        symbol = str(row.instrument)
        if row.held and isinstance(row.exit_reason, str):
            status = f"exit: {row.exit_reason}"
        elif symbol in best:
            status = "best"
        elif row.held and row.passed_recently:
            status = "frozen"
        elif row.held:
            status = "frozen too long"
        elif row.eligible:
            status = "passes"
        else:
            status = ""
        rows.append({
            "symbol": symbol, "order": position, "return_13w": _round(row.score, 4),
            "return_4w": _round(row.pace, 4), "atr_pct": _round(row.atr_pct, 4),
            "rsi": _round(row.rsi, 1), "passes": bool(row.eligible), "held": bool(row.held),
            "weight_now": _round(holdings.get(row.instrument, 0.0), 4) if row.held else None,
            "weight_target": _round(target.get(symbol, 0.0), 4) if rotation and (
                row.held or symbol in target) else None,
            "status": status,
        })
    return rows


def _stop_checks(held: Mapping[str, float], working: Sequence[connector.OpenOrder]) -> dict[str, Any]:
    """Whether every position has one resting stop for all of its shares."""
    stops = [o for o in working if o.is_stop and o.side == "SELL"]
    by_symbol: dict[str, list[connector.OpenOrder]] = {}
    for order in stops:
        by_symbol.setdefault(order.symbol, []).append(order)
    problems: list[str] = []
    table = []
    for symbol, quantity in sorted(held.items()):
        mine = by_symbol.get(symbol, [])
        covered = sum(o.quantity for o in mine)
        table.append({
            "symbol": symbol, "shares": quantity, "stop_shares": covered,
            "stop": mine[0].stop_price if len(mine) == 1 else None,
            "limit": mine[0].limit_price if len(mine) == 1 else None,
            "placed": mine[0].placed_at if len(mine) == 1 else None,
        })
        if not mine:
            problems.append(f"{symbol}: no stop for its {quantity:g} shares")
        elif len(mine) > 1:
            problems.append(f"{symbol}: {len(mine)} stops resting; there should be one")
        elif covered != quantity:
            problems.append(f"{symbol}: the stop is for {covered:g} shares, the position is {quantity:g}")
        if any(not o.good_till_cancelled for o in mine):
            problems.append(f"{symbol}: its stop is not good-till-cancelled")
    for symbol, orders in sorted(by_symbol.items()):
        if symbol not in held:
            label = symbol or "an order whose description could not be read"
            problems.append(
                f"{label}: {len(orders)} stop(s) resting with no position (order "
                f"{', '.join(o.order_id for o in orders)})"
            )
    return {"positions": table, "problems": problems, "ok": not problems}


# -- monitoring ------------------------------------------------------------------------------


def weekly_returns(growth: pd.Series, through: datetime) -> dict[str, float]:
    """Week-on-week returns of a daily growth index, keyed by the week's Friday.

    Only weeks that have ended by ``through`` are returned, and the first week
    of the index has no return: there is nothing before it to measure from.
    """
    if growth.empty:
        return {}
    by_week = growth.groupby([connector.week_close(d.date()) for d in growth.index]).last()
    by_week = by_week.loc[by_week.index <= through]
    changes = by_week.pct_change().dropna()
    return {week.date().isoformat(): float(value) for week, value in changes.items()}


def _first_live_week(monitor_from: str | None) -> date | None:
    """The Friday that closes the week of the first rotation on these rules."""
    if monitor_from is None:
        return None
    return connector.week_close(date.fromisoformat(monitor_from)).date()


def _record_rotation(state: dict[str, Any], decision_bar: pd.Timestamp, as_of: datetime,
                     orders: Sequence[Mapping[str, Any]], final: bool) -> None:
    key = decision_bar.date().isoformat()
    rotations = [r for r in state.get("rotations", []) if r.get("id") != key]
    previous = next((r for r in state.get("rotations", []) if r.get("id") == key), {})
    rotations.append({
        "id": key, "decision_time": (decision_bar.to_pydatetime() + timedelta(minutes=15)).isoformat(),
        "proposed_at": as_of.isoformat(), "final": final,
        "proposals": [
            {"symbol": o["symbol"], "side": o["side"], "quantity": o["quantity"],
             "decision_price": o["decision_price"] or o["reference"], "limit": o["limit"],
             "reason": o["reason"]}
            for o in orders
        ],
        "fills": previous.get("fills", []), "opens": previous.get("opens", {}),
    })
    state["rotations"] = sorted(rotations, key=lambda r: r["id"])[-KEEP_ROTATIONS:]


def _match_fills(state: dict[str, Any], fills: Sequence[connector.Trade],
                 prices: Mapping[str, Prices]) -> None:
    """Attach each rotation's fills, and the open the backtest would have used."""
    for rotation in state.get("rotations", []):
        decided = datetime.fromisoformat(rotation["decision_time"])
        wanted = {(p["symbol"], p["side"]) for p in rotation["proposals"]}
        seen = {f["trade_id"] for f in rotation.get("fills", [])}
        for trade in fills:
            if trade.asset_class != "STK" or trade.from_stop or trade.trade_id in seen:
                continue
            if not decided < trade.at <= decided + FILL_WINDOW:
                continue
            if (trade.symbol, trade.side) not in wanted:
                continue
            rotation.setdefault("fills", []).append({
                "trade_id": trade.trade_id, "symbol": trade.symbol, "side": trade.side,
                "quantity": trade.quantity, "price": trade.price,
                "commission": trade.commission, "at": trade.at.isoformat(),
            })
            seen.add(trade.trade_id)
        opens = rotation.setdefault("opens", {})
        for symbol in {p["symbol"] for p in rotation["proposals"]} - set(opens):
            frame = prices[symbol].traded if symbol in prices else None
            if frame is None:
                continue
            later = frame.loc[frame.index > decided]
            if not later.empty:
                opens[symbol] = float(later["open"].iloc[0])


def _records(state: Mapping[str, Any]) -> list[ExecutionRecord]:
    records = []
    for rotation in state.get("rotations", []):
        marks = {(p["symbol"], p["side"]): p["decision_price"] for p in rotation["proposals"]}
        for fill in rotation.get("fills", []):
            mark = marks.get((fill["symbol"], fill["side"]))
            if not mark:
                continue
            records.append(ExecutionRecord(
                rotation=rotation["id"], instrument=fill["symbol"], side=fill["side"].lower(),
                quantity=float(fill["quantity"]), decision_price=float(mark),
                fill_price=float(fill["price"]), commission=float(fill.get("commission", 0.0)),
                reference_open=rotation.get("opens", {}).get(fill["symbol"]),
            ))
    return records


def _compliance(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per rotation: what was proposed and how much of it traded."""
    out = []
    for rotation in state.get("rotations", [])[-6:]:
        filled: dict[tuple[str, str], float] = {}
        for fill in rotation.get("fills", []):
            key = (fill["symbol"], fill["side"])
            filled[key] = filled.get(key, 0.0) + float(fill["quantity"])
        lines = [
            {"symbol": p["symbol"], "side": p["side"], "proposed": p["quantity"],
             "filled": filled.get((p["symbol"], p["side"]), 0.0)}
            for p in rotation["proposals"]
        ]

        buys, sells = _done(lines, "BUY"), _done(lines, "SELL")
        out.append({
            "rotation": rotation["id"], "final": rotation.get("final", True),
            "buys_done": buys[0], "buys": buys[1],
            "sells_done": sells[0], "sells": sells[1], "orders": lines,
        })
    return out


def _done(lines: Sequence[Mapping[str, Any]], side: str) -> tuple[int, int]:
    mine = [line for line in lines if line["side"] == side]
    return sum(1 for line in mine if line["filled"] >= line["proposed"]), len(mine)


def _monitor(*, definition: Definition, kept: Mapping[str, Any], baseline, payloads: Payloads,
             prices: Mapping[str, Prices], decision_bar: pd.Timestamp, equity: float,
             fills: Sequence[connector.Trade], managed: Mapping[str, float],
             closes: Mapping[str, float], cash: float, as_of: datetime, notes: list[str],
             unit: str) -> dict[str, Any]:
    """The live record, the checks, and where the ladder stands after them."""
    state = json.loads(json.dumps(kept))  # a copy the caller's mapping does not share
    state.pop("previous_version", None)
    settings, review_settings = definition.monitoring, definition.review
    report: dict[str, Any] = {"live_from": review_settings.monitor_from}

    # The account's weekly returns, from the broker's time-weighted series.
    growth = (connector.performance(payloads.performance)
              if payloads.performance is not None else pd.Series(dtype="float64"))
    if payloads.performance is None:
        notes.append("no performance series was supplied; the live record was not extended")
    fresh = weekly_returns(growth, decision_bar.to_pydatetime())
    state["weekly_returns"] = {**state.get("weekly_returns", {}), **fresh}
    first = _first_live_week(review_settings.monitor_from)
    live = {week: value for week, value in sorted(state["weekly_returns"].items())
            if first is not None and date.fromisoformat(week) >= first}
    # Weeks before the live record are kept only as far back as the broker
    # serves them anyway; the document should not grow with history it never uses.
    state["weekly_returns"] = {
        week: value for week, value in sorted(state["weekly_returns"].items())
        if week in fresh or week in live
    }
    returns = list(live.values())
    report["weeks"] = len(returns)
    report["returns"] = {week: _round(value, 6) for week, value in live.items()}

    # A second opinion on the last week: what the positions held then did.
    last = state.get("last_review")
    report["last_week"] = None
    week_key = decision_bar.date().isoformat()
    if last and last.get("decision_bar") and last["decision_bar"] < week_key and week_key in fresh:
        before = sum(q * last["closes"].get(s, 0.0) for s, q in last["positions"].items())
        now = sum(q * closes.get(s, last["closes"].get(s, 0.0))
                  for s, q in last["positions"].items())
        base = before + float(last.get("cash") or 0.0)
        own = (now - before) / base if base > 0 else None
        weeks_apart = (date.fromisoformat(week_key) - date.fromisoformat(last["decision_bar"])).days // 7
        report["last_week"] = {
            "account": _round(fresh[week_key], 6), "positions_held_before": _round(own, 6),
            "weeks_since_last_review": weeks_apart,
        }
        traded = any(last["positions"].get(s) != q for s, q in managed.items()) or set(
            last["positions"]) != set(managed)
        if (own is not None and weeks_apart == 1 and not traded
                and abs(own - fresh[week_key]) > RETURN_TOLERANCE):
            notes.append(
                f"last week the account returned {fresh[week_key]:+.2%} by the broker's series "
                f"but the positions held moved {own:+.2%}, with no trade between: check for a "
                f"deposit, a dividend or a corporate action"
            )

    # Execution: each rotation's fills against its decision prices.
    _match_fills(state, fills, prices)
    records = _records(state)
    report["compliance"] = _compliance(state)

    ladder = dict(state.get("ladder") or empty_state("")["ladder"])
    current = DegradationState(ladder.get("state", "normal"))
    report["state_before"] = current.value

    if baseline is None:
        notes.append(
            "no published baseline for this strategy version (baselines/); the ladder was "
            "not judged. Build it with `ql monitor baseline` and `ql review baseline`."
        )
        report.update({"judged": False, "state": current.value, "reasons": [],
                       "ladder": ladder})
        state["ladder"] = ladder
        return {"state": state, "report": report}

    thresholds = Thresholds(
        reduce_percentile=settings.reduce_percentile, halt_percentile=settings.halt_percentile,
        reduce_break_probability=settings.reduce_break_probability,
        halt_break_probability=settings.halt_break_probability,
        reduce_shortfall_multiple=settings.reduce_shortfall_multiple,
        halt_shortfall_alpha_share=settings.halt_shortfall_alpha_share,
        halt_shortfall_cycles=settings.halt_shortfall_cycles,
    )

    def judge(live_returns: Sequence[float]):
        return assess(
            baseline.returns, live_returns, records,
            modeled_bps=baseline.modeled_bps,
            expected_rotation_return=baseline.expected_rotation_return,
            sleeve_equity=equity, thresholds=thresholds, paths=settings.bootstrap_paths,
            block=max(float(settings.bootstrap_block_weeks), 1.0),
            hazard_bars=max(float(settings.changepoint_hazard_weeks), 2.0),
            trend_min_bars=max(int(math.ceil(settings.trend_min_weeks)), 3),
            horizon=max(int(round(settings.horizon_weeks)), 1), unit=unit,
        )

    full = judge(returns)
    acting = full
    burn_in = len(returns) < review_settings.burn_in_weeks
    if burn_in:
        # Shown in full, acted on in part: with this few weeks only what does
        # not rest on the return series may move the ladder.
        acting = judge([])
    recommended, reasons = acting.recommended, list(acting.reasons)

    # The loss breaker: one week so bad the backtest almost never had one.
    breaker = None
    if returns:
        bar = list(live)[-1]
        threshold = float(np.quantile(baseline.returns, definition.automation.loss_breaker_percentile))
        if bar not in state.get("breaker_bars", []):
            state["breaker_bars"] = [*state.get("breaker_bars", []), bar][-KEEP_ROTATIONS:]
            if returns[-1] < threshold:
                breaker = (
                    f"loss breaker: the week to {bar} returned {returns[-1]:.1%}, below the "
                    f"backtest's {definition.automation.loss_breaker_percentile:.1%} quantile "
                    f"({threshold:.1%})"
                )
                recommended = DegradationState.HALTED
                reasons.append(f"halted: {breaker}")
        report["breaker"] = {"threshold": _round(threshold, 6), "last_week": _round(returns[-1], 6),
                             "fired": breaker is not None}

    # The ladder's own rules: a halt sticks; monitoring lifts only what it imposed.
    after = current
    why = "; ".join(reasons) or "monitoring found nothing outside its bands"
    paused_by_a_person = (current is DegradationState.REDUCE_ONLY
                          and ladder.get("by") not in ("monitor", "system"))
    if recommended is DegradationState.HALTED:
        after = DegradationState.HALTED
    elif current is DegradationState.HALTED or paused_by_a_person:
        after = current
    else:
        after = recommended
    if after is not current:
        ladder = {"state": after.value, "by": "monitor", "reason": why,
                  "since": as_of.isoformat()}
    state["ladder"] = ladder

    report.update({
        "judged": True, "state": after.value, "reasons": reasons,
        "burn_in": burn_in, "burn_in_weeks": review_settings.burn_in_weeks,
        "shown_but_not_acted_on": list(full.reasons) if burn_in else [],
        "notes": list(full.notes), "ladder": ladder,
        "baseline": {"version": baseline.strategy_version, "built_at": baseline.built_at,
                     "modeled_bps": _round(baseline.modeled_bps, 2),
                     "expected_rotation_return": _round(baseline.expected_rotation_return, 6)},
        "drawdown": _plain(full.drawdown), "break_probability": _round(full.break_probability, 4),
        "trend": _plain(full.trend), "shortfall": _plain(full.shortfall),
    })
    return {"state": state, "report": report}


def _plain(value: Any) -> Any:
    if value is None:
        return None
    raw = asdict(value)
    return {k: (_round(v, 6) if isinstance(v, float) else
                {a: _round(b, 4) for a, b in v.items()} if isinstance(v, Mapping) else v)
            for k, v in raw.items()}


# -- performance -----------------------------------------------------------------------------


def _performance(payloads: Payloads, prices: Mapping[str, Prices], definition: Definition,
                 decision_bar: pd.Timestamp) -> dict[str, Any]:
    """The account beside the benchmark over the same weeks, to the decision bar."""
    growth = (connector.performance(payloads.performance)
              if payloads.performance is not None else pd.Series(dtype="float64"))
    through = decision_bar.to_pydatetime()
    out: dict[str, Any] = {"through": decision_bar.date().isoformat(),
                           "benchmark": definition.review.benchmark, "windows": {}}
    if growth.empty:
        return out
    account = growth.groupby([connector.week_close(d.date()) for d in growth.index]).last()
    account = account.loc[account.index <= through]
    bench_prices = prices.get(definition.review.benchmark)
    bench = bench_prices.total_return["close"] if bench_prices is not None else None

    def since(start: datetime | None) -> tuple[float | None, float | None]:
        if start is None or account.empty:
            return None, None
        base = account.loc[account.index <= start]
        mine = float(account.iloc[-1] / base.iloc[-1] - 1.0) if not base.empty else None
        theirs = None
        if bench is not None:
            before = bench.loc[bench.index <= start]
            upto = bench.loc[bench.index <= through]
            if not before.empty and not upto.empty:
                theirs = float(upto.iloc[-1] / before.iloc[-1] - 1.0)
        return mine, theirs

    # Whole weeks only: the benchmark is known by its weekly closes here, and a
    # window that ends mid-week for one and on a Friday for the other compares
    # two different stretches of time.
    windows: dict[str, datetime | None] = {"last_week": through - timedelta(days=7)}
    if definition.review.performance_from:
        windows["since_" + definition.review.performance_from] = connector.last_closed_week(
            datetime.combine(date.fromisoformat(definition.review.performance_from),
                             datetime.max.time(), tzinfo=timezone.utc))
    first = _first_live_week(definition.review.monitor_from)
    if first is not None and connector.week_close(first) <= through:
        windows["live_record"] = connector.week_close(first) - timedelta(days=7)
    for name, start in windows.items():
        mine, theirs = since(start)
        out["windows"][name] = {
            "account": _round(mine, 6), "benchmark": _round(theirs, 6),
            "difference": _round(mine - theirs, 6) if mine is not None and theirs is not None else None,
        }
    return out


# -- a page of text --------------------------------------------------------------------------


def summary(result: Mapping[str, Any]) -> str:
    """The review in a screen of plain text, for the session that ran it."""
    lines: list[str] = []
    add = lines.append
    add(f"{result['week'].upper()} WEEK — decided on the close of {result['decision_bar']} "
        f"({result['strategy_version']})")
    add(f"next rebalance: {result['next_rebalance']}"
        + (f"; this one is {result['this_rebalance']}" if result["this_rebalance"] else ""))
    if result.get("data_behind"):
        add(f"NOT FINAL — no history reaches the week that ended {result['data_behind']}: what "
            f"follows is last week's picture. Fetch the weekly histories again, then run the "
            f"review again")
    data = result["data"]
    add(f"data: {data['with_bars']} stocks with bars (at least {data['bars']} weeks), "
        f"{data['dividends_applied']} dividends applied"
        + (f"; missing: {', '.join(data['missing'])}" if data["missing"] else "")
        + (f"; stale: {', '.join(data['stale'])}" if data["stale"] else ""))
    account = result["account"]
    add(f"account: net liquidation {account['net_liquidation']:,.2f}, cash {account['cash']:,.2f}, "
        f"{len(account['positions'])} positions")
    sel = result["selection"]
    add(f"selection: {sel['passing']} pass; best {sel['limit'] or 'all'}: {', '.join(sel['best']) or 'none'}")
    add(f"  frozen: {', '.join(sel['frozen']) or 'none'}")
    if sel["exits"]:
        add(f"  exits triggered: {', '.join(sel['exits'])}")
    if sel["out_of_time"]:
        add(f"  frozen more than its rotations: {', '.join(sel['out_of_time'])}")
    watch = result["monitoring"]
    add(f"monitoring: {watch['state']}" + (f" (was {watch['state_before']})"
                                             if watch["state"] != watch["state_before"] else "")
        + f"; {watch['weeks']} live weeks since {watch['live_from']}"
        + ("" if watch.get("judged") else "; NOT JUDGED (no baseline)"))
    for reason in watch.get("reasons", []):
        add(f"  {reason}")
    for reason in watch.get("shown_but_not_acted_on", []):
        add(f"  shown, not acted on before week {watch['burn_in_weeks']}: {reason}")
    checks = result["stop_checks"]
    add("stops: " + ("every position has one stop for all its shares" if checks["ok"]
                     else f"{len(checks['problems'])} problem(s)"))
    for problem in checks["problems"]:
        add(f"  {problem}")
    if result["week"] == "rebalance":
        contracts = result.get("contracts", {})

        def named(symbols: list[str]) -> str:
            return ", ".join(f"{s} ({contracts[s]})" if s in contracts else s for s in symbols)

        if result["history_needed"]:
            add("NOT FINAL — no current weekly history for these; fetch it, then run the "
                "review again: " + named(result["history_needed"]))
        if result["quotes_needed"]:
            add("NOT FINAL — quote these, then run the review again: "
                + named(result["quotes_needed"]))
        add(f"orders ({len(result['orders'])}):")
        for order in result["orders"]:
            add(f"  {order['side']:4} {order['quantity']:g} {order['symbol']} LIMIT "
                f"{order['limit']:.2f} DAY  ({order['reason']}, about {order['value']:,.0f})")
        for order in result["orders_held_back"]:
            add(f"  held back: {order['side']} {order['quantity']:g} {order['symbol']} — "
                f"{order['held_back']}")
        if result["cash_after"]:
            add(f"  cash after: {result['cash_after']['cash']:,.2f}")
        stops = result["stops"]
        if stops["replace"]:
            add(f"stops to cancel ({len(stops['cancel'])}) and place ({len(stops['place'])}):")
            for stop in stops["place"]:
                add(f"  SELL {stop['quantity']:g} {stop['symbol']} STP {stop['stop']:.2f}"
                    + (f" LMT {stop['limit']:.2f}" if stop["limit"] else "") + " GTC")
    else:
        add("no orders this week; stops stay where they are")
    perf = result["performance"]["windows"]
    for name, row in perf.items():
        if row["account"] is not None:
            bench = f"{row['benchmark']:+.2%}" if row["benchmark"] is not None else "n/a"
            add(f"performance, {name.replace('_', ' ')}: account {row['account']:+.2%}, "
                f"{result['performance']['benchmark']} {bench}")
    for note in result["notes"]:
        add(f"note: {note}")
    return "\n".join(lines)
