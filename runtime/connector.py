"""Reading what the IBKR connector returns, and finding it in a session's log.

The weekly review (``runtime.review``) runs where there is no gateway: in a
session that reaches the broker through the IBKR connector, whose answers are
JSON documents. This module turns those documents into the frames and records
the rest of the system works with, and nothing else. No rule lives here.

Two things about the connector's weekly bars are easy to get wrong:

**They are adjusted for splits, not for dividends.** The store the strategy was
validated on holds prices adjusted for both (``data.adjustments``), and the
project's expert is explicit that signals belong on the total-return series in
research and in production alike: an ex-dividend drop is an accounting step,
not a loss. The connector lists the cash dividends in its window, so the
adjustment is rebuilt here: every bar before the week a dividend went ex is
scaled by ``1 - dividend / previous week's close``. Checked against the store
on the connector's answers of 2026-10-09, the rebuilt closes of all 53 stocks
agree within 0.12% over two years; left unadjusted they are up to 11.6% apart.

**The first and last bars are not weeks.** The window opens on its first day,
whatever weekday that is, and closes with a bar for the day of the request.
Bars are therefore gathered by calendar week, the window's opening week is
dropped, and a week counts only once it has ended.

A session's log is where the payloads are. A session cannot hand a tool's
result to a program except by typing it out again, which costs the result
twice and invites a wrong digit in a price. The log already holds every result
verbatim, so :class:`SessionLog` reads them from there.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from contracts.errors import ContractViolation

#: Weekly bars close with the US session on Friday. The store stamps them at
#: 21:00 UTC, and the strategy's rotation calendar counts from that stamp.
WEEK_CLOSE_HOUR = 21
PRICE_FIELDS = ("open", "high", "low", "close")

#: Where Claude Code keeps session logs.
LOG_ROOT = Path.home() / ".claude" / "projects"
#: The name the review's state is kept under between sessions. A session that
#: reads a document whose path ends with this has handed the review its state.
STATE_DOCUMENT = "review-state.json"


# -- weekly bars -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dividend:
    ex_date: date
    amount: float
    currency: str = "USD"


def week_close(day: date) -> datetime:
    """The Friday 21:00 UTC that closes the calendar week ``day`` is in."""
    monday = day - timedelta(days=day.weekday())
    friday = monday + timedelta(days=4)
    return datetime(friday.year, friday.month, friday.day, WEEK_CLOSE_HOUR, tzinfo=timezone.utc)


def last_closed_week(as_of: datetime) -> datetime:
    """The most recent week close that is not after ``as_of``."""
    close = week_close(as_of.date())
    while close > as_of:
        close -= timedelta(days=7)
    return close


def _day(stamp: str) -> date:
    return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).date()


def weekly_bars(payload: Mapping[str, Any], as_of: datetime) -> pd.DataFrame:
    """Closed weekly bars from a ``get_price_history`` answer, as traded.

    Indexed by each week's close (Friday 21:00 UTC), oldest first, with open,
    high, low, close and volume. Split-adjusted, as IBKR serves them; not
    adjusted for dividends (see :func:`adjusted`).

    Raises:
        ContractViolation: If the answer is not a weekly price history.
    """
    if payload.get("chart_step") not in (None, 604800):
        raise ContractViolation(
            f"expected weekly bars (a step of 604800 seconds); got {payload.get('chart_step')}"
        )
    times = payload.get("time")
    if not times or any(name not in payload for name in PRICE_FIELDS):
        raise ContractViolation("the price history has no bars")
    raw = pd.DataFrame({
        "week": [week_close(_day(t)) for t in times],
        **{name: [float(v) for v in payload[name]] for name in PRICE_FIELDS},
        "volume": [float(v) for v in payload.get("volume") or [0.0] * len(times)],
    })
    # The window's first bar starts on the window's first day, so it is a part
    # of a week: its high and low are not the week's.
    raw = raw.loc[raw["week"] != raw["week"].iloc[0]]
    weeks = raw.groupby("week", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    weeks.index.name = None
    return weeks.loc[weeks.index <= as_of]


def dividends(payload: Mapping[str, Any]) -> tuple[Dividend, ...]:
    """The cash dividends a ``get_price_history`` answer lists, oldest first."""
    found = []
    for action in payload.get("corp_actions") or ():
        if action.get("type") != "CashDividends":
            continue
        stamp = str(action["date"])
        found.append(Dividend(
            ex_date=date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])),
            amount=float(action["value"]), currency=str(action.get("currency", "USD")),
        ))
    return tuple(sorted(found, key=lambda d: d.ex_date))


def other_actions(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Corporate actions that are not cash dividends, for a person to look at.

    The bars are already adjusted for splits. What is not known is whether a
    dividend paid *before* a split is listed per old share or per new one, so a
    split with earlier dividends in the window is worth comparing with the store.
    """
    return tuple(
        f"{a.get('type')} {a.get('value', '')} on {a.get('date')}".replace("  ", " ")
        for a in payload.get("corp_actions") or () if a.get("type") != "CashDividends"
    )


def adjusted(bars: pd.DataFrame, payouts: tuple[Dividend, ...]) -> pd.DataFrame:
    """``bars`` on the total-return scale: every dividend taken out of the past.

    A dividend that went ex in some week scales every *earlier* week by
    ``1 - amount / close of the week before``. The ex-week itself and
    everything after are left as traded, so the last bar is always the price
    on the screen. Volume is left alone.
    """
    if bars.empty or not payouts:
        return bars.copy()
    scale = pd.Series(1.0, index=bars.index)
    for payout in payouts:
        ex_week = week_close(payout.ex_date)
        earlier = bars.index < ex_week
        if not earlier.any() or ex_week > bars.index[-1]:
            continue
        before = float(bars.loc[earlier, "close"].iloc[-1])
        if before <= 0 or payout.amount <= 0 or payout.amount >= before:
            continue
        scale[earlier] *= 1.0 - payout.amount / before
    out = bars.copy()
    for name in PRICE_FIELDS:
        out[name] = out[name] * scale
    return out


# -- the account -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    contract_id: int
    quantity: float
    average_cost: float
    market_price: float
    currency: str
    asset_class: str


@dataclass(frozen=True, slots=True)
class OpenOrder:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    stop_price: float | None
    limit_price: float | None
    good_till_cancelled: bool
    status: str
    placed_at: str

    @property
    def is_stop(self) -> bool:
        return self.order_type.upper().startswith("STOP") or self.order_type.upper() == "STP"


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    at: datetime
    order_type: str
    asset_class: str

    @property
    def from_stop(self) -> bool:
        return "STOP" in self.order_type.upper()


def positions(payload: Mapping[str, Any]) -> tuple[Position, ...]:
    return tuple(
        Position(
            symbol=str(p["contract_description"]).split()[0].upper(),
            contract_id=int(p["contract_id"]), quantity=float(p["position"]),
            average_cost=float(p.get("average_price") or 0.0),
            market_price=float(p.get("market_price") or 0.0),
            currency=str(p.get("currency", "USD")), asset_class=str(p.get("asset_class", "STK")),
        )
        for p in payload.get("positions") or () if float(p.get("position") or 0.0) != 0.0
    )


_STOP_AT = re.compile(r"\bSTP\s+([0-9][0-9.,]*)")
_SYMBOL_IN = re.compile(r"^(?:Buy|Sell)\s+[0-9.,]+\s+(\S+)", re.IGNORECASE)
_DONE = frozenset({"FILLED", "CANCELLED", "CANCELED", "INACTIVE", "REJECTED", "EXPIRED"})


def open_orders(payload: Mapping[str, Any]) -> tuple[OpenOrder, ...]:
    """Working orders from a ``get_account_orders`` answer.

    The connector does not give a stop order's trigger as a field; it is in the
    order's description (``STP 356.91 LMT 355.13, GTC``), and so is the symbol
    (``Sell 2 PANW``). An order whose description cannot be read is kept, with
    an empty symbol, so that it is reported rather than lost.
    """
    found = []
    for order in payload.get("orders") or ():
        status = str(order.get("order_status", "")).upper()
        if status in _DONE:
            continue
        first, second = str(order.get("primary_description", "")), str(
            order.get("secondary_description", ""))
        symbol = _SYMBOL_IN.match(first)
        stop = _STOP_AT.search(second)
        limit = order.get("limit_price")
        found.append(OpenOrder(
            order_id=str(order.get("order_id", "")),
            symbol=symbol.group(1).upper() if symbol else "",
            side=str(order.get("side", "")).upper(),
            order_type=str(order.get("order_type", "")).upper(),
            quantity=float(order.get("remaining_shares_qty") or order.get("total_shares_qty") or 0),
            stop_price=float(stop.group(1).replace(",", "")) if stop else None,
            limit_price=float(limit) if limit not in (None, "") else None,
            good_till_cancelled="GTC" in second.upper(),
            status=status, placed_at=str(order.get("order_time", "")),
        ))
    return tuple(found)


def trades(payload: Mapping[str, Any]) -> tuple[Trade, ...]:
    return tuple(
        Trade(
            trade_id=str(t["trade_id"]), order_id=str(t.get("order_id", "")),
            symbol=str(t["symbol"]).upper(), side=str(t["side"]).upper(),
            quantity=float(t["size"]), price=float(t["price"]),
            commission=float(t.get("commission") or 0.0),
            at=datetime.fromisoformat(str(t["trade_time"]).replace("Z", "+00:00")),
            order_type=str(t.get("order_type", "")), asset_class=str(t.get("sec_type", "STK")),
        )
        for t in payload.get("trades") or ()
    )


def net_liquidation(payload: Mapping[str, Any]) -> float:
    return float(payload["net_liquidation"])


def cash(payload: Mapping[str, Any]) -> float:
    return float(payload["total_cash_value"])


def performance(payload: Mapping[str, Any]) -> pd.Series:
    """The account's cumulative return by day, from the longest window served.

    ``1 + cps``: a growth index that starts near one, dated by session. The
    connector says whether it is time-weighted; a money-weighted series is
    refused, because a deposit would then read as a gain.
    """
    if str(payload.get("portfolio_measure", "TWR")).upper() != "TWR":
        raise ContractViolation(
            "the account reports money-weighted returns; monitoring needs time-weighted ones"
        )
    best: Mapping[str, Any] | None = None
    for account in (payload.get("accounts") or {}).values():
        for window in (account.get("periods") or {}).values():
            if best is None or len(window.get("dates") or ()) > len(best.get("dates") or ()):
                best = window
    if not best or not best.get("dates"):
        return pd.Series(dtype="float64")
    days = [date(int(d[:4]), int(d[4:6]), int(d[6:8])) for d in best["dates"]]
    return pd.Series([1.0 + float(c) for c in best["cps"]], index=pd.to_datetime(days)).sort_index()


def quote(payload: Mapping[str, Any]) -> float | None:
    """A price to act on from a ``get_price_snapshot`` answer: last, else the midpoint."""
    last = (payload.get("last") or {}).get("price")
    if last:
        return float(last)
    book = payload.get("bid-ask") or payload.get("bid_ask") or {}
    bid, ask = book.get("bid"), book.get("ask")
    if bid and ask:
        return (float(bid) + float(ask)) / 2.0
    prior = payload.get("prior-close") or payload.get("prior_close") or {}
    value = prior.get("price") if isinstance(prior, Mapping) else prior
    return float(value) if value else None


def watchlist(payload: Mapping[str, Any]) -> dict[int, str]:
    """Contract id to ticker, from a ``get_watchlist`` answer."""
    return {
        int(row["contract_id_ex"]): str(row["contract_description"]).split()[0].upper()
        for row in payload.get("instruments") or ()
        if str(row.get("contract_id_ex", "")).isdigit()
    }


# -- a session's log -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Call:
    tool: str
    arguments: Mapping[str, Any]
    result: Any
    #: When the answer was logged (ISO 8601, so it sorts as text). Empty if the
    #: log does not say.
    at: str = ""


def latest_log(root: Path = LOG_ROOT) -> Path:
    """The session log written to most recently: the running session's own."""
    logs = sorted(root.glob("*/*.jsonl"), key=os.path.getmtime)
    if not logs:
        raise ContractViolation(f"no session log under {root}")
    return logs[-1]


def session_logs(log: Path) -> list[Path]:
    """A session's log and those of the helpers it started.

    A session may hand the fetching to a helper so that the answers do not fill
    its own context; the helper's log sits in a folder named after the session,
    and holds the answers just the same.
    """
    log = Path(log)
    return [log, *sorted((log.with_suffix("") / "subagents").glob("*.jsonl"))]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, Mapping) and part.get("type", "text") == "text"
        )
    return ""


def calls(log: Path) -> Iterator[Call]:
    """Every tool call in a session log that returned JSON, in the order made."""
    pending: dict[str, tuple[str, Mapping[str, Any]]] = {}
    with Path(log).open(encoding="utf-8") as lines:
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            content = (entry.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_use":
                    pending[str(block.get("id"))] = (
                        str(block.get("name", "")), block.get("input") or {})
                elif block.get("type") == "tool_result":
                    made = pending.pop(str(block.get("tool_use_id")), None)
                    if made is None or block.get("is_error"):
                        continue
                    try:
                        result = json.loads(_text(block.get("content")))
                    except ValueError:
                        continue
                    yield Call(tool=made[0], arguments=made[1], result=result,
                               at=str(entry.get("timestamp") or ""))


@dataclass(slots=True)
class Payloads:
    """The connector's answers a review needs, as they were given.

    The newest answer to each question wins, so calling a tool again -- for a
    fresher quote, or after a failed call -- replaces what was there.
    """

    history: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    quotes: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    account: Mapping[str, Any] | None = None
    positions: Mapping[str, Any] | None = None
    orders: Mapping[str, Any] | None = None
    trades: Mapping[str, Any] | None = None
    performance: Mapping[str, Any] | None = None
    #: The state document the session read back (``STATE_DOCUMENT``), as text.
    state: str | None = None
    #: Contract id to ticker, as far as the session's answers named them.
    names: dict[int, str] = field(default_factory=dict)
    unnamed: list[int] = field(default_factory=list)
    #: Weekly histories left out because they were asked for without corporate
    #: actions: such an answer looks exactly like a stock that pays no dividend.
    ignored: list[str] = field(default_factory=list)

    def over(self, older: Payloads) -> Payloads:
        """These answers laid over an earlier collection: the newer one wins.

        A long session's log is cut back when its context is compacted, and the
        answers made before that go with it. Collecting after each batch of
        calls, and laying each collection over the last, keeps them.
        """
        merged = Payloads(
            history={**older.history, **self.history},
            quotes={**older.quotes, **self.quotes},
            state=self.state if self.state is not None else older.state,
            names={**older.names, **self.names},
        )
        for name in ("account", "positions", "orders", "trades", "performance"):
            mine = getattr(self, name)
            setattr(merged, name, mine if mine is not None else getattr(older, name))
        merged.unnamed = [c for c in self.unnamed if c not in merged.names]
        merged.ignored = [symbol for symbol in self.ignored if symbol not in merged.history]
        return merged

    def save(self, folder: Path) -> None:
        """One file per answer, so a run can be repeated from the folder alone."""
        folder = Path(folder)
        (folder / "history").mkdir(parents=True, exist_ok=True)
        (folder / "quotes").mkdir(parents=True, exist_ok=True)
        for symbol, payload in self.history.items():
            (folder / "history" / f"{symbol}.json").write_text(json.dumps(payload))
        for symbol, payload in self.quotes.items():
            (folder / "quotes" / f"{symbol}.json").write_text(json.dumps(payload))
        for name in ("account", "positions", "orders", "trades", "performance"):
            payload = getattr(self, name)
            if payload is not None:
                (folder / f"{name}.json").write_text(json.dumps(payload))
        if self.state is not None:
            (folder / STATE_DOCUMENT).write_text(self.state)
        if self.names:
            (folder / "names.json").write_text(
                json.dumps({str(contract): symbol for contract, symbol in self.names.items()}))

    @classmethod
    def load(cls, folder: Path) -> Payloads:
        folder = Path(folder)
        if not folder.is_dir():
            raise ContractViolation(f"no folder of connector answers at {folder}")
        found = cls()
        for path in sorted((folder / "history").glob("*.json")):
            found.history[path.stem.upper()] = json.loads(path.read_text())
        for path in sorted((folder / "quotes").glob("*.json")):
            found.quotes[path.stem.upper()] = json.loads(path.read_text())
        for name in ("account", "positions", "orders", "trades", "performance"):
            path = folder / f"{name}.json"
            if path.exists():
                setattr(found, name, json.loads(path.read_text()))
        if (folder / STATE_DOCUMENT).exists():
            found.state = (folder / STATE_DOCUMENT).read_text()
        if (folder / "names.json").exists():
            found.names = {int(contract): str(symbol) for contract, symbol
                           in json.loads((folder / "names.json").read_text()).items()}
        return found


def gather(log: Path, tickers: Mapping[int, str] | None = None) -> Payloads:
    """Collect a review's inputs from a session log.

    Reads the log and the logs of any helpers the session started
    (:func:`session_logs`). Price histories and quotes are asked for by contract
    id, so a ticker has to come from somewhere: the session's own ``get_watchlist`` and
    ``get_account_positions`` answers, and ``tickers`` for anything else. A
    weekly history whose contract cannot be named is listed in ``unnamed``.
    """
    names: dict[int, str] = dict(tickers or {})
    # By the time each answer was logged, across the session and its helpers:
    # "the newest answer wins" has to mean newest, not last file read. The sort
    # is stable, so answers a log does not date keep the order they were made in.
    made = sorted((call for path in session_logs(log) for call in calls(path)),
                  key=lambda call: call.at)
    for call in made:
        if call.tool.endswith("get_watchlist"):
            names.update(watchlist(call.result))
        elif call.tool.endswith("get_account_positions"):
            names.update({p.contract_id: p.symbol for p in positions(call.result)})

    found = Payloads(names=names)
    for call in made:
        tool = call.tool.rsplit("__", 1)[-1]
        if tool == "get_price_history":
            if call.arguments.get("step") != "ONE_WEEK" or "time" not in call.result:
                continue
            contract = int(call.arguments.get("contract_id", 0))
            symbol = names.get(contract)
            if symbol is None:
                if contract not in found.unnamed:
                    found.unnamed.append(contract)
                continue
            if call.arguments.get("include_corporate_actions") not in (True, "true"):
                # Without them the dividends are not in the answer, and the
                # signals would be computed on a price series, not a return one.
                if symbol not in found.history and symbol not in found.ignored:
                    found.ignored.append(symbol)
                continue
            found.history[symbol] = call.result
            if symbol in found.ignored:
                found.ignored.remove(symbol)
        elif tool == "get_price_snapshot":
            symbol = names.get(int(call.arguments.get("contract_id", 0)))
            if symbol is not None:
                found.quotes[symbol] = call.result
        elif tool == "get_account_summary":
            found.account = call.result
        elif tool == "get_account_positions":
            found.positions = call.result
        elif tool == "get_account_orders":
            found.orders = call.result
        elif tool == "get_account_trades":
            found.trades = call.result
        elif tool == "get_pa_performance_all_periods":
            found.performance = call.result
        elif str(call.arguments.get("path", "")).endswith(STATE_DOCUMENT) and str(
                call.arguments.get("method", "")).endswith("read"):
            found.state = _document(call.result)
    return found


def _document(result: Any) -> str | None:
    """A document a session read: its text, wherever the tool put it.

    A small document comes back inline; a large one is written to a file and
    the answer names the file.
    """
    if not isinstance(result, Mapping):
        return None
    content = result.get("content")
    if isinstance(content, str) and content.strip():
        return content
    for value in result.values():
        # A path, whatever the tool named the file: short, one line, and there.
        if (isinstance(value, str) and 0 < len(value) < 1024 and "\n" not in value
                and value.startswith(("/", "~")) and Path(value).expanduser().is_file()):
            return Path(value).expanduser().read_text(encoding="utf-8")
    return None
