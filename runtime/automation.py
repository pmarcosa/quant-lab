"""Running the live cycle without a person, inside limits a person set.

Nothing here is on by default. Two things must both be true before an order
leaves without a typed approval:

1. the strategy's config allows it -- ``automation.mode`` is ``exits`` or
   ``full``, the most the system may do alone; and
2. a person has **armed** it, typing a phrase (``ql live auto arm``). A halt
   disarms it, and nothing re-arms it but a person.

What the project's expert requires of an automatic mode (consulted 2026-09-23),
and how each is met here:

* **Pre-flight gates on every automatic send.** Data quality and freshness
  (the proposal's own checks: data age, newer-bar staleness); state integrity
  and zero reconciliation mismatch (a clean reconciliation within the hour);
  the degradation ladder (halted sends nothing; reduce-only sends only exits);
  margin (IBKR's cushion above ``automation.min_cushion`` for anything that
  adds exposure). The expert's fifth gate, signal robustness under worst-case
  input perturbation, applies to models with fitted inputs; this system's
  robustness to noise is tested offline by the funnel's jitter gate instead.
* **Hard limits per cycle.** Order size, cycle turnover, a loss circuit
  breaker, and an order count. Three are adapted rather than copied: the
  expert's 5% order cap and 30-40% turnover cap describe a strategy that slices
  orders and rarely rotates wholesale, and a concentrated momentum book opens
  25-50% positions and replaces most of itself at a rotation. So all three are
  the backtest's own extremes: an order 10% larger than any the backtest
  sent, or a rotation busier than 99% of the backtest's (and than one whole
  book at the gross cap), is held for a person; a bar worse than the
  backtest's 0.5% quantile halts. The expert's 4%-a-day loss limit, applied to weekly bars,
  would fire in ordinary weeks.
* **Staging.** Exits first; everything only on evidence. ``full`` in live
  money requires ``graduation_weeks`` (8) armed for exits, no reconciliation
  mismatch, and exit shortfall within 1.2x the modelled cost -- from the live
  or the paper journal of the same strategy. The expert's third criterion, a
  calibrated Brier score, applies to models that emit probabilities; this one
  does not. A person may override with a written reason, which is recorded.
* **Failure.** A failed step ends the cycle with nothing sent; the next run
  starts from a fresh sync. Rejected orders are not retried -- the next
  proposal is computed from the book as it then is. Never automatic: lifting
  a halt, re-arming, trading on an unreconciled book, changing a limit.

**One adaptation on execution order.** The expert sends exits, waits for their
fills, then sends entries. Opening-auction orders all execute in the same
auction, so waiting would move the entries off the price the backtest assumes;
they are sent together, exits first, and the margin check covers the funding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from contracts.errors import ContractViolation, QuantLabError
from contracts.execution import split_legs
from contracts.live import DegradationState, EventKind, TradingMode
from runtime.config import AUTOMATION_MODES
from runtime.journal import Journal, intent_from_dict
from runtime.live import LiveSession, SyncReport, closing_part

SCOPES = ("exits", "full")

#: A reconciliation older than this does not vouch for the book an automatic
#: send is computed on.
RECONCILIATION_MAX_AGE = timedelta(minutes=60)


def arm_phrase(session: LiveSession, scope: str) -> str:
    """What a person types to arm. Longer for live money, as for approvals."""
    phrase = f"AUTO {scope.upper()} {session.config.strategy_id}"
    return f"LIVE {phrase}" if session.config.mode is TradingMode.LIVE else phrase


# -- graduation ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Graduation:
    """The evidence for moving from automatic exits to automatic everything."""

    source: str
    weeks_exits: float
    mismatches: int
    exit_fills: int
    shortfall_ratio: float | None
    required_weeks: float
    required_ratio: float

    @property
    def reasons(self) -> tuple[str, ...]:
        out = []
        if self.weeks_exits < self.required_weeks:
            out.append(
                f"{self.weeks_exits:.1f} weeks armed for exits; {self.required_weeks:g} required"
            )
        if self.mismatches:
            out.append(f"{self.mismatches} reconciliation mismatch(es) since exits were armed")
        if self.exit_fills == 0:
            out.append("no automatic exit has filled yet")
        elif self.shortfall_ratio is not None and self.shortfall_ratio > self.required_ratio:
            out.append(
                f"exit shortfall is {self.shortfall_ratio:.2f}x the modelled cost; "
                f"at most {self.required_ratio:g}x"
            )
        return tuple(out)

    @property
    def ok(self) -> bool:
        return not self.reasons


def graduation(session: LiveSession) -> Graduation:
    """Evidence from this journal, or the paper journal of the same strategy."""
    journals = [(session.config.mode.value, session.journal)]
    if session.config.mode is TradingMode.LIVE:
        paper = session.config.live_dir / "paper-journal.jsonl"
        if paper.exists():
            journals.append(("paper", Journal(paper)))
    found = [_evidence(session, name, journal) for name, journal in journals]
    return max(found, key=lambda g: (g.ok, g.weeks_exits))


def _evidence(session: LiveSession, name: str, journal: Journal) -> Graduation:
    a = session.config.automation
    now = session.clock()
    weeks = 0.0
    armed_since = first_armed = None
    for event in journal.events(EventKind.AUTOMATION):
        action = event.payload.get("action")
        if action == "arm" and event.payload.get("scope") == "exits":
            armed_since = event.at
            first_armed = first_armed or event.at
        elif action in ("arm", "disarm") and armed_since is not None:
            weeks += (event.at - armed_since).total_seconds() / 604_800
            armed_since = None
    if armed_since is not None:
        weeks += (now - armed_since).total_seconds() / 604_800
    mismatches = 0
    if first_armed is not None:
        mismatches = sum(
            1 for e in journal.events(EventKind.RECONCILIATION)
            if e.at >= first_armed and e.payload.get("status") == "mismatch"
        )
    # Shortfall of automatic exits against the price at decision time.
    auto_orders = {
        oid for e in journal.events(EventKind.APPROVAL) if e.payload.get("by") == "automation"
        for oid in e.payload.get("orders", ())
    }
    proposals = {e.payload["proposal_id"]: e.payload for e in journal.events(EventKind.PROPOSAL)}
    order_proposal = {
        oid: e.payload["proposal_id"] for e in journal.events(EventKind.APPROVAL)
        for oid in e.payload.get("orders", ())
    }
    costs, notional, fills = 0.0, 0.0, 0
    for event in journal.events(EventKind.FILL):
        p = event.payload
        oid = p["client_order_id"]
        if oid not in auto_orders or p.get("stop"):
            continue
        mark = proposals.get(order_proposal.get(oid, ""), {}).get("marks", {}).get(p["instrument"])
        if not mark:
            continue
        direction = 1.0 if p["side"] == "buy" else -1.0
        costs += float(p["quantity"]) * direction * (float(p["price"]) - mark)
        notional += float(p["quantity"]) * mark
        fills += 1
    ratio = None
    if notional > 0 and session.config.baseline_path.exists():
        from runtime.monitor import Baseline

        modeled = Baseline.load(session.config.baseline_path).modeled_bps
        if modeled > 0:
            ratio = (10_000.0 * costs / notional) / modeled
    return Graduation(
        source=f"{name} journal", weeks_exits=weeks, mismatches=mismatches, exit_fills=fills,
        shortfall_ratio=ratio, required_weeks=a.graduation_weeks,
        required_ratio=a.graduation_shortfall_multiple,
    )


# -- arming ---------------------------------------------------------------------------------


def arm(session: LiveSession, scope: str, typed: str, override: str | None = None) -> None:
    """A person allows the system to send orders alone, up to ``scope``.

    Raises:
        ContractViolation: If the config does not allow the scope, the system is
            halted, the phrase is wrong, or -- for ``full`` with live money --
            the graduation evidence is missing and no override is given.
    """
    if scope not in SCOPES:
        raise ContractViolation(f"scope must be one of {', '.join(SCOPES)}")
    ceiling = session.config.automation.mode
    if AUTOMATION_MODES.index(ceiling) < AUTOMATION_MODES.index(scope):
        raise ContractViolation(
            f"the config allows automation up to {ceiling!r}; set automation.mode to "
            f"{scope!r} in the strategy's config first"
        )
    if not session.journal.is_open:
        raise ContractViolation("the sleeve is not open; run `ql live init` first")
    if session.state() is DegradationState.HALTED:
        raise ContractViolation("the system is halted; clear it before arming automation")
    expected = arm_phrase(session, scope)
    if typed.strip() != expected:
        raise ContractViolation(f"confirmation did not match; type exactly: {expected}")
    payload: dict[str, Any] = {"action": "arm", "scope": scope, "by": "person"}
    if scope == "full" and session.config.mode is TradingMode.LIVE:
        evidence = graduation(session)
        payload["graduation"] = {"ok": evidence.ok, "reasons": list(evidence.reasons),
                                 "source": evidence.source}
        if not evidence.ok:
            if override is None or len(override.strip()) < 20:
                raise ContractViolation(
                    "full automation with live money needs the exits stage first: "
                    + "; ".join(evidence.reasons)
                    + ". To proceed anyway, give --override with a written reason."
                )
            payload["override"] = override.strip()
    session.journal.append(EventKind.AUTOMATION, session.clock(), payload)


def disarm(session: LiveSession, reason: str) -> None:
    if len(reason.strip()) < 10:
        raise ContractViolation("disarming needs a reason (at least ten characters)")
    if session.automation_scope() is None:
        raise ContractViolation("automation is not armed")
    session.journal.append(EventKind.AUTOMATION, session.clock(), {
        "action": "disarm", "reason": reason.strip(), "by": "person",
    })


# -- the gates --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gate:
    """One pre-flight check. ``blocks`` says what failing it withholds."""

    name: str
    passed: bool
    detail: str
    blocks: str = "entries"  # "entries" or "all"


@dataclass(frozen=True, slots=True)
class AutoDecision:
    proposal_id: str
    scope: str
    sent: tuple[str, ...]
    held: tuple[str, ...]
    gates: tuple[Gate, ...]

    @property
    def failed(self) -> tuple[Gate, ...]:
        return tuple(g for g in self.gates if not g.passed)


def auto_approve(session: LiveSession) -> AutoDecision | None:
    """Send what the armed scope and the gates allow of the pending proposal.

    Returns ``None`` when automation is not armed or nothing is pending.
    """
    scope = session.automation_scope()
    if scope is None:
        return None
    ceiling = session.config.automation.mode
    if ceiling == "manual":
        return None
    if AUTOMATION_MODES.index(ceiling) < AUTOMATION_MODES.index(scope):
        scope = ceiling  # the config was lowered after arming: it wins
    event = session.pending_proposal()
    if event is None:
        return None
    pid = event.payload["proposal_id"]
    sent_before = session.sent_orders(pid)
    rows = [r for r in event.payload["intents"] if r["client_order_id"] not in sent_before]
    gates: list[Gate] = []

    if event.payload.get("liquidation"):
        gates.append(Gate("liquidation", False, "a liquidation is always a person's decision", "all"))
    try:
        event, book, state = session.approvable(pid)
        gates.append(Gate("proposal", True, "latest, unexpired, current bar, sleeve unchanged", "all"))
    except ContractViolation as error:
        gates.append(Gate("proposal", False, str(error), "all"))
        book = session.book()
        state = session.state()
    rec = session.journal.last(EventKind.RECONCILIATION)
    now = session.clock()
    clean = (
        rec is not None and rec.payload.get("status") != "mismatch"
        and now - rec.at <= RECONCILIATION_MAX_AGE
    )
    gates.append(Gate(
        "reconciliation", clean,
        "clean, within the hour" if clean else "no clean reconciliation in the last hour", "all",
    ))
    a = session.config.automation
    gates.append(Gate(
        "order count", len(rows) <= a.max_orders, f"{len(rows)} orders, at most {a.max_orders}",
        "all",
    ))

    # What adding exposure additionally needs.
    gates.append(Gate(
        "ladder", state.permits_entries, f"the system is {state.value.replace('_', '-')}",
    ))
    snapshot = session.journal.last(EventKind.SNAPSHOT)
    cushion = snapshot.payload.get("account_cushion") if snapshot else None
    gates.append(Gate(
        "margin cushion", cushion is None or cushion >= a.min_cushion,
        "not reported" if cushion is None else f"{cushion:.0%}, at least {a.min_cushion:.0%}",
    ))
    marks = event.payload.get("marks", {})
    equity = float(event.payload.get("equity") or 0.0)
    worst, traded = 0.0, 0.0
    for row in rows:
        intent = intent_from_dict(row)
        price = float(marks.get(str(intent.instrument), 0.0))
        _, opening = split_legs(book.quantity(intent.instrument), intent.side, intent.quantity)
        if equity > 0:
            worst = max(worst, opening * price / equity)
            traded += intent.quantity * price / equity
    cap = _order_limit(session)
    gates.append(Gate(
        "order size", cap is not None and worst <= cap,
        "no baseline to judge order size against; run `ql monitor baseline`" if cap is None
        else f"largest order opens {worst:.0%} of equity; at most {cap:.0%}",
    ))
    limit = _turnover_limit(session)
    gates.append(Gate(
        "turnover", limit is not None and traded <= limit,
        "no baseline to judge turnover against; run `ql monitor baseline`" if limit is None
        else f"{traded:.0%} of equity traded; at most {limit:.0%}",
    ))

    blocked_all = any(not g.passed for g in gates if g.blocks == "all")
    entries_ok = not blocked_all and all(g.passed for g in gates)
    to_send = []
    if not blocked_all:
        for row in rows:
            intent = intent_from_dict(row)
            if scope == "full" and entries_ok:
                to_send.append(intent)
                continue
            exit_part = closing_part(intent, book.quantity(intent.instrument))
            if exit_part is not None:
                to_send.append(exit_part)
    sent_now = {i.client_order_id: i.quantity for i in to_send}
    whole = all(
        sent_now.get(r["client_order_id"]) == float(r["quantity"]) for r in rows
    )
    held = tuple(r["client_order_id"] for r in rows
                 if r["client_order_id"] not in {i.client_order_id for i in to_send})
    decision = AutoDecision(
        proposal_id=pid, scope=scope, sent=tuple(i.client_order_id for i in to_send),
        held=held, gates=tuple(gates),
    )
    if to_send:
        session.send(event, to_send, confirmation="auto", by="automation", partial=not whole)
    if to_send or not _already_recorded(session, pid, held):
        session.journal.append(EventKind.AUTOMATION, session.clock(), {
            "action": "decision", "proposal_id": pid, "scope": scope,
            "sent": list(decision.sent), "held": list(held),
            "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail, "blocks": g.blocks}
                      for g in gates],
        })
    return decision


def _already_recorded(session: LiveSession, pid: str, held: tuple[str, ...]) -> bool:
    for event in reversed(session.journal.events(EventKind.AUTOMATION)):
        if event.payload.get("action") == "decision" and event.payload.get("proposal_id") == pid:
            return tuple(event.payload.get("held", ())) == held
    return False


def _baseline(session: LiveSession):
    from runtime.monitor import Baseline

    if not session.config.baseline_path.exists():
        return None
    try:
        return Baseline.load(session.config.baseline_path)
    except ContractViolation:
        return None


def _order_limit(session: LiveSession) -> float | None:
    """The largest share of equity one automatic order may open."""
    configured = session.config.automation.max_order_fraction
    if configured is not None:
        return configured
    baseline = _baseline(session)
    if baseline is None or not baseline.order_shares:
        return None
    return 1.1 * max(baseline.order_shares)


def _turnover_limit(session: LiveSession) -> float | None:
    """The most one automatic cycle may trade, as a share of equity.

    The backtest's own tail, but never less than one whole book at the gross
    cap: building the book from cash, or replacing it once, is what the
    strategy does, however rarely it did so in the backtest.
    """
    baseline = _baseline(session)
    if baseline is None or not baseline.rotation_turnover:
        return None
    tail = float(np.quantile(baseline.rotation_turnover, session.config.automation.turnover_percentile))
    return max(tail, session.config.risk.max_gross)


# -- the loss breaker --------------------------------------------------------------------------


def loss_breaker(session: LiveSession) -> str | None:
    """Halt when the latest bar's return is below the backtest's tail quantile.

    Judged once per bar, so a person who has looked and cleared the halt is
    not halted again for the same bar. Returns the reason when it fires.
    """
    from runtime.monitor import live_returns

    baseline = _baseline(session)
    if baseline is None or not baseline.returns:
        return None
    labels, _, returns = live_returns(session)
    if not returns:
        return None
    bar = labels[-1]
    for event in session.journal.events(EventKind.AUTOMATION):
        if event.payload.get("action") == "breaker" and event.payload.get("bar") == bar:
            return None
    pct = session.config.automation.loss_breaker_percentile
    threshold = float(np.quantile(baseline.returns, pct))
    fired = returns[-1] < threshold
    session.journal.append(EventKind.AUTOMATION, session.clock(), {
        "action": "breaker", "bar": bar, "return": returns[-1], "threshold": threshold,
        "fired": fired,
    })
    if not fired:
        return None
    reason = (
        f"loss breaker: the {session.interval.noun} to {bar} returned {returns[-1]:.1%}, below "
        f"the backtest's {pct:.1%} quantile ({threshold:.1%})"
    )
    session.degrade(DegradationState.HALTED, reason, by="automation")
    return reason


# -- the cycle ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CycleReport:
    started_at: datetime
    sync: SyncReport | None = None
    refreshed: str | None = None
    monitored: str | None = None
    breaker: str | None = None
    proposal_id: str | None = None
    proposed_orders: int = 0
    decision: AutoDecision | None = None
    notes: tuple[str, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors


def should_propose(session: LiveSession) -> tuple[bool, str]:
    """Whether the cycle should make a proposal now, and why not if not."""
    if not session.state().permits_proposals:
        return False, "the system is halted; an exit is a person's decision"
    if session.rotation_orders_working():
        return False, "orders from the last approval are still working"
    latest = session.market().schedule[-1]
    last = session.journal.last(EventKind.PROPOSAL)
    if last is not None and datetime.fromisoformat(last.payload["decision_time"]) >= latest:
        return False, f"the {session.interval.noun} closing {latest:%Y-%m-%d %H:%M} has a proposal"
    return True, ""


def run_cycle(
    session: LiveSession,
    refresh: Callable[[], str] | None = None,
    monitor: Callable[[], str] | None = None,
) -> CycleReport:
    """Refresh, sync, check, monitor, propose, and -- if armed -- send.

    Idempotent: run it as often as a scheduler likes. A bar is proposed on
    once; an order is sent once; the loss breaker judges a bar once. With
    automation disarmed it does everything except send, so a person only has
    to approve.
    """
    started = session.clock()
    notes: list[str] = []
    errors: list[str] = []
    refreshed = monitored = breaker = proposal_id = None
    proposed = 0
    if refresh is not None:
        try:
            refreshed = refresh()
        except QuantLabError as error:
            errors.append(f"refresh: {error}")
    try:
        report = session.sync()
    except QuantLabError as error:
        errors.append(f"sync: {error}")
        return CycleReport(started_at=started, refreshed=refreshed, notes=tuple(notes),
                           errors=tuple(errors))
    notes.extend(report.notes)
    if session.automation_scope() is not None:
        breaker = loss_breaker(session)
    if monitor is not None:
        try:
            monitored = monitor()
        except QuantLabError as error:
            notes.append(f"monitor: {error}")
    go, why = should_propose(session)
    if go:
        try:
            proposal = session.propose()
            proposal_id, proposed = proposal.proposal_id, len(proposal.orders)
        except QuantLabError as error:
            errors.append(f"propose: {error}")
    else:
        notes.append(f"no new proposal: {why}")
    decision = None
    try:
        decision = auto_approve(session)
    except QuantLabError as error:
        errors.append(f"automatic approval: {error}")
    return CycleReport(
        started_at=started, sync=report, refreshed=refreshed, monitored=monitored,
        breaker=breaker, proposal_id=proposal_id, proposed_orders=proposed,
        decision=decision, notes=tuple(notes), errors=tuple(errors),
    )


# -- scheduling on the Mac ---------------------------------------------------------------------


def schedule_times(interval) -> list[tuple[int | None, int, int]]:
    """(weekday or None for every day, hour, minute), in the Mac's local time.

    Written for a Mac in Spain (CET/CEST). Weekly: Saturday morning, after
    Friday's bar is complete, to refresh, propose and -- if armed -- send the
    opening-auction orders; Monday after the US open to record the fills and
    place the stops; and each weekday evening after the close to record any
    stop that fired. Daily: each weekday after the bar is available (21:15
    UTC, 23:15 in summer) and each weekday after the open. Hourly: every hour
    through the US session.
    """
    from contracts.temporal import BarInterval

    if interval is BarInterval.WEEK:
        return [(6, 10, 0), (1, 16, 5)] + [(d, 22, 45) for d in range(1, 6)]
    if interval is BarInterval.DAY:
        return [(d, 23, 40) for d in range(1, 6)] + [(d, 16, 5) for d in range(1, 6)]
    if interval is BarInterval.HOUR:
        # Every hour from 14:32 to 22:32 on weekdays covers the US session in
        # Spain's time through the weeks when only one side has changed the
        # clocks. A run with no new bar does nothing, so the spare runs are free.
        return [(d, h, 32) for d in range(1, 6) for h in range(14, 23)]
    raise ContractViolation(
        "a minute-bar strategy needs a process that runs through the session, not a "
        "calendar job; that runner is not built"
    )


def launchd_plist(label: str, command: list[str], workdir: Path, log: Path,
                  times: list[tuple[int | None, int, int]]) -> str:
    """A LaunchAgent that runs ``command`` at ``times``. macOS runs a missed
    job when the Mac wakes; a Mac that is off misses it."""
    entries = []
    for weekday, hour, minute in times:
        day = f"<key>Weekday</key><integer>{weekday}</integer>" if weekday is not None else ""
        entries.append(
            f"    <dict>{day}<key>Hour</key><integer>{hour}</integer>"
            f"<key>Minute</key><integer>{minute}</integer></dict>"
        )
    args = "\n".join(f"    <string>{_xml(a)}</string>" for a in command)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{_xml(label)}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>WorkingDirectory</key><string>{_xml(str(workdir))}</string>
  <key>StandardOutPath</key><string>{_xml(str(log))}</string>
  <key>StandardErrorPath</key><string>{_xml(str(log))}</string>
  <key>StartCalendarInterval</key>
  <array>
{chr(10).join(entries)}
  </array>
</dict>
</plist>
"""


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
