"""``ql`` — one command for everything a person does with the system.

    ql data status                      what is cached, what is stored, how old
    ql data refresh                     fetch the latest weekly bars (needs the gateway)
    ql data fetch --symbols A,B         add instruments to the universe (needs the gateway)
    ql data import A=a.json             add instruments without the gateway
    ql data ingest --rebuild            rebuild the store from the cache
    ql backtest [--report]              run the configured strategy over history
    ql funnel                           the five research gates
    ql live init|sync|status|propose|approve|reject|stops|reconcile|adjust|pause|halt|clear|journal
    ql monitor baseline|run             build the reference, then judge live results
    ql report render FILE.json          re-render a saved report

Every command that touches the broker connects with ``configs/live.yaml`` and
disconnects when it is done. Nothing is ever sent without ``ql live approve``,
which asks you to type the proposal's confirmation phrase.

Run ``ql <command> --help`` for the options of each.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.errors import ContractViolation, QuantLabError
from contracts.live import DegradationState, EventKind
from runtime.config import DEFAULT_CONFIG, ROOT, LiveConfig, load_config

STORE = ROOT / "var" / "store"
CACHE = ROOT / "data" / "ibkr_cache"
SCRIPTS = ROOT / "scripts"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Context:
    """What every command needs. Tests replace the broker factory and the clock."""

    config_path: Path = DEFAULT_CONFIG
    store: Path = STORE
    cache: Path = CACHE
    broker_factory: Callable[[LiveConfig], Any] | None = None
    clock: Callable[[], datetime] = now_utc
    input: Callable[[str], str] = input
    out: Callable[[str], None] = print
    _config: LiveConfig | None = field(default=None, init=False)
    _broker: Any = field(default=None, init=False)

    def config(self) -> LiveConfig:
        if self._config is None:
            self._config = load_config(self.config_path)
        return self._config

    def broker(self):
        if self._broker is None:
            config = self.config()
            if self.broker_factory is not None:
                self._broker = self.broker_factory(config)
            else:  # pragma: no cover - needs a running gateway
                from execution.ibkr import IBKRBroker

                g = config.gateway
                self.out(f"connecting to {g.host}:{g.port} as {config.account} "
                         f"({config.mode.value}) ...")
                self._broker = IBKRBroker.connect(
                    g.host, g.port, g.client_id, config.account, config.mode, g.timeout_seconds
                )
        return self._broker

    def session(self, with_broker: bool = True):
        from runtime.live import LiveSession

        return LiveSession(self.config(), self.broker() if with_broker else None,
                           self.store, clock=self.clock)

    def close(self) -> None:
        if self._broker is not None and hasattr(self._broker, "disconnect"):
            with contextlib.suppress(Exception):  # best effort on the way out
                self._broker.disconnect()
        self._broker = None


# -- helpers ----------------------------------------------------------------------


def _money(v: float | None) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _table(ctx: Context, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    cells = [[str(h) for h in header]] + [["—" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    for n, row in enumerate(cells):
        ctx.out("  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                          for i, (c, w) in enumerate(zip(row, widths, strict=True))))
        if n == 0:
            ctx.out("  ".join("-" * w for w in widths))


def _run_script(name: str, argv: Sequence[str]) -> int:
    """Delegate to one of the scripts in ``scripts/``: they stay the single implementation."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ql_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return int(module.main(list(argv)) or 0)


def _report_paths(ctx: Context, stem: str) -> tuple[Path, Path]:
    base = ctx.config().reports_dir if ctx.config_path.exists() else ROOT / "state" / "reports"
    stamp = ctx.clock().strftime("%Y%m%d-%H%M%S")
    return base / f"{stem}-{stamp}.json", base / f"{stem}-{stamp}.html"


def _write_report(ctx: Context, report, stem: str) -> Path:
    from reports.html import write

    data_path, page_path = _report_paths(ctx, stem)
    report.save(data_path)
    write(report, page_path)
    ctx.out(f"report     {page_path}")
    ctx.out(f"data       {data_path}")
    return page_path


# -- data -----------------------------------------------------------------------------


def cmd_data_status(ctx: Context, args) -> int:
    from data.vendor import cache_inventory
    from runtime.wiring import load_market

    inventory = cache_inventory(ctx.cache, "weekly")
    ctx.out(f"cache      {len(inventory)} weekly instruments in {ctx.cache}")
    if args.verbose and len(inventory):
        _table(ctx, list(inventory.columns), inventory.astype(str).values.tolist())
    try:
        market = load_market(ctx.store)
    except Exception as error:
        ctx.out(f"store      not usable ({error}); run `ql data ingest --rebuild`")
        return 1
    last = market.schedule[-1]
    age = (ctx.clock() - last).total_seconds() / 86400
    ctx.out(f"store      {len(market.schedule)} weeks, {market.schedule[0].date()} to {last.date()}")
    ctx.out(f"data age   {age:.1f} days since the last complete week closed")
    if age > 10:
        ctx.out("           stale: run `ql data refresh` (gateway) before proposing")
    return 0


def cmd_data_refresh(ctx: Context, args) -> int:
    from data.bitemporal import BitemporalStore
    from runtime.refresh import refresh_weekly

    store = BitemporalStore(ctx.store, "bars_1week")
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    results = refresh_weekly(ctx.broker(), ctx.cache, store, ctx.clock(), symbols=symbols,
                             duration=args.duration)
    failed = [r for r in results if r.error]
    _table(ctx, ("instrument", "new weeks", "revised", "last week", "error"),
           [(r.instrument, r.new_weeks, r.revised_weeks, r.last_week, r.error or "")
            for r in results if r.error or r.new_weeks or r.revised_weeks or args.verbose])
    ctx.out(f"\n{len(results)} instruments, {sum(r.new_weeks for r in results)} new weeks, "
            f"{sum(r.revised_weeks for r in results)} revisions, {len(failed)} failed")
    if any(r.revised_weeks for r in results):
        ctx.out("revisions are kept as new versions: backtests as-of an earlier date still "
                "see what was known then")
    return 1 if failed else 0


def cmd_data_passthrough(script: str, gateway: bool = False):
    def run(ctx: Context, args) -> int:
        argv = list(args.rest)
        if gateway and ctx.config_path.exists() and not any(
            a.startswith("--port") for a in argv
        ):
            g = ctx.config().gateway
            argv += ["--host", g.host, "--port", str(g.port), "--client-id", str(g.client_id)]
        return _run_script(script, argv)
    return run


# -- backtest ---------------------------------------------------------------------------


def cmd_backtest(ctx: Context, args) -> int:
    from contracts.identifiers import RunId
    from engine.decide import SizingPolicy
    from execution.simulated import CostModel
    from risk.rules import GrossExposureLimit, ProtectiveStop, RiskSupervisor
    from runtime.reporting import backtest_report
    from runtime.research import periodic_returns, precompute_indicators, run_once
    from runtime.wiring import load_market
    from strategies.momentum import MomentumParams, WeeklyMomentum
    from validation.ledger import ResearchLedger, Study
    from validation.metrics import summarise

    defaults = ctx.config() if ctx.config_path.exists() else None
    s = defaults.strategy if defaults else None
    top = args.top or (s.top_n if s else 4)
    lookback = args.lookback or (s.lookback_weeks if s else 13)
    rebalance = args.rebalance_weeks or (s.rebalance_weeks if s else 4)
    stop = args.stop if args.stop is not None else (defaults.risk.stop_distance if defaults else 0.12)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    market = load_market(ctx.store, start=start)
    params = MomentumParams(rebalance_weeks=rebalance, top_n=top, lookback_weeks=lookback)
    strategy = WeeklyMomentum(params, precomputed=precompute_indicators(market, params))
    supervisor = RiskSupervisor(
        rules=(GrossExposureLimit(1.0),), stop=ProtectiveStop(stop) if stop > 0 else None
    )
    schedule = list(market.schedule)[params.warmup_weeks + params.min_history_weeks:]
    costs = CostModel(commission_bps=args.cost_bps, slippage_bps=args.slippage_bps)
    result = run_once(market, strategy, schedule, RunId(f"bt-{top}-{lookback}-{rebalance}"),
                      costs, SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.005),
                      supervisor=supervisor)

    # Every backtest is a trial. Recording it keeps the Deflated Sharpe honest.
    returns = periodic_returns(result)
    window = f"{schedule[0].date()}..{schedule[-1].date()}"
    note = f"stop={stop} cost={args.cost_bps} slip={args.slippage_bps}"
    ledger = ResearchLedger(Path(args.ledger))
    with Study("manual", ledger) as study:
        if study.existing(strategy.version, window, note) is None:
            sd = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
            study.evaluate(strategy.version, window, {
                "sharpe": float(returns.mean() / sd) if sd > 0 else 0.0,
                "final_equity": result.final_equity, "periods": float(returns.size),
            }, returns=returns, note=note)

    stats = summarise(result.equity_curve(), periods_per_year=market.periods_per_year)
    ctx.out(f"strategy     {strategy.version}")
    ctx.out(f"settings     top {top} · lookback {lookback}w · rotate every {rebalance}w · "
            f"stop {stop:.0%} · costs {args.cost_bps}+{args.slippage_bps} bp")
    ctx.out(f"window       {window} ({len(result.steps)} weekly marks)")
    ctx.out(f"CAGR         {stats.cagr:.1%}")
    ctx.out(f"volatility   {stats.volatility:.1%}")
    ctx.out(f"Sharpe       {stats.sharpe:.2f}")
    ctx.out(f"max drawdown {stats.max_drawdown:.1%}")
    ctx.out(f"final equity {stats.final_equity:,.0f} from 100,000")
    ctx.out(f"stops fired  {result.stops_fired()}")
    ctx.out(f"ledger       recorded in study 'manual' ({ledger.count('manual')} manual trials)")
    if args.report:
        settings = {"top_n": top, "lookback_weeks": lookback, "rebalance_weeks": rebalance,
                    "stop_distance": stop, "commission_bps": args.cost_bps,
                    "slippage_bps": args.slippage_bps, "window": window}
        report = backtest_report(result, market, f"Backtest — {strategy.version}", settings,
                                 ctx.clock(), benchmark=args.benchmark)
        _write_report(ctx, report, "backtest")
    return 0


# -- live ---------------------------------------------------------------------------------


def _print_status(ctx: Context, status: dict[str, Any]) -> None:
    ctx.out(f"mode        {status['mode']}  account {status['account']}")
    ctx.out(f"state       {status['state'].upper()}")
    ctx.out(f"equity      {_money(status['sleeve_equity'])}  (cash {_money(status['cash'])})")
    hours = status.get("last_snapshot_hours")
    ctx.out(f"last sync   {'never' if hours is None else f'{hours:.1f} h ago'}")
    ctx.out(f"reconciled  {status.get('last_reconciliation') or 'never'}")
    ctx.out(f"pending     {status.get('pending_proposal') or 'no proposal waiting'}")
    if status["positions"]:
        ctx.out("")
        _table(ctx, ("instrument", "quantity", "avg cost", "mark", "stop"), [
            (p["instrument"], f"{p['quantity']:g}", _money(p["average_cost"]),
             _money(p.get("mark")), _money(p.get("stop")))
            for p in status["positions"]
        ])


def cmd_live_init(ctx: Context, args) -> int:
    session = ctx.session()
    adopt = [a.strip() for a in args.adopt.split(",")] if args.adopt else []
    book = session.init(adopt=adopt)
    ctx.out(f"opened the {ctx.config().mode.value} sleeve: cash {_money(book.cash)}, "
            f"{len(book.positions)} adopted positions")
    report = session.sync()
    ctx.out(f"first sync: reconciliation {report.reconciliation.status}")
    return 0


def cmd_live_sync(ctx: Context, args) -> int:
    session = ctx.session()
    report = session.sync()
    ctx.out(f"fills       {report.new_fills} new")
    ctx.out(f"orders      {report.status_changes} status changes")
    ctx.out(f"stops       {report.stops_placed} placed, {report.stops_cancelled} cancelled")
    ctx.out(f"equity      {_money(report.snapshot.get('sleeve_equity'))}")
    ctx.out(f"reconcile   {report.reconciliation.status.upper()}")
    for finding in report.reconciliation.findings:
        ctx.out(f"  {finding.line()}")
    if report.reconciliation.status == "mismatch":
        ctx.out("\nthe broker and the sleeve disagree; the system is HALTED. See the manual, "
                "'Incidents: reconciliation mismatch'.")
        return 1
    return 0


def cmd_live_status(ctx: Context, args) -> int:
    session = ctx.session(with_broker=not args.offline)
    if not session.journal.is_open:
        config = ctx.config()
        if not args.offline:
            snapshot = session.broker.account_snapshot()
            ctx.out(f"connected   {config.account} ({config.mode.value}); net liquidation "
                    f"{_money(snapshot.net_liquidation)}, cash {_money(snapshot.cash)}")
        ctx.out(f"sleeve      not open yet; run `ql live init` to open the {config.mode.value} "
                "sleeve")
        return 0
    _print_status(ctx, session.status())
    if args.offline:
        ctx.out("\n(offline: stop prices come from the broker and are not shown)")
    return 0


def _print_proposal(ctx: Context, p) -> None:
    kind = "LIQUIDATION" if p.liquidation else "ROTATION" if p.rotation else "HOLD WEEK"
    ctx.out(f"proposal    {p.proposal_id}  ({kind})")
    ctx.out(f"decision    week closing {p.decision_time.date()}  · state {p.state.value.upper()}")
    ctx.out(f"expires     {p.expires_at.isoformat(timespec='minutes')}")
    ctx.out(f"equity      {_money(p.equity)}  (cash {_money(p.cash)})")
    names = sorted(set(p.current_weights) | set(p.target_weights))
    if names:
        ctx.out("")
        _table(ctx, ("instrument", "now", "target"), [
            (n, f"{p.current_weights.get(n, 0):.1%}", f"{p.target_weights.get(n, 0):.1%}")
            for n in names
        ])
    ctx.out("")
    if p.orders:
        _table(ctx, ("side", "instrument", "quantity", "type", "est. price", "est. value",
                     "of equity"), [
            (o.intent.side.value.upper(), str(o.intent.instrument), f"{o.intent.quantity:g}",
             f"{o.intent.order_type.value} {o.intent.time_in_force.value}",
             _money(o.estimated_price), _money(o.estimated_value), f"{o.share_of_equity:.1%}")
            for o in p.orders
        ])
    else:
        ctx.out("no orders: nothing to approve this week")
    for finding in p.findings:
        ctx.out(f"  {finding}")


def cmd_live_propose(ctx: Context, args) -> int:
    session = ctx.session()
    proposal = session.propose(liquidate=args.liquidate)
    _print_proposal(ctx, proposal)
    if proposal.orders:
        ctx.out(f"\nto send: ql live approve {proposal.proposal_id}")
        ctx.out(f"to decline: ql live reject {proposal.proposal_id} --reason \"...\"")
    return 0


def cmd_live_approve(ctx: Context, args) -> int:
    session = ctx.session()
    pending = session.pending_proposal()
    if pending is None:
        raise ContractViolation("no proposal is waiting; run `ql live propose`")
    pid = args.proposal_id or pending.payload["proposal_id"]
    phrase = session.confirmation_phrase(pid)
    payload = pending.payload
    ctx.out(f"proposal {pid} · {len(payload['intents'])} orders · state {payload['state']}")
    for row in payload["intents"]:
        ctx.out(f"  {row['side'].upper():<4} {row['quantity']:>10g} {row['instrument']:<8} "
                f"{row['order_type']} {row.get('time_in_force', '')}")
    mode = ctx.config().mode.value.upper()
    typed = args.confirm if args.confirm is not None else ctx.input(
        f"\n{mode} account {ctx.config().account}. Type '{phrase}' to send, anything else to stop: "
    )
    if typed.strip() != phrase:
        ctx.out("not confirmed; nothing was sent")
        return 1
    sent = session.approve(pid, typed)
    for intent, state in sent:
        ctx.out(f"  sent {intent.side.value.upper():<4} {intent.quantity:g} {intent.instrument} "
                f"-> {state.status.value} {state.message or ''}".rstrip())
    ctx.out(f"\n{len(sent)} orders sent. They fill at the next opening auction; run "
            "`ql live sync` after the open to record fills and place stops.")
    return 0


def cmd_live_reject(ctx: Context, args) -> int:
    session = ctx.session(with_broker=False)
    session.reject(args.proposal_id, args.reason)
    ctx.out(f"rejected {args.proposal_id}; recorded as an override")
    return 0


def cmd_live_stops(ctx: Context, args) -> int:
    placed, cancelled = ctx.session().place_stops()
    ctx.out(f"stops: {placed} placed, {cancelled} cancelled")
    return 0


def cmd_live_reconcile(ctx: Context, args) -> int:
    result = ctx.session().reconcile(record=True)
    ctx.out(f"reconciliation {result.status.upper()}")
    for finding in result.findings:
        ctx.out(f"  {finding.line()}")
    return 1 if result.status == "mismatch" else 0


def cmd_live_adjust(ctx: Context, args) -> int:
    session = ctx.session(with_broker=False)
    book = session.adjust(args.instrument, args.quantity, args.reason,
                          average_cost=args.average_cost, cash_delta=args.cash_delta)
    ctx.out(f"recorded. sleeve cash {_money(book.cash)}, {len(book.positions)} positions")
    ctx.out("run `ql live reconcile` to confirm the sleeve now agrees with the broker")
    return 0


def cmd_live_restrict(to: DegradationState):
    def run(ctx: Context, args) -> int:
        state = ctx.session(with_broker=False).restrict(to, args.reason)
        ctx.out(f"state is now {state.value.upper()}")
        return 0
    return run


def cmd_live_clear(ctx: Context, args) -> int:
    to = DegradationState(args.to)
    # Clearing reconciles against the broker first: a halt is never lifted over
    # a sleeve that disagrees with the account.
    state = ctx.session().clear(to, args.reason)
    ctx.out(f"state is now {state.value.upper()}")
    return 0


def cmd_live_journal(ctx: Context, args) -> int:
    session = ctx.session(with_broker=False)
    kinds = [EventKind(k) for k in args.kind] if args.kind else []
    events = list(session.journal.events(*kinds))[-args.tail:]
    for e in events:
        summary = json.dumps(e.payload, default=str)
        if len(summary) > 160 and not args.full:
            summary = summary[:157] + "..."
        ctx.out(f"{e.sequence:>5} {e.at.isoformat(timespec='minutes')} {e.kind.value:<15} {summary}")
    return 0


# -- monitor ----------------------------------------------------------------------------


def cmd_monitor_baseline(ctx: Context, args) -> int:
    from runtime.monitor import baseline_path, build_baseline

    session = ctx.session(with_broker=False)
    baseline = build_baseline(session, ctx.clock())
    path = baseline_path(session)
    baseline.save(path)
    ctx.out(f"baseline   {baseline.strategy_version}")
    ctx.out(f"history    {baseline.first_week} to {baseline.last_week}, "
            f"{len(baseline.weekly_returns)} weekly returns")
    ctx.out(f"modelled   {baseline.modeled_bps:.1f} bp execution cost per unit traded")
    ctx.out(f"expected   {baseline.expected_rotation_return:.2%} gross return per rotation")
    ctx.out(f"saved      {path}")
    return 0


def cmd_monitor_run(ctx: Context, args) -> int:
    from runtime.monitor import run_monitor
    from runtime.reporting import live_report

    session = ctx.session(with_broker=False)
    report = run_monitor(session, apply=not args.dry_run, benchmark=args.benchmark)
    a = report.assessment
    change = ("" if report.state_after == report.state_before
              else f" (was {report.state_before.value.upper()})")
    ctx.out(f"state      {report.state_after.value.upper()}{change}"
            + ("  [dry run: not applied]" if args.dry_run else ""))
    ctx.out(f"live weeks {a.weeks}")
    if a.drawdown:
        ctx.out(f"drawdown   {-a.drawdown.live_drawdown:.1%}, deeper than "
                f"{a.drawdown.percentile:.0%} of bootstrapped backtest windows")
    if a.break_probability is not None:
        ctx.out(f"break      {a.break_probability:.0%} probability the return process changed")
    if a.trend:
        ctx.out(f"trend      {a.trend.weekly_slope:+.2%}/week, CI {a.trend.low:+.2%} … "
                f"{a.trend.high:+.2%}" + ("" if a.trend.judged else " (not judged yet)"))
    if a.shortfall:
        ctx.out(f"shortfall  {a.shortfall.mean_bps:.1f} bp per rotation vs "
                f"{a.shortfall.modeled_bps:.1f} bp modelled")
    for reason in a.reasons:
        ctx.out(f"  {reason}")
    for problem in report.health.problems:
        ctx.out(f"  health: {problem}")
    if not args.no_report:
        status = session.status()
        page = live_report(report, status, asdict(ctx.config().monitoring), args.benchmark)
        _write_report(ctx, page, f"live-{report.mode}")
    return 0


# -- reports -------------------------------------------------------------------------------


def cmd_report_render(ctx: Context, args) -> int:
    from reports.document import Report
    from reports.html import write

    source = Path(args.path)
    report = Report.load(source)
    target = Path(args.out) if args.out else source.with_suffix(".html")
    write(report, target)
    ctx.out(f"report     {target}")
    return 0


def cmd_report_list(ctx: Context, args) -> int:
    base = ctx.config().reports_dir if ctx.config_path.exists() else ROOT / "state" / "reports"
    pages = sorted(base.glob("*.html"))[-args.tail:]
    if not pages:
        ctx.out(f"no reports in {base}")
    for page in pages:
        ctx.out(str(page))
    return 0


# -- parser ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ql", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to live.yaml")
    parser.add_argument("--store", default=None, help="Path to the bitemporal store")
    top = parser.add_subparsers(dest="command", required=True)

    data = top.add_parser("data", help="Price data and the universe").add_subparsers(
        dest="sub", required=True)
    p = data.add_parser("status", help="What is cached and stored, and how old it is")
    p.add_argument("-v", "--verbose", action="store_true", help="List every instrument")
    p.set_defaults(func=cmd_data_status)
    p = data.add_parser("refresh", help="Fetch the latest weekly bars for the universe (gateway)")
    p.add_argument("--symbols", default=None, help="Only these, comma-separated")
    p.add_argument("--duration", default="2 Y", help="How far back to re-fetch (IBKR syntax)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_data_refresh)
    for name, script, gw, text in (
        ("fetch", "fetch_ibkr", True, "Add instruments from the gateway (scripts/fetch_ibkr.py)"),
        ("import", "import_ibkr_json", False, "Add instruments from saved JSON payloads"),
        ("ingest", "ingest_ibkr_cache", False, "Load the cache into the store (--rebuild)"),
    ):
        p = data.add_parser(name, help=text, add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(func=cmd_data_passthrough(script, gw))

    p = top.add_parser("backtest", help="Backtest the configured strategy over history")
    p.add_argument("--top", type=int, default=None, help="Names held (default: config)")
    p.add_argument("--lookback", type=int, default=None, help="Momentum lookback, weeks")
    p.add_argument("--rebalance-weeks", type=int, default=None, help="Weeks between rotations")
    p.add_argument("--stop", type=float, default=None, help="Stop distance; 0 disables")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--start", default=None, help="Earliest week, ISO date")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--ledger", default=str(ROOT / "state" / "research.jsonl"))
    p.add_argument("--report", action="store_true", help="Write an HTML + JSON report")
    p.set_defaults(func=cmd_backtest)

    p = top.add_parser("funnel", help="The five research gates (scripts/run_funnel.py)",
                       add_help=False)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=lambda ctx, args: _run_script("run_funnel", args.rest))

    live = top.add_parser("live", help="The live (or paper) sleeve").add_subparsers(
        dest="sub", required=True)
    p = live.add_parser("init", help="Open the sleeve, once")
    p.add_argument("--adopt", default="", help="Existing holdings to hand over, comma-separated")
    p.set_defaults(func=cmd_live_init)
    live.add_parser("sync", help="Record fills, place stops, snapshot, reconcile").set_defaults(
        func=cmd_live_sync)
    p = live.add_parser("status", help="State, equity, positions and stops")
    p.add_argument("--offline", action="store_true", help="Do not connect to the gateway")
    p.set_defaults(func=cmd_live_status)
    p = live.add_parser("propose", help="Compute this week's orders. Sends nothing")
    p.add_argument("--liquidate", action="store_true", help="Propose selling everything")
    p.set_defaults(func=cmd_live_propose)
    p = live.add_parser("approve", help="Send the pending proposal, after typed confirmation")
    p.add_argument("proposal_id", nargs="?", default=None)
    p.add_argument("--confirm", default=None,
                   help="The confirmation phrase, instead of typing it at the prompt")
    p.set_defaults(func=cmd_live_approve)
    p = live.add_parser("reject", help="Decline the pending proposal, with a reason")
    p.add_argument("proposal_id")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_live_reject)
    live.add_parser("stops", help="Place missing protective stops").set_defaults(
        func=cmd_live_stops)
    live.add_parser("reconcile", help="Compare the sleeve with the broker").set_defaults(
        func=cmd_live_reconcile)
    p = live.add_parser("adjust", help="Correct the sleeve's record, with a reason")
    p.add_argument("--instrument", default=None)
    p.add_argument("--quantity", type=float, default=None, help="The correct absolute quantity")
    p.add_argument("--average-cost", type=float, default=None)
    p.add_argument("--cash-delta", type=float, default=0.0,
                   help="Cash added (+) or withdrawn (-) from the sleeve")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_live_adjust)
    for name, to, text in (
        ("pause", DegradationState.REDUCE_ONLY, "Stop new buys (reduce-only) until cleared"),
        ("halt", DegradationState.HALTED, "Stop all rotations until cleared"),
    ):
        p = live.add_parser(name, help=text)
        p.add_argument("--reason", required=True)
        p.set_defaults(func=cmd_live_restrict(to))
    p = live.add_parser("clear", help="Lift a pause or halt, with a reason")
    p.add_argument("--to", choices=[s.value for s in DegradationState], default="normal")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_live_clear)
    p = live.add_parser("journal", help="Read the event journal")
    p.add_argument("--tail", type=int, default=30)
    p.add_argument("--kind", action="append", choices=[k.value for k in EventKind])
    p.add_argument("--full", action="store_true", help="Do not shorten payloads")
    p.set_defaults(func=cmd_live_journal)

    monitor = top.add_parser("monitor", help="Judge live results").add_subparsers(
        dest="sub", required=True)
    monitor.add_parser("baseline", help="Backtest the configured strategy as the reference"
                       ).set_defaults(func=cmd_monitor_baseline)
    p = monitor.add_parser("run", help="Every check, the resulting state, and a dashboard")
    p.add_argument("--dry-run", action="store_true", help="Do not change the state")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--benchmark", default="SPY")
    p.set_defaults(func=cmd_monitor_run)

    report = top.add_parser("report", help="Saved reports").add_subparsers(dest="sub", required=True)
    p = report.add_parser("render", help="Re-render a saved JSON report as HTML")
    p.add_argument("path")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report_render)
    p = report.add_parser("list", help="The most recent report pages")
    p.add_argument("--tail", type=int, default=10)
    p.set_defaults(func=cmd_report_list)
    return parser


def main(argv: Sequence[str] | None = None, context: Context | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx = context or Context()
    if args.config:
        ctx.config_path = Path(args.config)
    if args.store:
        ctx.store = Path(args.store)
    try:
        return args.func(ctx, args)
    except QuantLabError as error:
        # Every refusal the system makes is one of these, with a message written
        # for a person. Anything else is a bug and keeps its traceback.
        ctx.out(f"error: {error}")
        return 2
    except KeyboardInterrupt:
        ctx.out("\ninterrupted; nothing further was sent")
        return 130
    finally:
        ctx.close()


if __name__ == "__main__":
    raise SystemExit(main())
