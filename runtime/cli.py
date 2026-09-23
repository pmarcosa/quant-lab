"""``ql`` — one command for everything a person does with the system.

    ql strategies                       every configured strategy, its account and sleeve
    ql data status [--interval day]     what is cached, what is stored, how old
    ql data refresh                     fetch the latest complete bars (needs the gateway)
    ql data backfill --interval hour    page back through years of intraday history
    ql data fetch --symbols A,B         add instruments to the universe (needs the gateway)
    ql data import A=a.json             add instruments without the gateway
    ql data ingest --rebuild            rebuild the store from the cache
    ql backtest [--report]              run the configured strategy over history
    ql funnel                           the five research gates
    ql live init|sync|status|propose|approve|reject|stops|reconcile|adjust|pause|halt|clear|journal
    ql live cycle                       the whole cycle in one command (for a scheduler)
    ql live auto status|arm|disarm|schedule   automatic sending, off until armed
    ql monitor baseline|run             build the reference, then judge live results
    ql report render FILE.json          re-render a saved report

Each deployed strategy has its own config, ``configs/strategies/<id>.yaml``, and
its own IBKR account. With one strategy configured every command uses it; with
several, say which: ``ql --strategy <id> live sync``. The id is written on every
order the strategy sends (IBKR's Order Ref, ``ql-<id>.<hash>``).

Every command that touches the broker connects with that strategy's config and
disconnects when it is done. Nothing is sent without ``ql live approve`` -- which
asks you to type the proposal's confirmation phrase -- unless the config allows
automation (``automation.mode``) *and* a person has armed it with
``ql live auto arm``; ``ql live cycle`` then sends what its gates allow.

Data commands work on the strategy's bar size (weekly, daily, hourly, minute);
``--interval`` overrides it.

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
from runtime.config import (
    CONFIGS,
    ROOT,
    LiveConfig,
    check_accounts,
    config_paths,
    interval_of,
    load_config,
)

STORE = ROOT / "var" / "store"
CACHE = ROOT / "data" / "ibkr_cache"
SCRIPTS = ROOT / "scripts"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Context:
    """What every command needs. Tests replace the broker factory and the clock.

    Which strategy a command acts on: ``--config PATH``, else ``--strategy NAME``
    (``configs/strategies/<NAME>.yaml``; normally the file is named after its
    ``strategy_id``, and a paper twin of a live strategy is a second file with
    the same id), else the only strategy configured. With
    several configured and neither given, the command refuses and lists them --
    guessing which account to trade in is not a default.
    """

    config_path: Path | None = None
    store: Path = STORE
    cache: Path = CACHE
    broker_factory: Callable[[LiveConfig], Any] | None = None
    clock: Callable[[], datetime] = now_utc
    input: Callable[[str], str] = input
    out: Callable[[str], None] = print
    strategy: str | None = None
    configs_root: Path = CONFIGS
    _config: LiveConfig | None = field(default=None, init=False)
    _broker: Any = field(default=None, init=False)

    def resolve_config_path(self) -> Path:
        if self.config_path is not None:
            return Path(self.config_path)
        if self.strategy:
            path = self.configs_root / "strategies" / f"{self.strategy}.yaml"
            if not path.exists():
                known = [q.stem for q in config_paths(self.configs_root)]
                raise ContractViolation(
                    f"no config for strategy {self.strategy!r} at {path}; configured: "
                    f"{known or 'none'}"
                )
            return path
        found = config_paths(self.configs_root)
        if len(found) == 1:
            return found[0]
        if not found:
            raise ContractViolation(
                "no strategy is configured. Copy configs/live.example.yaml to "
                "configs/strategies/<id>.yaml and fill it in (manual, section 7)."
            )
        raise ContractViolation(
            f"several strategies are configured ({', '.join(q.stem for q in found)}); "
            f"say which with --strategy <id>"
        )

    def has_config(self) -> bool:
        try:
            self.resolve_config_path()
        except ContractViolation:
            return False
        return True

    def config(self) -> LiveConfig:
        if self._config is None:
            path = self.resolve_config_path()
            config = load_config(path)
            # One strategy per account, and unique ids, across every config --
            # checked whenever any one of them is used.
            others = [
                load_config(q) for q in config_paths(self.configs_root)
                if q.resolve() != path.resolve()
            ]
            check_accounts([config, *others])
            self._config = config
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
                         f"({config.mode.value}, strategy {config.strategy_id}) ...")
                self._broker = IBKRBroker.connect(
                    g.host, g.port, g.client_id, config.account, config.mode,
                    g.timeout_seconds, order_prefix=_prefix(config),
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


def _prefix(config: LiveConfig) -> str:
    from contracts.execution import order_prefix
    from contracts.identifiers import PortfolioId
    from runtime.live import TENANT

    return order_prefix(PortfolioId(TENANT, config.strategy_id))


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
    base = ctx.config().reports_dir if ctx.has_config() else ROOT / "state" / "reports"
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


def cmd_strategies(ctx: Context, args) -> int:
    from runtime.journal import Journal

    paths = config_paths(ctx.configs_root)
    if not paths:
        ctx.out("no strategy is configured; see the manual, section 7")
        return 0
    configs = [load_config(q) for q in paths]
    check_accounts(configs)
    rows = []
    for c in configs:
        journal = Journal(c.journal_path)
        opened = "open" if journal.is_open else "not opened"
        rows.append((c.strategy_id, c.strategy.name, interval_of(c).frequency,
                     c.mode.value, c.account, f"{c.sleeve_capital:,.0f}", opened))
    _table(ctx, ("id", "strategy", "bars", "mode", "account", "capital", "sleeve"), rows)
    return 0


def _interval(ctx: Context, args):
    """The bar size a data command works on: ``--interval``, else the strategy's."""
    from contracts.temporal import BarInterval

    chosen = getattr(args, "interval", None)
    if chosen:
        return BarInterval.parse(chosen)
    if ctx.has_config():
        return interval_of(ctx.config())
    return BarInterval.WEEK


def cmd_data_status(ctx: Context, args) -> int:
    from data.vendor import cache_inventory
    from runtime.wiring import load_market

    interval = _interval(ctx, args)
    inventory = cache_inventory(ctx.cache, interval.frequency)
    ctx.out(f"cache      {len(inventory)} {interval.frequency} instruments in {ctx.cache}")
    if args.verbose and len(inventory):
        _table(ctx, list(inventory.columns), inventory.astype(str).values.tolist())
    try:
        market = load_market(ctx.store, interval=interval)
    except Exception as error:
        ctx.out(f"store      not usable ({error}); run `ql data ingest --rebuild`")
        return 1
    last = market.schedule[-1]
    age = (ctx.clock() - last).total_seconds() / 3600
    ctx.out(f"store      {len(market.schedule)} {interval.noun}s, "
            f"{market.schedule[0].date()} to {last.date()}")
    ctx.out(f"data age   {age / 24:.1f} days since the last complete {interval.noun} closed")
    from runtime.config import DEFAULT_DATA_AGE_HOURS

    limit = (
        ctx.config().monitoring.data_age_hours(interval) if ctx.has_config()
        else DEFAULT_DATA_AGE_HOURS[interval]
    )
    if age > limit:
        ctx.out("           stale: run `ql data refresh` (gateway) before proposing")
    return 0


def cmd_data_refresh(ctx: Context, args) -> int:
    from data.bitemporal import BitemporalStore
    from runtime.refresh import refresh

    interval = _interval(ctx, args)
    store = BitemporalStore(ctx.store, f"bars_{interval.value.lower()}")
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    results = refresh(ctx.broker(), ctx.cache, store, ctx.clock(), interval,
                      symbols=symbols, duration=args.duration)
    failed = [r for r in results if r.error]
    noun = interval.noun
    _table(ctx, ("instrument", f"new {noun}s", "revised", f"last {noun}", "error"),
           [(r.instrument, r.new_bars, r.revised_bars, r.last_bar, r.error or "")
            for r in results if r.error or r.new_bars or r.revised_bars or args.verbose])
    ctx.out(f"\n{len(results)} instruments, {sum(r.new_bars for r in results)} new {noun}s, "
            f"{sum(r.revised_bars for r in results)} revisions, {len(failed)} failed")
    if any(r.revised_bars for r in results):
        ctx.out("revisions are kept as new versions: backtests as-of an earlier date still "
                "see what was known then")
    return 1 if failed else 0


def cmd_data_backfill(ctx: Context, args) -> int:
    """Page back through the broker's history until the cache holds ``--years``."""
    from contracts.temporal import BarInterval
    from data.vendor import cache_inventory
    from runtime.refresh import BACKFILL_CHUNKS, PACING_SECONDS, backfill

    interval = _interval(ctx, args)
    if interval is BarInterval.WEEK and not args.interval:
        interval = BarInterval.HOUR
    symbols = ([s.strip() for s in args.symbols.split(",")] if args.symbols
               else list(cache_inventory(ctx.cache, "weekly")["symbol"]))
    chunk = args.chunk or BACKFILL_CHUNKS[interval]
    ctx.out(f"backfill   {len(symbols)} instruments, {interval.frequency} bars, {args.years:g} "
            f"years, {chunk} per request, {PACING_SECONDS:g} s apart (IBKR's pacing limit)")
    results = backfill(ctx.broker(), ctx.cache, interval, ctx.clock(), args.years, symbols,
                       chunk=chunk)
    _table(ctx, ("instrument", "bars added", "first bar", "requests", "note"), [
        (r.instrument, r.bars_added, r.first_bar or "—", r.requests,
         r.error or ("the broker has nothing older" if r.exhausted else ""))
        for r in results
    ])
    ctx.out("\nnext: `ql data ingest --rebuild` to load the cache into the store")
    return 1 if any(r.error for r in results) else 0


def cmd_data_passthrough(script: str, gateway: bool = False):
    def run(ctx: Context, args) -> int:
        argv = list(args.rest)
        if gateway and ctx.has_config() and not any(a.startswith("--port") for a in argv):
            g = ctx.config().gateway
            argv += ["--host", g.host, "--port", str(g.port), "--client-id", str(g.client_id)]
        return _run_script(script, argv)
    return run


# -- backtest ---------------------------------------------------------------------------


def cmd_backtest(ctx: Context, args) -> int:
    from dataclasses import replace as _replace

    from contracts.identifiers import RunId
    from engine.decide import SizingPolicy
    from execution.simulated import CostModel
    from risk.rules import (
        GrossExposureLimit,
        NetExposureLimit,
        ProtectiveStop,
        RiskSupervisor,
        ShortSales,
    )
    from runtime.config import FinancingSettings, LeverageSettings, RiskSettings, StrategySettings
    from runtime.reporting import backtest_report
    from runtime.research import periodic_returns, run_once
    from runtime.strategies import build_strategy
    from runtime.wiring import load_market
    from validation.ledger import ResearchLedger, Study
    from validation.metrics import summarise

    config = ctx.config() if ctx.has_config() else None
    chosen = config.strategy if config else StrategySettings()
    risk = config.risk if config else RiskSettings()
    lev = config.leverage if config else LeverageSettings()
    fin = config.financing if config else FinancingSettings()
    if args.leverage is not None:
        # A research override: a fixed leverage, with the cap raised to meet it.
        lev = _replace(lev, target=args.leverage, cvar_target=None,
                       floor=min(lev.floor, args.leverage))
        cap = max(risk.max_gross, args.leverage)
        risk = _replace(risk, max_gross=cap, max_net=max(risk.net_cap, cap))
    if args.margin_rate is not None:
        fin = _replace(fin, margin_rate=args.margin_rate)
    schedule_rule = lev.schedule(risk.max_gross)
    params = dict(chosen.params)
    overrides = {"top_n": args.top, "lookback_weeks": args.lookback,
                 "rebalance_weeks": args.rebalance_weeks}
    given = {k: v for k, v in overrides.items() if v}
    if given and chosen.name != "weekly-momentum":
        raise ContractViolation(
            f"--top/--lookback/--rebalance-weeks are weekly-momentum parameters; the "
            f"configured strategy is {chosen.name}. Edit its params in the config instead."
        )
    params.update(given)
    stop = args.stop if args.stop is not None else risk.stop_distance

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    interval = build_strategy(chosen.name, params).filtration_spec.interval
    market = load_market(ctx.store, interval=interval, start=start)
    strategy = build_strategy(chosen.name, params, market)
    supervisor = RiskSupervisor(
        rules=(
            ShortSales(allowed=risk.allow_short),
            GrossExposureLimit(risk.max_gross),
            NetExposureLimit(risk.min_net, risk.net_cap),
        ),
        stop=ProtectiveStop(stop) if stop > 0 else None,
    )
    # A strategy that needs a warm-up says so in its parameters; the backtest
    # window starts after it, so its first bars are not reported as flat returns.
    spec = getattr(strategy, "params", None)
    warmup = (
        int(getattr(spec, "warmup_weeks", 0)) + int(getattr(spec, "min_history_weeks", 0))
        if spec is not None else 0
    )
    schedule = list(market.schedule)[warmup:]
    costs = CostModel(commission_bps=args.cost_bps, slippage_bps=args.slippage_bps)
    run_name = "bt-" + "-".join(str(v) for _, v in sorted(params.items()))[:40]
    result = run_once(market, strategy, schedule, RunId(run_name), costs,
                      SizingPolicy(cash_buffer=0.01, min_trade_fraction=0.005,
                                   allow_short=risk.allow_short),
                      supervisor=supervisor,
                      leverage=None if schedule_rule.is_static_unlevered else schedule_rule,
                      financing=fin.model())

    # Every backtest is a trial. Recording it keeps the Deflated Sharpe honest.
    returns = periodic_returns(result)
    window = f"{schedule[0].date()}..{schedule[-1].date()}"
    note = f"stop={stop} cost={args.cost_bps} slip={args.slippage_bps}"
    if risk.allow_short:
        note += f" short gross={risk.max_gross} net=[{risk.min_net},{risk.net_cap}]"
    if not schedule_rule.is_static_unlevered:
        note += (f" lev={lev.target}/{risk.max_gross}"
                 + (f" cvar={lev.cvar_target}" if lev.cvar_target else "")
                 + f" margin={fin.margin_rate}")
    ledger = ResearchLedger(Path(args.ledger))
    with Study("manual", ledger) as study:
        if study.existing(strategy.version, window, note) is None:
            sd = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
            study.evaluate(strategy.version, window, {
                "sharpe": float(returns.mean() / sd) if sd > 0 else 0.0,
                "final_equity": result.final_equity, "periods": float(returns.size),
            }, returns=returns, note=note)

    stats = summarise(result.equity_curve(), periods_per_year=market.periods_per_year)
    shown = " · ".join(f"{k} {v}" for k, v in sorted(params.items()))
    ctx.out(f"strategy     {strategy.version} ({chosen.name}, {interval.frequency})")
    ctx.out(f"settings     {shown} · stop {stop:.0%} · costs {args.cost_bps}+{args.slippage_bps} bp"
            + (" · shorts allowed" if risk.allow_short else ""))
    ctx.out(f"window       {window} ({len(result.steps)} {interval.frequency} marks)")
    ctx.out(f"CAGR         {stats.cagr:.1%}")
    ctx.out(f"volatility   {stats.volatility:.1%}")
    ctx.out(f"Sharpe       {stats.sharpe:.2f}")
    ctx.out(f"max drawdown {stats.max_drawdown:.1%}")
    ctx.out(f"final equity {stats.final_equity:,.0f} from 100,000")
    ctx.out(f"stops fired  {result.stops_fired()}")
    if not schedule_rule.is_static_unlevered or result.financing_paid > 0:
        levels = [s.leverage for s in result.steps]
        ctx.out(f"leverage     average {sum(levels) / len(levels):.2f}x, range "
                f"{min(levels):.2f}-{max(levels):.2f}x (cap {risk.max_gross:.2f}x)")
        ctx.out(f"financing    {result.financing_paid:,.0f} paid (margin {fin.margin_rate:.2%}, "
                f"borrow {fin.borrow_fee:.2%} a year)")
        if result.min_cushion is not None:
            ctx.out(f"cushion      thinnest {result.min_cushion:.0%} (modelled Reg T maintenance)")
    ctx.out(f"ledger       recorded in study 'manual' ({ledger.count('manual')} manual trials)")
    if risk.allow_short:
        ctx.out("note         shorts pay a flat borrow fee here; hard-to-borrow names cost more, "
                "and recalls are not modelled")
    if args.report:
        settings = {**params, "strategy": chosen.name, "interval": interval.frequency,
                    "stop_distance": stop, "commission_bps": args.cost_bps,
                    "slippage_bps": args.slippage_bps, "window": window}
        report = backtest_report(result, market, f"Backtest — {strategy.version}", settings,
                                 ctx.clock(), benchmark=args.benchmark)
        _write_report(ctx, report, "backtest")
    return 0


# -- live ---------------------------------------------------------------------------------


def _print_strategy(ctx: Context) -> None:
    config = ctx.config()
    ctx.out(f"strategy    {config.strategy_id} ({config.strategy.name}, "
            f"{interval_of(config).frequency} bars)")


def _print_status(ctx: Context, status: dict[str, Any]) -> None:
    ctx.out(f"mode        {status['mode']}  account {status['account']}")
    ctx.out(f"state       {status['state'].upper()}")
    ctx.out(f"automation  {status.get('automation') or 'off: every order needs your approval'}")
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
    _print_strategy(ctx)
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
    unit = session.interval.noun
    ctx.out(f"history    {baseline.first_bar} to {baseline.last_bar}, "
            f"{len(baseline.returns)} returns, one per {unit}")
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
    ctx.out(f"strategy   {report.strategy_id} ({report.mode})")
    ctx.out(f"live {a.unit}s {a.periods}")
    if a.drawdown:
        ctx.out(f"drawdown   {-a.drawdown.live_drawdown:.1%}, deeper than "
                f"{a.drawdown.percentile:.0%} of bootstrapped backtest windows")
    if a.break_probability is not None:
        ctx.out(f"break      {a.break_probability:.0%} probability the return process changed")
    if a.trend:
        ctx.out(f"trend      {a.trend.slope:+.2%}/{a.unit}, CI {a.trend.low:+.2%} … "
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


# -- automation ----------------------------------------------------------------------------


def cmd_live_cycle(ctx: Context, args) -> int:
    """Everything a week (or a day) needs, in order; sends only if armed."""
    from runtime.automation import run_cycle

    session = ctx.session()
    stamp = ctx.clock().strftime("%Y-%m-%d %H:%M UTC")
    ctx.out(f"cycle      {stamp}  strategy {session.config.strategy_id} "
            f"({session.config.mode.value}), automation "
            f"{session.automation_scope() or 'not armed'}")

    def refresh() -> str:
        from data.bitemporal import BitemporalStore
        from runtime.refresh import refresh as fetch

        interval = session.interval
        store = BitemporalStore(ctx.store, f"bars_{interval.value.lower()}")
        results = fetch(session.broker, ctx.cache, store, ctx.clock(), interval)
        failed = [r for r in results if r.error]
        return (f"{sum(r.new_bars for r in results)} new {interval.noun}s, "
                f"{len(failed)} of {len(results)} instruments failed")

    def monitor() -> str:
        from runtime.monitor import run_monitor
        from runtime.reporting import live_report

        report = run_monitor(session, apply=True)
        page = live_report(report, session.status(), asdict(session.config.monitoring), "SPY")
        _write_report(ctx, page, f"live-{report.mode}")
        return report.state_after.value

    result = run_cycle(session, refresh=None if args.no_refresh else refresh,
                       monitor=None if args.no_monitor else monitor)
    if result.refreshed:
        ctx.out(f"data       {result.refreshed}")
    if result.sync is not None:
        s = result.sync
        ctx.out(f"sync       {s.new_fills} fills, {s.stops_placed} stops placed, "
                f"reconcile {s.reconciliation.status.upper()}, state {s.state.value.upper()}")
    if result.breaker:
        ctx.out(f"HALTED     {result.breaker}")
    if result.monitored:
        ctx.out(f"monitor    state {result.monitored.upper()}")
    if result.proposal_id:
        ctx.out(f"proposal   {result.proposal_id} with {result.proposed_orders} orders")
    if result.decision is not None:
        d = result.decision
        ctx.out(f"automatic  scope {d.scope}: {len(d.sent)} sent, {len(d.held)} held for a person")
        for gate in d.failed:
            ctx.out(f"  gate {gate.name}: {gate.detail}")
    elif session.pending_proposal() is not None:
        pid = session.pending_proposal().payload["proposal_id"]
        ctx.out(f"waiting    {pid} needs `ql live approve {pid}`")
    for note in result.notes:
        ctx.out(f"  {note}")
    for error in result.errors:
        ctx.out(f"error: {error}")
    return 0 if result.ok else 1


def cmd_live_auto(ctx: Context, args) -> int:
    from runtime import automation

    session = ctx.session(with_broker=False)
    if args.action == "status":
        config = session.config.automation
        ctx.out(f"allowed    {config.mode} (automation.mode in the config)")
        ctx.out(f"armed      {session.automation_scope() or 'no'}")
        evidence = automation.graduation(session)
        ctx.out(f"evidence   {evidence.weeks_exits:.1f} weeks armed for exits "
                f"({evidence.source}), {evidence.exit_fills} automatic exit fills, "
                f"{evidence.mismatches} mismatches"
                + (f", shortfall {evidence.shortfall_ratio:.2f}x modelled"
                   if evidence.shortfall_ratio is not None else ""))
        ctx.out("graduated  " + ("yes" if evidence.ok else "no: " + "; ".join(evidence.reasons)))
        last = [e for e in session.journal.events(EventKind.AUTOMATION)][-5:]
        for event in last:
            payload = event.payload
            ctx.out(f"  {event.at:%Y-%m-%d %H:%M}  {payload.get('action')}  "
                    + ", ".join(f"{k}={v}" for k, v in payload.items()
                                if k in ("scope", "reason", "proposal_id", "sent", "held", "fired")))
        return 0
    if args.action == "arm":
        phrase = automation.arm_phrase(session, args.scope)
        typed = args.confirm if args.confirm is not None else ctx.input(
            f"type {phrase} to let the system send {args.scope} orders on its own: ")
        automation.arm(session, args.scope, typed, override=args.override)
        ctx.out(f"armed for {args.scope}. `ql live cycle` now sends what the gates allow; "
                f"`ql live auto disarm` stops it.")
        return 0
    if args.action == "disarm":
        automation.disarm(session, args.reason or "")
        ctx.out("disarmed: nothing is sent without a typed approval")
        return 0
    if args.action == "schedule":
        import sys

        config = session.config
        times = automation.schedule_times(session.interval)
        label = f"com.quantlab.{config.strategy_id}.{config.mode.value}.cycle"
        name = ctx.resolve_config_path().stem
        command = [sys.executable, "-m", "runtime.cli", "--strategy", name, "live", "cycle"]
        folder = config.live_dir / "launchd"
        folder.mkdir(parents=True, exist_ok=True)
        plist = folder / f"{label}.plist"
        plist.write_text(automation.launchd_plist(
            label, command, ROOT, config.live_dir / "cycle.log", times))
        days = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", None: "daily"}
        ctx.out(f"wrote      {plist}")
        ctx.out("runs at   " + ", ".join(f"{days[d]} {h:02d}:{m:02d}" for d, h, m in times)
                + " (the Mac's local time)")
        ctx.out("install:  cp '" + str(plist) + "' ~/Library/LaunchAgents/ && "
                "launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/" + plist.name)
        ctx.out("remove:   launchctl bootout gui/$(id -u)/" + label)
        return 0
    raise ContractViolation(f"unknown action {args.action}")


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
    base = ctx.config().reports_dir if ctx.has_config() else ROOT / "state" / "reports"
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
    parser.add_argument("--strategy", default=None,
                        help="Acts on configs/strategies/<name>.yaml (normally named "
                             "after its strategy_id)")
    parser.add_argument("--config", default=None, help="Path to a strategy config file")
    parser.add_argument("--store", default=None, help="Path to the bitemporal store")
    top = parser.add_subparsers(dest="command", required=True)

    p = top.add_parser("strategies", help="Every configured strategy, its account and state")
    p.set_defaults(func=cmd_strategies)

    data = top.add_parser("data", help="Price data and the universe").add_subparsers(
        dest="sub", required=True)
    p = data.add_parser("status", help="What is cached and stored, and how old it is")
    p.add_argument("-v", "--verbose", action="store_true", help="List every instrument")
    p.add_argument("--interval", default=None,
                   help="week, day, hour or minute (default: the strategy's)")
    p.set_defaults(func=cmd_data_status)
    p = data.add_parser("refresh", help="Fetch the latest bars for the universe (gateway)")
    p.add_argument("--symbols", default=None, help="Only these, comma-separated")
    p.add_argument("--duration", default=None,
                   help="How far back to re-fetch (IBKR syntax; default by interval)")
    p.add_argument("--interval", default=None,
                   help="week, day, hour or minute (default: the strategy's)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_data_refresh)
    p = data.add_parser("backfill", help="Page back through intraday history (gateway)")
    p.add_argument("--interval", default=None, help="hour or minute (default: hour)")
    p.add_argument("--years", type=float, default=5.0)
    p.add_argument("--symbols", default=None, help="Comma-separated; default: the universe")
    p.add_argument("--chunk", default=None, help='Per request, IBKR syntax (e.g. "1 Y")')
    p.set_defaults(func=cmd_data_backfill)
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
    p.add_argument("--leverage", type=float, default=None,
                   help="Research override: a fixed gross leverage (e.g. 1.3)")
    p.add_argument("--margin-rate", type=float, default=None,
                   help="Annual interest on borrowed cash (default: config, 0.055)")
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
    p = live.add_parser("cycle", help="Refresh, sync, monitor, propose; send only if armed")
    p.add_argument("--no-refresh", action="store_true", help="Skip the data refresh")
    p.add_argument("--no-monitor", action="store_true", help="Skip the monitoring pass")
    p.set_defaults(func=cmd_live_cycle)
    p = live.add_parser("auto", help="Automatic sending: status, arm, disarm, schedule")
    p.add_argument("action", choices=["status", "arm", "disarm", "schedule"])
    p.add_argument("--scope", choices=["exits", "full"], default="exits")
    p.add_argument("--confirm", default=None, help="The arming phrase, instead of the prompt")
    p.add_argument("--override", default=None,
                   help="Arm full with live money without the exits-stage evidence (a reason)")
    p.add_argument("--reason", default=None, help="Why, when disarming")
    p.set_defaults(func=cmd_live_auto)
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
    if args.strategy:
        ctx.strategy = args.strategy
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
