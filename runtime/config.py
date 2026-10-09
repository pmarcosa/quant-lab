"""Live configuration: one YAML file per strategy, validated before anything connects.

Each deployed strategy has its own file, ``configs/strategies/<id>.yaml``, and
its own IBKR account. The files are gitignored because they name real brokerage
accounts; ``configs/live.example.yaml`` is the committed template.
(``configs/live.yaml``, the single-file layout from before strategies had ids,
is still read.)

Every setting that can put money at risk is checked here, at load, rather than
where it is used -- so a mistake shows up as a refusal to start instead of as a
wrong order. The checks that matter most:

- **every strategy has an id**, and the id is on every order it sends, in its
  journal and in its state folder. Two strategies can never claim each other's
  fills or share a journal.
- **mode and port must agree.** Paper and live use different gateway ports; a
  config saying ``paper`` with a live port is refused.
- **mode and account must agree.** ``DU…`` is paper, ``U…`` is live.
- **one strategy per account** (:func:`check_accounts`). Several strategies in
  one account need virtual sub-portfolios with netting and pro-rata fills,
  which are not built; separate linked accounts are the chosen design.
- **the sleeve is bounded.** The strategy trades a declared amount of capital.
- **leverage is capped twice.** ``leverage.target`` may not exceed
  ``risk.max_gross``, and the gross-exposure rule enforces the cap on every
  decision whatever the leverage schedule asks for.
- **automation is opt-in, twice.** ``automation.mode`` is the most the system
  may do without a person; nothing is automatic until a person also *arms* it
  with a typed phrase (``ql live auto arm``).

**One definition, several users.** What a strategy *is* -- its rules, how it
trades, its risk limits and monitoring thresholds -- can live in a committed
file, ``configs/definitions/<name>.yaml``, with no account in it. A private
config takes it in with ``extends:`` and adds the account; the weekly review
(``ql review``), which runs where no private config exists, reads the
definition directly. Both then run the same strategy by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from contracts.errors import ContractViolation
from contracts.execution import MAX_TAG_LENGTH, TimeInForce, is_valid_strategy_id
from contracts.live import TradingMode
from contracts.temporal import BarInterval

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
STRATEGY_CONFIGS = CONFIGS / "strategies"
DEFINITIONS = CONFIGS / "definitions"
LEGACY_CONFIG = CONFIGS / "live.yaml"
DEFAULT_CONFIG = LEGACY_CONFIG
STATE = ROOT / "state"

#: IB Gateway and TWS listen on different ports for paper and live. A config
#: whose mode and port disagree is almost always one pointed at the wrong
#: account.
PAPER_PORTS = frozenset({4002, 7497})
LIVE_PORTS = frozenset({4001, 7496})

#: How old the latest complete bar may be before proposing is refused, by bar
#: size. Weekly: a missed Saturday refresh. Daily: a long weekend plus a
#: holiday. Intraday cannot be judged properly without a session calendar,
#: which is not built; three days only catches a refresh that stopped.
DEFAULT_DATA_AGE_HOURS = {
    BarInterval.WEEK: 240.0,
    BarInterval.DAY: 100.0,
    BarInterval.HOUR: 72.0,
    BarInterval.MINUTE: 72.0,
}


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 17
    timeout_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class StrategySettings:
    """Which strategy, with which parameters.

    ``name`` is a key of ``runtime.strategies.STRATEGIES``. The parameters are
    the strategy's own and are passed through untouched; they are part of its
    version, so changing one deploys a different, unvalidated strategy.
    """

    name: str = "weekly-momentum"
    params: Mapping[str, Any] = field(
        default_factory=lambda: {"rebalance_weeks": 4, "top_n": 4, "lookback_weeks": 13}
    )
    #: A named list of symbols (``data/universes/<name>.txt``) or a path to one.
    #: Unset: every instrument in the store, so fetching a symbol widens it.
    universe: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """How rotation orders reach the market.

    ``auto`` sends them to the opening auction (OPG) for daily and weekly
    strategies -- the live counterpart of the backtest's next-open fill -- and
    as day market orders for intraday ones, where the next bar's open is not an
    auction.

    The rest says how weights become orders and what an order costs. It is read
    by the backtest, by the monitoring baseline and by the live proposal alike,
    so the three describe one way of trading: a strategy sized one way live and
    another in its baseline would be compared with something it is not.
    """

    time_in_force: str = "auto"
    #: Fraction of equity held back from sizing (``SizingPolicy.cash_buffer``).
    cash_buffer: float = 0.01
    #: Adjustments worth less than this fraction of equity are not sent.
    no_trade_band: float = 0.005
    #: Send limit orders this far through the decision price instead of market
    #: orders. ``None``: market orders.
    limit_band: float | None = None
    #: ``bps``: ``commission_bps`` of each order's value. ``ibkr-tiered`` or
    #: ``ibkr-fixed``: that plan, in dollars per order.
    commission: str = "bps"
    commission_bps: float = 10.0
    slippage_bps: float = 10.0
    #: With a plan: the typical share price per-share fees are charged at, since
    #: stored history is split-adjusted. 0 charges on the stored quantity.
    share_price: float = 100.0

    def resolve(self, interval: BarInterval) -> TimeInForce:
        if self.time_in_force == "auto":
            return TimeInForce.DAY if interval.is_intraday else TimeInForce.OPG
        return TimeInForce(self.time_in_force)

    def sizing(self, interval: BarInterval, **more):
        """The sizing rules these settings ask for; ``more`` adds or overrides."""
        from engine.decide import SizingPolicy

        return SizingPolicy(**{
            "cash_buffer": self.cash_buffer, "min_trade_fraction": self.no_trade_band,
            "limit_band": self.limit_band, "time_in_force": self.resolve(interval), **more,
        })

    def costs(self):
        """The cost model these settings ask for."""
        from execution.simulated import COMMISSION_PLANS, CostModel

        plan = None
        if self.commission != "bps":
            plan = COMMISSION_PLANS[self.commission].with_reference_price(
                self.share_price if self.share_price > 0 else None
            )
        return CostModel(commission_bps=self.commission_bps, slippage_bps=self.slippage_bps,
                         schedule=plan)


@dataclass(frozen=True, slots=True)
class RiskSettings:
    stop_distance: float = 0.12
    #: Make the stop a stop-limit that sells no lower than this fraction under
    #: the stop (0.005: half a percent). ``None``: a plain stop, which always
    #: fills; a stop-limit does not when the price gaps through its limit.
    stop_limit_offset: float | None = None

    def stop(self):
        """The protective stop these settings ask for, or None when it is off."""
        from risk.rules import ProtectiveStop

        if self.stop_distance <= 0:
            return None
        return ProtectiveStop(distance=self.stop_distance, limit_offset=self.stop_limit_offset)
    max_gross: float = 1.0
    #: Net exposure band, longs minus shorts over equity. ``max_net`` empty
    #: means "the same as ``max_gross``": for a long-only book net *is* gross,
    #: so a separate, lower cap would silently forbid the leverage the gross
    #: cap allows.
    max_net: float | None = None
    min_net: float = -1.0

    @property
    def net_cap(self) -> float:
        return self.max_gross if self.max_net is None else self.max_net
    #: Short sales are refused unless this is on. The account must also be a
    #: margin account; IBKR refuses shorts in a cash account.
    allow_short: bool = False
    #: Refuse new shorts whose annual borrow fee is above this, when the broker
    #: reports a fee. ``None`` disables the fee check.
    max_borrow_fee: float | None = None
    #: No single order may exceed this share of sleeve equity. A sanity bound,
    #: far above what the strategy ever asks for; it exists to stop a unit error
    #: from becoming an order.
    max_order_fraction: float = 0.6


@dataclass(frozen=True, slots=True)
class LeverageSettings:
    """How much of the strategy's intent to hold, above or at 100% gross.

    Defaults hold exactly the strategy's weights: no borrowing. See
    ``risk.leverage.LeverageSchedule`` for each setting and where it comes from.
    The hard cap is ``risk.max_gross``, not a setting here.
    """

    target: float = 1.0
    cvar_target: float | None = None
    drawdown_start: float = 0.10
    drawdown_full: float = 0.35
    convexity: float = 2.0
    step_up_per_week: float = 0.05
    floor: float = 1.0
    cushion_warning: float = 0.35
    cushion_critical: float = 0.25

    def schedule(self, maximum: float):
        from risk.leverage import LeverageSchedule

        return LeverageSchedule(
            target=self.target, maximum=maximum, cvar_target=self.cvar_target,
            drawdown_start=self.drawdown_start, drawdown_full=self.drawdown_full,
            convexity=self.convexity, step_up_per_week=self.step_up_per_week,
            floor=min(self.floor, maximum), cushion_warning=self.cushion_warning,
            cushion_critical=self.cushion_critical,
        )

    @property
    def borrows(self) -> bool:
        """Whether this can ever ask for more than 100% gross."""
        return self.target > 1.0 or self.cvar_target is not None


@dataclass(frozen=True, slots=True)
class FinancingSettings:
    """Annual carrying costs, charged in backtests and accrued in the live sleeve.

    ``margin_rate``: interest on borrowed cash. IBKR Pro, USD, first tier, was
    5.38% (benchmark + 1.5%) in September 2026; check the current rate and
    your tier. ``borrow_fee``: the fee on shorted stock, a general-collateral
    assumption -- hard-to-borrow names cost far more.
    """

    margin_rate: float = 0.055
    borrow_fee: float = 0.005

    def model(self):
        from engine.financing import FinancingModel

        return FinancingModel(margin_rate=self.margin_rate, borrow_fee=self.borrow_fee)


#: What automation may do, in increasing order. The project's expert: automate
#: exits first, then everything, and only on evidence from the first stage.
AUTOMATION_MODES = ("manual", "exits", "full")


@dataclass(frozen=True, slots=True)
class AutomationSettings:
    """The most the system may do without a person, and the gates it must pass.

    ``mode`` is a ceiling, not a switch: nothing runs unattended until a person
    arms it (``ql live auto arm``), and a halt disarms it. Every threshold here
    is from the project's expert, adapted where noted in ``runtime/automation.py``.
    """

    mode: str = "manual"
    #: No automatic order may open more than this share of equity. Empty: 10%
    #: more than the largest order the baseline backtest ever sent. The expert
    #: suggests 5% for strategies that slice orders; a concentrated book opens
    #: 25-50% positions by design, so the bound follows what the strategy does.
    max_order_fraction: float | None = None
    #: A cycle whose turnover exceeds this percentile of the backtest's
    #: rotations is held for a person rather than sent.
    turnover_percentile: float = 0.99
    #: At most this many orders in one automatic cycle.
    max_orders: int = 20
    #: A bar whose sleeve return is below this percentile of the backtest's
    #: per-bar returns halts the system -- the loss circuit breaker.
    loss_breaker_percentile: float = 0.005
    #: No automatic entries below this margin cushion.
    min_cushion: float = 0.35
    #: Evidence required in live mode before arming ``full``: this many weeks
    #: armed for ``exits``, with no reconciliation mismatch, and exit
    #: shortfall at most this multiple of the modelled cost.
    graduation_weeks: float = 8.0
    graduation_shortfall_multiple: float = 1.2


@dataclass(frozen=True, slots=True)
class MonitoringSettings:
    """Thresholds for the degradation ladder, from the project's expert.

    Each pair is (move to reduce-only, move to halted). See
    ``validation.monitoring`` for what each statistic measures and why.

    **Durations are in calendar weeks, whatever the strategy's bar size.** The
    expert's rule for changing frequency: the market's memory and the expected
    time between regimes are properties of calendar time, so they are fixed in
    calendar time and translated into bars (``BarInterval.bars_per_week``). A
    six-week bootstrap block is six bars for a weekly strategy and thirty for a
    daily one; the setting does not change.
    """

    #: Percentile of the bootstrapped drawdown distribution for a window the
    #: same length as the live record. The expert gives the 80th and 95th, with
    #: the 99th as the high-tolerance alternative for halting. Measured on
    #: healthy synthetic data (tests/test_monitoring.py), halting at the 95th
    #: stopped a working system in 5-13% of weekly assessments -- within most
    #: years, run weekly. At the 99th it is 0-5%, while a real collapse is still
    #: halted in most cases within a quarter. With a declared drawdown tolerance
    #: of 30-40%, the 99th is the default.
    reduce_percentile: float = 0.80
    halt_percentile: float = 0.99
    #: Posterior probability, from online changepoint detection, that the
    #: return process has changed during live trading.
    reduce_break_probability: float = 0.20
    halt_break_probability: float = 0.50
    #: Implementation shortfall as a multiple of what the backtest modelled.
    reduce_shortfall_multiple: float = 1.5
    #: Shortfall consuming more than this share of the expected gross return per
    #: rotation, for ``halt_shortfall_cycles`` consecutive rotations, halts.
    halt_shortfall_alpha_share: float = 0.5
    halt_shortfall_cycles: int = 2
    #: The robust PnL trend is only judged after this many weeks of live
    #: record; before that a slope is noise, not evidence.
    trend_min_weeks: float = 26
    #: Longest live window compared with the bootstrap, in weeks.
    horizon_weeks: float = 52
    #: Stationary-bootstrap settings for the reference distributions.
    bootstrap_paths: int = 5000
    bootstrap_block_weeks: float = 6.0
    #: Expected weeks between regime changes, for the changepoint prior.
    changepoint_hazard_weeks: float = 250.0
    #: The latest complete bar may not be older than this when proposing.
    #: ``None``: by bar size (``DEFAULT_DATA_AGE_HOURS``).
    max_data_age_hours: float | None = None

    def data_age_hours(self, interval: BarInterval) -> float:
        if self.max_data_age_hours is not None:
            return float(self.max_data_age_hours)
        return DEFAULT_DATA_AGE_HOURS[interval]


@dataclass(frozen=True, slots=True)
class ReviewSettings:
    """The weekly review from broker-connector data (``ql review``).

    The review proposes; a person approves each order at the broker. So its
    orders are day limit orders a little through a reference price, where the
    gateway path sends market orders to the opening auction.
    """

    #: A buy is limited to the reference price plus this fraction, a sell to
    #: the price less it: marketable, but not at any price.
    limit_offset: float = 0.005
    #: The Monday of the first rotation traded on this strategy version
    #: (``YYYY-MM-DD``). The live record monitoring judges starts there: weeks
    #: traded under other rules belong to another strategy. ``None``: no live
    #: record yet, so monitoring reports and judges nothing.
    monitor_from: str | None = None
    #: Live weeks before the drawdown and changepoint checks may move the
    #: state. They are computed and shown from the first week; the project's
    #: expert asks for a burn-in before acting on them, and for the execution
    #: cost check and the loss breaker to act from the start.
    burn_in_weeks: int = 12
    #: The benchmark reported beside the account.
    benchmark: str = "SPY"
    #: A date (``YYYY-MM-DD``) the report also measures the account and the
    #: benchmark from, besides the last week and the live record. Optional.
    performance_from: str | None = None


@dataclass(frozen=True, slots=True)
class LiveConfig:
    strategy_id: str
    mode: TradingMode
    account: str
    sleeve_capital: float
    gateway: GatewaySettings = field(default_factory=GatewaySettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    leverage: LeverageSettings = field(default_factory=LeverageSettings)
    financing: FinancingSettings = field(default_factory=FinancingSettings)
    automation: AutomationSettings = field(default_factory=AutomationSettings)
    review: ReviewSettings = field(default_factory=ReviewSettings)
    #: Hours a proposal stays approvable. Orders go to the opening auction, so a
    #: proposal made on Saturday must still be valid on Monday morning. Approval
    #: is also refused as soon as a newer bar exists, whatever this says.
    proposal_ttl_hours: float = 60.0
    #: ``dedicated``: the account belongs to this strategy, and anything else
    #: in it -- a position the sleeve does not hold, extra shares -- stops the
    #: system. ``shared``: other holdings are expected and only reported.
    account_scope: str = "dedicated"
    #: Tickers held in the account that the strategy must never touch.
    unmanaged: frozenset[str] = frozenset()
    state_dir: Path = STATE
    source: Path | None = None

    @property
    def live_dir(self) -> Path:
        """This strategy's own folder: journal, baseline."""
        return self.state_dir / "live" / self.strategy_id

    @property
    def journal_path(self) -> Path:
        return self.live_dir / f"{self.mode.value}-journal.jsonl"

    @property
    def baseline_path(self) -> Path:
        return self.live_dir / f"{self.mode.value}-baseline.json"

    @property
    def reports_dir(self) -> Path:
        return self.state_dir / "reports" / self.strategy_id

    def validate(self) -> LiveConfig:
        if not is_valid_strategy_id(self.strategy_id):
            raise ContractViolation(
                f"config: strategy_id {self.strategy_id!r} must be 1-{MAX_TAG_LENGTH} "
                f"characters of lowercase letters, digits and single dashes, starting "
                f"with a letter (it is written on every order sent to IBKR)"
            )
        if not self.account:
            raise ContractViolation("config: account is required")
        if not self.mode.admits(self.account):
            raise ContractViolation(
                f"config: account {self.account} does not look like a {self.mode.value} "
                f"account (paper accounts start with DU, live with U)"
            )
        port = self.gateway.port
        if self.mode is TradingMode.PAPER and port in LIVE_PORTS:
            raise ContractViolation(
                f"config: mode is paper but port {port} is a live port"
            )
        if self.mode is TradingMode.LIVE and port in PAPER_PORTS:
            raise ContractViolation(
                f"config: mode is live but port {port} is a paper port"
            )
        if self.sleeve_capital <= 0:
            raise ContractViolation("config: sleeve_capital must be positive")
        if self.strategy.universe:
            from runtime.wiring import universe_list

            universe_list(self.strategy.universe)  # refuses a name that resolves to nothing
        if self.account_scope not in ("dedicated", "shared"):
            raise ContractViolation("config: account_scope must be 'dedicated' or 'shared'")
        if self.execution.time_in_force not in ("auto", *(t.value for t in TimeInForce)):
            raise ContractViolation(
                "config: execution.time_in_force must be auto, opg, day or gtc"
            )
        e = self.execution
        if e.commission not in ("bps", "ibkr-tiered", "ibkr-fixed"):
            raise ContractViolation(
                "config: execution.commission must be bps, ibkr-tiered or ibkr-fixed"
            )
        if not 0 <= e.cash_buffer < 1 or e.no_trade_band < 0:
            raise ContractViolation(
                "config: execution.cash_buffer must be in [0, 1) and no_trade_band not negative"
            )
        if e.limit_band is not None and not 0 <= e.limit_band < 1:
            raise ContractViolation("config: execution.limit_band must be in [0, 1), or null")
        if e.commission_bps < 0 or e.slippage_bps < 0 or e.share_price < 0:
            raise ContractViolation("config: execution costs cannot be negative")
        r = self.risk
        if not 0 < r.stop_distance < 1 and r.stop_distance != 0:
            raise ContractViolation("config: risk.stop_distance must be in (0, 1), or 0")
        if r.stop_limit_offset is not None and not 0 <= r.stop_limit_offset < 1:
            raise ContractViolation("config: risk.stop_limit_offset must be in [0, 1), or null")
        if not 0 < r.max_order_fraction <= 1:
            raise ContractViolation("config: risk.max_order_fraction must be in (0, 1]")
        if r.max_gross <= 0:
            raise ContractViolation("config: risk.max_gross must be positive")
        if not r.min_net <= r.net_cap:
            raise ContractViolation("config: risk.min_net must not exceed risk.max_net")
        lv = self.leverage
        if lv.target > r.max_gross:
            raise ContractViolation(
                f"config: leverage.target {lv.target} is above risk.max_gross {r.max_gross}, "
                f"the hard cap; raise the cap deliberately or lower the target"
            )
        if lv.borrows and r.max_gross <= 1.0:
            raise ContractViolation(
                "config: leverage asks to borrow but risk.max_gross is 1.0, which forbids it"
            )
        if lv.borrows and not r.allow_short and r.net_cap < r.max_gross:
            raise ContractViolation(
                f"config: risk.max_net {r.net_cap} is below risk.max_gross {r.max_gross}; for a "
                f"long-only book net equals gross, so the lower cap would forbid the leverage. "
                f"Remove risk.max_net or raise it."
            )
        try:
            lv.schedule(r.max_gross)
            self.financing.model()
        except ContractViolation as error:
            raise ContractViolation(f"config: {error}") from error
        a = self.automation
        if a.mode not in AUTOMATION_MODES:
            raise ContractViolation(
                f"config: automation.mode must be one of {', '.join(AUTOMATION_MODES)}"
            )
        if a.max_order_fraction is not None and not 0 < a.max_order_fraction <= r.max_order_fraction:
            raise ContractViolation(
                "config: automation.max_order_fraction must be in (0, risk.max_order_fraction]"
            )
        if not 0.5 < a.turnover_percentile <= 1.0 or not 0 < a.loss_breaker_percentile < 0.5:
            raise ContractViolation("config: automation percentiles are out of range")
        m = self.monitoring
        if not 0.5 < m.reduce_percentile < m.halt_percentile < 1:
            raise ContractViolation(
                "config: monitoring percentiles must satisfy 0.5 < reduce < halt < 1"
            )
        if not 0 < m.reduce_break_probability < m.halt_break_probability < 1:
            raise ContractViolation(
                "config: break probabilities must satisfy 0 < reduce < halt < 1"
            )
        if m.bootstrap_paths < 500:
            raise ContractViolation("config: bootstrap_paths below 500 gives a noisy percentile")
        if self.proposal_ttl_hours <= 0:
            raise ContractViolation("config: proposal_ttl_hours must be positive")
        v = self.review
        if not 0 <= v.limit_offset < 0.1 or v.burn_in_weeks < 0:
            raise ContractViolation(
                "config: review.limit_offset must be in [0, 0.1) and burn_in_weeks not negative"
            )
        for label, day in (("monitor_from", v.monitor_from),
                           ("performance_from", v.performance_from)):
            if day is not None:
                try:
                    date.fromisoformat(str(day))
                except ValueError as error:
                    raise ContractViolation(
                        f"config: review.{label} must be a date, YYYY-MM-DD; got {day!r}"
                    ) from error
        from runtime.strategies import build_strategy  # validates name and params

        build_strategy(self.strategy.name, self.strategy.params)
        return self


# -- loading -------------------------------------------------------------------------


def load_config(path: Path | None = None) -> LiveConfig:
    """Read and validate one strategy's live configuration.

    Raises:
        ContractViolation: If the file is missing or any setting is unsafe.
    """
    source = Path(path) if path else DEFAULT_CONFIG
    if not source.exists():
        raise ContractViolation(
            f"no config at {source}. Copy configs/live.example.yaml to "
            f"configs/strategies/<strategy_id>.yaml and fill in the account."
        )
    raw = _read(source)
    _refuse_unknown_keys(raw, source)
    try:
        strategy_id = str(raw.get("strategy_id") or "").strip()
        if not strategy_id:
            raise ContractViolation(
                f"config {source.name}: strategy_id is required -- a short id such as "
                f"'momentum' that is written on every order this strategy sends"
            )
        config = LiveConfig(
            strategy_id=strategy_id,
            mode=TradingMode(str(raw["mode"]).lower()),
            account=str(raw["account"]).strip().upper(),
            sleeve_capital=float(raw["sleeve_capital"]),
            gateway=GatewaySettings(**(raw.get("gateway") or {})),
            strategy=_strategy_settings(raw.get("strategy") or {}),
            execution=ExecutionSettings(**(raw.get("execution") or {})),
            risk=RiskSettings(**(raw.get("risk") or {})),
            monitoring=MonitoringSettings(**_monitoring(raw.get("monitoring") or {})),
            leverage=LeverageSettings(**(raw.get("leverage") or {})),
            financing=FinancingSettings(**(raw.get("financing") or {})),
            automation=AutomationSettings(**(raw.get("automation") or {})),
            review=_review(raw.get("review") or {}),
            proposal_ttl_hours=float(raw.get("proposal_ttl_hours", 60.0)),
            account_scope=str(raw.get("account_scope", "dedicated")).lower(),
            unmanaged=frozenset(s.upper() for s in raw.get("unmanaged") or ()),
            state_dir=Path(raw.get("state_dir") or STATE),
            source=source,
        )
    except KeyError as missing:
        raise ContractViolation(f"config: missing required setting {missing}") from missing
    except (TypeError, ValueError) as error:
        raise ContractViolation(f"config: {error}") from error
    return config.validate()


#: Every top-level key a config may have, and the dataclass behind each section.
_SECTIONS = {
    "gateway": GatewaySettings, "strategy": StrategySettings, "execution": ExecutionSettings,
    "risk": RiskSettings, "monitoring": MonitoringSettings, "leverage": LeverageSettings,
    "financing": FinancingSettings, "automation": AutomationSettings, "review": ReviewSettings,
}
_TOP_LEVEL = frozenset({
    "strategy_id", "mode", "account", "sleeve_capital", "proposal_ttl_hours",
    "account_scope", "unmanaged", "state_dir", "extends", *_SECTIONS,
})
#: What a definition may not carry: it is committed, and these name an account.
_PRIVATE = frozenset({"mode", "account", "sleeve_capital", "gateway", "state_dir", "extends"})


def _read(source: Path) -> dict[str, Any]:
    """One config file as a mapping, with the definition it ``extends`` under it.

    ``extends`` names a file relative to the one that has it. The definition's
    settings are the base and the file's own are laid over them: section by
    section, key by key, and for the strategy parameter by parameter. A config
    that extends a definition therefore restates nothing, and one that overrides
    a rule says so in one visible line.
    """
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base_path = (source.parent / str(parent)).resolve()
    if not base_path.exists():
        raise ContractViolation(f"config {source.name}: extends {parent!r}, which is not there")
    base: dict[str, Any] = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if "extends" in base:
        raise ContractViolation(
            f"config {source.name}: {base_path.name} extends another file; one level only"
        )
    carried = sorted(set(base) & _PRIVATE)
    if carried:
        raise ContractViolation(
            f"definition {base_path.name} carries {carried}; a definition names no account"
        )
    return _overlay(base, raw)


def _overlay(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _overlay(merged[key], value)
        else:
            merged[key] = value
    return merged


def _review(raw: Mapping[str, Any]) -> ReviewSettings:
    values = dict(raw)
    for key in ("monitor_from", "performance_from"):  # YAML reads a bare date as a date
        if values.get(key) is not None:
            values[key] = str(values[key])
    return ReviewSettings(**values)


@dataclass(frozen=True, slots=True)
class Definition:
    """What a strategy is, with no account: rules, trading, limits, thresholds.

    The part of a config that can be published. ``ql review`` runs from one.
    """

    strategy_id: str
    strategy: StrategySettings
    execution: ExecutionSettings
    risk: RiskSettings
    monitoring: MonitoringSettings
    automation: AutomationSettings
    review: ReviewSettings
    unmanaged: frozenset[str] = frozenset()
    source: Path | None = None


def load_definition(path: Path) -> Definition:
    """Read a committed definition, checked as strictly as a live config.

    The checks are the live config's own, run on the definition with a stand-in
    paper account, so a setting refused there is refused here too.

    Raises:
        ContractViolation: If the file is missing, names an account, or holds
            a setting a live config would refuse.
    """
    source = Path(path)
    if not source.exists():
        known = sorted(p.stem for p in DEFINITIONS.glob("*.yaml")) if DEFINITIONS.is_dir() else []
        raise ContractViolation(
            f"no definition at {source}; known: {', '.join(known) or 'none'} (in {DEFINITIONS})"
        )
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    _refuse_unknown_keys(raw, source)
    carried = sorted(set(raw) & _PRIVATE)
    if carried:
        raise ContractViolation(
            f"definition {source.name} carries {carried}; a definition names no account"
        )
    strategy_id = str(raw.get("strategy_id") or "").strip()
    if not strategy_id:
        raise ContractViolation(f"definition {source.name}: strategy_id is required")
    try:
        checked = LiveConfig(
            strategy_id=strategy_id, mode=TradingMode.PAPER, account="DU0000000",
            sleeve_capital=1.0,
            strategy=_strategy_settings(raw.get("strategy") or {}),
            execution=ExecutionSettings(**(raw.get("execution") or {})),
            risk=RiskSettings(**(raw.get("risk") or {})),
            monitoring=MonitoringSettings(**_monitoring(raw.get("monitoring") or {})),
            leverage=LeverageSettings(**(raw.get("leverage") or {})),
            financing=FinancingSettings(**(raw.get("financing") or {})),
            automation=AutomationSettings(**(raw.get("automation") or {})),
            review=_review(raw.get("review") or {}),
            unmanaged=frozenset(s.upper() for s in raw.get("unmanaged") or ()),
            source=source,
        ).validate()
    except (TypeError, ValueError) as error:
        raise ContractViolation(f"definition {source.name}: {error}") from error
    return Definition(
        strategy_id=strategy_id, strategy=checked.strategy, execution=checked.execution,
        risk=checked.risk, monitoring=checked.monitoring, automation=checked.automation,
        review=checked.review, unmanaged=checked.unmanaged, source=source,
    )


def _refuse_unknown_keys(raw: Mapping[str, Any], source: Path) -> None:
    """A key the loader does not read is refused, never ignored.

    An ignored key is a setting the user believes is in force and is not: a
    universe written at the top level instead of under ``strategy:`` once let a
    backtest choose from the whole store while its config said otherwise.
    """
    from dataclasses import fields

    unknown = sorted(set(raw) - _TOP_LEVEL)
    if not unknown:
        return
    hints = []
    for key in unknown:
        homes = [name for name, cls in _SECTIONS.items()
                 if key in {f.name for f in fields(cls)}]
        hints.append(f"{key!r} (did you mean it under {homes[0]}:?)" if homes else repr(key))
    raise ContractViolation(
        f"config {source.name}: unknown setting(s) {', '.join(hints)}. A setting the "
        f"loader does not read would be silently ignored, so it is refused. Section "
        f"settings are indented under their section, e.g.\n"
        f"  strategy:\n    name: weekly-momentum\n    universe: <name>"
    )


def _strategy_settings(raw: Mapping[str, Any]) -> StrategySettings:
    """Accept ``{name, params}``, or the older flat momentum keys."""
    if "name" in raw or "params" in raw or "universe" in raw:
        unknown = set(raw) - {"name", "params", "universe"}
        if unknown:
            raise ContractViolation(f"config: unknown strategy settings {sorted(unknown)}")
        universe = raw.get("universe")
        return StrategySettings(
            name=str(raw.get("name", "weekly-momentum")), params=dict(raw.get("params") or {}),
            universe=str(universe) if universe else None,
        )
    legacy = {k: v for k, v in raw.items() if k in ("rebalance_weeks", "top_n", "lookback_weeks")}
    unknown = set(raw) - set(legacy)
    if unknown:
        raise ContractViolation(
            f"config: unknown strategy settings {sorted(unknown)}; use strategy.name "
            f"and strategy.params"
        )
    return StrategySettings(params={**StrategySettings().params, **legacy})


def _monitoring(raw: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(raw)
    if "max_data_age_days" in values:  # the older spelling
        days = values.pop("max_data_age_days")
        values.setdefault("max_data_age_hours", None if days is None else float(days) * 24)
    return values


def config_paths(root: Path = CONFIGS) -> list[Path]:
    """Every strategy config: ``configs/strategies/*.yaml`` plus a legacy ``live.yaml``."""
    found = sorted((root / "strategies").glob("*.yaml")) if (root / "strategies").is_dir() else []
    legacy = root / "live.yaml"
    if legacy.exists():
        found.append(legacy)
    return found


def check_accounts(configs: Iterable[LiveConfig]) -> None:
    """Refuse two strategies with the same id, or two in one account, per mode.

    The same id may appear once per mode: a strategy's paper and live configs
    share it, and their journals, accounts and orders are already apart.

    One account per strategy is the design: IBKR then keeps each strategy's
    positions, cash and orders apart natively. Two strategies sharing an
    account would need the virtual sub-portfolio machinery -- netting of
    opposite orders, pro-rata allocation of fills, reconciliation against the
    sum -- that is deliberately not built.
    """
    by_id: dict[tuple[str, str], Path | None] = {}
    by_account: dict[tuple[str, str], str] = {}
    for config in configs:
        ident = (config.mode.value, config.strategy_id)
        if ident in by_id:
            raise ContractViolation(
                f"strategy id {config.strategy_id!r} is used by two {config.mode.value} "
                f"configs ({by_id[ident]} and {config.source}); ids must be unique"
            )
        by_id[ident] = config.source
        key = (config.mode.value, config.account)
        other = by_account.get(key)
        if other is not None:
            raise ContractViolation(
                f"strategies {other!r} and {config.strategy_id!r} both use account "
                f"{config.account} in {config.mode.value} mode. Each strategy needs its "
                f"own IBKR account (open a linked account); sharing one is not supported."
            )
        by_account[key] = config.strategy_id


def interval_of(config: LiveConfig) -> BarInterval:
    """The strategy's bar size, from the strategy itself."""
    from runtime.strategies import build_strategy

    return build_strategy(config.strategy.name, config.strategy.params).filtration_spec.interval
