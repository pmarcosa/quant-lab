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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """How rotation orders reach the market.

    ``auto`` sends them to the opening auction (OPG) for daily and weekly
    strategies -- the live counterpart of the backtest's next-open fill -- and
    as day market orders for intraday ones, where the next bar's open is not an
    auction.
    """

    time_in_force: str = "auto"

    def resolve(self, interval: BarInterval) -> TimeInForce:
        if self.time_in_force == "auto":
            return TimeInForce.DAY if interval.is_intraday else TimeInForce.OPG
        return TimeInForce(self.time_in_force)


@dataclass(frozen=True, slots=True)
class RiskSettings:
    stop_distance: float = 0.12
    max_gross: float = 1.0
    #: Net exposure band, longs minus shorts over equity. Binds only for
    #: strategies that can be short; for long-only ones the top equals gross.
    max_net: float = 1.0
    min_net: float = -1.0
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
        if self.account_scope not in ("dedicated", "shared"):
            raise ContractViolation("config: account_scope must be 'dedicated' or 'shared'")
        if self.execution.time_in_force not in ("auto", *(t.value for t in TimeInForce)):
            raise ContractViolation(
                "config: execution.time_in_force must be auto, opg, day or gtc"
            )
        r = self.risk
        if not 0 < r.stop_distance < 1 and r.stop_distance != 0:
            raise ContractViolation("config: risk.stop_distance must be in (0, 1), or 0")
        if not 0 < r.max_order_fraction <= 1:
            raise ContractViolation("config: risk.max_order_fraction must be in (0, 1]")
        if r.max_gross <= 0:
            raise ContractViolation("config: risk.max_gross must be positive")
        if not r.min_net <= r.max_net:
            raise ContractViolation("config: risk.min_net must not exceed risk.max_net")
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
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
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


def _strategy_settings(raw: Mapping[str, Any]) -> StrategySettings:
    """Accept ``{name, params}``, or the older flat momentum keys."""
    if "name" in raw or "params" in raw:
        return StrategySettings(
            name=str(raw.get("name", "weekly-momentum")), params=dict(raw.get("params") or {})
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
