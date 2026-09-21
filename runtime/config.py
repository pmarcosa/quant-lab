"""Live configuration: one YAML file, validated before anything connects.

The file is ``configs/live.yaml`` and is gitignored, because it names a real
brokerage account. ``configs/live.example.yaml`` is the committed template.

Every setting that can put money at risk is checked here, at load, rather than
where it is used — so a mistake shows up as a refusal to start instead of as a
wrong order. The checks that matter most:

- **mode and port must agree.** Paper and live use different gateway ports; a
  config saying ``paper`` with a live port is refused.
- **mode and account must agree.** ``DU…`` is paper, ``U…`` is live.
- **the sleeve is bounded.** The strategy trades a declared amount of capital,
  not the whole account. Everything else in the account is invisible to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from contracts.errors import ContractViolation
from contracts.live import TradingMode

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "live.yaml"
STATE = ROOT / "state"

#: IB Gateway and TWS listen on different ports for paper and live. A config
#: whose mode and port disagree is almost always one pointed at the wrong
#: account.
PAPER_PORTS = frozenset({4002, 7497})
LIVE_PORTS = frozenset({4001, 7496})


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 17
    timeout_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class StrategySettings:
    rebalance_weeks: int = 4
    top_n: int = 4
    lookback_weeks: int = 13


@dataclass(frozen=True, slots=True)
class RiskSettings:
    stop_distance: float = 0.12
    max_gross: float = 1.0
    #: No single order may exceed this share of sleeve equity. A sanity bound,
    #: far above what the strategy ever asks for; it exists to stop a unit error
    #: from becoming an order.
    max_order_fraction: float = 0.6


@dataclass(frozen=True, slots=True)
class MonitoringSettings:
    """Thresholds for the degradation ladder, from the project's expert.

    Each pair is (move to reduce-only, move to halted). See
    ``validation.monitoring`` for what each statistic measures and why.
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
    #: The robust PnL trend is only judged after this many live weeks; before
    #: that a slope is noise, not evidence.
    trend_min_weeks: int = 26
    #: Stationary-bootstrap settings for the reference distributions.
    bootstrap_paths: int = 5000
    bootstrap_block_weeks: float = 6.0
    #: Expected weeks between regime changes, for the changepoint prior.
    changepoint_hazard_weeks: float = 250.0
    #: Bars older than this many days mean the data is stale.
    max_data_age_days: int = 10


@dataclass(frozen=True, slots=True)
class LiveConfig:
    mode: TradingMode
    account: str
    sleeve_capital: float
    gateway: GatewaySettings = field(default_factory=GatewaySettings)
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    #: Hours a proposal stays approvable. Orders go to the opening auction, so a
    #: proposal made on Saturday must still be valid on Monday morning.
    proposal_ttl_hours: float = 60.0
    #: Tickers held in the account that the strategy must never touch.
    unmanaged: frozenset[str] = frozenset()
    state_dir: Path = STATE

    @property
    def journal_path(self) -> Path:
        return self.state_dir / "live" / f"{self.mode.value}-journal.jsonl"

    @property
    def reports_dir(self) -> Path:
        return self.state_dir / "reports"

    def validate(self) -> LiveConfig:
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
        if not 0 < self.risk.stop_distance < 1 and self.risk.stop_distance != 0:
            raise ContractViolation("config: risk.stop_distance must be in (0, 1), or 0")
        if not 0 < self.risk.max_order_fraction <= 1:
            raise ContractViolation("config: risk.max_order_fraction must be in (0, 1]")
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
        return self


def load_config(path: Path | None = None) -> LiveConfig:
    """Read and validate the live configuration.

    Raises:
        ContractViolation: If the file is missing or any setting is unsafe.
    """
    source = Path(path) if path else DEFAULT_CONFIG
    if not source.exists():
        raise ContractViolation(
            f"no config at {source}. Copy configs/live.example.yaml to configs/live.yaml "
            f"and fill in your account."
        )
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    try:
        config = LiveConfig(
            mode=TradingMode(str(raw["mode"]).lower()),
            account=str(raw["account"]).strip(),
            sleeve_capital=float(raw["sleeve_capital"]),
            gateway=GatewaySettings(**(raw.get("gateway") or {})),
            strategy=StrategySettings(**(raw.get("strategy") or {})),
            risk=RiskSettings(**(raw.get("risk") or {})),
            monitoring=MonitoringSettings(**(raw.get("monitoring") or {})),
            proposal_ttl_hours=float(raw.get("proposal_ttl_hours", 60.0)),
            unmanaged=frozenset(s.upper() for s in raw.get("unmanaged") or ()),
            state_dir=Path(raw.get("state_dir") or STATE),
        )
    except KeyError as missing:
        raise ContractViolation(f"config: missing required setting {missing}") from missing
    except (TypeError, ValueError) as error:
        raise ContractViolation(f"config: {error}") from error
    return config.validate()
