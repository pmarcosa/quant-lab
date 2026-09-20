"""Addressable identifiers.

Every durable object in the system is named by a value object, never by a file
path. Today the store behind them is a folder; tomorrow it can be a database with
one row per tenant, and nothing above this module has to change.

The important one is :class:`StrategyVersion`. Parameters are part of a
strategy's identity, not a setting applied to it: a momentum strategy with a
13-week lookback and the same strategy with 26 weeks are two different things,
and validation evidence earned by one does not transfer to the other. Encoding
that in the identifier makes it impossible to lose track of.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from contracts.errors import ContractViolation

#: Identifiers are restricted to this shape so they are safe in a path, a URL and
#: a database key without escaping.
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def _check_slug(value: str, field: str) -> str:
    if not _SLUG.match(value):
        raise ContractViolation(
            f"{field} must match {_SLUG.pattern!r}; got {value!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class TenantId:
    """Who owns a portfolio. Today there is one; the code never assumes that."""

    value: str

    def __post_init__(self) -> None:
        _check_slug(self.value, "TenantId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class PortfolioId:
    """One book of positions, belonging to one tenant."""

    tenant: TenantId
    name: str

    def __post_init__(self) -> None:
        _check_slug(self.name, "PortfolioId.name")

    def __str__(self) -> str:
        return f"{self.tenant}/{self.name}"


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """A tradable thing, named by something that does not change.

    A ticker is not an identifier: companies rename, merge and relist, and a
    backtest keyed on today's tickers quietly rewrites history. This holds the
    durable key (IBKR contract id, FIGI, or an internal surrogate); the ticker is
    a point-in-time attribute that lives in the instrument's record.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or self.value.strip() != self.value:
            raise ContractViolation(f"InstrumentId must be non-empty and trimmed; got {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class StrategyId:
    """A family of strategies — the idea, not a particular parameterisation."""

    value: str

    def __post_init__(self) -> None:
        _check_slug(self.value, "StrategyId")

    def __str__(self) -> str:
        return self.value


def params_fingerprint(params: Mapping[str, Any]) -> str:
    """A stable short hash of a parameter set.

    Canonical JSON with sorted keys, so the same parameters always produce the
    same fingerprint regardless of insertion order, and any change to any value
    produces a different one.

    Args:
        params: The strategy's parameters. Must be JSON-serialisable.

    Raises:
        ContractViolation: If the parameters cannot be canonicalised, which means
            they carry something unhashable and the version could not be pinned.
    """
    try:
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=None)
    except TypeError as exc:
        raise ContractViolation(
            f"strategy parameters must be JSON-serialisable to be part of a version: {exc}"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class StrategyVersion:
    """A strategy family plus the exact parameters it runs with.

    Two versions differing in one parameter are two different strategies for every
    purpose that matters: validation evidence, the research ledger, the trial
    count behind a Deflated Sharpe Ratio, and what a report is allowed to claim.
    """

    strategy: StrategyId
    params_hash: str
    code_version: str = "dev"

    def __post_init__(self) -> None:
        if not re.match(r"^[0-9a-f]{12}$", self.params_hash):
            raise ContractViolation(
                f"params_hash must be a 12-character hex digest; got {self.params_hash!r}. "
                f"Build it with params_fingerprint()."
            )

    @classmethod
    def of(
        cls, strategy: StrategyId, params: Mapping[str, Any], code_version: str = "dev"
    ) -> StrategyVersion:
        """Build a version from a strategy and its parameters."""
        return cls(strategy=strategy, params_hash=params_fingerprint(params), code_version=code_version)

    def __str__(self) -> str:
        return f"{self.strategy}@{self.params_hash}"


@dataclass(frozen=True, slots=True)
class RunId:
    """One execution of the engine: a backtest, a paper session, a live session."""

    value: str

    def __post_init__(self) -> None:
        _check_slug(self.value, "RunId")

    def __str__(self) -> str:
        return self.value
