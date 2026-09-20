"""Who is asking, what they may run, and how much discretion they hold.

Today there is one user, one portfolio and no licences. This module exists
anyway, because the alternative is a system that assumes a single implicit user —
and retrofitting identity into that touches every file that reads configuration
or writes a result.

The interface asks three questions: who are you, what have you got, and may you
do this. The local implementation answers "the user", "everything", "yes" in about
thirty lines. A later implementation can answer from a subscription service
without anything above this module changing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from contracts.identifiers import PortfolioId, StrategyId, StrategyVersion, TenantId


class DiscretionMode(str, Enum):
    """Who makes the final call on each order.

    ``PROPOSE_AND_APPROVE``
        The system proposes; a human approves each order before it is sent.
    ``AUTOMATIC``
        The system sends orders without per-order human action.

    This is a first-class setting rather than a code path because both modes must
    exist and be switchable per portfolio. It is also the boundary that decides
    what kind of service is being provided, so it is recorded on every order and
    every report rather than being implicit in how the process was launched.
    """

    PROPOSE_AND_APPROVE = "propose_and_approve"
    AUTOMATIC = "automatic"


@dataclass(frozen=True, slots=True)
class Entitlement:
    """Permission for one tenant to run one strategy on one portfolio."""

    portfolio: PortfolioId
    strategy: StrategyId
    discretion: frozenset[DiscretionMode]

    def permits(self, mode: DiscretionMode) -> bool:
        """Whether this entitlement covers the requested discretion mode."""
        return mode in self.discretion


@runtime_checkable
class AccessPort(Protocol):
    """Identity, entitlements and authorisation. The product seam."""

    def principal(self) -> TenantId:
        """Who this session acts as."""
        ...

    def portfolios(self) -> Sequence[PortfolioId]:
        """Portfolios the principal may operate."""
        ...

    def entitlements(self, portfolio: PortfolioId) -> Sequence[Entitlement]:
        """What the principal may run on this portfolio."""
        ...

    def authorise(
        self,
        portfolio: PortfolioId,
        version: StrategyVersion,
        discretion: DiscretionMode,
    ) -> None:
        """Permit this run, or raise.

        Raises:
            NotEntitled: If the principal may not run this strategy on this
                portfolio with this level of discretion.
        """
        ...
