"""The local, single-user implementation of the access seam.

A null object in the design-pattern sense: it satisfies the interface completely
while granting everything, so the rest of the system can be written as if
identity, entitlements and licences already exist.

Nothing here is throwaway. When a product needs real accounts, this class is
joined by a second implementation and the call sites do not move.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from contracts.access import AccessPort, DiscretionMode, Entitlement
from contracts.errors import NotEntitled
from contracts.identifiers import PortfolioId, StrategyId, StrategyVersion, TenantId

#: Discretion modes a locally-run system may use. Both, because the owner of the
#: machine is also the owner of the money.
ALL_DISCRETION = frozenset(DiscretionMode)


@dataclass(frozen=True, slots=True)
class LocalOwner(AccessPort):
    """One owner, their own portfolios, every strategy, both discretion modes.

    Args:
        tenant: The single principal.
        books: The portfolios they operate.
        allowed: Strategies they may run, or None for all.
        discretion: Permitted discretion modes.
    """

    tenant: TenantId
    books: tuple[PortfolioId, ...]
    allowed: frozenset[StrategyId] | None = None
    discretion: frozenset[DiscretionMode] = ALL_DISCRETION

    @classmethod
    def single_portfolio(cls, tenant: str, portfolio: str) -> LocalOwner:
        """Build the common case: one person, one book."""
        owner = TenantId(tenant)
        return cls(tenant=owner, books=(PortfolioId(owner, portfolio),))

    def principal(self) -> TenantId:
        return self.tenant

    def portfolios(self) -> Sequence[PortfolioId]:
        return self.books

    def entitlements(self, portfolio: PortfolioId) -> Sequence[Entitlement]:
        if portfolio not in self.books:
            return ()
        strategies = self.allowed if self.allowed is not None else ()
        return tuple(
            Entitlement(portfolio=portfolio, strategy=strategy, discretion=self.discretion)
            for strategy in sorted(strategies, key=str)
        )

    def authorise(
        self,
        portfolio: PortfolioId,
        version: StrategyVersion,
        discretion: DiscretionMode,
    ) -> None:
        """Permit the run, or say precisely why not."""
        if portfolio not in self.books:
            raise NotEntitled(
                f"{self.tenant} does not operate {portfolio}; "
                f"known portfolios: {[str(b) for b in self.books]}"
            )
        if self.allowed is not None and version.strategy not in self.allowed:
            raise NotEntitled(f"{self.tenant} is not entitled to run {version.strategy}")
        if discretion not in self.discretion:
            raise NotEntitled(
                f"{discretion.value} is not permitted for {portfolio}; "
                f"permitted: {sorted(m.value for m in self.discretion)}"
            )
