"""The error vocabulary shared across layers.

Every failure a caller might reasonably handle gets its own type. Everything else
is a bug and should raise the ordinary built-ins.
"""

from __future__ import annotations


class QuantLabError(Exception):
    """Base class for every error this system raises on purpose."""


class ContractViolation(QuantLabError):
    """A value broke an invariant its type promised to uphold."""


class CausalityViolation(QuantLabError):
    """Something asked for information that did not exist at the decision time.

    This is raised rather than silently returning the data, because a look-ahead
    that fails loudly during development costs an afternoon and one that passes
    silently costs a live account.
    """


class NotEntitled(QuantLabError):
    """The caller is not permitted to run this strategy on this portfolio."""


class StateIntegrityError(QuantLabError):
    """Persisted state failed its checksum, or is from an incompatible version.

    Recovery must stop here. Trading on a half-written position snapshot is worse
    than not trading at all.
    """
