"""Time, and the rule that a decision may only use what existed when it was made.

Three timestamps, never one:

``event_time``
    The moment the world the datum describes happened — the close of a weekly
    bar, the end of a fiscal quarter.
``available_time``
    The moment the system could first have known it — when the vendor published
    it, when the file landed. Always at or after ``event_time``.
``decision_time``
    The moment a decision is being made. Everything a strategy sees is filtered
    by it.

The point of separating them is that they routinely differ. A weekly bar closes
Friday and a rotation happens Monday; a restated figure is published months after
the quarter it describes. Storing one timestamp collapses that distinction and
produces a backtest that knew things early.

:class:`Filtration` is how this becomes structural rather than a convention. A
strategy is handed a filtration, not a database. It has no method that takes a
date range, so there is nothing to pass a future date to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId

T = TypeVar("T")


def utc(moment: datetime) -> datetime:
    """Normalise a datetime to UTC, refusing naive ones.

    Naive datetimes are the quiet source of off-by-one-session errors: the same
    literal means different instants depending on where the process runs.

    Args:
        moment: An aware datetime.

    Raises:
        ContractViolation: If ``moment`` has no timezone.
    """
    if moment.tzinfo is None:
        raise ContractViolation(
            f"{moment!r} is timezone-naive. Every timestamp in this system is aware; "
            f"attach a timezone at the boundary where the data enters."
        )
    return moment.astimezone(timezone.utc)


class BarInterval(str, Enum):
    """Sampling frequencies the system understands.

    Everything that depends on the bar size asks this enum rather than assuming
    one: annualisation, how many bars make a week, how long a bar lasts, what
    the vendor calls it. A module that hard-codes "52" or "weekly" is a module
    that silently breaks the day a daily or intraday strategy arrives.
    """

    MINUTE = "1Min"
    HOUR = "1Hour"
    DAY = "1Day"
    WEEK = "1Week"

    @classmethod
    def parse(cls, name: str) -> BarInterval:
        """Accept the enum value or a plain word: ``week``, ``weekly``, ``1Day``..."""
        key = str(name).strip().lower()
        aliases = {
            "1min": cls.MINUTE, "minute": cls.MINUTE, "min": cls.MINUTE,
            "1hour": cls.HOUR, "hour": cls.HOUR, "hourly": cls.HOUR,
            "1day": cls.DAY, "day": cls.DAY, "daily": cls.DAY,
            "1week": cls.WEEK, "week": cls.WEEK, "weekly": cls.WEEK,
        }
        if key not in aliases:
            raise ContractViolation(
                f"unknown bar interval {name!r}; use minute, hour, day or week"
            )
        return aliases[key]

    @property
    def pandas_freq(self) -> str:
        """The pandas offset alias for this interval."""
        return {"1Min": "min", "1Hour": "h", "1Day": "D", "1Week": "W-FRI"}[self.value]

    @property
    def periods_per_year(self) -> int:
        """Bars per year, for annualising.

        Daily uses 252 trading days and weekly 52; these are the conventions every
        metric in the system annualises with, kept in one place so two modules can
        never disagree.
        """
        return {"1Min": 252 * 390, "1Hour": 252 * 7, "1Day": 252, "1Week": 52}[self.value]

    @property
    def bars_per_week(self) -> float:
        """Bars in one trading week. Converts calendar settings into bar counts.

        Monitoring parameters that describe the market's memory -- a bootstrap
        block, the expected time between regime changes, a minimum sample -- are
        fixed in calendar time and translated into bars through this, so the
        same setting means the same thing at every frequency.
        """
        return self.periods_per_year / 52.0

    @property
    def duration(self) -> timedelta:
        """How long one bar lasts, in wall-clock time (a trading day for DAY)."""
        return {
            "1Min": timedelta(minutes=1), "1Hour": timedelta(hours=1),
            "1Day": timedelta(days=1), "1Week": timedelta(weeks=1),
        }[self.value]

    @property
    def is_intraday(self) -> bool:
        return self in (BarInterval.MINUTE, BarInterval.HOUR)

    @property
    def frequency(self) -> str:
        """The word used for cache folders and universe files."""
        return {"1Min": "minute", "1Hour": "hourly", "1Day": "daily", "1Week": "weekly"}[self.value]

    @property
    def noun(self) -> str:
        """One bar, in words: for reports and messages."""
        return {"1Min": "minute", "1Hour": "hour", "1Day": "day", "1Week": "week"}[self.value]


@dataclass(frozen=True, slots=True)
class FiltrationSpec:
    """How a strategy samples the world, and how late its inputs arrive.

    Attributes:
        interval: The bar size the strategy reasons in.
        observation_lag_bars: Bars between a bar closing and the strategy being
            allowed to act on it. One means "act on the next bar", which is the
            honest default: you cannot trade the close you are still observing.
    """

    interval: BarInterval
    observation_lag_bars: int = 1

    def __post_init__(self) -> None:
        if self.observation_lag_bars < 0:
            raise ContractViolation(
                f"observation_lag_bars cannot be negative; got {self.observation_lag_bars}. "
                f"A negative lag is look-ahead written as a setting."
            )


@dataclass(frozen=True, slots=True)
class Observation(Generic[T]):
    """A value, plus when it happened and when it could first be known.

    Attributes:
        event_time: When the described event occurred.
        available_time: When the system could first have known it.
        payload: The value itself.
    """

    event_time: datetime
    available_time: datetime
    payload: T

    def __post_init__(self) -> None:
        event = utc(self.event_time)
        available = utc(self.available_time)
        if available < event:
            raise ContractViolation(
                f"available_time {available.isoformat()} precedes event_time {event.isoformat()}: "
                f"the system cannot have known this before it happened."
            )
        object.__setattr__(self, "event_time", event)
        object.__setattr__(self, "available_time", available)

    def knowable_at(self, decision_time: datetime) -> bool:
        """Whether this observation may be used for a decision at ``decision_time``."""
        moment = utc(decision_time)
        return self.event_time <= moment and self.available_time <= moment


@runtime_checkable
class Filtration(Protocol):
    """Everything knowable at one instant — the only view a strategy gets.

    Deliberately has no method that accepts a date range or an end date. The
    decision time is fixed when the filtration is constructed by the engine, so a
    strategy has no way to ask for the future: the mistake is unavailable rather
    than discouraged.
    """

    @property
    def decision_time(self) -> datetime:
        """The instant this view is pinned to."""
        ...

    def history(self, instrument: InstrumentId, field: str, count: int) -> pd.Series:
        """The last ``count`` values of ``field`` for ``instrument``, oldest first.

        Shorter than ``count`` when less history is knowable; never longer, and
        never containing anything not yet available.
        """
        ...

    def frame(
        self, instruments: Sequence[InstrumentId], field: str, count: int
    ) -> pd.DataFrame:
        """``history`` for several instruments at once, aligned on one index."""
        ...

    def is_available(self, instrument: InstrumentId, min_bars: int = 1) -> bool:
        """Whether the instrument had listed and has at least ``min_bars`` of history.

        This is point-in-time availability, not a data-quality check. It is what
        stops a backtest of 2011 from selecting an instrument that listed in 2019.
        """
        ...

    def metadata(self, instrument: InstrumentId) -> dict[str, Any]:
        """Attributes of the instrument as they stood at the decision time."""
        ...
