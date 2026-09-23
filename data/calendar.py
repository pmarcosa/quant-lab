"""The New York Stock Exchange's sessions: which days trade, and from when to when.

Daily and weekly bars can ignore this -- a missing holiday bar is simply not
there. Intraday bars cannot. The project's expert lists the session calendar
as a prerequisite for trading intraday: without it the system cannot tell a
holiday from a data outage, or a 13:00 half-day close from a missing afternoon,
and a bar from the pre-market is a different market (thin book, wide spreads)
that has to be kept apart from the regular session.

The rules and the one-off closures (a national day of mourning, a hurricane)
come from ``exchange_calendars``, a maintained library, rather than from a
hand-written list here: a calendar is data that goes stale, and a stale one
fails silently. It is an optional dependency (``pip install -e ".[intraday]"``,
included in ``dev``); only intraday strategies need it.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd

from contracts.errors import ContractViolation

EXCHANGE = "XNYS"


@lru_cache(maxsize=1)
def _calendar():
    try:
        import exchange_calendars
    except ImportError as missing:
        raise ContractViolation(
            "intraday bars need the exchange calendar: pip install -e \".[intraday]\" "
            "(the exchange_calendars package)"
        ) from missing
    return exchange_calendars.get_calendar(EXCHANGE, start="1995-01-01")


def is_available() -> bool:
    """Whether the calendar can be loaded here."""
    try:
        _calendar()
    except ContractViolation:
        return False
    return True


def session_bounds(day: date) -> tuple[datetime, datetime] | None:
    """The regular session's open and close on ``day``, in UTC; ``None`` if closed."""
    calendar = _calendar()
    label = pd.Timestamp(day)
    if not calendar.is_session(label):
        return None
    return (
        calendar.session_open(label).to_pydatetime().astimezone(timezone.utc),
        calendar.session_close(label).to_pydatetime().astimezone(timezone.utc),
    )


def is_early_close(day: date) -> bool:
    """A half day: the session closes before 16:00 New York (13:00 on those days)."""
    bounds = session_bounds(day)
    if bounds is None:
        return False
    return bool(pd.Timestamp(day) in _calendar().early_closes)


def in_regular_session(moment: datetime) -> bool:
    """Whether ``moment`` falls inside a regular session: open <= t < close."""
    stamp = pd.Timestamp(moment)
    if stamp.tzinfo is None:
        raise ContractViolation("a session check needs a timezone-aware time")
    ny = stamp.tz_convert("America/New_York")
    bounds = session_bounds(ny.date())
    return bounds is not None and bounds[0] <= stamp.to_pydatetime() < bounds[1]


def regular_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Only the bars that *start* inside a regular session.

    IBKR labels intraday bars with their start, so a bar is regular when its
    label is. Extended-hours bars are dropped rather than kept with a flag:
    nothing in the system trades outside the regular session, and a statistic
    computed over both mixes two markets.
    """
    if bars.empty:
        return bars
    keep = [in_regular_session(label.to_pydatetime()) for label in bars.index]
    return bars.loc[keep]


def last_closed_bar(now: datetime, duration: timedelta) -> datetime | None:
    """When the most recent intraday bar of ``duration`` that has finished closed.

    Bars are aligned to the session open; the session's last bar may be short
    (an hourly bar from 15:30 to 16:00, or anything on a half day), and it
    closes at the session close. ``None`` if no session has had a bar close.
    """
    stamp = pd.Timestamp(now)
    if stamp.tzinfo is None:
        raise ContractViolation("last_closed_bar needs a timezone-aware time")
    day = stamp.tz_convert("America/New_York").date()
    for _ in range(10):
        bounds = session_bounds(day)
        if bounds is not None:
            open_, close = bounds
            if now >= close:
                return close
            if now >= open_ + duration:
                elapsed = int((now - open_) / duration)
                return open_ + elapsed * duration
        day = day - timedelta(days=1)
        now = datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc)
    return None
