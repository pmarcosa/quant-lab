"""Loading price files into the bitemporal store.

A vendor CSV carries one timestamp per bar. Turning it into two — when the bar
happened, and when it could first have been acted on — is a judgement that has to
be made somewhere, and making it here, once and explicitly, is better than each
caller assuming zero.

For US equity sessions the bar's ``event_time`` is the session close rather than
the file's midnight date, because a strategy deciding on Friday evening and one
deciding at Friday midnight see different worlds. ``available_time`` adds a short
publication lag for the print to settle. How long a strategy must then wait before
acting is a separate matter, carried by ``FiltrationSpec.observation_lag_bars``.

**Weekly bars are stamped at the end of their week, not the start.** IBKR labels a
weekly bar with the week's first trading day, but its close is Friday's. Until
2026-09-21 the ingest used the label, so every weekly bar was dated four days
before it could have existed. In a backtest that was harmless — every decision
still used bar *k*'s close and filled at bar *k+1*'s open. Live it would
not have been harmless: a bar fetched on Saturday would have carried a Monday
event time, and the decision could not have been scheduled after it honestly.

**Split weeks are merged first.** See :func:`merge_split_weeks`. That merge is a
data change, and it moves the backtest; the relabelling on its own does not, and
``tests/test_data_layer.py`` pins both halves separately.

**The close is 21:00 UTC,** which is 16:00 New York in winter and an hour after
the close in summer. Using the later time in both seasons means a bar is never
stamped available before its session actually ended; the cost is that a summer
decision waits an hour it did not strictly need to.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from contracts.errors import ContractViolation
from contracts.identifiers import InstrumentId
from contracts.temporal import BarInterval
from data.bitemporal import BitemporalStore
from data.universe import PointInTimeUniverse, derive_memberships

#: US equity regular session close, in UTC, taken at its later seasonal value so
#: that no bar is ever stamped before its session closed. See the module note.
US_SESSION_CLOSE = time(21, 0, tzinfo=timezone.utc)

#: How long after the close a bar is treated as final.
PUBLICATION_LAG = timedelta(minutes=15)


def load_price_csv(path: Path) -> pd.DataFrame:
    """Read a price CSV indexed by timestamp.

    Args:
        path: File with a ``timestamp`` column plus OHLCV payload.

    Raises:
        ContractViolation: If the file has no ``timestamp`` column.
    """
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ContractViolation(f"{path.name} has no 'timestamp' column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.set_index("timestamp").sort_index()


def _resolve(interval: BarInterval | None, week_ending: bool) -> BarInterval:
    """The bar size, from the explicit interval or the older ``week_ending`` flag."""
    if interval is not None:
        return interval
    return BarInterval.WEEK if week_ending else BarInterval.DAY


def bar_close(
    label: datetime, week_ending: bool = False, interval: BarInterval | None = None
) -> datetime:
    """When the bar labelled ``label`` actually closed.

    Args:
        label: The vendor's label for the bar.
        week_ending: Legacy spelling of ``interval=BarInterval.WEEK``.
        interval: The bar size. Weekly bars close on the Friday of the label's
            ISO week, whatever day the vendor labelled them with; daily bars at
            that day's session close; intraday bars, which IBKR labels with
            their *start*, one bar length after the label. An intraday bar
            stamped at its start would be visible to a decision it had not
            finished forming for -- a lookahead of exactly one bar.
    """
    kind = _resolve(interval, week_ending)
    if kind.is_intraday:
        stamp = pd.Timestamp(label)
        if stamp.tzinfo is None:
            raise ContractViolation(
                f"intraday bar label {label!r} has no timezone; intraday data must "
                f"arrive with explicit UTC timestamps"
            )
        close = (stamp + kind.duration).to_pydatetime().astimezone(timezone.utc)
        return _within_session(stamp.to_pydatetime(), close)
    day = pd.Timestamp(label).date()
    if kind is BarInterval.WEEK:
        day = day + timedelta(days=4 - day.weekday())
    return datetime.combine(day, US_SESSION_CLOSE.replace(tzinfo=None), tzinfo=timezone.utc)


def _within_session(start: datetime, close: datetime) -> datetime:
    """An intraday bar ends at the session close if that comes first.

    The last hourly bar of a session runs 15:30-16:00, and every bar on a half
    day ends by 13:00. Without the exchange calendar the nominal close is kept:
    later than the truth, so never a look-ahead, only a later decision.
    """
    from data import calendar

    if not calendar.is_available():
        return close
    bounds = calendar.session_bounds(pd.Timestamp(start).tz_convert("America/New_York").date())
    if bounds is None:
        return close
    return min(close, bounds[1])


def to_observations(
    bars: pd.DataFrame,
    week_ending: bool = False,
    available_at: datetime | None = None,
    interval: BarInterval | None = None,
) -> pd.DataFrame:
    """Attach event and availability timestamps to session bars.

    Args:
        bars: OHLCV indexed by the vendor's bar label.
        week_ending: Legacy spelling of ``interval=BarInterval.WEEK``.
        available_at: When these rows became known, if later than the close plus
            the publication lag. Historical files leave it unset; a live refresh
            passes the moment of the fetch, because that is the truth.
        interval: The bar size; see :func:`bar_close`.
    """
    kind = _resolve(interval, week_ending)
    closes = pd.to_datetime(
        [bar_close(d, interval=kind) for d in bars.index], utc=True
    )
    available = closes + PUBLICATION_LAG
    if available_at is not None:
        floor = pd.Timestamp(available_at).tz_convert("UTC")
        available = available.where(available >= floor, floor)
    frame = bars.reset_index(drop=True)
    frame.insert(0, "event_time", closes)
    frame.insert(1, "available_time", available)
    return frame


def merge_split_weeks(bars: pd.DataFrame) -> pd.DataFrame:
    """One bar per ISO week: open of the first, extremes, close of the last.

    IBKR occasionally returns a week as two bars — around holidays, or where a
    session was split. Treated as two observations they add a phantom week to
    every rolling indicator: a ten-week SMA silently becomes a nine-and-a-bit
    week one. Across this cache that was 76 phantom weeks in 29 instruments.
    Merging them is the only reading under which "weekly" means one per week.
    """
    if bars.empty:
        return bars
    labels = pd.to_datetime(bars.index)
    iso = labels.isocalendar()
    key = pd.MultiIndex.from_arrays([iso.year.values, iso.week.values])
    if not key.duplicated().any():
        return bars
    frame = bars.copy()
    frame["_label"] = labels
    frame.index = key
    aggregation = {"_label": "first"}
    for column, rule in (
        ("open", "first"), ("high", "max"), ("low", "min"), ("close", "last"), ("volume", "sum")
    ):
        if column in frame.columns:
            aggregation[column] = rule
    merged = frame.groupby(level=[0, 1], sort=True).agg(aggregation)
    merged.index = pd.DatetimeIndex(merged.pop("_label"), name=bars.index.name)
    return merged.sort_index()


def weeks_from_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly bars from daily ones, without the window's first, partial week.

    IBKR gives split- and dividend-adjusted prices only for bars of a day or
    less, so weekly bars are built here from daily ones (:func:`merge_split_weeks`).
    A window of daily bars ("2 Y" back from now) almost always opens mid-week:
    its first ISO week holds only that week's last few sessions. Kept, it is a
    short bar with the wrong open, and merged into a cache where fresh bars win
    it would overwrite the complete week already there. Dropping it costs one
    week at the far end of the window (for a name that listed inside the
    window, its listing week).
    """
    if daily.empty:
        return daily
    return merge_split_weeks(daily).iloc[1:]


def complete_bars(
    bars: pd.DataFrame,
    now: datetime,
    week_ending: bool = False,
    interval: BarInterval | None = None,
) -> pd.DataFrame:
    """Only the bars whose session had closed, and been published, by ``now``.

    Replaces the old "drop the last row" rule, which was wrong in both
    directions: it discarded a finished week fetched on a Saturday, and it would
    have kept a half-finished one if the vendor had sent two in-progress rows.
    """
    if bars.empty:
        return bars
    kind = _resolve(interval, week_ending)
    closes = [bar_close(d, interval=kind) + PUBLICATION_LAG for d in bars.index]
    keep = [close <= now for close in closes]
    return bars.loc[keep]


def ingest_directory(
    source: Path,
    store: BitemporalStore,
    week_ending: bool = False,
    symbols: Sequence[str] | None = None,
    now: datetime | None = None,
    interval: BarInterval | None = None,
) -> dict[str, int]:
    """Load every price CSV in a directory into the store.

    Args:
        source: Directory of ``<SYMBOL>.csv`` files.
        store: Destination.
        week_ending: Legacy spelling of ``interval=BarInterval.WEEK``.
        symbols: Restrict to these symbols; all of them when omitted.
        now: Bars not yet closed and published by this moment are left out —
            a partial bar's high, low and close are not final, and a partial bar
            is a live-versus-backtest discrepancy waiting to happen.
        interval: The bar size of the files.

    Returns:
        Symbol to number of observations written.
    """
    if not source.is_dir():
        raise ContractViolation(f"no such directory: {source}")
    kind = _resolve(interval, week_ending)
    written: dict[str, int] = {}
    wanted = set(symbols) if symbols else None
    for path in sorted(source.glob("*.csv")):
        symbol = path.stem
        if wanted is not None and symbol not in wanted:
            continue
        bars = load_price_csv(path)
        if kind is BarInterval.WEEK:
            bars = merge_split_weeks(bars)
        bars = complete_bars(bars, now or datetime.now(timezone.utc), interval=kind)
        if bars.empty:
            continue
        written[symbol] = store.append(
            InstrumentId(symbol), to_observations(bars, interval=kind)
        )
    return written


def universe_from_store(
    store: BitemporalStore, still_trading_after: datetime
) -> PointInTimeUniverse:
    """Derive membership windows from the data the store actually holds.

    Approximate by construction — see ``data/universe.py`` for exactly which bias
    this removes and which it does not.
    """
    now = datetime.now(timezone.utc)
    spans = []
    for instrument in store.instruments():
        known = store.as_of(instrument, now)
        if known.empty:
            continue
        spans.append(
            (
                instrument,
                known.index[0].to_pydatetime(),
                known.index[-1].to_pydatetime(),
            )
        )
    return derive_memberships(spans, still_trading_after)
