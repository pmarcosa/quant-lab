"""The research ledger: every trial, written by the engine, never by the researcher.

The Deflated Sharpe Ratio needs to know how many strategies were tried before the
winner was chosen. That number cannot be reconstructed afterwards — nobody
remembers the configurations that looked bad and were dropped, and those are
exactly the ones that make the survivor look good. **Trials not recorded at the
moment they run are lost permanently, and the DSR for that line of research is
uncomputable forever.**

So the ledger has one property above all others: **the researcher cannot choose
what goes in it.** There is no `skip`, no `delete`, no filter argument. The only
way to evaluate a strategy is through :meth:`ResearchLedger.record`, and the only
way to avoid recording a trial is not to run it. This is a file-drawer problem
solved structurally rather than by discipline.

Append-only, one JSON object per line, flushed and fsynced per write. A crash
mid-study loses nothing already written, and a partially written final line is
detected on read rather than silently parsed.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from contracts.errors import ContractViolation, StateIntegrityError
from contracts.identifiers import StrategyVersion
from contracts.temporal import utc

#: Bumped when the row schema changes, so old studies stay readable and are not
#: silently mixed with new ones. Format 2 adds the date of every return.
LEDGER_FORMAT = 2
READABLE_FORMATS = (1, 2)


@dataclass(frozen=True, slots=True)
class Trial:
    """One evaluation of one strategy version on one data window.

    A "trial" is any evaluation of a loss function — a backtest, a fold of a
    cross-validation, a single point in a parameter sweep. The count that matters
    for the DSR is the count of *evaluations*, not of ideas or of published
    results, which is why the granularity is this fine.

    Attributes:
        study: The question being asked. Trials are compared within a study.
        version: Strategy family plus parameter fingerprint.
        recorded_at: When the evaluation finished.
        window: Which data it saw, as an opaque label (dates, fold id, seed).
        metrics: Whatever was measured. ``sharpe`` is expected by the DSR.
        returns: The periodic return series, kept so that later analysis can
            compute the correlation between trials. Without it, N_eff cannot be
            estimated and the DSR has to assume every trial was independent,
            which over-penalises a sweep of near-identical parameters.
        note: Free text for the human.
    """

    study: str
    version: StrategyVersion
    recorded_at: datetime
    window: str
    metrics: Mapping[str, float]
    returns: tuple[float, ...] = ()
    note: str = ""
    #: When each return was earned (ISO timestamps, one per return). With them,
    #: trials run over different windows can be aligned on the dates they share
    #: and de-correlated, instead of each counting as a whole trial.
    dates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))
        object.__setattr__(self, "metrics", dict(self.metrics))
        if not self.study:
            raise ContractViolation("a trial must belong to a named study")
        for key, value in self.metrics.items():
            if not isinstance(value, (int, float)) or value != value:
                raise ContractViolation(f"metric {key!r} is not a finite number: {value!r}")
        if self.dates and len(self.dates) != len(self.returns):
            raise ContractViolation(
                f"{len(self.dates)} dates for {len(self.returns)} returns: each return needs "
                f"exactly one date"
            )

    @property
    def key(self) -> str:
        """Identity of what was evaluated: version plus window."""
        return f"{self.version.strategy}|{self.version.params_hash}|{self.window}"

    def to_row(self) -> dict[str, Any]:
        return {
            "format": LEDGER_FORMAT,
            "study": self.study,
            "strategy": str(self.version.strategy),
            "params_hash": self.version.params_hash,
            "code_version": self.version.code_version,
            "recorded_at": self.recorded_at.isoformat(),
            "window": self.window,
            "metrics": dict(self.metrics),
            "returns": list(self.returns),
            "dates": list(self.dates),
            "note": self.note,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Trial:
        from contracts.identifiers import StrategyId

        if row.get("format") not in READABLE_FORMATS:
            raise StateIntegrityError(
                f"ledger row is format {row.get('format')!r}, this code reads "
                f"{READABLE_FORMATS}"
            )
        return cls(
            study=row["study"],
            version=StrategyVersion(
                strategy=StrategyId(row["strategy"]),
                params_hash=row["params_hash"],
                code_version=row["code_version"],
            ),
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            window=row["window"],
            metrics=row["metrics"],
            returns=tuple(row.get("returns", ())),
            note=row.get("note", ""),
            dates=tuple(row.get("dates", ())),
        )


@dataclass(frozen=True, slots=True)
class ResearchLedger:
    """An append-only record of every trial run.

    Attributes:
        path: The JSONL file. Created on first write.
    """

    path: Path

    def record(self, trial: Trial) -> None:
        """Append a trial. The only mutating operation there is.

        Durability matters more than speed here: a study is thousands of cheap
        evaluations and one expensive fact, which is how many there were. The
        write is flushed and fsynced so a crash cannot lose the count.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trial.to_row(), separators=(",", ":"), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def __iter__(self) -> Iterator[Trial]:
        """Every trial, oldest first.

        Raises:
            StateIntegrityError: On a truncated or corrupt line. A ledger that
                cannot be read in full is not a ledger with a gap — it is a
                ledger whose count is wrong, and the count is the whole point.
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StateIntegrityError(
                        f"{self.path}:{number} is not valid JSON; the trial count cannot "
                        f"be trusted while this line is unreadable"
                    ) from error
                yield Trial.from_row(row)

    def trials(self, study: str | None = None) -> tuple[Trial, ...]:
        """Every trial, optionally filtered to one study."""
        return tuple(t for t in self if study is None or t.study == study)

    def count(self, study: str | None = None) -> int:
        """How many evaluations were run. The N the DSR needs."""
        return len(self.trials(study))

    def best(self, study: str, metric: str = "sharpe") -> Trial | None:
        """The trial with the highest ``metric``. The one selection bias favours."""
        candidates = [t for t in self.trials(study) if metric in t.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.metrics[metric])

    def research_line(
        self, strategy: str
    ) -> tuple[np.ndarray, tuple[Trial, ...], tuple[Trial, ...]]:
        """Every trial of one strategy family, whatever study recorded it.

        The expert (2026-09-24): a new universe, other positions or other
        rotation weeks are the same line of research, not a new hypothesis, so
        all of its trials count toward one Deflated Sharpe -- manual backtests
        included. Controls (another strategy family) are not candidates and are
        left out.

        Trials are aligned on their dates. The common window is the span of one
        of the trials that the most trials cover (at least half as long as the
        typical trial, so a short run cannot shrink everyone's window to itself),
        and every trial covering it is cut to it and de-correlated with the rest.
        A trial without dates has them inferred from its label's window when the
        count matches exactly, one per week; failing that, from the one dated
        window of its exact length, if there is only one. Otherwise, and for
        trials that do not cover the window, it is counted at face value by the
        caller -- the conservative choice.

        Returns:
            The ``T x N`` matrix of the aligned trials (for N_eff), those trials,
            and the others.
        """
        line = [t for t in self if str(t.version.strategy) == strategy]
        with_returns = [t for t in line if t.returns]
        if not with_returns:
            return np.empty((0, 0)), (), tuple(line)
        dated = {}
        for t in with_returns:
            keys = _date_keys(t)
            if keys is not None:
                dated[id(t)] = keys
        # A trial with no dates and no window in its label ("full", from before
        # either existed) is placed on a dated window of the same length -- the
        # rule the ledger used before dates -- but only when exactly one dated
        # window has that length. Otherwise it stays at face value.
        spans_by_length: dict[int, set[tuple[str, ...]]] = {}
        for keys in dated.values():
            spans_by_length.setdefault(len(keys), set()).add(keys)
        for t in with_returns:
            spans = spans_by_length.get(len(t.returns), set())
            if id(t) not in dated and len(spans) == 1:
                dated[id(t)] = next(iter(spans))
        window = _common_window([dated[id(t)] for t in with_returns if id(t) in dated])
        columns, kept = [], []
        for t in with_returns:
            keys = dated.get(id(t))
            if window is not None and keys is not None:
                position = {k: i for i, k in enumerate(keys)}
                if all(k in position for k in window):
                    columns.append([t.returns[position[k]] for k in window])
                    kept.append(t)
        if window is None:  # nothing dated: the old rule, the most common length
            lengths = [len(t.returns) for t in with_returns]
            longest = max(set(lengths), key=lengths.count)
            kept = [t for t in with_returns if len(t.returns) == longest]
            columns = [list(t.returns) for t in kept]
        others = tuple(t for t in line if not any(t is k for k in kept))
        matrix = np.column_stack([np.asarray(c, dtype=float) for c in columns])
        return matrix, tuple(kept), others

    def returns_matrix(self, study: str) -> tuple[np.ndarray, tuple[Trial, ...]]:
        """Trial returns as a ``T x N`` matrix, for estimating how many were distinct.

        Only trials whose return series are all the same length are included:
        correlating series of different lengths would mean aligning them, and a
        misalignment here quietly changes N_eff and therefore the DSR.
        """
        candidates = [t for t in self.trials(study) if t.returns]
        if not candidates:
            return np.empty((0, 0)), ()
        lengths = {len(t.returns) for t in candidates}
        longest = max(lengths, key=lambda n: sum(1 for t in candidates if len(t.returns) == n))
        kept = tuple(t for t in candidates if len(t.returns) == longest)
        matrix = np.column_stack([np.asarray(t.returns, dtype=float) for t in kept])
        return matrix, kept


@dataclass
class Study:
    """A named question, and the ledger writes that answering it produces.

    Used as a context manager so that the recording is not something a caller
    remembers to do:

        with Study("stop-sweep", ledger) as study:
            for stop in candidates:
                study.evaluate(version, window, run_it)

    ``evaluate`` records before it returns. There is no path through this class
    that runs a strategy without writing a row — including one that raises, which
    is recorded as a failed trial rather than vanishing.
    """

    name: str
    ledger: ResearchLedger
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recorded: int = 0

    def __enter__(self) -> Study:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def existing(self, version: StrategyVersion, window: str, note: str) -> Trial | None:
        """A trial already recorded for exactly this evaluation, if there is one.

        Lets a long study resume after an interruption without repeating work,
        and without inflating the trial count by recording the same evaluation
        twice. Note that this cannot be used to *avoid* recording: the row it
        finds is one that was already written.
        """
        for trial in self.ledger.trials(self.name):
            if trial.version == version and trial.window == window and trial.note == note:
                return trial
        return None

    def evaluate(
        self,
        version: StrategyVersion,
        window: str,
        metrics: Mapping[str, float],
        returns: Sequence[float] = (),
        note: str = "",
        dates: Sequence[str] = (),
    ) -> Trial:
        """Record one evaluation. Returns the trial that was written."""
        trial = Trial(
            study=self.name,
            version=version,
            recorded_at=datetime.now(timezone.utc),
            window=window,
            metrics=metrics,
            returns=tuple(float(r) for r in returns),
            note=note,
            dates=tuple(str(d) for d in dates),
        )
        self.ledger.record(trial)
        self.recorded += 1
        return trial


_WINDOW = re.compile(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})")


def _date_keys(trial: Trial) -> tuple[str, ...] | None:
    """The trial's returns keyed by date, or None if they cannot be dated.

    Stored dates are used as they are (by session date for daily and weekly
    bars, by timestamp for intraday ones). A trial recorded before dates were
    stored gets weekly dates from its label's window ``A..B``, but only when
    that window holds exactly one week per return; anything else stays undated.
    """
    if trial.dates:
        days = tuple(d[:10] for d in trial.dates)
        return days if len(set(days)) == len(days) else tuple(trial.dates)
    match = _WINDOW.match(trial.window)
    if not match:
        return None
    first, last = (datetime.fromisoformat(v) for v in match.groups())
    weeks = (last - first).days // 7
    if (last - first).days % 7 or weeks != len(trial.returns):
        return None
    return tuple((first + timedelta(weeks=i + 1)).date().isoformat() for i in range(weeks))


def _common_window(series: list[tuple[str, ...]]) -> tuple[str, ...] | None:
    """The span, among the trials' own, that the most trials cover."""
    if not series:
        return None
    typical = float(np.median([len(s) for s in series]))
    sets = [set(s) for s in series]
    best, best_score = None, (-1, -1)
    for candidate in {s for s in series if len(s) >= max(2, typical / 2)}:
        covering = sum(1 for keys in sets if keys.issuperset(candidate))
        score = (covering, len(candidate))
        if score > best_score:
            best, best_score = candidate, score
    return best
