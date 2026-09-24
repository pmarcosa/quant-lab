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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from contracts.errors import ContractViolation, StateIntegrityError
from contracts.identifiers import StrategyVersion
from contracts.temporal import utc

#: Bumped when the row schema changes, so old studies stay readable and are not
#: silently mixed with new ones.
LEDGER_FORMAT = 1


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))
        object.__setattr__(self, "metrics", dict(self.metrics))
        if not self.study:
            raise ContractViolation("a trial must belong to a named study")
        for key, value in self.metrics.items():
            if not isinstance(value, (int, float)) or value != value:
                raise ContractViolation(f"metric {key!r} is not a finite number: {value!r}")

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
            "note": self.note,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Trial:
        from contracts.identifiers import StrategyId

        if row.get("format") != LEDGER_FORMAT:
            raise StateIntegrityError(
                f"ledger row is format {row.get('format')!r}, this code reads {LEDGER_FORMAT}"
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

        Returns:
            The ``T x N`` matrix of the trials sharing the most common series
            length (for N_eff), those trials, and the others. The others cannot
            be correlated without aligning dates, so a caller counts them at face
            value: the conservative choice.
        """
        line = [t for t in self if str(t.version.strategy) == strategy]
        with_returns = [t for t in line if t.returns]
        if not with_returns:
            return np.empty((0, 0)), (), tuple(line)
        lengths = [len(t.returns) for t in with_returns]
        longest = max(set(lengths), key=lengths.count)
        kept = tuple(t for t in with_returns if len(t.returns) == longest)
        others = tuple(t for t in line if t not in kept)
        matrix = np.column_stack([np.asarray(t.returns, dtype=float) for t in kept])
        return matrix, kept, others

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
        )
        self.ledger.record(trial)
        self.recorded += 1
        return trial
