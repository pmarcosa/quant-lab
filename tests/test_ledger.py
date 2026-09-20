"""The research ledger.

Most of these check that the ledger records what it should. The ones that matter
check that a researcher *cannot* avoid recording — the file-drawer problem is
solved by the shape of the API, not by remembering to be honest.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from contracts.errors import ContractViolation, StateIntegrityError
from contracts.identifiers import StrategyVersion
from tests.conftest import at
from validation.ledger import LEDGER_FORMAT, ResearchLedger, Study, Trial


def version(**params) -> StrategyVersion:
    return StrategyVersion.of("momentum", params or {"lookback": 13})


@pytest.fixture
def ledger(tmp_path) -> ResearchLedger:
    return ResearchLedger(tmp_path / "research.jsonl")


# -- the point of the thing --------------------------------------------------


def test_there_is_no_way_to_evaluate_without_recording(ledger):
    """The file-drawer problem, closed structurally.

    A researcher who runs twenty configurations and reports the best one has
    made a selection from twenty, and the DSR needs to know that. The API gives
    no way to run one without writing it: no skip flag, no delete, no filter.
    """
    assert not hasattr(ledger, "delete")
    assert not hasattr(ledger, "remove")
    with Study("sweep", ledger) as study:
        for lookback in range(8, 28):
            study.evaluate(version(lookback=lookback), "2009-2026", {"sharpe": 0.1 * lookback})
    assert ledger.count("sweep") == 20
    assert study.recorded == 20


def test_a_disappointing_trial_is_recorded_like_any_other(ledger):
    with Study("sweep", ledger) as study:
        study.evaluate(version(a=1), "w", {"sharpe": 1.4})
        study.evaluate(version(a=2), "w", {"sharpe": -0.9})
    sharpes = sorted(t.metrics["sharpe"] for t in ledger.trials("sweep"))
    assert sharpes == [-0.9, 1.4], "the bad one is the whole reason the count matters"


def test_the_best_trial_is_the_one_selection_bias_favours(ledger):
    with Study("sweep", ledger) as study:
        for i, sharpe in enumerate([0.2, 1.9, 0.7]):
            study.evaluate(version(a=i), "w", {"sharpe": sharpe})
    best = ledger.best("sweep")
    assert best is not None
    assert best.metrics["sharpe"] == 1.9


def test_studies_do_not_contaminate_each_other(ledger):
    with Study("stops", ledger) as first:
        first.evaluate(version(a=1), "w", {"sharpe": 1.0})
    with Study("cadence", ledger) as second:
        second.evaluate(version(b=1), "w", {"sharpe": 2.0})
        second.evaluate(version(b=2), "w", {"sharpe": 2.1})
    assert ledger.count("stops") == 1
    assert ledger.count("cadence") == 2
    assert ledger.count() == 3


# -- durability --------------------------------------------------------------


def test_the_ledger_is_append_only_on_disk(ledger):
    with Study("s", ledger) as study:
        study.evaluate(version(a=1), "w", {"sharpe": 1.0})
        first = ledger.path.read_text()
        study.evaluate(version(a=2), "w", {"sharpe": 1.1})
    second = ledger.path.read_text()
    assert second.startswith(first), "earlier rows are never rewritten"
    assert len(second.splitlines()) == 2


def test_a_truncated_line_is_an_error_not_a_silent_gap(ledger):
    with Study("s", ledger) as study:
        study.evaluate(version(a=1), "w", {"sharpe": 1.0})
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"study": "s", "metri')
    with pytest.raises(StateIntegrityError, match="cannot be trusted"):
        ledger.trials()


def test_a_row_from_a_different_format_is_refused(ledger):
    row = {"format": LEDGER_FORMAT + 1, "study": "s"}
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(StateIntegrityError, match="format"):
        ledger.trials()


def test_a_trial_survives_a_round_trip(ledger):
    with Study("s", ledger) as study:
        written = study.evaluate(
            version(a=1), "2009-2026", {"sharpe": 1.23, "cagr": 0.2},
            returns=[0.01, -0.02, 0.03], note="first real run",
        )
    (read_back,) = ledger.trials("s")
    assert read_back == written


def test_an_empty_ledger_reads_as_empty(ledger):
    assert ledger.trials() == ()
    assert ledger.count() == 0
    assert ledger.best("anything") is None


# -- what the DSR needs ------------------------------------------------------


def test_returns_are_kept_so_trials_can_be_correlated(ledger):
    """Without the series, N_eff cannot be estimated and every trial in a sweep
    has to be counted as independent -- which over-penalises the DSR."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.02, 200)
    with Study("sweep", ledger) as study:
        for i in range(5):
            nudged = base + rng.normal(0, 0.001, 200)
            study.evaluate(version(a=i), "w", {"sharpe": 1.0}, returns=nudged)
    matrix, kept = ledger.returns_matrix("sweep")
    assert matrix.shape == (200, 5)
    assert len(kept) == 5
    correlation = np.corrcoef(matrix, rowvar=False)
    assert correlation.min() > 0.9, "near-identical parameters give near-identical returns"


def test_trials_of_differing_length_are_not_silently_aligned(ledger):
    with Study("s", ledger) as study:
        study.evaluate(version(a=1), "w", {"sharpe": 1.0}, returns=[0.1] * 100)
        study.evaluate(version(a=2), "w", {"sharpe": 1.0}, returns=[0.1] * 100)
        study.evaluate(version(a=3), "w", {"sharpe": 1.0}, returns=[0.1] * 50)
    matrix, kept = ledger.returns_matrix("s")
    assert matrix.shape == (100, 2), "the odd one out is dropped, not padded"
    assert len(kept) == 2


def test_a_trial_needs_a_study_and_finite_metrics():
    with pytest.raises(ContractViolation, match="named study"):
        Trial(study="", version=version(), recorded_at=at(2026), window="w", metrics={})
    with pytest.raises(ContractViolation, match="finite"):
        Trial(
            study="s", version=version(), recorded_at=at(2026), window="w",
            metrics={"sharpe": float("nan")},
        )


def test_parameters_distinguish_trials_of_the_same_strategy(ledger):
    with Study("sweep", ledger) as study:
        a = study.evaluate(version(lookback=13), "w", {"sharpe": 1.0})
        b = study.evaluate(version(lookback=26), "w", {"sharpe": 1.0})
    assert a.version.strategy == b.version.strategy
    assert a.key != b.key, "same family, different strategy"
