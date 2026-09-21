"""Live state: the journal, the replayed book, the configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from contracts.errors import ContractViolation, StateIntegrityError
from contracts.execution import Fill, Side
from contracts.identifiers import InstrumentId
from contracts.live import DegradationState, EventKind, TradingMode
from runtime.config import load_config
from runtime.journal import (
    Journal,
    book_fingerprint,
    fill_from_dict,
    fill_to_dict,
    sleeve_book,
)
from tests.conftest import at

AAA, BBB = InstrumentId("AAA"), InstrumentId("BBB")


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal.jsonl")


def open_sleeve(journal, cash=10_000.0, positions=()):
    journal.append(
        EventKind.OPENED,
        at(2026, 9, 18),
        {"cash": cash, "positions": [dict(p) for p in positions]},
    )


def fill(instrument, side, quantity, price, moment, commission=1.0, oid="o1"):
    return Fill(
        client_order_id=oid, instrument=instrument, side=side,
        quantity=quantity, price=price, at=moment, commission=commission,
    )


# -- modes and states --------------------------------------------------------


def test_paper_and_live_accounts_are_told_apart():
    assert TradingMode.PAPER.admits("DU1234567")
    assert not TradingMode.PAPER.admits("U1234567")
    assert TradingMode.LIVE.admits("U1234567")
    assert not TradingMode.LIVE.admits("DU1234567"), "a paper account is not a live one"
    assert not TradingMode.LIVE.admits("F1234567")


def test_the_degradation_ladder_only_removes_permissions():
    normal, reduce, halted = (
        DegradationState.NORMAL, DegradationState.REDUCE_ONLY, DegradationState.HALTED,
    )
    assert normal.permits_buys and normal.permits_proposals
    assert not reduce.permits_buys and reduce.permits_proposals
    assert not halted.permits_buys and not halted.permits_proposals
    assert normal.worst(halted) is halted
    assert reduce.worst(normal) is reduce


# -- the journal -------------------------------------------------------------


def test_events_get_strictly_increasing_sequence_numbers(journal):
    first = journal.append(EventKind.NOTE, at(2026, 9, 18), {"text": "a"})
    second = journal.append(EventKind.NOTE, at(2026, 9, 18), {"text": "b"})
    assert (first.sequence, second.sequence) == (1, 2)
    reopened = Journal(journal.path)
    assert reopened.append(EventKind.NOTE, at(2026, 9, 19), {}).sequence == 3


def test_a_truncated_line_is_an_integrity_error(journal):
    journal.append(EventKind.NOTE, at(2026, 9, 18), {"text": "a"})
    with journal.path.open("a") as handle:
        handle.write('{"format":1,"sequ')
    with pytest.raises(StateIntegrityError, match="cannot be trusted"):
        list(journal)


def test_an_edited_journal_is_detected(journal):
    journal.append(EventKind.NOTE, at(2026, 9, 18), {"text": "a"})
    journal.append(EventKind.NOTE, at(2026, 9, 18), {"text": "b"})
    lines = journal.path.read_text().splitlines()
    journal.path.write_text("\n".join([lines[1], lines[0]]) + "\n")
    with pytest.raises(StateIntegrityError, match="edited or merged"):
        list(journal)


def test_fills_round_trip_through_the_journal():
    original = fill(AAA, Side.BUY, 10, 100.5, at(2026, 9, 21, 14), commission=1.25)
    assert fill_from_dict(fill_to_dict(original)) == original


# -- the replayed book -------------------------------------------------------


def test_the_book_is_rebuilt_from_opening_fills_and_adjustments(journal, portfolio):
    open_sleeve(journal, cash=10_000.0, positions=[
        {"instrument": "AAA", "quantity": 20, "average_cost": 50.0},
    ])
    journal.append(EventKind.FILL, at(2026, 9, 21, 14),
                   fill_to_dict(fill(BBB, Side.BUY, 10, 100.0, at(2026, 9, 21, 14))))
    journal.append(EventKind.ADJUSTMENT, at(2026, 9, 22), {
        "instrument": "AAA", "quantity": 25, "average_cost": 50.0,
        "cash_delta": 0.0, "reason": "broker shows 25 after a stock dividend",
    })
    book = sleeve_book(journal, portfolio)
    assert book.quantity(AAA) == 25
    assert book.quantity(BBB) == 10
    assert book.cash == pytest.approx(10_000 - 1_000 - 1.0)


def test_an_adjustment_without_a_reason_is_refused(journal, portfolio):
    open_sleeve(journal)
    journal.append(EventKind.ADJUSTMENT, at(2026, 9, 22), {"cash_delta": 50.0})
    with pytest.raises(StateIntegrityError, match="no reason"):
        sleeve_book(journal, portfolio)


def test_a_sleeve_without_an_opening_balance_cannot_be_read(journal, portfolio):
    with pytest.raises(StateIntegrityError, match="ql live init"):
        sleeve_book(journal, portfolio)


def test_a_sleeve_opens_exactly_once(journal, portfolio):
    open_sleeve(journal)
    open_sleeve(journal)
    with pytest.raises(StateIntegrityError, match="opens once"):
        sleeve_book(journal, portfolio)


def test_a_late_reported_fill_is_applied_in_journal_order(journal, portfolio):
    """The journal's order is authoritative; the fill's economics are unchanged."""
    open_sleeve(journal)
    journal.append(EventKind.NOTE, at(2026, 9, 21, 16), {"text": "later event"})
    early = fill(AAA, Side.BUY, 10, 100.0, at(2026, 9, 21, 14))
    journal.append(EventKind.FILL, at(2026, 9, 21, 17), fill_to_dict(early))
    book = sleeve_book(journal, portfolio)
    assert book.quantity(AAA) == 10
    assert book.positions[AAA].average_cost == 100.0


def test_the_fingerprint_changes_when_the_book_does(journal, portfolio):
    open_sleeve(journal)
    before = book_fingerprint(sleeve_book(journal, portfolio))
    journal.append(EventKind.FILL, at(2026, 9, 21, 14),
                   fill_to_dict(fill(AAA, Side.BUY, 1, 10.0, at(2026, 9, 21, 14))))
    after = book_fingerprint(sleeve_book(journal, portfolio))
    assert before != after
    assert after == book_fingerprint(sleeve_book(Journal(journal.path), portfolio)), (
        "the same journal always gives the same fingerprint"
    )


# -- configuration -----------------------------------------------------------


def write_config(tmp_path: Path, **overrides) -> Path:
    base = {
        "mode": "paper", "account": "DU1234567", "sleeve_capital": 25_000,
        "gateway": {"port": 4002},
    }
    base.update(overrides)
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump(base))
    return path


def test_the_example_config_is_valid():
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "configs" / "live.example.yaml")
    assert config.mode is TradingMode.PAPER


def test_a_live_mode_with_a_paper_account_is_refused(tmp_path):
    with pytest.raises(ContractViolation, match="does not look like a live"):
        load_config(write_config(tmp_path, mode="live", gateway={"port": 4001}))


def test_a_paper_mode_on_a_live_port_is_refused(tmp_path):
    with pytest.raises(ContractViolation, match="live port"):
        load_config(write_config(tmp_path, gateway={"port": 4001}))


def test_a_live_mode_on_a_paper_port_is_refused(tmp_path):
    with pytest.raises(ContractViolation, match="paper port"):
        load_config(write_config(
            tmp_path, mode="live", account="U7654321", gateway={"port": 4002}
        ))


def test_a_missing_setting_is_named(tmp_path):
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump({"mode": "paper", "account": "DU1"}))
    with pytest.raises(ContractViolation, match="sleeve_capital"):
        load_config(path)


def test_a_missing_file_says_what_to_do(tmp_path):
    with pytest.raises(ContractViolation, match="live.example.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_paper_and_live_keep_separate_journals(tmp_path):
    paper = load_config(write_config(tmp_path))
    live = load_config(write_config(
        tmp_path, mode="live", account="U7654321", gateway={"port": 4001}
    ))
    assert paper.journal_path != live.journal_path
    assert "paper" in paper.journal_path.name and "live" in live.journal_path.name


def test_unmanaged_tickers_are_normalised(tmp_path):
    config = load_config(write_config(tmp_path, unmanaged=["vwce", "Gld"]))
    assert config.unmanaged == frozenset({"VWCE", "GLD"})

