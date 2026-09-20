"""The access seam and crash-safe state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from access.local import LocalOwner
from contracts.access import AccessPort, DiscretionMode
from contracts.errors import NotEntitled, StateIntegrityError
from contracts.identifiers import PortfolioId, StrategyId, StrategyVersion, TenantId
from runtime.snapshot import SNAPSHOT_FORMAT, SnapshotStore

VERSION = StrategyVersion.of(StrategyId("xs_momentum"), {"top_n": 4})


# -- access ------------------------------------------------------------------


def test_the_local_owner_satisfies_the_port() -> None:
    """The seam is real from day one, even with a trivial implementation behind it."""
    assert isinstance(LocalOwner.single_portfolio("user", "ibkr-main"), AccessPort)


def test_the_owner_may_run_anything_on_their_own_book(portfolio: PortfolioId) -> None:
    owner = LocalOwner.single_portfolio("user", "ibkr-main")
    owner.authorise(portfolio, VERSION, DiscretionMode.AUTOMATIC)
    owner.authorise(portfolio, VERSION, DiscretionMode.PROPOSE_AND_APPROVE)


def test_another_tenants_portfolio_is_refused() -> None:
    owner = LocalOwner.single_portfolio("user", "ibkr-main")
    stranger = PortfolioId(TenantId("someone"), "their-book")
    with pytest.raises(NotEntitled, match="does not operate"):
        owner.authorise(stranger, VERSION, DiscretionMode.AUTOMATIC)


def test_an_unentitled_strategy_is_refused(portfolio: PortfolioId, tenant: TenantId) -> None:
    """The shape a strategy catalogue takes later: entitlements, checked at the seam."""
    owner = LocalOwner(tenant, (portfolio,), allowed=frozenset({StrategyId("regime_hmm")}))
    with pytest.raises(NotEntitled, match="not entitled"):
        owner.authorise(portfolio, VERSION, DiscretionMode.AUTOMATIC)


def test_discretion_can_be_withheld(portfolio: PortfolioId, tenant: TenantId) -> None:
    """Automatic execution is a switch, not a code path — so it can be switched off."""
    supervised = LocalOwner(
        tenant, (portfolio,), discretion=frozenset({DiscretionMode.PROPOSE_AND_APPROVE})
    )
    supervised.authorise(portfolio, VERSION, DiscretionMode.PROPOSE_AND_APPROVE)
    with pytest.raises(NotEntitled, match="not permitted"):
        supervised.authorise(portfolio, VERSION, DiscretionMode.AUTOMATIC)


# -- state -------------------------------------------------------------------


def test_state_survives_a_round_trip(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "state.json")
    store.save({"positions": {"265598": 12}, "cash": 1234.56})
    recovered = store.load()
    assert recovered is not None
    assert recovered.state["positions"] == {"265598": 12}


def test_a_first_run_has_nothing_to_recover(tmp_path: Path) -> None:
    assert SnapshotStore(tmp_path / "absent.json").load() is None


def test_a_corrupted_snapshot_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """Resuming from a half-written position file is worse than not resuming."""
    path = tmp_path / "state.json"
    store = SnapshotStore(path)
    store.save({"positions": {"a": 1}})
    envelope = json.loads(path.read_text())
    envelope["payload"] = '{"positions":{"a":999}}'
    path.write_text(json.dumps(envelope))
    with pytest.raises(StateIntegrityError, match="checksum"):
        store.load()


def test_the_backup_rescues_a_damaged_primary(tmp_path: Path) -> None:
    """The cheapest redundancy: a copy of the thing you cannot afford to lose."""
    path = tmp_path / "state.json"
    store = SnapshotStore(path)
    store.save({"generation": 1})
    store.save({"generation": 2})
    path.write_text("{ truncated")
    recovered = store.load()
    assert recovered is not None
    assert recovered.state["generation"] == 1


def test_an_incompatible_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = SnapshotStore(path)
    store.save({"a": 1})
    envelope = json.loads(path.read_text())
    envelope["format"] = SNAPSHOT_FORMAT + 1
    path.write_text(json.dumps(envelope))
    with pytest.raises(StateIntegrityError, match="format"):
        store.load()


def test_unserialisable_state_fails_loudly(tmp_path: Path) -> None:
    """A silent failure would leave a stale snapshot looking current."""
    store = SnapshotStore(tmp_path / "state.json")
    with pytest.raises(StateIntegrityError, match="not serialisable"):
        store.save({"handle": {1, 2, 3}})


def test_no_partial_file_is_left_behind(tmp_path: Path) -> None:
    """Writes are atomic: readers see the whole old version or the whole new one."""
    path = tmp_path / "state.json"
    store = SnapshotStore(path)
    store.save({"a": 1})
    store.save({"a": 2})
    assert not list(tmp_path.glob("*.tmp"))
    assert store.load().state == {"a": 2}
