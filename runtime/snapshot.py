"""Persisted state that survives being killed at any instant.

The system must be safe to terminate at any point — power cut, crash, a kill
during a rotation. There is no orderly shutdown that correctness depends on.
That means two things:

**Writes are atomic.** State is written to a temporary file, flushed to disk, and
renamed over the target. A rename is atomic on POSIX, so a reader sees either the
whole previous version or the whole new one, never a half-written file. The
previous version is kept as a sibling, because the cheapest redundancy is a copy
of the thing you cannot afford to lose.

**Reads are verified.** Every snapshot carries a checksum of its payload. A
mismatch raises rather than returning a plausible-looking dictionary, because
resuming from a corrupted position file is worse than not resuming.

The order at startup matters and is not incidental: recover state, verify it,
*then* open the connection to the broker. A process that connects first and reads
its positions second can place an order while still believing it holds nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.errors import StateIntegrityError

#: Bumped when the on-disk shape changes incompatibly. A snapshot from an older
#: format is refused rather than guessed at.
SNAPSHOT_FORMAT = 1


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode(value: Any) -> str:
    """Serialise the few non-JSON types state legitimately carries.

    Deliberately narrow. A blanket ``default=str`` would make every object
    serialisable by turning it into its repr, so a set or a custom object would
    save without complaint and come back as a string that cannot be turned into
    the original. A snapshot that saves successfully and restores wrongly is worse
    than one that refuses to save.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(
        f"{type(value).__name__} has no defined snapshot encoding; convert it to a "
        f"primitive before saving so it can be restored exactly."
    )


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A verified state snapshot.

    Attributes:
        state: The recovered payload.
        written_at: When it was persisted.
        format: On-disk format version it was written with.
    """

    state: Mapping[str, Any]
    written_at: datetime
    format: int = SNAPSHOT_FORMAT


class SnapshotStore:
    """Atomic, checksummed state on disk.

    Args:
        path: Where the snapshot lives. Its backup sits beside it as ``.bak``.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._backup = self._path.with_suffix(self._path.suffix + ".bak")

    @property
    def path(self) -> Path:
        return self._path

    def save(self, state: Mapping[str, Any]) -> Snapshot:
        """Persist state atomically, keeping the previous version as a backup.

        Args:
            state: JSON-serialisable state.

        Raises:
            StateIntegrityError: If the state cannot be serialised, which would
                otherwise leave a stale snapshot silently in place.
        """
        written_at = datetime.now(timezone.utc)
        try:
            payload = json.dumps(state, sort_keys=True, separators=(",", ":"), default=_encode)
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError(f"state is not serialisable: {exc}") from exc

        envelope = {
            "format": SNAPSHOT_FORMAT,
            "written_at": written_at.isoformat(),
            "checksum": _digest(payload),
            "payload": payload,
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._backup.write_bytes(self._path.read_bytes())

        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(envelope, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self._path)
        return Snapshot(state=dict(state), written_at=written_at)

    def _read(self, path: Path) -> Snapshot:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateIntegrityError(f"{path.name} is unreadable: {exc}") from exc

        if envelope.get("format") != SNAPSHOT_FORMAT:
            raise StateIntegrityError(
                f"{path.name} is format {envelope.get('format')}, this build expects "
                f"{SNAPSHOT_FORMAT}. Migrate it deliberately rather than guessing."
            )
        payload = envelope.get("payload", "")
        if _digest(payload) != envelope.get("checksum"):
            raise StateIntegrityError(
                f"{path.name} failed its checksum: the file is truncated or corrupted. "
                f"Recovery stops here; reconcile against the broker before trading."
            )
        return Snapshot(
            state=json.loads(payload),
            written_at=datetime.fromisoformat(envelope["written_at"]),
            format=envelope["format"],
        )

    def load(self) -> Snapshot | None:
        """Recover state, falling back to the backup if the primary is damaged.

        Returns:
            The snapshot, or None if there is nothing to recover — a first run.

        Raises:
            StateIntegrityError: If both copies exist and both fail verification.
        """
        if not self._path.exists():
            return None
        try:
            return self._read(self._path)
        except StateIntegrityError:
            if not self._backup.exists():
                raise
            return self._read(self._backup)
