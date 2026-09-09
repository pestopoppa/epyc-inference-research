"""Native durable Journal read/ACK ownership for bounded consumers."""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autokernel import journal as journal_module


class FeedOwnerError(RuntimeError):
    """Native read/ACK ownership is busy, stale, or unavailable."""


class FeedOwnerPending(FeedOwnerError):
    """Bounded pre-admission work must be retried without side effects."""


class FeedOwnerBusy(FeedOwnerPending):
    """The native owner is temporarily held by another operation."""


@dataclass(frozen=True)
class DrainLimits:
    max_events: int = 32
    max_bytes: int = 4 * 1024 * 1024
    max_seconds: float = 0.25
    max_shards: int = 64

    def __post_init__(self) -> None:
        if (not isinstance(self.max_events, int) or isinstance(self.max_events, bool)
                or self.max_events < 1 or not isinstance(self.max_bytes, int)
                or isinstance(self.max_bytes, bool) or self.max_bytes < 1
                or not isinstance(self.max_seconds, (int, float))
                or isinstance(self.max_seconds, bool)
                or not math.isfinite(float(self.max_seconds)) or self.max_seconds <= 0
                or not isinstance(self.max_shards, int)
                or isinstance(self.max_shards, bool) or self.max_shards < 1):
            raise ValueError("drain limits must be positive")


@dataclass(frozen=True)
class TailBatch:
    events: tuple[journal_module.JournalEntry, ...]
    positions: tuple[tuple[int, int, int], ...]
    bytes_read: int
    source_frontier: int
    cursor_frontier: int
    proof_pending: bool = False

    @property
    def lag(self) -> int:
        return max(0, self.source_frontier - self.cursor_frontier)


@dataclass(frozen=True)
class OwnedTailBatch:
    tail: TailBatch
    durable_frontier: int
    owner_token: str

    def __post_init__(self) -> None:
        if (not isinstance(self.durable_frontier, int)
                or isinstance(self.durable_frontier, bool)
                or self.durable_frontier < 0):
            raise FeedOwnerError("durable frontier must be a non-negative integer")
        if not isinstance(self.owner_token, str) or not self.owner_token:
            raise FeedOwnerError("current read/ACK owner token is required")
        if any(row.seq > self.durable_frontier for row in self.tail.events):
            raise FeedOwnerError("visible Journal bytes exceed the owner's durable frontier")


class BoundedReadAckOwner(Protocol):
    def read_owned(self, reader_id: str, limits: DrainLimits, *,
                   deadline: float) -> OwnedTailBatch: ...

    def ack_owned(self, reader_id: str, seq: int,
                  position: tuple[int, int, int], *, owner_token: str,
                  deadline: float) -> None: ...


class JournalFeedOwner:
    """Exclusive reader owner backed by Journal's successful-fsync frontier.

    ``max_seconds`` is cooperative for regular-file I/O. Lock admission is
    nonblocking; a completed call that crossed its deadline is surfaced through
    ``last_overrun`` and is never described as a hard syscall deadline.
    """

    def __init__(self, journal: journal_module.Journal, reader_id: str,
                 *, clock=time.monotonic) -> None:
        self.journal = journal
        self.reader_id = reader_id
        self.clock = clock
        digest = hashlib.sha256(reader_id.encode()).hexdigest()[:24]
        self._lease_path = Path(journal.root) / f".feed-reader-{digest}.lock"
        self._lease_fd = os.open(
            self._lease_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600)
        lease_stat = os.fstat(self._lease_fd)
        if (not stat.S_ISREG(lease_stat.st_mode) or lease_stat.st_nlink != 1
                or lease_stat.st_uid != os.geteuid() or lease_stat.st_mode & 0o077):
            os.close(self._lease_fd)
            self._lease_fd = -1
            raise FeedOwnerError(
                "native feed owner lease must be private, singly-linked, and owned")
        try:
            fcntl.flock(self._lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._lease_fd)
            self._lease_fd = -1
            raise FeedOwnerError("another process owns this Journal feed reader") from exc
        self._lease_identity = (lease_stat.st_dev, lease_stat.st_ino,
                                lease_stat.st_mode, lease_stat.st_nlink)
        self._secret = secrets.token_bytes(32)
        self._guard = threading.Lock()
        self._issued: dict[str, tuple[
            OwnedTailBatch, dict[int, tuple[int, int, int, int, int, int]],
            journal_module.DurablePublication]] = {}
        self._current: OwnedTailBatch | None = None
        self._current_limits: DrainLimits | None = None
        self._current_acked = True
        self._checkpoint: dict[str, int] | None = None
        self.last_overrun: dict[str, float | str | int] | None = None
        try:
            # Small ownership/frontier check only.  Seal history is validated
            # resumably by bounded reads rather than scanned in this constructor.
            self.journal.durable_publication()
            self.journal.register_reader(reader_id)
        except Exception as exc:
            self.close()
            raise FeedOwnerError(f"native durable publication refused: {exc}") from exc

    def _validate_lease(self) -> None:
        if self._lease_fd < 0:
            raise FeedOwnerError("native feed owner is closed")
        descriptor = os.fstat(self._lease_fd)
        path = os.stat(self._lease_path, follow_symlinks=False)
        identity = (descriptor.st_dev, descriptor.st_ino,
                    descriptor.st_mode, descriptor.st_nlink)
        if identity != self._lease_identity or identity != (
                path.st_dev, path.st_ino, path.st_mode, path.st_nlink):
            raise FeedOwnerError("native feed owner lease path was replaced")

    def close(self) -> None:
        with self._guard:
            if self._lease_fd >= 0:
                fcntl.flock(self._lease_fd, fcntl.LOCK_UN)
                os.close(self._lease_fd)
                self._lease_fd = -1

    def _token(self, native: journal_module.DurableFeedBatch) -> str:
        body = {
            "reader_id": self.reader_id,
            "era": native.publication.era,
            "source": [native.publication.root_device, native.publication.root_inode],
            "durable_frontier": native.publication.durable_frontier,
            "cursor_frontier": native.cursor_frontier,
            "positions": native.positions,
            "events": [[row.seq, row.event_id] for row in native.entries],
        }
        return hmac.new(self._secret, json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode(),
            hashlib.sha256).hexdigest()

    def read_owned(self, reader_id: str, limits: DrainLimits, *,
                   deadline: float) -> OwnedTailBatch:
        if self.clock() > deadline:
            raise FeedOwnerPending("cooperative read deadline expired before admission")
        if not self._guard.acquire(blocking=False):
            raise FeedOwnerBusy("native feed owner is busy in another thread")
        try:
            self._validate_lease()
            if reader_id != self.reader_id:
                raise FeedOwnerError("reader is not the current native owner")
            if (self._current is not None and self._current.tail.events
                    and not self._current_acked):
                if self._current_limits != limits:
                    raise FeedOwnerError(
                        "prior owned batch awaits ACK under different bounds")
                return self._current
            native = self.journal.read_durable_batch(
                reader_id, max_events=limits.max_events, max_bytes=limits.max_bytes,
                max_shards=limits.max_shards)
            tail = TailBatch(native.entries, native.positions, native.bytes_read,
                             native.publication.durable_frontier,
                             native.cursor_frontier, native.proof_pending)
            token = self._token(native)
            owned = OwnedTailBatch(tail, native.publication.durable_frontier, token)
            identities = {}
            for position in native.positions:
                identities[position[0]] = self.journal.durable_shard_identity(position[0])
            self._issued = {token: (owned, identities, native.publication)}
            self._current = owned
            self._current_limits = limits
            self._current_acked = not bool(tail.events)
            if self.clock() > deadline:
                self.last_overrun = {"operation": "read", "deadline": deadline,
                                     "resolved_at": self.clock()}
            return owned
        except BlockingIOError as exc:
            raise FeedOwnerBusy("native Journal owner is busy") from exc
        except journal_module.JournalError as exc:
            raise FeedOwnerError(f"native durable publication refused: {exc}") from exc
        finally:
            self._guard.release()

    def ack_owned(self, reader_id: str, seq: int,
                  position: tuple[int, int, int], *, owner_token: str,
                  deadline: float) -> None:
        if self.clock() > deadline:
            raise FeedOwnerPending("cooperative ACK deadline expired before admission")
        if not self._guard.acquire(blocking=False):
            raise FeedOwnerBusy("native feed owner is busy in another thread")
        try:
            self._validate_lease()
            if reader_id != self.reader_id:
                raise FeedOwnerError("reader is not the current native owner")
            issued = self._issued.get(owner_token)
            if issued is None:
                raise FeedOwnerError("owner token is stale or was not issued here")
            owned, identities, issued_publication = issued
            matches = [(row.seq, pos) for row, pos in zip(
                owned.tail.events, owned.tail.positions, strict=True)]
            if (seq, position) not in matches:
                raise FeedOwnerError("ACK position was not returned by this owner token")
            publication = self.journal.durable_publication()
            if (publication.era != issued_publication.era
                    or (publication.root_device, publication.root_inode) != (
                        issued_publication.root_device, issued_publication.root_inode)):
                raise FeedOwnerError("publication ownership era changed")
            try:
                self.journal.ack_durable_cursor(
                    reader_id, seq, position, era=publication.era,
                    source_identity=(publication.root_device, publication.root_inode),
                    shard_identity=identities[position[0]],
                    max_shards=self._current_limits.max_shards)
            except BlockingIOError as exc:
                raise FeedOwnerBusy("native Journal cursor owner is busy") from exc
            except journal_module.CursorProofPending as exc:
                raise FeedOwnerPending("native durable ACK proof is pending") from exc
            except journal_module.JournalError as exc:
                raise FeedOwnerError(
                    f"native durable ACK refused: {exc}") from exc
            self._checkpoint = {
                "shard": position[0], "offset": position[1], "line": position[2],
                "device": identities[position[0]][0], "inode": identities[position[0]][1],
            }
            self._current_acked = True
            if self.clock() > deadline:
                self.last_overrun = {"operation": "ack", "seq": seq,
                                     "deadline": deadline, "resolved_at": self.clock()}
        finally:
            self._guard.release()

    def read(self, limits: DrainLimits) -> TailBatch:
        owned = self.read_owned(
            self.reader_id, limits, deadline=self.clock() + limits.max_seconds)
        self._current = owned
        return owned.tail

    def ack(self, seq: int, position: tuple[int, int, int]) -> None:
        if self._current is None:
            raise FeedOwnerError("ACK has no current owned batch")
        self.ack_owned(self.reader_id, seq, position,
                       owner_token=self._current.owner_token,
                       deadline=self.clock() + 0.25)

    def checkpoint(self) -> dict[str, int] | None:
        return None if self._checkpoint is None else dict(self._checkpoint)

    def restore(self, seq: int, checkpoint: dict[str, int]) -> None:
        cursor, raw = self.journal.durable_cursor_position(self.reader_id)
        expected = {
            "shard": (raw.get("feed_position") or [None])[0],
            "offset": (raw.get("feed_position") or [None, None])[1],
            "line": (raw.get("feed_position") or [None, None, None])[2],
            "device": (raw.get("feed_shard_identity") or [None])[0],
            "inode": (raw.get("feed_shard_identity") or [None, None])[1],
        }
        if cursor.last_seq != seq or expected != checkpoint:
            raise FeedOwnerError("projection checkpoint disagrees with native owned cursor")
        self._checkpoint = dict(checkpoint)
