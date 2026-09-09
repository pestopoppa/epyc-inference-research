"""Bounded Journal -> Vidya ledger -> scoped-evidence projection.

The native AutoKernel Journal is the only WAL.  This module owns only a
rebuildable projection and the Journal's registered reader cursor.  A source
event is acknowledged after (and only after) every Vidya frame and the derived
projection have been fsynced.

Per-arm levels intentionally do not become effect findings.  A future producer
must journal an exact, prospective effect question and comparison receipt before
``finding_projector`` may return a :class:`Finding`; absent that seam the planner
view is explicitly ``unknown``.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
import fcntl
from pathlib import Path
from typing import Any, Callable, Mapping

from autokernel import journal as journal_module
from autokernel.loop import scoped_evidence as evidence
from autokernel.loop.journal_feed_owner import (
    BoundedReadAckOwner,
    DrainLimits,
    JournalFeedOwner,
    OwnedTailBatch,
    TailBatch,
)

STATE_SCHEMA = "epyc.autokernel.evidence_feed_checkpoint.v2"
READER_ID = "autokernel-vidya-evidence-v1"
PROJECTION_NAME = "evidence-index.json"
MAX_PROJECTION_BYTES = 1024 * 1024
__all__ = ["BoundedReadAckOwner", "DrainLimits", "OwnedTailBatch", "TailBatch",
           "EvidenceFeed", "EvidenceFeedWorker"]
_DB_COLUMNS = {
    "frames": ("id", "digest"),
    "events": ("id", "digest", "seq", "measurement_id"),
    "event_measurements": ("event_id", "measurement_id"),
    "measurements": ("id", "value"),
    "profile_terminals": ("id", "value", "seq"),
    "evicted": ("kind", "key"),
    "findings": ("id", "value", "frontier"),
    "invalidations": ("id", "value", "frontier"),
    "quarantines": ("id", "value", "frontier"),
}
class FeedError(RuntimeError):
    """The feed cannot safely advance its native cursor."""


class TailError(FeedError):
    """The cursor-addressed bounded tail is missing or corrupt."""


class FeedProjectionPending(FeedError):
    """Cooperative deadline passed; the retained exact projection can retry."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise FeedError("projection destination is a symlink")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _entry(raw: bytes, shard_index: int, line_number: int) -> journal_module.JournalEntry:
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TailError(f"shard {shard_index} line {line_number}: invalid JSON") from exc
    expected = {"journal_schema", "event_id", "seq", "kind", "campaign_id",
                "record_id", "written_at", "payload"}
    if not isinstance(obj, dict) or set(obj) != expected:
        raise TailError(f"shard {shard_index} line {line_number}: malformed envelope")
    if obj.get("journal_schema") != journal_module.JOURNAL_ENTRY_SCHEMA:
        raise TailError(f"shard {shard_index} line {line_number}: unsupported journal schema")
    if (not isinstance(obj.get("seq"), int) or isinstance(obj.get("seq"), bool)
            or obj["seq"] < 1 or not isinstance(obj.get("event_id"), str)
            or not obj["event_id"] or not isinstance(obj.get("kind"), str)
            or obj["kind"] not in journal_module.KINDS
            or not isinstance(obj.get("payload"), dict)):
        raise TailError(f"shard {shard_index} line {line_number}: invalid envelope fields")
    return journal_module.JournalEntry(
        event_id=obj["event_id"], seq=obj["seq"], kind=obj["kind"],
        campaign_id=obj["campaign_id"], record_id=obj["record_id"],
        written_at=obj["written_at"], payload=obj["payload"],
        shard_index=shard_index, line_number=line_number)


def _last_complete_line(path: str, max_bytes: int) -> bytes | None:
    """Read only the final newline-terminated record, tolerating a torn tail."""
    size = os.path.getsize(path)
    if size == 0:
        return None
    window = min(1 << 16, max_bytes)
    while True:
        window = min(window, size, max_bytes)
        with open(path, "rb") as handle:
            handle.seek(size - window)
            tail = handle.read(window)
        end = tail.rfind(b"\n")
        if end == -1:
            if window == size:
                return None
            if window == max_bytes:
                raise TailError(f"final journal record exceeds max_bytes={max_bytes}")
            window *= 4
            continue
        start = tail.rfind(b"\n", 0, end)
        if start == -1 and window < size:
            if window == max_bytes:
                raise TailError(f"final journal record exceeds max_bytes={max_bytes}")
            window *= 4
            continue
        line = tail[start + 1:end]
        if not line.strip():
            raise TailError(f"{path}: final complete journal line is blank")
        return line


@contextmanager
def _journal_snapshot_lock(journal: journal_module.Journal):
    """Take the native writer lock without waiting beyond a drain budget."""
    lock_path = Path(journal.root) / journal_module.LOCK_NAME
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TailError("native Journal writer is busy") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class BoundedJournalTail:
    """Fixture-only raw reader retained for fault comparison.

    Production ``EvidenceFeed`` uses :class:`JournalFeedOwner`; this helper
    deliberately has no successful-fsync publication authority.
    """

    def __init__(self, journal: journal_module.Journal, reader_id: str = READER_ID,
                 *, clock: Callable[[], float] = time.monotonic) -> None:
        self.journal = journal
        self.reader_id = reader_id
        self.clock = clock
        # A consumer never creates a source journal.  An absent source is not a
        # healthy empty campaign.
        self.journal.shards()
        self.journal.register_reader(reader_id)
        self._cached_cursor_seq: int | None = None
        self._cached_position: tuple[int, int, int] | None = None
        self._cached_identity: tuple[int, int] | None = None

    def _frontier(self, max_record_bytes: int) -> int:
        # Mirror Journal's append-side bounded tail lookup; do not call
        # read_all merely to report lag.
        for shard in reversed(self.journal.shards()):
            raw = _last_complete_line(shard.path, max_record_bytes)
            if raw is not None:
                return _entry(raw, shard.index, -1).seq
        return 0

    def _locate(self, last_seq: int, max_record_bytes: int, *,
                shards: list[Any] | None = None) -> tuple[int, int, int]:
        """One initialization/validation scan to the exact cursor boundary."""
        shards = self.journal.shards() if shards is None else shards
        if last_seq == 0:
            return shards[0].index, 0, 0
        previous = 0
        for shard in shards:
            offset = 0
            line_number = 0
            with open(shard.path, "rb") as handle:
                while True:
                    raw = handle.readline(max_record_bytes + 1)
                    if not raw:
                        break
                    if len(raw) > max_record_bytes:
                        raise TailError(
                            f"journal record exceeds max_bytes={max_record_bytes}")
                    line_number += 1
                    if not raw.endswith(b"\n"):
                        if shard.index == shards[-1].index:
                            break
                        raise TailError("non-final shard has a torn tail")
                    row = _entry(raw[:-1], shard.index, line_number)
                    if row.seq != previous + 1:
                        raise TailError(
                            f"journal sequence discontinuity {previous} -> {row.seq}")
                    previous = row.seq
                    offset += len(raw)
                    if row.seq == last_seq:
                        stat_result = os.fstat(handle.fileno())
                        self._cached_identity = (stat_result.st_dev, stat_result.st_ino)
                        return shard.index, offset, line_number
                    if row.seq > last_seq:
                        raise TailError(f"cursor {last_seq} is not present in the journal")
        raise TailError(f"cursor {last_seq} is beyond source frontier {previous}")

    def read(self, limits: DrainLimits) -> TailBatch:
        deadline = self.clock() + limits.max_seconds
        # Hold append exclusion only while capturing the small control snapshot.
        # Cursor recovery/location and bounded record reads can be expensive and
        # deliberately run after the writer lock is released.
        with _journal_snapshot_lock(self.journal):
            cursor = self.journal.cursor(self.reader_id)
            if cursor is None:
                raise TailError(f"registered cursor {self.reader_id!r} disappeared")
            if (cursor.reader_id != self.reader_id
                    or not isinstance(cursor.last_seq, int)
                    or isinstance(cursor.last_seq, bool) or cursor.last_seq < 0):
                raise TailError("registered cursor identity/position is malformed")
            shards = self.journal.shards()
            source_frontier = self._frontier(limits.max_bytes)
        by_index = {shard.index: shard for shard in shards}
        cached_ok = False
        if self._cached_cursor_seq == cursor.last_seq and self._cached_position is not None:
            cached_shard = by_index.get(self._cached_position[0])
            if cached_shard is not None:
                current = os.stat(cached_shard.path, follow_symlinks=False)
                cached_ok = (self._cached_identity == (current.st_dev, current.st_ino)
                             and current.st_size >= self._cached_position[1])
        if cached_ok:
            start_shard, start_offset, start_line = self._cached_position
        else:
            start_shard, start_offset, start_line = self._locate(
                cursor.last_seq, limits.max_bytes, shards=shards)
            self._cached_cursor_seq = cursor.last_seq
            self._cached_position = (start_shard, start_offset, start_line)
        rows: list[journal_module.JournalEntry] = []
        positions: list[tuple[int, int, int]] = []
        used = 0
        expected = cursor.last_seq + 1
        for shard in shards:
            if shard.index < start_shard:
                continue
            offset = start_offset if shard.index == start_shard else 0
            line_number = start_line if shard.index == start_shard else 0
            with open(shard.path, "rb") as handle:
                handle.seek(offset)
                while (len(rows) < limits.max_events and self.clock() <= deadline
                       and expected <= source_frontier):
                    remaining = limits.max_bytes - used
                    if remaining <= 0:
                        break
                    raw = handle.readline(remaining + 1)
                    if not raw:
                        break
                    if len(raw) > remaining:
                        if not rows:
                            raise TailError(
                                f"next journal record exceeds max_bytes={limits.max_bytes}")
                        break
                    if not raw.endswith(b"\n"):
                        if shard.index != shards[-1].index:
                            raise TailError("non-final shard has a torn tail")
                        break
                    line_number += 1
                    row = _entry(raw[:-1], shard.index, line_number)
                    if row.seq != expected:
                        raise TailError(
                            f"journal sequence discontinuity {expected - 1} -> {row.seq}")
                    rows.append(row)
                    used += len(raw)
                    positions.append((shard.index, handle.tell(), line_number))
                    expected += 1
                if (len(rows) >= limits.max_events or used >= limits.max_bytes
                        or self.clock() > deadline or expected > source_frontier):
                    break
        return TailBatch(tuple(rows), tuple(positions), used,
                         source_frontier, cursor.last_seq)

    def frontier(self, max_record_bytes: int) -> int:
        """Read the visible complete-line frontier while excluding append."""
        with _journal_snapshot_lock(self.journal):
            return self._frontier(max_record_bytes)

    def ack(self, seq: int, position: tuple[int, int, int]) -> None:
        """Update the in-process offset only after Journal cursor durability."""
        self._cached_cursor_seq = seq
        self._cached_position = position
        shard = next(item for item in self.journal.shards() if item.index == position[0])
        stat_result = os.stat(shard.path, follow_symlinks=False)
        self._cached_identity = (stat_result.st_dev, stat_result.st_ino)

    def restore(self, seq: int, checkpoint: Mapping[str, Any]) -> None:
        expected = {"shard", "offset", "line", "device", "inode"}
        if (not isinstance(checkpoint, Mapping) or set(checkpoint) != expected
                or any(not isinstance(checkpoint[name], int)
                       or isinstance(checkpoint[name], bool) or checkpoint[name] < 0
                       for name in expected)):
            raise TailError("persisted tail position is malformed")
        shard = next((item for item in self.journal.shards()
                      if item.index == checkpoint["shard"]), None)
        if shard is None:
            raise TailError("persisted tail shard is unavailable")
        stat_result = os.stat(shard.path, follow_symlinks=False)
        if ((stat_result.st_dev, stat_result.st_ino)
                != (checkpoint["device"], checkpoint["inode"])
                or stat_result.st_size < checkpoint["offset"]):
            raise TailError("persisted tail shard identity/size changed")
        if seq:
            window = min(checkpoint["offset"], 4 * 1024 * 1024)
            with open(shard.path, "rb") as handle:
                handle.seek(checkpoint["offset"] - window)
                raw = handle.read(window)
            if not raw.endswith(b"\n"):
                raise TailError("persisted tail offset is not a record boundary")
            prior = raw.rfind(b"\n", 0, len(raw) - 1)
            if prior < 0 and checkpoint["offset"] > window:
                raise TailError("persisted tail record exceeds validation bound")
            row = _entry(raw[prior + 1:-1], checkpoint["shard"], checkpoint["line"])
            if row.seq != seq:
                raise TailError("persisted tail position disagrees with cursor")
        self._cached_cursor_seq = seq
        self._cached_position = (
            checkpoint["shard"], checkpoint["offset"], checkpoint["line"])
        self._cached_identity = (checkpoint["device"], checkpoint["inode"])

    def checkpoint(self) -> dict[str, int] | None:
        if self._cached_position is None or self._cached_identity is None:
            return None
        return {"shard": self._cached_position[0], "offset": self._cached_position[1],
                "line": self._cached_position[2], "device": self._cached_identity[0],
                "inode": self._cached_identity[1]}


def _load_vidya(root_repo: Path) -> tuple[Any, Any, Any, Any, Any]:
    vidya = root_repo / "scripts" / "vidya"
    if not vidya.is_dir():
        raise FeedError(f"explicit root repo has no scripts/vidya: {root_repo}")
    value = str(vidya)
    if value not in sys.path:
        sys.path.insert(0, value)
    adapter = importlib.import_module("adapters.autokernel_unified_arm")
    profile_adapter = importlib.import_module("adapters.autokernel_profile")
    claim_tuple = importlib.import_module("claim_tuple")
    frames = importlib.import_module("frames")
    ledger = importlib.import_module("ledger")
    expected_root = root_repo.resolve()
    for module in (adapter, profile_adapter, claim_tuple, frames, ledger):
        module_path = Path(module.__file__).resolve()
        if expected_root not in module_path.parents:
            raise FeedError(
                f"loaded {module.__name__} from {module_path}, outside explicit root repo")
    return adapter, profile_adapter, claim_tuple, frames, ledger


class EvidenceFeed:
    """Single-writer, restartable projection consumer."""

    def __init__(self, **kwargs) -> None:
        self._lock_fd = -1
        try:
            self._initialize(**kwargs)
        except BaseException:
            # SQLite and both leases must be released by their creating thread,
            # including failures midway through recovery/module loading.
            self.close()
            raise

    def _initialize(self, *, source_root: Path, corpus_root: Path, ledger_path: Path,
                 store_root: Path, root_repo: Path, current_epoch: str,
                 reader_id: str = READER_ID,
                 finding_projector: Callable[..., evidence.Finding | None] | None = None,
                 scope_verifier: Callable[..., bool | str] | None = None,
                 use_verifier: Callable[..., bool | str] | None = None,
                 result_verifier: Callable[..., bool | str] | None = None,
                 support_rule_identity: str | None = None,
                 loaded_projection: Any = None,
                 max_projection_entries: int = 10_000) -> None:
        for label, path in (("source_root", source_root), ("corpus_root", corpus_root),
                            ("root_repo", root_repo)):
            if not Path(path).is_absolute():
                raise ValueError(f"{label} must be an explicit absolute path")
        if not Path(ledger_path).is_absolute() or not Path(store_root).is_absolute():
            raise ValueError("ledger_path and store_root must be explicit absolute paths")
        if (not isinstance(max_projection_entries, int)
                or isinstance(max_projection_entries, bool)
                or not 1 <= max_projection_entries <= 10_000):
            raise ValueError("max_projection_entries must be an integer in [1, 10000]")
        self.store_root = Path(store_root)
        self.store_root.mkdir(parents=True, exist_ok=True)
        if self.store_root.is_symlink() or not self.store_root.is_dir():
            raise FeedError("store_root must be a real directory, not a symlink")
        lock_path = self.store_root / ".evidence-feed.lock"
        self._lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT
                                | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._lock_fd)
            self._lock_fd = -1
            raise FeedError("another evidence feed owns this projection") from exc
        self.journal = journal_module.Journal(str(source_root))
        # The projection lock protects its files; this reader lease protects the
        # shared Journal cursor even when a misconfigured second instance names
        # a different store root.
        self.journal.shards()
        if not isinstance(reader_id, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*", reader_id):
            self.close()
            raise FeedError("reader_id is malformed")
        try:
            self.tail = JournalFeedOwner(self.journal, reader_id)
        except Exception as exc:
            self.close()
            raise FeedError("another evidence feed owns this Journal reader") from exc
        self.reader_id = reader_id
        self.corpus_root = Path(corpus_root)
        self.store_path = self.store_root / PROJECTION_NAME
        if loaded_projection is None:
            (self.adapter, self.profile_adapter, self.claim_tuple,
             self.frames, ledger_mod) = _load_vidya(Path(root_repo))
        else:
            from .feed_runtime import LoadedFeedProjection
            if not isinstance(loaded_projection, LoadedFeedProjection):
                raise FeedError("feed projection must be a concrete loaded source closure")
            self.adapter = loaded_projection.adapter
            self.profile_adapter = loaded_projection.profile_adapter
            self.claim_tuple = loaded_projection.claim_tuple
            self.frames = loaded_projection.frames
            ledger_mod = loaded_projection.ledger
        self.ledger_module = ledger_mod
        self.ledger = ledger_mod.Ledger(ledger_path)
        self.current_epoch = current_epoch
        self.finding_projector = finding_projector
        self._index_kwargs = {
            "scope_verifier": scope_verifier, "use_verifier": use_verifier,
            "result_verifier": result_verifier,
            "support_rule_identity": support_rule_identity,
        }
        self.max_projection_entries = max_projection_entries
        self._pending_ack_reconciliation: int | None = None
        self._pending_db_reconciliation = False
        self._recovery_batch: TailBatch | None = None
        self._failed = False
        self.state = self._load_state()
        cursor = self.journal.cursor(self.reader_id)
        if (cursor is not None and cursor.last_seq == self.state["acknowledged_frontier"]
                and self.state["ack_position"] is not None):
            self.tail.restore(cursor.last_seq, self.state["ack_position"])
        self.identity_path = self.store_root / "evidence-identities.sqlite3"
        if self.identity_path.is_symlink():
            raise FeedError("derived identity index must not be a symlink")
        self._db = sqlite3.connect(self.identity_path)
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                schema TEXT NOT NULL, source_device INTEGER NOT NULL,
                source_inode INTEGER NOT NULL, frontier INTEGER NOT NULL,
                projection_digest TEXT NOT NULL, derived_digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS frames (id TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, digest TEXT NOT NULL, seq INTEGER NOT NULL,
                measurement_id TEXT);
            CREATE TABLE IF NOT EXISTS event_measurements (
                event_id TEXT NOT NULL, measurement_id TEXT NOT NULL,
                PRIMARY KEY(event_id, measurement_id));
            CREATE TABLE IF NOT EXISTS measurements (
                id TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS profile_terminals (
                id TEXT PRIMARY KEY, value TEXT NOT NULL, seq INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS evicted (
                kind TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY(kind, key));
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY, value TEXT NOT NULL, frontier INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS invalidations (
                id TEXT PRIMARY KEY, value TEXT NOT NULL, frontier INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS quarantines (
                id TEXT PRIMARY KEY, value TEXT NOT NULL, frontier INTEGER NOT NULL);
        """)
        self._ledger_frames: OrderedDict[str, str] = OrderedDict()
        self._ledger_identity: tuple[int, int, int] | None = None
        self._ledger_offset = 0
        self._ledger_next_seq = 0
        self._ledger_link = ledger_mod.GENESIS_PREV_HASH
        self._events: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._measurements: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._findings: OrderedDict[str, evidence.Finding] = OrderedDict()
        self._invalidations: deque[dict[str, Any]] = deque(
            maxlen=self.max_projection_entries)
        self._quarantines: deque[dict[str, Any]] = deque(
            maxlen=self.max_projection_entries)
        self._global_quarantine = False
        self._diagnostics: deque[dict[str, Any]] = deque(
            maxlen=self.max_projection_entries)
        self._db_xor = 0
        self._open_identity_projection()
        self._clear_derived_table("frames")
        self._refresh_ledger_index(force=True)
        self._load_bounded_projection()
        self._db.commit()
        if self._pending_db_reconciliation:
            self._save_state()
            self._pending_db_reconciliation = False
        if self._pending_ack_reconciliation is not None:
            # A crash after the Journal cursor commit but before the final
            # projection save leaves the cursor one ahead of the persisted ACK.
            # Reconcile only after the native prefix, Ledger identities, cached
            # projection, and rebuilt EvidenceIndex have all validated.  The
            # source Journal remains the sole WAL throughout recovery.
            self.state["acknowledged_frontier"] = self._pending_ack_reconciliation
            self.state["ack_position"] = None
            self._save_state()
            self._pending_ack_reconciliation = None

    def _open_identity_projection(self) -> None:
        source = self.state["source_identity"]
        meta = self._db.execute(
            "SELECT schema, source_device, source_inode, frontier, projection_digest, "
            "derived_digest "
            "FROM meta WHERE singleton = 1").fetchone()
        if meta is None:
            if self.state["projected_frontier"] != 0:
                raise FeedError("derived identity projection is missing after cursor advance")
            self._db_xor = self._compute_derived_xor()
            if self._db_xor:
                raise FeedError("derived identity rows exist without their bound metadata")
            self._db.execute(
                "INSERT INTO meta VALUES (1, ?, ?, ?, ?, ?, ?)",
                (STATE_SCHEMA, source["device"], source["inode"], 0,
                 self.state["projection_digest"], self.state["derived_digest"]))
            return
        expected = (STATE_SCHEMA, source["device"], source["inode"],
                    self.state["projected_frontier"], self.state["projection_digest"],
                    self.state["derived_digest"])
        if tuple(meta) != expected:
            same_identity = tuple(meta[:3]) == expected[:3]
            db_frontier, db_digest = meta[3], meta[4]
            if (same_identity and db_frontier == self.state["projected_frontier"] + 1):
                batch = self.tail.read(DrainLimits(
                    max_events=1, max_bytes=4 * 1024 * 1024, max_seconds=1))
                if not batch.events or batch.events[0].seq != db_frontier:
                    raise FeedError("DB-ahead projection lacks its unacknowledged Journal row")
                expected_digest = _digest({
                    "prior": self.state["projection_digest"], "seq": db_frontier,
                    "event": _digest(batch.events[0].envelope())})
                if expected_digest != db_digest:
                    raise FeedError("DB-ahead projection digest does not bind Journal row")
                self.state["projected_frontier"] = db_frontier
                self.state["projection_digest"] = db_digest
                self.state["derived_digest"] = meta[5]
                self._pending_db_reconciliation = True
                self._recovery_batch = batch
            else:
                raise FeedError("derived identity projection disagrees with durable checkpoint")
        integrity = self._db.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise FeedError("derived identity projection fails SQLite integrity check")
        self._db_xor = self._compute_derived_xor()
        if f"{self._db_xor:064x}" != self.state["derived_digest"]:
            raise FeedError("derived identity projection content checksum mismatch")

    @staticmethod
    def _derived_row_token(table: str, row: tuple[Any, ...]) -> int:
        return int(hashlib.sha256(_canonical({"table": table, "row": row})).hexdigest(), 16)

    def _compute_derived_xor(self) -> int:
        result = 0
        for table, columns in _DB_COLUMNS.items():
            query = f"SELECT {', '.join(columns)} FROM {table} ORDER BY {', '.join(columns)}"
            for row in self._db.execute(query):
                result ^= self._derived_row_token(table, tuple(row))
        return result

    def _set_derived_row(self, table: str, key: tuple[Any, ...],
                         row: tuple[Any, ...]) -> None:
        columns = _DB_COLUMNS[table]
        key_columns = columns[:len(key)]
        where = " AND ".join(f"{column} = ?" for column in key_columns)
        prior = self._db.execute(
            f"SELECT {', '.join(columns)} FROM {table} WHERE {where}", key).fetchone()
        if prior is not None:
            self._db_xor ^= self._derived_row_token(table, tuple(prior))
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{column}=excluded.{column}"
                            for column in columns[len(key):])
        action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
        self._db.execute(
            f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT({', '.join(key_columns)}) {action}", row)
        self._db_xor ^= self._derived_row_token(table, row)
        self.state["derived_digest"] = f"{self._db_xor:064x}"

    def _clear_derived_table(self, table: str) -> None:
        columns = _DB_COLUMNS[table]
        for row in self._db.execute(f"SELECT {', '.join(columns)} FROM {table}"):
            self._db_xor ^= self._derived_row_token(table, tuple(row))
        self._db.execute(f"DELETE FROM {table}")
        self.state["derived_digest"] = f"{self._db_xor:064x}"

    def _load_bounded_projection(self) -> None:
        for event_id, digest, seq, measurement_id in reversed(self._db.execute(
                "SELECT id, digest, seq, measurement_id FROM events "
                "ORDER BY seq DESC LIMIT ?", (self.max_projection_entries,)).fetchall()):
            self._bounded_put(self._events, event_id, {
                "digest": digest, "seq": seq, "measurement_id": measurement_id})
        for measurement_id, value in self._db.execute(
                "SELECT m.id, m.value FROM measurements m JOIN ("
                "SELECT id AS event_id, measurement_id, seq FROM events "
                "WHERE measurement_id IS NOT NULL UNION "
                "SELECT a.event_id, a.measurement_id, e.seq FROM event_measurements a "
                "JOIN events e ON e.id=a.event_id) x ON x.measurement_id=m.id "
                "ORDER BY x.seq DESC LIMIT ?",
                (self.max_projection_entries,)):
            self._bounded_put(self._measurements, measurement_id, json.loads(value))
        for _finding_id, value, _frontier in reversed(self._db.execute(
                "SELECT id, value, frontier FROM findings ORDER BY frontier DESC LIMIT ?",
                (self.max_projection_entries,)).fetchall()):
            finding = evidence.Finding.from_dict(json.loads(value))
            self._findings[finding.finding_id] = finding
        for table, target in (("invalidations", self._invalidations),
                              ("quarantines", self._quarantines)):
            rows = self._db.execute(
                f"SELECT value FROM {table} ORDER BY frontier DESC LIMIT ?",
                (self.max_projection_entries,)).fetchall()
            target.extend(json.loads(row[0]) for row in reversed(rows))
        self._global_quarantine = self._db.execute(
            "SELECT 1 FROM quarantines WHERE json_extract(value, '$.global_scope') = 1 "
            "LIMIT 1").fetchone() is not None
        self._rebuild_bounded_index()

    def _bounded_put(self, cache: OrderedDict[str, Any], key: str, value: Any) -> Any:
        cache[key] = value
        cache.move_to_end(key)
        evicted = None
        if len(cache) > self.max_projection_entries:
            evicted = cache.popitem(last=False)
        return evicted

    def _rebuild_bounded_index(self) -> None:
        kwargs = ({key: value for key, value in self._index_kwargs.items()
                   if value is not None} if self.finding_projector is not None else {})
        self._index = evidence.EvidenceIndex(
            self._findings.values(), self._invalidations,
            current_epoch=self.current_epoch,
            projection_available=(self.state["readiness"] != "outage"
                                  and not self._global_quarantine), **kwargs)
        for item in self._quarantines:
            self._index.ingest_quarantine(item)

    def close(self) -> None:
        if getattr(self, "_db", None) is not None:
            self._db.close()
            self._db = None
        if getattr(self, "tail", None) is not None:
            self.tail.close()
            self.tail = None
        if getattr(self, "_lock_fd", -1) >= 0:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1

    def __enter__(self) -> "EvidenceFeed":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort failed-init cleanup
        try:
            self.close()
        except Exception:
            pass

    def _empty_state(self) -> dict[str, Any]:
        source_stat = os.stat(self.journal.root, follow_symlinks=False)
        return {"schema": STATE_SCHEMA, "reader_id": self.reader_id,
                "source_root": self.journal.root, "current_epoch": self.current_epoch,
                "source_identity": {"device": source_stat.st_dev,
                                    "inode": source_stat.st_ino},
                "source_frontier": 0, "projected_frontier": 0,
                "acknowledged_frontier": 0,
                "projection_digest": _digest([]),
                "derived_digest": "0" * 64,
                "ack_position": None,
                "readiness": "unknown", "diagnostic_count": 0,
                "quarantine_count": 0, "last_diagnostic": None,
                "last_quarantine": None, "checksum": ""}

    def _load_state(self) -> dict[str, Any]:
        if not self.store_path.exists():
            cursor = self.journal.cursor(self.reader_id)
            if cursor is None or cursor.last_seq != 0:
                raise FeedError("projection is missing while the native cursor has advanced")
            return self._empty_state()
        if self.store_path.is_symlink():
            raise FeedError("projection state must not be a symlink")
        if self.store_path.stat().st_size > MAX_PROJECTION_BYTES:
            raise FeedError("projection state exceeds its byte bound")
        try:
            state = json.loads(self.store_path.read_text())
            expected = {"schema", "reader_id", "source_root", "current_epoch",
                        "source_identity",
                        "source_frontier", "projected_frontier",
                        "acknowledged_frontier", "projection_digest", "derived_digest",
                        "ack_position",
                        "readiness",
                        "diagnostic_count", "quarantine_count",
                        "last_diagnostic", "last_quarantine", "checksum"}
            if not isinstance(state, dict) or set(state) != expected:
                raise ValueError("projection state has missing or unknown fields")
            checksum = state.pop("checksum")
            if (_digest(state) != checksum or state.get("schema") != STATE_SCHEMA
                    or state.get("reader_id") != self.reader_id
                    or state.get("source_root") != self.journal.root
                    or state.get("current_epoch") != self.current_epoch):
                raise ValueError("identity or checksum mismatch")
            source_stat = os.stat(self.journal.root, follow_symlinks=False)
            if state.get("source_identity") != {
                    "device": source_stat.st_dev, "inode": source_stat.st_ino}:
                raise ValueError("native source incarnation changed")
            state["checksum"] = checksum
            cursor = self.journal.cursor(self.reader_id)
            if cursor is None:
                raise ValueError("registered source cursor disappeared")
            projected = state.get("projected_frontier")
            acknowledged = state.get("acknowledged_frontier")
            if (not isinstance(projected, int) or isinstance(projected, bool)
                    or not isinstance(acknowledged, int) or isinstance(acknowledged, bool)
                    or not ((cursor.last_seq == acknowledged == projected)
                            or (cursor.last_seq == acknowledged
                                and projected == cursor.last_seq + 1)
                            or (projected == cursor.last_seq
                                and cursor.last_seq == acknowledged + 1))):
                raise ValueError(
                    "source cursor rolled back or advanced beyond the durable projection")
            if (projected == cursor.last_seq
                    and cursor.last_seq == acknowledged + 1):
                self._pending_ack_reconciliation = cursor.last_seq
            return state
        except Exception as exc:
            raise FeedError(f"projection state is corrupt: {exc}") from exc

    def _save_state(self) -> None:
        self._db.execute(
            "UPDATE meta SET frontier = ?, projection_digest = ?, derived_digest = ? "
            "WHERE singleton = 1",
            (self.state["projected_frontier"], self.state["projection_digest"],
             self.state["derived_digest"]))
        self._db.commit()
        unsigned = {key: value for key, value in self.state.items() if key != "checksum"}
        self.state["checksum"] = _digest(unsigned)
        if len(_canonical(self.state)) + 1 > MAX_PROJECTION_BYTES:
            raise FeedError("projection state exceeds its byte bound")
        try:
            _atomic_json(self.store_path, self.state)
        except Exception:
            self._failed = True
            raise

    def _refresh_ledger_index(self, *, force: bool = False) -> None:
        try:
            stat_result = self.ledger.path.stat()
            identity = (stat_result.st_dev, stat_result.st_ino, stat_result.st_size)
        except FileNotFoundError:
            identity = None
        if not force and identity == self._ledger_identity:
            return
        rotated = (identity is None or self._ledger_identity is None
                   or identity[:2] != self._ledger_identity[:2]
                   or identity[2] < self._ledger_offset)
        if force or rotated:
            self._ledger_frames = OrderedDict()
            self._ledger_offset = 0
            self._ledger_next_seq = 0
            self._ledger_link = self.ledger_module.GENESIS_PREV_HASH
            for record, size in self._iter_ledger_records():
                self._accept_ledger_record(record)
                self._ledger_offset += size
        elif identity[2] > self._ledger_offset:
            with open(self.ledger.path, "rb") as handle:
                handle.seek(self._ledger_offset)
                while True:
                    start = handle.tell()
                    raw = handle.readline()
                    if not raw or not raw.endswith(b"\n"):
                        break
                    try:
                        record = self.ledger_module.LedgerRecord.from_obj(json.loads(raw))
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        raise FeedError("incremental Vidya ledger tail is malformed") from exc
                    self._accept_ledger_record(record)
                    self._ledger_offset = handle.tell()
                    if self._ledger_offset <= start:  # pragma: no cover - defensive
                        raise FeedError("Vidya ledger tail made no progress")
        try:
            stat_result = self.ledger.path.stat()
            self._ledger_identity = (stat_result.st_dev, stat_result.st_ino,
                                     stat_result.st_size)
        except FileNotFoundError:
            self._ledger_identity = None

    def _iter_ledger_records(self):
        if not self.ledger.path.exists():
            return
        with open(self.ledger.path, "rb") as handle:
            while True:
                raw = handle.readline()
                if not raw:
                    return
                if not raw.endswith(b"\n"):
                    return
                if not raw.strip():
                    continue
                try:
                    record = self.ledger_module.LedgerRecord.from_obj(json.loads(raw))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise FeedError("Vidya ledger record is malformed") from exc
                yield record, len(raw)

    def _accept_ledger_record(self, record: Any) -> None:
        if (record.seq != self._ledger_next_seq or record.prev_hash != self._ledger_link
                or record.frame_hash != self.ledger_module.content_hash(record.frame)):
            raise FeedError("Vidya ledger tail fails sequence/hash-chain validation")
        frame_id = record.frame.get("frame_id")
        if isinstance(frame_id, str):
            digest = _digest(record.frame)
            prior = self._ledger_frames.get(frame_id)
            if prior is not None and prior != digest:
                raise FeedError(f"Vidya ledger reuses frame_id {frame_id!r}")
            self._set_derived_row("frames", (frame_id,), (frame_id, digest))
            self._bounded_put(self._ledger_frames, frame_id, digest)
        self._ledger_next_seq += 1
        self._ledger_link = self.ledger_module._link_hash(
            record.prev_hash, record.frame_hash, record.seq)

    def _ledger_contains(self, frame_id: str, digest: str) -> bool:
        cached = self._ledger_frames.get(frame_id)
        if cached is not None:
            if cached != digest:
                raise FeedError(f"conflicting Vidya frame {frame_id}")
            return True
        found = self._db.execute(
            "SELECT digest FROM frames WHERE id = ?", (frame_id,)).fetchone()
        if found is None:
            return False
        if found[0] != digest:
            raise FeedError(f"conflicting Vidya frame {frame_id}")
        self._bounded_put(self._ledger_frames, frame_id, digest)
        return True

    def _append_frames(self, frames: list[dict[str, Any]], *, replay: bool = False) -> None:
        # Sequential peer appends are tailed from the last verified byte.  The
        # underlying Ledger remains explicitly single-writer; it exposes no
        # cross-process publication lock, so concurrent writers are not claimed safe.
        self._refresh_ledger_index()
        for frame in frames:
            frame_id = frame["frame_id"]
            digest = _digest(frame)
            if self._ledger_contains(frame_id, digest):
                continue
            if replay:
                raise FeedError(f"projected Vidya frame {frame_id} is absent from ledger")
            self.ledger.append(frame)
            self._refresh_ledger_index()

    def _diagnose(self, event_id: str, reason: str, *, replay: bool) -> None:
        item = {"event_id": event_id, "reason": reason}
        self._diagnostics.append(item)
        if not replay:
            self.state["diagnostic_count"] += 1
            self.state["last_diagnostic"] = item

    def _cache_finding(self, finding: evidence.Finding) -> None:
        self._set_derived_row(
            "findings", (finding.finding_id,),
            (finding.finding_id, _canonical(finding.to_dict()).decode(), finding.frontier))
        evicted = self._bounded_put(self._findings, finding.finding_id, finding)
        if evicted is None:
            self._index.ingest_finding(finding)
            return
        old = evicted[1]
        for dep in old.claim_key.dependency_identities:
            self._set_derived_row("evicted", ("dependency", dep), ("dependency", dep))
        if old.conclusion in {"refutation", "conflict", "retraction"}:
            signature = evidence._mandatory_signature_digest(old.claim_key)
            self._set_derived_row("evicted", ("signature", signature),
                                  ("signature", signature))
        self._rebuild_bounded_index()

    def _cache_invalidation(self, item: dict[str, Any]) -> None:
        self._set_derived_row(
            "invalidations", (item["event_id"],),
            (item["event_id"], _canonical(item).decode(), item["frontier"]))
        evicted = self._invalidations[0] if len(self._invalidations) == self.max_projection_entries else None
        self._invalidations.append(item)
        if evicted is not None:
            dep = evicted["dependency_id"]
            self._set_derived_row("evicted", ("dependency", dep), ("dependency", dep))
            self._rebuild_bounded_index()
        else:
            self._index.ingest_invalidation(item)

    def _cache_quarantine(self, item: dict[str, Any]) -> None:
        self._set_derived_row(
            "quarantines", (item["event_id"],),
            (item["event_id"], _canonical(item).decode(), item["frontier"]))
        evicted = self._quarantines[0] if len(self._quarantines) == self.max_projection_entries else None
        self._quarantines.append(item)
        if evicted is not None:
            for dep in evicted["affected_dependencies"]:
                self._set_derived_row(
                    "evicted", ("dependency", dep), ("dependency", dep))
            if evicted["global_scope"]:
                self._global_quarantine = True
            self._rebuild_bounded_index()
        else:
            self._index.ingest_quarantine(item)

    def _prior_event(self, row: journal_module.JournalEntry) -> dict[str, Any] | None:
        cached = self._events.get(row.event_id)
        if cached is not None:
            return cached
        found = self._db.execute(
            "SELECT digest, seq, measurement_id FROM events WHERE id = ?",
            (row.event_id,)).fetchone()
        return (None if found is None else
                {"digest": found[0], "seq": found[1], "measurement_id": found[2]})

    def _measurement_ids_for_event(self, event_id: str) -> tuple[str, ...]:
        values = {measurement_id for measurement_id, item in self._measurements.items()
                  if item.get("event_id") == event_id}
        scalar = self._db.execute(
            "SELECT measurement_id FROM events WHERE id = ?", (event_id,)).fetchone()
        if scalar is not None and isinstance(scalar[0], str):
            values.add(scalar[0])
        values.update(row[0] for row in self._db.execute(
            "SELECT measurement_id FROM event_measurements WHERE event_id = ?",
            (event_id,)).fetchall())
        return tuple(sorted(values))

    def _delete_derived_row(self, table: str, key: tuple[Any, ...]) -> None:
        columns = _DB_COLUMNS[table]
        key_columns = columns[:len(key)]
        where = " AND ".join(f"{column} = ?" for column in key_columns)
        prior = self._db.execute(
            f"SELECT {', '.join(columns)} FROM {table} WHERE {where}", key).fetchone()
        if prior is None:
            return
        self._db_xor ^= self._derived_row_token(table, tuple(prior))
        self._db.execute(f"DELETE FROM {table} WHERE {where}", key)
        self.state["derived_digest"] = f"{self._db_xor:064x}"

    def _prior_measurement(self, measurement_id: str,
                           _before_seq: int) -> dict[str, Any] | None:
        cached = self._measurements.get(measurement_id)
        if cached is not None:
            return cached
        found = self._db.execute(
            "SELECT value FROM measurements WHERE id = ?", (measurement_id,)).fetchone()
        if found is None:
            return None
        value = json.loads(found[0])
        self._bounded_put(self._measurements, measurement_id, value)
        return value

    def _quarantine(self, row: journal_module.JournalEntry, reason: str,
                    *, affected_dependencies: tuple[str, ...] = (),
                    replay: bool = False) -> None:
        item = {
            "schema": evidence.QUARANTINE_SCHEMA,
            "event_id": row.event_id, "event_digest": _digest(row.envelope()),
            "reason": reason, "affected_dependencies": list(affected_dependencies),
            "global_scope": not bool(affected_dependencies), "frontier": row.seq}
        self._global_quarantine = self._global_quarantine or item["global_scope"]
        self._cache_quarantine(item)
        if not replay:
            self.state["quarantine_count"] += 1
            self.state["last_quarantine"] = item
        self.state["readiness"] = "unknown"
        self._index.set_projection_available(not self._global_quarantine)

    def _retract_prior(self, measurement_id: str,
                       row: journal_module.JournalEntry, *, replay: bool) -> tuple[str, ...]:
        prior = self._prior_measurement(measurement_id, row.seq) or {}
        for frame_id in prior.get("frame_ids", []):
            frame = self.frames.make_frame(
                frame_type="epyc.vidya/frame/retraction/v1",
                assertion={"retracts": frame_id,
                           "reason": "conflicting native carrier reused measurement_id"},
                provenance={"method": "autokernel-evidence-feed/v1", "about": frame_id},
                actor="autokernel-evidence-feed/v1", authority_scope="measurement",
                created_at=row.written_at)
            self._append_frames([frame], replay=replay)
        finding_id = prior.get("finding_id")
        finding = self._findings.get(finding_id)
        if finding is None and isinstance(finding_id, str):
            found = self._db.execute(
                "SELECT value FROM findings WHERE id = ?", (finding_id,)).fetchone()
            if found is not None:
                finding = evidence.Finding.from_dict(json.loads(found[0]))
        dependencies = (finding.claim_key.dependency_identities
                        if finding is not None else {})
        if dependencies:
            for dependency_id in sorted(dependencies):
                prior_generation = self._db.execute(
                    "SELECT max(CAST(json_extract(value, '$.generation') AS INTEGER)) "
                    "FROM invalidations WHERE json_extract(value, '$.dependency_id') = ?",
                    (dependency_id,)).fetchone()[0] or 0
                invalidation = {
                    "schema": evidence.INVALIDATION_SCHEMA,
                    "event_id": f"{row.event_id}:{_digest(dependency_id)[:16]}",
                    "dependency_id": dependency_id,
                    "generation": prior_generation + 1,
                    "kind": "retraction", "frontier": row.seq}
                self._cache_invalidation(invalidation)
        prior["conflicted"] = True
        self._set_derived_row(
            "measurements", (measurement_id,),
            (measurement_id, _canonical(prior).decode()))
        return tuple(sorted(dependencies))

    def _process_measurement(self, row: journal_module.JournalEntry, *, replay: bool) -> None:
        measurement_id = row.payload.get("measurement_id")
        semantic_digest = _digest(row.payload)
        prior = self._prior_measurement(str(measurement_id), row.seq)
        if prior is not None:
            if prior.get("semantic_digest") != semantic_digest:
                dependencies = self._retract_prior(str(measurement_id), row, replay=replay)
                self._quarantine(
                    row, "conflicting carrier reused native measurement_id",
                    affected_dependencies=dependencies, replay=replay)
                return
        projected = self.adapter.project_journal_event(
            row.envelope(), corpus_root=self.corpus_root)
        if prior is not None:
            self._diagnose(row.event_id, "exact_duplicate", replay=replay)
            return
        if projected is None:
            self._diagnose(row.event_id, "diagnostic_zero_tuple", replay=replay)
            value = {
                "semantic_digest": semantic_digest, "event_id": row.event_id,
                "frame_ids": [], "finding_id": None, "conflicted": False}
            key = str(measurement_id)
            self._set_derived_row(
                "measurements", (key,), (key, _canonical(value).decode()))
            self._bounded_put(self._measurements, str(measurement_id), value)
            return
        grade = self.claim_tuple.grade(projected)
        frames = self.claim_tuple.to_frames(
            projected, as_of=row.written_at, adapter_id=self.adapter.ADAPTER_ID)
        self._append_frames(frames, replay=replay)
        finding = None
        if self.finding_projector is not None:
            finding = self.finding_projector(projected, grade, row, row.seq)
            if finding is not None:
                finding = evidence.Finding.from_dict(finding.to_dict())
                self._cache_finding(finding)
        if finding is None:
            self.state["readiness"] = "unknown"
            self._diagnose(
                row.event_id,
                "no prospective effect-question/comparison receipt; level only",
                replay=replay)
        else:
            self.state["readiness"] = "projected"
        value = {
            "semantic_digest": semantic_digest, "event_id": row.event_id,
            "frame_ids": [frame["frame_id"] for frame in frames],
            "finding_id": finding.finding_id if finding is not None else None,
            "conflicted": False}
        key = str(measurement_id)
        self._set_derived_row(
            "measurements", (key,), (key, _canonical(value).decode()))
        self._bounded_put(self._measurements, str(measurement_id), value)

    @staticmethod
    def _profile_terminal_key(payload: Mapping[str, Any]) -> str | None:
        worker = payload.get("worker_id")
        generation = payload.get("worker_generation")
        request_id = payload.get("request_id")
        stage_id = payload.get("stage_id")
        if (payload.get("event") != "WORKER_RESULT_ACCEPTED"
                or not isinstance(worker, str) or not worker
                or type(generation) is not int or generation < 1
                or not isinstance(request_id, str) or not request_id.startswith("profile-")
                or not isinstance(stage_id, str) or not stage_id.startswith("target-profile-")
                or request_id.removeprefix("profile-")
                   != stage_id.removeprefix("target-profile-")):
            return None
        return f"{worker}:{generation}"

    def _remember_profile_terminal(self, row: journal_module.JournalEntry) -> None:
        if self.profile_adapter is None:
            return
        key = self._profile_terminal_key(row.payload)
        if key is None:
            return
        encoded = _canonical(row.envelope()).decode()
        prior = self._db.execute(
            "SELECT value, seq FROM profile_terminals WHERE id = ?", (key,)).fetchone()
        if prior is not None:
            if prior != (encoded, row.seq):
                raise FeedError("profile terminal identity was reused with different bytes")
            return
        count = self._db.execute("SELECT count(*) FROM profile_terminals").fetchone()[0]
        if count >= self.max_projection_entries:
            raise FeedProjectionPending(
                "unjoined profile terminal capacity exhausted; source cursor remains before event")
        self._set_derived_row(
            "profile_terminals", (key,), (key, encoded, row.seq))

    @staticmethod
    def _profile_terminal_ref(envelope: Mapping[str, Any]) -> str | None:
        payload = envelope.get("payload")
        if not isinstance(payload, Mapping):
            return None
        data = payload.get("data")
        if (not isinstance(data, Mapping) or data.get("accepted") is not True
                or data.get("reason") is not None
                or not isinstance(data.get("result_digest"), str)):
            return None
        fields = ("worker_id", "worker_generation", "request_id", "plan_digest",
                  "lineage_id", "stage_id", "grant_id", "grant_generation", "container_id")
        if any(name not in payload for name in fields):
            return None
        body = {name: payload[name] for name in fields}
        body.update(return_code=0, result_digest=data["result_digest"], accepted=True, reason=None)
        return "lifecycle:" + _digest(body)

    def _release_failed_profile_terminal(
            self, row: journal_module.JournalEntry, *, replay: bool) -> None:
        payload = row.payload
        if self.profile_adapter is None or payload.get("outcome") != "failed":
            return
        references = payload.get("terminal_refs")
        if not isinstance(references, list) or not references:
            return
        released = 0
        for key, encoded in self._db.execute(
                "SELECT id, value FROM profile_terminals").fetchall():
            envelope = json.loads(encoded)
            terminal = envelope.get("payload", {})
            if (terminal.get("lineage_id") != payload.get("transition_id")
                    or envelope.get("campaign_id") != row.campaign_id
                    or terminal.get("campaign_id") != payload.get("campaign_id")
                    or terminal.get("config_digest") != payload.get("config_digest")
                    or terminal.get("config_generation") != payload.get("config_generation")
                    or terminal.get("supervisor_incarnation")
                       != payload.get("supervisor_incarnation")
                    or self._profile_terminal_ref(envelope) not in references):
                continue
            self._delete_derived_row("profile_terminals", (key,))
            released += 1
        if released:
            self._diagnose(
                row.event_id,
                f"released {released} profile terminal after exact failed driver settlement",
                replay=replay)

    def _process_profile(self, row: journal_module.JournalEntry, *, replay: bool) -> None:
        if self.profile_adapter is None:
            self._diagnose(row.event_id, "profile projector is absent from installed ROOT closure",
                           replay=replay)
            return
        verifier = row.payload.get("verifier_ref")
        if not isinstance(verifier, str) or not verifier.startswith("controller-worker:"):
            self._quarantine(
                row, "profile publication lacks its original worker reference", replay=replay)
            return
        parts = verifier.removeprefix("controller-worker:").rsplit(":", 1)
        if len(parts) != 2:
            self._quarantine(
                row, "profile publication worker reference is malformed", replay=replay)
            return
        try:
            generation = int(parts[1])
        except ValueError:
            self._quarantine(
                row, "profile publication worker generation is malformed", replay=replay)
            return
        key = f"{parts[0]}:{generation}"
        found = self._db.execute(
            "SELECT value FROM profile_terminals WHERE id = ?", (key,)).fetchone()
        if found is None:
            self._diagnose(
                row.event_id,
                "diagnostic_zero_tuple: original accepted profile terminal is unavailable",
                replay=replay)
            return
        terminal = json.loads(found[0])
        native = {"terminal": terminal, "profile": row.envelope()}
        self.profile_adapter.joined_journal_pair(native)
        try:
            projected = self.profile_adapter.project_journal_pair(
                native, corpus_root=self.corpus_root)
        except self.claim_tuple.ProjectionError:
            # A durable PROFILE_VERIFIED has already consumed this exact
            # terminal.  Local compact-artifact/schema refusal is a permanent
            # zero-tuple/quarantine disposition, not a future terminal join.
            self._delete_derived_row("profile_terminals", (key,))
            raise
        if (not isinstance(projected, tuple) or len(projected) != 2
                or any(not isinstance(item, self.claim_tuple.ClaimTuple) for item in projected)):
            raise FeedError("profile projector did not return the exact measurement/integrity pair")
        semantic_digest = _digest(native)
        prepared: list[tuple[Any, tuple[str, str, list[str]], list[dict[str, Any]]]] = []
        for item in projected:
            measurement_id = item.measurement_id
            prior = self._prior_measurement(measurement_id, row.seq)
            if prior is not None and prior.get("semantic_digest") != semantic_digest:
                dependencies: set[str] = set()
                prior_event = prior.get("event_id")
                associated = (self._measurement_ids_for_event(prior_event)
                              if isinstance(prior_event, str) else (measurement_id,))
                for prior_measurement_id in associated:
                    dependencies.update(self._retract_prior(
                        prior_measurement_id, row, replay=replay))
                self._delete_derived_row("profile_terminals", (key,))
                self._quarantine(
                    row, "conflicting profile carrier reused native measurement_id",
                    affected_dependencies=tuple(sorted(dependencies)), replay=replay)
                return
            grade = self.claim_tuple.grade(item)
            frames = self.claim_tuple.to_frames(
                item, as_of=row.written_at, adapter_id=self.profile_adapter.ADAPTER_ID)
            prepared.append((item, grade, frames))
        for item, _grade, frames in prepared:
            self._append_frames(frames, replay=replay)
            value = {"semantic_digest": semantic_digest, "event_id": row.event_id,
                     "frame_ids": [frame["frame_id"] for frame in frames],
                     "finding_id": None, "conflicted": False}
            self._set_derived_row(
                "measurements", (item.measurement_id,),
                (item.measurement_id, _canonical(value).decode()))
            self._set_derived_row(
                "event_measurements", (row.event_id, item.measurement_id),
                (row.event_id, item.measurement_id))
            self._bounded_put(self._measurements, item.measurement_id, value)
        self._delete_derived_row("profile_terminals", (key,))
        self.state["readiness"] = "unknown"
        self._diagnose(
            row.event_id,
            "profile observation and receipt integrity only; no production-validation authority",
            replay=replay)

    def _invalidate_source_target(self, row: journal_module.JournalEntry, *, replay: bool) -> None:
        target = row.payload.get("target_event_id")
        matched = self._measurement_ids_for_event(target) if isinstance(target, str) else ()
        for measurement_id in matched:
            self._prior_measurement(measurement_id, row.seq)
            self._retract_prior(measurement_id, row, replay=replay)
        self.state["readiness"] = "unknown"
        self._diagnose(
            row.event_id, "source invalidation projected" if matched else
            "source invalidation target was not a projected measurement", replay=replay)

    def _process(self, row: journal_module.JournalEntry, *, replay: bool = False) -> None:
        digest = _digest(row.envelope())
        prior = self._prior_event(row)
        if prior is not None:
            if prior["digest"] != digest:
                dependencies = set()
                for measurement_id in self._measurement_ids_for_event(row.event_id):
                    dependencies.update(self._retract_prior(
                        measurement_id, row, replay=replay))
                self._quarantine(row, "event_id reused with conflicting bytes",
                                 affected_dependencies=tuple(sorted(dependencies)), replay=replay)
            return
        try:
            if row.kind == journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED:
                self._process_measurement(row, replay=replay)
            elif row.kind == journal_module.KIND_WORKER_LIFECYCLE:
                self._remember_profile_terminal(row)
            elif (row.kind == journal_module.KIND_ACTOR_PREPARATION
                  and row.payload.get("event") == "PROFILE_VERIFIED"):
                self._process_profile(row, replay=replay)
            elif row.kind == journal_module.KIND_UNIFIED_DRIVER_SETTLED:
                self._release_failed_profile_terminal(row, replay=replay)
            elif row.kind in {journal_module.KIND_SUPERSEDED,
                              journal_module.KIND_RETRIEVAL_SUPERSEDED}:
                self._invalidate_source_target(row, replay=replay)
            else:
                self._diagnose(row.event_id, "operational_zero_tuple", replay=replay)
        except (OSError, FeedError):
            # Storage failures are retryable outages.  The source cursor stays
            # put so the Journal remains the sole durable queue.
            self.state["readiness"] = "outage"
            self._index.set_projection_available(False)
            raise
        except Exception as exc:
            # Projection/schema/callback failures are durable unknowns, never winners.
            self._quarantine(
                row, f"projection refused: {type(exc).__name__}: {exc}", replay=replay)
        event_value = {
            "digest": digest, "seq": row.seq,
            "measurement_id": (row.payload.get("measurement_id")
                               if row.kind == journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED
                               else None)}
        self._set_derived_row(
            "events", (row.event_id,),
            (row.event_id, event_value["digest"], row.seq,
             event_value["measurement_id"]))
        self._bounded_put(self._events, row.event_id, event_value)

    def planner_view(self, requested_scope: Mapping[str, Any], claim: evidence.ClaimKey,
                     intended_use: str) -> dict[str, Any]:
        """Return the planner's two-state supported/unknown projection."""
        result = self._index.retrieve(requested_scope, claim, intended_use)
        keys = [("dependency", dep) for dep in claim.dependency_identities]
        keys.append(("signature", evidence._mandatory_signature_digest(claim)))
        evicted = any(self._db.execute(
            "SELECT 1 FROM evicted WHERE kind = ? AND key = ?", key).fetchone()
                      is not None for key in keys)
        return {"status": ("supported" if result.complete_for_intended_use and not evicted
                            else "unknown"),
                "result": result.to_dict(), "source_frontier": self.state["source_frontier"],
                "projection_frontier": self.state["projected_frontier"],
                "evicted_projection": evicted}

    def drain_once(self, limits: DrainLimits = DrainLimits()) -> dict[str, Any]:
        if self._failed:
            raise FeedError("evidence feed is poisoned after a failed projection checkpoint")
        deadline = self.tail.clock() + limits.max_seconds
        if self._recovery_batch is not None:
            batch = self._recovery_batch
            if (len(batch.events) > limits.max_events
                    or batch.bytes_read > limits.max_bytes):
                raise FeedError("recovery batch exceeds requested drain bounds")
            self._recovery_batch = None
        else:
            batch = self.tail.read(limits)
        self.state["source_frontier"] = batch.source_frontier
        if batch.proof_pending:
            result = self.snapshot(bytes_read=batch.bytes_read, events_read=0)
            result["proof_pending"] = True
            return result
        processed = 0
        for row, position in zip(batch.events, batch.positions, strict=True):
            if self.tail.clock() > deadline:
                break
            if row.seq == self.state["projected_frontier"]:
                projected = self._db.execute(
                    "SELECT digest FROM events WHERE seq = ?", (row.seq,)).fetchone()
                if projected != (_digest(row.envelope()),):
                    raise FeedError("durable projection-ahead row disagrees with Journal")
                # A prior same-process attempt may have crossed the projection
                # boundary and then expired before its checkpoint. Persist that
                # exact in-memory frontier before either cursor ACK below.
                self._save_state()
                self.tail.ack(row.seq, position)
                self.state["acknowledged_frontier"] = row.seq
                self.state["ack_position"] = self.tail.checkpoint()
                processed += 1
                continue
            if self.state["readiness"] == "outage":
                self.state["readiness"] = "unknown"
                self._index.set_projection_available(not self._global_quarantine)
            self._process(row)
            self.state["projected_frontier"] = row.seq
            self.state["projection_digest"] = _digest({
                "prior": self.state["projection_digest"],
                "seq": row.seq, "event": _digest(row.envelope())})
            # Projection is durable before the source ACK.
            if self.tail.clock() > deadline:
                raise FeedProjectionPending("drain deadline expired before projection commit")
            self._save_state()
            self.tail.ack(row.seq, position)
            self.state["acknowledged_frontier"] = row.seq
            self.state["ack_position"] = self.tail.checkpoint()
            processed += 1
        cursor = self.journal.cursor(self.reader_id)
        cursor_seq = cursor.last_seq if cursor is not None else 0
        if processed:
            self._save_state()
        return self.snapshot(cursor_seq=cursor_seq, bytes_read=batch.bytes_read,
                             events_read=processed)

    def index(self) -> evidence.EvidenceIndex:
        return self._index

    def snapshot(self, *, cursor_seq: int | None = None, bytes_read: int = 0,
                 events_read: int = 0) -> dict[str, Any]:
        if cursor_seq is None:
            cursor = self.journal.cursor(self.reader_id)
            cursor_seq = cursor.last_seq if cursor is not None else 0
        source = int(self.state["source_frontier"])
        return {"schema": "epyc.autokernel.evidence_feed_snapshot.v1",
                "reader_id": self.reader_id, "readiness": self.state["readiness"],
                "source_frontier": source, "cursor_frontier": cursor_seq,
                "projection_frontier": self.state["projected_frontier"],
                "lag": max(0, source - cursor_seq), "events_read": events_read,
                "bytes_read": bytes_read,
                "quarantine_count": self.state["quarantine_count"],
                "finding_count": len(self._findings),
                "projection_checksum": self.state["checksum"]}


class EvidenceFeedWorker:
    """One owned polling thread with bounded deterministic backoff."""

    def __init__(self, feed: EvidenceFeed, *, limits: DrainLimits = DrainLimits(),
                 poll_seconds: float = 0.1, max_backoff_seconds: float = 2.0,
                 multiplier: float = 2.0) -> None:
        values = (poll_seconds, max_backoff_seconds, multiplier)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) for value in values)
                or poll_seconds <= 0 or max_backoff_seconds < poll_seconds
                or multiplier < 1):
            raise ValueError("invalid worker backoff")
        self.feed, self.limits = feed, limits
        self.poll_seconds, self.max_backoff_seconds = poll_seconds, max_backoff_seconds
        self.multiplier = multiplier
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="autokernel-evidence-feed",
                                        daemon=False)
        self._thread.start()

    def _run(self) -> None:
        delay = self.poll_seconds
        while not self._stop.is_set():
            try:
                snap = self.feed.drain_once(self.limits)
                self.last_error = None
                delay = self.poll_seconds if snap["events_read"] else min(
                    self.max_backoff_seconds, delay * self.multiplier)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                delay = min(self.max_backoff_seconds, delay * self.multiplier)
            self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "corpus-root", "ledger-path", "store-root", "root-repo"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--current-epoch", required=True)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--max-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-seconds", type=float, default=.25)
    args = parser.parse_args(argv)
    feed = EvidenceFeed(source_root=args.source_root, corpus_root=args.corpus_root,
                        ledger_path=args.ledger_path, store_root=args.store_root,
                        root_repo=args.root_repo, current_epoch=args.current_epoch)
    print(json.dumps(feed.drain_once(DrainLimits(args.max_events, args.max_bytes,
                                                args.max_seconds)), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
