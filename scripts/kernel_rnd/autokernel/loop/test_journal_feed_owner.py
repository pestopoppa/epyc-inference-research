from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from autokernel import journal as journal_module
from autokernel.loop.journal_feed_owner import (
    DrainLimits,
    FeedOwnerError,
    FeedOwnerPending,
    JournalFeedOwner,
)


def _journal(tmp_path: Path, *, shard_bytes: int = 64 * 1024 * 1024):
    journal = journal_module.Journal(
        str(tmp_path / "source"), campaign_id="c", max_shard_bytes=shard_bytes)
    journal.initialize()
    return journal


def _append(journal, name: str):
    return journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                          {"proposal_ref": name, "reason": "fixture"})


def test_actual_owner_reads_acks_and_restarts_without_prefix_scan(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    for number in range(4):
        _append(journal, f"p{number}")
    owner = JournalFeedOwner(journal, "feed")
    first = owner.read_owned("feed", DrainLimits(max_events=2),
                             deadline=time.monotonic() + 1)
    owner.ack_owned("feed", 2, first.tail.positions[1],
                    owner_token=first.owner_token, deadline=time.monotonic() + 1)
    checkpoint = owner.checkpoint()
    owner.close()

    restarted = JournalFeedOwner(journal_module.Journal(journal.root), "feed")
    monkeypatch.setattr(restarted.journal, "scan", lambda *_a, **_k:
                        pytest.fail("ordinary restart scanned prefix"))
    second = restarted.read(DrainLimits(max_events=2))
    assert [row.seq for row in second.events] == [3, 4]
    assert checkpoint and checkpoint["offset"] > 0


def test_ordinary_tail_validates_only_new_seals_once(tmp_path, monkeypatch):
    journal = _journal(tmp_path, shard_bytes=1)
    for number in range(24):
        _append(journal, f"p{number}")
    owner = JournalFeedOwner(journal, "feed")
    batch = owner.read_owned(
        "feed", DrainLimits(max_events=24), deadline=time.monotonic() + 1)
    owner.ack_owned("feed", 24, batch.tail.positions[-1],
                    owner_token=batch.owner_token, deadline=time.monotonic() + 1)
    _append(journal, "new")
    seal_opens = 0
    real_read_metadata = journal_module._read_feed_metadata

    def observe_open(path, label):
        nonlocal seal_opens
        if ".feed-shards" in os.fspath(path):
            seal_opens += 1
        return real_read_metadata(path, label)

    monkeypatch.setattr(journal_module, "_read_feed_metadata", observe_open)
    tail = owner.read_owned("feed", DrainLimits(max_events=1),
                            deadline=time.monotonic() + 1)
    owner.ack_owned("feed", 25, tail.tail.positions[0],
                    owner_token=tail.owner_token, deadline=time.monotonic() + 1)
    assert seal_opens == 1


def test_failed_producer_fsync_leaves_visible_but_unowned_bytes(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    real_fsync = journal_module.os.fsync
    monkeypatch.setattr(journal_module.os, "fsync", lambda _fd: (
        _ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        _append(journal, "visible-only")
    monkeypatch.setattr(journal_module.os, "fsync", real_fsync)
    assert (Path(journal.root) / "events.jsonl").read_bytes().endswith(b"\n")
    with pytest.raises(FeedOwnerError, match="publication"):
        JournalFeedOwner(journal, "feed")


def test_rotation_repair_and_explicit_old_writer_recovery(tmp_path):
    journal = _journal(tmp_path, shard_bytes=1)
    _append(journal, "first")
    journal.rotate()
    with open(Path(journal.root) / "events_1.jsonl", "ab") as handle:
        handle.write(b'{"torn":')
    _append(journal, "after-repair")
    owner = JournalFeedOwner(journal, "feed")
    assert [row.seq for row in owner.read(DrainLimits(max_events=8)).events] == [1, 2, 3]
    owner.close()

    os.unlink(Path(journal.root) / journal_module.FEED_PUBLICATION_NAME)
    _append(journal, "old-unaware-writer")
    with pytest.raises(FeedOwnerError, match="publication"):
        JournalFeedOwner(journal, "old-feed")
    recovered = journal.recover_durable_publication()
    assert recovered.durable_frontier == 4
    assert JournalFeedOwner(journal, "recovered-feed").read(DrainLimits()).events


def test_same_size_out_of_band_mutation_invalidates_publication(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "owned")
    path = Path(journal.root) / "events.jsonl"
    raw = bytearray(path.read_bytes())
    raw[-2] = ord(" ") if raw[-2] != ord(" ") else ord("x")
    path.write_bytes(raw)
    with pytest.raises(journal_module.CursorError, match="outside the owner"):
        journal.durable_publication()


def test_token_binds_exact_position_and_survives_unrelated_append(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    batch = owner.read_owned("feed", DrainLimits(), deadline=time.monotonic() + 1)
    position = batch.tail.positions[0]
    with pytest.raises(FeedOwnerError, match="not returned"):
        owner.ack_owned("feed", 1, (position[0], position[1] - 1, position[2]),
                        owner_token=batch.owner_token, deadline=time.monotonic() + 1)
    _append(journal, "unrelated-later")
    owner.ack_owned("feed", 1, position, owner_token=batch.owner_token,
                    deadline=time.monotonic() + 1)
    assert journal.cursor("feed").last_seq == 1


def test_owned_ack_accepts_append_then_rotation_after_issue(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    batch = owner.read_owned("feed", DrainLimits(max_events=1),
                             deadline=time.monotonic() + 1)
    _append(journal, "two")
    journal.rotate()
    owner.ack_owned("feed", 1, batch.tail.positions[0],
                    owner_token=batch.owner_token, deadline=time.monotonic() + 1)
    assert journal.cursor("feed").last_seq == 1


def test_next_owned_read_accepts_acknowledged_shard_growth_then_rotation(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    first = owner.read_owned("feed", DrainLimits(max_events=1),
                             deadline=time.monotonic() + 1)
    owner.ack_owned("feed", 1, first.tail.positions[0],
                    owner_token=first.owner_token, deadline=time.monotonic() + 1)
    _append(journal, "two")
    journal.rotate()
    _append(journal, "three")
    second = owner.read_owned("feed", DrainLimits(max_events=8),
                              deadline=time.monotonic() + 1)
    assert [row.seq for row in second.tail.events] == [2, 3]


@pytest.mark.parametrize("mutation", ["same_size", "replacement"])
def test_retired_cursor_shard_mutation_is_not_authorized_as_growth(
        tmp_path, mutation):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    first = owner.read_owned("feed", DrainLimits(max_events=1),
                             deadline=time.monotonic() + 1)
    owner.ack_owned("feed", 1, first.tail.positions[0],
                    owner_token=first.owner_token, deadline=time.monotonic() + 1)
    _append(journal, "two")
    journal.rotate()
    path = Path(journal.root) / "events.jsonl"
    if mutation == "same_size":
        raw = bytearray(path.read_bytes())
        raw[-2] = ord(" ") if raw[-2] != ord(" ") else ord("x")
        path.write_bytes(raw)
    else:
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
    with pytest.raises(FeedOwnerError, match="seal|identity"):
        owner.read_owned("feed", DrainLimits(), deadline=time.monotonic() + 1)


def test_bounded_read_never_resolves_shards_before_durable_cursor(tmp_path,
                                                                  monkeypatch):
    journal = _journal(tmp_path, shard_bytes=1)
    for index in range(7):
        _append(journal, f"row-{index}")
    owner = JournalFeedOwner(journal, "feed")
    first = owner.read_owned("feed", DrainLimits(max_events=5),
                             deadline=time.monotonic() + 1)
    owner.ack_owned("feed", 5, first.tail.positions[-1],
                    owner_token=first.owner_token, deadline=time.monotonic() + 1)
    seen = []
    real_ref = journal._feed_shard_ref

    def checked_ref(index):
        seen.append(index)
        assert index >= first.tail.positions[-1][0]
        return real_ref(index)

    monkeypatch.setattr(journal, "_feed_shard_ref", checked_ref)
    second = owner.read_owned("feed", DrainLimits(max_events=1),
                              deadline=time.monotonic() + 1)
    assert [row.seq for row in second.tail.events] == [6]
    assert min(seen) == first.tail.positions[-1][0]


def test_one_event_read_does_not_eagerly_materialize_later_shards(tmp_path,
                                                                  monkeypatch):
    journal = _journal(tmp_path, shard_bytes=1)
    for index in range(8):
        _append(journal, f"row-{index}")
    owner = JournalFeedOwner(journal, "feed")

    class GuardedRange:
        def __iter__(self):
            yield 0
            raise AssertionError("read consumed a shard beyond its event bound")

    def guarded_range(start, stop):
        assert (start, stop) == (0, 8)
        return GuardedRange()

    monkeypatch.setattr(journal_module, "range", guarded_range, raising=False)
    batch = owner.read_owned("feed", DrainLimits(max_events=1),
                             deadline=time.monotonic() + 1)
    assert [row.seq for row in batch.tail.events] == [1]


def test_seal_proof_progress_is_bounded_durable_and_resumes_after_restart(tmp_path):
    journal = _journal(tmp_path, shard_bytes=1)
    for index in range(7):
        _append(journal, f"row-{index}")
    owner = JournalFeedOwner(journal, "feed")
    limits = DrainLimits(max_events=1, max_shards=2)
    first = owner.read_owned("feed", limits, deadline=time.monotonic() + 1)
    assert first.tail.proof_pending and first.tail.events == ()
    _, checkpoint = journal.durable_cursor_position("feed")
    assert checkpoint["feed_position"] == [0, 0, 0]
    assert journal._feed_cursor("feed")[1]["feed_proof_next"] == 2
    owner.close()

    restarted = JournalFeedOwner(journal_module.Journal(journal.root), "feed")
    second = restarted.read_owned("feed", limits, deadline=time.monotonic() + 1)
    assert second.tail.proof_pending and second.tail.events == ()
    assert journal._feed_cursor("feed")[1]["feed_proof_next"] == 4
    third = restarted.read_owned("feed", limits, deadline=time.monotonic() + 1)
    assert third.tail.proof_pending and third.tail.events == ()
    assert journal._feed_cursor("feed")[1]["feed_proof_next"] == 6
    fourth = restarted.read_owned("feed", limits, deadline=time.monotonic() + 1)
    assert [row.seq for row in fourth.tail.events] == [1]


def test_many_empty_rotations_make_bounded_durable_scan_progress(tmp_path):
    journal = _journal(tmp_path)
    for _ in range(9):
        journal.rotate()
    _append(journal, "after-empty-shards")
    owner = JournalFeedOwner(journal, "feed")
    limits = DrainLimits(max_events=1, max_shards=2)
    pending = 0
    while True:
        batch = owner.read_owned("feed", limits, deadline=time.monotonic() + 1)
        if batch.tail.events:
            break
        assert batch.tail.proof_pending
        pending += 1
        assert pending < 12
        owner.close()
        owner = JournalFeedOwner(journal_module.Journal(journal.root), "feed")
    assert [row.seq for row in batch.tail.events] == [1]
    assert pending >= 4


def test_proof_progress_survives_archive_reseal_without_prefix_replay(tmp_path,
                                                                      monkeypatch):
    journal = _journal(tmp_path, shard_bytes=1)
    for index in range(3):
        _append(journal, f"row-{index}")
    owner = JournalFeedOwner(journal, "feed")
    batch = owner.read_owned("feed", DrainLimits(max_events=3),
                             deadline=time.monotonic() + 1)
    owner.ack_owned("feed", 3, batch.tail.positions[-1],
                    owner_token=batch.owner_token, deadline=time.monotonic() + 1)
    owner.close()
    assert journal.archive_retired_shards()
    _append(journal, "row-3")

    resumed = JournalFeedOwner(journal_module.Journal(journal.root), "feed")
    monkeypatch.setattr(resumed.journal, "scan", lambda *_args, **_kwargs:
                        pytest.fail("archived prefix was rescanned"))
    tail = resumed.read_owned("feed", DrainLimits(max_events=1),
                              deadline=time.monotonic() + 1)
    assert [row.seq for row in tail.tail.events] == [4]


def test_mutation_after_proof_before_retired_shard_open_is_refused(tmp_path,
                                                                   monkeypatch):
    journal = _journal(tmp_path, shard_bytes=1)
    for index in range(3):
        _append(journal, f"row-{index}")
    owner = JournalFeedOwner(journal, "feed")
    real_advance = journal._advance_durable_proof

    def mutate_after_proof(*args, **kwargs):
        result = real_advance(*args, **kwargs)
        path = Path(journal.root) / "events_1.jsonl"
        raw = bytearray(path.read_bytes())
        raw[-2] = ord(" ") if raw[-2] != ord(" ") else ord("x")
        path.write_bytes(raw)
        return result

    monkeypatch.setattr(journal, "_advance_durable_proof", mutate_after_proof)
    with pytest.raises(FeedOwnerError, match="proved identity"):
        owner.read_owned("feed", DrainLimits(max_events=3),
                         deadline=time.monotonic() + 1)


@pytest.mark.parametrize("kind", ["publication", "cursor", "seal"])
def test_feed_metadata_is_bounded_regular_and_nonblocking(tmp_path, kind):
    journal = _journal(tmp_path, shard_bytes=1)
    _append(journal, "one")
    if kind == "publication":
        path = Path(journal.root) / journal_module.FEED_PUBLICATION_NAME
    elif kind == "cursor":
        journal.register_reader("feed")
        path = Path(journal.root) / journal_module.CURSOR_DIRNAME / "feed.json"
    else:
        _append(journal, "two")
        path = Path(journal.root) / journal_module.FEED_SHARD_DIRNAME / "shard_0.json"
    path.unlink()
    os.mkfifo(path)
    if kind == "publication":
        with pytest.raises(journal_module.CursorError, match="bounded regular"):
            journal.durable_publication()
    elif kind == "cursor":
        with pytest.raises(journal_module.CursorError, match="bounded regular"):
            journal._feed_cursor("feed")
    else:
        owner = JournalFeedOwner(journal, "feed")
        with pytest.raises(FeedOwnerError, match="bounded regular"):
            owner.read_owned("feed", DrainLimits(), deadline=time.monotonic() + 1)


def test_oversized_seal_is_refused_with_one_shard_budget(tmp_path):
    journal = _journal(tmp_path, shard_bytes=1)
    _append(journal, "one")
    _append(journal, "two")
    seal = Path(journal.root) / journal_module.FEED_SHARD_DIRNAME / "shard_0.json"
    seal.write_bytes(b"x" * (journal_module.MAX_FEED_METADATA_BYTES + 1))
    owner = JournalFeedOwner(journal, "feed")
    with pytest.raises(FeedOwnerError, match="bounded regular"):
        owner.read_owned("feed", DrainLimits(max_shards=1),
                         deadline=time.monotonic() + 1)


@pytest.mark.parametrize("kind", ["publication", "cursor"])
def test_oversized_publication_and_cursor_are_refused(tmp_path, kind):
    journal = _journal(tmp_path)
    _append(journal, "one")
    if kind == "publication":
        path = Path(journal.root) / journal_module.FEED_PUBLICATION_NAME
    else:
        journal.register_reader("feed")
        path = Path(journal.root) / journal_module.CURSOR_DIRNAME / "feed.json"
    path.write_bytes(b"x" * (journal_module.MAX_FEED_METADATA_BYTES + 1))
    with pytest.raises(journal_module.CursorError, match="bounded regular"):
        if kind == "publication":
            journal.durable_publication()
        else:
            journal._feed_cursor("feed")


def test_ack_rotation_proof_is_typed_pending_and_resumable(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    limits = DrainLimits(max_events=1, max_shards=1)
    batch = owner.read_owned("feed", limits, deadline=time.monotonic() + 1)
    for _ in range(3):
        journal.rotate()
    for expected_next in (1, 2):
        with pytest.raises(FeedOwnerPending):
            owner.ack_owned("feed", 1, batch.tail.positions[0],
                            owner_token=batch.owner_token,
                            deadline=time.monotonic() + 1)
        assert journal._feed_cursor("feed")[1]["feed_proof_next"] == expected_next
    owner.ack_owned("feed", 1, batch.tail.positions[0],
                    owner_token=batch.owner_token, deadline=time.monotonic() + 1)
    assert journal.cursor("feed").last_seq == 1


def test_same_active_append_during_read_is_a_safe_monotonic_extension(tmp_path,
                                                                     monkeypatch):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    real_ref = journal._feed_shard_ref
    appended = False

    def append_once(index):
        nonlocal appended
        ref = real_ref(index)
        if not appended:
            appended = True
            _append(journal, "two")
        return ref

    monkeypatch.setattr(journal, "_feed_shard_ref", append_once)
    batch = owner.read_owned("feed", DrainLimits(max_events=1),
                             deadline=time.monotonic() + 1)
    assert [row.seq for row in batch.tail.events] == [1]


def test_v1_publication_requires_explicit_recovery_into_v2(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "one")
    path = Path(journal.root) / journal_module.FEED_PUBLICATION_NAME
    body = json.loads(path.read_text())
    body["schema"] = "epyc.autokernel.feed_publication.v1"
    body["checksum"] = journal_module.schemas.content_hash({
        key: value for key, value in body.items() if key != "checksum"})
    path.write_text(json.dumps(body))
    with pytest.raises(journal_module.CursorError, match="schema"):
        journal.durable_publication()
    recovered = journal.recover_durable_publication()
    assert recovered.durable_frontier == 1
    assert json.loads(path.read_text())["schema"] == journal_module.FEED_PUBLICATION_SCHEMA


def test_token_refuses_recovery_era_change_and_unowned_later_shard(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    batch = owner.read_owned("feed", DrainLimits(), deadline=time.monotonic() + 1)
    journal.recover_durable_publication()
    with pytest.raises(FeedOwnerError, match="era changed"):
        owner.ack_owned("feed", 1, batch.tail.positions[0],
                        owner_token=batch.owner_token, deadline=time.monotonic() + 1)

    with open(Path(journal.root) / "events_1.jsonl", "ab") as handle:
        handle.write(b"unowned\n")
    with pytest.raises(journal_module.CursorError, match="unowned bytes"):
        journal.durable_publication()


def test_lease_replacement_and_second_owner_fail_closed(tmp_path):
    journal = _journal(tmp_path)
    owner = JournalFeedOwner(journal, "feed")
    with pytest.raises(FeedOwnerError, match="another process"):
        JournalFeedOwner(journal, "feed")
    lease = owner._lease_path
    replacement = lease.with_suffix(".replacement")
    replacement.write_text("replacement")
    os.replace(replacement, lease)
    with pytest.raises(FeedOwnerError, match="replaced"):
        owner.read(DrainLimits())


@pytest.mark.parametrize("mutation", ["hardlink", "public_mode"])
def test_initial_lease_must_be_private_singly_linked_owned_file(tmp_path, mutation):
    journal = _journal(tmp_path)
    digest = hashlib.sha256(b"feed").hexdigest()[:24]
    lease = Path(journal.root) / f".feed-reader-{digest}.lock"
    lease.write_text("lease")
    if mutation == "hardlink":
        os.link(lease, tmp_path / "second-link")
    else:
        lease.chmod(0o666)
    with pytest.raises(FeedOwnerError, match="private, singly-linked, and owned"):
        JournalFeedOwner(journal, "feed")


def test_outstanding_batch_refuses_retry_under_different_bounds(tmp_path):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    owner.read_owned("feed", DrainLimits(max_events=2), deadline=time.monotonic() + 1)
    with pytest.raises(FeedOwnerError, match="different bounds"):
        owner.read_owned("feed", DrainLimits(max_events=1),
                         deadline=time.monotonic() + 1)


def test_close_serializes_with_inflight_owned_operation(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    owner = JournalFeedOwner(journal, "feed")
    entered = threading.Event()
    release = threading.Event()
    real_read = journal.read_durable_batch

    def paused_read(*args, **kwargs):
        entered.set()
        assert release.wait(1)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(journal, "read_durable_batch", paused_read)
    reader = threading.Thread(target=lambda: owner.read(DrainLimits()))
    reader.start()
    assert entered.wait(1)
    closer = threading.Thread(target=owner.close)
    closer.start()
    time.sleep(0.02)
    assert closer.is_alive()
    release.set()
    reader.join(1)
    closer.join(1)
    assert not reader.is_alive() and not closer.is_alive()


def test_write_lock_is_thread_owned_and_distinct_process_excluded(tmp_path):
    journal = _journal(tmp_path)
    attempted = threading.Event()
    refused = []

    def contend():
        try:
            with journal.write_lock(blocking=False):
                pass
        except BlockingIOError:
            refused.append(True)
        attempted.set()

    with journal.write_lock():
        thread = threading.Thread(target=contend)
        thread.start()
        assert attempted.wait(1)
        code = (
            "from autokernel import journal as j\n"
            f"x=j.Journal({journal.root!r})\n"
            "try:\n"
            "  with x.write_lock(blocking=False): raise SystemExit(3)\n"
            "except BlockingIOError: raise SystemExit(0)\n")
        result = subprocess.run(
            [sys.executable, "-c", code], check=False,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
    thread.join(1)
    assert refused == [True] and result.returncode == 0


def test_cooperative_late_ack_is_committed_and_surfaced(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    _append(journal, "one")
    owner = JournalFeedOwner(journal, "feed")
    batch = owner.read_owned("feed", DrainLimits(), deadline=time.monotonic() + 1)
    real_ack = journal.ack_durable_cursor

    def slow_ack(*args, **kwargs):
        time.sleep(0.02)
        return real_ack(*args, **kwargs)

    monkeypatch.setattr(journal, "ack_durable_cursor", slow_ack)
    owner.ack_owned("feed", 1, batch.tail.positions[0],
                    owner_token=batch.owner_token, deadline=time.monotonic() + 0.001)
    assert journal.cursor("feed").last_seq == 1
    assert owner.last_overrun and owner.last_overrun["operation"] == "ack"
