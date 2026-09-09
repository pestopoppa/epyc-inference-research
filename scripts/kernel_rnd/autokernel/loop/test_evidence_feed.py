from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from autokernel import journal as journal_module
from autokernel.loop import scoped_evidence as evidence
from autokernel.loop.journal_feed_owner import FeedOwnerError
from autokernel.loop.evidence_feed import (
    BoundedJournalTail,
    DrainLimits,
    EvidenceFeed,
    EvidenceFeedWorker,
    FeedError,
    OwnedTailBatch,
    TailError,
)


def _root_repo() -> Path:
    configured = os.environ.get("EPYC_ROOT_REPO")
    candidates = [Path(configured)] if configured else []
    research = Path(__file__).resolve().parents[4]
    candidates.extend([
        research.parent / research.name.replace("-research-", "-root-"),
        Path("/workspace"),
    ])
    for candidate in candidates:
        if (candidate / "scripts/vidya/adapters/autokernel_unified_arm.py").is_file():
            return candidate.resolve()
    pytest.skip("paired epyc-root checkout is unavailable")


def _root_fixture(root_repo: Path, corpus_root: Path, *, status="measurement") -> tuple[dict, dict]:
    path = root_repo / "tests/vidya/test_autokernel_unified_arm.py"
    spec = importlib.util.spec_from_file_location("root_unified_arm_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    carrier = module.carrier_fixture(corpus_root, status=status)
    stored = module._write(corpus_root, "sealed-carrier.json", carrier)
    return carrier, stored


def _native_event(source_root: Path, corpus_root: Path, root_repo: Path,
                  *, status="measurement"):
    carrier, stored = _root_fixture(root_repo, corpus_root, status=status)
    journal = journal_module.Journal(str(source_root), campaign_id="campaign-1")
    journal.initialize()
    journal.append(
        journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED,
        {"schema": "epyc.autokernel.unified_arm_capture.v1",
         "measurement_id": carrier["measurement_id"],
         "carrier": carrier, "artifact": stored},
        record_id=carrier["measurement_id"],
    )
    return journal, carrier


def _feed(tmp_path: Path, root_repo: Path, **kwargs) -> EvidenceFeed:
    return EvidenceFeed(
        source_root=tmp_path / "source", corpus_root=tmp_path / "corpus",
        ledger_path=tmp_path / "vidya" / "ledger.jsonl",
        store_root=tmp_path / "store", root_repo=root_repo,
        current_epoch="epoch-1", **kwargs)


def _prospective_fixture_finding(dependency: str = "dep:a"):
    """Trusted in-process stand-in for the not-yet-authored producer receipt."""
    digest = "a" * 64
    other = "b" * 64

    def project(_tuple, grade, row, frontier):
        claim = evidence.ClaimKey.from_dict({
            "schema": evidence.CLAIM_KEY_SCHEMA,
            "target_scope": {"target": dependency, "backend": "llama_cpu",
                             "model": "m", "quant": "q8", "workload": "decode",
                             "allocation": "cores0-47"},
            "control_identity": {"recipe": "anchor", "digest": digest},
            "intervention_identity": {"recipe": "candidate", "digest": other},
            "mechanism_identity": {"name": "fixture", "implementation_digest": digest},
            "estimand": "level", "metric": "aggregate_tok_s",
            "metric_direction": "higher",
            "effect_question": {"kind": "absolute_effect_bound", "bound": 2.0,
                                "unit": "percent"},
            "dependency_identities": {dependency: digest},
        })
        return evidence.Finding.from_dict({
            "schema": evidence.FINDING_SCHEMA, "finding_id": f"finding:{row.event_id}",
            "source": {"schema": evidence.SOURCE_REF_SCHEMA,
                       "event_id": row.event_id, "artifact_digest": digest,
                       "locator": f"journal:{row.seq}"},
            "claim_key": claim.to_dict(), "conclusion": "positive", "value": 3.0,
            "tested_scope": dict(claim.target_scope),
            "tested_question": dict(claim.effect_question),
            "raw_grade": {"Q": grade[0], "T": grade[1]}, "epoch": "epoch-1",
            "record_class": "strict_search",
            "dependency_generations": {dependency: 0},
            "intended_use_disposition": {"intended_use": "screen_out",
                                           "disposition": "certificate_candidate"},
            "authority_reference": "prospective-fixture-receipt", "frontier": frontier,
        })
    return project


def test_actual_journal_adapter_ledger_index_path_is_bounded_and_unknown(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    snapshot = feed.drain_once(DrainLimits(max_events=1, max_bytes=1_000_000,
                                           max_seconds=1.0))
    assert snapshot["events_read"] == 1 and snapshot["lag"] == 0
    assert snapshot["readiness"] == "unknown" and snapshot["finding_count"] == 0
    assert journal.cursor(feed.reader_id).last_seq == 1
    records = feed.ledger.read_all()
    assert len(records) == 3
    support = next(row.frame for row in records
                   if row.frame["frame_type"].endswith("evidence_supports_claim/v1"))
    assert support["assertion"]["grade"] == {"Q": "Witnessed", "T": "Attested"}
    assert "no prospective effect-question" in str(feed._diagnostics)
    assert feed.index().findings == ()


def test_steady_measurement_drain_does_not_rescan_ledger(tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    ledger_path = tmp_path / "vidya" / "ledger.jsonl"
    ledger_path.parent.mkdir()
    ledger_path.touch()
    feed = _feed(tmp_path, root_repo)
    # Establish the existing Ledger instance's public single-writer fast path
    # through append itself; package C must not seed its private cache fields.
    feed.ledger.append({"frame_id": "single-writer-owner", "frame_type": "fixture"})
    monkeypatch.setattr(feed.ledger, "read_all", lambda **_kwargs: pytest.fail(
        "steady-state scan"))
    feed.drain_once()


def test_compact_checkpoint_runs_beyond_capacity_and_restarts(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    for number in range(12):
        journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": f"p{number}", "reason": "fixture"})
    feed = _feed(tmp_path, root_repo, max_projection_entries=2)
    for _ in range(4):
        feed.drain_once(DrainLimits(max_events=3, max_bytes=100_000, max_seconds=1))
    assert journal.cursor(feed.reader_id).last_seq == 12
    checkpoint = json.loads(feed.store_path.read_text())
    assert set(checkpoint).isdisjoint(
        {"events", "measurements", "frame_ids", "findings", "invalidations"})
    assert feed._diagnostics.maxlen == 2 and len(feed._diagnostics) == 2
    assert len(feed._events) <= 2 and len(feed._measurements) <= 2
    feed.close()

    resumed = _feed(tmp_path, root_repo, max_projection_entries=2)
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p12", "reason": "fixture"})
    resumed.drain_once()
    assert journal.cursor(resumed.reader_id).last_seq == 13


def test_substantial_prefix_new_event_is_constant_checkpoint_and_empty_is_no_write(
        tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    for number in range(80):
        journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": f"p{number}", "reason": "fixture"})
    feed = _feed(tmp_path, root_repo, max_projection_entries=3)
    feed.drain_once(DrainLimits(max_events=80, max_bytes=1_000_000, max_seconds=2))
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "next", "reason": "fixture"})
    serialized = []
    visited = []
    real_atomic = importlib.import_module(
        "autokernel.loop.evidence_feed")._atomic_json
    real_process = feed._process

    def observe(path, value):
        serialized.append(len(json.dumps(value, sort_keys=True)))
        return real_atomic(path, value)

    monkeypatch.setattr("autokernel.loop.evidence_feed._atomic_json", observe)
    monkeypatch.setattr(feed, "_process", lambda row: (
        visited.append(row.seq), real_process(row))[1])
    feed.drain_once(DrainLimits(max_events=1, max_bytes=100_000, max_seconds=1))
    assert visited == [81] and len(serialized) == 2 and max(serialized) < 2_000
    checkpoint_identity = (feed.store_path.stat().st_ino,
                           feed.store_path.stat().st_mtime_ns)
    serialized.clear()
    empty = feed.drain_once()
    assert empty["events_read"] == 0 and serialized == []
    assert (feed.store_path.stat().st_ino,
            feed.store_path.stat().st_mtime_ns) == checkpoint_identity


def test_incremental_ledger_tail_handles_peer_append_partial_repair_and_rotation(
        tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    source = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    source.initialize()
    ledger_path = tmp_path / "vidya" / "ledger.jsonl"
    ledger_path.parent.mkdir()
    ledger_path.touch()
    feed = _feed(tmp_path, root_repo)
    peer = feed.ledger_module.Ledger(ledger_path)
    peer.append({"frame_id": "peer-1", "frame_type": "fixture"})
    monkeypatch.setattr(feed.ledger, "read_all", lambda **_kwargs: pytest.fail(
        "ordinary peer append caused full scan"))
    feed._refresh_ledger_index()
    assert "peer-1" in feed._ledger_frames
    monkeypatch.undo()

    with open(ledger_path, "ab") as handle:
        handle.write(b'{"seq":')
    feed._refresh_ledger_index()
    peer.append({"frame_id": "peer-2", "frame_type": "fixture"})
    feed._refresh_ledger_index()
    assert {"peer-1", "peer-2"}.issubset(feed._ledger_frames)

    replacement = ledger_path.with_suffix(".rotated")
    replacement.write_bytes(ledger_path.read_bytes())
    os.replace(replacement, ledger_path)
    feed._refresh_ledger_index()
    assert {"peer-1", "peer-2"}.issubset(feed._ledger_frames)


def test_evidence_index_cap_evicts_locally_and_keeps_unrelated_cached_claim(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(
        tmp_path, root_repo, max_projection_entries=2,
        finding_projector=_prospective_fixture_finding(),
        scope_verifier=lambda *_: True, use_verifier=lambda *_: True,
        result_verifier=lambda *_: True,
        support_rule_identity="test:prospective-receipt:v1")
    row = journal.read_all()[0]
    projected = feed.adapter.project_journal_event(
        row.envelope(), corpus_root=feed.corpus_root)
    grade = feed.claim_tuple.grade(projected)
    made = []
    for number, dep in enumerate(("dep:a", "dep:b", "dep:c"), start=1):
        item = _prospective_fixture_finding(dep)(projected, grade, row, number).to_dict()
        item["finding_id"] = f"finding:{number}"
        item["frontier"] = number
        made.append(evidence.Finding.from_dict(item))
        feed._cache_finding(made[-1])
    assert len(feed._findings) == len(feed.index().findings) == 2
    assert feed.planner_view(
        made[0].claim_key.target_scope, made[0].claim_key, "screen_out")["status"] == "unknown"
    assert feed.planner_view(
        made[1].claim_key.target_scope, made[1].claim_key, "screen_out")["status"] == "supported"

def test_semantic_duplicate_and_conflict_use_full_native_payload(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, carrier = _native_event(
        tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    feed.drain_once()
    first = journal.read_all()[0].envelope()
    journal.append(first["kind"], first["payload"], record_id=first["record_id"])
    feed.drain_once()
    identity = feed._measurements[carrier["measurement_id"]]
    assert len(identity["semantic_digest"]) == 64
    assert feed._diagnostics[-1]["reason"] == "exact_duplicate"
    assert len(feed.ledger.read_all()) == 3

    conflict_payload = json.loads(json.dumps(first["payload"]))
    # Copying the producer's inner carrier_digest cannot disguise changed
    # semantic bytes elsewhere in the payload.
    conflict_payload["artifact"]["sha256"] = "0" * 64
    conflict = journal.append(first["kind"], conflict_payload,
                              record_id=first["record_id"])
    feed.drain_once()
    assert identity["conflicted"] is True
    assert feed.state["last_quarantine"]["event_id"] == conflict.event_id
    assert len(feed.ledger.read_all()) == 6  # three supports + three retractions
    assert journal.cursor(feed.reader_id).last_seq == 3


def test_cursor_tail_honours_event_limit_without_read_all(tmp_path, monkeypatch):
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    for number in range(3):
        journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": f"p{number}", "reason": "fixture"})
    tail = BoundedJournalTail(journal, "bounded-reader")
    monkeypatch.setattr(journal, "read_all", lambda: pytest.fail("unbounded read_all"))
    first = tail.read(DrainLimits(max_events=1, max_bytes=4096, max_seconds=1))
    assert [row.seq for row in first.events] == [1] and first.lag == 3
    journal.commit_cursor("bounded-reader", 1)
    tail.ack(1, first.positions[0])
    second = tail.read(DrainLimits(max_events=1, max_bytes=4096, max_seconds=1))
    assert [row.seq for row in second.events] == [2]


def test_cursor_location_recovery_runs_outside_journal_owner_lock(tmp_path, monkeypatch):
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p1", "reason": "fixture"})
    journal.register_reader("bounded-reader")
    journal.commit_cursor("bounded-reader", 1)
    tail = BoundedJournalTail(journal, "bounded-reader")
    real_locate = tail._locate
    observed_unlocked = False
    module = importlib.import_module("autokernel.loop.evidence_feed")

    def observe_location(*args, **kwargs):
        nonlocal observed_unlocked
        contender = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
        with module._journal_snapshot_lock(contender):
            observed_unlocked = True
        return real_locate(*args, **kwargs)

    monkeypatch.setattr(tail, "_locate", observe_location)
    tail.read(DrainLimits(max_events=1, max_bytes=4096, max_seconds=1))
    assert observed_unlocked


def test_visible_line_after_producer_fsync_failure_is_not_durable_evidence(
        tmp_path, monkeypatch):
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    real_fsync = journal_module.os.fsync

    def fail_fsync(_descriptor):
        raise OSError("fixture producer fsync failure")

    monkeypatch.setattr(journal_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="producer fsync failure"):
        journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": "visible-only", "reason": "fixture"})
    monkeypatch.setattr(journal_module.os, "fsync", real_fsync)

    shard = tmp_path / "source" / "events.jsonl"
    assert shard.read_bytes().endswith(b"\n")
    tail = BoundedJournalTail(journal, "fsync-failure-reader")
    visible = tail.read(DrainLimits(max_events=1, max_bytes=4096, max_seconds=1))
    assert [row.seq for row in visible.events] == [1]
    with pytest.raises(FeedOwnerError, match="durable frontier"):
        OwnedTailBatch(visible, durable_frontier=0,
                       owner_token="fixture-owner-generation")
    assert journal.cursor("fsync-failure-reader").last_seq == 0


def test_busy_cursor_writer_refuses_owned_ack_without_waiting_for_writer(
        tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p1", "reason": "fixture"})
    feed = _feed(tmp_path, root_repo)
    checkpointed = threading.Event()
    writer_has_lock = threading.Event()
    real_save = feed._save_state
    save_calls = 0

    def save_then_expose_ack_race():
        nonlocal save_calls
        save_calls += 1
        real_save()
        if save_calls == 1:
            checkpointed.set()
            assert writer_has_lock.wait(1)

    def hold_current_journal_owner():
        assert checkpointed.wait(1)
        contender = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
        with contender.write_lock():
            writer_has_lock.set()
            time.sleep(0.08)

    monkeypatch.setattr(feed, "_save_state", save_then_expose_ack_race)
    holder = threading.Thread(target=hold_current_journal_owner)
    holder.start()
    limit = 0.02
    limits = DrainLimits(max_events=1, max_bytes=4096, max_seconds=limit)
    with pytest.raises((BlockingIOError, FeedOwnerError)):
        feed.drain_once(limits)
    assert journal.cursor(feed.reader_id).last_seq == 0
    assert feed.state["projected_frontier"] == 1
    holder.join(1)

    assert not holder.is_alive()
    snapshot = feed.drain_once(limits)
    assert snapshot["events_read"] == 1
    assert journal.cursor(feed.reader_id).last_seq == 1
    feed.close()


def test_restart_uses_exact_derived_index_without_prefix_scan(tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    for number in range(20):
        journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": f"p{number}", "reason": "fixture"})
    feed = _feed(tmp_path, root_repo, max_projection_entries=2)
    feed.drain_once(DrainLimits(max_events=20, max_bytes=1_000_000, max_seconds=2))
    feed.close()

    resumed = _feed(tmp_path, root_repo, max_projection_entries=2)
    monkeypatch.setattr(resumed.journal, "scan", lambda *_args, **_kwargs:
                        pytest.fail("acknowledged prefix rescanned"))
    monkeypatch.setattr(resumed.journal, "read_all", lambda *_args, **_kwargs:
                        pytest.fail("acknowledged prefix materialized"))
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "new", "reason": "fixture"})
    resumed.drain_once()
    assert journal.cursor(resumed.reader_id).last_seq == 21


def test_restart_after_archived_acknowledged_shards_does_not_replay_prefix(
        tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal = journal_module.Journal(
        str(tmp_path / "source"), campaign_id="c", max_shard_bytes=1)
    journal.initialize()
    for number in range(3):
        journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": f"p{number}", "reason": "fixture"})
    feed = _feed(tmp_path, root_repo)
    feed.drain_once(DrainLimits(max_events=3, max_bytes=100_000, max_seconds=1))
    feed.close()
    assert journal.archive_retired_shards()

    resumed = _feed(tmp_path, root_repo)
    monkeypatch.setattr(resumed.journal, "scan", lambda *_args, **_kwargs:
                        pytest.fail("archived acknowledged prefix rescanned"))
    monkeypatch.setattr(resumed.journal, "read_all", lambda *_args, **_kwargs:
                        pytest.fail("archived acknowledged prefix materialized"))
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p3", "reason": "fixture"})
    resumed.drain_once()
    assert journal.cursor(resumed.reader_id).last_seq == 4


def test_db_ahead_checkpoint_failure_poisons_instance_and_recovers(tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    module = importlib.import_module("autokernel.loop.evidence_feed")
    monkeypatch.setattr(module, "_atomic_json", lambda *_args: (
        _ for _ in ()).throw(OSError("checkpoint failed after DB commit")))
    with pytest.raises(OSError, match="after DB commit"):
        feed.drain_once()
    assert journal.cursor(feed.reader_id).last_seq == 0
    with pytest.raises(FeedError, match="poisoned"):
        feed.drain_once()
    feed.close()
    monkeypatch.undo()

    resumed = _feed(tmp_path, root_repo)
    assert resumed.state["projected_frontier"] == 1
    resumed.drain_once()
    assert journal.cursor(resumed.reader_id).last_seq == 1
    assert len(resumed.ledger.read_all()) == 3
    resumed.close()
    restarted = _feed(tmp_path, root_repo)
    assert restarted.state["acknowledged_frontier"] == 1


def test_valid_sqlite_row_mutation_fails_derived_checksum(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    feed.drain_once()
    identity_path = feed.identity_path
    feed.close()
    with sqlite3.connect(identity_path) as connection:
        connection.execute("UPDATE events SET digest = ?", ("0" * 64,))
    with pytest.raises(FeedError, match="content checksum"):
        _feed(tmp_path, root_repo)


def test_json_ahead_of_derived_projection_is_refused(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    state = dict(feed.state)
    store_path = feed.store_path
    feed.close()
    state["projected_frontier"] = 1
    state["projection_digest"] = "a" * 64
    unsigned = {key: value for key, value in state.items() if key != "checksum"}
    state["checksum"] = importlib.import_module(
        "autokernel.loop.evidence_feed")._digest(unsigned)
    store_path.write_text(json.dumps(state))
    with pytest.raises(FeedError, match="disagrees"):
        _feed(tmp_path, root_repo)


def test_cached_tail_refuses_truncated_source_incarnation(tmp_path):
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p1", "reason": "fixture"})
    tail = BoundedJournalTail(journal, "bounded-reader")
    first = tail.read(DrainLimits(max_events=1, max_bytes=4096, max_seconds=1))
    journal.commit_cursor("bounded-reader", 1)
    tail.ack(1, first.positions[0])
    (tmp_path / "source" / "events.jsonl").write_bytes(b"")
    with pytest.raises(TailError, match="beyond source frontier"):
        tail.read(DrainLimits(max_events=1, max_bytes=4096, max_seconds=1))


def test_operational_event_emits_zero_frames_and_advances_cursor(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p1", "reason": "fixture"})
    feed = _feed(tmp_path, root_repo)
    snapshot = feed.drain_once()
    assert snapshot["events_read"] == 1
    assert feed.ledger.read_all() == []
    assert journal.cursor(feed.reader_id).last_seq == 1
    assert feed._diagnostics[-1]["reason"] == "operational_zero_tuple"


def test_diagnostic_zero_tuple_still_records_semantic_identity(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    _, carrier = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo,
                               status="diagnostic")
    feed = _feed(tmp_path, root_repo)
    feed.drain_once()
    identity = feed._measurements[carrier["measurement_id"]]
    assert len(identity["semantic_digest"]) == 64
    assert identity["frame_ids"] == []
    assert feed.ledger.read_all() == []


def test_unsupported_native_version_is_quarantined_and_unknown(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    source = tmp_path / "source"
    journal = journal_module.Journal(str(source), campaign_id="c")
    journal.initialize()
    envelope = {"journal_schema": journal_module.JOURNAL_ENTRY_SCHEMA,
                "event_id": "unsupported-event", "seq": 1,
                "kind": journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED,
                "campaign_id": "c", "record_id": "0" * 64,
                "written_at": "2026-09-09T00:00:00Z",
                "payload": {"schema": "epyc.autokernel.unified_arm_capture.v999"}}
    with open(source / "events.jsonl", "a") as handle:
        handle.write(json.dumps(envelope, sort_keys=True) + "\n")
    journal.recover_durable_publication()
    feed = _feed(tmp_path, root_repo)
    snapshot = feed.drain_once()
    assert snapshot["readiness"] == "unknown"
    assert snapshot["quarantine_count"] == 1
    assert journal.cursor(feed.reader_id).last_seq == 1


def test_ledger_outage_keeps_native_cursor_unacknowledged(tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    monkeypatch.setattr(feed.ledger, "append", lambda *_: (_ for _ in ()).throw(
        OSError("ledger unavailable")))
    with pytest.raises(OSError, match="ledger unavailable"):
        feed.drain_once()
    assert journal.cursor(feed.reader_id).last_seq == 0


def test_partial_ledger_append_retries_without_duplicate_frames(tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    real_append = feed.ledger.append
    calls = 0

    def fail_second(frame):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("mid-batch ledger outage")
        return real_append(frame)

    monkeypatch.setattr(feed.ledger, "append", fail_second)
    with pytest.raises(OSError, match="mid-batch"):
        feed.drain_once()
    assert journal.cursor(feed.reader_id).last_seq == 0
    monkeypatch.setattr(feed.ledger, "append", real_append)
    feed.drain_once()
    assert len(feed.ledger.read_all()) == 3
    assert journal.cursor(feed.reader_id).last_seq == 1


def test_restart_after_projection_before_cursor_ack_deduplicates_frames(tmp_path,
                                                                       monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    monkeypatch.setattr(feed.tail, "ack", lambda *_args, **_kwargs: (
        _ for _ in ()).throw(OSError("crash before ack")))
    with pytest.raises(OSError, match="crash before ack"):
        feed.drain_once()
    assert len(feed.ledger.read_all()) == 3
    assert journal.cursor(feed.reader_id).last_seq == 0
    feed.close()

    resumed = _feed(tmp_path, root_repo)
    resumed.drain_once()
    assert len(resumed.ledger.read_all()) == 3
    assert journal.cursor(feed.reader_id).last_seq == 1


def test_deadline_after_projection_same_process_retry_persists_before_ack(
        tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    feed._recovery_batch = feed.tail.read(
        DrainLimits(max_events=1, max_bytes=100_000, max_seconds=1))
    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(feed.tail, "clock", lambda: next(ticks))
    with pytest.raises(FeedError, match="deadline expired"):
        feed.drain_once(DrainLimits(max_events=1, max_bytes=100_000, max_seconds=1))
    assert feed.state["projected_frontier"] == 1
    assert journal.cursor(feed.reader_id).last_seq == 0
    assert (not feed.store_path.exists()
            or json.loads(feed.store_path.read_text())["projected_frontier"] == 0)

    monkeypatch.setattr(feed.tail, "clock", lambda: 0.0)
    monkeypatch.setattr(feed.tail, "ack", lambda *_args: (_ for _ in ()).throw(
        OSError("crash at source ACK")))
    with pytest.raises(OSError, match="crash at source ACK"):
        feed.drain_once(DrainLimits(max_events=1, max_bytes=100_000, max_seconds=1))
    assert journal.cursor(feed.reader_id).last_seq == 0
    assert json.loads(feed.store_path.read_text())["projected_frontier"] == 1
    finding_count = len(feed.ledger.read_all())
    feed.close()

    resumed = _feed(tmp_path, root_repo)
    assert resumed.state["projected_frontier"] == 1
    assert resumed.state["acknowledged_frontier"] == 0
    resumed.drain_once()
    assert resumed.state["acknowledged_frontier"] == 1
    assert len(resumed.ledger.read_all()) == finding_count
    resumed.close()


def test_restart_after_cursor_ack_then_next_event_survives_second_restart(
        tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    real_save = feed._save_state
    saves = 0

    def crash_after_cursor_commit():
        nonlocal saves
        saves += 1
        if saves == 2:
            raise OSError("crash after ack")
        return real_save()

    monkeypatch.setattr(feed, "_save_state", crash_after_cursor_commit)
    with pytest.raises(OSError, match="crash after ack"):
        feed.drain_once()
    assert journal.cursor(feed.reader_id).last_seq == 1
    persisted = json.loads(feed.store_path.read_text())
    assert persisted["projected_frontier"] == 1
    assert persisted["acknowledged_frontier"] == 0
    feed.close()

    resumed = _feed(tmp_path, root_repo)
    reconciled = json.loads(resumed.store_path.read_text())
    assert reconciled["acknowledged_frontier"] == 1
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p2", "reason": "fixture"})
    resumed.drain_once()
    assert journal.cursor(resumed.reader_id).last_seq == 2
    resumed.close()

    restarted = _feed(tmp_path, root_repo)
    assert restarted.state["projected_frontier"] == 2
    assert restarted.state["acknowledged_frontier"] == 2


def test_projection_store_failure_prevents_cursor_ack(tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    journal.append(journal_module.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "p1", "reason": "fixture"})
    feed = _feed(tmp_path, root_repo)
    monkeypatch.setattr(feed, "_save_state", lambda: (_ for _ in ()).throw(
        OSError("projection store unavailable")))
    with pytest.raises(OSError, match="projection store unavailable"):
        feed.drain_once()
    assert journal.cursor(feed.reader_id).last_seq == 0


def test_measurement_store_failure_recovers_from_ledger_identity(tmp_path, monkeypatch):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    monkeypatch.setattr(feed, "_save_state", lambda: (_ for _ in ()).throw(
        OSError("projection store unavailable")))
    with pytest.raises(OSError, match="projection store unavailable"):
        feed.drain_once()
    assert len(feed.ledger.read_all()) == 3
    assert journal.cursor(feed.reader_id).last_seq == 0
    feed.close()
    resumed = _feed(tmp_path, root_repo)
    resumed.drain_once()
    assert len(resumed.ledger.read_all()) == 3
    assert journal.cursor(resumed.reader_id).last_seq == 1


def test_source_supersession_invalidates_finding_before_ack(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    journal, _ = _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(
        tmp_path, root_repo, finding_projector=_prospective_fixture_finding(),
        scope_verifier=lambda *_: True, use_verifier=lambda *_: True,
        result_verifier=lambda *_: True,
        support_rule_identity="test:prospective-receipt:v1")
    feed.drain_once()
    claim = feed.index().findings[0].claim_key
    proposal = feed.index().proposal_snapshot(claim, intended_use="screen_out")
    target = journal.read_all()[0].event_id
    journal.append_superseded(target, "fixture source withdrawal")
    feed.drain_once()
    assert journal.cursor(feed.reader_id).last_seq == 2
    assert feed.index().dependency_generations["dep:a"] == 1
    admission = evidence.EvidenceIndex.admit_cached(
        proposal, feed.index().fence_snapshot(), intended_use="screen_out",
        authority_verifier=lambda _: True)
    assert admission.status == "stale"


def test_projection_cache_never_restores_verifier_authority(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(
        tmp_path, root_repo, finding_projector=_prospective_fixture_finding(),
        scope_verifier=lambda *_: True, use_verifier=lambda *_: True,
        result_verifier=lambda *_: True,
        support_rule_identity="test:prospective-receipt:v1")
    feed.drain_once()
    claim = feed.index().findings[0].claim_key
    assert feed.planner_view(claim.target_scope, claim, "screen_out")["status"] == "supported"
    feed.close()
    restored = _feed(tmp_path, root_repo)
    assert restored.planner_view(
        claim.target_scope, claim, "screen_out")["status"] == "unknown"


def test_persisted_acknowledgement_detects_cursor_rollback(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    _native_event(tmp_path / "source", tmp_path / "corpus", root_repo)
    feed = _feed(tmp_path, root_repo)
    feed.drain_once()
    feed.close()
    cursor_path = tmp_path / "source" / "cursors" / f"{feed.reader_id}.json"
    cursor = json.loads(cursor_path.read_text())
    cursor["last_seq"] = 0
    cursor_path.write_text(json.dumps(cursor))
    with pytest.raises(FeedError, match="rolled back"):
        _feed(tmp_path, root_repo)


def test_worker_start_is_idempotent_and_stop_joins_without_thread_leak():
    entered = threading.Event()

    class StubFeed:
        def drain_once(self, _limits):
            entered.set()
            return {"events_read": 0}

    worker = EvidenceFeedWorker(StubFeed(), poll_seconds=.01,
                                max_backoff_seconds=.02)
    worker.start()
    first = worker._thread
    worker.start()
    assert worker._thread is first and entered.wait(1)
    worker.stop()
    assert worker.join(1)
    assert not first.is_alive()


def test_cli_paths_are_explicit_and_relative_paths_refuse(tmp_path):
    with pytest.raises(ValueError, match="explicit absolute"):
        EvidenceFeed(source_root=Path("source"), corpus_root=tmp_path,
                     ledger_path=tmp_path / "ledger", store_root=tmp_path,
                     root_repo=tmp_path, current_epoch="epoch")


@pytest.mark.parametrize("kwargs", [
    {"max_events": 1.5}, {"max_events": True}, {"max_bytes": 1.5},
    {"max_seconds": float("nan")}, {"max_seconds": float("inf")},
])
def test_drain_limits_are_closed_finite_and_integral(kwargs):
    with pytest.raises(ValueError):
        DrainLimits(**kwargs)


def test_consumer_refuses_to_create_missing_source_or_share_projection(tmp_path):
    root_repo = _root_repo()
    (tmp_path / "corpus").mkdir()
    with pytest.raises(journal_module.JournalError):
        _feed(tmp_path, root_repo)

    journal = journal_module.Journal(str(tmp_path / "source"), campaign_id="c")
    journal.initialize()
    first = _feed(tmp_path, root_repo)
    with pytest.raises(FeedError, match="another evidence feed"):
        _feed(tmp_path, root_repo)
    with pytest.raises(FeedError, match="Journal reader"):
        EvidenceFeed(source_root=tmp_path / "source", corpus_root=tmp_path / "corpus",
                     ledger_path=tmp_path / "other-ledger.jsonl",
                     store_root=tmp_path / "other-store", root_repo=root_repo,
                     current_epoch="epoch-1")
    first.close()
