"""Journal-authoritative Bundle persistence and recovery regressions."""
from contextlib import contextmanager
import json
from unittest import mock

import pytest

from autokernel import journal
from autokernel.loop import accumulate as A


def _linear(*commits):
    positions = {commit: index for index, commit in enumerate(commits)}
    return lambda older, newer: (
        older in positions and newer in positions
        and positions[older] <= positions[newer]
    )


def _bundle(*, pct=7.5, cadence=2):
    return A.Bundle(champion_of_record="cor0", tip="k2",
                    keeps=["m1", "m2"], compounded_bench_pct=pct,
                    keeps_since_serving_gate=cadence)


def _book(store):
    return journal.Journal(str(store / A.JOURNAL_DIRNAME))


def _saved_events(store):
    return [entry for entry in _book(store).read_all()
            if entry.kind == journal.KIND_LOOP_BUNDLE_SAVED]


def _tree_bytes(root):
    return {str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None)
            for path in sorted(root.rglob("*"))}


def test_native_bundle_payload_is_narrow_and_digest_bound():
    snapshot = _bundle().to_dict()
    payload = A._saved_payload(snapshot, provenance="current_snapshot")
    assert journal._validate_native_payload(
        journal.KIND_LOOP_BUNDLE_SAVED, payload) == []
    bad = dict(payload, snapshot_sha256="0" * 64)
    assert any("content hash" in reason for reason in
               journal._validate_native_payload(journal.KIND_LOOP_BUNDLE_SAVED, bad))
    bad = dict(payload, claim_eligible=True)
    assert any("unknown field" in reason for reason in
               journal._validate_native_payload(journal.KIND_LOOP_BUNDLE_SAVED, bad))


def test_current_snapshots_require_v2_and_explicit_validity():
    snapshot = _bundle().to_dict()
    assert snapshot["schema"] == A.BUNDLE_SCHEMA_V2
    missing = dict(snapshot)
    missing.pop("measurement_validity")
    violations = journal.validate_loop_bundle_saved_payload(
        A._saved_payload(missing, provenance="current_snapshot"))
    assert any("missing required" in reason for reason in violations)
    v1 = dict(snapshot, schema=A.BUNDLE_SCHEMA_V1)
    v1.pop("measurement_validity")
    violations = journal.validate_loop_bundle_saved_payload(
        A._saved_payload(v1, provenance="current_snapshot"))
    assert any("only as imported legacy" in reason for reason in violations)
    with pytest.raises(ValueError, match="unknown bundle schema"):
        A.Bundle.from_dict(dict(snapshot, schema="epyc.autokernel.accumulator_bundle.v3"))


def test_v2_rejects_null_cadence():
    snapshot = _bundle().to_dict()
    snapshot["keeps_since_serving_gate"] = None
    with pytest.raises(ValueError, match="non-negative integer"):
        A.Bundle.from_dict(snapshot)
    violations = journal.validate_loop_bundle_saved_payload(
        A._saved_payload(snapshot, provenance="current_snapshot"))
    assert any("required non-negative integer in v2" in reason
               for reason in violations)


def test_v2_forces_old_v1_reader_to_refuse_rollback():
    snapshot = _bundle().to_dict()

    def old_reader(body):
        if body.get("schema") != A.BUNDLE_SCHEMA_V1:
            raise ValueError("unknown bundle schema")

    with pytest.raises(ValueError, match="unknown bundle schema"):
        old_reader(snapshot)


def test_save_journals_before_projection_and_recovers_prepublication_fault(tmp_path):
    bundle = _bundle()
    with mock.patch.object(A.status, "write_json", side_effect=OSError("projection fault")):
        with pytest.raises(OSError, match="projection fault"):
            bundle.save(tmp_path)
    assert len(_saved_events(tmp_path)) == 1
    assert not (tmp_path / A.Bundle.FILENAME).exists()
    restored, _ = A.load_bundle(
        tmp_path, anchor_commit="k2", is_ancestor=_linear("cor0", "k2"))
    assert restored.to_dict() == bundle.to_dict()


def test_journal_append_failure_never_publishes_projection(tmp_path):
    bundle = _bundle()
    with mock.patch.object(journal.Journal, "append",
                           side_effect=OSError("journal fsync failed")), \
             mock.patch.object(A.status, "write_json") as publish:
        with pytest.raises(OSError, match="journal fsync failed"):
            bundle.save(tmp_path)
    publish.assert_not_called()
    assert not (tmp_path / A.Bundle.FILENAME).exists()


def test_postpublication_fault_leaves_replayable_journal_and_projection(tmp_path):
    bundle = _bundle()
    real_write = A.status.write_json

    def publish_then_fail(*args, **kwargs):
        real_write(*args, **kwargs)
        raise OSError("directory fsync uncertainty")

    with mock.patch.object(A.status, "write_json", side_effect=publish_then_fail):
        with pytest.raises(OSError, match="directory fsync uncertainty"):
            bundle.save(tmp_path)
    assert len(_saved_events(tmp_path)) == 1
    assert json.loads((tmp_path / A.Bundle.FILENAME).read_text()) == bundle.to_dict()
    restored, _ = A.load_bundle(
        tmp_path, anchor_commit="k2", is_ancestor=_linear("cor0", "k2"))
    assert restored.to_dict() == bundle.to_dict()


@pytest.mark.parametrize("projection", [None, "{truncated"])
def test_deleted_or_truncated_json_replays_last_snapshot(tmp_path, projection):
    bundle = _bundle()
    bundle.save(tmp_path)
    path = tmp_path / A.Bundle.FILENAME
    if projection is None:
        path.unlink()
    else:
        path.write_text(projection, encoding="utf-8")
    restored, _ = A.load_bundle(
        tmp_path, anchor_commit="k2", is_ancestor=_linear("cor0", "k2"))
    assert restored.to_dict() == bundle.to_dict()
    assert json.loads(path.read_text()) == bundle.to_dict()


def test_repeated_identical_save_is_one_journal_event(tmp_path):
    bundle = _bundle()
    bundle.save(tmp_path)
    bundle.save(tmp_path)
    assert len(_saved_events(tmp_path)) == 1


def test_valid_legacy_json_is_imported_with_original_snapshot(tmp_path):
    legacy = {
        "schema": A.Bundle.LEGACY_SCHEMA,
        "champion_of_record": "cor0",
        "tip": "k2",
        "keeps": ["m1", "m2"],
        "compounded_bench_pct": 5.19,
        "keeps_since_serving_gate": 2,
    }
    (tmp_path / A.Bundle.FILENAME).write_text(json.dumps(legacy), encoding="utf-8")
    restored, note = A.load_bundle(
        tmp_path, anchor_commit="k2", is_ancestor=_linear("cor0", "k2"))
    event = _saved_events(tmp_path)[0]
    assert event.payload["provenance"] == "imported_legacy_state"
    assert event.payload["snapshot"] == legacy
    assert restored.champion_of_record == "cor0"
    assert restored.keeps == ["m1", "m2"]
    assert restored.measurement_validity == A.MEASUREMENT_UNKNOWN_LEGACY
    assert "imported_legacy_state" in note

    # Projection repair and a later save cannot duplicate the original import
    # or relabel its unknown measurement as newly current.
    restored_again, _ = A.load_bundle(
        tmp_path, anchor_commit="k2", is_ancestor=_linear("cor0", "k2"))
    restored_again.save(tmp_path)
    assert len(_saved_events(tmp_path)) == 1
    assert _saved_events(tmp_path)[0].payload["provenance"] == "imported_legacy_state"


def test_invalid_legacy_ancestry_never_appends_bundle_snapshot(tmp_path):
    legacy = {
        "schema": A.Bundle.LEGACY_SCHEMA,
        "champion_of_record": "gone",
        "tip": "gone",
        "keeps": ["m1"],
        "compounded_bench_pct": 4.0,
        "keeps_since_serving_gate": 1,
    }
    (tmp_path / A.Bundle.FILENAME).write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(A.BundleRecoveryRequired, match="invalid COR/tip ancestry"):
        A.load_bundle(tmp_path, anchor_commit="anchor",
                      is_ancestor=_linear("anchor"))
    assert _saved_events(tmp_path) == []


def test_corrupted_journal_refuses_instead_of_using_valid_json(tmp_path):
    bundle = _bundle()
    bundle.save(tmp_path)
    events_path = tmp_path / A.JOURNAL_DIRNAME / journal.BASE_SHARD_NAME
    envelope = json.loads(events_path.read_text().splitlines()[0])
    envelope["payload"]["snapshot"]["compounded_bench_pct"] = 99.0
    events_path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(A.BundleRecoveryRequired, match="snapshot_sha256"):
        A.load_bundle(tmp_path, anchor_commit="k2",
                      is_ancestor=_linear("cor0", "k2"))


def test_journal_read_oserror_is_typed_recovery_refusal(tmp_path):
    _bundle().save(tmp_path)
    with mock.patch.object(journal.Journal, "read_all",
                           side_effect=OSError("media unavailable")):
        with pytest.raises(A.BundleRecoveryRequired, match="journal read failure"):
            A.load_bundle(tmp_path, anchor_commit="k2",
                          is_ancestor=_linear("cor0", "k2"))


@pytest.mark.parametrize("existing_directory", [False, True])
def test_new_empty_store_initializes_durable_baseline(tmp_path, existing_directory):
    store = tmp_path / "new-store"
    if existing_directory:
        store.mkdir()
    restored, note = A.load_bundle(
        store, anchor_commit="anchor", is_ancestor=_linear("anchor"))
    assert restored.champion_of_record == restored.tip == "anchor"
    assert restored.keeps == [] and restored.keeps_since_serving_gate == 0
    assert restored.compounded_bench_pct == 0.0
    assert "new empty baseline" in note
    assert len(_saved_events(store)) == 1
    before = _tree_bytes(store)
    again, _ = A.load_bundle(
        store, anchor_commit="anchor", is_ancestor=_linear("anchor"))
    assert again.to_dict() == restored.to_dict()
    assert _tree_bytes(store) == before


@pytest.mark.parametrize("existing_directory", [False, True])
def test_read_only_missing_state_does_not_create_baseline(tmp_path, existing_directory):
    store = tmp_path / "new-store"
    if existing_directory:
        store.mkdir()
    with pytest.raises(A.BundleRecoveryRequired, match="no accumulator state"):
        A.load_bundle(store, anchor_commit="anchor", is_ancestor=_linear("anchor"),
                      read_only=True)
    assert store.exists() is existing_directory
    assert _tree_bytes(store) == {}


@pytest.mark.parametrize("historical_name", ["experiments.db", "anchor-gen-001", "status.json"])
def test_populated_store_without_bundle_cannot_be_reinitialized(tmp_path, historical_name):
    (tmp_path / historical_name).write_text("original", encoding="utf-8")
    before = _tree_bytes(tmp_path)
    with pytest.raises(A.BundleRecoveryRequired, match="no accumulator state"):
        A.load_bundle(tmp_path, anchor_commit="anchor", is_ancestor=_linear("anchor"))
    assert _tree_bytes(tmp_path) == before


def test_unreadable_legacy_only_state_still_refuses(tmp_path):
    (tmp_path / A.Bundle.FILENAME).write_text("{bad", encoding="utf-8")
    with pytest.raises(A.BundleRecoveryRequired, match="unreadable or invalid"):
        A.load_bundle(tmp_path, anchor_commit="anchor", is_ancestor=lambda a, b: True)


def test_new_store_rejects_symlink_without_writing_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    store = tmp_path / "store"
    store.symlink_to(target, target_is_directory=True)
    with pytest.raises(A.BundleRecoveryRequired, match="symlink"):
        A.load_bundle(store, anchor_commit="anchor", is_ancestor=_linear("anchor"))
    assert list(target.iterdir()) == []


def test_new_store_does_not_overwrite_racing_original_snapshot(tmp_path, monkeypatch):
    original_initialize = journal.Journal.initialize
    def initialize_with_original(book):
        original_initialize(book)
        if not book.read_all():
            with book.write_lock():
                A._append_snapshot_locked(book, _bundle().to_dict(),
                                          provenance="current_snapshot")
    monkeypatch.setattr(journal.Journal, "initialize", initialize_with_original)
    with pytest.raises(A.BundleRecoveryRequired, match="changed during initialization"):
        A.load_bundle(tmp_path, anchor_commit="anchor", is_ancestor=_linear("anchor"))
    assert _saved_events(tmp_path)[0].payload["snapshot"] == _bundle().to_dict()


@pytest.mark.parametrize(
    ("bundle", "anchor", "reason"),
    [
        (A.Bundle("cor0", "sibling"), "anchor", "invalid COR/tip ancestry"),
        (A.Bundle("cor0", "k2"), "k1", "invalid tip/anchor ancestry"),
    ],
)
def test_invalid_cor_or_tip_ancestry_refuses(tmp_path, bundle, anchor, reason):
    bundle.save(tmp_path)
    with pytest.raises(A.BundleRecoveryRequired, match=reason):
        A.load_bundle(tmp_path, anchor_commit=anchor,
                      is_ancestor=_linear("cor0", "k1", "k2", "anchor"))


def test_recovery_transition_holds_journal_lock_through_projection(tmp_path):
    _bundle().save(tmp_path)
    original_lock = journal.Journal.write_lock
    lock_depth = [0]

    @contextmanager
    def tracked_lock(book):
        with original_lock(book):
            lock_depth[0] += 1
            try:
                yield
            finally:
                lock_depth[0] -= 1

    def ancestor(_older, _newer):
        assert lock_depth[0] > 0
        return True

    real_write = A.status.write_json

    def projection(*args, **kwargs):
        assert lock_depth[0] > 0
        return real_write(*args, **kwargs)

    with mock.patch.object(journal.Journal, "write_lock", tracked_lock), \
             mock.patch.object(A.status, "write_json", side_effect=projection):
        A.load_bundle(tmp_path, anchor_commit="k3", is_ancestor=ancestor)


def test_advanced_tip_clears_threshold_inheritance_but_retains_four_keep_cadence(tmp_path):
    bundle = A.Bundle(champion_of_record="cor0", tip="k3",
                      keeps=["m1", "m2", "m3"],
                      compounded_bench_pct=50.0,
                      keeps_since_serving_gate=3)
    bundle.save(tmp_path)
    restored, _ = A.load_bundle(
        tmp_path, anchor_commit="k4", is_ancestor=_linear("cor0", "k3", "k4", "k5"))
    policy = A.AccumulatorPolicy()
    assert restored.champion_of_record == "cor0"
    assert restored.keeps == ["m1", "m2", "m3"]
    assert restored.compounded_bench_pct == 50.0
    assert restored.measurement_validity == A.MEASUREMENT_STALE_TIP_ADVANCE
    assert A.gate_trigger(restored, 3.0, policy) is None
    restored.add_keep("m4", "k5", 1.0)
    assert restored.measurement_validity == A.MEASUREMENT_CURRENT
    assert A.gate_trigger(restored, 3.0, policy) == "cadence"


def test_read_only_legacy_replay_does_not_import_or_publish(tmp_path):
    legacy = {
        "schema": A.Bundle.LEGACY_SCHEMA,
        "champion_of_record": "cor0",
        "tip": "k2",
        "keeps": ["m1"],
        "compounded_bench_pct": 3.0,
        "keeps_since_serving_gate": 1,
    }
    projection = tmp_path / A.Bundle.FILENAME
    original = json.dumps(legacy)
    projection.write_text(original, encoding="utf-8")
    restored, _ = A.load_bundle(
        tmp_path, anchor_commit="k2", is_ancestor=_linear("cor0", "k2"),
        read_only=True)
    assert restored.measurement_validity == A.MEASUREMENT_UNKNOWN_LEGACY
    assert not (tmp_path / A.JOURNAL_DIRNAME).exists()
    assert projection.read_text() == original


def test_read_only_existing_journal_requires_preexisting_lock(tmp_path):
    _bundle().save(tmp_path)
    lock = tmp_path / A.JOURNAL_DIRNAME / journal.LOCK_NAME
    lock.unlink()
    before = _tree_bytes(tmp_path)
    with pytest.raises(A.BundleRecoveryRequired, match="requires existing journal lock"):
        A.load_bundle(tmp_path, anchor_commit="k2",
                      is_ancestor=_linear("cor0", "k2"), read_only=True)
    assert not lock.exists()
    assert _tree_bytes(tmp_path) == before


def test_read_only_journal_replay_leaves_entire_tree_unchanged(tmp_path):
    bundle = _bundle()
    bundle.save(tmp_path)
    projection = tmp_path / A.Bundle.FILENAME
    projection.write_text("{truncated", encoding="utf-8")
    before = _tree_bytes(tmp_path)
    restored, _ = A.load_bundle(
        tmp_path, anchor_commit="k2", is_ancestor=_linear("cor0", "k2"),
        read_only=True)
    assert restored.to_dict() == bundle.to_dict()
    assert _tree_bytes(tmp_path) == before
