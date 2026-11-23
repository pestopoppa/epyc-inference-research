"""Pooled phase records preserve interruption and integration recovery state."""
import pytest

from .. import journal, schemas
from . import gates, loop, phase_journal, pipeline
from .test_pipeline import _comparison, _Critic, _Planner, _workers


def _reconciliation(tmp_path, row, disposition="not_applied"):
    import hashlib
    import json
    path = tmp_path / f"reconciliation-{row['attempt_id']}.json"
    path.write_text(json.dumps({"schema": "epyc.autokernel.integration_reconciliation.v1",
        "attempt_id": row["attempt_id"], "parent": row["parent"],
        "mutation": row["mutation"], "disposition": disposition}))
    return {"disposition": disposition, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _owner(tmp_path):
    book = journal.Journal(str(tmp_path / "journal"))
    book.initialize()
    return book, phase_journal.PhaseJournal(book)


def test_pooled_gate_and_measurement_are_journaled_and_replayed(tmp_path):
    book, owner = _owner(tmp_path)
    head = "a" * 40
    outcomes = pipeline.run_pool(
        workers=_workers(1), make_planner=lambda _: _Planner(), make_critic=lambda _: _Critic(),
        build_context=dict, make_gate=lambda _: lambda *_: (True, [gates.Verdict("compile", True)]),
        make_measure=lambda _: lambda *_: _comparison(0.0, floor=1.0),
        commit=lambda *_: pytest.fail("no keep"), champion_head=lambda: head,
        reset_to_champion=lambda _: head, record=lambda _: None, iterations=1,
        phase_journal=owner)
    assert outcomes[0].status == "measured_null"
    entries = [item for item in book.read_all() if item.kind == journal.KIND_LOOP_PHASE]
    assert [(item.payload["phase"], item.payload["event"]) for item in entries] == [
        ("iteration", "BEGIN"), ("gate", "BEGIN"), ("gate", "COMPLETE"),
        ("measurement", "BEGIN"), ("measurement", "COMPLETE"), ("iteration", "COMPLETE")]
    assert all(item.payload["parent"] == head for item in entries)
    assert all(item.payload["hypothesis"] == _Planner().hypothesis.to_dict()
               for item in entries if item.payload["phase"] != "iteration")
    assert all(item.payload["hypothesis"] == {"mechanism_id": "unknown"}
               for item in entries if item.payload["phase"] == "iteration")
    assert len({item.payload["draw_id"] for item in entries}) == 1
    assert {item["recovery_state"] for item in phase_journal.project(entries).values()} == {"complete"}
    owner.close()


def test_interrupted_units_invalid_and_integration_requires_verified_reconciliation(tmp_path):
    book, owner = _owner(tmp_path)
    common = {"schema": phase_journal.SCHEMA, "event": "BEGIN", "phase": "measurement",
              "attempt_id": "unit", "draw_id": "draw-1", "lane": "lane0", "parent": "a" * 40,
              "mechanism": "fixture", "hypothesis": {"mechanism_id": "fixture"},
              "hypothesis_digest": schemas.content_hash({"mechanism_id": "fixture"}),
              "mutation": None, "result": None}
    owner._append(common)
    owner.close()
    replay = phase_journal.PhaseJournal(book)
    assert replay.phases["unit"]["recovery_state"] == "invalid_interrupted"
    intent = {**common, "attempt_id": "integration", "phase": "integration"}
    replay._append(intent)
    replay._append({**intent, "event": "MUTATION_INTENT", "mutation": {
        "kind": "source", "repo": "/repo", "branch": "test", "candidate_tree": "b" * 40}})
    replay.close()
    with pytest.raises(phase_journal.PhaseJournalError, match="reconciliation"):
        phase_journal.PhaseJournal(book)
    replay = phase_journal.PhaseJournal(book, allow_reconcile=True)
    assert replay.phases["integration"]["recovery_state"] == "unresolved_integration"
    with pytest.raises(loop.RunAborted, match="reconciliation"):
        with replay.phase("gate", "lane0", "a" * 40, "next", "draw-2", {"mechanism_id": "next"}):
            pass
    with pytest.raises(phase_journal.PhaseJournalError):
        replay.reconcile("integration", lambda _: {"disposition": "applied", "sha256": "bad"})
    def verify_isolated(row):
        proof = _reconciliation(tmp_path, row)
        row["mutation"]["branch"] = "tampered"
        return proof

    replay.reconcile("integration", verify_isolated)
    assert replay.phases["integration"]["mutation"]["branch"] == "test"
    replay.reconcile("integration", lambda row: _reconciliation(tmp_path, row))
    with replay.phase("gate", "lane0", "a" * 40, "next", "draw-2", {"mechanism_id": "next"}):
        pass
    replay.close()
    restarted = phase_journal.PhaseJournal(book)
    assert restarted.phases["integration"]["event"] == "RECONCILED"
    restarted.close()


def test_pre_mutation_refusal_does_not_create_uncertain_integration(tmp_path):
    book, owner = _owner(tmp_path)
    with pytest.raises(ValueError):
        with owner.phase("integration", "lane0", "a" * 40, "fixture", "draw-1", {"mechanism_id": "fixture"}):
            raise ValueError("veto before mutation")
    owner.close()
    recovered = phase_journal.PhaseJournal(book)
    assert not recovered._unresolved()
    recovered.close()


def test_mutation_failure_is_unresolved_and_blocks_subsequent_phases(tmp_path):
    book, owner = _owner(tmp_path)
    with pytest.raises(loop.RunAborted, match="unresolved"):
        with owner.phase("integration", "lane0", "a" * 40, "fixture", "draw-1", {"mechanism_id": "fixture"}):
            owner.mark_integration({"kind": "source", "repo": "/repo", "branch": "test",
                                    "candidate_tree": "b" * 40})
            raise OSError("lost completion")
    assert owner._unresolved()[0]["event"] == "UNKNOWN"
    owner.close()
    replay = phase_journal.PhaseJournal(book, allow_reconcile=True)
    with pytest.raises(loop.RunAborted, match="reconciliation"):
        with replay.phase("measurement", "lane0", "a" * 40, "next", "draw-2", {"mechanism_id": "next"}):
            pass
    replay.close()


def test_owner_lease_excludes_another_loop(tmp_path):
    book, owner = _owner(tmp_path)
    with pytest.raises(phase_journal.PhaseJournalError, match="another loop"):
        phase_journal.PhaseJournal(book)
    owner.close()
    successor = phase_journal.PhaseJournal(book)
    successor.close()


def test_phase_event_rejects_resealed_or_malformed_provenance(tmp_path):
    book, owner = _owner(tmp_path)
    row = {"schema": phase_journal.SCHEMA, "event": "BEGIN", "phase": "gate",
           "attempt_id": "id", "draw_id": "draw", "lane": "lane0", "parent": "a" * 40,
           "mechanism": "fixture", "hypothesis": {"mechanism_id": "fixture"},
           "hypothesis_digest": "f" * 64, "mutation": None, "result": None}
    with pytest.raises(ValueError, match="provenance digest"):
        book.append(journal.KIND_LOOP_PHASE, row)
    owner.close()


def test_runtime_integration_binds_exact_snapshots_and_completion(tmp_path):
    book, owner = _owner(tmp_path)
    mutation = {"kind": "runtime", "anchor_snapshot": "a" * 64,
                "candidate_snapshot": "b" * 64}
    import hashlib
    import json
    selection = tmp_path / "runtime-selection.json"
    selection.write_text(json.dumps({"schema": "epyc.autokernel.direct_runtime_selection.v1",
                                     "current_recipe": {"snapshot_digest": "b" * 64}}))
    reference = {"path": str(selection), "sha256": hashlib.sha256(selection.read_bytes()).hexdigest()}
    with owner.phase("integration", "lane0", "c" * 40, "fixture", "draw-1",
                     {"mechanism_id": "fixture"}):
        owner.mark_integration(mutation)
        owner.bind_runtime_completion(reference)
    owner.close()
    rows = [entry.payload for entry in book.read_all() if entry.kind == journal.KIND_LOOP_PHASE]
    assert [row["event"] for row in rows] == ["BEGIN", "MUTATION_INTENT", "COMPLETE"]
    assert rows[1]["mutation"] == rows[2]["mutation"] == mutation
    replay = phase_journal.PhaseJournal(book)
    replay.close()
    selection.unlink()
    with pytest.raises(phase_journal.PhaseJournalError, match="proof unavailable"):
        phase_journal.PhaseJournal(book)


def test_unresolved_integration_blocks_bundle_republication_until_reconciled(tmp_path):
    from . import accumulate
    bundle = accumulate.Bundle(champion_of_record="anchor", tip="anchor")
    bundle.save(tmp_path)
    book = journal.Journal(str(tmp_path / accumulate.JOURNAL_DIRNAME))
    owner = phase_journal.PhaseJournal(book)
    hypothesis = {"mechanism_id": "fixture"}
    begin = {"schema": phase_journal.SCHEMA, "event": "BEGIN", "phase": "integration",
             "attempt_id": "integration", "draw_id": "draw", "lane": "lane0",
             "parent": "anchor", "mechanism": "fixture", "hypothesis": hypothesis,
             "hypothesis_digest": schemas.content_hash(hypothesis), "mutation": None, "result": None}
    owner._append(begin)
    owner._append({**begin, "event": "MUTATION_INTENT", "mutation": {
        "kind": "source", "repo": "/repo", "branch": "test", "candidate_tree": "b" * 40}})
    owner.close()
    projection = tmp_path / accumulate.Bundle.FILENAME
    before = projection.read_bytes()
    for read_only in (False, True):
        with pytest.raises(accumulate.BundleRecoveryRequired, match="unresolved integration"):
            accumulate.load_bundle(tmp_path, anchor_commit="advanced", is_ancestor=lambda *_: True,
                                   read_only=read_only)
    assert projection.read_bytes() == before
    assert accumulate.would_refuse_anchor(tmp_path, anchor_commit="advanced",
                                          is_ancestor=lambda *_: True)[0]
    assert "unresolved integration" in str(accumulate.peek_refusal(
        tmp_path, anchor_commit="advanced", is_ancestor=lambda *_: True))
    owner = phase_journal.PhaseJournal(book, allow_reconcile=True)
    owner.reconcile("integration", lambda row: _reconciliation(tmp_path, row))
    owner.close()
    proof_path = tmp_path / "reconciliation-integration.json"
    proof_bytes = proof_path.read_bytes()
    proof_path.unlink()
    with pytest.raises(accumulate.BundleRecoveryRequired, match="proof unavailable"):
        accumulate.load_bundle(tmp_path, anchor_commit="anchor", is_ancestor=lambda a, b: a == b)
    proof_path.write_bytes(proof_bytes)
    restored, _ = accumulate.load_bundle(tmp_path, anchor_commit="anchor",
                                         is_ancestor=lambda a, b: a == b)
    assert restored.champion_of_record == "anchor"


def test_completed_rollback_is_replayed_with_its_durable_receipt(tmp_path):
    import hashlib
    book, owner = _owner(tmp_path)
    receipt = tmp_path / "rollback.json"
    import json
    refs = {}
    for field, phase, tip in (("first", "initial", "c" * 40),
                              ("repeat", "repeat", "c" * 40),
                              ("restored_tip_recheck", "restored-prior-tip", "a" * 40)):
        path = tmp_path / f"{phase}.json"
        path.write_text(json.dumps({"schema": "epyc.autokernel.accumulator_comparison.v1",
            "phase": phase, "mechanism_id": "fixture", "prior_tip": "a" * 40,
            "measured_tip": tip, "comparison": {}}))
        refs[field] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt.write_text(json.dumps({
        "schema": "epyc.autokernel.accumulator_comparison_resolution.v1",
        "outcome": "interaction_regression", "mechanism_id": "fixture",
        "prior_tip": "a" * 40, "candidate_tip": "c" * 40, "candidate_tree": "b" * 40,
        **refs}))
    proof = {"path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
    with pytest.raises(loop.InteractionRegression):
        with owner.phase("integration", "lane0", "a" * 40, "fixture", "draw-1",
                         {"mechanism_id": "fixture"}):
            owner.mark_integration({"kind": "source", "repo": "/repo", "branch": "test",
                                    "candidate_tree": "b" * 40})
            owner.mark_rollback(proof)
            raise loop.InteractionRegression("restored")
    owner.close()
    replay = phase_journal.PhaseJournal(book)
    assert not replay._unresolved()
    assert list(replay.phases.values())[0]["event"] == "ROLLED_BACK"
    replay.close()
    (tmp_path / "repeat.json").unlink()
    with pytest.raises(phase_journal.PhaseJournalError, match="proof unavailable"):
        phase_journal.PhaseJournal(book)


def test_restored_regression_state_survives_restart_and_forces_next_bench(tmp_path):
    from . import accumulate
    bundle = accumulate.Bundle(champion_of_record="anchor", tip="anchor",
        measurement_validity=accumulate.MEASUREMENT_STALE_RESTORED_REGRESSION)
    bundle.save(tmp_path)
    restored, _ = accumulate.load_bundle(tmp_path, anchor_commit="anchor",
                                         is_ancestor=lambda a, b: a == b)
    assert restored.measurement_validity == accumulate.MEASUREMENT_STALE_RESTORED_REGRESSION
    assert accumulate.bench_due(restored, accumulate.AccumulatorPolicy(), every_keeps=5) == \
        "stale_measurement"


def test_journal_failure_stops_before_side_effect(tmp_path, monkeypatch):
    book, owner = _owner(tmp_path)
    calls = []

    def fail_append(*_args, **_kwargs):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(book, "append", fail_append)
    with pytest.raises(loop.RunAborted, match="phase journal failed"):
        with owner.phase("gate", "lane0", "a" * 40, "fixture", "draw-1",
                         {"mechanism_id": "fixture"}):
            calls.append("executed")
    assert calls == []
    owner.close()


def test_lost_completion_after_mutation_is_recovered_as_unresolved(tmp_path, monkeypatch):
    book, owner = _owner(tmp_path)
    original = book.append

    def lose_completion(kind, payload, **kwargs):
        if payload["event"] == "COMPLETE":
            raise OSError("injected completion failure")
        return original(kind, payload, **kwargs)

    monkeypatch.setattr(book, "append", lose_completion)
    with pytest.raises(loop.RunAborted, match="phase journal failed"):
        with owner.phase("integration", "lane0", "a" * 40, "fixture", "draw-1",
                         {"mechanism_id": "fixture"}) as completion:
            owner.mark_integration({"kind": "source", "repo": "/repo", "branch": "test",
                                    "candidate_tree": "b" * 40})
            completion["value"] = "c" * 40
    owner.close()
    monkeypatch.setattr(book, "append", original)
    with pytest.raises(phase_journal.PhaseJournalError, match="reconciliation"):
        phase_journal.PhaseJournal(book)


def test_pooled_journal_failure_refuses_the_gate(tmp_path, monkeypatch):
    book, owner = _owner(tmp_path)
    original = book.append
    calls = []

    def fail_gate(kind, payload, **kwargs):
        if payload["phase"] == "gate" and payload["event"] == "BEGIN":
            raise OSError("injected gate intent failure")
        return original(kind, payload, **kwargs)

    monkeypatch.setattr(book, "append", fail_gate)
    with pytest.raises(loop.RunAborted, match="phase journal"):
        pipeline.run_pool(
            workers=_workers(1), make_planner=lambda _: _Planner(), make_critic=lambda _: _Critic(),
            build_context=dict, make_gate=lambda _: lambda *_: calls.append("gate"),
            make_measure=lambda _: lambda *_: pytest.fail("measurement launched"),
            commit=lambda *_: pytest.fail("integration launched"), champion_head=lambda: "a" * 40,
            reset_to_champion=lambda _: "a" * 40, record=lambda _: None, iterations=1,
            phase_journal=owner)
    assert calls == []
    owner.close()


def test_runtime_integration_without_selection_receipt_stays_unresolved(tmp_path):
    book, owner = _owner(tmp_path)
    with pytest.raises(loop.RunAborted, match="not provable"):
        with owner.phase("integration", "lane0", "c" * 40, "fixture", "draw-1",
                         {"mechanism_id": "fixture"}):
            owner.mark_integration({"kind": "runtime", "anchor_snapshot": "a" * 64,
                                    "candidate_snapshot": "b" * 64})
    assert owner._unresolved()[0]["event"] == "UNKNOWN"
    owner.close()
    with pytest.raises(phase_journal.PhaseJournalError, match="reconciliation"):
        phase_journal.PhaseJournal(book)
