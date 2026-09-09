"""Adversarial pure tests for candidate and validation state contracts."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from . import candidate_manifest as cm
from . import resolved_recipe as rr


def _h(char: str) -> str:
    return char * 64


def _artifact(role: str, path: str, char: str) -> dict:
    return {"schema": rr.ARTIFACT_SCHEMA, "role": role, "path": path, "sha256": _h(char)}


def _source(*, commit="1" * 40, tree="2" * 40, object_format="sha1") -> dict:
    return {"schema": cm.SOURCE_SCHEMA, "repo_id": "research", "path": "/repo",
            "object_format": object_format, "commit": commit, "tree": tree}


def _target(*, recipe="7", revision="6", backend="gpu", workload="8", build_digest="") -> dict:
    return {"schema": cm.TARGET_SCHEMA, "target_id": f"target-{revision}",
            "target_revision_digest": _h(revision),
            "backend": backend, "build_execution_digest": build_digest,
            "resolved_recipe_execution_digest": _h(recipe),
            "resolved_recipe_snapshot_digest": _h("9"), "model_digest": _h("a"),
            "drafter_digest": None, "workload_digest": _h(workload),
            "production_required": True}


def _change(field: str, previous: str, current: str) -> dict:
    return {"schema": cm.CHANGE_SCHEMA, "field_id": field,
            "previous_digest": _h(previous), "current_digest": _h(current)}


def _keep(keep_id: str, parent: str, *, request=None, kind="runtime", previous="7",
          current="b", field="runtime:recipe", dependencies=()) -> dict:
    return {"schema": cm.KEEP_SCHEMA, "request_id": request or f"request-{keep_id}",
            "keep_id": keep_id, "parent_manifest_digest": parent, "kind": kind,
            "changes": [_change(field, previous, current)], "affected_scopes": ["gpu"],
            "dependencies": list(dependencies)}


def _manifest(*, parent="0", recipe="7", sources=None, keeps=(), manifest_id="candidate"):
    sources = sources or [_source()]
    parsed_sources = tuple(cm.SourceIdentity.from_dict(item) for item in sources)
    source_digest = cm.source_set_digest(parsed_sources)
    build = {"schema": cm.BUILD_SCHEMA, "build_id": "default",
             "build_recipe_digest": _h("3"),
             "source_set_digest": source_digest,
             "executable": _artifact("executable", "/build/bin/llama-server", "4"),
             "dsos": [_artifact("dso", "/build/bin/libggml.so", "5")]}
    return {"schema": cm.CANDIDATE_SCHEMA, "manifest_id": manifest_id,
            "parent_manifest_digest": _h(parent), "production_ref_digest": _h("f"),
            "sources": sources, "builds": [build],
            "targets": [_target(recipe=recipe,
                                 build_digest=cm.BuildIdentity.from_dict(build).execution_digest)],
            "keeps": list(keeps), "dependency_digest": _h("e")}


def _base() -> cm.CandidateManifest:
    return cm.CandidateManifest.from_dict(_manifest())


def _next(previous: cm.CandidateManifest, index: int, *, field="runtime:recipe",
          dependencies=()) -> cm.CandidateManifest:
    digit = "abcdef"[index]
    keep = _keep(f"k{index}", previous.manifest_digest, previous=(
        previous.targets[0].resolved_recipe_execution_digest[0]), current=digit,
        field=field, dependencies=dependencies)
    raw = _manifest(parent=previous.manifest_digest[0], recipe=digit,
                    keeps=[item.to_dict() for item in previous.keeps] + [keep],
                    manifest_id=f"candidate-{index}")
    raw["parent_manifest_digest"] = previous.manifest_digest
    raw["targets"][0]["backend"] = previous.targets[0].backend
    return cm.CandidateManifest.from_dict(raw)


def _row(row_id: str, candidate: cm.CandidateManifest, comparator: cm.CandidateManifest,
         *, backend="gpu", required=True, row_kind="production") -> cm.ValidationRow:
    target = candidate.targets[0]
    control = comparator.targets[0]
    return cm.ValidationRow.from_dict({
        "schema": cm.ROW_SCHEMA, "row_id": row_id, "row_kind": row_kind,
        "required": required, "target_revision_digest": target.target_revision_digest,
        "control_target_revision_digest": control.target_revision_digest,
        "backend": backend, "candidate_build_digest": target.build_execution_digest,
        "control_build_digest": control.build_execution_digest,
        "model_digest": target.model_digest, "drafter_digest": target.drafter_digest,
        "category": "OPTIMUM" if required else "CANDIDATE",
        "control_recipe_digest": control.resolved_recipe_execution_digest,
        "candidate_recipe_digest": target.resolved_recipe_execution_digest,
        "instrument_digest": _h("c"), "protocol_id": "P-TEST",
        "objective_digest": _h("d"), "workload_digest": target.workload_digest,
        "exact_candidate_required": True})


def _row_set(rows) -> cm.RequiredRowSet:
    direct = cm.RequiredRowSet("rows-v1", tuple(rows))
    return cm.RequiredRowSet.from_dict(direct.to_dict())


def _receipt(batch, row_id) -> cm.RowReceipt:
    return cm.RowReceipt.from_dict({
        "schema": cm.RECEIPT_SCHEMA, "batch_id": batch.batch_id,
        "candidate_manifest_digest": batch.candidate_manifest_digest,
        "comparator_manifest_digest": batch.comparator_manifest_digest,
        "row_set_digest": batch.row_set_digest, "row_id": row_id,
        "native_evidence_ref": f"evidence/{row_id}", "native_evidence_digest": _h("1"),
        "intended_use": "validate", "use_disposition": "permitted"})


def _loo_kwargs(candidate, row_set):
    plans, results = {}, {}
    for keep in candidate.keeps:
        plan = cm.plan_loo(candidate, keep.keep_id)
        plans[keep.keep_id] = plan
        results[keep.keep_id] = cm.LOOResult.from_dict({
            "schema": cm.LOO_RESULT_SCHEMA, "plan_digest": plan.plan_digest,
            "candidate_manifest_digest": candidate.manifest_digest,
            "keep_id": keep.keep_id, "derived_manifest_digest": plan.derived_manifest_digest,
            "row_set_digest": row_set.row_set_digest, "receipt_digests": [_h("1")],
            "disposition": "neutral", "evidence_digest": _h("2"),
            "deletion_authorized": False})
    return {"loo_plans": plans, "loo_results": results,
            "loo_verifier": lambda *_: True}


def _batch(candidate, comparator, rows, *, predecessor=None, states=None, batch_id="batch-1"):
    row_set = _row_set(rows)
    states = states or [cm.ValidationRowState(row.row_id, "pending", None, None) for row in rows]
    batch = cm.ValidationBatch(batch_id, candidate.manifest_digest, comparator.manifest_digest,
                               row_set.row_set_digest, predecessor, candidate.manifest_digest,
                               0, 0, tuple(item.keep_id for item in candidate.keeps),
                               tuple(row.row_id for row in rows if row.required),
                               tuple(states))
    return batch.validated(), row_set


def _state(base, *, cadence=0, validated=None, summaries=()):
    return cm.CandidateState.from_dict({
        "schema": cm.STATE_SCHEMA, "production_ref_digest": base.production_ref_digest,
        "integration_tip": base.manifest_digest, "validated_candidate": validated,
        "keeps_since_gate": cadence, "validation_debt": [], "active_batches": [],
        "threshold_generation": 0, "covered_threshold_generation": 0,
        "completed_batches": [], "integrated_requests": {},
        "outstanding_summary_refs": list(summaries), "stale_summary_refs": []})


def test_real_sha1_sources_round_trip_and_mixed_formats_refuse():
    manifest = _base()
    assert len(manifest.sources[0].commit) == 40
    assert cm.CandidateManifest.from_dict(manifest.to_dict()) == manifest
    raw = _manifest()
    raw["sources"][0]["commit"] = "1" * 64
    with pytest.raises(cm.CandidateError, match="does not match sha1"):
        cm.CandidateManifest.from_dict(raw)
    source256 = _source(commit="1" * 64, tree="2" * 64, object_format="sha256")
    assert cm.SourceIdentity.from_dict(source256).object_format == "sha256"
    source256["tree"] = "2" * 40
    with pytest.raises(cm.CandidateError, match="does not match sha256"):
        cm.SourceIdentity.from_dict(source256)


def test_each_target_selects_an_exact_build_from_multi_build_set():
    raw = _manifest()
    second = dict(raw["builds"][0])
    second["build_id"] = "cpu"
    second["executable"] = _artifact("executable", "/cpu/llama-server", "b")
    parsed = cm.BuildIdentity.from_dict(second)
    raw["builds"].append(second)
    raw["targets"].append(_target(recipe="c", revision="d", backend="cpu",
                                          build_digest=parsed.execution_digest))
    manifest = cm.CandidateManifest.from_dict(raw)
    assert len(manifest.builds) == 2
    raw["targets"][1]["build_execution_digest"] = _h("0")
    with pytest.raises(cm.CandidateError, match="unknown build"):
        cm.CandidateManifest.from_dict(raw)


def test_manifest_is_deeply_immutable_and_direct_objects_are_revalidated():
    manifest = _base()
    with pytest.raises(FrozenInstanceError):
        manifest.manifest_id = "x"
    with pytest.raises(FrozenInstanceError):
        manifest.targets[0].backend = "cpu"
    malformed = replace(manifest, sources=())
    with pytest.raises(cm.CandidateError, match="source repositories"):
        malformed.validated()


def test_unknown_schema_digest_tamper_duplicate_and_dependency_order_refuse():
    raw = _manifest()
    raw["unknown"] = 1
    with pytest.raises(cm.CandidateError, match="unknown"):
        cm.CandidateManifest.from_dict(raw)
    base = _base()
    duplicate = _keep("same", base.manifest_digest)
    raw = _manifest(parent=base.manifest_digest[0], recipe="b", keeps=[duplicate, duplicate])
    raw["parent_manifest_digest"] = base.manifest_digest
    with pytest.raises(cm.CandidateError, match="duplicate keep"):
        cm.CandidateManifest.from_dict(raw)
    forward = _keep("k1", base.manifest_digest, dependencies=("later",))
    raw["keeps"] = [forward]
    with pytest.raises(cm.CandidateError, match="forward/cyclic"):
        cm.CandidateManifest.from_dict(raw)


def test_source_change_cannot_claim_a_build_from_different_source_identity():
    base = _base()
    sources = [_source(commit="3" * 40, tree="4" * 40)]
    raw = _manifest(parent=base.manifest_digest[0], sources=sources)
    raw["parent_manifest_digest"] = base.manifest_digest
    raw["keeps"] = [_keep("source", base.manifest_digest, kind="source",
                           previous="1", current="9", field="source:set")]
    with pytest.raises(cm.CandidateError, match="source/build identity"):
        cm.CandidateManifest.from_dict(raw)


def test_runtime_loo_reuses_binary_while_overlap_dependency_and_source_are_not_faked():
    base = _base()
    first = _next(base, 0)
    plan = cm.plan_loo(first, "k0")
    assert plan.status == "planned"
    assert plan.binary_execution_digest == first.build_set_digest
    overlapping = _next(first, 1)
    assert cm.plan_loo(overlapping, "k0").status == "nonidentifiable"
    raw = first.to_dict()
    raw["manifest_id"] = "dependent"
    raw["parent_manifest_digest"] = first.manifest_digest
    raw["keeps"].append(_keep("dependent", first.manifest_digest, previous="c",
                              current="a", field="runtime:other", dependencies=("k0",)))
    dependent = cm.CandidateManifest.from_dict(raw)
    assert cm.plan_loo(dependent, "k0").status == "nonidentifiable"

    source_digest = base.builds[0].source_set_digest
    keep = _keep("source", base.manifest_digest, kind="source", current=source_digest[0],
                 field="source:set")
    keep["changes"][0]["current_digest"] = source_digest
    raw = _manifest(parent=base.manifest_digest[0], keeps=[keep])
    raw["parent_manifest_digest"] = base.manifest_digest
    source_candidate = cm.CandidateManifest.from_dict(raw)
    assert cm.plan_loo(source_candidate, "source").status == "unsupported"


def test_neutral_loo_never_grants_deletion_authority():
    row = {"schema": cm.LOO_RESULT_SCHEMA, "plan_digest": _h("1"),
           "candidate_manifest_digest": _h("3"), "keep_id": "keep",
           "derived_manifest_digest": _h("4"), "row_set_digest": _h("5"),
           "receipt_digests": [_h("6")],
           "disposition": "neutral", "evidence_digest": _h("2"),
           "deletion_authorized": False}
    assert not cm.LOOResult.from_dict(row).deletion_authorized
    row["deletion_authorized"] = True
    with pytest.raises(cm.CandidateError, match="cannot grant"):
        cm.LOOResult.from_dict(row)


def test_correctness_equivalence_requires_trusted_exactly_bound_receipt():
    receipt = cm.EquivalenceReceipt.from_dict({
        "schema": cm.EQUIVALENCE_SCHEMA, "source_digest": _h("1"),
        "destination_digest": _h("2"), "workload_digest": _h("3"),
        "dependency_digest": _h("4"), "intended_use": "correctness",
        "native_evidence_digest": _h("5")})
    assert not cm.equivalence_permits(actual_digest=_h("1"), required_digest=_h("2"),
                                      use="correctness", receipt=receipt)
    assert cm.equivalence_permits(actual_digest=_h("1"), required_digest=_h("2"),
                                  use="correctness", workload_digest=_h("3"),
                                  dependency_digest=_h("4"), receipt=receipt,
                                  verifier=lambda _: True)
    assert not cm.equivalence_permits(actual_digest=_h("1"), required_digest=_h("2"),
                                      use="timing", receipt=receipt)
    assert not cm.equivalence_permits(actual_digest=_h("1"), required_digest=_h("2"),
                                      use="exact", receipt=receipt)


def test_four_keeps_survive_serialization_and_duplicate_request_is_idempotent():
    base, state = _base(), _state(_base())
    previous = base
    for index in range(4):
        candidate = _next(previous, index)
        state, due = cm.integrate_candidate(
            state, previous, candidate, request_id=f"integrate-{index}")
        previous = candidate
        state = cm.CandidateState.from_dict(state.to_dict())
    assert due and state.gate_due and state.keeps_since_gate == 4
    candidate = _next(previous, 4)
    duplicate, duplicate_due = cm.integrate_candidate(
        state, previous, candidate, request_id="new")
    again, _ = cm.integrate_candidate(duplicate, previous, candidate, request_id="new")
    assert again == duplicate and duplicate_due


def test_threshold_due_is_persisted_until_an_attempted_gate_completes():
    base, candidate = _base(), _next(_base(), 0)
    state, due = cm.integrate_candidate(_state(base), base, candidate,
                                        request_id="threshold", threshold_signal=True)
    assert due and cm.CandidateState.from_dict(state.to_dict()).threshold_generation == 1
    row = _row("gpu", candidate, base)
    rows = _row_set([row])
    missing, _ = _batch(candidate, base, [row], states=[
        cm.ValidationRowState("gpu", "prerequisite_missing", "no grant", None)])
    missing = replace(missing, accounted_keeps_since_gate=state.keeps_since_gate,
                      accounted_threshold_generation=state.threshold_generation).validated()
    state = cm.start_batch(state, missing, rows, candidate, base)
    state = cm.complete_batch(state, missing)
    assert state.gate_due and state.keeps_since_gate == 1


def test_new_threshold_generation_survives_completion_of_older_batch():
    base, first = _base(), _next(_base(), 0)
    state, _ = cm.integrate_candidate(_state(base), base, first, request_id="first",
                                      threshold_signal=True)
    row = _row("gpu", first, base)
    rows = _row_set([row])
    batch, _ = _batch(first, base, [row])
    batch = replace(batch, accounted_keeps_since_gate=1,
                    accounted_threshold_generation=1).validated()
    state = cm.start_batch(state, batch, rows, first, base)
    second = _next(first, 1)
    state, _ = cm.integrate_candidate(state, first, second, request_id="second",
                                      threshold_signal=True)
    receipt = _receipt(batch, "gpu")
    completed = replace(batch, rows=(cm.ValidationRowState("gpu", "passed", None, receipt),))
    state = cm.complete_batch(cm.CandidateState.from_dict(state.to_dict()), completed)
    assert state.covered_threshold_generation == 1
    assert state.threshold_generation == 2 and state.gate_due


def test_start_batch_retries_active_and_completed_before_current_tip_checks():
    comparator, candidate = _base(), _next(_base(), 0)
    row = _row("gpu", candidate, comparator)
    rows = _row_set([row])
    batch, _ = _batch(candidate, comparator, [row])
    state = replace(_state(comparator), integration_tip=candidate.manifest_digest)
    active = cm.start_batch(state, batch, rows, candidate, comparator)
    moved = replace(active, integration_tip=_h("0")).validated()
    assert cm.start_batch(moved, batch, rows, candidate, comparator) == moved
    changed = replace(batch, accounted_keeps_since_gate=1).validated()
    with pytest.raises(cm.TransitionError, match="different payload"):
        cm.start_batch(moved, changed, rows, candidate, comparator)
    terminal = replace(batch, rows=(cm.ValidationRowState("gpu", "failed", "failed", None),))
    completed = cm.complete_batch(active, terminal)
    assert cm.start_batch(completed, terminal, rows, candidate, comparator) == completed


def test_integration_request_conflict_parent_cas_and_production_change_refuse():
    base, candidate, state = _base(), _next(_base(), 0), _state(_base())
    state, _ = cm.integrate_candidate(state, base, candidate, request_id="request")
    with pytest.raises(cm.TransitionError, match="different payload"):
        cm.integrate_candidate(state, candidate, _next(candidate, 1), request_id="request")
    with pytest.raises(cm.TransitionError, match="CAS"):
        cm.integrate_candidate(_state(base), candidate, _next(candidate, 1), request_id="other")


def test_integration_refuses_silent_unrelated_target_change():
    base, candidate = _base(), _next(_base(), 0)
    raw = candidate.to_dict()
    raw["targets"][0]["model_digest"] = _h("d")
    forged = cm.CandidateManifest.from_dict(raw)
    with pytest.raises(cm.TransitionError, match="not declared"):
        cm.integrate_candidate(_state(base), base, forged, request_id="forged")


def test_empty_required_obligations_and_missing_loo_coverage_refuse_start():
    comparator, candidate = _base(), _next(_base(), 0)
    seed = _row("seed", candidate, comparator, required=False, row_kind="seed")
    direct = cm.RequiredRowSet("empty-required", (seed,))
    with pytest.raises(cm.CandidateError, match="at least one required"):
        cm.RequiredRowSet.from_dict(direct.to_dict())
    row = _row("gpu", candidate, comparator)
    rows = _row_set([row])
    batch, _ = _batch(candidate, comparator, [row])
    batch = replace(batch, required_loo_keep_ids=()).validated()
    state = replace(_state(comparator), integration_tip=candidate.manifest_digest)
    with pytest.raises(cm.TransitionError, match="LOO coverage"):
        cm.start_batch(state, batch, rows, candidate, comparator)


def test_row_receipt_mismatch_and_duplicate_terminal_result_refuse():
    comparator, candidate = _base(), _next(_base(), 0)
    row = _row("gpu", candidate, comparator)
    batch, rows = _batch(candidate, comparator, [row])
    receipt = _receipt(batch, "gpu")
    passed = cm.ValidationRowState("gpu", "passed", None, receipt)
    updated = cm.record_row(batch, rows, passed)
    assert cm.record_row(updated, rows, passed) == updated
    failed = cm.ValidationRowState("gpu", "failed", "late", None)
    with pytest.raises(cm.TransitionError, match="terminal"):
        cm.record_row(updated, rows, failed)
    wrong = replace(receipt, comparator_manifest_digest=_h("0"))
    with pytest.raises(cm.TransitionError, match="identities"):
        cm.record_row(batch, rows, cm.ValidationRowState("gpu", "passed", None, wrong))


def test_completion_cannot_retarget_frozen_batch_and_pass_labels_retain_debt():
    comparator, candidate = _base(), _next(_base(), 0)
    row = _row("gpu", candidate, comparator)
    rows = _row_set([row])
    proto, _ = _batch(candidate, comparator, [row])
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(proto, "gpu"))
    completed = replace(proto, rows=(passed,)).validated()
    state = replace(_state(comparator), integration_tip=candidate.manifest_digest)
    state = cm.start_batch(state, proto, rows, candidate, comparator)
    retargeted = replace(completed, comparator_manifest_digest=_h("0"))
    with pytest.raises((cm.CandidateError, cm.TransitionError)):
        cm.complete_batch(state, retargeted)
    state = cm.complete_batch(state, completed)
    assert proto.batch_id in state.validation_debt


def test_optional_pending_row_does_not_block_required_batch_completion():
    comparator, candidate = _base(), _next(_base(), 0)
    required = _row("gpu", candidate, comparator)
    seed = _row("seed", candidate, comparator, required=False, row_kind="seed")
    rows = _row_set([required, seed])
    batch, _ = _batch(candidate, comparator, [required, seed])
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(batch, "gpu"))
    completed = replace(batch, rows=(passed, batch.rows[1])).validated()
    state = replace(_state(comparator), integration_tip=candidate.manifest_digest)
    state = cm.complete_batch(cm.start_batch(state, batch, rows, candidate, comparator), completed)
    assert any(item.row_id == "seed" and item.status == "pending"
               for item in state.completed_batches[0].rows)


def _completed_state(candidate, comparator, row_states, row_set, *, validated=None,
                     summaries=(), integration=None):
    batch = cm.ValidationBatch("batch-1", candidate.manifest_digest,
                               comparator.manifest_digest, row_set.row_set_digest,
                               validated, candidate.manifest_digest, 0, 0,
                               tuple(item.keep_id for item in candidate.keeps),
                               tuple(row.row_id for row in row_set.rows if row.required),
                               tuple(row_states)).validated()
    state = _state(comparator, validated=validated, summaries=summaries)
    state = replace(state, integration_tip=candidate.manifest_digest)
    state = cm.start_batch(state, batch, row_set, candidate, comparator)
    if integration is not None:
        state = replace(state, integration_tip=integration.manifest_digest,
                        keeps_since_gate=1).validated()
    return cm.complete_batch(state, batch), batch


def test_cpu_pass_gpu_missing_blocks_but_optional_seed_missing_does_not():
    comparator_raw = _manifest()
    comparator_raw["targets"][0]["backend"] = "both"
    comparator = cm.CandidateManifest.from_dict(comparator_raw)
    candidate = _next(comparator, 0)
    cpu = _row("cpu", candidate, comparator, backend="cpu")
    gpu = _row("gpu", candidate, comparator)
    rows = _row_set([cpu, gpu])
    proto, _ = _batch(candidate, comparator, [cpu, gpu])
    cpu_pass = cm.ValidationRowState("cpu", "passed", None, _receipt(proto, "cpu"))
    gpu_missing = cm.ValidationRowState("gpu", "prerequisite_missing", "no GPU", None)
    state, batch = _completed_state(candidate, comparator, [cpu_pass, gpu_missing], rows)
    with pytest.raises(cm.TransitionError, match="not passed"):
        cm.advance_validated(state, batch, rows, candidate, comparator,
                             verifier=lambda *_: True)

    comparator, candidate = _base(), _next(_base(), 0)
    required_gpu = _row("required-gpu", candidate, comparator)
    seed = _row("seed", candidate, comparator, required=False, row_kind="seed")
    rows = _row_set([required_gpu, seed])
    proto, _ = _batch(candidate, comparator, [required_gpu, seed])
    cpu_pass = cm.ValidationRowState("required-gpu", "passed", None,
                                     _receipt(proto, "required-gpu"))
    seed_missing = cm.ValidationRowState("seed", "prerequisite_missing", "optional", None)
    state, batch = _completed_state(candidate, comparator, [cpu_pass, seed_missing], rows)
    advanced = cm.advance_validated(state, batch, rows, candidate, comparator,
                                    verifier=lambda *_: True, **_loo_kwargs(candidate, rows))
    assert advanced.validated_candidate == candidate.manifest_digest


def test_untrusted_json_pass_wrong_recipe_and_wrong_predecessor_never_advance():
    comparator, candidate = _base(), _next(_base(), 0)
    row = _row("g2-conc", candidate, comparator)
    rows = _row_set([row])
    proto, _ = _batch(candidate, comparator, [row])
    passed = cm.ValidationRowState(row.row_id, "passed", None, _receipt(proto, row.row_id))
    state, batch = _completed_state(candidate, comparator, [passed], rows)
    with pytest.raises(cm.TrustedVerificationRequired):
        cm.advance_validated(state, batch, rows, candidate, comparator, verifier=None)
    wrong_row = replace(row, candidate_recipe_digest=_h("0"))
    wrong_rows = _row_set([wrong_row])
    with pytest.raises((cm.CandidateError, cm.TransitionError)):
        cm.advance_validated(state, batch, wrong_rows, candidate, comparator,
                             verifier=lambda *_: True)
    bad_cas = replace(batch, expected_validated_predecessor=_h("0"))
    with pytest.raises(cm.TransitionError, match="predecessor"):
        cm.advance_validated(state, bad_cas, rows, candidate, comparator,
                             verifier=lambda *_: True)


def test_loaded_completed_batch_cannot_omit_production_backend_obligation():
    comparator_raw = _manifest()
    comparator_raw["targets"][0]["backend"] = "both"
    comparator = cm.CandidateManifest.from_dict(comparator_raw)
    candidate = _next(comparator, 0)
    gpu = _row("gpu", candidate, comparator)
    rows = _row_set([gpu])
    batch, _ = _batch(candidate, comparator, [gpu])
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(batch, "gpu"))
    completed = replace(batch, rows=(passed,)).validated()
    state = replace(_state(comparator), integration_tip=candidate.manifest_digest,
                    completed_batches=(completed,), validation_debt=(completed.batch_id,)).validated()
    calls = []
    with pytest.raises(cm.TransitionError, match="omits"):
        cm.advance_validated(state, completed, rows, candidate, comparator,
                             verifier=lambda *_: calls.append(True) or True,
                             **_loo_kwargs(candidate, rows))
    assert calls == []


def test_loo_debt_needs_bound_eligible_result_and_trusted_verifier():
    comparator, candidate = _base(), _next(_base(), 0)
    plan = cm.plan_loo(candidate, "k0")
    assert plan.derived_manifest_digest is not None
    row = _row("gpu", candidate, comparator)
    rows = _row_set([row])
    proto, _ = _batch(candidate, comparator, [row])
    proto = replace(proto, required_loo_keep_ids=("k0",)).validated()
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(proto, "gpu"))
    completed = replace(proto, rows=(passed,)).validated()
    state = replace(_state(comparator), integration_tip=candidate.manifest_digest)
    state = cm.complete_batch(cm.start_batch(state, proto, rows, candidate, comparator),
                              completed)
    result = cm.LOOResult.from_dict({
        "schema": cm.LOO_RESULT_SCHEMA, "plan_digest": plan.plan_digest,
        "candidate_manifest_digest": candidate.manifest_digest, "keep_id": "k0",
        "derived_manifest_digest": plan.derived_manifest_digest,
        "row_set_digest": rows.row_set_digest, "receipt_digests": [_h("1")],
        "disposition": "neutral", "evidence_digest": _h("2"),
        "deletion_authorized": False})
    with pytest.raises(cm.TrustedVerificationRequired):
        cm.advance_validated(state, completed, rows, candidate, comparator,
                             verifier=lambda *_: True, loo_plans={"k0": plan},
                             loo_results={"k0": result})
    advanced = cm.advance_validated(
        state, completed, rows, candidate, comparator, verifier=lambda *_: True,
        loo_plans={"k0": plan}, loo_results={"k0": result},
        loo_verifier=lambda *_: True)
    assert completed.batch_id not in advanced.validation_debt


def test_trusted_verifier_exceptions_fail_as_typed_transition_refusals():
    comparator, candidate = _base(), _next(_base(), 0)
    row = _row("gpu", candidate, comparator)
    rows = _row_set([row])
    proto, _ = _batch(candidate, comparator, [row])
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(proto, "gpu"))
    state, batch = _completed_state(candidate, comparator, [passed], rows)

    def explode(*_):
        raise LookupError("adapter unavailable")

    with pytest.raises(cm.TransitionError, match="verifier errored"):
        cm.advance_validated(state, batch, rows, candidate, comparator,
                             verifier=explode, **_loo_kwargs(candidate, rows))


def test_inconclusive_completed_gate_resets_cadence_but_retains_debt_and_refusal_does_not():
    comparator, candidate = _base(), _next(_base(), 0)
    row = _row("gpu", candidate, comparator)
    rows = _row_set([row])
    batch, _ = _batch(candidate, comparator, [row], states=[
        cm.ValidationRowState("gpu", "inconclusive", "wide interval", None)])
    state = replace(_state(comparator, cadence=4), integration_tip=candidate.manifest_digest)
    batch = replace(batch, accounted_keeps_since_gate=state.keeps_since_gate).validated()
    state = cm.start_batch(state, batch, rows, candidate, comparator)
    refused = cm.refuse_gate_start(state, batch.batch_id, "grant unavailable")
    assert refused.keeps_since_gate == 4
    completed = cm.complete_batch(state, batch)
    assert completed.keeps_since_gate == 0
    assert batch.batch_id in completed.validation_debt


def test_frozen_old_batch_can_advance_its_candidate_after_integration_moves():
    comparator, candidate = _base(), _next(_base(), 0)
    newer = _next(candidate, 1)
    row = _row("gpu", candidate, comparator)
    rows = _row_set([row])
    proto, _ = _batch(candidate, comparator, [row])
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(proto, "gpu"))
    state, batch = _completed_state(candidate, comparator, [passed], rows,
                                    summaries=("summary-old",), integration=newer)
    advanced = cm.advance_validated(state, batch, rows, candidate, comparator,
                                    verifier=lambda *_: True, **_loo_kwargs(candidate, rows))
    assert advanced.integration_tip == newer.manifest_digest
    assert advanced.keeps_since_gate == 1
    assert advanced.validated_candidate == candidate.manifest_digest
    assert advanced.stale_summary_refs == ("summary-old",)
