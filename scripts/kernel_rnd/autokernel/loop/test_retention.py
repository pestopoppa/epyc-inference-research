from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from .. import storage
from . import pool, retention as R


def expiry():
    return {"schema": R.EXPIRY_SCHEMA, "campaign_id": "campaign",
            "sha256": "a" * 64, "durability_class": "hash_and_provenance_only",
            "expirable_kind": "rejected_candidate_build_tree",
            "reason": "rejected candidate", "rule_id": "retention-v1",
            "actor": "retention-worker", "preconditions": {
                "candidate_id": "candidate", "candidate_status": "rejected",
                "champion_status": "none", "evaluation_events_journaled": True}}


def node(name, *, kind="build_dir", deps=(), path=None,
         klass="permanent_large", size=None):
    return {"schema": R.NODE_SCHEMA, "artifact_id": name, "artifact_kind": kind,
            "dependencies": list(deps), "path": path, "retention_class": klass,
            "declared_size_bytes": size,
            "expiry": expiry() if klass == "expirable" else None}


def root(name, kind, artifacts):
    return {"schema": R.ROOT_SCHEMA, "root_id": name, "root_kind": kind,
            "artifact_ids": list(artifacts)}


def snapshot(nodes, roots, *, generation=1, uncertain=()):
    return R.RetentionSnapshot.create(snapshot_id="snapshot", generation=generation,
                                      nodes=nodes, roots=roots,
                                      uncertain_scopes=uncertain)


@pytest.mark.parametrize("kind", sorted(R.ROOT_KINDS))
def test_every_declared_root_kind_retains_its_complete_closure(kind):
    snap = snapshot([node("root"), node("dep", kind="source_ref"),
                     node("leaf", kind="evidence_record")],
                    [root(kind, kind, ["root"])])
    raw = snap.to_dict()
    raw["nodes"][0]["dependencies"] = ["dep"]
    raw["nodes"][1]["dependencies"] = ["leaf"]
    snap = R.RetentionSnapshot.create(snapshot_id="snapshot", generation=1,
                                      nodes=raw["nodes"], roots=raw["roots"])
    plan = R.plan_retention(snap)
    assert plan.complete
    assert plan.retained_ids == ("dep", "leaf", "root")
    assert plan.expirable_ids == ()


def test_runpath_shared_dso_and_pending_old_candidate_are_retained():
    nodes = [node("pending", deps=("candidate-build",)),
             node("candidate-build", deps=("shared-dso",), path="/owned/build"),
             node("shared-dso", kind="shared_dso_dir", path="/owned/dso"),
             node("spent", klass="expirable", path="/owned/spent", size=123)]
    snap = snapshot(nodes, [root("batch", "pending_validation", ["pending"])])
    plan = R.plan_retention(snap)
    assert plan.retained_ids == ("candidate-build", "pending", "shared-dso")
    assert plan.retained_paths == ("/owned/build", "/owned/dso")
    assert plan.expirable_ids == ("spent",)
    assert plan.expirable_total_bytes == 123


def test_orphan_intent_cycles_and_duplicate_root_reasons_are_deterministic():
    snap = snapshot([node("a", deps=("b",)), node("b", deps=("a",))], [
        root("orphan", "orphan_ref", ["a"]),
        root("intent", "integration_intent", ["b"]),
    ])
    plan = R.plan_retention(snap)
    assert plan.retained_ids == ("a", "b")
    assert {"integration_intent:intent", "orphan_ref:orphan"}.issubset(
        plan.retained_reasons["a"])
    assert plan.retained_reasons["a"] == tuple(sorted(plan.retained_reasons["a"]))
    assert R.RetentionPlan.from_dict(plan.to_dict()) == plan


def test_missing_reference_and_uncertain_scope_withhold_all_reclamation():
    candidate = node("spent", klass="expirable", path="/owned/spent", size=8)
    missing = snapshot([node("root", deps=("absent",)), candidate],
                       [root("production", "production", ["root"])])
    plan = R.plan_retention(missing)
    assert not plan.complete and plan.expirable_ids == ()
    assert "missing artifact absent" in plan.unknown_closure[0]
    uncertain = R.plan_retention(snapshot(
        [candidate], [root("evidence", "retained_evidence", ["spent"])],
        uncertain=("journal cursor unavailable",)))
    assert not uncertain.complete and uncertain.expirable_ids == ()
    dangling_candidate = node(
        "dangling", deps=("missing-dso",), klass="expirable", path="/owned/dangling")
    holder = node("holder")
    dangling = R.plan_retention(snapshot(
        [holder, dangling_candidate], [root("p", "production", ["holder"])]))
    assert not dangling.complete and dangling.expirable_ids == ()


def test_newer_or_corrupt_schema_and_bool_generation_refuse_closed():
    snap = snapshot([node("a")], [root("p", "production", ["a"])]).to_dict()
    for mutation in (
            lambda row: row.update(schema="epyc.autokernel.retention_snapshot.v2"),
            lambda row: row.update(generation=True),
            lambda row: row["nodes"][0].update(dependencies=["x", "x"])):
        bad = R.RetentionSnapshot.from_dict(snap).to_dict()
        mutation(bad)
        with pytest.raises(R.RetentionRefused):
            R.RetentionSnapshot.from_dict(bad)


def test_direct_inputs_are_deeply_frozen_and_digest_checked():
    dependencies = ["dep"]
    source = node("a", deps=dependencies)
    snap = snapshot([source, node("dep")], [root("p", "production", ["a"])])
    dependencies.append("late")
    source["dependencies"].append("also-late")
    assert snap.nodes[0].dependencies == ("dep",)
    with pytest.raises(FrozenInstanceError):
        snap.generation = 2
    forged = snap.to_dict()
    forged["generation"] = 2
    with pytest.raises(R.RetentionRefused, match="digest"):
        R.RetentionSnapshot.from_dict(forged)


def test_snapshot_generation_gate_rejects_changed_generation_or_content():
    first = snapshot([node("a")], [root("p", "production", ["a"])])
    plan = R.plan_retention(first)
    changed = snapshot([node("a")], [root("p", "production", ["a"])], generation=2)
    with pytest.raises(R.RetentionRefused, match="no longer matches"):
        R.validate_plan_snapshot(plan, changed)


def test_unified_prune_remains_non_destructive_without_native_authority(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    generations = [store / f"anchor-gen-{number:03d}" for number in range(1, 4)]
    for generation in generations:
        generation.mkdir()
    snap = snapshot([node("current", path=str(generations[0])),
                     node("old", path=str(generations[1]))],
                    [root("production", "production", ["current"])])
    plan = R.plan_retention(snap)
    report = pool.prune_anchor_generations(
        store, keep=1, unified=True, retention_plan=plan, retention_snapshot=snap)
    assert report.status == "retention_unknown" and not report.removed
    assert "tombstone expiry consumer" in report.retention_unknown[0]
    assert all(generation.exists() for generation in generations)

    changed = snapshot([node("current", path=str(generations[0]))],
                       [root("production", "production", ["current"])], generation=2)
    refused = pool.prune_anchor_generations(
        store, keep=1, unified=True, retention_plan=plan, retention_snapshot=changed)
    assert refused.status == "retention_unknown" and not refused.removed


def test_existing_storage_policy_dry_planner_is_the_only_expiry_authority(
        tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    target = owned / "candidate"
    target.mkdir(parents=True)
    (target / "file").write_bytes(b"fixture")
    candidate = node("spent", klass="expirable", path=str(target), size=None)
    snap = snapshot([candidate], [root("production", "production", [
        "keeper"])])
    # A missing root deliberately withholds the candidate.
    incomplete = R.plan_retention(snap)
    with pytest.raises(R.RetentionRefused, match="unknown"):
        R.plan_expiry_candidates(incomplete, snap, storage.StoragePolicy(
            campaign_quota_gb=1, owned_roots=(str(owned),)))
    complete = snapshot([candidate, node("keeper")], [
        root("production", "production", ["keeper"])])
    plan = R.plan_retention(complete)
    calls = []

    def existing_planner(artifact, policy, *, now=None):
        calls.append((artifact, policy, now))
        return "existing-storage-dry-plan"

    monkeypatch.setattr(storage, "plan_expiry", existing_planner)
    outcomes = R.plan_expiry_candidates(plan, complete, storage.StoragePolicy(
        campaign_quota_gb=1, owned_roots=(str(owned),)))
    assert outcomes == ("existing-storage-dry-plan",)
    assert calls[0][0].retention_class == "expirable"
    assert target.exists()


def test_permanent_and_source_classifications_never_become_expiry_candidates():
    permanent = snapshot([node("permanent")], [
        root("production", "production", ["permanent"])])
    assert R.plan_retention(permanent).expirable_ids == ()
    with pytest.raises(R.RetentionRefused, match="source refs"):
        R.ArtifactNode.from_dict(node(
            "source", kind="source_ref", klass="expirable", path="/owned/source"))


def test_path_overlap_retains_node_and_dependency_closure_to_fixed_point():
    snap = snapshot([
        node("serving", path="/owned/build/lib/libggml.so"),
        node("build", deps=("dso",), klass="expirable", path="/owned/build"),
        node("dso", klass="expirable", path="/owned/dso"),
    ], [root("production", "production", ["serving"])])
    plan = R.plan_retention(snap)
    assert plan.expirable_ids == ()
    assert plan.retained_ids == ("build", "dso", "serving")
    assert plan.retained_reasons["build"] == (
        "physical_path_overlap:serving",)
    assert plan.retained_reasons["dso"] == (
        "physical_path_overlap:serving",)


@pytest.mark.parametrize("path", ["/owned/../other", "/owned/./build", "/owned//build"])
def test_noncanonical_artifact_paths_are_refused_without_filesystem_resolution(path):
    with pytest.raises(R.RetentionRefused, match="lexically canonical"):
        R.ArtifactNode.from_dict(node("bad", path=path))
