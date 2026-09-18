"""Hermetic tests for the unified-loop campaign enrollment boundary."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest
import yaml

from . import campaign


def _artifact(kind: str, ref: str, digit: str = "a") -> dict:
    return {"schema": campaign.ARTIFACT_SCHEMA, "kind": kind, "ref": ref,
            "path": f"/artifacts/{kind}/{ref}", "sha256": digit * 64}


def _target(target_id: str, *, request_id: str = "request-1", backend: str = "gpu",
            model: str = "model-a", build: str = "build-a", recipe: str = "recipe-a",
            baseline: str | None = "production-v9", roles=("candidate",),
            obligations=("gpu-serving",), context: int = 16384,
            concurrency: int = 4) -> dict:
    row = {"schema": campaign.TARGET_SCHEMA, "request_id": request_id,
           "target_id": target_id, "backend": backend, "model_ref": model,
           "build_ref": build, "recipe_ref": recipe, "context": context,
           "concurrency": concurrency, "speculation": "self_draft", "env": {"KNOB": "1"},
           "metric": "aggregate_tok_s", "metric_direction": "higher",
           "roles": list(roles), "required_obligations": list(obligations)}
    if baseline is not None:
        row["baseline_ref"] = baseline
    return row


def _manifest(*, request_id: str = "request-1", production=None, seeds=None) -> dict:
    return {"schema": campaign.MANIFEST_SCHEMA, "campaign_id": "unified-ak-test",
            "request_id": request_id, "source_snapshot": {"kernel": "ef81196d5",
                                                            "recipes": "1d9733f1"},
            "resources": {"schema": campaign.RESOURCE_SCHEMA,
                          "cpu_logical": [0, 1], "gpu_ids": ["ROCm0"],
                          "stage_timeout_s": 900, "build_timeout_s": 1800,
                          "build_jobs": 8, "max_builds": 2},
            "objective_ref": "objective/aggregate-throughput-v1",
            "actors": {"planner": "gpt-5.6-sol", "critic": "fable-5.1"},
            "fallbacks": {"planner": ["gpt-5.5"], "critic": []},
            "production": production if production is not None else [],
            "seeds": seeds if seeds is not None else []}


def _registry(*, baseline_digit: str = "d") -> dict:
    return {"source": {"ef81196d5": _artifact("source", "ef81196d5", "1"),
                       "1d9733f1": _artifact("source", "1d9733f1", "2")},
            "model": {"model-a": _artifact("model", "model-a", "a"),
                      "model-b": _artifact("model", "model-b", "b")},
            "build": {"build-a": _artifact("build", "build-a", "c"),
                      "production-v9": _artifact("build", "production-v9", baseline_digit)},
            "recipe": {"recipe-a": _artifact("recipe", "recipe-a", "e")}}


def test_json_and_yaml_load_the_same_versioned_manifest(tmp_path):
    raw = _manifest(production=[_target("prod")])
    json_path = tmp_path / "campaign.json"
    yaml_path = tmp_path / "campaign.yaml"
    json_path.write_text(json.dumps(raw))
    yaml_path.write_text(yaml.safe_dump(raw))
    left = campaign.load_manifest(json_path)
    right = campaign.load_manifest(yaml_path)
    assert left == right
    assert campaign.CampaignManifest.from_dict(left.to_dict()) == left
    assert left.manifest_digest == right.manifest_digest


def test_manifest_and_nested_records_are_immutable():
    loaded = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    with pytest.raises(FrozenInstanceError):
        loaded.request_id = "changed"
    with pytest.raises(TypeError):
        loaded.source_snapshot[0][1] = "changed"
    assert isinstance(loaded.production, tuple)
    assert isinstance(loaded.production[0].env, tuple)


@pytest.mark.parametrize("field,value", [
    ("stage_timeout_s", 0), ("build_timeout_s", float("inf")),
    ("build_jobs", True), ("max_builds", -1),
])
def test_resource_limits_are_strict_positive_finite_integers(field, value):
    raw = _manifest(production=[_target("prod")])
    raw["resources"][field] = value
    with pytest.raises(campaign.ManifestError, match=field):
        campaign.CampaignManifest.from_dict(raw)


def test_cpu_gpu_and_both_targets_require_their_declared_resources():
    for backend, empty_field in (("cpu", "cpu_logical"), ("gpu", "gpu_ids"),
                                 ("both", "cpu_logical"), ("both", "gpu_ids")):
        raw = _manifest(production=[_target("prod", backend=backend)])
        raw["resources"][empty_field] = []
        with pytest.raises(campaign.ManifestError, match="require non-empty"):
            campaign.CampaignManifest.from_dict(raw)


def test_unknown_schema_fields_and_implicit_fallbacks_are_refused():
    raw = _manifest(production=[_target("prod")])
    raw["surprise"] = True
    with pytest.raises(campaign.ManifestError, match="unknown"):
        campaign.CampaignManifest.from_dict(raw)
    raw = _manifest(production=[_target("prod")])
    del raw["fallbacks"]["critic"]
    with pytest.raises(campaign.ManifestError, match="explicitly name exactly"):
        campaign.CampaignManifest.from_dict(raw)


def test_pure_resolution_produces_exact_local_identities_and_round_trips():
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    resolved = campaign.resolve_manifest(manifest, registry_snapshot=_registry())
    assert resolved.targets[0].status == "ready"
    assert resolved.targets[0].execution.model.sha256 == "a" * 64
    assert resolved.targets[0].baseline.ref == "production-v9"
    assert campaign.ResolvedCampaign.from_dict(resolved.to_dict()) == resolved


def test_injected_resolver_does_no_implicit_io_and_missing_is_per_target():
    calls = []
    registry = _registry()

    def resolver(kind, ref, snapshot):
        calls.append((kind, ref))
        return snapshot.get(kind, {}).get(ref)

    raw = _manifest(production=[_target("good"), _target("missing", model="off-disk")])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=registry,
                                         resolve_artifact=resolver)
    by_id = {target.target_ids[0]: target for target in resolved.targets}
    assert by_id["good"].status == "ready"
    assert by_id["missing"].status == "missing_artifact"
    assert by_id["missing"].missing == ("model:missing",)
    assert ("model", "off-disk") in calls


def test_unsupported_capability_is_distinct_and_does_not_block_peer():
    def resolver(kind, ref, snapshot):
        if kind == "recipe" and ref == "recipe-unsupported":
            return {"status": "unsupported_capability", "reason": "backend has no RPC"}
        return _registry().get(kind, {}).get(ref)

    raw = _manifest(production=[_target("good"),
                                _target("unsupported", recipe="recipe-unsupported")])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot={}, resolve_artifact=resolver)
    by_id = {target.target_ids[0]: target for target in resolved.targets}
    assert by_id["good"].status == "ready"
    assert by_id["unsupported"].status == "unsupported_capability"
    assert "recipe:unsupported:backend has no RPC" in by_id["unsupported"].missing


def test_alias_duplicates_union_roles_and_obligations_with_one_seed_boost():
    production = _target("production-alias", roles=("production",),
                         obligations=("cpu-regression",))
    seed1 = _target("seed-alias-1", roles=("candidate",), obligations=("gpu-serving",))
    seed2 = _target("seed-alias-2", roles=("diagnostic",), obligations=("loo",))
    raw = _manifest(production=[production], seeds=[seed1, seed2])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=_registry())
    assert len(resolved.targets) == 1
    target = resolved.targets[0]
    assert target.target_ids == ("production-alias", "seed-alias-1", "seed-alias-2")
    assert target.enrolled_as == ("production", "seed")
    assert target.roles == ("candidate", "diagnostic", "production")
    assert target.required_obligations == ("cpu-regression", "gpu-serving", "loo")
    assert target.seed_boost_units == 1


def test_an_identical_duplicate_seed_is_a_noop_not_a_second_boost():
    seed = _target("same-seed")
    raw = _manifest(seeds=[seed, dict(seed)])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=_registry())
    assert len(resolved.targets) == 1
    assert resolved.targets[0].target_ids == ("same-seed",)
    assert resolved.targets[0].seed_boost_units == 1


def test_same_target_id_cannot_name_two_workloads():
    raw = _manifest(production=[_target("same")],
                    seeds=[_target("same", context=8192)])
    with pytest.raises(campaign.ManifestError, match="conflicting workloads"):
        campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                  registry_snapshot=_registry())


def test_complete_workload_fields_move_the_signature():
    base = campaign.CampaignManifest.from_dict(_manifest(production=[_target("a")]))
    changed = campaign.CampaignManifest.from_dict(
        _manifest(production=[_target("a", concurrency=8)]))
    one = campaign.resolve_manifest(base, registry_snapshot=_registry()).targets[0]
    two = campaign.resolve_manifest(changed, registry_snapshot=_registry()).targets[0]
    assert one.workload_signature != two.workload_signature


def test_same_request_is_idempotent_and_conflicting_same_request_refuses():
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    first = campaign.resolve_manifest(manifest, registry_snapshot=_registry())
    assert campaign.resolve_manifest(manifest, registry_snapshot={}, previous=first) is first
    conflict = campaign.CampaignManifest.from_dict(
        _manifest(production=[_target("prod", concurrency=8)]))
    with pytest.raises(campaign.ManifestError, match="conflicting manifest"):
        campaign.resolve_manifest(conflict, registry_snapshot=_registry(), previous=first)


def test_moved_baseline_is_reported_and_the_pinned_identity_is_retained():
    first_manifest = campaign.CampaignManifest.from_dict(
        _manifest(production=[_target("prod")]))
    first = campaign.resolve_manifest(first_manifest, registry_snapshot=_registry(baseline_digit="d"))
    raw = _manifest(request_id="request-2",
                    production=[_target("prod", request_id="request-2")])
    second = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                       registry_snapshot=_registry(baseline_digit="f"),
                                       previous=first)
    assert second.targets[0].revision == 2
    assert second.targets[0].status == "mismatched_ref"
    assert second.targets[0].baseline.sha256 == "d" * 64
    assert second.targets[0].missing == ("baseline_ref_moved",)


def test_an_explicit_new_baseline_ref_creates_a_new_pinned_revision():
    first_manifest = campaign.CampaignManifest.from_dict(
        _manifest(production=[_target("prod")]))
    registry = _registry()
    first = campaign.resolve_manifest(first_manifest, registry_snapshot=registry)
    registry["build"]["production-v10"] = _artifact("build", "production-v10", "f")
    raw = _manifest(request_id="request-2",
                    production=[_target("prod", request_id="request-2",
                                        baseline="production-v10")])
    second = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                       registry_snapshot=registry, previous=first)
    assert second.targets[0].revision == 2
    assert second.targets[0].status == "ready"
    assert second.targets[0].baseline.ref == "production-v10"


def test_cpu_gpu_and_both_views_include_shared_targets():
    raw = _manifest(production=[_target("cpu", backend="cpu", model="model-a"),
                                _target("gpu", backend="gpu", model="model-b"),
                                _target("both", backend="both", context=8192)])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=_registry())
    assert {t.execution.backend for t in resolved.targets_for("cpu")} == {"cpu", "both"}
    assert {t.execution.backend for t in resolved.targets_for("gpu")} == {"gpu", "both"}
    assert resolved.targets_for("both") == resolved.targets


def test_previous_from_another_campaign_is_never_an_identity_source():
    first = campaign.resolve_manifest(
        campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")])),
        registry_snapshot=_registry())
    raw = _manifest(request_id="request-2", production=[_target("prod", request_id="request-2")])
    raw["campaign_id"] = "another-campaign"
    with pytest.raises(campaign.ManifestError, match="different campaign"):
        campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                  registry_snapshot=_registry(), previous=first)


def test_content_identical_artifact_aliases_deduplicate_despite_refs_and_paths():
    registry = _registry()
    registry["model"]["model-alias"] = {
        **_artifact("model", "model-alias", "a"), "path": "/another/model/copy"}
    raw = _manifest(seeds=[_target("one", model="model-a"),
                           _target("two", model="model-alias")])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=registry)
    assert len(resolved.targets) == 1
    assert resolved.targets[0].target_ids == ("one", "two")


def test_same_execution_with_different_baselines_remains_two_complete_obligations():
    registry = _registry()
    registry["build"]["other-baseline"] = _artifact("build", "other-baseline", "f")
    raw = _manifest(production=[
        _target("old-base", baseline="production-v9", obligations=("gpu-serving",)),
        _target("new-base", baseline="other-baseline", obligations=("cpu-regression",)),
    ])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=registry)
    assert len(resolved.targets) == 2
    assert {item.baseline.ref for item in resolved.targets} == {
        "production-v9", "other-baseline"}
    assert {ob for item in resolved.targets for ob in item.required_obligations} == {
        "gpu-serving", "cpu-regression"}


def test_same_content_aliases_with_different_baseline_labels_are_refused():
    registry = _registry()
    registry["build"]["production-v9-alias"] = {
        **_artifact("build", "production-v9-alias", "d"),
        "path": "/another/production-v9-copy"}
    raw = _manifest(seeds=[_target("one", baseline="production-v9"),
                           _target("two", baseline="production-v9-alias")])
    with pytest.raises(campaign.ManifestError, match="different baseline refs"):
        campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                  registry_snapshot=registry)


def test_distinct_missing_baseline_refs_do_not_merge():
    raw = _manifest(seeds=[
        _target("one", baseline="missing-base-a"),
        _target("two", baseline="missing-base-b"),
    ])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=_registry())
    assert len(resolved.targets) == 2
    assert {item.baseline_ref for item in resolved.targets} == {
        "missing-base-a", "missing-base-b"}


def test_execution_description_round_trips_and_signature_tampering_is_refused():
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    target = campaign.resolve_manifest(manifest, registry_snapshot=_registry()).targets[0]
    assert target.execution.context == 16384
    assert target.execution.concurrency == 4
    assert target.execution.speculation == "self_draft"
    assert dict(target.execution.env) == {"KNOB": "1"}
    assert target.execution.metric_direction == "higher"
    tampered = target.to_dict()
    tampered["execution"]["context"] = 8192
    with pytest.raises(campaign.ManifestError, match="does not match"):
        campaign.TargetRevision.from_dict(tampered)


def test_execution_artifact_cannot_be_relabelled_away_from_its_resolved_ref():
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    row = campaign.resolve_manifest(manifest, registry_snapshot=_registry()).targets[0].to_dict()
    row["execution"]["model_ref"] = "different-label"
    with pytest.raises(campaign.ManifestError, match="does not match its ref"):
        campaign.TargetRevision.from_dict(row)


def test_distinct_missing_drafter_refs_do_not_alias():
    one = _target("one")
    one["drafter_ref"] = "missing-drafter-a"
    two = _target("two")
    two["drafter_ref"] = "missing-drafter-b"
    resolved = campaign.resolve_manifest(
        campaign.CampaignManifest.from_dict(_manifest(seeds=[one, two])),
        registry_snapshot=_registry())
    assert len(resolved.targets) == 2
    assert all(item.status == "missing_artifact" for item in resolved.targets)


def test_each_kind_ref_resolves_once_against_a_deep_frozen_copy():
    registry = _registry()
    calls = {}

    def resolver(kind, ref, snapshot):
        calls[(kind, ref)] = calls.get((kind, ref), 0) + 1
        with pytest.raises(TypeError):
            snapshot[kind][ref]["sha256"] = "0" * 64
        # Mutating the caller's object cannot move the resolver's frozen snapshot.
        registry[kind][ref]["sha256"] = "f" * 64
        return snapshot.get(kind, {}).get(ref)

    raw = _manifest(seeds=[_target("one"), _target("two")])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=registry,
                                         resolve_artifact=resolver)
    assert all(count == 1 for count in calls.values())
    assert resolved.targets[0].execution.model.sha256 == "a" * 64


def test_source_snapshot_is_resolved_once_and_missing_source_marks_each_target():
    registry = _registry()
    del registry["source"]["ef81196d5"]
    raw = _manifest(production=[_target("one")],
                    seeds=[_target("two", model="model-b")])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                         registry_snapshot=registry)
    assert dict(resolved.source_snapshot)["kernel"] is None
    assert all(item.status == "missing_artifact" for item in resolved.targets)
    assert all("source:kernel:source:missing" in item.missing for item in resolved.targets)


def test_omitted_baseline_pins_the_current_resolved_build():
    manifest = campaign.CampaignManifest.from_dict(
        _manifest(seeds=[_target("seed", baseline=None)]))
    resolved = campaign.resolve_manifest(manifest, registry_snapshot=_registry())
    target = resolved.targets[0]
    assert target.status == "ready"
    assert target.baseline == target.execution.build


def test_omitted_baseline_stays_pinned_when_build_ref_moves_between_requests():
    first_manifest = campaign.CampaignManifest.from_dict(
        _manifest(seeds=[_target("seed", baseline=None)]))
    first = campaign.resolve_manifest(first_manifest, registry_snapshot=_registry())
    moved = _registry()
    moved["build"]["build-a"] = _artifact("build", "build-a", "f")
    raw = _manifest(request_id="request-2",
                    seeds=[_target("seed", request_id="request-2", baseline=None)])
    second = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                       registry_snapshot=moved, previous=first)
    target = second.targets[0]
    assert target.status == "mismatched_ref"
    assert target.execution.build.sha256 == "f" * 64
    assert target.baseline.sha256 == "c" * 64
    assert target.missing == ("implicit_baseline_build_moved",)


def test_empty_environment_value_is_preserved_without_weakening_text_fields():
    raw = _manifest(production=[_target("prod")])
    raw["production"][0]["env"] = {"EMPTY_OK": ""}
    manifest = campaign.CampaignManifest.from_dict(raw)
    resolved = campaign.resolve_manifest(manifest, registry_snapshot=_registry())
    assert dict(resolved.targets[0].execution.env) == {"EMPTY_OK": ""}
    assert campaign.ResolvedCampaign.from_dict(resolved.to_dict()) == resolved

    raw["actors"]["planner"] = ""
    with pytest.raises(campaign.ManifestError, match="must be a non-empty string"):
        campaign.CampaignManifest.from_dict(raw)


@pytest.mark.parametrize(("mutation", "reason"), [
    (lambda row: row.update(seed_boost_units=2), "seed_boost_units"),
    (lambda row: row.update(enrolled_as=["invented"]), "provenance"),
    (lambda row: row.update(model=None), "unknown"),
])
def test_deserialized_target_revision_refuses_authority_weakening(mutation, reason):
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    row = campaign.resolve_manifest(manifest, registry_snapshot=_registry()).targets[0].to_dict()
    mutation(row)
    with pytest.raises(campaign.ManifestError, match=reason):
        campaign.TargetRevision.from_dict(row)


def test_deserialized_ready_target_requires_every_core_identity():
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    row = campaign.resolve_manifest(manifest, registry_snapshot=_registry()).targets[0].to_dict()
    row["execution"]["recipe"] = None
    row["workload_signature"] = campaign._workload_signature(
        campaign.ResolvedExecution.from_dict(row["execution"]),
        campaign.ArtifactIdentity.from_dict(row["baseline"], kind="build",
                                            ref=row["baseline"]["ref"]),
        row["baseline_ref"])
    with pytest.raises(campaign.ManifestError, match="needs model/build/recipe/baseline"):
        campaign.TargetRevision.from_dict(row)


def test_deserialized_ready_target_requires_declared_drafter_identity():
    raw = _manifest(production=[_target("prod")])
    raw["production"][0]["drafter_ref"] = "model-b"
    target = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                       registry_snapshot=_registry()).targets[0]
    row = target.to_dict()
    row["execution"]["drafter"] = None
    execution = campaign.ResolvedExecution.from_dict(row["execution"])
    row["workload_signature"] = campaign._workload_signature(
        execution, campaign.ArtifactIdentity.from_dict(
            row["baseline"], kind="build", ref=row["baseline"]["ref"]),
        row["baseline_ref"])
    with pytest.raises(campaign.ManifestError, match="declared drafter"):
        campaign.TargetRevision.from_dict(row)


def test_deserialized_seed_boost_requires_seed_provenance():
    target = campaign.resolve_manifest(
        campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")])),
        registry_snapshot=_registry()).targets[0]
    row = target.to_dict()
    row["seed_boost_units"] = 1
    with pytest.raises(campaign.ManifestError, match="match seed provenance"):
        campaign.TargetRevision.from_dict(row)


def test_deserialized_resolved_campaign_rechecks_backend_resources():
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    row = campaign.resolve_manifest(manifest, registry_snapshot=_registry()).to_dict()
    row["resources"]["gpu_ids"] = []
    with pytest.raises(campaign.ManifestError, match="require GPU resources"):
        campaign.ResolvedCampaign.from_dict(row)


def test_load_refuses_non_manifest_suffix_without_guessing(tmp_path):
    path = tmp_path / "campaign.txt"
    path.write_text("{}")
    with pytest.raises(campaign.ManifestError, match="suffix"):
        campaign.load_manifest(path)
