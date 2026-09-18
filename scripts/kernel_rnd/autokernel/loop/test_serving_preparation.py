"""Hermetic preparation inputs; no kernel build, model, grant, or control PASS."""
from dataclasses import replace
import time

import pytest

from ..evaluator import api, statistics as st
from ..evaluator.test_statistics import make_controls, make_rule
from . import scheduling, serving_preparation as prep
from . import experiment_plan as ep, measurement_capture as mc, observation_binding as ob
from . import planned_serving as ps, serving
from .resolved_recipe import resolve_canonical_launch
from .test_experiment_plan import plan_dict
from .test_unified_planner import canonical_recipe
from .test_unified_worker import _prompt


def statistics():
    rule = make_rule()
    return prep.ServingStatisticsDeclaration(
        "original-seed", make_controls(calibration_block_count=2), rule,
        st.StoppingRuleCommitment.commit(rule, campaign_id="calibration-campaign",
                                         committed_at="2026-09-09T00:00:00+00:00"),
        st.StratumSplitRule("split-original", "original-seed", 0.25,
                            st.RotationSchedule("bundle-rotation", 1)),
        "sign_martingale_predictable_lambda/v1", st.EFFECT_SCALE_RELATIVE,
        "improvement", 0.0,
        st.OwningProtocolRepRule("explicit-fixture-protocol", st.REP_RULE_FLOOR, 5,
                                 "fixture declaration: original independent blocks"))


def neutral_copy(anchor):
    artifacts = {"model": anchor.model.to_dict(), "drafter": None,
                 "executable": {**anchor.executable.to_dict(),
                                "path": "/private-copy/bin/llama-server"},
                 "dsos": [item.to_dict() for item in anchor.dsos]}
    candidate = resolve_canonical_launch(
        anchor.template, build_dir="/private-copy",
        command_argv=("/private-copy/bin/llama-server", *anchor.command_argv[1:]),
        topology_prefix=anchor.topology_prefix, launch_environment=dict(anchor.launch_env),
        artifact_identities=artifacts, backend=anchor.backend,
        environment_policy=anchor.environment_policy, port=anchor.port,
        runtime_binary_dir="/private-copy/bin", runtime_ld_paths=anchor.runtime_ld_paths,
        provenance={**dict(anchor.provenance),
                    "material": "explicit-byte-identical-copy"})
    return prep.PreparationArmPair("neutral", anchor, candidate, "original-copy-reference")


def declaration(*, neutral=False, max_attempts=1):
    anchor = canonical_recipe()
    return prep.ServingPreparationDeclaration(
        "calibration-original", "calibration-campaign", "1" * 64,
        "2026-09-09T00:00:01+00:00",
        {"backend": "cpu", "phase": "decode", "cell_class": "serving",
         "model_sha256": anchor.model.sha256, "quant": "explicit-Q8_0",
         "metric": "aggregate_tok_s", "metric_direction": "higher_better",
         "estimator_id": "original-estimator", "epoch": "original-epoch"},
        statistics(), prep.PreparationArmPair("aa", anchor, anchor, None),
        neutral_copy(anchor) if neutral else None, "2" * 64,
        scheduling.ResourceVector(0.5, (), 1024), 60, 5,
        {"original-owner": "3" * 64},
        prep.PreparationRetryPolicy(max_attempts, ("failed", "contaminated")))


def test_full_original_statistics_and_declaration_round_trip():
    original = declaration(neutral=True)
    restored = prep.ServingPreparationDeclaration.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert restored.digest == original.digest
    assert restored.statistics.commitment.verify(restored.statistics.stopping_rule).outcome == api.schemas.PASS
    assert restored.statistics.campaign_seed == "original-seed"
    assert restored.preparation_debt == ()


def test_missing_neutral_is_explicit_preparation_debt_not_fake_material():
    original = declaration()
    assert original.preparation_debt == ("neutral_material_unavailable",)
    assert prep.ServingPreparationDeclaration.from_dict(original.to_dict()).neutral_pair is None


def test_preparation_frame_and_sources_are_detached_and_frozen():
    raw = declaration().to_dict()
    restored = prep.ServingPreparationDeclaration.from_dict(raw)
    raw["frame"]["quant"] = "mutated"
    raw["source_identities"]["original-owner"] = "4" * 64
    assert restored.frame["quant"] == "explicit-Q8_0"
    assert restored.source_identities["original-owner"] == "3" * 64
    with pytest.raises(TypeError):
        restored.frame["epoch"] = "mutated"


@pytest.mark.parametrize("field,value", [
    ("max_stage_seconds", float("inf")), ("teardown_seconds", float("nan")),
    ("max_stage_seconds", True), ("teardown_seconds", 0),
])
def test_explicit_budgets_are_finite_positive(field, value):
    with pytest.raises(prep.PreparationRefused, match="finite and positive"):
        replace(declaration(), **{field: value})


def test_no_runtime_dimension_is_invented_for_true_aa():
    original = declaration().aa_pair
    assert original.anchor.to_dict() == original.candidate.to_dict()
    assert "dimension" not in original.to_dict()
    with pytest.raises(prep.PreparationRefused, match="exact anchor"):
        replace(original, candidate=canonical_recipe(threads=8))


def test_neutral_cannot_be_relabelled_aa_or_changed_runtime():
    original = declaration().aa_pair
    with pytest.raises(prep.PreparationRefused, match="byte-identical"):
        replace(original, kind="neutral", neutral_material_ref="pretend-neutral")
    with pytest.raises(prep.PreparationRefused, match="byte-identical"):
        replace(neutral_copy(original.anchor), candidate=canonical_recipe(threads=8))


def test_original_commitment_must_predate_preparation():
    original = declaration()
    with pytest.raises(prep.PreparationRefused, match="later than"):
        replace(original, issued_at="2026-09-08T00:00:00+00:00")
    with pytest.raises(prep.PreparationRefused, match="original commitment"):
        replace(original.statistics, stopping_rule=make_rule(ceiling=21))
    with pytest.raises(prep.PreparationRefused, match="seeds differ"):
        replace(original.statistics, campaign_seed="retrospective-seed")


def test_closed_input_schemas_and_original_construction_registry():
    raw = declaration().to_dict()
    raw["statistics"]["controls"]["fabricated_threshold"] = 1.0
    with pytest.raises(prep.PreparationRefused, match="campaign controls fields differ"):
        prep.ServingPreparationDeclaration.from_dict(raw)
    with pytest.raises(st.ConstructionNotImplemented):
        replace(statistics(), construction_id="caller-owned-construction")


def request(tmp_path, *, kind="aa", block_start=0, blocks=2, attempt=0, max_attempts=1):
    original = declaration(neutral=kind == "neutral", max_attempts=max_attempts)
    pair = original.aa_pair if kind == "aa" else original.neutral_pair
    prompts = _prompt(pair.anchor.template)
    original = replace(original, prompt_manifest_digest=prep._digest(prompts.to_dict()))
    store = mc.ArtifactStore(tmp_path / "instrument")
    try:
        loaded = ob.seal_loaded_instrument(store=store, measurement_callable=serving._measure_once,
            fence_clock=time.monotonic, serving_timer=time.time).to_dict()
    finally:
        store.close()
    plan = plan_dict(n=blocks, instrument="serving", unit="process")
    plan.update(schema=ep.PLAN_SCHEMA_V2, phase="observation", record_class="observation",
        intended_use="explore", campaign_id=original.campaign_id,
        target_revision=original.target_revision, epoch=original.frame["epoch"],
        metric=original.frame["metric"], estimator_id=original.frame["estimator_id"],
        metric_direction="higher", changed_factors=[], calibration_ref=None,
        loaded_instrument=loaded, stopping={"kind": "fixed_n", "n_per_arm": blocks, "paired": True},
        anchor_identity=ps.arm_identity(pair.anchor.template, pair.anchor, loaded_instrument=loaded),
        candidate_identity=ps.arm_identity(pair.candidate.template, pair.candidate, loaded_instrument=loaded))
    schedule = st.OrderSchedule.derive(campaign_seed=original.statistics.campaign_seed,
        candidate_id=f"{original.declaration_id}:{kind}",
        base_blocks=original.statistics.controls.calibration_block_count, attempt=attempt)
    units, membership = [], []
    for local in range(blocks):
        index = local + block_start
        arms = ("anchor", "candidate") if schedule.order_for(index) == st.ORDER_ANCHOR_FIRST else (
            "candidate", "anchor")
        material = f"explicit-material-{kind}-{index}"
        row = {"block_index": index, "material_unit_id": material,
               "stratum": original.statistics.split_rule.assign(material)}
        for offset, arm in enumerate(arms):
            unit_id = f"{kind}-{index}-{arm}-attempt-{attempt}"
            row[f"{arm}_unit_id"] = unit_id
            units.append({"unit_id": unit_id, "arm": arm, "process_id": f"process:{unit_id}",
                "expected_prompt_ids": ["p1"], "order_index": 2 * local + offset, "pair_id": local})
        membership.append(row)
    plan["expected_units"] = units
    stage = scheduling.StageProposal(f"calibration:{kind}:{block_start}:attempt:{attempt}", 1.0, "cpu",
        original.target_revision, "original-alias", None, False, None, "calibration", 30,
        original.resources, True, original.digest, "calibration", False, (), False)
    return prep.CalibrationPreparationRequest(original, kind, ep.ExperimentPlan.from_dict(plan),
                                             prompts, tuple(membership), stage, attempt)


@pytest.mark.parametrize("kind", ["aa", "neutral"])
def test_original_request_native_process_membership_round_trip(tmp_path, kind):
    original = request(tmp_path, kind=kind)
    restored = prep.CalibrationPreparationRequest.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert len(restored.plan.expected_units) == 2 * len(restored.block_membership)
    assert len({unit.process_id for unit in restored.plan.expected_units}) == 4
    assert restored.plan.schema == ep.PLAN_SCHEMA_V2
    assert restored.plan.changed_factors == ()
    assert restored.plan.intended_use == "explore"


def test_wrong_original_order_and_reused_material_refuse(tmp_path):
    raw = request(tmp_path).to_dict()
    rows = raw["plan"]["expected_units"]
    rows[0]["order_index"], rows[1]["order_index"] = rows[1]["order_index"], rows[0]["order_index"]
    with pytest.raises(prep.PreparationRefused, match="original order differs"):
        prep.CalibrationPreparationRequest.from_dict(raw)
    raw = request(tmp_path).to_dict()
    raw["block_membership"][1]["material_unit_id"] = raw["block_membership"][0]["material_unit_id"]
    with pytest.raises(prep.PreparationRefused, match="material unit is reused"):
        prep.CalibrationPreparationRequest.from_dict(raw)


def test_no_prompt_as_independent_process_rep_and_wrong_budget_refuse(tmp_path):
    original = request(tmp_path)
    raw = original.to_dict()
    raw["plan"]["expected_units"][1]["process_id"] = raw["plan"]["expected_units"][0]["process_id"]
    with pytest.raises(ep.PlanValidationError, match="reuses process_id"):
        prep.CalibrationPreparationRequest.from_dict(raw)
    with pytest.raises(prep.PreparationRefused, match="resource budget"):
        replace(original, stage_proposal=replace(original.stage_proposal,
                                                estimated_duration_seconds=100))


def test_calibration_may_not_claim_strict_search_or_adopt_changed_prompt(tmp_path):
    raw = request(tmp_path).to_dict()
    raw["plan"]["intended_use"] = "rank"
    with pytest.raises(prep.PreparationRefused, match="original raw frame"):
        prep.CalibrationPreparationRequest.from_dict(raw)
    original = request(tmp_path)
    with pytest.raises(prep.PreparationRefused, match="prompt manifest differs"):
        replace(original, declaration=replace(original.declaration,
                                               prompt_manifest_digest="f" * 64))


def retry_requests(tmp_path):
    return tuple(request(tmp_path, block_start=block, blocks=1, attempt=attempt, max_attempts=2)
                 for block in range(2) for attempt in range(2))


def test_retry_membership_is_preissued_bounded_reversed_and_process_independent(tmp_path):
    requests = retry_requests(tmp_path)
    prep.validate_pool_membership(requests)
    for original, retry in (requests[:2], requests[2:]):
        assert original.chunk_identity == retry.chunk_identity
        assert original.logical_membership == retry.logical_membership
        assert original.plan.expected_units[0].arm != retry.plan.expected_units[0].arm
        assert not ({unit.process_id for unit in original.plan.expected_units}
                    & {unit.process_id for unit in retry.plan.expected_units})
    with pytest.raises(prep.PreparationRefused, match="prospectively"):
        prep.validate_pool_membership(requests[:-1])
    with pytest.raises(prep.PreparationRefused, match="isolate one independent pair"):
        request(tmp_path, max_attempts=2)
    with pytest.raises(prep.PreparationRefused, match="outside the original"):
        replace(requests[0], attempt_ordinal=2)


def test_retry_disposition_retains_valid_blocks_and_exhausts_without_reseed(tmp_path):
    first, retry_first, second, retry_second = requests = retry_requests(tmp_path)
    empty = prep.preparation_disposition(requests, {})
    assert empty["pending"] == (first.digest, second.digest)
    settled = {first.digest: {"outcome": "calibration"}, second.digest: {"outcome": "invalid"}}
    disposition = prep.preparation_disposition(requests, settled)
    assert disposition["pending"] == (retry_second.digest,)
    assert disposition["collected"] == (first.digest,)
    assert retry_first.digest not in disposition["pending"]
    settled[retry_second.digest] = {"outcome": "failed"}
    exhausted = prep.preparation_disposition(requests, settled)
    assert exhausted["pending"] == ()
    assert exhausted["exhausted"] == (second.chunk_identity,)
    assert exhausted["collected"] == (first.digest,)
    settled[retry_first.digest] = {"outcome": "calibration"}
    with pytest.raises(prep.PreparationRefused, match="without an eligible predecessor"):
        prep.preparation_disposition(requests, settled)


def test_global_original_pool_rejects_retry_budget_change_and_reused_process(tmp_path):
    requests = list(retry_requests(tmp_path))
    requests[1] = replace(requests[1], stage_proposal=replace(
        requests[1].stage_proposal, estimated_duration_seconds=29))
    with pytest.raises(prep.PreparationRefused, match="original scheduling budget"):
        prep.validate_pool_membership(tuple(requests))
    requests = list(retry_requests(tmp_path))
    raw = requests[1].to_dict()
    raw["plan"]["expected_units"][0]["process_id"] = requests[0].plan.expected_units[0].process_id
    requests[1] = prep.CalibrationPreparationRequest.from_dict(raw)
    with pytest.raises(prep.PreparationRefused, match="process is reused"):
        prep.validate_pool_membership(tuple(requests))


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, float("inf")])
def test_retry_policy_requires_explicit_finite_attempt_count(maximum):
    with pytest.raises(prep.PreparationRefused, match="finite attempts"):
        prep.PreparationRetryPolicy(maximum, ("failed", "contaminated"))


def test_request_count_and_byte_bounds_precede_any_parser_work(monkeypatch):
    def must_not_parse(_value):
        pytest.fail("individual request parsed before configuration budget check")
    monkeypatch.setattr(prep.CalibrationPreparationRequest, "from_dict", must_not_parse)
    with pytest.raises(prep.PreparationRefused, match="count exceeds"):
        prep.bounded_requests([{}, {}], max_requests=1)
    with pytest.raises(prep.PreparationRefused, match="bytes exceed"):
        prep.bounded_requests([{"oversize": "x" * prep.MAX_CONFIGURATION_BYTES}], max_requests=1)


@pytest.mark.parametrize("case", [
    None, "metric", "metric_direction", "prompt_manifest", "max_stage_seconds",
    "teardown_seconds", "instrument_id", "missing_execution", "extra_manifest_prompt",
    "unit_prompt_count", "unknown_unit_prompt",
])
def test_public_scheduler_materializes_true_aa_v3_without_schema_upgrade(tmp_path, case):
    from . import unified_driver as driver, unified_worker as worker
    from .test_unified_driver_plan_versions import _stack
    with _stack(tmp_path, version=2) as (original_driver, engine, _plan, _pair, _loaded):
        original = request(tmp_path)
        target = next(iter(original_driver.runtime_anchors.recipes))
        anchor = original_driver.runtime_anchors.recipes[target]
        stats = original.declaration.statistics
        stats = replace(stats, commitment=replace(stats.commitment,
                            campaign_id=original_driver.resolved.campaign_id),
                        controls=replace(stats.controls, calibration_block_count=2))
        declared = replace(original.declaration,
            campaign_id=original_driver.resolved.campaign_id, target_revision=target,
            aa_pair=prep.PreparationArmPair("aa", anchor, anchor, None),
            resources=engine.config.capacity,
            statistics=stats, frame={**dict(original.declaration.frame),
                "quant": original_driver.profiles[target].quant})
        arm = ps.arm_identity(anchor.template, anchor,
                             loaded_instrument=prep._plain(original.plan.loaded_instrument))
        plan = ep.ExperimentPlan.from_dict({**original.plan.to_dict(),
            "campaign_id": declared.campaign_id, "target_revision": target,
            "anchor_identity": arm, "candidate_identity": arm})
        stage = replace(original.stage_proposal, target_revision=target,
                        eligibility_ref=declared.digest, frontier_id=target,
                        production_frontier=True, estimated_claims=declared.resources,
                        full_region=True)
        original = replace(original, declaration=declared, plan=plan, stage_proposal=stage)
        if case in {"metric", "metric_direction"}:
            frame = {**prep._plain(declared.frame), case: (
                "foreign_metric" if case == "metric" else "lower_better")}
            declared = replace(declared, frame=frame)
            plan = ep.ExperimentPlan.from_dict({**plan.to_dict(),
                case: "foreign_metric" if case == "metric" else "lower"})
            original = replace(original, declaration=declared, plan=plan,
                stage_proposal=replace(stage, eligibility_ref=declared.digest))
        elif case in {"extra_manifest_prompt", "unit_prompt_count"}:
            manifest = original.prompts.to_dict()
            manifest.pop("digest")
            manifest["prompts"].append({**manifest["prompts"][0], "prompt_id": "p2"})
            prompts = ps.FrozenPromptManifest.from_dict({
                **manifest, "digest": prep._digest(manifest)})
            declared = replace(declared, prompt_manifest_digest=prep._digest(prompts.to_dict()))
            if case == "unit_prompt_count":
                row = plan.to_dict()
                for unit in row["expected_units"]:
                    unit["expected_prompt_ids"] = ["p1", "p2"]
                plan = ep.ExperimentPlan.from_dict(row)
            original = replace(original, declaration=declared, plan=plan, prompts=prompts,
                stage_proposal=replace(stage, eligibility_ref=declared.digest))
        elif case == "unknown_unit_prompt":
            row = plan.to_dict()
            row["expected_units"][0]["expected_prompt_ids"] = ["unknown"]
            original = replace(original, plan=ep.ExperimentPlan.from_dict(row))
        execution = driver.ExecutionInput(target, original.prompts,
            declared.max_stage_seconds, declared.teardown_seconds,
            original.plan.loaded_instrument["identity_sha256"])
        if case in {"max_stage_seconds", "teardown_seconds"}:
            execution = replace(execution, **{case: getattr(execution, case) + 1})
        elif case == "instrument_id":
            execution = replace(execution, instrument_id="f" * 64)
        elif case == "prompt_manifest":
            manifest = original.prompts.to_dict()
            manifest.pop("digest")
            manifest["version"] += "-foreign"
            execution = replace(execution, prompt_manifest=ps.FrozenPromptManifest.from_dict(
                {**manifest, "digest": prep._digest(manifest)}))
        kwargs = dict(
            resolved_campaign=original_driver.resolved, controller=original_driver.controller,
            scheduler_engine=engine, profiles=original_driver.profiles,
            evidence_index=original_driver.evidence, runtime_anchors=original_driver.runtime_anchors,
            runtime_dimensions={}, experiment_plans={}, profile_requests={}, actor_identities={},
            native_artifact_sink_ref=original_driver.sink_ref,
            execution_inputs={} if case == "missing_execution" else {target: execution},
            calibration_requests=(original,), executable_work_kinds={driver.CALIBRATION_WORK_KIND})
        if case not in {None, "extra_manifest_prompt"}:
            with pytest.raises(driver.DriverRefused, match="calibration"):
                driver.UnifiedCampaignDriver(**kwargs)
            return
        instance = driver.UnifiedCampaignDriver(**kwargs)
        outcome = instance.tick(now=1.0)
        assert outcome.status == "intent_recorded", outcome.to_dict()
        assert instance.issued_work_kind(outcome) == driver.CALIBRATION_WORK_KIND
        prepared = instance.materialize_calibration(outcome)
        assert prepared.schema == worker.PREPARED_SCHEMA_V3
        assert prepared.native_observed is True
        assert prepared.plan.to_dict() == original.plan.to_dict()
        assert prepared.runtime_pair.anchor == prepared.runtime_pair.candidate
        assert prepared.dispatch["execution_authorized"] is False
        assert prepared.plan.changed_factors == ()
        assert worker.PreparedPlannedServingStage.from_dict(prepared.to_dict()) == prepared
        with pytest.raises(driver.DriverRefused, match="not a runtime comparison"):
            instance.materialize_runtime(outcome)
        # A v3 payload cannot enter old parsers or silently take legacy completion.
        body = prepared.body()
        body["schema"] = worker.PREPARED_SCHEMA_V2
        with pytest.raises(worker.WorkerBridgeRefused, match="typed input is invalid"):
            worker.PreparedPlannedServingStage.from_dict({**body, "prepared_digest": worker._digest(body)})
        instance.preparation_owner.close()
