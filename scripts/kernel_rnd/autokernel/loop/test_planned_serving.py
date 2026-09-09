"""Pure planned-serving tests using fake admission and measurement consumers."""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager

import pytest

from .. import schemas
from . import experiment_plan as ep
from . import planned_serving as ps
from . import serving
from .test_experiment_plan import plan_dict
from .test_experiment_plan import _v2_plan_dict
from .test_resolved_recipe import _policy, _resolve


def _prompt(prompt_id: str, text: str, recipe: serving.Recipe) -> dict:
    body = {"prompt": text, "n_predict": recipe.n_predict,
            "temperature": recipe.temperature, "top_p": recipe.top_p,
            "top_k": recipe.top_k, "cache_prompt": False}
    import hashlib
    import json
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()
    return {"prompt_id": prompt_id, **body, "request_digest": digest}


def _prompts(recipe: serving.Recipe) -> ps.FrozenPromptManifest:
    body = {"schema": ps.PROMPT_SCHEMA, "version": "prompts-v1",
            "prompts": [_prompt("p1", "first exact prompt", recipe),
                        _prompt("p2", "second exact prompt", recipe)]}
    return ps.FrozenPromptManifest.from_dict({**body, "digest": schemas.content_hash(body)})


def _recipes(*, cpu=True):
    base = serving.Recipe(name="anchor", model="/m", np=2,
                          device="none" if cpu else "ROCm0", ngl=0 if cpu else 99)
    policy = _policy("KNOB")
    anchor = _resolve(base, backend="cpu" if cpu else "gpu", policy=policy)
    candidate_template = base.with_env(name="candidate", KNOB="1")
    candidate = _resolve(candidate_template, backend="cpu" if cpu else "gpu", policy=policy)
    return base, candidate_template, anchor, candidate


def _plan(anchor_template, candidate_template, anchor, candidate):
    body = plan_dict(n=1, instrument="serving")
    body["metric"] = "aggregate_tok_s"
    body["anchor_identity"] = ps.arm_identity(anchor_template, anchor)
    body["candidate_identity"] = ps.arm_identity(candidate_template, candidate)
    body["required_witnesses"] = ["identity", "teardown"]
    return ep.ExperimentPlan.from_dict(body)


def _v2_plan(anchor_template, candidate_template, anchor, candidate):
    body = _v2_plan_dict()
    body["metric"] = "aggregate_tok_s"
    instrument = body["loaded_instrument"]
    body["anchor_identity"] = ps.arm_identity(
        anchor_template, anchor, loaded_instrument=instrument)
    body["candidate_identity"] = ps.arm_identity(
        candidate_template, candidate, loaded_instrument=instrument)
    body["required_witnesses"] = ["identity", "teardown"]
    body["expected_units"] = body["expected_units"][:2]
    body["stopping"]["n_per_arm"] = 1
    return ep.ExperimentPlan.from_dict(body)


class Provider:
    def __init__(self, *, deadline=100.0, terminal=True, lineage="lineage-1"):
        self.deadline = deadline
        self.terminal = terminal
        self.lineage = lineage
        self.admitted = []

    def admit(self, plan_digest, unit, stages):
        self.admitted.append((unit.unit_id, stages))
        return ps.StageFence(f"fence-{unit.unit_id}", unit.unit_id, unit.process_id,
                             self.lineage, "grant-1", "container-1", "monotonic",
                             self.deadline)

    @contextmanager
    def guard(self, fence):
        yield ps.ExecutionGuard(fence.fence_id, fence.unit_id,
                                fence.process_generation_id, fence.lineage_id,
                                fence.grant_id, fence.container_id, True, True)

    def complete(self, fence, observation):
        return ps.StageCompletion(fence.fence_id, self.terminal,
                                  {"identity": ep.Witness("pass", f"id:{fence.unit_id}"),
                                   "teardown": ep.Witness("pass", f"td:{fence.unit_id}")},
                                  "clean" if self.terminal else "rejected",
                                  None if self.terminal else "drained")


def _measure(calls, *, partial=False, terminal=True):
    def measure(template, build_dir, port, **kwargs):
        requests = kwargs["frozen_requests"]
        calls.append((template.name, build_dir, port, requests,
                      kwargs["resolved_recipe"].execution_digest))
        import hashlib
        rows = [{"phase": "measurement", "slot_index": index,
                 "prompt_id": prompt_id, "request_sha256": hashlib.sha256(body).hexdigest(),
                 "predicted_n": template.n_predict, "predicted_per_second": 10.0,
                 "terminal": terminal, "error": None}
                for index, (prompt_id, body) in enumerate(requests)]
        if partial:
            rows.pop()
        kwargs["observation"].append({
            "schema": "epyc.autokernel.serving_observation.v1", "process_pid": 123,
            "requests": rows, "residency": {"status": "not_applicable"},
            "teardown": "terminated", "failure": None})
        return 20.0
    return measure


def test_cpu_runtime_recipes_share_binary_consume_exact_order_and_prompt_bytes():
    at, ct, anchor, candidate = _recipes()
    assert anchor.executable.sha256 == candidate.executable.sha256
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    calls, artifacts = [], []
    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
        artifact_sink=artifacts.append, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=_measure(calls))
    assert [item[0] for item in calls] == [unit.arm for unit in plan.expected_units]
    assert [raw.unit_id for raw in result.raw_units] == [
        unit.unit_id for unit in plan.expected_units]
    assert result.admissible_view.complete and result.use_status == "policy_undefined"
    assert len(artifacts) == 2 * len(plan.expected_units)
    assert calls[0][3][0] == ("p1", prompts.prompts[0].body)


def test_v2_requires_factory_and_carries_one_reference_per_actual_unit():
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _v2_plan(at, ct, anchor, candidate), _prompts(at)
    kwargs = dict(plan=plan, anchor_template=at, candidate_template=ct,
                  anchor_recipe=anchor, candidate_recipe=candidate, prompts=prompts,
                  stage_provider=Provider(), artifact_sink=lambda _: None,
                  lineage_id="lineage-1", clock=lambda: 1.0,
                  measure=_measure([]))
    with pytest.raises(ps.TrustedStageProviderRequired, match="observation authority"):
        ps.run_planned_comparison(**kwargs)

    class Factory:
        def __init__(self):
            self.created = []

        def create(self, *, unit, fence, recipe):
            session = object()
            self.created.append((unit.unit_id, fence.fence_id, recipe.execution_digest, session))
            return session

        def finish_reference(self, *, unit, session):
            assert self.created[-1][3] is session
            body = {"schema": "epyc.autokernel.lifecycle_observation_reference.v1",
                    "observation_id": f"obs-{unit.unit_id}", "unit_id": unit.unit_id,
                    "process_generation_id": unit.process_id, "fence_id": f"fence-{unit.unit_id}",
                    "active_claim_ref": "claim:1", "target_pid": 101,
                    "target_start_ticks": 100, "descendant_binding_ref": "descendant:1",
                    "worker_id": "worker-1",
                    "worker_generation": 1, "grant_id": "grant-1", "grant_generation": 1,
                    "container_id": "container-1", "instrument_identity_sha256": "c" * 64,
                    "observation_content_sha256": "e" * 64, "shutdown_status": "resolved",
                    "successor_permitted": True,
                    "artifact": {"locator": f"{unit.unit_id}.json", "sha256": "f" * 64,
                                 "verified": True}}
            return {**body, "reference_digest": schemas.content_hash(body)}

    factory, artifacts = Factory(), []
    kwargs["artifact_sink"] = artifacts.append
    result = ps.run_planned_comparison(**kwargs, observation_session_factory=factory)
    assert result.schema == ps.RUN_SCHEMA_V2
    assert len(result.lifecycle_observation_references) == 2
    assert all(row["schema"] == ps.ARTIFACT_SCHEMA_V2 for row in artifacts)
    assert [row["kind"] for row in artifacts] == [
        "native_observation", "completed_attempt", "native_observation", "completed_attempt"]


def test_v2_unresolved_observer_fences_successor_unit():
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _v2_plan(at, ct, anchor, candidate), _prompts(at)

    class UnresolvedFactory:
        def create(self, *, unit, fence, recipe):
            return unit.unit_id

        def finish_reference(self, *, unit, session):
            body = {"schema": "epyc.autokernel.lifecycle_observation_reference.v1",
                    "observation_id": f"obs-{unit.unit_id}", "unit_id": unit.unit_id,
                    "process_generation_id": unit.process_id, "fence_id": f"fence-{unit.unit_id}",
                    "active_claim_ref": "claim:1", "target_pid": 101,
                    "target_start_ticks": 100, "descendant_binding_ref": "descendant:1",
                    "worker_id": "worker-1",
                    "worker_generation": 1, "grant_id": "grant-1", "grant_generation": 1,
                    "container_id": "container-1", "instrument_identity_sha256": "c" * 64,
                    "observation_content_sha256": "e" * 64, "shutdown_status": "unresolved",
                    "successor_permitted": False,
                    "artifact": {"locator": f"{unit.unit_id}.json", "sha256": "f" * 64,
                                 "verified": True}}
            return {**body, "reference_digest": schemas.content_hash(body)}

    calls = []
    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
        artifact_sink=lambda _: None, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=_measure(calls), observation_session_factory=UnresolvedFactory())
    assert len(calls) == 1
    assert not result.execution_complete
    assert result.paused_reason == "lifecycle observer shutdown is unresolved"
    assert len(result.lifecycle_observation_references) == 1


def test_default_refuses_and_identity_mismatch_precedes_provider_or_measurement():
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    with pytest.raises(ps.TrustedStageProviderRequired):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts, stage_provider=None,
            artifact_sink=lambda _: None, lineage_id="lineage-1")
    bad = ep.ExperimentPlan.from_dict({**plan.to_dict(),
        "candidate_identity": {**dict(plan.candidate_identity), "model_digest": "0" * 64}})
    provider, calls = Provider(), []
    with pytest.raises(ps.PlannedServingError, match="identity"):
        ps.run_planned_comparison(
            bad, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts, stage_provider=provider,
            artifact_sink=lambda _: None, lineage_id="lineage-1", measure=_measure(calls))
    assert not provider.admitted and not calls


@pytest.mark.parametrize("partial,terminal", [(True, True), (False, False)])
def test_partial_or_nonterminal_request_is_raw_but_not_admissible(partial, terminal):
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    artifacts = []
    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
        artifact_sink=artifacts.append, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=_measure([], partial=partial, terminal=terminal))
    assert len(result.raw_units) == 0
    assert not result.admissible_view.complete and not result.execution_complete
    assert all(not row.terminal for row in result.raw_units)


def test_stale_fence_and_missing_containment_refuse_before_measurement():
    at, ct, anchor, candidate = _recipes(cpu=False)
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    calls = []
    with pytest.raises(ps.PlannedServingError, match="expired"):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(deadline=0.0),
            artifact_sink=lambda _: None, lineage_id="lineage-1", clock=lambda: 1.0,
            measure=_measure(calls))
    assert calls == []

    class Unsafe(Provider):
        @contextmanager
        def guard(self, fence):
            yield replace(ps.ExecutionGuard(
                fence.fence_id, fence.unit_id, fence.process_generation_id,
                fence.lineage_id, fence.grant_id, fence.container_id, True, True),
                owned_descendants=False)

    with pytest.raises(ps.UnsupportedContainment):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts, stage_provider=Unsafe(),
            artifact_sink=lambda _: None, lineage_id="lineage-1", clock=lambda: 1.0,
            measure=_measure(calls))


def test_completion_callback_failure_preserves_native_observation_first():
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    artifacts = []

    class BrokenCompletion(Provider):
        def complete(self, fence, observation):
            raise RuntimeError("lost adapter")

    with pytest.raises(ps.PlannedServingError, match="completion provider"):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts,
            stage_provider=BrokenCompletion(), artifact_sink=artifacts.append,
            lineage_id="lineage-1", clock=lambda: 1.0, measure=_measure([]))
    assert len(artifacts) == 1 and artifacts[0]["kind"] == "native_observation"


def test_provider_pause_returns_coherent_partial_run_without_successor():
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)

    class PauseSecond(Provider):
        def admit(self, plan_digest, unit, stages):
            if self.admitted:
                raise ps.StagePaused("grant revoked")
            return super().admit(plan_digest, unit, stages)

    provider, calls, artifacts = PauseSecond(), [], []
    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=provider,
        artifact_sink=artifacts.append, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=_measure(calls))
    assert result.paused_reason == "grant revoked"
    assert len(calls) == 1 and len(result.raw_units) == 1
    assert not result.admissible_view.complete


def test_completed_unit_continuation_needs_permission_verifier_and_new_lineage():
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    first = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
        artifact_sink=lambda _: None, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=_measure([]))
    with pytest.raises(ps.PlannedServingError, match="does not permit"):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
            artifact_sink=lambda _: None, lineage_id="lineage-2",
            previous_raws=first.raw_units, previous_lineage_id="lineage-1")
    allowed = ep.ExperimentPlan.from_dict({**plan.to_dict(), "continuation_allowed": True})
    # Plan digest changed, so old units cannot be laundered into the new plan.
    with pytest.raises(ps.PlannedServingError, match="eligible"):
        ps.run_planned_comparison(
            allowed, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts,
            stage_provider=Provider(lineage="lineage-2"), artifact_sink=lambda _: None,
            lineage_id="lineage-2", previous_raws=first.raw_units,
            previous_lineage_id="lineage-1", continuation_verifier=lambda *_: True)


def test_allowed_completed_units_reuse_exact_plan_with_new_lineage_only():
    at, ct, anchor, candidate = _recipes()
    plan = _plan(at, ct, anchor, candidate)
    plan = ep.ExperimentPlan.from_dict({**plan.to_dict(), "continuation_allowed": True})
    prompts = _prompts(at)
    first = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
        artifact_sink=lambda _: None, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=_measure([]))
    artifacts, calls, verified_lineages = [], [], []
    resumed = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts,
        stage_provider=Provider(lineage="lineage-2"), artifact_sink=artifacts.append,
        lineage_id="lineage-2", previous_raws=first.raw_units,
        previous_lineage_id="lineage-1",
        continuation_verifier=lambda *args: verified_lineages.append(args[3]) or True,
        measure=_measure(calls))
    assert not calls and resumed.admissible_view.complete
    assert verified_lineages == ["lineage-1", "lineage-1"]
    assert all(item["kind"] == "continued_unit" for item in artifacts)


def test_continuation_rejects_foreign_and_nonprefix_units_before_admission():
    at, ct, anchor, candidate = _recipes()
    plan = _plan(at, ct, anchor, candidate)
    plan = ep.ExperimentPlan.from_dict({**plan.to_dict(), "continuation_allowed": True})
    prompts = _prompts(at)
    first = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
        artifact_sink=lambda _: None, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=_measure([]))
    provider = Provider(lineage="lineage-2")
    foreign = ep.RawUnit.from_dict({**first.raw_units[0].to_dict(), "unit_id": "foreign"})
    with pytest.raises(ps.PlannedServingError, match="foreign"):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts, stage_provider=provider,
            artifact_sink=lambda _: None, lineage_id="lineage-2", previous_raws=(foreign,),
            previous_lineage_id="lineage-1", continuation_verifier=lambda *_: True)
    assert not provider.admitted
    with pytest.raises(ps.PlannedServingError, match="prefix"):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts, stage_provider=provider,
            artifact_sink=lambda _: None, lineage_id="lineage-2",
            previous_raws=(first.raw_units[1],), previous_lineage_id="lineage-1",
            continuation_verifier=lambda *_: True)
    assert not provider.admitted


def test_every_prompt_count_is_prevalidated_before_first_admission():
    at, ct, anchor, candidate = _recipes()
    plan = _plan(at, ct, anchor, candidate)
    body = plan.to_dict()
    body["expected_units"][1]["expected_prompt_ids"] = ["p1"]
    plan = ep.ExperimentPlan.from_dict(body)
    provider = Provider()
    with pytest.raises(ps.PlannedServingError, match="count"):
        ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=_prompts(at), stage_provider=provider,
            artifact_sink=lambda _: None, lineage_id="lineage-1")
    assert not provider.admitted


def test_malformed_observation_is_retained_then_typed_as_partial_refusal():
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    artifacts = []

    def malformed(*_, **kwargs):
        kwargs["observation"].extend([{"requests": []}, "bad-observation"])
        return 20.0

    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=Provider(),
        artifact_sink=artifacts.append, lineage_id="lineage-1", clock=lambda: 1.0,
        measure=malformed)
    assert not result.raw_units and result.paused_reason
    native = artifacts[0]
    assert native["observations"][0]["requests"] == ()
    assert native["observations"][1]["malformed_observation_type"] == "str"
