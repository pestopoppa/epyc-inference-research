"""Execute a frozen serving ExperimentPlan through the existing launcher consumer.

This runner has no admission authority.  A trusted stage provider and explicit raw-artifact
sink are mandatory; ordinary serialized labels cannot substitute for either boundary.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Any, Protocol

from .. import schemas
from . import experiment_plan as ep
from . import resolved_recipe as rr
from . import serving


PROMPT_SCHEMA = "epyc.autokernel.frozen_prompt_manifest.v1"
ARTIFACT_SCHEMA = "epyc.autokernel.planned_serving_artifact.v1"
RUN_SCHEMA = "epyc.autokernel.planned_serving_run.v1"
STAGES = ("setup", "load", "placement", "readback", "warmup", "request", "teardown")


class PlannedServingError(RuntimeError):
    """Frozen plan, identity, prompt, continuation, or consumer contract refused."""


class TrustedStageProviderRequired(PlannedServingError):
    """Execution was requested without an injected trusted stage-fence provider."""


class UnsupportedContainment(PlannedServingError):
    """The provider cannot prove bounded deadline and owned-descendant containment."""


class StagePaused(PlannedServingError):
    """Trusted provider revoked or paused admission before the next unit."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise PlannedServingError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlannedServingError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True)
class FrozenPrompt:
    prompt_id: str
    prompt: str
    n_predict: int
    temperature: float
    top_p: float
    top_k: int
    cache_prompt: bool
    request_digest: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenPrompt":
        fields = {"prompt_id", "prompt", "n_predict", "temperature", "top_p", "top_k",
                  "cache_prompt", "request_digest"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise PlannedServingError("frozen prompt has missing or unknown fields")
        if isinstance(value["n_predict"], bool) or not isinstance(value["n_predict"], int) \
                or value["n_predict"] <= 0:
            raise PlannedServingError("frozen prompt n_predict must be positive integer")
        if isinstance(value["top_k"], bool) or not isinstance(value["top_k"], int) \
                or value["top_k"] < 0:
            raise PlannedServingError("frozen prompt top_k must be non-negative integer")
        numeric = (value["temperature"], value["top_p"])
        if any(isinstance(item, bool) or not isinstance(item, (int, float))
               or not math.isfinite(float(item)) for item in numeric):
            raise PlannedServingError("frozen prompt sampling values must be finite")
        if value["cache_prompt"] is not False:
            raise PlannedServingError("planned independent requests require cache_prompt=false")
        result = cls(_text(value["prompt_id"], "prompt_id"),
                     _text(value["prompt"], "prompt"), value["n_predict"],
                     float(value["temperature"]), float(value["top_p"]), value["top_k"],
                     False, _sha(value["request_digest"], "request_digest"))
        if result.request_digest != hashlib.sha256(result.body).hexdigest():
            raise PlannedServingError("frozen prompt request digest mismatch")
        return result

    @property
    def body(self) -> bytes:
        return _canonical({"prompt": self.prompt, "n_predict": self.n_predict,
                           "temperature": self.temperature, "top_p": self.top_p,
                           "top_k": self.top_k, "cache_prompt": self.cache_prompt})

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "prompt": self.prompt,
                "n_predict": self.n_predict, "temperature": self.temperature,
                "top_p": self.top_p, "top_k": self.top_k,
                "cache_prompt": self.cache_prompt, "request_digest": self.request_digest}


@dataclass(frozen=True)
class FrozenPromptManifest:
    version: str
    prompts: tuple[FrozenPrompt, ...]
    digest: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenPromptManifest":
        if not isinstance(value, Mapping) or set(value) != {"schema", "version", "prompts",
                                                              "digest"} \
                or value["schema"] != PROMPT_SCHEMA:
            raise PlannedServingError("unsupported or malformed frozen prompt manifest")
        if not isinstance(value["prompts"], Sequence) or isinstance(value["prompts"], (str, bytes)):
            raise PlannedServingError("frozen prompt manifest prompts must be an array")
        prompts = tuple(FrozenPrompt.from_dict(item) for item in value["prompts"])
        if not prompts or len({item.prompt_id for item in prompts}) != len(prompts):
            raise PlannedServingError("frozen prompt IDs must be nonempty and unique")
        version = _text(value["version"], "prompt manifest version")
        body = {"schema": PROMPT_SCHEMA, "version": version,
                "prompts": [item.to_dict() for item in prompts]}
        digest = _sha(value["digest"], "prompt manifest digest")
        if digest != schemas.content_hash(body):
            raise PlannedServingError("frozen prompt manifest digest mismatch")
        return cls(version, prompts, digest)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PROMPT_SCHEMA, "version": self.version,
                "prompts": [item.to_dict() for item in self.prompts], "digest": self.digest}

    def requests(self, prompt_ids: Sequence[str], recipe: serving.Recipe) \
            -> tuple[tuple[str, bytes], ...]:
        by_id = {item.prompt_id: item for item in self.prompts}
        try:
            selected = tuple(by_id[item] for item in prompt_ids)
        except KeyError as exc:
            raise PlannedServingError(f"plan names unknown frozen prompt {exc.args[0]!r}") from exc
        for item in selected:
            if (item.n_predict, item.temperature, item.top_p, item.top_k) != (
                    recipe.n_predict, recipe.temperature, recipe.top_p, recipe.top_k):
                raise PlannedServingError("frozen prompt workload differs from arm recipe")
        return tuple((item.prompt_id, item.body) for item in selected)


@dataclass(frozen=True)
class StageFence:
    fence_id: str
    unit_id: str
    process_generation_id: str
    lineage_id: str
    grant_id: str
    container_id: str
    clock_domain: str
    valid_until: float


@dataclass(frozen=True)
class ExecutionGuard:
    fence_id: str
    unit_id: str
    process_generation_id: str
    lineage_id: str
    grant_id: str
    container_id: str
    deadline_includes_teardown: bool
    owned_descendants: bool


@dataclass(frozen=True)
class StageCompletion:
    fence_id: str
    terminal: bool
    stage_witnesses: Mapping[str, ep.Witness]
    recorded_screen: str
    reason: str | None


class TrustedStageProvider(Protocol):
    def admit(self, plan_digest: str, unit: ep.UnitSpec,
              stages: tuple[str, ...]) -> StageFence: ...

    def complete(self, fence: StageFence, observation: Mapping[str, Any]) \
            -> StageCompletion: ...

    def guard(self, fence: StageFence): ...


ArtifactSink = Callable[[Mapping[str, Any]], None]
Measure = Callable[..., float]
ContinuationVerifier = Callable[
    [ep.RawUnit, ep.ExperimentPlan, FrozenPromptManifest, str], bool]


@dataclass(frozen=True)
class PlannedServingRun:
    schema: str
    plan_digest: str
    prompt_manifest_digest: str
    lineage_id: str
    anchor_identity: Mapping[str, Any]
    candidate_identity: Mapping[str, Any]
    raw_units: tuple[ep.RawUnit, ...]
    admissible_view: ep.AdmissibleUnitView
    use_status: str
    execution_complete: bool
    paused_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "plan_digest": self.plan_digest,
                "prompt_manifest_digest": self.prompt_manifest_digest,
                "lineage_id": self.lineage_id,
                "anchor_identity": dict(self.anchor_identity),
                "candidate_identity": dict(self.candidate_identity),
                "raw_units": [item.to_dict() for item in self.raw_units],
                "admissible_view": self.admissible_view.to_dict(),
                "use_status": self.use_status,
                "execution_complete": self.execution_complete,
                "paused_reason": self.paused_reason}


def _validated_fence(value: Any, spec: ep.UnitSpec, lineage_id: str,
                     clock_domain: str) -> StageFence:
    if not isinstance(value, StageFence):
        raise PlannedServingError("trusted provider did not return a StageFence")
    for field in ("fence_id", "unit_id", "process_generation_id", "lineage_id",
                  "grant_id", "container_id", "clock_domain"):
        _text(getattr(value, field), f"stage fence {field}")
    if isinstance(value.valid_until, bool) or not isinstance(value.valid_until, (int, float)) \
            or not math.isfinite(float(value.valid_until)):
        raise PlannedServingError("stage fence deadline must be finite in provider clock domain")
    if (value.unit_id != spec.unit_id or value.process_generation_id != spec.process_id
            or value.lineage_id != lineage_id or value.clock_domain != clock_domain):
        raise PlannedServingError("trusted provider returned a mismatched stage fence")
    return value


def _validated_guard(value: Any, fence: StageFence) -> ExecutionGuard:
    if not isinstance(value, ExecutionGuard):
        raise UnsupportedContainment("provider did not enter a typed enclosing execution guard")
    identity = (value.fence_id, value.unit_id, value.process_generation_id, value.lineage_id,
                value.grant_id, value.container_id)
    expected = (fence.fence_id, fence.unit_id, fence.process_generation_id, fence.lineage_id,
                fence.grant_id, fence.container_id)
    if identity != expected or type(value.deadline_includes_teardown) is not bool \
            or type(value.owned_descendants) is not bool:
        raise UnsupportedContainment("enclosing execution guard identity is inconsistent")
    if not value.deadline_includes_teardown or not value.owned_descendants:
        raise UnsupportedContainment(
            "enclosing worker lacks bounded stage+teardown or owned-descendant containment")
    return value


def _validated_completion(value: Any, fence: StageFence) -> StageCompletion:
    if not isinstance(value, StageCompletion) or value.fence_id != fence.fence_id \
            or type(value.terminal) is not bool:
        raise PlannedServingError("trusted provider returned mismatched completion")
    if value.recorded_screen not in {"clean", "flagged_but_retained", "rejected"}:
        raise PlannedServingError("trusted provider returned invalid recorded screen")
    if value.reason is not None:
        _text(value.reason, "stage completion reason")
    if (value.recorded_screen == "clean") != (value.reason is None):
        raise PlannedServingError("stage completion screen/reason fields disagree")
    witnesses: dict[str, ep.Witness] = {}
    if not isinstance(value.stage_witnesses, Mapping):
        raise PlannedServingError("stage completion witnesses must be a mapping")
    for name, witness in value.stage_witnesses.items():
        _text(name, "stage witness name")
        try:
            witnesses[name] = ep.Witness.from_dict(witness.to_dict(), f"stage witness {name}")
        except Exception as exc:
            raise PlannedServingError(f"invalid stage witness {name!r}: {exc}") from exc
    return StageCompletion(value.fence_id, value.terminal,
                           MappingProxyType(witnesses), value.recorded_screen, value.reason)


def _dso_digest(recipe: rr.ResolvedRecipe) -> str:
    return schemas.content_hash([{"load_name": item.path.rsplit("/", 1)[-1],
                                  "sha256": item.sha256} for item in recipe.dsos])


def arm_identity(template: serving.Recipe, recipe: rr.ResolvedRecipe) -> dict[str, Any]:
    """Canonical exact identity expected in an ExperimentPlan arm mapping."""
    return {"template_hash": template.recipe_hash,
            "resolved_execution_digest": recipe.execution_digest,
            "resolved_snapshot_digest": recipe.snapshot_digest,
            "workload_digest": schemas.content_hash(recipe.workload.to_dict()),
            "model_digest": recipe.model.sha256,
            "drafter_digest": recipe.drafter.sha256 if recipe.drafter else None,
            "executable_digest": recipe.executable.sha256,
            "dso_set_digest": _dso_digest(recipe)}


def _normalized_plan(plan: ep.ExperimentPlan) -> ep.ExperimentPlan:
    try:
        return ep.ExperimentPlan.from_dict(plan.to_dict())
    except Exception as exc:
        raise PlannedServingError(f"invalid experiment plan: {exc}") from exc


def run_planned_comparison(plan: ep.ExperimentPlan, *,
                           anchor_template: serving.Recipe,
                           candidate_template: serving.Recipe,
                           anchor_recipe: rr.ResolvedRecipe,
                           candidate_recipe: rr.ResolvedRecipe,
                           prompts: FrozenPromptManifest,
                           stage_provider: TrustedStageProvider | None,
                           artifact_sink: ArtifactSink | None,
                           lineage_id: str,
                           clock: Callable[[], float] = time.monotonic,
                           clock_domain: str = "monotonic",
                           measure: Measure = serving._measure_once,
                           previous_raws: Sequence[ep.RawUnit] = (),
                           previous_lineage_id: str | None = None,
                           continuation_verifier: ContinuationVerifier | None = None) \
        -> PlannedServingRun:
    """Run exactly the plan's frozen unit order through fresh `_measure_once` launches."""
    plan = _normalized_plan(plan)
    prompts = FrozenPromptManifest.from_dict(prompts.to_dict())
    lineage_id = _text(lineage_id, "execution lineage")
    clock_domain = _text(clock_domain, "provider clock domain")
    if plan.instrument_class != "serving":
        raise PlannedServingError("planned serving requires instrument_class=serving")
    if stage_provider is None:
        raise TrustedStageProviderRequired("trusted stage-fence provider is not connected")
    if artifact_sink is None:
        raise PlannedServingError("explicit raw artifact sink is required")
    if previous_raws:
        if not plan.continuation_allowed:
            raise PlannedServingError("plan does not permit completed-unit continuation")
        if continuation_verifier is None:
            raise PlannedServingError("trusted continuation verifier is not connected")
        previous_lineage_id = _text(previous_lineage_id, "previous execution lineage")
        if previous_lineage_id == lineage_id:
            raise PlannedServingError("continued execution requires a new lineage")
    arms = {"anchor": (anchor_template, anchor_recipe),
            "candidate": (candidate_template, candidate_recipe)}
    expected_identities = {"anchor": dict(plan.anchor_identity),
                           "candidate": dict(plan.candidate_identity)}
    actual_identities: dict[str, dict[str, Any]] = {}
    for arm, (template, recipe) in arms.items():
        recipe.validate_launch(template, recipe.build_dir, recipe.port)
        actual_identities[arm] = arm_identity(template, recipe)
        if expected_identities[arm] != actual_identities[arm]:
            raise PlannedServingError(f"{arm} resolved identity does not match frozen plan")
        if plan.metric != recipe.workload.metric:
            raise PlannedServingError(f"{arm} workload metric does not match frozen plan")
    requests_by_unit: dict[str, tuple[tuple[str, bytes], ...]] = {}
    for spec in plan.expected_units:
        requests = prompts.requests(spec.expected_prompt_ids, arms[spec.arm][0])
        if len(requests) != arms[spec.arm][0].np:
            raise PlannedServingError(
                f"unit {spec.unit_id!r} frozen request count differs from arm np")
        requests_by_unit[spec.unit_id] = requests

    raws: list[ep.RawUnit] = []
    prior_by_id: dict[str, ep.RawUnit] = {}
    for raw in previous_raws:
        try:
            normalized = ep.RawUnit.from_dict(raw.to_dict())
        except Exception as exc:
            raise PlannedServingError(f"invalid previous raw unit: {exc}") from exc
        if normalized.unit_id in prior_by_id:
            raise PlannedServingError("previous completed units contain duplicate unit IDs")
        prior_by_id[normalized.unit_id] = normalized
    expected_order = [item.unit_id for item in sorted(
        plan.expected_units, key=lambda item: item.order_index)]
    if set(prior_by_id) - set(expected_order):
        raise PlannedServingError("previous completed units contain foreign unit IDs")
    retained_order = [unit_id for unit_id in expected_order if unit_id in prior_by_id]
    if retained_order != expected_order[:len(retained_order)]:
        raise PlannedServingError("continued units must form the frozen completed prefix")
    expected_by_id = {item.unit_id: item for item in plan.expected_units}
    for unit_id in retained_order:
        prior, spec = prior_by_id[unit_id], expected_by_id[unit_id]
        structurally_complete = (prior.plan_digest == plan.digest
                                 and prior.arm == spec.arm
                                 and prior.process_id == spec.process_id
                                 and prior.prompt_ids == spec.expected_prompt_ids
                                 and prior.observed_order_index == spec.order_index
                                 and prior.terminal and prior.recorded_screen != "rejected"
                                 and all(prior.witnesses.get(name) is not None
                                         and prior.witnesses[name].status == "pass"
                                         and prior.witnesses[name].ref is not None
                                         for name in plan.required_witnesses))
        try:
            verified = continuation_verifier(prior, plan, prompts, previous_lineage_id) \
                if continuation_verifier and previous_lineage_id else False
        except Exception as exc:
            raise PlannedServingError("trusted continuation verifier errored") from exc
        if not structurally_complete or verified is not True:
            raise PlannedServingError("previous unit is not eligible for exact continuation")
    paused_reason: str | None = None
    for spec in sorted(plan.expected_units, key=lambda item: item.order_index):
        template, recipe = arms[spec.arm]
        frozen_requests = requests_by_unit[spec.unit_id]
        prior = prior_by_id.get(spec.unit_id)
        if prior is not None:
            reuse_body = {"schema": ARTIFACT_SCHEMA, "kind": "continued_unit",
                          "plan_digest": plan.digest,
                          "unit_id": spec.unit_id, "arm": spec.arm,
                          "process_generation_id": spec.process_id,
                          "lineage_id": lineage_id,
                          "continued_from_lineage_id": previous_lineage_id,
                          "continued_from_artifact_digest": prior.artifact_digest,
                          "prompt_manifest_digest": prompts.digest,
                          "prompt_ids": list(prior.prompt_ids),
                          "comparison_identities": actual_identities,
                          "terminal": True, "value": prior.value,
                          "recorded_screen": prior.recorded_screen,
                          "reason": prior.reason}
            artifact_digest = schemas.content_hash(reuse_body)
            artifact_sink(_freeze(dict(reuse_body, artifact_digest=artifact_digest)))
            raws.append(ep.RawUnit.from_dict({**prior.to_dict(),
                                              "artifact_digest": artifact_digest}))
            continue
        try:
            admitted = stage_provider.admit(plan.digest, spec, STAGES)
        except StagePaused as exc:
            paused_reason = str(exc)
            break
        except Exception as exc:
            raise PlannedServingError("trusted stage admission provider errored") from exc
        fence = _validated_fence(admitted, spec, lineage_id, clock_domain)
        if clock() >= fence.valid_until:
            raise PlannedServingError("stage fence expired before launch")
        observations: list[dict[str, Any]] = []
        value: float | None = None
        error: str | None = None
        guard_factory = getattr(stage_provider, "guard", None)
        if not callable(guard_factory):
            raise UnsupportedContainment("provider has no enclosing execution guard")
        try:
            with guard_factory(fence) as guard:
                _validated_guard(guard, fence)
                value = float(measure(template, recipe.build_dir, recipe.port,
                                      resolved_recipe=recipe,
                                      frozen_requests=frozen_requests,
                                      observation=observations))
        except Exception as exc:
            if isinstance(exc, UnsupportedContainment):
                raise
            error = f"{type(exc).__name__}: {exc}"
        if value is not None and not math.isfinite(value):
            error = "measurement returned a non-finite value"
            value = None
        retained_observations = [dict(item) if isinstance(item, Mapping) else {
            "malformed_observation_type": type(item).__name__} for item in observations]
        if observations and isinstance(observations[-1], Mapping):
            observation = dict(observations[-1])
        else:
            if observations:
                error = error or "measurement returned a non-mapping observation"
            observation = {"schema": "epyc.autokernel.serving_observation.v1",
                           "process_pid": None, "requests": [], "residency": {},
                           "teardown": "unknown", "failure": error}
        if not isinstance(observation.get("requests"), list):
            error = error or "measurement observation requests must be a list"
            observation = dict(observation, requests=[])
        native_body = {"schema": ARTIFACT_SCHEMA, "kind": "native_observation",
                       "plan_digest": plan.digest, "unit_id": spec.unit_id,
                       "arm": spec.arm, "process_generation_id": spec.process_id,
                       "lineage_id": lineage_id, "fence_id": fence.fence_id,
                       "grant_id": fence.grant_id, "container_id": fence.container_id,
                       "prompt_manifest_digest": prompts.digest,
                       "comparison_identities": actual_identities,
                       "observations": retained_observations,
                       "selected_observation": observation, "value": value, "error": error}
        native_digest = schemas.content_hash(native_body)
        artifact_sink(_freeze(dict(native_body, artifact_digest=native_digest)))
        try:
            completion = _validated_completion(stage_provider.complete(fence, observation), fence)
        except Exception as exc:
            if isinstance(exc, PlannedServingError):
                raise
            raise PlannedServingError("trusted stage completion provider errored") from exc
        expired = clock() > fence.valid_until
        all_request_rows = observation.get("requests") if isinstance(observation, Mapping) else []
        request_rows = [row for row in all_request_rows
                        if isinstance(row, Mapping) and row.get("phase") == "measurement"]
        prompt_ids = tuple(row.get("prompt_id") for row in request_rows
                           if isinstance(row, Mapping) and isinstance(row.get("prompt_id"), str))
        expected_hashes = tuple(hashlib.sha256(body).hexdigest()
                                for _, body in frozen_requests)
        exact_requests = (prompt_ids == spec.expected_prompt_ids
                          and tuple(row.get("request_sha256") for row in request_rows)
                          == expected_hashes
                          and all(row.get("terminal") is True and row.get("error") is None
                                  for row in request_rows))
        terminal = bool(completion.terminal and not expired and error is None and value is not None
                        and exact_requests and observation.get("teardown") in {"terminated", "killed"})
        reason = error
        if expired:
            reason = "stage fence expired during bounded unit"
        elif not exact_requests:
            reason = "request membership or terminal completion mismatch"
        elif not completion.terminal:
            reason = completion.reason or "stage completion is not terminal"
        original_screen = completion.recorded_screen
        screen = original_screen if terminal else "rejected"
        if screen != "clean" and reason is None:
            reason = completion.reason or "recorded screen was not clean"
        witnesses = dict(completion.stage_witnesses)
        artifact_body = {"schema": ARTIFACT_SCHEMA, "kind": "completed_attempt",
                         "plan_digest": plan.digest,
                         "unit_id": spec.unit_id, "arm": spec.arm,
                         "process_generation_id": spec.process_id,
                         "lineage_id": lineage_id, "fence_id": fence.fence_id,
                         "prompt_manifest_digest": prompts.digest,
                         "prompt_ids": list(prompt_ids),
                         "comparison_identities": actual_identities,
                         "native_observation_digest": native_digest,
                         "stage_witnesses": {key: item.to_dict()
                                             for key, item in witnesses.items()},
                         "terminal": terminal, "value": value,
                         "provider_recorded_screen": original_screen,
                         "recorded_screen": screen, "reason": reason}
        artifact_digest = schemas.content_hash(artifact_body)
        artifact_sink(_freeze(dict(artifact_body, artifact_digest=artifact_digest)))
        if not terminal:
            paused_reason = reason or "unit did not complete"
            break
        raws.append(ep.RawUnit.from_dict({"schema": ep.UNIT_SCHEMA,
                    "plan_digest": plan.digest, "unit_id": spec.unit_id, "arm": spec.arm,
                    "process_id": spec.process_id, "prompt_ids": list(prompt_ids),
                    "terminal": terminal, "value": value, "witnesses": {
                        key: item.to_dict() for key, item in witnesses.items()},
                    "recorded_screen": screen, "reason": reason,
                    "artifact_digest": artifact_digest,
                    "observed_order_index": spec.order_index}))
    view = ep.admissible_units(plan, raws)
    return PlannedServingRun(RUN_SCHEMA, plan.digest, prompts.digest, lineage_id,
                             _freeze(actual_identities["anchor"]),
                             _freeze(actual_identities["candidate"]),
                             tuple(raws), view, "policy_undefined", view.complete,
                             paused_reason)


__all__ = ["ARTIFACT_SCHEMA", "PROMPT_SCHEMA", "RUN_SCHEMA", "FrozenPrompt",
           "FrozenPromptManifest", "ExecutionGuard", "PlannedServingError",
           "PlannedServingRun", "STAGES", "StageCompletion", "StageFence", "StagePaused",
           "TrustedStageProvider",
           "TrustedStageProviderRequired", "UnsupportedContainment", "arm_identity",
           "run_planned_comparison"]
