"""Factual parent evidence for one owned v2 planned-serving unit.

The controller supplies ParentUnitContext from its own held claim and descendant
event, never from a child request. The child supplies only a sealed artifact
reference. This module does not acquire resources, decide scientific validity,
classify foreign processes, or confer correctness/GPU/purpose warrants.

Supported witnesses are deliberately narrow: identity (native artifact joins),
request_completeness, placement (observed affinity/allowed NUMA equality), and
runtime_readback (the recipe's supported /proc/status readbacks). A pass is a
statement about the named facts, not inference placement, a kernel gain or a gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping

from . import experiment_plan as ep
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import observation_binding as ob
from . import planned_serving as ps
from . import resolved_recipe as rr
from . import serving
from . import worker_lifecycle as wl

COMPLETION_REQUEST_SCHEMA = "epyc.autokernel.planned_worker_unit_completion_request.v2"
RECEIPT_SCHEMA = "epyc.autokernel.native_parent_unit_evidence.v1"
READBACK_SCHEMA = "epyc.autokernel.parent_proc_status_readback.v1"
SUPPORTED_STATUS_FIELDS = frozenset({"THP_enabled"})
SUPPORTED_WITNESSES = frozenset({"identity", "request_completeness", "placement",
                                 "runtime_readback"})
TARGET_PLACEMENT_PHASES = ("load", "placement", "health", "warmup", "measurement")
_NATIVE_FIELDS = frozenset({"schema", "kind", "plan_digest", "unit_id", "arm",
    "process_generation_id", "lineage_id", "fence_id", "grant_id", "container_id",
    "worker_identity", "observed_started_at", "observed_ended_at",
    "prompt_manifest_digest", "comparison_identities", "observations",
    "selected_observation", "value", "error", "lifecycle_observation", "artifact_digest"})


class NativeEvidenceRefused(RuntimeError):
    """Wrong ownership, identity, artifact, or closed wire contract."""


def _plain(value: Any) -> Any:
    return ob._plain(value)


def _hash(value: Mapping[str, Any]) -> str:
    return wl._digest(_plain(value))


def _closed(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise NativeEvidenceRefused(f"{label} has missing or unknown fields")
    return _plain(value)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if _plain(actual) != _plain(expected):
        raise NativeEvidenceRefused(f"{label} differs from parent authority")


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise NativeEvidenceRefused("timestamp must be text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise NativeEvidenceRefused("timestamp lacks timezone")
    return parsed


def _artifact(value: Any) -> mc.StoredArtifact:
    # 'verified' is grammar only. ArtifactStore.read/verify reopens the actual bytes.
    row = _closed(value, {"locator", "sha256", "verified"}, "artifact reference")
    if type(row["verified"]) is not bool:
        raise NativeEvidenceRefused("artifact verified field must be boolean")
    wl._sha(row["sha256"], "artifact digest")
    wl._text(row["locator"], "artifact locator")
    return mc.StoredArtifact(**row)


@dataclass(frozen=True)
class ParentUnitContext:
    """Trusted constructor-only inputs; this is not a child-deserializable grant."""

    plan: ep.ExperimentPlan
    unit_id: str
    recipe: rr.ResolvedRecipe | rr.CanonicalResolvedRecipe
    prompts: ps.FrozenPromptManifest
    fence: ps.StageFence
    binding: ob.ObservationUnitBinding
    descendant_event: Mapping[str, Any]
    active_claim: Mapping[str, Any]
    nonce: str
    template: serving.Recipe | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.plan, ep.ExperimentPlan)
                or not isinstance(self.prompts, ps.FrozenPromptManifest)
                or not isinstance(self.fence, ps.StageFence)
                or not isinstance(self.binding, ob.ObservationUnitBinding)
                or not isinstance(self.recipe, (rr.ResolvedRecipe, rr.CanonicalResolvedRecipe))):
            raise NativeEvidenceRefused("parent unit inputs must be concrete typed records")
        object.__setattr__(self, "plan", ep.ExperimentPlan.from_dict(self.plan.to_dict()))
        object.__setattr__(self, "recipe", rr.resolved_recipe_from_dict(self.recipe.to_dict()))
        template = self.template or getattr(self.recipe, "template", None)
        if not isinstance(template, serving.Recipe):
            raise NativeEvidenceRefused("resolved recipe needs its concrete serving template")
        object.__setattr__(self, "template", serving.Recipe.from_dict(template.to_dict()))
        self.recipe.validate_launch(self.template, self.recipe.build_dir, self.recipe.port)
        object.__setattr__(self, "prompts", ps.FrozenPromptManifest.from_dict(self.prompts.to_dict()))
        object.__setattr__(self, "binding", ob.ObservationUnitBinding.from_dict(self.binding.to_dict()))
        event = wl.validate_event(_plain(self.descendant_event))
        if event["event"] != "OWNED_DESCENDANT_CAPTURED":
            raise NativeEvidenceRefused("parent event is not a captured descendant")
        claim = _closed(self.active_claim, {"schema", "grant_id", "grant_generation",
            "container_id", "held_claim", "active_claim_ref", "claim_digest"}, "active claim")
        claim_digest = claim.pop("claim_digest")
        _equal(claim_digest, _hash(claim), "active claim digest")
        _equal(claim["schema"], wl.ACTIVE_OBSERVATION_CLAIM_SCHEMA, "active claim schema")
        claim["claim_digest"] = claim_digest
        object.__setattr__(self, "descendant_event", ob._freeze(event))
        object.__setattr__(self, "active_claim", ob._freeze(claim))
        wl._text(self.nonce, "parent nonce")
        if self.plan.schema != ep.PLAN_SCHEMA_V2:
            raise NativeEvidenceRefused("parent producer requires a frozen v2 plan")
        unit = self.unit
        binding, fence = self.binding, self.fence
        wl._finite(fence.valid_until, "parent fence deadline", positive=True)
        _equal(event["plan_digest"], self.plan.digest, "event plan")
        _equal(event["campaign_id"], self.plan.campaign_id, "event campaign")
        for name in ("unit_id", "process_generation_id", "fence_id"):
            _equal(event["data"][name], getattr(binding, name), f"event {name}")
            _equal(getattr(fence, name), getattr(binding, name), f"fence {name}")
        _equal((binding.unit_id, binding.process_generation_id),
               (unit.unit_id, unit.process_id), "bound unit")
        _equal(binding.clock_domain, fence.clock_domain, "clock domain")
        _equal(event["data"]["binding_digest"], binding.to_dict()["binding_digest"], "binding digest")
        _equal(event["data"]["process"]["boot_id"], binding.boot_id, "boot identity")
        _equal(event["data"]["container_identity"],
               binding.worker_binding["container_identity"], "container identity")
        for event_key, binding_key in (("worker_id", "worker_id"),
                ("worker_generation", "worker_incarnation"), ("grant_id", "grant_id"),
                ("grant_generation", "grant_generation")):
            _equal(event[event_key], binding.worker_binding[binding_key], event_key)
        for event_key, fence_key in (("worker_id", "worker_id"),
                ("worker_generation", "worker_incarnation"), ("grant_id", "grant_id"),
                ("container_id", "container_id"), ("lineage_id", "lineage_id"),
                ("supervisor_id", "supervisor_id"),
                ("supervisor_incarnation", "supervisor_incarnation"),
                ("config_generation", "config_generation")):
            _equal(event[event_key], getattr(fence, fence_key), f"event/fence {event_key}")
        _equal(event["container_id"], binding.container_id, "binding container")
        for name in ("grant_id", "grant_generation", "container_id"):
            _equal(claim[name], event[name], f"claim {name}")
        _equal(claim["held_claim"], binding.held_claim, "held resources")
        if (self.recipe.backend == "cpu" and binding.held_claim["gpu_devices"]
                or self.recipe.backend == "gpu" and not binding.held_claim["gpu_devices"]):
            raise NativeEvidenceRefused("recipe backend differs from the held device claim")
        _equal(claim["active_claim_ref"], binding.active_claim_ref, "claim reference")
        _equal(self.plan.loaded_instrument, binding.instrument.to_dict(), "loaded instrument")
        if not binding.instrument.configuration_complete:
            raise NativeEvidenceRefused("loaded instrument configuration is incomplete")
        expected_identity = (self.plan.anchor_identity if unit.arm == "anchor"
                             else self.plan.candidate_identity)
        _equal(ps.arm_identity(self.template, self.recipe,
                              loaded_instrument=binding.instrument.to_dict()),
               expected_identity, "resolved arm identity")
        self.prompts.requests(unit.expected_prompt_ids, self.template)

    @property
    def unit(self) -> ep.UnitSpec:
        found = [unit for unit in self.plan.expected_units if unit.unit_id == self.unit_id]
        if len(found) != 1:
            raise NativeEvidenceRefused("parent unit is absent or ambiguous")
        return found[0]

    @property
    def identity(self) -> dict[str, Any]:
        return {"plan_digest": self.plan.digest, "unit_id": self.unit_id,
            "fence_id": self.fence.fence_id, "binding_digest": self.binding.to_dict()["binding_digest"],
            "descendant_event_digest": _hash(self.descendant_event),
            "recipe_digest": self.recipe.execution_digest,
            "active_claim_digest": self.active_claim["claim_digest"]}


def completion_request(context: ParentUnitContext, native: mc.StoredArtifact) -> dict[str, Any]:
    """Closed artifact-only request for the owner to thread through the v2 socket."""
    row = {"schema": COMPLETION_REQUEST_SCHEMA, "nonce": context.nonce,
           "sequence": context.unit.order_index + 1, "fence_id": context.fence.fence_id,
           "native_observation": native.to_dict()}
    return {**row, "request_digest": _hash(row)}


@dataclass(frozen=True)
class NativeUnitEvidenceResult:
    completion: ps.StageCompletion
    receipt: mc.StoredArtifact
    receipt_digest: str


class NativeUnitEvidenceProducer:
    """One parent-owned unit; use in the existing bounded evidence thread.

    capture_runtime_readback must run at a live health/warmup/measurement boundary.
    evaluate never samples a dead process retrospectively. runtime_probe is a
    concrete reader for fixtures/host integration, not a permissive verifier.
    """

    def __init__(self, *, store: mc.ArtifactStore, context: ParentUnitContext,
                 runtime_probe: lo.FilesystemProbe | None = None,
                 scientific_adapters: Any = None) -> None:
        from .native_scientific_witness import ParentScientificWitnessAdapters
        if type(store) is not mc.ArtifactStore or type(context) is not ParentUnitContext:
            raise NativeEvidenceRefused("concrete artifact store and parent context required")
        if runtime_probe is not None and type(runtime_probe) is not lo.FilesystemProbe:
            raise NativeEvidenceRefused("runtime probe must be the bounded filesystem reader")
        self.store, self.context = store, context
        if scientific_adapters is not None and type(scientific_adapters) is not ParentScientificWitnessAdapters:
            raise NativeEvidenceRefused("scientific evidence requires the closed concrete adapter registry")
        self.scientific_adapters = scientific_adapters
        if scientific_adapters is not None:
            from .native_producer_source import PRODUCER_SOURCE_SCHEMA_V2, validate_producer_source_closure
            original_instrument = store.read(context.binding.instrument.artifact.locator,
                                             context.binding.instrument.artifact.sha256)
            closure = validate_producer_source_closure(
                original_instrument["used_constants"].get("producer_source_closure"))
            if closure["schema"] != PRODUCER_SOURCE_SCHEMA_V2:
                raise NativeEvidenceRefused("scientific adapter was not bound before plan issuance")
            _equal(closure["scientific_adapters"], scientific_adapters.source_identity(),
                   "prospectively selected scientific adapter")
        self._probe = runtime_probe if runtime_probe is not None else lo.FilesystemProbe()
        self._readbacks: list[mc.StoredArtifact] = []
        self._request_digest: str | None = None
        self._result: NativeUnitEvidenceResult | None = None
        self._source_pins = {
            "module_artifact": lo.prepare_artifact_identity(Path(__file__)),
            "constants": {"receipt_schema": RECEIPT_SCHEMA,
                "completion_request_schema": COMPLETION_REQUEST_SCHEMA,
                "supported_status_fields": sorted(SUPPORTED_STATUS_FIELDS),
                "supported_witnesses": sorted(SUPPORTED_WITNESSES),
                "target_placement_phases": list(TARGET_PLACEMENT_PHASES)},
            "callables": {function.__qualname__: lo.callable_identity(function) for function in (
                ParentUnitContext.__post_init__, type(self).evaluate, type(self)._join_native,
                type(self)._join_observation, type(self)._requests, type(self)._coverage,
                type(self)._placement, type(self)._runtime, type(self).capture_runtime_readback,
                _plain, _hash, _closed, _equal, _timestamp, _artifact,
                lo.validate_observation, lo.physical_footprint, lo._intervals,
                serving.verify_env_readback, ob.validate_reopened_observation,
                wl.validate_event)}}
        if self.scientific_adapters is not None:
            self._source_pins["scientific_adapters"] = _plain(self.scientific_adapters.source_identity())

    def capture_runtime_readback(self, *, phase: str) -> mc.StoredArtifact:
        if phase not in ("health", "warmup", "measurement"):
            raise NativeEvidenceRefused("runtime readback requires a live request-phase boundary")
        if self._result is not None or len(self._readbacks) >= self.context.binding.budgets["max_samples"]:
            raise NativeEvidenceRefused("runtime readback is terminal or over its existing sample budget")
        context, probe = self.context, self._probe
        process = _plain(context.descendant_event["data"]["process"])
        expected = tuple(context.recipe.readback_expectations)
        started = time.monotonic()
        if started >= context.fence.valid_until:
            raise NativeEvidenceRefused("runtime readback fence expired")
        text, error = None, None
        try:
            budget = context.binding.budgets
            _equal(probe.boot_id(budget), process["boot_id"], "readback boot")
            for when in ("before", "after"):
                if when == "after":
                    text = lo._bounded_text(probe.proc_root / str(process["pid"]) / "status",
                                            budget["max_read_bytes"], "parent status readback")
                observed = probe.process_identity(process["pid"], budget)
                _equal((observed["pid"], observed["start_ticks"]),
                       (process["pid"], process["start_ticks"]), f"readback identity {when}")
                probe.verify_container(process["pid"],
                    _plain(context.binding.worker_binding["container_identity"]), budget)
            _equal(probe.boot_id(budget), process["boot_id"], "readback final boot")
        except Exception as exc:
            text, error = None, f"{type(exc).__name__}: {exc}"
        ended = time.monotonic()
        if ended > context.fence.valid_until:
            text, error = None, "readback returned after fence deadline"
        body = {"schema": READBACK_SCHEMA, "identity": context.identity,
            "process": process, "phase": phase, "clock_domain": context.fence.clock_domain,
            "started_monotonic_s": started, "ended_monotonic_s": ended,
            "expected": [list(item) for item in expected], "status_text": text, "error": error,
            "source": {"proc_root": str(probe.proc_root), "boot_id_path": str(probe.boot_id_path),
                       "reader": lo.callable_identity(lo._bounded_text),
                       "capture": lo.callable_identity(type(self).capture_runtime_readback)}}
        ref = self.store.write(f"parent-status-readback:{_hash(body)}", body)
        self._readbacks.append(ref)
        return ref

    def evaluate(self, request: Mapping[str, Any]) -> NativeUnitEvidenceResult:
        packet = _closed(request, set(("schema", "nonce", "sequence", "fence_id",
            "native_observation", "request_digest")), "v2 completion request")
        digest = packet.pop("request_digest")
        _equal(digest, _hash(packet), "completion request digest")
        context, unit = self.context, self.context.unit
        _equal(packet["schema"], COMPLETION_REQUEST_SCHEMA, "completion schema")
        _equal(packet["nonce"], context.nonce, "completion nonce")
        if type(packet["sequence"]) is not int:
            raise NativeEvidenceRefused("completion sequence must be an integer")
        _equal(packet["sequence"], unit.order_index + 1, "completion sequence")
        _equal(packet["fence_id"], context.fence.fence_id, "completion fence")
        if self._result is not None:
            _equal(digest, self._request_digest, "unit completion retry")
            return self._result
        native_ref = _artifact(packet["native_observation"])
        native = _closed(self.store.read(native_ref.locator, native_ref.sha256),
                         _NATIVE_FIELDS, "native observation")
        native_digest = native["artifact_digest"]
        _equal(native_digest, _hash({k: v for k, v in native.items() if k != "artifact_digest"}),
               "native artifact digest")
        self.store.verify(f"raw:{native_digest}", native)
        self._join_native(native)
        reference = ob.LifecycleObservationReference.from_dict(native["lifecycle_observation"])
        link = ob.validate_reopened_observation(reference, store=self.store,
            expected={"unit_id": unit.unit_id, "process_generation_id": unit.process_id,
                "fence_id": context.fence.fence_id, "active_claim_ref": context.binding.active_claim_ref,
                "container_id": context.binding.container_id,
                "capture_context": _plain(context.binding.worker_binding)},
            instrument=context.binding.instrument)
        observation = lo.validate_observation(_plain(self.store.read(
            reference.artifact.locator, reference.artifact.sha256)))
        self._join_observation(native, observation, reference)
        requests_ok, request_facts = self._requests(native)
        coverage_ok, coverage = self._coverage(observation)
        placement = self._placement(observation, coverage_ok)
        runtime = self._runtime(observation)
        findings = {
            "identity": {"status": "pass", "reason": "exact parent/native artifact joins",
                         "facts": context.identity},
            "request_completeness": {"status": "pass" if requests_ok else "fail",
                "reason": "frozen terminal request/value join" if requests_ok else "request membership/completion/value mismatch",
                "facts": request_facts},
            "placement": placement, "runtime_readback": runtime}
        if self.scientific_adapters is not None:
            _equal(self.scientific_adapters.source_identity(),
                   self._source_pins["scientific_adapters"], "selected scientific adapter source")
            findings.update(_plain(self.scientific_adapters.findings(context, native, link,
                tuple(self._readbacks), store=self.store)))
        for name in set(context.plan.required_witnesses) | set(("correctness", "contention", "purpose", "residency")):
            findings.setdefault(name, {"status": "unknown", "reason": "no owning evidence adapter implemented",
                                       "facts": {"backend": context.recipe.backend}})
        # No environmental/scientific policy is invented here. Even a fully factual
        # unit remains flagged; the existing intended-use consumer owns eligibility.
        terminal = bool(requests_ok and reference.successor_permitted
                        and observation["ended_monotonic_s"] <= context.fence.valid_until)
        body = {"schema": RECEIPT_SCHEMA, "identity": context.identity,
            "completion_request_digest": digest, "native_observation": native_ref.to_dict(),
            "lifecycle_observation": reference.to_dict(),
            "parent_descendant_event": _plain(context.descendant_event),
            "parent_active_claim": _plain(context.active_claim),
            "parent_observation_binding": context.binding.to_dict(),
            "recipe_source_pins": context.recipe.to_dict(),
            "producer_source_pin": lo.callable_identity(type(self).evaluate),
            "producer_supporting_pins": self._source_pins,
            "runtime_readbacks": [item.to_dict() for item in self._readbacks],
            "phase_coverage": coverage, "phase_facts": _plain(link.phase_facts),
            "findings": findings, "terminal": terminal,
            "scientific_authority": "none; owning plan/protocol required"}
        receipt_digest = _hash(body)
        receipt = self.store.write(f"parent-unit-evidence:{receipt_digest}", body)
        witnesses = MappingProxyType({name: ep.Witness(row["status"],
            f"parent-unit-evidence:{receipt_digest}#{name}" if row["status"] != "unknown" else None)
            for name, row in findings.items()})
        completion = ps.StageCompletion(context.fence.fence_id, terminal, witnesses,
            "flagged_but_retained" if terminal else "rejected",
            "factual evidence only; unsupported witnesses remain unknown")
        result = NativeUnitEvidenceResult(completion, receipt, receipt_digest)
        self._request_digest, self._result = digest, result
        return result

    def _join_native(self, native: Mapping[str, Any]) -> None:
        context, unit = self.context, self.context.unit
        expected = {"schema": ps.ARTIFACT_SCHEMA_V2, "kind": "native_observation",
            "plan_digest": context.plan.digest, "unit_id": unit.unit_id, "arm": unit.arm,
            "process_generation_id": unit.process_id, "lineage_id": context.fence.lineage_id,
            "fence_id": context.fence.fence_id, "grant_id": context.fence.grant_id,
            "container_id": context.fence.container_id, "prompt_manifest_digest": context.prompts.digest,
            "comparison_identities": {"anchor": _plain(context.plan.anchor_identity),
                                      "candidate": _plain(context.plan.candidate_identity)},
            "worker_identity": {key: getattr(context.fence, key) for key in (
                "supervisor_id", "supervisor_incarnation", "config_generation",
                "worker_id", "worker_incarnation")}}
        for name, value in expected.items():
            _equal(native[name], value, f"native {name}")
        observations = native["observations"]
        if not isinstance(observations, list) or len(observations) != 1:
            raise NativeEvidenceRefused("one process unit requires one selected native observation")
        _equal(native["selected_observation"], observations[0], "selected observation")
        _equal(observations[0].get("process_pid"),
               context.descendant_event["data"]["process"]["pid"], "native target PID")

    def _join_observation(self, native: Mapping[str, Any], observation: Mapping[str, Any],
                          reference: ob.LifecycleObservationReference) -> None:
        context, binding = self.context, self.context.binding
        _equal(reference.descendant_binding_ref, _hash(context.descendant_event), "parent descendant event reference")
        _equal(observation["target_binding"], {
            **_plain(context.descendant_event["data"]["process"]),
            "worker_binding": _plain(binding.worker_binding),
            "binding_ref": _hash(context.descendant_event)}, "parent target binding")
        for name, expected in {"recipe_identity_digest": context.recipe.execution_digest,
                "backend": context.recipe.backend,
                "instrument_identity_digest": binding.instrument.identity_sha256,
                "requested_effective_state": binding.requested_effective_state,
                "worker_binding": binding.worker_binding, "clock_domain": binding.clock_domain,
                "boot_id": binding.boot_id, "cadence_s": binding.cadence_s,
                "gap_limit_s": binding.gap_limit_s, "budgets": binding.budgets}.items():
            _equal(observation[name], expected, f"observed {name}")
        held = observation["held_claim"]
        _equal({key: held[key] for key in ("logical_cpus", "gpu_devices")}, binding.held_claim, "observed held claim")
        physical = lo.physical_footprint(binding.held_claim["logical_cpus"], observation["topology"],
                                         max_cpu_ids=binding.budgets["max_cpu_ids"])
        _equal(held["physical_cpus"], physical, "physical claim footprint")
        requested_physical = lo.physical_footprint(binding.requested_effective_state["logical_cpus"],
            observation["topology"], max_cpu_ids=binding.budgets["max_cpu_ids"])
        if not set(requested_physical).issubset(physical):
            raise NativeEvidenceRefused("requested execution footprint exceeds the held physical claim")
        intervals, _issues = lo._intervals(observation["samples"], set(physical), binding.gap_limit_s)
        _equal(observation["intervals"], intervals, "replayed sample intervals")
        times = [_timestamp(native["observed_started_at"]), _timestamp(observation["started_at"]),
                 _timestamp(context.descendant_event["occurred_at"]),
                 _timestamp(observation["ended_at"]), _timestamp(native["observed_ended_at"])]
        if times != sorted(times):
            raise NativeEvidenceRefused("parent capture/lifecycle/native observation windows do not join")
        for sample in observation["samples"]:
            if not (observation["started_monotonic_s"] <= sample["marker_monotonic_s"]
                    <= sample["read_started_monotonic_s"] <= sample["read_ended_monotonic_s"]
                    <= observation["ended_monotonic_s"]):
                raise NativeEvidenceRefused("sample lies outside its lifecycle window")
            target = sample["target"]
            if target is not None:
                _equal((target["pid"], target["start_ticks"]),
                       (reference.target_pid, reference.target_start_ticks), "sample target identity")
                _equal(target["container_identity"], binding.worker_binding["container_identity"], "sample target container")

    def _requests(self, native: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        context = self.context
        selected = native["selected_observation"]
        rows = selected.get("requests")
        if not isinstance(rows, list):
            return False, {"reason": "requests are not an array"}
        rows = [row for row in rows if isinstance(row, Mapping) and row.get("phase") == "measurement"]
        expected = context.prompts.requests(context.unit.expected_prompt_ids, context.template)
        expected_hashes = [hashlib.sha256(body).hexdigest() for _, body in expected]
        prompts_by_id = {prompt.prompt_id: prompt for prompt in context.prompts.prompts}
        expected_counts = [prompts_by_id[prompt_id].n_predict for prompt_id, _ in expected]
        rates = [row.get("predicted_per_second") for row in rows]
        rates_valid = all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in rates)
        ok = (len(rows) == len(expected) and [row.get("prompt_id") for row in rows] == [pid for pid, _ in expected]
              and [row.get("request_sha256") for row in rows] == expected_hashes
              and [row.get("predicted_n") for row in rows] == expected_counts
              and [row.get("slot_index") for row in rows] == list(range(len(rows)))
              and all(row.get("terminal") is True and row.get("error") is None
                      and type(row.get("predicted_n")) is int and row["predicted_n"] > 0 for row in rows)
              and rates_valid and type(native["value"]) in (int, float)
              and math.isfinite(native["value"]) and sum(rates) == native["value"]
              and native["error"] is None and selected.get("failure") is None
              and selected.get("teardown") in {"terminated", "killed"})
        return bool(ok), {"expected_prompt_ids": list(context.unit.expected_prompt_ids),
            "expected_request_hashes": expected_hashes, "measurement_requests": rows,
            "expected_predicted_n": expected_counts,
            "value": native["value"], "teardown": selected.get("teardown")}

    @staticmethod
    def _coverage(observation: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        counts = {phase: sum(sample["status"] == "observed" and sample["phase"] == phase
                             for sample in observation["samples"]) for phase in lo.PHASES}
        gaps = [index for index, interval in enumerate(observation["intervals"])
                if interval["status"] != "observed"]
        complete = (all(counts.values()) and not gaps and not observation["issues"]
                    and observation["completeness"] == "complete"
                    and observation["shutdown"]["status"] == "resolved")
        return bool(complete), {"observed_samples_by_phase": counts,
            "unknown_intervals": gaps, "issues": observation["issues"], "complete": bool(complete)}

    def _placement(self, observation: Mapping[str, Any], coverage_ok: bool) -> dict[str, Any]:
        requested = _plain(self.context.binding.requested_effective_state)
        samples = []
        mismatch = False
        boundaries = observation["phase_boundaries"]
        phase_windows = {item["phase"]: (item["monotonic_s"],
            boundaries[index + 1]["monotonic_s"] if index + 1 < len(boundaries)
            else observation["ended_monotonic_s"])
            for index, item in enumerate(boundaries)}
        for index, sample in enumerate(observation["samples"]):
            target = sample["target"]
            if target is None:
                continue
            cpus, nodes = target["cpus_allowed"], target["mems_allowed"]
            if (not isinstance(cpus, list) or not cpus or not isinstance(nodes, list) or not nodes
                    or any(type(value) is not int or value < 0 for value in cpus + nodes)):
                raise NativeEvidenceRefused("observed placement is malformed")
            physical = lo.physical_footprint(cpus, observation["topology"],
                                           max_cpu_ids=self.context.binding.budgets["max_cpu_ids"])
            _equal(target["physical_affinity_footprint"], physical, "observed target footprint")
            matches = cpus == requested["logical_cpus"] and nodes == requested["numa_nodes"]
            mismatch |= not matches
            window = phase_windows.get(sample["phase"])
            in_phase = bool(window and window[0] <= sample["marker_monotonic_s"]
                <= sample["read_started_monotonic_s"] <= sample["read_ended_monotonic_s"]
                <= window[1])
            samples.append({"sample_index": index, "phase": sample["phase"],
                "status": sample["status"], "logical_cpus": cpus, "allowed_numa_nodes": nodes,
                "numa_maps": target["numa_maps"], "matches": matches,
                "within_declared_phase": in_phase})
        target_counts = {phase: sum(row["phase"] == phase and row["status"] == "observed"
            and row["within_declared_phase"] for row in samples)
            for phase in TARGET_PLACEMENT_PHASES}
        status = "fail" if mismatch else "pass" if coverage_ok and all(target_counts.values()) else "unknown"
        return {"status": status, "reason": "observed allowed affinity/NUMA equality; no page-placement or isolation claim",
                "facts": {"requested": requested, "samples": samples,
                    "target_samples_by_required_phase": target_counts}}

    def _runtime(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        expected = tuple(self.context.recipe.readback_expectations)
        facts = []
        if not expected or any(field not in SUPPORTED_STATUS_FIELDS for field, _ in expected):
            return {"status": "unknown", "reason": "no supported declared proc/status readback", "facts": {"expected": expected}}
        for ref in self._readbacks:
            row = _plain(self.store.read(ref.locator, ref.sha256))
            _equal(row["identity"], self.context.identity, "parent readback identity")
            _equal(row["expected"], [list(item) for item in expected], "readback expectations")
            start, end = row["started_monotonic_s"], row["ended_monotonic_s"]
            in_window = observation["started_monotonic_s"] <= start <= end <= observation["ended_monotonic_s"]
            boundaries = observation["phase_boundaries"]
            phase_windows = [(item["monotonic_s"],
                boundaries[index + 1]["monotonic_s"] if index + 1 < len(boundaries)
                else observation["ended_monotonic_s"])
                for index, item in enumerate(boundaries) if item["phase"] == row["phase"]]
            in_phase = any(begin <= start <= end <= finish for begin, finish in phase_windows)
            status, reason = "unknown", row["error"] or "readback is outside the lifecycle window"
            if in_window and not in_phase:
                reason = "readback does not join its declared phase window"
            if in_window and in_phase and row["error"] is None and row["status_text"] is not None:
                try:
                    observed = serving.verify_env_readback(self.context.template,
                        row["process"]["pid"], status_text=row["status_text"], expectations=expected)
                    status, reason = "pass", "declared live process readback matched"
                except serving.EnvReadbackFailed as exc:
                    observed = serving._status_fields(row["status_text"])
                    status = "fail" if all(field in observed for field, _ in expected) else "unknown"
                    reason = str(exc)
            else:
                observed = None
            facts.append({"artifact": ref.to_dict(), "phase": row["phase"], "status": status,
                          "reason": reason, "observed": observed, "within_lifecycle": in_window,
                          "within_declared_phase": in_phase})
        statuses = [row["status"] for row in facts]
        status = "fail" if "fail" in statuses else "pass" if statuses and all(x == "pass" for x in statuses) else "unknown"
        return {"status": status, "reason": "parent-captured declared proc/status readback only",
                "facts": {"expected": expected, "samples": facts}}


__all__ = ["COMPLETION_REQUEST_SCHEMA", "NativeEvidenceRefused", "NativeUnitEvidenceProducer",
           "NativeUnitEvidenceResult", "ParentUnitContext", "completion_request"]
