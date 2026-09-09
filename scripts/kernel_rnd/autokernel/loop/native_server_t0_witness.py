"""Parent-issued same-server v1 T0 evidence; never CLI-to-server transfer.

Only the original concrete NativeT0WitnessAdapter can supply owning issuance.
All HTTP bytes are reopened through the published native recorder. No live
runner, resource acquisition, device API, or arbitrary verifier callback exists
in this module. v1 proves content-byte coherence only; unsupported gates remain
explicitly missing in a complete owning report.
"""
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Mapping

from ..evaluator import api, correctness
from ..execution import server_generation as sg
from ..execution import t0_provider as t0
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_scientific_witness as scientific
from . import native_server_response as raw
from . import observation_binding as ob
from . import worker_lifecycle as wl

ADAPTER_ID = "epyc.autokernel.native_server_t0_witness.v1"
INPUT_SCHEMA = "epyc.autokernel.parent_server_t0_inputs.v1"
RECEIPT_SCHEMA = "epyc.autokernel.parent_server_t0_witness.v1"
PAIR_SCHEMA = "epyc.autokernel.parent_server_t0_pair.v1"
PAIR_REFERENCE_SCHEMA = "epyc.autokernel.parent_server_t0_pair_reference.v1"
REPLAY_COVERAGE = "original_server_content_bytes_and_owning_t0_reducer_replay"
ADAPTER_ROLES = ("__init__", "prepare_model_identity", "register_owning_inputs", "_original",
    "_observe", "_generation", "_anchor", "_reduce", "_build", "evaluate", "reopen",
    "_pair_body", "finalize_pair", "reopen_pair", "reopen_pair_reports", "source_identity")
HELPER_ROLES = ("_source_pins", "_original_identity", "_unknown_surface")
MAX_UNITS = scientific.MAX_UNITS


def _source_pins() -> Mapping[str, Any]:
    from . import native_parent_evidence as npe
    from . import native_parent_receipt_replay as replay
    from . import native_final_trial as final
    functions = (sg.ServerGenerationEvidence.__init__, sg.ServerGenerationEvidence.__post_init__,
        sg.ServerGenerationEvidence.is_greedy, sg.collect_server_coherence,
        scientific.NativeT0WitnessAdapter._replay, scientific.NativeT0WitnessAdapter._model_fact,
        scientific._native_artifact_facts, scientific._immutable, scientific._project,
        correctness.seal_static_t0_bundle, correctness.evaluate_t0_with_static_bundle,
        correctness.T0CorrectnessRunner.evaluate, correctness.StaticEvidenceProvider.evidence_for,
        npe.NativeUnitEvidenceProducer._join_native, npe.NativeUnitEvidenceProducer._join_observation,
        replay.IssuedNativeEvidenceRegistry.snapshot, replay.NativeParentReceiptReplayer._replay_unit,
        ParentServerT0PairReference.__init__, ParentServerT0PairReference.to_dict)
    return ob._freeze({"modules": [lo.prepare_artifact_identity(Path(item.__file__))
                                  for item in (sg, scientific, t0, correctness, npe, replay)],
        "callables": [lo.callable_identity(function) for function in functions],
        "raw_source": raw.source_identity(), "final_trial_source": final.source_identity()})


def _original_identity(issued, context) -> api.AnchorIdentity:
    """Name the original owning collection; never copy EvaluationRequest.anchor."""
    plan, request, linkage = issued.plan, issued.request, issued.evidence.linkage
    recipe = context.recipe
    if type(linkage) is not correctness.LinkageEvidence:
        raise scientific.ScientificWitnessRefused("original owning linkage capture is unavailable")
    scientific._same((plan.candidate.binary, plan.candidate.build_dir),
        (recipe.executable.path, recipe.build_dir), "original server tool/build")
    scientific._same((request.artifact.binary_sha256, linkage.binary_sha256),
        (recipe.executable.sha256, recipe.executable.sha256), "original server executable bytes")
    scientific._same((request.artifact.source_sha256, request.artifact.linkage_sha256),
        (plan.candidate.source_sha256, linkage.linkage_sha256), "original source/linkage")
    scientific._same(tuple(t0._launch_env(plan.candidate.library_path, plan.base_env,
        plan.parameter_env)), tuple(recipe.launch_env), "original server launch environment")
    scientific._same(request.backend, {"cpu": "llama_cpu", "gpu": "llama_gpu"}[recipe.backend],
                     "original server backend")
    # This is the original issued source identity, not proof that all source
    # gates passed. The complete original report remains part of the receipt.
    return api.AnchorIdentity(source_commit=plan.candidate.source_commit,
        binary_sha256=linkage.binary_sha256, linkage_sha256=linkage.linkage_sha256)


def _unknown_surface() -> correctness.ChangeSurface:
    return correctness.ChangeSurface(None, None, None, None, (), (), None, None,
        (), False, "server-t0:original-mechanical-surface-unavailable")


@dataclass(frozen=True)
class _Inputs:
    context: Any
    original_context: Any
    original_artifact: mc.StoredArtifact
    requests: tuple[api.EvaluationRequest, ...]
    anchor_unit_id: str | None
    artifact: mc.StoredArtifact
    body: Mapping[str, Any]


@dataclass(frozen=True)
class _Observed:
    context: Any
    native_artifact: mc.StoredArtifact
    lifecycle_link: Any
    runtime_readbacks: tuple[mc.StoredArtifact, ...]


@dataclass(frozen=True)
class _Issued:
    observed: _Observed
    anchor: _Observed | None
    reference: scientific.WitnessEvidenceReference
    body: Mapping[str, Any]


@dataclass(frozen=True, init=False)
class ParentServerT0PairReference:
    """Scientific receipt only; never a capture or current-result capability."""
    artifact: mc.StoredArtifact
    digest: str

    def __init__(self, artifact: mc.StoredArtifact, digest: str) -> None:
        if type(artifact) is not mc.StoredArtifact or artifact.verified is not True:
            raise scientific.ScientificWitnessRefused("concrete pair artifact required")
        wl._sha(digest, "pair receipt digest")
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PAIR_REFERENCE_SCHEMA,
                "artifact": self.artifact.to_dict(), "digest": self.digest}


class NativeServerT0WitnessAdapter:
    def __init__(self, *, max_units: int = MAX_UNITS,
                 owning_issuer: scientific.NativeT0WitnessAdapter | None = None) -> None:
        if type(max_units) is not int or not 1 <= max_units <= MAX_UNITS:
            raise scientific.ScientificWitnessRefused("invalid server T0 registry bound")
        if owning_issuer is None:
            owning_issuer = scientific.NativeT0WitnessAdapter(max_units=max_units)
        if type(owning_issuer) is not scientific.NativeT0WitnessAdapter:
            raise scientific.ScientificWitnessRefused("original concrete owning T0 issuer required")
        self._max_units, self._owning_issuer = max_units, owning_issuer
        self._lock = threading.RLock()
        self._inputs: dict[tuple[str, str], _Inputs] = {}
        self._inflight: set[tuple[str, str]] = set()
        self._observed: dict[tuple[str, str], _Observed] = {}
        self._issued: dict[tuple[str, str], _Issued] = {}
        self._pairs: dict[str, tuple[ParentServerT0PairReference, Mapping[str, Any]]] = {}
        self._pair_inflight: set[str] = set()
        from .native_final_trial import NativeFinalTrialOwner
        self._final_trial_owner = NativeFinalTrialOwner(correctness_adapter=self)

    @property
    def max_units(self) -> int:
        return self._max_units

    @property
    def owning_issuer(self) -> scientific.NativeT0WitnessAdapter:
        return self._owning_issuer

    @property
    def final_trial_owner(self):
        return self._final_trial_owner

    def prepare_model_identity(self, **kwargs) -> mc.StoredArtifact:
        """Forward the existing scheduled original inventory protocol unchanged."""
        return self._owning_issuer.prepare_model_identity(**kwargs)

    def register_owning_inputs(self, *, context: Any, store: mc.ArtifactStore,
                              original_context: Any, original_artifact: mc.StoredArtifact,
                              request_by_slot: tuple[api.EvaluationRequest, ...],
                              anchor_unit_id: str | None = None) -> mc.StoredArtifact:
        from .native_parent_evidence import ParentUnitContext
        if (type(context) is not ParentUnitContext or type(original_context) is not ParentUnitContext
                or type(store) is not mc.ArtifactStore or type(original_artifact) is not mc.StoredArtifact
                or type(request_by_slot) is not tuple
                or len(request_by_slot) != len(context.unit.expected_prompt_ids)
                or any(type(item) is not api.EvaluationRequest for item in request_by_slot)):
            raise scientific.ScientificWitnessRefused("exact typed parent/owning/slot inputs required")
        if anchor_unit_id is not None:
            anchors = [unit for unit in context.plan.expected_units
                       if unit.unit_id == anchor_unit_id and unit.arm == "anchor"]
            if len(anchors) != 1 or anchor_unit_id == context.unit_id:
                raise scientific.ScientificWitnessRefused("independent original anchor unit required")
        key = (context.plan.digest, context.unit_id)
        with self._lock:
            prior = self._inputs.get(key)
            if key in self._inflight:
                raise scientific.ScientificWitnessRefused("server T0 unit is already in flight")
            if prior is None and len(self._inputs) + len(self._inflight) >= self.max_units:
                raise scientific.ScientificWitnessRefused("server T0 registry capacity exhausted")
            self._inflight.add(key)
        try:
            original = self._owning_issuer._replay(original_context, store)
            scientific._same(original.original_artifact.to_dict(), original_artifact.to_dict(),
                             "original concrete T0 issuance reference")
            identity = _original_identity(original, context)
            scientific._same(original_context.recipe.model.to_dict(), context.recipe.model.to_dict(),
                             "original owning model entry")
            scientific._same([item.to_dict() for item in original_context.recipe.dsos],
                             [item.to_dict() for item in context.recipe.dsos], "original owning DSO set")
            requests = tuple(scientific._immutable(item) for item in request_by_slot)
            for item in requests:
                scientific._same(item.artifact, original.request.artifact, "original slot artifact frame")
                scientific._same((item.campaign_id, item.backend, item.tier),
                    (context.plan.campaign_id, original.request.backend, "T0"), "original slot request scope")
                scientific._same(item.evaluator, original.request.evaluator, "original slot evaluator")
            body = {"schema": INPUT_SCHEMA, "frame": ob._plain(scientific._frame(context)),
                "original_frame": ob._plain(scientific._frame(original_context)),
                "original_t0": original_artifact.to_dict(),
                "original_t0_digest": wl._digest(ob._plain(original.original_body)),
                "original_identity": scientific._project(identity),
                "request_by_slot": [scientific._project(item) for item in requests],
                "policy": scientific._project(original.policy), "anchor_unit_id": anchor_unit_id,
                "source": self.source_identity()}
            if prior is not None:
                scientific._same(body, prior.body, "duplicate original server T0 inputs")
                self._original(prior, store)
                return prior.artifact
            artifact = store.write(f"parent-server-t0-inputs:{wl._digest(body)}", body)
            inputs = _Inputs(context, original_context, original_artifact, requests,
                             anchor_unit_id, artifact, ob._freeze(body))
            with self._lock:
                self._inputs[key] = inputs
            return artifact
        finally:
            with self._lock:
                self._inflight.discard(key)

    def _original(self, inputs: _Inputs, store: mc.ArtifactStore):
        original = self._owning_issuer._replay(inputs.original_context, store)
        scientific._same(original.original_artifact.to_dict(), inputs.original_artifact.to_dict(),
                         "original owning registry receipt")
        scientific._same(wl._digest(ob._plain(original.original_body)),
                         inputs.body["original_t0_digest"], "original owning full body")
        body = store.read(inputs.artifact.locator, inputs.artifact.sha256)
        scientific._same(body, inputs.body, "original server input artifact")
        store.verify(f"parent-server-t0-inputs:{wl._digest(ob._plain(body))}", body)
        scientific._same(self.source_identity(), body["source"], "original selected server T0 source")
        scientific._same(scientific._project(_original_identity(original, inputs.context)),
                         body["original_identity"], "original server identity")
        model_before, model_after = (original.original_body[name]
                                     for name in ("model_before", "model_after"))
        if model_before != model_after or model_after["error"] is not None:
            raise scientific.ScientificWitnessRefused("original owning model inventory is unproven")
        scientific._same(self.owning_issuer._model_fact(inputs.context, store), model_after,
                         "original complete model inventory continuity")
        before, after = (original.original_body[name]
                        for name in ("native_artifacts_before", "native_artifacts_after"))
        if before != after or any(item["error"] is not None or item["observed"] is None
                or item["expected"]["sha256"] != item["observed"]["sha256"] for item in before):
            raise scientific.ScientificWitnessRefused("original native executable/DSO identity is unproven")
        return original

    def _observe(self, context, native, lifecycle_link, runtime_readbacks, store):
        from .native_parent_evidence import NativeUnitEvidenceProducer
        native = ob._plain(native)
        verifier = NativeUnitEvidenceProducer(store=store, context=context)
        verifier._join_native(native)
        reference = lifecycle_link.reference
        observation = lo.validate_observation(ob._plain(store.read(
            reference.artifact.locator, reference.artifact.sha256)))
        verifier._join_observation(native, observation, reference)
        selected = native["selected_observation"]
        pid = context.descendant_event["data"]["process"]["pid"]
        scientific._same(selected["process_pid"], pid, "actual server target PID")
        requests = context.prompts.requests(context.unit.expected_prompt_ids, context.template)
        rows = raw.reopen_unit(selected.get("server_responses"), store=store,
            expected_frame=raw._frame(context.plan, context.unit, context.fence,
                                      context.recipe, context.prompts),
            expected_requests=requests, expected_pid=pid)
        receipt = selected["server_responses"]
        if not (observation["started_monotonic_s"] <= receipt["request_started_monotonic_s"]
                <= receipt["request_ended_monotonic_s"] <= receipt["retention_started_monotonic_s"]
                <= receipt["retention_ended_monotonic_s"] <= observation["ended_monotonic_s"]):
            raise scientific.ScientificWitnessRefused("server response escaped original lifecycle")
        boundaries = {name: [item["monotonic_s"] for item in observation["phase_boundaries"]
                            if item["phase"] == name] for name in ("warmup", "measurement", "teardown")}
        if any(len(items) != 1 for items in boundaries.values()):
            raise scientific.ScientificWitnessRefused("server phase boundaries are incomplete or ambiguous")
        for item in rows:
            phase = item["raw"]["phase"]
            start = boundaries[phase][0]
            end = boundaries["measurement" if phase == "warmup" else "teardown"][0]
            if not start <= item["raw"]["started_monotonic_s"] <= item["raw"]["ended_monotonic_s"] <= end:
                raise scientific.ScientificWitnessRefused("server response phase interval differs")
        measurement_ends = [sample["marker_monotonic_s"] for sample in observation["samples"]
            if sample["kind"] == "checkpoint" and sample["marker_label"] == "measurement_end"]
        instrument = lo.validate_instrument_identity(ob._plain(store.read(
            context.binding.instrument.artifact.locator, context.binding.instrument.artifact.sha256)))
        scientific._same(instrument["sha256"], context.binding.instrument.identity_sha256,
                         "original marker capability instrument")
        markers = instrument["used_constants"].get("parent_window_markers")
        if markers is not None:
            scientific._same(markers, ["health", "warmup", "measurement", "measurement_end", "teardown"],
                             "supported original parent window markers")
            if not measurement_ends:
                raise scientific.ScientificWitnessRefused("required original measurement_end is missing")
        if measurement_ends:
            if len(measurement_ends) != 1 or not boundaries["measurement"][0] <= measurement_ends[0] <= boundaries["teardown"][0]:
                raise scientific.ScientificWitnessRefused("original measurement_end is ambiguous or out of phase")
            if any(row["raw"]["ended_monotonic_s"] > measurement_ends[0] for row in rows
                   if row["raw"]["phase"] == "measurement"):
                raise scientific.ScientificWitnessRefused("server response exceeds original measurement_end")
        artifact = store.verify(f"raw:{native['artifact_digest']}", native)
        observed = _Observed(context, artifact, lifecycle_link, tuple(runtime_readbacks))
        return observed, rows

    @staticmethod
    def _generation(row) -> sg.ServerGenerationEvidence:
        request, response, captured = row["request"], row["response"], row["raw"]
        error, content, count = captured["error"], None, None
        if isinstance(response, Mapping):
            timings = response.get("timings")
            value = timings.get("predicted_n") if isinstance(timings, Mapping) else None
            count = value if type(value) is int and value >= 0 else None
        if error is None:
            if isinstance(response, Mapping) and response.get("error") is not None:
                error = "original server response contains an explicit error"
            elif not isinstance(response, Mapping) or type(response.get("content")) is not str:
                error = "original response has no content string"
            elif response.get("stop") is not True or count != request["n_predict"]:
                error = "original response is not exact terminal requested work"
            else:
                settings = response.get("generation_settings")
                if settings is not None and (not isinstance(settings, Mapping)
                        or any(key in settings and (type(settings[key]) is bool
                               or settings[key] != request[key])
                               for key in ("temperature", "top_k", "top_p", "n_predict"))):
                    error = "original response sampling settings contradict frozen request"
                else:
                    content = response["content"]
        return sg.ServerGenerationEvidence(prompt=request["prompt"], prompt_ref=captured["prompt_id"],
            n_predict=request["n_predict"], temperature=request["temperature"], top_k=request["top_k"],
            slot_index=captured["slot_index"], request_sha256=captured["request_sha256"],
            response_sha256=captured["response_sha256"], content=content, delivered_n=count,
            receipt_ref=f"{row['artifact']['locator']}#{row['artifact']['sha256']}", error=error)

    def _anchor(self, observed: _Observed | None, request, generation, store):
        if observed is None:
            return None, None
        key = (observed.context.plan.digest, observed.context.unit_id)
        with self._lock:
            inputs = self._inputs.get(key)
        if inputs is None:
            raise scientific.ScientificWitnessRefused("original server anchor input issuance is lost")
        original = self._original(inputs, store)
        native = store.read(observed.native_artifact.locator, observed.native_artifact.sha256)
        _, rows = self._observe(observed.context, native, observed.lifecycle_link,
                                observed.runtime_readbacks, store)
        selected = [row for row in rows if row["raw"]["phase"] == "measurement"
                    and row["raw"]["slot_index"] == generation.slot_index]
        if len(selected) != 1:
            raise scientific.ScientificWitnessRefused("original server anchor slot is missing or duplicated")
        anchor_generation = self._generation(selected[0])
        scientific._same((anchor_generation.prompt, anchor_generation.prompt_ref,
            anchor_generation.request_sha256, anchor_generation.n_predict),
            (generation.prompt, generation.prompt_ref, generation.request_sha256, generation.n_predict),
            "original anchor exact prompt/request/slot")
        identity = _original_identity(original, observed.context)
        scientific._same(request.anchor, identity, "original server anchor full identity")
        content = anchor_generation.content
        anchor = t0.AnchorCapture(source_commit=identity.source_commit,
            binary_sha256=identity.binary_sha256, linkage_sha256=identity.linkage_sha256,
            output_digests=() if content is None or not content else (t0.sha256_text(content),),
            output_lengths=() if content is None or not content else (len(content),),
            determinism_class="not_measured", delivered_units=anchor_generation.delivered_n,
            capture_refs=(anchor_generation.receipt_ref,))
        return anchor, anchor_generation

    @staticmethod
    def _reduce(original, request, generation, anchor):
        coherence = sg.collect_server_coherence(generation, anchor)
        surface = original.evidence.change_surface if original is not None else _unknown_surface()
        # No old dynamic gate is copied, including CLI linkage/coherence/dispatch.
        dynamic = correctness.DynamicT0Evidence(control_role=None, change_surface=surface,
            op_suite=None, reference=None, boundary_shapes=None, dispatch_trace=None,
            coherence=coherence, determinism=None, linkage=None, anti_reward_hacking=None)
        bundle = None
        if original is not None and not original.evidence.source_candidate:
            original_report = correctness.T0CorrectnessRunner(
                provider=correctness.StaticEvidenceProvider({original.request.candidate_id: original.evidence}),
                policy=original.policy).evaluate(original.request)
            # The owning bundle contract describes a complete passing source/build
            # report. Do not turn a partially passing old report into static authority.
            if not (original_report.failed or original_report.unevaluated or original_report.demoted_gates):
                bundle = correctness.seal_static_t0_bundle(original.request, original_report)
        if bundle is not None:
            report = correctness.evaluate_t0_with_static_bundle(request, bundle, dynamic, original.policy)
            return report, scientific._project(dynamic), bundle.to_dict()
        evidence = correctness.T0Evidence(control_role=None, change_surface=surface,
            symbols=None, build=None, diff=None, static_analysis=None, sanitizers=None,
            op_suite=None, reference=None, boundary_shapes=None, dispatch_trace=None,
            state_safety=None, coherence=coherence, determinism=None, linkage=None,
            anti_reward_hacking=None, source_candidate=original.evidence.source_candidate)
        report = correctness.T0CorrectnessRunner(
            provider=correctness.StaticEvidenceProvider({request.candidate_id: evidence}),
            policy=original.policy).evaluate(request)
        return report, scientific._project(evidence), None

    def _build(self, observed: _Observed, anchor: _Observed | None, store: mc.ArtifactStore):
        key = (observed.context.plan.digest, observed.context.unit_id)
        with self._lock:
            inputs = self._inputs.get(key)
        if inputs is None:
            raise scientific.ScientificWitnessRefused("original server T0 input registry is lost")
        original = self._original(inputs, store)
        native = store.read(observed.native_artifact.locator, observed.native_artifact.sha256)
        _, rows = self._observe(observed.context, native, observed.lifecycle_link,
                                observed.runtime_readbacks, store)
        measurement = tuple(row for row in rows if row["raw"]["phase"] == "measurement")
        if len(measurement) != len(inputs.requests):
            raise scientific.ScientificWitnessRefused("full original server slot set differs")
        if anchor is not None:
            scientific._same(anchor.context.plan.digest, observed.context.plan.digest, "anchor original plan")
            scientific._same(anchor.context.recipe.model.to_dict(),
                             observed.context.recipe.model.to_dict(), "anchor original model entry")
            scientific._same(anchor.context.unit_id, inputs.anchor_unit_id, "selected original anchor unit")
            with self._lock:
                anchor_inputs = self._inputs.get((anchor.context.plan.digest, anchor.context.unit_id))
            if anchor_inputs is None:
                raise scientific.ScientificWitnessRefused("original anchor inputs are lost")
            anchor_original = self._original(anchor_inputs, store)
            scientific._same(anchor_original.original_body["model_after"],
                             original.original_body["model_after"], "full original anchor model inventory")
        defects = [f"{row['raw']['phase']} slot {row['raw']['slot_index']}: {generation.error}"
                   for row in rows if (generation := self._generation(row)).error is not None]
        if native.get("error") is not None or native["selected_observation"].get("failure") is not None:
            defects.append("native unit records an original execution failure")
        slots = []
        for index, (request, row) in enumerate(zip(inputs.requests, measurement)):
            generation = self._generation(row)
            scientific._same(generation.slot_index, index, "complete server slot order")
            anchor_capture, anchor_generation = self._anchor(anchor, request, generation, store)
            report, evidence, bundle = self._reduce(original, request, generation, anchor_capture)
            slots.append({"slot_index": index, "prompt_id": generation.prompt_ref,
                "generation": scientific._project(generation),
                "anchor_generation": scientific._project(anchor_generation),
                "request": scientific._project(request), "evidence": evidence,
                "static_bundle": bundle, "report": report.to_dict()})
        failed = any(slot["report"]["failed"] for slot in slots)
        unknown = any(slot["report"]["unevaluated"] or slot["report"]["demoted_gates"]
                      or not slot["report"]["anchor_bound"] for slot in slots)
        report = {"slots": slots, "gate_ids": list(correctness.T0_GATE_IDS),
                  "coverage": REPLAY_COVERAGE}
        return {"schema": RECEIPT_SCHEMA, "witness": "correctness", "adapter_id": ADAPTER_ID,
            "frame": ob._plain(scientific._frame(observed.context)),
            "native_observation": observed.native_artifact.to_dict(),
            "lifecycle_reference": observed.lifecycle_link.reference.to_dict(),
            "runtime_readbacks": [item.to_dict() for item in observed.runtime_readbacks],
            "original_owning_inputs": inputs.artifact.to_dict(),
            "original_anchor_native": None if anchor is None else anchor.native_artifact.to_dict(),
            "source": self.source_identity(), "report": report, "scope_defects": defects,
            "status": "fail" if failed or defects else "unknown" if unknown else "pass",
            "replay_coverage": REPLAY_COVERAGE}

    def evaluate(self, context, native, lifecycle_link, runtime_readbacks, *, store):
        if type(store) is not mc.ArtifactStore:
            raise scientific.ScientificWitnessRefused("concrete original artifact store required")
        key = (context.plan.digest, context.unit_id)
        with self._lock:
            inputs, issued = self._inputs.get(key), self._issued.get(key)
        if inputs is None:
            raise scientific.ScientificWitnessRefused("original server T0 inputs unavailable")
        scientific._same(scientific._frame(context), scientific._frame(inputs.context), "original unit frame")
        observed, _ = self._observe(context, native, lifecycle_link, runtime_readbacks, store)
        if issued is not None:
            scientific._same(observed.native_artifact.to_dict(), issued.observed.native_artifact.to_dict(),
                             "duplicate native original response")
            scientific._same(observed.lifecycle_link.to_dict(), issued.observed.lifecycle_link.to_dict(),
                             "duplicate original lifecycle link")
            scientific._same([item.to_dict() for item in observed.runtime_readbacks],
                [item.to_dict() for item in issued.observed.runtime_readbacks],
                "duplicate original runtime readbacks")
            scientific._same(self._build(issued.observed, issued.anchor, store), issued.body,
                             "complete original server T0 replay")
            return issued.reference
        with self._lock:
            if key in self._inflight:
                raise scientific.ScientificWitnessRefused("server T0 unit is already in flight")
            self._inflight.add(key)
            self._observed[key] = observed
            anchor = None if inputs.anchor_unit_id is None else self._observed.get(
                (context.plan.digest, inputs.anchor_unit_id))
        try:
            body = self._build(observed, anchor, store)
            digest = wl._digest(body)
            reference = scientific.WitnessEvidenceReference(
                store.write(f"parent-scientific-witness:{digest}", body), digest)
            with self._lock:
                self._issued[key] = _Issued(observed, anchor, reference, ob._freeze(body))
            return reference
        finally:
            with self._lock:
                self._inflight.discard(key)

    def reopen(self, reference, *, context, native, lifecycle_link, runtime_readbacks, store):
        if type(reference) is not scientific.WitnessEvidenceReference:
            raise scientific.ScientificWitnessRefused("concrete server T0 witness reference required")
        body = store.read(reference.artifact.locator, reference.artifact.sha256)
        scientific._same(wl._digest(ob._plain(body)), reference.digest, "server T0 receipt digest")
        store.verify(f"parent-scientific-witness:{reference.digest}", body)
        expected = self.evaluate(context, native, lifecycle_link, runtime_readbacks, store=store)
        scientific._same(reference.to_dict(), expected.to_dict(), "original server T0 receipt replay")
        return ob._freeze(ob._plain(body))

    def _pair_body(self, *, plan, registry, store):
        from . import native_parent_receipt_replay as replay
        if type(registry) is not replay.IssuedNativeEvidenceRegistry:
            raise scientific.ScientificWitnessRefused("original concrete parent registry required")
        entries = registry.snapshot(plan=plan, store=store)
        ordered = []
        for entry in entries:
            context = entry.context
            if entry.scientific_adapters is None or entry.scientific_adapters.correctness is not self:
                raise scientific.ScientificWitnessRefused("pair selected a different original scientific issuer")
            key = (plan.digest, context.unit_id)
            with self._lock:
                issued, inputs = self._issued.get(key), self._inputs.get(key)
            if issued is None or inputs is None:
                raise scientific.ScientificWitnessRefused("pair lacks original server unit issuance")
            native = store.read(issued.observed.native_artifact.locator,
                                issued.observed.native_artifact.sha256)
            original = self.reopen(issued.reference, context=context, native=native,
                lifecycle_link=issued.observed.lifecycle_link,
                runtime_readbacks=issued.observed.runtime_readbacks, store=store)
            finding = entry.body["findings"].get("correctness")
            if not isinstance(finding, Mapping):
                raise scientific.ScientificWitnessRefused("original parent scientific finding unavailable")
            scientific._same(finding.get("facts", {}).get("receipt"), issued.reference.to_dict(),
                             "original parent scientific issuance")
            scientific._same(entry.body["native_observation"], issued.observed.native_artifact.to_dict(),
                             "original parent native observation")
            scientific._same(entry.body["lifecycle_observation"],
                issued.observed.lifecycle_link.reference.to_dict(), "original parent lifecycle")
            scientific._same(entry.body["runtime_readbacks"],
                [item.to_dict() for item in issued.observed.runtime_readbacks], "original parent readbacks")
            anchor = None
            if inputs.anchor_unit_id is not None:
                with self._lock:
                    anchor_issued = self._issued.get((plan.digest, inputs.anchor_unit_id))
                if anchor_issued is None:
                    raise scientific.ScientificWitnessRefused("selected complete original anchor is unavailable")
                anchor = anchor_issued.observed
            final = self._build(issued.observed, anchor, store)
            owning = self._original(inputs, store)
            ordered.append({"unit_id": context.unit_id, "arm": context.unit.arm,
                "process_generation_id": context.unit.process_id, "pair_id": context.unit.pair_id,
                "order_index": context.unit.order_index, "original_frame": ob._plain(scientific._frame(context)),
                "parent_unit_receipt": entry.result.receipt.to_dict(),
                "parent_unit_receipt_digest": entry.result.receipt_digest,
                "original_unit_scientific_receipt": issued.reference.to_dict(),
                "original_unit_status": original["status"], "original_inputs": inputs.artifact.to_dict(),
                "original_owning_t0": inputs.original_artifact.to_dict(),
                "original_model_preparation": ob._plain(owning.original_body["model_after"]),
                "selected_anchor_unit_id": inputs.anchor_unit_id,
                "final_evidence": final})
        return {"schema": PAIR_SCHEMA, "plan_digest": plan.digest, "plan": plan.to_dict(),
                "instrument_reference": ob._plain(plan.loaded_instrument),
                "adapter_source_identity": self.source_identity(),
                "original_issuer_source": self.owning_issuer.source_identity(),
                "ordered_units": ordered, "coverage": REPLAY_COVERAGE}

    def finalize_pair(self, *, plan, registry, store) -> ParentServerT0PairReference:
        # No wait for future units. snapshot refuses unless every original unit exists.
        key = plan.digest
        with self._lock:
            prior = self._pairs.get(key)
            if key in self._pair_inflight:
                raise scientific.ScientificWitnessRefused("pair finalization already in flight")
            if prior is None and len(self._pairs) >= self.max_units:
                raise scientific.ScientificWitnessRefused("pair registry capacity exhausted")
            self._pair_inflight.add(key)
        try:
            body = self._pair_body(plan=plan, registry=registry, store=store)
            if prior is not None:
                scientific._same(body, prior[1], "immutable final pair replay")
                reopened = store.read(prior[0].artifact.locator, prior[0].artifact.sha256)
                scientific._same(reopened, body, "original final pair bytes")
                store.verify(f"parent-server-t0-pair:{prior[0].digest}", body)
                return prior[0]
            digest = wl._digest(body)
            reference = ParentServerT0PairReference(
                store.write(f"parent-server-t0-pair:{digest}", body), digest)
            with self._lock:
                self._pairs[key] = (reference, ob._freeze(body))
            return reference
        finally:
            with self._lock:
                self._pair_inflight.discard(key)

    def reopen_pair(self, reference, *, plan, registry, store):
        if type(reference) is not ParentServerT0PairReference:
            raise scientific.ScientificWitnessRefused("concrete scientific pair reference required")
        with self._lock:
            original = self._pairs.get(plan.digest)
        if original is None:
            raise scientific.ScientificWitnessRefused("original pair issuance is lost")
        scientific._same(reference.to_dict(), original[0].to_dict(), "original pair reference")
        expected = self.finalize_pair(plan=plan, registry=registry, store=store)
        scientific._same(reference.to_dict(), expected.to_dict(), "replayed pair reference")
        return ob._freeze(ob._plain(original[1]))

    def reopen_pair_reports(self, reference, *, plan, registry, store):
        """Return actual owning T0Report objects, not deserialized report assertions."""
        body = self.reopen_pair(reference, plan=plan, registry=registry, store=store)
        reports = {}
        for item in body["ordered_units"]:
            key = (plan.digest, item["unit_id"])
            with self._lock:
                inputs, issued = self._inputs[key], self._issued[key]
                anchor_issued = (None if inputs.anchor_unit_id is None else
                                self._issued[(plan.digest, inputs.anchor_unit_id)])
            observed = issued.observed
            original = self._original(inputs, store)
            native = store.read(observed.native_artifact.locator, observed.native_artifact.sha256)
            _, rows = self._observe(observed.context, native, observed.lifecycle_link,
                                    observed.runtime_readbacks, store)
            measured = [row for row in rows if row["raw"]["phase"] == "measurement"]
            unit_reports = []
            for index, (request, row) in enumerate(zip(inputs.requests, measured)):
                generation = self._generation(row)
                anchor, _ = self._anchor(None if anchor_issued is None else anchor_issued.observed,
                                         request, generation, store)
                report, _evidence, _bundle = self._reduce(original, request, generation, anchor)
                scientific._same(report.to_dict(),
                    item["final_evidence"]["report"]["slots"][index]["report"],
                    "original final owning T0Report")
                unit_reports.append(report)
            reports[item["unit_id"]] = tuple(unit_reports)
        from types import MappingProxyType
        return MappingProxyType(reports)

    def source_identity(self) -> Mapping[str, Any]:
        from .native_final_trial import NativeFinalTrialOwner
        if type(self.owning_issuer) is not scientific.NativeT0WitnessAdapter:
            raise scientific.ScientificWitnessRefused("selected original issuer type changed")
        if type(self.final_trial_owner) is not NativeFinalTrialOwner or self.final_trial_owner.correctness_adapter is not self:
            raise scientific.ScientificWitnessRefused("selected final trial owner changed")
        return {"adapter_id": ADAPTER_ID,
            "configuration": {"max_units": self.max_units,
                              "owning_issuer_max_units": self.owning_issuer.max_units},
            "callables": [{"role": name, "identity": lo.callable_identity(getattr(type(self), name))}
                          for name in ADAPTER_ROLES],
            "helpers": [{"role": name, "identity": lo.callable_identity(globals()[name])}
                        for name in HELPER_ROLES],
            "owning_issuer": self.owning_issuer.source_identity(),
            "owning_source_pins": ob._plain(_source_pins())}


def validate_server_source(value: Any) -> Mapping[str, Any]:
    """Closed historical v1 server adapter grammar, not today's source equality."""
    from .native_producer_source import _closed, _callables, _identity
    row = _closed(value, ("adapter_id", "configuration", "callables", "helpers",
                         "owning_issuer", "owning_source_pins"), "server T0 adapter source")
    if row["adapter_id"] != ADAPTER_ID:
        raise scientific.ScientificWitnessRefused("unsupported server T0 adapter ID")
    config = _closed(row["configuration"], ("max_units", "owning_issuer_max_units"),
                     "server T0 configuration")
    if any(type(value) is not int or not 1 <= value <= MAX_UNITS for value in config.values()):
        raise scientific.ScientificWitnessRefused("server T0 configuration bound is invalid")
    row["configuration"] = config
    row["callables"] = _callables(row["callables"], ADAPTER_ROLES)
    row["helpers"] = _callables(row["helpers"], HELPER_ROLES)
    owner = scientific.validate_scientific_source({"schema": scientific.ADAPTERS_SCHEMA,
        "correctness": row["owning_issuer"], "purpose": None, "contention": None, "residency": None})
    if owner["correctness"] is None or owner["correctness"]["configuration"]["max_units"] != config["owning_issuer_max_units"]:
        raise scientific.ScientificWitnessRefused("original T0 issuer configuration differs")
    row["owning_issuer"] = owner["correctness"]
    pins = _closed(row["owning_source_pins"], ("modules", "callables", "raw_source", "final_trial_source"),
                   "server T0 owning source pins")
    if type(pins["modules"]) not in (list, tuple) or len(pins["modules"]) != 6:
        raise scientific.ScientificWitnessRefused("server T0 source module set differs")
    pins["modules"] = [lo._validate_artifact(item) for item in pins["modules"]]
    if type(pins["callables"]) not in (list, tuple) or len(pins["callables"]) != 19:
        raise scientific.ScientificWitnessRefused("server T0 owning callable set differs")
    pins["callables"] = [_identity(item) for item in pins["callables"]]
    retained = _closed(pins["raw_source"], ("schema", "max_response_bytes", "max_request_bytes",
        "max_slots", "max_total_raw_bytes", "callables"), "original raw server source")
    if retained["schema"] != raw.RESPONSE_SCHEMA or any(type(retained[name]) is not int
            or retained[name] <= 0 for name in ("max_response_bytes", "max_request_bytes",
                                               "max_slots", "max_total_raw_bytes")):
        raise scientific.ScientificWitnessRefused("original raw server source configuration differs")
    if type(retained["callables"]) not in (list, tuple) or len(retained["callables"]) != 14:
        raise scientific.ScientificWitnessRefused("original raw server callable set differs")
    retained["callables"] = [_identity(item) for item in retained["callables"]]
    pins["raw_source"] = retained
    from .native_final_trial import validate_source_identity
    pins["final_trial_source"] = validate_source_identity(pins["final_trial_source"])
    row["owning_source_pins"] = pins
    return ob._freeze(ob._plain(row))


def server_source_identities(value: Any) -> tuple[Mapping[str, Any], ...]:
    from .native_final_trial import source_identities
    row = validate_server_source(value)
    owner = {"schema": scientific.ADAPTERS_SCHEMA, "correctness": row["owning_issuer"],
             "purpose": None, "contention": None, "residency": None}
    return (tuple(item["identity"] for item in (*row["callables"], *row["helpers"]))
            + tuple(row["owning_source_pins"]["callables"])
            + tuple(row["owning_source_pins"]["raw_source"]["callables"])
            + source_identities(row["owning_source_pins"]["final_trial_source"])
            + scientific.scientific_source_identities(owner))
