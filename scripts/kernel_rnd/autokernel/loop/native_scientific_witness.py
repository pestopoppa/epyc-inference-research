"""Parent-issued T0 collection and deterministic owning-reducer replay.

No child document can construct an issuance. Initial collection uses the actual
owning provider under its original held claim. Replay reopens the original raw
bytes and rescales no data: StaticEvidenceProvider/T0CorrectnessRunner rescore
the immutable, originally collected evidence. This is not independent raw-output
re-parsing, a fresh observation, or a transfer between different tool identities.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

from .. import schemas
from ..evaluator import api, correctness
from ..evaluator import c3_epyc_tensor_capture as tensor_capture
from ..execution import t0_provider as t0
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import observation_binding as ob
from . import worker_lifecycle as wl

ADAPTERS_SCHEMA = "epyc.autokernel.parent_scientific_witness_adapters.v1"
ADAPTERS_SCHEMA_V2 = "epyc.autokernel.parent_scientific_witness_adapters.v2"
T0_SCHEMA = "epyc.autokernel.parent_issued_t0_evidence.v1"
WITNESS_SCHEMA = "epyc.autokernel.parent_scientific_witness.v1"
ADAPTER_ID = "epyc.autokernel.native_t0_witness.v1"
SLOTS = ("correctness", "purpose", "contention", "residency")
ADAPTER_ROLES = ("prepare_model_identity", "_model_fact", "collect_issued", "_collect",
                 "_replay", "evaluate", "_scope_defects", "reopen")
MODEL_SCHEMA = "epyc.autokernel.parent_verified_model_inventory.v1"
MAX_CAPTURES = 4096
MAX_UNITS = 1024
REPLAY_COVERAGE = "original_collection_identity_and_owning_reducer_replay"


class ScientificWitnessRefused(ValueError):
    pass


def _same(actual: Any, expected: Any, label: str) -> None:
    if _project(actual) != _project(expected):
        raise ScientificWitnessRefused(f"{label} differs from original parent issuance")


def _project(value: Any) -> Any:
    """Closed owning-record projection; never repr(), imports, or object identity."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ScientificWitnessRefused("T0 mapping keys must be text")
        return {key: _project(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_project(item) for item in value]
    if type(value) is frozenset and all(type(item) is str for item in value):
        return {"frozen_string_set": sorted(value)}
    if is_dataclass(value) and not isinstance(value, type):
        if not type(value).__module__.startswith("autokernel."):
            raise ScientificWitnessRefused("T0 record is not an installed owning record")
        return {"record_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {field.name: _project(getattr(value, field.name))
                           for field in fields(value)}}
    raise ScientificWitnessRefused("unsupported mutable or noncanonical T0 input")


def _immutable(value: Any) -> Any:
    # Preserve the actual validated owning type, recursively copying every field.
    # There is deliberately no inverse from untrusted JSON into these objects.
    _project(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if type(value) in (tuple, list):
        return tuple(_immutable(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        if not type(value).__dataclass_params__.frozen:
            raise ScientificWitnessRefused("owning T0 records must be frozen dataclasses")
        result = object.__new__(type(value))
        for field in fields(value):
            object.__setattr__(result, field.name, _immutable(getattr(value, field.name)))
        return result
    return value


def _source_functions() -> tuple[Any, ...]:
    return (t0.ExecutedT0EvidenceProvider.evidence_for,
        t0.ExecutedT0EvidenceProvider._execute, t0.SubprocessRunner.run,
        t0.capture_anchor, t0._complete_anchor_toolchain,
        t0.ExecutedT0EvidenceProvider.collect_op_suite,
        t0.ExecutedT0EvidenceProvider.collect_boundary_shapes,
        t0.ExecutedT0EvidenceProvider.collect_dispatch_trace,
        t0.ExecutedT0EvidenceProvider.collect_linkage,
        t0.ExecutedT0EvidenceProvider.collect_sanitizers,
        t0.ExecutedT0EvidenceProvider.collect_coherence,
        t0.ExecutedT0EvidenceProvider.collect_determinism,
        t0.ExecutedT0EvidenceProvider.collect_static_analysis,
        t0.ExecutedT0EvidenceProvider.collect_state_safety,
        t0.ExecutedT0EvidenceProvider.collect_anti_reward_hacking,
        correctness.StaticEvidenceProvider.evidence_for,
        correctness.T0CorrectnessRunner.evaluate, correctness.evaluate_t0,
        correctness.check_symbol_and_registration_preservation,
        correctness.check_clean_build_from_snapshot, correctness.check_semantic_diff_conformance,
        correctness.check_schema_and_diff_policy, correctness.check_static_and_compile,
        correctness.check_asan, correctness.check_ubsan, correctness.check_backend_op_units,
        correctness.check_exact_reference_comparison, correctness.check_unseen_boundary_shapes,
        correctness.check_affected_surface_reconciliation, correctness.check_no_fallback_dispatch_proof,
        correctness.check_state_rollback_teardown_race, correctness.check_output_coherence,
        correctness.check_determinism_class, correctness.check_binary_and_linkage_identity,
        correctness.check_anti_reward_hacking, correctness.demote_anchor_requiring_passes,
        tensor_capture.CaptureModelIdentity.validate, tensor_capture._checked_regular_file,
        tensor_capture._sha256_file, _project, _immutable, _frame, _CaptureRecorder.run,
        _native_artifact_facts, _file_stats)


def _source_pins() -> Mapping[str, Any]:
    # Explicit loaded collectors and every top-level owning gate. Module receipts
    # retain supporting source; this does not claim a transitive loaded call graph.
    return ob._freeze({"modules": [lo.prepare_artifact_identity(Path(module.__file__))
                       for module in (t0, correctness, api, schemas, tensor_capture)] +
                      [lo.prepare_artifact_identity(Path(__file__))],
        "callables": [lo.callable_identity(function) for function in _source_functions()]})


def _native_artifact_facts(context: Any) -> tuple[Mapping[str, Any], ...]:
    facts = []
    # Model inventory is verified once by prepare_model_identity, outside units.
    for artifact in (context.recipe.executable, *context.recipe.dsos):
        try:
            observed = lo.prepare_artifact_identity(Path(artifact.path))
            facts.append({"expected": artifact.to_dict(), "observed": observed,
                          "error": None})
        except (OSError, lo.ObservationError) as exc:
            facts.append({"expected": artifact.to_dict(), "observed": None,
                          "error": type(exc).__name__})
    return tuple(ob._freeze(row) for row in facts)


def _file_stats(paths: tuple[str, ...]) -> tuple[Mapping[str, Any], ...]:
    """Continuity with an original hash receipt; these fields are NOT content hashes."""
    rows = []
    for name in paths:
        path = Path(name)
        info = path.lstat()
        if path.is_symlink() or not path.is_file():
            raise ScientificWitnessRefused("verified inventory member is no longer a regular file")
        rows.append({"path": name, "dev": info.st_dev, "ino": info.st_ino,
            "size": info.st_size, "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "mode": info.st_mode, "nlink": info.st_nlink})
    return tuple(ob._freeze(row) for row in rows)


def _model_source_pins() -> Mapping[str, Any]:
    return ob._freeze({"module": lo.prepare_artifact_identity(Path(tensor_capture.__file__)),
        "callables": [lo.callable_identity(function) for function in (
            tensor_capture.CaptureModelIdentity.validate, tensor_capture._checked_regular_file,
            tensor_capture._sha256_file, _file_stats)]})


def _frame(context: Any) -> Mapping[str, Any]:
    from .native_parent_evidence import ParentUnitContext
    if type(context) is not ParentUnitContext:
        raise ScientificWitnessRefused("concrete original parent unit context required")
    return ob._freeze({"identity": context.identity, "arm": context.unit.arm,
        "process_generation_id": context.unit.process_id,
        "worker_binding": ob._plain(context.binding.worker_binding),
        "container_id": context.binding.container_id,
        "active_claim": ob._plain(context.active_claim),
        "recipe": context.recipe.to_dict(), "prompts": context.prompts.to_dict(),
        "expected_prompt_ids": list(context.unit.expected_prompt_ids),
        "parent_descendant_event": ob._plain(context.descendant_event)})


@dataclass(frozen=True)
class WitnessEvidenceReference:
    artifact: mc.StoredArtifact
    digest: str
    witness: str = "correctness"
    schema: str = WITNESS_SCHEMA

    def __post_init__(self) -> None:
        if (type(self.artifact) is not mc.StoredArtifact or self.witness != "correctness"
                or self.schema != WITNESS_SCHEMA):
            raise ScientificWitnessRefused("unsupported scientific witness reference")
        wl._sha(self.digest, "scientific witness digest")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "witness": self.witness,
                "artifact": self.artifact.to_dict(), "digest": self.digest}


class _CaptureRecorder:
    """A fixed recorder around the owning live runner, never a verifier callback."""
    def __init__(self, runner: t0.SubprocessRunner, store: mc.ArtifactStore) -> None:
        if type(runner) is not t0.SubprocessRunner:
            raise ScientificWitnessRefused("initial T0 collection requires the installed live runner")
        self.runner, self.store = runner, store
        self.records: list[Mapping[str, Any]] = []

    def run(self, argv, *, env, cwd, timeout_s):
        if len(self.records) >= MAX_CAPTURES:
            raise ScientificWitnessRefused("original T0 capture bound exceeded")
        requested = {"argv": list(argv), "env": sorted([list(item) for item in env.items()]),
                     "cwd": cwd, "timeout_s": timeout_s}
        capture = self.runner.run(argv, env=env, cwd=cwd, timeout_s=timeout_s)
        if type(capture) is not t0.CompletedProcess:
            raise ScientificWitnessRefused("owning runner did not return a complete capture")
        _same(capture.argv, argv, "actual T0 argv")
        _same(capture.cwd, cwd, "actual T0 cwd")
        # This version selects the ordinary owning runner, not a sandbox wrapper.
        # A future sandbox selection must specify its exact env transformation.
        _same(sorted(capture.env), sorted(env.items()), "actual T0 environment")
        body = {"schema": "epyc.autokernel.parent_t0_raw_capture.v1",
                "sequence": len(self.records), "invocation": requested,
                "capture": capture.to_dict(), "capture_ref": t0.capture_ref(capture)}
        ref = self.store.write(f"parent-t0-raw:{wl._digest(body)}", body)
        self.records.append(ob._freeze({"artifact": ref.to_dict(), "body": body}))
        return capture


@dataclass(frozen=True)
class _IssuedT0:
    frame: Mapping[str, Any]
    plan: t0.T0ExecutionPlan
    request: api.EvaluationRequest
    policy: correctness.T0Policy
    anchor: t0.AnchorCapture | None
    evidence: correctness.T0Evidence
    report: Mapping[str, Any]
    raw: tuple[Mapping[str, Any], ...]
    source_pins: Mapping[str, Any]
    original_artifact: mc.StoredArtifact
    original_body: Mapping[str, Any]


class NativeT0WitnessAdapter:
    """One parent-owned bounded issuance registry, with no JSON restore API.

    collect_issued runs live owning collection and must be scheduled by the claim
    owner. evaluate/reopen never run processes and work after claim release. A
    new adapter instance cannot recover authority from an artifact's presence.
    """
    def __init__(self, *, max_units: int = MAX_UNITS) -> None:
        if type(max_units) is not int or not 1 <= max_units <= MAX_UNITS:
            raise ScientificWitnessRefused("invalid T0 issuance registry bound")
        self._max_units = max_units
        self._lock = threading.RLock()
        self._issued: dict[tuple[str, str], _IssuedT0] = {}
        self._inflight: dict[tuple[str, str], str] = {}
        self._failed: set[tuple[str, str]] = set()
        self._models: dict[tuple[str, str], tuple[mc.StoredArtifact, Mapping[str, Any]]] = {}
        self._models_inflight: set[tuple[str, str]] = set()

    @property
    def max_units(self) -> int:
        return self._max_units

    def prepare_model_identity(self, *, store: mc.ArtifactStore,
                               identity: tensor_capture.CaptureModelIdentity,
                               entry_path: Path, preparation_claim: Any) -> mc.StoredArtifact:
        """Scheduled owner preparation; full hashing happens once, before unit lifecycles.

        The inventory digest and entry-member digest are different named facts.
        The native recipe continues to use its entry-file SHA, never the inventory
        digest. Reuse only checks original receipt/source and stable file metadata.
        """
        if type(store) is not mc.ArtifactStore or type(identity) is not tensor_capture.CaptureModelIdentity:
            raise ScientificWitnessRefused("concrete model inventory/store required")
        claim_id = t0.require_claim(preparation_claim, what="scheduled model identity preparation")
        manifest_bytes = lo._bounded_bytes(Path(identity.model_manifest),
                                           tensor_capture.MAX_JSON_BYTES, "model manifest")
        _same(hashlib.sha256(manifest_bytes).hexdigest(), identity.model_manifest_sha256,
              "original model manifest bytes")
        manifest = json.loads(manifest_bytes)
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "model_path", "files"}
                or manifest["schema"] != "epyc.autokernel.model_identity.v1"
                or not isinstance(manifest["files"], list)
                or not 1 <= len(manifest["files"]) <= MAX_CAPTURES):
            raise ScientificWitnessRefused("model manifest is not a bounded owning inventory")
        root = Path(identity.model_id).absolute()
        _same(str(root), manifest["model_path"], "inventory root")
        members = {}
        for row in manifest["files"]:
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise ScientificWitnessRefused("model inventory member fields differ")
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ScientificWitnessRefused("model member escapes its inventory")
            path = str(root if root.is_file() and row["path"] == "." else root / relative)
            if path in members:
                raise ScientificWitnessRefused("model inventory member is duplicated")
            members[path] = row["sha256"]
        entry = str(Path(entry_path).absolute())
        if entry not in members:
            raise ScientificWitnessRefused("native model entry is absent from original inventory")
        key = (entry, members[entry])
        paths = tuple(sorted(members)) + (str(Path(identity.model_manifest).absolute()),)
        before = _file_stats(paths)
        pins = _model_source_pins()
        with self._lock:
            prior = self._models.get(key)
            if prior is None:
                if key in self._models_inflight:
                    raise ScientificWitnessRefused("model identity preparation is already in flight")
                if len(self._models) + len(self._models_inflight) >= self.max_units:
                    raise ScientificWitnessRefused("model identity registry capacity exhausted")
                self._models_inflight.add(key)
        if prior is not None:
            artifact, body = prior
            _same(body["inventory_identity"], identity.to_dict(), "reused model identity")
            _same(body["source_pins"], pins, "reused owning model verifier")
            _same(body["verified_file_stats"], before, "reused verified model continuity")
            _same(store.read(artifact.locator, artifact.sha256), body, "original model receipt")
            return artifact
        try:
            identity.validate()  # The existing owning complete-inventory/full-byte verifier.
            after = _file_stats(paths)
            _same(after, before, "model files changed during original verification")
            _same(_model_source_pins(), pins, "owning model verifier changed")
            _same(t0.require_claim(preparation_claim, what="model identity preparation close"),
                  claim_id, "model preparation claim")
            body = {"schema": MODEL_SCHEMA, "inventory_identity": identity.to_dict(),
                "manifest": manifest, "manifest_raw_utf8": manifest_bytes.decode("utf-8"),
                "entry_path": entry, "entry_sha256": members[entry],
                "inventory_sha256": identity.model_sha256,
                "verified_file_stats": ob._plain(after), "source_pins": ob._plain(pins),
                "preparation_claim_id": claim_id,
                "authority": "original full-byte inventory verification; later stats prove continuity only"}
            artifact = store.write(f"parent-verified-model:{wl._digest(body)}", body)
            _same(t0.require_claim(
                preparation_claim, what="model identity preparation publication close"),
                claim_id, "model preparation publication claim")
            with self._lock:
                self._models[key] = (artifact, ob._freeze(body))
            return artifact
        finally:
            with self._lock:
                self._models_inflight.discard(key)

    def _model_fact(self, context: Any, store: mc.ArtifactStore) -> Mapping[str, Any]:
        key = (context.recipe.model.path, context.recipe.model.sha256)
        with self._lock:
            original = self._models.get(key)
        if original is None:
            return ob._freeze({"original": None, "continuity": None,
                               "error": "original verified model inventory unavailable"})
        artifact, body = original
        _same(store.read(artifact.locator, artifact.sha256), body, "original model inventory receipt")
        _same(_model_source_pins(), body["source_pins"], "original owning model verifier")
        paths = tuple(item["path"] for item in body["verified_file_stats"])
        try:
            observed = _file_stats(paths)
            error = None if observed == body["verified_file_stats"] else "model continuity changed"
        except (OSError, ScientificWitnessRefused) as exc:
            observed, error = (), type(exc).__name__
        return ob._freeze({"original": artifact.to_dict(), "continuity": ob._plain(observed),
                           "error": error})

    def collect_issued(self, *, context: Any, store: mc.ArtifactStore,
                       plan: t0.T0ExecutionPlan, request: api.EvaluationRequest,
                       policy: correctness.T0Policy, claim: Any,
                       anchor_capture: t0.AnchorCapture | None = None) -> mc.StoredArtifact:
        """Explicit live owner entry point; never invoked by a child's completion."""
        if (type(store) is not mc.ArtifactStore or type(plan) is not t0.T0ExecutionPlan
                or type(request) is not api.EvaluationRequest
                or type(policy) is not correctness.T0Policy
                or anchor_capture is not None and type(anchor_capture) is not t0.AnchorCapture):
            raise ScientificWitnessRefused("concrete original T0 inputs required")
        frame = _frame(context)
        key = (context.plan.digest, context.unit_id)
        plan, request, policy, anchor_capture = (
            _immutable(item) for item in (plan, request, policy, anchor_capture))
        fingerprint = wl._digest({"frame": _project(frame), "plan": _project(plan),
            "request": _project(request), "policy": _project(policy),
            "anchor": _project(anchor_capture), "artifact_root": str(store.root)})
        with self._lock:
            prior = self._issued.get(key)
            if prior is None:
                if key in self._failed:
                    raise ScientificWitnessRefused("original T0 collection failed; a new unit is required")
                if key in self._inflight:
                    if self._inflight[key] != fingerprint:
                        raise ScientificWitnessRefused("inflight original T0 inputs conflict")
                    raise ScientificWitnessRefused("original T0 collection is already in flight")
                if len(self._issued) + len(self._inflight) + len(self._failed) >= self.max_units:
                    raise ScientificWitnessRefused("T0 issuance registry capacity exhausted")
                self._inflight[key] = fingerprint
        if prior is not None:
            _same(fingerprint, prior.original_body["issuance_input_sha256"], "duplicate original T0 inputs")
            return self._replay(context, store).original_artifact
        try:
            issued = self._collect(context=context, store=store, frame=frame, plan=plan,
                request=request, policy=policy, claim=claim, anchor_capture=anchor_capture,
                fingerprint=fingerprint)
            with self._lock:
                if self._inflight.get(key) != fingerprint or key in self._issued:
                    raise ScientificWitnessRefused("original T0 reservation changed")
                self._issued[key] = issued
                del self._inflight[key]
            return issued.original_artifact
        except BaseException:
            with self._lock:
                self._inflight.pop(key, None)
                if key not in self._issued:
                    self._failed.add(key)
            raise

    def _collect(self, *, context, store, frame, plan, request, policy, claim, anchor_capture, fingerprint):
        """All subprocess, source/file reads and store I/O are outside the registry lock."""
        claim_id = t0.require_claim(claim, what="original native T0 collection")
        _same(claim_id, context.binding.active_claim_ref, "original T0 claim")
        pins = _source_pins()
        native_before = _native_artifact_facts(context)
        model_before = self._model_fact(context, store)
        recorder = _CaptureRecorder(t0.SubprocessRunner(), store)
        started = time.monotonic()
        anchor_origin = "supplied_unreplayed" if anchor_capture is not None else "absent"
        anchor_refs = ()
        if anchor_capture is None and plan.anchor is not None:
            seeds = (() if plan.generation is None else
                     tuple(plan.generation.seed for _ in range(plan.determinism_runs)))
            anchor_capture = _immutable(t0.capture_anchor(plan=plan, runner=recorder,
                claim=claim, generation_seeds=seeds, oracle_ids=plan.oracle_ids))
            anchor_origin = "owning_capture"
            anchor_refs = tuple(anchor_capture.capture_refs)
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=recorder,
            claim=claim, anchor_capture=anchor_capture)
        evidence = _immutable(provider.evidence_for(request))
        ended = time.monotonic()
        native_after = _native_artifact_facts(context)
        model_after = self._model_fact(context, store)
        _same(t0.require_claim(claim, what="original native T0 collection close"),
              claim_id, "original T0 closing claim")
        _same(_source_pins(), pins, "original T0 source closure")
        report = correctness.T0CorrectnessRunner(
            provider=correctness.StaticEvidenceProvider({request.candidate_id: evidence}),
            policy=policy).evaluate(request)
        _same(tuple(row["body"]["capture_ref"] for row in recorder.records),
              anchor_refs + tuple(provider.capture_refs), "complete original T0 capture sequence")
        body = {"schema": T0_SCHEMA, "frame": ob._plain(frame),
            "plan": _project(plan), "request": _project(request),
            "policy": _project(policy), "anchor": _project(anchor_capture),
            "anchor_origin": anchor_origin, "issuance_input_sha256": fingerprint,
            "evidence": _project(evidence), "report": report.to_dict(),
            "raw_captures": [ob._plain(item) for item in recorder.records],
            "source_pins": ob._plain(pins), "claim_id": claim_id,
            "native_artifacts_before": ob._plain(native_before),
            "native_artifacts_after": ob._plain(native_after),
            "model_before": ob._plain(model_before), "model_after": ob._plain(model_after),
            "started_monotonic_s": started, "ended_monotonic_s": ended,
            "replay_coverage": REPLAY_COVERAGE}
        artifact = store.write(f"parent-issued-t0:{wl._digest(body)}", body)
        return _IssuedT0(frame, plan, request, policy, anchor_capture,
            evidence, ob._freeze(report.to_dict()), tuple(recorder.records), pins,
            artifact, ob._freeze(body))

    def _replay(self, context: Any, store: mc.ArtifactStore) -> _IssuedT0:
        with self._lock:
            issued = self._issued.get((context.plan.digest, context.unit_id))
        if issued is None:
            raise ScientificWitnessRefused("original parent-issued T0 evidence is unavailable")
        _same(_frame(context), issued.frame, "complete original T0 frame")
        _same(_source_pins(), issued.source_pins, "original T0 source pins")
        body = store.read(issued.original_artifact.locator, issued.original_artifact.sha256)
        _same(body, issued.original_body, "original T0 artifact bytes")
        store.verify(f"parent-issued-t0:{wl._digest(ob._plain(body))}", body)
        model_ref = issued.original_body["model_before"]["original"]
        if model_ref is not None:
            key = (context.recipe.model.path, context.recipe.model.sha256)
            with self._lock:
                original_model = self._models.get(key)
            if original_model is None:
                raise ScientificWitnessRefused("original model issuance is unavailable")
            artifact, model_body = original_model
            _same(artifact.to_dict(), model_ref, "original model reference")
            reopened_model = store.read(artifact.locator, artifact.sha256)
            _same(reopened_model, model_body, "original complete model inventory receipt")
            store.verify(f"parent-verified-model:{wl._digest(ob._plain(reopened_model))}", reopened_model)
            _same(_model_source_pins(), model_body["source_pins"], "original model verifier source")
        for index, row in enumerate(issued.raw):
            ref = mc.StoredArtifact(**ob._plain(row["artifact"]))
            raw = store.read(ref.locator, ref.sha256)
            _same(raw, row["body"], "original ordered T0 raw capture")
            _same(raw["sequence"], index, "original T0 capture cardinality/order")
            store.verify(f"parent-t0-raw:{wl._digest(ob._plain(raw))}", raw)
        report = correctness.T0CorrectnessRunner(
            provider=correctness.StaticEvidenceProvider({issued.request.candidate_id: issued.evidence}),
            policy=issued.policy).evaluate(issued.request)
        _same(report.to_dict(), issued.report, "complete owning T0 reducer replay")
        return issued

    def evaluate(self, context: Any, native: Mapping[str, Any], lifecycle_link: Any,
                 runtime_readbacks: tuple[mc.StoredArtifact, ...], *,
                 store: mc.ArtifactStore) -> WitnessEvidenceReference:
        issued = self._replay(context, store)
        from .native_parent_evidence import NativeUnitEvidenceProducer
        verifier = NativeUnitEvidenceProducer(store=store, context=context)
        verifier._join_native(native)
        reference = lifecycle_link.reference
        observation = lo.validate_observation(ob._plain(store.read(
            reference.artifact.locator, reference.artifact.sha256)))
        verifier._join_observation(native, observation, reference)
        mismatch = self._scope_defects(context, issued, observation)
        report = issued.report
        status = ("unknown" if mismatch else "fail" if report["failed"] else
                  "unknown" if report["unevaluated"] or not report["anchor_bound"]
                  or report["demoted_gates"] else "pass")
        body = {"schema": WITNESS_SCHEMA, "witness": "correctness",
            "adapter_id": ADAPTER_ID, "frame": ob._plain(_frame(context)),
            "native_artifact_digest": native["artifact_digest"],
            "lifecycle_reference": reference.to_dict(),
            "runtime_readbacks": [ref.to_dict() for ref in runtime_readbacks],
            "original_t0": issued.original_artifact.to_dict(),
            "original_t0_digest": wl._digest(ob._plain(issued.original_body)),
            "source": self.source_identity(), "report": ob._plain(report),
            "scope_defects": mismatch, "status": status,
            "replay_coverage": REPLAY_COVERAGE}
        digest = wl._digest(body)
        return WitnessEvidenceReference(store.write(f"parent-scientific-witness:{digest}", body), digest)

    @staticmethod
    def _scope_defects(context: Any, issued: _IssuedT0,
                       observation: Mapping[str, Any]) -> list[str]:
        """Identity matching only; no tool-family or request equivalence policy."""
        defects = []
        request, plan, recipe = issued.request, issued.plan, context.recipe
        if request.campaign_id != context.plan.campaign_id:
            defects.append("T0 request campaign differs from native campaign")
        if issued.original_body["anchor_origin"] == "supplied_unreplayed":
            defects.append("supplied T0 anchor lacks original raw-capture issuance in this adapter")
        model_before, model_after = (issued.original_body[name] for name in ("model_before", "model_after"))
        if model_before != model_after or model_before["error"] is not None:
            defects.append("original complete model inventory lacks stable unit readback")
        before = issued.original_body["native_artifacts_before"]
        after = issued.original_body["native_artifacts_after"]
        if before != after or any(row["observed"] is None or row["error"] is not None
                or row["observed"]["sha256"] != row["expected"]["sha256"] for row in before):
            defects.append("native executable/DSO bytes were not stable and exact during T0")
        expected_env = t0._launch_env(plan.candidate.library_path, plan.base_env, plan.parameter_env)
        if tuple(expected_env) != tuple(recipe.launch_env):
            defects.append("T0 launch environment differs from native recipe")
        if (request.artifact.binary_sha256 != recipe.executable.sha256
                or plan.candidate.binary != recipe.executable.path):
            defects.append("T0 tool artifact is not the native serving artifact")
        if plan.candidate.build_dir != recipe.build_dir:
            defects.append("T0 build directory is not the native build")
        if request.backend != {"cpu": "llama_cpu", "gpu": "llama_gpu"}[recipe.backend]:
            defects.append("T0 backend differs from native backend")
        if request.artifact.source_sha256 != plan.candidate.source_sha256:
            defects.append("T0 request source differs from original plan")
        generation = plan.generation
        prompts = [item for item in context.prompts.prompts
                   if item.prompt_id in context.unit.expected_prompt_ids]
        if generation is None or len(prompts) != 1 or (
                generation.prompt, generation.prompt_ref, generation.n_predict,
                generation.temperature, generation.top_k) != (
                prompts[0].prompt, prompts[0].prompt_id, prompts[0].n_predict,
                prompts[0].temperature, prompts[0].top_k):
            defects.append("T0 generation is not the complete frozen native request set")
        if generation is None or tuple(generation.extra_argv).count("-m") != 1:
            defects.append("T0 generation has no unambiguous original model binding")
        elif generation.extra_argv[generation.extra_argv.index("-m") + 1:] != (recipe.model.path,):
            defects.append("T0 generation model differs or carries unsupported trailing arguments")
        body = issued.original_body
        if not (observation["started_monotonic_s"] <= body["started_monotonic_s"]
                <= body["ended_monotonic_s"] <= observation["ended_monotonic_s"]):
            defects.append("original T0 collection is outside the unit lifecycle interval")
        return defects

    def source_identity(self) -> Mapping[str, Any]:
        return {"adapter_id": ADAPTER_ID, "configuration": {"max_units": self.max_units},
                "callables": [{"role": name, "identity": lo.callable_identity(getattr(type(self), name))}
                    for name in ADAPTER_ROLES],
                "owning_source_pins": ob._plain(_source_pins())}

    def reopen(self, reference: WitnessEvidenceReference, *, context: Any,
               native: Mapping[str, Any], lifecycle_link: Any,
               runtime_readbacks: tuple[mc.StoredArtifact, ...],
               store: mc.ArtifactStore) -> Mapping[str, Any]:
        if type(reference) is not WitnessEvidenceReference:
            raise ScientificWitnessRefused("concrete scientific receipt reference required")
        body = store.read(reference.artifact.locator, reference.artifact.sha256)
        _same(wl._digest(ob._plain(body)), reference.digest, "scientific receipt digest")
        store.verify(f"parent-scientific-witness:{reference.digest}", body)
        expected = self.evaluate(context, native, lifecycle_link, runtime_readbacks, store=store)
        _same(reference.to_dict(), expected.to_dict(), "independently replayed scientific receipt")
        return ob._freeze(ob._plain(body))


@dataclass(frozen=True)
class ParentScientificWitnessAdapters:
    correctness: NativeT0WitnessAdapter | None = None
    purpose: None = None
    contention: None = None
    residency: None = None
    schema: str = ADAPTERS_SCHEMA

    def __post_init__(self) -> None:
        from .native_server_t0_witness import NativeServerT0WitnessAdapter
        if type(self.correctness) is NativeServerT0WitnessAdapter:
            if self.schema not in (ADAPTERS_SCHEMA, ADAPTERS_SCHEMA_V2) or any(
                    getattr(self, name) is not None for name in SLOTS[1:]):
                raise ScientificWitnessRefused("unsupported server T0 adapter registry slots")
            object.__setattr__(self, "schema", ADAPTERS_SCHEMA_V2)
            return
        if (self.schema != ADAPTERS_SCHEMA or self.correctness is not None
                and type(self.correctness) is not NativeT0WitnessAdapter
                or any(getattr(self, name) is not None for name in SLOTS[1:])):
            raise ScientificWitnessRefused("only the concrete installed T0 adapter is supported")

    def source_identity(self) -> Mapping[str, Any]:
        return {"schema": self.schema, **{name: None if getattr(self, name) is None
                else getattr(self, name).source_identity() for name in SLOTS}}

    def findings(self, context: Any, native: Mapping[str, Any], lifecycle_link: Any,
                 runtime_readbacks: tuple[mc.StoredArtifact, ...], *,
                 store: mc.ArtifactStore) -> Mapping[str, Any]:
        result = {}
        for name in SLOTS:
            adapter = getattr(self, name)
            if adapter is None:
                continue
            try:
                reference = adapter.evaluate(context, native, lifecycle_link,
                                             runtime_readbacks, store=store)
                body = adapter.reopen(reference, context=context, native=native,
                    lifecycle_link=lifecycle_link, runtime_readbacks=runtime_readbacks, store=store)
                result[name] = {"status": body["status"],
                    "reason": (body["replay_coverage"] if self.schema == ADAPTERS_SCHEMA_V2
                               else REPLAY_COVERAGE), "facts": {"receipt": reference.to_dict(),
                    "report": ob._plain(body["report"]), "scope_defects": list(body["scope_defects"]),
                    "adapter_source": ob._plain(body["source"])}}
            except Exception as exc:
                result[name] = {"status": "unknown", "reason": "owning T0 evidence unavailable",
                    "facts": {"refusal_type": type(exc).__name__, "detail": str(exc)}}
        return ob._freeze(result)


def validate_scientific_source(value: Any) -> Mapping[str, Any]:
    """Closed historical grammar; no comparison with today's installation here."""
    from .native_producer_source import _closed, _identity, _callables
    row = _closed(value, ("schema", *SLOTS), "scientific adapter source")
    if row["schema"] == ADAPTERS_SCHEMA_V2:
        from .native_server_t0_witness import validate_server_source
        if row["correctness"] is None or any(row[name] is not None for name in SLOTS[1:]):
            raise ScientificWitnessRefused("server T0 registry requires exact concrete correctness slot")
        row["correctness"] = validate_server_source(row["correctness"])
        return ob._freeze(ob._plain(row))
    if row["schema"] != ADAPTERS_SCHEMA or any(row[name] is not None for name in SLOTS[1:]):
        raise ScientificWitnessRefused("unsupported scientific adapter source schema/slots")
    if row["correctness"] is not None:
        adapter = _closed(row["correctness"],
            ("adapter_id", "configuration", "callables", "owning_source_pins"), "T0 adapter source")
        if adapter["adapter_id"] != ADAPTER_ID:
            raise ScientificWitnessRefused("unsupported concrete T0 adapter")
        config = _closed(adapter["configuration"], ("max_units",), "T0 adapter configuration")
        if type(config["max_units"]) is not int or not 1 <= config["max_units"] <= MAX_UNITS:
            raise ScientificWitnessRefused("unsupported T0 adapter bound")
        adapter["configuration"] = config
        adapter["callables"] = _callables(adapter["callables"], ADAPTER_ROLES)
        pins = _closed(adapter["owning_source_pins"], ("modules", "callables"), "T0 owning source")
        if not isinstance(pins["modules"], (list, tuple)) or len(pins["modules"]) != 6:
            raise ScientificWitnessRefused("T0 owning module scope differs")
        pins["modules"] = [lo._validate_artifact(item) for item in pins["modules"]]
        functions = _source_functions()
        if not isinstance(pins["callables"], (list, tuple)) or len(pins["callables"]) != len(functions):
            raise ScientificWitnessRefused("T0 owning callable scope differs")
        pins["callables"] = [_identity(item) for item in pins["callables"]]
        for item, function in zip(pins["callables"], functions):
            if (item["module"], item["qualname"]) != (function.__module__, function.__qualname__):
                raise ScientificWitnessRefused("T0 owning callable roles differ or are reordered")
        adapter["owning_source_pins"] = pins
        row["correctness"] = adapter
    return ob._freeze(ob._plain(row))


def scientific_source_identities(value: Any) -> tuple[Mapping[str, Any], ...]:
    row = validate_scientific_source(value)
    adapter = row["correctness"]
    if row["schema"] == ADAPTERS_SCHEMA_V2:
        from .native_server_t0_witness import server_source_identities
        return server_source_identities(adapter)
    if adapter is None:
        return ()
    return tuple(item["identity"] for item in adapter["callables"]) + tuple(
        adapter["owning_source_pins"]["callables"])


def installed_scientific_adapters(value: Any) -> ParentScientificWitnessAdapters:
    """Reconstruct selected installed TYPE/CONFIG only; never restore its issuance."""
    row = validate_scientific_source(value)
    adapter = row["correctness"]
    if row["schema"] == ADAPTERS_SCHEMA_V2:
        from .native_server_t0_witness import NativeServerT0WitnessAdapter
        config = adapter["configuration"]
        selected = ParentScientificWitnessAdapters(correctness=NativeServerT0WitnessAdapter(
            max_units=config["max_units"],
            owning_issuer=NativeT0WitnessAdapter(max_units=config["owning_issuer_max_units"])))
        _same(selected.source_identity(), row, "installed server scientific adapter source")
        return selected
    selected = ParentScientificWitnessAdapters(correctness=None if adapter is None else
        NativeT0WitnessAdapter(max_units=adapter["configuration"]["max_units"]))
    _same(selected.source_identity(), row, "installed scientific adapter source")
    return selected


__all__ = ["NativeT0WitnessAdapter", "ParentScientificWitnessAdapters",
           "WitnessEvidenceReference", "ScientificWitnessRefused", "ADAPTERS_SCHEMA"]
