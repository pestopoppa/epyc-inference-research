"""Parent-final scientific evidence over immutable child-native observations.

No launch, acquisition, statistical policy or grade lives here. Original child
diagnostics remain unchanged. A new final view is produced by the existing
admissibility reducer after replaying the original parent scientific issuances.
Control/calibration collection is not implemented: their slots explicitly refuse
anything except None until a concrete owning issuance API is installed.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Mapping

from . import experiment_plan as ep
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_parent_evidence as npe
from . import native_parent_receipt_replay as replay
from . import native_scientific_witness as scientific
from . import observation_binding as ob
from . import unified_worker as uw
from . import worker_lifecycle as wl

TRIAL_SCHEMA = "epyc.autokernel.parent_final_trial.v1"
REFERENCE_SCHEMA = "epyc.autokernel.parent_final_trial_reference.v1"
CAPTURE_SCHEMA = "epyc.autokernel.unified_arm_capture.v3"
PRODUCER_ID = "epyc.autokernel.measurement_capture/v3"
SOURCE_SCHEMA = "epyc.autokernel.parent_final_trial_source.v1"
OWNER_ROLES = ("__init__", "_build", "finalize", "reopen", "captures", "prevalidate")
HELPER_ROLES = ("source_identity", "validate_source_identity", "source_identities",
                "project_final_rows", "_capture_payloads")


@dataclass(frozen=True, init=False)
class ParentFinalTrialReference:
    artifact: mc.StoredArtifact
    digest: str

    def __init__(self, artifact: mc.StoredArtifact, digest: str) -> None:
        if type(artifact) is not mc.StoredArtifact or artifact.verified is not True:
            raise scientific.ScientificWitnessRefused("concrete final trial artifact required")
        wl._sha(digest, "final trial digest")
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "digest", digest)

    def to_dict(self):
        return {"schema": REFERENCE_SCHEMA, "artifact": self.artifact.to_dict(), "digest": self.digest}


@dataclass(frozen=True)
class ValidatedParentFinalTrial:
    reference: ParentFinalTrialReference
    plan_digest: str
    original_capture_refs: Mapping[str, Any]
    final_raw_units: tuple[ep.RawUnit, ...]
    final_view: ep.AdmissibleUnitView
    t0_reports_by_unit: Mapping[str, tuple[Any, ...]]
    control_reference: None
    calibration_reference: None
    source_identity: Mapping[str, Any]


@dataclass(frozen=True)
class _IssuedFinal:
    prepared: Any
    start: Any
    terminal: Any
    result: Any
    fence: Any
    registry: replay.IssuedNativeEvidenceRegistry
    pair_reference: Any
    reference: ParentFinalTrialReference
    body: Mapping[str, Any]


def source_identity():
    from .driver_execution import _terminal_body
    # Includes the newly extracted original reader, not a disk-only warrant.
    functions = (uw.reopen_deferred_result, replay.IssuedNativeEvidenceRegistry.snapshot,
        replay.NativeParentReceiptReplayer._replay_unit, ep.admissible_units,
        mc._MeasurementCaptureBuilder._diagnostic, mc._MeasurementCaptureBuilder._environment,
        ParentFinalTrialReference.__init__, ParentFinalTrialReference.to_dict, _terminal_body)
    return {"schema": SOURCE_SCHEMA, "trial_schema": TRIAL_SCHEMA,
        "reference_schema": REFERENCE_SCHEMA, "capture_schema": CAPTURE_SCHEMA,
        "producer_id": PRODUCER_ID, "module": lo.prepare_artifact_identity(Path(__file__)),
        "owner": [{"role": name, "identity": lo.callable_identity(getattr(NativeFinalTrialOwner, name))}
                  for name in OWNER_ROLES],
        "helpers": [{"role": name, "identity": lo.callable_identity(globals()[name])}
                    for name in HELPER_ROLES],
        "dependencies": [lo.callable_identity(function) for function in functions]}


def validate_source_identity(value):
    from .native_producer_source import _closed, _callables, _identity
    row = _closed(value, ("schema", "trial_schema", "reference_schema", "capture_schema",
        "producer_id", "module", "owner", "helpers", "dependencies"), "final trial source")
    if (row["schema"], row["trial_schema"], row["reference_schema"], row["capture_schema"],
            row["producer_id"]) != (SOURCE_SCHEMA, TRIAL_SCHEMA, REFERENCE_SCHEMA, CAPTURE_SCHEMA, PRODUCER_ID):
        raise scientific.ScientificWitnessRefused("final trial source contract differs")
    row["module"] = lo._validate_artifact(row["module"])
    row["owner"] = _callables(row["owner"], OWNER_ROLES)
    row["helpers"] = _callables(row["helpers"], HELPER_ROLES)
    if type(row["dependencies"]) not in (tuple, list) or len(row["dependencies"]) != 9:
        raise scientific.ScientificWitnessRefused("final trial dependency closure differs")
    row["dependencies"] = [_identity(item) for item in row["dependencies"]]
    return ob._freeze(ob._plain(row))


def source_identities(value):
    row = validate_source_identity(value)
    return tuple(item["identity"] for item in (*row["owner"], *row["helpers"])) + tuple(row["dependencies"])


def project_final_rows(plan, original_rows, pair_body):
    """Mechanical new-record projection followed by the unchanged owning reducer.

    This pure helper alone confers no issuance authority; callers must first
    reopen the concrete original registries and pair through NativeFinalTrialOwner.
    """
    plan = ep.ExperimentPlan.from_dict(plan.to_dict())
    originals = tuple(ep.RawUnit.from_dict(row) for row in original_rows)
    final_units = pair_body["ordered_units"]
    expected = tuple(unit.unit_id for unit in sorted(plan.expected_units, key=lambda unit: unit.order_index))
    if (tuple(row.unit_id for row in originals) != expected
            or tuple(row["unit_id"] for row in final_units) != expected
            or pair_body["plan_digest"] != plan.digest):
        raise scientific.ScientificWitnessRefused("final row projection requires exact full ordered membership")
    projected = []
    for original, evidence in zip(originals, final_units):
        row = original.to_dict()
        old = original.witnesses.get("correctness")
        status = evidence["final_evidence"]["status"]
        if status not in ("pass", "fail", "unknown"):
            raise scientific.ScientificWitnessRefused("unsupported owning final witness status")
        # Preserve every original fail. A newly established fail is also not lost.
        if old is None or old.status == "unknown" or status == "fail" and old.status != "fail":
            reference = (None if status == "unknown" else
                         f"parent-server-t0-pair:{wl._digest(ob._plain(pair_body))}#{original.unit_id}")
            row["witnesses"]["correctness"] = ep.Witness(status, reference).to_dict()
        projected.append(ep.RawUnit.from_dict(row))
    rows = tuple(projected)
    return rows, ep.admissible_units(plan, rows)


def _capture_payloads(body, reference, store):
    """New v3 sources; original v2 carrier/view/attempt bytes are never modified."""
    plan = ep.ExperimentPlan.from_dict(body["plan"])
    view = body["final_admissible_view"]
    result = []
    for arm in ("anchor", "candidate"):
        original_ref = body["original_arm_captures"][arm]
        original = ob._plain(store.read(original_ref["artifact"]["locator"],
                                        original_ref["artifact"]["sha256"]))
        store.verify(f"carrier:{original_ref['measurement_id']}", original)
        rows = [row for row in view["selected_rows"] if row["arm"] == arm]
        attempts = [item["document"] for item in original["raw_artifacts"]
                    if item["document"].get("kind") == "completed_attempt"]
        diagnostic = mc._MeasurementCaptureBuilder._diagnostic(
            plan, arm, rows, attempts, original["raw_artifacts"])
        values = [float(row["value"]) for row in rows]
        measurement = None
        if diagnostic is None and values:
            if plan.estimator_id != "median.v1":
                diagnostic = f"unsupported estimator {plan.estimator_id!r}"
            else:
                measurement = {"metric": plan.metric, "value": mc.median(values),
                    "unit": "t/s", "independent_unit": plan.unit, "direction": plan.metric_direction,
                    "independent_n": len(values), "reps_basis": "scored independent process launches",
                    "per_launch_values": values}
        identity = {"producer": PRODUCER_ID, "capture_schema": CAPTURE_SCHEMA,
            "plan_digest": plan.digest, "lineage_id": original["lineage_id"], "arm": arm,
            "instrument_identity_sha256": original["loaded_instrument"]["identity_sha256"],
            "parent_final_trial_digest": reference.digest}
        measurement_id = wl._digest(identity)
        carrier = {key: value for key, value in original.items() if key != "carrier_digest"}
        carrier.update(schema=CAPTURE_SCHEMA, producer=PRODUCER_ID, measurement_id=measurement_id,
            arm_locator=f"parent-final-serving:{plan.digest}:{original['lineage_id']}:{arm}:{reference.digest}",
            admissible_view=ob._plain(view), status="measurement" if measurement is not None else "diagnostic",
            diagnostic_reason=diagnostic, measurement=measurement,
            original_arm_capture=ob._plain(original_ref), parent_final_trial=reference.to_dict())
        carrier["carrier_digest"] = wl._digest(carrier)
        result.append((measurement_id, carrier))
    return tuple(result)


class NativeFinalTrialOwner:
    def __init__(self, *, correctness_adapter):
        from .native_server_t0_witness import NativeServerT0WitnessAdapter
        if type(correctness_adapter) is not NativeServerT0WitnessAdapter:
            raise scientific.ScientificWitnessRefused("concrete original server correctness owner required")
        self.correctness_adapter = correctness_adapter
        self._lock = threading.RLock()
        self._issued: dict[str, _IssuedFinal] = {}
        self._inflight: set[str] = set()

    def _build(self, *, prepared, start, terminal, result, fence, registry, store):
        from .driver_execution import _terminal_body
        if type(store) is not mc.ArtifactStore or store.root != prepared.artifact_root:
            raise scientific.ScientificWitnessRefused("final trial store differs from original preparation")
        reopened, captures = uw.reopen_deferred_result(result, prepared=prepared, start=start,
                                                      terminal=terminal, fence=fence)
        row = reopened.to_dict()
        plan = prepared.plan
        if prepared.schema != uw.PREPARED_SCHEMA_V2:
            raise scientific.ScientificWitnessRefused("final trial requires original native v2 capture")
        expected = [unit.unit_id for unit in sorted(plan.expected_units, key=lambda unit: unit.order_index)]
        if (row["completed_unit_ids"] != expected
                or [item["unit_id"] for item in row["lifecycle_observation_references"]] != expected
                or [item["unit_id"] for item in row["run"]["raw_units"]] != expected):
            raise scientific.ScientificWitnessRefused("final trial refuses incomplete original unit/lifecycle prefix")
        entries = registry.snapshot(plan=plan, store=store)
        native_by_id, attempt_by_id, arm_captures = {}, {}, {}
        for measurement_id, payload in captures:
            carrier = payload["carrier"]
            if (payload["schema"] != mc.CAPTURE_SCHEMA_V2 or carrier["schema"] != mc.CAPTURE_SCHEMA_V2
                    or carrier["arm"] in arm_captures):
                raise scientific.ScientificWitnessRefused("final trial requires two unique original v2 arms")
            arm_captures[carrier["arm"]] = {"measurement_id": measurement_id,
                "carrier_digest": carrier["carrier_digest"], "artifact": ob._plain(payload["artifact"])}
            for item in carrier["raw_artifacts"]:
                document = item["document"]
                if document.get("kind") not in ("native_observation", "completed_attempt"):
                    continue
                target = native_by_id if document["kind"] == "native_observation" else attempt_by_id
                if document["unit_id"] in target:
                    raise scientific.ScientificWitnessRefused("duplicate original native/attempt unit")
                target[document["unit_id"]] = ob._plain(document)
        if set(arm_captures) != {"anchor", "candidate"} or set(native_by_id) != set(expected) or set(attempt_by_id) != set(expected):
            raise scientific.ScientificWitnessRefused("final trial original full arm membership differs")
        for entry in entries:
            context = entry.context
            scientific._same(context.nonce, start.nonce, "original final worker nonce")
            scientific._same(context.prompts.to_dict(), prepared.prompts.to_dict(), "original final prompts")
            for field, original_field in (("worker_id", "worker_id"),
                    ("worker_generation", "worker_generation"), ("grant_id", "grant_id"),
                    ("grant_generation", "grant_generation"), ("container_id", "container_id"),
                    ("lineage_id", "lineage_id"), ("config_digest", "config_digest"),
                    ("config_generation", "config_generation"), ("supervisor_id", "supervisor_id"),
                    ("supervisor_incarnation", "supervisor_incarnation")):
                scientific._same(context.descendant_event[field], getattr(start, original_field),
                                 f"original final {field}")
            if entry.scientific_adapters is None or entry.scientific_adapters.correctness is not self.correctness_adapter:
                raise scientific.ScientificWitnessRefused("final trial original adapter instance differs")
            replay.NativeParentReceiptReplayer._replay_unit(entry, native_by_id[context.unit_id],
                attempt_by_id[context.unit_id], store)
        pair_reference = self.correctness_adapter.finalize_pair(plan=plan, registry=registry, store=store)
        pair = self.correctness_adapter.reopen_pair(pair_reference, plan=plan, registry=registry, store=store)
        original_rows = row["run"]["raw_units"]
        original_view = ep.admissible_units(plan, (ep.RawUnit.from_dict(item) for item in original_rows))
        scientific._same(row["run"]["admissible_view"], original_view.to_dict(), "original child admissible view")
        final_rows, view = project_final_rows(plan, original_rows, pair)
        body = {"schema": TRIAL_SCHEMA, "plan": plan.to_dict(), "plan_digest": plan.digest,
            "prepared_digest": prepared.prepared_digest, "result_reference": result.to_dict(),
            "worker_start": start.to_dict(), "accepted_terminal": _terminal_body(terminal),
            "result_fence": fence.to_dict(), "original_arm_captures": arm_captures,
            "original_unit_receipts": [{"unit_id": entry.context.unit_id,
                "receipt": entry.result.receipt.to_dict(), "digest": entry.result.receipt_digest}
                for entry in entries],
            "scientific_pair_reference": pair_reference.to_dict(),
            "control_reference": None, "calibration_reference": None,
            "original_raw_units": original_rows, "original_admissible_view": original_view.to_dict(),
            "final_rows": [{"original_unit_digest": wl._digest(original.to_dict()),
                "original_parent_receipt": entry.result.receipt.to_dict(),
                "final_scientific_reference": pair_reference.to_dict(), "row": final.to_dict()}
                for original, entry, final in zip((ep.RawUnit.from_dict(item) for item in original_rows), entries, final_rows)],
            "final_admissible_view": view.to_dict(),
            "source_identity": {"finalizer": source_identity(),
                                "selected_scientific_adapter": self.correctness_adapter.source_identity()}}
        return body, pair_reference

    def finalize(self, *, prepared, start, terminal, result, fence, registry, store,
                 control_evidence=None, calibration=None) -> ParentFinalTrialReference:
        if control_evidence is not None or calibration is not None:
            raise scientific.ScientificWitnessRefused("original control/calibration issuer is not installed")
        key = result.result_digest
        with self._lock:
            prior = self._issued.get(key)
            if key in self._inflight:
                raise scientific.ScientificWitnessRefused("final trial already in flight")
            if prior is None and len(self._issued) >= self.correctness_adapter.max_units:
                raise scientific.ScientificWitnessRefused("final trial registry capacity exhausted")
            self._inflight.add(key)
        try:
            body, pair = self._build(prepared=prepared, start=start, terminal=terminal,
                result=result, fence=fence, registry=registry, store=store)
            if prior is not None:
                if registry is not prior.registry:
                    raise scientific.ScientificWitnessRefused("final trial original registry instance changed")
                scientific._same(body, prior.body, "immutable final trial replay")
                reopened = store.read(prior.reference.artifact.locator, prior.reference.artifact.sha256)
                scientific._same(reopened, body, "original final trial bytes")
                store.verify(f"parent-final-trial:{prior.reference.digest}", body)
                return prior.reference
            digest = wl._digest(body)
            reference = ParentFinalTrialReference(store.write(f"parent-final-trial:{digest}", body), digest)
            issued = _IssuedFinal(prepared, start, terminal, result, fence, registry, pair,
                                  reference, ob._freeze(body))
            with self._lock:
                self._issued[key] = issued
            return reference
        finally:
            with self._lock:
                self._inflight.discard(key)

    def reopen(self, reference, *, registry, store) -> ValidatedParentFinalTrial:
        if type(reference) is not ParentFinalTrialReference:
            raise scientific.ScientificWitnessRefused("concrete final trial reference required")
        with self._lock:
            found = [item for item in self._issued.values() if item.reference == reference]
        if len(found) != 1 or found[0].registry is not registry:
            raise scientific.ScientificWitnessRefused("original final trial issuance unavailable")
        issued = found[0]
        expected = self.finalize(prepared=issued.prepared, start=issued.start, terminal=issued.terminal,
            result=issued.result, fence=issued.fence, registry=registry, store=store)
        scientific._same(reference.to_dict(), expected.to_dict(), "original final trial reference")
        reports = self.correctness_adapter.reopen_pair_reports(issued.pair_reference,
            plan=issued.prepared.plan, registry=registry, store=store)
        rows = tuple(ep.RawUnit.from_dict(ob._plain(item["row"])) for item in issued.body["final_rows"])
        view = ep.admissible_units(issued.prepared.plan, rows)
        return ValidatedParentFinalTrial(reference, issued.prepared.plan.digest,
            issued.body["original_arm_captures"], rows, view, reports, None, None,
            issued.body["source_identity"])

    def captures(self, reference, *, registry, store):
        self.reopen(reference, registry=registry, store=store)
        body = ob._plain(store.read(reference.artifact.locator, reference.artifact.sha256))
        out = []
        for measurement_id, carrier in _capture_payloads(body, reference, store):
            artifact = store.write(f"carrier:{measurement_id}", carrier)
            payload = {"schema": CAPTURE_SCHEMA, "measurement_id": measurement_id,
                       "carrier": carrier, "artifact": artifact.to_dict()}
            out.append((measurement_id, ob._freeze(payload)))
        return tuple(out)

    def prevalidate(self, validator, measurement_id, payload):
        from . import native_capture_control as nc
        if type(validator) is not nc.NativeCaptureValidator or validator.parent_receipt_replayer is None:
            raise scientific.ScientificWitnessRefused("final capture requires original parent replay scope")
        replayer = validator.parent_receipt_replayer
        with replayer._lock:
            registry = replayer._registry
        if type(registry) is not replay.IssuedNativeEvidenceRegistry:
            raise scientific.ScientificWitnessRefused("final capture original registry unavailable")
        row = ob._plain(payload)
        if type(row) is not dict or set(row) != {"schema", "measurement_id", "carrier", "artifact"} or row["schema"] != CAPTURE_SCHEMA or row["measurement_id"] != measurement_id:
            raise scientific.ScientificWitnessRefused("closed final capture payload differs")
        ref = npe._closed(row["carrier"].get("parent_final_trial"),
                          {"schema", "artifact", "digest"}, "parent final reference")
        if ref["schema"] != REFERENCE_SCHEMA:
            raise scientific.ScientificWitnessRefused("final trial reference schema differs")
        reference = ParentFinalTrialReference(npe._artifact(ref["artifact"]), ref["digest"])
        validated = self.reopen(reference, registry=registry, store=validator.store)
        body = ob._plain(validator.store.read(reference.artifact.locator, reference.artifact.sha256))
        expected = dict(_capture_payloads(body, reference, validator.store)).get(measurement_id)
        scientific._same(row["carrier"], expected, "independently rebuilt final carrier")
        original_prevalidations = []
        for original in validated.original_capture_refs.values():
            original_carrier = ob._plain(validator.store.read(original["artifact"]["locator"],
                                                             original["artifact"]["sha256"]))
            original_payload = {"schema": mc.CAPTURE_SCHEMA_V2,
                "measurement_id": original["measurement_id"], "carrier": original_carrier,
                "artifact": ob._plain(original["artifact"])}
            original_prevalidations.append(validator.prevalidate(original["measurement_id"], original_payload))
        generations = {item.grant_generation for item in original_prevalidations}
        if len(generations) != 1:
            raise scientific.ScientificWitnessRefused("original final capture grant generations differ")
        artifact = validator.store.verify(f"carrier:{measurement_id}", row["carrier"])
        scientific._same(row["artifact"], artifact.to_dict(), "final carrier artifact reference")
        context = mc.CaptureContext.from_dict(row["carrier"]["capture_context"])
        validator._validate_binding(context)
        digest = wl._digest(row)
        native = nc.ValidatedNativeCapture(measurement_id, digest, row["carrier"]["status"], False,
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        links = tuple(link for item in original_prevalidations for link in item.observation_links)
        return nc.PrevalidatedNativeCapture(measurement_id, digest, context,
                                           next(iter(generations)), links, native)
