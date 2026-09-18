"""Pinned bridge to the canonical ROOT ClaimTuple projection and grade.

The bridge deliberately has no grading or scientific policy of its own.  Its
default is the complete, explicitly published native-v2 source closure.  The
legacy v1 pin remains available only when selected explicitly; old receipts are
never inferred to have the prospective evidence carried by native v2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import threading
from types import ModuleType
from typing import Any, Mapping

from .. import schemas
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import validation_claim_receipt as cr
from . import validation_consumer as vc
from .validation_projection_source import ProjectionSourceClosure


PROJECTOR_NAME = "autokernel-unified-arm-measurement"
CLAIM_TUPLE_SHA256 = "375d46450d2fa01314ebcbddfba26411f3901f7e0df751773fd35a43f00606bb"
ARM_PROJECTOR_SHA256 = "590b1ac656b0123517a12fa194003bfed148bd3cf25553f986ee3a48ef7b0eae"
ARM_ADAPTER_ID = "vidya.adapters.autokernel_unified_arm/v1"
AUTHORITY_ID = "autokernel.validation.semantic-owner/v1"
FINAL_V2_ROOT_COMMIT: str | None = "2010b713f6a02d18c36799eac7c35c1361c06764"
FINAL_V2_CLAIM_TUPLE_SHA256: str | None = CLAIM_TUPLE_SHA256
FINAL_V2_ARM_PROJECTOR_SHA256: str | None = \
    "ad87bdec7afc4d07f04fe48375adf4fe476a20423481b2be662e776a2b590192"
FINAL_V2_ADAPTER_ID: str | None = ARM_ADAPTER_ID
FINAL_V2_MEASUREMENT_CAPTURE_SHA256: str | None = \
    "04cacacc8576048ff18a96e2c332ca2e2ea59bfdbfcc0acc451c1939f0ad3123"
FINAL_V2_OBSERVATION_BINDING_SHA256: str | None = \
    "fa893b554d6ab73222f3e003ffdefa7e062651adbce0de6014951731e6d0eed3"

MEASUREMENT_CAPTURE_PRODUCER_V2 = "epyc.autokernel.measurement_capture/v2"
CAPTURE_SCHEMA_V2 = "epyc.autokernel.unified_arm_capture.v2"
OBSERVATION_BINDING_PRODUCER = "scripts/kernel_rnd/autokernel/loop/observation_binding.py"
OBSERVATION_BINDING_SCHEMAS = {
    "loaded_instrument_reference":
        "epyc.autokernel.loaded_serving_instrument_reference.v1",
    "observation_unit_binding": "epyc.autokernel.observation_unit_binding.v1",
    "lifecycle_observation_reference":
        "epyc.autokernel.lifecycle_observation_reference.v1",
    "lifecycle_observation_link": "epyc.autokernel.lifecycle_observation_link.v1",
}
_MAX_PINNED_SOURCE_BYTES = 8 * 1024 * 1024
_LOAD_LOCK = threading.RLock()


class SemanticAdapterError(RuntimeError):
    """The installed canonical source or an evidence binding is unavailable."""


def _digest(path: Path) -> str:
    return hashlib.sha256(_source_bytes(path)).hexdigest()


def _source_bytes(path: Path, expected_sha256: str | None = None) -> bytes:
    """Read one bounded stable regular file once, without following a symlink."""
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK
                             | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_size < 0
                    or before.st_size > _MAX_PINNED_SOURCE_BYTES):
                raise SemanticAdapterError("pinned source is not a bounded regular file")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise SemanticAdapterError("pinned source ended before its fixed size")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise SemanticAdapterError("pinned source grew while being read")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        named = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise SemanticAdapterError("pinned source cannot be opened safely") from exc
    def identity(item):
        return (item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
                item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(named):
        raise SemanticAdapterError("pinned source changed while being read")
    source = b"".join(chunks)
    if expected_sha256 is not None \
            and hashlib.sha256(source).hexdigest() != expected_sha256:
        raise SemanticAdapterError("canonical ROOT projector source is not the pinned version")
    return source


def _load(name: str, path: Path, source: bytes | None = None) -> ModuleType:
    """Execute exactly one captured source byte string; never consult bytecode."""
    path = Path(path)
    source = _source_bytes(path) if source is None else bytes(source)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SemanticAdapterError(f"cannot load pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(name)
    sys.modules[name] = module
    try:
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior
        raise
    return module


@dataclass(frozen=True)
class CanonicalGrade:
    source_grade: str
    trace_grade: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionSourcePin:
    root_commit: str
    claim_tuple_sha256: str
    adapter_sha256: str
    adapter_id: str
    measurement_capture_source_sha256: str = ""
    observation_binding_source_sha256: str = ""

    def to_dict(self, *, capture_schema: str) -> dict[str, str]:
        return {"root_commit": self.root_commit,
                "claim_tuple_sha256": self.claim_tuple_sha256,
                "adapter_sha256": self.adapter_sha256,
                "adapter_id": self.adapter_id, "projector_name": PROJECTOR_NAME,
                "capture_schema": capture_schema,
                "measurement_capture_source_sha256":
                    self.measurement_capture_source_sha256,
                "measurement_capture_producer_id": MEASUREMENT_CAPTURE_PRODUCER_V2,
                "measurement_capture_schema": CAPTURE_SCHEMA_V2,
                "observation_binding_source_sha256":
                    self.observation_binding_source_sha256,
                "observation_binding_producer_id": OBSERVATION_BINDING_PRODUCER,
                "observation_binding_schemas": dict(OBSERVATION_BINDING_SCHEMAS)}


def _final_v2_source_pin() -> ProjectionSourcePin | None:
    values = (FINAL_V2_ROOT_COMMIT, FINAL_V2_CLAIM_TUPLE_SHA256,
              FINAL_V2_ARM_PROJECTOR_SHA256, FINAL_V2_ADAPTER_ID,
              FINAL_V2_MEASUREMENT_CAPTURE_SHA256,
              FINAL_V2_OBSERVATION_BINDING_SHA256)
    if not all(isinstance(value, str) and value for value in values):
        return None
    root, claim, adapter, adapter_id, capture, observation = values
    return ProjectionSourcePin(root, claim, adapter, adapter_id, capture, observation)


class PinnedRootProjection:
    """An explicitly hash-pinned installation of ROOT's registered projector."""

    def __init__(self, root: Path, *, source_pin: ProjectionSourcePin | None = None,
                 source_closure: ProjectionSourceClosure | None = None):
        root = Path(root).resolve()
        self.source_closure = None
        if source_closure is not None:
            if source_pin is not None or type(source_closure) is not ProjectionSourceClosure:
                raise SemanticAdapterError("select one concrete current or historical source")
            from .feed_runtime import LoadedFeedProjection
            loaded = LoadedFeedProjection.load(root, source_closure.root_source_sha256)
            self.root, self.loaded_projection = root, loaded
            self.claim_tuple, self.arm_adapter = loaded.claim_tuple, loaded.adapter
            self.projector = loaded.claim_tuple.registered()[PROJECTOR_NAME]
            self.source_closure = source_closure
            self.source_pin = None
            from .validation_projection_source import source_identity
            self._installed_source_identity = source_identity(self)
            return
        claim_path = root / "scripts/vidya/claim_tuple.py"
        adapter_path = root / "scripts/vidya/adapters/autokernel_unified_arm.py"
        pin = source_pin or _final_v2_source_pin() or ProjectionSourcePin(
            "8fa14b0f8b53259db4c8a5991094bd06347b9a1e",
            CLAIM_TUPLE_SHA256, ARM_PROJECTOR_SHA256, ARM_ADAPTER_ID,
            "0" * 64, "0" * 64)
        with _LOAD_LOCK:
            claim_source = _source_bytes(claim_path, pin.claim_tuple_sha256)
            adapter_source = _source_bytes(adapter_path, pin.adapter_sha256)
            prior = sys.modules.get("claim_tuple")
            claim = _load("claim_tuple", claim_path, claim_source)
            try:
                adapter = _load(
                    "_autokernel_validation_pinned_unified_arm", adapter_path,
                    adapter_source)
            finally:
                if prior is None:
                    sys.modules.pop("claim_tuple", None)
                else:
                    sys.modules["claim_tuple"] = prior
        if adapter.ADAPTER_ID != pin.adapter_id:
            raise SemanticAdapterError("canonical arm adapter id changed")
        projector = claim.registered().get(PROJECTOR_NAME)
        if projector is None or projector is not adapter.project:
            raise SemanticAdapterError("canonical arm projector is not installed as registered")
        self.root = root
        self.claim_tuple = claim
        self.arm_adapter = adapter
        self.projector = projector
        self.source_pin = pin

    @property
    def native_v2_available(self) -> bool:
        return bool(
            self.source_closure is None and FINAL_V2_ROOT_COMMIT
            and FINAL_V2_CLAIM_TUPLE_SHA256
            and FINAL_V2_ARM_PROJECTOR_SHA256
            and FINAL_V2_ADAPTER_ID
            and FINAL_V2_MEASUREMENT_CAPTURE_SHA256
            and FINAL_V2_OBSERVATION_BINDING_SHA256
            and self.source_pin.root_commit == FINAL_V2_ROOT_COMMIT
            and self.source_pin.claim_tuple_sha256 == FINAL_V2_CLAIM_TUPLE_SHA256
            and self.source_pin.adapter_sha256 == FINAL_V2_ARM_PROJECTOR_SHA256
            and self.source_pin.adapter_id == FINAL_V2_ADAPTER_ID
            and self.source_pin.measurement_capture_source_sha256
            == FINAL_V2_MEASUREMENT_CAPTURE_SHA256
            and self.source_pin.observation_binding_source_sha256
            == FINAL_V2_OBSERVATION_BINDING_SHA256
            and hasattr(self.arm_adapter, "CAPTURE_SCHEMA_V2"))

    def _current_receipt_body(self, event, *, source_store, source_reference):
        from .validation_projection_source import build_receipt
        try:
            return build_receipt(self, event, source_store=source_store,
                                 source_reference=source_reference)
        except Exception as exc:
            raise SemanticAdapterError("canonical current source/provenance refused") from exc

    def _receipt_body(self, event: Mapping[str, Any], *, source_store: mc.ArtifactStore,
                      source_reference: Mapping[str, str]) -> dict[str, Any]:
        if self.source_closure is not None:
            return mc._plain(self._current_receipt_body(event, source_store=source_store,
                                                       source_reference=source_reference))
        try:
            payload = event["payload"]
            carrier = payload["carrier"]
            record_id = event["record_id"]
            loaded = carrier["loaded_instrument"]
            observations = carrier["lifecycle_observations"]
        except (KeyError, TypeError) as exc:
            raise SemanticAdapterError("canonical v2 event is malformed") from exc
        if (not isinstance(payload, Mapping) or not isinstance(carrier, Mapping)
                or not isinstance(loaded, Mapping) or not isinstance(observations, list)
                or payload.get("schema") != CAPTURE_SCHEMA_V2
                or carrier.get("schema") != CAPTURE_SCHEMA_V2
                or carrier.get("producer") != MEASUREMENT_CAPTURE_PRODUCER_V2
                or loaded.get("schema")
                != OBSERVATION_BINDING_SCHEMAS["loaded_instrument_reference"]
                or any(not isinstance(item, Mapping)
                       or item.get("schema")
                       != OBSERVATION_BINDING_SCHEMAS["lifecycle_observation_reference"]
                       for item in observations)):
            raise SemanticAdapterError("canonical v2 producer/schema identity differs")
        try:
            projected = self.arm_adapter.project_journal_event(
                dict(event), corpus_root=source_store.root)
        except (self.claim_tuple.ProjectionError, KeyError, TypeError, ValueError) as exc:
            raise SemanticAdapterError("canonical projector refused the v2 source") from exc
        if projected is None:
            raise SemanticAdapterError("canonical v2 source projected no measurement")
        claim = mc._plain(asdict(projected))
        grade, trace, reasons = self.claim_tuple.grade(projected)
        extra = projected.extra
        capture_schema = extra.get("capture_schema")
        if capture_schema != getattr(self.arm_adapter, "CAPTURE_SCHEMA_V2", None):
            raise SemanticAdapterError("canonical receipt requires native-v2 source")
        comparison = extra.get("comparison_identities")
        binding = {"measurement_id": record_id, "arm": extra.get("arm"),
                   "plan_digest": extra.get("plan_digest"),
                   "lineage_id": carrier.get("lineage_id"),
                   "comparison_identities_digest": schemas.content_hash(comparison),
                   "instrument_identity_sha256": extra.get("instrument_identity_sha256")}
        body = {"schema": cr.RECEIPT_SCHEMA, "producer": cr.PRODUCER_ID,
                "source_identity": self.source_pin.to_dict(capture_schema=capture_schema),
                "source_event": dict(source_reference),
                "projection": {"claim_tuple": claim,
                    "claim_tuple_digest": schemas.content_hash(claim),
                    "source_grade": grade, "trace_grade": trace,
                    "reasons": list(reasons)},
                "native_binding": binding,
                # These pins identify the verifier source loaded now.  Existing
                # native-v2 carriers do not bind the capture and observation
                # producer implementations that were loaded for the run, so
                # their provenance cannot be upgraded retrospectively.
                "authority_scope": "compatibility_only"}
        body["receipt_id"] = "claim-grade-" + schemas.content_hash(body)[:24]
        return dict(cr.validate_receipt_body(body))

    def produce_receipt(self, *, source_store: mc.ArtifactStore,
                        source_locator: str, source_sha256: str,
                        receipt_store: mc.ArtifactStore) -> cr.ClaimGradeReceiptReference:
        """Reopen one exact v2 Journal event and seal its canonical projection/grade."""
        source_reference = {"locator": source_locator, "sha256": source_sha256}
        event = mc._plain(source_store.read(source_locator, source_sha256))
        body = self._receipt_body(
            event, source_store=source_store, source_reference=source_reference)
        artifact = receipt_store.write("canonical-claim-grade-receipt", body)
        return cr.ClaimGradeReceiptReference(
            body["receipt_id"], artifact.locator, artifact.sha256)

    def reopen_receipt(self, *, reference: cr.ClaimGradeReceiptReference,
                       source_store: mc.ArtifactStore,
                       receipt_store: mc.ArtifactStore) -> Mapping[str, Any]:
        """Reproject exact source bytes and compare the sole grader's full result."""
        reference = cr.ClaimGradeReceiptReference.from_dict(reference.to_dict())
        try:
            body = cr.validate_receipt_body(mc._plain(
                receipt_store.read(reference.locator, reference.sha256)))
        except (ValueError, mc.CaptureError, mc.SecureRuntimeError) as exc:
            raise SemanticAdapterError("canonical grade receipt cannot be reopened") from exc
        if self.source_closure is not None:
            from .validation_projection_source import source_identity
            expected_source = mc._plain(source_identity(self))
        else:
            expected_source = self.source_pin.to_dict(capture_schema=body["source_identity"]["capture_schema"])
        if (body["receipt_id"] != reference.receipt_id
                or mc._plain(body["source_identity"]) != expected_source):
            raise SemanticAdapterError("canonical grade receipt source identity changed")
        source = body["source_event"]
        try:
            event = mc._plain(source_store.read(source["locator"], source["sha256"]))
            expected = self._receipt_body(
                event, source_store=source_store, source_reference=source)
        except (ValueError, mc.CaptureError, mc.SecureRuntimeError) as exc:
            raise SemanticAdapterError("canonical grade source cannot be reopened") from exc
        if mc._plain(body) != expected:
            raise SemanticAdapterError("canonical grade receipt differs from reprojection")
        return body

    def grade_v1(self, source: Mapping[str, Any], *, receipt_locator: str,
                 receipt_sha256: str, corpus_root: Path | None = None) -> CanonicalGrade:
        """Project and grade v1 bytes; this is diagnostic, never v2 authority."""
        try:
            rows = self.arm_adapter.native_rows(
                dict(source), receipt_locator=receipt_locator,
                receipt_sha256=receipt_sha256, attestation_present=True,
                corpus_root=corpus_root)
        except self.claim_tuple.ProjectionError as exc:
            raise SemanticAdapterError("canonical projector refused the native source") from exc
        if len(rows) != 1:
            raise SemanticAdapterError("canonical source is not a complete measurement row")
        grade, trace, reasons = self.claim_tuple.grade(self.projector(rows[0]))
        return CanonicalGrade(grade, trace, tuple(reasons))


@dataclass(frozen=True)
class ReopenedClaimReceiptPair:
    plan: ep.ExperimentPlan
    view: ep.AdmissibleUnitView
    anchor: Mapping[str, Any]
    candidate: Mapping[str, Any]
    anchor_carrier: Mapping[str, Any]
    candidate_carrier: Mapping[str, Any]


@dataclass(frozen=True)
class RegisteredClaimGradeVerifier:
    """One concrete receipt owner for historical facts and strict grade checking.

    A retained native validator may identify an application installation, but
    historical replay never invokes it or reconstructs its original registry.
    """

    projection: PinnedRootProjection
    source_store: mc.ArtifactStore
    receipt_store: mc.ArtifactStore
    native_validator: Any = None

    def __post_init__(self):
        from .native_capture_control import NativeCaptureValidator
        if (type(self.projection) is not PinnedRootProjection
                or type(self.source_store) is not mc.ArtifactStore
                or type(self.receipt_store) is not mc.ArtifactStore
                or self.native_validator is not None
                and type(self.native_validator) is not NativeCaptureValidator):
            raise SemanticAdapterError("claim-grade verifier requires concrete installed dependencies")

    def reopen_pair(self, pair: cr.ClaimGradeReceiptPairReference) -> ReopenedClaimReceiptPair:
        from . import observation_binding as ob
        from . import validation_projection_source as current
        if type(pair) is not cr.ClaimGradeReceiptPairReference or self.projection.source_closure is None:
            raise SemanticAdapterError("current historical replay requires a concrete receipt pair/source")
        pair = cr.ClaimGradeReceiptPairReference.from_dict(pair.to_dict())
        receipts = ValidationSemanticAdapter(self.projection).reopen_receipt_pair(
            anchor=pair.anchor, candidate=pair.candidate,
            source_store=self.source_store, receipt_store=self.receipt_store)
        native = []
        for body in receipts:
            source = body["source_event"]
            event = mc._plain(self.source_store.read(source["locator"], source["sha256"]))
            native.append(current.reopen_native(self.projection, event, self.source_store))
        _a_projection, a, plan, view, _a_provenance = native[0]
        _c_projection, c, other_plan, other_view, _c_provenance = native[1]
        if (a["measurement_id"] != pair.anchor_measurement_id
                or c["measurement_id"] != pair.candidate_measurement_id
                or plan.to_dict() != other_plan.to_dict() or view.to_dict() != other_view.to_dict()
                or view.view_digest != pair.admissible_view_digest
                or a["parent_final_trial"] != c["parent_final_trial"]
                or receipts[0]["source_identity"] != receipts[1]["source_identity"]):
            raise SemanticAdapterError("canonical receipt pair differs from original full final evidence")
        return ReopenedClaimReceiptPair(plan, view, receipts[0], receipts[1], ob._freeze(a), ob._freeze(c))

    def verify(self, plan: ep.ExperimentPlan, view: ep.AdmissibleUnitView,
               pair: cr.ClaimGradeReceiptPairReference) -> ep.AdmissibleUnitView:
        """Strict continuation interface; a diagnostic cannot discharge a grade."""
        plan = ep.ExperimentPlan.from_dict(plan.to_dict())
        supplied = ep._validated_view(plan, view)
        actual = self.reopen_pair(pair)
        if (actual.plan.to_dict() != plan.to_dict() or actual.view.to_dict() != supplied.to_dict()
                or not actual.view.complete or any(
                    body["projection"].get("status") != "measurement"
                    or (body["projection"].get("source_grade"), body["projection"].get("trace_grade"))
                    != ("Witnessed", "Attested") or body["authority_scope"] != "final_pinned_source"
                    for body in (actual.anchor, actual.candidate))):
            raise SemanticAdapterError("canonical receipt pair lacks complete qualified measurement evidence")
        return actual.view


class ValidationSemanticAdapter:
    """Consumer-shaped adapter that preserves current unavailable authority."""

    def __init__(self, projection: PinnedRootProjection):
        self.projection = projection

    def reopen_receipt_pair(
            self, *, anchor: cr.ClaimGradeReceiptReference,
            candidate: cr.ClaimGradeReceiptReference,
            source_store: mc.ArtifactStore,
            receipt_store: mc.ArtifactStore) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Reopen two canonical receipts and enforce their comparison identity."""
        left = self.projection.reopen_receipt(
            reference=anchor, source_store=source_store, receipt_store=receipt_store)
        right = self.projection.reopen_receipt(
            reference=candidate, source_store=source_store, receipt_store=receipt_store)
        a_binding, c_binding = left["native_binding"], right["native_binding"]
        if (a_binding["arm"] != "anchor" or c_binding["arm"] != "candidate"
                or a_binding["plan_digest"] != c_binding["plan_digest"]
                or a_binding["lineage_id"] != c_binding["lineage_id"]
                or a_binding["comparison_identities_digest"]
                != c_binding["comparison_identities_digest"]
                or a_binding["instrument_identity_sha256"]
                != c_binding["instrument_identity_sha256"]):
            raise SemanticAdapterError("canonical receipt pair identity differs")
        if self.projection.source_closure is not None and (
                a_binding["parent_final_trial"] != c_binding["parent_final_trial"]
                or a_binding["final_view_digest"] != c_binding["final_view_digest"]):
            raise SemanticAdapterError("canonical receipt pair final identity differs")
        return left, right

    def evaluate_receipt_pair(
            self, plan: ep.ExperimentPlan, *,
            anchor: cr.ClaimGradeReceiptReference,
            candidate: cr.ClaimGradeReceiptReference,
            source_store: mc.ArtifactStore,
            receipt_store: mc.ArtifactStore) -> vc.SemanticDecision:
        """Consume canonical receipts while keeping selection/promotion distinct."""
        plan = ep.ExperimentPlan.from_dict(plan.to_dict())
        left, right = self.reopen_receipt_pair(
            anchor=anchor, candidate=candidate, source_store=source_store,
            receipt_store=receipt_store)
        if self.projection.source_closure is not None and any(
                item["native_binding"]["plan_digest"] != plan.digest for item in (left, right)):
            raise SemanticAdapterError("current canonical receipt pair belongs to a different plan")
        if self.projection.source_closure is not None and any(
                item["projection"]["status"] == "diagnostic" for item in (left, right)):
            return vc.SemanticDecision("Unavailable", "Unavailable", "validate_production", False,
                ("canonical final-trial projection is diagnostic; no ClaimTuple or grade exists",))
        grades = {(item["projection"]["source_grade"],
                   item["projection"]["trace_grade"]) for item in (left, right)}
        authority_scopes = {item["authority_scope"] for item in (left, right)}
        binding = left["native_binding"]
        reasons: list[str] = []
        if binding["plan_digest"] != plan.digest:
            reasons.append("canonical receipt pair belongs to a different plan")
        if grades != {("Witnessed", "Attested")}:
            reasons.append("canonical receipt pair is below Witnessed/Attested")
        if self.projection.source_closure is None and not self.projection.native_v2_available:
            reasons.append("native-v2 projector/producer identity is compatibility-only")
        if authority_scopes != {"final_pinned_source"}:
            reasons.append("native producer source identity is compatibility-only")
        # Canonical grading discharges the missing grader connection only.  It
        # does not decide the experimental serving comparison or production use.
        reasons.append("owning validation objective/serving decision is required")
        return vc.SemanticDecision(
            next(iter(grades))[0] if len(grades) == 1 else "Unavailable",
            next(iter(grades))[1] if len(grades) == 1 else "Unavailable",
            "validate_production", False, tuple(reasons))

    def evaluate(self, plan: ep.ExperimentPlan,
                 evidence: Mapping[str, Any], *,
                 claim_grade_verifier: RegisteredClaimGradeVerifier | None = None) -> vc.SemanticDecision:
        plan = ep.ExperimentPlan.from_dict(plan.to_dict())
        if not isinstance(evidence, Mapping):
            raise SemanticAdapterError("semantic evidence must be a mapping")
        if claim_grade_verifier is not None:
            if type(claim_grade_verifier) is not RegisteredClaimGradeVerifier:
                raise SemanticAdapterError("semantic receipt verifier must be the concrete registration")
            actual = claim_grade_verifier.reopen_pair(evidence.get("claim_grade_pair"))
            if actual.plan.to_dict() != plan.to_dict():
                raise SemanticAdapterError("semantic receipt pair belongs to another plan")
            if any(body["projection"]["status"] == "diagnostic" for body in (actual.anchor, actual.candidate)):
                return vc.SemanticDecision("Unavailable", "Unavailable", "validate_production", False,
                    ("canonical final-trial projection is diagnostic; no ClaimTuple or grade exists",
                     "complete_native_measurement_pair", "original_control_evidence_unavailable",
                     "original_calibration_evidence_unavailable"))
            # The same strict registration serves the future owning qualified
            # serving-decision route; source/grade proof alone is not that rule.
            claim_grade_verifier.verify(plan, actual.view, evidence["claim_grade_pair"])
            return vc.SemanticDecision("Witnessed", "Attested", "validate_production", False,
                ("owning qualified measurement/serving decision is required",))
        return vc.SemanticDecision(
            "Unavailable", "Unavailable", "validate_production", False,
            ("canonical native-v2 projector is not installed",
             "owning production-validation numerical policy/result is undefined"))

    def registered_authority(self, transaction_verifier_id: str, *,
                             source_store: mc.ArtifactStore | None = None,
                             receipt_store: mc.ArtifactStore | None = None,
                             native_validator: Any = None) \
            -> vc.RegisteredSemanticAuthority:
        """Expose a real capability whose current result is an explicit refusal."""
        if source_store is not None or receipt_store is not None or native_validator is not None:
            verifier = RegisteredClaimGradeVerifier(self.projection, source_store, receipt_store, native_validator)
            return vc.RegisteredSemanticAuthority(AUTHORITY_ID, transaction_verifier_id,
                RegisteredServingSemanticEvaluator(self, verifier), fixture_only=False,
                claim_grade_verifier=verifier)
        return vc.RegisteredSemanticAuthority(
            AUTHORITY_ID, transaction_verifier_id, self.evaluate, fixture_only=False)


@dataclass(frozen=True)
class RegisteredServingSemanticEvaluator:
    adapter: ValidationSemanticAdapter
    claim_grade_verifier: RegisteredClaimGradeVerifier

    def __post_init__(self):
        if (type(self.adapter) is not ValidationSemanticAdapter
                or type(self.claim_grade_verifier) is not RegisteredClaimGradeVerifier
                or self.adapter.projection is not self.claim_grade_verifier.projection):
            raise SemanticAdapterError("semantic evaluator requires concrete installed dependencies")

    def __call__(self, plan, evidence):
        return self.adapter.evaluate(plan, evidence, claim_grade_verifier=self.claim_grade_verifier)


__all__ = ["ARM_ADAPTER_ID", "ARM_PROJECTOR_SHA256", "AUTHORITY_ID",
           "CLAIM_TUPLE_SHA256", "CanonicalGrade", "PinnedRootProjection",
           "ProjectionSourceClosure", "RegisteredClaimGradeVerifier", "RegisteredServingSemanticEvaluator",
           "SemanticAdapterError", "ValidationSemanticAdapter"]
