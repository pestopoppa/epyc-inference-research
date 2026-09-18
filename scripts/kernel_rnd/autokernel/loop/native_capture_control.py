"""Validate prospective native carriers before the controller journals them.

This module has no writer, WAL, admission, or scientific-use authority.  It validates
the frozen producer contract, verifies already-published artifacts without creating
them, and requires a typed result fence supplied by the actual worker lifecycle owner.
The campaign controller remains responsible for serialized lookup/append/replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from statistics import median
from typing import Any, Callable, Mapping

from .. import schemas
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import observation_binding as ob
from . import planned_serving as ps


class NativeCaptureRefused(mc.CaptureError):
    """A carrier lacks exact structure, artifacts, or trusted result ownership."""


@dataclass(frozen=True)
class PrevalidatedNativeCapture:
    """Large immutable v2 artifact work completed outside the controller mutex."""

    measurement_id: str
    payload_digest: str
    context: mc.CaptureContext
    grant_generation: int
    observation_links: tuple[ob.ValidatedObservationLink, ...]
    validated: ValidatedNativeCapture


@dataclass(frozen=True)
class CurrentOwnerToken:
    """Short-lived controller-mutex token for one exact prevalidated payload."""

    measurement_id: str
    payload_digest: str
    grant_generation: int
    fence: TrustedWorkerResultFence


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeCaptureRefused(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise NativeCaptureRefused(f"{label} must be lowercase SHA-256")
    return value


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise NativeCaptureRefused(f"{label} must be a positive integer")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise NativeCaptureRefused("JSON objects require non-empty text keys")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise NativeCaptureRefused("capture values must be finite canonical JSON")


@dataclass(frozen=True)
class NativeCaptureBinding:
    campaign_id: str
    config_digest: str
    config_generation: int
    supervisor_id: str
    supervisor_incarnation: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeCaptureBinding":
        fields = {"campaign_id", "config_digest", "config_generation",
                  "supervisor_id", "supervisor_incarnation"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise NativeCaptureRefused("native capture binding has missing/unknown fields")
        return cls(_text(value["campaign_id"], "binding campaign_id"),
                   _sha(value["config_digest"], "binding config_digest"),
                   _positive(value["config_generation"], "binding config_generation"),
                   _text(value["supervisor_id"], "binding supervisor_id"),
                   _positive(value["supervisor_incarnation"],
                             "binding supervisor_incarnation"))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TrustedWorkerResultFence:
    campaign_id: str
    config_digest: str
    config_generation: int
    supervisor_id: str
    supervisor_incarnation: int
    worker_id: str
    worker_incarnation: int
    grant_id: str
    container_id: str
    lineage_id: str
    current: bool
    result_accepted: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrustedWorkerResultFence":
        fields = {"campaign_id", "config_digest", "config_generation",
                  "supervisor_id", "supervisor_incarnation", "worker_id",
                  "worker_incarnation", "grant_id", "container_id", "lineage_id",
                  "current", "result_accepted"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise NativeCaptureRefused("trusted result fence has missing/unknown fields")
        for name in ("current", "result_accepted"):
            if not isinstance(value[name], bool):
                raise NativeCaptureRefused(f"trusted result fence {name} must be boolean")
        return cls(
            _text(value["campaign_id"], "fence campaign_id"),
            _sha(value["config_digest"], "fence config_digest"),
            _positive(value["config_generation"], "fence config_generation"),
            _text(value["supervisor_id"], "fence supervisor_id"),
            _positive(value["supervisor_incarnation"], "fence supervisor_incarnation"),
            _text(value["worker_id"], "fence worker_id"),
            _positive(value["worker_incarnation"], "fence worker_incarnation"),
            _text(value["grant_id"], "fence grant_id"),
            _text(value["container_id"], "fence container_id"),
            _text(value["lineage_id"], "fence lineage_id"),
            value["current"], value["result_accepted"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


FenceProvider = Callable[[str, mc.CaptureContext], TrustedWorkerResultFence]


@dataclass(frozen=True)
class ValidatedNativeCapture:
    measurement_id: str
    payload_digest: str
    status: str
    scientific_progress: bool
    _payload_json: str

    def payload(self) -> dict[str, Any]:
        value = json.loads(self._payload_json)
        assert isinstance(value, dict)
        return value


class NativeCaptureValidator:
    """Validate one new carrier against controller identity and current result fence."""

    def __init__(self, *, binding: NativeCaptureBinding, store: mc.ArtifactStore,
                 fence_provider: FenceProvider | None = None,
                 observation_verifiers: ob.ParentObservationVerifiers =
                 ob.ParentObservationVerifiers(),
                 parent_receipt_replayer: Any | None = None) -> None:
        if not isinstance(binding, NativeCaptureBinding):
            raise NativeCaptureRefused("binding must be NativeCaptureBinding")
        self.binding = NativeCaptureBinding.from_dict(binding.to_dict())
        if not isinstance(store, mc.ArtifactStore):
            raise NativeCaptureRefused("store must be the accepted ArtifactStore")
        self.store = store
        if fence_provider is not None and not callable(fence_provider):
            raise NativeCaptureRefused("fence_provider must be callable")
        self.fence_provider = fence_provider
        if not isinstance(observation_verifiers, ob.ParentObservationVerifiers):
            raise NativeCaptureRefused("observation verifiers must be the concrete adapter set")
        self.observation_verifiers = observation_verifiers
        if parent_receipt_replayer is not None:
            from .native_parent_receipt_replay import NativeParentReceiptReplayer
            if type(parent_receipt_replayer) is not NativeParentReceiptReplayer:
                raise NativeCaptureRefused("parent receipt replay requires its concrete adapter")
        self.parent_receipt_replayer = parent_receipt_replayer

    def validate(self, measurement_id: str,
                 payload: Mapping[str, Any]) -> ValidatedNativeCapture:
        measurement_id = _sha(measurement_id, "measurement_id")
        row = _plain(payload)
        fields = {"schema", "measurement_id", "carrier", "artifact"}
        if not isinstance(row, dict) or set(row) != fields:
            raise NativeCaptureRefused("native capture payload has missing/unknown fields")
        if row["schema"] == mc.CAPTURE_SCHEMA_V2:
            raise NativeCaptureRefused(
                "v2 capture requires outside-lock prevalidation and a current-owner token")
        if row["schema"] != mc.CAPTURE_SCHEMA:
            raise NativeCaptureRefused("unsupported native capture payload schema")
        if row["measurement_id"] != measurement_id:
            raise NativeCaptureRefused("payload measurement_id differs from callback identity")
        carrier = row["carrier"]
        if not isinstance(carrier, dict):
            raise NativeCaptureRefused("native capture carrier must be an object")
        try:
            self._validate_carrier(measurement_id, carrier)
            context = mc.CaptureContext.from_dict(carrier["capture_context"])
        except NativeCaptureRefused:
            raise
        except Exception as exc:
            raise NativeCaptureRefused("native carrier failed producer-contract validation") from exc
        self._validate_binding(context)
        self._validate_fence(measurement_id, context)
        try:
            self._verify_artifacts(measurement_id, carrier, row["artifact"])
        except NativeCaptureRefused:
            raise
        except Exception as exc:
            raise NativeCaptureRefused(
                "native carrier raw/artifact validation failed") from exc
        return ValidatedNativeCapture(
            measurement_id, schemas.content_hash(row), carrier["status"], False,
            json.dumps(row, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False))

    def prevalidate(self, measurement_id: str,
                    payload: Mapping[str, Any]) -> PrevalidatedNativeCapture:
        """Perform all potentially large v2 reads before controller serialization."""
        from . import native_final_trial as final
        if isinstance(payload, Mapping) and payload.get("schema") == final.CAPTURE_SCHEMA:
            from .native_server_t0_witness import NativeServerT0WitnessAdapter
            if self.parent_receipt_replayer is None:
                raise NativeCaptureRefused("final capture original parent replay is unavailable")
            selected = self.parent_receipt_replayer.selected_scientific_adapters()
            if selected is None or type(selected.correctness) is not NativeServerT0WitnessAdapter:
                raise NativeCaptureRefused("final capture requires the concrete original server T0 owner")
            return selected.correctness.final_trial_owner.prevalidate(self, measurement_id, payload)
        measurement_id = _sha(measurement_id, "measurement_id")
        row = _plain(payload)
        if (not isinstance(row, dict)
                or set(row) != {"schema", "measurement_id", "carrier", "artifact"}
                or row["schema"] != mc.CAPTURE_SCHEMA_V2
                or row["measurement_id"] != measurement_id):
            raise NativeCaptureRefused("v2 native capture payload is malformed")
        carrier = row["carrier"]
        if not isinstance(carrier, dict):
            raise NativeCaptureRefused("v2 native carrier must be an object")
        self._validate_carrier(measurement_id, carrier)
        context = mc.CaptureContext.from_dict(carrier["capture_context"])
        self._validate_binding(context)
        self._verify_artifacts(measurement_id, carrier, row["artifact"])
        grant_generation, links = self._verify_observations(carrier)
        from .native_producer_source import verify_capture_producer_source
        verify_capture_producer_source(carrier, store=self.store, validator=self)
        from .native_parent_receipt_replay import has_parent_receipt_refs
        if self.parent_receipt_replayer is not None:
            self.parent_receipt_replayer.replay(carrier, store=self.store)
        elif has_parent_receipt_refs(carrier):
            raise NativeCaptureRefused("original parent-issued receipt replay authority is unavailable")
        validated = ValidatedNativeCapture(
            measurement_id, schemas.content_hash(row), carrier["status"], False,
            json.dumps(row, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False))
        return PrevalidatedNativeCapture(measurement_id, schemas.content_hash(row),
                                         context, grant_generation, links, validated)

    def validate_prevalidated(self, prevalidated: PrevalidatedNativeCapture,
                              token: CurrentOwnerToken) -> ValidatedNativeCapture:
        """Check only exact current ownership; caller holds the controller mutex."""
        if type(prevalidated) is not PrevalidatedNativeCapture \
                or type(token) is not CurrentOwnerToken:
            raise NativeCaptureRefused("typed prevalidation/current-owner token required")
        if (token.measurement_id != prevalidated.measurement_id
                or token.payload_digest != prevalidated.payload_digest
                or isinstance(token.grant_generation, bool)
                or token.grant_generation < 1
                or token.grant_generation != prevalidated.grant_generation):
            raise NativeCaptureRefused("current-owner token binds another payload")
        fence = TrustedWorkerResultFence.from_dict(token.fence.to_dict())
        self._validate_supplied_fence(prevalidated.measurement_id,
                                      prevalidated.context, fence)
        return prevalidated.validated

    def _validate_binding(self, context: mc.CaptureContext) -> None:
        expected = self.binding
        if (context.campaign_id != expected.campaign_id
                or context.config_digest != expected.config_digest
                or context.config_generation != expected.config_generation
                or context.supervisor_id != expected.supervisor_id
                or context.supervisor_incarnation != expected.supervisor_incarnation):
            raise NativeCaptureRefused("capture context differs from controller binding")

    def _validate_fence(self, measurement_id: str, context: mc.CaptureContext) -> None:
        if self.fence_provider is None:
            raise NativeCaptureRefused("trusted worker-result fence provider is not connected")
        try:
            supplied = self.fence_provider(measurement_id, context)
        except Exception as exc:
            raise NativeCaptureRefused("trusted worker-result fence callback failed") from exc
        if not isinstance(supplied, TrustedWorkerResultFence):
            raise NativeCaptureRefused(
                "trusted worker-result fence provider returned an untyped value")
        fence = TrustedWorkerResultFence.from_dict(supplied.to_dict())
        self._validate_supplied_fence(measurement_id, context, fence)

    def _validate_supplied_fence(self, measurement_id: str, context: mc.CaptureContext,
                                 fence: TrustedWorkerResultFence) -> None:
        pairs = (
            (fence.campaign_id, context.campaign_id),
            (fence.config_digest, context.config_digest),
            (fence.config_generation, context.config_generation),
            (fence.supervisor_id, context.supervisor_id),
            (fence.supervisor_incarnation, context.supervisor_incarnation),
            (fence.worker_id, context.worker_id),
            (fence.worker_incarnation, context.worker_incarnation),
            (fence.grant_id, context.grant_id),
            (fence.container_id, context.container_id),
            (fence.lineage_id, context.lineage_id),
        )
        if any(actual != expected for actual, expected in pairs):
            raise NativeCaptureRefused("trusted worker-result fence identity mismatch")
        if not fence.current:
            raise NativeCaptureRefused("trusted worker-result fence is stale")
        if not fence.result_accepted:
            raise NativeCaptureRefused("trusted worker-result fence did not accept this result")

    def _validate_carrier(self, measurement_id: str, carrier: dict[str, Any]) -> None:
        fields = {
            "schema", "producer", "measurement_id", "arm", "arm_locator", "plan",
            "prompt_manifest", "prompt_manifest_digest", "lineage_id",
            "comparison_identities", "source_identity", "capture_context",
            "admissible_view", "raw_artifacts", "environment_verdicts", "status",
            "diagnostic_reason", "measurement", "claim", "category", "phase",
            "record_class", "intended_use", "protocol_id", "protocol_status",
            "instrument_id", "interval", "carrier_digest",
        }
        v2 = carrier.get("schema") == mc.CAPTURE_SCHEMA_V2
        if v2:
            fields |= {"loaded_instrument", "lifecycle_observations"}
        if set(carrier) != fields:
            raise NativeCaptureRefused("native carrier has missing/unknown fields")
        if ((not v2 and (carrier["schema"] != mc.CAPTURE_SCHEMA
                         or carrier["producer"] != mc.PRODUCER_ID))
                or (v2 and carrier["producer"] != mc.PRODUCER_ID_V2)):
            raise NativeCaptureRefused("native carrier schema/producer is unsupported")
        if carrier["measurement_id"] != measurement_id:
            raise NativeCaptureRefused("carrier measurement_id differs from callback identity")
        arm = carrier["arm"]
        if not isinstance(arm, str) or arm not in {"anchor", "candidate"}:
            raise NativeCaptureRefused("native carrier arm is invalid")
        plan = ep.ExperimentPlan.from_dict(carrier["plan"])
        identity = {"producer": carrier["producer"], "plan_digest": plan.digest,
                    "lineage_id": carrier["lineage_id"], "arm": arm}
        if v2:
            if plan.schema != ep.PLAN_SCHEMA_V2 \
                    or carrier["loaded_instrument"] != _plain(plan.loaded_instrument):
                raise NativeCaptureRefused("v2 carrier loaded instrument differs from plan")
            identity |= {"capture_schema": mc.CAPTURE_SCHEMA_V2,
                         "instrument_identity_sha256":
                             carrier["loaded_instrument"].get("identity_sha256")}
        expected_id = schemas.content_hash(identity)
        if expected_id != measurement_id:
            raise NativeCaptureRefused("measurement_id does not bind plan/lineage/arm")
        if carrier["arm_locator"] != \
                f"planned-serving:{plan.digest}:{carrier['lineage_id']}:{arm}":
            raise NativeCaptureRefused("arm locator does not bind plan/lineage/arm")
        prompts = ps.FrozenPromptManifest.from_dict(carrier["prompt_manifest"])
        if carrier["prompt_manifest_digest"] != prompts.digest:
            raise NativeCaptureRefused("prompt manifest digest mismatch")
        context = mc.CaptureContext.from_dict(carrier["capture_context"])
        if (plan.campaign_id != context.campaign_id
                or plan.protocol_ref != context.protocol_id
                or plan.protocol_status != context.protocol_status
                or carrier["lineage_id"] != context.lineage_id
                or carrier["protocol_id"] != context.protocol_id
                or carrier["protocol_status"] != context.protocol_status
                or carrier["instrument_id"] != context.instrument_id):
            raise NativeCaptureRefused("carrier plan/protocol/context binding mismatch")
        for key in ("category", "phase", "record_class", "intended_use"):
            if carrier[key] != getattr(plan, key):
                raise NativeCaptureRefused(f"carrier {key} differs from frozen plan")
        _text(carrier["claim"], "carrier claim")
        identities = carrier["comparison_identities"]
        if not isinstance(identities, dict) or set(identities) != {"anchor", "candidate"}:
            raise NativeCaptureRefused("comparison identities must contain both arms")
        expected_identities = {
            "anchor": _plain(plan.anchor_identity),
            "candidate": _plain(plan.candidate_identity),
        }
        if identities != expected_identities:
            raise NativeCaptureRefused(
                "comparison identities differ from the frozen plan")
        source = context.source_identities[arm]
        if carrier["source_identity"] != _plain(source):
            raise NativeCaptureRefused("carrier source identity differs from capture context")
        for identity_arm in ("anchor", "candidate"):
            identity = identities[identity_arm]
            identity_source = context.source_identities[identity_arm]
            if (not isinstance(identity, dict)
                    or identity.get("model_digest") != identity_source["model_sha256"]
                    or identity.get("executable_digest") != identity_source["build_sha256"]
                    or identity.get("template_hash") != identity_source["recipe_hash"]):
                raise NativeCaptureRefused(
                    "carrier execution identity differs from source context")
        self._validate_view(plan, carrier["admissible_view"])
        self._validate_environment(carrier["environment_verdicts"])
        body = dict(carrier)
        digest = body.pop("carrier_digest")
        if _sha(digest, "carrier_digest") != schemas.content_hash(body):
            raise NativeCaptureRefused("carrier digest mismatch")

    @staticmethod
    def _validate_view(plan: ep.ExperimentPlan, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise NativeCaptureRefused("admissible view must be an object")
        fields = {"schema", "plan_digest", "selected_rows", "rejection_reasons",
                  "missing_expected_units", "independent_n", "complete", "view_digest"}
        if set(value) != fields or not isinstance(value["selected_rows"], list):
            raise NativeCaptureRefused("admissible view is malformed")
        try:
            view = ep.AdmissibleUnitView(
                value["schema"], value["plan_digest"],
                tuple(ep.RawUnit.from_dict(item) for item in value["selected_rows"]),
                value["rejection_reasons"], tuple(value["missing_expected_units"]),
                value["independent_n"], value["complete"], value["view_digest"])
            normalized = ep._validated_view(plan, view)
        except Exception as exc:
            raise NativeCaptureRefused("admissible view failed structural validation") from exc
        if normalized.to_dict() != _plain(value):
            raise NativeCaptureRefused("admissible view is not canonical")

    @staticmethod
    def _validate_status(plan: ep.ExperimentPlan, carrier: Mapping[str, Any],
                         attempts: list[Mapping[str, Any]],
                         raw_artifacts: list[Mapping[str, Any]]) -> None:
        arm = carrier["arm"]
        rows = [row for row in carrier["admissible_view"]["selected_rows"]
                if row["arm"] == arm]
        try:
            diagnostic = mc.NativeMeasurementSink._diagnostic(
                plan, arm, rows, attempts, raw_artifacts)
        except Exception as exc:
            raise NativeCaptureRefused(
                "native carrier diagnostic derivation failed") from exc
        supported_measurement = (
            plan.instrument_class == "serving"
            and plan.metric == "aggregate_tok_s"
            and plan.metric_direction == "higher"
            and plan.unit == "process"
            and plan.estimator_id == "median.v1"
            and plan.estimand == "level"
            and (carrier["instrument_id"] == "planned-serving/v1"
                 if carrier["schema"] == mc.CAPTURE_SCHEMA else
                 carrier["instrument_id"]
                 == carrier["loaded_instrument"]["identity_sha256"])
        )
        if diagnostic is None and not supported_measurement:
            raise NativeCaptureRefused(
                "unsupported native instrument/metric/unit/estimator/estimand semantics")
        values = [float(row["value"]) for row in rows]
        expected_measurement = None
        if diagnostic is None and values:
            if plan.estimator_id != "median.v1":
                diagnostic = f"unsupported estimator {plan.estimator_id!r}"
            else:
                expected_measurement = {
                    "metric": plan.metric, "value": median(values), "unit": "t/s",
                    "independent_unit": plan.unit,
                    "direction": plan.metric_direction,
                    "independent_n": len(values),
                    "reps_basis": "scored independent process launches",
                    "per_launch_values": values,
                }
        expected_status = "measurement" if expected_measurement is not None else "diagnostic"
        if (carrier["status"] != expected_status
                or carrier["diagnostic_reason"] != diagnostic
                or not NativeCaptureValidator._same_json(
                    carrier["measurement"], expected_measurement)
                or carrier["claim"] !=
                f"{arm} {plan.metric} for frozen plan {plan.plan_id}"):
            raise NativeCaptureRefused(
                "carrier status/scalar is not derived from retained native inputs")

    @staticmethod
    def _same_json(left: Any, right: Any) -> bool:
        try:
            return json.dumps(left, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False) == json.dumps(
                                  right, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _validate_environment(value: Any) -> None:
        if not isinstance(value, list):
            raise NativeCaptureRefused("environment verdicts must be a list")
        allowed = {
            "contention": {"clean", "contaminated", "unknown"},
            "placement": {"proven", "refuted", "unknown"},
            "residency": {"proven", "refuted", "unknown", "not_applicable"},
        }
        for row in value:
            if not isinstance(row, Mapping) or set(row) != {
                    "unit_id", "contention", "placement", "residency"}:
                raise NativeCaptureRefused("environment verdict row is malformed")
            _text(row["unit_id"], "environment unit_id")
            for name, states in allowed.items():
                item = row[name]
                if (not isinstance(item, Mapping) or set(item) != {"verdict", "ref"}
                        or not isinstance(item["verdict"], str)
                        or item["verdict"] not in states
                        or (item["ref"] is not None
                            and not isinstance(item["ref"], str))):
                    raise NativeCaptureRefused(
                        f"environment {name} verdict is malformed")

    def _verify_artifacts(self, measurement_id: str, carrier: Mapping[str, Any],
                          carrier_artifact: Any) -> None:
        raw_artifacts = carrier["raw_artifacts"]
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise NativeCaptureRefused("carrier requires retained raw artifacts")
        intervals: list[tuple[datetime, datetime, str, str]] = []
        documents: dict[str, Mapping[str, Any]] = {}
        for index, item in enumerate(raw_artifacts):
            if not isinstance(item, Mapping) or set(item) != {"document", "stored"}:
                raise NativeCaptureRefused(f"raw_artifacts[{index}] is malformed")
            document = item["document"]
            if not isinstance(document, Mapping):
                raise NativeCaptureRefused(f"raw_artifacts[{index}].document must be object")
            document = dict(document)
            artifact_digest = document.pop("artifact_digest", None)
            if _sha(artifact_digest, "raw artifact digest") != schemas.content_hash(document):
                raise NativeCaptureRefused("raw artifact digest mismatch")
            document["artifact_digest"] = artifact_digest
            if artifact_digest in documents:
                raise NativeCaptureRefused("raw artifact digest is duplicated")
            documents[artifact_digest] = document
            expected_artifact_schema = (ps.ARTIFACT_SCHEMA_V2
                                        if carrier["schema"] == mc.CAPTURE_SCHEMA_V2
                                        else ps.ARTIFACT_SCHEMA)
            selected_artifact = (document.get("schema") == ps.ARTIFACT_SCHEMA_V3
                                 and document.get("kind") == "completed_attempt"
                                 and expected_artifact_schema == ps.ARTIFACT_SCHEMA_V2)
            if document.get("schema") != expected_artifact_schema and not selected_artifact:
                raise NativeCaptureRefused("raw artifact schema is unsupported")
            kind = document.get("kind")
            if not isinstance(kind, str) or kind not in {
                    "continued_unit", "native_observation", "completed_attempt"}:
                raise NativeCaptureRefused("raw artifact kind is unsupported")
            shapes = {
                "continued_unit": {
                    "schema", "kind", "plan_digest", "unit_id", "arm",
                    "process_generation_id", "lineage_id",
                    "continued_from_lineage_id", "continued_from_artifact_digest",
                    "prompt_manifest_digest", "prompt_ids", "comparison_identities",
                    "terminal", "value", "recorded_screen", "reason",
                    "artifact_digest",
                },
                "native_observation": {
                    "schema", "kind", "plan_digest", "unit_id", "arm",
                    "process_generation_id", "lineage_id", "fence_id", "grant_id",
                    "container_id", "worker_identity", "observed_started_at",
                    "observed_ended_at", "prompt_manifest_digest",
                    "comparison_identities", "observations", "selected_observation",
                    "value", "error", "artifact_digest",
                },
                "completed_attempt": {
                    "schema", "kind", "plan_digest", "unit_id", "arm",
                    "process_generation_id", "lineage_id", "fence_id", "grant_id",
                    "container_id", "worker_identity", "observed_started_at",
                    "observed_ended_at", "prompt_manifest_digest", "prompt_ids",
                    "comparison_identities", "native_observation_digest",
                    "stage_witnesses", "terminal", "value",
                    "provider_recorded_screen", "recorded_screen", "reason",
                    "artifact_digest",
                },
            }
            if expected_artifact_schema == ps.ARTIFACT_SCHEMA_V2:
                if kind == "continued_unit":
                    raise NativeCaptureRefused(
                        "v2 continuation requires an original v2 observation reference")
                shapes["native_observation"] |= {"lifecycle_observation"}
                shapes["completed_attempt"] |= {
                    "lifecycle_observation_content_sha256"}
            if selected_artifact:
                shapes["completed_attempt"] |= {"selected_range"}
            if set(document) != shapes[kind]:
                raise NativeCaptureRefused(f"raw {kind} artifact shape is not closed")
            if selected_artifact:
                from .planned_unit_selection import SelectedPlanUnitRange, SelectionRefused
                try:
                    scope = SelectedPlanUnitRange.from_dict(document["selected_range"])
                    units = scope.units(ep.ExperimentPlan.from_dict(carrier["plan"]))
                    if len(units) != 1 or (document.get("unit_id"), document.get("arm"),
                            document.get("process_generation_id"), tuple(document.get("prompt_ids", ()))) != (
                            units[0].unit_id, units[0].arm, units[0].process_id,
                            units[0].expected_prompt_ids):
                        raise SelectionRefused("selected artifact differs from exact original unit")
                except (SelectionRefused, ep.PlanValidationError) as exc:
                    raise NativeCaptureRefused("selected artifact range/membership differs") from exc
            if document.get("arm") != carrier["arm"]:
                raise NativeCaptureRefused("raw artifact belongs to another carrier arm")
            if (document.get("plan_digest") != schemas.content_hash(carrier["plan"])
                    or document.get("prompt_manifest_digest") !=
                    carrier["prompt_manifest_digest"]
                    or document.get("comparison_identities") !=
                    carrier["comparison_identities"]):
                raise NativeCaptureRefused(
                    "raw artifact differs from carrier plan/prompt/identity binding")
            for field in ("lineage_id", "grant_id", "container_id"):
                if field in document and document[field] != carrier["capture_context"][field]:
                    raise NativeCaptureRefused(
                        f"raw artifact {field} differs from capture context")
            if document["kind"] != "continued_unit":
                worker = document.get("worker_identity")
                context = carrier["capture_context"]
                expected_worker = {
                    "supervisor_id": context["supervisor_id"],
                    "supervisor_incarnation": context["supervisor_incarnation"],
                    "config_generation": context["config_generation"],
                    "worker_id": context["worker_id"],
                    "worker_incarnation": context["worker_incarnation"],
                }
                if worker != expected_worker:
                    raise NativeCaptureRefused(
                        "raw artifact worker identity differs from capture context")
            try:
                verified = self.store.verify(f"raw:{artifact_digest}", document)
            except mc.CaptureError as exc:
                raise NativeCaptureRefused("raw artifact byte verification failed") from exc
            if item["stored"] != verified.to_dict():
                raise NativeCaptureRefused("raw artifact locator/hash differs from verified bytes")
            if document.get("kind") == "native_observation":
                intervals.append(self._native_interval(document))
        selected = carrier["admissible_view"]["selected_rows"]
        for row in selected:
            if row["arm"] != carrier["arm"]:
                continue
            document = documents.get(row["artifact_digest"])
            if (not isinstance(document, Mapping)
                    or document.get("kind") not in {"completed_attempt", "continued_unit"}
                    or document.get("unit_id") != row["unit_id"]
                    or document.get("process_generation_id") != row["process_id"]
                    or document.get("value") != row["value"]):
                raise NativeCaptureRefused(
                    "admissible row lacks its exact retained completed attempt")
        attempts = [document for document in documents.values()
                    if document.get("kind") == "completed_attempt"]
        selected_attempts = [document for document in attempts
                             if document["schema"] == ps.ARTIFACT_SCHEMA_V3]
        if selected_attempts and (len(attempts) != 1 or any(
                document.get("unit_id") != selected_attempts[0]["unit_id"]
                for document in documents.values())):
            raise NativeCaptureRefused("selected capture mixes original unit ranges")
        plan = ep.ExperimentPlan.from_dict(carrier["plan"])
        self._rederive_view(plan, carrier, documents, attempts)
        self._validate_measurement_links(plan, carrier, documents, attempts)
        expected_environment = self._expected_environment(
            attempts, carrier["comparison_identities"][carrier["arm"]].get("backend"))
        if carrier["environment_verdicts"] != expected_environment:
            raise NativeCaptureRefused(
                "environment verdicts differ from retained completed attempts")
        self._validate_status(plan, carrier, attempts, raw_artifacts)
        expected_interval = None
        if intervals:
            first = min(intervals, key=lambda item: item[0])
            last = max(intervals, key=lambda item: item[1])
            expected_interval = {"start": first[2], "end": last[3]}
        if carrier["interval"] != expected_interval:
            raise NativeCaptureRefused("carrier interval differs from retained raw observations")
        try:
            sealed = self.store.verify(f"carrier:{measurement_id}", carrier)
        except mc.CaptureError as exc:
            raise NativeCaptureRefused("carrier artifact byte verification failed") from exc
        if carrier_artifact != sealed.to_dict():
            raise NativeCaptureRefused("carrier locator/hash differs from verified bytes")

    def _verify_observations(self, carrier: Mapping[str, Any]) \
            -> tuple[int, tuple[ob.ValidatedObservationLink, ...]]:
        """Reopen v2 observation/instrument bytes; never trust child verdict labels."""
        if carrier["schema"] != mc.CAPTURE_SCHEMA_V2:
            raise NativeCaptureRefused("observation verification is v2-only")
        try:
            instrument = ob.LoadedInstrumentReference.from_dict(
                carrier["loaded_instrument"])
        except Exception as exc:
            raise NativeCaptureRefused("v2 loaded instrument reference is invalid") from exc
        refs = carrier["lifecycle_observations"]
        if not isinstance(refs, list):
            raise NativeCaptureRefused("v2 lifecycle observations must be an array")
        native = [item["document"] for item in carrier["raw_artifacts"]
                  if item["document"].get("kind") == "native_observation"]
        attempts = [item["document"] for item in carrier["raw_artifacts"]
                    if item["document"].get("kind") == "completed_attempt"]
        by_unit = {item.get("unit_id"): item for item in native}
        if (len(by_unit) != len(native) or len(refs) != len(native)
                or {item.get("unit_id") for item in refs} != set(by_unit)):
            raise NativeCaptureRefused(
                "v2 requires exactly one lifecycle observation per executed arm unit")
        links: list[ob.ValidatedObservationLink] = []
        for raw_ref in refs:
            try:
                reference = ob.LifecycleObservationReference.from_dict(raw_ref)
                document = by_unit[reference.unit_id]
                if (document.get("process_generation_id") != reference.process_generation_id
                        or document.get("lifecycle_observation") != reference.to_dict()):
                    raise NativeCaptureRefused(
                        "lifecycle observation differs from raw unit binding")
                matching = [row for row in attempts if row.get("unit_id") == reference.unit_id]
                if (len(matching) != 1
                        or matching[0].get("lifecycle_observation_content_sha256")
                           != reference.observation_content_sha256):
                    raise NativeCaptureRefused(
                        "completed attempt does not bind lifecycle observation content")
                links.append(ob.validate_reopened_observation(reference, store=self.store,
                    expected={"plan_digest": schemas.content_hash(carrier["plan"]),
                              "unit_id": reference.unit_id,
                              "process_generation_id": reference.process_generation_id,
                              "fence_id": document["fence_id"],
                              "active_claim_ref": reference.active_claim_ref,
                              "container_id": carrier["capture_context"]["container_id"],
                              "capture_context": carrier["capture_context"]},
                    instrument=instrument, verifiers=self.observation_verifiers))
            except NativeCaptureRefused:
                raise
            except Exception as exc:
                raise NativeCaptureRefused(
                    "parent lifecycle observation validation failed") from exc
        generations = {row["grant_generation"] for row in refs}
        if len(generations) != 1:
            raise NativeCaptureRefused("v2 observations disagree on grant generation")
        return next(iter(generations)), tuple(links)

    @staticmethod
    def _rederive_view(plan: ep.ExperimentPlan, carrier: Mapping[str, Any],
                       documents: Mapping[str, Mapping[str, Any]],
                       attempts: list[Mapping[str, Any]]) -> None:
        specs = {spec.unit_id: spec for spec in plan.expected_units}
        rows = []
        for attempt in attempts:
            if attempt.get("terminal") is not True:
                continue
            spec = specs.get(attempt.get("unit_id"))
            if spec is None:
                raise NativeCaptureRefused("completed attempt is absent from frozen plan")
            rows.append(ep.RawUnit.from_dict({
                "schema": ep.UNIT_SCHEMA,
                "plan_digest": attempt.get("plan_digest"),
                "unit_id": attempt.get("unit_id"), "arm": attempt.get("arm"),
                "process_id": attempt.get("process_generation_id"),
                "prompt_ids": attempt.get("prompt_ids"),
                "terminal": attempt.get("terminal"), "value": attempt.get("value"),
                "witnesses": attempt.get("stage_witnesses"),
                "recorded_screen": attempt.get("recorded_screen"),
                "reason": attempt.get("reason"),
                "artifact_digest": attempt.get("artifact_digest"),
                "observed_order_index": spec.order_index,
            }))
        continued = [document for document in documents.values()
                     if document.get("kind") == "continued_unit"]
        selected_by_digest = {
            row["artifact_digest"]: row
            for row in carrier["admissible_view"]["selected_rows"]}
        for document in continued:
            row = selected_by_digest.get(document["artifact_digest"])
            if row is None:
                raise NativeCaptureRefused(
                    "continued artifact lacks its selected historical raw unit")
            if (document.get("unit_id") != row["unit_id"]
                    or document.get("arm") != row["arm"]
                    or document.get("process_generation_id") != row["process_id"]
                    or document.get("prompt_ids") != row["prompt_ids"]
                    or document.get("terminal") != row["terminal"]
                    or document.get("value") != row["value"]
                    or document.get("recorded_screen") != row["recorded_screen"]
                    or document.get("reason") != row["reason"]):
                raise NativeCaptureRefused(
                    "continued artifact differs from selected historical raw unit")
            rows.append(ep.RawUnit.from_dict(row))
        selected = [row for row in carrier["admissible_view"]["selected_rows"]
                    if row["arm"] == carrier["arm"]]
        reconstructed = {row.unit_id: row.to_dict() for row in rows}
        if any(row["unit_id"] not in reconstructed
               or not NativeCaptureValidator._same_json(
                   row, reconstructed[row["unit_id"]]) for row in selected):
            raise NativeCaptureRefused(
                "selected arm view differs from retained attempt artifacts")
        selected_ids = {row["unit_id"] for row in selected}
        reasons = carrier["admissible_view"]["rejection_reasons"]
        for unit_id in reconstructed.keys() - selected_ids:
            if unit_id not in reasons:
                raise NativeCaptureRefused(
                    "retained terminal attempt is absent from view without rejection")

    @staticmethod
    def _validate_measurement_links(
            plan: ep.ExperimentPlan, carrier: Mapping[str, Any],
            documents: Mapping[str, Mapping[str, Any]],
            attempts: list[Mapping[str, Any]]) -> None:
        """Re-derive every measured launch from its retained request rows."""
        if carrier["status"] != "measurement":
            return
        arm = carrier["arm"]
        rows = [row for row in carrier["admissible_view"]["selected_rows"]
                if row["arm"] == arm]
        specs = {spec.unit_id: spec for spec in plan.expected_units}
        prompt_manifest = ps.FrozenPromptManifest.from_dict(carrier["prompt_manifest"])
        prompts = {prompt.prompt_id: prompt for prompt in prompt_manifest.prompts}
        for row in rows:
            spec = specs.get(row["unit_id"])
            matches = [attempt for attempt in attempts
                       if attempt.get("unit_id") == row["unit_id"]]
            if spec is None or len(matches) != 1:
                raise NativeCaptureRefused(
                    "measured row lacks exactly one declared completed attempt")
            attempt = matches[0]
            native_observation = documents.get(attempt.get("native_observation_digest"))
            if (not isinstance(native_observation, Mapping)
                    or native_observation.get("kind") != "native_observation"):
                raise NativeCaptureRefused(
                    "completed attempt lacks its exact native observation")
            common = ("unit_id", "arm", "process_generation_id", "lineage_id",
                      "fence_id", "grant_id", "container_id", "worker_identity",
                      "observed_started_at", "observed_ended_at",
                      "prompt_manifest_digest", "comparison_identities")
            if any(attempt.get(name) != native_observation.get(name) for name in common):
                raise NativeCaptureRefused(
                    "completed attempt and native observation bindings differ")
            _text(attempt.get("fence_id"), "completed attempt fence_id")
            selected = native_observation.get("selected_observation")
            observations = native_observation.get("observations")
            if (not isinstance(selected, Mapping) or not isinstance(observations, list)
                    or not observations or observations[-1] != selected
                    or selected.get("schema") != "epyc.autokernel.serving_observation.v1"
                    or selected.get("teardown") not in {"terminated", "killed"}
                    or selected.get("failure") is not None
                    or native_observation.get("error") is not None):
                raise NativeCaptureRefused("native selected observation is not terminal/exact")
            if set(selected) != {
                    "schema", "process_pid", "requests", "residency", "teardown",
                    "failure"}:
                raise NativeCaptureRefused("native selected observation shape is not closed")
            requests = selected.get("requests")
            if not isinstance(requests, list):
                raise NativeCaptureRefused("native selected requests must be a list")
            measured = [request for request in requests
                        if isinstance(request, Mapping)
                        and request.get("phase") == "measurement"]
            if len(measured) != len(spec.expected_prompt_ids):
                raise NativeCaptureRefused("native measurement request membership is incomplete")
            rates = []
            for index, (request, prompt_id) in enumerate(
                    zip(measured, spec.expected_prompt_ids, strict=True)):
                prompt = prompts.get(prompt_id)
                rate = request.get("predicted_per_second")
                if (set(request) != {"phase", "slot_index", "prompt_id",
                                     "request_sha256", "predicted_n",
                                     "predicted_per_second", "terminal", "error"}
                        or prompt is None or request.get("slot_index") != index
                        or request.get("prompt_id") != prompt_id
                        or request.get("request_sha256") != prompt.request_digest
                        or request.get("predicted_n") != prompt.n_predict
                        or request.get("terminal") is not True
                        or request.get("error") is not None
                        or isinstance(rate, bool)
                        or not isinstance(rate, (int, float))
                        or not math.isfinite(float(rate))):
                    raise NativeCaptureRefused(
                        "native request row differs from frozen prompt/terminal result")
                rates.append(float(rate))
            if (row["prompt_ids"] != list(spec.expected_prompt_ids)
                    or attempt.get("prompt_ids") != row["prompt_ids"]
                    or attempt.get("process_generation_id") != spec.process_id
                    or native_observation.get("process_generation_id") != spec.process_id
                    or attempt.get("terminal") is not True
                    or attempt.get("value") != row["value"]
                    or native_observation.get("value") != row["value"]
                    or sum(rates) != float(row["value"])
                    or attempt.get("stage_witnesses") != row["witnesses"]
                    or attempt.get("provider_recorded_screen") != row["recorded_screen"]
                    or attempt.get("recorded_screen") != row["recorded_screen"]
                    or attempt.get("reason") != row["reason"]
                    or attempt.get("artifact_digest") != row["artifact_digest"]):
                raise NativeCaptureRefused(
                    "measured raw unit/process/value/witness links are inconsistent")

    @staticmethod
    def _expected_environment(attempts: list[Mapping[str, Any]],
                              backend: Any) -> list[dict[str, Any]]:
        labels = {
            "contention": {"pass": "clean", "fail": "contaminated",
                           "unknown": "unknown"},
            "placement": {"pass": "proven", "fail": "refuted",
                          "unknown": "unknown"},
            "residency": {"pass": "proven", "fail": "refuted",
                          "unknown": "unknown"},
        }
        out = []
        for attempt in attempts:
            witnesses = attempt.get("stage_witnesses")
            witnesses = witnesses if isinstance(witnesses, Mapping) else {}
            row: dict[str, Any] = {"unit_id": attempt.get("unit_id")}
            for name, states in labels.items():
                witness = witnesses.get(name)
                status = witness.get("status") if isinstance(witness, Mapping) else "unknown"
                ref = witness.get("ref") if isinstance(witness, Mapping) else None
                if name == "residency" and backend == "cpu":
                    row[name] = {"verdict": "not_applicable", "ref": None}
                else:
                    row[name] = {"verdict": states.get(status, "unknown"), "ref": ref}
            out.append(row)
        return out

    @staticmethod
    def _native_interval(document: Mapping[str, Any]) \
            -> tuple[datetime, datetime, str, str]:
        values = []
        for key in ("observed_started_at", "observed_ended_at"):
            text = _text(document.get(key), key)
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise NativeCaptureRefused(f"{key} must be ISO-8601") from exc
            if parsed.tzinfo is None:
                raise NativeCaptureRefused(f"{key} must include timezone")
            values.append((parsed, text))
        if values[1][0] < values[0][0]:
            raise NativeCaptureRefused("native observation interval is reversed")
        return values[0][0], values[1][0], values[0][1], values[1][1]


__all__ = ["CurrentOwnerToken", "FenceProvider", "NativeCaptureBinding",
           "NativeCaptureRefused", "NativeCaptureValidator",
           "PrevalidatedNativeCapture", "TrustedWorkerResultFence",
           "ValidatedNativeCapture"]
