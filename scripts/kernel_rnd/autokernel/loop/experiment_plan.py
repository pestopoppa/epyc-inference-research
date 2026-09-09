#!/usr/bin/env python3
"""Immutable experiment plans and evidence-use checks.

This module validates plans and recorded units.  It performs no execution and
does not grade claims.  In particular, a protocol label is data, not authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, cast

from .. import schemas


PLAN_SCHEMA = "epyc.autokernel.experiment_plan.v1"
UNIT_SCHEMA = "epyc.autokernel.raw_unit.v1"
VIEW_SCHEMA = "epyc.autokernel.admissible_unit_view.v1"
CALIBRATION_SCHEMA = "epyc.autokernel.calibration_receipt.v1"
DISPOSITION_SCHEMA = "epyc.autokernel.evidence_use_disposition.v1"

INSTRUMENT_CLASSES = frozenset({"bench", "serving"})
CATEGORIES = frozenset({"OPTIMUM", "BASELINE", "CANDIDATE"})
PHASES = frozenset({"discovery", "confirmation", "observation", "release"})
PROTOCOL_STATUSES = frozenset({"ratified", "unratified", "unknown"})
RECORD_CLASSES = frozenset({
    "discovery_screen", "strict_search", "observation", "registered_claim",
})
COMPARISON_KINDS = frozenset({
    "mechanism", "assembled_candidate", "best_supported_recipe",
})
ESTIMANDS = frozenset({"level", "dispersion"})
METRIC_DIRECTIONS = frozenset({"higher", "lower"})
UNITS = frozenset({"arm", "session", "process"})
ARMS = frozenset({"anchor", "candidate"})
SCREEN_STATES = frozenset({"clean", "flagged_but_retained", "rejected"})
WITNESS_STATES = frozenset({"pass", "fail", "unknown"})
CLAIM_USES = frozenset({
    "bank", "validate", "validate_production", "certify", "certify_transfer",
    "certify_overlap", "headline", "release",
})
KNOWN_USES = frozenset({"explore", "nominate", "rank"}) | CLAIM_USES

# Exact protocol identity can be checked structurally; its semantic attestations
# require a trusted adapter that is deliberately absent from v1.
A2_PROTOCOL_REF = "P-AK-SEARCH-1-A2"
RECORD_PHASE = {
    "discovery_screen": "discovery",
    "strict_search": "confirmation",
    "observation": "observation",
    "registered_claim": "release",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

__all__ = [
    "PLAN_SCHEMA", "UNIT_SCHEMA", "VIEW_SCHEMA", "CALIBRATION_SCHEMA",
    "DISPOSITION_SCHEMA", "A2_PROTOCOL_REF",
    "PlanValidationError", "UnsupportedStoppingRule", "UnitSpec",
    "ExperimentPlan", "Witness", "RawUnit", "AdmissibleUnitView",
    "UseDisposition", "admissible_units", "eligibility",
    "CalibrationReceipt", "CalibrationDisposition", "CalibrationCache",
    "calibration_applicability",
]


class PlanValidationError(ValueError):
    """A versioned plan/evidence object violates its declared schema."""


class UnsupportedStoppingRule(PlanValidationError):
    """The stopping rule is well identified but has no implementation here."""


def _exact(obj: Mapping[str, Any], fields: set[str], label: str) -> None:
    if not isinstance(obj, Mapping):
        raise PlanValidationError(f"{label}: expected object")
    missing = fields - set(obj)
    extra = set(obj) - fields
    if missing or extra:
        bits = []
        if missing:
            bits.append("missing " + ", ".join(sorted(missing)))
        if extra:
            bits.append("unknown " + ", ".join(sorted(extra)))
        raise PlanValidationError(f"{label}: {'; '.join(bits)}")


def _text(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{label}: expected non-empty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlanValidationError(f"{label}: expected integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanValidationError(f"{label}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PlanValidationError(f"{label}: expected finite number")
    return result


def _digest(value: Any, label: str) -> str:
    value = _text(value, label)
    value = cast(str, value)
    if not _SHA256.fullmatch(value):
        raise PlanValidationError(f"{label}: expected lowercase SHA-256")
    return value


def _enum(value: Any, choices: frozenset[str], label: str) -> str:
    value = _text(value, label)
    value = cast(str, value)
    if value not in choices:
        raise PlanValidationError(f"{label}: unsupported {value!r}")
    return value


def _freeze(value: Any, label: str) -> Any:
    """Validate canonical JSON data and recursively make it immutable."""
    try:
        schemas.canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{label}: {exc}") from exc
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item, f"{label}.{key}")
                                 for key, item in value.items()})
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze(item, f"{label}[]") for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strings(value: Any, label: str, *, allow_empty: bool = False,
             unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanValidationError(f"{label}: expected array")
    items = tuple(_text(item, f"{label}[]") for item in value)
    if not allow_empty and not items:
        raise PlanValidationError(f"{label}: must not be empty")
    if unique and len(set(items)) != len(items):
        raise PlanValidationError(f"{label}: duplicate identifiers")
    return items  # type: ignore[return-value]


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    arm: str
    process_id: str
    expected_prompt_ids: tuple[str, ...]
    order_index: int
    pair_id: int | None

    FIELDS = {"unit_id", "arm", "process_id", "expected_prompt_ids",
              "order_index", "pair_id"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "UnitSpec":
        _exact(obj, cls.FIELDS, "UnitSpec")
        pair_id = obj["pair_id"]
        if pair_id is not None:
            pair_id = _integer(pair_id, "UnitSpec.pair_id")
        return cls(
            unit_id=_text(obj["unit_id"], "UnitSpec.unit_id"),  # type: ignore[arg-type]
            arm=_enum(obj["arm"], ARMS, "UnitSpec.arm"),
            process_id=_text(obj["process_id"], "UnitSpec.process_id"),  # type: ignore[arg-type]
            expected_prompt_ids=_strings(obj["expected_prompt_ids"],
                                         "UnitSpec.expected_prompt_ids"),
            order_index=_integer(obj["order_index"], "UnitSpec.order_index"),
            pair_id=pair_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"unit_id": self.unit_id, "arm": self.arm,
                "process_id": self.process_id,
                "expected_prompt_ids": list(self.expected_prompt_ids),
                "order_index": self.order_index, "pair_id": self.pair_id}


@dataclass(frozen=True)
class ExperimentPlan:
    schema: str
    plan_id: str
    campaign_id: str
    target_revision: str
    epoch: str
    instrument_class: str
    category: str
    phase: str
    protocol_ref: str | None
    protocol_status: str
    record_class: str
    intended_use: str
    comparison_kind: str
    estimand: str
    metric: str
    metric_direction: str
    estimator_id: str
    unit: str
    changed_factors: tuple[str, ...]
    anchor_identity: Mapping[str, Any]
    candidate_identity: Mapping[str, Any]
    expected_units: tuple[UnitSpec, ...]
    stopping: Mapping[str, Any]
    required_witnesses: tuple[str, ...]
    calibration_ref: str | None
    policy_snapshot: Mapping[str, Any]
    continuation_allowed: bool

    FIELDS = {"schema", "plan_id", "campaign_id", "target_revision", "epoch",
              "instrument_class", "category", "phase", "protocol_ref",
              "protocol_status", "record_class", "intended_use",
              "comparison_kind", "estimand", "metric", "metric_direction",
              "estimator_id", "unit", "changed_factors", "anchor_identity",
              "candidate_identity", "expected_units", "stopping",
              "required_witnesses", "calibration_ref", "policy_snapshot",
              "continuation_allowed"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ExperimentPlan":
        _exact(obj, cls.FIELDS, "ExperimentPlan")
        if obj["schema"] != PLAN_SCHEMA:
            raise PlanValidationError(
                f"ExperimentPlan.schema: unsupported {obj['schema']!r}")
        units_obj = obj["expected_units"]
        if not isinstance(units_obj, (list, tuple)):
            raise PlanValidationError("ExperimentPlan.expected_units: expected array")
        units = tuple(UnitSpec.from_dict(item) for item in units_obj)
        if not units:
            raise PlanValidationError("ExperimentPlan.expected_units: must not be empty")
        unit_ids = [item.unit_id for item in units]
        if len(set(unit_ids)) != len(unit_ids):
            raise PlanValidationError("ExperimentPlan.expected_units: duplicate unit_id")
        indices = [item.order_index for item in units]
        if sorted(indices) != list(range(len(units))) or len(set(indices)) != len(indices):
            raise PlanValidationError(
                "ExperimentPlan.expected_units: order_index must be unique and contiguous")

        unit_kind = _enum(obj["unit"], UNITS, "ExperimentPlan.unit")
        if unit_kind == "process":
            process_ids = [item.process_id for item in units]
            if len(set(process_ids)) != len(process_ids):
                raise PlanValidationError(
                    "ExperimentPlan.expected_units: process unit reuses process_id")

        stopping_obj = obj["stopping"]
        _exact(stopping_obj, {"kind", "n_per_arm", "paired"},
               "ExperimentPlan.stopping")
        kind = _text(stopping_obj["kind"], "ExperimentPlan.stopping.kind")
        if kind != "fixed_n":
            raise UnsupportedStoppingRule(
                f"ExperimentPlan.stopping.kind: unsupported {kind!r}")
        n_per_arm = _integer(stopping_obj["n_per_arm"],
                             "ExperimentPlan.stopping.n_per_arm", minimum=1)
        paired = stopping_obj["paired"]
        if not isinstance(paired, bool):
            raise PlanValidationError("ExperimentPlan.stopping.paired: expected boolean")
        counts = {arm: sum(item.arm == arm for item in units) for arm in ARMS}
        if counts != {"anchor": n_per_arm, "candidate": n_per_arm}:
            raise PlanValidationError(
                "ExperimentPlan.expected_units: fixed_n requires n_per_arm units for each arm")

        phase = _enum(obj["phase"], PHASES, "ExperimentPlan.phase")
        record_class = _enum(obj["record_class"], RECORD_CLASSES,
                             "ExperimentPlan.record_class")
        if RECORD_PHASE[record_class] != phase:
            raise PlanValidationError(
                "ExperimentPlan: record_class/phase mismatch; "
                f"{record_class!r} requires {RECORD_PHASE[record_class]!r}")
        if phase == "discovery":
            if paired or any(item.pair_id is not None for item in units):
                raise PlanValidationError(
                    "ExperimentPlan.discovery: pair_id must be null and stopping unpaired")
        elif not paired and any(item.pair_id is not None for item in units):
            raise PlanValidationError(
                "ExperimentPlan.unpaired: every pair_id must be null")
        if phase == "confirmation" and not paired:
            raise PlanValidationError("ExperimentPlan.confirmation: paired fixed_n required")
        if paired:
            by_pair: dict[int, list[UnitSpec]] = {}
            for item in units:
                if item.pair_id is None:
                    raise PlanValidationError(
                        "ExperimentPlan.paired: every unit requires pair_id")
                by_pair.setdefault(item.pair_id, []).append(item)
            if sorted(by_pair) != list(range(n_per_arm)):
                raise PlanValidationError(
                    "ExperimentPlan.paired: pair slots must be contiguous")
            if any(len(pair) != 2 or {item.arm for item in pair} != ARMS
                   for pair in by_pair.values()):
                raise PlanValidationError(
                    "ExperimentPlan.paired: each pair requires two distinct arms")
            if phase == "confirmation":
                ordered = sorted(units, key=lambda item: item.order_index)
                for slot in range(n_per_arm):
                    block = ordered[2 * slot:2 * slot + 2]
                    if ({item.pair_id for item in block} != {slot}
                            or {item.arm for item in block} != ARMS):
                        raise PlanValidationError(
                            "ExperimentPlan.confirmation: each pair must occupy "
                            "adjacent declared order slots with opposite arms")

        policy = obj["policy_snapshot"]
        _exact(policy, {"reference", "digest"}, "ExperimentPlan.policy_snapshot")
        policy = {"reference": _text(policy["reference"],
                                     "ExperimentPlan.policy_snapshot.reference"),
                  "digest": _digest(policy["digest"],
                                    "ExperimentPlan.policy_snapshot.digest")}
        if not isinstance(obj["continuation_allowed"], bool):
            raise PlanValidationError(
                "ExperimentPlan.continuation_allowed: expected boolean")
        intended_use = _enum(obj["intended_use"], KNOWN_USES,
                             "ExperimentPlan.intended_use")
        anchor = obj["anchor_identity"]
        candidate = obj["candidate_identity"]
        if not isinstance(anchor, Mapping) or not anchor:
            raise PlanValidationError("ExperimentPlan.anchor_identity: non-empty object required")
        if not isinstance(candidate, Mapping) or not candidate:
            raise PlanValidationError("ExperimentPlan.candidate_identity: non-empty object required")
        return cls(
            schema=PLAN_SCHEMA,
            plan_id=_text(obj["plan_id"], "ExperimentPlan.plan_id"),  # type: ignore[arg-type]
            campaign_id=_text(obj["campaign_id"], "ExperimentPlan.campaign_id"),  # type: ignore[arg-type]
            target_revision=_text(obj["target_revision"], "ExperimentPlan.target_revision"),  # type: ignore[arg-type]
            epoch=_text(obj["epoch"], "ExperimentPlan.epoch"),  # type: ignore[arg-type]
            instrument_class=_enum(obj["instrument_class"], INSTRUMENT_CLASSES,
                                   "ExperimentPlan.instrument_class"),
            category=_enum(obj["category"], CATEGORIES, "ExperimentPlan.category"),
            phase=phase,
            protocol_ref=_text(obj["protocol_ref"], "ExperimentPlan.protocol_ref", nullable=True),
            protocol_status=_enum(obj["protocol_status"], PROTOCOL_STATUSES,
                                  "ExperimentPlan.protocol_status"),
            record_class=record_class,
            intended_use=intended_use,
            comparison_kind=_enum(obj["comparison_kind"], COMPARISON_KINDS,
                                  "ExperimentPlan.comparison_kind"),
            estimand=_enum(obj["estimand"], ESTIMANDS, "ExperimentPlan.estimand"),
            metric=_text(obj["metric"], "ExperimentPlan.metric"),  # type: ignore[arg-type]
            metric_direction=_enum(obj["metric_direction"], METRIC_DIRECTIONS,
                                   "ExperimentPlan.metric_direction"),
            estimator_id=_text(obj["estimator_id"], "ExperimentPlan.estimator_id"),  # type: ignore[arg-type]
            unit=unit_kind,
            changed_factors=_strings(obj["changed_factors"],
                                     "ExperimentPlan.changed_factors", allow_empty=True),
            anchor_identity=_freeze(anchor, "ExperimentPlan.anchor_identity"),
            candidate_identity=_freeze(candidate, "ExperimentPlan.candidate_identity"),
            expected_units=units,
            stopping=_freeze({"kind": kind, "n_per_arm": n_per_arm,
                              "paired": paired}, "ExperimentPlan.stopping"),
            required_witnesses=_strings(obj["required_witnesses"],
                                        "ExperimentPlan.required_witnesses", allow_empty=True),
            calibration_ref=_text(obj["calibration_ref"],
                                  "ExperimentPlan.calibration_ref", nullable=True),
            policy_snapshot=_freeze(policy, "ExperimentPlan.policy_snapshot"),
            continuation_allowed=obj["continuation_allowed"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: (_thaw(getattr(self, name))) for name in self.FIELDS
                if name != "expected_units"} | {
                    "expected_units": [item.to_dict() for item in self.expected_units]}

    @property
    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class Witness:
    status: str
    ref: str | None

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any], label: str) -> "Witness":
        _exact(obj, {"status", "ref"}, label)
        return cls(_enum(obj["status"], WITNESS_STATES, f"{label}.status"),
                   _text(obj["ref"], f"{label}.ref", nullable=True))

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "ref": self.ref}


@dataclass(frozen=True)
class RawUnit:
    schema: str
    plan_digest: str
    unit_id: str
    arm: str
    process_id: str
    prompt_ids: tuple[str, ...]
    terminal: bool
    value: float
    witnesses: Mapping[str, Witness]
    recorded_screen: str
    reason: str | None
    artifact_digest: str
    observed_order_index: int

    FIELDS = {"schema", "plan_digest", "unit_id", "arm", "process_id",
              "prompt_ids", "terminal", "value", "witnesses",
              "recorded_screen", "reason", "artifact_digest",
              "observed_order_index"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "RawUnit":
        _exact(obj, cls.FIELDS, "RawUnit")
        if obj["schema"] != UNIT_SCHEMA:
            raise PlanValidationError(f"RawUnit.schema: unsupported {obj['schema']!r}")
        witness_obj = obj["witnesses"]
        if not isinstance(witness_obj, Mapping) or any(
                not isinstance(key, str) or not key for key in witness_obj):
            raise PlanValidationError("RawUnit.witnesses: expected text-keyed object")
        witnesses = MappingProxyType({
            key: Witness.from_dict(value, f"RawUnit.witnesses.{key}")
            for key, value in witness_obj.items()})
        if not isinstance(obj["terminal"], bool):
            raise PlanValidationError("RawUnit.terminal: expected boolean")
        screen = _enum(obj["recorded_screen"], SCREEN_STATES,
                       "RawUnit.recorded_screen")
        reason = _text(obj["reason"], "RawUnit.reason", nullable=True)
        if screen == "clean" and reason is not None:
            raise PlanValidationError("RawUnit.reason: clean unit must have null reason")
        if screen != "clean" and reason is None:
            raise PlanValidationError("RawUnit.reason: non-clean unit requires reason")
        return cls(
            schema=UNIT_SCHEMA,
            plan_digest=_digest(obj["plan_digest"], "RawUnit.plan_digest"),
            unit_id=_text(obj["unit_id"], "RawUnit.unit_id"),  # type: ignore[arg-type]
            arm=_enum(obj["arm"], ARMS, "RawUnit.arm"),
            process_id=_text(obj["process_id"], "RawUnit.process_id"),  # type: ignore[arg-type]
            prompt_ids=_strings(obj["prompt_ids"], "RawUnit.prompt_ids", unique=False),
            terminal=obj["terminal"], value=_finite(obj["value"], "RawUnit.value"),
            witnesses=witnesses, recorded_screen=screen, reason=reason,
            artifact_digest=_digest(obj["artifact_digest"], "RawUnit.artifact_digest"),
            observed_order_index=_integer(obj["observed_order_index"],
                                          "RawUnit.observed_order_index"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "plan_digest": self.plan_digest,
                "unit_id": self.unit_id, "arm": self.arm,
                "process_id": self.process_id, "prompt_ids": list(self.prompt_ids),
                "terminal": self.terminal, "value": self.value,
                "witnesses": {key: value.to_dict()
                              for key, value in self.witnesses.items()},
                "recorded_screen": self.recorded_screen, "reason": self.reason,
                "artifact_digest": self.artifact_digest,
                "observed_order_index": self.observed_order_index}


@dataclass(frozen=True)
class AdmissibleUnitView:
    schema: str
    plan_digest: str
    selected_rows: tuple[RawUnit, ...]
    rejection_reasons: Mapping[str, tuple[str, ...]]
    missing_expected_units: tuple[str, ...]
    independent_n: Mapping[str, int]
    complete: bool
    view_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "plan_digest": self.plan_digest,
                "selected_rows": [row.to_dict() for row in self.selected_rows],
                "rejection_reasons": {key: list(value)
                                      for key, value in self.rejection_reasons.items()},
                "missing_expected_units": list(self.missing_expected_units),
                "independent_n": dict(self.independent_n),
                "complete": self.complete, "view_digest": self.view_digest}


def _normalized_plan(plan: ExperimentPlan) -> ExperimentPlan:
    if not isinstance(plan, ExperimentPlan):
        raise PlanValidationError("expected ExperimentPlan")
    try:
        return ExperimentPlan.from_dict(plan.to_dict())
    except PlanValidationError:
        raise
    except Exception as exc:
        raise PlanValidationError(f"invalid directly constructed ExperimentPlan: {exc}") from exc


def _normalized_raw(row: RawUnit) -> RawUnit:
    if not isinstance(row, RawUnit):
        raise PlanValidationError("admissible_units: every row must be RawUnit")
    try:
        return RawUnit.from_dict(row.to_dict())
    except PlanValidationError:
        raise
    except Exception as exc:
        raise PlanValidationError(f"invalid directly constructed RawUnit: {exc}") from exc


def admissible_units(plan: ExperimentPlan,
                     raws: Iterable[RawUnit]) -> AdmissibleUnitView:
    """Return the one immutable unit view supplied to every later consumer."""
    plan = _normalized_plan(plan)
    rows = tuple(_normalized_raw(row) for row in raws)
    expected = {item.unit_id: item for item in plan.expected_units}
    by_id: dict[str, list[RawUnit]] = {}
    for row in rows:
        by_id.setdefault(row.unit_id, []).append(row)
    reasons: dict[str, list[str]] = {}

    def reject(unit_id: str, reason: str) -> None:
        bucket = reasons.setdefault(unit_id, [])
        if reason not in bucket:
            bucket.append(reason)

    for unit_id, instances in by_id.items():
        if len(instances) != 1:
            reject(unit_id, "duplicate raw unit_id")
        if unit_id not in expected:
            reject(unit_id, "unit_id not declared by plan")
        for row in instances:
            spec = expected.get(unit_id)
            if spec is None:
                continue
            if row.plan_digest != plan.digest:
                reject(unit_id, "wrong plan digest")
            if row.arm != spec.arm:
                reject(unit_id, "wrong arm")
            if row.process_id != spec.process_id:
                reject(unit_id, "wrong process_id")
            if row.observed_order_index != spec.order_index:
                reject(unit_id, "wrong observed order")
            if len(set(row.prompt_ids)) != len(row.prompt_ids):
                reject(unit_id, "duplicate prompt IDs")
            missing_prompts = set(spec.expected_prompt_ids) - set(row.prompt_ids)
            extra_prompts = set(row.prompt_ids) - set(spec.expected_prompt_ids)
            if missing_prompts:
                reject(unit_id, "missing expected prompts")
            if extra_prompts:
                reject(unit_id, "extra prompts")
            if not missing_prompts and not extra_prompts and row.prompt_ids != spec.expected_prompt_ids:
                reject(unit_id, "prompt order mismatch")
            if not row.terminal:
                reject(unit_id, "unit is not terminal")
            if not math.isfinite(row.value):
                reject(unit_id, "non-finite value")
            if row.recorded_screen == "rejected":
                reject(unit_id, f"predeclared screen rejected: {row.reason}")
            for name in plan.required_witnesses:
                witness = row.witnesses.get(name)
                if witness is None:
                    reject(unit_id, f"missing required witness {name}")
                elif witness.status != "pass" or witness.ref is None:
                    reject(unit_id, f"required witness {name} is not passed with a ref")

    missing = tuple(item.unit_id for item in plan.expected_units
                    if item.unit_id not in by_id)
    if bool(plan.stopping["paired"]):
        pairs: dict[int, tuple[str, ...]] = {}
        for spec in plan.expected_units:
            if spec.pair_id is None:
                raise PlanValidationError(
                    "paired plan contains a unit without pair_id")
            pairs.setdefault(spec.pair_id, tuple())
            pairs[spec.pair_id] += (spec.unit_id,)
        for members in pairs.values():
            bad = [member for member in members
                   if member in missing or member in reasons]
            if bad:
                for member in members:
                    reject(member, "paired counterpart missing or invalid")

    selected = tuple(row for row in rows
                     if len(by_id[row.unit_id]) == 1
                     and row.unit_id in expected and row.unit_id not in reasons)
    selected = tuple(sorted(selected, key=lambda row: expected[row.unit_id].order_index))
    counts = MappingProxyType({arm: sum(row.arm == arm for row in selected)
                               for arm in sorted(ARMS)})
    complete = (bool(selected) and not missing and not reasons
                and len(selected) == len(plan.expected_units))
    body = {"schema": VIEW_SCHEMA, "plan_digest": plan.digest,
            "selected_rows": [row.to_dict() for row in selected],
            "rejection_reasons": {key: value for key, value in sorted(reasons.items())},
            "missing_expected_units": list(missing), "independent_n": dict(counts),
            "complete": complete}
    return AdmissibleUnitView(
        VIEW_SCHEMA, plan.digest, selected,
        MappingProxyType({key: tuple(value) for key, value in sorted(reasons.items())}),
        missing, counts, complete, schemas.content_hash(body))


@dataclass(frozen=True)
class UseDisposition:
    schema: str
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "status": self.status,
                "reasons": list(self.reasons)}


def _disposition(status: str, reasons: Iterable[str]) -> UseDisposition:
    return UseDisposition(DISPOSITION_SCHEMA, status, tuple(dict.fromkeys(reasons)))


def _validated_view(plan: ExperimentPlan,
                    view: AdmissibleUnitView) -> AdmissibleUnitView:
    if not isinstance(view, AdmissibleUnitView):
        raise PlanValidationError("expected AdmissibleUnitView")
    if view.schema != VIEW_SCHEMA:
        raise PlanValidationError(f"unsupported unit-view schema {view.schema!r}")
    if view.plan_digest != plan.digest:
        raise PlanValidationError("unit view belongs to a different plan")
    rows = tuple(_normalized_raw(row) for row in view.selected_rows)
    if len({row.unit_id for row in rows}) != len(rows):
        raise PlanValidationError("unit view contains duplicate selected unit IDs")
    if not isinstance(view.complete, bool):
        raise PlanValidationError("unit view complete must be boolean")
    if not isinstance(view.rejection_reasons, Mapping):
        raise PlanValidationError("unit view rejection_reasons must be a mapping")
    reasons: dict[str, tuple[str, ...]] = {}
    for unit_id, values in view.rejection_reasons.items():
        if not isinstance(unit_id, str) or not unit_id:
            raise PlanValidationError("unit view rejection key must be text")
        reasons[unit_id] = _strings(values, f"unit view rejection_reasons.{unit_id}")
    missing = _strings(view.missing_expected_units,
                       "unit view missing_expected_units", allow_empty=True)
    if not isinstance(view.independent_n, Mapping) or set(view.independent_n) != ARMS:
        raise PlanValidationError("unit view independent_n requires both arms")
    counts = {arm: _integer(view.independent_n[arm],
                            f"unit view independent_n.{arm}") for arm in ARMS}
    actual_counts = {arm: sum(row.arm == arm for row in rows) for arm in ARMS}
    if counts != actual_counts:
        raise PlanValidationError("unit view independent_n does not match selected rows")
    selected_ids = {row.unit_id for row in rows}
    expected_ids = {unit.unit_id for unit in plan.expected_units}
    complete = (bool(rows) and selected_ids == expected_ids and not reasons and not missing)
    if view.complete != complete:
        raise PlanValidationError("unit view complete is inconsistent with its rows/reasons")
    if complete and not admissible_units(plan, rows).complete:
        raise PlanValidationError("unit view selected rows are not admissible for the plan")
    normalized = AdmissibleUnitView(
        VIEW_SCHEMA, plan.digest, rows,
        MappingProxyType(dict(sorted(reasons.items()))), missing,
        MappingProxyType(dict(sorted(counts.items()))), complete, view.view_digest)
    body = normalized.to_dict()
    supplied_digest = body.pop("view_digest")
    if not isinstance(supplied_digest, str) or schemas.content_hash(body) != supplied_digest:
        raise PlanValidationError("unit view digest does not verify")
    return normalized


def eligibility(plan: ExperimentPlan, view: AdmissibleUnitView,
                intended_use: str, *, current_epoch: str,
                registered_claim_grade: Any = None) -> UseDisposition:
    """Check structural evidence use.  This deliberately does not grade claims."""
    try:
        plan = _normalized_plan(plan)
        view = _validated_view(plan, view)
    except PlanValidationError as exc:
        return _disposition("refused", (f"invalid structural evidence: {exc}",))
    if intended_use not in KNOWN_USES:
        return _disposition("refused", (f"unsupported intended use {intended_use!r}",))
    refusals: list[str] = []
    undefined: list[str] = []
    if view.plan_digest != plan.digest:
        refusals.append("unit view belongs to a different plan")
    if intended_use != plan.intended_use:
        refusals.append("requested use differs from immutable plan")
    if (plan.category == "BASELINE" or plan.instrument_class == "bench") and intended_use in {
            "headline", "release"}:
        refusals.append("BASELINE or bench evidence cannot support headline/release use")
    if plan.category == "BASELINE" and intended_use in {
            "validate_production", "certify_transfer", "certify_overlap"}:
        refusals.append(
            "BASELINE evidence cannot validate production or certify transfer/overlap")
    if plan.record_class == "observation" and intended_use in CLAIM_USES:
        refusals.append("observation cannot be converted into a claim")
    if plan.record_class == "discovery_screen" and intended_use not in {"explore", "nominate"}:
        refusals.append("discovery evidence is limited to explore/nominate")
    if (plan.record_class in {"discovery_screen", "strict_search"}
            and intended_use == "rank" and current_epoch != plan.epoch):
        refusals.append("cross-epoch search magnitude cannot rank")
    if intended_use != "explore" and not view.complete:
        refusals.append("complete admissible-unit view required")

    if plan.record_class == "discovery_screen":
        if plan.phase != "discovery":
            refusals.append("discovery_screen requires discovery phase")
        if plan.protocol_ref != A2_PROTOCOL_REF or plan.protocol_status != "ratified":
            refusals.append("ratified exact Annex K A2 protocol required")
        if plan.stopping["n_per_arm"] != 3 or plan.stopping["paired"]:
            refusals.append("A2 requires separate fixed 3 anchor and 3 candidate units")
        if len(plan.changed_factors) != 1:
            refusals.append("A2 requires exactly one declared changed factor")
        if intended_use == "nominate":
            # No semantic bank/frame/sole-factor adapter is wired in v1.  Labels
            # and arbitrary witness references cannot bootstrap that authority.
            undefined.append(
                "nomination missing registered runtime-attestation verifier for "
                "sealed bank identity, sole-factor semantics, and zero new anchor launches")

    if plan.record_class == "strict_search" and intended_use in {"headline", "release"}:
        refusals.append("strict_search evidence cannot support headline/release use")
    if plan.record_class == "strict_search" and intended_use in CLAIM_USES:
        if not plan.stopping["paired"]:
            refusals.append("strict-search gate requires paired plan")
        if plan.calibration_ref is None:
            refusals.append("strict-search gate requires registered calibration reference")
        # A string/dict supplied by an actor is intentionally never a ClaimTuple.
        undefined.append("shared registered ClaimTuple grader adapter is not integrated")
    if plan.record_class == "registered_claim" and intended_use in CLAIM_USES:
        undefined.append("shared registered ClaimTuple grader adapter is not integrated")

    if refusals:
        return _disposition("refused", refusals + undefined)
    if undefined:
        return _disposition("policy_undefined", undefined)
    return _disposition("permitted", ())


@dataclass(frozen=True)
class CalibrationReceipt:
    schema: str
    unit: str
    harness: str
    n: int
    interval: Mapping[str, Any]
    estimator_id: str
    metric: str
    value: float
    anchor_identity: Mapping[str, Any]
    candidate_identity: Mapping[str, Any]
    raw_sample_digest: str
    unit_ids: tuple[str, ...]
    contention_model: Mapping[str, Any]
    host_state: Mapping[str, Any]
    policy_ref: Mapping[str, Any]

    FIELDS = {"schema", "unit", "harness", "n", "interval",
              "estimator_id", "metric", "value", "anchor_identity",
              "candidate_identity", "raw_sample_digest", "unit_ids",
              "contention_model", "host_state", "policy_ref"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "CalibrationReceipt":
        _exact(obj, cls.FIELDS, "CalibrationReceipt")
        if obj["schema"] != CALIBRATION_SCHEMA:
            raise PlanValidationError(
                f"CalibrationReceipt.schema: unsupported {obj['schema']!r}")
        unit = _enum(obj["unit"], UNITS, "CalibrationReceipt.unit")
        n = _integer(obj["n"], "CalibrationReceipt.n", minimum=24)
        unit_ids = _strings(obj["unit_ids"], "CalibrationReceipt.unit_ids")
        if len(unit_ids) != n:
            raise PlanValidationError("CalibrationReceipt.unit_ids: n mismatch")
        interval = obj["interval"]
        _exact(interval, {"lower", "upper", "confidence", "method_ref"},
               "CalibrationReceipt.interval")
        lower = _finite(interval["lower"], "CalibrationReceipt.interval.lower")
        upper = _finite(interval["upper"], "CalibrationReceipt.interval.upper")
        confidence = _finite(interval["confidence"],
                             "CalibrationReceipt.interval.confidence")
        if lower > upper:
            raise PlanValidationError("CalibrationReceipt.interval: lower exceeds upper")
        if not 0.0 < confidence < 1.0:
            raise PlanValidationError(
                "CalibrationReceipt.interval.confidence: expected 0 < value < 1")
        policy = obj["policy_ref"]
        _exact(policy, {"reference", "digest"}, "CalibrationReceipt.policy_ref")
        policy = {"reference": _text(policy["reference"],
                                     "CalibrationReceipt.policy_ref.reference"),
                  "digest": _digest(policy["digest"],
                                    "CalibrationReceipt.policy_ref.digest")}
        mappings = {}
        for name in ("anchor_identity", "candidate_identity", "contention_model",
                     "host_state"):
            value = obj[name]
            if not isinstance(value, Mapping) or not value:
                raise PlanValidationError(f"CalibrationReceipt.{name}: non-empty object required")
            mappings[name] = _freeze(value, f"CalibrationReceipt.{name}")
        return cls(
            CALIBRATION_SCHEMA,
            unit, _text(obj["harness"], "CalibrationReceipt.harness"),  # type: ignore[arg-type]
            n, _freeze({"lower": lower, "upper": upper,
                        "confidence": confidence,
                        "method_ref": _text(interval["method_ref"],
                                            "CalibrationReceipt.interval.method_ref")},
                       "CalibrationReceipt.interval"),
            _text(obj["estimator_id"], "CalibrationReceipt.estimator_id"),  # type: ignore[arg-type]
            _text(obj["metric"], "CalibrationReceipt.metric"),  # type: ignore[arg-type]
            _finite(obj["value"], "CalibrationReceipt.value"),
            mappings["anchor_identity"], mappings["candidate_identity"],
            _digest(obj["raw_sample_digest"], "CalibrationReceipt.raw_sample_digest"),
            unit_ids, mappings["contention_model"], mappings["host_state"],
            _freeze(policy, "CalibrationReceipt.policy_ref"))

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in self.FIELDS}

    @property
    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class CalibrationDisposition:
    status: str
    reasons: tuple[str, ...]
    cache_hit: bool = False
    replay_cache_hit: bool = False


@dataclass
class CalibrationCache:
    """Separate immutable raw-replay and plan-applicability caches."""
    _replay_entries: dict[tuple[str, ...], float] = field(default_factory=dict)
    _applicability_entries: dict[tuple[str, ...], CalibrationDisposition] = field(
        default_factory=dict)

    def __len__(self) -> int:
        return len(self._applicability_entries)

    @property
    def replay_entries(self) -> int:
        return len(self._replay_entries)

    @property
    def applicability_entries(self) -> int:
        return len(self._applicability_entries)


def _normalized_receipt(receipt: CalibrationReceipt) -> CalibrationReceipt:
    if not isinstance(receipt, CalibrationReceipt):
        raise PlanValidationError("expected CalibrationReceipt")
    try:
        return CalibrationReceipt.from_dict(receipt.to_dict())
    except PlanValidationError:
        raise
    except Exception as exc:
        raise PlanValidationError(
            f"invalid directly constructed CalibrationReceipt: {exc}") from exc


def calibration_applicability(
        receipt: CalibrationReceipt, plan: ExperimentPlan, *,
        registered_estimators: Mapping[str, Callable[[CalibrationReceipt], float]] | None,
        registered_rule_id: str | None,
        applicability_rule: Callable[[CalibrationReceipt, ExperimentPlan], bool | str] | None,
        cache: CalibrationCache | None = None) -> CalibrationDisposition:
    """Replay once and ask a registered rule whether a receipt applies.

    The rule callback is mandatory: digest equality alone is not calibration
    transfer authority.  Callback identities must be versioned by their registry
    key/rule ID; changing code without changing that identity violates this API.
    """
    receipt = _normalized_receipt(receipt)
    plan = _normalized_plan(plan)
    if not registered_rule_id or not callable(applicability_rule):
        return CalibrationDisposition(
            "policy_undefined", ("registered calibration applicability rule required",))
    estimator = (registered_estimators or {}).get(receipt.estimator_id)
    if estimator is None:
        return CalibrationDisposition(
            "policy_undefined", (f"unsupported registered estimator {receipt.estimator_id!r}",))
    applicability_key = (
        receipt.digest, plan.digest, plan.policy_snapshot["digest"],
        receipt.estimator_id, registered_rule_id)
    if cache is not None and applicability_key in cache._applicability_entries:
        prior = cache._applicability_entries[applicability_key]
        return CalibrationDisposition(prior.status, prior.reasons, True, True)

    replay_key = (receipt.digest, receipt.estimator_id,
                  plan.policy_snapshot["digest"])
    replay_cache_hit = cache is not None and replay_key in cache._replay_entries
    if replay_cache_hit:
        replayed = cache._replay_entries[replay_key]
    else:
        try:
            replayed = _finite(estimator(receipt), "registered estimator replay")
        except PlanValidationError as exc:
            return CalibrationDisposition(
                "recalibration_required", (str(exc),), False, False)
        except Exception as exc:  # transient callback failure is never cached
            return CalibrationDisposition(
                "recalibration_required",
                (f"registered estimator replay failed: {exc}",), False, False)
        if cache is not None:
            cache._replay_entries[replay_key] = replayed

    reasons: list[str] = []
    if receipt.unit != plan.unit:
        reasons.append("calibration unit differs from plan unit")
    if receipt.metric != plan.metric:
        reasons.append("calibration metric differs from plan metric")
    if receipt.estimator_id != plan.estimator_id:
        reasons.append("calibration estimator differs from plan estimator")
    if receipt.policy_ref != plan.policy_snapshot:
        reasons.append("calibration policy dependency differs from plan")
    if plan.calibration_ref != receipt.digest:
        reasons.append("plan does not reference this calibration receipt")
    if replayed != receipt.value:
        reasons.append("registered estimator replay does not exactly match saved value")
    try:
        rule_result = applicability_rule(receipt, plan)
    except Exception as exc:  # transient callback failure is never cached
        return CalibrationDisposition(
            "recalibration_required",
            tuple(reasons + [f"registered applicability rule failed: {exc}"]),
            False, replay_cache_hit)
    else:
        if rule_result is not True:
            reasons.append(str(rule_result) if isinstance(rule_result, str)
                           else "registered applicability rule refused transfer")
    result = CalibrationDisposition(
        "applicable" if not reasons else "recalibration_required", tuple(reasons),
        False, replay_cache_hit)
    if cache is not None:
        cache._applicability_entries[applicability_key] = result
    return result
