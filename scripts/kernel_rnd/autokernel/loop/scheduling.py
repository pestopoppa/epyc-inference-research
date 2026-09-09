"""Deterministic bounded opportunity selection and held-claim accounting.

This module is deliberately not an admission controller.  A selection is an
offline proposal with ``execution_authorized=False``; only native receipts
supplied by a future trusted consumer can account actual held service.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from bisect import bisect_left, insort
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

CONFIG_SCHEMA = "epyc.autokernel.scheduler_config.v1"
VECTOR_SCHEMA = "epyc.autokernel.resource_vector.v1"
PROPOSAL_SCHEMA = "epyc.autokernel.stage_proposal.v1"
RECEIPT_SCHEMA = "epyc.autokernel.held_claim_receipt.v1"
OUTAGE_SCHEMA = "epyc.autokernel.scheduler_outage.v1"
SEED_SCHEMA = "epyc.autokernel.seed_account.v1"
STATE_SCHEMA = "epyc.autokernel.scheduler_state.v1"
SELECTION_SCHEMA = "epyc.autokernel.stage_selection.v1"
ACCOUNTING_SCHEMA = "epyc.autokernel.accounting_view.v1"
RECEIPT_RECORD_SCHEMA = "epyc.autokernel.accounted_receipt.v1"
OPERATIONAL_SCHEMA = "epyc.autokernel.scheduler_operational_projection.v1"
SELECTION_PREVIEW_SCHEMA = "epyc.autokernel.scheduler_selection_preview.v1"
ACCOUNTING_PREVIEW_SCHEMA = "epyc.autokernel.scheduler_accounting_preview.v1"

STAGE_CLASSES = frozenset({
    "search", "prerequisite", "calibration", "validation", "reject_audit",
    "maintenance", "build", "teardown",
})
RESERVATION_KINDS = frozenset({"seed", "calibration", "validation", "reject_audit",
                               "maintenance", "full_region"})
OUTAGE_KINDS = frozenset({"authority", "resource"})
BACKENDS = frozenset({"cpu", "gpu", "both"})
OUTCOMES = frozenset({"valid_comparison", "invalid", "failed", "prerequisite",
                      "calibration", "validation", "reject_audit", "maintenance"})


class SchedulingRefused(ValueError):
    """An input or transition cannot be proven within this accounting model."""


@dataclass(frozen=True, slots=True)
class OperationalProjection:
    """Bounded scheduler state used by serialized issue transactions.

    Receipt history is represented by its already-computed accounting view and
    counts, never copied into a steady-state transaction.
    """

    body: Mapping[str, Any]
    schema: str = OPERATIONAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OPERATIONAL_SCHEMA or not isinstance(self.body, Mapping):
            raise SchedulingRefused("operational projection is invalid")
        canonical = json.loads(_canonical(self.body))
        if not isinstance(canonical, dict) or canonical.get("schema") != OPERATIONAL_SCHEMA:
            raise SchedulingRefused("operational projection body/schema differs")
        object.__setattr__(self, "body", _deep_freeze(canonical))

    @property
    def projection_digest(self) -> str:
        return digest(_plain(self.body))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "projection_digest": self.projection_digest,
                "body": _plain(self.body)}


@dataclass(frozen=True, slots=True)
class SelectionPreview:
    prior: OperationalProjection
    selection: "Selection"
    after: OperationalProjection
    preview_digest: str
    schema: str = SELECTION_PREVIEW_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "prior_digest": self.prior.projection_digest,
                "selection": self.selection.to_dict(),
                "after_digest": self.after.projection_digest,
                "preview_digest": self.preview_digest}


@dataclass(frozen=True, slots=True)
class AccountingPreview:
    prior: OperationalProjection
    selection: "Selection"
    receipt: "HeldClaimReceipt"
    outcome: str
    after: OperationalProjection
    preview_digest: str
    schema: str = ACCOUNTING_PREVIEW_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "prior_digest": self.prior.projection_digest,
                "selection": self.selection.to_dict(), "receipt": self.receipt.to_dict(),
                "outcome": self.outcome, "after_digest": self.after.projection_digest,
                "preview_digest": self.preview_digest}


def _exact(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchedulingRefused(f"{label} must be an object")
    keys = tuple(value.keys())
    if any(not isinstance(key, str) for key in keys):
        raise SchedulingRefused(f"{label} keys must be strings")
    actual = set(keys)
    if actual != fields:
        raise SchedulingRefused(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}")
    return value


def _text(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SchedulingRefused(f"{label} must be non-empty text")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0,
            maximum: float | None = None) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < minimum
            or (maximum is not None and float(value) > maximum)):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise SchedulingRefused(f"{label} must be finite and >= {minimum}{suffix}")
    return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchedulingRefused(f"{label} must be an integer >= {minimum}")
    return value


def _enum(value: Any, allowed: frozenset[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SchedulingRefused(f"{label} is unsupported")
    return value


def _texts(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SchedulingRefused(f"{label} must be an array")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if len(result) != len(set(result)):
        raise SchedulingRefused(f"{label} contains duplicates")
    return tuple(str(item) for item in result)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _normalize(instance, cls):
    if isinstance(instance, cls):
        return cls.from_dict(instance.to_dict())
    return cls.from_dict(instance)


@dataclass(frozen=True, slots=True)
class ResourceVector:
    physical_region_fraction: float
    gpu_devices: tuple[str, ...]
    memory_reservation_bytes: int
    schema: str = VECTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VECTOR_SCHEMA:
            raise SchedulingRefused("resource vector schema is unsupported")
        object.__setattr__(self, "physical_region_fraction", _number(
            self.physical_region_fraction, "physical_region_fraction", maximum=1.0))
        object.__setattr__(self, "gpu_devices", _texts(self.gpu_devices, "gpu_devices"))
        object.__setattr__(self, "memory_reservation_bytes", _integer(
            self.memory_reservation_bytes, "memory_reservation_bytes"))
        if self.gpu_devices and self.physical_region_fraction <= 0:
            raise SchedulingRefused("GPU claims must also declare a host CPU fraction")

    @classmethod
    def from_dict(cls, value: Any) -> "ResourceVector":
        row = _exact(value, frozenset({"schema", "physical_region_fraction",
                                      "gpu_devices", "memory_reservation_bytes"}),
                     "resource vector")
        return cls(**dict(row))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "physical_region_fraction": self.physical_region_fraction,
                "gpu_devices": list(self.gpu_devices),
                "memory_reservation_bytes": self.memory_reservation_bytes}

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    config_id: str
    max_stage_seconds: float
    noncoverage_slots: int
    reservation_slots: Mapping[str, tuple[int, ...]]
    reservation_shares: Mapping[str, float]
    campaign_attempt_cap: int
    campaign_charged_seconds_cap: float
    seed_attempt_cap: int
    seed_charged_seconds_cap: float
    capacity: ResourceVector
    weights_source: str
    apportionment_rule: str
    adaptive_rule_id: str | None
    normal_weight: float = 1.0
    seed_weight: float = 2.0
    seed_valid_comparison_cap: int = 3
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise SchedulingRefused("scheduler config schema is unsupported")
        object.__setattr__(self, "config_id", _text(self.config_id, "config_id"))
        object.__setattr__(self, "max_stage_seconds", _number(
            self.max_stage_seconds, "max_stage_seconds", minimum=1e-12))
        object.__setattr__(self, "noncoverage_slots", _integer(
            self.noncoverage_slots, "noncoverage_slots"))
        object.__setattr__(self, "campaign_attempt_cap", _integer(
            self.campaign_attempt_cap, "campaign_attempt_cap", minimum=1))
        object.__setattr__(self, "campaign_charged_seconds_cap", _number(
            self.campaign_charged_seconds_cap, "campaign_charged_seconds_cap", minimum=1e-12))
        object.__setattr__(self, "seed_attempt_cap", _integer(
            self.seed_attempt_cap, "seed_attempt_cap", minimum=1))
        object.__setattr__(self, "seed_charged_seconds_cap", _number(
            self.seed_charged_seconds_cap, "seed_charged_seconds_cap", minimum=1e-12))
        object.__setattr__(self, "normal_weight", _number(
            self.normal_weight, "normal_weight", minimum=1.0, maximum=3.0))
        object.__setattr__(self, "seed_weight", _number(
            self.seed_weight, "seed_weight", minimum=1.0, maximum=3.0))
        if self.normal_weight != 1.0 or self.seed_weight != 2.0:
            raise SchedulingRefused("v1 fixed weights are normal=1 and seed=2")
        if _integer(self.seed_valid_comparison_cap, "seed_valid_comparison_cap", minimum=1) != 3:
            raise SchedulingRefused("v1 seed boost ends after exactly three valid comparisons")
        object.__setattr__(self, "capacity", _normalize(self.capacity, ResourceVector))
        for name in ("weights_source", "apportionment_rule"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "adaptive_rule_id", _text(
            self.adaptive_rule_id, "adaptive_rule_id", nullable=True))
        if self.adaptive_rule_id is not None:
            raise SchedulingRefused("scheduler v1 uses fixed weights; adaptation is not configured")
        if not isinstance(self.reservation_slots, Mapping):
            raise SchedulingRefused("reservation_slots must be an object")
        frozen: dict[str, tuple[int, ...]] = {}
        occupied: dict[int, str] = {}
        for kind, slots in self.reservation_slots.items():
            kind = _enum(kind, RESERVATION_KINDS, "reservation kind")
            if not isinstance(slots, (list, tuple)):
                raise SchedulingRefused("reservation slots must be arrays")
            normalized = tuple(_integer(slot, "reservation slot") for slot in slots)
            if len(normalized) != len(set(normalized)):
                raise SchedulingRefused("reservation kind repeats a slot")
            for slot in normalized:
                if slot >= self.noncoverage_slots:
                    raise SchedulingRefused("reservation slot is outside noncoverage K")
                if slot in occupied:
                    raise SchedulingRefused(
                        f"reservation slot {slot} conflicts: {occupied[slot]} versus {kind}")
                occupied[slot] = kind
            frozen[kind] = normalized
        object.__setattr__(self, "reservation_slots", MappingProxyType(frozen))
        if not isinstance(self.reservation_shares, Mapping):
            raise SchedulingRefused("reservation_shares must be an object")
        shares: dict[str, float] = {}
        for kind, share in self.reservation_shares.items():
            kind = _enum(kind, RESERVATION_KINDS, "reservation share kind")
            shares[kind] = _number(share, "reservation share", maximum=1.0)
            expected = (len(frozen.get(kind, ())) / self.noncoverage_slots
                        if self.noncoverage_slots else 0.0)
            if not math.isclose(shares[kind], expected, rel_tol=0.0, abs_tol=1e-12):
                raise SchedulingRefused("reservation share does not match frozen slot allocation")
        if set(shares) != set(frozen) or sum(shares.values()) > 1.0 + 1e-12:
            raise SchedulingRefused("reservation shares must exactly cover reserved slot kinds")
        object.__setattr__(self, "reservation_shares", MappingProxyType(shares))

    @classmethod
    def from_dict(cls, value: Any) -> "SchedulerConfig":
        fields = frozenset({
            "schema", "config_id", "max_stage_seconds", "noncoverage_slots",
            "reservation_slots", "reservation_shares", "campaign_attempt_cap",
            "campaign_charged_seconds_cap",
            "seed_attempt_cap", "seed_charged_seconds_cap", "capacity", "weights_source",
            "apportionment_rule", "adaptive_rule_id", "normal_weight", "seed_weight",
            "seed_valid_comparison_cap",
        })
        row = dict(_exact(value, fields, "scheduler config"))
        row["capacity"] = ResourceVector.from_dict(row["capacity"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "config_id": self.config_id,
                "max_stage_seconds": self.max_stage_seconds,
                "noncoverage_slots": self.noncoverage_slots,
                "reservation_slots": {key: list(value) for key, value in self.reservation_slots.items()},
                "reservation_shares": dict(self.reservation_shares),
                "campaign_attempt_cap": self.campaign_attempt_cap,
                "campaign_charged_seconds_cap": self.campaign_charged_seconds_cap,
                "seed_attempt_cap": self.seed_attempt_cap,
                "seed_charged_seconds_cap": self.seed_charged_seconds_cap,
                "capacity": self.capacity.to_dict(), "weights_source": self.weights_source,
                "apportionment_rule": self.apportionment_rule,
                "adaptive_rule_id": self.adaptive_rule_id,
                "normal_weight": self.normal_weight, "seed_weight": self.seed_weight,
                "seed_valid_comparison_cap": self.seed_valid_comparison_cap}

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    @property
    def policy_digest(self) -> str:
        value = self.to_dict()
        value.pop("capacity")
        return digest(value)


@dataclass(frozen=True, slots=True)
class StageProposal:
    proposal_id: str
    submitted_at: float
    backend: str
    target_revision: str
    alias_identity: str
    frontier_id: str | None
    production_frontier: bool
    seed_id: str | None
    stage_class: str
    estimated_duration_seconds: float
    estimated_claims: ResourceVector
    eligible: bool
    eligibility_ref: str
    reservation_kind: str | None
    full_region: bool
    compatibility_authority_refs: tuple[str, ...]
    safe_chunking_declared: bool
    schema: str = PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROPOSAL_SCHEMA:
            raise SchedulingRefused("stage proposal schema is unsupported")
        for name in ("proposal_id", "target_revision", "alias_identity", "eligibility_ref"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "backend", _enum(self.backend, BACKENDS, "backend"))
        object.__setattr__(self, "frontier_id", _text(self.frontier_id, "frontier_id", nullable=True))
        object.__setattr__(self, "seed_id", _text(self.seed_id, "seed_id", nullable=True))
        object.__setattr__(self, "submitted_at", _number(self.submitted_at, "submitted_at"))
        object.__setattr__(self, "stage_class", _enum(
            self.stage_class, STAGE_CLASSES, "stage_class"))
        object.__setattr__(self, "estimated_duration_seconds", _number(
            self.estimated_duration_seconds, "estimated_duration_seconds", minimum=1e-12))
        object.__setattr__(self, "estimated_claims", _normalize(
            self.estimated_claims, ResourceVector))
        for name in ("production_frontier", "eligible", "full_region",
                     "safe_chunking_declared"):
            if not isinstance(getattr(self, name), bool):
                raise SchedulingRefused(f"{name} must be boolean")
        if self.production_frontier != (self.frontier_id is not None):
            raise SchedulingRefused("production frontier flag and frontier_id disagree")
        if self.reservation_kind is not None:
            object.__setattr__(self, "reservation_kind", _enum(
                self.reservation_kind, RESERVATION_KINDS, "reservation_kind"))
        object.__setattr__(self, "compatibility_authority_refs", _texts(
            self.compatibility_authority_refs, "compatibility_authority_refs"))
        if self.full_region and self.estimated_claims.physical_region_fraction != 1.0:
            raise SchedulingRefused("full_region proposal must claim the full physical region")

    @classmethod
    def from_dict(cls, value: Any) -> "StageProposal":
        fields = frozenset({
            "schema", "proposal_id", "submitted_at", "backend", "target_revision",
            "alias_identity", "frontier_id", "production_frontier", "seed_id",
            "stage_class", "estimated_duration_seconds", "estimated_claims", "eligible",
            "eligibility_ref", "reservation_kind", "full_region",
            "compatibility_authority_refs", "safe_chunking_declared",
        })
        row = dict(_exact(value, fields, "stage proposal"))
        row["estimated_claims"] = ResourceVector.from_dict(row["estimated_claims"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "proposal_id": self.proposal_id,
                "submitted_at": self.submitted_at, "backend": self.backend,
                "target_revision": self.target_revision, "alias_identity": self.alias_identity,
                "frontier_id": self.frontier_id, "production_frontier": self.production_frontier,
                "seed_id": self.seed_id, "stage_class": self.stage_class,
                "estimated_duration_seconds": self.estimated_duration_seconds,
                "estimated_claims": self.estimated_claims.to_dict(), "eligible": self.eligible,
                "eligibility_ref": self.eligibility_ref,
                "reservation_kind": self.reservation_kind, "full_region": self.full_region,
                "compatibility_authority_refs": list(self.compatibility_authority_refs),
                "safe_chunking_declared": self.safe_chunking_declared}

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class HeldClaimReceipt:
    receipt_id: str
    proposal_id: str
    backend: str
    stage_class: str
    started_at: float
    ended_at: float
    ownership_generation: int
    allocation_generation: int
    physical_claim_ids: tuple[str, ...]
    physical_region_fraction: float
    gpu_device_ids: tuple[str, ...]
    memory_reservation_bytes: int
    affinity_cores: tuple[str, ...]
    beneficiary_shares: Mapping[str, float]
    schema: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise SchedulingRefused("held receipt schema is unsupported")
        for name in ("receipt_id", "proposal_id", "backend"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "stage_class", _enum(
            self.stage_class, STAGE_CLASSES, "receipt stage_class"))
        start = _number(self.started_at, "started_at")
        end = _number(self.ended_at, "ended_at")
        if end <= start:
            raise SchedulingRefused("receipt interval must have positive duration")
        object.__setattr__(self, "started_at", start)
        object.__setattr__(self, "ended_at", end)
        object.__setattr__(self, "ownership_generation", _integer(
            self.ownership_generation, "ownership_generation", minimum=1))
        object.__setattr__(self, "allocation_generation", _integer(
            self.allocation_generation, "allocation_generation", minimum=1))
        object.__setattr__(self, "physical_claim_ids", _texts(
            self.physical_claim_ids, "physical_claim_ids"))
        object.__setattr__(self, "physical_region_fraction", _number(
            self.physical_region_fraction, "physical_region_fraction", maximum=1.0))
        object.__setattr__(self, "gpu_device_ids", _texts(
            self.gpu_device_ids, "gpu_device_ids"))
        object.__setattr__(self, "memory_reservation_bytes", _integer(
            self.memory_reservation_bytes, "memory_reservation_bytes"))
        object.__setattr__(self, "affinity_cores", _texts(self.affinity_cores, "affinity_cores"))
        if self.gpu_device_ids and self.physical_region_fraction <= 0:
            raise SchedulingRefused("GPU receipt must include its host CPU claim")
        if ((self.physical_region_fraction > 0 or self.gpu_device_ids
             or self.memory_reservation_bytes > 0) and not self.physical_claim_ids):
            raise SchedulingRefused("held resources require an explicit physical claim ID")
        if not isinstance(self.beneficiary_shares, Mapping) or not self.beneficiary_shares:
            raise SchedulingRefused("beneficiary_shares must be a non-empty object")
        shares: dict[str, float] = {}
        for beneficiary, share in self.beneficiary_shares.items():
            beneficiary = _text(beneficiary, "beneficiary")
            shares[beneficiary] = _number(share, "beneficiary share", minimum=1e-12,
                                          maximum=1.0)
        if not math.isclose(sum(shares.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise SchedulingRefused("beneficiary shares must sum exactly to one within 1e-12")
        object.__setattr__(self, "beneficiary_shares", MappingProxyType(shares))

    @classmethod
    def from_dict(cls, value: Any) -> "HeldClaimReceipt":
        fields = frozenset({
            "schema", "receipt_id", "proposal_id", "backend", "stage_class", "started_at",
            "ended_at", "ownership_generation", "allocation_generation",
            "physical_claim_ids", "physical_region_fraction", "gpu_device_ids",
            "memory_reservation_bytes", "affinity_cores", "beneficiary_shares",
        })
        return cls(**dict(_exact(value, fields, "held claim receipt")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "receipt_id": self.receipt_id,
                "proposal_id": self.proposal_id, "backend": self.backend,
                "stage_class": self.stage_class, "started_at": self.started_at,
                "ended_at": self.ended_at, "ownership_generation": self.ownership_generation,
                "allocation_generation": self.allocation_generation,
                "physical_claim_ids": list(self.physical_claim_ids),
                "physical_region_fraction": self.physical_region_fraction,
                "gpu_device_ids": list(self.gpu_device_ids),
                "memory_reservation_bytes": self.memory_reservation_bytes,
                "affinity_cores": list(self.affinity_cores),
                "beneficiary_shares": dict(self.beneficiary_shares)}

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class Outage:
    outage_id: str
    kind: str
    started_at: float
    ended_at: float | None
    reason: str
    backend: str | None = None
    frontier_id: str | None = None
    schema: str = OUTAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OUTAGE_SCHEMA:
            raise SchedulingRefused("outage schema is unsupported")
        object.__setattr__(self, "outage_id", _text(self.outage_id, "outage_id"))
        object.__setattr__(self, "kind", _enum(self.kind, OUTAGE_KINDS, "outage kind"))
        start = _number(self.started_at, "outage started_at")
        end = None if self.ended_at is None else _number(self.ended_at, "outage ended_at")
        if end is not None and end < start:
            raise SchedulingRefused("outage ends before it starts")
        object.__setattr__(self, "started_at", start)
        object.__setattr__(self, "ended_at", end)
        object.__setattr__(self, "reason", _text(self.reason, "outage reason"))
        if self.backend is not None:
            object.__setattr__(self, "backend", _enum(
                self.backend, BACKENDS, "outage backend"))
        object.__setattr__(self, "frontier_id", _text(
            self.frontier_id, "outage frontier_id", nullable=True))

    @classmethod
    def from_dict(cls, value: Any) -> "Outage":
        fields = frozenset({"schema", "outage_id", "kind", "started_at", "ended_at", "reason",
                            "backend", "frontier_id"})
        return cls(**dict(_exact(value, fields, "outage")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "outage_id": self.outage_id, "kind": self.kind,
                "started_at": self.started_at, "ended_at": self.ended_at, "reason": self.reason,
                "backend": self.backend, "frontier_id": self.frontier_id}

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class SeedAccount:
    seed_id: str
    backend: str
    target_revision: str
    alias_identity: str
    seed_ids: tuple[str, ...]
    first_submitted_at: float
    attempts: int = 0
    charged_seconds: float = 0.0
    valid_comparisons: int = 0
    boosted: bool = True
    schema: str = SEED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEED_SCHEMA:
            raise SchedulingRefused("seed account schema is unsupported")
        for name in ("seed_id", "backend", "target_revision", "alias_identity"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "seed_ids", _texts(self.seed_ids, "seed_ids"))
        if self.seed_id not in self.seed_ids:
            raise SchedulingRefused("canonical seed_id must be present in seed_ids")
        object.__setattr__(self, "first_submitted_at", _number(
            self.first_submitted_at, "first_submitted_at"))
        object.__setattr__(self, "attempts", _integer(self.attempts, "seed attempts"))
        object.__setattr__(self, "charged_seconds", _number(
            self.charged_seconds, "seed charged_seconds"))
        object.__setattr__(self, "valid_comparisons", _integer(
            self.valid_comparisons, "seed valid_comparisons"))
        if not isinstance(self.boosted, bool):
            raise SchedulingRefused("seed boosted must be boolean")

    @classmethod
    def from_dict(cls, value: Any) -> "SeedAccount":
        fields = frozenset({"schema", "seed_id", "backend", "target_revision",
                            "alias_identity", "seed_ids", "first_submitted_at", "attempts",
                            "charged_seconds", "valid_comparisons", "boosted"})
        return cls(**dict(_exact(value, fields, "seed account")))

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in (
            "schema", "seed_id", "backend", "target_revision", "alias_identity",
            "seed_ids", "first_submitted_at", "attempts", "charged_seconds", "valid_comparisons",
            "boosted")}


@dataclass(frozen=True, slots=True)
class AccountedReceipt:
    receipt_id: str
    receipt_digest: str
    selection_digest: str
    outcome: str
    schema: str = RECEIPT_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_RECORD_SCHEMA:
            raise SchedulingRefused("accounted receipt schema is unsupported")
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, "receipt_id"))
        for name in ("receipt_digest", "selection_digest"):
            value = _text(getattr(self, name), name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise SchedulingRefused(f"{name} must be a lowercase SHA-256")
        object.__setattr__(self, "outcome", _enum(self.outcome, OUTCOMES, "receipt outcome"))

    @classmethod
    def from_dict(cls, value: Any) -> "AccountedReceipt":
        fields = frozenset({"schema", "receipt_id", "receipt_digest",
                            "selection_digest", "outcome"})
        return cls(**dict(_exact(value, fields, "accounted receipt")))

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in (
            "schema", "receipt_id", "receipt_digest", "selection_digest", "outcome")}


def _number_map(value: Any, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise SchedulingRefused(f"{label} must be an object")
    result: dict[str, float] = {}
    for key, item in value.items():
        key = _text(key, f"{label} key")
        result[key] = _number(item, f"{label}.{key}")
    return MappingProxyType(result)


def _text_map(value: Any, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise SchedulingRefused(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        key = _text(key, f"{label} key")
        result[key] = _text(item, f"{label}.{key}")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class SchedulerState:
    scheduler_id: str
    config_digest: str
    policy_digest: str
    accounting_epoch: int
    capacity: ResourceVector
    capacity_digest: str
    round_number: int
    frozen_frontier: tuple[str, ...]
    coverage_debt: Mapping[str, str]
    used_coverage: Mapping[str, str]
    used_noncoverage: tuple[str, ...]
    skipped_noncoverage: tuple[int, ...]
    round_reservations: Mapping[str, str]
    round_seed_used: bool
    deficits: Mapping[str, float]
    seed_accounts: tuple[SeedAccount, ...]
    campaign_attempts: int
    campaign_charged_seconds: float
    receipts: tuple[HeldClaimReceipt, ...]
    accounted_receipts: tuple[AccountedReceipt, ...]
    successor_fences: tuple[str, ...]
    issued_selection_digests: tuple[str, ...]
    existing_stage_until: float | None
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATE_SCHEMA:
            raise SchedulingRefused("scheduler state schema is unsupported")
        object.__setattr__(self, "scheduler_id", _text(self.scheduler_id, "scheduler_id"))
        for name in ("config_digest", "policy_digest", "capacity_digest"):
            value = _text(getattr(self, name), name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise SchedulingRefused(f"{name} must be a lowercase SHA-256")
        object.__setattr__(self, "accounting_epoch", _integer(
            self.accounting_epoch, "accounting_epoch", minimum=1))
        object.__setattr__(self, "capacity", _normalize(self.capacity, ResourceVector))
        if self.capacity_digest != self.capacity.digest:
            raise SchedulingRefused("capacity digest does not match capacity vector")
        object.__setattr__(self, "round_number", _integer(
            self.round_number, "round_number"))
        object.__setattr__(self, "frozen_frontier", _texts(
            self.frozen_frontier, "frozen_frontier"))
        object.__setattr__(self, "coverage_debt", _text_map(
            self.coverage_debt, "coverage_debt"))
        object.__setattr__(self, "used_coverage", _text_map(
            self.used_coverage, "used_coverage"))
        if not set(self.used_coverage).issubset(self.frozen_frontier):
            raise SchedulingRefused("used coverage is outside frozen frontier")
        object.__setattr__(self, "used_noncoverage", _texts(
            self.used_noncoverage, "used_noncoverage"))
        if not isinstance(self.skipped_noncoverage, (list, tuple)):
            raise SchedulingRefused("skipped_noncoverage must be an array")
        skipped = tuple(_integer(slot, "skipped noncoverage slot")
                        for slot in self.skipped_noncoverage)
        if skipped != tuple(sorted(set(skipped))):
            raise SchedulingRefused("skipped noncoverage slots must be unique and sorted")
        object.__setattr__(self, "skipped_noncoverage", skipped)
        object.__setattr__(self, "round_reservations", _text_map(
            self.round_reservations, "round_reservations"))
        for slot in self.round_reservations:
            if not slot.isdigit():
                raise SchedulingRefused("round reservation slot keys must be decimal integers")
        for kind in self.round_reservations.values():
            _enum(kind, RESERVATION_KINDS, "round reservation kind")
        if not isinstance(self.round_seed_used, bool):
            raise SchedulingRefused("round_seed_used must be boolean")
        object.__setattr__(self, "deficits", _number_map(self.deficits, "deficits"))
        if not isinstance(self.seed_accounts, (list, tuple)):
            raise SchedulingRefused("seed_accounts must be an array")
        seeds = tuple(_normalize(item, SeedAccount) for item in self.seed_accounts)
        if len({seed.seed_id for seed in seeds}) != len(seeds):
            raise SchedulingRefused("duplicate seed accounts")
        all_seed_ids = [seed_id for seed in seeds for seed_id in seed.seed_ids]
        if len(all_seed_ids) != len(set(all_seed_ids)):
            raise SchedulingRefused("seed ID occurs in multiple accounts")
        object.__setattr__(self, "seed_accounts", seeds)
        object.__setattr__(self, "campaign_attempts", _integer(
            self.campaign_attempts, "campaign_attempts"))
        object.__setattr__(self, "campaign_charged_seconds", _number(
            self.campaign_charged_seconds, "campaign_charged_seconds"))
        if not isinstance(self.receipts, (list, tuple)):
            raise SchedulingRefused("receipts must be an array")
        receipts = tuple(_normalize(item, HeldClaimReceipt) for item in self.receipts)
        ids = [receipt.receipt_id for receipt in receipts]
        if len(ids) != len(set(ids)):
            raise SchedulingRefused("state contains duplicate receipt IDs")
        _check_receipt_overlaps(receipts)
        object.__setattr__(self, "receipts", receipts)
        if not isinstance(self.accounted_receipts, (list, tuple)):
            raise SchedulingRefused("accounted_receipts must be an array")
        records = tuple(_normalize(item, AccountedReceipt) for item in self.accounted_receipts)
        record_ids = [record.receipt_id for record in records]
        if len(record_ids) != len(set(record_ids)) or set(record_ids) != set(ids):
            raise SchedulingRefused("accounted receipt records must exactly bind receipt IDs")
        by_receipt = {receipt.receipt_id: receipt for receipt in receipts}
        if any(record.receipt_digest != by_receipt[record.receipt_id].digest
               for record in records):
            raise SchedulingRefused("accounted receipt digest does not match receipt")
        object.__setattr__(self, "accounted_receipts", records)
        object.__setattr__(self, "successor_fences", _texts(
            self.successor_fences, "successor_fences"))
        issued = _texts(self.issued_selection_digests, "issued_selection_digests")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
               for value in issued):
            raise SchedulingRefused("issued selection digests must be lowercase SHA-256")
        object.__setattr__(self, "issued_selection_digests", issued)
        if self.existing_stage_until is not None:
            object.__setattr__(self, "existing_stage_until", _number(
                self.existing_stage_until, "existing_stage_until"))

    @classmethod
    def from_dict(cls, value: Any) -> "SchedulerState":
        fields = frozenset({
            "schema", "scheduler_id", "config_digest", "policy_digest", "accounting_epoch",
            "capacity", "capacity_digest", "round_number", "frozen_frontier", "used_coverage",
            "coverage_debt", "used_noncoverage", "skipped_noncoverage",
            "round_reservations", "round_seed_used", "deficits",
            "seed_accounts", "campaign_attempts",
            "campaign_charged_seconds", "receipts", "accounted_receipts",
            "successor_fences", "issued_selection_digests", "existing_stage_until",
        })
        row = dict(_exact(value, fields, "scheduler state"))
        row["capacity"] = ResourceVector.from_dict(row["capacity"])
        row["seed_accounts"] = tuple(SeedAccount.from_dict(item) for item in row["seed_accounts"])
        row["receipts"] = tuple(HeldClaimReceipt.from_dict(item) for item in row["receipts"])
        row["accounted_receipts"] = tuple(
            AccountedReceipt.from_dict(item) for item in row["accounted_receipts"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "scheduler_id": self.scheduler_id,
                "config_digest": self.config_digest, "policy_digest": self.policy_digest,
                "accounting_epoch": self.accounting_epoch,
                "capacity": self.capacity.to_dict(),
                "capacity_digest": self.capacity_digest, "round_number": self.round_number,
                "frozen_frontier": list(self.frozen_frontier),
                "coverage_debt": dict(self.coverage_debt),
                "used_coverage": dict(self.used_coverage),
                "used_noncoverage": list(self.used_noncoverage),
                "skipped_noncoverage": list(self.skipped_noncoverage),
                "round_reservations": dict(self.round_reservations),
                "round_seed_used": self.round_seed_used,
                "deficits": dict(self.deficits),
                "seed_accounts": [seed.to_dict() for seed in self.seed_accounts],
                "campaign_attempts": self.campaign_attempts,
                "campaign_charged_seconds": self.campaign_charged_seconds,
                "receipts": [receipt.to_dict() for receipt in self.receipts],
                "accounted_receipts": [record.to_dict() for record in self.accounted_receipts],
                "successor_fences": list(self.successor_fences),
                "issued_selection_digests": list(self.issued_selection_digests),
                "existing_stage_until": self.existing_stage_until}


@dataclass(frozen=True, slots=True)
class Selection:
    status: str
    reasons: tuple[str, ...]
    proposal: StageProposal | None
    slot_kind: str | None
    slot_index: int | None
    round_number: int
    scheduler_id: str
    config_digest: str
    accounting_epoch: int
    capacity_digest: str
    proposal_digest: str | None
    service_bound_seconds: float
    outage_seconds: float
    existing_stage_seconds: float = 0.0
    execution_authorized: bool = False
    schema: str = SELECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SELECTION_SCHEMA or self.status not in {
                "selected", "refused", "waiting", "complete"}:
            raise SchedulingRefused("selection schema/status is unsupported")
        object.__setattr__(self, "reasons", _texts(self.reasons, "selection reasons"))
        if self.proposal is not None:
            object.__setattr__(self, "proposal", _normalize(self.proposal, StageProposal))
        if (self.status == "selected") != (self.proposal is not None):
            raise SchedulingRefused("selected status and proposal presence disagree")
        allowed_slots = RESERVATION_KINDS | {"coverage", "noncoverage"}
        if self.slot_kind is not None:
            object.__setattr__(self, "slot_kind", _enum(
                self.slot_kind, allowed_slots, "selection slot_kind"))
        if self.slot_index is not None:
            object.__setattr__(self, "slot_index", _integer(
                self.slot_index, "selection slot_index"))
        if self.status == "selected" and (self.slot_kind is None or self.slot_index is None):
            raise SchedulingRefused("selected opportunity requires a concrete slot")
        object.__setattr__(self, "round_number", _integer(
            self.round_number, "selection round_number"))
        object.__setattr__(self, "scheduler_id", _text(
            self.scheduler_id, "selection scheduler_id"))
        for name in ("config_digest", "capacity_digest"):
            value = _text(getattr(self, name), f"selection {name}")
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise SchedulingRefused(f"selection {name} must be a lowercase SHA-256")
        object.__setattr__(self, "accounting_epoch", _integer(
            self.accounting_epoch, "selection accounting_epoch", minimum=1))
        if self.proposal_digest is None:
            if self.proposal is not None:
                raise SchedulingRefused("selected proposal requires its digest")
        else:
            value = _text(self.proposal_digest, "selection proposal_digest")
            if (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                    or self.proposal is None or value != self.proposal.digest):
                raise SchedulingRefused("selection proposal digest does not match proposal")
        object.__setattr__(self, "service_bound_seconds", _number(
            self.service_bound_seconds, "selection service_bound_seconds"))
        object.__setattr__(self, "outage_seconds", _number(
            self.outage_seconds, "selection outage_seconds"))
        object.__setattr__(self, "existing_stage_seconds", _number(
            self.existing_stage_seconds, "selection existing_stage_seconds"))
        if self.execution_authorized is not False:
            raise SchedulingRefused("scheduler selection cannot authorize execution")

    @classmethod
    def from_dict(cls, value: Any) -> "Selection":
        fields = frozenset({"schema", "status", "reasons", "proposal", "slot_kind",
                            "slot_index", "round_number", "scheduler_id", "config_digest",
                            "accounting_epoch", "capacity_digest", "proposal_digest",
                            "service_bound_seconds",
                            "outage_seconds", "existing_stage_seconds",
                            "execution_authorized"})
        row = dict(_exact(value, fields, "selection"))
        if row["proposal"] is not None:
            row["proposal"] = StageProposal.from_dict(row["proposal"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "status": self.status, "reasons": list(self.reasons),
                "proposal": None if self.proposal is None else self.proposal.to_dict(),
                "slot_kind": self.slot_kind, "slot_index": self.slot_index,
                "round_number": self.round_number,
                "scheduler_id": self.scheduler_id, "config_digest": self.config_digest,
                "accounting_epoch": self.accounting_epoch,
                "capacity_digest": self.capacity_digest,
                "proposal_digest": self.proposal_digest,
                "service_bound_seconds": self.service_bound_seconds,
                "outage_seconds": self.outage_seconds,
                "existing_stage_seconds": self.existing_stage_seconds,
                "execution_authorized": False}

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class AccountingView:
    receipt_count: int
    held_seconds: float
    physical_region_seconds: float
    gpu_device_seconds: Mapping[str, float]
    memory_byte_seconds: float
    beneficiary_seconds: Mapping[str, float]
    view_digest: str
    schema: str = ACCOUNTING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ACCOUNTING_SCHEMA:
            raise SchedulingRefused("accounting view schema is unsupported")
        object.__setattr__(self, "receipt_count", _integer(
            self.receipt_count, "receipt_count"))
        for name in ("held_seconds", "physical_region_seconds", "memory_byte_seconds"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        object.__setattr__(self, "gpu_device_seconds", _number_map(
            self.gpu_device_seconds, "gpu_device_seconds"))
        object.__setattr__(self, "beneficiary_seconds", _number_map(
            self.beneficiary_seconds, "beneficiary_seconds"))
        value = _text(self.view_digest, "view_digest")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise SchedulingRefused("view_digest must be a lowercase SHA-256")
        body = {"schema": self.schema, "receipt_count": self.receipt_count,
                "held_seconds": self.held_seconds,
                "physical_region_seconds": self.physical_region_seconds,
                "gpu_device_seconds": dict(self.gpu_device_seconds),
                "memory_byte_seconds": self.memory_byte_seconds,
                "beneficiary_seconds": dict(self.beneficiary_seconds)}
        if self.view_digest != digest(body):
            raise SchedulingRefused("accounting view digest does not match its content")

    @classmethod
    def from_dict(cls, value: Any) -> "AccountingView":
        fields = frozenset({"schema", "receipt_count", "held_seconds",
                            "physical_region_seconds", "gpu_device_seconds",
                            "memory_byte_seconds", "beneficiary_seconds", "view_digest"})
        return cls(**dict(_exact(value, fields, "accounting view")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "receipt_count": self.receipt_count,
                "held_seconds": self.held_seconds,
                "physical_region_seconds": self.physical_region_seconds,
                "gpu_device_seconds": dict(self.gpu_device_seconds),
                "memory_byte_seconds": self.memory_byte_seconds,
                "beneficiary_seconds": dict(self.beneficiary_seconds),
                "view_digest": self.view_digest}


def initial_state(config: SchedulerConfig | Mapping[str, Any], scheduler_id: str) -> SchedulerState:
    config = _normalize(config, SchedulerConfig)
    return SchedulerState(
        scheduler_id=_text(scheduler_id, "scheduler_id"), config_digest=config.digest,
        policy_digest=config.policy_digest,
        accounting_epoch=1, capacity=config.capacity,
        capacity_digest=config.capacity.digest, round_number=0,
        frozen_frontier=(), coverage_debt={}, used_coverage={}, used_noncoverage=(),
        skipped_noncoverage=(), deficits={},
        round_reservations={}, round_seed_used=False,
        seed_accounts=(), campaign_attempts=0, campaign_charged_seconds=0.0,
        receipts=(), accounted_receipts=(), successor_fences=(),
        issued_selection_digests=(), existing_stage_until=None)


def _outage_seconds(outages: Sequence[Outage], now: float) -> float:
    """Return union wall time; simultaneous outage labels never double charge time."""
    intervals = sorted((outage.started_at,
                        now if outage.ended_at is None else min(now, outage.ended_at))
                       for outage in outages)
    total = 0.0
    start = end = None
    for left, right in intervals:
        if right <= left:
            continue
        if start is None:
            start, end = left, right
        elif left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total if start is None else total + end - start


def _outage_applies(outage: Outage, proposal: StageProposal) -> bool:
    return ((outage.backend is None or outage.backend == proposal.backend)
            and (outage.frontier_id is None or outage.frontier_id == proposal.frontier_id))


def _dominant_estimate(capacity: ResourceVector, proposal: StageProposal) -> float:
    vector = proposal.estimated_claims
    terms: list[float] = []
    if capacity.physical_region_fraction:
        terms.append(vector.physical_region_fraction / capacity.physical_region_fraction)
    elif vector.physical_region_fraction:
        return math.inf
    if capacity.gpu_devices:
        terms.append(len(vector.gpu_devices) / len(capacity.gpu_devices))
    elif vector.gpu_devices:
        return math.inf
    if capacity.memory_reservation_bytes:
        terms.append(vector.memory_reservation_bytes / capacity.memory_reservation_bytes)
    elif vector.memory_reservation_bytes:
        return math.inf
    return max(terms, default=0.0) * proposal.estimated_duration_seconds


def _validate_state_for_config(config: SchedulerConfig, state: SchedulerState) -> None:
    if state.config_digest != config.digest:
        raise SchedulingRefused("state/config digest mismatch")
    expected_reservations = {
        str(slot): kind for kind, slots in config.reservation_slots.items() for slot in slots
    }
    if len(state.used_noncoverage) + len(state.skipped_noncoverage) > config.noncoverage_slots:
        raise SchedulingRefused("state has consumed more than K noncoverage opportunities")
    if any(slot >= config.noncoverage_slots for slot in state.skipped_noncoverage):
        raise SchedulingRefused("state skipped a noncoverage slot outside K")
    if state.policy_digest != config.policy_digest:
        raise SchedulingRefused("state scheduling policy digest mismatch")
    boosted_backends = [seed.backend for seed in state.seed_accounts if seed.boosted]
    if len(boosted_backends) != len(set(boosted_backends)):
        raise SchedulingRefused("state has more than one boosted seed for a backend")
    for seed in state.seed_accounts:
        if seed.boosted and (seed.attempts >= config.seed_attempt_cap
                             or seed.charged_seconds >= config.seed_charged_seconds_cap
                             or seed.valid_comparisons >= config.seed_valid_comparison_cap):
            raise SchedulingRefused("state retains a boost after a seed cap was reached")
    if state.round_number == 0:
        if (state.frozen_frontier or state.coverage_debt or state.used_coverage
                or state.used_noncoverage or state.skipped_noncoverage
                or state.round_reservations or state.round_seed_used):
            raise SchedulingRefused("unstarted state contains round-local accounting")
    elif dict(state.round_reservations) != expected_reservations:
        raise SchedulingRefused("state round reservations do not match frozen configuration")


def _check_receipt_overlaps(receipts: Sequence[HeldClaimReceipt]) -> None:
    by_claim: dict[str, list[tuple[float, float, str]]] = {}
    for receipt in receipts:
        for claim in (*receipt.physical_claim_ids, *receipt.gpu_device_ids):
            by_claim.setdefault(claim, []).append(
                (receipt.started_at, receipt.ended_at, receipt.receipt_id))
    for claim, intervals in by_claim.items():
        intervals.sort()
        for prior, current in zip(intervals, intervals[1:]):
            if current[0] < prior[1]:
                raise SchedulingRefused(
                    f"overlapping reuse of physical claim {claim}: {prior[2]}/{current[2]}")


def charge_receipts(receipts: Sequence[HeldClaimReceipt | Mapping[str, Any]]) -> AccountingView:
    normalized = tuple(_normalize(receipt, HeldClaimReceipt) for receipt in receipts)
    by_id: dict[str, HeldClaimReceipt] = {}
    for receipt in normalized:
        prior = by_id.get(receipt.receipt_id)
        if prior is not None and prior.digest != receipt.digest:
            raise SchedulingRefused("receipt ID reused for different content")
        by_id[receipt.receipt_id] = receipt
    unique = tuple(by_id.values())
    _check_receipt_overlaps(unique)
    physical = 0.0
    memory = 0.0
    gpu: dict[str, float] = {}
    attribution: dict[str, float] = {}
    for receipt in unique:
        duration = receipt.ended_at - receipt.started_at
        physical += duration * receipt.physical_region_fraction
        memory += duration * receipt.memory_reservation_bytes
        for device in receipt.gpu_device_ids:
            gpu[device] = gpu.get(device, 0.0) + duration
        for beneficiary, share in receipt.beneficiary_shares.items():
            attribution[beneficiary] = attribution.get(beneficiary, 0.0) + duration * share
    body = {"schema": ACCOUNTING_SCHEMA, "receipt_count": len(unique),
            "held_seconds": float(sum(
                receipt.ended_at - receipt.started_at for receipt in unique)),
            "physical_region_seconds": physical, "gpu_device_seconds": gpu,
            "memory_byte_seconds": memory, "beneficiary_seconds": attribution}
    return AccountingView(receipt_count=len(unique),
                          held_seconds=body["held_seconds"],
                          physical_region_seconds=physical,
                          gpu_device_seconds=gpu, memory_byte_seconds=memory,
                          beneficiary_seconds=attribution, view_digest=digest(body))


class SchedulerEngine:
    """Campaign-scoped indexed owner for bounded selection and accounting.

    Construction/replay validates the complete serialized history once.  Hot-path
    selection never reads receipts; accounting uses an ID dict, affected claim
    interval lists and compact totals.
    """

    def __init__(self, config: SchedulerConfig | Mapping[str, Any],
                 state: SchedulerState | Mapping[str, Any]) -> None:
        self.config = _normalize(config, SchedulerConfig)
        source = _normalize(state, SchedulerState)
        _validate_state_for_config(self.config, source)
        for name in ("scheduler_id", "config_digest", "policy_digest", "accounting_epoch",
                     "capacity", "capacity_digest", "round_number", "round_seed_used",
                     "campaign_attempts", "campaign_charged_seconds", "existing_stage_until",
                     "issued_selection_digests"):
            setattr(self, name, getattr(source, name))
        self.frozen_frontier = list(source.frozen_frontier)
        self.coverage_debt = dict(source.coverage_debt)
        self.used_coverage = dict(source.used_coverage)
        self.used_noncoverage = list(source.used_noncoverage)
        self.skipped_noncoverage = list(source.skipped_noncoverage)
        self.round_reservations = dict(source.round_reservations)
        self.deficits = dict(source.deficits)
        self.seed_accounts = list(source.seed_accounts)
        self._seed_by_identity: dict[tuple[str, str, str], SeedAccount] = {}
        self._seed_by_id: dict[str, SeedAccount] = {}
        self._seed_position: dict[tuple[str, str, str], int] = {}
        self._boosted_by_backend: dict[str, SeedAccount] = {}
        self._seed_queue: dict[str, list[tuple[float, str, tuple[str, str, str]]]] = {}
        for index, seed in enumerate(self.seed_accounts):
            identity = (seed.backend, seed.target_revision, seed.alias_identity)
            if identity in self._seed_by_identity:
                raise SchedulingRefused("duplicate seed immutable identity")
            self._seed_by_identity[identity] = seed
            self._seed_position[identity] = index
            for seed_id in seed.seed_ids:
                if seed_id in self._seed_by_id:
                    raise SchedulingRefused("seed ID occurs in multiple accounts")
                self._seed_by_id[seed_id] = seed
            if seed.boosted:
                self._boosted_by_backend[seed.backend] = seed
            elif (seed.attempts < self.config.seed_attempt_cap
                  and seed.charged_seconds < self.config.seed_charged_seconds_cap
                  and seed.valid_comparisons < self.config.seed_valid_comparison_cap):
                heapq.heappush(self._seed_queue.setdefault(seed.backend, []),
                               (seed.first_submitted_at, seed.seed_id, identity))
        self.successor_fences = list(source.successor_fences)
        self._receipts = list(source.receipts)
        self._records = list(source.accounted_receipts)
        self._receipt_by_id = {receipt.receipt_id: receipt for receipt in source.receipts}
        self._record_by_id = {record.receipt_id: record for record in source.accounted_receipts}
        self._claim_intervals: dict[str, list[tuple[float, float, str]]] = {}
        self._held_seconds = self._physical_seconds = self._memory_seconds = 0.0
        self._gpu_seconds: dict[str, float] = {}
        self._beneficiary_seconds: dict[str, float] = {}
        for receipt in source.receipts:
            self._index_receipt(receipt, replay=True)
        for intervals in self._claim_intervals.values():
            intervals.sort()
        self._seed_commitment = 0
        for seed in self.seed_accounts:
            self._seed_commitment ^= int(digest(seed.to_dict()), 16)
        self._outstanding_preview: SelectionPreview | None = None
        self._outstanding_accounting: AccountingPreview | None = None
        self._seed_preview_undo: list[tuple[str, Any]] | None = None

    def _operational_body(self) -> dict[str, Any]:
        return {
            "schema": OPERATIONAL_SCHEMA,
            "scheduler_id": self.scheduler_id,
            "config_digest": self.config_digest,
            "policy_digest": self.policy_digest,
            "accounting_epoch": self.accounting_epoch,
            "capacity": self.capacity.to_dict(),
            "capacity_digest": self.capacity_digest,
            "round_number": self.round_number,
            "frozen_frontier": list(self.frozen_frontier),
            "coverage_debt": dict(self.coverage_debt),
            "used_coverage": dict(self.used_coverage),
            "used_noncoverage": list(self.used_noncoverage),
            "skipped_noncoverage": list(self.skipped_noncoverage),
            "round_reservations": dict(self.round_reservations),
            "round_seed_used": self.round_seed_used,
            "deficits": dict(self.deficits),
            "seed_account_count": len(self.seed_accounts),
            "seed_commitment": f"{self._seed_commitment:064x}",
            "campaign_attempts": self.campaign_attempts,
            "campaign_charged_seconds": self.campaign_charged_seconds,
            "successor_fences": list(self.successor_fences),
            "issued_selection_digests": list(self.issued_selection_digests),
            "existing_stage_until": self.existing_stage_until,
            "accounting_view": self.accounting_view().to_dict(),
        }

    def operational_projection(self) -> OperationalProjection:
        """Return state whose size is independent of receipt-history length."""
        return OperationalProjection(self._operational_body())

    def _restore_operational(self, projection: OperationalProjection) -> None:
        body = projection.body
        if (body["scheduler_id"] != self.scheduler_id
                or body["config_digest"] != self.config_digest
                or body["policy_digest"] != self.policy_digest
                or body["accounting_epoch"] != self.accounting_epoch
                or body["capacity_digest"] != self.capacity_digest
                or _plain(body["capacity"]) != self.capacity.to_dict()
                or _plain(body["accounting_view"]) != self.accounting_view().to_dict()):
            raise SchedulingRefused("operational projection differs from scheduler authority")
        self.round_number = body["round_number"]
        self.frozen_frontier = list(body["frozen_frontier"])
        self.coverage_debt = dict(body["coverage_debt"])
        self.used_coverage = dict(body["used_coverage"])
        self.used_noncoverage = list(body["used_noncoverage"])
        self.skipped_noncoverage = list(body["skipped_noncoverage"])
        self.round_reservations = dict(body["round_reservations"])
        self.round_seed_used = body["round_seed_used"]
        self.deficits = dict(body["deficits"])
        if (body["seed_account_count"] != len(self.seed_accounts)
                or body["seed_commitment"] != f"{self._seed_commitment:064x}"):
            raise SchedulingRefused("operational seed commitment differs")
        self.campaign_attempts = body["campaign_attempts"]
        self.campaign_charged_seconds = body["campaign_charged_seconds"]
        self.successor_fences = list(body["successor_fences"])
        self.issued_selection_digests = tuple(body["issued_selection_digests"])
        self.existing_stage_until = body["existing_stage_until"]

    def preview_selection(self, proposals: Sequence[StageProposal | Mapping[str, Any]], *,
                          now: float,
                          outages: Sequence[Outage | Mapping[str, Any]] = ()) -> SelectionPreview:
        """Preview one issue without mutating the persistent engine."""
        normalized = tuple(_normalize(item, StageProposal) for item in proposals)
        for proposal in normalized:
            if proposal.seed_id is not None:
                self._seed_for(proposal)
        prior = self.operational_projection()
        try:
            selection = self.select_stage(normalized, now=now, outages=outages,
                                          _register_new_seeds=False)
            after = self.operational_projection()
        finally:
            self._restore_operational(prior)
        body = {"schema": SELECTION_PREVIEW_SCHEMA,
                "prior_digest": prior.projection_digest,
                "selection": selection.to_dict(),
                "after_digest": after.projection_digest}
        preview = SelectionPreview(prior, selection, after, digest(body))
        self._outstanding_preview = preview
        return preview

    def apply_preview(self, preview: SelectionPreview) -> bool:
        """Apply an exact locally-derived preview, refusing stale or forged state."""
        if not isinstance(preview, SelectionPreview) or preview is not self._outstanding_preview:
            raise SchedulingRefused("selection preview is not locally derived")
        body = {"schema": preview.schema,
                "prior_digest": preview.prior.projection_digest,
                "selection": preview.selection.to_dict(),
                "after_digest": preview.after.projection_digest}
        if preview.schema != SELECTION_PREVIEW_SCHEMA or digest(body) != preview.preview_digest:
            raise SchedulingRefused("selection preview integrity differs")
        current = self.operational_projection().projection_digest
        if current == preview.after.projection_digest:
            return False
        if current != preview.prior.projection_digest:
            raise SchedulingRefused("selection preview is stale")
        self._restore_operational(preview.after)
        return True

    def preview_accounting(self, selection: Selection | Mapping[str, Any],
                           receipt: HeldClaimReceipt | Mapping[str, Any], *,
                           outcome: str) -> AccountingPreview:
        """Validate one settlement and derive its compact result without retaining it."""
        selection = _normalize(selection, Selection)
        receipt = _normalize(receipt, HeldClaimReceipt)
        outcome = _enum(outcome, OUTCOMES, "stage outcome")
        prior = self.operational_projection()
        gpu_prior = {key: self._gpu_seconds.get(key) for key in receipt.gpu_device_ids}
        beneficiary_prior = {key: self._beneficiary_seconds.get(key)
                             for key in receipt.beneficiary_shares}
        scalar_prior = (self._held_seconds, self._physical_seconds, self._memory_seconds)
        receipt_count = len(self._receipts)
        record_count = len(self._records)
        prior_receipt = self._receipt_by_id.get(receipt.receipt_id)
        prior_record = self._record_by_id.get(receipt.receipt_id)
        seed_prior = (self._seed_for(selection.proposal)
                      if selection.proposal is not None
                      and selection.proposal.seed_id is not None else None)
        seed_backend = seed_prior.backend if seed_prior is not None else None
        queue_preexisting = seed_backend in self._seed_queue if seed_backend is not None else False
        if self._seed_preview_undo is not None:
            raise SchedulingRefused("nested accounting preview is unsupported")
        self._seed_preview_undo = []
        try:
            changed = self.account_stage(selection, receipt, outcome=outcome)
            if not changed:
                raise SchedulingRefused("accounting preview receipt was already applied")
            after = self.operational_projection()
        finally:
            seed_undo, self._seed_preview_undo = self._seed_preview_undo, None
            appended = (prior_receipt is None and prior_record is None
                        and len(self._receipts) == receipt_count + 1
                        and len(self._records) == record_count + 1
                        and self._receipts[-1].receipt_id == receipt.receipt_id
                        and self._records[-1].receipt_id == receipt.receipt_id)
            if appended:
                self._receipts.pop()
                self._records.pop()
                self._receipt_by_id.pop(receipt.receipt_id, None)
                self._record_by_id.pop(receipt.receipt_id, None)
                point = (receipt.started_at, receipt.ended_at, receipt.receipt_id)
                for claim in (*receipt.physical_claim_ids, *receipt.gpu_device_ids):
                    intervals = self._claim_intervals.get(claim, [])
                    try:
                        intervals.remove(point)
                    except ValueError:
                        pass
                    if not intervals:
                        self._claim_intervals.pop(claim, None)
            self._held_seconds, self._physical_seconds, self._memory_seconds = scalar_prior
            for key, value in gpu_prior.items():
                if value is None:
                    self._gpu_seconds.pop(key, None)
                else:
                    self._gpu_seconds[key] = value
            for key, value in beneficiary_prior.items():
                if value is None:
                    self._beneficiary_seconds.pop(key, None)
                else:
                    self._beneficiary_seconds[key] = value
            for kind, item in reversed(seed_undo or []):
                if kind == "replace":
                    original, updated = item
                    identity = (updated.backend, updated.target_revision,
                                updated.alias_identity)
                    current = self._seed_by_identity[identity]
                    if current != updated:
                        raise SchedulingRefused(
                            "seed changed outside the accounting preview")
                    self._replace_seed(current, original)
                else:
                    backend, queued = item
                    heapq.heappush(self._seed_queue.setdefault(backend, []), queued)
            if seed_backend is not None and not queue_preexisting \
                    and not self._seed_queue.get(seed_backend):
                self._seed_queue.pop(seed_backend, None)
            self._restore_operational(prior)
        body = {"schema": ACCOUNTING_PREVIEW_SCHEMA,
                "prior_digest": prior.projection_digest,
                "selection": selection.to_dict(), "receipt": receipt.to_dict(),
                "outcome": outcome, "after_digest": after.projection_digest}
        preview = AccountingPreview(prior, selection, receipt, outcome, after, digest(body))
        self._outstanding_accounting = preview
        return preview

    def apply_accounting_preview(self, preview: AccountingPreview) -> bool:
        if (not isinstance(preview, AccountingPreview)
                or preview is not self._outstanding_accounting):
            raise SchedulingRefused("accounting preview is not locally derived")
        body = {"schema": preview.schema,
                "prior_digest": preview.prior.projection_digest,
                "selection": preview.selection.to_dict(),
                "receipt": preview.receipt.to_dict(), "outcome": preview.outcome,
                "after_digest": preview.after.projection_digest}
        if preview.schema != ACCOUNTING_PREVIEW_SCHEMA or digest(body) != preview.preview_digest:
            raise SchedulingRefused("accounting preview integrity differs")
        current = self.operational_projection().projection_digest
        if current == preview.after.projection_digest:
            prior = self._record_by_id.get(preview.receipt.receipt_id)
            if prior is None:
                raise SchedulingRefused("accounting retry lacks its durable receipt")
            return False
        if current != preview.prior.projection_digest:
            raise SchedulingRefused("accounting preview is stale")
        changed = self.account_stage(preview.selection, preview.receipt, outcome=preview.outcome)
        if self.operational_projection().projection_digest != preview.after.projection_digest:
            raise SchedulingRefused("accounting application differs from preview")
        return changed

    def _seed_identity(self, proposal: StageProposal) -> tuple[str, str, str]:
        return proposal.backend, proposal.target_revision, proposal.alias_identity

    def _seed_for(self, proposal: StageProposal) -> SeedAccount:
        identity = self._seed_identity(proposal)
        by_id = self._seed_by_id.get(proposal.seed_id)
        if by_id is not None:
            if (by_id.backend, by_id.target_revision, by_id.alias_identity) != identity:
                raise SchedulingRefused("seed ID reused for a different immutable identity")
            return by_id
        seed = self._seed_by_identity.get(identity)
        if seed is not None:
            return seed
        raise SchedulingRefused("seed proposal was not registered in scheduler state")

    def _replace_seed(self, original: SeedAccount, updated: SeedAccount) -> None:
        identity = (original.backend, original.target_revision, original.alias_identity)
        if identity != (updated.backend, updated.target_revision, updated.alias_identity):
            raise SchedulingRefused("seed immutable identity cannot change")
        if self._seed_preview_undo is not None:
            self._seed_preview_undo.append(("replace", (original, updated)))
        self._seed_commitment ^= (int(digest(original.to_dict()), 16)
                                  ^ int(digest(updated.to_dict()), 16))
        self.seed_accounts[self._seed_position[identity]] = updated
        self._seed_by_identity[identity] = updated
        for seed_id in updated.seed_ids:
            self._seed_by_id[seed_id] = updated
        if updated.boosted:
            self._boosted_by_backend[updated.backend] = updated
        elif self._boosted_by_backend.get(updated.backend) == original:
            self._boosted_by_backend.pop(updated.backend, None)

    def _register_seeds(self, proposals: Sequence[StageProposal]) -> None:
        for proposal in sorted((item for item in proposals if item.seed_id is not None),
                               key=lambda item: (item.submitted_at, item.proposal_id)):
            identity = self._seed_identity(proposal)
            matched = self._seed_by_id.get(proposal.seed_id)
            if matched is not None and (
                    matched.backend, matched.target_revision, matched.alias_identity) != identity:
                raise SchedulingRefused("seed ID reused for a different immutable identity")
            if matched is None:
                matched = self._seed_by_identity.get(identity)
            if matched is not None:
                if proposal.seed_id not in matched.seed_ids:
                    self._replace_seed(matched, SeedAccount(
                        seed_id=matched.seed_id, backend=matched.backend,
                        target_revision=matched.target_revision,
                        alias_identity=matched.alias_identity,
                        seed_ids=matched.seed_ids + (proposal.seed_id,),
                        first_submitted_at=matched.first_submitted_at,
                        attempts=matched.attempts, charged_seconds=matched.charged_seconds,
                        valid_comparisons=matched.valid_comparisons, boosted=matched.boosted))
                continue
            boosted = proposal.backend not in self._boosted_by_backend
            seed = SeedAccount(
                seed_id=proposal.seed_id, backend=proposal.backend,
                target_revision=proposal.target_revision, alias_identity=proposal.alias_identity,
                seed_ids=(proposal.seed_id,), first_submitted_at=proposal.submitted_at,
                boosted=boosted)
            self._seed_position[identity] = len(self.seed_accounts)
            self.seed_accounts.append(seed)
            self._seed_commitment ^= int(digest(seed.to_dict()), 16)
            self._seed_by_identity[identity] = seed
            self._seed_by_id[proposal.seed_id] = seed
            if boosted:
                self._boosted_by_backend[proposal.backend] = seed
            else:
                heapq.heappush(self._seed_queue.setdefault(proposal.backend, []),
                               (seed.first_submitted_at, seed.seed_id, identity))

    def _seed_available(self, proposal: StageProposal) -> bool:
        seed = self._seed_for(proposal)
        return (seed.attempts < self.config.seed_attempt_cap
                and seed.charged_seconds < self.config.seed_charged_seconds_cap)

    def _promote_seed(self, backend: str) -> None:
        if backend in self._boosted_by_backend:
            return
        queue = self._seed_queue.setdefault(backend, [])
        while queue:
            queued = heapq.heappop(queue)
            if self._seed_preview_undo is not None:
                self._seed_preview_undo.append(("queue_pop", (backend, queued)))
            _submitted, _seed_id, identity = queued
            seed = self._seed_by_identity[identity]
            if (seed.boosted or seed.attempts >= self.config.seed_attempt_cap
                    or seed.charged_seconds >= self.config.seed_charged_seconds_cap
                    or seed.valid_comparisons >= self.config.seed_valid_comparison_cap):
                continue
            self._replace_seed(seed, SeedAccount(**(seed.to_dict() | {"boosted": True})))
            return

    def _start_round(self, proposals: Sequence[StageProposal]) -> None:
        self.round_number += 1
        self.frozen_frontier = sorted({proposal.frontier_id for proposal in proposals
                                       if proposal.production_frontier and proposal.eligible})
        self.used_coverage = {}
        self.used_noncoverage = []
        self.skipped_noncoverage = []
        self.round_reservations = {
            str(slot): kind for kind, slots in self.config.reservation_slots.items()
            for slot in slots
        }
        self.round_seed_used = False
        self.issued_selection_digests = ()

    def _coverage_done(self) -> bool:
        return set(self.used_coverage) == set(self.frozen_frontier)

    def _next_noncoverage_slot(self) -> int:
        return len(self.used_noncoverage) + len(self.skipped_noncoverage)

    def _proposal_issue(self, proposal: StageProposal) -> str | None:
        if proposal.estimated_duration_seconds > self.config.max_stage_seconds:
            return "is oversized beyond D; safe_chunking_declared is not a bounded chunk descriptor"
        vector = proposal.estimated_claims
        if (vector.physical_region_fraction > self.capacity.physical_region_fraction
                or not set(vector.gpu_devices).issubset(self.capacity.gpu_devices)
                or vector.memory_reservation_bytes > self.capacity.memory_reservation_bytes):
            return "has infeasible resource demand"
        return None

    def _active_outages(self, proposal: StageProposal, outages: Sequence[Outage]) -> tuple[Outage, ...]:
        return tuple(outage for outage in outages
                     if outage.ended_at is None and _outage_applies(outage, proposal))

    def _bound(self) -> float:
        return (len(self.frozen_frontier) + self.config.noncoverage_slots) \
            * self.config.max_stage_seconds

    def _selection(self, status: str, reasons: Sequence[str], proposal: StageProposal | None,
                   slot_kind: str | None, slot_index: int | None, outages: Sequence[Outage],
                   now: float) -> Selection:
        remaining = (0.0 if self.existing_stage_until is None
                     else max(0.0, self.existing_stage_until - now))
        selected = Selection(
            status=status, reasons=tuple(dict.fromkeys(reasons)), proposal=proposal,
            slot_kind=slot_kind, slot_index=slot_index, round_number=self.round_number,
            scheduler_id=self.scheduler_id, config_digest=self.config_digest,
            accounting_epoch=self.accounting_epoch, capacity_digest=self.capacity_digest,
            proposal_digest=None if proposal is None else proposal.digest,
            service_bound_seconds=self._bound(), outage_seconds=_outage_seconds(outages, now),
            existing_stage_seconds=remaining)
        if status == "selected" and selected.digest not in self.issued_selection_digests:
            self.issued_selection_digests = self.issued_selection_digests + (selected.digest,)
        return selected

    def _has_required_noncoverage(self, proposals: Sequence[StageProposal],
                                  outages: Sequence[Outage]) -> bool:
        slot = self._next_noncoverage_slot()
        if slot >= self.config.noncoverage_slots:
            return False
        if str(slot) in self.round_reservations:
            return True
        eligible = [proposal for proposal in proposals
                    if proposal.eligible and not proposal.production_frontier
                    and (proposal.seed_id is None or self._seed_available(proposal))]
        if any(proposal.seed_id is not None and self._seed_available(proposal)
               and self._proposal_issue(proposal) is None
               and not self._active_outages(proposal, outages)
               for proposal in eligible) and not self.round_seed_used:
            return True
        return any(self._proposal_issue(proposal) is None
                   and not self._active_outages(proposal, outages) for proposal in eligible)

    def select_stage(self, proposals: Sequence[StageProposal | Mapping[str, Any]], *, now: float,
                     outages: Sequence[Outage | Mapping[str, Any]] = (),
                     _register_new_seeds: bool = True) -> Selection:
        proposals = tuple(_normalize(proposal, StageProposal) for proposal in proposals)
        outages = tuple(_normalize(outage, Outage) for outage in outages)
        now = _number(now, "now")
        ids = [proposal.proposal_id for proposal in proposals]
        if len(ids) != len(set(ids)):
            raise SchedulingRefused("duplicate proposal IDs")
        outage_by_id: dict[str, Outage] = {}
        for outage in outages:
            prior = outage_by_id.get(outage.outage_id)
            if prior is not None and prior.digest != outage.digest:
                raise SchedulingRefused("outage ID reused for different content")
            outage_by_id[outage.outage_id] = outage
        outages = tuple(outage_by_id.values())
        if any(outage.started_at > now
               or (outage.ended_at is not None and outage.ended_at > now)
               for outage in outages):
            raise SchedulingRefused("outage event is later than the selection observation time")
        if _register_new_seeds:
            self._register_seeds(proposals)
        eligible = tuple(proposal for proposal in proposals if proposal.eligible)
        if self.round_number == 0:
            self._start_round(eligible)
        elif self._coverage_done() and (
                self._next_noncoverage_slot() >= self.config.noncoverage_slots
                or not self._has_required_noncoverage(eligible, outages)):
            if eligible:
                self._start_round(eligible)
            else:
                return self._selection("complete", ("no eligible work remains",), None,
                                       None, None, outages, now)
        if self.successor_fences:
            return self._selection("refused", tuple(self.successor_fences), None,
                                   None, None, outages, now)
        if self.issued_selection_digests:
            return self._selection("waiting", ("issued selection awaits settlement",), None,
                                   None, None, outages, now)
        if self.campaign_attempts >= self.config.campaign_attempt_cap:
            return self._selection("refused", ("campaign attempt budget exhausted",), None,
                                   None, None, outages, now)
        if self.campaign_charged_seconds >= self.config.campaign_charged_seconds_cap:
            return self._selection("refused", ("campaign charged-time budget exhausted",), None,
                                   None, None, outages, now)
        if self.existing_stage_until is not None and self.existing_stage_until > now:
            return self._selection("waiting", ("existing admitted bounded stage is running",),
                                   None, None, None, outages, now)
        unavailable: list[str] = []
        permanent = False
        for index, frontier in enumerate(self.frozen_frontier):
            if frontier in self.used_coverage:
                continue
            candidates = [proposal for proposal in eligible if proposal.frontier_id == frontier]
            ready = []
            for proposal in candidates:
                issue = self._proposal_issue(proposal)
                blocked = self._active_outages(proposal, outages)
                if issue is not None:
                    unavailable.append(f"frozen frontier {frontier} {issue}")
                    permanent = True
                elif blocked:
                    unavailable.extend(f"{item.kind} outage for {frontier}: {item.reason}"
                                       for item in blocked)
                else:
                    ready.append(proposal)
            if ready:
                chosen = min(ready, key=lambda item: (item.submitted_at, item.proposal_id))
                reasons = ["frozen production coverage"]
                if unavailable:
                    reasons.append("other frozen frontiers are temporarily unavailable")
                return self._selection("selected", reasons, chosen, "coverage", index,
                                       outages, now)
            if not candidates:
                unavailable.append(f"frozen frontier {frontier} is temporarily ineligible")
        coverage_unavailable = tuple(unavailable) if not self._coverage_done() else ()
        continuously_ready = tuple(
            proposal for proposal in eligible if proposal.production_frontier
            and self._proposal_issue(proposal) is None
            and not self._active_outages(proposal, outages))
        if (coverage_unavailable
                and (self._next_noncoverage_slot() >= self.config.noncoverage_slots
                     or not self._has_required_noncoverage(eligible, outages))
                and continuously_ready):
            debt_reason = "; ".join(coverage_unavailable)
            for frontier in self.frozen_frontier:
                if frontier not in self.used_coverage:
                    self.coverage_debt.setdefault(frontier, debt_reason)
            self._start_round(continuously_ready)
            chosen = min(continuously_ready,
                         key=lambda item: (item.submitted_at, item.proposal_id))
            index = self.frozen_frontier.index(chosen.frontier_id)
            return self._selection(
                "selected",
                ("new round after suspending unavailable coverage debt", debt_reason),
                chosen, "coverage", index, outages, now)
        if permanent and coverage_unavailable:
            return self._selection("refused", coverage_unavailable, None,
                                   "coverage", None, outages, now)
        slot = self._next_noncoverage_slot()
        seeds = [proposal for proposal in eligible if proposal.seed_id is not None
                 and self._seed_available(proposal)]
        if seeds and self.config.noncoverage_slots < 1:
            return self._selection("refused", ("eligible seed requires K>=1",), None,
                                   None, None, outages, now)
        if slot >= self.config.noncoverage_slots:
            if coverage_unavailable:
                return self._selection("waiting", coverage_unavailable, None,
                                       "coverage", None, outages, now)
            return self._selection("complete", ("coverage round is complete",), None,
                                   None, None, outages, now)
        future_full_slots = sorted(
            int(index) for index, kind in self.round_reservations.items()
            if kind == "full_region" and int(index) >= slot)
        if future_full_slots and future_full_slots[0] > slot:
            # V1 has no trusted compatibility consumer. Preserve the due exclusive
            # reservation by leaving only earlier unreserved backfill unused.
            # A preceding seed or other frozen reservation keeps its exact slot.
            due = future_full_slots[0]
            prior_reserved = [index for index in range(slot, due)
                              if str(index) in self.round_reservations]
            skip_until = min(prior_reserved, default=due)
            self.skipped_noncoverage.extend(range(slot, skip_until))
            slot = skip_until
        reservation = self.round_reservations.get(str(slot))
        if seeds and not self.round_seed_used and reservation not in {None, "seed"}:
            future = any(index > slot and str(index) not in self.round_reservations
                         for index in range(self.config.noncoverage_slots))
            if not future:
                return self._selection(
                    "refused", ("reserved slots leave no required seed opportunity",), None,
                    None, slot, outages, now)
        if reservation == "seed":
            candidates = seeds
        elif reservation is not None:
            candidates = [proposal for proposal in eligible
                          if proposal.reservation_kind == reservation
                          or (reservation == "full_region" and proposal.full_region)]
        elif seeds and not self.round_seed_used:
            candidates, reservation = seeds, "seed"
        else:
            candidates = [proposal for proposal in eligible if not proposal.production_frontier
                          and (proposal.seed_id is None or self._seed_available(proposal))]
        ready = []
        unavailable = []
        permanent = False
        for proposal in candidates:
            issue = self._proposal_issue(proposal)
            blocked = self._active_outages(proposal, outages)
            if issue is not None:
                unavailable.append(f"proposal {proposal.proposal_id} {issue}")
                permanent = True
            elif blocked:
                unavailable.extend(f"{item.kind} outage for {proposal.proposal_id}: {item.reason}"
                                   for item in blocked)
            else:
                ready.append(proposal)
        if not ready:
            if unavailable:
                return self._selection("refused" if permanent else "waiting", unavailable, None,
                                       reservation or "noncoverage", slot, outages, now)
            if coverage_unavailable:
                return self._selection("waiting", coverage_unavailable, None,
                                       "coverage", None, outages, now)
            reason = (f"reserved {reservation} opportunity is due" if reservation is not None
                      else "no eligible noncoverage proposal")
            return self._selection("waiting", (reason,), None,
                                   reservation or "noncoverage", slot, outages, now)
        if reservation == "seed":
            chosen = min(ready, key=lambda proposal: (
                not self._seed_for(proposal).boosted,
                self._seed_for(proposal).first_submitted_at,
                self._seed_for(proposal).seed_id, proposal.proposal_id))
        else:
            chosen = min(ready, key=lambda proposal: (
                self.deficits.get(proposal.backend, 0.0)
                + _dominant_estimate(self.capacity, proposal)
                / (self.config.seed_weight if proposal.seed_id is not None
                   and self._seed_for(proposal).boosted else self.config.normal_weight),
                proposal.submitted_at, proposal.proposal_id))
        reasons = [f"{reservation or 'weighted-deficit'} opportunity"]
        if coverage_unavailable:
            reasons.append("frozen coverage is temporarily unavailable")
        return self._selection("selected", reasons,
                               chosen, reservation or "noncoverage", slot, outages, now)

    def _check_new_intervals(self, receipt: HeldClaimReceipt) -> None:
        for claim in (*receipt.physical_claim_ids, *receipt.gpu_device_ids):
            intervals = self._claim_intervals.get(claim, [])
            point = (receipt.started_at, receipt.ended_at, receipt.receipt_id)
            index = bisect_left(intervals, point)
            neighbors = intervals[max(0, index - 1):index + 1]
            for start, end, receipt_id in neighbors:
                if receipt.started_at < end and start < receipt.ended_at:
                    raise SchedulingRefused(
                        f"overlapping reuse of physical claim {claim}: "
                        f"{receipt_id}/{receipt.receipt_id}")

    def _index_receipt(self, receipt: HeldClaimReceipt, *, replay: bool = False) -> None:
        duration = receipt.ended_at - receipt.started_at
        self._held_seconds += duration
        self._physical_seconds += duration * receipt.physical_region_fraction
        self._memory_seconds += duration * receipt.memory_reservation_bytes
        for device in receipt.gpu_device_ids:
            self._gpu_seconds[device] = self._gpu_seconds.get(device, 0.0) + duration
        for beneficiary, share in receipt.beneficiary_shares.items():
            self._beneficiary_seconds[beneficiary] = (
                self._beneficiary_seconds.get(beneficiary, 0.0) + duration * share)
        for claim in (*receipt.physical_claim_ids, *receipt.gpu_device_ids):
            intervals = self._claim_intervals.setdefault(claim, [])
            point = (receipt.started_at, receipt.ended_at, receipt.receipt_id)
            if replay:
                intervals.append(point)
            elif not intervals or receipt.started_at >= intervals[-1][1]:
                intervals.append(point)
            else:
                insort(intervals, point)

    def account_stage(self, selection: Selection | Mapping[str, Any],
                      receipt: HeldClaimReceipt | Mapping[str, Any], *, outcome: str) -> bool:
        selection = _normalize(selection, Selection)
        receipt = _normalize(receipt, HeldClaimReceipt)
        outcome = _enum(outcome, OUTCOMES, "stage outcome")
        record = AccountedReceipt(receipt_id=receipt.receipt_id,
                                  receipt_digest=receipt.digest,
                                  selection_digest=selection.digest, outcome=outcome)
        prior = self._record_by_id.get(receipt.receipt_id)
        if prior is not None:
            if prior != record:
                raise SchedulingRefused("receipt ID conflicts with its original selection or outcome")
            return False
        if (selection.status != "selected" or selection.proposal is None
                or selection.digest not in self.issued_selection_digests):
            raise SchedulingRefused("selection was not issued by this scheduler round")
        if (selection.scheduler_id != self.scheduler_id
                or selection.config_digest != self.config_digest
                or selection.accounting_epoch != self.accounting_epoch
                or selection.capacity_digest != self.capacity_digest
                or selection.round_number != self.round_number
                or selection.service_bound_seconds != self._bound()):
            raise SchedulingRefused("selection does not match scheduler/config/epoch/round")
        proposal = selection.proposal
        if (receipt.proposal_id != proposal.proposal_id
                or receipt.stage_class != proposal.stage_class
                or receipt.backend != proposal.backend):
            raise SchedulingRefused("receipt does not match selected proposal identity")
        if selection.slot_kind == "coverage":
            frontier = proposal.frontier_id
            if (frontier is None or selection.slot_index is None
                    or selection.slot_index >= len(self.frozen_frontier)
                    or self.frozen_frontier[selection.slot_index] != frontier
                    or frontier in self.used_coverage):
                raise SchedulingRefused("coverage selection does not match its unused frozen slot")
        else:
            if selection.slot_index != self._next_noncoverage_slot():
                raise SchedulingRefused("noncoverage slot is not the next durable slot")
            reservation = self.round_reservations.get(str(selection.slot_index))
            if reservation is not None and selection.slot_kind != reservation:
                raise SchedulingRefused("selection does not match the frozen slot reservation")
            if reservation == "seed" and proposal.seed_id is None:
                raise SchedulingRefused("seed reservation requires a seed proposal")
            if (reservation == "full_region" and not proposal.full_region) or (
                    reservation not in {None, "seed", "full_region"}
                    and proposal.reservation_kind != reservation):
                raise SchedulingRefused("proposal does not satisfy the frozen slot reservation")
            if reservation is None and selection.slot_kind not in {"seed", "noncoverage"}:
                raise SchedulingRefused("unreserved slot cannot be relabelled as reserved")
        self._check_new_intervals(receipt)
        duration = receipt.ended_at - receipt.started_at
        violations = []
        if duration > self.config.max_stage_seconds:
            violations.append("actual held interval exceeded configured stage-plus-teardown D")
        if (receipt.physical_region_fraction > self.capacity.physical_region_fraction
                or not set(receipt.gpu_device_ids).issubset(self.capacity.gpu_devices)
                or receipt.memory_reservation_bytes > self.capacity.memory_reservation_bytes):
            violations.append("actual held claims exceeded configured capacity")
        if selection.slot_kind == "coverage":
            self.used_coverage[proposal.frontier_id] = proposal.proposal_id
            self.coverage_debt.pop(proposal.frontier_id, None)
        else:
            self.used_noncoverage.append(proposal.proposal_id)
        terms = []
        if self.capacity.physical_region_fraction:
            terms.append(duration * receipt.physical_region_fraction
                         / self.capacity.physical_region_fraction)
        if self.capacity.gpu_devices:
            terms.append(duration * len(receipt.gpu_device_ids) / len(self.capacity.gpu_devices))
        if self.capacity.memory_reservation_bytes:
            terms.append(duration * receipt.memory_reservation_bytes
                         / self.capacity.memory_reservation_bytes)
        weight = self.config.normal_weight
        if proposal.seed_id is not None:
            seed = self._seed_for(proposal)
            weight = self.config.seed_weight if seed.boosted else self.config.normal_weight
            comparisons = seed.valid_comparisons + (outcome == "valid_comparison")
            boosted = (seed.boosted and seed.attempts + 1 < self.config.seed_attempt_cap
                       and seed.charged_seconds + duration < self.config.seed_charged_seconds_cap
                       and comparisons < self.config.seed_valid_comparison_cap)
            updated = SeedAccount(
                seed_id=seed.seed_id, backend=seed.backend,
                target_revision=seed.target_revision, alias_identity=seed.alias_identity,
                seed_ids=seed.seed_ids, first_submitted_at=seed.first_submitted_at,
                attempts=seed.attempts + 1, charged_seconds=seed.charged_seconds + duration,
                valid_comparisons=comparisons, boosted=boosted)
            self._replace_seed(seed, updated)
            if not boosted:
                self._promote_seed(seed.backend)
        self.round_seed_used = self.round_seed_used or proposal.seed_id is not None
        self.deficits[proposal.backend] = (self.deficits.get(proposal.backend, 0.0)
                                           + max(terms, default=0.0) / weight)
        self.campaign_attempts += 1
        self.campaign_charged_seconds += duration
        self._receipts.append(receipt)
        self._records.append(record)
        self._receipt_by_id[receipt.receipt_id] = receipt
        self._record_by_id[receipt.receipt_id] = record
        self._index_receipt(receipt)
        self.successor_fences.extend(item for item in violations
                                     if item not in self.successor_fences)
        self.issued_selection_digests = ()
        return True

    def change_capacity(self, new_config: SchedulerConfig | Mapping[str, Any]) -> bool:
        new_config = _normalize(new_config, SchedulerConfig)
        if new_config.policy_digest != self.policy_digest:
            raise SchedulingRefused("capacity epoch cannot change scheduling policy")
        if new_config.capacity.digest == self.capacity_digest:
            if new_config.digest != self.config_digest:
                raise SchedulingRefused("new config digest changed without a capacity change")
            return False
        self.config = new_config
        self.config_digest = new_config.digest
        self.capacity = new_config.capacity
        self.capacity_digest = new_config.capacity.digest
        self.accounting_epoch += 1
        self.issued_selection_digests = ()
        return True

    def accounting_view(self) -> AccountingView:
        body = {"schema": ACCOUNTING_SCHEMA, "receipt_count": len(self._receipts),
                "held_seconds": self._held_seconds,
                "physical_region_seconds": self._physical_seconds,
                "gpu_device_seconds": dict(self._gpu_seconds),
                "memory_byte_seconds": self._memory_seconds,
                "beneficiary_seconds": dict(self._beneficiary_seconds)}
        return AccountingView(**{key: value for key, value in body.items() if key != "schema"},
                              view_digest=digest(body))

    def export_state(self) -> SchedulerState:
        return SchedulerState(
            scheduler_id=self.scheduler_id, config_digest=self.config_digest,
            policy_digest=self.policy_digest, accounting_epoch=self.accounting_epoch,
            capacity=self.capacity, capacity_digest=self.capacity_digest,
            round_number=self.round_number, frozen_frontier=tuple(self.frozen_frontier),
            coverage_debt=self.coverage_debt, used_coverage=self.used_coverage,
            used_noncoverage=tuple(self.used_noncoverage),
            skipped_noncoverage=tuple(self.skipped_noncoverage),
            round_reservations=self.round_reservations, round_seed_used=self.round_seed_used,
            deficits=self.deficits, seed_accounts=tuple(self.seed_accounts),
            campaign_attempts=self.campaign_attempts,
            campaign_charged_seconds=self.campaign_charged_seconds,
            receipts=tuple(self._receipts), accounted_receipts=tuple(self._records),
            successor_fences=tuple(self.successor_fences),
            issued_selection_digests=self.issued_selection_digests,
            existing_stage_until=self.existing_stage_until)


def select_stage(config: SchedulerConfig | Mapping[str, Any],
                 state: SchedulerState | Mapping[str, Any],
                 proposals: Sequence[StageProposal | Mapping[str, Any]], *, now: float,
                 outages: Sequence[Outage | Mapping[str, Any]] = ()) -> tuple[SchedulerState, Selection]:
    """Full-validation pure reference transition; use SchedulerEngine operationally."""
    engine = SchedulerEngine(config, state)
    selected = engine.select_stage(proposals, now=now, outages=outages)
    return engine.export_state(), selected


def account_stage(config: SchedulerConfig | Mapping[str, Any],
                  state: SchedulerState | Mapping[str, Any], selection: Selection,
                  receipt: HeldClaimReceipt | Mapping[str, Any], *, outcome: str) -> SchedulerState:
    """Full-validation pure reference transition; use SchedulerEngine operationally."""
    engine = SchedulerEngine(config, state)
    engine.account_stage(selection, receipt, outcome=outcome)
    return engine.export_state()


def change_capacity(state: SchedulerState | Mapping[str, Any],
                    new_config: SchedulerConfig | Mapping[str, Any]) -> SchedulerState:
    """Start an epoch under a config whose only semantic change is capacity."""
    source = _normalize(state, SchedulerState)
    config = _normalize(new_config, SchedulerConfig)
    if config.policy_digest != source.policy_digest:
        raise SchedulingRefused("capacity epoch cannot change scheduling policy")
    if config.capacity.digest == source.capacity_digest:
        if config.digest != source.config_digest:
            raise SchedulingRefused("new config digest changed without a capacity change")
        return source
    return SchedulerState.from_dict(source.to_dict() | {
        "config_digest": config.digest,
        "accounting_epoch": source.accounting_epoch + 1,
        "capacity": config.capacity.to_dict(),
        "capacity_digest": config.capacity.digest,
        "issued_selection_digests": [],
    })


def adaptive_weights(_config: SchedulerConfig | Mapping[str, Any],
                     _state: SchedulerState | Mapping[str, Any],
                     _records: Sequence[Mapping[str, Any]] = (), **_kwargs: Any) -> dict[str, Any]:
    """V1 is deliberately fixed; verified batch adaptation remains unwired."""
    return {"status": "adaptation_not_configured", "weights": {}}


__all__ = [
    "ACCOUNTING_SCHEMA", "CONFIG_SCHEMA", "OUTAGE_SCHEMA", "PROPOSAL_SCHEMA",
    "RECEIPT_RECORD_SCHEMA", "RECEIPT_SCHEMA", "SEED_SCHEMA", "SELECTION_SCHEMA", "STATE_SCHEMA",
    "VECTOR_SCHEMA", "AccountedReceipt", "AccountingView", "HeldClaimReceipt", "Outage",
    "ResourceVector", "SchedulerConfig", "SchedulerEngine", "SchedulerState",
    "SchedulingRefused", "SeedAccount", "Selection", "StageProposal",
    "account_stage", "adaptive_weights", "change_capacity", "charge_receipts", "digest",
    "initial_state", "select_stage",
]
