"""Pure, fail-closed GPU model-load admission policy.

This module performs no I/O except loading a sealed policy corpus.  It does not
acquire locks, sample hardware, launch a model, or trust actor prose.  Runtime
code supplies a typed request whose telemetry has already been captured; this
module returns the one deterministic admission decision those facts permit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .. import schemas
from .split_runtime_verifier import HotResidencyIdentity


POLICY_SCHEMA = "epyc.autokernel.gpu_load_admission_policy.v2"
DECISION_SCHEMA = "epyc.autokernel.gpu_load_admission_decision.v1"
MAX_POLICY_BYTES = 512 * 1024
MAX_PROFILES = 64
MAX_EXAMPLES = 256
MAX_FACTS = 48
MAX_TEXT = 4096
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
MODES = frozenset({"hot_resident", "cold_overlap", "cold_serialized"})


class AdmissionPolicyError(RuntimeError):
    pass


def _content_hash(value: Mapping[str, Any]) -> str:
    return schemas.content_hash(value)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise AdmissionPolicyError(f"{label} must be a SHA-256 digest")
    return value


def _text(value: object, label: str, *, identifier: bool = False) -> str:
    pattern = IDENTIFIER if identifier else None
    if (not isinstance(value, str) or not value or len(value) > MAX_TEXT
            or "\x00" in value or pattern is not None and not pattern.fullmatch(value)):
        raise AdmissionPolicyError(f"{label} is malformed or unbounded")
    return value


def _integer(value: object, label: str, *, minimum: int = 0,
             maximum: int = 1 << 60) -> int:
    if (isinstance(value, bool) or not isinstance(value, int)
            or not minimum <= value <= maximum):
        raise AdmissionPolicyError(f"{label} is outside its reviewed bounds")
    return value


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise AdmissionPolicyError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _freeze(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise AdmissionPolicyError("policy nesting exceeds the reviewed bound")
    if isinstance(value, Mapping):
        if len(value) > MAX_FACTS:
            raise AdmissionPolicyError("policy mapping exceeds the reviewed bound")
        return MappingProxyType({
            _text(key, "policy key"): _freeze(item, depth=depth + 1)
            for key, item in value.items()
        })
    if isinstance(value, list):
        if len(value) > MAX_EXAMPLES:
            raise AdmissionPolicyError("policy list exceeds the reviewed bound")
        return tuple(_freeze(item, depth=depth + 1) for item in value)
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (value != value or abs(value) == float("inf")):
            raise AdmissionPolicyError("policy contains a non-finite number")
        return value
    return _text(value, "policy string")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ReviewedProfile:
    profile_id: str
    model_path: str
    model_sha256: str
    model_bytes: int
    workload: str
    calls_per_arm: int
    device_id: str
    cold_load_host_bytes: int
    worst_case_loads_per_interval: int
    minimum_headroom_bytes_per_s: int
    telemetry_max_age_ms: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        _text(self.profile_id, "profile_id", identifier=True)
        path = Path(_text(self.model_path, "model_path"))
        if not path.is_absolute() or ".." in path.parts:
            raise AdmissionPolicyError("profile model_path must be absolute")
        _sha(self.model_sha256, "profile model")
        _integer(self.model_bytes, "profile model bytes", minimum=1)
        _text(self.workload, "profile workload", identifier=True)
        _integer(self.calls_per_arm, "profile calls", minimum=1, maximum=10_000)
        _text(self.device_id, "profile device", identifier=True)
        _integer(self.cold_load_host_bytes, "profile cold load", minimum=1)
        _integer(self.worst_case_loads_per_interval, "profile cadence",
                 minimum=1, maximum=1_000_000)
        _integer(self.minimum_headroom_bytes_per_s, "profile headroom", minimum=1)
        _integer(self.telemetry_max_age_ms, "profile telemetry age",
                 minimum=1, maximum=3_600_000)
        _sha(self.evidence_sha256, "profile evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id, "model_path": self.model_path,
            "model_sha256": self.model_sha256, "model_bytes": self.model_bytes,
            "workload": self.workload, "calls_per_arm": self.calls_per_arm,
            "device_id": self.device_id,
            "cold_load_host_bytes": self.cold_load_host_bytes,
            "worst_case_loads_per_interval": self.worst_case_loads_per_interval,
            "minimum_headroom_bytes_per_s": self.minimum_headroom_bytes_per_s,
            "telemetry_max_age_ms": self.telemetry_max_age_ms,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class PolicyExample:
    example_id: str
    polarity: str
    facts: Mapping[str, Any]
    missing: tuple[str, ...]
    mode: str
    rationale: str
    disqualifiers: tuple[str, ...]
    counterfactual: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.example_id, "example id", identifier=True)
        if self.polarity not in {"positive", "negative"}:
            raise AdmissionPolicyError("example polarity must be positive or negative")
        if self.mode not in MODES:
            raise AdmissionPolicyError("example mode is invalid")
        if self.polarity == "positive" and self.mode not in {"cold_overlap", "hot_resident"}:
            raise AdmissionPolicyError("positive example must describe reviewed overlap or future hot reuse")
        if self.polarity == "negative" and self.mode != "cold_serialized":
            raise AdmissionPolicyError("negative example must fail closed to serialization")
        if not isinstance(self.facts, Mapping) or not self.facts:
            raise AdmissionPolicyError("example facts must be a non-empty immutable mapping")
        for label, values in (("missing", self.missing),
                              ("disqualifiers", self.disqualifiers)):
            if not isinstance(values, tuple) or any(not isinstance(x, str) or not x
                                                    for x in values):
                raise AdmissionPolicyError(f"example {label} is malformed")
        if (not self.evidence or any(not isinstance(x, str)
                                    or not x.startswith("sha256:")
                                    or not SHA256.fullmatch(x.removeprefix("sha256:"))
                                    for x in self.evidence)):
            raise AdmissionPolicyError("example evidence is malformed")
        _text(self.rationale, "example rationale")
        _text(self.counterfactual, "example counterfactual")


@dataclass(frozen=True)
class PolicyCorpus:
    version: str
    policy_sha256: str
    file_sha256: str
    profiles: tuple[ReviewedProfile, ...]
    examples: tuple[PolicyExample, ...]
    sealed: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.version, "policy version", identifier=True)
        _sha(self.policy_sha256, "policy")
        _sha(self.file_sha256, "policy file")
        if (not self.profiles or len(self.profiles) > MAX_PROFILES
                or not self.examples or len(self.examples) > MAX_EXAMPLES):
            raise AdmissionPolicyError("policy requires bounded profiles and examples")
        if len({row.profile_id for row in self.profiles}) != len(self.profiles):
            raise AdmissionPolicyError("policy contains duplicate profile IDs")
        identities = [tuple((key, value) for key, value in row.to_dict().items()
                            if key not in {"profile_id", "evidence_sha256"})
                      for row in self.profiles]
        if len(set(identities)) != len(identities):
            raise AdmissionPolicyError("policy contains duplicate profile fact frames")
        if len({row.example_id for row in self.examples}) != len(self.examples):
            raise AdmissionPolicyError("policy contains duplicate example IDs")
        if {row.polarity for row in self.examples} != {"positive", "negative"}:
            raise AdmissionPolicyError("policy requires positive and negative examples")
        profile_ids = {row.profile_id for row in self.profiles}
        if any(row.facts.get("profile_id") not in profile_ids for row in self.examples):
            raise AdmissionPolicyError("policy example does not cite a reviewed profile")


@dataclass(frozen=True)
class AdmissionRequest:
    model_path: str
    model_sha256: str
    model_bytes: int
    workload: str
    calls_per_arm: int
    device_id: str
    cold_load_host_bytes: int
    worst_case_loads_per_interval: int
    telemetry_observed: bool
    telemetry_age_ms: int | None
    observed_headroom_bytes_per_s: int | None
    telemetry_receipt_sha256: str | None
    foreign_kfd_pids: tuple[int, ...] = ()
    runtime_manifest_sha256: str | None = None
    runtime_arm: str | None = None
    hot_residency: HotResidencyIdentity | None = None
    expected_hot_identity_sha256: str | None = None
    residency_revalidated: bool = False

    def __post_init__(self) -> None:
        path = Path(_text(self.model_path, "request model_path"))
        if not path.is_absolute() or ".." in path.parts:
            raise AdmissionPolicyError("request model_path must be absolute")
        _sha(self.model_sha256, "request model")
        _integer(self.model_bytes, "request model bytes", minimum=1)
        _text(self.workload, "request workload", identifier=True)
        _integer(self.calls_per_arm, "request calls", minimum=1, maximum=10_000)
        _text(self.device_id, "request device", identifier=True)
        _integer(self.cold_load_host_bytes, "request cold load", minimum=1)
        _integer(self.worst_case_loads_per_interval, "request cadence",
                 minimum=1, maximum=1_000_000)
        if not isinstance(self.telemetry_observed, bool):
            raise AdmissionPolicyError("telemetry_observed must be boolean")
        if self.telemetry_observed:
            _integer(self.telemetry_age_ms, "telemetry age", maximum=3_600_000)
            _integer(self.observed_headroom_bytes_per_s,
                     "observed headroom", minimum=0)
            _sha(self.telemetry_receipt_sha256, "telemetry receipt")
        elif any(value is not None for value in (
                self.telemetry_age_ms, self.observed_headroom_bytes_per_s,
                self.telemetry_receipt_sha256)):
            raise AdmissionPolicyError("unobserved telemetry may not carry asserted values")
        if (not isinstance(self.foreign_kfd_pids, tuple)
                or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                       for pid in self.foreign_kfd_pids)
                or len(set(self.foreign_kfd_pids)) != len(self.foreign_kfd_pids)):
            raise AdmissionPolicyError("foreign KFD PID evidence is malformed")
        hot_fields = (self.runtime_manifest_sha256, self.runtime_arm,
                      self.hot_residency, self.expected_hot_identity_sha256)
        if any(value is not None for value in hot_fields):
            if (not all(value is not None for value in hot_fields)
                    or not isinstance(self.hot_residency, HotResidencyIdentity)
                    or self.runtime_arm not in {"anchor", "candidate"}
                    or not self.residency_revalidated):
                raise AdmissionPolicyError("hot residency request is incomplete")
            _sha(self.runtime_manifest_sha256, "runtime manifest")
            _sha(self.expected_hot_identity_sha256, "expected hot identity")
        elif self.residency_revalidated:
            raise AdmissionPolicyError("residency cannot be revalidated without an identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path, "model_sha256": self.model_sha256,
            "model_bytes": self.model_bytes, "workload": self.workload,
            "calls_per_arm": self.calls_per_arm, "device_id": self.device_id,
            "cold_load_host_bytes": self.cold_load_host_bytes,
            "worst_case_loads_per_interval": self.worst_case_loads_per_interval,
            "telemetry_observed": self.telemetry_observed,
            "telemetry_age_ms": self.telemetry_age_ms,
            "observed_headroom_bytes_per_s": self.observed_headroom_bytes_per_s,
            "telemetry_receipt_sha256": self.telemetry_receipt_sha256,
            "foreign_kfd_pids": list(self.foreign_kfd_pids),
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "runtime_arm": self.runtime_arm,
            "hot_residency_identity_sha256": (
                self.hot_residency.identity_sha256 if self.hot_residency else None),
            "expected_hot_identity_sha256": self.expected_hot_identity_sha256,
            "residency_revalidated": self.residency_revalidated,
        }


@dataclass(frozen=True)
class AdmissionDecision:
    policy_version: str
    policy_sha256: str
    policy_file_sha256: str
    request: Mapping[str, Any]
    profile: Mapping[str, Any] | None
    actor_recommendation: str | None
    mode: str
    reason: str
    disqualifiers: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise AdmissionPolicyError("decision mode is invalid")
        _sha(self.policy_sha256, "decision policy")
        _sha(self.policy_file_sha256, "decision policy file")
        _sha(self.decision_sha256, "decision")
        if self.actor_recommendation is not None and self.actor_recommendation not in MODES:
            raise AdmissionPolicyError("decision recommendation is invalid")
        _text(self.reason, "decision reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "policy_file_sha256": self.policy_file_sha256,
            "request": _thaw(self.request),
            "profile": None if self.profile is None else _thaw(self.profile),
            "actor_recommendation": self.actor_recommendation,
            "mode": self.mode, "reason": self.reason,
            "disqualifiers": list(self.disqualifiers),
            "promotion_claim": False,
            "decision_sha256": self.decision_sha256,
        }


def _profile(raw: object) -> ReviewedProfile:
    keys = {"profile_id", "model_path", "model_sha256", "model_bytes", "workload",
            "calls_per_arm", "device_id", "cold_load_host_bytes",
            "worst_case_loads_per_interval", "minimum_headroom_bytes_per_s",
            "telemetry_max_age_ms", "evidence_sha256"}
    return ReviewedProfile(**dict(_exact(raw, keys, "profile")))


def _example(raw: object) -> PolicyExample:
    keys = {"id", "polarity", "facts", "missing", "mode", "rationale",
            "disqualifiers", "counterfactual", "evidence"}
    value = _exact(raw, keys, "example")
    for label in ("missing", "disqualifiers", "evidence"):
        if not isinstance(value[label], list):
            raise AdmissionPolicyError(f"example {label} must be a bounded list")
    facts = _freeze(value["facts"])
    return PolicyExample(
        example_id=_text(value["id"], "example id", identifier=True),
        polarity=value["polarity"], facts=facts,
        missing=tuple(value["missing"]), mode=value["mode"],
        rationale=value["rationale"],
        disqualifiers=tuple(value["disqualifiers"]),
        counterfactual=value["counterfactual"],
        evidence=tuple(value["evidence"]))


def load_policy_corpus(path: Path, *, expected_file_sha256: str) -> PolicyCorpus:
    """Load one exact, self-hashed policy corpus into deeply immutable types."""
    if not path.is_absolute() or path.is_symlink():
        raise AdmissionPolicyError("policy path must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AdmissionPolicyError("policy path cannot be resolved") from exc
    if path.absolute() != resolved or not resolved.is_file():
        raise AdmissionPolicyError("policy path traverses a symlink or is not regular")
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as exc:
        raise AdmissionPolicyError("policy corpus is unreadable") from exc
    if len(raw_bytes) > MAX_POLICY_BYTES:
        raise AdmissionPolicyError("policy corpus exceeds the reviewed byte bound")
    actual_file_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_file_sha != _sha(expected_file_sha256, "expected policy file"):
        raise AdmissionPolicyError("policy file bytes differ from the sealed digest")
    try:
        body = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionPolicyError("policy corpus is not JSON") from exc
    value = _exact(body, {"schema", "version", "policy_sha256", "profiles", "examples"},
                   "policy corpus")
    if value["schema"] != POLICY_SCHEMA:
        raise AdmissionPolicyError("policy schema is unsupported")
    expected_policy_sha = _content_hash(
        {key: item for key, item in value.items() if key != "policy_sha256"})
    if value["policy_sha256"] != expected_policy_sha:
        raise AdmissionPolicyError("policy corpus self-hash mismatch")
    if (not isinstance(value["profiles"], list)
            or not 1 <= len(value["profiles"]) <= MAX_PROFILES
            or not isinstance(value["examples"], list)
            or not 2 <= len(value["examples"]) <= MAX_EXAMPLES):
        raise AdmissionPolicyError("policy corpus cardinality is outside bounds")
    sealed = _freeze(value)
    return PolicyCorpus(
        version=_text(value["version"], "policy version", identifier=True),
        policy_sha256=_sha(value["policy_sha256"], "policy"),
        file_sha256=actual_file_sha,
        profiles=tuple(_profile(row) for row in value["profiles"]),
        examples=tuple(_example(row) for row in value["examples"]),
        sealed=sealed)


def _profile_mismatches(profile: ReviewedProfile,
                        request: AdmissionRequest) -> tuple[str, ...]:
    fields = ("model_path", "model_sha256", "model_bytes", "workload",
              "calls_per_arm", "device_id", "cold_load_host_bytes",
              "worst_case_loads_per_interval")
    return tuple(f"profile_mismatch:{field}" for field in fields
                 if getattr(profile, field) != getattr(request, field))


def _hot_identity_valid(request: AdmissionRequest) -> bool:
    hot = request.hot_residency
    if hot is None:
        return False
    body = {
        "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
        "runtime_manifest_sha256": hot.runtime_manifest_sha256,
        "arm": hot.arm,
        "reward_binary_sha256": hot.reward_binary_sha256,
        "hip_library_sha256": hot.hip_library_sha256,
        "model_path": str(hot.model_path), "model_sha256": hot.model_sha256,
        "device_id": hot.device_id, "kfd_pid": hot.kfd_pid,
        "boot_id": hot.boot_id, "process_start_ticks": hot.process_start_ticks,
        "mapped_local_sha256": dict(hot.mapped_local_sha256),
    }
    return all((
        _content_hash(body) == hot.identity_sha256,
        hot.identity_sha256 == request.expected_hot_identity_sha256,
        hot.runtime_manifest_sha256 == request.runtime_manifest_sha256,
        hot.arm == request.runtime_arm,
        str(hot.model_path) == request.model_path,
        hot.model_sha256 == request.model_sha256,
        hot.device_id == request.device_id,
        request.residency_revalidated,
        not request.foreign_kfd_pids,
    ))


def _decision(*, corpus: PolicyCorpus, request: AdmissionRequest,
              profile: ReviewedProfile | None, recommendation: str | None,
              mode: str, reason: str,
              disqualifiers: Sequence[str]) -> AdmissionDecision:
    body = {
        "schema": DECISION_SCHEMA, "policy_version": corpus.version,
        "policy_sha256": corpus.policy_sha256,
        "policy_file_sha256": corpus.file_sha256,
        "request": request.to_dict(),
        "profile": None if profile is None else profile.to_dict(),
        "actor_recommendation": recommendation, "mode": mode,
        "reason": reason, "disqualifiers": list(disqualifiers),
        "promotion_claim": False,
    }
    return AdmissionDecision(
        policy_version=corpus.version, policy_sha256=corpus.policy_sha256,
        policy_file_sha256=corpus.file_sha256,
        request=_freeze(body["request"]),
        profile=None if profile is None else _freeze(body["profile"]),
        actor_recommendation=recommendation, mode=mode, reason=reason,
        disqualifiers=tuple(disqualifiers),
        decision_sha256=_content_hash(body))


def arbitrate(corpus: PolicyCorpus, request: AdmissionRequest, *,
              actor_recommendation: str | None = None) -> AdmissionDecision:
    """Return the only mode permitted by sealed facts; prose cannot upgrade it."""
    if not isinstance(corpus, PolicyCorpus) or not isinstance(request, AdmissionRequest):
        raise AdmissionPolicyError("arbiter requires typed policy and request")
    if actor_recommendation is not None and actor_recommendation not in MODES:
        raise AdmissionPolicyError("actor recommendation is not a known mode")

    exact = [profile for profile in corpus.profiles
             if not _profile_mismatches(profile, request)]
    profile = exact[0] if len(exact) == 1 else None
    disqualifiers: list[str] = []

    if _hot_identity_valid(request):
        base_mode = "hot_resident"
        reason = "exact self-validating resident process identity was revalidated"
    elif request.hot_residency is not None:
        base_mode = "cold_serialized"
        reason = "declared hot residency did not match its sealed process identity"
        disqualifiers.append("hot_residency_mismatch")
    elif profile is not None:
        if request.foreign_kfd_pids:
            disqualifiers.append("foreign_kfd")
        if not request.telemetry_observed:
            disqualifiers.append("telemetry_missing")
        else:
            assert request.telemetry_age_ms is not None
            assert request.observed_headroom_bytes_per_s is not None
            if request.telemetry_age_ms > profile.telemetry_max_age_ms:
                disqualifiers.append("telemetry_stale")
            if request.observed_headroom_bytes_per_s < profile.minimum_headroom_bytes_per_s:
                disqualifiers.append("headroom_insufficient")
        if disqualifiers:
            base_mode = "cold_serialized"
            reason = "reviewed profile lacked complete safe overlap observations"
        else:
            base_mode = "cold_overlap"
            reason = "all reviewed profile facts and explicit headroom telemetry matched"
    else:
        base_mode = "cold_serialized"
        reason = "no unique exact reviewed profile matched the request"
        if not corpus.profiles:
            disqualifiers.append("profile_missing")
        else:
            for candidate in corpus.profiles:
                disqualifiers.extend(_profile_mismatches(candidate, request))
            disqualifiers = sorted(set(disqualifiers)) or ["profile_ambiguous"]

    # Actor text can request the safest fallback.  It cannot create overlap,
    # create residency, or switch between two fact-derived nonserialized modes.
    if actor_recommendation == "cold_serialized" and base_mode != "cold_serialized":
        disqualifiers.append("actor_requested_safer_serialization")
        mode = "cold_serialized"
        reason = "actor recommendation downgraded fact-authorized mode to serialization"
    else:
        mode = base_mode
        if actor_recommendation is not None and actor_recommendation != base_mode:
            disqualifiers.append("actor_recommendation_not_authoritative")

    return _decision(corpus=corpus, request=request, profile=profile,
                     recommendation=actor_recommendation, mode=mode,
                     reason=reason, disqualifiers=tuple(sorted(set(disqualifiers))))


def validate_decision_receipt(value: Mapping[str, Any]) -> None:
    """Fail closed if any decision field changed after sealing."""
    keys = {"schema", "policy_version", "policy_sha256", "policy_file_sha256",
            "request", "profile",
            "actor_recommendation", "mode", "reason", "disqualifiers",
            "promotion_claim", "decision_sha256"}
    raw = _exact(value, keys, "decision receipt")
    if (raw["schema"] != DECISION_SCHEMA or raw["mode"] not in MODES
            or raw["promotion_claim"] is not False):
        raise AdmissionPolicyError("decision receipt authority is invalid")
    expected = _content_hash({key: item for key, item in raw.items()
                              if key != "decision_sha256"})
    if raw["decision_sha256"] != expected:
        raise AdmissionPolicyError("decision receipt self-hash mismatch")
