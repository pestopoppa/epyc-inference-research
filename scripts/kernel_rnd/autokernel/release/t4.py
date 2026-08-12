#!/usr/bin/env python3
"""T4 post-cutover activation and watch-window evaluator.

T3 answers whether a sealed candidate is releasable.  T4 answers two different
questions after an operator has executed the cutover:

* did the cutover activate the intended binaries, linkage and processes; and
* has the predeclared production watch window observed a regression?

This module only evaluates captured JSON receipts.  It has no process, network,
clock, inference, benchmark, production-write or rollback capability.  A bad
receipt produces a recommendation to raise an operator decision package; it
never performs the action itself.  The watch arithmetic remains owned by
``release.packager`` so there is one implementation of the §11.5 bands and
``later_of(duration, volume)`` close condition.

CLI::

    python3 -m scripts.kernel_rnd.autokernel.release.t4 --request t4-request.json

The CLI reads one request and writes the result to stdout.  Exit 0 means continue
or recommend keep, 2 means raise an operator decision package, 3 means evidence
is incomplete, and 64 means the input was refused.  It never writes a file.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .. import schemas
from . import packager, t3

__all__ = [
    "TIER", "MODULE_ID", "REQUEST_SCHEMA", "RESULT_SCHEMA",
    "ACTIVATION_MANIFEST_SCHEMA", "RECORD_CLASS", "PROBE_ROLE_CANARY",
    "PROBE_TRANSPORT_HEALTH", "PROBE_API_HEALTH", "PROBE_SPEECH_SMOKE",
    "PROBE_KINDS", "GLOBAL_PROBE_KINDS", "RECOMMEND_CONTINUE", "RECOMMEND_KEEP",
    "RECOMMEND_DECISION", "RECOMMEND_INCOMPLETE", "RECOMMENDATIONS", "EXIT_OK",
    "EXIT_DECISION", "EXIT_INCOMPLETE", "EXIT_INPUT", "T4Error", "T4InputError",
    "LiveRoleExpectation", "LiveRoleReceipt", "ProbeReceipt", "RollbackAnchorReceipt",
    "T4Request", "T4Result", "T4Runner", "activation_manifest",
    "activation_manifest_sha256", "evaluate_t4", "audit_no_live_or_mutating_capability",
    "main",
]

TIER = "T4"
MODULE_ID = "autokernel.release.t4/v1"
REQUEST_SCHEMA = "epyc.autokernel.t4_request.v1"
RESULT_SCHEMA = "epyc.autokernel.t4_result.v1"
ACTIVATION_MANIFEST_SCHEMA = "epyc.autokernel.t4_activation_manifest.v1"
RECORD_CLASS = (
    "RECOMMENDATION — NOT A CLAIM OR AN ACTION. T4 evaluates operator-supplied "
    "post-cutover receipts and cannot restart, roll back, freeze or write production."
)

PROBE_ROLE_CANARY = "role_canary"
PROBE_TRANSPORT_HEALTH = "transport_health"
PROBE_API_HEALTH = "api_health"
PROBE_SPEECH_SMOKE = "speech_smoke"
PROBE_KINDS = (
    PROBE_ROLE_CANARY,
    PROBE_TRANSPORT_HEALTH,
    PROBE_API_HEALTH,
    PROBE_SPEECH_SMOKE,
)
GLOBAL_PROBE_KINDS = (
    PROBE_TRANSPORT_HEALTH,
    PROBE_API_HEALTH,
    PROBE_SPEECH_SMOKE,
)

RECOMMEND_CONTINUE = "continue_watch"
RECOMMEND_KEEP = "recommend_keep_and_request_window_close"
RECOMMEND_DECISION = "raise_operator_decision_package"
RECOMMEND_INCOMPLETE = "incomplete_evidence"
RECOMMENDATIONS = (
    RECOMMEND_CONTINUE,
    RECOMMEND_KEEP,
    RECOMMEND_DECISION,
    RECOMMEND_INCOMPLETE,
)

EXIT_OK = 0
EXIT_DECISION = 2
EXIT_INCOMPLETE = 3
EXIT_INPUT = 64


class T4Error(Exception):
    """Base for T4 input and evaluator refusals."""


class T4InputError(T4Error):
    """The supplied material is incomplete, ambiguous or internally inconsistent."""


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise T4InputError(f"{label}: required non-empty ISO-8601 timestamp")
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise T4InputError(f"{label}: invalid ISO-8601 timestamp: {exc}") from exc
    if parsed.tzinfo is None:
        parsed = datetime.combine(parsed.date(), parsed.time(), tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise T4InputError(f"{label}: required non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    try:
        return schemas.require.sha256(value, label, error=T4InputError)
    except TypeError as exc:  # compatibility with older require helpers
        raise T4InputError(f"{label}: invalid sha256") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise T4InputError(f"{label}: required object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise T4InputError(f"{label}: required array")
    return value


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    row = _mapping(value, label)
    actual = set(row)
    if actual != keys:
        raise T4InputError(
            f"{label}: unknown={sorted(actual - keys)}, missing={sorted(keys - actual)}")
    return row


def _check(outcome: str, *reasons: str) -> schemas.Check:
    return schemas.Check(outcome, tuple(reasons))


def _worst(checks: Sequence[schemas.Check]) -> schemas.Check:
    rank = {schemas.PASS: 0, schemas.COULD_NOT_CHECK: 1, schemas.FAIL: 2}
    outcome = schemas.PASS
    reasons: list[str] = []
    for check in checks:
        if rank[check.outcome] > rank[outcome]:
            outcome = check.outcome
        reasons.extend(check.reasons)
    return _check(outcome, *reasons)


@dataclass(frozen=True)
class LiveRoleExpectation:
    """Identity the release package says an affected role must run."""

    role: str
    backend: str
    binary_path: str
    binary_sha256: str
    linkage_root: str
    linkage_sha256: str

    def __post_init__(self) -> None:
        _text(self.role, "LiveRoleExpectation.role")
        if self.backend not in schemas.BACKENDS:
            raise T4InputError(
                f"LiveRoleExpectation.backend: {self.backend!r} not in {sorted(schemas.BACKENDS)}")
        _text(self.binary_path, "LiveRoleExpectation.binary_path")
        _sha(self.binary_sha256, "LiveRoleExpectation.binary_sha256")
        _text(self.linkage_root, "LiveRoleExpectation.linkage_root")
        _sha(self.linkage_sha256, "LiveRoleExpectation.linkage_sha256")

    def to_dict(self) -> dict:
        return {
            "role": self.role, "backend": self.backend,
            "binary_path": self.binary_path, "binary_sha256": self.binary_sha256,
            "linkage_root": self.linkage_root, "linkage_sha256": self.linkage_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LiveRoleExpectation":
        row = _exact(value, {"role", "backend", "binary_path", "binary_sha256",
                             "linkage_root", "linkage_sha256"}, "expected_roles[]")
        return cls(**row)


def activation_manifest(*, package_id: str, candidate_era: str,
                        expected_roles: Sequence[LiveRoleExpectation]) -> dict:
    """Canonical pre-cutover identity manifest bound into the release package."""
    _text(package_id, "activation_manifest.package_id")
    _text(candidate_era, "activation_manifest.candidate_era")
    if not expected_roles or not all(
            isinstance(item, LiveRoleExpectation) for item in expected_roles):
        raise T4InputError("activation_manifest.expected_roles: required typed roles")
    roles = [item.to_dict() for item in sorted(expected_roles, key=lambda item: item.role)]
    if len({item["role"] for item in roles}) != len(roles):
        raise T4InputError("activation_manifest.expected_roles: duplicate role")
    return {
        "schema": ACTIVATION_MANIFEST_SCHEMA,
        "package_id": package_id,
        "candidate_era": candidate_era,
        "expected_roles": roles,
    }


def activation_manifest_sha256(*, package_id: str, candidate_era: str,
                               expected_roles: Sequence[LiveRoleExpectation]) -> str:
    return schemas.content_hash(activation_manifest(
        package_id=package_id, candidate_era=candidate_era,
        expected_roles=expected_roles))


@dataclass(frozen=True)
class LiveRoleReceipt:
    """Captured live process, binary and linkage identity for one role."""

    role: str
    backend: str
    pid: int
    enumerated_role_pids: tuple[int, ...]
    process_start_ticks: int
    boot_id: str
    process_started_at: str
    captured_at: str
    binary_path: str
    binary_sha256: str
    linkage_root: str
    linkage_sha256: str
    linkage_verifier: str
    linkage_exit_code: int
    evidence_ref: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _text(self.role, "LiveRoleReceipt.role")
        _text(self.backend, "LiveRoleReceipt.backend")
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid < 1:
            raise T4InputError("LiveRoleReceipt.pid: required positive integer")
        if (not isinstance(self.enumerated_role_pids, tuple)
                or not self.enumerated_role_pids
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 1
                       for item in self.enumerated_role_pids)
                or len(set(self.enumerated_role_pids)) != len(self.enumerated_role_pids)):
            raise T4InputError(
                "LiveRoleReceipt.enumerated_role_pids: unique positive-integer tuple")
        if (isinstance(self.process_start_ticks, bool)
                or not isinstance(self.process_start_ticks, int)
                or self.process_start_ticks < 1):
            raise T4InputError("LiveRoleReceipt.process_start_ticks: required positive integer")
        _text(self.boot_id, "LiveRoleReceipt.boot_id")
        started = _timestamp(self.process_started_at, "LiveRoleReceipt.process_started_at")
        captured = _timestamp(self.captured_at, "LiveRoleReceipt.captured_at")
        if captured < started:
            raise T4InputError("LiveRoleReceipt.captured_at precedes process_started_at")
        _text(self.binary_path, "LiveRoleReceipt.binary_path")
        _sha(self.binary_sha256, "LiveRoleReceipt.binary_sha256")
        _text(self.linkage_root, "LiveRoleReceipt.linkage_root")
        _sha(self.linkage_sha256, "LiveRoleReceipt.linkage_sha256")
        _text(self.linkage_verifier, "LiveRoleReceipt.linkage_verifier")
        if isinstance(self.linkage_exit_code, bool) or not isinstance(self.linkage_exit_code, int):
            raise T4InputError("LiveRoleReceipt.linkage_exit_code: required integer")
        _text(self.evidence_ref, "LiveRoleReceipt.evidence_ref")
        _sha(self.evidence_sha256, "LiveRoleReceipt.evidence_sha256")

    def to_dict(self) -> dict:
        return {
            "role": self.role, "backend": self.backend, "pid": self.pid,
            "enumerated_role_pids": list(self.enumerated_role_pids),
            "process_start_ticks": self.process_start_ticks, "boot_id": self.boot_id,
            "process_started_at": self.process_started_at, "captured_at": self.captured_at,
            "binary_path": self.binary_path, "binary_sha256": self.binary_sha256,
            "linkage_root": self.linkage_root, "linkage_sha256": self.linkage_sha256,
            "linkage_verifier": self.linkage_verifier,
            "linkage_exit_code": self.linkage_exit_code, "evidence_ref": self.evidence_ref,
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LiveRoleReceipt":
        keys = {"role", "backend", "pid", "enumerated_role_pids", "process_start_ticks", "boot_id",
                "process_started_at", "captured_at",
                "binary_path", "binary_sha256", "linkage_root", "linkage_verifier",
                "linkage_sha256", "linkage_exit_code", "evidence_ref", "evidence_sha256"}
        row = _exact(value, keys, "live_roles[]")
        return cls(**{**row, "enumerated_role_pids": tuple(_array(
            row["enumerated_role_pids"], "live_roles[].enumerated_role_pids"))})


@dataclass(frozen=True)
class ProbeReceipt:
    """Captured role-canary or stack-smoke result; PASS is derived, not accepted."""

    probe_kind: str
    role: Optional[str]
    observed_at: str
    exit_code: int
    status_code: Optional[int]
    semantic_success: Optional[bool]
    evidence_ref: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.probe_kind not in PROBE_KINDS:
            raise T4InputError(
                f"ProbeReceipt.probe_kind: {self.probe_kind!r} not in {list(PROBE_KINDS)}")
        if self.probe_kind == PROBE_ROLE_CANARY:
            _text(self.role, "ProbeReceipt.role")
        elif self.role is not None:
            raise T4InputError(
                f"ProbeReceipt({self.probe_kind}): global probe role must be null")
        _timestamp(self.observed_at, "ProbeReceipt.observed_at")
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise T4InputError("ProbeReceipt.exit_code: required integer")
        if self.status_code is not None and (
                isinstance(self.status_code, bool) or not isinstance(self.status_code, int)):
            raise T4InputError("ProbeReceipt.status_code: integer or null")
        if self.semantic_success is not None and not isinstance(self.semantic_success, bool):
            raise T4InputError("ProbeReceipt.semantic_success: bool or null")
        _text(self.evidence_ref, "ProbeReceipt.evidence_ref")
        _sha(self.evidence_sha256, "ProbeReceipt.evidence_sha256")

    def to_dict(self) -> dict:
        return {
            "probe_kind": self.probe_kind, "role": self.role,
            "observed_at": self.observed_at, "exit_code": self.exit_code,
            "status_code": self.status_code, "semantic_success": self.semantic_success,
            "evidence_ref": self.evidence_ref, "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ProbeReceipt":
        keys = {"probe_kind", "role", "observed_at", "exit_code", "status_code",
                "semantic_success", "evidence_ref", "evidence_sha256"}
        return cls(**_exact(value, keys, "probes[]"))


@dataclass(frozen=True)
class RollbackAnchorReceipt:
    anchor_ref: str
    verified_at: str
    available: bool
    immutable: bool
    evidence_ref: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _text(self.anchor_ref, "RollbackAnchorReceipt.anchor_ref")
        _timestamp(self.verified_at, "RollbackAnchorReceipt.verified_at")
        if not isinstance(self.available, bool) or not isinstance(self.immutable, bool):
            raise T4InputError("RollbackAnchorReceipt available/immutable: required bools")
        _text(self.evidence_ref, "RollbackAnchorReceipt.evidence_ref")
        _sha(self.evidence_sha256, "RollbackAnchorReceipt.evidence_sha256")

    def to_dict(self) -> dict:
        return {
            "anchor_ref": self.anchor_ref, "verified_at": self.verified_at,
            "available": self.available, "immutable": self.immutable,
            "evidence_ref": self.evidence_ref, "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RollbackAnchorReceipt":
        keys = {"anchor_ref", "verified_at", "available", "immutable", "evidence_ref",
                "evidence_sha256"}
        return cls(**_exact(value, keys, "rollback_anchor"))


def _window_from_dict(value: Any) -> packager.WatchWindow:
    keys = {"schema", "window_id", "package_id", "owner", "output_class",
            "incumbent_era", "candidate_era", "comparison_method", "affected_roles",
            "duration_rule", "min_duration_days", "min_volume_by_role", "signals",
            "bands_fixed_at", "bands_sha256", "opens_at", "close_step",
            "rollback_anchor_ref", "activation_manifest_ref",
            "activation_manifest_sha256", "rollback_anchor_rule", "computed_by"}
    row = _exact(value, keys, "watch_window")
    if row["schema"] != packager.WATCH_WINDOW_SCHEMA:
        raise T4InputError("watch_window.schema is not the packager watch schema")
    close = _exact(row["close_step"], {"owner", "action", "verdict_required",
                                       "verdict_vocabulary", "unclosed_state"},
                   "watch_window.close_step")
    bands = []
    band_keys = {"signal_id", "source", "alarm_rule", "unit", "lower", "upper", "mde",
                 "basis_ref", "noise_reference_ref", "roles"}
    for index, raw in enumerate(_array(row["signals"], "watch_window.signals")):
        band = _exact(raw, band_keys, f"watch_window.signals[{index}]")
        bands.append(packager.WatchSignalBand(
            signal_id=band["signal_id"], unit=band["unit"], lower=band["lower"],
            upper=band["upper"], mde=band["mde"], basis_ref=band["basis_ref"],
            noise_reference_ref=band["noise_reference_ref"],
            roles=tuple(_array(band["roles"], f"watch_window.signals[{index}].roles"))))
    affected_roles = tuple(_array(row["affected_roles"], "watch_window.affected_roles"))
    min_volume = dict(_mapping(row["min_volume_by_role"],
                               "watch_window.min_volume_by_role"))
    verdicts = tuple(_array(close["verdict_vocabulary"],
                            "watch_window.close_step.verdict_vocabulary"))
    window = packager.WatchWindow(
        window_id=row["window_id"], package_id=row["package_id"], owner=row["owner"],
        incumbent_era=row["incumbent_era"], candidate_era=row["candidate_era"],
        comparison_method=row["comparison_method"], affected_roles=affected_roles,
        min_duration_days=row["min_duration_days"],
        min_volume_by_role=min_volume, bands=tuple(bands),
        bands_fixed_at=row["bands_fixed_at"], opens_at=row["opens_at"],
        close_step=packager.WatchWindowCloseStep(
            owner=close["owner"], action=close["action"],
            verdict_required=close["verdict_required"],
            verdict_vocabulary=verdicts,
            unclosed_state=close["unclosed_state"]),
        rollback_anchor_ref=row["rollback_anchor_ref"],
        activation_manifest_ref=row["activation_manifest_ref"],
        activation_manifest_sha256=row["activation_manifest_sha256"])
    expected = window.to_dict()
    if dict(row) != expected:
        changed = sorted(key for key in keys if row.get(key) != expected.get(key))
        raise T4InputError(f"watch_window derived fields were altered: {changed}")
    return window


def _progress_from_dict(value: Any) -> packager.WatchWindowProgress:
    row = _exact(value, {"now", "volume_by_role", "bands_sha256", "observations"},
                 "progress")
    observations = []
    for index, raw in enumerate(_array(row["observations"], "progress.observations")):
        observation = _exact(raw, {"signal_id", "value", "observed_at", "era_label",
                                   "samples_ref"}, f"progress.observations[{index}]")
        observations.append(packager.WatchObservation(**observation))
    return packager.WatchWindowProgress(
        now=row["now"], volume_by_role=dict(_mapping(
            row["volume_by_role"], "progress.volume_by_role")),
        bands_sha256=row["bands_sha256"], observations=tuple(observations))


@dataclass(frozen=True)
class T4Request:
    request_id: str
    cutover_at: str
    watch_window: packager.WatchWindow
    expected_roles: tuple[LiveRoleExpectation, ...]
    live_roles: tuple[LiveRoleReceipt, ...]
    probes: tuple[ProbeReceipt, ...]
    rollback_anchor: RollbackAnchorReceipt
    progress: packager.WatchWindowProgress

    def __post_init__(self) -> None:
        _text(self.request_id, "T4Request.request_id")
        cutover = _timestamp(self.cutover_at, "T4Request.cutover_at")
        if not isinstance(self.watch_window, packager.WatchWindow):
            raise T4InputError("T4Request.watch_window: required WatchWindow")
        if cutover != _timestamp(self.watch_window.opens_at, "watch_window.opens_at"):
            raise T4InputError(
                "T4Request.cutover_at must equal watch_window.opens_at; otherwise the "
                "stale-process boundary and watch-window boundary differ")
        if not self.expected_roles:
            raise T4InputError("T4Request.expected_roles: required non-empty tuple")
        if not all(isinstance(item, LiveRoleExpectation) for item in self.expected_roles):
            raise T4InputError("T4Request.expected_roles: invalid member")
        if not all(isinstance(item, LiveRoleReceipt) for item in self.live_roles):
            raise T4InputError("T4Request.live_roles: invalid member")
        if not all(isinstance(item, ProbeReceipt) for item in self.probes):
            raise T4InputError("T4Request.probes: invalid member")
        if not isinstance(self.rollback_anchor, RollbackAnchorReceipt):
            raise T4InputError("T4Request.rollback_anchor: invalid receipt")
        if not isinstance(self.progress, packager.WatchWindowProgress):
            raise T4InputError("T4Request.progress: invalid progress")
        affected = set(self.watch_window.affected_roles)
        expected = [item.role for item in self.expected_roles]
        live = [item.role for item in self.live_roles]
        if len(expected) != len(set(expected)) or set(expected) != affected:
            raise T4InputError(
                f"T4Request.expected_roles must cover affected roles exactly: {sorted(affected)}")
        actual_manifest_sha256 = activation_manifest_sha256(
            package_id=self.watch_window.package_id,
            candidate_era=self.watch_window.candidate_era,
            expected_roles=self.expected_roles)
        if actual_manifest_sha256 != self.watch_window.activation_manifest_sha256:
            raise T4InputError(
                "T4Request.expected_roles differ from the activation manifest fixed in "
                "the release package before cutover")
        if len(live) != len(set(live)) or set(live) != affected:
            raise T4InputError(
                f"T4Request.live_roles must cover affected roles exactly: {sorted(affected)}")
        canaries = [probe.role for probe in self.probes
                    if probe.probe_kind == PROBE_ROLE_CANARY]
        if len(canaries) != len(set(canaries)) or set(canaries) != affected:
            raise T4InputError(
                f"T4Request role canaries must cover affected roles exactly: {sorted(affected)}")
        global_kinds = [probe.probe_kind for probe in self.probes
                        if probe.probe_kind in GLOBAL_PROBE_KINDS]
        if len(global_kinds) != len(set(global_kinds)) or set(global_kinds) != set(
                GLOBAL_PROBE_KINDS):
            raise T4InputError(
                f"T4Request requires exactly one of each global probe: {list(GLOBAL_PROBE_KINDS)}")
        if self.rollback_anchor.anchor_ref != self.watch_window.rollback_anchor_ref:
            raise T4InputError("rollback receipt names a different anchor than the watch window")
        opens = _timestamp(self.watch_window.opens_at, "watch_window.opens_at")
        now = _timestamp(self.progress.now, "progress.now")
        if now < opens:
            raise T4InputError("progress.now precedes the cutover/watch opening")
        for receipt in self.live_roles:
            if _timestamp(receipt.captured_at, "live receipt captured_at") > now:
                raise T4InputError("live receipt was captured after progress.now")
        for probe in self.probes:
            observed = _timestamp(probe.observed_at, "probe observed_at")
            if observed < opens or observed > now:
                raise T4InputError("probe observation falls outside cutover..progress.now")
        verified = _timestamp(self.rollback_anchor.verified_at, "rollback verified_at")
        if verified < opens or verified > now:
            raise T4InputError("rollback verification falls outside cutover..progress.now")
        for observation in self.progress.observations:
            observed = _timestamp(observation.observed_at, "watch observation observed_at")
            if observed < opens or observed > now:
                raise T4InputError("watch observation falls outside cutover..progress.now")

    def to_dict(self) -> dict:
        return {
            "schema": REQUEST_SCHEMA, "request_id": self.request_id,
            "cutover_at": self.cutover_at, "watch_window": self.watch_window.to_dict(),
            "expected_roles": [item.to_dict() for item in self.expected_roles],
            "live_roles": [item.to_dict() for item in self.live_roles],
            "probes": [item.to_dict() for item in self.probes],
            "rollback_anchor": self.rollback_anchor.to_dict(),
            "progress": self.progress.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "T4Request":
        keys = {"schema", "request_id", "cutover_at", "watch_window", "expected_roles",
                "live_roles", "probes", "rollback_anchor", "progress"}
        row = _exact(value, keys, "request")
        if row["schema"] != REQUEST_SCHEMA:
            raise T4InputError(f"request.schema: expected {REQUEST_SCHEMA!r}")
        for name in ("expected_roles", "live_roles", "probes"):
            _array(row[name], f"request.{name}")
        return cls(
            request_id=row["request_id"], cutover_at=row["cutover_at"],
            watch_window=_window_from_dict(row["watch_window"]),
            expected_roles=tuple(LiveRoleExpectation.from_dict(v)
                                 for v in row["expected_roles"]),
            live_roles=tuple(LiveRoleReceipt.from_dict(v) for v in row["live_roles"]),
            probes=tuple(ProbeReceipt.from_dict(v) for v in row["probes"]),
            rollback_anchor=RollbackAnchorReceipt.from_dict(row["rollback_anchor"]),
            progress=_progress_from_dict(row["progress"]))

    def fingerprint(self) -> str:
        return schemas.content_hash(self.to_dict())


def _role_check(expectation: LiveRoleExpectation, receipt: LiveRoleReceipt,
                cutover: datetime) -> schemas.Check:
    reasons = []
    for name in ("backend", "binary_path", "binary_sha256", "linkage_root",
                 "linkage_sha256"):
        if getattr(receipt, name) != getattr(expectation, name):
            reasons.append(
                f"{expectation.role}: live {name} {getattr(receipt, name)!r} differs from "
                f"package expectation {getattr(expectation, name)!r}")
    if _timestamp(receipt.process_started_at, "process_started_at") < cutover:
        reasons.append(
            f"{expectation.role}: pid {receipt.pid} started before the cutover and is stale")
    if set(receipt.enumerated_role_pids) != {receipt.pid}:
        reasons.append(
            f"{expectation.role}: role process enumeration {list(receipt.enumerated_role_pids)} "
            f"is not the single expected pid {receipt.pid}; a peer may be stale")
    if not receipt.linkage_verifier.endswith(t3.LINKAGE_VERIFIER_RELPATH):
        reasons.append(
            f"{expectation.role}: linkage was not captured by {t3.LINKAGE_VERIFIER_RELPATH}")
    if receipt.linkage_exit_code != 0:
        reasons.append(
            f"{expectation.role}: linkage verifier exited {receipt.linkage_exit_code}")
    return _check(schemas.FAIL, *reasons) if reasons else _check(schemas.PASS)


def _probe_check(probe: ProbeReceipt) -> schemas.Check:
    label = probe.role or probe.probe_kind
    missing = []
    failures = []
    if probe.semantic_success is None:
        missing.append(f"{label}: probe has no semantic result")
    elif probe.semantic_success is False:
        failures.append(f"{label}: probe semantic result is false")
    if probe.exit_code != 0:
        failures.append(f"{label}: probe exited {probe.exit_code}")
    if (probe.probe_kind in (PROBE_TRANSPORT_HEALTH, PROBE_API_HEALTH)
            and probe.status_code is None):
        missing.append(f"{label}: HTTP health probe has no status code")
    if probe.status_code is not None and not 200 <= probe.status_code < 300:
        failures.append(f"{label}: probe HTTP status is {probe.status_code}")
    if failures:
        return _check(schemas.FAIL, *failures, *missing)
    if missing:
        return _check(schemas.COULD_NOT_CHECK, *missing)
    return _check(schemas.PASS)


@dataclass(frozen=True)
class T4Result:
    request_id: str
    request_sha256: str
    recommendation: str
    activation_check: schemas.Check
    role_checks: Mapping[str, schemas.Check]
    probe_checks: Mapping[str, schemas.Check]
    rollback_check: schemas.Check
    watch: packager.WatchWindowRecommendation

    def __post_init__(self) -> None:
        if self.recommendation not in RECOMMENDATIONS:
            raise T4InputError("T4Result.recommendation: unknown value")

    def to_dict(self) -> dict:
        def encode(check: schemas.Check) -> dict:
            return {"outcome": check.outcome, "reasons": list(check.reasons)}

        return {
            "schema": RESULT_SCHEMA, "tier": TIER, "module_id": MODULE_ID,
            "record_class": RECORD_CLASS, "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "recommendation": self.recommendation,
            "activation_check": encode(self.activation_check),
            "role_checks": {key: encode(value) for key, value in self.role_checks.items()},
            "probe_checks": {key: encode(value) for key, value in self.probe_checks.items()},
            "rollback_check": encode(self.rollback_check),
            "watch": self.watch.to_dict(),
        }


def evaluate_t4(request: T4Request) -> T4Result:
    """Evaluate captured T4 material without collecting or mutating anything."""
    if not isinstance(request, T4Request):
        raise T4InputError(f"evaluate_t4: expected T4Request, got {type(request).__name__}")
    cutover = _timestamp(request.cutover_at, "cutover_at")
    expectation = {item.role: item for item in request.expected_roles}
    role_checks = {
        item.role: _role_check(expectation[item.role], item, cutover)
        for item in request.live_roles
    }
    probe_checks = {}
    for probe in request.probes:
        key = f"{probe.probe_kind}:{probe.role}" if probe.role else probe.probe_kind
        probe_checks[key] = _probe_check(probe)
    rollback_reasons = []
    if not request.rollback_anchor.available:
        rollback_reasons.append("rollback anchor is not available")
    if not request.rollback_anchor.immutable:
        rollback_reasons.append("rollback anchor is not verified immutable")
    rollback_check = (_check(schemas.FAIL, *rollback_reasons) if rollback_reasons
                      else _check(schemas.PASS))
    activation = _worst(tuple(role_checks.values()) + tuple(probe_checks.values())
                        + (rollback_check,))
    watch = packager.evaluate_watch_window(request.watch_window, request.progress)
    if activation.outcome == schemas.FAIL or watch.alarms:
        recommendation = RECOMMEND_DECISION
    elif activation.outcome == schemas.COULD_NOT_CHECK or watch.unevaluable:
        recommendation = RECOMMEND_INCOMPLETE
    elif watch.recommendation == packager.WATCH_CLOSE_NO_REGRESSION:
        recommendation = RECOMMEND_KEEP
    else:
        recommendation = RECOMMEND_CONTINUE
    return T4Result(
        request_id=request.request_id, request_sha256=request.fingerprint(),
        recommendation=recommendation, activation_check=activation,
        role_checks=role_checks, probe_checks=probe_checks,
        rollback_check=rollback_check, watch=watch)


class T4Runner:
    """The release-tier evaluator seam for post-cutover material."""

    tier = TIER

    def evaluate_release(self, request: Any) -> T4Result:
        if not isinstance(request, T4Request):
            raise T4InputError(
                f"T4Runner.evaluate_release: expected T4Request, got {type(request).__name__}")
        return evaluate_t4(request)


def audit_no_live_or_mutating_capability(source: Optional[str] = None) -> schemas.Check:
    """Prove the evaluator has no process/network/clock/write capability."""
    text = Path(__file__).read_text(encoding="utf-8") if source is None else source
    tree = ast.parse(text)
    reasons = []
    denied_imports = {"os", "subprocess", "shutil", "signal", "socket", "requests", "httpx"}
    denied_calls = {"write_text", "write_bytes", "unlink", "rename", "replace", "mkdir",
                    "rmdir", "touch", "chmod", "chown", "now", "utcnow"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in denied_imports:
                    reasons.append(f"imports denied module {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in denied_imports:
                reasons.append(f"imports from denied module {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in denied_calls:
                reasons.append(f"calls denied capability .{node.func.attr}()")
    return _check(schemas.FAIL, *reasons) if reasons else _check(schemas.PASS)


def _exit_code(result: T4Result) -> int:
    if result.recommendation == RECOMMEND_DECISION:
        return EXIT_DECISION
    if result.recommendation == RECOMMEND_INCOMPLETE:
        return EXIT_INCOMPLETE
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate captured AutoKernel T4 receipts")
    parser.add_argument("--request", required=True, help="request JSON path, or - for stdin")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read() if args.request == "-" else Path(args.request).read_text(
            encoding="utf-8")
        document = json.loads(raw)
        result = T4Runner().evaluate_release(T4Request.from_dict(document))
    except (OSError, json.JSONDecodeError, T4Error, packager.PackagerError,
            TypeError, ValueError) as exc:
        print(json.dumps({"schema": RESULT_SCHEMA, "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return EXIT_INPUT
    print(json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")))
    return _exit_code(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
