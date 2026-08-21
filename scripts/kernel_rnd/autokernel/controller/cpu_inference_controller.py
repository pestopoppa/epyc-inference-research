#!/usr/bin/env python3
"""Durable controller for full AutoKernel CPU inference campaigns.

This module deliberately does not implement a second build, correctness, claim,
or benchmark path.  It selects one pre-authored candidate at a time and invokes
``autokernel.campaign`` which already owns, in order:

* isolated source materialization and candidate build;
* current T0 correctness;
* the governed CPU-region claim and per-model-call inference window;
* calibrated, interleaved anchor/candidate ``llama-bench`` comparison; and
* release, production-tree immutability proof, and durable evidence journaling.

The missing product seam was lifecycle: ordered candidate selection, immutable
input authority, truthful science accounting, and restart without replaying a
sealed operation.  Those are the only responsibilities added here.

Validation is side-effect free.  Execution is available only through the CLI's
paired ``--execute --i-hold-the-host`` flags; the controller injects the
campaign's corresponding flags itself, so a manifest cannot forge operator
host ownership.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Protocol, Sequence

from .. import campaign, journal, storage


SCHEMA = "epyc.autokernel.cpu_inference_controller_manifest.v1"
CANDIDATE_SCHEMA = "epyc.autokernel.cpu_inference_candidate.v1"
STATE_SCHEMA = "epyc.autokernel.cpu_inference_controller_state.v2"
RECEIPT_SCHEMA = "epyc.autokernel.cpu_inference_controller_receipt.v2"
HASH = re.compile(r"^[0-9a-f]{64}$")
ID = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
CONTROLLER_ID = re.compile(r"^ak-[a-z0-9](?:[a-z0-9._-]{0,123}[a-z0-9])?$")
CANDIDATE_ID = re.compile(r"^akc-[a-z0-9](?:[a-z0-9._-]{0,122}[a-z0-9])?$")
HYPOTHESIS_ID = re.compile(r"^akh-[a-z0-9](?:[a-z0-9._-]{0,122}[a-z0-9])?$")

_MODE_FLAGS = frozenset({
    "--execute", "--dry-run", "--i-hold-the-host", "--json",
    "--screening-only", "--create-screening-baseline",
    "--screening-baseline-bank",
})
_SINGLE_VALUE_FLAGS = frozenset({
    "--campaign-id", "--candidate-id", "--candidate",
    "--proposal-manifest", "--least-commitment-capture-plan",
    "--matched-experiment-id", "--source-patch-manifest",
    "--source-prerequisite-package", "--fresh-source-prerequisite-plan",
    "--calibration-bundle", "--physical-envelope", "--ranked-units",
    "--backend", "--blocks", "--recipe", "--model", "--reps",
    "--nominal-khz", "--journal-root", "--hypothesis",
    "--hypothesis-store",
})
_PATH_FLAGS = frozenset({
    "--proposal-manifest", "--least-commitment-capture-plan",
    "--source-patch-manifest", "--source-prerequisite-package",
    "--fresh-source-prerequisite-plan", "--calibration-bundle",
    "--physical-envelope", "--ranked-units", "--model",
    "--hypothesis-store",
})
_TREE_PATH_FLAGS = frozenset({"--calibration-bundle"})
_REQUIRED_FLAGS = frozenset({
    "--campaign-id", "--candidate-id", "--candidate",
    "--proposal-manifest", "--source-patch-manifest",
    "--calibration-bundle", "--backend", "--model",
    "--nominal-khz", "--journal-root", "--hypothesis",
    "--hypothesis-store",
})
_CAMPAIGN_RESULT_KEYS = frozenset({
    "schema", "state", "campaign_id", "candidate_id", "spec", "steps",
    "t0", "decision", "pairs", "preflight", "production_unchanged",
    "releases", "error", "executed", "screening_only", "non_promotable",
    "journal_error", "screening_report", "ok", "grammar",
})
_DECISION_KEYS = frozenset({
    "keep", "reason", "blocks", "min_delta", "median_relative",
    "contribution_floor", "calibration_evidence_ref", "drift_bound",
    "anchor_drift", "deltas", "relatives", "anchors", "orders",
})


class CpuInferenceControllerError(RuntimeError):
    """A fail-closed controller/configuration refusal."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CpuInferenceControllerError(
            "controller material is not canonical finite JSON") from exc


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_json(data: bytes, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CpuInferenceControllerError(
                    f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CpuInferenceControllerError(
            f"{label} contains non-finite JSON value {value}")

    try:
        return json.loads(data.decode("utf-8", "strict"),
                          object_pairs_hook=object_pairs,
                          parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuInferenceControllerError(
            f"{label} is not strict UTF-8 JSON") from exc


def _read_regular(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    if not path.is_absolute():
        raise CpuInferenceControllerError(f"{label} path must be absolute")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CpuInferenceControllerError(
            f"{label} cannot be opened as a regular non-symlink file") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CpuInferenceControllerError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev, value.st_ino, value.st_uid,
                stat.S_IMODE(value.st_mode), value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after):
            raise CpuInferenceControllerError(f"{label} changed while hashing")
        try:
            current = path.lstat()
        except OSError as exc:
            raise CpuInferenceControllerError(
                f"{label} disappeared after hashing") from exc
        if identity(after) != identity(current):
            raise CpuInferenceControllerError(
                f"{label} pathname identity changed while hashing")
        return b"".join(chunks), after
    finally:
        os.close(fd)


def _lexically_safe_absolute(path: Path, label: str) -> None:
    raw = str(path)
    if (not raw.startswith("/") or "\0" in raw or "//" in raw
            or any(part in {"", ".", ".."} for part in path.parts[1:])
            or os.path.normpath(raw) != raw):
        raise CpuInferenceControllerError(
            f"{label} must be a canonical absolute path without traversal")


def _assert_no_symlink_ancestry(path: Path, label: str) -> None:
    _lexically_safe_absolute(path, label)
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise CpuInferenceControllerError(
                f"{label} ancestry cannot be attested") from exc
        if stat.S_ISLNK(info.st_mode):
            raise CpuInferenceControllerError(
                f"{label} has symlink ancestry at {current}")


def _contains_or_is_contained(left: Path, right: Path) -> bool:
    left_s, right_s = str(left), str(right)
    try:
        common = os.path.commonpath((left_s, right_s))
    except ValueError:
        return False
    return common in {left_s, right_s}


def _validate_controller_path(path: Path, label: str, *, root: Path | None = None) -> None:
    """Refuse controller-owned material in repositories or production trees."""
    _assert_no_symlink_ancestry(path, label)
    if ".git" in path.parts:
        raise CpuInferenceControllerError(f"{label} may not be inside .git")
    if root is not None:
        _assert_no_symlink_ancestry(root, "controller output root")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CpuInferenceControllerError(
                f"{label} escapes the controller output root") from exc
    # An output below a repository is mutable through that repository; an
    # output above it could contain and overwrite the repository.  Check every
    # existing ancestor for the former and the owned root itself for the latter.
    for ancestor in (path, *path.parents):
        git = ancestor / ".git"
        if (git.exists() or git.is_symlink()) and _contains_or_is_contained(path, git):
            raise CpuInferenceControllerError(
                f"{label} is inside or contains Git metadata at {git}")
    if root is None and path.is_dir():
        for current, dirs, files in os.walk(path, topdown=True, followlinks=False):
            dirs.sort()
            files.sort()
            if ".git" in dirs or ".git" in files:
                raise CpuInferenceControllerError(
                    f"{label} contains Git metadata below {current}")
    for production in campaign.worktree.frozen_tree_paths():
        frozen = Path(production)
        _lexically_safe_absolute(frozen, "frozen production tree")
        if _contains_or_is_contained(path, frozen):
            raise CpuInferenceControllerError(
                f"{label} is inside or contains frozen production tree {frozen}")


def _hash_path(path: Path, kind: str) -> str:
    """Hash a stable file or complete directory closure without following links."""
    if kind == "file":
        return hashlib.sha256(_read_regular(path, f"artifact {path}")[0]).hexdigest()
    if kind != "tree":
        raise CpuInferenceControllerError("artifact kind must be file or tree")
    if not path.is_absolute():
        raise CpuInferenceControllerError("artifact tree path must be absolute")
    try:
        root_before = path.lstat()
    except OSError as exc:
        raise CpuInferenceControllerError(f"artifact tree {path} is unavailable") from exc
    if not stat.S_ISDIR(root_before.st_mode) or path.is_symlink():
        raise CpuInferenceControllerError(
            f"artifact tree {path} is not a non-symlink directory")
    rows: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(path, topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        current_path = Path(current)
        for name in tuple(dirs):
            entry = current_path / name
            info = entry.lstat()
            if not stat.S_ISDIR(info.st_mode) or entry.is_symlink():
                raise CpuInferenceControllerError(
                    f"artifact tree contains non-directory or symlink {entry}")
            rows.append({
                "path": entry.relative_to(path).as_posix(), "kind": "directory",
                "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid,
            })
        for name in files:
            entry = current_path / name
            data, info = _read_regular(entry, f"artifact tree member {entry}")
            rows.append({
                "path": entry.relative_to(path).as_posix(), "kind": "file",
                "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid,
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    root_after = path.lstat()
    def root_identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev, value.st_ino, value.st_uid,
            stat.S_IMODE(value.st_mode), value.st_mtime_ns, value.st_ctime_ns)
    if root_identity(root_before) != root_identity(root_after):
        raise CpuInferenceControllerError(
            f"artifact tree {path} changed while hashing")
    return _sha({"root": str(path), "entries": rows})


@dataclass(frozen=True)
class ArtifactBinding:
    path: Path
    kind: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactBinding":
        if (not isinstance(value, Mapping)
                or set(value) != {"path", "kind", "sha256"}
                or not isinstance(value.get("path"), str)
                or not Path(value["path"]).is_absolute()
                or value.get("kind") not in {"file", "tree"}
                or not isinstance(value.get("sha256"), str)
                or not HASH.fullmatch(value["sha256"])):
            raise CpuInferenceControllerError("artifact binding is malformed")
        return cls(Path(value["path"]), value["kind"], value["sha256"])

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "kind": self.kind,
                "sha256": self.sha256}

    def revalidate(self) -> None:
        if _hash_path(self.path, self.kind) != self.sha256:
            raise CpuInferenceControllerError(
                f"artifact identity changed: {self.path}")


def _parse_args(args: tuple[str, ...]) -> dict[str, str]:
    if (not args or any(not isinstance(value, str) or not value or "\0" in value
                        for value in args)
            or any(value.startswith("--") and "=" in value for value in args)):
        raise CpuInferenceControllerError(
            "campaign arguments must be non-empty separate tokens")
    parsed: dict[str, str] = {}
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in _MODE_FLAGS:
            raise CpuInferenceControllerError(
                f"campaign manifest may not select execution mode {flag}")
        if flag in {"--device", "--device-name"}:
            raise CpuInferenceControllerError(
                "CPU campaign may not declare GPU devices")
        if flag not in _SINGLE_VALUE_FLAGS:
            raise CpuInferenceControllerError(
                f"campaign argument {flag!r} is outside the reviewed CPU surface")
        if flag in parsed:
            raise CpuInferenceControllerError(
                f"campaign argument {flag} is repeated")
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            raise CpuInferenceControllerError(
                f"campaign argument {flag} lacks one value")
        parsed[flag] = args[index + 1]
        index += 2
    missing = sorted(_REQUIRED_FLAGS - set(parsed))
    if missing:
        raise CpuInferenceControllerError(
            f"CPU campaign arguments are missing {missing}")
    if parsed["--backend"] != campaign.BACKEND_CPU:
        raise CpuInferenceControllerError("CPU controller requires --backend llama_cpu")
    source_authorities = set(parsed) & {
        "--source-prerequisite-package", "--fresh-source-prerequisite-plan"}
    if len(source_authorities) != 1:
        raise CpuInferenceControllerError(
            "CPU source campaign requires exactly one current correctness prerequisite authority")
    physical = set(parsed) & {"--physical-envelope", "--ranked-units"}
    if len(physical) != 1:
        raise CpuInferenceControllerError(
            "CPU campaign requires exactly one physical-envelope authority")
    # The controller's allowlist is the authority over *which* knobs may be
    # forwarded.  The campaign parser remains the authority over their types,
    # choices, and argparse grammar.  Exercise it during validate-only so a
    # malformed int cannot survive until the host-owning execution boundary.
    try:
        with redirect_stderr(io.StringIO()):
            normalized = campaign.build_parser().parse_args(list(args))
    except SystemExit as exc:
        raise CpuInferenceControllerError(
            "campaign arguments do not parse under the current campaign grammar") from exc
    destinations = {
        "--campaign-id": "campaign_id", "--candidate-id": "candidate_id",
        "--candidate": "candidate_ref", "--backend": "backend",
        "--blocks": "blocks", "--recipe": "recipe_id", "--model": "model",
        "--reps": "reps", "--nominal-khz": "nominal_khz",
        "--journal-root": "journal_root", "--hypothesis": "hypothesis",
        "--proposal-manifest": "proposal_manifest",
        "--least-commitment-capture-plan": "least_commitment_capture_plan",
        "--matched-experiment-id": "matched_experiment_id",
        "--source-patch-manifest": "source_patch_manifest",
        "--source-prerequisite-package": "source_prerequisite_package",
        "--fresh-source-prerequisite-plan": "fresh_source_prerequisite_plan",
        "--calibration-bundle": "calibration_bundle",
        "--physical-envelope": "physical_envelope",
        "--ranked-units": "ranked_units",
        "--hypothesis-store": "hypothesis_store",
    }
    if set(destinations) != _SINGLE_VALUE_FLAGS:
        raise CpuInferenceControllerError(
            "controller campaign normalization table is incomplete")
    for flag, destination in destinations.items():
        if flag not in parsed:
            continue
        actual = getattr(normalized, destination)
        expected: Any = parsed[flag]
        if flag in {"--blocks", "--reps", "--nominal-khz"}:
            expected = int(expected)
        if actual != expected:
            raise CpuInferenceControllerError(
                f"campaign argument {flag} did not normalize exactly")
    for flag in _PATH_FLAGS | {"--journal-root"}:
        if flag in parsed:
            _lexically_safe_absolute(Path(parsed[flag]), f"campaign argument {flag}")
    if "--blocks" in parsed and int(parsed["--blocks"]) < 2:
        raise CpuInferenceControllerError("campaign --blocks must be at least 2")
    if "--reps" in parsed and int(parsed["--reps"]) < 1:
        raise CpuInferenceControllerError("campaign --reps must be positive")
    if int(parsed["--nominal-khz"]) < 1:
        raise CpuInferenceControllerError("campaign --nominal-khz must be positive")
    if not CONTROLLER_ID.fullmatch(parsed["--campaign-id"]):
        raise CpuInferenceControllerError("campaign id is outside the strict CPU namespace")
    if not CANDIDATE_ID.fullmatch(parsed["--candidate-id"]):
        raise CpuInferenceControllerError("candidate id is outside the strict CPU namespace")
    if not HYPOTHESIS_ID.fullmatch(parsed["--hypothesis"]):
        raise CpuInferenceControllerError("hypothesis id is outside the strict CPU namespace")
    if not parsed["--candidate"].strip():
        raise CpuInferenceControllerError("candidate ref must be non-empty")
    return parsed


@dataclass(frozen=True)
class CpuCandidate:
    candidate_id: str
    hypothesis_id: str
    campaign_args: tuple[str, ...]
    artifacts: tuple[ArtifactBinding, ...]
    candidate_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CpuCandidate":
        required = {"schema", "candidate_id", "hypothesis_id", "campaign_args",
                    "artifacts", "candidate_sha256"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise CpuInferenceControllerError("CPU candidate schema is not exact")
        if (value.get("schema") != CANDIDATE_SCHEMA
                or not isinstance(value.get("candidate_id"), str)
                or not CANDIDATE_ID.fullmatch(value["candidate_id"])
                or not isinstance(value.get("hypothesis_id"), str)
                or not HYPOTHESIS_ID.fullmatch(value["hypothesis_id"])
                or not isinstance(value.get("campaign_args"), list)
                or not isinstance(value.get("artifacts"), list)
                or not isinstance(value.get("candidate_sha256"), str)
                or not HASH.fullmatch(value["candidate_sha256"])):
            raise CpuInferenceControllerError("CPU candidate identity is malformed")
        args = tuple(value["campaign_args"])
        parsed = _parse_args(args)
        if (parsed["--candidate-id"] != value["candidate_id"]
                or parsed["--hypothesis"] != value["hypothesis_id"]):
            raise CpuInferenceControllerError(
                "candidate identity differs from its campaign arguments")
        artifacts = tuple(ArtifactBinding.from_mapping(row)
                          for row in value["artifacts"])
        if (tuple(str(item.path) for item in artifacts)
                != tuple(sorted(str(item.path) for item in artifacts))
                or len({item.path for item in artifacts}) != len(artifacts)):
            raise CpuInferenceControllerError(
                "candidate artifacts must be unique and path-sorted")
        expected_paths = {Path(parsed[flag]) for flag in _PATH_FLAGS if flag in parsed}
        declared_path_values = [Path(parsed[flag]) for flag in sorted(_PATH_FLAGS)
                                if flag in parsed]
        if len(expected_paths) != len(declared_path_values):
            raise CpuInferenceControllerError(
                "distinct campaign inputs may not alias one artifact path")
        if expected_paths != {item.path for item in artifacts}:
            raise CpuInferenceControllerError(
                "candidate artifact bindings differ from path-valued campaign inputs")
        kinds = {item.path: item.kind for item in artifacts}
        for flag in _PATH_FLAGS:
            if flag not in parsed:
                continue
            expected_kind = "tree" if flag in _TREE_PATH_FLAGS else "file"
            if kinds[Path(parsed[flag])] != expected_kind:
                raise CpuInferenceControllerError(
                    f"campaign input {flag} requires a {expected_kind} artifact")
        body = {key: value[key] for key in required - {"candidate_sha256"}}
        if _sha(body) != value["candidate_sha256"]:
            raise CpuInferenceControllerError("candidate self-hash mismatch")
        return cls(value["candidate_id"], value["hypothesis_id"], args,
                   artifacts, value["candidate_sha256"])

    @property
    def parsed_args(self) -> dict[str, str]:
        return _parse_args(self.campaign_args)

    def revalidate(self, output_root: Path) -> None:
        parsed = self.parsed_args
        expected_journal = output_root / "campaign-journal"
        if Path(parsed["--journal-root"]) != expected_journal:
            raise CpuInferenceControllerError(
                "candidate journal root differs from controller-owned root")
        _validate_controller_path(expected_journal, "campaign journal root",
                                  root=output_root)
        for artifact in self.artifacts:
            artifact.revalidate()

    def operation_key(self, controller_id: str) -> str:
        return _sha({
            "schema": "epyc.autokernel.cpu_inference_operation.v1",
            "controller_id": controller_id,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "campaign_args": list(self.campaign_args),
        })


@dataclass(frozen=True)
class ControllerManifest:
    controller_id: str
    output_root: Path
    max_scientific_attempts: int
    candidates: tuple[CpuCandidate, ...]
    manifest_sha256: str

    @classmethod
    def load(cls, path: Path) -> "ControllerManifest":
        value = _strict_json(_read_regular(path, "controller manifest")[0],
                             "controller manifest")
        required = {"schema", "controller_id", "output_root",
                    "max_scientific_attempts", "candidates", "manifest_sha256"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise CpuInferenceControllerError("controller manifest schema is not exact")
        if (value.get("schema") != SCHEMA
                or not isinstance(value.get("controller_id"), str)
                or not CONTROLLER_ID.fullmatch(value["controller_id"])
                or not isinstance(value.get("output_root"), str)
                or not Path(value["output_root"]).is_absolute()
                or isinstance(value.get("max_scientific_attempts"), bool)
                or not isinstance(value.get("max_scientific_attempts"), int)
                or value["max_scientific_attempts"] < 1
                or not isinstance(value.get("candidates"), list)
                or not value["candidates"]
                or not isinstance(value.get("manifest_sha256"), str)
                or not HASH.fullmatch(value["manifest_sha256"])):
            raise CpuInferenceControllerError("controller manifest identity is malformed")
        body = {key: value[key] for key in required - {"manifest_sha256"}}
        if _sha(body) != value["manifest_sha256"]:
            raise CpuInferenceControllerError("controller manifest self-hash mismatch")
        candidates = tuple(CpuCandidate.from_mapping(row)
                           for row in value["candidates"])
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise CpuInferenceControllerError("controller repeats candidate identity")
        output_root = Path(value["output_root"])
        _validate_controller_path(output_root, "controller output root")
        for candidate in candidates:
            candidate.revalidate(output_root)
            if candidate.parsed_args["--campaign-id"] != value["controller_id"]:
                raise CpuInferenceControllerError(
                    "candidate campaign id differs from controller id")
        return cls(value["controller_id"], output_root,
                   value["max_scientific_attempts"], candidates,
                   value["manifest_sha256"])

    def revalidate(self) -> None:
        _validate_controller_path(self.output_root, "controller output root")
        for candidate in self.candidates:
            candidate.revalidate(self.output_root)


class CampaignRunner(Protocol):
    def run(self, args: Sequence[str]) -> tuple[int, Mapping[str, Any] | None]: ...


class InProcessCampaignRunner:
    """Invoke the sole campaign entrypoint and parse its automation result."""

    def run(self, args: Sequence[str]) -> tuple[int, Mapping[str, Any] | None]:
        output = io.StringIO()
        code = campaign.main(tuple(args), out=output)
        data = output.getvalue().encode("utf-8")
        if not data.strip():
            return code, None
        value = _strict_json(data, "campaign JSON output")
        if not isinstance(value, Mapping):
            raise CpuInferenceControllerError(
                "campaign JSON output is not an object")
        return code, value


class StateStore:
    _KEYS = {"schema", "controller_id", "manifest_sha256", "next_index",
             "scientific_attempts", "iterations", "inflight", "complete",
             "terminal_reason", "updated_at", "state_sha256"}

    def __init__(self, config: ControllerManifest) -> None:
        self.config = config
        self.root = config.output_root
        self.path = self.root / "state.json"
        self.lock_path = self.root / ".controller.lock"
        self.book = journal.Journal(
            str(self.root / "controller-journal"),
            campaign_id=config.controller_id)
        for path, label in (
                (self.root, "controller output root"),
                (self.path, "controller state path"),
                (self.lock_path, "controller lock path"),
                (Path(self.book.root), "controller journal path"),
                (self.root / "receipts", "controller receipt directory")):
            _validate_controller_path(path, label,
                                      root=None if path == self.root else self.root)

    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _validate_controller_path(self.lock_path, "controller lock path", root=self.root)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
                     | os.O_NOFOLLOW, 0o600)
        handle = os.fdopen(fd, "r+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def load(self) -> dict[str, Any]:
        _validate_controller_path(self.path, "controller state path", root=self.root)
        if not self.path.exists():
            return {
                "schema": STATE_SCHEMA, "controller_id": self.config.controller_id,
                "manifest_sha256": self.config.manifest_sha256,
                "next_index": 0, "scientific_attempts": 0, "iterations": [],
                "inflight": None, "complete": False, "terminal_reason": None,
                "updated_at": None, "state_sha256": None,
            }
        value = _strict_json(_read_regular(self.path, "controller state")[0],
                             "controller state")
        if not isinstance(value, Mapping) or set(value) != self._KEYS:
            raise CpuInferenceControllerError("controller state schema is not exact")
        body = {key: value[key] for key in self._KEYS - {"state_sha256"}}
        if (value.get("schema") != STATE_SCHEMA
                or value.get("controller_id") != self.config.controller_id
                or value.get("manifest_sha256") != self.config.manifest_sha256
                or not isinstance(value.get("state_sha256"), str)
                or value["state_sha256"] != _sha(body)):
            raise CpuInferenceControllerError(
                "controller state identity or self-hash changed")
        if (isinstance(value.get("next_index"), bool)
                or not isinstance(value.get("next_index"), int)
                or not 0 <= value["next_index"] <= len(self.config.candidates)
                or isinstance(value.get("scientific_attempts"), bool)
                or not isinstance(value.get("scientific_attempts"), int)
                or value["scientific_attempts"] < 0
                or value["scientific_attempts"] >
                   self.config.max_scientific_attempts
                or not isinstance(value.get("iterations"), list)
                or len(value["iterations"]) != value["next_index"]
                or not isinstance(value.get("complete"), bool)):
            raise CpuInferenceControllerError("controller state counters are malformed")
        derived = sum(1 for row in value["iterations"]
                      if isinstance(row, Mapping)
                      and row.get("scientific_budget_spent") is True)
        if derived != value["scientific_attempts"]:
            raise CpuInferenceControllerError(
                "controller science counter is not derived from dispositions")
        self._validate_semantics(value)
        return dict(value)

    def _validate_semantics(self, value: Mapping[str, Any]) -> None:
        row_keys = {
            "candidate_id", "hypothesis_id", "candidate_sha256", "status",
            "classification", "scientific_budget_spent", "keep",
            "receipt_path", "receipt_file_sha256", "result_sha256",
            "operation_key", "effect_fraction", "reason_code", "completed_at",
        }
        scientific_classes = {
            "candidate": True, "screened_out": False,
            "correctness_falsified": False,
        }
        for index, row in enumerate(value["iterations"]):
            candidate = self.config.candidates[index]
            if (not isinstance(row, Mapping) or set(row) != row_keys
                    or row.get("candidate_id") != candidate.candidate_id
                    or row.get("hypothesis_id") != candidate.hypothesis_id
                    or row.get("candidate_sha256") != candidate.candidate_sha256
                    or row.get("operation_key") != candidate.operation_key(
                        self.config.controller_id)
                    or not isinstance(row.get("status"), str)
                    or not isinstance(row.get("completed_at"), str)
                    or not row["completed_at"]):
                raise CpuInferenceControllerError(
                    "controller iteration identity or schema is malformed")
            classification = row.get("classification")
            science = row.get("scientific_budget_spent")
            if classification in scientific_classes:
                expected_keep = scientific_classes[classification]
                expected_status = (campaign.STATE_DECIDED
                                   if classification in {"candidate", "screened_out"}
                                   else campaign.STATE_T0_FAILED)
                if (science is not True or row.get("keep") is not expected_keep
                        or row.get("status") != expected_status
                        or not isinstance(row.get("receipt_path"), str)
                        or Path(row["receipt_path"]) != self.root / "receipts" / (
                            f"{candidate.candidate_id}.json")
                        or not isinstance(row.get("receipt_file_sha256"), str)
                        or not HASH.fullmatch(row["receipt_file_sha256"])
                        or not isinstance(row.get("result_sha256"), str)
                        or not HASH.fullmatch(row["result_sha256"])
                        or (classification in {"candidate", "screened_out"}
                            and (not isinstance(row.get("effect_fraction"), (int, float))
                                 or isinstance(row.get("effect_fraction"), bool)
                                 or not math.isfinite(row["effect_fraction"])))
                        or (classification == "correctness_falsified"
                            and row.get("effect_fraction") is not None)
                        or row.get("reason_code") is not None):
                    raise CpuInferenceControllerError(
                        "scientific CPU iteration is not exactly evidence-bound")
            elif classification == "infrastructure_ambiguous":
                if (science is not False or row.get("keep") is not None
                        or row.get("effect_fraction") is not None):
                    raise CpuInferenceControllerError(
                        "infrastructure ambiguity cannot consume science or keep")
                receipt_values = (row.get("receipt_path"),
                                  row.get("receipt_file_sha256"),
                                  row.get("result_sha256"))
                if any(item is None for item in receipt_values) != all(
                        item is None for item in receipt_values):
                    raise CpuInferenceControllerError(
                        "infrastructure receipt binding is partial")
                if receipt_values[0] is not None and (
                        Path(receipt_values[0]) != self.root / "receipts" / (
                            f"{candidate.candidate_id}.json")
                        or not HASH.fullmatch(str(receipt_values[1]))
                        or not HASH.fullmatch(str(receipt_values[2]))):
                    raise CpuInferenceControllerError(
                        "infrastructure receipt identity is malformed")
                if row.get("reason_code") is not None and (
                        not isinstance(row["reason_code"], str)
                        or not ID.fullmatch(row["reason_code"])):
                    raise CpuInferenceControllerError(
                        "infrastructure reason code is malformed")
                if receipt_values[0] is None:
                    exact_reasons = {
                        "interrupted_without_terminal":
                            "inflight_operation_has_no_sealed_terminal",
                    }
                    status = row.get("status")
                    if status == "campaign_entrypoint_refused":
                        reason = row.get("reason_code")
                        if (not isinstance(reason, str)
                                or not re.fullmatch(r"campaign_exit_[0-9]+_without_result",
                                                    reason)):
                            raise CpuInferenceControllerError(
                                "entrypoint ambiguity lacks its exact exit reason")
                    elif exact_reasons.get(status) != row.get("reason_code"):
                        raise CpuInferenceControllerError(
                            "receipt-free infrastructure ambiguity is not a typed terminal")
                elif row.get("reason_code") is not None:
                    raise CpuInferenceControllerError(
                        "evidence-bound infrastructure ambiguity carries an extra reason")
            else:
                raise CpuInferenceControllerError(
                    "controller iteration classification is unknown")
            if row.get("receipt_path") is not None:
                receipt_path = Path(row["receipt_path"])
                _validate_controller_path(
                    receipt_path, "referenced CPU result receipt", root=self.root)
                receipt_bytes = _read_regular(
                    receipt_path, "referenced CPU result receipt")[0]
                if hashlib.sha256(receipt_bytes).hexdigest() != row[
                        "receipt_file_sha256"]:
                    raise CpuInferenceControllerError(
                        "referenced CPU result receipt file hash changed")
                receipt_result = _load_receipt(
                    receipt_path, candidate,
                    candidate.operation_key(self.config.controller_id))
                if _sha(receipt_result) != row["result_sha256"]:
                    raise CpuInferenceControllerError(
                        "referenced CPU campaign result identity changed")
                # A self-hash proves only byte identity.  Re-run the complete
                # typed disposition on every restart so release, production
                # immutability, T0 and statistical semantics cannot be replaced
                # together with a freshly recomputed receipt/state hash.
                disposition = _result_disposition(
                    candidate, receipt_result, self.root,
                    candidate.operation_key(self.config.controller_id))
                for key in ("status", "classification",
                            "scientific_budget_spent", "keep", "effect_fraction"):
                    if row.get(key) != disposition[key]:
                        raise CpuInferenceControllerError(
                            "controller iteration differs from rederived result disposition")
        inflight = value.get("inflight")
        if inflight is not None:
            keys = {"candidate_index", "candidate_id", "candidate_sha256",
                    "operation_key", "started_at"}
            index = inflight.get("candidate_index") if isinstance(inflight, Mapping) else None
            if (not isinstance(inflight, Mapping) or set(inflight) != keys
                    or isinstance(index, bool) or not isinstance(index, int)
                    or index != value["next_index"]
                    or not 0 <= index < len(self.config.candidates)
                    or inflight.get("candidate_id") !=
                       self.config.candidates[index].candidate_id
                    or inflight.get("candidate_sha256") !=
                       self.config.candidates[index].candidate_sha256
                    or inflight.get("operation_key") !=
                       self.config.candidates[index].operation_key(
                           self.config.controller_id)
                    or not isinstance(inflight.get("started_at"), str)
                    or not inflight["started_at"]):
                raise CpuInferenceControllerError(
                    "inflight candidate checkpoint is malformed")
        terminal_reason = value.get("terminal_reason")
        if value["complete"]:
            expected = (
                "scientific_budget_exhausted"
                if value["scientific_attempts"] >=
                   self.config.max_scientific_attempts
                else "candidate_portfolio_exhausted"
                if value["next_index"] >= len(self.config.candidates)
                else None)
            if terminal_reason != expected or inflight is not None:
                raise CpuInferenceControllerError(
                    "complete controller state lacks its exact terminal condition")
        elif terminal_reason is not None:
            raise CpuInferenceControllerError(
                "nonterminal controller state carries a terminal reason")

    def save(self, state: dict[str, Any], phase: str) -> None:
        state["updated_at"] = _now()
        state["state_sha256"] = _sha({key: state[key] for key in self._KEYS
                                      if key != "state_sha256"})
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _validate_controller_path(self.path, "controller state path", root=self.root)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(state, sort_keys=True, indent=2,
                                    allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.book.initialize()
        self.book.append(journal.KIND_STOP_STATE, {
            "state": f"cpu_inference_{phase}",
            "controller_state_sha256": state["state_sha256"],
            "next_index": state["next_index"],
            "scientific_attempts": state["scientific_attempts"],
        })


def _t0_gates(value: Any) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    if (not isinstance(value, Mapping)
            or set(value) != {"all_pass", "report_ref", "gates"}
            or not isinstance(value.get("all_pass"), bool)
            or not isinstance(value.get("report_ref"), str)
            or not value["report_ref"].strip()
            or not isinstance(value.get("gates"), list)
            or not value["gates"]):
        raise CpuInferenceControllerError("T0 result is not an exact nonempty gate set")
    rows: list[tuple[str, str, tuple[str, ...]]] = []
    for row in value["gates"]:
        if (not isinstance(row, list) or len(row) != 3
                or not isinstance(row[0], str) or not row[0].strip()
                or row[1] not in {campaign.schemas.PASS, campaign.schemas.FAIL,
                                  campaign.schemas.COULD_NOT_CHECK}
                or not isinstance(row[2], list)
                or any(not isinstance(reason, str) for reason in row[2])):
            raise CpuInferenceControllerError("T0 gate row is malformed")
        rows.append((row[0], row[1], tuple(row[2])))
        if row[1] != campaign.schemas.PASS and not row[2]:
            raise CpuInferenceControllerError(
                "T0 FAIL/COULD_NOT_CHECK gate must carry a reason")
    if len({row[0] for row in rows}) != len(rows):
        raise CpuInferenceControllerError("T0 gate identities are repeated")
    expected_all = all(row[1] == campaign.schemas.PASS for row in rows)
    if value["all_pass"] is not expected_all:
        raise CpuInferenceControllerError("T0 all_pass disagrees with typed gates")
    return tuple(rows)


def _exact_decision(spec: Mapping[str, Any], t0: Mapping[str, Any],
                    pairs_value: Any, decision: Any) -> float:
    blocks = spec.get("blocks_precommitted")
    if (isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2
            or not isinstance(pairs_value, list) or len(pairs_value) != blocks
            or not isinstance(decision, Mapping) or set(decision) != _DECISION_KEYS):
        raise CpuInferenceControllerError(
            "CPU pair count and decision do not match the precommitted blocks")
    pairs: list[campaign.Pair] = []
    for index, row in enumerate(pairs_value):
        if (not isinstance(row, Mapping)
                or set(row) != {"block_index", "anchor", "candidate", "order",
                                    "delta", "relative"}
                or row.get("block_index") != index):
            raise CpuInferenceControllerError("CPU pair geometry is not exact")
        for field in ("anchor", "candidate", "delta", "relative"):
            value = row.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise CpuInferenceControllerError("CPU pair contains a non-finite value")
        try:
            pair = campaign.Pair(index, row["anchor"], row["candidate"], row["order"])
        except (TypeError, ValueError) as exc:
            raise CpuInferenceControllerError("CPU pair is invalid") from exc
        expected_pair = pair.to_dict()
        if any(not math.isclose(float(row[key]), float(expected_pair[key]),
                                rel_tol=1e-12, abs_tol=1e-12)
               for key in ("delta", "relative")):
            raise CpuInferenceControllerError("CPU pair derived values disagree")
        pairs.append(pair)
    _t0_gates(t0)
    try:
        expected = campaign.decide(
            pairs,
            t0=campaign.T0Outcome(all_pass=True),
            blocks_precommitted=blocks,
            drift_bound=decision.get("drift_bound"),
            contribution_floor=decision.get("contribution_floor"),
            calibration_evidence_ref=decision.get("calibration_evidence_ref"),
        ).to_dict()
    except (TypeError, ValueError, campaign.AcceptRuleMisuse) as exc:
        raise CpuInferenceControllerError("CPU decision cannot be rederived") from exc
    if dict(decision) != expected:
        raise CpuInferenceControllerError(
            "CPU decision/effect differs from the exact paired observations")
    return float(expected["median_relative"])


def _result_disposition(candidate: CpuCandidate, result: Mapping[str, Any],
                        output_root: Path, operation_key: str) -> dict[str, Any]:
    _canonical(result)
    parsed = candidate.parsed_args
    spec = result.get("spec")
    if (set(result) != _CAMPAIGN_RESULT_KEYS
            or result.get("schema") != "epyc.autokernel.campaign_result.v1"
            or not isinstance(spec, Mapping)
            or result.get("campaign_id") != parsed["--campaign-id"]
            or result.get("candidate_id") != candidate.candidate_id
            or spec.get("campaign_id") != parsed["--campaign-id"]
            or spec.get("candidate_id") != candidate.candidate_id
            or spec.get("candidate_ref") != parsed["--candidate"]
            or spec.get("backend") != campaign.BACKEND_CPU
            or spec.get("journal_root") != str(output_root / "campaign-journal")
            or result.get("executed") is not True
            or result.get("screening_only") is not False
            or result.get("non_promotable") is not False
            or result.get("screening_report") is not None
            or result.get("grammar") != "SEARCH RECORD, NOT A CLAIM"
            or not isinstance(result.get("steps"), list)
            or not isinstance(result.get("pairs"), list)):
        raise CpuInferenceControllerError(
            "campaign result differs from the selected full CPU operation")
    if not HASH.fullmatch(operation_key) or operation_key != candidate.operation_key(
            parsed["--campaign-id"]):
        raise CpuInferenceControllerError("campaign result operation binding changed")
    blocks = spec.get("blocks_precommitted")
    if (isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2
            or "--blocks" in parsed and blocks != int(parsed["--blocks"])
            or "--reps" in parsed and spec.get("reps") != int(parsed["--reps"])
            or spec.get("model") != parsed["--model"]):
        raise CpuInferenceControllerError(
            "campaign result spec differs from normalized candidate arguments")
    state = result.get("state")
    decision = result.get("decision")
    if state == campaign.STATE_DECIDED:
        t0 = result.get("t0")
        if (not isinstance(decision, Mapping) or set(decision) != _DECISION_KEYS
                or not isinstance(decision.get("keep"), bool)
                or not result["pairs"]
                or result.get("journal_error") is not None
                or result.get("ok") is not True
                or result.get("error") is not None):
            raise CpuInferenceControllerError("decided CPU result lacks an exact decision")
        gates = _t0_gates(t0)
        if any(row[1] != campaign.schemas.PASS for row in gates):
            raise CpuInferenceControllerError("decided CPU result lacks all-PASS T0")
        effect = _exact_decision(spec, t0, result["pairs"], decision)
        classification = "candidate" if decision["keep"] else "screened_out"
        science = True
        keep: bool | None = decision["keep"]
    elif state == campaign.STATE_T0_FAILED:
        t0 = result.get("t0")
        if (decision is not None
                or result["pairs"]
                or result.get("journal_error") is not None
                or result.get("ok") is not True):
            raise CpuInferenceControllerError("T0 failure unexpectedly carries speed decision")
        gates = _t0_gates(t0)
        outcomes = {row[1] for row in gates}
        if campaign.schemas.COULD_NOT_CHECK in outcomes:
            classification = "infrastructure_ambiguous"
            science = False
            keep = None
        elif campaign.schemas.FAIL in outcomes:
            if result.get("error") is not None:
                raise CpuInferenceControllerError(
                    "correctness-falsified T0 carries an unrelated error")
            classification = "correctness_falsified"
            science = True
            keep = False
        elif outcomes == {campaign.schemas.PASS} and isinstance(
                result.get("error"), str) and result["error"].strip():
            # T0 itself passed, but the governed evaluator admission to timing
            # refused.  This is a typed infrastructure terminal, never a
            # correctness falsification and never science.
            classification = "infrastructure_ambiguous"
            science = False
            keep = None
        else:
            raise CpuInferenceControllerError("T0 failure has no typed failure")
        effect = None
    elif state in {campaign.STATE_PREFLIGHT_REFUSED, campaign.STATE_ERROR}:
        if (decision is not None or result["pairs"]
                or state == campaign.STATE_ERROR and result.get("ok") is not False):
            raise CpuInferenceControllerError(
                "infrastructure terminal unexpectedly carries speed decision")
        classification = "infrastructure_ambiguous"
        science = False
        keep = None
        effect = None
    else:
        raise CpuInferenceControllerError(
            f"executing CPU campaign returned inadmissible state {state!r}")
    releases = result.get("releases")
    if (not isinstance(releases, list) or any(
            not isinstance(row, Mapping)
            or set(row) != {"name", "released", "detail"}
            or not isinstance(row.get("name"), str) or not row["name"].strip()
            or row.get("released") is not True
            or not isinstance(row.get("detail"), str)
            for row in releases)
            or len({row["name"] for row in releases}) != len(releases)):
        raise CpuInferenceControllerError(
            "campaign result does not prove release of every acquired resource")
    if state in {campaign.STATE_DECIDED, campaign.STATE_T0_FAILED}:
        names = {row.get("name") for row in releases}
        if not {"cpu_region_claim", "campaign_worktree"} <= names:
            raise CpuInferenceControllerError(
                "CPU post-claim terminal lacks claim/worktree release evidence")
    unchanged = result.get("production_unchanged")
    if (not isinstance(unchanged, Mapping)
            or set(unchanged) != {"outcome", "reasons"}
            or unchanged.get("outcome") != campaign.schemas.PASS
            or not isinstance(unchanged.get("reasons"), list)
            or any(not isinstance(reason, str) for reason in unchanged["reasons"])):
        raise CpuInferenceControllerError(
            "CPU disposition lacks production immutability PASS")
    return {
        "status": state, "classification": classification,
        "scientific_budget_spent": science, "keep": keep,
        "effect_fraction": effect,
    }


def _write_receipt(root: Path, candidate: CpuCandidate,
                   operation_key: str,
                   result: Mapping[str, Any]) -> tuple[Path, str, str]:
    directory = root / "receipts"
    _validate_controller_path(directory, "CPU receipt directory", root=root)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_sha = _sha(result)
    body = {
        "schema": RECEIPT_SCHEMA, "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate.candidate_sha256,
        "operation_key": operation_key,
        "campaign_result_sha256": result_sha, "campaign_result": dict(result),
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    path = directory / f"{candidate.candidate_id}.json"
    _validate_controller_path(path, "CPU result receipt", root=root)
    encoded = json.dumps(receipt, sort_keys=True, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
    if path.exists():
        existing = _read_regular(path, "CPU result receipt")[0]
        if existing != encoded:
            raise CpuInferenceControllerError(
                "existing CPU result receipt differs from sealed operation")
    else:
        with path.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    file_sha = hashlib.sha256(encoded).hexdigest()
    return path, file_sha, result_sha


def _load_receipt(path: Path, candidate: CpuCandidate,
                  operation_key: str) -> Mapping[str, Any]:
    raw = _read_regular(path, "CPU result receipt")[0]
    value = _strict_json(raw, "CPU result receipt")
    if not isinstance(value, Mapping):
        raise CpuInferenceControllerError("CPU result receipt is not an object")
    required = {"schema", "candidate_id", "candidate_sha256", "operation_key",
                "campaign_result_sha256", "campaign_result", "receipt_sha256"}
    body = {key: value[key] for key in required - {"receipt_sha256"}} \
        if set(value) == required else None
    if (body is None or value.get("schema") != RECEIPT_SCHEMA
            or value.get("candidate_id") != candidate.candidate_id
            or value.get("candidate_sha256") != candidate.candidate_sha256
            or value.get("operation_key") != operation_key
            or not isinstance(value.get("campaign_result"), Mapping)
            or value.get("campaign_result_sha256") != _sha(value["campaign_result"])
            or value.get("receipt_sha256") != _sha(body)):
        raise CpuInferenceControllerError("CPU result receipt binding changed")
    return value["campaign_result"]


def _journal_terminal(config: ControllerManifest,
                      candidate: CpuCandidate,
                      operation_key: str) -> Mapping[str, Any] | None:
    book = journal.Journal(
        str(config.output_root / "campaign-journal"),
        campaign_id=config.controller_id)
    if not Path(book.root).exists():
        return None
    entries = [entry for entry in book.read_all()
               if entry.kind == journal.KIND_STOP_STATE
               and isinstance(entry.payload, Mapping)
               and entry.payload.get("campaign_id") == config.controller_id
               and isinstance(entry.payload.get("result"), Mapping)
               and entry.payload["result"].get("candidate_id") == candidate.candidate_id]
    if not entries:
        return None
    for entry in entries:
        _result_disposition(candidate, entry.payload["result"],
                            config.output_root, operation_key)
    hashes = {_sha(entry.payload["result"]) for entry in entries}
    if len(hashes) != 1:
        raise CpuInferenceControllerError(
            "campaign journal contains conflicting terminal results")
    return entries[-1].payload["result"]


def _iteration(candidate: CpuCandidate, disposition: Mapping[str, Any], *,
               operation_key: str,
               receipt_path: Path | None, receipt_file_sha256: str | None,
               result_sha256: str | None, reason_code: str | None = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "hypothesis_id": candidate.hypothesis_id,
        "candidate_sha256": candidate.candidate_sha256,
        "operation_key": operation_key,
        "status": disposition["status"],
        "classification": disposition["classification"],
        "scientific_budget_spent": disposition["scientific_budget_spent"],
        "keep": disposition["keep"],
        "effect_fraction": disposition.get("effect_fraction"),
        "receipt_path": None if receipt_path is None else str(receipt_path),
        "receipt_file_sha256": receipt_file_sha256,
        "result_sha256": result_sha256,
        "reason_code": reason_code,
        "completed_at": _now(),
    }


def _append_iteration(state: dict[str, Any], row: Mapping[str, Any]) -> None:
    state["iterations"].append(dict(row))
    state["next_index"] += 1
    state["scientific_attempts"] = sum(
        1 for item in state["iterations"]
        if item.get("scientific_budget_spent") is True)
    state["inflight"] = None


def _finish_terminal(state: dict[str, Any], config: ControllerManifest) -> None:
    if state["scientific_attempts"] >= config.max_scientific_attempts:
        state["complete"] = True
        state["terminal_reason"] = "scientific_budget_exhausted"
    elif state["next_index"] >= len(config.candidates):
        state["complete"] = True
        state["terminal_reason"] = "candidate_portfolio_exhausted"


def run_controller(config: ControllerManifest, *,
                   runner: CampaignRunner | None = None) -> dict[str, Any]:
    """Run the ordered CPU portfolio; restart never replays sealed science."""
    storage.assert_not_scratch(
        config.output_root, what="CPU inference controller evidence root")
    config.revalidate()
    selected_runner = runner or InProcessCampaignRunner()
    store = StateStore(config)
    lock = store.lock()
    try:
        state = store.load()
        if state["complete"]:
            return state

        # A process may have died after campaign journaling but before the
        # controller receipt/state checkpoint.  Reconcile exact private receipt
        # first, then the campaign's append-only terminal.  If neither exists,
        # the statistical key is ambiguous and is never re-run.
        if state["inflight"] is not None:
            inflight = state["inflight"]
            index = inflight.get("candidate_index") if isinstance(inflight, Mapping) else None
            if (isinstance(index, bool) or not isinstance(index, int)
                    or index != state["next_index"]
                    or not 0 <= index < len(config.candidates)):
                raise CpuInferenceControllerError("inflight candidate checkpoint is malformed")
            candidate = config.candidates[index]
            operation_key = candidate.operation_key(config.controller_id)
            if (inflight.get("candidate_sha256") != candidate.candidate_sha256
                    or inflight.get("candidate_id") != candidate.candidate_id
                    or inflight.get("operation_key") != operation_key):
                raise CpuInferenceControllerError("inflight candidate identity changed")
            receipt_path = config.output_root / "receipts" / f"{candidate.candidate_id}.json"
            _validate_controller_path(
                receipt_path, "CPU result receipt", root=config.output_root)
            if receipt_path.exists():
                result = _load_receipt(receipt_path, candidate, operation_key)
            else:
                result = _journal_terminal(config, candidate, operation_key)
            if result is not None:
                disposition = _result_disposition(
                    candidate, result, config.output_root, operation_key)
                path, file_sha, result_sha = _write_receipt(
                    config.output_root, candidate, operation_key, result)
                _append_iteration(state, _iteration(
                    candidate, disposition, operation_key=operation_key,
                    receipt_path=path,
                    receipt_file_sha256=file_sha, result_sha256=result_sha))
                _finish_terminal(state, config)
                store.save(state, "reconciled_terminal")
            else:
                disposition = {
                    "status": "interrupted_without_terminal",
                    "classification": "infrastructure_ambiguous",
                    "scientific_budget_spent": False, "keep": None,
                    "effect_fraction": None,
                }
                _append_iteration(state, _iteration(
                    candidate, disposition, operation_key=operation_key,
                    receipt_path=None,
                    receipt_file_sha256=None, result_sha256=None,
                    reason_code="inflight_operation_has_no_sealed_terminal"))
                _finish_terminal(state, config)
                store.save(state, "interrupted_ambiguous")

        while (not state["complete"]
               and state["scientific_attempts"] < config.max_scientific_attempts
               and state["next_index"] < len(config.candidates)):
            index = state["next_index"]
            candidate = config.candidates[index]
            candidate.revalidate(config.output_root)
            operation_key = candidate.operation_key(config.controller_id)
            state["inflight"] = {
                "candidate_index": index, "candidate_id": candidate.candidate_id,
                "candidate_sha256": candidate.candidate_sha256,
                "operation_key": operation_key,
                "started_at": _now(),
            }
            store.save(state, "candidate_started")
            args = (*candidate.campaign_args,
                    "--execute", "--i-hold-the-host", "--json")
            code, result = selected_runner.run(args)
            if result is None:
                disposition = {
                    "status": "campaign_entrypoint_refused",
                    "classification": "infrastructure_ambiguous",
                    "scientific_budget_spent": False, "keep": None,
                    "effect_fraction": None,
                }
                _append_iteration(state, _iteration(
                    candidate, disposition, operation_key=operation_key,
                    receipt_path=None,
                    receipt_file_sha256=None, result_sha256=None,
                    reason_code=f"campaign_exit_{code}_without_result"))
                _finish_terminal(state, config)
                store.save(state, "entrypoint_refused")
                continue
            disposition = _result_disposition(
                candidate, result, config.output_root, operation_key)
            if code not in {0, 1}:
                raise CpuInferenceControllerError(
                    "campaign returned a result under an inadmissible exit status")
            if (code == 0) != bool(result.get("ok")):
                raise CpuInferenceControllerError(
                    "campaign exit status disagrees with result.ok")
            path, file_sha, result_sha = _write_receipt(
                config.output_root, candidate, operation_key, result)
            _append_iteration(state, _iteration(
                candidate, disposition, operation_key=operation_key,
                receipt_path=path,
                receipt_file_sha256=file_sha, result_sha256=result_sha))
            _finish_terminal(state, config)
            store.save(state, "candidate_terminal")
        _finish_terminal(state, config)
        if state["complete"] and state["state_sha256"] is None:
            store.save(state, "complete")
        return state
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--i-hold-the-host", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ControllerManifest.load(Path(args.manifest))
        if not args.execute:
            if args.i_hold_the_host:
                raise CpuInferenceControllerError(
                    "--i-hold-the-host has no meaning without --execute")
            print(json.dumps({
                "status": "validated", "inference_executed": False,
                "controller_id": config.controller_id,
                "manifest_sha256": config.manifest_sha256,
                "candidate_count": len(config.candidates),
            }, sort_keys=True))
            return 0
        if not args.i_hold_the_host:
            raise CpuInferenceControllerError(
                "--execute requires --i-hold-the-host")
        state = run_controller(config)
        print(json.dumps({
            "status": "complete" if state["complete"] else "stopped",
            "inference_executed": True,
            "state_sha256": state["state_sha256"],
            "scientific_attempts": state["scientific_attempts"],
            "terminal_reason": state["terminal_reason"],
        }, sort_keys=True))
        return 0 if state["complete"] else 1
    except (CpuInferenceControllerError, OSError, ValueError, TypeError) as exc:
        print(f"refusing CPU inference controller: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
