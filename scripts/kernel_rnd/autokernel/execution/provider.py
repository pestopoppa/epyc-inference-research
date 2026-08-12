#!/usr/bin/env python3
"""Fail-closed identity and installation roots for external kernel providers.

Provider implementations may be useful search oracles without being eligible
AutoKernel source.  This module owns the filesystem half of that boundary: a
provider prefix is an isolated candidate location, never a shared system ROCm
installation and never one of the frozen production kernel trees.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import schemas
from . import sandbox
from . import worktree


class ProviderIsolationError(ValueError):
    """A provider prefix could alter shared or production state."""


class ProviderBuildError(RuntimeError):
    """A source provider could not be acquired or built with governed identity."""


SOURCE_ACQUISITION_SCHEMA = "epyc.autokernel.provider_source_acquisition.v1"
SOURCE_BUILD_PLAN_SCHEMA = "epyc.autokernel.provider_source_build_plan.v1"
SOURCE_BUILD_RECEIPT_SCHEMA = "epyc.autokernel.provider_source_build_receipt.v1"
SOURCE_BUILD_AUTHORITY = (
    "provider_candidate_only_requires_clean_llama_gpu_integration")


PROHIBITED_PROVIDER_PREFIXES = (
    "/opt/rocm",
    "/usr",
    *worktree.PRODUCTION_TREES,
    *worktree.PRODUCTION_TREE_ALIASES,
)


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


@dataclass(frozen=True)
class IsolatedProviderPrefix:
    """An absolute provider root proven outside shared and frozen trees."""

    path: str

    @classmethod
    def create(cls, path: str, *, prohibited: Iterable[str] =
               PROHIBITED_PROVIDER_PREFIXES) -> "IsolatedProviderPrefix":
        if not isinstance(path, str) or not path or not os.path.isabs(path):
            raise ProviderIsolationError("provider isolation root must be absolute")
        resolved = os.path.realpath(path)
        if resolved == os.path.sep:
            raise ProviderIsolationError("filesystem root is not an isolated provider prefix")
        blocked = tuple(os.path.realpath(item) for item in prohibited)
        matches = tuple(root for root in blocked
                        if _under(resolved, root) or _under(root, resolved))
        if matches:
            raise ProviderIsolationError(
                f"provider isolation root {resolved!r} overlaps prohibited prefix "
                f"{matches[0]!r}")
        return cls(resolved)

    def child(self, *parts: str) -> Path:
        child = Path(self.path, *parts).resolve(strict=False)
        if not _under(str(child), self.path):
            raise ProviderIsolationError("provider child path escapes its isolated prefix")
        return child


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(source_root), *args), text=True,
        capture_output=True, check=False)
    if completed.returncode:
        raise ProviderBuildError(
            f"provider source git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _tracked_source_manifest(source_root: Path) -> tuple[dict[str, str], str]:
    raw = subprocess.run(
        ("git", "-C", str(source_root), "ls-files", "-z"),
        capture_output=True, check=False)
    if raw.returncode:
        raise ProviderBuildError("provider source tracked-file enumeration failed")
    rows: dict[str, str] = {}
    for encoded in raw.stdout.split(b"\0"):
        if not encoded:
            continue
        path_text = encoded.decode("utf-8")
        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProviderBuildError("provider source contains an escaping tracked path")
        path = (source_root / relative).resolve(strict=True)
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ProviderBuildError(
                f"provider tracked file escapes or is not regular: {path_text}")
        rows[path_text] = _file_sha256(path)
    if not rows:
        raise ProviderBuildError("provider source has no tracked files")
    ordered = dict(sorted(rows.items()))
    return ordered, _canonical_sha256(ordered)


def acquire_source(source_root: str, *, source_commit: str,
                   artifact_sha256: str) -> dict[str, Any]:
    """Verify an already-present source checkout; never clone or contact a network."""
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir() or not Path(root, ".git").exists():
        raise ProviderBuildError("provider source must be an existing Git checkout")
    for prohibited in ("/opt/rocm", *worktree.PRODUCTION_TREES,
                       *worktree.PRODUCTION_TREE_ALIASES):
        blocked = os.path.realpath(prohibited)
        if _under(str(root), blocked) or _under(blocked, str(root)):
            raise ProviderBuildError(
                f"provider source overlaps shared or frozen tree {blocked!r}")
    if _git(root, "rev-parse", "HEAD") != source_commit:
        raise ProviderBuildError("provider source HEAD differs from the declared commit")
    if _git(root, "status", "--porcelain"):
        raise ProviderBuildError("provider source checkout is not clean")
    files, observed_artifact = _tracked_source_manifest(root)
    if observed_artifact != artifact_sha256:
        raise ProviderBuildError(
            "provider source artifact digest differs from its tracked-file manifest")
    receipt = {
        "schema": SOURCE_ACQUISITION_SCHEMA,
        "source_root": str(root), "source_commit": source_commit,
        "artifact_sha256": observed_artifact, "tracked_files": files,
        "network_accessed": False, "source_modified": False,
        "authority": SOURCE_BUILD_AUTHORITY,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def source_artifact_sha256(source_root: str) -> str:
    """Return the canonical tracked-file digest expected by a provider reference."""
    root = Path(source_root).resolve(strict=True)
    return _tracked_source_manifest(root)[1]


def _manifest_identity(path: str, expected_sha256: str, label: str) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ProviderBuildError(f"{label} must be a regular file")
    observed = _file_sha256(resolved)
    if observed != expected_sha256:
        raise ProviderBuildError(f"{label} digest differs from provider reference")
    return {"path": str(resolved), "sha256": observed}


@dataclass(frozen=True)
class SourceProviderBuildPlan:
    provider_reference: Mapping[str, Any]
    provider_reference_sha256: str
    source_acquisition: Mapping[str, Any]
    isolation_root: str
    argv: tuple[str, ...]
    executable: Mapping[str, str]
    expected_outputs: tuple[str, ...]
    toolchain_manifest: Mapping[str, str]
    linkage_manifest: Mapping[str, str]
    license_file: Mapping[str, str]
    plan_sha256: str = ""

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_BUILD_PLAN_SCHEMA,
            "provider_reference": dict(self.provider_reference),
            "provider_reference_sha256": self.provider_reference_sha256,
            "source_acquisition": dict(self.source_acquisition),
            "isolation_root": self.isolation_root,
            "argv": list(self.argv),
            "executable": dict(self.executable),
            "expected_outputs": list(self.expected_outputs),
            "toolchain_manifest": dict(self.toolchain_manifest),
            "linkage_manifest": dict(self.linkage_manifest),
            "license_file": dict(self.license_file),
            "network_allowed": False,
            "shared_rocm_mutation_allowed": False,
            "production_tree_mutation_allowed": False,
            "authority": SOURCE_BUILD_AUTHORITY,
        }

    def __post_init__(self) -> None:
        observed = _canonical_sha256(self._unsigned_dict())
        if self.plan_sha256 and self.plan_sha256 != observed:
            raise ProviderBuildError("provider build plan digest differs from its fields")
        object.__setattr__(self, "plan_sha256", observed)

    def to_dict(self) -> dict[str, Any]:
        document = self._unsigned_dict()
        observed = _canonical_sha256(document)
        if observed != self.plan_sha256:
            raise ProviderBuildError("provider build plan mutated after compilation")
        document["plan_sha256"] = self.plan_sha256
        return document


def compile_source_build(
    provider_reference: Mapping[str, Any], *, source_root: str,
    argv: Sequence[str], expected_outputs: Sequence[str],
    toolchain_manifest: str, linkage_manifest: str, license_file: str,
) -> SourceProviderBuildPlan:
    """Compile a no-network, isolated build plan for source-available ROCm code."""
    reference = json.loads(json.dumps(provider_reference))
    violations = schemas.validate_provider_reference(reference)
    if violations:
        raise ProviderBuildError("invalid provider reference: " + "; ".join(violations))
    if (reference["source_mode"] != "source"
            or reference["evidence_authority"] != "candidate_eligible"
            or reference["target_backend"] != "llama_gpu"
            or reference["kind"] not in {
                "rocm_library", "third_party_source", "compiler_toolchain"}):
        raise ProviderBuildError(
            "source build requires a candidate-eligible external provider integrated via llama_gpu")
    prefix = IsolatedProviderPrefix.create(reference["isolation_root"])
    if Path(prefix.path).exists():
        raise ProviderBuildError("provider isolation root must not exist before its build")
    acquisition = acquire_source(
        source_root, source_commit=reference["source_commit"],
        artifact_sha256=reference["artifact_sha256"])
    source_path = Path(acquisition["source_root"])
    if (_under(prefix.path, str(source_path))
            or _under(str(source_path), prefix.path)):
        raise ProviderBuildError("provider prefix and source checkout must not overlap")
    command = tuple(argv)
    if (not command or not os.path.isabs(command[0])
            or any(not isinstance(item, str) or not item or "\0" in item
                   for item in command)):
        raise ProviderBuildError(
            "provider build argv must be non-empty strings with an absolute executable")
    if prefix.path not in command:
        raise ProviderBuildError(
            "provider build argv must carry the exact isolated prefix as one argument")
    executable_path = Path(command[0]).resolve(strict=True)
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ProviderBuildError("provider build executable is not an executable regular file")
    outputs = tuple(expected_outputs)
    if not outputs or len(outputs) != len(set(outputs)):
        raise ProviderBuildError("provider build requires unique expected outputs")
    for item in outputs:
        relative = Path(item)
        if (not isinstance(item, str) or not item or relative.is_absolute()
                or ".." in relative.parts or str(relative) in {".", ""}):
            raise ProviderBuildError("provider expected output must stay under its prefix")
    toolchain = _manifest_identity(
        toolchain_manifest, reference["toolchain_manifest_sha256"],
        "toolchain manifest")
    linkage = _manifest_identity(
        linkage_manifest, reference["linkage_manifest_sha256"],
        "linkage manifest")
    license_identity = {
        "path": str(Path(license_file).resolve(strict=True)),
        "sha256": _file_sha256(Path(license_file).resolve(strict=True)),
        "declared_check": reference["license_check"],
    }
    license_path = Path(license_identity["path"])
    if (not license_path.is_relative_to(source_path)
            or str(license_path.relative_to(source_path))
            not in acquisition["tracked_files"]):
        raise ProviderBuildError("provider license file must be tracked inside source")
    return SourceProviderBuildPlan(
        provider_reference=reference,
        provider_reference_sha256=_canonical_sha256(reference),
        source_acquisition=acquisition, isolation_root=prefix.path,
        argv=command,
        executable={"path": str(executable_path),
                    "sha256": _file_sha256(executable_path)},
        expected_outputs=outputs,
        toolchain_manifest=toolchain, linkage_manifest=linkage,
        license_file=license_identity)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execute_source_build(
    plan: SourceProviderBuildPlan, *, receipt_root: str,
    authorize_build: bool = False, timeout_seconds: float = 3600,
) -> dict[str, Any]:
    """Execute an exact plan inside the no-network write-confined sandbox."""
    if not isinstance(plan, SourceProviderBuildPlan):
        raise TypeError("plan must be a SourceProviderBuildPlan")
    if authorize_build is not True:
        raise ProviderBuildError("provider build requires explicit authorize_build=True")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0):
        raise ProviderBuildError("provider build timeout must be positive")
    prefix = Path(plan.isolation_root)
    if prefix.exists():
        raise ProviderBuildError("provider isolation root appeared before execution")
    if not prefix.parent.is_dir():
        raise ProviderBuildError("provider isolation parent does not exist")
    receipts = Path(receipt_root).resolve(strict=True)
    if not receipts.is_dir() or receipts == prefix or receipts.is_relative_to(prefix):
        raise ProviderBuildError("provider receipt root must exist outside the candidate prefix")
    source_path = Path(plan.source_acquisition["source_root"])
    if receipts == source_path or receipts.is_relative_to(source_path) \
            or source_path.is_relative_to(receipts):
        raise ProviderBuildError("provider receipt root and source checkout must not overlap")
    executable_path = Path(plan.argv[0]).resolve(strict=True)
    if ({"path": str(executable_path), "sha256": _file_sha256(executable_path)}
            != dict(plan.executable)):
        raise ProviderBuildError("provider build executable identity drifted")
    plan.to_dict()
    prefix.mkdir(mode=0o700)
    Path(prefix, "tmp").mkdir(mode=0o700)
    policy = sandbox.SandboxPolicy(
        writable_root=str(prefix), writable_device_paths=("/dev/null",))
    sandbox_receipt = receipts / "provider-build-sandbox.json"
    stdout_path = receipts / "provider-build.stdout"
    stderr_path = receipts / "provider-build.stderr"
    started = _utc_now()
    env = os.environ.copy()
    env.update({"AK_PROVIDER_PREFIX": str(prefix), "TMPDIR": str(prefix / "tmp")})
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    teardown: dict[str, Any] | None = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            policy.wrap(plan.argv, receipt_path=str(sandbox_receipt)),
            cwd=plan.source_acquisition["source_root"], env=env,
            stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            returncode = process.wait(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=5)
        finally:
            teardown = sandbox.cleanup_cgroup(policy, process.pid)
    if process.poll() is None:
        raise ProviderBuildError("captured provider build PID remained alive after teardown")
    ended = _utc_now()
    sandbox_document = sandbox.read_receipt(sandbox_receipt)
    sandbox.verify_receipt(sandbox_document, policy=policy, argv=plan.argv,
                           pid=process.pid)
    source_after = acquire_source(
        plan.source_acquisition["source_root"],
        source_commit=plan.provider_reference["source_commit"],
        artifact_sha256=plan.provider_reference["artifact_sha256"])
    outputs = []
    output_errors = []
    if returncode == 0 and not timed_out:
        for relative_text in plan.expected_outputs:
            relative = Path(relative_text)
            try:
                output = (prefix / relative).resolve(strict=True)
            except OSError:
                output_errors.append(
                    f"provider build omitted expected output {relative_text!r}")
                continue
            if not output.is_relative_to(prefix) or not output.is_file():
                output_errors.append(
                    f"provider build output is not a confined regular file {relative_text!r}")
                continue
            outputs.append({"path": relative_text, "sha256": _file_sha256(output),
                            "size_bytes": output.stat().st_size})
    complete = returncode == 0 and not timed_out and not output_errors
    receipt = {
        "schema": SOURCE_BUILD_RECEIPT_SCHEMA,
        "status": "complete" if complete else "failed",
        "started_at": started, "ended_at": ended,
        "plan": plan.to_dict(), "source_after": source_after,
        "sandbox_receipt": {"path": str(sandbox_receipt),
                            "sha256": _file_sha256(sandbox_receipt)},
        "teardown": teardown, "pid": process.pid, "returncode": returncode,
        "timed_out": timed_out,
        "stdout": {"path": str(stdout_path), "sha256": _file_sha256(stdout_path)},
        "stderr": {"path": str(stderr_path), "sha256": _file_sha256(stderr_path)},
        "outputs": outputs, "output_errors": output_errors,
        "network_accessed": False,
        "shared_rocm_mutated": False, "production_tree_mutated": False,
        "candidate_bankable_without_llama_gpu_integration": False,
        "authority": SOURCE_BUILD_AUTHORITY,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


__all__ = [
    "IsolatedProviderPrefix", "PROHIBITED_PROVIDER_PREFIXES",
    "ProviderBuildError", "ProviderIsolationError", "SOURCE_ACQUISITION_SCHEMA",
    "SOURCE_BUILD_AUTHORITY", "SOURCE_BUILD_PLAN_SCHEMA",
    "SOURCE_BUILD_RECEIPT_SCHEMA", "SourceProviderBuildPlan", "acquire_source",
    "compile_source_build", "execute_source_build", "source_artifact_sha256",
]
