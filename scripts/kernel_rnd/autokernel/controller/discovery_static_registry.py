"""Static, non-JSON construction of the GPU source discovery build seam.

This is deliberately the only source builder that the deployment launcher may
register.  It uses the typed worktree/source-candidate/build APIs: no actor
path, argv, CMake flag, or production path is accepted from planner output.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Any, Mapping

from .. import source_candidate
from .. import schemas
from ..evaluator import integrity
from ..execution import t0_provider, worktree
from . import discovery_controller as controller
from . import gpu_source_evidence as evidence
from . import gpu_source_proofs
from . import split_runtime_verifier


class StaticRegistryError(RuntimeError):
    pass


_REWARD_FILES = ("llama-bench", "libllama-bench-impl.so", "libllama-common.so",
                 "libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so")
_HIP = "libggml-hip.so"
_BUILDER_SCHEMA = "epyc.autokernel.static_gpu_source_builder.v3"
_BUILD_KEY_SCHEMA = "epyc.autokernel.gpu_source_build_key.v1"
_BUILD_REF_SCHEMA = "epyc.autokernel.gpu_source_build_ref.v1"
_BUILD_INTENT_SCHEMA = "epyc.autokernel.gpu_source_build_intent.v1"
_BUILD_TERMINAL_SCHEMA = "epyc.autokernel.gpu_source_build_terminal.v1"
_TEARDOWN_SCHEMA = "epyc.autokernel.source_materialization_teardown.v2"
_SOURCE_FAILURE_MESSAGE_MAX_BYTES = 2048
_REQUIRED_TARGETS = ("llama-bench", "test-backend-ops")
_CORRECTNESS_CAPABILITY_SCHEMA = "epyc.autokernel.backend_ops_property_capability.v1"
_CORRECTNESS_CAPABILITY_SEED = 2026081301
_BUILD_ENV_NAMES = (
    "PATH", "HOME", "LANG", "LC_ALL", "ROCM_PATH", "HIP_PATH",
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH", "PKG_CONFIG_PATH", "CMAKE_PREFIX_PATH", "CC",
    "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "MAKEFLAGS",
)
_SEALED_BUILD_PATH = "/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin"


def _digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StaticRegistryError(f"runtime artifact is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_failure_message(value: object) -> str | None:
    """Return a bounded one-line diagnostic safe for a sealed terminal."""
    if not isinstance(value, str) or not value or not value.isprintable():
        return None
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return None
    if len(encoded) > _SOURCE_FAILURE_MESSAGE_MAX_BYTES:
        return None
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _regular_directory(path: Path, label: str, *, create: bool = False) -> Path:
    if create and not path.exists() and not path.is_symlink():
        path.mkdir(parents=True)
    if path.is_symlink() or not path.is_dir():
        raise StaticRegistryError(f"{label} must be a regular directory: {path}")
    return path.resolve(strict=True)


def _sealed_write(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    """Create one self-hashed receipt without replacing prior evidence."""
    if path.exists() or path.is_symlink():
        raise StaticRegistryError(f"sealed receipt already exists: {path}")
    parent = _regular_directory(path.parent, "receipt parent")
    payload = dict(body)
    payload["receipt_sha256"] = schemas.content_hash(payload)
    raw = _canonical(payload)
    temporary = parent / f".{path.name}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            # `replace` would silently overwrite a receipt installed between
            # the pre-check and publication.  A hard-link publication is
            # same-filesystem and fails atomically when the destination exists.
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
            raise
    except FileExistsError as exc:
        raise StaticRegistryError(f"concurrent temporary receipt collision: {temporary}") from exc
    return path.resolve(strict=True), hashlib.sha256(raw).hexdigest()


def _sealed_read(path: Path, *, schema: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StaticRegistryError(f"{label} is absent or not a regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise StaticRegistryError(f"{label} must have one regular-file link")
            with os.fdopen(descriptor, "rb") as handle:
                raw = handle.read()
                after = os.fstat(handle.fileno())
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_nlink) !=
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                     after.st_nlink)):
                raise StaticRegistryError(f"{label} changed while being read")
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        body = json.loads(raw.decode("utf-8", "strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticRegistryError(f"{label} is not JSON") from exc
    if not isinstance(body, dict) or body.get("schema") != schema:
        raise StaticRegistryError(f"{label} schema mismatch")
    payload = {key: value for key, value in body.items() if key != "receipt_sha256"}
    if body.get("receipt_sha256") != schemas.content_hash(payload):
        raise StaticRegistryError(f"{label} self-hash mismatch")
    return body


def _within(path: Path, root: Path, label: str, *, directory: bool = False,
            single_link: bool = False) -> Path:
    if path.is_symlink():
        raise StaticRegistryError(f"{label} may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StaticRegistryError(f"{label} escapes its sealed cache root") from exc
    if directory and not resolved.is_dir():
        raise StaticRegistryError(f"{label} is not a directory")
    if not directory and not resolved.is_file():
        raise StaticRegistryError(f"{label} is not a file")
    if not directory and single_link and resolved.stat().st_nlink != 1:
        raise StaticRegistryError(f"{label} must have exactly one hard link")
    return resolved


@contextmanager
def _exclusive_lock(path: Path):
    parent = _regular_directory(path.parent, "build lock parent", create=True)
    target = parent / path.name
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        if target.is_symlink() or not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise StaticRegistryError("build cache lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _resolved_regular(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StaticRegistryError(f"runtime artifact cannot be resolved: {path}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise StaticRegistryError(f"runtime artifact is not a resolved regular file: {path}")
    return resolved


def _identity(*, root: Path, commit: str, build: Path) -> gpu_source_proofs.BuildIdentity:
    binary = build / "bin" / "llama-bench"
    hip = build / "bin" / _HIP
    cache = build / "CMakeCache.txt"
    tree = integrity.hash_source_tree(root, exclude_dir_names=(".git",)).sha256
    hip_real = _resolved_regular(hip)
    linkage = _linkage_sha(build)
    return gpu_source_proofs.BuildIdentity(
        source_commit=commit, source_sha256=tree, binary_sha256=_digest(binary),
        hip_library_sha256=_digest(hip_real), config_sha256=_digest(cache), linkage_sha256=linkage)


def _linkage_body(build: Path) -> dict[str, object]:
    binary = build / "bin" / "llama-bench"
    hip = _resolved_regular(build / "bin" / _HIP)
    topology = {name: os.readlink(build / "bin" / name)
                for name in ("libggml-hip.so", "libggml-hip.so.0")
                if (build / "bin" / name).is_symlink()}
    return {"binary": _digest(binary), "hip": _digest(hip), "topology": topology}


def _linkage_sha(build: Path) -> str:
    return hashlib.sha256(json.dumps(_linkage_body(build), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _write_linkage_carrier(path: Path, build: Path) -> tuple[Path, str]:
    raw = json.dumps(_linkage_body(build), sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _correctness_capability_environment(build: Path) -> dict[str, str]:
    bindir = (build / "bin").resolve(strict=True)
    return {
        "HIP_VISIBLE_DEVICES": "-1",
        "LD_LIBRARY_PATH": f"{bindir}:/opt/rocm/lib",
        "PATH": "/opt/rocm/bin:/usr/bin:/bin",
        "ROCM_PATH": "/opt/rocm",
    }


def _attest_correctness_capability(
        build: Path, *, arm: str,
        runner: Any | None = None) -> dict[str, Any]:
    """Execute the reviewed planted-defect test before a build may be complete."""
    if arm not in {"anchor", "candidate"}:
        raise StaticRegistryError("correctness capability arm is invalid")
    build = build.resolve(strict=True)
    binary = build / "bin" / "test-backend-ops"
    try:
        lexical = binary.lstat()
    except OSError as exc:
        raise StaticRegistryError(f"{arm} correctness binary is absent") from exc
    if (not stat.S_ISREG(lexical.st_mode) or lexical.st_nlink != 1
            or not os.access(binary, os.X_OK) or binary.resolve(strict=True) != binary):
        raise StaticRegistryError(
            f"{arm} correctness binary is not a single-link regular executable")
    hip = _resolved_regular(build / "bin" / _HIP)
    argv = t0_provider.backend_ops_property_self_test_argv(
        str(binary), _CORRECTNESS_CAPABILITY_SEED)
    environment = _correctness_capability_environment(build)
    if runner is None:
        runner = subprocess.run
    try:
        completed = runner(
            argv, check=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, env=environment)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StaticRegistryError(
            f"{arm} correctness capability self-test could not execute") from exc
    try:
        stdout = completed.stdout.decode("utf-8", "strict")
        stderr = completed.stderr.decode("utf-8", "strict")
    except (AttributeError, UnicodeDecodeError) as exc:
        raise StaticRegistryError(
            f"{arm} correctness capability output is unreadable") from exc
    if completed.returncode != 0 or stdout:
        raise StaticRegistryError(
            f"{arm} correctness capability self-test did not return its exact stderr contract")
    try:
        result = t0_provider.parse_backend_ops_property_self_test(
            stderr, expected_suite_seed=_CORRECTNESS_CAPABILITY_SEED)
    except t0_provider.InstrumentCapabilityError as exc:
        raise StaticRegistryError(
            f"{arm} correctness capability self-test failed: {exc}") from exc
    return {
        "schema": _CORRECTNESS_CAPABILITY_SCHEMA,
        "arm": arm,
        "binary": {"path": str(binary), "sha256": _digest(binary)},
        "hip_library": {"path": str(hip), "sha256": _digest(hip)},
        "argv": list(argv),
        "environment": dict(sorted(environment.items())),
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr": stderr,
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "result": {
            "suite_seed": result.suite_seed,
            "sensitivity": result.sensitivity,
            "specificity": result.specificity,
            "planted": result.planted,
            "clean": result.clean,
        },
        "promotion_claim": False,
    }


def _verify_correctness_capability_receipt(
        *, receipt: Path, receipt_sha256: str, binary: Path,
        binary_sha256: str, build: Path, arm: str) -> dict[str, Any]:
    reopened_binary = _within(
        binary, build, f"{arm} correctness binary", single_link=True)
    if (reopened_binary != binary or not os.access(reopened_binary, os.X_OK)
            or _digest(reopened_binary) != binary_sha256):
        raise StaticRegistryError(f"{arm} correctness binary identity changed")
    if _digest(receipt) != receipt_sha256:
        raise StaticRegistryError(f"{arm} correctness capability receipt changed")
    body = _sealed_read(
        receipt, schema=_CORRECTNESS_CAPABILITY_SCHEMA,
        label=f"{arm} correctness capability receipt")
    expected_argv = list(t0_provider.backend_ops_property_self_test_argv(
        str(binary), _CORRECTNESS_CAPABILITY_SEED))
    hip = _resolved_regular(build / "bin" / _HIP)
    expected = {
        "arm": arm,
        "binary": {"path": str(binary), "sha256": binary_sha256},
        "hip_library": {"path": str(hip), "sha256": _digest(hip)},
        "argv": expected_argv,
        "environment": dict(sorted(_correctness_capability_environment(build).items())),
        "exit_code": 0,
        "stdout": "",
        "stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "stderr_sha256": hashlib.sha256(str(body.get("stderr", "")).encode()).hexdigest(),
        "promotion_claim": False,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise StaticRegistryError(f"{arm} correctness capability receipt identity mismatch")
    try:
        parsed = t0_provider.parse_backend_ops_property_self_test(
            str(body.get("stderr", "")), expected_suite_seed=_CORRECTNESS_CAPABILITY_SEED)
    except t0_provider.InstrumentCapabilityError as exc:
        raise StaticRegistryError(
            f"{arm} correctness capability receipt is not a passing self-test") from exc
    if body.get("result") != {
            "suite_seed": parsed.suite_seed, "sensitivity": parsed.sensitivity,
            "specificity": parsed.specificity, "planted": parsed.planted,
            "clean": parsed.clean}:
        raise StaticRegistryError(f"{arm} correctness capability result changed")
    return body


def _bound(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path.resolve()), "sha256": _digest(path)}


def _instrument_authority(*, instrument_path: Path, production_commit: str,
                          instrument_branch: str, instrument_commit: str) -> dict[str, str]:
    """Prove the instrument ref and its *object tree*, without checking it out.

    ``Anchor.fingerprint`` deliberately protects the frozen checkout from the
    worktree operations below.  It is not an identity for a descendant commit:
    the frozen checkout remains at the production commit.  This separate
    listing digest binds the exact approved instrument object that snapshots
    and builds start from.
    """
    root = instrument_path.resolve(strict=True)
    def git(*argv: str, binary: bool = False) -> str | bytes:
        result = subprocess.run(("git", "-C", str(root), *argv),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise StaticRegistryError(
                f"could not verify sealed measurement instrument Git object: {' '.join(argv)}")
        return result.stdout if binary else result.stdout.decode("utf-8", "strict").strip()
    branch_commit = git("rev-parse", f"refs/heads/{instrument_branch}")
    if branch_commit != instrument_commit:
        raise StaticRegistryError("instrument branch no longer resolves to its sealed commit")
    # Both the commit object and the tree must exist locally.  Listing every
    # tracked entry gives a SHA-256 fingerprint independent from the frozen
    # checkout's TreeFingerprint and is stable across worktree locations.
    git("cat-file", "-e", f"{instrument_commit}^{{commit}}")
    tree = git("rev-parse", f"{instrument_commit}^{{tree}}")
    listing = git("ls-tree", "-r", "-z", "--full-tree", instrument_commit, binary=True)
    assert isinstance(listing, bytes)
    ancestor = subprocess.run(("git", "-C", str(root), "merge-base", "--is-ancestor",
                               production_commit, instrument_commit), stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if ancestor.returncode:
        raise StaticRegistryError("instrument commit is not a descendant of frozen production")
    body = {"schema": "epyc.autokernel.measurement_instrument_authority.v1",
            "production_base_commit": production_commit,
            "instrument_branch": instrument_branch,
            "instrument_commit": instrument_commit,
            "instrument_tree": tree,
            "tree_listing_sha256": hashlib.sha256(listing).hexdigest()}
    body["authority_sha256"] = schemas.content_hash(body)
    return body


def _verify_selected_gpu_blobs(*, production_path: Path, production_commit: str,
                               instrument_path: Path, instrument_commit: str,
                               paths: tuple[str, ...]) -> dict[str, str]:
    """Prove a planner-visible GPU patch has the same base in both eras.

    The instrument adds deterministic measurement support but is not permitted
    to silently change a source surface the planner was briefed from frozen
    production.  A reviewed divergent template would need a separate explicit
    authority mechanism; none is admitted by this builder today.
    """
    if not paths:
        raise StaticRegistryError("selected GPU source scope is empty")
    result: dict[str, str] = {}
    for path in paths:
        if not path.startswith("ggml/src/ggml-cuda/"):
            raise StaticRegistryError("static builder only admits reviewed GPU kernel files")
        def blob(repo: Path, commit: str) -> bytes:
            call = subprocess.run(("git", "-C", str(repo), "show", f"{commit}:{path}"),
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, check=False)
            if call.returncode:
                raise StaticRegistryError(f"selected GPU path lacks sealed base blob: {path}")
            return call.stdout
        production = blob(production_path.resolve(strict=True), production_commit)
        instrument = blob(instrument_path.resolve(strict=True), instrument_commit)
        if production != instrument:
            raise StaticRegistryError(
                f"instrument GPU source diverges from frozen production for {path}; explicit review required")
        result[path] = hashlib.sha256(production).hexdigest()
    return result


def _boot_id(proc_root: Path) -> str:
    value = (proc_root / "sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    if not value:
        raise StaticRegistryError("kernel boot identity is unavailable")
    return value


def _start_ticks(proc_root: Path, pid: int) -> int:
    # /proc/<pid>/stat's second field may contain spaces/parens.  The final
    # right-paren unambiguously starts fields 3..; starttime is field 22.
    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    tail = raw.rsplit(")", 1)[1].split()
    ticks = int(tail[19])
    if ticks < 1:
        raise StaticRegistryError("owned KFD process has invalid start ticks")
    return ticks


def runtime_maps_sampler(*, proc_root: Path = Path("/proc")) -> evidence.RuntimeMapsSampler:
    """Return the production callback used while rocprof's owned KFD PID lives.

    The invocation's context is constructed by the evidence producer, not the
    planner.  We rebuild the split closure from disk at the sampling instant and
    bind the exact KFD descendant/starttick to its maps image.
    """
    root = proc_root.resolve()
    def sample(invocation: evidence.CommandInvocation, launcher_pid: int,
               residency: evidence.GpuResidencySample) -> Mapping[str, Any]:
        context = invocation.runtime_maps_context
        if not isinstance(context, Mapping):
            raise StaticRegistryError("runtime maps callback lacks sealed invocation context")
        arm = context.get("arm")
        shared = context.get("shared_runtime")
        model = context.get("model")
        if (arm not in {"anchor", "candidate"} or not isinstance(shared, Mapping)
                or not isinstance(model, Mapping)):
            raise StaticRegistryError("runtime maps callback context is malformed")
        try:
            receipt = Path(str(shared["runtime_receipt"]["path"])).resolve(strict=True)
            runtime_body = json.loads(receipt.read_text(encoding="utf-8"))
            runtime_root = Path(str(runtime_body["split_runtime_manifest"]["root"])).resolve(strict=True)
            model_path = Path(str(model["path"])).resolve(strict=True)
            model_sha = str(context["model_sha256"])
            device_id = str(context["device_id"])
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise StaticRegistryError("runtime maps callback cannot load sealed context") from exc
        manifest = split_runtime_verifier.verify_split_runtime(runtime_root)
        identities = []
        for kfd_pid in sorted(set(residency.kfd_pids)):
            try:
                maps = (root / str(kfd_pid) / "maps").read_text(encoding="utf-8")
                identities.append(split_runtime_verifier.verify_runtime_maps(
                    manifest, arm=str(arm), maps_text=maps, model_path=model_path,
                    model_sha256=model_sha, device_id=device_id, kfd_pid=kfd_pid,
                    boot_id=_boot_id(root), process_start_ticks=_start_ticks(root, kfd_pid)))
            except (OSError, ValueError, split_runtime_verifier.SplitRuntimeError):
                # rocprof and helper wrappers can be KFD clients themselves;
                # only the uniquely proven full reward/model closure is valid.
                continue
        if len(identities) != 1:
            raise StaticRegistryError(
                "runtime maps must prove exactly one owned KFD process for the sealed arm")
        return identities[0].to_dict()
    return sample


def evidence_identity_files_for_build(
    build: controller.GpuSourceBuild, *, manifest: evidence.BoundInputFile,
    model: evidence.BoundInputFile, workload: evidence.BoundInputFile,
    runtime_config: evidence.BoundInputFile,
) -> evidence.EvidenceIdentityFiles:
    """Reconstruct every evidence carrier only from the sealed materialization.

    This intentionally refuses caller-supplied source paths.  Snapshot trees
    have already been torn down; their complete source TreeDigest receipts are
    the durable identity.
    """
    required = (build.materialization_receipt, build.materialization_sha256,
                build.anchor_source_tree_receipt, build.anchor_source_tree_sha256,
                build.candidate_source_tree_receipt, build.candidate_source_tree_sha256,
                build.measurement_binary, build.reward_runtime_sha256)
    if any(value is None for value in required):
        raise StaticRegistryError("source build lacks durable materialization/source/runtime receipts")
    assert build.materialization_receipt is not None and build.materialization_sha256 is not None
    if _digest(build.materialization_receipt) != build.materialization_sha256:
        raise StaticRegistryError("materialization carrier changed after build")
    try:
        body = json.loads(build.materialization_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticRegistryError("materialization receipt is not JSON") from exc
    if body.get("receipt_sha256") != schemas.content_hash(
            {key: value for key, value in body.items() if key != "receipt_sha256"}):
        raise StaticRegistryError("materialization receipt self-hash mismatch")
    if (body.get("schema") != "epyc.autokernel.gpu_source_materialization.v1"
            or body.get("candidate_identity") != vars(build.candidate_identity)
            or body.get("anchor_identity") != vars(build.anchor_identity)):
        raise StaticRegistryError("materialization does not bind returned build identities")
    refs = body.get("build_identity_files")
    if not isinstance(refs, Mapping) or set(refs) != {"anchor", "candidate"}:
        raise StaticRegistryError("materialization lacks sealed build identity carriers")
    def files(arm: str, identity: gpu_source_proofs.BuildIdentity) -> evidence.BuildIdentityFiles:
        row = refs[arm]
        if not isinstance(row, Mapping):
            raise StaticRegistryError("materialization build identity carrier is malformed")
        expected = ("source_identity", "binary", "hip_library", "config", "linkage")
        if set(row) != set(expected):
            raise StaticRegistryError("materialization carrier keys are incomplete")
        values = {key: evidence._bound_from_dict(row[key]) for key in expected}
        result = evidence.BuildIdentityFiles(**values)
        evidence._verify_build_files(result, identity, arm)
        return result
    shared_row = body.get("shared_runtime")
    if not isinstance(shared_row, Mapping):
        raise StaticRegistryError("materialization lacks shared reward runtime carriers")
    shared = evidence.SharedRewardRuntimeFiles(**{
        key: evidence._bound_from_dict(shared_row[key]) for key in (
            "measurement_binary", "runtime_receipt", "anchor_hip_library",
            "candidate_hip_library")})
    if shared.measurement_binary.path != build.measurement_binary:
        raise StaticRegistryError("materialization reward binary differs from returned build")
    return evidence.EvidenceIdentityFiles(
        candidate=files("candidate", build.candidate_identity),
        anchor=files("anchor", build.anchor_identity), manifest=manifest, model=model,
        workload=workload, runtime_config=runtime_config,
        materialization=evidence.BoundInputFile("materialization", build.materialization_receipt,
                                                 build.materialization_sha256),
        shared_runtime=shared)


def correctness_capability_files_for_build(
        build: controller.GpuSourceBuild, *, arm: str,
) -> tuple[evidence.BoundInputFile, evidence.BoundInputFile]:
    """Reopen one builder-terminal capability and its exact diagnostic binary."""
    if arm not in {"anchor", "candidate"}:
        raise StaticRegistryError("correctness capability arm is invalid")
    binary = getattr(build, f"{arm}_correctness_binary")
    binary_sha256 = getattr(build, f"{arm}_correctness_binary_sha256")
    receipt = getattr(build, f"{arm}_correctness_capability_receipt")
    receipt_sha256 = getattr(build, f"{arm}_correctness_capability_sha256")
    arm_build = build.anchor_build if arm == "anchor" else build.candidate_build
    if (not isinstance(binary, Path) or not isinstance(binary_sha256, str)
            or not isinstance(receipt, Path) or not isinstance(receipt_sha256, str)
            or binary != arm_build / "bin" / "test-backend-ops"):
        raise StaticRegistryError(
            f"source build lacks its {arm} correctness capability identity")
    _verify_correctness_capability_receipt(
        receipt=receipt, receipt_sha256=receipt_sha256,
        binary=binary, binary_sha256=binary_sha256,
        build=arm_build, arm=arm)
    return (evidence.BoundInputFile("executable", binary, binary_sha256),
            evidence.BoundInputFile("instrument_capability", receipt, receipt_sha256))


def verify_build_authority(build: controller.GpuSourceBuild, *, production_path: Path,
                           production_branch: str, production_commit: str,
                           instrument_path: Path, instrument_branch: str,
                           instrument_commit: str) -> None:
    """Revalidate both roots against a durable build receipt before evidence.

    Evidence may be produced long after snapshots were torn down (including an
    S2/cache reuse), so the authority must be re-established from the receipt
    and current Git refs rather than trusted from a prior builder invocation.
    """
    if build.materialization_receipt is None or build.materialization_sha256 is None:
        raise StaticRegistryError("build authority requires a materialization receipt")
    if _digest(build.materialization_receipt) != build.materialization_sha256:
        raise StaticRegistryError("materialization receipt carrier changed")
    try:
        materialization = json.loads(build.materialization_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticRegistryError("materialization authority receipt is not JSON") from exc
    if materialization.get("receipt_sha256") != schemas.content_hash(
            {key: value for key, value in materialization.items() if key != "receipt_sha256"}):
        raise StaticRegistryError("materialization authority receipt self-hash mismatch")
    expected_production = {"path": str(production_path.resolve(strict=True)),
                           "branch": production_branch, "commit": production_commit}
    if materialization.get("production_base_authority") != expected_production:
        raise StaticRegistryError("materialization production authority differs from deployment")
    expected_instrument = _instrument_authority(
        instrument_path=instrument_path, production_commit=production_commit,
        instrument_branch=instrument_branch, instrument_commit=instrument_commit)
    if materialization.get("instrument_authority") != expected_instrument:
        raise StaticRegistryError("materialization instrument authority differs from deployment")
    if (build.anchor_identity.source_commit != instrument_commit
            or materialization.get("anchor_commit") != instrument_commit):
        raise StaticRegistryError("build anchor does not originate from the sealed instrument")


def _source_tree_receipt(*, path: Path, root: Path, commit: str) -> tuple[Path, str]:
    """Persist a complete, self-hashed TreeDigest before teardown."""
    tree = integrity.hash_source_tree(root, exclude_dir_names=(".git",))
    body = {
        "schema": "epyc.autokernel.source_tree_identity.v1",
        "source_commit": commit,
        "root_provenance": str(root.resolve()),
        "exclusions": [".git"],
        "tree": tree.to_dict(),
    }
    body["receipt_sha256"] = schemas.content_hash(body)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    return path, _digest(path)


def _production_authority(*, production_path: Path, production_branch: str,
                          production_commit: str) -> dict[str, str]:
    root = production_path.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise StaticRegistryError("frozen production path is not a regular directory")
    def git(*argv: str) -> str:
        result = subprocess.run(("git", "-C", str(root), *argv), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=False)
        if result.returncode:
            raise StaticRegistryError(
                f"could not revalidate frozen production authority: {' '.join(argv)}")
        return result.stdout.strip()
    if (git("rev-parse", "HEAD") != production_commit
            or git("branch", "--show-current") != production_branch
            or git("status", "--porcelain", "--untracked-files=no")):
        raise StaticRegistryError("frozen production checkout changed from deployment authority")
    return {"path": str(root), "branch": production_branch, "commit": production_commit}


def _program_identity(name: str, *, path_value: str) -> dict[str, str] | None:
    resolved_name = shutil.which(name, path=path_value)
    if resolved_name is None:
        return None
    requested = Path(resolved_name)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise StaticRegistryError(f"build tool cannot be resolved: {name}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise StaticRegistryError(f"build tool is not a regular file: {name}")
    return {"requested": str(requested.absolute()), "resolved": str(resolved),
            "sha256": _digest(resolved)}


def _path_identity(path: Path) -> dict[str, str] | None:
    if not path.exists() and not path.is_symlink():
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StaticRegistryError(f"build tool cannot be resolved: {path}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise StaticRegistryError(f"build tool is not a regular file: {path}")
    return {"requested": str(path.absolute()), "resolved": str(resolved),
            "sha256": _digest(resolved)}


def _build_environment_and_toolchain() -> tuple[dict[str, str], dict[str, Any]]:
    """Freeze the small environment actually supplied to both build phases."""
    environment = {name: os.environ[name] for name in _BUILD_ENV_NAMES if name in os.environ}
    # Do not let a controller/container wrapper's ephemeral PATH become part of
    # the compiler selection or make otherwise identical S1/S2 keys diverge.
    environment["PATH"] = _SEALED_BUILD_PATH
    path_value = _SEALED_BUILD_PATH
    programs = {name: _program_identity(name, path_value=path_value)
                for name in ("cmake", "cc", "c++", "make", "ninja", "hipcc")}
    if programs["cmake"] is None or programs["cc"] is None or programs["c++"] is None:
        raise StaticRegistryError("sealed build toolchain requires cmake, cc, and c++")
    # Resolve the compilers rather than letting a later PATH edit choose them.
    environment["CC"] = str(programs["cc"]["resolved"])
    environment["CXX"] = str(programs["c++"]["resolved"])
    rocm_root = Path(environment.get("ROCM_PATH", environment.get("HIP_PATH", "/opt/rocm")))
    rocm_programs = {relative: _path_identity(rocm_root / relative) for relative in (
        "bin/hipcc", "llvm/bin/clang", "llvm/bin/clang++", "llvm/bin/ld.lld")}
    toolchain = {
        "schema": "epyc.autokernel.build_toolchain.v1",
        "programs": programs,
        "rocm_root": str(rocm_root.resolve(strict=False)),
        "rocm_programs": rocm_programs,
        "dynamic_environment": {
            "TMPDIR": "<arm-build-dir>/.autokernel-tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    toolchain["toolchain_sha256"] = schemas.content_hash(toolchain)
    return environment, toolchain


def _verify_source_identity_receipt(path: Path, *, expected_sha256: str,
                                    identity: gpu_source_proofs.BuildIdentity,
                                    cache_root: Path) -> None:
    path = _within(path, cache_root, "source identity receipt")
    if _digest(path) != expected_sha256:
        raise StaticRegistryError("source identity receipt carrier changed")
    body = _sealed_read(path, schema="epyc.autokernel.source_tree_identity.v1",
                        label="source identity receipt")
    tree = body.get("tree")
    if (body.get("source_commit") != identity.source_commit
            or not isinstance(tree, Mapping)
            or tree.get("sha256") != identity.source_sha256
            or tree.get("listing_is_complete") is not True):
        raise StaticRegistryError("source identity receipt differs from build identity")


def _copy_regular(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise StaticRegistryError(f"runtime object is not regular: {source}")
    shutil.copyfile(source, target)
    target.chmod(source.stat().st_mode & 0o777)


def _copy_topology(source_dir: Path, destination: Path, *, names: frozenset[str]) -> dict[str, str]:
    """Copy a complete local ELF closure while preserving only relative symlinks."""
    selected = sorted(path for path in source_dir.iterdir() if path.name in names)
    if not selected:
        raise StaticRegistryError(f"no selected runtime closure in {source_dir}")
    selected_names = {path.name for path in selected}
    manifest: dict[str, str] = {}
    for source in selected:
        target = destination / source.name
        if source.is_symlink():
            link = os.readlink(source)
            if (os.path.isabs(link) or Path(link).parts != (Path(link).name,)
                    or link not in selected_names):
                raise StaticRegistryError(f"runtime symlink {source} escapes its sealed closure")
            target.symlink_to(link)
            manifest[source.name] = f"symlink:{link}"
        else:
            _copy_regular(source, target)
            manifest[source.name] = f"file:{_digest(target)}"
    return manifest


def _local_closure(source_dir: Path, roots: tuple[str, ...]) -> frozenset[str]:
    """Return the complete one-directory SONAME topology for reviewed roots."""
    todo = list(roots)
    selected: set[str] = set()
    while todo:
        name = todo.pop()
        if name in selected:
            continue
        path = source_dir / name
        if not path.exists() and not path.is_symlink():
            raise StaticRegistryError(f"runtime closure lacks required artifact {path}")
        if path.is_symlink():
            link = os.readlink(path)
            # SONAME chains must remain local and direct: no escaped
            # ../../ build path can enter the reward closure.
            if os.path.isabs(link) or Path(link).parts != (Path(link).name,):
                raise StaticRegistryError(f"runtime SONAME link {path} is not local/direct")
            todo.append(link)
        elif not path.is_file():
            raise StaticRegistryError(f"runtime closure object is not a regular file {path}")
        selected.add(name)
    return frozenset(selected)


def _elf_dynamic(path: Path) -> dict[str, object] | None:
    """Read the dynamic contract of a real object; fixtures may be opaque bytes."""
    resolved = _resolved_regular(path)
    if resolved.read_bytes()[:4] != b"\x7fELF":
        return None
    completed = subprocess.run(("/usr/bin/readelf", "-d", str(resolved)),
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode:
        raise StaticRegistryError(f"readelf failed for sealed runtime object {resolved}")
    import re
    needed = tuple(re.findall(r"Shared library: \[(.+?)\]", completed.stdout))
    soname_rows = re.findall(r"Library soname: \[(.+?)\]", completed.stdout)
    runpaths = re.findall(r"(?:RUNPATH|RPATH).*?\[(.+?)\]", completed.stdout)
    if len(soname_rows) > 1:
        raise StaticRegistryError(f"runtime object has multiple SONAMEs: {resolved}")
    return {"soname": soname_rows[0] if soname_rows else None,
            "needed": needed, "runpath": tuple(runpaths)}


def _verify_elf_closure(common: Path, loaders: Mapping[str, Path],
                        invariant: Mapping[str, str]) -> dict[str, object]:
    """Verify common non-HIP closure and only-local HIP arm SONAMEs for real ELF."""
    common_rows = {name: _elf_dynamic(common / name) for name in invariant}
    real_rows = {name: row for name, row in common_rows.items() if row is not None}
    if not real_rows:
        return {"kind": "opaque_test_fixture"}
    common_names = set(invariant)
    for name, row in real_rows.items():
        assert isinstance(row, dict)
        for needed in row["needed"]:  # type: ignore[index]
            if needed == "libggml-hip.so.0":
                continue
            if needed.startswith("lib") and needed not in common_names:
                raise StaticRegistryError(f"common reward closure misses NEEDED {needed} from {name}")
    arms: dict[str, object] = {}
    for arm, directory in loaders.items():
        hip = _elf_dynamic(directory / "libggml-hip.so.0")
        if hip is None:
            raise StaticRegistryError("mixed opaque/ELF HIP closure is unsafe")
        if hip["soname"] != "libggml-hip.so.0":
            raise StaticRegistryError(f"{arm} HIP object has unexpected SONAME {hip['soname']!r}")
        # A HIP-only loader directory cannot smuggle another reward DSO.
        extras = sorted(path.name for path in directory.iterdir()
                        if path.name not in {"libggml-hip.so", "libggml-hip.so.0",
                                             (directory / "libggml-hip.so.0").resolve().name})
        if extras:
            raise StaticRegistryError(f"{arm} loader closure contains non-HIP extras: {extras}")
        arms[arm] = hip
    return {"kind": "elf_dynamic", "common": real_rows, "arms": arms}


@dataclass(frozen=True)
class SharedRewardRuntime:
    """Two complete loader directories with one byte-identical reward closure."""
    root: Path
    measurement_binary: Path
    common_loader_dir: Path
    anchor_loader_dir: Path
    candidate_loader_dir: Path
    receipt_path: Path

    @classmethod
    def materialize(cls, *, root: Path, anchor_build: Path, candidate_build: Path,
                    verifier: Any = split_runtime_verifier.verify_split_runtime) -> "SharedRewardRuntime":
        root.mkdir(parents=True, exist_ok=False)
        anchor_bin, candidate_bin = anchor_build / "bin", candidate_build / "bin"
        common = root / "common"; common.mkdir()
        loaders = {"anchor": root / "anchor-hip", "candidate": root / "candidate-hip"}
        for path in loaders.values(): path.mkdir()
        # Both arms execute one full byte-identical non-HIP closure.  Do not
        # put a whole build/bin on LD_LIBRARY_PATH: it would swap the reward
        # path together with the HIP DSO.
        common_names = _local_closure(anchor_bin, _REWARD_FILES)
        candidate_common_names = _local_closure(candidate_bin, _REWARD_FILES)
        if candidate_common_names != common_names:
            raise StaticRegistryError("candidate non-HIP reward SONAME topology differs from anchor")
        for name in common_names:
            anchor_item, candidate_item = anchor_bin / name, candidate_bin / name
            if anchor_item.is_symlink() != candidate_item.is_symlink():
                raise StaticRegistryError(f"candidate reward topology differs for {name}")
            if anchor_item.is_symlink():
                if os.readlink(anchor_item) != os.readlink(candidate_item):
                    raise StaticRegistryError(f"candidate reward link differs for {name}")
            # Candidate non-HIP objects can embed the candidate commit without
            # entering the shared reward path.  The split-runtime verifier
            # below validates ABI/topology/SONAME/NEEDED compatibility; only
            # the anchor copies are ever executable reward objects.
        invariant = _copy_topology(anchor_bin, common, names=common_names)
        if "llama-bench" not in invariant:
            raise StaticRegistryError("anchor reward closure lacks llama-bench")
        hip_names = {
            arm: _local_closure(source, ("libggml-hip.so", "libggml-hip.so.0"))
            for arm, source in (("anchor", anchor_bin), ("candidate", candidate_bin))}
        hip = {arm: _copy_topology(source, loaders[arm], names=hip_names[arm])
               for arm, source in (("anchor", anchor_bin), ("candidate", candidate_bin))}
        for arm in hip:
            if not {"libggml-hip.so", "libggml-hip.so.0"}.issubset(hip[arm]):
                raise StaticRegistryError(f"{arm} HIP closure lacks SONAME links")
            resolved = (loaders[arm] / "libggml-hip.so.0").resolve(strict=True)
            if resolved.parent != loaders[arm] or resolved.is_symlink():
                raise StaticRegistryError(f"{arm} HIP SONAME does not resolve to local regular DSO")
        if _digest((loaders["anchor"] / "libggml-hip.so.0").resolve()) == _digest((loaders["candidate"] / "libggml-hip.so.0").resolve()):
            raise StaticRegistryError("candidate HIP DSO is byte-identical to anchor")
        try:
            verified = verifier(root)
        except split_runtime_verifier.SplitRuntimeError as exc:
            raise StaticRegistryError(f"split reward runtime verifier refused closure: {exc}") from exc
        body = {"schema": "epyc.autokernel.shared_reward_runtime.v1",
                "authority": "nonpromotable_candidate_only_discovery",
                "measurement_binary_sha256": _digest(common / "llama-bench"),
                "invariant_files": invariant,
                "anchor_hip_topology": hip["anchor"], "candidate_hip_topology": hip["candidate"],
                "anchor_hip_sha256": _digest((loaders["anchor"] / "libggml-hip.so.0").resolve()),
                "candidate_hip_sha256": _digest((loaders["candidate"] / "libggml-hip.so.0").resolve()),
                "split_runtime_manifest": verified.to_dict(),
                "promotion_claim": False}
        body["receipt_sha256"] = schemas.content_hash(body)
        receipt = root / "reward-runtime.json"
        receipt.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
        return cls(root=root, measurement_binary=common / "llama-bench", common_loader_dir=common,
                   anchor_loader_dir=loaders["anchor"], candidate_loader_dir=loaders["candidate"],
                   receipt_path=receipt)


@dataclass(frozen=True)
class StaticGpuSourceBuilder:
    """Fresh production descendants, source mutation, clean snapshots, and builds."""
    production_path: Path
    production_branch: str
    instrument_path: Path
    operations_root: Path
    build_root: Path
    cmake_defines: tuple[tuple[str, str], ...]
    correctness_capability_runner: Any | None = None

    def _sealed_cmake_defines(self) -> tuple[tuple[str, str], ...]:
        required = {
            "CMAKE_BUILD_RPATH_USE_ORIGIN": "ON",
            "CMAKE_BUILD_RPATH": "$ORIGIN;/opt/rocm/lib",
            "CMAKE_INSTALL_RPATH": "$ORIGIN;/opt/rocm/lib",
        }
        supplied = dict(self.cmake_defines)
        if len(supplied) != len(self.cmake_defines):
            raise StaticRegistryError("duplicate CMake definitions are not a sealed build contract")
        for key, value in required.items():
            if key in supplied and supplied[key] != value:
                raise StaticRegistryError(f"source build may not override sealed {key}")
            supplied[key] = value
        return tuple(sorted(supplied.items()))

    def _contract(self, candidate: controller.PlannedCandidate,
                  permit: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        instrument_branch = permit.get("instrument_branch")
        deployment_sha = permit.get("deployment_config_sha256")
        if not isinstance(instrument_branch, str) or not instrument_branch:
            raise StaticRegistryError("static source builder requires a sealed instrument branch")
        if (not isinstance(deployment_sha, str) or len(deployment_sha) != 64
                or any(char not in "0123456789abcdef" for char in deployment_sha)):
            raise StaticRegistryError("static source builder requires sealed deployment authority")
        production_root = self.production_path.resolve(strict=True)
        instrument_root = self.instrument_path.resolve(strict=True)
        if production_root == instrument_root:
            raise StaticRegistryError("instrument repository must be separate from frozen production")
        mutable_roots = (self.operations_root.resolve(strict=False),
                         self.build_root.resolve(strict=False))
        if (mutable_roots[0] == mutable_roots[1]
                or mutable_roots[0].is_relative_to(mutable_roots[1])
                or mutable_roots[1].is_relative_to(mutable_roots[0])):
            raise StaticRegistryError("build root and cache root may not overlap")
        for mutable in mutable_roots:
            for protected in (production_root, instrument_root):
                if mutable == protected or mutable.is_relative_to(protected) or protected.is_relative_to(mutable):
                    raise StaticRegistryError("build/cache roots may not overlap protected source repositories")
        _regular_directory(self.operations_root, "operations root", create=True)
        _regular_directory(self.build_root, "build root", create=True)
        environment, toolchain = _build_environment_and_toolchain()
        production = _production_authority(
            production_path=self.production_path, production_branch=self.production_branch,
            production_commit=candidate.source_manifest.production_base_commit)
        instrument = _instrument_authority(
            instrument_path=self.instrument_path,
            production_commit=candidate.source_manifest.production_base_commit,
            instrument_branch=instrument_branch,
            instrument_commit=candidate.source_manifest.instrument_commit)
        selected = _verify_selected_gpu_blobs(
            production_path=self.production_path,
            production_commit=candidate.source_manifest.production_base_commit,
            instrument_path=self.instrument_path,
            instrument_commit=candidate.source_manifest.instrument_commit,
            paths=candidate.source_manifest.declared_files)
        effective_defines = dict(self._sealed_cmake_defines())
        effective_defines.setdefault("GGML_CCACHE", "OFF")
        try:
            proposal_sha256 = schemas.content_hash(dict(candidate.proposal))
        except (TypeError, ValueError) as exc:
            raise StaticRegistryError("candidate proposal is not canonical build authority") from exc
        contract: dict[str, Any] = {
            "schema": _BUILD_KEY_SCHEMA,
            "builder_schema": _BUILDER_SCHEMA,
            "deployment_config_sha256": deployment_sha,
            "production_base_authority": production,
            "instrument_authority": instrument,
            "patch_bundle_sha256": candidate.source_manifest.patch_bundle_sha256,
            "patch_sha256": candidate.source_manifest.patch_sha256,
            "proposal_sha256": proposal_sha256,
            "selected_gpu_base_blobs": selected,
            "cmake_defines": [list(item) for item in sorted(effective_defines.items())],
            "build_type": "Release",
            "parallelism": {"jobs": 1, "cpu_list": None, "load_average_cap": None},
            "required_targets": list(_REQUIRED_TARGETS),
            "build_environment": dict(sorted(environment.items())),
            "toolchain": toolchain,
            "operations_root": str(self.operations_root.resolve(strict=False)),
            "build_root": str(self.build_root.resolve(strict=False)),
        }
        contract["build_key"] = schemas.content_hash(contract)
        return contract, environment

    @staticmethod
    def _request_key(contract: Mapping[str, Any]) -> str:
        return schemas.content_hash({
            "schema": "epyc.autokernel.gpu_source_build_request.v1",
            "deployment_config_sha256": contract["deployment_config_sha256"],
            "production_base_authority": contract["production_base_authority"],
            "instrument_authority": contract["instrument_authority"],
            "patch_bundle_sha256": contract["patch_bundle_sha256"],
            "proposal_sha256": contract["proposal_sha256"],
            "builder_schema": contract["builder_schema"],
        })

    @staticmethod
    def _build_projection(build: controller.GpuSourceBuild) -> dict[str, Any]:
        fields = {
            "anchor_build": str(build.anchor_build),
            "candidate_build": str(build.candidate_build),
            "candidate_identity": vars(build.candidate_identity),
            "anchor_identity": vars(build.anchor_identity),
            "measurement_binary": str(build.measurement_binary),
            "common_loader_dir": str(build.common_loader_dir),
            "anchor_loader_dir": str(build.anchor_loader_dir),
            "candidate_loader_dir": str(build.candidate_loader_dir),
            "reward_runtime_sha256": build.reward_runtime_sha256,
            "build_key": build.build_key,
            "materialization_receipt": str(build.materialization_receipt),
            "materialization_sha256": build.materialization_sha256,
            "anchor_source_tree_receipt": str(build.anchor_source_tree_receipt),
            "anchor_source_tree_sha256": build.anchor_source_tree_sha256,
            "candidate_source_tree_receipt": str(build.candidate_source_tree_receipt),
            "candidate_source_tree_sha256": build.candidate_source_tree_sha256,
            "anchor_correctness_binary": str(build.anchor_correctness_binary),
            "anchor_correctness_binary_sha256": build.anchor_correctness_binary_sha256,
            "candidate_correctness_binary": str(build.candidate_correctness_binary),
            "candidate_correctness_binary_sha256": build.candidate_correctness_binary_sha256,
            "anchor_correctness_capability_receipt": str(
                build.anchor_correctness_capability_receipt),
            "anchor_correctness_capability_sha256":
                build.anchor_correctness_capability_sha256,
            "candidate_correctness_capability_receipt": str(
                build.candidate_correctness_capability_receipt),
            "candidate_correctness_capability_sha256":
                build.candidate_correctness_capability_sha256,
            "teardown_receipt": str(build.teardown_receipt),
            "teardown_sha256": build.teardown_sha256,
        }
        if any(value in (None, "None") for value in fields.values()):
            raise StaticRegistryError("completed build projection is incomplete")
        return fields

    def _validate_ref(self, path: Path, *, request_key: str, build_key: str) -> None:
        row = _sealed_read(path, schema=_BUILD_REF_SCHEMA, label="build cache ref")
        if (row.get("request_key") != request_key or row.get("build_key") != build_key
                or row.get("builder_schema") != _BUILDER_SCHEMA):
            raise StaticRegistryError("build cache ref differs from the sealed request")

    def _reopen(self, *, cache_root: Path, build_root: Path, operation_key: str,
                expected_intent: Mapping[str, Any], contract: Mapping[str, Any]) -> controller.GpuSourceBuild:
        intent_path = cache_root / "intent.json"
        intent = _sealed_read(intent_path, schema=_BUILD_INTENT_SCHEMA,
                              label="build cache intent")
        if any(intent.get(key) != value for key, value in expected_intent.items()):
            raise StaticRegistryError("build cache intent differs from the canonical build contract")
        terminal_path = cache_root / "terminal.json"
        if not terminal_path.exists() and not terminal_path.is_symlink():
            raise StaticRegistryError("build cache is incomplete after a prior intent; refusing rebuild")
        terminal = _sealed_read(terminal_path, schema=_BUILD_TERMINAL_SCHEMA,
                                label="build cache terminal")
        if (terminal.get("build_key") != contract["build_key"]
                or terminal.get("intent_file_sha256") != _digest(intent_path)):
            raise StaticRegistryError("build cache terminal does not close its intent")
        if terminal.get("state") != "complete":
            if (terminal.get("state") == "failed"
                    and terminal.get("failure_type") ==
                    source_candidate.SourceCandidateError.__name__):
                # The ref, canonical intent, terminal self-hash, build key,
                # and exact intent-file digest have all been revalidated above.
                # Only that complete identity chain may recover the original
                # typed authoring refusal; every other failed/tampered cache
                # remains a non-reusable registry error.
                failure_message = terminal.get("failure_message")
                if (failure_message is not None
                        and _source_failure_message(failure_message) is None):
                    raise StaticRegistryError(
                        "source candidate failed terminal has an unsafe diagnostic")
                detail = (f": {failure_message}"
                          if isinstance(failure_message, str) else "")
                raise source_candidate.SourceCandidateError(
                    "sealed prior build transaction rejected source candidate "
                    f"authoring for build {contract['build_key']}{detail}")
            raise StaticRegistryError("prior build transaction is terminal but not reusable")
        raw = terminal.get("build")
        if not isinstance(raw, Mapping):
            raise StaticRegistryError("completed build transaction has no typed projection")
        try:
            build = controller.GpuSourceBuild(
                anchor_build=Path(str(raw["anchor_build"])),
                candidate_build=Path(str(raw["candidate_build"])),
                candidate_identity=gpu_source_proofs.BuildIdentity(**dict(raw["candidate_identity"])),
                anchor_identity=gpu_source_proofs.BuildIdentity(**dict(raw["anchor_identity"])),
                measurement_binary=Path(str(raw["measurement_binary"])),
                common_loader_dir=Path(str(raw["common_loader_dir"])),
                anchor_loader_dir=Path(str(raw["anchor_loader_dir"])),
                candidate_loader_dir=Path(str(raw["candidate_loader_dir"])),
                reward_runtime_sha256=str(raw["reward_runtime_sha256"]),
                operation_key=operation_key,
                build_key=str(raw["build_key"]),
                materialization_receipt=Path(str(raw["materialization_receipt"])),
                materialization_sha256=str(raw["materialization_sha256"]),
                anchor_source_tree_receipt=Path(str(raw["anchor_source_tree_receipt"])),
                anchor_source_tree_sha256=str(raw["anchor_source_tree_sha256"]),
                candidate_source_tree_receipt=Path(str(raw["candidate_source_tree_receipt"])),
                candidate_source_tree_sha256=str(raw["candidate_source_tree_sha256"]),
                anchor_correctness_binary=Path(str(raw["anchor_correctness_binary"])),
                anchor_correctness_binary_sha256=str(
                    raw["anchor_correctness_binary_sha256"]),
                candidate_correctness_binary=Path(str(raw["candidate_correctness_binary"])),
                candidate_correctness_binary_sha256=str(
                    raw["candidate_correctness_binary_sha256"]),
                anchor_correctness_capability_receipt=Path(str(
                    raw["anchor_correctness_capability_receipt"])),
                anchor_correctness_capability_sha256=str(
                    raw["anchor_correctness_capability_sha256"]),
                candidate_correctness_capability_receipt=Path(str(
                    raw["candidate_correctness_capability_receipt"])),
                candidate_correctness_capability_sha256=str(
                    raw["candidate_correctness_capability_sha256"]),
                teardown_receipt=Path(str(raw["teardown_receipt"])),
                teardown_sha256=str(raw["teardown_sha256"]),
            )
        except (KeyError, TypeError, ValueError, controller.DiscoveryControllerError) as exc:
            raise StaticRegistryError("completed build projection is malformed") from exc
        if build.build_key != contract["build_key"]:
            raise StaticRegistryError("cached build identity is not keyed by its build contract")
        _within(build.anchor_build, build_root, "anchor build", directory=True)
        _within(build.candidate_build, build_root, "candidate build", directory=True)
        materialization_path = _within(
            build.materialization_receipt, cache_root, "materialization receipt")
        materialization = _sealed_read(
            materialization_path, schema="epyc.autokernel.gpu_source_materialization.v1",
            label="materialization receipt")
        if (materialization.get("build_key") != contract["build_key"]
                or materialization.get("build_contract") != contract
                or materialization.get("operation_key") != contract["build_key"]
                or materialization.get("manifest_sha256") != contract["patch_bundle_sha256"]
                or materialization.get("production_base_authority") != contract["production_base_authority"]
                or materialization.get("instrument_authority") != contract["instrument_authority"]):
            raise StaticRegistryError("materialization receipt differs from sealed build contract")
        if _digest(materialization_path) != build.materialization_sha256:
            raise StaticRegistryError("materialization receipt bytes changed")
        if (_production_authority(
                production_path=self.production_path, production_branch=self.production_branch,
                production_commit=str(contract["production_base_authority"]["commit"]))
                != contract["production_base_authority"]):
            raise StaticRegistryError("production authority changed after cached build")
        current_instrument = _instrument_authority(
            instrument_path=self.instrument_path,
            production_commit=str(contract["production_base_authority"]["commit"]),
            instrument_branch=str(contract["instrument_authority"]["instrument_branch"]),
            instrument_commit=str(contract["instrument_authority"]["instrument_commit"]))
        if current_instrument != contract["instrument_authority"]:
            raise StaticRegistryError("instrument authority changed after cached build")
        selected = _verify_selected_gpu_blobs(
            production_path=self.production_path,
            production_commit=str(contract["production_base_authority"]["commit"]),
            instrument_path=self.instrument_path,
            instrument_commit=str(contract["instrument_authority"]["instrument_commit"]),
            paths=tuple(sorted(contract["selected_gpu_base_blobs"])))
        if selected != contract["selected_gpu_base_blobs"]:
            raise StaticRegistryError("selected source blobs changed after cached build")
        _verify_source_identity_receipt(
            build.anchor_source_tree_receipt,
            expected_sha256=str(build.anchor_source_tree_sha256),
            identity=build.anchor_identity, cache_root=cache_root)
        _verify_source_identity_receipt(
            build.candidate_source_tree_receipt,
            expected_sha256=str(build.candidate_source_tree_sha256),
            identity=build.candidate_identity, cache_root=cache_root)
        for arm in ("anchor", "candidate"):
            binary = getattr(build, f"{arm}_correctness_binary")
            binary_sha256 = getattr(build, f"{arm}_correctness_binary_sha256")
            receipt = getattr(build, f"{arm}_correctness_capability_receipt")
            receipt_sha256 = getattr(build, f"{arm}_correctness_capability_sha256")
            assert isinstance(binary, Path) and isinstance(binary_sha256, str)
            assert isinstance(receipt, Path) and isinstance(receipt_sha256, str)
            arm_build = build.anchor_build if arm == "anchor" else build.candidate_build
            expected_binary = arm_build / "bin" / "test-backend-ops"
            if binary != expected_binary or _digest(binary) != binary_sha256:
                raise StaticRegistryError(
                    f"cached {arm} correctness binary identity changed")
            _within(binary, build_root, f"{arm} correctness binary", single_link=True)
            _within(receipt, cache_root, f"{arm} correctness capability", single_link=True)
            _verify_correctness_capability_receipt(
                receipt=receipt, receipt_sha256=receipt_sha256,
                binary=binary, binary_sha256=binary_sha256,
                build=arm_build, arm=arm)
        capability_refs = materialization.get("correctness_capabilities")
        if not isinstance(capability_refs, Mapping) or set(capability_refs) != {
                "anchor", "candidate"}:
            raise StaticRegistryError(
                "materialization correctness capability map is malformed")
        for arm in ("anchor", "candidate"):
            receipt = getattr(build, f"{arm}_correctness_capability_receipt")
            receipt_sha256 = getattr(build, f"{arm}_correctness_capability_sha256")
            if capability_refs[arm] != {
                    "role": "correctness_capability", "path": str(receipt.resolve()),
                    "sha256": receipt_sha256}:
                raise StaticRegistryError(
                    f"materialization {arm} correctness capability binding changed")
        refs = materialization.get("build_identity_files")
        if not isinstance(refs, Mapping) or set(refs) != {"anchor", "candidate"}:
            raise StaticRegistryError("materialization build carrier map is malformed")
        for arm, identity in (("anchor", build.anchor_identity),
                              ("candidate", build.candidate_identity)):
            row = refs[arm]
            if not isinstance(row, Mapping):
                raise StaticRegistryError("materialization build carrier row is malformed")
            values = {key: evidence._bound_from_dict(row[key]) for key in (
                "source_identity", "binary", "hip_library", "config", "linkage")}
            for value in values.values():
                allowed_root = cache_root if value.role in {"source_identity", "linkage"} else build_root
                _within(value.path, allowed_root, f"{arm} {value.role}",
                        single_link=value.role in {"source_identity", "linkage"})
            evidence._verify_build_files(evidence.BuildIdentityFiles(**values), identity, arm)
        shared_row = materialization.get("shared_runtime")
        if not isinstance(shared_row, Mapping):
            raise StaticRegistryError("materialization shared runtime is malformed")
        shared = evidence.SharedRewardRuntimeFiles(**{
            key: evidence._bound_from_dict(shared_row[key]) for key in (
                "measurement_binary", "runtime_receipt", "anchor_hip_library",
                "candidate_hip_library")})
        for item in (shared.measurement_binary, shared.runtime_receipt,
                     shared.anchor_hip_library, shared.candidate_hip_library):
            _within(item.path, cache_root, f"shared runtime {item.role}")
            evidence._verify_bound(item)
        runtime_body = _sealed_read(shared.runtime_receipt.path,
                                    schema="epyc.autokernel.shared_reward_runtime.v1",
                                    label="shared reward runtime receipt")
        try:
            runtime_root = Path(str(runtime_body["split_runtime_manifest"]["root"]))
            verified_runtime = split_runtime_verifier.verify_split_runtime(runtime_root)
        except (KeyError, TypeError, OSError, split_runtime_verifier.SplitRuntimeError) as exc:
            raise StaticRegistryError("cached split reward runtime cannot be revalidated") from exc
        if runtime_body["split_runtime_manifest"] != verified_runtime.to_dict():
            raise StaticRegistryError("cached split reward runtime identity changed")
        expected_runtime_paths = {
            "root": verified_runtime.root.resolve(strict=True),
            "measurement": verified_runtime.reward_binary.resolve(strict=True),
            "common": verified_runtime.common_dir.resolve(strict=True),
            "anchor": verified_runtime.anchor_hip_dir.resolve(strict=True),
            "candidate": verified_runtime.candidate_hip_dir.resolve(strict=True),
        }
        for label, path in (("common", build.common_loader_dir),
                            ("anchor", build.anchor_loader_dir),
                            ("candidate", build.candidate_loader_dir)):
            _within(path, verified_runtime.root, f"cached {label} loader directory",
                    directory=True)
        actual_runtime_paths = {
            "root": runtime_root,
            "measurement": build.measurement_binary,
            "common": build.common_loader_dir,
            "anchor": build.anchor_loader_dir,
            "candidate": build.candidate_loader_dir,
        }
        if actual_runtime_paths != expected_runtime_paths:
            raise StaticRegistryError("cached runner loader paths differ from verified split runtime")
        if (build.measurement_binary != shared.measurement_binary.path
                or build.reward_runtime_sha256 != shared.runtime_receipt.sha256
                or build.anchor_identity.hip_library_sha256 != shared.anchor_hip_library.sha256
                or build.candidate_identity.hip_library_sha256 != shared.candidate_hip_library.sha256):
            raise StaticRegistryError("cached reward runtime differs from build identities")
        teardown_path = _within(build.teardown_receipt, cache_root, "teardown receipt")
        teardown = _sealed_read(teardown_path, schema=_TEARDOWN_SCHEMA,
                                label="teardown receipt")
        if (_digest(teardown_path) != build.teardown_sha256
                or teardown.get("build_key") != contract["build_key"]
                or teardown.get("errors") != []):
            raise StaticRegistryError("cached build teardown is incomplete")
        receipts = teardown.get("receipts")
        if not isinstance(receipts, list) or len(receipts) != 3:
            raise StaticRegistryError("cached build teardown does not cover all worktrees")
        actor = materialization.get("actor_worktree")
        builds = materialization.get("builds")
        if not isinstance(actor, Mapping) or not isinstance(builds, Mapping):
            raise StaticRegistryError("materialization lacks worktree provenance")
        expected_worktree_paths = {str(Path(str(actor.get("path", ""))).resolve())}
        for build_receipt in builds.values():
            plan = build_receipt.get("plan") if isinstance(build_receipt, Mapping) else None
            if not isinstance(plan, Mapping):
                raise StaticRegistryError("cached build receipt lacks a source snapshot")
            expected_worktree_paths.add(str(Path(str(plan.get("source_root", ""))).resolve()))
        if len(expected_worktree_paths) != 3:
            raise StaticRegistryError("materialization worktree provenance is not one actor plus two snapshots")
        actual_worktree_paths = {
            str(Path(str(receipt.get("worktree_path", ""))).resolve())
            for receipt in receipts if isinstance(receipt, Mapping)}
        if actual_worktree_paths != expected_worktree_paths:
            raise StaticRegistryError("teardown receipts do not bind the materialized worktrees")
        for receipt in receipts:
            if (not isinstance(receipt, Mapping)
                    or receipt.get("worktree_removed") is not True
                    or receipt.get("branch_exists_after") is not False
                    or receipt.get("all_production_trees_unchanged") is not True):
                raise StaticRegistryError("cached worktree teardown proof is not clean")
            worktree_path = Path(str(receipt.get("worktree_path", "")))
            if worktree_path.exists() or worktree_path.is_symlink():
                raise StaticRegistryError("a supposedly torn-down build worktree still exists")
        if not isinstance(builds, Mapping) or len(builds) != 2:
            raise StaticRegistryError("cached materialization lacks both build receipts")
        for build_receipt in builds.values():
            if (not isinstance(build_receipt, Mapping)
                    or build_receipt.get("succeeded") is not True
                    or build_receipt.get("log_disagrees_with_exit_code") is not False):
                raise StaticRegistryError("cached build receipt is not successful")
            log = _within(Path(str(build_receipt.get("log_path", ""))), cache_root,
                          "cached build log", single_link=True)
            if _digest(log) != build_receipt.get("log_sha256"):
                raise StaticRegistryError("cached build log changed")
            facts = build_receipt.get("facts")
            if (not isinstance(facts, Mapping)
                    or not set(_REQUIRED_TARGETS).issubset(set(facts.get("built_targets", [])))):
                raise StaticRegistryError("cached build receipt lacks required targets")
            plan = build_receipt.get("plan")
            expected_cmake = str(contract["toolchain"]["programs"]["cmake"]["resolved"])
            expected_parallelism = contract["parallelism"]
            if (not isinstance(plan, Mapping)
                    or plan.get("targets") != contract["required_targets"]
                    or plan.get("build_type") != contract["build_type"]
                    or plan.get("cmake_defines") != contract["cmake_defines"]
                    or plan.get("parallelism") != expected_parallelism
                    or not isinstance(plan.get("configure_command"), list)
                    or not isinstance(plan.get("build_command"), list)
                    or expected_cmake not in plan["configure_command"]
                    or expected_cmake not in plan["build_command"]):
                raise StaticRegistryError("cached build plan differs from sealed build contract")
        return build

    def build(self, candidate: controller.PlannedCandidate, _authorization: Any,
              _permit: Mapping[str, Any]) -> controller.GpuSourceBuild:
        operation_key = _permit.get("operation_key")
        if (not isinstance(operation_key, str) or len(operation_key) != 64
                or any(char not in "0123456789abcdef" for char in operation_key)):
            raise StaticRegistryError("static builder requires the sealed controller operation key")
        contract, environment = self._contract(candidate, _permit)
        build_key = str(contract["build_key"])
        request_key = self._request_key(contract)
        cache_base = self.operations_root / "build-cache"
        _regular_directory(cache_base, "build cache", create=True)
        refs_root = _regular_directory(cache_base / "refs", "build cache refs", create=True)
        entries_root = _regular_directory(cache_base / "entries", "build cache entries", create=True)
        locks_root = _regular_directory(cache_base / "locks", "build cache locks", create=True)
        cache_root = entries_root / build_key
        keyed_build_root = self.build_root / build_key
        expected_intent = {
            "schema": _BUILD_INTENT_SCHEMA,
            "authority": "nonpromotable_candidate_only_discovery",
            "promotion_claim": False,
            "request_key": request_key,
            "build_key": build_key,
            "build_contract": contract,
        }
        with _exclusive_lock(locks_root / f"request-{request_key}.lock"):
            with _exclusive_lock(locks_root / f"build-{build_key}.lock"):
                ref_path = refs_root / f"{request_key}.json"
                created_ref = False
                cache_present = cache_root.exists() or cache_root.is_symlink()
                if ref_path.exists() or ref_path.is_symlink():
                    self._validate_ref(ref_path, request_key=request_key, build_key=build_key)
                else:
                    if cache_present:
                        raise StaticRegistryError(
                            "build cache entry exists without its sealed ref; refusing reuse")
                    _sealed_write(ref_path, {
                        "schema": _BUILD_REF_SCHEMA, "builder_schema": _BUILDER_SCHEMA,
                        "request_key": request_key, "build_key": build_key,
                        "promotion_claim": False})
                    created_ref = True
                if cache_present:
                    _regular_directory(cache_root, "build cache entry")
                    return self._reopen(cache_root=cache_root,
                                        build_root=keyed_build_root,
                                        operation_key=operation_key,
                                        expected_intent=expected_intent, contract=contract)
                if not created_ref:
                    raise StaticRegistryError(
                        "build ref exists without its cache transaction; refusing rebuild")
                cache_root.mkdir()
                _intent_path, intent_file_sha = _sealed_write(
                    cache_root / "intent.json", expected_intent)
                try:
                    build = self._build_uncached(
                        candidate, _authorization,
                        {**dict(_permit), "operation_key": build_key,
                         "build_contract": contract, "build_environment": environment,
                         "cache_root": str(cache_root)})
                except Exception as exc:
                    failure = {
                        "schema": _BUILD_TERMINAL_SCHEMA, "build_key": build_key,
                        "intent_file_sha256": intent_file_sha, "state": "failed",
                        "failure_type": type(exc).__name__, "promotion_claim": False}
                    if isinstance(exc, source_candidate.SourceCandidateError):
                        message = _source_failure_message(str(exc))
                        if message is not None:
                            failure["failure_message"] = message
                    _sealed_write(cache_root / "terminal.json", failure)
                    raise
                _sealed_write(cache_root / "terminal.json", {
                    "schema": _BUILD_TERMINAL_SCHEMA, "build_key": build_key,
                    "intent_file_sha256": intent_file_sha, "state": "complete",
                    "build": self._build_projection(build), "promotion_claim": False})
                return self._reopen(cache_root=cache_root,
                                    build_root=keyed_build_root,
                                    operation_key=operation_key,
                                    expected_intent=expected_intent, contract=contract)

    def _build_uncached(self, candidate: controller.PlannedCandidate, _authorization: Any,
                        _permit: Mapping[str, Any]) -> controller.GpuSourceBuild:
        operation_key = _permit.get("operation_key")
        if not isinstance(operation_key, str) or len(operation_key) != 64:
            raise StaticRegistryError("uncached builder requires its canonical build key")
        contract = _permit.get("build_contract")
        environment = _permit.get("build_environment")
        cache_root_raw = _permit.get("cache_root")
        if (not isinstance(contract, Mapping) or contract.get("build_key") != operation_key
                or not isinstance(environment, Mapping) or not isinstance(cache_root_raw, str)):
            raise StaticRegistryError("uncached builder lacks its sealed build transaction")
        cache_root = Path(cache_root_raw).resolve(strict=True)
        # Production remains verification-only.  Every mutable worktree is
        # created from the independently sealed experimental instrument repo.
        if self.instrument_path.resolve(strict=True) == self.production_path.resolve(strict=True):
            raise StaticRegistryError("instrument repository must be separate from frozen production")
        instrument_branch = _permit.get("instrument_branch")
        if not isinstance(instrument_branch, str) or not instrument_branch:
            raise StaticRegistryError("static source builder requires a sealed instrument branch")
        anchor = worktree.resolve_anchor(self.instrument_path, instrument_branch,
                                         expected_commit=candidate.source_manifest.instrument_commit)
        instrument_authority = _instrument_authority(
            instrument_path=self.instrument_path,
            production_commit=candidate.source_manifest.production_base_commit,
            instrument_branch=instrument_branch,
            instrument_commit=candidate.source_manifest.instrument_commit)
        selected_gpu_blobs = _verify_selected_gpu_blobs(
            production_path=self.production_path,
            production_commit=candidate.source_manifest.production_base_commit,
            instrument_path=self.instrument_path,
            instrument_commit=candidate.source_manifest.instrument_commit,
            paths=candidate.source_manifest.declared_files)
        campaign_root = cache_root / "worktrees"
        build_root = self.build_root / operation_key
        if campaign_root.exists() or campaign_root.is_symlink():
            raise StaticRegistryError("fresh build transaction already has a worktree root")
        if build_root.exists() or build_root.is_symlink():
            raise StaticRegistryError("fresh build transaction already has a keyed build root")
        campaign_root.mkdir(); build_root.mkdir()
        actor: worktree.Worktree | None = None
        actor_proof: Any | None = None
        snapshots: list[worktree.Worktree] = []
        operation_dir = cache_root / "receipts"
        operation_dir.mkdir(parents=True, exist_ok=True)
        completed: dict[str, Any] | None = None
        try:
            actor, actor_proof = worktree.create_campaign_worktree(
                anchor, candidate.source_manifest.campaign_id, root=campaign_root)
            applied = source_candidate.apply_source_candidate(candidate.source_manifest,
                                                               proposal=candidate.proposal, actor=actor)
            anchor_snapshot, _ = worktree.create_snapshot_worktree(
                self.instrument_path, anchor.commit,
                worktree.snapshot_worktree_path(candidate.source_manifest.campaign_id,
                                                "akc-anchor", root=campaign_root))
            snapshots.append(anchor_snapshot)
            candidate_snapshot, _ = worktree.create_snapshot_worktree(
                self.instrument_path, applied.candidate_commit,
                worktree.snapshot_worktree_path(candidate.source_manifest.campaign_id,
                                                candidate.source_manifest.candidate_id, root=campaign_root))
            snapshots.append(candidate_snapshot)
            parallel = worktree.BuildParallelism(jobs=1)
            plans = []
            for ident, snapshot in (("akc-anchor", anchor_snapshot),
                                    (candidate.source_manifest.candidate_id, candidate_snapshot)):
                build_dir = worktree.default_build_dir(candidate.source_manifest.campaign_id, ident,
                                                       root=build_root)
                plans.append((ident, snapshot, build_dir, worktree.BuildPlan(
                    source_root=snapshot.path, build_dir=build_dir, actor_worktree=actor.path,
                    parallelism=parallel, targets=_REQUIRED_TARGETS,
                    cmake_defines=self._sealed_cmake_defines(),
                    cmake=str(contract["toolchain"]["programs"]["cmake"]["resolved"]))))
            results = []
            for ident, snapshot, build_dir, plan in plans:
                log = cache_root / "logs" / f"{ident}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                result = worktree.run_build(plan, log_path=log,
                                            env={str(key): str(value)
                                                 for key, value in environment.items()})
                if not result.succeeded or result.log_disagrees_with_exit_code:
                    raise StaticRegistryError(f"clean build did not succeed for {ident}")
                if not set(_REQUIRED_TARGETS).issubset(set(result.facts.built_targets)):
                    raise StaticRegistryError(f"build did not prove required targets for {ident}")
                results.append((ident, snapshot, build_dir, result))
            by_id = {ident: (snapshot, build_dir, result) for ident, snapshot, build_dir, result in results}
            arm_builds = {
                "anchor": Path(by_id["akc-anchor"][1].path),
                "candidate": Path(
                    by_id[candidate.source_manifest.candidate_id][1].path),
            }
            correctness_capabilities: dict[str, tuple[Path, str, Path, str]] = {}
            for arm, arm_build in arm_builds.items():
                capability = _attest_correctness_capability(
                    arm_build, arm=arm, runner=self.correctness_capability_runner)
                capability_path, capability_sha = _sealed_write(
                    operation_dir / f"{arm}-correctness-capability.json", capability)
                correctness_binary = arm_build / "bin" / "test-backend-ops"
                correctness_capabilities[arm] = (
                    correctness_binary, _digest(correctness_binary),
                    capability_path, capability_sha)
            anchor_root = Path(anchor_snapshot.path.path); candidate_root = Path(candidate_snapshot.path.path)
            anchor_identity = _identity(root=anchor_root, commit=anchor.commit,
                                        build=Path(by_id["akc-anchor"][1].path))
            candidate_identity = _identity(root=candidate_root, commit=applied.candidate_commit,
                                           build=Path(by_id[candidate.source_manifest.candidate_id][1].path))
            anchor_source_receipt, anchor_source_sha = _source_tree_receipt(
                path=operation_dir / "anchor-source-tree.json", root=anchor_root,
                commit=anchor.commit)
            candidate_source_receipt, candidate_source_sha = _source_tree_receipt(
                path=operation_dir / "candidate-source-tree.json", root=candidate_root,
                commit=applied.candidate_commit)
            if (anchor_source_sha == anchor_identity.source_sha256
                    or candidate_source_sha == candidate_identity.source_sha256):
                raise StaticRegistryError("source tree carrier must not be mistaken for tree digest")
            anchor_linkage, anchor_linkage_sha = _write_linkage_carrier(
                operation_dir / "anchor-linkage.json", Path(by_id["akc-anchor"][1].path))
            candidate_linkage, candidate_linkage_sha = _write_linkage_carrier(
                operation_dir / "candidate-linkage.json",
                Path(by_id[candidate.source_manifest.candidate_id][1].path))
            if (anchor_linkage_sha != anchor_identity.linkage_sha256
                    or candidate_linkage_sha != candidate_identity.linkage_sha256):
                raise StaticRegistryError("linkage carrier does not reproduce build identity")
            def identity_files(*, arm: str, build_dir: Path, source_receipt: Path) -> dict[str, dict[str, str]]:
                return {
                    "source_identity": _bound(source_receipt, "source_identity"),
                    "binary": _bound(build_dir / "bin" / "llama-bench", "binary"),
                    "hip_library": _bound(_resolved_regular(build_dir / "bin" / _HIP), "hip_library"),
                    "config": _bound(build_dir / "CMakeCache.txt", "config"),
                    "linkage": _bound(anchor_linkage if arm == "anchor" else candidate_linkage,
                                      "linkage"),
                }
            anchor_files = identity_files(
                arm="anchor", build_dir=Path(by_id["akc-anchor"][1].path),
                source_receipt=anchor_source_receipt)
            candidate_files = identity_files(
                arm="candidate", build_dir=Path(by_id[candidate.source_manifest.candidate_id][1].path),
                source_receipt=candidate_source_receipt)
            runtime = SharedRewardRuntime.materialize(
                root=cache_root / "runtime",
                anchor_build=Path(by_id["akc-anchor"][1].path),
                candidate_build=Path(by_id[candidate.source_manifest.candidate_id][1].path),
                verifier=split_runtime_verifier.verify_split_runtime)
            # `create_*worktree` independently proves the experimental checkout
            # was unchanged around each operation.  Re-read the sealed branch
            # object as well, so a ref move cannot be hidden between actor
            # creation and durable evidence materialization.
            if _instrument_authority(
                    instrument_path=self.instrument_path,
                    production_commit=candidate.source_manifest.production_base_commit,
                    instrument_branch=instrument_branch,
                    instrument_commit=candidate.source_manifest.instrument_commit) != instrument_authority:
                raise StaticRegistryError("instrument authority changed during source materialization")
            materialization = {
                "schema": "epyc.autokernel.gpu_source_materialization.v1",
                "authority": "nonpromotable_candidate_only_discovery",
                "operation_key": operation_key,
                "build_key": operation_key,
                "build_contract": dict(contract),
                "actor_worktree": actor.to_record(),
                "actor_proof": actor_proof.to_dict(),
                "manifest_sha256": candidate.source_manifest.patch_bundle_sha256,
                "production_base_authority": {
                    **dict(contract["production_base_authority"]),
                },
                "instrument_authority": instrument_authority,
                "selected_gpu_base_blobs": selected_gpu_blobs,
                "applied": {
                    "candidate_commit": applied.candidate_commit,
                    "actual_files": list(applied.actual_files),
                    "actual_hunk_ids": list(applied.actual_hunk_ids),
                    "actual_symbols": list(applied.actual_symbols),
                    "commit_argv": list(applied.commit_argv),
                    "mutation_receipt": dict(applied.mutation_receipt),
                    "diff_sha256": hashlib.sha256(applied.diff_text.encode()).hexdigest(),
                },
                "anchor_commit": anchor.commit,
                "candidate_source_commit": applied.candidate_commit,
                "candidate_source_sha256": candidate_identity.source_sha256,
                "patch_applied": True,
                "production_tree": False,
                "builds": {ident: result.to_dict() for ident, _snapshot, _build, result in results},
                "anchor_identity": vars(anchor_identity),
                "candidate_identity": vars(candidate_identity),
                "anchor_source_tree_receipt": str(anchor_source_receipt),
                "anchor_source_tree_receipt_sha256": anchor_source_sha,
                "candidate_source_tree_receipt": str(candidate_source_receipt),
                "candidate_source_tree_receipt_sha256": candidate_source_sha,
                "source_identity_receipts": {
                    "anchor": _bound(anchor_source_receipt, "source_identity"),
                    "candidate": _bound(candidate_source_receipt, "source_identity"),
                },
                "build_identity_files": {"anchor": anchor_files, "candidate": candidate_files},
                "correctness_capabilities": {
                    arm: _bound(values[2], "correctness_capability")
                    for arm, values in correctness_capabilities.items()},
                "shared_runtime": {
                    "measurement_binary": _bound(runtime.measurement_binary, "reward_binary"),
                    "runtime_receipt": _bound(runtime.receipt_path, "runtime_receipt"),
                    "anchor_hip_library": _bound(
                        (runtime.anchor_loader_dir / "libggml-hip.so.0").resolve(strict=True),
                        "runtime_hip"),
                    "candidate_hip_library": _bound(
                        (runtime.candidate_loader_dir / "libggml-hip.so.0").resolve(strict=True),
                        "runtime_hip"),
                },
                "reward_runtime_receipt": str(runtime.receipt_path),
                "reward_runtime_sha256": _digest(runtime.receipt_path),
                "promotion_claim": False,
            }
            materialization["receipt_sha256"] = schemas.content_hash(materialization)
            materialization_path = operation_dir / "materialization.json"
            materialization_path.write_text(json.dumps(materialization, sort_keys=True) + "\n", encoding="utf-8")
            completed = {
                "anchor_build": Path(by_id["akc-anchor"][1].path),
                "candidate_build": Path(by_id[candidate.source_manifest.candidate_id][1].path),
                "candidate_identity": candidate_identity, "anchor_identity": anchor_identity,
                "measurement_binary": runtime.measurement_binary,
                "common_loader_dir": runtime.common_loader_dir,
                "anchor_loader_dir": runtime.anchor_loader_dir,
                "candidate_loader_dir": runtime.candidate_loader_dir,
                "reward_runtime_sha256": _digest(runtime.receipt_path),
                "operation_key": None, "build_key": operation_key,
                "materialization_receipt": materialization_path,
                "materialization_sha256": _digest(materialization_path),
                "anchor_source_tree_receipt": anchor_source_receipt,
                "anchor_source_tree_sha256": anchor_source_sha,
                "candidate_source_tree_receipt": candidate_source_receipt,
                "candidate_source_tree_sha256": candidate_source_sha,
                "anchor_correctness_binary": correctness_capabilities["anchor"][0],
                "anchor_correctness_binary_sha256": correctness_capabilities["anchor"][1],
                "candidate_correctness_binary": correctness_capabilities["candidate"][0],
                "candidate_correctness_binary_sha256": correctness_capabilities["candidate"][1],
                "anchor_correctness_capability_receipt": correctness_capabilities["anchor"][2],
                "anchor_correctness_capability_sha256": correctness_capabilities["anchor"][3],
                "candidate_correctness_capability_receipt": correctness_capabilities["candidate"][2],
                "candidate_correctness_capability_sha256": correctness_capabilities["candidate"][3],
            }
        finally:
            receipts = []
            teardown_errors: list[str] = []
            for snapshot in snapshots:
                try:
                    receipts.append(worktree.teardown_worktree(snapshot).to_dict())
                except Exception as exc:  # retain all teardown attempts before refusing
                    teardown_errors.append(f"{snapshot.path.path}: {exc}")
            if actor is not None:
                try:
                    receipts.append(worktree.teardown_worktree(actor).to_dict())
                except Exception as exc:
                    teardown_errors.append(f"{actor.path.path}: {exc}")
            teardown = {"schema": _TEARDOWN_SCHEMA,
                        "operation_key": operation_key, "build_key": operation_key,
                        "receipts": receipts,
                        "errors": teardown_errors, "promotion_claim": False}
            teardown["receipt_sha256"] = schemas.content_hash(teardown)
            receipt_path = operation_dir / "teardown.json"
            receipt_path.write_text(json.dumps(teardown, sort_keys=True) + "\n", encoding="utf-8")
            if teardown_errors:
                raise StaticRegistryError("one or more governed worktrees could not be torn down")
        if completed is None:  # defensive: the originating build exception was re-raised by finally
            raise StaticRegistryError("static source build did not produce a complete result")
        return controller.GpuSourceBuild(**completed, teardown_receipt=receipt_path,
                                         teardown_sha256=_digest(receipt_path))
