"""Governed real-workload tensor capture for the INF-48 EPYC suite.

This module supplies the producer boundary that the offline C3 reducer cannot
provide.  Planning and receipt validation are inference-free.  The only call
that may execute a workload is :func:`execute_capture`, and it requires an
explicit ``authorize_inference=True`` argument.  A capture receipt establishes
tensor identity only; it is never correctness, performance, or promotion
evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import schemas
from . import c3_epyc_suite as c3


PLAN_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_plan.v1"
MANIFEST_SCHEMA = "epyc.autokernel.c3_epyc_tensor_manifest.v1"
RECEIPT_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_receipt.v1"
COMPLETION_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_completion.v1"
CAPTURE_KIND = "real_model_inference_tensor_capture"
AUTHORITY = "tensor_identity_only_no_correctness_speedup_or_promotion"
TARGET_ARCH = c3.TARGET_ARCH


class TensorCaptureRefusal(ValueError):
    """A tensor capture is not sufficiently identified or governed."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TensorCaptureRefusal(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if not schemas.SHA256_RE.fullmatch(value) or schemas.is_placeholder_digest(value):
        raise TensorCaptureRefusal(f"{label} must be a non-placeholder SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise TensorCaptureRefusal(f"{label} must be a full lowercase commit")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TensorCaptureRefusal(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise TensorCaptureRefusal(
            f"{label} fields differ from schema; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_regular_file(path: Path, expected_sha256: str, label: str,
                          *, root: Path | None = None) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise TensorCaptureRefusal(f"{label} must not be a symlink")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise TensorCaptureRefusal(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise TensorCaptureRefusal(f"{label} must be a singly-linked regular file")
    resolved = path.resolve()
    if root is not None and not resolved.is_relative_to(Path(root).resolve()):
        raise TensorCaptureRefusal(f"{label} escaped its capture root")
    observed = _sha256_file(resolved)
    if observed != _sha(expected_sha256, f"{label}.sha256"):
        raise TensorCaptureRefusal(
            f"{label} hash mismatch: expected {expected_sha256}, observed {observed}")
    return resolved


@dataclass(frozen=True)
class TensorSpec:
    name: str
    role: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.name, "tensor.name")
        _text(self.role, "tensor.role")
        _text(self.dtype, "tensor.dtype")
        if not self.shape or any(isinstance(item, bool) or not isinstance(item, int)
                                 or item <= 0 for item in self.shape):
            raise TensorCaptureRefusal("tensor.shape must contain positive integers")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "dtype": self.dtype,
                "shape": list(self.shape)}


@dataclass(frozen=True)
class CaptureSourceIdentity:
    repository_root: Path
    source_commit: str
    clean: bool
    producer_file: str
    producer_file_sha256: str
    producer_id: str

    def validate(self) -> Path:
        root = Path(self.repository_root)
        if not root.is_absolute() or not root.is_dir():
            raise TensorCaptureRefusal("source repository_root must be an absolute directory")
        root = root.resolve()
        _commit(self.source_commit, "source_commit")
        if self.clean is not True:
            raise TensorCaptureRefusal("tensor producer source tree must be clean")
        head = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"), text=True,
            capture_output=True, check=False)
        status_result = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"), text=True,
            capture_output=True, check=False)
        if head.returncode != 0 or status_result.returncode != 0:
            raise TensorCaptureRefusal("cannot inspect tensor producer source identity")
        if head.stdout.strip() != self.source_commit or status_result.stdout.strip():
            raise TensorCaptureRefusal(
                "tensor producer source tree differs from the declared clean commit")
        relative = Path(_text(self.producer_file, "producer_file"))
        if relative.is_absolute() or ".." in relative.parts:
            raise TensorCaptureRefusal("producer_file must remain inside its source tree")
        _text(self.producer_id, "producer_id")
        return _checked_regular_file(
            root / relative, self.producer_file_sha256, "tensor producer file", root=root)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_root": str(Path(self.repository_root).resolve()),
            "source_commit": self.source_commit,
            "clean": self.clean,
            "producer_file": self.producer_file,
            "producer_file_sha256": self.producer_file_sha256,
            "producer_id": self.producer_id,
        }


@dataclass(frozen=True)
class CaptureModelIdentity:
    model_id: str
    model_manifest: Path
    model_manifest_sha256: str
    model_sha256: str

    def validate(self) -> None:
        manifest_path = _checked_regular_file(
            Path(self.model_manifest), self.model_manifest_sha256, "model manifest")
        manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")),
                            "model manifest")
        _exact_keys(manifest, {"schema", "model_path", "files"}, "model manifest")
        if manifest["schema"] != "epyc.autokernel.model_identity.v1":
            raise TensorCaptureRefusal("unsupported model manifest schema")
        model_path = Path(_text(self.model_id, "model_id"))
        if not model_path.is_absolute() or not model_path.exists():
            raise TensorCaptureRefusal("model_id must be an existing absolute local path")
        model_path = model_path.resolve()
        if manifest["model_path"] != str(model_path):
            raise TensorCaptureRefusal("model manifest names a different model")
        rows = manifest["files"]
        if not isinstance(rows, list) or not rows:
            raise TensorCaptureRefusal("model manifest files must be a non-empty list")
        declared: dict[str, str] = {}
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"model manifest.files[{index}]")
            _exact_keys(row, {"path", "sha256"}, f"model manifest.files[{index}]")
            relative = _text(row["path"], f"model manifest.files[{index}].path")
            if relative in declared or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise TensorCaptureRefusal("model manifest contains unsafe or duplicate paths")
            declared[relative] = _sha(row["sha256"], f"model file {relative}.sha256")
        actual = ({".": model_path} if model_path.is_file() else {
            str(path.relative_to(model_path)): path
            for path in model_path.rglob("*") if path.is_file()
        })
        if set(declared) != set(actual):
            raise TensorCaptureRefusal("model manifest is not a complete exact file inventory")
        for relative, path in actual.items():
            _checked_regular_file(path, declared[relative], f"model file {relative}",
                                  root=model_path if model_path.is_dir() else model_path.parent)
        material = {"model_path": str(model_path), "files": [
            {"path": relative, "sha256": declared[relative]}
            for relative in sorted(declared)
        ]}
        if hashlib.sha256(_canonical(material).encode()).hexdigest() != _sha(
                self.model_sha256, "model_sha256"):
            raise TensorCaptureRefusal("model_sha256 differs from the complete model inventory")

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id,
                "model_manifest": str(Path(self.model_manifest).resolve()),
                "model_manifest_sha256": self.model_manifest_sha256,
                "model_sha256": self.model_sha256}


@dataclass(frozen=True)
class CaptureToolchainIdentity:
    manifest: Path
    manifest_sha256: str
    python_executable: Path
    python_executable_sha256: str
    torch_version: str
    hip_version: str
    triton_version: str

    def validate(self) -> None:
        manifest_path = _checked_regular_file(
            Path(self.manifest), self.manifest_sha256, "toolchain manifest")
        executable = _checked_regular_file(
            Path(self.python_executable), self.python_executable_sha256,
            "Python executable")
        if not os.access(executable, os.X_OK):
            raise TensorCaptureRefusal("Python executable is not executable")
        for value, label in ((self.torch_version, "torch_version"),
                             (self.hip_version, "hip_version"),
                             (self.triton_version, "triton_version")):
            _text(value, label)
        document = _read_json(manifest_path, "toolchain manifest")
        _exact_keys(document, {"schema", "python_executable",
                               "python_executable_sha256", "torch_version",
                               "hip_version", "triton_version"},
                    "toolchain manifest")
        expected = self.to_dict()
        if document["schema"] != "epyc.autokernel.c3_epyc_capture_toolchain.v1" \
                or any(document[key] != expected[key] for key in (
                    "python_executable", "python_executable_sha256", "torch_version",
                    "hip_version", "triton_version")):
            raise TensorCaptureRefusal("toolchain manifest differs from declared identity")

    def to_dict(self) -> dict[str, str]:
        return {"manifest": str(Path(self.manifest).resolve()),
                "manifest_sha256": self.manifest_sha256,
                "python_executable": str(Path(self.python_executable).resolve()),
                "python_executable_sha256": self.python_executable_sha256,
                "torch_version": self.torch_version, "hip_version": self.hip_version,
                "triton_version": self.triton_version}


@dataclass(frozen=True)
class TensorCapturePlan:
    case_id: str
    workload_id: str
    stage: str
    token_count: int
    device_id: str
    source: CaptureSourceIdentity
    model: CaptureModelIdentity
    toolchain: CaptureToolchainIdentity
    recipe_ref: str
    recipe_sha256: str
    tensors: tuple[TensorSpec, ...]
    output_root: Path
    timeout_seconds: int
    plan_sha256: str

    @property
    def dispatch_branch(self) -> str:
        return "n_le_1350" if self.token_count <= 1350 else "n_gt_1350"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA, "capture_kind": CAPTURE_KIND,
            "case_id": self.case_id, "workload_id": self.workload_id,
            "stage": self.stage, "token_count": self.token_count,
            "dispatch_branch": self.dispatch_branch,
            "architecture": TARGET_ARCH, "device_id": self.device_id,
            "source": self.source.to_dict(), "model": self.model.to_dict(),
            "toolchain": self.toolchain.to_dict(), "recipe_ref": self.recipe_ref,
            "recipe_sha256": self.recipe_sha256,
            "tensors": [item.to_dict() for item in self.tensors],
            "output_root": str(Path(self.output_root).resolve()),
            "timeout_seconds": self.timeout_seconds,
            "authority": AUTHORITY, "plan_sha256": self.plan_sha256,
        }


def prepare_capture_plan(*, case_id: str, workload_id: str, stage: str,
                         token_count: int, device_id: str,
                         source: CaptureSourceIdentity, model: CaptureModelIdentity,
                         toolchain: CaptureToolchainIdentity, recipe_ref: str,
                         recipe_sha256: str, tensors: Sequence[TensorSpec],
                         output_root: Path, timeout_seconds: int = 3600) -> TensorCapturePlan:
    """Validate immutable inputs and compile a prospective capture plan."""
    if case_id not in {case.case_id for case in c3.epyc_op_suite()}:
        raise TensorCaptureRefusal("capture plan names a case outside the exact EPYC suite")
    if stage not in {"prefill", "decode"}:
        raise TensorCaptureRefusal("capture stage must be prefill or decode")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise TensorCaptureRefusal("token_count must be a positive integer")
    source.validate()
    model.validate()
    toolchain.validate()
    recipe_path = Path(_text(recipe_ref, "recipe_ref"))
    if not recipe_path.is_absolute():
        raise TensorCaptureRefusal("capture recipe_ref must be an absolute file")
    recipe_path = _checked_regular_file(
        recipe_path, recipe_sha256, "capture recipe")
    tensor_tuple = tuple(tensors)
    if not tensor_tuple or len({item.name for item in tensor_tuple}) != len(tensor_tuple):
        raise TensorCaptureRefusal("capture tensors must be non-empty and uniquely named")
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise TensorCaptureRefusal("capture output_root must be absolute")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise TensorCaptureRefusal("capture output_root must be absent or empty")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int)
            or timeout_seconds <= 0 or timeout_seconds > 21600):
        raise TensorCaptureRefusal("timeout_seconds must be in 1..21600")
    material = {
        "schema": PLAN_SCHEMA, "capture_kind": CAPTURE_KIND,
        "case_id": _text(case_id, "case_id"),
        "workload_id": _text(workload_id, "workload_id"), "stage": stage,
        "token_count": token_count,
        "dispatch_branch": "n_le_1350" if token_count <= 1350 else "n_gt_1350",
        "architecture": TARGET_ARCH, "device_id": _text(device_id, "device_id"),
        "source": source.to_dict(), "model": model.to_dict(),
        "toolchain": toolchain.to_dict(), "recipe_ref": str(recipe_path),
        "recipe_sha256": _sha(recipe_sha256, "recipe_sha256"),
        "tensors": [item.to_dict() for item in tensor_tuple],
        "output_root": str(output_root.resolve()), "timeout_seconds": timeout_seconds,
        "authority": AUTHORITY,
    }
    digest = hashlib.sha256(_canonical(material).encode()).hexdigest()
    return TensorCapturePlan(
        case_id=case_id, workload_id=workload_id, stage=stage,
        token_count=token_count, device_id=device_id, source=source, model=model,
        toolchain=toolchain, recipe_ref=str(recipe_path), recipe_sha256=recipe_sha256,
        tensors=tensor_tuple, output_root=output_root.resolve(),
        timeout_seconds=timeout_seconds, plan_sha256=digest)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, json.JSONDecodeError) as exc:
        raise TensorCaptureRefusal(f"cannot read {label}: {exc}") from exc


def bind_capture_outputs(plan: TensorCapturePlan) -> dict[str, Any]:
    """Re-hash exact tensor bytes and project an identity-only receipt."""
    manifest_path = Path(plan.output_root) / "captured_tensor_manifest.json"
    manifest = _read_json(manifest_path, "captured tensor manifest")
    required = {"schema", "capture_kind", "synthetic", "plan_sha256", "case_id",
                "workload_id", "model_sha256", "source_commit",
                "toolchain_manifest_sha256", "architecture", "device_id", "stage",
                "token_count", "dispatch_branch", "tensors"}
    _exact_keys(manifest, required, "captured tensor manifest")
    expected_header = {
        "schema": MANIFEST_SCHEMA, "capture_kind": CAPTURE_KIND, "synthetic": False,
        "plan_sha256": plan.plan_sha256, "case_id": plan.case_id,
        "workload_id": plan.workload_id, "model_sha256": plan.model.model_sha256,
        "source_commit": plan.source.source_commit,
        "toolchain_manifest_sha256": plan.toolchain.manifest_sha256,
        "architecture": TARGET_ARCH, "device_id": plan.device_id, "stage": plan.stage,
        "token_count": plan.token_count, "dispatch_branch": plan.dispatch_branch,
    }
    drift = [key for key, value in expected_header.items() if manifest.get(key) != value]
    if drift:
        raise TensorCaptureRefusal(f"captured tensor manifest identity drifted at {drift}")
    rows = manifest["tensors"]
    if not isinstance(rows, list) or len(rows) != len(plan.tensors):
        raise TensorCaptureRefusal("captured tensor manifest has the wrong tensor count")
    observed = []
    for index, (raw, spec) in enumerate(zip(rows, plan.tensors)):
        label = f"captured tensor manifest.tensors[{index}]"
        row = _mapping(raw, label)
        _exact_keys(row, {"name", "role", "dtype", "shape", "path", "nbytes",
                          "sha256"}, label)
        if {key: row[key] for key in ("name", "role", "dtype", "shape")} != spec.to_dict():
            raise TensorCaptureRefusal(f"{label} identity or order differs from the plan")
        relative = Path(_text(row["path"], f"{label}.path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise TensorCaptureRefusal(f"{label}.path must remain inside output_root")
        path = _checked_regular_file(plan.output_root / relative, row["sha256"],
                                     f"captured tensor {spec.name}", root=plan.output_root)
        nbytes = row["nbytes"]
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0 \
                or path.stat().st_size != nbytes:
            raise TensorCaptureRefusal(f"captured tensor {spec.name} size differs from manifest")
        observed.append(dict(row))
    manifest_sha256 = _sha256_file(manifest_path)
    receipt = {
        "schema": RECEIPT_SCHEMA, "capture_kind": CAPTURE_KIND,
        "authority": AUTHORITY, "plan_sha256": plan.plan_sha256,
        "case_id": plan.case_id, "workload_id": plan.workload_id,
        "model_sha256": plan.model.model_sha256,
        "source_commit": plan.source.source_commit,
        "producer_file_sha256": plan.source.producer_file_sha256,
        "toolchain_manifest_sha256": plan.toolchain.manifest_sha256,
        "architecture": TARGET_ARCH, "device_id": plan.device_id,
        "stage": plan.stage, "token_count": plan.token_count,
        "dispatch_branch": plan.dispatch_branch,
        "tensor_manifest": str(manifest_path.resolve()),
        "tensor_manifest_sha256": manifest_sha256, "tensors": observed,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt).encode()).hexdigest()
    return receipt


def _run_producer(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(tuple(argv), text=True, capture_output=True, check=False, **kwargs)


def execute_capture(
        plan: TensorCapturePlan, *, authorize_inference: bool = False,
        run: Callable[..., subprocess.CompletedProcess[str]] = _run_producer,
        environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Execute the exact pinned producer and bind outputs after authorization.

    The producer receives the canonical plan JSON on stdin and must implement
    the ``--epyc-c3-tensor-capture-v1`` protocol.  The injected ``run`` seam is
    for process-isolation tests; it cannot change the exact argv or completion
    acknowledgement this function validates.
    """
    if not authorize_inference:
        raise TensorCaptureRefusal("tensor capture requires explicit inference authorization")
    # Revalidate mutable identities immediately before and after the producer.
    plan.source.validate()
    plan.model.validate()
    plan.toolchain.validate()
    producer_path = (Path(plan.source.repository_root).resolve()
                     / plan.source.producer_file).resolve()
    environment_source = os.environ if environ is None else environ
    if environment_source.get("HSA_OVERRIDE_GFX_VERSION"):
        raise TensorCaptureRefusal("HSA_OVERRIDE_GFX_VERSION would spoof physical gfx90a")
    child_environment = {
        key: environment_source[key]
        for key in ("PATH", "LD_LIBRARY_PATH", "ROCM_PATH", "HIP_VISIBLE_DEVICES")
        if key in environment_source
    }
    child_environment["PYTHONNOUSERSITE"] = "1"
    argv = (
        str(Path(plan.toolchain.python_executable).resolve()), str(producer_path),
        "--epyc-c3-tensor-capture-v1", "--output-root", str(plan.output_root),
    )
    try:
        result = run(
            argv, input=_canonical(plan.to_dict()) + "\n",
            cwd=str(Path(plan.source.repository_root).resolve()),
            env=child_environment, timeout=plan.timeout_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TensorCaptureRefusal(f"tensor producer process failed: {exc}") from exc
    if not isinstance(result, subprocess.CompletedProcess) or result.returncode != 0:
        code = getattr(result, "returncode", "invalid-result")
        raise TensorCaptureRefusal(f"tensor producer exited nonzero: {code}")
    try:
        completion = _mapping(json.loads(result.stdout), "tensor producer completion")
    except (TypeError, json.JSONDecodeError) as exc:
        raise TensorCaptureRefusal("tensor producer completion is not valid JSON") from exc
    _exact_keys(completion, {"schema", "plan_sha256", "output_root"},
                "tensor producer completion")
    if completion != {"schema": COMPLETION_SCHEMA,
                       "plan_sha256": plan.plan_sha256,
                       "output_root": str(plan.output_root)}:
        raise TensorCaptureRefusal("tensor producer completion differs from the plan")
    plan.source.validate()
    plan.model.validate()
    plan.toolchain.validate()
    receipt = bind_capture_outputs(plan)
    receipt_path = Path(plan.output_root) / "tensor_capture_receipt.json"
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    descriptor: int | None = None
    try:
        descriptor = os.open(
            receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except OSError as exc:
        raise TensorCaptureRefusal(f"cannot publish exclusive tensor capture receipt: {exc}") \
            from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return receipt


def load_capture_receipt(path: Path) -> Mapping[str, Any]:
    """Load and re-hash a capture receipt, manifest, and every tensor file."""
    receipt = _read_json(Path(path), "tensor capture receipt")
    required = {"schema", "capture_kind", "authority", "plan_sha256", "case_id",
                "workload_id", "model_sha256", "source_commit", "producer_file_sha256",
                "toolchain_manifest_sha256", "architecture", "device_id", "stage",
                "token_count", "dispatch_branch", "tensor_manifest",
                "tensor_manifest_sha256", "tensors", "receipt_sha256"}
    _exact_keys(receipt, required, "tensor capture receipt")
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["capture_kind"] != CAPTURE_KIND \
            or receipt["authority"] != AUTHORITY or receipt["architecture"] != TARGET_ARCH:
        raise TensorCaptureRefusal("tensor capture receipt overstates or drifts authority")
    for field in ("plan_sha256", "model_sha256", "producer_file_sha256",
                  "toolchain_manifest_sha256", "tensor_manifest_sha256"):
        _sha(receipt[field], field)
    _commit(receipt["source_commit"], "source_commit")
    _text(receipt["case_id"], "case_id")
    _text(receipt["workload_id"], "workload_id")
    _text(receipt["device_id"], "device_id")
    if receipt["stage"] not in {"prefill", "decode"}:
        raise TensorCaptureRefusal("receipt stage must be prefill or decode")
    token_count = receipt["token_count"]
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise TensorCaptureRefusal("receipt token_count must be a positive integer")
    expected_branch = "n_le_1350" if token_count <= 1350 else "n_gt_1350"
    if receipt["dispatch_branch"] != expected_branch:
        raise TensorCaptureRefusal("receipt dispatch branch disagrees with token_count")
    material = dict(receipt)
    claimed = _sha(material.pop("receipt_sha256"), "receipt_sha256")
    if hashlib.sha256(_canonical(material).encode()).hexdigest() != claimed:
        raise TensorCaptureRefusal("tensor capture receipt self-hash mismatch")
    manifest_path = _checked_regular_file(
        Path(_text(receipt["tensor_manifest"], "tensor_manifest")),
        receipt["tensor_manifest_sha256"], "captured tensor manifest")
    manifest = _read_json(manifest_path, "captured tensor manifest")
    _exact_keys(manifest, {"schema", "capture_kind", "synthetic", "plan_sha256",
                           "case_id", "workload_id", "model_sha256", "source_commit",
                           "toolchain_manifest_sha256", "architecture", "device_id",
                           "stage", "token_count", "dispatch_branch", "tensors"},
                "captured tensor manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise TensorCaptureRefusal("unsupported captured tensor manifest schema")
    if manifest.get("synthetic") is not False or manifest.get("capture_kind") != CAPTURE_KIND:
        raise TensorCaptureRefusal("synthetic or non-real tensor manifest is inadmissible")
    shared_fields = ("plan_sha256", "case_id", "workload_id", "model_sha256",
                     "source_commit", "toolchain_manifest_sha256", "architecture",
                     "device_id", "stage", "token_count", "dispatch_branch")
    drift = [field for field in shared_fields
             if manifest.get(field) != receipt.get(field)]
    if drift:
        raise TensorCaptureRefusal(
            f"receipt and tensor manifest identity drifted at {drift}")
    rows = receipt["tensors"]
    if not isinstance(rows, list) or not rows or manifest.get("tensors") != rows:
        raise TensorCaptureRefusal("receipt and tensor manifest rows differ")
    names: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"receipt.tensors[{index}]")
        _exact_keys(row, {"name", "role", "dtype", "shape", "path", "nbytes",
                          "sha256"}, f"receipt.tensors[{index}]")
        if not isinstance(row["shape"], list):
            raise TensorCaptureRefusal("receipt tensor shape must be a list")
        spec = TensorSpec(_text(row["name"], "tensor.name"),
                          _text(row["role"], "tensor.role"),
                          _text(row["dtype"], "tensor.dtype"), tuple(row["shape"]))
        if spec.name in names:
            raise TensorCaptureRefusal("receipt repeats a tensor name")
        names.add(spec.name)
        if isinstance(row["nbytes"], bool) or not isinstance(row["nbytes"], int) \
                or row["nbytes"] <= 0:
            raise TensorCaptureRefusal("receipt tensor nbytes must be positive")
        _sha(row["sha256"], "tensor.sha256")
        tensor_path = Path(row["path"])
        if not tensor_path.is_absolute():
            tensor_path = manifest_path.parent / tensor_path
        path_checked = _checked_regular_file(
            tensor_path, row["sha256"], f"receipt tensor {index}", root=manifest_path.parent)
        if path_checked.stat().st_size != row["nbytes"]:
            raise TensorCaptureRefusal(f"receipt tensor {index} size mismatch")
    return receipt


__all__ = [
    "AUTHORITY", "CAPTURE_KIND", "COMPLETION_SCHEMA", "MANIFEST_SCHEMA", "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
    "CaptureModelIdentity", "CaptureSourceIdentity", "CaptureToolchainIdentity",
    "TensorCapturePlan", "TensorCaptureRefusal", "TensorSpec", "bind_capture_outputs",
    "execute_capture", "load_capture_receipt", "prepare_capture_plan",
]
