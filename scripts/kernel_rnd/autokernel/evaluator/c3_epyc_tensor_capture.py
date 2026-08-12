"""Governed real-workload tensor capture for the INF-48 EPYC suite.

This module supplies the producer boundary that the offline C3 reducer cannot
provide.  Planning and receipt validation are inference-free.  The only call
that may execute a workload is :func:`execute_capture`, and it requires explicit
authorization plus live, identity-matched CPU and MI210 claims.  A capture
receipt establishes tensor identity only; it is never correctness, performance,
or promotion evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import schemas
from ..execution import cpu_region_claim
from ..execution import device_sampler
from ..resource import device_claim
from . import c3_epyc_suite as c3


REQUEST_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_request.v1"
PLAN_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_plan.v2"
MANIFEST_SCHEMA = "epyc.autokernel.c3_epyc_tensor_manifest.v1"
RECEIPT_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_receipt.v1"
COMPLETION_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_completion.v1"
CAPTURE_KIND = "real_model_inference_tensor_capture"
AUTHORITY = "tensor_identity_only_no_correctness_speedup_or_promotion"
TARGET_ARCH = c3.TARGET_ARCH
WINDOW_SCHEMA = "epyc.autokernel.c3_epyc_tensor_capture_window.v2"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_TENSORS = 128
MAX_TENSOR_RANK = 8
MAX_TENSOR_BYTES = 8 * 1024 * 1024 * 1024
MAX_CAPTURE_BYTES = 64 * 1024 * 1024 * 1024
MAX_PROCESS_OUTPUT_BYTES = 1024 * 1024


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


def _reject_governed_path(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if ".git" in resolved.parts:
        raise TensorCaptureRefusal(f"{label} must not be inside .git")
    frozen = Path("/mnt/raid0/llm/llama.cpp").resolve()
    if resolved == frozen or resolved.is_relative_to(frozen):
        raise TensorCaptureRefusal(f"{label} must not touch the frozen production tree")
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
        if len(self.shape) > MAX_TENSOR_RANK:
            raise TensorCaptureRefusal(f"tensor rank exceeds {MAX_TENSOR_RANK}")
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
        root = _reject_governed_path(Path(self.repository_root), "source repository_root")
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
        model_path = _reject_governed_path(Path(_text(self.model_id, "model_id")), "model_id")
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
    campaign_id: str
    case_id: str
    workload_id: str
    stage: str
    token_count: int
    device_id: str
    device_claim_id: str
    device_visible_ordinal: int
    device_inventory: Path
    device_inventory_sha256: str
    cpu_list: str
    source: CaptureSourceIdentity
    model: CaptureModelIdentity
    toolchain: CaptureToolchainIdentity
    recipe_ref: str
    recipe_sha256: str
    tensors: tuple[TensorSpec, ...]
    output_root: Path
    timeout_seconds: int
    capture_command: tuple[str, ...]
    runtime_environment: Mapping[str, str]
    plan_sha256: str

    @property
    def dispatch_branch(self) -> str:
        return "n_le_1350" if self.token_count <= 1350 else "n_gt_1350"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA, "capture_kind": CAPTURE_KIND,
            "campaign_id": self.campaign_id, "case_id": self.case_id,
            "workload_id": self.workload_id,
            "stage": self.stage, "token_count": self.token_count,
            "dispatch_branch": self.dispatch_branch,
            "architecture": TARGET_ARCH, "device_id": self.device_id,
            "device_claim_id": self.device_claim_id,
            "device_visible_ordinal": self.device_visible_ordinal,
            "device_inventory": str(Path(self.device_inventory).resolve()),
            "device_inventory_sha256": self.device_inventory_sha256,
            "cpu_list": self.cpu_list,
            "source": self.source.to_dict(), "model": self.model.to_dict(),
            "toolchain": self.toolchain.to_dict(), "recipe_ref": self.recipe_ref,
            "recipe_sha256": self.recipe_sha256,
            "tensors": [item.to_dict() for item in self.tensors],
            "output_root": str(Path(self.output_root).resolve()),
            "timeout_seconds": self.timeout_seconds,
            "capture_command": list(self.capture_command),
            "runtime_environment": dict(self.runtime_environment),
            "authority": AUTHORITY, "plan_sha256": self.plan_sha256,
        }


def prepare_capture_plan(*, campaign_id: str | None = None, case_id: str,
                         workload_id: str, stage: str,
                         token_count: int, device_id: str,
                         device_claim_id: str = "mi210_0",
                         device_visible_ordinal: int = 0,
                         device_inventory: Path | None = None,
                         device_inventory_sha256: str | None = None,
                         cpu_list: str | None = None,
                         source: CaptureSourceIdentity, model: CaptureModelIdentity,
                         toolchain: CaptureToolchainIdentity, recipe_ref: str,
                         recipe_sha256: str, tensors: Sequence[TensorSpec],
                         output_root: Path, timeout_seconds: int = 3600,
                         capture_command: Sequence[str] | None = None,
                         runtime_environment: Mapping[str, str] | None = None) -> TensorCapturePlan:
    """Validate immutable inputs and compile a prospective capture plan."""
    if case_id not in {case.case_id for case in c3.epyc_op_suite()}:
        raise TensorCaptureRefusal("capture plan names a case outside the exact EPYC suite")
    if stage not in {"prefill", "decode"}:
        raise TensorCaptureRefusal("capture stage must be prefill or decode")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise TensorCaptureRefusal("token_count must be a positive integer")
    campaign_id = _text(campaign_id or workload_id, "campaign_id")
    if device_claim_id != "mi210_0":
        raise TensorCaptureRefusal("EPYC C3/C5 capture requires logical device claim mi210_0")
    if isinstance(device_visible_ordinal, bool) or device_visible_ordinal != 0:
        raise TensorCaptureRefusal("EPYC C3/C5 capture requires MI210 visible ordinal 0")
    if device_inventory is None or device_inventory_sha256 is None:
        raise TensorCaptureRefusal("capture requires a hash-bound device inventory")
    inventory_path = _checked_regular_file(
        Path(device_inventory), device_inventory_sha256, "device inventory")
    inventory = _read_json(inventory_path, "device inventory")
    _exact_keys(inventory, {"schema", "logical_device_id", "pci_bdf",
                            "visible_ordinal", "architecture"}, "device inventory")
    if inventory != {"schema": "epyc.autokernel.device_inventory.v1",
                      "logical_device_id": device_claim_id, "pci_bdf": device_id,
                      "visible_ordinal": device_visible_ordinal,
                      "architecture": TARGET_ARCH}:
        raise TensorCaptureRefusal("device inventory differs from logical/physical plan")
    cpu_list = cpu_region_claim.gpu_host_cpu_list() if cpu_list is None else cpu_list
    if cpu_list != cpu_region_claim.gpu_host_cpu_list():
        raise TensorCaptureRefusal(
            "EPYC C3/C5 capture requires the codified MI210 host CPU footprint")
    source.validate()
    model.validate()
    toolchain.validate()
    recipe_path = Path(_text(recipe_ref, "recipe_ref"))
    if not recipe_path.is_absolute():
        raise TensorCaptureRefusal("capture recipe_ref must be an absolute file")
    recipe_path = _checked_regular_file(
        recipe_path, recipe_sha256, "capture recipe")
    tensor_tuple = tuple(tensors)
    if (not tensor_tuple or len(tensor_tuple) > MAX_TENSORS
            or len({item.name for item in tensor_tuple}) != len(tensor_tuple)):
        raise TensorCaptureRefusal("capture tensors must be non-empty and uniquely named")
    raw_output_root = Path(output_root)
    if raw_output_root.is_symlink():
        raise TensorCaptureRefusal("capture output_root must not be a symlink")
    output_root = _reject_governed_path(raw_output_root, "capture output_root")
    if not output_root.is_absolute():
        raise TensorCaptureRefusal("capture output_root must be absolute")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise TensorCaptureRefusal("capture output_root must be absent or empty")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int)
            or timeout_seconds <= 0 or timeout_seconds > 21600):
        raise TensorCaptureRefusal("timeout_seconds must be in 1..21600")
    producer_path = (Path(source.repository_root).resolve() / source.producer_file).resolve()
    command = tuple(capture_command or (
        str(Path(toolchain.python_executable).resolve()), str(producer_path)))
    expected_command = (str(Path(toolchain.python_executable).resolve()), str(producer_path))
    if command != expected_command:
        raise TensorCaptureRefusal(
            "capture_command must name the exact pinned Python and producer file")
    runtime = dict(runtime_environment or {})
    if set(runtime) != {"LD_LIBRARY_PATH", "ROCM_PATH"}:
        raise TensorCaptureRefusal(
            "runtime_environment must bind exact LD_LIBRARY_PATH and ROCM_PATH")
    rocm_path = Path(_text(runtime["ROCM_PATH"], "ROCM_PATH"))
    if not rocm_path.is_absolute() or not rocm_path.is_dir():
        raise TensorCaptureRefusal("ROCM_PATH must be an existing absolute directory")
    ld_entries = runtime["LD_LIBRARY_PATH"].split(":")
    if not ld_entries or any(not Path(item).is_absolute() or not Path(item).is_dir()
                             for item in ld_entries):
        raise TensorCaptureRefusal("LD_LIBRARY_PATH must contain existing absolute directories")
    material = {
        "schema": PLAN_SCHEMA, "capture_kind": CAPTURE_KIND,
        "campaign_id": campaign_id, "case_id": _text(case_id, "case_id"),
        "workload_id": _text(workload_id, "workload_id"), "stage": stage,
        "token_count": token_count,
        "dispatch_branch": "n_le_1350" if token_count <= 1350 else "n_gt_1350",
        "architecture": TARGET_ARCH, "device_id": _text(device_id, "device_id"),
        "device_claim_id": device_claim_id,
        "device_visible_ordinal": device_visible_ordinal, "cpu_list": cpu_list,
        "device_inventory": str(inventory_path),
        "device_inventory_sha256": _sha(device_inventory_sha256,
                                         "device_inventory_sha256"),
        "source": source.to_dict(), "model": model.to_dict(),
        "toolchain": toolchain.to_dict(), "recipe_ref": str(recipe_path),
        "recipe_sha256": _sha(recipe_sha256, "recipe_sha256"),
        "tensors": [item.to_dict() for item in tensor_tuple],
        "output_root": str(output_root.resolve()), "timeout_seconds": timeout_seconds,
        "capture_command": list(command),
        "runtime_environment": runtime,
        "authority": AUTHORITY,
    }
    digest = hashlib.sha256(_canonical(material).encode()).hexdigest()
    return TensorCapturePlan(
        campaign_id=campaign_id, case_id=case_id, workload_id=workload_id, stage=stage,
        token_count=token_count, device_id=device_id, source=source, model=model,
        device_claim_id=device_claim_id, device_visible_ordinal=device_visible_ordinal,
        device_inventory=inventory_path,
        device_inventory_sha256=device_inventory_sha256, cpu_list=cpu_list,
        toolchain=toolchain, recipe_ref=str(recipe_path), recipe_sha256=recipe_sha256,
        tensors=tensor_tuple, output_root=output_root.resolve(),
        timeout_seconds=timeout_seconds, capture_command=command,
        runtime_environment=runtime, plan_sha256=digest)


_REQUEST_FIELDS = {
    "schema", "campaign_id", "case_id", "workload_id", "stage", "token_count",
    "architecture", "device_id", "device_claim_id", "device_visible_ordinal",
    "cpu_list", "source", "model", "toolchain", "recipe_ref", "recipe_sha256",
    "device_inventory", "device_inventory_sha256", "tensors", "output_root",
    "timeout_seconds", "capture_command", "runtime_environment",
}


def _compile_request(document: Mapping[str, Any]) -> TensorCapturePlan:
    _exact_keys(document, _REQUEST_FIELDS, "tensor capture request")
    if document["schema"] != REQUEST_SCHEMA:
        raise TensorCaptureRefusal("unsupported tensor capture request schema")
    if document["architecture"] != TARGET_ARCH:
        raise TensorCaptureRefusal("tensor capture request must name physical gfx90a")
    source_raw = _mapping(document["source"], "source")
    _exact_keys(source_raw, {"repository_root", "source_commit", "clean",
                             "producer_file", "producer_file_sha256", "producer_id"},
                "source")
    model_raw = _mapping(document["model"], "model")
    _exact_keys(model_raw, {"model_id", "model_manifest", "model_manifest_sha256",
                            "model_sha256"}, "model")
    toolchain_raw = _mapping(document["toolchain"], "toolchain")
    _exact_keys(toolchain_raw, {"manifest", "manifest_sha256", "python_executable",
                                "python_executable_sha256", "torch_version",
                                "hip_version", "triton_version"}, "toolchain")
    tensor_rows = document["tensors"]
    if not isinstance(tensor_rows, list):
        raise TensorCaptureRefusal("tensors must be a list")
    tensors: list[TensorSpec] = []
    for index, raw in enumerate(tensor_rows):
        row = _mapping(raw, f"tensors[{index}]")
        _exact_keys(row, {"name", "role", "dtype", "shape"}, f"tensors[{index}]")
        if not isinstance(row["shape"], list):
            raise TensorCaptureRefusal(f"tensors[{index}].shape must be a list")
        tensors.append(TensorSpec(row["name"], row["role"], row["dtype"],
                                  tuple(row["shape"])))
    command = document["capture_command"]
    if not isinstance(command, list) or not command \
            or any(not isinstance(item, str) or not item for item in command):
        raise TensorCaptureRefusal("capture_command must be a non-empty string list")
    return prepare_capture_plan(
        campaign_id=document["campaign_id"], case_id=document["case_id"],
        workload_id=document["workload_id"], stage=document["stage"],
        token_count=document["token_count"], device_id=document["device_id"],
        device_claim_id=document["device_claim_id"],
        device_visible_ordinal=document["device_visible_ordinal"],
        device_inventory=Path(document["device_inventory"]),
        device_inventory_sha256=document["device_inventory_sha256"],
        cpu_list=document["cpu_list"],
        source=CaptureSourceIdentity(
            repository_root=Path(source_raw["repository_root"]),
            source_commit=source_raw["source_commit"], clean=source_raw["clean"],
            producer_file=source_raw["producer_file"],
            producer_file_sha256=source_raw["producer_file_sha256"],
            producer_id=source_raw["producer_id"]),
        model=CaptureModelIdentity(
            model_id=model_raw["model_id"], model_manifest=Path(model_raw["model_manifest"]),
            model_manifest_sha256=model_raw["model_manifest_sha256"],
            model_sha256=model_raw["model_sha256"]),
        toolchain=CaptureToolchainIdentity(
            manifest=Path(toolchain_raw["manifest"]),
            manifest_sha256=toolchain_raw["manifest_sha256"],
            python_executable=Path(toolchain_raw["python_executable"]),
            python_executable_sha256=toolchain_raw["python_executable_sha256"],
            torch_version=toolchain_raw["torch_version"],
            hip_version=toolchain_raw["hip_version"],
            triton_version=toolchain_raw["triton_version"]),
        recipe_ref=document["recipe_ref"], recipe_sha256=document["recipe_sha256"],
        tensors=tensors, output_root=Path(document["output_root"]),
        timeout_seconds=document["timeout_seconds"], capture_command=command,
        runtime_environment=_mapping(document["runtime_environment"],
                                     "runtime_environment"))


def compile_capture_manifest(path: Path) -> TensorCapturePlan:
    """Compile an exact request manifest without executing inference."""
    return _compile_request(_read_json(Path(path), "tensor capture request"))


def load_capture_plan(path: Path) -> TensorCapturePlan:
    """Reload a compiled plan and prove that its self-hash and inputs still match."""
    document = _read_json(Path(path), "tensor capture plan")
    required = (_REQUEST_FIELDS - {"schema"}) | {
        "schema", "capture_kind", "dispatch_branch", "authority", "plan_sha256"}
    _exact_keys(document, required, "tensor capture plan")
    request = {key: document[key] for key in _REQUEST_FIELDS}
    request["schema"] = REQUEST_SCHEMA
    plan = _compile_request(request)
    if document != plan.to_dict():
        raise TensorCaptureRefusal("compiled tensor capture plan identity drifted")
    return plan


def _publish_json_exclusive(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o444)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except OSError as exc:
        raise TensorCaptureRefusal(f"cannot publish exclusive {path.name}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise TensorCaptureRefusal(f"{label} exceeds {MAX_JSON_BYTES} bytes")
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
    if (not isinstance(rows, list) or len(rows) != len(plan.tensors)
            or len(rows) > MAX_TENSORS):
        raise TensorCaptureRefusal("captured tensor manifest has the wrong tensor count")
    observed = []
    total_bytes = 0
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
                or nbytes > MAX_TENSOR_BYTES or path.stat().st_size != nbytes:
            raise TensorCaptureRefusal(f"captured tensor {spec.name} size differs from manifest")
        total_bytes += nbytes
        if total_bytes > MAX_CAPTURE_BYTES:
            raise TensorCaptureRefusal("captured tensors exceed the total byte ceiling")
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
    """Run one captured process group with bounded output and KFD overlap sampling."""
    input_text = kwargs.pop("input", None)
    timeout = kwargs.pop("timeout", None)
    poll_check = kwargs.pop("poll_check", None)
    process = subprocess.Popen(
        tuple(argv), text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True, **kwargs)
    pgid = process.pid
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    overflow: set[str] = set()
    residency: list[dict[str, Any]] = []
    stop = threading.Event()
    monitor_failure: list[str] = []

    def drain(name: str, stream: Any) -> None:
        while True:
            chunk = stream.buffer.read(65536)
            if not chunk:
                return
            remaining = MAX_PROCESS_OUTPUT_BYTES - len(buffers[name])
            if remaining > 0:
                buffers[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow.add(name)

    def sample_kfd() -> None:
        started = time.monotonic()
        while not stop.wait(0.05):
            users: list[int] = []
            for raw in Path("/proc").iterdir():
                if not raw.name.isdigit():
                    continue
                pid = int(raw.name)
                try:
                    if os.getpgid(pid) != pgid:
                        continue
                    if any(path.resolve() == Path("/dev/kfd")
                           for path in (raw / "fd").iterdir()):
                        users.append(pid)
                except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                    continue
            residency.append({"offset_s": time.monotonic() - started,
                              "kfd_pids": sorted(users)})

    def monitor_boundary() -> None:
        if poll_check is None:
            return
        while not stop.wait(0.25):
            try:
                poll_check()
            except Exception as exc:  # fail closed and stop only our process group
                monitor_failure.append(f"{type(exc).__name__}: {exc}")
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return

    readers = [threading.Thread(target=drain, args=(name, stream), daemon=True)
               for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))]
    sampler = threading.Thread(target=sample_kfd, daemon=True)
    monitor = threading.Thread(target=monitor_boundary, daemon=True)
    for thread in readers:
        thread.start()
    sampler.start()
    monitor.start()
    failure: BaseException | None = None
    try:
        assert process.stdin is not None
        process.stdin.write(input_text or "")
        process.stdin.close()
        process.wait(timeout=timeout)
    except BaseException as exc:
        failure = exc
    finally:
        stop.set()
        sampler.join(timeout=2)
        monitor.join(timeout=2)
        # A successful parent may still have left descendants. Always terminate
        # the exact captured group, then prove that no member remains.
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.killpg(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
        for thread in readers:
            thread.join(timeout=2)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            pass
        else:
            raise TensorCaptureRefusal("tensor producer process group survived teardown")
    if failure is not None:
        raise failure
    if monitor_failure:
        raise TensorCaptureRefusal(
            "resource claim/revocation poll failed: " + "; ".join(monitor_failure))
    if overflow:
        raise TensorCaptureRefusal(
            "tensor producer output exceeded cap: " + ", ".join(sorted(overflow)))
    result = subprocess.CompletedProcess(
        tuple(argv), process.returncode,
        buffers["stdout"].decode("utf-8", errors="strict"),
        buffers["stderr"].decode("utf-8", errors="replace"))
    result.residency_witness = {
        "schema": "epyc.autokernel.kfd_process_group_witness.v1",
        "process_group_id": pgid, "samples": residency,
        "overlap_observed": any(row["kfd_pids"] for row in residency),
    }
    return result


def _validate_held_claims(plan: TensorCapturePlan, cpu_claim: Any,
                          gpu_claim: Any, *, device_lock_root: Path | None) -> None:
    if cpu_claim is None or gpu_claim is None:
        raise TensorCaptureRefusal("tensor capture requires held CPU and MI210 claims")
    try:
        cpu_receipt = cpu_claim.receipt().to_dict()
        gpu_receipt = gpu_claim.receipt().to_dict()
        cpu_check = cpu_claim.verify_held()
        gpu_check = device_claim.check_device_claim_held(
            gpu_receipt, lock_root=device_lock_root)
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        raise TensorCaptureRefusal(f"cannot validate resource claims: {exc}") from exc
    if cpu_check.outcome != schemas.PASS:
        raise TensorCaptureRefusal(
            "CPU resource claim is not held: " + "; ".join(cpu_check.reasons))
    if gpu_check.outcome != schemas.PASS:
        raise TensorCaptureRefusal(
            "MI210 resource claim is not held: " + "; ".join(gpu_check.reasons))
    if (cpu_receipt.get("cpu_list") != plan.cpu_list
            or cpu_receipt.get("campaign_id") != plan.campaign_id):
        raise TensorCaptureRefusal("CPU resource claim identity differs from the plan")
    if (gpu_receipt.get("device_id") != plan.device_claim_id
            or gpu_receipt.get("campaign_id") != plan.campaign_id):
        raise TensorCaptureRefusal("MI210 resource claim identity differs from the plan")


def execute_capture(
        plan: TensorCapturePlan, *, authorize_inference: bool = False,
        cpu_claim: Any = None, gpu_claim: Any = None,
        device_lock_root: Path | None = None,
        execution_evidence: dict[str, Any] | None = None,
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
    _validate_held_claims(plan, cpu_claim, gpu_claim,
                          device_lock_root=device_lock_root)
    # Revalidate mutable identities immediately before and after the producer.
    plan.source.validate()
    plan.model.validate()
    plan.toolchain.validate()
    producer_path = (Path(plan.source.repository_root).resolve()
                     / plan.source.producer_file).resolve()
    environment_source = os.environ if environ is None else environ
    if environment_source.get("HSA_OVERRIDE_GFX_VERSION"):
        raise TensorCaptureRefusal("HSA_OVERRIDE_GFX_VERSION would spoof physical gfx90a")
    for variable in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if variable in environment_source and environment_source[variable] != str(
                plan.device_visible_ordinal):
            raise TensorCaptureRefusal(f"{variable} differs from the capture plan")
    child_environment = {
        key: environment_source[key]
        for key in ("PATH",)
        if key in environment_source
    }
    child_environment.update(plan.runtime_environment)
    child_environment["PYTHONNOUSERSITE"] = "1"
    child_environment["HIP_VISIBLE_DEVICES"] = str(plan.device_visible_ordinal)
    child_environment["ROCR_VISIBLE_DEVICES"] = str(plan.device_visible_ordinal)
    if tuple(plan.capture_command) != (
            str(Path(plan.toolchain.python_executable).resolve()), str(producer_path)):
        raise TensorCaptureRefusal("capture command identity drifted from pinned producer")
    argv = (*plan.capture_command,
        "--epyc-c3-tensor-capture-v1", "--output-root", str(plan.output_root),
    )
    def poll_boundary() -> None:
        _validate_held_claims(plan, cpu_claim, gpu_claim,
                              device_lock_root=device_lock_root)
        if hasattr(gpu_claim, "revocation") and gpu_claim.revocation() is not None:
            raise TensorCaptureRefusal("MI210 claim was revoked during capture")
    try:
        result = run(
            argv, input=_canonical(plan.to_dict()) + "\n",
            cwd=str(Path(plan.source.repository_root).resolve()),
            env=child_environment, timeout=plan.timeout_seconds,
            poll_check=poll_boundary)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TensorCaptureRefusal(f"tensor producer process failed: {exc}") from exc
    if not isinstance(result, subprocess.CompletedProcess) or result.returncode != 0:
        code = getattr(result, "returncode", "invalid-result")
        raise TensorCaptureRefusal(f"tensor producer exited nonzero: {code}")
    witness = getattr(result, "residency_witness", None)
    if not isinstance(witness, Mapping) or witness.get("schema") != \
            "epyc.autokernel.kfd_process_group_witness.v1" \
            or witness.get("overlap_observed") is not True:
        raise TensorCaptureRefusal(
            "tensor producer lacks overlapping process-group KFD residency evidence")
    if execution_evidence is not None:
        execution_evidence["kfd_residency"] = dict(witness)
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
    _validate_held_claims(plan, cpu_claim, gpu_claim,
                          device_lock_root=device_lock_root)
    receipt = bind_capture_outputs(plan)
    receipt_path = Path(plan.output_root) / "tensor_capture_receipt.json"
    _publish_json_exclusive(receipt_path, receipt)
    return receipt


def finalize_capture_window(
        plan: TensorCapturePlan, *, tensor_receipt: Mapping[str, Any],
        open_cpu_claim: Mapping[str, Any], open_device_claim: Mapping[str, Any],
        released_cpu_claim: Mapping[str, Any], released_device_claim: Mapping[str, Any],
        device_sampling: Mapping[str, Any], kfd_residency: Mapping[str, Any],
        claim_journal: Path) -> dict[str, Any]:
    """Bind the complete claimed/sampled capture window after both releases."""
    if not released_cpu_claim.get("released_at") or not released_device_claim.get(
            "released_at"):
        raise TensorCaptureRefusal("capture window requires released CPU and device claims")
    for before, after, label in ((open_cpu_claim, released_cpu_claim, "CPU"),
                                 (open_device_claim, released_device_claim, "device")):
        if before.get("claim_id") != after.get("claim_id"):
            raise TensorCaptureRefusal(f"{label} claim changed across capture window")
    if kfd_residency.get("overlap_observed") is not True:
        raise TensorCaptureRefusal("capture window lacks overlapping KFD residency")
    if device_sampling.get("schema") != "epyc.autokernel.device_sampling_receipt.v1" \
            or not device_sampling.get("samples"):
        raise TensorCaptureRefusal("capture window lacks overlapping numeric device samples")
    journal_path = _checked_regular_file(
        claim_journal, _sha256_file(claim_journal), "claim journal")
    tensor_path = _checked_regular_file(
        plan.output_root / "tensor_capture_receipt.json",
        _sha256_file(plan.output_root / "tensor_capture_receipt.json"),
        "tensor capture receipt")
    if load_capture_receipt(tensor_path) != tensor_receipt:
        raise TensorCaptureRefusal("capture window tensor receipt drifted")
    document = {
        "schema": WINDOW_SCHEMA, "authority": AUTHORITY,
        "plan_sha256": plan.plan_sha256,
        "tensor_capture_receipt": str(tensor_path),
        "tensor_capture_receipt_sha256": _sha256_file(tensor_path),
        "open_cpu_claim": dict(open_cpu_claim),
        "open_device_claim": dict(open_device_claim),
        "released_cpu_claim": dict(released_cpu_claim),
        "released_device_claim": dict(released_device_claim),
        "device_sampling": dict(device_sampling),
        "kfd_residency": dict(kfd_residency),
        "claim_journal": str(journal_path),
        "claim_journal_sha256": _sha256_file(journal_path),
        "device_inventory": str(plan.device_inventory),
        "device_inventory_sha256": plan.device_inventory_sha256,
        "runtime_environment": dict(plan.runtime_environment),
    }
    document["receipt_sha256"] = hashlib.sha256(_canonical(document).encode()).hexdigest()
    return document


def load_capture_window_receipt(path: Path) -> Mapping[str, Any]:
    document = _read_json(Path(path), "tensor capture window receipt")
    required = {"schema", "authority", "plan_sha256", "tensor_capture_receipt",
                "tensor_capture_receipt_sha256", "open_cpu_claim", "open_device_claim",
                "released_cpu_claim", "released_device_claim", "device_sampling",
                "kfd_residency", "claim_journal", "claim_journal_sha256",
                "device_inventory", "device_inventory_sha256", "runtime_environment",
                "receipt_sha256"}
    _exact_keys(document, required, "tensor capture window receipt")
    if document["schema"] != WINDOW_SCHEMA or document["authority"] != AUTHORITY:
        raise TensorCaptureRefusal("capture window schema or authority drifted")
    material = dict(document)
    claimed = _sha(material.pop("receipt_sha256"), "receipt_sha256")
    if hashlib.sha256(_canonical(material).encode()).hexdigest() != claimed:
        raise TensorCaptureRefusal("capture window self-hash mismatch")
    for field in ("tensor_capture_receipt", "claim_journal", "device_inventory"):
        _checked_regular_file(Path(document[field]), document[f"{field}_sha256"], field)
    if not _mapping(document["released_cpu_claim"], "released_cpu_claim").get("released_at") \
            or not _mapping(document["released_device_claim"],
                            "released_device_claim").get("released_at"):
        raise TensorCaptureRefusal("capture window claims are not released")
    try:
        open_cpu = cpu_region_claim.RegionClaimReceipt.from_dict(
            _mapping(document["open_cpu_claim"], "open_cpu_claim"))
        released_cpu = cpu_region_claim.RegionClaimReceipt.from_dict(
            _mapping(document["released_cpu_claim"], "released_cpu_claim"))
        open_gpu = device_claim.ClaimReceipt.from_dict(
            _mapping(document["open_device_claim"], "open_device_claim"))
        released_gpu = device_claim.ClaimReceipt.from_dict(
            _mapping(document["released_device_claim"], "released_device_claim"))
    except (TypeError, ValueError) as exc:
        raise TensorCaptureRefusal(f"capture window claim receipt invalid: {exc}") from exc
    if (open_cpu.claim_id != released_cpu.claim_id
            or open_gpu.claim_id != released_gpu.claim_id
            or open_cpu.released_at is not None or open_gpu.released_at is not None):
        raise TensorCaptureRefusal("capture window open/released claim identities differ")
    kfd = _mapping(document["kfd_residency"], "kfd_residency")
    _exact_keys(kfd, {"schema", "process_group_id", "samples", "overlap_observed"},
                "kfd_residency")
    if kfd["schema"] != "epyc.autokernel.kfd_process_group_witness.v1" \
            or kfd["overlap_observed"] is not True or not isinstance(kfd["samples"], list) \
            or not kfd["samples"]:
        raise TensorCaptureRefusal("capture window lacks KFD overlap")
    if not any(isinstance(row, Mapping) and row.get("kfd_pids")
               and isinstance(row.get("offset_s"), (int, float))
               and math.isfinite(float(row["offset_s"])) for row in kfd["samples"]):
        raise TensorCaptureRefusal("capture window KFD samples do not witness residency")
    sampling = _mapping(document["device_sampling"], "device_sampling")
    sampling_required = {"schema", "sampler_id", "device_id", "source", "started_at",
                         "ended_at", "interval_s", "duration_s", "command",
                         "sample_count", "max_gap_s", "samples", "sha256"}
    _exact_keys(sampling, sampling_required, "device_sampling")
    sampling_material = dict(sampling)
    sampling_sha = _sha(sampling_material.pop("sha256"), "device_sampling.sha256")
    if hashlib.sha256(_canonical(sampling_material).encode()).hexdigest() != sampling_sha \
            or sampling["schema"] != "epyc.autokernel.device_sampling_receipt.v1" \
            or sampling["sample_count"] != len(sampling["samples"]) \
            or not sampling["samples"]:
        raise TensorCaptureRefusal("capture window device sampling receipt is invalid")
    for row in sampling["samples"]:
        row = _mapping(row, "device sample")
        for field in ("offset_s", "sclk_mhz", "mclk_mhz", "power_w", "temperature_c"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or value < 0:
                raise TensorCaptureRefusal(f"device sample {field} is not finite/non-negative")
    tensor = load_capture_receipt(Path(document["tensor_capture_receipt"]))
    if tensor["plan_sha256"] != document["plan_sha256"]:
        raise TensorCaptureRefusal("capture window plan differs from tensor receipt")
    return document


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser(
        "compile", help="validate a request manifest and emit a no-inference plan")
    compile_parser.add_argument("--manifest", type=Path, required=True)
    compile_parser.add_argument("--plan", type=Path, required=True)
    execute_parser = subparsers.add_parser(
        "execute", help="acquire governed claims and execute a compiled plan")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--claim-journal", type=Path, required=True)
    execute_parser.add_argument("--claim-timeout-seconds", type=float, default=600.0)
    execute_parser.add_argument("--execute", action="store_true")
    execute_parser.add_argument("--authorize-inference", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compile":
            plan = compile_capture_manifest(args.manifest)
            _publish_json_exclusive(args.plan, plan.to_dict())
            print(json.dumps({"plan": str(args.plan.resolve()),
                              "plan_sha256": plan.plan_sha256}, sort_keys=True))
            return 0
        plan = load_capture_plan(args.plan)
        if not args.execute or not args.authorize_inference:
            raise TensorCaptureRefusal(
                "execution requires both --execute and --authorize-inference")
        journal_path = _reject_governed_path(args.claim_journal, "claim journal")
        if not journal_path.is_absolute() or journal_path.is_relative_to(
                Path("/mnt/raid0/llm/tmp")):
            raise TensorCaptureRefusal("claim journal must be an absolute durable path")
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal = cpu_region_claim.RegionClaimJournal(journal_path)
        claim_args = {
            "purpose": f"AutoKernel C3/C5 tensor capture {plan.case_id}",
            "campaign_id": plan.campaign_id, "journal": journal,
            "timeout_s": args.claim_timeout_seconds,
            "max_hold_s": float(plan.timeout_seconds + 300),
        }
        cpu_held = None
        gpu_held = None
        sampling_session = None
        sampling_receipt = None
        evidence: dict[str, Any] = {}
        open_cpu: Mapping[str, Any] | None = None
        open_gpu: Mapping[str, Any] | None = None
        released_cpu: Mapping[str, Any] | None = None
        released_gpu: Mapping[str, Any] | None = None
        try:
            cpu_held = cpu_region_claim.acquire_cpu_region_claim(
                plan.cpu_list, role="autokernel", **claim_args)
            gpu_held = device_claim.acquire_device_claim(plan.device_claim_id, **claim_args)
            open_cpu = cpu_held.receipt().to_dict()
            open_gpu = gpu_held.receipt().to_dict()
            if gpu_held.revocation() is not None:
                raise TensorCaptureRefusal("MI210 claim was revoked before capture")
            sampling_session = device_sampler.RocmSmiSampler(
                device_index=plan.device_visible_ordinal, interval_s=0.25).start()
            receipt = execute_capture(
                plan, authorize_inference=True, cpu_claim=cpu_held,
                gpu_claim=gpu_held, execution_evidence=evidence)
            sampling_receipt = sampling_session.stop().to_dict()
            sampling_session = None
            if gpu_held.revocation() is not None:
                raise TensorCaptureRefusal("MI210 claim was revoked during capture")
        finally:
            if sampling_session is not None:
                try:
                    sampling_session.stop()
                except Exception:
                    pass
            try:
                if gpu_held is not None:
                    released_gpu = gpu_held.release().to_dict()
            finally:
                if cpu_held is not None:
                    released_cpu = cpu_held.release().to_dict()
        if any(value is None for value in (open_cpu, open_gpu, released_cpu,
                                            released_gpu, sampling_receipt)):
            raise TensorCaptureRefusal("capture window did not complete all evidence boundaries")
        window = finalize_capture_window(
            plan, tensor_receipt=receipt, open_cpu_claim=open_cpu,
            open_device_claim=open_gpu, released_cpu_claim=released_cpu,
            released_device_claim=released_gpu, device_sampling=sampling_receipt,
            kfd_residency=evidence.get("kfd_residency", {}),
            claim_journal=journal_path)
        window_path = plan.output_root / "tensor_capture_window_receipt.json"
        _publish_json_exclusive(window_path, window)
        print(json.dumps({"receipt": str(window_path.resolve()),
                          "receipt_sha256": window["receipt_sha256"]}, sort_keys=True))
        return 0
    except TensorCaptureRefusal as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=os.sys.stderr)
        return 2


__all__ = [
    "AUTHORITY", "CAPTURE_KIND", "COMPLETION_SCHEMA", "MANIFEST_SCHEMA", "PLAN_SCHEMA",
    "RECEIPT_SCHEMA", "REQUEST_SCHEMA", "WINDOW_SCHEMA",
    "CaptureModelIdentity", "CaptureSourceIdentity", "CaptureToolchainIdentity",
    "TensorCapturePlan", "TensorCaptureRefusal", "TensorSpec", "bind_capture_outputs",
    "compile_capture_manifest", "execute_capture", "finalize_capture_window",
    "load_capture_plan", "load_capture_receipt", "load_capture_window_receipt",
    "main", "prepare_capture_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
