"""Fail-closed verification for an AutoKernel split GPU reward runtime.

The reward executable and every non-HIP local DSO live in one immutable
``common`` directory.  Each arm directory contains only the three-link HIP
SONAME chain.  The verifier also checks a captured ``/proc/<pid>/maps`` image;
filesystem layout alone cannot prove which object the dynamic loader mapped.

This module does not launch a model, profiler, or actor.  The default ELF
reader invokes only ``/usr/bin/readelf`` and is injectable for hardware-free
tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Callable, Mapping


SCHEMA = "epyc.autokernel.split_reward_runtime.v1"
MAPS_SCHEMA = "epyc.autokernel.split_reward_runtime_maps.v1"
RESIDENCY_SCHEMA = "epyc.autokernel.hot_gpu_residency.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_PREFIXES = ("libggml", "libllama")
_COMMON_FAMILIES = (
    "libllama-common.so",
    "libllama.so",
    "libggml.so",
    "libggml-base.so",
    "libggml-cpu.so",
)
_COMMON_SINGLETONS = frozenset({"llama-bench", "libllama-bench-impl.so"})
_HIP_LINK = "libggml-hip.so"
_HIP_SONAME = "libggml-hip.so.0"


class SplitRuntimeError(RuntimeError):
    pass


class RuntimeMapsIncomplete(SplitRuntimeError):
    """The process is valid but has not mapped the full runtime closure yet."""


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SplitRuntimeError(f"expected regular runtime file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ElfIdentity:
    soname: str | None
    needed: tuple[str, ...]
    runpath: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"soname": self.soname, "needed": list(self.needed),
                "runpath": list(self.runpath)}


ElfReader = Callable[[Path], ElfIdentity]


def readelf_identity(path: Path) -> ElfIdentity:
    """Read dynamic identity without executing the object under review."""
    completed = subprocess.run(
        ("/usr/bin/readelf", "-dW", str(path)), check=False,
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"})
    if completed.returncode:
        raise SplitRuntimeError(
            f"readelf refused {path}: {completed.stderr[-400:]}")
    needed: list[str] = []
    runpath: list[str] = []
    soname: str | None = None
    for line in completed.stdout.splitlines():
        match = re.search(r"\((NEEDED|SONAME|RUNPATH|RPATH)\).*\[([^]]*)\]", line)
        if not match:
            continue
        kind, value = match.groups()
        if kind == "NEEDED":
            needed.append(value)
        elif kind == "SONAME":
            soname = value
        else:
            runpath.extend(item for item in value.split(":") if item)
    if not needed:
        raise SplitRuntimeError(f"ELF dynamic section is vacuous: {path}")
    return ElfIdentity(soname=soname, needed=tuple(needed),
                       runpath=tuple(runpath))


@dataclass(frozen=True)
class RuntimeFile:
    name: str
    kind: str
    sha256: str | None = None
    target: str | None = None
    elf: ElfIdentity | None = None

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {"name": self.name, "kind": self.kind}
        if self.sha256 is not None:
            body["sha256"] = self.sha256
        if self.target is not None:
            body["target"] = self.target
        if self.elf is not None:
            body["elf"] = self.elf.to_dict()
        return body


@dataclass(frozen=True)
class SplitRuntimeManifest:
    root: Path
    common_dir: Path
    anchor_hip_dir: Path
    candidate_hip_dir: Path
    reward_binary: Path
    common_files: tuple[RuntimeFile, ...]
    anchor_hip_files: tuple[RuntimeFile, ...]
    candidate_hip_files: tuple[RuntimeFile, ...]
    anchor_hip_sha256: str
    candidate_hip_sha256: str
    manifest_sha256: str

    def arm_environment(self, arm: str) -> Mapping[str, str]:
        hip = {"anchor": self.anchor_hip_dir,
               "candidate": self.candidate_hip_dir}.get(arm)
        if hip is None:
            raise SplitRuntimeError(f"unknown runtime arm: {arm}")
        # No ambient loader variables survive.  Helpers are resolved only from
        # the system-owned directories in this fixed PATH.
        return MappingProxyType({
            "PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH": f"{hip}:{self.common_dir}:/opt/rocm/lib",
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "root": str(self.root),
            "common_dir": str(self.common_dir),
            "anchor_hip_dir": str(self.anchor_hip_dir),
            "candidate_hip_dir": str(self.candidate_hip_dir),
            "reward_binary": str(self.reward_binary),
            "common_files": [item.to_dict() for item in self.common_files],
            "anchor_hip_files": [item.to_dict() for item in self.anchor_hip_files],
            "candidate_hip_files": [item.to_dict() for item in self.candidate_hip_files],
            "anchor_hip_sha256": self.anchor_hip_sha256,
            "candidate_hip_sha256": self.candidate_hip_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


def _relative_link(directory: Path, path: Path, allowed: frozenset[str]) -> RuntimeFile:
    target = os.readlink(path)
    target_path = Path(target)
    if target_path.is_absolute() or len(target_path.parts) != 1 or target not in allowed:
        raise SplitRuntimeError(f"runtime symlink escapes exact closure: {path} -> {target}")
    resolved = path.resolve(strict=True)
    if resolved.parent != directory or resolved.is_symlink():
        raise SplitRuntimeError(f"runtime symlink does not resolve locally: {path}")
    return RuntimeFile(name=path.name, kind="symlink", target=target)


def _family_names(entries: frozenset[str], stem: str) -> tuple[str, str, str]:
    soname = f"{stem}.0"
    versions = sorted(name for name in entries
                      if name.startswith(soname + ".") and name != soname)
    if len(versions) != 1 or stem not in entries or soname not in entries:
        raise SplitRuntimeError(f"{stem} requires .so -> .so.0 -> one versioned file")
    return stem, soname, versions[0]


def _inspect_directory(directory: Path, expected: frozenset[str], *,
                       elf_reader: ElfReader) -> tuple[RuntimeFile, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise SplitRuntimeError(f"runtime directory is unsafe: {directory}")
    actual = frozenset(path.name for path in directory.iterdir())
    if actual != expected:
        raise SplitRuntimeError(
            f"runtime directory membership differs: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}")
    records: list[RuntimeFile] = []
    for name in sorted(actual):
        path = directory / name
        if path.is_symlink():
            records.append(_relative_link(directory, path, actual))
        elif path.is_file():
            records.append(RuntimeFile(name=name, kind="file", sha256=_sha(path),
                                       elf=elf_reader(path)))
        else:
            raise SplitRuntimeError(f"non-file runtime member: {path}")
    return tuple(records)


def _require_chain(records: tuple[RuntimeFile, ...], stem: str) -> RuntimeFile:
    by_name = {row.name: row for row in records}
    soname = f"{stem}.0"
    versions = [row for row in records if row.name.startswith(soname + ".")]
    if (len(versions) != 1 or by_name[stem].kind != "symlink"
            or by_name[stem].target != soname
            or by_name[soname].kind != "symlink"
            or by_name[soname].target != versions[0].name
            or versions[0].kind != "file" or versions[0].elf is None
            or versions[0].elf.soname != soname):
        raise SplitRuntimeError(
            f"{stem} topology/ELF identity is not .so -> .so.0 -> one matching SONAME DSO")
    return versions[0]


def verify_split_runtime(root: Path, *, elf_reader: ElfReader = readelf_identity) -> SplitRuntimeManifest:
    """Verify exact on-disk split closure and return its content identity."""
    root = root.resolve(strict=True)
    common = root / "common"
    anchor = root / "anchor-hip"
    candidate = root / "candidate-hip"
    common_entries = frozenset(path.name for path in common.iterdir())
    expected_common = set(_COMMON_SINGLETONS)
    for family in _COMMON_FAMILIES:
        expected_common.update(_family_names(common_entries, family))
    common_records = _inspect_directory(common, frozenset(expected_common),
                                        elf_reader=elf_reader)
    hip_names: dict[str, frozenset[str]] = {}
    for label, directory in (("anchor", anchor), ("candidate", candidate)):
        entries = frozenset(path.name for path in directory.iterdir())
        hip_names[label] = frozenset(_family_names(entries, _HIP_LINK))
    anchor_records = _inspect_directory(anchor, hip_names["anchor"],
                                        elf_reader=elf_reader)
    candidate_records = _inspect_directory(candidate, hip_names["candidate"],
                                           elf_reader=elf_reader)

    for family in _COMMON_FAMILIES:
        _require_chain(common_records, family)
    anchor_hip_object = _require_chain(anchor_records, _HIP_LINK)
    candidate_hip_object = _require_chain(candidate_records, _HIP_LINK)

    regular_common = {row.name: row for row in common_records if row.kind == "file"}
    regular_anchor = {row.name: row for row in anchor_records if row.kind == "file"}
    regular_candidate = {row.name: row for row in candidate_records if row.kind == "file"}
    reward = regular_common.get("llama-bench")
    bench_impl = regular_common.get("libllama-bench-impl.so")
    if (reward is None or reward.elf is None or reward.elf.soname is not None
            or bench_impl is None or bench_impl.elf is None
            or bench_impl.elf.soname != "libllama-bench-impl.so"
            or "libllama-bench-impl.so" not in reward.elf.needed):
        raise SplitRuntimeError("shared reward executable has invalid ELF identity")

    # Every llama/ggml dependency must resolve inside common, except the one
    # intended HIP SONAME.  RUNPATH is recorded and may contain only absolute
    # paths or $ORIGIN expressions; relative cwd-dependent lookup is refused.
    provided = set(expected_common) | {_HIP_SONAME}
    for row in (*regular_common.values(), *regular_anchor.values(), *regular_candidate.values()):
        assert row.elf is not None
        missing = [name for name in row.elf.needed
                   if name.startswith(_LOCAL_PREFIXES) and name not in provided]
        if missing:
            raise SplitRuntimeError(f"unclosed local ELF dependencies for {row.name}: {missing}")
        # An embedded build-directory RUNPATH turns a missing sealed object into
        # a silent fallback to an unsealed tree.  The copied closure may consult
        # itself and the system ROCm tree only.
        if any(not (entry == "$ORIGIN" or entry.startswith("$ORIGIN/")
                   or entry == "/opt/rocm/lib"
                   or entry.startswith("/opt/rocm/lib/"))
               for entry in row.elf.runpath):
            raise SplitRuntimeError(f"unsealed ELF RUNPATH in {row.name}")
    for records in (regular_anchor, regular_candidate):
        if len(records) != 1:
            raise SplitRuntimeError("HIP arm contains more than one regular DSO")
        only = next(iter(records.values()))
        if only.elf is None or only.elf.soname != _HIP_SONAME:
            raise SplitRuntimeError("HIP versioned DSO has wrong SONAME")

    if anchor_hip_object.sha256 is None or candidate_hip_object.sha256 is None:
        raise SplitRuntimeError("HIP chain lacks hashed regular objects")

    anchor_real = (anchor / _HIP_SONAME).resolve(strict=True)
    candidate_real = (candidate / _HIP_SONAME).resolve(strict=True)
    anchor_sha, candidate_sha = _sha(anchor_real), _sha(candidate_real)
    if anchor_sha == candidate_sha:
        raise SplitRuntimeError("HIP arm DSOs are byte-identical")
    body = {
        "schema": SCHEMA,
        "root": str(root),
        "common": [item.to_dict() for item in common_records],
        "anchor_hip": [item.to_dict() for item in anchor_records],
        "candidate_hip": [item.to_dict() for item in candidate_records],
        "anchor_hip_sha256": anchor_sha,
        "candidate_hip_sha256": candidate_sha,
    }
    return SplitRuntimeManifest(
        root=root, common_dir=common, anchor_hip_dir=anchor,
        candidate_hip_dir=candidate, reward_binary=common / "llama-bench",
        common_files=common_records, anchor_hip_files=anchor_records,
        candidate_hip_files=candidate_records, anchor_hip_sha256=anchor_sha,
        candidate_hip_sha256=candidate_sha, manifest_sha256=_content_hash(body))


def _maps_paths(text: str) -> frozenset[Path]:
    paths: set[Path] = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5].removesuffix(" (deleted)")
        paths.add(Path(raw).resolve(strict=True))
    return frozenset(paths)


@dataclass(frozen=True)
class HotResidencyIdentity:
    runtime_manifest_sha256: str
    arm: str
    reward_binary_sha256: str
    hip_library_sha256: str
    model_path: Path
    model_sha256: str
    device_id: str
    kfd_pid: int
    boot_id: str
    process_start_ticks: int
    mapped_local_sha256: Mapping[str, str]
    identity_sha256: str

    def __post_init__(self) -> None:
        if (self.arm not in {"anchor", "candidate"}
                or not all(SHA256_RE.fullmatch(value) for value in (
                    self.runtime_manifest_sha256, self.reward_binary_sha256,
                    self.hip_library_sha256, self.model_sha256,
                    self.identity_sha256))
                or not self.model_path.is_absolute() or not self.model_path.is_file()
                or not self.device_id or self.kfd_pid <= 0 or not self.boot_id
                or self.process_start_ticks <= 0
                or not self.mapped_local_sha256
                or any(not Path(path).is_absolute() or not SHA256_RE.fullmatch(digest)
                       for path, digest in self.mapped_local_sha256.items())):
            raise SplitRuntimeError("hot residency identity is malformed")
        frozen = MappingProxyType(dict(self.mapped_local_sha256))
        object.__setattr__(self, "mapped_local_sha256", frozen)
        body = {"schema": MAPS_SCHEMA,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
                "arm": self.arm,
                "reward_binary_sha256": self.reward_binary_sha256,
                "hip_library_sha256": self.hip_library_sha256,
                "model_path": str(self.model_path), "model_sha256": self.model_sha256,
                "device_id": self.device_id, "kfd_pid": self.kfd_pid,
                "boot_id": self.boot_id, "process_start_ticks": self.process_start_ticks,
                "mapped_local_sha256": dict(frozen)}
        if _content_hash(body) != self.identity_sha256:
            raise SplitRuntimeError("hot residency identity self-hash mismatch")

    def same_resident_process(self, other: "HotResidencyIdentity") -> bool:
        return all(getattr(self, field) == getattr(other, field) for field in (
            "runtime_manifest_sha256", "arm", "reward_binary_sha256",
            "hip_library_sha256", "model_sha256", "device_id", "kfd_pid",
            "boot_id", "process_start_ticks", "identity_sha256"))

    def to_dict(self) -> dict[str, object]:
        return {"schema": RESIDENCY_SCHEMA,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
                "arm": self.arm,
                "reward_binary_sha256": self.reward_binary_sha256,
                "hip_library_sha256": self.hip_library_sha256,
                "model_path": str(self.model_path), "model_sha256": self.model_sha256,
                "device_id": self.device_id, "kfd_pid": self.kfd_pid,
                "boot_id": self.boot_id, "process_start_ticks": self.process_start_ticks,
                "mapped_local_sha256": dict(self.mapped_local_sha256),
                "identity_sha256": self.identity_sha256}


def verify_runtime_maps(manifest: SplitRuntimeManifest, *, arm: str, maps_text: str,
                        model_path: Path, model_sha256: str, device_id: str,
                        kfd_pid: int, boot_id: str,
                        process_start_ticks: int) -> HotResidencyIdentity:
    """Bind actual mapped objects and a process-lifetime marker to one arm."""
    if arm not in {"anchor", "candidate"}:
        raise SplitRuntimeError("runtime maps arm is invalid")
    if not SHA256_RE.fullmatch(model_sha256) or kfd_pid <= 0 \
            or process_start_ticks <= 0 or not boot_id or not device_id:
        raise SplitRuntimeError("runtime residency identity is incomplete")
    paths = _maps_paths(maps_text)
    hip_dir = manifest.anchor_hip_dir if arm == "anchor" else manifest.candidate_hip_dir
    expected_hip = (hip_dir / _HIP_SONAME).resolve(strict=True)
    expected_model = model_path.resolve(strict=True)
    allowed_roots = (manifest.common_dir, manifest.anchor_hip_dir,
                     manifest.candidate_hip_dir)
    local = {path for path in paths if any(path == root or root in path.parents
                                           for root in allowed_roots)}
    wrong_arm = manifest.candidate_hip_dir if arm == "anchor" else manifest.anchor_hip_dir
    if any(path == wrong_arm or wrong_arm in path.parents for path in local):
        raise SplitRuntimeError("runtime maps contain the opposite HIP arm")
    expected_common = {path.resolve(strict=True) for path in manifest.common_dir.iterdir()
                       if path.is_file()}
    allowed_local = expected_common | {expected_hip}
    if local - allowed_local:
        raise SplitRuntimeError("runtime maps contain an unsealed local object")
    if expected_model in paths and _sha(expected_model) != model_sha256:
        raise SplitRuntimeError("runtime maps model bytes changed after verification")
    if manifest.reward_binary.resolve(strict=True) not in paths:
        raise RuntimeMapsIncomplete("runtime maps omit shared reward executable")
    if expected_hip not in paths:
        raise RuntimeMapsIncomplete("runtime maps omit intended HIP SONAME object")
    if expected_model not in paths:
        raise RuntimeMapsIncomplete("runtime maps omit the sealed resident model")
    if not expected_common.issubset(local):
        missing = sorted(str(path) for path in expected_common - local)
        raise RuntimeMapsIncomplete(f"runtime maps omit common closure objects: {missing}")
    mapped = {str(path): _sha(path) for path in sorted(local)}
    expected_common_hashes: dict[str, str] = {}
    for record in manifest.common_files:
        if record.kind != "file" or record.sha256 is None:
            continue
        resolved = (manifest.common_dir / record.name).resolve(strict=True)
        expected_common_hashes[str(resolved)] = record.sha256
    actual_common_hashes = {path: digest for path, digest in mapped.items()
                            if Path(path) != expected_hip}
    if actual_common_hashes != expected_common_hashes:
        raise SplitRuntimeError("mapped common closure bytes changed after verification")
    reward_sha = _sha(manifest.reward_binary)
    hip_sha = _sha(expected_hip)
    expected_hip_sha = (manifest.anchor_hip_sha256 if arm == "anchor"
                        else manifest.candidate_hip_sha256)
    if hip_sha != expected_hip_sha:
        raise SplitRuntimeError("mapped HIP DSO changed after verification")
    body = {"schema": MAPS_SCHEMA,
            "runtime_manifest_sha256": manifest.manifest_sha256, "arm": arm,
            "reward_binary_sha256": reward_sha, "hip_library_sha256": hip_sha,
            "model_path": str(expected_model), "model_sha256": model_sha256,
            "device_id": device_id, "kfd_pid": kfd_pid, "boot_id": boot_id,
            "process_start_ticks": process_start_ticks,
            "mapped_local_sha256": mapped}
    return HotResidencyIdentity(
        runtime_manifest_sha256=manifest.manifest_sha256, arm=arm,
        reward_binary_sha256=reward_sha, hip_library_sha256=hip_sha,
        model_path=expected_model, model_sha256=model_sha256,
        device_id=device_id, kfd_pid=kfd_pid, boot_id=boot_id,
        process_start_ticks=process_start_ticks,
        mapped_local_sha256=MappingProxyType(mapped),
        identity_sha256=_content_hash(body))


def validate_arm_pair(anchor: HotResidencyIdentity,
                      candidate: HotResidencyIdentity) -> None:
    """Prove mapped local closure equality except the intended HIP object."""
    if anchor.arm != "anchor" or candidate.arm != "candidate":
        raise SplitRuntimeError("runtime identity pair has wrong arms")
    if anchor.runtime_manifest_sha256 != candidate.runtime_manifest_sha256:
        raise SplitRuntimeError("runtime arms use different closure manifests")
    if anchor.reward_binary_sha256 != candidate.reward_binary_sha256:
        raise SplitRuntimeError("runtime arms use different reward executables")
    anchor_common = {Path(path).name: digest for path, digest in anchor.mapped_local_sha256.items()
                     if not Path(path).name.startswith(_HIP_LINK)}
    candidate_common = {Path(path).name: digest for path, digest in candidate.mapped_local_sha256.items()
                        if not Path(path).name.startswith(_HIP_LINK)}
    if anchor_common != candidate_common:
        raise SplitRuntimeError("mapped non-HIP runtime differs between arms")
    if anchor.hip_library_sha256 == candidate.hip_library_sha256:
        raise SplitRuntimeError("mapped HIP runtime does not differ between arms")
