#!/usr/bin/env python3
"""Production-only P-GPU-1 Qwen3.6-27B Q8 DFlash base-vs-spec runner.

Dry-run is the default.  ``--execute`` is intentionally guarded: it requires a
clean production-consolidated-v9 HIP tree, then runs five fresh-server
replicates *per arm*.  Every replicate executes the same immutable prompt pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server")
DEFAULT_SOURCE_ROOT = Path("/mnt/raid0/llm/llama.cpp")
DEFAULT_TARGET_MODEL = Path("/mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf")
DEFAULT_DRAFTER_MODEL = Path("/mnt/raid0/llm/models/dflash/Qwen3.6-27B-DFlash/Qwen3.6-27B-DFlash-f16-canonical-v3.gguf")
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data/gpu-mi210/qwen36-27b-q8-dflash-pgpu1-v9"
DEFAULT_REPS = 5
DEFAULT_PORT_BASE = 19880
DEFAULT_CONTEXT = 4096
DEFAULT_MAX_TOKENS = 512
DEFAULT_MIN_COMPLETION_TOKENS = 96
MIN_EXPLANATION_WORDS = 8
DEFAULT_SEED = 424242
DEFAULT_STARTUP_TIMEOUT_S = 600
DEFAULT_REQUEST_TIMEOUT_S = 900
TARGET_CACHE_K = "q8_0"
TARGET_CACHE_V = "q8_0"
DRAFTER_CACHE_K = "q8_0"
DRAFTER_CACHE_V = "q8_0"
PROMPT_SPECS = (
    ("primes", "Write a concise explanation of how to enumerate every prime below 30 and sum them. Do not enumerate rejected or composite candidates one by one. Then end with exactly `RESULT_JSON: {\"primes\":[...],\"sum\":...}` using the computed values. The final JSON is machine-validated."),
    ("nested_flatten", "Write a concise explanation of a deterministic encounter-order flatten of {\"a\":[1,{\"b\":[2,3]}],\"c\":{\"d\":4},\"e\":[5]}. Then end with exactly `RESULT_JSON: {\"values\":[...]}` using the computed scalar order. The final JSON is machine-validated."),
    ("normalize", "Write a concise explanation of normalization of [0,2,3,5] by its sum, including the zero-sum edge case. The input total is 10. Then end with exactly `RESULT_JSON: {\"normalized\":[...],\"sum\":...}` using the computed normalized values. JSON `sum` must be the sum of the normalized values. Do not report the input total in JSON. The final JSON is machine-validated."),
)
PROMPTS = tuple(text for _, text in PROMPT_SPECS)
PGPU1_WARMUP_POLICY = "no warm-up requests; no discarded reps; fresh server per rep; graph recapture remains inside each measured fresh-server replicate"
CPU_INTERFERENCE_POLICY = "CPU production stack quiesced and verified before the window; no concurrent llama-server, AutoPilot, or KFD model-owner process is permitted"
EXPECTED_BRANCH = "production-consolidated-v9"
ROLLBACK_BRANCH = "production-consolidated-v8"
PROMOTION_ATTESTATION_SCHEMA = "epyc.kernel_promotion_attestation.v1"
CANDIDATE_SMOKE_SCHEMA = "epyc.qwen36_27b_q8_dflash_candidate_smoke.v1"
SAFE_PATH = "/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
BASE_ENVIRONMENT = {"PATH": SAFE_PATH, "LANG": "C", "LC_ALL": "C"}
PINNED_VISIBLE_DEVICE_ENVIRONMENT = {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"}
SCRUBBED_ENV_PREFIXES = ("GGML_", "HSA_", "HIP_", "ROCR_")
SCRUBBED_ENV_NAMES = ("LD_PRELOAD",)
GOVERNANCE_REPO = Path("/mnt/raid0/llm/epyc-root")
PROMOTION_ATTESTATION_RELATIVE_PATH = Path("handoffs/active/v9-kernel-promotion-attestation.json")
PROMOTION_ATTESTATION_PATH = GOVERNANCE_REPO / PROMOTION_ATTESTATION_RELATIVE_PATH
TARGET_MODEL_BYTES = 29_047_084_160
TARGET_MODEL_SHA256 = "9408dcb356cc061a05c139e5647cbde0698ff980c6a69f7fc214e9989f86cfa8"
DRAFTER_MODEL_BYTES = 3_471_497_600
DRAFTER_MODEL_SHA256 = "27cfa437c226ade7fdac8009206e163566a1418674789322ecf07c6f1a3def17"
TARGET_OFFLOAD_LAYER_COUNT = 66
DRAFTER_OFFLOAD_LAYER_COUNT = 6
TARGET_POSITIVE_KV_BUFFER_COUNT = 1
DRAFTER_POSITIVE_KV_BUFFER_COUNT = 1
SOURCE_UNTRACKED_ALLOWLIST = {
    ".gitnexusignore": "local GitNexus configuration; not a llama.cpp build input",
    "tools/math-tools/": "operator-owned unrelated tool subtree; not linked into llama-server",
}


@dataclass(frozen=True)
class Arm:
    name: str
    speculative: bool


BASE_ARM = Arm("base", False)
DFLASH_ARM = Arm("dflash", True)
ARMS = (BASE_ARM, DFLASH_ARM)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_identity(value: os.stat_result) -> dict[str, int]:
    return {"dev": value.st_dev, "inode": value.st_ino, "bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def stable_file_identity(path: Path) -> dict[str, Any]:
    """Hash a regular non-symlink file only when its pathname stays stable."""
    try:
        resolved = path.resolve(strict=True)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("path is not a regular file")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or stat_identity(before) != stat_identity(opened):
                raise ValueError("path changed while opening")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
            after_fd = os.fstat(handle.fileno())
        after_path = path.stat()
        if stat_identity(before) != stat_identity(after_fd) or stat_identity(before) != stat_identity(after_path):
            raise ValueError("path changed while hashing")
        return {"path": str(path), "resolved_path": str(resolved), **stat_identity(before), "sha256": digest.hexdigest(), "stable": True}
    except (OSError, ValueError) as exc:
        return {"path": str(path), "resolved_path": str(path.resolve()), "stable": False, "identity_error": str(exc)}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence_env() -> dict[str, str]:
    return {**BASE_ENVIRONMENT, **PINNED_VISIBLE_DEVICE_ENVIRONMENT}


def scrubbed_parent_env_keys() -> list[str]:
    return sorted(
        key
        for key in os.environ
        if key in SCRUBBED_ENV_NAMES or any(key.startswith(prefix) for prefix in SCRUBBED_ENV_PREFIXES)
    )


def subprocess_env_is_allowed(env: dict[str, str]) -> bool:
    if any(not Path(entry).is_absolute() for entry in str(env.get("PATH") or "").split(":")):
        return False
    if any(env.get(key) != value for key, value in evidence_env().items()):
        return False
    if set(env) - {*evidence_env(), "LD_LIBRARY_PATH"}:
        return False
    library_path = env.get("LD_LIBRARY_PATH")
    expected_library_path = f"{DEFAULT_BINARY.parent}:/opt/rocm/lib"
    return library_path is None or library_path == expected_library_path


def run_capture(argv: list[str], timeout: int = 30, env: dict[str, str] | None = None) -> dict[str, Any]:
    selected_env = evidence_env() if env is None else dict(env)
    if not subprocess_env_is_allowed(selected_env):
        return {"argv": argv, "environment": selected_env, "returncode": None, "stdout": "", "stderr": "", "exec_error": "non-allowlisted subprocess environment"}
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False, env=selected_env)
        return {"argv": argv, "environment": selected_env, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"argv": argv, "environment": selected_env, "returncode": None, "stdout": "", "stderr": "", "exec_error": repr(exc)}


def rocm_dynamic_command(pid: int | None = None) -> list[str]:
    command = ["rocm-smi", "--showpids"]
    if pid is not None:
        command.extend(["--showpidgpus", str(pid)])
    command.extend(["--showmeminfo", "vram", "--showuse"])
    return command


def collect_rocm_snapshot(pid: int | None = None) -> dict[str, Any]:
    commands = [
        rocm_dynamic_command(pid),
        ["rocm-smi", "--showclocks"],
        ["rocm-smi", "--showpower"],
        ["rocm-smi", "--showtemp"],
    ]
    if shutil.which("rocm-smi", path=SAFE_PATH) is None:
        return {"available": False, "reason": "rocm-smi not found", "commands": commands}
    return {"available": True, "captures": [run_capture(command, timeout=30) for command in commands]}


def collect_dynamic_rocm_snapshot(pid: int) -> dict[str, Any]:
    command = rocm_dynamic_command(pid)
    return {"available": shutil.which("rocm-smi", path=SAFE_PATH) is not None, "captures": [run_capture(command, timeout=30)]}


def collect_hardware_state() -> dict[str, Any]:
    return {
        "gpu_product": run_capture(["rocm-smi", "--showproductname"], timeout=30),
        "gfx_target": run_capture(["rocminfo"], timeout=30),
        "rocm_runtime": run_capture(["hipconfig", "--version"], timeout=30),
        "rocm_driver": run_capture(["rocm-smi", "--showdriverversion"], timeout=30),
        "kernel": run_capture(["uname", "-srvmo"], timeout=10),
    }


def hardware_state_is_valid(state: dict[str, Any]) -> bool:
    required = ("gpu_product", "gfx_target", "rocm_runtime", "rocm_driver", "kernel")
    if any((state.get(key) or {}).get("returncode") != 0 for key in required):
        return False
    return (
        "mi210" in str(state["gpu_product"].get("stdout") or "").lower()
        and "gfx90a" in str(state["gfx_target"].get("stdout") or "").lower()
        and bool(str(state["rocm_runtime"].get("stdout") or "").strip())
        and bool(str(state["rocm_driver"].get("stdout") or "").strip())
        and bool(str(state["kernel"].get("stdout") or "").strip())
    )


def proc_fd_owners(target: str | None = None, listener_only: bool = False) -> dict[str, Any]:
    try:
        listener_inodes: dict[str, int] = {}
        if listener_only:
            for name in ("tcp", "tcp6"):
                lines = (Path("/proc/net") / name).read_text(encoding="utf-8").splitlines()[1:]
                for line in lines:
                    fields = line.split()
                    if len(fields) >= 10 and fields[3] == "0A":
                        listener_inodes[fields[9]] = int(fields[1].rsplit(":", 1)[1], 16)
        owners: list[dict[str, Any]] = []
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                for fd in (proc_dir / "fd").iterdir():
                    link = os.readlink(fd)
                    if target and link == target:
                        owners.append({"pid": int(proc_dir.name), "fd": fd.name, "target": link})
                    if listener_only:
                        match = re.fullmatch(r"socket:\[(\d+)\]", link)
                        if match and match.group(1) in listener_inodes:
                            owners.append({"pid": int(proc_dir.name), "fd": fd.name, "target": link, "port": listener_inodes[match.group(1)]})
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return {"returncode": 0, "owners": owners}
    except (OSError, ValueError) as exc:
        return {"returncode": None, "owners": [], "exec_error": repr(exc)}


def exact_process_owners(names: tuple[str, ...]) -> dict[str, Any]:
    owners: list[dict[str, Any]] = []
    try:
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                comm = (proc_dir / "comm").read_text(encoding="utf-8").strip()
                exe_link = os.readlink(proc_dir / "exe")
                exe = Path(exe_link).name
                if comm in names or exe in names:
                    owners.append({"pid": int(proc_dir.name), "comm": comm, "exe": exe, "exe_path": exe_link, "exe_resolved": str(Path(exe_link).resolve())})
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        commands = {
            name: {
                "argv": ["/proc", "exact-comm-or-exe", name],
                "returncode": 0 if any(owner["comm"] == name or owner["exe"] == name for owner in owners) else 1,
                "stdout": "\n".join(
                    f"{owner['pid']} {owner['exe_path']}"
                    for owner in owners
                    if owner["comm"] == name or owner["exe"] == name
                ),
                "stderr": "",
            }
            for name in names
        }
        return {"commands": commands, "proc_owners": owners, "returncode": 0}
    except OSError as exc:
        return {"commands": {}, "proc_owners": [], "returncode": None, "exec_error": repr(exc)}


def cmdline_process_owners(names: tuple[str, ...]) -> dict[str, Any]:
    """Return exact /proc argv-basename owners without a pattern process tool."""
    owners: list[dict[str, Any]] = []
    try:
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                argv = [part.decode("utf-8", errors="replace") for part in (proc_dir / "cmdline").read_bytes().split(b"\0") if part]
                if any(Path(argument).name in names for argument in argv):
                    owners.append({"pid": int(proc_dir.name), "argv": argv})
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return {
            "argv": ["/proc", "argv-basename", *names],
            "returncode": 0 if owners else 1,
            "stdout": "\n".join(f"{owner['pid']} {' '.join(owner['argv'])}" for owner in owners),
            "stderr": "",
            "owners": owners,
        }
    except OSError as exc:
        return {"argv": ["/proc", "argv-basename", *names], "returncode": None, "stdout": "", "stderr": "", "owners": [], "exec_error": repr(exc)}


def process_snapshot() -> dict[str, Any]:
    return {
        "model_binaries": exact_process_owners(("llama-server", "llama-cli", "llama-bench")),
        "autopilot": cmdline_process_owners(("autopilot.py", "autopilot_supervisor.py")),
        "listeners_lsof": run_capture(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], timeout=10),
        "listeners_proc": proc_fd_owners(listener_only=True),
        "kfd_lsof": run_capture(["lsof", "-nP", "/dev/kfd"], timeout=10),
        "kfd_proc": proc_fd_owners(target="/dev/kfd"),
        "rocm_pids": run_capture(["rocm-smi", "--showpids"], timeout=30),
    }


def proc_maps(pid: int) -> dict[str, Any]:
    try:
        return {"returncode": 0, "stdout": (Path("/proc") / str(pid) / "maps").read_text(encoding="utf-8")}
    except (OSError, UnicodeError) as exc:
        return {"returncode": None, "stdout": "", "exec_error": repr(exc)}


def decode_proc_maps_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def mapped_files(maps: dict[str, Any]) -> list[dict[str, Any]]:
    if maps.get("returncode") != 0:
        return []
    mapped: list[dict[str, Any]] = []
    for line in str(maps.get("stdout") or "").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        major_minor = fields[3].split(":", 1)
        if len(major_minor) != 2:
            continue
        try:
            mapped.append({"path": decode_proc_maps_path(fields[5]), "dev": os.makedev(int(major_minor[0], 16), int(major_minor[1], 16)), "inode": int(fields[4])})
        except ValueError:
            continue
    return mapped


def mapped_identity_matches(mapped: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        Path(str(mapped.get("path") or "")).resolve() == Path(str(expected.get("resolved_path") or expected.get("path") or "")).resolve()
        and mapped.get("dev") == expected.get("dev")
        and mapped.get("inode") == expected.get("inode")
    )


def live_artifacts_valid(pid: int, expected_binding: dict[str, Any], require_drafter: bool) -> tuple[bool, str]:
    server = expected_binding.get("server") or {}
    binary = server.get("artifact") or {}
    if binary.get("stable") is not True:
        return False, "preflight server artifact identity is unstable"
    try:
        exe = os.stat(Path("/proc") / str(pid) / "exe")
    except OSError as exc:
        return False, f"cannot inspect /proc/{pid}/exe: {exc}"
    if stat_identity(exe) != {key: binary.get(key) for key in ("dev", "inode", "bytes", "mtime_ns")}:
        return False, f"/proc/{pid}/exe identity differs from preflight server artifact"
    maps = proc_maps(pid)
    if maps.get("returncode") != 0:
        return False, f"cannot inspect /proc/{pid}/maps"
    entries = mapped_files(maps)
    expected_libs = server.get("local_llama_ggml_libraries") or []
    actual_libs = [entry for entry in entries if Path(entry["path"]).name.startswith(("libllama", "libggml")) and Path(entry["path"]).parent == Path(str(server.get("artifact", {}).get("resolved_path") or server.get("path") or "")).parent]
    unique_actual_libs = {
        (str(Path(entry["path"]).resolve()), entry["dev"], entry["inode"]): entry
        for entry in actual_libs
    }
    unique_expected_libs = {
        (str(Path(str(entry.get("resolved_path") or entry.get("path") or "")).resolve()), entry.get("dev"), entry.get("inode")): entry
        for entry in expected_libs
    }
    if len(unique_actual_libs) != len(unique_expected_libs) or set(unique_actual_libs) != set(unique_expected_libs):
        return False, "mapped local libllama/libggml artifacts differ from preflight identities"
    models = expected_binding.get("models") or {}
    target = models.get("target") or {}
    drafter = models.get("drafter") or {}
    if not any(mapped_identity_matches(entry, target) for entry in entries):
        return False, "target GGUF is not mapped with its preflight identity"
    drafter_mapped = any(mapped_identity_matches(entry, drafter) for entry in entries)
    drafter_path_mapped = any(
        Path(entry["path"]).resolve() == Path(str(drafter.get("resolved_path") or drafter.get("path") or "")).resolve()
        for entry in entries
    )
    if require_drafter and not drafter_mapped:
        return False, "DFlash drafter GGUF is not mapped with its preflight identity"
    if not require_drafter and drafter_path_mapped:
        return False, "base arm mapped the forbidden DFlash drafter GGUF"
    return True, "ok"


def runtime_env(binary: Path) -> dict[str, str]:
    library_dir = str(binary.parent)
    return {
        **evidence_env(),
        "LD_LIBRARY_PATH": f"{library_dir}:/opt/rocm/lib",
    }


def git_state(source_root: Path) -> dict[str, Any]:
    return {
        "source_root": str(source_root),
        "branch": run_capture(["git", "-C", str(source_root), "branch", "--show-current"]),
        "commit": run_capture(["git", "-C", str(source_root), "rev-parse", "HEAD"]),
        "tracked_diff": run_capture(["git", "-C", str(source_root), "diff", "--name-only"]),
        "index_diff": run_capture(["git", "-C", str(source_root), "diff", "--cached", "--name-only"]),
        "untracked": run_capture(["git", "-C", str(source_root), "ls-files", "--others", "--exclude-standard"]),
    }


def binary_identity(binary: Path, source_root: Path) -> dict[str, Any]:
    env = runtime_env(binary)
    server_version = run_capture([str(binary), "--version"], timeout=30, env=env) if binary.is_file() else None
    ldd = run_capture(["ldd", str(binary)], timeout=30, env=env) if binary.is_file() else None
    artifact = stable_file_identity(binary)
    return {
        "binary": str(binary),
        "artifact": artifact,
        "binary_sha256": artifact.get("sha256"),
        "ld_library_path": env["LD_LIBRARY_PATH"],
        "environment": env,
        "scrubbed_parent_env_keys": scrubbed_parent_env_keys(),
        "server_version": server_version,
        "ldd": ldd,
        "local_llama_ggml_libraries": local_library_identities(ldd),
        "git": git_state(source_root) if source_root.is_dir() else None,
    }


def immutable_model_identity(path: Path) -> dict[str, Any]:
    return stable_file_identity(path)


def immutable_file_identity(path: Path) -> dict[str, Any]:
    return stable_file_identity(path)


def local_library_identities(ldd: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not ldd or ldd.get("returncode") != 0:
        return []
    identities: list[dict[str, Any]] = []
    for line in str(ldd.get("stdout") or "").splitlines():
        match = re.match(r"^\s*(lib(?:llama|ggml)[^\s]*)\s+=>\s+(/[^\s]+)\s+\(", line)
        if match is None:
            continue
        ldd_path = Path(match.group(2))
        resolved_path = str(ldd_path)
        try:
            resolved = ldd_path.resolve(strict=True)
            resolved_path = str(resolved)
            if resolved.parent != DEFAULT_BINARY.parent.resolve():
                raise ValueError("resolved library is outside the canonical binary directory")
            identity = immutable_file_identity(resolved)
            # Preserve the SONAME link reported by ldd while hashing the real file.
            identity["path"] = str(ldd_path)
        except (OSError, RuntimeError, ValueError) as exc:
            identity = {
                "path": str(ldd_path),
                "resolved_path": resolved_path,
                "stable": False,
                "identity_error": str(exc),
            }
        identity["soname"] = match.group(1)
        identities.append(identity)
    return sorted(identities, key=lambda item: (str(item["soname"]), str(item["resolved_path"])))


def git_ref_commit(repo: Path, ref: str) -> str | None:
    command = run_capture(["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"])
    value = str(command.get("stdout") or "").strip()
    return value if command.get("returncode") == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


def verified_governance_attestation(path: Path) -> tuple[bytes | None, dict[str, Any] | None, str]:
    if path != PROMOTION_ATTESTATION_PATH:
        return None, None, f"attestation must use the exact canonical governance path: {PROMOTION_ATTESTATION_PATH}"
    try:
        leaf = path.lstat()
        if not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode):
            return None, None, "canonical attestation leaf must be a regular non-symlink file"
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (leaf.st_dev, leaf.st_ino):
                return None, None, "canonical attestation leaf changed while it was opened"
            raw = handle.read()
    except OSError as exc:
        return None, None, f"attestation governance provenance cannot be read: {exc}"
    relative = str(PROMOTION_ATTESTATION_RELATIVE_PATH)
    tracked = run_capture(["git", "-C", str(GOVERNANCE_REPO), "ls-files", "--error-unmatch", "--", relative])
    worktree = run_capture(["git", "-C", str(GOVERNANCE_REPO), "diff", "--quiet", "HEAD", "--", relative])
    index = run_capture(["git", "-C", str(GOVERNANCE_REPO), "diff", "--cached", "--quiet", "--", relative])
    head_blob = run_capture(["git", "-C", str(GOVERNANCE_REPO), "rev-parse", f"HEAD:{relative}"])
    if any(command.get("returncode") != 0 or command.get("exec_error") for command in (tracked, worktree, index, head_blob)):
        return None, None, "attestation must be tracked, unstaged, unmodified, and present in governance HEAD"
    blob_header = f"blob {len(raw)}\0".encode()
    disk_blob = hashlib.sha1(blob_header + raw, usedforsecurity=False).hexdigest()
    if disk_blob != head_blob.get("stdout", "").strip():
        return None, None, "attestation bytes differ from the governance HEAD object"
    identity = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "governance_head_blob": disk_blob,
    }
    return raw, identity, "ok"


def load_promotion_attestation(path: Path, expected_head: str, expected_server_sha256: str) -> tuple[dict[str, Any] | None, str]:
    if not path.is_absolute() or path.suffix != ".json":
        return None, "attestation must be an existing absolute JSON file"
    raw, identity, provenance_reason = verified_governance_attestation(path)
    if raw is None or identity is None:
        return None, provenance_reason
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        return None, f"attestation cannot be read as JSON: {exc}"
    required = {
        "schema": PROMOTION_ATTESTATION_SCHEMA,
        "status": "production_promoted_pending_gpu_certification",
        "production_branch": EXPECTED_BRANCH,
        "production_head": expected_head,
        "frozen": False,
    }
    if not isinstance(document, dict) or any(document.get(key) != value for key, value in required.items()):
        return None, "attestation must be the provisional v9 promotion record for this execution, not a final frozen record"
    server = document.get("server_binary")
    if not isinstance(server, dict) or server.get("path") != str(DEFAULT_BINARY) or server.get("sha256") != expected_server_sha256:
        return None, "attestation server binary path/SHA256 does not match the execution"
    if not isinstance(document.get("promoted_at"), str) or not document["promoted_at"].strip():
        return None, "attestation promoted_at is missing"
    try:
        promoted_at = datetime.fromisoformat(document["promoted_at"].replace("Z", "+00:00"))
    except ValueError:
        return None, "attestation promoted_at is not an ISO-8601 timestamp"
    if promoted_at.tzinfo is None:
        return None, "attestation promoted_at must include a timezone"
    rollback = document.get("rollback")
    if not isinstance(rollback, dict):
        return None, "attestation rollback metadata is missing"
    if rollback.get("branch") != ROLLBACK_BRANCH or not re.fullmatch(r"[0-9a-f]{40}", str(rollback.get("head") or "")):
        return None, f"attestation rollback branch/head must identify {ROLLBACK_BRANCH} exactly"
    backup_ref = rollback.get("backup_ref")
    source_ref = rollback.get("source_ref")
    expected_backup_ref = f"refs/heads/{ROLLBACK_BRANCH}"
    expected_source_ref = f"refs/heads/{EXPECTED_BRANCH}"
    if backup_ref != expected_backup_ref or source_ref != expected_source_ref:
        return None, f"attestation rollback refs must be the canonical {ROLLBACK_BRANCH} backup and {EXPECTED_BRANCH} production refs"
    backup_commit = git_ref_commit(DEFAULT_SOURCE_ROOT, backup_ref)
    source_commit = git_ref_commit(DEFAULT_SOURCE_ROOT, source_ref)
    if backup_commit is None or source_commit is None:
        return None, "attestation rollback refs must resolve in the canonical production repository"
    if backup_commit != rollback["head"] or source_commit != expected_head or backup_commit == source_commit:
        return None, "attestation rollback refs do not resolve to the distinct attested rollback/production commits"
    return {**identity, "document": document}, "ok"


def harness_identity() -> dict[str, Any]:
    script = Path(__file__).resolve()
    state = git_state(RESEARCH_ROOT)
    relative = str(script.relative_to(RESEARCH_ROOT))
    return {
        **stable_file_identity(script), "git": state,
        "tracked": run_capture(["git", "-C", str(RESEARCH_ROOT), "ls-files", "--error-unmatch", relative]),
        "worktree_unchanged": run_capture(["git", "-C", str(RESEARCH_ROOT), "diff", "--quiet", "HEAD", "--", relative]),
        "index_unchanged": run_capture(["git", "-C", str(RESEARCH_ROOT), "diff", "--cached", "--quiet", "--", relative]),
    }


def harness_valid(identity: dict[str, Any]) -> tuple[bool, str]:
    if identity.get("stable") is not True or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256") or "")):
        return False, "harness artifact identity is unstable or unhashed"
    for key in ("tracked", "worktree_unchanged", "index_unchanged"):
        command = identity.get(key) or {}
        if command.get("returncode") != 0 or command.get("exec_error"):
            return False, f"harness is not committed and unchanged: {key}"
    return True, "ok"


def version_commit_prefix(version_capture: dict[str, Any] | None) -> str | None:
    """Return llama.cpp's parenthesized lowercase commit token from its version line."""
    if not isinstance(version_capture, dict):
        return None
    output = "\n".join(
        value
        for value in (version_capture.get("stdout"), version_capture.get("stderr"))
        if isinstance(value, str) and value
    )
    match = re.search(r"(?m)^version: [^\r\n]*\(([0-9a-f]{9,40})\)\r?$", output)
    return match.group(1) if match is not None else None


def production_identity_valid(identity: dict[str, Any], expected_head: str, expected_server_sha256: str) -> tuple[bool, str]:
    git = identity.get("git") or {}
    branch = ((git.get("branch") or {}).get("stdout") or "").strip()
    tracked_diff = ((git.get("tracked_diff") or {}).get("stdout") or "").strip()
    index_diff = ((git.get("index_diff") or {}).get("stdout") or "").strip()
    untracked = [line.strip() for line in ((git.get("untracked") or {}).get("stdout") or "").splitlines() if line.strip()]
    commit = ((git.get("commit") or {}).get("stdout") or "").strip()
    version_commit = version_commit_prefix(identity.get("server_version"))
    ldd = ((identity.get("ldd") or {}).get("stdout") or "")
    source_root = Path(str(git.get("source_root") or "")).resolve()
    binary = Path(str(identity.get("binary") or "")).resolve()
    expected_binary = (DEFAULT_SOURCE_ROOT / "build-hip/bin/llama-server").resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head) or not re.fullmatch(r"[0-9a-f]{64}", expected_server_sha256):
        return False, "frozen production HEAD and server SHA256 must be supplied exactly"
    if not identity.get("binary_sha256"):
        return False, "canonical binary is missing or unhashed"
    if (identity.get("artifact") or {}).get("stable") is not True:
        return False, "canonical binary artifact identity is unstable"
    if identity.get("environment") != runtime_env(DEFAULT_BINARY):
        return False, "binary identity was captured with a non-canonical environment"
    for command in (git.get("branch") or {}, git.get("commit") or {}, git.get("tracked_diff") or {}, git.get("index_diff") or {}, git.get("untracked") or {}, identity.get("server_version") or {}, identity.get("ldd") or {}):
        if command.get("returncode") != 0 or command.get("exec_error"):
            return False, "required git/version/ldd command failed"
    if branch != EXPECTED_BRANCH:
        return False, f"expected branch {EXPECTED_BRANCH}, got {branch or '<none>'}"
    if source_root != DEFAULT_SOURCE_ROOT.resolve() or binary != expected_binary:
        return False, "source root or binary is not the canonical production HIP path"
    if tracked_diff or index_diff:
        return False, "production tracked/index state is dirty"
    disallowed = [
        path
        for path in untracked
        if not any(path.startswith(prefix) if prefix.endswith("/") else path == prefix for prefix in SOURCE_UNTRACKED_ALLOWLIST)
    ]
    if disallowed:
        return False, f"production has non-allowlisted untracked files: {disallowed}"
    if commit != expected_head or identity.get("binary_sha256") != expected_server_sha256:
        return False, "production HEAD or server SHA256 differs from frozen expected identity"
    if version_commit is None or not expected_head.startswith(version_commit):
        return False, "llama-server version does not contain a valid production commit prefix"
    version_output = "\n".join(
        value
        for value in ((identity.get("server_version") or {}).get("stdout"), (identity.get("server_version") or {}).get("stderr"))
        if isinstance(value, str) and value
    )
    if "HIP" not in version_output and "ROCm" not in version_output and "hip" not in ldd.lower():
        return False, "binary does not prove a HIP/ROCm backend"
    libraries = identity.get("local_llama_ggml_libraries")
    if not isinstance(libraries, list) or not libraries:
        return False, "local llama/ggml shared-library identities are missing"
    sonames = [str(item.get("soname") or "") for item in libraries if isinstance(item, dict)]
    if not any(name.startswith("libllama") for name in sonames) or not any(name.startswith("libggml") for name in sonames):
        return False, "ldd output does not prove both libllama and libggml provenance"
    for library in libraries:
        if (
            not isinstance(library, dict)
            or Path(str(library.get("resolved_path") or "")).parent != expected_binary.parent
            or library.get("stable") is not True
            or not isinstance(library.get("bytes"), int)
            or library["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(library.get("sha256") or ""))
        ):
            return False, f"invalid or non-canonical local shared-library identity: {library}"
    return True, "ok"


def model_identities_valid(target: dict[str, Any], drafter: dict[str, Any]) -> tuple[bool, str]:
    if Path(str(target.get("path") or "")).resolve() != DEFAULT_TARGET_MODEL.resolve() or target.get("bytes") != TARGET_MODEL_BYTES or target.get("sha256") != TARGET_MODEL_SHA256:
        return False, "Qwen3.6-27B Q8 target size/SHA256 mismatch"
    if Path(str(drafter.get("path") or "")).resolve() != DEFAULT_DRAFTER_MODEL.resolve() or drafter.get("bytes") != DRAFTER_MODEL_BYTES or drafter.get("sha256") != DRAFTER_MODEL_SHA256:
        return False, "Qwen3.6-27B DFlash F16 size/SHA256 mismatch"
    return True, "ok"


def execution_binding(
    binary: dict[str, Any],
    target: dict[str, Any],
    drafter: dict[str, Any],
    harness: dict[str, Any],
    attestation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "server": {
            "path": binary.get("binary"),
            "sha256": binary.get("binary_sha256"),
            "artifact": binary.get("artifact"),
            "local_llama_ggml_libraries": binary.get("local_llama_ggml_libraries"),
        },
        "models": {"target": target, "drafter": drafter},
        "harness": {key: harness.get(key) for key in ("path", "resolved_path", "dev", "inode", "bytes", "mtime_ns", "sha256", "stable")},
        "promotion_attestation": {"path": attestation.get("path"), "sha256": attestation.get("sha256")},
    }


def validate_execution_identity(
    args: argparse.Namespace,
    target_identity: dict[str, Any],
    drafter_identity: dict[str, Any],
) -> dict[str, Any]:
    binary = binary_identity(args.binary, args.source_root)
    production_ok, production_reason = production_identity_valid(
        binary,
        args.expected_production_head,
        args.expected_server_sha256,
    )
    attestation, attestation_reason = load_promotion_attestation(
        args.attestation_ref,
        args.expected_production_head,
        args.expected_server_sha256,
    )
    harness = harness_identity()
    harness_ok, harness_reason = harness_valid(harness)
    models_ok, models_reason = model_identities_valid(target_identity, drafter_identity)
    binding = execution_binding(binary, target_identity, drafter_identity, harness, attestation or {})
    valid = production_ok and attestation is not None and harness_ok and models_ok
    return {
        "valid": valid,
        "binding": binding,
        "binary": binary,
        "production_valid": production_ok,
        "production_reason": production_reason,
        "attestation": attestation,
        "attestation_valid": attestation is not None,
        "attestation_reason": attestation_reason,
        "harness": harness,
        "harness_valid": harness_ok,
        "harness_reason": harness_reason,
        "models_valid": models_ok,
        "models_reason": models_reason,
    }


def snapshot_is_valid(snapshot: dict[str, Any]) -> bool:
    captures = snapshot.get("captures") or []
    if not snapshot.get("available") or not captures or any(item.get("returncode") != 0 or not str(item.get("stdout") or "").strip() for item in captures):
        return False
    text = "\n".join(str(item.get("stdout") or "") for item in captures).lower()
    dynamic = "kfd processes" in text and "gpu use (%)" in text and "vram total used memory (b)" in text
    if len(captures) == 1:
        return dynamic
    return dynamic and ("sclk" in text or "clock" in text) and "power" in text and ("temperature" in text or "temp" in text)


def parse_rocm_pid_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    text = "\n".join(str(item.get("stdout") or "") for item in snapshot.get("captures", []))
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = re.fullmatch(
            r"\s*([1-9][0-9]*)\s+(\S+)\s+(\S+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*",
            line,
        )
        if match:
            rows.append({"pid": int(match.group(1)), "process_name": match.group(2), "gpus": match.group(3), "vram_bytes": int(match.group(4))})
            continue
        pid_prefix = re.match(r"^\s*([1-9][0-9]*)\s+", line)
        if pid_prefix:
            rows.append({"pid": int(pid_prefix.group(1)), "process_name": None, "gpus": None, "vram_bytes": 0, "malformed": True, "raw": line})
    return rows


def parse_rocm_pid_gpu_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    text = "\n".join(str(item.get("stdout") or "") for item in snapshot.get("captures", []))
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"\s*PID\s+([1-9][0-9]*)\s+is using\s+([0-9]+)\s+DRM device\(s\):?\s*", line)
        if match is None:
            continue
        pid = int(match.group(1))
        declared_count = int(match.group(2))
        devices: list[int] = []
        malformed = False
        for offset in range(1, declared_count + 1):
            if index + offset >= len(lines):
                malformed = True
                break
            device_match = re.fullmatch(r"\s*([0-9]+)\s*", lines[index + offset])
            if device_match is None:
                malformed = True
                break
            devices.append(int(device_match.group(1)))
        rows.append(
            {
                "pid": pid,
                "declared_device_count": declared_count,
                "devices": devices,
                "malformed": malformed or len(devices) != declared_count,
            }
        )
    return rows


def vram_used_mb(snapshot: dict[str, Any]) -> float | None:
    text = "\n".join(str(item.get("stdout") or "") for item in snapshot.get("captures", []))
    bytes_match = re.search(r"VRAM Total Used Memory \(B\):\s*([0-9]+)", text, flags=re.IGNORECASE)
    if bytes_match:
        return int(bytes_match.group(1)) / (1024.0 * 1024.0)
    match = re.search(r"(?:vram[^\n]*?used|used[^\n]*?vram|used)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mib|mb)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def vram_settled(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_mb = vram_used_mb(before)
    after_mb = vram_used_mb(after)
    return before_mb is not None and after_mb is not None and after_mb <= before_mb + 64.0


def process_guard_clean(snapshot: dict[str, Any], port: int | None = None) -> tuple[bool, str]:
    model_binaries = snapshot.get("model_binaries") or {}
    if model_binaries.get("returncode") != 0 or model_binaries.get("exec_error"):
        return False, "required exact model-process proof failed"
    for name, command in (model_binaries.get("commands") or {}).items():
        if command.get("returncode") not in (0, 1) or command.get("exec_error"):
            return False, f"exact process proof failed: {name}"
        if command.get("returncode") == 0:
            return False, f"process blocker: {name}"
    if model_binaries.get("proc_owners"):
        return False, "model binary process owner present"
    for key in ("autopilot",):
        command = snapshot.get(key) or {}
        if command.get("returncode") not in (0, 1) or command.get("exec_error"):
            return False, f"required process proof failed: {key}"
        if command.get("returncode") == 0:
            return False, f"process blocker: {key}"
    for key in ("listeners_lsof", "kfd_lsof"):
        command = snapshot.get(key) or {}
        if command.get("returncode") not in (0, 1) or command.get("exec_error"):
            return False, f"required process proof failed: {key}"
    for key in ("listeners_proc", "kfd_proc"):
        if (snapshot.get(key) or {}).get("returncode") != 0:
            return False, f"required /proc proof failed: {key}"
    if (snapshot.get("kfd_lsof") or {}).get("returncode") == 0 or (snapshot.get("kfd_proc") or {}).get("owners"):
        return False, "KFD owner present"
    rocm = snapshot.get("rocm_pids") or {}
    if rocm.get("returncode") != 0:
        return False, "rocm-smi --showpids failed"
    rocm_text = str(rocm.get("stdout") or "").lower()
    if "no kfd" not in rocm_text and re.search(r"\bpid\s*[:=]?\s*[1-9][0-9]*\b", rocm_text):
        return False, "ROCm reports a KFD PID owner"
    listeners = str((snapshot.get("listeners_lsof") or {}).get("stdout") or "")
    proc_ports = [int(owner["port"]) for owner in (snapshot.get("listeners_proc") or {}).get("owners", [])]
    if port is not None and (re.search(rf":{port}\b", listeners) or port in proc_ports):
        return False, f"listener already occupies port {port}"
    return True, "ok"


def live_binding_is_valid(processes: dict[str, Any], rocm: dict[str, Any], pid: int, port: int, binary: Path, expected_binding: dict[str, Any] | None = None, require_drafter: bool = False) -> tuple[bool, str]:
    if not snapshot_is_valid(rocm):
        return False, "live ROCm snapshot is incomplete"
    model = processes.get("model_binaries") or {}
    if model.get("returncode") != 0:
        return False, "exact model-process scan failed"
    owners = {int(item["pid"]) for item in model.get("proc_owners", [])}
    if owners != {pid}:
        return False, f"model-process owners are not exactly server PID {pid}: {sorted(owners)}"
    owner_rows = [item for item in model.get("proc_owners", []) if int(item["pid"]) == pid]
    if len(owner_rows) != 1 or Path(str(owner_rows[0].get("exe_resolved") or "")).resolve() != binary.resolve():
        return False, f"/proc/{pid}/exe does not resolve to the frozen server binary"
    commands = model.get("commands") or {}
    if (commands.get("llama-server") or {}).get("returncode") != 0 or any((commands.get(name) or {}).get("returncode") != 1 for name in ("llama-cli", "llama-bench")):
        return False, "exact model command ownership is contaminated"
    if (processes.get("autopilot") or {}).get("returncode") != 1:
        return False, "AutoPilot is present or its proof failed"
    kfd_owners = {int(item["pid"]) for item in (processes.get("kfd_proc") or {}).get("owners", [])}
    if kfd_owners != {pid}:
        return False, f"/dev/kfd owners are not exactly server PID {pid}: {sorted(kfd_owners)}"
    kfd_lsof = processes.get("kfd_lsof") or {}
    if kfd_lsof.get("returncode") != 0 or not re.search(rf"\b{pid}\b", str(kfd_lsof.get("stdout") or "")):
        return False, "lsof does not bind /dev/kfd to server PID"
    port_owners = {int(item["pid"]) for item in (processes.get("listeners_proc") or {}).get("owners", []) if int(item.get("port", -1)) == port}
    if port_owners != {pid}:
        return False, f"port {port} listener owners are not exactly server PID {pid}: {sorted(port_owners)}"
    listeners = processes.get("listeners_lsof") or {}
    if listeners.get("returncode") != 0 or not re.search(rf"\b{pid}\b.*:{port}\b", str(listeners.get("stdout") or "")):
        return False, "lsof does not bind target listener to server PID"
    rocm_rows = parse_rocm_pid_rows(rocm)
    rocm_pids = {int(row["pid"]) for row in rocm_rows}
    if rocm_pids != {pid} or not all(
        int(row["vram_bytes"]) > 0
        and not row.get("malformed")
        and re.fullmatch(r"[1-9][0-9]*", str(row.get("gpus") or ""))
        for row in rocm_rows
    ):
        return False, f"ROCm PID mapping is not solely resident server PID {pid}: {rocm_rows}"
    gpu_rows = parse_rocm_pid_gpu_rows(rocm)
    if gpu_rows != [
        {
            "pid": pid,
            "declared_device_count": 1,
            "devices": [0],
            "malformed": False,
        }
    ]:
        return False, f"ROCm physical-device mapping is not exactly GPU 0 for server PID {pid}: {gpu_rows}"
    if expected_binding is not None:
        artifacts_valid, artifacts_reason = live_artifacts_valid(pid, expected_binding, require_drafter)
        if not artifacts_valid:
            return False, artifacts_reason
    return True, "ok"


def collect_live_binding_evidence(pid: int, port: int, binary: Path, phase: str, expected_binding: dict[str, Any] | None = None, require_drafter: bool = False) -> dict[str, Any]:
    sample_started = time.monotonic()
    processes_before = process_snapshot()
    rocm = collect_dynamic_rocm_snapshot(pid)
    processes_after = process_snapshot()
    sample_ended = time.monotonic()
    before_valid, before_reason = live_binding_is_valid(processes_before, rocm, pid, port, binary, expected_binding, require_drafter)
    after_valid, after_reason = live_binding_is_valid(processes_after, rocm, pid, port, binary, expected_binding, require_drafter)
    valid = before_valid and after_valid
    reason = "ok" if valid else f"before={before_reason}; after={after_reason}"
    return {
        "phase": phase,
        "captured_at": utc_now(),
        "sample_started_monotonic": sample_started,
        "sample_ended_monotonic": sample_ended,
        "processes_before": processes_before,
        "rocm": rocm,
        "processes_after": processes_after,
        "valid": valid,
        "reason": reason,
    }


def query_with_live_samples(port: int, body: dict[str, Any], timeout_s: int, pid: int, binary: Path, prompt_index: int, expected_binding: dict[str, Any] | None = None, require_drafter: bool = False) -> tuple[dict[str, Any], float, dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    sampler_ready = threading.Event()
    request_active = threading.Event()

    def sample() -> None:
        sampler_ready.set()
        request_active.wait()
        while not stop.is_set():
            samples.append(collect_live_binding_evidence(pid, port, binary, f"during_request_{prompt_index}_{len(samples) + 1}", expected_binding, require_drafter))
            stop.wait(0.25)

    sampler = threading.Thread(target=sample, name=f"pgpu1-rocm-sampler-{prompt_index}", daemon=True)
    sampler.start()
    if not sampler_ready.wait(timeout=5):
        stop.set()
        sampler.join(timeout=5)
        raise RuntimeError("request lifecycle sampler did not start")
    started = time.monotonic()
    request_active.set()
    try:
        response = query_chat(port, body, timeout_s)
    finally:
        ended = time.monotonic()
        stop.set()
        sampler.join(timeout=35)
    elapsed = ended - started
    contained = [
        sample
        for sample in samples
        if sample["sample_started_monotonic"] >= started and sample["sample_ended_monotonic"] <= ended
    ]
    if sampler.is_alive() or not contained or any(not sample["valid"] for sample in contained):
        raise RuntimeError(f"request lifecycle binding failed: {samples}")
    lifecycle = {
        "request_started_monotonic": started,
        "request_ended_monotonic": ended,
        "elapsed_s": elapsed,
        "samples": samples,
        "fully_contained_sample_count": len(contained),
        "fully_contained_valid": True,
    }
    return response, elapsed, lifecycle


def server_argv(args: argparse.Namespace, arm: Arm, port: int) -> list[str]:
    argv = [
        str(args.binary), "-m", str(args.target_model), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(args.context), "-ngl", "all", "-dev", "ROCm0", "-ot", "token_embd.weight=ROCm0", "-fa", "on",
        "--cache-type-k", TARGET_CACHE_K, "--cache-type-v", TARGET_CACHE_V,
        "--seed", str(args.seed), "--temp", "0", "--top-k", "1", "--top-p", "1", "--jinja",
        "--reasoning", "off", "--reasoning-budget", "0", "-v",
    ]
    if arm.speculative:
        argv.extend([
            "-md", str(args.drafter_model), "--spec-draft-device", "ROCm0", "--spec-draft-ngl", "all",
            "--spec-type", "draft-dflash", "--spec-draft-n-max", "15", "--spec-draft-n-min", "0",
            "--spec-draft-p-min", "0", "--spec-draft-type-k", DRAFTER_CACHE_K,
            "--spec-draft-type-v", DRAFTER_CACHE_V,
        ])
    return argv


def request_body(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "stream": False,
    }


def wait_for_health(port: int, timeout_s: int, proc: subprocess.Popen[str] | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server exited before health became ready (returncode={proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - retained in timeout diagnostic
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"llama-server health check timed out: {last_error}")


def query_chat(port: int, body: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc


def terminate(proc: subprocess.Popen[str]) -> dict[str, Any]:
    evidence: dict[str, Any] = {"pid": proc.pid, "term_sent": False, "kill_sent": False, "errors": []}
    try:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            evidence["term_sent"] = True
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                evidence["kill_sent"] = True
                proc.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        evidence["errors"].append(repr(exc))
    group = run_capture(["pgrep", "-g", str(proc.pid)], timeout=10)
    if group.get("returncode") == 0:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            evidence["kill_sent"] = True
        except ProcessLookupError:
            pass
        except OSError as exc:
            evidence["errors"].append(repr(exc))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            group = run_capture(["pgrep", "-g", str(proc.pid)], timeout=10)
            if group.get("returncode") == 1:
                break
            time.sleep(0.1)
    if proc.poll() is None:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            evidence["errors"].append(repr(exc))
    evidence.update({"completed": proc.poll() is not None, "dead": proc.poll() is not None and group.get("returncode") == 1, "returncode": proc.returncode, "process_group": group})
    return evidence


LOG_PREFIX = r"^\d+\.\d+\.\d+\.\d+\s+[A-Z]\s+(?:[A-Za-z0-9_~]+\s+){0,2}[A-Za-z0-9_~]+\s*:\s*"


def _positive_model_buffers(section: str, layer_count: int) -> list[float]:
    pattern = re.compile(
        rf"{LOG_PREFIX}offloaded {layer_count}/{layer_count} layers to GPU\s*\n"
        rf"{LOG_PREFIX}ROCm0 model buffer size =\s*([0-9]+(?:\.[0-9]+)?) MiB\s*$",
        flags=re.MULTILINE,
    )
    return [float(match.group(1)) for match in pattern.finditer(section) if float(match.group(1)) > 0]


def _positive_kv_buffers(section: str, *, cache_k: str, cache_v: str) -> list[dict[str, float]]:
    pattern = re.compile(
        rf"{LOG_PREFIX}ROCm0 KV buffer size =\s*([0-9]+(?:\.[0-9]+)?) MiB\s*\n"
        rf"{LOG_PREFIX}size =\s*([0-9]+(?:\.[0-9]+)?) MiB .*"
        rf"K \({re.escape(cache_k)}\):\s*([0-9]+(?:\.[0-9]+)?) MiB, "
        rf"V \({re.escape(cache_v)}\):\s*([0-9]+(?:\.[0-9]+)?) MiB\s*$",
        flags=re.MULTILINE,
    )
    rows = [
        {
            "device_buffer_mib": float(match.group(1)),
            "total_mib": float(match.group(2)),
            "k_mib": float(match.group(3)),
            "v_mib": float(match.group(4)),
        }
        for match in pattern.finditer(section)
    ]
    return [row for row in rows if all(value > 0 and math.isfinite(value) for value in row.values())]


def parse_log_residency(log_text: str, arm: Arm) -> dict[str, Any]:
    target_load = re.search(
        rf"{LOG_PREFIX}loading model '{re.escape(str(DEFAULT_TARGET_MODEL))}'\s*$",
        log_text,
        flags=re.MULTILINE,
    )
    draft_load = re.search(
        rf"{LOG_PREFIX}loading draft model '{re.escape(str(DEFAULT_DRAFTER_MODEL))}'\s*$",
        log_text,
        flags=re.MULTILINE,
    )
    split_at = draft_load.start() if draft_load is not None else len(log_text)
    target_section = log_text[:split_at]
    draft_section = log_text[split_at:] if draft_load is not None else ""
    target_models = _positive_model_buffers(target_section, TARGET_OFFLOAD_LAYER_COUNT)
    target_kv = _positive_kv_buffers(
        target_section, cache_k=TARGET_CACHE_K, cache_v=TARGET_CACHE_V
    )
    draft_models = _positive_model_buffers(draft_section, DRAFTER_OFFLOAD_LAYER_COUNT)
    draft_kv = _positive_kv_buffers(
        draft_section, cache_k=DRAFTER_CACHE_K, cache_v=DRAFTER_CACHE_V
    )
    target_valid = target_load is not None and len(target_models) == 1 and len(target_kv) == TARGET_POSITIVE_KV_BUFFER_COUNT
    draft_valid = (
        draft_load is not None and len(draft_models) == 1 and len(draft_kv) == DRAFTER_POSITIVE_KV_BUFFER_COUNT
        if arm.speculative
        else draft_load is None and not draft_models and not draft_kv
    )
    return {
        "passed": target_valid and draft_valid,
        "target_model_load_exact": target_load is not None,
        "target_positive_rocm0_model_buffers_mib": target_models,
        "target_positive_kv_buffers": target_kv,
        "target_valid": target_valid,
        "drafter_model_load_exact": draft_load is not None,
        "drafter_positive_rocm0_model_buffers_mib": draft_models,
        "drafter_positive_q8_kv_buffers": draft_kv,
        "drafter_valid": draft_valid,
        "proof_contract": (
            "anchored exact layer offload plus positive ROCm0 model/KV buffers; "
            "target and DFlash drafter K/V=q8_0 are independently required"
        ),
    }


def strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field} must be an integer, got {type(value).__name__}")
    return value


def strict_real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be a finite number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{field} must be finite")
    return result


def timings_from_response(response: dict[str, Any], *, speculative: bool) -> dict[str, Any]:
    timings = response.get("timings") or {}
    usage = response.get("usage") or {}
    prompt_ms = timings.get("prompt_ms")
    decode_ms = timings.get("predicted_ms")
    if prompt_ms is None or decode_ms is None:
        raise RuntimeError("response lacks required prompt_ms or predicted_ms timing")
    if speculative and ("draft_n" not in timings or "draft_n_accepted" not in timings):
        raise RuntimeError("response lacks draft counters")
    prompt_tokens_raw = usage["prompt_tokens"] if "prompt_tokens" in usage else timings.get("prompt_n")
    completion_tokens_raw = usage["completion_tokens"] if "completion_tokens" in usage else timings.get("predicted_n")
    result = {
        "prompt_tokens": strict_int(prompt_tokens_raw, "prompt_tokens"),
        "completion_tokens": strict_int(completion_tokens_raw, "completion_tokens"),
        "prompt_ms": strict_real(prompt_ms, "prompt_ms"),
        "decode_ms": strict_real(decode_ms, "predicted_ms"),
        "prompt_tps": strict_real(timings["prompt_per_second"], "prompt_per_second") if timings.get("prompt_per_second") is not None else None,
        "decode_tps": strict_real(timings["predicted_per_second"], "predicted_per_second") if timings.get("predicted_per_second") is not None else None,
        "draft_n": strict_int(timings["draft_n"], "draft_n") if speculative else 0,
        "draft_n_accepted": strict_int(timings["draft_n_accepted"], "draft_n_accepted") if speculative else 0,
    }
    if result["prompt_tokens"] <= 0 or result["completion_tokens"] <= 0 or result["prompt_ms"] <= 0 or result["decode_ms"] <= 0:
        raise RuntimeError("response has non-positive token or timing fields")
    if any(value is not None and value <= 0 for value in (result["prompt_tps"], result["decode_tps"])):
        raise RuntimeError("response has non-positive throughput fields")
    if speculative and (result["draft_n"] <= 0 or result["draft_n_accepted"] < 0 or result["draft_n_accepted"] > result["draft_n"]):
        raise RuntimeError("response has invalid DFlash draft counters")
    return result


def response_sanity(content: str) -> dict[str, Any]:
    printable = sum(character.isprintable() or character in "\n\r\t" for character in content)
    printable_ratio = printable / len(content) if content else 0.0
    words = re.findall(r"\b\w+\b", content.lower())
    max_word_run = 0
    current_run = 0
    previous = None
    for word in words:
        current_run = current_run + 1 if word == previous else 1
        max_word_run = max(max_word_run, current_run)
        previous = word
    max_char_run = max((len(match.group(0)) for match in re.finditer(r"(.)\1*", content)), default=0)
    unique_word_ratio = len(set(words)) / len(words) if words else 0.0
    passed = bool(content.strip()) and printable_ratio >= 0.98 and max_word_run < 20 and max_char_run < 64 and (len(words) < 30 or unique_word_ratio >= 0.05)
    return {"passed": passed, "printable_ratio": printable_ratio, "max_consecutive_word_run": max_word_run, "max_consecutive_character_run": max_char_run, "unique_word_ratio": unique_word_ratio, "word_count": len(words)}


def semantic_validation(prompt_id: str, content: str) -> dict[str, Any]:
    match = re.fullmatch(r"(?s)(?P<explanation>.+?)\s+RESULT_JSON:\s*(?P<result>\{[^\n]*\})\s*", content.strip())
    if match is None:
        return {"passed": False, "reason": "output must contain an explanation followed by one terminal RESULT_JSON object"}
    explanation_word_count = len(re.findall(r"\b[\w'-]+\b", match.group("explanation")))
    if explanation_word_count < MIN_EXPLANATION_WORDS:
        return {
            "passed": False,
            "reason": (
                f"explanation has {explanation_word_count} lexical words; "
                f"minimum is {MIN_EXPLANATION_WORDS}"
            ),
            "explanation_word_count": explanation_word_count,
        }
    try:
        value = json.loads(match.group("result"))
    except json.JSONDecodeError as exc:
        return {"passed": False, "reason": f"invalid RESULT_JSON: {exc}"}
    if prompt_id == "primes":
        primes = value.get("primes") if isinstance(value, dict) else None
        total = value.get("sum") if isinstance(value, dict) else None
        passed = (
            isinstance(primes, list)
            and all(isinstance(item, int) and not isinstance(item, bool) for item in primes)
            and primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            and isinstance(total, int)
            and not isinstance(total, bool)
            and total == 129
            and set(value) == {"primes", "sum"}
        )
    elif prompt_id == "nested_flatten":
        values = value.get("values") if isinstance(value, dict) else None
        passed = isinstance(values, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in values) and values == [1, 2, 3, 4, 5] and set(value) == {"values"}
    elif prompt_id == "normalize":
        normalized = value.get("normalized") if isinstance(value, dict) else None
        total = value.get("sum") if isinstance(value, dict) else None
        passed = isinstance(normalized, list) and len(normalized) == 4 and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in normalized) and all(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9) for actual, expected in zip(normalized, (0.0, 0.2, 0.3, 0.5), strict=True)) and isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(float(total)) and math.isclose(float(total), 1.0, rel_tol=0.0, abs_tol=1e-9) and set(value) == {"normalized", "sum"}
    else:
        return {"passed": False, "reason": f"unknown prompt id: {prompt_id}"}
    return {"passed": passed, "reason": "ok" if passed else "machine-checkable result is incorrect", "parsed_result": value, "explanation_word_count": explanation_word_count}


def finish_reason_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise RuntimeError("response does not contain exactly one completion choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise RuntimeError(f"response did not finish normally: {finish_reason!r}")
    return finish_reason


def median_mad(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "mad": None}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise RuntimeError("median/MAD inputs must be finite real numbers")
    median = float(statistics.median(values))
    return {"n": len(values), "median": median, "mad": float(statistics.median([abs(value - median) for value in values]))}


def summarize_arm(replicates: list[dict[str, Any]], arm: Arm, expected_reps: int | None = None) -> dict[str, Any]:
    ok = [rep for rep in replicates if rep.get("status") == "ok"]
    draft_n = sum(strict_int(rep.get("draft_n") or 0, "draft_n") for rep in ok)
    accepted = sum(strict_int(rep.get("draft_n_accepted") or 0, "draft_n_accepted") for rep in ok)
    expected = len(replicates) if expected_reps is None else expected_reps
    return {
        "replicates": len(replicates), "expected_replicates": expected, "ok_replicates": len(ok), "all_ok": len(replicates) == expected and len(ok) == expected,
        "prompt_tps": median_mad([strict_real(rep["prompt_tps"], "prompt_tps") for rep in ok if rep.get("prompt_tps") is not None]),
        "decode_tps": median_mad([strict_real(rep["decode_tps"], "decode_tps") for rep in ok if rep.get("decode_tps") is not None]),
        "draft_n": draft_n if arm.speculative else 0,
        "draft_n_accepted": accepted if arm.speculative else 0,
        "draft_acceptance_rate": (accepted / draft_n) if arm.speculative and draft_n else (0.0 if not arm.speculative else None),
        "draft_acceptance_rate_per_rep": median_mad([strict_real(rep["draft_acceptance_rate"], "draft_acceptance_rate") for rep in ok if rep.get("draft_acceptance_rate") is not None]) if arm.speculative else {"n": len(ok), "median": 0.0, "mad": 0.0},
        "draft_counters": "required_parsed" if arm.speculative else "not_applicable_zero",
    }


def matrix_cardinality_valid(results: list[dict[str, Any]], reps: int) -> tuple[bool, str]:
    expected_prompt_ids = [prompt_id for prompt_id, _ in PROMPT_SPECS]
    if len(results) != len(ARMS) * reps:
        return False, f"expected {len(ARMS) * reps} replicate results, got {len(results)}"
    for arm in ARMS:
        rows = [row for row in results if row.get("arm") == arm.name]
        rep_ids = [row.get("rep") for row in rows]
        if (
            len(rows) != reps
            or any(isinstance(rep_id, bool) or not isinstance(rep_id, int) for rep_id in rep_ids)
            or len(set(rep_ids)) != reps
            or sorted(rep_ids) != list(range(1, reps + 1))
        ):
            return False, f"{arm.name} replicate IDs are incomplete or duplicated: {rep_ids}"
        for row in rows:
            if row.get("status") != "ok":
                return False, f"{arm.name} rep {row.get('rep')} is not successful"
            records = row.get("records")
            if row.get("prompt_count") != len(PROMPT_SPECS) or not isinstance(records, list) or len(records) != len(PROMPT_SPECS):
                return False, f"{arm.name} rep {row.get('rep')} has incomplete prompt cardinality"
            if [record.get("prompt_index") for record in records] != list(range(1, len(PROMPT_SPECS) + 1)):
                return False, f"{arm.name} rep {row.get('rep')} has invalid prompt indices"
            if [record.get("prompt_id") for record in records] != expected_prompt_ids:
                return False, f"{arm.name} rep {row.get('rep')} has invalid prompt IDs"
            if any(record.get("finish_reason") != "stop" for record in records):
                return False, f"{arm.name} rep {row.get('rep')} has a non-stop finish reason"
            if any((record.get("semantic_validation") or {}).get("passed") is not True for record in records):
                return False, f"{arm.name} rep {row.get('rep')} has a failed semantic validator"
            for record in records:
                lifecycle = record.get("request_lifecycle") or {}
                sample_count = lifecycle.get("fully_contained_sample_count")
                if (
                    lifecycle.get("fully_contained_valid") is not True
                    or isinstance(sample_count, bool)
                    or not isinstance(sample_count, int)
                    or sample_count < 1
                ):
                    return False, f"{arm.name} rep {row.get('rep')} lacks a fully contained valid request sample"
    return True, "ok"


def candidate_smoke_projection(
    results: list[dict[str, Any]],
    *,
    initial_rocm: dict[str, Any],
    final_rocm: dict[str, Any],
    identity: dict[str, Any],
    final_port: int | None,
) -> dict[str, Any]:
    """Render the old smoke evidence shape from canonical-runner results.

    This is deliberately an observation-only compatibility projection.  It
    preserves the useful per-cell semantic and cleanup evidence from the
    one-off smoke, but it is never a promotion or performance verdict.
    """
    expected_prompt_ids = [prompt_id for prompt_id, _ in PROMPT_SPECS]
    cells: list[dict[str, Any]] = []
    for result in results:
        records = result.get("records")
        records = records if isinstance(records, list) else []
        prompt_ids = [record.get("prompt_id") for record in records if isinstance(record, dict)]
        all_finished_stop = bool(records) and all(record.get("finish_reason") == "stop" for record in records if isinstance(record, dict))
        all_semantic_passed = bool(records) and all((record.get("semantic_validation") or {}).get("passed") is True for record in records if isinstance(record, dict))
        all_completion_floor_passed = bool(records) and all(
            isinstance(record.get("completion_tokens"), int) and record["completion_tokens"] >= DEFAULT_MIN_COMPLETION_TOKENS
            for record in records
            if isinstance(record, dict)
        )
        cleanup = result.get("cleanup") if isinstance(result.get("cleanup"), dict) else {}
        cells.append(
            {
                "arm": result.get("arm"),
                "rep": result.get("rep"),
                "status": result.get("status"),
                "error": result.get("error"),
                "prompt_count": result.get("prompt_count"),
                "prompt_ids": prompt_ids,
                "all_finished_stop": all_finished_stop,
                "all_semantic_passed": all_semantic_passed,
                "all_completion_floor_passed": all_completion_floor_passed,
                "draft_n": result.get("draft_n"),
                "draft_n_accepted": result.get("draft_n_accepted"),
                "draft_acceptance_rate": result.get("draft_acceptance_rate"),
                "cleanup": {
                    "pid": cleanup.get("pid"),
                    "pid_dead": cleanup.get("dead") is True,
                    "returncode": cleanup.get("returncode"),
                    "settled": result.get("post_cleanup_vram_settled") is True,
                    "signal": cleanup.get("signal"),
                },
                "records": records,
            }
        )
    base_count = sum(cell["arm"] == BASE_ARM.name for cell in cells)
    dflash_count = sum(cell["arm"] == DFLASH_ARM.name for cell in cells)
    cardinality_valid = bool(cells) and base_count == dflash_count and all(
        cell["prompt_count"] == len(expected_prompt_ids) and cell["prompt_ids"] == expected_prompt_ids for cell in cells
    )
    all_cells_ok = cardinality_valid and all(cell["status"] == "ok" for cell in cells)
    all_finished_stop = all(cell["all_finished_stop"] for cell in cells)
    all_semantic_passed = all(cell["all_semantic_passed"] for cell in cells)
    all_completion_floor_passed = all(cell["all_completion_floor_passed"] for cell in cells)
    cleanup_valid = all(cell["cleanup"]["pid_dead"] and cell["cleanup"]["settled"] for cell in cells)
    dflash_cells = [cell for cell in cells if cell["arm"] == DFLASH_ARM.name]
    draft_n_total = sum(int(cell["draft_n"] or 0) for cell in dflash_cells)
    draft_n_accepted_total = sum(int(cell["draft_n_accepted"] or 0) for cell in dflash_cells)
    status = "ok" if all((all_cells_ok, all_finished_stop, all_semantic_passed, all_completion_floor_passed, cleanup_valid)) else "failed"
    return {
        "schema": CANDIDATE_SMOKE_SCHEMA,
        "status": status,
        "non_gating": True,
        "observation_only": True,
        "reps_per_arm": base_count if base_count == dflash_count else None,
        "expected_prompt_ids": expected_prompt_ids,
        "cardinality_valid": cardinality_valid,
        "all_cells_ok": all_cells_ok,
        "all_finished_stop": all_finished_stop,
        "all_semantic_passed": all_semantic_passed,
        "all_completion_floor_passed": all_completion_floor_passed,
        "cleanup_valid": cleanup_valid,
        "draft_n_total": draft_n_total,
        "draft_n_accepted_total": draft_n_accepted_total,
        "draft_acceptance_rate": (draft_n_accepted_total / draft_n_total) if draft_n_total else None,
        "identity": identity,
        "final_port": final_port,
        "initial_rocm": initial_rocm,
        "final_rocm": final_rocm,
        "cells": cells,
    }


def build_plan(args: argparse.Namespace, model_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    cells = []
    for rep in range(1, args.reps + 1):
        ordered_arms = ARMS if rep % 2 else tuple(reversed(ARMS))
        for arm in ordered_arms:
            cells.append({"arm": arm.name, "rep": rep, "port": args.port_base + len(cells), "prompt_count": len(PROMPTS)})
    return {
        "schema": "epyc.qwen36_27b_q8_dflash_pgpu1.plan.v1", "created_at": utc_now(), "execute": args.execute,
        "production_named_kernel_required": True, "required_branch": EXPECTED_BRANCH, "reps_per_arm": args.reps,
        "arm_order": "paired by rep; base-first on odd reps and dflash-first on even reps",
        "rep_policy": "n >= 5 per arm; n >= 10 is required before making any <=2% claim; this runner does not make a <=2% claim",
        "target_model": model_identity, "drafter_model": args.drafter_identity,
        "fixed_prompt_pack": [{"id": prompt_id, "text": text} for prompt_id, text in PROMPT_SPECS], "prompt_pack_sha256": sha256_text("\n".join(f"{prompt_id}:{text}" for prompt_id, text in PROMPT_SPECS)),
        "seed": args.seed, "max_tokens": args.max_tokens, "min_completion_tokens": args.min_completion_tokens,
        "min_explanation_words": MIN_EXPLANATION_WORDS,
        "target_kv_quant": {"k": TARGET_CACHE_K, "v": TARGET_CACHE_V},
        "drafter_kv_quant": {"k": DRAFTER_CACHE_K, "v": DRAFTER_CACHE_V},
        "provisional_promotion_identity": {"expected_head": args.expected_production_head or "execute_required", "expected_server_sha256": args.expected_server_sha256 or "execute_required"},
        "source_untracked_allowlist": SOURCE_UNTRACKED_ALLOWLIST,
        "warmup_discard_policy": PGPU1_WARMUP_POLICY, "cpu_interference_policy": CPU_INTERFERENCE_POLICY,
        "harness": harness_identity(), "attestation_ref": str(args.attestation_ref) if args.attestation_ref else "execute_required",
        "cells": cells,
    }


def poll_post_cleanup(before_rocm: dict[str, Any], port: int, memory_samples: list[dict[str, Any]]) -> tuple[bool, str]:
    last_reason = "post-cleanup state did not settle"
    for attempt in range(1, 11):
        processes = process_snapshot()
        rocm = collect_rocm_snapshot()
        memory_samples.append({"phase": f"after_cleanup_poll_{attempt}", "rocm": rocm, "processes": processes})
        clean, reason = process_guard_clean(processes, port)
        if snapshot_is_valid(rocm) and clean and vram_settled(before_rocm, rocm):
            return True, "ok"
        last_reason = reason if not clean else "ROCm snapshot failed or VRAM did not settle"
        time.sleep(2)
    return False, last_reason


def run_replicate(args: argparse.Namespace, arm: Arm, rep: int, port: int, output_dir: Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    rep_dir = output_dir / "runs" / f"{arm.name}_rep{rep}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    if expected_identity.get("valid") is not True:
        raise RuntimeError("replicate received an invalid execution identity")
    argv = server_argv(args, arm, port)
    write_json(rep_dir / "server_argv.json", argv)
    write_json(rep_dir / "environment.json", {"exact_server_environment": runtime_env(args.binary), "scrubbed_parent_env_keys": scrubbed_parent_env_keys()})
    write_json(rep_dir / "expected_execution_identity.json", expected_identity)
    write_json(rep_dir / "model_identity_refs.json", expected_identity["binding"]["models"])
    (rep_dir / "prompts.json").write_text(json.dumps([{"id": prompt_id, "text": text} for prompt_id, text in PROMPT_SPECS], indent=2) + "\n", encoding="utf-8")
    before_processes = process_snapshot()
    clean, reason = process_guard_clean(before_processes, port)
    before_rocm = collect_rocm_snapshot()
    memory_samples = [{"phase": "before_launch", "rocm": before_rocm, "processes": before_processes}]
    log_path = rep_dir / "server.stderr"
    proc: subprocess.Popen[str] | None = None
    records: list[dict[str, Any]] = []
    cleanup: dict[str, Any] | None = None
    try:
        if not clean:
            raise RuntimeError(reason)
        if not snapshot_is_valid(before_rocm):
            raise RuntimeError("pre-launch ROCm snapshot incomplete")
        with log_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=stderr, text=True, start_new_session=True, env=runtime_env(args.binary))
        wait_for_health(port, args.startup_timeout, proc)
        resident_rocm = collect_rocm_snapshot(proc.pid)
        resident_processes = process_snapshot()
        resident_valid, resident_reason = live_binding_is_valid(resident_processes, resident_rocm, proc.pid, port, args.binary, expected_identity["binding"], arm.speculative)
        memory_samples.append({"phase": "after_health", "rocm": resident_rocm, "processes": resident_processes, "valid": resident_valid, "reason": resident_reason})
        if not resident_valid:
            raise RuntimeError(f"resident PID/device binding failed: {resident_reason}")
        for index, (prompt_id, prompt) in enumerate(PROMPT_SPECS, 1):
            body = request_body(prompt, args)
            write_json(rep_dir / f"request_{index}.json", body)
            response, elapsed, lifecycle = query_with_live_samples(port, body, args.request_timeout, proc.pid, args.binary, index, expected_identity["binding"], arm.speculative)
            memory_samples.extend(lifecycle["samples"])
            write_json(rep_dir / f"response_{index}.json", response)
            finish_reason = finish_reason_from_response(response)
            content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"response {index} lacks nonempty assistant content")
            reasoning_content = ((response.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content")
            if reasoning_content not in (None, ""):
                raise RuntimeError(f"response {index} violated fixed reasoning-off policy")
            sanity = response_sanity(content)
            if not sanity["passed"]:
                raise RuntimeError(f"response {index} failed anti-garbage check: {sanity}")
            semantics = semantic_validation(prompt_id, content)
            if not semantics["passed"]:
                raise RuntimeError(f"response {index} failed semantic validation: {semantics}")
            row = {"prompt_index": index, "prompt_id": prompt_id, "elapsed_s": elapsed, "request_lifecycle": lifecycle, "finish_reason": finish_reason, "assistant_content": content, "assistant_content_sha256": sha256_text(content), "reasoning_content": reasoning_content, "response_sanity": sanity, "semantic_validation": semantics, **timings_from_response(response, speculative=arm.speculative)}
            if arm.speculative and row["draft_n"] <= 0:
                raise RuntimeError(f"DFlash draft_n must be positive for prompt {index}")
            if row["completion_tokens"] < args.min_completion_tokens:
                raise RuntimeError(f"completion floor failed for prompt {index}: {row['completion_tokens']} < {args.min_completion_tokens}")
            records.append(row)
            after_prompt = collect_live_binding_evidence(proc.pid, port, args.binary, f"after_request_{index}", expected_identity["binding"], arm.speculative)
            memory_samples.append(after_prompt)
            if not after_prompt["valid"]:
                raise RuntimeError(f"post-request PID/device binding failed: {after_prompt['reason']}")
        request_rocm = collect_rocm_snapshot(proc.pid)
        request_processes = process_snapshot()
        request_valid, request_reason = live_binding_is_valid(request_processes, request_rocm, proc.pid, port, args.binary, expected_identity["binding"], arm.speculative)
        memory_samples.append({"phase": "after_all_requests", "rocm": request_rocm, "processes": request_processes, "valid": request_valid, "reason": request_reason})
        if not request_valid:
            raise RuntimeError(f"after-request PID/device binding failed: {request_reason}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        residency = parse_log_residency(log_text, arm)
        if not residency["passed"]:
            raise RuntimeError(f"GPU residency proof failed: {residency}")
        result = {
            "arm": arm.name, "rep": rep, "status": "ok", "prompt_count": len(records),
            "prompt_tps": sum(row["prompt_tokens"] for row in records) / (sum(row["prompt_ms"] for row in records) / 1000.0),
            "decode_tps": sum(row["completion_tokens"] for row in records) / (sum(row["decode_ms"] for row in records) / 1000.0),
            "prompt_ms": sum(row["prompt_ms"] for row in records), "decode_ms": sum(row["decode_ms"] for row in records),
            "completion_tokens": sum(row["completion_tokens"] for row in records),
            "draft_n": sum(row["draft_n"] for row in records) if arm.speculative else 0,
            "draft_n_accepted": sum(row["draft_n_accepted"] for row in records) if arm.speculative else 0,
            "draft_acceptance_rate": (sum(row["draft_n_accepted"] for row in records) / sum(row["draft_n"] for row in records)) if arm.speculative and sum(row["draft_n"] for row in records) else None,
            "records": records, "draft_counters": "required_parsed" if arm.speculative else "not_applicable_zero", "residency": residency,
        }
    except Exception as exc:  # noqa: BLE001 - preserve complete failure evidence
        result = {"arm": arm.name, "rep": rep, "status": "error", "error": repr(exc), "draft_counters": "required_parsed" if arm.speculative else "not_applicable_zero"}
    finally:
        if proc is not None:
            cleanup_rocm = collect_rocm_snapshot(proc.pid)
            memory_samples.append({"phase": "before_cleanup", "rocm": cleanup_rocm})
            cleanup = terminate(proc)
        settled, reason = poll_post_cleanup(before_rocm, port, memory_samples)
        result["post_cleanup_vram_settled"] = settled
        if not cleanup or not cleanup.get("dead") or not settled:
            result["status"] = "cleanup_failed"
            result["cleanup_error"] = reason if settled is False else "server death proof failed"
    result["memory_samples"] = memory_samples
    result["cleanup"] = cleanup
    write_json(rep_dir / "result.json", result)
    return result


def execute(args: argparse.Namespace, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    initial_identity = validate_execution_identity(args, args.target_identity, args.drafter_identity)
    identity = initial_identity["binary"]
    write_json(output_dir / "binary_identity.json", identity)
    valid, reason = initial_identity["production_valid"], initial_identity["production_reason"]
    attestation, attestation_reason = initial_identity["attestation"], initial_identity["attestation_reason"]
    before_processes = process_snapshot()
    clean, interference_reason = process_guard_clean(before_processes)
    initial_rocm = collect_rocm_snapshot()
    hardware_state = collect_hardware_state()
    hardware_valid = hardware_state_is_valid(hardware_state)
    write_json(output_dir / "hardware_state.json", hardware_state)
    harness = initial_identity["harness"]
    harness_ok, harness_reason = initial_identity["harness_valid"], initial_identity["harness_reason"]
    models_ok, models_reason = initial_identity["models_valid"], initial_identity["models_reason"]
    pre_binding = initial_identity["binding"]
    write_json(output_dir / "pre_execution_binding.json", pre_binding)
    guard = {
        "production_identity_valid": valid,
        "reason": reason,
        "expected_production_head": args.expected_production_head,
        "expected_server_sha256": args.expected_server_sha256,
        "source_untracked_allowlist": SOURCE_UNTRACKED_ALLOWLIST,
        "cpu_interference_policy": CPU_INTERFERENCE_POLICY,
        "processes_before": before_processes,
        "initial_rocm": initial_rocm,
        "hardware_state": hardware_state,
        "hardware_valid": hardware_valid,
        "interference_valid": clean,
        "interference_reason": interference_reason,
        "harness": harness,
        "harness_valid": harness_ok,
        "harness_reason": harness_reason,
        "model_identities_valid": models_ok,
        "model_identities_reason": models_reason,
        "attestation_ref": str(args.attestation_ref),
        "attestation": attestation,
        "attestation_valid": attestation is not None,
        "attestation_reason": attestation_reason,
    }
    write_json(output_dir / "guard_state.json", guard)
    if not valid:
        raise RuntimeError(reason)
    if not clean:
        raise RuntimeError(interference_reason)
    if not snapshot_is_valid(initial_rocm):
        raise RuntimeError("initial ROCm state capture failed")
    if not hardware_valid:
        raise RuntimeError("GPU model/gfx target, ROCm runtime/driver, or kernel evidence is incomplete")
    if not harness_ok:
        raise RuntimeError(harness_reason)
    if not models_ok:
        raise RuntimeError(models_reason)
    if attestation is None:
        raise RuntimeError(attestation_reason)
    results: list[dict[str, Any]] = []
    replicate_binding_checks: list[dict[str, Any]] = []
    arms_by_name = {arm.name: arm for arm in ARMS}
    for cell in plan["cells"]:
        arm = arms_by_name[str(cell["arm"])]
        rep = int(cell["rep"])
        rep_dir = output_dir / "runs" / f"{arm.name}_rep{rep}"
        before_identity = validate_execution_identity(args, args.target_identity, args.drafter_identity)
        before_unchanged = before_identity["binding"] == pre_binding
        write_json(rep_dir / "binding_before.json", {**before_identity, "binding_unchanged": before_unchanged})
        if not before_identity["valid"] or not before_unchanged:
            result = {
                "arm": arm.name,
                "rep": rep,
                "status": "error",
                "error": "pre-replicate execution identity is invalid or differs from the initial binding",
                "draft_counters": "required_parsed" if arm.speculative else "not_applicable_zero",
            }
            write_json(rep_dir / "result.json", result)
            results.append(result)
            replicate_binding_checks.append({"arm": arm.name, "rep": rep, "before": before_identity, "after": None, "valid": False})
            break
        try:
            result = run_replicate(args, arm, rep, int(cell["port"]), output_dir, before_identity)
        except Exception as exc:  # noqa: BLE001 - post-replicate identity must still be captured
            result = {
                "arm": arm.name,
                "rep": rep,
                "status": "error",
                "error": repr(exc),
                "draft_counters": "required_parsed" if arm.speculative else "not_applicable_zero",
            }
        after_identity = validate_execution_identity(args, args.target_identity, args.drafter_identity)
        after_unchanged = after_identity["binding"] == before_identity["binding"] == pre_binding
        binding_valid = before_identity["valid"] and after_identity["valid"] and before_unchanged and after_unchanged
        write_json(rep_dir / "binding_after.json", {**after_identity, "binding_unchanged": after_unchanged, "replicate_binding_valid": binding_valid})
        replicate_binding_checks.append({"arm": arm.name, "rep": rep, "before": before_identity, "after": after_identity, "valid": binding_valid})
        if not binding_valid:
            result["status"] = "error"
            result["binding_error"] = "post-replicate production/ref/binary/library/attestation/harness identity changed or became invalid"
        write_json(rep_dir / "result.json", result)
        results.append(result)
        if result.get("status") != "ok":
            break
    cardinality_valid, cardinality_reason = matrix_cardinality_valid(results, args.reps)
    summaries = {arm.name: summarize_arm([row for row in results if row["arm"] == arm.name], arm, args.reps) for arm in ARMS}
    hash_observations: list[dict[str, Any]] = []
    for rep in range(1, args.reps + 1):
        base = next((row for row in results if row.get("arm") == "base" and row.get("rep") == rep and row.get("status") == "ok"), None)
        dflash = next((row for row in results if row.get("arm") == "dflash" and row.get("rep") == rep and row.get("status") == "ok"), None)
        if base and dflash:
            for base_record, dflash_record in zip(base["records"], dflash["records"], strict=True):
                hash_observations.append({"rep": rep, "prompt_index": base_record["prompt_index"], "base_sha256": base_record["assistant_content_sha256"], "dflash_sha256": dflash_record["assistant_content_sha256"], "exact_match": base_record["assistant_content_sha256"] == dflash_record["assistant_content_sha256"]})
    base_median = summaries["base"]["decode_tps"]["median"]
    dflash_median = summaries["dflash"]["decode_tps"]["median"]
    ratio = (float(dflash_median) / float(base_median)) if base_median and dflash_median else None
    if ratio is not None and not math.isfinite(ratio):
        raise RuntimeError("base-vs-DFlash decode throughput ratio is not finite")
    final_processes = process_snapshot()
    final_rocm = collect_rocm_snapshot()
    final_clean, final_reason = process_guard_clean(final_processes)
    if final_clean:
        for cell in plan["cells"]:
            final_clean, final_reason = process_guard_clean(final_processes, int(cell["port"]))
            if not final_clean:
                break
    final_vram_settled = snapshot_is_valid(final_rocm) and vram_settled(initial_rocm, final_rocm)
    post_target = immutable_model_identity(args.target_model)
    post_drafter = immutable_model_identity(args.drafter_model)
    final_identity = validate_execution_identity(args, post_target, post_drafter)
    post_binding = final_identity["binding"]
    binding_unchanged = post_binding == pre_binding
    binding_reason = "ok" if binding_unchanged else "binary/library/model/harness/attestation identity changed during execution"
    post_identity = {
        "binding": post_binding,
        "binding_unchanged": binding_unchanged,
        "binding_reason": binding_reason,
        "production_valid": final_identity["production_valid"],
        "production_reason": final_identity["production_reason"],
        "models_valid": final_identity["models_valid"],
        "models_reason": final_identity["models_reason"],
        "harness_valid": final_identity["harness_valid"],
        "harness_reason": final_identity["harness_reason"],
        "attestation_valid": final_identity["attestation_valid"],
        "attestation_reason": final_identity["attestation_reason"],
    }
    write_json(output_dir / "post_execution_binding.json", post_identity)
    per_replicate_bindings_valid = len(replicate_binding_checks) == len(plan["cells"]) and all(check["valid"] for check in replicate_binding_checks)
    execution_binding_valid = binding_unchanged and final_identity["valid"] and per_replicate_bindings_valid
    final_guard_valid = final_clean and final_vram_settled and execution_binding_valid
    status = "ok" if cardinality_valid and all(summary["all_ok"] for summary in summaries.values()) and final_guard_valid else "failed"
    candidate_smoke = candidate_smoke_projection(
        results,
        initial_rocm=initial_rocm,
        final_rocm=final_rocm,
        identity=pre_binding,
        final_port=int(plan["cells"][-1]["port"]) if plan["cells"] else None,
    )
    write_json(output_dir / "candidate_smoke_summary.json", candidate_smoke)
    return {"schema": "epyc.qwen36_27b_q8_dflash_pgpu1.summary.v1", "created_at": utc_now(), "status": status, "production_named_kernel": True, "required_branch": EXPECTED_BRANCH, "attestation_ref": str(args.attestation_ref), "attestation_sha256": attestation["sha256"], "n": args.reps, "rep_policy": plan["rep_policy"], "median": True, "mad": True, "arm_order": plan["arm_order"], "results": results, "matrix_cardinality_valid": cardinality_valid, "matrix_cardinality_reason": cardinality_reason, "arm_summaries": summaries, "base_vs_dflash": {"decode_tps_ratio": ratio, "direction": "higher_better"}, "cross_arm_hash_observation": {"contract": "distribution-lossless, not byte-exact greedy", "decision_gating": False, "quality_equivalence_claimed": False, "rows": hash_observations}, "warmup_discard_policy": PGPU1_WARMUP_POLICY, "cpu_interference_policy": CPU_INTERFERENCE_POLICY, "hardware_state": hardware_state, "target_kv_quant": {"k": TARGET_CACHE_K, "v": TARGET_CACHE_V}, "drafter_kv_quant": {"k": DRAFTER_CACHE_K, "v": DRAFTER_CACHE_V}, "replicate_binding_checks": replicate_binding_checks, "per_replicate_bindings_valid": per_replicate_bindings_valid, "execution_binding_valid": execution_binding_valid, "post_execution_identity": post_identity, "final_process_guard": final_processes, "final_rocm": final_rocm, "final_clean": final_clean, "final_reason": final_reason, "final_vram_settled": final_vram_settled, "final_guard_valid": final_guard_valid, "post_cleanup_vram_sample": "after_cleanup ROCm snapshots in every replicate", "candidate_smoke_projection": {"path": "candidate_smoke_summary.json", "schema": CANDIDATE_SMOKE_SCHEMA, "non_gating": True}}


def run_audit(output_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pgpu1_artifact_completeness_audit import audit_artifact  # local artifact-only auditor
    report = audit_artifact(output_dir)
    write_json(output_dir / "completeness_audit.json", report)
    return report


def render_operator_run_script(args: argparse.Namespace) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Execute only in the approved P-GPU-1 quiet window using the provisional v9 promotion record.",
        ': "${QWEN36_PGPU1_PROVISIONAL_ATTESTATION_REF:?set provisional promotion attestation reference}"',
        ': "${QWEN36_PGPU1_PROMOTED_HEAD:?set promoted 40-hex production HEAD}"',
        ': "${QWEN36_PGPU1_PROMOTED_SERVER_SHA256:?set promoted 64-hex llama-server SHA256}"',
        "cd " + str(RESEARCH_ROOT),
        "exec " + " ".join([
            "/usr/bin/env", "-i",
            shlex_quote(f"PATH={SAFE_PATH}"), shlex_quote("LANG=C"), shlex_quote("LC_ALL=C"),
            shlex_quote("HIP_VISIBLE_DEVICES=0"), shlex_quote("ROCR_VISIBLE_DEVICES=0"),
            shlex_quote(sys.executable), shlex_quote(str(Path(__file__).resolve())), "--execute",
            "--output-dir", shlex_quote(str(args.output_dir)), "--binary", shlex_quote(str(args.binary)),
            "--source-root", shlex_quote(str(args.source_root)), "--reps", str(args.reps),
            "--context", str(args.context), "--max-tokens", str(args.max_tokens),
            "--min-completion-tokens", str(args.min_completion_tokens), "--seed", str(args.seed),
            "--attestation-ref", '"$QWEN36_PGPU1_PROVISIONAL_ATTESTATION_REF"',
            "--expected-production-head", '"$QWEN36_PGPU1_PROMOTED_HEAD"',
            "--expected-server-sha256", '"$QWEN36_PGPU1_PROMOTED_SERVER_SHA256"',
        ]),
        "",
    ])


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\\"'\\\"'") + "'"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--drafter-model", type=Path, default=DEFAULT_DRAFTER_MODEL)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--min-completion-tokens", type=int, default=DEFAULT_MIN_COMPLETION_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--attestation-ref", type=Path)
    parser.add_argument("--expected-production-head", default="")
    parser.add_argument("--expected-server-sha256", default="")
    args = parser.parse_args(argv)
    if args.reps < DEFAULT_REPS:
        parser.error("P-GPU-1 requires --reps >= 5 per arm")
    if args.target_model != DEFAULT_TARGET_MODEL or args.drafter_model != DEFAULT_DRAFTER_MODEL:
        parser.error("Qwen3.6-27B P-GPU-1 runner uses fixed target and DFlash model paths")
    if args.binary != DEFAULT_BINARY or args.source_root != DEFAULT_SOURCE_ROOT:
        parser.error("Qwen3.6-27B P-GPU-1 runner uses the canonical production HIP binary and source root")
    if (args.context, args.max_tokens, args.min_completion_tokens, args.seed) != (DEFAULT_CONTEXT, DEFAULT_MAX_TOKENS, DEFAULT_MIN_COMPLETION_TOKENS, DEFAULT_SEED):
        parser.error("context, max tokens, completion floor, and seed are fixed protocol constants")
    if args.execute and args.attestation_ref is None:
        parser.error("--execute requires --attestation-ref")
    if args.execute and not re.fullmatch(r"[0-9a-f]{40}", args.expected_production_head):
        parser.error("--execute requires a lowercase 40-hex --expected-production-head")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_server_sha256):
        parser.error("--execute requires a lowercase 64-hex --expected-server-sha256")
    if args.execute:
        attestation, reason = load_promotion_attestation(args.attestation_ref, args.expected_production_head, args.expected_server_sha256)
        if attestation is None:
            parser.error(f"invalid --attestation-ref: {reason}")
        args.attestation_identity = attestation
    else:
        args.attestation_identity = None
    # Large GGUF hashes are deliberately deferred until a live run.  A live run
    # hashes each immutable artifact once and every replicate references it.
    args.target_identity = {"path": str(args.target_model), "bytes": args.target_model.stat().st_size if args.target_model.is_file() else None, "sha256": None}
    args.drafter_identity = {"path": str(args.drafter_model), "bytes": args.drafter_model.stat().st_size if args.drafter_model.is_file() else None, "sha256": None}
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute:
        args.output_dir = args.output_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.execute:
        if not args.target_model.is_file() or not args.drafter_model.is_file():
            failure = {"schema": "epyc.qwen36_27b_q8_dflash_pgpu1.summary.v1", "status": "failed", "production_named_kernel_required": True, "error": "required target or drafter model file is absent"}
            write_json(args.output_dir / "summary.json", failure)
            return 1
        args.target_identity = immutable_model_identity(args.target_model)
        args.drafter_identity = immutable_model_identity(args.drafter_model)
    plan = build_plan(args, args.target_identity)
    write_json(args.output_dir / "plan.json", plan)
    operator_script = args.output_dir / "operator_run.sh"
    operator_script.write_text(render_operator_run_script(args), encoding="utf-8")
    operator_script.chmod(0o755)
    if not args.execute:
        write_json(args.output_dir / "summary.json", {"schema": "epyc.qwen36_27b_q8_dflash_pgpu1.summary.v1", "status": "prepared_no_inference", "production_named_kernel_required": True, "n": args.reps, "median": True, "mad": True, "warmup_discard_policy": PGPU1_WARMUP_POLICY, "cpu_interference_policy": CPU_INTERFERENCE_POLICY, "draft_n_accepted": 0, "post_cleanup_vram_sample": "required on execute", "harness": harness_identity()})
        print(f"prepared: {args.output_dir}")
        return 0
    try:
        summary = execute(args, args.output_dir, plan)
    except Exception as exc:  # noqa: BLE001 - failed gates must leave a durable summary
        summary = {"schema": "epyc.qwen36_27b_q8_dflash_pgpu1.summary.v1", "status": "failed", "production_named_kernel_required": True, "attestation_ref": str(args.attestation_ref), "error": repr(exc), "harness": harness_identity()}
    write_json(args.output_dir / "summary.json", summary)
    audit = run_audit(args.output_dir)
    if summary["status"] != "ok" or audit["status"] != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
