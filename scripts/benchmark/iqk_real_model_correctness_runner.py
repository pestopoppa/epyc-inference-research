#!/usr/bin/env python3
"""Decision-grade B2 candidate-CPU IQK correctness/coherence/safety attestation.

The default mode creates a non-inference plan.  ``--execute`` is deliberately
required because this gate starts six large fresh CPU servers (three models x
GGML_IQK=0/1).  It is not a throughput benchmark and never requires byte-equal
responses: each arm must independently satisfy the same deterministic semantic
contract.  The resulting summary is an externally bindable correctness,
coherence, and numerical-safety attestation for P-BENCH-PREFILL-1.
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
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
BINARY = SOURCE_ROOT / "build-v8-cpu/bin/llama-server"
EXPECTED_BRANCH = "experimental-v8-refresh-20260724"
EXPECTED_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
# Frozen candidate identity, not a caller-supplied assertion.
EXPECTED_SERVER_SHA256 = "c6accc0d5bf935e85c56a16a5f837a70774005876d2326628702325e73b6704b"
EXPECTED_LOCAL_LIBRARY_SHA256 = {
    "libggml-base.so.0.16.0": "f47cc4ad6ab59ea39de7e5fd302f79ba62626cdf84b48332144dfdfa34af0cde",
    "libggml-cpu.so.0.16.0": "26c3c98a289764c11752751faf380b592ab07b00191d289637b4079d2e1a5e90",
    "libggml.so.0.16.0": "ed67a5d9340c256abdcd9b2729871d3dbab6f979031e752bd4c92c2f73a5dacd",
    "libllama-common.so.0.0.10107": "c80f532417d58c52aee40bb25734661b5c7a3f74763c17d0c57921ad83d72bca",
    "libllama-server-impl.so": "ae40403742641096519c0c336e8d2d4bb8c54a8d7404004d940b42d970eb5d31",
    "libllama.so.0.0.10107": "f1e5a2e0976fa4f96d9f78775634b7e4869af8f6549eed384ebc94435eebee8e",
}
LLVM20_LIBDIR = Path("/usr/lib/llvm-20/lib")
OPENMP_RUNTIME = LLVM20_LIBDIR / "libomp.so.5"
EXPECTED_OPENMP_RUNTIME_SHA256 = "98b1f8225260f138243e8e3e7578b83802e998a240f841dc1944a908bf1aee70"
OUTPUT_DIR = RESEARCH_ROOT / "data/kernel-v8-candidate/iqk-real-model-correctness"
CPUSET = "0-95"
THREADS = 96
SEED = 424242
CONTEXT = 4096
MAX_TOKENS = 192
STARTUP_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 900
SETTLE_S = 2
MAX_UPTIME_SECONDS = 7 * 24 * 60 * 60

# The native path added in b8ad9d292.  IQ1_M (29) intentionally remains a
# fallback; Hy3 is included because its IQ2/IQ3 tensors are newly covered.
NATIVE_IQ_TYPE_CODES = {16, 17, 18, 21, 22}
EXPECTED_NATIVE_TYPES_BY_MODEL = {
    "qwen_next_iq2": {21, 22},
    "glm52_ud_iq2": {16, 18, 22},
    "hy3_iq1_m": {16, 18},
}
ACTIVE_RE = re.compile(r"^\[iqk\] ACTIVE: .*?\btype=(\d+)\b", re.MULTILINE)
FATAL_RUNTIME_RE = re.compile(
    r"\b(?:assertion(?:\s+failed)?|fatal(?:\s+error)?|segmentation fault|floating point exception|"
    r"non[- ]?finite|\bnan\b|\binf\b)\b",
    re.IGNORECASE,
)
BENIGN_EOG_LOGIT_BIAS_INF_RE = re.compile(
    r"^(?:(?:\d+)\.\d{2}\.\d{3}\.\d{3} I cmn  )?"
    r"common_init_: added \S+ logit bias = -inf$"
)
SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
LOCAL_LIB_RE = re.compile(r"^\s*(?:libllama|libggml)\S*\s*=>\s*(\S+)", re.MULTILINE)
OPENMP_LIB_RE = re.compile(r"^\s*(lib(?:gomp|omp)\S*)\s*=>\s*(\S+)", re.MULTILINE)
LDD_LINE_RE = re.compile(
    r"^\s*(?P<binding>(?:\S+\s+=>\s+\S+|linux-vdso\.so\.1|/\S+))\s+\(0x[0-9a-f]+\)$"
)
VERSION_BUILD_COMMIT_RE = re.compile(r"^version:\s+\d+\s+\(([0-9a-f]{7,40})\)\s*$", re.MULTILINE)
MEMORY_KIB_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached")
THP_MEMINFO_KEYS = (
    "AnonHugePages", "ShmemHugePages", "ShmemPmdMapped", "FileHugePages", "FilePmdMapped",
    "HugePages_Total", "HugePages_Free", "HugePages_Rsvd", "HugePages_Surp", "Hugepagesize",
    "Hugetlb", "DirectMap2M",
)

BASE_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/home/node",
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "PYTHONNOUSERSITE": "1",
    "LD_LIBRARY_PATH": f"{BINARY.parent}:{LLVM20_LIBDIR}",
    "OMP_NUM_THREADS": str(THREADS),
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false",
    "KMP_BLOCKTIME": "10",
    # CPU-only is both launch intent and an auditable process contract.
    "ROCR_VISIBLE_DEVICES": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "CUDA_VISIBLE_DEVICES": "-1",
}


@dataclass(frozen=True)
class Model:
    name: str
    path: Path


MODELS = (
    Model("qwen_next_iq2", Path("/mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf")),
    Model("glm52_ud_iq2", Path("/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf")),
    Model("hy3_iq1_m", Path("/mnt/raid0/llm/models/hy3-angelslim/Hy3-IQ1_M-mtp.gguf")),
)


class GateFailure(RuntimeError):
    """Failure whose run directory has already received durable evidence."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def stable_file_identity(path: Path) -> dict[str, Any]:
    """Hash a regular file while rejecting replacement/modification races."""
    if not path.is_file():
        raise GateFailure(f"required file is missing: {path}")
    resolved = path.resolve()
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    before_tuple = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_tuple = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_tuple != after_tuple:
        raise GateFailure(f"file identity changed while hashing: {resolved}")
    return {"path": str(resolved), "device": after.st_dev, "inode": after.st_ino,
            "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def child_env(iqk: int) -> dict[str, str]:
    if iqk not in (0, 1):
        raise ValueError("GGML_IQK arm must be exactly 0 or 1")
    return {**BASE_ENV, "GGML_IQK": str(iqk)}


def capture_environment(env: dict[str, str] | None) -> dict[str, str]:
    """Return one of the two exact allowlisted subprocess environments."""
    selected = child_env(0) if env is None else dict(env)
    if selected not in (child_env(0), child_env(1)):
        raise GateFailure(f"subprocess environment is outside the exact allowlist: {selected}")
    return selected


def run_capture(argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    selected_env = capture_environment(env)
    try:
        proc = subprocess.run(argv, text=True, capture_output=True, check=False, env=selected_env, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": argv, "environment": selected_env, "returncode": None, "stdout": "", "stderr": repr(exc), "ok": False}
    return {"argv": argv, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
            "environment": selected_env, "ok": proc.returncode == 0}


def require_ok(capture: dict[str, Any], purpose: str) -> None:
    if not capture.get("ok"):
        raise GateFailure(f"required {purpose} failed: {capture}")


def git_value(*args: str) -> str:
    capture = run_capture(["git", "-C", str(SOURCE_ROOT), *args], env=child_env(0))
    require_ok(capture, "git " + " ".join(args))
    return str(capture["stdout"]).strip()


def resolve_version_build_commit(version: dict[str, Any]) -> dict[str, str]:
    """Resolve the llama-server version's abbreviated build commit fail-closed."""
    output = f"{version.get('stdout', '')}\n{version.get('stderr', '')}"
    version_lines = re.findall(r"^version:.*$", output, re.MULTILINE)
    if not version_lines:
        raise GateFailure("candidate server version does not contain a build commit")
    if len(version_lines) != 1:
        raise GateFailure("candidate server version contains an ambiguous build commit witness")
    match = VERSION_BUILD_COMMIT_RE.fullmatch(version_lines[0])
    if match is None:
        raise GateFailure("candidate server version build commit is malformed")
    abbreviated = match.group(1)
    resolved = git_value("rev-parse", "--verify", f"{abbreviated}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise GateFailure("candidate server version build commit did not resolve to a full commit ID")
    if resolved != EXPECTED_HEAD:
        raise GateFailure("candidate server version build commit does not resolve to the expected HEAD")
    return {"abbreviated": abbreviated, "resolved": resolved}


def normalize_ldd_evidence(ldd: dict[str, Any]) -> dict[str, Any]:
    """Remove only ASLR addresses after strictly validating every ldd binding line."""
    stdout = ldd.get("stdout")
    if not isinstance(stdout, str) or not stdout:
        raise GateFailure("ldd evidence lacks nonempty stdout")
    bindings: list[str] = []
    for line in stdout.splitlines():
        match = LDD_LINE_RE.fullmatch(line)
        if match is None:
            raise GateFailure(f"ldd contains malformed or unrecognized binding line: {line!r}")
        bindings.append(match.group("binding"))
    if not bindings:
        raise GateFailure("ldd evidence lacks binding lines")
    return {**ldd, "stdout": "\n".join(bindings)}


def local_library_identities(binary: Path) -> dict[str, Any]:
    ldd = run_capture(["ldd", str(binary)], env=child_env(0))
    require_ok(ldd, "ldd candidate server")
    ldd = normalize_ldd_evidence(ldd)
    targets: list[Path] = []
    for target_text in LOCAL_LIB_RE.findall(str(ldd["stdout"])):
        target = Path(target_text)
        if target_text == "not" or not target.is_file() or target.resolve().parent != binary.parent.resolve():
            raise GateFailure(f"candidate local llama/ggml library is not resolved from build-v8-cpu/bin: {target_text}")
        targets.append(target.resolve())
    if not targets or len(set(targets)) != len(targets):
        raise GateFailure(f"ldd does not give a unique complete local llama/ggml set: {ldd}")
    libraries = [stable_file_identity(item) for item in sorted(targets)]
    actual = {Path(item["path"]).name: item["sha256"] for item in libraries}
    if actual != EXPECTED_LOCAL_LIBRARY_SHA256:
        raise GateFailure(
            "candidate local llama/ggml library filename/SHA set does not match frozen v8 candidate: "
            f"actual={actual} expected={EXPECTED_LOCAL_LIBRARY_SHA256}"
        )
    openmp_matches = OPENMP_LIB_RE.findall(str(ldd["stdout"]))
    if len(openmp_matches) != 1:
        raise GateFailure(f"ldd must resolve exactly one OpenMP runtime: {ldd}")
    soname, target_text = openmp_matches[0]
    target = Path(target_text)
    resolved = target.resolve()
    expected_dir = LLVM20_LIBDIR.resolve()
    if (target_text == "not" or target.parent.resolve() != expected_dir
            or resolved.parent != expected_dir or resolved.name != OPENMP_RUNTIME.name):
        raise GateFailure(
            "candidate OpenMP runtime must resolve to LLVM 20 libomp.so.5: "
            f"soname={soname} target={target_text} resolved={resolved}"
        )
    runtime = stable_file_identity(resolved)
    if runtime["sha256"] != EXPECTED_OPENMP_RUNTIME_SHA256:
        raise GateFailure("LLVM 20 OpenMP runtime SHA256 does not match the frozen candidate runtime")
    return {"ldd": ldd, "libraries": libraries, "filename_sha256": actual,
            "openmp_runtime": {"soname": soname, **runtime}}


def discover_shards(entry: Path) -> list[Path]:
    if not entry.is_file():
        raise GateFailure(f"model entry missing: {entry}")
    match = SHARD_RE.match(entry.name)
    if not match:
        return [entry]
    count, index = int(match.group("count")), int(match.group("index"))
    if index != 1 or count < 2:
        raise GateFailure(f"split model entry must be shard 00001: {entry}")
    paths = [entry.parent / f"{match.group('prefix')}-{part:05d}-of-{count:05d}.gguf" for part in range(1, count + 1)]
    missing = [str(item) for item in paths if not item.is_file()]
    if missing:
        raise GateFailure(f"split model is incomplete: {missing}")
    return paths


def model_identity(model: Model) -> dict[str, Any]:
    shards = [stable_file_identity(shard) for shard in discover_shards(model.path)]
    return {"name": model.name, "entry_path": str(model.path), "shards": shards,
            "shard_count": len(shards), "total_bytes": sum(int(item["bytes"]) for item in shards)}


def execution_identity() -> dict[str, Any]:
    if git_value("branch", "--show-current") != EXPECTED_BRANCH or git_value("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise GateFailure("candidate branch or HEAD does not match the frozen v8 candidate")
    if git_value("status", "--porcelain=v1"):
        raise GateFailure("candidate source must be completely clean before IQK gate")
    binary = stable_file_identity(BINARY)
    if binary["sha256"] != EXPECTED_SERVER_SHA256:
        raise GateFailure("candidate llama-server SHA256 does not match the frozen candidate")
    version = run_capture([str(BINARY), "--version"], env=child_env(0))
    require_ok(version, "candidate llama-server --version")
    version_build_commit = resolve_version_build_commit(version)
    return {
        "candidate": {"source_root": str(SOURCE_ROOT), "branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD,
                      "source_clean": True, "version": version, "version_build_commit": version_build_commit, "binary": binary,
                      "local_libraries": local_library_identities(BINARY)},
        "runner": stable_file_identity(Path(__file__).resolve()),
        "models": {model.name: model_identity(model) for model in MODELS},
        "environment_arms": {str(iqk): child_env(iqk) for iqk in (0, 1)},
    }


def process_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            comm = (proc / "comm").read_text(encoding="utf-8").strip()
            argv0 = (proc / "cmdline").read_bytes().split(b"\0", 1)[0].decode("utf-8", errors="replace")
        except FileNotFoundError:
            continue
        except OSError as exc:
            # Uninspectable, potentially relevant processes invalidate a quiet CPU claim.
            rows.append({"pid": int(proc.name), "comm": None, "argv0": None, "unreadable": repr(exc)})
            continue
        if comm in {"llama-server", "llama-cli", "llama-bench"} or "llama" in Path(argv0).name:
            try:
                exe = str((proc / "exe").resolve())
            except OSError:
                exe = None
            rows.append({"pid": int(proc.name), "comm": comm, "argv0": argv0, "exe": exe})
    return rows


def kfd_ownership() -> dict[str, Any]:
    """Resolve /dev/kfd ownership, with lsof required on /proc permission loss."""
    owners: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            fds = list((proc / "fd").iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            unreadable.append({"pid": int(proc.name), "reason": repr(exc)})
            continue
        for fd in fds:
            try:
                if os.readlink(fd) == "/dev/kfd":
                    owners.append({"pid": int(proc.name), "fd": fd.name})
            except FileNotFoundError:
                continue
            except OSError as exc:
                unreadable.append({"pid": int(proc.name), "fd": fd.name, "reason": repr(exc)})
    fallback = None
    if unreadable:
        executable = shutil.which("lsof")
        if executable is None:
            raise GateFailure("/proc KFD ownership is unreadable and lsof fallback is unavailable")
        fallback = run_capture([executable, "-t", "/dev/kfd"], env=child_env(0))
        if fallback["returncode"] not in (0, 1):
            raise GateFailure(f"/proc KFD ownership is unreadable and lsof fallback failed: {fallback}")
        for line in str(fallback["stdout"]).splitlines():
            try:
                pid = int(line.strip())
            except ValueError as exc:
                raise GateFailure(f"unparseable lsof KFD owner: {line!r}") from exc
            if pid != os.getpid() and not any(item["pid"] == pid for item in owners):
                owners.append({"pid": pid, "fd": "lsof:/dev/kfd"})
    return {"users": owners, "unreadable_processes": unreadable, "lsof_fallback": fallback}


def autopilot_processes() -> list[dict[str, Any]]:
    """Fail closed if an AutoPilot process could contend with this CPU gate."""
    owners: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            comm = (proc / "comm").read_text(encoding="utf-8").strip()
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise GateFailure(f"cannot inspect potential autopilot ownership for pid {proc.name}: {exc}") from exc
        if "autopilot" in comm.lower() or "autopilot" in cmdline.lower():
            owners.append({"pid": int(proc.name), "comm": comm, "cmdline": cmdline})
    return owners


def _read_required(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise GateFailure(f"required host-state file is unreadable: {path}: {exc}") from exc


def _active_thp(value: str) -> str | None:
    match = re.search(r"\[(\w+)\]", value)
    return match.group(1) if match else None


def parse_memory_state(text: str) -> dict[str, Any]:
    """Parse the full THP/hugepage pool contract with explicit units."""
    raw_rows: dict[str, tuple[int, str]] = {}
    required = set(MEMORY_KIB_KEYS).union(THP_MEMINFO_KEYS)
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if key not in required:
            continue
        if not separator or key in raw_rows:
            raise GateFailure(f"malformed or duplicate required meminfo field: {line!r}")
        match = re.fullmatch(r"\s*(\d+)(?:\s+(kB))?\s*", value)
        if match is None:
            raise GateFailure(f"malformed required meminfo field: {line!r}")
        numeric, unit = int(match.group(1)), match.group(2)
        count_field = key.startswith("HugePages_")
        if (count_field and unit is not None) or (not count_field and unit != "kB"):
            raise GateFailure(f"unexpected meminfo unit for {key}: {unit!r}")
        raw_rows[key] = (numeric, unit or "count")
    missing = sorted(required.difference(raw_rows))
    if missing:
        raise GateFailure(f"missing required THP/hugepage meminfo fields: {missing}")
    memory_kib = {key: raw_rows[key][0] for key in MEMORY_KIB_KEYS}
    thp_meminfo = {key: {"value": raw_rows[key][0], "unit": raw_rows[key][1]} for key in THP_MEMINFO_KEYS}
    total = memory_kib["MemTotal"]
    if total <= 0 or any(value < 0 or value > total for value in memory_kib.values() if value != total):
        raise GateFailure(f"base meminfo values are inconsistent: {memory_kib}")
    if thp_meminfo["HugePages_Free"]["value"] > thp_meminfo["HugePages_Total"]["value"]:
        raise GateFailure("HugePages_Free exceeds HugePages_Total")
    return {"memory_kib": memory_kib, "thp_meminfo": thp_meminfo, "meminfo_raw": text}


def host_snapshot() -> dict[str, Any]:
    governors = {
        str(path): _read_required(path)
        for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
    }
    memory = parse_memory_state(_read_required(Path("/proc/meminfo")))
    return {
        "captured_at": utc_now(),
        "uptime_seconds": float(_read_required(Path("/proc/uptime")).split()[0]),
        "governors": governors,
        "thp_enabled": {"raw": _read_required(Path("/sys/kernel/mm/transparent_hugepage/enabled"))},
        "thp_defrag": {"raw": _read_required(Path("/sys/kernel/mm/transparent_hugepage/defrag"))},
        "numa_balancing": _read_required(Path("/proc/sys/kernel/numa_balancing")),
        **memory,
        "llama_ownership": process_rows(),
        "autopilot_processes": autopilot_processes(),
        "kfd_ownership": kfd_ownership(),
    }


def require_host_state(snapshot: dict[str, Any], *, expected_llama_pid: int | None) -> None:
    if snapshot["uptime_seconds"] > MAX_UPTIME_SECONDS:
        raise GateFailure(f"host uptime exceeds one week: {snapshot['uptime_seconds']}")
    governors = snapshot["governors"]
    if not governors or any(value != "performance" for value in governors.values()):
        raise GateFailure("not every CPU governor is performance")
    if _active_thp(snapshot["thp_enabled"]["raw"]) != "always":
        raise GateFailure("THP enabled mode is not always")
    if _active_thp(snapshot["thp_defrag"]["raw"]) != "always":
        raise GateFailure("THP defrag mode is not always")
    if snapshot["numa_balancing"] != "0":
        raise GateFailure("kernel.numa_balancing is not 0")
    memory = snapshot.get("memory_kib")
    thp_meminfo = snapshot.get("thp_meminfo")
    raw_meminfo = snapshot.get("meminfo_raw")
    if not isinstance(memory, dict) or not isinstance(thp_meminfo, dict) or not isinstance(raw_meminfo, str):
        raise GateFailure("full THP/hugepage memory evidence is absent")
    # Re-parse the raw capture and compare it exactly, preventing a caller from
    # asserting a reduced/edited memory view in a synthetic snapshot.
    parsed_memory = parse_memory_state(raw_meminfo)
    if parsed_memory["memory_kib"] != memory or parsed_memory["thp_meminfo"] != thp_meminfo:
        raise GateFailure("THP/hugepage memory evidence does not match raw meminfo")
    if snapshot["autopilot_processes"]:
        raise GateFailure(f"autopilot is active during CPU IQK gate: {snapshot['autopilot_processes']}")
    kfd = snapshot["kfd_ownership"]
    if kfd["unreadable_processes"] and (not kfd["lsof_fallback"] or kfd["lsof_fallback"]["returncode"] not in (0, 1)):
        raise GateFailure("KFD ownership is unresolved through both /proc and lsof")
    if kfd["users"]:
        raise GateFailure(f"CPU-only run has /dev/kfd owner(s): {kfd['users']}")
    rows = snapshot["llama_ownership"]
    if expected_llama_pid is None:
        if rows:
            raise GateFailure(f"CPU IQK gate requires no llama processes: {rows}")
    else:
        expected = [row for row in rows if row.get("pid") == expected_llama_pid and row.get("exe") == str(BINARY.resolve())]
        if len(expected) != 1 or len(rows) != 1:
            raise GateFailure(f"CPU-only run has unexpected llama ownership: {rows}")


def parse_nodes(text: str) -> list[int]:
    match = re.search(r"available:\s*(\d+)\s+nodes\s*\(([^)]+)\)", text)
    if not match:
        raise GateFailure(f"cannot parse numactl hardware nodes: {text!r}")
    nodes: list[int] = []
    for piece in match.group(2).split(","):
        if "-" in piece:
            left, right = (int(value) for value in piece.split("-", 1))
            nodes.extend(range(left, right + 1))
        else:
            nodes.append(int(piece))
    if len(nodes) != int(match.group(1)) or nodes != list(range(nodes[-1] + 1)):
        raise GateFailure(f"unexpected noncontiguous NUMA layout: {nodes}")
    return nodes


def exact_process_env(pid: int, iqk: int, proc_root: Path = Path("/proc")) -> dict[str, str]:
    try:
        entries = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    except OSError as exc:
        raise GateFailure(f"cannot read child environment: {exc}") from exc
    actual: dict[str, str] = {}
    for item in filter(None, entries):
        if b"=" not in item:
            raise GateFailure("malformed child environment")
        key, value = item.split(b"=", 1)
        key_text = key.decode()
        if key_text in actual:
            raise GateFailure(f"duplicate child environment variable: {key_text}")
        actual[key_text] = value.decode()
    expected = child_env(iqk)
    if actual != expected:
        raise GateFailure(f"child environment is not the exact sanitized allowlist: actual={actual} expected={expected}")
    return actual


def mapped_openmp_runtime(pid: int, runtime: dict[str, Any], proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Prove the launched server maps the same pinned LLVM 20 runtime."""
    try:
        maps = (proc_root / str(pid) / "maps").read_text(encoding="utf-8")
    except OSError as exc:
        raise GateFailure(f"cannot read child runtime maps: {exc}") from exc
    expected = str(Path(str(runtime["path"])).resolve())
    mapped = {line.rsplit(maxsplit=1)[-1] for line in maps.splitlines() if len(line.split()) >= 6}
    if expected not in mapped:
        raise GateFailure(f"child does not map the pinned LLVM 20 OpenMP runtime: {expected}")
    return {"path": expected, "sha256": runtime["sha256"], "maps": maps}


def validate_numastat_totals(values: list[float], nodes: list[int]) -> None:
    if len(values) != len(nodes) + 1:
        raise GateFailure(f"numastat Total row has wrong column count: {values}")
    node_values, total = values[:-1], values[-1]
    if any(not math.isfinite(value) or value <= 0 for value in node_values) or not math.isfinite(total) or total <= 0:
        raise GateFailure(f"numastat lacks finite positive residency on all NUMA nodes: {values}")
    if abs(sum(node_values) - total) > max(0.1, 0.01 * len(nodes)):
        raise GateFailure(f"numastat node sum does not match Total: {values}")


def parse_cpu_list(value: str) -> set[int]:
    """Parse Linux Cpus_allowed_list syntax without accepting ambiguous input."""
    if not value:
        raise GateFailure("empty Cpus_allowed_list")
    cpus: set[int] = set()
    for piece in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", piece):
            raise GateFailure(f"malformed Cpus_allowed_list: {value!r}")
        if "-" in piece:
            left, right = (int(item) for item in piece.split("-", 1))
            if left > right:
                raise GateFailure(f"malformed descending CPU range: {value!r}")
            cpus.update(range(left, right + 1))
        else:
            cpus.add(int(piece))
    return cpus


def thread_affinity_evidence(
    pid: int,
    proc_root: Path = Path("/proc"),
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Capture a stable, fail-closed affinity witness for every server thread."""
    expected = parse_cpu_list(CPUSET)
    task_dir = proc_root / str(pid) / "task"
    for attempt in range(1, max_attempts + 1):
        try:
            before = sorted(path.name for path in task_dir.iterdir() if path.name.isdecimal())
        except OSError as exc:
            raise GateFailure(f"cannot list child threads: {exc}") from exc
        if not before:
            raise GateFailure("child has no observable threads")
        threads: list[dict[str, Any]] = []
        churned = False
        for tid in before:
            try:
                status = (task_dir / tid / "status").read_text(encoding="utf-8")
            except FileNotFoundError:
                churned = True
                break
            except OSError as exc:
                raise GateFailure(f"cannot read child thread status for TID {tid}: {exc}") from exc
            matches = re.findall(r"^Cpus_allowed_list:\s*(\S*)\s*$", status, re.MULTILINE)
            if len(matches) != 1:
                raise GateFailure(f"TID {tid} lacks exactly one Cpus_allowed_list")
            allowed_list = matches[0]
            allowed = parse_cpu_list(allowed_list)
            if not allowed <= expected:
                raise GateFailure(f"TID {tid} affinity escapes required CPU set {CPUSET}: {allowed_list}")
            threads.append({"tid": int(tid), "cpus_allowed_list": allowed_list, "cpus": sorted(allowed)})
        try:
            after = sorted(path.name for path in task_dir.iterdir() if path.name.isdecimal())
        except OSError as exc:
            raise GateFailure(f"cannot relist child threads: {exc}") from exc
        if churned or before != after:
            continue
        union = set().union(*(set(thread["cpus"]) for thread in threads))
        if union != expected:
            raise GateFailure(
                f"thread affinity union does not exactly cover required CPU set {CPUSET}: {sorted(union)}"
            )
        return {
            "status": "pass", "attempt": attempt, "expected_cpus_allowed_list": CPUSET,
            "threads": threads, "union_cpus": sorted(union),
        }
    raise GateFailure(f"child thread list did not stabilize within {max_attempts} snapshots")


def live_cpu_evidence(pid: int, iqk: int, openmp_runtime: dict[str, Any]) -> dict[str, Any]:
    host = host_snapshot()
    require_host_state(host, expected_llama_pid=pid)
    expected_nodes_capture = run_capture(["numactl", "--hardware"], env=child_env(iqk))
    require_ok(expected_nodes_capture, "numactl hardware")
    nodes = parse_nodes(str(expected_nodes_capture["stdout"]))
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        numa_maps = Path(f"/proc/{pid}/numa_maps").read_text(encoding="utf-8")
    except OSError as exc:
        raise GateFailure(f"cannot read child placement state: {exc}") from exc
    affinity = thread_affinity_evidence(pid)
    policies = {line.split()[1] for line in numa_maps.splitlines() if len(line.split()) >= 2}
    expected_policy = f"interleave:0-{nodes[-1]}"
    if policies != {expected_policy}:
        raise GateFailure(f"child numa_maps does not prove all-node interleave: {policies}")
    numastat = run_capture(["numastat", "-p", str(pid)], env=child_env(iqk))
    require_ok(numastat, "live NUMA residency")
    header = next((line for line in str(numastat["stdout"]).splitlines() if "Total" in line and "Node" in line), "")
    actual_nodes = [int(value) for value in re.findall(r"Node\s+(\d+)", header)]
    total_line = next((line.split() for line in str(numastat["stdout"]).splitlines() if line.split()[:1] == ["Total"]), [])
    if actual_nodes != nodes or len(total_line) != len(nodes) + 2:
        raise GateFailure(f"numastat cannot prove all-node residency: {numastat}")
    try:
        values = [float(value) for value in total_line[1:]]
    except ValueError as exc:
        raise GateFailure("numastat total row is not numeric") from exc
    validate_numastat_totals(values, nodes)
    return {"status": "pass", "captured_at": utc_now(), "pid": pid, "environment": exact_process_env(pid, iqk),
            "openmp_runtime": mapped_openmp_runtime(pid, openmp_runtime),
            "host": host, "nodes": nodes, "thread_affinity": affinity,
            "status_file": status, "numa_maps": numa_maps, "numastat": numastat,
            "numastat_total_mib": values}


def assert_clean_host() -> dict[str, Any]:
    snapshot = host_snapshot()
    require_host_state(snapshot, expected_llama_pid=None)
    return {**snapshot, "status": "pass"}


def server_argv(model: Model, iqk: int, port: int) -> list[str]:
    argv = ["taskset", "-c", CPUSET, "numactl", "--interleave=all", str(BINARY), "-m", str(model.path),
            "--host", "127.0.0.1", "--port", str(port), "-c", str(CONTEXT), "-t", str(THREADS),
            "-tb", str(THREADS), "-fa", "on", "-dev", "none", "-ngl", "0", "--no-op-offload", "--no-mmap",
            "--seed", str(SEED), "--temp", "0", "--top-k", "1", "--top-p", "1",
            "--reasoning", "off", "--reasoning-budget", "0", "--jinja"]
    validate_cpu_argv(argv)
    return argv


def validate_cpu_argv(argv: list[str]) -> None:
    if "--no-mmap" not in argv or "--mmap" in argv:
        raise GateFailure("CPU IQK recipe must use --no-mmap and must not contain --mmap")
    for prohibited in ("--gpu-layers", "--gpu-layers-draft", "--device"):
        if prohibited in argv:
            raise GateFailure(f"CPU IQK recipe contains conflicting GPU placement flag: {prohibited}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(port: int) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:  # final timeout has the useful signal
            pass
        time.sleep(1)
    raise GateFailure("server health check timed out")


def tasks_for(model: Model) -> list[dict[str, str]]:
    return [
        # This exact payload is already a passing Hy3 CPU task.  Model identity
        # belongs to the attested server/model artifact, not generated prose.
        {"name": "exact_json", "prompt": "Return exactly this JSON object and no other text: {\"status\":\"ok\",\"model\":\"hy3\"}."},
        {"name": "math_37_plus_58", "prompt": "What is 37 + 58? Return only the decimal integer."},
        {"name": "needle", "prompt": "The required audit needle is IQK-DELTA-9421. Return that exact token once and no other text."},
        {"name": "routing_tradeoffs", "prompt": "In 45 to 110 words, explain tradeoffs among compute cost, bandwidth, load balancing, and routing overhead in a mixture-of-experts inference router. Use all four named concepts and give no headings."},
    ]


def request_body(prompt: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS, "seed": SEED,
            "temperature": 0, "top_k": 1, "top_p": 1, "stream": False, "logprobs": True}


def query(port: int, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise GateFailure(f"completion HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    if not isinstance(value, dict):
        raise GateFailure("completion response is not a JSON object")
    return value


def finite_positive(value: Any, name: str, *, zero_ok: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateFailure(f"{name} is nonnumeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not zero_ok):
        raise GateFailure(f"{name} is nonfinite or invalid")
    return number


def finite_native_number(value: Any, name: str, *, zero_ok: bool = True) -> int | float:
    if type(value) not in (int, float):
        raise GateFailure(f"{name} is not a strict native JSON number")
    number = float(value)
    if not math.isfinite(number) or (not zero_ok and number == 0):
        raise GateFailure(f"{name} is nonfinite or invalid")
    return value


def logprob_evidence(choice: dict[str, Any]) -> dict[str, Any]:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict) or not isinstance(logprobs.get("content"), list) or not logprobs["content"]:
        raise GateFailure("completion lacks nonempty per-token logprobs")
    empty_indices = [
        index for index, item in enumerate(logprobs["content"])
        if isinstance(item, dict) and item.get("token") == ""
    ]
    tokens: list[dict[str, Any]] = []
    for index, item in enumerate(logprobs["content"]):
        if not isinstance(item, dict) or not isinstance(item.get("token"), str):
            raise GateFailure(f"logprobs.content[{index}] lacks a returned completion token")
        byte_values = item.get("bytes")
        if not isinstance(byte_values, list) or any(type(value) is not int or not 0 <= value <= 255 for value in byte_values):
            raise GateFailure(f"logprobs.content[{index}].bytes is not a JSON byte list")
        token = item["token"]
        terminal_eog_empty = token == ""
        if terminal_eog_empty:
            if empty_indices != [len(logprobs["content"]) - 1] or byte_values != []:
                raise GateFailure("completion may contain exactly one terminal empty EOG token with bytes=[]")
        elif list(token.encode("utf-8")) != byte_values:
            raise GateFailure(f"logprobs.content[{index}].bytes does not match token UTF-8")
        tokens.append({
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "bytes": byte_values,
            "terminal_eog_empty": terminal_eog_empty,
            "logprob": finite_native_number(item.get("logprob"), f"logprobs.content[{index}].logprob"),
        })
    return {"status": "pass", "token_count": len(tokens), "terminal_eog_empty": bool(empty_indices), "tokens": tokens}


def strict_native_integral(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateFailure(f"{name} is not a positive native integral count")
    return value


def flatten_usage_counters(usage: dict[str, Any]) -> dict[str, int | float]:
    """Accept only finite native numeric usage leaves under unambiguous dotted paths."""
    counters: dict[str, int | float] = {}

    def visit(node: dict[str, Any], prefix: str = "") -> None:
        if not node:
            raise GateFailure(f"usage counter object is empty at {prefix or '<root>'}")
        for key, value in node.items():
            if not isinstance(key, str) or not key or "." in key:
                raise GateFailure(f"usage counter key is ambiguous at {prefix or '<root>'}: {key!r}")
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                visit(value, path)
            elif type(value) in (int, float):
                if path in counters:
                    raise GateFailure(f"duplicate flattened usage counter path: {path}")
                counters[path] = finite_native_number(value, f"usage.{path}")
            else:
                raise GateFailure(f"usage.{path} is not a strict native JSON number")

    visit(usage)
    return counters


def content_from_response(response: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise GateFailure("completion lacks first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise GateFailure("completion lacks textual assistant content")
    if str(message.get("reasoning_content") or "").strip():
        raise GateFailure("reasoning-off arm emitted reasoning_content")
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise GateFailure("completion lacks timings")
    prompt_n = strict_native_integral(timings.get("prompt_n"), "timings.prompt_n")
    predicted_n = strict_native_integral(timings.get("predicted_n"), "timings.predicted_n")
    telemetry = {
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prompt_ms": finite_positive(timings.get("prompt_ms"), "timings.prompt_ms"),
        "predicted_ms": finite_positive(timings.get("predicted_ms"), "timings.predicted_ms"),
    }
    for key, value in timings.items():
        finite_native_number(value, f"timings.{key}")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        raise GateFailure("completion usage counters are not an object")
    completion_tokens = strict_native_integral(usage.get("completion_tokens"), "usage.completion_tokens")
    counters = flatten_usage_counters(usage)
    logprobs = logprob_evidence(choices[0])
    if logprobs["token_count"] != predicted_n or logprobs["token_count"] != completion_tokens:
        raise GateFailure(
            "completion logprob token count does not match timings.predicted_n and usage.completion_tokens"
        )
    return message["content"], {"timings": telemetry, "counters": counters}, logprobs


def validate_semantics(task_name: str, content: str, model: Model) -> dict[str, Any]:
    stripped = content.strip()
    if task_name == "exact_json":
        expected = {"status": "ok", "model": "hy3"}
        try:
            actual = json.loads(stripped, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, json.JSONDecodeError) as exc:
            raise GateFailure("exact_json task did not return finite valid JSON") from exc
        if actual != expected:
            raise GateFailure(f"exact_json task mismatch: {actual!r}")
    elif task_name == "math_37_plus_58":
        if stripped != "95":
            raise GateFailure(f"math task mismatch: {stripped!r}")
    elif task_name == "needle":
        if stripped != "IQK-DELTA-9421" or stripped.count("IQK-DELTA-9421") != 1:
            raise GateFailure(f"needle task mismatch: {stripped!r}")
    elif task_name == "routing_tradeoffs":
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", stripped)
        lower = stripped.lower()
        required = ("compute", "bandwidth", "load balancing", "routing", "overhead")
        if not 45 <= len(words) <= 110 or any(token not in lower for token in required):
            raise GateFailure("routing coherence task is out of bounds or omits a required tradeoff")
    else:
        raise ValueError(task_name)
    # Reject degenerate but technically patterned content in all task modes.
    nonspace = [char for char in stripped if not char.isspace()]
    if not nonspace or len(set(nonspace)) < min(4, len(nonspace)):
        raise GateFailure("response fails anti-garbage diversity check")
    return {"task": task_name, "status": "pass", "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def active_log_evidence(log_path: Path, model: Model, iqk: int) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if BENIGN_EOG_LOGIT_BIAS_INF_RE.fullmatch(line):
            continue
        fatal = FATAL_RUNTIME_RE.search(line)
        if fatal:
            raise GateFailure(f"server stderr contains fatal/nonfinite runtime diagnostic: {fatal.group(0)!r}")
    types = [int(value) for value in ACTIVE_RE.findall(text)]
    if iqk == 0:
        if types:
            raise GateFailure(f"GGML_IQK=0 emitted IQK active logs: {types}")
        return {"status": "pass", "iqk": 0, "active_type_codes": []}
    native = sorted(set(types).intersection(NATIVE_IQ_TYPE_CODES))
    required = EXPECTED_NATIVE_TYPES_BY_MODEL[model.name]
    if not required.issubset(native):
        raise GateFailure(
            f"GGML_IQK=1 did not prove all required native IQ types for {model.name}; "
            f"required={sorted(required)} observed={types}"
        )
    return {"status": "pass", "iqk": 1, "active_type_codes": sorted(set(types)), "native_type_codes": native,
            "native_type_names": {"16": "IQ2_XXS", "17": "IQ2_XS", "18": "IQ3_XXS", "21": "IQ3_S", "22": "IQ2_S"}}


def group_members(pgid: int) -> list[int]:
    capture = run_capture(["ps", "-eo", "pid=,pgid="], env=child_env(0))
    require_ok(capture, "process group verification")
    members = []
    for row in str(capture["stdout"]).splitlines():
        pieces = row.split()
        if len(pieces) == 2 and pieces[1] == str(pgid):
            members.append(int(pieces[0]))
    return members


def cleanup_pgrep() -> dict[str, Any]:
    """Require the exact production server name to be absent after cleanup."""
    capture = run_capture(["pgrep", "-x", "llama-server"], env=child_env(0))
    if capture["returncode"] == 1:
        return capture
    if capture["returncode"] == 0:
        raise GateFailure(f"llama-server remains after cleanup: {capture}")
    raise GateFailure(f"cleanup pgrep failed: {capture}")


def cleanup(proc: subprocess.Popen[str], port: int) -> dict[str, Any]:
    pgid = os.getpgid(proc.pid)
    signals: list[str] = []
    if proc.poll() is None:
        signals.append("SIGTERM")
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            signals.append("SIGKILL")
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=30)
    time.sleep(SETTLE_S)
    members = group_members(pgid)
    if members:
        signals.append("SIGKILL-descendants")
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(SETTLE_S)
        members = group_members(pgid)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            port_closed = False
    except OSError:
        port_closed = True
    pgrep = cleanup_pgrep()
    host = assert_clean_host()
    result = {"status": "pass" if proc.poll() is not None and not members and port_closed else "fail",
              "signals": signals, "leader_dead": proc.poll() is not None, "members": members,
              "port_closed": port_closed, "pgrep": pgrep, "host": host}
    if result["status"] != "pass":
        raise GateFailure(f"cleanup proof failed: {result}")
    return result


def run_arm(run_dir: Path, model: Model, iqk: int, openmp_runtime: dict[str, Any]) -> dict[str, Any]:
    arm_dir = run_dir / "arms" / f"{model.name}_iqk{iqk}"
    arm_dir.mkdir(parents=True, exist_ok=False)
    port = find_free_port()
    argv = server_argv(model, iqk, port)
    write_json(arm_dir / "server_argv.json", argv)
    write_json(arm_dir / "expected_environment.json", child_env(iqk))
    write_json(arm_dir / "expected_openmp_runtime.json", openmp_runtime)
    write_json(arm_dir / "tasks.json", tasks_for(model))
    proc: subprocess.Popen[str] | None = None
    rows: list[dict[str, Any]] = []
    primary_error = None
    cleanup_record: dict[str, Any] | None = None
    iqk_log_record: dict[str, Any] | None = None
    numerical_safety: dict[str, Any] | None = None
    identity_before: dict[str, Any] | None = None
    identity_after: dict[str, Any] | None = None
    try:
        identity_before = execution_identity()
        with (arm_dir / "server.stderr").open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=stderr, text=True,
                                    start_new_session=True, env=child_env(iqk))
        wait_for_health(port)
        for index, task in enumerate(tasks_for(model), 1):
            body = request_body(task["prompt"])
            write_json(arm_dir / f"request_{index}.json", body)
            pre = live_cpu_evidence(proc.pid, iqk, openmp_runtime)
            write_json(arm_dir / f"request_{index}_pre_evidence.json", pre)
            response: dict[str, Any] | None = None
            request_error: Exception | None = None
            try:
                response = query(port, body)
            except Exception as exc:  # post-boundary evidence is mandatory even on HTTP failure
                request_error = exc
            post = live_cpu_evidence(proc.pid, iqk, openmp_runtime)
            write_json(arm_dir / f"request_{index}_post_evidence.json", post)
            if request_error:
                raise request_error
            assert response is not None
            write_json(arm_dir / f"response_{index}.json", response)
            content, telemetry, logprobs = content_from_response(response)
            semantic = validate_semantics(task["name"], content, model)
            rows.append({"task": task["name"], "telemetry": telemetry, "semantic": semantic, "logprobs": logprobs})
        iqk_log_record = active_log_evidence(arm_dir / "server.stderr", model, iqk)
        write_json(arm_dir / "iqk_log_evidence.json", iqk_log_record)
        numerical_safety = {
            "status": "pass", "scope": "real-model completion token logprobs and server stderr only",
            "logprob_token_count": sum(int(row["logprobs"]["token_count"]) for row in rows),
            "timings_decision_use": "non-decision observational only",
        }
    except Exception as exc:  # retain all raw evidence, then make summary fail
        primary_error = repr(exc)
    finally:
        if proc is None:
            cleanup_record = {"status": "fail", "error": "server never started"}
        else:
            try:
                cleanup_record = cleanup(proc, port)
            except Exception as exc:
                cleanup_record = {"status": "fail", "error": repr(exc)}
        try:
            iqk_log_record = active_log_evidence(arm_dir / "server.stderr", model, iqk)
            write_json(arm_dir / "iqk_log_evidence.json", iqk_log_record)
        except Exception as exc:
            primary_error = primary_error or repr(exc)
        try:
            identity_after = execution_identity()
            if identity_before != identity_after:
                primary_error = primary_error or "runtime artifact identity changed during arm"
        except Exception as exc:
            primary_error = primary_error or repr(exc)
        result = {"status": "pass" if not primary_error and cleanup_record.get("status") == "pass" and len(rows) == 4 and iqk_log_record and numerical_safety else "fail",
                  "model": model.name, "iqk": iqk, "rows": rows, "primary_error": primary_error,
                  "cleanup": cleanup_record, "iqk_log_evidence": iqk_log_record,
                  "numerical_safety": numerical_safety,
                  "runtime_identity": {"before": identity_before, "after": identity_after}}
        write_json(arm_dir / "result.json", result)
    return result


def plan() -> dict[str, Any]:
    cells = [{"model": model.name, "model_path": str(model.path), "iqk": iqk, "tasks": [task["name"] for task in tasks_for(model)]}
             for model in MODELS for iqk in (0, 1)]
    return {"schema": "epyc.iqk_real_model_correctness.plan.v1", "created_at": utc_now(), "protocol_scope": "B2 correctness/coherence/numerical-safety; non-bit-exact semantic parity", "candidate": {"branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD, "binary": str(BINARY)}, "fixed_cpu_recipe": {"cpuset": CPUSET, "numa": "interleave=all", "threads": THREADS, "device": "none", "ngl": 0, "no_op_offload": True, "seed": SEED, "temperature": 0, "reasoning": "off"}, "excluded": ["qwen3.5-122B", "Laguna UD-IQ2_M CPU"], "cells": cells}


def summarize(results: list[dict[str, Any]], identity: dict[str, Any]) -> dict[str, Any]:
    expected = {(model.name, iqk) for model in MODELS for iqk in (0, 1)}
    actual = {(str(row.get("model")), row.get("iqk")) for row in results}
    passed = actual == expected and len(results) == len(expected) and all(row.get("status") == "pass" for row in results)
    for row in results:
        if len(row.get("rows", [])) != 4:
            passed = False
        log_evidence = row.get("iqk_log_evidence")
        if not isinstance(log_evidence, dict) or log_evidence.get("status") != "pass":
            passed = False
        elif row.get("iqk") == 0:
            if log_evidence.get("active_type_codes") != []:
                passed = False
        elif row.get("iqk") == 1:
            active = log_evidence.get("active_type_codes")
            native = log_evidence.get("native_type_codes")
            required = EXPECTED_NATIVE_TYPES_BY_MODEL.get(str(row.get("model")))
            if (not isinstance(active, list) or not isinstance(native, list) or required is None
                    or not required.issubset(set(native))
                    or not set(native).issubset(NATIVE_IQ_TYPE_CODES) or not set(native).issubset(set(active))):
                passed = False
        else:
            passed = False
        numerical = row.get("numerical_safety")
        if not isinstance(numerical, dict) or numerical.get("status") != "pass" or numerical.get("scope") != "real-model completion token logprobs and server stderr only":
            passed = False
        for task in row.get("rows", []):
            if task.get("semantic", {}).get("status") != "pass":
                passed = False
            if task.get("logprobs", {}).get("status") != "pass" or not task.get("logprobs", {}).get("token_count"):
                passed = False
            telemetry = task.get("telemetry", {})
            for number in telemetry.get("timings", {}).values():
                finite_positive(number, "recorded response telemetry")
            for number in telemetry.get("counters", {}).values():
                finite_native_number(number, "recorded response counter")
    return {"schema": "epyc.iqk_real_model_correctness.attestation.v1", "created_at": utc_now(),
            "status": "pass" if passed else "fail", "attestation_roles": {"correctness": passed, "coherence": passed, "numerical_safety": passed},
            "decision_gate": {"handoff": "iqk-iquant-enablement B2", "b2_gate_passed": passed, "promotion_decision": False,
                              "semantic_contract": "IQK arms are not bit-exact; both independently satisfy fixed tasks",
                              "timings": "non-decision observational only"},
            "identity": identity, "arms": results}


def execute(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    if run_dir.exists():
        raise GateFailure(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        preflight = assert_clean_host()
        write_json(run_dir / "host_preflight.json", preflight)
        before = execution_identity()
        write_json(run_dir / "identity_pre.json", before)
        openmp_runtime = before["candidate"]["local_libraries"]["openmp_runtime"]
        results = [run_arm(run_dir, model, iqk, openmp_runtime) for model in MODELS for iqk in (0, 1)]
        after = execution_identity()
        write_json(run_dir / "identity_post.json", after)
        if before != after:
            raise GateFailure("binary, libraries, runner, model shards, environment, or source provenance changed during gate")
        summary = summarize(results, before)
    except Exception as exc:
        summary = {"schema": "epyc.iqk_real_model_correctness.attestation.v1", "status": "fail", "error": repr(exc),
                   "attestation_roles": {"correctness": False, "coherence": False, "numerical_safety": False}}
    write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), **summary}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the six fresh-server inference arms")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args(argv)
    if args.binary != BINARY or args.source_root != SOURCE_ROOT:
        parser.error("candidate binary and source root are fixed; caller-supplied provenance is forbidden")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", plan())
    if not args.execute:
        write_json(args.output_dir / "summary.json", {"schema": "epyc.iqk_real_model_correctness.attestation.v1", "status": "prepared_no_inference", "attestation_roles": {"correctness": False, "coherence": False, "numerical_safety": False}})
        print(f"prepared: {args.output_dir}")
        return 0
    summary = execute(args.output_dir)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
