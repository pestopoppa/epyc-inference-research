#!/usr/bin/env python3
"""Strict CPU-only, observation-grade Laguna DFlash characterization.

There is no ratified CPU speculative-decoding throughput protocol.  This
harness may establish a lossless server-path observation, never a promotion
gate or a reopening of the March DFlash CPU NO-GO.  Dry-run is the default.
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
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = Path("/mnt/raid0/llm/llama.cpp-experimental")
CANONICAL_BINARY = CANONICAL_SOURCE / "build-v8-cpu/bin/llama-server"
EXPECTED_BRANCH = "experimental-v8-refresh-20260724"
EXPECTED_HEAD = "1977a5d78a5a9c0b1e0050105f8741b7d0a00284"
EXPECTED_SERVER_SHA256 = "d255be4e7735da2b18ff958e6a02400bc1ec86104fa8e8530bb3519fed0e8302"
EXPECTED_LOCAL_LIBRARY_SHA256 = {
    "libggml-base.so.0.16.0": "c289b60b2cc1ddb741f0b9aba0c320d7f56bda8a106e4295ac9af3336a2fc2b4",
    "libggml-cpu.so.0.16.0": "4b1a63e842afa0a57857fcb68c9f3e4906b007526d561122603958c9118c45c5",
    "libggml.so.0.16.0": "ed67a5d9340c256abdcd9b2729871d3dbab6f979031e752bd4c92c2f73a5dacd",
    "libllama-common.so.0.0.10102": "99e9dde64c440091fb2e77b49cb74bf50137a724275f2ffde8299756453a4574",
    "libllama-server-impl.so": "8248f8f1d69d80c0af92483e0a060b34059826d967fbfe88e6f629a61973fa5a",
    "libllama.so.0.0.10102": "293d19eb0bc7c4cde9fbc48d0d38a0a0f9c499f639e302fea36f31035d87f0b8",
}
LLVM20_LIBDIR = Path("/usr/lib/llvm-20/lib")
OPENMP_RUNTIME = LLVM20_LIBDIR / "libomp.so.5"
EXPECTED_OPENMP_RUNTIME_SHA256 = "98b1f8225260f138243e8e3e7578b83802e998a240f841dc1944a908bf1aee70"
Q4_MODEL = Path("/mnt/raid0/llm/models/Laguna-S-2.1-GGUF/laguna-s-2.1-Q4_K_M.gguf")
Q8_MODEL = Path("/dev/shm/laguna-s-2.1-Q8_0.gguf")
DRAFTER_MODEL = Path("/mnt/raid0/llm/models/Laguna-S-2.1-GGUF/laguna-s-2.1-DFlash-BF16.gguf")
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data/cpu-laguna-dflash-observation"
Q8_BYTES = 128_750_823_168
Q8_SHA256 = "d946b221d69f2c5f87a986952bcd3cfb75831e5a6a2184e626e361663e1bfe2b"
Q4_BYTES = 75173103200
Q4_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
DRAFTER_BYTES = 2233764000
DRAFTER_SHA256 = "24614292a4477f3ae5203c3875edcde0bc219f02616a9c9f65791e29b18a67ee"
REPS = 5
CONTEXT = 4096
MAX_TOKENS = 320
MIN_COMPLETION_TOKENS = 96
SEED = 424242
THREADS = 96
CPUSET = "0-95"
STARTUP_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 900
SETTLE_S = 2
MONITOR_INTERVAL_S = 1.0
MIN_MONITORED_INTERVAL_S = 0.25
# Four busy cores is 4.17% of the pinned 96-core team: large enough to exceed
# timer/IRQ noise, small enough to reject contention material to this observation.
MAX_EXTERNAL_CPU_CORES = 4.0
MAX_SWAP_IO_PAGES = 0
WARMUP_MAX_TOKENS = 128
WARMUP_PROMPT = (
    "Warm up the deterministic decode path by writing the integers from 1 through 80 "
    "inclusive in ascending order, separated by commas, with no other text."
)
WARMUP_POLICY = {
    "mode": "fixed_unmeasured_request_per_fresh_server",
    "prompt_sha256": hashlib.sha256(WARMUP_PROMPT.encode()).hexdigest(),
    "max_tokens": WARMUP_MAX_TOKENS,
    "seed": SEED,
    "measured_prompt_order": [1, 2, 3],
}
OMP_ENV = {
    "OMP_NUM_THREADS": "96",
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false",
    "KMP_BLOCKTIME": "10",
}
EXECUTION_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CONTROL_ENV = {
    "PATH": f"{EXECUTION_PATH}:/opt/rocm/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
EXECUTION_ENV = {
    "PATH": EXECUTION_PATH,
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": f"{CANONICAL_BINARY.parent}:{LLVM20_LIBDIR}",
    "GGML_IQK": "1",
    **OMP_ENV,
}
IQK_ACTIVE_RE = re.compile(r"^\[iqk\] ACTIVE:.*?\btype=(\d+)\b.*$", re.MULTILINE)
THP_MEMINFO_KEYS = (
    "AnonHugePages",
    "ShmemHugePages",
    "ShmemPmdMapped",
    "FileHugePages",
    "FilePmdMapped",
    "HugePages_Total",
    "HugePages_Free",
    "HugePages_Rsvd",
    "HugePages_Surp",
    "Hugepagesize",
    "Hugetlb",
    "DirectMap2M",
    "DirectMap1G",
)
PROMPTS = (
    "Determine every prime integer from 10 through 50 inclusive and compute their sum. Explain a reliable method for deciding which integers are prime, show enough intermediate reasoning that the result can be checked, and then include exactly one dedicated line beginning `PRIMES:` followed only by the ascending prime values separated by commas. Include exactly one dedicated line beginning `SUM:` followed only by the integer sum. Other prose may vary, but do not put prose on either result line.",
    'Flatten every scalar in this JSON value: {"z":[3,{"b":false,"a":null}],"a":{"y":"hi","x":[2,1]}}. Visit object keys in ascending lexicographic order and array elements in index order. Treat numbers, strings, booleans, and null as scalars. Explain the traversal in prose, then include exactly one dedicated line beginning `FLAT:` followed only by the resulting valid JSON array. Other prose may vary, but do not put prose on the result line.',
    "Normalize the nonnegative list [0, 2, 3, 5] by dividing each value by the total. Under the required zero-total policy, normalizing [0, 0, 0] returns a same-length all-zero vector. Explain the calculation and why the zero-total branch is necessary. Then include exactly one dedicated line beginning `NORMALIZED:` followed only by the first result as a valid JSON array, and exactly one dedicated line beginning `ZERO_CASE:` followed only by the zero-total result as a valid JSON array. Other prose may vary, but do not put prose on either result line.",
)
SEMANTIC_TASKS = ("prime_list_and_sum", "nested_json_flatten", "normalization_and_zero_policy")
OBSERVATION_POLICY = {
    "decision_grade": False,
    "promotion_gate": False,
    "protocol_id": None,
    "measurement_class": "observation_only_no_ratified_cpu_spec_dec_protocol",
    "march_no_go_reopened": False,
    "acceptance_and_throughput_use": "characterization_only_not_a_promotion_or_no_go_verdict",
    "functional_equality_use": "non_gating_output_stability_observation_only",
    "speculative_semantics": "distribution_lossless_not_byte_exact_greedy",
    "host_window": "warmed_bounded_interference_observation_not_clean_host_claim",
    "external_cpu_ceiling_cores": MAX_EXTERNAL_CPU_CORES,
    "swap_io_page_ceiling": MAX_SWAP_IO_PAGES,
}


@dataclass(frozen=True)
class Arm:
    name: str
    speculative: bool


@dataclass(frozen=True)
class Lane:
    name: str
    model: Path


class RunFailure(RuntimeError):
    def __init__(self, message: str, run_dir: Path | None = None):
        super().__init__(message)
        self.run_dir = run_dir


class LiveEvidenceFailure(RuntimeError):
    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


BASE = Arm("base", False)
DFLASH = Arm("dflash", True)
ARMS = (BASE, DFLASH)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_stamp() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact missing: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    before_tuple = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_tuple = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_tuple != after_tuple:
        raise RuntimeError(f"artifact identity changed while hashing: {path}")
    return {
        "path": str(path),
        "device": after.st_dev,
        "inode": after.st_ino,
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def run_capture(argv: list[str], *, timeout: int = 30, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=dict(CONTROL_ENV) if env is None else env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"argv": argv, "ok": False, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {"argv": argv, "ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def child_env() -> dict[str, str]:
    return dict(EXECUTION_ENV)


def command_required(capture: dict[str, Any], name: str) -> None:
    if not capture.get("ok"):
        raise RuntimeError(f"required state capture failed: {name}: {capture.get('stderr', '')}")


def require_commands() -> dict[str, str]:
    required = ("cat", "free", "git", "grep", "ldd", "lsof", "numactl", "numastat", "ps", "rocm-smi", "taskset")
    missing = [name for name in required if shutil.which(name, path=CONTROL_ENV["PATH"]) is None]
    if missing:
        raise RuntimeError(f"required command(s) unavailable: {missing}")
    return {name: str(shutil.which(name, path=CONTROL_ENV["PATH"])) for name in required}


def parse_process_table(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            rows.append({"pid": int(parts[0]), "comm": parts[1]})
        except ValueError:
            continue
    return rows


def proc_exe_path(pid: int) -> Path:
    return Path(f"/proc/{pid}/exe").resolve()


def exact_llama_processes(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in processes:
        if row["pid"] == os.getpid() or row["comm"] not in {"llama-server", "llama-cli", "llama-bench"}:
            continue
        try:
            exe_path = proc_exe_path(row["pid"])
        except OSError:
            result.append({**row, "exe": None, "reason": "cannot resolve executable ownership"})
            continue
        if exe_path.name == row["comm"]:
            result.append({**row, "exe": str(exe_path), "reason": "exact llama executable"})
    return result


def proc_cmdline(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"cannot read process command line for pid {pid}: {exc}") from exc
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def is_autopilot_process(comm: str, cmdline: str) -> bool:
    return "autopilot" in comm.lower() or "autopilot" in cmdline.lower()


def observed_autopilot_processes(
    processes: list[dict[str, Any]],
    proc_root: Path = Path("/proc"),
) -> list[dict[str, Any]]:
    matches = []
    for row in processes:
        cmdline = proc_cmdline(row["pid"], proc_root)
        if is_autopilot_process(row["comm"], cmdline or ""):
            matches.append({**row, "cmdline": cmdline})
    return matches


def tcp_listeners() -> list[dict[str, Any]]:
    listeners = []
    for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError as exc:
            raise RuntimeError(f"required TCP listener capture failed: {path}: {exc}") from exc
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
                inode = int(fields[9])
            except (IndexError, ValueError):
                raise RuntimeError(f"unparseable TCP listener row: {line}")
            if inode <= 0:
                raise RuntimeError(f"TCP listener has invalid socket inode: {line}")
            listeners.append({"family": path.name, "port": port, "inode": inode, "raw": line})
    return listeners


def kfd_fd_users(proc_root: Path = Path("/proc"), current_pid: int | None = None) -> dict[str, Any]:
    """Return processes holding an exact `/dev/kfd` descriptor.

    Process disappearance between `/proc` enumeration and fd inspection is a
    race and is recorded implicitly by omission.  Any other inspection failure
    is conservative: the preflight must fail rather than declare a clean host.
    """
    current_pid = os.getpid() if current_pid is None else current_pid
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"required /proc enumeration failed: {exc}") from exc
    users = []
    unreadable = []
    for process_dir in process_dirs:
        if not process_dir.name.isdigit() or int(process_dir.name) == current_pid:
            continue
        try:
            fd_dir = process_dir / "fd"
            fd_paths = list(fd_dir.iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            unreadable.append({"pid": int(process_dir.name), "reason": repr(exc)})
            continue
        except OSError as exc:
            if not process_dir.exists():
                continue
            raise RuntimeError(f"failed inspecting process fd ownership: {process_dir}: {exc}") from exc
        for fd_path in fd_paths:
            try:
                target = os.readlink(fd_path)
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                unreadable.append({"pid": int(process_dir.name), "reason": repr(exc)})
                break
            except OSError as exc:
                if not process_dir.exists():
                    break
                raise RuntimeError(f"failed reading process fd target: {fd_path}: {exc}") from exc
            if target != "/dev/kfd":
                continue
            try:
                comm = (process_dir / "comm").read_text(encoding="utf-8").strip()
                exe = str((process_dir / "exe").resolve())
            except FileNotFoundError:
                break
            except OSError as exc:
                if not process_dir.exists():
                    break
                raise RuntimeError(f"failed recording /dev/kfd owner: {process_dir}: {exc}") from exc
            users.append({"pid": int(process_dir.name), "comm": comm, "exe": exe, "fd": fd_path.name})
    lsof = None
    if unreadable:
        lsof = run_capture(["lsof", "-n", "-P", "-Fpcn", "/dev/kfd"], timeout=30)
        # lsof's documented no-match status is 1; anything else is a failed
        # fallback and cannot prove that unreadable processes do not own KFD.
        if lsof.get("returncode") not in (0, 1):
            raise RuntimeError(f"cannot cross-check unreadable /proc fd owners: {lsof}")
        for line in lsof.get("stdout", "").splitlines():
            if not line.startswith("p"):
                continue
            try:
                pid = int(line[1:])
            except ValueError as exc:
                raise RuntimeError(f"unparseable lsof /dev/kfd record: {line}") from exc
            if pid != current_pid and not any(user["pid"] == pid for user in users):
                users.append({"pid": pid, "comm": None, "exe": None, "fd": "lsof:/dev/kfd"})
    return {"users": users, "unreadable_processes": unreadable, "lsof_fallback": lsof}


def process_snapshot() -> dict[str, Any]:
    captures = {
        "all_processes": run_capture(["ps", "-eo", "pid=,comm="], timeout=20),
        "rocm_owners": run_capture(["rocm-smi", "--showpidgpus"], timeout=30),
    }
    for name, capture in captures.items():
        command_required(capture, name)
    processes = parse_process_table(captures["all_processes"]["stdout"])
    rocm_text = captures["rocm_owners"]["stdout"].lower()
    captures["exact_llama_processes"] = exact_llama_processes(processes)
    captures["autopilot_processes"] = observed_autopilot_processes(processes)
    captures["tcp_listeners"] = tcp_listeners()
    captures["kfd_fd_snapshot"] = kfd_fd_users()
    captures["kfd_users"] = captures["kfd_fd_snapshot"]["users"]
    captures["kfd_owner"] = bool(captures["kfd_users"])
    captures["rocm_owner"] = "no kfd pids" not in rocm_text and "no process" not in rocm_text
    return captures


def read_required_text(path: Path, name: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"required {name} state is unreadable: {path}: {exc}") from exc
    if not value:
        raise RuntimeError(f"required {name} state is empty: {path}")
    return value


def active_thp_mode(value: str, name: str) -> str:
    matches = re.findall(r"\[([A-Za-z0-9_+-]+)\]", value)
    if len(matches) != 1:
        raise RuntimeError(f"required {name} active mode is malformed: {value!r}")
    return matches[0]


def parse_thp_meminfo(text: str) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        key = line.partition(":")[0]
        if key not in THP_MEMINFO_KEYS:
            continue
        if key in rows:
            raise RuntimeError(f"duplicate THP meminfo field: {key}")
        match = re.fullmatch(rf"{re.escape(key)}:\s+(\d+)(?:\s+(kB))?", line)
        if not match:
            raise RuntimeError(f"malformed THP meminfo field: {line!r}")
        unit = match.group(2)
        count_field = key.startswith("HugePages_")
        if (count_field and unit is not None) or (not count_field and unit != "kB"):
            raise RuntimeError(f"unexpected THP meminfo unit for {key}: {unit!r}")
        rows[key] = {"value": int(match.group(1)), "unit": unit or "count"}
    missing = sorted(set(THP_MEMINFO_KEYS).difference(rows))
    if missing:
        raise RuntimeError(f"missing required THP meminfo fields: {missing}")
    if rows["HugePages_Free"]["value"] > rows["HugePages_Total"]["value"]:
        raise RuntimeError("HugePages_Free exceeds HugePages_Total")
    return rows


def host_tuning_snapshot(
    cpu_root: Path = Path("/sys/devices/system/cpu"),
    thp_root: Path = Path("/sys/kernel/mm/transparent_hugepage"),
    numa_balancing_path: Path = Path("/proc/sys/kernel/numa_balancing"),
    meminfo_path: Path = Path("/proc/meminfo"),
) -> dict[str, Any]:
    online_specification = read_required_text(cpu_root / "online", "online CPU")
    online_cpus = parse_id_set(online_specification, "online CPU")
    if not online_cpus:
        raise RuntimeError("online CPU set is empty")
    governors = {}
    for cpu in online_cpus:
        path = cpu_root / f"cpu{cpu}/cpufreq/scaling_governor"
        governors[str(path)] = read_required_text(path, f"CPU {cpu} scaling governor")
    if any(value != "performance" for value in governors.values()):
        raise RuntimeError(f"not every online CPU scaling governor is performance: {governors}")
    thp_enabled_raw = read_required_text(thp_root / "enabled", "THP enabled")
    thp_defrag_raw = read_required_text(thp_root / "defrag", "THP defrag")
    thp_enabled_active = active_thp_mode(thp_enabled_raw, "THP enabled")
    thp_defrag_active = active_thp_mode(thp_defrag_raw, "THP defrag")
    if thp_enabled_active != "always" or thp_defrag_active != "always":
        raise RuntimeError(
            "THP active modes must both be always: "
            f"enabled={thp_enabled_active!r} defrag={thp_defrag_active!r}"
        )
    numa_balancing = read_required_text(numa_balancing_path, "kernel.numa_balancing")
    if numa_balancing != "0":
        raise RuntimeError(f"kernel.numa_balancing must be 0: {numa_balancing!r}")
    meminfo_raw = read_required_text(meminfo_path, "meminfo")
    return {
        "online_cpu_specification": online_specification,
        "online_cpus": online_cpus,
        "scaling_governors": governors,
        "transparent_hugepage": {
            "enabled": {"raw": thp_enabled_raw, "active": thp_enabled_active},
            "defrag": {"raw": thp_defrag_raw, "active": thp_defrag_active},
        },
        "numa_balancing": numa_balancing,
        "thp_meminfo": parse_thp_meminfo(meminfo_raw),
        "meminfo_raw": meminfo_raw,
    }


def system_snapshot() -> dict[str, Any]:
    captures = {
        "processes": process_snapshot(),
        "numactl_hardware": run_capture(["numactl", "--hardware"], timeout=20),
        "memory": run_capture(["free", "-h"], timeout=20),
        "host_tuning": host_tuning_snapshot(),
    }
    for name, capture in captures.items():
        if name not in {"processes", "host_tuning"}:
            command_required(capture, name)
    return captures


def ensure_quiet_cpu_only(snapshot: dict[str, Any]) -> None:
    processes = snapshot["processes"]
    if processes["exact_llama_processes"] or processes["autopilot_processes"] or processes["kfd_owner"] or processes["rocm_owner"]:
        raise RuntimeError(f"contaminated CPU-only window: {processes}")


def parse_id_set(specification: str, name: str) -> list[int]:
    if not specification or not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", specification):
        raise RuntimeError(f"cannot parse {name} set: {specification!r}")
    values = set()
    for part in specification.split(","):
        bounds = [int(value) for value in part.split("-", 1)]
        start, end = (bounds[0], bounds[0]) if len(bounds) == 1 else bounds
        if start > end:
            raise RuntimeError(f"cannot parse descending {name} range: {part}")
        values.update(range(start, end + 1))
    return sorted(values)


def numactl_available_nodes(capture: dict[str, Any]) -> list[int]:
    command_required(capture, "numactl hardware node set")
    matches = re.findall(r"^available:\s+(\d+)\s+nodes?\s+\(([^)]+)\)\s*$", capture["stdout"], re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"cannot establish exact available NUMA nodes: {capture}")
    declared_count, specification = matches[0]
    nodes = parse_id_set(specification, "NUMA node")
    if int(declared_count) != len(nodes):
        raise RuntimeError(f"numactl node count disagrees with node set: count={declared_count} nodes={nodes}")
    return nodes


def parse_numastat_residency(capture: dict[str, Any], expected_nodes: list[int]) -> dict[str, Any]:
    command_required(capture, "live numastat residency")
    lines = capture["stdout"].splitlines()
    header_rows = [line for line in lines if "Total" in line and re.search(r"\bNode\s+\d+\b", line)]
    if len(header_rows) != 1:
        raise RuntimeError(f"cannot establish numastat node columns: {capture}")
    header_nodes = [int(value) for value in re.findall(r"\bNode\s+(\d+)\b", header_rows[0])]
    if header_nodes != expected_nodes:
        raise RuntimeError(f"numastat nodes do not match all available nodes: expected={expected_nodes} actual={header_nodes}")
    total_rows = [line.split() for line in lines if line.split() and line.split()[0] == "Total"]
    if len(total_rows) != 1 or len(total_rows[0]) != len(expected_nodes) + 2:
        raise RuntimeError(f"cannot establish numastat Total residency row: {capture}")
    try:
        values = [float(value) for value in total_rows[0][1:]]
    except ValueError as exc:
        raise RuntimeError(f"numastat Total row is nonnumeric: {total_rows[0]}") from exc
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError(f"numastat Total row is negative or nonfinite: {values}")
    node_mib, total_mib = values[:-1], values[-1]
    if total_mib <= 0 or any(value <= 0 for value in node_mib):
        raise RuntimeError(f"numastat does not show positive residency on every interleave node: {values}")
    if abs(sum(node_mib) - total_mib) > max(0.1, 0.01 * len(expected_nodes)):
        raise RuntimeError(f"numastat node residency does not sum to Total: {values}")
    return {
        "nodes": expected_nodes,
        "node_mib": {str(node): node_mib[index] for index, node in enumerate(expected_nodes)},
        "total_mib": total_mib,
    }


def target_process_environment(pid: int, proc_root: Path = Path("/proc")) -> dict[str, str]:
    path = proc_root / str(pid) / "environ"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read target process environment for pid {pid}: {exc}") from exc
    entries = [entry for entry in raw.split(b"\0") if entry]
    environment: dict[str, str] = {}
    for entry in entries:
        if b"=" not in entry:
            raise RuntimeError(f"malformed target process environment entry for pid {pid}: {entry!r}")
        key, value = entry.split(b"=", 1)
        decoded_key = key.decode("utf-8")
        if decoded_key in environment:
            raise RuntimeError(f"duplicate target process environment key for pid {pid}: {decoded_key}")
        environment[decoded_key] = value.decode("utf-8")
    expected = child_env()
    if environment != expected:
        raise RuntimeError(f"target process environment differs from exact allowlist: actual={environment} expected={expected}")
    return environment


def identity_stat_tuple(identity: dict[str, Any]) -> tuple[int, int, int]:
    return (identity["device"], identity["inode"], identity["bytes"])


def target_executable_evidence(
    pid: int,
    expected_identity: dict[str, Any],
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    proc_exe = proc_root / str(pid) / "exe"
    try:
        resolved = proc_exe.resolve(strict=True)
        stat_result = proc_exe.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot establish target executable identity for pid {pid}: {exc}") from exc
    expected_path = Path(expected_identity["path"]).resolve(strict=True)
    actual_tuple = (stat_result.st_dev, stat_result.st_ino, stat_result.st_size)
    if resolved != expected_path or actual_tuple != identity_stat_tuple(expected_identity):
        raise RuntimeError(
            "live executable is not the pinned candidate: "
            f"path={resolved} identity={actual_tuple} expected_path={expected_path} "
            f"expected_identity={identity_stat_tuple(expected_identity)}"
        )
    return {
        "proc_exe": str(proc_exe),
        "resolved_path": str(resolved),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "bytes": stat_result.st_size,
        "sha256": expected_identity["sha256"],
    }


def target_mapped_runtime_evidence(
    pid: int,
    runtime_artifacts: dict[str, Any],
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    expected_identities = [
        *runtime_artifacts["local_llama_ggml_libraries"],
        runtime_artifacts["openmp_runtime"],
    ]
    expected_by_path = {
        Path(identity["path"]).resolve(strict=True): identity
        for identity in expected_identities
    }
    maps_path = proc_root / str(pid) / "maps"
    try:
        rows = maps_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read target process maps for pid {pid}: {exc}") from exc
    mapped: dict[Path, dict[str, Any]] = {}
    relevant_runtime_maps = []
    for row in rows:
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        if fields[5].endswith(" (deleted)"):
            raise RuntimeError(f"target process has a deleted mapped file: {fields[5]}")
        path = Path(fields[5])
        name = path.name
        is_openmp = name.startswith(("libgomp.so", "libomp.so"))
        is_llama_ggml = name.startswith(("libllama", "libggml")) and ".so" in name
        if not (is_openmp or is_llama_ggml):
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"cannot establish relevant mapped runtime identity: {path}: {exc}") from exc
        relevant_runtime_maps.append(str(resolved))
        if resolved not in expected_by_path:
            raise RuntimeError(f"target process maps an unpinned OpenMP or llama/ggml runtime: {resolved}")
        stat_result = path.stat()
        expected = expected_by_path[resolved]
        actual_tuple = (stat_result.st_dev, stat_result.st_ino, stat_result.st_size)
        if actual_tuple != identity_stat_tuple(expected):
            raise RuntimeError(
                f"mapped runtime identity changed for {resolved}: "
                f"actual={actual_tuple} expected={identity_stat_tuple(expected)}"
            )
        mapped[resolved] = {
            "path": str(resolved),
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
            "bytes": stat_result.st_size,
            "sha256": expected["sha256"],
        }
    missing = sorted(str(path) for path in set(expected_by_path).difference(mapped))
    if missing:
        raise RuntimeError(f"target process does not map every pinned runtime artifact: {missing}")
    return {
        "maps_path": str(maps_path),
        "mapped_runtime_artifacts": [mapped[path] for path in sorted(mapped)],
        "relevant_runtime_maps": sorted(set(relevant_runtime_maps)),
    }


def target_listener_evidence(
    pid: int,
    port: int,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    matching = [listener for listener in tcp_listeners() if listener["port"] == port]
    if len(matching) != 1:
        raise RuntimeError(f"expected exactly one listener for port {port}: {matching}")
    fd_root = proc_root / str(pid) / "fd"
    try:
        fd_paths = list(fd_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot enumerate target socket ownership for pid {pid}: {exc}") from exc
    owned: list[dict[str, Any]] = []
    for fd_path in fd_paths:
        try:
            target = os.readlink(fd_path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot read target socket fd {fd_path}: {exc}") from exc
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            owned.append({"fd": fd_path.name, "inode": int(match.group(1))})
    listener = matching[0]
    owners = [row for row in owned if row["inode"] == listener["inode"]]
    if len(owners) != 1:
        raise RuntimeError(
            f"listener inode {listener['inode']} on port {port} is not owned exactly once by pid {pid}: {owners}"
        )
    return {"pid": pid, "port": port, "listener": listener, "target_fd": owners[0]}


def parse_proc_stat_row(text: str, pid: int) -> dict[str, Any]:
    closing = text.rfind(")")
    opening = text.find("(")
    if opening <= 0 or closing <= opening:
        raise RuntimeError(f"malformed /proc stat row for pid {pid}")
    try:
        parsed_pid = int(text[:opening].strip())
        comm = text[opening + 1:closing]
        fields = text[closing + 1:].split()
        pgrp = int(fields[2])
        ticks = int(fields[11]) + int(fields[12])
        starttime = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"malformed /proc stat fields for pid {pid}") from exc
    if parsed_pid != pid or ticks < 0 or starttime <= 0:
        raise RuntimeError(f"invalid /proc stat identity for pid {pid}")
    return {"pid": pid, "comm": comm, "pgrp": pgrp, "cpu_ticks": ticks, "starttime": starttime}


def proc_cpu_busy_ticks(proc_root: Path = Path("/proc")) -> int:
    text = read_required_text(proc_root / "stat", "aggregate CPU stat")
    rows = [line for line in text.splitlines() if line.startswith("cpu ")]
    if len(rows) != 1:
        raise RuntimeError("aggregate /proc/stat CPU row is missing or duplicated")
    try:
        values = [int(value) for value in rows[0].split()[1:]]
    except ValueError as exc:
        raise RuntimeError("aggregate /proc/stat CPU row is nonnumeric") from exc
    if len(values) < 8 or any(value < 0 for value in values):
        raise RuntimeError("aggregate /proc/stat CPU row is incomplete or negative")
    return sum(values[index] for index in (0, 1, 2, 5, 6, 7))


def proc_swap_io_pages(proc_root: Path = Path("/proc")) -> dict[str, int]:
    text = read_required_text(proc_root / "vmstat", "VM stat")
    values: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"pswpin", "pswpout"}:
            try:
                values[fields[0]] = int(fields[1])
            except ValueError as exc:
                raise RuntimeError(f"nonnumeric VM stat row: {line}") from exc
    if set(values) != {"pswpin", "pswpout"} or any(value < 0 for value in values.values()):
        raise RuntimeError(f"required swap IO counters are missing or negative: {values}")
    return values


def request_interference_snapshot(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    target_stat_path = proc_root / str(pid) / "stat"
    try:
        target = parse_proc_stat_row(target_stat_path.read_text(encoding="utf-8"), pid)
        process_dirs = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot capture request process state for pid {pid}: {exc}") from exc
    forbidden = []
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        process_pid = int(process_dir.name)
        try:
            comm = (process_dir / "comm").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot read process name during monitored request: {process_dir}: {exc}") from exc
        if process_pid == pid:
            continue
        cmdline = proc_cmdline(process_pid, proc_root)
        if cmdline is None:
            continue
        if comm in {"llama-server", "llama-cli", "llama-bench"} or is_autopilot_process(comm, cmdline):
            forbidden.append({"pid": process_pid, "comm": comm, "cmdline": cmdline})
    return {
        "captured_at": utc_now(),
        "monotonic_s": time.monotonic(),
        "clock_ticks_per_second": int(os.sysconf("SC_CLK_TCK")),
        "aggregate_cpu_busy_ticks": proc_cpu_busy_ticks(proc_root),
        "target": target,
        "swap_io_pages": proc_swap_io_pages(proc_root),
        "forbidden_processes": forbidden,
        "kfd_fd_snapshot": kfd_fd_users(proc_root=proc_root),
    }


def validate_request_monitor_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 2:
        raise RuntimeError("request monitor did not capture both start and end state")
    for sample in samples:
        if sample["forbidden_processes"]:
            raise RuntimeError(
                f"competing inference or AutoPilot process observed during monitored request: "
                f"{sample['forbidden_processes']}"
            )
        if sample["kfd_fd_snapshot"]["users"]:
            raise RuntimeError("KFD owner observed during monitored CPU request")
    intervals = []
    cpu_intervals = 0
    for before, after in zip(samples, samples[1:], strict=False):
        elapsed = after["monotonic_s"] - before["monotonic_s"]
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"request monitor interval is nonpositive or nonfinite: {elapsed}")
        if (
            before["target"]["pid"] != after["target"]["pid"]
            or before["target"]["starttime"] != after["target"]["starttime"]
        ):
            raise RuntimeError("target process identity changed during monitored request")
        hz = before["clock_ticks_per_second"]
        if hz <= 0 or after["clock_ticks_per_second"] != hz:
            raise RuntimeError("request monitor clock tick rate changed or is invalid")
        busy_delta = after["aggregate_cpu_busy_ticks"] - before["aggregate_cpu_busy_ticks"]
        target_delta = after["target"]["cpu_ticks"] - before["target"]["cpu_ticks"]
        if busy_delta < 0 or target_delta < 0:
            raise RuntimeError(
                f"request monitor CPU counters regressed or disagree: busy={busy_delta} target={target_delta}"
            )
        swap_in = after["swap_io_pages"]["pswpin"] - before["swap_io_pages"]["pswpin"]
        swap_out = after["swap_io_pages"]["pswpout"] - before["swap_io_pages"]["pswpout"]
        if swap_in < 0 or swap_out < 0 or swap_in > MAX_SWAP_IO_PAGES or swap_out > MAX_SWAP_IO_PAGES:
            raise RuntimeError(f"swap IO contaminated monitored request: pswpin={swap_in} pswpout={swap_out}")
        if elapsed < MIN_MONITORED_INTERVAL_S:
            intervals.append({
                "elapsed_s": elapsed,
                "cpu_evaluation": "skipped_below_clock_tick_resolution",
                "swap_in_pages": swap_in,
                "swap_out_pages": swap_out,
            })
            continue
        if target_delta > busy_delta:
            raise RuntimeError(
                f"request monitor CPU counters disagree: busy={busy_delta} target={target_delta}"
            )
        external_cores = (busy_delta - target_delta) / (elapsed * hz)
        if not math.isfinite(external_cores) or external_cores > MAX_EXTERNAL_CPU_CORES:
            raise RuntimeError(
                f"material external CPU interference exceeded {MAX_EXTERNAL_CPU_CORES} cores: {external_cores}"
            )
        cpu_intervals += 1
        intervals.append({
            "elapsed_s": elapsed,
            "aggregate_busy_tick_delta": busy_delta,
            "target_tick_delta": target_delta,
            "external_cpu_cores": external_cores,
            "swap_in_pages": swap_in,
            "swap_out_pages": swap_out,
        })
    if cpu_intervals == 0:
        raise RuntimeError(
            f"request monitor captured no interval of at least {MIN_MONITORED_INTERVAL_S} seconds"
        )
    return {
        "status": "pass",
        "sample_count": len(samples),
        "interval_s": MONITOR_INTERVAL_S,
        "minimum_cpu_interval_s": MIN_MONITORED_INTERVAL_S,
        "external_cpu_ceiling_cores": MAX_EXTERNAL_CPU_CORES,
        "swap_io_page_ceiling": MAX_SWAP_IO_PAGES,
        "intervals": intervals,
        "samples": samples,
    }


def monitored_query(
    port: int,
    body: dict[str, Any],
    pid: int,
) -> tuple[dict[str, Any] | None, dict[str, Any], Exception | None]:
    samples = [request_interference_snapshot(pid)]
    monitor_errors: list[Exception] = []
    stop = threading.Event()

    def sample_until_stopped() -> None:
        while not stop.wait(MONITOR_INTERVAL_S):
            try:
                samples.append(request_interference_snapshot(pid))
            except Exception as exc:  # noqa: BLE001 - surfaced in the monitor artifact
                monitor_errors.append(exc)
                return

    thread = threading.Thread(target=sample_until_stopped, name="laguna-request-monitor", daemon=False)
    thread.start()
    response = None
    query_error: Exception | None = None
    try:
        response = query_chat(port, body)
    except Exception as exc:  # noqa: BLE001 - monitor completion remains mandatory
        query_error = exc
    finally:
        stop.set()
        thread.join(timeout=max(5.0, MONITOR_INTERVAL_S * 2))
        if thread.is_alive():
            monitor_errors.append(RuntimeError("request monitor thread did not terminate"))
        try:
            samples.append(request_interference_snapshot(pid))
        except Exception as exc:  # noqa: BLE001 - surfaced in the monitor artifact
            monitor_errors.append(exc)
    try:
        evidence = validate_request_monitor_samples(samples)
        if monitor_errors:
            raise RuntimeError(f"request monitor capture failed: {[repr(error) for error in monitor_errors]}")
    except Exception as exc:
        evidence = {
            "status": "fail",
            "error": repr(exc),
            "capture_errors": [repr(error) for error in monitor_errors],
            "samples": samples,
        }
        if query_error is None:
            query_error = exc
    return response, evidence, query_error


def live_cpu_evidence(
    pid: int,
    port: int,
    runtime_artifacts: dict[str, Any],
) -> dict[str, Any]:
    snapshot = system_snapshot()
    evidence: dict[str, Any] = {"status": "fail", "captured_at": utc_now(), "system_snapshot": snapshot}
    try:
        processes = snapshot["processes"]
        exact = processes["exact_llama_processes"]
        if len(exact) != 1 or exact[0]["pid"] != pid:
            raise RuntimeError(f"live server process evidence is not exactly the launched CPU server: {exact}")
        if processes["autopilot_processes"] or processes["kfd_owner"] or processes["rocm_owner"]:
            raise RuntimeError(f"GPU or interfering owner observed while CPU server is live: {processes}")
        evidence["target_executable"] = target_executable_evidence(pid, runtime_artifacts["server"])
        evidence["target_mapped_runtime"] = target_mapped_runtime_evidence(pid, runtime_artifacts)
        evidence["target_listener"] = target_listener_evidence(pid, port)
        affinity = run_capture(["taskset", "-pc", str(pid)], timeout=20)
        evidence["affinity"] = affinity
        command_required(affinity, "live taskset placement")
        affinity_rows = [
            line.rsplit(":", 1)[1].strip()
            for line in affinity["stdout"].splitlines()
            if "current affinity list:" in line
        ]
        if affinity_rows != [CPUSET]:
            raise RuntimeError(f"live server CPU affinity is not pinned to {CPUSET}: {affinity}")
        expected_nodes = numactl_available_nodes(snapshot["numactl_hardware"])
        evidence["available_numa_nodes"] = expected_nodes
        evidence["target_process_policy"] = target_process_policy(pid, expected_nodes)
        evidence["target_process_environment"] = target_process_environment(pid)
        numastat = run_capture(["numastat", "-p", str(pid)], timeout=20)
        evidence["numastat"] = numastat
        evidence["numastat_residency"] = parse_numastat_residency(numastat, expected_nodes)
    except Exception as exc:
        evidence["error"] = repr(exc)
        raise LiveEvidenceFailure(str(exc), evidence) from exc
    evidence["status"] = "pass"
    return evidence


def write_boundary_evidence(
    path: Path,
    pid: int,
    port: int,
    runtime_artifacts: dict[str, Any],
) -> dict[str, Any]:
    try:
        evidence = live_cpu_evidence(pid, port, runtime_artifacts)
    except LiveEvidenceFailure as exc:
        write_json(path, exc.evidence)
        raise
    except Exception as exc:
        write_json(path, {"status": "fail", "captured_at": utc_now(), "error": repr(exc)})
        raise
    write_json(path, evidence)
    return evidence


def target_process_policy(pid: int, expected_nodes: list[int], proc_root: Path = Path("/proc")) -> dict[str, Any]:
    status_path = proc_root / str(pid) / "status"
    numa_maps_path = proc_root / str(pid) / "numa_maps"
    try:
        status = status_path.read_text(encoding="utf-8")
        numa_maps = numa_maps_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read target process CPU/NUMA policy for pid {pid}: {exc}") from exc
    allowed_rows = [line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("Cpus_allowed_list:")]
    if allowed_rows != [CPUSET]:
        raise RuntimeError(f"target process Cpus_allowed_list is not exactly {CPUSET}: {allowed_rows}")
    map_rows = [line.split() for line in numa_maps.splitlines() if line.strip()]
    if not map_rows or any(len(row) < 2 for row in map_rows):
        raise RuntimeError("target process numa_maps is empty or malformed")
    policies = [row[1] for row in map_rows]
    unique_policies = sorted(set(policies))
    if len(unique_policies) != 1 or not unique_policies[0].startswith(("interleave:", "interleave=")):
        raise RuntimeError(f"target process numa_maps does not prove a consistent interleave policy: {unique_policies}")
    interleave_nodes = parse_id_set(re.split(r"[:=]", unique_policies[0], maxsplit=1)[1], "interleave NUMA node")
    if interleave_nodes != expected_nodes:
        raise RuntimeError(f"target process interleave nodes are not all available nodes: expected={expected_nodes} actual={interleave_nodes}")
    return {
        "status_path": str(status_path),
        "cpus_allowed_list": allowed_rows[0],
        "numa_maps_path": str(numa_maps_path),
        "interleave_policy": unique_policies[0],
        "interleave_nodes": interleave_nodes,
        "numa_map_rows": len(map_rows),
        "numa_maps": numa_maps,
    }


def git_capture(argv: list[str]) -> dict[str, Any]:
    capture = run_capture(["git", "-C", str(CANONICAL_SOURCE), *argv])
    command_required(capture, "git " + " ".join(argv))
    return capture


def validate_candidate_provenance() -> dict[str, Any]:
    if not CANONICAL_BINARY.is_file():
        raise FileNotFoundError(f"candidate binary missing: {CANONICAL_BINARY}")
    head = git_capture(["rev-parse", "HEAD"])["stdout"].strip()
    branch = git_capture(["branch", "--show-current"])["stdout"].strip()
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"candidate identity mismatch: branch={branch} head={head}")
    tracked = git_capture(["status", "--porcelain=v1"])["stdout"].splitlines()
    if tracked:
        raise RuntimeError(f"candidate source must have zero tracked/index/untracked changes: {tracked}")
    version = run_capture([str(CANONICAL_BINARY), "--version"], env=child_env())
    command_required(version, "llama-server --version")
    version_text = version["stdout"] + version["stderr"]
    if head[:12] not in version_text and head[:7] not in version_text:
        raise RuntimeError("llama-server --version does not contain candidate HEAD short SHA")
    binary_identity = stable_file_identity(CANONICAL_BINARY)
    if binary_identity["sha256"] != EXPECTED_SERVER_SHA256:
        raise RuntimeError("candidate llama-server SHA256 does not match the frozen v8 candidate")
    ldd = run_capture(["ldd", str(CANONICAL_BINARY)], env=child_env())
    command_required(ldd, "ldd llama-server")
    library_targets: list[Path] = []
    openmp_targets: list[Path] = []
    for row in ldd["stdout"].splitlines():
        if "libgomp.so" in row or "libomp.so" in row:
            target = row.split("=>", 1)[1].split("(", 1)[0].strip() if "=>" in row else ""
            if not target or target == "not found":
                raise RuntimeError(f"candidate ldd OpenMP runtime is unresolved: {row}")
            openmp_targets.append(Path(target).resolve())
        if "libllama" not in row and "libggml" not in row:
            continue
        target = row.split("=>", 1)[1].split("(", 1)[0].strip() if "=>" in row else ""
        if not target or target == "not found" or not Path(target).resolve().parent == CANONICAL_BINARY.parent:
            raise RuntimeError(f"candidate ldd library does not resolve in build-v8-cpu/bin: {row}")
        library_targets.append(Path(target).resolve())
    if not library_targets:
        raise RuntimeError("ldd does not expose libllama/libggml candidate targets")
    if len(set(library_targets)) != len(library_targets):
        raise RuntimeError(f"ldd exposes duplicate candidate llama/ggml targets: {library_targets}")
    library_identities = [stable_file_identity(path) for path in sorted(library_targets)]
    library_sha256 = {Path(identity["path"]).name: identity["sha256"] for identity in library_identities}
    if library_sha256 != EXPECTED_LOCAL_LIBRARY_SHA256:
        raise RuntimeError(
            "candidate local llama/ggml library set or SHA256 does not match the frozen v8 candidate: "
            f"actual={library_sha256} expected={EXPECTED_LOCAL_LIBRARY_SHA256}"
        )
    expected_openmp = OPENMP_RUNTIME.resolve(strict=True)
    if openmp_targets != [expected_openmp]:
        raise RuntimeError(
            f"candidate must resolve exactly the canonical LLVM20 OpenMP runtime: "
            f"actual={openmp_targets} expected={[expected_openmp]}"
        )
    openmp_identity = stable_file_identity(expected_openmp)
    if openmp_identity["sha256"] != EXPECTED_OPENMP_RUNTIME_SHA256:
        raise RuntimeError("canonical LLVM20 OpenMP runtime SHA256 does not match the pinned artifact")
    return {
        "source_root": str(CANONICAL_SOURCE), "binary": str(CANONICAL_BINARY), "branch": branch, "head": head,
        "source_clean": True,
        "version": version, "ldd": ldd, "ldd_candidate_targets": [str(path) for path in library_targets],
        "local_library_identities": library_identities,
        "openmp_runtime_identity": openmp_identity,
        "environment": child_env(), "binary_identity": binary_identity,
    }


def checked_model_identity(label: str, path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    identity = stable_file_identity(path)
    if identity["bytes"] != expected_bytes:
        raise ValueError(f"{label} model has {identity['bytes']} bytes, expected {expected_bytes}")
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} model SHA256 does not match the pinned Laguna artifact")
    return identity


def validate_q8(path: Path) -> dict[str, Any]:
    if path.name.endswith(".part"):
        raise ValueError("Q8 model must be complete; .part files are never accepted")
    return checked_model_identity("Q8", path, Q8_BYTES, Q8_SHA256)


def lanes() -> tuple[Lane, Lane]:
    return (Lane("q4_k_m", Q4_MODEL), Lane("q8_0", Q8_MODEL))


def validate_models() -> dict[str, Any]:
    if any("iq2" in path.name.lower() for path in (Q4_MODEL, Q8_MODEL, DRAFTER_MODEL)):
        raise ValueError("IQ2 models are forbidden in this CPU-only observation")
    return {
        "q4": checked_model_identity("Q4", Q4_MODEL, Q4_BYTES, Q4_SHA256),
        "q8": validate_q8(Q8_MODEL),
        "drafter": checked_model_identity("drafter", DRAFTER_MODEL, DRAFTER_BYTES, DRAFTER_SHA256),
    }


def collect_execution_identity() -> dict[str, Any]:
    candidate = validate_candidate_provenance()
    models = validate_models()
    artifacts = {
        "server": candidate["binary_identity"],
        "local_llama_ggml_libraries": candidate["local_library_identities"],
        "openmp_runtime": candidate["openmp_runtime_identity"],
        "models": models,
        "runner": stable_file_identity(Path(__file__).resolve()),
    }
    return {"candidate": candidate, "models": models, "artifacts": artifacts}


def require_matching_postflight(preflight: dict[str, Any], postflight: dict[str, Any]) -> None:
    if postflight["artifacts"] != preflight["artifacts"]:
        raise RuntimeError(
            "execution artifact identity changed between preflight and postflight: "
            f"pre={preflight['artifacts']} post={postflight['artifacts']}"
        )


def server_argv(lane: Lane, arm: Arm, port: int) -> list[str]:
    server = [
        str(CANONICAL_BINARY), "-m", str(lane.model), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(CONTEXT), "-t", str(THREADS), "-tb", str(THREADS), "-fa", "on", "-dev", "none",
        "-ngl", "0", "--no-op-offload", "--no-mmap", "--seed", str(SEED), "--temp", "0", "--top-k", "1",
        "--top-p", "1", "--reasoning", "off", "--reasoning-budget", "0", "--jinja",
    ]
    if arm.speculative:
        server.extend([
            "-md", str(DRAFTER_MODEL), "--spec-draft-device", "none", "--spec-draft-ngl", "0",
            "--spec-type", "draft-dflash", "--spec-draft-n-max", "15", "--spec-draft-n-min", "0",
            "--spec-draft-p-min", "0", "--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0",
        ])
    argv = ["taskset", "-c", CPUSET, "numactl", "--interleave=all", *server]
    validate_server_argv(argv)
    return argv


def validate_server_argv(argv: list[str]) -> None:
    if argv.count("--no-mmap") != 1 or "--mmap" in argv:
        raise RuntimeError("server argv must contain exactly one --no-mmap and no conflicting --mmap")


def request_body(prompt: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS, "seed": SEED, "temperature": 0, "top_k": 1, "top_p": 1, "stream": False}


def warmup_body() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": WARMUP_PROMPT}],
        "max_tokens": WARMUP_MAX_TOKENS,
        "seed": SEED,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "stream": False,
    }


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
        except Exception:  # noqa: BLE001 - final timeout is the useful diagnostic
            pass
        time.sleep(1)
    raise TimeoutError("llama-server health check timed out")


def query_chat(port: int, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def metric_ms(timings: dict[str, Any], name: str) -> float:
    value = timings.get(name)
    if value is None:
        raise RuntimeError(f"response has no required timings.{name}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"response has nonnumeric timings.{name}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise RuntimeError(f"response has nonpositive or nonfinite timings.{name}")
    return numeric


def positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"response has non-integral {name}")
    if value < 0 or (value == 0 and not allow_zero):
        raise RuntimeError(f"response has invalid {name}")
    return value


def finite_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"non-numeric {name}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or (numeric == 0 and not allow_zero):
        raise RuntimeError(f"nonpositive or nonfinite {name}")
    return numeric


def anti_garbage_validity(content: str) -> dict[str, Any]:
    stripped = content.strip()
    alphabetic = sum(character.isalpha() for character in stripped)
    nonspace = [character for character in stripped if not character.isspace()]
    unique_nonspace = len(set(nonspace))
    longest_run = 0
    run = 0
    previous = None
    for character in nonspace:
        run = run + 1 if character == previous else 1
        longest_run = max(longest_run, run)
        previous = character
    valid = len(stripped) >= 64 and alphabetic >= 24 and unique_nonspace >= 8 and longest_run < max(16, len(nonspace) // 2)
    return {"valid": valid, "characters": len(stripped), "alphabetic_characters": alphabetic, "unique_nonspace_characters": unique_nonspace, "longest_nonspace_run": longest_run}


def unique_result_line(content: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
    matches = pattern.findall(content)
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(f"semantic validation requires exactly one nonempty {label}: result line")
    return matches[0]


def strict_json_array(text: str, label: str) -> list[Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(text, parse_constant=reject_nonfinite)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"semantic validation found invalid JSON in {label}: result") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"semantic validation requires a JSON array in {label}: result")
    return value


def validate_prime_semantics(content: str) -> dict[str, Any]:
    prime_text = unique_result_line(content, "PRIMES")
    if not re.fullmatch(r"\d+(?:[ \t]*,[ \t]*\d+)*", prime_text):
        raise RuntimeError("semantic validation found malformed PRIMES result")
    primes = [int(value.strip()) for value in prime_text.split(",")]
    expected_primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if primes != expected_primes:
        raise RuntimeError(f"semantic validation found an incorrect prime list: {primes}")
    sum_text = unique_result_line(content, "SUM")
    if not re.fullmatch(r"\d+", sum_text) or int(sum_text) != 311:
        raise RuntimeError(f"semantic validation found an incorrect prime sum: {sum_text}")
    return {"task": SEMANTIC_TASKS[0], "valid": True, "primes": primes, "sum": int(sum_text)}


def validate_flatten_semantics(content: str) -> dict[str, Any]:
    flattened = strict_json_array(unique_result_line(content, "FLAT"), "FLAT")
    expected = [2, 1, "hi", 3, None, False]
    canonical = json.dumps(flattened, separators=(",", ":"), ensure_ascii=True)
    expected_canonical = json.dumps(expected, separators=(",", ":"), ensure_ascii=True)
    if canonical != expected_canonical:
        raise RuntimeError(f"semantic validation found an incorrect flattened sequence: {flattened}")
    return {"task": SEMANTIC_TASKS[1], "valid": True, "flattened": flattened}


def finite_numeric_vector(value: list[Any], label: str, expected_length: int) -> list[float]:
    if len(value) != expected_length:
        raise RuntimeError(f"semantic validation found the wrong {label} vector length")
    numeric = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise RuntimeError(f"semantic validation found a nonnumeric or nonfinite {label} value")
        numeric.append(float(item))
    return numeric


def validate_normalization_semantics(content: str) -> dict[str, Any]:
    normalized = finite_numeric_vector(
        strict_json_array(unique_result_line(content, "NORMALIZED"), "NORMALIZED"),
        "NORMALIZED",
        4,
    )
    zero_case = finite_numeric_vector(
        strict_json_array(unique_result_line(content, "ZERO_CASE"), "ZERO_CASE"),
        "ZERO_CASE",
        3,
    )
    expected = [0.0, 0.2, 0.3, 0.5]
    if any(abs(actual - wanted) > 1e-9 for actual, wanted in zip(normalized, expected, strict=True)):
        raise RuntimeError(f"semantic validation found an incorrect normalized vector: {normalized}")
    if any(abs(value) > 1e-12 for value in zero_case):
        raise RuntimeError(f"semantic validation found an incorrect zero-total result: {zero_case}")
    return {
        "task": SEMANTIC_TASKS[2],
        "valid": True,
        "normalized": normalized,
        "normalized_sum": sum(normalized),
        "zero_case": zero_case,
    }


def validate_prompt_semantics(content: str, prompt_index: int) -> dict[str, Any]:
    validators = (validate_prime_semantics, validate_flatten_semantics, validate_normalization_semantics)
    if isinstance(prompt_index, bool) or not 1 <= prompt_index <= len(validators):
        raise RuntimeError(f"semantic validation received invalid prompt index: {prompt_index}")
    return validators[prompt_index - 1](content)


def iqk_engagement_evidence(log_path: Path, lane: Lane) -> dict[str, Any]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"cannot read server stderr for IQK engagement: {log_path}: {exc}") from exc
    active_lines = [line for line in text.splitlines() if "[iqk] ACTIVE:" in line]
    parsed_lines = IQK_ACTIVE_RE.findall(text)
    if len(parsed_lines) != len(active_lines):
        raise RuntimeError(f"one or more IQK ACTIVE lines is malformed: {active_lines}")
    type_codes = [int(value) for value in parsed_lines]
    if lane.name == "q4_k_m":
        if 12 not in type_codes:
            raise RuntimeError(f"Q4_K_M lane did not prove [iqk] ACTIVE type=12 engagement: {type_codes}")
        if any(type_code in {16, 17, 18, 21, 22} for type_code in type_codes):
            raise RuntimeError(f"Q4_K_M lane falsely reported a native IQ weight path: {type_codes}")
        expectation = "Q4_K IQK type=12 active"
    elif lane.name == "q8_0":
        if active_lines:
            raise RuntimeError(f"Q8_0 lane falsely reported an IQK ACTIVE path: {active_lines}")
        expectation = "no IQK active path because GGML_IQK_Q8_0 is absent"
    else:
        raise RuntimeError(f"unknown CPU Laguna lane for IQK evidence: {lane.name}")
    return {
        "status": "pass",
        "lane": lane.name,
        "expectation": expectation,
        "active_lines": active_lines,
        "active_type_codes": sorted(set(type_codes)),
        "raw_log_identity": stable_file_identity(log_path),
    }


def validate_warmup_response(response: dict[str, Any]) -> dict[str, Any]:
    timings = response.get("timings")
    choices = response.get("choices")
    if not isinstance(timings, dict) or not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("warmup response is missing exact timings or single-choice structure")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("warmup response has no nonempty string content")
    return {
        "status": "pass",
        "prompt_tokens": positive_int(timings.get("prompt_n"), "warmup timings.prompt_n"),
        "completion_tokens": positive_int(timings.get("predicted_n"), "warmup timings.predicted_n"),
        "prompt_ms": metric_ms(timings, "prompt_ms"),
        "decode_ms": metric_ms(timings, "predicted_ms"),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def response_row(response: dict[str, Any], arm: Arm, prompt_index: int) -> dict[str, Any]:
    timings = response.get("timings") or {}
    choices = response.get("choices") or []
    message = choices[0].get("message") or {} if choices and isinstance(choices[0], dict) else {}
    message = message if isinstance(message, dict) else {}
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and str(reasoning_content).strip():
        raise RuntimeError("response contains reasoning_content despite the pinned reasoning-off contract")
    content = str(message.get("content") or "")
    row = {
        "prompt_tokens": positive_int(timings.get("prompt_n"), "timings.prompt_n"),
        "completion_tokens": positive_int(timings.get("predicted_n"), "timings.predicted_n"),
        "prompt_ms": metric_ms(timings, "prompt_ms"), "decode_ms": metric_ms(timings, "predicted_ms"),
        "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    row["semantic_validation"] = validate_prompt_semantics(content, prompt_index)
    validity = anti_garbage_validity(content)
    row["validity"] = validity
    if not validity["valid"] or row["completion_tokens"] < MIN_COMPLETION_TOKENS:
        raise RuntimeError("coherence completion floor failed")
    if arm.speculative:
        if "draft_n" not in timings or "draft_n_accepted" not in timings:
            raise RuntimeError("DFlash response lacks explicit draft_n/draft_n_accepted telemetry")
        row["draft_n"] = positive_int(timings["draft_n"], "timings.draft_n")
        row["draft_n_accepted"] = positive_int(timings["draft_n_accepted"], "timings.draft_n_accepted")
        if row["draft_n_accepted"] > row["draft_n"]:
            raise RuntimeError("DFlash accepted counter exceeds generated counter")
    return row


def process_group_members(pgid: int) -> list[int]:
    capture = run_capture(["ps", "-eo", "pid=,pgid="])
    command_required(capture, "process-group snapshot")
    members = []
    for row in capture["stdout"].splitlines():
        parts = row.split()
        if len(parts) == 2 and int(parts[1]) == pgid:
            members.append(int(parts[0]))
    return members


def port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        socket_closed = sock.connect_ex(("127.0.0.1", port)) != 0
    return socket_closed and all(listener["port"] != port for listener in tcp_listeners())


def cleanup(proc: subprocess.Popen[str], port: int) -> dict[str, Any]:
    pgid = os.getpgid(proc.pid)
    signals = []
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
    members = process_group_members(pgid)
    if members:
        signals.append("SIGKILL-descendants")
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(SETTLE_S)
        members = process_group_members(pgid)
    alive = []
    for pid in members:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            alive.append(pid)
    post = system_snapshot()
    port_is_closed = port_closed(port)
    valid = proc.poll() is not None and not members and not alive and port_is_closed and not post["processes"]["exact_llama_processes"] and not post["processes"]["autopilot_processes"] and not post["processes"]["kfd_owner"] and not post["processes"]["rocm_owner"]
    result = {"status": "pass" if valid else "fail", "leader_dead": proc.poll() is not None, "process_group_members": members, "verified_dead_pids": alive, "port_closed": port_is_closed, "signals": signals, "post_settle": post}
    if not valid:
        raise RuntimeError(f"gate-critical cleanup failure: {result}")
    return result


def weighted_tps(rows: list[dict[str, Any]], tokens_key: str, ms_key: str) -> float:
    tokens = sum(positive_int(row.get(tokens_key), f"{tokens_key} row") for row in rows)
    milliseconds = sum(finite_number(row.get(ms_key), f"{ms_key} row") for row in rows)
    rate = tokens * 1000.0 / milliseconds
    return finite_number(rate, f"weighted throughput {tokens_key}/{ms_key}")


def median_mad(values: list[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("cannot summarize an empty metric series")
    checked = [finite_number(value, "summary metric") for value in values]
    median = finite_number(float(statistics.median(checked)), "summary median")
    mad = finite_number(float(statistics.median(abs(value - median) for value in checked)), "summary MAD", allow_zero=True)
    return {"n": len(checked), "median": median, "mad": mad}


def run_replicate(
    lane: Lane,
    arm: Arm,
    rep: int,
    run_dir: Path,
    model_identity: dict[str, Any],
    runtime_artifacts: dict[str, Any],
    schedule_position: int,
    lane_position: int,
    pair_position: int,
) -> dict[str, Any]:
    rep_dir = run_dir / "runs" / f"{lane.name}_{arm.name}_rep{rep}"
    rep_dir.mkdir(parents=True, exist_ok=False)
    port = find_free_port()
    argv = server_argv(lane, arm, port)
    write_json(rep_dir / "server_argv.json", argv)
    write_json(rep_dir / "child_environment_expected.json", child_env())
    write_json(rep_dir / "model_identity.json", model_identity)
    write_json(rep_dir / "prompts.json", list(PROMPTS))
    write_json(rep_dir / "warmup_policy.json", WARMUP_POLICY)
    proc: subprocess.Popen[str] | None = None
    rows: list[dict[str, Any]] = []
    warmup_result: dict[str, Any] | None = None
    primary_error = None
    cleanup_error = None
    engagement_error = None
    result: dict[str, Any] = {
        "status": "error",
        "lane": lane.name,
        "arm": arm.name,
        "rep": rep,
        "schedule_position": schedule_position,
        "lane_position": lane_position,
        "pair_position": pair_position,
        "port": port,
        "prompt_rows": rows,
    }
    try:
        with (rep_dir / "server.stderr").open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                text=True,
                start_new_session=True,
                env=child_env(),
            )
        wait_for_health(port)
        warmup_request = warmup_body()
        write_json(rep_dir / "warmup_request.json", warmup_request)
        write_boundary_evidence(
            rep_dir / "warmup_pre_evidence.json",
            proc.pid,
            port,
            runtime_artifacts,
        )
        warmup_response, warmup_monitor, warmup_error = monitored_query(
            port,
            warmup_request,
            proc.pid,
        )
        write_json(rep_dir / "warmup_monitor.json", warmup_monitor)
        write_boundary_evidence(
            rep_dir / "warmup_post_evidence.json",
            proc.pid,
            port,
            runtime_artifacts,
        )
        if warmup_monitor.get("status") != "pass" and warmup_error is None:
            warmup_error = RuntimeError(f"warmup monitor failed without an exception: {warmup_monitor}")
        if warmup_error is not None:
            raise warmup_error
        if warmup_response is None:
            raise RuntimeError("warmup request returned no response")
        write_json(rep_dir / "warmup_response.json", warmup_response)
        warmup_result = validate_warmup_response(warmup_response)
        write_json(rep_dir / "warmup_result.json", warmup_result)
        for index, prompt in enumerate(PROMPTS, 1):
            body = request_body(prompt)
            write_json(rep_dir / f"request_{index}.json", body)
            write_boundary_evidence(
                rep_dir / f"request_{index}_pre_evidence.json",
                proc.pid,
                port,
                runtime_artifacts,
            )
            response, monitor, request_error = monitored_query(port, body, proc.pid)
            write_json(rep_dir / f"request_{index}_monitor.json", monitor)
            write_boundary_evidence(
                rep_dir / f"request_{index}_post_evidence.json",
                proc.pid,
                port,
                runtime_artifacts,
            )
            if monitor.get("status") != "pass" and request_error is None:
                request_error = RuntimeError(f"request monitor failed without an exception: {monitor}")
            if request_error is not None:
                raise request_error
            if response is None:
                raise RuntimeError("request returned no response")
            write_json(rep_dir / f"response_{index}.json", response)
            row = response_row(response, arm, index)
            row["prompt_index"] = index
            rows.append(row)
        result = {"status": "ok", "lane": lane.name, "arm": arm.name, "rep": rep, "schedule_position": schedule_position, "lane_position": lane_position, "pair_position": pair_position, "port": port, "warmup": warmup_result, "prompt_rows": rows, "prompt_tps": weighted_tps(rows, "prompt_tokens", "prompt_ms"), "decode_tps": weighted_tps(rows, "completion_tokens", "decode_ms"), "completion_tokens": sum(row["completion_tokens"] for row in rows), "draft_n": sum(row["draft_n"] for row in rows) if arm.speculative else None, "draft_n_accepted": sum(row["draft_n_accepted"] for row in rows) if arm.speculative else None}
    except Exception as exc:  # noqa: BLE001 - write a complete primary-failure artifact
        primary_error = repr(exc)
    finally:
        if proc is None:
            cleanup_error = "server process was not created"
            result["cleanup"] = {"status": "fail", "error": cleanup_error}
        else:
            try:
                result["cleanup"] = cleanup(proc, port)
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure alongside primary error
                cleanup_error = repr(exc)
                result["cleanup"] = {"status": "fail", "error": cleanup_error}
        if result["cleanup"].get("status") == "pass":
            try:
                result["iqk_engagement"] = iqk_engagement_evidence(rep_dir / "server.stderr", lane)
            except Exception as exc:  # noqa: BLE001 - engagement is required post-shutdown evidence
                engagement_error = repr(exc)
                result["iqk_engagement"] = {"status": "fail", "error": engagement_error}
        else:
            engagement_error = "IQK engagement not evaluated because cleanup did not pass"
            result["iqk_engagement"] = {"status": "fail", "error": engagement_error}
        write_json(rep_dir / "iqk_engagement.json", result["iqk_engagement"])
        result["primary_error"] = primary_error
        result["cleanup_error"] = cleanup_error
        result["engagement_error"] = engagement_error
        if primary_error or cleanup_error or engagement_error:
            result["status"] = "error"
        write_json(rep_dir / "result.json", result)
    return result


def balanced_schedule() -> list[dict[str, Any]]:
    cells = []
    stable_lanes = lanes()
    for rep in range(1, REPS + 1):
        lane_order = stable_lanes if rep % 2 else tuple(reversed(stable_lanes))
        for lane_position, lane in enumerate(lane_order, 1):
            lane_index = stable_lanes.index(lane)
            arm_order = ARMS if (rep + lane_index) % 2 else tuple(reversed(ARMS))
            for pair_position, arm in enumerate(arm_order, 1):
                cells.append({
                    "schedule_position": len(cells) + 1,
                    "lane_position": lane_position,
                    "pair_position": pair_position,
                    "lane": lane.name,
                    "arm": arm.name,
                    "rep": rep,
                    "prompt_count": len(PROMPTS),
                    "seed": SEED,
                })
    validate_schedule_contract(cells)
    return cells


def validate_schedule_contract(cells: list[dict[str, Any]]) -> None:
    if len(cells) != len(lanes()) * len(ARMS) * REPS:
        raise RuntimeError(f"schedule must contain exactly 20 cells: {len(cells)}")
    if [cell.get("schedule_position") for cell in cells] != list(range(1, len(cells) + 1)):
        raise RuntimeError("schedule positions are not exact and contiguous")
    expected_keys = {
        (lane.name, arm.name, rep)
        for lane in lanes()
        for arm in ARMS
        for rep in range(1, REPS + 1)
    }
    actual_keys = {(cell.get("lane"), cell.get("arm"), cell.get("rep")) for cell in cells}
    if actual_keys != expected_keys or len(actual_keys) != len(cells):
        raise RuntimeError("schedule does not contain each lane/arm/rep cell exactly once")
    lane_first_counts = {lane.name: 0 for lane in lanes()}
    arm_first_counts = {arm.name: 0 for arm in ARMS}
    for rep in range(1, REPS + 1):
        rep_cells = [cell for cell in cells if cell["rep"] == rep]
        if len(rep_cells) != len(lanes()) * len(ARMS):
            raise RuntimeError(f"rep {rep} does not contain both paired lanes")
        lane_groups = [rep_cells[index:index + len(ARMS)] for index in range(0, len(rep_cells), len(ARMS))]
        expected_lane_order = lanes() if rep % 2 else tuple(reversed(lanes()))
        if [group[0]["lane"] for group in lane_groups] != [lane.name for lane in expected_lane_order]:
            raise RuntimeError(f"rep {rep} lane order is not counterbalanced")
        lane_first_counts[lane_groups[0][0]["lane"]] += 1
        for lane_position, group in enumerate(lane_groups, 1):
            if len({cell["lane"] for cell in group}) != 1:
                raise RuntimeError(f"rep {rep} breaks a base/DFlash lane pair")
            if [cell["lane_position"] for cell in group] != [lane_position, lane_position]:
                raise RuntimeError(f"rep {rep} lane positions are invalid")
            if [cell["pair_position"] for cell in group] != [1, 2]:
                raise RuntimeError(f"rep {rep} pair positions are invalid")
            arm_first_counts[group[0]["arm"]] += 1
    if abs(lane_first_counts[lanes()[0].name] - lane_first_counts[lanes()[1].name]) > 1:
        raise RuntimeError(f"lane-first counts are not counterbalanced: {lane_first_counts}")
    if len(set(arm_first_counts.values())) != 1:
        raise RuntimeError(f"arm-first counts are not globally balanced: {arm_first_counts}")
    for lane in lanes():
        lane_pairs = [
            [cell for cell in cells if cell["lane"] == lane.name and cell["rep"] == rep]
            for rep in range(1, REPS + 1)
        ]
        first_counts = {
            arm.name: sum(pair[0]["arm"] == arm.name for pair in lane_pairs)
            for arm in ARMS
        }
        if abs(first_counts[BASE.name] - first_counts[DFLASH.name]) > 1:
            raise RuntimeError(f"within-lane arm order is not counterbalanced for {lane.name}: {first_counts}")


def build_plan() -> dict[str, Any]:
    return {"schema": "epyc.laguna_cpu_dflash_observation.plan.v3", "created_at": utc_now(), "recipe": {"context": CONTEXT, "max_tokens": MAX_TOKENS, "min_completion_tokens": MIN_COMPLETION_TOKENS, "seed": SEED, "threads": THREADS, "cpuset": CPUSET, "ggml_iqk": "1", "mmap": False, "warmup_policy": WARMUP_POLICY, "host_requirements": {"scaling_governor": "performance_on_every_online_cpu", "thp_enabled_active": "always", "thp_defrag_active": "always", "numa_balancing": "0", "request_external_cpu_ceiling_cores": MAX_EXTERNAL_CPU_CORES, "request_swap_io_page_ceiling": MAX_SWAP_IO_PAGES}, "prompt_pack": list(PROMPTS), "semantic_tasks": list(SEMANTIC_TASKS), "prompt_count": len(PROMPTS)}, "observation_policy": OBSERVATION_POLICY, "schedule_contract": "rep_outer_lane_counterbalanced_arm_paired_and_globally_counterbalanced", "cells": balanced_schedule()}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    equality = []
    expected_keys = {(lane.name, arm.name, rep) for lane in lanes() for arm in ARMS for rep in range(1, REPS + 1)}
    actual_keys = {(row.get("lane"), row.get("arm"), row.get("rep")) for row in results}
    if actual_keys != expected_keys or len(results) != len(expected_keys):
        raise RuntimeError(f"required exact 20-cell key set mismatch: expected={expected_keys} actual={actual_keys}")
    for row in results:
        positive_int(row.get("schedule_position"), "schedule_position")
        positive_int(row.get("lane_position"), "lane_position")
        positive_int(row.get("pair_position"), "pair_position")
        positive_int(row.get("rep"), "rep")
    expected_schedule = [
        (cell["schedule_position"], cell["lane_position"], cell["pair_position"], cell["lane"], cell["arm"], cell["rep"])
        for cell in balanced_schedule()
    ]
    actual_schedule = [
        (row.get("schedule_position"), row.get("lane_position"), row.get("pair_position"), row.get("lane"), row.get("arm"), row.get("rep"))
        for row in results
    ]
    if actual_schedule != expected_schedule:
        raise RuntimeError(f"balanced paired schedule mismatch: expected={expected_schedule} actual={actual_schedule}")
    if any(row.get("cleanup", {}).get("status") != "pass" for row in results):
        raise RuntimeError("one or more cells has failed cleanup")
    if any(row.get("warmup", {}).get("status") != "pass" for row in results):
        raise RuntimeError("one or more cells lacks a successful fixed unmeasured warmup")
    if any(
        row.get("iqk_engagement", {}).get("status") != "pass"
        or row["iqk_engagement"].get("lane") != row.get("lane")
        or (
            row.get("lane") == "q4_k_m"
            and 12 not in row["iqk_engagement"].get("active_type_codes", [])
        )
        or (
            row.get("lane") == "q8_0"
            and row["iqk_engagement"].get("active_type_codes") != []
        )
        for row in results
    ):
        raise RuntimeError("one or more cells lacks valid lane-specific post-shutdown IQK engagement evidence")
    for row in results:
        finite_number(row.get("prompt_tps"), "replicate prompt_tps")
        finite_number(row.get("decode_tps"), "replicate decode_tps")
        completion_tokens = positive_int(row.get("completion_tokens"), "replicate completion_tokens")
        prompt_completion_tokens = 0
        for prompt in row.get("prompt_rows", []):
            positive_int(prompt.get("prompt_tokens"), "prompt_tokens")
            prompt_completion_tokens += positive_int(prompt.get("completion_tokens"), "completion_tokens")
            finite_number(prompt.get("prompt_ms"), "prompt_ms")
            finite_number(prompt.get("decode_ms"), "decode_ms")
        if completion_tokens != prompt_completion_tokens:
            raise RuntimeError(
                f"replicate completion token total mismatch: row={completion_tokens} prompts={prompt_completion_tokens}"
            )
        if row["arm"] == DFLASH.name:
            generated = positive_int(row.get("draft_n"), "replicate draft_n")
            accepted = positive_int(row.get("draft_n_accepted"), "replicate draft_n_accepted")
            if accepted > generated:
                raise RuntimeError("replicate accepted draft count exceeds generated count")
            finite_number(accepted / generated, "replicate draft acceptance")
        elif row.get("draft_n") is not None or row.get("draft_n_accepted") is not None:
            raise RuntimeError("base replicate unexpectedly contains draft metrics")
    for lane in lanes():
        for arm in ARMS:
            rows = [row for row in results if row["lane"] == lane.name and row["arm"] == arm.name]
            exact_reps = {row["rep"] for row in rows} == set(range(1, REPS + 1))
            exact_prompts = all({prompt["prompt_index"] for prompt in row["prompt_rows"]} == set(range(1, len(PROMPTS) + 1)) for row in rows)
            exact_semantics = all(
                isinstance(prompt.get("prompt_index"), int)
                and not isinstance(prompt["prompt_index"], bool)
                and 1 <= prompt["prompt_index"] <= len(SEMANTIC_TASKS)
                and prompt.get("semantic_validation", {}).get("valid") is True
                and prompt["semantic_validation"].get("task") == SEMANTIC_TASKS[prompt["prompt_index"] - 1]
                for row in rows
                for prompt in row["prompt_rows"]
            )
            if len(rows) != REPS or not exact_reps or not exact_prompts or not exact_semantics or any(row["status"] != "ok" or len(row["prompt_rows"]) != len(PROMPTS) for row in rows):
                raise RuntimeError(f"required complete replicates missing for {lane.name}/{arm.name}")
            key = f"{lane.name}_{arm.name}"
            summary = {"replicates": len(rows), "prompt_tps": median_mad([row["prompt_tps"] for row in rows]), "decode_tps": median_mad([row["decode_tps"] for row in rows])}
            if arm.speculative:
                total_draft = sum(positive_int(row["draft_n"], "summary draft_n") for row in rows)
                total_accepted = sum(positive_int(row["draft_n_accepted"], "summary draft_n_accepted") for row in rows)
                acceptance = finite_number(total_accepted / total_draft, "summary draft acceptance")
                per_rep = []
                for row in rows:
                    per_rep.append({
                        "rep": row["rep"],
                        "generated": row["draft_n"],
                        "accepted": row["draft_n_accepted"],
                        "acceptance": finite_number(row["draft_n_accepted"] / row["draft_n"], "per-rep draft acceptance"),
                    })
                summary["draft_counters"] = {"generated": total_draft, "accepted": total_accepted, "acceptance": acceptance, "per_rep": per_rep}
            else:
                summary["draft_counters"] = "not_applicable"
            summaries[key] = summary
        for rep in range(1, REPS + 1):
            base = next(row for row in results if row["lane"] == lane.name and row["arm"] == BASE.name and row["rep"] == rep)
            dflash = next(row for row in results if row["lane"] == lane.name and row["arm"] == DFLASH.name and row["rep"] == rep)
            for prompt_index in range(1, len(PROMPTS) + 1):
                base_row = next(row for row in base["prompt_rows"] if row["prompt_index"] == prompt_index)
                dflash_row = next(row for row in dflash["prompt_rows"] if row["prompt_index"] == prompt_index)
                equality.append({"lane": lane.name, "rep": rep, "prompt_index": prompt_index, "base_hash": base_row["content_sha256"], "dflash_hash": dflash_row["content_sha256"], "exact_equal": base_row["content_sha256"] == dflash_row["content_sha256"]})
        base_decode = summaries[f"{lane.name}_base"]["decode_tps"]["median"]
        dflash_decode = summaries[f"{lane.name}_dflash"]["decode_tps"]["median"]
        summaries[f"{lane.name}_dflash"]["decode_ratio_vs_base_higher_better"] = finite_number(
            dflash_decode / base_decode,
            "DFlash decode ratio",
        )
    equality_rate = sum(row["exact_equal"] for row in equality) / len(equality)
    return {"schema": "epyc.laguna_cpu_dflash_observation.summary.v3", "created_at": utc_now(), "status": "ok", "arm_summaries": summaries, "output_stability_observation": {"non_gating": True, "contract": "distribution_lossless_not_byte_exact_greedy", "rows": equality, "exact_equality_rate": equality_rate}, "observation_policy": OBSERVATION_POLICY}


def execute(output_dir: Path) -> dict[str, Any]:
    run_dir: Path | None = None
    try:
        commands = require_commands()
        preflight_identity = collect_execution_identity()
        models = preflight_identity["models"]
        preflight = system_snapshot()
        ensure_quiet_cpu_only(preflight)
        run_dir = output_dir / run_stamp()
        if run_dir.exists():
            raise RuntimeError(f"refusing existing exact run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        identity = {**preflight_identity, "required_commands": commands}
        write_json(run_dir / "identity.json", identity)
        write_json(run_dir / "preflight.json", preflight)
        schedule = balanced_schedule()
        write_json(run_dir / "schedule.json", schedule)
        results = []
        lanes_by_name = {lane.name: lane for lane in lanes()}
        arms_by_name = {arm.name: arm for arm in ARMS}
        for cell in schedule:
            result = run_replicate(
                lanes_by_name[cell["lane"]],
                arms_by_name[cell["arm"]],
                cell["rep"],
                run_dir,
                models,
                preflight_identity["artifacts"],
                cell["schedule_position"],
                cell["lane_position"],
                cell["pair_position"],
            )
            results.append(result)
            if result.get("cleanup", {}).get("status") != "pass":
                raise RuntimeError("cleanup fault aborts before the next replicate")
        postflight_identity = collect_execution_identity()
        write_json(run_dir / "postflight_identity.json", postflight_identity)
        require_matching_postflight(preflight_identity, postflight_identity)
        summary = summarize(results)
        write_json(run_dir / "summary.json", summary)
        return {"run_dir": str(run_dir), **summary}
    except Exception as exc:
        if run_dir is not None:
            write_json(run_dir / "failure.json", {"error": repr(exc), "run_dir": str(run_dir), "created_at": utc_now()})
        raise RunFailure(repr(exc), run_dir) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=CANONICAL_BINARY)
    parser.add_argument("--source-root", type=Path, default=CANONICAL_SOURCE)
    parser.add_argument("--q4-model", type=Path, default=Q4_MODEL)
    parser.add_argument("--q8-model", type=Path, default=Q8_MODEL)
    parser.add_argument("--drafter-model", type=Path, default=DRAFTER_MODEL)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--min-completion-tokens", type=int, default=MIN_COMPLETION_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args(argv)
    if args.binary != CANONICAL_BINARY or args.source_root != CANONICAL_SOURCE or args.q4_model != Q4_MODEL or args.drafter_model != DRAFTER_MODEL:
        parser.error("candidate provenance and Q4/DFlash model paths are fixed")
    if args.q8_model != Q8_MODEL or args.q8_model.name.endswith(".part") or "iq2" in args.q8_model.name.lower():
        parser.error("Q8 lane must be a complete non-IQ2 artifact")
    if (args.reps, args.context, args.max_tokens, args.min_completion_tokens, args.seed, args.threads) != (REPS, CONTEXT, MAX_TOKENS, MIN_COMPLETION_TOKENS, SEED, THREADS):
        parser.error("CPU Laguna DFlash recipe is fixed; command-line recipe drift is forbidden")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", build_plan())
    if not args.execute:
        write_json(args.output_dir / "summary.json", {"schema": "epyc.laguna_cpu_dflash_observation.summary.v3", "status": "prepared_no_inference", "observation_policy": OBSERVATION_POLICY})
        print(f"prepared: {args.output_dir}")
        return 0
    try:
        summary = execute(args.output_dir)
    except RunFailure as exc:
        summary = {"schema": "epyc.laguna_cpu_dflash_observation.summary.v3", "status": "failed", "error": repr(exc), "run_dir": str(exc.run_dir) if exc.run_dir else None, "observation_policy": OBSERVATION_POLICY}
    write_json(args.output_dir / "summary.json", summary)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
