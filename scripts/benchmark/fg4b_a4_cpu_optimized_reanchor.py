#!/usr/bin/env python3
"""Proposal-only FG-4b A4 CPU re-anchor using the production serving shape.

This is deliberately not a llama-bench wrapper.  The retired 18.36 t/s FG-4b
observation omitted native MTP and did not exercise the serving configuration
whose ``baseline_tps`` it was intended to refresh.  This runner starts one
frozen-v8 CPU llama-server with the live A4 topology, warms it, then records
the server-reported decode rate of repeated fixed long chat completions.

Default mode is dry-run.  Execution is intentionally gated by an explicit
operator window acknowledgement *and* the region-lock wrapper holding every
region touched by the configured affinity.  The runner emits evidence and a
non-applying JSON-patch proposal; it never writes either registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
REGION_LOCK = ORCHESTRATOR_ROOT / "scripts/region-lock"
LLAMA_ROOT = Path("/mnt/raid0/llm/llama.cpp")
LLAMA_SERVER = LLAMA_ROOT / "build/bin/llama-server"
GIT = Path("/usr/bin/git")
MODEL = Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf")
CPU_LIST = "0-47,96-143"
PHYSICAL_REGIONS = ("q0", "q1")
PORT = 19080
CTX = 32768
UBATCH = 8192
THREADS = 96
REPS = 5
N_PREDICT = 512
WARMUP_TOKENS = 64
WARMUP_CONSECUTIVE = 3
WARMUP_MAX_ATTEMPTS = 8
WARMUP_RELATIVE_TOLERANCE = 0.05
PROTOCOL_ID = "FG-4b/A4-CPU-optimized-server-v1"
PROTOCOL_ATTESTATION_SCHEMA = "epyc.fg4b_a4_cpu_optimized_server_protocol_review.v1"
EXPECTED_LLAMA_BRANCH = "production-consolidated-v8"
EXPECTED_LLAMA_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/architect-27b-finetunes-v8-20260726"

sys.path.insert(0, str(REPO_ROOT / "scripts/lib"))
sys.path.insert(0, str(BENCHMARK_DIR))
from canonical_recipe import (  # noqa: E402
    CANONICAL_OMP_ENV,
    EXPECTED_LIBS_V6_IQK,
    assert_binary_resolves_correctly,
    assert_canonical_env,
    build_canonical_env,
)
from server_np_sweep import (  # noqa: E402
    collect_attestation,
    ensure_clean_runtime,
    find_llama_processes,
    host_health_warnings,
    run_capture,
    start_server,
    stop_server,
)


class ReanchorRefusal(RuntimeError):
    """A required decision-grade invariant was not proven."""


@dataclass(frozen=True)
class DecodeSample:
    ordinal: int
    predicted_n: int
    predicted_per_second: float
    prompt_n: int
    response_chars: int
    finish_reason: str
    timings: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    """Read a required Git identity value from this runner's worktree."""
    result = subprocess.run(
        [str(GIT), "-C", str(REPO_ROOT), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ReanchorRefusal(
            "cannot establish authoritative runner Git identity: "
            + result.stderr.strip()
        )
    return result.stdout.strip()


def instrument_identity() -> dict[str, str]:
    """Return the clean, tracked source identity which a receipt must bind."""
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ReanchorRefusal("authoritative runner worktree is dirty")
    repository = _git_output("remote", "get-url", "origin")
    commit = _git_output("rev-parse", "HEAD")
    tree = _git_output("rev-parse", "HEAD^{tree}")
    try:
        path = Path(__file__).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ReanchorRefusal("authoritative runner is outside its Git worktree") from exc
    _git_output("ls-files", "--error-unmatch", "--", path)
    return {
        "repository": repository,
        "repository_commit": commit,
        "repository_tree": tree,
        "path": path,
        "sha256": sha256(Path(__file__).resolve()),
    }


def protocol_contract() -> dict[str, Any]:
    """Exact contract that the human-reviewed protocol receipt must bind."""
    return {
        "protocol_id": PROTOCOL_ID,
        "metric": "llama-server timings.predicted_per_second",
        "metric_direction": "higher_is_better",
        "model": str(MODEL),
        "binary": str(LLAMA_SERVER),
        "cpu_list": CPU_LIST,
        "physical_regions": list(PHYSICAL_REGIONS),
        "threads": THREADS,
        "ctx": CTX,
        "ubatch": UBATCH,
        "np": 1,
        "native_mtp_draft_max": 4,
        "n_predict": N_PREDICT,
        "ignore_eos": True,
        "required_finish_reason": "length",
        "measured_reps": REPS,
        "aggregation": ["median", "median_absolute_deviation"],
        "warmup": {
            "tokens": WARMUP_TOKENS,
            "consecutive_samples": WARMUP_CONSECUTIVE,
            "relative_tolerance": WARMUP_RELATIVE_TOLERANCE,
            "max_attempts": WARMUP_MAX_ATTEMPTS,
        },
        "cold_cache_preparation": {
            "sync": True,
            "drop_caches": 3,
            "after_clean_host_gate": True,
            "before_server_start": True,
        },
        "per_request_witness": {
            "exclusive_inference_process_tree": True,
            "thread_affinity": {
                "expected_cpu_list": CPU_LIST,
                "thread_union_exact": True,
                "no_thread_outside_expected": True,
                "worker_thread_union_exact": True,
            },
        },
        "durable_publish": "fsync_files_and_staging_dir_then_parent_before_and_after_atomic_rename",
    }


def validate_protocol_attestation(path: Path) -> dict[str, Any]:
    """Validate a human-reviewed receipt against this exact instrument."""
    if not path.is_file():
        raise ReanchorRefusal(f"reviewed protocol attestation not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReanchorRefusal("reviewed protocol attestation is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ReanchorRefusal("reviewed protocol attestation must be an object")
    if payload.get("schema") != PROTOCOL_ATTESTATION_SCHEMA:
        raise ReanchorRefusal("reviewed protocol attestation schema mismatch")
    if payload.get("status") != "ratified":
        raise ReanchorRefusal("optimized-server protocol is not ratified")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ReanchorRefusal("reviewed protocol_id mismatch")
    if payload.get("contract") != protocol_contract():
        raise ReanchorRefusal("reviewed protocol contract does not exactly match the runner")
    identity = instrument_identity()
    if payload.get("instrument_sha256") != identity["sha256"]:
        raise ReanchorRefusal("reviewed protocol receipt does not bind this exact instrument hash")
    instrument = payload.get("instrument")
    if not isinstance(instrument, dict):
        raise ReanchorRefusal("reviewed protocol receipt lacks instrument identity")
    required_identity = {
        "repository": identity["repository"],
        "repository_commit": identity["repository_commit"],
        "repository_tree": identity["repository_tree"],
        "path": identity["path"],
    }
    if any(not isinstance(instrument.get(key), str) for key in required_identity):
        raise ReanchorRefusal("reviewed protocol receipt has incomplete instrument identity")
    if {key: instrument[key] for key in required_identity} != required_identity:
        raise ReanchorRefusal("reviewed protocol receipt instrument identity mismatch")
    amendment = payload.get("human_amendment")
    if not isinstance(amendment, dict):
        raise ReanchorRefusal("reviewed protocol receipt lacks human_amendment")
    amendment_path = Path(str(amendment.get("path") or ""))
    amendment_hash = str(amendment.get("sha256") or "")
    if (
        not amendment_path.is_absolute()
        or not amendment_path.is_file()
        or len(amendment_hash) != 64
        or sha256(amendment_path) != amendment_hash
    ):
        raise ReanchorRefusal("human amendment path/hash is absent or mismatched")
    reviewed_at = payload.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise ReanchorRefusal("reviewed protocol receipt lacks reviewed_at")
    try:
        reviewed = datetime.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise ReanchorRefusal("reviewed_at is not an ISO datetime") from exc
    if reviewed.tzinfo is None:
        raise ReanchorRefusal("reviewed_at must include a timezone")
    if not str(payload.get("reviewer") or "").strip():
        raise ReanchorRefusal("reviewed protocol receipt lacks reviewer attribution")
    return payload | {
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha256(path),
    }


def build_server_command(*, port: int = PORT) -> list[str]:
    """Return the frozen-v8, single-instance A4 serving command.

    Do not feed this through ``canonical_recipe.apply_canonical_prefix``: that
    helper enforces the retired all-host/interleaved llama-bench shape.  A4's
    best serving topology is node-0, no-numactl, and this explicit taskset.
    """
    return [
        "taskset", "-c", CPU_LIST, str(LLAMA_SERVER),
        "-m", str(MODEL), "-t", str(THREADS), "-c", str(CTX), "-np", "1",
        "-ub", str(UBATCH), "-ctk", "q8_0", "-ctv", "q8_0",
        "--flash-attn", "on", "--jinja", "--mlock", "--device", "none",
        "--device-draft", "none", "--reasoning", "off",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
        "--port", str(port), "--log-colors", "off",
    ]


def build_env() -> dict[str, str]:
    env = build_canonical_env({"KMP_BLOCKTIME": "10", "GGML_IQK_Q8_0": "1"})
    assert_canonical_env(env)
    if env.get("KMP_BLOCKTIME") != "10" or env.get("GGML_IQK_Q8_0") != "1":
        raise ReanchorRefusal("canonical serving environment is incomplete")
    return env


def host_memory_numa_snapshot() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    numa_path = Path("/proc/sys/kernel/numa_balancing")
    return {
        "captured_at": utc_now(),
        "meminfo": meminfo,
        "numa_balancing": (
            numa_path.read_text(encoding="utf-8").strip()
            if numa_path.is_file()
            else None
        ),
    }


def prepare_cold_cache() -> dict[str, Any]:
    """Synchronize storage and drop page cache, failing closed."""
    before = host_memory_numa_snapshot()
    sync_command = ["sync"]
    sync_result = subprocess.run(
        sync_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if sync_result.returncode != 0:
        raise ReanchorRefusal(
            f"sync failed during cold-cache preparation: {sync_result.stdout.strip()}"
        )
    drop_command = ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"]
    drop_result = subprocess.run(
        drop_command,
        check=False,
        text=True,
        input="3\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    record = {
        "started_at": before["captured_at"],
        "finished_at": utc_now(),
        "sync": {
            "command": sync_command,
            "returncode": sync_result.returncode,
            "output": sync_result.stdout,
        },
        "drop_caches": {
            "command": drop_command,
            "input": "3",
            "returncode": drop_result.returncode,
            "output": drop_result.stdout,
        },
        "before": before,
        "after": host_memory_numa_snapshot(),
    }
    if drop_result.returncode != 0:
        raise ReanchorRefusal(
            "drop_caches=3 unavailable; refusing measurement: "
            + drop_result.stdout.strip()
        )
    return record


def _region_status() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [str(REGION_LOCK), "status", "--json"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
    )
    if proc.returncode:
        raise ReanchorRefusal(f"cannot read region-lock status: {proc.stdout.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ReanchorRefusal("region-lock status did not return JSON") from exc
    if not isinstance(payload, list):
        raise ReanchorRefusal("region-lock status must return a list")
    return payload


def ancestor_pids(pid: int | None = None) -> set[int]:
    """Return the current process and its Linux ancestor chain."""
    current = os.getpid() if pid is None else pid
    result: set[int] = set()
    while current > 0 and current not in result:
        result.add(current)
        try:
            fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
            current = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
    return result


def verify_held_footprint(
    *,
    claim_tag: str,
    claim_role: str = "bench",
) -> list[dict[str, Any]]:
    """Prove q0,q1 are held by this runner's region-lock ancestor."""
    if not claim_tag.strip():
        raise ReanchorRefusal("--region-claim-tag must be non-empty")
    rows = _region_status()
    by_region = {
        str(row.get("region")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("region"), str)
    }
    if set(by_region) != {"q0", "q1", "q2", "q3"}:
        raise ReanchorRefusal("region-lock status does not contain q0..q3 exactly once")
    ancestors = ancestor_pids()
    owner_pids: set[int] = set()
    for region in PHYSICAL_REGIONS:
        row = by_region[region]
        if row.get("global_held") is not True:
            raise ReanchorRefusal(
                "execution requires region-lock coverage for the actual A4 CPU "
                f"footprint {list(PHYSICAL_REGIONS)}; {region} is not held. "
                "A q2-only claim is not resource protection."
            )
        holders = row.get("holders")
        if not isinstance(holders, list):
            raise ReanchorRefusal(f"{region} has no attributable lock holder")
        matches = [
            holder
            for holder in holders
            if isinstance(holder, dict)
            and holder.get("role") == claim_role
            and holder.get("request_tag") == claim_tag
            and set(holder.get("regions") or []) == set(PHYSICAL_REGIONS)
        ]
        if len(matches) != 1:
            raise ReanchorRefusal(
                f"{region} is not held by exactly one {claim_role!r}/{claim_tag!r} "
                f"holder covering {list(PHYSICAL_REGIONS)}"
            )
        try:
            holder_pid = int(matches[0]["pid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReanchorRefusal(f"{region} holder has no valid PID") from exc
        if holder_pid not in ancestors:
            raise ReanchorRefusal(
                f"{region} holder pid {holder_pid} is not in this runner's ancestor chain"
            )
        owner_pids.add(holder_pid)
    if len(owner_pids) != 1:
        raise ReanchorRefusal(
            f"q0 and q1 are attributed to different holder PIDs: {sorted(owner_pids)}"
        )
    return rows


def _http_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise ReanchorRefusal(f"server request returned HTTP {response.status}")
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ReanchorRefusal(f"server request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ReanchorRefusal("server response is not a JSON object")
    return result


def wait_for_health(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            pass
        time.sleep(1)
    raise ReanchorRefusal(f"llama-server on port {port} did not become healthy")


def completion_payload(n_predict: int) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": (
            "Write a detailed technical explanation of deterministic CPU inference "
            "measurement. Continue until the requested length; do not summarize early."
        )}],
        "max_tokens": n_predict,
        "temperature": 0.3,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_sample(
    response: dict[str, Any],
    ordinal: int,
    *,
    expected_tokens: int = N_PREDICT,
) -> DecodeSample:
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise ReanchorRefusal("response lacks timings")
    predicted_n = int(timings.get("predicted_n") or 0)
    predicted_tps = float(timings.get("predicted_per_second") or 0.0)
    prompt_n = int(timings.get("prompt_n") or 0)
    choices = response.get("choices")
    text = ""
    finish_reason = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = str(choices[0].get("finish_reason") or "")
        message = choices[0].get("message")
        if isinstance(message, dict):
            text = str(message.get("content") or "")
    if predicted_n != expected_tokens:
        raise ReanchorRefusal(
            f"decode sample {ordinal} returned {predicted_n} tokens; "
            f"expected exactly {expected_tokens}"
        )
    if finish_reason != "length":
        raise ReanchorRefusal(
            f"decode sample {ordinal} finish_reason={finish_reason!r}; expected 'length'"
        )
    if predicted_tps <= 0:
        raise ReanchorRefusal(f"long-decode sample {ordinal} has no positive predicted_per_second")
    return DecodeSample(
        ordinal,
        predicted_n,
        predicted_tps,
        prompt_n,
        len(text),
        finish_reason,
        dict(timings),
    )


def warmup_is_stable(samples: list[DecodeSample]) -> bool:
    if len(samples) < WARMUP_CONSECUTIVE:
        return False
    values = [
        sample.predicted_per_second
        for sample in samples[-WARMUP_CONSECUTIVE:]
    ]
    center = statistics.median(values)
    return center > 0 and (max(values) - min(values)) / center <= WARMUP_RELATIVE_TOLERANCE


def collect_warmup_samples(
    request: Any,
) -> list[DecodeSample]:
    samples: list[DecodeSample] = []
    for ordinal in range(1, WARMUP_MAX_ATTEMPTS + 1):
        response = request(completion_payload(WARMUP_TOKENS))
        samples.append(
            parse_sample(
                response,
                ordinal,
                expected_tokens=WARMUP_TOKENS,
            )
        )
        if warmup_is_stable(samples):
            return samples
    raise ReanchorRefusal(
        f"warmup failed to reach {WARMUP_CONSECUTIVE} consecutive positive "
        f"samples within {WARMUP_RELATIVE_TOLERANCE:.0%} after "
        f"{WARMUP_MAX_ATTEMPTS} attempts"
    )


def expected_affinity() -> set[int]:
    return parse_cpu_list(CPU_LIST)


def parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ReanchorRefusal("empty CPU-affinity segment")
        start, end = (part.split("-", 1) + [part])[:2]
        try:
            lower, upper = int(start), int(end)
        except ValueError as exc:
            raise ReanchorRefusal(f"invalid CPU-affinity segment {part!r}") from exc
        if lower < 0 or upper < lower:
            raise ReanchorRefusal(f"invalid CPU-affinity range {part!r}")
        result.update(range(lower, upper + 1))
    return result


def _thread_affinity_from_status(status_path: Path) -> set[int]:
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReanchorRefusal(f"cannot read thread affinity status: {status_path}") from exc
    for line in lines:
        if line.startswith("Cpus_allowed_list:"):
            value = line.split(":", 1)[1].strip()
            if not value:
                raise ReanchorRefusal(f"thread affinity status is empty: {status_path}")
            return parse_cpu_list(value)
    raise ReanchorRefusal(f"thread affinity status lacks Cpus_allowed_list: {status_path}")


def read_thread_affinities(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[int, set[int]]:
    """Read every server thread's kernel affinity mask from procfs."""
    task_dir = proc_root / str(pid) / "task"
    try:
        tids = sorted(int(path.name) for path in task_dir.iterdir() if path.name.isdigit())
    except OSError as exc:
        raise ReanchorRefusal(f"cannot enumerate server threads: {task_dir}") from exc
    if pid not in tids:
        raise ReanchorRefusal(f"server leader thread {pid} is absent from {task_dir}")
    if len(tids) < 2:
        raise ReanchorRefusal(
            f"server pid {pid} has no worker threads; cannot prove OpenMP coverage"
        )
    return {
        tid: _thread_affinity_from_status(task_dir / str(tid) / "status")
        for tid in tids
    }


def verify_live_affinity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    thread_affinities = read_thread_affinities(pid, proc_root=proc_root)
    expected = expected_affinity()
    observed_union = set().union(*thread_affinities.values())
    outside = {
        tid: sorted(cpus - expected)
        for tid, cpus in thread_affinities.items()
        if cpus - expected
    }
    if outside:
        raise ReanchorRefusal(
            f"live llama-server pid {pid} has thread affinity outside expected mask: {outside}"
        )
    if observed_union != expected:
        raise ReanchorRefusal(
            f"live llama-server pid {pid} thread-affinity union mismatch: "
            f"expected {sorted(expected)}, got {sorted(observed_union)}"
        )
    worker_affinities = {tid: cpus for tid, cpus in thread_affinities.items() if tid != pid}
    worker_union = set().union(*worker_affinities.values())
    if worker_union != expected:
        raise ReanchorRefusal(
            f"live llama-server pid {pid} worker thread coverage mismatch: "
            f"expected {sorted(expected)}, got {sorted(worker_union)}"
        )
    return {
        "server_pid": pid,
        "expected_cpu_list": CPU_LIST,
        "expected_cpus": sorted(expected),
        "thread_affinities": {
            str(tid): sorted(cpus) for tid, cpus in thread_affinities.items()
        },
        "thread_union": sorted(observed_union),
        "worker_thread_ids": sorted(worker_affinities),
        "worker_thread_union": sorted(worker_union),
    }


def process_tree_pids(root_pid: int) -> set[int]:
    result = {root_pid}
    proc = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    if proc.returncode:
        raise ReanchorRefusal("cannot enumerate server process tree")
    pairs: list[tuple[int, int]] = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            pairs.append((int(fields[0]), int(fields[1])))
    changed = True
    while changed:
        changed = False
        for pid, parent in pairs:
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def verify_exclusive_server(pid: int) -> dict[str, Any]:
    allowed = process_tree_pids(pid)
    processes = find_llama_processes()
    try:
        inference_pids = {int(row["pid"]) for row in processes}
    except (KeyError, TypeError, ValueError) as exc:
        raise ReanchorRefusal("live inference process inventory is malformed") from exc
    if pid not in inference_pids:
        raise ReanchorRefusal(f"expected llama-server pid {pid} is absent")
    competitors = sorted(inference_pids - allowed)
    if competitors:
        raise ReanchorRefusal(
            f"competing inference processes detected before request: {competitors}"
        )
    return {
        "captured_at": utc_now(),
        "server_pid": pid,
        "allowed_process_tree_pids": sorted(allowed),
        "live_inference_processes": processes,
        "live_affinity": verify_live_affinity(pid),
    }


def _run_identity() -> dict[str, Any]:
    branch = run_capture([str(GIT), "-C", str(LLAMA_ROOT), "branch", "--show-current"])
    commit = run_capture([str(GIT), "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"])
    if branch.strip() != EXPECTED_LLAMA_BRANCH or commit.strip() != EXPECTED_LLAMA_COMMIT:
        raise ReanchorRefusal(
            f"frozen-v8 identity mismatch: branch={branch!r} commit={commit!r}"
        )
    env = build_env()
    assert_binary_resolves_correctly(
        str(LLAMA_SERVER),
        EXPECTED_LIBS_V6_IQK,
        env=env,
    )
    return {
        "llama_branch": branch.strip(), "llama_commit": commit.strip(),
        "binary_sha256": sha256(LLAMA_SERVER), "model_sha256": sha256(MODEL),
        "instrument_sha256": sha256(Path(__file__).resolve()),
        "binary_version": run_capture([str(LLAMA_SERVER), "--version"]),
        "binary_realpath": str(LLAMA_SERVER.resolve(strict=True)),
        "expected_binary_realpath": str(LLAMA_SERVER),
        "expected_shared_libraries": list(EXPECTED_LIBS_V6_IQK),
        "research_commit": run_capture([str(GIT), "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
    }


def proposal(
    evidence: dict[str, Any],
    *,
    evidence_file_sha256: str,
) -> dict[str, Any]:
    """Return a non-applying registry proposal; no registry path is opened."""
    return {
        "schema": "epyc.registry_patch_proposal.v1",
        "mode": "proposal_only",
        "must_not_apply_automatically": True,
        "apply_eligibility": (
            "review_required"
            if evidence.get("decision_grade") is True
            else "candidate_protocol_pending_ratification"
        ),
        "metric_semantics": (
            "llama-server timings.predicted_per_second; median and MAD of five "
            "exact 512-token, single-request, stability-warmed production-shaped decodes"
        ),
        "not_comparable_to": ["llama-bench tg512", "P-BENCH-3 short task-rate"],
        "intended_registry_field_targets": [
            "roles.frontdoor.performance.baseline_tps",
            "roles.frontdoor.performance.benchmark_date",
        ],
        "evidence_sha256": evidence_file_sha256,
        "evidence_hash_semantics": "exact_written_evidence_json_bytes",
        "requires_human_review": True,
    }


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "protocol_status": "candidate_protocol_pending_ratification",
        "decision_grade": False,
        "server_command": build_server_command(port=args.port),
        "env": {key: build_env()[key] for key in sorted((*CANONICAL_OMP_ENV, "KMP_BLOCKTIME", "GGML_IQK_Q8_0"))},
        "required_regions": list(PHYSICAL_REGIONS), "reps": REPS,
        "n_predict": N_PREDICT, "metric": "timings.predicted_per_second",
        "protocol_contract": protocol_contract(),
        "registry_mutation": False,
        "proposal_apply_eligible": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_content_hash_manifest(staging: Path) -> Path:
    manifest_path = staging / "content-hashes.json"
    records = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path != manifest_path and path.name != "COMPLETE":
            records.append(
                {
                    "path": str(path.relative_to(staging)),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_json(
        manifest_path,
        {
            "schema": "epyc.content_hash_manifest.v1",
            "files": records,
        },
    )
    return manifest_path


def atomic_publish(staging: Path, output: Path) -> None:
    """Durably rename a complete staged directory or remove it on failure."""
    if output.exists():
        raise ReanchorRefusal(f"terminal output already exists: {output}")
    published = False
    parent_fd: int | None = None
    try:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        parent_fd = os.open(staging.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(parent_fd)
        os.replace(staging, output)
        published = True
        os.fsync(parent_fd)
    except Exception as publish_error:
        cleanup_target = output if published else staging
        shutil.rmtree(cleanup_target, ignore_errors=True)
        if cleanup_target.exists():
            raise ReanchorRefusal(
                f"publish failed and cleanup could not remove {cleanup_target}"
            ) from publish_error
        if parent_fd is not None:
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def execute(args: argparse.Namespace) -> Path:
    if not args.i_have_operator_grant:
        raise ReanchorRefusal("--execute requires --i-have-operator-grant")
    if args.reviewed_protocol_attestation is None:
        raise ReanchorRefusal(
            "--execute requires --reviewed-protocol-attestation; "
            "candidate protocol remains pending human ratification"
        )
    if not args.region_claim_tag:
        raise ReanchorRefusal("--execute requires --region-claim-tag")
    if not LLAMA_SERVER.is_file() or not MODEL.is_file():
        raise ReanchorRefusal("frozen-v8 llama-server or A4 model is missing")
    protocol_attestation = validate_protocol_attestation(
        args.reviewed_protocol_attestation
    )
    region_before = verify_held_footprint(
        claim_tag=args.region_claim_tag,
        claim_role=args.region_claim_role,
    )
    ensure_clean_runtime()
    attestation = collect_attestation()
    warnings = host_health_warnings(attestation)
    if warnings:
        raise ReanchorRefusal("host-health preconditions failed: " + "; ".join(warnings))
    identity = _run_identity()
    cache_preparation = prepare_cold_cache()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / args.run_id
    if output.exists():
        raise ReanchorRefusal(f"terminal output already exists: {output}")
    staging = output_root / f".{args.run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    command = build_server_command(port=args.port)
    env = build_env()
    write_json(staging / "server-command.json", command)
    proc = None
    samples: list[DecodeSample] = []
    warmup_samples: list[DecodeSample] = []
    request_witnesses: list[dict[str, Any]] = []
    teardown: dict[str, Any] | None = None
    try:
        try:
            proc = start_server(command, env, staging / "server.log")
            wait_for_health(args.port, args.startup_timeout)
            live_affinity = verify_live_affinity(proc.pid)

            def request(payload: dict[str, Any]) -> dict[str, Any]:
                request_witnesses.append(verify_exclusive_server(proc.pid))
                return _http_json(
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    payload,
                    args.request_timeout,
                )

            warmup_responses: list[dict[str, Any]] = []

            def warmup_request(payload: dict[str, Any]) -> dict[str, Any]:
                response = request(payload)
                warmup_responses.append(response)
                return response

            warmup_samples = collect_warmup_samples(warmup_request)
            for ordinal, response in enumerate(warmup_responses, start=1):
                write_json(staging / f"warmup-response-{ordinal}.json", response)
            for ordinal in range(1, REPS + 1):
                response = request(completion_payload(N_PREDICT))
                sample = parse_sample(response, ordinal)
                samples.append(sample)
                write_json(staging / f"response-{ordinal}.json", response)
        finally:
            if proc is not None:
                teardown = stop_server(proc)

        region_after = verify_held_footprint(
            claim_tag=args.region_claim_tag,
            claim_role=args.region_claim_role,
        )
        if len(samples) != REPS:
            raise ReanchorRefusal("incomplete long-decode sample set")
        expected_witnesses = len(warmup_samples) + REPS
        if len(request_witnesses) != expected_witnesses:
            raise ReanchorRefusal(
                f"request witness count {len(request_witnesses)} does not match "
                f"warmups+measured {expected_witnesses}"
            )
        values = [sample.predicted_per_second for sample in samples]
        median_tps = statistics.median(values)
        mad_tps = statistics.median(
            [abs(value - median_tps) for value in values]
        )
        evidence = {
            "schema": "epyc.fg4b_a4_cpu_optimized_server_evidence.v2",
            "created_at": utc_now(),
            "protocol_id": PROTOCOL_ID,
            "protocol_status": "ratified_receipt_validated",
            "protocol_attestation": protocol_attestation,
            "metric": "llama-server timings.predicted_per_second",
            "metric_direction": "higher_is_better",
            "metric_semantics": (
                "median and MAD of five server-reported decode tokens/s samples "
                "from exact 512-token, np=1 chat completions after stability warmup"
            ),
            "median_tokens_per_second": median_tps,
            "median_absolute_deviation_tokens_per_second": mad_tps,
            "mean_tokens_per_second_observation": statistics.mean(values),
            "samples": [asdict(sample) for sample in samples],
            "warmup": {
                "disposition": "cold_samples_excluded_until_stable",
                "stable": True,
                "attempts": len(warmup_samples),
                "contract": protocol_contract()["warmup"],
                "samples": [asdict(sample) for sample in warmup_samples],
                "request_witnesses": request_witnesses[:len(warmup_samples)],
            },
            "top_serving_spec": {
                "cpu_list": CPU_LIST,
                "live_affinity": live_affinity,
                "threads": THREADS,
                "ctx": CTX,
                "ubatch": UBATCH,
                "np": 1,
                "native_mtp_draft_max": 4,
                "numactl": "none",
                "reasoning": "off",
                "ignore_eos": True,
            },
            "topology_derivation": {
                "source": (
                    "/mnt/raid0/llm/epyc-orchestrator/src/runtime/"
                    "instance_topology.py"
                ),
                "cpu_list": CPU_LIST,
                "physical_regions": list(PHYSICAL_REGIONS),
                "rule": (
                    "hyper-thread siblings are stripped before mapping "
                    "physical cores to atomic regions"
                ),
            },
            "runtime_identity": identity,
            "host_attestation": attestation,
            "cold_cache_preparation": cache_preparation,
            "measured_request_witnesses": request_witnesses[len(warmup_samples):],
            "cold_warm_disposition": {
                "cold_start_samples_used_in_metric": False,
                "measurement_started_after_stability_gate": True,
                "host_uptime_seconds": attestation.get("uptime_seconds"),
                "scaling_governors": attestation.get("scaling_governors"),
                "numa_balancing": attestation.get("numa_balancing"),
            },
            "environment": {
                key: env[key]
                for key in sorted(
                    (*CANONICAL_OMP_ENV, "KMP_BLOCKTIME", "GGML_IQK_Q8_0")
                )
            },
            "region_claim": {
                "tag": args.region_claim_tag,
                "role": args.region_claim_role,
                "regions": list(PHYSICAL_REGIONS),
                "status_before": region_before,
                "status_after": region_after,
            },
            "teardown": teardown,
            "decision_grade": True,
            "proposal_only": True,
        }
        evidence_path = staging / "evidence.json"
        write_json(evidence_path, evidence)
        write_json(
            staging / "registry-patch-proposal.json",
            proposal(
                evidence,
                evidence_file_sha256=sha256(evidence_path),
            ),
        )
        write_content_hash_manifest(staging)
        (staging / "COMPLETE").write_text("", encoding="utf-8")
        atomic_publish(staging, output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run inference; default is dry-run")
    parser.add_argument("--i-have-operator-grant", action="store_true")
    parser.add_argument(
        "--reviewed-protocol-attestation",
        type=Path,
        default=None,
        help="human-reviewed receipt binding the exact protocol and instrument hash",
    )
    parser.add_argument(
        "--region-claim-tag",
        default=None,
        help="exact tag passed to the enclosing region-lock invocation",
    )
    parser.add_argument(
        "--region-claim-role",
        default="bench",
        help="exact role passed to the enclosing region-lock invocation",
    )
    parser.add_argument("--run-id", default=f"fg4b-a4-cpu-optimized-server-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps(dry_run_payload(args), indent=2, sort_keys=True))
        return 0
    try:
        output = execute(args)
    except ReanchorRefusal as exc:
        print(f"FG-4b optimized server re-anchor refused: {exc}", file=sys.stderr)
        return 2
    print(f"FG-4b optimized server evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
