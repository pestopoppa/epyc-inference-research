#!/usr/bin/env python3
"""P-BENCH-3 multi-server NUMA x np sweep for E5 batched-decode cells.

Sibling of server_np_sweep.py (the E1 single-server harness). One invocation
runs the cells of ONE model group (drop_caches + re-warm between models is an
OPERATOR step, never harness-run): per cell it launches N llama-server
instances (per-instance `taskset -c` pinning; `numactl --interleave=all` ONLY
when the instance's manifest policy says so), verifies LIVE affinity via the
orchestrator affinity_preflight cell-manifest mode as a HARD gate, drives the
pinned P-BENCH-3 prompt batch closed-loop across N x K streams, and tears the
instances down with a ps-verified kill.

Cells are described by e5-cell-manifest/1 JSON files (schema owner:
e5_cell_manifests.py — the ONLY cross-owner import is
{SCHEMA_VERSION, validate_cell_manifest}). The env is composed from
scripts/lib/canonical_recipe.py constants (GGML_IQK=1 is a member of
CANONICAL_OMP_ENV) plus KMP_BLOCKTIME=10 — there is deliberately NO private
env copy in this module (audit 2a recipe-drift nit).

DEFAULT IS DRY-RUN: without --execute nothing is ever spawned. --execute
additionally requires --i-have-operator-grant (benches run only in
operator-approved quiet windows).

Offline summarizer: --summarize-run RUN_DIR emits the iso-T comparison tables
and the pre-registered R1-R4 rule evaluation (R3 refuses to price the eval
lane until a fresh current-arm baseline row is supplied).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

_BENCHMARK_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCHMARK_DIR.parents[1]
for _p in (str(_REPO_ROOT / "scripts" / "lib"), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from canonical_recipe import (  # noqa: E402
    CANONICAL_OMP_ENV,
    FREQ_BOOST_MIN_CORES,
    FREQ_BOOST_THRESHOLD_KHZ,
    assert_canonical_env,
    build_canonical_env,
)
from server_np_sweep import (  # noqa: E402
    collect_attestation,
    ensure_clean_runtime,
    host_health_warnings,
    percentile,
    run_capture,
    start_server,
    utc_now,
    wait_for_health,
    write_jsonl,
)

try:
    import httpx
except ImportError:  # tests and dry-run never need it; --execute does
    httpx = None


EXPECTED_SCHEMA_VERSION = "e5-cell-manifest/1"
EXPECTED_PROTOCOL_ID = "P-BENCH-3"
BENCH_PORT_MIN = 19000
BENCH_PORT_MAX = 19999
DEFAULT_LLAMA_SERVER = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-server")
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "data" / "batched_decode"
DEFAULT_AFFINITY_PREFLIGHT = Path(
    "/mnt/raid0/llm/epyc-orchestrator/scripts/server/affinity_preflight.py"
)
KMP_BLOCKTIME_VALUE = "10"
TRIM_DEFINITION = (
    "steady-state window = requests whose full lifetime lies within "
    "[first request completion, last request start]; trimmed tasks/hour = "
    "steady request count / window duration"
)
E1_COMPARABILITY_CAVEAT = (
    "E5 K=1 cells are NOT byte-comparable to E1 rows (different -c convention: "
    "E5 uses -c max(8192, per_stream_ctx*np), E1 used fixed -c 32768) — "
    "direction cross-check only."
)
R1_MARGIN = 0.10
R2_P95_RATIO = 3.0
R2_P95_ABSOLUTE_MS = 60_000.0
R2_HOLD_FRACTION = 0.70
R4_PEAK_FRACTION = 0.90
GARBAGE_GATE_MAX_PARSE_FAILURES = 2
ISO_T_WHOLE_MACHINE = (8, 16, 32)   # {C1b@T/2 vs C3@T/4}; gemma-style: {C1full@T vs C3@T/4}
ISO_T_HALF_MACHINE = (16, 32)       # {C1@T vs C2@T/2}
SCALING_K = (4, 8)                  # {C1@K vs C1b@K}
KVU_PROBE_ESCALATION = 0.05         # M06: >=5% split-vs-unified probe delta escalates
# Scout paired-probe variants must never be conflated with the canonical
# (config_id, np) cell in R1/R2/R4 picks (review F3/F5): tagged via
# stage_b_families (preferred) with a cell_id-suffix fallback for rows that
# predate the tag propagation.
VARIANT_FAMILY_TAGS = ("scout_kvu_probe", "scout_dense_c1_shape_pair")
VARIANT_CELL_ID_SUFFIXES = ("-kvu", "-scout-full")
# The pre-registered Stage-B grid has no C1b@1 / C2@1 anchor; R2 falls back to
# the same-per-instance-shape K=1 baseline with the substitution recorded
# (review F5: C1b instances are halves like C1@1; C2 instances are quarters
# like C3@1).
R2_K1_PROXY = {"C1b": "C1", "C2": "C3"}
TRIM_BASIS_CAVEAT = (
    "aggregate tasks/hour falls back from trimmed to raw when the steady-state "
    "window is empty (structural at high K: the 43-prompt closed loop leaves "
    "few second-round requests); raw includes ramp+drain and systematically "
    "understates steady throughput — every R1/R2/R4 comparison records the "
    "per-cell metric basis and flags mixed-basis pairs (review F3)"
)


# ---------------------------------------------------------------------------
# Cell manifests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    cpu_list: str
    port: int
    threads: int
    numactl_policy: str


@dataclass(frozen=True)
class Cell:
    path: Path
    manifest: dict[str, Any]
    instances: tuple[Instance, ...]

    @property
    def cell_id(self) -> str:
        return str(self.manifest.get("cell_id") or self.path.stem)

    @property
    def model_key(self) -> str:
        return str(self.manifest.get("model_key") or "")

    @property
    def model_path(self) -> str:
        return str(self.manifest.get("model_path") or "")

    @property
    def config_id(self) -> str:
        return str(self.manifest.get("config_id") or "")

    @property
    def np(self) -> int:
        return int(self.manifest.get("np") or 0)

    @property
    def spec_dec(self) -> dict[str, Any]:
        block = self.manifest.get("spec_dec")
        return block if isinstance(block, dict) else {}

    @property
    def kv(self) -> dict[str, Any]:
        block = self.manifest.get("kv")
        return block if isinstance(block, dict) else {}

    @property
    def prompt_caps(self) -> dict[str, Any]:
        block = self.manifest.get("prompt_caps")
        return block if isinstance(block, dict) else {}

    @property
    def prompt_batch(self) -> dict[str, Any]:
        block = self.manifest.get("prompt_batch")
        return block if isinstance(block, dict) else {}

    @property
    def decision_grade_intent(self) -> bool:
        return bool(self.manifest.get("decision_grade_intent"))

    @property
    def total_streams(self) -> int:
        return len(self.instances) * self.np


def _manifest_interface() -> Any:
    """Import the manifests owner's schema interface (cross-owner contract).

    The harness consumes ONLY {SCHEMA_VERSION, validate_cell_manifest}; tests
    monkeypatch this accessor so harness tests never depend on the manifests
    owner's implementation.
    """
    import e5_cell_manifests

    return e5_cell_manifests


def load_cell_manifest(path: Path) -> Cell:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"cell manifest is not a JSON object: {path}")
    raw_instances = manifest.get("instances")
    instances: list[Instance] = []
    if isinstance(raw_instances, list):
        for row in raw_instances:
            if not isinstance(row, dict):
                raise RuntimeError(f"instance row is not an object: {path}")
            instances.append(
                Instance(
                    cpu_list=str(row.get("cpu_list") or ""),
                    port=int(row.get("port") or 0),
                    threads=int(row.get("threads") or 0),
                    numactl_policy=str(row.get("numactl_policy") or "none"),
                )
            )
    return Cell(path=path, manifest=manifest, instances=tuple(instances))


def harness_refusals(cells: list[Cell]) -> list[str]:
    """Harness-side fail-closed checks (independent of the schema validator)."""
    errors: list[str] = []
    if not cells:
        return ["no cell manifests supplied"]
    model_keys = sorted({cell.model_key for cell in cells})
    if len(model_keys) != 1:
        errors.append(
            "mixed model_keys in one invocation: "
            f"{model_keys} — one invocation = one model group (drop_caches + "
            "re-warm between models is an operator step)"
        )
    first_qids = cells[0].prompt_batch.get("qids")
    for cell in cells:
        name = cell.path.name
        version = cell.manifest.get("schema_version")
        if version != EXPECTED_SCHEMA_VERSION:
            errors.append(
                f"{name}: unknown schema_version {version!r} "
                f"(expected {EXPECTED_SCHEMA_VERSION!r}); refusing (fail closed on drift)"
            )
        protocol = cell.manifest.get("protocol_id")
        if protocol != EXPECTED_PROTOCOL_ID:
            errors.append(
                f"{name}: protocol_id {protocol!r} is not {EXPECTED_PROTOCOL_ID!r}"
            )
        if not cell.instances:
            errors.append(f"{name}: no instances")
        for inst in cell.instances:
            if not (BENCH_PORT_MIN <= inst.port <= BENCH_PORT_MAX):
                errors.append(
                    f"{name}: port {inst.port} outside bench range "
                    f"[{BENCH_PORT_MIN},{BENCH_PORT_MAX}] — prod ports are refused"
                )
            if inst.numactl_policy not in ("none", "interleave=all"):
                errors.append(
                    f"{name}: unknown numactl_policy {inst.numactl_policy!r}"
                )
        if cell.np <= 0:
            errors.append(f"{name}: np must be positive")
        selection = cell.prompt_batch.get("selection")
        if selection != "pinned_qids":
            errors.append(
                f"{name}: prompt_batch.selection {selection!r} is not 'pinned_qids' — "
                "re-sampling from the rebuilt pool is forbidden (E7 pool drift)"
            )
        if cell.prompt_batch.get("qids") != first_qids:
            errors.append(
                f"{name}: prompt_batch.qids differ across cells in one invocation"
            )
        cap = cell.prompt_caps.get("max_total_in_flight")
        if isinstance(cap, int) and cell.total_streams > cap:
            errors.append(
                f"{name}: len(instances)*np = {cell.total_streams} exceeds "
                f"max_total_in_flight {cap}"
            )
    return errors


def revalidate_cells(cells: list[Cell]) -> list[str]:
    """Re-validate every manifest via the schema owner's validator (fail closed)."""
    try:
        schema = _manifest_interface()
    except ImportError:
        return [
            "e5_cell_manifests is not importable; cannot re-validate cell "
            "manifests — refusing (the schema owner's validator is a hard gate)"
        ]
    if getattr(schema, "SCHEMA_VERSION", None) != EXPECTED_SCHEMA_VERSION:
        return [
            f"e5_cell_manifests.SCHEMA_VERSION={getattr(schema, 'SCHEMA_VERSION', None)!r} "
            f"does not match harness expectation {EXPECTED_SCHEMA_VERSION!r}; "
            "refusing (schema drift)"
        ]
    errors: list[str] = []
    for cell in cells:
        for violation in schema.validate_cell_manifest(cell.manifest):
            errors.append(f"{cell.path.name}: {violation}")
    return errors


# ---------------------------------------------------------------------------
# Prompt batch (pinned-qid replay)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSpec:
    qid: str
    suite: str
    prompt: str


def load_pinned_prompts(pool_path: Path, cell: Cell) -> list[PromptSpec]:
    """Replay the pinned qid batch from the question pool (never re-sample)."""
    batch = cell.prompt_batch
    qids = batch.get("qids")
    if not isinstance(qids, list) or not qids:
        raise RuntimeError(f"{cell.path.name}: prompt_batch.qids missing or empty")
    max_chars = cell.prompt_caps.get("max_prompt_chars")
    wanted = {str(qid) for qid in qids}
    found: dict[str, PromptSpec] = {}
    with pool_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("__pool_metadata__"):
                continue
            qid = str(row.get("id") or "")
            if qid not in wanted or qid in found:
                continue
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise RuntimeError(f"pinned qid {qid} has no usable prompt text")
            context = row.get("context")
            full_prompt = (
                f"{context}\n\n{prompt}" if isinstance(context, str) and context else prompt
            )
            if isinstance(max_chars, int) and len(full_prompt) > max_chars:
                raise RuntimeError(
                    f"pinned qid {qid} prompt is {len(full_prompt)} chars, over the "
                    f"max_prompt_chars cap {max_chars} — refusing (fail-closed guard "
                    "against qid-pinning bypass / pool drift)"
                )
            found[qid] = PromptSpec(
                qid=qid,
                suite=str(row.get("suite") or "unknown"),
                prompt=full_prompt,
            )
    missing = [str(qid) for qid in qids if str(qid) not in found]
    if missing:
        raise RuntimeError(
            f"pinned qids missing from pool {pool_path}: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    return [found[str(qid)] for qid in qids]


# ---------------------------------------------------------------------------
# Command + env construction
# ---------------------------------------------------------------------------


def build_instance_command(*, binary: Path, cell: Cell, inst: Instance) -> list[str]:
    """Per-instance launch command: taskset pinning + per-instance numactl policy.

    Deliberately NOT canonical_recipe.CANONICAL_PREFIX (that is the
    single-instance full-machine 0-95 + interleave=all recipe); halves and
    quarters run taskset-only (first-touch locality, matching stack_numa
    production wiring).
    """
    cmd = ["taskset", "-c", inst.cpu_list]
    if inst.numactl_policy == "interleave=all":
        cmd += ["numactl", "--interleave=all"]
    elif inst.numactl_policy != "none":
        raise RuntimeError(f"unknown numactl_policy {inst.numactl_policy!r}")
    kv = cell.kv
    cmd += [
        str(binary),
        "-m",
        cell.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(inst.port),
        "-np",
        str(cell.np),
        "-c",
        str(int(cell.manifest.get("ctx") or 0)),
        "-t",
        str(inst.threads),
        "-ub",
        str(int(cell.manifest.get("ubatch_size") or 0)),
        "-ctk",
        str(kv.get("type_k") or "q8_0"),
        "-ctv",
        str(kv.get("type_v") or "q8_0"),
    ]
    if kv.get("flash_attn", True):
        cmd += ["--flash-attn", "on"]
    if cell.manifest.get("jinja", True):
        cmd.append("--jinja")
    if cell.manifest.get("mlock", True):
        cmd.append("--mlock")
    # GPU contamination guard: the v7 binary is HIP-capable and would otherwise
    # auto-select ROCm0 (orchestrator_stack._append_cpu_only_device_args).
    cmd += ["--device", "none"]
    spec = cell.spec_dec
    if spec.get("enabled"):
        draft_model = spec.get("draft_model_path")
        if isinstance(draft_model, str) and draft_model and (
            os.path.realpath(draft_model) != os.path.realpath(cell.model_path)
        ):
            cmd += ["-md", draft_model]
        if spec.get("spec_type"):
            cmd += ["--spec-type", str(spec["spec_type"])]
        if spec.get("draft_max") is not None:
            cmd += ["--spec-draft-n-max", str(spec["draft_max"])]
        if spec.get("draft_min") is not None:
            cmd += ["--spec-draft-n-min", str(spec["draft_min"])]
        if spec.get("draft_p_min") is not None:
            cmd += ["--draft-p-min", str(spec["draft_p_min"])]
        if spec.get("draft_p_split") is not None:
            cmd += ["--draft-p-split", str(spec["draft_p_split"])]
        if spec.get("threads_draft") is not None:
            cmd += ["--threads-draft", str(spec["threads_draft"])]
        ngram_mod = spec.get("ngram_mod")
        if isinstance(ngram_mod, dict):
            for key, flag in (
                ("n_min", "--spec-ngram-mod-n-min"),
                ("n_max", "--spec-ngram-mod-n-max"),
                ("n_match", "--spec-ngram-mod-n-match"),
            ):
                if ngram_mod.get(key) is not None:
                    cmd += [flag, str(ngram_mod[key])]
        cmd += ["--device-draft", "none"]
    if kv.get("kv_unified"):
        cmd.append("-kvu")
    cmd += ["--log-colors", "off"]
    return cmd


def build_cell_env(llama_server: Path) -> dict[str, str]:
    """Canonical env for every instance of a cell — no private env copies.

    build_canonical_env carries the canonical OMP stack + GGML_IQK=1 + the
    LLVM20 libomp LD_LIBRARY_PATH prepend; KMP_BLOCKTIME=10 preserves the E1
    harness idle-spin fix (feedback_ik_llamacpp_omp_idle_spin).
    """
    env = build_canonical_env(extra_vars={"KMP_BLOCKTIME": KMP_BLOCKTIME_VALUE})
    bindir = str(llama_server.parent)
    parts = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
    if bindir not in parts:
        env["LD_LIBRARY_PATH"] = ":".join([bindir] + parts)
    return env


def check_env_expectation(env: dict[str, str], cell: Cell) -> list[str]:
    """Verify the composed env satisfies the manifest env_expectation block."""
    errors: list[str] = []
    expectation = cell.manifest.get("env_expectation")
    expectation = expectation if isinstance(expectation, dict) else {}
    expected_iqk = str(expectation.get("ggml_iqk") or "1")
    if env.get("GGML_IQK") != expected_iqk:
        errors.append(
            f"{cell.cell_id}: env GGML_IQK={env.get('GGML_IQK')!r} does not satisfy "
            f"env_expectation.ggml_iqk={expected_iqk!r} (v7 iqk runtime gate)"
        )
    expected_blocktime = str(expectation.get("kmp_blocktime") or KMP_BLOCKTIME_VALUE)
    if env.get("KMP_BLOCKTIME") != expected_blocktime:
        errors.append(
            f"{cell.cell_id}: env KMP_BLOCKTIME={env.get('KMP_BLOCKTIME')!r} does not "
            f"satisfy env_expectation.kmp_blocktime={expected_blocktime!r}"
        )
    return errors


SYS_CPU_ROOT = Path("/sys/devices/system/cpu")
FREQ_SAMPLE_INTERVAL_S = 10.0


def read_physical_core_freqs(base: Path = SYS_CPU_ROOT) -> list[int]:
    """Per-PHYSICAL-core scaling_cur_freq (SMT siblings deduped via topology).

    FREQ_BOOST_MIN_CORES (80) is calibrated "of 96" PHYSICAL cores
    (canonical_recipe); counting all 192 logical entries double-counts SMT
    siblings (review F1). Siblings share one clock domain, so the per-core
    frequency is the max over the sibling group.
    """
    groups: dict[str, int] = {}
    for cpu_dir in base.glob("cpu[0-9]*"):
        try:
            freq = int(
                (cpu_dir / "cpufreq" / "scaling_cur_freq").read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            continue
        key: str | None = None
        for name in ("core_cpus_list", "thread_siblings_list"):
            try:
                key = (cpu_dir / "topology" / name).read_text(encoding="utf-8").strip()
                break
            except OSError:
                continue
        if key is None:
            key = cpu_dir.name  # no topology info: count the logical cpu alone
        groups[key] = max(groups.get(key, 0), freq)
    return list(groups.values())


def cpu_freq_throttle_warnings(freqs: list[int] | None = None) -> list[str]:
    """UNDER-LOAD throttle gate: >= FREQ_BOOST_MIN_CORES physical cores boosting.

    canonical_recipe defines this as an under-load expectation ("under load,
    expect ALL 96 cores boosting above 2.5 GHz"). It is only meaningful while
    the machine is loaded — the E5 driver window, sampled via FreqSampler: on
    an idle quiet host amd-pstate parks cores near base clock and an
    instantaneous reading false-fails (review F1). Idle-time gating uses
    cpu_freq_static_warnings instead.
    """
    if freqs is None:
        freqs = read_physical_core_freqs()
    if not freqs:
        return ["no scaling_cur_freq readable; cannot verify CPU boost state"]
    boosting = sum(1 for freq in freqs if freq >= FREQ_BOOST_THRESHOLD_KHZ)
    if boosting < FREQ_BOOST_MIN_CORES:
        return [
            f"only {boosting}/{len(freqs)} physical cores at >= "
            f"{FREQ_BOOST_THRESHOLD_KHZ} kHz under load "
            f"(need {FREQ_BOOST_MIN_CORES}); host may be throttled "
            "(feedback_host_throttle_check)"
        ]
    return []


def cpu_freq_static_warnings(base: Path = SYS_CPU_ROOT) -> list[str]:
    """Idle-valid throttle indicators (run-start + per-cell precondition).

    The FREQ_BOOST gate is an under-load expectation, so instantaneous
    scaling_cur_freq on a quiet host cannot gate (review F1). At idle only
    STATIC throttle state is meaningful: the global cpufreq boost flag and
    per-CPU scaling_max_freq caps below the boost threshold. The under-load
    check runs during each cell's driver window (FreqSampler +
    cpu_freq_throttle_warnings).
    """
    warnings: list[str] = []
    boost = base / "cpufreq" / "boost"
    try:
        if boost.read_text(encoding="utf-8").strip() == "0":
            warnings.append(
                "cpufreq global boost flag is 0 — the host cannot boost "
                "(throttled state; feedback_host_throttle_check)"
            )
    except OSError:
        pass  # amd-pstate active mode exposes no global boost file
    capped = 0
    total = 0
    for path in base.glob("cpu[0-9]*/cpufreq/scaling_max_freq"):
        try:
            max_freq = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        total += 1
        if max_freq < FREQ_BOOST_THRESHOLD_KHZ:
            capped += 1
    if total and capped:
        warnings.append(
            f"{capped}/{total} logical CPUs have scaling_max_freq < "
            f"{FREQ_BOOST_THRESHOLD_KHZ} kHz — frequency caps below the boost "
            "threshold (feedback_host_throttle_check)"
        )
    return warnings


class FreqSampler:
    """Under-load CPU-frequency sampler for the per-cell throttle gate.

    Samples per-physical-core frequencies while the closed-loop driver holds
    the machine under load, keeping the best (most-boosting) sample; the gate
    (cpu_freq_throttle_warnings) is evaluated post-hoc against that sample.
    A cell shorter than the sample interval reports status "not_sampled"
    instead of a false idle reading (review F1: never gate on idle freqs).
    """

    def __init__(
        self,
        interval_s: float = FREQ_SAMPLE_INTERVAL_S,
        read_fn: Callable[[], list[int]] | None = None,
    ) -> None:
        self.interval_s = interval_s
        self._read = read_fn or read_physical_core_freqs
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.best_freqs: list[int] | None = None

    @staticmethod
    def _boost_count(freqs: list[int]) -> int:
        return sum(1 for freq in freqs if freq >= FREQ_BOOST_THRESHOLD_KHZ)

    def _loop(self) -> None:
        # Event.wait returns True once stop() is called: samples are only ever
        # taken while the driver still holds load, never on the idle host after.
        while not self._stop_event.wait(self.interval_s):
            freqs = self._read()
            if not freqs:
                continue
            self.samples += 1
            if self.best_freqs is None or (
                self._boost_count(freqs) > self._boost_count(self.best_freqs)
            ):
                self.best_freqs = freqs

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def result(self) -> dict[str, Any]:
        if self.samples == 0 or self.best_freqs is None:
            return {
                "status": "not_sampled",
                "samples": 0,
                "warnings": [],
                "note": (
                    "cell shorter than the sample interval; under-load boost "
                    "state not evaluated (never gated on idle frequencies)"
                ),
            }
        warnings = cpu_freq_throttle_warnings(self.best_freqs)
        return {
            "status": "warning" if warnings else "ok",
            "samples": self.samples,
            "boosting_physical_cores": self._boost_count(self.best_freqs),
            "n_physical_cores": len(self.best_freqs),
            "threshold_khz": FREQ_BOOST_THRESHOLD_KHZ,
            "min_boosting_cores": FREQ_BOOST_MIN_CORES,
            "warnings": warnings,
        }


# ---------------------------------------------------------------------------
# Teardown (ps-verified kill)
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    ps = subprocess.run(
        ["ps", "-p", str(pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ps.returncode == 0


def stop_instance(
    proc: subprocess.Popen[str],
    *,
    timeout_s: float = 30.0,
    poll_timeout_s: float = 20.0,
    poll_interval_s: float = 0.5,
) -> dict[str, Any]:
    """SIGTERM the process group, escalate to SIGKILL, then poll ps until GONE."""
    result: dict[str, Any] = {
        "pid": proc.pid,
        "signal": None,
        "returncode": None,
        "killed": False,
        "ps_verified_dead": False,
    }
    if proc.poll() is None:
        result["signal"] = "SIGTERM"
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            result["signal"] = "SIGKILL"
            result["killed"] = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
    result["returncode"] = proc.returncode
    deadline = time.monotonic() + poll_timeout_s
    while True:
        if not _pid_alive(proc.pid):
            result["ps_verified_dead"] = True
            return result
        if proc.poll() is not None:
            # The child was reaped by our own wait(): it cannot appear in ps
            # anymore, so a visible pid here is necessarily a RECYCLED pid on
            # an UNRELATED process — never signal it (pid-wraparound SIGKILL
            # hazard, review F6). The instance itself is verifiably dead.
            result["ps_verified_dead"] = True
            result["note"] = (
                "pid visible in ps after reap: recycled pid belongs to an "
                "unrelated process; not signaled"
            )
            return result
        if time.monotonic() >= deadline:
            break
        if not result["killed"]:
            # Still visible mid-poll and not yet reaped: escalate.
            result["signal"] = "SIGKILL"
            result["killed"] = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"instance pid {proc.pid} still visible in ps after SIGKILL escalation; "
        "refusing to proceed with surviving llama processes"
    )


def teardown_cell(
    procs: list[tuple[Instance, subprocess.Popen[str]]],
) -> list[dict[str, Any]]:
    """Stop every instance; raise after attempting all if any survives."""
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for inst, proc in procs:
        try:
            stop_result = stop_instance(proc)
        except Exception as exc:  # noqa: BLE001 — collect, then refuse
            failures.append(f"port {inst.port} pid {proc.pid}: {exc}")
            stop_result = {"pid": proc.pid, "error": str(exc), "ps_verified_dead": False}
        stop_result["port"] = inst.port
        results.append(stop_result)
    if failures:
        raise RuntimeError(
            "cell teardown left surviving instances:\n"
            + "\n".join(f"  {failure}" for failure in failures)
        )
    return results


# ---------------------------------------------------------------------------
# Affinity preflight (HARD cell gate, subprocess contract)
# ---------------------------------------------------------------------------


def run_affinity_preflight(
    *,
    preflight_script: Path,
    cell: Cell,
    pid_map: dict[str, int],
    artifact_path: Path,
    timeout_s: float = 120.0,
) -> tuple[int, dict[str, Any] | None, str]:
    """Invoke affinity_preflight.py --cell-manifest as a subprocess.

    Exit != 0 means the cell MUST be aborted (no warn-and-continue path).
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(preflight_script),
        "--cell-manifest",
        str(cell.path),
        "--pid-map",
        json.dumps(pid_map, sort_keys=True),
        "--output",
        str(artifact_path),
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    artifact: dict[str, Any] | None = None
    if artifact_path.exists():
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact = None
    return proc.returncode, artifact, proc.stdout


# ---------------------------------------------------------------------------
# Closed-loop N x K driver
# ---------------------------------------------------------------------------


@dataclass
class StreamRequestRecord:
    cell_id: str
    qid: str
    suite: str
    request_index: int
    stream_id: int
    instance_port: int
    success: bool
    start_s: float
    first_token_s: float | None
    end_s: float
    ttft_ms: float | None
    latency_ms: float
    predicted_tokens: int
    prompt_tokens: int
    predicted_tps: float
    draft_n: int | None = None
    draft_n_accepted: int | None = None
    http_status: int | None = None
    error: str = ""
    response_text: str = ""
    timings: dict[str, Any] = field(default_factory=dict)


def stream_instance_assignment(n_instances: int, np_level: int) -> list[int]:
    """Stream s is permanently bound to instance s % N."""
    if n_instances <= 0 or np_level <= 0:
        return []
    return [stream % n_instances for stream in range(n_instances * np_level)]


def run_cell_driver(
    *,
    cell: Cell,
    prompts: list[PromptSpec],
    send_fn: Callable[..., StreamRequestRecord],
    on_record: Callable[[StreamRequestRecord], None] | None = None,
) -> list[StreamRequestRecord]:
    """Closed-loop driver: N x K permanently-bound streams pull from one queue.

    The shared queue realizes "dispatch round-robin to the next free stream
    until each prompt completes exactly once" — a fixed work quantum identical
    across cells.
    """
    assignment = stream_instance_assignment(len(cell.instances), cell.np)
    work: Queue[tuple[int, PromptSpec]] = Queue()
    for index, prompt in enumerate(prompts):
        work.put((index, prompt))
    records: list[StreamRequestRecord] = []
    lock = threading.Lock()

    def stream_worker(stream_id: int) -> None:
        instance = cell.instances[assignment[stream_id]]
        while True:
            try:
                index, prompt = work.get_nowait()
            except Empty:
                return
            record = send_fn(
                stream_id=stream_id,
                instance=instance,
                prompt=prompt,
                request_index=index,
            )
            with lock:
                records.append(record)
                if on_record is not None:
                    on_record(record)

    total_streams = len(assignment)
    with ThreadPoolExecutor(max_workers=total_streams) as pool:
        futures = [pool.submit(stream_worker, stream) for stream in range(total_streams)]
        for future in futures:
            future.result()
    records.sort(key=lambda record: record.request_index)
    return records


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one llama-server SSE line; returns None for non-data/DONE lines."""
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def send_streaming_completion(
    *,
    cell: Cell,
    instance: Instance,
    prompt: PromptSpec,
    stream_id: int,
    request_index: int,
    n_predict: int,
    timeout_s: float,
) -> StreamRequestRecord:
    """POST /completion with stream=true; TTFT = first content chunk timestamp."""
    if httpx is None:
        raise RuntimeError("httpx is required for --execute (pip install httpx)")
    sampling = cell.manifest.get("sampling")
    sampling = sampling if isinstance(sampling, dict) else {}
    payload: dict[str, Any] = {
        "prompt": prompt.prompt,
        "n_predict": n_predict,
        "temperature": float(sampling.get("temperature", 0.0)),
        "cache_prompt": False,
        "stream": True,
    }
    if sampling.get("seed") is not None:
        payload["seed"] = int(sampling["seed"])
    url = f"http://127.0.0.1:{instance.port}/completion"
    start = time.perf_counter()
    first_token: float | None = None
    text_parts: list[str] = []
    timings: dict[str, Any] = {}
    status: int | None = None
    error = ""
    try:
        with httpx.Client(timeout=timeout_s) as client:
            with client.stream("POST", url, json=payload) as response:
                status = response.status_code
                for line in response.iter_lines():
                    chunk = parse_sse_line(line)
                    if chunk is None:
                        continue
                    content = chunk.get("content")
                    if content:
                        if first_token is None:
                            first_token = time.perf_counter()
                        text_parts.append(str(content))
                    if isinstance(chunk.get("timings"), dict):
                        timings = chunk["timings"]
        success = status == 200
        if not success:
            error = f"HTTP {status}"
    except Exception as exc:  # noqa: BLE001 — recorded per-request, run continues
        success = False
        error = str(exc)
    end = time.perf_counter()
    return StreamRequestRecord(
        cell_id=cell.cell_id,
        qid=prompt.qid,
        suite=prompt.suite,
        request_index=request_index,
        stream_id=stream_id,
        instance_port=instance.port,
        success=success,
        start_s=start,
        first_token_s=first_token,
        end_s=end,
        ttft_ms=((first_token - start) * 1000.0) if first_token is not None else None,
        latency_ms=(end - start) * 1000.0,
        predicted_tokens=int(timings.get("predicted_n") or 0),
        prompt_tokens=int(timings.get("prompt_n") or 0),
        predicted_tps=float(timings.get("predicted_per_second") or 0.0),
        draft_n=(int(timings["draft_n"]) if timings.get("draft_n") is not None else None),
        draft_n_accepted=(
            int(timings["draft_n_accepted"])
            if timings.get("draft_n_accepted") is not None
            else None
        ),
        http_status=status,
        error=error,
        response_text="".join(text_parts),
        timings=dict(timings),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def trimmed_aggregate(records: list[StreamRequestRecord]) -> dict[str, Any]:
    """Ramp-trimmed tasks/hour (see TRIM_DEFINITION; stored with the number)."""
    successes = [record for record in records if record.success]
    base: dict[str, Any] = {
        "trim_definition": TRIM_DEFINITION,
        "steady_count": 0,
        "window_seconds": 0.0,
        "tasks_per_hour_trimmed": 0.0,
    }
    if not successes:
        return base
    ramp_end = min(record.end_s for record in successes)
    drain_start = max(record.start_s for record in successes)
    window_s = drain_start - ramp_end
    steady = [
        record
        for record in successes
        if record.start_s >= ramp_end and record.end_s <= drain_start
    ]
    base["steady_count"] = len(steady)
    base["window_seconds"] = max(window_s, 0.0)
    if steady and window_s > 0:
        base["tasks_per_hour_trimmed"] = len(steady) / window_s * 3600.0
    return base


def resolve_sampling(cell: Cell) -> dict[str, Any]:
    """Effective sampling regime for the cell (recorded in row + run manifest).

    A manifest "sampling" block wins; the default is temperature-0 E1 parity.
    Review F4 (spec-completeness gap, operator call pending): accept rates
    measured under greedy decoding may misrepresent production spec-dec
    (feedback_production_sampling_seed_not_temp0 argues production temp +
    seed 42) — the effective regime is therefore always recorded so the
    choice is auditable, and dry-run flags decision-grade cells that carry
    no explicit block.
    """
    sampling = cell.manifest.get("sampling")
    if isinstance(sampling, dict):
        return {
            "temperature": float(sampling.get("temperature", 0.0)),
            "seed": sampling.get("seed"),
            "source": "manifest",
        }
    return {"temperature": 0.0, "seed": None, "source": "default_e1_parity"}


def summarize_cell(
    *,
    cell: Cell,
    records: list[StreamRequestRecord],
    wall_s: float,
    env: dict[str, str],
    instance_pids: dict[int, int],
    affinity: dict[str, Any],
    run_overrides_active: bool,
    host_warnings: list[str],
    cell_host_warnings: list[str] | None = None,
    throttle_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successes = [record for record in records if record.success]
    latencies = [record.latency_ms for record in successes]
    ttfts = [record.ttft_ms for record in successes if record.ttft_ms is not None]
    total_predicted = sum(record.predicted_tokens for record in successes)
    total_prompt = sum(record.prompt_tokens for record in successes)
    per_tps = [record.predicted_tps for record in successes if record.predicted_tps > 0]
    draft_totals = [record.draft_n for record in successes if record.draft_n is not None]
    accepted_totals = [
        record.draft_n_accepted
        for record in successes
        if record.draft_n_accepted is not None
    ]
    accept_rate: float | None = None
    if draft_totals and sum(draft_totals) > 0:
        accept_rate = sum(accepted_totals) / sum(draft_totals) if accepted_totals else 0.0
    per_stream: dict[str, dict[str, Any]] = {}
    for record in successes:
        bucket = per_stream.setdefault(
            str(record.stream_id), {"count": 0, "latencies": []}
        )
        bucket["count"] += 1
        bucket["latencies"].append(record.latency_ms)
    per_stream_summary = {
        stream: {
            "count": bucket["count"],
            "p50_latency_ms": percentile(bucket["latencies"], 0.50),
            "p95_latency_ms": percentile(bucket["latencies"], 0.95),
        }
        for stream, bucket in sorted(per_stream.items(), key=lambda kv: int(kv[0]))
    }
    total = len(records)
    success_count = len(successes)
    # Per-cell preconditions are hard gates too (protocol decision 5, review
    # F2/F7): a mid-run host-health flip or an under-load throttle warning
    # demotes THIS cell, not just the run.
    throttle_warnings = list((throttle_check or {}).get("warnings") or [])
    gate_warnings = list(host_warnings) + list(cell_host_warnings or []) + throttle_warnings
    hard_gates_passed = bool(affinity.get("live_affinity_verified")) and not gate_warnings
    decision_grade = (
        cell.decision_grade_intent and hard_gates_passed and not run_overrides_active
    )
    return {
        "timestamp": utc_now(),
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "cell_id": cell.cell_id,
        "window": cell.manifest.get("window"),
        "model_key": cell.model_key,
        "model_path": cell.model_path,
        "quant": cell.manifest.get("quant"),
        "architecture": cell.manifest.get("architecture"),
        "config_id": cell.config_id,
        "np": cell.np,
        "n_instances": len(cell.instances),
        "total_streams": cell.total_streams,
        "instances": [asdict(inst) for inst in cell.instances],
        "instance_pids": instance_pids,
        "ctx": cell.manifest.get("ctx"),
        "per_stream_ctx": cell.manifest.get("per_stream_ctx"),
        "ubatch_size": cell.manifest.get("ubatch_size"),
        "kv": cell.kv,
        "kv_unified": bool(cell.kv.get("kv_unified")),
        "spec_dec": cell.spec_dec,
        "draft_accept_rate": accept_rate,
        "draft_n_total": sum(draft_totals) if draft_totals else None,
        "draft_n_accepted_total": sum(accepted_totals) if accepted_totals else None,
        "ggml_iqk": env.get("GGML_IQK"),
        "total_count": total,
        "success_count": success_count,
        "error_rate": ((total - success_count) / total) if total else 1.0,
        "wall_seconds": wall_s,
        "tasks_per_hour_raw": (success_count / wall_s * 3600.0) if wall_s > 0 else 0.0,
        **trimmed_aggregate(records),
        "aggregate_predicted_tps": (total_predicted / wall_s) if wall_s > 0 else 0.0,
        "predicted_tokens_total": total_predicted,
        "prompt_tokens_total": total_prompt,
        "per_request_tps_mean": statistics.mean(per_tps) if per_tps else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "per_stream": per_stream_summary,
        "affinity_artifact": affinity.get("artifact_path"),
        "live_affinity_verified": affinity.get("live_affinity_verified"),
        "sampling": resolve_sampling(cell),
        "stage_b_families": cell.manifest.get("stage_b_families"),
        "host_health_warnings_at_cell": list(cell_host_warnings or []),
        "throttle_check": throttle_check,
        "decision_grade_intent": cell.decision_grade_intent,
        "decision_grade": decision_grade,
        "cell_error": None,
    }


CSV_FIELDS = [
    "timestamp",
    "protocol_id",
    "cell_id",
    "model_key",
    "quant",
    "config_id",
    "np",
    "n_instances",
    "total_streams",
    "success_count",
    "total_count",
    "error_rate",
    "wall_seconds",
    "tasks_per_hour_raw",
    "tasks_per_hour_trimmed",
    "aggregate_predicted_tps",
    "p50_latency_ms",
    "p95_latency_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "draft_accept_rate",
    "kv_unified",
    "ggml_iqk",
    "live_affinity_verified",
    "decision_grade",
    "cell_error",
]


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({fieldname: row.get(fieldname) for fieldname in CSV_FIELDS})


def failed_cell_row(cell: Cell, error: str, affinity: dict[str, Any] | None = None) -> dict[str, Any]:
    affinity = affinity or {}
    return {
        "timestamp": utc_now(),
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "cell_id": cell.cell_id,
        "window": cell.manifest.get("window"),
        "model_key": cell.model_key,
        "model_path": cell.model_path,
        "quant": cell.manifest.get("quant"),
        "architecture": cell.manifest.get("architecture"),
        "config_id": cell.config_id,
        "np": cell.np,
        "n_instances": len(cell.instances),
        "total_streams": cell.total_streams,
        "kv_unified": bool(cell.kv.get("kv_unified")),
        "success_count": 0,
        "total_count": 0,
        "error_rate": 1.0,
        "tasks_per_hour_raw": 0.0,
        "tasks_per_hour_trimmed": 0.0,
        "affinity_artifact": affinity.get("artifact_path"),
        "live_affinity_verified": affinity.get("live_affinity_verified"),
        "decision_grade_intent": cell.decision_grade_intent,
        "decision_grade": False,
        "cell_error": error,
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_cell_execute(
    *,
    cell: Cell,
    prompts: list[PromptSpec],
    args: argparse.Namespace,
    output_dir: Path,
    env: dict[str, str],
    host_warnings: list[str],
) -> dict[str, Any]:
    """Launch, gate, warm, drive, and tear down one cell. Returns the cell row."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[Instance, subprocess.Popen[str]]] = []
    affinity_info: dict[str, Any] = {}
    overrides_active = bool(args.allow_host_health_warning or args.skip_clean_check)
    try:
        # 1) Per-cell env drift tripwire + manifest expectation.
        assert_canonical_env(env)
        env_errors = check_env_expectation(env, cell)
        if env_errors:
            raise RuntimeError("; ".join(env_errors))

        # 1b) Per-cell host preconditions (protocol decision 5: throttle /
        # numa_balancing / clean-runtime are PER-CELL gates — a 4-6h Stage-B
        # window can flip mid-run; review F2/F7). Only idle-valid static
        # indicators gate here; the under-load frequency check runs during
        # the driver (FreqSampler).
        cell_host_warnings = (
            host_health_warnings(collect_attestation()) + cpu_freq_static_warnings()
        )
        if cell_host_warnings:
            write_jsonl(
                output_dir / "events.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "cell_precondition_warning",
                    "cell_id": cell.cell_id,
                    "warnings": cell_host_warnings,
                },
            )
            if not args.allow_host_health_warning:
                raise RuntimeError(
                    f"{cell.cell_id}: per-cell host-health preconditions failed "
                    "(mid-run flip): " + "; ".join(cell_host_warnings)
                )

        # 2) Sequential launch + health per instance.
        for inst in cell.instances:
            cmd = build_instance_command(
                binary=args.llama_server, cell=cell, inst=inst
            )
            log_path = log_dir / f"{cell.cell_id}-{inst.port}.log"
            proc = start_server(cmd, env, log_path)
            procs.append((inst, proc))
            wait_for_health(inst.port, args.startup_timeout, proc)

        # 3) Affinity HARD gate (no warn-and-continue path).
        pid_map = {str(inst.port): proc.pid for inst, proc in procs}
        artifact_path = output_dir / "affinity" / f"{cell.cell_id}.json"
        preflight_rc, artifact, preflight_output = run_affinity_preflight(
            preflight_script=args.affinity_preflight,
            cell=cell,
            pid_map=pid_map,
            artifact_path=artifact_path,
        )
        affinity_info = {
            "artifact_path": str(artifact_path),
            "live_affinity_verified": bool(
                (artifact or {}).get("live_affinity_verified")
            ),
        }
        write_jsonl(
            output_dir / "events.jsonl",
            {
                "timestamp": utc_now(),
                "event": "affinity_preflight",
                "cell_id": cell.cell_id,
                "returncode": preflight_rc,
                "artifact": str(artifact_path),
                "output_tail": preflight_output[-2000:] if preflight_output else "",
            },
        )
        if preflight_rc != 0:
            raise AffinityPreflightFailure(
                f"affinity preflight exited {preflight_rc} for {cell.cell_id}"
            )

        # 4) Per-instance pinned warm-up through EACH instance's own port.
        warmup = cell.manifest.get("warmup")
        warmup = warmup if isinstance(warmup, dict) else {}
        warmup_prompts = int(warmup.get("prompts") or 0)
        warmup_n_predict = int(warmup.get("n_predict") or 32)
        warmup_records: list[StreamRequestRecord] = []
        for inst_index, (inst, _proc) in enumerate(procs):
            for widx in range(warmup_prompts):
                prompt = prompts[widx % len(prompts)]
                warmup_records.append(
                    send_streaming_completion(
                        cell=cell,
                        instance=inst,
                        prompt=prompt,
                        stream_id=-1 - inst_index,
                        request_index=-1 - widx,
                        n_predict=warmup_n_predict,
                        timeout_s=args.request_timeout,
                    )
                )
        with (output_dir / "warmup_requests.jsonl").open("a", encoding="utf-8") as fh:
            for record in warmup_records:
                row = asdict(record)
                row.pop("response_text", None)
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        # 5) Closed-loop N x K driver over the pinned batch.
        n_predict = int(cell.prompt_caps.get("n_predict") or 256)
        requests_fh = (output_dir / "requests.jsonl").open("a", encoding="utf-8")
        responses_fh = (output_dir / "responses.jsonl").open("a", encoding="utf-8")
        write_lock = threading.Lock()

        def persist(record: StreamRequestRecord) -> None:
            # Incremental per-request persistence (feedback_incremental_persistence).
            with write_lock:
                meta = asdict(record)
                response_text = meta.pop("response_text", "")
                requests_fh.write(json.dumps(meta, sort_keys=True) + "\n")
                requests_fh.flush()
                responses_fh.write(
                    json.dumps(
                        {
                            "cell_id": record.cell_id,
                            "qid": record.qid,
                            "instance_port": record.instance_port,
                            "stream": record.stream_id,
                            "response_text": response_text,
                            "timings": record.timings,
                            "http_status": record.http_status,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                responses_fh.flush()

        def send(**kwargs: Any) -> StreamRequestRecord:
            return send_streaming_completion(
                cell=cell,
                n_predict=n_predict,
                timeout_s=args.request_timeout,
                **kwargs,
            )

        # Under-load throttle sampling runs only while the driver holds load
        # (review F1: the FREQ_BOOST gate is an under-load expectation).
        freq_sampler = FreqSampler()
        start = time.perf_counter()
        freq_sampler.start()
        try:
            records = run_cell_driver(
                cell=cell, prompts=prompts, send_fn=send, on_record=persist
            )
        finally:
            freq_sampler.stop()
            requests_fh.close()
            responses_fh.close()
        wall_s = time.perf_counter() - start
        throttle_check = freq_sampler.result()
        if throttle_check.get("warnings"):
            write_jsonl(
                output_dir / "events.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "cell_throttle_warning",
                    "cell_id": cell.cell_id,
                    "throttle_check": throttle_check,
                },
            )

        return summarize_cell(
            cell=cell,
            records=records,
            wall_s=wall_s,
            env=env,
            instance_pids={inst.port: proc.pid for inst, proc in procs},
            affinity=affinity_info,
            run_overrides_active=overrides_active,
            host_warnings=host_warnings,
            cell_host_warnings=cell_host_warnings,
            throttle_check=throttle_check,
        )
    finally:
        if procs:
            try:
                stop_results = teardown_cell(procs)
                write_jsonl(
                    output_dir / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "event": "cell_teardown",
                        "cell_id": cell.cell_id,
                        "results": stop_results,
                    },
                )
                ensure_clean_runtime()
            except Exception as exc:
                # Surviving llama processes: the run MUST stop (never launch the
                # next cell over survivors).
                raise TeardownFailure(str(exc)) from exc


class AffinityPreflightFailure(RuntimeError):
    """Raised when the affinity preflight HARD gate fails for a cell."""


class TeardownFailure(RuntimeError):
    """Raised when a cell teardown leaves surviving processes; aborts the run."""


def build_run_manifest(
    *,
    args: argparse.Namespace,
    cells: list[Cell],
    prompts: list[PromptSpec],
    output_dir: Path,
    dry_run: bool,
    attestation: dict[str, Any] | None,
    warnings: list[str],
    decision_grade: bool,
) -> dict[str, Any]:
    first = cells[0]
    return {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "era": first.manifest.get("era"),
        "dry_run": dry_run,
        "output_dir": str(output_dir),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "cell_manifest"
        }
        | {"cell_manifest": [str(path) for path in (args.cell_manifest or [])]},
        "model": {
            "model_key": first.model_key,
            "model_path": first.model_path,
            "quant": first.manifest.get("quant"),
            "architecture": first.manifest.get("architecture"),
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "manifest_path": str(cell.path),
                "config_id": cell.config_id,
                "np": cell.np,
                "window": cell.manifest.get("window"),
                "decision_grade_intent": cell.decision_grade_intent,
                "kv_unified": bool(cell.kv.get("kv_unified")),
                "sampling": resolve_sampling(cell),
            }
            for cell in cells
        ],
        "prompt_batch": {
            "source": first.prompt_batch.get("source"),
            "selection": first.prompt_batch.get("selection"),
            "pinned_from": first.prompt_batch.get("pinned_from"),
            "qids": [prompt.qid for prompt in prompts],
            "count": len(prompts),
        },
        "attestation": attestation,
        "host_health_warnings": warnings,
        "overrides": {
            "allow_host_health_warning": bool(args.allow_host_health_warning),
            "skip_clean_check": bool(args.skip_clean_check),
        },
        "decision_grade": decision_grade,
    }


def write_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def cmd_sweep(args: argparse.Namespace) -> int:
    cells = collect_cells(args)
    errors = harness_refusals(cells) + revalidate_cells(cells)
    if errors:
        raise RuntimeError(
            "cell manifests refused:\n" + "\n".join(f"- {error}" for error in errors)
        )
    pool_path = (
        args.question_pool
        if args.question_pool is not None
        else Path(str(cells[0].prompt_batch.get("source") or ""))
    )
    if not str(pool_path):
        raise RuntimeError("no question pool: manifest prompt_batch.source is empty")
    prompts = load_pinned_prompts(pool_path, cells[0])

    output_dir = args.output_root / args.run_id

    if not args.execute:
        # DRY-RUN (default): validate + print the resolved plan; never spawn.
        env = build_cell_env(args.llama_server)
        assert_canonical_env(env)
        print(f"[dry-run] run_id={args.run_id} model_key={cells[0].model_key}")
        print(
            "[dry-run] env delta: "
            + " ".join(
                f"{key}={env[key]}"
                for key in sorted(list(CANONICAL_OMP_ENV) + ["KMP_BLOCKTIME"])
            )
        )
        for cell in cells:
            env_errors = check_env_expectation(env, cell)
            if env_errors:
                raise RuntimeError("; ".join(env_errors))
            print(f"[dry-run] cell {cell.cell_id} (config {cell.config_id} np={cell.np}):")
            for inst in cell.instances:
                cmd = build_instance_command(
                    binary=args.llama_server, cell=cell, inst=inst
                )
                print("  " + " ".join(cmd))
            preflight_cmd = [
                sys.executable,
                str(args.affinity_preflight),
                "--cell-manifest",
                str(cell.path),
                "--pid-map",
                "{<port>: <pid>, ...}",
                "--output",
                str(output_dir / "affinity" / f"{cell.cell_id}.json"),
            ]
            print("  preflight: " + " ".join(preflight_cmd))
            if cell.decision_grade_intent and not isinstance(
                cell.manifest.get("sampling"), dict
            ):
                print(
                    f"  [note] {cell.cell_id}: no manifest 'sampling' block — the "
                    "driver defaults to temperature 0.0 (E1 parity). Accept rates "
                    "under greedy decoding may misrepresent production spec-dec "
                    "(feedback_production_sampling_seed_not_temp0: production temp "
                    "+ seed 42); the operator must bless one regime before "
                    "decision-grade windows (review F4)."
                )
        manifest = build_run_manifest(
            args=args,
            cells=cells,
            prompts=prompts,
            output_dir=output_dir,
            dry_run=True,
            attestation=None,
            warnings=[],
            decision_grade=False,
        )
        write_run_manifest(output_dir, manifest)
        print(f"[dry-run] wrote {output_dir / 'manifest.json'}; no process spawned")
        return 0

    # ---- EXECUTE path (operator-granted quiet window only) ----
    if not args.llama_server.exists():
        raise FileNotFoundError(f"llama-server binary not found: {args.llama_server}")
    for cell in cells:
        if not Path(cell.model_path).exists():
            raise FileNotFoundError(
                f"{cell.cell_id}: model file not found: {cell.model_path}"
            )
    if not args.skip_clean_check:
        ensure_clean_runtime()

    env = build_cell_env(args.llama_server)
    assert_canonical_env(env)

    attestation = collect_attestation()
    # Run-start gating uses IDLE-VALID static throttle indicators only: the
    # canonical FREQ_BOOST gate is an under-load expectation and false-fails
    # on the required-quiet bench host (review F1). The under-load check runs
    # per cell during the driver window (FreqSampler in run_cell_execute).
    warnings = host_health_warnings(attestation) + cpu_freq_static_warnings()
    if warnings and not args.allow_host_health_warning:
        formatted = "\n".join(f"- {warning}" for warning in warnings)
        raise RuntimeError(
            "host-health preconditions failed; refusing decision-grade P-BENCH-3 run.\n"
            f"{formatted}\n"
            "Use --allow-host-health-warning only for scout/non-gating cells."
        )
    overrides_active = bool(args.allow_host_health_warning or args.skip_clean_check)
    run_decision_grade = not warnings and not overrides_active

    attestation = attestation | {
        "ggml_iqk": env.get("GGML_IQK"),
        "kv_unified_per_cell": {
            cell.cell_id: bool(cell.kv.get("kv_unified")) for cell in cells
        },
        "omp_env": {key: env.get(key) for key in sorted(CANONICAL_OMP_ENV)}
        | {"KMP_BLOCKTIME": env.get("KMP_BLOCKTIME")},
        "ld_library_path": env.get("LD_LIBRARY_PATH"),
        "binary": str(args.llama_server),
        "binary_version": run_capture([str(args.llama_server), "--version"]),
        "api": "n/a (direct /completion; no orchestrator API in the loop)",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selected_prompts.jsonl").open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            fh.write(json.dumps(asdict(prompt), sort_keys=True) + "\n")

    manifest = build_run_manifest(
        args=args,
        cells=cells,
        prompts=prompts,
        output_dir=output_dir,
        dry_run=False,
        attestation=attestation,
        warnings=warnings,
        decision_grade=run_decision_grade,
    )
    write_run_manifest(output_dir, manifest)

    for cell in cells:
        print(f"[{utc_now()}] cell={cell.cell_id} config={cell.config_id} np={cell.np}", flush=True)
        try:
            row = run_cell_execute(
                cell=cell,
                prompts=prompts,
                args=args,
                output_dir=output_dir,
                env=env,
                host_warnings=warnings,
            )
        except AffinityPreflightFailure as exc:
            row = failed_cell_row(
                cell,
                "affinity_preflight_failed",
                {
                    "artifact_path": str(output_dir / "affinity" / f"{cell.cell_id}.json"),
                    "live_affinity_verified": False,
                },
            )
            if cell.decision_grade_intent:
                run_decision_grade = False
            write_jsonl(
                output_dir / "events.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "cell_failed",
                    "cell_id": cell.cell_id,
                    "error": str(exc),
                },
            )
        except TeardownFailure as exc:
            row = failed_cell_row(cell, f"teardown_failed: {exc}")
            run_decision_grade = False
            write_jsonl(output_dir / "cells.jsonl", row)
            write_csv_row(output_dir / "summary.csv", row)
            manifest["decision_grade"] = run_decision_grade
            manifest["aborted"] = f"teardown_failed at {cell.cell_id}: {exc}"
            write_run_manifest(output_dir, manifest)
            raise
        except Exception as exc:  # noqa: BLE001 — record the cell, keep the run honest
            row = failed_cell_row(cell, str(exc))
            if cell.decision_grade_intent:
                run_decision_grade = False
            write_jsonl(
                output_dir / "events.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "cell_failed",
                    "cell_id": cell.cell_id,
                    "error": str(exc),
                },
            )
        if cell.decision_grade_intent and not row.get("decision_grade"):
            # Any per-cell hard-gate demotion (host-health flip, under-load
            # throttle warning, failed gate) propagates to the run manifest.
            run_decision_grade = False
        write_jsonl(output_dir / "cells.jsonl", row)
        write_csv_row(output_dir / "summary.csv", row)
        print(
            "  tasks/hour raw={:.2f} trimmed={:.2f} p95={:.0f}ms err={:.1%}".format(
                float(row.get("tasks_per_hour_raw") or 0.0),
                float(row.get("tasks_per_hour_trimmed") or 0.0),
                float(row.get("p95_latency_ms") or 0.0),
                float(row.get("error_rate") or 0.0),
            ),
            flush=True,
        )

    manifest["decision_grade"] = run_decision_grade
    manifest["completed_at"] = utc_now()
    write_run_manifest(output_dir, manifest)
    print(f"wrote {output_dir}", flush=True)
    return 0


def collect_cells(args: argparse.Namespace) -> list[Cell]:
    paths: list[Path] = list(args.cell_manifest or [])
    if args.manifest_dir is not None:
        paths.extend(sorted(args.manifest_dir.glob("*.json")))
    return [load_cell_manifest(path) for path in paths]


# ---------------------------------------------------------------------------
# Summarizer (--summarize-run; offline, no inference)
# ---------------------------------------------------------------------------


def load_cells_jsonl(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "cells.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no cells.jsonl in {run_dir}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cell_aggregate(row: dict[str, Any]) -> float:
    trimmed = float(row.get("tasks_per_hour_trimmed") or 0.0)
    return trimmed if trimmed > 0 else float(row.get("tasks_per_hour_raw") or 0.0)


def aggregate_basis(row: dict[str, Any]) -> str:
    """Which number cell_aggregate returned: 'trimmed' or the 'raw_fallback'.

    The raw fallback (empty steady-state window, structural at high K)
    includes ramp+drain and systematically understates steady throughput —
    every R1/R2/R4 comparison records the per-cell basis and flags
    mixed-basis pairs (review F3, TRIM_BASIS_CAVEAT).
    """
    if float(row.get("tasks_per_hour_trimmed") or 0.0) > 0:
        return "trimmed"
    if float(row.get("tasks_per_hour_raw") or 0.0) > 0:
        return "raw_fallback"
    return "none"


def _row_families(row: dict[str, Any]) -> list[str]:
    fams = row.get("stage_b_families")
    return [str(fam) for fam in fams] if isinstance(fams, list) else []


def _is_variant_row(row: dict[str, Any]) -> bool:
    """Deliberate paired-probe variants (unified-KV probe, dense full-shape
    scout pair) must not be conflated with the canonical (config_id, np) cell
    in R1/R2/R4 picks (review F3/F5); they are compared explicitly in
    evaluate_scout_probes instead."""
    if bool(row.get("kv_unified")):
        return True
    if any(fam in VARIANT_FAMILY_TAGS for fam in _row_families(row)):
        return True
    cell_id = str(row.get("cell_id") or "")
    return any(cell_id.endswith(suffix) for suffix in VARIANT_CELL_ID_SUFFIXES)


def apply_garbage_gate(
    rows: list[dict[str, Any]], scores_path: Path | None
) -> list[str]:
    """Mark degraded cells from an offline-score JSONL; returns degraded cell ids.

    Score rows are per-question: {"cell_id", "qid", "parse_ok": bool,
    "repetition_loop": bool}. Gate: parse-failures <= 2/43 and zero
    repetition-loop flags, else the cell's speed is demoted to observation.
    """
    if scores_path is None:
        return []
    parse_failures: dict[str, int] = {}
    repetition_flags: dict[str, int] = {}
    with scores_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            score = json.loads(line)
            cell_id = str(score.get("cell_id") or "")
            if not cell_id:
                continue
            if score.get("parse_ok") is False:
                parse_failures[cell_id] = parse_failures.get(cell_id, 0) + 1
            if score.get("repetition_loop"):
                repetition_flags[cell_id] = repetition_flags.get(cell_id, 0) + 1
    degraded: list[str] = []
    for row in rows:
        cell_id = str(row.get("cell_id") or "")
        failures = parse_failures.get(cell_id, 0)
        repetitions = repetition_flags.get(cell_id, 0)
        if failures > GARBAGE_GATE_MAX_PARSE_FAILURES or repetitions > 0:
            row["degraded"] = True
            row["degraded_reason"] = (
                f"parse_failures={failures} repetition_flags={repetitions}; "
                "speed demoted to observation"
            )
            degraded.append(cell_id)
    return degraded


def _usable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row.get("cell_error")
        and not row.get("degraded")
        and float(row.get("success_count") or 0) > 0
    ]


def _find(rows: list[dict[str, Any]], config_id: str, np_level: int) -> dict[str, Any] | None:
    """Canonical cell for (config_id, np): paired-probe variants excluded."""
    matches = [
        row
        for row in rows
        if row.get("config_id") == config_id
        and int(row.get("np") or 0) == np_level
        and not _is_variant_row(row)
    ]
    if not matches:
        return None
    return max(matches, key=cell_aggregate)


def _pair_verdict(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    label: str,
    prefer_on_tie: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"pair": label}
    if left is None or right is None:
        entry["status"] = "insufficient_data"
        return entry
    left_rate = cell_aggregate(left)
    right_rate = cell_aggregate(right)
    entry["cells"] = {
        left["cell_id"]: round(left_rate, 3),
        right["cell_id"]: round(right_rate, 3),
    }
    entry["metric_basis"] = {
        left["cell_id"]: aggregate_basis(left),
        right["cell_id"]: aggregate_basis(right),
    }
    entry["mixed_metric_basis"] = aggregate_basis(left) != aggregate_basis(right)
    if entry["mixed_metric_basis"]:
        entry["basis_note"] = (
            "mixed metric basis (trimmed vs raw_fallback): raw includes "
            "ramp+drain and understates steady-state — verdict caveated, "
            "not decision-grade (review F3)"
        )
    if left_rate <= 0 and right_rate <= 0:
        entry["status"] = "insufficient_data"
        return entry
    low, high = sorted((left_rate, right_rate))
    margin = (high - low) / high if high > 0 else 0.0
    entry["margin"] = round(margin, 4)
    if margin >= R1_MARGIN:
        winner = left if left_rate > right_rate else right
        entry["status"] = "winner"
        entry["winner"] = winner["cell_id"]
        entry["winner_config"] = winner["config_id"]
    else:
        entry["status"] = "tie"
        entry["preferred"] = prefer_on_tie
        entry["note"] = (
            f"margin < {R1_MARGIN:.0%}: tie — prefer status-quo quarters ({prefer_on_tie})"
        )
    return entry


def _c1_is_whole_machine(rows: list[dict[str, Any]]) -> bool:
    """Gemma-style family: C1 is the full-machine shape and there is no C1b.

    The pre-registered gemma whole-machine pairs are {C1full@T vs C3@T/4}
    (batched-decode-measurement.md:363 'Gemma runs {1xfull, 4xq} only'; review
    F1/F2). Detection: a C1 row tagged whole_machine_T* / shaped 0-95, and NO
    usable C1b rows (a family with C1b keeps the pre-registered C1b-vs-C3
    pairing; half-shaped C1@T vs C3@T/4 would not be iso-resource).
    """
    if any(row.get("config_id") == "C1b" for row in rows):
        return False
    for row in rows:
        if row.get("config_id") != "C1":
            continue
        if any(fam.startswith("whole_machine_T") for fam in _row_families(row)):
            return True
        instances = row.get("instances")
        if (
            isinstance(instances, list)
            and len(instances) == 1
            and isinstance(instances[0], dict)
            and str(instances[0].get("cpu_list")) == "0-95"
        ):
            return True
    return False


def evaluate_r1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = _usable(rows)
    pairs: list[dict[str, Any]] = []
    c1_whole_machine = _c1_is_whole_machine(usable)
    for total in ISO_T_WHOLE_MACHINE:
        if c1_whole_machine:
            # Gemma money comparison under the same pre-registered margin/tie
            # rule (>=10% wins; smaller = tie -> prefer status-quo quarters).
            pairs.append(
                _pair_verdict(
                    _find(usable, "C1", total),
                    _find(usable, "C3", total // 4),
                    label=f"whole-machine T={total}: C1@{total} vs C3@{total // 4}",
                    prefer_on_tie="C3",
                )
            )
        else:
            pairs.append(
                _pair_verdict(
                    _find(usable, "C1b", total // 2),
                    _find(usable, "C3", total // 4),
                    label=f"whole-machine T={total}: C1b@{total // 2} vs C3@{total // 4}",
                    prefer_on_tie="C3",
                )
            )
    for total in ISO_T_HALF_MACHINE:
        pairs.append(
            _pair_verdict(
                _find(usable, "C1", total),
                _find(usable, "C2", total // 2),
                label=f"half-machine T={total}: C1@{total} vs C2@{total // 2}",
                prefer_on_tie="C3",
            )
        )
    scaling: list[dict[str, Any]] = []
    for k in SCALING_K:
        c1 = _find(usable, "C1", k)
        c1b = _find(usable, "C1b", k)
        if c1 is None or c1b is None:
            scaling.append({"k": k, "status": "insufficient_data"})
            continue
        base = cell_aggregate(c1)
        scaling.append(
            {
                "k": k,
                "status": "ok",
                "c1_tasks_per_hour": round(base, 3),
                "c1b_tasks_per_hour": round(cell_aggregate(c1b), 3),
                "c1b_over_c1": round(cell_aggregate(c1b) / base, 3) if base > 0 else None,
                "metric_basis": {
                    c1["cell_id"]: aggregate_basis(c1),
                    c1b["cell_id"]: aggregate_basis(c1b),
                },
                "mixed_metric_basis": aggregate_basis(c1) != aggregate_basis(c1b),
            }
        )
    k_star: int | None = None
    for total in ISO_T_WHOLE_MACHINE:
        big = _find(usable, "C1", total)
        quarters = _find(usable, "C3", total // 4)
        if big is None or quarters is None:
            continue
        if cell_aggregate(big) > cell_aggregate(quarters):
            k_star = total
            break
    return {
        "rule": "R1 crossover: iso-T winner needs >= 10% aggregate tasks/hour margin; "
        "smaller = tie -> prefer status-quo quarters (C3)",
        "c1_whole_machine": c1_whole_machine,
        "iso_t_pairs": pairs,
        "scaling_pairs": scaling,
        "k_star_roofline_flip": k_star,
        "anchors": {
            "C1@1": (row := _find(usable, "C1", 1)) and row.get("cell_id"),
            "C3@1": (row := _find(usable, "C3", 1)) and row.get("cell_id"),
        },
    }


def _k1_baseline(
    model_rows: list[dict[str, Any]], config_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    """K=1 p95 baseline for a config, with a same-shape proxy fallback.

    The pre-registered Stage-B grid has no C1b@1 / C2@1 anchor (review F5):
    when the config's own K=1 cell is absent, the same-per-instance-shape
    K=1 cell substitutes (C1@1 for C1b — half instances; C3@1 for C2 —
    quarter instances) and the substitution is recorded in rules.json.
    """
    base = _find(model_rows, config_id, 1)
    if base is not None:
        return base, None
    proxy_config = R2_K1_PROXY.get(config_id)
    if proxy_config:
        proxy = _find(model_rows, proxy_config, 1)
        if proxy is not None:
            return proxy, (
                f"{proxy_config}@1 substituted as the K=1 p95 baseline for "
                f"{config_id} (no {config_id}@1 in the pre-registered Stage-B "
                f"grid; same per-instance shape)"
            )
    return None, None


def evaluate_r2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = _usable(rows)
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in usable:
        key = f"{row.get('model_key')}+{row.get('quant')}"
        by_model.setdefault(key, []).append(row)
    verdicts: dict[str, Any] = {}
    for model_label, model_rows in sorted(by_model.items()):
        pareto = [
            row
            for row in model_rows
            if not any(
                other is not row
                and cell_aggregate(other) >= cell_aggregate(row)
                and float(other.get("p95_latency_ms") or 0.0)
                <= float(row.get("p95_latency_ms") or 0.0)
                and (
                    cell_aggregate(other) > cell_aggregate(row)
                    or float(other.get("p95_latency_ms") or 0.0)
                    < float(row.get("p95_latency_ms") or 0.0)
                )
                for other in model_rows
            )
        ]
        peak = max(model_rows, key=cell_aggregate)
        peak_rate = cell_aggregate(peak)
        peak_p95 = float(peak.get("p95_latency_ms") or 0.0)
        base, base_substitution = _k1_baseline(model_rows, str(peak.get("config_id")))
        verdict: dict[str, Any] = {
            "pareto": [
                {
                    "cell_id": row.get("cell_id"),
                    "aggregate_tasks_per_hour": round(cell_aggregate(row), 3),
                    "aggregate_basis": aggregate_basis(row),
                    "p95_latency_ms": row.get("p95_latency_ms"),
                }
                for row in sorted(pareto, key=cell_aggregate, reverse=True)
            ],
            "peak_cell": peak.get("cell_id"),
        }
        if base is None:
            verdict["lanes_real"] = None
            verdict["note"] = (
                "no K=1 cell (or same-shape proxy) for the peak config; "
                "cannot evaluate SLA"
            )
            verdicts[model_label] = verdict
            continue
        verdict["k1_baseline"] = {
            "cell_id": base.get("cell_id"),
            "proxy": base_substitution is not None,
            "note": base_substitution,
        }
        base_p95 = float(base.get("p95_latency_ms") or 0.0)
        sla_violated = (
            peak_p95 > R2_P95_RATIO * base_p95 or peak_p95 > R2_P95_ABSOLUTE_MS
        )

        def within_sla(row: dict[str, Any]) -> bool:
            own_base, _substitution = _k1_baseline(model_rows, str(row.get("config_id")))
            if own_base is None:
                return False
            own_base_p95 = float(own_base.get("p95_latency_ms") or 0.0)
            p95 = float(row.get("p95_latency_ms") or 0.0)
            return p95 <= R2_P95_RATIO * own_base_p95 and p95 <= R2_P95_ABSOLUTE_MS

        holder = next(
            (
                row
                for row in sorted(model_rows, key=cell_aggregate, reverse=True)
                if int(row.get("np") or 0) < int(peak.get("np") or 0)
                and cell_aggregate(row) >= R2_HOLD_FRACTION * peak_rate
                and within_sla(row)
            ),
            None,
        )
        verdict["peak_p95_over_k1"] = round(peak_p95 / base_p95, 3) if base_p95 > 0 else None
        verdict["sla_violated_at_peak"] = sla_violated
        verdict["holder_cell"] = holder.get("cell_id") if holder else None
        verdict["lanes_real"] = bool(sla_violated and holder is not None)
        bases = {str(peak.get("cell_id")): aggregate_basis(peak)}
        bases[str(base.get("cell_id"))] = aggregate_basis(base)
        if holder is not None:
            bases[str(holder.get("cell_id"))] = aggregate_basis(holder)
        verdict["metric_basis"] = bases
        verdict["mixed_metric_basis"] = len(set(bases.values())) > 1
        verdicts[model_label] = verdict
    return {
        "rule": "R2 lanes: lanes are real iff the peak-aggregate cell's p95 exceeds "
        "3x that config's K=1 p95 (or 60s absolute) while some lower-K cell holds "
        ">= 70% of peak within SLA",
        "models": verdicts,
    }


def evaluate_r3(
    rows: list[dict[str, Any]], baseline_path: Path | None
) -> dict[str, Any]:
    refusal_base = {
        "rule": "R3 eval-lane pricing (audit C2): the pre-v7 E2 rows "
        "(2.258/10.970 wall-min/eval) are demoted-to-prior and never gate",
    }
    if baseline_path is None:
        return refusal_base | {
            "status": "refused",
            "reason": "no --current-arm-baseline supplied; refusing to price the "
            "eval lane until a FRESH current-arm baseline row (v7 + core_v2 + "
            "WP-12 fleet layer) is measured",
        }
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    wall_minutes = baseline.get("wall_minutes_per_eval")
    attestation = baseline.get("attestation")
    attestation = attestation if isinstance(attestation, dict) else {}
    if not isinstance(wall_minutes, (int, float)) or isinstance(wall_minutes, bool) or wall_minutes <= 0:
        return refusal_base | {
            "status": "refused",
            "reason": "baseline row lacks a positive wall_minutes_per_eval",
        }
    items_per_eval = baseline.get("items_per_eval")
    if (
        not isinstance(items_per_eval, (int, float))
        or isinstance(items_per_eval, bool)
        or items_per_eval <= 0
    ):
        # Unit normalization (review F4): the core_v2 eval unit is 50 items
        # while the E5 cell batch is 43 P-BENCH-3 prompts — comparing
        # wall-minutes per differing work quantum bakes in a ~16% skew, larger
        # than R1's own 10% margin. Both sides must be priced per ITEM.
        return refusal_base | {
            "status": "refused",
            "reason": "baseline row lacks a positive items_per_eval — refusing "
            "to compare wall-minutes across differing work quanta (core_v2 "
            "eval = 50 items vs E5 batch = 43 prompts); supply the baseline's "
            "item count so both sides normalize to wall-minutes per item",
        }
    if attestation.get("api_worker_count") is None:
        return refusal_base | {
            "status": "refused",
            "reason": "baseline attestation lacks api_worker_count (C10-F1: the "
            "current arm is 6-uvicorn x per-process Semaphore(1); the worker "
            "count must be attested)",
        }
    baseline_minutes_per_item = wall_minutes / items_per_eval
    pricing = []
    for row in _usable(rows):
        wall_s = float(row.get("wall_seconds") or 0.0)
        cell_items = int(row.get("success_count") or 0)
        if wall_s <= 0 or cell_items <= 0:
            continue
        cell_minutes = wall_s / 60.0
        cell_minutes_per_item = cell_minutes / cell_items
        pricing.append(
            {
                "cell_id": row.get("cell_id"),
                "wall_minutes_per_batch": round(cell_minutes, 3),
                "batch_items": cell_items,
                "cell_wall_minutes_per_item": round(cell_minutes_per_item, 4),
                "current_arm_wall_minutes_per_eval": wall_minutes,
                "current_arm_items_per_eval": items_per_eval,
                "current_arm_wall_minutes_per_item": round(baseline_minutes_per_item, 4),
                "speedup_vs_current_arm": round(
                    baseline_minutes_per_item / cell_minutes_per_item, 3
                ),
            }
        )
    return refusal_base | {
        "status": "priced",
        "unit": "wall-minutes per item (both sides normalized; review F4)",
        "baseline": {
            "path": str(baseline_path),
            "wall_minutes_per_eval": wall_minutes,
            "items_per_eval": items_per_eval,
            "wall_minutes_per_item": round(baseline_minutes_per_item, 4),
            "api_worker_count": attestation.get("api_worker_count"),
        },
        "cells": pricing,
    }


def evaluate_r4(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MODEL-KEYED capability rows (never role-keyed)."""
    usable = _usable(rows)
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in usable:
        key = (str(row.get("model_key")), str(row.get("quant")))
        by_model.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (model_key, quant), model_rows in sorted(by_model.items()):
        # Capability picks come from canonical cells only; paired-probe
        # variants (kvu / dense-full scout pairs) are compared in
        # evaluate_scout_probes, never silently max'd in (review F3/F5).
        canonical_rows = [row for row in model_rows if not _is_variant_row(row)] or model_rows
        peak_rate = max(cell_aggregate(row) for row in canonical_rows)
        candidates = [
            row
            for row in canonical_rows
            if cell_aggregate(row) >= R4_PEAK_FRACTION * peak_rate
        ]
        recommended = min(
            candidates, key=lambda row: float(row.get("p95_latency_ms") or 0.0)
        )
        per_shape: dict[str, dict[str, Any]] = {}
        for row in model_rows:
            if _is_variant_row(row):
                continue  # paired-probe variants never seed capability rows
            config = str(row.get("config_id"))
            best = per_shape.get(config)
            if best is None or cell_aggregate(row) > best["aggregate_tasks_per_hour"]:
                per_shape[config] = {
                    "np": row.get("np"),
                    "aggregate_tasks_per_hour": round(cell_aggregate(row), 3),
                    "aggregate_basis": aggregate_basis(row),
                    "p95_latency_ms": row.get("p95_latency_ms"),
                }
        solo = _find(model_rows, "C1", 1) or _find(model_rows, "C3", 1)
        splitting: dict[str, float] = {}
        for k in SCALING_K:
            c1 = _find(model_rows, "C1", k)
            c1b = _find(model_rows, "C1b", k)
            if c1 is not None and c1b is not None and cell_aggregate(c1) > 0:
                splitting[f"C1b_over_C1_at_np{k}"] = round(
                    cell_aggregate(c1b) / cell_aggregate(c1), 3
                )
        output.append(
            {
                "model_key": model_key,
                "quant": quant,
                "recommended": {
                    "config_id": recommended.get("config_id"),
                    "np": recommended.get("np"),
                    "aggregate_tasks_per_hour": round(cell_aggregate(recommended), 3),
                    "aggregate_basis": aggregate_basis(recommended),
                    "p95_latency_ms": recommended.get("p95_latency_ms"),
                    "rule": "smallest-latency cell achieving >= 90% of peak aggregate",
                },
                "mixed_metric_basis": len(
                    {aggregate_basis(row) for row in canonical_rows}
                ) > 1,
                "solo_shape": (
                    {
                        "config_id": solo.get("config_id"),
                        "cell_id": solo.get("cell_id"),
                        "aggregate_tasks_per_hour": round(cell_aggregate(solo), 3),
                    }
                    if solo
                    else None
                ),
                "numa_splitting_potential": splitting or None,
                "per_shape_np_optimum": per_shape,
                "ctx_kv_config": {
                    "per_stream_ctx": recommended.get("per_stream_ctx"),
                    "ctx": recommended.get("ctx"),
                    "kv": recommended.get("kv"),
                },
                "spec_recipe": {
                    "spec_dec": recommended.get("spec_dec"),
                    "draft_accept_rate": recommended.get("draft_accept_rate"),
                    "accept_rates_per_cell": {
                        row["cell_id"]: row.get("draft_accept_rate") for row in model_rows
                    },
                },
                "kv_unified_attestation": {
                    row["cell_id"]: bool(row.get("kv_unified")) for row in model_rows
                },
            }
        )
    return output


def evaluate_scout_probes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicit paired-probe comparisons for the operator's Stage-B pruning.

    The W0 scout carries deliberate same-(config, np) variant pairs that
    _find deliberately excludes from R1/R2/R4 (review F3/F5): the qwen36
    C1@16 split-vs--kvu probe (>=5% delta escalates a Stage-B
    split-vs-unified arm to the operator, M06) and the dense-control C1
    half0-vs-full shape pair (Stage-B C1 adopts the winner).
    """
    usable = _usable(rows)
    kvu_pairs: list[dict[str, Any]] = []
    for kvu in (row for row in usable if bool(row.get("kv_unified"))):
        split = _find(usable, str(kvu.get("config_id")), int(kvu.get("np") or 0))
        entry: dict[str, Any] = {
            "kvu_cell": kvu.get("cell_id"),
            "split_cell": split.get("cell_id") if split else None,
        }
        split_rate = cell_aggregate(split) if split else 0.0
        kvu_rate = cell_aggregate(kvu)
        if split is None or split_rate <= 0:
            entry["status"] = "insufficient_data"
        else:
            delta = (kvu_rate - split_rate) / split_rate
            entry["status"] = "ok"
            entry["kvu_tasks_per_hour"] = round(kvu_rate, 3)
            entry["split_tasks_per_hour"] = round(split_rate, 3)
            entry["delta_fraction"] = round(delta, 4)
            entry["mixed_metric_basis"] = aggregate_basis(kvu) != aggregate_basis(split)
            entry["escalate_to_operator"] = abs(delta) >= KVU_PROBE_ESCALATION
            if entry["escalate_to_operator"]:
                entry["note"] = (
                    ">=5% split-vs-unified delta: escalate a Stage-B "
                    "split-vs-unified arm to the operator (protocol decision 1)"
                )
        kvu_pairs.append(entry)

    shape_pairs: list[dict[str, Any]] = []
    fulls = [
        row
        for row in usable
        if "scout_dense_c1_shape_pair" in _row_families(row)
        or str(row.get("cell_id") or "").endswith("-scout-full")
    ]
    for full in fulls:
        half = _find(usable, str(full.get("config_id")), int(full.get("np") or 0))
        entry = {
            "np": full.get("np"),
            "full_cell": full.get("cell_id"),
            "half_cell": half.get("cell_id") if half else None,
        }
        half_rate = cell_aggregate(half) if half else 0.0
        full_rate = cell_aggregate(full)
        if half is None or (half_rate <= 0 and full_rate <= 0):
            entry["status"] = "insufficient_data"
        else:
            entry["status"] = "ok"
            entry["full_tasks_per_hour"] = round(full_rate, 3)
            entry["half0_tasks_per_hour"] = round(half_rate, 3)
            entry["winner_shape"] = "full" if full_rate > half_rate else "half0"
            entry["mixed_metric_basis"] = aggregate_basis(full) != aggregate_basis(half)
            entry["note"] = (
                "Stage-B C1 adopts the winning dense-control shape (regenerate "
                "the W3 C1 cells on the full-machine shape if full wins)"
            )
        shape_pairs.append(entry)
    return {"kvu_split_pairs": kvu_pairs, "dense_c1_shape_pairs": shape_pairs}


def render_summary_md(rules: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# E5 NUMA x np sweep summary (P-BENCH-3)")
    lines.append("")
    lines.append(f"> {E1_COMPARABILITY_CAVEAT}")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append(
        "| cell | config | np | tasks/h raw | tasks/h trimmed | p50 ms | p95 ms "
        "| TTFT p50 ms | accept | kvu | grade | error |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        accept = row.get("draft_accept_rate")
        lines.append(
            "| {} | {} | {} | {:.2f} | {:.2f} | {:.0f} | {:.0f} | {:.0f} | {} | {} | {} | {} |".format(
                row.get("cell_id"),
                row.get("config_id"),
                row.get("np"),
                float(row.get("tasks_per_hour_raw") or 0.0),
                float(row.get("tasks_per_hour_trimmed") or 0.0),
                float(row.get("p50_latency_ms") or 0.0),
                float(row.get("p95_latency_ms") or 0.0),
                float(row.get("ttft_p50_ms") or 0.0),
                f"{accept:.3f}" if isinstance(accept, (int, float)) else "-",
                "on" if row.get("kv_unified") else "off",
                "DG" if row.get("decision_grade") else ("degraded" if row.get("degraded") else "obs"),
                row.get("cell_error") or "",
            )
        )
    lines.append("")
    lines.append("## R1 — iso-T crossover")
    lines.append("")
    for pair in rules["R1"]["iso_t_pairs"]:
        lines.append(f"- {pair.get('pair')}: {pair.get('status')}"
                     + (f" -> {pair.get('winner')}" if pair.get("winner") else "")
                     + (f" ({pair.get('note')})" if pair.get("note") else "")
                     + (" [MIXED METRIC BASIS — caveated]" if pair.get("mixed_metric_basis") else ""))
    lines.append(f"- K* roofline flip: {rules['R1']['k_star_roofline_flip']}")
    lines.append("")
    probes = rules.get("scout_probes") or {}
    if probes.get("kvu_split_pairs") or probes.get("dense_c1_shape_pairs"):
        lines.append("## Scout paired probes")
        lines.append("")
        for pair in probes.get("kvu_split_pairs", []):
            lines.append(
                f"- kvu probe {pair.get('kvu_cell')} vs {pair.get('split_cell')}: "
                f"{pair.get('status')} delta={pair.get('delta_fraction')} "
                f"escalate={pair.get('escalate_to_operator')}"
            )
        for pair in probes.get("dense_c1_shape_pairs", []):
            lines.append(
                f"- dense C1 shape {pair.get('full_cell')} vs {pair.get('half_cell')}: "
                f"{pair.get('status')} winner={pair.get('winner_shape')}"
            )
        lines.append("")
    lines.append("## R2 — lane reality")
    lines.append("")
    for model_label, verdict in rules["R2"]["models"].items():
        lines.append(f"- {model_label}: lanes_real={verdict.get('lanes_real')}")
    lines.append("")
    lines.append("## R3 — eval-lane pricing")
    lines.append("")
    lines.append(f"- status: {rules['R3'].get('status')}")
    if rules["R3"].get("reason"):
        lines.append(f"- reason: {rules['R3']['reason']}")
    lines.append("")
    lines.append("## R4 — model-keyed capability rows")
    lines.append("")
    for entry in rules["R4"]:
        recommended = entry["recommended"]
        lines.append(
            f"- {entry['model_key']}+{entry['quant']}: {recommended['config_id']}@np"
            f"{recommended['np']} ({recommended['aggregate_tasks_per_hour']} tasks/h)"
        )
    if rules.get("degraded_cells"):
        lines.append("")
        lines.append("## Degraded cells (garbage gate)")
        lines.append("")
        for cell_id in rules["degraded_cells"]:
            lines.append(f"- {cell_id}: speed demoted to observation")
    lines.append("")
    return "\n".join(lines)


def cmd_summarize(args: argparse.Namespace) -> int:
    run_dir = args.summarize_run
    rows = load_cells_jsonl(run_dir)
    degraded = apply_garbage_gate(rows, args.offline_scores)
    rules = {
        "created_at": utc_now(),
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "caveats": [E1_COMPARABILITY_CAVEAT, TRIM_BASIS_CAVEAT],
        "degraded_cells": degraded,
        "R1": evaluate_r1(rows),
        "R2": evaluate_r2(rows),
        "R3": evaluate_r3(rows, args.current_arm_baseline),
        "R4": evaluate_r4(rows),
        "scout_probes": evaluate_scout_probes(rows),
    }
    (run_dir / "rules.json").write_text(
        json.dumps(rules, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        render_summary_md(rules, rows), encoding="utf-8"
    )
    print(f"wrote {run_dir / 'rules.json'} and {run_dir / 'summary.md'}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-manifest",
        action="append",
        type=Path,
        default=[],
        help="e5-cell-manifest JSON; repeatable. All manifests in one "
        "invocation must share one model_key.",
    )
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually launch servers. DEFAULT IS DRY-RUN. Requires "
        "--i-have-operator-grant (operator-approved quiet window).",
    )
    parser.add_argument(
        "--i-have-operator-grant",
        action="store_true",
        help="explicit acknowledgment that the operator granted this "
        "inference window (no-inference-without-approval standing rule)",
    )
    parser.add_argument(
        "--run-id",
        default=f"e5-pbench3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument(
        "--affinity-preflight", type=Path, default=DEFAULT_AFFINITY_PREFLIGHT
    )
    parser.add_argument(
        "--question-pool",
        type=Path,
        default=None,
        help="override the manifest prompt_batch.source pool path",
    )
    parser.add_argument(
        "--allow-host-health-warning",
        action="store_true",
        help="scout-only; forces decision_grade=false in the run manifest",
    )
    parser.add_argument(
        "--skip-clean-check",
        action="store_true",
        help="scout-only escape hatch; forces decision_grade=false",
    )
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--summarize-run",
        type=Path,
        default=None,
        help="offline mode: read RUN_DIR/cells.jsonl and emit summary.md + rules.json",
    )
    parser.add_argument(
        "--current-arm-baseline",
        type=Path,
        default=None,
        help="fresh current-arm baseline row JSON for R3 (v7 + core_v2 + WP-12 "
        "fleet layer, attestation must record api_worker_count)",
    )
    parser.add_argument(
        "--offline-scores",
        type=Path,
        default=None,
        help="offline B7-score JSONL for the garbage gate (per-question rows)",
    )
    args = parser.parse_args(argv)
    if args.summarize_run is None:
        if not args.cell_manifest and args.manifest_dir is None:
            parser.error("supply --cell-manifest and/or --manifest-dir (or --summarize-run)")
        if args.execute and not args.i_have_operator_grant:
            parser.error(
                "--execute requires --i-have-operator-grant: benches run only in "
                "operator-approved quiet windows (feedback_no_concurrent_inference)"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize_run is not None:
        return cmd_summarize(args)
    return cmd_sweep(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
