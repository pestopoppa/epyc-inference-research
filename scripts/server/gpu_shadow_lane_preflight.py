#!/usr/bin/env python3
"""Guarded preflight probe for the GPU-resident shadow lane (plan-only default).

Pattern: scripts/benchmark/eval_batch_serving_probe.py (the E2 activation-probe
precedent) applied to the gpu_shadow_lane scaffold. When the operator runs it,
the probe verifies — WITHOUT touching any production process, launching any
server, or sending any inference:

  - VRAM budget on ROCm0 (rocm-smi meminfo, pinned to card 0) covers the
    planned tenant residency + KV estimate from
    orchestration/gpu_shadow_lane_np_ceiling.yaml
  - no foreign GPU compute PIDs (rocm-smi showpids); PIDs on the
    ``--expected-gpu-pid`` allowlist (default empty) are reported, not blocked
  - KFD state (/dev/kfd + /sys/class/kfd topology)
  - production HIP binary reports version 10107 (67a433bf4) — the v8 freeze
  - lane TCP port (18100) is free
  - tenant GGUF identity: streamed sha256 against the policy's pinned hash
    (~28 GiB file — expect a couple of minutes of read I/O; skippable via an
    explicit ``--skip-tenant-hash``, with the skip RECORDED in report and
    attestation)
  - live-affinity overlap taxonomy (P1-1/P1-2): overlap between the planned
    host cpuset (184-191, the GPU host-thread SMT rule) and every live
    llama-server's affinity is evaluated over PHYSICAL cores — cpu N and its
    SMT sibling N+96 fold together — and classified:
      * "unpinned"          — full-host masks (e.g. unpinned embedders inherit
                              0-191): informational only
      * "static-co-tenant"  — overlap EXPLAINED by a static NUMA_CONFIG cpuset
                              (Q1B quarters 72-95,168-191; architect_general
                              (the :8074 full instance again since the
                              2026-09-27 swap) / worker_general full 0-95 whose physical cores
                              88-95 are the lane's SMT siblings): WARNING —
                              these are the Step-4 contention-recert set,
                              not a launch conflict
      * "unexplained"       — a pinned mask overlapping the lane that matches
                              no static cpuset: BLOCKER (a healthy fleet never
                              produces this; --allow-smt-overlap does NOT
                              downgrade it)
  - the (np, per-slot context) plan is within the validated np_ceiling policy

Mutation scope: NONE in plan mode beyond writing the report directory.
``--apply`` adds exactly one artifact — ``preflight_attestation.json`` inside
the report directory, consumed by the activation choreography
(docs/gpu-shadow-lane.md) — and is refused while any blocker stands. The probe
never starts, stops, signals, or renices a process.

Exit codes: 0 = clean; 75 = blocked (matches the eval_batch probe convention);
argparse's 2 for usage errors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from scripts.server.gpu_shadow_lane import (
    LANE_BINARY,
    LANE_BINARY_COMMIT,
    LANE_BINARY_VERSION,
    LANE_DEVICE,
    LANE_HOST_CPUSET,
    LANE_PORT,
    MODE_MTP_OFF,
    VALID_MODES,
    NpCeilingPolicy,
    TENANT_CANDIDATE_ID,
    build_tenant_launch_plan,
    estimated_dynamic_gib,
    lane_enabled,
    load_np_ceiling_policy,
    np_ceiling,
)
from scripts.server.gpu_shadow_lane_lease import (
    SMT_SIBLING_OFFSET as _SMT_SIBLING_OFFSET,
    fold_cpus_to_physical,
)
from scripts.server.stack_numa import NUMA_CONFIG
from src.features import Features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "orchestration" / "reports"
KFD_DEV = Path("/dev/kfd")
KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ── Pure helpers (unit-tested; no subprocess, no filesystem) ─────────────────


def parse_cpu_list(spec: str) -> set[int]:
    """Parse a taskset-style cpu list ("72-95,168-191") into a set of ints."""
    cpus: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(chunk))
    return cpus


def cpuset_overlap(a: str, b: str) -> set[int]:
    return parse_cpu_list(a) & parse_cpu_list(b)


# P1-2: 192-CPU EPYC 9655 host; SMT sibling of cpu N is N+96 (and vice versa).
# Overlap math folds siblings together because two hyperthreads of one physical
# core contend for its execution resources — a process pinned to 88-95 and one
# pinned to 184-191 are PHYSICAL co-tenants even though their masks are disjoint.
# P2-1a UNIFIED (2026-07-28): the fold math now has ONE implementation,
# `gpu_shadow_lane_lease.fold_cpus_to_physical`, which this module re-expands to
# the sibling closure for REPORTING. The two representations differ only in
# which ids they return — the lease module answers in physical cores (88-95),
# the probe reports both hyperthreads (88-95 + 184-191) because an operator
# reading a blocker wants to see the mask that actually collided. The
# co-tenancy PREDICATE is identical, and now provably so: it is computed once.
HOST_CPU_COUNT = 192
SMT_SIBLING_OFFSET = _SMT_SIBLING_OFFSET


def smt_fold(cpus: set[int]) -> set[int]:
    """Union of a cpu set with its SMT siblings (physical-core closure).

    Delegates the fold to the canonical lease helper, then re-expands each
    physical core to both of its hyperthreads.
    """
    return {
        cpu
        for core in fold_cpus_to_physical(cpus)
        for cpu in (core, core + SMT_SIBLING_OFFSET)
    }


def folded_overlap(a: str, b: str) -> set[int]:
    """Cpuset intersection evaluated over the union of {cpus, SMT siblings}."""
    return smt_fold(parse_cpu_list(a)) & smt_fold(parse_cpu_list(b))


def static_smt_overlap_roles(host_cpuset: str = LANE_HOST_CPUSET) -> dict[str, list[int]]:
    """Production roles whose NUMA_CONFIG cpusets are physical co-tenants of the
    lane's host cpuset, SMT-sibling-folded (static topology fact — this is the
    Step-4 contention-recert set, not a live blocker).

    P1-2: folding makes the full-machine instances (0-95 — architect_general
    8074 [architect_critic until the 2026-09-27 ARCHITECT SWAP], frontdoor 8070,
    ingest_long_context 8085, worker_general 8072) visible
    as co-tenants of 184-191 — their physical cores 88-95 ARE the lane slice's
    SMT siblings — alongside the instances that name the lane's logical CPUs
    directly: the half-B instances (frontdoor 8180, worker_general 8182, ingest
    8285, cpuset 48-95,144-191) and the two GPU host-lane roles themselves
    (architect_critic 8083 [architect_general until the 2026-09-27 swap],
    worker_vision 8086).

    The parenthetical here used to read "frontdoor 8380, worker_general 8382,
    ingest 8485, vision_escalation 8087". Those ports are retired: the quarter
    fleet was replaced by 1 full + 2 halves on 2026-07-31 (8280/8380/8282/8382/
    8385/8485 freed, not to be revived) and :8087 went away when
    vision_escalation became an alias on worker_vision's :8086.
    """
    lane_folded = smt_fold(parse_cpu_list(host_cpuset))
    overlaps: dict[str, list[int]] = {}
    for role, cfg in NUMA_CONFIG.items():
        ports = []
        for cpu_list, port, _threads in cfg.get("instances", []):
            if smt_fold(parse_cpu_list(cpu_list)) & lane_folded:
                ports.append(port)
        if ports:
            overlaps[role] = ports
    return overlaps


def _static_instance_matches(proc_cpus: set[int]) -> list[str]:
    """Return ``role[port]`` labels of static NUMA_CONFIG instances explaining a
    process's affinity mask (exact match or subset of the instance cpuset)."""
    matches: list[str] = []
    for role, cfg in NUMA_CONFIG.items():
        for cpu_list, port, _threads in cfg.get("instances", []):
            instance_cpus = parse_cpu_list(cpu_list)
            if proc_cpus and proc_cpus <= instance_cpus:
                matches.append(f"{role}[{port}]")
    return matches


def classify_live_overlap(
    cpus_allowed_list: str | None,
    host_cpuset: str = LANE_HOST_CPUSET,
) -> tuple[str, list[int], list[str]]:
    """Classify a live process's affinity against the lane slice (P1-1 taxonomy).

    Returns ``(overlap_class, folded_overlap_cpus, static_matches)`` with
    overlap_class one of:
      - "none"             — no folded overlap with the lane slice
      - "unpinned"         — the mask covers every host CPU (scheduler-placed;
                             e.g. embedders inherit 0-191): informational
      - "static-co-tenant" — overlap explained by a static NUMA_CONFIG cpuset
                             (mask == or subset of an instance cpuset): warning,
                             Step-4 contention-recert set
      - "unexplained"      — pinned overlap matching no static cpuset: blocker
    """
    if not cpus_allowed_list:
        return "none", [], []
    proc_cpus = parse_cpu_list(cpus_allowed_list)
    overlap = sorted(smt_fold(proc_cpus) & smt_fold(parse_cpu_list(host_cpuset)))
    if not overlap:
        return "none", [], []
    if proc_cpus >= set(range(HOST_CPU_COUNT)):
        return "unpinned", overlap, []
    matches = _static_instance_matches(proc_cpus)
    if matches:
        return "static-co-tenant", overlap, matches
    return "unexplained", overlap, []


# P2-7: the lane serves on card 0 (ROCm0) ONLY. Parsing "whichever record has
# the fields" would silently attest the wrong device on a multi-GPU host, so
# meminfo is pinned to the card-0 record (rocm-smi JSON keys it "card0").
ROCM_CARD_KEY = "card0"


def parse_rocm_meminfo(payload: Any, card_key: str = ROCM_CARD_KEY) -> tuple[float, float] | None:
    """Extract (total_gib, used_gib) for CARD 0 from `rocm-smi --showmeminfo vram --json`."""
    if not isinstance(payload, dict):
        return None
    record = payload.get(card_key)
    if not isinstance(record, dict):
        return None
    total = used = None
    for key, value in record.items():
        key_l = str(key).lower()
        if "vram total memory" in key_l:
            total = value
        elif "used" in key_l and "vram" in key_l:
            used = value
    if total is None or used is None:
        return None
    try:
        return float(total) / float(1 << 30), float(used) / float(1 << 30)
    except (TypeError, ValueError):
        return None


_PID_KEY_RE = re.compile(r"pid\D*?(\d+)", re.IGNORECASE)


def parse_rocm_pids(payload: Any) -> list[int]:
    """Extract GPU compute PIDs from `rocm-smi --showpids --json` (tolerant).

    rocm-smi encodes the PID in the mapping KEY ("PID 12345": [...]); some
    variants use {"pid": N} records. Both are accepted.
    """
    pids: set[int] = set()

    def _walk(node: Any, key_hint: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                match = _PID_KEY_RE.search(key_text)
                if match:
                    pids.add(int(match.group(1)))
                _walk(value, key_text)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item, key_hint)
        else:
            if key_hint.lower() == "pid":
                try:
                    pid = int(str(node).strip())
                except (TypeError, ValueError):
                    return
                if pid > 0:
                    pids.add(pid)

    _walk(payload)
    return sorted(pid for pid in pids if pid > 0)


@dataclass
class LanePlan:
    tenant_id: str
    model_path: str
    model_vram_gib: float
    np_slots: int
    slot_context_tokens: int
    port: int
    host_cpuset: str
    dynamic_budget_gib: float
    np_ceiling: int | None
    estimated_dynamic_gib: float | None
    launch_argv: list[str] = field(default_factory=list)
    mode: str = MODE_MTP_OFF
    # P2-1: pinned tenant GGUF hash from the np_ceiling policy (None when the
    # policy has no pin — reported as a warning, never silently passed).
    expected_model_sha256: str | None = None


@dataclass
class PreflightFacts:
    flag_enabled: bool
    binary_exists: bool
    binary_version_output: str | None
    kfd_dev_present: bool
    kfd_topology_present: bool
    vram_total_gib: float | None
    vram_used_gib: float | None
    gpu_compute_pids: list[int]
    live_llama_processes: list[dict[str, Any]]  # {pid, cpus_allowed_list, overlap, overlap_class, static_matches}
    model_file_exists: bool
    static_smt_overlaps: dict[str, list[int]]
    errors: list[str] = field(default_factory=list)
    # P2-1: lane TCP port state (None = probe unavailable) + streamed tenant
    # GGUF sha256 (None when unreadable or skipped; skip is recorded).
    lane_port_in_use: bool | None = None
    model_sha256: str | None = None
    model_sha256_skipped: bool = False


def evaluate_preflight(
    facts: PreflightFacts,
    plan: LanePlan,
    *,
    require_enabled: bool = False,
    allow_existing_gpu_pids: bool = False,
    allow_smt_overlap: bool = False,
    expected_gpu_pids: frozenset[int] | set[int] = frozenset(),
) -> tuple[list[str], list[str], list[str]]:
    """Compute (blockers, warnings, infos). Pure — no I/O.

    P1-1 taxonomy: live-affinity overlaps are classified per process
    (``classify_live_overlap``); only UNEXPLAINED pinned overlaps block.
    ``allow_smt_overlap`` now affects ONLY the static-co-tenant class (it
    acknowledges the known co-tenants, demoting their warning to info) — it can
    no longer excuse an unexplained overlap. P2-8: ``expected_gpu_pids`` is the
    expected-tenant allowlist (default empty) for the foreign-PID check.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []

    if require_enabled and not facts.flag_enabled:
        blockers.append("gpu_shadow_lane feature flag is not enabled on this environment")
    elif not facts.flag_enabled:
        warnings.append("gpu_shadow_lane feature flag is off (expected pre-activation)")

    if not facts.binary_exists:
        blockers.append(f"production HIP binary missing: {LANE_BINARY}")
    else:
        version_output = facts.binary_version_output or ""
        if LANE_BINARY_VERSION not in version_output or LANE_BINARY_COMMIT not in version_output:
            blockers.append(
                "HIP binary version mismatch: expected "
                f"{LANE_BINARY_VERSION} ({LANE_BINARY_COMMIT}), got: "
                f"{version_output.strip()[:200] or '<no output>'}"
            )

    if not facts.kfd_dev_present or not facts.kfd_topology_present:
        blockers.append("KFD state not healthy (/dev/kfd or kfd topology missing)")

    if facts.vram_total_gib is None or facts.vram_used_gib is None:
        blockers.append("could not attest VRAM via rocm-smi meminfo")
    else:
        free_gib = facts.vram_total_gib - facts.vram_used_gib
        required = plan.estimated_dynamic_gib
        if required is None:
            warnings.append(
                "no KV arithmetic model for tenant; VRAM check limited to model residency"
            )
            required = 0.0
        # model weights + dynamic estimate must fit in currently-free VRAM
        if free_gib < plan.model_vram_gib + required:
            blockers.append(
                f"VRAM budget short: free {free_gib:.1f} GiB < model "
                f"{plan.model_vram_gib:.1f} GiB + dynamic estimate {required:.1f} GiB"
            )

    # P2-8: expected-tenant allowlist — allowlisted PIDs are reported (info),
    # everything else stays a blocker unless explicitly acknowledged.
    expected_present = sorted(set(facts.gpu_compute_pids) & set(expected_gpu_pids))
    foreign_pids = sorted(set(facts.gpu_compute_pids) - set(expected_gpu_pids))
    if expected_present:
        infos.append(f"expected GPU tenant PIDs present (allowlisted): {expected_present}")
    if foreign_pids:
        message = (
            f"foreign GPU compute PIDs present: {foreign_pids} "
            "(operator-owned processes are never killed; wait for them to vacate)"
        )
        if allow_existing_gpu_pids:
            warnings.append(message)
        else:
            blockers.append(message)

    # P1-1/P1-2: per-process overlap taxonomy over SMT-folded physical cores.
    # Classification is recomputed here from cpus_allowed_list (pure), so the
    # verdict never depends on what the gather layer pre-annotated.
    for proc in facts.live_llama_processes:
        overlap_class, overlap, matches = classify_live_overlap(
            proc.get("cpus_allowed_list"), plan.host_cpuset
        )
        if overlap_class == "none":
            continue
        detail = (
            f"pid {proc.get('pid')} (cpus {proc.get('cpus_allowed_list')}, "
            f"folded overlap {overlap})"
        )
        if overlap_class == "unpinned":
            infos.append(
                f"unpinned llama-server shares the lane slice via full-host mask: {detail} "
                "(scheduler-placed; not a launch conflict)"
            )
        elif overlap_class == "static-co-tenant":
            message = (
                f"static-co-tenant llama-server on the lane slice: {detail}, "
                f"explained by NUMA_CONFIG {matches} — Step-4 contention recert "
                "applies; not a launch conflict"
            )
            if allow_smt_overlap:
                infos.append(message + " [acknowledged via --allow-smt-overlap]")
            else:
                warnings.append(message)
        else:  # unexplained
            blockers.append(
                "UNEXPLAINED pinned llama-server overlap with planned host cpuset "
                f"{plan.host_cpuset}: {detail} — matches no static NUMA_CONFIG "
                "cpuset; refusing (not downgradable by --allow-smt-overlap)"
            )

    if facts.static_smt_overlaps:
        warnings.append(
            "static NUMA_CONFIG co-tenants of the lane SMT slice, sibling-folded "
            "(contention recert required at activation Step 4): "
            f"{facts.static_smt_overlaps}"
        )

    # P2-1: lane TCP port must be free.
    if facts.lane_port_in_use is True:
        blockers.append(f"lane port {plan.port} is already in use")
    elif facts.lane_port_in_use is None:
        warnings.append(f"lane port {plan.port} state could not be probed")

    if not facts.model_file_exists:
        blockers.append(f"tenant model file missing: {plan.model_path}")

    # P2-1: tenant GGUF identity (streamed sha256 vs the policy pin).
    if facts.model_sha256_skipped:
        warnings.append(
            "tenant sha256 verification SKIPPED (--skip-tenant-hash) — "
            "attestation records the skip; identity not verified"
        )
    elif plan.expected_model_sha256 is None:
        warnings.append(
            f"np_ceiling policy pins no sha256 for tenant {plan.tenant_id}; "
            "identity check limited to file existence"
        )
    elif facts.model_sha256 is None:
        if facts.model_file_exists:
            blockers.append("tenant model sha256 could not be computed")
    elif facts.model_sha256 != plan.expected_model_sha256:
        blockers.append(
            "tenant model sha256 mismatch: file "
            f"{facts.model_sha256} != policy pin {plan.expected_model_sha256}"
        )

    if plan.np_ceiling is None:
        blockers.append(
            f"np_ceiling policy has no validated operating point for tenant "
            f"{plan.tenant_id} at slot_context={plan.slot_context_tokens} within "
            f"budget {plan.dynamic_budget_gib} GiB — refuse, never extrapolate"
        )
    elif plan.np_slots > plan.np_ceiling:
        blockers.append(
            f"planned -np {plan.np_slots} exceeds validated ceiling {plan.np_ceiling} "
            f"for slot_context={plan.slot_context_tokens}"
        )

    blockers.extend(facts.errors)
    return blockers, warnings, infos


# ── Gather (subprocess/filesystem; not exercised by unit tests) ──────────────


def _run_json(cmd: list[str], timeout_s: float = 10.0) -> tuple[Any, str | None]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{cmd[0]}: {exc}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, f"{cmd[0]}: non-JSON output ({proc.stdout[:120]!r})"


def _binary_version_output() -> str | None:
    if not LANE_BINARY.exists():
        return None
    try:
        proc = subprocess.run(
            [str(LANE_BINARY), "--version"],
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def _live_llama_processes(host_cpuset: str) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "llama-server"], capture_output=True, text=True, check=False
        )
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for token in proc.stdout.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        status_path = Path(f"/proc/{pid}/status")
        cpus_allowed = None
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    cpus_allowed = line.split(":", 1)[1].strip()
                    break
        except OSError:
            continue
        # Report annotation only — evaluate_preflight reclassifies from the raw
        # mask via classify_live_overlap (P1-1/P1-2 fold + taxonomy).
        overlap_class, overlap, matches = classify_live_overlap(cpus_allowed, host_cpuset)
        records.append(
            {
                "pid": pid,
                "cpus_allowed_list": cpus_allowed,
                "overlap": overlap,
                "overlap_class": overlap_class,
                "static_matches": matches,
            }
        )
    return records


def _lane_port_in_use(port: int) -> bool | None:
    """TCP probe of the lane port on loopback (P2-1). None = probe failed."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return None


def _streamed_sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str | None:
    """Streamed sha256 of a large file (P2-1; ~28 GiB tenant GGUF — constant
    memory, expect minutes of read I/O)."""
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def gather_facts(plan: LanePlan, *, skip_tenant_hash: bool = False) -> PreflightFacts:
    errors: list[str] = []
    # P2-7: pin the query AND the parse to card 0 (the lane device).
    meminfo_payload, meminfo_error = _run_json(
        ["rocm-smi", "-d", "0", "--showmeminfo", "vram", "--json"]
    )
    if meminfo_error:
        errors.append(meminfo_error)
    meminfo = parse_rocm_meminfo(meminfo_payload)
    pids_payload, pids_error = _run_json(["rocm-smi", "--showpids", "--json"])
    if pids_error:
        errors.append(pids_error)
    model_file = Path(plan.model_path)
    model_file_exists = model_file.exists()
    model_sha256: str | None = None
    if model_file_exists and not skip_tenant_hash:
        model_sha256 = _streamed_sha256(model_file)
    return PreflightFacts(
        flag_enabled=lane_enabled(),
        binary_exists=LANE_BINARY.exists(),
        binary_version_output=_binary_version_output(),
        kfd_dev_present=KFD_DEV.exists(),
        kfd_topology_present=KFD_TOPOLOGY.exists(),
        vram_total_gib=meminfo[0] if meminfo else None,
        vram_used_gib=meminfo[1] if meminfo else None,
        gpu_compute_pids=parse_rocm_pids(pids_payload),
        live_llama_processes=_live_llama_processes(plan.host_cpuset),
        model_file_exists=model_file_exists,
        static_smt_overlaps=static_smt_overlap_roles(plan.host_cpuset),
        errors=errors,
        lane_port_in_use=_lane_port_in_use(plan.port),
        model_sha256=model_sha256,
        model_sha256_skipped=bool(skip_tenant_hash),
    )


# ── Plan construction + reporting ────────────────────────────────────────────


def build_plan(policy: NpCeilingPolicy, args: argparse.Namespace) -> LanePlan:
    tenant = policy.tenants[args.tenant]
    # Mode-aware row selection (P2-3). MTP on/off are different capacity
    # frontiers, and some tenants exist in only one of them — the A4 bridge was
    # only ever measured MTP ON, so reading tenant.budgets (the mtp_off rows)
    # would find nothing and report it as a missing PROFILE rather than as a
    # missing MODE. Resolve the mode first so the error names the real cause.
    mode = getattr(args, "mode", MODE_MTP_OFF)
    mode_policy = tenant.mode_policy(mode)
    if mode_policy is None:
        raise KeyError(
            f"tenant {args.tenant!r} has no validated frontier for mode {mode!r} "
            f"(have: {sorted(tenant.modes)}) — refuse, never fall back to the other mode"
        )
    rows = {row.name: row for row in mode_policy.budgets}
    if args.budget_profile not in rows:
        raise KeyError(
            f"budget profile {args.budget_profile!r} not in policy for {args.tenant!r} "
            f"in mode {mode!r} (have: {sorted(rows)})"
        )
    budget = rows[args.budget_profile].dynamic_budget_gib
    ceiling = np_ceiling(
        policy,
        args.tenant,
        dynamic_budget_gib=budget,
        slot_context_tokens=args.slot_context,
        mode=mode,
    )
    return LanePlan(
        tenant_id=args.tenant,
        model_path=tenant.model_path,
        model_vram_gib=tenant.model_vram_gib,
        np_slots=args.np,
        slot_context_tokens=args.slot_context,
        port=args.port,
        host_cpuset=LANE_HOST_CPUSET,
        dynamic_budget_gib=budget,
        np_ceiling=ceiling,
        estimated_dynamic_gib=estimated_dynamic_gib(
            tenant, np_slots=args.np, slot_context_tokens=args.slot_context
        ),
        launch_argv=build_tenant_launch_plan(
            model_path=tenant.model_path,
            np_slots=args.np,
            slot_context_tokens=args.slot_context,
            port=args.port,
        ),
        mode=mode,
        expected_model_sha256=tenant.model_sha256,
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    plan = report["plan"]
    lines = [
        "# GPU Shadow Lane Preflight",
        "",
        f"- status: `{report['status']}`",
        f"- mode: `{report['mode']}`",
        f"- tenant: `{plan['tenant_id']}`",
        f"- plan: `-np {plan['np_slots']}` x `{plan['slot_context_tokens']}` ctx on port `{plan['port']}` ({LANE_DEVICE})",
        f"- np_ceiling: `{plan['np_ceiling']}` (budget `{plan['dynamic_budget_gib']}` GiB)",
        f"- host cpuset: `{plan['host_cpuset']}`",
    ]
    tenant_hash = report.get("tenant_hash") or {}
    if tenant_hash:
        lines.append(
            f"- tenant sha256: `{tenant_hash.get('status')}` "
            f"(streamed hash of a ~28 GiB GGUF costs minutes of read I/O; "
            f"skip with --skip-tenant-hash, which is RECORDED)"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    if report.get("infos"):
        lines.extend(["", "## Informational", ""])
        lines.extend(f"- {info}" for info in report["infos"])
    lines.extend(["", "## Planned launch argv (NOT executed by this probe)", ""])
    lines.append("```bash\n" + " ".join(plan["launch_argv"]) + "\n```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default=TENANT_CANDIDATE_ID)
    parser.add_argument(
        "--budget-profile",
        default="phase2_resident_set",
        help="np_ceiling budget row to plan against (e.g. solo_resident, phase2_resident_set)",
    )
    parser.add_argument(
        "--mode",
        default=MODE_MTP_OFF,
        choices=list(VALID_MODES),
        help="launch mode to plan against (D6: MTP is launch-bound, not per-request)",
    )
    parser.add_argument("--np", type=int, default=8)
    parser.add_argument("--slot-context", type=int, default=8192)
    parser.add_argument("--port", type=int, default=LANE_PORT)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write preflight_attestation.json for the activation choreography. "
            "Refused while any blocker stands. This is the probe's ONLY mutation."
        ),
    )
    parser.add_argument("--require-enabled", action="store_true")
    parser.add_argument(
        "--allow-existing-gpu-pids",
        action="store_true",
        help="Downgrade foreign-GPU-PID blocker to a warning (explicit operator call).",
    )
    parser.add_argument(
        "--allow-smt-overlap",
        action="store_true",
        help=(
            "Acknowledge STATIC-CO-TENANT overlaps (demote their warning to "
            "informational). P1-1 narrowed this flag to that class only: "
            "unexplained pinned overlaps always block, unpinned full-host masks "
            "are informational anyway — a healthy fleet needs no flag."
        ),
    )
    parser.add_argument(
        "--skip-tenant-hash",
        action="store_true",
        help=(
            "Skip the streamed sha256 of the tenant GGUF (~28 GiB, minutes of "
            "read I/O). The skip is RECORDED in the report and attestation."
        ),
    )
    parser.add_argument(
        "--expected-gpu-pid",
        action="append",
        type=int,
        default=[],
        metavar="PID",
        help=(
            "Expected-tenant GPU PID allowlist (repeatable; default empty). "
            "Allowlisted PIDs are reported informationally instead of blocking "
            "the foreign-GPU-PID check."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = utc_stamp()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"gpu_shadow_lane_preflight_{stamp}")
    # The probe is itself the guarded tool: it reads the policy with an explicit
    # override so it can run BEFORE the flag is flipped, while the report always
    # records the environment's real flag state (facts.flag_enabled).
    policy = load_np_ceiling_policy(
        args.policy_path, feats=Features(gpu_shadow_lane=True)
    )
    plan = build_plan(policy, args)
    facts = gather_facts(plan, skip_tenant_hash=args.skip_tenant_hash)
    expected_gpu_pids = frozenset(args.expected_gpu_pid)
    blockers, warnings, infos = evaluate_preflight(
        facts,
        plan,
        require_enabled=args.require_enabled,
        allow_existing_gpu_pids=args.allow_existing_gpu_pids,
        allow_smt_overlap=args.allow_smt_overlap,
        expected_gpu_pids=expected_gpu_pids,
    )

    attested = False
    if args.apply and not blockers:
        attested = True
    status = "blocked" if blockers else ("attested" if attested else "plan_only_ok")
    allow_flags = {
        "allow_existing_gpu_pids": bool(args.allow_existing_gpu_pids),
        "allow_smt_overlap": bool(args.allow_smt_overlap),
        "skip_tenant_hash": bool(args.skip_tenant_hash),
        "expected_gpu_pids": sorted(expected_gpu_pids),
    }
    tenant_hash_status = (
        "skipped"
        if facts.model_sha256_skipped
        else "unpinned"
        if plan.expected_model_sha256 is None
        else "verified"
        if facts.model_sha256 == plan.expected_model_sha256
        else "mismatch"
        if facts.model_sha256 is not None
        else "unreadable"
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "mode": "apply" if args.apply else "plan-only",
        "blockers": blockers,
        "warnings": warnings,
        "infos": infos,
        "allow_flags": allow_flags,
        "tenant_hash": {
            "status": tenant_hash_status,
            "expected": plan.expected_model_sha256,
            "computed": facts.model_sha256,
        },
        "plan": asdict(plan),
        "facts": asdict(facts),
    }
    json_path, md_path = write_report(report, output_dir)
    if attested:
        # P2-2: the attestation derives every boolean from the ACTUAL check
        # results (not literals), records the warnings/infos it was granted
        # under, the allow-flags in force, and a freshness field.
        version_output = facts.binary_version_output or ""
        attestation_path = output_dir / "preflight_attestation.json"
        attestation_path.write_text(
            json.dumps(
                {
                    "lane": "gpu_shadow_lane",
                    "attested_at": report["generated_at"],
                    "generated_at": report["generated_at"],
                    "plan": report["plan"],
                    "warnings": warnings,
                    "infos": infos,
                    "allow_flags": allow_flags,
                    "tenant_hash": report["tenant_hash"],
                    "facts_summary": {
                        "vram_free_gib": (
                            None
                            if facts.vram_total_gib is None or facts.vram_used_gib is None
                            else round(facts.vram_total_gib - facts.vram_used_gib, 2)
                        ),
                        "binary_version_ok": bool(
                            facts.binary_exists
                            and LANE_BINARY_VERSION in version_output
                            and LANE_BINARY_COMMIT in version_output
                        ),
                        "kfd_ok": bool(
                            facts.kfd_dev_present and facts.kfd_topology_present
                        ),
                        "lane_port_free": facts.lane_port_in_use is False,
                        "gpu_compute_pids": facts.gpu_compute_pids,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if not args.summary_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {json_path}")
        print(f"wrote {md_path}")
    if blockers:
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
