"""Canonical cross-role contention matrix re-bench tool.

Phase F of `handoffs/active/cross-role-bw-aware-routing.md`. Replaces the
ad-hoc `/workspace/tmp/contention_matrix*.sh` scripts.

Reads NUMA_CONFIG to enumerate relevant role pairs, runs solo + parallel
HTTP benches against the live llama-server ports, writes a YAML matrix that
the gate (Phase A's `src/scheduling/contention.py`) consumes.

Smart-prune rule: once a pair is measured as catastrophic (ratio < 0.65),
do NOT include triples or larger N-way combinations that contain it — they'd
just re-confirm the catastrophic floor at high time cost. Skipped pairs are
recorded explicitly as `unknown_pairs` with `reason: skipped_due_to_pair_X`
so the gate's unknown-pair policy still applies.

Usage:
    # Re-bench the live stack (assumes stack is up + roles healthy)
    python scripts/server/contention_matrix.py

    # Subset — updates the measured rows and carries every unmeasured pair /
    # unknown-pair entry forward verbatim (never truncates the matrix), and is
    # stamped decision_grade=false because it is not a full re-measurement.
    python scripts/server/contention_matrix.py --roles frontdoor worker_general

    # Dry-run (show what would be measured, don't bench)
    python scripts/server/contention_matrix.py --dry-run

    # Validate the existing matrix without re-running
    python scripts/server/contention_matrix.py --validate-only

Output: writes `orchestration/contention_matrix.yaml` with full metadata
(topology hash, binary commit, host, measured_at). The gate uses the
topology hash to detect when NUMA_CONFIG has drifted from the measured state.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import logging
import platform
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Resolve repo paths regardless of cwd
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "server"))

# Device axis for the feasibility model (rider §2). Imported after the sys.path
# setup above, which is what makes `src.` resolvable when this file is run as a
# standalone script. `device_model` is pure and import-safe: it reads declared
# artifacts only and touches no process.
from src.registry.kernel_paths import server_binary  # noqa: E402
from src.scheduling.device_model import (  # noqa: E402
    DEFAULT_VRAM_HEADROOM_GIB,
    DeviceClass,
    resolve_device_classes,
    vram_capacity_gib as _declared_vram_capacity_gib,
    vram_fit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [matrix] %(levelname)s %(message)s")
log = logging.getLogger("contention_matrix")

PROMPT = (
    "Write a 100-word essay analyzing the philosophical implications of autonomous "
    "decision-making in artificial intelligence systems."
)
N_PREDICT = 100
DEFAULT_FLOOR = 0.85
CATASTROPHIC_FLOOR = 0.65  # below this, skip triples/quads containing the pair
TIMEOUT_PER_REQUEST_S = 180
DEFAULT_OUTPUT = REPO_ROOT / "orchestration" / "contention_matrix.yaml"
AUXILIARY_ROLES = {"eval_batch_frontdoor"}


@dataclass
class BenchResult:
    port: int
    role: str
    tps: float
    elapsed_s: float


@dataclass
class PairBench:
    roles: tuple[str, str]
    instance_a: dict[str, Any]
    instance_b: dict[str, Any]
    solo_a: BenchResult
    solo_b: BenchResult
    parallel_a: BenchResult
    parallel_b: BenchResult
    seq_aggregate_tps: float
    parallel_aggregate_tps: float
    ratio: float
    note: str = ""

    def verdict(self, floor: float = DEFAULT_FLOOR) -> str:
        if self.ratio >= 1.0:
            return "allow"
        if self.ratio >= floor:
            return "borderline"
        return "block"

    def to_yaml_dict(self, floor: float = DEFAULT_FLOOR) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "instance_a": self.instance_a,
            "instance_b": self.instance_b,
            "seq_aggregate_tps": round(self.seq_aggregate_tps, 2),
            "parallel_aggregate_tps": round(self.parallel_aggregate_tps, 2),
            "ratio": round(self.ratio, 2),
            "samples": 1,
            "verdict": self.verdict(floor),
            "note": self.note,
        }


# ── HTTP bench primitive ────────────────────────────────────────────


def _http_bench(port: int, n_predict: int = N_PREDICT, *, safe_sampling: bool = False) -> tuple[float, float]:
    """Single /completion call → (tps, elapsed_s). 0.0 on error.

    safe_sampling=True uses temp>0 + repetition penalty to avoid greedy
    degeneration loops that can produce un-parseable output. Decode speed is
    effectively unchanged; this only steers token selection. Use for the
    quarter-level N-way re-bench (2026-05-26 gemma4 degeneration crash).
    """
    try:
        import httpx
    except ImportError:
        log.error("httpx not available")
        return (0.0, 0.0)

    payload = {
        "prompt": PROMPT,
        "n_predict": n_predict,
        "temperature": 0.7 if safe_sampling else 0,
        "cache_prompt": False,
    }
    if safe_sampling:
        payload["repeat_penalty"] = 1.1
        payload["top_p"] = 0.95
    try:
        with httpx.Client(timeout=TIMEOUT_PER_REQUEST_S) as client:
            resp = client.post(f"http://localhost:{port}/completion", json=payload)
            resp.raise_for_status()
            data = resp.json()
        timings = data.get("timings", {})
        tps = float(timings.get("predicted_per_second", 0.0))
        ms = float(timings.get("predicted_ms", 0.0))
        return (tps, ms / 1000.0 if ms else 0.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("bench port=%d failed: %s", port, exc)
        return (0.0, 0.0)


def _port_healthy(port: int, timeout_s: float = 0.5) -> bool:
    try:
        import httpx

        resp = httpx.get(f"http://localhost:{port}/health", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False


class UnmeasuredLegError(RuntimeError):
    """A bench leg produced no measurement, so no ratio can be derived from it.

    `_http_bench` returns (0.0, 0.0) on timeout or HTTP error. That sentinel is
    NOT a measurement of zero throughput, and the aggregate arithmetic cannot
    tell the two apart — see `_unmeasured_legs` for why the failure is not
    merely lossy but SIGN-BIASED.
    """


def _unmeasured_legs(legs: dict[str, tuple[float, float]]) -> list[str]:
    """Names of legs whose (tps, elapsed_s) shows no measurement was obtained.

    WHY THIS EXISTS, and why the old arithmetic was worse than lossy.

    `par_time = max(par_a_el, par_b_el)` DISCARDS a failed leg: a timeout
    contributes 0.0, so max() silently returns the surviving leg's time. But
    `total_tokens` still counts BOTH legs, so tokens that were never generated
    get divided by one leg's elapsed time.

    THE ERROR IS ONE-DIRECTIONAL. Measured against the old arithmetic with
    5 s solo legs and DEFAULT_FLOOR = 0.85:

        both parallel legs 10 s (real 2x contention)   -> ratio 1.00, allow
        leg B TIMED OUT, leg A 10 s                    -> ratio 1.00, allow
        leg B TIMED OUT, leg A 5 s                     -> ratio 2.00, allow

    Note the middle row: when the failed leg is the one that WOULD have
    dominated max(), its timeout is not merely inflationary, it is INVISIBLE —
    the output is byte-identical to a healthy run. No input of this shape can
    produce a false `block`; every one of them reads as `allow`. So the
    arithmetic reports contention as most benign exactly when a server is
    collapsing under load, which is when it is least benign.

    A 2x48-thread overlapping pair on 48 physical cores is precisely the regime
    where the 180 s timeout bites, so the defect fires hardest on the
    measurement someone most wants to trust.

    The module already refuses to treat HTTP failures as throughput evidence
    for the pre-flight health check ("HTTP failures are not throughput
    evidence"); this applies the same rule to the bench legs themselves.
    """
    return sorted(
        name for name, (tps, elapsed_s) in legs.items()
        if elapsed_s <= 0.0 or tps <= 0.0
    )


def _bench_pair(
    role_a: str,
    port_a: int,
    role_b: str,
    port_b: int,
    *,
    instance_a: dict[str, Any] | None = None,
    instance_b: dict[str, Any] | None = None,
) -> PairBench:
    """Measure solo + parallel for one role pair. Returns a PairBench.

    Raises `UnmeasuredLegError` if any of the four legs failed to produce a
    measurement, rather than deriving a ratio from a partial sample.
    """
    log.info("bench pair: %s (port %d) + %s (port %d)", role_a, port_a, role_b, port_b)

    # Solo
    log.info("  solo %s:%d ...", role_a, port_a)
    solo_a_tps, solo_a_el = _http_bench(port_a)
    log.info("  solo %s:%d ...", role_b, port_b)
    solo_b_tps, solo_b_el = _http_bench(port_b)

    # Parallel
    log.info("  parallel %s:%d || %s:%d ...", role_a, port_a, role_b, port_b)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(_http_bench, port_a)
        fut_b = ex.submit(_http_bench, port_b)
        par_a_tps, par_a_el = fut_a.result()
        par_b_tps, par_b_el = fut_b.result()

    failed = _unmeasured_legs({
        f"solo_{role_a}:{port_a}": (solo_a_tps, solo_a_el),
        f"solo_{role_b}:{port_b}": (solo_b_tps, solo_b_el),
        f"parallel_{role_a}:{port_a}": (par_a_tps, par_a_el),
        f"parallel_{role_b}:{port_b}": (par_b_tps, par_b_el),
    })
    if failed:
        raise UnmeasuredLegError(
            f"{role_a} + {role_b}: no ratio derivable, these legs produced no "
            f"measurement: {', '.join(failed)}. A timed-out leg is not zero "
            f"throughput — deriving a ratio here would report contention as "
            f"MORE benign than it is (see _unmeasured_legs)."
        )

    total_tokens = N_PREDICT * 2
    seq_time = (solo_a_el + solo_b_el) or 0.001
    par_time = max(par_a_el, par_b_el) or 0.001
    seq_agg = total_tokens / seq_time
    par_agg = total_tokens / par_time
    ratio = par_agg / seq_agg if seq_agg > 0 else 0.0

    log.info(
        "  → seq=%.2f t/s par=%.2f t/s ratio=%.2f",
        seq_agg, par_agg, ratio,
    )

    return PairBench(
        roles=tuple(sorted([role_a, role_b])),  # type: ignore[arg-type]
        instance_a=instance_a or {"port": port_a},
        instance_b=instance_b or {"port": port_b},
        solo_a=BenchResult(port=port_a, role=role_a, tps=solo_a_tps, elapsed_s=solo_a_el),
        solo_b=BenchResult(port=port_b, role=role_b, tps=solo_b_tps, elapsed_s=solo_b_el),
        parallel_a=BenchResult(port=port_a, role=role_a, tps=par_a_tps, elapsed_s=par_a_el),
        parallel_b=BenchResult(port=port_b, role=role_b, tps=par_b_tps, elapsed_s=par_b_el),
        seq_aggregate_tps=seq_agg,
        parallel_aggregate_tps=par_agg,
        ratio=ratio,
    )


# ── Enumeration + smart-prune ───────────────────────────────────────


def _enumerate_full_pairs(numa_config: dict, role_filter: set[str] | None = None) -> list[tuple[str, str]]:
    """Enumerate (role_a, role_b) pairs of full instances. Sorted + deduped."""
    roles = sorted(r for r in numa_config.keys() if r not in AUXILIARY_ROLES)
    if role_filter:
        roles = [r for r in roles if r in role_filter]
    out: list[tuple[str, str]] = []
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            if not numa_config[a].get("instances") or not numa_config[b].get("instances"):
                continue
            out.append((a, b))
    return out


def _full_port(numa_config: dict, role: str) -> int | None:
    instances = numa_config.get(role, {}).get("instances", [])
    if not instances:
        return None
    return int(instances[0][1])


def _matrix_roles(numa_config: dict, role_filter: set[str] | None = None) -> list[str]:
    roles = sorted(r for r in numa_config if r not in AUXILIARY_ROLES)
    if role_filter:
        roles = [role for role in roles if role in role_filter]
    return roles


def _shape_label(idx: int, regions: frozenset) -> str:
    """Human-readable SHAPE name for an instance record's footprint.

    Derived from the atomic CPU regions the instance occupies — the only sound
    discriminator (see `canonical_shape_for_regions`). An unrecognised
    footprint gets a visibly non-committal `inst<idx>` so the label can never
    assert a shape the geometry does not support: the predecessor rule
    (`"full" if idx == 0 else f"q{idx - 1}"`) named every secondary instance a
    quarter, which since the 2026-07-30 quarter retirement mislabelled every
    live HALF as a quarter.
    """
    from src.runtime import instance_topology

    return instance_topology.canonical_shape_for_regions(regions) or f"inst{idx}"


def _instance_record(role: str, idx: int, port: int, regions: frozenset, numa_config: dict) -> dict[str, Any]:
    inst = numa_config[role]["instances"][idx]
    label = _shape_label(idx, regions)
    return {
        "role": role,
        "label": label,
        "instance_idx": idx,
        "port": int(port),
        "cpu_list": str(inst[0]),
        "threads": int(inst[2]),
        "regions": sorted(regions),
    }


def _live_role_instances(numa_config: dict, role: str) -> list[dict[str, Any]]:
    return [
        _instance_record(role, idx, port, regions, numa_config)
        for idx, port, regions in _role_footprints(numa_config, role, live_only=True)
    ]


def _select_live_pair_instances(
    numa_config: dict,
    role_a: str,
    role_b: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Choose live instances for a pair, preferring disjoint CPU regions.

    This is the v7/quarter-mode recert path: instance-0/full ports may be
    intentionally offline while quarter ports are live. A role with no healthy
    live instance is an infra blocker, not a zero-throughput measurement.

    The third element is the geometry marker: None for a DISJOINT placement
    (the geometry the gate's role-keyed lookup is read as), and
    "overlap_measured" for the overlapping fallback. A caller MUST NOT record
    the fallback as a plain `pairs:` row: an overlapping-geometry number
    substituted into the matrix is exactly what put the frontdoor+ingest 1.89
    "disjoint" row in front of the role-keyed gate (shape-keyed handoff APPEND
    2026-08-12, inverted marker polarity — the gate cannot tell the two
    geometries apart). `cmd_run` refuses the fallback into `unknown_pairs`.
    """
    a_live = _live_role_instances(numa_config, role_a)
    b_live = _live_role_instances(numa_config, role_b)
    if not a_live or not b_live:
        missing = [
            role
            for role, live in ((role_a, a_live), (role_b, b_live))
            if not live
        ]
        return None, None, f"missing_live_instance:{','.join(missing)}"

    def _sort_key(item: dict[str, Any]) -> tuple[int, int]:
        return (len(item["regions"]), int(item["instance_idx"]))

    for a in sorted(a_live, key=_sort_key):
        for b in sorted(b_live, key=_sort_key):
            if set(a["regions"]).isdisjoint(set(b["regions"])):
                return a, b, None
    # No disjoint placement exists (e.g. architect_full against a quarter role).
    # Measure the least-broad live pair so the matrix records the live contention
    # rather than silently using a dead primary port. The marker names the
    # substituted geometry; `cmd_run` refuses this fallback into `unknown_pairs`
    # rather than recording it as a measured pair row.
    return sorted(a_live, key=_sort_key)[0], sorted(b_live, key=_sort_key)[0], "overlap_measured"


# ── Environment metadata ────────────────────────────────────────────


def _binary_metadata(binary_path: Path) -> dict[str, str]:
    """llama-server binary path + git commit (extracted from --version output)."""
    info = {"llama_server_path": str(binary_path), "git_commit": ""}
    if not binary_path.exists():
        return info
    try:
        out = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True, text=True, timeout=10,
        )
        for line in (out.stdout + out.stderr).splitlines():
            if "build:" in line.lower():
                # Format: "build: 2ffbdbbba (8957)"
                parts = line.split()
                for token in parts:
                    if token.startswith("(") and token.endswith(")"):
                        continue
                    if len(token) >= 7 and all(c in "0123456789abcdef" for c in token):
                        info["git_commit"] = token
                        break
                break
    except Exception:
        pass
    return info


def _host_metadata() -> dict[str, str]:
    try:
        uptime = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        uptime = ""
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "uptime": uptime,
    }


# ── Host-health provenance ───────────────────────────────────────────
#
# ORIGIN. 2026-08-12, dry-running the OP-21 re-bench. `_host_metadata()` already
# collected `uptime` and `kernel`; `_emit_yaml` then wrote only `host:
# <hostname>` and threw both away. A matrix measured at 14.1 d uptime therefore
# landed INDISTINGUISHABLE from one measured on a freshly rebooted host, and no
# later reader could tell which they were holding. That is what makes a reboot
# window worthless: the number it buys cannot prove the host was clean.
#
# The rule set is NOT re-implemented here. `host_health_warnings()` (uptime /
# kernel.numa_balancing / stray llama processes) and `cpu_freq_static_warnings()`
# (cpufreq boost + scaling_max_freq caps) in epyc-inference-research are the
# single authority for MEASUREMENT.md P-BENCH-1/P-BENCH-3 host state. A second
# copy of those thresholds here would drift, and a drifted copy that renders
# "clean" is strictly worse than no field at all.
#
# FAIL-SAFE is the whole point of the block below: every path that cannot
# establish host state emits `host_health_status: "unknown"` plus an explicit
# warning naming what failed, and `decision_grade: false`. An absent or
# unreadable input must never render as a pass.

_RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
_RESEARCH_IMPORT_DIRS = (
    _RESEARCH_ROOT / "scripts" / "benchmark",
    _RESEARCH_ROOT / "scripts" / "lib",
)
_HOST_HEALTH_RULE_SOURCE = (
    "epyc-inference-research/scripts/benchmark/server_np_sweep.py:host_health_warnings"
    " + server_numa_np_sweep.py:cpu_freq_static_warnings"
)

HOST_HEALTH_CLEAN = "clean"
HOST_HEALTH_WARN = "warn"
HOST_HEALTH_UNKNOWN = "unknown"


def _load_host_health_rules():
    """Import the research repo's host-health rule set. Raises on failure.

    Cross-repo import by absolute path is the established pattern in this repo
    (see scripts/analysis/reviewer_policy_arm_ab.py, scripts/graph_router/…).
    Both modules are stdlib-only at import time, so this pulls in no server,
    no model and no inference.
    """
    for directory in _RESEARCH_IMPORT_DIRS:
        entry = str(directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import server_np_sweep  # noqa: PLC0415
    import server_numa_np_sweep  # noqa: PLC0415

    return server_np_sweep, server_numa_np_sweep


_PORT_ARG_RE = re.compile(r"(?:^|\s)--port(?:=|\s+)(\d+)(?=\s|$)")


def _expected_lineup_ports() -> frozenset[int]:
    """Every port the declared stack lineup may legitimately occupy.

    Union of the launch manifest's primary ``PORT_MAP``, every NUMA_CONFIG
    instance (fulls + halves/quarters) and the embedder pool. Declared artifacts
    only — no process is touched. An explicit-only lane (e.g. the GPU shadow
    lane) is deliberately NOT in the lineup: it is exactly the kind of extra
    tenant a contention measurement must not absorb silently.
    """
    import stack_manifest  # noqa: PLC0415
    from stack_numa import NUMA_CONFIG  # noqa: PLC0415

    ports = {int(p) for p in stack_manifest.PORT_MAP.values()}
    ports |= {
        int(inst[1])
        for cfg in NUMA_CONFIG.values()
        for inst in (cfg.get("instances") or [])
    }
    ports |= {int(p) for p in stack_manifest.EMBEDDER_PORTS}
    return frozenset(ports)


def _llama_process_port(proc: dict[str, Any]) -> int | None:
    """The ``--port`` a llama process was launched with, or None if unattributable."""
    args = str(proc.get("args") or proc.get("cmd") or "")
    match = _PORT_ARG_RE.search(args)
    return int(match.group(1)) if match else None


def _split_llama_processes(
    processes: list[dict[str, Any]], expected_ports: frozenset[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition observed llama processes into (lineup instrument, FOREIGN).

    A process is the instrument only if it listens on a declared lineup port.
    Everything else — an undeclared port, or no attributable port at all
    (llama-bench, llama-cli, a server launched via env) — is foreign.
    """
    lineup: list[dict[str, Any]] = []
    foreign: list[dict[str, Any]] = []
    for proc in processes:
        port = _llama_process_port(proc)
        (lineup if port is not None and port in expected_ports else foreign).append(proc)
    return lineup, foreign


def _host_health_probe(
    *,
    host_meta: dict[str, str] | None = None,
    extra_blockers: list[str] | None = None,
    load_rules=_load_host_health_rules,
    expected_ports=None,
    resolve_lineup=_expected_lineup_ports,
) -> dict[str, Any]:
    """Collect host-health provenance for the matrix artifact. Never raises.

    Returns a dict that is always fully populated. `status` is one of
    clean / warn / unknown; `decision_grade` is true ONLY when the gating
    warning list is empty AND the attestation was actually collected.

    `decision_grade` here attests HOST STATE ONLY. It says nothing about the
    statistical adequacy of the ratios (the pairwise emitter records
    `samples: 1` per cell) — see the comment written into the artifact.
    """
    meta = dict(host_meta if host_meta is not None else _host_metadata())
    probe: dict[str, Any] = {
        "hostname": meta.get("hostname", ""),
        "kernel": meta.get("kernel", ""),
        "uptime": meta.get("uptime", ""),
        "uptime_seconds": None,
        "numa_balancing": None,
        "scaling_governors": [],
        "loadavg": "",
        "llama_processes_at_attestation": None,
        "llama_processes_lineup": None,
        "llama_processes_foreign": [],
        "attestation_status": "unavailable",
        "attestation_error": "",
        "rule_source": _HOST_HEALTH_RULE_SOURCE,
        "status": HOST_HEALTH_UNKNOWN,
        "warnings": [],
        "structural_for_harness": [],
        "decision_grade": False,
        "decision_grade_blockers": list(extra_blockers or []),
    }

    try:
        np_sweep, numa_sweep = load_rules()
        attestation = np_sweep.collect_attestation()
        freq_warnings = list(numa_sweep.cpu_freq_static_warnings())

        # The FULL list is what the artifact records — nothing is filtered out
        # of the record.
        full = list(np_sweep.host_health_warnings(attestation)) + freq_warnings

        # The GATING list re-runs the SAME rule set against an attestation with
        # this harness's own instrument elided. A contention matrix benches the
        # LIVE stack, so llama-server presence is the instrument, not
        # contamination; if it gated, `decision_grade` could never be true and
        # the field would be vacuous. Deriving the structural subset by set
        # difference (never by matching a warning's spelling) keeps the rule
        # owner in charge: a reworded or newly added process-derived warning
        # re-classifies itself automatically.
        #
        # Only the DECLARED lineup is the instrument. On a shared host a
        # foreign llama process (another agent's server, a llama-bench) is
        # exactly the contamination a contention measurement must not absorb,
        # so it stays in the gating attestation and fires the rule there.
        observed = list(attestation.get("existing_llama_processes") or [])
        foreign_blockers: list[str] = []
        lineup_procs: list[dict[str, Any]] = []
        foreign_procs: list[dict[str, Any]] = []
        if observed:
            try:
                ports = frozenset(
                    int(p)
                    for p in (expected_ports if expected_ports is not None else resolve_lineup())
                )
                lineup_procs, foreign_procs = _split_llama_processes(observed, ports)
            except Exception as exc:  # noqa: BLE001 — unresolvable lineup = nothing is waived
                foreign_procs = observed
                foreign_blockers.append(
                    "expected stack lineup could not be resolved "
                    f"({type(exc).__name__}: {exc}); no llama process can be "
                    "attributed to the instrument, so none is waived"
                )
            if foreign_procs:
                described = ", ".join(
                    f"pid {p.get('pid', '?')} (port {_llama_process_port(p) or 'unattributable'})"
                    for p in foreign_procs
                )
                foreign_blockers.append(
                    f"{len(foreign_procs)} foreign llama process(es) outside the "
                    f"declared stack lineup: {described}"
                )
        instrument_elided = dict(attestation)
        instrument_elided["existing_llama_processes"] = foreign_procs
        gating = (
            list(np_sweep.host_health_warnings(instrument_elided))
            + freq_warnings
            + foreign_blockers
        )
        full = full + foreign_blockers

        structural = [w for w in full if w not in gating]

        # An unreadable /proc/uptime makes the uptime rule silently not fire —
        # a missing input that would otherwise render as a pass. Refuse it.
        uptime_seconds = attestation.get("uptime_seconds")
        if not isinstance(uptime_seconds, (int, float)):
            unreadable = (
                "host uptime could not be read (/proc/uptime); the uptime rule "
                "did not evaluate, so host cleanliness is UNKNOWN, not clean"
            )
            full.append(unreadable)
            gating.append(unreadable)
            probe["attestation_status"] = "incomplete"
        else:
            probe["attestation_status"] = "collected"

        probe.update(
            {
                "uptime_seconds": uptime_seconds if isinstance(uptime_seconds, (int, float)) else None,
                "numa_balancing": attestation.get("numa_balancing"),
                "scaling_governors": list(attestation.get("scaling_governors") or []),
                "loadavg": str(attestation.get("loadavg") or ""),
                "llama_processes_at_attestation": len(observed),
                "llama_processes_lineup": len(lineup_procs),
                "llama_processes_foreign": [
                    {"pid": str(p.get("pid", "")), "port": _llama_process_port(p)}
                    for p in foreign_procs
                ],
                "warnings": full,
                "structural_for_harness": structural,
            }
        )
        probe["decision_grade_blockers"] = list(extra_blockers or []) + gating
        if probe["attestation_status"] == "collected":
            probe["status"] = HOST_HEALTH_CLEAN if not gating else HOST_HEALTH_WARN
        else:
            probe["status"] = HOST_HEALTH_UNKNOWN
    except Exception as exc:  # noqa: BLE001 — the probe must never break a run
        reason = f"{type(exc).__name__}: {exc}"
        blocker = (
            "host health could not be determined "
            f"({_HOST_HEALTH_RULE_SOURCE} unavailable: {reason}); "
            "this artifact is NOT decision-grade"
        )
        probe["attestation_error"] = reason
        probe["warnings"] = [blocker]
        probe["decision_grade_blockers"] = list(extra_blockers or []) + [blocker]
        probe["status"] = HOST_HEALTH_UNKNOWN

    probe["decision_grade"] = (
        probe["status"] == HOST_HEALTH_CLEAN and not probe["decision_grade_blockers"]
    )
    return probe


def read_matrix_host_health(path: Path) -> dict[str, Any]:
    """Read a matrix's host-health stamp. A matrix without one is UNKNOWN.

    Read side of the same rule: a matrix written before this provenance block
    existed carries no attestation, and the ONLY honest reading of that is
    "unknown", never "clean". Non-gating by design — this reports, it does not
    admit or refuse.
    """
    record = {
        "status": HOST_HEALTH_UNKNOWN,
        "decision_grade": False,
        "warnings": [],
        "reason": "",
    }
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        record["reason"] = f"matrix could not be read: {type(exc).__name__}: {exc}"
        return record
    if not isinstance(data, dict) or "host_health_status" not in data:
        record["reason"] = (
            "matrix carries no host-health stamp (written before provenance was "
            "emitted); host state at measurement time is UNKNOWN, not clean"
        )
        return record
    record["status"] = str(data.get("host_health_status") or HOST_HEALTH_UNKNOWN)
    record["decision_grade"] = bool(data.get("decision_grade"))
    record["warnings"] = [str(w) for w in (data.get("host_health_warnings") or [])]
    return record


def describe_matrix_host_health(path: Path) -> list[str]:
    """Human-readable lines for the host-health stamp of a matrix on disk."""
    record = read_matrix_host_health(path)
    lines = [
        f"matrix host health: {record['status']} "
        f"(decision_grade={str(record['decision_grade']).lower()})"
    ]
    if record["reason"]:
        lines.append(f"  {record['reason']}")
    for warning in record["warnings"]:
        lines.append(f"  warning: {warning}")
    return lines


# ── YAML emission ────────────────────────────────────────────────────

# Top-level sections this emitter regenerates from a fresh bench run.  EVERY
# other top-level section in an existing matrix file is hand-authored runtime
# policy (`nway_light_roles`, `nway_heavy_roles`, `same_role`,
# `same_role_certifications`, `n_way`, `triples`,
# `n_way_full_instance_coarse`, ...) that `src/scheduling/contention.py`
# reads at admission time and that only the `bench-nway` / `bench-within-role`
# subcommands produce — as FRAGMENTS, for hand-merge.  A pairwise re-bench
# therefore MUST carry them through verbatim instead of truncating the file
# (origin: cd42def3 / a517793c silently deleted the whole N-way policy block
# while claiming only "matrix refreshed on the restored lineup", which
# degraded the N-way admission gate to conservative QUEUE).
_EMITTER_OWNED_SECTIONS = frozenset(
    {
        "version",
        "measured_at",
        "host",
        # Host-health provenance is regenerated from THIS run's probe. Carrying
        # a previous run's block forward would be the exact failure this block
        # exists to prevent: a stale "clean" attesting a run that never made it.
        "host_kernel",
        "host_uptime",
        "host_health_status",
        "host_health_warnings",
        "host_health_structural_for_harness",
        "decision_grade",
        "decision_grade_blockers",
        "host_provenance",
        "binary",
        "topology_hash",
        "default_floor",
        "pairs",
        "unknown_pairs",
    }
)

_TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")

# One `pairs:` / `unknown_pairs:` entry line as emitted by `_emit_yaml`:
# `  - roles: ['a', 'b']` — the capture is the roles-list literal.
_PAIR_ENTRY_RE = re.compile(r"^  - roles: \[(.*)\]\s*$")


def _split_top_level_sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a matrix YAML document into ordered (top_level_key, lines) blocks.

    Comment / blank lines immediately preceding a key are attached to that key's
    block so provenance comments travel with the section they annotate.
    """
    sections: list[tuple[str, list[str]]] = []
    pending: list[str] = []
    current_key: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_key is not None:
            sections.append((current_key, list(current_lines)))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            pending.append(line)
            continue
        m = _TOP_LEVEL_KEY_RE.match(line)
        if m:
            _flush()
            current_key = m.group(1)
            current_lines = [*pending, line]
            pending = []
            continue
        if current_key is None:
            # Content before any top-level key — malformed for our purposes.
            raise ValueError(f"unparseable line before first top-level key: {line!r}")
        current_lines.extend(pending)
        current_lines.append(line)
        pending = []
    _flush()
    return sections


def _carry_forward_sections(existing_path: Path | None) -> list[tuple[str, list[str]]]:
    """Return the hand-authored policy sections of an existing matrix file.

    Returns [] when there is no existing file.  Raises when the file exists but
    cannot be split — refusing to write is strictly better than silently
    dropping the runtime policy the admission gate reads.
    """
    if existing_path is None or not existing_path.exists():
        return []
    text = existing_path.read_text()
    if not text.strip():
        return []
    sections = _split_top_level_sections(text)
    return [(k, lines) for k, lines in sections if k not in _EMITTER_OWNED_SECTIONS]


def _split_pair_entries(block_lines: list[str]) -> list[tuple[tuple[str, str], list[str]]]:
    """Split a `pairs:` / `unknown_pairs:` block into (sorted-role-key, lines) units.

    Entries have the emitter's own shape: a `  - roles: [...]` line followed by
    indented field lines. The role key is parsed from the roles line; the
    entry's lines are preserved VERBATIM so a carried-forward row keeps its
    original numbers and notes. Content that is neither the section header nor
    an entry raises — refusing beats silently dropping a row.
    """
    entries: list[tuple[tuple[str, str], list[str]]] = []
    current_key: tuple[str, str] | None = None
    current_lines: list[str] = []
    pending: list[str] = []

    def _flush() -> None:
        if current_key is not None:
            entries.append((current_key, list(current_lines)))

    for line in block_lines:
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            pending.append(line)
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:$", line):
            continue  # the section header (`pairs:` / `unknown_pairs:`)
        m = _PAIR_ENTRY_RE.match(line)
        if m:
            _flush()
            roles = ast.literal_eval(f"[{m.group(1)}]")
            if not isinstance(roles, list) or len(roles) != 2:
                raise ValueError(f"pair entry roles not a 2-list: {line!r}")
            current_key = tuple(sorted(roles))  # type: ignore[arg-type]
            current_lines = [*pending, line]
            pending = []
            continue
        if current_key is None:
            raise ValueError(f"unparseable line before first pair entry: {line!r}")
        current_lines.extend(pending)
        current_lines.append(line)
        pending = []
    _flush()
    return entries


def _preserve_unmeasured_entries(
    existing_path: Path | None,
    measured_pair_keys: set[tuple[str, str]],
    fresh_unknown_keys: set[tuple[str, str]],
) -> tuple[list[tuple[tuple[str, str], list[str]]], list[tuple[tuple[str, str], list[str]]]]:
    """The existing matrix's `pairs:` / `unknown_pairs:` entries, verbatim, minus
    any role pair THIS run measured.

    Scoped (`--roles`) runs otherwise TRUNCATE the matrix: both sections are
    emitter-owned, so `_carry_forward_sections` drops them and the emitted
    file keeps only the measured subset (handoff APPEND 2026-08-12: 3 pairs
    in, 1 out — against the default output that destroyed 14 of 15 rows and
    degraded the admission gate to fail-closed for every other pair).
    Filtering is by the exact role-pair key, so a partially-scoped run
    preserves every pair that does not name a measured key.
    """
    if existing_path is None or not existing_path.exists():
        return [], []
    sections = _split_top_level_sections(existing_path.read_text())
    by_key = {key: lines for key, lines in sections}
    pair_entries = _split_pair_entries(by_key.get("pairs", [])) if "pairs" in by_key else []
    unknown_entries = (
        _split_pair_entries(by_key.get("unknown_pairs", []))
        if "unknown_pairs" in by_key
        else []
    )
    kept_pairs = [e for e in pair_entries if e[0] not in measured_pair_keys]
    kept_unknown = [e for e in unknown_entries if e[0] not in fresh_unknown_keys]
    return kept_pairs, kept_unknown


def _host_health_unknown_probe() -> dict[str, Any]:
    """The explicit UNKNOWN probe: what a missing probe must render as (fail-safe).

    Every emit path that cannot establish host state must stamp
    ``decision_grade: false`` with a warning naming what could not be
    established — an absent or unreadable input must never render as a pass.
    """
    return {
        "status": HOST_HEALTH_UNKNOWN,
        "warnings": [
            "host health was not probed by the emitter's caller; "
            "this artifact is NOT decision-grade"
        ],
        "structural_for_harness": [],
        "decision_grade": False,
        "decision_grade_blockers": [
            "host health was not probed by the emitter's caller"
        ],
        "attestation_status": "not_probed",
    }


def _host_health_stamp_lines(host_health: dict[str, Any] | None) -> list[str]:
    """The host-health provenance block, shared by the full-matrix emitter and the
    J4b N-way fragment writer (SC21, orchestrator 77e5a214). One rendering for
    both artifacts so the fragment can never drift from the matrix: ``status``
    clean | warn | unknown, the complete unfiltered warning list, the
    harness-structural subset, ``decision_grade`` (true ONLY when status is
    clean AND the blocker list is empty), and the ``host_provenance`` record.
    FAIL-SAFE: a caller without a probe is stamped explicit UNKNOWN, never a
    silently absent block that a later reader would read as clean.
    """
    if host_health is None:
        host_health = _host_health_unknown_probe()

    def _inline(value: Any) -> str:
        return json.dumps(value, sort_keys=True)

    def _str_list(key: str) -> list[str]:
        return [str(v) for v in (host_health.get(key) or [])]

    lines: list[str] = []
    lines.append("")
    lines.append("# Host-health provenance (P-BENCH-1/P-BENCH-3). `decision_grade` attests HOST")
    lines.append("# STATE ONLY: it does NOT assert statistical adequacy — every pair below is")
    lines.append("# `samples: 1`. status is clean | warn | unknown; `unknown` means the probe")
    lines.append("# could not establish host state and MUST NOT be read as clean.")
    lines.append("# `host_health_warnings` is the complete unfiltered rule output.")
    lines.append("# `host_health_structural_for_harness` is the subset waived from gating because")
    lines.append("# a contention matrix benches the LIVE stack by design (llama-server presence is")
    lines.append("# the instrument). Only processes on a DECLARED lineup port are waived; a foreign")
    lines.append("# llama process (undeclared or unattributable port) gates and is listed under")
    lines.append("# host_provenance.llama_processes_foreign.")
    lines.append(f'host_health_status: "{host_health.get("status", HOST_HEALTH_UNKNOWN)}"')
    warnings = _str_list("warnings")
    if warnings:
        lines.append("host_health_warnings:")
        for w in warnings:
            lines.append(f"  - {_inline(w)}")
    else:
        lines.append("host_health_warnings: []")
    structural = _str_list("structural_for_harness")
    if structural:
        lines.append("host_health_structural_for_harness:")
        for w in structural:
            lines.append(f"  - {_inline(w)}")
    else:
        lines.append("host_health_structural_for_harness: []")
    lines.append(f"decision_grade: {str(bool(host_health.get('decision_grade'))).lower()}")
    blockers = _str_list("decision_grade_blockers")
    if blockers:
        lines.append("decision_grade_blockers:")
        for w in blockers:
            lines.append(f"  - {_inline(w)}")
    else:
        lines.append("decision_grade_blockers: []")
    lines.append("host_provenance:")
    for key in (
        "hostname",
        "uptime_seconds",
        "numa_balancing",
        "scaling_governors",
        "loadavg",
        "llama_processes_at_attestation",
        "llama_processes_lineup",
        "llama_processes_foreign",
        "attestation_status",
        "attestation_error",
        "rule_source",
    ):
        lines.append(f"  {key}: {_inline(host_health.get(key))}")
    return lines


def _emit_yaml(
    pairs: list[PairBench],
    *,
    same_role_verdicts: dict[str, str] | None = None,
    unknown_pairs: list[tuple[str, str, str]] | None = None,
    topology_hash: str = "",
    binary: dict[str, str] | None = None,
    host: str = "",
    host_health: dict[str, Any] | None = None,
    floor: float = DEFAULT_FLOOR,
    preserve_sections: list[tuple[str, list[str]]] | None = None,
    preserved_pair_entries: list[tuple[tuple[str, str], list[str]]] | None = None,
    preserved_unknown_entries: list[tuple[tuple[str, str], list[str]]] | None = None,
) -> str:
    """Render the matrix as YAML (no PyYAML dump dependency — handle ourselves)."""
    def _inline(value: Any) -> str:
        return json.dumps(value, sort_keys=True)

    # FAIL-SAFE. A caller that does not pass a probe gets an explicit UNKNOWN,
    # never a silently absent block that a later reader would read as clean.
    if host_health is None:
        host_health = _host_health_unknown_probe()

    lines: list[str] = []
    lines.append("# Auto-generated by scripts/server/contention_matrix.py")
    lines.append(f"# Generated {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("version: 1")
    lines.append(f'measured_at: "{datetime.now(timezone.utc).isoformat()}"')
    lines.append(f'host: "{host}"')
    lines.append(f'host_kernel: {_inline(str(host_health.get("kernel", "")))}')
    lines.append(f'host_uptime: {_inline(str(host_health.get("uptime", "")))}')

    lines.extend(_host_health_stamp_lines(host_health))

    lines.append("")
    lines.append("binary:")
    if binary:
        for k, v in binary.items():
            lines.append(f'  {k}: "{v}"')
    lines.append(f'topology_hash: "{topology_hash}"')
    lines.append(f"default_floor: {floor}")

    lines.append("")
    lines.append("pairs:")
    for p in sorted(pairs, key=lambda x: x.ratio):  # block first, allow last
        d = p.to_yaml_dict(floor)
        lines.append(f'  - roles: [{", ".join(repr(r) for r in d["roles"])}]')
        lines.append(f"    instance_a: {_inline(d['instance_a'])}")
        lines.append(f"    instance_b: {_inline(d['instance_b'])}")
        lines.append(f"    seq_aggregate_tps: {d['seq_aggregate_tps']}")
        lines.append(f"    parallel_aggregate_tps: {d['parallel_aggregate_tps']}")
        lines.append(f"    ratio: {d['ratio']}")
        lines.append(f"    samples: {d['samples']}")
        lines.append(f'    verdict: "{d["verdict"]}"')
        if d.get("note"):
            lines.append(f'    note: "{d["note"]}"')

    # A scoped run must UPDATE the measured rows, not truncate the matrix: the
    # previous run's entries for role pairs this run did NOT measure are
    # carried forward verbatim (handoff APPEND 2026-08-12, `--roles` TRUNCATES).
    fresh_pair_keys = {tuple(sorted(p.roles)) for p in pairs}
    kept_pair_entries = [
        entry for entry in (preserved_pair_entries or []) if entry[0] not in fresh_pair_keys
    ]
    if kept_pair_entries:
        lines.append("  # carried forward verbatim — role pair not measured by this run")
        for _key, entry_lines in kept_pair_entries:
            lines.extend(entry_lines)

    if same_role_verdicts:
        lines.append("")
        lines.append("same_role:")
        for role in sorted(same_role_verdicts.keys()):
            lines.append(f'  - role: "{role}"')
            lines.append(f'    verdict: "{same_role_verdicts[role]}"')

    fresh_unknown_keys = {tuple(sorted([a, b])) for a, b, _reason in unknown_pairs or []}
    kept_unknown_entries = [
        entry for entry in (preserved_unknown_entries or []) if entry[0] not in fresh_unknown_keys
    ]
    if unknown_pairs or kept_unknown_entries:
        lines.append("")
        lines.append("unknown_pairs:")
        for a, b, reason in sorted(unknown_pairs or []):
            lines.append(f"  - roles: [{repr(a)}, {repr(b)}]")
            lines.append(f'    reason: "{reason}"')
        if kept_unknown_entries:
            lines.append("  # carried forward verbatim — role pair not measured by this run")
            for _key, entry_lines in kept_unknown_entries:
                lines.extend(entry_lines)

    # Carry every hand-authored runtime-policy section through verbatim.  A
    # freshly measured `same_role:` block (from same_role_verdicts) supersedes
    # the preserved one; everything else is preserved unconditionally.
    for key, block in preserve_sections or []:
        if key == "same_role" and same_role_verdicts:
            continue
        lines.append("")
        lines.extend(block)

    lines.append("")
    return "\n".join(lines)


# ── J4a: N-way candidate enumeration (no inference) ──────────────────
#
# Cross-role N-way closure (handoffs/active/cross-role-nway-contention-matrix.md).
# Reads the live role topology + the existing pairwise matrix and emits a
# deterministic, topology-stamped manifest of:
#   * candidate_sets   — non-trivial N-way active sets whose every constituent
#                        pair is bulk-allowed (ratio >= floor, measured); these
#                        REQUIRE J4b measurement before any cross-role launch.
#   * excluded_sets    — every other size>=3 set, each with the concrete
#                        lower-order evidence (block / below-floor / unknown
#                        pair, or a measured failed triple) that pruned it.
# Pairwise-allowed is a precondition only; nothing here is launch-certified.


def _pair_key_str(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


def _read_matrix_triples(matrix_path: Path) -> list[dict[str, Any]]:
    """Read the informational `triples:` block (not parsed by the dataclass)."""
    try:
        import yaml

        data = yaml.safe_load(matrix_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read triples from %s: %s", matrix_path, exc)
        return []
    out: list[dict[str, Any]] = []
    for entry in data.get("triples", []) or []:
        roles = entry.get("roles")
        if isinstance(roles, list) and len(roles) >= 3:
            out.append(
                {
                    "roles": tuple(sorted(roles)),
                    "ratio": float(entry.get("ratio", 0.0)),
                    "note": str(entry.get("note", "")),
                }
            )
    return out


def _classify_pair(matrix, a: str, b: str, floor: float) -> dict[str, Any]:
    """Bulk/background classification of a cross-role pair.

    bulk_allowed iff measured AND ratio >= floor. block / below-floor /
    unknown / missing are all NOT bulk-allowed (they queue or are unmeasured).
    """
    pair = matrix.get_pair(a, b)
    if pair is None:
        kind = "unknown" if matrix.is_unknown_pair(a, b) else "missing"
        return {"ratio": None, "verdict": kind, "bulk_allowed": False, "kind": kind}
    below = pair.ratio < floor
    return {
        "ratio": round(pair.ratio, 4),
        "verdict": pair.verdict,
        "bulk_allowed": (not below),
        "kind": "below_floor" if below else "allowed",
    }


def enumerate_n_way(
    numa_config: dict,
    matrix,
    *,
    floor: float,
    matrix_triples: list[dict[str, Any]] | None = None,
    max_size: int | None = None,
) -> dict[str, Any]:
    """Pure enumeration core. Returns the manifest body (no I/O, no timestamp)."""
    roles = sorted(r for r in numa_config if (numa_config[r].get("instances")))
    matrix_triples = matrix_triples or []
    failed_triples = {t["roles"] for t in matrix_triples if t["ratio"] < floor}

    # Lower-order pair evidence over every cross-role pair (deterministic order).
    pair_evidence: dict[str, dict[str, Any]] = {}
    for a, b in itertools.combinations(roles, 2):
        pair_evidence[_pair_key_str(a, b)] = _classify_pair(matrix, a, b, floor)

    hi = max_size or len(roles)
    candidate_sets: list[dict[str, Any]] = []
    excluded_sets: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []

    for size in range(3, hi + 1):
        for combo in itertools.combinations(roles, size):
            combo_sorted = tuple(sorted(combo))
            offending: list[str] = []
            min_ratio = None
            for a, b in itertools.combinations(combo_sorted, 2):
                ev = pair_evidence[_pair_key_str(a, b)]
                if not ev["bulk_allowed"]:
                    if ev["kind"] in ("unknown", "missing"):
                        offending.append(f"{a}+{b} {ev['kind']} pair (not measured)")
                    else:
                        offending.append(
                            f"{a}+{b} ratio {ev['ratio']} < floor {floor} ({ev['verdict']})"
                        )
                else:
                    r = ev["ratio"]
                    min_ratio = r if min_ratio is None else min(min_ratio, r)
            # measured-failed-triple pruning (superset of a known-bad triple)
            bad_sub = [
                "+".join(ft) for ft in failed_triples if set(ft).issubset(set(combo_sorted))
            ]

            if not offending and not bad_sub:
                candidate_sets.append(
                    {
                        "roles": list(combo_sorted),
                        "size": size,
                        "min_pair_ratio": min_ratio,
                        "constituent_pairs": [
                            _pair_key_str(a, b)
                            for a, b in itertools.combinations(combo_sorted, 2)
                        ],
                    }
                )
            else:
                reason = (
                    "contains_failed_triple"
                    if bad_sub and not offending
                    else "contains_below_floor_or_unknown_pair"
                )
                evidence = offending + [f"contains failed triple {s}" for s in bad_sub]
                excluded_sets.append(
                    {"roles": list(combo_sorted), "size": size, "reason": reason, "evidence": evidence}
                )

    # Discrepancy flags: an excluded set that the matrix actually measured >= floor
    # as an informational triple (pairwise-conservative vs N-way reality — the
    # exact phenomenon this closure exists to catch).
    excluded_role_sets = {tuple(e["roles"]) for e in excluded_sets}
    for t in matrix_triples:
        if t["roles"] in excluded_role_sets and t["ratio"] >= floor:
            flags.append(
                {
                    "roles": list(t["roles"]),
                    "issue": "excluded_by_pairwise_floor_but_measured_positive_as_triple",
                    "measured_triple_ratio": t["ratio"],
                    "recommendation": (
                        "J4b should measure this set explicitly; a constituent pair is "
                        "below the background floor while the measured triple is positive. "
                        "Reconsider for foreground-only allow or revisit the pair floor."
                    ),
                }
            )

    return {
        "floor": floor,
        "roles": roles,
        "pair_evidence": pair_evidence,
        "candidate_sets": sorted(candidate_sets, key=lambda x: (x["size"], x["roles"])),
        "excluded_sets": sorted(excluded_sets, key=lambda x: (x["size"], x["roles"])),
        "flags": flags,
        "summary": {
            "n_roles": len(roles),
            "n_candidates": len(candidate_sets),
            "n_excluded": len(excluded_sets),
            "max_size_enumerated": hi,
        },
    }


# Large models that are impractical to run on a 24-core quarter — an 80B/122B
# model on a quarter thrashes (every llama-server carries ~241-289 OMP threads;
# on 24 cores that is ~10x over-subscription → ~0.1 t/s, measured 2026-05-26;
# ingest-FULL on 48 cores is fine at ~16.8 t/s).
# Each NON_QUARTERABLE role co-runs (if at all) only via its instance-0 footprint:
#   - ingest_long_context: instance-0 is a HALF (0-47), so it co-runs as a half
#     against the other half (e.g. ingest 0-47 + vision 48-95 = disjoint, fine).
#   - architect_general: HISTORICAL. Instance-0 used to be the WHOLE machine
#     (0-95) with no half/quarter alternative, which made the role strictly
#     solo. That stopped being true in the 2026-08-01 W1 cutover: the role is
#     now Qwen3.6-27B on MI210 with a GPU HOST LANE, so its instance-0 is 8
#     host threads, its device is ROCm0, and it is one of the most
#     co-residency-friendly roles on the box. Membership here is now inert for
#     it (it has exactly one instance either way) and is kept only so the set
#     keeps meaning "never gets quartered", which is still accurate.
#     The whole-machine solo blocker did not disappear, it was RENAMED:
#     `architect_critic` held the 0-95 interleave=all instance on :8074.
#   - 2026-09-27 ARCHITECT SWAP (operator-decided): renamed BACK. architect_general
#     is again the whole-machine 0-95 instance (Flash-Next, :8074), so its
#     membership here is load-bearing again and accurate; architect_critic is now
#     the MI210 27B GPU host lane (one instance, never quartered).
# This is the per-role "quarterable" property the WP-5 placement policy needs:
# small MoE-light models (gemma4-26B, frontdoor-35B-A3B, vision-30B-A3B) quarter
# well; large models stay full/half; the GPU lanes do not quarter at all.
NON_QUARTERABLE: set[str] = {"ingest_long_context", "architect_general"}


def _role_footprints(
    numa_config: dict,
    role: str,
    *,
    live_only: bool = False,
) -> list[tuple[int, int, frozenset]]:
    """(instance_idx, port, regions) for every instance of `role`.

    For configured/planning enumeration, NON_QUARTERABLE roles expose only the
    full/primary instance (idx 0). Live recert is different: it must measure
    the healthy instances the v7 stack actually launched, including secondary
    ports when the primary port is intentionally down.
    """
    from src.runtime.instance_topology import cpu_list_to_regions
    insts = numa_config.get(role, {}).get("instances", []) or []
    if role in NON_QUARTERABLE and not live_only:
        insts = insts[:1]
    out = []
    for idx, inst in enumerate(insts):
        port = int(inst[1])
        if live_only and not _port_healthy(port):
            continue
        out.append((idx, port, frozenset(cpu_list_to_regions(inst[0]))))
    return out


@dataclass(frozen=True)
class FeasibilityVerdict:
    """Why a role-set can or cannot coexist, with the resource named.

    `reason` is deliberately SPECIFIC. The predecessor emitted a single bare
    `topology_infeasible` for every failure, which conflated "these two want
    the same physical cores" with "these two do not fit on the card" — two
    different resources with two different remedies.
    """

    feasible: bool
    reason: str  # "" when feasible
    evidence: str
    assignment: dict | None
    device_classes: dict[str, str]
    vram: dict | None = None
    reasons: tuple[str, ...] = ()


def _cpu_packing(cpu_roles, numa_config) -> tuple[dict | None, str]:
    """Mutually-disjoint cpuset packing over CPU-device roles only.

    Unchanged physics for CPU decode (2026-05-26 audit): they can co-run only
    on non-overlapping cpusets, so a whole-machine instance can be selected
    only when it is the sole CPU role; otherwise the role falls back to a
    smaller footprint. Returns (assignment, failure_reason).
    """
    foot = {r: _role_footprints(numa_config, r) for r in cpu_roles}
    missing = [r for r in cpu_roles if not foot[r]]
    if missing:
        return (None, "no_placeable_instance")
    # Try smaller footprints first (quarters before full) so the packing leaves
    # room; order roles by fewest options first to prune the search.
    for r in foot:
        foot[r] = sorted(foot[r], key=lambda t: (len(t[2]), t[0]))
    order = sorted(cpu_roles, key=lambda r: len(foot[r]))
    chosen: dict = {}
    used: set = set()

    def rec(i: int) -> bool:
        if i == len(order):
            return True
        r = order[i]
        for idx, port, regs in foot[r]:
            if used & regs:
                continue
            chosen[r] = (idx, port, sorted(regs))
            used.update(regs)
            if rec(i + 1):
                return True
            used.difference_update(regs)
            del chosen[r]
        return False

    if rec(0):
        return (
            {
                r: {"instance_idx": chosen[r][0], "port": chosen[r][1], "regions": chosen[r][2]}
                for r in cpu_roles
            },
            "",
        )
    return (None, "cpu_region_conflict")


def _gpu_placements(gpu_roles, numa_config) -> tuple[dict | None, str]:
    """Placements for GPU roles. A GPU role's host lane is SHAREABLE, so it
    reserves nothing — it is recorded for provenance, not for exclusion."""
    out: dict = {}
    for r in gpu_roles:
        foot = _role_footprints(numa_config, r)
        if not foot:
            return (None, "no_placeable_instance")
        idx, port, regs = sorted(foot, key=lambda t: (len(t[2]), t[0]))[0]
        out[r] = {
            "instance_idx": idx,
            "port": port,
            "regions": sorted(regs),
            "host_lane_shared": True,
        }
    return (out, "")


def assess_feasibility(
    roleset,
    numa_config,
    *,
    priors: dict | None = None,
    server_mode: dict | None = None,
    device_map: dict | None = None,
    vram_capacity_gib: float | None = None,
    vram_headroom_gib: float | None = None,
    allow_host_query: bool = False,
) -> FeasibilityVerdict:
    """Can these roles coexist? Answered per RESOURCE, not per cpuset.

    Artifact 1 + 2 of the device/load-axes rider. The set is partitioned by
    DEVICE and each partition is judged against the resource it actually
    consumes:

      * CPU-device instances claim their CPU regions EXCLUSIVELY. Two of them
        overlapping a region are infeasible together — unchanged behaviour.
      * GPU-device instances do NOT claim CPU regions exclusively. Their host
        lane is shareable light work (tokenise, sample, marshal) while the
        weights are VRAM-resident under `-ngl`. Instead they claim VRAM, and
        the GPU subset must FIT the device with headroom reserved.
      * GPU-vs-CPU region overlap is NOT a conflict. That false exclusion is
        the defect this function exists to remove.

    Device resolution RAISES (`DeviceResolutionError`) on an undeclared role or
    on a disagreement between the compiled priors and `gpu_host_lane`. That is
    intentional: a feasibility answer computed over a corrupted device map is
    worse than no answer.
    """
    roleset = tuple(roleset)
    devices = device_map or resolve_device_classes(roleset, numa_config=numa_config, priors=priors)
    device_classes = {r: devices[r].device_class.value for r in roleset}

    cpu_roles = [r for r in roleset if devices[r].device_class is DeviceClass.CPU]
    gpu_roles = [r for r in roleset if devices[r].device_class is DeviceClass.GPU]

    reasons: list[str] = []
    evidence: list[str] = []

    cpu_assign, cpu_reason = _cpu_packing(cpu_roles, numa_config)
    if cpu_reason:
        reasons.append(cpu_reason)
        if cpu_reason == "cpu_region_conflict":
            evidence.append(
                "no mutually-disjoint cpuset assignment exists over the CPU-device "
                f"subset {sorted(cpu_roles)} (over-subscribed or solo-only full instance)"
            )
        else:
            evidence.append(f"a CPU-device role in {sorted(cpu_roles)} declares no instance")

    gpu_assign, gpu_place_reason = _gpu_placements(gpu_roles, numa_config)
    if gpu_place_reason:
        reasons.append(gpu_place_reason)
        evidence.append(f"a GPU-device role in {sorted(gpu_roles)} declares no instance")

    vram_report: dict | None = None
    if gpu_roles:
        fit = vram_fit(
            gpu_roles,
            priors=priors,
            server_mode=server_mode,
            headroom_gib=vram_headroom_gib,
            capacity_gib=vram_capacity_gib,
            allow_host_query=allow_host_query,
        )
        vram_report = {
            "required_gib": fit.required_gib,
            "budget_gib": fit.budget_gib,
            "capacity_gib": fit.capacity_gib,
            "headroom_gib": fit.headroom_gib,
            "capacity_source": fit.capacity_source,
            "per_role_gib": fit.per_role,
            "slack_gib": fit.slack_gib,
            "gpu_roles": sorted(gpu_roles),
        }
        if fit.undeclared_roles:
            vram_report["undeclared_roles"] = list(fit.undeclared_roles)
        if not fit.ok:
            reasons.append(fit.reason)
            if fit.reason == "vram_capacity_exceeded":
                evidence.append(
                    f"GPU subset {sorted(gpu_roles)} needs {fit.required_gib} GiB VRAM "
                    f"but the budget is {fit.budget_gib} GiB "
                    f"({fit.capacity_gib} GiB capacity - {fit.headroom_gib} GiB reserved, "
                    f"source {fit.capacity_source})"
                )
            else:
                evidence.append(
                    "VRAM footprint is undeclared for "
                    f"{list(fit.undeclared_roles)} — cannot show the set fits, failing closed"
                )

    if reasons:
        return FeasibilityVerdict(
            feasible=False,
            reason=reasons[0],
            evidence="; ".join(evidence),
            assignment=None,
            device_classes=device_classes,
            vram=vram_report,
            reasons=tuple(reasons),
        )

    assignment = {**(cpu_assign or {}), **(gpu_assign or {})}
    return FeasibilityVerdict(
        feasible=True,
        reason="",
        evidence="",
        assignment=assignment,
        device_classes=device_classes,
        vram=vram_report,
        reasons=(),
    )


def feasible_assignment(roleset, numa_config, **kwargs) -> dict | None:
    """Device-aware placement for a role-set, or None if infeasible.

    Thin wrapper over `assess_feasibility` kept for callers that only need the
    placement. Prefer `assess_feasibility` when you need to know WHICH resource
    excluded the set — `None` here cannot distinguish a CPU region conflict
    from a VRAM shortfall.
    """
    return assess_feasibility(roleset, numa_config, **kwargs).assignment


def enumerate_feasible(
    numa_config: dict,
    *,
    max_size: int | None = None,
    priors: dict | None = None,
    server_mode: dict | None = None,
    vram_capacity_gib: float | None = None,
    vram_headroom_gib: float | None = None,
    allow_host_query: bool = False,
) -> dict[str, Any]:
    """Device-aware placement-feasibility enumeration.

    A role-set is a candidate iff (a) its CPU-device subset admits a mutually
    disjoint cpuset assignment and (b) its GPU-device subset fits VRAM with
    headroom reserved. Exclusions name the resource — `cpu_region_conflict`,
    `vram_capacity_exceeded`, `vram_declaration_missing`,
    `no_placeable_instance` — never a bare `topology_infeasible`. Throughput is
    NOT judged here; the assignment is the realizable placement to measure.

    The device map and the VRAM capacity are resolved ONCE, up front, so an
    undeclared role or a disagreeing declaration raises before any verdict is
    produced rather than silently colouring one combination.
    """
    roles = sorted(r for r in numa_config if numa_config[r].get("instances"))
    devices = resolve_device_classes(roles, numa_config=numa_config, priors=priors)

    capacity_source = "n/a (no GPU-device role in topology)"
    if any(d.device_class is DeviceClass.GPU for d in devices.values()):
        if vram_capacity_gib is None:
            vram_capacity_gib, capacity_source = _declared_vram_capacity_gib(
                allow_host_query=allow_host_query
            )
        else:
            capacity_source = "explicit"

    hi = max_size or len(roles)
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for size in range(2, hi + 1):
        for combo in itertools.combinations(roles, size):
            verdict = assess_feasibility(
                combo,
                numa_config,
                priors=priors,
                server_mode=server_mode,
                device_map=devices,
                vram_capacity_gib=vram_capacity_gib,
                vram_headroom_gib=vram_headroom_gib,
            )
            entry: dict[str, Any] = {
                "roles": list(combo),
                "size": size,
                "device_classes": verdict.device_classes,
            }
            if verdict.vram is not None:
                entry["vram"] = verdict.vram
            if verdict.feasible:
                entry["assignment"] = {
                    r: {
                        "port": verdict.assignment[r]["port"],
                        "regions": verdict.assignment[r]["regions"],
                        **(
                            {"host_lane_shared": True}
                            if verdict.assignment[r].get("host_lane_shared")
                            else {}
                        ),
                    }
                    for r in combo
                }
                candidates.append(entry)
            else:
                entry["reason"] = verdict.reason
                entry["reasons"] = list(verdict.reasons)
                entry["evidence"] = verdict.evidence
                excluded.append(entry)

    gpu_roles = sorted(r for r, d in devices.items() if d.device_class is DeviceClass.GPU)
    return {
        "roles": roles,
        "device_model": {
            "gpu_roles": gpu_roles,
            "cpu_roles": sorted(set(roles) - set(gpu_roles)),
            "device_classes": {r: d.device_class.value for r, d in devices.items()},
            "device_sources": {r: d.source for r, d in devices.items()},
            "vram_capacity_gib": vram_capacity_gib,
            "vram_capacity_source": capacity_source,
        },
        "candidate_sets": sorted(candidates, key=lambda x: (x["size"], x["roles"])),
        "excluded_sets": sorted(excluded, key=lambda x: (x["size"], x["roles"])),
        "summary": {
            "n_candidates": len(candidates),
            "n_excluded": len(excluded),
            "max_size": hi,
            "n_excluded_by_reason": {
                reason: sum(1 for e in excluded if e["reason"] == reason)
                for reason in sorted({e["reason"] for e in excluded})
            },
        },
    }


def cmd_enumerate(args: argparse.Namespace) -> int:
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import (
        load_contention_matrix,
        matrix_status,
        topology_fingerprint_for_matrix,
        MatrixStatus,
    )

    _feasibility = getattr(args, "feasibility", False)

    matrix_path = Path(args.matrix) if args.matrix else DEFAULT_OUTPUT
    matrix = None
    try:
        matrix = load_contention_matrix(matrix_path)
    except FileNotFoundError:
        if not _feasibility:
            raise
    live_hash = topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
    status = matrix_status(matrix_path, current_topology_hash=live_hash)
    if status != MatrixStatus.OK:
        # The N-way enumeration CONSUMES measured pair ratios, so a stale matrix
        # would silently prune real candidates — that path still refuses.
        #
        # The feasibility enumeration reads NO measurement at all: it is derived
        # entirely from NUMA_CONFIG plus the declared device (rider §4, artifact
        # 1 — "needs measurement? No"). Refusing it on the freshness of evidence
        # it never opens is a category error, and it is the reason this command
        # could not be run at all through three topology generations of matrix
        # drift. Warn loudly, stamp the status into the manifest, proceed.
        if _feasibility:
            log.warning(
                "matrix status %s (live hash %s) — proceeding anyway: the feasibility "
                "model is derived from topology + declared device and reads no "
                "measured cell. The stale matrix is recorded in the manifest.",
                status.value, live_hash,
            )
        else:
            log.error(
                "matrix status %s (live hash %s) — enumeration requires a fresh, "
                "topology-matching matrix; refusing to emit a manifest against stale evidence.",
                status.value, live_hash,
            )
            return 2

    if _feasibility:
        body = enumerate_feasible(
            NUMA_CONFIG,
            max_size=args.max_size,
            vram_headroom_gib=args.vram_headroom,
            allow_host_query=args.allow_host_query,
        )
    else:
        floor = args.floor if args.floor is not None else matrix.default_floor
        body = enumerate_n_way(
            NUMA_CONFIG,
            matrix,
            floor=floor,
            matrix_triples=_read_matrix_triples(matrix_path),
            max_size=args.max_size,
        )

    # Deterministic content hash (excludes timestamp/run_id) for the J4a
    # "deterministic across two dry runs" closure gate.
    content = json.dumps(body, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    try:
        src_bytes = matrix_path.read_bytes()
    except OSError:
        src_bytes = b""
    git_sha = ""
    try:
        git_sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        pass

    _feas = _feasibility
    manifest = {
        "task_id": "J4a-feasible" if _feas else "J4a",
        "model": "device_aware_disjoint_cpuset_plus_vram" if _feas else "full_instance_pairwise",
        "run_id": args.run_id or f"j4a-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topology_hash": live_hash,
        "content_hash": content_hash,
        "matrix_source": {
            "path": str(matrix_path),
            "sha256": hashlib.sha256(src_bytes).hexdigest(),
            "git_sha": git_sha,
            "topology_hash": matrix.topology_hash if matrix else "",
            "measured_at": matrix.measured_at if matrix else "",
            # Stamped for BOTH models. The feasibility model consumes nothing
            # from the matrix, but a reader must still be able to see what the
            # measured evidence looked like when the manifest was cut.
            "status": status.value,
            "consumed_by_this_model": not _feas,
        },
        **body,
    }

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ("j4a_feasible_manifest.json" if _feas else "j4a_candidate_manifest.json")
        out_path.write_text(json.dumps(manifest, indent=2) + "\n")
        log.info("wrote manifest → %s", out_path)
    else:
        print(json.dumps(manifest, indent=2))

    log.info(
        "enumeration (%s): %d candidates, %d excluded, %d flags (content_hash=%s)",
        manifest["model"],
        manifest["summary"]["n_candidates"],
        manifest["summary"]["n_excluded"],
        len(manifest.get("flags", [])),
        content_hash,
    )
    if _feas:
        dm = manifest.get("device_model", {})
        log.info(
            "  device model: GPU %s | CPU %s | VRAM capacity %s GiB (%s)",
            dm.get("gpu_roles"), dm.get("cpu_roles"),
            dm.get("vram_capacity_gib"), dm.get("vram_capacity_source"),
        )
        log.info("  exclusions by reason: %s", manifest["summary"].get("n_excluded_by_reason"))
    for c in manifest["candidate_sets"]:
        if "assignment" in c:
            devs = c.get("device_classes", {})
            tag = "".join(sorted({d[0].upper() for d in devs.values()})) if devs else "?"
            log.info(
                "  CANDIDATE [%s] %s ports=%s%s",
                tag, c["roles"],
                {r: c["assignment"][r]["port"] for r in c["roles"]},
                f" vram={c['vram']['required_gib']}/{c['vram']['budget_gib']}GiB" if c.get("vram") else "",
            )
        else:
            log.info("  CANDIDATE %s (min_pair_ratio=%s)", c["roles"], c.get("min_pair_ratio"))
    for f in manifest.get("flags", []):
        log.warning("  FLAG %s: %s (triple=%.2f)", f["roles"], f["issue"], f["measured_triple_ratio"])
    return 0


# ── J4b: N-way measurement (runs ALONE on host) ─────────────────────
#
# Measures each non-trivial candidate active set from the J4a manifest:
# solo each role, then all-K concurrent; ratio = parallel_aggregate /
# seq_aggregate over `samples` repetitions; gate on CV <= 0.05.
# Honors feedback_no_concurrent_inference: must run alone.


def _bench_nway(role_ports: list[tuple[str, int]], samples: int = 3, *, safe_sampling: bool = False) -> dict[str, Any]:
    """Solo + all-K-concurrent bench for one active set. Returns ratio/cv/verdict."""
    import statistics
    import functools
    bench = functools.partial(_http_bench, safe_sampling=safe_sampling)
    details: list[dict[str, Any]] = []
    ratios: list[float] = []
    for s in range(samples):
        solo: dict[str, tuple[float, float]] = {}
        for role, port in role_ports:
            solo[role] = bench(port)
        with ThreadPoolExecutor(max_workers=len(role_ports)) as ex:
            futs = {role: ex.submit(bench, port) for role, port in role_ports}
            par = {role: futs[role].result() for role, _ in role_ports}
        # Same sign-biased defect as _bench_pair, and this is the path the
        # OP-21 overlapping re-bench is meant to use, so it must refuse too.
        # An N-way set has MORE legs and therefore more chances for one to
        # time out, and max() discards every one that does.
        failed = _unmeasured_legs(
            {f"solo_{r}": v for r, v in solo.items()}
            | {f"parallel_{r}": v for r, v in par.items()}
        )
        if failed:
            raise UnmeasuredLegError(
                f"{'+'.join(r for r, _ in role_ports)} sample {s}: no ratio "
                f"derivable, these legs produced no measurement: "
                f"{', '.join(failed)}. A timed-out leg is not zero throughput."
            )

        total_tokens = N_PREDICT * len(role_ports)
        seq_time = sum(el for _, el in solo.values()) or 0.001
        par_time = max(el for _, el in par.values()) or 0.001
        seq_agg = total_tokens / seq_time
        par_agg = total_tokens / par_time
        ratio = par_agg / seq_agg if seq_agg > 0 else 0.0
        ratios.append(ratio)
        details.append({
            "sample": s,
            "solo_tps": {r: round(t, 2) for r, (t, _e) in solo.items()},
            "par_tps": {r: round(t, 2) for r, (t, _e) in par.items()},
            "seq_aggregate_tps": round(seq_agg, 2),
            "parallel_aggregate_tps": round(par_agg, 2),
            "ratio": round(ratio, 3),
        })
        log.info("  sample %d: seq=%.1f par=%.1f ratio=%.3f", s, seq_agg, par_agg, ratio)
    mean_ratio = statistics.mean(ratios)
    cv = (statistics.pstdev(ratios) / mean_ratio) if mean_ratio else 0.0
    verdict = "allow" if mean_ratio >= 1.0 else ("borderline" if mean_ratio >= DEFAULT_FLOOR else "block")
    return {
        "ratio": round(mean_ratio, 3),
        "cv": round(cv, 4),
        "samples": samples,
        "seq_aggregate_tps": round(statistics.mean(d["seq_aggregate_tps"] for d in details), 2),
        "parallel_aggregate_tps": round(statistics.mean(d["parallel_aggregate_tps"] for d in details), 2),
        "verdict": verdict,
        "per_sample": details,
    }


def cmd_bench_nway(args: argparse.Namespace) -> int:
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import (
        load_contention_matrix,
        matrix_status,
        topology_fingerprint_for_matrix,
        MatrixStatus,
    )

    matrix_path = DEFAULT_OUTPUT
    matrix = load_contention_matrix(matrix_path)
    live_hash = topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
    status = matrix_status(matrix_path, current_topology_hash=live_hash)
    if status != MatrixStatus.OK:
        log.error("matrix status %s (live hash %s) — refusing N-way bench against stale topology", status.value, live_hash)
        return 2

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("topology_hash") != live_hash:
        log.error("manifest topology_hash %s != live %s — re-run J4a enumerate first",
                  manifest.get("topology_hash"), live_hash)
        return 2

    # Each spec: (roleset_tuple, {role: port}). Feasible-manifest candidates
    # carry an `assignment` (disjoint quarter ports); full-manifest candidates
    # fall back to each role's full/primary port.
    setspecs: list[tuple[tuple, dict]] = []
    for c in manifest.get("candidate_sets", []):
        roles = tuple(c["roles"])
        if len(roles) < args.min_size or (args.max_size and len(roles) > args.max_size):
            continue
        if "assignment" in c:
            ports = {r: c["assignment"][r]["port"] for r in roles}
        else:
            ports = {r: _full_port(NUMA_CONFIG, r) for r in roles}
        setspecs.append((roles, ports))
    if args.include_flagged:
        for fl in manifest.get("flags", []):
            roles = tuple(sorted(fl["roles"]))
            if roles not in [s[0] for s in setspecs]:
                setspecs.append((roles, {r: _full_port(NUMA_CONFIG, r) for r in roles}))
    if not setspecs:
        log.warning("no candidate sets to measure")
        return 0

    results: list[dict[str, Any]] = []
    for roleset, ports in setspecs:
        role_ports = [(r, ports[r]) for r in roleset]
        if any(p is None for _r, p in role_ports):
            log.warning("skipping %s — missing port", roleset)
            continue
        log.info("=== N-way bench %s ports=%s safe_sampling=%s ===", list(roleset), [p for _r, p in role_ports], args.safe_sampling)
        b = _bench_nway(role_ports, samples=args.samples, safe_sampling=args.safe_sampling)
        entry = {
            "roles": sorted(roleset), "size": len(roleset),
            "topology_hash": live_hash, "ports": {r: p for r, p in role_ports},
            "measured_at": datetime.now(timezone.utc).isoformat(), **b,
        }
        results.append(entry)
        log.info("  → %s ratio=%.3f cv=%.4f verdict=%s", list(roleset), b["ratio"], b["cv"], b["verdict"])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SC21: the N-way fragment carries the SAME host-health stamp as the
    # full-matrix emitter (orchestrator 77e5a214), computed here at emit time
    # with the identical probe — never backfilled. `decision_grade` attests
    # HOST STATE only, not the statistical adequacy of the ratios.
    host_health = _host_health_probe(host_meta=_host_metadata())
    if host_health["status"] != HOST_HEALTH_CLEAN:
        log.warning(
            "host health %s — N-way fragment stamped decision_grade=false: %s",
            host_health["status"],
            "; ".join(host_health["decision_grade_blockers"]) or "(no blockers listed)",
        )

    (out_dir / "j4b_nway_results.json").write_text(json.dumps({
        "task_id": "J4b", "topology_hash": live_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest, "n_way": results,
        "host_health": host_health,
    }, indent=2) + "\n")

    # Emit a YAML block ready to append to contention_matrix.yaml. The
    # host-health stamp reuses the matrix emitter's own rendering, so a
    # hand-merged fragment carries the identical provenance fields.
    yb: list[str] = _host_health_stamp_lines(host_health)
    yb.append("n_way:")
    for e in results:
        yb.append(f"  - roles: [{', '.join(repr(r) for r in e['roles'])}]")
        yb.append(f"    size: {e['size']}")
        yb.append(f'    topology_hash: "{e["topology_hash"]}"')
        yb.append(f"    seq_aggregate_tps: {e['seq_aggregate_tps']}")
        yb.append(f"    parallel_aggregate_tps: {e['parallel_aggregate_tps']}")
        yb.append(f"    ratio: {e['ratio']}")
        yb.append(f"    samples: {e['samples']}")
        yb.append(f"    cv: {e['cv']}")
        yb.append(f'    verdict: "{e["verdict"]}"')
        yb.append(f'    measured_at: "{e["measured_at"]}"')
    (out_dir / "j4b_n_way_block.yaml").write_text("\n".join(yb) + "\n")
    log.info("wrote %s (%d sets) + YAML block", out_dir / "j4b_nway_results.json", len(results))
    for e in results:
        cv_ok = "CV_OK" if e["cv"] <= 0.05 else "CV_HIGH"
        log.info("  RESULT %s ratio=%.3f verdict=%s %s", e["roles"], e["ratio"], e["verdict"], cv_ok)
    return 0


# ── J5: within-role instance-pair bench (WP-6, runs ALONE) ──────────
#
# Re-measures same_role contention at instance-pair granularity. For each
# multi-instance quarterable role, benches every DISJOINT instance pair
# (overlapping pairs are a hard topology veto — never co-placed — so they are
# skipped, not measured). NON_QUARTERABLE roles (ingest/architect) are excluded:
# they are not quartered. Reuses _bench_nway keyed by instance label.


def cmd_bench_within_role(args: argparse.Namespace) -> int:
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import (
        load_contention_matrix,
        matrix_status,
        role_topology_fingerprint,
        topology_fingerprint_for_matrix,
        MatrixStatus,
    )

    matrix = load_contention_matrix(DEFAULT_OUTPUT)
    live_hash = topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
    status = matrix_status(DEFAULT_OUTPUT, current_topology_hash=live_hash)
    if status != MatrixStatus.OK and not args.allow_stale_matrix:
        log.error("matrix stale/missing for live topology %s — refusing within-role bench", live_hash)
        return 2
    if status != MatrixStatus.OK:
        log.warning(
            "matrix status %s for live topology %s — continuing because --allow-stale-matrix was set",
            status.value,
            live_hash,
        )

    roles = args.roles or [
        r for r in sorted(NUMA_CONFIG)
        if len(NUMA_CONFIG[r].get("instances", []) or []) >= 2 and r not in NON_QUARTERABLE
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    certs: dict[str, dict[str, Any]] = {}
    for role in roles:
        fps = _role_footprints(NUMA_CONFIG, role, live_only=args.live_only)
        labels = {idx: _shape_label(idx, regs) for idx, _p, regs in fps}
        pairs: list[dict[str, Any]] = []
        for (ia, pa, ra), (ib, pb, rb) in itertools.combinations(fps, 2):
            if ra & rb:
                continue  # overlapping cpusets — hard topology veto, never co-placed
            la, lb = labels[ia], labels[ib]
            log.info("=== within-role %s: %s(%d) + %s(%d) ===", role, la, pa, lb, pb)
            b = _bench_nway([(la, pa), (lb, pb)], samples=args.samples, safe_sampling=args.safe_sampling)
            pairs.append({
                "a": la, "b": lb, "port_a": pa, "port_b": pb,
                "ratio": b["ratio"], "cv": b["cv"], "verdict": b["verdict"],
                "seq_aggregate_tps": b["seq_aggregate_tps"], "parallel_aggregate_tps": b["parallel_aggregate_tps"],
            })
            log.info("  → %s %s+%s ratio=%.3f cv=%.4f %s", role, la, lb, b["ratio"], b["cv"], b["verdict"])
        out[role] = pairs
        live_ports = sorted({port for _idx, port, _regions in fps})
        ratios = [float(p.get("ratio", 0.0) or 0.0) for p in pairs]
        cvs = [float(p.get("cv", 0.0) or 0.0) for p in pairs]
        certs[role] = {
            "role": role,
            "mode": "live_only" if args.live_only else "configured",
            "topology_hash": role_topology_fingerprint(
                NUMA_CONFIG,
                role,
                live_ports=set(live_ports) if args.live_only else None,
            ),
            "live_ports": live_ports,
            "samples": args.samples,
            "min_ratio": min(ratios) if ratios else 0.0,
            "max_cv": max(cvs) if cvs else 0.0,
            "verdict": (
                "allow"
                if len(live_ports) >= 2 and pairs and all(p.get("verdict") == "allow" for p in pairs)
                else "block"
            ),
        }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    artifact_path = out_dir / "j5_within_role_results.json"
    for cert in certs.values():
        cert["measured_at"] = generated_at
        cert["artifact"] = str(artifact_path)
    (out_dir / "j5_within_role_results.json").write_text(json.dumps({
        "task_id": "J5", "topology_hash": live_hash,
        "matrix_status": status.value,
        "generated_at": generated_at,
        "same_role_instance_pairs": out,
        "same_role_certifications": certs,
    }, indent=2) + "\n")

    # Emit a same_role.instance_pairs YAML fragment.
    yb: list[str] = ["# same_role instance_pairs (J5 / WP-6) — disjoint pairs only"]
    for role, pairs in out.items():
        yb.append(f"  - role: \"{role}\"")
        yb.append("    instance_pairs:")
        for p in pairs:
            yb.append(f"      - {{a: {p['a']}, b: {p['b']}, ratio: {p['ratio']}, cv: {p['cv']}, verdict: {p['verdict']}}}")
    (out_dir / "j5_same_role_block.yaml").write_text("\n".join(yb) + "\n")
    cb: list[str] = ["same_role_certifications:"]
    for role in sorted(certs):
        cert = certs[role]
        cb.append(f'  - role: "{role}"')
        cb.append(f'    mode: "{cert["mode"]}"')
        cb.append(f'    measured_at: "{cert["measured_at"]}"')
        cb.append(f'    topology_hash: "{cert["topology_hash"]}"')
        cb.append(f'    live_ports: {cert["live_ports"]}')
        cb.append(f'    verdict: "{cert["verdict"]}"')
        cb.append(f'    samples: {cert["samples"]}')
        cb.append(f'    min_ratio: {round(cert["min_ratio"], 3)}')
        cb.append(f'    max_cv: {round(cert["max_cv"], 3)}')
        cb.append(f'    artifact: "{cert["artifact"]}"')
    (out_dir / "j5_same_role_certifications.yaml").write_text("\n".join(cb) + "\n")
    log.info("wrote %s", out_dir / "j5_within_role_results.json")
    for role, pairs in out.items():
        blocks = [f"{p['a']}+{p['b']}={p['ratio']}" for p in pairs if p["verdict"] == "block"]
        cert = certs.get(role, {})
        log.info(
            "  %s: %d live/configured ports, %d disjoint pairs, %d block %s cert=%s",
            role,
            len(cert.get("live_ports", [])),
            len(pairs),
            len(blocks),
            blocks or "",
            cert.get("verdict"),
        )
    return 0


# ── CLI ──────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> int:
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import topology_fingerprint

    role_filter = set(args.roles) if args.roles else None
    pairs_to_bench = _enumerate_full_pairs(NUMA_CONFIG, role_filter)
    log.info("enumerated %d cross-role pair combinations", len(pairs_to_bench))

    selected: list[tuple[str, str, dict[str, Any], dict[str, Any], str | None]] = []
    missing_live: list[tuple[str, str, str]] = []
    overlap_refused: list[tuple[str, str, str]] = []
    for a, b in pairs_to_bench:
        inst_a, inst_b, reason = _select_live_pair_instances(NUMA_CONFIG, a, b)
        if inst_a is None or inst_b is None:
            missing_live.append((a, b, reason or "missing_live_instance"))
            continue
        if reason:
            # The overlap fallback SUBSTITUTES an overlapping placement for a
            # role pair that has no disjoint live arrangement. Recording it as
            # a measured pair row would let the role-keyed gate read an
            # overlapping-geometry number as the pair's verdict — the exact
            # class of the shipped frontdoor+ingest 1.89 "disjoint" row (APPEND
            # 2026-08-12, inverted marker polarity: the marker fired on the
            # honest fallback while the substituted rows entered unmarked).
            # REFUSE: the pair lands in `unknown_pairs` and the gate's
            # unknown-pair policy applies. To measure a specific geometry,
            # use `bench-nway` with a hand-authored manifest that pins ports.
            overlap_refused.append(
                (
                    a,
                    b,
                    "overlap_substituted"
                    f" ({reason}): no disjoint live placement exists for this "
                    "role pair; an overlapping-geometry number must not enter "
                    "the role-keyed gate as the pair's verdict — re-bench the "
                    "geometry you want with `bench-nway` from a manifest",
                )
            )
            continue
        selected.append((a, b, inst_a, inst_b, reason))

    if args.dry_run:
        log.info("DRY RUN — would measure:")
        for a, b, inst_a, inst_b, reason in selected:
            note = f" ({reason})" if reason else ""
            log.info(
                "  %s.%s:%s + %s.%s:%s%s",
                a,
                inst_a["label"],
                inst_a["port"],
                b,
                inst_b["label"],
                inst_b["port"],
                note,
            )
        for a, b, reason in missing_live:
            log.error("  %s + %s cannot be measured: %s", a, b, reason)
        for a, b, reason in overlap_refused:
            log.error("  %s + %s REFUSED (overlap substitution): %s", a, b, reason)
        return 0

    if missing_live:
        for a, b, reason in missing_live:
            log.error("%s + %s cannot be measured: %s", a, b, reason)
        log.error(
            "refusing to write contention matrix: all production roles must have "
            "a healthy live instance; HTTP failures are not throughput evidence"
        )
        return 2
    if pairs_to_bench and not selected:
        log.error("refusing to write contention matrix: no live pairs selected")
        return 2

    measured: list[PairBench] = []
    blocked_pairs: list[tuple[str, str]] = []  # known catastrophic pairs
    # Overlap-substituted selections are refused, so `selected` can only carry
    # disjoint placements and no geometry marker survives into a pair row.
    skipped: list[tuple[str, str, str]] = list(overlap_refused)

    for a, b, inst_a, inst_b, reason in selected:
        try:
            pb = _bench_pair(
                a,
                int(inst_a["port"]),
                b,
                int(inst_b["port"]),
                instance_a=inst_a,
                instance_b=inst_b,
            )
        except UnmeasuredLegError as exc:
            # Drop the pair rather than bank a sign-biased ratio. Skipping is
            # visible in the emitted `skipped` section; a laundered `allow` is
            # not visible anywhere.
            log.error("  → SKIPPED, not measured: %s", exc)
            skipped.append((a, b, f"unmeasured_leg: {exc}"))
            continue
        measured.append(pb)
        if pb.ratio < CATASTROPHIC_FLOOR:
            blocked_pairs.append(pb.roles)

    zero_pairs = [pb.roles for pb in measured if pb.seq_aggregate_tps <= 0 or pb.parallel_aggregate_tps <= 0]
    if zero_pairs:
        log.error(
            "refusing to write contention matrix: zero-throughput bench result(s) for %s",
            zero_pairs,
        )
        return 2

    # Provenance must name the kernel the bench ACTUALLY ran on. A build-path
    # literal keeps reporting the old tree after a promotion repoints the store.
    binary_path = server_binary("cpu")
    bin_meta = _binary_metadata(binary_path)
    host_meta = _host_metadata()
    measured_roles = {role for pb in measured for role in pb.roles}
    if not measured_roles:
        log.error("refusing to write contention matrix: no pairs were measured")
        return 2
    topo_hash = topology_fingerprint(
        {role: NUMA_CONFIG[role] for role in _matrix_roles(NUMA_CONFIG, measured_roles)}
    )

    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    try:
        preserved = _carry_forward_sections(out_path)
    except ValueError as exc:
        log.error(
            "refusing to overwrite %s: existing matrix could not be parsed for "
            "hand-authored policy sections (%s); a blind rewrite would delete the "
            "runtime N-way / same-role policy the admission gate reads",
            out_path,
            exc,
        )
        return 2

    # Role-scoped runs measure only a subset of the role pairs. `pairs` /
    # `unknown_pairs` are emitter-owned, so without this a scoped run would
    # TRUNCATE the matrix to the measured subset (handoff APPEND 2026-08-12:
    # 3 pairs in, 1 out; against the default output that destroyed 14 of 15
    # measured rows). Carry the unmeasured entries forward verbatim so a
    # scoped re-bench UPDATES the measured rows instead of deleting the rest.
    # Full runs regenerate both sections wholesale.
    preserved_pair_entries: list[tuple[tuple[str, str], list[str]]] | None = None
    preserved_unknown_entries: list[tuple[tuple[str, str], list[str]]] | None = None
    if role_filter:
        measured_pair_keys = {tuple(sorted(pb.roles)) for pb in measured}
        fresh_unknown_keys = {tuple(sorted([a, b])) for a, b, _reason in skipped}
        preserved_pair_entries, preserved_unknown_entries = _preserve_unmeasured_entries(
            out_path, measured_pair_keys, fresh_unknown_keys
        )

    # A `--roles` run re-measures only a subset of the role pairs (unmeasured
    # rows are carried forward verbatim, but their freshness is whatever the
    # previous run stamped), so a role-restricted run is by construction not a
    # decision-grade matrix. It demotes; it never rescues.
    restriction_blockers = (
        [
            "run was role-restricted via --roles "
            f"({', '.join(sorted(role_filter))}); the emitted `pairs` block is "
            "truncated to those roles and is not a full matrix"
        ]
        if role_filter
        else []
    )
    host_health = _host_health_probe(
        host_meta=host_meta, extra_blockers=restriction_blockers
    )
    if host_health["status"] != HOST_HEALTH_CLEAN:
        log.warning(
            "host health %s — matrix will be stamped decision_grade=false: %s",
            host_health["status"],
            "; ".join(host_health["decision_grade_blockers"]) or "(no blockers listed)",
        )

    yaml_str = _emit_yaml(
        measured,
        unknown_pairs=skipped,
        topology_hash=topo_hash,
        binary=bin_meta,
        host=host_meta["hostname"],
        host_health=host_health,
        floor=DEFAULT_FLOOR,
        preserve_sections=preserved,
        preserved_pair_entries=preserved_pair_entries,
        preserved_unknown_entries=preserved_unknown_entries,
    )

    out_path.write_text(yaml_str)
    if preserved:
        log.info(
            "carried forward %d hand-authored policy section(s): %s",
            len(preserved),
            ", ".join(k for k, _ in preserved),
        )
    log.info("wrote %s (%d pairs, topology_hash=%s)", out_path, len(measured), topo_hash[:8])
    log.info("catastrophic pairs (ratio < %.2f): %s", CATASTROPHIC_FLOOR, blocked_pairs)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate the existing matrix against live NUMA_CONFIG."""
    from stack_numa import NUMA_CONFIG
    from src.scheduling.contention import (
        load_contention_matrix,
        matrix_status,
        topology_fingerprint,
        topology_fingerprint_for_matrix,
        MatrixStatus,
    )

    path = Path(args.output) if args.output else DEFAULT_OUTPUT
    matrix = None
    try:
        matrix = load_contention_matrix(path)
    except Exception:
        pass
    current_hash = (
        topology_fingerprint_for_matrix(NUMA_CONFIG, matrix)
        if matrix is not None
        else topology_fingerprint(NUMA_CONFIG)
    )
    status = matrix_status(path, current_topology_hash=current_hash)
    log.info("matrix status: %s (path=%s)", status.value, path)
    log.info("live topology hash: %s", current_hash)
    for line in describe_matrix_host_health(path):
        log.info("%s", line)

    if status == MatrixStatus.OK:
        m = load_contention_matrix(path)
        log.info("matrix loaded OK: %d pairs, %d same_role, %d unknown_pairs",
                 len(m.pairs), len(m.same_role), len(m.unknown_pairs))
        return 0
    if status == MatrixStatus.MISSING:
        log.error("matrix MISSING — run: python scripts/server/contention_matrix.py")
        return 2
    if status == MatrixStatus.STALE:
        try:
            m = load_contention_matrix(path)
            log.warning("matrix STALE — stored hash %s, live %s (or file > 30d)",
                        m.topology_hash, current_hash)
        except Exception as exc:
            log.warning("matrix STALE — additionally failed to load: %s", exc)
        log.warning("re-run: python scripts/server/contention_matrix.py")
        return 2
    if status == MatrixStatus.INVALID:
        log.error("matrix INVALID — file present but unparseable")
        return 3
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="re-bench + write matrix (default)")
    p_run.add_argument("--roles", nargs="+", help="restrict to these roles")
    p_run.add_argument("--dry-run", action="store_true", help="enumerate without benching")
    p_run.add_argument("--output", help="output YAML path (default: orchestration/contention_matrix.yaml)")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="check freshness against live NUMA_CONFIG")
    p_val.add_argument("--output", help="matrix YAML path to validate")
    p_val.set_defaults(func=cmd_validate)

    p_enum = sub.add_parser(
        "enumerate",
        help="J4a: emit N-way candidate/exclusion manifest (no inference)",
    )
    p_enum.add_argument("--matrix", help="input matrix YAML (default: orchestration/contention_matrix.yaml)")
    p_enum.add_argument("--output", help="output dir for j4a_candidate_manifest.json (default: stdout)")
    p_enum.add_argument("--floor", type=float, default=None, help="bulk/background floor (default: matrix default_floor)")
    p_enum.add_argument("--max-size", type=int, default=None, dest="max_size", help="max active-set size to enumerate (default: #roles)")
    p_enum.add_argument("--run-id", default=None, dest="run_id", help="stable run id for the manifest")
    p_enum.add_argument("--feasibility", action="store_true",
                        help="device-aware feasibility model: CPU-device roles must be pairwise "
                             "region-disjoint, GPU-device roles must fit VRAM, GPU-vs-CPU cpuset "
                             "overlap is not a conflict. Derived only — reads no measured cell, "
                             "so it does not require a fresh matrix")
    p_enum.add_argument("--vram-headroom", type=float, default=None, dest="vram_headroom",
                        help="GiB of VRAM to reserve in the capacity check "
                             f"(default {DEFAULT_VRAM_HEADROOM_GIB}, or $ORCHESTRATOR_VRAM_HEADROOM_GIB)")
    p_enum.add_argument("--allow-host-query", action="store_true", dest="allow_host_query",
                        help="permit a READ-ONLY `rocm-smi --showmeminfo vram` query when the "
                             "declared VRAM capacity artifact is unavailable")
    p_enum.set_defaults(func=cmd_enumerate)

    p_nway = sub.add_parser(
        "bench-nway",
        help="J4b: measure N-way candidate sets from a J4a manifest (runs ALONE)",
    )
    p_nway.add_argument("--manifest", required=True, help="J4a candidate manifest JSON")
    p_nway.add_argument("--output", required=True, help="output dir for j4b_nway_results.json")
    p_nway.add_argument("--samples", type=int, default=3, help="repetitions per set (CV gate)")
    p_nway.add_argument("--min-size", type=int, default=2, dest="min_size", help="skip candidate sets smaller than this")
    p_nway.add_argument("--max-size", type=int, default=None, dest="max_size", help="skip candidate sets larger than this")
    p_nway.add_argument("--safe-sampling", action="store_true", dest="safe_sampling",
                        help="temp>0 + repeat_penalty to avoid greedy degeneration (gemma4 crash mitigation)")
    p_nway.add_argument("--include-flagged", action="store_true",
                        help="also measure the manifest's discrepancy-flagged sets")
    p_nway.set_defaults(func=cmd_bench_nway)

    p_wr = sub.add_parser(
        "bench-within-role",
        help="J5 (WP-6): measure same-role disjoint instance pairs (runs ALONE)",
    )
    p_wr.add_argument("--roles", nargs="+", help="roles to sweep (default: multi-instance quarterable roles)")
    p_wr.add_argument("--output", required=True, help="output dir for j5_within_role_results.json")
    p_wr.add_argument("--samples", type=int, default=3)
    p_wr.add_argument("--safe-sampling", action="store_true", dest="safe_sampling",
                      help="temp>0 + repeat_penalty (gemma4 crash mitigation)")
    p_wr.add_argument("--live-only", action="store_true",
                      help="only bench instances whose /health endpoint is live")
    p_wr.add_argument("--allow-stale-matrix", action="store_true",
                      help="allow role-scoped recertification when the global matrix is stale")
    p_wr.set_defaults(func=cmd_bench_within_role)

    args = p.parse_args()
    if args.cmd is None:
        args.cmd = "run"
        args.roles = None
        args.dry_run = False
        args.output = None
        args.func = cmd_run
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
