#!/usr/bin/env python3
"""Per-cell page-cache reset for decision-grade timing cells, with a witness.

WHY THIS EXISTS (BRF-T3 E5 adjudication, 2026-11-02)
----------------------------------------------------
``server_numa_np_sweep.py`` dropped the page cache only between MODEL GROUPS
(an operator step), never per cell. A cell that followed another on the same
model therefore inherited a warm page cache laid out by the previous cell's
placement: in the a7 gemma run the C1b (2x half, policy none) cells'
``pages_by_node`` matched the preceding C1 (interleave=all) cells' layout
exactly — one layout across all 8 cells — so the C1b timings measured the C1
cache placement, not their own. None of the 20 Qwen manifests declared a
per-cell drop either. Record: epyc-root
``docs/design/backlog-refactor-2026-11/t3-evidence/e5-adjudication-20261102.json``.

THE POLICY (manifest field ``cache_reset``)
-------------------------------------------
``per_cell``   reset before EVERY cell (the default and the only policy a
               decision-grade cell may declare, unless it is a warm-cache control)
``per_group``  legacy: the operator drops between model groups; the harness does
               nothing per cell
``none``       no reset at all

A decision-grade cell with ``per_group`` or ``none`` is refused unless it sets
``warm_cache_control: true`` (a cell whose PURPOSE is to measure the warm-cache
case). ``warm_cache_control: true`` with ``per_cell`` is a contradiction and is
also refused. The rule lives here once (`cache_reset_violations`); the schema
owner (e5_cell_manifests.validate_cell_manifest) and the harness both call it.

THE MECHANISMS (reused, not invented)
------------------------------------
``drop_caches``  ``sync`` then ``sudo -n tee /proc/sys/vm/drop_caches`` <- ``3``,
                 the same argv fg4b_a4_cpu_optimized_reanchor.prepare_cold_cache
                 uses. HOST-WIDE: it evicts every co-resident workload's cache,
                 so it is allowed ONLY when this process tree holds the whole
                 host — the quiet-host grant: all four CPU regions q0..q3 AND
                 gpu-quiet EXCLUSIVE, both held by an ancestor of this process
                 (region-lock status, read; acquired by the caller, never here).
``numa_evict``   ``scripts/utils/numa_evict.py --nodes <cell nodes>`` — the
                 forcing per-node eviction (INF-70/C7), scoped to the NUMA nodes
                 the cell's memory can land on. Used whenever the whole host is
                 NOT held. It reclaims page cache only on those nodes and only up
                 to the target, so it is a free-memory floor, NOT a guaranteed
                 cold cache — the witness says so (``cold_cache_guaranteed``).

THE WITNESS (written into each cell's result row as ``cache_reset``)
-------------------------------------------------------------------
timestamp, policy, method and why it was chosen, the scoped nodes, the
whole-host evidence, the exact commands with return codes, and per-node
``free_kib_by_node`` / ``pages_by_node`` (file-backed page-cache pages, from
/sys/devices/system/node/nodeN/meminfo FilePages) BEFORE and AFTER.

Every side-effecting primitive is injectable; the module-level defaults are the
only place a real subprocess is spawned. Tests must inject fakes.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

WITNESS_SCHEMA = "cell-cache-reset-witness/1"

POLICY_PER_CELL = "per_cell"
POLICY_PER_GROUP = "per_group"
POLICY_NONE = "none"
CACHE_RESET_POLICIES = (POLICY_PER_CELL, POLICY_PER_GROUP, POLICY_NONE)

METHOD_DROP_CACHES = "drop_caches"
METHOD_NUMA_EVICT = "numa_evict"

WHOLE_HOST_REGIONS = ("q0", "q1", "q2", "q3")

_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
NUMA_EVICT = _RESEARCH_ROOT / "scripts" / "utils" / "numa_evict.py"
REGION_LOCK = Path("/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock")
SYS_NODE_ROOT = Path("/sys/devices/system/node")

DROP_SYNC_ARGV = ["sync"]
DROP_CACHES_ARGV = ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"]

# numa_evict sizing: never below the canonical pre-evict floor, and at least the
# model's per-node share — loading the model will take that memory on those
# nodes anyway, so forcing it free costs co-residents nothing the load would not.
DEFAULT_EVICT_FLOOR_GIB = 40
EVICT_MAX_TARGET_GIB = 198  # numa_evict MAX_TARGET_GIB (200) minus its 2 GiB headroom
GIB = 1 << 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Policy (pure)
# ---------------------------------------------------------------------------


def resolve_policy(manifest: dict[str, Any]) -> str:
    """The policy a cell runs under. Absent field: per_cell when decision-grade.

    Non-decision-grade cells without the field keep the legacy per_group
    behaviour, so no already-run scout's recipe moves underneath it.
    """
    declared = manifest.get("cache_reset")
    if declared is not None:
        return str(declared)
    return POLICY_PER_CELL if manifest.get("decision_grade_intent") is True else POLICY_PER_GROUP


def cache_reset_violations(manifest: dict[str, Any], *, require_declared: bool) -> list[str]:
    """Policy refusals for one cell manifest ([] = acceptable).

    ``require_declared`` is the schema owner's stricter form: a decision-grade
    cell must STATE its policy. The harness passes False and relies on the
    per_cell default for an absent field.
    """
    errors: list[str] = []
    declared = manifest.get("cache_reset")
    if declared is not None and declared not in CACHE_RESET_POLICIES:
        errors.append(
            f"cache_reset {declared!r} is not one of {list(CACHE_RESET_POLICIES)}"
        )
        return errors
    control = manifest.get("warm_cache_control")
    if control is not None and not isinstance(control, bool):
        errors.append(f"warm_cache_control {control!r} must be a boolean when present")
        return errors
    dgi = manifest.get("decision_grade_intent") is True
    if dgi and require_declared and declared is None:
        errors.append(
            "decision-grade cell must declare cache_reset (per_cell; per_group/none "
            "only with warm_cache_control=true) — a warm cache inherited from the "
            "previous cell is a placement confound (BRF-T3 E5)"
        )
    policy = resolve_policy(manifest)
    if dgi and policy in (POLICY_PER_GROUP, POLICY_NONE) and control is not True:
        errors.append(
            f"decision-grade cell declares cache_reset={policy!r} without "
            "warm_cache_control=true — refused: a decision-grade timing cell needs a "
            "per-cell cache reset plus witness (BRF-T3 E5)"
        )
    if control is True and policy == POLICY_PER_CELL:
        errors.append(
            "warm_cache_control=true contradicts cache_reset='per_cell' (a warm-cache "
            "control must not reset the cache)"
        )
    return errors


# ---------------------------------------------------------------------------
# Topology + snapshots (pure parsers; sysfs root injectable)
# ---------------------------------------------------------------------------


def parse_cpulist(spec: str) -> set[int]:
    """'0-3,8' -> {0,1,2,3,8}. Raises ValueError on garbage."""
    out: set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def read_cpu_to_node(base: Path = SYS_NODE_ROOT) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for node_dir in sorted(base.glob("node[0-9]*")):
        node = int(node_dir.name[4:])
        cpulist = node_dir / "cpulist"
        if cpulist.is_file():
            for cpu in parse_cpulist(cpulist.read_text(encoding="utf-8")):
                mapping[cpu] = node
    return mapping


def cell_memory_nodes(
    instances: Iterable[tuple[str, str]], cpu_to_node: dict[int, int]
) -> list[int]:
    """NUMA nodes a cell's memory can land on.

    ``interleave=all`` stripes over every node; ``none`` allocates first-touch
    local to the pinned CPUs, i.e. the nodes those CPUs belong to.
    """
    all_nodes = sorted(set(cpu_to_node.values()))
    nodes: set[int] = set()
    for cpu_list, policy in instances:
        if policy == "interleave=all":
            return all_nodes
        if policy.startswith("interleave="):
            nodes.update(int(n) for n in policy.split("=", 1)[1].split(","))
            continue
        for cpu in parse_cpulist(cpu_list):
            if cpu not in cpu_to_node:
                raise ValueError(f"cpu {cpu} of cpu_list {cpu_list!r} has no NUMA node")
            nodes.add(cpu_to_node[cpu])
    return sorted(nodes)


def parse_node_meminfo(text: str) -> dict[str, int]:
    """'Node 0 MemFree:  123 kB' lines -> {'MemFree': 123, ...} (kB)."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "Node" and fields[2].endswith(":"):
            try:
                out[fields[2][:-1]] = int(fields[3])
            except ValueError:
                continue
    return out


def node_memory_snapshot(base: Path = SYS_NODE_ROOT, page_kib: int | None = None) -> dict[str, Any]:
    """Per-node free memory and file-backed page-cache pages, now."""
    if page_kib is None:
        page_kib = os.sysconf("SC_PAGE_SIZE") // 1024
    free: dict[str, int] = {}
    pages: dict[str, int] = {}
    for node_dir in sorted(base.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        meminfo = node_dir / "meminfo"
        if not meminfo.is_file():
            continue
        info = parse_node_meminfo(meminfo.read_text(encoding="utf-8"))
        key = f"N{node_dir.name[4:]}"
        free[key] = info.get("MemFree", 0)
        pages[key] = info.get("FilePages", 0) // max(1, page_kib)
    return {
        "captured_at": utc_now(),
        "free_kib_by_node": free,
        "pages_by_node": pages,
        "pages_by_node_source": "sysfs nodeN/meminfo FilePages (page-cache pages per node)",
    }


# ---------------------------------------------------------------------------
# Whole-host evidence (pure check over region-lock status)
# ---------------------------------------------------------------------------


def ancestor_pids(pid: int | None = None, proc: Path = Path("/proc")) -> set[int]:
    current = os.getpid() if pid is None else pid
    result: set[int] = set()
    while current > 0 and current not in result:
        result.add(current)
        try:
            # comm may contain spaces; ppid is the 2nd field after the closing ')'.
            stat = (proc / str(current) / "stat").read_text(encoding="utf-8")
            current = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return result


def whole_host_evidence(status: dict[str, Any] | None, ancestors: set[int]) -> dict[str, Any]:
    """Does THIS process tree hold the whole host? (q0..q3 + gpu-quiet exclusive).

    ``status`` is ``region-lock status --json --gpu-quiet`` output. Holding is
    proved by holder PID membership in our ancestor chain, never by the absence
    of other processes (observing is not acquiring).
    """
    reasons: list[str] = []
    if not isinstance(status, dict):
        return {"held": False, "reasons": ["region-lock status unavailable"]}
    rows = status.get("regions")
    by_region = {
        str(row.get("region")): row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict)
    }
    for region in WHOLE_HOST_REGIONS:
        row = by_region.get(region)
        if not row or row.get("global_held") is not True:
            reasons.append(f"{region} not held")
            continue
        holders = row.get("holders") if isinstance(row.get("holders"), list) else []
        if not any(_pid(h) in ancestors for h in holders if isinstance(h, dict)):
            reasons.append(f"{region} held by a process outside this tree")
    quiet = status.get("gpu_quiet")
    if not isinstance(quiet, dict) or quiet.get("held") is not True:
        reasons.append("gpu-quiet not held")
    elif quiet.get("mode") != "exclusive":
        reasons.append(f"gpu-quiet held in mode {quiet.get('mode')!r}, not exclusive")
    else:
        holders = quiet.get("holders") if isinstance(quiet.get("holders"), list) else []
        if not any(
            _pid(h) in ancestors and h.get("mode") == "exclusive"
            for h in holders
            if isinstance(h, dict)
        ):
            reasons.append("gpu-quiet exclusive held by a process outside this tree")
    return {"held": not reasons, "reasons": reasons}


def _pid(holder: dict[str, Any]) -> int | None:
    try:
        return int(holder.get("pid"))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Real primitives (the only place a subprocess is spawned)
# ---------------------------------------------------------------------------


def read_region_lock_status(region_lock: Path = REGION_LOCK) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [str(region_lock), "status", "--json", "--gpu-quiet"],
            check=False, text=True, capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            return None
        payload = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cmd_record(argv: list[str], proc: subprocess.CompletedProcess[str], stdin: str | None = None) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "stdin": stdin,
        "returncode": proc.returncode,
        "output_tail": ((proc.stdout or "") + (proc.stderr or ""))[-2000:],
    }


def run_drop_caches() -> list[dict[str, Any]]:
    sync = subprocess.run(DROP_SYNC_ARGV, check=False, text=True, capture_output=True, timeout=120)
    records = [_cmd_record(DROP_SYNC_ARGV, sync)]
    if sync.returncode != 0:
        return records
    drop = subprocess.run(
        DROP_CACHES_ARGV, check=False, text=True, input="3\n", capture_output=True, timeout=30,
    )
    records.append(_cmd_record(DROP_CACHES_ARGV, drop, stdin="3"))
    return records


def numa_evict_argv(nodes: list[int], target_gib: int) -> list[str]:
    return [
        sys.executable, str(NUMA_EVICT),
        "--nodes", ",".join(str(n) for n in nodes),
        "--target-gib", str(target_gib),
    ]


def run_numa_evict(nodes: list[int], target_gib: int) -> list[dict[str, Any]]:
    argv = numa_evict_argv(nodes, target_gib)
    proc = subprocess.run(argv, check=False, text=True, capture_output=True, timeout=1800)
    return [_cmd_record(argv, proc)]


def evict_target_gib(model_bytes: int | None, n_nodes: int, floor_gib: int = DEFAULT_EVICT_FLOOR_GIB) -> int:
    share = 0
    if model_bytes and n_nodes > 0:
        share = math.ceil(model_bytes / n_nodes / GIB)
    return max(1, min(EVICT_MAX_TARGET_GIB, max(floor_gib, share)))


# ---------------------------------------------------------------------------
# The reset (orchestration; every primitive injectable)
# ---------------------------------------------------------------------------


def perform_cell_cache_reset(
    *,
    cell_id: str,
    manifest: dict[str, Any],
    instances: Iterable[tuple[str, str]],
    model_bytes: int | None = None,
    status_reader: Callable[[], dict[str, Any] | None] = read_region_lock_status,
    ancestors: Callable[[], set[int]] = ancestor_pids,
    cpu_to_node: Callable[[], dict[int, int]] = read_cpu_to_node,
    snapshot: Callable[[], dict[str, Any]] = node_memory_snapshot,
    drop_fn: Callable[[], list[dict[str, Any]]] = run_drop_caches,
    evict_fn: Callable[[list[int], int], list[dict[str, Any]]] = run_numa_evict,
    clock: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    """Reset the page cache for one cell per its policy; return the witness.

    Never raises for a failed reset: ``ok`` is False and ``error`` says why, and
    the caller decides (the sweep fails the cell — a decision-grade timing must
    not run on an unwitnessed cache).
    """
    policy = resolve_policy(manifest)
    witness: dict[str, Any] = {
        "schema": WITNESS_SCHEMA,
        "cell_id": cell_id,
        "timestamp": clock(),
        "policy": policy,
        "policy_declared": manifest.get("cache_reset") is not None,
        "warm_cache_control": manifest.get("warm_cache_control") is True,
        "performed": False,
        "method": None,
        "method_reason": None,
        "scope_nodes": None,
        "whole_host": None,
        "commands": [],
        "cold_cache_guaranteed": False,
        "before": None,
        "after": None,
        "finished_at": None,
        "ok": True,
        "error": None,
    }
    if policy != POLICY_PER_CELL:
        witness["method_reason"] = (
            "policy per_group: the harness does not reset per cell (operator drops "
            "between model groups)" if policy == POLICY_PER_GROUP
            else "policy none: no cache reset"
        )
        witness["before"] = snapshot()
        witness["finished_at"] = clock()
        return witness

    try:
        witness["before"] = snapshot()
        evidence = whole_host_evidence(status_reader(), ancestors())
        witness["whole_host"] = evidence
        if evidence["held"]:
            witness["method"] = METHOD_DROP_CACHES
            witness["method_reason"] = (
                "whole host held by this process tree (q0..q3 + gpu-quiet exclusive)"
            )
            witness["scope_nodes"] = "host"
            witness["commands"] = drop_fn()
            witness["cold_cache_guaranteed"] = True
        else:
            nodes = cell_memory_nodes(instances, cpu_to_node())
            if not nodes:
                raise ValueError("cell resolves to no NUMA nodes")
            target = evict_target_gib(model_bytes, len(nodes))
            witness["method"] = METHOD_NUMA_EVICT
            witness["method_reason"] = (
                "whole host NOT held (" + "; ".join(evidence["reasons"]) + "): host-wide "
                "drop_caches would evict co-resident workloads, so the forcing per-node "
                "eviction is scoped to this cell's NUMA nodes"
            )
            witness["scope_nodes"] = nodes
            witness["evict_target_gib"] = target
            witness["commands"] = evict_fn(nodes, target)
        failed = [c for c in witness["commands"] if c.get("returncode") != 0]
        if not witness["commands"] or failed:
            witness["ok"] = False
            witness["error"] = (
                f"{witness['method']} failed: "
                + "; ".join(f"{c.get('argv')} rc={c.get('returncode')}" for c in failed)
                if failed else f"{witness['method']} ran no command"
            )
        else:
            witness["performed"] = True
    except Exception as exc:  # noqa: BLE001 - the witness records it; the caller gates
        witness["ok"] = False
        witness["error"] = f"{type(exc).__name__}: {exc}"
    try:
        witness["after"] = snapshot()
    except Exception as exc:  # noqa: BLE001
        witness["ok"] = False
        witness["error"] = (witness["error"] or "") + f"; after-snapshot failed: {exc}"
    witness["finished_at"] = clock()
    return witness
