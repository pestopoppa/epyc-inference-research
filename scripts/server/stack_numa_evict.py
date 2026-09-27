"""Pre-launch NUMA eviction and post-launch placement observation (INF-70/C7).

ENABLED 2026-09-03 (operator: "C7 — proceed") for every CPU llama-server role
in `orchestration/stack_topology.yaml` (`numa_pre_evict_gib: 40`); GPU
host-lane roles never declare it and are refused here even if they did.

WHY
---
`numactl --interleave=all` is a per-allocation POLICY HINT, not a guarantee.
When the round robin reaches a node with no free pages the kernel silently
falls back to a node that has some, rather than reclaiming on the intended one.
Page cache counts as "not free", so on a box that has been serving for hours
the striping quietly collapses — same argv, same prefix, no warning, no error,
nothing in the log.

Measured 2026-09-02 (INF-70): a 98 GB model loaded under the canonical prefix
landed 57.7 / 10.7 / 8.0 / 17.7 GB across nodes 0-3 (61% on node 0 against an
even 25%) and decode measured 7.65 t/s against 10.09 t/s with clean placement —
**-25%**, entirely remote-node traffic on a bandwidth-bound decode. The same
audit found a live `sd-server` skewed 8/40/32/19%.

`stack_prewarm` is a different lever and does not cover this: it populates the
PAGE CACHE interleaved so a later mlock finds pages already spread. This module
does the opposite — it FREES pages on every node so a fresh anonymous
allocation can be spread at all.

THE MECHANISM, AND THE FORCING FORM
-----------------------------------
Per node: allocate and TOUCH G GiB under `numactl --membind=<node>`. The hard
binding leaves the kernel no fallback, so it reclaims that node's page cache to
satisfy the fault; freeing the allocation then leaves the pages genuinely free
on that node.

G must be sized the FORCING way (D8x root cause, 2026-09-03): whenever
`free < TARGET`, G = TARGET + 2. The first helper used `TARGET - free`, which
fits inside the already-free pages, reclaims nothing, and leaves free exactly
where it was — that is why runs kept skewing "even after eviction". G strictly
larger than free makes reclaim unavoidable; after release the node holds
>= TARGET + 2 free. A node at or above target gets no allocation. Verify per
node; two passes, because concurrent page-cache growth can steal pass one.

`plan_allocation_gib` / `run_eviction` below are the same rule as
epyc-inference-research/scripts/utils/numa_evict.py. The research helper is
reused when the copy on disk IS the forcing form (probed by marker, not
assumed — the shared clone can lag origin/main); otherwise this module's own
forcing implementation runs. Either way the stack never runs the weak form.

GPU ROLES
---------
`gpu_host_lane: true` roles (architect_critic since the 2026-09-27 ARCHITECT
SWAP — architect_general before —, worker_vision) hold their
weights in VRAM and pin host threads to 184-191 == NPS4 node 3. There is
nothing to interleave and a 42 GiB forced reclaim on node 3 would only evict
the page cache the GPU co-tenant relies on. `pre_evict_gib_for_role` returns
0 for them regardless of what the YAML says, and a test pins that.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

# The research repo's productized helper. Reused when it is the FORCING form.
RESEARCH_EVICT = "/mnt/raid0/llm/epyc-inference-research/scripts/utils/numa_evict.py"
# Presence of this symbol distinguishes the forcing helper from the 2026-09-02
# weak one (which sized `TARGET - free` inline in main()).
RESEARCH_FORCING_MARKER = "def plan_allocation_gib"

# Refuse absurd declarations: one node on this host is ~256 GiB.
MAX_PRE_EVICT_GIB = 200

# Forcing form: a short node is pushed to TARGET + FORCE_HEADROOM_GIB.
FORCE_HEADROOM_GIB = 2

# Verify-and-retry passes; pass two catches concurrent page-cache growth.
DEFAULT_PASSES = 2

# Whole-step budget. 42 GiB/node is seconds with numpy and ~40 s with the
# fallback loop; the cap stops a misconfiguration from stalling a stack start.
EVICT_TIMEOUT_S = 900
PER_NODE_TIMEOUT_S = 600


def pre_evict_gib_for_role(numa_cfg: dict | None) -> int:
    """Resolve `numa_pre_evict_gib` from a role's stack_topology entry.

    Absent, malformed or non-positive ⇒ 0 ⇒ the step does not run. A GPU host
    lane ⇒ 0 unconditionally (see GPU ROLES above). A value is never inferred.
    """
    if not isinstance(numa_cfg, dict):
        return 0
    if numa_cfg.get("gpu_host_lane"):
        return 0
    raw = numa_cfg.get("numa_pre_evict_gib", 0)
    try:
        gib = int(raw)
    except (TypeError, ValueError):
        return 0
    if gib <= 0:
        return 0
    return min(gib, MAX_PRE_EVICT_GIB)


# ---------------------------------------------------------------------------
# Sizing and the pass loop — same rule as the research helper; pure, fakeable
# ---------------------------------------------------------------------------

def plan_allocation_gib(
    free_mb: int, target_gib: int, headroom_gib: int = FORCE_HEADROOM_GIB
) -> int:
    """GiB to allocate-and-touch on a node with `free_mb` free (FORCING form).

    0 when the node is already at or above target; otherwise TARGET + headroom,
    strictly more than is free, so the kernel must reclaim on that node.
    """
    if free_mb // 1024 >= target_gib:
        return 0
    return target_gib + headroom_gib


def run_eviction(
    nodes: list[int],
    target_gib: int,
    *,
    query_free_mb,
    evict,
    passes: int = DEFAULT_PASSES,
    headroom_gib: int = FORCE_HEADROOM_GIB,
    log=lambda _msg: None,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Force >= target on every node, verifying per node, up to `passes` times.

    `query_free_mb()` -> {node: free MB}; `evict(node, gib)` -> bool. Returns
    (nodes still short after the last pass, allocations as (pass, node, gib)).
    """
    allocations: list[tuple[int, int, int]] = []
    free = query_free_mb()
    short = [n for n in nodes if free.get(n, 0) // 1024 < target_gib]
    for pass_no in range(1, max(1, passes) + 1):
        if not short:
            break
        for node in nodes:
            gib = plan_allocation_gib(free.get(node, 0), target_gib, headroom_gib)
            if gib == 0:
                continue
            log(f"pass {pass_no} node {node}: {free.get(node, 0)} MB free -> forcing {gib} GiB")
            allocations.append((pass_no, node, gib))
            if not evict(node, gib):
                log(f"pass {pass_no} node {node}: touch child FAILED")
        free = query_free_mb()
        short = [n for n in nodes if free.get(n, 0) // 1024 < target_gib]
        log(
            f"after pass {pass_no}: "
            + " ".join(f"n{n}={free.get(n, 0)}MB" for n in sorted(free))
            + (f" still short {short}" if short else " all at target")
        )
    return short, allocations


# ---------------------------------------------------------------------------
# Host plumbing (each of these is one subprocess; tests fake them)
# ---------------------------------------------------------------------------

def _node_ids() -> list[int]:
    """NUMA node ids from /sys, or [] when the host is not NUMA."""
    base = "/sys/devices/system/node"
    try:
        entries = os.listdir(base)
    except OSError:
        return []
    ids = []
    for name in entries:
        if name.startswith("node") and name[4:].isdigit():
            ids.append(int(name[4:]))
    return sorted(ids)


def parse_free_mb(numactl_h: str) -> dict[int, int]:
    """Parse `node N free: X MB` lines from `numactl -H` output."""
    free: dict[int, int] = {}
    for line in numactl_h.splitlines():
        f = line.split()
        if len(f) >= 4 and f[0] == "node" and f[2] == "free:":
            try:
                free[int(f[1])] = int(f[3])
            except ValueError:
                continue
    return free


def _query_free_mb() -> dict[int, int]:
    proc = subprocess.run(["numactl", "-H"], capture_output=True, text=True, timeout=30)
    return parse_free_mb(proc.stdout or "")


# The membind'd child: allocate `gib` GiB anonymous, touch every page, exit.
# numpy memset is the fast path (memory-bandwidth speed); the 4 KiB-stride
# Python loop is the dependency-free fallback. Both fault every page, which is
# the only thing that makes the kernel reclaim on this node.
_TOUCH_CHILD = (
    "import sys\n"
    "n = int(sys.argv[1]) << 30\n"
    "try:\n"
    "    import numpy as np\n"
    "    b = np.empty(n, dtype=np.uint8); b[:] = 1; assert b[0] == 1 and b[-1] == 1\n"
    "except ImportError:\n"
    "    import mmap\n"
    "    m = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)\n"
    "    for o in range(0, n, 4096):\n"
    "        m[o] = 1\n"
    "    m.close()\n"
)


def _touch_node(node: int, gib: int, timeout_s: int = PER_NODE_TIMEOUT_S) -> bool:
    cmd = ["numactl", f"--membind={node}", "--", "python3", "-c", _TOUCH_CHILD, str(gib)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def research_helper_is_forcing(path: str = RESEARCH_EVICT) -> bool:
    """True only when the research helper on disk is the FORCING form."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return RESEARCH_FORCING_MARKER in fh.read()
    except OSError:
        return False


def pre_evict_nodes(
    target_gib: int,
    *,
    timeout_s: int = EVICT_TIMEOUT_S,
    passes: int = DEFAULT_PASSES,
) -> tuple[bool, str]:
    """Force >= `target_gib` free on every NUMA node. Returns (ok, message).

    Never raises and never blocks a launch: a failure here degrades placement,
    it does not prevent serving. The caller logs the message either way.
    """
    if target_gib <= 0:
        return True, "disabled (numa_pre_evict_gib=0)"
    numactl = shutil.which("numactl")
    if numactl is None:
        return False, "numactl binary not found on PATH"
    nodes = _node_ids()
    if len(nodes) < 2:
        return True, "single-node host; nothing to evict"

    t0 = time.monotonic()
    if research_helper_is_forcing():
        cmd = ["python3", RESEARCH_EVICT, "--target-gib", str(target_gib), "--passes", str(passes)]
        source = "epyc-inference-research/scripts/utils/numa_evict.py (forcing form)"
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout_s}s ({source})"
        except Exception as exc:  # never block a launch
            return False, f"{type(exc).__name__}: {exc} ({source})"
        elapsed = time.monotonic() - t0
        tail = (proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else ""
        if proc.returncode != 0:
            return False, f"exit {proc.returncode} after {elapsed:.1f}s via {source}: {detail}"
        return True, f"{target_gib} GiB/node in {elapsed:.1f}s via {source}: {detail}"

    source = "inline forcing form" + (
        " (research helper on disk is the weak form)"
        if os.path.isfile(RESEARCH_EVICT) else " (research helper absent)"
    )
    lines: list[str] = []
    try:
        short, allocations = run_eviction(
            nodes,
            target_gib,
            query_free_mb=_query_free_mb,
            evict=lambda node, gib: _touch_node(node, gib, PER_NODE_TIMEOUT_S),
            passes=passes,
            log=lines.append,
        )
    except Exception as exc:  # never block a launch
        return False, f"{type(exc).__name__}: {exc} ({source})"
    elapsed = time.monotonic() - t0
    detail = lines[-1] if lines else "no node was short"
    if short:
        return False, (
            f"nodes {short} still below {target_gib} GiB after {passes} pass(es), "
            f"{elapsed:.1f}s via {source}: {detail}"
        )
    return True, (
        f"{target_gib} GiB/node in {elapsed:.1f}s via {source}: "
        f"{len(allocations)} forced allocation(s); {detail}"
    )


def placement_summary(pid: int) -> str:
    """One-line per-node fold of a live process's resident pages.

    Reads `/proc/<pid>/numa_maps` rather than shelling out to `numastat -p`.
    Same quantity — numastat aggregates exactly these per-node page counts —
    but with no subprocess on the launch path, no dependency on the separate
    `numastat` binary, and no perturbation of the callers' process bookkeeping.
    `scripts/server/affinity_preflight.py` already reads numa_maps for the same
    reason. Read-only observation: it gates nothing and never raises.
    """
    try:
        text = open(f"/proc/{pid}/numa_maps", errors="replace").read()
    except OSError as exc:
        return f"numa_maps unreadable: {exc}"
    return summarize_numa_maps(text)


def summarize_numa_maps(text: str) -> str:
    """Fold numa_maps text into 'n0=… n1=… total=… max=nodeX@YY% (even=ZZ%)'.

    Each line's `N<node>=<pages>` counts pages of THAT mapping's page size, so
    `kernelpagesize_kB` is applied per line — a THP-backed model mapping counts
    2 MiB per page, and ignoring it understates the very mappings that matter.
    """
    per_node: dict[int, float] = {}
    for line in text.splitlines():
        fields = line.split()
        page_kb = 4.0
        for tok in fields:
            if tok.startswith("kernelpagesize_kB="):
                try:
                    page_kb = float(tok.split("=", 1)[1])
                except ValueError:
                    page_kb = 4.0
        for tok in fields:
            if tok.startswith("N") and "=" in tok and tok[1:2].isdigit():
                key, _, val = tok.partition("=")
                try:
                    node = int(key[1:])
                    pages = float(val)
                except ValueError:
                    continue
                per_node[node] = per_node.get(node, 0.0) + pages * page_kb / 1024.0
    if not per_node:
        return "no NUMA-resident mappings"
    total = sum(per_node.values())
    if total <= 0:
        return "no resident pages yet"
    nodes = sorted(per_node)
    parts = " ".join(f"n{n}={per_node[n]:.0f}MB" for n in nodes)
    max_node = max(nodes, key=lambda n: per_node[n])
    max_share = 100.0 * per_node[max_node] / total
    even = 100.0 / len(nodes)
    return (
        f"{parts} total={total:.0f}MB "
        f"max=node{max_node}@{max_share:.1f}% (even={even:.0f}%)"
    )


__all__ = [
    "MAX_PRE_EVICT_GIB",
    "FORCE_HEADROOM_GIB",
    "DEFAULT_PASSES",
    "EVICT_TIMEOUT_S",
    "RESEARCH_EVICT",
    "pre_evict_gib_for_role",
    "plan_allocation_gib",
    "run_eviction",
    "parse_free_mb",
    "research_helper_is_forcing",
    "pre_evict_nodes",
    "placement_summary",
    "summarize_numa_maps",
]
