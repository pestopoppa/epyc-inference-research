#!/usr/bin/env python3
"""Force free pages onto every NUMA node before a CPU model load.

WHY THIS EXISTS (the mechanism, measured 2026-09-02, INF-70/C7)
---------------------------------------------------------------
``numactl --interleave=all`` is a *policy hint*, not a guarantee. The kernel
applies it per-allocation: when the round-robin lands on a node that has no
free pages, the allocator silently falls back to whichever node still has
some, rather than reclaiming on the intended node. Page cache counts as "not
free" for this purpose. On a box that has been serving for hours, three of the
four nodes can be effectively full of cache, and a large model then lands
wherever memory happens to be — with no error, no warning, and no change to
the command line.

Measured instance: the 98 GB Qwen3.8-Flash-Next IQ4_XS uniform artifact loaded
under the canonical recipe landed **57.7 / 10.7 / 8.0 / 17.7 GB** across nodes
0-3 (a 61% share on node 0 against an even 25%), and decode measured 7.65 t/s
against 10.09 t/s with clean placement — **-25%**, entirely from remote-node
traffic on a bandwidth-bound decode.

The fix, per node: allocate and *touch* N GiB of anonymous memory under
``numactl --membind=<node>``. Touching forces the fault, the fault forces the
kernel to reclaim that node's page cache (it cannot fall back — membind is a
hard constraint), and freeing the allocation leaves N GiB of genuinely free
pages on that node. Do this on every node and the subsequent
``--interleave=all`` load stripes evenly because every node can honour its
share.

This is prophylactic only. It does not prove placement — prove that in-window
with ``numa_placement_check.sh <pid>`` against the running model process.

USAGE
-----
    python3 scripts/utils/numa_evict.py                    # 40 GiB on all nodes
    python3 scripts/utils/numa_evict.py --target-gib 60
    python3 scripts/utils/numa_evict.py --nodes 0,2 --target-gib 40
    python3 scripts/utils/numa_evict.py --dry-run          # report only

Exit codes: 0 = every requested node reached the target; 1 = at least one node
is still short (the caller should treat the following measurement as suspect);
2 = usage / environment error.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

MIB = 1024 * 1024

# Refuse absurd requests: a single node on this class of host is ~256 GiB.
MAX_TARGET_GIB = 200


# ---------------------------------------------------------------------------
# Node inventory
# ---------------------------------------------------------------------------

def numactl_hardware() -> str:
    """Return `numactl -H` output, or raise RuntimeError if numactl is absent."""
    if shutil.which("numactl") is None:
        raise RuntimeError("numactl not found on PATH; cannot query or bind NUMA nodes")
    return subprocess.run(
        ["numactl", "-H"], check=True, capture_output=True, text=True
    ).stdout


def parse_free_mb(numactl_h: str) -> dict[int, int]:
    """Parse `node N free: X MB` lines out of `numactl -H` output.

    >>> parse_free_mb("node 0 free: 23001 MB\\nnode 1 free: 2649 MB\\n")
    {0: 23001, 1: 2649}
    """
    free: dict[int, int] = {}
    for line in numactl_h.splitlines():
        f = line.split()
        if len(f) >= 4 and f[0] == "node" and f[2] == "free:":
            try:
                free[int(f[1])] = int(f[3])
            except ValueError:
                continue
    return free


def parse_nodes_arg(spec: str, available: list[int]) -> list[int]:
    """Parse `--nodes` ('all', '0,2', '0-3') against the nodes that exist."""
    spec = spec.strip()
    if not spec or spec == "all":
        return sorted(available)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    unknown = [n for n in out if n not in available]
    if unknown:
        raise ValueError(f"no such NUMA node(s): {unknown} (have {sorted(available)})")
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Where did my own pages land? (/proc/self/numa_maps)
# ---------------------------------------------------------------------------

def parse_numa_maps(text: str, min_pages: int = 1000) -> dict[int, int]:
    """Sum anon pages per node across the big mappings of a numa_maps dump.

    Mappings smaller than `min_pages` are interpreter noise and are skipped, so
    the result describes the allocation we made and nothing else.
    """
    per_node: dict[int, int] = {}
    for line in text.splitlines():
        if "anon=" not in line:
            continue
        fields = line.split()
        nodes: dict[int, int] = {}
        for tok in fields:
            if tok.startswith("N") and "=" in tok and tok[1:2].isdigit():
                key, val = tok.split("=", 1)
                try:
                    nodes[int(key[1:])] = int(val)
                except ValueError:
                    continue
        if sum(nodes.values()) < min_pages:
            continue
        for node, pages in nodes.items():
            per_node[node] = per_node.get(node, 0) + pages
    return per_node


def _fmt_landing(per_node: dict[int, int]) -> str:
    page = os.sysconf("SC_PAGE_SIZE")
    return " ".join(
        f"n{n}={pages * page / MIB:.0f}MiB" for n, pages in sorted(per_node.items())
    )


# ---------------------------------------------------------------------------
# The child: allocate + touch under whatever binding the parent set
# ---------------------------------------------------------------------------

def touch_gib(gib: int) -> tuple[float, str]:
    """Allocate `gib` GiB of anonymous memory and touch every page.

    Fast path: numpy's `arr[:] = 1` is a single C-speed memset over the whole
    buffer, which faults every page in at memory bandwidth (~2 s for 30 GiB).
    Fallback: a Python loop over 4 KiB strides (~30 s for 30 GiB) — correct,
    just slow. Either way the pages are faulted, which is the only thing that
    makes the kernel reclaim on this node.

    Returns (seconds, backend-name). The buffer is freed by the caller
    returning; the child process then exits and the pages go back to the node.
    """
    nbytes = gib << 30
    try:
        import numpy as np  # noqa: PLC0415 - optional fast path
    except ImportError:
        np = None

    if np is not None:
        t0 = time.monotonic()
        buf = np.empty(nbytes, dtype=np.uint8)
        buf[:] = 1
        # Read one byte per page back so the compiler/allocator cannot have
        # elided anything; also proves the mapping is resident.
        assert buf[0] == 1 and buf[-1] == 1
        elapsed = time.monotonic() - t0
        landing = open("/proc/self/numa_maps").read()
        del buf
        return elapsed, "numpy-memset\n" + landing

    import mmap  # noqa: PLC0415

    t0 = time.monotonic()
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    for off in range(0, nbytes, 4096):
        m[off] = 1
    elapsed = time.monotonic() - t0
    landing = open("/proc/self/numa_maps").read()
    m.close()
    return elapsed, "python-loop\n" + landing


def _child_main(gib: int) -> int:
    elapsed, payload = touch_gib(gib)
    backend, _, maps = payload.partition("\n")
    per_node = parse_numa_maps(maps)
    print(
        f"    touched {gib} GiB in {elapsed:.2f}s via {backend}; "
        f"pages landed: {_fmt_landing(per_node) or '(no large mapping seen)'}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------

def evict_node(node: int, gib: int, timeout_s: int) -> bool:
    """Run one membind'd allocate-and-touch child. True if it succeeded."""
    cmd = [
        "numactl",
        f"--membind={node}",
        "--",
        sys.executable,
        os.path.abspath(__file__),
        "--child-gib",
        str(gib),
    ]
    try:
        proc = subprocess.run(cmd, timeout=timeout_s, text=True)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout_s}s on node {node}", file=sys.stderr)
        return False
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Force >= --target-gib free on every NUMA node before a CPU model load.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--target-gib", type=int, default=40,
                    help="free GiB required on each node (default: 40)")
    ap.add_argument("--nodes", default="all",
                    help="'all' (default), or a list like '0,2' / '0-3'")
    ap.add_argument("--dry-run", action="store_true",
                    help="report per-node free and what would be reclaimed; allocate nothing")
    ap.add_argument("--timeout-s", type=int, default=600,
                    help="per-node timeout for the allocate-and-touch child (default: 600)")
    ap.add_argument("--child-gib", type=int, default=None,
                    help=argparse.SUPPRESS)  # internal: the membind'd worker
    args = ap.parse_args(argv)

    if args.child_gib is not None:
        return _child_main(args.child_gib)

    if not 1 <= args.target_gib <= MAX_TARGET_GIB:
        print(f"--target-gib must be in 1..{MAX_TARGET_GIB}", file=sys.stderr)
        return 2

    try:
        hw = numactl_hardware()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    before = parse_free_mb(hw)
    if not before:
        print("ERROR: could not parse any 'node N free:' line from numactl -H", file=sys.stderr)
        return 2
    try:
        nodes = parse_nodes_arg(args.nodes, list(before))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"numa_evict: target {args.target_gib} GiB free per node; nodes {nodes}")
    print("  free BEFORE: " + " ".join(f"n{n}={before[n]}MB" for n in sorted(before)))

    for node in nodes:
        need = args.target_gib - before[node] // 1024 + 1
        if need <= 0:
            print(f"  node {node}: {before[node]} MB free — already at target, skipping")
            continue
        print(f"  node {node}: {before[node]} MB free -> reclaiming {need} GiB under --membind={node}")
        if args.dry_run:
            continue
        if not evict_node(node, need, args.timeout_s):
            print(f"  node {node}: eviction child FAILED", file=sys.stderr)

    after = parse_free_mb(numactl_hardware()) if not args.dry_run else before
    print("  free AFTER:  " + " ".join(f"n{n}={after.get(n, 0)}MB" for n in sorted(after)))

    short = [n for n in nodes if after.get(n, 0) // 1024 < args.target_gib]
    if args.dry_run:
        print("  (dry run — nothing allocated)")
        return 0
    if short:
        print(
            f"  WARNING: nodes {short} are still below {args.target_gib} GiB free. "
            "--interleave=all may still skew; verify with numa_placement_check.sh.",
            file=sys.stderr,
        )
        return 1
    print("  OK: every requested node is at or above target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
