"""Per-role instance topology — which atomic CPU regions each instance occupies.

Derived from NUMA_CONFIG (scripts/server/stack_numa.py). The single source
of truth for "which (role, instance_idx) pairs can run concurrently" is
the CPU-region overlap: two instances may run concurrently iff their
region sets are disjoint.

The four atomic regions partition the 96 physical cores of the EPYC 9655:
    q0 = cores 0-23   (NUMA node 0, half A)
    q1 = cores 24-47  (NUMA node 0, half B)
    q2 = cores 48-71  (NUMA node 1, half A)
    q3 = cores 72-95  (NUMA node 1, half B)

For each (role, instance_idx) we record the set of quarters it occupies.
The "full" instance for frontdoor (NUMA_NODE0 = 0-47) covers {q0, q1};
the "full" instance for worker_general (0-95) covers {q0, q1, q2, q3}.
Quarter instances cover exactly one region.

This module is import-safe — it performs no I/O at import and has no side
effects. It can be imported from anywhere in the orchestrator process
tree, and from tests, without coupling to running infrastructure. The one
lazy read is the host SMT thread-sibling map (sysfs, lscpu fallback), taken
only when `parse_cpu_list(..., smt_siblings="fold")` meets a CPU above the
primary range 0-95, and cached.

2026-10-04 (REGION-SIBLING-1) — `parse_cpu_list` used to DISCARD logical
CPUs 96-191 (the SMT siblings of 0-95), so `region-lock run --cpu-list
160-183` mapped to no region although it runs on cores 64-87. It now folds
siblings onto their physical core when asked (`smt_siblings="fold"`), and
the `region-lock run` CLI asks by default (`--fold-siblings`). The library
default stays "drop", so the in-process placement model and every other
caller keep their exact prior meaning.

2026-05-22 — added to support cross-process per-region locking
(`src/runtime/cpu_region_lock.py`). See progress entry for design notes.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import Iterable, Literal, Mapping


# The four atomic quarters of the EPYC 9655's 96 physical cores.
ATOMIC_REGIONS = ("q0", "q1", "q2", "q3")

# Physical core ranges for each atomic region. Inclusive on both ends.
REGION_CORE_RANGE: dict[str, tuple[int, int]] = {
    "q0": (0, 23),
    "q1": (24, 47),
    "q2": (48, 71),
    "q3": (72, 95),
}


# Highest primary-thread CPU id covered by REGION_CORE_RANGE (95 on this host).
MAX_PRIMARY_CPU = max(hi for _lo, hi in REGION_CORE_RANGE.values())

SYSFS_CPU_ROOT = Path("/sys/devices/system/cpu")

#: How `parse_cpu_list` treats logical CPUs above MAX_PRIMARY_CPU.
#:   "drop" — DEFAULT, unchanged legacy behaviour: discard CPUs above
#:            MAX_PRIMARY_CPU entirely. Every library caller relies on it: the
#:            in-process placement model (`build_instance_regions` and all its
#:            derivatives — per-call region claims, dispatch, contention,
#:            fleet, eval_tower, contention_matrix) deliberately treats
#:            HT-only instances (the GPU host lane on 184-191) as region-free,
#:            and AutoKernel's `loop/claim.py` (research repo) calls
#:            `cpu_list_to_regions(cpu_list)` positionally.
#:   "fold" — OPT-IN: map each logical CPU onto its physical core's primary
#:            thread (the lowest CPU id in its thread-sibling group), so cpu
#:            160 claims the same core as cpu 64. Used by the `region-lock run`
#:            CLI (`--fold-siblings`, on by default there only).
SmtSiblingMode = Literal["fold", "drop"]
_SMT_MODES = ("fold", "drop")


class CpuTopologyUnavailable(ValueError):
    """A logical CPU above the primary range could not be mapped to its core.

    Never degraded into "assume it maps to itself" or "drop it": either would
    make a region claim cover LESS than the CPUs the caller actually pins —
    under-exclusion, the direction that corrupts measurements.
    """


def _expand_cpu_ranges(text: str, *, source: str) -> set[int]:
    """Strict range expansion for kernel-provided lists (sysfs)."""
    out: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, _, hi_s = part.partition("-")
        if not lo_s.isdigit() or (hi_s and not hi_s.isdigit()):
            raise CpuTopologyUnavailable(f"{source}: {part!r} is not a cpu id or range")
        lo, hi = int(lo_s), int(hi_s or lo_s)
        out.update(range(lo, hi + 1))
    return out


def _sibling_map_from_sysfs(root: Path) -> dict[int, int]:
    """`{logical_cpu: primary_cpu}` from sysfs.

    Preferred source: `topology/thread_siblings_list` (primary = lowest id in
    the group). Fallback for CPUs lacking it: group by
    (`physical_package_id`, `core_id`). NB the raw `core_id` is a hardware id
    (on this Zen5 host cpu64 and cpu160 both read core_id 88), so it is only
    ever used as a GROUPING key, never as the core index the regions use.
    """
    try:
        entries = sorted(root.glob("cpu[0-9]*"))
    except OSError as exc:
        raise CpuTopologyUnavailable(f"cannot list {root}: {exc}") from exc
    mapping: dict[int, int] = {}
    by_core: dict[tuple[str, str], set[int]] = {}
    for entry in entries:
        suffix = entry.name[3:]
        if not suffix.isdigit():
            continue
        cpu = int(suffix)
        topo = entry / "topology"
        try:
            siblings = _expand_cpu_ranges(
                (topo / "thread_siblings_list").read_text(encoding="ascii"),
                source=str(topo / "thread_siblings_list"),
            )
        except OSError:
            siblings = set()
        if siblings:
            mapping[cpu] = min(siblings)
            continue
        try:
            key = (
                (topo / "physical_package_id").read_text(encoding="ascii").strip(),
                (topo / "core_id").read_text(encoding="ascii").strip(),
            )
        except OSError:
            # Offline CPU: no topology dir. Only an error if someone asks for it.
            continue
        by_core.setdefault(key, set()).add(cpu)
    for group in by_core.values():
        primary = min(group)
        for cpu in group:
            mapping.setdefault(cpu, primary)
    if not mapping:
        raise CpuTopologyUnavailable(f"{root}: no cpu exposed thread-sibling topology")
    return mapping


def _sibling_map_from_lscpu() -> dict[int, int]:
    """`{logical_cpu: primary_cpu}` derived from `lscpu -p=CPU,CORE,SOCKET`."""
    try:
        proc = subprocess.run(
            ["lscpu", "-p=CPU,CORE,SOCKET"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CpuTopologyUnavailable(f"lscpu unavailable: {exc}") from exc
    by_core: dict[tuple[str, str], set[int]] = {}
    for line in proc.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) < 3 or not fields[0].isdigit() or not fields[1]:
            continue
        by_core.setdefault((fields[2], fields[1]), set()).add(int(fields[0]))
    mapping = {cpu: min(group) for group in by_core.values() for cpu in group}
    if not mapping:
        raise CpuTopologyUnavailable("lscpu reported no CPU/CORE rows")
    return mapping


def read_sibling_map(sysfs_root: Path | None = None) -> dict[int, int]:
    """`{logical_cpu: primary_cpu}` for THIS host — sysfs first, lscpu second.

    Raises `CpuTopologyUnavailable` when neither source resolves. Nothing here
    hardcodes the +96 sibling offset; it is read from the kernel.
    """
    root = SYSFS_CPU_ROOT if sysfs_root is None else Path(sysfs_root)
    try:
        return _sibling_map_from_sysfs(root)
    except CpuTopologyUnavailable as sysfs_exc:
        try:
            return _sibling_map_from_lscpu()
        except CpuTopologyUnavailable as lscpu_exc:
            raise CpuTopologyUnavailable(f"{sysfs_exc}; fallback: {lscpu_exc}") from lscpu_exc


@functools.lru_cache(maxsize=1)
def _host_sibling_map() -> Mapping[int, int]:
    # lru_cache does not cache exceptions, so a transient failure is retried.
    return read_sibling_map()


def _iter_cpu_ids(cpu_list: str) -> Iterable[int]:
    """Lenient taskset-style expansion: malformed segments are skipped (legacy)."""
    if not cpu_list or not cpu_list.strip():
        return
    for segment in cpu_list.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "-" in segment:
            lo_s, hi_s = segment.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                continue
            yield from range(lo, hi + 1)
        else:
            try:
                yield int(segment)
            except ValueError:
                continue


def parse_cpu_list(
    cpu_list: str,
    *,
    smt_siblings: SmtSiblingMode = "drop",
    sibling_map: Mapping[int, int] | None = None,
) -> set[int]:
    """Parse a taskset-style cpu_list (e.g. '0-23,96-119') into the set of
    physical cores it occupies, named by their primary-thread CPU id (0-95).

    `smt_siblings="drop"` (DEFAULT, legacy): CPUs above MAX_PRIMARY_CPU are
    discarded, so '184-191' -> {}. Every in-process consumer relies on this.

    `smt_siblings="fold"` (opt-in): a CPU above MAX_PRIMARY_CPU is mapped to
    its core's primary thread via the host's thread-sibling topology (read
    lazily, only when such a CPU is present; `sibling_map` injects one for
    tests). '160-183' -> {64..87}. Raises `CpuTopologyUnavailable` if such a
    CPU cannot be resolved.

    Edge cases: empty string returns empty set; single ints work; whitespace
    is tolerated; malformed segments are skipped.
    """
    if smt_siblings not in _SMT_MODES:
        raise ValueError(f"smt_siblings must be one of {_SMT_MODES}, got {smt_siblings!r}")
    result: set[int] = set()
    for c in _iter_cpu_ids(cpu_list):
        if c < 0:
            continue
        if c <= MAX_PRIMARY_CPU:
            result.add(c)
            continue
        if smt_siblings == "drop":
            continue
        if sibling_map is None:
            sibling_map = _host_sibling_map()
        primary = sibling_map.get(c)
        if primary is None:
            raise CpuTopologyUnavailable(
                f"logical cpu {c} is above the primary range 0-{MAX_PRIMARY_CPU} and the "
                f"host thread-sibling topology does not name it; refusing to guess which "
                f"physical core it occupies"
            )
        if not 0 <= primary <= MAX_PRIMARY_CPU:
            raise CpuTopologyUnavailable(
                f"logical cpu {c} folds to primary cpu {primary}, outside 0-{MAX_PRIMARY_CPU}; "
                f"REGION_CORE_RANGE and this host's topology disagree"
            )
        result.add(primary)
    return result


def cores_to_regions(cores: Iterable[int]) -> frozenset[str]:
    """Return the set of atomic regions touched by an iterable of core IDs.

    A region is "touched" if at least one of its cores appears in `cores`.
    """
    touched: set[str] = set()
    for c in cores:
        for region, (lo, hi) in REGION_CORE_RANGE.items():
            if lo <= c <= hi:
                touched.add(region)
                break
    return frozenset(touched)


def cpu_list_to_regions(
    cpu_list: str,
    *,
    smt_siblings: SmtSiblingMode = "drop",
    sibling_map: Mapping[int, int] | None = None,
) -> frozenset[str]:
    """Combine `parse_cpu_list` + `cores_to_regions` — convenience for
    consumers that have a taskset-style cpu_list string in hand.

    Default `smt_siblings="drop"` is the unchanged legacy meaning; the
    `region-lock run` CLI opts into "fold" so a sibling list claims the same
    regions as its physical cores — see `parse_cpu_list`."""
    return cores_to_regions(
        parse_cpu_list(cpu_list, smt_siblings=smt_siblings, sibling_map=sibling_map)
    )


def build_instance_regions(numa_config: dict) -> dict[tuple[str, int], frozenset[str]]:
    """Derive {(role, instance_idx): regions} from a NUMA_CONFIG dict.

    Pure function — caller passes in the config (typically from
    `scripts.server.stack_numa.NUMA_CONFIG`). Tests pass synthetic
    configs.

    Returns one entry per (role, instance_idx). Instances with no CPU
    region overlap with the 0-95 physical cores (e.g. embedders pinned
    to HT-only ranges) get an empty frozenset — treat as non-conflicting
    in the lock layer.

    Primary-only BY CHOICE (library default `smt_siblings="drop"`): the live GPU
    host lane (architect_critic on 184-191, the siblings of 88-95) is
    region-free here, so GPU dispatch never takes a q3 lock. Folding it
    would make every GPU request contend q3 — a placement change, not a
    lock-CLI fix. Lane-vs-q3 co-tenancy is handled by
    `scripts/server/gpu_shadow_lane_lease.py`.
    """
    out: dict[tuple[str, int], frozenset[str]] = {}
    for role, cfg in (numa_config or {}).items():
        instances = cfg.get("instances", [])
        for idx, entry in enumerate(instances):
            if not entry:
                continue
            cpu_list = entry[0] if len(entry) > 0 else ""
            out[(role, idx)] = cpu_list_to_regions(cpu_list)
    return out


def instances_overlap(
    instance_regions: dict[tuple[str, int], frozenset[str]],
    a: tuple[str, int],
    b: tuple[str, int],
) -> bool:
    """True iff (role_a, idx_a) and (role_b, idx_b) share at least one
    atomic region — i.e. they cannot run concurrently without CPU
    contention.

    Useful for tests + diagnostics. The lock layer doesn't call this
    directly; it just acquires the union of region locks for each
    instance and lets fcntl handle the rest.
    """
    return bool(instance_regions.get(a, frozenset()) & instance_regions.get(b, frozenset()))


# ── Canonical shape naming ─────────────────────────────────────────────

# The seven footprints the machine can actually be partitioned into, keyed by
# the CANONICAL REGION SET — never by instance index and never by a thread
# count. Two prior derivations were wrong for reasons worth recording:
#
#   * instance index — `idx 0 -> "full", else f"q{idx-1}"` names every
#     non-primary instance a QUARTER regardless of its real footprint. Since
#     the 2026-07-30 quarter retirement the live lineup is full + two HALVES,
#     so that rule labels a two-region half "q0"/"q1", and labels a role whose
#     primary is itself a half (frontdoor, cpus 0-47) "full".
#   * thread ratio — `threads / full_threads` cannot separate a half from a
#     quarter, because `threads` counts LOGICAL cpus including SMT siblings:
#     a 48-thread quarter (24 cores x 2 SMT) and a 48-thread half both read
#     0.5 against a 96-thread full.
#
# Regions are the machine's own partitioning unit and are derived from the
# instance's own cpu_list, so they are the sound discriminator.
CANONICAL_SHAPES: dict[frozenset[str], str] = {
    frozenset({"q0", "q1", "q2", "q3"}): "full",
    frozenset({"q0", "q1"}): "half0",
    frozenset({"q2", "q3"}): "half1",
    frozenset({"q0"}): "q0",
    frozenset({"q1"}): "q1",
    frozenset({"q2"}): "q2",
    frozenset({"q3"}): "q3",
}


def canonical_shape_for_regions(regions) -> str | None:
    """Name the SHAPE of an instance from the atomic regions it occupies.

    Returns one of "full", "half0", "half1", "q0".."q3" — or ``None`` when the
    footprint is not a canonical shape (an exotic cross-node span such as
    {q0,q2}, a three-region span, or an empty region set as produced by a GPU
    role's HT-only host lane).

    ``None`` is deliberate: a caller that must name the instance anyway is
    obliged to pick a VISIBLY non-committal fallback (e.g. ``inst<idx>``)
    rather than silently asserting a shape the footprint does not have.
    Display-side code that needs a string for every input (the dashboard's
    ``_shape_for_regions``) keeps its own joined-region fallback.
    """
    return CANONICAL_SHAPES.get(frozenset(regions or ()))


# ── Derived module-level table (lazy import to avoid circulars) ─────────

_INSTANCE_REGIONS_CACHE: dict[tuple[str, int], frozenset[str]] | None = None


def get_instance_regions() -> dict[tuple[str, int], frozenset[str]]:
    """Return the live mapping derived from the orchestrator's NUMA_CONFIG.

    Memoized for cheap repeated access. The NUMA_CONFIG is effectively
    immutable at runtime, so caching is safe. Tests should use
    `build_instance_regions` directly with a synthetic config.
    """
    global _INSTANCE_REGIONS_CACHE
    if _INSTANCE_REGIONS_CACHE is None:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            _INSTANCE_REGIONS_CACHE = build_instance_regions(NUMA_CONFIG)
        except Exception:
            # Defensive: if stack_numa import path differs in some
            # deployment, return an empty mapping (lock layer treats
            # missing entries as no-CPU-conflict → no-op blocking).
            _INSTANCE_REGIONS_CACHE = {}
    return _INSTANCE_REGIONS_CACHE


# ── Topology-derived safe-N for autopilot fan-out (WP-1) ───────────────

def compute_max_safe_concurrency(numa_config: dict, role: str) -> int:
    """Return the largest N such that N concurrent requests for `role` can
    be placed on mutually-disjoint cpusets under the dispatcher's current
    full-first policy.

    The count = 1 (instance 0, "full" by NUMA_CONFIG convention) + the
    number of remaining instances whose region set is disjoint from both
    the full instance AND every other already-accepted instance, walked
    in NUMA-disjoint preference order from full.

    Boundary cases:
      * Role not in numa_config → 1.
      * Role has 0 or 1 instance → 1.
      * Role has full instance that covers ALL atomic regions (e.g.
        worker_general with `0-95`) → 1; every quarter overlaps full and
        cannot co-place under the full-first dispatcher.

    Pure function (no I/O). Tests pass synthetic NUMA_CONFIG dicts.

    Until WP-3 (within-role-placement-state-machine.md) lands forward KV
    migration, this is the operational ceiling for autopilot eval fan-out.
    With WP-3, a quarters-only configuration becomes reachable by evicting
    the full session and achievable concurrency rises to the largest
    disjoint subset of all instances (e.g. frontdoor: 3 → 4).
    """
    cfg = (numa_config or {}).get(role) if numa_config else None
    if not cfg:
        return 1
    instances = cfg.get("instances") or []
    if len(instances) <= 1:
        return 1

    regions_for: list[frozenset[str]] = [
        cpu_list_to_regions(entry[0]) if entry else frozenset()
        for entry in instances
    ]
    full_regions = regions_for[0]
    if not full_regions:
        return 1

    # Visit non-full instances in NUMA-disjoint-from-full preference order,
    # matching ConcurrencyAwareBackend._compute_quarter_preference.
    quarter_order = sorted(
        range(1, len(instances)),
        key=lambda i: (bool(full_regions & regions_for[i]), i),
    )

    accepted_union: set[str] = set(full_regions)
    safe_n = 1  # full always counted
    for q_idx in quarter_order:
        q_regions = regions_for[q_idx]
        if not q_regions:
            continue
        if accepted_union & q_regions:
            continue
        accepted_union |= q_regions
        safe_n += 1
    return safe_n


def compute_max_disjoint_live_concurrency(
    numa_config: dict,
    role: str,
    *,
    live_ports: set[int] | None = None,
) -> int:
    """Return the largest disjoint instance set for a live role fleet.

    This is intentionally separate from ``compute_max_safe_concurrency``. The
    legacy helper models the dispatcher's full-first policy, which is correct
    for a mixed/full stack. A quarter-only v7 stack does not have the full
    instance live, so its safe fan-out is the largest disjoint subset among the
    actually-live quarter ports.
    """
    cfg = (numa_config or {}).get(role) if numa_config else None
    if not cfg:
        return 1
    instances = cfg.get("instances") or []
    candidates: list[tuple[int, frozenset[str]]] = []
    for idx, entry in enumerate(instances):
        if not entry or len(entry) < 2:
            continue
        try:
            port = int(entry[1])
        except (TypeError, ValueError):
            continue
        if live_ports is not None and port not in live_ports:
            continue
        regions = cpu_list_to_regions(entry[0])
        if regions:
            candidates.append((idx, regions))
    if not candidates:
        return 1

    accepted_union: set[str] = set()
    safe_n = 0
    for _idx, regions in sorted(candidates, key=lambda item: (len(item[1]), item[0])):
        if accepted_union & regions:
            continue
        accepted_union |= set(regions)
        safe_n += 1
    return max(1, safe_n)


def full_instance_port(role: str, numa_config: dict | None = None) -> int | None:
    """Return the port declared for `role`'s full instance (NUMA_CONFIG idx 0).

    Liveness/alignment helper (DISPATCH-A). The dispatcher labels one endpoint
    per concurrency-aware role as the "full" (all-region) instance and, when it
    routes there, acquires idx-0's *whole-machine* region lock. If the endpoint
    wired into that slot is actually a quarter-sized server (a 24-core quarter
    impersonating the 96-core full), routing to it grabs every region lock and
    serializes the machine — the DISPATCH-A amplifier. This helper lets the
    dispatcher confirm the port it holds as "full" really is the topology's
    idx-0 port before it emits the full candidate.

    Returns None when the role, its instance list, or the port is unknown — the
    caller then preserves legacy behavior (no demotion) rather than guessing.
    Pure when `numa_config` is supplied; otherwise reads the live NUMA_CONFIG.
    """
    cfg: dict | None
    if numa_config is not None:
        cfg = numa_config.get(role) if numa_config else None
    else:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            cfg = NUMA_CONFIG.get(role)
        except Exception:
            cfg = None
    if not cfg:
        return None
    instances = cfg.get("instances") or []
    if not instances:
        return None
    entry = instances[0]
    if not entry or len(entry) < 2:
        return None
    try:
        return int(entry[1])
    except (TypeError, ValueError):
        return None


def topology_idx_for_port(
    role: str, port: int, numa_config: dict | None = None
) -> int | None:
    """Return the NUMA_CONFIG instance index for (`role`, `port`), or None.

    Resolves an endpoint's TRUE topology index (and therefore its atomic region
    set) by PORT rather than by list position (DISPATCH-A2). The dispatcher's
    per-quarter region locks are keyed by topology index; when a `full:`-labelled
    endpoint that is actually a quarter is demoted into the quarters pool, its
    lock MUST be the region matching its physical cores — which is the region of
    the NUMA_CONFIG instance whose port equals the endpoint's port, not the
    region implied by its position in the (full-stripped) quarter list.

    Returns None when the role or port is unknown — the caller then preserves
    legacy positional behavior rather than guessing a (wrong) region. Pure when
    `numa_config` is supplied; otherwise reads the live NUMA_CONFIG.
    """
    if numa_config is not None:
        cfg = numa_config.get(role) if numa_config else None
    else:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            cfg = NUMA_CONFIG.get(role)
        except Exception:
            cfg = None
    if not cfg:
        return None
    instances = cfg.get("instances") or []
    for idx, entry in enumerate(instances):
        if not entry or len(entry) < 2:
            continue
        try:
            if int(entry[1]) == int(port):
                return idx
        except (TypeError, ValueError):
            continue
    return None


def topology_instance_for_port(
    port: int, numa_config: dict | None = None
) -> tuple[str, int] | None:
    """Return the physical ``(topology_role, instance_idx)`` owning a port.

    Logical aliases normally resolve within their role, but launcher-only lanes
    such as ``eval_batch_frontdoor`` deliberately serve a frontdoor alias on a
    separate topology role. Port ownership is unambiguous in NUMA_CONFIG and is
    therefore the authoritative bridge for direct single-endpoint backends.
    """
    if numa_config is None:
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]

            numa_config = NUMA_CONFIG
        except Exception:
            numa_config = {}
    for role, cfg in (numa_config or {}).items():
        for idx, entry in enumerate((cfg or {}).get("instances") or []):
            if not entry or len(entry) < 2:
                continue
            try:
                if int(entry[1]) == int(port):
                    return str(role), idx
            except (TypeError, ValueError):
                continue
    return None


_MAX_SAFE_CONCURRENCY_CACHE: dict[str, int] = {}


def max_safe_concurrency(role: str) -> int:
    """Live counterpart to `compute_max_safe_concurrency` — reads the
    orchestrator's NUMA_CONFIG once and caches per-role results.

    Use this from runtime code (autopilot eval defaults, dispatcher
    placement). Use `compute_max_safe_concurrency` directly in tests
    with synthetic configs.
    """
    if role in _MAX_SAFE_CONCURRENCY_CACHE:
        return _MAX_SAFE_CONCURRENCY_CACHE[role]
    try:
        from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
    except Exception:
        _MAX_SAFE_CONCURRENCY_CACHE[role] = 1
        return 1
    n = compute_max_safe_concurrency(NUMA_CONFIG, role)
    _MAX_SAFE_CONCURRENCY_CACHE[role] = n
    return n
