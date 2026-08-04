#!/usr/bin/env python3
"""cpu_region_claim.py — ACQUISITION of a CPU region claim (invariant 9, P-AK-SEARCH-1).

WHY THIS MODULE EXISTS
----------------------
`resource/preflight.read_region_claims` READS the CPU region-lock namespace, and
`resource/device_claim.py` ACQUIRES the GPU device. Nothing acquired a CPU
region. Invariant 9 — *"Resources are acquired, not observed. Every CPU/GPU
benchmark or profiler run holds the appropriate region/device claim. Idle
sensing is never a claim"* — and `P-AK-SEARCH-1` denial 8 — *"no inference run
OUTSIDE A HELD CLAIM"* — were therefore **unsatisfiable rather than safe** on the
CPU side: an agent that wanted to benchmark legitimately had no way to obtain the
claim it was required to hold, and the only remaining options were to sense idle
(the `gpu_idle()` mistake, §10.4) or to run unclaimed. This module closes that.

It is the CPU counterpart of `device_claim.py` and follows its design, not a
second idiom: a lock file whose exclusion fact is `flock(LOCK_EX)`, liveness from
PID **plus** `/proc` start-time so a recycled PID cannot inherit a claim, a
three-valued liveness verdict in which `unknown` never authorises a takeover,
reclamation of a genuinely dead holder's lock after a grace period, and a
journal record written BEFORE any reclamation takes effect. Where it diverges,
the divergence is named and argued in *DELIBERATE DEVIATIONS* below.

INTEROPERATION, NOT A SECOND NAMESPACE
--------------------------------------
The orchestrator owns the CPU region lock (`epyc-orchestrator/src/runtime/
cpu_region_lock.py`). Two lock systems over one machine are worse than none,
because each is correct in isolation and neither excludes the other. So this
module writes **into the orchestrator's namespace, with the orchestrator's file
names, primitive and payload contract**:

  * path      `{lock_root}/cpu_region.{role}.{region}.lock`
  * exclusion `fcntl.flock(LOCK_EX)` — THE FLOCK IS THE FACT
  * global    `{lock_root}/cpu_region.GLOBAL.{region}.lock`, the role-agnostic
              cross-role mutex layer, held with NO payload (the orchestrator
              writes none there, and its `sweep_stale_region_lock_payloads`
              deliberately skips GLOBAL files — a payload we left there after a
              crash would be debris nothing in the fleet ever clears)
  * payload   `{schema_version: 1, pid, role, region, regions, instance_idx,
              request_tag, started_at}` plus AutoKernel-only extension keys, so
              `cpu_region_lock.read_region_lock_payload`,
              `active_region_holders`, `held_regions_by_role` and
              `preflight.read_region_claims` all keep working unchanged and a
              human running `region-lock` sees who holds what.
  * regions   `q0=0-23, q1=24-47, q2=48-71, q3=72-95` — the four atomic quarters
              of the EPYC 9655's 96 physical cores, mirrored from
              `src/runtime/instance_topology.REGION_CORE_RANGE`.

The region table and the payload shape are MIRRORED rather than imported, for the
reason `preflight.py` already records: importing `src.runtime.*` would make this
module depend on a second repository's import graph and on `sys.path` ordering
across three repos. Mirroring costs drift, and drift is contained structurally —
`test_cpu_region_claim.py` loads the orchestrator's `instance_topology.py` by
path when it is present and asserts the two tables are equal, so a change there
fails a test here instead of silently splitting the machine in two.

OVERLAP IS THE POINT
--------------------
A claim system that only compares equality is decorative. A claim on cores 0-95
and a claim on 48-143 are not equal and they conflict. Overlap is resolved in
two stages and both are structural:

  1. **cores → physical cores.** Logical CPUs 96-191 are the SMT siblings of
     physical cores 0-95 (`/sys/.../cpu184/topology/thread_siblings_list` reads
     `88,184`), so 184-191 and 88-95 are the SAME EIGHT PHYSICAL CORES. The
     recorded scar (`feedback_mi210_host_threads_smt_siblings`) is that GPU host
     threads belong on 184-191 and NOT on 88-95 — that is a choice about which
     *thread* to pin, not a claim that the cores are free. This module folds
     siblings onto their physical core, so a GPU host-thread claim on 184-191 and
     a canonical CPU baseline on 0-95 correctly CONFLICT.
  2. **physical cores → atomic regions**, and the lock set is the union of the
     regions touched. Two claims conflict iff their region sets intersect;
     0-47 and 48-95 are disjoint and may be held at the same time, which is the
     whole reason the orchestrator's namespace is per-region rather than global.

WHAT THIS MODULE GUARANTEES
---------------------------
1. **Exclusion is the kernel's.** `flock(LOCK_EX|LOCK_NB)` on never-unlinked lock
   files. Two claims cannot both hold an overlapping region, including two claims
   in the same process (flock conflicts between open file descriptions), so a
   self-conflict surfaces as a clean timeout rather than a double-booked machine.
2. **Acquisition is all-or-nothing and deadlock-free by construction.** Locks are
   taken in ONE total order — every `GLOBAL` region lock in sorted region order,
   then every per-role lock in sorted `(role, region)` order — which is a
   linear extension of the order `cpu_region_lock.cpu_region_lock` itself uses
   (GLOBAL-all-then-role-all, each region-sorted). A partial acquisition releases
   what it holds before propagating, so a timeout never strands a region.
3. **Liveness is PID + `/proc` start time + boot id, never a heartbeat.**
   Delegated to `device_claim.assess_holder_liveness` — one implementation, not
   two. A heartbeat written once is a birth certificate
   (INC-20260727-stale-heartbeat), and there is none here.
4. **A live holder is never stolen from.** Reclamation requires positive proof of
   death. `unknown` is a third outcome that refuses, journals a defect, and
   raises `CpuRegionClaimInconsistent`.
5. **A reclamation is journaled before it happens**, so a takeover cannot occur
   without a record of it (invariant 7).
6. **Every claim emits a receipt** whose `claim_id` is the string a caller puts
   in `evaluation_event.resource_claim_receipt`, and `check_footprint_covered`
   answers `P-AK-SEARCH-1` precondition 1 — *"A CPU region claim covering the
   exact footprint measured"* — against the argv's own `taskset -c` list.
7. **Nothing here signals, kills, or unlinks another process's lock.** There is
   no `kill`, no `signal`, no `subprocess`, and no name-pattern process lookup in
   this module (INC-20260731-broad-process-pattern-kills). Reading
   `/proc/<pid>/stat` for a PID a lock file names is a targeted read of a
   recorded id; it is the opposite of `pgrep`.

DELIBERATE DEVIATIONS FROM `device_claim.py` AND FROM `cpu_region_lock.py`
--------------------------------------------------------------------------
* **`device_claim` REFUSES every payload it cannot age; this module RECLAIMS
  orchestrator-shaped debris.** A payload written by `cpu_region_lock` under a
  FREE flock cannot belong to a live holder: it writes the payload only after
  taking the flock and clears it before closing, so payload-without-flock means
  the writer was killed between the two. That is the namespace owner's own
  stated semantics (*"The flock is the sole liveness and occupancy fact"*,
  `sweep_stale_region_lock_payloads`), and honouring it is what interoperation
  means. It is still reclaimed only after the grace period and still journaled.
  A payload we cannot **date** (no usable `started_at`/`acquired_at`) is refused,
  not reclaimed, and the refusal names `region-lock sweep` as the sanctioned
  cleanup.
* **Roles and regions are VALIDATED, not sanitized.** `cpu_region_lock` rewrites
  `/` to `_` in a role name; two distinct roles would then map onto one lock file
  (over-exclusion) or one role be addressable by two names that do not exclude
  each other. A bad role raises. Refusing a name the orchestrator would have
  accepted is safe; accepting one it maps elsewhere is not.
* **`timeout_s <= 0` means ONE attempt, not "block forever".** `cpu_region_lock`
  reads a non-positive timeout as no timeout; on a shared host that turns a typo
  into an unbounded hang. `None` blocks forever and has to be written out. This
  matches `device_claim` and diverges from `cpu_region_lock` in the visible
  direction.
* **SMT siblings fold onto their physical core; `cpu_region_lock.parse_cpu_list`
  DROPS logical CPUs 96-191 entirely.** For the same cpu list this module
  therefore takes a SUPERSET of the orchestrator's locks (a claim on 184-191
  takes q3; the orchestrator would take nothing). The divergence is
  over-exclusion, never under-exclusion, and over-exclusion costs concurrency
  while under-exclusion costs the measurement.
* **No revocation protocol.** `device_claim` has one because it invented its own
  namespace and could define it. The CPU namespace is the orchestrator's and has
  no revocation contract; adding `cpu_region.*.revoke.json` unilaterally would
  create exactly the second idiom this module exists to avoid. `max_hold_s` is
  recorded as an advisory `expires_at` so an over-long hold is visible to a
  human, and `check_claim_expiry` reports it. Nothing preempts a holder.

SCOPE
-----
No inference, no benchmark, no build, no process start/stop/signal, no
production tree. This module makes it possible to hold a claim; running anything
under one is the caller's job.

Requires Linux `/proc` (for liveness) — on a host without it, acquisition raises
rather than falling back to PID-only liveness, because PID-only liveness IS the
impersonation hole the design closes.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .. import storage as _storage
from ..resource import preflight as _pf
from ..resource.device_claim import (
    DEAD,
    KIND_ACQUIRED,
    KIND_DEFECT,
    KIND_RECLAIMED,
    KIND_RELEASED,
    LIVE,
    UNKNOWN,
    ClaimJournal,
    Liveness,
    assess_holder_liveness,
    current_holder_identity,
)
from ..schemas import COULD_NOT_CHECK, FAIL, PASS, Check, canonical_json

__all__ = [
    "CPU_REGION_CLAIM_SCHEMA",
    "RECEIPT_SCHEMA",
    "ORCHESTRATOR_PAYLOAD_SCHEMA_VERSION",
    "ATOMIC_REGIONS",
    "REGION_CORE_RANGE",
    "GLOBAL_MUTEX_ROLE",
    "MAX_PHYSICAL_CORE",
    "SYSFS_CPU_ROOT",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_POLL_S",
    "DEFAULT_STALE_GRACE_S",
    "DEFECT_LIVE_HOLDER_FREE_LOCK",
    "DEFECT_UNVERIFIABLE_CLAIM",
    "LIVENESS_STATES",
    "CpuRegionClaimError",
    "CpuRegionClaimTimeout",
    "CpuRegionClaimInconsistent",
    "CpuRegionClaimUnreadable",
    "CpuTopologyUnavailable",
    "LockRootDenied",
    "RegionClaimJournal",
    "RegionClaimReceipt",
    "RegionPlan",
    "CpuRegionClaim",
    "parse_cpu_list",
    "render_cpu_list",
    "read_sibling_map",
    "verify_host_topology",
    "physical_cores",
    "cores_to_regions",
    "cpu_list_to_regions",
    "regions_overlap",
    "cpu_lists_overlap",
    "region_lock_path",
    "global_region_lock_path",
    "default_region_lock_dir",
    "plan_region_claim",
    "acquire_cpu_region_claim",
    "cpu_region_claim",
    "canonical_cpu_baseline_cpu_list",
    "gpu_host_cpu_list",
    "probe_region_conflicts",
    "inspect_region_claims",
    "roles_present",
    "check_region_claim_held",
    "check_footprint_covered",
    "check_precondition_1",
    "check_receipt_self_consistent",
    "expected_lock_paths",
    "check_dispatch_exclusion",
    "check_claim_expiry",
]

# =============================================================================
# Identity, namespace and defaults
# =============================================================================

CPU_REGION_CLAIM_SCHEMA = "epyc.autokernel.cpu_region_claim.v1"
RECEIPT_SCHEMA = "epyc.autokernel.cpu_region_claim_receipt.v1"

#: The orchestrator's payload version. Written verbatim so
#: `preflight.read_region_claims` does not annotate our claims with "unknown
#: payload schema_version" — a note that degrades attribution in an evidence
#: record for no reason. Our own version lives in `autokernel_schema`.
ORCHESTRATOR_PAYLOAD_SCHEMA_VERSION = 1

#: MIRRORED from `epyc-orchestrator/src/runtime/instance_topology.py`. The four
#: atomic quarters partition the 96 physical cores of the EPYC 9655.
ATOMIC_REGIONS = ("q0", "q1", "q2", "q3")
REGION_CORE_RANGE: dict = {
    "q0": (0, 23),
    "q1": (24, 47),
    "q2": (48, 71),
    "q3": (72, 95),
}
ORCHESTRATOR_TOPOLOGY_SOURCE = (
    "epyc-orchestrator/src/runtime/instance_topology.py:REGION_CORE_RANGE"
)

#: The reserved pseudo-role of the role-AGNOSTIC cross-role mutex layer.
GLOBAL_MUTEX_ROLE = "GLOBAL"

#: The orchestrator's env flag for whether IT takes the GLOBAL layer. We always
#: take it (it is what makes overlap detection role-independent for AutoKernel
#: claims); this name exists so `check_dispatch_exclusion` can say precisely what
#: our GLOBAL hold does and does not prove about orchestrator dispatch.
ORCHESTRATOR_CROSS_ROLE_FLAG = "ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"

SYSFS_CPU_ROOT = Path("/sys/devices/system/cpu")

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_POLL_S = 0.05
DEFAULT_STALE_GRACE_S = 30.0

_MAX_PAYLOAD_BYTES = 65536
_MAX_CPU_ID = 4095   # matches `evaluator.recipes._cpu_list_members`

#: Highest physical core id, derived from the region table rather than typed, so
#: a topology change cannot leave a stale `95` behind.
MAX_PHYSICAL_CORE = max(hi for _lo, hi in REGION_CORE_RANGE.values())

# Defect classes, reusing `device_claim`'s vocabulary so one journal reads
# consistently across CPU and GPU claims.
DEFECT_LIVE_HOLDER_FREE_LOCK = "cpu_region_claim.live_holder_without_lock"
DEFECT_UNVERIFIABLE_CLAIM = "cpu_region_claim.unverifiable_claim"

STATE_HELD = "held"

#: The shared three-valued liveness vocabulary, re-exported from `device_claim`
#: so a caller reading a CPU journal record and a GPU one compares the same
#: strings. `unknown` is never collapsed into either of the others.
LIVENESS_STATES = (LIVE, DEAD, UNKNOWN)

#: The candidate-id prefix `api.EvaluationRequest.__post_init__`, `schemas._need_id`,
#: `integrity.BuildProvenance` and `worktree.BuildIdentity` all require. Duplicated
#: as a literal rather than imported: this module is a resource claim and must not
#: grow a dependency on the evaluator to mint an id. The literal is not left to be
#: kept in step by memory — `_require_disjoint_id_namespaces` below fails the
#: IMPORT if the two namespaces ever overlap again, and
#: `test_cpu_region_claim.TestTheClaimIdIsNotACandidateId` resolves it against the
#: real validator rather than against this copy.
_CANDIDATE_ID_PREFIX = "akc-"

#: `akclaim-`, NOT `akc-`. A claim id and a candidate id are different KINDS of
#: name, and until 2026-08-04 they were spelled the same: `akc-` is what
#: `api.EvaluationRequest` requires of a CANDIDATE id, so a claim id handed to a
#: parameter expecting a candidate id passed the one validator written to catch
#: exactly that substitution, and the record grammar rendered `res=akc-…` beside
#: `candidate=akc-…` with nothing to tell a reader which was which. A shared
#: prefix between two id kinds is how a validator becomes decorative.
#: `akclaim-` is the spelling `t0_provider`'s own fixtures already used.
_CLAIM_ID_PREFIX = "akclaim-"


def _require_disjoint_id_namespaces(claim_prefix: str, candidate_prefix: str) -> None:
    """Refuse a claim namespace an id validator could confuse with the candidate one.

    Disjointness here is PREFIX-disjointness in both directions, not inequality:
    if either prefix starts the other, then some id in one namespace satisfies the
    other's `startswith` test, and every validator in this package tests ids by
    prefix. `akclaim-` and `akc-` are disjoint because the fourth character is
    `l`, not `-` — which is exactly the kind of fact that survives an edit only if
    something checks it.

    Called at import so a re-merge of the two namespaces cannot reach a campaign:
    the package fails to load instead of minting ids that pass the wrong gate.
    """
    for name, value in (("claim", claim_prefix), ("candidate", candidate_prefix)):
        if not isinstance(value, str) or not value:
            raise ImportError(f"the {name} id prefix must be a non-empty string, got {value!r}")
    if claim_prefix.startswith(candidate_prefix) or candidate_prefix.startswith(claim_prefix):
        raise ImportError(
            f"claim id prefix {claim_prefix!r} and candidate id prefix {candidate_prefix!r} "
            "are not prefix-disjoint: an id minted in one namespace would satisfy the other "
            "namespace's validator, which is how a claim id passed where a candidate id "
            "belongs became undetectable. Pick a claim prefix that neither starts with nor "
            "is started by the candidate prefix")


_require_disjoint_id_namespaces(_CLAIM_ID_PREFIX, _CANDIDATE_ID_PREFIX)

#: Validated, never rewritten (see DELIBERATE DEVIATIONS). Path separators, the
#: `.lock` suffix's own dot rules and whitespace are all excluded; a leading
#: alphanumeric keeps `.`-prefixed hidden files out of the namespace.
_ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


def _region_table_is_a_partition() -> None:
    """Fail at import if the mirrored region table is not a clean partition.

    A silently overlapping or gapped mirror would put a core in two regions (or
    none), and overlap arithmetic computed from it would be wrong in the
    under-exclusion direction for the gapped case. Checked here so a bad edit
    cannot reach a benchmark.
    """
    seen: dict = {}
    for region, (lo, hi) in REGION_CORE_RANGE.items():
        if region not in ATOMIC_REGIONS:
            raise ImportError(f"region {region!r} is not in ATOMIC_REGIONS")
        if lo > hi:
            raise ImportError(f"region {region!r} has an inverted range {(lo, hi)}")
        for core in range(lo, hi + 1):
            if core in seen:
                raise ImportError(
                    f"core {core} is claimed by regions {seen[core]!r} and {region!r}"
                )
            seen[core] = region
    if set(REGION_CORE_RANGE) != set(ATOMIC_REGIONS):
        raise ImportError("REGION_CORE_RANGE and ATOMIC_REGIONS disagree")
    missing = [c for c in range(0, MAX_PHYSICAL_CORE + 1) if c not in seen]
    if missing:
        raise ImportError(f"the region table leaves cores {missing[:8]} in no region")


_region_table_is_a_partition()


# =============================================================================
# Exceptions — every one is a refusal, never a degraded success
# =============================================================================

class CpuRegionClaimError(RuntimeError):
    """Base for every CPU region claim failure."""


class CpuRegionClaimTimeout(CpuRegionClaimError):
    """The regions were held by someone else for the whole budget.

    The ordinary contention outcome and the only one a caller should retry.
    `conflicts` carries the machine-readable conflict report (which lock, whose
    pid, what attribution) so a caller can log WHO it lost to rather than
    reporting "busy".
    """

    def __init__(self, message: str, conflicts: Sequence = ()) -> None:
        super().__init__(message)
        self.conflicts = list(conflicts)


class CpuRegionClaimInconsistent(CpuRegionClaimError):
    """A lock is free but its payload cannot be shown to be abandoned."""


class CpuRegionClaimUnreadable(CpuRegionClaimError):
    """A claim payload exists but could not be read as a record."""


class CpuTopologyUnavailable(CpuRegionClaimError):
    """A logical CPU could not be mapped to a physical core.

    Never degraded into "assume it maps to itself": an unmapped sibling would
    silently drop the region it actually occupies, and the claim would then cover
    less than the measurement pins — under-exclusion, which is the direction that
    corrupts data rather than the one that costs concurrency.
    """


# =============================================================================
# CPU list parsing and the sibling fold
# =============================================================================

def parse_cpu_list(spec: Any, *, field: str = "cpu_list") -> frozenset:
    """Parse `0-95` / `184-191` / `0-3,8-11` into the set of logical cpu ids.

    Strict on purpose and matching `evaluator.recipes._cpu_list_members` rule for
    rule (empty element, inverted range, non-numeric token, implausible id): the
    footprint a claim covers and the footprint an argv pins must be parsed by the
    same grammar or "the claim covers the measurement" is not checkable.
    `test_cpu_region_claim.py` asserts the two parsers agree.
    """
    if not isinstance(spec, str):
        raise ValueError(f"{field}: expected a string cpu list, got {type(spec).__name__}")
    text = spec.strip()
    if not text:
        raise ValueError(f"{field}: an empty cpu list pins nothing and can claim nothing")
    members: set = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{field}: {text!r} has an empty range element")
        if "-" in part:
            lo_txt, _, hi_txt = part.partition("-")
            if not lo_txt.isdigit() or not hi_txt.isdigit():
                raise ValueError(f"{field}: {part!r} is not a cpu range")
            lo, hi = int(lo_txt), int(hi_txt)
            if lo > hi:
                raise ValueError(f"{field}: {part!r} is an inverted range")
        elif part.isdigit():
            lo = hi = int(part)
        else:
            raise ValueError(f"{field}: {part!r} is not a cpu id or range")
        if hi >= _MAX_CPU_ID:
            raise ValueError(f"{field}: cpu id {hi} is implausible (>= {_MAX_CPU_ID})")
        members.update(range(lo, hi + 1))
    return frozenset(members)


def render_cpu_list(cpus: Iterable) -> str:
    """Render a set of cpu ids back to the compact `0-23,88` form."""
    ordered = sorted(set(int(c) for c in cpus))
    if not ordered:
        return ""
    chunks = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        chunks.append((start, prev))
        start = prev = cpu
    chunks.append((start, prev))
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in chunks)


def read_sibling_map(sysfs_root: Optional[Path] = None) -> dict:
    """`{logical_cpu: physical_anchor_cpu}` from sysfs thread_siblings_list.

    The anchor is the LOWEST id in a sibling group, which on this host makes
    `{88: 88, 184: 88}` — the recorded fact behind
    `feedback_mi210_host_threads_smt_siblings`: 184-191 and 88-95 are eight
    physical cores addressed two ways, not sixteen cores.

    Raises `CpuTopologyUnavailable` if the topology cannot be read. It is only
    consulted for cpu ids ABOVE `MAX_PHYSICAL_CORE`; ids at or below it are their
    own anchor, which is precisely what `instance_topology.parse_cpu_list`
    assumes, so the common `0-95` case needs no sysfs at all and is deterministic
    on any host.
    """
    root = SYSFS_CPU_ROOT if sysfs_root is None else Path(sysfs_root)
    try:
        entries = sorted(root.glob("cpu[0-9]*"))
    except OSError as exc:
        raise CpuTopologyUnavailable(f"cannot list {root}: {exc}") from exc
    if not entries:
        raise CpuTopologyUnavailable(
            f"{root} exposes no cpuN entries; the SMT sibling fold cannot be resolved"
        )
    mapping: dict = {}
    for entry in entries:
        suffix = entry.name[3:]
        if not suffix.isdigit():
            continue
        cpu = int(suffix)
        siblings_file = entry / "topology" / "thread_siblings_list"
        try:
            raw = siblings_file.read_text(encoding="ascii")
        except OSError:
            # An offline CPU has no topology directory. Absence here is not an
            # error for the map as a whole; it becomes one only if somebody asks
            # to claim that specific cpu (`physical_cores` raises then).
            continue
        try:
            siblings = parse_cpu_list(raw.strip(), field=str(siblings_file))
        except ValueError as exc:
            raise CpuTopologyUnavailable(f"{siblings_file}: {exc}") from exc
        if not siblings:
            continue
        mapping[cpu] = min(siblings)
    if not mapping:
        raise CpuTopologyUnavailable(f"{root}: no cpu exposed a thread_siblings_list")
    return mapping


def physical_cores(cpus: Iterable, sibling_map: Optional[Mapping] = None) -> frozenset:
    """Fold logical cpus onto physical cores.

    `sibling_map=None` reads sysfs lazily, and ONLY if some requested cpu is
    above `MAX_PHYSICAL_CORE`. Passing an explicit map (in tests, or on a host
    whose sysfs is unavailable) is supported and is the deterministic path.
    """
    wanted = sorted(set(int(c) for c in cpus))
    if not wanted:
        return frozenset()
    resolved: set = set()
    needs_map = [c for c in wanted if c > MAX_PHYSICAL_CORE]
    if needs_map and sibling_map is None:
        sibling_map = read_sibling_map()
    for cpu in wanted:
        if cpu < 0:
            raise ValueError(f"negative cpu id {cpu}")
        if sibling_map is not None and cpu in sibling_map:
            anchor = int(sibling_map[cpu])
        elif cpu <= MAX_PHYSICAL_CORE:
            anchor = cpu
        else:
            raise CpuTopologyUnavailable(
                f"logical cpu {cpu} is above the physical range 0-{MAX_PHYSICAL_CORE} and the "
                f"thread-sibling topology does not name it, so the physical core it occupies "
                f"is unknown. Refusing to assume it maps to itself: a wrong fold makes the "
                f"claim cover LESS than the measurement pins."
            )
        if anchor > MAX_PHYSICAL_CORE:
            raise CpuTopologyUnavailable(
                f"logical cpu {cpu} folds to anchor {anchor}, which is itself outside the "
                f"physical range 0-{MAX_PHYSICAL_CORE}; the mirrored region table and this "
                f"host's topology disagree ({ORCHESTRATOR_TOPOLOGY_SOURCE})"
            )
        resolved.add(anchor)
    return frozenset(resolved)


def verify_host_topology(sysfs_root: Optional[Path] = None) -> Check:
    """Does THIS host's SMT enumeration match the assumption the fold is built on?

    `physical_cores` reads sysfs only when some requested cpu is above
    `MAX_PHYSICAL_CORE`, so for the overwhelmingly common `0-95` footprint the
    fold is computed from a MIRRORED table and never checked against the machine.
    That is correct on the EPYC 9655, where Linux enumerates one thread of each
    physical core as 0-95 and the siblings as 96-191. It is silently WRONG on any
    host that enumerates siblings adjacently (0,1 siblings; 2,3 siblings), where
    `0-95` is 48 physical cores and the region arithmetic would describe a
    machine that is not this one.

    This is the check nothing performed. It is deliberately NOT on the
    acquisition path — a sysfs walk per claim, to re-answer a question whose
    answer only changes across a reboot or a BIOS SMT change — but it is what a
    session must run once before its first benchmark, and what a host-change
    would fail.

    PASS — every cpu in `0..MAX_PHYSICAL_CORE` is its own sibling anchor and
           every higher cpu folds into that range.
    FAIL — the host disagrees with the mirrored table.
    COULD_NOT_CHECK — sysfs is unreadable (a container without
           `/sys/devices/system/cpu`); never PASS, because an unread topology is
           not a confirmed one.
    """
    try:
        mapping = read_sibling_map(sysfs_root)
    except CpuTopologyUnavailable as exc:
        return Check(COULD_NOT_CHECK, (f"host topology could not be read: {exc}",))
    mismatched = sorted(cpu for cpu, anchor in mapping.items()
                        if cpu <= MAX_PHYSICAL_CORE and anchor != cpu)
    if mismatched:
        return Check(FAIL, (
            f"cpus {mismatched[:8]} in 0-{MAX_PHYSICAL_CORE} are NOT their own sibling "
            f"anchor on this host (e.g. cpu {mismatched[0]} → {mapping[mismatched[0]]}); the "
            f"mirrored region table assumes anchor-block enumeration "
            f"({ORCHESTRATOR_TOPOLOGY_SOURCE}) and the region arithmetic would describe a "
            f"different machine",))
    escaped = sorted(cpu for cpu, anchor in mapping.items()
                     if cpu > MAX_PHYSICAL_CORE and anchor > MAX_PHYSICAL_CORE)
    if escaped:
        return Check(FAIL, (
            f"cpus {escaped[:8]} fold onto anchors outside 0-{MAX_PHYSICAL_CORE}, so they "
            f"occupy physical cores the region table does not describe",))
    return Check(PASS, (
        f"every cpu in 0-{MAX_PHYSICAL_CORE} is its own sibling anchor and all "
        f"{sum(1 for c in mapping if c > MAX_PHYSICAL_CORE)} higher cpus fold into that "
        f"range, matching {ORCHESTRATOR_TOPOLOGY_SOURCE}",))


def cores_to_regions(cores: Iterable) -> tuple:
    """Sorted tuple of the atomic regions touched by an iterable of PHYSICAL cores."""
    touched: set = set()
    for core in cores:
        for region, (lo, hi) in REGION_CORE_RANGE.items():
            if lo <= core <= hi:
                touched.add(region)
                break
        else:
            raise CpuTopologyUnavailable(
                f"physical core {core} is in no atomic region; the mirrored table covers "
                f"0-{MAX_PHYSICAL_CORE} ({ORCHESTRATOR_TOPOLOGY_SOURCE})"
            )
    return tuple(sorted(touched, key=ATOMIC_REGIONS.index))


def cpu_list_to_regions(cpu_list: str, sibling_map: Optional[Mapping] = None) -> tuple:
    """`"0-47"` → `("q0", "q1")`; `"184-191"` → `("q3",)` (the SMT fold)."""
    return cores_to_regions(physical_cores(parse_cpu_list(cpu_list), sibling_map))


def regions_overlap(a: Iterable, b: Iterable) -> frozenset:
    """The regions two claims would contend for. Empty ⇒ they may run together."""
    return frozenset(a) & frozenset(b)


def cpu_lists_overlap(a: str, b: str, sibling_map: Optional[Mapping] = None) -> frozenset:
    """Do two cpu lists contend? Returns the intersecting regions.

    `cpu_lists_overlap("0-95", "48-143")` is `{q2, q3}` — not equal, and not
    disjoint. `cpu_lists_overlap("0-47", "48-95")` is empty.
    """
    return regions_overlap(cpu_list_to_regions(a, sibling_map),
                           cpu_list_to_regions(b, sibling_map))


# =============================================================================
# Paths
# =============================================================================

def default_region_lock_dir(environ: Optional[Mapping] = None) -> Path:
    """The orchestrator's lock root. Resolution delegated, never re-derived.

    `preflight.default_region_lock_dir` already implements the orchestrator's
    precedence (`ORCHESTRATOR_TMP_DIR`, `ORCHESTRATOR_PATHS_TMP_DIR`,
    `/mnt/raid0/llm/tmp`). A third copy is a third thing to drift; if this
    resolved somewhere else we would stop excluding the fleet and everything
    would still look healthy. There is deliberately NO AutoKernel-specific
    override env var — tests pass `lock_root=` explicitly.
    """
    return _resolve_lock_root(_pf.default_region_lock_dir(environ))


class LockRootDenied(CpuRegionClaimError):
    """The requested lock root resolves onto a frozen production tree.

    A refusal, never a fallback to a safe default: a caller that asked for the
    wrong root has a wrong idea of where the fleet's exclusion namespace lives,
    and silently relocating it would leave the claim excluding nobody while
    every check reported healthy.
    """


def _resolve_lock_root(root: Any) -> Path:
    """Resolve a lock root and REFUSE it if it touches a frozen production tree.

    This is the structural half of hard boundary 1. Nothing else in this module
    was defending it, and the module CREATES what it locks: `_open_lock_fd` does
    `mkdir(parents=True)` plus `O_CREAT`, so a lock root inside
    `/mnt/raid0/llm/llama.cpp` would have written new directories and files into
    the v8 tree and broken `git status --porcelain` byte-identity. The route is
    not hypothetical — `default_region_lock_dir` honours `ORCHESTRATOR_TMP_DIR`
    and `ORCHESTRATOR_PATHS_TMP_DIR`, so an env var in a launcher redirects it,
    and `lock_root=` is a plain caller argument.

    Both containment directions are tested, for the reason
    `storage.plan_expiry` records: a root that CONTAINS a production tree is as
    dangerous as one inside it, because everything downstream walks and creates
    beneath it. `.git` is denied for the same reason storage denies it.

    `realpath` is what makes it structural rather than a string check: a symlink
    or a `..` segment that lands in a frozen tree resolves before the prefix
    test, and `storage.production_tree_forms()` supplies the realpath forms of
    the roots themselves (this repository's own working-tree identity rule makes
    `/workspace/repos/epyc-llama` a symlink onto one of them).
    """
    resolved = Path(_storage._norm(root))
    for tree in _storage.production_tree_forms():
        if _storage._under(str(resolved), tree):
            raise LockRootDenied(
                f"region-lock root {str(resolved)!r} is inside the FROZEN production tree "
                f"{tree!r}. This module creates directories and lock files under its root; "
                f"a production tree is inviolate (CLAUDE.md v8 freeze, invariant 3)."
            )
        if _storage._under(tree, str(resolved)):
            raise LockRootDenied(
                f"region-lock root {str(resolved)!r} CONTAINS the FROZEN production tree "
                f"{tree!r}: every path this module creates would be walked from a root that "
                f"spans production. Denied in both containment directions."
            )
    if ".git" in str(resolved).split("/"):
        raise LockRootDenied(
            f"region-lock root {str(resolved)!r} is inside a .git directory; repository "
            "internals are never an exclusion namespace"
        )
    return resolved


def _validated_role(role: Any) -> str:
    if not isinstance(role, str) or not _ROLE_RE.match(role):
        raise ValueError(
            f"invalid region-lock role {role!r}: must match {_ROLE_RE.pattern}. Roles are "
            "validated, never rewritten — the orchestrator maps '/' to '_', and silently "
            "rewriting one role into another's lock file breaks exclusion instead of fixing "
            "a typo."
        )
    return role


def _validated_region(region: Any) -> str:
    if region not in ATOMIC_REGIONS:
        raise ValueError(f"invalid atomic region {region!r}: expected one of {ATOMIC_REGIONS}")
    return region


def region_lock_path(role: str, region: str, lock_root: Optional[Any] = None) -> Path:
    """`{lock_root}/cpu_region.{role}.{region}.lock` — the orchestrator's name."""
    root = _resolve_lock_root(lock_root) if lock_root is not None else default_region_lock_dir()
    return root / f"cpu_region.{_validated_role(role)}.{_validated_region(region)}.lock"


def global_region_lock_path(region: str, lock_root: Optional[Any] = None) -> Path:
    """`{lock_root}/cpu_region.GLOBAL.{region}.lock` — the cross-role mutex layer."""
    root = _resolve_lock_root(lock_root) if lock_root is not None else default_region_lock_dir()
    return root / f"cpu_region.{GLOBAL_MUTEX_ROLE}.{_validated_region(region)}.lock"


def _new_id(prefix: str = _CLAIM_ID_PREFIX) -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def _utc_now_iso(now: Optional[float] = None) -> str:
    moment = (datetime.now(timezone.utc) if now is None
              else datetime.fromtimestamp(now, tz=timezone.utc))
    return moment.isoformat()


def _parse_iso(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise CpuRegionClaimUnreadable(f"{field_name}: expected ISO-8601, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CpuRegionClaimUnreadable(f"{field_name}: {value!r} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise CpuRegionClaimUnreadable(f"{field_name}: {value!r} has no timezone offset")
    return parsed


# =============================================================================
# The plan — one total lock order, computed before anything is opened
# =============================================================================

@dataclass(frozen=True)
class RegionPlan:
    """Everything an acquisition will touch, derived and ordered up front.

    Separating the plan from the acquisition is what makes the lock ORDER
    testable without taking a single lock: `plan.lock_steps` is the exact
    sequence `acquire_cpu_region_claim` walks, and a mutation that reordered it
    (roles before GLOBAL, unsorted regions) is a failing assertion rather than a
    deadlock discovered at 3am against another repository's process.
    """

    role: str
    cpu_list: str
    cpus: frozenset
    physical_cores: frozenset
    regions: tuple
    roles: tuple
    lock_steps: tuple      # ((lock_role, region, Path), …) in acquisition order
    lock_root: str

    @property
    def scope_id(self) -> str:
        return f"{self.role}:{'+'.join(self.regions)}"

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "cpu_list": self.cpu_list,
            "cpu_count": len(self.cpus),
            "physical_core_list": render_cpu_list(self.physical_cores),
            "physical_core_count": len(self.physical_cores),
            "regions": list(self.regions),
            "roles": list(self.roles),
            "lock_paths": [str(p) for _r, _g, p in self.lock_steps],
            "lock_root": self.lock_root,
        }


def plan_region_claim(
    cpu_list: str,
    *,
    role: str,
    co_roles: Iterable = (),
    lock_root: Optional[Any] = None,
    sibling_map: Optional[Mapping] = None,
) -> RegionPlan:
    """Derive the regions and the total lock order for a claim.

    Args:
        cpu_list: taskset-style list, e.g. the `taskset -c` argument of the
            constructed recipe argv (`recipes.ClaimFootprint.cpu_list`).
        role: the attribution role this claim holds under, e.g. `"autokernel"`.
            It becomes `cpu_region.{role}.{region}.lock`, which is what a human
            running `region-lock` and what `preflight.read_region_claims` see.
        co_roles: ADDITIONAL orchestrator roles whose per-role locks to hold.
            The GLOBAL layer excludes other AutoKernel claims unconditionally,
            but it excludes an ORCHESTRATOR dispatch only when the orchestrator
            was started with `ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT` on.
            Naming a role here holds ITS lock and therefore blocks its dispatch
            regardless of that flag. `roles_present()` lists the candidates and
            `check_dispatch_exclusion()` reports what a given claim does and does
            not exclude.

    The order is GLOBAL-all (region-sorted), then per-role (role-sorted, then
    region-sorted). That is a linear extension of the order
    `cpu_region_lock.cpu_region_lock` uses for its own single role, which is why
    an AutoKernel claim and an orchestrator dispatch cannot deadlock against each
    other: neither can hold a lock that the other acquires earlier in the order.
    """
    role = _validated_role(role)
    if role == GLOBAL_MUTEX_ROLE:
        raise ValueError(
            f"{GLOBAL_MUTEX_ROLE!r} is the reserved cross-role mutex pseudo-role and carries "
            "no attribution payload; claim under a real role name"
        )
    cpus = parse_cpu_list(cpu_list)
    cores = physical_cores(cpus, sibling_map)
    regions = cores_to_regions(cores)
    if not regions:
        raise ValueError(f"cpu list {cpu_list!r} touches no atomic region")

    extra = []
    for name in co_roles:
        validated = _validated_role(name)
        if validated == GLOBAL_MUTEX_ROLE:
            raise ValueError(
                f"{GLOBAL_MUTEX_ROLE!r} is always held and must not be listed in co_roles"
            )
        if validated != role:
            extra.append(validated)
    roles = tuple(sorted(set(extra) | {role}))

    root = _resolve_lock_root(lock_root) if lock_root is not None else default_region_lock_dir()
    steps = [(GLOBAL_MUTEX_ROLE, region, global_region_lock_path(region, root))
             for region in regions]
    for lock_role in roles:
        for region in regions:
            steps.append((lock_role, region, region_lock_path(lock_role, region, root)))

    return RegionPlan(
        role=role,
        cpu_list=render_cpu_list(cpus),
        cpus=cpus,
        physical_cores=cores,
        regions=regions,
        roles=roles,
        lock_steps=tuple(steps),
        lock_root=str(root),
    )


# =============================================================================
# Journal — a reclamation that is not recorded did not happen
# =============================================================================

class RegionClaimJournal:
    """Append-only JSONL sink for CPU region claim events.

    COMPOSED over `device_claim.ClaimJournal` rather than reimplemented: the
    fsync-before-return discipline, the single-`write()` append and the
    raise-on-malformed read are the same guarantees and there must not be two
    versions of them. The inherited record shape keys the subject as `device_id`;
    for a CPU claim that field carries the SCOPE id (`role:q0+q1`), and
    `detail.region_scope` states it under a name a human reading a mixed journal
    will not misread.

    `acquire_cpu_region_claim` accepts any object exposing
    `.append(kind, scope, detail)`, so a caller that wants ONE journal for its
    GPU and CPU claims can pass a bare `ClaimJournal`. There is no no-op default:
    invariant 7 makes every outcome durable, and a fail-open sink is the shape
    that has poisoned this project's stores before while everything reported
    healthy.
    """

    def __init__(self, path: Any) -> None:
        self._journal = ClaimJournal(path)

    @property
    def path(self) -> Path:
        return self._journal.path

    def append(self, kind: str, scope: str, detail: Mapping) -> dict:
        return self._journal.append(kind, scope, {"region_scope": scope, **dict(detail)})

    def read_all(self) -> list:
        return self._journal.read_all()


def _require_journal(journal: Any) -> Any:
    if journal is None:
        raise TypeError(
            "a CPU region claim requires a journal: reclamations must be durable before "
            "they take effect (invariant 7). Pass a RegionClaimJournal; there is no no-op "
            "default on purpose."
        )
    if not callable(getattr(journal, "append", None)):
        raise TypeError(
            f"journal {type(journal).__name__} has no callable .append(kind, scope, detail)"
        )
    return journal


# =============================================================================
# Receipt
# =============================================================================

_RECEIPT_FIELDS = (
    "schema", "claim_id", "role", "roles", "cpu_list", "physical_core_list", "regions",
    "lock_paths", "lock_root", "state", "holder_pid", "holder_start_ticks",
    "holder_boot_id", "host", "holder_label", "purpose", "campaign_id", "acquired_at",
    "expires_at", "released_at", "reclaimed_from",
)


def expected_lock_paths(roles: Iterable, regions: Iterable,
                        lock_root: Any) -> tuple:
    """The COMPLETE lock-path set a claim on `roles`×`regions` must hold.

    Derived exactly as `plan_region_claim` derives `lock_steps`: the GLOBAL
    mutex for every region, plus every role's lock for every region. Sorted, so
    it is comparable as a set and printable as a diff.
    """
    root = Path(lock_root)
    paths = [str(root / f"cpu_region.{GLOBAL_MUTEX_ROLE}.{_validated_region(r)}.lock")
             for r in regions]
    for role in sorted(set(roles)):
        for region in regions:
            paths.append(str(root /
                             f"cpu_region.{_validated_role(role)}."
                             f"{_validated_region(region)}.lock"))
    return tuple(sorted(paths))


def _receipt_inconsistencies(receipt: Mapping, *,
                             sibling_map: Optional[Mapping] = None) -> list:
    """Every way a receipt contradicts itself. Empty list ⇒ self-consistent.

    WHY THIS EXISTS — the recurring defect class in this package is a correct
    guard wired to the wrong input. `check_region_claim_held` iterates the
    receipt's own `lock_paths`, and `check_footprint_covered` compares a
    footprint against the receipt's own `regions`. Both fields are supplied by
    the party being gated, and neither checker re-derived them. Two concrete
    passes were demonstrated against the real module:

      * a receipt that had really claimed only `0-23` (regions `['q0']`), edited
        to say `cpu_list="0-95"`, `regions=["q0","q1","q2","q3"]`, PASSED
        `check_footprint_covered(receipt, "0-95")` — q1-q3 were free the whole
        time;
      * the same receipt with `lock_paths` truncated to the single payload-free
        `cpu_region.GLOBAL.q0.lock` PASSED `check_region_claim_held` with the
        reason "holds every lock it names". A conjunct satisfiable by deleting
        it is not a conjunct.

    Every one of these fields is DERIVED at acquisition from `cpu_list`,
    `role`/`roles` and `lock_root`, so equality with the re-derivation is an
    invariant of any honest receipt and a cheap, total refutation of a dishonest
    one. `physical_core_list` is checked for the same reason.

    Note what is NOT claimed here: self-consistency says the receipt describes a
    coherent claim, never that the claim is held. That is
    `check_region_claim_held`'s job and it reads the machine.
    """
    problems: list = []
    role = receipt.get("role")
    roles = receipt.get("roles")
    regions = receipt.get("regions")
    lock_paths = receipt.get("lock_paths")
    lock_root = receipt.get("lock_root")
    cpu_list = receipt.get("cpu_list")

    if not isinstance(roles, (list, tuple)) or not roles:
        problems.append(f"roles must be a non-empty sequence, got {roles!r}")
        roles = ()
    if not isinstance(regions, (list, tuple)) or not regions:
        problems.append(f"regions must be a non-empty sequence, got {regions!r}")
        regions = ()
    if not isinstance(lock_paths, (list, tuple)):
        problems.append(f"lock_paths must be a sequence, got {lock_paths!r}")
        lock_paths = ()
    if role is not None and roles and role not in roles:
        problems.append(f"role {role!r} is not among roles {list(roles)}")
    bad_regions = [r for r in regions if r not in ATOMIC_REGIONS]
    if bad_regions:
        problems.append(f"regions {bad_regions} are not atomic regions {list(ATOMIC_REGIONS)}")
    if list(regions) != sorted(regions, key=lambda r: ATOMIC_REGIONS.index(r)
                               if r in ATOMIC_REGIONS else 99):
        problems.append(f"regions {list(regions)} are not in canonical region order")

    if regions and roles and not bad_regions and isinstance(lock_root, str) and lock_root:
        try:
            expected = expected_lock_paths(roles, regions, lock_root)
        except ValueError as exc:
            problems.append(f"lock paths cannot be derived: {exc}")
        else:
            actual = tuple(sorted(str(p) for p in lock_paths))
            if actual != expected:
                missing = [p for p in expected if p not in actual]
                extra = [p for p in actual if p not in expected]
                problems.append(
                    f"lock_paths do not match the {len(expected)} locks that roles "
                    f"{sorted(set(roles))} × regions {list(regions)} require under "
                    f"{lock_root!r} — missing {missing}, unexpected {extra}"
                )
    elif not isinstance(lock_root, str) or not lock_root:
        problems.append(f"lock_root must be a non-empty string, got {lock_root!r}")
    if isinstance(lock_root, str) and lock_root:
        try:
            _resolve_lock_root(lock_root)
        except LockRootDenied as exc:
            problems.append(str(exc))
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"lock_root {lock_root!r} could not be resolved: {exc}")

    if isinstance(cpu_list, str) and regions and not bad_regions:
        try:
            cores = physical_cores(parse_cpu_list(cpu_list), sibling_map)
            derived = cores_to_regions(cores)
        except (ValueError, CpuTopologyUnavailable):
            # Unresolvable topology is NOT silently accepted as consistent — it
            # is reported by `check_receipt_self_consistent` as COULD_NOT_CHECK.
            # Deserialisation is the one place it is skipped, because refusing to
            # rebuild a `184-191` receipt on a host without sysfs would make a
            # stored record unreadable rather than unsafe.
            pass
        else:
            if tuple(regions) != derived:
                problems.append(
                    f"cpu_list {cpu_list!r} occupies regions {list(derived)}, but the receipt "
                    f"records {list(regions)}"
                )
            recorded_cores = receipt.get("physical_core_list")
            if isinstance(recorded_cores, str) and recorded_cores != render_cpu_list(cores):
                problems.append(
                    f"physical_core_list {recorded_cores!r} is not the fold of cpu_list "
                    f"{cpu_list!r} ({render_cpu_list(cores)!r})"
                )
    return problems


def check_receipt_self_consistent(receipt: Any,
                                  sibling_map: Optional[Mapping] = None) -> Check:
    """Does the receipt describe a claim that could actually have been taken?

    PASS — every derived field agrees with its derivation.
    FAIL — some field contradicts another; the receipt is fabricated or stale.
    COULD_NOT_CHECK — the receipt is not a mapping, or its `cpu_list` needs a
        thread-sibling topology this host cannot supply, so the
        `cpu_list ↔ regions` conjunct could not be evaluated. NOT a pass:
        an unevaluated conjunct is exactly the hole this function closes.
    """
    if isinstance(receipt, RegionClaimReceipt):
        receipt = receipt.to_dict()
    if not isinstance(receipt, Mapping):
        return Check(COULD_NOT_CHECK, (f"receipt is a {type(receipt).__name__}",))
    problems = _receipt_inconsistencies(receipt, sibling_map=sibling_map)
    if problems:
        return Check(FAIL, tuple(problems))
    cpu_list = receipt.get("cpu_list")
    if isinstance(cpu_list, str):
        try:
            physical_cores(parse_cpu_list(cpu_list), sibling_map)
        except CpuTopologyUnavailable as exc:
            return Check(COULD_NOT_CHECK, (
                f"cpu_list {cpu_list!r} could not be folded onto physical cores ({exc}), so "
                "the cpu_list↔regions conjunct was NOT evaluated",))
        except ValueError as exc:
            return Check(FAIL, (f"cpu_list {cpu_list!r} is unparseable: {exc}",))
    else:
        return Check(FAIL, (f"cpu_list must be a string, got {cpu_list!r}",))
    return Check(PASS, (
        f"receipt {receipt.get('claim_id')!r}: regions, physical_core_list and all "
        f"{len(receipt.get('lock_paths') or ())} lock paths re-derive from cpu_list "
        f"{cpu_list!r}, roles {sorted(set(receipt.get('roles') or ()))} and the lock root",))


@dataclass(frozen=True)
class RegionClaimReceipt:
    """Immutable snapshot of one CPU region claim.

    `claim_id` is the string that goes into
    `evaluation_event.resource_claim_receipt`. Without it an event asserts a
    number with no evidence that anything was kept off the cores while it was
    taken — which on a 96-core shared host is the difference between a
    measurement and an anecdote.
    """

    claim_id: str
    role: str
    roles: tuple
    cpu_list: str
    physical_core_list: str
    regions: tuple
    lock_paths: tuple
    lock_root: str
    state: str
    holder_pid: int
    holder_start_ticks: int
    holder_boot_id: str
    host: str
    purpose: str
    campaign_id: str
    acquired_at: str
    holder_label: Optional[str] = None
    expires_at: Optional[str] = None
    released_at: Optional[str] = None
    reclaimed_from: Optional[tuple] = None
    schema: str = RECEIPT_SCHEMA

    def to_dict(self) -> dict:
        out = {}
        for name in _RECEIPT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, tuple):
                value = [dict(v) if isinstance(v, Mapping) else v for v in value]
            out[name] = value
        return out

    @classmethod
    def from_dict(cls, obj: Mapping) -> "RegionClaimReceipt":
        """Rebuild a receipt. Raises on a missing or unknown field.

        Tolerating either would let a receipt round-trip into a DIFFERENT
        receipt, and the whole point of the id is that it means one thing.
        """
        if not isinstance(obj, Mapping):
            raise TypeError(f"receipt must be a mapping, got {type(obj).__name__}")
        missing = [n for n in _RECEIPT_FIELDS if n not in obj]
        if missing:
            raise ValueError(f"receipt is missing required fields: {missing}")
        unknown = [n for n in obj if n not in _RECEIPT_FIELDS]
        if unknown:
            raise ValueError(f"receipt carries unknown fields: {sorted(unknown)}")
        if obj["schema"] != RECEIPT_SCHEMA:
            raise ValueError(f"receipt schema {obj['schema']!r} != {RECEIPT_SCHEMA!r}")
        values = {n: obj[n] for n in _RECEIPT_FIELDS}
        for name in ("roles", "regions", "lock_paths"):
            values[name] = tuple(values[name])
        if values["reclaimed_from"] is not None:
            values["reclaimed_from"] = tuple(values["reclaimed_from"])
        problems = _receipt_inconsistencies(obj)
        if problems:
            raise ValueError(
                "receipt is internally inconsistent and cannot be reconstructed: "
                + "; ".join(problems)
                + ". Every field here is DERIVED from `cpu_list`, `role`/`roles` and "
                  "`lock_root` at acquisition; a receipt whose fields disagree describes a "
                  "claim that was never taken."
            )
        return cls(**values)


# =============================================================================
# Payload I/O — always performed while holding the flock
# =============================================================================

def _claim_payload(*, lock_role: str, region: str, plan: RegionPlan, holder: Mapping,
                   claim_id: str, purpose: str, campaign_id: str, acquired_at: str,
                   started_at: float, expires_at: Optional[str],
                   reclaimed_from: Optional[Mapping]) -> dict:
    """The on-disk attribution payload: the orchestrator's contract + our extension.

    The first block is byte-for-byte the orchestrator's
    `_lock_payload_for_region` shape. Anything reading this namespace today —
    `read_region_lock_payload`, `active_region_holders`,
    `preflight.read_region_claims` — keeps working, and `request_tag` is where a
    human sees that AutoKernel is the holder, because that is the field the
    orchestrator's "still waiting for region lock" log and preflight's
    attribution both quote.
    """
    return {
        # --- the orchestrator's contract ---------------------------------
        "schema_version": ORCHESTRATOR_PAYLOAD_SCHEMA_VERSION,
        "pid": holder["pid"],
        "role": lock_role,
        "region": region,
        "regions": list(plan.regions),
        "instance_idx": None,
        "request_tag": f"autokernel:{claim_id}",
        "started_at": started_at,
        # --- the AutoKernel extension ------------------------------------
        "autokernel_schema": CPU_REGION_CLAIM_SCHEMA,
        "claim_id": claim_id,
        "claim_role": plan.role,
        "state": STATE_HELD,
        "holder": dict(holder),
        "cpu_list": plan.cpu_list,
        "physical_core_list": render_cpu_list(plan.physical_cores),
        "lock_roles": list(plan.roles),
        "purpose": purpose,
        "campaign_id": campaign_id,
        "acquired_at": acquired_at,
        "expires_at": expires_at,
        "reclaimed_from": dict(reclaimed_from) if reclaimed_from else None,
    }


def _open_lock_fd(path: Path, *, create: bool = True) -> int:
    """Open a lock file. Never unlinked. `create=False` never makes one.

    `create` exists because CREATION IS NOT A READ. `_probe_lock_free` — the
    read-only half of `check_region_claim_held` — used to come through here with
    `O_CREAT` and `mkdir(parents=True)`, so probing a receipt that named a lock
    file which does not exist MATERIALISED it, together with every parent
    directory. Two consequences, both observed: the phantom then appeared in
    `roles_present()` and changed `check_dispatch_exclusion`'s verdict, and a
    checker — not just an acquisition — could write into whatever root the
    receipt named. A file that does not exist cannot be flocked by anyone, so
    the honest answer for the probe is "free", reached without touching the
    filesystem.

    O_CLOEXEC states the requirement explicitly: the whole purpose of this claim
    is to run a benchmark under it, and an inherited descriptor would keep the
    region locked after `release()` returned — a claim outliving its claimant,
    invisibly. Note what actually enforces it in CPython: PEP 446 has made every
    descriptor non-inheritable by default since 3.4, so this flag is belt and
    braces rather than the sole mechanism, and the test asserts the PROPERTY
    (`os.get_inheritable` is False) rather than the flag bit — a flag assertion
    would pass on a build with the flag removed and would therefore be a guard
    that cannot fail.

    flock (not fcntl/lockf) matches the orchestrator and is also the only correct
    choice: fcntl record locks are dropped when the process closes ANY descriptor
    to the file, so an unrelated open/close of the same path elsewhere in this
    process would silently release a held claim.
    """
    # Re-checked HERE and not only at plan time: this is the one place the module
    # creates anything, so the frozen-tree refusal is enforced at the syscall
    # rather than at whichever caller happened to build the path.
    _resolve_lock_root(path.parent)
    if not create:
        return os.open(path, os.O_RDWR | os.O_CLOEXEC)
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)


def _try_flock_ex(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return False
        raise


def _unlock_and_close(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_payload(fd: int, payload: Mapping) -> None:
    data = (canonical_json(payload) + "\n").encode("utf-8")
    os.ftruncate(fd, 0)
    os.pwrite(fd, data, 0)
    os.fsync(fd)


def _clear_payload(fd: int) -> None:
    os.ftruncate(fd, 0)
    os.fsync(fd)


def _parse_payload_bytes(raw: bytes, source: str) -> Optional[dict]:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CpuRegionClaimUnreadable(f"{source}: payload is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CpuRegionClaimUnreadable(
            f"{source}: payload is a {type(payload).__name__}, not an object"
        )
    return payload


def _read_payload_fd(fd: int, source: str) -> Optional[dict]:
    raw = os.pread(fd, _MAX_PAYLOAD_BYTES + 1, 0)
    if len(raw) > _MAX_PAYLOAD_BYTES:
        raise CpuRegionClaimUnreadable(
            f"{source}: payload exceeds {_MAX_PAYLOAD_BYTES} bytes; treating as corruption"
        )
    return _parse_payload_bytes(raw, source)


def _read_payload_path(path: Path, *, attempts: int = 5,
                       retry_s: float = 0.02) -> Optional[dict]:
    """Read a payload WITHOUT the lock (observer path), tolerating a torn write.

    A holder rewrites in place while holding the lock, so an unlocked reader can
    catch a partial write. The window is microseconds and the write happens twice
    per claim, so a short bounded retry closes it. After the retries a parse
    failure is reported as unreadable — never as "no claim", which would read as
    "region free".
    """
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with open(path, "rb") as fh:
                raw = fh.read(_MAX_PAYLOAD_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(raw) > _MAX_PAYLOAD_BYTES:
            raise CpuRegionClaimUnreadable(
                f"{path}: payload exceeds {_MAX_PAYLOAD_BYTES} bytes; treating as corruption "
                "(retrying cannot shrink it)"
            )
        try:
            return _parse_payload_bytes(raw, str(path))
        except CpuRegionClaimUnreadable as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(retry_s)
    raise CpuRegionClaimUnreadable(f"{path}: {last_exc}")


def _probe_lock_free(lock_path: Path) -> Optional[bool]:
    """True if nothing holds the flock right now. ADVISORY ONLY.

    Observation is not a claim (invariant 9): the answer is stale the instant it
    is returned. It exists for `check_region_claim_held` and for dashboards,
    never as a precondition for using cores.
    """
    try:
        fd = _open_lock_fd(lock_path, create=False)
    except FileNotFoundError:
        # No file, therefore no flock, therefore nothing is excluded. FREE is
        # the true answer and the one that makes `check_region_claim_held` FAIL.
        return True
    except OSError:
        return None
    try:
        if _try_flock_ex(fd):
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        return False
    except OSError:
        return None
    finally:
        os.close(fd)


# =============================================================================
# Classification of a payload found under a FREE lock
# =============================================================================

_TAKE = "take"
_WAIT = "wait"
_REFUSE = "refuse"

_KIND_AUTOKERNEL = "autokernel"
_KIND_ORCHESTRATOR = "orchestrator"


@dataclass(frozen=True)
class _Disposition:
    action: str
    reason: str
    liveness: Optional[Liveness] = None
    previous: Optional[dict] = None
    payload_kind: Optional[str] = None


def _payload_age_s(payload: Mapping, now: float) -> Optional[float]:
    """Seconds since the payload was written, or None if it cannot be dated.

    Both timestamp forms are accepted because both are in the namespace: ours
    (`acquired_at`, ISO-8601) and the orchestrator's (`started_at`, epoch float).
    """
    acquired = payload.get("acquired_at")
    if isinstance(acquired, str):
        try:
            return now - _parse_iso(acquired, "acquired_at").timestamp()
        except CpuRegionClaimUnreadable:
            pass
    started = payload.get("started_at")
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        return now - float(started)
    return None


def _classify(payload: Optional[dict], *, lock_path: Path, stale_grace_s: float,
              now: float) -> _Disposition:
    """Decide what to do with a payload found while we hold its FREE lock.

    TAKE   — no payload; a positively dead AutoKernel holder past the grace; or
             orchestrator debris past the grace (see below).
    WAIT   — a dead holder still inside the grace period; retry later.
    REFUSE — a live AutoKernel holder, a holder whose liveness cannot be
             determined, a payload that cannot be dated, or a payload whose shape
             belongs to neither contract. Never a takeover.

    The orchestrator-debris branch is the one considered divergence from
    `device_claim`, and it is the namespace owner's own rule: `cpu_region_lock`
    writes its payload only AFTER taking the flock and clears it BEFORE closing,
    so a payload under a free flock cannot belong to a live holder — its writer
    was killed between the two. `sweep_stale_region_lock_payloads` clears exactly
    this state without consulting PIDs at all, for exactly this reason.
    """
    if payload is None:
        return _Disposition(_TAKE, f"{lock_path.name} carries no attribution payload")

    if payload.get("autokernel_schema") == CPU_REGION_CLAIM_SCHEMA:
        liveness = assess_holder_liveness(payload.get("holder"))
        if liveness.state == LIVE:
            self_note = ""
            holder = payload.get("holder")
            if isinstance(holder, Mapping) and holder.get("pid") == os.getpid():
                self_note = (
                    " — the payload names THIS process, so an earlier claim released its "
                    "lock without clearing its payload"
                )
            return _Disposition(
                _REFUSE,
                f"{lock_path.name} is free but its recorded AutoKernel holder is alive "
                f"({liveness.reason}){self_note}. A live holder is never stolen from.",
                liveness=liveness, previous=payload, payload_kind=_KIND_AUTOKERNEL,
            )
        if liveness.state == UNKNOWN:
            return _Disposition(
                _REFUSE,
                f"{lock_path.name}: the recorded holder's liveness cannot be determined "
                f"({liveness.reason}); inability to evaluate is not evidence of death",
                liveness=liveness, previous=payload, payload_kind=_KIND_AUTOKERNEL,
            )
        age = _payload_age_s(payload, now)
        if age is None:
            return _Disposition(
                _REFUSE,
                f"{lock_path.name}: the holder is dead ({liveness.reason}) but the claim "
                "carries no usable acquired_at, so it cannot be aged against the reclaim "
                "grace period",
                liveness=liveness, previous=payload, payload_kind=_KIND_AUTOKERNEL,
            )
        if age < stale_grace_s:
            return _Disposition(
                _WAIT,
                f"{lock_path.name}: holder is dead ({liveness.reason}) but the claim is only "
                f"{age:.3f}s old, inside the {stale_grace_s:.3f}s reclaim grace",
                liveness=liveness, previous=payload, payload_kind=_KIND_AUTOKERNEL,
            )
        return _Disposition(
            _TAKE,
            f"{lock_path.name}: holder is dead ({liveness.reason}) and the claim is "
            f"{age:.3f}s old, past the {stale_grace_s:.3f}s grace",
            liveness=liveness, previous=payload, payload_kind=_KIND_AUTOKERNEL,
        )

    if payload.get("schema_version") == ORCHESTRATOR_PAYLOAD_SCHEMA_VERSION:
        age = _payload_age_s(payload, now)
        if age is None:
            return _Disposition(
                _REFUSE,
                f"{lock_path.name} carries an orchestrator payload with no usable "
                "`started_at`, so it cannot be aged; clear it with the orchestrator's own "
                "`region-lock sweep` (sweep_stale_region_lock_payloads) rather than "
                "having a second repo guess",
                previous=payload, payload_kind=_KIND_ORCHESTRATOR,
            )
        if age < stale_grace_s:
            return _Disposition(
                _WAIT,
                f"{lock_path.name} carries orchestrator debris written {age:.3f}s ago, "
                f"inside the {stale_grace_s:.3f}s grace",
                previous=payload, payload_kind=_KIND_ORCHESTRATOR,
            )
        return _Disposition(
            _TAKE,
            f"{lock_path.name} carries an orchestrator attribution payload (pid "
            f"{payload.get('pid')!r}, role {payload.get('role')!r}) under a FREE flock, "
            f"written {age:.3f}s ago: by that namespace's own contract the flock is the "
            "sole occupancy fact, so this is debris from a holder killed before cleanup",
            previous=payload, payload_kind=_KIND_ORCHESTRATOR,
        )

    return _Disposition(
        _REFUSE,
        f"{lock_path.name} carries a payload matching neither the AutoKernel claim schema "
        f"nor the orchestrator's schema_version {ORCHESTRATOR_PAYLOAD_SCHEMA_VERSION} "
        f"(keys: {sorted(payload)[:8]}); refusing to classify an unknown record as free",
        previous=payload,
    )


def _payload_summary(payload: Mapping) -> dict:
    holder = payload.get("holder")
    return {
        "claim_id": payload.get("claim_id"),
        "role": payload.get("role"),
        "claim_role": payload.get("claim_role"),
        "region": payload.get("region"),
        "pid": payload.get("pid"),
        "holder": dict(holder) if isinstance(holder, Mapping) else None,
        "acquired_at": payload.get("acquired_at"),
        "started_at": payload.get("started_at"),
        "purpose": payload.get("purpose"),
        "campaign_id": payload.get("campaign_id"),
        "request_tag": payload.get("request_tag"),
    }


# =============================================================================
# Conflict reporting — who did we lose to?
# =============================================================================

def probe_region_conflicts(plan: RegionPlan) -> list:
    """ADVISORY report of what currently holds any lock in `plan`.

    Observation, never a gate: acquire the claim instead of consulting this. It
    exists so a timeout can say WHO, and so an operator dashboard can show the
    machine. Degrades to an `error` entry rather than raising, because a claim
    failure must not be masked by a diagnostics failure.
    """
    wanted = {(lock_role, region) for lock_role, region, _p in plan.lock_steps}
    out: list = []
    try:
        claims = _pf.read_region_claims(Path(plan.lock_root),
                                        require_nonempty_namespace=False)
    except Exception as exc:   # noqa: BLE001 - diagnostics must not mask the failure
        return [{"error": f"region-lock namespace unreadable: {type(exc).__name__}: {exc}",
                 "lock_root": plan.lock_root}]
    for claim in claims:
        if (claim.role, claim.region) not in wanted:
            continue
        if not claim.held:
            continue
        payload = claim.payload if isinstance(claim.payload, Mapping) else None
        out.append({
            "role": claim.role,
            "region": claim.region,
            "lock_path": claim.lock_path,
            "holder_pids": list(claim.holders.holder_pids),
            "unattributed_holders": claim.holders.unattributed_holders,
            "attribution": _payload_summary(payload) if payload else None,
            "notes": list(claim.notes),
        })
    return out


def _conflict_note(conflicts: Sequence) -> str:
    if not conflicts:
        return "(no holder could be attributed; the lock may have just been released)"
    parts = []
    for entry in conflicts:
        if "error" in entry:
            parts.append(entry["error"])
            continue
        attribution = entry.get("attribution") or {}
        parts.append(
            f"{entry['role']}.{entry['region']} held by pid(s) {entry['holder_pids']}"
            f" tag={attribution.get('request_tag')!r} purpose={attribution.get('purpose')!r}"
        )
    return "; ".join(parts)


# =============================================================================
# The claim handle
# =============================================================================

class CpuRegionClaim:
    """A held CPU region claim. Construct via `acquire_cpu_region_claim`.

    Usable as a context manager and released exactly once, on the normal and the
    exception path alike. Release is idempotent so a `finally` after a partial
    failure is safe.
    """

    def __init__(self, *, plan: RegionPlan, holds: Sequence,
                 receipt: RegionClaimReceipt, journal: Any) -> None:
        self._plan = plan
        # ((lock_role, region, Path, fd), …) in ACQUISITION order; released LIFO.
        self._holds = list(holds)
        self._receipt = receipt
        self._journal = journal
        self._released = False
        self._final_receipt: Optional[RegionClaimReceipt] = None
        # Tracked apart from `_released`: dropping the locks is what makes the
        # release real, but a release nobody recorded is not durable (invariant
        # 7), so a failed journal write stays retryable instead of being latched.
        self._release_journaled = False
        self._release_context: Optional[dict] = None

    # -- identity ---------------------------------------------------------
    @property
    def claim_id(self) -> str:
        return self._receipt.claim_id

    @property
    def plan(self) -> RegionPlan:
        return self._plan

    @property
    def regions(self) -> tuple:
        return self._plan.regions

    @property
    def role(self) -> str:
        return self._plan.role

    def verify_held(self) -> Check:
        """RE-READ the machine and report whether this claim still excludes anyone.

        WHY THIS IS NOT A FLAG. `held` used to be `not self._released` — an
        in-memory boolean recording only that nobody in this process had called
        `release()`. `microbench.CpuRegionClaimAdapter.attest()` consumes exactly
        that boolean as its per-invocation attestation, and `microbench.HeldClaim`
        states the requirement in its own docstring: *"A conforming claim re-reads
        its own lock on every attest() call. Returning a cached PASS defeats the
        entire mid-run revocation check."* The flag could not defeat the one
        failure the re-read exists for.

        The failure is real, not hypothetical. `storage.EPHEMERAL_ROOTS` lists
        `/mnt/raid0/llm/tmp` — the region-lock root — as a SCRATCH root, i.e. one
        this project's own storage plane declares sweepable. If anything unlinks
        or replaces a lock file (a tmp sweep, a human tidying a 96%-full disk,
        the orchestrator's namespace being recreated), our flock survives on an
        ORPHANED INODE while the path every other actor tests is a fresh, free
        file. Nothing has been revoked and nothing errors; we simply stop
        excluding anyone, and the flag says `True` forever. A benchmark then runs
        contended and the evidence record asserts it was claimed.

        Checked per lock: the fd we hold and the path still name the SAME inode,
        and (for the attributed per-role locks) the payload read back through our
        own descriptor still carries our `claim_id`. Both are cheap syscalls on a
        file we already have open.

        FAIL — released, or a lock no longer excludes anyone under its path.
        COULD_NOT_CHECK — a lock could not be examined. NEVER PASS: an
        unverifiable claim is not a verified one.
        """
        if self._released:
            return Check(FAIL, (f"claim {self.claim_id} has been released",))
        reasons: list = []
        for lock_role, region, path, fd in self._holds:
            try:
                ours = os.fstat(fd)
            except OSError as exc:
                return Check(COULD_NOT_CHECK,
                             (f"cannot stat our own descriptor for {path}: {exc}",))
            try:
                ondisk = os.stat(path)
            except FileNotFoundError:
                return Check(FAIL, (
                    f"{path} no longer exists: our flock survives on an orphaned inode, so "
                    f"region {region} reads as FREE to every other actor and this claim "
                    f"excludes nobody",))
            except OSError as exc:
                return Check(COULD_NOT_CHECK, (f"cannot stat {path}: {exc}",))
            if (ondisk.st_dev, ondisk.st_ino) != (ours.st_dev, ours.st_ino):
                return Check(FAIL, (
                    f"{path} has been REPLACED (inode {ours.st_ino} → {ondisk.st_ino}): our "
                    f"flock is on the old inode and region {region} is claimable by anyone",))
            if lock_role == GLOBAL_MUTEX_ROLE:
                reasons.append(f"{path.name}: same inode (exclusion-only, no payload)")
                continue
            try:
                payload = _read_payload_fd(fd, str(path))
            except CpuRegionClaimUnreadable as exc:
                return Check(COULD_NOT_CHECK, (f"payload unreadable at {path}: {exc}",))
            if payload is None or payload.get("claim_id") != self.claim_id:
                return Check(FAIL, (
                    f"{path} no longer records claim {self.claim_id!r} "
                    f"(found {None if payload is None else payload.get('claim_id')!r}): the "
                    f"attribution this region carries is not ours",))
            reasons.append(f"{path.name}: same inode, payload records {self.claim_id}")
        return Check(PASS, tuple(reasons))

    @property
    def held(self) -> bool:
        """True only if `verify_held()` PASSES — fail-closed on anything else.

        Deliberately not `not self._released`; see `verify_held`. Consumers of
        this property (notably `microbench.CpuRegionClaimAdapter`) treat it as
        the attestation that the region is still excluded, so it must read the
        machine and must report False when it cannot.
        """
        return self.verify_held().outcome == PASS

    @property
    def lock_paths(self) -> tuple:
        return tuple(str(p) for _r, _g, p, _fd in self._holds)

    def receipt(self) -> RegionClaimReceipt:
        """Snapshot receipt. After release it carries `released_at`."""
        return self._final_receipt if self._final_receipt is not None else self._receipt

    def covers(self, cpu_list: str, sibling_map: Optional[Mapping] = None) -> bool:
        """Does this claim's held region set cover every core `cpu_list` pins?"""
        needed = cores_to_regions(physical_cores(parse_cpu_list(cpu_list), sibling_map))
        return set(needed).issubset(set(self._plan.regions))

    # -- release ----------------------------------------------------------
    def release(self) -> RegionClaimReceipt:
        """Release every lock (LIFO) and journal it. Idempotent.

        The journal step is RETRIED by a repeat call until it succeeds. Caching
        "already released" over a failed record write would make the durability
        step unretryable: the locks are gone, no later actor can reconstruct the
        release, and the outcome would be permanently missing (invariant 7).
        """
        first_call = not self._released
        if first_call:
            released_at = _utc_now_iso()
            clear_errors: list = []
            for lock_role, region, path, fd in reversed(self._holds):
                try:
                    if lock_role != GLOBAL_MUTEX_ROLE:
                        # Leaving our payload behind under a free lock is the
                        # exact unresolvable state `_classify` refuses to touch,
                        # so the clear is attempted for every lock even if an
                        # earlier one failed.
                        _clear_payload(fd)
                except OSError as exc:
                    clear_errors.append({
                        "lock_path": str(path), "role": lock_role, "region": region,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                finally:
                    # The locks go no matter what. Holding a quarter of the
                    # machine because a bookkeeping step failed is strictly worse
                    # than a missing record, and the record is written below.
                    _unlock_and_close(fd)
            self._released = True
            self._final_receipt = RegionClaimReceipt(
                **{**self._receipt.to_dict(),
                   "roles": tuple(self._receipt.roles),
                   "regions": tuple(self._receipt.regions),
                   "lock_paths": tuple(self._receipt.lock_paths),
                   "reclaimed_from": (tuple(self._receipt.reclaimed_from)
                                      if self._receipt.reclaimed_from else None),
                   "released_at": released_at}
            )
            self._release_context = {"released_at": released_at,
                                     "clear_errors": clear_errors}

        self._journal_release()
        assert self._final_receipt is not None
        if first_call and self._release_context["clear_errors"]:
            raise CpuRegionClaimError(
                f"claim {self.claim_id} released its locks but could not clear "
                f"{len(self._release_context['clear_errors'])} attribution payload(s): "
                f"{self._release_context['clear_errors']}. Those regions now carry a payload "
                "naming a LIVE process beside a free lock and are NOT claimable until a "
                "human (or `region-lock sweep`) truncates them. A defect was journaled."
            )
        return self._final_receipt

    def _journal_release(self) -> None:
        if self._release_journaled or self._release_context is None:
            return
        ctx = self._release_context
        if ctx["clear_errors"]:
            self._journal.append(KIND_DEFECT, self._plan.scope_id, {
                "defect_class": DEFECT_LIVE_HOLDER_FREE_LOCK,
                "reason": "attribution payloads could not be cleared on release, leaving a "
                          "payload naming a live process beside a free lock",
                "claim_id": self.claim_id,
                "failures": ctx["clear_errors"],
                "observer_pid": os.getpid(),
            })
        self._journal.append(KIND_RELEASED, self._plan.scope_id, {
            "claim_id": self.claim_id,
            "released_at": ctx["released_at"],
            "receipt": self._final_receipt.to_dict(),
            "payload_clear_errors": ctx["clear_errors"],
        })
        self._release_journaled = True

    def __enter__(self) -> "CpuRegionClaim":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
        return None


# =============================================================================
# Acquisition
# =============================================================================

def acquire_cpu_region_claim(
    cpu_list: str,
    *,
    purpose: str,
    campaign_id: str,
    journal: Any,
    role: str = "autokernel",
    co_roles: Iterable = (),
    holder_label: Optional[str] = None,
    timeout_s: Optional[float] = DEFAULT_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    stale_grace_s: float = DEFAULT_STALE_GRACE_S,
    max_hold_s: Optional[float] = None,
    lock_root: Optional[Any] = None,
    sibling_map: Optional[Mapping] = None,
    cancel_check: Optional[Callable] = None,
) -> CpuRegionClaim:
    """Acquire the CPU region claim covering `cpu_list`, or raise.

    Args:
        cpu_list: the taskset-style footprint to claim, e.g. `"0-95"` for the
            canonical CPU baseline or `"184-191"` for the GPU host threads. Take
            it from the constructed argv (`recipes.ClaimFootprint.cpu_list`), not
            from memory — the claim must cover what the command actually pins.
        purpose / campaign_id: required and non-empty. An unattributable claim is
            indistinguishable from a leaked one.
        journal: where reclamations and releases are recorded. Required.
        role: attribution role for the per-role lock files.
        co_roles: extra orchestrator roles to exclude (see `plan_region_claim`).
        timeout_s: budget. `None` blocks forever (write it out on purpose);
            `<= 0` makes exactly ONE attempt. Diverges from `cpu_region_lock`,
            where `<= 0` means block forever — deliberately, in the visible
            direction.
        stale_grace_s: how old an abandoned payload must be before its lock may
            be reclaimed.
        max_hold_s: declared maximum hold, recorded as an advisory `expires_at`.
            An expired claim is NEVER stolen; nothing here preempts a holder.

    Raises:
        CpuRegionClaimTimeout: contended for the whole budget. `.conflicts`
            names the holders.
        CpuRegionClaimInconsistent: a lock was free but its payload could not be
            shown to be abandoned. A defect is journaled and nothing is taken.
    """
    journal = _require_journal(journal)
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be a non-empty string")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be a non-empty string")
    if poll_s <= 0:
        raise ValueError(f"poll_s must be positive, got {poll_s!r}")
    if stale_grace_s < 0:
        raise ValueError(f"stale_grace_s must be >= 0, got {stale_grace_s!r}")
    if max_hold_s is not None and max_hold_s <= 0:
        raise ValueError(f"max_hold_s must be positive when set, got {max_hold_s!r}")

    plan = plan_region_claim(cpu_list, role=role, co_roles=co_roles,
                             lock_root=lock_root, sibling_map=sibling_map)
    holder = current_holder_identity(holder_label)

    deadline = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
    single_attempt = timeout_s is not None and timeout_s <= 0
    attempts = 0
    last_wait_reason = "every region was held for the whole budget"

    while True:
        attempts += 1
        if cancel_check is not None and cancel_check():
            raise CpuRegionClaimTimeout(
                f"CPU region claim on {plan.cpu_list!r} ({'+'.join(plan.regions)}) cancelled "
                f"before acquisition (purpose={purpose!r})",
                probe_region_conflicts(plan),
            )

        holds: list = []
        keep = False
        blocked = None
        reclaimed: list = []
        try:
            for lock_role, region, path in plan.lock_steps:
                fd = _open_lock_fd(path)
                taken = False
                try:
                    if not _try_flock_ex(fd):
                        blocked = f"{path.name} is held by another process"
                        break
                    if lock_role != GLOBAL_MUTEX_ROLE:
                        try:
                            payload = _read_payload_fd(fd, str(path))
                            disposition = _classify(payload, lock_path=path,
                                                    stale_grace_s=stale_grace_s,
                                                    now=time.time())
                        except CpuRegionClaimUnreadable as exc:
                            journal.append(KIND_DEFECT, plan.scope_id, {
                                "defect_class": DEFECT_UNVERIFIABLE_CLAIM,
                                "reason": str(exc),
                                "lock_path": str(path),
                                "observer_pid": os.getpid(),
                            })
                            raise CpuRegionClaimInconsistent(
                                f"{path}: {exc}. The region is NOT reclaimed — a payload that "
                                "cannot be parsed cannot be shown to be abandoned. A human "
                                "must confirm nothing is using these cores and truncate it."
                            ) from exc
                        if disposition.action == _REFUSE:
                            journal.append(KIND_DEFECT, plan.scope_id, {
                                "defect_class": DEFECT_LIVE_HOLDER_FREE_LOCK,
                                "reason": disposition.reason,
                                "liveness": (disposition.liveness.state
                                             if disposition.liveness else None),
                                "lock_path": str(path),
                                "recorded_claim": (_payload_summary(disposition.previous)
                                                   if disposition.previous else None),
                                "observer_pid": os.getpid(),
                            })
                            raise CpuRegionClaimInconsistent(
                                f"region {region} (role {lock_role}): {disposition.reason}"
                            )
                        if disposition.action == _WAIT:
                            blocked = disposition.reason
                            break
                        if disposition.previous is not None:
                            reclaimed.append((path, lock_role, region, disposition))
                    holds.append((lock_role, region, path, fd))
                    taken = True
                finally:
                    if not taken:
                        _unlock_and_close(fd)

            if blocked is None:
                started_at = time.time()
                acquired_at = _utc_now_iso(started_at)
                expires_at = (_utc_now_iso(started_at + max_hold_s)
                              if max_hold_s is not None else None)
                claim_id = _new_id()

                # Journal BEFORE the takeover: a reclamation that is not recorded
                # must not be able to happen. If this raises, the finally block
                # releases every lock and nothing on disk moved.
                reclaimed_from: list = []
                for path, lock_role, region, disposition in reclaimed:
                    summary = _payload_summary(disposition.previous)
                    summary["lock_path"] = str(path)
                    summary["payload_kind"] = disposition.payload_kind
                    reclaimed_from.append(summary)
                    journal.append(KIND_RECLAIMED, plan.scope_id, {
                        "claim_id": claim_id,
                        "reason": disposition.reason,
                        "payload_kind": disposition.payload_kind,
                        "liveness": (disposition.liveness.state
                                     if disposition.liveness else None),
                        "liveness_reason": (disposition.liveness.reason
                                            if disposition.liveness else None),
                        "stale_grace_s": stale_grace_s,
                        "lock_path": str(path),
                        "reclaimed_from": summary,
                        "reclaimed_by_pid": holder["pid"],
                    })

                for lock_role, region, path, fd in holds:
                    if lock_role == GLOBAL_MUTEX_ROLE:
                        # Exclusion only, exactly as the orchestrator does it:
                        # its sweeper SKIPS GLOBAL files, so a payload we left
                        # here after a crash is debris nothing ever clears.
                        continue
                    payload = _claim_payload(
                        lock_role=lock_role, region=region, plan=plan, holder=holder,
                        claim_id=claim_id, purpose=purpose, campaign_id=campaign_id,
                        acquired_at=acquired_at, started_at=started_at,
                        expires_at=expires_at,
                        reclaimed_from=next(
                            (r for r in reclaimed_from if r["lock_path"] == str(path)), None),
                    )
                    _write_payload(fd, payload)

                receipt = RegionClaimReceipt(
                    claim_id=claim_id,
                    role=plan.role,
                    roles=plan.roles,
                    cpu_list=plan.cpu_list,
                    physical_core_list=render_cpu_list(plan.physical_cores),
                    regions=plan.regions,
                    lock_paths=tuple(str(p) for _r, _g, p, _fd in holds),
                    lock_root=plan.lock_root,
                    state=STATE_HELD,
                    holder_pid=holder["pid"],
                    holder_start_ticks=holder["start_ticks"],
                    holder_boot_id=holder["boot_id"],
                    host=holder["host"],
                    holder_label=holder_label,
                    purpose=purpose,
                    campaign_id=campaign_id,
                    acquired_at=acquired_at,
                    expires_at=expires_at,
                    reclaimed_from=tuple(reclaimed_from) or None,
                )
                claim = CpuRegionClaim(plan=plan, holds=holds,
                                       receipt=receipt, journal=journal)
                try:
                    journal.append(KIND_ACQUIRED, plan.scope_id, {
                        "claim_id": claim_id,
                        "receipt": receipt.to_dict(),
                        "attempts": attempts,
                        "reclaimed": bool(reclaimed_from),
                        "plan": plan.to_dict(),
                    })
                except BaseException:
                    # Payloads are on disk and the locks are about to be dropped
                    # by the `finally` below. A payload naming this LIVE process
                    # beside a free lock is the unresolvable state `_classify`
                    # refuses to touch, so it must not survive a failed
                    # acquisition.
                    for lock_role, _region, path, fd in holds:
                        if lock_role != GLOBAL_MUTEX_ROLE:
                            _clear_payload(fd)
                    raise
                keep = True
                return claim

            last_wait_reason = blocked
        finally:
            if not keep:
                for _lock_role, _region, _path, fd in reversed(holds):
                    _unlock_and_close(fd)

        if single_attempt or (deadline is not None and time.monotonic() >= deadline):
            conflicts = probe_region_conflicts(plan)
            raise CpuRegionClaimTimeout(
                f"could not claim cpus {plan.cpu_list!r} (regions {'+'.join(plan.regions)}, "
                f"roles {list(plan.roles)}) within "
                f"{'a single attempt' if single_attempt else f'{timeout_s}s'}: "
                f"{last_wait_reason}. Holders: {_conflict_note(conflicts)}",
                conflicts,
            )
        time.sleep(poll_s)


@contextmanager
def cpu_region_claim(cpu_list: str, **kwargs: Any):
    """`with cpu_region_claim("0-95", …) as claim:` — released on exit."""
    claim = acquire_cpu_region_claim(cpu_list, **kwargs)
    try:
        yield claim
    finally:
        claim.release()


# =============================================================================
# Footprints sourced from the codified recipes, never retyped
# =============================================================================

def _recipes():
    """Import `evaluator.recipes` lazily.

    Lazy on purpose: `recipes` hashes `scripts/lib/canonical_recipe.py` at import
    and raises if it is unreadable. A claim on an explicit cpu list must not be
    blocked by an unrelated file, and a CPU-only campaign must not pay for the
    GPU launcher parse.
    """
    from ..evaluator import recipes as _r
    return _r


def canonical_cpu_baseline_cpu_list() -> str:
    """The canonical CPU baseline footprint — read from `CANONICAL_PREFIX`.

    `taskset -c 0-95 numactl --interleave=all` is the ratified prefix; the cpu
    list is the argument of its own `-c`, so a change there reaches this claim on
    the next import instead of being retyped here
    (`feedback_use_codified_recipes_not_memory`).
    """
    prefix = list(_recipes().CANONICAL_PREFIX)
    try:
        index = prefix.index("-c")
    except ValueError as exc:
        raise CpuRegionClaimError(
            f"the ratified CANONICAL_PREFIX {prefix} has no `-c` cpu list; refusing to guess "
            "the canonical baseline footprint"
        ) from exc
    if index + 1 >= len(prefix):
        raise CpuRegionClaimError(f"CANONICAL_PREFIX {prefix} ends after `-c`")
    return prefix[index + 1]


def gpu_host_cpu_list() -> str:
    """The MI210 host-thread footprint — read from the codified GPU launcher.

    Returns `184-191`, the SMT siblings of physical cores 88-95, NOT `88-95`.
    Sourced through `recipes.gpu_host_cpu_list()` so the provenance note that
    records `88-95` as SUPERSEDED stays the single home of that correction. Note
    what the fold means for exclusion: this claim takes region q3, so it conflicts
    with a CPU claim on 72-95 and with the canonical 0-95 baseline — the siblings
    are a different thread of the SAME physical cores.
    """
    return _recipes().gpu_host_cpu_list()


# =============================================================================
# Checkers — the evaluator side
# =============================================================================

def check_region_claim_held(receipt: Any, *, lock_root: Optional[Any] = None) -> Check:
    """Is the claim named by `receipt` actually held right now?

    PASS — every lock in the receipt is held AND every per-role payload names
           this claim.
    FAIL — any lock is free (the claim leaked), or a payload names a different
           claim (someone else holds the region).
    COULD_NOT_CHECK — a payload or lock file could not be read.

    The lock probe is required and is not redundant with the payload check: "the
    payload names me" is passable by writing a file, and a claim whose flock has
    been released excludes nobody while still reading as held.
    """
    if isinstance(receipt, RegionClaimReceipt):
        receipt = receipt.to_dict()
    if not isinstance(receipt, Mapping):
        return Check(COULD_NOT_CHECK, (f"receipt is a {type(receipt).__name__}",))
    claim_id = receipt.get("claim_id")
    lock_paths = receipt.get("lock_paths")
    if not isinstance(claim_id, str) or not isinstance(lock_paths, (list, tuple)):
        return Check(COULD_NOT_CHECK, ("receipt lacks claim_id/lock_paths",))
    if not lock_paths:
        return Check(FAIL, ("receipt names no lock files, so it excludes nothing",))

    # The set of locks to probe MUST NOT be the set the receipt chose to list.
    # A receipt truncated to the payload-free GLOBAL lock passed this function
    # with "holds every lock it names" while three quarters of the machine were
    # unclaimed. Re-derive first; only then is iterating `lock_paths` meaningful.
    consistent = check_receipt_self_consistent(receipt)
    if consistent.outcome != PASS:
        return Check(consistent.outcome if consistent.outcome == COULD_NOT_CHECK else FAIL,
                     ("the receipt does not describe a claim that could have been taken, so "
                      "what it names cannot be checked",) + tuple(consistent.reasons))

    root = Path(lock_root) if lock_root is not None else None
    reasons: list = []
    for raw_path in lock_paths:
        path = Path(raw_path)
        if root is not None:
            path = root / path.name
        free = _probe_lock_free(path)
        if free is None:
            return Check(COULD_NOT_CHECK, (f"could not probe the lock file {path}",))
        if free:
            return Check(FAIL, (
                f"claim {claim_id!r} names {path} but its flock is FREE: the claim leaked "
                "and nothing is excluding other processes from those cores",
            ))
        if path.name.startswith(f"cpu_region.{GLOBAL_MUTEX_ROLE}."):
            # Exclusion-only layer: held, and by the namespace contract it carries
            # NO payload, so "held" here cannot be attributed to THIS claim — the
            # orchestrator holds the same files when its cross-role placement flag
            # is on. Recorded as what it is rather than folded into the PASS
            # reason, so an evidence reader is not told we proved more than we did.
            reasons.append(f"{path.name} is held (unattributable: the GLOBAL layer "
                           f"carries no payload, so the holder may be another actor)")
            continue
        try:
            payload = _read_payload_path(path)
        except (CpuRegionClaimUnreadable, OSError) as exc:
            return Check(COULD_NOT_CHECK, (f"payload unreadable at {path}: {exc}",))
        if payload is None:
            return Check(FAIL, (f"{path} is locked but carries no claim payload",))
        if payload.get("claim_id") != claim_id:
            return Check(FAIL, (
                f"{path} is recorded to claim {payload.get('claim_id')!r}, not {claim_id!r}",
            ))
        reasons.append(f"{path.name} holds {claim_id}")
    return Check(PASS, tuple(reasons) or (f"claim {claim_id!r} holds every lock it names",))


def check_footprint_covered(receipt: Any, footprint: Any,
                            sibling_map: Optional[Mapping] = None) -> Check:
    """Does the claim cover every core the measurement pins? HALF of precondition 1.

    `footprint` is a cpu-list string or anything with a `.cpu_list` attribute —
    `recipes.ClaimFootprint` is the intended input, because it is DERIVED from
    the constructed argv's own `taskset -c` rather than declared beside it.

    FAIL is the case that matters: a claim on `0-47` under a command that pins
    `0-95` leaves half the machine unclaimed, and the resulting number is a
    measurement of a contended host.

    THIS FUNCTION READS NO LOCK. It answers "the claim described by this receipt
    would cover that footprint", not "a claim is held". A fabricated receipt for
    a claim that never existed passed it — that is not a defect in this function,
    it is the boundary of what a pure comparison can assert, and it was
    previously mislabelled: the docstring said it answered precondition 1
    outright. `check_precondition_1` conjoins it with the lock probe, and is what
    a caller about to run a benchmark must use.
    """
    cpu_list = getattr(footprint, "cpu_list", footprint)
    if isinstance(receipt, RegionClaimReceipt):
        receipt = receipt.to_dict()
    if not isinstance(receipt, Mapping):
        return Check(COULD_NOT_CHECK, (f"receipt is a {type(receipt).__name__}",))
    consistent = check_receipt_self_consistent(receipt, sibling_map)
    if consistent.outcome != PASS:
        # `regions` is the receipt's own assertion. Re-derived first, or a
        # receipt that really held only q0 while recording q0-q3 covers `0-95`.
        return Check(consistent.outcome if consistent.outcome == COULD_NOT_CHECK else FAIL,
                     ("coverage is undecidable against a receipt whose own fields "
                      "disagree",) + tuple(consistent.reasons))
    held_regions = receipt.get("regions")
    if not isinstance(held_regions, (list, tuple)) or not held_regions:
        return Check(COULD_NOT_CHECK, ("receipt names no regions",))
    try:
        needed = cores_to_regions(physical_cores(parse_cpu_list(cpu_list), sibling_map))
    except (ValueError, CpuTopologyUnavailable) as exc:
        return Check(COULD_NOT_CHECK, (f"footprint {cpu_list!r} could not be resolved: {exc}",))
    uncovered = [r for r in needed if r not in set(held_regions)]
    if uncovered:
        return Check(FAIL, (
            f"the measurement pins {cpu_list!r}, which occupies regions {list(needed)}, but "
            f"claim {receipt.get('claim_id')!r} holds only {list(held_regions)}: "
            f"{uncovered} are unclaimed",
        ))
    return Check(PASS, (
        f"claim {receipt.get('claim_id')!r} holds {list(held_regions)}, covering the "
        f"{list(needed)} that {cpu_list!r} pins",
    ))


def check_precondition_1(receipt: Any, footprint: Any, *,
                         lock_root: Optional[Any] = None,
                         sibling_map: Optional[Mapping] = None) -> Check:
    """`P-AK-SEARCH-1` precondition 1, whole: a HELD claim COVERING the footprint.

    The conjunction, in one call, because the two halves were separable and the
    coverage half PASSes on data alone. A caller that ran only
    `check_footprint_covered` before a benchmark would have proven that a receipt
    was arithmetically adequate and nothing about whether any lock existed.

    PASS only when both hold. Any COULD_NOT_CHECK propagates as COULD_NOT_CHECK
    (never as PASS); any FAIL wins outright.
    """
    covered = check_footprint_covered(receipt, footprint, sibling_map)
    held = check_region_claim_held(receipt, lock_root=lock_root)
    reasons = tuple(covered.reasons) + tuple(held.reasons)
    if covered.outcome == FAIL or held.outcome == FAIL:
        return Check(FAIL, reasons)
    if covered.outcome == COULD_NOT_CHECK or held.outcome == COULD_NOT_CHECK:
        return Check(COULD_NOT_CHECK, reasons)
    return Check(PASS, reasons)


def check_dispatch_exclusion(receipt: Any, *, lock_root: Optional[Any] = None,
                             environ: Optional[Mapping] = None) -> Check:
    """What does this claim actually keep OFF the cores it holds?

    PASS — every role with a lock file in the namespace, for every region held,
           is held BY THIS CLAIM. Nothing in the fleet can dispatch onto them.
    COULD_NOT_CHECK — some role's per-region lock is not held by this claim.
           Those roles are excluded only through the GLOBAL mutex layer, and the
           orchestrator consults that layer only when it was started with
           `ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT` enabled — a fact about
           ANOTHER process's environment that this one cannot read. Reporting it
           as PASS would be an assumption about a second repository's runtime
           config; reporting FAIL would accuse a namespace that may be fully
           excluded. Pass those roles as `co_roles=` to convert this to PASS.
    FAIL — the claim is not held at all.
    """
    held = check_region_claim_held(receipt, lock_root=lock_root)
    if held.outcome != PASS:
        return Check(held.outcome, ("dispatch exclusion is undecidable unless the claim is "
                                    "held",) + tuple(held.reasons))
    if isinstance(receipt, RegionClaimReceipt):
        receipt = receipt.to_dict()
    root = Path(lock_root) if lock_root is not None else Path(receipt.get("lock_root", ""))
    regions = set(receipt.get("regions") or ())
    claimed_roles = set(receipt.get("roles") or ())
    try:
        present = roles_present(root)
    except Exception as exc:   # noqa: BLE001 - reported, never silently passed
        return Check(COULD_NOT_CHECK, (f"cannot enumerate roles in {root}: {exc}",))
    outside = sorted(r for r in present if r not in claimed_roles and r != GLOBAL_MUTEX_ROLE)
    env = os.environ if environ is None else environ
    if outside:
        return Check(COULD_NOT_CHECK, (
            f"claim {receipt.get('claim_id')!r} holds regions {sorted(regions)} for roles "
            f"{sorted(claimed_roles)}, but the namespace also contains roles {outside}. Those "
            f"are excluded only via the GLOBAL mutex, which the orchestrator honours only "
            f"when started with {ORCHESTRATOR_CROSS_ROLE_FLAG} enabled (this process sees it "
            f"as {env.get(ORCHESTRATOR_CROSS_ROLE_FLAG)!r}, which says nothing about the "
            f"orchestrator's own environment). Pass co_roles={outside} to exclude them "
            f"unconditionally.",
        ))
    return Check(PASS, (
        f"claim {receipt.get('claim_id')!r} holds the GLOBAL mutex plus every role lock "
        f"present in {root} for regions {sorted(regions)}",
    ))


def check_claim_expiry(receipt: Any, *, now: Optional[float] = None) -> Check:
    """Has the claim outlived the maximum hold it declared?

    A FAIL is a reason to go and ask the holder to finish, never a licence to
    reclaim: expiry is a declaration BY the holder, not a fact ABOUT it, and this
    module never preempts anything.
    """
    if isinstance(receipt, RegionClaimReceipt):
        receipt = receipt.to_dict()
    if not isinstance(receipt, Mapping):
        return Check(COULD_NOT_CHECK, (f"receipt is a {type(receipt).__name__}",))
    expires_at = receipt.get("expires_at")
    if expires_at is None:
        return Check(COULD_NOT_CHECK, (
            f"claim {receipt.get('claim_id')!r} declared no maximum hold",))
    try:
        deadline = _parse_iso(expires_at, "expires_at")
    except CpuRegionClaimUnreadable as exc:
        return Check(COULD_NOT_CHECK, (str(exc),))
    moment = time.time() if now is None else now
    if moment <= deadline.timestamp():
        return Check(PASS, (f"claim {receipt.get('claim_id')!r} expires at {expires_at}",))
    return Check(FAIL, (
        f"claim {receipt.get('claim_id')!r} passed its declared expiry {expires_at}; ask the "
        "holder to release — an expired claim is still not reclaimable while its holder lives",
    ))


# =============================================================================
# Advisory views
# =============================================================================

def roles_present(lock_root: Optional[Any] = None) -> tuple:
    """Roles that have a lock file in the namespace, GLOBAL included.

    ADVISORY and incomplete BY CONSTRUCTION: a role that has never dispatched has
    no lock file, so absence here is not proof that a role cannot appear. It is
    an aid for choosing `co_roles`, never a gate.
    """
    root = _resolve_lock_root(lock_root) if lock_root is not None else default_region_lock_dir()
    found: set = set()
    for path in root.glob("cpu_region.*.*.lock"):
        stem = path.name[len("cpu_region."):-len(".lock")]
        role, _, region = stem.rpartition(".")
        if role and region:
            found.add(role)
    return tuple(sorted(found))


def inspect_region_claims(lock_root: Optional[Any] = None) -> dict:
    """Read-only diagnostic view of the whole region-lock namespace.

    ADVISORY: every field is stale the moment it is returned. Nothing may decide
    to use cores from this dict — that is the `gpu_idle()` mistake §10.4 forbids.
    Acquire the claim instead.
    """
    root = _resolve_lock_root(lock_root) if lock_root is not None else default_region_lock_dir()
    out: dict = {
        "lock_root": str(root),
        "advisory": True,
        "observed_at": _utc_now_iso(),
        "regions": {},
    }
    try:
        claims = _pf.read_region_claims(root, require_nonempty_namespace=False)
    except Exception as exc:   # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    for claim in claims:
        entry = {
            "role": claim.role,
            "region": claim.region,
            "lock_path": claim.lock_path,
            "held": claim.held,
            "holder_pids": list(claim.holders.holder_pids),
            "payload_is_stale": claim.payload_is_stale,
            "notes": list(claim.notes),
            "attribution": (_payload_summary(claim.payload)
                            if isinstance(claim.payload, Mapping) else None),
        }
        if isinstance(claim.payload, Mapping):
            entry["holder_liveness"] = (
                assess_holder_liveness(claim.payload.get("holder")).state
                if claim.payload.get("autokernel_schema") == CPU_REGION_CLAIM_SCHEMA
                else None
            )
        out["regions"].setdefault(claim.region, []).append(entry)
    out["held_regions"] = sorted(
        region for region, entries in out["regions"].items()
        if any(e["held"] for e in entries)
    )
    return out
