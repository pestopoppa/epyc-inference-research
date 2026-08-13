#!/usr/bin/env python3
"""preflight.py — THE single audited, read-only inference preflight (AK1, §3.5).

WHY THIS MODULE EXISTS
----------------------
Two standing rules of this host contradict each other, and until they are
reconciled in ONE place no evaluator can be protocol-conformant:

  * `CLAUDE.md` § Process Management bans `pkill`/`pgrep` on a name pattern.
    Origin INC-20260731-broad-process-pattern-kills: a broad pattern
    (`llama-server -m`) used to "clean up" a benchmark killed **another agent's
    running server, twice in one day**, and a pattern sweep killed `earlyoom`
    because its own argv contains `--ignore ^(llama-server|sd-server)$` — the
    sweep matched the guard process whose entire job is protecting the fleet
    from OOM kills. The incident's own words: the failure is *structural, not
    careless* — on a shared box any name-based pattern is a wildcard over other
    sessions' processes, and a guard's argv necessarily contains the names it
    guards.

  * The measurement protocols MANDATE the opposite. `bench-cpu.md:16-17` makes
    *"no concurrent inference (`pgrep llama` zombie check...)"* a P-BENCH-1
    precondition, `MEASUREMENT_POLICY.md:38` repeats it, and
    `gpu-cross-device.md:27-30` requires *"`llama-server` / AutoPilot / KFD PID
    checks before and after"* as mandatory P-GPU-1 evidence.

§3.5 resolves this in two stages, and this module implements BOTH, in two
clearly separated layers, with the weaker one always labelled as such:

  1. **TARGET — claim witness** (`claim_witness_preflight`). Establish "no
     concurrent inference" from WHO HOLDS THE CLAIM: the CPU region locks
     (`{tmp}/cpu_region.{role}.{region}.lock`, the cross-process exclusion fact
     used by every dispatch path) and the GPU device claim, plus enumeration of
     the cgroup/PIDs this process itself owns. A claim is authoritative about
     *intent* and is complete by construction; a process scan is a sample of a
     moving target that cannot even distinguish a resident-but-idle production
     server from an active decode. This layer is preferred whenever it can
     evaluate at all.

  2. **INTERIM — read-only name-pattern enumerator** (`interim_process_scan`).
     Reports matching PIDs and NEVER signals. §3.5's interim stage says the ban's
     blast radius was *pattern kills*, and that a read-only enumerator at ONE
     audited call site is a different object — *"but it must be that one call
     site, not `pgrep` sprinkled through the evaluator."* **This module is that
     call site.** No other module in the AutoKernel substrate may read processes
     by name pattern; anything that needs it calls here.

SIGNALLING IS STRUCTURALLY ABSENT, NOT MERELY AVOIDED
-----------------------------------------------------
This module imports no `signal`, no `subprocess`, no `ctypes`, no
`multiprocessing`, and contains no `kill`/`killpg`/`send_signal`/`terminate`
call, no `os.system`/`os.popen`, and no dynamic-attribute escape hatch
(`getattr`, `__import__`, `eval`, `exec`) through which one could be reached.
`audit_no_signalling_capability()` proves that from the module's own AST — string
literals and comments are ignored, so the docstring you are reading cannot defeat
it. That audit is a THREE-outcome checker like everything else here: if the
source cannot be read or parsed it returns COULD_NOT_CHECK, because absence of
evidence is not evidence of absence (`feedback_verify_negatives_before_
concluding_absence`). The audit proves the module cannot deliver a signal; it is
not a sandbox and does not claim to be one.

INABILITY TO EVALUATE IS A THIRD OUTCOME
----------------------------------------
`PreflightResult.verdict` is PASS, FAIL, or COULD_NOT_CHECK, and the API is
shaped so a caller cannot collapse the last one into either of the first two:

  * `PreflightResult.__bool__` RAISES. `if preflight(...):` is the exact bug this
    module exists to prevent, so it is a TypeError, not a silent misread.
  * `.passed` is True only for PASS.
  * `.require_pass()` raises `ConcurrentInferenceDetected` for FAIL and
    `PreflightIndeterminate` for COULD_NOT_CHECK — two distinct exception types,
    so an `except` clause cannot accidentally treat "the host is busy" and "I
    could not look" as the same event.

Every reader in here RAISES `PreflightUnavailable` rather than returning a
default: an unreadable claim root is NOT an empty claim root, a missing GPU claim
substrate is NOT an idle GPU, and an unreadable process is NOT an absent one.
This project has been bitten repeatedly by fail-open defaults that poisoned
stores while every component reported healthy
(`feedback_fail_open_defaults_conceal_their_own_corruption`), and a preflight
that fails open is worth less than no preflight at all, because it launders a
non-check into an attestation. For the same reason a region-lock directory that
exists but contains no lock files at all is COULD_NOT_CHECK by default, not
"no claims": on this host that namespace is never empty, so an empty read means
the path is wrong far more often than it means the fleet is idle.

WHAT THIS MODULE NEVER DOES
---------------------------
No signal. No process start or stop. No inference. No benchmark. No write of any
kind — every filesystem operation is a read. The result object is designed to be
journaled verbatim as the precondition attestation of the evaluation event it
gated (`to_dict()` is `schemas.canonical_json`-safe), because a precondition that
was checked but not recorded is indistinguishable from one that was skipped
(§3.7, invariant 7).
"""
from __future__ import annotations

import ast
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

# --- schemas import -------------------------------------------------------
# The verdict vocabulary has ONE source of truth (`schemas.PASS/FAIL/
# COULD_NOT_CHECK`); redefining the strings here would create a second one that
# drifts silently. The `sys.path` fallback is the repo's established idiom for
# running a module both as a package member and as a plain file, but mutating
# `sys.path` at import is itself AutoPilot scar item 12 (§2.5) — ambient import
# identity, *"which code scores your eval depends on which eval ran first"*. The
# assertion below is the mitigation: the module we bound MUST be the schemas.py
# sitting next to this package, or we refuse to import at all.
_SCHEMAS_PATH = Path(__file__).resolve().parent.parent / "schemas.py"
try:
    from .. import schemas as _schemas
except ImportError:
    _PKG_DIR = str(_SCHEMAS_PATH.parent)
    if _PKG_DIR not in sys.path:
        sys.path.insert(0, _PKG_DIR)
    import schemas as _schemas  # type: ignore[no-redef]

if Path(_schemas.__file__).resolve() != _SCHEMAS_PATH:
    raise ImportError(
        f"autokernel.resource.preflight bound a foreign schemas module: "
        f"{_schemas.__file__} is not {_SCHEMAS_PATH}"
    )

Check = _schemas.Check
PASS = _schemas.PASS
FAIL = _schemas.FAIL
COULD_NOT_CHECK = _schemas.COULD_NOT_CHECK

# =============================================================================
# Verdicts and their lattice
# =============================================================================

# Severity ordering for combining independent sub-checks. FAIL outranks
# COULD_NOT_CHECK outranks PASS: one conclusive failure decides the whole
# preflight, and one blind spot downgrades an otherwise clean result. There is
# deliberately no way to combine upward.
_VERDICT_SEVERITY = {PASS: 0, COULD_NOT_CHECK: 1, FAIL: 2}

# Which layer produced the verdict. Recorded in the attestation because a PASS
# from the interim scan is a WEAKER statement than a PASS from claim witness,
# and the difference must survive into the evidence record rather than being
# lost at the call site.
BASIS_CLAIM_WITNESS = "CLAIM_WITNESS"
BASIS_INTERIM_PROCESS_SCAN = "INTERIM_PROCESS_SCAN"
BASIS_NONE = "NONE"

BASES = frozenset({BASIS_CLAIM_WITNESS, BASIS_INTERIM_PROCESS_SCAN, BASIS_NONE})


def combine_verdicts(*verdicts: str) -> str:
    """Worst-of the given verdicts, under FAIL > COULD_NOT_CHECK > PASS."""
    if not verdicts:
        raise ValueError("combine_verdicts() needs at least one verdict")
    worst = PASS
    for verdict in verdicts:
        if verdict not in _VERDICT_SEVERITY:
            raise ValueError(f"not a verdict: {verdict!r}")
        if _VERDICT_SEVERITY[verdict] > _VERDICT_SEVERITY[worst]:
            worst = verdict
    return worst


# =============================================================================
# Exceptions
# =============================================================================


class PreflightUnavailable(RuntimeError):
    """A reader could not establish a fact. NEVER caught to produce a default.

    Raised by the low-level readers and triaged exactly once, by the checker
    that owns the verdict, into COULD_NOT_CHECK. The distinction matters: a
    reader that returned `[]` on an unreadable directory would make "nothing is
    running" and "I cannot see" the same value.
    """


class PreflightNotSatisfied(RuntimeError):
    """Base for `require_pass()` failures. Never raised directly.

    Carries the `result` that produced it. Without that, the recommended call
    site (`require_no_concurrent_inference`) would DISCARD the attestation on
    exactly the two outcomes §4 invariant 7 says must be durable — "failures,
    crashes, timeouts, rejected proposals, negative results ... remain in an
    append-only event journal". A caller that catches the exception can now
    journal `exc.result.to_dict()` verbatim.
    """

    def __init__(self, message: str, result: Optional["PreflightResult"] = None) -> None:
        super().__init__(message)
        self.result = result


class ConcurrentInferenceDetected(PreflightNotSatisfied):
    """FAIL — something else holds the resource. The run must not start."""


class PreflightIndeterminate(PreflightNotSatisfied):
    """COULD_NOT_CHECK — the precondition could not be established.

    A separate type from `ConcurrentInferenceDetected` on purpose: a caller that
    wants to retry, escalate, or record `EVALUATOR_COVERAGE_GAP` must be able to
    tell "the host is busy" from "I could not look", and a shared exception type
    would let one `except` clause silently merge them.
    """


# =============================================================================
# Scope — what this run needs exclusive use of
# =============================================================================


@dataclass(frozen=True)
class PreflightScope:
    """The resources a run needs, and the protocol precondition it satisfies.

    `cpu_regions=None` means THE WHOLE MACHINE — every region, i.e. the
    canonical `taskset -c 0-95` P-BENCH-1 recipe. None is the strict reading on
    purpose: a scope built wrong fails closed.
    """

    label: str
    cpu_regions: Optional[frozenset] = None
    gpu_devices: frozenset = frozenset()
    protocol_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("PreflightScope.label must be a non-empty string")
        if self.cpu_regions is not None:
            if not isinstance(self.cpu_regions, frozenset):
                raise TypeError("cpu_regions must be a frozenset or None (= whole machine)")
            for region in self.cpu_regions:
                if not isinstance(region, str) or not region:
                    raise ValueError(f"invalid cpu region id: {region!r}")
        if not isinstance(self.gpu_devices, frozenset):
            raise TypeError("gpu_devices must be a frozenset")
        for device in self.gpu_devices:
            if not isinstance(device, str) or not device:
                raise ValueError(f"invalid gpu device id: {device!r}")
        if self.cpu_regions == frozenset() and not self.gpu_devices:
            # A preflight over nothing would always PASS and would attest to
            # nothing at all — the most dangerous possible result.
            raise ValueError(
                "PreflightScope covers no CPU region and no GPU device; a preflight "
                "over an empty scope would attest to nothing"
            )

    @classmethod
    def whole_machine_cpu(cls, label: str, protocol_id: str = "P-BENCH-1") -> "PreflightScope":
        return cls(label=label, cpu_regions=None, protocol_id=protocol_id)

    @classmethod
    def cpu(cls, label: str, regions: Iterable, protocol_id: str = "P-BENCH-1") -> "PreflightScope":
        return cls(label=label, cpu_regions=frozenset(regions), protocol_id=protocol_id)

    @classmethod
    def gpu(cls, label: str, devices: Iterable, protocol_id: str = "P-GPU-1") -> "PreflightScope":
        return cls(
            label=label,
            cpu_regions=frozenset(),
            gpu_devices=frozenset(devices),
            protocol_id=protocol_id,
        )

    @property
    def covers_cpu(self) -> bool:
        return self.cpu_regions is None or bool(self.cpu_regions)

    def covers_region(self, region: str) -> bool:
        return self.cpu_regions is None or region in self.cpu_regions

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "cpu_regions": None if self.cpu_regions is None else sorted(self.cpu_regions),
            "gpu_devices": sorted(self.gpu_devices),
            "protocol_id": self.protocol_id,
        }


# =============================================================================
# /proc access — injectable so tests never depend on live host state
# =============================================================================


@dataclass(frozen=True)
class ProcSource:
    """Where process facts are read from, and who 'we' are.

    Injectable because a test that had to observe the live `/proc` of a shared
    box would be testing the box, not the code — and because the fixtures below
    must be able to construct process trees that would be unethical to create
    for real on this host.
    """

    root: Path = Path("/proc")
    self_pid: int = field(default_factory=os.getpid)

    def pid_dir(self, pid: int) -> Path:
        return self.root / str(pid)


def _read_proc_text(path: Path) -> Optional[str]:
    """Read a /proc file. None ⇒ the entry vanished (normal); else raises.

    A process disappearing mid-scan is the expected behaviour of a moving
    target. A permission error or an I/O error is NOT — it means we cannot see,
    and the caller must be able to tell those apart.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except ProcessLookupError:
        return None
    except OSError as exc:
        raise PreflightUnavailable(f"cannot read {path}: {exc}") from exc


def _read_proc_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except ProcessLookupError:
        return None
    except OSError as exc:
        raise PreflightUnavailable(f"cannot read {path}: {exc}") from exc


def _list_pids(proc: ProcSource) -> list:
    """Every numeric entry under the proc root, ascending."""
    try:
        with os.scandir(proc.root) as entries:
            pids = [int(e.name) for e in entries if e.name.isdigit()]
    except OSError as exc:
        raise PreflightUnavailable(f"cannot list {proc.root}: {exc}") from exc
    return sorted(pids)


def _parse_stat(text: str) -> Optional[dict]:
    """Parse /proc/<pid>/stat. Returns None if the line is malformed.

    `comm` is parenthesised and may itself contain spaces and parentheses, so
    the split is anchored on the LAST ')' — the standard parse. `comm` is also
    truncated to 15 characters by the kernel (TASK_COMM_LEN), which is why
    pattern matching below prefers argv[0] and treats comm as corroboration:
    `llama-perplexity` is 16 characters and never appears intact in comm.
    """
    close = text.rfind(")")
    open_paren = text.find("(")
    if close < 0 or open_paren < 0 or close < open_paren:
        return None
    comm = text[open_paren + 1:close]
    rest = text[close + 1:].split()
    if len(rest) < 20:
        return None
    try:
        return {
            "comm": comm,
            "state": rest[0],
            "ppid": int(rest[1]),
            # /proc(5) field 22, counting from 1 with `state` as field 3.
            "starttime_ticks": int(rest[19]),
        }
    except ValueError:
        return None


def _read_cmdline(proc: ProcSource, pid: int) -> Optional[list]:
    """argv as a list of str, or None if the process vanished.

    An empty list is meaningful and distinct from None: kernel threads and
    zombies have an empty cmdline.
    """
    raw = _read_proc_bytes(proc.pid_dir(pid) / "cmdline")
    if raw is None:
        return None
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _read_cgroup(proc: ProcSource, pid: int) -> Optional[str]:
    """The cgroup-v2 path of a pid ("0::/..."), or the first line, or None."""
    text = _read_proc_text(proc.pid_dir(pid) / "cgroup")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:]
    lines = text.splitlines()
    return lines[0] if lines else None


def _describe_pid(proc: ProcSource, pid: int) -> dict:
    """Identify a pid we already hold by IDENTITY, never by pattern.

    The distinction INC-20260731 turns on: reading `/proc/<pid>` for a pid
    obtained from a claim (or from the caller's own bookkeeping) is by-identity
    and has a blast radius of exactly one known process. Selecting pids by
    matching a name against every process on the box is the wildcard. This
    function is the by-identity direction and is used to answer "whose?" once a
    claim has already told us "which pid".
    """
    description: dict = {"pid": pid}
    try:
        stat_text = _read_proc_text(proc.pid_dir(pid) / "stat")
    except PreflightUnavailable as exc:
        description["unreadable"] = str(exc)
        return description
    if stat_text is None:
        description["vanished"] = True
        return description
    parsed = _parse_stat(stat_text)
    if parsed is not None:
        description["comm"] = parsed["comm"]
        description["ppid"] = parsed["ppid"]
        description["starttime_ticks"] = parsed["starttime_ticks"]
    try:
        argv = _read_cmdline(proc, pid)
        cgroup = _read_cgroup(proc, pid)
    except PreflightUnavailable as exc:
        description["unreadable"] = str(exc)
        return description
    if argv:
        description["argv0"] = argv[0]
        description["argv0_basename"] = os.path.basename(argv[0])
        description["cmdline"] = list(argv)
    if cgroup:
        description["cgroup"] = cgroup
    return description


def describe_pid(pid: int, proc: Optional[ProcSource] = None) -> dict:
    """Describe one already-identified PID without performing a pattern scan.

    This is the public, read-only identity lookup for callers that already got
    an exact PID from a claim or from :func:`interim_process_scan`.  It keeps
    targeted ``/proc`` reads in this signalling-incapable module instead of
    proliferating ad-hoc process readers.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError(f"pid must be a positive integer, got {pid!r}")
    return _describe_pid(proc or ProcSource(), pid)


# =============================================================================
# Owned scope — the PIDs and cgroup this process itself accounts for
# =============================================================================


@dataclass(frozen=True)
class OwnedScope:
    """Processes whose activity is OURS, so a claim they hold is not foreign.

    `incomplete` records why the set may be smaller than the truth. The
    asymmetry is deliberate and load-bearing: an incomplete ownership set can
    only ever cause a claim of ours to look foreign (a false FAIL, which fails
    closed), never a foreign claim to look like ours (a false PASS). So
    incompleteness is reported, but it does not by itself downgrade a PASS.
    """

    self_pid: int
    cgroup: Optional[str]
    pids: frozenset
    reasons: Mapping
    incomplete: tuple = ()

    def owns(self, pid: Optional[int]) -> bool:
        return pid is not None and pid in self.pids

    def to_dict(self) -> dict:
        return {
            "self_pid": self.self_pid,
            "cgroup": self.cgroup,
            "pids": sorted(self.pids),
            "reasons": {str(k): v for k, v in sorted(self.reasons.items())},
            "incomplete": list(self.incomplete),
        }


_MAX_ANCESTOR_DEPTH = 64


def read_own_scope(proc: Optional[ProcSource] = None) -> OwnedScope:
    """Enumerate self + ancestors + descendants, with our cgroup.

    ANCESTORS COUNT. The canonical bench runs under `region-lock run -- ...`,
    which acquires the region locks in the PARENT and execs the workload as a
    child; the holder of our own claim is therefore an ancestor, and a preflight
    that ignored ancestors would report its own wrapper as concurrent inference.
    PID 1 is excluded: it is everyone's ancestor, so treating it as ours would
    make an init-held claim structurally invisible.
    """
    proc = proc or ProcSource()
    self_pid = proc.self_pid
    if self_pid <= 1:
        # The mirror image of the PID-1 exclusion below, and the more dangerous
        # half. If WE are pid 1 (a container entrypoint in its own PID
        # namespace) then every process in the namespace is our descendant, so
        # the descendant walk would mark the entire machine as owned and no
        # claim could ever be foreign — a guaranteed PASS. Ownership is not
        # decidable from that vantage point, so say so.
        raise PreflightUnavailable(
            f"cannot separate owned from foreign processes while running as pid {self_pid}: "
            "every process in this PID namespace is a descendant of pid 1, so ownership "
            "would swallow every foreign claim"
        )
    incomplete: list = []
    reasons: dict = {self_pid: "self"}
    owned: set = {self_pid}

    # Ancestors: walk ppid upward. A malformed or unreadable link truncates the
    # chain and is recorded rather than assumed benign.
    current = self_pid
    for _ in range(_MAX_ANCESTOR_DEPTH):
        try:
            stat_text = _read_proc_text(proc.pid_dir(current) / "stat")
        except PreflightUnavailable as exc:
            incomplete.append(f"ancestor chain truncated at pid {current}: {exc}")
            break
        if stat_text is None:
            incomplete.append(f"ancestor chain truncated at pid {current}: entry vanished")
            break
        parsed = _parse_stat(stat_text)
        if parsed is None:
            incomplete.append(f"ancestor chain truncated at pid {current}: unparseable stat")
            break
        ppid = parsed["ppid"]
        if ppid <= 1:
            break
        if ppid in owned:
            incomplete.append(f"ancestor chain cycle at pid {ppid}")
            break
        owned.add(ppid)
        reasons[ppid] = "ancestor"
        current = ppid
    else:
        incomplete.append(f"ancestor chain exceeded {_MAX_ANCESTOR_DEPTH} levels")

    # Descendants: one pass over the pid table to build a child map, then BFS.
    children: dict = {}
    for pid in _list_pids(proc):
        try:
            stat_text = _read_proc_text(proc.pid_dir(pid) / "stat")
        except PreflightUnavailable as exc:
            incomplete.append(f"cannot read stat of pid {pid}: {exc}")
            continue
        if stat_text is None:
            continue
        parsed = _parse_stat(stat_text)
        if parsed is None:
            incomplete.append(f"unparseable stat for pid {pid}")
            continue
        children.setdefault(parsed["ppid"], []).append(pid)

    frontier = [self_pid]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, ()):
            if child in owned:
                continue
            owned.add(child)
            reasons[child] = "descendant"
            frontier.append(child)

    try:
        cgroup = _read_cgroup(proc, self_pid)
    except PreflightUnavailable as exc:
        cgroup = None
        incomplete.append(f"cannot read own cgroup: {exc}")

    return OwnedScope(
        self_pid=self_pid,
        cgroup=cgroup,
        pids=frozenset(owned),
        reasons=dict(reasons),
        incomplete=tuple(incomplete),
    )


# =============================================================================
# LAYER 1 (TARGET) — claim witness
# =============================================================================

# On-disk contract mirrored from the orchestrator's `cpu_region_lock`:
#   path    {tmp_dir}/cpu_region.{role}.{region}.lock
#   holder  an exclusive flock; THE FLOCK IS THE FACT
#   payload optional JSON attribution {schema_version, pid, role, region,
#           regions, instance_idx, request_tag, started_at}
# It is mirrored rather than imported: importing `src.runtime.cpu_region_lock`
# would make this module depend on a second repository's import graph (and on
# `sys.path` ordering across three repos — §2.5 item 12), for a read this small.
# The cost of mirroring is drift, and drift is contained by
# `require_nonempty_namespace` below: a wrong or renamed namespace reads as
# COULD_NOT_CHECK, never as "no claims".
_LOCK_GLOB = "cpu_region.*.*.lock"
_LOCK_PREFIX = "cpu_region."
_LOCK_SUFFIX = ".lock"
_GLOBAL_MUTEX_ROLE = "GLOBAL"
_KNOWN_PAYLOAD_SCHEMA_VERSIONS = frozenset({1})


def default_region_lock_dir(environ: Optional[Mapping] = None) -> Path:
    """Resolve the region-lock namespace the way the orchestrator does.

    Same precedence as `cpu_region_lock._tmp_dir()`: ORCHESTRATOR_TMP_DIR, then
    ORCHESTRATOR_PATHS_TMP_DIR, then the hard-coded /mnt/raid0/llm/tmp. If the
    two implementations ever disagree we would read an empty namespace, which is
    exactly the case `require_nonempty_namespace` refuses to call a PASS.
    """
    env = os.environ if environ is None else environ
    for name in ("ORCHESTRATOR_TMP_DIR", "ORCHESTRATOR_PATHS_TMP_DIR"):
        override = env.get(name)
        if override:
            return Path(override)
    return Path("/mnt/raid0/llm/tmp")


@dataclass(frozen=True)
class LockHolders:
    """Live holders and waiters of one lock file, from /proc/locks."""

    holder_pids: tuple = ()
    waiter_pids: tuple = ()
    # OFD locks (`OFDLCK`) report pid -1: a real holder that cannot be
    # attributed to a process. Counted separately so "held" and "held by
    # someone we can name" never get conflated.
    unattributed_holders: int = 0

    @property
    def held(self) -> bool:
        return bool(self.holder_pids) or self.unattributed_holders > 0


def _read_proc_locks(proc: ProcSource) -> dict:
    """Map (dev, inode) -> LockHolders from /proc/locks.

    Two parsing details matter:
      * A BLOCKED waiter is printed as `N: -> POSIX ...`, shifting every field
        by one. Misparsing it silently turns a waiter into a holder — i.e. an
        invented FAIL — so waiters are detected and tracked separately.
      * The 6th field is MAJOR:MINOR:INODE with the device in hex. Matching on
        the inode alone (as some callers do) can collide across filesystems, so
        the full triple is the key.
    """
    text = _read_proc_text(proc.root / "locks")
    if text is None:
        raise PreflightUnavailable(f"{proc.root / 'locks'} does not exist")
    holders: dict = {}
    waiters: dict = {}
    unattributed: dict = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        # `1: FLOCK  ADVISORY  WRITE <pid> <maj:min:ino> <start> <end>`
        #  0    1        2        3      4         5
        offset = 1 if parts[1] == "->" else 0
        if len(parts) < 6 + offset:
            continue
        pid_text = parts[4 + offset]
        dev_inode = parts[5 + offset]
        chunks = dev_inode.rsplit(":", 1)
        if len(chunks) != 2:
            continue
        dev_text, inode_text = chunks
        try:
            inode = int(inode_text)
        except ValueError:
            continue
        key = (dev_text.lower(), inode)
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if offset:
            waiters.setdefault(key, []).append(pid)
        elif pid < 0:
            unattributed[key] = unattributed.get(key, 0) + 1
        else:
            holders.setdefault(key, []).append(pid)
    keys = set(holders) | set(waiters) | set(unattributed)
    return {
        key: LockHolders(
            holder_pids=tuple(sorted(set(holders.get(key, ())))),
            waiter_pids=tuple(sorted(set(waiters.get(key, ())))),
            unattributed_holders=unattributed.get(key, 0),
        )
        for key in keys
    }


def _has_non_finite(obj: Any, depth: int = 0) -> bool:
    """True if `obj` contains a NaN/Infinity anywhere (any depth).

    `json.loads` accepts the non-standard `NaN`/`Infinity` literals, but
    `schemas.canonical_json` refuses them — correctly, since `nan != nan` makes
    a record unequal to its own round-trip. The attribution payload is arbitrary
    on-disk content written by another process, so without this screen a single
    `NaN` in a lock file makes the FAIL attestation UNSERIALISABLE: the run is
    correctly blocked and then the evidence of why cannot be journalled, which
    is §4 invariant 7 failing precisely when it matters.
    """
    if depth > 32:
        return True
    if isinstance(obj, float):
        return not math.isfinite(obj)
    if isinstance(obj, Mapping):
        return any(_has_non_finite(v, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_non_finite(v, depth + 1) for v in obj)
    return False


def _lock_key(path: Path) -> Optional[tuple]:
    """(dev-hex, inode) for a lock file, matching /proc/locks' encoding."""
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PreflightUnavailable(f"cannot stat {path}: {exc}") from exc
    return ("%02x:%02x" % (os.major(info.st_dev), os.minor(info.st_dev)), info.st_ino)


@dataclass(frozen=True)
class RegionClaim:
    """One CPU region lock file and what is (or is not) holding it."""

    role: str
    region: str
    lock_path: str
    holders: LockHolders
    payload: Optional[Mapping] = None
    payload_is_stale: bool = False
    notes: tuple = ()

    @property
    def held(self) -> bool:
        return self.holders.held

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "region": self.region,
            "lock_path": self.lock_path,
            "held": self.held,
            "holder_pids": list(self.holders.holder_pids),
            "waiter_pids": list(self.holders.waiter_pids),
            "unattributed_holders": self.holders.unattributed_holders,
            "payload": dict(self.payload) if isinstance(self.payload, Mapping) else None,
            "payload_is_stale": self.payload_is_stale,
            "notes": list(self.notes),
        }


def read_region_claims(
    lock_dir: Path,
    proc: Optional[ProcSource] = None,
    *,
    require_nonempty_namespace: bool = True,
) -> list:
    """Read every CPU region lock file and its live holders. Raises on doubt.

    Raises `PreflightUnavailable` if the namespace is missing, unreadable, or
    (by default) empty. An empty region-lock directory on this host means the
    path is wrong far more often than it means the fleet is idle, and reading it
    as "no claims" would manufacture a PASS out of a misconfiguration — the
    fail-open pattern that has poisoned stores here before.
    """
    proc = proc or ProcSource()
    lock_dir = Path(lock_dir)
    if not lock_dir.exists():
        raise PreflightUnavailable(f"region-lock namespace {lock_dir} does not exist")
    if not lock_dir.is_dir():
        raise PreflightUnavailable(f"region-lock namespace {lock_dir} is not a directory")
    try:
        lock_files = sorted(lock_dir.glob(_LOCK_GLOB))
    except OSError as exc:
        raise PreflightUnavailable(f"cannot list region-lock namespace {lock_dir}: {exc}") from exc
    if not lock_files and require_nonempty_namespace:
        raise PreflightUnavailable(
            f"region-lock namespace {lock_dir} contains no {_LOCK_GLOB} files; refusing to "
            "read an empty namespace as 'no claims' (pass require_nonempty_namespace=False "
            "to accept this deliberately)"
        )

    lock_table = _read_proc_locks(proc)
    claims: list = []
    unparsed: list = []
    for lock_file in lock_files:
        stem = lock_file.name[len(_LOCK_PREFIX):-len(_LOCK_SUFFIX)]
        role, _, region = stem.rpartition(".")
        if not role or not region:
            # Not our naming contract; refusing to guess which part is the
            # region is safer than attributing a claim to the wrong scope.
            unparsed.append(lock_file.name)
            continue
        key = _lock_key(lock_file)
        if key is None:
            unparsed.append(lock_file.name)
            continue
        holders = lock_table.get(key, LockHolders())
        notes: list = []
        payload: Optional[Mapping] = None
        raw = _read_proc_text(lock_file)
        if raw is not None and raw.strip():
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                notes.append("attribution payload is not valid JSON")
                loaded = None
            if isinstance(loaded, dict) and _has_non_finite(loaded):
                # Occupancy does not depend on attribution, so the flock still
                # counts; the payload is dropped rather than carried into an
                # attestation that could not then be written.
                notes.append(
                    "attribution payload contains a non-finite number (NaN/Infinity) or is "
                    "nested too deeply; dropped so the attestation stays canonical-JSON safe"
                )
                loaded = None
            if isinstance(loaded, dict):
                payload = loaded
                version = loaded.get("schema_version")
                if version not in _KNOWN_PAYLOAD_SCHEMA_VERSIONS:
                    # The flock still counts — attribution degrades, occupancy
                    # does not.
                    notes.append(f"unknown payload schema_version {version!r}")
            elif loaded is not None:
                notes.append("attribution payload is not an object")
        # A holder killed with SIGKILL never runs its cleanup, so its JSON
        # outlives its lock. The flock is the fact; a payload without a live
        # holder is debris, and reporting it as a claim would block every future
        # run on this host forever.
        stale = payload is not None and not holders.held
        if stale:
            notes.append("attribution payload present but no live holder (stale debris)")
        if (
            payload is not None
            and not stale
            and isinstance(payload.get("pid"), int)
            and holders.holder_pids
            and payload["pid"] not in holders.holder_pids
        ):
            # The flock and the JSON disagree about who is here. The flock wins
            # (it is the fact), but `_whose_from_claim` would otherwise quote a
            # `request_tag` written by a DIFFERENT process than the live holder,
            # putting a wrong attribution into a permanent evidence record.
            notes.append(
                f"attribution payload names pid {payload['pid']} but the live flock is held "
                f"by {list(holders.holder_pids)}; attribution is not trustworthy"
            )
        claims.append(
            RegionClaim(
                role=role,
                region=region,
                lock_path=str(lock_file),
                holders=holders,
                payload=payload,
                payload_is_stale=stale,
                notes=tuple(notes),
            )
        )
    if lock_files and not claims:
        # Every file matched the glob and NONE of them yielded a claim. That is
        # not "no claims" — it is a namespace whose NAME SHAPE we no longer
        # understand (contract drift), and returning [] here would manufacture a
        # PASS out of it. `require_nonempty_namespace` only catches a namespace
        # that is empty; this catches one that is unreadable in the other sense.
        raise PreflightUnavailable(
            f"region-lock namespace {lock_dir} has {len(lock_files)} matching file(s) but none "
            f"parse as cpu_region.<role>.<region>.lock ({unparsed[:5]}); refusing to read a "
            "namespace whose naming contract has drifted as 'no claims'"
        )
    return claims


@dataclass(frozen=True)
class GpuClaimWitness:
    """One held GPU device claim, as reported by the device-claim substrate.

    Every string field is validated. A `ClaimReceipt` from `device_claim.py`
    carries `holder_label: Optional[str]`, so the obvious bridge —
    `GpuClaimWitness(holder_label=receipt.holder_label, ...)` — quietly produced
    a witness whose label was `None`, and `claim_witness_preflight` renders that
    straight into a FAIL finding's `whose` as the literal string
    `"None (pid 8800, via ...)"`. `PreflightResult` already refuses a FAIL that
    carries no finding because an unactionable FAIL is indistinguishable from a
    bug in this module; a finding that cannot name whose is the same failure one
    level down. Build witnesses with `resource.claim_witness`, which supplies a
    real label.
    """

    device_id: str
    holder_pid: Optional[int]
    holder_label: str
    source: str
    acquired_at: Optional[str] = None

    def __post_init__(self) -> None:
        # Written out field by field rather than looped over names: a loop needs
        # `getattr`, and `getattr` is on this module's own forbidden-name list
        # (it is a dynamic-attribute escape hatch, and the AST audit that proves
        # this module cannot signal does not exempt the code that validates a
        # dataclass). The audit caught this the first time it was written.
        for name, value in (
            ("device_id", self.device_id),
            ("holder_label", self.holder_label),
            ("source", self.source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"GpuClaimWitness.{name} must be a non-empty string, got "
                    f"{value!r}; a witness that cannot say what is claimed or "
                    "whose it is produces an unactionable finding"
                )
        if self.holder_pid is not None:
            if isinstance(self.holder_pid, bool) or not isinstance(self.holder_pid, int):
                raise TypeError(
                    f"GpuClaimWitness.holder_pid must be an int or None, got "
                    f"{type(self.holder_pid).__name__}"
                )
        if self.acquired_at is not None and not isinstance(self.acquired_at, str):
            raise TypeError("GpuClaimWitness.acquired_at must be a string or None")

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "holder_pid": self.holder_pid,
            "holder_label": self.holder_label,
            "source": self.source,
            "acquired_at": self.acquired_at,
        }


# A reader returns the currently-held device claims, or RAISES
# PreflightUnavailable. It must never return [] to mean "I could not look":
# reporting an unclaimed GPU as free would be a fabricated P-GPU-1 precondition.
#
# AK2 built the substrate (`resource/device_claim.py`), and
# `resource/claim_witness.device_claim_witness_reader()` is the conforming
# reader over it. Supplying no reader remains COULD_NOT_CHECK — not because the
# claim plane is missing, but because a GPU preflight that inspected no claim
# plane learned nothing about the device.
GpuClaimReader = Callable[[], Iterable]


@dataclass(frozen=True)
class ClaimSources:
    """Where claim witness is read from."""

    region_lock_dir: Path
    gpu_claim_reader: Optional[GpuClaimReader] = None
    # default_factory, NOT `ProcSource()`: a bare instance here is built once at
    # class-creation time, so its `self_pid` would be frozen to whichever pid
    # imported this module. After any fork in the host program the default would
    # describe the PARENT, whose owned set is a SUPERSET of the child's — the one
    # direction that turns a foreign claim into "ours", i.e. a false PASS.
    proc: ProcSource = field(default_factory=ProcSource)
    require_nonempty_namespace: bool = True

    @classmethod
    def from_environment(cls, environ: Optional[Mapping] = None, **kwargs: Any) -> "ClaimSources":
        return cls(region_lock_dir=default_region_lock_dir(environ), **kwargs)


# =============================================================================
# LAYER 2 (INTERIM) — read-only name-pattern enumerator
# =============================================================================


class MatchField(str, Enum):
    """Which text a pattern is matched against.

    EXE_BASENAME is the default because INC-20260731's blast radius came from
    matching a whole command line: `earlyoom --ignore ^(llama-server|sd-server)$`
    matches "llama-server" in full-cmdline mode and is not remotely an inference
    process. Full-cmdline matching remains available, opt-in, and every match it
    makes outside argv[0] is classified as a MENTION rather than as inference.
    """

    EXE_BASENAME = "EXE_BASENAME"
    FULL_CMDLINE = "FULL_CMDLINE"


class Classification(str, Enum):
    INFERENCE_LIKE = "INFERENCE_LIKE"
    OWNED = "OWNED"
    GUARD_ARGV_ONLY = "GUARD_ARGV_ONLY"
    ARGV_MENTION_ONLY = "ARGV_MENTION_ONLY"


# How much a classification counts AGAINST the host being quiet. Used to pick
# the strongest classification across ALL patterns for one process, never the
# first one that happens to match: stopping at the first match makes the pattern
# TUPLE ORDER decide whether a process counts as inference, and a mere mention of
# an earlier pattern would then mask a later pattern matching argv[0] itself.
_CLASSIFICATION_STRENGTH = {
    Classification.ARGV_MENTION_ONLY: 0,
    Classification.GUARD_ARGV_ONLY: 1,
    Classification.INFERENCE_LIKE: 2,
}


# The names the protocols actually ask about: bench-cpu.md's `pgrep llama`
# zombie check and gpu-cross-device.md's llama-server / AutoPilot / KFD checks.
DEFAULT_INFERENCE_EXE_PATTERNS = (
    "llama-server",
    "llama-bench",
    "llama-cli",
    "llama-perplexity",
    "whisper-server",
    "whisper-cli",
    "qwentts-server",
    "sd-server",
)

# argv flags whose VALUE is a list of process names to protect or avoid. A
# pattern found only inside one of these is a guard's exclusion list, not an
# inference process — this is the earlyoom half of INC-20260731 encoded so the
# false positive cannot recur.
_GUARD_ARGV_FLAGS = ("--ignore", "--ignore-regex", "--prefer", "--avoid", "--exclude", "-N")

# Executables that are known guards. Their argv necessarily contains the names
# they guard, so they are classified by identity, never by their arguments.
_KNOWN_GUARD_EXECUTABLES = ("earlyoom", "oomd", "systemd-oomd", "nohang")


@dataclass(frozen=True)
class ProcessObservation:
    """One process the enumerator matched. Reported, never signalled."""

    pid: int
    classification: Classification
    matched_pattern: str
    matched_field: MatchField
    comm: Optional[str] = None
    argv0_basename: Optional[str] = None
    cmdline: tuple = ()
    starttime_ticks: Optional[int] = None
    cgroup: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "classification": self.classification.value,
            "matched_pattern": self.matched_pattern,
            "matched_field": self.matched_field.value,
            "comm": self.comm,
            "argv0_basename": self.argv0_basename,
            "cmdline": list(self.cmdline),
            "starttime_ticks": self.starttime_ticks,
            "cgroup": self.cgroup,
        }


@dataclass(frozen=True)
class ProcessScan:
    """The result of one read-only enumeration pass. Carries no action."""

    observations: tuple
    patterns: tuple
    match_field: MatchField
    scanned_pids: int
    vanished_pids: tuple = ()
    unreadable_pids: Mapping = field(default_factory=dict)

    def inference_like(self) -> tuple:
        return tuple(o for o in self.observations
                     if o.classification is Classification.INFERENCE_LIKE)

    def to_dict(self) -> dict:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "patterns": list(self.patterns),
            "match_field": self.match_field.value,
            "scanned_pids": self.scanned_pids,
            "vanished_pids": list(self.vanished_pids),
            "unreadable_pids": {str(k): v for k, v in sorted(self.unreadable_pids.items())},
        }


def _classify(
    argv: list,
    comm: Optional[str],
    pattern: str,
    match_field: MatchField,
) -> Optional[Classification]:
    """Return how `pattern` matched, or None if it did not match at all."""
    argv0_base = os.path.basename(argv[0]) if argv else None
    if argv0_base and pattern in argv0_base:
        return Classification.INFERENCE_LIKE
    # comm is the fallback identity for a process with no readable argv (a
    # zombie, or a kernel thread). It is truncated to 15 chars, so it can only
    # ever confirm a match, never refute one.
    if not argv and comm and pattern in comm:
        return Classification.INFERENCE_LIKE
    if match_field is not MatchField.FULL_CMDLINE:
        return None
    if argv0_base and os.path.basename(argv[0]) in _KNOWN_GUARD_EXECUTABLES:
        matched_elsewhere = any(pattern in token for token in argv[1:])
        return Classification.GUARD_ARGV_ONLY if matched_elsewhere else None
    for index, token in enumerate(argv[1:], start=1):
        if pattern not in token:
            continue
        previous = argv[index - 1]
        if previous in _GUARD_ARGV_FLAGS or any(
            previous.startswith(flag + "=") for flag in _GUARD_ARGV_FLAGS
        ):
            return Classification.GUARD_ARGV_ONLY
        if any(token.startswith(flag + "=") for flag in _GUARD_ARGV_FLAGS):
            return Classification.GUARD_ARGV_ONLY
        return Classification.ARGV_MENTION_ONLY
    return None


def interim_process_scan(
    patterns: Iterable = DEFAULT_INFERENCE_EXE_PATTERNS,
    *,
    proc: Optional[ProcSource] = None,
    owned: Optional[OwnedScope] = None,
    match_field: MatchField = MatchField.EXE_BASENAME,
) -> ProcessScan:
    """Enumerate processes whose executable matches a pattern. NEVER signals.

    This is §3.5's interim instrument and the ONLY sanctioned name-pattern
    process reader in the codebase. It reads `/proc`; it holds no lock, takes no
    action, and has no code path that could deliver a signal — see
    `audit_no_signalling_capability()`.

    A pid that vanishes mid-pass is recorded and skipped: this instrument is a
    sample of a moving target, which is precisely why the claim-witness layer is
    preferred. A pid that is present but UNREADABLE is recorded separately,
    because it downgrades the scan's verdict to COULD_NOT_CHECK — an
    unenumerable process is not an absent one.
    """
    proc = proc or ProcSource()
    patterns = tuple(patterns)
    if not patterns:
        raise ValueError("interim_process_scan() needs at least one pattern")
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"invalid pattern: {pattern!r}")

    pids = _list_pids(proc)
    observations: list = []
    vanished: list = []
    unreadable: dict = {}
    for pid in pids:
        try:
            stat_text = _read_proc_text(proc.pid_dir(pid) / "stat")
            if stat_text is None:
                vanished.append(pid)
                continue
            argv = _read_cmdline(proc, pid)
            if argv is None:
                vanished.append(pid)
                continue
            cgroup = _read_cgroup(proc, pid)
        except PreflightUnavailable as exc:
            unreadable[pid] = str(exc)
            continue
        parsed = _parse_stat(stat_text)
        comm = parsed["comm"] if parsed else None
        # STRONGEST match across every pattern, not the first one to hit. With
        # first-match-wins, `sd-server --lora-dir /models/llama-cli/x` is
        # classified ARGV_MENTION_ONLY (because "llama-cli" precedes "sd-server"
        # in the default tuple) and a live inference server reads as PASS.
        best: Optional[Classification] = None
        best_pattern: Optional[str] = None
        for pattern in patterns:
            classification = _classify(argv, comm, pattern, match_field)
            if classification is None:
                continue
            if best is None or (
                _CLASSIFICATION_STRENGTH[classification] > _CLASSIFICATION_STRENGTH[best]
            ):
                best, best_pattern = classification, pattern
            if best is Classification.INFERENCE_LIKE:
                break
        if best is not None:
            if owned is not None and owned.owns(pid):
                best = Classification.OWNED
            observations.append(
                ProcessObservation(
                    pid=pid,
                    classification=best,
                    matched_pattern=best_pattern,
                    matched_field=match_field,
                    comm=comm,
                    argv0_basename=os.path.basename(argv[0]) if argv else None,
                    cmdline=tuple(argv),
                    starttime_ticks=parsed["starttime_ticks"] if parsed else None,
                    cgroup=cgroup,
                )
            )
    return ProcessScan(
        observations=tuple(observations),
        patterns=patterns,
        match_field=match_field,
        scanned_pids=len(pids),
        vanished_pids=tuple(vanished),
        unreadable_pids=dict(unreadable),
    )


# =============================================================================
# Result
# =============================================================================


@dataclass(frozen=True)
class Finding:
    """One concrete reason a preflight failed: WHAT is running, and WHOSE."""

    kind: str
    what: str
    whose: str
    detail: Mapping = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "what": self.what,
            "whose": self.whose,
            "detail": dict(self.detail),
        }


class InterimScan(str, Enum):
    """Whether the weaker §3.5 instrument may decide this preflight.

    DENY is the default. Sliding from claim witness to a name-pattern scan
    without the call site saying so is the silent-degradation pattern this
    project keeps paying for, and the two instruments do not measure the same
    thing: a resident-but-idle production `llama-server` is not concurrent
    inference, and only the claim plane can tell the difference.
    """

    DENY = "DENY"
    ALLOW_LABELLED = "ALLOW_LABELLED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class PreflightResult:
    """A three-outcome preflight verdict with the evidence that produced it.

    Designed to be journaled verbatim as the P-BENCH-1 / P-GPU-1 precondition
    attestation of the run it gated: `to_dict()` is canonical-JSON safe, and it
    carries `basis`, so a reader can always tell whether the precondition was
    established by claim witness or by the weaker interim scan.
    """

    verdict: str
    basis: str
    scope: PreflightScope
    observed_at: str
    reasons: tuple = ()
    findings: tuple = ()
    notes: tuple = ()
    owned: Optional[OwnedScope] = None
    region_claims: tuple = ()
    gpu_claims: tuple = ()
    scan: Optional[ProcessScan] = None

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICT_SEVERITY:
            raise ValueError(f"invalid preflight verdict: {self.verdict!r}")
        if self.basis not in BASES:
            raise ValueError(f"invalid preflight basis: {self.basis!r}")
        if self.verdict == FAIL and not self.findings:
            # A FAIL that cannot say what and whose is unactionable and
            # indistinguishable from a bug in this module.
            raise ValueError("a FAIL preflight must carry at least one finding")
        if self.verdict == COULD_NOT_CHECK and not self.reasons:
            raise ValueError("a COULD_NOT_CHECK preflight must say why it could not check")

    def __bool__(self) -> bool:
        """Truth-testing is a TypeError, deliberately.

        `if preflight(...):` would read COULD_NOT_CHECK as falsy or PASS-ish
        depending on how the caller wrote the condition, and that single line is
        the entire failure mode this module exists to prevent. Making it raise
        turns a silent misreading into a stack trace.
        """
        raise TypeError(
            "PreflightResult has THREE outcomes and must not be truth-tested; "
            "COULD_NOT_CHECK is not a pass. Use .passed, .verdict, or .require_pass()."
        )

    @property
    def passed(self) -> bool:
        """True only for PASS. COULD_NOT_CHECK is falsy here on purpose."""
        return self.verdict == PASS

    @property
    def could_not_check(self) -> bool:
        return self.verdict == COULD_NOT_CHECK

    def require_pass(self) -> "PreflightResult":
        """Return self on PASS; raise a DISTINCT exception per failing outcome."""
        if self.verdict == PASS:
            return self
        summary = "; ".join(self.reasons) or "no reason recorded"
        if self.verdict == FAIL:
            detail = "; ".join(f"{f.what} ({f.whose})" for f in self.findings)
            raise ConcurrentInferenceDetected(
                f"concurrent inference detected for scope {self.scope.label!r}: {detail}",
                self,
            )
        raise PreflightIndeterminate(
            f"inference preflight for scope {self.scope.label!r} could not be established: "
            f"{summary}",
            self,
        )

    def as_check(self) -> Any:
        """The same verdict as a `schemas.Check`, for composing with §7 checkers."""
        return Check(self.verdict, tuple(self.reasons) or tuple(
            f"{f.what} ({f.whose})" for f in self.findings))

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "basis": self.basis,
            "scope": self.scope.to_dict(),
            "observed_at": self.observed_at,
            "reasons": list(self.reasons),
            "findings": [f.to_dict() for f in self.findings],
            "notes": list(self.notes),
            "owned": self.owned.to_dict() if self.owned is not None else None,
            "region_claims": [c.to_dict() for c in self.region_claims],
            "gpu_claims": [g.to_dict() for g in self.gpu_claims],
            "scan": self.scan.to_dict() if self.scan is not None else None,
        }


# =============================================================================
# The checkers
# =============================================================================


def claim_witness_preflight(
    scope: PreflightScope,
    sources: ClaimSources,
    *,
    owned: Optional[OwnedScope] = None,
    now: Callable[[], str] = _utc_now,
) -> PreflightResult:
    """§3.5 TARGET: decide from who holds the claims, not from who is running.

    PASS  — no claim in scope is held by anyone but us.
    FAIL  — a claim in scope is held by someone else (reported with what/whose).
    COULD_NOT_CHECK — the claim plane could not be read, or the scope includes a
            GPU device and no device-claim reader was supplied (§2.5: that
            substrate does not exist yet, so its silence means nothing).
    """
    reasons: list = []
    notes: list = []
    findings: list = []
    verdicts: list = []
    claims: tuple = ()
    gpu_claims: tuple = ()

    if owned is None:
        try:
            owned = read_own_scope(sources.proc)
        except PreflightUnavailable as exc:
            # Without ownership we cannot separate our own wrapper's claim from
            # a foreign one, so nothing downstream is decidable.
            return PreflightResult(
                verdict=COULD_NOT_CHECK,
                basis=BASIS_CLAIM_WITNESS,
                scope=scope,
                observed_at=now(),
                reasons=(f"cannot enumerate own process scope: {exc}",),
            )
    notes.extend(f"owned-scope enumeration incomplete: {r}" for r in owned.incomplete)

    if scope.covers_cpu:
        try:
            claims = tuple(
                read_region_claims(
                    sources.region_lock_dir,
                    sources.proc,
                    require_nonempty_namespace=sources.require_nonempty_namespace,
                )
            )
        except PreflightUnavailable as exc:
            verdicts.append(COULD_NOT_CHECK)
            reasons.append(f"CPU region claim witness unavailable: {exc}")
        else:
            cpu_verdict = PASS
            for claim in claims:
                notes.extend(f"{claim.role}.{claim.region}: {n}" for n in claim.notes)
                if not scope.covers_region(claim.region):
                    continue
                if not claim.held:
                    continue
                if claim.role == _GLOBAL_MUTEX_ROLE:
                    what = f"cross-role GLOBAL mutex on region {claim.region}"
                else:
                    what = f"CPU region {claim.region} claimed by role {claim.role!r}"
                foreign = [pid for pid in claim.holders.holder_pids if not owned.owns(pid)]
                if not foreign and not claim.holders.unattributed_holders:
                    notes.append(f"{claim.role}.{claim.region} is held by our own scope")
                    continue
                cpu_verdict = FAIL
                for pid in foreign:
                    findings.append(
                        Finding(
                            kind="cpu_region_claim",
                            what=what,
                            whose=_whose_from_claim(claim, pid, sources.proc),
                            detail={
                                "claim": claim.to_dict(),
                                "holder": _describe_pid(sources.proc, pid),
                            },
                        )
                    )
                if claim.holders.unattributed_holders:
                    # An OFD lock has no owning pid. It is unambiguously held —
                    # occupancy is a fact — but "whose" is not establishable, and
                    # guessing would be worse than saying so.
                    findings.append(
                        Finding(
                            kind="cpu_region_claim",
                            what=what,
                            whose="UNATTRIBUTED (open-file-description lock, /proc/locks pid -1)",
                            detail={"claim": claim.to_dict()},
                        )
                    )
            verdicts.append(cpu_verdict)
            if cpu_verdict == FAIL and owned.incomplete:
                reasons.append(
                    "note: owned-scope enumeration was incomplete, so a claim reported as "
                    "foreign could in principle be ours; the preflight fails closed"
                )

    if scope.gpu_devices:
        if sources.gpu_claim_reader is None:
            verdicts.append(COULD_NOT_CHECK)
            reasons.append(
                "no GPU device-claim reader supplied, so nothing inspected the device claim "
                "plane and GPU exclusivity cannot be witnessed; pass "
                "resource.claim_witness.device_claim_witness_reader(devices) (region-lock is "
                "CPU-only and src/gpu_lease.py is a process-local lease, so neither answers this)"
            )
        else:
            try:
                gpu_claims = tuple(sources.gpu_claim_reader())
            except PreflightUnavailable as exc:
                verdicts.append(COULD_NOT_CHECK)
                reasons.append(f"GPU device claim witness unavailable: {exc}")
            except Exception as exc:  # noqa: BLE001 - a broken reader is a blind spot
                verdicts.append(COULD_NOT_CHECK)
                reasons.append(f"GPU device claim reader raised {type(exc).__name__}: {exc}")
            else:
                gpu_verdict = PASS
                for gpu_claim in gpu_claims:
                    if gpu_claim.device_id not in scope.gpu_devices:
                        continue
                    if owned.owns(gpu_claim.holder_pid):
                        notes.append(f"GPU device {gpu_claim.device_id} is claimed by our scope")
                        continue
                    gpu_verdict = FAIL
                    findings.append(
                        Finding(
                            kind="gpu_device_claim",
                            what=f"GPU device {gpu_claim.device_id} is claimed",
                            whose=(
                                f"{gpu_claim.holder_label} "
                                f"(pid {gpu_claim.holder_pid}, via {gpu_claim.source})"
                            ),
                            detail={"claim": gpu_claim.to_dict()},
                        )
                    )
                verdicts.append(gpu_verdict)

    if not verdicts:
        # Unreachable for a valid scope (the constructor forbids an empty one),
        # but an empty verdict list must never silently become PASS.
        verdicts.append(COULD_NOT_CHECK)
        reasons.append("scope selected no claim class to check")

    return PreflightResult(
        verdict=combine_verdicts(*verdicts),
        basis=BASIS_CLAIM_WITNESS,
        scope=scope,
        observed_at=now(),
        reasons=tuple(reasons),
        findings=tuple(findings),
        notes=tuple(notes),
        owned=owned,
        region_claims=claims,
        gpu_claims=gpu_claims,
    )


def _whose_from_claim(claim: RegionClaim, pid: int, proc: ProcSource) -> str:
    """Human-readable attribution for a foreign region-lock holder."""
    payload = claim.payload if isinstance(claim.payload, Mapping) else {}
    tag = payload.get("request_tag") if not claim.payload_is_stale else None
    described = _describe_pid(proc, pid)
    comm = described.get("argv0_basename") or described.get("comm") or "unknown"
    parts = [f"pid {pid}", f"exe {comm}", f"role {claim.role!r}"]
    if tag:
        parts.append(f"tag {tag!r}")
    cgroup = described.get("cgroup")
    if cgroup:
        parts.append(f"cgroup {cgroup}")
    return ", ".join(parts)


def interim_scan_preflight(
    scope: PreflightScope,
    *,
    proc: Optional[ProcSource] = None,
    owned: Optional[OwnedScope] = None,
    patterns: Iterable = DEFAULT_INFERENCE_EXE_PATTERNS,
    match_field: MatchField = MatchField.EXE_BASENAME,
    now: Callable[[], str] = _utc_now,
    preferred_layer_reasons: Iterable = (),
) -> PreflightResult:
    """§3.5 INTERIM: decide from a read-only name-pattern enumeration.

    Explicitly the weaker instrument, and labelled as such in the result's
    `basis`. It cannot distinguish a resident-but-idle production server from an
    active decode, so its PASS is a weaker statement than a claim-witness PASS
    and its FAIL may be a false positive on a loaded-but-quiet server. It exists
    because §3.5 sanctions it as the interim substitute for the protocols'
    mandated `pgrep`, and it never signals.
    """
    proc = proc or ProcSource()
    reasons = list(preferred_layer_reasons)
    notes = ["verdict established by the INTERIM name-pattern enumerator, not by claim witness"]
    if owned is None:
        try:
            owned = read_own_scope(proc)
        except PreflightUnavailable as exc:
            reasons.append(f"cannot enumerate own process scope: {exc}")
            return PreflightResult(
                verdict=COULD_NOT_CHECK,
                basis=BASIS_INTERIM_PROCESS_SCAN,
                scope=scope,
                observed_at=now(),
                reasons=tuple(reasons),
                notes=tuple(notes),
            )
    try:
        scan = interim_process_scan(
            patterns, proc=proc, owned=owned, match_field=match_field
        )
    except PreflightUnavailable as exc:
        reasons.append(f"process enumeration unavailable: {exc}")
        return PreflightResult(
            verdict=COULD_NOT_CHECK,
            basis=BASIS_INTERIM_PROCESS_SCAN,
            scope=scope,
            observed_at=now(),
            reasons=tuple(reasons),
            notes=tuple(notes),
            owned=owned,
        )

    findings = tuple(
        Finding(
            kind="process",
            what=f"{observation.argv0_basename or observation.comm} matched "
                 f"{observation.matched_pattern!r}",
            whose=f"pid {observation.pid}, cgroup {observation.cgroup or 'unknown'}",
            detail=observation.to_dict(),
        )
        for observation in scan.inference_like()
    )
    verdicts = [FAIL if findings else PASS]
    if scope.gpu_devices:
        # A name-pattern scan of /proc witnesses NOTHING about GPU device
        # occupancy: it never opens /dev/kfd, never reads a device claim, and
        # cannot tell a resident server from one that is decoding on the device.
        # Without this, `interim_scan=ALLOW_LABELLED` turns the documented
        # "GPU exclusivity is never fabricated" rule into a fabricated P-GPU-1
        # precondition — a PASS for a device nothing looked at.
        verdicts.append(COULD_NOT_CHECK)
        reasons.append(
            "scope covers GPU device(s) "
            f"{sorted(scope.gpu_devices)}, which a name-pattern process scan cannot witness; "
            "GPU exclusivity needs the device-claim reader (§2.5/AK2)"
        )
    if scan.unreadable_pids:
        verdicts.append(COULD_NOT_CHECK)
        reasons.append(
            f"{len(scan.unreadable_pids)} process(es) could not be read, so the "
            "enumeration is not exhaustive"
        )
    notes.extend(
        f"{o.classification.value}: pid {o.pid} matched {o.matched_pattern!r} but is not "
        "counted as inference"
        for o in scan.observations
        if o.classification is not Classification.INFERENCE_LIKE
    )
    return PreflightResult(
        verdict=combine_verdicts(*verdicts),
        basis=BASIS_INTERIM_PROCESS_SCAN,
        scope=scope,
        observed_at=now(),
        reasons=tuple(reasons),
        findings=findings,
        notes=tuple(notes),
        owned=owned,
        scan=scan,
    )


def preflight(
    scope: PreflightScope,
    sources: ClaimSources,
    *,
    interim_scan: InterimScan = InterimScan.DENY,
    patterns: Iterable = DEFAULT_INFERENCE_EXE_PATTERNS,
    match_field: MatchField = MatchField.EXE_BASENAME,
    now: Callable[[], str] = _utc_now,
    scanner: Callable[..., PreflightResult] = interim_scan_preflight,
) -> PreflightResult:
    """Run the preflight, preferring claim witness and labelling any fallback.

    The claim-witness layer decides whenever it can evaluate at all — a PASS or
    a FAIL from it is returned untouched and the scan is never even run. The
    interim scan is consulted ONLY when claim witness returns COULD_NOT_CHECK,
    and only when the call site has explicitly allowed it. Corroborating one
    layer with the other is deliberately not offered: see the module notes.
    """
    if not isinstance(interim_scan, InterimScan):
        raise TypeError("interim_scan must be an InterimScan member")
    result = claim_witness_preflight(scope, sources, now=now)
    if result.verdict != COULD_NOT_CHECK:
        return result
    if interim_scan is InterimScan.DENY:
        return PreflightResult(
            verdict=COULD_NOT_CHECK,
            basis=result.basis,
            scope=result.scope,
            observed_at=result.observed_at,
            reasons=result.reasons,
            findings=result.findings,
            notes=result.notes + (
                "the INTERIM name-pattern enumerator was NOT consulted "
                "(interim_scan=DENY); pass InterimScan.ALLOW_LABELLED to accept the "
                "weaker §3.5 instrument deliberately",
            ),
            owned=result.owned,
            region_claims=result.region_claims,
            gpu_claims=result.gpu_claims,
        )
    return scanner(
        scope,
        proc=sources.proc,
        owned=result.owned,
        patterns=patterns,
        match_field=match_field,
        now=now,
        preferred_layer_reasons=result.reasons,
    )


def require_no_concurrent_inference(
    scope: PreflightScope,
    sources: ClaimSources,
    **kwargs: Any,
) -> PreflightResult:
    """The recommended call site: run the preflight and refuse anything but PASS.

    Returns the result so the caller can journal it as the precondition
    attestation — a precondition that was checked but not recorded is
    indistinguishable from one that was skipped. On FAIL/COULD_NOT_CHECK the
    attestation is on the raised exception as `exc.result`, and it must be
    journalled there too (§4 invariant 7: failures are durable).

    A PASS IS AN OBSERVATION, NOT A CLAIM. §4 invariant 9 — *"resources are
    acquired, not observed ... idle sensing is never a claim"* — means this
    function is a precondition to acquisition, never a substitute for it:
    nothing stops another process from taking the region lock in the interval
    between this PASS and your `cpu_region_lock(...)`. The correct sequence is
    preflight → acquire the claim → and only then run.
    """
    return preflight(scope, sources, **kwargs).require_pass()


# =============================================================================
# Structural no-signalling audit
# =============================================================================

# Import roots that can deliver a signal or spawn something that can.
_FORBIDDEN_IMPORT_ROOTS = frozenset({
    "signal", "subprocess", "multiprocessing", "ctypes", "psutil", "pty", "sh",
    "paramiko", "fabric", "asyncio",
    # Indirection roots. `importlib.import_module("os")` and
    # `operator.attrgetter("kill")` reach exactly what `__import__`/`getattr`
    # reach, so banning only the builtin spellings bans only the obvious spelling.
    "importlib", "runpy", "operator", "gc", "pickle",
})

# Identifiers that either deliver a signal, replace/duplicate this process, or
# open a dynamic escape hatch through which the first two could be reached
# without appearing in the AST as themselves.
_FORBIDDEN_NAMES = frozenset({
    "kill", "killpg", "raise_signal", "pthread_kill", "send_signal", "terminate",
    "system", "popen", "spawnl", "spawnle", "spawnv", "spawnve", "spawnvp",
    "execl", "execle", "execlp", "execv", "execve", "execvp", "execvpe",
    "fork", "forkpty", "abort", "posix_spawn", "posix_spawnp",
    "getattr", "setattr", "delattr", "__import__", "eval", "exec", "compile",
    # Every other way to reach an attribute by a computed name. Banning only
    # `getattr` leaves `vars(os)["kill"]`, `os.__dict__["kill"]`,
    # `globals()["k"]` and `operator.attrgetter("kill")(os)` all auditing clean,
    # which makes the "no dynamic escape hatch" claim narrower than it reads.
    "vars", "globals", "locals", "__dict__", "__builtins__", "__getattribute__",
    "import_module", "attrgetter", "methodcaller",
    "SIGKILL", "SIGTERM", "SIGINT", "SIGSTOP", "SIGHUP", "SIGQUIT",
})

SIGNALLING_AUDIT_TARGET = Path(__file__).resolve()


def audit_no_signalling_capability(source_path: Optional[Path] = None) -> Any:
    """Prove from the AST that this module cannot deliver a signal.

    PASS — no forbidden import, attribute, or name appears in executable code.
    FAIL — one does; the violations name the line.
    COULD_NOT_CHECK — the source could not be read or parsed, so the absence
            could not be established. Absence of evidence is not evidence of
            absence, and a self-audit that returns PASS when it cannot see its
            own source would be worse than no audit.

    Comments and string literals are invisible to this audit because it works on
    the parsed tree, which is why the module docstring can discuss `pkill` and
    signalling freely without weakening the guarantee. The audit proves the
    module contains no signal-delivery call and imports nothing that can make
    one; it is not a sandbox.
    """
    path = Path(source_path) if source_path is not None else SIGNALLING_AUDIT_TARGET
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return Check(COULD_NOT_CHECK, (f"cannot read {path}: {exc}",))
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return Check(COULD_NOT_CHECK, (f"cannot parse {path}: {exc}",))

    violations: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    violations.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_IMPORT_ROOTS:
                violations.append(f"line {node.lineno}: imports from {node.module}")
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    violations.append(f"line {node.lineno}: imports name {alias.name}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_NAMES:
                violations.append(f"line {node.lineno}: attribute access .{node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                violations.append(f"line {node.lineno}: reference to {node.id}")
    if violations:
        return Check(FAIL, tuple(violations))
    return Check(PASS)


__all__ = [
    "PASS", "FAIL", "COULD_NOT_CHECK", "Check", "combine_verdicts",
    "BASIS_CLAIM_WITNESS", "BASIS_INTERIM_PROCESS_SCAN", "BASIS_NONE", "BASES",
    "PreflightUnavailable", "PreflightNotSatisfied", "ConcurrentInferenceDetected",
    "PreflightIndeterminate",
    "PreflightScope", "ProcSource", "OwnedScope", "read_own_scope", "describe_pid",
    "ClaimSources", "GpuClaimWitness", "GpuClaimReader", "RegionClaim", "LockHolders",
    "read_region_claims", "default_region_lock_dir",
    "MatchField", "Classification", "ProcessObservation", "ProcessScan",
    "DEFAULT_INFERENCE_EXE_PATTERNS", "interim_process_scan",
    "Finding", "InterimScan", "PreflightResult",
    "claim_witness_preflight", "interim_scan_preflight", "preflight",
    "require_no_concurrent_inference",
    "audit_no_signalling_capability", "SIGNALLING_AUDIT_TARGET",
]
