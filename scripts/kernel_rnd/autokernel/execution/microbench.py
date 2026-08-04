"""The T1 paired-block microbench runner — the thing that produces a speed reading.

WHAT THIS IS
------------
`evaluator/recipes.py` CONSTRUCTS the argv and refuses to run it.
`evaluator/statistics.py` REDUCES paired blocks and refuses to produce them.
This module is the missing middle: it takes a recipe id, a candidate binding and
a named immutable anchor, runs alternating candidate/anchor invocations under a
held resource claim, parses what came back, and hands `statistics.PairedBlock`
objects to the reducer. It is the first module in this package that consumes
inference.

Its authority and its limits are P-AK-SEARCH-1 denial 8: *"no name-pattern
process check, no signal to any process THE LOOP DID NOT LAUNCH, no host reboot,
no privileged cache action outside the sanctioned path, no inference run OUTSIDE
A HELD CLAIM."* Each of those is enforced here structurally, not documented:

* **held claim** — `MicrobenchRunner` has no default claim and no `claim=None`
  branch that proceeds. `_attest_claim()` runs immediately before *every* spawn,
  not once at the top, so a claim revoked mid-run stops the run at the next
  invocation instead of at the next campaign. `CpuRegionClaimAdapter` makes that
  real by going back to the FILESYSTEM on every attestation:
  `CpuRegionClaim.held` is `not self._released`, an in-process flag no external
  event can move, so an adapter built on it alone would re-read a variable and
  call it a revocation check.
* **no name-pattern anything** — this module never enumerates processes, never
  matches a command line, and never calls `os.kill`. The only termination path
  is `Popen.terminate()`/`Popen.kill()` on the handle `SubprocessSpawner` itself
  created. `audit_no_name_pattern_process_paths()` proves it from the AST.
* **the frozen trees** — enforced upstream by `recipes._assert_arm_allows_binding`
  (denial 2), which is why the anchor arm here goes through `recipes.construct`
  exactly like the candidate rather than around it. Note what that check does
  NOT do: it exempts the ANCHOR arm entirely, because reading the frozen binary
  is not a write. Nothing therefore constrains where `anchor_binding` points,
  and the only thing standing between "the frozen anchor" and "a second
  candidate build wearing the anchor's name" is
  `MicrobenchRunner._check_anchor_identity`, which compares the digest of the
  binary about to run against the `api.AnchorIdentity` the plan declares.

THE FIVE THINGS THIS MODULE REFUSES TO GET WRONG
------------------------------------------------
1. **A blocked design cannot be mislabelled as a paired one.** The runner takes
   no `order` argument anywhere in its public surface; order comes from
   `statistics.OrderSchedule`, which derives it from the committed campaign seed.
   `assemble_block()` then re-derives the order from the arms that were OBSERVED
   to run and requires strict alternation, so a caller cannot hand it
   `[anchor, anchor, candidate, candidate]` with `order="anchor_first"` stapled
   on. That is `PairingViolation`, and there is no code path that produces a
   `PairedBlock` except through that check.

2. **The recipe, not a hand-typed line.** Every argv comes from
   `recipes.construct()`. The canonical CPU prefix
   (`taskset -c 0-95 numactl --interleave=all`), the canonical bench flags
   (`-t 96 -fa 1 -mmp 0` — note `-fa 1`, where `llama-bench` itself defaults to
   `-fa 0`) and the mandatory OMP stack are all sourced constants, and
   `check_recipe_discipline()` re-asserts them against the argv that is about to
   be spawned.

3. **A receipt that can actually be verified.** `evaluator/api.check_preconditions`
   passes precondition 6 on the mere PRESENCE of an `api.RecipeReceipt`, so a
   hand-typed argv is currently undetectable at that gate. `ExecutionReceipt`
   carries the resolved argv, the executed env, and the SHA-256 of the binary
   that ran; `verify_receipt()` recomputes all three and FAILS on a single
   mutated token. See `RECEIPT_VERIFICATION_REQUIREMENTS` for what the upstream
   gate would need to consume it.

4. **The raw vector, not a summary.** `MicrobenchRun.raw_vector()` emits every
   per-repetition `samples_ts` reading, per invocation, per arm, per block,
   alongside the `api.ScopeDenominator` of the cell. A full-machine gate applied
   to a partial-machine cell is a category error, and the denominator travelling
   with the samples is what makes it detectable after the fact.

5. **Honest failure.** A run that was contended, throttled, or short of its
   blocks does not emit a number. `MicrobenchRun.paired_blocks()` RAISES
   `RunRefused` rather than returning the blocks it did manage — a partial
   paired-block set is not a small measurement, it is a differently-shaped one.
   The refused run still carries its raw vector and its reasons, because a
   failure that is not durable is indistinguishable from a run that never
   happened.

WHY STDOUT GOES TO A FILE AND NOT A PIPE
----------------------------------------
*"Never pipe llama binaries through another process; it changes their
behaviour."* `SubprocessSpawner` redirects stdout and stderr to temporary FILES
and never constructs a `subprocess.PIPE`. This also removes the pipe-buffer
deadlock that a large `-o json` payload would otherwise create.

WHY ONE PID IS ENOUGH
---------------------
The canonical prefix is `taskset -c 0-95 numactl --interleave=all <binary> ...`.
Both `taskset` and `numactl` `execvp()` their target rather than forking it, so
the pid returned by `Popen` IS the `llama-bench` process by the time it is
running. Terminating that single captured pid therefore needs no process group
and no name pattern — which is the whole point, given INC-20260731.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .. import schemas, storage
from ..evaluator import api, integrity, recipes, statistics

__all__ = [
    # identity
    "RUNNER_ID", "TESTDATA_DIR",
    # errors
    "MicrobenchError", "ClaimNotHeld", "PairingViolation", "BenchOutputError",
    "RecipeOutputMismatch", "RunRefused", "SpawnFailure", "HostStateUnreadable",
    # claim seam
    "HeldClaim", "ClaimAttestation", "CpuRegionClaimAdapter",
    # host state
    "HostState", "HostStatePolicy", "read_host_state", "DEFAULT_BASE_ENV_KEYS",
    # env
    "assemble_env", "EnvAssembly",
    # receipt
    "ExecutionReceipt", "verify_receipt", "RECEIPT_VERIFICATION_REQUIREMENTS",
    "check_recipe_discipline",
    # output parsing
    "BenchRow", "parse_llama_bench_json", "LlamaBenchExpectation",
    "commit_prefix_match",
    # spawning
    "SpawnResult", "Spawner", "SubprocessSpawner", "RecordedSpawner",
    "ProductionTreeWrite", "assert_scratch_root_is_not_production",
    # pairing
    "ARM_ANCHOR", "ARM_CANDIDATE", "BlockPlan", "plan_blocks", "assemble_block",
    "Invocation", "BlockRecord",
    # the run
    "MicrobenchPlan", "MicrobenchRun", "MicrobenchRunner",
    # self-audit
    "audit_no_name_pattern_process_paths",
]

#: Identity of this runner. It goes into every receipt and every raw vector, so a
#: record naming it names the exact execution semantics that produced its samples.
RUNNER_ID = "autokernel.execution.microbench/v1"

TESTDATA_DIR = Path(__file__).resolve().parent / "testdata"

ARM_ANCHOR = "anchor"
ARM_CANDIDATE = "candidate"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# =============================================================================
# Errors — every one is a refusal. None has a degraded-result branch.
# =============================================================================

class MicrobenchError(Exception):
    """Base for every refusal this module makes."""


class ClaimNotHeld(MicrobenchError):
    """Denial 8: *"no inference run OUTSIDE A HELD CLAIM."*

    Raised before any process is spawned, and again before every subsequent
    invocation. There is deliberately no "claim could not be checked, proceeding"
    branch: an unverifiable claim is not a held claim.
    """


class PairingViolation(MicrobenchError):
    """The observed arm sequence is not an interleaved paired block.

    *"Blocked designs (candidate x n, then anchor x n) are forbidden — thermal
    and page-cache drift alias onto the arm effect."* This is raised rather than
    returned so that no `PairedBlock` can come into existence from a blocked run.
    """


class BenchOutputError(MicrobenchError):
    """The tool's output cannot be read as the sample vector it must carry."""


class RecipeOutputMismatch(MicrobenchError):
    """The output contradicts the recipe that was supposed to have produced it.

    `-fa 1` in argv but `flash_attn: false` in the JSON means the flag did not
    take effect. The number would be real and would measure the wrong thing,
    which is the most expensive class of benchmark defect this project has.
    """


class RunRefused(MicrobenchError):
    """A number was requested from a run that refused to produce one."""


class SpawnFailure(MicrobenchError):
    """The process could not be started, or did not finish within its bound."""


class HostStateUnreadable(MicrobenchError):
    """Host frequency or load could not be read at all."""


# =============================================================================
# The resource-claim seam
# =============================================================================

@dataclass(frozen=True)
class ClaimAttestation:
    """One attestation that the claim was held at a named instant.

    `check` is a three-outcome `schemas.Check`, and `COULD_NOT_CHECK` is NOT a
    pass here — see `HeldClaim`.
    """

    claim_id: str
    holder: str
    cpu_list: str
    observed_at: str
    check: schemas.Check

    def __post_init__(self) -> None:
        for name in ("claim_id", "holder", "cpu_list", "observed_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"claim attestation {name} must be a non-empty string")
        if not isinstance(self.check, schemas.Check):
            raise TypeError("claim attestation check must be a schemas.Check")

    @property
    def held(self) -> bool:
        return self.check.outcome == schemas.PASS

    def to_dict(self) -> dict:
        return {"claim_id": self.claim_id, "holder": self.holder,
                "cpu_list": self.cpu_list, "observed_at": self.observed_at,
                "outcome": self.check.outcome, "reasons": list(self.check.reasons)}


class HeldClaim(Protocol):
    """The claim this runner runs under. NOT implemented in this module.

    The GPU implementation is `resource/device_claim.py`; the CPU region-claim
    ACQUISITION is `execution/cpu_region_claim.py`, built in parallel with this
    file. This module deliberately depends on the PROTOCOL and not on either
    implementation, so the runner is testable without acquiring a real lock and
    so a future third claim kind needs no edit here.

    A conforming claim re-reads its own lock on every `attest()` call. Returning
    a cached PASS defeats the entire mid-run revocation check: the point of
    attesting before every invocation is that the answer can CHANGE.
    """

    claim_id: str

    def attest(self) -> ClaimAttestation:
        ...


def _cpu_region_claim_module():
    """Lazily import the sibling CPU-claim module. Returns None when absent.

    LAZY and inside the call on purpose: `microbench.py` must import cleanly
    whether or not the CPU-claim module is present, so the Protocol stays the
    contract and the adapter stays a convenience.
    """
    try:
        from . import cpu_region_claim                      # noqa: PLC0415 - see docstring
    except ImportError:
        return None
    return cpu_region_claim


class CpuRegionClaimAdapter:
    """Adapts `execution/cpu_region_claim.CpuRegionClaim` to the `HeldClaim` seam.

    WHY THIS RE-READS THE LOCK AND NOT `claim.held`
    -----------------------------------------------
    `CpuRegionClaim.held` is `not self._released` — an IN-PROCESS boolean that
    only this process's own `release()` can move. Asking it a second time cannot
    return a different answer for any reason external to this process, so an
    adapter built on it alone would re-read a variable, not a lock, and the
    runner's mid-run revocation check would be decorative: a claim whose flock
    was dropped, whose lock file was replaced, or whose payload now names a
    different claim would still attest PASS on every invocation of an hour-long
    campaign.

    So every `attest()` goes back to the filesystem through the sibling's own
    `check_region_claim_held(receipt)`, which probes each flock and reads each
    payload, plus `check_claim_expiry(receipt)`. `held` is still consulted —
    it is the one thing that catches a claim this process already released —
    but it is the cheapest of the four conjuncts, not the only one.

    COULD_NOT_CHECK from the lock probe is NOT a pass: an unverifiable claim is
    not a held one. A claim that declared no maximum hold yields COULD_NOT_CHECK
    from the expiry checker, and THAT one is recorded without failing, because a
    missing expiry is a property of the claim's declaration rather than evidence
    that the claim has gone.
    """

    def __init__(self, claim: Any, *, cpu_list: str) -> None:
        for attribute in ("claim_id", "held", "covers", "receipt"):
            if not hasattr(claim, attribute):
                raise TypeError(
                    f"{type(claim).__name__} has no {attribute!r}; it is not a "
                    f"cpu_region_claim.CpuRegionClaim and cannot be adapted. Implement "
                    f"the HeldClaim Protocol directly instead of loosening this check.")
        self._claim = claim
        self._cpu_list = cpu_list
        self.claim_id = str(claim.claim_id)

    def attest(self) -> ClaimAttestation:
        reasons: list = []
        notes: list = []
        if not self._claim.held:
            reasons.append("the region lock is no longer held by this process")
        elif not self._claim.covers(self._cpu_list):
            reasons.append(
                f"the held claim does not cover {self._cpu_list!r}; precondition 1 "
                f"requires a CPU region claim covering the EXACT footprint measured")
        else:
            reasons.extend(self._reread_locks(notes))
        outcome = schemas.FAIL if reasons else schemas.PASS
        return ClaimAttestation(
            claim_id=self.claim_id, holder=f"pid:{os.getpid()}", cpu_list=self._cpu_list,
            observed_at=_utc_now(),
            check=schemas.Check(outcome, tuple(reasons) or tuple(notes)))

    def _reread_locks(self, notes: list) -> list:
        """Go back to the filesystem. Anything short of PASS on the lock probe fails."""
        module = _cpu_region_claim_module()
        if module is None:
            return ["execution/cpu_region_claim.py could not be imported, so the region "
                    "lock could not be re-read; `claim.held` alone is an in-process flag "
                    "and an unverifiable claim is not a held one"]
        try:
            receipt = self._claim.receipt()
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            return [f"the claim could not produce a receipt to re-read its locks: {exc}"]

        reasons: list = []
        held = module.check_region_claim_held(receipt)
        if held.outcome != schemas.PASS:
            reasons.append(
                f"re-reading the region locks named by the receipt is {held.outcome}: "
                f"{'; '.join(held.reasons)}")
        else:
            notes.extend(held.reasons)

        expiry = module.check_claim_expiry(receipt)
        if expiry.outcome == schemas.FAIL:
            reasons.append(f"the claim has expired: {'; '.join(expiry.reasons)}")
        else:
            notes.extend(expiry.reasons)
        return reasons


# =============================================================================
# Host state — frequency and contention
# =============================================================================

#: The only ambient environment variables that survive into a measured process.
#: Everything else in the recipe env is declared by the recipe, so a stray
#: ambient `OMP_PROC_BIND` in the operator's shell cannot reach the benchmark.
DEFAULT_BASE_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")


@dataclass(frozen=True)
class HostState:
    """CPU frequency and load, as read at one instant, with its own provenance.

    `khz_by_cpu` is retained per-cpu rather than averaged because a throttle that
    hits one CCD is invisible in a mean and obvious in a minimum.
    """

    observed_at: str
    cpu_list: str
    khz_by_cpu: tuple
    driver_min_khz: Optional[int]
    driver_max_khz: Optional[int]
    load1: Optional[float]
    source: str
    unreadable: tuple = ()

    @property
    def min_khz(self) -> Optional[int]:
        values = [khz for _, khz in self.khz_by_cpu]
        return min(values) if values else None

    @property
    def median_khz(self) -> Optional[float]:
        values = sorted(khz for _, khz in self.khz_by_cpu)
        if not values:
            return None
        return statistics.median(tuple(float(v) for v in values))

    @property
    def readable(self) -> bool:
        return bool(self.khz_by_cpu)

    def to_dict(self) -> dict:
        return {
            "observed_at": self.observed_at,
            "cpu_list": self.cpu_list,
            "khz_by_cpu": [[cpu, khz] for cpu, khz in self.khz_by_cpu],
            "min_khz": self.min_khz,
            "median_khz": self.median_khz,
            "driver_min_khz": self.driver_min_khz,
            "driver_max_khz": self.driver_max_khz,
            "load1": self.load1,
            "source": self.source,
            "unreadable": list(self.unreadable),
        }


def _parse_cpu_list(cpu_list: str) -> tuple:
    """`"0-95"` / `"184-191"` / `"0,2,4"` -> a tuple of ints. Mirrors taskset's grammar."""
    if not isinstance(cpu_list, str) or not cpu_list.strip():
        raise ValueError("cpu_list must be a non-empty string")
    members: list = []
    for part in cpu_list.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"cpu_list {cpu_list!r} has an empty member")
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                lo, hi = int(low), int(high)
            except ValueError as exc:
                raise ValueError(f"cpu_list {cpu_list!r}: bad range {part!r}") from exc
            if hi < lo:
                raise ValueError(f"cpu_list {cpu_list!r}: descending range {part!r}")
            members.extend(range(lo, hi + 1))
        else:
            try:
                members.append(int(part))
            except ValueError as exc:
                raise ValueError(f"cpu_list {cpu_list!r}: bad member {part!r}") from exc
    return tuple(members)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _read_int_file(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def read_host_state(*, cpu_list: str, sysfs_root: Any = "/sys/devices/system/cpu",
                    proc_root: Any = "/proc", now: Callable[[], str] = _utc_now) -> HostState:
    """Read per-cpu scaling frequency and 1-minute load for the claimed footprint.

    `sysfs_root` and `proc_root` are injectable so the throttle guard can be
    tested against a synthesised sysfs — a guard that can only be exercised on a
    genuinely throttled machine is a guard that is never exercised.

    A cpu whose `scaling_cur_freq` cannot be read is recorded in `unreadable`
    rather than skipped silently, because "we could not see the throttled core"
    and "there was no throttled core" must not render identically.
    """
    sysfs = Path(sysfs_root)
    cpus = _parse_cpu_list(cpu_list)
    readings: list = []
    unreadable: list = []
    for cpu in cpus:
        khz = _read_int_file(sysfs / f"cpu{cpu}" / "cpufreq" / "scaling_cur_freq")
        if khz is None:
            unreadable.append(f"cpu{cpu}: scaling_cur_freq unreadable")
        else:
            readings.append((cpu, khz))

    first = cpus[0] if cpus else 0
    driver_min = _read_int_file(sysfs / f"cpu{first}" / "cpufreq" / "cpuinfo_min_freq")
    driver_max = _read_int_file(sysfs / f"cpu{first}" / "cpufreq" / "cpuinfo_max_freq")

    load1: Optional[float] = None
    try:
        fields = Path(proc_root, "loadavg").read_text(encoding="utf-8").split()
        load1 = float(fields[0])
    except (OSError, ValueError, IndexError):
        unreadable.append("loadavg unreadable")

    return HostState(
        observed_at=now(), cpu_list=cpu_list, khz_by_cpu=tuple(readings),
        driver_min_khz=driver_min, driver_max_khz=driver_max, load1=load1,
        source=str(sysfs), unreadable=tuple(unreadable))


@dataclass(frozen=True)
class HostStatePolicy:
    """When a host is too throttled or too busy to produce a usable number.

    `nominal_khz` has NO DEFAULT and is not derived from `cpuinfo_max_freq`.
    That file reports the single-core boost ceiling (4.51 GHz on this part); an
    all-core benchmark never reaches it, so a ratio against it would fail every
    healthy run and would be switched off within a week. The correct reference is
    the frequency this host actually sustained on this cell when it was known
    healthy, which only the operator can supply — so an absent `nominal_khz` is
    `COULD_NOT_CHECK`, never a pass.

    Origin of the frequency check at all: a multi-day host throttle held this
    machine at roughly 40% of its normal clock and silently poisoned every number
    taken during the window (`feedback_host_throttle_check`). The run that
    detects it must refuse, not annotate.

    `max_load_per_core` is evaluated at run OPEN only. During the run the
    benchmark itself saturates the claimed cores, so a mid-run load reading says
    nothing about foreign contention; mid-run contention is detected by
    re-attesting the CLAIM before every invocation, which is the layer that can
    actually attribute it.
    """

    nominal_khz: Optional[int] = None
    min_frequency_ratio: float = 0.85
    max_load_per_core: float = 0.25
    require_frequency: bool = True
    require_load: bool = True

    def __post_init__(self) -> None:
        if self.nominal_khz is not None:
            if isinstance(self.nominal_khz, bool) or not isinstance(self.nominal_khz, int) \
                    or self.nominal_khz <= 0:
                raise ValueError("nominal_khz must be a positive int or None")
        for name in ("min_frequency_ratio", "max_load_per_core"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")

    def check_frequency(self, state: HostState) -> schemas.Check:
        if not isinstance(state, HostState):
            raise TypeError("check_frequency takes a HostState")
        if not self.require_frequency:
            # NOT a PASS. A check the caller switched off did not happen, and
            # this module's own rule everywhere else is that COULD_NOT_CHECK is
            # not a pass — the same rule `recipes._check_binding_inputs` states
            # for `verify_inputs=False`. A success-shaped result from a disabled
            # guard is the fail-open shape that makes a guard worth nothing: the
            # multi-day throttle that poisoned every number taken in its window
            # would have produced exactly this record.
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "frequency checking was disabled by the caller "
                "(HostStatePolicy.require_frequency=False), so the host's clock was "
                "never read; a run under this policy does not emit a number",))
        if not state.readable:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "no cpu in the claimed footprint reported scaling_cur_freq; a multi-day "
                "host throttle has silently poisoned results here before, so an "
                "unverifiable frequency is not a passing frequency",) + state.unreadable)
        reasons: list = []
        min_khz = state.min_khz
        if state.unreadable:
            reasons.extend(state.unreadable)
        if state.driver_min_khz is not None and min_khz is not None \
                and min_khz <= state.driver_min_khz:
            return schemas.Check(schemas.FAIL, tuple(reasons) + (
                f"cpu frequency is pinned at the driver's own minimum "
                f"({min_khz} kHz <= cpuinfo_min_freq {state.driver_min_khz} kHz); "
                f"this is a throttled host, not a quiet one",))
        if self.nominal_khz is None:
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (
                "HostStatePolicy.nominal_khz was not supplied, so the observed "
                f"{min_khz} kHz cannot be compared against a healthy reference for this "
                "cell. cpuinfo_max_freq is the single-core boost ceiling and is NOT a "
                "valid all-core reference; record a healthy observation instead.",))
        floor = self.nominal_khz * self.min_frequency_ratio
        if min_khz is not None and min_khz < floor:
            return schemas.Check(schemas.FAIL, tuple(reasons) + (
                f"slowest claimed cpu is at {min_khz} kHz, below "
                f"{self.min_frequency_ratio:.2f} x nominal {self.nominal_khz} kHz "
                f"({floor:.0f} kHz); refusing to emit a number from a throttled host",))
        if reasons:
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons))
        return schemas.Check(schemas.PASS, (
            f"min {min_khz} kHz >= {floor:.0f} kHz "
            f"({self.min_frequency_ratio:.2f} x nominal {self.nominal_khz} kHz)",))

    def check_load(self, state: HostState, *, cpu_count: int) -> schemas.Check:
        if not self.require_load:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "contention checking was disabled by the caller "
                "(HostStatePolicy.require_load=False), so /proc/loadavg was never read; "
                "a benchmark taken under unmeasured contention is garbage data",))
        if state.load1 is None:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "/proc/loadavg could not be read; contention is unevaluable",))
        if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count < 1:
            raise ValueError("cpu_count must be a positive int")
        per_core = state.load1 / cpu_count
        if per_core > self.max_load_per_core:
            return schemas.Check(schemas.FAIL, (
                f"1-minute load {state.load1:.2f} over {cpu_count} claimed cores is "
                f"{per_core:.2f}/core, above the {self.max_load_per_core:.2f} ceiling; "
                f"a benchmark taken under this contention is garbage data and steals "
                f"from whoever else is on the host",))
        return schemas.Check(schemas.PASS, (
            f"1-minute load {state.load1:.2f} over {cpu_count} cores = "
            f"{per_core:.2f}/core",))

    def to_dict(self) -> dict:
        return {"nominal_khz": self.nominal_khz,
                "min_frequency_ratio": self.min_frequency_ratio,
                "max_load_per_core": self.max_load_per_core,
                "require_frequency": self.require_frequency,
                "require_load": self.require_load}


# =============================================================================
# Environment assembly
# =============================================================================

@dataclass(frozen=True)
class EnvAssembly:
    """The exact environment a measured process will receive, and why.

    The recipe's declared env always wins, and the ambient environment
    contributes ONLY the keys in `base_keys`. That is what stops an operator's
    exported `OMP_PROC_BIND=close` from reaching a benchmark whose recipe says
    `spread` — a deviation that would not show up anywhere in argv.

    `dropped_ambient` names every ambient variable whose VALUE did not reach the
    process, including the ones the recipe overrode. "Absent" and "overridden"
    are both "did not reach", and the interesting case is the second.
    """

    env: dict
    recipe_keys: tuple
    base_keys: tuple
    dropped_ambient: tuple

    def to_dict(self) -> dict:
        return {"env": dict(self.env), "recipe_keys": list(self.recipe_keys),
                "base_keys": list(self.base_keys),
                "dropped_ambient": list(self.dropped_ambient)}


def assemble_env(command_env: Mapping, *, environ: Optional[Mapping] = None,
                 base_keys: Sequence[str] = DEFAULT_BASE_ENV_KEYS) -> EnvAssembly:
    """Build the process env: recipe env, plus an allowlisted slice of the ambient one.

    Raises if the ambient slice would collide with a recipe key. By construction
    `DEFAULT_BASE_ENV_KEYS` is disjoint from every recipe env key, so a collision
    means someone widened the allowlist to include something the recipe controls,
    and silently letting the recipe win would hide that.
    """
    if not isinstance(command_env, Mapping):
        raise TypeError("command_env must be a mapping")
    ambient = os.environ if environ is None else environ
    recipe_keys = tuple(sorted(command_env))
    env = {str(k): str(v) for k, v in command_env.items()}

    used_base: list = []
    for key in base_keys:
        if key in env:
            raise ValueError(
                f"base env key {key!r} collides with a recipe-declared key; the recipe "
                f"controls that variable and an ambient value must not shadow or be "
                f"shadowed by it silently")
        value = ambient.get(key)
        if value is not None:
            env[key] = str(value)
            used_base.append(key)

    dropped = tuple(sorted(k for k, v in ambient.items() if env.get(k) != str(v)))
    return EnvAssembly(env=env, recipe_keys=recipe_keys, base_keys=tuple(used_base),
                       dropped_ambient=dropped)


# =============================================================================
# The receipt — precondition 6, made verifiable
# =============================================================================

#: What an upstream `verify_receipt()` would need in order to make precondition 6
#: bite. `api.check_preconditions` currently passes it on the mere PRESENCE of an
#: `api.RecipeReceipt`, whose three fields (constructor id, constructor sha, argv
#: sha) are all self-reported: nothing in the record says what argv actually ran,
#: so a hand-typed line accompanied by a copied receipt is indistinguishable from
#: a constructed one. Reported rather than patched in, because `evaluator/api.py`
#: is owned by another workstream this hour.
RECEIPT_VERIFICATION_REQUIREMENTS = (
    "the RESOLVED ARGV, token by token — `api.RecipeReceipt.argv_sha256` is a hash "
    "of an argv the record does not contain, so it can only be checked against a "
    "reconstruction, and there is nothing to reconstruct it from",
    "the EXECUTED ENV — the OMP stack and LD_LIBRARY_PATH change the measured "
    "number and appear nowhere in argv; two runs with identical argv and different "
    "OMP_PROC_BIND are different measurements with the same receipt",
    "the SHA-256 OF THE BINARY THAT RAN — `ToolBinding` names a path, and a path is "
    "not an identity on a host where the experimental tree is rebuilt between blocks",
    "the RECIPE ID PLUS REGISTRY ID — so `recipes.construct()` can be re-run and the "
    "argv re-derived independently rather than trusted",
    "a re-derivation, not a comparison of self-reported digests — a receipt that "
    "hashes its own claims is satisfied by any consistent forgery",
)


@dataclass(frozen=True)
class ExecutionReceipt:
    """Proof of which argv actually ran, in a form that can be re-derived.

    `api.RecipeReceipt` is the three-field object the record grammar wants; this
    is a superset that keeps the material those three hashes were taken OVER, so
    `verify_receipt()` can recompute them instead of comparing a digest to itself.

    TWO ENVIRONMENTS, deliberately:

    * `recipe_env` is what `recipes.construct()` declared. `argv_sha256` is taken
      over it, which is what makes this receipt's hash byte-identical to the
      constructor's own — two independent computations of one identity that must
      never drift.
    * `env` is what the process actually received: `recipe_env` plus the
      allowlisted ambient slice (`PATH`, `HOME`, …). `env_sha256` is taken over
      it.

    Collapsing them would force a choice between a receipt that disagrees with
    the constructor and a receipt that does not describe the process that ran.
    `verify_receipt()` additionally requires every `recipe_env` entry to survive
    into `env` unchanged, so the assembly step cannot quietly alter a
    recipe-controlled variable.
    """

    runner_id: str
    recipe_id: str
    registry_id: str
    arm: str
    constructor_id: str
    constructor_sha256: str
    argv_sha256: str
    argv: tuple
    recipe_env: dict
    env: dict
    env_sha256: str
    binary_path: str
    binary_sha256: str
    binary_size: int
    source_root: str
    library_path: str
    resolved_at: str

    def __post_init__(self) -> None:
        for name in ("runner_id", "recipe_id", "registry_id", "constructor_id",
                     "binary_path", "source_root", "library_path", "resolved_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"receipt.{name} must be a non-empty string")
        if self.arm not in recipes.ARMS:
            raise ValueError(f"receipt.arm: {self.arm!r} is not one of {list(recipes.ARMS)}")
        for name in ("constructor_sha256", "argv_sha256", "env_sha256", "binary_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_RE.match(value):
                raise ValueError(f"receipt.{name} must be a 64-hex SHA-256, got {value!r}")
        if not isinstance(self.argv, tuple) or not self.argv:
            raise ValueError("receipt.argv must be a non-empty tuple")
        for name in ("env", "recipe_env"):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"receipt.{name} must be a dict")
        if isinstance(self.binary_size, bool) or not isinstance(self.binary_size, int) \
                or self.binary_size < 0:
            raise ValueError("receipt.binary_size must be a non-negative int")

    @property
    def recipe_receipt(self) -> api.RecipeReceipt:
        """The three-field form that drops into `api.WindowAttestations.recipe`."""
        return api.RecipeReceipt(constructor_id=self.constructor_id,
                                 constructor_sha256=self.constructor_sha256,
                                 argv_sha256=self.argv_sha256)

    def render(self) -> str:
        return self.recipe_receipt.render()

    def to_dict(self) -> dict:
        return {
            "runner_id": self.runner_id, "recipe_id": self.recipe_id,
            "registry_id": self.registry_id, "arm": self.arm,
            "constructor_id": self.constructor_id,
            "constructor_sha256": self.constructor_sha256,
            "argv_sha256": self.argv_sha256, "argv": list(self.argv),
            "recipe_env": dict(self.recipe_env),
            "env": dict(self.env), "env_sha256": self.env_sha256,
            "binary_path": self.binary_path, "binary_sha256": self.binary_sha256,
            "binary_size": self.binary_size, "source_root": self.source_root,
            "library_path": self.library_path, "resolved_at": self.resolved_at,
        }


def _env_hash(env: Mapping) -> str:
    return schemas.content_hash({"env": {str(k): str(v) for k, v in env.items()}})


def _argv_hash(*, recipe_id: str, registry_id: str, arm: str, argv: Sequence[str],
               env: Mapping) -> str:
    """Exactly `recipes.construct`'s own argv_sha256 preimage, so the two agree."""
    return schemas.content_hash({
        "recipe_id": recipe_id, "registry_id": registry_id, "arm": arm,
        "argv": list(argv), "env": dict(env),
    })


def build_receipt(command: recipes.ConstructedCommand, *, env: Mapping,
                  binary_sha256: Optional[str] = None,
                  binary_size: Optional[int] = None,
                  now: Callable[[], str] = _utc_now) -> ExecutionReceipt:
    """Turn a `ConstructedCommand` plus the env that will really run into a receipt.

    The binary digest is streamed from disk when not supplied. It is NOT optional
    in the resulting object: a receipt that cannot say which build ran does not
    distinguish a candidate from its anchor when both live at the same path
    across a rebuild.
    """
    if not isinstance(command, recipes.ConstructedCommand):
        raise TypeError("build_receipt takes a recipes.ConstructedCommand")
    if binary_sha256 is None:
        binary_sha256 = integrity.sha256_file(command.binding.binary)
    if binary_size is None:
        binary_size = Path(command.binding.binary).stat().st_size
    return ExecutionReceipt(
        runner_id=RUNNER_ID,
        recipe_id=command.recipe_id,
        registry_id=command.registry_id,
        arm=command.arm,
        constructor_id=command.receipt.constructor_id,
        constructor_sha256=command.receipt.constructor_sha256,
        argv_sha256=command.receipt.argv_sha256,
        argv=tuple(command.argv),
        recipe_env=dict(command.env),
        env=dict(env),
        env_sha256=_env_hash(env),
        binary_path=command.binding.binary,
        binary_sha256=binary_sha256,
        binary_size=int(binary_size),
        source_root=command.binding.source_root,
        library_path=command.binding.library_path,
        resolved_at=now(),
    )


def verify_receipt(receipt: ExecutionReceipt, *, argv: Sequence[str], env: Mapping,
                   binary_sha256: Optional[str] = None,
                   reconstruct: bool = False) -> schemas.Check:
    """Recompute the receipt against what actually ran. FAILs on one changed token.

    This is what `api.check_preconditions`' precondition 6 cannot currently do.
    Three independent checks, and each one alone catches a real forgery:

    * `argv` and `env` are compared token by token AND by recomputed digest, so
      neither an edited argv with a stale hash nor a consistent re-hash of an
      edited argv passes.
    * `binary_sha256`, when supplied, is compared against the digest recorded at
      construction, which catches a rebuild between the receipt and the run.
    * `reconstruct=True` additionally re-runs `recipes.construct()` for the
      receipt's own recipe id and arm and requires the argv to come out
      identical. That is the only check that does not trust the receipt at all;
      it needs the binding to still exist on disk, so it is opt-in.
    """
    if not isinstance(receipt, ExecutionReceipt):
        raise TypeError("verify_receipt takes an ExecutionReceipt")
    argv = tuple(str(t) for t in argv)
    env = {str(k): str(v) for k, v in env.items()}
    reasons: list = []

    if argv != receipt.argv:
        reasons.append(
            f"argv does not match the receipt: receipt has {len(receipt.argv)} tokens, "
            f"observed {len(argv)}; first difference at index "
            f"{_first_difference(receipt.argv, argv)}")
    if env != receipt.env:
        differing = sorted(set(env) ^ set(receipt.env)) or \
            sorted(k for k in env if env[k] != receipt.env.get(k))
        reasons.append(f"env does not match the receipt; differing keys: {differing}")

    recomputed_env = _env_hash(env)
    if recomputed_env != receipt.env_sha256:
        reasons.append(f"env_sha256 {receipt.env_sha256[:12]} does not hash the observed "
                       f"env ({recomputed_env[:12]})")

    # `argv_sha256` is the CONSTRUCTOR's identity, taken over the recipe env, so
    # it is recomputed over that env and not over the assembled one.
    recomputed_argv = _argv_hash(recipe_id=receipt.recipe_id, registry_id=receipt.registry_id,
                                 arm=receipt.arm, argv=argv, env=receipt.recipe_env)
    if recomputed_argv != receipt.argv_sha256:
        reasons.append(f"argv_sha256 {receipt.argv_sha256[:12]} does not hash the "
                       f"(recipe, registry, arm, argv, recipe_env) tuple "
                       f"({recomputed_argv[:12]}); this receipt did not come from "
                       f"recipes.construct()")

    # Every recipe-declared variable must survive assembly unchanged. Without
    # this, the assembly step is an unaudited place to alter a measurement.
    altered = sorted(k for k, v in receipt.recipe_env.items() if env.get(k) != str(v))
    if altered:
        reasons.append(
            f"the executed env alters recipe-declared variables {altered}; the recipe "
            f"controls those and an assembly step must not change them")

    if binary_sha256 is not None and binary_sha256 != receipt.binary_sha256:
        reasons.append(f"the binary that ran ({binary_sha256[:12]}) is not the binary the "
                       f"receipt was taken over ({receipt.binary_sha256[:12]})")

    if reconstruct:
        try:
            binding = recipes.ToolBinding(binary=receipt.binary_path,
                                          source_root=receipt.source_root,
                                          library_path=receipt.library_path)
            rebuilt = recipes.construct(receipt.recipe_id, binding=binding,
                                        params=_params_from_argv(receipt),
                                        arm=receipt.arm, verify_inputs=False)
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons) + (
                f"independent reconstruction was requested but failed: {exc}",))
        if tuple(rebuilt.argv) != argv:
            reasons.append(
                "independent reconstruction from the recipe registry produced a different "
                f"argv at index {_first_difference(tuple(rebuilt.argv), argv)}; the "
                "observed command was not emitted by this recipe")

    return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons else schemas.Check(
        schemas.PASS, (f"argv ({len(argv)} tokens), env ({len(env)} keys) and binary digest "
                       f"all re-derive to the recorded receipt",))


def _first_difference(a: Sequence, b: Sequence) -> Optional[int]:
    for index in range(max(len(a), len(b))):
        if index >= len(a) or index >= len(b) or a[index] != b[index]:
            return index
    return None


def _params_from_argv(receipt: ExecutionReceipt) -> dict:
    """Recover the recipe params from the receipt's own argv, for reconstruction.

    Deliberately reads the ARGV rather than a stored params dict: a stored dict
    would be one more self-reported field, and the point of `reconstruct=True` is
    to trust nothing the receipt asserts about itself except the tokens it claims
    were executed.
    """
    argv = list(receipt.argv)
    flags = {"-m": "model", "-r": "reps", "-d": "n_depth", "-ub": "ubatch", "-b": "batch",
             "-o": "output_format", "-ngl": "n_gpu_layers", "-t": "threads"}
    params: dict = {}
    for index, token in enumerate(argv[:-1]):
        name = flags.get(token)
        if name is not None:
            params[name] = argv[index + 1]
    if "-p" in argv:
        params["n_prompt"] = argv[argv.index("-p") + 1]
    if "-n" in argv:
        params["n_gen"] = argv[argv.index("-n") + 1]
    spec = recipes.get_recipe(receipt.recipe_id)
    declared = spec.param_map
    out: dict = {}
    for key, value in params.items():
        if key not in declared:
            continue
        try:
            out[key] = int(value)
        except (TypeError, ValueError):
            out[key] = value
    # `-p 0` / `-n 0` are the other metric's sentinel, not a parameter of this one.
    for key in ("n_prompt", "n_gen"):
        if out.get(key) == 0 and key not in declared:
            out.pop(key, None)
    return {k: v for k, v in out.items() if k in declared}


def check_recipe_discipline(command: recipes.ConstructedCommand,
                            env: Mapping) -> schemas.Check:
    """Re-assert the codified recipe against the argv that is about to be spawned.

    `recipes.construct()` already validates this, so this is a second, independent
    reading immediately before execution — the window in which a caller could have
    edited `command.argv` after construction is exactly the window this closes.

    It checks the three things the scars are about: the canonical prefix (NUMA
    interleave and the taskset mask), flash-attention ON where `llama-bench`
    itself defaults to OFF, and the complete OMP stack.
    """
    reasons: list = []
    argv = list(command.argv)

    if command.backend == "llama_cpu":
        prefix = list(recipes.CANONICAL_PREFIX)
        if argv[:len(prefix)] != prefix:
            reasons.append(
                f"argv does not start with the ratified canonical prefix {prefix}; "
                f"NUMA policy drifted off interleave once on a 1.7% warm A/B and the "
                f"front door ended up at 46% of canonical")
        for key, value in recipes.CANONICAL_OMP_ENV.items():
            if env.get(key) != value:
                reasons.append(
                    f"OMP stack incomplete: {key}={env.get(key)!r}, canonical recipe "
                    f"requires {value!r}. The OMP stack is MANDATORY, not optional.")

    if command.tool == "llama-bench":
        if "-fa" not in argv:
            reasons.append("`-fa` is absent; llama-bench defaults to -fa 0 and production "
                           "runs flash-attention ON, so a default here is a real and "
                           "useless number")
        elif argv[argv.index("-fa") + 1] != "1":
            reasons.append(f"`-fa {argv[argv.index('-fa') + 1]}` — production runs "
                           f"flash-attention ON")
        fmt_index = argv.index("-o") if "-o" in argv else None
        if fmt_index is None or argv[fmt_index + 1] not in \
                recipes.LLAMA_BENCH_SAMPLE_BEARING_FORMATS:
            reasons.append(
                f"output format must be one of "
                f"{list(recipes.LLAMA_BENCH_SAMPLE_BEARING_FORMATS)}; every other format "
                f"stops at avg/stddev and carries no per-repetition sample vector")

    if command.discipline_outcome == schemas.FAIL:
        failing = [f.finding_id for f in command.discipline
                   if f.check.outcome == schemas.FAIL]
        reasons.append(f"the recipe constructor itself reported FAIL discipline findings: "
                       f"{failing}")

    return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons else schemas.Check(
        schemas.PASS, ("canonical prefix, flash-attention and the OMP stack are all as the "
                       "codified recipe declares them",))


# =============================================================================
# Output parsing — llama-bench `-o json`
# =============================================================================

@dataclass(frozen=True)
class BenchRow:
    """One `llama-bench` result row, with its per-repetition sample vector.

    `samples_ts` is the vector the reducer needs; `avg_ts` is retained only so a
    parse can be cross-checked against the tool's own summary. The reduction is
    never taken from `avg_ts`: a mean of a mean is not reproducible from raw
    samples, and P-AK-SEARCH-1's search-grade conjunction requires that it is.
    """

    build_commit: str
    n_prompt: int
    n_gen: int
    n_depth: int
    n_threads: int
    flash_attn: bool
    use_mmap: bool
    model_filename: str
    n_batch: Optional[int]
    n_ubatch: Optional[int]
    avg_ts: float
    stddev_ts: float
    samples_ts: tuple
    samples_ns: tuple
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def metric_samples(self) -> tuple:
        return self.samples_ts

    def to_dict(self) -> dict:
        return {"build_commit": self.build_commit, "n_prompt": self.n_prompt,
                "n_gen": self.n_gen, "n_depth": self.n_depth,
                "n_threads": self.n_threads, "flash_attn": self.flash_attn,
                "use_mmap": self.use_mmap, "model_filename": self.model_filename,
                "n_batch": self.n_batch, "n_ubatch": self.n_ubatch,
                "avg_ts": self.avg_ts, "stddev_ts": self.stddev_ts,
                "samples_ts": list(self.samples_ts), "samples_ns": list(self.samples_ns)}


def _row_int(entry: Mapping, key: str, index: int, *, default: Any = None) -> int:
    """Read an integer field out of a result row, or raise `BenchOutputError`.

    `int(entry["n_threads"])` on `"ninety-six"` raises `ValueError`, which is not
    a `MicrobenchError`, so it escapes the runner's refusal path entirely and
    takes the whole run's durable record with it. Every field the row asserts is
    supplied by the process being measured, and a producer feeding a trusted
    evaluator must not be able to choose the exception type its consumer raises.
    """
    value = entry.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BenchOutputError(
            f"result row {index}: {key}={value!r} is not a number; the tool emitted a "
            f"row this parser cannot read, which is a refusal, not a crash")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BenchOutputError(
            f"result row {index}: {key}={value!r} is not an integer ({exc})") from exc


def _row_float(entry: Mapping, key: str, index: int, *, default: Any = None) -> float:
    value = entry.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BenchOutputError(
            f"result row {index}: {key}={value!r} is not a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchOutputError(
            f"result row {index}: {key}={value!r} is not a float ({exc})") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise BenchOutputError(f"result row {index}: {key}={value!r} is not finite")
    return result


_COMMIT_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def commit_prefix_match(expected: str, actual: str) -> tuple:
    """Do two commit spellings name the same commit? Returns `(matched, reason)`.

    Prefix matching is required — `llama-bench` reports an abbreviated
    `build_commit` (`91745611f`) while an `api.AnchorIdentity` carries the full
    40-hex form. It is also the whole hole: `"".startswith(anything)` is False
    but `anything.startswith("")` is TRUE, so a row whose `build_commit` is an
    empty string satisfied a naive two-way prefix test against EVERY anchor, and
    a one-character `build_commit` satisfied one anchor in sixteen. Both are
    reachable: a tree built from a tarball, or from a shallow copy with no git
    metadata, reports exactly that.

    So: both spellings must be hex, at least 7 characters (git's own minimum
    abbreviation), and the shorter must be a true prefix of the longer.
    """
    expected = str(expected)
    actual = str(actual)
    for name, value in (("the anchor identity", expected), ("the result row", actual)):
        if not _COMMIT_TOKEN_RE.match(value):
            return False, (
                f"{name} reports build_commit {value!r}, which is not a 7-to-40 character "
                f"hex commit. An unreadable commit is not a matching one — an empty or "
                f"truncated build_commit used to satisfy the anchor check vacuously")
    shorter, longer = sorted((expected, actual), key=len)
    if not longer.lower().startswith(shorter.lower()):
        return False, (f"anchor commit {expected!r} and reported build_commit {actual!r} "
                       f"do not share a prefix")
    return True, ""


def _as_bool(value: Any) -> bool:
    """`flash_attn` is `false` in older builds and `1` in current ones. Both are read."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "enabled")
    raise BenchOutputError(f"cannot read {value!r} as a boolean flag")


def parse_llama_bench_json(text: str) -> tuple:
    """Parse `llama-bench -o json` output into `BenchRow`s. Raises rather than guessing.

    Refuses, specifically:

    * anything that is not a JSON array of objects — `-o md` and `-o csv` land
      here, and they are the formats that carry no sample vector at all;
    * a row without `samples_ts`, for the same reason;
    * a row whose `samples_ts` is empty or contains a non-finite or non-positive
      value, since a zero-throughput sample means the run did not measure.
    """
    if not isinstance(text, str):
        raise TypeError("parse_llama_bench_json takes the tool's stdout as a string")
    stripped = text.strip()
    if not stripped:
        raise BenchOutputError("llama-bench produced no stdout at all")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        head = stripped.splitlines()[0][:120] if stripped.splitlines() else ""
        raise BenchOutputError(
            f"stdout is not JSON ({exc}); first line was {head!r}. `-o md`, `-o csv` and "
            f"`-o sql` all parse-fail here on purpose: they print get_fields(), which "
            f"stops at avg_ns/stddev_ns and carries NO per-repetition samples."
        ) from exc
    if not isinstance(payload, list):
        raise BenchOutputError(
            f"expected a JSON array of result rows, got {type(payload).__name__}")
    if not payload:
        raise BenchOutputError("llama-bench emitted an empty result array")

    rows: list = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise BenchOutputError(f"result row {index} is not an object")
        missing = [k for k in ("samples_ts", "avg_ts", "n_threads", "build_commit")
                   if k not in entry]
        if missing:
            raise BenchOutputError(
                f"result row {index} is missing {missing}. A row without samples_ts came "
                f"from a non-sample-bearing output format; the reduction must be "
                f"reproducible from raw samples, so there is nothing usable here.")
        samples = entry.get("samples_ts")
        if not isinstance(samples, list) or not samples:
            raise BenchOutputError(f"result row {index}: samples_ts is empty or not a list")
        values: list = []
        for sample in samples:
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise BenchOutputError(f"result row {index}: sample {sample!r} is not a number")
            value = float(sample)
            if value != value or value in (float("inf"), float("-inf")):
                raise BenchOutputError(f"result row {index}: sample {sample!r} is not finite")
            if value <= 0.0:
                raise BenchOutputError(
                    f"result row {index}: sample {value!r} tokens/s is not positive; a "
                    f"zero-throughput repetition means the run did not measure")
            values.append(value)
        ns = entry.get("samples_ns")
        if isinstance(ns, list):
            samples_ns = tuple(_row_int({"v": v}, "v", index) for v in ns)
        else:
            samples_ns = ()
        rows.append(BenchRow(
            build_commit=str(entry.get("build_commit", "")),
            n_prompt=_row_int(entry, "n_prompt", index, default=0),
            n_gen=_row_int(entry, "n_gen", index, default=0),
            n_depth=_row_int(entry, "n_depth", index, default=0),
            n_threads=_row_int(entry, "n_threads", index),
            flash_attn=_as_bool(entry.get("flash_attn", False)),
            use_mmap=_as_bool(entry.get("use_mmap", False)),
            model_filename=str(entry.get("model_filename", "")),
            n_batch=entry.get("n_batch"),
            n_ubatch=entry.get("n_ubatch"),
            avg_ts=_row_float(entry, "avg_ts", index),
            stddev_ts=_row_float(entry, "stddev_ts", index, default=0.0),
            samples_ts=tuple(values),
            samples_ns=samples_ns,
            raw=dict(entry),
        ))
    return tuple(rows)


@dataclass(frozen=True)
class LlamaBenchExpectation:
    """What the recipe said, so the OUTPUT can be checked against it.

    This is the guard against the most expensive class of benchmark defect here:
    a flag that is present in argv and did not take effect. `-fa 1` with
    `"flash_attn": false` in the result row is a real number measuring a
    different kernel, and nothing else in the pipeline would notice.

    Built from ARGV rather than from the params dict, because argv is what the
    process received.
    """

    n_prompt: int
    n_gen: int
    n_threads: int
    flash_attn: bool
    reps: int
    model_filename: str
    n_depth: Optional[int] = None
    expected_build_commit: Optional[str] = None

    @classmethod
    def from_command(cls, command: recipes.ConstructedCommand, *,
                     expected_build_commit: Optional[str] = None
                     ) -> "LlamaBenchExpectation":
        argv = list(command.argv)

        def after(flag: str) -> Optional[str]:
            return argv[argv.index(flag) + 1] if flag in argv else None

        depth = after("-d")
        return cls(
            n_prompt=int(after("-p") or 0),
            n_gen=int(after("-n") or 0),
            n_threads=int(after("-t") or 0),
            flash_attn=(after("-fa") == "1"),
            reps=int(after("-r") or 0),
            model_filename=after("-m") or "",
            n_depth=int(depth) if depth is not None else None,
            expected_build_commit=expected_build_commit,
        )

    def check_row(self, row: BenchRow) -> schemas.Check:
        if not isinstance(row, BenchRow):
            raise TypeError("check_row takes a BenchRow")
        reasons: list = []
        if row.flash_attn != self.flash_attn:
            reasons.append(
                f"argv requested flash_attn={self.flash_attn} but the result row reports "
                f"flash_attn={row.flash_attn}; the flag did not take effect and this "
                f"number measures a different kernel than the recipe names")
        if row.n_threads != self.n_threads:
            reasons.append(f"argv requested -t {self.n_threads} but the row ran with "
                           f"n_threads={row.n_threads}")
        if row.n_prompt != self.n_prompt or row.n_gen != self.n_gen:
            reasons.append(f"argv requested (-p {self.n_prompt}, -n {self.n_gen}) but the "
                           f"row is (n_prompt={row.n_prompt}, n_gen={row.n_gen}); a "
                           f"different (prompt, decode) point is a different cell")
        if self.n_depth is not None and row.n_depth != self.n_depth:
            reasons.append(f"argv requested -d {self.n_depth} but the row ran at "
                           f"n_depth={row.n_depth}")
        if self.reps and len(row.samples_ts) != self.reps:
            reasons.append(
                f"argv requested -r {self.reps} but the row carries "
                f"{len(row.samples_ts)} samples; a short sample vector means repetitions "
                f"were dropped, and the rep count is a constitutional floor")
        if self.model_filename and row.model_filename != self.model_filename:
            reasons.append(f"argv named -m {self.model_filename!r} but the row reports "
                           f"model_filename={row.model_filename!r}")
        if self.expected_build_commit:
            matched, why = commit_prefix_match(self.expected_build_commit, row.build_commit)
            if not matched:
                reasons.append(
                    f"this arm is anchored at build_commit "
                    f"{self.expected_build_commit!r} but the row reports "
                    f"{row.build_commit!r} ({why}); the binary that ran is not the one "
                    f"the arm names, so the anchor is not the anchor")
        return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons else schemas.Check(
            schemas.PASS, ("the result row agrees with the argv that produced it",))

    def to_dict(self) -> dict:
        return {"n_prompt": self.n_prompt, "n_gen": self.n_gen,
                "n_threads": self.n_threads, "flash_attn": self.flash_attn,
                "reps": self.reps, "model_filename": self.model_filename,
                "n_depth": self.n_depth,
                "expected_build_commit": self.expected_build_commit}


# =============================================================================
# Spawning
# =============================================================================

@dataclass(frozen=True)
class SpawnResult:
    """What one process invocation produced."""

    argv: tuple
    returncode: int
    stdout: str
    stderr_tail: str
    pid: Optional[int]
    duration_s: float
    timed_out: bool = False
    terminated_by_runner: bool = False

    def to_dict(self) -> dict:
        return {"argv": list(self.argv), "returncode": self.returncode,
                "stdout_sha256": hashlib.sha256(self.stdout.encode("utf-8")).hexdigest(),
                "stdout_bytes": len(self.stdout.encode("utf-8")),
                "stderr_tail": self.stderr_tail, "pid": self.pid,
                "duration_s": self.duration_s, "timed_out": self.timed_out,
                "terminated_by_runner": self.terminated_by_runner}


class ProductionTreeWrite(MicrobenchError):
    """Something was about to create files inside a frozen production tree."""


def assert_scratch_root_is_not_production(root: Optional[str]) -> Optional[str]:
    """Resolve a scratch root and REFUSE it if it touches a frozen production tree.

    `SubprocessSpawner` is the only thing in this module that CREATES anything:
    it makes a temporary directory for the tool's stdout and stderr. With no
    `dir=`, `tempfile` honours `TMPDIR` from the runner's own ambient
    environment — so `export TMPDIR=/mnt/raid0/llm/llama.cpp/tmp` in the shell
    that launches a campaign puts new directories and files inside the v8 tree
    and breaks the `git status --porcelain` byte-identity that hard boundary 1
    is stated in terms of. An env var redirecting a write into a frozen tree is
    a route, not a hypothetical: `cpu_region_claim._resolve_lock_root` was
    hardened against exactly this shape for exactly this reason.

    So the root is resolved and prefix-tested here, and `dir=` is passed
    explicitly when the caller supplies one. `realpath` is what makes it
    structural: a symlink or a `..` segment that lands in a frozen tree resolves
    before the test. `None` (the `tempfile` default) is checked too, because the
    ambient `TMPDIR` is exactly the value that would carry the redirect.
    """
    candidate = root
    if candidate is None:
        candidate = os.environ.get("TMPDIR") or tempfile.gettempdir()
    resolved = storage._norm(candidate)
    for tree in storage.production_tree_forms():
        if storage._under(resolved, tree) or storage._under(tree, resolved):
            raise ProductionTreeWrite(
                f"the spawner's scratch root resolves to {resolved!r}, which touches the "
                f"FROZEN production tree {tree!r}. This module creates a temporary "
                f"directory for the tool's stdout, and `tempfile` honours TMPDIR, so this "
                f"would have written into a tree whose working copy must stay "
                f"byte-identical. Point TMPDIR somewhere else; nothing here is allowed to "
                f"create a file in a production tree, ever.")
    return root


class Spawner(Protocol):
    """The seam that actually starts a process.

    Injectable so that argv construction, env assembly, block pairing and output
    parsing are all testable without running a benchmark — which is the whole of
    what can be verified on a contended host.
    """

    spawner_id: str

    def run(self, argv: Sequence[str], env: Mapping, *, timeout_s: float,
            cwd: Optional[str] = None) -> SpawnResult:
        ...


class SubprocessSpawner:
    """The real one. Files, not pipes; one captured pid; TERM then KILL; verify dead.

    Process discipline, point by point, because each line is a scar:

    * stdout and stderr go to temporary FILES. Never `subprocess.PIPE` — *"never
      pipe llama binaries through another process; it changes their behaviour"*,
      and a `-o json` payload would deadlock a pipe buffer anyway.
    * the env is passed EXPLICITLY, so nothing ambient leaks into the measured
      process.
    * on timeout, the runner escalates `terminate()` then `kill()` on the
      `Popen` handle it created — never `os.kill`, never a process group, never a
      name pattern. INC-20260731: a name-pattern kill took out another session's
      `llama-server` twice and killed `earlyoom`, whose own argv contains the
      names it guards.
    * after killing, it WAITS and confirms the process is reaped before
      returning. `terminated_by_runner` is recorded either way.
    """

    spawner_id = "subprocess/v1"

    def __init__(self, *, term_grace_s: float = 10.0,
                 stderr_tail_bytes: int = 4096,
                 workdir_root: Optional[str] = None) -> None:
        if term_grace_s <= 0:
            raise ValueError("term_grace_s must be positive")
        self._term_grace_s = float(term_grace_s)
        self._stderr_tail_bytes = int(stderr_tail_bytes)
        self._workdir_root = assert_scratch_root_is_not_production(workdir_root)

    def run(self, argv: Sequence[str], env: Mapping, *, timeout_s: float,
            cwd: Optional[str] = None) -> SpawnResult:
        argv = [str(token) for token in argv]
        if not argv:
            raise ValueError("argv must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        env = {str(k): str(v) for k, v in env.items()}

        started = time.monotonic()
        timed_out = False
        terminated = False
        with tempfile.TemporaryDirectory(prefix="autokernel-microbench-",
                                         dir=self._workdir_root) as workdir:
            out_path = Path(workdir, "stdout")
            err_path = Path(workdir, "stderr")
            with out_path.open("wb") as out_fh, err_path.open("wb") as err_fh:
                try:
                    proc = subprocess.Popen(argv, stdout=out_fh, stderr=err_fh,
                                            stdin=subprocess.DEVNULL, env=env, cwd=cwd)
                except OSError as exc:
                    raise SpawnFailure(f"could not start {argv[0]!r}: {exc}") from exc
                pid = proc.pid
                try:
                    returncode = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminated = True
                    returncode = self._terminate(proc)
            stdout = out_path.read_text(encoding="utf-8", errors="replace")
            stderr = err_path.read_bytes()[-self._stderr_tail_bytes:].decode(
                "utf-8", errors="replace")

        return SpawnResult(argv=tuple(argv), returncode=returncode, stdout=stdout,
                           stderr_tail=stderr, pid=pid,
                           duration_s=time.monotonic() - started, timed_out=timed_out,
                           terminated_by_runner=terminated)

    def _terminate(self, proc: "subprocess.Popen") -> int:
        """SIGTERM, then SIGKILL, then confirm reaped. Only this handle, ever."""
        proc.terminate()
        try:
            return proc.wait(timeout=self._term_grace_s)
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        try:
            return proc.wait(timeout=self._term_grace_s)
        except subprocess.TimeoutExpired as exc:
            raise SpawnFailure(
                f"pid {proc.pid} survived SIGTERM and SIGKILL and was not reaped within "
                f"{self._term_grace_s}s; refusing to report success on a process that may "
                f"still be running and still holding the claimed cores"
            ) from exc


class RecordedSpawner:
    """Replays recorded tool output instead of launching anything.

    Ships in the module rather than in the test file on purpose: it is how the
    whole pipeline — argv, env, pairing, parsing, reduction — is exercised on a
    contended host, and how the operator dry-runs a campaign before spending an
    hour of a claim on it.

    `responses` is keyed by `(arm, invocation_index)` or by `arm` alone. Every
    call records the argv and env it was handed, so a test can assert on what
    WOULD have been executed.
    """

    spawner_id = "recorded/v1"

    def __init__(self, responses: Mapping, *, default: Optional[SpawnResult] = None) -> None:
        self._responses = dict(responses)
        self._default = default
        self.calls: list = []

    def run(self, argv: Sequence[str], env: Mapping, *, timeout_s: float,
            cwd: Optional[str] = None) -> SpawnResult:
        argv = tuple(str(t) for t in argv)
        index = len(self.calls)
        arm = ARM_ANCHOR if self._looks_like(argv, ARM_ANCHOR) else ARM_CANDIDATE
        self.calls.append({"argv": argv, "env": dict(env), "timeout_s": timeout_s,
                           "cwd": cwd, "index": index, "arm": arm})
        for key in ((arm, index), index, arm):
            if key in self._responses:
                return self._materialise(self._responses[key], argv)
        if self._default is not None:
            return self._materialise(self._default, argv)
        raise SpawnFailure(f"RecordedSpawner has no recorded response for {key!r} "
                           f"(call {index}, arm {arm})")

    @staticmethod
    def _looks_like(argv: Sequence[str], arm: str) -> bool:
        return any(f"/{arm}/" in token or token.endswith(f"/{arm}") for token in argv)

    @staticmethod
    def _materialise(response: Any, argv: Sequence[str]) -> SpawnResult:
        if isinstance(response, SpawnResult):
            return response
        if isinstance(response, str):
            return SpawnResult(argv=tuple(argv), returncode=0, stdout=response,
                               stderr_tail="", pid=None, duration_s=0.0)
        raise TypeError(f"recorded response must be a SpawnResult or stdout string, "
                        f"got {type(response).__name__}")


# =============================================================================
# Pairing — the part that must be impossible to mislabel
# =============================================================================

@dataclass(frozen=True)
class BlockPlan:
    """One block's arm sequence, derived — never declared.

    `arm_sequence` is computed in `__post_init__` from `order` and `pairs`, so
    there is no constructor anywhere that accepts an arbitrary sequence. That is
    requirement 1 made structural rather than documented: the only way to get a
    non-alternating sequence into this object is to edit this class.
    """

    block_index: int
    order: str
    pairs: int
    unit_id: str
    stratum: str
    segment: str = statistics.SEGMENT_BASE
    extension_round: Optional[int] = None
    arm_sequence: tuple = ()

    def __post_init__(self) -> None:
        if self.order not in statistics.ORDERS:
            raise ValueError(f"order {self.order!r} is not one of {list(statistics.ORDERS)}")
        if isinstance(self.pairs, bool) or not isinstance(self.pairs, int) or self.pairs < 1:
            raise ValueError("pairs must be a positive int")
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, int) \
                or self.block_index < 0:
            raise ValueError("block_index must be a non-negative int")
        first = ARM_ANCHOR if self.order == statistics.ORDER_ANCHOR_FIRST else ARM_CANDIDATE
        second = ARM_CANDIDATE if first == ARM_ANCHOR else ARM_ANCHOR
        derived = tuple((first, second)[i % 2] for i in range(self.pairs * 2))
        if self.arm_sequence and tuple(self.arm_sequence) != derived:
            raise PairingViolation(
                f"block {self.block_index}: arm_sequence {list(self.arm_sequence)} was "
                f"supplied and does not equal the interleaved sequence {list(derived)} "
                f"that order={self.order!r} with {self.pairs} pair(s) requires. A blocked "
                f"design (candidate x n, then anchor x n) is forbidden: thermal and "
                f"page-cache drift alias onto the arm effect.")
        object.__setattr__(self, "arm_sequence", derived)

    @property
    def invocations(self) -> int:
        return len(self.arm_sequence)

    def to_dict(self) -> dict:
        return {"block_index": self.block_index, "order": self.order, "pairs": self.pairs,
                "unit_id": self.unit_id, "stratum": self.stratum, "segment": self.segment,
                "extension_round": self.extension_round,
                "arm_sequence": list(self.arm_sequence)}


def plan_blocks(schedule: statistics.OrderSchedule, *, count: int, pairs: int,
                unit_ids: Sequence[str], stratum: str,
                segment: str = statistics.SEGMENT_BASE,
                extension_round: Optional[int] = None) -> tuple:
    """Derive every block's order from the committed campaign seed.

    The order is never a parameter of this function. `OrderSchedule.order_for(i)`
    is prefix-stable, so appending extension blocks cannot retroactively change
    the schedule of the base blocks and turn a conforming run into a
    non-conforming one.
    """
    if not isinstance(schedule, statistics.OrderSchedule):
        raise TypeError("plan_blocks takes a statistics.OrderSchedule")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive int")
    if not unit_ids:
        raise ValueError("unit_ids must name at least one measurement-material unit")
    units = list(unit_ids)
    offset = 0 if segment == statistics.SEGMENT_BASE else schedule.base_blocks
    return tuple(
        BlockPlan(block_index=offset + i, order=schedule.order_for(offset + i), pairs=pairs,
                  unit_id=units[i % len(units)], stratum=stratum, segment=segment,
                  extension_round=extension_round)
        for i in range(count)
    )


@dataclass(frozen=True)
class Invocation:
    """One executed measurement, with everything needed to re-derive it."""

    block_index: int
    position: int
    arm: str
    receipt: ExecutionReceipt
    spawn: SpawnResult
    row: Optional[BenchRow]
    samples: tuple
    claim: ClaimAttestation
    checks: tuple

    def to_dict(self) -> dict:
        return {"block_index": self.block_index, "position": self.position, "arm": self.arm,
                "recipe": self.receipt.render(), "receipt": self.receipt.to_dict(),
                "spawn": self.spawn.to_dict(),
                "row": self.row.to_dict() if self.row is not None else None,
                "samples": list(self.samples), "claim": self.claim.to_dict(),
                "checks": [[n, {"outcome": c.outcome, "reasons": list(c.reasons)}]
                           for n, c in self.checks]}


def assemble_block(plan: BlockPlan, invocations: Sequence[Invocation], *,
                   measured_at: Optional[str] = None) -> statistics.PairedBlock:
    """Turn observed invocations into a `statistics.PairedBlock`, or refuse.

    Requirement 1, enforced three ways — any one of them alone catches a blocked
    design, and all three are kept because they fail for different reasons:

    1. the OBSERVED arm sequence must equal the plan's derived sequence;
    2. the observed sequence must STRICTLY ALTERNATE, checked independently of
       the plan so a defect in `BlockPlan` cannot let one through;
    3. the resulting `order` is re-derived from the arm that actually ran FIRST,
       never copied from the plan, so a run cannot be labelled `anchor_first`
       because the plan said so while candidate actually went first.
    """
    if not isinstance(plan, BlockPlan):
        raise TypeError("assemble_block takes a BlockPlan")
    invocations = tuple(invocations)
    if not invocations:
        raise PairingViolation(f"block {plan.block_index}: no invocations were executed")
    for inv in invocations:
        if not isinstance(inv, Invocation):
            raise TypeError("every element must be an Invocation")

    observed = tuple(inv.arm for inv in invocations)
    if observed != plan.arm_sequence:
        raise PairingViolation(
            f"block {plan.block_index}: the arms that ran were {list(observed)} but the "
            f"plan requires {list(plan.arm_sequence)}. An unpaired A-then-B run is not a "
            f"paired block and must not be recorded as one.")
    for index in range(1, len(observed)):
        if observed[index] == observed[index - 1]:
            raise PairingViolation(
                f"block {plan.block_index}: positions {index - 1} and {index} both ran "
                f"{observed[index]!r}. Candidate and anchor MUST be interleaved within "
                f"every paired block; a blocked design lets thermal and page-cache drift "
                f"alias onto the arm effect.")
    if len(observed) % 2 or observed.count(ARM_ANCHOR) != observed.count(ARM_CANDIDATE):
        raise PairingViolation(
            f"block {plan.block_index}: {observed.count(ARM_ANCHOR)} anchor and "
            f"{observed.count(ARM_CANDIDATE)} candidate invocations. A block with unequal "
            f"arms is not paired.")

    derived_order = (statistics.ORDER_ANCHOR_FIRST if observed[0] == ARM_ANCHOR
                     else statistics.ORDER_CANDIDATE_FIRST)
    if derived_order != plan.order:
        raise PairingViolation(
            f"block {plan.block_index}: the plan declares order={plan.order!r} but "
            f"{observed[0]!r} ran first. The recorded order is derived from what ran, "
            f"never from what was declared.")

    anchor: list = []
    candidate: list = []
    for inv in invocations:
        if not inv.samples:
            raise PairingViolation(
                f"block {plan.block_index} position {inv.position}: the {inv.arm} arm "
                f"produced no samples; a block with one arm missing is not a paired block")
        (anchor if inv.arm == ARM_ANCHOR else candidate).extend(inv.samples)

    return statistics.PairedBlock(
        block_index=plan.block_index, unit_id=plan.unit_id, stratum=plan.stratum,
        order=derived_order, anchor_samples=tuple(anchor),
        candidate_samples=tuple(candidate), segment=plan.segment,
        extension_round=plan.extension_round, measured_at=measured_at)


@dataclass(frozen=True)
class BlockRecord:
    """One block: its plan, its invocations, its host state, and its paired block."""

    plan: BlockPlan
    invocations: tuple
    host_state_open: HostState
    host_state_close: Optional[HostState]
    paired_block: Optional[statistics.PairedBlock]
    checks: tuple
    refusals: tuple

    @property
    def complete(self) -> bool:
        return self.paired_block is not None and not self.refusals

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "invocations": [i.to_dict() for i in self.invocations],
            "host_state_open": self.host_state_open.to_dict(),
            "host_state_close": (self.host_state_close.to_dict()
                                 if self.host_state_close is not None else None),
            "paired_block": (self.paired_block.to_list()
                             if self.paired_block is not None else None),
            "checks": [[n, {"outcome": c.outcome, "reasons": list(c.reasons)}]
                       for n, c in self.checks],
            "refusals": list(self.refusals),
            "complete": self.complete,
        }


# =============================================================================
# The run
# =============================================================================

@dataclass(frozen=True)
class MicrobenchPlan:
    """Everything a campaign needs to run one candidate against one anchor.

    There is NO `order` field and no `arm_sequence` field, by design: order is a
    function of `campaign_seed` and `candidate_id` alone, and admitting it here
    would make the schedule declarable rather than derived.

    `anchor_binding` and `anchor` are separate because the first says where the
    anchor binary IS and the second says what it must BE. `anchor.source_commit`
    is checked against the `build_commit` that `llama-bench` itself reports, so a
    rebuilt-in-place anchor is caught in the output rather than assumed away.
    """

    recipe_id: str
    candidate_id: str
    campaign_seed: str
    candidate_binding: recipes.ToolBinding
    anchor_binding: recipes.ToolBinding
    anchor: api.AnchorIdentity
    params: Mapping
    base_blocks: int
    pairs_per_block: int
    unit_ids: tuple
    stratum: str = api.STRATUM_SELECTION
    timeout_s: float = 1800.0
    attempt: int = 0

    def __post_init__(self) -> None:
        for name in ("recipe_id", "candidate_id", "campaign_seed"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"plan.{name} must be a non-empty string")
        for name in ("candidate_binding", "anchor_binding"):
            if not isinstance(getattr(self, name), recipes.ToolBinding):
                raise TypeError(f"plan.{name} must be a recipes.ToolBinding")
        if not isinstance(self.anchor, api.AnchorIdentity):
            raise TypeError("plan.anchor must be an api.AnchorIdentity — a named immutable "
                            "anchor, not a path")
        if self.stratum not in api.STRATA:
            raise ValueError(f"plan.stratum {self.stratum!r} is not one of {list(api.STRATA)}")
        for name in ("base_blocks", "pairs_per_block"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"plan.{name} must be a positive int")
        if not self.unit_ids:
            raise ValueError("plan.unit_ids must name at least one measurement-material unit")
        if self.timeout_s <= 0:
            raise ValueError("plan.timeout_s must be positive")

    def schedule(self) -> statistics.OrderSchedule:
        """The order schedule this plan implies. Derived, never declared."""
        return statistics.OrderSchedule.derive(
            campaign_seed=self.campaign_seed, candidate_id=self.candidate_id,
            base_blocks=self.base_blocks, attempt=self.attempt)

    def to_dict(self) -> dict:
        return {"recipe_id": self.recipe_id, "candidate_id": self.candidate_id,
                "candidate_binding": self.candidate_binding.to_dict(),
                "anchor_binding": self.anchor_binding.to_dict(),
                "anchor": self.anchor.short(), "params": dict(self.params),
                "base_blocks": self.base_blocks, "pairs_per_block": self.pairs_per_block,
                "unit_ids": list(self.unit_ids), "stratum": self.stratum,
                "timeout_s": self.timeout_s, "attempt": self.attempt}


@dataclass(frozen=True)
class MicrobenchRun:
    """The result of a campaign leg — complete with a number, or refused with reasons.

    `paired_blocks()` RAISES on a refused run. That is requirement 5 made
    structural: there is no accessor that hands back the blocks a contended,
    throttled or short run did manage to produce, because a partial paired-block
    set is not a smaller measurement of the same thing.
    """

    plan: MicrobenchPlan
    runner_id: str
    started_at: str
    ended_at: str
    blocks: tuple
    refusals: tuple
    checks: tuple
    scope_denominator: api.ScopeDenominator
    claim_attestations: tuple
    candidate_receipt: Optional[ExecutionReceipt]
    anchor_receipt: Optional[ExecutionReceipt]

    @property
    def order_control(self) -> schemas.Check:
        """The reducer's own order control, RE-DERIVED at access time.

        Deliberately not a value frozen into `checks` at the end of the run. The
        runner runs the schedule it derived, so a control computed once inside
        the run is a tautology; the property that is NOT a tautology is that the
        blocks in THIS object were produced under THIS object's plan. Recomputing
        here is what makes restapling a different plan onto a completed run —
        a different campaign seed, a different attempt, a different candidate id
        — fail instead of silently relabelling the run's order provenance.
        """
        emitted = [b.paired_block for b in self.blocks if b.paired_block is not None]
        return self.plan.schedule().check_observed(emitted)

    @property
    def complete(self) -> bool:
        """True only when every requested block completed AND nothing was refused."""
        return (not self.refusals
                and len(self.blocks) == self.plan.base_blocks
                and all(b.complete for b in self.blocks)
                and self.order_control.outcome == schemas.PASS)

    def paired_blocks(self) -> tuple:
        """The blocks, for the reducer. Raises `RunRefused` on anything else."""
        if not self.complete:
            raise RunRefused(
                f"this run produced no admissible number and will not pretend otherwise. "
                f"{len(self.blocks)}/{self.plan.base_blocks} blocks completed; refusals: "
                f"{list(self.refusals) or ['(none, but a block is incomplete)']}. The raw "
                f"vector is still available via raw_vector() — a failure that is not "
                f"durable is indistinguishable from a run that never happened.")
        return tuple(b.paired_block for b in self.blocks)

    def raw_vector(self) -> dict:
        """Every per-repetition reading, per invocation, per arm, per block.

        The `scope_denominator` travels WITH the samples. A full-machine gate
        computed on a partial-machine cell is a category error, and this is what
        makes it detectable after the fact rather than only at the moment the
        gate is written.
        """
        return {
            "schema": "epyc.autokernel.microbench_raw_vector.v1",
            "runner_id": self.runner_id,
            "recipe_id": self.plan.recipe_id,
            "candidate_id": self.plan.candidate_id,
            # NOT a `campaign_seed_committed: True` boolean. That field asserted
            # pre-registration on the say-so of the module emitting it, while the
            # material needed to check it — the seed, and the schedule derived
            # from it — appeared nowhere in the record, so nothing downstream
            # could ever contradict it. What ships instead is the seed's digest
            # (comparable against the committed campaign record without
            # disclosing the seed) and the derived schedule, so a reader can
            # re-run `OrderSchedule.check_observed` on the blocks below.
            "campaign_seed_sha256": hashlib.sha256(
                self.plan.campaign_seed.encode("utf-8")).hexdigest(),
            # `OrderSchedule.to_dict()` deliberately withholds the seed, so the
            # REQUIRED orders are spelled out here: a reader compares them
            # against each block's own recorded order without needing the seed,
            # and binds the pair to the committed campaign record through the
            # digest above.
            "order_schedule": dict(self.plan.schedule().to_dict(),
                                   orders=list(self.plan.schedule().orders(
                                       self.plan.base_blocks))),
            "anchor": self.plan.anchor.short(),
            "anchor_identity": self.plan.anchor.to_dict(),
            "scope_denominator": self.scope_denominator.to_dict(),
            "scope_render": self.scope_denominator.render(),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "complete": self.complete,
            "order_control": {"outcome": self.order_control.outcome,
                              "reasons": list(self.order_control.reasons)},
            "refusals": list(self.refusals),
            "checks": [[n, {"outcome": c.outcome, "reasons": list(c.reasons)}]
                       for n, c in self.checks],
            "candidate_receipt": (self.candidate_receipt.to_dict()
                                  if self.candidate_receipt is not None else None),
            "anchor_receipt": (self.anchor_receipt.to_dict()
                               if self.anchor_receipt is not None else None),
            "claim_attestations": [a.to_dict() for a in self.claim_attestations],
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def to_dict(self) -> dict:
        return self.raw_vector()


class MicrobenchRunner:
    """Runs paired blocks. The only thing in this package that executes a benchmark.

    Construction requires a claim and a spawner; neither has a default. A default
    claim would be a fail-open version of denial 8, and a default spawner would
    make "did this actually run a benchmark?" un-inspectable at the call site.
    """

    def __init__(self, *, claim: HeldClaim, spawner: Spawner,
                 policy: Optional[HostStatePolicy] = None,
                 host_state: Callable[..., HostState] = read_host_state,
                 now: Callable[[], str] = _utc_now) -> None:
        if claim is None:
            raise ClaimNotHeld(
                "MicrobenchRunner requires a held resource claim. P-AK-SEARCH-1 denial 8: "
                "'no inference run OUTSIDE A HELD CLAIM'. There is no claim=None path.")
        if spawner is None:
            raise ValueError("MicrobenchRunner requires an explicit Spawner")
        self._claim = claim
        self._spawner = spawner
        self._policy = policy if policy is not None else HostStatePolicy()
        self._read_host_state = host_state
        self._now = now

    # -- claim -------------------------------------------------------------

    def _attest_claim(self, *, when: str, cpu_list: str) -> ClaimAttestation:
        """Re-read the claim. Called before EVERY spawn, never cached.

        Two conditions, not one. The claim must be HELD, and it must COVER the
        footprint the argv pins — precondition 1 is *"a CPU region claim covering
        the exact footprint measured"*, and a claim on a region that does not
        contain the `taskset` mask leaves the measured cores unprotected while
        looking, in every journal field, exactly like a claimed run.
        """
        attestation = self._claim.attest()
        if not isinstance(attestation, ClaimAttestation):
            raise TypeError("HeldClaim.attest() must return a ClaimAttestation")
        if not attestation.held:
            raise ClaimNotHeld(
                f"the resource claim was not held {when}: "
                f"{attestation.check.outcome} — {list(attestation.check.reasons)}. "
                f"COULD_NOT_CHECK is not a pass; an unverifiable claim is not a held one.")
        try:
            measured = set(_parse_cpu_list(cpu_list))
            claimed = set(_parse_cpu_list(attestation.cpu_list))
        except ValueError as exc:
            raise ClaimNotHeld(
                f"the claim's footprint could not be compared with the argv's {when}: "
                f"{exc}. An uncomparable footprint is not a covering one.") from exc
        uncovered = sorted(measured - claimed)
        if uncovered:
            raise ClaimNotHeld(
                f"the claim held {when} covers {attestation.cpu_list!r} but the argv pins "
                f"{cpu_list!r}; cpus {uncovered[:8]}"
                f"{'...' if len(uncovered) > 8 else ''} would be measured OUTSIDE the "
                f"claim. Precondition 1 requires a claim covering the exact footprint "
                f"measured.")
        return attestation

    # -- the anchor --------------------------------------------------------

    @staticmethod
    def _check_anchor_identity(plan: MicrobenchPlan,
                               receipt: Optional[ExecutionReceipt]) -> list:
        """Compare the anchor that WILL RUN against the anchor the plan NAMES.

        `api.AnchorIdentity` is a TRIPLE — source commit, binary SHA-256, linkage
        SHA-256 — and precondition 4 is that all three are re-verified, because
        *"a rebuilt anchor is a different anchor"*. Only the first of the three
        was ever checked here, and only indirectly, against the `build_commit`
        the measured process reports about itself.

        The second is the one that costs nothing: `build_receipt()` already
        digests the anchor binary on the way past. Not comparing it meant an
        `anchor_binding` pointing at an arbitrary second candidate build was
        admissible as "the anchor" — the `anchor` arm is exempt from
        `recipes._assert_arm_allows_binding` by design (reading the frozen tree
        is not a write), so nothing else constrains where it points.

        The third, `linkage_sha256`, cannot be computed without resolving the
        binary's dynamic libraries, which is `verify_ggml_linkage.sh` and belongs
        to the T0 provider. It is emitted as an explicit COULD_NOT_CHECK naming
        its delegate rather than omitted — an unverified conjunct that appears
        nowhere is indistinguishable from one that passed.
        """
        if receipt is None:
            return [("anchor_identity.binary_sha256", schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("the anchor binary could not be digested, so the anchor identity could "
                 "not be re-verified",)))]
        checks: list = []
        if receipt.binary_sha256 != plan.anchor.binary_sha256:
            checks.append(("anchor_identity.binary_sha256", schemas.Check(schemas.FAIL, (
                f"the plan names anchor binary {plan.anchor.binary_sha256[:12]} but "
                f"{receipt.binary_path} digests to {receipt.binary_sha256[:12]}. A rebuilt "
                f"anchor is a different anchor, and an anchor binding is exempt from the "
                f"production-tree refusal precisely because it is supposed to BE the named "
                f"immutable anchor.",))))
        else:
            checks.append(("anchor_identity.binary_sha256", schemas.Check(schemas.PASS, (
                f"the anchor binary at {receipt.binary_path} digests to the "
                f"{plan.anchor.binary_sha256[:12]} the plan names",))))
        checks.append(("anchor_identity.linkage_sha256", schemas.Check(
            schemas.COULD_NOT_CHECK, (
                f"anchor linkage {plan.anchor.linkage_sha256[:12]} was NOT re-verified "
                f"here: resolving the binary's dynamic libraries is "
                f"`verify_ggml_linkage.sh`, wired in execution/t0_provider.py. Recorded "
                f"rather than omitted — three trees run three ggml generations and a "
                f"binary that inherits another tree's ggml runs silently wrong.",))))
        return checks

    def _attest_binary(self, *, arm: str, command: recipes.ConstructedCommand,
                       receipt: ExecutionReceipt, when: str) -> None:
        """Re-digest the binary before EVERY spawn, not once per campaign.

        The receipt's whole argument for carrying a binary digest is that *"a
        path is not an identity on a host where the experimental tree is rebuilt
        between blocks"* — and this loop's entire purpose is to rebuild
        candidates. A digest taken once at run open and stamped onto eight
        invocations spanning an hour asserts of invocations 2..8 something that
        was only observed of invocation 1.
        """
        try:
            digest = integrity.sha256_file(command.binding.binary)
        except (OSError, integrity.IntegrityError) as exc:
            raise SpawnFailure(
                f"the {arm} binary {command.binding.binary} could not be digested "
                f"{when}: {exc}") from exc
        if digest != receipt.binary_sha256:
            raise SpawnFailure(
                f"the {arm} binary at {command.binding.binary} changed {when}: the receipt "
                f"was taken over {receipt.binary_sha256[:12]} and the file now digests to "
                f"{digest[:12]}. A rebuild mid-campaign makes every earlier block a "
                f"measurement of a different binary under one receipt.")

    # -- the run -----------------------------------------------------------

    def run(self, plan: MicrobenchPlan) -> MicrobenchRun:
        if not isinstance(plan, MicrobenchPlan):
            raise TypeError("run() takes a MicrobenchPlan")
        started_at = self._now()
        checks: list = []
        refusals: list = []
        attestations: list = []
        blocks: list = []

        commands = {
            ARM_CANDIDATE: recipes.construct(plan.recipe_id, binding=plan.candidate_binding,
                                             params=plan.params, arm=ARM_CANDIDATE,
                                             verify_inputs=False),
            ARM_ANCHOR: recipes.construct(plan.recipe_id, binding=plan.anchor_binding,
                                          params=plan.params, arm=ARM_ANCHOR,
                                          verify_inputs=False),
        }
        scope = commands[ARM_CANDIDATE].scope_denominator
        footprint = commands[ARM_CANDIDATE].claim_footprint

        envs: dict = {}
        receipts: dict = {}
        expectations: dict = {}
        for arm, command in commands.items():
            assembly = assemble_env(command.env)
            envs[arm] = assembly.env
            discipline = check_recipe_discipline(command, assembly.env)
            checks.append((f"recipe_discipline.{arm}", discipline))
            if discipline.outcome != schemas.PASS:
                refusals.append(f"{arm}: {'; '.join(discipline.reasons)}")
            try:
                receipts[arm] = build_receipt(command, env=assembly.env)
            except (OSError, integrity.IntegrityError) as exc:
                refusals.append(f"{arm}: cannot digest the binary that would run: {exc}")
            expectations[arm] = LlamaBenchExpectation.from_command(
                command,
                expected_build_commit=(plan.anchor.source_commit if arm == ARM_ANCHOR
                                       else None))
            for finding in command.discipline:
                if finding.check.outcome == schemas.PASS:
                    continue
                checks.append((f"delegated.{arm}.{finding.finding_id}", finding.check))
            # `verify_inputs=False` above turns every input check into one
            # COULD_NOT_CHECK. That is the right call at construction — the
            # binary's digest is taken here anyway and a missing model fails the
            # tool loudly — but `command.inputs_verified` being False must APPEAR
            # in the record. An unverified conjunct that appears nowhere is
            # indistinguishable from one that passed.
            if not command.inputs_verified:
                checks.append((f"inputs_verified.{arm}", schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    tuple(r for c in command.input_checks for r in c.reasons))))

        anchor_checks = self._check_anchor_identity(plan, receipts.get(ARM_ANCHOR))
        checks.extend(anchor_checks)
        for name, check in anchor_checks:
            if check.outcome == schemas.FAIL:
                refusals.append(f"{name}: {'; '.join(check.reasons)}")

        # Host state at OPEN. Contention is judged here and only here: once the
        # benchmark is running it saturates the claimed cores itself, so a
        # mid-run load reading measures this runner, not a foreign process.
        open_state = self._read_host_state(cpu_list=footprint.cpu_list)
        freq_check = self._policy.check_frequency(open_state)
        load_check = self._policy.check_load(open_state, cpu_count=footprint.cpu_count)
        checks.append(("host_frequency_open", freq_check))
        checks.append(("host_load_open", load_check))
        for name, check in (("frequency", freq_check), ("contention", load_check)):
            if check.outcome != schemas.PASS:
                refusals.append(f"host {name} at run open: {check.outcome} — "
                                f"{'; '.join(check.reasons)}")

        if refusals:
            return self._finish(plan, started_at, blocks, refusals, checks, scope,
                                attestations, receipts)

        schedule = statistics.OrderSchedule.derive(
            campaign_seed=plan.campaign_seed, candidate_id=plan.candidate_id,
            base_blocks=plan.base_blocks, attempt=plan.attempt)
        plans = plan_blocks(schedule, count=plan.base_blocks, pairs=plan.pairs_per_block,
                            unit_ids=plan.unit_ids, stratum=plan.stratum)

        for block_plan in plans:
            record = self._run_block(plan, block_plan, commands, envs, receipts,
                                     expectations, footprint, attestations)
            blocks.append(record)
            if not record.complete:
                refusals.extend(record.refusals)
                break

        return self._finish(plan, started_at, blocks, refusals, checks, scope,
                            attestations, receipts)

    def _run_block(self, plan: MicrobenchPlan, block_plan: BlockPlan, commands: Mapping,
                   envs: Mapping, receipts: Mapping, expectations: Mapping,
                   footprint: recipes.ClaimFootprint, attestations: list) -> BlockRecord:
        open_state = self._read_host_state(cpu_list=footprint.cpu_list)
        checks: list = [("host_frequency_block_open",
                         self._policy.check_frequency(open_state))]
        refusals: list = []
        invocations: list = []

        if checks[0][1].outcome != schemas.PASS:
            return BlockRecord(plan=block_plan, invocations=(), host_state_open=open_state,
                               host_state_close=None, paired_block=None,
                               checks=tuple(checks),
                               refusals=(f"block {block_plan.block_index}: host frequency "
                                         f"{checks[0][1].outcome} — "
                                         f"{'; '.join(checks[0][1].reasons)}",))

        for position, arm in enumerate(block_plan.arm_sequence):
            try:
                attestation = self._attest_claim(
                    when=f"before block {block_plan.block_index} position {position} "
                         f"({arm} arm)",
                    cpu_list=footprint.cpu_list)
            except ClaimNotHeld as exc:
                refusals.append(str(exc))
                break
            attestations.append(attestation)

            command = commands[arm]
            # Everything from here to the parse is driven by material this runner
            # does not control — the filesystem, the process, and the tool's own
            # stdout. A `MicrobenchError` raised out of `run()` would take the
            # whole campaign's raw vector with it, and "a failure that is not
            # durable is indistinguishable from a run that never happened". So
            # the loop's own errors become refusals; a `TypeError` from a Spawner
            # that breaks its contract still raises, because that is a defect in
            # the caller's code and not a fact about the host.
            try:
                self._attest_binary(
                    arm=arm, command=command, receipt=receipts[arm],
                    when=f"before block {block_plan.block_index} position {position}")
                spawn = self._spawner.run(command.argv, envs[arm],
                                          timeout_s=plan.timeout_s)
            except MicrobenchError as exc:
                refusals.append(f"block {block_plan.block_index} position {position} "
                                f"({arm}): {exc}")
                break
            except OSError as exc:
                refusals.append(f"block {block_plan.block_index} position {position} "
                                f"({arm}): the spawn failed with {type(exc).__name__}: "
                                f"{exc}")
                break
            if not isinstance(spawn, SpawnResult):
                raise TypeError(f"Spawner.run() must return a SpawnResult, got "
                                f"{type(spawn).__name__}")
            inv_checks: list = []
            row = None
            samples: tuple = ()

            if spawn.timed_out:
                refusals.append(
                    f"block {block_plan.block_index} position {position} ({arm}): the "
                    f"process exceeded {plan.timeout_s}s and was terminated by this "
                    f"runner; a truncated run is not a short one")
            elif spawn.returncode != 0:
                refusals.append(
                    f"block {block_plan.block_index} position {position} ({arm}): exit "
                    f"{spawn.returncode}; stderr tail: {spawn.stderr_tail[-400:]!r}")
            else:
                try:
                    rows = parse_llama_bench_json(spawn.stdout)
                except MicrobenchError as exc:
                    refusals.append(f"block {block_plan.block_index} position {position} "
                                    f"({arm}): {exc}")
                    rows = ()
                if len(rows) > 1:
                    refusals.append(
                        f"block {block_plan.block_index} position {position} ({arm}): the "
                        f"tool emitted {len(rows)} result rows. Each row is a different "
                        f"cell and they must not share one record; the recipe constructs "
                        f"exactly one (n_prompt, n_gen) point per invocation.")
                elif rows:
                    row = rows[0]
                    agreement = expectations[arm].check_row(row)
                    inv_checks.append(("output_matches_recipe", agreement))
                    if agreement.outcome != schemas.PASS:
                        refusals.append(f"block {block_plan.block_index} position "
                                        f"{position} ({arm}): "
                                        f"{'; '.join(agreement.reasons)}")
                    else:
                        samples = row.metric_samples

            invocations.append(Invocation(
                block_index=block_plan.block_index, position=position, arm=arm,
                receipt=receipts[arm], spawn=spawn, row=row, samples=samples,
                claim=attestation, checks=tuple(inv_checks)))

            if refusals:
                break

        close_state = self._read_host_state(cpu_list=footprint.cpu_list)
        close_freq = self._policy.check_frequency(close_state)
        checks.append(("host_frequency_block_close", close_freq))
        if close_freq.outcome != schemas.PASS:
            refusals.append(
                f"block {block_plan.block_index}: host frequency at block close is "
                f"{close_freq.outcome} — {'; '.join(close_freq.reasons)}; a throttle that "
                f"developed during the block invalidates the block it developed in")

        paired = None
        if not refusals:
            try:
                paired = assemble_block(block_plan, invocations,
                                        measured_at=close_state.observed_at)
            except PairingViolation as exc:
                refusals.append(str(exc))

        return BlockRecord(plan=block_plan, invocations=tuple(invocations),
                           host_state_open=open_state, host_state_close=close_state,
                           paired_block=paired, checks=tuple(checks),
                           refusals=tuple(refusals))

    def _finish(self, plan: MicrobenchPlan, started_at: str, blocks: list, refusals: list,
                checks: list, scope: api.ScopeDenominator, attestations: list,
                receipts: Mapping) -> MicrobenchRun:
        if len(blocks) < plan.base_blocks and not refusals:
            refusals.append(
                f"only {len(blocks)}/{plan.base_blocks} paired blocks completed; a run "
                f"short of its declared block count does not emit a number")
        run = MicrobenchRun(
            plan=plan, runner_id=RUNNER_ID, started_at=started_at, ended_at=self._now(),
            blocks=tuple(blocks), refusals=tuple(refusals), checks=tuple(checks),
            scope_denominator=scope, claim_attestations=tuple(attestations),
            candidate_receipt=receipts.get(ARM_CANDIDATE),
            anchor_receipt=receipts.get(ARM_ANCHOR))
        # The reducer's own order control is NOT copied into `checks` here.
        # `MicrobenchRun.order_control` re-derives it from the plan on every
        # read, `complete` is conjoined with it, and `raw_vector()` emits it —
        # one computation site, deliberately. A frozen copy taken at the end of
        # the run would be a second thing to stub, and inside this runner it can
        # only ever say PASS anyway: the loop runs the schedule it derived, so
        # the tautological verdict is worth nothing and the recomputed one is
        # what stops a completed run being relabelled with another plan.
        if run.order_control.outcome != schemas.PASS and not refusals \
                and len(blocks) == plan.base_blocks:
            refusals.append(
                f"the emitted blocks do not satisfy the order schedule derived from "
                f"the committed campaign seed: {run.order_control.outcome} — "
                f"{'; '.join(run.order_control.reasons)}")
            return replace(run, refusals=tuple(refusals))
        return run


# =============================================================================
# Structural self-audit
# =============================================================================

#: Every spelling of "find a process by what it is called and signal it". The
#: origin is INC-20260731-broad-process-pattern-kills: a `pkill`-style name match
#: killed another agent's `llama-server` twice, and killed `earlyoom`, because a
#: guard process's argv necessarily contains the names of the things it guards.
_FORBIDDEN_PROCESS_NAMES = frozenset({
    "pkill", "pgrep", "killall", "kill", "system", "popen", "spawnl", "spawnv",
    "spawnlp", "spawnvp", "execv", "execvp", "process_iter", "pidof",
})

#: `os.kill` and `signal.*` reach any pid, including pids this loop never
#: launched. Termination here goes through `Popen.terminate` / `Popen.kill`,
#: which can only reach the handle this module itself created.
_FORBIDDEN_ATTRIBUTES = frozenset({"kill", "killpg", "system", "popen", "psutil"})


#: Callables that START something. A string is dangerous only when it reaches
#: one of these — everywhere else it is prose.
_LAUNCHER_CALLEES = frozenset({
    "Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput",
    "system", "spawnl", "spawnv", "spawnlp", "spawnvp", "execv", "execvp", "execl",
})


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _launcher_string_arguments(tree: ast.AST) -> list:
    """Every string literal handed to something that starts a process.

    Scoping the scan to LAUNCHER CALLS is the difference between a guard and a
    guard that forbids its own idiom. This module must be able to NAME `pkill` —
    in `_FORBIDDEN_PROCESS_NAMES`, in its docstrings, in the incident reference —
    while being unable to RUN it. Scanning every string constant would flag the
    constant that defines the ban, and a guard that fails on its own vocabulary
    gets deleted within the week. So: a string matters when it reaches a
    launcher, and nowhere else.

    WITHIN a launcher call the scan is TOTAL — every string constant anywhere in
    the call's subtree, `f"..."` segments included. The narrower version walked
    only the call's direct arguments plus one level of list/tuple and looked only
    at `ast.Constant`, so `Popen([f"pkill", "-f", name])` audited clean:
    CPython parses `f"pkill"` as a `JoinedStr` wrapping the constant, not as a
    constant, and `Popen(["sh"] + [cmd])` hides the elements behind a `BinOp`. A
    guard that any spelling of its own target walks past is not a guard.
    """
    found: list = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _callee_name(node) not in _LAUNCHER_CALLEES:
            continue
        for operand in list(node.args) + [kw.value for kw in node.keywords]:
            for inner in ast.walk(operand):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    found.append(inner.value)
    return found


def _shell_true_launchers(tree: ast.AST) -> list:
    """Launcher calls with `shell=True` — a shell turns any string into a pattern."""
    found: list = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _callee_name(node) not in _LAUNCHER_CALLEES:
            continue
        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) \
                    and kw.value.value is True:
                found.append(_callee_name(node))
    return found


def audit_no_name_pattern_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from the AST that this module cannot find a process by name.

    A comment promising process discipline is worth nothing; this walks the
    module's own syntax tree. It permits `Popen.terminate()` and `Popen.kill()`
    on the handle this module created, and refuses `os.kill`, `os.killpg`,
    `os.system`, the `signal` module, `psutil`, and any process-matching utility
    named in a string that is handed to a call.
    """
    if source is None:
        source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    reasons: list = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ("psutil", "signal"):
                    reasons.append(f"imports {alias.name!r}: it can signal a pid this "
                                   f"module never launched")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in ("psutil", "signal"):
                reasons.append(f"imports from {node.module!r}")
            elif root == "os":
                # `from os import kill` unbinds the name from its module, and the
                # `os.kill` attribute check below then never sees it. The bare
                # call `kill(pid, 9)` reaches any pid on the host.
                for alias in node.names:
                    if alias.name in _FORBIDDEN_ATTRIBUTES or \
                            alias.name in _FORBIDDEN_PROCESS_NAMES:
                        reasons.append(
                            f"imports {alias.name!r} from os, which unbinds it from the "
                            f"module and puts a call that can reach ANY pid one bare name "
                            f"away; termination must go through the Popen handle this "
                            f"module created")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("os", "signal") \
                    and node.attr in _FORBIDDEN_ATTRIBUTES:
                reasons.append(
                    f"uses {node.value.id}.{node.attr}, which can reach any pid; "
                    f"termination must go through the Popen handle this module created")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("kill", "killpg", "system"):
            # A BARE name only. `proc.kill()` is an ast.Attribute and stays
            # permitted — it is the one termination path this module is allowed,
            # and a guard that forbids its own compliant idiom gets deleted.
            reasons.append(
                f"calls {node.func.id}() as a bare name; that is a module-level function "
                f"reaching an arbitrary pid, not Popen.{node.func.id}() on a handle this "
                f"module created")

    for literal in _launcher_string_arguments(tree):
        for token in re.split(r"[\s;|&]+", literal.strip().lower()):
            base = token.rsplit("/", 1)[-1].strip("'\"`")
            if base in _FORBIDDEN_PROCESS_NAMES:
                reasons.append(
                    f"hands {base!r} to a process launcher; INC-20260731 is exactly this, "
                    f"and a name pattern on a shared host is a wildcard over other "
                    f"sessions' processes — including guard processes whose own argv "
                    f"contains the names they guard")
    for callee in _shell_true_launchers(tree):
        reasons.append(f"calls {callee} with shell=True; a shell turns any argument into a "
                       f"pattern and makes the argv un-auditable")

    if reasons:
        return schemas.Check(schemas.FAIL, tuple(sorted(set(reasons))))
    return schemas.Check(schemas.PASS, (
        "no name-pattern process lookup, no os.kill/killpg, no signal module, no psutil; "
        "the only termination path is Popen.terminate()/kill() on a handle this module "
        "created itself",))
