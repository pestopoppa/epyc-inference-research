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

THE EXTENSION ROUND IS *EXTENDED*, NOT RE-DERIVED — AND WHY
-----------------------------------------------------------
For the CPU decode cell the calibration solves `B_min = 5` and `threshold = 10`,
and the sign-martingale e-value over 5 same-sign blocks tops out at **5.5687**
— the statistic is the SIGN of each block's effect, so the magnitude never
enters and a candidate at a true factor of 3.0 returns the same 5.5687 as one at
1.08. No candidate can cross on the base segment, whatever its real effect.
Every win therefore comes from the declared extension round (10 same-sign blocks
reach 42.29), which makes the question below load-bearing rather than academic:

    when a run extends, is the order schedule RE-DERIVED for the extension
    round, or EXTENDED from the base segment across two runner invocations?

**It is EXTENDED.** One `OrderSchedule`, whose `base_blocks` is `B_min`, indexed
straight through the base segment and every extension round. The argument is in
the code, not in a preference:

1. `OrderSchedule.order_for` has an explicit `index >= base_blocks` limb that
   FLIPS the base order — *"extension = fresh REVERSED-order pairs"*. Under
   re-derivation that limb is unreachable: a schedule re-derived for a 5-block
   round is only ever asked for indices 0..4, so the extension would repeat the
   base orders exactly. `statistics.BoundedExtension` refuses any `order` but
   `"reversed"`, so re-derivation contradicts the only extension the protocol
   can represent.
2. `plan_blocks` already offsets extension blocks by `schedule.base_blocks`: it
   takes the BASE schedule and continues its index line. It never re-derives.
3. `statistics.SequentialEvaluation` — the rule's own driver, the thing that
   TELLS a caller what the next block must be — refuses an `order_schedule`
   whose `base_blocks != b_min`, and issues `order_for(len(blocks))` straight
   through the extension. A re-derived, longer schedule is not constructible
   there at all; `CampaignStatistics.order_schedule()` does not even accept a
   block count.
4. `statistics._check_block_identity` requires the pooled submission to be
   `block_index == position` over 0..n-1. Base and extension are one index line
   by the time the reducer sees them, so they must be one index line when they
   are produced.
5. Prefix-stability is argued in `OrderSchedule`'s own docstring as *"adding
   extension blocks cannot retroactively change the schedule of the base
   blocks"*. That sentence is about ONE schedule being extended; under
   re-derivation there is nothing for it to be retroactive about.

The statistical reason underneath: the order assignment must be a pre-committed
function of (campaign seed, candidate, block index) fixed before any data is
seen. Extending evaluates that function at further indices. Re-deriving would
introduce a SECOND schedule whose `base_blocks` parameter depends on how many
blocks were actually run — which depends on the data — so the randomization
scheme itself would become data-dependent, and it would silently drop the
reversal. `MicrobenchPlan.extend()` therefore carries the base plan's
`campaign_seed`, `candidate_id`, `attempt` and `base_blocks` across unchanged,
`ExtensionAuthorization` refuses a `base_blocks` that disagrees with the rule's,
and `assemble_run_blocks()` refuses to pool two runs that are not the same
schedule (`ScheduleMismatch`). A mismatch is a hard error, never a relabel.

AND THE EXTENSION IS DECLARED, NOT GRANTED AFTER THE FACT
--------------------------------------------------------
Anytime-validity is the whole justification for the e-process, and it survives
looking but not rule-changing. So an extension round here is not something a
caller can hand itself: `ExtensionAuthorization` takes the CAMPAIGN — one
`statistics.CampaignStatistics` — and reads `max_rounds`, `blocks_per_round`,
the ceiling and the base length off it. Round `max_rounds + 1` cannot be
constructed, a round that would carry the run past `max_blocks_per_candidate`
cannot be constructed, and a plan with `segment="extension"` and no
authorization is refused at construction (`ExtensionNotDeclared`). There is no
unbounded extension because there is no way to say one.

It takes the campaign and NOT a `(StoppingRule, StoppingRuleCommitment)` pair,
which is what it used to take. `commitment.verify(rule)` compares the two
arguments to EACH OTHER, so a caller that wanted a rule the campaign never
committed did not have to mutate anything: it built the rule it wanted and
called `StoppingRuleCommitment.commit()` on it, and the verification passed by
construction. Red-teamed on 2026-08-04 — a licence for round 3 of a
`max_rounds=3` rule with a ceiling of 100, campaign id `"not-even-this-
campaign"` and `committed_at` in 2099, constructed cleanly, and rounds 1..3
were SPAWNED (real benchmark minutes, on a held claim) before anything refused
them. `CampaignStatistics` is the object the REDUCER already reads, it verifies
its own commitment at construction, and it additionally requires an accepted
calibration solved for this cell — so binding the licence to it is what makes
"the rule that licensed this round is the rule this evidence is reduced under"
a checkable fact instead of a pair of caller-supplied objects agreeing with
themselves.

Binding it at construction is necessary and not sufficient, because a caller can
build a second campaign. `assemble_run_blocks()` therefore takes the campaign
too, and refuses a round whose licence names a different one — that is the seam
where the pooled evidence is built, and a round licensed by another campaign
must not reach the reducer wearing this campaign's record.

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
import math
import os
import re
import subprocess
import statistics as python_statistics
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .. import journal as journal_module
from .. import schemas, storage
from ..evaluator import api, devices, integrity, recipes, statistics
from . import device_sampler as gpu_device_sampler
from . import instrument_integrity
from . import physical_bounds
from . import sandbox as process_sandbox


# A synchronized twin that is more than 20% slower in the median is not used as
# a corrected performance result. It is an integrity failure: the ordinary arm
# returned before device work was complete. The threshold is deliberately a
# large-divergence screen, not a noise-floor or speed-claim threshold.
STREAM_ESCAPE_MAX_MEDIAN_GAP_FRACTION = 0.20

# llama-bench emits tokens/s for both live campaign recipes. The envelope must
# use the same delivered unit; matching only the shape id would still allow a
# ceiling derived for output rows or requests to grade a token/s vector.
_DELIVERED_UNIT_BY_METRIC = {
    "decode_tokens_per_s": "token",
    "prefill_tokens_per_s": "token",
}

__all__ = [
    # identity
    "RUNNER_ID", "TESTDATA_DIR",
    # errors
    "MicrobenchError", "ClaimNotHeld", "PairingViolation", "BenchOutputError",
    "RecipeOutputMismatch", "RunRefused", "SpawnFailure", "HostStateUnreadable",
    "ExtensionNotDeclared", "ScheduleMismatch", "RunLedgerRequired",
    "RunAlreadyCompleted", "RunNotJournaled", "RunIdentityMismatch",
    "STREAM_ESCAPE_MAX_MEDIAN_GAP_FRACTION",
    # claim seam
    "HeldClaim", "ClaimAttestation", "CpuRegionClaimAdapter",
    # host state
    "HostState", "HostStatePolicy", "read_host_state", "DEFAULT_BASE_ENV_KEYS",
    "PackagePowerAttestation", "derive_package_power_attestation",
    "FREQUENCY_JUDGED", "FREQUENCY_UNEVALUABLE", "FREQUENCY_DEFERRED_IDLE",
    "FREQUENCY_CLASSIFICATIONS",
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
    "ExtensionAuthorization", "MicrobenchPlan", "MicrobenchRun", "CompletedRunLedger",
    "MicrobenchRunner",
    "assemble_run_blocks",
    # self-audit
    "audit_no_name_pattern_process_paths",
]

#: Identity of this runner. It goes into every receipt and every raw vector, so a
#: record naming it names the exact execution semantics that produced its samples.
RUNNER_ID = "autokernel.execution.microbench/v1"

TESTDATA_DIR = Path(__file__).resolve().parent / "testdata"

ARM_ANCHOR = "anchor"
ARM_CANDIDATE = "candidate"

#: Bound, not re-compiled — the digest shape has one owner. See `schemas.require`.
_SHA256_RE = schemas.SHA256_RE


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


class ExtensionNotDeclared(MicrobenchError):
    """An extension round the pre-committed stopping rule does not license.

    *"Extension follows the declared rule only … Any post-hoc change to the
    stopping rule voids every affected record."* Raised rather than returned,
    and raised at CONSTRUCTION of the authorization, so a plan for an undeclared
    round never comes into existence and therefore never reaches a spawn. The
    three ways to earn it: a round beyond the declared `max_rounds`, a round
    whose blocks would carry the run past `max_blocks_per_candidate`, and a
    `StoppingRule` that no longer hashes to the commitment made at campaign
    start — which is what "the caller granted itself an extension after seeing
    the result" actually looks like in a diff.
    """


class ScheduleMismatch(MicrobenchError):
    """Base and extension were produced under two different order schedules.

    The extension round EXTENDS the base segment's schedule; it does not
    re-derive one (see the module docstring). Two runs that do not agree on
    `(campaign_seed, candidate_id, attempt, base_blocks)` — or on the recipe,
    params, anchor and bindings that make them one instrument — are not one
    candidate's blocks, and pooling them to a pre-declared threshold would pool
    two experiments. A hard error, never a relabel.
    """


class RunLedgerRequired(MicrobenchError):
    """A durable run ledger is required for a path that can spend a round."""


class RunAlreadyCompleted(MicrobenchError):
    """This campaign/candidate/attempt/segment key already has a completed run."""


class RunNotJournaled(MicrobenchError):
    """Pooling was asked to consume a run the durable ledger does not contain."""


class RunIdentityMismatch(MicrobenchError):
    """The run supplied for pooling is not the run journaled under its key."""


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
    uptime_s: Optional[float] = None
    monotonic_s: Optional[float] = None
    package_by_cpu: tuple = ()
    package_energy_uj: tuple = ()

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
            "uptime_s": self.uptime_s,
            "monotonic_s": self.monotonic_s,
            "package_by_cpu": [[cpu, package] for cpu, package in self.package_by_cpu],
            "package_energy_uj": [
                {"package": package, "energy_uj": energy,
                 "max_energy_range_uj": maximum, "source": source}
                for package, energy, maximum, source in self.package_energy_uj
            ],
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


def _read_package_topology(*, cpus: Sequence[int], sysfs: Path,
                           unreadable: list) -> tuple:
    """Return CPU/package bindings and the distinct packages they name."""
    package_by_cpu: list[tuple[int, int]] = []
    packages: set[int] = set()
    for cpu in cpus:
        package = _read_int_file(sysfs / f"cpu{cpu}" / "topology"
                                 / "physical_package_id")
        if package is None:
            unreadable.append(f"cpu{cpu}: physical_package_id unreadable")
            continue
        package_by_cpu.append((cpu, package))
        packages.add(package)
    return tuple(package_by_cpu), tuple(sorted(packages))


def _read_package_energy(*, cpus: Sequence[int], sysfs: Path,
                         powercap_root: Path, unreadable: list) -> tuple:
    """Return ``(package_by_cpu, counters)`` from sysfs without a process probe.

    The counter is package-wide; a partition receipt therefore names both the
    claimed CPU list and the shared package.  It never calls the result
    lane-exclusive power.  That distinction is what AK-LN-3's cross-lane A/A
    can test rather than assume away.
    """
    package_by_cpu, packages = _read_package_topology(
        cpus=cpus, sysfs=sysfs, unreadable=unreadable)

    domains: dict[int, Path] = {}
    try:
        name_files = tuple(powercap_root.rglob("name"))
    except OSError:
        name_files = ()
    for name_path in name_files:
        try:
            name = name_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        match = re.fullmatch(r"package-(\d+)", name)
        if match:
            domains[int(match.group(1))] = name_path.parent

    counters: list[tuple[int, int, int, str]] = []
    for package in packages:
        domain = domains.get(package)
        if domain is None:
            unreadable.append(f"package{package}: powercap energy domain unavailable")
            continue
        energy = _read_int_file(domain / "energy_uj")
        maximum = _read_int_file(domain / "max_energy_range_uj")
        if energy is None or maximum is None or maximum <= 0:
            unreadable.append(f"package{package}: powercap energy counter unreadable")
            continue
        counters.append((package, energy, maximum, str(domain / "energy_uj")))
    return package_by_cpu, tuple(counters)


def read_host_state(*, cpu_list: str, sysfs_root: Any = "/sys/devices/system/cpu",
                    proc_root: Any = "/proc",
                    powercap_root: Any = "/sys/devices/virtual/powercap",
                    package_energy_reader: Optional[Callable[..., tuple]] = None,
                    now: Callable[[], str] = _utc_now,
                    monotonic: Callable[[], float] = time.monotonic) -> HostState:
    """Read per-cpu scaling frequency and 1-minute load for the claimed footprint.

    `sysfs_root` and `proc_root` are injectable so the throttle guard can be
    tested against a synthesised sysfs — a guard that can only be exercised on a
    genuinely throttled machine is a guard that is never exercised.

    ``package_energy_reader`` is the narrow privileged seam for hosts whose
    package counters are root-readable only. It receives the exact package ids
    derived from the claimed CPUs and returns ``(package, energy_uj,
    max_energy_range_uj, source)`` tuples. Frequency, load and topology remain
    direct host reads; a broker cannot replace or fabricate them.

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

    uptime_s: Optional[float] = None
    try:
        uptime_s = float(Path(proc_root, "uptime").read_text(encoding="utf-8").split()[0])
        if not math.isfinite(uptime_s) or uptime_s < 0:
            raise ValueError("invalid uptime")
    except (OSError, ValueError, IndexError):
        uptime_s = None

    if package_energy_reader is None:
        package_by_cpu, package_energy = _read_package_energy(
            cpus=cpus, sysfs=sysfs, powercap_root=Path(powercap_root),
            unreadable=unreadable)
    else:
        package_by_cpu, packages = _read_package_topology(
            cpus=cpus, sysfs=sysfs, unreadable=unreadable)
        try:
            package_energy = tuple(package_energy_reader(packages=packages))
        except Exception as exc:  # noqa: BLE001 - unreadable is evidence, not a crash
            package_energy = ()
            unreadable.append(
                f"package power broker unreadable: {type(exc).__name__}: {exc}")
    monotonic_s = float(monotonic())
    if not math.isfinite(monotonic_s) or monotonic_s < 0:
        raise HostStateUnreadable("monotonic host-state timestamp is invalid")

    return HostState(
        observed_at=now(), cpu_list=cpu_list, khz_by_cpu=tuple(readings),
        driver_min_khz=driver_min, driver_max_khz=driver_max, load1=load1,
        source=str(sysfs), unreadable=tuple(unreadable), uptime_s=uptime_s,
        monotonic_s=monotonic_s, package_by_cpu=package_by_cpu,
        package_energy_uj=package_energy)


#: How `check_frequency` arrived at its outcome. Three CLASSIFICATIONS, which are
#: not the same axis as the three OUTCOMES: `JUDGED` says the reading was
#: compared against the healthy reference (outcome PASS or FAIL), `UNEVALUABLE`
#: says the comparison could not be set up at all (disabled, unreadable, no
#: reference), and `DEFERRED_IDLE` says the comparison was well-formed but the
#: host was not under enough load for a clock reading to carry information.
#:
#: Callers need the classification because they must treat the two
#: COULD_NOT_CHECKs differently, and telling them apart by matching on the reason
#: PROSE is the kind of guard that dies at the first rewording. See
#: `frequency_verdict`.
FREQUENCY_JUDGED = "judged"
FREQUENCY_UNEVALUABLE = "unevaluable"
FREQUENCY_DEFERRED_IDLE = "deferred_idle"

FREQUENCY_CLASSIFICATIONS = (FREQUENCY_JUDGED, FREQUENCY_UNEVALUABLE,
                             FREQUENCY_DEFERRED_IDLE)

# The ratified canonical boost gate is 80 of 96 cores at or above 2.5 GHz.
# Expressed as a ratio so a narrower declared footprint has a reachable gate.
BOOST_THRESHOLD_KHZ = 2_500_000
BOOST_MIN_CORES = 80
BOOST_MIN_CORES_OF = 96


@dataclass(frozen=True)
class HostStatePolicy:
    """When a host is too throttled or too busy to produce a usable number.

    THE IDLE-FREQUENCY TRAP, measured on this host 2026-08-04
    ---------------------------------------------------------
    A clock reading taken from an IDLE host says nothing about whether that host
    is throttled. An idle EPYC parks its cores: this machine reported **16 cores
    above 2.5 GHz at idle and 117 under load**, and `cpuinfo_min_freq` here is
    1.2 GHz, so an idle core sits AT the driver's own minimum — which is the
    exact signature this policy used to call "a throttled host, not a quiet one".

    That made the run-open gate unsatisfiable, and not as a matter of degree.
    `check_load` PASSes only at `load1/core <= max_load_per_core`, and a clock
    reading only carries information at `load1/core >= max_load_per_core`. The
    two gates at run open are evaluated against the SAME denominator and the
    SAME constant in opposite directions, so **a host quiet enough to pass the
    contention gate is by construction too quiet for the frequency gate to
    mean anything.** Both configurations refused: with `nominal_khz` supplied the
    idle reading fell below the floor and FAILed, and without it the check was
    COULD_NOT_CHECK — and the runner refused on anything but PASS. `AutoKernel
    could not take a measurement on a perfectly healthy machine`, which is how a
    guard gets switched off within a week.

    The fix is to move the judgement to where it discriminates, not to soften it.
    `frequency_verdict` DEFERS on an idle host instead of failing it, the runner
    does not refuse a deferred reading, and the check still bites at block close
    — where the benchmark's own load is what makes the clock informative. The
    anti-fail-open control is in `MicrobenchRunner`: a run that never once
    managed to JUDGE the frequency under load emits no number. So an idle host
    no longer aborts the run, and a host that is throttled for the whole run
    still cannot produce a number.

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
    require_package_power: bool = False

    def __post_init__(self) -> None:
        if self.nominal_khz is not None:
            if isinstance(self.nominal_khz, bool) or not isinstance(self.nominal_khz, int) \
                    or self.nominal_khz <= 0:
                raise ValueError("nominal_khz must be a positive int or None")
        for name in ("min_frequency_ratio", "max_load_per_core"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if not isinstance(self.require_package_power, bool):
            raise TypeError("require_package_power must be a bool")

    def frequency_verdict(self, state: HostState, *,
                          under_load: bool = True) -> tuple:
        """`(classification, Check)`. THE single computation site for both.

        `under_load` is the CALLER's structural statement that the claimed
        footprint was being driven when `state` was read. It is not inferred
        from `/proc/loadavg`: load1 is a one-minute damped average, so it lags
        the thing it would be standing in for, and the runner does not have to
        guess — it knows whether it has just spawned work on these cores.

        With `under_load=False` a well-formed check DEFERS rather than judging.
        An idle EPYC parks its cores, so a low clock there is evidence of
        idleness and not of throttle (16 boosting idle vs 117 under load on this
        host, 2026-08-04). Deferred is COULD_NOT_CHECK — never a pass.

        Order is load-bearing. Configuration defects (disabled, unreadable, no
        healthy reference) are reported BEFORE the idle deferral, because they
        are properties of the setup rather than of the host, and deferring them
        would hide a missing `--nominal-khz` until the host happened to be busy.
        """
        if not isinstance(state, HostState):
            raise TypeError("frequency_verdict takes a HostState")
        if not isinstance(under_load, bool):
            raise TypeError("under_load must be a bool")
        frequency_unreadable = tuple(
            reason for reason in state.unreadable
            if "scaling_cur_freq" in reason
        )
        if not self.require_frequency:
            # NOT a PASS. A check the caller switched off did not happen, and
            # this module's own rule everywhere else is that COULD_NOT_CHECK is
            # not a pass — the same rule `recipes._check_binding_inputs` states
            # for `verify_inputs=False`. A success-shaped result from a disabled
            # guard is the fail-open shape that makes a guard worth nothing: the
            # multi-day throttle that poisoned every number taken in its window
            # would have produced exactly this record.
            return FREQUENCY_UNEVALUABLE, schemas.Check(schemas.COULD_NOT_CHECK, (
                "frequency checking was disabled by the caller "
                "(HostStatePolicy.require_frequency=False), so the host's clock was "
                "never read; a run under this policy does not emit a number",))
        if not state.readable:
            return FREQUENCY_UNEVALUABLE, schemas.Check(schemas.COULD_NOT_CHECK, (
                "no cpu in the claimed footprint reported scaling_cur_freq; a multi-day "
                "host throttle has silently poisoned results here before, so an "
                "unverifiable frequency is not a passing frequency",)
                + frequency_unreadable)
        reasons: list = []
        min_khz = state.min_khz
        if frequency_unreadable:
            reasons.extend(frequency_unreadable)
        # The idle deferral, and it must come BEFORE the driver-minimum test
        # below: parking at `cpuinfo_min_freq` is precisely what a healthy idle
        # EPYC does, so that test cannot tell an idle host from a throttled one
        # either. Everything below this line reads the clock as evidence about
        # the host's HEALTH, and that inference is only valid when the cores
        # were being asked to do something.
        #
        # It also comes before the missing-`nominal_khz` branch, which is a
        # configuration defect rather than a host reading. That is safe because
        # the defect is caught earlier and harder: `campaign.py` makes
        # `--nominal-khz` a REQUIREMENT of `--execute`, at argument-parse time,
        # so a run reaching here without one has already been refused.
        if not under_load:
            return FREQUENCY_DEFERRED_IDLE, schemas.Check(
                schemas.COULD_NOT_CHECK, tuple(reasons) + (
                    f"the claimed footprint was not under load when this reading was "
                    f"taken, so {min_khz} kHz is evidence of idleness and not of "
                    f"throttle: an idle EPYC parks its cores (16 boosting idle vs 117 "
                    f"under load on this host, 2026-08-04), and parks them AT "
                    f"cpuinfo_min_freq. DEFERRED, not passed — the run must still judge "
                    f"the frequency under its own load before it may emit a number.",))

        # The one throttle shape that needs no operator-supplied reference: the
        # WHOLE footprint is pinned at the driver's minimum.  Testing `min_khz`
        # here rejected a healthy 96-core run as soon as one core parked in the
        # few microseconds between process exit and the sysfs read.
        values = [khz for _, khz in state.khz_by_cpu]
        if state.driver_min_khz is not None and values \
                and max(values) <= state.driver_min_khz:
            return FREQUENCY_JUDGED, schemas.Check(schemas.FAIL, tuple(reasons) + (
                f"the whole claimed footprint is pinned at the driver's own minimum "
                f"(max {max(values)} kHz <= cpuinfo_min_freq "
                f"{state.driver_min_khz} kHz) while "
                f"the claimed footprint is under load; "
                f"this is a throttled host, not a quiet one",))
        if self.nominal_khz is None:
            return FREQUENCY_UNEVALUABLE, schemas.Check(
                schemas.COULD_NOT_CHECK, tuple(reasons) + (
                    "HostStatePolicy.nominal_khz was not supplied, so the observed "
                    f"{min_khz} kHz cannot be compared against a healthy reference for "
                    "this cell. cpuinfo_max_freq is the single-core boost ceiling and is "
                    "NOT a valid all-core reference; record a healthy observation "
                    "instead.",))
        floor = self.nominal_khz * self.min_frequency_ratio
        median_khz = state.median_khz
        if median_khz is not None and median_khz < floor:
            return FREQUENCY_JUDGED, schemas.Check(schemas.FAIL, tuple(reasons) + (
                f"median claimed cpu is at {median_khz:.0f} kHz, below "
                f"{self.min_frequency_ratio:.2f} x nominal {self.nominal_khz} kHz "
                f"({floor:.0f} kHz); refusing to emit a number from a throttled host",))
        required = (len(values) * BOOST_MIN_CORES + BOOST_MIN_CORES_OF - 1) \
            // BOOST_MIN_CORES_OF
        boosting = sum(khz >= BOOST_THRESHOLD_KHZ for khz in values)
        if boosting < required:
            return FREQUENCY_JUDGED, schemas.Check(schemas.FAIL, tuple(reasons) + (
                f"only {boosting}/{len(values)} claimed cpus are at or above "
                f"{BOOST_THRESHOLD_KHZ} kHz under load; the ratified quorum for this "
                f"footprint is {required}/{len(values)} (80/96 scaled)",))
        if reasons:
            return FREQUENCY_UNEVALUABLE, schemas.Check(schemas.COULD_NOT_CHECK,
                                                        tuple(reasons))
        return FREQUENCY_JUDGED, schemas.Check(schemas.PASS, (
            f"median {median_khz:.0f} kHz >= {floor:.0f} kHz and {boosting}/"
            f"{len(values)} cpus >= {BOOST_THRESHOLD_KHZ} kHz (required {required}), "
            "judged under load",))

    def check_frequency(self, state: HostState) -> schemas.Check:
        """The Check alone. `frequency_verdict` is the one that computes it."""
        return self.frequency_verdict(state)[1]

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

    def check_package_power_available(self, state: HostState) -> schemas.Check:
        """Check counter coverage before a claim; this is not yet an interval.

        A point reading cannot attest power.  It can prove that every package
        containing the declared CPU partition has a readable counter, avoiding
        a claim and paired block that are known in advance to be unreportable.
        The interval attestation is still derived from block-open/block-close
        states by :func:`derive_package_power_attestation`.
        """
        if not self.require_package_power:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "package-power checking was disabled by the caller; no interval may "
                "be treated as power-attested under this policy",))
        try:
            claimed_cpus = set(_parse_cpu_list(state.cpu_list))
        except ValueError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"the claimed CPU partition could not be parsed: {exc}",))
        topology = dict(state.package_by_cpu)
        missing_topology = sorted(claimed_cpus - set(topology))
        if missing_topology:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"CPU-to-package topology is unavailable for claimed CPUs "
                f"{missing_topology}",))
        packages = {topology[cpu] for cpu in claimed_cpus}
        counters = {package for package, _energy, _maximum, _source
                    in state.package_energy_uj}
        missing_counters = sorted(packages - counters)
        if missing_counters:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"package energy counters are unreadable for claimed packages "
                f"{missing_counters}; missing power evidence is not zero watts",))
        return schemas.Check(schemas.PASS, (
            f"every claimed CPU in {state.cpu_list} maps to packages "
            f"{sorted(packages)}, and each package has a readable energy counter; "
            "the block still needs a positive open/close interval",))

    def to_dict(self) -> dict:
        return {"nominal_khz": self.nominal_khz,
                "min_frequency_ratio": self.min_frequency_ratio,
                "max_load_per_core": self.max_load_per_core,
                "require_frequency": self.require_frequency,
                "require_load": self.require_load,
                "require_package_power": self.require_package_power}


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
    params: dict
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
        for name in ("env", "recipe_env", "params"):
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
            "params": dict(self.params),
            "env": dict(self.env), "env_sha256": self.env_sha256,
            "binary_path": self.binary_path, "binary_sha256": self.binary_sha256,
            "binary_size": self.binary_size, "source_root": self.source_root,
            "library_path": self.library_path, "resolved_at": self.resolved_at,
        }


def _env_hash(env: Mapping) -> str:
    return schemas.content_hash({"env": {str(k): str(v) for k, v in env.items()}})


def _argv_hash(*, recipe_id: str, registry_id: str, arm: str, argv: Sequence[str],
               env: Mapping, params: Mapping) -> str:
    """Exactly `recipes.construct`'s own argv_sha256 preimage, so the two agree."""
    return schemas.content_hash({
        "recipe_id": recipe_id, "registry_id": registry_id, "arm": arm,
        "argv": list(argv), "env": dict(env), "params": dict(params),
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
        params=dict(command.params),
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
                                 arm=receipt.arm, argv=argv, env=receipt.recipe_env,
                                 params=receipt.params)
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
                                        # Resolved optional params are stored as null so the
                                        # receipt hash binds the complete frame. Feeding those
                                        # nulls back as caller overrides is different: the
                                        # constructor correctly rejects `n_depth=None`. Omit
                                        # only nulls here and let the registry re-resolve them.
                                        params={k: v for k, v in receipt.params.items()
                                                if v is not None},
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
            # The registry licenses GGML_IQK as the one arm-local env variant.
            # Its declared value is already covered by the constructor's
            # canonical-env finding; rechecking against literal canonical `1`
            # here made the registered `0` arm impossible to execute.
            expected = command.env.get(key) if key == "GGML_IQK" else value
            if env.get(key) != expected:
                reasons.append(
                    f"OMP stack incomplete: {key}={env.get(key)!r}, constructed recipe "
                    f"requires {expected!r}. The OMP stack is MANDATORY, not optional.")

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
    n_gpu_layers: int
    flash_attn: bool
    use_mmap: bool
    model_filename: str
    n_batch: Optional[int]
    n_ubatch: Optional[int]
    avg_ts: float
    stddev_ts: float
    samples_ts: tuple
    samples_ns: tuple
    autokernel_hardened: bool = False
    autokernel_output_invariant: bool = False
    autokernel_hybrid_ab_complete: bool = False
    autokernel_input_working_set_bytes: int = 0
    autokernel_input_hashes: str = ""
    autokernel_input_addresses: str = ""
    autokernel_context_addresses: str = ""
    autokernel_output_hashes: str = ""
    autokernel_unsynchronized_samples_ns: str = ""
    autokernel_thread_set_stable: bool = False
    autokernel_escape_checks_complete: bool = False
    autokernel_thread_set_hashes: str = ""
    autokernel_device_sync_mode: str = ""
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def metric_samples(self) -> tuple:
        return self.samples_ts

    def to_dict(self) -> dict:
        return {"build_commit": self.build_commit, "n_prompt": self.n_prompt,
                "n_gen": self.n_gen, "n_depth": self.n_depth,
                "n_threads": self.n_threads, "n_gpu_layers": self.n_gpu_layers,
                "flash_attn": self.flash_attn,
                "use_mmap": self.use_mmap, "model_filename": self.model_filename,
                "n_batch": self.n_batch, "n_ubatch": self.n_ubatch,
                "avg_ts": self.avg_ts, "stddev_ts": self.stddev_ts,
                "samples_ts": list(self.samples_ts), "samples_ns": list(self.samples_ns),
                "autokernel_hardened": self.autokernel_hardened,
                "autokernel_output_invariant": self.autokernel_output_invariant,
                "autokernel_hybrid_ab_complete":
                    self.autokernel_hybrid_ab_complete,
                "autokernel_input_working_set_bytes":
                    self.autokernel_input_working_set_bytes,
                "autokernel_input_hashes": self.autokernel_input_hashes,
                "autokernel_input_addresses": self.autokernel_input_addresses,
                "autokernel_context_addresses": self.autokernel_context_addresses,
                "autokernel_output_hashes": self.autokernel_output_hashes,
                "autokernel_unsynchronized_samples_ns":
                    self.autokernel_unsynchronized_samples_ns,
                "autokernel_thread_set_stable": self.autokernel_thread_set_stable,
                "autokernel_escape_checks_complete":
                    self.autokernel_escape_checks_complete,
                "autokernel_thread_set_hashes": self.autokernel_thread_set_hashes,
                "autokernel_device_sync_mode": self.autokernel_device_sync_mode}


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
            n_gpu_layers=_row_int(entry, "n_gpu_layers", index, default=0),
            flash_attn=_as_bool(entry.get("flash_attn", False)),
            use_mmap=_as_bool(entry.get("use_mmap", False)),
            model_filename=str(entry.get("model_filename", "")),
            n_batch=entry.get("n_batch"),
            n_ubatch=entry.get("n_ubatch"),
            avg_ts=_row_float(entry, "avg_ts", index),
            stddev_ts=_row_float(entry, "stddev_ts", index, default=0.0),
            samples_ts=tuple(values),
            samples_ns=samples_ns,
            autokernel_hardened=_as_bool(entry.get("autokernel_hardened", False)),
            autokernel_output_invariant=_as_bool(
                entry.get("autokernel_output_invariant", False)),
            autokernel_hybrid_ab_complete=_as_bool(
                entry.get("autokernel_hybrid_ab_complete", False)),
            autokernel_input_working_set_bytes=_row_int(
                entry, "autokernel_input_working_set_bytes", index, default=0),
            autokernel_input_hashes=str(entry.get("autokernel_input_hashes", "")),
            autokernel_input_addresses=str(entry.get("autokernel_input_addresses", "")),
            autokernel_context_addresses=str(entry.get("autokernel_context_addresses", "")),
            autokernel_output_hashes=str(entry.get("autokernel_output_hashes", "")),
            autokernel_unsynchronized_samples_ns=str(
                entry.get("autokernel_unsynchronized_samples_ns", "")),
            autokernel_thread_set_stable=_as_bool(
                entry.get("autokernel_thread_set_stable", False)),
            autokernel_escape_checks_complete=_as_bool(
                entry.get("autokernel_escape_checks_complete", False)),
            autokernel_thread_set_hashes=str(
                entry.get("autokernel_thread_set_hashes", "")),
            autokernel_device_sync_mode=str(
                entry.get("autokernel_device_sync_mode", "")),
            raw=dict(entry),
        ))
    return tuple(rows)


_AUTOKERNEL_HASH_RE = re.compile(r"^[0-9a-f]{16}$")
_AUTOKERNEL_ADDRESS_RE = re.compile(r"^0x[0-9a-f]+$")


def _comma_values(value: str) -> tuple:
    return tuple(part for part in value.split(",") if part)


def _address_pairs(value: str, *, label: str, reps: int) -> tuple:
    pairs = _comma_values(value)
    if len(pairs) != reps:
        raise ValueError(f"{label} carries {len(pairs)} pairs for {reps} repetitions")
    flattened: list[str] = []
    for pair in pairs:
        parts = tuple(pair.split("/"))
        if len(parts) != 2 or any(not _AUTOKERNEL_ADDRESS_RE.fullmatch(p)
                                  for p in parts):
            raise ValueError(f"{label} contains malformed address pair {pair!r}")
        if parts[0] == parts[1]:
            raise ValueError(f"{label} did not rotate within pair {pair!r}")
        flattened.extend(parts)
    if len(set(flattened)) != len(flattened):
        raise ValueError(f"{label} reuses an address across repetitions")
    return tuple(flattened)


def _check_autokernel_hardening(row: BenchRow, *, reps: int,
                                n_gpu_layers: int) -> tuple:
    """Validate the trusted RVP-C6-8 receipt emitted by hardened llama-bench.

    The ranked speed sample is the full-device-synchronized SECOND execution of
    each unique content vector, through a second simultaneously live
    context/input allocation. The ordinary first execution remains a diagnostic
    twin. Thus content or pointer memoization cannot accelerate the ranked path,
    while a stale pointer cache is exposed by unequal logits between the
    address-rotated pair.
    """
    reasons: list[str] = []
    if not row.autokernel_hardened:
        reasons.append(
            "argv requested --autokernel-harden but the result row does not attest the "
            "hardened path; repeated same-buffer samples are not admissible T1 evidence")
    if not row.autokernel_output_invariant:
        reasons.append(
            "the hardened result did not attest bitwise output invariance across each "
            "same-content, address-rotated replicate")
    if not row.autokernel_hybrid_ab_complete:
        reasons.append(
            "the hardened result did not attest completion of the ordinary/full-device-"
            "synchronize hybrid A/B for every repetition")
    if row.autokernel_input_working_set_bytes <= 0:
        reasons.append("the hardened result reports no live rotated working set")
    if not row.autokernel_thread_set_stable:
        reasons.append(
            "the hardened result did not attest that the process thread set was stable "
            "across every timed repetition")
    if not row.autokernel_escape_checks_complete:
        reasons.append(
            "the hardened result did not attest completion of the thread/device escape "
            "checks for every timed repetition")

    input_hashes = _comma_values(row.autokernel_input_hashes)
    if len(input_hashes) != reps:
        reasons.append(
            f"autokernel_input_hashes carries {len(input_hashes)} values for {reps} "
            "repetitions")
    elif any(not _AUTOKERNEL_HASH_RE.fullmatch(value) for value in input_hashes):
        reasons.append("autokernel_input_hashes contains a malformed digest")
    elif len(set(input_hashes)) != len(input_hashes):
        reasons.append(
            "measured repetitions reused input content; a content-keyed cache could pay")

    for value, label in (
            (row.autokernel_input_addresses, "autokernel_input_addresses"),
            (row.autokernel_context_addresses, "autokernel_context_addresses")):
        try:
            _address_pairs(value, label=label, reps=reps)
        except ValueError as exc:
            reasons.append(str(exc))

    output_pairs = _comma_values(row.autokernel_output_hashes)
    if len(output_pairs) != reps:
        reasons.append(
            f"autokernel_output_hashes carries {len(output_pairs)} pairs for {reps} "
            "repetitions")
    else:
        for pair in output_pairs:
            parts = tuple(pair.split("/"))
            if len(parts) != 2 or any(not _AUTOKERNEL_HASH_RE.fullmatch(p)
                                      for p in parts):
                reasons.append(f"autokernel_output_hashes contains malformed pair {pair!r}")
            elif parts[0] != parts[1]:
                reasons.append(
                    f"output changed across an address-rotated replicate ({pair})")

    unsynchronized_text = _comma_values(row.autokernel_unsynchronized_samples_ns)
    unsynchronized_ns: list[int] = []
    if len(unsynchronized_text) != reps:
        reasons.append(
            f"autokernel_unsynchronized_samples_ns carries {len(unsynchronized_text)} "
            f"values for {reps} repetitions")
    else:
        for value in unsynchronized_text:
            try:
                parsed = int(value)
            except ValueError:
                reasons.append(
                    f"autokernel_unsynchronized_samples_ns contains non-integer "
                    f"value {value!r}")
                continue
            if parsed <= 0:
                reasons.append(
                    "autokernel_unsynchronized_samples_ns contains a non-positive "
                    f"duration {value!r}")
                continue
            unsynchronized_ns.append(parsed)
    if (n_gpu_layers > 0 and len(unsynchronized_ns) == reps
            and len(row.samples_ns) == reps):
        ordinary_median = python_statistics.median(unsynchronized_ns)
        synchronized_median = python_statistics.median(row.samples_ns)
        gap_fraction = synchronized_median / ordinary_median - 1.0
        if gap_fraction > STREAM_ESCAPE_MAX_MEDIAN_GAP_FRACTION:
            reasons.append(
                "the full-device-synchronized twin is "
                f"{gap_fraction:.1%} slower in the median than the ordinary twin, above "
                f"the declared {STREAM_ESCAPE_MAX_MEDIAN_GAP_FRACTION:.0%} stream-escape "
                "screen; this is an integrity flag, not a corrected speed measurement")

    thread_pairs = _comma_values(row.autokernel_thread_set_hashes)
    if len(thread_pairs) != reps:
        reasons.append(
            f"autokernel_thread_set_hashes carries {len(thread_pairs)} pairs for "
            f"{reps} repetitions")
    else:
        for pair in thread_pairs:
            parts = tuple(pair.split("/"))
            if len(parts) != 4 or any(not _AUTOKERNEL_HASH_RE.fullmatch(part)
                                      for part in parts):
                reasons.append(
                    f"autokernel_thread_set_hashes contains malformed pair {pair!r}")
            elif len(set(parts)) != 1:
                reasons.append(
                    "the process thread set changed across the ordinary/synchronized "
                    f"timed pair ({pair})")

    required_sync_mode = (
        "hip_full_device" if n_gpu_layers > 0 else "cpu_not_applicable")
    if row.autokernel_device_sync_mode != required_sync_mode:
        reasons.append(
            f"the recipe's n_gpu_layers={n_gpu_layers} requires device sync mode "
            f"{required_sync_mode!r}, but the result reports "
            f"{row.autokernel_device_sync_mode!r}")
    return tuple(reasons)


def check_gpu_device_sampling(
        receipt: Optional[gpu_device_sampler.DeviceSamplingReceipt], *,
        n_gpu_layers: int, live_subprocess: bool) -> schemas.Check:
    """Make the C3-4 in-window trace verdict-bearing on live GPU arms."""
    if not live_subprocess or n_gpu_layers <= 0:
        return schemas.Check(schemas.PASS, (
            "an in-window GPU trace is not required for this non-live or CPU-only arm",))
    if receipt is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the live GPU arm carries no 250 ms in-window device-state receipt",))
    if not isinstance(receipt, gpu_device_sampler.DeviceSamplingReceipt):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the live GPU arm carries an unrecognized device-state receipt type",))
    return receipt.device_state(
        nominal_sclk_mhz=gpu_device_sampler.MI210_NOMINAL_SCLK_MHZ,
        min_sclk_ratio=gpu_device_sampler.MI210_MIN_SCLK_RATIO).check()


def check_gpu_ranked_duration_windows(
        observed_ns: Sequence[Any],
        receipt: Optional[gpu_device_sampler.DeviceSamplingReceipt], *,
        n_gpu_layers: int, live_subprocess: bool) -> schemas.Check:
    """Apply RVP-C3-5 before any live GPU timing sample can be ranked."""
    if not live_subprocess or n_gpu_layers <= 0:
        return schemas.Check(schemas.PASS, (
            "the gfx90a absolute duration floor is not required for this non-live or "
            "CPU-only arm",))
    if not isinstance(receipt, gpu_device_sampler.DeviceSamplingReceipt):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the live GPU arm has no recognized device receipt to bind the local "
            "gfx90a duration floor to",))
    return devices.GFX90A_RANKED_DURATION_ADMISSION.check(
        observed_ns, device_id=receipt.device_id)


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
    n_gpu_layers: int
    flash_attn: bool
    reps: int
    model_filename: str
    n_depth: Optional[int] = None
    autokernel_seed: Optional[int] = None
    expected_build_commit: Optional[str] = None

    @classmethod
    def from_command(cls, command: recipes.ConstructedCommand, *,
                     expected_build_commit: Optional[str] = None
                     ) -> "LlamaBenchExpectation":
        argv = list(command.argv)

        def after(flag: str) -> Optional[str]:
            return argv[argv.index(flag) + 1] if flag in argv else None

        depth = after("-d")
        autokernel_seed = after("--autokernel-harden")
        return cls(
            n_prompt=int(after("-p") or 0),
            n_gen=int(after("-n") or 0),
            n_threads=int(after("-t") or 0),
            n_gpu_layers=int(after("-ngl") or -1),
            flash_attn=(after("-fa") == "1"),
            reps=int(after("-r") or 0),
            model_filename=after("-m") or "",
            n_depth=int(depth) if depth is not None else None,
            autokernel_seed=(int(autokernel_seed)
                             if autokernel_seed is not None else None),
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
        if row.n_gpu_layers != self.n_gpu_layers:
            reasons.append(
                f"argv requested n_gpu_layers={self.n_gpu_layers} but the row reports "
                f"n_gpu_layers={row.n_gpu_layers}; the offload split is a different cell")
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
        if self.autokernel_seed is not None:
            reasons.extend(_check_autokernel_hardening(
                row, reps=self.reps, n_gpu_layers=self.n_gpu_layers))
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
                "n_threads": self.n_threads, "n_gpu_layers": self.n_gpu_layers,
                "flash_attn": self.flash_attn,
                "reps": self.reps, "model_filename": self.model_filename,
                "n_depth": self.n_depth,
                "autokernel_seed": self.autokernel_seed,
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
    khz_peak_by_cpu: tuple = ()
    sandbox_receipt: Optional[dict] = None
    sandbox_teardown: Optional[dict] = None
    device_sampling_receipt: Optional[gpu_device_sampler.DeviceSamplingReceipt] = None
    scheduler_smt_receipt: Optional[dict] = None
    inference_window_receipt: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"argv": list(self.argv), "returncode": self.returncode,
                "stdout_sha256": hashlib.sha256(self.stdout.encode("utf-8")).hexdigest(),
                "stdout_bytes": len(self.stdout.encode("utf-8")),
                "stderr_tail": self.stderr_tail, "pid": self.pid,
                "duration_s": self.duration_s, "timed_out": self.timed_out,
                "terminated_by_runner": self.terminated_by_runner,
                "khz_peak_by_cpu": [[cpu, khz] for cpu, khz in self.khz_peak_by_cpu],
                "sandbox_receipt": self.sandbox_receipt,
                "sandbox_teardown": self.sandbox_teardown,
                "device_sampling_receipt": (
                    None if self.device_sampling_receipt is None
                    else self.device_sampling_receipt.to_dict()),
                "scheduler_smt_receipt": self.scheduler_smt_receipt,
                "inference_window_receipt": self.inference_window_receipt}


def _proc_cpu_ticks(cpus: Sequence[int]) -> dict[int, tuple[int, int]]:
    """Return ``cpu -> (busy_ticks, total_ticks)`` from one /proc/stat read.

    This deliberately reports kernel accounting ticks rather than inventing a
    percent from wall time.  A reader can derive utilization from the two
    snapshots and can see exactly which sibling threads were observed.
    """
    wanted = set(cpus)
    result: dict[int, tuple[int, int]] = {}
    try:
        lines = Path("/proc/stat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu[0-9]+", fields[0]):
            continue
        cpu = int(fields[0][3:])
        if cpu not in wanted:
            continue
        try:
            ticks = [int(value) for value in fields[1:]]
        except ValueError:
            continue
        if len(ticks) < 5:
            continue
        total = sum(ticks)
        idle = ticks[3] + ticks[4]
        result[cpu] = (total - idle, total)
    return result


def _thread_siblings(cpus: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """Read the host's SMT topology for this exact invocation footprint.

    Observation failure is carried explicitly in the receipt rather than
    assumed away; this evidence must not turn a sysfs race into a benchmark
    refusal after the benchmark already ran.
    """
    result: dict[int, tuple[int, ...]] = {}
    for cpu in sorted(set(cpus)):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            result[cpu] = tuple(sorted(_parse_cpu_list(
                path.read_text(encoding="ascii").strip())))
        except (OSError, ValueError):
            result[cpu] = ()
    return result


def _proc_scheduler_sample(pid: int) -> Optional[dict]:
    """Targeted scheduler facts for the process this spawner created.

    No process enumeration or command-name matching is involved.  The PID is
    the ``Popen`` handle's PID, so this is evidence about this invocation only.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        # Fields 14, 15, and 39, after state (field 3) occupies tail[0].
        sample = {"user_ticks": int(tail[11]), "system_ticks": int(tail[12]),
                  "last_processor": int(tail[36])}
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        allowed = next((line.split(":", 1)[1].strip() for line in status.splitlines()
                        if line.startswith("Cpus_allowed_list:")), None)
        sample["allowed_cpu_list"] = allowed
        return sample
    except (OSError, IndexError, ValueError):
        return None


class _SchedulerSmtSampler:
    """Best-effort, per-Popen scheduler/SMT receipt collector.

    The claimed process itself causes most utilization on its taskset CPUs.
    Recording both those threads and their SMT siblings makes a later noise
    attribution falsifiable: high sibling utilization is visible instead of an
    after-the-fact claim that affinity isolated physical cores.
    """

    def __init__(self, pid: int, cpu_list: str) -> None:
        self.pid = pid
        self.cpu_list = cpu_list
        try:
            self.requested_cpus = tuple(sorted(_parse_cpu_list(cpu_list)))
        except ValueError:
            self.requested_cpus = ()
        self.siblings = _thread_siblings(self.requested_cpus)
        self.observed_cpus = tuple(sorted(set(self.requested_cpus).union(
            *(set(value) for value in self.siblings.values()))))
        self.started_at = _utc_now()
        self.cpu_open = _proc_cpu_ticks(self.observed_cpus)
        self.first_process = _proc_scheduler_sample(pid)
        self.last_process = self.first_process
        self.processors: set[int] = set()
        if self.first_process is not None:
            self.processors.add(self.first_process["last_processor"])
        self.samples = 1 if self.first_process is not None else 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True,
                                        name=f"autokernel-scheduler-{pid}")
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            sample = _proc_scheduler_sample(self.pid)
            if sample is None:
                continue
            self.last_process = sample
            self.samples += 1
            self.processors.add(sample["last_processor"])

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=1.0)
        cpu_close = _proc_cpu_ticks(self.observed_cpus)
        per_cpu = {}
        for cpu in self.observed_cpus:
            opened, closed = self.cpu_open.get(cpu), cpu_close.get(cpu)
            if opened is None or closed is None:
                per_cpu[str(cpu)] = {"observed": False}
                continue
            busy = max(0, closed[0] - opened[0])
            total = max(0, closed[1] - opened[1])
            per_cpu[str(cpu)] = {"observed": True, "busy_ticks": busy,
                                 "total_ticks": total,
                                 "utilization": None if total == 0 else busy / total}
        return {
            "schema": "epyc.autokernel.scheduler_smt_invocation_receipt.v1",
            "pid": self.pid, "started_at": self.started_at, "ended_at": _utc_now(),
            "requested_cpu_list": self.cpu_list,
            "requested_cpus": list(self.requested_cpus),
            "thread_siblings": {str(cpu): list(siblings)
                                for cpu, siblings in sorted(self.siblings.items())},
            "observed_cpus": list(self.observed_cpus), "per_cpu": per_cpu,
            "process_scheduler": {"samples": self.samples,
                                  "first": self.first_process,
                                  "last": self.last_process,
                                  "processors_observed": sorted(self.processors)},
        }


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
                 workdir_root: Optional[str] = None,
                 sandbox_policy: Optional[process_sandbox.SandboxPolicy] = None,
                 device_sampler: Optional[gpu_device_sampler.RocmSmiSampler] = None) -> None:
        if term_grace_s <= 0:
            raise ValueError("term_grace_s must be positive")
        self._term_grace_s = float(term_grace_s)
        self._stderr_tail_bytes = int(stderr_tail_bytes)
        self._workdir_root = assert_scratch_root_is_not_production(workdir_root)
        if sandbox_policy is not None and not isinstance(
                sandbox_policy, process_sandbox.SandboxPolicy):
            raise TypeError("sandbox_policy must be a SandboxPolicy or None")
        self._sandbox_policy = sandbox_policy
        if device_sampler is not None and not isinstance(
                device_sampler, gpu_device_sampler.RocmSmiSampler):
            raise TypeError("device_sampler must be a RocmSmiSampler or None")
        self._device_sampler = device_sampler
        if sandbox_policy is not None:
            actual_root = storage._norm(workdir_root or os.environ.get("TMPDIR")
                                        or tempfile.gettempdir())
            allowed_root = storage._norm(sandbox_policy.writable_root)
            if not storage._under(actual_root, allowed_root):
                raise process_sandbox.SandboxError(
                    f"spawner workdir root {actual_root!r} is outside the sandbox's only "
                    f"writable tree {allowed_root!r}")

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
        peaks: dict[int, int] = {}
        stop_sampling = threading.Event()
        sampler = None
        sandbox_receipt = None
        sandbox_teardown = None
        device_sampling_receipt = None
        scheduler_smt_receipt = None
        with tempfile.TemporaryDirectory(prefix="autokernel-microbench-",
                                         dir=self._workdir_root) as workdir:
            evaluator_dir = Path(workdir, "evaluator")
            candidate_dir = Path(workdir, "candidate")
            evaluator_dir.mkdir()
            candidate_dir.mkdir()
            out_path = evaluator_dir / "stdout"
            err_path = evaluator_dir / "stderr"
            receipt_path = evaluator_dir / "sandbox-receipt.json"
            spawn_argv = argv
            invocation_policy = self._sandbox_policy
            if self._sandbox_policy is not None:
                # The campaign sandbox root and this outer workdir are owned by
                # the evaluator.  The process receives write authority over the
                # EMPTY candidate child only.  stdout/stderr and the activation
                # receipt live in the evaluator sibling, so candidate code
                # cannot rewrite the evidence used to judge its confinement.
                # The whole outer directory is deleted before the next arm, so
                # no process or filesystem memo survives between invocations.
                invocation_policy = process_sandbox.SandboxPolicy(
                    writable_root=str(candidate_dir),
                    cgroup_root=self._sandbox_policy.cgroup_root,
                    limits=self._sandbox_policy.limits,
                    token=self._sandbox_policy.token)
                spawn_argv = list(invocation_policy.wrap(
                    argv, receipt_path=str(receipt_path)))
                env = dict(env)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["TMPDIR"] = str(candidate_dir)
            with out_path.open("wb") as out_fh, err_path.open("wb") as err_fh:
                try:
                    proc = subprocess.Popen(spawn_argv, stdout=out_fh, stderr=err_fh,
                                            stdin=subprocess.DEVNULL, env=env, cwd=cwd,
                                            start_new_session=True)
                except OSError as exc:
                    raise SpawnFailure(f"could not start {argv[0]!r}: {exc}") from exc
                pid = proc.pid
                device_session = None
                if self._device_sampler is not None:
                    try:
                        device_session = self._device_sampler.start()
                    except gpu_device_sampler.DeviceSamplingError as exc:
                        self._terminate(proc)
                        raise SpawnFailure(
                            f"could not start the required in-window GPU sampler: {exc}") from exc
                try:
                    cpu_list = argv[argv.index("-c") + 1]
                    cpus = _parse_cpu_list(cpu_list)
                except (ValueError, IndexError):
                    cpus = ()
                    cpu_list = ""
                scheduler_sampler = (_SchedulerSmtSampler(pid, cpu_list)
                                     if cpus else None)
                if cpus:
                    def sample_while_alive() -> None:
                        while not stop_sampling.is_set():
                            for cpu in cpus:
                                khz = _read_int_file(Path(
                                    f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/"
                                    "scaling_cur_freq"))
                                if khz is not None:
                                    peaks[cpu] = max(peaks.get(cpu, 0), khz)
                            stop_sampling.wait(0.005)

                    sampler = threading.Thread(
                        target=sample_while_alive,
                        name=f"autokernel-cpufreq-{pid}", daemon=True)
                    sampler.start()
                try:
                    returncode = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminated = True
                    returncode = self._terminate(proc)
                finally:
                    stop_sampling.set()
                    if sampler is not None:
                        sampler.join(timeout=1.0)
                    if scheduler_sampler is not None:
                        scheduler_smt_receipt = scheduler_sampler.stop()
                    if device_session is not None:
                        try:
                            device_sampling_receipt = device_session.stop()
                        except gpu_device_sampler.DeviceSamplingError as exc:
                            raise SpawnFailure(
                                "the benchmark completed without a valid in-window GPU "
                                f"device-state trace: {exc}") from exc
            if self._sandbox_policy is not None:
                try:
                    sandbox_receipt = process_sandbox.read_receipt(receipt_path)
                    process_sandbox.verify_receipt(
                        sandbox_receipt, policy=invocation_policy, pid=pid, argv=argv)
                    sandbox_teardown = process_sandbox.cleanup_cgroup(
                        invocation_policy, pid)
                except process_sandbox.SandboxError as exc:
                    cleanup_note = ""
                    cgroup_path = invocation_policy.cgroup_path(pid)
                    if cgroup_path.exists():
                        try:
                            process_sandbox.cleanup_cgroup(invocation_policy, pid)
                            cleanup_note = "; the owned cgroup was drained after refusal"
                        except process_sandbox.SandboxError as cleanup_exc:
                            cleanup_note = (
                                "; additionally, owned-cgroup cleanup failed: "
                                f"{cleanup_exc}")
                    raise SpawnFailure(
                        f"candidate containment did not produce a verified receipt and "
                        f"teardown: {exc}{cleanup_note}") from exc
            stdout = out_path.read_text(encoding="utf-8", errors="replace")
            stderr = err_path.read_bytes()[-self._stderr_tail_bytes:].decode(
                "utf-8", errors="replace")

        return SpawnResult(argv=tuple(argv), returncode=returncode, stdout=stdout,
                           stderr_tail=stderr, pid=pid,
                           duration_s=time.monotonic() - started, timed_out=timed_out,
                           terminated_by_runner=terminated,
                           khz_peak_by_cpu=tuple(sorted(peaks.items())),
                           sandbox_receipt=sandbox_receipt,
                           sandbox_teardown=sandbox_teardown,
                           device_sampling_receipt=device_sampling_receipt,
                           scheduler_smt_receipt=scheduler_smt_receipt)

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
        # Mirrors `statistics.PairedBlock.__post_init__`. Checked HERE as well as
        # there because `assemble_block` copies these two fields onto the
        # PairedBlock: without this, a malformed plan is discovered only after
        # the block has been measured, as a `MaterialError` out of the middle of
        # a run rather than a refusal before a single process is spawned.
        if self.segment not in statistics.SEGMENTS:
            raise ValueError(f"segment {self.segment!r} is not one of "
                             f"{list(statistics.SEGMENTS)}")
        if self.segment == statistics.SEGMENT_EXTENSION:
            if isinstance(self.extension_round, bool) \
                    or not isinstance(self.extension_round, int) or self.extension_round < 1:
                raise ValueError(
                    "extension_round must be a positive int on an extension block; an "
                    "extension that cannot say which declared round it belongs to is "
                    "unstructured continuation")
        elif self.extension_round is not None:
            raise ValueError("extension_round must be None on a base block")
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

    Extension rounds CONTINUE the base schedule's index line rather than
    re-deriving one (module docstring, "the extension round is extended, not
    re-derived"): round *r* of `count` blocks starts at
    `base_blocks + (r - 1) * count`, so its orders are `order_for()`'s reversed
    limb and no two rounds can claim the same index. The unit is taken at the
    ABSOLUTE index for the same reason — restarting the unit cycle at every
    round would over-weight the first units of the cycle across a run that the
    reducer then treats as one block sequence.
    """
    if not isinstance(schedule, statistics.OrderSchedule):
        raise TypeError("plan_blocks takes a statistics.OrderSchedule")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive int")
    if not unit_ids:
        raise ValueError("unit_ids must name at least one measurement-material unit")
    if segment not in statistics.SEGMENTS:
        raise ValueError(f"segment {segment!r} is not one of {list(statistics.SEGMENTS)}")
    if segment == statistics.SEGMENT_EXTENSION:
        if isinstance(extension_round, bool) or not isinstance(extension_round, int) \
                or extension_round < 1:
            raise ValueError("plan_blocks needs the declared round number to place an "
                             "extension round on the schedule's index line")
        offset = schedule.base_blocks + (extension_round - 1) * count
    else:
        if extension_round is not None:
            raise ValueError("extension_round must be None on the base segment")
        offset = 0
    units = list(unit_ids)
    return tuple(
        BlockPlan(block_index=offset + i, order=schedule.order_for(offset + i), pairs=pairs,
                  unit_id=units[(offset + i) % len(units)], stratum=stratum, segment=segment,
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
class PackagePowerAttestation:
    """Package-counter delta over one partition's exact measurement window.

    EPYC exposes package energy, not per-core energy.  This receipt therefore
    says ``shared_package_window`` explicitly: it binds the partition's CPU
    mask to the packages it occupied and records the shared-package average. It
    never relabels a package counter as lane-exclusive power.
    """

    cpu_list: str
    duration_s: float
    average_watts_by_package: tuple
    counter_sources: tuple
    scope: str = "shared_package_window"

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "cpu_list": self.cpu_list,
            "duration_s": self.duration_s,
            "average_watts_by_package": [
                [package, watts] for package, watts in self.average_watts_by_package
            ],
            "counter_sources": [[package, source]
                                for package, source in self.counter_sources],
        }


def derive_package_power_attestation(open_state: HostState,
                                     close_state: Optional[HostState]) -> tuple:
    """Return ``(Check, attestation-or-None)``; missing energy never becomes zero."""
    if close_state is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "block has no close host state, so package energy has no interval",)), None
    if open_state.cpu_list != close_state.cpu_list:
        return schemas.Check(schemas.FAIL, (
            "open and close host states name different CPU partitions",)), None
    if open_state.package_by_cpu != close_state.package_by_cpu:
        return schemas.Check(schemas.FAIL, (
            "CPU-to-package topology changed across the measurement window",)), None
    if open_state.monotonic_s is None or close_state.monotonic_s is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "host states carry no monotonic timestamps for a power denominator",)), None
    duration = close_state.monotonic_s - open_state.monotonic_s
    if not math.isfinite(duration) or duration <= 0:
        return schemas.Check(schemas.FAIL, (
            f"package-power interval must be positive, got {duration!r}",)), None

    before = {package: (energy, maximum, source)
              for package, energy, maximum, source in open_state.package_energy_uj}
    after = {package: (energy, maximum, source)
             for package, energy, maximum, source in close_state.package_energy_uj}
    packages = sorted({package for _cpu, package in open_state.package_by_cpu})
    if not packages or any(package not in before or package not in after
                           for package in packages):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "one or more claimed CPU packages has no readable energy counter",)), None

    watts: list[tuple[int, float]] = []
    sources: list[tuple[int, str]] = []
    for package in packages:
        start, maximum, source = before[package]
        end, close_maximum, close_source = after[package]
        if maximum != close_maximum or source != close_source:
            return schemas.Check(schemas.FAIL, (
                f"package {package} energy-counter identity changed across the window",)), None
        delta = end - start if end >= start else maximum - start + end
        if delta < 0 or delta > maximum:
            return schemas.Check(schemas.FAIL, (
                f"package {package} energy delta {delta} is outside its counter range",)), None
        value = delta / 1_000_000.0 / duration
        if not math.isfinite(value) or value < 0:
            return schemas.Check(schemas.FAIL, (
                f"package {package} average power is invalid: {value!r}",)), None
        watts.append((package, value))
        sources.append((package, source))
    attestation = PackagePowerAttestation(
        cpu_list=open_state.cpu_list, duration_s=duration,
        average_watts_by_package=tuple(watts), counter_sources=tuple(sources))
    return schemas.Check(schemas.PASS, (
        f"package energy covers CPU partition {open_state.cpu_list} for {duration:.6f}s; "
        "values are shared-package, not lane-exclusive",)), attestation


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
        power_check, power = derive_package_power_attestation(
            self.host_state_open, self.host_state_close)
        return {
            "plan": self.plan.to_dict(),
            "invocations": [i.to_dict() for i in self.invocations],
            "host_state_open": self.host_state_open.to_dict(),
            "host_state_close": (self.host_state_close.to_dict()
                                 if self.host_state_close is not None else None),
            "package_power": {
                "check": {"outcome": power_check.outcome,
                          "reasons": list(power_check.reasons)},
                "attestation": None if power is None else power.to_dict(),
            },
            "paired_block": (self.paired_block.to_list()
                             if self.paired_block is not None else None),
            "checks": [[n, {"outcome": c.outcome, "reasons": list(c.reasons)}]
                       for n, c in self.checks],
            "refusals": list(self.refusals),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class ReplayedBlockRecord:
    """A completed block recovered from a durable checkpoint.

    Recovery deliberately does not try to recreate ``Invocation`` or
    ``SpawnResult`` objects: doing so would turn their constructors into a
    second raw-vector schema.  The original canonical block mapping is retained
    byte-for-byte for the final raw vector, while the paired block and plan are
    reconstructed and checked for the two operations the runner needs: schedule
    validation and reduction.
    """

    plan: BlockPlan
    paired_block: statistics.PairedBlock
    raw: Mapping

    @property
    def complete(self) -> bool:
        return True

    def to_dict(self) -> dict:
        # Return a JSON-value copy so a caller cannot mutate the checkpoint
        # mapping through the recovered run.
        return json.loads(schemas.canonical_json(self.raw))


def replay_completed_block(
        expected: BlockPlan, raw: Mapping, *, plan: "MicrobenchPlan | None" = None,
) -> ReplayedBlockRecord:
    """Validate one checkpoint block against its exact derived schedule slot.

    This is intentionally stricter than the final raw-vector reader: a prefix
    is executable authority because accepting it causes the runner to skip
    inference.  Any missing invocation receipt, order mismatch, gap, refusal,
    or disagreement between invocation samples and the paired reduction fails
    closed before the next process can be spawned.
    """
    if not isinstance(expected, BlockPlan):
        raise TypeError("replay_completed_block takes an expected BlockPlan")
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint block must be a mapping")
    if plan is not None and not isinstance(plan, MicrobenchPlan):
        raise TypeError("replay_completed_block plan must be a MicrobenchPlan or None")
    required = {
        "plan", "invocations", "host_state_open", "host_state_close",
        "package_power", "paired_block", "checks", "refusals", "complete",
    }
    if set(raw) != required:
        raise ValueError(
            "checkpoint block fields differ from the canonical BlockRecord schema: "
            f"missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}")
    if raw.get("plan") != expected.to_dict():
        raise ScheduleMismatch(
            f"checkpoint block {expected.block_index} plan/order/frame does not match "
            "the schedule derived from the current committed plan")
    if raw.get("complete") is not True or raw.get("refusals") != []:
        raise RunRefused(
            f"checkpoint block {expected.block_index} is not complete and refusal-free")
    invocations = raw.get("invocations")
    if not isinstance(invocations, list) or len(invocations) != expected.invocations:
        raise PairingViolation(
            f"checkpoint block {expected.block_index} has "
            f"{len(invocations) if isinstance(invocations, list) else 'non-list'} "
            f"invocations; the plan requires {expected.invocations}")
    anchor_samples: list[float] = []
    candidate_samples: list[float] = []
    for position, (invocation, arm) in enumerate(zip(invocations, expected.arm_sequence)):
        if not isinstance(invocation, Mapping):
            raise PairingViolation(
                f"checkpoint block {expected.block_index} invocation {position} "
                "is not a mapping")
        if invocation.get("block_index") != expected.block_index \
                or invocation.get("position") != position \
                or invocation.get("arm") != arm:
            raise PairingViolation(
                f"checkpoint block {expected.block_index} invocation {position} "
                "does not match the derived alternating arm sequence")
        receipt_raw = invocation.get("receipt")
        if not isinstance(receipt_raw, Mapping) \
                or not isinstance(invocation.get("recipe"), str) \
                or not invocation.get("recipe") \
                or not isinstance(invocation.get("spawn"), Mapping):
            raise ValueError(
                f"checkpoint block {expected.block_index} invocation {position} "
                "is missing its exact execution receipt, recipe, or spawn receipt")
        try:
            receipt = ExecutionReceipt(
                runner_id=receipt_raw["runner_id"],
                recipe_id=receipt_raw["recipe_id"],
                registry_id=receipt_raw["registry_id"], arm=receipt_raw["arm"],
                constructor_id=receipt_raw["constructor_id"],
                constructor_sha256=receipt_raw["constructor_sha256"],
                argv_sha256=receipt_raw["argv_sha256"],
                argv=tuple(receipt_raw["argv"]),
                recipe_env=dict(receipt_raw["recipe_env"]),
                params=dict(receipt_raw["params"]), env=dict(receipt_raw["env"]),
                env_sha256=receipt_raw["env_sha256"],
                binary_path=receipt_raw["binary_path"],
                binary_sha256=receipt_raw["binary_sha256"],
                binary_size=receipt_raw["binary_size"],
                source_root=receipt_raw["source_root"],
                library_path=receipt_raw["library_path"],
                resolved_at=receipt_raw["resolved_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint block {expected.block_index} invocation {position} "
                f"has a malformed execution receipt: {exc}") from exc
        binding = (expected.unit_id, arm)  # included in errors without hiding the slot
        receipt_check = verify_receipt(
            receipt, argv=receipt.argv, env=receipt.env,
            binary_sha256=receipt.binary_sha256)
        if receipt_check.outcome != schemas.PASS \
                or receipt.arm != arm \
                or invocation["recipe"] != receipt.render():
            raise ValueError(
                f"checkpoint block {expected.block_index} invocation {position} "
                f"receipt does not re-derive for {binding!r}: "
                f"{'; '.join(receipt_check.reasons)}")
        if plan is not None:
            arm_binding = (plan.candidate_binding if arm == ARM_CANDIDATE
                           else plan.anchor_binding)
            expected_params = plan.params_for(arm, expected.unit_id)
            if receipt.recipe_id != plan.recipe_id \
                    or any(receipt.params.get(key) != value
                           for key, value in expected_params.items()) \
                    or (receipt.binary_path, receipt.source_root,
                        receipt.library_path) != (
                            arm_binding.binary, arm_binding.source_root,
                            arm_binding.library_path):
                raise ValueError(
                    f"checkpoint block {expected.block_index} invocation {position} "
                    "execution receipt differs from the current recipe/frame/binding")
        claim = invocation.get("claim")
        if not isinstance(claim, Mapping) or claim.get("outcome") != schemas.PASS:
            raise ClaimNotHeld(
                f"checkpoint block {expected.block_index} invocation {position} "
                "was not attested under a held claim")
        samples = invocation.get("samples")
        if not isinstance(samples, list) or not samples:
            raise PairingViolation(
                f"checkpoint block {expected.block_index} invocation {position} "
                "has no sample vector")
        target = anchor_samples if arm == ARM_ANCHOR else candidate_samples
        target.extend(samples)

    value = raw.get("paired_block")
    if not isinstance(value, list) or len(value) != 9:
        raise PairingViolation(
            f"checkpoint block {expected.block_index} has no canonical paired block")
    paired = statistics.PairedBlock(
        block_index=value[0], unit_id=value[1], stratum=value[2], order=value[3],
        segment=value[4], extension_round=value[5], measured_at=value[6],
        anchor_samples=tuple(value[7]), candidate_samples=tuple(value[8]))
    expected_identity = (
        expected.block_index, expected.unit_id, expected.stratum, expected.order,
        expected.segment, expected.extension_round)
    observed_identity = (
        paired.block_index, paired.unit_id, paired.stratum, paired.order,
        paired.segment, paired.extension_round)
    if observed_identity != expected_identity:
        raise ScheduleMismatch(
            f"checkpoint block identity {observed_identity!r} != {expected_identity!r}")
    if paired.anchor_samples != tuple(anchor_samples) \
            or paired.candidate_samples != tuple(candidate_samples):
        raise PairingViolation(
            f"checkpoint block {expected.block_index} paired samples differ from its "
            "exact invocation receipts")
    return ReplayedBlockRecord(plan=expected, paired_block=paired, raw=raw)


# =============================================================================
# The run
# =============================================================================

@dataclass(frozen=True)
class ExtensionAuthorization:
    """The pre-committed licence for ONE declared extension round.

    It holds the CAMPAIGN, and READS every bound off it: `max_rounds`,
    `blocks_per_round`, `max_blocks_per_candidate` and the base segment's length
    are not arguments, so there is no spelling of this object in which a caller
    states its own budget. The only thing the caller supplies is which declared
    round this is.

    It does NOT take a `(StoppingRule, StoppingRuleCommitment)` pair, which is
    what it took until the 2026-08-04 red team. `commitment.verify(rule)`
    compares its two arguments to each other, and a caller that wanted a budget
    the campaign never declared never had to mutate anything — it built the rule
    it wanted and committed THAT. Verification passed by construction, a licence
    for round 3 of a `max_rounds=3` rule carrying campaign id
    `"not-even-this-campaign"` and `committed_at` in 2099 was accepted, and its
    three rounds were spawned before anything refused them. The campaign is the
    object the reducer already reads, it verifies its own commitment at
    construction, and it carries an accepted calibration for this cell — so a
    licence derived from it is a licence the evidence's own denominator issued.

    `base_blocks` is `campaign.b_min`, and it is the OTHER half of the schedule
    identity: the extension continues the base segment's index line, so a plan
    whose `base_blocks` disagrees is a re-derived schedule wearing an
    extension's name, and `MicrobenchPlan` refuses it.

    Construction is the first enforcement point but not the last: a caller can
    build a second campaign, so `assemble_run_blocks()` re-checks the licence
    against the campaign the evidence is being pooled for (`licence_for`).
    """

    campaign: statistics.CampaignStatistics
    round_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.campaign, statistics.CampaignStatistics):
            raise ExtensionNotDeclared(
                "an extension round is authorized by the CAMPAIGN — the same "
                "statistics.CampaignStatistics the reduction is computed under — not by a "
                "stopping rule and a commitment the caller minted together, which verify "
                "against each other and against nothing else")
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int) \
                or self.round_index < 1:
            raise ExtensionNotDeclared("extension round_index must be a positive int")
        verified = self.commitment.verify(self.rule)
        if verified.outcome != schemas.PASS:
            raise ExtensionNotDeclared(
                f"the stopping rule does not match the one committed at campaign start "
                f"({self.commitment.committed_at}): {verified.outcome} — "
                f"{'; '.join(verified.reasons)}. An extension the caller granted itself "
                f"after seeing the base segment is exactly what anytime-validity does not "
                f"license.")
        extension = self.rule.extension
        if extension.max_rounds < 1:
            raise ExtensionNotDeclared(
                f"rule {self.rule.rule_id!r} declares max_rounds={extension.max_rounds}: it "
                f"licenses NO extension round at all, so there is nothing to authorize")
        if self.round_index > extension.max_rounds:
            raise ExtensionNotDeclared(
                f"round {self.round_index} exceeds the declared maximum "
                f"{extension.max_rounds} of rule {self.rule.rule_id!r}. Extension follows "
                f"the declared rule only; there is no round the rule did not name.")
        last_block = self.base_blocks + self.round_index * extension.blocks_per_round
        if last_block > self.rule.max_blocks_per_candidate:
            raise ExtensionNotDeclared(
                f"round {self.round_index} would take the run to {last_block} blocks, past "
                f"the declared ceiling max_blocks_per_candidate="
                f"{self.rule.max_blocks_per_candidate} of rule {self.rule.rule_id!r}")

    @property
    def rule(self) -> statistics.StoppingRule:
        """The campaign's committed rule. Read, never supplied."""
        return self.campaign.stopping_rule

    @property
    def commitment(self) -> statistics.StoppingRuleCommitment:
        """The commitment the campaign was constructed against."""
        return self.campaign.stopping_rule_commitment

    @property
    def base_blocks(self) -> int:
        """The calibrated `B_min`. The base segment is exactly that long."""
        return self.campaign.b_min

    @property
    def blocks_per_round(self) -> int:
        return self.rule.extension.blocks_per_round

    @property
    def max_rounds(self) -> int:
        return self.rule.extension.max_rounds

    @property
    def first_block_index(self) -> int:
        """Where this round starts on the base segment's index line."""
        return self.base_blocks + (self.round_index - 1) * self.blocks_per_round

    def licence_for(self, campaign: statistics.CampaignStatistics) -> schemas.Check:
        """PASS only when this licence was issued by `campaign` itself.

        Construction binds the licence to A campaign; this binds it to THIS one.
        Nothing stops a caller from building a second `CampaignStatistics` with a
        permissive rule, licensing a round off it, and then pooling the round
        into the real campaign's evidence — and the resulting record would carry
        the other campaign's `rule_id`, `rule_content_hash` and `committed_at`
        while its number was reduced under this campaign's threshold. That is a
        record that overstates its own licence, and the reducer cannot see it:
        `PairedBlock` carries a segment and a round number, never an
        authorization.

        The identity compared is the `StoppingRuleCommitment` — campaign id,
        rule id, content hash and `committed_at`, all four at once — plus the
        campaign seed the schedule is derived from and the calibrated `B_min`
        the round's first index is measured off.
        """
        if not isinstance(campaign, statistics.CampaignStatistics):
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "no CampaignStatistics was supplied to check this licence against",))
        reasons = []
        if self.commitment != campaign.stopping_rule_commitment:
            reasons.append(
                f"the round is licensed by rule {self.commitment.rule_id!r} committed at "
                f"{self.commitment.committed_at} for campaign "
                f"{self.commitment.campaign_id!r}, but this evidence is reduced under rule "
                f"{campaign.stopping_rule_commitment.rule_id!r} committed at "
                f"{campaign.stopping_rule_commitment.committed_at} for campaign "
                f"{campaign.campaign_id!r}; a licence issued by another campaign licenses "
                "nothing here")
        if self.campaign.campaign_seed != campaign.campaign_seed:
            reasons.append(
                "the licence was issued under a different committed campaign seed, so the "
                "order schedule it authorizes is not the one this reduction checks against")
        if self.base_blocks != campaign.b_min:
            reasons.append(
                f"the licence places round {self.round_index} off a base segment of "
                f"{self.base_blocks} blocks but this campaign's calibrated B_min is "
                f"{campaign.b_min}")
        return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
            else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"round_index": self.round_index, "base_blocks": self.base_blocks,
                "first_block_index": self.first_block_index,
                "rule_id": self.rule.rule_id,
                "rule_content_hash": self.commitment.rule_content_hash,
                "committed_at": self.commitment.committed_at,
                "campaign_id": self.commitment.campaign_id,
                "extension": self.rule.extension.to_dict(),
                "max_blocks_per_candidate": self.rule.max_blocks_per_candidate}


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

    `segment` and `extension` are the extension-round producer. `base_blocks`
    keeps its meaning in BOTH segments — it is the length of the base segment
    and therefore the `base_blocks` of the one `OrderSchedule` the whole run
    uses — and it is `blocks_to_run` (not `base_blocks`) that says how many
    blocks THIS invocation produces. That split is the "extended, not
    re-derived" decision made structural: there is no way to build an extension
    plan whose schedule is derived for a different base length, because
    `schedule()` reads `base_blocks` and `ExtensionAuthorization` refuses a
    `base_blocks` that disagrees with it.
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
    #: Shared matched-experiment identity when two campaign records are the
    #: intervention and A/A arms of one experiment.  It is an identity, not a
    #: caller-declared order: ``schedule()`` derives the actual order from it.
    matched_experiment_id: Optional[str] = None
    candidate_instrument_root: Optional[str] = None
    anchor_instrument_root: Optional[str] = None
    candidate_param_overrides: Mapping = field(default_factory=dict)
    anchor_param_overrides: Mapping = field(default_factory=dict)
    #: Per-ranked-unit recipe changes (shape/value-distribution selectors only).
    #: Unlike ``unit_ids`` alone, these mappings change the command that runs.
    unit_param_overrides: Mapping = field(default_factory=dict)
    #: Units deliberately hostile to structured-input/shape short-circuiting.
    #: They remain ordinary ranked blocks; this label grants no gate-only path.
    anti_short_circuit_units: tuple = ()
    physical_envelopes: Mapping = field(default_factory=dict)
    stratum: str = api.STRATUM_SELECTION
    timeout_s: float = 1800.0
    attempt: int = 0
    segment: str = statistics.SEGMENT_BASE
    extension: Optional[ExtensionAuthorization] = None

    def __post_init__(self) -> None:
        for name in ("recipe_id", "candidate_id", "campaign_seed"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"plan.{name} must be a non-empty string")
        if self.matched_experiment_id is not None:
            if (not isinstance(self.matched_experiment_id, str)
                    or not self.matched_experiment_id.startswith("akm-")
                    or "\0" in self.matched_experiment_id):
                raise ValueError(
                    "plan.matched_experiment_id must be an akm- identity or None")
        for name in ("candidate_binding", "anchor_binding"):
            if not isinstance(getattr(self, name), recipes.ToolBinding):
                raise TypeError(f"plan.{name} must be a recipes.ToolBinding")
        if (self.candidate_instrument_root is None) != (self.anchor_instrument_root is None):
            raise ValueError(
                "candidate_instrument_root and anchor_instrument_root must be supplied "
                "together; one source tree cannot prove instrument identity")
        for name in ("candidate_instrument_root", "anchor_instrument_root"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"plan.{name} must be an absolute non-empty path")
            if value is not None and not os.path.isabs(value):
                raise ValueError(f"plan.{name} must be absolute, got {value!r}")
        if not isinstance(self.anchor, api.AnchorIdentity):
            raise TypeError("plan.anchor must be an api.AnchorIdentity — a named immutable "
                            "anchor, not a path")
        for name in ("params", "candidate_param_overrides", "anchor_param_overrides",
                     "unit_param_overrides", "physical_envelopes"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"plan.{name} must be a mapping")
        recipe = recipes.get_recipe(self.recipe_id)
        if recipe.tool == "llama-bench" and not self.params.get("autokernel_seed"):
            seed_material = (
                f"{self.campaign_seed}\0{self.candidate_id}\0{self.attempt}\0"
                f"{self.recipe_id}"
            )
            seed = int.from_bytes(
                hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big"
            ) & ((1 << 63) - 1)
            derived = dict(self.params)
            derived["autokernel_seed"] = seed or 1
            object.__setattr__(self, "params", derived)
        # The recipe registry currently declares one arm-local variant: GGML_IQK.
        # Keeping the allowlist here prevents this seam from becoming a general
        # way to benchmark the candidate and anchor under different cells.
        for name in ("candidate_param_overrides", "anchor_param_overrides"):
            unknown = sorted(set(getattr(self, name)) - {"ggml_iqk"})
            if unknown:
                raise ValueError(
                    f"plan.{name} contains {unknown}; only the recipe-declared "
                    "GGML_IQK env-flag variant may differ by arm")
        if self.stratum not in api.STRATA:
            raise ValueError(f"plan.stratum {self.stratum!r} is not one of {list(api.STRATA)}")
        for name in ("base_blocks", "pairs_per_block"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"plan.{name} must be a positive int")
        if not self.unit_ids:
            raise ValueError("plan.unit_ids must name at least one measurement-material unit")
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("plan.unit_ids must be unique")
        unknown_unit_params = sorted(set(self.unit_param_overrides) - set(self.unit_ids))
        if unknown_unit_params:
            raise ValueError(
                f"plan.unit_param_overrides names unknown units {unknown_unit_params}")
        normalized_unit_params = {}
        for unit_id, overrides in self.unit_param_overrides.items():
            if not isinstance(overrides, Mapping):
                raise TypeError(
                    f"plan.unit_param_overrides[{unit_id!r}] must be a mapping")
            unknown = sorted(set(overrides) - set(recipe.param_map))
            if unknown:
                raise ValueError(
                    f"plan.unit_param_overrides[{unit_id!r}] contains recipe-unknown "
                    f"parameters {unknown}")
            normalized_unit_params[unit_id] = json.loads(schemas.canonical_json(overrides))
        object.__setattr__(self, "unit_param_overrides", normalized_unit_params)
        if not isinstance(self.anti_short_circuit_units, tuple):
            raise TypeError("plan.anti_short_circuit_units must be a tuple")
        if len(set(self.anti_short_circuit_units)) != len(
                self.anti_short_circuit_units):
            raise ValueError("plan.anti_short_circuit_units must be unique")
        unknown_anti = sorted(set(self.anti_short_circuit_units) - set(self.unit_ids))
        if unknown_anti:
            raise ValueError(
                f"plan.anti_short_circuit_units names unknown units {unknown_anti}")
        if self.anti_short_circuit_units:
            normal = set(self.unit_ids) - set(self.anti_short_circuit_units)
            if not normal:
                raise ValueError(
                    "an anti-short-circuit ranked set also needs a normal control unit")
            if set(self.unit_param_overrides) != set(self.unit_ids):
                missing = sorted(set(self.unit_ids) - set(self.unit_param_overrides))
                raise ValueError(
                    "anti-short-circuit units must be real per-unit commands; "
                    f"unit_param_overrides is missing {missing}")
            if self.base_blocks < len(self.unit_ids):
                raise ValueError(
                    f"base_blocks={self.base_blocks} cannot rank all {len(self.unit_ids)} "
                    "declared units at least once")
            normal_frames = {
                schemas.canonical_json(self.unit_param_overrides[unit]) for unit in normal}
            for unit in self.anti_short_circuit_units:
                if schemas.canonical_json(self.unit_param_overrides[unit]) in normal_frames:
                    raise ValueError(
                        f"anti-short-circuit unit {unit!r} has the same recipe parameters "
                        "as a normal unit; relabelling one command does not price the hard case")
        unknown_envelopes = sorted(set(self.physical_envelopes) - set(self.unit_ids))
        if unknown_envelopes:
            raise ValueError(
                f"plan.physical_envelopes names unknown units {unknown_envelopes}")
        if self.physical_envelopes and set(self.physical_envelopes) != set(self.unit_ids):
            missing = sorted(set(self.unit_ids) - set(self.physical_envelopes))
            raise ValueError(
                f"plan.physical_envelopes is partial; missing units {missing}. A physical "
                "screen that grades only the easy cells is not a campaign screen")
        for unit_id, envelope in self.physical_envelopes.items():
            if not isinstance(envelope, physical_bounds.PhysicalEnvelope):
                raise TypeError(
                    f"plan.physical_envelopes[{unit_id!r}] must be a PhysicalEnvelope")
            if envelope.shape_id != unit_id:
                raise ValueError(
                    f"plan.physical_envelopes[{unit_id!r}] declares shape_id "
                    f"{envelope.shape_id!r}; the envelope key and measured unit must be "
                    "identical so a permissive shape cannot grade a harder cell")
            metric = recipes.get_recipe(self.recipe_id).metric
            expected_unit = _DELIVERED_UNIT_BY_METRIC.get(metric)
            if expected_unit is None:
                raise ValueError(
                    f"recipe {self.recipe_id!r} emits metric {metric!r}, which has no "
                    "declared physical-envelope unit; refusing to compare unlike units")
            if envelope.delivered_unit != expected_unit:
                raise ValueError(
                    f"plan.physical_envelopes[{unit_id!r}] is expressed in "
                    f"{envelope.delivered_unit!r}/s, but recipe {self.recipe_id!r} emits "
                    f"{metric!r} in {expected_unit!r}/s")
            frame_params = dict(self.params)
            frame_params.update(self.unit_param_overrides.get(unit_id, {}))
            frame = physical_bounds.measurement_frame_sha256(
                self.recipe_id, frame_params)
            if envelope.measurement_frame_sha256 != frame:
                raise ValueError(
                    f"plan.physical_envelopes[{unit_id!r}] is bound to measurement "
                    f"frame {envelope.measurement_frame_sha256}, but the exact recipe, "
                    f"model, and parameters derive {frame}")
        if self.timeout_s <= 0:
            raise ValueError("plan.timeout_s must be positive")
        if self.segment not in statistics.SEGMENTS:
            raise ValueError(f"plan.segment {self.segment!r} is not one of "
                             f"{list(statistics.SEGMENTS)}")
        if self.segment == statistics.SEGMENT_EXTENSION:
            if not isinstance(self.extension, ExtensionAuthorization):
                raise ExtensionNotDeclared(
                    "plan.segment is 'extension' but the plan carries no "
                    "ExtensionAuthorization. An extension round is a DECLARED "
                    "continuation of the pre-committed stopping rule; a run that can "
                    "extend itself by setting a string is peeking with extra steps.")
            if self.extension.base_blocks != self.base_blocks:
                raise ScheduleMismatch(
                    f"the authorization is for a base segment of "
                    f"{self.extension.base_blocks} blocks but the plan declares "
                    f"base_blocks={self.base_blocks}. The extension EXTENDS the base "
                    f"segment's order schedule — one schedule, indexed straight through — "
                    f"so a plan whose base length disagrees with the authorized one is a "
                    f"RE-DERIVED schedule, and its blocks would carry orders the "
                    f"pre-committed schedule never assigned.")
        elif self.extension is not None:
            raise ExtensionNotDeclared(
                "plan.extension is set on a BASE-segment plan. The base segment is not "
                "authorized by the extension rule and carrying its authorization here "
                "would make `segment` the only thing standing between a base run and an "
                "extension's index line; build the extension plan with `extend()`.")

    @property
    def extension_round(self) -> Optional[int]:
        """The declared round this plan produces, or None on the base segment."""
        return None if self.extension is None else self.extension.round_index

    @property
    def blocks_to_run(self) -> int:
        """How many blocks THIS invocation produces — not how long the base was."""
        return (self.base_blocks if self.segment == statistics.SEGMENT_BASE
                else self.extension.blocks_per_round)

    @property
    def block_index_offset(self) -> int:
        """Where this invocation's blocks start on the run's single index line."""
        return 0 if self.extension is None else self.extension.first_block_index

    def schedule(self) -> statistics.OrderSchedule:
        """The order schedule this plan implies. Derived, never declared.

        Identical for the base plan and every extension plan derived from it:
        `base_blocks` is the BASE segment's length in both, so `order_for()`
        reaches its reversed limb for extension indices. This method is the
        single place the "extended, not re-derived" decision lives.
        """
        return statistics.OrderSchedule.derive(
            campaign_seed=self.campaign_seed,
            candidate_id=(self.matched_experiment_id or self.candidate_id),
            base_blocks=self.base_blocks, attempt=self.attempt)

    def params_for(self, arm: str, unit_id: Optional[str] = None) -> dict:
        """Return the committed recipe parameters for one arm.

        `recipes._P_GGML_IQK` explicitly licenses an env-flag variant "one flag
        per arm".  Until this projection existed the execution layer constructed
        both arms from the same mapping, making that registered control
        unexecutable.  Every non-IQK parameter remains byte-for-byte shared.
        """
        if arm not in (ARM_CANDIDATE, ARM_ANCHOR):
            raise ValueError(f"arm must be candidate or anchor, got {arm!r}")
        if unit_id is not None and unit_id not in self.unit_ids:
            raise ValueError(f"unit_id {unit_id!r} is not declared by this plan")
        override = (self.candidate_param_overrides if arm == ARM_CANDIDATE
                    else self.anchor_param_overrides)
        merged = dict(self.params)
        if unit_id is not None:
            merged.update(self.unit_param_overrides.get(unit_id, {}))
        merged.update(override)
        return merged

    def extend(self, authorization: ExtensionAuthorization) -> "MicrobenchPlan":
        """The plan for one DECLARED extension round of this base plan.

        Every schedule-identity field — `campaign_seed`, `candidate_id`,
        `attempt`, `base_blocks` — rides across unchanged, which is what makes
        `assemble_run_blocks()`'s equality check something the honest path
        satisfies by construction rather than by care.

        Callable only on a base plan. Round 2 comes from the same base plan with
        `round_index=2`, not from round 1's plan: chaining would put the round
        arithmetic in two places, and the second place is where it goes wrong.
        """
        if self.segment != statistics.SEGMENT_BASE:
            raise ExtensionNotDeclared(
                "extend() takes the BASE plan. Every extension round is a continuation of "
                "the base segment's index line, so round N comes from the base plan with "
                "round_index=N — never from round N-1's plan.")
        if not isinstance(authorization, ExtensionAuthorization):
            raise ExtensionNotDeclared("extend() takes an ExtensionAuthorization")
        return replace(self, segment=statistics.SEGMENT_EXTENSION,
                       extension=authorization)

    def to_dict(self) -> dict:
        return {"recipe_id": self.recipe_id, "candidate_id": self.candidate_id,
                "candidate_binding": self.candidate_binding.to_dict(),
                "anchor_binding": self.anchor_binding.to_dict(),
                "anchor": self.anchor.short(), "params": dict(self.params),
                "candidate_param_overrides": dict(self.candidate_param_overrides),
                "anchor_param_overrides": dict(self.anchor_param_overrides),
                "unit_param_overrides": {
                    unit: dict(overrides)
                    for unit, overrides in sorted(self.unit_param_overrides.items())},
                "anti_short_circuit_units": list(self.anti_short_circuit_units),
                "base_blocks": self.base_blocks, "pairs_per_block": self.pairs_per_block,
                "unit_ids": list(self.unit_ids), "stratum": self.stratum,
                "physical_envelopes": {
                    unit: envelope.to_dict()
                    for unit, envelope in sorted(self.physical_envelopes.items())},
                "timeout_s": self.timeout_s, "attempt": self.attempt,
                "segment": self.segment, "extension_round": self.extension_round,
                "blocks_to_run": self.blocks_to_run,
                "block_index_offset": self.block_index_offset,
                "extension": (None if self.extension is None
                              else self.extension.to_dict())}


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
    unit_receipts: Mapping

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
        return self.plan.schedule().check_observed(
            emitted, first_index=self.plan.block_index_offset)

    @property
    def complete(self) -> bool:
        """True only when every requested block completed AND nothing was refused."""
        return (not self.refusals
                and len(self.blocks) == self.plan.blocks_to_run
                and all(b.complete for b in self.blocks)
                and self.order_control.outcome == schemas.PASS)

    def paired_blocks(self) -> tuple:
        """The blocks, for the reducer. Raises `RunRefused` on anything else."""
        if not self.complete:
            raise RunRefused(
                f"this run produced no admissible number and will not pretend otherwise. "
                f"{len(self.blocks)}/{self.plan.blocks_to_run} blocks completed; refusals: "
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
            # Attempt is part of the evidence identity, not just the schedule.
            # Attempts n and n+2 have the same parity-derived order; omitting it
            # would let a completed run be restapled onto a later retry key.
            "attempt": self.plan.attempt,
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
            # The orders spelled out are THIS invocation's window on the one
            # schedule — `first_block_index .. +blocks_to_run`. On the base
            # segment that is 0..base_blocks-1, unchanged; on an extension round
            # it is the reversed limb the round actually ran, so the reader is
            # comparing against the positions the blocks claim rather than
            # against the base segment's slots.
            "order_schedule": dict(
                self.plan.schedule().to_dict(),
                first_block_index=self.plan.block_index_offset,
                orders=[self.plan.schedule().order_for(self.plan.block_index_offset + i)
                        for i in range(self.plan.blocks_to_run)]),
            "segment": self.plan.segment,
            "extension_round": self.plan.extension_round,
            "extension_authorization": (None if self.plan.extension is None
                                        else self.plan.extension.to_dict()),
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
            "unit_receipts": {
                unit: {arm: receipt.to_dict() for arm, receipt in sorted(by_arm.items())}
                for unit, by_arm in sorted(self.unit_receipts.items())},
            "claim_attestations": [a.to_dict() for a in self.claim_attestations],
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def to_dict(self) -> dict:
        return self.raw_vector()

    @property
    def run_id(self) -> str:
        """Content identity of this runner return, including its raw samples."""
        return schemas.content_hash(self.raw_vector())


class CompletedRunLedger:
    """Durable one-run-per-declared-key ledger backed by the primary journal.

    The key is the pre-committed statistical identity that may be spent once:
    ``(campaign_id, candidate_id, attempt, segment, extension_round)``. A
    refused or partial runner return spends the attempt too; repeating it is a
    new ``attempt`` with the retry schedule, never a re-roll under the same key.
    """

    def __init__(self, journal: journal_module.Journal, *, campaign_id: str) -> None:
        if not isinstance(journal, journal_module.Journal):
            raise TypeError("CompletedRunLedger needs the campaign Journal")
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise ValueError("campaign_id must be a non-empty string")
        if journal.campaign_id not in (None, campaign_id):
            raise ValueError(
                f"journal is bound to campaign {journal.campaign_id!r}, not {campaign_id!r}")
        self.journal = journal
        self.campaign_id = campaign_id
        self.journal.initialize()

    def _key(self, plan: MicrobenchPlan) -> tuple:
        return (self.campaign_id, plan.candidate_id, plan.attempt,
                plan.segment, plan.extension_round)

    @staticmethod
    def _payload_key(payload: Mapping[str, Any]) -> tuple:
        return (payload.get("campaign_id"), payload.get("candidate_id"),
                payload.get("attempt"), payload.get("segment"),
                payload.get("extension_round"))

    def _entries(self, plan: MicrobenchPlan) -> tuple:
        key = self._key(plan)
        return tuple(
            entry for entry in self.journal.read_all()
            if entry.kind == journal_module.KIND_MICROBENCH_RUN_COMPLETED
            and self._payload_key(entry.payload) == key)

    def assert_fresh(self, plan: MicrobenchPlan) -> None:
        entries = self._entries(plan)
        if entries:
            ids = sorted({entry.payload.get("run_id") for entry in entries})
            raise RunAlreadyCompleted(
                f"declared run key {self._key(plan)!r} already completed as {ids}; "
                "repeat the measurement as attempt + 1, never as a re-roll of the "
                "observed attempt")

    def record(self, run: MicrobenchRun):
        if not isinstance(run, MicrobenchRun):
            raise TypeError("record() takes the completed MicrobenchRun")
        existing = self._entries(run.plan)
        same = [entry for entry in existing
                if entry.payload.get("run_id") == run.run_id]
        if same:
            return same[0]

        payload = {
            "campaign_id": self.campaign_id,
            "candidate_id": run.plan.candidate_id,
            "attempt": run.plan.attempt,
            "segment": run.plan.segment,
            "extension_round": run.plan.extension_round,
            "run_id": run.run_id,
            "completed_at": run.ended_at,
            "complete": run.complete,
            "raw_vector": run.raw_vector(),
        }
        entry = self.journal.append(
            journal_module.KIND_MICROBENCH_RUN_COMPLETED, payload,
            record_id=run.run_id)
        if existing:
            ids = sorted({old.payload.get("run_id") for old in existing} | {run.run_id})
            raise RunAlreadyCompleted(
                f"declared run key {self._key(run.plan)!r} completed more than once: "
                f"{ids}. The conflicting run was journaled, but neither run may be pooled.")
        return entry

    def assert_poolable(self, runs: Sequence[MicrobenchRun]) -> None:
        for run in runs:
            if not isinstance(run, MicrobenchRun):
                raise TypeError("assert_poolable takes MicrobenchRun values")
            entries = self._entries(run.plan)
            if not entries:
                raise RunNotJournaled(
                    f"run {run.run_id} under declared key {self._key(run.plan)!r} is not "
                    "in the durable completed-run ledger")
            ids = {entry.payload.get("run_id") for entry in entries}
            if len(ids) > 1:
                raise RunAlreadyCompleted(
                    f"declared run key {self._key(run.plan)!r} has conflicting completed "
                    f"runs {sorted(ids)}; selecting one after observing both is alpha spend")
            if run.run_id not in ids:
                raise RunIdentityMismatch(
                    f"pooling supplied run {run.run_id}, but declared key "
                    f"{self._key(run.plan)!r} is journaled as {sorted(ids)}")


class MicrobenchRunner:
    """Runs paired blocks. The only thing in this package that executes a benchmark.

    Construction requires a claim and a spawner; neither has a default. A default
    claim would be a fail-open version of denial 8, and a default spawner would
    make "did this actually run a benchmark?" un-inspectable at the call site.
    """

    def __init__(self, *, claim: HeldClaim, spawner: Spawner,
                 policy: Optional[HostStatePolicy] = None,
                 host_state: Callable[..., HostState] = read_host_state,
                 now: Callable[[], str] = _utc_now,
                 run_ledger: Optional[CompletedRunLedger] = None) -> None:
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
        if run_ledger is not None and not isinstance(run_ledger, CompletedRunLedger):
            raise TypeError("run_ledger must be a CompletedRunLedger or None")
        self._run_ledger = run_ledger

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
        checks.append(("anchor_identity.tool", MicrobenchRunner._check_anchor_tool(plan)))
        return checks

    @staticmethod
    def _check_anchor_tool(plan: MicrobenchPlan) -> schemas.Check:
        """Is the tool the plan's anchor NAMES the tool this run MEASURES with?

        `api.AnchorIdentity.tool` (2026-08-04) carries a RULE, not a label:
        *`binary_sha256` is the digest of the tool the record's `metric` was
        measured with*. This is the only place in the package where both halves
        of that rule are present at once — the plan's anchor and the recipe that
        is about to be spawned — so it is the only place the rule can be
        enforced rather than trusted.

        Without it the digest check alone does not close the hole. A driver that
        captures the anchor `llama-bench` correctly and then binds it
        `chain.bind_anchor(capture, tool="llama-cli")` — a copy-paste from the T0
        leg, which is the leg that legitimately names `llama-cli` — produces a
        plan whose digest matches the binary that runs, whose every existing
        check PASSes, and whose record renders `vs anchor llama-cli:<digest>/…`
        for a ratio `llama-bench` measured. `for_tool` refuses a RE-label, but the
        FIRST label is a free string and `bind_anchor` accepts any of them.

        COULD_NOT_CHECK when the anchor names no tool: a record predating the
        field is readable and must stay so, and an unnamed anchor is not evidence
        that the tool agrees. Recorded rather than omitted, for the reason the
        linkage conjunct above is — an unverified conjunct that appears nowhere
        is indistinguishable from one that passed.
        """
        measured_with = recipes.get_recipe(plan.recipe_id).tool
        if plan.anchor.tool is None:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"the plan's anchor names no tool, so whether its "
                f"{plan.anchor.binary_sha256[:12]} is the digest of the {measured_with!r} "
                f"this run measures with is UNOBSERVED. Bind it with "
                f"`chain.bind_anchor(capture, tool=…)`; not naming a tool is not evidence "
                f"that it is the right one",))
        if plan.anchor.tool != measured_with:
            return schemas.Check(schemas.FAIL, (
                f"the plan's anchor is named {plan.anchor.tool!r} but recipe "
                f"{plan.recipe_id!r} measures with {measured_with!r}. `binary_sha256` is "
                f"the digest of the tool the record's metric was measured with, so this "
                f"record would render `vs anchor {plan.anchor.tool}:"
                f"{plan.anchor.binary_sha256[:12]}…` as the denominator of a ratio "
                f"{measured_with} produced — a denominator naming a binary that never "
                f"ran. The digest check cannot see this: one anchor BUILD ships both "
                f"tools, so the label can be wrong while the bytes are right",))
        return schemas.Check(schemas.PASS, (
            f"the plan's anchor names {measured_with!r}, the tool recipe "
            f"{plan.recipe_id!r} measures with",))

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

    def run(
            self, plan: MicrobenchPlan, *,
            completed_prefix: Sequence[ReplayedBlockRecord] = (),
            original_started_at: Optional[str] = None,
            on_block_completed: Optional[Callable[[BlockRecord], None]] = None,
    ) -> MicrobenchRun:
        """Execute the unmeasured suffix of a plan.

        ``completed_prefix`` is admitted only through
        :func:`replay_completed_block`; callers cannot hand the runner an
        arbitrary partial result.  The callback runs synchronously immediately
        after a complete block is assembled and before the next block opens,
        which is the sole durable checkpoint boundary: invocations inside a
        block are never reusable.
        """
        if not isinstance(plan, MicrobenchPlan):
            raise TypeError("run() takes a MicrobenchPlan")
        if not isinstance(completed_prefix, Sequence) \
                or any(not isinstance(row, ReplayedBlockRecord)
                       for row in completed_prefix):
            raise TypeError(
                "completed_prefix must contain only replay_completed_block results")
        if on_block_completed is not None and not callable(on_block_completed):
            raise TypeError("on_block_completed must be callable or None")
        if plan.segment == statistics.SEGMENT_EXTENSION and self._run_ledger is None:
            raise RunLedgerRequired(
                "an extension round requires a durable CompletedRunLedger; otherwise the "
                "same declared round can be re-run after its result is observed")
        if self._run_ledger is not None:
            self._run_ledger.assert_fresh(plan)
        started_at = original_started_at or self._now()
        if not isinstance(started_at, str) or not started_at.strip():
            raise ValueError("original_started_at must be a non-empty timestamp")
        checks: list = []
        refusals: list = []
        attestations: list = []
        blocks: list = list(completed_prefix)
        # Every frequency classification this run produced, in order. `_finish`
        # refuses a run in which none of them is JUDGED: deferring the idle
        # readings is only safe if SOMETHING eventually judged the clock under
        # load, and this list is what makes that a structural property of the
        # run rather than a hope about the load average.
        freq_classifications: list = []
        process_identities: set[tuple[int, int]] = set()
        for recovered in completed_prefix:
            for invocation in recovered.raw["invocations"]:
                claim = invocation["claim"]
                attestations.append(ClaimAttestation(
                    claim_id=str(claim["claim_id"]), holder=str(claim["holder"]),
                    cpu_list=str(claim["cpu_list"]),
                    observed_at=str(claim["observed_at"]),
                    check=schemas.Check(str(claim["outcome"]),
                                        tuple(claim.get("reasons", ())))))
                sandbox_receipt = invocation["spawn"].get("sandbox_receipt")
                if isinstance(sandbox_receipt, Mapping) \
                        and isinstance(sandbox_receipt.get("pid"), int) \
                        and isinstance(sandbox_receipt.get("process_start_ticks"), int):
                    process_identities.add((sandbox_receipt["pid"],
                                            sandbox_receipt["process_start_ticks"]))
            for name, check in recovered.raw["checks"]:
                if name == "host_frequency_block_close" \
                        and isinstance(check, Mapping) \
                        and check.get("outcome") == schemas.PASS:
                    freq_classifications.append(FREQUENCY_JUDGED)

        commands = {
            unit: {
                ARM_CANDIDATE: recipes.construct(
                    plan.recipe_id, binding=plan.candidate_binding,
                    params=plan.params_for(ARM_CANDIDATE, unit),
                    arm=ARM_CANDIDATE, verify_inputs=False),
                ARM_ANCHOR: recipes.construct(
                    plan.recipe_id, binding=plan.anchor_binding,
                    params=plan.params_for(ARM_ANCHOR, unit),
                    arm=ARM_ANCHOR, verify_inputs=False),
            }
            for unit in plan.unit_ids
        }
        first_unit = plan.unit_ids[0]
        multi_unit = len(plan.unit_ids) > 1
        scope = commands[first_unit][ARM_CANDIDATE].scope_denominator
        footprint = commands[first_unit][ARM_CANDIDATE].claim_footprint

        envs: dict = {}
        receipts: dict = {}
        expectations: dict = {}
        for unit, unit_commands in commands.items():
            envs[unit] = {}
            receipts[unit] = {}
            expectations[unit] = {}
            for arm, command in unit_commands.items():
                if command.scope_denominator != scope or command.claim_footprint != footprint:
                    refusals.append(
                        f"{unit}/{arm}: per-unit parameters changed the claimed scope or "
                        "CPU footprint; ranked cases may vary work, never resource identity")
                assembly = assemble_env(command.env)
                envs[unit][arm] = assembly.env
                discipline = check_recipe_discipline(command, assembly.env)
                discipline_name = (f"recipe_discipline.{unit}.{arm}" if multi_unit
                                   else f"recipe_discipline.{arm}")
                checks.append((discipline_name, discipline))
                if discipline.outcome != schemas.PASS:
                    refusals.append(f"{unit}/{arm}: {'; '.join(discipline.reasons)}")
                try:
                    receipts[unit][arm] = build_receipt(command, env=assembly.env)
                except (OSError, integrity.IntegrityError) as exc:
                    refusals.append(
                        f"{unit}/{arm}: cannot digest the binary that would run: {exc}")
                expectations[unit][arm] = LlamaBenchExpectation.from_command(
                    command,
                    expected_build_commit=(plan.anchor.source_commit
                                           if arm == ARM_ANCHOR else None))
                for finding in command.discipline:
                    if finding.check.outcome == schemas.PASS:
                        continue
                    delegated_name = (
                        f"delegated.{unit}.{arm}.{finding.finding_id}" if multi_unit
                        else f"delegated.{arm}.{finding.finding_id}")
                    checks.append((delegated_name, finding.check))
                # `verify_inputs=False` above turns every input check into one
                # COULD_NOT_CHECK. It must remain visible in the record.
                if not command.inputs_verified:
                    inputs_name = (f"inputs_verified.{unit}.{arm}" if multi_unit
                                   else f"inputs_verified.{arm}")
                    checks.append((inputs_name, schemas.Check(
                        schemas.COULD_NOT_CHECK,
                        tuple(r for c in command.input_checks for r in c.reasons))))

            anchor_checks = self._check_anchor_identity(
                plan, receipts[unit].get(ARM_ANCHOR))
            for name, check in anchor_checks:
                qualified = f"{name}.{unit}" if multi_unit else name
                checks.append((qualified, check))
                if check.outcome == schemas.FAIL:
                    refusals.append(f"{qualified}: {'; '.join(check.reasons)}")

        if getattr(self._spawner, "spawner_id", None) == "subprocess/v1":
            if plan.candidate_instrument_root is None:
                source_pin = schemas.Check(schemas.FAIL, (
                    "live measurement has no explicit candidate/anchor instrument source "
                    "roots; ToolBinding.source_root names a build closure and is not a "
                    "translation-unit authority",))
            else:
                source_pin = instrument_integrity.compare_manifest_to_anchor(
                    candidate_root=plan.candidate_instrument_root,
                    anchor_root=plan.anchor_instrument_root)
            checks.append(("measurement_source_pin", source_pin))
            if source_pin.outcome != schemas.PASS:
                refusals.append(
                    "measurement source pin: " + "; ".join(source_pin.reasons))
        else:
            checks.append(("measurement_source_pin", schemas.Check(
                schemas.COULD_NOT_CHECK, (
                    "recorded/fixture spawner launched no measured binary; the live "
                    "SubprocessSpawner re-hashes candidate and anchor translation units "
                    "at run open and before every invocation",))))

        if getattr(self._spawner, "spawner_id", None) == "subprocess/v1":
            if not plan.physical_envelopes:
                physical_screen = schemas.Check(schemas.FAIL, (
                    "live measurement has no predeclared RVP-C6-4 physical envelope; "
                    "a missing speed-of-light screen cannot be interpreted as below the "
                    "ceiling",))
            else:
                physical_screen = schemas.Check(schemas.PASS, (
                    f"RVP-C6-4 envelopes are predeclared for all {len(plan.unit_ids)} "
                    "measurement units",))
        else:
            physical_screen = schemas.Check(schemas.COULD_NOT_CHECK, (
                "recorded/fixture spawner launches no live benchmark; the live runner "
                "requires a per-unit physical envelope and checks every emitted sample",))
        checks.append(("physical_speed_of_light_predeclared", physical_screen))
        if physical_screen.outcome == schemas.FAIL:
            refusals.append("physical speed-of-light screen: "
                            + "; ".join(physical_screen.reasons))

        # Host state at OPEN. Contention is judged here and only here: once the
        # benchmark is running it saturates the claimed cores itself, so a
        # mid-run load reading measures this runner, not a foreign process.
        open_state = self._read_host_state(cpu_list=footprint.cpu_list)
        # `under_load=False` is a fact about this line's position in the loop,
        # not a reading: nothing has been spawned yet, so the claimed footprint
        # is idle by construction.
        freq_class, freq_check = self._policy.frequency_verdict(
            open_state, under_load=False)
        freq_classifications.append(freq_class)
        load_check = self._policy.check_load(open_state, cpu_count=footprint.cpu_count)
        checks.append(("host_frequency_open", freq_check))
        checks.append(("host_load_open", load_check))
        # The frequency reading at run open is DEFERRED, not refused, when the
        # host is idle. It has to be: `load_check` PASSes only below the very
        # load threshold above which the clock reading means anything, so a host
        # that passes contention here can never pass frequency here. Refusing on
        # it made the gate unsatisfiable on a healthy machine — see
        # HostStatePolicy's docstring. Every OTHER non-PASS still refuses, so a
        # disabled, unreadable or reference-less check is as fatal as it was.
        if freq_class != FREQUENCY_DEFERRED_IDLE and freq_check.outcome != schemas.PASS:
            refusals.append(f"host frequency at run open: {freq_check.outcome} — "
                            f"{'; '.join(freq_check.reasons)}")
        if self._policy.require_load and load_check.outcome != schemas.PASS:
            refusals.append(f"host contention at run open: {load_check.outcome} — "
                            f"{'; '.join(load_check.reasons)}")

        if refusals:
            return self._finish(plan, started_at, blocks, refusals, checks, scope,
                                attestations, receipts, freq_classifications)

        # This must use the plan's one derivation site: a matched intervention
        # and A/A control share ``matched_experiment_id`` as their schedule key.
        schedule = plan.schedule()
        # `count` is `blocks_to_run`, not `base_blocks`: on an extension round
        # the schedule still has the BASE segment's length (that is what makes
        # `order_for` reverse past it) while the number of blocks to produce is
        # the rule's declared `blocks_per_round`.
        plans = plan_blocks(schedule, count=plan.blocks_to_run, pairs=plan.pairs_per_block,
                            unit_ids=plan.unit_ids, stratum=plan.stratum,
                            segment=plan.segment, extension_round=plan.extension_round)

        if len(completed_prefix) > len(plans):
            raise ScheduleMismatch(
                f"checkpoint prefix has {len(completed_prefix)} blocks but the current "
                f"plan declares only {len(plans)}")
        for index, recovered in enumerate(completed_prefix):
            if recovered.plan != plans[index]:
                raise ScheduleMismatch(
                    f"checkpoint prefix block {index} no longer matches the exact "
                    "derived plan/order slot")

        for block_plan in plans[len(completed_prefix):]:
            record = self._run_block(plan, block_plan, commands, envs, receipts,
                                     expectations, footprint, attestations,
                                     freq_classifications, process_identities)
            blocks.append(record)
            if not record.complete:
                refusals.extend(record.refusals)
                break
            if on_block_completed is not None:
                on_block_completed(record)

        return self._finish(plan, started_at, blocks, refusals, checks, scope,
                            attestations, receipts, freq_classifications)

    def _run_block(self, plan: MicrobenchPlan, block_plan: BlockPlan, commands: Mapping,
                   envs: Mapping, receipts: Mapping, expectations: Mapping,
                   footprint: recipes.ClaimFootprint, attestations: list,
                   freq_classifications: list,
                   process_identities: set) -> BlockRecord:
        open_state = self._read_host_state(cpu_list=footprint.cpu_list)
        # Between blocks the claimed cores are idle by construction too: the
        # previous block's last invocation has already exited.
        open_class, open_freq = self._policy.frequency_verdict(
            open_state, under_load=False)
        freq_classifications.append(open_class)
        checks: list = [("host_frequency_block_open", open_freq)]
        refusals: list = []
        invocations: list = []
        unit_commands = commands[block_plan.unit_id]
        unit_envs = envs[block_plan.unit_id]
        unit_receipts = receipts[block_plan.unit_id]
        unit_expectations = expectations[block_plan.unit_id]

        # Same deferral as at run open, and for the same reason: between blocks
        # the claimed cores are idle by construction, so the first block would
        # otherwise refuse on every healthy host. The block CLOSE reading below
        # is the one taken under the benchmark's own load, and it still refuses.
        if open_class != FREQUENCY_DEFERRED_IDLE and open_freq.outcome != schemas.PASS:
            return BlockRecord(plan=block_plan, invocations=(), host_state_open=open_state,
                               host_state_close=None, paired_block=None,
                               checks=tuple(checks),
                               refusals=(f"block {block_plan.block_index}: host frequency "
                                         f"{open_freq.outcome} — "
                                         f"{'; '.join(open_freq.reasons)}",))

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

            command = unit_commands[arm]
            # Everything from here to the parse is driven by material this runner
            # does not control — the filesystem, the process, and the tool's own
            # stdout. A `MicrobenchError` raised out of `run()` would take the
            # whole campaign's raw vector with it, and "a failure that is not
            # durable is indistinguishable from a run that never happened". So
            # the loop's own errors become refusals; a `TypeError` from a Spawner
            # that breaks its contract still raises, because that is a defect in
            # the caller's code and not a fact about the host.
            inv_checks: list = []
            try:
                if getattr(self._spawner, "spawner_id", None) == "subprocess/v1":
                    if plan.candidate_instrument_root is None:
                        source_pin = schemas.Check(schemas.FAIL, (
                            "live measurement has no explicit instrument source roots",))
                    else:
                        source_pin = instrument_integrity.compare_manifest_to_anchor(
                            candidate_root=plan.candidate_instrument_root,
                            anchor_root=plan.anchor_instrument_root)
                    inv_checks.append(("measurement_source_pin", source_pin))
                    if source_pin.outcome != schemas.PASS:
                        raise SpawnFailure(
                            "measurement translation unit changed before the invocation: "
                            + "; ".join(source_pin.reasons))
                self._attest_binary(
                    arm=arm, command=command, receipt=unit_receipts[arm],
                    when=f"before block {block_plan.block_index} position {position}")
                spawn = self._spawner.run(command.argv, unit_envs[arm],
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
            if getattr(self._spawner, "spawner_id", None) == "subprocess/v1":
                receipt = spawn.sandbox_receipt
                if receipt is None:
                    refusals.append(
                        f"block {block_plan.block_index} position {position} ({arm}): "
                        "the live subprocess carries no C6 activation receipt; an "
                        "unsandboxed candidate process is not an admissible arm")
                else:
                    identity = (receipt["pid"], receipt["process_start_ticks"])
                    if identity in process_identities:
                        refusals.append(
                            f"block {block_plan.block_index} position {position} ({arm}): "
                            f"process identity {identity} was already used by another arm; "
                            "an import/init-gated variant requires a fresh process per arm")
                    else:
                        process_identities.add(identity)
                        inv_checks.append(("fresh_process_per_arm", schemas.Check(
                            schemas.PASS, (
                                f"fresh pid/start identity {identity} under invocation-only "
                                "writable state; no process or filesystem memo survives "
                                "from the preceding arm",))))
            device_state_check = check_gpu_device_sampling(
                spawn.device_sampling_receipt,
                n_gpu_layers=unit_expectations[arm].n_gpu_layers,
                live_subprocess=(
                    getattr(self._spawner, "spawner_id", None) == "subprocess/v1"))
            inv_checks.append(("gpu_device_state_window", device_state_check))
            if device_state_check.outcome != schemas.PASS:
                refusals.append(
                    f"block {block_plan.block_index} position {position} ({arm}): "
                    + "; ".join(device_state_check.reasons))
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
                    agreement = unit_expectations[arm].check_row(row)
                    inv_checks.append(("output_matches_recipe", agreement))
                    if agreement.outcome != schemas.PASS:
                        refusals.append(f"block {block_plan.block_index} position "
                                        f"{position} ({arm}): "
                                        f"{'; '.join(agreement.reasons)}")
                    else:
                        duration_check = check_gpu_ranked_duration_windows(
                            row.samples_ns, spawn.device_sampling_receipt,
                            n_gpu_layers=unit_expectations[arm].n_gpu_layers,
                            live_subprocess=(
                                getattr(self._spawner, "spawner_id", None)
                                == "subprocess/v1"))
                        inv_checks.append(("gpu_absolute_duration_window", duration_check))
                        if duration_check.outcome != schemas.PASS:
                            refusals.append(
                                f"block {block_plan.block_index} position {position} "
                                f"({arm}): {'; '.join(duration_check.reasons)}")
                        samples = row.metric_samples
                        envelope = plan.physical_envelopes.get(block_plan.unit_id)
                        if envelope is not None:
                            physical_check = envelope.check_throughput(samples)
                            inv_checks.append(("physical_speed_of_light", physical_check))
                            if physical_check.outcome != schemas.PASS:
                                refusals.append(
                                    f"block {block_plan.block_index} position {position} "
                                    f"({arm}): {'; '.join(physical_check.reasons)}")

            invocations.append(Invocation(
                block_index=block_plan.block_index, position=position, arm=arm,
                receipt=unit_receipts[arm], spawn=spawn, row=row, samples=samples,
                claim=attestation, checks=tuple(inv_checks)))

            if refusals:
                break

        # Block CLOSE is the reading that matters. The benchmark has just driven
        # the claimed footprint, so the clock here is evidence about the host
        # rather than about its idleness, and a FAIL is a real throttle. This is
        # where the guard the idle deferral preserved actually bites.
        # `bool(invocations)` and not a bare True: a block that broke out before
        # spawning anything did NOT drive the footprint, and claiming it did
        # would let an empty block manufacture the JUDGED reading that the
        # run-level control in `_finish` is looking for.
        close_state = self._read_host_state(cpu_list=footprint.cpu_list)
        # A short benchmark can park every core between `wait()` returning and
        # this sysfs read.  The real spawner therefore samples while its exact
        # captured PID is alive and returns the per-cpu peaks.  Prefer those
        # in-process readings; the close read remains the fallback for recorded
        # spawners and the provenance still carries both sampling points.
        peak: dict[int, int] = {}
        for invocation in invocations:
            for cpu, khz in invocation.spawn.khz_peak_by_cpu:
                peak[cpu] = max(peak.get(cpu, 0), khz)
        if peak:
            close_state = replace(
                close_state, khz_by_cpu=tuple(sorted(peak.items())),
                source=f"{close_state.source}+subprocess_lifetime_peak")
        close_class, close_freq = self._policy.frequency_verdict(
            close_state, under_load=bool(invocations))
        freq_classifications.append(close_class)
        checks.append(("host_frequency_block_close", close_freq))
        package_power_check, _package_power = derive_package_power_attestation(
            open_state, close_state)
        checks.append(("host_package_power_block", package_power_check))
        if self._policy.require_package_power \
                and package_power_check.outcome != schemas.PASS:
            refusals.append(
                f"block {block_plan.block_index}: package-power attestation "
                f"{package_power_check.outcome} — "
                f"{'; '.join(package_power_check.reasons)}")
        if close_class != FREQUENCY_DEFERRED_IDLE and close_freq.outcome != schemas.PASS:
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
                receipts: Mapping,
                freq_classifications: Optional[Sequence[str]] = None) -> MicrobenchRun:
        # THE CONTROL ON THE IDLE DEFERRAL. Deferring an idle frequency reading
        # is only sound because the run is about to load the host itself; if it
        # never did — every reading deferred, start to finish — then the throttle
        # guard never ran, and the multi-day throttle that poisoned every number
        # in its window (`feedback_host_throttle_check`) would sail straight
        # through. A guard that can be satisfied by never evaluating it is worth
        # nothing, so a run with blocks and no JUDGED reading emits no number.
        #
        # Scoped to runs that actually produced blocks: a run that refused
        # before planning any has a real refusal already and does not need a
        # second, more confusing one.
        classifications = tuple(freq_classifications or ())
        if blocks and FREQUENCY_JUDGED not in classifications:
            refusals.append(
                f"the host frequency was never judged under load: "
                f"{len(classifications)} reading(s), classified "
                f"{sorted(set(classifications))}, and not one of them was "
                f"{FREQUENCY_JUDGED!r}. Idle readings are DEFERRED so a healthy quiet "
                f"host can start a run; a run that stays under the load threshold to the "
                f"end never exercised the throttle guard at all, and this host has sat at "
                f"-60% for days undetected. No number is emitted.")
        if len(blocks) < plan.blocks_to_run and not refusals:
            refusals.append(
                f"only {len(blocks)}/{plan.blocks_to_run} paired blocks completed; a run "
                f"short of its declared block count does not emit a number")
        single_receipts = receipts.get(plan.unit_ids[0], {}) \
            if len(plan.unit_ids) == 1 else {}
        run = MicrobenchRun(
            plan=plan, runner_id=RUNNER_ID, started_at=started_at, ended_at=self._now(),
            blocks=tuple(blocks), refusals=tuple(refusals), checks=tuple(checks),
            scope_denominator=scope, claim_attestations=tuple(attestations),
            candidate_receipt=single_receipts.get(ARM_CANDIDATE),
            anchor_receipt=single_receipts.get(ARM_ANCHOR),
            unit_receipts=receipts)
        # The reducer's own order control is NOT copied into `checks` here.
        # `MicrobenchRun.order_control` re-derives it from the plan on every
        # read, `complete` is conjoined with it, and `raw_vector()` emits it —
        # one computation site, deliberately. A frozen copy taken at the end of
        # the run would be a second thing to stub, and inside this runner it can
        # only ever say PASS anyway: the loop runs the schedule it derived, so
        # the tautological verdict is worth nothing and the recomputed one is
        # what stops a completed run being relabelled with another plan.
        if run.order_control.outcome != schemas.PASS and not refusals \
                and len(blocks) == plan.blocks_to_run:
            refusals.append(
                f"the emitted blocks do not satisfy the order schedule derived from "
                f"the committed campaign seed: {run.order_control.outcome} — "
                f"{'; '.join(run.order_control.reasons)}")
            run = replace(run, refusals=tuple(refusals))
        if self._run_ledger is not None:
            self._run_ledger.record(run)
        return run


# =============================================================================
# Pooling the base segment and its extension rounds
# =============================================================================

def assemble_run_blocks(base_run: MicrobenchRun,
                        extension_runs: Sequence[MicrobenchRun] = (), *,
                        campaign: statistics.CampaignStatistics,
                        run_ledger: Optional[CompletedRunLedger] = None) -> tuple:
    """The one block sequence the reducer reads: base segment, then whole rounds.

    Something has to concatenate a base run with the extension rounds that
    followed it, and doing it at the call site with `base + ext` is exactly
    where two different schedules get pooled into one candidate's evidence.
    `statistics._check_extension_structure` would catch some of that at
    reduction time — interleaving, a short round, more rounds than declared —
    but it cannot see what it is not told: it never sees the campaign seed, the
    attempt, the recipe or the bindings, so two runs of DIFFERENT candidates or
    under two different seeds reduce cleanly if their segments line up. This
    function refuses first, and raises, because a pooled block set is not
    something to journal as INVALID — it is a set that should never have been
    built.

    The runs must be the same plan in everything except `segment` and
    `extension` (a total comparison rather than a field list: a field list is a
    thing to forget to update, and every field of a plan is either a
    measurement condition or a refusal bound, both of which must hold across a
    pooled run). Rounds must be consecutive from 1, each exactly once, and the
    resulting indices must be contiguous from 0 — the shape
    `statistics._check_block_identity` requires downstream.

    Every run must be `complete`; an incomplete one raises `RunRefused` through
    `paired_blocks()`, which is the same refusal a caller would have got from
    the run on its own.

    `campaign` is REQUIRED and has no default. This is the only place the runs
    and the campaign whose threshold they will be judged against are in the same
    frame, so it is the only place "was this round licensed by THIS campaign?"
    can be asked — `ExtensionAuthorization.licence_for`. A default of `None`
    here would be the fail-open version of the whole guard: the honest caller
    passes it, the pooled-in round from somewhere else does not, and the check
    would be skipped exactly when it matters.
    """
    if not isinstance(base_run, MicrobenchRun):
        raise TypeError("assemble_run_blocks takes a MicrobenchRun as the base run")
    if not isinstance(campaign, statistics.CampaignStatistics):
        raise TypeError(
            "assemble_run_blocks needs the statistics.CampaignStatistics this evidence "
            "will be reduced under; the pooled set is where a round licensed by another "
            "campaign would otherwise enter this campaign's record")
    runs = tuple(extension_runs)
    for run in runs:
        if not isinstance(run, MicrobenchRun):
            raise TypeError("every extension run must be a MicrobenchRun")
    if runs and run_ledger is None:
        raise RunLedgerRequired(
            "pooling extension evidence requires the durable CompletedRunLedger that "
            "recorded every completed run under its declared key")
    if run_ledger is not None:
        if not isinstance(run_ledger, CompletedRunLedger):
            raise TypeError("run_ledger must be a CompletedRunLedger")
        run_ledger.assert_poolable((base_run,) + runs)
    if base_run.plan.segment != statistics.SEGMENT_BASE:
        raise ScheduleMismatch(
            f"the base run's plan is segment {base_run.plan.segment!r}; the base segment "
            f"is what an extension extends and it cannot itself be an extension round")
    if base_run.plan.campaign_seed != campaign.campaign_seed:
        raise ScheduleMismatch(
            "the base run was planned under a different committed campaign seed than the "
            "campaign it is being pooled for; the order schedule this run obeyed is not "
            "the one the reduction will check it against")
    if base_run.plan.base_blocks != campaign.b_min:
        raise ScheduleMismatch(
            f"the base segment is {base_run.plan.base_blocks} blocks but this campaign's "
            f"calibrated B_min is {campaign.b_min}; the base segment is exactly B_min "
            f"blocks and everything beyond it is a declared extension round")

    identity = replace(base_run.plan, segment=statistics.SEGMENT_BASE, extension=None)
    rounds: dict = {}
    for run in runs:
        plan = run.plan
        if plan.segment != statistics.SEGMENT_EXTENSION or plan.extension is None:
            raise ScheduleMismatch(
                f"a run passed as an extension round carries segment {plan.segment!r} with "
                f"no authorization; only a declared extension round may follow the base "
                f"segment")
        licence = plan.extension.licence_for(campaign)
        if licence.outcome != schemas.PASS:
            raise ExtensionNotDeclared(
                f"extension round {plan.extension.round_index} is not licensed by this "
                f"campaign: {licence.outcome} — {'; '.join(licence.reasons)}")
        if replace(plan, segment=statistics.SEGMENT_BASE, extension=None) != identity:
            raise ScheduleMismatch(
                f"extension round {plan.extension.round_index} was produced under a "
                f"different plan than the base segment (seed/candidate/attempt/base_blocks "
                f"identify the ONE order schedule; recipe, params, anchor and bindings "
                f"identify the ONE instrument). Pooling them to the pre-declared threshold "
                f"would pool two experiments.")
        if plan.extension.round_index in rounds:
            raise ScheduleMismatch(
                f"extension round {plan.extension.round_index} was submitted twice; a "
                f"round is a fixed number of fresh pairs, not a resubmittable one")
        rounds[plan.extension.round_index] = run

    expected = list(range(1, len(rounds) + 1))
    if sorted(rounds) != expected:
        raise ScheduleMismatch(
            f"extension rounds {sorted(rounds)} are not consecutive from 1; a run cannot "
            f"skip a declared round and keep the ones after it")

    blocks: list = list(base_run.paired_blocks())
    for round_index in expected:
        blocks.extend(rounds[round_index].paired_blocks())
    for position, block in enumerate(blocks):
        if block.block_index != position:
            raise ScheduleMismatch(
                f"the pooled blocks are not one contiguous index line: position "
                f"{position} carries block_index {block.block_index}. Base and extension "
                f"are ONE schedule, indexed straight through.")
    return tuple(blocks)


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
