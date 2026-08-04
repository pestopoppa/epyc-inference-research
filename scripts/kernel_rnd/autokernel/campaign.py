#!/usr/bin/env python3
"""campaign.py — the campaign driver. THE ENTRYPOINT this package did not have.

    python3 -m scripts.kernel_rnd.autokernel.campaign --help      # from the repo root

WHY THIS FILE EXISTS
--------------------
Until it was written, `grep -rln '__main__|argparse|def main('` over every
non-test module in this package returned NOTHING. 94k lines, 5,695 passing
tests, and no way to start any of it. That — not a missing gate, not a missing
statistic — is why AutoKernel has produced no results. The executors, the
evaluator, the journal and the seams (`execution/chain.py`) were all built and
green; nothing composed them into something a shell could run.

So this module is a DRIVER, not machinery. It owns exactly four things that
live nowhere else:

  1. **The order of the loop**, and the two places that order is load-bearing —
     T0 before any speed number at all, and the claim/teardown discipline on
     every exit path including the failure ones.
  2. **The accept rule**, which is thirty lines of arithmetic justified below
     from a measurement rather than from principle.
  3. **The falsifier-before-compute gate** (`--hypothesis`). The driver is what
     SPENDS the claim, so the gate that decides whether a question may be spent
     on belongs here and nowhere else. Without `--hypothesis` a campaign is
     EXPLORATORY and the record says so in as many words; with it, the claim is
     acquired through `hypotheses.claim_for_hypothesis` and a question with no
     falsifier — or a placeholder one — cannot reach a claim at all. See the
     section by that name below for what was broken before it was wired.
  4. **The boundary.** `MODULES_THE_DRIVER_USES` below is the whole of what a
     campaign needs. `MODULES_DELIBERATELY_NOT_USED` is the rest of the
     package, and `test_campaign.py` enforces both against this file's own AST.
     Nothing here deletes anything; the boundary is drawn so that deletion
     stays a one-line decision for the operator rather than one this file
     pre-empts.

`execution/README.md` is the runbook for everything below, and
`execution/test_execution_chain.py::ChainLeg` is the reference composition of
the seams. Read them before editing the host path.

THE EVIDENCE THIS FILE IS BUILT ON
----------------------------------
`data/autokernel_aa_20260804/` — four A/A runs of IDENTICAL code, on a quiet
host, under the ratified canonical recipe. It is the first real measurement this
package has ever had, and it settles three design questions that used to be
argued:

    pp512   899.95  894.70  867.16  886.16     between-run CV 1.62%, spread 3.70%
    tg128    52.76   52.31   51.62   50.52     between-run CV 1.88%, spread 4.32%

  * **A single-run A/B here can be fooled by ~4% of pure noise.** An n=1 strict
    `<` accept rule is a coin flip with a decimal point.
  * **Decode declined MONOTONICALLY across four consecutive runs** — 52.76 →
    52.31 → 51.62 → 50.52. That is drift, not scatter, and its consequence is
    the important one: an A/B that runs candidate-then-anchor charges the second
    arm a systematic ~4% penalty, and MORE REPETITIONS DO NOT REMOVE IT — they
    measure the drift more precisely. Interleaved paired blocks are therefore
    the minimum correct design. `microbench.BlockPlan` already derives the
    alternating sequence structurally and refuses a blocked design; this driver
    never plans blocks any other way.
  * **1.6–1.9% CV does not justify an e-process.** Pairing, a pre-committed N
    and a median cover it. `evaluator/statistics.py` solved a harder problem
    than we have, and its e-process is not imported here — see the boundary.

DRY RUN IS THE DEFAULT, AND THAT IS A SAFETY PROPERTY
-----------------------------------------------------
`--dry-run` is on unless `--execute` is passed. A driver on a shared host that
can benchmark by accident is worse than no driver: two of the six A/A runs were
destroyed by a legitimate co-tenant bringing up seven `llama-server` instances,
and the co-tenant did nothing wrong — nothing told it the host was in use.
A dry run composes every step, prints the exact argv and env that WOULD be
spawned, and executes nothing at all. It emits no speed number, because a dry
run that produces a number is a real run with a flag on it.

`--execute` additionally requires `--i-hold-the-host`, spelled out, because the
loop below spends hours of a claim.

THE ACCEPT RULE
---------------
`decide()` is the whole of it. In words:

  1. **T0 all-PASS, lexicographically prior.** If T0 does not pass, NO SPEED
     NUMBER IS COMPUTED AT ALL — `run_campaign` returns before the blocks are
     planned, and `decide()` raises if called anyway. Throughput is
     reward-hackable: deleting the computation is the fastest kernel there is.
     The predecessor harness (`kernel_eval.sh`) tested `MUL_MAT` only, so a
     kernel that broke `MUL_MAT_ID` — MoE dispatch, on EVERY token in
     production — passed it cleanly. `require_op_suite_covers_moe_dispatch()`
     below makes that specific hole structural.
  2. Over the PRE-COMMITTED N alternating pairs, `delta_i = candidate_i -
     anchor_i`, and `relative_i = delta_i / anchor_i`.
  3. **The order control, first of all.** `statistics.OrderSchedule` draws each
     block's order from a per-block hash of the campaign seed — a coin flip per
     block, NOT an alternation — so five blocks land all one way once in
     sixteen runs, and that run is a sequential A/B. It is INADMISSIBLE in both
     directions: the within-block slot effect is perfectly confounded with the
     candidate effect there, and the anchor control below cannot see it either
     (a host that boosts at each block's start and sags inside it leaves the
     anchor series flat across blocks). The cost is a false negative once in
     sixteen runs, which is the direction this rule is deliberately wrong in.
  4. **The neutral control, next**: if the ANCHOR arm — identical code from
     block to block — moved further across the run than an A/A of identical
     code did, the run is INADMISSIBLE. Interleaving handles drift within a
     comparison; nothing else tells you the instrument moved at all, and *a
     campaign that never runs a control cannot distinguish "this kernel is 4%
     faster" from "this kernel ran first"*. It costs no extra claim time: the
     anchor blocks the run already produced ARE that A/A.
  5. **KEEP iff `min(delta) > 0` AND `median(relative) > drift_bound`.**
  6. Otherwise REVERT.

Why those two conjuncts and no more, from the A/A rather than from theory:

  * `min(delta) > 0` — every pair must favour the candidate. Under a null the
    sign of each pair is a coin flip, so P(all 5 positive) = 1/32 = 3.1%. It
    also has the property the median alone does not: one adverse block sinks
    the candidate, which is the right posture when the alternative is spending
    a claim window confirming something the instrument cannot resolve.
  * `median(relative) > drift_bound` — the sign test alone is still fooled by
    *systematic* drift, which is exactly what the A/A found. `DRIFT_BOUND_BY_METRIC`
    is derived (at import, from the numbers, not typed) as the largest
    single-step relative change between ADJACENT A/A runs: 3.08% for pp512,
    2.13% for tg128. A candidate whose median relative gain does not exceed the
    step the instrument itself takes between two runs of identical code has not
    been shown to be faster than nothing.
  * N is PRE-COMMITTED on the spec and `decide()` refuses any other count, so
    there is no optional stopping, no multiplicity, and no e-process to make
    inert. This closes, by construction, the one open manufacture-a-crossing
    hole in `execution/README.md` §6.5 (re-run a declared round until it
    crosses): there is no round to re-run.

The rule is deliberately conservative. It cannot rank a +2% win on decode. That
is the honest reading of a host whose own A/A spread is 4.3%, and the remedy is
a quieter host or a bigger effect — not a looser rule.

EVERY EXIT PATH RELEASES
------------------------
`ResourceLedger` releases in reverse acquisition order, exactly once, from a
`finally`, and keeps going past a failing release rather than abandoning the
rest. A driver that dies holding a claim is how the next session finds the host
locked by a corpse. The production-tree immutability proof runs on EVERY exit
path too, successful or not: it is the most important thing this file checks and
it must not be skipped by the failure that made it interesting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from math import ceil
# The STDLIB median, deliberately. `evaluator.statistics` is NOT imported: its
# e-process solved a harder problem than the measured 1.6–1.9% CV poses, and it
# made the gate unpassable at B_min (threshold 10, ceiling 5.5687 at every
# effect size). See MODULES_DELIBERATELY_NOT_USED.
from statistics import median
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from . import journal as journal_module
from . import schemas, storage
from .controller import do_not_repeat, hypotheses
from .evaluator import api, correctness, devices, recipes
from .execution import chain, cpu_region_claim, microbench, t0_provider, worktree
from .resource import claim_witness, device_claim, preflight

__all__ = [
    "MODULES_THE_DRIVER_USES",
    "MODULES_DELIBERATELY_NOT_USED",
    "AA_PP512_RUNS",
    "AA_TG128_RUNS",
    "adjacent_relative_steps",
    "drift_bound_from",
    "DRIFT_BOUND_BY_METRIC",
    "BOOST_THRESHOLD_KHZ",
    "BOOST_MIN_CORES",
    "BOOST_MIN_CORES_OF",
    "required_boosting_cores",
    "LOADED_ENOUGH_TO_JUDGE_BOOST",
    "check_boost_under_load",
    "MOE_DISPATCH_OP",
    "require_op_suite_covers_moe_dispatch",
    "Pair",
    "AcceptDecision",
    "T0Outcome",
    "AcceptRuleMisuse",
    "decide",
    "EXPLORATORY_NOTE",
    "HypothesisBindingError",
    "campaign_purpose",
    "authorize_for",
    "CampaignSpec",
    "Step",
    "ResourceLedger",
    "ReleaseRecord",
    "CampaignOps",
    "DryRunOps",
    "HostOps",
    "CampaignResult",
    "run_campaign",
    "build_parser",
    "main",
]


# =============================================================================
# The boundary. Enforced by test_campaign.TestTheBoundaryIsStructural.
# =============================================================================

#: Every module this driver may import, and the single reason each is essential.
#: Each reason is a real incident or a fact measured on this host, not a taste.
MODULES_THE_DRIVER_USES: Mapping[str, str] = {
    "schemas": "one record shape; PASS/FAIL/COULD_NOT_CHECK",
    "storage": "the 2026-07-04 async-prefetch win was written to /mnt/raid0/llm/tmp/ "
               "and that directory no longer exists; assert_not_scratch refuses it",
    "journal": "AutoPilot lost 232 trials / ~16 days to an unjournaled loop",
    "controller.hypotheses": "the falsifier-before-compute gate. `claim_for_hypothesis` "
                             "calls itself THE ONLY route from a hypothesis to a "
                             "resource claim and had ZERO non-test callers until "
                             "2026-08-04: this driver acquired the region claim "
                             "directly, so the rule was enforced on nothing. The driver "
                             "is what SPENDS the claim, so the gate belongs here",
    "controller.do_not_repeat": "not a taste and not 'just in case': "
                                "`authorize_claim(ledger=…)` has NO default and "
                                "`claim_for_hypothesis` raises `LedgerNotConsulted` on a "
                                "token carrying no verdict, so a SPENDABLE token cannot "
                                "be minted without a real ledger. "
                                "`compile_for_tracker(tracker)` is the only honest one — "
                                "a driver-local stub returning 'nothing matched' would "
                                "be the fail-open this package refuses everywhere else",
    "evaluator.api": "a Verdict/identity is constructible only through this module",
    "evaluator.correctness": "T0. Throughput is reward-hackable; T0 is what makes "
                             "deleting the computation lose",
    "evaluator.devices": "a GPU cell must not be satisfied by 'Device 0: CPU'",
    "evaluator.recipes": "argv from a hashed constructor. Production once drifted off "
                         "NUMA interleave and the front door ended up at 46% of canonical",
    "execution.chain": "the seams; a hand-written evidence record is what T0 exists to refuse",
    "execution.cpu_region_claim": "TODO-free: two A/A runs were destroyed by a legitimate "
                                  "co-tenant because we held no claim",
    "execution.microbench": "paired ALTERNATING blocks — the measured monotone drift makes "
                            "any sequential design charge the second arm ~4%",
    "execution.t0_provider": "the executed T0 evidence provider",
    "execution.worktree": "no candidate exists without it; production stays byte-identical",
    "resource.claim_witness": "a claim is witnessed, not asserted",
    "resource.device_claim": "the GPU half of the same invariant",
    "resource.preflight": "INC-20260731: a name-pattern kill took out another agent's "
                          "server twice, and earlyoom, whose argv names what it guards",
}

#: The rest of the package, and why a campaign does not need it. NOT a deletion
#: list — deletion is the operator's call and stays a separate one-line decision.
#: This is the boundary that makes that decision easy to make.
#:
#: 2026-08-04: the operator MADE that decision. `release`, `adapters` and
#: `surface` — and the AK4 strategy plane under `controller` — were removed
#: (~79,600 lines, recoverable from the tag `autokernel-preserve-20260804`).
#: Their rows are gone from this table rather than kept as epitaphs, because
#: `test_campaign.TestTheBoundaryIsStructural.test_every_declared_module_is_real`
#: is right: a boundary that names modules which do not exist is a boundary
#: nobody can check. The argument each row carried is preserved where it is
#: still checkable — `FOOTPRINT.md` for the reachability record and
#: `test_campaign_footprint.DELETED_BY_OPERATOR` for the prefixes that may
#: never come back onto this path.
#:
#: 2026-08-04, second correction: `controller` LEFT this table. It said the
#: campaign path "reaches none of it… a hypothesis store is read by the agent
#: proposing, not by the driver measuring". The first half of that is right about
#: STRATEGY and wrong about CLAIMS: the driver is what SPENDS the claim, so the
#: falsifier-before-compute gate belongs where the spending happens, and while it
#: sat on the other side of this boundary `claim_for_hypothesis` — "the ONLY route
#: from a hypothesis to a resource claim" — had zero non-test callers. Two modules
#: under `controller` are now named in `MODULES_THE_DRIVER_USES`, one by one; the
#: prefix is NOT open, and `test_campaign_footprint.CONTROLLER_ALLOWED` is the
#: matching allow-list on the closure side (a prefix allowance there would
#: silently re-admit anything added under `controller/` later).
MODULES_DELIBERATELY_NOT_USED: Mapping[str, str] = {
    "evaluator.integrity": "one of two coexisting derivations of the same §8.5.1 gates; "
                           "reached transitively through chain.py, which is the single "
                           "derivation this driver consumes",
    "evaluator.surface": "the other one, and `derived superset-of traced` is unsatisfiable "
                         "against a full-suite trace",
    "evaluator.statistics": "IMPORT-FOR-CONSTANTS was permitted and is not needed. Its "
                            "e-process made the gate UNPASSABLE (threshold 10, ceiling "
                            "5.5687 for every effect size) and the fix's authorization was "
                            "self-certifying. The measured 1.6-1.9% CV does not justify "
                            "it; a median over paired deltas covers it. Optional stopping "
                            "stays off, structurally, because N is pre-committed",
}


# =============================================================================
# The measured A/A, and the drift bound derived from it
# =============================================================================

#: `data/autokernel_aa_20260804/README.md`, runs A B C D, in order.
AA_PP512_RUNS: tuple = (899.95, 894.70, 867.16, 886.16)
AA_TG128_RUNS: tuple = (52.76, 52.31, 51.62, 50.52)

#: Where the numbers above live. A campaign that cites them cites a path in git.
AA_EVIDENCE_REF = "data/autokernel_aa_20260804/README.md"


def adjacent_relative_steps(series: Sequence[float]) -> tuple:
    """The relative change between each pair of CONSECUTIVE runs.

    Adjacent, not against the first run, because the quantity a paired design
    is exposed to is ONE step: within a block the anchor and the candidate are
    neighbours in time, so the drift the pairing fails to difference out is the
    drift between two adjacent measurements, not the drift across the run.
    """
    values = [float(v) for v in series]
    if len(values) < 2:
        raise ValueError("a drift bound needs at least two consecutive observations")
    return tuple((values[i + 1] - values[i]) / values[i] for i in range(len(values) - 1))


def drift_bound_from(series: Sequence[float]) -> float:
    """The largest single-step relative move an A/A of IDENTICAL code produced.

    This is the number a candidate's median relative gain has to beat. It is
    computed here from the recorded series rather than typed, so re-running the
    A/A and updating `AA_*_RUNS` updates the rule — and so that no one can
    quietly loosen the bound without changing a measurement.
    """
    return max(abs(step) for step in adjacent_relative_steps(series))


#: Keyed by `recipes.RecipeSpec.metric`, because the bound is a property of the
#: cell that was measured, not of the campaign. pp512 -> 0.0308, tg128 -> 0.0213.
DRIFT_BOUND_BY_METRIC: Mapping[str, float] = {
    "prefill_tokens_per_s": drift_bound_from(AA_PP512_RUNS),
    "decode_tokens_per_s": drift_bound_from(AA_TG128_RUNS),
}

#: For a cell with no A/A of its own. The LARGEST measured bound, not an
#: average: an unmeasured cell gets the most conservative bound we have evidence
#: for, never the most convenient one.
DEFAULT_DRIFT_BOUND: float = max(DRIFT_BOUND_BY_METRIC.values())


def drift_bound_for_metric(metric: str) -> float:
    return DRIFT_BOUND_BY_METRIC.get(metric, DEFAULT_DRIFT_BOUND)


# =============================================================================
# The idle-frequency trap
# =============================================================================

#: `scripts/lib/canonical_recipe.py`, restated here only as the values this
#: check compares against; `preflight_canonical.py` reads the same two.
BOOST_THRESHOLD_KHZ = 2_500_000
BOOST_MIN_CORES = 80
#: …**of 96**, and the denominator is carried rather than left in a comment.
#: A count without its footprint is unsatisfiable on any cell that does not pin
#: 96 cores: the GPU cell pins `184-191`, EIGHT cores, so "80 boosting" cannot be
#: reached by a healthy MI210 host and the gate FAILs the compliant path.
BOOST_MIN_CORES_OF = 96


def required_boosting_cores(cpu_count: int) -> int:
    """The boost floor for a footprint of `cpu_count` cores. A RATIO, not a count.

    The ratified constant is `FREQ_BOOST_MIN_CORES = 80  # of 96`, and 96 is the
    canonical CPU footprint (`taskset -c 0-95`). Applied verbatim to the GPU
    cell's eight host cores it is unreachable — 8 boosting cores out of 8 is a
    perfectly healthy machine and `80` is not a number it can produce. A guard
    that FAILs its own compliant path gets switched off, which is how the
    throttle check stops existing.

    On the canonical footprint this returns exactly `BOOST_MIN_CORES`, so the
    ratified gate is unchanged where it was ratified.
    """
    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count < 1:
        raise ValueError("cpu_count must be a positive int")
    if cpu_count >= BOOST_MIN_CORES_OF:
        return BOOST_MIN_CORES
    return max(1, ceil(cpu_count * BOOST_MIN_CORES / BOOST_MIN_CORES_OF))

#: Below this 1-minute-load-per-core, the boost count says nothing.
#:
#: DERIVED, not chosen: it is `microbench.HostStatePolicy.max_load_per_core`, the
#: ceiling above which this package already refuses to START a run. The two
#: readings then partition cleanly — below it the host is quiet enough to bench
#: on and its cores are parked, at or above it the host is doing real work — and
#: a campaign's own canonical bench saturates the claimed cores (`-t 96` on 96
#: cores is ~1.0/core), so the check is evaluated exactly where it is valid.
#: Measured on this host 2026-08-04: 16 cores above 2.5 GHz at load 3.3
#: (0.034/core), 117 under our own bench. The gate is written for the second
#: reading and aborts on the first, which is a healthy machine.
LOADED_ENOUGH_TO_JUDGE_BOOST: float = microbench.HostStatePolicy().max_load_per_core


def check_boost_under_load(*, boosting_cores: int, load1: Optional[float],
                           cpu_count: int) -> schemas.Check:
    """The canonical recipe's boost-count gate, evaluated only where it is valid.

    THE TRAP, measured 2026-08-04 and the reason this function is not a
    two-line inline: `FREQ_BOOST_MIN_CORES=80` cores above
    `FREQ_BOOST_THRESHOLD_KHZ=2_500_000` **fails on a healthy IDLE machine.**
    An idle EPYC parks its cores; this host reported 16 cores above 2.5 GHz at
    idle and 117 under load. `preflight_canonical.gates_tripwire_and_freq`
    avoids the trap by generating the load itself — it spawns a `llama-bench`
    to make the reading meaningful — and a campaign preflight that copies the
    THRESHOLD without copying the LOAD aborts on a perfectly good host, at
    which point the first thing anyone does is switch the check off.

    So the three outcomes are three outcomes:

      * not under load -> COULD_NOT_CHECK, with the reason. Never PASS: an
        unevaluated throttle check is not a passing one, and this host has sat
        at -60% for days undetected (`feedback_host_throttle_check`).
      * under load, >= `required_boosting_cores(cpu_count)` boosting -> PASS.
      * under load, fewer -> FAIL. The guard still bites, which is the point of
        fixing it rather than deleting it.

    The floor is a RATIO of the claimed footprint, not the bare ratified count:
    see `required_boosting_cores`. `80 of 96` on the GPU cell's eight cores is
    the same defect in the other direction — a gate that FAILs a healthy host.
    """
    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count < 1:
        raise ValueError("cpu_count must be a positive int")
    if isinstance(boosting_cores, bool) or not isinstance(boosting_cores, int) \
            or boosting_cores < 0:
        raise ValueError("boosting_cores must be a non-negative int")
    if load1 is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "/proc/loadavg could not be read, so it is unknown whether the boost count "
            f"({boosting_cores} core(s) at or above {BOOST_THRESHOLD_KHZ} kHz) was taken "
            "under load; an idle host parks its cores and reads as throttled",))
    per_core = float(load1) / cpu_count
    if per_core < LOADED_ENOUGH_TO_JUDGE_BOOST:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"1-minute load {float(load1):.2f} over {cpu_count} cores is {per_core:.3f}/core, "
            f"below {LOADED_ENOUGH_TO_JUDGE_BOOST:.2f} — the host is IDLE, and the boost "
            f"count ({boosting_cores} core(s) at or above {BOOST_THRESHOLD_KHZ} kHz) is not "
            "evaluable there. Measured on this host 2026-08-04: 16 cores boosting at idle, "
            "117 under load. Re-read this during the first anchor block, not before it.",))
    required = required_boosting_cores(cpu_count)
    if boosting_cores >= required:
        return schemas.Check(schemas.PASS, (
            f"{boosting_cores} core(s) at or above {BOOST_THRESHOLD_KHZ} kHz under load "
            f"{float(load1):.2f} ({per_core:.2f}/core), at or above the {required} the "
            f"canonical recipe requires of a {cpu_count}-core footprint "
            f"({BOOST_MIN_CORES} of {BOOST_MIN_CORES_OF})",))
    return schemas.Check(schemas.FAIL, (
        f"only {boosting_cores} core(s) at or above {BOOST_THRESHOLD_KHZ} kHz under load "
        f"{float(load1):.2f} ({per_core:.2f}/core); a {cpu_count}-core footprint requires "
        f"{required} ({BOOST_MIN_CORES} of {BOOST_MIN_CORES_OF}). "
        "This is the multi-day-uptime hysteresis (feedback_host_throttle_check); a number "
        "taken here is poisoned. The fix is a host reboot, which is the operator's.",))


# =============================================================================
# The MoE-dispatch hole, made structural
# =============================================================================

#: The op the predecessor harness did not test.
MOE_DISPATCH_OP = "MUL_MAT_ID"


def require_op_suite_covers_moe_dispatch(ops: Sequence[str]) -> tuple:
    """Refuse a T0 op suite that omits `MUL_MAT_ID`. Returns the normalized ops.

    `kernel_eval.sh` tested `MUL_MAT` only. `MUL_MAT_ID` is MoE expert dispatch
    and it runs on EVERY TOKEN of the production worker
    (`Qwen3-Coder-30B-A3B`, `gemma4-26B-A4B`), so a kernel that broke it passed
    the predecessor's correctness screen cleanly and would have been ranked on
    speed. A campaign that measures a MoE model and does not test MoE dispatch
    is measuring an untested code path, and that is refused here rather than
    noted in a runbook.
    """
    normalized = tuple(str(op).strip().upper() for op in ops if str(op).strip())
    if not normalized:
        raise ValueError("the T0 op suite must name at least one ggml op")
    if MOE_DISPATCH_OP not in normalized:
        raise ValueError(
            f"the T0 op suite {list(normalized)} does not cover {MOE_DISPATCH_OP}. The "
            "predecessor harness (kernel_eval.sh) tested MUL_MAT only, so a kernel that "
            f"broke {MOE_DISPATCH_OP} — MoE expert dispatch, executed on EVERY token in "
            "production — passed it cleanly. Speed is reward-hackable and correctness is "
            "what makes deleting the computation lose; a suite that skips the op the "
            "production model spends its time in is not that.")
    return normalized


# =============================================================================
# The accept rule
# =============================================================================

class AcceptRuleMisuse(Exception):
    """The accept rule was asked a question it must not answer."""


@dataclass(frozen=True)
class Pair:
    """One paired block's two arms, reduced to one number each.

    `anchor` and `candidate` are the block's per-arm reduction (the median of
    that arm's repetitions), and `order` records which arm ran FIRST inside the
    block. The order is carried because it is what makes the pairing honest: a
    run whose blocks are all `anchor_first` has not differenced out the within-
    block drift, it has given the candidate the later — slower — slot every
    time.
    """

    block_index: int
    anchor: float
    candidate: float
    order: str

    def __post_init__(self) -> None:
        for name in ("anchor", "candidate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Pair.{name} must be a number")
            if value <= 0:
                raise ValueError(
                    f"Pair.{name}={value!r}: a throughput of zero or less is not a "
                    "measurement, and dividing by it would manufacture a relative delta")
        if self.order not in ("anchor_first", "candidate_first"):
            raise ValueError(
                f"Pair.order {self.order!r} must be 'anchor_first' or 'candidate_first' — "
                "the two orders microbench.BlockPlan derives")

    @property
    def delta(self) -> float:
        return float(self.candidate) - float(self.anchor)

    @property
    def relative(self) -> float:
        return self.delta / float(self.anchor)

    def to_dict(self) -> dict:
        return {"block_index": self.block_index, "anchor": self.anchor,
                "candidate": self.candidate, "order": self.order,
                "delta": self.delta, "relative": self.relative}


@dataclass(frozen=True)
class T0Outcome:
    """T0's answer, reduced to the one bit the loop branches on, plus the detail."""

    all_pass: bool
    gates: tuple = ()          # ((gate_id, outcome, (reason, ...)), ...)
    report_ref: Optional[str] = None

    @property
    def failures(self) -> tuple:
        return tuple(g for g in self.gates if g[1] == schemas.FAIL)

    def to_dict(self) -> dict:
        return {"all_pass": self.all_pass, "report_ref": self.report_ref,
                "gates": [[gid, outcome, list(reasons)] for gid, outcome, reasons
                          in self.gates]}


def anchor_drift(pairs: Sequence[Pair]) -> float:
    """How much the ANCHOR arm moved across this run. The in-run neutral control.

    THE GAP THIS CLOSES: interleaved paired blocks handle drift WITHIN a
    comparison, but nothing in a candidate-vs-anchor design tells you the
    instrument moved at all — *"a campaign that never runs a control cannot
    distinguish 'this kernel is 4% faster' from 'this kernel ran first'"*. The
    A/A found decode declining monotonically across four consecutive runs of
    identical code, so this is not hypothetical.

    It is computed from the anchor readings the run ALREADY produced, in block
    order, and costs no extra claim time: those five anchor blocks ARE an A/A of
    identical code taken during this run, on this host, under this claim. The
    alternative — a dedicated anchor-vs-anchor control arm through
    `evaluator/controls.py` and `execution/control_runner.py` (3,951 lines) —
    measures the same quantity and doubles the claim window to do it. If a
    campaign ever needs the drift attributed to a MECHANISM rather than merely
    detected, that is the upgrade; detection is what the accept rule needs and
    detection is free.

    Returns the largest single-step relative move between consecutive blocks'
    anchors — the same statistic, on the same footing, as `drift_bound_from`.
    """
    series = [p.anchor for p in sorted(pairs, key=lambda p: p.block_index)]
    if len(series) < 2:
        raise ValueError("an in-run drift control needs at least two blocks")
    return max(abs(step) for step in adjacent_relative_steps(series))


@dataclass(frozen=True)
class AcceptDecision:
    """KEEP or REVERT, with every number the rule read.

    There is no `score`, no rank and no e-value. The rule is a conjunction of
    two comparisons and the record says which one failed.
    """

    keep: bool
    reason: str
    blocks: int
    min_delta: Optional[float] = None
    median_relative: Optional[float] = None
    drift_bound: Optional[float] = None
    anchor_drift: Optional[float] = None
    deltas: tuple = ()
    relatives: tuple = ()
    anchors: tuple = ()
    orders: tuple = ()

    def to_dict(self) -> dict:
        return {"keep": self.keep, "reason": self.reason, "blocks": self.blocks,
                "min_delta": self.min_delta, "median_relative": self.median_relative,
                "drift_bound": self.drift_bound, "anchor_drift": self.anchor_drift,
                "deltas": list(self.deltas), "relatives": list(self.relatives),
                "anchors": list(self.anchors), "orders": list(self.orders)}


def decide(pairs: Sequence[Pair], *, t0: T0Outcome, blocks_precommitted: int,
           drift_bound: float) -> AcceptDecision:
    """The accept rule. Thirty lines, justified from the A/A in the module docstring.

    Raises `AcceptRuleMisuse` — never returns a decision — when:

      * T0 did not all-PASS. A wrong kernel gets no speed rank AT ALL, so there
        is no branch here that reads a delta after a failed T0. `run_campaign`
        never plans a block in that case either; this is the second lock on the
        same door, because the ordering is the whole property and one lock is a
        convention.
      * the number of pairs is not the PRE-COMMITTED N. That is what makes
        optional stopping impossible rather than discouraged: a run cannot be
        extended until it crosses, because a longer run is not admissible input.
    """
    if not isinstance(t0, T0Outcome):
        raise TypeError("decide(t0=...) takes a T0Outcome")
    if not t0.all_pass:
        raise AcceptRuleMisuse(
            "decide() was called after a T0 that did not all-PASS "
            f"(failures: {[g[0] for g in t0.failures]}). T0 is lexicographically prior: a "
            "candidate that is wrong gets no speed number at all, because throughput is "
            "reward-hackable and deleting the computation is the fastest kernel there is.")
    if isinstance(blocks_precommitted, bool) or not isinstance(blocks_precommitted, int) \
            or blocks_precommitted < 2:
        raise ValueError("blocks_precommitted must be an int >= 2; one pair is n=1, and the "
                         "A/A shows n=1 here is a coin flip with a decimal point")
    items = list(pairs)
    for item in items:
        if not isinstance(item, Pair):
            raise TypeError("decide(pairs=...) takes campaign.Pair instances")
    if len(items) != blocks_precommitted:
        raise AcceptRuleMisuse(
            f"{len(items)} pair(s) were submitted against a pre-committed N of "
            f"{blocks_precommitted}. N is fixed before the run so that there is no optional "
            "stopping and no multiplicity to correct for; a run of a different length is a "
            "different design, and accepting it here is exactly the 'run the round again "
            "until it crosses' hole (execution/README.md 6.5).")
    if isinstance(drift_bound, bool) or not isinstance(drift_bound, (int, float)) \
            or drift_bound <= 0:
        raise ValueError("drift_bound must be a positive fraction, e.g. 0.0213 for tg128")

    ordered = sorted(items, key=lambda p: p.block_index)
    orders = tuple(p.order for p in ordered)
    deltas = tuple(p.delta for p in ordered)
    relatives = tuple(p.relative for p in ordered)
    anchors = tuple(p.anchor for p in ordered)
    min_delta = min(deltas)
    median_relative = float(median(relatives))
    moved = anchor_drift(ordered)
    common = {"blocks": len(items), "min_delta": min_delta,
              "median_relative": median_relative, "drift_bound": float(drift_bound),
              "anchor_drift": moved, "deltas": deltas, "relatives": relatives,
              "anchors": anchors, "orders": orders}

    # ORDER CONTROL, before anything else is read. `Pair.order` was recorded and
    # never inspected, which made it documentation. `statistics.OrderSchedule`
    # draws each block's order from a per-block hash of the campaign seed — a
    # COIN FLIP PER BLOCK, not an alternation — so five blocks land all one way
    # once in sixteen runs, and such a run is a sequential A/B wearing a paired
    # design's record.
    #
    # It is refused in BOTH directions, and refusing only the direction that
    # flatters the candidate would be the mistake. The quantity that would say
    # which direction a within-block position effect runs in is the WITHIN-block
    # slot difference, and in an all-one-way run that is perfectly confounded
    # with the candidate effect — it is the one number this design cannot
    # measure. The anchor-arm control below does not see it either: a host that
    # boosts at the start of each block and sags inside it (which is what an
    # EPYC does) produces a FLAT anchor series across blocks and a systematic
    # several-percent gap inside every one of them.
    #
    # So all-one-way is inadmissible, full stop, and the cost is a false
    # NEGATIVE once every sixteen five-block runs. That is the direction this
    # rule is deliberately wrong in.
    if len(set(orders)) < 2:
        return AcceptDecision(
            keep=False,
            reason=(f"REVERT (inadmissible): all {len(items)} blocks ran {orders[0]!r}, so "
                    "no block ever gave the other arm the earlier slot. The order draw is "
                    "a per-block coin flip (`statistics.OrderSchedule` hashes the campaign "
                    "seed per block), not an alternation, and a run that lands all one way "
                    "is a sequential A/B wearing a paired design's record: any within-"
                    "block position effect is added to the candidate effect and cannot be "
                    "separated from it, in EITHER direction. Re-run under a fresh campaign "
                    "seed — `OrderSchedule.retry()` reverses every element, so it turns a "
                    "5-0 draw into a 5-0 draw the other way and does NOT fix this."),
            **common)

    # THE NEUTRAL CONTROL, and it is FIRST because it is about whether this run
    # is admissible at all, not about the candidate. If the anchor arm — which
    # is identical code from block to block — moved further across this run than
    # an A/A of identical code did, the instrument moved during the measurement
    # and the deltas are not attributable to the kernel.
    if moved > drift_bound:
        return AcceptDecision(
            keep=False,
            reason=(f"REVERT (inadmissible): the ANCHOR arm moved {moved:.4%} across this "
                    f"run, more than the {drift_bound:.4%} an A/A of identical code "
                    f"produced ({AA_EVIDENCE_REF}). The instrument moved during the "
                    "measurement, so no delta in it is attributable to the candidate — "
                    "this is the control that separates 'this kernel is faster' from "
                    "'this kernel ran first'."),
            **common)

    if min_delta <= 0:
        adverse = [p.block_index for p in items if p.delta <= 0]
        return AcceptDecision(
            keep=False,
            reason=(f"REVERT: {len(adverse)} of {len(items)} paired block(s) did not favour "
                    f"the candidate (blocks {adverse}, worst delta {min_delta:+.4f}). Every "
                    "pre-committed pair must favour the candidate; under a null the sign of "
                    f"each pair is a coin flip, so all-{len(items)} is a 1-in-"
                    f"{2 ** len(items)} event."),
            **common)
    if median_relative <= drift_bound:
        return AcceptDecision(
            keep=False,
            reason=(f"REVERT: median relative gain {median_relative:+.4%} does not exceed "
                    f"the drift bound {drift_bound:.4%} — the largest single-step move this "
                    f"host produced between two consecutive runs of IDENTICAL code "
                    f"({AA_EVIDENCE_REF}). Every pair favoured the candidate, which a "
                    "systematic drift also produces; the sign test alone cannot tell them "
                    "apart and this conjunct is what does."),
            **common)
    return AcceptDecision(
        keep=True,
        reason=(f"KEEP: all {len(items)} pre-committed paired blocks favoured the candidate "
                f"(worst delta {min_delta:+.4f}) and the median relative gain "
                f"{median_relative:+.4%} exceeds the measured drift bound "
                f"{drift_bound:.4%} ({AA_EVIDENCE_REF})."),
        **common)


# =============================================================================
# The falsifier-before-compute gate — `--hypothesis`
# =============================================================================
#
# THE DEFECT THIS CLOSES. `controller/hypotheses.py::claim_for_hypothesis`
# documents itself as *"The ONLY route from a hypothesis to a resource claim"*
# and enforces the operator-approved rule that a falsifier is OPTIONAL when a
# hypothesis is written and MANDATORY before a claim is spent on it. Until
# 2026-08-04 it had ZERO non-test callers: `HostOps.acquire_claim` called
# `cpu_region_claim.acquire_cpu_region_claim` directly, so the gate enforced
# nothing. That is the fifth instance of one shape in this package —
# `_WORKTREE_MUTATING_SUBCOMMANDS` referenced only by its own definition,
# `OrderSchedule.retry()` with no caller, `check_do_not_repeat` with no ledger,
# control seed rotation declared/hashed/callerless — and it is why the wiring
# lives in the DRIVER rather than beside the proposer: the driver is what
# SPENDS the claim.
#
# TWO MODES, AND THE RECORD DISTINGUISHES THEM
#
#   * no `--hypothesis` -> EXPLORATORY. The claim is acquired exactly as before.
#     The record SAYS so (`EXPLORATORY_NOTE`), because an unexplained absence and
#     a declared exploratory run must not read the same afterwards — that is the
#     same confusion `ClaimAuthorization.do_not_repeat_outcome` refuses one field
#     over ("nobody asked" has no default there either).
#   * `--hypothesis akh-…` -> the claim is acquired THROUGH
#     `claim_for_hypothesis`, whose type gate cannot be satisfied by a question
#     with an absent or placeholder falsifier.
#
# AND THE REFUSAL IS AT THE DOOR. `authorize_for` runs in `main()`, before the
# ops object is used, before the region claim, before the worktree, before the
# build — the same reason `unimplemented_seams` is checked there. A gate that
# fires an hour into a held window has already cost the thing it was protecting.

#: What the record says when no question was declared. Written into the campaign
#: record so "exploratory by decision" is a positive statement rather than a
#: field that happens to be missing.
EXPLORATORY_NOTE = (
    "EXPLORATORY: no --hypothesis was declared, so this campaign is bound to no "
    "question and resolves none. The claim was acquired directly rather than through "
    "hypotheses.claim_for_hypothesis, which is legal only because nothing here claims "
    "to be testing a stated prediction."
)

#: Who the ledger records as having authorized the spend. The DRIVER, by name:
#: the authorization is machine-made at campaign start, and a record that named a
#: person would be attributing a decision nobody made at that moment.
AUTHORIZED_BY = "campaign.py"


class HypothesisBindingError(Exception):
    """`--hypothesis` was given but the campaign cannot bind it to a record.

    Distinct from every refusal `controller/hypotheses.py` raises: those are the
    GATE saying no (no falsifier, a placeholder, an unknown question, a receipted
    repeat). This one is the DRIVER saying it was not given enough to ask the
    gate at all — and the two must not be confused, because the second is fixed
    by adding a flag and the first is not fixed by anything the caller can type.
    """


def campaign_purpose(spec: "CampaignSpec") -> str:
    """What a claim's own receipt says this hold is for, before any question.

    `cpu_region_claim` refuses an unattributable claim, so this string is not
    decoration. On the hypothesis path it is the PREFIX that
    `ClaimAuthorization.claim_purpose` appends the falsifier to, which is how the
    resource record and the question record end up saying the same thing without
    anyone keeping them in step.
    """
    return f"AutoKernel campaign {spec.campaign_id} / {spec.candidate_id}"


def authorize_for(spec: "CampaignSpec", hypothesis_id: str, *,
                  store_path: Optional[str] = None, dry_run: bool = True,
                  authorized_by: str = AUTHORIZED_BY):
    """Mint the ONE token that lets this campaign's claim be spent on a question.

    Raises rather than returning a degraded value, and every refusal below is
    somebody else's:

    * an absent falsifier          -> `FalsifierRequiredBeforeCompute` (from
      `ClaimAuthorization.__post_init__`, which is the one place that refusal is
      worded);
    * a placeholder falsifier      -> `FalsifierMissing`, from the STORE loader,
      naming the file and the entry index. A different type and a different
      message from the line above, deliberately: the two states are distinct all
      the way down, and collapsing them is the defect the hypothesis work closed;
    * an unknown id                -> `UnknownHypothesis`. Never treated as
      "no hypothesis declared": a typo must not silently downgrade a bound
      campaign into an exploratory one;
    * a receipted repeat           -> `RepeatsAReceiptedNegative` (§8.4, §19.2).

    `dry_run` is carried into the PURPOSE rather than used to skip the gate. A
    gate that only runs under `--execute` is a gate wired to the one mode no test
    can exercise, which is the defect this whole function exists to remove; and a
    CLAIM_AUTHORIZED record that did not say a composition pass minted it would
    be a record of a spend that never happened.
    """
    if not str(hypothesis_id or "").strip():
        raise HypothesisBindingError("--hypothesis needs a hypothesis id (akh-…)")
    if not spec.journal_root:
        raise HypothesisBindingError(
            f"--hypothesis {hypothesis_id} names a question whose authorization is a "
            "DURABLE record, and --journal-root is where this campaign's records live. "
            "Authorizing into a root nobody declared would spend a claim against a "
            "question whose ledger the next session cannot find. Pass --journal-root.")
    root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
    book = journal_module.Journal(root, campaign_id=spec.campaign_id)
    book.initialize()
    tracker = hypotheses.HypothesisTracker(
        journal_=book, root=root, campaign_id=spec.campaign_id)
    # The operator's drop-in file, if this campaign has one. `intake` is IDEMPOTENT
    # and is what turns a line the operator typed into a tracked question; passing
    # None is how a campaign says it has no operator channel, and is NOT the same
    # as pointing at a path that may not be mounted (`load()` raises on absence).
    store = (hypotheses.OperatorHypothesisStore(store_path)
             if store_path else None)
    tracker.intake(store)
    # Recompiled here, per campaign, rather than held: `compile_for_tracker` reads
    # BOTH halves off the same tracker so the journal and the hypothesis ledger
    # can never be from two different campaigns.
    ledger = do_not_repeat.compile_for_tracker(tracker)
    purpose = campaign_purpose(spec) + (
        " [DRY RUN: composed only, no claim was spent]" if dry_run else "")
    return tracker.authorize_claim(hypothesis_id, purpose=purpose,
                                   authorized_by=authorized_by, ledger=ledger)


# =============================================================================
# The pre-committed campaign spec
# =============================================================================

DEFAULT_BLOCKS = 5
BACKEND_CPU = "llama_cpu"
BACKEND_GPU = "llama_gpu"
BACKENDS = (BACKEND_CPU, BACKEND_GPU)

#: The default cell, per backend: the ratified canonical llama-bench decode slice.
DEFAULT_RECIPE_BY_BACKEND = {
    BACKEND_CPU: "t1b.llama_cpu.llama_bench_decode.v1",
    BACKEND_GPU: "t1b.llama_gpu.llama_bench_decode.v1",
}

#: Production, frozen. `worktree.resolve_anchor(expected_commit=...)` turns
#: "I believe production is at v8" into a checked precondition, and
#: `create_campaign_worktree` re-resolves the tip and raises `StaleAnchor` if it
#: moved (CLAUDE.md step 1; INC-20260706-iqk-missing-subsystem).
PRODUCTION_REPO = "/mnt/raid0/llm/llama.cpp"
PRODUCTION_BRANCH = "production-consolidated-v8"
PRODUCTION_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


@dataclass(frozen=True)
class CampaignSpec:
    """Everything that must be fixed BEFORE the claim is acquired.

    Frozen, and every field that governs the accept rule is here rather than on
    the runner, because a parameter the runner can choose after seeing a number
    is not a pre-commitment. `blocks` in particular: `decide()` refuses any
    other count.
    """

    campaign_id: str
    candidate_id: str
    candidate_ref: str
    backend: str = BACKEND_CPU
    blocks: int = DEFAULT_BLOCKS
    recipe_id: Optional[str] = None
    model: Optional[str] = None
    reps: int = 5
    n_gen: int = 128
    n_prompt: int = 512
    t0_ops: tuple = ("MUL_MAT", MOE_DISPATCH_OP)
    #: The recipe's `-dev` ids, e.g. `("ROCm0",)`. An IDENTIFIER, checked by the
    #: recipe registry's own `_P_DEVICE_ID` domain.
    devices: tuple = ()
    #: What the runtime CALLS those devices, e.g. `("AMD Instinct MI210",)`.
    #: A different thing from the id, and the one `evaluator/devices.py` grades:
    #: a GPU cell must not be satisfied by "Device 0: CPU".
    device_names: tuple = ()
    device_index: int = 0
    n_gpu_layers: int = 99
    journal_root: Optional[str] = None
    build_root: str = "/mnt/raid0/llm/ak-build"
    claim_journal_path: str = "/mnt/raid0/llm/ak-claims/region.jsonl"
    max_hold_s: int = 6 * 3600
    #: The `hypotheses.ClaimAuthorization` this campaign's claim will be spent
    #: through, or `None` for an EXPLORATORY campaign. It is on the SPEC and not
    #: on the ops object for the same reason `blocks` is: it must be fixed before
    #: the claim, it governs what the claim's own receipt says, and it has to
    #: reach the durable record — `to_dict()` carries it, so "what did we spend
    #: the card on" and "what would have refuted it" are one lookup.
    authorization: Optional[Any] = None
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not str(self.campaign_id).startswith("ak-"):
            raise ValueError(f"campaign_id {self.campaign_id!r} must start with 'ak-' "
                             "(api.EvaluationRequest requires it)")
        if not str(self.candidate_id).startswith("akc-"):
            raise ValueError(f"candidate_id {self.candidate_id!r} must start with 'akc-'. "
                             "Claim ids are 'akclaim-' and the two namespaces are refused "
                             "at import by cpu_region_claim; do not merge them here")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend {self.backend!r} must be one of {list(BACKENDS)}")
        if isinstance(self.blocks, bool) or not isinstance(self.blocks, int) \
                or self.blocks < 2:
            raise ValueError("blocks (the PRE-COMMITTED N) must be an int >= 2")
        if not str(self.candidate_ref).strip():
            raise ValueError("candidate_ref must name the patch or branch under test")
        if self.authorization is not None:
            if not isinstance(self.authorization, hypotheses.ClaimAuthorization):
                raise TypeError(
                    "authorization must be a hypotheses.ClaimAuthorization or None; got "
                    f"{type(self.authorization).__name__}. A claim is spent on a question "
                    "through a capability, never through a string somebody typed")
            if self.authorization.campaign_id not in (None, self.campaign_id):
                raise ValueError(
                    f"authorization was minted for campaign "
                    f"{self.authorization.campaign_id!r}, not {self.campaign_id!r}. A "
                    "token that travelled between campaigns would charge this run's "
                    "claim to another run's question")
        object.__setattr__(self, "t0_ops", require_op_suite_covers_moe_dispatch(self.t0_ops))
        if self.recipe_id is None:
            object.__setattr__(self, "recipe_id", DEFAULT_RECIPE_BY_BACKEND[self.backend])
        spec = recipes.get_recipe(self.recipe_id)
        if spec.backend != self.backend:
            raise ValueError(f"recipe {self.recipe_id!r} is a {spec.backend!r} recipe but "
                             f"the campaign declares backend {self.backend!r}")
        if self.backend == BACKEND_GPU:
            if not self.devices:
                raise ValueError(
                    "a llama_gpu campaign must name the device it claims (--device ROCm0)")
            if not self.device_names:
                raise ValueError(
                    "a llama_gpu campaign must also declare what the runtime CALLS that "
                    "device (--device-name 'AMD Instinct MI210'). The id and the name are "
                    "different things: 'ROCm0' is an argv token and grades nothing, while "
                    "the NAME is what evaluator/devices.py reads — a GPU cell must not be "
                    "satisfied by 'Device 0: CPU'")
            check = devices.check_device_names(self.device_names,
                                               expected_lane=devices.GPU)
            if check.outcome != schemas.PASS:
                # COULD_NOT_CHECK is refused as well: an unrecognised name
                # establishes NEITHER lane, and a GPU campaign that cannot show
                # it is on a GPU is not a GPU campaign.
                raise ValueError(f"declared device names {list(self.device_names)} are not "
                                 f"a GPU lane ({check.outcome}): {list(check.reasons)}")
        elif self.devices or self.device_names:
            raise ValueError("a llama_cpu campaign claims no device; drop --device")
        if self.journal_root is not None:
            storage.assert_not_scratch(self.journal_root, what="campaign journal root")
        storage.assert_not_scratch(self.build_root, what="campaign build root")
        # Construct BOTH arms' argv now, before a claim is taken and before a
        # build is started. `recipes.construct` executes nothing, so this costs
        # nothing and turns "the cell is missing a required parameter" from a
        # failure an hour into a claim window into a refusal at argument-parse
        # time (feedback_bench_max_opt_and_config_probe_first).
        render_bench_commands(self)

    # -- derived -----------------------------------------------------------

    @property
    def recipe(self) -> Any:
        return recipes.get_recipe(self.recipe_id)

    @property
    def metric(self) -> str:
        return self.recipe.metric

    @property
    def drift_bound(self) -> float:
        return drift_bound_for_metric(self.metric)

    @property
    def hypothesis_id(self) -> Optional[str]:
        """The question this campaign is bound to, or None. Read off the TOKEN.

        Never a second field. A campaign carrying an id beside an authorization
        could have the two disagree, and the one a reader would trust is the
        string rather than the capability.
        """
        return None if self.authorization is None else self.authorization.hypothesis_id

    @property
    def claim_purpose(self) -> str:
        """What the region claim's own receipt will say, in BOTH modes.

        On the hypothesis path this is the token's `claim_purpose` verbatim —
        the same string `claim_for_hypothesis` passes to the acquirer, which is
        why the composition pass can print it and the executed run cannot print
        anything else.
        """
        if self.authorization is None:
            return campaign_purpose(self) + " [exploratory: no --hypothesis declared]"
        return self.authorization.claim_purpose

    @property
    def hypothesis_record(self) -> dict:
        """What the durable record says about the question — in BOTH modes.

        An absent key would make "we chose to explore" and "somebody forgot the
        flag existed" the same record, which is the one distinction this whole
        seam is for. Same discipline as
        `ClaimAuthorization.do_not_repeat_outcome`, which has no default for the
        same reason one field over.
        """
        if self.authorization is None:
            return {"bound": False, "hypothesis_id": None, "falsifier": None,
                    "note": EXPLORATORY_NOTE}
        return {"bound": True,
                "hypothesis_id": self.authorization.hypothesis_id,
                "falsifier": self.authorization.falsifier,
                "authorization": self.authorization.to_dict()}

    @property
    def cpu_list(self) -> str:
        """The footprint the ARGV pins, DERIVED from the argv. Never retyped.

        `recipes.ClaimFootprint` is computed from the constructed command's own
        `taskset -c` list precisely so the claim cannot drift from the mask the
        command applies, and this property is the only place the campaign asks
        what to claim.

        It is not `CANONICAL_PREFIX[2]`, and the difference is not cosmetic:
        that constant is `0-95`, and the GPU cell pins `184-191` — the device's
        node-local SMT siblings. A campaign that claimed `0-95` and ran on
        `184-191` would leave every measured core unprotected while looking, in
        every journal field, exactly like a claimed run — and
        `MicrobenchRunner._attest_claim` would refuse it an hour into the
        window, after the build.
        """
        return render_bench_commands(self)["candidate"]["cpu_list"]

    @property
    def cpu_count(self) -> int:
        """How many cores that footprint has, from the same derivation.

        Carried because the ratified boost gate is `80 of 96` and 96 is a
        FOOTPRINT: on the GPU cell's eight cores the bare count is unreachable.
        """
        return int(render_bench_commands(self)["candidate"]["cpu_count"])

    @property
    def worktree_path(self) -> str:
        """The campaign worktree, as a plain path.

        `campaign_worktree_path` returns a `SandboxPath` — a TYPE that cannot
        name a frozen tree, which is why the worktree layer takes it. `str()`
        here, at the one boundary that needs a string, rather than anywhere the
        type would be lost.
        """
        return worktree.campaign_worktree_path(self.campaign_id).path

    @property
    def build_dir(self) -> str:
        return os.path.join(self.build_root, self.campaign_id, self.candidate_id)

    @property
    def bench_params(self) -> dict:
        params: dict = {"reps": self.reps}
        if self.model:
            params["model"] = self.model
        declared = self.recipe.param_map
        if "n_gen" in declared:
            params["n_gen"] = self.n_gen
        if "n_prompt" in declared:
            params["n_prompt"] = self.n_prompt
        if self.backend == BACKEND_GPU:
            params["device_id"] = self.devices[0]
            params["device_index"] = self.device_index
            params["n_gpu_layers"] = self.n_gpu_layers
        return params

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id, "candidate_id": self.candidate_id,
            "candidate_ref": self.candidate_ref, "backend": self.backend,
            "blocks_precommitted": self.blocks, "recipe_id": self.recipe_id,
            "metric": self.metric, "drift_bound": self.drift_bound,
            "drift_bound_evidence": AA_EVIDENCE_REF,
            "model": self.model, "reps": self.reps, "n_gen": self.n_gen,
            "n_prompt": self.n_prompt,
            "t0_ops": list(self.t0_ops), "devices": list(self.devices),
            "device_names": list(self.device_names),
            "cpu_list": self.cpu_list, "worktree": self.worktree_path,
            "build_dir": self.build_dir, "journal_root": self.journal_root,
            "created_at": self.created_at,
            "hypothesis": self.hypothesis_record,
            "claim_purpose": self.claim_purpose,
            "anchor": {"repo": PRODUCTION_REPO, "branch": PRODUCTION_BRANCH,
                       "expected_commit": PRODUCTION_COMMIT},
        }


# =============================================================================
# Steps — the composition, printable
# =============================================================================

@dataclass(frozen=True)
class Step:
    """One step of the loop, with what it WOULD do rendered for review."""

    index: int
    name: str
    what: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"[{self.index:>2}] {self.name}", f"     {self.what}"]
        for key in sorted(self.detail):
            value = self.detail[key]
            if isinstance(value, (list, tuple)):
                rendered = " ".join(str(v) for v in value)
            elif isinstance(value, dict):
                rendered = json.dumps(value, sort_keys=True)
            else:
                rendered = str(value)
            if len(rendered) > 4000:
                rendered = rendered[:4000] + " ...(truncated)"
            lines.append(f"       {key}: {rendered}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "what": self.what,
                "detail": json.loads(json.dumps(self.detail, default=str))}


# =============================================================================
# The resource ledger — every exit path releases
# =============================================================================

@dataclass(frozen=True)
class ReleaseRecord:
    name: str
    released: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "released": self.released, "detail": self.detail}


class ResourceLedger:
    """Acquire-and-register; release in reverse, exactly once, from a `finally`.

    Three properties, each of which had to be a property and not a convention:

      * **reverse order.** The worktree is torn down BEFORE the claim is
        released (`execution/README.md` step 8): teardown reads and writes the
        campaign worktree, and doing that outside the claim window is doing it
        on a host somebody else has already been told is free.
      * **exactly once.** `release_all()` is idempotent, so calling it from a
        `finally` that is reached twice (an exception during cleanup) cannot
        double-release a `flock`.
      * **keeps going.** A release that raises is RECORDED and the remaining
        releases still run. The alternative — abandoning the rest on the first
        failure — is precisely how a claim outlives the process that took it.
    """

    def __init__(self) -> None:
        self._held: list = []
        self._records: Optional[tuple] = None

    def hold(self, name: str, release: Callable[[], Any]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("a held resource must be named")
        if not callable(release):
            raise TypeError("release must be callable")
        if self._records is not None:
            raise RuntimeError(
                f"cannot register {name!r}: this ledger has already released. A resource "
                "acquired after teardown has no owner and no releaser.")
        self._held.append((name, release))

    @property
    def holding(self) -> tuple:
        return tuple(name for name, _ in self._held)

    @property
    def released(self) -> bool:
        return self._records is not None

    def release_all(self) -> tuple:
        if self._records is not None:
            return self._records
        records: list = []
        for name, release in reversed(self._held):
            try:
                detail = release()
            # BaseException, for the same reason `run_campaign` catches it: the
            # realistic way a campaign ends is Ctrl-C, and a SECOND Ctrl-C
            # arrives during teardown. Catching only `Exception` here would let
            # that second interrupt strand every release after the one it hit —
            # the claim among them — which is the exact failure this ledger
            # exists to prevent.
            except BaseException as exc:  # noqa: BLE001 - a failing release must not stop the rest
                records.append(ReleaseRecord(
                    name=name, released=False,
                    detail=f"{type(exc).__name__}: {exc}"))
            else:
                records.append(ReleaseRecord(name=name, released=True,
                                             detail="released" if detail is None
                                             else str(detail)[:400]))
        self._records = tuple(records)
        self._held = []
        return self._records


# =============================================================================
# The side-effecting seam
# =============================================================================

class CampaignOps(Protocol):
    """Everything in the loop that touches the host. One Protocol, two impls.

    The loop's ORDER lives in `run_campaign` and is tested against a spy; the
    host contact lives here. That split is what lets `test_campaign` prove "no
    speed number is computed after a failed T0" and "every failure path releases
    the claim" without a host, a claim or a build.
    """

    def preflight(self, spec: CampaignSpec) -> schemas.Check: ...
    def acquire_claim(self, spec: CampaignSpec) -> Any: ...
    def release_claim(self, claim: Any) -> Any: ...
    def create_worktree(self, spec: CampaignSpec) -> Any: ...
    def apply_candidate(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def build(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def run_t0(self, spec: CampaignSpec, build: Any) -> T0Outcome: ...
    def run_paired_blocks(self, spec: CampaignSpec, build: Any,
                          claim: Any) -> Optional[Sequence[Pair]]: ...
    def teardown_worktree(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def keep_or_revert(self, spec: CampaignSpec, tree: Any,
                       decision: Optional[AcceptDecision]) -> Any: ...
    def prove_production_unchanged(self, spec: CampaignSpec) -> schemas.Check: ...
    def journal(self, spec: CampaignSpec, payload: Mapping[str, Any]) -> Any: ...


#: Opaque handles a composition pass hands back where the host path hands back a
#: claim, a worktree or a build. They are deliberately NOT `None`: `run_campaign`
#: skips keep-or-revert when there is no worktree, so a `None` here would drop a
#: step out of the very composition the operator is reviewing.
_DRY_RUN_CLAIM = "<dry-run: no claim was acquired>"
_DRY_RUN_WORKTREE = "<dry-run: no worktree was created>"
_DRY_RUN_BUILD = "<dry-run: nothing was built>"


class DryRunOps:
    """Composes and records every step. Executes NOTHING. The default.

    It emits **no speed number** — `run_paired_blocks` returns `None`, not a
    plausible-looking vector. A dry run that produces a number is a real run
    with a flag on it, and a fabricated one would be indistinguishable in the
    record from a measured one, which is the fail-open shape this package
    refuses everywhere else.

    What it DOES produce is the exact argv and env `microbench` would spawn,
    built by the same hashed constructor (`recipes.dry_run`) the real path uses,
    so what is reviewed here is what would run.
    """

    executes = False

    def __init__(self, *, out: Optional[Any] = None) -> None:
        self.steps: list = []
        self._out = out if out is not None else sys.stdout

    @property
    def calls(self) -> list:
        """The step names, DERIVED from the steps. Never a second list.

        It was a parallel list appended beside `steps`, which is two spellings
        of one fact — the shape `chain.py` argues about at length — and the one
        that can disagree is the one the tests read.
        """
        return [step.name for step in self.steps]

    # -- recording ---------------------------------------------------------

    def _step(self, name: str, what: str, **detail: Any) -> Step:
        step = Step(index=len(self.steps) + 1, name=name, what=what, detail=detail)
        self.steps.append(step)
        print(step.render(), file=self._out)
        return step

    # -- the seam ----------------------------------------------------------

    def preflight(self, spec: CampaignSpec) -> schemas.Check:
        self._step(
            "preflight",
            "would verify: frozen trees clean at their ratified commits; no concurrent "
            "inference, where anything but PASS refuses (the claim-witness layer's own "
            "rule); no overlapping region claim; SMT topology "
            "(cpu_region_claim.verify_host_topology); and the boost count UNDER LOAD only.",
            frozen_trees=list(worktree.frozen_tree_paths()),
            production=f"{PRODUCTION_REPO}@{PRODUCTION_BRANCH}#{PRODUCTION_COMMIT[:12]}",
            cpu_list=spec.cpu_list,
            boost_gate=(f">= {required_boosting_cores(spec.cpu_count)} of this cell's "
                        f"{spec.cpu_count} claimed core(s) at or above "
                        f"{BOOST_THRESHOLD_KHZ} kHz ({BOOST_MIN_CORES} of "
                        f"{BOOST_MIN_CORES_OF}, scaled to the footprint), evaluated ONLY "
                        f"at load/core >= {LOADED_ENOUGH_TO_JUDGE_BOOST} (idle reads as "
                        f"throttled: 16 boosting idle vs 117 under load)"))
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "dry run: nothing was read from the host",))

    def acquire_claim(self, spec: CampaignSpec) -> Any:
        """Composes the acquisition — THROUGH the same door the host path uses.

        A composition pass that skipped `claim_for_hypothesis` would compose a
        loop different from the one `--execute` runs, at exactly the step the
        gate is on. So the door is crossed here too; only the acquirer is
        different, and it acquires nothing.
        """
        def acquire(*, purpose: str, **detail: Any) -> Any:
            self._step("acquire_claim",
                       "would acquire the CPU region claim covering the argv's own "
                       "footprint and bind it for BOTH consumers (chain.bind_claim: "
                       "t0_provider reads verify_held()/covers(), microbench calls "
                       "attest()).",
                       cpu_list=spec.cpu_list, role="autokernel",
                       campaign_id=spec.campaign_id,
                       claim_journal=spec.claim_journal_path,
                       max_hold_s=spec.max_hold_s,
                       # The claim's own receipt, verbatim. On the hypothesis path
                       # `claim_for_hypothesis` supplies it and it carries the
                       # falsifier; the driver never gets to write its own.
                       purpose=purpose, **detail)
            return _DRY_RUN_CLAIM

        if spec.authorization is None:
            return acquire(purpose=spec.claim_purpose)
        return hypotheses.claim_for_hypothesis(spec.authorization, acquire)

    def release_claim(self, claim: Any) -> Any:
        self._step("release_claim", "would release the CPU region claim.")
        return "dry run"

    def create_worktree(self, spec: CampaignSpec) -> Any:
        self._step("create_worktree",
                   "would re-resolve the CURRENT production tip and add a campaign "
                   "worktree off it (StaleAnchor if it moved).",
                   worktree=spec.worktree_path,
                   branch=f"ak/{spec.campaign_id}/{spec.candidate_id}")
        # A handle, not None: `run_campaign` skips keep-or-revert when there is
        # no worktree to keep or revert, so returning None here would silently
        # drop a step from the composition the operator is reviewing.
        return _DRY_RUN_WORKTREE

    def apply_candidate(self, spec: CampaignSpec, tree: Any) -> Any:
        self._step("apply_candidate",
                   "would apply the candidate change and commit it with an EXPLICIT "
                   "pathspec (never `git add .` in a shared clone).",
                   candidate_ref=spec.candidate_ref)
        return None

    def build(self, spec: CampaignSpec, tree: Any) -> Any:
        self._step("build",
                   "would configure and build with GGML_CCACHE=OFF forced and the load "
                   "average cap as a PRECONDITION (HostTooContended before configure).",
                   build_dir=spec.build_dir,
                   targets=["llama-cli", "llama-bench", "test-backend-ops"])
        return _DRY_RUN_BUILD

    def run_t0(self, spec: CampaignSpec, build: Any) -> T0Outcome:
        self._step("t0",
                   "would run T0 correctness FIRST. If any gate FAILs, the campaign stops "
                   "here and NO speed number is computed at all.",
                   ops=list(spec.t0_ops),
                   moe_dispatch_covered=MOE_DISPATCH_OP in spec.t0_ops)
        return T0Outcome(all_pass=False, gates=(
            ("dry_run", schemas.COULD_NOT_CHECK,
             ("dry run: T0 was composed but not executed",)),))

    def run_paired_blocks(self, spec: CampaignSpec, build: Any,
                          claim: Any) -> Optional[Sequence[Pair]]:
        rendered = render_bench_commands(spec)
        self._step(
            "paired_blocks",
            f"would run {spec.blocks} PRE-COMMITTED alternating paired blocks "
            "(anchor, candidate, anchor, candidate, ...). microbench.BlockPlan derives the "
            "arm sequence from the order and refuses a blocked design; the measured "
            "monotone drift is why.",
            blocks=spec.blocks, pairs_per_block=1,
            anchor_argv=rendered["anchor"]["argv"],
            candidate_argv=rendered["candidate"]["argv"],
            # BOTH envs, not one labelled `env`. They differ in exactly one key
            # and it is the load-bearing one: each arm's LD_LIBRARY_PATH points at
            # its OWN build's libs. Rendering only the anchor's invited the reader
            # to conclude the candidate runs against production's ggml — which is
            # the single worst thing that could silently be true here, because a
            # candidate linked to the anchor's libggml measures the anchor no
            # matter what it changed, and reports a clean null. The three source
            # trees run three ggml generations for the same reason.
            anchor_env=rendered["anchor"]["env"],
            candidate_env=rendered["candidate"]["env"],
            drift_bound=f"{spec.drift_bound:.4%} ({AA_EVIDENCE_REF})")
        return None

    def teardown_worktree(self, spec: CampaignSpec, tree: Any) -> Any:
        self._step("teardown_worktree",
                   "would tear the campaign worktree down and witness every frozen tree.",
                   witness_trees=list(worktree.frozen_tree_paths()))
        return "dry run"

    def keep_or_revert(self, spec: CampaignSpec, tree: Any,
                       decision: Optional[AcceptDecision]) -> Any:
        self._step("keep_or_revert",
                   "would keep the candidate branch on KEEP and delete it on REVERT.",
                   decision="none (dry run computed no delta)" if decision is None
                   else ("KEEP" if decision.keep else "REVERT"))
        return None

    def prove_production_unchanged(self, spec: CampaignSpec) -> schemas.Check:
        self._step("prove_production_unchanged",
                   "would fingerprint every frozen tree before and after and prove them "
                   "byte-identical. This runs on EVERY exit path, including the failing "
                   "ones.",
                   trees=list(worktree.frozen_tree_paths()))
        return schemas.Check(schemas.COULD_NOT_CHECK, ("dry run: nothing was fingerprinted",))

    def journal(self, spec: CampaignSpec, payload: Mapping[str, Any]) -> Any:
        self._step("journal",
                   "would append the terminal state to the append-only journal, fsynced, "
                   "under a root that is NOT a sweepable scratch directory.",
                   journal_root=spec.journal_root or "(none declared: --journal-root)",
                   state=payload.get("state"))
        return None


def render_bench_commands(spec: CampaignSpec) -> dict:
    """Both arms' argv and env, from the HASHED constructor. Executes nothing.

    `recipes.dry_run` is not a mode that suppresses an execution path — that
    module has none (`recipes.audit_no_execution_paths`). It is the same
    construction the real run uses, rendered. Bindings are placeholders under
    the campaign's own build/worktree roots and inputs are not verified, because
    in a dry run neither binary exists yet; what is being reviewed is the ARGV,
    which is the thing that drifted to 46% of canonical when nobody reviewed it.
    """
    out: dict = {}
    tool = spec.recipe.tool
    for arm, root in (("anchor", os.path.join(PRODUCTION_REPO, "build")),
                      ("candidate", spec.build_dir)):
        # `ToolBinding.source_root` is the BUILD root, not the git worktree —
        # `BuildPlan` puts the build directory OUTSIDE the worktree by
        # construction (the clean-build gate FAILs a build dir inside the
        # actor's tree), and `ToolBinding` requires the binary to resolve under
        # its own source root so that no arm can pick up another tree's ggml.
        # `bench_binding` in the reference composition binds it the same way.
        bindir = os.path.join(root, "bin")
        binding = recipes.ToolBinding(binary=os.path.join(bindir, tool),
                                      source_root=root, library_path=bindir)
        payload = recipes.dry_run(spec.recipe_id, binding=binding,
                                  params=spec.bench_params, arm=arm,
                                  verify_inputs=False)
        out[arm] = {"argv": list(payload["argv"]), "env": dict(payload["env"]),
                    # The footprint DERIVED from this argv's own `taskset -c`,
                    # which is what the claim must cover. See `CampaignSpec.cpu_list`.
                    "cpu_list": payload["claim_footprint"]["cpu_list"],
                    "cpu_count": payload["claim_footprint"]["cpu_count"]}
    both = {out[arm]["cpu_list"] for arm in out}
    if len(both) != 1:
        raise recipes.RecipeDriftError(
            f"the two arms pin different CPU footprints {sorted(both)}; one claim cannot "
            "cover both and a paired block would measure them on different cores")
    return out


class HostOps:
    """The real one. Touches the host, spends the claim, spawns the benchmarks.

    NEVER EXERCISED BY THE TEST SUITE, and it must not pretend otherwise: every
    test in this package runs on recorded output, and no line below has been run
    against a kernel. What IS covered is the composition it performs, in
    `execution/test_execution_chain.py::ChainLeg` — read that before editing
    anything here, because the seams it crosses are the ones where two modules
    have records with the same name and different fields.

    Two deliberate limits, stated rather than hidden:

      * **T0's richer evidence surfaces are supplied, not derived here.** The
        symbol diff, the diff policy and the change surface each need the
        PROPOSAL's own declarations (declared surface files, declared symbol
        deltas, registration patterns) and their producers live in
        `chain.symbol_evidence` / `chain.diff_policy_evidence` /
        `chain.change_surface_from`. Pass `t0_evidence=` to wire them; the
        reference wiring is `ChainLeg.t0_evidence_inputs`. Omitted, those gates
        read COULD_NOT_CHECK, which is true, rather than PASS, which would not
        be.
      * **The op suite must cover `MUL_MAT_ID`** — enforced on the spec, not
        here, so it cannot be bypassed by constructing ops at this layer.
    """

    executes = True

    #: The methods below that still raise `NotImplementedError`, and what each
    #: needs. `unimplemented_seams()` reads this and reports which are STILL the
    #: base class's, so a subclass that overrides one clears it automatically —
    #: a hand-maintained "is it ready" flag would be a second source of truth
    #: that goes stale in the direction that says yes.
    SEAMS_A_CAMPAIGN_MUST_SUPPLY: Mapping[str, str] = {
        "apply_candidate":
            "the candidate mutation. worktree.GitRepo carries no content-mutating git "
            "verb by construction; a campaign supplies the change the way a proposal "
            "does. Reference wiring: execution/test_execution_chain.py::ChainLeg",
        "_anchor_identity_for_bench":
            "the T1 anchor identity, MEASURED off the anchor `llama-bench` "
            "(chain.bind_anchor(capture, tool='llama-bench')). api.AnchorIdentity."
            "binary_sha256 is single-valued and llama-cli != llama-bench, so T0's "
            "triple cannot name both",
        "t0_evidence":
            "the anchor capture and the richer T0 surfaces, which need the PROPOSAL's "
            "own declarations. Pass HostOps(t0_evidence=...); the reference wiring is "
            "ChainLeg.t0_evidence_inputs",
        "nominal_khz":
            "the healthy all-core clock for this cell, which only the operator can "
            "supply (--nominal-khz). Without it every frequency reading in the run "
            "classifies FREQUENCY_UNEVALUABLE, so the throttle guard is never once "
            "JUDGED under load and MicrobenchRunner emits no number — the claim and "
            "the build are spent for nothing. cpuinfo_max_freq is the single-core "
            "boost ceiling and is NOT a valid all-core reference",
    }

    def unimplemented_seams(self) -> tuple:
        """Which required seams are still unsupplied. Empty means runnable.

        Checked BEFORE the claim, in `main`, because the alternative is what the
        code does otherwise: acquire a region claim, create a worktree, spend
        forty minutes building, and THEN raise `NotImplementedError` on the
        first line that needed a value nobody supplied. Probe the config before
        the long run (`feedback_bench_max_opt_and_config_probe_first`).

        Derived from what is actually bound — an overridden method and a
        supplied `t0_evidence` each clear their own entry — rather than from a
        flag someone has to remember to flip.
        """
        missing = [name for name in ("apply_candidate", "_anchor_identity_for_bench")
                   if getattr(type(self), name, None) is getattr(HostOps, name, None)]
        if self._t0_evidence is None:
            missing.append("t0_evidence")
        if self._nominal_khz is None:
            missing.append("nominal_khz")
        return tuple(sorted(missing))

    def __init__(self, *, spawner: Optional[Any] = None,
                 t0_evidence: Optional[Callable[..., Mapping[str, Any]]] = None,
                 nominal_khz: Optional[int] = None) -> None:
        self._spawner = spawner
        self._t0_evidence = t0_evidence
        self._nominal_khz = nominal_khz
        self._claim_binding: Optional[Any] = None
        self._device_claims: list = []
        self._build_state: dict = {}
        self._fingerprints: dict = {}

    # -- 0. preflight ------------------------------------------------------

    def preflight(self, spec: CampaignSpec) -> schemas.Check:
        """Host canonical, and nobody else on the cores. Reads; acquires nothing.

        A PASS here is an OBSERVATION, never a claim — nothing stops another
        process taking the region in the interval between this and
        `acquire_claim`. The sequence is preflight -> acquire -> run, and the
        claim is the thing that makes the run defensible.
        """
        reasons: list = []
        outcome = schemas.PASS

        def fold(check: schemas.Check, label: str, *, hard: bool = False) -> None:
            """Fold one check in. `hard` means COULD_NOT_CHECK is a refusal.

            The default is soft because the boost gate's COULD_NOT_CHECK is the
            NORMAL pre-run reading on a healthy idle host, and aborting there is
            the trap `check_boost_under_load` exists to escape.

            `hard=True` is for the concurrent-inference layer, whose own
            recommended call site is `require_no_concurrent_inference` —
            *"refuse anything but PASS"*. "I could not tell whether another
            session is benchmarking" must not start a benchmark: that is the
            fail-open direction, and it is the one that destroyed two of the six
            A/A runs on 2026-08-04.
            """
            nonlocal outcome
            if check.outcome == schemas.PASS:
                return
            reasons.extend(f"{label}: {r}" for r in check.reasons)
            if check.outcome == schemas.FAIL or hard:
                outcome = schemas.FAIL
            elif outcome != schemas.FAIL:
                outcome = schemas.COULD_NOT_CHECK

        # The frozen trees, and the fingerprints the exit-path proof compares to.
        for tree in worktree.frozen_tree_paths():
            repo = worktree.GitRepo(tree)
            self._fingerprints[tree] = worktree.fingerprint_tree(repo)

        fold(cpu_region_claim.verify_host_topology(), "topology")

        scope = (preflight.PreflightScope.gpu(spec.campaign_id, spec.devices)
                 if spec.backend == BACKEND_GPU
                 else preflight.PreflightScope.whole_machine_cpu(spec.campaign_id))
        # `device_claim_witness_reader(device_ids)` takes the ids it is a witness
        # FOR — they are required-positional, and calling it bare raised
        # TypeError on the first line of every executing run. `gpu_claim_sources`
        # is the wiring helper for both planes and defaults the two lock roots to
        # each other, which is the thing that must not diverge: two roots is how
        # the CPU and GPU planes stop excluding one another.
        sources = claim_witness.gpu_claim_sources(
            spec.devices, region_lock_dir=str(cpu_region_claim.default_region_lock_dir()))
        result = preflight.preflight(scope, sources)
        fold(schemas.Check(result.verdict, tuple(result.reasons)), "concurrent_inference",
             hard=True)

        state = microbench.read_host_state(cpu_list=spec.cpu_list)
        boosting = sum(1 for _cpu, khz in state.khz_by_cpu if khz >= BOOST_THRESHOLD_KHZ)
        boost = check_boost_under_load(boosting_cores=boosting, load1=state.load1,
                                       cpu_count=len(state.khz_by_cpu) or 1)
        # COULD_NOT_CHECK here is the IDLE case and is expected before a run:
        # it is folded, so it degrades the preflight rather than aborting it.
        fold(boost, "boost_under_load")

        policy = microbench.HostStatePolicy(nominal_khz=self._nominal_khz)
        fold(policy.check_load(state, cpu_count=len(state.khz_by_cpu) or 1), "load")

        if outcome == schemas.PASS:
            return schemas.Check(schemas.PASS, (
                f"host canonical for {spec.cpu_list}; frozen trees fingerprinted",))
        return schemas.Check(outcome, tuple(reasons))

    # -- 1. claim ----------------------------------------------------------

    def acquire_claim(self, spec: CampaignSpec) -> Any:
        """Acquire the region claim, and every device claim, or hold NOTHING.

        THE HOLE THIS CLOSES: the ledger registers the releaser only once this
        method RETURNS. Everything after `acquire_cpu_region_claim` — the seam
        check, and every device claim after the first — could raise, and the
        region claim was then held by a process on its way out with no releaser
        anywhere. "Released by the ledger" was true only on the happy path.

        So this method is transactional: anything acquired here is released here
        if the acquisition as a whole does not complete. After it returns, the
        ledger owns all of it through `release_claim`.

        THE ONE CALL SITE. `acquire_cpu_region_claim` is named exactly once
        below, inside a closure both modes go through, so the hypothesis gate
        cannot be bypassed by a second acquisition growing beside it — the same
        one-door discipline `audit_falsifier_required_before_claim` proves from
        the AST on the other side of the seam. On the bound path the acquirer is
        handed to `claim_for_hypothesis`, which supplies `purpose` off the token
        and refuses one from the caller: the falsifier the claim is being spent
        against therefore lands in the CLAIM JOURNAL, not only in the hypothesis
        ledger.
        """
        journal = cpu_region_claim.RegionClaimJournal(
            storage.assert_not_scratch(spec.claim_journal_path, what="claim journal"))

        def acquire(*, purpose: str) -> Any:
            return cpu_region_claim.acquire_cpu_region_claim(
                spec.cpu_list, role="autokernel", purpose=purpose,
                campaign_id=spec.campaign_id, journal=journal,
                timeout_s=600.0, max_hold_s=float(spec.max_hold_s))

        if spec.authorization is None:
            claim = acquire(purpose=spec.claim_purpose)
        else:
            claim = hypotheses.claim_for_hypothesis(spec.authorization, acquire)
        try:
            binding = chain.bind_claim(claim, cpu_list=spec.cpu_list)
            satisfies = chain.check_claim_satisfies_both_seams(claim, cpu_list=spec.cpu_list)
            if satisfies.outcome != schemas.PASS:
                # Raised here so the campaign stops before a TypeError an hour
                # into the window — and, since nothing outside this method knows
                # about the claim yet, released here too.
                raise RuntimeError(
                    "the acquired claim does not satisfy both consumer seams: "
                    + "; ".join(satisfies.reasons))
            self._claim_binding = binding
            for device_id in spec.devices:
                self._device_claims.append(device_claim.acquire_device_claim(
                    device_id, purpose=f"AutoKernel {spec.campaign_id}",
                    campaign_id=spec.campaign_id, journal=journal))
        except BaseException:
            self._release_device_claims()
            self._claim_binding = None
            try:
                claim.release()
            except BaseException:  # noqa: BLE001 - the original failure is the news
                pass
            raise
        return claim

    def _release_device_claims(self) -> list:
        """Release every device claim, in reverse, past a failing one."""
        detail: list = []
        while self._device_claims:
            held = self._device_claims.pop()
            try:
                detail.append(held.release().to_dict())
            except BaseException as exc:  # noqa: BLE001
                detail.append({"released": False, "error": f"{type(exc).__name__}: {exc}"})
        return detail

    def release_claim(self, claim: Any) -> Any:
        """Release the DEVICE claims and then the region claim. Both, always.

        The device claims used to be acquired and never released at all: nothing
        held a reference to them, so the flocks survived until the process died
        and the lock files kept naming a holder that no longer existed. The next
        GPU campaign then met either a stale-grace wait or
        `DeviceClaimInconsistent` — a corpse holding the MI210.
        """
        self._claim_binding = None
        devices_released = self._release_device_claims()
        region = claim.release().to_dict() if claim is not None else None
        return {"region": region, "devices": devices_released}

    # -- 2. worktree -------------------------------------------------------

    def create_worktree(self, spec: CampaignSpec) -> Any:
        repo = worktree.GitRepo(PRODUCTION_REPO)
        anchor = worktree.resolve_anchor(repo, PRODUCTION_BRANCH,
                                         expected_commit=PRODUCTION_COMMIT)
        tree, proof = worktree.create_campaign_worktree(anchor, spec.campaign_id)
        if not proof.holds:
            # The worktree EXISTS at this point and the ledger has not been told
            # about it — this method has not returned. Raising bare would leave
            # it on disk under the campaign id, and the next attempt at the same
            # campaign fails on a directory nobody remembers creating. Tear it
            # down here, and carry whatever that costs into the message rather
            # than replacing the mutation report with it.
            teardown_note = "campaign worktree torn down"
            try:
                worktree.teardown_worktree(
                    tree, witness_trees=list(worktree.frozen_tree_paths()))
            except BaseException as exc:  # noqa: BLE001 - the mutation is the news
                teardown_note = (f"AND the campaign worktree at {tree.path} could not be "
                                 f"torn down: {type(exc).__name__}: {exc}")
            raise worktree.ProductionMutated(
                f"creating the campaign worktree moved the production tree: "
                f"{list(proof.differences)} ({teardown_note})")
        return tree

    def apply_candidate(self, spec: CampaignSpec, tree: Any) -> Any:
        """Apply the candidate change into the worktree, pathspec-limited.

        NOT implemented as a generic patch applier: `worktree.GitRepo` carries
        no content-mutating verb by construction, and inventing one here would
        put a write path on the class whose whole point is that it has none.
        A campaign supplies the mutation the same way a proposal does.
        """
        raise NotImplementedError(
            f"apply_candidate({spec.candidate_ref!r}): the candidate mutation is the "
            "proposal's, not the driver's. Wire it by subclassing HostOps and overriding "
            "this one method, or land the change in the worktree before --execute. "
            "worktree.GitRepo deliberately carries no content-mutating git verb.")

    # -- 3. build ----------------------------------------------------------

    def build(self, spec: CampaignSpec, tree: Any) -> Any:
        plan = worktree.BuildPlan(
            source_root=tree.path,
            build_dir=worktree.default_build_dir(spec.campaign_id, spec.candidate_id),
            actor_worktree=tree.path,
            parallelism=worktree.BuildParallelism(jobs=64, load_average_cap=8.0),
            targets=("llama-cli", "llama-bench", "test-backend-ops"),
            cmake="/usr/bin/cmake")
        log_path = os.path.join(spec.build_root, spec.campaign_id,
                                f"{spec.candidate_id}.log")
        result = worktree.run_build(plan, log_path=log_path)
        self._build_state = {"plan": plan, "result": result, "tree": tree}
        return result

    # -- 4. T0 -------------------------------------------------------------

    def run_t0(self, spec: CampaignSpec, build: Any) -> T0Outcome:
        """T0 first, and its failure ENDS the campaign — see `run_campaign`.

        The evidence assembly crosses four seams that `chain.py` argues at
        length and that a driver must not hand-write: the build receipt is
        projected (one field INVERTS), the artifact digests are RE-MEASURED from
        disk (or the clean-build gate becomes `x == x`), the anchor is bound PER
        TOOL, and the claim is bound for both Protocols.
        """
        if self._claim_binding is None:
            raise RuntimeError("run_t0 was reached without a bound claim")
        result = self._build_state["result"]
        tree = self._build_state["tree"]
        plan = self._build_state["plan"]

        snapshot = _source_tree_digest(tree.path.path)
        identity = worktree.build_identity(
            result, candidate_id=spec.candidate_id, campaign_id=spec.campaign_id,
            worktree=tree, snapshot=snapshot,
            output_binary=os.path.join(plan.build_dir.path, "bin", "llama-cli"),
            toolchain="cmake + GNU make",
            libraries={"libggml.so.0":
                       os.path.join(plan.build_dir.path, "bin", "libggml.so.0"),
                       "libggml-base.so.0":
                       os.path.join(plan.build_dir.path, "bin", "libggml-base.so.0")})
        build_ev = chain.build_evidence(identity)                       # seam 1
        if build_ev.worst.outcome != schemas.PASS:
            return T0Outcome(all_pass=False, gates=(
                ("build_evidence", build_ev.worst.outcome, tuple(build_ev.worst.reasons)),))

        candidate = chain.candidate_build_for(identity)                 # seam 3
        extra = dict(self._t0_evidence(spec=spec, identity=identity,
                                       build_evidence=build_ev)) if self._t0_evidence \
            else {}
        anchor_capture = extra.pop("anchor_capture", None)
        if anchor_capture is None:
            raise NotImplementedError(
                "run_t0 needs the anchor capture for this tool. Supply it (with the "
                "richer T0 surfaces) through HostOps(t0_evidence=...); the reference "
                "wiring is execution/test_execution_chain.ChainLeg.bind_anchor. "
                "t0_provider.capture_anchor MEASURES it — it is never typed.")
        t0_anchor = chain.bind_anchor(anchor_capture, tool="llama-cli")  # seam 3

        t0_plan = t0_provider.T0ExecutionPlan(
            candidate=candidate,
            tools=t0_provider.ToolPaths(
                bash="/bin/bash",
                verify_ggml_linkage_sh=str(recipes.REPO_ROOT / "scripts" / "utils"
                                           / "verify_ggml_linkage.sh"),
                cmake="/usr/bin/cmake"),
            op_suite=t0_provider.OpSuitePlan(
                backend_filter="CPU" if spec.backend == BACKEND_CPU
                else recipes.GPU_VISIBLE_DEVICE_NAME,
                ops=spec.t0_ops,
                suite_id="test-backend-ops/v1",
                suite_source_sha256=identity.snapshot_sha256),
            dispatch=t0_provider.DispatchTracePlan(derived_surface=spec.t0_ops),
            generation=t0_provider.GenerationPlan(
                prompt="The capital of France is", prompt_ref="ak-prompt-001",
                n_predict=32, seed=42),
            determinism_runs=2, cache_state="cold", state_safety_probe=False,
            oracle_ids=(f"oracle://{PRODUCTION_BRANCH}",),
            build=build_ev.provenance,
            **extra)
        provider = t0_provider.ExecutedT0EvidenceProvider(
            plan=t0_plan, runner=t0_provider.SubprocessRunner(),
            claim=self._claim_binding.t0_claim,
            anchor_capture=t0_anchor.capture)
        request = self._evaluation_request(spec, identity=identity, anchor=t0_anchor)
        report = correctness.T0CorrectnessRunner(
            provider=provider, policy=correctness.T0Policy()).evaluate(request)
        gates = tuple((g.gate_id, g.check.outcome, tuple(g.check.reasons))
                      for g in report.gates)
        return T0Outcome(
            all_pass=all(outcome == schemas.PASS for _gid, outcome, _r in gates),
            gates=gates, report_ref=report.policy_ref)

    def _evaluation_request(self, spec: CampaignSpec, *, identity: Any,
                            anchor: Any) -> api.EvaluationRequest:
        """The request T0 is evaluated against. Digests MEASURED, not copied.

        `chain.measure_artifact_identity` re-walks the source tree and re-hashes
        the binary rather than reading them off the build receipt: the clean-
        build gate compares receipt against measurement, and filling both sides
        from the receipt turns two of its four sub-checks into `x == x`.
        """
        plan = self._build_state["plan"]
        tree = self._build_state["tree"]
        binary = os.path.join(plan.build_dir.path, "bin", "llama-cli")
        artifact = chain.measure_artifact_identity(          # seam 2
            source_root=tree.path.path, binary=binary,
            linkage_sha256=anchor.capture.linkage_sha256)
        command = self._construct(spec, arm="candidate")
        return api.EvaluationRequest(
            event_id=f"ake-{spec.campaign_id}-{spec.candidate_id}-t0",
            campaign_id=spec.campaign_id, candidate_id=spec.candidate_id,
            tier="T0", backend=spec.backend, phase=command.phase,
            cell_class=command.cell_class, protocol_id="P-AK-SEARCH-1",
            artifact=artifact, anchor=anchor.identity,
            evaluator=api.EvaluatorIdentity(
                evaluator_id="autokernel.campaign/v1",
                evaluator_sha256=storage.hash_file(__file__)),
            scope_denominator=command.scope_denominator,
            scope_manifest_sha256=identity.snapshot_sha256,
            co_residency="single",
            determinism=api.DeterminismReport(determinism_class="bitwise_stable"),
            metric=command.metric, metric_direction=command.metric_direction,
            reps=spec.reps, created_at=spec.created_at,
            campaign_controls=None, calibration=None)

    def _construct(self, spec: CampaignSpec, *, arm: str) -> Any:
        plan = self._build_state["plan"]
        tool = spec.recipe.tool
        # The BUILD root, not the worktree — see `render_bench_commands`.
        root = (os.path.join(PRODUCTION_REPO, "build") if arm == "anchor"
                else plan.build_dir.path)
        bindir = os.path.join(root, "bin")
        binding = recipes.ToolBinding(binary=os.path.join(bindir, tool),
                                      source_root=root, library_path=bindir)
        return recipes.construct(spec.recipe_id, binding=binding,
                                 params=spec.bench_params, arm=arm)

    # -- 5. the paired blocks ---------------------------------------------

    def run_paired_blocks(self, spec: CampaignSpec, build: Any,
                          claim: Any) -> Optional[Sequence[Pair]]:
        """N pre-committed ALTERNATING paired blocks. No extension round.

        `microbench.MicrobenchPlan` derives the arm sequence from the campaign
        seed and `BlockPlan` refuses a blocked design outright, so the
        interleaving the A/A proved mandatory is structural rather than a
        parameter. `base_blocks == spec.blocks` and no `ExtensionAuthorization`
        is ever built: this driver's accept rule does not extend, so §6.5's
        re-run-until-it-crosses hole has nothing to re-run.
        """
        if self._claim_binding is None:
            raise RuntimeError("run_paired_blocks was reached without a bound claim")
        candidate_cmd = self._construct(spec, arm="candidate")
        anchor_cmd = self._construct(spec, arm="anchor")
        anchor_identity = self._anchor_identity_for_bench(spec)
        plan = microbench.MicrobenchPlan(
            recipe_id=spec.recipe_id, candidate_id=spec.candidate_id,
            campaign_seed=f"{spec.campaign_id}/{spec.created_at}",
            candidate_binding=candidate_cmd.binding,
            anchor_binding=anchor_cmd.binding,
            anchor=anchor_identity,
            params=spec.bench_params,
            base_blocks=spec.blocks, pairs_per_block=1,
            unit_ids=(f"{spec.recipe_id}:{spec.model or 'declared-model'}",),
            stratum=api.STRATUM_SELECTION)
        # The schedule is a per-block coin flip and a 5-0 draw is inadmissible
        # (see `decide`). It is knowable from the plan alone, so it is checked
        # HERE — before the blocks are spent — rather than only after them. The
        # remedy is a fresh campaign seed, so the run is refused with the claim
        # still in hand and nothing measured.
        drawn = plan.schedule.orders(spec.blocks)
        if len(set(drawn)) < 2:
            raise RuntimeError(
                f"the order schedule for campaign seed {plan.campaign_seed!r} draws "
                f"{drawn[0]!r} for all {spec.blocks} blocks. That is a sequential A/B, and "
                "`decide()` refuses it after the fact; refusing it here costs no blocks. "
                "Re-run under a fresh campaign seed (retry() reverses every element and "
                "would produce the mirror-image degenerate draw).")
        runner = microbench.MicrobenchRunner(
            claim=self._claim_binding.microbench_claim,
            policy=microbench.HostStatePolicy(nominal_khz=self._nominal_khz),
            spawner=self._spawner or microbench.SubprocessSpawner())
        run = runner.run(plan)
        return pairs_from_run(run)

    def _anchor_identity_for_bench(self, spec: CampaignSpec) -> api.AnchorIdentity:
        raise NotImplementedError(
            "the T1 anchor identity must be MEASURED off the anchor `llama-bench` "
            "(chain.bind_anchor(capture, tool='llama-bench')), and tied to T0's anchor by "
            "chain.check_anchor_build_is_one_build. api.AnchorIdentity.binary_sha256 is "
            "single-valued and llama-cli and llama-bench are different files, so one "
            "triple cannot name both. Supply it by overriding this method.")

    # -- 6. teardown -------------------------------------------------------

    def teardown_worktree(self, spec: CampaignSpec, tree: Any) -> Any:
        receipt = worktree.teardown_worktree(
            tree, witness_trees=list(worktree.frozen_tree_paths()))
        return receipt.to_dict()

    def keep_or_revert(self, spec: CampaignSpec, tree: Any,
                       decision: Optional[AcceptDecision]) -> Any:
        """KEEP is a branch that survives teardown; REVERT is one that does not.

        Nothing is promoted here and nothing is merged. The release plane is
        what SHIPS a champion, and this driver's job ends at a banked result.
        """
        return {"keep": bool(decision and decision.keep),
                "branch": getattr(getattr(tree, "branch", None), "name", None)}

    # -- 7. the invariant ---------------------------------------------------

    def prove_production_unchanged(self, spec: CampaignSpec) -> schemas.Check:
        if not self._fingerprints:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "no pre-run fingerprint was taken (the preflight did not run), so "
                "byte-identity cannot be proved. That is not a pass.",))
        reasons: list = []
        for tree, before in sorted(self._fingerprints.items()):
            after = worktree.fingerprint_tree(worktree.GitRepo(tree))
            proof = worktree.prove_unchanged(before, after)
            if not proof.holds:
                reasons.extend(f"{tree}: {d}" for d in proof.differences)
        if reasons:
            return schemas.Check(schemas.FAIL, tuple(reasons))
        return schemas.Check(schemas.PASS, (
            f"{len(self._fingerprints)} frozen tree(s) byte-identical",))

    # -- 8. durability ------------------------------------------------------

    def journal(self, spec: CampaignSpec, payload: Mapping[str, Any]) -> Any:
        """Journal the terminal state, fsynced, before the process can exit.

        AutoPilot lost 232 trials / ~16 days to a loop that held its results in
        memory. `Journal.append` returns only after `fsync`, and it refuses a
        malformed payload rather than coercing one.
        """
        if not spec.journal_root:
            return None
        root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
        book = journal_module.Journal(root, campaign_id=spec.campaign_id)
        book.initialize()
        entry = book.append(journal_module.KIND_STOP_STATE, dict(payload))
        return entry.event_id


def _source_tree_digest(root: str) -> Any:
    """The candidate worktree's snapshot digest, for `worktree.build_identity`.

    THE ONE `integrity` CALL IN THIS FILE, and it is made through `chain` — the
    module whose entire job is the projection between `integrity`'s derivation
    and `correctness`'s shape — rather than by importing `integrity` here.
    Importing it directly would make this driver a SECOND consumer of a
    derivation the package already has two of (`integrity.py` over ELF tables
    and parsed diffs, `correctness.py` over declared evidence objects), and
    converging those two is an AK4 decision, not a driver's. Pinned by
    `test_campaign.TestTheBoundaryIsStructural`.
    """
    return chain.integrity.hash_source_tree(root, exclude_dir_names=(".git",))


def pairs_from_run(run: Any) -> tuple:
    """`microbench.MicrobenchRun` -> the accept rule's input. Refuses a partial run.

    `run.paired_blocks()` RAISES on an incomplete or refused run and that
    refusal is deliberately not caught: *"do not reach for the partial blocks"*.
    Each arm is reduced to the MEDIAN of its repetitions — the same reduction
    the A/A used, and one that a single stalled repetition cannot drag.
    """
    out: list = []
    for block in run.paired_blocks():
        out.append(Pair(block_index=block.block_index,
                        anchor=float(median(block.anchor_samples)),
                        candidate=float(median(block.candidate_samples)),
                        order=block.order))
    return tuple(out)


# =============================================================================
# The loop
# =============================================================================

STATE_COMPOSED = "dry_run_composed"
STATE_PREFLIGHT_REFUSED = "preflight_refused"
STATE_T0_FAILED = "t0_failed"
STATE_DECIDED = "decided"
STATE_ERROR = "error"


@dataclass(frozen=True)
class CampaignResult:
    """One candidate's whole outcome, including the ones that are not numbers."""

    spec: CampaignSpec
    state: str
    steps: tuple
    t0: Optional[T0Outcome] = None
    decision: Optional[AcceptDecision] = None
    pairs: tuple = ()
    preflight: Optional[schemas.Check] = None
    production_unchanged: Optional[schemas.Check] = None
    releases: tuple = ()
    error: Optional[str] = None
    #: Whether this run actually touched the host. A composition pass proves
    #: nothing about the frozen trees because it read nothing from them.
    executed: bool = False
    #: Why the durable record could not be written, when one was asked for.
    journal_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """A campaign is OK when it terminated cleanly and left production alone.

        Note what is NOT in here: whether the candidate won. A REVERT is a
        successful campaign. Banking is not the same as winning.

        On an EXECUTING run the immutability proof must be a PASS, not merely
        not-a-FAIL. `_finish` converts a proof that RAISED into COULD_NOT_CHECK
        and says in its own reason that this "outranks everything else in this
        run" — and the run then exited 0 anyway. COULD_NOT_CHECK is exactly the
        state a proof reaches when the thing it inspects has been disturbed, so
        treating it as clean is the fail-open shape (`feedback_fail_open_
        defaults_conceal_their_own_corruption`). A composition pass is exempt
        because it fingerprinted nothing and claims nothing.
        """
        if self.state == STATE_ERROR:
            return False
        # A result that could not be written down is a result AutoPilot lost 232
        # trials / ~16 days to. The journal failure was printed to stderr and the
        # process exited 0; a wrapper reading the exit code learned nothing.
        if self.journal_error and self.spec.journal_root:
            return False
        if self.executed:
            if self.production_unchanged is None \
                    or self.production_unchanged.outcome != schemas.PASS:
                return False
        elif self.production_unchanged is not None \
                and self.production_unchanged.outcome == schemas.FAIL:
            return False
        return all(record.released for record in self.releases)

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.campaign_result.v1",
            "state": self.state,
            "campaign_id": self.spec.campaign_id,
            "candidate_id": self.spec.candidate_id,
            "spec": self.spec.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "t0": self.t0.to_dict() if self.t0 else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "pairs": [p.to_dict() for p in self.pairs],
            "preflight": None if self.preflight is None else {
                "outcome": self.preflight.outcome, "reasons": list(self.preflight.reasons)},
            "production_unchanged": None if self.production_unchanged is None else {
                "outcome": self.production_unchanged.outcome,
                "reasons": list(self.production_unchanged.reasons)},
            "releases": [r.to_dict() for r in self.releases],
            "error": self.error,
            "executed": self.executed,
            "journal_error": self.journal_error,
            "ok": self.ok,
            "grammar": "SEARCH RECORD, NOT A CLAIM",
        }


def run_campaign(spec: CampaignSpec, ops: Any) -> CampaignResult:
    """One candidate, once, through the whole loop. Never raises past the ledger.

    The order below is the deliverable, and two positions in it are the reason
    this function exists rather than a runbook paragraph:

      * **T0 is before the blocks, and its failure returns.** There is no path
        from a failed T0 to a `run_paired_blocks` call, and `decide()` refuses
        to be called anyway. A wrong kernel gets no speed rank at all.
      * **The `finally` runs the releases and the immutability proof on EVERY
        path**, including the one where the body raised. A driver that dies
        holding a claim is how the next session finds the host locked by a
        corpse; a driver that skips the production-tree proof on the failing
        path skips it exactly when it mattered.
    """
    ledger = ResourceLedger()
    # A composition pass (`--dry-run`) walks the same steps and takes no branch
    # that depends on a result, because it produced none.
    executes = bool(getattr(ops, "executes", True))
    t0: Optional[T0Outcome] = None
    decision: Optional[AcceptDecision] = None
    pairs: tuple = ()
    pre: Optional[schemas.Check] = None
    state = STATE_ERROR
    error: Optional[str] = None
    tree = None

    try:
        pre = ops.preflight(spec)
        if pre.outcome == schemas.FAIL:
            state = STATE_PREFLIGHT_REFUSED
            error = "; ".join(pre.reasons)
            return _finish(spec, ops, ledger, state=state, t0=t0, decision=decision,
                           pairs=pairs, pre=pre, error=error, tree=tree)

        claim = ops.acquire_claim(spec)
        ledger.hold("cpu_region_claim", lambda: ops.release_claim(claim))

        tree = ops.create_worktree(spec)
        captured_tree = tree
        ledger.hold("campaign_worktree",
                    lambda: ops.teardown_worktree(spec, captured_tree))

        ops.apply_candidate(spec, tree)
        build = ops.build(spec, tree)

        t0 = ops.run_t0(spec, build)
        if executes and not t0.all_pass:
            # STOP. No speed number is computed at all. This is the ONE branch
            # a composition pass does not take, because in a composition pass
            # T0 was not executed and therefore did not fail — it produced
            # nothing. Every other step is walked identically, and
            # `test_campaign` pins the two call orders against each other so the
            # loop cannot acquire a second spelling.
            state = STATE_T0_FAILED
            return _finish(spec, ops, ledger, state=state, t0=t0, decision=None,
                           pairs=(), pre=pre, error=None, tree=tree)

        observed = ops.run_paired_blocks(spec, build, claim)
        if observed is None:
            if executes:
                raise RuntimeError(
                    "run_paired_blocks returned no observations on an EXECUTING run. A run "
                    "that produced nothing must raise or refuse; returning None is how a "
                    "dry run says 'composed, not measured', and an executing run must not "
                    "be able to say it.")
            state = STATE_COMPOSED
            return _finish(spec, ops, ledger, state=state, t0=t0, decision=None,
                           pairs=(), pre=pre, error=None, tree=tree)

        pairs = tuple(observed)
        decision = decide(pairs, t0=t0, blocks_precommitted=spec.blocks,
                          drift_bound=spec.drift_bound)
        state = STATE_DECIDED
        return _finish(spec, ops, ledger, state=state, t0=t0, decision=decision,
                       pairs=pairs, pre=pre, error=None, tree=tree)
    # BaseException, not Exception, and deliberately: the realistic way a
    # campaign ends early is Ctrl-C an hour into a claim window, and
    # `KeyboardInterrupt` is not an `Exception`. Catching only `Exception` here
    # would leave the one interruption an operator actually performs as the one
    # path that leaks the claim.
    except BaseException as exc:  # noqa: BLE001 - the ledger must run on EVERY path
        error = f"{type(exc).__name__}: {exc}"
        state = STATE_ERROR
        return _finish(spec, ops, ledger, state=state, t0=t0, decision=decision,
                       pairs=pairs, pre=pre, error=error, tree=tree,
                       traceback_text=traceback.format_exc())


def _finish(spec: CampaignSpec, ops: Any, ledger: ResourceLedger, *, state: str,
            t0: Optional[T0Outcome], decision: Optional[AcceptDecision],
            pairs: tuple, pre: Optional[schemas.Check], error: Optional[str],
            tree: Any, traceback_text: Optional[str] = None) -> CampaignResult:
    """Keep-or-revert, release everything, prove production, journal. Always.

    Ordered so that the two things that must happen even when everything else
    failed happen last and unconditionally: the releases, and the proof that the
    frozen trees are byte-identical. `keep_or_revert` runs first because it needs
    the worktree the ledger is about to tear down, and its own failure is
    recorded rather than allowed to skip the release.
    """
    if tree is not None:
        try:
            ops.keep_or_revert(spec, tree, decision)
        # BaseException: `keep_or_revert` runs BEFORE the releases, so an
        # interrupt here — the second Ctrl-C, the one during teardown — would
        # otherwise skip `release_all()` entirely and leak the claim.
        except BaseException as exc:  # noqa: BLE001
            error = "; ".join(x for x in (error, f"keep_or_revert: {exc}") if x)

    releases = ledger.release_all()

    try:
        unchanged = ops.prove_production_unchanged(spec)
    except BaseException as exc:  # noqa: BLE001
        unchanged = schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the production-tree immutability proof itself raised: {exc}. That is not a "
            "pass, and it outranks everything else in this run.",))

    result = CampaignResult(
        spec=spec, state=state, steps=tuple(getattr(ops, "steps", ())),
        t0=t0, decision=decision, pairs=pairs, preflight=pre,
        production_unchanged=unchanged, releases=releases,
        # The same expression `run_campaign` branches on, read from the same
        # object, so the record cannot disagree with the loop about whether the
        # host was touched.
        executed=bool(getattr(ops, "executes", True)),
        error="\n".join(x for x in (error, traceback_text) if x) or None)

    try:
        ops.journal(spec, {"state": state, "campaign_id": spec.campaign_id,
                           "result": result.to_dict()})
    # A failed journal must not HIDE the result — it is returned either way —
    # but it must not be silent in the exit code either. A wrapper that reads
    # only the status learned nothing from a warning on stderr, and an
    # unjournaled loop is how AutoPilot lost 232 trials / ~16 days.
    except BaseException as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: the result could not be journaled: {detail}", file=sys.stderr)
        # Frozen dataclass: rebuilt rather than mutated, so the record and its
        # `ok` cannot disagree.
        result = replace(result, journal_error=detail)
    return result


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m scripts.kernel_rnd.autokernel.campaign",
        description="AutoKernel campaign driver — one candidate through the whole loop.",
        epilog="DRY RUN IS THE DEFAULT. --execute additionally requires "
               "--i-hold-the-host, because the loop spends hours of a claim on a "
               "shared machine.")
    parser.add_argument("--campaign-id", default="ak-0001",
                        help="campaign id; must start with 'ak-' (default: ak-0001)")
    parser.add_argument("--candidate-id", default="akc-0001",
                        help="candidate id; must start with 'akc-' (default: akc-0001)")
    parser.add_argument("--candidate", dest="candidate_ref", default="(none declared)",
                        help="the patch file or branch under test")
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND_CPU)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS,
                        help=f"the PRE-COMMITTED number of paired blocks "
                             f"(default: {DEFAULT_BLOCKS}). Fixed before the run: the "
                             f"accept rule refuses any other count, which is what makes "
                             f"optional stopping impossible rather than discouraged.")
    parser.add_argument("--recipe", dest="recipe_id", default=None,
                        choices=list(recipes.RECIPE_IDS) + [None],
                        help="codified recipe id (default: the canonical decode slice "
                             "for the backend)")
    parser.add_argument("--model", default=None, help="absolute path to the GGUF")
    parser.add_argument("--reps", type=int, default=5, help="llama-bench -r (default: 5)")
    parser.add_argument("--device", action="append", default=[],
                        help="GPU device id (e.g. ROCm0), repeatable; required for "
                             "--backend llama_gpu")
    parser.add_argument("--device-name", action="append", default=[],
                        help="what the runtime CALLS that device (e.g. 'AMD Instinct "
                             "MI210'). Graded by evaluator/devices.py: a GPU cell must "
                             "not be satisfied by 'Device 0: CPU'")
    parser.add_argument("--nominal-khz", type=int, default=None,
                        help="the healthy all-core clock for this cell, in kHz, from a "
                             "recorded healthy observation. Required by --execute: "
                             "without it the throttle guard reads COULD_NOT_CHECK and "
                             "MicrobenchRunner refuses at run open. NOT cpuinfo_max_freq "
                             "— that is the single-core boost ceiling")
    parser.add_argument("--journal-root", default=None,
                        help="append-only journal root; refused if it is a sweepable "
                             "scratch path (the 2026-07-04 win was written to "
                             "/mnt/raid0/llm/tmp/ and that directory no longer exists)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                       help="compose and print every step, execute nothing (DEFAULT)")
    group.add_argument("--execute", dest="dry_run", action="store_false",
                       help="actually run. Requires --i-hold-the-host.")
    parser.add_argument("--i-hold-the-host", action="store_true", default=False,
                        help="attest that this session owns the machine for the claim "
                             "window. Required by --execute.")
    parser.add_argument("--hypothesis", default=None, metavar="AKH-ID",
                        help="bind this campaign to ONE stated question (akh-…). The "
                             "claim is then acquired through "
                             "hypotheses.claim_for_hypothesis, so a question with no "
                             "falsifier — or a placeholder one ('tbd') — cannot reach a "
                             "claim, and the refusal happens here rather than an hour "
                             "into a held window. WITHOUT it the campaign is EXPLORATORY "
                             "and the record says so. Requires --journal-root")
    parser.add_argument("--hypothesis-store", default=None, metavar="PATH",
                        help="the operator's drop-in JSON file (HYPOTHESES.md). Its "
                             "entries are taken into the ledger before the claim is "
                             "authorized, which is what makes a line the operator typed "
                             "a tracked question. Omit it when the id is already tracked")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="print the result document as JSON on stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, out: Optional[Any] = None,
         ops: Optional[Any] = None) -> int:
    """Compose and run one campaign. Returns a process exit code.

    `0` means the campaign terminated cleanly and production is byte-identical —
    NOT that the candidate won. A REVERT is a successful campaign.
    """
    stream = out if out is not None else sys.stdout
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    if not args.dry_run and not args.i_hold_the_host:
        print("--execute requires --i-hold-the-host. This driver spawns benchmarks on a "
              "shared machine; two of the six A/A runs on 2026-08-04 were destroyed by a "
              "legitimate co-tenant, and the co-tenant did nothing wrong.", file=sys.stderr)
        return 2

    try:
        spec = CampaignSpec(
            campaign_id=args.campaign_id, candidate_id=args.candidate_id,
            candidate_ref=args.candidate_ref, backend=args.backend, blocks=args.blocks,
            recipe_id=args.recipe_id, model=args.model, reps=args.reps,
            devices=tuple(args.device), device_names=tuple(args.device_name),
            journal_root=args.journal_root)
    except (ValueError, TypeError, storage.StorageError, recipes.RecipeError) as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return 2

    if ops is None:
        ops = (DryRunOps(out=stream) if args.dry_run
               else HostOps(nominal_khz=args.nominal_khz))

    # BEFORE the claim, and before the banner: `HostOps` is written and has
    # never been run, and four of its seams still need a value only the
    # campaign can supply. Each raises where it is REACHED — after the region
    # claim, after the worktree, after the build — so the cost of discovering
    # it is the claim window. Refusing here costs argv-parse time. A run that
    # never started must not print a line saying EXECUTING either.
    unimplemented = getattr(ops, "unimplemented_seams", None)
    if not args.dry_run and callable(unimplemented):
        pending = unimplemented()
        if pending:
            print("refusing to --execute: this ops object cannot complete a run. The "
                  "following seams still need a value the campaign must supply:",
                  file=sys.stderr)
            for name in pending:
                why = HostOps.SEAMS_A_CAMPAIGN_MUST_SUPPLY.get(name, "")
                print(f"  {name}: {why}", file=sys.stderr)
            print("Subclass HostOps and override them (or pass --dry-run, which composes "
                  "the whole loop and executes nothing).", file=sys.stderr)
            return 2

    # THE GATE, and it is here for the same reason the block above is: nothing
    # has been acquired yet, no worktree exists, and the banner has not printed a
    # line saying a bound campaign started. A hypothesis with no falsifier, a
    # placeholder one, an unknown id or a receipted repeat all stop on this
    # statement. `--hypothesis` is NOT downgraded to exploratory by any failure:
    # a typo that silently produced an unbound run would be worse than the typo.
    if args.hypothesis is not None:
        try:
            spec = replace(spec, authorization=authorize_for(
                spec, args.hypothesis, store_path=args.hypothesis_store,
                dry_run=args.dry_run))
        except (HypothesisBindingError, hypotheses.HypothesisError,
                do_not_repeat.DoNotRepeatError, storage.StorageError,
                journal_module.JournalError, ValueError, TypeError, OSError) as exc:
            print(f"refusing to start: --hypothesis {args.hypothesis}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    print(f"AutoKernel campaign {spec.campaign_id} / {spec.candidate_id} — "
          f"{'DRY RUN (nothing will be executed)' if args.dry_run else 'EXECUTING'}",
          file=stream)
    print("  question    "
          + (f"{spec.hypothesis_id} — falsifier: {spec.authorization.falsifier}"
             if spec.authorization is not None
             else "EXPLORATORY (no --hypothesis; this run resolves no question)"),
          file=stream)
    print(f"  cell        {spec.recipe_id}  metric={spec.metric}", file=stream)
    print(f"  accept      min(delta) > 0 over {spec.blocks} pre-committed pairs AND "
          f"median(relative) > {spec.drift_bound:.4%}", file=stream)
    print(f"  drift bound measured, not assumed: {AA_EVIDENCE_REF}", file=stream)
    print("", file=stream)

    result = run_campaign(spec, ops)

    print("", file=stream)
    print(f"state: {result.state}", file=stream)
    if result.decision is not None:
        print(result.decision.reason, file=stream)
    if result.error:
        print(f"error: {result.error.splitlines()[0]}", file=stream)
    for record in result.releases:
        marker = "released" if record.released else "NOT RELEASED"
        print(f"  {record.name}: {marker} ({record.detail})", file=stream)
    if result.production_unchanged is not None:
        print(f"  production trees: {result.production_unchanged.outcome}", file=stream)
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str),
              file=stream)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised by test_campaign via main()
    sys.exit(main())
