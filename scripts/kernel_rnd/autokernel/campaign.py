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
  5. **KEEP iff `min(delta) > 0` AND `median(relative) > contribution_floor`.**
     The floor and minimum block count come from the accepted, cell-local live
     calibration; they are not copied from a different phase or metric.
  6. Otherwise REVERT.

Why those two conjuncts and no more, from the A/A rather than from theory:

  * `min(delta) > 0` — every pair must favour the candidate. Under a null the
    sign of each pair is a coin flip; the current bundle supplies the minimum
    admissible N before any observations exist. It
    also has the property the median alone does not: one adverse block sinks
    the candidate, which is the right posture when the alternative is spending
    a claim window confirming something the instrument cannot resolve.
  * `median(relative) > contribution_floor` — the current identity-bound
    five-control bundle supplies the floor and B_min. The historical v8 3% /
    B_min=12 result and earlier 2.131% decode step are not v9 ranking authority.
    `DRIFT_BOUND_BY_METRIC` remains a separate neutral/instrument-movement
    control derived from the adjacent A/A readings.
  * N is PRE-COMMITTED on the spec and `decide()` refuses any other count, so
    there is no optional stopping, no multiplicity, and no e-process to make
    inert. This closes, by construction, the one open manufacture-a-crossing
    hole in `execution/README.md` §6.5 (re-run a declared round until it
    crosses): there is no round to re-run.

The rule is deliberately conservative and cell-local. It cannot rank a +2% win,
and it refuses live ranking entirely for a recipe without accepted calibration.

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
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from math import ceil, isclose, isfinite
from pathlib import Path
# The STDLIB median, deliberately. `evaluator.statistics` is NOT imported: its
# e-process solved a harder problem than the measured 1.6–1.9% CV poses, and it
# made the gate unpassable at B_min (threshold 10, ceiling 5.5687 at every
# effect size). See MODULES_DELIBERATELY_NOT_USED.
from statistics import median
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from . import (artifact_diff, candidate_record, dashboard, least_commitment_capture,
               journal as journal_module, source_candidate,
               source_prerequisite_package, source_prerequisite_producer)
from . import schemas, storage
from .controller import do_not_repeat, hypotheses
from .evaluator import api, correctness, devices, recipes
from .execution import (chain, control_runner, cpu_region_claim, device_sampler,
                        inference_window, instrument_integrity, microbench, physical_bounds,
                        powercap_broker, provider, sandbox, screening_baseline,
                        t0_provider, worktree)
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
    "RANKED_UNITS_SCHEMA", "RANKED_UNIT_NORMAL",
    "RANKED_UNIT_ANTI_SHORT_CIRCUIT", "RankedUnitSpec",
    "ranked_units_from_mapping",
    "CampaignSpec",
    "PRODUCTION_REPO", "PRODUCTION_BRANCH", "PRODUCTION_COMMIT",
    "MEASUREMENT_REPO", "MEASUREMENT_BRANCH", "MEASUREMENT_COMMIT",
    "MEASUREMENT_BUILD_ROOT",
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

# T0 deliberately occupies the whole CPU claim.  Linux's one-minute load
# average therefore remains high for a while after the final T0 child has been
# contained and reaped.  T1's run-open contention gate must not reinterpret
# that known, already-finished work as a foreign co-tenant.  This is a phase
# boundary, not a relaxation: after the fixed decay interval we take three
# independent low-load observations, each paired with a fresh claim witness.
POST_T0_QUIET_BARRIER_S = 65.0
POST_T0_QUIET_SAMPLES = 3
POST_T0_QUIET_SAMPLE_INTERVAL_S = 5.0

# A completed arm's build can remain in Linux's one-minute load EWMA after its
# owned children are reaped.  The following arm must not treat that known,
# already-finished work as a reason to abandon before T0; wait only until the
# same declared build cap becomes true, then let `run_build` recheck at spawn.
# The one-minute EWMA can still exceed the cap after the first 65 seconds when
# a full 64-way predecessor build has just completed.  Three minutes remains a
# bounded, claim-held settling interval while covering that measured tail.
BUILD_LOAD_SETTLE_TIMEOUT_S = 180.0
BUILD_LOAD_SETTLE_INTERVAL_S = 5.0


# =============================================================================
# The boundary. Enforced by test_campaign.TestTheBoundaryIsStructural.
# =============================================================================

#: Every module this driver may import, and the single reason each is essential.
#: Each reason is a real incident or a fact measured on this host, not a taste.
MODULES_THE_DRIVER_USES: Mapping[str, str] = {
    "artifact_diff": "AK-TR-6 compile-only register/scratch/instruction movement vetoes "
                     "a GPU claim before the behavioural T0 provider can launch",
    "candidate_record": "every executed candidate is durably recorded from the exact "
                        "built snapshot and evaluation event identities",
    "least_commitment_capture": "a live IQK result must carry the predeclared, hash-bound "
                                "diagnostics and measured outcome reducers needed by the "
                                "observe-only AK-WM-2 archive",
    "source_candidate": "source-changing proposals consume one immutable embedded patch "
                        "bundle through the guarded worktree mutation boundary",
    "source_prerequisite_package": "source candidates may rank only after the archived "
                                   "raw sensitivity/hostile/checker CSV bytes are re-reduced "
                                   "and rebound to the exact live source, binary and evaluator",
    "source_prerequisite_producer": "fresh source receipts reuse the campaign's already-held "
                                    "claims and built candidate, then enter the identical "
                                    "content-addressed archive/reducer boundary before T0",
    "dashboard": "the fsynced terminal campaign result must reach the operator surface; "
                 "the exporter is derived and cannot make an old journal entry fresh",
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
    "execution.control_runner": "the accepted five-control/calibration bundle is the "
                                "authority for prospective EVALUATION_EVENT reductions; "
                                "the live loop may not invent its own e-value or panel",
    "execution.chain": "the seams; a hand-written evidence record is what T0 exists to refuse",
    "execution.instrument_integrity": "RVP-C6-1: candidate reward translation units "
                                      "must equal the reviewed measurement overlay before "
                                      "T0 or T1 can launch",
    "execution.inference_window": "CPU and GPU candidate preparation may overlap, while "
                                  "every actual model-load/inference call is serialized by "
                                  "one recoverable host-wide mutex and receipts its wait/release",
    "execution.cpu_region_claim": "TODO-free: two A/A runs were destroyed by a legitimate "
                                  "co-tenant because we held no claim",
    "execution.device_sampler": "RVP-C3-4: a GPU result needs numeric power/clock/temperature "
                                "samples every 250 ms across each exact subprocess lifetime; "
                                "two endpoint text blobs cannot reveal a mid-run excursion",
    "execution.microbench": "paired ALTERNATING blocks — the measured monotone drift makes "
                            "any sequential design charge the second arm ~4%",
    "execution.physical_bounds": "RVP-C6-4 refuses a live throughput sample that exceeds "
                                 "the predeclared compute-or-memory speed of light; the "
                                 "candidate cannot author its own work denominator",
    "execution.powercap_broker": "root-owned package counters are read by one captured, "
                                 "networkless read-only container; candidate code remains "
                                 "non-root and cannot reach its control plane",
    "execution.provider": "provider roots are realpath-resolved and must not overlap "
                          "shared ROCm/system prefixes or frozen production trees",
    "execution.sandbox": "C6: code authored by the loop executes under Landlock, seccomp, "
                          "non-root finite rlimits and an owned cgroup whose empty teardown "
                          "is verified; an agent tool allowlist does not constrain a binary",
    "execution.screening_baseline": "amortized exact-frame anchor banks support noisy "
                                    "candidate-only discovery but cannot grant rank authority",
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

#: Where the legacy exploratory numbers above live. They remain the independent
#: anchor-movement control; they are no longer the candidate contribution floor.
AA_EVIDENCE_REF = "data/autokernel_aa_20260804/README.md"

#: Historical v8 control evidence.  These constants are regression fixtures,
#: never v9 ranking authority.  A live campaign consumes a current bundle via
#: ``--calibration-bundle`` and verifies both source identities first.
HISTORICAL_CALIBRATION_EVIDENCE_REF = "data/autokernel_controls_3pct_20260805/"
HISTORICAL_CALIBRATED_RECIPE_ID = "t1b.llama_cpu.llama_bench_prefill.v1"
# The current governed CPU IQK experiment is a one-invocation paired-block
# instrument.  This is deliberately separate from CampaignSpec's generic
# legacy default: the matched-pair producer below binds every executable IQK
# pair to this value, while unrelated campaign families retain their own frame.
IQK_MATCHED_PAIR_REPS = 1
HISTORICAL_CONTRIBUTION_FLOOR = 0.03
HISTORICAL_B_MIN_BLOCKS = 12
HISTORICAL_MAX_BLOCKS = 20
HISTORICAL_NOISE_FLOOR_PHI = 0.049206882811302755
HISTORICAL_MDE = 0.027408174371940427


@dataclass(frozen=True)
class LeanCalibration:
    """Accepted outputs the session-driven rule is licensed to consume."""

    recipe_id: str
    contribution_floor: float
    b_min_blocks: int
    max_blocks: int
    noise_floor_phi: float
    mde: float
    production_commit: str
    measurement_commit: str
    evidence_ref: str
    evaluation_authority: Optional[control_runner.LiveEvaluationAuthority] = None
    # A current calibration must also measure the anchor over the exact
    # precommitted T1 window it licenses.  These remain optional for direct
    # arithmetic fixtures; an executing campaign refuses their absence below.
    anchor_motion_bound: Optional[float] = None
    anchor_motion_window_blocks: Optional[int] = None
    anchor_motion_evidence_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "recipe_id": self.recipe_id,
            "contribution_floor": self.contribution_floor,
            "b_min_blocks": self.b_min_blocks,
            "max_blocks": self.max_blocks,
            "noise_floor_phi": self.noise_floor_phi,
            "mde": self.mde,
            "production_commit": self.production_commit,
            "measurement_commit": self.measurement_commit,
            "evidence_ref": self.evidence_ref,
            "anchor_motion_bound": self.anchor_motion_bound,
            "anchor_motion_window_blocks": self.anchor_motion_window_blocks,
            "anchor_motion_evidence_ref": self.anchor_motion_evidence_ref,
        }


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

#: §10.7's maximum admissible uptime. Crossing it never reboots the shared
#: host; it returns a refusal that belongs in an operator decision package.
MAX_HOST_UPTIME_S = 7 * 24 * 60 * 60


def check_host_uptime(uptime_s: Optional[float]) -> schemas.Check:
    """Refuse a measurement window after one week without mutating the host."""
    if isinstance(uptime_s, bool) or not isinstance(uptime_s, (int, float)):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "/proc/uptime was unreadable, so the one-week host-health ceiling could not "
            "be evaluated",))
    if not isfinite(float(uptime_s)) or uptime_s < 0:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"/proc/uptime returned an invalid value {uptime_s!r}",))
    if uptime_s > MAX_HOST_UPTIME_S:
        return schemas.Check(schemas.FAIL, (
            f"host uptime {uptime_s / 86400:.2f} days exceeds the one-week ceiling; "
            "refuse measurement and route a reboot decision package to the operator",))
    return schemas.Check(schemas.PASS, (
        f"host uptime {uptime_s / 86400:.2f} days is within the one-week ceiling",))


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
class CandidateOnlyObservation:
    """One discovery call compared descriptively with an amortized bank.

    This is intentionally not a :class:`Pair`: no anchor ran in this block and
    therefore there is no within-block arm order.  Treating ``candidate_only``
    as a Pair order would either lie about execution or violate Pair's paired
    design invariant.
    """

    block_index: int
    baseline_reference: float
    candidate: float

    def __post_init__(self) -> None:
        for name in ("baseline_reference", "candidate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or value <= 0:
                raise ValueError(f"CandidateOnlyObservation.{name} must be positive")

    @property
    def delta(self) -> float:
        return float(self.candidate) - float(self.baseline_reference)

    @property
    def relative(self) -> float:
        return self.delta / float(self.baseline_reference)

    def to_dict(self) -> dict:
        return {"block_index": self.block_index,
                "observation_kind": "candidate_only",
                "baseline_reference": self.baseline_reference,
                "candidate": self.candidate, "delta": self.delta,
                "relative": self.relative}


@dataclass(frozen=True)
class T0Outcome:
    """T0's answer, reduced to the one bit the loop branches on, plus the detail."""

    all_pass: bool
    gates: tuple = ()          # ((gate_id, outcome, (reason, ...)), ...)
    report_ref: Optional[str] = None
    # The evaluator-native records are retained for the prospective journal
    # writer.  The flattened triples above remain the small accept-loop seam.
    gate_results: tuple = ()

    @property
    def failures(self) -> tuple:
        return tuple(g for g in self.gates if g[1] == schemas.FAIL)

    def to_dict(self) -> dict:
        return {"all_pass": self.all_pass, "report_ref": self.report_ref,
                "gates": [[gid, outcome, list(reasons)] for gid, outcome, reasons
                          in self.gates]}


class T0EvaluationAdmissionRefusal(RuntimeError):
    """The measured T0 gates passed, but their governed event cannot admit T1.

    A gate-only ``T0Outcome`` is deliberately a small loop seam.  It is not,
    however, sufficient authority to spend the first timing block: the event
    must also cite the exact ratified protocol and validate under the event
    schema.  This typed refusal lets the driver retain T0's terminal state and
    skip T1 rather than accidentally classifying a bad evidence record as a
    generic driver failure after timing has begun.
    """


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
    contribution_floor: Optional[float] = None
    calibration_evidence_ref: Optional[str] = None
    drift_bound: Optional[float] = None
    anchor_drift: Optional[float] = None
    deltas: tuple = ()
    relatives: tuple = ()
    anchors: tuple = ()
    orders: tuple = ()

    def to_dict(self) -> dict:
        return {"keep": self.keep, "reason": self.reason, "blocks": self.blocks,
                "min_delta": self.min_delta, "median_relative": self.median_relative,
                "contribution_floor": self.contribution_floor,
                "calibration_evidence_ref": self.calibration_evidence_ref,
                "drift_bound": self.drift_bound, "anchor_drift": self.anchor_drift,
                "deltas": list(self.deltas), "relatives": list(self.relatives),
                "anchors": list(self.anchors), "orders": list(self.orders)}


def decide(pairs: Sequence[Pair], *, t0: T0Outcome, blocks_precommitted: int,
           drift_bound: float, contribution_floor: Optional[float] = None,
           calibration_evidence_ref: Optional[str] = None) -> AcceptDecision:
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
    if contribution_floor is None:
        # Compatibility for arithmetic/unit callers that exercise the original
        # rule directly.  The executing campaign always supplies its accepted,
        # cell-local calibration below.
        contribution_floor = float(drift_bound)
    if calibration_evidence_ref is None:
        calibration_evidence_ref = "direct arithmetic caller (no live authority)"
    if not isinstance(calibration_evidence_ref, str) or not calibration_evidence_ref.strip():
        raise ValueError("calibration_evidence_ref must be a non-empty string")
    if isinstance(contribution_floor, bool) or not isinstance(
            contribution_floor, (int, float)) or contribution_floor <= 0:
        raise ValueError("contribution_floor must be a positive fraction")

    ordered = sorted(items, key=lambda p: p.block_index)
    orders = tuple(p.order for p in ordered)
    deltas = tuple(p.delta for p in ordered)
    relatives = tuple(p.relative for p in ordered)
    anchors = tuple(p.anchor for p in ordered)
    min_delta = min(deltas)
    median_relative = float(median(relatives))
    moved = anchor_drift(ordered)
    common = {"blocks": len(items), "min_delta": min_delta,
              "median_relative": median_relative,
              "contribution_floor": float(contribution_floor),
              "calibration_evidence_ref": calibration_evidence_ref,
              "drift_bound": float(drift_bound),
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
    if median_relative <= contribution_floor:
        return AcceptDecision(
            keep=False,
            reason=(f"REVERT: median relative gain {median_relative:+.4%} does not exceed "
                    f"the predeclared contribution floor {contribution_floor:.4%} "
                    f"({calibration_evidence_ref}). Every pair favoured the candidate, "
                    "but a gain below the campaign's calibrated target is not a KEEP."),
            **common)
    return AcceptDecision(
        keep=True,
        reason=(f"KEEP: all {len(items)} pre-committed paired blocks favoured the candidate "
                f"(worst delta {min_delta:+.4f}) and the median relative gain "
                f"{median_relative:+.4%} exceeds the predeclared contribution floor "
                f"{contribution_floor:.4%} ({calibration_evidence_ref}); anchor movement "
                f"also stayed within {drift_bound:.4%} ({AA_EVIDENCE_REF})."),
        **common)


def screening_decision(pairs: Sequence[CandidateOnlyObservation], *,
                       blocks_precommitted: int) -> AcceptDecision:
    """Record a bounded discovery observation without acceptance authority.

    This deliberately does not call :func:`decide`: screening has no executed
    T0, no stabilization boundary, and fewer than calibration's rankable
    blocks.  It may report a directional median, but `keep` is structurally
    false so callers cannot confuse a cheap screen with a confirmation.
    """
    items = tuple(pairs)
    if len(items) != blocks_precommitted or not all(
            isinstance(p, CandidateOnlyObservation) for p in items):
        raise AcceptRuleMisuse(
            "screening requires exactly its precommitted candidate-only observations")
    ordered = tuple(sorted(items, key=lambda p: p.block_index))
    return AcceptDecision(
        keep=False, blocks=len(ordered),
        reason=("SCREENING_ONLY: bounded directional observation; no T0, no post-T0 "
                "stabilization, no calibration-rank authority, and hard-forbidden "
                "from KEEP/archive/promotion. A fresh confirmation campaign is required."),
        min_delta=min(p.delta for p in ordered),
        median_relative=float(median(tuple(p.relative for p in ordered))),
        deltas=tuple(p.delta for p in ordered), relatives=tuple(p.relative for p in ordered),
        anchors=tuple(p.baseline_reference for p in ordered), orders=())


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

#: CPU defaults to the one cell with accepted live calibration.  Decode and GPU
#: remain valid dry-run targets, but live ranking refuses until each has its own
#: calibration rather than borrowing the prefill result.
DEFAULT_RECIPE_BY_BACKEND = {
    BACKEND_CPU: HISTORICAL_CALIBRATED_RECIPE_ID,
    BACKEND_GPU: "t1b.llama_gpu.llama_bench_decode.v1",
}

#: Serving production, frozen. `worktree.resolve_anchor(expected_commit=...)`
#: turns "I believe production is at v9" into a checked precondition.
PRODUCTION_REPO = "/mnt/raid0/llm/llama.cpp"
PRODUCTION_BRANCH = "production-consolidated-v9"
PRODUCTION_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"

#: Reviewed measurement source.  The hardened instrument commit is a clean
#: descendant of the frozen v9 serving commit: candidate worktrees start here
#: so evaluator-only and correctness fixes are present without modifying
#: production.
#: Kernel proposals may change kernel sources but RVP-C6-1 requires all reward
#: translation units to remain byte-identical to this commit.
MEASUREMENT_REPO = "/mnt/raid0/llm/autokernel/worktrees/ak-final-q6k-20260813"
MEASUREMENT_BRANCH = "experimental-v9-autokernel-t0-final-q6k-20260813"
MEASUREMENT_COMMIT = "f744cc220e722d1bda93783959471d44f8e118b0"
MEASUREMENT_BUILD_ROOT = os.path.join(MEASUREMENT_REPO, "build-ak-t0-cpu-f744cc220")

# ``llama-cli`` now starts an embedded HTTP server and talks to it over a
# loopback socket. Candidate T0 is deliberately network-denied, so use the
# direct API completion executable for the behavioural probe instead. This is
# a measurement-tool identity, not a candidate choice.
T0_GENERATION_TOOL = "llama-completion"


def _validate_calibration_raw_measurement(
        raw: Mapping[str, Any], *, recipe_id: str,
        frame: Mapping[str, Any], expected_blocks: int, label: str) -> None:
    """Re-derive a raw control trace's declared fresh-pair aggregation.

    The calibration summary and typed authority describe derived statistics.
    This check independently binds those statistics to the physical block plan
    and invocation count that produced them, so changing a declaration to five
    pairs cannot make one-pair evidence look like an aggregated decode block.
    """
    frame_key, token_key, expected_pairs = \
        control_runner.calibration_frame_contract(frame, recipe_id=recipe_id)
    if raw.get("recipe_id") != recipe_id:
        raise ValueError(f"calibration {label} raw evidence names another recipe")
    for arm in ("candidate", "anchor"):
        receipt = raw.get(f"{arm}_receipt")
        params = receipt.get("params") if isinstance(receipt, Mapping) else None
        if not isinstance(params, Mapping) \
                or receipt.get("recipe_id") != recipe_id \
                or params.get(token_key) != frame.get(frame_key) \
                or params.get("reps") != frame.get("reps") \
                or params.get("ggml_iqk") != frame.get(f"{arm}_ggml_iqk"):
            raise ValueError(
                f"calibration {label} {arm} receipt is outside the declared frame")
    blocks = raw.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != expected_blocks:
        raise ValueError(
            f"calibration {label} evidence does not span its declared block count")
    expected_samples = expected_pairs * int(frame["reps"])
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping) or block.get("complete") is not True \
                or block.get("refusals") != []:
            raise ValueError(f"calibration {label} block {index} is incomplete")
        plan = block.get("plan")
        invocations = block.get("invocations")
        paired = block.get("paired_block")
        if not isinstance(plan, Mapping) \
                or plan.get("block_index") != index \
                or plan.get("pairs") != expected_pairs \
                or not isinstance(plan.get("arm_sequence"), list) \
                or len(plan["arm_sequence"]) != 2 * expected_pairs \
                or plan["arm_sequence"].count("anchor") != expected_pairs \
                or plan["arm_sequence"].count("candidate") != expected_pairs:
            raise ValueError(
                f"calibration {label} block {index} has the wrong fresh-pair plan")
        if not isinstance(invocations, list) or len(invocations) != 2 * expected_pairs:
            raise ValueError(
                f"calibration {label} block {index} has the wrong invocation count")
        samples_by_arm = {"anchor": [], "candidate": []}
        for position, invocation in enumerate(invocations):
            if not isinstance(invocation, Mapping) \
                    or invocation.get("position") != position \
                    or invocation.get("arm") != plan["arm_sequence"][position] \
                    or not isinstance(invocation.get("samples"), list) \
                    or len(invocation["samples"]) != frame["reps"]:
                raise ValueError(
                    f"calibration {label} block {index} invocation {position} "
                    "does not match its plan/repetition frame")
            samples_by_arm[invocation["arm"]].extend(invocation["samples"])
        if not isinstance(paired, list) or len(paired) != 9 or paired[0] != index \
                or paired[7] != samples_by_arm["anchor"] \
                or paired[8] != samples_by_arm["candidate"] \
                or len(paired[7]) != expected_samples \
                or len(paired[8]) != expected_samples:
            raise ValueError(
                f"calibration {label} block {index} does not preserve its raw "
                "per-arm aggregation material")


def load_calibration_bundle(path: os.PathLike[str] | str) -> LeanCalibration:
    """Load accepted live controls bound to the exact production/instrument pair.

    The v8 bundle is intentionally rejected on v9.  Calibration is local to an
    era, recipe and measurement instrument; a numerically valid old bundle is
    evidence, but it is not authority to rank a current candidate.
    """
    root = Path(path).resolve()

    def read(name: str) -> Mapping[str, Any]:
        try:
            value = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"calibration bundle {name}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"calibration bundle {name} must contain a JSON object")
        return value

    declaration = read("campaign_declaration.json")
    source = read("runtime-source-label.json")
    summary = read("summary.json")
    if declaration.get("schema") != "epyc.autokernel.live_control_campaign_declaration.v1":
        raise ValueError("calibration bundle declaration has the wrong schema")
    if source.get("schema") != "epyc.autokernel.runtime_source_label.v1":
        raise ValueError("calibration bundle runtime source label has the wrong schema")
    source_body = dict(source)
    source_sha = source_body.pop("source_sha256", None)
    if source_sha != schemas.content_hash(source_body):
        raise ValueError("calibration bundle runtime source label hash does not verify")
    if declaration.get("source_sha256") != source_sha:
        raise ValueError("calibration declaration is not bound to its runtime source label")
    if source.get("production_source_commit") != PRODUCTION_COMMIT:
        raise ValueError(
            "calibration production commit is stale: "
            f"{source.get('production_source_commit')!r} != {PRODUCTION_COMMIT!r}")
    if source.get("measurement_instrument_commit") != MEASUREMENT_COMMIT:
        raise ValueError(
            "calibration measurement instrument is stale or absent: "
            f"{source.get('measurement_instrument_commit')!r} != {MEASUREMENT_COMMIT!r}")
    if not source.get("binary_copy_exact"):
        raise ValueError("calibration measurement binary copy was not exact")
    if summary.get("production_source_commit") != PRODUCTION_COMMIT:
        raise ValueError("calibration summary names a different production commit")
    if summary.get("campaign_id") != declaration.get("campaign_id"):
        raise ValueError("calibration summary and declaration name different campaigns")
    if summary.get("state") != "controls_complete" or not summary.get("may_rank"):
        raise ValueError("calibration controls are not complete and rank-authorizing")
    if not summary.get("binary_copy_exact"):
        raise ValueError("calibration summary does not attest an exact binary copy")
    calibration = summary.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration summary has no calibration record")
    outputs = calibration.get("outputs")
    attempts = calibration.get("attempts")
    if not isinstance(outputs, Mapping) or not outputs.get("accepted"):
        raise ValueError("calibration outputs are absent or unaccepted")
    if not isinstance(attempts, list):
        raise ValueError("calibration attempts are absent")
    accepted_attempts = [a for a in attempts
                         if isinstance(a, Mapping) and a.get("accepted")]
    if len(accepted_attempts) != 1:
        raise ValueError("calibration must carry exactly one accepted solve attempt")
    mde = accepted_attempts[0].get("mde")
    if not isinstance(mde, Mapping) or not mde.get("found"):
        raise ValueError("accepted calibration has no solved MDE")
    recipe_id = declaration.get("recipe_id")
    values = {
        "contribution_floor": declaration.get("contribution_floor"),
        "b_min_blocks": outputs.get("b_min_blocks"),
        "max_blocks": declaration.get("max_blocks_per_candidate"),
        "noise_floor_phi": outputs.get("noise_floor_phi"),
        "mde": mde.get("value"),
    }
    if not isinstance(recipe_id, str) or recipe_id not in recipes.RECIPE_IDS:
        raise ValueError("calibration declaration names an unknown recipe")
    frame = declaration.get("calibration_frame")
    control_runner.calibration_frame_contract(frame, recipe_id=recipe_id)
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"calibration {name} must be positive")
    if int(values["b_min_blocks"]) > int(values["max_blocks"]):
        raise ValueError("calibration B_min exceeds the declared candidate ceiling")
    if float(values["mde"]) > float(values["contribution_floor"]):
        raise ValueError("calibration MDE exceeds the declared contribution floor")
    anchor_motion = summary.get("anchor_motion")
    declared_motion_blocks = declaration.get("anchor_motion_window_blocks")
    declared_settling = declaration.get("anchor_motion_settling")
    declared_between_leg_policy = declaration.get("between_leg_policy")
    amendment_path = root / "resume_amendment.json"
    if amendment_path.is_file():
        amendment = read("resume_amendment.json")
        amendment_body = dict(amendment)
        amendment_sha = amendment_body.pop("amendment_sha256", None)
        if amendment_sha != schemas.content_hash(amendment_body) \
                or amendment.get("schema") \
                != "epyc.autokernel.live_control_resume_amendment.v1" \
                or amendment.get("campaign_id") != declaration.get("campaign_id") \
                or amendment.get("original_declaration_sha256") \
                != schemas.content_hash(declaration) \
                or amendment.get("replaced_field") != "anchor_motion_settling" \
                or amendment.get("original_value") != declared_settling:
            raise ValueError("calibration resume amendment does not verify")
        declared_settling = amendment.get("replacement_value")
        declared_between_leg_policy = amendment.get("added_between_leg_policy")
    if not isinstance(anchor_motion, Mapping):
        raise ValueError(
            "calibration bundle has no fresh campaign-length anchor-motion authority")
    if anchor_motion.get("schema") != "epyc.autokernel.anchor_motion_window.v1":
        raise ValueError("calibration anchor-motion authority has the wrong schema")
    if (isinstance(declared_motion_blocks, bool)
            or not isinstance(declared_motion_blocks, int)
            or declared_motion_blocks < 2):
        raise ValueError("calibration declaration has no valid anchor-motion window length")
    if not isinstance(declared_settling, Mapping):
        raise ValueError("calibration declaration has no anchor-motion settling contract")
    if anchor_motion.get("window_blocks") != declared_motion_blocks:
        raise ValueError("calibration anchor-motion window differs from its declaration")
    if anchor_motion.get("settling") != declared_settling:
        raise ValueError("calibration anchor-motion settling receipt differs from declaration")
    settling_ref = anchor_motion.get("settling_receipt_ref")
    expected_settling_path = (root / "anchor_motion_settling.json").resolve()
    if settling_ref != str(expected_settling_path):
        raise ValueError("calibration anchor-motion settling receipt has the wrong path")
    try:
        settling_receipt = json.loads(expected_settling_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"calibration anchor-motion settling receipt: {exc}") from exc
    if not isinstance(settling_receipt, Mapping):
        raise ValueError("calibration anchor-motion settling receipt must be an object")
    unsigned_settling = dict(settling_receipt)
    settling_sha = unsigned_settling.pop("receipt_sha256", None)
    if settling_sha != schemas.content_hash(unsigned_settling) \
            or anchor_motion.get("settling_receipt_sha256") != settling_sha \
            or settling_receipt.get("settling") != declared_settling:
        raise ValueError("calibration anchor-motion settling receipt does not verify")
    settling_samples = settling_receipt.get("samples")
    expected_samples = declared_settling.get("required_samples")
    transition_v2 = declared_settling.get("schema") \
        == "epyc.autokernel.anchor_motion_transition.v2"
    if not isinstance(settling_samples, list) or len(settling_samples) != expected_samples:
        raise ValueError("calibration anchor-motion transition has the wrong sample count")
    if transition_v2:
        if not isinstance(declared_between_leg_policy, Mapping):
            raise ValueError("calibration transition has no between-leg policy")
        for sample in settling_samples:
            if not isinstance(sample, Mapping):
                raise ValueError("calibration transition sample is not an object")
            body = dict(sample)
            observation_sha = body.pop("observation_sha256", None)
            witness = sample.get("inference_witness")
            if observation_sha != schemas.content_hash(body) \
                    or sample.get("policy") != declared_between_leg_policy \
                    or sample.get("ordinary_load", {}).get("disposition") \
                    != "recorded_as_noise_not_a_gate" \
                    or sample.get("claim_attestation", {}).get("outcome") \
                    != schemas.PASS \
                    or not isinstance(witness, Mapping) \
                    or witness.get("competing") is not False \
                    or sample.get("inference_witness_error") is not None:
                raise ValueError(
                    "calibration transition lacks a clean claim/inference witness")
    elif any(not isinstance(sample, Mapping)
             or sample.get("load", {}).get("outcome") != schemas.PASS
             or sample.get("claim_attestation", {}).get("outcome") != schemas.PASS
             for sample in settling_samples):
        # Historical bundles retain their original quiet-settling semantics.
        raise ValueError("calibration anchor-motion settling receipt has no all-PASS samples")
    raw_ref = anchor_motion.get("raw_ref")
    label = anchor_motion.get("label")
    if not isinstance(raw_ref, str) or not isinstance(label, str) or not label:
        raise ValueError("calibration anchor-motion authority lacks a raw evidence reference")
    raw_path = (root / "raw" / f"{label}.json").resolve()
    if raw_path != (root / "raw" / f"{label}.json") or raw_ref != str(raw_path):
        raise ValueError("calibration anchor-motion raw reference escapes or differs from bundle")
    try:
        raw_motion = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"calibration anchor-motion raw evidence: {exc}") from exc
    if not isinstance(raw_motion, Mapping):
        raise ValueError("calibration anchor-motion raw evidence must be an object")
    if anchor_motion.get("raw_sha256") != schemas.content_hash(raw_motion):
        raise ValueError("calibration anchor-motion raw evidence hash does not verify")
    if raw_motion.get("recipe_id") != recipe_id:
        raise ValueError("calibration anchor-motion raw evidence names another recipe")
    candidate_receipt = raw_motion.get("candidate_receipt")
    anchor_receipt = raw_motion.get("anchor_receipt")
    if not isinstance(frame, Mapping) or not isinstance(candidate_receipt, Mapping) \
            or not isinstance(anchor_receipt, Mapping):
        raise ValueError("calibration anchor-motion evidence lacks the declared A/A frame")
    raw_blocks = raw_motion.get("blocks")
    _validate_calibration_raw_measurement(
        raw_motion, recipe_id=recipe_id, frame=frame,
        expected_blocks=declared_motion_blocks, label=label)
    assert isinstance(raw_blocks, list)
    anchor_medians = []
    for index, block in enumerate(raw_blocks):
        paired = block["paired_block"]
        anchor_medians.append(float(median(paired[7])))
    measured_bound = drift_bound_from(anchor_medians)
    bound = anchor_motion.get("bound")
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) or bound <= 0 \
            or not isclose(float(bound), measured_bound, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("calibration anchor-motion bound does not re-derive from raw evidence")
    authority = None
    # Legacy/synthetic loader fixtures intentionally carry only the three files
    # above. They remain valid arithmetic fixtures, but cannot drive the live
    # prospective writer. A real executing HostOps run requires these files and
    # refuses later at the request boundary if they are absent.
    if (root / "calibration.json").is_file() or (root / "control_sweep.json").is_file():
        authority = control_runner.load_live_evaluation_authority(root)
        if authority.campaign_controls.contribution_floor != float(values["contribution_floor"]) \
                or authority.calibration.b_min_blocks != int(values["b_min_blocks"]) \
                or authority.campaign_controls.max_blocks_per_candidate != int(values["max_blocks"]) \
                or authority.calibration.noise_floor_phi != float(values["noise_floor_phi"]) \
                or authority.mde != float(values["mde"]):
            raise ValueError("calibration summary disagrees with live evaluation authority")
        for raw_label, count_key in (
                ("aa_calibration", "calibration_blocks"),
                ("neutral_calibration", "neutral_blocks")):
            count = declaration.get(count_key)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(
                    f"calibration declaration has no valid {count_key}")
            raw_control = read(f"raw/{raw_label}.json")
            _validate_calibration_raw_measurement(
                raw_control, recipe_id=recipe_id, frame=frame,
                expected_blocks=count, label=raw_label)
    return LeanCalibration(
        recipe_id=recipe_id,
        contribution_floor=float(values["contribution_floor"]),
        b_min_blocks=int(values["b_min_blocks"]),
        max_blocks=int(values["max_blocks"]),
        noise_floor_phi=float(values["noise_floor_phi"]),
        mde=float(values["mde"]),
        production_commit=PRODUCTION_COMMIT,
        measurement_commit=MEASUREMENT_COMMIT,
        evidence_ref=str(root),
        evaluation_authority=authority,
        anchor_motion_bound=float(bound),
        anchor_motion_window_blocks=declared_motion_blocks,
        anchor_motion_evidence_ref=raw_ref,
    )

RANKED_UNITS_SCHEMA = "epyc.autokernel.ranked-units.v1"
RANKED_UNIT_NORMAL = "normal"
RANKED_UNIT_ANTI_SHORT_CIRCUIT = "anti_short_circuit"
RANKED_UNIT_KINDS = (RANKED_UNIT_NORMAL, RANKED_UNIT_ANTI_SHORT_CIRCUIT)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


@dataclass(frozen=True)
class RankedUnitSpec:
    """One real recipe variant that receives blocks in the ranked stream."""

    unit_id: str
    kind: str
    params: Mapping[str, Any]
    physical_envelope: physical_bounds.PhysicalEnvelope

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise ValueError("ranked unit_id must be a non-empty string")
        if self.kind not in RANKED_UNIT_KINDS:
            raise ValueError(
                f"ranked unit kind {self.kind!r} must be one of {list(RANKED_UNIT_KINDS)}")
        if not isinstance(self.params, Mapping):
            raise TypeError("ranked unit params must be a mapping")
        object.__setattr__(self, "params", json.loads(schemas.canonical_json(self.params)))
        if not isinstance(self.physical_envelope, physical_bounds.PhysicalEnvelope):
            raise TypeError("ranked unit physical_envelope must be a PhysicalEnvelope")
        if self.physical_envelope.shape_id != self.unit_id:
            raise ValueError(
                f"ranked unit {self.unit_id!r} carries physical shape "
                f"{self.physical_envelope.shape_id!r}; they must be identical")

    def to_dict(self) -> dict:
        return {"unit_id": self.unit_id, "kind": self.kind,
                "params": dict(self.params),
                "physical_envelope": self.physical_envelope.to_dict()}


def ranked_units_from_mapping(payload: Mapping[str, Any]) -> tuple[RankedUnitSpec, ...]:
    """Parse the strict C6-10 ranked-unit manifest without executing anything."""
    if not isinstance(payload, Mapping):
        raise ValueError("ranked-unit manifest must be a JSON object")
    unknown = sorted(set(payload) - {"schema", "units"})
    if unknown:
        raise ValueError(f"ranked-unit manifest has unknown fields {unknown}")
    if payload.get("schema") != RANKED_UNITS_SCHEMA:
        raise ValueError(f"ranked-unit manifest schema must be {RANKED_UNITS_SCHEMA!r}")
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("ranked-unit manifest units must be a non-empty list")
    parsed = []
    required = {"unit_id", "kind", "params", "physical_envelope"}
    for index, item in enumerate(units):
        if not isinstance(item, Mapping):
            raise ValueError(f"ranked-unit manifest units[{index}] must be an object")
        missing = sorted(required - set(item))
        extra = sorted(set(item) - required)
        if missing or extra:
            raise ValueError(
                f"ranked-unit manifest units[{index}] missing={missing}, extra={extra}")
        parsed.append(RankedUnitSpec(
            unit_id=item["unit_id"], kind=item["kind"], params=item["params"],
            physical_envelope=physical_bounds.PhysicalEnvelope.from_mapping(
                item["physical_envelope"])))
    return tuple(parsed)


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
    #: Validated current-schema proposal record. Optional for composition-only legacy dry
    #: runs; mandatory on the executing CLI before any claim or mutation.
    proposal: Optional[Mapping[str, Any]] = None
    #: Immutable embedded source artifact, loaded completely before any claim.
    source_patch: Optional[source_candidate.SourcePatchManifest] = None
    #: Immutable raw correctness archive for a source candidate. It is loaded
    #: before the claim and identity-bound after the candidate build exists.
    source_prerequisite_package: Optional[
        source_prerequisite_package.SourcePrerequisitePackage] = None
    #: Predeclared execute-in-this-window producer plan. Mutually exclusive
    #: with an archive: fresh and resume are two modes, never two authorities.
    fresh_source_prerequisite_plan: Optional[
        source_prerequisite_producer.FreshSourcePrerequisitePlan] = None
    #: Prospective observe-only diagnostic plan.  It is fixed before a claim and
    #: supplies no selection authority; the campaign only journals its diagnostics
    #: and reduces the immutable outcome functions after measurement.
    least_commitment_plan: Optional[least_commitment_capture.CapturePlan] = None
    #: Shared identity for a matched intervention/control experiment. Unlike
    #: campaign_id/candidate_id, this value is deliberately identical across
    #: both completed campaigns and owns every randomized ordering/holdout seed.
    matched_experiment_id: Optional[str] = None
    #: Accepted, identity-bound live calibration.  Supplied by the CLI bundle
    #: loader; absent means composition-only and carries no ranking authority.
    calibration: Optional[LeanCalibration] = None
    #: Predeclared RVP-C6-4 envelope for this exact measurement-material unit.
    #: Optional only for a composition-only dry run; the CLI refuses live work
    #: without it before a claim, mutation, build, or subprocess exists.
    physical_envelope: Optional[physical_bounds.PhysicalEnvelope] = None
    #: Optional multi-unit C6-10 ranked set. When present it replaces the single
    #: physical_envelope and must include both a normal control and at least one
    #: actual anti-short-circuit recipe variant.
    ranked_units: tuple[RankedUnitSpec, ...] = ()
    #: The `hypotheses.ClaimAuthorization` this campaign's claim will be spent
    #: through, or `None` for an EXPLORATORY campaign. It is on the SPEC and not
    #: on the ops object for the same reason `blocks` is: it must be fixed before
    #: the claim, it governs what the claim's own receipt says, and it has to
    #: reach the durable record — `to_dict()` carries it, so "what did we spend
    #: the card on" and "what would have refuted it" are one lookup.
    authorization: Optional[Any] = None
    #: Discovery-only measurement: bounded, explicitly non-promotable and
    #: forbidden from producing candidate/archive authority.
    screening_only: bool = False
    screening_baseline: Optional[screening_baseline.BaselineBank] = None
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
        if not isinstance(self.screening_only, bool):
            raise TypeError("screening_only must be bool")
        if self.screening_only:
            if self.blocks > 3:
                raise ValueError("screening-only runs are capped at 3 paired blocks")
            if self.least_commitment_plan is not None or self.matched_experiment_id is not None:
                raise ValueError("screening-only runs cannot carry archive/held-out bindings")
            if not isinstance(self.screening_baseline, screening_baseline.BaselineBank):
                raise ValueError("screening-only runs require an immutable screening baseline bank")
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
        if self.source_patch is not None and not isinstance(
                self.source_patch, source_candidate.SourcePatchManifest):
            raise TypeError("source_patch must be a SourcePatchManifest or None")
        if self.source_prerequisite_package is not None and not isinstance(
                self.source_prerequisite_package,
                source_prerequisite_package.SourcePrerequisitePackage):
            raise TypeError(
                "source_prerequisite_package must be a SourcePrerequisitePackage or None")
        if self.fresh_source_prerequisite_plan is not None and not isinstance(
                self.fresh_source_prerequisite_plan,
                source_prerequisite_producer.FreshSourcePrerequisitePlan):
            raise TypeError("fresh_source_prerequisite_plan must be a "
                            "FreshSourcePrerequisitePlan or None")
        if (self.source_prerequisite_package is not None
                and self.fresh_source_prerequisite_plan is not None):
            raise ValueError("archive/resume and fresh source prerequisite modes are "
                             "mutually exclusive")
        if self.proposal is None and (self.source_prerequisite_package is not None
                                      or self.fresh_source_prerequisite_plan is not None):
            raise ValueError("source prerequisites require a source proposal")
        if self.least_commitment_plan is not None and not isinstance(
                self.least_commitment_plan, least_commitment_capture.CapturePlan):
            raise TypeError("least_commitment_plan must be a CapturePlan or None")
        if self.least_commitment_plan is not None and self.proposal is None:
            raise ValueError("least_commitment_plan requires a proposal")
        if self.least_commitment_plan is not None:
            if (not isinstance(self.matched_experiment_id, str)
                    or not self.matched_experiment_id.startswith("akm-")
                    or "\0" in self.matched_experiment_id):
                raise ValueError(
                    "least_commitment_plan requires matched_experiment_id starting "
                    "with 'akm-'")
            if self.least_commitment_plan.raw.get("matched_experiment_id") \
                    != self.matched_experiment_id:
                raise ValueError(
                    "least-commitment plan matched_experiment_id differs from campaign")
        elif self.matched_experiment_id is not None:
            raise ValueError(
                "matched_experiment_id has no meaning without least_commitment_plan")
        if self.proposal is not None:
            proposal = json.loads(schemas.canonical_json(self.proposal))
            violations = schemas.validate_proposal(proposal)
            if violations:
                raise ValueError("proposal manifest is invalid: " + "; ".join(violations))
            if proposal["campaign_id"] != self.campaign_id:
                raise ValueError(
                    f"proposal campaign_id {proposal['campaign_id']!r} does not match "
                    f"campaign {self.campaign_id!r}"
                )
            provider_reference = proposal["provider_reference"]
            try:
                provider.IsolatedProviderPrefix.create(
                    provider_reference["isolation_root"])
            except provider.ProviderIsolationError as exc:
                raise ValueError(f"proposal provider isolation is invalid: {exc}") from exc
            if provider_reference["target_backend"] != self.backend:
                raise ValueError(
                    f"proposal provider target {provider_reference['target_backend']!r} "
                    f"does not match campaign backend {self.backend!r}"
                )
            # The provider identity is part of the measurement material.  A
            # proposal from another kernel/instrument era must be rejected at
            # composition time, before it can reach the journal, claim, or
            # build.  Calibration independently binds this same commit in
            # ``load_calibration_bundle``; accepting either side alone would
            # make the resulting evidence impossible to interpret.
            if (provider_reference["source_mode"] == "source"
                    and provider_reference["source_commit"] != MEASUREMENT_COMMIT):
                raise ValueError(
                    "proposal provider source commit is not the campaign measurement "
                    f"instrument: {provider_reference['source_commit']!r} != "
                    f"{MEASUREMENT_COMMIT!r}")
            object.__setattr__(self, "proposal", proposal)
            self._validate_arm_parameter_surface(proposal)
            if proposal["change_class"] == "parameter" and self.source_patch is not None:
                raise ValueError("parameter campaigns may not carry a source patch")
            if proposal["change_class"] == "parameter" \
                    and self.source_prerequisite_package is not None:
                raise ValueError("parameter campaigns may not carry source prerequisites")
            if proposal["change_class"] == "parameter" \
                    and self.fresh_source_prerequisite_plan is not None:
                raise ValueError("parameter campaigns may not carry a fresh source plan")
            if proposal["change_class"] != "parameter":
                if self.source_patch is not None:
                    self.source_patch.bind(
                        proposal=proposal, campaign_id=self.campaign_id,
                        candidate_id=self.candidate_id,
                        production_base_commit=PRODUCTION_COMMIT,
                        instrument_commit=MEASUREMENT_COMMIT)
                if self.source_prerequisite_package is not None:
                    self.source_prerequisite_package.bind_campaign(
                        proposal=proposal, campaign_id=self.campaign_id,
                        candidate_id=self.candidate_id)
                if self.fresh_source_prerequisite_plan is not None:
                    self.fresh_source_prerequisite_plan.bind_campaign(
                        proposal=proposal, campaign_id=self.campaign_id,
                        candidate_id=self.candidate_id)
        if self.calibration is not None and not isinstance(
                self.calibration, LeanCalibration):
            raise TypeError("calibration must be a LeanCalibration or None")
        if self.physical_envelope is not None and not isinstance(
                self.physical_envelope, physical_bounds.PhysicalEnvelope):
            raise TypeError("physical_envelope must be a PhysicalEnvelope or None")
        if not isinstance(self.ranked_units, tuple):
            raise TypeError("ranked_units must be a tuple")
        if any(not isinstance(unit, RankedUnitSpec) for unit in self.ranked_units):
            raise TypeError("ranked_units must contain RankedUnitSpec values")
        if self.ranked_units and self.physical_envelope is not None:
            raise ValueError(
                "physical_envelope and ranked_units are mutually exclusive; every "
                "ranked unit carries its own exact envelope")
        object.__setattr__(self, "t0_ops", require_op_suite_covers_moe_dispatch(self.t0_ops))
        if self.recipe_id is None:
            object.__setattr__(self, "recipe_id", DEFAULT_RECIPE_BY_BACKEND[self.backend])
        spec = recipes.get_recipe(self.recipe_id)
        if spec.backend != self.backend:
            raise ValueError(f"recipe {self.recipe_id!r} is a {spec.backend!r} recipe but "
                             f"the campaign declares backend {self.backend!r}")
        if self.calibration is not None:
            if self.calibration.recipe_id != self.recipe_id:
                raise ValueError(
                    f"calibration recipe {self.calibration.recipe_id!r} does not match "
                    f"campaign recipe {self.recipe_id!r}")
            if self.calibration.production_commit != PRODUCTION_COMMIT \
                    or self.calibration.measurement_commit != MEASUREMENT_COMMIT:
                raise ValueError("calibration identities do not match the live anchors")
            # A current bundle's A/A dispersion and its anchor-motion control
            # answer different questions.  The former sets power; the latter
            # says whether this exact-length T1 window is attributable at all.
            # Do not silently fall back to the old exploratory four-run bound.
            motion = self.calibration.anchor_motion_bound
            window = self.calibration.anchor_motion_window_blocks
            if (motion is None) != (window is None):
                raise ValueError(
                    "calibration anchor-motion bound and window must be present together")
            if motion is not None:
                if isinstance(motion, bool) or not isinstance(motion, (int, float)) \
                        or motion <= 0:
                    raise ValueError(
                        "live calibration lacks a positive fresh anchor-motion bound")
                if isinstance(window, bool) or not isinstance(window, int) or window < 2:
                    raise ValueError(
                        "live calibration lacks a valid anchor-motion window length")
                if self.blocks != window:
                    raise ValueError(
                        f"campaign blocks={self.blocks} differs from the calibration's "
                        f"anchor-motion window of {window}; recalibrate this exact T1 length")
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
        if self.screening_only:
            assert self.screening_baseline is not None
            self.screening_baseline.admit({
                "recipe_id": self.recipe_id, "backend": self.backend,
                "model_sha256": (storage.hash_file(self.model)
                                 if self.model and Path(self.model).is_file() else None),
                "instrument_commit": MEASUREMENT_COMMIT,
                "production_commit": PRODUCTION_COMMIT,
                "boot_sha256": schemas.content_hash({
                    "boot_id": Path("/proc/sys/kernel/random/boot_id")
                    .read_text(encoding="utf-8").strip()}),
                "anchor_ggml_iqk": self.anchor_param_overrides.get("ggml_iqk"),
                "reps": self.reps,
                "n_prompt": (self.n_prompt if self.recipe.phase == "prefill" else 0),
                "n_gen": self.n_gen,
            })
        if self.calibration is not None \
                and self.calibration.evaluation_authority is not None:
            frame = self.calibration.evaluation_authority.calibration_frame
            # Legacy/synthetic authorities intentionally lack this prospective
            # binding.  A loaded live bundle cannot: its loader refuses a
            # missing or malformed frame.  The comparison binds its A/A
            # dispersion to the exact baseline arm rather than merely to a
            # similarly named recipe.
            if frame is not None:
                frame_key, token_key, _pairs = \
                    control_runner.calibration_frame_contract(
                        frame, recipe_id=self.recipe_id)
                actual = {
                    "recipe_id": self.recipe_id,
                    frame_key: getattr(self, token_key),
                    "reps": self.reps,
                    "anchor_ggml_iqk": self.anchor_param_overrides.get(
                        "ggml_iqk", recipes.CANONICAL_OMP_ENV["GGML_IQK"]),
                }
                expected = {
                    "recipe_id": frame["recipe_id"],
                    frame_key: frame[frame_key],
                    "reps": frame["reps"],
                    "anchor_ggml_iqk": frame["anchor_ggml_iqk"],
                }
                if actual != expected:
                    raise ValueError(
                        "calibration_frame does not match this campaign's exact "
                        f"anchor recipe: calibrated={expected}, requested={actual}")
        if self.physical_envelope is not None \
                and self.physical_envelope.shape_id != self.measurement_unit_id:
            raise ValueError(
                f"physical envelope shape_id {self.physical_envelope.shape_id!r} does "
                f"not match campaign unit {self.measurement_unit_id!r}")
        if self.physical_envelope is not None:
            self._check_physical_envelope_frame(
                self.measurement_unit_id, self.physical_envelope, self.bench_params)
        if self.ranked_units:
            unit_ids = tuple(unit.unit_id for unit in self.ranked_units)
            if len(set(unit_ids)) != len(unit_ids):
                raise ValueError("ranked_units must have unique unit_id values")
            kinds = {unit.kind for unit in self.ranked_units}
            if kinds != set(RANKED_UNIT_KINDS):
                raise ValueError(
                    "ranked_units must include at least one normal control and at least "
                    "one anti_short_circuit unit")
            if self.blocks < len(self.ranked_units):
                raise ValueError(
                    f"blocks={self.blocks} cannot rank all {len(self.ranked_units)} units")
            declared_params = set(self.recipe.param_map)
            normal_frames = {
                schemas.canonical_json(unit.params) for unit in self.ranked_units
                if unit.kind == RANKED_UNIT_NORMAL}
            for unit in self.ranked_units:
                unknown = sorted(set(unit.params) - declared_params)
                if unknown:
                    raise ValueError(
                        f"ranked unit {unit.unit_id!r} contains recipe-unknown params "
                        f"{unknown}")
                render_bench_commands(
                    self, params={**self.bench_params, **dict(unit.params)})
                self._check_physical_envelope_frame(
                    unit.unit_id, unit.physical_envelope,
                    {**self.bench_params, **dict(unit.params)})
                if (unit.kind == RANKED_UNIT_ANTI_SHORT_CIRCUIT
                        and schemas.canonical_json(unit.params) in normal_frames):
                    raise ValueError(
                        f"anti-short-circuit unit {unit.unit_id!r} has the same recipe "
                        "params as a normal control; relabelling one command does not "
                        "price the hard case")
        if self.least_commitment_plan is not None:
            least_commitment_capture.bind_executed_factor_frame(
                self.least_commitment_plan,
                matched_experiment_id=str(self.matched_experiment_id),
                factors=self.matched_factor_frame)

    def _check_physical_envelope_frame(
            self, unit_id: str, envelope: physical_bounds.PhysicalEnvelope,
            params: Mapping[str, Any]) -> None:
        if envelope.delivered_unit != "token":
            raise ValueError(
                f"physical envelope for {unit_id!r} is expressed in "
                f"{envelope.delivered_unit!r}/s, but the registered llama-bench "
                "campaign recipes emit token/s")
        derived = physical_bounds.measurement_frame_sha256(self.recipe_id, params)
        if envelope.measurement_frame_sha256 != derived:
            raise ValueError(
                f"physical envelope for {unit_id!r} is bound to measurement frame "
                f"{envelope.measurement_frame_sha256}, but the exact recipe, model, and "
                f"parameters derive {derived}")

    def _validate_arm_parameter_surface(self, proposal: Mapping[str, Any]) -> None:
        """Validate the one recipe-declared arm-local parameter comparison.

        ``MicrobenchPlan`` already carries candidate/anchor overrides and limits
        them to ``ggml_iqk``. The campaign must project the proposal into that
        existing seam rather than constructing both arms from the same mapping.
        """
        if proposal["change_class"] != "parameter":
            return
        surface = proposal["change"]["parameter_surface"]
        if set(surface) != {"candidate", "anchor"}:
            raise ValueError(
                "a parameter proposal must declare parameter_surface with exactly "
                "candidate and anchor mappings")
        for arm in ("candidate", "anchor"):
            values = surface[arm]
            if not isinstance(values, Mapping):
                raise ValueError(f"parameter_surface.{arm} must be a mapping")
            unknown = sorted(set(values) - {"ggml_iqk"})
            if unknown:
                raise ValueError(
                    f"parameter_surface.{arm} contains {unknown}; the recipe registry "
                    "licenses only the GGML_IQK arm-local variant")
            if set(values) != {"ggml_iqk"} or values["ggml_iqk"] not in ("0", "1"):
                raise ValueError(
                    f"parameter_surface.{arm} must declare ggml_iqk as '0' or '1'")
        if surface["candidate"] == surface["anchor"]:
            if self.least_commitment_plan is not None \
                    and self.least_commitment_plan.role == "control" \
                    and surface["candidate"]["ggml_iqk"] == "0":
                return
            raise ValueError(
                "a parameter proposal gives candidate and anchor the same GGML_IQK "
                "value; it declares no comparison. Only a hash-bound least-commitment "
                "role=control plan may "
                "declare the production-setting A/A control")

    # -- derived -----------------------------------------------------------

    @property
    def recipe(self) -> Any:
        return recipes.get_recipe(self.recipe_id)

    @property
    def metric(self) -> str:
        return self.recipe.metric

    @property
    def drift_bound(self) -> float:
        if self.calibration is not None \
                and self.calibration.anchor_motion_bound is not None:
            return float(self.calibration.anchor_motion_bound)
        # Direct arithmetic and historical regression fixtures retain their
        # fixed fixture bound. Executing campaigns must instead use the fresh
        # bundle authority validated in __post_init__.
        return drift_bound_for_metric(self.metric)

    @property
    def contribution_floor(self) -> Optional[float]:
        return None if self.calibration is None else self.calibration.contribution_floor

    @property
    def hypothesis_id(self) -> Optional[str]:
        """The question this campaign is bound to, or None. Read off the TOKEN.

        Never a second field. A campaign carrying an id beside an authorization
        could have the two disagree, and the one a reader would trust is the
        string rather than the capability.
        """
        return None if self.authorization is None else self.authorization.hypothesis_id

    @property
    def proposal_id(self) -> Optional[str]:
        return None if self.proposal is None else self.proposal["proposal_id"]

    @property
    def measurement_unit_id(self) -> str:
        return f"{self.recipe_id}:{self.model or 'declared-model'}"

    @property
    def ranked_unit_ids(self) -> tuple[str, ...]:
        return (tuple(unit.unit_id for unit in self.ranked_units)
                if self.ranked_units else (self.measurement_unit_id,))

    @property
    def ranked_unit_param_overrides(self) -> dict:
        return ({unit.unit_id: dict(unit.params) for unit in self.ranked_units}
                if self.ranked_units else {})

    @property
    def anti_short_circuit_units(self) -> tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.ranked_units
                     if unit.kind == RANKED_UNIT_ANTI_SHORT_CIRCUIT)

    @property
    def candidate_param_overrides(self) -> dict:
        if self.proposal is None or self.proposal["change_class"] != "parameter":
            return {}
        return dict(self.proposal["change"]["parameter_surface"]["candidate"])

    @property
    def anchor_param_overrides(self) -> dict:
        if self.proposal is None or self.proposal["change_class"] != "parameter":
            return {}
        return dict(self.proposal["change"]["parameter_surface"]["anchor"])

    def params_for_arm(self, arm: str, *, params: Optional[Mapping[str, Any]] = None
                       ) -> dict:
        if arm not in ("candidate", "anchor"):
            raise ValueError(f"arm must be candidate or anchor, got {arm!r}")
        merged = dict(self.bench_params if params is None else params)
        merged.update(self.candidate_param_overrides if arm == "candidate"
                      else self.anchor_param_overrides)
        return merged

    def t0_parameter_env_for_arm(self, arm: str) -> tuple:
        """Project the validated recipe parameter into T0's closed env registry."""
        if arm not in ("candidate", "anchor"):
            raise ValueError(f"arm must be candidate or anchor, got {arm!r}")
        overrides = (self.candidate_param_overrides if arm == "candidate"
                     else self.anchor_param_overrides)
        return (() if not overrides else (("GGML_IQK", overrides["ggml_iqk"]),))

    @property
    def physical_envelopes(self) -> dict:
        if self.ranked_units:
            return {unit.unit_id: unit.physical_envelope for unit in self.ranked_units}
        return ({} if self.physical_envelope is None
                else {self.measurement_unit_id: self.physical_envelope})

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
        return self.bench_params_for(self.matched_experiment_id)

    @property
    def fresh_pairs_per_block(self) -> int:
        """Physical fresh-process pairs contributing to one statistical block."""
        authority = (None if self.calibration is None else
                     self.calibration.evaluation_authority)
        frame = None if authority is None else authority.calibration_frame
        if frame is None:
            return 1
        _frame_key, _token_key, pairs = \
            control_runner.calibration_frame_contract(
                frame, recipe_id=self.recipe_id)
        return pairs

    def bench_params_for(self, matched_experiment_id: Optional[str]) -> dict:
        """Derive benchmark parameters for a legacy or declared matched frame."""
        seed_material = (
            f"matched\0{matched_experiment_id}\0{self.recipe_id}"
            if matched_experiment_id is not None else
            f"{self.campaign_id}\0{self.candidate_id}\0{self.recipe_id}")
        autokernel_seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        params: dict = {"reps": self.reps,
                        "autokernel_seed": autokernel_seed or 1}
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

    @property
    def suite_seed(self) -> int:
        """Deterministic T0 tensor seed fixed by campaign or matched identity."""
        return self.suite_seed_for(self.matched_experiment_id)

    def suite_seed_for(self, matched_experiment_id: Optional[str]) -> int:
        material = (
            f"t0-suite\0matched\0{matched_experiment_id}\0{self.recipe_id}"
            if matched_experiment_id is not None else
            f"t0-suite\0{self.campaign_id}\0{self.candidate_id}\0{self.recipe_id}")
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")

    @property
    def schedule_seed(self) -> str:
        return self.schedule_seed_for(self.matched_experiment_id)

    def schedule_seed_for(self, matched_experiment_id: Optional[str]) -> str:
        return (
            f"ak-schedule/v1:matched:{matched_experiment_id}:{self.recipe_id}"
            if matched_experiment_id is not None else
            f"{self.campaign_id}/{self.created_at}")

    @property
    def holdout_selection_seed(self) -> str:
        return self.holdout_selection_seed_for(self.matched_experiment_id)

    def holdout_selection_seed_for(
            self, matched_experiment_id: Optional[str]) -> str:
        return (
            f"ak-holdout/v1:matched:{matched_experiment_id}:"
            f"{self.suite_seed_for(matched_experiment_id)}"
            if matched_experiment_id is not None else
            f"{self.campaign_id}/{self.suite_seed}")

    @property
    def matched_factor_frame(self) -> dict:
        """Derive every execution axis that must match except the intervention."""
        if self.least_commitment_plan is None or self.matched_experiment_id is None:
            raise ValueError("matched factor frame requires a least-commitment plan")
        return self.matched_factor_frame_for(self.matched_experiment_id)

    def matched_factor_frame_for(
            self, matched_experiment_id: str, *,
            physical_envelope: Optional[physical_bounds.PhysicalEnvelope] = None,
    ) -> dict:
        """Derive a plan-generation frame without weakening live admission."""
        if (not isinstance(matched_experiment_id, str)
                or not matched_experiment_id.startswith("akm-")
                or "\0" in matched_experiment_id):
            raise ValueError("matched factor frame requires an akm- identity")
        if not self.model:
            raise ValueError("matched factor frame requires the measured model")
        model_path = storage.assert_not_scratch(self.model, what="matched model")
        if not os.path.isfile(model_path):
            raise ValueError(f"matched model is not a file: {model_path}")
        calibration = self.calibration
        if calibration is None:
            raise ValueError("matched factor frame requires accepted calibration")
        selected_envelope = physical_envelope or self.physical_envelope
        if self.ranked_units and physical_envelope is not None:
            raise ValueError(
                "a physical-envelope override cannot replace ranked units")
        if self.ranked_units:
            envelope: Any = [unit.to_dict() for unit in self.ranked_units]
        elif selected_envelope is not None:
            envelope = selected_envelope.to_dict()
        else:
            raise ValueError("matched factor frame requires a physical envelope")
        provider_reference = (
            dict(self.proposal["provider_reference"])
            if self.proposal is not None else None)
        return {
            "matched_experiment_id": matched_experiment_id,
            "candidate_ref": self.candidate_ref,
            "backend": self.backend,
            "recipe_id": self.recipe_id,
            "metric": self.metric,
            "measurement_unit_id": self.measurement_unit_id,
            "model_path": model_path,
            "model_sha256": storage.hash_file(model_path),
            "reps": self.reps,
            "blocks": self.blocks,
            "fresh_pairs_per_block": self.fresh_pairs_per_block,
            "n_gen": self.n_gen,
            "n_prompt": self.n_prompt,
            "t0_ops": list(self.t0_ops),
            "devices": list(self.devices),
            "device_names": list(self.device_names),
            "device_index": self.device_index,
            "n_gpu_layers": self.n_gpu_layers,
            "cpu_list": self.cpu_list,
            "autokernel_seed": self.bench_params_for(
                matched_experiment_id)["autokernel_seed"],
            "suite_seed": self.suite_seed_for(matched_experiment_id),
            "schedule_seed": self.schedule_seed_for(matched_experiment_id),
            "holdout_selection_seed": self.holdout_selection_seed_for(
                matched_experiment_id),
            "calibration": {
                "recipe_id": calibration.recipe_id,
                "contribution_floor": calibration.contribution_floor,
                "b_min_blocks": calibration.b_min_blocks,
                "max_blocks": calibration.max_blocks,
                "noise_floor_phi": calibration.noise_floor_phi,
                "mde": calibration.mde,
                "production_commit": calibration.production_commit,
                "measurement_commit": calibration.measurement_commit,
                "evidence_ref": calibration.evidence_ref,
            },
            "physical_envelope": envelope,
            "production_commit": PRODUCTION_COMMIT,
            "measurement_commit": MEASUREMENT_COMMIT,
            "claim_journal_path": self.claim_journal_path,
            "max_hold_s": self.max_hold_s,
            "provider_reference": provider_reference,
            "ggml_iqk": self.candidate_param_overrides.get("ggml_iqk"),
        }

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id, "candidate_id": self.candidate_id,
            "candidate_ref": self.candidate_ref, "backend": self.backend,
            # These identities are required by the prospective held-out
            # projector.  A later campaign must be able to prove that an
            # out-of-regime observation measured the same immutable candidate
            # frame without trusting the path or the projector's memory.
            "model_sha256": (
                storage.hash_file(self.model)
                if self.model and os.path.isfile(self.model) else None),
            "production_commit": PRODUCTION_COMMIT,
            "measurement_commit": MEASUREMENT_COMMIT,
            "blocks_precommitted": self.blocks, "recipe_id": self.recipe_id,
            "metric": self.metric, "drift_bound": self.drift_bound,
            "drift_bound_evidence": (
                AA_EVIDENCE_REF if self.calibration is None
                else self.calibration.anchor_motion_evidence_ref),
            "calibration": None if self.calibration is None else {
                "contribution_floor": self.calibration.contribution_floor,
                "b_min_blocks": self.calibration.b_min_blocks,
                "max_blocks": self.calibration.max_blocks,
                "noise_floor_phi": self.calibration.noise_floor_phi,
                "mde": self.calibration.mde,
                "evidence_ref": self.calibration.evidence_ref,
                "anchor_motion_bound": self.calibration.anchor_motion_bound,
                "anchor_motion_window_blocks": (
                    self.calibration.anchor_motion_window_blocks),
                "anchor_motion_evidence_ref": (
                    self.calibration.anchor_motion_evidence_ref),
            },
            "model": self.model, "reps": self.reps, "n_gen": self.n_gen,
            "n_prompt": self.n_prompt,
            "fresh_pairs_per_block": self.fresh_pairs_per_block,
            "suite_seed": self.suite_seed,
            "schedule_seed": self.schedule_seed,
            "holdout_selection_seed": self.holdout_selection_seed,
            "matched_experiment_id": self.matched_experiment_id,
            "t0_ops": list(self.t0_ops), "devices": list(self.devices),
            "device_names": list(self.device_names),
            "device_index": self.device_index,
            "n_gpu_layers": self.n_gpu_layers,
            "cpu_list": self.cpu_list, "worktree": self.worktree_path,
            "build_dir": self.build_dir, "journal_root": self.journal_root,
            "created_at": self.created_at,
            "proposal": None if self.proposal is None else {
                "proposal_id": self.proposal_id,
                "schema": self.proposal["schema"],
                "representation_frame_sha256": self.proposal[
                    "representation_contract"
                ]["frame_sha256"],
            },
            "source_patch_bundle_sha256": (
                None if self.source_patch is None else self.source_patch.patch_bundle_sha256),
            "source_prerequisite_package_sha256": (
                None if self.source_prerequisite_package is None
                else self.source_prerequisite_package.package_sha256),
            "fresh_source_prerequisite_plan_sha256": (
                None if self.fresh_source_prerequisite_plan is None
                else self.fresh_source_prerequisite_plan.plan_sha256),
            "least_commitment_capture_plan_sha256": (
                None if self.least_commitment_plan is None
                else self.least_commitment_plan.plan_sha256),
            "physical_envelope": (
                None if self.physical_envelope is None
                else self.physical_envelope.to_dict()),
            "ranked_units": [unit.to_dict() for unit in self.ranked_units],
            "hypothesis": self.hypothesis_record,
            "claim_purpose": self.claim_purpose,
            "anchor": {"repo": PRODUCTION_REPO, "branch": PRODUCTION_BRANCH,
                       "expected_commit": PRODUCTION_COMMIT},
            "measurement_instrument": {
                "repo": MEASUREMENT_REPO,
                "branch": MEASUREMENT_BRANCH,
                "expected_commit": MEASUREMENT_COMMIT,
                "parent_production_commit": PRODUCTION_COMMIT,
                "build_root": MEASUREMENT_BUILD_ROOT,
            },
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

    def record_proposal(self, spec: CampaignSpec) -> Any: ...
    def preflight(self, spec: CampaignSpec) -> schemas.Check: ...
    def acquire_claim(self, spec: CampaignSpec) -> Any: ...
    def release_claim(self, claim: Any) -> Any: ...
    def create_worktree(self, spec: CampaignSpec) -> Any: ...
    def apply_candidate(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def build(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def run_t0(self, spec: CampaignSpec, build: Any) -> T0Outcome: ...
    def settle_after_t0(self, spec: CampaignSpec, claim: Any) -> Any: ...
    def admit_t1_after_t0(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def run_paired_blocks(self, spec: CampaignSpec, build: Any,
                          claim: Any) -> Optional[Sequence[Pair]]: ...
    def teardown_worktree(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def keep_or_revert(self, spec: CampaignSpec, tree: Any,
                       decision: Optional[AcceptDecision]) -> Any: ...
    def prove_production_unchanged(self, spec: CampaignSpec) -> schemas.Check: ...
    def close_evaluation_window(self, spec: CampaignSpec, tree: Any) -> Any: ...
    def journal_evaluation(self, spec: CampaignSpec, result: Any) -> Any: ...
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

    def record_proposal(self, spec: CampaignSpec) -> Any:
        self._step(
            "record_proposal",
            "would validate and fsync the current proposal schema before preflight, "
            "claim, mutation, or build.",
            proposal_id=spec.proposal_id,
            representation_frame_sha256=spec.proposal["representation_contract"][
                "frame_sha256"
            ],
        )
        return None

    def preflight(self, spec: CampaignSpec) -> schemas.Check:
        self._step(
            "preflight",
            "would verify: frozen trees clean at their ratified commits; no concurrent "
            "inference, where anything but PASS refuses (the claim-witness layer's own "
            "rule); no overlapping region claim; SMT topology "
            "(cpu_region_claim.verify_host_topology); and the boost count UNDER LOAD only.",
            frozen_trees=list(worktree.frozen_tree_paths()),
            production=f"{PRODUCTION_REPO}@{PRODUCTION_BRANCH}#{PRODUCTION_COMMIT[:12]}",
            measurement_instrument=(
                f"{MEASUREMENT_REPO}@{MEASUREMENT_BRANCH}#{MEASUREMENT_COMMIT[:12]}"),
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
                   "would re-resolve the reviewed measurement-instrument tip, prove it "
                   "is the one-child overlay on current production v9, and add a campaign "
                   "worktree off it (StaleAnchor if either anchor moved).",
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
                   "would configure and build with GGML_CCACHE=OFF forced; ordinary host "
                   "load is recorded as noise and does not delay the build.",
                   build_dir=spec.build_dir,
                   targets=[T0_GENERATION_TOOL, "llama-bench", "test-backend-ops"])
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

    def settle_after_t0(self, spec: CampaignSpec, claim: Any) -> None:
        self._step(
            "post_t0_ready_boundary",
            "would retain every T0 sandbox-teardown receipt, then immediately record one "
            "claim-witnessed host-noise and competing-inference witness before T1.",
            samples=1, ordinary_load_policy="recorded_not_blocking")

    def run_paired_blocks(self, spec: CampaignSpec, build: Any,
                          claim: Any) -> Optional[Sequence[Pair]]:
        rendered = render_bench_commands(spec)
        if spec.screening_only:
            assert spec.screening_baseline is not None
            self._step(
                "paired_blocks",
                f"would run exactly {spec.blocks} candidate-only discovery calls against "
                "the sealed reusable baseline bank; zero anchors, no T0, no settling, "
                "and no promotion authority.",
                candidate_invocations=spec.blocks, anchor_invocations=0,
                baseline_sha256=spec.screening_baseline.to_dict()["baseline_sha256"],
                candidate_argv=rendered["candidate"]["argv"],
                candidate_env=rendered["candidate"]["env"],
                ordinary_load_policy="recorded_not_blocking")
            return None
        self._step(
            "paired_blocks",
            f"would run {spec.blocks} PRE-COMMITTED alternating paired blocks "
            "(anchor, candidate, anchor, candidate, ...). microbench.BlockPlan derives the "
            "arm sequence from the order and refuses a blocked design; the measured "
            "monotone drift is why.",
            blocks=spec.blocks, pairs_per_block=spec.fresh_pairs_per_block,
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


def render_bench_commands(spec: CampaignSpec, *,
                          params: Optional[Mapping[str, Any]] = None) -> dict:
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
    runtime_parameter_screen = (
        spec.screening_only
        and spec.proposal is not None
        and spec.proposal["change_class"] == "parameter"
    )
    for arm, root in (("anchor", MEASUREMENT_BUILD_ROOT),
                      ("candidate", (MEASUREMENT_BUILD_ROOT
                                     if runtime_parameter_screen else spec.build_dir))):
        # Rendering is a preview over placeholder roots; both the binary and
        # library directory are intentionally under the binding source root.
        # The live candidate path is bound as an external build in _construct,
        # where the real build-prefixed root and receipt are available.
        bindir = os.path.join(root, "bin")
        binding = recipes.ToolBinding(
            binary=os.path.join(bindir, tool), source_root=root,
            library_path=bindir)
        payload = recipes.dry_run(spec.recipe_id, binding=binding,
                                  params=spec.params_for_arm(arm, params=params),
                                  arm=arm,
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


class _RecordedGateRunner:
    """Evaluator gate seam over gates already produced by the executed T0."""

    def __init__(self, gates: Sequence[api.GateResult]) -> None:
        self._gates = tuple(gates)

    def run_gates(self, request: api.EvaluationRequest) -> tuple:
        return self._gates


class HostOps:
    """The real one. Touches the host, spends the claim, spawns the benchmarks.

    NEVER EXERCISED BY THE TEST SUITE, and it must not pretend otherwise: every
    test in this package runs on recorded output, and no line below has been run
    against a kernel. What IS covered is the composition it performs, in
    `execution/test_execution_chain.py::ChainLeg` — read that before editing
    anything here, because the seams it crosses are the ones where two modules
    have records with the same name and different fields.

    Two deliberate limits, stated rather than hidden:

      * **Source-changing proposals still supply their richer T0 evidence.** The
        built-in IQK parameter adapter derives its empty diff, registered
        dispatch surface, symbol tables and reference applicability. A source
        mutation has proposal-specific files, symbols and registration patterns;
        pass `t0_evidence=` for that path. `unimplemented_seams()` refuses a
        source campaign before the claim when it is absent.
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
        "t0_evidence":
            "the anchor capture and the richer T0 surfaces, which need the PROPOSAL's "
            "own declarations. Pass HostOps(t0_evidence=...); the reference wiring is "
            "ChainLeg.t0_evidence_inputs",
        "source_prerequisites":
            "a source proposal requires either an immutable archive or a predeclared "
            "fresh producer plan. Pass one of --source-prerequisite-package / "
            "--fresh-source-prerequisite-plan; parameter proposals reject both",
        "nominal_khz":
            "the healthy all-core clock for this cell, which only the operator can "
            "supply (--nominal-khz). Without it every frequency reading in the run "
            "classifies FREQUENCY_UNEVALUABLE, so the throttle guard is never once "
            "JUDGED under load and MicrobenchRunner emits no number — the claim and "
            "the build are spent for nothing. cpuinfo_max_freq is the single-core "
            "boost ceiling and is NOT a valid all-core reference",
    }

    def unimplemented_seams(self, spec: Optional[CampaignSpec] = None) -> tuple:
        """Which required seams are still unsupplied. Empty means runnable.

        Checked BEFORE the claim, in `main`, because the alternative is what the
        code does otherwise: acquire a region claim, create a worktree, spend
        forty minutes building, and THEN raise `NotImplementedError` on the
        first line that needed a value nobody supplied. Probe the config before
        the long run (`feedback_bench_max_opt_and_config_probe_first`).

        Derived from what is actually bound and from the proposal class rather
        than from a flag someone has to remember to flip.
        """
        missing = []
        parameter_only = bool(
            spec is not None and spec.proposal is not None
            and spec.proposal.get("change_class") == "parameter")
        if (not parameter_only and (spec is None or spec.source_patch is None)
                and getattr(type(self), "apply_candidate", None) is HostOps.apply_candidate):
            missing.append("apply_candidate")
        if self._t0_evidence is None and not parameter_only:
            missing.append("t0_evidence")
        if (spec is not None and spec.proposal is not None and not parameter_only
                and spec.source_prerequisite_package is None
                and spec.fresh_source_prerequisite_plan is None):
            missing.append("source_prerequisites")
        if self._nominal_khz is None:
            missing.append("nominal_khz")
        return tuple(sorted(missing))

    def __init__(self, *, spawner: Optional[Any] = None,
                 t0_evidence: Optional[Callable[..., Mapping[str, Any]]] = None,
                 nominal_khz: Optional[int] = None,
                 host_state: Optional[Callable[..., microbench.HostState]] = None,
                 source_prerequisite_runner: Optional[Any] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._spawner = spawner
        self._t0_evidence = t0_evidence
        self._nominal_khz = nominal_khz
        self._read_host_state = host_state or microbench.read_host_state
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._sleep = sleep
        self._source_prerequisite_runner = source_prerequisite_runner
        self._claim_binding: Optional[Any] = None
        self._device_claims: list = []
        self._build_state: dict = {}
        self._build_load_settlement: tuple[float, ...] = ()
        self._fingerprints: dict = {}
        self._t0_anchor_binding: Optional[Any] = None
        self._preflight_check: Optional[schemas.Check] = None
        self._preflight_open_receipt: Optional[Mapping[str, Any]] = None
        self._preflight_close_receipt: Optional[Mapping[str, Any]] = None
        self._host_open: Optional[Any] = None
        self._claim_release_receipt: Optional[Mapping[str, Any]] = None
        self._claim_open_receipt: Optional[Mapping[str, Any]] = None
        self._claim_close_receipt: Optional[Mapping[str, Any]] = None
        self._claim_open_check: Optional[schemas.Check] = None
        self._claim_close_check: Optional[schemas.Check] = None
        self._claim_same_holder_check: Optional[schemas.Check] = None
        self._no_concurrent_close: Optional[schemas.Check] = None
        self._t0_request: Optional[api.EvaluationRequest] = None
        self._t0_report: Optional[correctness.T0Report] = None
        self._t0_gate_results: tuple = ()
        self._t0_event_request: Optional[api.EvaluationRequest] = None
        self._t0_evaluation_event: Optional[dict] = None
        self._t0_started = False
        self._t0_capture_archive: Optional[t0_provider.DirectoryCaptureSink] = None
        self._t0_capture_refs: tuple[str, ...] = ()
        self._post_t0_settlement: Optional[Mapping[str, Any]] = None
        self._microbench_run: Optional[microbench.MicrobenchRun] = None
        self._screening_report: Optional[Mapping[str, Any]] = None
        self._t1_request: Optional[api.EvaluationRequest] = None
        self._storage_open: Optional[schemas.Check] = None
        self._storage_close: Optional[schemas.Check] = None
        self._host_close: Optional[Any] = None
        self._host_health_close: Optional[schemas.Check] = None
        self._anchors_at_close: dict[str, api.AnchorIdentity] = {}
        self._anchor_close_checks: dict[str, schemas.Check] = {}
        self._evaluator_bundle_snapshot: Optional[api.EvaluatorIdentity] = None
        self._evaluator_bundle_snapshot_ref: Optional[str] = None
        self._evaluator_close_check: Optional[schemas.Check] = None
        self._runtime_close_check: Optional[schemas.Check] = None
        self._recipe_receipts: dict[tuple[str, str], api.RecipeReceipt] = {}
        self._source_application: Optional[source_candidate.AppliedSourceCandidate] = None
        self._build_identity: Optional[worktree.BuildIdentity] = None
        self._build_snapshot: Optional[worktree.Worktree] = None
        self._cached_evaluation_events: Optional[tuple] = None
        self._cached_candidate_record: Optional[dict] = None

    def calibration_gate(self, spec: CampaignSpec) -> schemas.Check:
        """Refuse live ranking outside the accepted cell-local calibration."""
        if spec.screening_only:
            # CampaignSpec has already admitted the immutable bank against the
            # exact boot/source/model/recipe frame.  A discovery screen ranks
            # only for top-K nomination and is structurally non-promotable, so
            # strict paired-T1 calibration authority is neither applicable nor
            # required here.
            return schemas.Check(schemas.PASS, (
                "screening-only uses its exact-frame sealed baseline bank; "
                "strict paired-T1 calibration is not selection authority",))
        calibration = spec.calibration
        if calibration is None:
            return schemas.Check(schemas.FAIL, (
                f"recipe {spec.recipe_id!r} has no accepted cell-local live calibration; "
                "a calibration from another phase or recipe cannot license ranking",))
        if not calibration.b_min_blocks <= spec.blocks <= calibration.max_blocks:
            return schemas.Check(schemas.FAIL, (
                f"blocks={spec.blocks} is outside the accepted calibration range "
                f"[{calibration.b_min_blocks}, {calibration.max_blocks}] from "
                f"{calibration.evidence_ref}",))
        return schemas.Check(schemas.PASS, ())

    @staticmethod
    def _storage_check(spec: CampaignSpec) -> schemas.Check:
        if spec.calibration is None or spec.calibration.evaluation_authority is None \
                or not spec.journal_root:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "storage floor or journal root was not available",))
        try:
            stat = os.statvfs(spec.journal_root)
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"could not read journal filesystem headroom: {exc}",))
        free = stat.f_bavail * stat.f_frsize
        floor = spec.calibration.evaluation_authority.campaign_controls.storage_floor_bytes_free
        if free < floor:
            return schemas.Check(schemas.FAIL, (
                f"journal filesystem has {free} bytes free, below floor {floor}",))
        return schemas.Check(schemas.PASS, (
            f"journal filesystem has {free} bytes free, at or above floor {floor}",))

    @staticmethod
    def _t0_evaluator_policy(spec: CampaignSpec) -> correctness.T0Policy:
        """Return the fixed evaluator policy for an executed campaign.

        The policy lives on the evaluator side of the boundary, but construction
        happens here because the campaign supplies the backend identity.  It is
        intentionally explicit: :class:`T0Policy` has no defaults, and a bare
        construction would turn a protocol omission into a live-run failure.
        """
        return correctness.T0Policy(
            required_backend_ops=correctness.MANDATORY_BACKEND_OPS,
            symbol_shrinkage_reject_ratio=0.6,
            diff_ceiling=correctness.DiffComplexityCeiling(
                backend=spec.backend,
                max_changed_lines=400,
                max_files_touched=10,
                shared_core_forces_review=True),
            determinism_min_runs=2,
            coherence_tolerance_floor=0.98,
            policy_ref="ak-policy/v1")

    # -- 0. preflight ------------------------------------------------------

    @staticmethod
    def _evaluator_bundle_files() -> tuple[Path, ...]:
        """Return the complete driver-side evaluator closure to seal at start."""
        direct = (
            Path(__file__), Path(api.__file__), Path(correctness.__file__),
            Path(schemas.__file__), Path(recipes.__file__),
            Path(control_runner.__file__), Path(source_prerequisite_producer.__file__),
        )
        # A module can be part of both the driver's direct evaluator and a
        # nested authority's complete closure.  `correctness.py` is currently
        # such a module: T0 imports it directly and the source-prerequisite
        # reducer also binds it.  Form the union over the exact Path identities
        # here so the seal remains complete without listing the same bytes
        # twice.  Deliberately do *not* resolve paths while de-duplicating:
        # lexical/symlink aliases still reach `_bind_evaluator_identity` as two
        # entries and fail its duplicate/ambiguity guard closed.
        return tuple(dict.fromkeys(
            direct + source_prerequisite_package.evaluator_source_files()))

    def _bind_evaluator_identity(self, spec: CampaignSpec) -> api.EvaluatorIdentity:
        """Persist the loaded evaluator's start-time bytes before a claim.

        A close-window reread of the shared checkout is not a test of the
        evaluator this Python process executed: another session can legitimately
        update the checkout while T0/T1 run.  Seal once, then use that durable
        start-time identity for every request and the close comparison.
        """
        if self._evaluator_bundle_snapshot is not None:
            return self._evaluator_bundle_snapshot
        if spec.calibration is None or spec.calibration.evaluation_authority is None:
            raise RuntimeError("evaluator bundle seal requires accepted live authority")
        if not spec.journal_root:
            raise RuntimeError("evaluator bundle seal requires an executing campaign journal")
        root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
        bundle_root = Path(root) / "evaluator-bundle"
        files_root = bundle_root / "files"
        if bundle_root.exists():
            raise RuntimeError(
                f"evaluator bundle directory already exists at {bundle_root}; "
                "a campaign id may not reuse a prior evaluator seal")
        files_root.mkdir(parents=True, mode=0o700)
        source_hashes: dict[str, str] = {}
        records: list[dict[str, str]] = []
        try:
            files = tuple(sorted(self._evaluator_bundle_files(), key=str))
            if not files or len({str(path) for path in files}) != len(files):
                raise RuntimeError("evaluator bundle closure is empty or contains duplicate paths")
            for index, path in enumerate(files):
                source = Path(path)
                content = source.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                source_key = str(source)
                destination = files_root / f"{index:02d}-{digest}-{source.name}"
                destination.write_bytes(content)
                destination.chmod(0o400)
                source_hashes[source_key] = digest
                records.append({"source": source_key, "sha256": digest,
                                "snapshot": str(destination.relative_to(bundle_root))})
            # A mutation while the seal is being made is a pre-start integrity
            # failure.  Later edits are deliberately irrelevant to this loaded
            # evaluator, but this race must fail before a claim is acquired.
            for source_key, digest in source_hashes.items():
                if storage.hash_file(source_key) != digest:
                    raise RuntimeError(
                        f"evaluator source {source_key!r} changed while its start-time "
                        "bundle was being sealed")
            authority = spec.calibration.evaluation_authority
            identity = api.EvaluatorIdentity(
                id="autokernel.campaign-live-evaluation/v1",
                bundle_sha256=schemas.content_hash(source_hashes),
                runtime_source_label_ref=authority.runtime_source_label_ref,
            )
            manifest = {
                "schema": "epyc.autokernel.evaluator-bundle.v1",
                "campaign_id": spec.campaign_id,
                "identity": identity.to_dict(), "files": records,
            }
            manifest["snapshot_sha256"] = schemas.content_hash(manifest)
            manifest_path = bundle_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n",
                                     encoding="utf-8")
            manifest_path.chmod(0o400)
        except BaseException:
            # Do not leave a partial directory that a same-id retry could call a
            # seal.  It contains only bytes written by this method.
            if bundle_root.exists():
                for path in sorted(bundle_root.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                bundle_root.rmdir()
            raise
        self._evaluator_bundle_snapshot = identity
        self._evaluator_bundle_snapshot_ref = str(manifest_path)
        return identity

    def record_proposal(self, spec: CampaignSpec) -> Any:
        """Fsync the current proposal schema before host work; resume is idempotent."""
        if spec.proposal is None or not spec.journal_root:
            raise RuntimeError(
                "an executing campaign requires a current-schema proposal and --journal-root")
        root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
        book = journal_module.Journal(root, campaign_id=spec.campaign_id)
        book.initialize()
        for entry in book.read_all():
            if (
                entry.kind == journal_module.KIND_PROPOSAL_RECORDED
                and entry.record_id == spec.proposal_id
            ):
                if schemas.content_hash(entry.payload) != schemas.content_hash(spec.proposal):
                    raise RuntimeError(
                        f"proposal id {spec.proposal_id!r} already names different bytes"
                    )
                return entry.event_id
        return book.append(
            journal_module.KIND_PROPOSAL_RECORDED, dict(spec.proposal)
        ).event_id

    def preflight(self, spec: CampaignSpec) -> schemas.Check:
        """Host canonical, and nobody else on the cores. Reads; acquires nothing.

        A PASS here is an OBSERVATION, never a claim — nothing stops another
        process taking the region in the interval between this and
        `acquire_claim`. The sequence is preflight -> acquire -> run, and the
        claim is the thing that makes the run defensible.
        """
        if not spec.journal_root:
            return schemas.Check(schemas.FAIL, (
                "an executing campaign requires --journal-root before any claim or T0 "
                "inference; completed benchmark attempts cannot be machine-enforced in "
                "volatile memory",))
        # Arithmetic/preflight-only callers have no live evaluator authority.
        # Real T0/T1 requests require it and are sealed here, before any claim.
        if spec.calibration is not None and spec.calibration.evaluation_authority is not None:
            self._bind_evaluator_identity(spec)
        self._storage_open = self._storage_check(spec)
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

        # Serving and reward-instrument identities are distinct.  The latter is
        # an evaluator-only one-commit overlay on production and is the source
        # from which both the T1 anchor and every candidate are built.
        try:
            production_anchor = worktree.resolve_anchor(
                worktree.GitRepo(PRODUCTION_REPO), PRODUCTION_BRANCH,
                expected_commit=PRODUCTION_COMMIT)
            measurement_repo = worktree.GitRepo(MEASUREMENT_REPO)
            measurement_anchor = worktree.resolve_anchor(
                measurement_repo, MEASUREMENT_BRANCH,
                expected_commit=MEASUREMENT_COMMIT)
        except worktree.WorktreeError as exc:
            fold(schemas.Check(schemas.FAIL, (str(exc),)), "source_anchor", hard=True)
        else:
            self._fingerprints[MEASUREMENT_REPO] = measurement_anchor.fingerprint
            if measurement_anchor.fingerprint.head_commit != MEASUREMENT_COMMIT:
                fold(schemas.Check(schemas.FAIL, (
                    "measurement working tree HEAD is not the reviewed instrument commit",)),
                    "measurement_instrument", hard=True)
            if measurement_anchor.fingerprint.symbolic_ref != MEASUREMENT_BRANCH:
                fold(schemas.Check(schemas.FAIL, (
                    f"measurement working tree is on "
                    f"{measurement_anchor.fingerprint.symbolic_ref!r}, required "
                    f"{MEASUREMENT_BRANCH!r}",)), "measurement_instrument", hard=True)
            if measurement_anchor.fingerprint.status_porcelain:
                fold(schemas.Check(schemas.FAIL, (
                    "measurement working tree is dirty; source pinning must name committed "
                    "bytes only",)), "measurement_instrument", hard=True)
            parents = measurement_repo.commit_parents(MEASUREMENT_COMMIT)
            if not measurement_repo.is_ancestor(
                    production_anchor.commit, MEASUREMENT_COMMIT):
                fold(schemas.Check(schemas.FAIL, (
                    f"measurement commit parents are {parents!r}; required a commit "
                    f"descending from current production {production_anchor.commit!r}",)),
                    "measurement_instrument", hard=True)

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
        self._preflight_open_receipt = result.to_dict()
        self._preflight_check = result.as_check()
        fold(self._preflight_check, "concurrent_inference",
             hard=True)

        state = self._read_host_state(cpu_list=spec.cpu_list)
        self._host_open = state
        fold(check_host_uptime(getattr(state, "uptime_s", None)), "host_uptime")
        boosting = sum(1 for _cpu, khz in state.khz_by_cpu if khz >= BOOST_THRESHOLD_KHZ)
        # Retain the point-in-time frequency/load snapshot as a non-gating
        # diagnostic.  The 1-minute load average after a preceding matched arm
        # is that arm's decaying self-load while the instantaneous frequencies
        # below are from now-idle, parked cores.  Combining those two clocks
        # made a healthy paired control look throttled.  Frequency and package
        # power remain fail-closed under the campaign's own measured block
        # load.  COULD_NOT_CHECK is executable below; FAIL remains reserved for
        # checks whose observation window overlaps the phenomenon.
        check_boost_under_load(
            boosting_cores=boosting, load1=state.load1,
            cpu_count=len(state.khz_by_cpu) or 1)

        policy = microbench.HostStatePolicy(
            nominal_khz=self._nominal_khz,
            require_load=False,
            require_package_power=(spec.backend == BACKEND_CPU))
        # Ordinary host load is recorded noise under the ratified discovery
        # policy.  Concurrent model inference is still rejected above by the
        # claim/witness preflight; retain ``require_load=False``'s explicit
        # diagnostic as executable COULD_NOT_CHECK instead of folding it into
        # a campaign refusal.
        policy.check_load(state, cpu_count=len(state.khz_by_cpu) or 1)
        if spec.backend == BACKEND_CPU:
            # A point reading is only an availability preflight.  Exact power
            # is derived later from each block's open/close counter interval.
            # Hard refusal here avoids acquiring the machine for a CPU campaign
            # whose required host receipt is known to be impossible.
            fold(policy.check_package_power_available(state),
                 "package_power_available", hard=True)

        if outcome == schemas.PASS:
            return schemas.Check(schemas.PASS, tuple(reasons) + (
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
                spec.cpu_list,
                role=(inference_window.WINDOWED_CPU_ROLE
                      if spec.backend == BACKEND_CPU else "autokernel"),
                purpose=purpose,
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
                # `max_hold_s` DECLARES the window, and declaring it is what writes
                # `expires_at` into the claim payload. Without it,
                # `check_claim_expiry()` returns COULD_NOT_CHECK forever rather than
                # FAIL ("claim … declared no maximum hold, so expiry cannot be
                # evaluated", device_claim.py) — so the one expiry check this fleet
                # already owns was disarmed for exactly the claim that monopolised
                # the MI210 through the night of 2026-08-11/12, while the CPU claim
                # three lines above had been declaring its window all along.
                #
                # THE VALUE IS `spec.max_hold_s`, not a fresh constant, and that is
                # the point: both claims are taken in this one transaction, for one
                # campaign, and released together in `release_claim`. Two claims
                # covering the same window that declare different deadlines are a
                # defect by construction, so the window is declared ONCE on the spec
                # (default 6 h, `CampaignSpec.max_hold_s`) and both acquirers quote
                # it. A campaign that legitimately needs longer raises it in one
                # place and both claims move together.
                #
                # It stays ADVISORY. An expired claim is never stolen: expiry is a
                # declaration by the holder, not a fact about it, and a FAIL is a
                # reason to call `request_revocation` — quiesce-and-drain, honoured
                # by the holder at its own boundary — never a licence to reclaim.
                # Arming the check changes what can be OBSERVED about a claim and
                # nothing about who may take the device.
                self._device_claims.append(device_claim.acquire_device_claim(
                    device_id, purpose=f"AutoKernel {spec.campaign_id}",
                    campaign_id=spec.campaign_id, journal=journal,
                    max_hold_s=float(spec.max_hold_s)))
            region_receipt = claim.receipt().to_dict()
            device_receipts = [held.receipt().to_dict() for held in self._device_claims]
            self._claim_open_receipt = {
                "region": region_receipt, "devices": device_receipts}
            self._claim_open_check = schemas.Check.worst_of((
                claim.verify_held(),
                *(device_claim.check_device_claim_held(receipt)
                  for receipt in device_receipts),
            ))
            if self._claim_open_check.outcome != schemas.PASS:
                raise RuntimeError(
                    "resource claim could not be verified immediately after acquisition: "
                    + "; ".join(self._claim_open_check.reasons))
        except BaseException:
            self._release_device_claims()
            self._claim_binding = None
            try:
                claim.release()
            except BaseException:  # noqa: BLE001 - the original failure is the news
                pass
            raise
        return claim

    @staticmethod
    def _claim_holder_identity(receipt: Mapping[str, Any]) -> tuple:
        """Stable identities for every claim plane in one window snapshot."""
        region = receipt.get("region")
        devices = receipt.get("devices", ())
        values = []
        if isinstance(region, Mapping):
            identity = (region.get("claim_id"), region.get("holder_pid"),
                        region.get("holder_start_ticks"), region.get("holder_boot_id"))
            if any(value is None or value == "" for value in identity):
                return ()
            values.append(("region", *identity))
        for item in devices if isinstance(devices, (list, tuple)) else ():
            if isinstance(item, Mapping):
                identity = (item.get("device_id"), item.get("claim_id"),
                            item.get("holder_pid"), item.get("holder_start_ticks"),
                            item.get("holder_boot_id"))
                if any(value is None or value == "" for value in identity):
                    return ()
                values.append(identity)
            else:
                return ()
        return tuple(sorted(values, key=lambda value: str(value[0])))

    @classmethod
    def _check_same_claim_holder(cls, opened_receipt: Mapping[str, Any],
                                 closed_receipt: Mapping[str, Any]) -> schemas.Check:
        opened = cls._claim_holder_identity(opened_receipt)
        closed = cls._claim_holder_identity(closed_receipt)
        return schemas.Check(
            schemas.PASS if opened and opened == closed else schemas.FAIL,
            () if opened and opened == closed else
            (f"claim holder identity incomplete or moved: "
             f"open={opened!r}, close={closed!r}",))

    def close_evaluation_window(self, spec: CampaignSpec, tree: Any) -> None:
        """Re-attest the live window while every claim is still held.

        This method deliberately performs fresh reads.  A release receipt can
        prove that locks were eventually dropped, but it cannot prove that the
        same locks, anchor, evaluator, and host remained valid through the last
        measured block.
        """
        unknown = lambda reason: schemas.Check(schemas.COULD_NOT_CHECK, (reason,))

        claim = None if self._claim_binding is None else self._claim_binding.claim
        claim_checks = []
        close_devices = []
        if claim is None:
            claim_checks.append(unknown("CPU region claim was unavailable at window close"))
            close_region = None
        else:
            claim_checks.append(claim.verify_held())
            close_region = claim.receipt().to_dict()
        for held in self._device_claims:
            receipt = held.receipt().to_dict()
            close_devices.append(receipt)
            claim_checks.append(device_claim.check_device_claim_held(receipt))
        self._claim_close_receipt = {"region": close_region, "devices": close_devices}
        self._claim_close_check = schemas.Check.worst_of(claim_checks)
        if self._claim_open_receipt is None:
            self._claim_same_holder_check = unknown(
                "claim identity was not retained at window open")
        else:
            self._claim_same_holder_check = self._check_same_claim_holder(
                self._claim_open_receipt, self._claim_close_receipt)

        try:
            scope = (preflight.PreflightScope.gpu(spec.campaign_id, spec.devices)
                     if spec.backend == BACKEND_GPU else
                     preflight.PreflightScope.whole_machine_cpu(spec.campaign_id))
            sources = claim_witness.gpu_claim_sources(
                spec.devices,
                region_lock_dir=str(cpu_region_claim.default_region_lock_dir()))
            result = preflight.preflight(scope, sources)
            self._preflight_close_receipt = result.to_dict()
            self._no_concurrent_close = result.as_check()
        except BaseException as exc:  # noqa: BLE001 - becomes three-valued evidence
            self._no_concurrent_close = unknown(
                f"concurrent-inference close preflight raised: {type(exc).__name__}: {exc}")

        host_checks = []
        try:
            state = self._read_host_state(cpu_list=spec.cpu_list)
            self._host_close = state
            policy = microbench.HostStatePolicy(
                nominal_khz=self._nominal_khz,
                require_package_power=(spec.backend == BACKEND_CPU))
            # Do not apply the idle/open load policy here: load1 still includes
            # this campaign's own just-finished benchmark.  The runner's actual
            # pre-spawn load check below is the contention observation; close
            # contributes uptime and counter availability.
            host_checks.append(check_host_uptime(getattr(state, "uptime_s", None)))
            if spec.backend == BACKEND_CPU:
                host_checks.append(policy.check_package_power_available(state))
        except BaseException as exc:  # noqa: BLE001
            host_checks.append(unknown(
                f"host close-state read raised: {type(exc).__name__}: {exc}"))
        if self._microbench_run is not None:
            host_checks.extend(
                check for name, check in self._microbench_run.checks
                if name == "host_load_open")
            for block in self._microbench_run.blocks:
                host_checks.extend(
                    check for name, check in block.checks
                    if name in ("host_frequency_block_close",
                                "host_package_power_block"))
                for invocation in block.invocations:
                    host_checks.extend(
                        check for name, check in invocation.checks
                        if name == "gpu_device_state_window")
            host_checks.append(self._microbench_run.order_control)
            host_checks.append(schemas.Check(
                schemas.PASS if self._microbench_run.complete else schemas.FAIL,
                () if self._microbench_run.complete else
                ("microbench run was incomplete at window close",)))
        self._host_health_close = schemas.Check.worst_of(host_checks)
        self._storage_close = self._storage_check(spec)

        requests = tuple(request for request in (self._t0_request, self._t1_request)
                         if request is not None)
        if not requests:
            self._evaluator_close_check = unknown(
                "no executed evaluator identity was available at window close")
        else:
            for request in requests:
                tool = request.anchor.tool or T0_GENERATION_TOOL
                try:
                    capture = t0_provider.capture_anchor_identity(
                        anchor=self._measurement_anchor_build(tool), tools=self._t0_tools(),
                        runner=t0_provider.SubprocessRunner(),
                        base_env=tuple(sorted(
                            self._construct(spec, arm="anchor").env.items())),
                        parameter_env=spec.t0_parameter_env_for_arm("anchor"))
                    binding = chain.bind_anchor(capture, tool=tool)
                    self._anchors_at_close[tool] = binding.identity
                    self._anchor_close_checks[tool] = request.anchor.identity_matches(
                        binding.identity)
                except BaseException as exc:  # noqa: BLE001
                    self._anchor_close_checks[tool] = unknown(
                        f"anchor close capture for {tool} raised: "
                        f"{type(exc).__name__}: {exc}")

        authority = (None if spec.calibration is None else
                     spec.calibration.evaluation_authority)
        if authority is None:
            self._runtime_close_check = unknown(
                "typed live evaluation authority was unavailable at window close")
            if self._evaluator_close_check is None:
                self._evaluator_close_check = unknown(
                    "typed live evaluation authority was unavailable at window close")
        else:
            try:
                current = control_runner.load_live_evaluation_authority(
                    authority.evidence_ref)
                same_authority = current == authority
                self._runtime_close_check = schemas.Check(
                    schemas.PASS if same_authority else schemas.FAIL,
                    () if same_authority else
                    ("live calibration/control authority changed during the window",))
            except BaseException as exc:  # noqa: BLE001
                self._runtime_close_check = unknown(
                    f"runtime authority close read raised: {type(exc).__name__}: {exc}")
            if requests:
                try:
                    current_evaluator = self._evaluator_identity(authority)
                    same_evaluator = all(
                        current_evaluator == request.evaluator for request in requests)
                    self._evaluator_close_check = schemas.Check(
                        schemas.PASS if same_evaluator else schemas.FAIL,
                        () if same_evaluator else
                        ("evaluator identity changed during the evaluation window",))
                except BaseException as exc:  # noqa: BLE001
                    self._evaluator_close_check = unknown(
                        f"evaluator close hash raised: {type(exc).__name__}: {exc}")

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
        receipt = {"region": region, "devices": devices_released}
        self._claim_release_receipt = receipt
        return receipt

    # -- 2. worktree -------------------------------------------------------

    def create_worktree(self, spec: CampaignSpec) -> Any:
        repo = worktree.GitRepo(MEASUREMENT_REPO)
        anchor = worktree.resolve_anchor(repo, MEASUREMENT_BRANCH,
                                         expected_commit=MEASUREMENT_COMMIT)
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
        """Apply a candidate, with a built-in path only for no-source parameters.

        The first campaign is the registered ``ggml_iqk`` parameter comparison.
        Its candidate and anchor are the same reviewed source and differ only in
        the arm-local environment projected by ``recipes``.  Treating that as a
        patch would create a fake commit.  The adapter instead proves the fresh
        worktree has an empty source diff and records the validated parameter
        surface.  Source mutations remain campaign-specific and fail closed.
        """
        if spec.proposal is not None and spec.proposal.get("change_class") == "parameter":
            diff_text = tree.unified_diff_from_source()
            if not isinstance(diff_text, str):
                raise TypeError("campaign worktree unified_diff_from_source() must return str")
            if diff_text.strip():
                raise RuntimeError(
                    "a parameter-only proposal reached a worktree with a source diff; the "
                    "comparison is no longer one-factor and will not be relabelled parameter")
            return {
                "change_class": "parameter",
                "source_diff_sha256": schemas.content_hash({"diff": diff_text}),
                "candidate": dict(spec.candidate_param_overrides),
                "anchor": dict(spec.anchor_param_overrides),
                "source_mutated": False,
            }
        if spec.proposal is None or spec.source_patch is None:
            raise RuntimeError("source candidate requires proposal and immutable source patch")
        self._source_application = source_candidate.apply_source_candidate(
            spec.source_patch, proposal=spec.proposal, actor=tree)
        return self._source_application

    # -- 3. build ----------------------------------------------------------

    def _settle_build_load(self, plan: worktree.BuildPlan) -> tuple[float, ...]:
        """Take one noise receipt; ordinary host activity never delays a build."""
        del plan
        return (float(os.getloadavg()[0]),)

    def build(self, spec: CampaignSpec, tree: Any) -> Any:
        if isinstance(tree, worktree.Worktree):
            snapshot_path = worktree.snapshot_worktree_path(
                spec.campaign_id, spec.candidate_id)
            snapshot, proof = worktree.create_snapshot_worktree(
                tree.repo, tree.head_commit(), snapshot_path)
            if not proof.holds:
                raise worktree.ProductionMutated(
                    f"creating build snapshot changed the source tree: {proof.differences}")
            self._build_snapshot = snapshot
        else:  # test-double compatibility; executing HostOps always has the typed tree
            snapshot = tree
        plan = worktree.BuildPlan(
            source_root=snapshot.path,
            build_dir=worktree.default_build_dir(spec.campaign_id, spec.candidate_id),
            actor_worktree=tree.path,
            parallelism=worktree.BuildParallelism(jobs=64, load_average_cap=None),
            targets=(T0_GENERATION_TOOL, "llama-bench", "test-backend-ops"),
            cmake_defines=(("LLAMA_FATAL_WARNINGS", "ON"),
                           ("LLAMA_BUILD_EXAMPLES", "ON")),
            cmake="/usr/bin/cmake")
        log_path = os.path.join(spec.build_root, spec.campaign_id,
                                f"{spec.candidate_id}.log")
        self._build_load_settlement = self._settle_build_load(plan)
        result = worktree.run_build(plan, log_path=log_path)
        self._build_state = {"plan": plan, "result": result, "tree": snapshot,
                             "mutation_tree": tree,
                             "load_settlement": self._build_load_settlement}
        return result

    # -- 4. T0 -------------------------------------------------------------

    @staticmethod
    def _t0_tools() -> t0_provider.ToolPaths:
        return t0_provider.ToolPaths(
            bash="/bin/bash",
            verify_ggml_linkage_sh=str(
                recipes.REPO_ROOT / "scripts" / "utils" / "verify_ggml_linkage.sh"),
            cmake="/usr/bin/cmake")

    @staticmethod
    def _measurement_anchor_build(tool: str) -> t0_provider.AnchorBuild:
        bindir = os.path.join(MEASUREMENT_BUILD_ROOT, "bin")
        return t0_provider.AnchorBuild(
            worktree=MEASUREMENT_REPO, source_commit=MEASUREMENT_COMMIT,
            binary=os.path.join(bindir, tool), library_path=bindir)

    @staticmethod
    def _t0_generation_plan(spec: CampaignSpec) -> t0_provider.GenerationPlan:
        """The T0 graph/output probe, bound to this campaign's exact model.

        The direct completion executable needs the model explicitly to
        construct a graph and generate output. Keep it in the typed plan,
        rather than relying on an ambient default or a build-local model path.
        ``llama-cli`` is intentionally not used: its current implementation
        starts a loopback HTTP server, while T0's candidate sandbox is
        network-denied.
        """
        return t0_provider.GenerationPlan(
            prompt="The capital of France is", prompt_ref="ak-prompt-001",
            n_predict=32, seed=42, extra_argv=("-m", spec.model))

    @staticmethod
    def _t0_capture_sink(spec: CampaignSpec) -> t0_provider.DirectoryCaptureSink:
        """The durable archive behind every T0 ``akcap:`` evidence reference.

        Capture before the first anchor generation: a model/containment failure
        is the case where stdout, stderr, and the Landlock receipt are most
        needed, and a sink installed only for the candidate loses half of the
        comparison.  This directory is evaluator-owned beside the journal, not
        inside the candidate's writable sandbox.
        """
        if not spec.journal_root:
            raise RuntimeError("T0 capture archive needs the executing campaign journal root")
        root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
        capture_root = os.path.join(root, "t0-captures")
        os.makedirs(capture_root, mode=0o700, exist_ok=True)
        return t0_provider.DirectoryCaptureSink(capture_root)

    def _evaluator_identity(self, authority: control_runner.LiveEvaluationAuthority
                            ) -> api.EvaluatorIdentity:
        """Return the immutable evaluator identity sealed before preflight."""
        identity = self._evaluator_bundle_snapshot
        if identity is None:
            raise RuntimeError(
                "evaluator identity requested before the immutable start-time bundle "
                "was sealed")
        if identity.runtime_source_label_ref != authority.runtime_source_label_ref:
            raise RuntimeError(
                "live authority runtime-source label differs from the start-time "
                "evaluator bundle")
        return identity

    def _parameter_t0_evidence(self, spec: CampaignSpec, *, identity: Any,
                               build_evidence: Any) -> dict:
        """Derive every non-behavioural T0 surface for a no-source IQK arm."""
        if spec.proposal is None or spec.proposal.get("change_class") != "parameter":
            raise RuntimeError("the built-in T0 evidence adapter only licenses parameter proposals")
        tree = self._build_state["tree"]
        plan = self._build_state["plan"]
        diff_text = tree.unified_diff_from_source()
        if diff_text.strip():
            raise RuntimeError("parameter T0 evidence refuses a non-empty source diff")
        if self._t0_anchor_binding is None:
            raise RuntimeError(
                f"parameter T0 evidence requires the measured {T0_GENERATION_TOOL} anchor")

        anchor_lib_capture = t0_provider.capture_anchor_identity(
            anchor=self._measurement_anchor_build("libggml.so.0"),
            tools=self._t0_tools(), runner=t0_provider.SubprocessRunner(),
            base_env=tuple(sorted(self._construct(spec, arm="anchor").env.items())),
            parameter_env=spec.t0_parameter_env_for_arm("anchor"))
        anchor_lib = chain.bind_anchor(anchor_lib_capture, tool="libggml.so.0")
        one_build = chain.check_anchor_build_is_one_build(
            (self._t0_anchor_binding, anchor_lib))
        if one_build.outcome != schemas.PASS:
            raise chain.AnchorNotOneAnchor("; ".join(one_build.reasons))

        symbols = chain.iqk_parameter_symbol_evidence(
            anchor_binary=self._measurement_anchor_build("libggml.so.0").binary,
            candidate_binary=os.path.join(plan.build_dir.path, "bin", "libggml.so.0"),
            anchor=anchor_lib, proposal=spec.proposal,
            anchor_root=MEASUREMENT_REPO, candidate_root=tree.path.path)

        diff = chain.diff_policy_evidence(
            diff_text=diff_text, worktree_root=tree.path.path,
            declared_surface_files=(),
            envelope=correctness.ChangeClassEnvelope(
                change_class="parameter", max_changed_lines=1, max_files_touched=1),
            # A clean build is intentionally made in a detached worktree at the
            # committed candidate snapshot.  The diff record's legacy
            # ``branch_name`` field still requires a non-empty provenance label,
            # but a snapshot does not have (and must not invent) a branch.  Bind
            # that label to the actual detached HEAD: a generic "detached" loses
            # the source identity that T0 is supposed to preserve.
            branch_name=(tree.branch.name if tree.branch else
                         f"detached@{tree.head_commit()}"),
            commit_argv=(),
            record_schema_violations=())

        declared_ops = tuple(spec.proposal["change"].get("predicted_affected_surface", ()))
        derivation_ref = (
            "autokernel.parameter-registry/ggml_iqk/v1:"
            + schemas.content_hash({
                "parameter_surface": spec.proposal["change"]["parameter_surface"],
                "t0_ops": list(spec.t0_ops),
            })[:32])
        surface = correctness.ChangeSurface(
            derived_touches_memory=False,
            derived_touches_threading=False,
            derived_touches_dispatch=True,
            derived_touches_persistent_state=False,
            derived_ops=tuple(spec.t0_ops), derived_files=(),
            declared_touches_memory=False, declared_touches_threading=False,
            declared_ops=declared_ops, touches_shared_core_header=False,
            derivation_ref=derivation_ref)
        surface_evidence = chain.ChangeSurfaceEvidence(
            surface=surface,
            affected={"registered_parameter": "ggml_iqk", "source_diff": "empty"},
            checks=(("registered_parameter_surface", schemas.Check(
                schemas.PASS, (
                    "the validated ggml_iqk registry projects dispatch=true and "
                    "memory/threading/persistent-state=false",))),),
            realized_edit={"change_class": "parameter", "source_mutated": False})

        return chain.t0_plan_evidence(
            symbols=symbols, diff=diff, change_surface=surface_evidence)

    def _source_prerequisites_for_t0(
            self, spec: CampaignSpec, *, identity: worktree.BuildIdentity,
            candidate: t0_provider.CandidateBuild,
            evaluator: api.EvaluatorIdentity
            ) -> tuple[correctness.SourcePrerequisiteEvidence, ...]:
        """Re-reduce the preloaded archive against one exact completed build."""
        if spec.proposal is None or spec.proposal.get("change_class") == "parameter":
            if (spec.source_prerequisite_package is not None
                    or spec.fresh_source_prerequisite_plan is not None):
                raise source_prerequisite_package.SourcePrerequisitePackageError(
                    "parameter/no-source campaign carried source prerequisites")
            return ()
        package = spec.source_prerequisite_package
        if package is None and spec.fresh_source_prerequisite_plan is not None:
            if not spec.journal_root:
                raise source_prerequisite_package.SourcePrerequisitePackageError(
                    "fresh source prerequisites require the durable campaign journal")
            runner = self._source_prerequisite_runner or t0_provider.SubprocessRunner(
                sandbox_policy=self._candidate_sandbox_policy(spec))
            try:
                package = source_prerequisite_producer.FreshSourcePrerequisiteProducer(
                    runner=runner).produce_or_resume(
                        plan=spec.fresh_source_prerequisite_plan,
                        journal_root=spec.journal_root, candidate=candidate,
                        candidate_source_sha256=identity.snapshot_sha256,
                        evaluator_bundle_sha256=evaluator.bundle_sha256,
                        base_env=tuple(sorted(
                            self._construct(spec, arm="candidate").env.items())),
                        parameter_env=spec.t0_parameter_env_for_arm("candidate"),
                        cpu_claim=self._claim_binding.t0_claim,
                        cpu_list=spec.cpu_list, held_devices=tuple(self._device_claims),
                        require_device=spec.backend == BACKEND_GPU)
            except (source_prerequisite_producer.FreshSourcePrerequisiteError,
                    OSError, storage.StorageError) as exc:
                raise source_prerequisite_package.SourcePrerequisitePackageError(
                    f"fresh producer refused: {exc}") from exc
        if package is None:
            raise source_prerequisite_package.SourcePrerequisitePackageError(
                "source candidate has neither an archive nor a fresh producer plan")
        try:
            binary_sha256 = storage.hash_file(candidate.test_backend_ops)
        except (OSError, storage.StorageError) as exc:
            raise source_prerequisite_package.SourcePrerequisitePackageError(
                f"cannot hash the live candidate test-backend-ops binary: {exc}") from exc
        return package.materialize(
            candidate_source_sha256=identity.snapshot_sha256,
            candidate_binary_sha256=binary_sha256,
            evaluator_bundle_sha256=evaluator.bundle_sha256)

    def _stop_t0_early(self, request: api.EvaluationRequest, gate_id: str,
                       check: schemas.Check) -> T0Outcome:
        """Retain a typed non-rate gate for a refusal before behavioural T0."""
        gate = api.GateResult(
            gate_id=gate_id, gate_class=api.GATE_INTEGRITY, check=check,
            requires_anchor=False,
            evidence_ref="sha256:" + schemas.content_hash({
                "gate_id": gate_id, "outcome": check.outcome,
                "reasons": list(check.reasons)}),
            notes=("pre-behavioural T0 refusal; no rate measurement exists",))
        self._t0_request = request
        self._t0_gate_results = (gate,)
        return T0Outcome(
            all_pass=False,
            gates=((gate_id, check.outcome, tuple(check.reasons)),),
            report_ref=gate.evidence_ref, gate_results=(gate,))

    @staticmethod
    def _require_t0_artifacts(plan: worktree.BuildPlan) -> dict:
        """Return the exact candidate T0 paths only when they are usable.

        A successful `cmake --build --target …` exit does not attest that every
        target the T0 composition consumes was emitted.  Checking this before
        `build_identity()` is deliberate: missing output is a build failure,
        not an artifact-hashing or T0 failure, and must spend neither evidence
        collection nor a correctness run.
        """
        bindir = os.path.join(plan.build_dir.path, "bin")
        artifacts = {
            T0_GENERATION_TOOL: os.path.join(bindir, T0_GENERATION_TOOL),
            "test-backend-ops": os.path.join(bindir, "test-backend-ops"),
            "libggml.so.0": os.path.join(bindir, "libggml.so.0"),
            "libggml-base.so.0": os.path.join(bindir, "libggml-base.so.0"),
        }
        invalid = []
        for name, path in artifacts.items():
            if not os.path.isfile(path):
                invalid.append(f"{name}: missing or not a regular file ({path})")
            elif name in {T0_GENERATION_TOOL, "test-backend-ops"} and not os.access(path, os.X_OK):
                invalid.append(f"{name}: not executable ({path})")
        if invalid:
            raise RuntimeError(
                "run_t0 refuses missing/unusable required build artifacts before "
                "artifact hashing: " + "; ".join(invalid))
        return artifacts

    def run_t0(self, spec: CampaignSpec, build: Any) -> T0Outcome:
        """T0 first, and its failure ENDS the campaign — see `run_campaign`.

        The evidence assembly crosses four seams that `chain.py` argues at
        length and that a driver must not hand-write: the build receipt is
        projected (one field INVERTS), the artifact digests are RE-MEASURED from
        disk (or the clean-build gate becomes `x == x`), the anchor is bound PER
        TOOL, and the claim is bound for both Protocols.
        """
        self._t0_started = True
        if self._claim_binding is None:
            raise RuntimeError("run_t0 was reached without a bound claim")
        result = self._build_state["result"]
        tree = self._build_state["tree"]
        plan = self._build_state["plan"]

        # Do not turn a compiler failure into a later artifact-hashing failure.
        # `build_identity()` intentionally re-measures outputs from disk, but a
        # failed build has no candidate artifact eligible for that measurement.
        # Refuse before touching either the source tree or output paths so the
        # failure remains a build outcome, never a T0/evidence outcome.
        if not result.succeeded:
            raise RuntimeError(
                "run_t0 refuses a failed build before artifact hashing "
                f"(exit_code={result.exit_code!r})")
        artifacts = self._require_t0_artifacts(plan)

        snapshot = _source_tree_digest(tree.path.path)
        identity = worktree.build_identity(
            result, candidate_id=spec.candidate_id, campaign_id=spec.campaign_id,
            worktree=tree, snapshot=snapshot,
            output_binary=artifacts[T0_GENERATION_TOOL],
            toolchain="cmake + GNU make",
            libraries={name: artifacts[name] for name in
                       ("libggml.so.0", "libggml-base.so.0")})
        candidate_capture = t0_provider.capture_anchor_identity(
            anchor=t0_provider.AnchorBuild(
                worktree=tree.path.path, source_commit=tree.head_commit(),
                binary=artifacts[T0_GENERATION_TOOL],
                library_path=os.path.join(plan.build_dir.path, "bin")),
            tools=self._t0_tools(), runner=t0_provider.SubprocessRunner(),
            base_env=tuple(sorted(self._construct(spec, arm="candidate").env.items())),
            parameter_env=spec.t0_parameter_env_for_arm("candidate"))
        identity = replace(identity, linkage_sha256=candidate_capture.linkage_sha256)
        self._build_identity = identity
        early_anchor_capture = t0_provider.capture_anchor_identity(
            anchor=self._measurement_anchor_build(T0_GENERATION_TOOL),
            tools=self._t0_tools(), runner=t0_provider.SubprocessRunner(),
            base_env=tuple(sorted(self._construct(spec, arm="anchor").env.items())),
            parameter_env=spec.t0_parameter_env_for_arm("anchor"))
        early_anchor = chain.bind_anchor(early_anchor_capture, tool=T0_GENERATION_TOOL)
        early_request = self._evaluation_request(
            spec, identity=identity, anchor_identity=early_anchor.identity,
            device_state=None,
            candidate_linkage_sha256=candidate_capture.linkage_sha256,
            determinism=api.DeterminismReport(
                determinism_class="not_measured", same_seed_repeat_runs=0))

        source_pin = instrument_integrity.compare_manifest_to_anchor(
            candidate_root=tree.path.path, anchor_root=MEASUREMENT_REPO)
        if source_pin.outcome != schemas.PASS:
            return self._stop_t0_early(
                early_request, "t0.measurement_source_pin", source_pin)

        build_ev = chain.build_evidence(identity)                       # seam 1
        if build_ev.worst.outcome != schemas.PASS:
            return self._stop_t0_early(
                early_request, "t0.build_evidence", build_ev.worst)

        candidate = chain.candidate_build_for(identity)                 # seam 3
        extra = dict(self._t0_evidence(spec=spec, identity=identity,
                                       build_evidence=build_ev)) if self._t0_evidence \
            else {}
        if "source_prerequisites" in extra:
            return self._stop_t0_early(
                early_request, correctness.GID_OP_UNITS, schemas.Check(
                    schemas.COULD_NOT_CHECK, (
                        "t0_evidence attempted to supply source prerequisites outside "
                        "the immutable campaign package boundary",)))
        if spec.proposal is not None and spec.proposal.get("change_class") != "parameter":
            try:
                extra["source_prerequisites"] = self._source_prerequisites_for_t0(
                    spec, identity=identity, candidate=candidate,
                    evaluator=early_request.evaluator)
            except source_prerequisite_package.SourcePrerequisitePackageError as exc:
                return self._stop_t0_early(
                    early_request, correctness.GID_OP_UNITS, schemas.Check(
                        schemas.COULD_NOT_CHECK,
                        (f"source prerequisite package refused: {exc}",)))
        # AK-TR-6: keep this before ExecutedT0EvidenceProvider construction.
        # That provider launches op/generation processes while collecting the
        # behavioural evidence, so discovering a compile-only veto later would
        # already have spent the GPU wall-time this check exists to save.
        artifact_evidence = extra.pop("artifact_diff", None)
        device_state = extra.pop("device_state", None)
        artifact_check = artifact_diff.require_confirmed_for_t1(artifact_evidence)
        if spec.backend == BACKEND_GPU and artifact_check.outcome != schemas.PASS:
            return self._stop_t0_early(
                early_request, "t0.compile_artifact_diff", artifact_check)
        capture_sink = self._t0_capture_sink(spec)
        self._t0_capture_archive = capture_sink
        anchor_capture = extra.pop("anchor_capture", None)
        if anchor_capture is None:
            anchor_plan = t0_provider.T0ExecutionPlan(
                candidate=candidate,
                tools=self._t0_tools(),
                op_suite=t0_provider.OpSuitePlan(
                    backend_filter="CPU" if spec.backend == BACKEND_CPU
                    else recipes.GPU_VISIBLE_DEVICE_NAME,
                ops=spec.t0_ops, suite_id="test-backend-ops/v1",
                suite_source_sha256=identity.snapshot_sha256,
                suite_seed=spec.suite_seed, capabilities=None),
                dispatch=t0_provider.DispatchTracePlan(derived_surface=spec.t0_ops),
                anchor=self._measurement_anchor_build(T0_GENERATION_TOOL),
                generation=self._t0_generation_plan(spec),
                backend=spec.backend,
                base_env=tuple(sorted(self._construct(spec, arm="anchor").env.items())),
                parameter_env=spec.t0_parameter_env_for_arm("anchor"))
            anchor_capture = t0_provider.capture_anchor(
                plan=anchor_plan,
                runner=t0_provider.SubprocessRunner(
                    sandbox_policy=self._candidate_sandbox_policy(spec)),
                claim=self._claim_binding.t0_claim,
                sink=capture_sink,
                generation_seeds=(42, 42),
                oracle_ids=(f"oracle://{MEASUREMENT_BRANCH}",))
        t0_anchor = chain.bind_anchor(anchor_capture, tool=T0_GENERATION_TOOL)  # seam 3
        self._t0_anchor_binding = t0_anchor
        if spec.proposal is not None and spec.proposal.get("change_class") == "parameter":
            derived = self._parameter_t0_evidence(
                spec, identity=identity, build_evidence=build_ev)
            for name, value in derived.items():
                extra.setdefault(name, value)

        t0_plan = t0_provider.T0ExecutionPlan(
            candidate=candidate,
            tools=self._t0_tools(),
            op_suite=t0_provider.OpSuitePlan(
                backend_filter="CPU" if spec.backend == BACKEND_CPU
                else recipes.GPU_VISIBLE_DEVICE_NAME,
                ops=spec.t0_ops,
                suite_id="test-backend-ops/v1",
                suite_source_sha256=identity.snapshot_sha256,
                suite_seed=spec.suite_seed, capabilities=None),
            dispatch=t0_provider.DispatchTracePlan(derived_surface=spec.t0_ops),
            generation=self._t0_generation_plan(spec),
            holdout=(t0_provider.HoldoutPlan(
                unseen_case_filter="type_a=(q4_K|iq4_xs)",
                boundary_case_filter="n=1",
                selection_rule_id="ak.iqk-heldout/v1",
                selection_seed=spec.holdout_selection_seed,
                visible_to_planner=False)
                if spec.proposal is not None
                and spec.proposal.get("change_class") == "parameter" else None),
            determinism_runs=2, cache_state="cold", state_safety_probe=False,
            candidate_diff_text=(
                self._source_application.diff_text
                if self._source_application is not None
                else tree.unified_diff_from_source()),
            oracle_ids=(f"oracle://{MEASUREMENT_BRANCH}",),
            base_env=tuple(sorted(self._construct(spec, arm="candidate").env.items())),
            parameter_env=spec.t0_parameter_env_for_arm("candidate"),
            build=build_ev.provenance,
            **extra)
        provider = t0_provider.ExecutedT0EvidenceProvider(
            plan=t0_plan,
            runner=t0_provider.SubprocessRunner(
                sandbox_policy=self._candidate_sandbox_policy(spec)),
            claim=self._claim_binding.t0_claim,
            sink=capture_sink,
            anchor_capture=t0_anchor.capture)
        provisional = self._evaluation_request(
            spec, identity=identity, anchor_identity=t0_anchor.identity,
            device_state=device_state,
            candidate_linkage_sha256=candidate_capture.linkage_sha256,
            determinism=api.DeterminismReport(
                determinism_class="not_measured", same_seed_repeat_runs=0))
        evidence = provider.evidence_for(provisional)
        measured_determinism = (
            api.DeterminismReport(
                determinism_class=evidence.determinism.measured_class(),
                same_seed_repeat_runs=evidence.determinism.runs)
            if evidence.determinism is not None else api.DeterminismReport(
                determinism_class="not_measured", same_seed_repeat_runs=0)
        )
        request = replace(provisional, determinism=measured_determinism)
        report = correctness.evaluate_t0(
            request, evidence, self._t0_evaluator_policy(spec))
        # Retain every durable capture reference across BOTH T0 legs.  The
        # quiet boundary below reads these exact bytes rather than assuming
        # subprocess completion implies sandbox teardown.
        self._t0_capture_refs = tuple(dict.fromkeys(
            tuple(anchor_capture.capture_refs) + tuple(provider.capture_refs)))
        self._t0_request = request
        self._t0_report = report
        self._t0_gate_results = report.gates
        gates = tuple((g.gate_id, g.check.outcome, tuple(g.check.reasons))
                      for g in report.gates)
        return T0Outcome(
            all_pass=all(outcome == schemas.PASS for _gid, outcome, _r in gates),
            gates=gates, report_ref=report.policy_ref, gate_results=report.gates)

    def _evaluation_request(self, spec: CampaignSpec, *, identity: Any,
                            anchor_identity: api.AnchorIdentity,
                            candidate_linkage_sha256: str,
                            determinism: api.DeterminismReport,
                            device_state: Any = None) -> api.EvaluationRequest:
        """The request T0 is evaluated against. Digests MEASURED, not copied.

        `chain.measure_artifact_identity` re-walks the source tree and re-hashes
        the binary rather than reading them off the build receipt: the clean-
        build gate compares receipt against measurement, and filling both sides
        from the receipt turns two of its four sub-checks into `x == x`.
        """
        plan = self._build_state["plan"]
        tree = self._build_state["tree"]
        binary = os.path.join(plan.build_dir.path, "bin", T0_GENERATION_TOOL)
        artifact = chain.measure_artifact_identity(          # seam 2
            source_root=tree.path.path, binary=binary,
            linkage_sha256=candidate_linkage_sha256)
        command = self._construct(spec, arm="candidate")
        self._recipe_receipts[("T0", anchor_identity.tool or T0_GENERATION_TOOL)] = \
            command.receipt
        if spec.calibration is None or spec.calibration.evaluation_authority is None:
            raise RuntimeError("live evaluation request requires accepted typed authority")
        authority = spec.calibration.evaluation_authority
        return api.EvaluationRequest(
            event_id=f"ake-{spec.campaign_id}-{spec.candidate_id}-t0",
            campaign_id=spec.campaign_id, candidate_id=spec.candidate_id,
            tier="T0", backend=spec.backend, phase=command.phase,
            cell_class=command.cell_class, protocol_id=api.PROTOCOL_VERSIONED_ID,
            artifact=artifact, anchor=anchor_identity,
            evaluator=self._evaluator_identity(authority),
            scope_denominator=command.scope_denominator,
            scope_manifest_sha256=identity.snapshot_sha256,
            co_residency="single",
            determinism=determinism,
            metric=command.metric, metric_direction=command.metric_direction,
            reps=spec.reps,
            change_class=("parameter" if spec.proposal is None
                          else spec.proposal["change_class"]),
            anchor_tier="T0", transfer_ratio_to=(), created_at=spec.created_at,
            campaign_controls=authority.campaign_controls,
            calibration=authority.calibration, device_state=device_state,
            suite_seed=spec.suite_seed)

    def _construct(self, spec: CampaignSpec, *, arm: str) -> Any:
        tool = spec.recipe.tool
        # The source identity is always the git worktree/snapshot.  Candidate
        # build artifacts are intentionally outside it (clean-build gate), so
        # do not misidentify the build directory as the source root.
        # Both arms execute out-of-tree build artifacts.  Their source identity
        # is the corresponding git checkout, never the build directory.
        reuse_parameter_anchor = (spec.screening_only and arm == "candidate"
                                  and spec.proposal is not None
                                  and spec.proposal["change_class"] == "parameter")
        root = (MEASUREMENT_REPO if arm == "anchor" or reuse_parameter_anchor
                else self._build_state["tree"].path.path)
        artifact_root = (MEASUREMENT_BUILD_ROOT
                         if arm == "anchor" or reuse_parameter_anchor
                         else self._build_state["plan"].build_dir.path)
        bindir = os.path.join(artifact_root, "bin")
        binary = os.path.join(bindir, tool)
        # Preserve both identities in the binding so execution cannot silently
        # resolve libraries from another tree.  This applies to the anchor as
        # well as the candidate: the anchor build is also external to its git
        # checkout.
        binding = recipes.ToolBinding.for_external_build(
            binary=binary, source_root=root, build_root=artifact_root,
            library_path=bindir)
        return recipes.construct(spec.recipe_id, binding=binding,
                                 params=spec.params_for_arm(arm), arm=arm)

    def create_screening_baseline(self, spec: CampaignSpec, *, output: str | Path) -> dict[str, Any]:
        """Execute and seal exactly three bound anchor calls for a discovery batch."""
        if self._claim_binding is None:
            raise RuntimeError("screening baseline creation requires a held claim")
        target = Path(output)
        if target.exists():
            raise RuntimeError(f"refusing to overwrite existing screening baseline {target}")
        storage.assert_not_scratch(target, what="screening baseline bank")
        anchor = self._construct(spec, arm="anchor")
        policy = self._candidate_sandbox_policy(spec)
        spawner = self._spawner or microbench.SubprocessSpawner(
            workdir_root=policy.writable_root, sandbox_policy=policy)
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        frame = {"recipe_id": spec.recipe_id, "backend": spec.backend,
                 "model_sha256": storage.hash_file(spec.model),
                 "instrument_commit": MEASUREMENT_COMMIT,
                 "production_commit": PRODUCTION_COMMIT,
                 "boot_sha256": schemas.content_hash({"boot_id": boot_id}),
                 "anchor_ggml_iqk": spec.anchor_param_overrides.get("ggml_iqk"),
                 "reps": spec.reps,
                 "n_prompt": (spec.n_prompt if spec.recipe.phase == "prefill" else 0),
                 "n_gen": spec.n_gen}
        witness = screening_baseline.competing_inference_witness()
        if witness["competing"]:
            raise RuntimeError("competing model inference occupies claimed screening compute")
        bank = screening_baseline.create(
            frame=frame, anchor_command=anchor.to_dict(),
            invoke_anchor=lambda: screening_baseline.invoke_command(
                command=anchor, spawner=spawner), anchor_count=3)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(bank.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return {"path": str(target), "baseline_sha256": bank.to_dict()["baseline_sha256"],
                "anchor_invocations": 3, "inference_witness": witness,
                "non_promotable": True}

    # -- 5. the paired blocks ---------------------------------------------

    def admit_t1_after_t0(self, spec: CampaignSpec, tree: Any) -> None:
        """Close, validate and retain T0's event before the first T1 block.

        T0's raw gates are necessary but not sufficient: an unversioned
        protocol citation, a voided window, or an event-schema failure makes
        the T0 evidence unusable.  Previously the event was first constructed
        during finalization, after T1 had already spent timing blocks.  Capture
        the T0 close boundary now and make its emitted, schema-valid PASS event
        the sole admission ticket for T1.
        """
        if self._t0_request is None or not self._t0_gate_results:
            raise T0EvaluationAdmissionRefusal(
                "T0 produced no retained evaluation request and gate set; T1 is refused")
        if self._t0_request.protocol_id != api.PROTOCOL_VERSIONED_ID:
            raise T0EvaluationAdmissionRefusal(
                "T0 event cites " + repr(self._t0_request.protocol_id) +
                ", not exact ratified protocol " + repr(api.PROTOCOL_VERSIONED_ID))

        # This is the T0 window's close, while its claim and worktree are
        # still live.  The final close later belongs to T1 and must not replace
        # the T0 receipt we are about to validate and retain.
        self.close_evaluation_window(spec, tree)
        evidence = [gate.to_dict() for gate in self._t0_gate_results]
        request = self._event_request(self._t0_request, evidence, "t0")
        window = self._window_attestations(
            spec, request,
            raw_evidence_ref="sha256:" + schemas.content_hash(evidence),
            rate_run=None)
        outcome = api.TierDispatcher(gate_runners={
            "T0": _RecordedGateRunner(self._t0_gate_results),
        }).dispatch(request, window, effect=None)
        if outcome.event is None or outcome.event_violations \
                or outcome.verdict.status != api.STATUS_PASS:
            raise T0EvaluationAdmissionRefusal(
                "T0 evaluation event cannot admit T1: "
                f"status={outcome.verdict.status!r} "
                f"blocked={outcome.event_blocked_reason!r} "
                f"violations={list(outcome.event_violations)}")
        self._t0_event_request = request
        self._t0_evaluation_event = outcome.event

    def settle_after_t0(self, spec: CampaignSpec, claim: Any) -> Mapping[str, Any]:
        """Prove T0 has drained and immediately record a fresh host-noise receipt.

        The T1 load gate itself is intentionally unchanged.  What changes is
        the attribution of its input: a full-width T0 leg is known work owned
        by this held claim, and one-minute load is a trailing statistic.  The
        There is deliberately no quiet-period sleep or load ceiling: services,
        builds and filesystem activity are measurement noise, not a reason to
        consume a claimed host by waiting.  A missing teardown, claim witness,
        or competing inference identity remains a hard refusal.
        """
        if self._t0_capture_archive is None or not self._t0_capture_refs:
            raise RuntimeError(
                "T0 passed without retained durable sandbox captures; cannot establish "
                "the post-T0 quiet boundary before T1")
        teardown_receipts = []
        for ref in self._t0_capture_refs:
            capture = self._t0_capture_archive.get(ref)
            teardown = capture.sandbox_teardown
            if not isinstance(teardown, Mapping) \
                    or teardown.get("verified_empty") is not True \
                    or teardown.get("removed") is not True:
                raise RuntimeError(
                    f"T0 capture {ref!r} lacks a verified removed sandbox cgroup; "
                    "cannot attribute the following load decay to completed owned work")
            teardown_receipts.append({"capture_ref": ref, "teardown": dict(teardown)})

        policy = microbench.HostStatePolicy(
            nominal_khz=self._nominal_khz,
            require_load=False,
            require_package_power=(spec.backend == BACKEND_CPU))
        held = microbench.CpuRegionClaimAdapter(claim, cpu_list=spec.cpu_list)
        samples = []
        for index in range(1):
            attestation = held.attest()
            if not attestation.held:
                raise RuntimeError(
                    f"post-T0 quiet sample {index + 1}/{POST_T0_QUIET_SAMPLES} has no "
                    f"held claim witness: {attestation.check.outcome} — "
                    f"{'; '.join(attestation.check.reasons)}")
            state = self._read_host_state(cpu_list=spec.cpu_list)
            load = policy.check_load(state, cpu_count=len(state.khz_by_cpu) or 1)
            witness = screening_baseline.competing_inference_witness()
            if witness["competing"]:
                raise RuntimeError("competing model inference occupies claimed AutoKernel compute")
            samples.append({
                "index": index + 1,
                "host_state": state.to_dict(),
                "claim_attestation": {
                    "claim_id": attestation.claim_id, "holder": attestation.holder,
                    "cpu_list": attestation.cpu_list,
                    "observed_at": attestation.observed_at,
                    "outcome": attestation.check.outcome,
                    "reasons": list(attestation.check.reasons),
                },
                "load": {"outcome": load.outcome, "reasons": list(load.reasons)},
                "inference_witness": witness,
            })
        receipt = {
            "schema": "epyc.autokernel.post_t0_ready_boundary.v1",
            "campaign_id": spec.campaign_id,
            "ordinary_load_policy": "recorded_not_blocking",
            "required_samples": 1,
            "t0_sandbox_teardowns": teardown_receipts,
            "samples": samples,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        receipt["receipt_id"] = "akq-" + schemas.content_hash(receipt)[:24]
        # The event is the durable receipt, not merely a hash folded into a
        # later evaluation event.  A failure after this boundary (including a
        # T1 refusal) must not erase the proof that T0's sandbox children were
        # gone and that all three fresh samples were quiet under this claim.
        self._journal_post_t0_quiet_boundary(spec, receipt)
        self._post_t0_settlement = receipt
        return receipt

    @staticmethod
    def _journal_post_t0_quiet_boundary(spec: CampaignSpec,
                                        receipt: Mapping[str, Any]) -> None:
        if not spec.journal_root:
            raise RuntimeError("post-T0 quiet boundary needs an executing campaign journal")
        root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
        book = journal_module.Journal(root, campaign_id=spec.campaign_id)
        book.initialize()
        receipt_id = receipt["receipt_id"]
        with book.write_lock():
            prior = [entry for entry in book.read_all()
                     if entry.kind == journal_module.KIND_POST_T0_QUIET_BOUNDARY
                     and entry.record_id == receipt_id]
            if prior:
                if schemas.content_hash(prior[0].payload) != schemas.content_hash(receipt):
                    raise RuntimeError(
                        f"post-T0 quiet receipt {receipt_id!r} names different bytes")
                return
            book.append(journal_module.KIND_POST_T0_QUIET_BOUNDARY, receipt)

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
        if spec.screening_only:
            assert spec.screening_baseline is not None
            sandbox_policy = self._candidate_sandbox_policy(spec)
            spawner = self._spawner or microbench.SubprocessSpawner(
                workdir_root=sandbox_policy.writable_root, sandbox_policy=sandbox_policy)
            if spec.backend == BACKEND_CPU:
                spawner = inference_window.WindowedSpawner(
                    spawner, inference_window.InferenceCallWindow(timeout_s=600.0))
            frame = {"recipe_id": spec.recipe_id, "backend": spec.backend,
                     "model_sha256": storage.hash_file(spec.model),
                     "instrument_commit": MEASUREMENT_COMMIT,
                     "production_commit": PRODUCTION_COMMIT,
                     "boot_sha256": schemas.content_hash({"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()}),
                     "anchor_ggml_iqk": spec.anchor_param_overrides.get("ggml_iqk"),
                     "reps": spec.reps,
                     "n_prompt": (spec.n_prompt if spec.recipe.phase == "prefill" else 0),
                     "n_gen": spec.n_gen}
            witness = screening_baseline.competing_inference_witness()
            report = screening_baseline.screen(
                bank=spec.screening_baseline, frame=frame,
                invoke_candidate=lambda: screening_baseline.invoke_command(
                    command=candidate_cmd, spawner=spawner),
                competing_inference=bool(witness["competing"]),
                candidate_command=candidate_cmd.to_dict())
            report["inference_witness"] = witness
            self._screening_report = report
            center = float(report["baseline_center"])
            return tuple(CandidateOnlyObservation(index, center, value)
                         for index, value in enumerate(report["candidate_samples"]))
        anchor_cmd = self._construct(spec, arm="anchor")
        anchor_identity = self._anchor_identity_for_bench(spec)
        self._recipe_receipts[("T1", anchor_identity.tool or "llama-bench")] = \
            candidate_cmd.receipt
        self._t1_request = self._t1_evaluation_request(
            spec, command=candidate_cmd, anchor=anchor_identity)
        plan = microbench.MicrobenchPlan(
            recipe_id=spec.recipe_id, candidate_id=spec.candidate_id,
            campaign_seed=spec.schedule_seed,
            matched_experiment_id=spec.matched_experiment_id,
            candidate_binding=candidate_cmd.binding,
            anchor_binding=anchor_cmd.binding,
            anchor=anchor_identity,
            candidate_instrument_root=self._build_state["tree"].path.path,
            anchor_instrument_root=MEASUREMENT_REPO,
            params=spec.bench_params,
            candidate_param_overrides=spec.candidate_param_overrides,
            anchor_param_overrides=spec.anchor_param_overrides,
            unit_param_overrides=spec.ranked_unit_param_overrides,
            anti_short_circuit_units=spec.anti_short_circuit_units,
            base_blocks=spec.blocks,
            pairs_per_block=spec.fresh_pairs_per_block,
            unit_ids=spec.ranked_unit_ids,
            physical_envelopes=spec.physical_envelopes,
            stratum=api.STRATUM_SELECTION)
        # The complete base segment is known before any block is spent, so its
        # temporal-order counterbalance is a pre-spend invariant.  A 15-block
        # matched pair must be 8/7, never a random 10/5 split that lets thermal
        # drift favour one campaign arm.
        drawn = plan.schedule().orders(spec.blocks)
        imbalance = abs(2 * drawn.count(drawn[0]) - len(drawn))
        if imbalance > 1:
            raise RuntimeError(
                f"the order schedule for campaign key {plan.matched_experiment_id or plan.candidate_id!r} "
                f"has base imbalance {imbalance} across {spec.blocks} blocks; refusing before "
                "measurement because interleaved order must be counterbalanced within one.")
        sandbox_policy = self._candidate_sandbox_policy(spec)
        spawner = self._spawner or microbench.SubprocessSpawner(
            workdir_root=sandbox_policy.writable_root,
            sandbox_policy=sandbox_policy,
            device_sampler=(
                device_sampler.RocmSmiSampler(device_index=spec.device_index)
                if spec.backend == BACKEND_GPU else None))
        if spec.backend == BACKEND_CPU:
            spawner = inference_window.WindowedSpawner(
                spawner, inference_window.InferenceCallWindow(timeout_s=600.0))
        runner = microbench.MicrobenchRunner(
            claim=self._claim_binding.microbench_claim,
            policy=microbench.HostStatePolicy(
                nominal_khz=self._nominal_khz,
                require_load=False,
                require_package_power=(spec.backend == BACKEND_CPU)),
            spawner=spawner,
            host_state=self._read_host_state,
            run_ledger=self._completed_run_ledger(spec))
        run = runner.run(plan)
        self._microbench_run = run
        pairs = pairs_from_run(run)
        if spec.backend == BACKEND_CPU:
            self._require_inference_window_receipts(run)
        return pairs

    @staticmethod
    def _require_inference_window_receipts(run: Any) -> None:
        """Refuse a CPU result unless every model call proves lock release.

        The host-wide region claim may be borrowed by GPU discovery only because
        CPU inference is serialized at the much narrower call boundary.  That is
        an evidence-bearing contract, not an implementation detail: a completed
        strict run without one released receipt per invocation must never rank.
        """
        raw = run.raw_vector()
        blocks = raw.get("blocks") if isinstance(raw, Mapping) else None
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError(
                "completed CPU run has no blocks carrying inference-window receipts")
        checked = 0
        for block_index, block in enumerate(blocks):
            invocations = block.get("invocations") if isinstance(block, Mapping) else None
            if not isinstance(invocations, list) or not invocations:
                raise RuntimeError(
                    f"CPU block {block_index} has no inference-windowed invocations")
            for invocation_index, invocation in enumerate(invocations):
                spawn = (invocation.get("spawn")
                         if isinstance(invocation, Mapping) else None)
                receipt = (spawn.get("inference_window_receipt")
                           if isinstance(spawn, Mapping) else None)
                location = f"block {block_index} invocation {invocation_index}"
                if not isinstance(receipt, Mapping):
                    raise RuntimeError(
                        f"CPU {location} lacks an inference-window receipt")
                waited_s = receipt.get("waited_s")
                held_s = receipt.get("held_s")
                if (receipt.get("schema")
                        != "epyc.autokernel.inference_call_window.v1"
                        or receipt.get("lock_path")
                        != str(inference_window.DEFAULT_LOCK_PATH)
                        or receipt.get("scope") != "model_load_and_inference_only"
                        or receipt.get("released") is not True
                        or not isinstance(waited_s, (int, float))
                        or not isfinite(waited_s) or waited_s < 0
                        or not isinstance(held_s, (int, float))
                        or not isfinite(held_s) or held_s < 0):
                    raise RuntimeError(
                        f"CPU {location} has a malformed or unreleased "
                        "inference-window receipt")
                checked += 1
        if checked == 0:  # defensive: the non-empty checks above imply this.
            raise RuntimeError("completed CPU run has no inference-window receipts")

    def _t1_evaluation_request(self, spec: CampaignSpec, *, command: Any,
                               anchor: api.AnchorIdentity) -> api.EvaluationRequest:
        """Bind the T1 request to the benchmark binary's own linkage identity."""
        if spec.calibration is None or spec.calibration.evaluation_authority is None:
            raise RuntimeError("T1 evaluation requires accepted typed authority")
        plan = self._build_state["plan"]
        tree = self._build_state["tree"]
        bindir = os.path.join(plan.build_dir.path, "bin")
        capture = t0_provider.capture_anchor_identity(
            anchor=t0_provider.AnchorBuild(
                worktree=tree.path.path, source_commit=tree.head_commit(),
                binary=os.path.join(bindir, "llama-bench"), library_path=bindir),
            tools=self._t0_tools(), runner=t0_provider.SubprocessRunner(),
            base_env=tuple(sorted(command.env.items())),
            parameter_env=spec.t0_parameter_env_for_arm("candidate"))
        artifact = chain.measure_artifact_identity(
            source_root=tree.path.path, binary=os.path.join(bindir, "llama-bench"),
            linkage_sha256=capture.linkage_sha256)
        authority = spec.calibration.evaluation_authority
        determinism = (
            self._t0_request.determinism if self._t0_request is not None
            else api.DeterminismReport(
                determinism_class="not_measured", same_seed_repeat_runs=0)
        )
        return api.EvaluationRequest(
            event_id=f"ake-{spec.campaign_id}-{spec.candidate_id}-t1",
            campaign_id=spec.campaign_id, candidate_id=spec.candidate_id,
            tier="T1", backend=spec.backend, phase=command.phase,
            cell_class=command.cell_class, protocol_id=api.PROTOCOL_VERSIONED_ID,
            artifact=artifact, anchor=anchor,
            evaluator=self._evaluator_identity(authority),
            scope_denominator=command.scope_denominator,
            scope_manifest_sha256=(
                self._t0_request.scope_manifest_sha256
                if self._t0_request is not None else artifact.source_sha256),
            co_residency="single", determinism=determinism,
            metric=command.metric, metric_direction=command.metric_direction,
            reps=spec.reps,
            change_class=("parameter" if spec.proposal is None
                          else spec.proposal["change_class"]),
            anchor_tier="T1", transfer_ratio_to=(), created_at=spec.created_at,
            campaign_controls=authority.campaign_controls,
            calibration=authority.calibration, device_state=None,
            suite_seed=spec.suite_seed)

    @staticmethod
    def _candidate_sandbox_policy(spec: CampaignSpec) -> sandbox.SandboxPolicy:
        """The only live-campaign route to a process runner.

        The evaluator creates the directory; the candidate receives write
        authority over exactly that directory after Landlock activates.  The
        journal itself is a sibling and therefore remains read-only to code the
        loop authored.  A missing durable journal root is already illegal for
        execution and is refused here again so no caller can fall back to /tmp.
        """
        if not spec.journal_root:
            raise RuntimeError(
                "candidate sandbox needs the executing campaign's durable journal root")
        journal_root = storage.assert_not_scratch(
            spec.journal_root, what="campaign journal root")
        root = os.path.realpath(os.path.join(
            journal_root, spec.campaign_id, "candidate-sandbox"))
        if not storage._under(root, journal_root):
            raise RuntimeError("candidate sandbox escaped the campaign journal root")
        for production in worktree.frozen_tree_paths():
            if storage._under(root, production) or storage._under(production, root):
                raise RuntimeError(
                    f"candidate sandbox {root!r} touches frozen tree {production!r}")
        os.makedirs(root, mode=0o700, exist_ok=True)
        limits = sandbox.ResourceLimits(cpu_time_s=max(60, min(spec.max_hold_s, 8 * 3600)))
        return sandbox.SandboxPolicy(writable_root=root, limits=limits)

    def _completed_run_ledger(self, spec: CampaignSpec) -> microbench.CompletedRunLedger:
        """The executing path cannot spend a benchmark leg without durability."""
        if not spec.journal_root:
            raise RuntimeError(
                "an executing campaign requires --journal-root before paired blocks; the "
                "completed-run key cannot be machine-enforced in volatile memory")
        root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
        return microbench.CompletedRunLedger(
            journal_module.Journal(root, campaign_id=spec.campaign_id),
            campaign_id=spec.campaign_id)

    def _anchor_identity_for_bench(self, spec: CampaignSpec) -> api.AnchorIdentity:
        capture = t0_provider.capture_anchor_identity(
            anchor=self._measurement_anchor_build("llama-bench"),
            tools=self._t0_tools(),
            runner=t0_provider.SubprocessRunner(),
            base_env=tuple(sorted(self._construct(spec, arm="anchor").env.items())),
            parameter_env=spec.t0_parameter_env_for_arm("anchor"))
        binding = chain.bind_anchor(capture, tool="llama-bench")
        if self._t0_anchor_binding is None:
            raise RuntimeError(
                f"the T1 anchor was requested before T0 captured its {T0_GENERATION_TOOL} anchor")
        same_build = chain.check_anchor_build_is_one_build(
            (self._t0_anchor_binding, binding))
        if same_build.outcome != schemas.PASS:
            raise chain.AnchorNotOneAnchor("; ".join(same_build.reasons))
        return binding.identity

    # -- 6. teardown -------------------------------------------------------

    def teardown_worktree(self, spec: CampaignSpec, tree: Any) -> Any:
        snapshot_receipt = None
        if self._build_snapshot is not None:
            snapshot_receipt = worktree.teardown_worktree(
                self._build_snapshot, witness_trees=list(worktree.frozen_tree_paths()))
            self._build_snapshot = None
        receipt = worktree.teardown_worktree(
            tree, witness_trees=list(worktree.frozen_tree_paths()))
        return {"snapshot": None if snapshot_receipt is None else snapshot_receipt.to_dict(),
                "campaign": receipt.to_dict()}

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

    def _window_attestations(self, spec: CampaignSpec, request: api.EvaluationRequest,
                             *, raw_evidence_ref: str,
                             rate_run: Optional[microbench.MicrobenchRun]
                             ) -> api.WindowAttestations:
        authority = spec.calibration.evaluation_authority
        if authority is None:
            raise RuntimeError("evaluation window has no typed live authority")
        unknown = lambda reason: schemas.Check(schemas.COULD_NOT_CHECK, (reason,))
        release_material = (self._claim_release_receipt
                            if isinstance(self._claim_release_receipt, Mapping) else
                            {"unavailable": "claim release receipt was not retained"})
        claim_ref = "sha256:" + schemas.content_hash(release_material)
        open_check = self._claim_open_check or unknown(
            "resource claims were not freshly checked at window open")
        close_check = self._claim_close_check or unknown(
            "resource claims were not freshly checked at window close")
        holder_check = self._claim_same_holder_check or unknown(
            "resource-claim holder continuity was not checked")
        concurrent = schemas.Check.worst_of((
            self._preflight_check or unknown(
                "concurrent-inference preflight was not retained at open"),
            self._no_concurrent_close or unknown(
                "concurrent-inference preflight was not repeated at close"),
        ))
        preflight_material = {
            "open": self._preflight_open_receipt,
            "close": self._preflight_close_receipt,
        }
        host_material = {
            "open": None if self._host_open is None else self._host_open.to_dict(),
            "close": None if self._host_close is None else self._host_close.to_dict(),
            "post_t0_quiet_boundary": self._post_t0_settlement,
            "raw_evidence_ref": raw_evidence_ref,
        }
        host_ref = "sha256:" + schemas.content_hash(host_material)
        host_health = self._host_health_close or unknown(
            "host health was not re-attested at window close")
        anchor_tool = request.anchor.tool or T0_GENERATION_TOOL
        anchor_identity_check = self._anchor_close_checks.get(anchor_tool) or unknown(
            f"anchor identity for {anchor_tool} was not re-captured at window close")
        if rate_run is not None:
            anchor_values = [median(block.anchor_samples)
                             for block in rate_run.paired_blocks()]
            anchor_value = median(anchor_values)
            low, high = authority.calibration.anchor_gate_band
            anchor_band = schemas.Check(
                schemas.PASS if low <= anchor_value <= high else schemas.FAIL,
                (f"anchor median {anchor_value:.9g} vs calibrated band [{low:.9g}, "
                 f"{high:.9g}]",))
            anchor_gate = schemas.Check.worst_of((
                anchor_band,
                anchor_identity_check,
            ))
            order_check = rate_run.order_control
            strata = schemas.Check(
                schemas.PASS if all(block.stratum == api.STRATUM_SELECTION
                                    for block in rate_run.paired_blocks())
                else schemas.FAIL,
                ("all completed blocks are in the selection stratum",))
            rule = schemas.Check(
                schemas.PASS if len(rate_run.paired_blocks()) == spec.blocks
                and len(rate_run.paired_blocks()) <=
                authority.campaign_controls.max_blocks_per_candidate else schemas.FAIL,
                (f"realized {len(rate_run.paired_blocks())} of precommitted "
                 f"{spec.blocks} blocks",))
            order_seed = rate_run.plan.campaign_seed
        else:
            anchor_gate = schemas.Check.worst_of((
                schemas.Check(
                    schemas.PASS if self._t0_gate_results else schemas.COULD_NOT_CHECK,
                    ("T0 anchor-bound gate set completed",)),
                anchor_identity_check,
            ))
            order_check = schemas.Check(
                schemas.COULD_NOT_CHECK, ("no rate-order schedule applies to T0",))
            strata = schemas.Check(
                schemas.COULD_NOT_CHECK, ("no selection stratum applies to T0",))
            rule = schemas.Check(schemas.PASS, ("T0 precedes the speed stopping rule",))
            order_seed = f"{spec.campaign_id}/t0"
        recipe_receipt = self._recipe_receipts.get((request.tier, anchor_tool))
        return api.WindowAttestations(
            resource_claim_receipt=claim_ref,
            resource_claim_open=open_check, resource_claim_close=close_check,
            resource_claim_same_holder=holder_check,
            no_concurrent_inference=concurrent,
            preflight_attestation_ref="sha256:" + schemas.content_hash(
                preflight_material),
            host_receipt=host_ref, host_health=host_health,
            anchor_at_open=request.anchor,
            anchor_at_close=self._anchors_at_close.get(anchor_tool),
            anchor_gate=anchor_gate,
            evaluator_bundle=self._evaluator_close_check or unknown(
                "evaluator bundle was not re-hashed at window close"),
            runtime_source_label=self._runtime_close_check or unknown(
                "runtime source label was not re-read at window close"),
            recipe=recipe_receipt,
            storage_open=self._storage_open or schemas.Check(
                schemas.COULD_NOT_CHECK, ("storage headroom was not retained at open",)),
            storage_close=self._storage_close or unknown(
                "storage headroom was not re-read at window close"), strata=strata,
            stopping_rule_id=authority.stopping_rule_id,
            rule_immutability=rule, order_randomized=order_check,
            order_seed=order_seed, aa_cadence=authority.aa_cadence,
            controls=authority.controls, calibration=schemas.Check(
                schemas.PASS, (authority.evidence_ref,)),
            control_definitions_immutable=authority.control_definitions_immutable,
            raw_evidence_ref=raw_evidence_ref)

    @staticmethod
    def _event_request(request: api.EvaluationRequest, evidence: Any,
                       suffix: str) -> api.EvaluationRequest:
        digest = schemas.content_hash(evidence)
        return replace(request, event_id=(
            f"ake-{request.campaign_id}-{request.candidate_id}-{suffix}-{digest[:16]}"))

    @staticmethod
    def _parameter_intervention_gate(
            t0_gates: Sequence[api.GateResult]) -> api.GateResult:
        """Bind a parameter T1 effect to T0's canonical real-path proof.

        The T0 provider owns the gate's identity.  Do not reconstruct a
        historical spelling here: that would turn a passing real-path proof
        into an uncheckable T1 mechanism gate at the campaign/archive seam.
        """
        dispatch = next((gate for gate in t0_gates
                         if gate.gate_id == correctness.GID_NO_FALLBACK), None)
        check = (dispatch.check if dispatch is not None else schemas.Check(
            schemas.COULD_NOT_CHECK,
            ("T0 emitted no no-fallback dispatch identity",)))
        return api.GateResult(
            gate_id="t1.parameter_intervention_explained",
            gate_class=api.GATE_MECHANISM,
            check=check,
            evidence_ref=None if dispatch is None else dispatch.evidence_ref,
            notes=("the registered GGML_IQK arm-local intervention and its "
                   "no-fallback real-path trace are bound to this T1 effect",),
        )

    def _evaluation_events(self, spec: CampaignSpec) -> tuple:
        """Build T0 then T1 from retained executed evidence; never from display pairs."""
        events = []
        t0_request = None
        if self._t0_evaluation_event is not None:
            if self._t0_event_request is None:
                raise RuntimeError("retained T0 event has no matching request")
            t0_request = self._t0_event_request
            events.append(self._t0_evaluation_event)
        elif self._t0_request is not None and self._t0_gate_results:
            evidence = [gate.to_dict() for gate in self._t0_gate_results]
            t0_request = self._event_request(self._t0_request, evidence, "t0")
            window = self._window_attestations(
                spec, t0_request,
                raw_evidence_ref="sha256:" + schemas.content_hash(evidence),
                rate_run=None)
            outcome = api.TierDispatcher(gate_runners={
                "T0": _RecordedGateRunner(self._t0_gate_results),
            }).dispatch(t0_request, window, effect=None)
            if outcome.event is None or outcome.event_violations:
                raise RuntimeError(
                    "T0 evaluation event was not schema-valid: "
                    f"blocked={outcome.event_blocked_reason!r} "
                    f"violations={list(outcome.event_violations)}")
            events.append(outcome.event)
        if self._microbench_run is not None and self._t1_request is not None:
            blocks = self._microbench_run.paired_blocks()
            raw_ref = "sha256:" + schemas.content_hash(
                [block.to_list() for block in blocks])
            anchor = replace(
                self._t1_request.anchor,
                measurement_event_ids=(() if t0_request is None
                                       else (t0_request.event_id,)))
            request = replace(self._t1_request, anchor=anchor)
            request = self._event_request(
                request, [block.to_list() for block in blocks], "t1")
            window = self._window_attestations(
                spec, request, raw_evidence_ref=raw_ref,
                rate_run=self._microbench_run)
            effect = control_runner.reduce_live_blocks(
                request, blocks, spec.calibration.evaluation_authority)
            t1_gates = list(self._t0_gate_results)
            if spec.proposal is not None \
                    and spec.proposal.get("change_class") == "parameter":
                t1_gates.append(self._parameter_intervention_gate(
                    self._t0_gate_results))
            outcome = api.TierDispatcher(gate_runners={
                "T1": _RecordedGateRunner(t1_gates),
            }).dispatch(request, window, effect=effect)
            if outcome.event is None or outcome.event_violations:
                raise RuntimeError(
                    "T1 evaluation event was not schema-valid: "
                    f"blocked={outcome.event_blocked_reason!r} "
                    f"violations={list(outcome.event_violations)}")
            if not spec.model:
                raise RuntimeError("prospective belief capture requires the measured model")
            event = control_runner.attach_belief_capture(
                outcome.event, effect_scale="relative",
                model_id=os.path.basename(spec.model),
                model_sha256=storage.hash_file(spec.model),
                producer_sha256=request.evaluator.bundle_sha256)
            violations = schemas.validate_evaluation_event(event)
            if violations:
                raise RuntimeError(
                    "belief-capture event failed schema validation: "
                    + "; ".join(violations))
            events.append(event)
        return tuple(events)

    def prepare_durable_records(self, spec: CampaignSpec, *, state: str,
                                decision: Optional[AcceptDecision]) -> None:
        """Materialize evaluation and candidate bytes while the built snapshot lives."""
        if self._cached_evaluation_events is not None:
            return
        events = self._evaluation_events(spec)
        self._cached_evaluation_events = events
        if spec.screening_only:
            # A screen may journal its raw terminal observation, but no
            # candidate record means no completed-campaign adapter, archive,
            # champion, or promotion path can consume it.
            return
        # Evaluation events are durable on every terminal path, but candidate
        # records and least-commitment verdicts are *decision-derived*.  An
        # error/refusal has no AcceptDecision, so trying to materialize it
        # would turn the primary campaign failure into a misleading secondary
        # CapturePlanError.  Keep the already-built event(s) cached for
        # journaling, and derive records only from a decided terminal result.
        if state != STATE_DECIDED:
            return
        if self._build_identity is None or self._build_snapshot is None \
                or self._t0_request is None or spec.proposal is None:
            return
        event_ids = tuple(event["event_id"] for event in events)
        event = events[-1] if events else None
        evaluator = self._t0_request.evaluator
        # `decide()` owns the precommitted paired-block rule, but it runs before
        # the final T1 event is reduced after the window closes.  The evaluator
        # can truthfully withhold the speed rank at that later boundary (for
        # example, because final close evidence voids the window or the effect
        # misses MDE).  A durable ``evaluating`` candidate is eligible for the
        # completed-campaign adapter to bank, so a raw KEEP cannot be enough.
        #
        # Read the exact final T1 projection which the adapter consumes.  A
        # missing field is not a historical default: it proves no rank was
        # journaled, and therefore cannot preserve a candidate for banking.
        t1_events = tuple(item for item in events
                          if isinstance(item, Mapping) and item.get("tier") == "T1")
        final_t1 = t1_events[-1] if t1_events else None
        performance = (final_t1.get("performance")
                       if isinstance(final_t1, Mapping) else None)
        discipline = (performance.get("search_discipline")
                      if isinstance(performance, Mapping) else None)
        t1_speed_rank_admissible = (
            discipline.get("speed_rank_admissible") is True
            if isinstance(discipline, Mapping) else False)
        keep_is_rankable = bool(decision is not None and decision.keep
                                and t1_speed_rank_admissible)
        status = "evaluating" if keep_is_rankable else "rejected"
        derived_tokens = (
            tuple(f"file:{path}" for path in self._source_application.actual_files)
            if self._source_application is not None else ("flag:GGML_IQK",))
        derived_verdicts = {
            "campaign_state": state,
            "accept_decision": None if decision is None else decision.to_dict(),
            "final_t1_speed_rank_admissible": t1_speed_rank_admissible,
            "keep_is_rankable": keep_is_rankable,
        }
        if spec.least_commitment_plan is not None:
            derived_verdicts["least_commitment"] = least_commitment_capture.materialize(
                spec.least_commitment_plan, decision=decision,
                calibration=spec.calibration,
                executed_factors=spec.matched_factor_frame)
        self._cached_candidate_record = candidate_record.build_candidate_record(
            proposal=spec.proposal, candidate_id=spec.candidate_id,
            campaign_id=spec.campaign_id, production_base_commit=PRODUCTION_COMMIT,
            instrument_commit=MEASUREMENT_COMMIT,
            source_commit=self._build_snapshot.head_commit(),
            actor=self._build_snapshot, identity=self._build_identity,
            build_result=self._build_state.get("result"),
            source_application=self._source_application, status=status,
            evaluator_id=evaluator.id,
            evaluator_bundle_sha256=evaluator.bundle_sha256,
            evaluator_runtime_source_label_ref=evaluator.runtime_source_label_ref,
            resource_claim_receipt=(
                event["resource_claim_receipt"] if event is not None
                else schemas.content_hash(self._claim_close_receipt or {})),
            host_receipt=(event["host_receipt"] if event is not None
                          else schemas.content_hash({"open": str(self._host_open),
                                                    "close": str(self._host_close)})),
            evaluation_event_ids=event_ids,
            derived_surface_tokens=derived_tokens,
            dispatch_predicates=(),
            protocol_ids=tuple(sorted({event["claim_grammar"]["protocol_id"]
                                       for event in events})) or ("P-AK-SEARCH-1/v1",),
            same_seed_repeat_runs=self._t0_request.determinism.same_seed_repeat_runs,
            derived_verdicts=derived_verdicts,
            created_at=spec.created_at)

    def journal_evaluation(self, spec: CampaignSpec, result: Any) -> tuple:
        """Append prospective events idempotently, before terminal STOP_STATE."""
        if not spec.journal_root:
            raise RuntimeError("executed evaluation has no durable journal root")
        root = storage.assert_not_scratch(spec.journal_root, what="campaign journal root")
        book = journal_module.Journal(root, campaign_id=spec.campaign_id)
        book.initialize()
        appended = []
        events = (self._cached_evaluation_events if self._cached_evaluation_events is not None
                  else self._evaluation_events(spec))
        if self._t0_started and not events and getattr(result, "error", None):
            refusal = {
                "campaign_id": spec.campaign_id,
                "candidate_id": spec.candidate_id,
                "stage": "run_t0.before_event_emission",
                "error": result.error,
                "rate_measured": False,
            }
            refusal_id = "akt0r-" + schemas.content_hash(refusal)[:24]
            with book.write_lock():
                prior = [entry for entry in book.read_all()
                         if entry.kind == journal_module.KIND_T0_REFUSAL
                         and entry.record_id == refusal_id]
                if prior:
                    if schemas.content_hash(prior[0].payload) != schemas.content_hash(refusal):
                        raise RuntimeError(
                            f"T0 refusal id {refusal_id!r} names different bytes")
                    appended.append(prior[0].event_id)
                else:
                    appended.append(book.append(
                        journal_module.KIND_T0_REFUSAL, refusal,
                        record_id=refusal_id).event_id)
        for event in events:
            with book.write_lock():
                prior = [entry for entry in book.read_all()
                         if entry.kind == journal_module.KIND_EVALUATION_EVENT
                         and entry.record_id == event["event_id"]]
                if prior:
                    if schemas.content_hash(prior[0].payload) != schemas.content_hash(event):
                        raise RuntimeError(
                            f"evaluation id {event['event_id']!r} names different bytes")
                    appended.append(prior[0].event_id)
                    continue
                appended.append(book.append(
                    journal_module.KIND_EVALUATION_EVENT, event).event_id)
        if self._cached_candidate_record is not None:
            appended.append(candidate_record.append_candidate_idempotent(
                book, self._cached_candidate_record,
                kind=journal_module.KIND_CANDIDATE_RECORDED))
        return tuple(appended)

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
        # The event above is the primary record and is already fsynced.  The
        # dashboard is a derived view: failure to refresh it is loud, but must
        # not turn a successfully journaled campaign into `journal_error`.
        try:
            dashboard.export_terminal_entry(entry)
        except Exception as exc:  # derived presentation failure, never record loss
            print(f"WARNING: terminal result journaled as {entry.event_id}, but the "
                  f"dashboard export failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
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
    #: Candidate-only discovery receipt. Present only for SCREENING_ONLY and
    #: explicitly carries zero anchor invocations/non-promotable authority.
    screening_report: Optional[Mapping[str, Any]] = None

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
            "screening_only": self.spec.screening_only,
            "non_promotable": self.spec.screening_only,
            "journal_error": self.journal_error,
            "screening_report": (None if self.screening_report is None
                                  else dict(self.screening_report)),
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
        calibration_gate = getattr(ops, "calibration_gate", None)
        if executes and callable(calibration_gate):
            pre = calibration_gate(spec)
            if pre.outcome != schemas.PASS:
                state = STATE_PREFLIGHT_REFUSED
                error = "; ".join(pre.reasons)
                return _finish(
                    spec, ops, ledger, state=state, t0=t0, decision=decision,
                    pairs=pairs, pre=pre, error=error, tree=tree,
                )
        if spec.proposal is not None:
            ops.record_proposal(spec)
        pre = ops.preflight(spec)
        if pre.outcome == schemas.FAIL:
            state = STATE_PREFLIGHT_REFUSED
            error = "; ".join(pre.reasons)
            return _finish(spec, ops, ledger, state=state, t0=t0, decision=decision,
                           pairs=pairs, pre=pre, error=error, tree=tree)

        claim = ops.acquire_claim(spec)
        ledger.hold("cpu_region_claim", lambda: ops.release_claim(claim))

        parameter_screen = (
            spec.screening_only
            and spec.proposal is not None
            and spec.proposal["change_class"] == "parameter"
        )
        if parameter_screen:
            # A registered runtime-parameter screen changes no source or build
            # material.  Both arms must execute the exact instrument sealed by
            # the baseline bank; creating a candidate worktree/build is wasted
            # work and, worse, gives the two arms different artifact identities.
            # ``HostOps._construct`` projects the candidate command from that
            # one instrument and changes only the registered parameter surface.
            tree = None
            build = None
        else:
            tree = ops.create_worktree(spec)
            captured_tree = tree
            ledger.hold("campaign_worktree",
                        lambda: ops.teardown_worktree(spec, captured_tree))

            ops.apply_candidate(spec, tree)
            build = ops.build(spec, tree)

        if spec.screening_only:
            # Screening intentionally reuses the already-selected measurement
            # build path but does not pay T0 or its post-work quiet boundary.
            # The resulting observation is non-promotable by construction.
            t0 = None
        else:
            t0 = ops.run_t0(spec, build)
        if not spec.screening_only and executes and not t0.all_pass:
            # STOP. No speed number is computed at all. This is the ONE branch
            # a composition pass does not take, because in a composition pass
            # T0 was not executed and therefore did not fail — it produced
            # nothing. Every other step is walked identically, and
            # `test_campaign` pins the two call orders against each other so the
            # loop cannot acquire a second spelling.
            state = STATE_T0_FAILED
            return _finish(spec, ops, ledger, state=state, t0=t0, decision=None,
                           pairs=(), pre=pre, error=None, tree=tree)

        settle = getattr(ops, "settle_after_t0", None)
        if not spec.screening_only and callable(settle):
            # A T0 PASS is not yet a T1 run-open condition: its own full-width
            # work remains in load1.  The concrete HostOps boundary preserves
            # all teardown receipts and proves quietness under the held claim.
            settle(spec, claim)

        admit_t1 = getattr(ops, "admit_t1_after_t0", None)
        if not spec.screening_only and executes and callable(admit_t1):
            try:
                admit_t1(spec, tree)
            except T0EvaluationAdmissionRefusal as exc:
                state = STATE_T0_FAILED
                return _finish(spec, ops, ledger, state=state, t0=t0, decision=None,
                               pairs=(), pre=pre, error=str(exc), tree=tree)

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
        decision = (screening_decision(pairs, blocks_precommitted=spec.blocks)
                    if spec.screening_only else decide(
                        pairs, t0=t0, blocks_precommitted=spec.blocks,
                        drift_bound=spec.drift_bound,
                        contribution_floor=spec.contribution_floor,
                        calibration_evidence_ref=(
                            None if spec.calibration is None else spec.calibration.evidence_ref),
                    ))
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

    # The last truthful time to attest a measurement window is while its claim
    # and worktree are still held.  Release receipts are evidence that cleanup
    # happened; they are not substitutes for re-reading the live resources.
    close_window = getattr(ops, "close_evaluation_window", None)
    if bool(getattr(ops, "executes", True)) and callable(close_window):
        try:
            close_window(spec, tree)
        except BaseException as exc:  # noqa: BLE001 - release must still run
            state = STATE_ERROR
            error = "; ".join(x for x in (
                error, f"close_evaluation_window: {type(exc).__name__}: {exc}") if x)

    prepare_records = getattr(ops, "prepare_durable_records", None)
    if bool(getattr(ops, "executes", True)) and callable(prepare_records):
        try:
            prepare_records(spec, state=state, decision=decision)
        except BaseException as exc:  # noqa: BLE001 - release must still run
            state = STATE_ERROR
            error = "; ".join(x for x in (
                error, f"prepare_durable_records: {type(exc).__name__}: {exc}") if x)

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
        screening_report=getattr(ops, "_screening_report", None),
        error="\n".join(x for x in (error, traceback_text) if x) or None)

    evaluation_writer = getattr(ops, "journal_evaluation", None)
    if result.executed and callable(evaluation_writer):
        try:
            evaluation_writer(spec, result)
        except BaseException as exc:  # noqa: BLE001 - durability failure is terminal
            detail = f"evaluation_event: {type(exc).__name__}: {exc}"
            print(f"WARNING: evaluated evidence could not be journaled: {detail}",
                  file=sys.stderr)
            result = replace(result, journal_error=detail)

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
        prior = result.journal_error
        result = replace(result, journal_error="; ".join(
            item for item in (prior, detail) if item))
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
    parser.add_argument(
        "--proposal-manifest",
        default=None,
        metavar="PATH",
        help="validated current-schema proposal JSON. Required by --execute and "
             "fsynced before host work",
    )
    parser.add_argument(
        "--least-commitment-capture-plan", default=None, metavar="PATH",
        help="hash-bound prospective AK-WM-2 diagnostic/control plan. Required by "
             "executing IQK parameter campaigns so a clean result is archive-usable",
    )
    parser.add_argument(
        "--matched-experiment-id", default=None, metavar="AKM-ID",
        help="shared intervention/control identity that owns benchmark, T0, and "
             "holdout randomization; required with --least-commitment-capture-plan",
    )
    parser.add_argument(
        "--source-patch-manifest", default=None, metavar="PATH",
        help="immutable source-patch.v1 JSON with embedded bytes; required for "
             "source-changing --execute campaigns",
    )
    source_prerequisite_group = parser.add_mutually_exclusive_group()
    source_prerequisite_group.add_argument(
        "--source-prerequisite-package", default=None, metavar="PATH",
        help="immutable content-addressed archive/resume package containing all three "
             "raw source-candidate correctness receipts. Loaded before any claim and "
             "reduced again against the live build before T0",
    )
    source_prerequisite_group.add_argument(
        "--fresh-source-prerequisite-plan", default=None, metavar="PATH",
        help="strict predeclared plan for producing all three raw source-candidate "
             "receipts after the candidate build, under this campaign's already-held "
             "CPU/device claims. Loaded before any claim; execute-only",
    )
    parser.add_argument(
        "--calibration-bundle",
        default=None,
        metavar="DIR",
        help="accepted live-control bundle for the exact production commit, measurement "
             "instrument and recipe. Required by --execute; stale-era bundles refuse",
    )
    physical_group = parser.add_mutually_exclusive_group()
    physical_group.add_argument(
        "--physical-envelope",
        default=None,
        metavar="PATH",
        help="predeclared RVP-C6-4 physical-envelope JSON for the exact measured "
             "unit. Required by --execute and checked on every emitted sample",
    )
    physical_group.add_argument(
        "--ranked-units",
        default=None,
        metavar="PATH",
        help="strict epyc.autokernel.ranked-units.v1 JSON: normal and anti-short-"
             "circuit recipe variants, each with its exact physical envelope. "
             "Replaces --physical-envelope and puts every unit in the ranked stream",
    )
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND_CPU)
    parser.add_argument("--blocks", type=int, default=None,
                        help=f"the PRE-COMMITTED number of paired blocks "
                             f"(default: accepted B_min for a calibrated recipe, otherwise "
                             f"{DEFAULT_BLOCKS}). Fixed before the run: the "
                             f"accept rule refuses any other count, which is what makes "
                             f"optional stopping impossible rather than discouraged.")
    parser.add_argument("--screening-only", action="store_true", default=False,
                        help="max-three-block non-promotable discovery screen; skips T0 and "
                             "post-T0 stabilization and cannot KEEP/archive/promote")
    parser.add_argument("--screening-baseline-bank", default=None, metavar="PATH",
                        help="sealed reusable anchor baseline required by --screening-only")
    parser.add_argument("--create-screening-baseline", default=None, metavar="PATH",
                        help="execute exactly three bound anchor calls and seal a fresh reusable "
                             "non-promotable screening baseline bank at PATH")
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
    # ``--json`` is an automation contract: stdout (or the injected ``out``
    # stream) must contain exactly one parseable result document. Keep the
    # detailed composition trace, but move it to stderr in JSON mode.
    detail_stream = sys.stderr if args.as_json else stream

    if not args.dry_run and not args.i_hold_the_host:
        print("--execute requires --i-hold-the-host. This driver spawns benchmarks on a "
              "shared machine; two of the six A/A runs on 2026-08-04 were destroyed by a "
              "legitimate co-tenant, and the co-tenant did nothing wrong.", file=sys.stderr)
        return 2

    proposal = None
    if args.proposal_manifest is not None:
        try:
            proposal = json.loads(Path(args.proposal_manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"refusing to start: --proposal-manifest: {exc}", file=sys.stderr)
            return 2
    if not args.dry_run and proposal is None:
        print(
            "refusing to --execute: --proposal-manifest is required before any claim, "
            "mutation, or build",
            file=sys.stderr,
        )
        return 2

    least_commitment_plan = None
    if args.least_commitment_capture_plan is not None:
        if proposal is None:
            print("refusing to start: --least-commitment-capture-plan requires "
                  "--proposal-manifest", file=sys.stderr)
            return 2
        try:
            least_commitment_plan = least_commitment_capture.load(
                args.least_commitment_capture_plan, proposal=proposal,
                campaign_id=args.campaign_id, candidate_id=args.candidate_id)
        except (OSError, json.JSONDecodeError, TypeError,
                least_commitment_capture.CapturePlanError) as exc:
            print(f"refusing to start: --least-commitment-capture-plan: {exc}",
                  file=sys.stderr)
            return 2
    if not args.dry_run and not args.screening_only \
            and args.create_screening_baseline is None \
            and isinstance(proposal, Mapping) \
            and proposal.get("change_class") == "parameter" \
            and least_commitment_plan is None:
        print("refusing to --execute: IQK parameter campaigns require "
              "--least-commitment-capture-plan before any claim, mutation, build, "
              "or benchmark; otherwise a clean result cannot enter AK-WM-2",
              file=sys.stderr)
        return 2

    baseline_bank = None
    if args.screening_baseline_bank is not None:
        try:
            baseline_bank = screening_baseline.load(args.screening_baseline_bank)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"refusing to start: --screening-baseline-bank: {exc}", file=sys.stderr)
            return 2
    if args.screening_only and baseline_bank is None:
        print("refusing to start: --screening-only requires --screening-baseline-bank", file=sys.stderr)
        return 2
    if args.create_screening_baseline is not None and args.screening_only:
        print("refusing to start: --create-screening-baseline and --screening-only are separate modes",
              file=sys.stderr)
        return 2
    if not args.dry_run and least_commitment_plan is not None \
            and least_commitment_plan.raw["capture_mode"] != "measured":
        print("refusing to --execute: architecture_regression_fixture cannot supply "
              "least-commitment evidence", file=sys.stderr)
        return 2

    source_patch = None
    if args.source_patch_manifest is not None:
        try:
            source_patch = source_candidate.load_source_patch_manifest(
                args.source_patch_manifest)
        except (OSError, ValueError, TypeError, source_candidate.SourceCandidateError) as exc:
            print(f"refusing to start: --source-patch-manifest: {exc}", file=sys.stderr)
            return 2

    source_prerequisites = None
    if args.source_prerequisite_package is not None:
        try:
            source_prerequisites = (
                source_prerequisite_package.load_source_prerequisite_package(
                    args.source_prerequisite_package))
        except (OSError, ValueError, TypeError,
                source_prerequisite_package.SourcePrerequisitePackageError) as exc:
            print(f"refusing to start: --source-prerequisite-package: {exc}",
                  file=sys.stderr)
            return 2

    fresh_source_plan = None
    if args.fresh_source_prerequisite_plan is not None:
        try:
            fresh_source_plan = (
                source_prerequisite_producer.load_fresh_source_prerequisite_plan(
                    args.fresh_source_prerequisite_plan))
        except (OSError, ValueError, TypeError,
                source_prerequisite_producer.FreshSourcePrerequisiteError) as exc:
            print(f"refusing to start: --fresh-source-prerequisite-plan: {exc}",
                  file=sys.stderr)
            return 2

    selected_calibration = None
    if args.calibration_bundle is not None:
        try:
            selected_calibration = load_calibration_bundle(args.calibration_bundle)
        except (ValueError, TypeError) as exc:
            print(f"refusing to start: --calibration-bundle: {exc}", file=sys.stderr)
            return 2
    if not args.dry_run and not args.screening_only and args.create_screening_baseline is None and selected_calibration is None:
        print(
            "refusing to --execute: --calibration-bundle is required; historical or "
            "cross-cell constants carry no live ranking authority",
            file=sys.stderr,
        )
        return 2

    physical_envelope = None
    ranked_units: tuple[RankedUnitSpec, ...] = ()
    if args.physical_envelope is not None:
        try:
            physical_payload = json.loads(
                Path(args.physical_envelope).read_text(encoding="utf-8"))
            physical_envelope = physical_bounds.PhysicalEnvelope.from_mapping(
                physical_payload)
        except (OSError, json.JSONDecodeError,
                physical_bounds.PhysicalBoundError) as exc:
            print(f"refusing to start: --physical-envelope: {exc}", file=sys.stderr)
            return 2
    if args.ranked_units is not None:
        try:
            ranked_payload = json.loads(
                Path(args.ranked_units).read_text(encoding="utf-8"))
            ranked_units = ranked_units_from_mapping(ranked_payload)
        except (OSError, json.JSONDecodeError, ValueError, TypeError,
                physical_bounds.PhysicalBoundError) as exc:
            print(f"refusing to start: --ranked-units: {exc}", file=sys.stderr)
            return 2
    if not args.dry_run and not args.screening_only and args.create_screening_baseline is None and physical_envelope is None and not ranked_units:
        print(
            "refusing to --execute: --physical-envelope or --ranked-units is required "
            "before any claim, mutation, build, or benchmark",
            file=sys.stderr,
        )
        return 2

    resolved_recipe_id = (
        args.recipe_id
        if args.recipe_id is not None
        else (selected_calibration.recipe_id if selected_calibration is not None
              else DEFAULT_RECIPE_BY_BACKEND[args.backend])
    )
    resolved_blocks = (
        args.blocks
        if args.blocks is not None
        else (3 if (args.screening_only or args.create_screening_baseline is not None) else selected_calibration.b_min_blocks
              if selected_calibration is not None else DEFAULT_BLOCKS)
    )

    try:
        spec = CampaignSpec(
            campaign_id=args.campaign_id, candidate_id=args.candidate_id,
            candidate_ref=args.candidate_ref, backend=args.backend, blocks=resolved_blocks,
            recipe_id=resolved_recipe_id, model=args.model, reps=args.reps,
            devices=tuple(args.device), device_names=tuple(args.device_name),
            journal_root=args.journal_root, proposal=proposal,
            source_patch=source_patch,
            source_prerequisite_package=source_prerequisites,
            fresh_source_prerequisite_plan=fresh_source_plan,
            least_commitment_plan=least_commitment_plan,
            matched_experiment_id=args.matched_experiment_id,
            calibration=(None if args.create_screening_baseline is not None else selected_calibration),
            physical_envelope=(None if args.create_screening_baseline is not None else physical_envelope),
            ranked_units=(() if args.create_screening_baseline is not None else ranked_units),
            screening_only=args.screening_only, screening_baseline=baseline_bank)
    except (ValueError, TypeError, storage.StorageError, recipes.RecipeError,
            source_candidate.SourceCandidateError,
            source_prerequisite_package.SourcePrerequisitePackageError,
            source_prerequisite_producer.FreshSourcePrerequisiteError) as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        return 2

    if (not args.dry_run and spec.source_prerequisite_package is not None
            and spec.source_prerequisite_package.capture_mode != "measured"):
        print(
            "refusing to --execute: --source-prerequisite-package was captured in "
            "dry_run mode; it cannot supply correctness authority",
            file=sys.stderr,
        )
        return 2

    # Ranking authority is cell-local.  The five-control bundle calibrated the
    # CPU prefill cell only; silently applying it to decode or GPU is the exact
    # cross-cell transfer that calibration exists to prevent.  This refusal is
    # before HostOps construction, host reads, a claim, mutation, or subprocess.
    if not args.dry_run and not args.screening_only and args.create_screening_baseline is None:
        calibration = spec.calibration
        if calibration is None:
            print(
                f"refusing to --execute: recipe {spec.recipe_id!r} has no accepted "
                "cell-local live calibration; run or select a calibrated recipe first",
                file=sys.stderr,
            )
            return 2
        if not calibration.b_min_blocks <= spec.blocks <= calibration.max_blocks:
            print(
                f"refusing to --execute: --blocks {spec.blocks} is outside the accepted "
                f"range [{calibration.b_min_blocks}, {calibration.max_blocks}] from "
                f"{calibration.evidence_ref}",
                file=sys.stderr,
            )
            return 2

    broker: Optional[powercap_broker.PowercapBroker] = None
    if ops is None:
        if args.dry_run:
            ops = DryRunOps(out=detail_stream)
        else:
            # Root-owned package counters are read through a networkless,
            # read-only broker. It starts lazily on the first host snapshot, so
            # every pre-claim refusal above remains side-effect free.
            broker = powercap_broker.PowercapBroker()
            ops = HostOps(nominal_khz=args.nominal_khz,
                          host_state=broker.read_host_state)

    if args.create_screening_baseline is not None:
        if args.dry_run:
            print("screening baseline creation DRY RUN: no claim, subprocess, or output file", file=detail_stream)
            return 0
        if not isinstance(ops, HostOps):
            print("refusing to start: baseline creation requires concrete HostOps", file=sys.stderr)
            return 2
        pre = ops.preflight(spec)
        if pre.outcome == schemas.FAIL:
            print("refusing to start: baseline preflight: " + "; ".join(pre.reasons), file=sys.stderr)
            return 2
        claim = ops.acquire_claim(spec)
        try:
            receipt = ops.create_screening_baseline(spec, output=args.create_screening_baseline)
        finally:
            ops.release_claim(claim)
            if broker is not None:
                broker.close()
        print(json.dumps(receipt, sort_keys=True) if args.as_json else
              f"screening baseline sealed: {receipt['path']} ({receipt['baseline_sha256']})",
              file=stream)
        return 0

    # BEFORE the claim, and before the banner: source-changing campaigns still
    # have proposal-specific seams; the IQK
    # parameter campaign has a built-in adapter. Check before any host work.
    # Refusing here costs argv-parse time. A run that
    # never started must not print a line saying EXECUTING either.
    unimplemented = getattr(ops, "unimplemented_seams", None)
    if not args.dry_run and callable(unimplemented):
        pending = unimplemented(spec)
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
          file=detail_stream)
    print("  question    "
          + (f"{spec.hypothesis_id} — falsifier: {spec.authorization.falsifier}"
             if spec.authorization is not None
             else "EXPLORATORY (no --hypothesis; this run resolves no question)"),
          file=detail_stream)
    print(f"  cell        {spec.recipe_id}  metric={spec.metric}", file=detail_stream)
    if spec.screening_only:
        print("  authority   SCREENING_ONLY / NON_PROMOTABLE: no T0, stabilization, "
              "KEEP, archive, or promotion", file=detail_stream)
        print("  accept      candidate-only top-K nomination under unquantified noise; "
              "strict confirmation required", file=detail_stream)
    elif spec.calibration is None:
        print("  accept      UNCALIBRATED CELL — dry-run composition only; live ranking "
              "will refuse", file=detail_stream)
    else:
        print(f"  accept      min(delta) > 0 over {spec.blocks} pre-committed pairs AND "
              f"median(relative) > {spec.contribution_floor:.4%}", file=detail_stream)
        print(f"  calibration B_min={spec.calibration.b_min_blocks}, "
              f"MDE={spec.calibration.mde:.4%}: "
              f"{spec.calibration.evidence_ref}", file=detail_stream)
    if not spec.screening_only:
        print(f"  anchor movement bound: {spec.drift_bound:.4%} from {AA_EVIDENCE_REF}",
              file=detail_stream)
    print("", file=detail_stream)

    try:
        result = run_campaign(spec, ops)
    finally:
        if broker is not None:
            broker.close()

    print("", file=detail_stream)
    print(f"state: {result.state}", file=detail_stream)
    if result.decision is not None:
        print(result.decision.reason, file=detail_stream)
    if result.error:
        print(f"error: {result.error.splitlines()[0]}", file=detail_stream)
    for record in result.releases:
        marker = "released" if record.released else "NOT RELEASED"
        print(f"  {record.name}: {marker} ({record.detail})", file=detail_stream)
    if result.production_unchanged is not None:
        print(f"  production trees: {result.production_unchanged.outcome}",
              file=detail_stream)
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str),
              file=stream)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised by test_campaign via main()
    sys.exit(main())
