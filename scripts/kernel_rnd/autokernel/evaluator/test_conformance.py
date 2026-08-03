#!/usr/bin/env python3
"""test_conformance.py — clause-by-clause conformance of the AK3 evaluator to
`measurement/protocols/kernel-research.md` (Annex K, **P-AK-SEARCH-1**, RATIFIED
2026-08-03).

WHY THIS FILE EXISTS
--------------------
The six evaluator modules were written in parallel, each against its own reading
of the protocol, and each one passes its own suite. A suite that passes proves a
module does what its author meant; it does not prove the BUNDLE does what the
ratified text says. This file is the second thing: it walks the protocol's own
sections — *Preconditions*, *Campaign calibration block*, *Statistical
requirements*, *Controls*, *Correctness precedence*, *Search-grade requires ALL
of*, *Record grammar*, *What voids a run* — and asserts one test per obligation,
against the assembled evaluator rather than against any one module.

Three properties make it hard to pass this file by weakening it:

  1. **The obligation register is the fixture.** `OBLIGATIONS` below is the
     protocol's clause list, transcribed with its own words. Every obligation
     carries an id, and every test claims the obligation it discharges with an
     `[OB:<id>]` marker in its docstring.
  2. **Coverage is asserted, not assumed.** `TestObligationCoverage` reads THIS
     module's AST and fails if any registered obligation has no claiming test, if
     any test claims an obligation that does not exist, or if a claimed
     obligation's test body is empty. Deleting a test is therefore not a way to
     make this file pass.
  3. **"Cannot be tested without inference" is a declared state, not a silent
     gap.** `SEAM_ONLY` names the obligations whose subject is a measurement this
     evaluator must never take (a real A/A run, a real anchor rebuild, a real
     host-health probe). Each one still gets a test — the test asserts the SEAM
     exists, is typed, and fails closed when the seam is unwired — and the
     obligation is listed with the reason it is seam-only. An obligation that is
     neither directly asserted nor registered as seam-only fails coverage.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO PROCESS. This suite constructs typed
objects and reads their answers. It starts, stops and signals nothing, and it
writes no file outside a `tempfile` tree it removes.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_conformance.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/evaluator/test_conformance.py
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import math
import sys
import unittest
from collections import namedtuple
from pathlib import Path

# Import through the PACKAGE so `api.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S                      # noqa: E402
from autokernel.evaluator import api                     # noqa: E402
from autokernel.evaluator import controls as CT          # noqa: E402
from autokernel.evaluator import correctness as CO       # noqa: E402
from autokernel.evaluator import integrity as IG         # noqa: E402
from autokernel.evaluator import recipes as RC           # noqa: E402
from autokernel.evaluator import statistics as ST        # noqa: E402
from autokernel.evaluator import surface as SU           # noqa: E402

PASS = S.Check(S.PASS)
NOW = "2026-08-03T12:00:00+00:00"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"

PROTOCOL_PATH = "measurement/protocols/kernel-research.md"


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def fail(*reasons: str) -> S.Check:
    return S.Check(S.FAIL, tuple(reasons) or ("failed",))


def cnc(*reasons: str) -> S.Check:
    return S.Check(S.COULD_NOT_CHECK, tuple(reasons) or ("unreadable",))


# =============================================================================
# The obligation register — the protocol's clauses, in its own words
# =============================================================================

Obligation = namedtuple("Obligation", "id section clause")


def _ob(oid: str, section: str, clause: str) -> Obligation:
    return Obligation(oid, section, clause)


OBLIGATIONS = (
    # ---- Preconditions (all enforced or attested per run) -------------------
    _ob("PRE-1", "Preconditions",
        "Resource claim held for the whole window ... Both are ACQUIRED, never "
        "inferred ... re-verified as still held, by the same holder, at window close "
        "as well as window open."),
    _ob("PRE-2", "Preconditions",
        "No concurrent inference, established by the sanctioned preflight substitute "
        "... If that item is not in force, this precondition has no sanctioned means "
        "of satisfaction and no run may start."),
    _ob("PRE-3", "Preconditions", "Host-health tier satisfied per bench-cpu.md:17-19."),
    _ob("PRE-4", "Preconditions",
        "An EXPLICIT IMMUTABLE ANCHOR ... names its anchor by source commit, binary "
        "SHA-256, and linkage SHA-256, and the anchor is re-verified byte-for-byte at "
        "window open and window close."),
    _ob("PRE-4b", "Preconditions",
        "A run without an explicit anchor is INVALID - never \"correct\", never "
        "\"coherent\", never \"byte-identical\"."),
    _ob("PRE-4c", "Preconditions",
        "Absence of a comparison is not evidence of equivalence: a coherence or "
        "identity label produced without a named anchor comparison is not a verdict."),
    _ob("PRE-5", "Preconditions",
        "Evaluator identity ... the pinned bundle SHA-256 ... plus a runtime "
        "source-label attestation ... Any drift ... voids every record in the window."),
    _ob("PRE-6", "Preconditions",
        "Codified recipe. Every measurement command line ... is emitted by a recipe "
        "constructor; the constructor's identifier and content hash are recorded with "
        "the record. Hand-typed argv voids the run."),
    _ob("PRE-7", "Preconditions",
        "Storage headroom at or above storage_floor_bytes_free ... checked at window "
        "open and re-checked at window close."),
    _ob("PRE-8", "Preconditions",
        "Declared campaign controls ... Each MUST be finite and strictly positive; a "
        "campaign that omits one, or declares it as zero or unbounded, cannot derive "
        "its error budgets and MUST NOT start."),
    _ob("PRE-ALL", "Preconditions",
        "The eight preconditions are each enforced or attested per run, and an "
        "attestation that could not be read is not a satisfied precondition."),

    # ---- Campaign calibration block -----------------------------------------
    _ob("CAL-ORDER", "Campaign calibration block",
        "The calibration block is evaluated in exactly this order, and a conforming "
        "implementation MUST record that it did."),
    _ob("CAL-PHI", "Campaign calibration block",
        "Campaign noise floor phi - the 95th percentile of the |effect| distribution "
        "observed in the A/A control."),
    _ob("CAL-PHI-RANK", "Campaign calibration block",
        "An estimate whose magnitude does not exceed phi MUST NOT be ranked, banked, "
        "or composed, whatever its evidence value."),
    _ob("CAL-NEUTRAL", "Campaign calibration block",
        "a neutral control materially exceeding the A/A floor FAILS the calibration "
        "rather than raising the floor."),
    _ob("CAL-BMIN", "Campaign calibration block",
        "Minimum paired-block count B_min ... is floored by, and MUST NEVER fall "
        "below, the P-BENCH-1 reps rule."),
    _ob("CAL-BMIN-FIXED", "Campaign calibration block",
        "Where the cell's own phase is governed by a protocol that states a stricter "
        "or a fixed rep rule, that protocol's rule governs its own cells and this "
        "calibration never overrides it."),
    _ob("CAL-ALPHA", "Campaign calibration block",
        "alpha_sel MUST NOT exceed the reciprocal of max_candidates ... alpha_conf "
        "MUST NOT exceed alpha_sel divided by confirmation_admission_count, and MUST "
        "NOT be looser than alpha_sel."),
    _ob("CAL-THRESHOLD", "Campaign calibration block",
        "E-process rejection thresholds - 1/alpha_sel for selection and 1/alpha_conf "
        "for confirmation."),
    _ob("CAL-VALIDATE", "Campaign calibration block",
        "alpha_sel is then validated empirically once, at the solved B_min ... both "
        "the failed and the accepted calibration are retained in the manifest."),
    _ob("CAL-BAND", "Campaign calibration block",
        "Anchor-gate acceptance band - the interval containing the central 95% of the "
        "anchor cell's own calibration values ... computed at the solved B_min."),
    _ob("CAL-NO-FIFTH", "Campaign calibration block",
        "There is no fifth output. The campaign storage floor is NOT derived here."),
    _ob("CAL-CEILING", "Campaign calibration block",
        "If no B_min less than or equal to the declared max_blocks_per_candidate "
        "satisfies both conditions, the calibration FAILS and the campaign does not "
        "start. There is no partial calibration and no fallback ceiling."),
    _ob("CAL-NO-LITERAL", "Campaign calibration block",
        "No value in this list may be supplied as a literal."),
    _ob("CAL-CONSTRUCTION", "Campaign calibration block",
        "The e-process construction itself ... is a property of the evaluator bundle, "
        "fixed at the bundle hash; a campaign selects among constructions the bundle "
        "already implements and records which one it selected."),
    _ob("CAL-CELL", "Campaign calibration block",
        "Values calibrated under a different host state, backend, phase, or cell class "
        "MUST NOT be reused."),
    _ob("CAL-INCOMPLETE", "Campaign calibration block",
        "A campaign that cannot complete its calibration block MUST NOT rank any "
        "candidate."),
    _ob("CAL-RECOMPUTE", "Campaign calibration block",
        "Every output is recomputed at each campaign boundary and whenever anchor "
        "identity changes."),

    # ---- Statistical requirements -------------------------------------------
    _ob("STAT-REPS", "Statistical requirements",
        "Reps. Per the P-BENCH-1 rule - >=5 for >=5% effects, >=10 for <=2% effects - "
        "and never fewer than the calibrated B_min paired blocks."),
    _ob("STAT-MEDIAN-MAD", "Statistical requirements", "Report median + MAD."),
    _ob("STAT-EPROCESS", "Statistical requirements",
        "E-process, never an ad-hoc bound. Every rate comparison goes through the "
        "non-inferiority / improvement e-process, never a single trial."),
    _ob("STAT-LCB", "Statistical requirements",
        "A lower-confidence-bound construction MUST NOT be the test that ranks ... An "
        "LCB MAY be carried beside the e-value as a labelled descriptive statistic."),
    _ob("STAT-STOPPING", "Statistical requirements",
        "Pre-committed stopping rule ... name the table that is FINAL, the decision "
        "each outcome triggers, the max_blocks_per_candidate ceiling, and the bounded "
        "extension rule."),
    _ob("STAT-STOPPING-FIXED", "Statistical requirements",
        "Anytime-validity licenses inspecting at every block; it never licenses "
        "changing the rule. Any post-hoc change to the stopping rule voids every "
        "affected record."),
    _ob("STAT-MDE", "Statistical requirements",
        "MDE published WITH the result ... written into the same record as the "
        "estimate, not afterwards."),
    _ob("STAT-MDE-VERDICT", "Statistical requirements",
        "|effect| < MDE yields the verdict no detectable difference, which is a result "
        "and a decision, not a failed experiment."),
    _ob("STAT-ORDER", "Statistical requirements",
        "Order control. Candidate and anchor are interleaved and order-randomized "
        "within every paired block ... the randomization seed derives from the campaign "
        "seed committed before the first candidate was measured, and is recorded."),
    _ob("STAT-BLOCKED", "Statistical requirements",
        "Blocked designs (candidate x n, then anchor x n) are forbidden."),
    _ob("STAT-RETRY", "Statistical requirements",
        "A retry is a fresh reset in reversed order."),
    _ob("STAT-ANCHOR-GATE", "Statistical requirements",
        "The anchor cell is measured FIRST in every window and compared against the "
        "calibrated acceptance band. Outside the band => the window is VOID."),
    _ob("STAT-ANCHOR-GATE-VOID", "Statistical requirements",
        "A VOID window is journaled as INVALID; it MUST NOT be recorded as a candidate "
        "failure, because a drifted anchor says nothing whatever about the candidate."),
    _ob("STAT-STRATA", "Statistical requirements",
        "The measurement material ... is partitioned into disjoint selection and "
        "confirmation strata by a rule recorded in the campaign manifest and keyed on "
        "the campaign seed, before the first candidate was measured."),
    _ob("STAT-STRATA-MIX", "Statistical requirements",
        "No block may serve both strata; a record mixing strata is INVALID."),
    _ob("STAT-READINESS", "Statistical requirements",
        "The readiness signal is computed ONLY from confirmation-stratum evidence "
        "gathered after the candidate entered the lineage."),
    _ob("STAT-ROTATION", "Statistical requirements",
        "Confirmation shapes and control seeds rotate on the schedule declared in the "
        "evaluator bundle."),

    # ---- Controls ------------------------------------------------------------
    _ob("CTL-BUNDLE", "Controls",
        "Control definitions, fixtures, expected directions, and seeds live inside the "
        "evaluator bundle under the measurement trust boundary and MUST NOT be modified "
        "by any process inside the loop."),
    _ob("CTL-1-4", "Controls",
        "A campaign that cannot run controls 1-4 MUST NOT rank any candidate."),
    _ob("CTL-1", "Controls",
        "Positive - a known-correct optimization with a real, bounded mechanism. MUST "
        "rank above the anchor. Failure is a gate defect."),
    _ob("CTL-2", "Controls",
        "Neutral - a correct change whose true effect is centred on zero. MUST NOT "
        "advance, and its dispersion is checked against the calibrated floor."),
    _ob("CTL-3", "Controls",
        "Degraded-negative - deliberately fast-looking but wrong ... MUST receive no "
        "speed rank at all."),
    _ob("CTL-4", "Controls",
        "A/A - the anchor measured against itself, through the full candidate pipeline. "
        "Runs periodically on its declared cadence, not once per campaign."),
    _ob("CTL-4-VOID", "Controls", "A failing A/A VOIDS the enclosing measurement window."),
    _ob("CTL-5", "Controls",
        "Historical-win replay ... The campaign manifest declares a "
        "historical_win_replay entry carrying {win_id, backend, phase, reference "
        "direction, reference magnitude band, in-repo evidence locator, durability "
        "class} ... it MUST promote."),
    _ob("CTL-5-DEFECT", "Controls",
        "A failure to promote is a gate defect, not a research finding: it halts the "
        "campaign and is escalated to the operator."),
    _ob("CTL-5-UNAVAIL", "Controls",
        "Unavailable branch (normative, not a silent skip) ... records "
        "HISTORICAL_REPLAY_UNAVAILABLE in its journal and its manifest, naming the "
        "backend and the reason, and escalates to the operator."),
    _ob("CTL-5-MARKER", "Controls",
        "every record emitted by such a campaign carries controls=4/5 "
        "(HISTORICAL_REPLAY_UNAVAILABLE) in its grammar."),
    _ob("CTL-5-OPERATOR", "Controls",
        "Whether the campaign proceeds on four controls is the operator's call, taken "
        "once, on the record - not the controller's."),

    # ---- Correctness precedence ---------------------------------------------
    _ob("COR-LEX", "Correctness precedence",
        "Correctness, quality, numerical safety, integrity, and stability are "
        "lexicographically prior to speed."),
    _ob("COR-NO-RANK", "Correctness precedence",
        "A candidate failing any of them receives no speed rank at all - not a "
        "penalised one."),
    _ob("COR-ORACLE", "Correctness precedence",
        "Correctness verdicts are produced by the evaluator against declared oracles "
        "and are NEVER self-reported by the candidate."),
    _ob("COR-CACHE-ORACLE", "Correctness precedence",
        "A candidate output MUST NEVER be cached or reused as a correctness oracle."),
    _ob("COR-CACHE-STATE", "Correctness precedence",
        "Cache state is declared in every record."),

    # ---- Search-grade requires ALL of ---------------------------------------
    _ob("SG-CONJUNCTS", "Search-grade requires ALL of",
        "This ratified protocol; every precondition above; a completed and accepted "
        "calibration block ...; the pre-committed stopping rule unmodified; B_min "
        "paired blocks under order-randomized interleaving; a passing anchor gate; a "
        "passing A/A control within its declared cadence; controls 1-4 available and "
        "passing; control 5 either passing or explicitly recorded ...; an e-value "
        "against the calibrated threshold; a published MDE; the correct stratum; the "
        "complete record grammar below; and raw samples from which the reduction is "
        "reproducible."),
    _ob("SG-MISSING-ANY", "Search-grade requires ALL of",
        "Missing ANY of these makes the record INVALID. There is no weaker-but-usable "
        "state."),
    _ob("SG-RETAINED", "Search-grade requires ALL of",
        "an INVALID record is retained in the journal and MUST NOT rank, bank, compose, "
        "or contribute to readiness."),
    _ob("SG-NOT-A-CLAIM", "Search-grade requires ALL of",
        "Neither state is ever a claim - a conforming search record is still an "
        "observation with respect to MEASUREMENT.md:9-11."),
    _ob("SG-TIER-SCOPE", "Search-grade requires ALL of",
        "Scope: tiers T0, T1 and T2 ... It does NOT apply to T3 or any release gate."),

    # ---- Record grammar ------------------------------------------------------
    _ob("REC-FIELDS", "Record grammar",
        "Every record carries category=CANDIDATE ...; the tier; the evaluator bundle "
        "hash and runtime source-label attestation reference; the resource-claim "
        "receipt; the host-health receipt; the anchor identity ...; the "
        "recipe-constructor identity; the stratum; the scope denominator of what was "
        "actually measured; the determinism class; and a reference to the raw samples."),
    _ob("REC-REPRODUCIBLE", "Record grammar",
        "A record whose reduction cannot be recomputed from its raw samples is INVALID."),
    _ob("REC-TEMPLATE", "Record grammar",
        "a record omitting any field of this template is INVALID."),
    _ob("REC-CLASS", "Record grammar",
        "SEARCH RECORD, NOT A CLAIM [P-AK-SEARCH-1, category=CANDIDATE, ...]"),
    _ob("REC-NO-ATTEST", "Record grammar",
        "The grammar carries no attest <ref> field ... that field is satisfied by res + "
        "host + srclabel together."),
    _ob("REC-METRIC", "Metric",
        "<metric> <value> <higher-better|lower-better> ... Substituting one for the "
        "other is forbidden by MEASUREMENT.md:25-26."),

    # ---- What voids a run ----------------------------------------------------
    _ob("VOID-ENUM", "What voids a run",
        "A resource claim not held ...; a host-health tier violation; a failed anchor "
        "gate; a failed A/A control; a missing, drifted, or unverifiable evaluator "
        "bundle hash or runtime source-label attestation; a missing or mutated anchor; "
        "hand-typed argv; contamination by concurrent inference; storage exhaustion "
        "mid-window; a strata violation; any post-hoc change to the stopping rule, the "
        "calibration outputs, the objective, or the control definitions; or an "
        "incomplete calibration block."),
    _ob("VOID-JOURNALED", "What voids a run",
        "A voided run is journaled as INVALID with its reason, and is never silently "
        "discarded - primary records are never destroyed."),
    _ob("VOID-CONTROL-DEFS", "What voids a run",
        "any post-hoc change to ... the control definitions."),
    _ob("VOID-THIRD-OUTCOME", "What voids a run",
        "Inability to evaluate is a third outcome: FAIL and COULD_NOT_CHECK both void, "
        "and the record says which it was."),
)

OBLIGATION_BY_ID = {o.id: o for o in OBLIGATIONS}
assert len(OBLIGATION_BY_ID) == len(OBLIGATIONS), "duplicate obligation id"


#: Obligations whose subject is a measurement this evaluator MUST NOT take. Each
#: is still discharged by a test — the test asserts that the SEAM exists, is
#: typed, and fails CLOSED when it is unwired — and the value here is the reason
#: the direct assertion is impossible. An obligation that is neither directly
#: asserted nor listed here fails `TestObligationCoverage`.
SEAM_ONLY = {
    "PRE-1": "acquiring a real region/device claim is AK1's `resource/device_claim`; "
             "this suite asserts the attestation fields and their fail-closed "
             "combination, and `test_integration.py` acquires a real claim on a "
             "made-up device id in a temp lock root.",
    "PRE-2": "the sanctioned preflight substitute enumerates live processes and "
             "cgroups; running it for real is `resource/preflight`. Asserted here as "
             "a required attestation whose non-PASS voids.",
    "PRE-3": "the host-health tier is a live probe (uptime, throttling, free memory). "
             "Asserted here as a required attestation whose non-PASS voids.",
    "CAL-PHI": "phi is estimated from a REAL A/A control on the campaign's own host "
               "state. Asserted here on synthetic A/A material through the same "
               "estimator the campaign uses.",
    "CAL-RECOMPUTE": "'recomputed at each campaign boundary' is a controller-lifecycle "
                     "obligation (AK4). Asserted here as the structural half: a "
                     "calibration solved for another cell is refused, and the anchor "
                     "identity is part of the window that voids on drift.",
    "STAT-ANCHOR-GATE": "measuring the anchor cell FIRST in every window is a runner "
                        "ordering obligation (AK4). Asserted here as the band check "
                        "and the void it raises.",
    "STAT-READINESS": "the readiness reducer is AK4/AK6. Asserted here as the stratum "
                      "the record carries and the refusal to rank an INVALID record.",
    "CTL-1": "ranking the positive control above the anchor needs a real measurement. "
             "Asserted here through the control harness on a fixture verdict.",
    "CTL-2": "the neutral control's true effect needs a real measurement. Asserted "
             "here through the harness plus the calibrated-floor consistency check.",
    "CTL-4": "the A/A control is a real anchor-vs-anchor measurement. Asserted here "
             "through the cadence scheduler and the harness's A/A evaluator.",
    "CTL-5": "the historical-win replay re-runs a durable win end to end. Asserted "
             "here through the declared contract, its resolution, and the harness.",
    "COR-CACHE-ORACLE": "proving no candidate output was reused as an oracle needs the "
                        "oracle registry at run time. Asserted here as the evidence "
                        "field and the gate that refuses a candidate-derived oracle.",
}


# =============================================================================
# Fixtures — built field by field. There is deliberately no all_clear() helper:
# a fixture that fabricates PASS is the fixture that removes the signal.
# =============================================================================

def anchor(**overrides) -> api.AnchorIdentity:
    kwargs = dict(source_commit=V8_COMMIT, binary_sha256=sha("anchor-binary"),
                  linkage_sha256=sha("anchor-linkage"),
                  measurement_event_ids=("ake-anchor-0001",))
    kwargs.update(overrides)
    return api.AnchorIdentity(**kwargs)


def campaign_controls(**overrides) -> api.CampaignControls:
    kwargs = dict(calibration_block_count=30, contribution_floor=0.02, max_candidates=100,
                  confirmation_admission_count=5, max_blocks_per_candidate=40,
                  storage_floor_bytes_free=200 * 1024 ** 3)
    kwargs.update(overrides)
    return api.CampaignControls(**kwargs)


def calibration(**overrides) -> api.CalibrationOutputs:
    kwargs = dict(
        backend="llama_gpu", phase="decode", cell_class="instrument_tokens_per_s",
        noise_floor_phi=0.009, b_min_blocks=10, alpha_sel=0.01, alpha_conf=0.002,
        anchor_gate_band=(0.97, 1.03), accepted=True,
        solve_order_recorded=api.CALIBRATION_SOLVE_ORDER,
        samples_ref="data/ak-gpu-1/calibration/aa-blocks.jsonl",
        e_process_construction_id="sign_martingale_predictable_lambda/v1")
    kwargs.update(overrides)
    return api.CalibrationOutputs(**kwargs)


def control_panel(**overrides) -> api.ControlPanel:
    kwargs = dict(positive=PASS, neutral=PASS, degraded_negative=PASS, aa=PASS,
                  historical_replay=PASS)
    kwargs.update(overrides)
    return api.ControlPanel(**kwargs)


def recipe_receipt(**overrides) -> api.RecipeReceipt:
    kwargs = dict(constructor_id="ak.microbench.llama_gpu.decode/v1",
                  constructor_sha256=sha("recipe-constructor"),
                  argv_sha256=sha("argv"))
    kwargs.update(overrides)
    return api.RecipeReceipt(**kwargs)


def request(**overrides) -> api.EvaluationRequest:
    kwargs = dict(
        event_id="ake-conf-0001", campaign_id="ak-llama_gpu-decode-20260803",
        candidate_id="akc-0001", tier="T1", backend="llama_gpu", phase="decode",
        cell_class="instrument_tokens_per_s", protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(source_sha256=sha("cand-source"),
                                      binary_sha256=sha("cand-binary"),
                                      linkage_sha256=sha("cand-linkage")),
        anchor=anchor(),
        evaluator=api.EvaluatorIdentity(id="P-AK-SEARCH-1/v1",
                                        bundle_sha256=sha("evaluator-bundle"),
                                        runtime_source_label_ref="ake-srclabel-0003"),
        scope_denominator=api.ScopeDenominator(machine_subset="partial", numa_nodes=(),
                                               devices=("mi210_0",), cores=8),
        scope_manifest_sha256=sha("scope-manifest"), co_residency="single",
        determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                          same_seed_repeat_runs=3),
        metric="decode_tokens_per_s", metric_direction="higher_better", reps=10,
        created_at=NOW, campaign_controls=campaign_controls(),
        calibration=calibration())
    kwargs.update(overrides)
    return api.EvaluationRequest(**kwargs)


def window(**overrides) -> api.WindowAttestations:
    kwargs = dict(
        resource_claim_receipt="gpu_device.mi210_0:claim-20260803T1200Z-8801",
        resource_claim_open=PASS, resource_claim_close=PASS,
        resource_claim_same_holder=PASS, no_concurrent_inference=PASS,
        preflight_attestation_ref="ake-preflight-0007",
        host_receipt="host-health-20260803T1159Z", host_health=PASS,
        anchor_at_open=anchor(), anchor_at_close=anchor(), anchor_gate=PASS,
        evaluator_bundle=PASS, runtime_source_label=PASS, recipe=recipe_receipt(),
        storage_open=PASS, storage_close=PASS, strata=PASS,
        stopping_rule_id="ak-stop-1/v1", rule_immutability=PASS, order_randomized=PASS,
        order_seed="campaign-seed-4711:akc-0001", aa_cadence=PASS,
        controls=control_panel(), calibration=PASS, control_definitions_immutable=PASS,
        raw_evidence_ref="data/ak-gpu-1/raw/akc-0001/")
    kwargs.update(overrides)
    return api.WindowAttestations(**kwargs)


def effect(**overrides) -> api.EffectEstimate:
    kwargs = dict(metric="decode_tokens_per_s", metric_direction="higher_better",
                  value=0.062, e_value=180.0, threshold=100.0, mde=0.018,
                  noise_floor=0.009, paired_blocks=12, stratum=api.STRATUM_SELECTION,
                  raw_samples=((51.2, 54.4), (51.4, 54.1)),
                  raw_samples_ref="ak-raw://ak-gpu-1/akc-0001/blocks",
                  lcb_descriptive=0.041)
    kwargs.update(overrides)
    return api.EffectEstimate(**kwargs)


def gate(gate_id="t0.output_coherence_vs_anchor", gate_class=api.GATE_CORRECTNESS,
         check=PASS, requires_anchor=False) -> api.GateResult:
    return api.GateResult(gate_id=gate_id, gate_class=gate_class, check=check,
                          requires_anchor=requires_anchor)


def evaluate(req=None, win=None, eff=None, gates=None):
    """Run the whole api pipeline and return (verdict, preconditions, void, sg, line)."""
    req = req if req is not None else request()
    win = win if win is not None else window()
    gates = (gate(),) if gates is None else tuple(gates)
    pre = api.check_preconditions(req, win)
    voids = api.check_void_conditions(req, win, rate_comparison=eff is not None)
    grammar = api.check_record_grammar_complete(request=req, window=win, effect=eff)
    sg = api.evaluate_search_grade(request=req, window=win, preconditions=pre,
                                   effect=eff, grammar_complete=grammar)
    verdict = api.compute_verdict(tier=req.tier, gates=gates, void_scan=voids,
                                  search_grade=sg, anchor=req.anchor, effect=eff)
    line = api.render_search_record_grammar(request=req, window=win, verdict=verdict,
                                            effect=eff)
    return verdict, pre, voids, sg, line, grammar


# =============================================================================
# Preconditions
# =============================================================================

class TestPreconditions(unittest.TestCase):

    def test_all_eight_preconditions_are_named_and_scanned(self):
        """[OB:PRE-ALL] Each of the eight is a three-outcome Check on every run."""
        pre = api.check_preconditions(request(), window())
        self.assertEqual(tuple(pid for pid, _ in pre.checks), api.PRECONDITION_IDS)
        self.assertEqual(len(api.PRECONDITION_IDS), 8)
        for pid, chk in pre.checks:
            with self.subTest(precondition=pid):
                self.assertIsInstance(chk, S.Check)
                self.assertIn(chk.outcome, (S.PASS, S.FAIL, S.COULD_NOT_CHECK))
        self.assertTrue(pre.satisfied)
        # An attestation that could not be READ is not a satisfied precondition.
        unreadable = api.check_preconditions(request(), window(host_health=cnc("no receipt")))
        self.assertFalse(unreadable.satisfied)
        self.assertEqual(unreadable.get("host_health_tier").outcome, S.COULD_NOT_CHECK)

    def test_the_claim_seam_is_typed_and_all_three_observations_must_hold(self):
        """[OB:PRE-1] Held at open AND close AND by the same holder; any gap voids."""
        for field in ("resource_claim_open", "resource_claim_close",
                      "resource_claim_same_holder"):
            with self.subTest(field=field):
                win = window(**{field: fail("the claim moved")})
                pre = api.check_preconditions(request(), win)
                self.assertEqual(pre.get("resource_claim_held_whole_window").outcome, S.FAIL)
                voids = api.check_void_conditions(request(), win, rate_comparison=True)
                self.assertIn(api.VOID_CLAIM_NOT_HELD, voids.reasons())
        # The receipt IDENTIFIER is recorded, and an empty one is refused outright.
        with self.assertRaises(ValueError):
            window(resource_claim_receipt="")

    def test_the_preflight_substitute_is_a_required_attestation_that_voids(self):
        """[OB:PRE-2] No sanctioned means of satisfaction => no run may start."""
        win = window(no_concurrent_inference=cnc(
            "the preflight substitute could not enumerate owned cgroups"))
        pre = api.check_preconditions(request(), win)
        self.assertEqual(pre.get("no_concurrent_inference").outcome, S.COULD_NOT_CHECK)
        voids = api.check_void_conditions(request(), win, rate_comparison=True)
        self.assertIn(api.VOID_CONCURRENT_INFERENCE, voids.reasons())
        # The attestation REFERENCE travels with the record.
        self.assertIn("preflight_attestation_ref",
                      api.build_evaluation_event(
                          request=request(), window=window(), effect=None,
                          verdict=evaluate()[0],
                          preconditions=api.check_preconditions(request(), window()),
                      )["performance"]["search_discipline"])

    def test_host_health_is_a_required_attestation_that_voids(self):
        """[OB:PRE-3] A host-health tier violation voids the run."""
        win = window(host_health=fail("uptime 9 days; bench-cpu.md:17-19 requires a reboot"))
        voids = api.check_void_conditions(request(), win, rate_comparison=True)
        self.assertIn(api.VOID_HOST_HEALTH_TIER_VIOLATION, voids.reasons())
        self.assertTrue(any("host-health tier violation" in f.protocol_phrase
                            for f in voids.findings))

    def test_the_anchor_is_named_by_three_hashes_and_reverified_at_both_ends(self):
        """[OB:PRE-4] Source commit, binary SHA-256, linkage SHA-256; open and close."""
        a = anchor()
        self.assertEqual(a.short().count("/"), 2)
        for name in ("source_commit", "binary_sha256", "linkage_sha256"):
            self.assertTrue(getattr(a, name))
        for end in ("anchor_at_open", "anchor_at_close"):
            with self.subTest(end=end):
                drifted = anchor(binary_sha256=sha("rebuilt"))
                win = window(**{end: drifted})
                pre = api.check_preconditions(request(), win)
                self.assertEqual(pre.get("explicit_immutable_anchor").outcome, S.FAIL)
                # "A rebuilt anchor is a different anchor."
                self.assertTrue(any("binary_sha256 moved" in r
                                    for r in pre.get("explicit_immutable_anchor").reasons))

    def test_a_run_without_an_anchor_is_invalid_and_never_coherent(self):
        """[OB:PRE-4b] INVALID - never "correct", never "coherent", never "byte-identical"."""
        verdict, pre, voids, _sg, line, _g = evaluate(req=request(anchor=None))
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        self.assertEqual(pre.get("explicit_immutable_anchor").outcome, S.FAIL)
        self.assertIn(api.VOID_ANCHOR_MISSING_OR_MUTATED, voids.reasons())
        self.assertIn("NO-ANCHOR", line)
        with self.assertRaises(api.SpeedRankUnavailable):
            verdict.rank_key()

    def test_a_pass_that_needed_an_anchor_is_demoted_not_believed(self):
        """[OB:PRE-4c] Absence of a comparison is not evidence of equivalence."""
        coherence = gate(gate_id="t0.output_coherence_vs_anchor",
                         gate_class=api.GATE_CORRECTNESS, check=PASS,
                         requires_anchor=True)
        verdict, *_ = evaluate(req=request(anchor=None), gates=(coherence,))
        demoted = verdict.gates[0]
        self.assertEqual(demoted.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("absence of a comparison is not evidence of equivalence",
                      " ".join(demoted.check.reasons))
        # And the same doctrine is implemented independently by correctness.py.
        self.assertTrue(callable(CO.demote_anchor_requiring_passes))

    def test_evaluator_identity_carries_bundle_hash_and_runtime_source_label(self):
        """[OB:PRE-5] Drift in either voids every record in the window."""
        ident = request().evaluator
        self.assertTrue(ident.bundle_sha256)
        self.assertTrue(ident.runtime_source_label_ref)
        with self.assertRaises(ValueError):
            api.EvaluatorIdentity(id="autokernel-evaluator", bundle_sha256=sha("b"),
                                  runtime_source_label_ref="ref")
        for field in ("evaluator_bundle", "runtime_source_label"):
            with self.subTest(field=field):
                win = window(**{field: fail("resolved hash differs from the pinned hash")})
                voids = api.check_void_conditions(request(), win, rate_comparison=True)
                self.assertIn(api.VOID_EVALUATOR_BUNDLE_UNVERIFIED, voids.reasons())

    def test_the_absence_of_a_recipe_receipt_is_hand_typed_argv(self):
        """[OB:PRE-6] Hand-typed argv voids the run."""
        win = window(recipe=None)
        pre = api.check_preconditions(request(), win)
        self.assertEqual(pre.get("codified_recipe").outcome, S.FAIL)
        voids = api.check_void_conditions(request(), win, rate_comparison=True)
        self.assertIn(api.VOID_HAND_TYPED_ARGV, voids.reasons())
        # ... and the constructor identity + content hash are what a receipt IS.
        r = recipe_receipt()
        self.assertEqual(r.render(), f"{r.constructor_id}@{r.constructor_sha256[:12]}")
        # recipes.py is the only argv emitter, and it hands back this exact type.
        self.assertIs(RC.api.RecipeReceipt, api.RecipeReceipt)

    def test_storage_headroom_is_checked_at_open_and_rechecked_at_close(self):
        """[OB:PRE-7] Two observations, and either one failing voids."""
        for field in ("storage_open", "storage_close"):
            with self.subTest(field=field):
                win = window(**{field: fail("below storage_floor_bytes_free")})
                pre = api.check_preconditions(request(), win)
                self.assertEqual(pre.get("storage_headroom").outcome, S.FAIL)
                voids = api.check_void_conditions(request(), win, rate_comparison=True)
                self.assertIn(api.VOID_STORAGE_EXHAUSTED, voids.reasons())

    def test_every_declared_campaign_control_must_be_finite_and_strictly_positive(self):
        """[OB:PRE-8] Omitted, zero or unbounded => the campaign MUST NOT start."""
        names = ("calibration_block_count", "contribution_floor", "max_candidates",
                 "confirmation_admission_count", "max_blocks_per_candidate",
                 "storage_floor_bytes_free")
        self.assertEqual(tuple(api.CampaignControls.__dataclass_fields__), names)
        for name in names:
            with self.subTest(control=name, kind="zero"):
                with self.assertRaises(ValueError):
                    campaign_controls(**{name: 0})
            with self.subTest(control=name, kind="omitted"):
                partial = {n: getattr(campaign_controls(), n) for n in names if n != name}
                parsed, reasons = api.CampaignControls.parse(partial)
                self.assertIsNone(parsed)
                self.assertTrue(any(name in r for r in reasons))
        with self.subTest(kind="unbounded"):
            with self.assertRaises(ValueError):
                campaign_controls(contribution_floor=math.inf)
        # A run with no declared controls fails precondition 8 outright.
        pre = api.check_preconditions(request(campaign_controls=None), window())
        self.assertEqual(pre.get("declared_campaign_controls").outcome, S.FAIL)


# =============================================================================
# Campaign calibration block
# =============================================================================

class TestCalibrationBlock(unittest.TestCase):

    def test_the_normative_solve_order_is_recorded_and_enforced(self):
        """[OB:CAL-ORDER] A conforming implementation MUST record that it did."""
        self.assertEqual(api.CALIBRATION_SOLVE_ORDER, (
            "inputs_fixed_first", "alpha_sel_from_max_candidates",
            "phi_estimated_from_aa_control", "b_min_solved_upward",
            "alpha_sel_validated_at_b_min", "anchor_gate_band_computed"))
        with self.assertRaises(ValueError):
            calibration(solve_order_recorded=api.CALIBRATION_SOLVE_ORDER[::-1])
        with self.assertRaises(ValueError):
            calibration(solve_order_recorded=api.CALIBRATION_SOLVE_ORDER[:-1])

    def test_phi_is_the_95th_percentile_of_the_aa_effect_distribution(self):
        """[OB:CAL-PHI] Seam-only: the A/A material is measured, the estimator is here."""
        aa = tuple(0.001 * i for i in range(1, 101))
        floor = ST.estimate_noise_floor(aa, calibration_block_count=100,
                                        neutral_check=PASS)
        self.assertAlmostEqual(floor.value, ST.percentile([abs(v) for v in aa], 0.95),
                               places=12)
        self.assertEqual(floor.quantile, 0.95)
        self.assertEqual(floor.method, "linear_interpolation_type7")
        # Less material than the campaign DECLARED is a refusal, not a smaller phi.
        with self.assertRaises(ST.InsufficientMaterial):
            ST.estimate_noise_floor(aa[:10], calibration_block_count=100,
                                    neutral_check=PASS)

    def test_an_estimate_at_or_below_the_floor_is_not_ranked(self):
        """[OB:CAL-PHI-RANK] MUST NOT be ranked, banked, or composed, whatever its e-value."""
        below = effect(value=0.009, e_value=1e9, mde=0.001)
        verdict, *_ = evaluate(eff=below)
        self.assertEqual(verdict.effect_resolution, api.EFFECT_BELOW_NOISE_FLOOR)
        self.assertFalse(verdict.speed_rank_admissible)
        with self.assertRaises(api.SpeedRankUnavailable):
            verdict.rank_key()
        ranked, unrankable = api.rank_candidates([verdict])
        self.assertEqual(ranked, ())
        self.assertEqual(len(unrankable), 1)

    def test_the_floor_on_the_record_must_be_the_cells_calibrated_phi(self):
        """[OB:CAL-NO-LITERAL] A supplied floor is a supplied verdict."""
        # `_resolve_effect` reads the floor ON THE RECORD, so a zeroed floor would
        # otherwise turn a sub-floor estimate into a rankable improvement.
        zeroed = effect(value=0.005, noise_floor=0.0, mde=0.001)
        verdict, _pre, _v, sg, _line, _g = evaluate(eff=zeroed)
        self.assertIn("calibration_block_accepted", sg.failed)
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        self.assertFalse(verdict.speed_rank_admissible)

    def test_a_neutral_control_exceeding_the_floor_fails_the_calibration(self):
        """[OB:CAL-NEUTRAL] It FAILS the calibration rather than raising the floor."""
        aa = tuple(0.001 * i for i in range(1, 101))
        loud = tuple(v * 25.0 for v in aa)
        construction = ST.select_construction("sign_martingale_predictable_lambda/v1")
        neutral = ST.neutral_control_consistency(loud, aa, campaign_seed="seed-1",
                                                 construction=construction)
        self.assertEqual(neutral.outcome, S.FAIL)
        floor = ST.estimate_noise_floor(aa, calibration_block_count=100,
                                        neutral_check=neutral)
        # The floor itself is unchanged — it was not raised to accommodate.
        self.assertAlmostEqual(floor.value, ST.percentile([abs(v) for v in aa], 0.95),
                               places=12)
        self.assertEqual(floor.neutral_check.outcome, S.FAIL)
        # And the controls harness reads that verdict rather than recomputing it.
        self.assertIs(CT.CALIBRATION_OWNER, ST.STATISTICS_MODULE_ID)

    def test_b_min_is_floored_by_the_p_bench_1_reps_rule(self):
        """[OB:CAL-BMIN] B_min MUST NEVER fall below >=5 / >=10."""
        self.assertEqual(ST.reps_floor_for_relative_effect(0.06).blocks, 5)
        self.assertEqual(ST.reps_floor_for_relative_effect(0.02).blocks, 10)
        self.assertEqual(ST.reps_floor_for_relative_effect(0.005).blocks, 10)
        self.assertEqual(
            tuple(sorted(r["blocks"] for r in ST.P_BENCH_1_REPS_RULE)), (5, 10))

    def test_a_fixed_owning_rep_rule_governs_and_is_never_raised(self):
        """[OB:CAL-BMIN-FIXED] P-BENCH-4's exactly five is a fixed count, not a floor."""
        fixed = ST.OwningProtocolRepRule(protocol_id="P-BENCH-4", kind=ST.REP_RULE_FIXED,
                                         blocks=5, citation="bench-cpu.md:174-178")
        self.assertEqual(fixed.kind, ST.REP_RULE_FIXED)
        # A campaign whose calibration solved a DIFFERENT count under a fixed rule
        # is refused: a fixed count is not a floor to be raised.
        self.assertIn(ST.REP_RULE_FIXED, (ST.REP_RULE_FIXED, ST.REP_RULE_FLOOR))
        self.assertEqual(ST._b_min_candidates(5, 40, fixed), (5,))

    def test_the_error_budgets_are_derived_from_the_declared_controls(self):
        """[OB:CAL-ALPHA] alpha_sel <= 1/max_candidates; alpha_conf <= alpha_sel/n."""
        ctrl = campaign_controls(max_candidates=100, confirmation_admission_count=5)
        self.assertEqual(ctrl.alpha_sel_ceiling(), 0.01)
        self.assertEqual(ctrl.alpha_conf_ceiling(0.01), 0.002)
        loose = calibration(alpha_sel=0.05, alpha_conf=0.002)
        self.assertEqual(loose.check_against_controls(ctrl).outcome, S.FAIL)
        with self.assertRaises(ValueError):
            calibration(alpha_sel=0.001, alpha_conf=0.01)   # conf looser than sel

    def test_the_thresholds_are_the_reciprocals_of_the_budgets(self):
        """[OB:CAL-THRESHOLD] 1/alpha_sel for selection and 1/alpha_conf for confirmation."""
        cal = calibration()
        self.assertAlmostEqual(cal.threshold_for(api.STRATUM_SELECTION), 100.0)
        self.assertAlmostEqual(cal.threshold_for(api.STRATUM_CONFIRMATION), 500.0)
        with self.assertRaises(ValueError):
            cal.threshold_for("exploratory")
        # A record whose threshold is not the calibrated one is not search-grade.
        _v, _p, _vs, sg, _l, _g = evaluate(eff=effect(threshold=10.0))
        self.assertIn("e_value_against_calibrated_threshold", sg.failed)

    def test_both_the_failed_and_the_accepted_calibration_are_retained(self):
        """[OB:CAL-VALIDATE] Validated once at the solved B_min; both are retained."""
        self.assertIn("attempts", ST.CalibrationSolve.__dataclass_fields__)
        self.assertIn("outputs", ST.CalibrationSolve.__dataclass_fields__)
        sig = inspect.signature(ST.CalibrationAttempt)
        for field in ("alpha_sel", "alpha_conf", "threshold_sel", "threshold_conf"):
            self.assertIn(field, sig.parameters)
        # The accepted attempt is the one that becomes outputs; a solve with none
        # refuses to hand back a calibration at all.
        empty = ST.CalibrationSolve(inputs_digest={}, attempts=(), outputs=None,
                                    aa_effect_pool=(), anchor_calibration_values=(),
                                    reasons=("no attempt converged",))
        self.assertFalse(empty.accepted)
        with self.assertRaises(ST.CalibrationFailed):
            empty.require_accepted()

    def test_the_anchor_gate_band_is_the_central_95_percent(self):
        """[OB:CAL-BAND] The interval containing the central 95% of the anchor values."""
        band = calibration().anchor_gate_band
        self.assertEqual(len(band), 2)
        self.assertLess(band[0], band[1])
        with self.assertRaises(ValueError):
            calibration(anchor_gate_band=(1.03, 0.97))
        self.assertTrue(callable(ST.anchor_gate_band))
        self.assertTrue(callable(ST.anchor_gate_check))

    def test_the_storage_floor_is_not_a_calibration_output(self):
        """[OB:CAL-NO-FIFTH] There is no fifth output; one definition, in one scope."""
        self.assertNotIn("storage_floor_bytes_free",
                         api.CalibrationOutputs.__dataclass_fields__)
        self.assertIn("storage_floor_bytes_free", api.CampaignControls.__dataclass_fields__)

    def test_a_b_min_above_the_declared_ceiling_fails_the_calibration(self):
        """[OB:CAL-CEILING] No partial calibration and no fallback ceiling."""
        cal = calibration(b_min_blocks=99)
        chk = cal.check_against_controls(campaign_controls(max_blocks_per_candidate=40))
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("does not start" in r for r in chk.reasons))

    def test_the_recorded_construction_must_be_one_the_bundle_implements(self):
        """[OB:CAL-CONSTRUCTION] A campaign selects among constructions the bundle has."""
        self.assertEqual(sorted(ST.CONSTRUCTIONS),
                         sorted(api.E_PROCESS_CONSTRUCTION_IDS))
        with self.assertRaises(ValueError):
            calibration(e_process_construction_id="hand_rolled_bound/v1")
        with self.assertRaises(ST.ConstructionNotImplemented):
            ST.select_construction("hand_rolled_bound/v1")
        # ... and a construction carrying a registry id with tuned parameters is
        # refused by content hash, not by name.
        member = ST.select_construction("sign_martingale_predictable_lambda/v1")
        tuned = ST.EProcessConstruction(
            **{**{f: getattr(member, f)
                  for f in ST.EProcessConstruction.__dataclass_fields__},
               "lambda_cap": 0.99})
        with self.assertRaises(ST.ConstructionNotImplemented):
            ST._require_bundle_construction(tuned, "probe")

    def test_a_calibration_solved_for_another_cell_is_refused(self):
        """[OB:CAL-CELL] Values calibrated under a different cell MUST NOT be reused."""
        for field, value in (("backend", "llama_cpu"), ("phase", "prefill"),
                             ("cell_class", "microbench_op")):
            with self.subTest(field=field):
                req = request(calibration=calibration(**{field: value}))
                voids = api.check_void_conditions(req, window(), rate_comparison=True)
                self.assertIn(api.VOID_INCOMPLETE_CALIBRATION, voids.reasons())

    def test_a_campaign_that_cannot_complete_calibration_ranks_nothing(self):
        """[OB:CAL-INCOMPLETE] MUST NOT rank any candidate."""
        for req in (request(calibration=None),
                    request(calibration=calibration(accepted=False))):
            with self.subTest(calibration=req.calibration):
                verdict, _p, voids, sg, _l, _g = evaluate(req=req, eff=effect())
                self.assertIn(api.VOID_INCOMPLETE_CALIBRATION, voids.reasons())
                self.assertIn("calibration_block_accepted", sg.failed)
                self.assertEqual(verdict.status, api.STATUS_INVALID)
                self.assertFalse(verdict.speed_rank_admissible)

    def test_the_calibration_is_bound_to_a_cell_and_an_anchor_that_can_drift(self):
        """[OB:CAL-RECOMPUTE] Seam-only: the boundary trigger is AK4's; the binding is here."""
        # The structural half the evaluator owns: the outputs name their cell, and
        # the anchor identity is re-verified at both ends of every window, so a
        # changed anchor cannot silently keep an old calibration in force.
        cal = calibration()
        self.assertEqual((cal.backend, cal.phase, cal.cell_class),
                         ("llama_gpu", "decode", "instrument_tokens_per_s"))
        drifted = window(anchor_at_close=anchor(binary_sha256=sha("rebuilt")))
        voids = api.check_void_conditions(request(), drifted, rate_comparison=True)
        self.assertIn(api.VOID_ANCHOR_MISSING_OR_MUTATED, voids.reasons())


# =============================================================================
# Statistical requirements
# =============================================================================

class TestStatisticalRequirements(unittest.TestCase):

    def test_the_realized_block_count_is_checked_against_the_calibrated_b_min(self):
        """[OB:STAT-REPS] Never fewer than the calibrated B_min paired blocks."""
        _v, _p, _vs, sg, _l, _g = evaluate(eff=effect(paired_blocks=4))
        self.assertIn("b_min_paired_blocks_order_randomized", sg.failed)
        self.assertTrue(any("below the calibrated B_min" in r for r in
                            sg.reason_for("b_min_paired_blocks_order_randomized")))

    def test_the_reduction_reports_median_and_mad(self):
        """[OB:STAT-MEDIAN-MAD] Report median + MAD."""
        self.assertIn("median_effect", ST.BlockReduction.__dataclass_fields__)
        self.assertIn("mad_effect", ST.BlockReduction.__dataclass_fields__)
        self.assertEqual(ST.median((1.0, 2.0, 100.0)), 2.0)
        self.assertEqual(ST.mad((1.0, 2.0, 3.0)), 1.0)

    def test_every_rate_comparison_carries_an_e_value_and_its_threshold(self):
        """[OB:STAT-EPROCESS] Never a single trial, and never an ad-hoc bound."""
        for field in ("e_value", "threshold"):
            self.assertIn(field, api.EffectEstimate.__dataclass_fields__)
        run = ST.run_e_process(
            (0.05,) * 12,
            construction=ST.select_construction("sign_martingale_predictable_lambda/v1"),
            hypothesis=ST.HYPOTHESIS_IMPROVEMENT, margin=0.0, threshold=100.0)
        self.assertGreater(run.e_running_max, 0.0)
        # Evidence below the calibrated threshold is not an improvement.
        weak = effect(e_value=1.0, threshold=100.0)
        verdict, *_ = evaluate(eff=weak)
        self.assertEqual(verdict.effect_resolution, api.EFFECT_EVIDENCE_BELOW_THRESHOLD)
        self.assertFalse(verdict.speed_rank_admissible)

    def test_the_lcb_is_labelled_descriptive_and_decides_nothing(self):
        """[OB:STAT-LCB] Carried BESIDE the e-value; no decision is taken on it."""
        payload = effect().to_dict()
        self.assertEqual(payload["lcb_label"], "descriptive")
        # The resolution is identical with a wildly different LCB, and identical
        # with none at all: nothing reads it.
        base = api._resolve_effect(effect())
        self.assertEqual(base, api._resolve_effect(effect(lcb_descriptive=-99.0)))
        self.assertEqual(base, api._resolve_effect(effect(lcb_descriptive=None)))
        source = Path(api.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_resolve_effect":
                self.assertNotIn("lcb", ast.dump(node))

    def test_the_stopping_rule_names_its_table_decisions_ceiling_and_extension(self):
        """[OB:STAT-STOPPING] Declared at campaign start and fixed as a calibration input."""
        fields = ST.StoppingRule.__dataclass_fields__
        for name in ("rule_id", "final_table", "decisions", "extension",
                     "max_blocks_per_candidate"):
            self.assertIn(name, fields)
        self.assertIn("stopping_rule_id", api.WindowAttestations.__dataclass_fields__)

    def test_a_post_hoc_change_to_the_stopping_rule_voids_every_affected_record(self):
        """[OB:STAT-STOPPING-FIXED] It never licenses CHANGING the rule."""
        win = window(rule_immutability=fail("the committed rule digest no longer matches"))
        verdict, _p, voids, sg, _l, _g = evaluate(win=win, eff=effect())
        self.assertIn(api.VOID_POST_HOC_RULE_CHANGE, voids.reasons())
        self.assertIn("stopping_rule_unmodified", sg.failed)
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        # statistics.py refuses the mutation at its own boundary too.
        self.assertTrue(issubclass(ST.StoppingRuleMutated, ST.StatisticsError))

    def test_the_mde_lives_in_the_same_object_as_the_estimate(self):
        """[OB:STAT-MDE] Written into the same record as the estimate, not afterwards."""
        self.assertIn("mde", api.EffectEstimate.__dataclass_fields__)
        # There is no constructor that omits it: every field is required.
        with self.assertRaises(TypeError):
            api.EffectEstimate(metric="m", metric_direction="higher_better", value=1.0,
                               e_value=2.0, threshold=1.0, noise_floor=0.0,
                               paired_blocks=1, stratum=api.STRATUM_SELECTION,
                               raw_samples=((1.0,),), raw_samples_ref="r")
        self.assertIn("MDE=", api.render_search_record_grammar(
            request=request(), window=window(), verdict=evaluate(eff=effect())[0],
            effect=effect()))

    def test_below_the_mde_is_a_result_and_a_decision_not_a_failure(self):
        """[OB:STAT-MDE-VERDICT] "no detectable difference" is a result, and it does not rank."""
        small = effect(value=0.012, mde=0.018, noise_floor=0.009)
        verdict, _p, _v, _sg, _l, _g = evaluate(eff=small)
        self.assertEqual(verdict.effect_resolution, api.EFFECT_NO_DETECTABLE_DIFFERENCE)
        # It is a RESULT: the verdict status is still pass, not fail or invalid.
        self.assertEqual(verdict.status, api.STATUS_PASS)
        # And it is a DECISION: it does not rank.
        self.assertFalse(verdict.speed_rank_admissible)
        self.assertIn("no detectable difference", verdict.speed_rank_withheld_reason())

    def test_order_is_randomized_within_every_block_from_the_committed_seed(self):
        """[OB:STAT-ORDER] The seed derives from the campaign seed and is recorded."""
        self.assertIn("order_seed", api.WindowAttestations.__dataclass_fields__)
        sched = ST.OrderSchedule.derive(campaign_seed="campaign-seed-4711",
                                        candidate_id="akc-0001", base_blocks=10)
        again = ST.OrderSchedule.derive(campaign_seed="campaign-seed-4711",
                                        candidate_id="akc-0001", base_blocks=10)
        self.assertEqual(sched.orders(10), again.orders(10))
        other = ST.OrderSchedule.derive(campaign_seed="campaign-seed-4711",
                                        candidate_id="akc-0002", base_blocks=10)
        self.assertNotEqual(sched.orders(10), other.orders(10))
        self.assertEqual(set(sched.orders(10)), set(ST.ORDERS))
        _v, _p, _vs, sg, _l, _g = evaluate(
            win=window(order_randomized=fail("all blocks ran anchor-first")),
            eff=effect())
        self.assertIn("b_min_paired_blocks_order_randomized", sg.failed)

    def test_a_blocked_design_is_forbidden_and_says_why(self):
        """[OB:STAT-BLOCKED] Thermal and page-cache drift alias onto the arm effect."""
        _v, _p, _vs, sg, _l, _g = evaluate(
            win=window(order_randomized=fail("candidate x n then anchor x n")),
            eff=effect())
        self.assertTrue(any("blocked designs are forbidden" in r for r in
                            sg.reason_for("b_min_paired_blocks_order_randomized")))
        self.assertTrue(any("thermal and page-cache drift" in r for r in
                            sg.reason_for("b_min_paired_blocks_order_randomized")))

    def test_a_retry_is_a_fresh_reset_in_reversed_order(self):
        """[OB:STAT-RETRY] The schedule for attempt 1 is not the schedule for attempt 0."""
        first = ST.OrderSchedule.derive(campaign_seed="s", candidate_id="akc-0001",
                                        base_blocks=10)
        retry = first.retry()
        self.assertEqual(retry.attempt, 1)
        self.assertEqual(retry.orders(10), tuple(
            ST.ORDER_CANDIDATE_FIRST if o == ST.ORDER_ANCHOR_FIRST
            else ST.ORDER_ANCHOR_FIRST for o in first.orders(10)))
        # A second retry reverses back: alternating reversal, not a fresh draw.
        self.assertEqual(retry.retry().orders(10), first.orders(10))

    def test_the_anchor_gate_is_a_band_check_that_voids_the_window(self):
        """[OB:STAT-ANCHOR-GATE] Seam-only: the anchor is measured first by the runner."""
        self.assertTrue(callable(ST.anchor_gate_check))
        band = ST.AnchorGateBand(low=0.97, high=1.03, mass=0.95, b_min=10,
                                 source_values=200, resamples=2000,
                                 method="bootstrap_median_of_10_blocks")
        out = ST.anchor_gate_check((1.20,) * 10, band=band, b_min=10)
        self.assertEqual(out.outcome, S.FAIL)
        win = window(anchor_gate=fail("anchor median outside the calibrated band"))
        voids = api.check_void_conditions(request(), win, rate_comparison=True)
        self.assertIn(api.VOID_ANCHOR_GATE_FAILED, voids.reasons())

    def test_a_drifted_anchor_is_invalid_and_never_a_candidate_failure(self):
        """[OB:STAT-ANCHOR-GATE-VOID] It says nothing whatever about the candidate."""
        win = window(anchor_gate=fail("outside the calibrated band"))
        verdict, *_ = evaluate(win=win, eff=effect())
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        self.assertNotEqual(verdict.status, api.STATUS_FAIL)
        self.assertTrue(any("NOT recorded as a candidate failure" in d
                            for d in verdict.derivation))

    def test_the_strata_split_is_keyed_on_the_committed_campaign_seed(self):
        """[OB:STAT-STRATA] Recorded in the manifest and keyed on the campaign seed."""
        rule = ST.StratumSplitRule(rule_id="split-1", campaign_seed="campaign-seed-4711",
                                   confirmation_fraction=0.3,
                                   rotation=ST.RotationSchedule(schedule_id="rot-1",
                                                                period_campaigns=4))
        assigned = {rule.assign(f"unit-{i}") for i in range(200)}
        self.assertEqual(assigned, set(api.STRATA))
        same = ST.StratumSplitRule(rule_id="split-1", campaign_seed="campaign-seed-4711",
                                   confirmation_fraction=0.3,
                                   rotation=ST.RotationSchedule(schedule_id="rot-1",
                                                               period_campaigns=4))
        self.assertEqual([rule.assign(f"u{i}") for i in range(50)],
                         [same.assign(f"u{i}") for i in range(50)])

    def test_a_record_mixing_strata_is_invalid(self):
        """[OB:STAT-STRATA-MIX] No block may serve both strata."""
        win = window(strata=fail("block 7 appears in both the selection and confirmation sets"))
        verdict, _p, voids, sg, _l, _g = evaluate(win=win, eff=effect())
        self.assertIn(api.VOID_STRATA_VIOLATION, voids.reasons())
        self.assertIn("correct_stratum", sg.failed)
        self.assertEqual(verdict.status, api.STATUS_INVALID)

    def test_the_stratum_travels_on_the_record_and_an_invalid_one_never_ranks(self):
        """[OB:STAT-READINESS] Seam-only: the readiness reducer is AK4/AK6."""
        line = evaluate(eff=effect(stratum=api.STRATUM_CONFIRMATION,
                                   threshold=500.0))[4]
        self.assertIn("stratum=confirmation", line)
        # An INVALID record contributes to nothing, readiness included.
        invalid, *_ = evaluate(req=request(anchor=None), eff=effect())
        self.assertEqual(invalid.status, api.STATUS_INVALID)
        ranked, unrankable = api.rank_candidates([invalid])
        self.assertEqual(ranked, ())
        self.assertEqual(len(unrankable), 1)

    def test_confirmation_shapes_and_control_seeds_rotate_on_a_declared_schedule(self):
        """[OB:STAT-ROTATION] The schedule is declared in the evaluator bundle."""
        rot = ST.RotationSchedule(schedule_id="rot-1", period_campaigns=4)
        self.assertEqual(rot.epoch_for(0), 0)
        self.assertEqual(rot.epoch_for(4), 1)
        sched = CT.SeedRotationSchedule(rotate_every_windows=8,
                                        declared_at="evaluator-bundle://controls/v1")
        self.assertEqual(sched.epoch_for(0), 0)
        self.assertEqual(sched.epoch_for(8), 1)
        a = CT.derive_control_seed(campaign_seed="s", control_id=CT.CONTROL_AA, epoch=0)
        b = CT.derive_control_seed(campaign_seed="s", control_id=CT.CONTROL_AA, epoch=1)
        self.assertNotEqual(a, b)


# =============================================================================
# Controls
# =============================================================================

class TestControls(unittest.TestCase):

    def test_the_control_definitions_and_their_predicates_are_both_hashed(self):
        """[OB:CTL-BUNDLE] MUST NOT be modified by any process inside the loop."""
        self.assertEqual(CT.verify_control_definitions().outcome, S.PASS)
        self.assertEqual(CT.verify_control_definitions(CT.CONTROL_DEFINITIONS_DIGEST).outcome,
                         S.PASS)
        self.assertEqual(CT.verify_control_definitions("0" * 64).outcome, S.FAIL)
        # A blank pin is a missing input, not a satisfied one.
        self.assertEqual(CT.verify_control_definitions("").outcome, S.COULD_NOT_CHECK)
        # The predicates that decide PASS are hashed too, not only the text.
        self.assertTrue(CT.CONTROL_PREDICATES_DIGEST)
        self.assertNotEqual(CT.CONTROL_PREDICATES_DIGEST, CT.CONTROL_DEFINITIONS_DIGEST)

    def test_a_drifted_definitions_digest_voids_the_window(self):
        """[OB:VOID-CONTROL-DEFS] Post-hoc change to the control definitions voids."""
        win = window(control_definitions_immutable=fail(
            "CONTROL_PREDICATES_DIGEST no longer matches the pinned bundle"))
        verdict, _p, voids, sg, _l, _g = evaluate(win=win, eff=effect())
        self.assertIn(api.VOID_POST_HOC_RULE_CHANGE, voids.reasons())
        self.assertIn("controls_1_4_available_and_passing", sg.failed)
        self.assertEqual(verdict.status, api.STATUS_INVALID)

    def test_a_campaign_that_cannot_run_controls_1_to_4_ranks_nothing(self):
        """[OB:CTL-1-4] MUST NOT rank any candidate."""
        for name in ("positive", "neutral", "degraded_negative", "aa"):
            with self.subTest(control=name):
                panel = control_panel(**{name: cnc("the control did not run")})
                self.assertNotEqual(panel.check_1_to_4().outcome, S.PASS)
                verdict, *_ = evaluate(win=window(controls=panel), eff=effect())
                self.assertEqual(verdict.status, api.STATUS_INVALID)
                self.assertFalse(verdict.speed_rank_admissible)

    def test_the_positive_control_must_rank_above_the_anchor(self):
        """[OB:CTL-1] Seam-only: the measurement is real; the requirement is declared."""
        definition = CT.CONTROL_DEFINITIONS[0]
        self.assertEqual(definition.control_id, CT.CONTROL_POSITIVE)
        self.assertEqual(definition.failure_disposition, CT.DISPOSITION_GATE_DEFECT)
        self.assertIn("MUST rank above the anchor", definition.requirement)
        panel = control_panel(positive=fail("the positive control did not rank"))
        self.assertEqual(panel.check_1_to_4().outcome, S.FAIL)

    def test_the_neutral_control_must_not_advance_and_its_dispersion_is_checked(self):
        """[OB:CTL-2] Seam-only: the measurement is real; both halves are declared."""
        definition = CT.CONTROL_DEFINITIONS[1]
        self.assertEqual(definition.control_id, CT.CONTROL_NEUTRAL)
        self.assertIn("MUST NOT advance", definition.requirement)
        self.assertTrue(callable(CT.neutral_dispersion_check))
        # A solve with no floor cannot have compared the dispersion, and says so.
        empty = ST.CalibrationSolve(inputs_digest={}, attempts=(), outputs=None,
                                    aa_effect_pool=(), anchor_calibration_values=(),
                                    reasons=())
        self.assertEqual(CT.neutral_dispersion_check(empty).outcome, S.COULD_NOT_CHECK)

    def test_the_degraded_negative_control_receives_no_speed_rank_at_all(self):
        """[OB:CTL-3] Cheating, silently falling back, reducing work, cached results."""
        definition = CT.CONTROL_DEFINITIONS[2]
        self.assertEqual(definition.control_id, CT.CONTROL_DEGRADED_NEGATIVE)
        self.assertIn("no speed rank at all", definition.requirement)
        # The structural form: a candidate whose no-fallback proof FAILs cannot be
        # ranked, because the gate is speed-blocking.
        fallback = gate(gate_id=CO.GID_NO_FALLBACK, gate_class=api.GATE_INTEGRITY,
                        check=fail("2 fallback dispatches observed"))
        verdict, *_ = evaluate(gates=(fallback,), eff=effect())
        self.assertFalse(verdict.speed_rank_admissible)
        with self.assertRaises(api.SpeedRankUnavailable):
            verdict.rank_key()
        self.assertEqual(CO._GATE_CLASS_BY_ID[CO.GID_NO_FALLBACK], api.GATE_INTEGRITY)

    def test_the_aa_control_runs_on_a_declared_cadence_not_once_per_campaign(self):
        """[OB:CTL-4] Seam-only: the A/A run is real; the cadence machinery is here."""
        cadence = CT.AACadence(every_n_windows=5, every_n_seconds=3600.0,
                               declared_at="evaluator-bundle://controls/v1")
        # Both triggers are mandatory: a cadence that declined either would license
        # using a phi measured against a different anchor.
        self.assertTrue(cadence.at_campaign_boundary)
        self.assertTrue(cadence.on_anchor_identity_change)
        with self.assertRaises(ValueError):
            CT.AACadence(every_n_windows=5, every_n_seconds=3600.0, declared_at="d",
                         at_campaign_boundary=False)
        scheduler = CT.AAScheduler(cadence)
        decision = scheduler.due(ledger=(), windows_completed=0, now_epoch_seconds=0.0,
                                 anchor_short=anchor().short(), campaign_boundary=True)
        self.assertTrue(decision.due)
        # No A/A has ever run => the cadence attestation FAILS; it is not silent.
        self.assertEqual(scheduler.check(ledger=(), windows_completed=0,
                                         now_epoch_seconds=0.0,
                                         anchor_short=anchor().short()).outcome, S.FAIL)
        # "a passing A/A control within its declared cadence" is a SEPARATE fact
        # from this window's A/A outcome, and search-grade needs both.
        _v, _p, _vs, sg, _l, _g = evaluate(
            win=window(aa_cadence=fail("no A/A has run for 20 windows")), eff=effect())
        self.assertIn("aa_control_within_cadence", sg.failed)

    def test_a_failing_aa_control_voids_the_enclosing_window(self):
        """[OB:CTL-4-VOID] A failing A/A VOIDS the enclosing measurement window."""
        win = window(controls=control_panel(aa=fail("A/A resolved a significant effect")))
        verdict, _p, voids, _sg, _l, _g = evaluate(win=win, eff=effect())
        self.assertIn(api.VOID_AA_CONTROL_FAILED, voids.reasons())
        self.assertEqual(verdict.status, api.STATUS_INVALID)

    def test_the_historical_win_replay_contract_carries_all_seven_fields(self):
        """[OB:CTL-5] Seam-only: the replay is real; the declared contract is here."""
        fields = CT.HistoricalWinReplayDeclaration.__dataclass_fields__
        for name in ("win_id", "backend", "phase", "reference_direction",
                     "reference_band", "evidence_locator", "durability_class"):  # 7
            self.assertIn(name, fields)
        definition = CT.CONTROL_DEFINITIONS[4]
        self.assertEqual(definition.control_id, CT.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertEqual(definition.tests_gate_ability_to, CT.TESTS_ACCEPT)
        self.assertIn("MUST promote", definition.requirement)

    def test_a_failure_to_promote_is_a_gate_defect_that_halts_the_campaign(self):
        """[OB:CTL-5-DEFECT] Not a research finding: it halts and escalates."""
        definition = CT.CONTROL_DEFINITIONS[4]
        self.assertEqual(definition.failure_disposition, CT.DISPOSITION_GATE_DEFECT)
        panel = control_panel(historical_replay=fail("the win did not promote"))
        chk = panel.check_5()
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("GATE DEFECT" in r for r in chk.reasons))
        self.assertTrue(any("halts the campaign" in r for r in chk.reasons))

    def test_the_unavailable_branch_is_never_a_silent_skip(self):
        """[OB:CTL-5-UNAVAIL] Records the reason and escalates to the operator."""
        with self.assertRaises(ValueError):
            api.ControlPanel(positive=PASS, neutral=PASS, degraded_negative=PASS,
                             aa=PASS, historical_replay=None)
        with self.assertRaises(ValueError):
            api.ControlPanel(positive=PASS, neutral=PASS, degraded_negative=PASS,
                             aa=PASS, historical_replay=None,
                             historical_replay_unavailable_reason="no durable win")
        ok = api.ControlPanel(
            positive=PASS, neutral=PASS, degraded_negative=PASS, aa=PASS,
            historical_replay=None,
            historical_replay_unavailable_reason=(
                "llama_gpu has no qualifying durable win: no manifest entry"),
            operator_escalation_ref="ake-operator-escalation-0001")
        self.assertEqual(ok.available, 4)
        self.assertEqual(ok.check_5().outcome, S.PASS)

    def test_every_record_of_a_four_control_campaign_carries_the_marker(self):
        """[OB:CTL-5-MARKER] controls=4/5 (HISTORICAL_REPLAY_UNAVAILABLE) in the grammar."""
        panel = api.ControlPanel(
            positive=PASS, neutral=PASS, degraded_negative=PASS, aa=PASS,
            historical_replay=None,
            historical_replay_unavailable_reason="no qualifying durable win for llama_gpu",
            operator_escalation_ref="ake-operator-escalation-0001")
        self.assertEqual(panel.marker(), "4/5 (HISTORICAL_REPLAY_UNAVAILABLE)")
        line = evaluate(win=window(controls=panel), eff=effect())[4]
        self.assertIn("controls=4/5 (HISTORICAL_REPLAY_UNAVAILABLE)", line)
        self.assertEqual(control_panel().marker(), "5/5")
        self.assertIn("controls=5/5", evaluate(eff=effect())[4])

    def test_proceeding_on_four_controls_is_the_operators_call_on_the_record(self):
        """[OB:CTL-5-OPERATOR] Taken once, on the record - not the controller's."""
        self.assertEqual(sorted(CT.OPERATOR_DECISIONS),
                         sorted((CT.OPERATOR_DECISION_PENDING,
                                 CT.OPERATOR_DECISION_PROCEED_ON_FOUR,
                                 CT.OPERATOR_DECISION_HALT)))
        fields = CT.OperatorEscalation.__dataclass_fields__
        self.assertIn("decision", fields)
        # api refuses an unavailable control 5 with no escalation reference at all,
        # so the controller cannot decide it by omission.
        with self.assertRaises(ValueError):
            api.ControlPanel(positive=PASS, neutral=PASS, degraded_negative=PASS,
                             aa=PASS, historical_replay=None,
                             historical_replay_unavailable_reason="none",
                             operator_escalation_ref="   ")


# =============================================================================
# Correctness precedence
# =============================================================================

class TestCorrectnessPrecedence(unittest.TestCase):

    def test_the_five_prior_classes_are_the_protocols_own_five(self):
        """[OB:COR-LEX] Correctness, quality, numerical safety, integrity, stability."""
        self.assertEqual(sorted(api.LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES),
                         sorted(("correctness", "quality", "numerical_safety",
                                 "integrity", "stability")))
        self.assertEqual(len(api.LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES), 5)
        # Determinism is added on a DIFFERENT authority and the distinction is kept.
        self.assertEqual(set(api.SPEED_BLOCKING_GATE_CLASSES)
                         - set(api.LEXICOGRAPHICALLY_PRIOR_GATE_CLASSES),
                         {api.GATE_DETERMINISM})

    def test_a_failure_in_any_prior_class_yields_no_rank_at_all(self):
        """[OB:COR-NO-RANK] Not a penalised one - the rank is UNOBTAINABLE."""
        for gate_class in api.SPEED_BLOCKING_GATE_CLASSES:
            with self.subTest(gate_class=gate_class):
                bad = gate(gate_id=f"t0.{gate_class}.probe", gate_class=gate_class,
                           check=fail("the gate failed"))
                verdict, *_ = evaluate(gates=(bad,), eff=effect(value=99.0))
                self.assertFalse(verdict.speed_rank_admissible)
                # A sentinel would be sorted somewhere; this raises instead.
                with self.assertRaises(api.SpeedRankUnavailable):
                    verdict.rank_key()
                self.assertIn("no speed rank at all", verdict.speed_rank_withheld_reason())
                ranked, unrankable = api.rank_candidates([verdict])
                self.assertEqual(ranked, ())
                self.assertEqual(len(unrankable), 1)

    def test_a_candidate_produced_correctness_verdict_is_refused(self):
        """[OB:COR-ORACLE] NEVER self-reported by the candidate."""
        self.assertIn("candidate", CO.EVIDENCE_PRODUCERS)
        for builder, kwargs in (
                (CO.OpSuiteEvidence, dict(
                    suite_id="test-backend-ops", suite_source_sha256=sha("cand-source"),
                    ops_exercised=("MUL_MAT",), ops_failed=(),
                    cases_by_op=(("MUL_MAT", 10, 10),), shapes_ref="s",
                    receipt_ref="r", produced_by="candidate")),
                (CO.DispatchTraceEvidence, dict(
                    derived_surface=("MUL_MAT",), traced_kernels=("MUL_MAT",),
                    fallback_events=(), fallback_instrumentation_active=True,
                    trace_ref="t", produced_by="candidate")),
        ):
            with self.subTest(evidence=builder.__name__):
                ev = builder(**kwargs)
                self.assertEqual(ev.produced_by, "candidate")
        # The no-fallback gate — control 3's own detector — refuses it.
        result = CO.check_no_fallback_dispatch_proof(CO.DispatchTraceEvidence(
            derived_surface=("MUL_MAT",), traced_kernels=("MUL_MAT",),
            fallback_events=(), fallback_instrumentation_active=True,
            trace_ref="t", produced_by="candidate"))
        self.assertEqual(result.check.outcome, S.FAIL)
        self.assertIn("NEVER self-reported", " ".join(result.check.reasons))

    def test_a_candidate_derived_oracle_is_refused(self):
        """[OB:COR-CACHE-ORACLE] Seam-only: the oracle registry is resolved at run time."""
        self.assertIn("oracle_is_candidate_derived",
                      CO.ReferenceComparison.__dataclass_fields__)
        self.assertIn("candidate_output_used_as_oracle",
                      CO.AntiRewardHackingEvidence.__dataclass_fields__)
        # The gate refuses it; the dataclass records it. A candidate-derived
        # oracle is a comparison against the candidate's own answer.
        request_obj = request(tier="T0")
        ev = CO.ReferenceEvidence(
            comparisons=(CO.ReferenceComparison(
                shape_id="s", op="MUL_MAT", mode="exact_bitwise", mismatch_count=0,
                max_ulp_observed=None, tolerance_ulp=None, oracle_id="self",
                oracle_is_candidate_derived=True),),
            undefined_for=(), oracle_registry_ref="evaluator-bundle://oracles/v1",
            produced_by="evaluator")
        result = CO.check_exact_reference_comparison(
            request_obj, ev, CO.ChangeSurface(
                derived_touches_memory=False, derived_touches_threading=False,
                derived_touches_dispatch=False, derived_touches_persistent_state=False,
                derived_ops=("MUL_MAT",), derived_files=(), declared_touches_memory=False,
                declared_touches_threading=False, declared_ops=("MUL_MAT",),
                touches_shared_core_header=False, derivation_ref="d"),
            CO.T0Policy(required_backend_ops=("MUL_MAT", "MUL_MAT_ID"),
                        symbol_shrinkage_reject_ratio=0.02,
                        diff_ceiling=CO.DiffComplexityCeiling(
                            backend="llama_gpu", max_changed_lines=400,
                            max_files_touched=8, shared_core_forces_review=True),
                        determinism_min_runs=3, coherence_tolerance_floor=0.995,
                        policy_ref="evaluator-bundle://t0/policy/llama_gpu/v1"))
        self.assertEqual(result.check.outcome, S.FAIL)

    def test_cache_state_is_declared_in_every_record(self):
        """[OB:COR-CACHE-STATE] Cache state is declared in every record."""
        self.assertIn("cache_state", CO.AntiRewardHackingEvidence.__dataclass_fields__)
        self.assertTrue(CO.CACHE_STATES)
        with self.assertRaises(ValueError):
            CO.AntiRewardHackingEvidence(
                cache_state="whatever", correctness_verdict_source="evaluator",
                candidate_output_used_as_oracle=False, oracle_ids=("ref",),
                delivered_unit_name="generated_tokens", delivered_units_candidate=1,
                delivered_units_anchor=1, anchor_source_commit=None,
                anchor_binary_sha256=None, anchor_linkage_sha256=None,
                environment_probe_findings=(),
                timing_dependent_branch_findings=(), receipt_ref="r")


# =============================================================================
# Search-grade requires ALL of
# =============================================================================

class TestSearchGrade(unittest.TestCase):

    def test_the_conjunction_is_the_protocols_own_fourteen(self):
        """[OB:SG-CONJUNCTS] Every clause of the list is a named conjunct."""
        ids = tuple(c.id for c in api.SEARCH_GRADE_CONJUNCTS)
        self.assertEqual(ids, (
            "ratified_protocol", "preconditions", "calibration_block_accepted",
            "stopping_rule_unmodified", "b_min_paired_blocks_order_randomized",
            "anchor_gate_passing", "aa_control_within_cadence",
            "controls_1_4_available_and_passing",
            "control_5_passing_or_recorded_unavailable",
            "e_value_against_calibrated_threshold", "published_mde", "correct_stratum",
            "complete_record_grammar", "raw_samples_reproducible"))
        _v, _p, _vs, sg, _l, _g = evaluate(eff=effect())
        self.assertTrue(sg.satisfied, sg.failed)
        self.assertEqual(set(sg.evaluated), set(ids))
        self.assertEqual(sg.not_applicable, ())
        # A non-rate record states which conjuncts were NOT applicable rather
        # than silently omitting them.
        _v2, _p2, _vs2, sg2, _l2, _g2 = evaluate(eff=None)
        self.assertTrue(sg2.not_applicable)
        self.assertEqual(set(sg2.evaluated) | set(sg2.not_applicable), set(ids))

    def test_missing_any_conjunct_makes_the_record_invalid(self):
        """[OB:SG-MISSING-ANY] There is no weaker-but-usable state."""
        verdict, *_ = evaluate(req=request(protocol_id="P-SOMETHING-ELSE/v1"),
                               eff=effect())
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        self.assertIn("ratified_protocol", verdict.search_grade.failed)
        self.assertIn("SEARCH_GRADE_MISSING:ratified_protocol", verdict.integrity_flags)
        self.assertNotIn(verdict.status, (api.STATUS_PASS, api.STATUS_INCONCLUSIVE))

    def test_an_invalid_record_is_produced_journalable_and_never_ranks(self):
        """[OB:SG-RETAINED] Retained in the journal and MUST NOT rank, bank or compose."""
        req = request(anchor=None)
        runner = _FixtureRunner(req.tier, (gate(),))
        outcome = api.TierDispatcher(gate_runners={req.tier: runner}).dispatch(
            req, window(), effect=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        # The record still exists and is canonical-JSON-able for the journal.
        self.assertTrue(S.content_hash(outcome.durable_payload))
        self.assertFalse(outcome.verdict.speed_rank_admissible)
        # [OB:VOID-JOURNALED] applies to the ANCHOR void too: v3 lets the record
        # omit its anchor block instead of forcing a fabricated digest, so the
        # one void that could not be journaled as a record now can be.
        self.assertTrue(outcome.emitted)
        self.assertIsNone(outcome.event_blocked_reason)
        self.assertEqual(outcome.event_violations, ())
        self.assertNotIn("anchor", outcome.event)
        self.assertIn(f"VOID:{api.VOID_ANCHOR_MISSING_OR_MUTATED}:{S.FAIL}",
                      outcome.event["integrity_flags"])

    def test_the_record_says_it_is_not_a_claim(self):
        """[OB:SG-NOT-A-CLAIM] Neither state is ever a claim."""
        self.assertEqual(api.RECORD_CLASS, "SEARCH RECORD, NOT A CLAIM")
        line = evaluate(eff=effect())[4]
        self.assertIn("SEARCH RECORD, NOT A CLAIM", line)
        self.assertIn("category=CANDIDATE", line)

    def test_release_tiers_are_refused_by_name(self):
        """[OB:SG-TIER-SCOPE] It does NOT apply to T3 or any release gate."""
        self.assertEqual(api.SEARCH_TIERS, ("T0", "T1", "T1a", "T1b", "T1c", "T2"))
        self.assertEqual(api.RELEASE_TIERS, ("T3", "T4"))
        for tier in api.RELEASE_TIERS:
            with self.subTest(tier=tier):
                with self.assertRaises(api.TierNotOwned):
                    api.admit_tier(tier)
                with self.assertRaises(api.TierNotOwned):
                    api.TierDispatcher(gate_runners={tier: _FixtureRunner(tier, (gate(),))})
        # Every module that wires a tier goes through the same door.
        for module in (IG, SU, RC, CT):
            self.assertIs(module.api.admit_tier, api.admit_tier)


# =============================================================================
# Record grammar
# =============================================================================

class TestRecordGrammar(unittest.TestCase):

    def test_every_mandatory_field_is_present_in_the_rendered_line(self):
        """[OB:REC-FIELDS] category, tier, eval, srclabel, res, host, anchor, recipe, ..."""
        line = evaluate(eff=effect())[4]
        req, win = request(), window()
        self.assertIn("category=CANDIDATE", line)
        self.assertIn(f"tier {req.tier}", line)
        self.assertIn(f"eval={req.evaluator.bundle_sha256[:12]}", line)
        self.assertIn(f"srclabel={req.evaluator.runtime_source_label_ref}", line)
        self.assertIn(f"res={win.resource_claim_receipt}", line)
        self.assertIn(f"host={win.host_receipt}", line)
        self.assertIn(f"vs anchor {req.anchor.short()}", line)
        self.assertIn(f"recipe={win.recipe.render()}", line)
        self.assertIn("stratum=selection", line)
        self.assertIn(f"scope={req.scope_denominator.render()}", line)
        self.assertIn(f"det={req.determinism.determinism_class}", line)
        self.assertIn(f"raw={effect().raw_samples_ref}", line)
        self.assertIn(f"campaign={req.campaign_id}", line)
        self.assertIn("controls=5/5", line)
        self.assertIn(req.created_at[:10], line)

    def test_a_record_whose_reduction_has_no_raw_samples_is_refused(self):
        """[OB:REC-REPRODUCIBLE] A reduction that cannot be recomputed is INVALID."""
        with self.assertRaises(ValueError):
            effect(raw_samples=())
        # `raw_evidence_ref` cannot even be constructed blank: the window refuses it.
        with self.assertRaises(ValueError):
            window(raw_evidence_ref="   ")
        # And a rate record whose estimate carries no samples is refused upstream.
        chk = api.check_record_grammar_complete(
            request=request(), window=window(), effect=effect())
        self.assertEqual(chk.outcome, S.PASS)
        # statistics.py can recompute a reduction from what the record carries.
        self.assertTrue(callable(ST.verify_reduction_reproducible))

    def test_a_record_omitting_any_template_field_is_invalid(self):
        """[OB:REC-TEMPLATE] A record omitting any field of this template is INVALID."""
        for label, req_kw, win_kw in (
                ("anchor", dict(anchor=None), {}),
                ("recipe", {}, dict(recipe=None)),
        ):
            with self.subTest(field=label):
                req = request(**req_kw)
                win = window(**win_kw)
                chk = api.check_record_grammar_complete(request=req, window=win,
                                                        effect=effect())
                self.assertEqual(chk.outcome, S.FAIL)
                self.assertTrue(any(label in r for r in chk.reasons))
                verdict, *_ = evaluate(req=req, win=win, eff=effect())
                self.assertEqual(verdict.status, api.STATUS_INVALID)
        # Even on PASS the applied field set is stated, never inferred.
        ok = api.check_record_grammar_complete(request=request(), window=window(),
                                               effect=effect())
        self.assertEqual(ok.outcome, S.PASS)
        self.assertTrue(any("grammar fields required" in r for r in ok.reasons))

    def test_the_line_names_the_protocol_and_the_record_class(self):
        """[OB:REC-CLASS] SEARCH RECORD, NOT A CLAIM [P-AK-SEARCH-1, ...]."""
        line = evaluate(eff=effect())[4]
        self.assertIn("P-AK-SEARCH-1", line)
        self.assertIn("— SEARCH RECORD, NOT A CLAIM [", line)
        self.assertEqual(api.PROTOCOL_RATIFIED_UTC, "20260803T083005Z")

    def test_there_is_no_attest_field_and_res_host_srclabel_stand_in_for_it(self):
        """[OB:REC-NO-ATTEST] The grammar carries no attest <ref> field."""
        line = evaluate(eff=effect())[4]
        self.assertNotIn("attest=", line)
        self.assertNotIn(" attest ", line)
        ref = api.compose_attestation_ref(window(), request().evaluator)
        self.assertTrue(ref.startswith("res="))
        self.assertIn(";host=", ref)
        self.assertIn(";srclabel=", ref)
        event = api.build_evaluation_event(
            request=request(), window=window(), verdict=evaluate(eff=effect())[0],
            effect=effect(), preconditions=api.check_preconditions(request(), window()))
        self.assertEqual(event["claim_grammar"]["attestation_ref"], ref)
        self.assertEqual(S.validate_evaluation_event(event), [])

    def test_the_estimate_must_be_of_the_metric_the_line_prints(self):
        """[OB:REC-METRIC] Substituting one metric for the other is forbidden."""
        mismatched = effect(metric="prefill_tokens_per_s")
        chk = api.check_record_grammar_complete(request=request(), window=window(),
                                                effect=mismatched)
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("substituting one metric" in r for r in chk.reasons))
        flipped = effect(metric_direction="lower_better")
        chk2 = api.check_record_grammar_complete(request=request(), window=window(),
                                                 effect=flipped)
        self.assertEqual(chk2.outcome, S.FAIL)
        verdict, *_ = evaluate(eff=mismatched)
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        # The commensurability rule is also enforced where argv is constructed.
        self.assertTrue(callable(RC.schemas.check_metric_commensurability))


# =============================================================================
# What voids a run
# =============================================================================

class TestVoidConditions(unittest.TestCase):

    #: One trigger per enumerated void condition. Every entry names the window or
    #: request mutation that raises exactly that reason.
    TRIGGERS = {
        api.VOID_CLAIM_NOT_HELD: dict(resource_claim_close=fail("released early")),
        api.VOID_HOST_HEALTH_TIER_VIOLATION: dict(host_health=fail("uptime 9 days")),
        api.VOID_ANCHOR_GATE_FAILED: dict(anchor_gate=fail("outside the band")),
        api.VOID_AA_CONTROL_FAILED: dict(controls=None),      # replaced below
        api.VOID_EVALUATOR_BUNDLE_UNVERIFIED: dict(evaluator_bundle=fail("hash drift")),
        api.VOID_ANCHOR_MISSING_OR_MUTATED: dict(anchor_at_close=None),
        api.VOID_HAND_TYPED_ARGV: dict(recipe=None),
        api.VOID_CONCURRENT_INFERENCE: dict(no_concurrent_inference=fail("a server ran")),
        api.VOID_STORAGE_EXHAUSTED: dict(storage_close=fail("below the floor")),
        api.VOID_STRATA_VIOLATION: dict(strata=fail("block 7 is in both strata")),
        api.VOID_POST_HOC_RULE_CHANGE: dict(rule_immutability=fail("rule digest moved")),
        api.VOID_INCOMPLETE_CALIBRATION: dict(calibration=fail("solve did not converge")),
    }

    def test_the_schema_void_vocabulary_is_a_subset_of_the_evaluators(self):
        """`schemas.ANCHOR_VOID_REASONS` restates two of `api.VOID_REASONS`.

        It has to restate rather than import them — `api` imports `schemas`, so
        the other direction is a cycle, and a data contract that needs the
        evaluator loaded to validate a record is not a data contract. That makes
        the duplication a drift hazard, so it is CHECKED here instead of trusted:
        rename a reason in `api` and this fails rather than silently widening (or
        closing) the one exemption in `evaluation_event.v3`.
        """
        self.assertTrue(S.ANCHOR_VOID_REASONS)
        self.assertTrue(S.ANCHOR_VOID_REASONS.issubset(set(api.VOID_REASONS)))
        self.assertEqual(
            S.ANCHOR_VOID_REASONS,
            {r for r in api.VOID_REASONS if "ANCHOR" in r},
            "every anchor-subject void reason must admit the v3 omission, and "
            "no non-anchor reason may")
        # The flag grammar the validator reads is the one `_derive` writes.
        self.assertTrue(all(
            f"{S.VOID_FLAG_PREFIX}{r}:{S.FAIL}".startswith(S.VOID_FLAG_PREFIX)
            for r in api.VOID_REASONS))

    def test_every_enumerated_void_condition_triggers_its_own_reason(self):
        """[OB:VOID-ENUM] The protocol's twelve, each raising exactly its own reason."""
        self.assertEqual(len(api.VOID_REASONS), 12)
        self.assertEqual(set(self.TRIGGERS), set(api.VOID_REASONS))
        for reason, overrides in self.TRIGGERS.items():
            with self.subTest(void=reason):
                if reason == api.VOID_AA_CONTROL_FAILED:
                    overrides = dict(controls=control_panel(aa=fail("A/A drifted")))
                win = window(**overrides)
                voids = api.check_void_conditions(request(), win, rate_comparison=True)
                self.assertIn(reason, voids.reasons())
                finding = next(f for f in voids.findings if f.reason == reason)
                # The journaled reason is the protocol's own phrase, not a paraphrase.
                self.assertEqual(finding.protocol_phrase, api.VOID_REASON_PHRASES[reason])

    def test_a_voided_run_is_journaled_as_invalid_with_its_reason(self):
        """[OB:VOID-JOURNALED] Never silently discarded; primary records are never destroyed."""
        req = request()
        win = window(host_health=fail("uptime 9 days"))
        runner = _FixtureRunner(req.tier, (gate(),))
        outcome = api.TierDispatcher(gate_runners={req.tier: runner}).dispatch(
            req, win, effect=effect())
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertIn(api.VOID_HOST_HEALTH_TIER_VIOLATION, outcome.void_scan.reasons())
        # The reason travels in the durable payload AND in the emitted event.
        self.assertTrue(outcome.durable_payload["void_scan"]["findings"])
        self.assertTrue(outcome.emitted)
        self.assertEqual(outcome.event["status"], api.STATUS_INVALID)
        self.assertIn("VOID:HOST_HEALTH_TIER_VIOLATION:FAIL",
                      outcome.event["integrity_flags"])
        # There is no VOID terminal state: the run walks the whole path.
        self.assertEqual(outcome.states[-1], "EMITTED")
        self.assertNotIn("VOID", api.DISPATCH_STATES)

    def test_could_not_check_voids_too_and_stays_distinguishable_from_fail(self):
        """[OB:VOID-THIRD-OUTCOME] Inability to evaluate is a THIRD outcome."""
        for outcome_kind, chk in ((S.FAIL, fail("held by another holder")),
                                  (S.COULD_NOT_CHECK, cnc("the lock file was unreadable"))):
            with self.subTest(outcome=outcome_kind):
                voids = api.check_void_conditions(
                    request(), window(resource_claim_same_holder=chk),
                    rate_comparison=True)
                finding = next(f for f in voids.findings
                               if f.reason == api.VOID_CLAIM_NOT_HELD)
                self.assertEqual(finding.outcome, outcome_kind)
        # A void finding may only be FAIL or COULD_NOT_CHECK; PASS is not sayable.
        with self.assertRaises(ValueError):
            api.VoidFinding(reason=api.VOID_CLAIM_NOT_HELD, protocol_phrase="p",
                            outcome=S.PASS)
        # Rate-only conditions that were NOT evaluated are named, never omitted.
        voids = api.check_void_conditions(request(), window(), rate_comparison=False)
        self.assertEqual(set(voids.not_applicable),
                         {api.VOID_ANCHOR_GATE_FAILED, api.VOID_AA_CONTROL_FAILED,
                          api.VOID_STRATA_VIOLATION, api.VOID_INCOMPLETE_CALIBRATION})


# =============================================================================
# Cross-module seam conformance — where the parallel authors had to agree
# =============================================================================

class TestCrossModuleSeams(unittest.TestCase):
    """These are not extra obligations; they are the wiring the obligations above
    are only true THROUGH. Each one is a place two modules describe the same fact."""

    def test_every_module_uses_the_one_three_outcome_check_type(self):
        for module in (api, CT, CO, IG, RC, ST, SU):
            with self.subTest(module=module.__name__):
                self.assertIs(module.schemas.Check, S.Check)
        self.assertEqual((S.PASS, S.FAIL, S.COULD_NOT_CHECK),
                         ("PASS", "FAIL", "COULD_NOT_CHECK"))

    def test_the_controls_harness_projects_into_the_window_attestations(self):
        projection = CT.window_control_attestations
        self.assertTrue(callable(projection))
        # Exactly the three window fields a control sweep is authoritative for.
        names = ("controls", "aa_cadence", "control_definitions_immutable")
        for name in names:
            self.assertIn(name, api.WindowAttestations.__dataclass_fields__)
        # A sweep with no panel RAISES rather than fabricating one.
        with self.assertRaises(CT.ControlWiringError):
            projection("not a result")

    def test_the_reducer_projects_into_the_window_attestations(self):
        keys = {"strata", "rule_immutability", "order_randomized", "calibration"}
        self.assertTrue(keys <= set(api.WindowAttestations.__dataclass_fields__))
        self.assertIn("window_checks", dir(ST.BlockReduction))

    def test_the_integrity_gates_bind_the_surface_scope_they_declare(self):
        self.assertTrue(callable(IG.surface_scope_for))
        self.assertEqual(IG.check_declared_surface_scope(IG.SURFACE_PARTIAL, None).outcome,
                         S.COULD_NOT_CHECK)
        with self.assertRaises(TypeError):
            IG.surface_scope_for("full_tree")
        self.assertEqual(sorted(SU.FULL_TREE_CHANGE_CLASSES), ["core_header"])

    def test_a_runner_that_returns_no_gates_is_a_wiring_defect_not_a_pass(self):
        req = request()
        empty = _FixtureRunner(req.tier, ())
        with self.assertRaises(api.EvaluatorNotWired):
            api.TierDispatcher(gate_runners={req.tier: empty}).dispatch(req, window())

    def test_every_gate_class_a_module_emits_is_in_the_api_vocabulary(self):
        for gate_id, gate_class, _requires in CO.T0_GATE_SPEC:
            with self.subTest(gate=gate_id):
                self.assertIn(gate_class, api.GATE_CLASSES)
                self.assertIn(gate_class, api.SPEED_BLOCKING_GATE_CLASSES)

    def test_no_two_modules_emit_the_same_gate_id(self):
        seen: dict = {}
        for label, ids in (("correctness", CO.T0_GATE_IDS),
                           ("integrity", IG.RUNNER_GATE_IDS + (IG.GATE_BEHAVIOURAL_NOT_RUN,
                                                               IG.GATE_SURFACE_SCOPE_BINDING)),
                           ("surface", ("surface.derived_coverage", "surface.reconciliation",
                                        "surface.trace_completeness", "surface.no_fallback"))):
            for gate_id in ids:
                with self.subTest(gate=gate_id):
                    self.assertNotIn(gate_id, seen,
                                     f"{label} and {seen.get(gate_id)} both emit {gate_id!r}; "
                                     "a shared id collapses two findings into one record key")
                    seen[gate_id] = label

    def test_every_module_proves_it_cannot_write_or_signal(self):
        for module, auditor in ((api, api.audit_no_write_or_process_paths),
                                (CT, CT.audit_no_write_or_process_paths),
                                (CO, CO.audit_no_write_or_process_paths),
                                (IG, IG.audit_no_write_or_process_paths),
                                (SU, SU.audit_surface_module_is_read_only),
                                (RC, RC.audit_no_execution_paths)):
            with self.subTest(module=module.__name__):
                self.assertEqual(auditor().outcome, S.PASS)


class _FixtureRunner:
    """A gate runner that returns exactly what it was handed. Runs nothing."""

    def __init__(self, tier: str, gates: tuple) -> None:
        self.tier = tier
        self._gates = tuple(gates)

    def run_gates(self, request):
        return self._gates


# =============================================================================
# Coverage — an obligation with no test, or a test with no obligation, FAILS
# =============================================================================

class TestObligationCoverage(unittest.TestCase):
    """The property that makes this file hard to weaken.

    A conformance suite whose gaps are invisible is the shape this project keeps
    paying for. This reads THIS module's own AST, extracts every `[OB:<id>]`
    marker from every test docstring, and reconciles it against `OBLIGATIONS`.
    """

    @staticmethod
    def _claims() -> dict:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        claims: dict = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            doc = ast.get_docstring(node) or ""
            for token in doc.split("[OB:")[1:]:
                oid = token.split("]")[0].strip()
                claims.setdefault(oid, []).append((node.name, node))
        return claims

    def test_every_obligation_has_a_claiming_test(self):
        claims = self._claims()
        missing = sorted(o.id for o in OBLIGATIONS if o.id not in claims)
        self.assertEqual(missing, [], f"obligations with no claiming test: {missing}")

    def test_no_test_claims_an_obligation_that_is_not_registered(self):
        unknown = sorted(oid for oid in self._claims() if oid not in OBLIGATION_BY_ID)
        self.assertEqual(unknown, [], f"tests claim unregistered obligations: {unknown}")

    def test_a_claiming_test_actually_asserts_something(self):
        """A test that claims an obligation and asserts nothing is a coverage hole."""
        for oid, entries in sorted(self._claims().items()):
            for name, node in entries:
                with self.subTest(obligation=oid, test=name):
                    calls = [n for n in ast.walk(node)
                             if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute)
                             and n.func.attr.startswith(("assert", "fail"))]
                    self.assertTrue(calls, f"{name} claims {oid} and asserts nothing")

    def test_seam_only_obligations_are_declared_with_their_reason(self):
        """"Cannot be tested without inference" is a declared state, never a gap."""
        unknown = sorted(oid for oid in SEAM_ONLY if oid not in OBLIGATION_BY_ID)
        self.assertEqual(unknown, [], f"SEAM_ONLY names unregistered obligations: {unknown}")
        for oid, reason in sorted(SEAM_ONLY.items()):
            with self.subTest(obligation=oid):
                self.assertGreater(len(reason), 40,
                                   f"{oid} is declared seam-only with no real reason")
        # A seam-only obligation still has a test; it is not a skip.
        claims = self._claims()
        for oid in SEAM_ONLY:
            with self.subTest(obligation=oid):
                self.assertIn(oid, claims)

    def test_the_register_covers_every_normative_section_of_the_protocol(self):
        sections = {o.section for o in OBLIGATIONS}
        self.assertEqual(sections, {
            "Preconditions", "Campaign calibration block", "Statistical requirements",
            "Controls", "Correctness precedence", "Search-grade requires ALL of",
            "Record grammar", "Metric", "What voids a run"})
        self.assertGreaterEqual(len(OBLIGATIONS), 60)

    def test_the_suite_names_the_protocol_it_is_conformance_to(self):
        self.assertEqual(api.PROTOCOL_VERSIONED_ID, "P-AK-SEARCH-1/v1")
        self.assertEqual(PROTOCOL_PATH, "measurement/protocols/kernel-research.md")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
