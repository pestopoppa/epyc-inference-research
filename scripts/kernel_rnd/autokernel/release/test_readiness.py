#!/usr/bin/env python3
"""test_readiness.py — the regression barrier for the T2 estimator's guarantees.

WHY THIS FILE EXISTS
--------------------
`readiness.py` is where three specific mistakes would be cheapest to make and
most expensive to detect: a number that came out of a sentence, a scalar folding
two backends, and a signal that quietly became a trigger. Each of those is
unrepresentable in the module; this suite is what keeps them unrepresentable.

Organised by the obligation under test rather than by function, because that is
how an auditor reads it:

  * `measurement/protocols/kernel-research.md` — P-AK-SEARCH-1 (Annex K, RATIFIED
    2026-08-03): authorization 5 (advisory readiness signal computed by a
    deterministic reducer over journaled records), denial 5 (a readiness signal is
    not a freeze trigger), denial 9 (no new instrument by composition), the
    selection/confirmation split, correctness precedence, the controls marker,
    and P-AK-SEARCH-1-A1 clause 1 (mechanism plausibility).
  * `handoffs/active/autokernel-research-loop.md` — §1.2, §1.6, §9.6, §9.7, §9.8,
    §4 invariants 14 and 15, AK-D3, AK-D4, AK-D12, AK-D22.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO MODEL CALL, NO PROCESS, NO FILE WRITTEN.
Every e-process here is run over synthetic per-block effects by
`evaluator/statistics.py` itself, so the fixtures exercise the real reducer
rather than a hand-built stand-in for one.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/release/test_readiness.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/release/test_readiness.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import unittest

# RELATIVE, not `sys.path.insert` + `from autokernel import …`. Under
# `unittest discover -t .` the flat idiom loads a SECOND copy of `schemas` and
# `evaluator.api`, so `isinstance(x, api.Verdict)` is False for a Verdict this
# suite built and every cross-module guard degrades to a silent no-op. README,
# "Import convention".
from .. import schemas as S
from ..evaluator import api
from ..evaluator import statistics as st
from . import readiness as R

PASS = S.Check(S.PASS)
NOW = "2026-08-03T12:00:00+00:00"
LATER = "2026-08-03T14:00:00+00:00"
ENTERED = "2026-08-03T10:00:00+00:00"
BEFORE = "2026-08-03T09:00:00+00:00"

CHAMPION_ID = "akc-champion-0001"
MEMBER_A = "akc-member-0001"
MEMBER_B = "akc-member-0002"
CAMPAIGN = "ak-llama_cpu-20260803"

CONSTRUCTION = st.select_construction("sign_martingale_predictable_lambda/v1")
THRESHOLD = 100.0
NI_MARGIN = 0.02

DECODE_PROTOCOL = "P-BENCH-1"
PREFILL_PROTOCOL = "P-BENCH-PREFILL-1"
GPU_PROTOCOL = "P-GPU-1"


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def anchor(tag: str = "v8") -> api.AnchorIdentity:
    return api.AnchorIdentity(
        source_commit=hashlib.sha1(f"{tag}-commit".encode("utf-8")).hexdigest(),
        binary_sha256=sha(f"{tag}-binary"),
        linkage_sha256=sha(f"{tag}-linkage"),
        measurement_event_ids=(f"ake-anchor-{tag}",),
    )


def gates_ok() -> tuple:
    return (
        api.GateResult("op_reference_shapes", api.GATE_CORRECTNESS, PASS,
                       requires_anchor=True),
        api.GateResult("no_fallback_dispatch", api.GATE_INTEGRITY, PASS),
        api.GateResult("ppl_margin", api.GATE_QUALITY, PASS),
        api.GateResult("state_rollback", api.GATE_STABILITY, PASS),
    )


def gates_correctness_failed() -> tuple:
    return (
        api.GateResult("op_reference_shapes", api.GATE_CORRECTNESS,
                       S.Check(S.FAIL, ("adversarial shape mismatched the oracle",)),
                       requires_anchor=True),
    ) + gates_ok()[1:]


def gates_mechanism_failed() -> tuple:
    return gates_ok() + (
        api.GateResult("predicted_counter", api.GATE_MECHANISM,
                       S.Check(S.FAIL, ("the predicted counter did not move",))),
    )


def e_run(*, effect: float, blocks: int = 16, hypothesis: str, margin: float,
          threshold: float = THRESHOLD) -> st.EProcessRun:
    """A real e-process over `blocks` synthetic per-block oriented effects."""
    return st.run_e_process(tuple([effect] * blocks), construction=CONSTRUCTION,
                            hypothesis=hypothesis, margin=margin, threshold=threshold)


def estimate(*, value: float, run: st.EProcessRun, mde: float = 0.02,
             floor: float = 0.01, stratum: str = api.STRATUM_CONFIRMATION,
             metric: str = "decode_tokens_per_s",
             direction: str = "higher_better", lcb=None,
             raw_ref: str = "ak-raw://champion/decode/blocks.jsonl"
             ) -> api.EffectEstimate:
    return api.EffectEstimate(
        metric=metric, metric_direction=direction, value=value,
        e_value=run.e_running_max, threshold=run.threshold, mde=mde, noise_floor=floor,
        paired_blocks=run.blocks, stratum=stratum, raw_samples=(41.1, 43.6),
        raw_samples_ref=raw_ref, lcb_descriptive=lcb)


def verdict(effect, *, tier: str = "T2", gates=None, anchor_=None, voids=(),
            search_grade=None) -> api.Verdict:
    return api.compute_verdict(
        tier=tier, gates=gates if gates is not None else gates_ok(),
        void_scan=api.VoidScan(tuple(voids), api.VOID_REASONS, ()),
        search_grade=(search_grade if search_grade is not None
                      else api.SearchGradeResult(True, (), (), (), ())),
        anchor=anchor_ if anchor_ is not None else anchor(), effect=effect)


def evidence(*, value: float, effect_per_block: float, hypothesis: str, margin: float,
             blocks: int = 16, gates=None, anchor_=None, voids=(), mde: float = 0.02,
             floor: float = 0.01, stratum: str = api.STRATUM_CONFIRMATION,
             metric: str = "decode_tokens_per_s", direction: str = "higher_better",
             lcb=None, raw_ref: str = "ak-raw://champion/decode/blocks.jsonl",
             threshold: float = THRESHOLD, tier: str = "T2") -> R.PhaseEvidence:
    run = e_run(effect=effect_per_block, blocks=blocks, hypothesis=hypothesis,
                margin=margin, threshold=threshold)
    est = estimate(value=value, run=run, mde=mde, floor=floor, stratum=stratum,
                   metric=metric, direction=direction, lcb=lcb, raw_ref=raw_ref)
    return R.PhaseEvidence(
        verdict=verdict(est, tier=tier, gates=gates, anchor_=anchor_, voids=voids),
        e_process=run)


def non_inferior_evidence(**over) -> R.PhaseEvidence:
    kwargs = dict(value=0.06, effect_per_block=0.06,
                  hypothesis=st.HYPOTHESIS_NON_INFERIORITY, margin=NI_MARGIN)
    kwargs.update(over)
    return evidence(**kwargs)


def improving_evidence(**over) -> R.PhaseEvidence:
    kwargs = dict(value=0.06, effect_per_block=0.06,
                  hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0)
    kwargs.update(over)
    return evidence(**kwargs)


def cell(cell_id: str = "cell-decode-a", **over) -> R.T2Cell:
    kwargs = dict(
        cell_id=cell_id, backend="llama_cpu", phase="decode",
        protocol_id=DECODE_PROTOCOL, cell_class="instrument_tokens_per_s",
        role=R.CELL_ROLE_PROTECTED, architecture_class="moe", regime="batch1",
        recipe_class="production_optimal", co_residency="single",
        production_share=0.4, candidate_id=CHAMPION_ID, event_id=f"ake-{cell_id}",
        measured_at=NOW, non_inferiority=None, improvement=None,
        protects_roles=("worker",))
    kwargs.update(over)
    if kwargs["non_inferiority"] is None:
        kwargs["non_inferiority"] = non_inferior_evidence()
    return R.T2Cell(**kwargs)


def champion(**over) -> R.ChampionLineage:
    kwargs = dict(combined_candidate_id=CHAMPION_ID, source_tree="llama.cpp",
                  anchor=anchor(), entered_lineage_at=ENTERED,
                  member_candidate_ids=(MEMBER_A, MEMBER_B))
    kwargs.update(over)
    return R.ChampionLineage(**kwargs)


def objective(**over) -> R.ObjectiveSpec:
    kwargs = dict(backend="llama_cpu", phases=("prefill", "decode"),
                  protocol_by_phase={"prefill": PREFILL_PROTOCOL,
                                     "decode": DECODE_PROTOCOL},
                  improvement_quantifier=R.QUANTIFIER_BACKEND_WIDE)
    kwargs.update(over)
    return R.ObjectiveSpec(**kwargs)


def matrix_spec(**over) -> R.T2MatrixSpec:
    kwargs = dict(backend="llama_cpu",
                  required_coverage=(("moe", "batch1"),),
                  t1_paired_blocks_by_phase={"prefill": 10, "decode": 10},
                  t1_sentinel_ids=frozenset({"sent-t1"}),
                  required_capacity_kinds=(R.CAPACITY_RAM,),
                  effect_scale=st.EFFECT_SCALE_RELATIVE)
    kwargs.update(over)
    return R.T2MatrixSpec(**kwargs)


def mechanisms() -> tuple:
    return (
        R.MechanismConfirmation(member_candidate_id=MEMBER_A,
                                predicted_mechanism="fewer L3 misses per token",
                                confirmed=True, event_id="ake-mech-a"),
        R.MechanismConfirmation(member_candidate_id=MEMBER_B,
                                predicted_mechanism="one fewer kernel launch",
                                confirmed=True, event_id="ake-mech-b"),
    )


def capacity() -> tuple:
    return (R.CapacityDelta(kind=R.CAPACITY_RAM, backend="llama_cpu", delta=0.0,
                            event_id="ake-cap-ram", measured_at=NOW),)


def green_cells() -> tuple:
    """A matrix that satisfies every §9.7 requirement and §1.6's objective."""
    decode = cell("cell-decode-a", improvement=improving_evidence())
    decode_co = cell("cell-decode-co", co_residency="co_resident:big-quarters",
                     event_id="ake-cell-decode-co")
    prefill = cell("cell-prefill-a", phase="prefill", protocol_id=PREFILL_PROTOCOL,
                   non_inferiority=non_inferior_evidence(
                       metric="prefill_tokens_per_s",
                       raw_ref="ak-raw://champion/prefill/blocks.jsonl"))
    sent_t1 = cell("sent-t1", role=R.CELL_ROLE_NON_TARGET, production_share=0.0)
    sent_t2 = cell("sent-t2", role=R.CELL_ROLE_DISPATCHER_BOUNDARY,
                   production_share=0.0)
    return (decode, decode_co, prefill, sent_t1, sent_t2)


def green_signal(**over) -> R.ReadinessSignal:
    kwargs = dict(backend="llama_cpu", campaign_id=CAMPAIGN, champion=champion(),
                  objective=objective(), spec=matrix_spec(), cells=green_cells(),
                  controls_marker=R.CONTROLS_COMPLETE,
                  evaluator_bundle_sha256=sha("evaluator-bundle"),
                  computed_at=LATER, capacity_deltas=capacity(),
                  mechanisms=mechanisms())
    kwargs.update(over)
    return R.compute_readiness(**kwargs)


# ===========================================================================
# 0. Fixture honesty — the green matrix really is green
# ===========================================================================

class GreenFixtureTest(unittest.TestCase):
    """If the baseline fixture is not MET, every negative test below is vacuous."""

    def test_the_green_matrix_meets_the_objective(self):
        signal = green_signal()
        self.assertEqual(signal.standing, R.STANDING_MET, signal.blockers)
        self.assertEqual(signal.blockers, ())
        self.assertEqual(signal.matrix.overall.outcome, S.PASS)

    def test_every_phase_is_non_inferior_and_one_improves(self):
        signal = green_signal()
        for standing in signal.phases:
            self.assertEqual(standing.non_inferior.outcome, S.PASS, standing.phase)
        improved = [s.phase for s in signal.phases if s.improved.outcome == S.PASS]
        self.assertEqual(improved, ["decode"])


# ===========================================================================
# 1. AK-D12 — there is no cross-device composite, and there cannot be one
# ===========================================================================

class NoCrossDeviceCompositeTest(unittest.TestCase):

    def test_compute_readiness_refuses_a_cell_from_another_backend(self):
        foreign = cell("cell-gpu", backend="llama_gpu", protocol_id=DECODE_PROTOCOL)
        with self.assertRaises(R.CrossBackendComposite) as caught:
            green_signal(cells=green_cells() + (foreign,))
        self.assertIn("one backend", str(caught.exception))

    def test_an_objective_for_another_backend_is_refused(self):
        with self.assertRaises(R.CrossBackendComposite):
            green_signal(objective=objective(
                backend="llama_gpu", phases=("decode",),
                protocol_by_phase={"decode": GPU_PROTOCOL}))

    def test_a_matrix_spec_for_another_backend_is_refused(self):
        with self.assertRaises(R.CrossBackendComposite):
            green_signal(spec=matrix_spec(backend="llama_gpu"))

    def test_a_capability_objective_for_another_backend_is_refused(self):
        objective_item = R.CapabilityObjective(
            objective_id="cap-1", backend="llama_gpu",
            utility_model_sha256=sha("utility"), declared_at=ENTERED,
            runnable=PASS, correctness_floor=PASS, quality_floor=PASS,
            resource_budget=PASS, event_id="ake-cap")
        with self.assertRaises(R.CrossBackendComposite):
            green_signal(capability_objectives=(objective_item,))

    def test_composite_readiness_is_a_dead_end_that_explains_itself(self):
        with self.assertRaises(R.CrossBackendComposite) as caught:
            R.composite_readiness(1, 2, weights=(0.5, 0.5))
        text = str(caught.exception)
        self.assertIn("MEASUREMENT.md:83-84", text)
        self.assertIn("gpu-cross-device.md:106-111", text)

    def test_a_report_refuses_two_signals_for_one_backend(self):
        signal = green_signal()
        with self.assertRaises(R.CrossBackendComposite):
            R.compute_readiness_report(campaign_id=CAMPAIGN, computed_at=LATER,
                                       signals=(signal, signal))

    def test_the_cross_backend_view_is_labelled_and_cannot_gate(self):
        report = R.compute_readiness_report(
            campaign_id=CAMPAIGN, computed_at=LATER, signals=(green_signal(),))
        view = R.cross_backend_analysis_view(report)
        self.assertFalse(view.gates)
        self.assertIn("NEVER GATES", view.label)
        with self.assertRaises(R.CrossBackendComposite):
            view.as_gate()

    def test_the_view_keeps_each_rows_own_protocol_and_carries_no_aggregate(self):
        report = R.compute_readiness_report(
            campaign_id=CAMPAIGN, computed_at=LATER, signals=(green_signal(),))
        view = R.cross_backend_analysis_view(report)
        protocols = {row["protocol_id"] for row in view.rows}
        self.assertEqual(protocols, {DECODE_PROTOCOL, PREFILL_PROTOCOL})
        for row in view.rows:
            self.assertTrue(row["not_commensurable_with_other_rows"])
        rendered = view.to_dict()
        self.assertNotIn("total", rendered)
        self.assertNotIn("weighted", rendered)

    def test_a_view_cannot_be_built_claiming_it_gates(self):
        with self.assertRaises(R.CrossBackendComposite):
            R.CrossBackendAnalysisView(
                label=R.CrossBackendAnalysisView.LABEL, rows=(), gates=True)

    def test_a_view_cannot_drop_its_label(self):
        with self.assertRaises(R.CrossBackendComposite):
            R.CrossBackendAnalysisView(label="readiness", rows=())

    def test_a_report_exposes_backends_and_no_scalar(self):
        report = R.compute_readiness_report(
            campaign_id=CAMPAIGN, computed_at=LATER, signals=(green_signal(),))
        self.assertEqual(report.backends, ("llama_cpu",))
        for name in ("score", "total", "composite", "combined", "overall_gain"):
            self.assertFalse(hasattr(report, name), name)

    def test_the_module_cannot_express_a_weighted_average(self):
        """The AST audit is the enforcement; this is the regression barrier."""
        self.assertEqual(R.audit_no_weighting_or_averaging().outcome, S.PASS)

    def test_the_audit_actually_catches_a_weighted_average(self):
        bad = "def f(a, b, wa, wb):\n    return a * wa + b * wb\n"
        result = R.audit_no_weighting_or_averaging(bad)
        self.assertEqual(result.outcome, S.FAIL)
        self.assertTrue(any("Mult" in reason for reason in result.reasons))

    def test_the_audit_catches_a_pooled_reducer(self):
        bad = "def f(xs):\n    return sum(xs)\n"
        self.assertEqual(R.audit_no_weighting_or_averaging(bad).outcome, S.FAIL)

    def test_the_audit_catches_a_second_median(self):
        bad = "from autokernel.evaluator import statistics as st\n" \
              "def f(xs):\n    return st.median(xs)\n"
        self.assertEqual(R.audit_no_weighting_or_averaging(bad).outcome, S.FAIL)

    def test_the_audit_catches_a_numeric_library_import(self):
        bad = "import statistics\n"
        self.assertEqual(R.audit_no_weighting_or_averaging(bad).outcome, S.FAIL)

    def test_unparsable_source_is_could_not_check_not_pass(self):
        self.assertEqual(R.audit_no_weighting_or_averaging("def (").outcome,
                         S.COULD_NOT_CHECK)


# ===========================================================================
# 2. AK-D3 — the signal reports; it does not trigger
# ===========================================================================

class ReportsButDoesNotTriggerTest(unittest.TestCase):

    def test_the_signal_declares_itself_not_a_trigger(self):
        signal = green_signal()
        self.assertFalse(signal.is_trigger)
        self.assertIn("NOT A TRIGGER", signal.signal_class)
        self.assertFalse(signal.to_dict()["is_trigger"])

    def test_freeze_eligibility_is_a_dead_end(self):
        with self.assertRaises(R.TriggerAuthorityError) as caught:
            R.freeze_eligibility(green_signal())
        text = str(caught.exception)
        self.assertIn("four human-only trust boundaries", text)
        self.assertIn("a human executes it", text)

    def test_the_reference_comparison_is_advisory_and_cannot_be_made_binding(self):
        signal = green_signal(reference=R.ReferencePolicy(reference_point_gain=0.25,
                                                          reference_lcb_gain=0.20))
        figure = signal.figure_for("decode")
        self.assertIsNotNone(figure.reference)
        self.assertTrue(figure.reference.advisory)
        self.assertFalse(figure.reference.to_dict()["is_trigger"])

    def test_missing_the_reference_does_not_change_the_standing(self):
        """The +25%/+20% figure is advisory: it never denies a met objective."""
        policy = R.ReferencePolicy(reference_point_gain=0.25, reference_lcb_gain=0.20)
        with_reference = green_signal(reference=policy)
        without = green_signal()
        self.assertEqual(with_reference.standing, without.standing)
        self.assertEqual(with_reference.standing, R.STANDING_MET)
        figure = with_reference.figure_for("decode")
        self.assertEqual(figure.reference.point_at_or_above.outcome, S.FAIL)

    def test_the_lcb_comparison_is_labelled_descriptive(self):
        policy = R.ReferencePolicy(reference_point_gain=0.02, reference_lcb_gain=0.01)
        decode = cell("cell-decode-a",
                      non_inferiority=non_inferior_evidence(lcb=0.03),
                      improvement=improving_evidence(lcb=0.03))
        cells = (decode,) + green_cells()[1:]
        signal = green_signal(cells=cells, reference=policy)
        figure = signal.figure_for("decode")
        rendered = figure.reference.to_dict()
        self.assertEqual(rendered["lcb_at_or_above"]["label"], "descriptive")
        self.assertEqual(rendered["lcb_at_or_above"]["outcome"], S.PASS)

    def test_an_absolute_effect_scale_makes_the_percentage_yardstick_uncheckable(self):
        policy = R.ReferencePolicy(reference_point_gain=0.25, reference_lcb_gain=0.20)
        signal = green_signal(
            spec=matrix_spec(effect_scale=st.EFFECT_SCALE_ABSOLUTE), reference=policy)
        figure = signal.figure_for("decode")
        self.assertEqual(figure.reference.point_at_or_above.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(figure.reference.lcb_at_or_above.outcome, S.COULD_NOT_CHECK)

    def test_a_reference_comparison_cannot_declare_itself_binding(self):
        with self.assertRaises(R.TriggerAuthorityError):
            R.ReferenceComparison(
                effect_scale=st.EFFECT_SCALE_RELATIVE, reference_point_gain=0.25,
                reference_lcb_gain=0.2, observed_point=0.3,
                observed_lcb_descriptive=0.25, point_at_or_above=PASS,
                lcb_at_or_above=PASS, advisory=False)


# ===========================================================================
# 3. Invariant 14 — every number is carried out of one record
# ===========================================================================

class ReadinessIsComputedNotNarratedTest(unittest.TestCase):

    def test_the_figure_is_a_named_cells_own_oriented_estimate(self):
        signal = green_signal()
        figure = signal.figure_for("decode")
        source = None
        for candidate in green_cells():
            if candidate.cell_id == figure.cell_id:
                source = candidate
        self.assertIsNotNone(source)
        self.assertEqual(figure.value, source.oriented_effect())
        self.assertEqual(figure.event_id, source.event_id)
        self.assertEqual(figure.paired_blocks, source.estimate.paired_blocks)

    def test_the_figure_is_the_weakest_protected_cell(self):
        strong = cell("cell-strong", non_inferiority=non_inferior_evidence(value=0.09),
                      improvement=improving_evidence(value=0.09))
        weak = cell("cell-weak", non_inferiority=non_inferior_evidence(value=0.03),
                    event_id="ake-cell-weak")
        cells = (strong, weak) + green_cells()[1:]
        signal = green_signal(cells=cells)
        figure = signal.figure_for("decode")
        self.assertEqual(figure.cell_id, "cell-weak")
        self.assertEqual(figure.value, 0.03)
        self.assertEqual(figure.best_cell_id, "cell-strong")
        self.assertEqual(figure.best_value, 0.09)

    def test_the_figure_is_never_a_value_no_cell_measured(self):
        strong = cell("cell-strong", non_inferiority=non_inferior_evidence(value=0.09),
                      improvement=improving_evidence(value=0.09))
        weak = cell("cell-weak", non_inferiority=non_inferior_evidence(value=0.03),
                    event_id="ake-cell-weak")
        signal = green_signal(cells=(strong, weak) + green_cells()[1:])
        measured = {0.09, 0.03, 0.06}
        self.assertIn(signal.figure_for("decode").value, measured)

    def test_a_lower_better_metric_is_oriented_by_the_statistics_module(self):
        lower = cell(
            "cell-lower",
            non_inferiority=non_inferior_evidence(
                value=-0.06, metric="decode_ms_per_token", direction="lower_better"))
        standing = R.cell_standing(lower)
        self.assertEqual(standing.oriented_effect,
                         st.orient(-0.06, "lower_better"))
        self.assertGreater(standing.oriented_effect, 0)

    def test_the_figure_only_admits_confirmation_stratum_evidence(self):
        with self.assertRaises(R.StratumViolation):
            R.ReadinessFigure(
                backend="llama_cpu", phase="decode", protocol_id=DECODE_PROTOCOL,
                kind=R.ReadinessFigure.KIND_WEAKEST_PROTECTED_CELL,
                cell_id="c", event_id="e", value=0.05, metric="m",
                metric_direction="higher_better", e_value=200.0, threshold=100.0,
                mde=0.02, noise_floor=0.01, paired_blocks=16,
                stratum=api.STRATUM_SELECTION, lcb_descriptive=None,
                best_cell_id="c", best_value=0.05)

    def test_a_figure_has_exactly_one_kind_and_it_is_a_selection(self):
        with self.assertRaises(R.CellInadmissible):
            R.ReadinessFigure(
                backend="llama_cpu", phase="decode", protocol_id=DECODE_PROTOCOL,
                kind="pooled_mean", cell_id="c", event_id="e", value=0.05, metric="m",
                metric_direction="higher_better", e_value=200.0, threshold=100.0,
                mde=0.02, noise_floor=0.01, paired_blocks=16,
                stratum=api.STRATUM_CONFIRMATION, lcb_descriptive=None,
                best_cell_id="c", best_value=0.05)

    def test_the_figure_hands_the_controller_a_per_phase_series_not_one_number(self):
        signal = green_signal()
        fields = signal.figure_for("decode").observation_fields()
        self.assertEqual(set(fields), {"readiness", "source_event_id", "stratum"})
        self.assertEqual(fields["stratum"], api.STRATUM_CONFIRMATION)
        self.assertEqual(len(signal.figures), 2)


# ===========================================================================
# 4. §1.6 — per-backend, per-phase non-inferiority plus improvement
# ===========================================================================

class ObjectiveTest(unittest.TestCase):

    def test_a_phase_that_is_not_non_inferior_denies_the_objective(self):
        regressing = cell(
            "cell-prefill-a", phase="prefill", protocol_id=PREFILL_PROTOCOL,
            non_inferiority=non_inferior_evidence(
                value=-0.05, effect_per_block=-0.05,
                metric="prefill_tokens_per_s",
                raw_ref="ak-raw://champion/prefill/blocks.jsonl"))
        cells = green_cells()[:2] + (regressing,) + green_cells()[3:]
        signal = green_signal(cells=cells)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)
        self.assertIn(R.BLOCK_DETECTABLE_DEGRADATION, signal.blockers)

    def test_no_improvement_anywhere_denies_the_objective(self):
        cells = (cell("cell-decode-a"),) + green_cells()[1:]
        signal = green_signal(cells=cells)
        self.assertEqual(signal.standing, R.STANDING_UNDETERMINED)
        self.assertIn(R.BLOCK_IMPROVEMENT_EVIDENCE_ABSENT, signal.blockers)

    def test_a_failed_improvement_e_process_is_a_fail_not_a_silence(self):
        """One window, two nulls: inside the NI margin, short of improvement."""
        flat = cell("cell-decode-a",
                    non_inferiority=non_inferior_evidence(value=-0.01,
                                                          effect_per_block=-0.01),
                    improvement=improving_evidence(value=-0.01,
                                                   effect_per_block=-0.01))
        self.assertTrue(flat.non_inferiority.crossed)
        self.assertFalse(flat.improvement.crossed)
        signal = green_signal(cells=(flat,) + green_cells()[1:])
        decode = [s for s in signal.phases if s.phase == "decode"][0]
        self.assertEqual(decode.non_inferior.outcome, S.PASS)
        self.assertEqual(decode.improved.outcome, S.FAIL)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)

    def test_both_improvement_quantifiers_are_computed_and_reported(self):
        signal = green_signal()
        self.assertEqual(signal.improvement_backend_wide.outcome, S.PASS)
        self.assertEqual(signal.improvement_per_protected_cell.outcome,
                         S.COULD_NOT_CHECK)

    def test_the_declared_quantifier_is_the_one_that_decides(self):
        strict = green_signal(objective=objective(
            improvement_quantifier=R.QUANTIFIER_PER_PROTECTED_CELL))
        self.assertEqual(strict.standing, R.STANDING_UNDETERMINED)
        lenient = green_signal()
        self.assertEqual(lenient.standing, R.STANDING_MET)

    def test_the_quantifier_has_no_default(self):
        with self.assertRaises(TypeError):
            R.ObjectiveSpec(backend="llama_cpu", phases=("decode",),
                            protocol_by_phase={"decode": DECODE_PROTOCOL})

    def test_an_unknown_quantifier_is_refused_with_the_ambiguity_named(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            objective(improvement_quantifier="whatever")
        self.assertIn("§1.6 does not disambiguate", str(caught.exception))

    def test_a_phase_with_no_protected_cell_is_not_measured(self):
        cells = tuple(c for c in green_cells() if c.phase != "prefill")
        signal = green_signal(cells=cells)
        self.assertIn(R.BLOCK_PHASE_NOT_MEASURED, signal.blockers)
        self.assertEqual(signal.standing, R.STANDING_UNDETERMINED)

    def test_a_sentinel_improvement_cannot_satisfy_the_objective(self):
        """A speed-up on a path nobody runs is not why a lineage ships."""
        sentinel = cell("sent-t1", role=R.CELL_ROLE_NON_TARGET, production_share=0.0,
                        improvement=improving_evidence())
        cells = (cell("cell-decode-a"),) + green_cells()[1:3] + (sentinel,) \
            + green_cells()[4:]
        signal = green_signal(cells=cells)
        decode = [s for s in signal.phases if s.phase == "decode"][0]
        self.assertNotEqual(decode.improved.outcome, S.PASS)


class NonInferiorityDerivationTest(unittest.TestCase):

    def test_a_crossed_non_inferiority_e_process_passes(self):
        standing = R.cell_standing(cell())
        self.assertEqual(standing.non_inferiority.outcome, S.PASS)
        self.assertIn("rejecting H0", standing.non_inferiority.reasons[0])

    def test_a_detectable_degradation_with_no_evidence_fails_and_says_so(self):
        degraded = cell("cell-bad", non_inferiority=non_inferior_evidence(
            value=-0.05, effect_per_block=-0.05))
        standing = R.cell_standing(degraded)
        self.assertEqual(standing.non_inferiority.outcome, S.FAIL)
        reason = standing.non_inferiority.reasons[0]
        self.assertIn("detectable degradation", reason)
        self.assertIn("not a test of inferiority", reason)
        self.assertIn(R.BLOCK_DETECTABLE_DEGRADATION, standing.blockers)

    def test_an_undetectable_difference_is_could_not_check_not_fail(self):
        """12 blocks leave the e-process short of its threshold; |effect| < MDE."""
        tiny = cell("cell-tiny", non_inferiority=non_inferior_evidence(
            value=-0.001, effect_per_block=-0.001, blocks=12))
        self.assertFalse(tiny.non_inferiority.crossed)
        standing = R.cell_standing(tiny)
        self.assertEqual(standing.non_inferiority.outcome, S.COULD_NOT_CHECK)
        self.assertIn(R.BLOCK_NON_INFERIORITY_EVIDENCE_ABSENT, standing.blockers)
        self.assertIn("no detectable difference", standing.non_inferiority.reasons[0])

    def test_a_sub_floor_estimate_with_no_evidence_never_ranks(self):
        sub_floor = cell("cell-floor", non_inferiority=non_inferior_evidence(
            value=0.005, effect_per_block=-0.005, floor=0.01, blocks=12))
        standing = R.cell_standing(sub_floor)
        self.assertEqual(standing.non_inferiority.outcome, S.COULD_NOT_CHECK)
        self.assertIn("MUST NOT be ranked", standing.non_inferiority.reasons[0])

    def test_a_sub_floor_effect_with_crossed_evidence_is_still_non_inferior(self):
        """The floor bars ranking and banking; it does not bar 'not worse'.

        *"An estimate whose magnitude does not exceed phi MUST NOT be ranked,
        banked, or composed"* — none of which is what the non-inferiority half of
        §1.6 asks. A change that is genuinely neutral IS non-inferior, and calling
        that unevaluable would make a neutral-but-correct member unlandable.
        """
        neutral = cell("cell-neutral", non_inferiority=non_inferior_evidence(
            value=0.005, effect_per_block=0.005, floor=0.01, blocks=16))
        self.assertTrue(neutral.non_inferiority.crossed)
        standing = R.cell_standing(neutral)
        self.assertEqual(standing.non_inferiority.outcome, S.PASS)
        self.assertEqual(standing.improvement.outcome, S.COULD_NOT_CHECK)

    def test_fail_and_could_not_check_have_the_same_consequence(self):
        """Both withhold the objective; only the operator-facing reason differs."""
        degraded_cell = cell("cell-decode-a", non_inferiority=non_inferior_evidence(
            value=-0.05, effect_per_block=-0.05))
        undetectable_cell = cell("cell-decode-a", non_inferiority=non_inferior_evidence(
            value=-0.001, effect_per_block=-0.001, blocks=12))
        degraded = green_signal(cells=(degraded_cell,) + green_cells()[1:])
        undetectable = green_signal(cells=(undetectable_cell,) + green_cells()[1:])
        self.assertEqual(R.cell_standing(degraded_cell).non_inferiority.outcome, S.FAIL)
        self.assertEqual(R.cell_standing(undetectable_cell).non_inferiority.outcome,
                         S.COULD_NOT_CHECK)
        self.assertNotEqual(degraded.standing, R.STANDING_MET)
        self.assertNotEqual(undetectable.standing, R.STANDING_MET)

    def test_a_non_inferiority_statement_cannot_be_an_improvement_e_process(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            cell("cell-x", non_inferiority=improving_evidence())
        self.assertIn("does not test it", str(caught.exception))

    def test_an_improvement_statement_cannot_be_a_non_inferiority_e_process(self):
        with self.assertRaises(R.CellInadmissible):
            cell("cell-x", improvement=non_inferior_evidence())

    def test_non_inferiority_holds_even_when_the_point_estimate_is_negative(self):
        """Crossing H0: oriented > -margin is non-inferiority, not a speed-up."""
        run = e_run(effect=0.06, hypothesis=st.HYPOTHESIS_NON_INFERIORITY,
                    margin=NI_MARGIN)
        est = estimate(value=-0.03, run=run)
        item = R.PhaseEvidence(verdict=verdict(est), e_process=run)
        standing = R.cell_standing(cell("cell-x", non_inferiority=item))
        self.assertEqual(standing.non_inferiority.outcome, S.PASS)
        self.assertIn("non-inferior within the declared margin",
                      standing.non_inferiority.reasons[0])


# ===========================================================================
# 5. Correctness precedence — no speed reading behind a failed prior gate
# ===========================================================================

class CorrectnessPrecedenceTest(unittest.TestCase):

    def test_a_failed_correctness_gate_yields_no_speed_standing(self):
        broken = cell("cell-broken", non_inferiority=non_inferior_evidence(
            gates=gates_correctness_failed()))
        standing = R.cell_standing(broken)
        self.assertEqual(standing.non_inferiority.outcome, S.COULD_NOT_CHECK)
        self.assertIn(R.BLOCK_CELL_FAILED_PRIOR_GATE, standing.blockers)
        self.assertIn("no speed standing at all", standing.non_inferiority.reasons[0])

    def test_a_voided_window_is_invalid_and_never_a_candidate_failure(self):
        void = api.VoidFinding(reason=api.VOID_AA_CONTROL_FAILED,
                               protocol_phrase="a failing A/A VOIDS the window",
                               outcome=S.FAIL)
        voided = cell("cell-void",
                      non_inferiority=non_inferior_evidence(voids=(void,)))
        standing = R.cell_standing(voided)
        self.assertIn(R.BLOCK_CELL_INVALID, standing.blockers)
        self.assertEqual(standing.non_inferiority.outcome, S.COULD_NOT_CHECK)

    def test_an_inconclusive_record_is_distinct_from_an_invalid_one(self):
        item = non_inferior_evidence(gates=gates_mechanism_failed())
        self.assertEqual(item.verdict.status, api.STATUS_INCONCLUSIVE)
        standing = R.cell_standing(cell("cell-inconclusive", non_inferiority=item))
        self.assertIn(R.BLOCK_CELL_INCONCLUSIVE, standing.blockers)
        self.assertNotIn(R.BLOCK_CELL_INVALID, standing.blockers)

    def test_a_broken_cell_blocks_the_backend_standing(self):
        broken = cell("cell-decode-a", non_inferiority=non_inferior_evidence(
            gates=gates_correctness_failed()), improvement=improving_evidence())
        signal = green_signal(cells=(broken,) + green_cells()[1:])
        self.assertEqual(signal.standing, R.STANDING_UNDETERMINED)
        self.assertIn(R.BLOCK_CELL_FAILED_PRIOR_GATE, signal.blockers)


# ===========================================================================
# 6. §9.7 — the T2 matrix requirements, each one checked
# ===========================================================================

class ComposedChampionOnlyTest(unittest.TestCase):

    def test_a_member_candidates_cell_is_refused(self):
        member_cell = cell("cell-decode-a", candidate_id=MEMBER_A)
        with self.assertRaises(R.ChampionMismatch) as caught:
            green_signal(cells=(member_cell,) + green_cells()[1:])
        self.assertIn("never by adding local percentages", str(caught.exception))

    def test_a_champion_cannot_be_its_own_member(self):
        with self.assertRaises(R.ChampionMismatch):
            champion(member_candidate_ids=(CHAMPION_ID,))

    def test_the_champion_must_live_in_the_backends_source_tree(self):
        with self.assertRaises(R.ChampionMismatch) as caught:
            green_signal(champion=champion(source_tree="whisper.cpp"))
        self.assertIn("champions are per SOURCE TREE", str(caught.exception))


class CoverageTest(unittest.TestCase):

    def test_an_uncovered_architecture_regime_is_a_coverage_gap(self):
        signal = green_signal(spec=matrix_spec(
            required_coverage=(("moe", "batch1"), ("dense", "long_context"))))
        self.assertEqual(signal.matrix.coverage.outcome, S.FAIL)
        self.assertIn(R.BLOCK_COVERAGE_GAP, signal.blockers)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)

    def test_a_matrix_that_names_no_affected_architecture_cannot_be_checked(self):
        with self.assertRaises(R.MatrixSpecInvalid):
            matrix_spec(required_coverage=())

    def test_only_protected_cells_close_a_coverage_gap(self):
        sentinel_only = cell("sent-only", role=R.CELL_ROLE_NON_TARGET,
                             architecture_class="dense", regime="long_context",
                             production_share=0.0)
        signal = green_signal(
            cells=green_cells() + (sentinel_only,),
            spec=matrix_spec(required_coverage=(("moe", "batch1"),
                                                ("dense", "long_context"))))
        self.assertEqual(signal.matrix.coverage.outcome, S.FAIL)


class RepetitionsTest(unittest.TestCase):

    def test_t2_must_run_stronger_paired_repetitions_than_t1(self):
        signal = green_signal(spec=matrix_spec(
            t1_paired_blocks_by_phase={"prefill": 16, "decode": 16}))
        self.assertEqual(signal.matrix.repetitions.outcome, S.FAIL)
        self.assertIn(R.BLOCK_REPETITIONS_NOT_STRONGER_THAN_T1, signal.blockers)

    def test_equal_repetitions_are_not_stronger(self):
        weak = cell("cell-decode-a",
                    non_inferiority=non_inferior_evidence(blocks=16),
                    improvement=improving_evidence(blocks=16))
        signal = green_signal(cells=(weak,) + green_cells()[1:],
                              spec=matrix_spec(t1_paired_blocks_by_phase={
                                  "prefill": 10, "decode": 16}))
        self.assertEqual(signal.matrix.repetitions.outcome, S.FAIL)

    def test_an_undeclared_t1_count_is_could_not_check_not_pass(self):
        signal = green_signal(spec=matrix_spec(
            t1_paired_blocks_by_phase={"decode": 10}))
        self.assertEqual(signal.matrix.repetitions.outcome, S.COULD_NOT_CHECK)
        self.assertIn(R.BLOCK_REPETITIONS_NOT_STRONGER_THAN_T1, signal.blockers)

    def test_a_spec_with_no_t1_counts_at_all_is_refused(self):
        with self.assertRaises(R.MatrixSpecInvalid):
            matrix_spec(t1_paired_blocks_by_phase={})


class SentinelBreadthTest(unittest.TestCase):

    def test_dropping_a_t1_sentinel_fails(self):
        cells = tuple(c for c in green_cells() if c.cell_id != "sent-t1")
        signal = green_signal(cells=cells)
        self.assertEqual(signal.matrix.sentinels.outcome, S.FAIL)
        self.assertIn(R.BLOCK_SENTINEL_SET_NOT_BROADER, signal.blockers)

    def test_matching_t1_exactly_is_not_broader(self):
        cells = tuple(c for c in green_cells() if c.cell_id != "sent-t2")
        signal = green_signal(cells=cells)
        self.assertEqual(signal.matrix.sentinels.outcome, S.FAIL)
        self.assertIn("strict superset", signal.matrix.sentinels.reasons[0])

    def test_a_regressing_non_target_sentinel_blocks(self):
        bad_sentinel = cell("sent-t1", role=R.CELL_ROLE_NON_TARGET,
                            production_share=0.0,
                            non_inferiority=non_inferior_evidence(
                                value=-0.05, effect_per_block=-0.05))
        cells = green_cells()[:3] + (bad_sentinel,) + green_cells()[4:]
        signal = green_signal(cells=cells)
        self.assertEqual(signal.matrix.non_target.outcome, S.FAIL)
        self.assertIn(R.BLOCK_NON_TARGET_REGRESSION, signal.blockers)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)


class CoResidentCellTest(unittest.TestCase):

    def test_llama_cpu_requires_at_least_one_co_resident_cell(self):
        cells = tuple(c for c in green_cells() if not c.is_co_resident)
        signal = green_signal(cells=cells)
        self.assertEqual(signal.matrix.co_resident.outcome, S.FAIL)
        self.assertIn(R.BLOCK_CO_RESIDENT_CELL_ABSENT, signal.blockers)
        self.assertIn("bandwidth-bound", signal.matrix.co_resident.reasons[0])

    def test_a_campaign_cannot_declare_its_way_out_of_the_requirement(self):
        spec = matrix_spec(extra_co_resident_backends=frozenset())
        self.assertTrue(spec.co_resident_required)
        self.assertIn("llama_cpu", R.CO_RESIDENT_REQUIRED_BACKENDS)

    def test_an_adapter_may_add_a_backend_to_the_requirement(self):
        spec = R.T2MatrixSpec(
            backend="llama_gpu", required_coverage=(("moe", "batch1"),),
            t1_paired_blocks_by_phase={"decode": 10},
            t1_sentinel_ids=frozenset(), required_capacity_kinds=(R.CAPACITY_VRAM,),
            effect_scale=st.EFFECT_SCALE_RELATIVE,
            extra_co_resident_backends=frozenset({"llama_gpu"}))
        self.assertTrue(spec.co_resident_required)

    def test_a_backend_without_the_requirement_passes_the_check(self):
        spec = R.T2MatrixSpec(
            backend="llama_gpu", required_coverage=(("moe", "batch1"),),
            t1_paired_blocks_by_phase={"decode": 10},
            t1_sentinel_ids=frozenset(), required_capacity_kinds=(R.CAPACITY_VRAM,),
            effect_scale=st.EFFECT_SCALE_RELATIVE)
        self.assertFalse(spec.co_resident_required)


class CapacityTest(unittest.TestCase):

    def test_an_undeclared_capacity_requirement_is_could_not_check(self):
        signal = green_signal(spec=matrix_spec(required_capacity_kinds=()))
        self.assertEqual(signal.matrix.capacity.outcome, S.COULD_NOT_CHECK)
        self.assertIn(R.BLOCK_CAPACITY_REQUIREMENT_UNDECLARED, signal.blockers)
        self.assertEqual(signal.standing, R.STANDING_UNDETERMINED)

    def test_a_missing_declared_capacity_delta_blocks(self):
        signal = green_signal(capacity_deltas=())
        self.assertEqual(signal.matrix.capacity.outcome, S.COULD_NOT_CHECK)
        self.assertIn(R.BLOCK_CAPACITY_DELTA_ABSENT, signal.blockers)

    def test_a_capacity_regression_fails(self):
        lost = (R.CapacityDelta(kind=R.CAPACITY_RAM, backend="llama_cpu",
                                delta=-1024.0, event_id="ake-cap", measured_at=NOW),)
        signal = green_signal(capacity_deltas=lost)
        self.assertEqual(signal.matrix.capacity.outcome, S.FAIL)
        self.assertIn(R.BLOCK_CAPACITY_REGRESSION, signal.blockers)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)

    def test_another_backends_capacity_delta_does_not_satisfy_this_one(self):
        foreign = (R.CapacityDelta(kind=R.CAPACITY_RAM, backend="llama_gpu",
                                   delta=0.0, event_id="ake-cap", measured_at=NOW),)
        signal = green_signal(capacity_deltas=foreign)
        self.assertEqual(signal.matrix.capacity.outcome, S.COULD_NOT_CHECK)


class MechanismTest(unittest.TestCase):

    def test_an_unconfirmed_mechanism_blocks_the_lineage(self):
        unconfirmed = (
            R.MechanismConfirmation(
                member_candidate_id=MEMBER_A, predicted_mechanism="fewer L3 misses",
                confirmed=False, event_id="ake-mech-a",
                explanation="the counter moved the other way and nobody knows why"),
            mechanisms()[1],
        )
        signal = green_signal(mechanisms=unconfirmed)
        self.assertEqual(signal.matrix.mechanism.outcome, S.FAIL)
        self.assertIn(R.BLOCK_MECHANISM_UNCONFIRMED, signal.blockers)

    def test_a_member_with_no_confirmation_at_all_is_could_not_check(self):
        signal = green_signal(mechanisms=mechanisms()[:1])
        self.assertEqual(signal.matrix.mechanism.outcome, S.COULD_NOT_CHECK)
        self.assertIn(R.BLOCK_MECHANISM_UNCONFIRMED, signal.blockers)

    def test_an_unconfirmed_mechanism_needs_a_recorded_explanation(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            R.MechanismConfirmation(member_candidate_id=MEMBER_A,
                                    predicted_mechanism="magic", confirmed=False,
                                    event_id="ake-mech")
        self.assertIn("keep measuring, not to land", str(caught.exception))

    def test_a_confirmation_for_a_non_member_is_flagged(self):
        stray = mechanisms() + (
            R.MechanismConfirmation(member_candidate_id="akc-stranger",
                                    predicted_mechanism="x", confirmed=True,
                                    event_id="ake-mech-x"),)
        signal = green_signal(mechanisms=stray)
        self.assertEqual(signal.matrix.mechanism.outcome, S.COULD_NOT_CHECK)


class AnchorTest(unittest.TestCase):

    def test_a_cell_measured_against_another_anchor_is_anchor_moved(self):
        drifted = cell("cell-decode-a", improvement=None,
                       non_inferiority=non_inferior_evidence(anchor_=anchor("v7")))
        signal = green_signal(cells=(drifted,) + green_cells()[1:])
        self.assertEqual(signal.matrix.anchor_agreement.outcome, S.FAIL)
        self.assertIn(R.BLOCK_ANCHOR_MOVED, signal.blockers)
        self.assertIn("superseded, not reinterpreted",
                      signal.matrix.anchor_agreement.reasons[0])

    def test_an_anchorless_cell_is_invalid_and_never_coherent(self):
        run = e_run(effect=0.06, hypothesis=st.HYPOTHESIS_NON_INFERIORITY,
                    margin=NI_MARGIN)
        est = estimate(value=0.06, run=run)
        anchorless = R.PhaseEvidence(
            verdict=api.compute_verdict(
                tier="T2", gates=gates_ok(),
                void_scan=api.VoidScan((), api.VOID_REASONS, ()),
                search_grade=api.SearchGradeResult(True, (), (), (), ()),
                anchor=None, effect=est),
            e_process=run)
        standing = R.cell_standing(cell("cell-x", non_inferiority=anchorless))
        self.assertIn(R.BLOCK_CELL_INVALID, standing.blockers)
        self.assertEqual(standing.non_inferiority.outcome, S.COULD_NOT_CHECK)


class StrataAndLineageTest(unittest.TestCase):

    def test_selection_stratum_evidence_is_refused(self):
        with self.assertRaises(R.StratumViolation) as caught:
            cell("cell-x", non_inferiority=non_inferior_evidence(
                stratum=api.STRATUM_SELECTION))
        self.assertIn("structurally unfit", str(caught.exception))

    def test_evidence_gathered_before_lineage_entry_blocks(self):
        early = cell("cell-decode-a", measured_at=BEFORE,
                     improvement=improving_evidence())
        signal = green_signal(cells=(early,) + green_cells()[1:])
        self.assertEqual(signal.matrix.lineage_ordering.outcome, S.FAIL)
        self.assertIn(R.BLOCK_CONFIRMATION_EVIDENCE_PREDATES_LINEAGE, signal.blockers)

    def test_a_naive_timestamp_cannot_be_ordered_and_is_refused(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            cell("cell-x", measured_at="2026-08-03T12:00:00")
        self.assertIn("cannot be ordered", str(caught.exception))

    def test_the_two_statements_must_come_from_the_same_window(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            cell("cell-x", improvement=improving_evidence(
                raw_ref="ak-raw://another/window.jsonl"))
        self.assertIn("do not co-occur", str(caught.exception))

    def test_differing_block_counts_are_two_windows(self):
        with self.assertRaises(R.CellInadmissible):
            cell("cell-x", improvement=improving_evidence(blocks=20))


class ProtocolBoundaryTest(unittest.TestCase):

    def test_a_cell_citing_another_phases_protocol_is_refused(self):
        crossed = cell("cell-decode-a", protocol_id=PREFILL_PROTOCOL,
                       improvement=improving_evidence())
        with self.assertRaises(R.ProtocolBoundaryCrossed) as caught:
            green_signal(cells=(crossed,) + green_cells()[1:])
        self.assertIn("its own protocol", str(caught.exception))

    def test_a_phase_the_objective_does_not_declare_is_refused(self):
        objective_decode_only = objective(
            phases=("decode",), protocol_by_phase={"decode": DECODE_PROTOCOL})
        with self.assertRaises(R.ProtocolBoundaryCrossed):
            green_signal(objective=objective_decode_only)

    def test_two_phases_may_share_one_protocol_when_the_backend_does(self):
        gpu = R.ObjectiveSpec(
            backend="llama_gpu", phases=("prefill", "decode"),
            protocol_by_phase={"prefill": GPU_PROTOCOL, "decode": GPU_PROTOCOL},
            improvement_quantifier=R.QUANTIFIER_BACKEND_WIDE)
        self.assertEqual(gpu.protocol_for("prefill"), GPU_PROTOCOL)
        self.assertEqual(gpu.protocol_for("decode"), GPU_PROTOCOL)

    def test_a_protocol_for_an_undeclared_phase_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            objective(phases=("decode",),
                      protocol_by_phase={"decode": DECODE_PROTOCOL,
                                         "prefill": PREFILL_PROTOCOL})

    def test_a_phase_with_no_protocol_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            objective(phases=("decode", "prefill"),
                      protocol_by_phase={"decode": DECODE_PROTOCOL})


# ===========================================================================
# 7. Invariant 15 — production-optimal only, and weights never weight
# ===========================================================================

class ProductionRecipeTest(unittest.TestCase):

    def test_an_off_recipe_cell_is_not_admissible_to_readiness(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            cell("cell-x", recipe_class="baseline")
        self.assertIn("never justifies or vetoes a release", str(caught.exception))

    def test_the_objective_pins_the_production_optimal_recipe_class(self):
        with self.assertRaises(R.CellInadmissible):
            objective(recipe_class="baseline")

    def test_production_share_cannot_change_any_verdict(self):
        """§1.6 withdrew the production-weighted composite; the share only orders."""
        flat = tuple(
            R.T2Cell(**{**{f.name: getattr(c, f.name)
                           for f in c.__dataclass_fields__.values()},
                        "production_share": 0.0})
            for c in green_cells())
        heavy = tuple(
            R.T2Cell(**{**{f.name: getattr(c, f.name)
                           for f in c.__dataclass_fields__.values()},
                        "production_share": 1.0})
            for c in green_cells())
        first = green_signal(cells=flat)
        second = green_signal(cells=heavy)
        self.assertEqual(first.standing, second.standing)
        self.assertEqual(first.figure_for("decode").value,
                         second.figure_for("decode").value)
        self.assertEqual(first.blockers, second.blockers)


# ===========================================================================
# 8. The phase trade is the operator's decision, at freeze time
# ===========================================================================

def regressing_prefill_cells() -> tuple:
    regressing = cell(
        "cell-prefill-a", phase="prefill", protocol_id=PREFILL_PROTOCOL,
        non_inferiority=non_inferior_evidence(
            value=-0.03, effect_per_block=-0.03, metric="prefill_tokens_per_s",
            raw_ref="ak-raw://champion/prefill/blocks.jsonl"))
    return green_cells()[:2] + (regressing,) + green_cells()[3:]


class PhaseTradeTest(unittest.TestCase):

    def test_an_undeclared_trade_is_a_regression_not_an_exception(self):
        signal = green_signal(cells=regressing_prefill_cells())
        self.assertEqual(signal.phase_trade.status,
                         R.PhaseTradeAssessment.STATUS_NOT_PREDECLARED)
        self.assertFalse(signal.phase_trade.operator_decision_required)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)

    def test_a_pre_declared_trade_inside_its_band_still_does_not_meet_the_objective(self):
        exception = R.PhaseTradeException(
            regressing_phase="prefill", band=(-0.05, -0.01), expected_gain=0.05,
            roles=("worker",), declared_at=BEFORE)
        signal = green_signal(cells=regressing_prefill_cells(),
                              objective=objective(phase_trade_exception=exception))
        self.assertEqual(signal.phase_trade.status,
                         R.PhaseTradeAssessment.STATUS_WITHIN_BAND)
        self.assertTrue(signal.phase_trade.operator_decision_required)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)
        self.assertIn(R.BLOCK_PHASE_TRADE_DECISION_REQUIRED, signal.blockers)

    def test_the_assessment_says_out_loud_that_it_does_not_decide(self):
        exception = R.PhaseTradeException(
            regressing_phase="prefill", band=(-0.05, -0.01), expected_gain=0.05,
            roles=("worker",), declared_at=BEFORE)
        signal = green_signal(cells=regressing_prefill_cells(),
                              objective=objective(phase_trade_exception=exception))
        joined = " ".join(signal.phase_trade.reasons)
        self.assertIn("operator decision at freeze time", joined)
        self.assertIn("not a freeze trigger", joined)

    def test_a_regression_outside_the_declared_band_is_outside_it(self):
        exception = R.PhaseTradeException(
            regressing_phase="prefill", band=(-0.01, -0.005), expected_gain=0.05,
            roles=("worker",), declared_at=BEFORE)
        signal = green_signal(cells=regressing_prefill_cells(),
                              objective=objective(phase_trade_exception=exception))
        self.assertEqual(signal.phase_trade.status,
                         R.PhaseTradeAssessment.STATUS_OUTSIDE_BAND)

    def test_an_exception_for_another_phase_does_not_stretch(self):
        exception = R.PhaseTradeException(
            regressing_phase="decode", band=(-0.05, -0.01), expected_gain=0.05,
            roles=("worker",), declared_at=BEFORE)
        signal = green_signal(cells=regressing_prefill_cells(),
                              objective=objective(phase_trade_exception=exception))
        self.assertEqual(signal.phase_trade.status,
                         R.PhaseTradeAssessment.STATUS_OUTSIDE_BAND)

    def test_a_trade_with_no_expected_gain_is_refused_at_declaration(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            R.PhaseTradeException(regressing_phase="prefill", band=(-0.05, -0.01),
                                  expected_gain=0.0, roles=("worker",),
                                  declared_at=BEFORE)
        self.assertIn("a regression with paperwork", str(caught.exception))

    def test_a_trade_band_that_is_not_a_regression_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            R.PhaseTradeException(regressing_phase="prefill", band=(-0.01, 0.05),
                                  expected_gain=0.05, roles=("worker",),
                                  declared_at=BEFORE)

    def test_an_unscoped_exception_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            R.PhaseTradeException(regressing_phase="prefill", band=(-0.05, -0.01),
                                  expected_gain=0.05, roles=(), declared_at=BEFORE)

    def test_an_exception_for_a_phase_the_objective_does_not_declare_is_refused(self):
        exception = R.PhaseTradeException(
            regressing_phase="prefill", band=(-0.05, -0.01), expected_gain=0.05,
            roles=("worker",), declared_at=BEFORE)
        with self.assertRaises(R.CellInadmissible):
            objective(phases=("decode",),
                      protocol_by_phase={"decode": DECODE_PROTOCOL},
                      phase_trade_exception=exception)

    def test_no_regression_means_the_assessment_is_not_applicable(self):
        signal = green_signal()
        self.assertEqual(signal.phase_trade.status,
                         R.PhaseTradeAssessment.STATUS_NOT_APPLICABLE)
        self.assertFalse(signal.phase_trade.operator_decision_required)


# ===========================================================================
# 9. §9.8 — capability objectives enter only through a fixed utility model
# ===========================================================================

class CapabilityObjectiveTest(unittest.TestCase):

    def _objective(self, **over):
        kwargs = dict(objective_id="cap-122b-iq2", backend="llama_cpu",
                      utility_model_sha256=sha("utility-model"), declared_at=BEFORE,
                      runnable=PASS, correctness_floor=PASS, quality_floor=PASS,
                      resource_budget=PASS, event_id="ake-cap-1")
        kwargs.update(over)
        return R.CapabilityObjective(**kwargs)

    def test_a_matching_campaign_start_digest_admits_the_capability(self):
        signal = green_signal(
            capability_objectives=(self._objective(),),
            campaign_start_utility_model_sha256=sha("utility-model"))
        self.assertEqual(signal.capabilities[0].admitted.outcome, S.PASS)
        self.assertEqual(signal.standing, R.STANDING_MET)

    def test_a_drifted_utility_model_blocks_the_capability(self):
        signal = green_signal(
            capability_objectives=(self._objective(),),
            campaign_start_utility_model_sha256=sha("something-else"))
        self.assertEqual(signal.capabilities[0].admitted.outcome, S.FAIL)
        self.assertIn(R.BLOCK_CAPABILITY_UTILITY_MODEL_DRIFTED, signal.blockers)

    def test_an_unsupplied_campaign_start_digest_is_could_not_check(self):
        signal = green_signal(capability_objectives=(self._objective(),))
        capability = signal.capabilities[0]
        self.assertEqual(capability.utility_model_fixed_at_campaign_start.outcome,
                         S.COULD_NOT_CHECK)
        self.assertIn(R.BLOCK_CAPABILITY_UTILITY_MODEL_DRIFTED, signal.blockers)

    def test_a_failed_floor_denies_admission(self):
        failing = self._objective(
            quality_floor=S.Check(S.FAIL, ("PPL margin exceeded",)))
        signal = green_signal(
            capability_objectives=(failing,),
            campaign_start_utility_model_sha256=sha("utility-model"))
        self.assertEqual(signal.capabilities[0].admitted.outcome, S.FAIL)

    def test_a_placeholder_utility_digest_is_refused(self):
        with self.assertRaises(R.CapabilityObjectiveInvalid):
            self._objective(utility_model_sha256="0" * 64)


# ===========================================================================
# 10. §9.7 — the T2 trigger authorizes a measurement window, nothing else
# ===========================================================================

class TriggerTest(unittest.TestCase):

    def _call(self, **over):
        kwargs = dict(composed_champion_passed_t0_t1=PASS,
                      winners_accumulated_interaction_dominant=S.Check(
                          S.FAIL, ("two winners banked, no interaction risk",)),
                      readiness_could_change_materially=S.Check(
                          S.FAIL, ("the champion has not moved",)),
                      capability_objective_runnable=S.Check(
                          S.FAIL, ("nothing newly runnable",)))
        kwargs.update(over)
        return R.evaluate_t2_trigger(**kwargs)

    def test_any_one_condition_fires_the_trigger(self):
        decision = self._call(readiness_could_change_materially=S.Check(
            S.PASS, ("four compatible winners since the last T2",)))
        self.assertEqual(decision.outcome, R.TRIGGER_RUN_T2)
        self.assertEqual(decision.satisfied, (R.TRIGGER_READINESS_COULD_CHANGE,))

    def test_no_condition_holds(self):
        self.assertEqual(self._call().outcome, R.TRIGGER_HOLD)

    def test_an_ungreen_composed_champion_holds_even_when_a_condition_fires(self):
        decision = self._call(
            composed_champion_passed_t0_t1=S.Check(S.FAIL, ("T0 build failed",)),
            readiness_could_change_materially=PASS)
        self.assertEqual(decision.outcome, R.TRIGGER_HOLD)
        self.assertIn("FULL composed champion", decision.reasons[0])

    def test_an_unknown_precondition_is_could_not_evaluate_not_hold(self):
        decision = self._call(composed_champion_passed_t0_t1=S.Check(
            S.COULD_NOT_CHECK, ("the T1 record for the composition is missing",)))
        self.assertEqual(decision.outcome, R.TRIGGER_COULD_NOT_EVALUATE)

    def test_all_conditions_unknown_is_could_not_evaluate(self):
        unknown = S.Check(S.COULD_NOT_CHECK, ("no record",))
        decision = self._call(winners_accumulated_interaction_dominant=unknown,
                              readiness_could_change_materially=unknown,
                              capability_objective_runnable=unknown)
        self.assertEqual(decision.outcome, R.TRIGGER_COULD_NOT_EVALUATE)

    def test_the_trigger_says_what_it_authorizes(self):
        decision = self._call(capability_objective_runnable=PASS)
        rendered = decision.to_dict()
        self.assertEqual(rendered["authorizes"], "one T2 measurement window")
        self.assertFalse(rendered["is_trigger"])

    def test_an_undeclared_condition_name_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            R.TriggerDecision(outcome=R.TRIGGER_RUN_T2, satisfied=("because_i_said",),
                              reasons=(), precondition=PASS)


# ===========================================================================
# 11. Controls marker, rendering, and record hygiene
# ===========================================================================

class MarkerAndRenderingTest(unittest.TestCase):

    def test_the_four_control_marker_rides_on_the_signal(self):
        signal = green_signal(controls_marker=R.CONTROLS_REPLAY_UNAVAILABLE)
        self.assertIn("HISTORICAL_REPLAY_UNAVAILABLE", signal.controls_marker)
        self.assertIn("HISTORICAL_REPLAY_UNAVAILABLE",
                      R.render_readiness_line(signal, "decode"))

    def test_a_four_control_campaign_is_not_blocked_by_the_controller(self):
        """Whether to proceed on four controls is the operator's call, not ours."""
        signal = green_signal(controls_marker=R.CONTROLS_REPLAY_UNAVAILABLE)
        self.assertEqual(signal.standing, R.STANDING_MET)

    def test_an_undeclared_controls_marker_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            green_signal(controls_marker="most of them")

    def test_the_rendered_line_labels_the_signal_and_names_its_evidence(self):
        signal = green_signal()
        line = R.render_readiness_line(signal, "decode")
        self.assertIn(R.SIGNAL_CLASS, line)
        self.assertIn("NOT A CLAIM", line)
        self.assertIn("NOT A TRIGGER", line)
        self.assertIn(f"tier={R.TIER}", line)
        self.assertIn(f"protocol={DECODE_PROTOCOL}", line)
        self.assertIn(signal.anchor.short(), line)
        self.assertIn("stratum=confirmation", line)
        self.assertIn(st.STATISTICS_MODULE_ID, line)
        for token in ("blocks=", "e=", "thr=", "MDE=", "floor="):
            self.assertIn(token, line)

    def test_rendering_an_undeclared_phase_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            R.render_readiness_line(green_signal(), "audio")

    def test_the_projection_carries_no_authority_flavoured_key(self):
        signal = green_signal()
        self.assertEqual(S.find_authority_flavoured_keys(signal.to_dict()), [])
        report = R.compute_readiness_report(campaign_id=CAMPAIGN, computed_at=LATER,
                                            signals=(signal,))
        self.assertEqual(S.find_authority_flavoured_keys(report.to_dict()), [])

    def test_the_projection_is_canonically_serialisable(self):
        signal = green_signal()
        self.assertTrue(S.canonical_json(signal.to_dict()))

    def test_the_signal_names_the_reducer_that_produced_it(self):
        signal = green_signal()
        self.assertEqual(signal.reducer_id, R.MODULE_ID)
        self.assertEqual(signal.statistics_module_id, st.STATISTICS_MODULE_ID)


# ===========================================================================
# 12. Tier ownership and module hygiene
# ===========================================================================

class TierAndHygieneTest(unittest.TestCase):

    def test_the_estimator_reads_t2_records_only(self):
        run = e_run(effect=0.06, hypothesis=st.HYPOTHESIS_NON_INFERIORITY,
                    margin=NI_MARGIN)
        est = estimate(value=0.06, run=run)
        with self.assertRaises(R.CellInadmissible) as caught:
            R.PhaseEvidence(verdict=verdict(est, tier="T1"), e_process=run)
        self.assertIn("T3 is a release gate", str(caught.exception))

    def test_t3_is_not_reachable_from_this_module(self):
        with self.assertRaises(api.TierNotOwned):
            api.admit_tier("T3")

    def test_the_module_writes_nothing_and_signals_nothing(self):
        self.assertEqual(R.audit_no_write_or_process_paths().outcome, S.PASS)

    def test_the_write_audit_actually_catches_a_write(self):
        bad = "import os\ndef f():\n    os.remove('/tmp/x')\n"
        self.assertEqual(R.audit_no_write_or_process_paths(bad).outcome, S.FAIL)

    def test_an_e_process_and_an_estimate_from_different_runs_are_refused(self):
        run = e_run(effect=0.06, hypothesis=st.HYPOTHESIS_NON_INFERIORITY,
                    margin=NI_MARGIN)
        other = e_run(effect=0.06, blocks=20,
                      hypothesis=st.HYPOTHESIS_NON_INFERIORITY, margin=NI_MARGIN)
        est = estimate(value=0.06, run=run)
        with self.assertRaises(R.CellInadmissible) as caught:
            R.PhaseEvidence(verdict=verdict(est), e_process=other)
        self.assertIn("same run", str(caught.exception))

    def test_an_estimate_without_its_e_process_is_refused(self):
        run = e_run(effect=0.06, hypothesis=st.HYPOTHESIS_NON_INFERIORITY,
                    margin=NI_MARGIN)
        est = estimate(value=0.06, run=run)
        with self.assertRaises(R.CellInadmissible) as caught:
            R.PhaseEvidence(verdict=verdict(est))
        self.assertIn("the hypothesis", str(caught.exception))

    def test_evidence_can_be_built_from_the_reducers_own_output(self):
        self.assertTrue(callable(R.PhaseEvidence.from_reduction))
        with self.assertRaises(R.CellInadmissible):
            R.PhaseEvidence.from_reduction("not a reduction", None)

    def test_every_declared_blocker_is_a_member_of_the_vocabulary(self):
        with self.assertRaises(R.CellInadmissible):
            R.CellStanding(cell_id="c", backend="llama_cpu", phase="decode",
                           role=R.CELL_ROLE_PROTECTED, event_id="e",
                           non_inferiority=PASS, improvement=PASS,
                           blockers=("MADE_UP",), oriented_effect=0.0)

    def test_the_module_declares_the_one_objective_rule_the_schema_knows(self):
        self.assertIn(R.OBJECTIVE_RULE, S.OBJECTIVE_RULES)

    def test_the_capacity_kinds_are_the_three_the_design_names(self):
        self.assertEqual(set(R.CAPACITY_KINDS),
                         {"vram_bytes_free", "ram_bytes_free", "context_tokens"})


# ===========================================================================
# 13. Red-team regressions — each one is a defect that WAS reachable
#
# Every test in this section failed against the module as first written. They
# are grouped together and named for the hole rather than the function, because
# the next person to touch these code paths needs to know what reopening one
# costs, not which line it lives on.
# ===========================================================================

class FigureObeysCorrectnessPrecedenceTest(unittest.TestCase):
    """A rank-inadmissible cell may not BE the readiness number.

    `cell_standing()` withheld the speed *Check* for a failed prior gate, but
    `_phase_figure` selected over `oriented_effect()` without consulting the
    verdict at all. A cell whose correctness gate FAILED therefore supplied
    `figure.value`, `figure.best_value`, the `+25%` advisory comparison, the line
    the operator reads, and `observation_fields()` — which is the series
    `controller.guards` runs the plateau/stop rule over. *"A candidate failing any
    of them receives no speed rank at all — not a penalised one"*, and being
    selected as the weakest or the best IS a rank.
    """

    def test_a_correctness_failed_cell_does_not_supply_the_figure(self):
        broken = cell("cell-decode-broken", event_id="ake-broken",
                      non_inferiority=non_inferior_evidence(
                          value=0.99, effect_per_block=0.99,
                          gates=gates_correctness_failed()))
        self.assertEqual(broken.non_inferiority.verdict.status, api.STATUS_FAIL)
        self.assertEqual(broken.oriented_effect(), 0.99)
        standing = R.phase_standing(backend="llama_cpu", phase="decode",
                                    objective=objective(), cells=(broken,))
        self.assertIsNone(standing.figure)

    def test_an_invalid_cell_does_not_supply_the_figure(self):
        void = api.VoidFinding(reason=api.VOID_AA_CONTROL_FAILED,
                               protocol_phrase="a failing A/A VOIDS the window",
                               outcome=S.FAIL)
        voided = cell("cell-void", event_id="ake-void",
                      non_inferiority=non_inferior_evidence(
                          value=0.80, effect_per_block=0.80, voids=(void,)))
        self.assertEqual(voided.non_inferiority.verdict.status, api.STATUS_INVALID)
        standing = R.phase_standing(backend="llama_cpu", phase="decode",
                                    objective=objective(), cells=(voided,))
        self.assertIsNone(standing.figure)

    def test_a_failed_cell_is_not_the_best_cell_and_not_the_advisory_comparison(self):
        """The whole leak, end to end: figure, best, reference and rendered line."""
        broken = cell("cell-decode-broken", event_id="ake-broken",
                      non_inferiority=non_inferior_evidence(
                          value=0.99, effect_per_block=0.99,
                          gates=gates_correctness_failed()))
        signal = green_signal(cells=(broken,) + green_cells()[1:],
                              reference=R.ReferencePolicy(reference_point_gain=0.25,
                                                          reference_lcb_gain=0.20))
        figure = signal.figure_for("decode")
        self.assertIsNotNone(figure)
        self.assertNotEqual(figure.cell_id, "cell-decode-broken")
        self.assertNotEqual(figure.best_cell_id, "cell-decode-broken")
        self.assertNotEqual(figure.value, 0.99)
        self.assertNotEqual(figure.best_value, 0.99)
        # The advisory +25% comparison read PASS off the failed cell's 0.99.
        self.assertEqual(figure.reference.point_at_or_above.outcome, S.FAIL)
        self.assertNotIn("ake-broken", R.render_readiness_line(signal, "decode"))
        # And the number handed to the controller's plateau series is not its 0.99.
        self.assertNotEqual(figure.observation_fields()["readiness"], 0.99)
        self.assertIn(R.BLOCK_CELL_FAILED_PRIOR_GATE, signal.blockers)

    def test_a_phase_of_only_inadmissible_cells_has_no_figure_rather_than_a_wrong_one(self):
        broken = cell("cell-decode-broken", non_inferiority=non_inferior_evidence(
            value=0.99, effect_per_block=0.99, gates=gates_correctness_failed()))
        cells = (broken,) + tuple(c for c in green_cells()
                                  if c.phase != "decode" or c.role != R.CELL_ROLE_PROTECTED)
        signal = green_signal(cells=cells)
        self.assertIsNone(signal.figure_for("decode"))
        self.assertIn("no protected-cell figure",
                      R.render_readiness_line(signal, "decode"))
        self.assertNotEqual(signal.standing, R.STANDING_MET)

    def test_a_detectable_degradation_still_supplies_its_figure(self):
        """The fix must not delete the signal it exists to report.

        A cell that PASSED every prior gate and then measured a degradation is
        rank-inadmissible for a different reason — the RESOLUTION, not correctness
        — and its number is exactly what the operator needs. Excluding it would
        make a regression invisible, which is the failure mode in the other
        direction.
        """
        signal = green_signal(cells=regressing_prefill_cells())
        figure = signal.figure_for("prefill")
        self.assertIsNotNone(figure)
        self.assertLess(figure.value, 0.0)


class CapacityDeltasAreNotDeduplicatedTest(unittest.TestCase):
    """A second record for one axis must not overwrite the first.

    `{delta.kind: delta}` kept the LAST record per kind, so a measured regression
    followed by a clean re-measurement of the same axis read PASS with no blocker
    — and the same two records in the other order read FAIL. Order-dependence in
    a deterministic reducer is a defect on its own; this one's preferred direction
    was to hide the regression.
    """

    def _capacity_pair(self):
        regressed = R.CapacityDelta(kind=R.CAPACITY_RAM, backend="llama_cpu",
                                    delta=-1.0, event_id="ake-cap-regressed",
                                    measured_at=NOW)
        clean = R.CapacityDelta(kind=R.CAPACITY_RAM, backend="llama_cpu", delta=0.0,
                                event_id="ake-cap-clean", measured_at=LATER)
        return regressed, clean

    def test_a_regression_is_not_overwritten_by_a_later_clean_record(self):
        regressed, clean = self._capacity_pair()
        signal = green_signal(capacity_deltas=(regressed, clean))
        self.assertEqual(signal.matrix.capacity.outcome, S.FAIL)
        self.assertIn(R.BLOCK_CAPACITY_REGRESSION, signal.blockers)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)

    def test_the_capacity_verdict_does_not_depend_on_record_order(self):
        regressed, clean = self._capacity_pair()
        forward = green_signal(capacity_deltas=(regressed, clean))
        reverse = green_signal(capacity_deltas=(clean, regressed))
        self.assertEqual(forward.matrix.capacity.outcome, reverse.matrix.capacity.outcome)
        self.assertEqual(forward.standing, reverse.standing)

    def test_every_measured_axis_record_is_reported_not_just_the_last(self):
        clean_a = R.CapacityDelta(kind=R.CAPACITY_RAM, backend="llama_cpu", delta=0.0,
                                  event_id="ake-cap-a", measured_at=NOW)
        clean_b = R.CapacityDelta(kind=R.CAPACITY_RAM, backend="llama_cpu", delta=1.0,
                                  event_id="ake-cap-b", measured_at=LATER)
        signal = green_signal(capacity_deltas=(clean_a, clean_b))
        self.assertEqual(signal.matrix.capacity.outcome, S.PASS)
        reasons = " ".join(signal.matrix.capacity.reasons)
        self.assertIn("ake-cap-a", reasons)
        self.assertIn("ake-cap-b", reasons)


class MechanismConfirmationsAreNotDeduplicatedTest(unittest.TestCase):
    """An unconfirmed mechanism must not be overwritten by a later confirmation.

    `{conf.member_candidate_id: conf}` kept the LAST record per member, so a
    `confirmed=False` finding — the exact thing `P-AK-SEARCH-1-A1` clause 1 blocks
    on — disappeared behind a subsequent `confirmed=True` record for the same
    member, and the standing read `objective_met` with an empty blocker list.
    """

    def _contradictory(self):
        unconfirmed = R.MechanismConfirmation(
            member_candidate_id=MEMBER_A, predicted_mechanism="fewer L3 misses",
            confirmed=False, event_id="ake-mech-a-negative",
            explanation="the predicted counter did not move on the composition")
        confirmed = R.MechanismConfirmation(
            member_candidate_id=MEMBER_A, predicted_mechanism="fewer L3 misses",
            confirmed=True, event_id="ake-mech-a-positive")
        return unconfirmed, confirmed, mechanisms()[1]

    def test_an_unconfirmed_member_is_not_overwritten_by_a_later_confirmation(self):
        unconfirmed, confirmed, member_b = self._contradictory()
        signal = green_signal(mechanisms=(unconfirmed, confirmed, member_b))
        self.assertEqual(signal.matrix.mechanism.outcome, S.FAIL)
        self.assertIn(R.BLOCK_MECHANISM_UNCONFIRMED, signal.blockers)
        self.assertEqual(signal.standing, R.STANDING_NOT_MET)

    def test_the_mechanism_verdict_does_not_depend_on_record_order(self):
        unconfirmed, confirmed, member_b = self._contradictory()
        forward = green_signal(mechanisms=(unconfirmed, confirmed, member_b))
        reverse = green_signal(mechanisms=(confirmed, unconfirmed, member_b))
        self.assertEqual(forward.matrix.mechanism.outcome,
                         reverse.matrix.mechanism.outcome)
        self.assertEqual(forward.standing, reverse.standing)


class EvaluatorBundleIsADigestTest(unittest.TestCase):
    """`eval=` in the operator's line must be a digest, not a label.

    Precondition 5 pins the evaluator bundle by hash. The field was validated as
    a non-empty STRING, so `'not-a-hash'` and the all-zero placeholder both rode
    into the rendered line as `eval=<12 chars>`, where they are indistinguishable
    from a pinned evaluator. `_sha256` already refuses placeholders elsewhere in
    this module for exactly that reason.
    """

    def test_a_placeholder_digest_is_refused(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            green_signal(evaluator_bundle_sha256="0" * 64)
        self.assertIn("placeholder digest", str(caught.exception))

    def test_a_free_text_label_is_refused(self):
        with self.assertRaises(R.CellInadmissible) as caught:
            green_signal(evaluator_bundle_sha256="pinned-evaluator-bundle")
        self.assertIn("64-hex lowercase digest", str(caught.exception))

    def test_a_real_digest_is_accepted_and_abbreviated_in_the_line(self):
        signal = green_signal()
        self.assertIn(f"eval={sha('evaluator-bundle')[:12]}",
                      R.render_readiness_line(signal, "decode"))


class ObjectiveCoversEveryDeclaredPhaseTest(unittest.TestCase):
    """§1.6's conjunction cannot be satisfied by deleting one of its conjuncts.

    *"Both prefill and decode throughput must be non-inferior … and at least one
    must improve."* An `ObjectiveSpec` naming only `decode` reached
    `objective_met` on decode evidence alone, with an EMPTY blocker list and
    nothing anywhere in the signal recording that prefill was never asked about.
    `schemas.PHASES_BY_BACKEND` already knew both phases; the module used it only
    as a membership filter.
    """

    def test_a_decode_only_objective_cannot_produce_a_readiness_signal(self):
        decode_only = R.ObjectiveSpec(
            backend="llama_cpu", phases=("decode",),
            protocol_by_phase={"decode": DECODE_PROTOCOL},
            improvement_quantifier=R.QUANTIFIER_BACKEND_WIDE)
        decode_cells = tuple(c for c in green_cells() if c.phase == "decode")
        with self.assertRaises(R.CellInadmissible) as caught:
            green_signal(objective=decode_only, cells=decode_cells)
        self.assertIn("§1.6 quantifies over", str(caught.exception))

    def test_the_gpu_backend_is_held_to_the_same_two_phases(self):
        gpu_only = R.ObjectiveSpec(
            backend="llama_gpu", phases=("prefill",),
            protocol_by_phase={"prefill": GPU_PROTOCOL},
            improvement_quantifier=R.QUANTIFIER_BACKEND_WIDE)
        gpu_cell = cell("cell-gpu-prefill", backend="llama_gpu", phase="prefill",
                        protocol_id=GPU_PROTOCOL)
        with self.assertRaises(R.CellInadmissible):
            green_signal(backend="llama_gpu", objective=gpu_only, cells=(gpu_cell,),
                         spec=matrix_spec(backend="llama_gpu"))

    def test_the_full_phase_set_is_still_accepted(self):
        self.assertEqual(green_signal().standing, R.STANDING_MET)


class SelfAuditsAreBoundToThisModuleTest(unittest.TestCase):
    """An audit satisfiable by deleting its subject audits nothing.

    Both self-audits accept a caller-supplied `source` and returned PASS when
    handed the empty string: no multiplication in `""`, therefore "this module
    cannot express a weighted average". The FAIL paths are unaffected — a
    fabricated snippet that really does contain `a * wa + b * wb` still FAILs —
    but a clean bill of health is now issued only for source that is this module.
    """

    def test_the_weighting_audit_does_not_pass_the_empty_string(self):
        result = R.audit_no_weighting_or_averaging("")
        self.assertEqual(result.outcome, S.COULD_NOT_CHECK)
        self.assertIn("deleting what it inspects", result.reasons[0])

    def test_the_write_audit_does_not_pass_the_empty_string(self):
        self.assertEqual(R.audit_no_write_or_process_paths("").outcome,
                         S.COULD_NOT_CHECK)

    def test_neither_audit_passes_an_innocent_stand_in_module(self):
        stand_in = "MODULE_ID = 'something.else/v1'\ndef f(x):\n    return x\n"
        self.assertEqual(R.audit_no_weighting_or_averaging(stand_in).outcome,
                         S.COULD_NOT_CHECK)
        self.assertEqual(R.audit_no_write_or_process_paths(stand_in).outcome,
                         S.COULD_NOT_CHECK)

    def test_both_audits_still_pass_on_this_module_itself(self):
        self.assertEqual(R.audit_no_weighting_or_averaging().outcome, S.PASS)
        self.assertEqual(R.audit_no_write_or_process_paths().outcome, S.PASS)

    def test_both_audits_still_fail_on_a_real_violation(self):
        self.assertEqual(
            R.audit_no_weighting_or_averaging(
                "def f(a, b, wa, wb):\n    return a * wa + b * wb\n").outcome, S.FAIL)
        self.assertEqual(
            R.audit_no_write_or_process_paths(
                "import os\ndef f():\n    os.remove('/tmp/x')\n").outcome, S.FAIL)


class StandingIsDerivedNotStampedTest(unittest.TestCase):
    """The one field a caller could simply set.

    Everything else in this module is structural, but `standing` was a plain
    dataclass field: `dataclasses.replace(signal, standing='objective_met',
    blockers=())` produced a `ReadinessSignal` that rendered `standing=met` with
    an empty blocker list while its own phase standings said
    `improved=COULD_NOT_CHECK`. `api.Verdict` re-derives its status in
    `__post_init__` and raises `VerdictTampering`; §4 invariant 14 asks for the
    same lock here — *"the LLM may request, never declare."*
    """

    def _not_met_signal(self) -> R.ReadinessSignal:
        signal = green_signal(
            cells=tuple(c for c in green_cells() if c.cell_id != "cell-decode-a")
            + (cell("cell-decode-a"),))
        self.assertNotEqual(signal.standing, R.STANDING_MET)
        return signal

    def test_a_standing_cannot_be_upgraded_to_met(self):
        signal = self._not_met_signal()
        with self.assertRaises(R.StandingNotDerived) as caught:
            dataclasses.replace(signal, standing=R.STANDING_MET, blockers=())
        self.assertIn("do not follow from this signal's own evidence",
                      str(caught.exception))

    def test_blockers_cannot_be_dropped_while_keeping_the_standing(self):
        signal = self._not_met_signal()
        self.assertTrue(signal.blockers)
        with self.assertRaises(R.StandingNotDerived):
            dataclasses.replace(signal, blockers=())

    def test_a_met_standing_cannot_be_downgraded_either(self):
        """The lock is on disagreement, not on optimism: both directions raise."""
        with self.assertRaises(R.StandingNotDerived):
            dataclasses.replace(green_signal(), standing=R.STANDING_NOT_MET)

    def test_an_undeclared_blocker_is_refused(self):
        with self.assertRaises(R.CellInadmissible):
            dataclasses.replace(green_signal(), blockers=("MADE_UP",))

    def test_a_faithful_copy_survives(self):
        """The guard must not forbid its own idiom."""
        signal = green_signal()
        self.assertEqual(dataclasses.replace(signal).standing, signal.standing)

    def test_the_declared_quantifier_still_decides_the_re_derivation(self):
        strict = green_signal(objective=objective(
            improvement_quantifier=R.QUANTIFIER_PER_PROTECTED_CELL))
        self.assertEqual(strict.standing, R.STANDING_UNDETERMINED)
        self.assertEqual(dataclasses.replace(strict).standing,
                         R.STANDING_UNDETERMINED)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
