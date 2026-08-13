#!/usr/bin/env python3
"""test_statistics.py — the regression barrier for the AK3 reducer's guarantees.

WHY THIS FILE EXISTS
--------------------
Every defect this project has recorded in a statistical instrument was visible
in the source and asserted nowhere: a threshold that was typed rather than
derived, a sample extended until the answer settled, an LCB standing in for a
test, an MDE computed after the estimate, and a maximum-over-candidates reported
as the candidate's own effect. `statistics.py` makes each of those
unrepresentable; this suite is what keeps them unrepresentable.

The properties under test, each traceable to a clause of
`measurement/protocols/kernel-research.md` (P-AK-SEARCH-1, RATIFIED 2026-08-03):

  * *"every threshold is derived, none is supplied"* — `api.CalibrationOutputs`
    has exactly one producer, and it produces one only when every condition the
    protocol names has been evaluated and passed;
  * the normative SOLVE ORDER is executed and recorded;
  * the e-process is a genuine supermartingale (measured false-positive rate at
    or below the calibrated budget) and is ANYTIME-VALID;
  * the stopping rule cannot be extended past its declared bound, cannot be
    mutated mid-run, and cannot be looked at at an undeclared point;
  * the MDE does not depend on the candidate's data — "published WITH the
    result" is a checkable fact, not a promise;
  * `B_min` never falls below the P-BENCH-1 reps rule, and a FIXED owning rule
    is not raised;
  * the anchor gate VOIDS outside its calibrated band;
  * selection and confirmation are disjoint, and confirmation evidence gathered
    before lineage entry is refused;
  * the LCB is labelled `descriptive`, is not a test, and no decision reads it.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO PROCESS, NO FILE IS WRITTEN. The suite
also proves that last property of the module under test by running api's own AST
audit against `statistics.py`'s source.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_statistics.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/evaluator/test_statistics.py
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
import unittest
from pathlib import Path

# Import through the PACKAGE so `api.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S                    # noqa: E402
from autokernel.evaluator import api                   # noqa: E402
from autokernel.evaluator import statistics as st      # noqa: E402

CAMPAIGN_SEED = "campaign-seed-0001"
SCALE = st.EFFECT_SCALE_RELATIVE
HIGHER = "higher_better"
LOWER = "lower_better"
NOW = "2026-08-03T12:00:00+00:00"


# =============================================================================
# Fixtures — synthetic samples only. Nothing here runs, builds or measures.
# =============================================================================

def make_blocks(n, *, effect, noise, seed, stratum=api.STRATUM_SELECTION,
                unit_prefix="u", start=0, base=100.0, reps=3):
    """`n` paired blocks whose true relative effect is `effect`.

    The two arms are drawn independently around the same base, so the A/A
    fixture (`effect=0`) has exactly the dispersion a real A/A control would.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        anchor = tuple(base + rng.gauss(0, noise * base) for _ in range(reps))
        a_med = sorted(anchor)[reps // 2]
        cand = tuple(a_med * (1.0 + effect) + rng.gauss(0, noise * base)
                     for _ in range(reps))
        idx = start + i
        out.append(st.PairedBlock(
            block_index=idx, unit_id=f"{unit_prefix}-{idx}", stratum=stratum,
            order=st.ORDER_ANCHOR_FIRST if idx % 2 == 0 else st.ORDER_CANDIDATE_FIRST,
            anchor_samples=anchor, candidate_samples=cand))
    return tuple(out)


def make_rule(*, max_rounds=1, blocks_per_round=5, ceiling=20, futility=None):
    return st.StoppingRule(
        rule_id="ak-stop-1", final_table="t1a_paired_block_table",
        decisions=(("evidence_threshold_crossed", "compose_into_champion_lineage"),
                   ("extension_exhausted", "abandon"),
                   ("block_ceiling_reached", "abandon"))
        + ((("futility_stop", "abandon"),) if futility is not None else ()),
        extension=st.BoundedExtension(max_rounds=max_rounds,
                                      blocks_per_round=blocks_per_round),
        max_blocks_per_candidate=ceiling, futility=futility)


def make_controls(**over):
    kwargs = dict(calibration_block_count=200, contribution_floor=0.10, max_candidates=10,
                  confirmation_admission_count=2, max_blocks_per_candidate=20,
                  storage_floor_bytes_free=10 * 2 ** 30)
    kwargs.update(over)
    return api.CampaignControls(**kwargs)


def make_inputs(**over):
    construction = over.pop("construction",
                            st.select_construction("sign_martingale_predictable_lambda/v1"))
    rule = over.pop("stopping_rule", make_rule())
    controls = over.pop("controls", make_controls())
    kwargs = dict(
        backend="llama_cpu", phase="decode", cell_class="microbench_op",
        campaign_seed=CAMPAIGN_SEED, controls=controls, stopping_rule=rule,
        construction=construction, effect_scale=SCALE, metric_direction=HIGHER,
        hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
        aa_blocks=make_blocks(200, effect=0.0, noise=0.01, seed=1, unit_prefix="aa"),
        neutral_blocks=make_blocks(60, effect=0.0, noise=0.01, seed=2, unit_prefix="nt"),
        anchor_calibration_values=_anchor_values(),
        samples_ref="ak-raw://calibration/0001")
    kwargs.update(over)
    return st.CalibrationInputs(**kwargs)


def _anchor_values(n=200, seed=3, base=100.0, sd=1.0):
    rng = random.Random(seed)
    return tuple(base + rng.gauss(0, sd) for _ in range(n))


def make_request(*, calibration, controls, candidate_id="akc-0001", direction=HIGHER,
                 backend="llama_cpu", phase="decode", cell_class="microbench_op",
                 tier="T1a"):
    return api.EvaluationRequest(
        event_id="ake-0001", campaign_id="ak-1", candidate_id=candidate_id, tier=tier,
        backend=backend, phase=phase, cell_class=cell_class, protocol_id=api.PROTOCOL_ID,
        artifact=api.ArtifactIdentity(source_sha256="a" * 64, binary_sha256="b" * 64,
                                      linkage_sha256="c" * 64),
        # Real digests, not `d`*40 / `e`*64: an anchor is the one identity that
        # may never be a placeholder (`schemas.is_placeholder_digest`), and a
        # fixture that fills it in would not survive `build_evaluation_event`.
        anchor=api.AnchorIdentity(
            source_commit=hashlib.sha1(b"anchor-commit").hexdigest(),
            binary_sha256=hashlib.sha256(b"anchor-binary").hexdigest(),
            linkage_sha256=hashlib.sha256(b"anchor-linkage").hexdigest()),
        evaluator=api.EvaluatorIdentity(id="autokernel.evaluator/v1",
                                        bundle_sha256="1" * 64,
                                        runtime_source_label_ref="srclabel-0001"),
        scope_denominator=api.ScopeDenominator(machine_subset="partial",
                                               numa_nodes=(0,), devices=(), cores=24),
        scope_manifest_sha256="3" * 64, co_residency="single",
        determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                          same_seed_repeat_runs=3),
        metric="decode_tokens_per_second", metric_direction=direction, reps=3,
        change_class="parameter", anchor_tier=tier, transfer_ratio_to=(),
        created_at=NOW, campaign_controls=controls, calibration=calibration)


class _Campaign:
    """One accepted calibration plus the campaign state built on it.

    Built once per process because the calibration solve is the expensive part
    and it is deterministic — the same inputs give the same outputs, which is
    itself asserted in `TestDeterminism`.
    """

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            controls = make_controls()
            rule = make_rule()
            inputs = make_inputs(controls=controls, stopping_rule=rule)
            solve = st.solve_calibration(inputs)
            cal = solve.require_accepted()
            commitment = st.StoppingRuleCommitment.commit(
                rule, campaign_id="ak-1", committed_at=NOW)
            split = st.StratumSplitRule(
                rule_id="split-1", campaign_seed=CAMPAIGN_SEED,
                confirmation_fraction=0.3,
                rotation=st.RotationSchedule(schedule_id="rot-1", period_campaigns=4))
            campaign = st.CampaignStatistics(
                campaign_id="ak-1", campaign_seed=CAMPAIGN_SEED, effect_scale=SCALE,
                hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, stopping_rule=rule,
                stopping_rule_commitment=commitment, split_rule=split,
                construction=inputs.construction, calibration=cal,
                aa_effect_pool=solve.aa_effect_pool,
                anchor_calibration_values=solve.anchor_calibration_values)
            cls._cache = (inputs, solve, cal, rule, commitment, split, campaign, controls)
        return cls._cache


def selection_unit(split, index):
    """A unit id the recorded split rule assigns to the selection stratum."""
    unit = f"sel-{index}"
    while split.assign(unit) != api.STRATUM_SELECTION:
        unit += "x"
    return unit


def confirmation_unit(split, index):
    unit = f"conf-{index}"
    while split.assign(unit) != api.STRATUM_CONFIRMATION:
        unit += "x"
    return unit


def run_candidate(campaign, *, effect, noise=0.01, seed=99, direction=HIGHER,
                  candidate_id="akc-0001", stratum=api.STRATUM_SELECTION):
    """Drive a candidate through the rule, returning (evaluation, blocks)."""
    split = campaign.split_rule
    seq = campaign.sequential_evaluation(candidate_id=candidate_id, stratum=stratum,
                                         metric_direction=direction)
    rng = random.Random(seed)
    blocks = []
    while not seq.terminal:
        req = seq.next_block_request()
        anchor = tuple(100.0 + rng.gauss(0, noise * 100.0) for _ in range(3))
        a_med = sorted(anchor)[1]
        cand = tuple(a_med * (1.0 + effect) + rng.gauss(0, noise * 100.0)
                     for _ in range(3))
        unit = (selection_unit(split, req.block_index)
                if stratum == api.STRATUM_SELECTION
                else confirmation_unit(split, req.block_index))
        block = st.PairedBlock(
            block_index=req.block_index, unit_id=unit, stratum=stratum, order=req.order,
            anchor_samples=anchor, candidate_samples=cand, segment=req.segment,
            extension_round=req.extension_round, measured_at=NOW)
        seq.submit_block(block)
        blocks.append(block)
    return seq, tuple(blocks)


# =============================================================================
# T1a absolute measurement floor — local paired A/A only
# =============================================================================

class TestMinimumMeasurableDuration(unittest.TestCase):

    def test_floor_is_derived_from_local_paired_absolute_spread(self):
        anchors = (100.0, 101.0, 102.0, 103.0, 104.0)
        candidates = (100.1, 100.8, 102.3, 102.6, 104.5)
        differences = [abs(a - b) for a, b in zip(anchors, candidates)]
        expected_spread = st.percentile(differences, 0.95)
        floor = st.derive_minimum_measurable_duration(
            anchors, candidates, relative_noise_budget=0.03,
            samples_ref="fixture:local-paired-aa-us")
        self.assertAlmostEqual(floor.aa_absolute_spread_us, expected_spread)
        self.assertAlmostEqual(floor.min_measurable_us, expected_spread / 0.03)
        self.assertEqual(floor.aa_pair_count, 5)
        self.assertEqual(
            floor.check_observed_us(floor.min_measurable_us / 2).outcome,
            S.COULD_NOT_CHECK)
        self.assertEqual(
            floor.check_observed_us(floor.min_measurable_us).outcome, S.PASS)

    def test_zero_spread_cannot_manufacture_a_zero_us_floor(self):
        with self.assertRaisesRegex(st.MaterialError, "zero"):
            st.derive_minimum_measurable_duration(
                (10.0,) * 5, (10.0,) * 5, relative_noise_budget=0.03,
                samples_ref="fixture:zero-spread")

    def test_capability_refuses_a_literal_that_breaks_the_derivation(self):
        with self.assertRaisesRegex(ValueError, "must equal"):
            api.MinimumMeasurableDuration(
                min_measurable_us=10.0, aa_absolute_spread_us=1.0,
                relative_noise_budget=0.05, aa_pair_count=5,
                samples_ref="fixture:not-derived")


# =============================================================================
# Robust reduction and the named quantile estimator
# =============================================================================

class TestRobustReduction(unittest.TestCase):

    def test_median_and_mad_are_the_reported_pair(self):
        self.assertEqual(st.median([1, 2, 3]), 2)
        self.assertEqual(st.median([1, 2, 3, 4]), 2.5)
        self.assertEqual(st.mad([1, 2, 3, 4, 100]), 1.0)

    def test_median_of_nothing_raises_rather_than_returning_zero(self):
        with self.assertRaises(st.MaterialError):
            st.median([])
        with self.assertRaises(st.MaterialError):
            st.mad([])

    def test_non_finite_sample_raises(self):
        for bad in (float("nan"), float("inf"), None, True, "3"):
            with self.assertRaises(st.MaterialError):
                st.median([1.0, bad])

    def test_percentile_estimator_is_named_and_interpolating(self):
        self.assertEqual(st.PERCENTILE_METHOD, "linear_interpolation_type7")
        self.assertAlmostEqual(st.percentile([0, 1, 2, 3, 4], 0.5), 2.0)
        self.assertAlmostEqual(st.percentile([0, 10], 0.95), 9.5)
        self.assertAlmostEqual(st.percentile([5], 0.95), 5.0)

    def test_min_samples_for_quantile_is_derived_not_tabulated(self):
        self.assertEqual(st.min_samples_for_quantile(0.95), 20)
        self.assertEqual(st.min_samples_for_quantile(0.99), 100)
        self.assertEqual(st.min_samples_for_quantile(0.5), 2)


# =============================================================================
# The constitutional rep floor
# =============================================================================

class TestRepsFloor(unittest.TestCase):

    def test_p_bench_1_limbs(self):
        self.assertEqual(st.reps_floor_for_relative_effect(0.05).blocks, 5)
        self.assertEqual(st.reps_floor_for_relative_effect(0.12).blocks, 5)
        self.assertEqual(st.reps_floor_for_relative_effect(0.02).blocks, 10)
        self.assertEqual(st.reps_floor_for_relative_effect(0.005).blocks, 10)

    def test_undefined_band_takes_the_stricter_limb_and_says_so(self):
        floor = st.reps_floor_for_relative_effect(0.03)
        self.assertEqual(floor.blocks, 10)
        self.assertTrue(floor.conservative)
        self.assertIn("2%", floor.note)
        self.assertEqual(floor.band, "undefined_band_2pct_to_5pct")

    def test_zero_or_unbounded_floor_refuses(self):
        for bad in (0, -0.01, float("inf"), float("nan")):
            with self.assertRaises(st.MaterialError):
                st.reps_floor_for_relative_effect(bad)

    def test_fixed_owning_rule_is_not_a_floor_to_be_raised(self):
        fixed = st.OwningProtocolRepRule(
            protocol_id="P-BENCH-4", kind=st.REP_RULE_FIXED, blocks=5,
            citation="bench-cpu.md:174-178 — exactly five, no retry, replace, "
                     "discard or pooling")
        # A campaign whose contribution floor would demand 10 blocks still gets
        # exactly 5, because the owning protocol's count is fixed.
        self.assertEqual(st._b_min_candidates(10, 20, fixed), (5,))
        self.assertEqual(st._start_blocks(st.reps_floor_for_relative_effect(0.01), fixed), 5)

    def test_floor_kind_owning_rule_raises_the_start_but_never_lowers_it(self):
        floor_rule = st.OwningProtocolRepRule(
            protocol_id="P-SHED-1", kind=st.REP_RULE_FLOOR, blocks=10,
            citation="gpu-cross-device.md:146-147")
        self.assertEqual(st._start_blocks(st.reps_floor_for_relative_effect(0.20),
                                          floor_rule), 10)
        self.assertEqual(st._start_blocks(st.reps_floor_for_relative_effect(0.001),
                                          floor_rule), 10)


# =============================================================================
# Order control
# =============================================================================

class TestOrderControl(unittest.TestCase):

    def test_schedule_is_derived_from_the_committed_campaign_seed(self):
        a = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED, candidate_id="akc-1",
                                    base_blocks=8)
        b = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED, candidate_id="akc-1",
                                    base_blocks=8)
        c = st.OrderSchedule.derive(campaign_seed="other-seed", candidate_id="akc-1",
                                    base_blocks=8)
        self.assertEqual(a.orders(8), b.orders(8))
        self.assertNotEqual(a.orders(8), c.orders(8))

    def test_schedule_is_prefix_stable_so_extending_cannot_invalidate_the_base(self):
        sched = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                        candidate_id="akc-1", base_blocks=5)
        self.assertEqual(sched.orders(5), sched.orders(12)[:5])

    def test_fifteen_block_schedule_is_counterbalanced_within_one(self):
        """A declared 15-block base cannot leave temporal order 10/5 or worse."""
        sched = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                        candidate_id="akm-iqk-0001", base_blocks=15)
        orders = sched.orders(15)
        self.assertLessEqual(
            abs(orders.count(st.ORDER_ANCHOR_FIRST)
                - orders.count(st.ORDER_CANDIDATE_FIRST)), 1)

    def test_extension_blocks_are_reversed_order(self):
        sched = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                        candidate_id="akc-1", base_blocks=5)
        for i in range(5):
            self.assertNotEqual(sched.order_for(i), sched.order_for(i + 5))

    def test_retry_is_a_fresh_reset_in_reversed_order(self):
        sched = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                        candidate_id="akc-1", base_blocks=6)
        retry = sched.retry()
        for i in range(6):
            self.assertNotEqual(sched.order_for(i), retry.order_for(i))
        self.assertEqual(retry.attempt, sched.attempt + 1)
        # A retry is a REVERSAL, not a re-randomization: reversing twice returns
        # the original order, and that is what makes "reversed on retry"
        # verifiable at all.
        self.assertEqual(sched.orders(6), retry.retry().orders(6))

    def test_deriving_at_an_attempt_is_the_same_schedule_as_retrying_to_it(self):
        """THE BITE: `retry()` has no caller outside these tests.

        Everything that builds a schedule builds it with `derive()` —
        `MicrobenchPlan.schedule`, `CampaignStatistics.order_schedule` — so if
        `derive` ignores the attempt, `retry()` is the only implementation of
        *"a retry is a fresh reset in reversed order"* and nothing calls it: a
        retried run re-ran the IDENTICAL order sequence, on both sides at once,
        so nothing failed and the aliasing the reversal exists to break was
        re-committed. Both spellings of attempt `n` must be one schedule.
        """
        first = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                        candidate_id="akc-1", base_blocks=6)
        walked = first
        for attempt in (1, 2, 3):
            walked = walked.retry()
            derived = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                              candidate_id="akc-1", base_blocks=6,
                                              attempt=attempt)
            self.assertEqual(derived, walked)
            self.assertEqual(derived.orders(6), walked.orders(6))
        self.assertNotEqual(
            first.orders(6),
            st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED, candidate_id="akc-1",
                                    base_blocks=6, attempt=1).orders(6))

    def test_attempt_zero_is_unchanged_and_a_second_retry_reverses_back(self):
        """Compliant control: the first attempt's schedule must not have moved."""
        base = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                       candidate_id="akc-1", base_blocks=6)
        self.assertFalse(base.reversed_schedule)
        self.assertEqual(
            base.orders(6),
            st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED, candidate_id="akc-1",
                                    base_blocks=6, attempt=2).orders(6))

    def test_blocked_design_is_named_as_such(self):
        sched = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                        candidate_id="akc-1", base_blocks=8)
        blocked = tuple(
            st.PairedBlock(block_index=i, unit_id=f"u-{i}", stratum=api.STRATUM_SELECTION,
                           order=st.ORDER_ANCHOR_FIRST, anchor_samples=(1.0,),
                           candidate_samples=(1.0,))
            for i in range(8))
        chk = sched.check_observed(blocked)
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("BLOCKED design" in r for r in chk.reasons))

    def test_no_blocks_is_could_not_check_not_pass(self):
        sched = st.OrderSchedule.derive(campaign_seed=CAMPAIGN_SEED,
                                        candidate_id="akc-1", base_blocks=4)
        self.assertEqual(sched.check_observed(()).outcome, S.COULD_NOT_CHECK)


# =============================================================================
# The e-process
# =============================================================================

class TestEProcess(unittest.TestCase):

    def setUp(self):
        self.construction = st.select_construction("sign_martingale_predictable_lambda/v1")

    def test_only_registry_constructions_are_selectable(self):
        with self.assertRaises(st.ConstructionNotImplemented):
            st.select_construction("my_own_bound/v1")
        with self.assertRaises(st.ConstructionNotImplemented):
            st.select_construction(None)

    def test_construction_identity_is_a_content_hash_over_its_parameters(self):
        a = st.select_construction("sign_martingale_predictable_lambda/v1")
        b = st.select_construction("sign_martingale_fixed_lambda/v1")
        self.assertNotEqual(a.content_hash(), b.content_hash())
        self.assertEqual(a.content_hash(), a.content_hash())
        self.assertIn("mde_power_target", a.to_dict())

    def test_betting_fraction_is_predictable_and_capped(self):
        c = self.construction
        self.assertEqual(c.lambda_for(()), c.lambda_init)
        self.assertLessEqual(c.lambda_for((1.0,) * 10), c.lambda_cap)
        self.assertGreaterEqual(c.lambda_for((-1.0,) * 10), 0.0)
        # The moment form and the sample form are the same definition.
        past = (1.0, -1.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(
            c.lambda_for(past),
            c.lambda_from_moments(len(past), sum(past), sum(x * x for x in past)))

    def test_wealth_stays_positive_for_every_admissible_lambda(self):
        for cid in sorted(st.CONSTRUCTIONS):
            c = st.select_construction(cid)
            run = st.run_e_process([-1.0] * 60, construction=c,
                                   hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                                   threshold=20.0)
            self.assertTrue(all(math.isfinite(w) for w in run.log_wealth))
            self.assertGreater(run.e_final, 0.0)

    def test_false_positive_rate_is_at_or_below_the_budget(self):
        """A supermartingale, measured. Ville's bound, not asserted but sampled."""
        alpha = 0.1
        threshold = 1.0 / alpha
        rng = random.Random(4242)
        crossings = 0
        trials = 400
        for _ in range(trials):
            oriented = [rng.gauss(0.0, 0.01) for _ in range(20)]
            run = st.run_e_process(oriented, construction=self.construction,
                                   hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                                   threshold=threshold)
            if run.crossed:
                crossings += 1
        self.assertLessEqual(crossings / trials, alpha,
                             f"{crossings}/{trials} crossings exceeds alpha={alpha}")

    def test_anytime_validity_reports_the_running_maximum(self):
        """A process that crossed and fell back still rejected."""
        oriented = [1.0] * 12 + [-1.0] * 12
        run = st.run_e_process(oriented, construction=self.construction,
                               hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                               threshold=5.0)
        self.assertTrue(run.crossed)
        self.assertGreater(run.e_running_max, run.e_final)
        self.assertLess(run.e_final, 5.0)

    def test_threshold_at_or_below_one_is_refused(self):
        for bad in (1.0, 0.5, 0.0, -3.0, float("inf")):
            with self.assertRaises(st.MaterialError):
                st.run_e_process([0.1], construction=self.construction,
                                 hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                                 threshold=bad)

    def test_zero_blocks_is_not_a_measurement(self):
        with self.assertRaises(st.InsufficientMaterial):
            st.run_e_process([], construction=self.construction,
                             hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                             threshold=10.0)

    def test_improvement_hypothesis_refuses_a_smuggled_margin(self):
        with self.assertRaises(st.MaterialError):
            st.null_boundary_for(st.HYPOTHESIS_IMPROVEMENT, 0.02)
        self.assertEqual(st.null_boundary_for(st.HYPOTHESIS_NON_INFERIORITY, 0.02), -0.02)
        with self.assertRaises(st.MaterialError):
            st.null_boundary_for(st.HYPOTHESIS_NON_INFERIORITY, 0.0)

    def test_an_unrepresentable_e_value_is_refused_not_clipped(self):
        run = st.EProcessRun(
            construction_id="sign_martingale_predictable_lambda/v1",
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, null_boundary=0.0,
            threshold=10.0, log_threshold=math.log(10.0), log_wealth=(1e6,),
            lambdas=(0.5,), signs=(1.0,), log_e_final=1e6, log_e_running_max=1e6,
            first_crossing_block=1)
        with self.assertRaises(st.EValueNotRepresentable) as cm:
            _ = run.e_running_max
        self.assertIn("log_e_running_max", str(cm.exception))
        self.assertEqual(run.log_e_running_max, 1e6)   # the exact value survives

    def test_sign_statistic_is_robust_to_a_single_outlier(self):
        base = [0.01] * 12
        outlier = list(base)
        outlier[3] = 500.0
        a = st.run_e_process(base, construction=self.construction,
                             hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                             threshold=10.0)
        b = st.run_e_process(outlier, construction=self.construction,
                             hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                             threshold=10.0)
        self.assertEqual(a.log_e_running_max, b.log_e_running_max)


# =============================================================================
# The pre-committed stopping rule
# =============================================================================

class TestStoppingRule(unittest.TestCase):

    def test_unbounded_extension_is_unrepresentable(self):
        for bad in (None, math.inf, 3.5, "many", -1):
            with self.assertRaises(st.StoppingRuleViolation):
                st.BoundedExtension(max_rounds=bad, blocks_per_round=5)

    def test_extension_must_be_reversed_and_pooled(self):
        with self.assertRaises(st.StoppingRuleViolation):
            st.BoundedExtension(max_rounds=1, blocks_per_round=5, order="same")
        with self.assertRaises(st.StoppingRuleViolation):
            st.BoundedExtension(max_rounds=1, blocks_per_round=5, pooled=False)

    def test_rule_must_name_the_final_table(self):
        with self.assertRaises(st.StoppingRuleViolation):
            st.StoppingRule(rule_id="r", final_table="",
                            decisions=(("evidence_threshold_crossed", "retain"),
                                       ("extension_exhausted", "abandon"),
                                       ("block_ceiling_reached", "abandon")),
                            extension=st.BoundedExtension(max_rounds=0, blocks_per_round=1),
                            max_blocks_per_candidate=10)

    def test_every_reachable_outcome_needs_a_declared_decision(self):
        with self.assertRaises(st.StoppingRuleViolation) as cm:
            st.StoppingRule(rule_id="r", final_table="t",
                            decisions=(("evidence_threshold_crossed", "retain"),),
                            extension=st.BoundedExtension(max_rounds=0, blocks_per_round=1),
                            max_blocks_per_candidate=10)
        self.assertIn("block_ceiling_reached", str(cm.exception))

    def test_an_unreachable_outcome_may_not_be_declared(self):
        with self.assertRaises(st.StoppingRuleViolation):
            st.StoppingRule(rule_id="r", final_table="t",
                            decisions=(("evidence_threshold_crossed", "retain"),
                                       ("extension_exhausted", "abandon"),
                                       ("block_ceiling_reached", "abandon"),
                                       ("futility_stop", "abandon")),
                            extension=st.BoundedExtension(max_rounds=0, blocks_per_round=1),
                            max_blocks_per_candidate=10)

    def test_a_search_record_may_not_declare_a_forbidden_decision(self):
        for forbidden in ("promote", "deploy", "freeze", "ship_it", "revert_production"):
            with self.assertRaises(st.StoppingRuleViolation) as cm:
                st.StoppingRule(rule_id="r", final_table="t",
                                decisions=(("evidence_threshold_crossed", forbidden),
                                           ("extension_exhausted", "abandon"),
                                           ("block_ceiling_reached", "abandon")),
                                extension=st.BoundedExtension(max_rounds=0,
                                                              blocks_per_round=1),
                                max_blocks_per_candidate=10)
            self.assertIn("denial 1", str(cm.exception))

    def test_only_the_exact_parameter_free_futility_form_exists(self):
        with self.assertRaises(st.StoppingRuleViolation):
            st.FutilityRule(kind="looks_unpromising")
        self.assertEqual(st.FutilityRule().kind, st.FUTILITY_UNREACHABLE_THRESHOLD)

    def test_commitment_detects_a_post_hoc_rule_change(self):
        rule = make_rule()
        commitment = st.StoppingRuleCommitment.commit(rule, campaign_id="ak-1",
                                                      committed_at=NOW)
        self.assertEqual(commitment.verify(rule).outcome, S.PASS)
        mutated = make_rule(max_rounds=3)
        chk = commitment.verify(mutated)
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("VOIDS every affected record" in r for r in chk.reasons))

    def test_max_total_blocks_respects_the_declared_ceiling(self):
        rule = make_rule(max_rounds=10, blocks_per_round=5, ceiling=20)
        self.assertEqual(rule.max_total_blocks(5), 20)
        rule2 = make_rule(max_rounds=1, blocks_per_round=5, ceiling=100)
        self.assertEqual(rule2.max_total_blocks(5), 10)


class TestSequentialEvaluation(unittest.TestCase):

    def setUp(self):
        (_i, _s, self.cal, self.rule, self.commitment, self.split, self.campaign,
         self.controls) = _Campaign.get()

    def test_a_real_improvement_crosses_and_takes_the_declared_decision(self):
        seq, blocks = run_candidate(self.campaign, effect=0.15)
        self.assertEqual(seq.outcome, "evidence_threshold_crossed")
        decision = seq.decide()
        self.assertEqual(decision.decision, "compose_into_champion_lineage")
        self.assertIn(decision.decision, st.AUTHORIZED_DECISIONS)
        self.assertGreaterEqual(len(blocks), self.campaign.b_min)

    def test_a_null_candidate_exhausts_the_extension_and_is_abandoned(self):
        seq, blocks = run_candidate(self.campaign, effect=0.0, seed=7)
        self.assertIn(seq.outcome, ("extension_exhausted", "block_ceiling_reached"))
        self.assertEqual(seq.decide().decision, "abandon")
        self.assertEqual(len(blocks), self.rule.max_total_blocks(self.campaign.b_min))

    def test_the_rule_never_stops_before_b_min(self):
        seq = self.campaign.sequential_evaluation(
            candidate_id="akc-0001", stratum=api.STRATUM_SELECTION, metric_direction=HIGHER)
        for i in range(self.campaign.b_min - 1):
            req = seq.next_block_request()
            seq.submit_block(st.PairedBlock(
                block_index=req.block_index, unit_id=selection_unit(self.split, i),
                stratum=api.STRATUM_SELECTION, order=req.order,
                anchor_samples=(100.0,), candidate_samples=(500.0,),
                segment=req.segment, extension_round=req.extension_round))
            self.assertFalse(seq.terminal)

    def test_continuing_past_termination_raises(self):
        seq, _ = run_candidate(self.campaign, effect=0.15)
        self.assertTrue(seq.terminal)
        with self.assertRaises(st.StoppingRuleViolation) as cm:
            seq.next_block_request()
        self.assertIn("Extension follows the declared rule ONLY",
                      str(cm.exception).replace("\n", " "))

    def test_a_block_the_rule_did_not_request_is_refused(self):
        seq = self.campaign.sequential_evaluation(
            candidate_id="akc-0001", stratum=api.STRATUM_SELECTION, metric_direction=HIGHER)
        good = st.PairedBlock(block_index=0, unit_id=selection_unit(self.split, 0),
                              stratum=api.STRATUM_SELECTION, order=st.ORDER_ANCHOR_FIRST,
                              anchor_samples=(100.0,), candidate_samples=(110.0,))
        with self.assertRaises(st.StoppingRuleViolation):
            seq.submit_block(good)          # nothing was requested yet
        req = seq.next_block_request()
        wrong_order = st.PairedBlock(
            block_index=req.block_index, unit_id=selection_unit(self.split, 0),
            stratum=api.STRATUM_SELECTION,
            order=(st.ORDER_CANDIDATE_FIRST if req.order == st.ORDER_ANCHOR_FIRST
                   else st.ORDER_ANCHOR_FIRST),
            anchor_samples=(100.0,), candidate_samples=(110.0,))
        with self.assertRaises(st.StoppingRuleViolation):
            seq.submit_block(wrong_order)

    def test_deciding_before_termination_is_peeking_and_raises(self):
        seq = self.campaign.sequential_evaluation(
            candidate_id="akc-0001", stratum=api.STRATUM_SELECTION, metric_direction=HIGHER)
        with self.assertRaises(st.StoppingRuleViolation):
            seq.decide()

    def test_futility_stops_when_the_threshold_is_unreachable(self):
        rule = make_rule(max_rounds=1, blocks_per_round=5, ceiling=20,
                         futility=st.FutilityRule())
        commitment = st.StoppingRuleCommitment.commit(rule, campaign_id="ak-1",
                                                      committed_at=NOW)
        campaign = st.CampaignStatistics(
            campaign_id="ak-1", campaign_seed=CAMPAIGN_SEED, effect_scale=SCALE,
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, stopping_rule=rule,
            stopping_rule_commitment=commitment, split_rule=self.split,
            construction=self.campaign.construction, calibration=self.cal,
            aa_effect_pool=self.campaign.aa_effect_pool,
            anchor_calibration_values=self.campaign.anchor_calibration_values)
        seq, blocks = run_candidate(campaign, effect=-0.30, seed=11)
        self.assertEqual(seq.outcome, "futility_stop")
        self.assertEqual(seq.decide().decision, "abandon")
        self.assertLess(len(blocks), rule.max_total_blocks(campaign.b_min))

    def test_a_rule_mutated_mid_run_is_caught_at_the_next_block(self):
        bad_commitment = st.StoppingRuleCommitment(
            campaign_id="ak-1", rule_id=self.rule.rule_id,
            rule_content_hash="0" * 64, committed_at=NOW)
        with self.assertRaises(st.StoppingRuleMutated):
            st.SequentialEvaluation(
                rule=self.rule, commitment=bad_commitment,
                construction=self.campaign.construction, b_min=self.campaign.b_min,
                threshold=self.campaign.threshold_for(api.STRATUM_SELECTION),
                hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                metric_direction=HIGHER, effect_scale=SCALE,
                order_schedule=self.campaign.order_schedule("akc-0001"))


# =============================================================================
# The calibration block
# =============================================================================

class TestCalibrationSolve(unittest.TestCase):

    def setUp(self):
        (self.inputs, self.solve, self.cal, self.rule, _c, _sp, self.campaign,
         self.controls) = _Campaign.get()

    def test_calibration_outputs_have_exactly_one_construction_site(self):
        """*"every threshold is derived, none is supplied"* — as a source fact.

        If a second `api.CalibrationOutputs(...)` ever appears in this module, a
        second route to a threshold exists, and the first thing such a route
        does in practice is accept a literal.
        """
        source = Path(st.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("api.CalibrationOutputs("), 1)
        self.assertIn("outputs = api.CalibrationOutputs(", source)
        self.assertTrue(self.solve.accepted)
        self.assertIsInstance(self.solve.outputs, api.CalibrationOutputs)

    def test_the_accepted_solve_is_canonicalizable_for_the_manifest(self):
        S.canonical_json(self.solve.to_dict())
        self.assertEqual(self.solve.to_dict()["solve_order"],
                         list(api.CALIBRATION_SOLVE_ORDER))

    def test_the_normative_solve_order_is_recorded(self):
        self.assertEqual(tuple(self.cal.solve_order_recorded), api.CALIBRATION_SOLVE_ORDER)
        for attempt in self.solve.attempts:
            self.assertEqual(tuple(attempt.solve_order_recorded),
                             api.CALIBRATION_SOLVE_ORDER)

    def test_alpha_sel_comes_from_max_candidates_and_alpha_conf_from_the_split(self):
        self.assertLessEqual(self.cal.alpha_sel, 1.0 / self.controls.max_candidates)
        self.assertLessEqual(self.cal.alpha_conf,
                             self.cal.alpha_sel / self.controls.confirmation_admission_count)
        self.assertLessEqual(self.cal.alpha_conf, self.cal.alpha_sel)
        self.assertEqual(self.cal.check_against_controls(self.controls).outcome, S.PASS)

    def test_thresholds_are_the_reciprocals_of_the_derived_budgets(self):
        self.assertAlmostEqual(self.cal.threshold_for(api.STRATUM_SELECTION),
                               1.0 / self.cal.alpha_sel)
        self.assertAlmostEqual(self.cal.threshold_for(api.STRATUM_CONFIRMATION),
                               1.0 / self.cal.alpha_conf)
        self.assertGreater(self.cal.threshold_for(api.STRATUM_CONFIRMATION),
                           self.cal.threshold_for(api.STRATUM_SELECTION))

    def test_phi_is_the_p95_of_the_aa_effect_magnitudes(self):
        magnitudes = [abs(e) for e in self.solve.aa_effect_pool]
        self.assertAlmostEqual(self.cal.noise_floor_phi, st.percentile(magnitudes, 0.95))

    def test_b_min_is_at_or_above_the_p_bench_1_floor(self):
        floor = st.reps_floor_for_relative_effect(
            self.inputs.relative_contribution_floor())
        self.assertGreaterEqual(self.cal.b_min_blocks, floor.blocks)

    def test_both_calibration_conditions_hold_at_the_solved_b_min(self):
        attempt = self.solve.attempts[-1]
        self.assertTrue(attempt.accepted)
        self.assertLessEqual(attempt.condition_a.rate, attempt.alpha_sel)
        self.assertTrue(attempt.mde.found)
        self.assertLessEqual(attempt.mde.value, self.controls.contribution_floor)
        self.assertEqual(attempt.alpha_validation.outcome, S.PASS)

    def test_anchor_gate_band_is_a_central_95_percent_interval(self):
        low, high = self.cal.anchor_gate_band
        self.assertLess(low, high)
        centre = st.median(self.solve.anchor_calibration_values)
        self.assertLess(low, centre)
        self.assertGreater(high, centre)

    def test_calibration_is_deterministic_for_identical_inputs(self):
        again = st.solve_calibration(make_inputs())
        self.assertTrue(again.accepted)
        self.assertEqual(again.outputs.to_dict(), self.cal.to_dict())

    def test_an_unaccepted_solve_refuses_to_hand_over_outputs(self):
        bad = st.solve_calibration(make_inputs(
            controls=make_controls(contribution_floor=1e-9, max_blocks_per_candidate=12),
            stopping_rule=make_rule(max_rounds=1, blocks_per_round=2, ceiling=12)))
        self.assertFalse(bad.accepted)
        self.assertIsNone(bad.outputs)
        with self.assertRaises(st.CalibrationFailed):
            bad.require_accepted()
        self.assertTrue(bad.reasons)

    def test_failed_attempts_are_retained_in_the_solve(self):
        bad = st.solve_calibration(make_inputs(
            controls=make_controls(contribution_floor=1e-9, max_blocks_per_candidate=12),
            stopping_rule=make_rule(max_rounds=1, blocks_per_round=2, ceiling=12)))
        self.assertTrue(bad.attempts)
        self.assertFalse(bad.attempts[-1].accepted)
        self.assertTrue(bad.attempts[-1].reasons)
        # Canonicalizable, so it can be written into the manifest verbatim.
        S.canonical_json(bad.to_dict())

    def test_a_ceiling_that_disagrees_with_the_manifest_stops_the_solve(self):
        solve = st.solve_calibration(make_inputs(
            controls=make_controls(max_blocks_per_candidate=30)))
        self.assertFalse(solve.accepted)
        self.assertTrue(any("held constant" in r for r in solve.reasons))

    def test_a_calibration_below_the_declared_block_count_refuses(self):
        with self.assertRaises(st.InsufficientMaterial):
            st.estimate_noise_floor([0.01] * 30, calibration_block_count=200,
                                    neutral_check=S.Check(S.PASS))

    def test_a_p95_over_too_few_points_is_the_maximum_and_is_refused(self):
        with self.assertRaises(st.InsufficientMaterial):
            st.estimate_noise_floor([0.01, -0.02, 0.03], calibration_block_count=3,
                                    neutral_check=S.Check(S.PASS))

    def test_a_zero_noise_floor_would_admit_everything_and_is_refused(self):
        with self.assertRaises(st.InsufficientMaterial):
            st.estimate_noise_floor([0.0] * 40, calibration_block_count=40,
                                    neutral_check=S.Check(S.PASS))

    def test_neutral_check_is_a_required_part_of_output_1(self):
        with self.assertRaises(st.MaterialError):
            st.estimate_noise_floor([0.01 * i for i in range(1, 41)],
                                    calibration_block_count=40, neutral_check=True)

    def test_an_inflated_neutral_control_fails_the_calibration(self):
        inflated = make_blocks(60, effect=0.0, noise=0.04, seed=5, unit_prefix="nt")
        solve = st.solve_calibration(make_inputs(neutral_blocks=inflated))
        self.assertFalse(solve.accepted)
        self.assertTrue(any("materially exceeds the A/A floor" in r for r in solve.reasons))
        self.assertTrue(any("rather than raising the floor" in r for r in solve.reasons))
        # The floor itself is unchanged by the failure — it is not raised.
        self.assertEqual(solve.attempts[-1].noise_floor.value, self.cal.noise_floor_phi)

    def test_a_well_behaved_neutral_control_passes(self):
        aa = [st.block_effect(b, scale=SCALE) for b in self.inputs.aa_blocks]
        nt = [st.block_effect(b, scale=SCALE) for b in self.inputs.neutral_blocks]
        chk = st.neutral_control_consistency(
            nt, aa, campaign_seed=CAMPAIGN_SEED, construction=self.inputs.construction)
        self.assertEqual(chk.outcome, S.PASS)

    def test_neutral_check_is_could_not_check_on_thin_material(self):
        chk = st.neutral_control_consistency(
            [0.01] * 5, [0.01] * 40, campaign_seed=CAMPAIGN_SEED,
            construction=self.inputs.construction)
        self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)

    def test_a_campaign_control_that_is_zero_or_unbounded_cannot_start(self):
        for bad in ({"max_candidates": 0}, {"contribution_floor": 0.0},
                    {"contribution_floor": float("inf")},
                    {"max_blocks_per_candidate": 0}):
            with self.assertRaises(ValueError):
                make_controls(**bad)

    def test_required_disjoint_windows_is_derived(self):
        self.assertEqual(st.required_disjoint_windows(0.1), 10)
        self.assertEqual(st.required_disjoint_windows(0.02), 50)
        with self.assertRaises(st.MaterialError):
            st.required_disjoint_windows(0.0)

    def test_thin_validation_material_is_could_not_check_and_blocks_acceptance(self):
        """0/3 windows cannot demonstrate a rate at or below 1/50."""
        thin = make_blocks(60, effect=0.0, noise=0.01, seed=1, unit_prefix="aa")
        solve = st.solve_calibration(make_inputs(
            aa_blocks=thin,
            controls=make_controls(calibration_block_count=60, max_candidates=50)))
        self.assertFalse(solve.accepted)
        self.assertTrue(any("disjoint A/A windows" in r or "needs at least" in r
                            for r in solve.reasons))

    def test_a_fixed_owning_rep_rule_is_honoured_exactly(self):
        fixed = st.OwningProtocolRepRule(
            protocol_id="P-BENCH-4", kind=st.REP_RULE_FIXED, blocks=5,
            citation="bench-cpu.md:174-178")
        solve = st.solve_calibration(make_inputs(owning_rep_rule=fixed))
        if solve.accepted:
            self.assertEqual(solve.outputs.b_min_blocks, 5)
        else:
            # If the fixed count cannot satisfy the conditions the calibration
            # FAILS; it is never raised to a count that would.
            self.assertTrue(all(a.b_min in (None, 5) for a in solve.attempts))


class TestCalibrationInputValidation(unittest.TestCase):

    def test_a_modified_construction_is_not_selectable_by_a_campaign(self):
        """A campaign selects among the constructions the bundle implements."""
        real = st.select_construction("sign_martingale_predictable_lambda/v1")
        tuned = st.EProcessConstruction(
            construction_id=real.construction_id, statistic=real.statistic,
            betting_form=real.betting_form, lambda_cap=0.9,          # <- tuned
            lambda_init=real.lambda_init, lambda_fixed=real.lambda_fixed,
            mde_power_target=0.5,                                    # <- tuned
            mde_resamples=real.mde_resamples,
            mde_max_doublings=real.mde_max_doublings,
            mde_search_tolerance=real.mde_search_tolerance,
            crossing_rate_resamples=real.crossing_rate_resamples,
            band_resamples=real.band_resamples,
            neutral_permutation_reps=real.neutral_permutation_reps,
            lcb_bootstrap_iterations=real.lcb_bootstrap_iterations,
            description=real.description)
        self.assertNotEqual(tuned.content_hash(), real.content_hash())
        with self.assertRaises(st.ConstructionNotImplemented):
            make_inputs(construction=tuned)

    def test_the_anchor_cell_values_cannot_be_substituted_away(self):
        with self.assertRaises(st.MaterialError):
            make_inputs(anchor_calibration_values=())

    def test_relative_scale_refuses_a_non_positive_anchor(self):
        bad = (st.PairedBlock(block_index=0, unit_id="u-0", stratum=api.STRATUM_SELECTION,
                              order=st.ORDER_ANCHOR_FIRST, anchor_samples=(0.0,),
                              candidate_samples=(1.0,)),)
        with self.assertRaises(st.EffectScaleError):
            st.block_effect(bad[0], scale=st.EFFECT_SCALE_RELATIVE)
        self.assertEqual(st.block_effect(bad[0], scale=st.EFFECT_SCALE_ABSOLUTE), 1.0)


class TestEProcessConstructionFields(unittest.TestCase):

    def test_construction_dict_round_trips_into_a_registry_member(self):
        for cid, con in st.CONSTRUCTIONS.items():
            rebuilt = st.EProcessConstruction(
                construction_id=con.construction_id, statistic=con.statistic,
                betting_form=con.betting_form, lambda_cap=con.lambda_cap,
                lambda_init=con.lambda_init, lambda_fixed=con.lambda_fixed,
                mde_power_target=con.mde_power_target, mde_resamples=con.mde_resamples,
                mde_max_doublings=con.mde_max_doublings,
                mde_search_tolerance=con.mde_search_tolerance,
                crossing_rate_resamples=con.crossing_rate_resamples,
                band_resamples=con.band_resamples,
                neutral_permutation_reps=con.neutral_permutation_reps,
                lcb_bootstrap_iterations=con.lcb_bootstrap_iterations,
                description=con.description)
            self.assertEqual(rebuilt.content_hash(), con.content_hash())
            self.assertEqual(cid, con.construction_id)

    def test_an_out_of_range_betting_cap_is_refused(self):
        with self.assertRaises(st.ConstructionNotImplemented):
            st.EProcessConstruction(
                construction_id="x/v1", statistic="paired_block_effect_sign",
                betting_form="predictable_grow_approximation", lambda_cap=1.0,
                lambda_init=0.1, lambda_fixed=None, mde_power_target=0.8,
                mde_resamples=10, mde_max_doublings=4, mde_search_tolerance=0.01,
                crossing_rate_resamples=10, band_resamples=10,
                neutral_permutation_reps=10, lcb_bootstrap_iterations=10, description="x")

    def test_an_unimplemented_statistic_is_refused(self):
        with self.assertRaises(st.ConstructionNotImplemented):
            st.EProcessConstruction(
                construction_id="lcb/v1", statistic="lower_confidence_bound",
                betting_form="predictable_grow_approximation", lambda_cap=0.5,
                lambda_init=0.1, lambda_fixed=None, mde_power_target=0.8,
                mde_resamples=10, mde_max_doublings=4, mde_search_tolerance=0.01,
                crossing_rate_resamples=10, band_resamples=10,
                neutral_permutation_reps=10, lcb_bootstrap_iterations=10, description="x")


# =============================================================================
# The MDE
# =============================================================================

class TestMDE(unittest.TestCase):

    def setUp(self):
        (self.inputs, self.solve, self.cal, self.rule, _c, _sp, self.campaign,
         self.controls) = _Campaign.get()
        self.reducer = st.PairedBlockReducer(self.campaign)

    def test_mde_does_not_depend_on_the_candidates_data(self):
        """"published WITH the result, not after seeing it" — as a property."""
        request = make_request(calibration=self.cal, controls=self.controls)
        _s1, blocks_a = run_candidate(self.campaign, effect=0.15, seed=21)
        reduction_a = self.reducer.reduce(request, blocks_a)
        # A second candidate with a completely different effect, reduced by a
        # FRESH reducer so no cache can be responsible for the equality.
        fresh = st.PairedBlockReducer(self.campaign)
        blocks_b = tuple(
            st.PairedBlock(block_index=b.block_index, unit_id=b.unit_id,
                           stratum=b.stratum, order=b.order,
                           anchor_samples=b.anchor_samples,
                           candidate_samples=tuple(v * 3.0 for v in b.candidate_samples),
                           segment=b.segment, extension_round=b.extension_round,
                           measured_at=b.measured_at)
            for b in blocks_a)
        reduction_b = fresh.reduce(request, blocks_b)
        self.assertNotAlmostEqual(reduction_a.median_effect, reduction_b.median_effect)
        self.assertEqual(reduction_a.mde.value, reduction_b.mde.value)

    def test_mde_shrinks_as_blocks_grow(self):
        oriented = self.campaign.aa_oriented(HIGHER)
        threshold = self.campaign.threshold_for(api.STRATUM_SELECTION)
        small = st.solve_mde(oriented, block_count=5, rule=self.rule,
                             construction=self.campaign.construction,
                             hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                             threshold=threshold, campaign_seed=CAMPAIGN_SEED)
        large = st.solve_mde(oriented, block_count=14, rule=self.rule,
                             construction=self.campaign.construction,
                             hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                             threshold=threshold, campaign_seed=CAMPAIGN_SEED)
        self.assertTrue(small.found and large.found)
        self.assertLess(large.value, small.value)

    def test_the_returned_mde_has_its_power_measured_not_extrapolated(self):
        oriented = self.campaign.aa_oriented(HIGHER)
        mde = st.solve_mde(oriented, block_count=self.campaign.b_min, rule=self.rule,
                           construction=self.campaign.construction,
                           hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                           threshold=self.campaign.threshold_for(api.STRATUM_SELECTION),
                           campaign_seed=CAMPAIGN_SEED)
        self.assertTrue(mde.found)
        self.assertGreaterEqual(mde.achieved_power, mde.power_target)
        self.assertEqual(mde.method, "common_random_number_resampling")

    def test_an_undetectable_cell_says_so_instead_of_returning_a_number(self):
        tight = st.select_construction("sign_martingale_fixed_lambda/v1")
        rule = make_rule(max_rounds=0, blocks_per_round=1, ceiling=20)
        dispersed = [(-1.0) ** i * 0.01 * (1 + i % 7) for i in range(40)]
        # A threshold no 3-block window can reach at ANY shift: three wins at
        # lambda=0.25 cannot produce an e-value of 1e6. Report not-found with a
        # reason; never return a number that was never measured to work.
        mde = st.solve_mde(dispersed, block_count=3, rule=rule, construction=tight,
                           hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                           threshold=1e6, campaign_seed=CAMPAIGN_SEED)
        self.assertFalse(mde.found)
        self.assertIsNotNone(mde.reason)
        self.assertEqual(mde.achieved_power, 0.0)

    def test_zero_dispersion_material_refuses(self):
        with self.assertRaises(st.InsufficientMaterial):
            st.solve_mde([0.0] * 40, block_count=5, rule=self.rule,
                         construction=self.campaign.construction,
                         hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                         threshold=10.0, campaign_seed=CAMPAIGN_SEED)


# =============================================================================
# Crossing rates — the rule's own realized error rate
# =============================================================================

class TestCrossingRates(unittest.TestCase):

    def setUp(self):
        (_i, _s, self.cal, self.rule, _c, _sp, self.campaign, self.controls) = _Campaign.get()
        self.oriented = self.campaign.aa_oriented(HIGHER)
        self.threshold = self.campaign.threshold_for(api.STRATUM_SELECTION)

    def test_the_full_rule_including_extension_holds_the_budget_on_aa(self):
        rate = st.resampled_crossing_rate(
            self.oriented, block_count=self.campaign.b_min, rule=self.rule,
            construction=self.campaign.construction,
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, threshold=self.threshold,
            campaign_seed=CAMPAIGN_SEED)
        self.assertLessEqual(rate.rate, self.cal.alpha_sel)
        self.assertEqual(rate.window_length,
                         self.rule.max_total_blocks(self.campaign.b_min))
        self.assertEqual(rate.method, "resampled_aa_windows")

    def test_a_positive_shift_raises_the_crossing_rate(self):
        null = st.resampled_crossing_rate(
            self.oriented, block_count=self.campaign.b_min, rule=self.rule,
            construction=self.campaign.construction,
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, threshold=self.threshold,
            campaign_seed=CAMPAIGN_SEED)
        shifted = st.resampled_crossing_rate(
            self.oriented, block_count=self.campaign.b_min, rule=self.rule,
            construction=self.campaign.construction,
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, threshold=self.threshold,
            campaign_seed=CAMPAIGN_SEED, shift=0.30)
        self.assertGreater(shifted.rate, null.rate)

    def test_the_empirical_replay_reports_how_many_windows_it_had(self):
        rate = st.empirical_crossing_rate(
            self.oriented, block_count=self.campaign.b_min, rule=self.rule,
            construction=self.campaign.construction,
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, threshold=self.threshold)
        self.assertEqual(rate.method, "disjoint_aa_windows")
        self.assertEqual(rate.windows,
                         len(self.oriented) // self.rule.max_total_blocks(
                             self.campaign.b_min))
        self.assertAlmostEqual(rate.resolution, 1.0 / rate.windows)
        self.assertIsNone(rate.seed)

    def test_material_thinner_than_one_window_refuses(self):
        with self.assertRaises(st.InsufficientMaterial):
            st.empirical_crossing_rate(
                self.oriented[:3], block_count=self.campaign.b_min, rule=self.rule,
                construction=self.campaign.construction,
                hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                threshold=self.threshold)


# =============================================================================
# The anchor gate
# =============================================================================

class TestAnchorGate(unittest.TestCase):

    def setUp(self):
        (_i, self.solve, self.cal, _r, _c, _sp, self.campaign,
         _ctl) = _Campaign.get()

    def test_an_in_band_anchor_passes(self):
        centre = st.median(self.solve.anchor_calibration_values)
        chk = st.anchor_gate_check([centre] * self.cal.b_min_blocks,
                                   band=self.cal.anchor_gate_band,
                                   b_min=self.cal.b_min_blocks)
        self.assertEqual(chk.outcome, S.PASS)

    def test_a_drifted_anchor_voids_and_is_not_a_candidate_failure(self):
        low, high = self.cal.anchor_gate_band
        chk = st.anchor_gate_check([high * 1.5] * self.cal.b_min_blocks,
                                   band=self.cal.anchor_gate_band,
                                   b_min=self.cal.b_min_blocks)
        self.assertEqual(chk.outcome, S.FAIL)
        joined = " ".join(chk.reasons)
        self.assertIn("VOID", joined)
        self.assertIn("NOT a candidate", joined)
        self.assertLess(low, high)

    def test_an_unmeasured_anchor_is_could_not_check_never_pass(self):
        self.assertEqual(
            st.anchor_gate_check([], band=self.cal.anchor_gate_band,
                                 b_min=self.cal.b_min_blocks).outcome, S.COULD_NOT_CHECK)
        self.assertEqual(
            st.anchor_gate_check([100.0], band=None,
                                 b_min=self.cal.b_min_blocks).outcome, S.COULD_NOT_CHECK)

    def test_a_gate_at_a_different_reduction_size_is_a_different_gate(self):
        centre = st.median(self.solve.anchor_calibration_values)
        chk = st.anchor_gate_check([centre], band=self.cal.anchor_gate_band,
                                   b_min=self.cal.b_min_blocks)
        self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)

    def test_degenerate_anchor_material_refuses_a_zero_width_band(self):
        with self.assertRaises(st.InsufficientMaterial):
            st.anchor_gate_band((100.0,) * 40, b_min=5,
                                construction=self.campaign.construction,
                                campaign_seed=CAMPAIGN_SEED)


# =============================================================================
# Selection / confirmation split
# =============================================================================

class TestStrata(unittest.TestCase):

    def setUp(self):
        (_i, _s, _c, _r, _cm, self.split, self.campaign, _ctl) = _Campaign.get()

    def test_the_partition_is_deterministic_and_keyed_on_the_campaign_seed(self):
        other = st.StratumSplitRule(
            rule_id="split-1", campaign_seed="another-seed", confirmation_fraction=0.3,
            rotation=st.RotationSchedule(schedule_id="rot-1", period_campaigns=4))
        units = [f"shape-{i}" for i in range(400)]
        mine = self.split.partition(units)
        theirs = other.partition(units)
        self.assertEqual(mine, self.split.partition(units))
        self.assertNotEqual(mine, theirs)

    def test_the_partition_is_disjoint_and_covers_the_material(self):
        units = [f"shape-{i}" for i in range(500)]
        part = self.split.partition(units)
        sel = set(part[api.STRATUM_SELECTION])
        conf = set(part[api.STRATUM_CONFIRMATION])
        self.assertEqual(sel & conf, set())
        self.assertEqual(sel | conf, set(units))
        self.assertGreater(len(conf), 0)
        self.assertGreater(len(sel), 0)

    def test_the_confirmation_fraction_is_approximately_honoured(self):
        units = [f"shape-{i}" for i in range(2000)]
        conf = self.split.partition(units)[api.STRATUM_CONFIRMATION]
        self.assertAlmostEqual(len(conf) / 2000, 0.3, delta=0.05)

    def test_rotation_changes_the_partition_on_its_declared_schedule(self):
        rotation = st.RotationSchedule(schedule_id="rot-1", period_campaigns=4)
        early = st.StratumSplitRule(rule_id="split-1", campaign_seed=CAMPAIGN_SEED,
                                    confirmation_fraction=0.3, rotation=rotation,
                                    campaign_ordinal=0)
        same_epoch = st.StratumSplitRule(rule_id="split-1", campaign_seed=CAMPAIGN_SEED,
                                         confirmation_fraction=0.3, rotation=rotation,
                                         campaign_ordinal=3)
        next_epoch = st.StratumSplitRule(rule_id="split-1", campaign_seed=CAMPAIGN_SEED,
                                         confirmation_fraction=0.3, rotation=rotation,
                                         campaign_ordinal=4)
        units = [f"shape-{i}" for i in range(300)]
        self.assertEqual(early.partition(units), same_epoch.partition(units))
        self.assertNotEqual(early.partition(units), next_epoch.partition(units))

    def test_a_record_mixing_strata_fails(self):
        mixed = (
            st.PairedBlock(block_index=0, unit_id=selection_unit(self.split, 0),
                           stratum=api.STRATUM_SELECTION, order=st.ORDER_ANCHOR_FIRST,
                           anchor_samples=(1.0,), candidate_samples=(1.0,)),
            st.PairedBlock(block_index=1, unit_id=confirmation_unit(self.split, 1),
                           stratum=api.STRATUM_CONFIRMATION,
                           order=st.ORDER_CANDIDATE_FIRST,
                           anchor_samples=(1.0,), candidate_samples=(1.0,)),
        )
        chk = self.split.check_blocks(mixed)
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("mixing strata is INVALID" in r for r in chk.reasons))

    def test_a_mislabelled_block_fails_against_the_recorded_rule(self):
        unit = confirmation_unit(self.split, 5)
        block = st.PairedBlock(block_index=0, unit_id=unit,
                               stratum=api.STRATUM_SELECTION,
                               order=st.ORDER_ANCHOR_FIRST, anchor_samples=(1.0,),
                               candidate_samples=(1.0,))
        self.assertEqual(self.split.check_blocks((block,)).outcome, S.FAIL)

    def test_confirmation_material_may_not_appear_in_planner_context(self):
        conf = confirmation_unit(self.split, 9)
        sel = selection_unit(self.split, 9)
        self.assertEqual(self.split.check_planner_context([sel]).outcome, S.PASS)
        self.assertEqual(self.split.check_planner_context([sel, conf]).outcome, S.FAIL)

    def test_a_proposal_targeting_a_confirmation_shape_is_rejected(self):
        conf = confirmation_unit(self.split, 11)
        chk = self.split.check_proposal_targets([conf])
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("BEFORE it consumes a window" in r for r in chk.reasons))

    def test_selection_evidence_may_not_report_readiness(self):
        blocks = (st.PairedBlock(block_index=0, unit_id=selection_unit(self.split, 0),
                                 stratum=api.STRATUM_SELECTION,
                                 order=st.ORDER_ANCHOR_FIRST, anchor_samples=(1.0,),
                                 candidate_samples=(1.0,), measured_at=NOW),)
        chk = self.split.check_confirmation_admissible(blocks, lineage_entry_at=NOW)
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("upward-biased" in r for r in chk.reasons))

    def test_confirmation_evidence_before_lineage_entry_is_refused(self):
        early = st.PairedBlock(block_index=0, unit_id=confirmation_unit(self.split, 0),
                               stratum=api.STRATUM_CONFIRMATION,
                               order=st.ORDER_ANCHOR_FIRST, anchor_samples=(1.0,),
                               candidate_samples=(1.0,),
                               measured_at="2026-08-01T00:00:00+00:00")
        late = st.PairedBlock(block_index=1, unit_id=confirmation_unit(self.split, 1),
                              stratum=api.STRATUM_CONFIRMATION,
                              order=st.ORDER_CANDIDATE_FIRST, anchor_samples=(1.0,),
                              candidate_samples=(1.0,),
                              measured_at="2026-08-03T00:00:00+00:00")
        entry = "2026-08-02T00:00:00+00:00"
        self.assertEqual(
            self.split.check_confirmation_admissible((early,),
                                                     lineage_entry_at=entry).outcome, S.FAIL)
        self.assertEqual(
            self.split.check_confirmation_admissible((late,),
                                                     lineage_entry_at=entry).outcome, S.PASS)

    def test_untimestamped_blocks_are_could_not_check_never_pass(self):
        block = st.PairedBlock(block_index=0, unit_id=confirmation_unit(self.split, 0),
                               stratum=api.STRATUM_CONFIRMATION,
                               order=st.ORDER_ANCHOR_FIRST, anchor_samples=(1.0,),
                               candidate_samples=(1.0,))
        self.assertEqual(
            self.split.check_confirmation_admissible(
                (block,), lineage_entry_at=NOW).outcome, S.COULD_NOT_CHECK)
        self.assertEqual(
            self.split.check_confirmation_admissible(
                (block,), lineage_entry_at=None).outcome, S.COULD_NOT_CHECK)

    def test_a_degenerate_split_fraction_is_refused(self):
        rotation = st.RotationSchedule(schedule_id="rot-1", period_campaigns=4)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(st.MaterialError):
                st.StratumSplitRule(rule_id="s", campaign_seed=CAMPAIGN_SEED,
                                    confirmation_fraction=bad, rotation=rotation)

    def test_the_rotation_schedule_lives_in_the_bundle(self):
        with self.assertRaises(st.MaterialError):
            st.RotationSchedule(schedule_id="rot", period_campaigns=4,
                                declared_in="campaign_manifest")


# =============================================================================
# The reducer
# =============================================================================

class TestReducer(unittest.TestCase):

    def setUp(self):
        (_i, self.solve, self.cal, self.rule, self.commitment, self.split,
         self.campaign, self.controls) = _Campaign.get()
        self.reducer = st.PairedBlockReducer(self.campaign)
        self.request = make_request(calibration=self.cal, controls=self.controls)

    def test_a_conforming_run_produces_a_complete_effect_estimate(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        estimate = self.reducer.reduce_blocks(self.request, blocks)
        self.assertIsInstance(estimate, api.EffectEstimate)
        self.assertEqual(estimate.paired_blocks, len(blocks))
        self.assertEqual(estimate.threshold,
                         self.cal.threshold_for(api.STRATUM_SELECTION))
        self.assertEqual(estimate.noise_floor, self.cal.noise_floor_phi)
        self.assertGreater(estimate.mde, 0.0)
        self.assertEqual(api._resolve_effect(estimate), api.EFFECT_IMPROVEMENT)

    def test_the_estimate_carries_the_e_value_its_threshold_the_mde_and_the_floor(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        estimate = self.reducer.reduce_blocks(self.request, blocks)
        as_dict = estimate.to_dict()
        for key in ("e_value", "threshold", "mde", "noise_floor", "paired_blocks",
                    "stratum", "raw_samples_ref"):
            self.assertIn(key, as_dict)
        self.assertEqual(as_dict["lcb_label"], "descriptive")

    def test_a_below_floor_effect_is_not_a_small_win(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.0, noise=0.0005, seed=44)
        reduction = self.reducer.reduce(self.request, blocks)
        if reduction.estimate is not None:
            self.assertIn(api._resolve_effect(reduction.estimate),
                          (api.EFFECT_BELOW_NOISE_FLOOR,
                           api.EFFECT_NO_DETECTABLE_DIFFERENCE,
                           api.EFFECT_EVIDENCE_BELOW_THRESHOLD))

    def test_a_regression_is_signed_correctly_for_a_lower_better_metric(self):
        request = make_request(calibration=self.cal, controls=self.controls,
                               direction=LOWER)
        _seq, blocks = run_candidate(self.campaign, effect=-0.15, seed=52,
                                     direction=LOWER)
        reduction = self.reducer.reduce(request, blocks)
        self.assertLess(reduction.median_effect, 0.0)
        for oriented, signed in zip(reduction.oriented_effects, reduction.block_effects):
            self.assertAlmostEqual(oriented, -signed)

    def test_too_few_blocks_is_refused_not_reported(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        short = blocks[:self.campaign.b_min - 1]
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(self.request, short)
        self.assertIn("below the calibrated B_min", str(cm.exception))
        self.assertIsInstance(cm.exception.reduction, st.BlockReduction)

    def test_inadmissible_reduces_to_an_exception_never_to_none(self):
        """`None` would make api.TierDispatcher skip the rate-only voids."""
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        mixed = blocks[:-1] + (st.PairedBlock(
            block_index=blocks[-1].block_index,
            unit_id=confirmation_unit(self.split, 77),
            stratum=api.STRATUM_CONFIRMATION, order=blocks[-1].order,
            anchor_samples=blocks[-1].anchor_samples,
            candidate_samples=blocks[-1].candidate_samples,
            segment=blocks[-1].segment, extension_round=blocks[-1].extension_round),)
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(self.request, mixed)
        reduction = cm.exception.reduction
        self.assertIsNone(reduction.estimate)
        self.assertEqual(reduction.check("stratum_partition").outcome, S.FAIL)
        self.assertEqual(reduction.window_checks["strata"].outcome, S.FAIL)

    def test_a_blocked_design_is_refused(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        blocked = tuple(
            st.PairedBlock(block_index=b.block_index, unit_id=b.unit_id,
                           stratum=b.stratum, order=st.ORDER_ANCHOR_FIRST,
                           anchor_samples=b.anchor_samples,
                           candidate_samples=b.candidate_samples, segment=b.segment,
                           extension_round=b.extension_round)
            for b in blocks)
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(self.request, blocked)
        self.assertEqual(cm.exception.reduction.check("order_control").outcome, S.FAIL)

    def test_a_calibration_from_a_different_cell_is_refused(self):
        request = make_request(calibration=self.cal, controls=self.controls,
                               phase="prefill")
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(request, blocks)
        self.assertEqual(cm.exception.reduction.check("calibration_cell").outcome, S.FAIL)

    def test_more_extension_rounds_than_declared_are_refused(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.0, seed=7)
        extra = blocks + tuple(
            st.PairedBlock(block_index=len(blocks) + i,
                           unit_id=selection_unit(self.split, 200 + i),
                           stratum=api.STRATUM_SELECTION,
                           order=self.campaign.order_schedule(
                               self.request.candidate_id).order_for(len(blocks) + i),
                           anchor_samples=(100.0,), candidate_samples=(101.0,),
                           segment=st.SEGMENT_EXTENSION, extension_round=2)
            for i in range(5))
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(self.request, extra)
        joined = " ".join(r for _n, chk in cm.exception.reduction.checks
                          for r in chk.reasons)
        self.assertIn("declared", joined)

    def test_zero_blocks_is_refused_rather_than_returning_none(self):
        with self.assertRaises(st.InsufficientMaterial):
            self.reducer.reduce(self.request, ())

    def test_the_reduction_recomputes_from_the_records_own_raw_samples(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        estimate = self.reducer.reduce_blocks(self.request, blocks)
        chk = st.verify_reduction_reproducible(estimate, self.reducer, self.request)
        self.assertEqual(chk.outcome, S.PASS)

    def test_tampered_raw_samples_fail_the_reproducibility_check(self):
        """A robust reducer alone would not catch this; the content hash does.

        Changing ONE sample can leave the median, the sign-based e-value and the
        MDE identical — that is what robustness means. The record's raw-samples
        reference is content-addressed, so the edit is still caught.
        """
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        estimate = self.reducer.reduce_blocks(self.request, blocks)
        tampered_raw = list(estimate.raw_samples)
        first = list(tampered_raw[0])
        first[8] = tuple(v * 4.0 for v in first[8])       # inflate the candidate arm
        tampered_raw[0] = tuple(first)
        tampered = api.EffectEstimate(
            metric=estimate.metric, metric_direction=estimate.metric_direction,
            value=estimate.value, e_value=estimate.e_value, threshold=estimate.threshold,
            mde=estimate.mde, noise_floor=estimate.noise_floor,
            paired_blocks=estimate.paired_blocks, stratum=estimate.stratum,
            raw_samples=tuple(tampered_raw), raw_samples_ref=estimate.raw_samples_ref,
            lcb_descriptive=estimate.lcb_descriptive)
        chk = st.verify_reduction_reproducible(tampered, self.reducer, self.request)
        self.assertEqual(chk.outcome, S.FAIL)

    def test_raw_samples_ref_is_content_addressed_by_default(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        estimate = self.reducer.reduce_blocks(self.request, blocks)
        expected = "sha256:" + S.content_hash([b.to_list() for b in blocks])
        self.assertEqual(estimate.raw_samples_ref, expected)

    def test_the_reported_e_value_is_the_running_maximum(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        reduction = self.reducer.reduce(self.request, blocks)
        self.assertAlmostEqual(reduction.estimate.e_value,
                               reduction.e_process.e_running_max)
        self.assertGreaterEqual(reduction.e_process.log_e_running_max,
                                reduction.e_process.log_e_final)

    def test_the_extension_pools_to_the_same_pre_declared_threshold(self):
        """*"pooled to a pre-declared threshold"* — the threshold never moves."""
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        self.assertGreater(len(blocks), self.campaign.b_min)
        base_only = self.reducer.reduce(self.request, blocks[:self.campaign.b_min])
        pooled = self.reducer.reduce(self.request, blocks)
        self.assertEqual(base_only.threshold, pooled.threshold)
        self.assertEqual(pooled.threshold,
                         self.cal.threshold_for(api.STRATUM_SELECTION))

    def test_the_confirmation_stratum_is_judged_at_a_tighter_threshold(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31,
                                     stratum=api.STRATUM_CONFIRMATION)
        reduction = self.reducer.reduce(self.request, blocks)
        self.assertEqual(reduction.threshold,
                         self.cal.threshold_for(api.STRATUM_CONFIRMATION))
        self.assertGreater(reduction.threshold,
                           self.cal.threshold_for(api.STRATUM_SELECTION))

    def test_the_reducer_satisfies_the_api_effect_reducer_seam(self):
        self.assertTrue(hasattr(self.reducer, "construction_id"))
        self.assertTrue(callable(self.reducer.reduce_blocks))
        self.assertEqual(self.reducer.construction_id,
                         self.cal.e_process_construction_id)


class TestCampaignStatisticsGuards(unittest.TestCase):

    def setUp(self):
        (_i, self.solve, self.cal, self.rule, self.commitment, self.split,
         self.campaign, self.controls) = _Campaign.get()

    def _build(self, **over):
        kwargs = dict(
            campaign_id="ak-1", campaign_seed=CAMPAIGN_SEED, effect_scale=SCALE,
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
            stopping_rule=self.rule, stopping_rule_commitment=self.commitment,
            split_rule=self.split, construction=self.campaign.construction,
            calibration=self.cal, aa_effect_pool=self.campaign.aa_effect_pool,
            anchor_calibration_values=self.campaign.anchor_calibration_values)
        kwargs.update(over)
        return st.CampaignStatistics(**kwargs)

    def test_a_mutated_rule_cannot_start_a_campaign(self):
        with self.assertRaises(st.StoppingRuleMutated):
            self._build(stopping_rule=make_rule(max_rounds=4))

    def test_a_construction_the_calibration_did_not_record_is_refused(self):
        other = st.select_construction("sign_martingale_fixed_lambda/v1")
        with self.assertRaises(st.ConstructionNotImplemented):
            self._build(construction=other)

    def test_a_split_rule_keyed_on_another_seed_is_refused(self):
        other = st.StratumSplitRule(
            rule_id="split-1", campaign_seed="another", confirmation_fraction=0.3,
            rotation=st.RotationSchedule(schedule_id="rot-1", period_campaigns=4))
        with self.assertRaises(st.MaterialError):
            self._build(split_rule=other)

    def test_an_unaccepted_calibration_cannot_rank_a_candidate(self):
        unaccepted = api.CalibrationOutputs(
            backend=self.cal.backend, phase=self.cal.phase, cell_class=self.cal.cell_class,
            noise_floor_phi=self.cal.noise_floor_phi, b_min_blocks=self.cal.b_min_blocks,
            alpha_sel=self.cal.alpha_sel, alpha_conf=self.cal.alpha_conf,
            anchor_gate_band=self.cal.anchor_gate_band, accepted=False,
            solve_order_recorded=api.CALIBRATION_SOLVE_ORDER,
            samples_ref=self.cal.samples_ref,
            e_process_construction_id=self.cal.e_process_construction_id)
        with self.assertRaises(st.CalibrationFailed):
            self._build(calibration=unaccepted)

    def test_an_empty_aa_pool_would_make_the_mde_uncomputable_and_is_refused(self):
        with self.assertRaises(st.MaterialError):
            self._build(aa_effect_pool=())

    def test_a_fixed_owning_rule_disagreeing_with_b_min_is_refused(self):
        fixed = st.OwningProtocolRepRule(
            protocol_id="P-BENCH-4", kind=st.REP_RULE_FIXED,
            blocks=self.cal.b_min_blocks + 1, citation="bench-cpu.md:174-178")
        with self.assertRaises(st.CalibrationFailed):
            self._build(owning_rep_rule=fixed)


# =============================================================================
# The descriptive LCB
# =============================================================================

class TestDescriptiveLCB(unittest.TestCase):

    def setUp(self):
        (_i, _s, self.cal, _r, _c, _sp, self.campaign, self.controls) = _Campaign.get()

    def test_the_lcb_is_labelled_descriptive_and_denies_being_a_test(self):
        lcb = st.descriptive_lcb([0.1, 0.12, 0.09, 0.11],
                                 campaign_seed=CAMPAIGN_SEED, candidate_id="akc-1",
                                 construction=self.campaign.construction)
        self.assertEqual(lcb.label, "descriptive")
        self.assertFalse(lcb.is_a_test)
        self.assertIn("MUST NOT rank", lcb.warning)
        self.assertIn("sequential", lcb.warning)

    def test_the_lcb_is_seeded_and_reproducible(self):
        args = dict(campaign_seed=CAMPAIGN_SEED, candidate_id="akc-1",
                    construction=self.campaign.construction)
        a = st.descriptive_lcb([0.1, 0.12, 0.09, 0.11], **args)
        b = st.descriptive_lcb([0.1, 0.12, 0.09, 0.11], **args)
        self.assertEqual(a.value, b.value)
        self.assertEqual(a.seed, b.seed)

    def test_no_resolution_in_the_module_reads_the_lcb(self):
        """The LCB is carried BESIDE the e-value and decides nothing."""
        reducer = st.PairedBlockReducer(self.campaign)
        request = make_request(calibration=self.cal, controls=self.controls)
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        estimate = reducer.reduce_blocks(request, blocks)
        with_lcb = api._resolve_effect(estimate)
        without_lcb = api._resolve_effect(api.EffectEstimate(
            metric=estimate.metric, metric_direction=estimate.metric_direction,
            value=estimate.value, e_value=estimate.e_value, threshold=estimate.threshold,
            mde=estimate.mde, noise_floor=estimate.noise_floor,
            paired_blocks=estimate.paired_blocks, stratum=estimate.stratum,
            raw_samples=estimate.raw_samples, raw_samples_ref=estimate.raw_samples_ref,
            lcb_descriptive=None))
        self.assertEqual(with_lcb, without_lcb)


# =============================================================================
# Determinism, provenance and the no-write property
# =============================================================================

class TestDeterminism(unittest.TestCase):

    def test_seed_derivation_is_stable_and_purpose_separated(self):
        a = st.derive_seed(CAMPAIGN_SEED, "mde", 5)
        b = st.derive_seed(CAMPAIGN_SEED, "mde", 5)
        c = st.derive_seed(CAMPAIGN_SEED, "order", 5)
        d = st.derive_seed("other", "mde", 5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_an_empty_campaign_seed_is_refused(self):
        for bad in ("", "   ", None, 17):
            with self.assertRaises(st.MaterialError):
                st.derive_seed(bad, "mde")

    def test_the_reduction_is_bit_identical_across_reducers(self):
        (_i, _s, cal, _r, _c, _sp, campaign, controls) = _Campaign.get()
        request = make_request(calibration=cal, controls=controls)
        _seq, blocks = run_candidate(campaign, effect=0.15, seed=31)
        a = st.PairedBlockReducer(campaign).reduce(request, blocks)
        b = st.PairedBlockReducer(campaign).reduce(request, blocks)
        self.assertEqual(a.to_dict(), b.to_dict())
        S.canonical_json(a.to_dict())


class TestNoWriteOrProcessPaths(unittest.TestCase):

    def test_statistics_module_cannot_write_or_signal(self):
        source = Path(st.__file__).read_text(encoding="utf-8")
        chk = api.audit_no_write_or_process_paths(source, module_id=st.MODULE_ID)
        self.assertEqual(chk.outcome, S.PASS, f"audit findings: {list(chk.reasons)}")

    def test_the_audit_would_notice_a_write_path(self):
        chk = api.audit_no_write_or_process_paths(
            "import os\ndef f(p):\n    p.write_text('x')\n")
        self.assertEqual(chk.outcome, S.FAIL)

    def test_stdlib_statistics_is_not_imported(self):
        source = Path(st.__file__).read_text(encoding="utf-8")
        self.assertNotIn("\nimport statistics", source)
        self.assertNotIn("\nfrom statistics import", source)


class TestModuleSurface(unittest.TestCase):

    def test_every_exported_name_exists(self):
        missing = [name for name in st.__all__ if not hasattr(st, name)]
        self.assertEqual(missing, [])

    def test_errors_are_evaluator_errors_so_one_except_catches_the_family(self):
        for name in ("StatisticsError", "CalibrationFailed", "InsufficientMaterial",
                     "StoppingRuleMutated", "StoppingRuleViolation",
                     "ReductionInadmissible", "ConstructionNotImplemented",
                     "EValueNotRepresentable", "MaterialError"):
            self.assertTrue(issubclass(getattr(st, name), api.EvaluatorError), name)

    def test_the_authorized_decision_set_is_the_protocols_own(self):
        for decision in ("rank_against_anchor", "retain", "abandon", "branch",
                         "compose_into_champion_lineage", "select_next_experiment",
                         "request_readiness_computation"):
            self.assertIn(decision, st.AUTHORIZED_DECISIONS)
        for forbidden in ("promote", "deploy", "freeze", "cutover", "buy", "close"):
            self.assertNotIn(forbidden, st.AUTHORIZED_DECISIONS)


# =============================================================================
# Red-team regressions (2026-08-03)
#
# Every test below reproduces a defect that was PRESENT and SILENT in the first
# version of `statistics.py`, and each one failed before its fix. They are kept
# together because they share a shape: in every case the module already carried
# prose asserting the property, and a check that appeared to enforce it could be
# satisfied by removing the thing it inspected.
# =============================================================================

class TestEProcessTiesAtTheNullBoundary(unittest.TestCase):
    """The supermartingale must hold under the null the module itself STATES.

    Before the fix `X_b` was `sign(delta)` in {-1, 0, +1}, giving
    `E[X_b] = 2*P(> boundary) + P(= boundary) - 1`, which is POSITIVE whenever
    the boundary carries an atom and `P(> boundary) = 1/2`. That distribution is
    squarely inside the docstring's own null, so the wealth was a SUBmartingale
    and Ville's bound did not hold. The existing supermartingale test could not
    see it: its fixture draws `rng.gauss(0, 0.01)`, a continuous law with no
    atom anywhere, so it never generated the case that breaks the proof.
    """

    def setUp(self):
        self.construction = st.select_construction("sign_martingale_predictable_lambda/v1")

    def _crossing_rate(self, *, p_tie, alpha, blocks, trials=4000, seed=20260803):
        rng = random.Random(seed)
        crossings = 0
        for _ in range(trials):
            oriented = []
            for _ in range(blocks):
                u = rng.random()
                # P(> 0) = 1/2 exactly: the boundary of the stated null.
                oriented.append(1.0 if u < 0.5 else (0.0 if u < 0.5 + p_tie else -1.0))
            if st.run_e_process(oriented, construction=self.construction,
                                hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                                threshold=1.0 / alpha).crossed:
                crossings += 1
        return crossings / trials

    def test_a_tie_heavy_null_does_not_exceed_alpha(self):
        for p_tie in (0.0, 0.2, 0.4, 0.6):
            for alpha, blocks in ((0.10, 20), (0.05, 30)):
                rate = self._crossing_rate(p_tie=p_tie, alpha=alpha, blocks=blocks)
                self.assertLessEqual(
                    rate, alpha,
                    f"tie rate {p_tie}: {rate} crossings exceeds alpha={alpha} at "
                    f"{blocks} blocks — the wealth is not a supermartingale under the "
                    "null this construction states")

    def test_a_block_exactly_on_the_boundary_scores_against_the_candidate(self):
        """The tie rule is visible in the recorded signs, not just in the total."""
        run = st.run_e_process([0.0, 0.0, 0.0], construction=self.construction,
                               hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                               threshold=10.0)
        self.assertEqual(list(run.signs), [-1.0, -1.0, -1.0])
        self.assertLess(run.log_e_running_max, 0.0)
        self.assertFalse(run.crossed)

    def test_ties_cannot_manufacture_wealth_against_an_identical_candidate(self):
        """A candidate byte-identical to the anchor is the tie-generating case."""
        run = st.run_e_process([0.0] * 40, construction=self.construction,
                               hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                               threshold=10.0)
        self.assertLessEqual(run.e_running_max, 1.0)

    def test_a_non_inferiority_margin_ties_the_same_way(self):
        run = st.run_e_process([-0.02] * 12, construction=self.construction,
                               hypothesis=st.HYPOTHESIS_NON_INFERIORITY, margin=0.02,
                               threshold=10.0)
        self.assertEqual(set(run.signs), {-1.0})
        self.assertFalse(run.crossed)


class TestTheConstructionCannotBeTuned(unittest.TestCase):
    """*"a campaign selects among constructions the bundle already implements."*

    `CalibrationInputs` refused a tuned construction. `CampaignStatistics` and
    `SequentialEvaluation` — the two objects that decide what actually RUNS —
    accepted one, so a campaign could record
    `e_process_construction_id="sign_martingale_predictable_lambda/v1"` in its
    manifest while betting at `lambda_cap=0.99` and deriving its MDE at 5% power.
    Measured on identical blocks that inflated the e-value 23x and shrank the
    MDE 3x, and both reductions were admissible.
    """

    def setUp(self):
        (_i, self.solve, self.cal, self.rule, self.commitment, self.split,
         self.campaign, self.controls) = _Campaign.get()
        real = st.select_construction("sign_martingale_predictable_lambda/v1")
        self.impostor = st.EProcessConstruction(
            construction_id=real.construction_id,          # the registry's OWN id
            statistic=real.statistic, betting_form=real.betting_form,
            lambda_cap=0.99, lambda_init=0.99, lambda_fixed=None,
            mde_power_target=0.05, mde_resamples=real.mde_resamples,
            mde_max_doublings=real.mde_max_doublings,
            mde_search_tolerance=real.mde_search_tolerance,
            crossing_rate_resamples=real.crossing_rate_resamples,
            band_resamples=real.band_resamples,
            neutral_permutation_reps=real.neutral_permutation_reps,
            lcb_bootstrap_iterations=real.lcb_bootstrap_iterations,
            description=real.description)

    def test_campaign_statistics_refuses_a_tuned_construction(self):
        with self.assertRaises(st.ConstructionNotImplemented) as cm:
            st.CampaignStatistics(
                campaign_id="ak-1", campaign_seed=CAMPAIGN_SEED, effect_scale=SCALE,
                hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                stopping_rule=self.rule, stopping_rule_commitment=self.commitment,
                split_rule=self.split, construction=self.impostor, calibration=self.cal,
                aa_effect_pool=self.solve.aa_effect_pool,
                anchor_calibration_values=self.solve.anchor_calibration_values)
        self.assertIn("fixed at the bundle hash", str(cm.exception))

    def test_sequential_evaluation_refuses_a_tuned_construction(self):
        with self.assertRaises(st.ConstructionNotImplemented):
            st.SequentialEvaluation(
                rule=self.rule, commitment=self.commitment, construction=self.impostor,
                b_min=self.campaign.b_min,
                threshold=self.campaign.threshold_for(api.STRATUM_SELECTION),
                hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                metric_direction=HIGHER, effect_scale=SCALE,
                order_schedule=self.campaign.order_schedule("akc-0001"))

    def test_an_unregistered_id_is_refused_too(self):
        stranger = st.EProcessConstruction(
            construction_id="sign_martingale_hand_rolled/v1",
            statistic=self.impostor.statistic, betting_form=self.impostor.betting_form,
            lambda_cap=0.5, lambda_init=0.1, lambda_fixed=None, mde_power_target=0.8,
            mde_resamples=10, mde_max_doublings=4, mde_search_tolerance=0.01,
            crossing_rate_resamples=10, band_resamples=10, neutral_permutation_reps=10,
            lcb_bootstrap_iterations=10, description="x")
        with self.assertRaises(st.ConstructionNotImplemented):
            st.CampaignStatistics(
                campaign_id="ak-1", campaign_seed=CAMPAIGN_SEED, effect_scale=SCALE,
                hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                stopping_rule=self.rule, stopping_rule_commitment=self.commitment,
                split_rule=self.split, construction=stranger, calibration=self.cal,
                aa_effect_pool=self.solve.aa_effect_pool,
                anchor_calibration_values=self.solve.anchor_calibration_values)

    def test_a_faithfully_rebuilt_construction_is_still_admitted(self):
        """The guard is on CONTENT, not on object identity — a deserialized
        construction that hashes to the bundle's is the bundle's."""
        real = st.select_construction("sign_martingale_predictable_lambda/v1")
        rebuilt = st.EProcessConstruction(**{
            k: getattr(real, k) for k in
            ("construction_id", "statistic", "betting_form", "lambda_cap", "lambda_init",
             "lambda_fixed", "mde_power_target", "mde_resamples", "mde_max_doublings",
             "mde_search_tolerance", "crossing_rate_resamples", "band_resamples",
             "neutral_permutation_reps", "lcb_bootstrap_iterations", "description")})
        self.assertIsNot(rebuilt, real)
        campaign = st.CampaignStatistics(
            campaign_id="ak-1", campaign_seed=CAMPAIGN_SEED, effect_scale=SCALE,
            hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0, stopping_rule=self.rule,
            stopping_rule_commitment=self.commitment, split_rule=self.split,
            construction=rebuilt, calibration=self.cal,
            aa_effect_pool=self.solve.aa_effect_pool,
            anchor_calibration_values=self.solve.anchor_calibration_values)
        self.assertEqual(campaign.construction.content_hash(), real.content_hash())


class TestTamperCheckCannotBeSkipped(unittest.TestCase):
    """Check 1 was conditional on the ref's PREFIX, so deleting the prefix
    deleted the check — and check 2 alone cannot see an edited sample, which the
    function's own docstring says. A tampered record then returned PASS."""

    def setUp(self):
        (_i, _s, self.cal, _r, _c, _sp, self.campaign, self.controls) = _Campaign.get()
        self.reducer = st.PairedBlockReducer(self.campaign)
        self.request = make_request(calibration=self.cal, controls=self.controls)

    def _tamper(self, estimate):
        raw = list(estimate.raw_samples)
        first = list(raw[0])
        first[8] = tuple(v * 4.0 for v in first[8])       # inflate the candidate arm
        raw[0] = tuple(first)
        return api.EffectEstimate(
            metric=estimate.metric, metric_direction=estimate.metric_direction,
            value=estimate.value, e_value=estimate.e_value, threshold=estimate.threshold,
            mde=estimate.mde, noise_floor=estimate.noise_floor,
            paired_blocks=estimate.paired_blocks, stratum=estimate.stratum,
            raw_samples=tuple(raw), raw_samples_ref=estimate.raw_samples_ref,
            lcb_descriptive=estimate.lcb_descriptive)

    def test_a_non_content_addressed_ref_is_never_a_pass(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        reduction = self.reducer.reduce(self.request, blocks,
                                        raw_samples_ref="ak-raw://journal/0001")
        chk = st.verify_reduction_reproducible(self._tamper(reduction.estimate),
                                               self.reducer, self.request)
        self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(any("content-addressed" in r for r in chk.reasons))

    def test_an_untampered_record_with_a_bare_ref_is_also_not_a_pass(self):
        """The third outcome is about the REF, not about the tampering: a record
        whose samples cannot be checked is not a record whose samples checked."""
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        reduction = self.reducer.reduce(self.request, blocks,
                                        raw_samples_ref="ak-raw://journal/0002")
        chk = st.verify_reduction_reproducible(reduction.estimate, self.reducer,
                                               self.request)
        self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)

    def test_the_content_addressed_path_still_passes_and_still_catches_tampering(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        estimate = self.reducer.reduce_blocks(self.request, blocks)
        self.assertEqual(
            st.verify_reduction_reproducible(estimate, self.reducer,
                                             self.request).outcome, S.PASS)
        self.assertEqual(
            st.verify_reduction_reproducible(self._tamper(estimate), self.reducer,
                                             self.request).outcome, S.FAIL)


class TestAnchorGateReductionSizeIsBound(unittest.TestCase):
    """*"a gate evaluated at a different reduction size is a different gate."*

    The guard compared the observation count against a `b_min` the CALLER
    supplied, and the bare `(low, high)` pair carries no size of its own — so
    declaring `b_min=1` gated a single unreduced anchor sample against a band
    bootstrapped over medians of B_min blocks, and returned PASS. The band now
    carries its own calibration size and a disagreeing caller is COULD_NOT_CHECK.
    """

    def setUp(self):
        (_i, self.solve, self.cal, _r, _c, _sp, self.campaign, _ctl) = _Campaign.get()
        self.band = st.anchor_gate_band(
            self.solve.anchor_calibration_values, b_min=self.cal.b_min_blocks,
            construction=self.campaign.construction, campaign_seed=CAMPAIGN_SEED)
        self.centre = st.median(self.solve.anchor_calibration_values)

    def test_a_band_object_refuses_a_smaller_declared_reduction_size(self):
        chk = st.anchor_gate_check([self.centre], band=self.band, b_min=1)
        self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(any("different gate" in r for r in chk.reasons))

    def test_the_calibration_outputs_bind_the_size_too(self):
        chk = st.anchor_gate_check([self.centre], band=self.cal, b_min=1)
        self.assertEqual(chk.outcome, S.COULD_NOT_CHECK)

    def test_the_bound_forms_still_gate_normally(self):
        low, high = self.band.as_tuple()
        n = self.cal.b_min_blocks
        self.assertEqual(
            st.anchor_gate_check([self.centre] * n, band=self.band, b_min=n).outcome,
            S.PASS)
        self.assertEqual(
            st.anchor_gate_check([high * 1.5] * n, band=self.cal, b_min=n).outcome,
            S.FAIL)
        self.assertLess(low, high)

    def test_a_bare_pair_says_the_size_was_caller_asserted(self):
        chk = st.anchor_gate_check([self.centre], band=self.cal.anchor_gate_band, b_min=1)
        self.assertTrue(any("CALLER-ASSERTED" in r for r in chk.reasons),
                        "an unbound reduction size must be visible in the record")

    def test_a_malformed_band_is_could_not_check_not_a_traceback(self):
        for bad in (("a", "b"), (float("nan"), float("nan")), (5.0, 5.0), (9.0, 1.0), 7):
            self.assertEqual(
                st.anchor_gate_check([self.centre] * 5, band=bad, b_min=5).outcome,
                S.COULD_NOT_CHECK, f"band={bad!r}")

    def test_a_nonsense_b_min_is_could_not_check(self):
        for bad in (0, -3, 2.5, True, None):
            self.assertEqual(
                st.anchor_gate_check([self.centre] * 5, band=self.band, b_min=bad).outcome,
                S.COULD_NOT_CHECK, f"b_min={bad!r}")


class TestLineageEntryIsAnInstantNotAString(unittest.TestCase):
    """Confirmation evidence was ordered against lineage entry by comparing the
    two timestamps AS STRINGS. Lexicographic order is not chronological order
    across two legal spellings: `"2026-8-2T00:00:00+00:00"` sorts above
    `"2026-08-02T12:00:00+00:00"`, so a block measured twelve hours BEFORE
    lineage entry passed the winner's-curse gate — and any non-timestamp string
    that happened to sort high passed it too."""

    def setUp(self):
        (_i, _s, _c, _r, _cm, self.split, _cp, _ctl) = _Campaign.get()

    def _block(self, measured_at, index=0):
        return st.PairedBlock(
            block_index=index, unit_id=confirmation_unit(self.split, index),
            stratum=api.STRATUM_CONFIRMATION, order=st.ORDER_ANCHOR_FIRST,
            anchor_samples=(1.0,), candidate_samples=(1.0,), measured_at=measured_at)

    def test_an_earlier_instant_that_sorts_later_as_a_string_is_refused(self):
        """The exact defect, stated twice: as a string the block is AFTER lineage
        entry, and as an instant it is five hours BEFORE it."""
        entry = "2026-08-02T12:00:00Z"
        measured = "2026-08-02T14:00:00+05:00"          # == 09:00Z
        self.assertGreater(measured, entry, "the string comparison says 'after'")
        chk = self.split.check_confirmation_admissible((self._block(measured),),
                                                       lineage_entry_at=entry)
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("part of what selected it" in r for r in chk.reasons))

    def test_a_z_suffixed_entry_orders_correctly_against_an_offset_block(self):
        entry = "2026-08-02T12:00:00Z"
        self.assertEqual(self.split.check_confirmation_admissible(
            (self._block("2026-08-02T08:00:00+00:00"),),
            lineage_entry_at=entry).outcome, S.FAIL)
        self.assertEqual(self.split.check_confirmation_admissible(
            (self._block("2026-08-02T16:00:00+00:00"),),
            lineage_entry_at=entry).outcome, S.PASS)

    def test_offsets_are_honoured_not_ignored(self):
        """13:00+02:00 is 11:00Z — BEFORE a 12:00Z entry, though it sorts after."""
        self.assertEqual(self.split.check_confirmation_admissible(
            (self._block("2026-08-02T13:00:00+02:00"),),
            lineage_entry_at="2026-08-02T12:00:00Z").outcome, S.FAIL)

    def test_an_unorderable_stamp_cannot_enter_the_record_at_all(self):
        """Defence in depth: the checker's COULD_NOT_CHECK branch stays, but a
        block carrying a stamp nothing can order is refused at construction, so
        it never reaches the journal in the first place."""
        for bad in ("zzz-not-a-timestamp", "9999", "", "2026-08-02T16:00:00"):
            with self.assertRaises(st.MaterialError, msg=f"measured_at={bad!r}"):
                self._block(bad)

    def test_an_unorderable_lineage_entry_is_could_not_check(self):
        for bad in ("later", "2026-08-02T12:00:00"):
            self.assertEqual(self.split.check_confirmation_admissible(
                (self._block(NOW),), lineage_entry_at=bad).outcome, S.COULD_NOT_CHECK)

    def test_an_elided_offender_list_states_the_total(self):
        blocks = tuple(self._block("2026-08-01T00:00:00+00:00", index=i)
                       for i in range(20))
        chk = self.split.check_confirmation_admissible(
            blocks, lineage_entry_at="2026-08-02T12:00:00+00:00")
        self.assertEqual(chk.outcome, S.FAIL)
        joined = " ".join(chk.reasons)
        self.assertIn("of 20", joined)
        self.assertIn("elided", joined)


class TestOneBlockIsNotBMinBlocks(unittest.TestCase):
    """`reduce_blocks` accepted the SAME measured block B_min times.

    Every "block" then carries the identical sign, the wealth grows at the
    betting cap every step, and a single measurement is reported as
    `blocks=B_min` paired blocks under order-randomized interleaving. Order
    control cannot see it — each position's order can be made to match the
    schedule — so independence has to be checked as block IDENTITY.
    """

    def setUp(self):
        (_i, _s, self.cal, self.rule, _c, self.split, self.campaign,
         self.controls) = _Campaign.get()
        self.reducer = st.PairedBlockReducer(self.campaign)
        self.request = make_request(calibration=self.cal, controls=self.controls)
        self.schedule = self.campaign.order_schedule(self.request.candidate_id)

    def test_a_replayed_block_is_refused(self):
        rep = tuple(st.PairedBlock(
            block_index=0, unit_id=selection_unit(self.split, i),
            stratum=api.STRATUM_SELECTION, order=self.schedule.order_for(i),
            anchor_samples=(100.0, 100.0, 100.0),
            candidate_samples=(115.0, 115.0, 115.0), measured_at=NOW)
            for i in range(self.campaign.b_min))
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(self.request, rep)
        self.assertEqual(cm.exception.reduction.check("block_identity").outcome, S.FAIL)
        self.assertIsNone(cm.exception.reduction.estimate)

    def test_a_block_that_is_not_at_the_index_it_claims_is_refused(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        shifted = tuple(st.PairedBlock(
            block_index=b.block_index + 100, unit_id=b.unit_id, stratum=b.stratum,
            order=b.order, anchor_samples=b.anchor_samples,
            candidate_samples=b.candidate_samples, segment=b.segment,
            extension_round=b.extension_round, measured_at=b.measured_at)
            for b in blocks)
        reduction = self.reducer.reduce(self.request, shifted)
        self.assertEqual(reduction.check("block_identity").outcome, S.FAIL)

    def test_a_conforming_run_still_passes_block_identity(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        reduction = self.reducer.reduce(self.request, blocks)
        self.assertEqual(reduction.check("block_identity").outcome, S.PASS)
        self.assertIsNotNone(reduction.estimate)


class TestTheBaseSegmentIsAlwaysChecked(unittest.TestCase):
    """The base-count guard was `len(blocks) > b_min`, so a submission of exactly
    B_min blocks skipped it: B_min blocks with ZERO base blocks, all labelled
    "extension round 1", were admitted. An extension round that arrives before
    the base segment it extends is not an extension."""

    def setUp(self):
        (_i, _s, self.cal, self.rule, _c, self.split, self.campaign,
         self.controls) = _Campaign.get()
        self.reducer = st.PairedBlockReducer(self.campaign)
        self.request = make_request(calibration=self.cal, controls=self.controls)
        self.schedule = self.campaign.order_schedule(self.request.candidate_id)

    def _all_extension(self):
        return tuple(st.PairedBlock(
            block_index=i, unit_id=selection_unit(self.split, i),
            stratum=api.STRATUM_SELECTION, order=self.schedule.order_for(i),
            anchor_samples=(100.0, 100.0, 100.0),
            candidate_samples=(115.0, 115.0, 115.0), segment=st.SEGMENT_EXTENSION,
            extension_round=1, measured_at=NOW)
            for i in range(self.campaign.b_min))

    def test_b_min_blocks_with_no_base_segment_are_refused(self):
        chk = st._check_extension_structure(self._all_extension(),
                                            b_min=self.campaign.b_min, rule=self.rule)
        self.assertEqual(chk.outcome, S.FAIL)
        self.assertTrue(any("base segment" in r for r in chk.reasons))

    def test_the_seam_refuses_them_too(self):
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(self.request, self._all_extension())
        self.assertEqual(
            cm.exception.reduction.check("extension_structure").outcome, S.FAIL)

    def test_a_conforming_base_only_run_still_passes(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        base_only = blocks[:self.campaign.b_min]
        self.assertEqual(
            st._check_extension_structure(base_only, b_min=self.campaign.b_min,
                                          rule=self.rule).outcome, S.PASS)


class TestOverExtensionIsJournaledNotRaised(unittest.TestCase):
    """A run that used MORE blocks than its declared ceiling crashed the reducer.

    `max_total_blocks` CLAMPED a `b_min` above `max_blocks_per_candidate` to the
    ceiling, producing a window shorter than the base segment it was meant to
    contain; `_replay` then raised `InsufficientMaterial("replay needs at least
    b_min=21 blocks, got 20")` out of `reduce()`, with no checks attached — so
    the one run the protocol most wants journaled as INVALID (*"Extension
    follows the declared rule only"*) was the one that produced a traceback
    about internal bookkeeping instead of a record.
    """

    def setUp(self):
        (_i, _s, self.cal, self.rule, _c, self.split, self.campaign,
         self.controls) = _Campaign.get()
        self.reducer = st.PairedBlockReducer(self.campaign)
        self.request = make_request(calibration=self.cal, controls=self.controls)
        self.schedule = self.campaign.order_schedule(self.request.candidate_id)

    def _over_ceiling_blocks(self):
        n = self.rule.max_blocks_per_candidate + 1
        b_min = self.campaign.b_min
        return tuple(st.PairedBlock(
            block_index=i, unit_id=selection_unit(self.split, i),
            stratum=api.STRATUM_SELECTION, order=self.schedule.order_for(i),
            anchor_samples=(100.0, 100.1, 99.9),
            candidate_samples=(115.0, 115.1, 114.9),
            segment=st.SEGMENT_BASE if i < b_min else st.SEGMENT_EXTENSION,
            extension_round=None if i < b_min else 1, measured_at=NOW)
            for i in range(n))

    def test_the_ceiling_is_refused_not_clamped(self):
        ceiling = self.rule.max_blocks_per_candidate
        self.assertEqual(self.rule.max_total_blocks(ceiling), ceiling)
        with self.assertRaises(st.StoppingRuleViolation) as cm:
            self.rule.max_total_blocks(ceiling + 1)
        self.assertIn("max_blocks_per_candidate", str(cm.exception))

    def test_an_over_extended_run_comes_back_as_a_reduction(self):
        """Still refused, still journalable — and now the reason NAMES it.

        Until 2026-08-04 the over-extension surfaced as `mde_derivable` FAIL,
        because `reduce()` handed `len(blocks)` to `mde_for` and
        `max_total_blocks` refused the over-ceiling `b_min`. That was the right
        verdict reached through the wrong organ: the record said *"no MDE could
        be derived"*, which reads as a bookkeeping failure, about a run whose
        actual finding is that it used more blocks than its rule licenses —
        exactly the complaint in this class's own docstring, one layer along.

        `reduce()` now derives the MDE from `campaign.b_min` (it must: the
        realized count asks `solve_mde` for a window the rule cannot license,
        which is optimistic by 18.8-43.5%), so the MDE is derivable here. The
        over-extension is named by `block_count` and by the new `mde_window`,
        which bounds the realized count by `b_min + max_rounds * blocks_per_round`
        — TIGHTER than `block_count`'s `max_blocks_per_candidate`, so it also
        catches a run between the two.
        """
        reduction = self.reducer.reduce(self.request, self._over_ceiling_blocks())
        self.assertEqual(reduction.check("block_count").outcome, S.FAIL)
        self.assertEqual(reduction.check("mde_window").outcome, S.FAIL)
        self.assertIn("licenses at most",
                      " ".join(reduction.check("mde_window").reasons))
        self.assertTrue(reduction.mde.found)
        self.assertIsNone(reduction.estimate)
        S.canonical_json(reduction.to_dict())          # still journalable

    def test_a_run_between_the_two_ceilings_is_caught_by_mde_window_alone(self):
        """THE BITE for `mde_window`: `block_count` cannot see this run.

        `max_blocks_per_candidate` is the campaign-wide ceiling; the rule's own
        budget for ONE candidate is `b_min + max_rounds * blocks_per_round`. When
        the second is smaller than the first, a run between them passes
        `block_count` and is described by an MDE for a shorter window than it
        actually ran.
        """
        b_min = self.campaign.b_min
        rule = self.rule
        licensed = rule.max_total_blocks(b_min)
        if licensed >= rule.max_blocks_per_candidate:
            self.skipTest("this campaign's extension budget exhausts the ceiling, "
                          "so there is no gap between the two bounds")
        n = licensed + 1
        blocks = tuple(st.PairedBlock(
            block_index=i, unit_id=selection_unit(self.split, i),
            stratum=api.STRATUM_SELECTION, order=self.schedule.order_for(i),
            anchor_samples=(100.0, 100.1, 99.9),
            candidate_samples=(115.0, 115.1, 114.9),
            segment=st.SEGMENT_BASE if i < b_min else st.SEGMENT_EXTENSION,
            extension_round=None if i < b_min else 1, measured_at=NOW)
            for i in range(n))
        reduction = self.reducer.reduce(self.request, blocks)
        self.assertEqual(reduction.check("block_count").outcome, S.PASS)
        self.assertEqual(reduction.check("mde_window").outcome, S.FAIL)
        self.assertIsNone(reduction.estimate)

    def test_the_declared_window_is_the_compliant_control_for_mde_window(self):
        """A run of exactly the licensed length passes `mde_window`."""
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        reduction = self.reducer.reduce(self.request, blocks)
        self.assertEqual(reduction.check("mde_window").outcome, S.PASS,
                         reduction.check("mde_window").reasons)
        self.assertLessEqual(len(blocks),
                             self.rule.max_total_blocks(self.campaign.b_min))

    def test_the_seam_raises_the_journalable_refusal(self):
        with self.assertRaises(st.ReductionInadmissible) as cm:
            self.reducer.reduce_blocks(self.request, self._over_ceiling_blocks())
        joined = " ".join(r for _n, chk in cm.exception.reduction.checks
                          for r in chk.reasons)
        self.assertIn("max_blocks_per_candidate", joined)

    def test_a_conforming_block_count_still_publishes_an_mde(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        reduction = self.reducer.reduce(self.request, blocks)
        self.assertTrue(reduction.mde.found)
        self.assertEqual(reduction.check("mde_derivable").outcome, S.PASS)


class TestMaterialRefusalsStayJournalable(unittest.TestCase):
    """*"A voided run is journaled as INVALID with its reason."*

    `reduce()` documented "never raises on a non-conforming run" and then raised
    a bare `MaterialError` when a block's anchor arm medianed to zero under a
    relative scale — carrying no checks, so the run could not be journaled with
    its reason. The refusal stands (there is nothing to reduce), but it now
    carries the checks a `BlockReduction` would have carried.
    """

    def setUp(self):
        (_i, _s, self.cal, _r, _c, _sp, self.campaign, self.controls) = _Campaign.get()
        self.reducer = st.PairedBlockReducer(self.campaign)
        self.request = make_request(calibration=self.cal, controls=self.controls)

    def test_a_degenerate_anchor_arm_refuses_with_journalable_checks(self):
        _seq, blocks = run_candidate(self.campaign, effect=0.15, seed=31)
        bad = list(blocks)
        b0 = bad[0]
        bad[0] = st.PairedBlock(
            block_index=b0.block_index, unit_id=b0.unit_id, stratum=b0.stratum,
            order=b0.order, anchor_samples=(0.0, 0.0, 0.0),
            candidate_samples=b0.candidate_samples, segment=b0.segment,
            extension_round=b0.extension_round, measured_at=b0.measured_at)
        with self.assertRaises(st.MaterialError) as cm:
            self.reducer.reduce_blocks(self.request, tuple(bad))
        checks = dict(getattr(cm.exception, "checks", ()))
        self.assertEqual(checks["effect_scale"].outcome, S.FAIL)
        # The api.WindowAttestations fields this reduction is authoritative for
        # are present, so the window can still be attested as INVALID.
        for name in ("stratum_partition", "stopping_rule_unmodified", "order_control",
                     "calibration_cell"):
            self.assertIn(name, checks)

    def test_zero_blocks_refuses_with_a_reason_too(self):
        with self.assertRaises(st.InsufficientMaterial) as cm:
            self.reducer.reduce(self.request, ())
        self.assertEqual(dict(getattr(cm.exception, "checks", ()))["block_count"].outcome,
                         S.FAIL)


class CombineChecksIsTheOneLatticeTest(unittest.TestCase):
    """`statistics._combine_checks` delegates to `schemas.Check.worst_of`.

    Its result gates whether an `EffectEstimate` is built at all, so "no
    admissibility check ran" must not read as "admissible".
    """

    def test_an_empty_admissibility_vector_is_could_not_check_and_never_pass(self):
        combined = st._combine_checks([])
        self.assertEqual(combined.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(combined.passed)

    def test_reasons_are_prefixed_with_the_outcome_that_raised_them(self):
        combined = st._combine_checks([S.Check(S.COULD_NOT_CHECK, ("no raw samples",)),
                                       S.Check(S.FAIL, ("mde window exceeded",))])
        self.assertEqual(combined.outcome, S.FAIL)
        self.assertEqual(combined.reasons, ("[COULD_NOT_CHECK] no raw samples",
                                            "[FAIL] mde window exceeded"))

    def test_a_non_check_element_raises(self):
        with self.assertRaises(TypeError):
            st._combine_checks([S.Check(S.PASS), None])

    def test_the_delegation_is_real_and_not_a_reimplementation(self):
        for vector in ([], [S.Check(S.PASS)],
                       [S.Check(S.PASS), S.Check(S.COULD_NOT_CHECK, ("x",))],
                       [S.Check(S.FAIL, ("y",))]):
            with self.subTest(vector=[c.outcome for c in vector]):
                self.assertEqual(st._combine_checks(vector), S.Check.worst_of(vector))


if __name__ == "__main__":
    unittest.main(verbosity=2)
