#!/usr/bin/env python3
"""test_control_runner.py — the five controls, executed.

WHAT THIS SUITE IS FOR
----------------------
`evaluator/controls.py` was complete and had never run a control: `ControlRunner`
was a Protocol with no implementation, and every control result in the tree was a
`ControlObservation` handed over by a fixture object. This suite exercises the
implementation, and — more to the point — it exercises the three things the
implementation is supposed to make impossible:

  1. a control scored down a path a candidate never takes;
  2. five controls sharing one seed for the life of a campaign; and
  3. a `ControlPanelResult` that says `may_rank` with no control ever run.

Every guard here has BOTH a failing case and a compliant-path control, because a
guard whose FAIL branch is unreachable is a guard nobody has shown to work, and a
guard that also refuses the correct idiom is a guard that will be deleted.

WHAT IT DOES NOT DO
-------------------
NO inference, NO benchmark, NO build, NO model, NO server, NO claim. No process
is started, stopped or signalled, and nothing outside `tempfile` is written. The
"measurements" are deterministic `random.Random(seed)` numbers standing in for
recorded material; nothing here is a measurement of anything and nothing here may
be reported as one. The first real control run happens under a held claim, on an
uncontended host, with this code unchanged.

Run:
    python3 -W error::ResourceWarning -m unittest \\
        scripts.kernel_rnd.autokernel.execution.test_control_runner
"""
from __future__ import annotations

import dataclasses
import hashlib
import random
import sys
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S                        # noqa: E402
from autokernel.evaluator import api                       # noqa: E402
from autokernel.evaluator import controls as CT            # noqa: E402
from autokernel.evaluator import statistics as ST          # noqa: E402
from autokernel.execution import control_runner as CR      # noqa: E402

PASS = S.Check(S.PASS)
NOW = "2026-08-03T12:00:00+00:00"
CAMPAIGN = "ak-llama_cpu-prefill-20260803"
CAMPAIGN_SEED = "ak-campaign-seed-controls-0001"
CONSTRUCTION_ID = "sign_martingale_predictable_lambda/v1"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# =============================================================================
# The campaign — solved once, exactly as a real one is
# =============================================================================

def _calibration_blocks(count, *, effect, noise, seed, stratum, prefix, split):
    rng = random.Random(seed)
    blocks = []
    for i in range(count):
        anchor_arm = tuple(100.0 + rng.gauss(0, noise * 100.0) for _ in range(3))
        med = sorted(anchor_arm)[1]
        cand_arm = tuple(med * (1.0 + effect) + rng.gauss(0, noise * 100.0)
                         for _ in range(3))
        unit = f"{prefix}-{i}"
        while split.assign(unit) != stratum:
            unit += "x"
        blocks.append(ST.PairedBlock(
            block_index=i, unit_id=unit, stratum=stratum,
            order=(ST.ORDER_ANCHOR_FIRST if i % 2 == 0 else ST.ORDER_CANDIDATE_FIRST),
            anchor_samples=anchor_arm, candidate_samples=cand_arm, measured_at=NOW))
    return tuple(blocks)


class Campaign:
    """Everything fixed before the first control is measured. Solved once."""

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is not None:
            return cls._cache
        controls_decl = api.CampaignControls(
            calibration_block_count=200, contribution_floor=0.10, max_candidates=10,
            confirmation_admission_count=2, max_blocks_per_candidate=20,
            storage_floor_bytes_free=200 * 1024 ** 3)
        rule = ST.StoppingRule(
            rule_id="ak-stop-controls/v1", final_table="t1_paired_block_table",
            decisions=(("evidence_threshold_crossed", "compose_into_champion_lineage"),
                       ("extension_exhausted", "abandon"),
                       ("block_ceiling_reached", "abandon")),
            extension=ST.BoundedExtension(max_rounds=1, blocks_per_round=5),
            max_blocks_per_candidate=20)
        construction = ST.select_construction(CONSTRUCTION_ID)
        split = ST.StratumSplitRule(
            rule_id="ak-split-controls/v1", campaign_seed=CAMPAIGN_SEED,
            confirmation_fraction=0.3,
            rotation=ST.RotationSchedule(schedule_id="ak-rot-controls/v1",
                                         period_campaigns=4))
        rng = random.Random(3)
        anchor_values = tuple(100.0 + rng.gauss(0, 1.0) for _ in range(200))
        inputs = ST.CalibrationInputs(
            backend="llama_cpu", phase="prefill", cell_class="instrument_tokens_per_s",
            campaign_seed=CAMPAIGN_SEED, controls=controls_decl, stopping_rule=rule,
            construction=construction, effect_scale=ST.EFFECT_SCALE_RELATIVE,
            metric_direction="higher_better", hypothesis=ST.HYPOTHESIS_IMPROVEMENT,
            margin=0.0,
            aa_blocks=_calibration_blocks(200, effect=0.0, noise=0.01, seed=1,
                                          stratum=api.STRATUM_SELECTION, prefix="aa",
                                          split=split),
            neutral_blocks=_calibration_blocks(60, effect=0.0, noise=0.01, seed=2,
                                               stratum=api.STRATUM_SELECTION,
                                               prefix="nt", split=split),
            anchor_calibration_values=anchor_values,
            samples_ref=f"ak-raw://{CAMPAIGN}/calibration/0001")
        solve = ST.solve_calibration(inputs)
        outputs = solve.require_accepted()
        commitment = ST.StoppingRuleCommitment.commit(rule, campaign_id=CAMPAIGN,
                                                      committed_at=NOW)
        stats = ST.CampaignStatistics(
            campaign_id=CAMPAIGN, campaign_seed=CAMPAIGN_SEED,
            effect_scale=ST.EFFECT_SCALE_RELATIVE,
            hypothesis=ST.HYPOTHESIS_IMPROVEMENT, margin=0.0, stopping_rule=rule,
            stopping_rule_commitment=commitment, split_rule=split,
            construction=construction, calibration=outputs,
            aa_effect_pool=solve.aa_effect_pool,
            anchor_calibration_values=solve.anchor_calibration_values)
        cls._cache = (controls_decl, rule, construction, split, solve, outputs, stats)
        return cls._cache


def anchor_identity() -> api.AnchorIdentity:
    return api.AnchorIdentity(
        source_commit=V8_COMMIT, binary_sha256=sha("anchor-binary"),
        linkage_sha256=sha("anchor-linkage"),
        measurement_event_ids=("ake-anchor-0001",))


# =============================================================================
# The candidate-side wiring: a tier gate runner, a dispatcher, a pipeline
# =============================================================================

#: The one artifact whose op suite genuinely fails. The gate runner keys on the
#: ARTIFACT, never on the candidate id and never on anything naming a control:
#: `_GateRunner` must be as blind to control-ness as the pipeline is, or this
#: suite would be proving a property of its own fixture.
DEGRADED_SOURCE = sha("degraded-negative-source")


class _GateRunner:
    """A tier gate runner. Stands in for the op suites, and measures the artifact."""

    def __init__(self, tier: str):
        self.tier = tier
        self.requests = []

    def run_gates(self, request):
        self.requests.append(request)
        correctness = (
            S.Check(S.FAIL, ("the op suite disagreed with the reference on 11/64 "
                             "shapes; the kernel silently falls back and reports the "
                             "cached result",))
            if request.artifact.source_sha256 == DEGRADED_SOURCE else PASS)
        return (
            api.GateResult(gate_id="ops-suite", gate_class=api.GATE_CORRECTNESS,
                           check=correctness, requires_anchor=True,
                           evidence_ref="ak-raw://ops/0001"),
            api.GateResult(gate_id="numerics", gate_class=api.GATE_NUMERICAL_SAFETY,
                           check=PASS, requires_anchor=True),
        )


class _RecordingPipeline:
    """Wraps the real pipeline and records what it was handed.

    It records and forwards; it does not answer. A recording stub that produced
    its own outcomes would make every assertion below a statement about the stub.
    """

    def __init__(self, inner):
        self._inner = inner
        self.pipeline_id = inner.pipeline_id
        self.submissions = []

    def evaluate(self, submission):
        self.submissions.append(submission)
        return self._inner.evaluate(submission)


def build_pipeline():
    stats = Campaign.get()[6]
    dispatcher = api.TierDispatcher(gate_runners={
        "T0": _GateRunner("T0"), "T1": _GateRunner("T1"), "T2": _GateRunner("T2")})
    reducer = ST.PairedBlockReducer(stats)
    return _RecordingPipeline(CR.DispatchPipeline(dispatcher=dispatcher,
                                                  reducer=reducer))


# =============================================================================
# The fixtures — the five controls' recorded material
# =============================================================================

def _arms(count, *, effect, noise, seed):
    """`(anchor_blocks, candidate_blocks)` — per-block sample vectors."""
    rng = random.Random(seed)
    anchor_blocks, candidate_blocks = [], []
    for _ in range(count):
        anchor_arm = tuple(100.0 + rng.gauss(0, noise * 100.0) for _ in range(3))
        med = sorted(anchor_arm)[1]
        anchor_blocks.append(anchor_arm)
        candidate_blocks.append(
            tuple(med * (1.0 + effect) + rng.gauss(0, noise * 100.0) for _ in range(3)))
    return tuple(anchor_blocks), tuple(candidate_blocks)


def _fixture(control_id, *, tier, effect, seed, source_tag, blocks, available=True,
             unavailable_reason=None, measured_at=NOW):
    definition = next(d for d in CT.CONTROL_DEFINITIONS if d.control_id == control_id)
    anchor_blocks, candidate_blocks = _arms(blocks, effect=effect, noise=0.01, seed=seed)
    if not available:
        anchor_blocks, candidate_blocks = (), ()
    return CR.ControlFixture(
        fixture_id=definition.fixture_id,
        control_id=control_id,
        tier=tier,
        candidate_id=f"akc-control-{control_id.replace('_', '-')}",
        artifact=api.ArtifactIdentity(
            source_sha256=sha(source_tag), binary_sha256=sha(source_tag + "-bin"),
            linkage_sha256=sha(source_tag + "-link")),
        determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                          same_seed_repeat_runs=3),
        created_at=NOW,
        measured_at=measured_at,
        stratum=api.STRATUM_SELECTION,
        anchor_samples=anchor_blocks,
        candidate_samples=candidate_blocks,
        available=available,
        unavailable_reason=unavailable_reason,
    )


def build_fixtures(*, blocks=None, degraded_effect=0.90, replay_effect=0.36,
                   positive_effect=0.30, seed_offset=0):
    # Exactly B_min base blocks. The count is the CAMPAIGN's, solved by the
    # calibration block: fewer is below the calibrated floor and more is an
    # undeclared extension, and both are refused by the reducer — for a control
    # exactly as for a candidate.
    stats = Campaign.get()[6]
    blocks = (stats.b_min + stats.stopping_rule.extension.blocks_per_round
              if blocks is None else blocks)
    return (
        # `seed_offset` stands in for RE-MEASURING: a later window's controls are
        # a fresh run, not a replay of the recorded one, so every arm differs.
        _fixture(CT.CONTROL_POSITIVE, tier="T1", effect=positive_effect,
                 seed=11 + seed_offset,
                 source_tag="positive-source", blocks=blocks),
        _fixture(CT.CONTROL_NEUTRAL, tier="T1", effect=0.0, seed=12 + seed_offset,
                 source_tag="neutral-source", blocks=blocks),
        # "Deliberately fast-looking but wrong": it IS fast. The gate is supposed
        # to withhold the rank anyway, which is the whole point of the control.
        _fixture(CT.CONTROL_DEGRADED_NEGATIVE, tier="T1", effect=degraded_effect,
                 seed=13 + seed_offset, source_tag="degraded-negative-source",
                 blocks=blocks),
        _fixture(CT.CONTROL_AA, tier="T1", effect=0.0, seed=14 + seed_offset,
                 source_tag="aa-source", blocks=blocks),
        _fixture(CT.CONTROL_HISTORICAL_WIN_REPLAY, tier="T2", effect=replay_effect,
                 seed=15 + seed_offset, source_tag="iqk-prefill-port", blocks=blocks),
    )


def fixture_set(fixtures=None):
    fixtures = build_fixtures() if fixtures is None else fixtures
    digest = S.content_hash(CR._fixture_payload(fixtures))
    return CR.resolve_fixture_set(fixtures=fixtures, pinned_digest=digest,
                                  source_label="evaluator-bundle@ak3-control-tests")


# =============================================================================
# Bindings, bundle, contexts
# =============================================================================

def window_template() -> api.WindowAttestations:
    a = anchor_identity()
    return api.WindowAttestations(
        resource_claim_receipt="akclaim-region-0001",
        resource_claim_open=PASS, resource_claim_close=PASS,
        resource_claim_same_holder=PASS, no_concurrent_inference=PASS,
        preflight_attestation_ref="ake-preflight-0001",
        host_receipt="host-health-20260803T1159Z", host_health=PASS,
        anchor_at_open=a, anchor_at_close=a, anchor_gate=PASS,
        evaluator_bundle=PASS, runtime_source_label=PASS,
        recipe=api.RecipeReceipt(
            constructor_id="ak.microbench.llama_cpu.prefill/v1",
            constructor_sha256=sha("recipe-constructor"), argv_sha256=sha("argv")),
        storage_open=PASS, storage_close=PASS, strata=PASS,
        stopping_rule_id="ak-stop-controls/v1", rule_immutability=PASS,
        order_randomized=PASS,
        # A placeholder, and the suite asserts it never reaches a runner: the
        # per-control seed is derived and replaces it.
        order_seed="PLACEHOLDER-SEED-MUST-NOT-BE-USED",
        aa_cadence=PASS,
        controls=api.ControlPanel(positive=PASS, neutral=PASS,
                                  degraded_negative=PASS, aa=PASS,
                                  historical_replay=PASS),
        calibration=PASS, control_definitions_immutable=PASS,
        raw_evidence_ref=f"data/{CAMPAIGN}/raw/controls/")


def binding() -> CR.CampaignBinding:
    controls_decl, _rule, _cons, _split, _solve, outputs, _stats = Campaign.get()
    return CR.CampaignBinding(
        campaign_id=CAMPAIGN, backend="llama_cpu", phase="prefill",
        cell_class="instrument_tokens_per_s", protocol_id=api.PROTOCOL_VERSIONED_ID,
        evaluator=api.EvaluatorIdentity(
            id="P-AK-SEARCH-1/v1", bundle_sha256=sha("evaluator-bundle"),
            runtime_source_label_ref="ake-srclabel-0001"),
        scope_denominator=api.ScopeDenominator(
            machine_subset="full", numa_nodes=(), devices=(), cores=96),
        scope_manifest_sha256=sha("scope-manifest"), co_residency="single",
        metric="prefill_tokens_per_s", metric_direction="higher_better", reps=10,
        change_class="parameter",
        anchor=anchor_identity(), campaign_controls=controls_decl,
        calibration=outputs)


def declaration() -> CT.HistoricalWinReplayDeclaration:
    return CT.HistoricalWinReplayDeclaration(
        win_id="iqk-prefill-port", backend="llama_cpu", phase="prefill",
        reference_direction="higher_better",
        reference_band=CT.ReferenceBand(low=0.30, high=0.45),
        evidence_locator="data/ak-x/iqk-prefill-port.json",
        durability_class="carried_in_git")


def bundle(rotate_every_windows=10) -> CT.ControlBundle:
    return CT.resolve_control_bundle(
        pinned_definitions_digest=CT.CONTROL_DEFINITIONS_DIGEST,
        aa_cadence=CT.AACadence(every_n_windows=5, every_n_seconds=3600.0,
                                declared_at=NOW),
        seed_rotation=CT.SeedRotationSchedule(
            rotate_every_windows=rotate_every_windows, declared_at=NOW),
        historical_win_replays=(declaration(),),
        source_label="evaluator-bundle@ak3-control-tests")


def available_resolution() -> CT.HistoricalWinResolution:
    return CT.HistoricalWinResolution(
        backend="llama_cpu", available=True, declaration=declaration(),
        durability_outcome=S.PASS,
        check=S.Check(S.PASS, ("fixture: resolves in-repo",)))


def run_context(window_id="akw-0001") -> CT.ControlRunContext:
    return CT.ControlRunContext(
        campaign_id=CAMPAIGN, backend="llama_cpu", phase="prefill",
        cell_class="instrument_tokens_per_s", window_id=window_id, tier="T1",
        seed="PLACEHOLDER-SEED-MUST-NOT-BE-USED", anchor=anchor_identity(),
        declaration=declaration())


def control_context(window_id="akw-0001", historical=None) -> CT.ControlContext:
    solve = Campaign.get()[4]
    return CT.ControlContext(
        campaign_id=CAMPAIGN, backend="llama_cpu", phase="prefill",
        cell_class="instrument_tokens_per_s", window_id=window_id,
        historical=available_resolution() if historical is None else historical,
        neutral_dispersion=CT.neutral_dispersion_check(solve),
        calibration=Campaign.get()[5])


class Harness:
    """Assembles the whole stack. Every test that needs a sweep builds one."""

    def __init__(self, *, fixtures=None, rotate_every_windows=10):
        self.pipeline = build_pipeline()
        self.fixture_set = fixture_set(fixtures)
        self.binding = binding()
        self.stats = Campaign.get()[6]
        self.runner = CR.ExecutedControlRunner(
            pipeline=self.pipeline, fixtures=self.fixture_set, binding=self.binding,
            campaign_statistics=self.stats)
        self.bundle = bundle(rotate_every_windows)
        self.harness = CT.ControlHarness(bundle=self.bundle, runner=self.runner)
        self.sweep = CR.ControlSweep(harness=self.harness, campaign_seed=CAMPAIGN_SEED)

    def run(self, *, windows_completed=0, last_rotation_epoch=0, window_id="akw-0001",
            window=None):
        return self.sweep.run(
            run_context=run_context(window_id), context=control_context(window_id),
            window=window_template() if window is None else window,
            aa_cadence=PASS, windows_completed=windows_completed,
            last_rotation_epoch=last_rotation_epoch)


# =============================================================================
# 1. One code path
# =============================================================================

class TestOneCodePath(unittest.TestCase):
    """*"A control that runs down a different code path proves nothing about the
    path that matters."* Structural, not documented."""

    def test_every_control_is_handed_to_the_pipeline_as_a_candidate_submission(self):
        h = Harness()
        result = h.run()
        self.assertEqual(len(h.pipeline.submissions), 5)
        for submission in h.pipeline.submissions:
            self.assertIsInstance(submission, CR.CandidateSubmission)
            self.assertIsInstance(submission.request, api.EvaluationRequest)
            # A control is submitted with a candidate's identity. If the pipeline
            # could recognise the id shape it could branch on it.
            self.assertTrue(submission.request.candidate_id.startswith("akc-"))
        self.assertTrue(result.may_rank)

    def test_the_submission_type_carries_no_marker_and_no_effect(self):
        self.assertEqual(
            CR.audit_submission_carries_no_control_marker().outcome, S.PASS)
        names = {f.name for f in dataclasses.fields(CR.CandidateSubmission)}
        self.assertEqual(names, {"request", "window", "blocks"})

    def test_the_marker_audit_has_a_reachable_fail_branch(self):
        """The compliant path passes above; this is the same guard, biting."""

        @dataclasses.dataclass(frozen=True)
        class _Marked:
            request: object
            window: object
            blocks: tuple
            is_control: bool = False

        check = CR.audit_submission_carries_no_control_marker(_Marked)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("is_control", " ".join(check.reasons))

    def test_a_non_dataclass_submission_type_is_could_not_check(self):
        check = CR.audit_submission_carries_no_control_marker(object)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_the_single_path_audit_passes_on_the_real_module(self):
        self.assertEqual(CR.audit_single_evaluation_path().outcome, S.PASS)

    def test_the_single_path_audit_catches_a_second_route_to_a_verdict(self):
        source = Path(CR.__file__).read_text(encoding="utf-8")
        tampered = source.replace(
            "        submission = self._submission_for(fixture, context)",
            "        submission = self._submission_for(fixture, context)\n"
            "        _sneaky = self._dispatcher.dispatch(submission.request, None)")
        self.assertNotEqual(tampered, source)
        check = CR.audit_single_evaluation_path(tampered)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("dispatch", " ".join(check.reasons))

    def test_the_single_path_audit_catches_a_second_call_site(self):
        source = Path(CR.__file__).read_text(encoding="utf-8")
        tampered = source.replace(
            "            outcome = self._pipeline.evaluate(submission)",
            "            outcome = (self._pipeline.evaluate(submission)\n"
            "                       if fixture.available\n"
            "                       else self._pipeline.evaluate(submission))")
        self.assertNotEqual(tampered, source)
        check = CR.audit_single_evaluation_path(tampered)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("exactly one", " ".join(check.reasons))

    def test_unparseable_source_is_could_not_check_not_a_pass(self):
        self.assertEqual(
            CR.audit_single_evaluation_path("def broken(:").outcome, S.COULD_NOT_CHECK)
        self.assertEqual(
            CR.audit_single_evaluation_path("x = 1").outcome, S.COULD_NOT_CHECK)

    def test_the_pipeline_computes_the_effect_the_control_is_scored_on(self):
        """A control cannot report the number it is about to be ranked by.

        The estimate on the verdict is the REDUCER's, and its `raw_samples` are
        byte-for-byte the blocks that were submitted — so the number the control
        is scored on is derived from the material, never handed over with it.
        """
        h = Harness()
        result = h.run()
        for submission in h.pipeline.submissions:
            self.assertFalse(hasattr(submission, "effect"))
            self.assertTrue(submission.blocks)
        positive_submission = next(s for s in h.pipeline.submissions
                                   if "positive" in s.request.candidate_id)
        positive_observation = next(o for o in result.observations
                                    if o.control_id == CT.CONTROL_POSITIVE)
        estimate = positive_observation.verdict.effect
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.raw_samples,
                         tuple(b.to_tuple() for b in positive_submission.blocks))
        self.assertEqual(estimate.paired_blocks, len(positive_submission.blocks))

    def test_a_pipeline_returning_the_wrong_type_is_a_wiring_error(self):
        class _Bad:
            pipeline_id = "bad/v1"

            def evaluate(self, submission):
                return {"status": "OK"}

        runner = CR.ExecutedControlRunner(
            pipeline=_Bad(), fixtures=fixture_set(), binding=binding(),
            campaign_statistics=Campaign.get()[6])
        runner.open_window(window_id="akw-0001", window=window_template())
        definition = CT.CONTROL_DEFINITIONS[0]
        with self.assertRaises(CR.PipelineNotWired):
            runner.run_control(definition, dataclasses.replace(
                run_context(), seed="a-derived-seed"))

    def test_there_is_no_default_pipeline_reducer_or_statistics(self):
        for kwargs in (
                {"pipeline": None},
                {"fixtures": None},
                {"binding": None},
                {"campaign_statistics": None}):
            base = dict(pipeline=build_pipeline(), fixtures=fixture_set(),
                        binding=binding(), campaign_statistics=Campaign.get()[6])
            base.update(kwargs)
            with self.subTest(**{k: "None" for k in kwargs}):
                with self.assertRaises(CR.PipelineNotWired):
                    CR.ExecutedControlRunner(**base)

    def test_the_dispatch_pipeline_requires_both_halves(self):
        dispatcher = api.TierDispatcher(gate_runners={"T1": _GateRunner("T1")})
        reducer = ST.PairedBlockReducer(Campaign.get()[6])
        with self.assertRaises(CR.PipelineNotWired):
            CR.DispatchPipeline(dispatcher=dispatcher, reducer=None)
        with self.assertRaises(CR.PipelineNotWired):
            CR.DispatchPipeline(dispatcher=object(), reducer=reducer)
        with self.assertRaises(CR.PipelineNotWired):
            CR.DispatchPipeline(dispatcher=dispatcher, reducer=reducer,
                                pipeline_id="unversioned")
        # Compliant path.
        CR.DispatchPipeline(dispatcher=dispatcher, reducer=reducer)


# =============================================================================
# 2. Seed rotation
# =============================================================================

class TestSeedRotation(unittest.TestCase):
    """`derive_control_seed`, `ControlBundle.seed_for` and
    `SeedRotationSchedule.check_rotation` were declared, hashed, and had no
    caller. These are the tests that fail if the caller goes away again."""

    def test_each_control_gets_its_own_seed(self):
        h = Harness()
        seeds = h.sweep.seed_ledger(windows_completed=0)
        self.assertEqual(len(seeds), 5)
        self.assertEqual(len({s.seed for s in seeds}), 5)
        self.assertEqual([s.control_id for s in seeds], list(CT.CONTROL_IDS))

    def test_the_seeds_are_the_ratified_derivation_not_a_local_one(self):
        h = Harness()
        for assignment in h.sweep.seed_ledger(windows_completed=37):
            self.assertEqual(
                assignment.seed,
                CT.derive_control_seed(campaign_seed=CAMPAIGN_SEED,
                                       control_id=assignment.control_id,
                                       epoch=assignment.epoch))
            self.assertEqual(assignment.epoch, 3)  # 37 // 10

    def test_the_placeholder_seed_never_reaches_a_runner(self):
        h = Harness()
        h.run()
        placeholder = run_context().seed
        derived = {s.seed for s in h.sweep.seed_ledger(windows_completed=0)}
        for submission in h.pipeline.submissions:
            self.assertNotEqual(submission.window.order_seed, placeholder)
            self.assertIn(submission.window.order_seed, derived)

    def test_rotating_the_epoch_rotates_the_measured_units(self):
        """A seed that changes nothing about what was measured is a rotation
        schedule with no subject."""
        first = Harness()
        first.run(windows_completed=0, last_rotation_epoch=0)
        second = Harness()
        second.run(windows_completed=10, last_rotation_epoch=1)

        def units(h):
            return {tuple(b.unit_id for b in s.blocks) for s in h.pipeline.submissions}

        self.assertEqual(len(units(first)), 5)
        self.assertTrue(units(first).isdisjoint(units(second)))

    def test_the_same_epoch_reproduces_the_same_units(self):
        """Rotation is a schedule, not randomness: within an epoch it is stable."""
        a, b = Harness(), Harness()
        a.run(windows_completed=3, last_rotation_epoch=0)
        b.run(windows_completed=7, last_rotation_epoch=0)
        self.assertEqual(
            [tuple(bl.unit_id for bl in s.blocks) for s in a.pipeline.submissions],
            [tuple(bl.unit_id for bl in s.blocks) for s in b.pipeline.submissions])

    def test_an_overdue_rotation_blocks_the_sweep(self):
        h = Harness()
        result = h.run(windows_completed=20, last_rotation_epoch=0)
        self.assertEqual(result.rotation_check.outcome, S.FAIL)
        self.assertIsNone(result.panel_result)
        self.assertFalse(result.may_rank)
        self.assertEqual(h.pipeline.submissions, [])
        self.assertIn("rotation schedule", result.blocked_reason)

    def test_rotating_ahead_of_schedule_also_blocks(self):
        h = Harness()
        result = h.run(windows_completed=0, last_rotation_epoch=4)
        self.assertEqual(result.rotation_check.outcome, S.FAIL)
        self.assertIsNone(result.panel_result)

    def test_the_compliant_rotation_path_sweeps_and_may_rank(self):
        """The control for the two guards above: the correct idiom is not refused."""
        h = Harness()
        result = h.run(windows_completed=20, last_rotation_epoch=2)
        self.assertEqual(result.rotation_check.outcome, S.PASS)
        self.assertIsNotNone(result.panel_result)
        self.assertTrue(result.may_rank)

    def test_run_all_requires_the_rotation_inputs(self):
        h = Harness()
        with self.assertRaises(TypeError):
            h.harness.run_all(run_context=run_context(),
                              historical=available_resolution())

    def test_a_substituted_seed_derivation_is_refused(self):
        """A derivation that stops keying on the control id is a derivation that
        has been replaced, and five identical seeds must not look like a sweep."""
        h = Harness()
        original = CT.derive_control_seed
        try:
            CT.derive_control_seed = lambda **kwargs: "one-seed-for-all"
            with self.assertRaises(CT.ControlWiringError):
                h.harness.seed_plan(campaign_seed=CAMPAIGN_SEED, windows_completed=0)
        finally:
            CT.derive_control_seed = original
        # Compliant path, with the real derivation restored.
        self.assertEqual(
            len(set(h.harness.seed_plan(campaign_seed=CAMPAIGN_SEED,
                                        windows_completed=0).values())), 5)

    def test_seed_plan_refuses_unusable_rotation_inputs(self):
        h = Harness()
        for kwargs in ({"campaign_seed": "", "windows_completed": 0},
                       {"campaign_seed": CAMPAIGN_SEED, "windows_completed": -1},
                       {"campaign_seed": CAMPAIGN_SEED, "windows_completed": True}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    h.harness.seed_plan(**kwargs)


# =============================================================================
# 3. The panel's mint
# =============================================================================

class TestPanelCarriesProofItWasRun(unittest.TestCase):
    """`ControlPanelResult` had no mint token: one built by hand with an all-PASS
    panel yielded `may_rank=True` with no control ever run."""

    def _real_result(self):
        return Harness().run().panel_result

    def test_the_harness_can_mint_one(self):
        result = self._real_result()
        self.assertIsInstance(result, CT.ControlPanelResult)
        self.assertTrue(result.may_rank)

    def test_a_hand_built_all_pass_panel_result_is_refused(self):
        real = self._real_result()
        with self.assertRaises(CT.ControlPanelForged):
            CT.ControlPanelResult(
                outcomes=real.outcomes, panel=real.panel, historical=real.historical,
                escalation=None, aa_cadence=PASS, definitions_check=PASS,
                observations=real.observations, context=real.context)

    def test_replace_cannot_manufacture_one(self):
        real = self._real_result()
        with self.assertRaises(CT.ControlPanelForged):
            dataclasses.replace(real, aa_cadence=PASS)

    def test_the_second_lock_holds_even_with_the_mint_token(self):
        """Reaching in for the token buys nothing: the outcomes must still follow
        from the observations the result carries."""
        real = self._real_result()
        # Same object, minted the same way — the compliant control for this test.
        CT.ControlPanelResult(
            outcomes=real.outcomes, panel=real.panel, historical=real.historical,
            escalation=real.escalation, aa_cadence=real.aa_cadence,
            definitions_check=real.definitions_check,
            observations=real.observations, context=real.context,
            blocked_reason=real.blocked_reason, mint=CT._PANEL_MINT_TOKEN)
        # Now swap the degraded-negative control's FAILING observation out and
        # claim the outcomes anyway.
        kept = tuple(o for o in real.observations
                     if o.control_id != CT.CONTROL_DEGRADED_NEGATIVE)
        with self.assertRaises(CT.ControlPanelForged) as raised:
            CT.ControlPanelResult(
                outcomes=real.outcomes, panel=real.panel, historical=real.historical,
                escalation=real.escalation, aa_cadence=real.aa_cadence,
                definitions_check=real.definitions_check,
                observations=kept, context=real.context,
                blocked_reason=real.blocked_reason, mint=CT._PANEL_MINT_TOKEN)
        self.assertIn("does not follow from the observations", str(raised.exception))

    def test_a_result_reporting_a_different_historical_resolution_is_refused(self):
        real = self._real_result()
        other = CT.HistoricalWinResolution(
            backend="llama_cpu", available=True, declaration=declaration(),
            durability_outcome=S.PASS,
            check=S.Check(S.PASS, ("a second answer to the same question",)))
        with self.assertRaises(CT.ControlPanelForged):
            CT.ControlPanelResult(
                outcomes=real.outcomes, panel=real.panel, historical=other,
                escalation=real.escalation, aa_cadence=real.aa_cadence,
                definitions_check=real.definitions_check,
                observations=real.observations, context=real.context,
                mint=CT._PANEL_MINT_TOKEN)

    def test_every_passing_control_is_backed_by_a_minted_verdict(self):
        result = self._real_result()
        for observation in result.observations:
            self.assertTrue(observation.ran)
            self.assertIsInstance(observation.verdict, api.Verdict)

    def test_the_journalled_result_carries_the_observations(self):
        result = self._real_result()
        payload = result.to_dict()
        S.canonical_json(payload)
        self.assertEqual(len(payload["observations"]), 5)
        self.assertTrue(payload["may_rank"])


# =============================================================================
# 4. The five controls, scored
# =============================================================================

class TestTheFiveControls(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.h = Harness()
        cls.result = cls.h.run()
        cls.panel = cls.result.panel_result

    def _control_outcome(self, control_id):
        return self.panel.outcome_for(control_id)

    def test_the_positive_control_ranks_above_the_anchor(self):
        outcome = self._control_outcome(CT.CONTROL_POSITIVE)
        self.assertEqual(outcome.check.outcome, S.PASS)
        verdict = self.panel.observations[0].verdict
        self.assertEqual(verdict.effect_resolution, api.EFFECT_IMPROVEMENT)
        self.assertIsNotNone(verdict.rank_key())

    def test_the_neutral_control_does_not_advance(self):
        outcome = self._control_outcome(CT.CONTROL_NEUTRAL)
        self.assertEqual(outcome.check.outcome, S.PASS)
        observation = next(o for o in self.panel.observations
                           if o.control_id == CT.CONTROL_NEUTRAL)
        self.assertNotEqual(observation.verdict.effect_resolution,
                            api.EFFECT_IMPROVEMENT)
        self.assertTrue(observation.abs_effects)

    def test_the_degraded_negative_control_receives_no_speed_rank_at_all(self):
        outcome = self._control_outcome(CT.CONTROL_DEGRADED_NEGATIVE)
        self.assertEqual(outcome.check.outcome, S.PASS)
        observation = next(o for o in self.panel.observations
                           if o.control_id == CT.CONTROL_DEGRADED_NEGATIVE)
        # It really is fast, and it really gets no rank. That is the control.
        self.assertGreater(abs(observation.verdict.effect.value), 0.5)
        with self.assertRaises(api.SpeedRankUnavailable):
            observation.verdict.rank_key()

    def test_the_aa_control_finds_no_significant_effect(self):
        outcome = self._control_outcome(CT.CONTROL_AA)
        self.assertEqual(outcome.check.outcome, S.PASS)
        observation = next(o for o in self.panel.observations
                           if o.control_id == CT.CONTROL_AA)
        self.assertIsNotNone(observation.verdict.effect)
        self.assertNotIn(observation.verdict.effect_resolution,
                         (api.EFFECT_IMPROVEMENT, api.EFFECT_REGRESSION))
        self.assertFalse(self.panel.voids_window)

    def test_the_historical_win_replay_promotes(self):
        outcome = self._control_outcome(CT.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertEqual(outcome.check.outcome, S.PASS)
        observation = next(o for o in self.panel.observations
                           if o.control_id == CT.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertTrue(observation.promoted)
        self.assertEqual(observation.observed_direction, "higher_better")
        self.assertTrue(declaration().reference_band.contains(
            observation.observed_magnitude))

    def test_the_panel_licenses_ranking_and_names_five_of_five(self):
        self.assertTrue(self.panel.may_rank)
        self.assertFalse(self.panel.halts_campaign)
        self.assertEqual(self.panel.gate_defects, ())
        self.assertIn("5/5", self.panel.marker)

    def test_the_window_attestations_projection_is_complete(self):
        fields = CT.window_control_attestations(self.panel)
        self.assertEqual(set(fields),
                         {"controls", "aa_cadence", "control_definitions_immutable"})
        self.assertIsInstance(fields["controls"], api.ControlPanel)

    def test_a_fixture_cannot_declare_itself_promoted(self):
        names = {f.name for f in dataclasses.fields(CR.ControlFixture)}
        self.assertNotIn("promoted", names)
        self.assertFalse(names & {"expected_direction", "expected_outcome", "effect"})

    def test_a_replay_that_does_not_promote_is_a_gate_defect(self):
        """The accept-side control, biting: the same code path, a win that no
        longer crosses. *"A failure to promote is a gate defect, not a research
        finding."*"""
        weak = build_fixtures(replay_effect=0.0005)
        h = Harness(fixtures=weak)
        panel = h.run().panel_result
        observation = next(o for o in panel.observations
                           if o.control_id == CT.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertFalse(observation.promoted)
        self.assertEqual(
            panel.outcome_for(CT.CONTROL_HISTORICAL_WIN_REPLAY).check.outcome, S.FAIL)
        self.assertTrue(panel.halts_campaign)
        self.assertFalse(panel.may_rank)
        self.assertEqual([d.control_id for d in panel.gate_defects],
                         [CT.CONTROL_HISTORICAL_WIN_REPLAY])

    def test_a_positive_control_that_does_not_rank_is_a_gate_defect(self):
        flat = build_fixtures(positive_effect=0.0003)
        h = Harness(fixtures=flat)
        panel = h.run().panel_result
        self.assertEqual(panel.outcome_for(CT.CONTROL_POSITIVE).check.outcome, S.FAIL)
        self.assertIn(CT.CONTROL_POSITIVE, [d.control_id for d in panel.gate_defects])
        self.assertFalse(panel.may_rank)

    def test_an_unavailable_fixture_is_could_not_check_never_a_pass(self):
        fixtures = list(build_fixtures())
        fixtures[1] = _fixture(CT.CONTROL_NEUTRAL, tier="T1", effect=0.0, seed=12,
                               source_tag="neutral-source", blocks=0, available=False,
                               unavailable_reason="the neutral candidate did not build")
        h = Harness(fixtures=tuple(fixtures))
        panel = h.run().panel_result
        outcome = panel.outcome_for(CT.CONTROL_NEUTRAL)
        self.assertEqual(outcome.check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(panel.may_rank)

    def test_the_observations_name_the_runner_the_fixture_and_the_seed(self):
        for observation in self.panel.observations:
            notes = " ".join(observation.notes)
            self.assertIn(CR.CONTROL_RUNNER_ID, notes)
            self.assertIn("fixture=", notes)
            self.assertIn("order_seed=", notes)


# =============================================================================
# 5. The fixtures are under the trust boundary
# =============================================================================

class TestFixturesArePinned(unittest.TestCase):
    """*"Control definitions, FIXTURES, expected directions, and seeds live inside
    the evaluator bundle … and MUST NOT be modified by any process inside the
    loop."* Only the definitions had a pin."""

    def test_the_correct_pin_resolves(self):
        fixtures = build_fixtures()
        digest = S.content_hash(CR._fixture_payload(fixtures))
        resolved = CR.resolve_fixture_set(fixtures=fixtures, pinned_digest=digest,
                                          source_label="bundle@test")
        self.assertEqual(resolved.digest, digest)

    def test_a_wrong_pin_is_refused(self):
        with self.assertRaises(CR.FixtureBundleDrift):
            CR.resolve_fixture_set(fixtures=build_fixtures(),
                                   pinned_digest=sha("some-other-bundle"),
                                   source_label="bundle@test")

    def test_editing_one_sample_moves_the_digest(self):
        original = build_fixtures()
        pin = S.content_hash(CR._fixture_payload(original))
        edited = list(original)
        arms = list(edited[0].anchor_samples)
        arms[0] = tuple(v * 0.5 for v in arms[0])
        edited[0] = dataclasses.replace(edited[0], anchor_samples=tuple(arms))
        with self.assertRaises(CR.FixtureBundleDrift):
            CR.resolve_fixture_set(fixtures=tuple(edited), pinned_digest=pin,
                                   source_label="bundle@test")

    def test_a_supplied_digest_that_does_not_describe_the_contents_is_refused(self):
        with self.assertRaises(CR.FixtureBundleDrift):
            CR.ControlFixtureSet(fixtures=build_fixtures(),
                                 digest=sha("not-derived"), source_label="bundle@test")

    def test_a_missing_fixture_is_a_wiring_error(self):
        fixtures = tuple(f for f in build_fixtures()
                         if f.control_id != CT.CONTROL_AA)
        resolved = fixture_set(fixtures)
        definition = next(d for d in CT.CONTROL_DEFINITIONS
                          if d.control_id == CT.CONTROL_AA)
        with self.assertRaises(CR.FixtureNotDeclared):
            resolved.for_definition(definition)

    def test_a_fixture_answering_for_another_control_is_refused(self):
        """Looked up by the DEFINITION's fixture_id, so a set that answers with
        material the definition does not name is caught rather than hashed."""
        fixtures = list(build_fixtures())
        aa_definition = next(d for d in CT.CONTROL_DEFINITIONS
                             if d.control_id == CT.CONTROL_AA)
        fixtures[0] = dataclasses.replace(fixtures[0],
                                          fixture_id=aa_definition.fixture_id)
        with self.assertRaises(ValueError):
            # Two fixtures now claim the A/A fixture id.
            CR.ControlFixtureSet(
                fixtures=tuple(fixtures),
                digest=S.content_hash(CR._fixture_payload(tuple(fixtures))),
                source_label="bundle@test")

    def test_the_replay_fixture_must_run_at_t2(self):
        with self.assertRaises(ValueError):
            _fixture(CT.CONTROL_HISTORICAL_WIN_REPLAY, tier="T1", effect=0.36,
                     seed=15, source_tag="iqk", blocks=4)

    def test_an_available_fixture_with_no_blocks_is_refused(self):
        with self.assertRaises(ValueError):
            _fixture(CT.CONTROL_POSITIVE, tier="T1", effect=0.3, seed=1,
                     source_tag="x", blocks=0)

    def test_unpaired_arms_are_refused(self):
        good = build_fixtures()[0]
        with self.assertRaises(ValueError):
            dataclasses.replace(good,
                                candidate_samples=good.candidate_samples[:-1])

    def test_blocks_for_needs_the_campaigns_order_schedule_and_split_rule(self):
        fixture = build_fixtures()[0]
        stats = Campaign.get()[6]
        schedule = stats.order_schedule(fixture.candidate_id)
        per_round = stats.stopping_rule.extension.blocks_per_round
        with self.assertRaises(CR.PipelineNotWired):
            fixture.blocks_for(seed="s", schedule=None, split_rule=stats.split_rule,
                               base_blocks=stats.b_min, blocks_per_round=per_round)
        with self.assertRaises(CR.PipelineNotWired):
            fixture.blocks_for(seed="s", schedule=schedule, split_rule=None,
                               base_blocks=stats.b_min, blocks_per_round=per_round)
        with self.assertRaises(CR.PipelineNotWired):
            fixture.blocks_for(seed="s", schedule=schedule,
                               split_rule=stats.split_rule, base_blocks=0,
                               blocks_per_round=per_round)
        # Compliant path: the campaign's own discipline, and it is used.
        blocks = fixture.blocks_for(seed="s", schedule=schedule,
                                    split_rule=stats.split_rule,
                                    base_blocks=stats.b_min,
                                    blocks_per_round=per_round)
        self.assertEqual([b.segment for b in blocks],
                         [ST.SEGMENT_BASE] * stats.b_min
                         + [ST.SEGMENT_EXTENSION] * per_round)
        self.assertEqual([b.order for b in blocks],
                         list(schedule.orders(len(blocks))))
        for block in blocks:
            self.assertEqual(stats.split_rule.assign(block.unit_id), block.stratum)


# =============================================================================
# 6. The A/A arm reaching the calibration solve
# =============================================================================

class TestCalibrationJoin(unittest.TestCase):
    """*"phi is estimated from the A/A control."* What this module emits must be
    what `statistics.CalibrationSolve` consumes — checked by solving with it."""

    @classmethod
    def setUpClass(cls):
        cls.h = Harness()
        cls.result = cls.h.run()

    def test_the_material_is_the_blocks_the_controls_were_measured_on(self):
        material = CR.calibration_material(self.h.runner, self.result)
        aa_fixture = self.h.fixture_set.for_definition(
            next(d for d in CT.CONTROL_DEFINITIONS if d.control_id == CT.CONTROL_AA))
        submitted = next(s for s in self.h.runner.submissions
                         if s.request.candidate_id == aa_fixture.candidate_id)
        self.assertEqual(material["aa_blocks"], submitted.blocks)
        self.assertTrue(material["neutral_blocks"])
        for block in material["aa_blocks"] + material["neutral_blocks"]:
            self.assertIsInstance(block, ST.PairedBlock)

    def _inputs(self, controls_decl):
        _cd, rule, construction, _split, _solve, _outputs, stats = Campaign.get()
        return CR.build_calibration_inputs(
            runner=self.h.runner, result=self.result, binding=self.h.binding,
            campaign_seed=CAMPAIGN_SEED, campaign_controls=controls_decl,
            stopping_rule=rule, construction=construction,
            effect_scale=ST.EFFECT_SCALE_RELATIVE,
            hypothesis=ST.HYPOTHESIS_IMPROVEMENT, margin=0.0,
            anchor_calibration_values=stats.anchor_calibration_values,
            samples_ref=f"ak-raw://{CAMPAIGN}/controls/0001")

    def test_the_solver_consumes_what_this_module_emits(self):
        """The compatibility proof: the solve READS the A/A material and refuses
        on a protocol clause about it, rather than on its shape.

        A sweep's A/A arm is far shorter than a declared 200-block calibration
        block, so this campaign's own declaration cannot be satisfied from one
        window — and *"phi is estimated over AT LEAST the declared count"* is the
        clause that says so. That refusal is the evidence the material arrived
        intact: a shape mismatch would have raised out of `CalibrationInputs`
        before any clause was reached.
        """
        controls_decl = Campaign.get()[0]
        inputs = self._inputs(controls_decl)
        self.assertIsInstance(inputs, ST.CalibrationInputs)
        self.assertEqual(len(inputs.aa_effects()), len(inputs.aa_blocks))
        solve = CT.run_calibration_block(inputs)   # the one door into the solver
        self.assertIsInstance(solve, ST.CalibrationSolve)
        self.assertFalse(solve.accepted)
        self.assertIn("calibration_block_count", " ".join(solve.reasons))

    def test_rotating_a_static_fixture_does_not_produce_a_second_measurement(self):
        """The seed rotates the LABEL. It does not rotate the samples.

        This is what the suite previously called "twelve DISTINCT arms, not one
        arm counted twelve times" and proved by counting `unit_id`s — which is
        counting labels. The arms are byte-identical across epochs, so twelve
        epochs over a static fixture are ten measurements wearing 120 names, and
        `estimate_noise_floor`'s *"phi is estimated over AT LEAST the declared
        count"* would have been satisfied by the names.
        """
        epochs = 12
        h = Harness()
        results = [h.run(windows_completed=w * 10, last_rotation_epoch=w,
                         window_id=f"akw-{w:04d}")
                   for w in range(epochs)]
        for result in results:
            self.assertTrue(result.may_rank)
        aa_fixture = h.fixture_set.for_definition(
            next(d for d in CT.CONTROL_DEFINITIONS if d.control_id == CT.CONTROL_AA))
        arms = [s.blocks for s in h.runner.admitted_submissions
                if s.request.candidate_id == aa_fixture.candidate_id]
        self.assertEqual(len(arms), epochs)
        # The labels differ …
        self.assertEqual(len({b.unit_id for arm in arms for b in arm}),
                         epochs * len(arms[0]))
        # … and the measurements do not.
        self.assertEqual(
            len({(b.anchor_samples, b.candidate_samples) for arm in arms for b in arm}),
            len(arms[0]))
        with self.assertRaises(CR.CalibrationMaterialRelabelled):
            CR.calibration_material(h.runner, results[-1])

    def test_the_pooled_aa_history_solves_and_is_accepted(self):
        """*"Runs periodically on its declared cadence, not once per campaign."*

        The compliant path, and the only one: each window resolves its own
        FRESHLY MEASURED fixture set under its own claim, and the windows' A/A
        arms are pooled. Twelve windows of genuinely distinct material reach the
        declared count and the solve accepts.
        """
        windows = 12
        materials = []
        for w in range(windows):
            # Fresh material per window — a real A/A run, not a replay of one.
            fixtures = build_fixtures(seed_offset=1000 + 10 * w)
            h = Harness(fixtures=fixtures)
            result = h.run(window_id=f"akw-{w:04d}")
            self.assertTrue(result.may_rank)
            materials.append(CR.calibration_material(h.runner, result))
        material = CR.pool_calibration_material(materials)
        blocks_per_window = len(materials[0]["aa_blocks"])
        self.assertEqual(len(material["aa_blocks"]), windows * blocks_per_window)
        # The pool is distinct MEASUREMENTS, checked on the samples themselves.
        self.assertEqual(
            len({(b.anchor_samples, b.candidate_samples)
                 for b in material["aa_blocks"]}),
            windows * blocks_per_window)

        _cd, rule, construction, _split, _s, _o, stats = Campaign.get()
        controls_decl = dataclasses.replace(
            Campaign.get()[0], calibration_block_count=len(material["aa_blocks"]))
        inputs = CR.build_calibration_inputs(
            runner=h.runner, result=result, binding=h.binding,
            campaign_seed=CAMPAIGN_SEED, campaign_controls=controls_decl,
            stopping_rule=rule, construction=construction,
            effect_scale=ST.EFFECT_SCALE_RELATIVE,
            hypothesis=ST.HYPOTHESIS_IMPROVEMENT, margin=0.0,
            anchor_calibration_values=stats.anchor_calibration_values,
            samples_ref=f"ak-raw://{CAMPAIGN}/controls/pooled",
            material=material)
        solve = CT.run_calibration_block(inputs)
        self.assertIsInstance(solve, ST.CalibrationSolve)
        self.assertTrue(solve.attempts, solve.reasons)
        self.assertTrue(solve.accepted, solve.reasons)
        self.assertIsInstance(solve.require_accepted(), api.CalibrationOutputs)
        self.assertGreater(solve.require_accepted().noise_floor_phi, 0.0)
        neutral = CT.neutral_dispersion_check(solve)
        self.assertIsInstance(neutral, S.Check)
        self.assertEqual(neutral.outcome, S.PASS, neutral.reasons)

    def test_a_pool_of_one_window_repeated_is_refused(self):
        """The compliant-path control's mirror: pooling is not multiplication."""
        h = Harness()
        result = h.run()
        one = CR.calibration_material(h.runner, result)
        self.assertTrue(CR.pool_calibration_material([one])["aa_blocks"])
        with self.assertRaises(CR.CalibrationMaterialRelabelled):
            CR.pool_calibration_material([one, one])
        with self.assertRaises(CR.CalibrationMaterialMissing):
            CR.pool_calibration_material([])

    def test_build_calibration_inputs_refuses_relabelled_supplied_material(self):
        h = Harness()
        result = h.run()
        one = CR.calibration_material(h.runner, result)
        doubled = {k: tuple(v) + tuple(
            dataclasses.replace(b, unit_id=b.unit_id + "-copy") for b in v)
            for k, v in one.items()}
        _cd, rule, construction, _split, _s, _o, stats = Campaign.get()
        with self.assertRaises(CR.CalibrationMaterialRelabelled):
            CR.build_calibration_inputs(
                runner=h.runner, result=result, binding=h.binding,
                campaign_seed=CAMPAIGN_SEED, campaign_controls=Campaign.get()[0],
                stopping_rule=rule, construction=construction,
                effect_scale=ST.EFFECT_SCALE_RELATIVE,
                hypothesis=ST.HYPOTHESIS_IMPROVEMENT, margin=0.0,
                anchor_calibration_values=stats.anchor_calibration_values,
                samples_ref="ak-raw://x", material=doubled)

    def test_the_cell_comes_from_the_binding_the_controls_ran_under(self):
        controls_decl, rule, construction, _split, _s, _o, stats = Campaign.get()
        inputs = CR.build_calibration_inputs(
            runner=self.h.runner, result=self.result, binding=self.h.binding,
            campaign_seed=CAMPAIGN_SEED, campaign_controls=controls_decl,
            stopping_rule=rule, construction=construction,
            effect_scale=ST.EFFECT_SCALE_RELATIVE,
            hypothesis=ST.HYPOTHESIS_IMPROVEMENT, margin=0.0,
            anchor_calibration_values=stats.anchor_calibration_values,
            samples_ref="ak-raw://x")
        self.assertEqual(
            (inputs.backend, inputs.phase, inputs.cell_class, inputs.metric_direction),
            (self.h.binding.backend, self.h.binding.phase, self.h.binding.cell_class,
             self.h.binding.metric_direction))

    def test_a_sweep_that_did_not_run_the_aa_control_cannot_calibrate(self):
        fixtures = list(build_fixtures())
        fixtures[3] = _fixture(CT.CONTROL_AA, tier="T1", effect=0.0, seed=14,
                               source_tag="aa-source", blocks=0, available=False,
                               unavailable_reason="the A/A arm did not run")
        h = Harness(fixtures=tuple(fixtures))
        result = h.run()
        with self.assertRaises(CR.CalibrationMaterialMissing):
            CR.calibration_material(h.runner, result)

    def test_calibration_material_refuses_the_wrong_types(self):
        with self.assertRaises(TypeError):
            CR.calibration_material(object(), self.result)
        with self.assertRaises(TypeError):
            CR.calibration_material(self.h.runner, object())


# =============================================================================
# 7. Red-team regressions — each of these passed before the guard it names
# =============================================================================

class TestTheWindowIsLiveNotATemplate(unittest.TestCase):
    """A window's attestations are facts about ONE window.

    `CampaignBinding` used to carry a `window_template`, and `window_for()`
    cloned it for every control of every window. Twelve windows of controls all
    attested window one's `resource_claim_receipt` — a claim released eleven
    windows earlier — with `resource_claim_same_holder` frozen at whatever it was
    when the object was built. Denial 8's *"no inference run OUTSIDE A HELD
    CLAIM"* is not satisfied by a receipt copied from an earlier window.
    """

    @staticmethod
    def _window(receipt, host):
        return dataclasses.replace(window_template(), resource_claim_receipt=receipt,
                                   host_receipt=host)

    def test_each_window_submits_under_its_own_claim_receipt(self):
        h = Harness()
        h.run(windows_completed=0, last_rotation_epoch=0, window_id="akw-0001",
              window=self._window("akclaim-region-0001", "host-A"))
        h.run(windows_completed=1, last_rotation_epoch=0, window_id="akw-0002",
              window=self._window("akclaim-region-0002", "host-B"))
        by_window = {}
        for submission in h.runner.submissions:
            by_window.setdefault(submission.window.resource_claim_receipt,
                                 set()).add(submission.window.host_receipt)
        # THE BITE: with a frozen template this is a single receipt, twice.
        self.assertEqual(by_window,
                         {"akclaim-region-0001": {"host-A"},
                          "akclaim-region-0002": {"host-B"}})

    def test_a_control_cannot_be_assembled_with_no_window_open(self):
        h = Harness()
        definition = CT.CONTROL_DEFINITIONS[0]
        with self.assertRaises(CR.WindowBindingStale):
            h.runner.run_control(definition,
                                 dataclasses.replace(run_context(), seed="a-seed"))

    def test_the_sweep_closes_the_window_even_when_the_sweep_raises(self):
        h = Harness()
        h.run(window_id="akw-0001")
        # The window is closed on the way out, so a control cannot be run
        # between sweeps under the receipts of the one that just finished.
        with self.assertRaises(CR.WindowBindingStale):
            h.runner.run_control(CT.CONTROL_DEFINITIONS[0],
                                 dataclasses.replace(run_context(), seed="a-seed"))

    def test_a_run_context_naming_another_window_is_refused(self):
        h = Harness()
        h.runner.open_window(window_id="akw-0001", window=window_template())
        with self.assertRaises(CR.WindowBindingStale):
            h.runner.run_control(CT.CONTROL_DEFINITIONS[0],
                                 dataclasses.replace(run_context("akw-0009"),
                                                     seed="a-seed"))

    def test_a_window_cannot_be_reopened_with_fresh_receipts(self):
        h = Harness()
        h.runner.open_window(window_id="akw-0001",
                             window=self._window("akclaim-0001", "host-sick"))
        h.runner.close_window()
        with self.assertRaises(CR.WindowBindingStale):
            h.runner.open_window(window_id="akw-0001",
                                 window=self._window("akclaim-0001", "host-healthy"))
        # Compliant path: a NEW window opens.
        h.runner.open_window(window_id="akw-0002",
                             window=self._window("akclaim-0002", "host-healthy"))

    def test_a_window_whose_claim_did_not_open_cannot_be_swept(self):
        """*"No inference run OUTSIDE A HELD CLAIM."*"""
        h = Harness()
        no_claim = dataclasses.replace(
            window_template(),
            resource_claim_open=S.Check(S.FAIL, ("the region claim was not granted",)))
        with self.assertRaises(CR.WindowBindingStale):
            h.runner.open_window(window_id="akw-0001", window=no_claim)
        for outcome in (S.COULD_NOT_CHECK,):
            with self.subTest(outcome=outcome):
                with self.assertRaises(CR.WindowBindingStale):
                    h.runner.open_window(
                        window_id=f"akw-{outcome}",
                        window=dataclasses.replace(
                            window_template(),
                            resource_claim_open=S.Check(outcome, ("claim unknown",))))
        # Compliant path: a window whose claim opened sweeps.
        h.runner.open_window(window_id="akw-0002", window=window_template())

    def test_open_window_has_no_default_window(self):
        h = Harness()
        with self.assertRaises(CR.WindowBindingStale):
            h.runner.open_window(window_id="akw-0001", window=None)

    def test_a_sweep_needs_a_runner_that_can_have_a_window_opened(self):
        class _NoWindow:
            runner_id = "no-window/v1"

            def run_control(self, definition, context):  # pragma: no cover
                raise AssertionError("must not be reached")

        harness = CT.ControlHarness(bundle=bundle(), runner=_NoWindow())
        with self.assertRaises(CR.PipelineNotWired):
            CR.ControlSweep(harness=harness, campaign_seed=CAMPAIGN_SEED)


class TestTheVerdictMustDescribeTheSubmission(unittest.TestCase):
    """`api.Verdict` names no candidate and no event.

    `_observation_from` projected `outcome.verdict` onto the control it happened
    to be running with nothing tying the two together, so a pipeline returning
    another candidate's verdict produced a control observation reporting a
    measurement that control never took. `EffectEstimate.raw_samples` is the one
    field that binds a verdict to the material it was reduced from.
    """

    class _SwapsTheVerdict:
        """Evaluates every submission against the FIRST one's blocks."""

        def __init__(self, inner):
            self._inner = inner
            self.pipeline_id = inner.pipeline_id
            self._first = None

        def evaluate(self, submission):
            if self._first is None:
                self._first = submission
                return self._inner.evaluate(submission)
            return self._inner.evaluate(
                dataclasses.replace(submission, blocks=self._first.blocks))

    def test_a_verdict_reduced_from_other_material_is_refused(self):
        h = Harness()
        h.runner._pipeline = self._SwapsTheVerdict(h.pipeline)
        with self.assertRaises(CR.PipelineNotWired):      # THE BITE
            h.run()

    def test_a_verdict_reduced_from_the_submitted_material_is_projected(self):
        """Compliant-path control."""
        h = Harness()
        result = h.run()
        self.assertTrue(result.may_rank, result.blocked_reason)
        for observation in result.observations:
            self.assertTrue(observation.ran)


class TestCalibrationMaterialReadsTheSweep(unittest.TestCase):
    """`result` was type-checked and then never read.

    The docstring claimed it stopped material being taken out of a BLOCKED sweep.
    It did not: an off-schedule rotation returned `panel_result=None`, and the A/A
    arm came out of `runner.submissions` regardless. A parameter that is validated
    and discarded is a conjunct satisfiable by deleting it.
    """

    def test_a_blocked_sweep_cannot_calibrate(self):
        h = Harness()
        h.run(windows_completed=0, last_rotation_epoch=0, window_id="akw-0000")
        blocked = h.run(windows_completed=100, last_rotation_epoch=0,
                        window_id="akw-0001")
        self.assertIsNone(blocked.panel_result)
        self.assertFalse(blocked.may_rank)
        with self.assertRaises(CR.SweepNotLicensed):     # THE BITE
            CR.calibration_material(h.runner, blocked)

    def test_a_licensed_sweep_still_calibrates(self):
        """Compliant-path control: the guard must not refuse the correct idiom."""
        h = Harness()
        result = h.run()
        material = CR.calibration_material(h.runner, result)
        self.assertTrue(material["aa_blocks"])
        self.assertTrue(material["neutral_blocks"])


class TestOnlyControlsThatRanCalibrate(unittest.TestCase):
    """Blocks the reducer refused as not search-grade were being pooled into φ.

    `run_control` appends the submission BEFORE calling the pipeline, so a
    reduction refused for being below `B_min` — or for an undeclared extension —
    left its blocks in `submissions`, and `calibration_material` read them. φ is
    the floor every candidate is judged against; estimating it from blocks the
    reducer had just rejected is the fail-open one layer down.
    """

    @staticmethod
    def _short_aa():
        fixtures = list(build_fixtures())
        fixtures[3] = _fixture(CT.CONTROL_AA, tier="T1", effect=0.0, seed=14,
                               source_tag="aa-source", blocks=3)
        return Harness(fixtures=tuple(fixtures))

    def test_a_refused_reduction_contributes_no_calibration_material(self):
        h = self._short_aa()
        result = h.run()
        aa = next(o for o in result.observations if o.control_id == CT.CONTROL_AA)
        self.assertFalse(aa.ran)
        self.assertIn("search-grade", aa.could_not_run_reason)
        # The submission was still MADE, and is still on the audit trail …
        self.assertTrue(any(s.request.candidate_id.endswith("aa")
                            for s in h.runner.submissions))
        # … and it is not admitted, so it cannot calibrate. THE BITE: this used
        # to return the three rejected blocks.
        with self.assertRaises(CR.CalibrationMaterialMissing):
            CR.calibration_material(h.runner, result)

    def test_a_control_that_ran_is_admitted(self):
        """Compliant-path control."""
        h = Harness()
        h.run()
        self.assertEqual(len(h.runner.admitted_submissions),
                         len(h.runner.submissions))


class TestTheRotationLedger(unittest.TestCase):
    """`check_rotation` compares two numbers the gated party supplies.

    The per-control seeds are derived from `windows_completed`, so a caller that
    reports the matching `last_rotation_epoch` passes by construction. Freezing
    `windows_completed` therefore runs a whole campaign on one holdout with the
    rotation check reading PASS every window — the exact defect it was written
    for. The sweep's own ledger of served window counts is the one input the
    caller does not supply.
    """

    def test_the_rotation_check_alone_passes_a_frozen_clock(self):
        """The vacuity, stated as a test so it cannot be quietly re-introduced."""
        schedule = CT.SeedRotationSchedule(rotate_every_windows=10, declared_at=NOW)
        for frozen in (0, 7, 9):
            check = schedule.check_rotation(windows_completed=frozen,
                                            last_rotation_epoch=frozen // 10)
            self.assertEqual(check.outcome, S.PASS)

    def test_a_repeated_window_count_is_refused(self):
        h = Harness()
        h.run(windows_completed=5, last_rotation_epoch=0, window_id="akw-0001")
        with self.assertRaises(CR.RotationLedgerViolation):     # THE BITE
            h.run(windows_completed=5, last_rotation_epoch=0, window_id="akw-0002")

    def test_a_rewound_window_count_is_refused(self):
        h = Harness()
        h.run(windows_completed=5, last_rotation_epoch=0, window_id="akw-0001")
        with self.assertRaises(CR.RotationLedgerViolation):
            h.run(windows_completed=4, last_rotation_epoch=0, window_id="akw-0002")

    def test_an_advancing_window_count_sweeps(self):
        """Compliant-path control."""
        h = Harness()
        for w in range(4):
            result = h.run(windows_completed=w, last_rotation_epoch=0,
                           window_id=f"akw-{w:04d}")
            self.assertTrue(result.may_rank, result.blocked_reason)


class TestBlockProvenance(unittest.TestCase):
    """Every control block used to be emitted with `measured_at=None`.

    `PairedBlock.measured_at` is *"what orders confirmation evidence against
    lineage entry"*. Unstamped, a control arm recorded last month and one measured
    under tonight's held claim are the same record, and nothing downstream can
    tell them apart.
    """

    def test_every_emitted_block_carries_the_fixtures_measurement_stamp(self):
        h = Harness()
        h.run()
        stamps = {b.measured_at for s in h.runner.submissions for b in s.blocks}
        self.assertEqual(stamps, {NOW})     # THE BITE: this used to be {None}

    def test_a_fixture_with_an_unorderable_stamp_is_refused(self):
        for bad in ("2026-08-03T12:00:00", "not-a-time", "   "):
            with self.subTest(measured_at=bad):
                with self.assertRaises(ValueError):
                    _fixture(CT.CONTROL_POSITIVE, tier="T1", effect=0.3, seed=1,
                             source_tag="p", blocks=5, measured_at=bad)

    def test_an_orderable_stamp_is_accepted(self):
        """Compliant-path control."""
        fixture = _fixture(CT.CONTROL_POSITIVE, tier="T1", effect=0.3, seed=1,
                           source_tag="p", blocks=5,
                           measured_at="2026-08-04T09:30:00+00:00")
        self.assertEqual(fixture.measured_at, "2026-08-04T09:30:00+00:00")


class TestTheSinglePathAuditFollowsHelpers(unittest.TestCase):
    """The audit walked the ClassDef only.

    A second route to a verdict placed in a module-level helper and called from
    `run_control` was invisible — which is precisely where it would be put by
    anyone working around the check.
    """

    def test_a_second_path_hidden_in_a_module_level_helper_is_caught(self):
        source = Path(CR.__file__).read_text(encoding="utf-8")
        tampered = source.replace(
            "def _definition_for(control_id: str)",
            "def _shortcut(pipeline, request, window):\n"
            "    return pipeline._dispatcher.dispatch(request, window, effect=None)\n"
            "\n"
            "\n"
            "def _definition_for(control_id: str)", 1)
        tampered = tampered.replace(
            "        submission = self._submission_for(fixture, context)",
            "        submission = self._submission_for(fixture, context)\n"
            "        _shortcut(self._pipeline, submission.request, submission.window)", 1)
        self.assertNotEqual(tampered, source)
        check = CR.audit_single_evaluation_path(tampered)
        self.assertEqual(check.outcome, S.FAIL)          # THE BITE
        self.assertIn("_shortcut", " ".join(check.reasons))
        self.assertIn("dispatch", " ".join(check.reasons))

    def test_the_real_module_still_passes(self):
        """Compliant-path control: the widened audit must not refuse the module
        it audits — `DispatchPipeline.evaluate` legitimately calls `dispatch` and
        `reduce_blocks`, and it is not reachable from the runner as a helper."""
        self.assertEqual(CR.audit_single_evaluation_path().outcome, S.PASS)


# =============================================================================
# 8. This module runs nothing and writes nothing
# =============================================================================

class TestNoProcessOrWritePaths(unittest.TestCase):
    """`control_runner.py` assembles submissions. Building and benchmarking are
    other modules', under a held claim; denial 8's *"no inference run OUTSIDE A
    HELD CLAIM"* is satisfiable here only because this module runs nothing."""

    def test_the_module_has_no_write_or_process_paths(self):
        source = Path(CR.__file__).read_text(encoding="utf-8")
        check = api.audit_no_write_or_process_paths(source, module_id=CR.MODULE_ID)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_the_auditor_would_notice_a_process_call(self):
        source = Path(CR.__file__).read_text(encoding="utf-8")
        tampered = source.replace("import ast", "import ast\nimport subprocess")
        self.assertNotEqual(tampered, source)
        self.assertEqual(
            api.audit_no_write_or_process_paths(
                tampered, module_id=CR.MODULE_ID).outcome, S.FAIL)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
