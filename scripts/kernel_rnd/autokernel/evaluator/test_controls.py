#!/usr/bin/env python3
"""Tests for `controls.py` — the five-control harness and the calibration block.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_controls.py
    python3 -W error::ResourceWarning -m unittest \\
        scripts/kernel_rnd/autokernel/evaluator/test_controls.py

WHAT THESE TESTS ARE FOR
------------------------
Each of the five controls is a claim about the EVALUATOR, so each test here is a
test of a test. The three that matter most, and would be the easiest to write
vacuously:

  * the degraded-negative control is checked by calling `Verdict.rank_key()` and
    treating a returned key as the control FAILING — and separately by proving
    that "no rank because the effect was small" is NOT accepted as "the gate caught
    the cheat";
  * the historical-win replay is proved to be unavailable-but-recorded rather than
    silently skipped, and proved to block ranking while the operator's call is
    pending; and
  * the control definitions are proved tamper-evident by actually rebinding the
    module attribute and asserting the resolve refuses.

No test constructs a passing fixture by removing the signal under test: there is
deliberately no `all_pass()` helper, every `WindowAttestations` is built field by
field, and the estimator fixture is named a fixture in its own class name.
"""
from __future__ import annotations

import contextlib
import dataclasses
import tempfile
import unittest
from pathlib import Path

from .. import schemas, storage
from . import api, controls
from . import statistics as ak_statistics

_HERE = str(Path(__file__).resolve().parent)
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _sha(seed: str) -> str:
    return schemas.content_hash({"seed": seed})


def _anchor(seed: str = "anchor") -> api.AnchorIdentity:
    return api.AnchorIdentity(
        source_commit=_COMMIT,
        binary_sha256=_sha(f"{seed}-binary"),
        linkage_sha256=_sha(f"{seed}-linkage"),
        measurement_event_ids=("ake-anchor-001",),
    )


def _gate(gate_id: str, gate_class: str, outcome: str = schemas.PASS,
          *, requires_anchor: bool = False) -> api.GateResult:
    return api.GateResult(
        gate_id=gate_id, gate_class=gate_class,
        check=schemas.Check(outcome, () if outcome == schemas.PASS else ("fixture",)),
        requires_anchor=requires_anchor)


def _effect(value: float, *, e_value: float = 100.0, threshold: float = 20.0,
            mde: float = 0.05, noise_floor: float = 0.02, blocks: int = 10,
            stratum: str = api.STRATUM_SELECTION,
            direction: str = "higher_better") -> api.EffectEstimate:
    return api.EffectEstimate(
        metric="decode_tps", metric_direction=direction, value=value,
        e_value=e_value, threshold=threshold, mde=mde, noise_floor=noise_floor,
        paired_blocks=blocks, stratum=stratum,
        raw_samples=(1.0, 2.0, 3.0), raw_samples_ref="ev-raw-001")


def _search_grade(satisfied: bool = True, failed: tuple = ()) -> api.SearchGradeResult:
    return api.SearchGradeResult(
        satisfied=satisfied,
        evaluated=tuple(c.id for c in api.SEARCH_GRADE_CONJUNCTS),
        failed=failed, not_applicable=(),
        reasons=tuple((cid, ("fixture",)) for cid in failed))


def _void_scan(*reasons: str) -> api.VoidScan:
    findings = tuple(
        api.VoidFinding(reason=reason, protocol_phrase=api.VOID_REASON_PHRASES[reason],
                        outcome=schemas.FAIL, detail=("fixture",))
        for reason in reasons)
    return api.VoidScan(findings=findings, evaluated=api.VOID_REASONS, not_applicable=())


def _verdict(*, tier: str = "T1", gates: tuple = (),
             effect=None, anchor=None, voids: tuple = (),
             search_satisfied: bool = True) -> api.Verdict:
    """Build a real `api.Verdict` through the only constructor that exists."""
    return api.compute_verdict(
        tier=tier,
        gates=gates or (_gate("g-correct", api.GATE_CORRECTNESS),),
        void_scan=_void_scan(*voids),
        search_grade=_search_grade(search_satisfied),
        anchor=_anchor() if anchor is None else anchor,
        effect=effect,
    )


def _cadence(**kwargs) -> controls.AACadence:
    defaults = {"every_n_windows": 5, "every_n_seconds": 3600.0,
                "declared_at": "2026-08-03T00:00:00Z"}
    defaults.update(kwargs)
    return controls.AACadence(**defaults)


def _rotation(**kwargs) -> controls.SeedRotationSchedule:
    defaults = {"rotate_every_windows": 10, "declared_at": "2026-08-03T00:00:00Z"}
    defaults.update(kwargs)
    return controls.SeedRotationSchedule(**defaults)


def _declaration(**kwargs) -> controls.HistoricalWinReplayDeclaration:
    defaults = {
        "win_id": "iqk-prefill-port",
        "backend": "llama_cpu",
        "phase": "prefill",
        "reference_direction": "higher_better",
        "reference_band": controls.ReferenceBand(low=0.30, high=0.45),
        "evidence_locator": "/repo/data/ak-x/iqk.json",
        "durability_class": "carried_in_git",
    }
    defaults.update(kwargs)
    return controls.HistoricalWinReplayDeclaration(**defaults)


def _bundle(**kwargs) -> controls.ControlBundle:
    defaults = {
        "pinned_definitions_digest": controls.CONTROL_DEFINITIONS_DIGEST,
        "aa_cadence": _cadence(),
        "seed_rotation": _rotation(),
        "historical_win_replays": (_declaration(),),
        "source_label": "evaluator-bundle@fixture",
    }
    defaults.update(kwargs)
    return controls.resolve_control_bundle(**defaults)


def _escalation(decision: str = controls.OPERATOR_DECISION_PROCEED_ON_FOUR,
                **kwargs) -> controls.OperatorEscalation:
    defaults = {
        "escalation_ref": "op-esc-2026-08-03-01",
        "raised_at": "2026-08-03T09:00:00Z",
        "decision": decision,
    }
    if decision != controls.OPERATOR_DECISION_PENDING:
        defaults["decided_at"] = "2026-08-03T10:00:00Z"
        defaults["decided_by"] = "operator"
    defaults.update(kwargs)
    return controls.OperatorEscalation(**defaults)


def _available(declaration=None) -> controls.HistoricalWinResolution:
    declaration = declaration or _declaration()
    return controls.HistoricalWinResolution(
        backend=declaration.backend, available=True, declaration=declaration,
        durability_outcome=schemas.PASS,
        check=schemas.Check(schemas.PASS, ("fixture: resolves in-repo",)))


def _unavailable(reason: str = "no entry for this backend",
                 outcome: str = schemas.FAIL) -> controls.HistoricalWinResolution:
    return controls.HistoricalWinResolution(
        backend="llama_cpu", available=False, declaration=None,
        marker=controls.HISTORICAL_REPLAY_UNAVAILABLE,
        check=schemas.Check(outcome, (reason,)))


def _context(**kwargs) -> controls.ControlContext:
    defaults = {
        "campaign_id": "ak-llama_cpu-fixture-20260803",
        "backend": "llama_cpu",
        "phase": "prefill",
        "cell_class": "microbench",
        "window_id": "win-001",
        "historical": _available(),
        "neutral_dispersion": schemas.Check(schemas.PASS, ("fixture",)),
    }
    defaults.update(kwargs)
    return controls.ControlContext(**defaults)


class _FixtureRunner:
    """A fixture, and named one. Runs nothing; hands back prepared observations."""

    runner_id = "ak3-test-fixture-runner/v1"

    def __init__(self, observations: dict):
        self._observations = dict(observations)
        self.calls = []

    def run_control(self, definition, context):
        self.calls.append((definition.control_id, context.window_id))
        try:
            return self._observations[definition.control_id]
        except KeyError:
            raise AssertionError(
                f"fixture runner has no observation for {definition.control_id!r}") from None


def _passing_observation(control_id: str) -> controls.ControlObservation:
    """A control result that satisfies that control's requirement. Built per
    control from a REAL verdict, never from a hand-stamped PASS."""
    if control_id == controls.CONTROL_POSITIVE:
        return controls.ControlObservation(
            control_id=control_id, ran=True,
            verdict=_verdict(tier="T1", effect=_effect(0.30)))
    if control_id == controls.CONTROL_NEUTRAL:
        return controls.ControlObservation(
            control_id=control_id, ran=True, verdict=_verdict(effect=_effect(0.001)),
            abs_effects=(0.001, 0.002))
    if control_id == controls.CONTROL_DEGRADED_NEGATIVE:
        return controls.ControlObservation(
            control_id=control_id, ran=True,
            verdict=_verdict(
                gates=(_gate("g-correct", api.GATE_CORRECTNESS, schemas.FAIL),),
                effect=_effect(0.90)))
    if control_id == controls.CONTROL_AA:
        return controls.ControlObservation(
            control_id=control_id, ran=True, verdict=_verdict(effect=_effect(0.001)),
            abs_effects=(0.001, 0.002))
    return controls.ControlObservation(
        control_id=control_id, ran=True,
        verdict=_verdict(tier="T2", effect=_effect(0.36)), promoted=True,
        observed_magnitude=0.36, observed_direction="higher_better")


def _other_observations(exclude: str) -> tuple:
    return tuple(_passing_observation(cid) for cid in controls.CONTROL_IDS
                 if cid != exclude)


def _all_observations() -> tuple:
    return tuple(_passing_observation(cid) for cid in controls.CONTROL_IDS)


@contextlib.contextmanager
def _rebound_definitions(definitions: tuple):
    """Rebind the module's control definitions, and always put them back."""
    original = controls.CONTROL_DEFINITIONS
    controls.CONTROL_DEFINITIONS = definitions
    try:
        yield
    finally:
        controls.CONTROL_DEFINITIONS = original


# =============================================================================
# The definitions and the hash-pinned bundle
# =============================================================================

class TestControlDefinitions(unittest.TestCase):

    def test_five_controls_in_ordinal_order(self):
        self.assertEqual(tuple(d.control_id for d in controls.CONTROL_DEFINITIONS),
                         controls.CONTROL_IDS)
        self.assertEqual([d.ordinal for d in controls.CONTROL_DEFINITIONS], [1, 2, 3, 4, 5])
        self.assertEqual(len(controls.MANDATORY_CONTROL_IDS), 4)
        self.assertEqual(controls.ACCEPT_SIDE_CONTROL_ID,
                         controls.CONTROL_HISTORICAL_WIN_REPLAY)

    def test_four_test_rejection_and_exactly_one_tests_acceptance(self):
        accepting = [d for d in controls.CONTROL_DEFINITIONS
                     if d.tests_gate_ability_to == controls.TESTS_ACCEPT]
        self.assertEqual([d.control_id for d in accepting],
                         [controls.CONTROL_HISTORICAL_WIN_REPLAY])
        rejecting = [d for d in controls.CONTROL_DEFINITIONS
                     if d.tests_gate_ability_to == controls.TESTS_REJECT]
        self.assertEqual(len(rejecting), 4)

    def test_failure_dispositions_are_the_protocols_own(self):
        by_id = {d.control_id: d for d in controls.CONTROL_DEFINITIONS}
        self.assertEqual(by_id[controls.CONTROL_POSITIVE].failure_disposition,
                         controls.DISPOSITION_GATE_DEFECT)
        self.assertEqual(by_id[controls.CONTROL_HISTORICAL_WIN_REPLAY].failure_disposition,
                         controls.DISPOSITION_GATE_DEFECT)
        self.assertEqual(by_id[controls.CONTROL_AA].failure_disposition,
                         controls.DISPOSITION_VOIDS_WINDOW)
        for cid in (controls.CONTROL_NEUTRAL, controls.CONTROL_DEGRADED_NEGATIVE):
            self.assertEqual(by_id[cid].failure_disposition,
                             controls.DISPOSITION_BLOCKS_RANKING)

    def test_requirements_carry_the_ratified_phrases(self):
        by_id = {d.control_id: d for d in controls.CONTROL_DEFINITIONS}
        self.assertIn("Failure is a gate defect",
                      by_id[controls.CONTROL_POSITIVE].requirement)
        self.assertIn("no speed rank at all",
                      by_id[controls.CONTROL_DEGRADED_NEGATIVE].requirement)
        self.assertIn("VOIDS", by_id[controls.CONTROL_AA].requirement)
        self.assertIn("not once per campaign", by_id[controls.CONTROL_AA].requirement)
        self.assertIn("gate defect, not a research finding",
                      by_id[controls.CONTROL_HISTORICAL_WIN_REPLAY].requirement)

    def test_positive_control_is_pinned_to_t1(self):
        by_id = {d.control_id: d for d in controls.CONTROL_DEFINITIONS}
        self.assertIn("T1", by_id[controls.CONTROL_POSITIVE].required_tiers)

    def test_definition_is_frozen(self):
        definition = controls.CONTROL_DEFINITIONS[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            definition.requirement = "MUST do whatever the actor wants"

    def test_definition_refuses_a_release_tier(self):
        with self.assertRaises(api.TierNotOwned):
            controls.ControlDefinition(
                control_id=controls.CONTROL_POSITIVE, ordinal=1, mandatory=True,
                tests_gate_ability_to=controls.TESTS_REJECT, purpose="p",
                requirement="r", failure_disposition=controls.DISPOSITION_GATE_DEFECT,
                fixture_id="f", expected_direction="d", required_tiers=("T3",))

    def test_digest_is_stable_and_recomputed(self):
        self.assertEqual(controls.CONTROL_DEFINITIONS_DIGEST,
                         controls._current_definitions_digest())
        self.assertEqual(controls.verify_control_definitions().outcome, schemas.PASS)
        self.assertEqual(
            controls.verify_control_definitions(
                controls.CONTROL_DEFINITIONS_DIGEST).outcome, schemas.PASS)

    def test_wrong_pin_fails_and_names_both_digests(self):
        check = controls.verify_control_definitions(_sha("some-other-bundle"))
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("do not match the campaign pin", " ".join(check.reasons))

    def test_unusable_pin_is_could_not_check_not_pass(self):
        self.assertEqual(controls.verify_control_definitions("").outcome,
                         schemas.COULD_NOT_CHECK)
        self.assertEqual(controls.verify_control_definitions(None).outcome, schemas.PASS)


class TestControlBundle(unittest.TestCase):

    def test_resolve_with_the_pin_succeeds(self):
        bundle = _bundle()
        self.assertEqual(bundle.definitions_digest, controls.CONTROL_DEFINITIONS_DIGEST)
        self.assertEqual(len(bundle.definitions), 5)
        self.assertEqual(bundle.definition(controls.CONTROL_AA).ordinal, 4)

    def test_resolve_requires_a_pin(self):
        with self.assertRaises(ValueError):
            _bundle(pinned_definitions_digest="")

    def test_resolve_refuses_a_wrong_pin(self):
        with self.assertRaises(controls.ControlBundleDrift):
            _bundle(pinned_definitions_digest=_sha("not-this-bundle"))

    def test_actor_cannot_alter_a_definition_structurally(self):
        """Rebinding the module attribute is the actor's best move. It fails."""
        tampered = tuple(
            dataclasses.replace(d, requirement="MAY do anything")
            if d.control_id == controls.CONTROL_DEGRADED_NEGATIVE else d
            for d in controls.CONTROL_DEFINITIONS)
        with _rebound_definitions(tampered):
            self.assertNotEqual(controls._current_definitions_digest(),
                                controls.CONTROL_DEFINITIONS_DIGEST)
            with self.assertRaises(controls.ControlBundleDrift):
                _bundle()
        # And the original is restored, so the drift was detected, not persisted.
        self.assertEqual(controls._current_definitions_digest(),
                         controls.CONTROL_DEFINITIONS_DIGEST)

    def test_dropping_a_control_is_refused(self):
        with _rebound_definitions(controls.CONTROL_DEFINITIONS[:4]):
            with self.assertRaises(controls.ControlBundleDrift):
                _bundle()

    def test_bundle_refuses_a_digest_that_does_not_describe_it(self):
        bundle = _bundle()
        with self.assertRaises(controls.ControlBundleDrift):
            dataclasses.replace(bundle, definitions_digest=_sha("lies"))
        with self.assertRaises(controls.ControlBundleDrift):
            dataclasses.replace(bundle, campaign_digest=_sha("lies"))

    def test_campaign_digest_covers_the_declared_bindings(self):
        one = _bundle()
        two = _bundle(aa_cadence=_cadence(every_n_windows=6))
        self.assertEqual(one.definitions_digest, two.definitions_digest)
        self.assertNotEqual(one.campaign_digest, two.campaign_digest)

    def test_campaign_pin_mismatch_is_drift(self):
        with self.assertRaises(controls.ControlBundleDrift):
            _bundle(pinned_campaign_digest=_sha("yesterdays-bindings"))

    def test_reverify_detects_a_post_hoc_binding_change(self):
        bundle = _bundle()
        good = bundle.reverify(
            pinned_definitions_digest=controls.CONTROL_DEFINITIONS_DIGEST,
            pinned_campaign_digest=bundle.campaign_digest)
        self.assertEqual(good.outcome, schemas.PASS)
        bad = bundle.reverify(
            pinned_definitions_digest=controls.CONTROL_DEFINITIONS_DIGEST,
            pinned_campaign_digest=_sha("different"))
        self.assertEqual(bad.outcome, schemas.FAIL)
        self.assertIn("post-hoc change", " ".join(bad.reasons))

    def test_bundle_dict_is_canonicalizable(self):
        schemas.canonical_json(_bundle().to_dict())


class TestSeedRotation(unittest.TestCase):

    def test_seed_is_deterministic_and_epoch_sensitive(self):
        a = controls.derive_control_seed(campaign_seed="s", control_id="aa", epoch=0)
        b = controls.derive_control_seed(campaign_seed="s", control_id="aa", epoch=0)
        c = controls.derive_control_seed(campaign_seed="s", control_id="aa", epoch=1)
        d = controls.derive_control_seed(campaign_seed="s", control_id="neutral", epoch=0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_seed_refuses_an_unknown_control(self):
        with self.assertRaises(ValueError):
            controls.derive_control_seed(campaign_seed="s", control_id="nope", epoch=0)

    def test_schedule_refuses_a_never_rotating_interval(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                _rotation(rotate_every_windows=bad)

    def test_rotation_due_but_not_taken_is_a_coverage_defect(self):
        schedule = _rotation(rotate_every_windows=10)
        self.assertEqual(schedule.epoch_for(0), 0)
        self.assertEqual(schedule.epoch_for(25), 2)
        check = schedule.check_rotation(windows_completed=25, last_rotation_epoch=0)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("evaluator coverage defect", " ".join(check.reasons))
        self.assertEqual(
            schedule.check_rotation(windows_completed=25, last_rotation_epoch=2).outcome,
            schemas.PASS)

    def test_rotating_ahead_of_schedule_is_also_refused(self):
        schedule = _rotation(rotate_every_windows=10)
        check = schedule.check_rotation(windows_completed=5, last_rotation_epoch=3)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("post-hoc change", " ".join(check.reasons))

    def test_bundle_derives_the_seed_for_the_current_epoch(self):
        bundle = _bundle(seed_rotation=_rotation(rotate_every_windows=4))
        first = bundle.seed_for(campaign_seed="cs", control_id="aa", windows_completed=3)
        second = bundle.seed_for(campaign_seed="cs", control_id="aa", windows_completed=4)
        self.assertNotEqual(first, second)


# =============================================================================
# The A/A periodic scheduling contract
# =============================================================================

def _ledger_entry(**kwargs) -> controls.AALedgerEntry:
    defaults = {"window_id": "win-aa-1", "ran_at_epoch_seconds": 1000.0,
                "windows_completed_at_run": 0, "outcome": schemas.PASS,
                "anchor_short": "abc/def/ghi"}
    defaults.update(kwargs)
    return controls.AALedgerEntry(**defaults)


class TestAACadence(unittest.TestCase):

    def test_cadence_refuses_unbounded_or_zero_intervals(self):
        for kwargs in ({"every_n_windows": 0}, {"every_n_seconds": 0.0},
                       {"every_n_seconds": float("inf")}):
            with self.assertRaises(ValueError):
                _cadence(**kwargs)

    def test_cadence_cannot_decline_the_boundary_triggers(self):
        with self.assertRaises(ValueError):
            _cadence(at_campaign_boundary=False)
        with self.assertRaises(ValueError):
            _cadence(on_anchor_identity_change=False)

    def test_empty_ledger_is_due_and_fails_the_attestation(self):
        scheduler = controls.AAScheduler(_cadence())
        decision = scheduler.due(ledger=(), windows_completed=0,
                                 now_epoch_seconds=1000.0, anchor_short="a/b/c")
        self.assertTrue(decision.due)
        self.assertIn("not once per campaign", " ".join(decision.reasons))
        check = scheduler.check(ledger=(), windows_completed=0,
                                now_epoch_seconds=1000.0, anchor_short="a/b/c")
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_within_cadence_passes(self):
        scheduler = controls.AAScheduler(_cadence(every_n_windows=5,
                                                  every_n_seconds=3600.0))
        ledger = (_ledger_entry(anchor_short="a/b/c"),)
        check = scheduler.check(ledger=ledger, windows_completed=2,
                                now_epoch_seconds=2000.0, anchor_short="a/b/c")
        self.assertEqual(check.outcome, schemas.PASS)

    def test_window_count_trigger(self):
        scheduler = controls.AAScheduler(_cadence(every_n_windows=5))
        ledger = (_ledger_entry(anchor_short="a/b/c"),)
        decision = scheduler.due(ledger=ledger, windows_completed=5,
                                 now_epoch_seconds=1001.0, anchor_short="a/b/c")
        self.assertTrue(decision.due)
        self.assertEqual(decision.windows_since_last, 5)

    def test_wall_clock_trigger_fires_even_with_few_windows(self):
        scheduler = controls.AAScheduler(_cadence(every_n_windows=100,
                                                  every_n_seconds=60.0))
        ledger = (_ledger_entry(anchor_short="a/b/c"),)
        decision = scheduler.due(ledger=ledger, windows_completed=1,
                                 now_epoch_seconds=1000.0 + 61.0, anchor_short="a/b/c")
        self.assertTrue(decision.due)
        self.assertIn("since the last A/A, cadence is every", " ".join(decision.reasons))

    def test_anchor_identity_change_makes_it_due(self):
        scheduler = controls.AAScheduler(_cadence())
        ledger = (_ledger_entry(anchor_short="a/b/c"),)
        decision = scheduler.due(ledger=ledger, windows_completed=0,
                                 now_epoch_seconds=1001.0, anchor_short="x/y/z")
        self.assertTrue(decision.due)
        self.assertIn("rebuilt anchor is a different anchor", " ".join(decision.reasons))

    def test_campaign_boundary_makes_it_due(self):
        scheduler = controls.AAScheduler(_cadence())
        ledger = (_ledger_entry(anchor_short="a/b/c"),)
        decision = scheduler.due(ledger=ledger, windows_completed=0,
                                 now_epoch_seconds=1001.0, anchor_short="a/b/c",
                                 campaign_boundary=True)
        self.assertTrue(decision.due)

    def test_last_aa_failed_fails_the_attestation_even_within_cadence(self):
        scheduler = controls.AAScheduler(_cadence())
        ledger = (_ledger_entry(outcome=schemas.FAIL, anchor_short="a/b/c"),)
        check = scheduler.check(ledger=ledger, windows_completed=0,
                                now_epoch_seconds=1001.0, anchor_short="a/b/c")
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("voids the enclosing measurement window", " ".join(check.reasons))

    def test_last_aa_could_not_check_is_not_reported_as_a_failure(self):
        scheduler = controls.AAScheduler(_cadence())
        ledger = (_ledger_entry(outcome=schemas.COULD_NOT_CHECK, anchor_short="a/b/c"),)
        check = scheduler.check(ledger=ledger, windows_completed=0,
                                now_epoch_seconds=1001.0, anchor_short="a/b/c")
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("not the same as a failed A/A", " ".join(check.reasons))

    def test_drift_exposure_is_advisory_and_counts_windows(self):
        scheduler = controls.AAScheduler(_cadence())
        ledger = (
            _ledger_entry(window_id="aa-1", windows_completed_at_run=2,
                          outcome=schemas.PASS, anchor_short="a/b/c"),
            _ledger_entry(window_id="aa-2", windows_completed_at_run=7,
                          outcome=schemas.FAIL, anchor_short="a/b/c"),
        )
        exposure = scheduler.drift_exposure(ledger=ledger, windows_completed=9)
        self.assertEqual(exposure["label"], "advisory")
        self.assertEqual(exposure["windows_since_last_passing_aa"], 7)
        self.assertEqual(exposure["last_passing_aa_window_id"], "aa-1")
        self.assertIn("operator decision", exposure["authority_note"])

    def test_drift_exposure_with_no_passing_aa(self):
        scheduler = controls.AAScheduler(_cadence())
        exposure = scheduler.drift_exposure(ledger=(), windows_completed=4)
        self.assertEqual(exposure["windows_since_last_passing_aa"], 4)
        self.assertIsNone(exposure["last_passing_aa_window_id"])

    def test_unreadable_ledger_raises_rather_than_degrading(self):
        scheduler = controls.AAScheduler(_cadence())
        with self.assertRaises(TypeError):
            scheduler.check(ledger=[{"window_id": "aa"}], windows_completed=0,
                            now_epoch_seconds=1.0, anchor_short="a/b/c")
        with self.assertRaises(TypeError):
            scheduler.due(ledger="aa", windows_completed=0, now_epoch_seconds=1.0,
                          anchor_short="a/b/c")

    def test_ledger_entry_refuses_a_non_check_outcome(self):
        with self.assertRaises(ValueError):
            _ledger_entry(outcome="OK")


# =============================================================================
# Control 5 — the declared contract, resolved at run time
# =============================================================================

class TestHistoricalWinReplayContract(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="_ak_controls_test_", dir=_HERE)
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "epyc-inference-research"
        (self.repo / "data" / "ak-x").mkdir(parents=True)
        self.tracked_path = self.repo / "data" / "ak-x" / "iqk.json"
        self.tracked_path.write_text('{"win": "iqk"}', encoding="utf-8")
        self.untracked_path = self.repo / "data" / "ak-x" / "loose.json"
        self.untracked_path.write_text("{}", encoding="utf-8")
        self.index = storage.StaticTrackedIndex(self.repo, ["data/ak-x/iqk.json"])

    def test_declaration_carries_exactly_the_protocols_fields(self):
        declaration = _declaration()
        self.assertEqual(
            set(declaration.to_dict()),
            {"win_id", "backend", "phase", "reference_direction", "reference_band",
             "evidence_locator", "durability_class", "evidence_sha256",
             "evidence_provenance"})

    def test_declaration_refuses_an_unclassified_citation(self):
        with self.assertRaises(ValueError):
            _declaration(durability_class="probably_fine")

    def test_declaration_refuses_an_unknown_backend_or_direction(self):
        with self.assertRaises(ValueError):
            _declaration(backend="llama_tpu")
        with self.assertRaises(ValueError):
            _declaration(reference_direction="bigger_is_nicer")

    def test_reference_band_must_be_an_interval(self):
        with self.assertRaises(ValueError):
            controls.ReferenceBand(low=0.4, high=0.4)
        with self.assertRaises(ValueError):
            controls.ReferenceBand(low=-0.1, high=0.4)
        band = controls.ReferenceBand(low=0.3, high=0.45)
        self.assertTrue(band.contains(0.3))
        self.assertTrue(band.contains(0.45))
        self.assertFalse(band.contains(0.46))

    def test_parse_returns_reasons_instead_of_raising(self):
        declaration, reasons = controls.HistoricalWinReplayDeclaration.parse("nope")
        self.assertIsNone(declaration)
        self.assertTrue(reasons)
        declaration, reasons = controls.HistoricalWinReplayDeclaration.parse(
            {"win_id": "w", "backend": "llama_cpu"})
        self.assertIsNone(declaration)
        self.assertTrue(reasons)
        good = {
            "win_id": "w", "backend": "llama_cpu", "phase": "prefill",
            "reference_direction": "higher_better",
            "reference_band": {"low": 0.3, "high": 0.4},
            "evidence_locator": "/repo/data/x.json",
            "durability_class": "carried_in_git",
        }
        declaration, reasons = controls.HistoricalWinReplayDeclaration.parse(good)
        self.assertEqual(reasons, ())
        self.assertEqual(declaration.win_id, "w")

    def test_resolution_requires_a_tracked_index(self):
        with self.assertRaises(controls.ControlsError):
            controls.resolve_historical_win_replay(
                declarations=(_declaration(),), backend="llama_cpu", tracked_index=None)

    def test_no_entry_for_the_backend_is_unavailable_and_named(self):
        resolution = controls.resolve_historical_win_replay(
            declarations=(_declaration(backend="llama_gpu"),),
            backend="llama_cpu", tracked_index=self.index)
        self.assertFalse(resolution.available)
        self.assertEqual(resolution.marker, controls.HISTORICAL_REPLAY_UNAVAILABLE)
        self.assertEqual(resolution.check.outcome, schemas.FAIL)
        self.assertIn("llama_cpu", resolution.reason())

    def test_two_entries_for_one_backend_is_refused_not_arbitrated(self):
        resolution = controls.resolve_historical_win_replay(
            declarations=(_declaration(win_id="a"), _declaration(win_id="b")),
            backend="llama_cpu", tracked_index=self.index)
        self.assertFalse(resolution.available)
        self.assertIn("the contract names one supplier", resolution.reason())

    def test_evidence_that_resolves_in_repo_is_available(self):
        resolution = controls.resolve_historical_win_replay(
            declarations=(_declaration(evidence_locator=str(self.tracked_path)),),
            backend="llama_cpu", tracked_index=self.index)
        self.assertTrue(resolution.available)
        self.assertEqual(resolution.check.outcome, schemas.PASS)
        self.assertEqual(resolution.durability_outcome, schemas.PASS)

    def test_evidence_git_does_not_carry_is_unavailable(self):
        resolution = controls.resolve_historical_win_replay(
            declarations=(_declaration(evidence_locator=str(self.untracked_path)),),
            backend="llama_cpu", tracked_index=self.index)
        self.assertFalse(resolution.available)
        self.assertEqual(resolution.check.outcome, schemas.FAIL)
        self.assertEqual(resolution.marker, controls.HISTORICAL_REPLAY_UNAVAILABLE)
        self.assertIn("does not resolve in-repo", resolution.reason())

    def test_a_scratch_citation_is_unavailable(self):
        resolution = controls.resolve_historical_win_replay(
            declarations=(_declaration(evidence_locator="/mnt/raid0/llm/tmp/iqk.json"),),
            backend="llama_cpu", tracked_index=self.index)
        self.assertFalse(resolution.available)
        self.assertEqual(resolution.check.outcome, schemas.FAIL)
        self.assertIn("scratch", resolution.reason().lower())

    def test_unanswerable_durability_is_could_not_check_not_fail(self):
        """Fail closed, but never conflated: unavailable-because-unknown is its own
        outcome, and the record keeps it distinguishable from a real absence."""
        outside = Path(self._tmp.name) / "elsewhere.json"
        outside.write_text("{}", encoding="utf-8")
        resolution = controls.resolve_historical_win_replay(
            declarations=(_declaration(evidence_locator=str(outside),
                                       durability_class="hash_and_provenance_only",
                                       evidence_sha256=_sha("x"),
                                       evidence_provenance="fixture"),),
            backend="llama_cpu", tracked_index=self.index)
        self.assertIn(resolution.check.outcome, (schemas.PASS, schemas.COULD_NOT_CHECK,
                                                 schemas.FAIL))
        if resolution.check.outcome == schemas.COULD_NOT_CHECK:
            self.assertFalse(resolution.available)
            self.assertEqual(resolution.marker, controls.HISTORICAL_REPLAY_UNAVAILABLE)

    def test_durable_untracked_inside_the_tree_resolves(self):
        resolution = controls.resolve_historical_win_replay(
            declarations=(_declaration(evidence_locator=str(self.untracked_path),
                                       durability_class="durable_untracked"),),
            backend="llama_cpu", tracked_index=self.index)
        self.assertTrue(resolution.available)

    def test_unavailable_resolution_must_carry_the_marker(self):
        with self.assertRaises(ValueError):
            controls.HistoricalWinResolution(
                backend="llama_cpu", available=False, declaration=None,
                check=schemas.Check(schemas.FAIL, ("no entry",)))

    def test_available_resolution_must_carry_its_declaration(self):
        with self.assertRaises(ValueError):
            controls.HistoricalWinResolution(
                backend="llama_cpu", available=True, declaration=None,
                check=schemas.Check(schemas.PASS))


class TestOperatorEscalation(unittest.TestCase):

    def test_a_decision_must_name_who_took_it_and_when(self):
        with self.assertRaises(ValueError):
            controls.OperatorEscalation(
                escalation_ref="op-1", raised_at="t",
                decision=controls.OPERATOR_DECISION_PROCEED_ON_FOUR)
        with self.assertRaises(ValueError):
            controls.OperatorEscalation(
                escalation_ref="op-1", raised_at="t",
                decision=controls.OPERATOR_DECISION_HALT, decided_at="t2")

    def test_pending_is_a_real_state(self):
        escalation = _escalation(controls.OPERATOR_DECISION_PENDING)
        self.assertIsNone(escalation.decided_by)

    def test_unknown_decision_is_refused(self):
        with self.assertRaises(ValueError):
            controls.OperatorEscalation(escalation_ref="op-1", raised_at="t",
                                        decision="probably_fine")


# =============================================================================
# Per-control evaluation — one test per way each control can be wrong
# =============================================================================

class TestPositiveControl(unittest.TestCase):

    def _evaluate_one(self, observation, context=None):
        harness = controls.ControlHarness(bundle=_bundle(), runner=_FixtureRunner({}))
        result = harness.evaluate(
            observations=(observation,) + _other_observations(observation.control_id),
            context=context or _context(),
            aa_cadence=schemas.Check(schemas.PASS))
        return result.outcome_for(observation.control_id)

    def test_improvement_at_t1_passes(self):
        verdict = _verdict(tier="T1", effect=_effect(0.30))
        self.assertTrue(verdict.speed_rank_admissible)
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_POSITIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.PASS)
        self.assertEqual(outcome.disposition, controls.DISPOSITION_SATISFIED)

    def test_no_rank_is_a_gate_defect(self):
        verdict = _verdict(tier="T1", effect=_effect(0.001))  # below the noise floor
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_POSITIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertEqual(outcome.disposition, controls.DISPOSITION_GATE_DEFECT)
        self.assertIn("Failure is a gate defect", " ".join(outcome.check.reasons))

    def test_wrong_tier_fails(self):
        verdict = _verdict(tier="T2", effect=_effect(0.30))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_POSITIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertIn("ran at tier 'T2'", " ".join(outcome.check.reasons))

    def test_a_void_window_is_could_not_check_not_a_defect(self):
        verdict = _verdict(tier="T1", effect=_effect(0.30),
                           voids=(api.VOID_HOST_HEALTH_TIER_VIOLATION,))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_POSITIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("says nothing whatever about the gate",
                      " ".join(outcome.check.reasons))

    def test_not_run_is_could_not_check_with_its_reason(self):
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_POSITIVE, ran=False,
            could_not_run_reason="build failed"))
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(outcome.disposition, controls.DISPOSITION_GATE_DEFECT)
        self.assertIn("build failed", " ".join(outcome.check.reasons))


class TestNeutralControl(unittest.TestCase):

    def _evaluate_one(self, observation, context=None):
        harness = controls.ControlHarness(bundle=_bundle(), runner=_FixtureRunner({}))
        result = harness.evaluate(
            observations=(observation,) + _other_observations(observation.control_id),
            context=context or _context(),
            aa_cadence=schemas.Check(schemas.PASS))
        return result.outcome_for(observation.control_id)

    def test_not_advancing_passes(self):
        verdict = _verdict(effect=_effect(0.001))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_NEUTRAL, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.PASS)

    def test_spurious_advance_fails(self):
        verdict = _verdict(effect=_effect(0.30))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_NEUTRAL, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertEqual(outcome.disposition, controls.DISPOSITION_BLOCKS_RANKING)
        self.assertIn("ADVANCED", " ".join(outcome.check.reasons))

    def test_dispersion_failure_is_reported_as_a_calibration_failure(self):
        verdict = _verdict(effect=_effect(0.001))
        context = _context(neutral_dispersion=schemas.Check(
            schemas.FAIL, ("p95 0.9 exceeds phi 0.1",)))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_NEUTRAL, ran=True, verdict=verdict), context)
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertIn("rather than raising the floor", " ".join(outcome.check.reasons))

    def test_unchecked_dispersion_is_could_not_check(self):
        verdict = _verdict(effect=_effect(0.001))
        context = _context(neutral_dispersion=schemas.Check(
            schemas.COULD_NOT_CHECK, ("no neutral samples",)))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_NEUTRAL, ran=True, verdict=verdict), context)
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)

    def test_rejected_for_an_unrelated_reason_does_not_count_as_discrimination(self):
        verdict = _verdict(
            gates=(_gate("g-correct", api.GATE_CORRECTNESS, schemas.FAIL),),
            effect=_effect(0.001))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_NEUTRAL, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("has not tested the discrimination", " ".join(outcome.check.reasons))


class TestDegradedNegativeControl(unittest.TestCase):

    def _evaluate_one(self, observation, context=None):
        harness = controls.ControlHarness(bundle=_bundle(), runner=_FixtureRunner({}))
        result = harness.evaluate(
            observations=(observation,) + _other_observations(observation.control_id),
            context=context or _context(),
            aa_cadence=schemas.Check(schemas.PASS))
        return result.outcome_for(observation.control_id)

    def test_caught_by_a_correctness_gate_passes(self):
        verdict = _verdict(
            gates=(_gate("g-correct", api.GATE_CORRECTNESS, schemas.FAIL),
                   _gate("g-perf", api.GATE_PERFORMANCE)),
            effect=_effect(0.90))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_DEGRADED_NEGATIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.PASS)

    def test_receiving_a_speed_rank_fails(self):
        """The check calls the exact method a ranking loop calls."""
        verdict = _verdict(effect=_effect(0.90))
        self.assertIsNotNone(verdict.rank_key())  # the leak this control must catch
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_DEGRADED_NEGATIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertIn("RECEIVED a speed rank", " ".join(outcome.check.reasons))

    def test_unranked_only_because_the_effect_was_small_is_not_a_catch(self):
        """A cheating candidate that happens to measure slow is not a caught one."""
        verdict = _verdict(effect=_effect(0.001))
        self.assertFalse(verdict.speed_rank_admissible)
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_DEGRADED_NEGATIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertIn("NO correctness-class gate rejected it",
                      " ".join(outcome.check.reasons))

    def test_could_not_check_gate_is_not_a_rejection(self):
        verdict = _verdict(
            gates=(_gate("g-correct", api.GATE_CORRECTNESS, schemas.COULD_NOT_CHECK),),
            effect=_effect(0.90))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_DEGRADED_NEGATIVE, ran=True, verdict=verdict))
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("COULD_NOT_CHECK is not a rejection",
                      " ".join(outcome.check.reasons))

    def test_a_void_window_excuses_an_absent_rank_but_not_a_present_one(self):
        void_verdict = _verdict(effect=_effect(0.90),
                                voids=(api.VOID_CLAIM_NOT_HELD,))
        outcome = self._evaluate_one(controls.ControlObservation(
            control_id=controls.CONTROL_DEGRADED_NEGATIVE, ran=True,
            verdict=void_verdict))
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)


class TestAAControl(unittest.TestCase):

    def _result(self, observation, context=None, aa_cadence=None):
        harness = controls.ControlHarness(bundle=_bundle(), runner=_FixtureRunner({}))
        return harness.evaluate(
            observations=(observation,) + _other_observations(observation.control_id),
            context=context or _context(),
            aa_cadence=aa_cadence or schemas.Check(schemas.PASS))

    def test_no_detectable_difference_passes(self):
        verdict = _verdict(effect=_effect(0.001))
        result = self._result(controls.ControlObservation(
            control_id=controls.CONTROL_AA, ran=True, verdict=verdict))
        self.assertEqual(result.outcome_for(controls.CONTROL_AA).check.outcome,
                         schemas.PASS)
        self.assertFalse(result.voids_window)

    def test_a_significant_effect_fails_and_voids_the_window(self):
        verdict = _verdict(effect=_effect(0.30))
        result = self._result(controls.ControlObservation(
            control_id=controls.CONTROL_AA, ran=True, verdict=verdict))
        outcome = result.outcome_for(controls.CONTROL_AA)
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertEqual(outcome.disposition, controls.DISPOSITION_VOIDS_WINDOW)
        self.assertTrue(result.voids_window)
        self.assertFalse(result.may_rank)

    def test_a_regression_also_fails(self):
        verdict = _verdict(effect=_effect(-0.30))
        result = self._result(controls.ControlObservation(
            control_id=controls.CONTROL_AA, ran=True, verdict=verdict))
        self.assertEqual(result.outcome_for(controls.CONTROL_AA).check.outcome,
                         schemas.FAIL)

    def test_an_invalid_aa_is_could_not_check_and_still_voids(self):
        verdict = _verdict(effect=_effect(0.001), search_satisfied=False)
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        result = self._result(controls.ControlObservation(
            control_id=controls.CONTROL_AA, ran=True, verdict=verdict))
        self.assertEqual(result.outcome_for(controls.CONTROL_AA).check.outcome,
                         schemas.COULD_NOT_CHECK)
        self.assertTrue(result.voids_window)

    def test_aa_failing_is_never_a_gate_defect(self):
        verdict = _verdict(effect=_effect(0.30))
        result = self._result(controls.ControlObservation(
            control_id=controls.CONTROL_AA, ran=True, verdict=verdict))
        self.assertEqual(result.gate_defects, ())
        self.assertFalse(result.halts_campaign)


class TestHistoricalReplayControl(unittest.TestCase):

    def _result(self, observation=None, context=None, escalation=None):
        harness = controls.ControlHarness(bundle=_bundle(), runner=_FixtureRunner({}))
        observations = _other_observations(controls.CONTROL_HISTORICAL_WIN_REPLAY)
        if observation is not None:
            observations = observations + (observation,)
        return harness.evaluate(
            observations=observations, context=context or _context(),
            aa_cadence=schemas.Check(schemas.PASS), escalation=escalation)

    def _promoted(self, **kwargs) -> controls.ControlObservation:
        defaults = {
            "control_id": controls.CONTROL_HISTORICAL_WIN_REPLAY,
            "ran": True,
            "verdict": _verdict(tier="T2", effect=_effect(0.36)),
            "promoted": True,
            "observed_magnitude": 0.36,
            "observed_direction": "higher_better",
        }
        defaults.update(kwargs)
        return controls.ControlObservation(**defaults)

    def test_a_promoting_replay_passes_and_the_panel_is_five_of_five(self):
        result = self._result(self._promoted())
        self.assertEqual(
            result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY).check.outcome,
            schemas.PASS)
        self.assertEqual(result.marker, "5/5")
        self.assertTrue(result.may_rank)
        self.assertFalse(result.halts_campaign)

    def test_failure_to_promote_is_a_gate_defect_that_halts(self):
        result = self._result(self._promoted(promoted=False))
        outcome = result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertEqual(outcome.disposition, controls.DISPOSITION_GATE_DEFECT)
        self.assertIn("did NOT promote", " ".join(outcome.check.reasons))
        self.assertTrue(result.halts_campaign)
        self.assertEqual([d.control_id for d in result.gate_defects],
                         [controls.CONTROL_HISTORICAL_WIN_REPLAY])
        self.assertFalse(result.may_rank)

    def test_promoting_outside_the_declared_band_fails(self):
        result = self._result(self._promoted(observed_magnitude=0.90,
                                             verdict=_verdict(tier="T2",
                                                              effect=_effect(0.90))))
        outcome = result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertIn("outside its declared reference band",
                      " ".join(outcome.check.reasons))

    def test_promoting_in_the_wrong_direction_fails(self):
        result = self._result(self._promoted(observed_direction="lower_better"))
        self.assertEqual(
            result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY).check.outcome,
            schemas.FAIL)

    def test_no_magnitude_is_could_not_check_not_pass(self):
        result = self._result(self._promoted(observed_magnitude=None))
        self.assertEqual(
            result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY).check.outcome,
            schemas.COULD_NOT_CHECK)

    def test_an_observation_must_say_whether_it_promoted(self):
        with self.assertRaises(ValueError):
            controls.ControlObservation(
                control_id=controls.CONTROL_HISTORICAL_WIN_REPLAY, ran=True,
                verdict=_verdict(tier="T2", effect=_effect(0.36)))

    # -- the unavailable branch -------------------------------------------------

    def test_unavailable_without_escalation_blocks_ranking(self):
        result = self._result(context=_context(historical=_unavailable()))
        self.assertIsNone(result.panel)
        self.assertFalse(result.may_rank)
        self.assertIn("the operator's call", result.blocked_reason)
        outcome = result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertIn(controls.HISTORICAL_REPLAY_UNAVAILABLE,
                      " ".join(outcome.check.reasons))

    def test_unavailable_and_pending_blocks_ranking(self):
        result = self._result(
            context=_context(historical=_unavailable()),
            escalation=_escalation(controls.OPERATOR_DECISION_PENDING))
        self.assertIsNone(result.panel)
        self.assertFalse(result.may_rank)
        self.assertIn("pending", result.blocked_reason)

    def test_unavailable_and_halted_halts_the_campaign(self):
        result = self._result(
            context=_context(historical=_unavailable()),
            escalation=_escalation(controls.OPERATOR_DECISION_HALT))
        self.assertTrue(result.halts_campaign)
        self.assertFalse(result.may_rank)

    def test_operator_authorised_four_controls_marks_every_record(self):
        result = self._result(
            context=_context(historical=_unavailable("no durable win for llama_cpu")),
            escalation=_escalation(controls.OPERATOR_DECISION_PROCEED_ON_FOUR))
        self.assertIsNotNone(result.panel)
        self.assertEqual(result.marker,
                         f"4/5 ({controls.HISTORICAL_REPLAY_UNAVAILABLE})")
        self.assertTrue(result.may_rank)
        self.assertEqual(result.panel.check_5().outcome, schemas.PASS)
        self.assertIn("no durable win for llama_cpu",
                      result.panel.historical_replay_unavailable_reason)
        self.assertEqual(result.panel.operator_escalation_ref, "op-esc-2026-08-03-01")

    def test_unavailable_is_never_counted_as_a_gate_defect(self):
        result = self._result(
            context=_context(historical=_unavailable()),
            escalation=_escalation(controls.OPERATOR_DECISION_PROCEED_ON_FOUR))
        self.assertEqual(result.gate_defects, ())

    def test_running_a_control_resolved_as_unavailable_is_a_wiring_error(self):
        harness = controls.ControlHarness(bundle=_bundle(), runner=_FixtureRunner({}))
        with self.assertRaises(controls.ControlWiringError):
            controls._evaluate_historical(
                harness.bundle.definition(controls.CONTROL_HISTORICAL_WIN_REPLAY),
                self._promoted(),
                _context(historical=_unavailable()))


# =============================================================================
# The harness
# =============================================================================

def _run_context(**kwargs) -> controls.ControlRunContext:
    defaults = {
        "campaign_id": "ak-llama_cpu-fixture-20260803",
        "backend": "llama_cpu",
        "phase": "prefill",
        "cell_class": "microbench",
        "window_id": "win-001",
        "tier": "T1",
        "seed": _sha("seed"),
        "anchor": _anchor(),
    }
    defaults.update(kwargs)
    return controls.ControlRunContext(**defaults)


class TestControlHarness(unittest.TestCase):

    def _harness(self, observations=None):
        runner = _FixtureRunner(
            observations if observations is not None
            else {cid: _passing_observation(cid) for cid in controls.CONTROL_IDS})
        return controls.ControlHarness(bundle=_bundle(), runner=runner), runner

    def test_a_runner_is_mandatory(self):
        with self.assertRaises(controls.ControlWiringError):
            controls.ControlHarness(bundle=_bundle(), runner=None)
        with self.assertRaises(controls.ControlWiringError):
            controls.ControlHarness(bundle=_bundle(), runner=object())

    def test_run_all_calls_the_runner_once_per_control(self):
        harness, runner = self._harness()
        observations = harness.run_all(run_context=_run_context(),
                                       historical=_available(),
                                       campaign_seed="ak-seed-fixture",
                                       windows_completed=0)
        self.assertEqual(len(observations), 5)
        self.assertEqual([cid for cid, _ in runner.calls], list(controls.CONTROL_IDS))

    def test_run_all_skips_control_5_when_it_is_unavailable(self):
        harness, runner = self._harness()
        observations = harness.run_all(run_context=_run_context(),
                                       historical=_unavailable(),
                                       campaign_seed="ak-seed-fixture",
                                       windows_completed=0)
        self.assertEqual(len(observations), 4)
        self.assertNotIn(controls.CONTROL_HISTORICAL_WIN_REPLAY,
                         [cid for cid, _ in runner.calls])

    def test_a_runner_answering_for_the_wrong_control_raises(self):
        wrong = {cid: _passing_observation(controls.CONTROL_AA)
                 for cid in controls.CONTROL_IDS}
        harness, _ = self._harness(wrong)
        with self.assertRaises(controls.ControlWiringError):
            harness.run_all(run_context=_run_context(), historical=_available(),
                            campaign_seed="ak-seed-fixture", windows_completed=0)

    def test_a_runner_returning_the_wrong_type_raises(self):
        class _Bad:
            runner_id = "bad"

            def run_control(self, definition, context):
                return {"outcome": "PASS"}

        harness = controls.ControlHarness(bundle=_bundle(), runner=_Bad())
        with self.assertRaises(controls.ControlWiringError):
            harness.run_all(run_context=_run_context(), historical=_available(),
                            campaign_seed="ak-seed-fixture", windows_completed=0)

    def test_a_missing_observation_is_not_run_not_a_pass(self):
        harness, _ = self._harness()
        result = harness.evaluate(
            observations=_other_observations(controls.CONTROL_NEUTRAL),
            context=_context(), aa_cadence=schemas.Check(schemas.PASS))
        outcome = result.outcome_for(controls.CONTROL_NEUTRAL)
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(outcome.disposition, controls.DISPOSITION_NOT_RUN)
        self.assertFalse(result.may_rank)

    def test_two_observations_for_one_control_raises(self):
        harness, _ = self._harness()
        duplicated = _all_observations() + (_passing_observation(controls.CONTROL_AA),)
        with self.assertRaises(controls.ControlWiringError):
            harness.evaluate(observations=duplicated, context=_context(),
                             aa_cadence=schemas.Check(schemas.PASS))

    def test_all_five_passing_permits_ranking(self):
        harness, _ = self._harness()
        result = harness.evaluate(observations=_all_observations(), context=_context(),
                                  aa_cadence=schemas.Check(schemas.PASS))
        self.assertTrue(result.may_rank)
        self.assertFalse(result.halts_campaign)
        self.assertFalse(result.voids_window)
        self.assertEqual(result.marker, "5/5")
        self.assertEqual(result.panel.check_1_to_4().outcome, schemas.PASS)

    def test_any_of_controls_1_to_4_failing_blocks_ranking(self):
        harness, _ = self._harness()
        for cid in controls.MANDATORY_CONTROL_IDS:
            broken = controls.ControlObservation(
                control_id=cid, ran=False, could_not_run_reason="fixture: did not run")
            result = harness.evaluate(
                observations=_other_observations(cid) + (broken,),
                context=_context(), aa_cadence=schemas.Check(schemas.PASS))
            with self.subTest(control=cid):
                self.assertFalse(result.may_rank)

    def test_drifted_definitions_block_ranking(self):
        harness, _ = self._harness()
        result = harness.evaluate(
            observations=_all_observations(), context=_context(),
            aa_cadence=schemas.Check(schemas.PASS),
            pinned_definitions_digest=_sha("a-different-bundle"))
        self.assertEqual(result.definitions_check.outcome, schemas.FAIL)
        self.assertFalse(result.may_rank)

    def test_derived_properties_cannot_be_stamped(self):
        harness, _ = self._harness()
        result = harness.evaluate(observations=_all_observations(), context=_context(),
                                  aa_cadence=schemas.Check(schemas.PASS))
        self.assertTrue(result.may_rank)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.may_rank = True
        with self.assertRaises(TypeError):
            dataclasses.replace(result, may_rank=True)
        # `replace()` used to hand back a mutated result, and the guarantee was
        # only that `may_rank` could not be stamped ON one. It is now stronger:
        # `replace()` cannot produce a ControlPanelResult AT ALL, because the
        # InitVar mint defaults to None on the rebuild. A stripped panel is no
        # longer a thing that exists to be reasoned about.
        with self.assertRaises(controls.ControlPanelForged):
            dataclasses.replace(result, panel=None)

    def test_result_must_cover_every_control(self):
        harness, _ = self._harness()
        result = harness.evaluate(observations=_all_observations(), context=_context(),
                                  aa_cadence=schemas.Check(schemas.PASS))
        # Same reason: the missing-control check is still in __post_init__, but
        # `replace()` no longer reaches it — it is refused at the mint.
        with self.assertRaises(controls.ControlPanelForged):
            dataclasses.replace(result, outcomes=result.outcomes[:4])

    def test_result_dict_is_canonicalizable_and_names_the_requirements(self):
        harness, _ = self._harness()
        result = harness.evaluate(observations=_all_observations(), context=_context(),
                                  aa_cadence=schemas.Check(schemas.PASS))
        payload = result.to_dict()
        schemas.canonical_json(payload)
        self.assertEqual(len(payload["outcomes"]), 5)
        self.assertIn("Failure is a gate defect", payload["outcomes"][0]["requirement"])

    def test_gate_defect_finding_refuses_a_control_the_protocol_did_not_name(self):
        with self.assertRaises(ValueError):
            controls.GateDefectFinding(control_id=controls.CONTROL_NEUTRAL,
                                       protocol_phrase="p", outcome=schemas.FAIL,
                                       detail=())


# =============================================================================
# Plugging into `api` — the panel is what the rest of the evaluator reads
# =============================================================================

def _campaign_controls(**kwargs) -> api.CampaignControls:
    defaults = {
        "calibration_block_count": 30,
        "contribution_floor": 6.0,
        "max_candidates": 20,
        "confirmation_admission_count": 5,
        "max_blocks_per_candidate": 40,
        "storage_floor_bytes_free": 10 ** 10,
    }
    defaults.update(kwargs)
    return api.CampaignControls(**defaults)


def _window(*, panel, aa_cadence, anchor, calibration=schemas.Check(schemas.PASS),
            control_definitions_immutable=schemas.Check(schemas.PASS)):
    """Built field by field on purpose: `WindowAttestations` has no defaults, and a
    convenience all-clear helper is the fixture that removes the signal."""
    passing = schemas.Check(schemas.PASS)
    return api.WindowAttestations(
        resource_claim_receipt="claim-001",
        resource_claim_open=passing,
        resource_claim_close=passing,
        resource_claim_same_holder=passing,
        no_concurrent_inference=passing,
        preflight_attestation_ref="pf-001",
        host_receipt="host-001",
        host_health=passing,
        anchor_at_open=anchor,
        anchor_at_close=anchor,
        anchor_gate=passing,
        evaluator_bundle=passing,
        runtime_source_label=passing,
        recipe=api.RecipeReceipt(constructor_id="ak-recipe/v1",
                                 constructor_sha256=_sha("recipe"),
                                 argv_sha256=_sha("argv")),
        storage_open=passing,
        storage_close=passing,
        strata=passing,
        stopping_rule_id="ak-stop/v1",
        rule_immutability=passing,
        order_randomized=passing,
        order_seed=_sha("order"),
        aa_cadence=aa_cadence,
        controls=panel,
        calibration=calibration,
        control_definitions_immutable=control_definitions_immutable,
        raw_evidence_ref="raw-001",
    )


def _request(*, anchor, calibration, campaign_controls, tier="T1"):
    return api.EvaluationRequest(
        event_id="ake-001", campaign_id="ak-llama_cpu-fixture-20260803",
        candidate_id="akc-001", tier=tier, backend="llama_cpu", phase="prefill",
        cell_class="microbench", protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(source_sha256=_sha("src"),
                                      binary_sha256=_sha("bin"),
                                      linkage_sha256=_sha("link")),
        anchor=anchor,
        evaluator=api.EvaluatorIdentity(id="ak-evaluator/v1",
                                        bundle_sha256=_sha("bundle"),
                                        runtime_source_label_ref="srclabel-001"),
        scope_denominator=api.ScopeDenominator(machine_subset="full", numa_nodes=(),
                                               devices=(), cores=96),
        scope_manifest_sha256=_sha("scope"), co_residency="single",
        determinism=api.DeterminismReport(determinism_class="bitwise_stable",
                                          same_seed_repeat_runs=3),
        metric="decode_tps", metric_direction="higher_better", reps=10,
        change_class="parameter", anchor_tier=tier, transfer_ratio_to=(),
        created_at="2026-08-03T12:00:00Z", campaign_controls=campaign_controls,
        calibration=calibration)


def _calibration_outputs(**kwargs) -> api.CalibrationOutputs:
    """An accepted calibration for the fixture cell.

    Built directly as `api.CalibrationOutputs` — the type `api` owns — because the
    SOLVE that produces one is `statistics.py`'s and is covered by its own suite.
    Re-deriving it here would be the duplication these tests exist to forbid.
    """
    defaults = {
        "backend": "llama_cpu", "phase": "prefill", "cell_class": "microbench",
        "noise_floor_phi": 0.02, "b_min_blocks": 10,
        "alpha_sel": 1.0 / 20, "alpha_conf": (1.0 / 20) / 5,
        "anchor_gate_band": (0.95, 1.05), "accepted": True,
        "solve_order_recorded": api.CALIBRATION_SOLVE_ORDER,
        "samples_ref": "ak-cal-raw-001",
        "e_process_construction_id": "sign_martingale_predictable_lambda/v1",
    }
    defaults.update(kwargs)
    return api.CalibrationOutputs(**defaults)


class _FixtureGateRunner:
    tier = "T1"

    def run_gates(self, request):
        return (_gate("g-correct", api.GATE_CORRECTNESS, requires_anchor=True),
                _gate("g-quality", api.GATE_QUALITY),
                _gate("g-perf", api.GATE_PERFORMANCE))


class TestPanelPlugsIntoApi(unittest.TestCase):

    def setUp(self):
        harness = controls.ControlHarness(
            bundle=_bundle(),
            runner=_FixtureRunner({cid: _passing_observation(cid)
                                   for cid in controls.CONTROL_IDS}))
        self.harness = harness
        self.controls_result = harness.evaluate(
            observations=_all_observations(), context=_context(),
            aa_cadence=schemas.Check(schemas.PASS))
        self.calibration = _calibration_outputs()
        self.anchor = _anchor()
        self.campaign_controls = _campaign_controls()

    def test_a_full_dispatch_with_five_passing_controls_is_search_grade(self):
        window = _window(panel=self.controls_result.panel,
                         aa_cadence=self.controls_result.aa_cadence,
                         anchor=self.anchor)
        request = _request(anchor=self.anchor, calibration=self.calibration,
                           campaign_controls=self.campaign_controls)
        dispatcher = api.TierDispatcher(gate_runners={"T1": _FixtureGateRunner()})
        outcome = dispatcher.dispatch(request, window, effect=_effect(0.30))
        self.assertEqual(outcome.verdict.search_grade.failed, ())
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        self.assertTrue(outcome.verdict.speed_rank_admissible)
        self.assertEqual(outcome.event_violations, ())
        self.assertIn("controls=5/5", outcome.grammar_line)

    def test_a_failing_aa_control_voids_the_window_through_api(self):
        failing = self.harness.evaluate(
            observations=_other_observations(controls.CONTROL_AA) + (
                controls.ControlObservation(control_id=controls.CONTROL_AA, ran=True,
                                            verdict=_verdict(effect=_effect(0.30))),),
            context=_context(), aa_cadence=schemas.Check(schemas.PASS))
        window = _window(panel=failing.panel, aa_cadence=failing.aa_cadence,
                         anchor=self.anchor)
        request = _request(anchor=self.anchor, calibration=self.calibration,
                           campaign_controls=self.campaign_controls)
        scan = api.check_void_conditions(request, window, rate_comparison=True)
        self.assertIn(api.VOID_AA_CONTROL_FAILED, scan.reasons())
        dispatcher = api.TierDispatcher(gate_runners={"T1": _FixtureGateRunner()})
        result = dispatcher.dispatch(request, window, effect=_effect(0.30))
        self.assertEqual(result.verdict.status, api.STATUS_INVALID)
        with self.assertRaises(api.SpeedRankUnavailable):
            result.verdict.rank_key()

    def test_the_four_of_five_marker_reaches_the_record_grammar(self):
        four = self.harness.evaluate(
            observations=_other_observations(controls.CONTROL_HISTORICAL_WIN_REPLAY),
            context=_context(historical=_unavailable("no durable win")),
            aa_cadence=schemas.Check(schemas.PASS),
            escalation=_escalation(controls.OPERATOR_DECISION_PROCEED_ON_FOUR))
        window = _window(panel=four.panel, aa_cadence=four.aa_cadence,
                         anchor=self.anchor)
        request = _request(anchor=self.anchor, calibration=self.calibration,
                           campaign_controls=self.campaign_controls)
        dispatcher = api.TierDispatcher(gate_runners={"T1": _FixtureGateRunner()})
        outcome = dispatcher.dispatch(request, window, effect=_effect(0.30))
        self.assertIn(f"controls=4/5 ({controls.HISTORICAL_REPLAY_UNAVAILABLE})",
                      outcome.grammar_line)
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)

    def test_a_failing_aa_cadence_costs_search_grade(self):
        window = _window(
            panel=self.controls_result.panel,
            aa_cadence=schemas.Check(schemas.FAIL, ("A/A has never run",)),
            anchor=self.anchor)
        request = _request(anchor=self.anchor, calibration=self.calibration,
                           campaign_controls=self.campaign_controls)
        dispatcher = api.TierDispatcher(gate_runners={"T1": _FixtureGateRunner()})
        outcome = dispatcher.dispatch(request, window, effect=_effect(0.30))
        self.assertIn("aa_control_within_cadence", outcome.verdict.search_grade.failed)
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)


# =============================================================================
# The calibration block — delegation to statistics.py, and the join back
# =============================================================================

def _noise_floor(neutral_check: schemas.Check) -> ak_statistics.NoiseFloor:
    return ak_statistics.NoiseFloor(
        value=0.02, blocks=30, quantile=0.95,
        method=ak_statistics.PERCENTILE_METHOD,
        declared_calibration_block_count=30, neutral_check=neutral_check)


def _attempt(noise_floor) -> ak_statistics.CalibrationAttempt:
    return ak_statistics.CalibrationAttempt(
        attempt=0, alpha_sel=0.05, alpha_conf=0.01, threshold_sel=20.0,
        threshold_conf=100.0,
        reps_floor=ak_statistics.RepsFloor(
            blocks=10, band="<=2%", relative_effect=0.0003,
            citation="bench-cpu.md:21-22", conservative=False, note="fixture"),
        start_blocks=10, b_min=10, noise_floor=noise_floor, mde=None,
        condition_a=None, alpha_validation=schemas.Check(schemas.PASS), band=None,
        accepted=True, reasons=())


def _solve(attempts) -> ak_statistics.CalibrationSolve:
    return ak_statistics.CalibrationSolve(
        inputs_digest={"backend": "llama_cpu"}, attempts=tuple(attempts),
        outputs=None, aa_effect_pool=(), anchor_calibration_values=(), reasons=())


class TestCalibrationIsOwnedByStatistics(unittest.TestCase):
    """This module supplies the material and reads the neutral verdict back. It
    does not solve, and these tests assert the absence of a second solver."""

    def test_controls_declares_who_owns_the_solve(self):
        self.assertEqual(controls.CALIBRATION_OWNER,
                         ak_statistics.STATISTICS_MODULE_ID)

    def test_controls_implements_no_calibration_solver(self):
        for name in ("CalibrationSolver", "CalibrationEstimator", "solve_calibration",
                     "p_bench_1_floor", "TIGHTENING_CONSTRUCTIONS"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(controls, name),
                                 f"{name} would be a second source of truth for the "
                                 "calibration clause")

    def test_run_calibration_block_refuses_anything_but_the_owners_input(self):
        with self.assertRaises(controls.ControlWiringError):
            controls.run_calibration_block({"backend": "llama_cpu"})
        with self.assertRaises(controls.ControlWiringError):
            controls.run_calibration_block(None)

    def test_run_calibration_block_delegates(self):
        calls = []
        real = ak_statistics.solve_calibration
        sentinel = object()
        ak_statistics.solve_calibration = lambda inputs: (calls.append(inputs) or sentinel)
        try:
            fake = object.__new__(ak_statistics.CalibrationInputs)
            self.assertIs(controls.run_calibration_block(fake), sentinel)
        finally:
            ak_statistics.solve_calibration = real
        self.assertEqual(calls, [fake])

    def test_neutral_verdict_is_read_out_of_the_solve(self):
        passing = _solve([_attempt(_noise_floor(schemas.Check(schemas.PASS, ("ok",))))])
        self.assertEqual(controls.neutral_dispersion_check(passing).outcome, schemas.PASS)
        failing = _solve([_attempt(_noise_floor(schemas.Check(
            schemas.FAIL, ("neutral p95 exceeds phi",))))])
        self.assertEqual(controls.neutral_dispersion_check(failing).outcome, schemas.FAIL)

    def test_the_last_attempt_with_a_floor_is_the_one_read(self):
        solve = _solve([
            _attempt(_noise_floor(schemas.Check(schemas.FAIL, ("first attempt",)))),
            _attempt(_noise_floor(schemas.Check(schemas.PASS, ("after tightening",)))),
        ])
        check = controls.neutral_dispersion_check(solve)
        self.assertEqual(check.outcome, schemas.PASS)

    def test_a_solve_with_no_noise_floor_is_could_not_check_not_pass(self):
        check = controls.neutral_dispersion_check(_solve([_attempt(None)]))
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("uncomputed consistency check is not a passing one",
                      " ".join(check.reasons))
        self.assertEqual(controls.neutral_dispersion_check(_solve([])).outcome,
                         schemas.COULD_NOT_CHECK)

    def test_neutral_dispersion_check_refuses_a_foreign_object(self):
        with self.assertRaises(controls.ControlWiringError):
            controls.neutral_dispersion_check({"attempts": []})

    def test_the_extracted_verdict_feeds_the_neutral_control(self):
        """End to end: the solve's neutral check reaches the control evaluator."""
        solve = _solve([_attempt(_noise_floor(schemas.Check(
            schemas.FAIL, ("neutral p95 0.9 exceeds phi 0.02",))))])
        context = _context(neutral_dispersion=controls.neutral_dispersion_check(solve))
        harness = controls.ControlHarness(bundle=_bundle(), runner=_FixtureRunner({}))
        result = harness.evaluate(observations=_all_observations(), context=context,
                                  aa_cadence=schemas.Check(schemas.PASS))
        outcome = result.outcome_for(controls.CONTROL_NEUTRAL)
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertIn("rather than raising the floor", " ".join(outcome.check.reasons))
        self.assertFalse(result.may_rank)


# =============================================================================
# The module's own audit
# =============================================================================

class TestModuleAudit(unittest.TestCase):

    def test_controls_module_has_no_write_or_process_path(self):
        self.assertEqual(controls.audit_no_write_or_process_paths().outcome,
                         schemas.PASS)

    def test_the_audit_detects_a_write_path(self):
        check = controls.audit_no_write_or_process_paths(
            "import os\ndef f(p):\n    os.remove(p)\n")
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_the_audit_detects_a_process_launch(self):
        check = controls.audit_no_write_or_process_paths(
            "import subprocess\ndef f():\n    subprocess.Popen(['ls'])\n")
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_unparseable_source_is_could_not_check(self):
        self.assertEqual(
            controls.audit_no_write_or_process_paths("def (:\n").outcome,
            schemas.COULD_NOT_CHECK)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
