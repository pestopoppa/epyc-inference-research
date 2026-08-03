#!/usr/bin/env python3
"""Regression tests for defects found by an adversarial review of `controls.py`.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_controls_redteam.py
    python3 -W error::ResourceWarning -m unittest \\
        scripts/kernel_rnd/autokernel/evaluator/test_controls_redteam.py

Kept in its own module rather than appended to `test_controls.py`: that file is
the builder's and the evaluator package took writes from five parallel sessions
while this review ran. Every test below names the defect it pins, and each one
FAILS against the module as originally written.

The fixtures are `test_controls`' own, imported as a namespace so its TestCases
are not collected twice: `unittest` walks this module's `dir()`, and a module
object is not a `TestCase`.
"""
from __future__ import annotations

import dataclasses
import unittest

from .. import schemas
from . import api, controls
from . import test_controls as fx


def _definition(control_id: str) -> controls.ControlDefinition:
    return {d.control_id: d for d in controls.CONTROL_DEFINITIONS}[control_id]


def _harness() -> controls.ControlHarness:
    return controls.ControlHarness(bundle=fx._bundle(), runner=fx._FixtureRunner({}))


def _result(observation, *, context=None, aa_cadence=None, escalation=None, **kwargs):
    """Evaluate one substituted observation against four passing ones."""
    others = fx._other_observations(observation.control_id)
    return _harness().evaluate(
        observations=others + (observation,),
        context=context or fx._context(),
        aa_cadence=aa_cadence or schemas.Check(schemas.PASS),
        escalation=escalation, **kwargs)


# =============================================================================
# The compliant path — a guard against fixes that forbid their own idiom
# =============================================================================

class TestTheCompliantPathStillPasses(unittest.TestCase):
    """Every tightening below must leave the conforming case conforming."""

    def test_five_passing_controls_still_licence_ranking(self):
        result = _harness().evaluate(
            observations=fx._all_observations(), context=fx._context(),
            aa_cadence=schemas.Check(schemas.PASS))
        self.assertTrue(result.may_rank)
        self.assertEqual(result.marker, "5/5")
        self.assertFalse(result.halts_campaign)
        self.assertFalse(result.voids_window)
        self.assertEqual(result.gate_defects, ())

    def test_the_operator_authorised_four_control_path_still_ranks(self):
        result = _harness().evaluate(
            observations=fx._other_observations(controls.CONTROL_HISTORICAL_WIN_REPLAY),
            context=fx._context(historical=fx._unavailable()),
            aa_cadence=schemas.Check(schemas.PASS),
            escalation=fx._escalation(controls.OPERATOR_DECISION_PROCEED_ON_FOUR))
        self.assertTrue(result.may_rank)
        self.assertFalse(result.halts_campaign)
        self.assertEqual(result.marker,
                         f"4/5 ({controls.HISTORICAL_REPLAY_UNAVAILABLE})")


# =============================================================================
# Defect 1 — the A/A control passed by not measuring anything
# =============================================================================

class TestAAControlCannotPassByNotMeasuring(unittest.TestCase):
    """*"the anchor measured against itself, through the full candidate pipeline
    ... it calibrates the false-positive rate and it is what detects host drift
    mid-campaign."*

    The evaluator asked only whether the resolution was `improvement` or
    `regression`. Deleting the effect estimate — the thing it inspects — made it
    return PASS with no reasons at all.
    """

    def _aa(self, verdict) -> controls.ControlObservation:
        return controls.ControlObservation(
            control_id=controls.CONTROL_AA, ran=True, verdict=verdict)

    def test_an_aa_with_no_effect_estimate_is_could_not_check(self):
        verdict = fx._verdict(effect=None)
        self.assertEqual(verdict.status, api.STATUS_PASS)
        self.assertEqual(verdict.effect_resolution, api.EFFECT_NOT_MEASURED)
        check = controls._evaluate_aa(_definition(controls.CONTROL_AA),
                                      self._aa(verdict), fx._context())
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("measured nothing", " ".join(check.reasons))

    def test_an_unmeasured_aa_cannot_licence_ranking(self):
        result = _result(self._aa(fx._verdict(effect=None)))
        self.assertEqual(result.outcome_for(controls.CONTROL_AA).check.outcome,
                         schemas.COULD_NOT_CHECK)
        self.assertFalse(result.may_rank)

    def test_an_aa_whose_own_verdict_failed_is_not_a_passing_aa(self):
        verdict = fx._verdict(
            gates=(fx._gate("g-correct", api.GATE_CORRECTNESS, schemas.FAIL),),
            effect=None)
        self.assertEqual(verdict.status, api.STATUS_FAIL)
        check = controls._evaluate_aa(_definition(controls.CONTROL_AA),
                                      self._aa(verdict), fx._context())
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("uncalibrated", " ".join(check.reasons))

    def test_an_inconclusive_aa_is_not_a_passing_aa(self):
        verdict = fx._verdict(
            gates=(fx._gate("g-correct", api.GATE_CORRECTNESS),
                   fx._gate("g-mech", api.GATE_MECHANISM, schemas.FAIL)),
            effect=fx._effect(0.001))
        self.assertEqual(verdict.status, api.STATUS_INCONCLUSIVE)
        check = controls._evaluate_aa(_definition(controls.CONTROL_AA),
                                      self._aa(verdict), fx._context())
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_a_significant_aa_effect_is_still_a_FAIL_not_a_could_not_check(self):
        """The tightening must not swallow the failure the control exists for."""
        check = controls._evaluate_aa(_definition(controls.CONTROL_AA),
                                      self._aa(fx._verdict(effect=fx._effect(0.30))),
                                      fx._context())
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("VOIDS the enclosing measurement window", " ".join(check.reasons))


# =============================================================================
# Defect 2 — the accept-side control's direction check was skippable by omission
# =============================================================================

class TestHistoricalReplayDirectionIsChecked(unittest.TestCase):
    """The manifest declares a `reference direction` for control 5. An observation
    that simply omitted its direction skipped the comparison silently, while an
    omitted MAGNITUDE was already COULD_NOT_CHECK — the same omission, two answers.
    """

    def test_a_replay_with_no_direction_is_could_not_check(self):
        result = _result(fx._passing_observation(
            controls.CONTROL_HISTORICAL_WIN_REPLAY).__class__(
                control_id=controls.CONTROL_HISTORICAL_WIN_REPLAY, ran=True,
                verdict=fx._verdict(tier="T2", effect=fx._effect(0.36)),
                promoted=True, observed_magnitude=0.36, observed_direction=None))
        outcome = result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertEqual(outcome.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("reported no direction", " ".join(outcome.check.reasons))
        self.assertFalse(result.may_rank)

    def test_a_replay_in_the_wrong_direction_is_still_a_FAIL(self):
        observation = controls.ControlObservation(
            control_id=controls.CONTROL_HISTORICAL_WIN_REPLAY, ran=True,
            verdict=fx._verdict(tier="T2", effect=fx._effect(0.36)), promoted=True,
            observed_magnitude=0.36, observed_direction="lower_better")
        result = _result(observation)
        self.assertEqual(
            result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY).check.outcome,
            schemas.FAIL)

    def test_a_failure_to_promote_outranks_an_unchecked_direction(self):
        """A FAIL must never be softened into COULD_NOT_CHECK by a missing field."""
        observation = controls.ControlObservation(
            control_id=controls.CONTROL_HISTORICAL_WIN_REPLAY, ran=True,
            verdict=fx._verdict(tier="T2", effect=fx._effect(0.36)), promoted=False,
            observed_magnitude=None, observed_direction=None)
        result = _result(observation)
        outcome = result.outcome_for(controls.CONTROL_HISTORICAL_WIN_REPLAY)
        self.assertEqual(outcome.check.outcome, schemas.FAIL)
        self.assertIn("did NOT promote", " ".join(outcome.check.reasons))


# =============================================================================
# Defect 3 — the predicate table was outside every digest
# =============================================================================

class TestTheControlPredicatesAreTamperEvident(unittest.TestCase):
    """*"Control definitions, fixtures, expected directions, and seeds live inside
    the evaluator bundle under the measurement trust boundary and MUST NOT be
    modified by any process inside the loop."*

    The five definitions hashed; the five predicates that decide whether a control
    passed did not. Rebinding one entry of `_EVALUATORS` is a strictly easier move
    than rebinding `CONTROL_DEFINITIONS`, and it produced a 5/5 panel and
    `may_rank == True` with the definitions digest reporting PASS.
    """

    def setUp(self):
        self._saved = dict(controls._EVALUATORS)

    def tearDown(self):
        controls._EVALUATORS.clear()
        controls._EVALUATORS.update(self._saved)

    def test_substituting_a_predicate_is_detected(self):
        controls._EVALUATORS[controls.CONTROL_DEGRADED_NEGATIVE] = \
            lambda d, o, c: schemas.Check(schemas.PASS)
        check = controls.verify_control_definitions(controls.CONTROL_DEFINITIONS_DIGEST)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("EVALUATORS changed in-process", " ".join(check.reasons))

    def test_substituting_a_predicate_refuses_the_resolve(self):
        controls._EVALUATORS[controls.CONTROL_POSITIVE] = \
            lambda d, o, c: schemas.Check(schemas.PASS)
        with self.assertRaises(controls.ControlBundleDrift):
            fx._bundle()

    def test_a_substitute_wearing_the_originals_name_is_still_detected(self):
        """Matching `__module__`/`__qualname__` is not matching the code."""
        def impostor(definition, observation, context):
            return schemas.Check(schemas.PASS)
        original = controls._EVALUATORS[controls.CONTROL_DEGRADED_NEGATIVE]
        impostor.__module__ = original.__module__
        impostor.__qualname__ = original.__qualname__
        impostor.__name__ = original.__name__
        controls._EVALUATORS[controls.CONTROL_DEGRADED_NEGATIVE] = impostor
        self.assertEqual(
            controls.verify_control_definitions(
                controls.CONTROL_DEFINITIONS_DIGEST).outcome,
            schemas.FAIL)

    def test_a_liar_predicate_cannot_produce_a_rankable_panel(self):
        """End to end: a degraded-negative that RECEIVED a rank must not pass."""
        for cid in controls.CONTROL_IDS:
            controls._EVALUATORS[cid] = lambda d, o, c: schemas.Check(schemas.PASS)
        cheating = controls.ControlObservation(
            control_id=controls.CONTROL_DEGRADED_NEGATIVE, ran=True,
            verdict=fx._verdict(effect=fx._effect(0.90)))
        with self.assertRaises(controls.ControlBundleDrift):
            controls.ControlHarness(bundle=fx._bundle(), runner=fx._FixtureRunner({}))
        # And if the bundle was resolved BEFORE the swap, the window-close
        # re-verification still refuses to licence the window.
        controls._EVALUATORS.clear()
        controls._EVALUATORS.update(self._saved)
        harness = _harness()
        for cid in controls.CONTROL_IDS:
            controls._EVALUATORS[cid] = lambda d, o, c: schemas.Check(schemas.PASS)
        result = harness.evaluate(
            observations=fx._other_observations(controls.CONTROL_DEGRADED_NEGATIVE)
            + (cheating,),
            context=fx._context(), aa_cadence=schemas.Check(schemas.PASS))
        self.assertEqual(result.definitions_check.outcome, schemas.FAIL)
        self.assertFalse(result.may_rank)

    def test_the_published_digest_is_not_the_baseline_that_is_verified(self):
        """Rebinding `CONTROL_DEFINITIONS` AND the published digest beside it — the
        actor's obvious second move — used to produce a clean PASS."""
        tampered = tuple(dataclasses.replace(d, requirement="anything goes")
                         for d in controls.CONTROL_DEFINITIONS)
        saved_defs = controls.CONTROL_DEFINITIONS
        saved_digest = controls.CONTROL_DEFINITIONS_DIGEST
        try:
            controls.CONTROL_DEFINITIONS = tampered
            controls.CONTROL_DEFINITIONS_DIGEST = controls._current_definitions_digest()
            self.assertEqual(controls.verify_control_definitions().outcome, schemas.FAIL)
            with self.assertRaises(controls.ControlBundleDrift):
                controls.resolve_control_bundle(
                    pinned_definitions_digest=controls.CONTROL_DEFINITIONS_DIGEST,
                    aa_cadence=fx._cadence(), seed_rotation=fx._rotation(),
                    historical_win_replays=(fx._declaration(),),
                    source_label="evaluator-bundle@fixture")
        finally:
            controls.CONTROL_DEFINITIONS = saved_defs
            controls.CONTROL_DEFINITIONS_DIGEST = saved_digest


# =============================================================================
# Defect 4 — a supplied-but-empty pin degraded to no pin
# =============================================================================

class TestAnUnusablePinIsNotASatisfiedPin(unittest.TestCase):

    def test_an_empty_definitions_pin_does_not_fall_back(self):
        result = _harness().evaluate(
            observations=fx._all_observations(), context=fx._context(),
            aa_cadence=schemas.Check(schemas.PASS), pinned_definitions_digest="")
        self.assertEqual(result.definitions_check.outcome, schemas.COULD_NOT_CHECK)
        self.assertFalse(result.may_rank)

    def test_a_blank_definitions_pin_does_not_fall_back(self):
        result = _harness().evaluate(
            observations=fx._all_observations(), context=fx._context(),
            aa_cadence=schemas.Check(schemas.PASS), pinned_definitions_digest="   ")
        self.assertEqual(result.definitions_check.outcome, schemas.COULD_NOT_CHECK)
        self.assertFalse(result.may_rank)

    def test_an_absent_pin_still_falls_back_to_the_bundles_own_digest(self):
        result = _harness().evaluate(
            observations=fx._all_observations(), context=fx._context(),
            aa_cadence=schemas.Check(schemas.PASS), pinned_definitions_digest=None)
        self.assertEqual(result.definitions_check.outcome, schemas.PASS)
        self.assertTrue(result.may_rank)


# =============================================================================
# Defect 5 — `may_rank` ignored the cadence attestation it was handed
# =============================================================================

class TestTheCadenceAttestationGatesRanking(unittest.TestCase):
    """Search-grade requires *"a passing A/A control within its declared cadence"*.
    The panel's `aa` field is THIS window's A/A outcome; the cadence attestation is
    a different fact, and `may_rank` consulted `definitions_check` — also not one of
    controls 1-4 — while ignoring it."""

    def test_a_failing_cadence_blocks_ranking(self):
        result = _harness().evaluate(
            observations=fx._all_observations(), context=fx._context(),
            aa_cadence=schemas.Check(schemas.FAIL, ("no A/A has run for 20 windows",)))
        self.assertEqual(result.panel.check_1_to_4().outcome, schemas.PASS)
        self.assertFalse(result.may_rank)

    def test_an_uncheckable_cadence_blocks_ranking(self):
        result = _harness().evaluate(
            observations=fx._all_observations(), context=fx._context(),
            aa_cadence=schemas.Check(schemas.COULD_NOT_CHECK, ("ledger unreadable",)))
        self.assertFalse(result.may_rank)


# =============================================================================
# Defect 6 — the cadence attestation could not be told about a campaign boundary
# =============================================================================

class TestCadenceTriggersReachTheAttestation(unittest.TestCase):

    def _ledger(self, **kwargs):
        defaults = {"window_id": "aa-w1", "ran_at_epoch_seconds": 1000.0,
                    "windows_completed_at_run": 10, "outcome": schemas.PASS,
                    "anchor_short": "anchor-a"}
        defaults.update(kwargs)
        return (controls.AALedgerEntry(**defaults),)

    def test_a_campaign_boundary_reaches_check(self):
        scheduler = controls.AAScheduler(fx._cadence())
        ledger = self._ledger()
        in_cadence = dict(ledger=ledger, windows_completed=10,
                          now_epoch_seconds=1001.0, anchor_short="anchor-a")
        self.assertEqual(scheduler.check(**in_cadence).outcome, schemas.PASS)
        boundary = scheduler.check(campaign_boundary=True, **in_cadence)
        self.assertEqual(boundary.outcome, schemas.FAIL)
        self.assertIn("campaign boundary", " ".join(boundary.reasons))

    def test_a_ledger_ahead_of_this_window_is_not_a_satisfied_cadence(self):
        """A ledger the caller's counters cannot be reconciled with produced a
        negative interval, which compared as comfortably inside the cadence."""
        scheduler = controls.AAScheduler(fx._cadence())
        ledger = self._ledger(windows_completed_at_run=100, ran_at_epoch_seconds=9e9)
        check = scheduler.check(ledger=ledger, windows_completed=1,
                                now_epoch_seconds=1000.0, anchor_short="anchor-a")
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("recorded ahead of this window", " ".join(check.reasons))
        decision = scheduler.due(ledger=ledger, windows_completed=1,
                                 now_epoch_seconds=1000.0, anchor_short="anchor-a")
        self.assertTrue(decision.due)


# =============================================================================
# Defect 7 — a stamped disposition defeated a derived consequence
# =============================================================================

class TestADispositionCannotContradictItsCheck(unittest.TestCase):
    """`ControlPanelResult.gate_defects` skips outcomes stamped
    `unavailable_recorded`. Nothing stopped a FAILING control 1 from carrying that
    stamp, and the gate defect — the whole point of control 1 — vanished."""

    def test_satisfied_requires_a_passing_check(self):
        with self.assertRaises(ValueError):
            controls.ControlOutcome(
                control_id=controls.CONTROL_POSITIVE,
                definition=_definition(controls.CONTROL_POSITIVE),
                check=schemas.Check(schemas.FAIL, ("the positive control did not rank",)),
                disposition=controls.DISPOSITION_SATISFIED)

    def test_a_passing_check_cannot_carry_a_failure_disposition(self):
        with self.assertRaises(ValueError):
            controls.ControlOutcome(
                control_id=controls.CONTROL_AA,
                definition=_definition(controls.CONTROL_AA),
                check=schemas.Check(schemas.PASS),
                disposition=controls.DISPOSITION_VOIDS_WINDOW)

    def test_only_control_5_has_an_unavailable_branch(self):
        for cid in controls.MANDATORY_CONTROL_IDS:
            with self.subTest(control=cid):
                with self.assertRaises(ValueError):
                    controls.ControlOutcome(
                        control_id=cid, definition=_definition(cid),
                        check=schemas.Check(schemas.FAIL, ("stamped",)),
                        disposition=controls.DISPOSITION_UNAVAILABLE_RECORDED)

    def test_control_5s_own_unavailable_dispositions_are_still_constructible(self):
        for outcome in (schemas.PASS, schemas.FAIL):
            with self.subTest(outcome=outcome):
                controls.ControlOutcome(
                    control_id=controls.CONTROL_HISTORICAL_WIN_REPLAY,
                    definition=_definition(controls.CONTROL_HISTORICAL_WIN_REPLAY),
                    check=schemas.Check(outcome, ("recorded",)),
                    disposition=controls.DISPOSITION_UNAVAILABLE_RECORDED)


# =============================================================================
# Defect 8 — an absent manifest entry was journaled as a GATE DEFECT
# =============================================================================

class TestUnavailabilityIsNotAGateDefect(unittest.TestCase):
    """*"A failure to promote is a gate defect, not a research finding."* An entry
    the manifest never declared is the UNAVAILABLE branch, not a failure to
    promote. Recording it as a gate defect stamps the phrase "it MUST promote" on
    a win that was never replayed, and publishes `escalation_required` for a
    finding about the gate that no evidence supports."""

    def _unavailable_result(self, escalation=None):
        return _harness().evaluate(
            observations=fx._other_observations(controls.CONTROL_HISTORICAL_WIN_REPLAY),
            context=fx._context(historical=fx._unavailable()),
            aa_cadence=schemas.Check(schemas.PASS), escalation=escalation)

    def test_no_escalation_is_blocked_but_is_not_a_gate_defect(self):
        result = self._unavailable_result()
        self.assertEqual(result.gate_defects, ())
        self.assertFalse(result.may_rank)
        self.assertIsNone(result.panel)
        self.assertTrue(result.halts_campaign)
        self.assertIn("the operator's call", result.blocked_reason)

    def test_a_pending_escalation_is_blocked_but_is_not_a_gate_defect(self):
        result = self._unavailable_result(
            fx._escalation(controls.OPERATOR_DECISION_PENDING))
        self.assertEqual(result.gate_defects, ())
        self.assertFalse(result.may_rank)
        self.assertTrue(result.halts_campaign)

    def test_a_halt_decision_still_halts(self):
        result = self._unavailable_result(fx._escalation(controls.OPERATOR_DECISION_HALT))
        self.assertTrue(result.halts_campaign)
        self.assertFalse(result.may_rank)

    def test_an_actual_failure_to_promote_is_still_a_gate_defect(self):
        """The distinction must cut both ways or it is not a distinction."""
        observation = controls.ControlObservation(
            control_id=controls.CONTROL_HISTORICAL_WIN_REPLAY, ran=True,
            verdict=fx._verdict(tier="T2", effect=fx._effect(0.36)), promoted=False,
            observed_magnitude=0.36, observed_direction="higher_better")
        result = _result(observation)
        self.assertEqual([d.control_id for d in result.gate_defects],
                         [controls.CONTROL_HISTORICAL_WIN_REPLAY])
        self.assertTrue(result.halts_campaign)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
