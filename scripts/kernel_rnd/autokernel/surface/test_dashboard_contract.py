#!/usr/bin/env python3
"""test_dashboard_contract.py — the AK6 `/kernel` contract v2 and its producer.

WHAT THIS SUITE IS DEFENDING
----------------------------
One scar, stated once so every test below can point at it:

> Today's `/kernel` page is **absence-tolerant over a missing directory** — it
> renders clean when its producer is dead, which is the exact shape of AutoPilot
> dying at trial 1302 and staying dead ~23 HOURS with every dashboard green.

So the assertions that matter here are not "the happy document validates". They
are the ones that fail if a dead producer can be made to look alive:

  * a section cannot be omitted, and an omitted one is not a healthy one;
  * `degraded` and `unreported_sections` cannot be stamped independently of the
    sections they summarise;
  * `produced_at` cannot be hand-set, cannot be lifted by a live host reading, and
    is `null` — not `now` — when nothing reported;
  * `None` is not an accepted input for a section.

FIXTURES ARE REUSED, NEVER RE-DECLARED. `test_readiness.green_signal` and
`test_packager.release_package` are imported as-is, so a drift in either owning
module's fixture drifts this suite too — which is the only way the seam assertions
here stay honest. `controller/test_state_machine.py` is the one exception and
`ObserveControllerTest` says why: it imports the package flat, so its objects are
instances of a SECOND copy of every class and no `isinstance` seam would bite.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO PROCESS, NO PRODUCTION WRITE. Every byte
written by this suite goes into a `tempfile.TemporaryDirectory` it creates and
removes.

Run:
    python3 -m unittest scripts.kernel_rnd.autokernel.surface.test_dashboard_contract
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import pathlib
import tempfile
import unittest

# Import convention (README): package-relative, so this suite shares ONE copy of
# `schemas` with the modules under test. A flat import loads a second copy and
# every `isinstance` guard silently degrades to a no-op.
from .. import schemas as S
from .. import storage
from .. import journal as J
from ..controller import composition as COMP
from ..controller import state_machine as SM
from ..release import packager
from ..release import readiness as R
from ..release import test_packager as FPK
from ..release import test_readiness as FRD
from ..resource import device_claim
from . import dashboard_contract as DC

CAMPAIGN = FRD.CAMPAIGN
NOW = "2026-08-03T12:00:00+00:00"
LATER = "2026-08-03T14:00:00+00:00"
LONG_AGO = "2026-07-01T00:00:00+00:00"
EXPORTED_AT = "2026-08-03T15:30:00+00:00"


# =============================================================================
# Fixtures — the owning modules' objects, assembled the way a caller would
# =============================================================================

def transition(**over) -> SM.Transition:
    fields = {"seq": 7, "from_state": SM.T1_SEARCH_EVAL, "to_state": SM.POST_RUN_CRITIC,
              "trigger": "controller", "reason": "T1 banked", "at": NOW}
    fields.update(over)
    return SM.Transition(**fields)


def controller(**over) -> DC.ControllerObservation:
    fields = {"campaign_id": CAMPAIGN, "state": SM.POST_RUN_CRITIC, "seq": 7,
              "stopped": False, "last_transition": transition()}
    fields.update(over)
    return DC.ControllerObservation(**fields)


def champion_record(**over) -> dict:
    record = {
        "schema": S.SCHEMA_CHAMPION,
        "source_tree": "llama.cpp",
        "anchor_commit": "67a433bf45a8a091d83b4ea0b32ff0735fd51800",
        "branch": "ak/champion/llama-20260802",
        "member_candidates": ["akc-20260803-0001"],
        "combined_candidate_id": "akc-20260803-0009",
        "last_t0": {"event_id": "ake-20260803-0002", "status": "pass"},
        "last_t1": {"event_id": "ake-20260803-0001", "status": "pass"},
        "last_t2": None,
        "readiness": {
            "by_backend": {"llama_cpu": {"prefill": {}, "decode": {}}},
            "reference_signal": "point +2.1% / LCB +0.8% versus anchor on 6 cells",
        },
        "affected_surface_union_sha256": FRD.sha("surface-union"),
        "storage_gb": 12.0,
        "blocking_conditions": [],
        "created_at": NOW,
    }
    record.update(over)
    return record


def readiness_report(**over) -> R.ReadinessReport:
    fields = {"campaign_id": CAMPAIGN, "computed_at": LATER,
              "signals": (FRD.green_signal(),)}
    fields.update(over)
    return R.compute_readiness_report(**fields)


def headroom(**over) -> DC.HeadroomObservation:
    fields = {
        "storage_state": storage.StorageState(
            state=storage.STORAGE_OK, free_bytes=200 * 1024 ** 3,
            total_bytes=3750 * 1024 ** 3, floor_bytes=50 * 1024 ** 3),
        "quota_state": storage.QuotaState(
            state=storage.QUOTA_OK, used_bytes=10 * 1024 ** 3,
            limit_bytes=100 * 1024 ** 3, fraction=0.1),
        "measured_at": EXPORTED_AT,
    }
    fields.update(over)
    return DC.HeadroomObservation(**fields)


def claim_receipt(**over) -> device_claim.ClaimReceipt:
    fields = {"claim_id": "akclaim-0001", "device_id": "gpu0",
              "lock_path": "/mnt/raid0/llm/tmp/gpu_device.gpu0.lock",
              "state": "held", "holder_pid": 4242, "holder_start_ticks": 99,
              "holder_boot_id": "boot-1", "host": "epyc", "purpose": "T1 eval",
              "campaign_id": CAMPAIGN, "acquired_at": EXPORTED_AT}
    fields.update(over)
    return device_claim.ClaimReceipt(**fields)


def claims(**over) -> DC.ClaimsObservation:
    fields = {"receipts": (claim_receipt(),), "observed_at": EXPORTED_AT}
    fields.update(over)
    return DC.ClaimsObservation(**fields)


#: A sentinel distinct from `None`, because `None` is the value under test: an
#: override helper that treated `None` as "not supplied" would silently substitute
#: the default and every assertion about refusing `None` would pass vacuously.
_UNSET = object()


def inputs(**over) -> DC.ContractInputs:
    """Everything reported, nothing degraded — the baseline every test perturbs."""
    report = over.pop("readiness", _UNSET)
    if report is _UNSET:
        report = readiness_report()
    package = over.pop("release_package", _UNSET)
    if package is _UNSET:
        package = FPK.release_package()
    head = over.pop("headroom", _UNSET)
    if head is _UNSET:
        head = headroom()
    ctrl = over.pop("controller", _UNSET)
    if ctrl is _UNSET:
        ctrl = controller()
    champ = over.pop("champion", _UNSET)
    if champ is _UNSET:
        champ = champion_record()
    blocking = over.pop("blocking", _UNSET)
    if blocking is _UNSET:
        # Derived from the SAME objects the contract is built from. Anything else
        # is refused by `_assert_blocking_agrees_with_its_own_inputs`, which is the
        # point: a panel derived from other objects is a panel that can read clear
        # while the sections beside it read blocked.
        blocking = DC.derive_blocking_conditions(
            controller=ctrl, readiness_report=report, release_package=package,
            headroom=head, champion=champ)
    fields = {
        "campaign_id": CAMPAIGN,
        "controller": ctrl,
        "champion": champ,
        "readiness": report,
        "headroom": head,
        "blocking": blocking,
        "claims": claims(),
        "release_package": package,
        "exported_at": EXPORTED_AT,
    }
    fields.update(over)
    return DC.ContractInputs(**fields)


def document(**over) -> dict:
    return DC.build_contract(inputs(**over))


def _signal_fields(signal: R.ReadinessSignal) -> dict:
    """The signal's real fields. `dataclasses.fields`, never `__dataclass_fields__`:
    the latter also lists `ClassVar` pseudo-fields (`is_trigger`, `signal_class`,
    `reducer_id`), which are not constructor arguments.
    """
    return {f.name: getattr(signal, f.name) for f in dataclasses.fields(signal)}


def stored_standing(signal: R.ReadinessSignal, standing: str, blockers: tuple):
    """A signal whose STORED standing contradicts its own green phases.

    Written after construction, not through it, and the reason is itself a
    guarantee worth naming: `ReadinessSignal.__post_init__` REFUSES a standing its
    evidence does not derive (`StandingNotDerived`), so the readiness module is the
    only place a standing can come from. That is exactly the invariant the drift
    test below is protecting — the surface must READ this field, because it is the
    one nobody else is allowed to compute. Forcing the contradiction past the
    owner's guard is the only way to catch a producer that quietly recomputes it.
    """
    forced = copy.copy(signal)
    object.__setattr__(forced, "standing", standing)
    object.__setattr__(forced, "blockers", blockers)
    return forced


# =============================================================================
# 0 — fixture honesty: the baseline really is undegraded
# =============================================================================

class BaselineFixtureTest(unittest.TestCase):
    """If the baseline is already degraded, every absence test below is vacuous."""

    def test_the_baseline_document_validates_and_is_not_degraded(self):
        doc = document()
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])
        self.assertFalse(doc["degraded"])
        self.assertEqual(doc["unreported_sections"], [])
        self.assertEqual(sorted(doc["sections"]), sorted(S.DASHBOARD_SECTIONS))
        for name, section in doc["sections"].items():
            self.assertEqual(section["status"], S.SECTION_OBSERVED, name)

    def test_the_baseline_readiness_signal_is_green(self):
        signal = FRD.green_signal()
        self.assertEqual(signal.standing, R.STANDING_MET, signal.blockers)

    def test_the_baseline_package_is_ready(self):
        self.assertEqual(FPK.release_package().state, packager.STATE_READY)


# =============================================================================
# 1 — the schema: absence is a value, and it cannot be omitted
# =============================================================================

class AbsenceIsRepresentableTest(unittest.TestCase):

    def test_every_section_is_mandatory(self):
        """A dropped section is a VIOLATION, not a default. This is the whole
        contract: the old panel rendered clean because nothing objected to a
        producer that reported nothing at all.
        """
        for name in S.DASHBOARD_SECTIONS:
            with self.subTest(section=name):
                doc = document()
                del doc["sections"][name]
                violations = S.validate_kernel_dashboard_v2(doc)
                self.assertTrue(violations)
                self.assertTrue(any(f"sections.{name}" in v for v in violations),
                                violations)

    def test_an_unknown_section_is_refused(self):
        doc = document()
        doc["sections"]["telemetry"] = {"status": S.SECTION_OBSERVED, "as_of": None}
        self.assertTrue(any("telemetry" in v
                            for v in S.validate_kernel_dashboard_v2(doc)))

    def test_an_unreported_section_must_say_why(self):
        doc = document()
        doc["sections"][S.DASHBOARD_SECTION_CHAMPION] = {
            "status": S.SECTION_NOT_REPORTED, "as_of": None}
        doc["degraded"] = True
        doc["unreported_sections"] = [S.DASHBOARD_SECTION_CHAMPION]
        doc["produced_at"] = doc["generated_at"] = S.dashboard_liveness_timestamp(
            doc["sections"])
        violations = S.validate_kernel_dashboard_v2(doc)
        self.assertTrue(any("reason" in v for v in violations), violations)

    def test_an_unreported_section_may_not_carry_a_timestamp(self):
        """A `not_reported` section with an `as_of` would lift `produced_at`, which
        is the dead producer wearing the live one's timestamp.
        """
        doc = document()
        doc["sections"][S.DASHBOARD_SECTION_CHAMPION] = {
            "status": S.SECTION_NOT_REPORTED, "as_of": LATER, "reason": "no champion"}
        self.assertTrue(any("as_of" in v
                            for v in S.validate_kernel_dashboard_v2(doc)))

    def test_an_observed_liveness_section_must_carry_a_record_time(self):
        doc = document()
        doc["sections"][S.DASHBOARD_SECTION_CAMPAIGN]["as_of"] = None
        violations = S.validate_kernel_dashboard_v2(doc)
        self.assertTrue(any("as_of" in v for v in violations), violations)

    def test_a_live_host_section_may_be_observed_without_a_record_time(self):
        """The compliant-path control for the rule above: `headroom` and
        `resource_claims` are live readings with no journaled record behind them,
        and requiring an `as_of` there would forbid the legitimate case.
        """
        for name in (S.DASHBOARD_SECTION_HEADROOM, S.DASHBOARD_SECTION_CLAIMS):
            with self.subTest(section=name):
                doc = document()
                doc["sections"][name]["as_of"] = None
                self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])


class DerivedSummariesCannotBeStampedTest(unittest.TestCase):
    """`degraded` and `unreported_sections` are recomputed by the validator."""

    def test_a_document_may_not_claim_health_while_a_section_is_unreported(self):
        doc = document(champion=DC.Unreported(reason="no champion yet"))
        self.assertTrue(doc["degraded"])
        doc["degraded"] = False
        violations = S.validate_kernel_dashboard_v2(doc)
        self.assertTrue(any("degraded" in v for v in violations), violations)

    def test_the_unreported_list_may_not_disagree_with_the_sections(self):
        doc = document(champion=DC.Unreported(reason="no champion yet"))
        doc["unreported_sections"] = []
        violations = S.validate_kernel_dashboard_v2(doc)
        self.assertTrue(any("unreported_sections" in v for v in violations),
                        violations)

    def test_an_omitted_section_counts_as_unreported(self):
        """The two absences a consumer must never have to tell apart."""
        sections = document()["sections"]
        del sections[S.DASHBOARD_SECTION_CLAIMS]
        self.assertIn(S.DASHBOARD_SECTION_CLAIMS,
                      S.dashboard_unreported_sections(sections))


# =============================================================================
# 2 — liveness: a dead loop cannot be made to look alive
# =============================================================================

class LivenessIsDerivedTest(unittest.TestCase):

    def test_produced_at_is_the_newest_loop_record_timestamp(self):
        doc = document()
        expected = max(doc["sections"][name]["as_of"]
                       for name in S.DASHBOARD_LIVENESS_SECTIONS)
        self.assertEqual(doc["produced_at"], expected)

    def test_produced_at_may_not_be_hand_stamped(self):
        """THE anti-scar assertion. A producer that could write `produced_at`
        itself could report a controller that stopped in July as fresh today.
        """
        doc = document()
        doc["produced_at"] = doc["generated_at"] = EXPORTED_AT
        violations = S.validate_kernel_dashboard_v2(doc)
        self.assertTrue(any("produced_at" in v for v in violations), violations)

    def test_a_live_host_reading_cannot_lift_produced_at(self):
        """The bite for `DASHBOARD_LIVENESS_SECTIONS`. The controller last moved a
        month ago; disk and claims were measured seconds ago. If host readings
        counted, this document would read as fresh — a surface process that is
        merely alive reporting a controller that is dead, which is the 23-hour
        outage rebuilt one layer up.
        """
        stale = controller(last_transition=transition(at=LONG_AGO))
        doc = document(
            controller=stale,
            champion=DC.Unreported(reason="champion view not rebuilt since the stop"),
            readiness=DC.Unreported(reason="no readiness signal since the stop"),
            release_package=DC.Unreported(reason="no package"),
            blocking=DC.Unreported(reason="blocking view not computed"))
        self.assertEqual(doc["sections"][S.DASHBOARD_SECTION_HEADROOM]["as_of"],
                         EXPORTED_AT)
        self.assertEqual(doc["sections"][S.DASHBOARD_SECTION_CLAIMS]["as_of"],
                         EXPORTED_AT)
        self.assertEqual(doc["produced_at"], LONG_AGO)   # the bite
        self.assertNotEqual(doc["produced_at"], doc["exported_at"])

    def test_a_fully_dead_loop_produces_a_null_timestamp_and_a_degraded_flag(self):
        """Everything a consumer needs to render the ABSENCE rather than a clean
        panel: no timestamp at all, `degraded` true, and all seven sections named.
        """
        doc = document(
            controller=DC.Unreported(reason="controller ledger unreadable"),
            champion=DC.Unreported(reason="no champion view"),
            readiness=DC.Unreported(reason="no readiness report"),
            headroom=DC.Unreported(reason="storage probe failed", refused=True),
            blocking=DC.Unreported(reason="not derivable without the controller"),
            claims=DC.Unreported(reason="claim root unreadable"),
            release_package=DC.Unreported(reason="no package"))
        self.assertIsNone(doc["produced_at"])
        self.assertIsNone(doc["generated_at"])
        self.assertTrue(doc["degraded"])
        self.assertEqual(doc["unreported_sections"], sorted(S.DASHBOARD_SECTIONS))
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])
        for section in doc["sections"].values():
            self.assertTrue(section["reason"])

    def test_a_no_op_re_export_does_not_move_the_semantic_timestamp(self):
        """Re-exporting the same state an hour later changes `exported_at` and
        NOTHING else. `server.py` classifies freshness "from semantic run
        timestamps, not file mtime" precisely so a re-export cannot read as fresh;
        this is that property held at the producer end.
        """
        first = document()
        second = document(exported_at="2026-08-03T23:59:00+00:00")
        self.assertNotEqual(first["exported_at"], second["exported_at"])
        self.assertEqual(first["produced_at"], second["produced_at"])
        first.pop("exported_at"), second.pop("exported_at")
        self.assertEqual(S.canonical_json(first), S.canonical_json(second))

    def test_generated_at_carries_the_same_semantic_value_as_produced_at(self):
        """v1 compatibility, and it must be the SEMANTIC value: the deployed hub
        reads `generated_at`, so putting the export time there would make an old
        reader classify a dead loop as fresh.
        """
        doc = document()
        self.assertEqual(doc["generated_at"], doc["produced_at"])
        doc["generated_at"] = EXPORTED_AT
        self.assertTrue(any("generated_at" in v
                            for v in S.validate_kernel_dashboard_v2(doc)))

    def test_run_identity_lets_a_consumer_compare_two_exports_without_the_filesystem(self):
        doc = document()
        run = doc["producer"]["run"]
        self.assertEqual(run["campaign_id"], CAMPAIGN)
        self.assertEqual(run["controller_seq"], 7)
        self.assertEqual(run["controller_state"], SM.POST_RUN_CRITIC)
        self.assertEqual(run["ledger_receipt"], transition().receipt)
        self.assertEqual(doc["producer"]["module_id"], DC.MODULE_ID)

    def test_run_identity_is_null_when_no_controller_reported(self):
        """A producing run is not invented. If the controller did not report, the
        document says so rather than naming a run nobody can check.
        """
        doc = document(controller=DC.Unreported(reason="ledger unreadable"),
                       blocking=DC.Unreported(reason="not derivable"))
        self.assertIsNone(doc["producer"]["run"])
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])


# =============================================================================
# 3 — DERIVE, never restate
# =============================================================================

class EveryFieldComesFromItsOwnerTest(unittest.TestCase):

    def test_standing_is_copied_from_the_signal_not_recomputed(self):
        """THE drift bite. The signal is handed a standing that contradicts its own
        green phases; the contract must report the OWNER's value. A producer that
        re-derived standing from the phases would answer `objective_met` here and
        would then disagree with the module a freeze decision actually consults.
        """
        signal = FRD.green_signal()
        contradictory = stored_standing(signal, R.STANDING_NOT_MET,
                                        (R.BLOCK_ANCHOR_MOVED,))
        self.assertEqual(
            [p.non_inferior.outcome for p in contradictory.phases],
            [S.PASS] * len(contradictory.phases))
        doc = document(readiness=readiness_report(signals=(contradictory,)))
        backend = doc["sections"][S.DASHBOARD_SECTION_BACKEND_STANDING]["backends"]
        self.assertEqual(backend["llama_cpu"]["standing"], R.STANDING_NOT_MET)
        self.assertEqual(backend["llama_cpu"]["blockers"], [R.BLOCK_ANCHOR_MOVED])
        self.assertEqual(backend["llama_cpu"]["reducer_id"], signal.reducer_id)

    def test_storage_numbers_are_copied_from_storage_state(self):
        state = storage.StorageState(state=storage.DISK_PRESSURE, free_bytes=3,
                                     total_bytes=100, floor_bytes=50,
                                     reasons=("free below floor",))
        doc = document(headroom=headroom(storage_state=state))
        block = doc["sections"][S.DASHBOARD_SECTION_HEADROOM]["storage"]
        self.assertEqual(block["free_bytes"], 3)
        self.assertEqual(block["state"], storage.DISK_PRESSURE)
        self.assertIs(block["pressured"], state.pressured)
        self.assertEqual(block["reasons"], ["free below floor"])

    def test_package_state_is_copied_from_the_packager(self):
        package = FPK.release_package()
        doc = document(release_package=package)
        block = doc["sections"][S.DASHBOARD_SECTION_RELEASE_PACKAGE]
        self.assertEqual(block["state"], package.state)
        self.assertEqual(block["package_id"], package.package_id)
        self.assertEqual(block["requires_human_code_review"],
                         package.requires_human_code_review)
        self.assertEqual(block["executed_by"], packager.EXECUTED_BY)

    def test_champion_membership_is_copied_from_the_champion_record(self):
        record = champion_record()
        block = document()["sections"][S.DASHBOARD_SECTION_CHAMPION]
        self.assertEqual(block["combined_candidate_id"],
                         record["combined_candidate_id"])
        self.assertEqual(block["member_candidate_ids"], record["member_candidates"])
        self.assertEqual(block["readiness"]["reference_signal"],
                         record["readiness"]["reference_signal"])
        self.assertEqual(block["branch"], record["branch"])

    def test_an_invalid_champion_record_is_refused_not_rendered(self):
        bad = champion_record(branch="production-consolidated-v8")
        with self.assertRaises(DC.ContractInputError):
            document(champion=bad)

    def test_a_champion_without_a_record_time_is_refused(self):
        bad = champion_record()
        del bad["created_at"]
        with self.assertRaises(DC.ContractInputError):
            document(champion=bad)

    def test_a_could_not_check_verdict_stays_distinguishable(self):
        """A `COULD_NOT_CHECK` folded into PASS or FAIL turns "we could not tell"
        into an answer — the same defect class as a clean panel over a hole.
        """
        rendered = DC._check_dict(S.Check(S.COULD_NOT_CHECK, ("no anchor",)))
        self.assertEqual(rendered,
                         {"outcome": S.COULD_NOT_CHECK, "reasons": ["no anchor"]})


class BlockingConditionsComeFromTheirOwnersTest(unittest.TestCase):

    def test_every_emitted_kind_belongs_to_an_owning_vocabulary(self):
        """No private taxonomy. Each `kind` is a constant owned by the module that
        decides the block, so the surface cannot drift from the code that blocks.
        """
        owned = (set(SM.STOP_STATES) | set(R.BLOCKERS)
                 | {storage.DISK_PRESSURE, packager.STATE_BLOCKED}
                 # The champion record's own vocabulary: the entries it may carry
                 # in `blocking_conditions` are composition's constants, and
                 # `CHAMPION_BLOCKED_UNNAMED` is defined beside the champion schema
                 # rather than here — a fallback name minted by the surface would be
                 # the private taxonomy this test exists to forbid.
                 | {COMP.BLOCKING_REANCHOR_REMEASURE, S.CHAMPION_BLOCKED_UNNAMED}
                 | {status.upper() for status in (
                     R.PhaseTradeAssessment.STATUS_OUTSIDE_BAND,
                     R.PhaseTradeAssessment.STATUS_NOT_PREDECLARED,
                     R.PhaseTradeAssessment.STATUS_WITHIN_BAND,
                     R.PhaseTradeAssessment.STATUS_NOT_APPLICABLE)})
        blocked = stored_standing(FRD.green_signal(), R.STANDING_NOT_MET,
                                  (R.BLOCK_ANCHOR_MOVED, R.BLOCK_COVERAGE_GAP))
        stopped = controller(state=SM.ANCHOR_MOVED, stopped=True,
                             last_transition=transition(to_state=SM.ANCHOR_MOVED))
        observation = DC.derive_blocking_conditions(
            controller=stopped,
            readiness_report=readiness_report(signals=(blocked,)),
            release_package=FPK.release_package(),
            headroom=headroom(storage_state=storage.StorageState(
                state=storage.DISK_PRESSURE, free_bytes=1, total_bytes=2,
                floor_bytes=3, reasons=("free below floor",))),
            champion=champion_record(
                blocking_conditions=["REANCHOR_PENDING_REMEASURE"],
                last_t1={"event_id": "ake-1", "status": "fail"}))
        kinds = {condition.kind for condition in observation.conditions}
        self.assertTrue(kinds)
        self.assertEqual(kinds - owned, set())
        self.assertIn(SM.ANCHOR_MOVED, kinds)
        self.assertIn(R.BLOCK_COVERAGE_GAP, kinds)
        self.assertIn(storage.DISK_PRESSURE, kinds)

    def test_an_evaluator_coverage_gap_is_named_by_the_controllers_constant(self):
        gap = FakeGap(missing_class="gpu_correctness", blocked_lineage="akc-0009",
                      owner="inference", deadline="2026-08-10T00:00:00+00:00")
        observation = DC.derive_blocking_conditions(
            controller=controller(), readiness_report=readiness_report(),
            release_package=FPK.release_package(), coverage_gaps=(gap,))
        kinds = {c.kind: c for c in observation.conditions}
        self.assertIn(SM.EVALUATOR_COVERAGE_GAP, kinds)
        self.assertEqual(kinds[SM.EVALUATOR_COVERAGE_GAP].owner, "inference")
        self.assertEqual(kinds[SM.EVALUATOR_COVERAGE_GAP].deadline,
                         "2026-08-10T00:00:00+00:00")

    def test_the_successful_terminus_is_not_reported_as_a_block(self):
        """`RELEASE_PACKAGE_READY` is in `STOP_STATES` because the loop stopped
        advancing, not because anything is wrong. Reporting the finish line as a
        blocker is how a surface teaches an operator to ignore its blockers.
        """
        done = controller(state=SM.RELEASE_PACKAGE_READY, stopped=True,
                          last_transition=transition(to_state=SM.RELEASE_PACKAGE_READY))
        observation = DC.derive_blocking_conditions(
            controller=done, readiness_report=readiness_report(),
            release_package=FPK.release_package())
        self.assertNotIn(SM.RELEASE_PACKAGE_READY,
                         {c.kind for c in observation.conditions})

    def test_a_stopped_controller_on_any_other_state_is_reported(self):
        """The compliant-path control for the exclusion above: it excludes exactly
        one state, not stop reporting in general.
        """
        for state in SM.STOP_STATES:
            if state == SM.RELEASE_PACKAGE_READY:
                continue
            with self.subTest(state=state):
                observation = DC.derive_blocking_conditions(
                    controller=controller(state=state, stopped=True,
                                          last_transition=transition(to_state=state)),
                    readiness_report=DC.Unreported(reason="none"),
                    release_package=DC.Unreported(reason="none"))
                self.assertIn(state, {c.kind for c in observation.conditions})

    def test_the_derivation_is_stable_across_two_identical_calls(self):
        args = dict(controller=controller(), readiness_report=readiness_report(),
                    release_package=FPK.release_package(), headroom=headroom())
        first = DC.derive_blocking_conditions(**args)
        second = DC.derive_blocking_conditions(**args)
        self.assertEqual([c.to_dict() for c in first.conditions],
                         [c.to_dict() for c in second.conditions])


class FakeGap:
    """A `controller.context.CoverageGap`-shaped record, by field name only.

    Deliberately structural: `derive_blocking_conditions` reads four attributes and
    nothing else, so importing the controller's context plane here would couple this
    suite to a module the producer does not need in order to name a gap.
    """

    def __init__(self, *, missing_class, blocked_lineage, owner, deadline):
        self.missing_class = missing_class
        self.blocked_lineage = blocked_lineage
        self.owner = owner
        self.deadline = deadline


# =============================================================================
# 4 — inputs: None is not an absence, it is a defect
# =============================================================================

class NoneIsNotAnInputTest(unittest.TestCase):

    def test_none_is_refused_for_every_section(self):
        """The reflex that built every absence-tolerant panel in this project is
        `None`. It has to cost something at the input boundary, or the reason the
        operator reads when the panel is empty never gets written down.
        """
        for name in ("controller", "champion", "readiness", "headroom",
                     "blocking", "claims", "release_package"):
            with self.subTest(field=name):
                with self.assertRaises(DC.ContractInputError) as caught:
                    inputs(**{name: None})
                self.assertIn("Unreported", str(caught.exception))

    def test_unreported_requires_a_reason(self):
        with self.assertRaises(DC.ContractInputError):
            DC.Unreported(reason="")

    def test_a_wrong_type_is_refused(self):
        with self.assertRaises(DC.ContractInputError):
            inputs(readiness={"standing": "objective_met"})

    def test_an_unreported_section_renders_with_its_reason(self):
        doc = document(claims=DC.Unreported(reason="claim root unreadable",
                                            refused=True))
        section = doc["sections"][S.DASHBOARD_SECTION_CLAIMS]
        self.assertEqual(section["status"], S.SECTION_REFUSED)
        self.assertEqual(section["reason"], "claim root unreadable")
        self.assertTrue(doc["degraded"])
        self.assertEqual(doc["unreported_sections"], [S.DASHBOARD_SECTION_CLAIMS])

    def test_a_campaign_id_must_be_a_campaign_id(self):
        with self.assertRaises(DC.ContractInputError):
            inputs(campaign_id="llama_cpu-20260803")


class ObserveControllerTest(unittest.TestCase):
    """The phase comes off a REAL `ControllerStateMachine`, not a stub.

    `controller/test_state_machine.py` has a ready-made fixture and it is NOT
    reused here, for the one reason the README calls out: that suite imports the
    package FLAT (`from autokernel.controller import state_machine`), which loads a
    SECOND copy of every module. A `Transition` built there is not an instance of
    the `Transition` this producer imported, so every `isinstance` guard in the
    seam would degrade to a silent no-op — the guard would still be there and would
    still be checking nothing, which is the failure mode this whole suite is about.
    So the machine is built here, under the package-relative identity.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir="/mnt/raid0/llm")
        self.addCleanup(self.tmp.cleanup)
        self.journal = J.Journal(os.path.join(self.tmp.name, "journal"),
                                 campaign_id=CAMPAIGN)
        self.journal.initialize()

    def machine(self) -> SM.ControllerStateMachine:
        return SM.ControllerStateMachine(
            journal_=self.journal, root=os.path.join(self.tmp.name, "controller"),
            campaign_id=CAMPAIGN)

    def walked(self) -> SM.ControllerStateMachine:
        machine = self.machine()
        for step in (SM.DISCOVER, SM.SELECT_TARGET):
            machine.transition(step, trigger="test", reason="walk to fixture state")
        return machine

    def test_the_transition_the_ledger_returns_is_the_one_this_module_guards_on(self):
        """Anti-vacuity for the isinstance guard above: if these two classes were
        ever different objects, the guard would pass everything and the test below
        would prove nothing.
        """
        machine = self.walked()
        self.assertIsInstance(machine.ledger.read().transitions[-1], SM.Transition)

    def test_the_observation_matches_the_machines_own_accessors(self):
        machine = self.walked()
        observed = DC.observe_controller(machine, campaign_id=CAMPAIGN)
        self.assertEqual(observed.state, machine.state)
        self.assertEqual(observed.seq, machine.seq)
        self.assertEqual(observed.stopped, machine.is_stopped())
        self.assertEqual(observed.last_transition.to_state, SM.SELECT_TARGET)

    def test_an_empty_ledger_is_refused_rather_than_reported_as_a_phase(self):
        """A machine that has recorded nothing has no phase to report. Reporting
        `BOOTSTRAP` here would be the surface inventing a state the ledger does not
        support — the AutoPilot derived-view scar in miniature.
        """
        with self.assertRaises(DC.ContractInputError):
            DC.observe_controller(self.machine(), campaign_id=CAMPAIGN)

    def test_the_observed_document_validates_end_to_end(self):
        machine = self.walked()
        observed = DC.observe_controller(machine, campaign_id=CAMPAIGN)
        doc = document(controller=observed,
                       blocking=DC.derive_blocking_conditions(
                           controller=observed, readiness_report=readiness_report(),
                           release_package=FPK.release_package()))
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])
        self.assertEqual(doc["producer"]["run"]["controller_seq"], machine.seq)


# =============================================================================
# 5 — the one write, and the fence around it
# =============================================================================

class ExportDestinationTest(unittest.TestCase):

    def test_the_default_path_is_accepted(self):
        self.assertEqual(DC.assert_exportable_destination(DC.DEFAULT_EXPORT_PATH),
                         DC.DEFAULT_EXPORT_PATH)

    def test_the_default_path_is_durable_by_the_storage_modules_own_test(self):
        """Not asserted in prose: `storage.is_scratch_path` is the SSOT, and the
        path this module replaces failed exactly this check.
        """
        self.assertFalse(storage.is_scratch_path(DC.DEFAULT_EXPORT_PATH))
        self.assertTrue(storage.is_scratch_path(
            "/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_dashboard.json"))

    def test_every_human_only_target_is_refused(self):
        for candidate in (
                "/mnt/raid0/llm/production-consolidated-v9/kernel_dashboard.json",
                "/mnt/raid0/llm/kernels/production/kernel_dashboard.json",
                "/workspace/orchestration/instrument_eras.yaml",
                "/workspace/orchestration/autopilot_baseline.yaml",
                "/workspace/coordination/session-bus/human_only_paths.yaml"):
            with self.subTest(path=candidate):
                with self.assertRaises(DC.ExportDestinationRefused):
                    DC.assert_exportable_destination(candidate)

    def test_a_symlink_into_a_production_tree_is_refused(self):
        """The bypass a literal-string guard misses. This repository's own
        working-tree identity rule makes symlinked roots the NORMAL case, so a
        check that only reads the string as written is not a check.
        """
        with tempfile.TemporaryDirectory() as tmp:
            link = pathlib.Path(tmp) / "surface"
            link.symlink_to("/mnt/raid0/llm/llama.cpp")
            with self.assertRaises(DC.ExportDestinationRefused) as caught:
                DC.assert_exportable_destination(str(link / "kernel_dashboard.json"))
            self.assertIn("production", str(caught.exception))

    def test_a_scratch_destination_is_refused(self):
        for candidate in ("/tmp/kernel_dashboard.json",
                          "/mnt/raid0/llm/tmp/kernel_dashboard.json",
                          "/dev/shm/kernel_dashboard.json"):
            with self.subTest(path=candidate):
                with self.assertRaises(DC.ExportDestinationRefused):
                    DC.assert_exportable_destination(candidate)

    def test_a_non_json_destination_and_a_directory_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DC.ExportDestinationRefused):
                DC.assert_exportable_destination(os.path.join(tmp, "surface.txt"))
            with self.assertRaises(DC.ExportDestinationRefused):
                DC.assert_exportable_destination(tmp)

    def test_an_ordinary_durable_destination_is_accepted(self):
        """The compliant-path control: the guard forbids human-only targets and
        scratch, never writing as such. A guard that refused everything would pass
        every test above while making the module useless.
        """
        with tempfile.TemporaryDirectory(dir="/mnt/raid0/llm") as tmp:
            target = os.path.join(tmp, "kernel_dashboard.json")
            self.assertEqual(DC.assert_exportable_destination(target), target)


class ExportTest(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir="/mnt/raid0/llm")
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, "kernel_dashboard.json")

    def test_the_export_round_trips_and_still_validates(self):
        doc = document()
        receipt = DC.export_contract(doc, path=self.target)
        self.assertEqual(receipt.path, self.target)
        reloaded = json.loads(pathlib.Path(self.target).read_text(encoding="utf-8"))
        self.assertEqual(reloaded, doc)
        self.assertEqual(S.validate_kernel_dashboard_v2(reloaded), [])
        self.assertEqual(receipt.sha256, S.content_hash(doc))
        self.assertEqual(receipt.produced_at, doc["produced_at"])
        self.assertFalse(receipt.degraded)

    def test_a_degraded_export_reports_its_absence_in_the_receipt(self):
        doc = document(champion=DC.Unreported(reason="no champion view"))
        receipt = DC.export_contract(doc, path=self.target)
        self.assertTrue(receipt.degraded)
        self.assertEqual(receipt.unreported_sections,
                         (S.DASHBOARD_SECTION_CHAMPION,))

    def test_an_invalid_document_is_refused_before_anything_is_written(self):
        """The file on disk is never a document the schema would reject: a consumer
        that meets a malformed export has to guess, and the guess that comes
        naturally is "treat it as empty".
        """
        doc = document()
        doc["degraded"] = False
        del doc["sections"][S.DASHBOARD_SECTION_CHAMPION]
        with self.assertRaises(DC.ContractInvalid):
            DC.export_contract(doc, path=self.target)
        self.assertFalse(os.path.exists(self.target))

    def test_a_refused_destination_writes_nothing(self):
        with self.assertRaises(DC.ExportDestinationRefused):
            DC.export_contract(document(), path="/tmp/kernel_dashboard.json")
        self.assertFalse(os.path.exists("/tmp/kernel_dashboard.json"))

    def test_the_write_leaves_no_temporary_behind(self):
        DC.export_contract(document(), path=self.target)
        leftovers = [name for name in os.listdir(self.tmp.name)
                     if name.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        self.assertEqual(os.listdir(self.tmp.name), ["kernel_dashboard.json"])

    def test_a_re_export_replaces_in_place_and_keeps_the_semantic_timestamp(self):
        first = DC.export_contract(document(), path=self.target)
        second = DC.export_contract(
            document(exported_at="2026-08-04T09:00:00+00:00"), path=self.target)
        self.assertEqual(first.produced_at, second.produced_at)
        self.assertNotEqual(first.exported_at, second.exported_at)
        self.assertEqual(os.listdir(self.tmp.name), ["kernel_dashboard.json"])

    def test_the_exported_bytes_are_canonical(self):
        doc = document()
        DC.export_contract(doc, path=self.target)
        self.assertEqual(pathlib.Path(self.target).read_bytes(),
                         S.canonical_bytes(doc))


# =============================================================================
# 6 — version handling: the consumer will meet both
# =============================================================================

LEGACY_V1 = {
    "db_present": True,
    "runs": [{"ts": "2026-07-30T10:00:00+00:00", "model": "qwen", "ok": True}],
    "pareto": [],
    "best_per_model": [],
    "totals": {"runs": 1, "correct": 1, "failed": 0, "models": 1},
    "generated_at": "2026-07-30T10:05:00+00:00",
    "observation_notice": "Every number here is an OBSERVATION (MEASUREMENT.md).",
}


class VersionHandlingTest(unittest.TestCase):

    def test_the_legacy_unlabelled_document_is_still_readable(self):
        """v1 carries no `schema` key at all, which is the whole reason the version
        is being made explicit now. A validator that demanded the label would make
        every real v1 file invalid and push a reader toward "unknown, render empty".
        """
        self.assertIsNone(LEGACY_V1.get("schema"))
        self.assertEqual(S.detect_kernel_dashboard_version(LEGACY_V1),
                         S.SCHEMA_KERNEL_DASHBOARD_V1)
        self.assertEqual(S.validate_kernel_dashboard(LEGACY_V1), [])

    def test_a_v2_document_is_detected_as_v2(self):
        doc = document()
        self.assertEqual(S.detect_kernel_dashboard_version(doc),
                         S.SCHEMA_KERNEL_DASHBOARD_V2)
        self.assertEqual(S.validate_kernel_dashboard(doc), [])

    def test_a_malformed_v2_document_is_not_demoted_to_v1(self):
        """A broken v2 misread as a valid v1 renders as an empty-but-clean panel —
        the failure this contract exists to prevent, arriving through the reader
        instead of the producer.
        """
        doc = document()
        del doc["sections"][S.DASHBOARD_SECTION_CHAMPION]
        self.assertEqual(S.detect_kernel_dashboard_version(doc),
                         S.SCHEMA_KERNEL_DASHBOARD_V2)
        self.assertTrue(S.validate_kernel_dashboard(doc))

    def test_an_unrecognisable_document_is_reported_not_guessed(self):
        for candidate in ({}, {"hello": "world"}, [], "text", None):
            with self.subTest(candidate=candidate):
                self.assertIsNone(S.detect_kernel_dashboard_version(candidate))
                self.assertTrue(S.validate_kernel_dashboard(candidate))

    def test_a_v1_document_with_a_broken_timestamp_is_reported(self):
        broken = copy.deepcopy(LEGACY_V1)
        broken["runs"][0]["ts"] = "yesterday"
        self.assertTrue(S.validate_kernel_dashboard(broken))

    def test_both_versions_have_a_validator_and_neither_is_a_journal_record(self):
        """Dispatch lives in `KERNEL_DASHBOARD_VALIDATORS`, not `SCHEMA_REGISTRY`.

        The boundary is deliberate. `SCHEMA_REGISTRY` is what `validate_record`
        dispatches journal lines through, so everything in it is something a
        journal may contain; this contract is a DERIVED EXPORT of those records,
        and appending one to a journal would put a rendering where evidence
        belongs. It could not be dispatched there honestly either: a legacy v1
        document carries no `schema` key, which `validate_record` requires.
        """
        self.assertEqual(set(S.KERNEL_DASHBOARD_VALIDATORS),
                         {S.SCHEMA_KERNEL_DASHBOARD_V1, S.SCHEMA_KERNEL_DASHBOARD_V2})
        self.assertEqual(S.SCHEMA_KERNEL_DASHBOARD, S.SCHEMA_KERNEL_DASHBOARD_V2)
        self.assertNotIn(S.SCHEMA_KERNEL_DASHBOARD_V2, S.SCHEMA_REGISTRY)
        self.assertNotIn(S.SCHEMA_KERNEL_DASHBOARD_V1, S.SCHEMA_REGISTRY)
        # ...and a dashboard document handed to the journal reader is reported as
        # an unknown record rather than quietly accepted.
        self.assertTrue(S.validate_record(document()))

    def test_the_retrieval_view_knows_both_versions(self):
        """An absent `NON_RETRIEVABLE_FIELDS` entry makes `retrievable_view` raise
        on an otherwise valid record, which is a crash in a reader, not a schema
        question.
        """
        doc = document()
        self.assertEqual(S.retrievable_view(doc), doc)
        labelled_v1 = dict(LEGACY_V1, schema=S.SCHEMA_KERNEL_DASHBOARD_V1)
        self.assertEqual(S.retrievable_view(labelled_v1), labelled_v1)


# =============================================================================
# 7 — the record carries no authority, and canonicalises
# =============================================================================

class RecordDisciplineTest(unittest.TestCase):

    def test_the_contract_carries_no_authority_flavoured_key(self):
        self.assertEqual(S.find_authority_flavoured_keys(document()), [])

    def test_an_authority_flavoured_key_is_refused(self):
        doc = document()
        doc["approved_for_cutover"] = True
        self.assertTrue(S.validate_kernel_dashboard_v2(doc))

    def test_the_document_canonicalises(self):
        """Other modules content-hash this record; a tuple or a NaN in it would
        make the export unequal to its own reload.
        """
        doc = document()
        self.assertEqual(json.loads(S.canonical_json(doc)), doc)

    def test_the_observation_notice_says_what_an_empty_panel_means(self):
        notice = document()["observation_notice"]
        self.assertIn("OBSERVATION", notice)
        self.assertIn("produced_at", notice)


# =============================================================================
# 8 — the adversarial pass: what the module let through before it was attacked
# =============================================================================
#
# Every test in this section corresponds to a defect that was PRESENT and
# demonstrated on the built module, not to a hypothetical. They are grouped
# together and named after the attack rather than the fix, because the next
# person to change this module needs to know what the guard is holding back.


class TheExporterClockCannotDateADerivedSectionTest(unittest.TestCase):
    """ATTACK C. `blocking_conditions` was a liveness section whose `as_of` came
    from an argument. Handing it the exporter's wall clock made a month-dead loop
    render `produced_at: now`, `degraded: false`, and the document validated.
    """

    def _month_old_inputs(self, **over):
        """Every JOURNALED record a month old — and nothing else changed. The host
        readings (`headroom`, `resource_claims`) stay measured seconds ago, which is
        what a live exporter over a dead loop actually looks like.
        """
        ctrl = controller(last_transition=transition(at=LONG_AGO))
        champ = champion_record(created_at=LONG_AGO)
        report = readiness_report(computed_at=LONG_AGO)
        package = FPK.release_package(created_at=LONG_AGO)
        return dict(controller=ctrl, champion=champ, readiness=report,
                    release_package=package, **over)

    def test_derive_blocking_conditions_takes_no_timestamp_at_all(self):
        """The parameter is GONE, not validated. A caller cannot pass what the
        signature does not accept, and a signature is checked by every call site
        at once.
        """
        import inspect
        parameters = inspect.signature(DC.derive_blocking_conditions).parameters
        self.assertNotIn("as_of", parameters)

    def test_a_supplied_blocking_timestamp_is_refused(self):
        """The other door: `BlockingObservation` is public, so the guard lives in
        `__post_init__` rather than only in the derivation helper.
        """
        conditions = (DC.BlockingCondition(
            kind=SM.ANCHOR_MOVED, origin="controller_stop",
            detail="stopped in July", since=LONG_AGO),)
        with self.assertRaises(DC.ContractInputError) as caught:
            DC.BlockingObservation(conditions=conditions, as_of=EXPORTED_AT)
        self.assertIn("DERIVED", str(caught.exception))

    def test_the_blocking_timestamp_is_the_newest_record_it_carries(self):
        """The compliant-path control: the field still gets a value, and the value
        is the conditions' own newest record time — passing that value explicitly
        is accepted, because it is not a stamp, it is agreement.
        """
        conditions = (
            DC.BlockingCondition(kind=SM.ANCHOR_MOVED, origin="controller_stop",
                                 detail="stopped in July", since=LONG_AGO),
            DC.BlockingCondition(kind=R.BLOCK_COVERAGE_GAP, origin="readiness",
                                 detail="gap", since=NOW),
        )
        self.assertEqual(DC.BlockingObservation(conditions=conditions).as_of, NOW)
        self.assertEqual(
            DC.BlockingObservation(conditions=conditions, as_of=NOW).as_of, NOW)

    def test_a_condition_with_no_record_time_does_not_default_to_now(self):
        gap = FakeGap(missing_class="gpu_correctness", blocked_lineage="akc-0009",
                      owner="inference", deadline="2026-08-10T00:00:00+00:00")
        observation = DC.derive_blocking_conditions(
            controller=controller(), readiness_report=readiness_report(),
            release_package=FPK.release_package(), coverage_gaps=(gap,))
        gaps = [c for c in observation.conditions if c.origin == "evaluator_coverage"]
        self.assertTrue(gaps)
        self.assertIsNone(gaps[0].since)

    def test_a_derived_section_is_not_a_liveness_source(self):
        """The second, independent guard. Even if a record timestamp somehow got
        into this section, it may not ESTABLISH freshness: every condition in it is
        restated from a section that already speaks for itself.
        """
        self.assertNotIn(S.DASHBOARD_SECTION_BLOCKING, S.DASHBOARD_LIVENESS_SECTIONS)
        self.assertIn(S.DASHBOARD_SECTION_BLOCKING, S.DASHBOARD_SECTIONS)

    def test_a_month_dead_loop_cannot_read_as_fresh_or_undegraded(self):
        """THE attack, end to end. Before the fix this document validated with
        `produced_at` equal to the export time and `degraded: false`.
        """
        doc = document(**self._month_old_inputs())
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])
        self.assertEqual(doc["produced_at"], LONG_AGO)
        self.assertNotEqual(doc["produced_at"], doc["exported_at"])

    def test_a_dead_loop_with_a_live_exporter_still_produces_a_null_timestamp(self):
        """The purest form: every owner silent, the exporter alive and stamping.
        `produced_at` must be null — the value every consumer classifies as
        `missing` — and not the moment the exporter happened to run.
        """
        dead = DC.Unreported(reason="producer did not report")
        blocking = DC.derive_blocking_conditions(
            controller=dead, readiness_report=dead, release_package=dead,
            headroom=dead, champion=dead)
        doc = document(controller=dead, champion=dead, readiness=dead, headroom=dead,
                       claims=dead, release_package=dead, blocking=blocking)
        self.assertIsNone(doc["produced_at"])
        self.assertIsNone(doc["generated_at"])
        self.assertTrue(doc["degraded"])
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])

    def test_the_panel_names_the_owners_that_did_not_report(self):
        """"No open blocking conditions" and "nobody told me about the blocking
        conditions" rendered identically. Now the panel says which.
        """
        dead = DC.Unreported(reason="producer did not report")
        blocking = DC.derive_blocking_conditions(
            controller=dead, readiness_report=dead, release_package=dead,
            headroom=dead, champion=dead)
        self.assertEqual(
            sorted(blocking.unreported_owners),
            ["champion", "controller", "headroom", "readiness", "release_package"])
        doc = document(controller=dead, champion=dead, readiness=dead, headroom=dead,
                       claims=dead, release_package=dead, blocking=blocking)
        section = doc["sections"][S.DASHBOARD_SECTION_BLOCKING]
        self.assertEqual(section["open"], [])
        self.assertTrue(section["unreported_owners"])

    def test_a_fully_reported_panel_names_no_silent_owner(self):
        """The compliant-path control: the baseline names nobody, so the list above
        is a signal rather than noise every document carries.
        """
        section = document()["sections"][S.DASHBOARD_SECTION_BLOCKING]
        self.assertEqual(section["unreported_owners"], [])


class TheBlockingPanelMayNotContradictItsOwnDocumentTest(unittest.TestCase):
    """ATTACK D. `blocking` was whatever the caller handed over, checked against
    nothing — so the panel an operator reads to answer "is anything wrong?" could
    be empty while the sections beside it named three separate blocks.
    """

    EMPTY = property(lambda self: DC.BlockingObservation())

    def test_an_empty_panel_is_refused_while_readiness_reports_a_blocker(self):
        blocked = stored_standing(FRD.green_signal(), R.STANDING_NOT_MET,
                                  (R.BLOCK_ANCHOR_MOVED,))
        with self.assertRaises(DC.ContractInputError) as caught:
            document(readiness=readiness_report(signals=(blocked,)),
                     blocking=DC.BlockingObservation())
        self.assertIn(R.BLOCK_ANCHOR_MOVED, str(caught.exception))

    def test_an_empty_panel_is_refused_while_the_controller_is_stopped(self):
        for state in sorted(SM.STOP_STATES):
            if state == SM.RELEASE_PACKAGE_READY:
                continue
            with self.subTest(state=state):
                stopped = controller(state=state, stopped=True,
                                     last_transition=transition(to_state=state))
                with self.assertRaises(DC.ContractInputError):
                    document(controller=stopped, blocking=DC.BlockingObservation())

    def test_an_empty_panel_is_refused_while_the_champion_is_held(self):
        held = champion_record(blocking_conditions=[SM.EVALUATOR_COVERAGE_GAP],
                               last_t1={"event_id": "ake-1", "status": "fail"})
        with self.assertRaises(DC.ContractInputError) as caught:
            document(champion=held, blocking=DC.BlockingObservation())
        self.assertIn(SM.EVALUATOR_COVERAGE_GAP, str(caught.exception))

    def test_a_champion_condition_reaches_the_panel_with_its_own_name(self):
        held = champion_record(blocking_conditions=[SM.EVALUATOR_COVERAGE_GAP],
                               last_t1={"event_id": "ake-1", "status": "fail"})
        doc = document(champion=held)
        panel = doc["sections"][S.DASHBOARD_SECTION_BLOCKING]["open"]
        kinds = {entry["kind"]: entry for entry in panel}
        self.assertIn(SM.EVALUATOR_COVERAGE_GAP, kinds)
        self.assertEqual(kinds[SM.EVALUATOR_COVERAGE_GAP]["origin"], "champion")
        self.assertEqual(doc["sections"][S.DASHBOARD_SECTION_CHAMPION]
                         ["blocking_conditions"], [SM.EVALUATOR_COVERAGE_GAP])

    def test_a_champion_condition_not_in_the_machine_vocabulary_is_still_surfaced(self):
        """A prose entry cannot be a `kind` (the schema requires UPPER_SNAKE), and
        dropping it would be silence. It is carried under the fallback name the
        CHAMPION SCHEMA owns, with its own text as the detail.
        """
        held = champion_record(blocking_conditions=["waiting on the operator"],
                               last_t1={"event_id": "ake-1", "status": "fail"})
        doc = document(champion=held)
        panel = doc["sections"][S.DASHBOARD_SECTION_BLOCKING]["open"]
        folded = [e for e in panel if e["kind"] == S.CHAMPION_BLOCKED_UNNAMED]
        self.assertEqual(len(folded), 1)
        self.assertIn("waiting on the operator", folded[0]["detail"])
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])

    def test_a_panel_derived_from_the_same_inputs_is_accepted(self):
        """THE compliant-path control. The check must forbid contradiction, not
        forbid a blocked campaign from being reported at all.
        """
        blocked = stored_standing(FRD.green_signal(), R.STANDING_NOT_MET,
                                  (R.BLOCK_ANCHOR_MOVED,))
        doc = document(readiness=readiness_report(signals=(blocked,)))
        kinds = {entry["kind"]
                 for entry in doc["sections"][S.DASHBOARD_SECTION_BLOCKING]["open"]}
        self.assertIn(R.BLOCK_ANCHOR_MOVED, kinds)
        self.assertEqual(S.validate_kernel_dashboard_v2(doc), [])

    def test_more_conditions_than_this_module_can_rebuild_are_allowed(self):
        """Containment, not equality: evaluator coverage gaps come from
        `controller.context`, which `ContractInputs` does not carry. A caller must
        be able to report MORE than the contract can rebuild — never less.
        """
        gap = FakeGap(missing_class="gpu_correctness", blocked_lineage="akc-0009",
                      owner="inference", deadline="2026-08-10T00:00:00+00:00")
        rich = DC.derive_blocking_conditions(
            controller=controller(), readiness_report=readiness_report(),
            release_package=FPK.release_package(), headroom=headroom(),
            champion=champion_record(), coverage_gaps=(gap,))
        doc = document(blocking=rich)
        kinds = {entry["kind"]
                 for entry in doc["sections"][S.DASHBOARD_SECTION_BLOCKING]["open"]}
        self.assertIn(SM.EVALUATOR_COVERAGE_GAP, kinds)

    def test_an_absent_panel_is_still_allowed_and_still_degrades(self):
        """The other compliant path: `Unreported` is honest — it renders as absence
        and counts toward `degraded`. It is the SILENTLY EMPTY panel that lies.
        """
        blocked = stored_standing(FRD.green_signal(), R.STANDING_NOT_MET,
                                  (R.BLOCK_ANCHOR_MOVED,))
        doc = document(readiness=readiness_report(signals=(blocked,)),
                       blocking=DC.Unreported(reason="blocking view not computed"))
        self.assertTrue(doc["degraded"])
        self.assertIn(S.DASHBOARD_SECTION_BLOCKING, doc["unreported_sections"])


class TheOneWriterCannotReachAShardCheckoutTest(unittest.TestCase):
    """ATTACK A. The destination guard refused production trees, human-only targets
    and scratch — and accepted `epyc-inference-research/data/kernel_dashboard.json`
    and `/workspace/handoffs/active/kernel_dashboard.json`. Both are clones shared
    with other live sessions, and this file is rewritten on every export.
    """

    def test_a_destination_inside_a_git_working_tree_is_refused(self):
        with tempfile.TemporaryDirectory(dir="/mnt/raid0/llm") as tmp:
            checkout = pathlib.Path(tmp) / "repo"
            (checkout / "data").mkdir(parents=True)
            (checkout / ".git").mkdir()
            target = checkout / "data" / "kernel_dashboard.json"
            with self.assertRaises(DC.ExportDestinationRefused) as caught:
                DC.assert_exportable_destination(str(target))
            self.assertIn("git working tree", str(caught.exception))

    def test_a_linked_worktree_is_refused_too(self):
        """`git worktree add` leaves a `.git` FILE, not a directory, and this
        repository's identity rule makes linked worktrees and shared clones normal.
        A guard that only tested `is_dir()` would pass this.
        """
        with tempfile.TemporaryDirectory(dir="/mnt/raid0/llm") as tmp:
            checkout = pathlib.Path(tmp) / "worktree"
            checkout.mkdir()
            (checkout / ".git").write_text("gitdir: /elsewhere/.git/worktrees/w\n",
                                           encoding="utf-8")
            with self.assertRaises(DC.ExportDestinationRefused):
                DC.assert_exportable_destination(
                    str(checkout / "kernel_dashboard.json"))

    def test_the_real_research_checkout_is_refused(self):
        """Not a synthetic tree: the actual clone this suite runs inside, which the
        guard accepted before.
        """
        root = pathlib.Path(__file__).resolve().parents[4]
        self.assertTrue((root / ".git").exists(), root)
        with self.assertRaises(DC.ExportDestinationRefused):
            DC.assert_exportable_destination(str(root / "data" / "kernel.json"))

    def test_the_checkout_is_recognised_through_the_symlinked_working_tree(self):
        """`/workspace/repos/<name>` is a SYMLINK to `/mnt/raid0/llm/<name>` — the
        repository's own working-tree identity rule — so the two paths are one
        clone. A guard that tested the string as written would refuse one spelling
        of the same destination and accept the other.
        """
        link = pathlib.Path("/workspace/repos/epyc-inference-research")
        if not link.is_symlink():
            self.skipTest("the symlinked working tree is not present on this host")
        with self.assertRaises(DC.ExportDestinationRefused):
            DC.assert_exportable_destination(str(link / "data" / "kernel.json"))

    def test_a_double_slash_prefix_does_not_walk_past_the_guards(self):
        """A leading `//` is a POSIX special case that `normpath` PRESERVES, and it
        has already defeated `_is_within`-style prefix tests in two adapters here.
        Every guard in this function tests the RESOLVED path, so the trick has to
        die on all four; `..` traversal is checked in the same breath.
        """
        for candidate in ("//mnt/raid0/llm/llama.cpp/kernel_dashboard.json",
                          "//mnt/raid0/llm/tmp/kernel_dashboard.json",
                          "//mnt/raid0/llm/kernels/production/kernel_dashboard.json",
                          "/mnt/raid0/llm/autokernel/surface/../../llama.cpp/x.json",
                          "/mnt/raid0/llm/autokernel/../tmp/x.json"):
            with self.subTest(path=candidate):
                with self.assertRaises(DC.ExportDestinationRefused):
                    DC.assert_exportable_destination(candidate)

    def test_the_default_destination_is_outside_every_checkout(self):
        """The compliant-path control, and the assertion the module's own comment
        was making in prose: the durable path is accepted BECAUSE no ancestor of it
        is a working tree.
        """
        self.assertIsNone(DC._enclosing_checkout(DC.DEFAULT_EXPORT_PATH))
        self.assertEqual(DC.assert_exportable_destination(DC.DEFAULT_EXPORT_PATH),
                         DC.DEFAULT_EXPORT_PATH)


class TheWriteIsAllOrNothingTest(unittest.TestCase):
    """ATTACK C, second form. `os.write` may write fewer bytes than it is given and
    its return value was discarded, so a short write installed a TRUNCATED document
    atomically over the previous honest one — and returned a receipt claiming the
    full length and the full document's hash.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir="/mnt/raid0/llm")
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, "kernel_dashboard.json")
        self.real_write = os.write

    def _short_write(self, fd, data):
        return self.real_write(fd, data[: len(data) // 2])

    def test_a_short_write_is_refused_and_the_good_document_survives(self):
        good = document()
        DC.export_contract(good, path=self.target)
        DC.os.write = self._short_write
        self.addCleanup(setattr, DC.os, "write", self.real_write)
        try:
            with self.assertRaises(DC.ContractInvalid):
                DC.export_contract(document(exported_at="2026-08-04T09:00:00+00:00"),
                                   path=self.target)
        finally:
            DC.os.write = self.real_write
        reloaded = json.loads(pathlib.Path(self.target).read_text(encoding="utf-8"))
        self.assertEqual(reloaded, good)
        self.assertEqual(S.validate_kernel_dashboard_v2(reloaded), [])

    def test_a_failed_write_leaves_no_temporary_behind(self):
        DC.os.write = self._short_write
        self.addCleanup(setattr, DC.os, "write", self.real_write)
        try:
            with self.assertRaises(DC.ContractInvalid):
                DC.export_contract(document(), path=self.target)
        finally:
            DC.os.write = self.real_write
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_a_write_that_makes_no_progress_does_not_spin(self):
        DC.os.write = lambda fd, data: 0
        self.addCleanup(setattr, DC.os, "write", self.real_write)
        try:
            with self.assertRaises(DC.ContractInvalid) as caught:
                DC.export_contract(document(), path=self.target)
        finally:
            DC.os.write = self.real_write
        self.assertIn("stopped making progress", str(caught.exception))

    def test_a_partial_write_still_completes_when_the_kernel_takes_it_in_pieces(self):
        """THE compliant-path control. A short write is NORMAL; the defect was
        ignoring it, not the shortness. Written in three chunks, the export must
        succeed and the file must be byte-identical to the canonical form.
        """
        state = {"calls": 0}

        def chunked(fd, data):
            state["calls"] += 1
            return self.real_write(fd, data[: max(1, len(data) // 3)])

        DC.os.write = chunked
        self.addCleanup(setattr, DC.os, "write", self.real_write)
        try:
            doc = document()
            receipt = DC.export_contract(doc, path=self.target)
        finally:
            DC.os.write = self.real_write
        self.assertGreater(state["calls"], 1)
        self.assertEqual(pathlib.Path(self.target).read_bytes(), S.canonical_bytes(doc))
        self.assertEqual(receipt.bytes_written, os.path.getsize(self.target))

    def test_the_rename_is_flushed_to_the_directory_too(self):
        """Atomic-until-the-power-goes is not atomic. The second fsync — on the
        containing directory — is the half that gets left out.
        """
        import stat
        fsynced = []
        real_fsync = os.fsync

        def recording(fd):
            fsynced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
            return real_fsync(fd)

        DC.os.fsync = recording
        self.addCleanup(setattr, DC.os, "fsync", real_fsync)
        try:
            DC.export_contract(document(), path=self.target)
        finally:
            DC.os.fsync = real_fsync
        self.assertIn(True, fsynced, "the containing directory was never fsynced")
        self.assertIn(False, fsynced, "the file itself was never fsynced")


class NoVerdictIsRenderedAsNullTest(unittest.TestCase):
    """ATTACK E. `_check_dict` returned None for anything it did not recognise, and
    `null` is what an ABSENT check looks like — a consumer reads it as nothing to
    report. Every `Check` field on the owning dataclasses is typed and guarded at
    construction, so an unrecognised one is a caller handing over an object nobody
    verified.
    """

    def test_an_unrecognised_verdict_is_refused_rather_than_rendered_as_null(self):
        for value in (None, "pass", {"outcome": "pass"}, 1):
            with self.subTest(value=value):
                with self.assertRaises(DC.ContractInputError):
                    DC._check_dict(value)

    def test_a_real_verdict_still_renders_including_could_not_check(self):
        for outcome in (S.PASS, S.FAIL, S.COULD_NOT_CHECK):
            with self.subTest(outcome=outcome):
                self.assertEqual(DC._check_dict(S.Check(outcome, ("why",))),
                                 {"outcome": outcome, "reasons": ["why"]})


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
