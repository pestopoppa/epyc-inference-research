#!/usr/bin/env python3
"""test_release_integration.py — the AK5/AK6/AK8/AK9 seams, driven end to end.

WHY THIS FILE EXISTS
--------------------
`plan.py`, `readiness.py`, `t3.py`, `packager.py` and the three adapters were each
written and red-teamed on their own, and each one's suite proves it is internally
consistent. None of those suites can see a SEAM. A seam defect looks exactly like
correct behaviour from both sides: `plan.py` re-derives a drop verdict and refuses a
forged one, `t3.py` reads the same verdict off a normalised view and never
re-derives it — both modules are right about themselves and the pair drops a
backend's whole matrix on a boolean nobody computed. That class of defect is what
this file exists to catch, and every one of them was found by writing it:

  * a §3.2 drop verdict that `plan.py` refuses and `t3.py` accepts;
  * §1.6's conjunction adjudicated in `t3.py` over caller-supplied STANDINGS with
    nothing joining them to the measured matrix — delete every prefill cell, keep
    the prefill standing, and the release gate returned PASS;
  * `serving_runtime` refused by name at three of the four release-plane doors and
    silently admitted at the fourth;
  * a release receipt licensing claims on a FAILing run, which AK6 then rendered
    onto the operator's first page as "claims licensed: N";
  * a `CellResult` free to relabel the plan's cell — including `co_resident`, which
    is how the §10.2 phase 4 co-residency requirement gets satisfied by a run that
    was never co-resident;
  * a waiver attributed to `autokernel` verifying as human-attested, because the
    machine-actor vocabulary lives in AK6 and the waiver verifier is in AK5.

The chain driven here is the real one: an operator freeze request through the AK4
state machine, a sealed champion, a compiled release plan with one backend changed
and one confirmed unchanged and dropped under receipt, a T3 run through the wired
`ReleaseTierEvaluator` seam producing each of PASS / FAIL / PASS_WITH_WAIVER, the
sealed bundle, and an assembled package carrying its drafted era rows and AutoPilot
rebaseline note.

THE CARDINAL RULE, AS A TEST. `TestNoProductionWritePathsAnywhere` parses the AST —
from the filesystem, RECURSIVELY, so a module added tomorrow is covered by default —
and proves that nothing can write a production branch, move a stable kernel symlink,
write an era-registry row or apply an AutoPilot baseline. It is deliberately NOT a
grep: the words appear in almost every docstring, which is the point of parsing
calls instead.

The two rules have DIFFERENT SCOPES, and conflating them is what left the audit
covering 18 of 42 modules:

  * RULE 1 — "writes or spawns nothing at all" — is true of `release/` and
    `adapters/` and FALSE of the rest of the package by design. `journal.py`
    appends, `storage.py` expires artifacts, `resource/device_claim.py` takes lock
    files, `controller/state_machine.py` persists state. Widening rule 1 to them
    would be a guard forbidding the package's own legitimate idiom.
  * RULE 2 — "no call names a human-only target" — is not a property of a plane, and
    it binds HARDEST on exactly the modules rule 1 cannot reach, because those are
    the ones holding `rmtree`, `mkdir`, `replace` and `subprocess`. It runs over the
    whole package. The auditor always could find such a call; it was simply never
    handed the source.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO PROCESS, NO PRODUCTION WRITE. The only
bytes this suite writes go into `tempfile.TemporaryDirectory` instances it creates
and removes (the AK4 state machine journals to disk by design; that is the only
reason a temp dir appears at all).

Run:
    python3 -m unittest scripts.kernel_rnd.autokernel.release.test_release_integration
    python3 -W error::ResourceWarning -m unittest \
        scripts.kernel_rnd.autokernel.release.test_release_integration
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
import unittest

from .. import journal as journal_mod
from .. import schemas
from ..adapters import qwentts_tts, serving_runtime, whisper_stt
from ..controller import state_machine as sm
from ..evaluator import api as evaluator_api
from . import packager, plan, readiness, t3
# The unit suites' fixtures are REUSED, never re-declared. An integration test that
# builds its own fixtures proves that two hand-written shapes agree, which is the one
# thing nobody needed proved; reusing them means a drift in either module's fixture
# drifts this chain too, and the seam assertion is what catches it.
from . import test_packager as FPK
from . import test_plan as FPL
from . import test_t3 as FT3

#: The preserved WAIVE-Q8 attestation, READ from `test_t3.py` rather than re-typed:
#: an operator attestation retyped is an operator attestation edited.
V8_WAIVER = FT3.V8_WAIVER

NOW = FPK.NOW
LATER = "2026-08-04T18:00:00Z"   # after the fixture escalation deadline
CAMPAIGN = "ak-v9"
V8_HEAD = FPK.V8_HEAD
CANDIDATE_COMMIT = FPL.CAND_COMMIT


# =============================================================================
# The bridge: a compiled `plan.ReleasePlan` driven into a `t3.T3Request`
#
# This is the seam under test, so it is written the way a caller would have to
# write it — through the three published adapters and nothing else.
# =============================================================================

def _with_reps(cell: t3.Cell, reps: int) -> t3.Cell:
    """The plan-derived cell plus its measured rep count.

    `plan.ReleaseCell` has no `reps` field and could not: the compiler runs before
    anything is measured. Every other facet is carried through unchanged, because
    `T3Request` now refuses a result whose cell contradicts the planned cell of the
    same id — see `TestSeamRegressions.test_a_result_may_not_relabel_the_planned_cell`.
    """
    fields = {f.name: getattr(cell, f.name) for f in cell.__dataclass_fields__.values()}
    fields["reps"] = reps
    return t3.Cell(**fields)


EXTRA_PHASES = (t3.PHASE_BACKEND_CORRECTNESS, t3.PHASE_QUALITY,
                t3.PHASE_STABILITY, t3.PHASE_CAPACITY_UTILITY)


def extra_cells(backend: str = "llama_cpu") -> tuple:
    """The evidence classes the compiler declares requirements for but does not enumerate."""
    return tuple(
        t3.Cell(cell_id=f"{backend}.{phase_id}", backend=backend, release_phase=phase_id,
                protocol_id=t3.RELEASE_PROTOCOL_ID,
                recipe_class=t3.RECIPE_PRODUCTION_OPTIMAL, metric="pass_fail",
                metric_direction="higher_better",
                claim=f"{backend} {phase_id} parity", reps=1)
        for phase_id in EXTRA_PHASES)


class Chain:
    """One compiled plan carried through every published plan->T3 adapter."""

    def __init__(self):
        self.compiled = FPL.compile_ok()
        self.view = t3.release_plan_view_from_compiled(
            self.compiled,
            incumbent_branch="production-consolidated-v8",
            incumbent_commit=V8_HEAD,
            incumbent_version_number=8,
            extra_cells=extra_cells())
        self.unchanged = t3.unchanged_results_from_plan(self.compiled)
        self.receipts = t3.transfer_receipts_from_plan(
            self.compiled, incumbent_commit=V8_HEAD)

    # -- derived views -------------------------------------------------------

    def performance_cells(self, phase: str) -> tuple:
        return tuple(c for c in self.view.cells
                     if c.release_phase == t3.PHASE_PERFORMANCE_MATRIX
                     and c.workload_phase == phase)

    def results(self, failing_cell_id: str = "") -> tuple:
        out = []
        for cell in self.view.cells:
            reps = 10 if cell.release_phase == t3.PHASE_PERFORMANCE_MATRIX else 1
            check = (t3._fail(f"{cell.cell_id}: regressed against the production anchor")
                     if cell.cell_id == failing_cell_id else schemas.Check(schemas.PASS))
            out.append(t3.CellResult(
                cell=_with_reps(cell, reps), check=check,
                raw_samples_ref=f"data/ak-v9/{cell.cell_id}.jsonl",
                reducer_id="median_mad/v1"))
        return tuple(out)

    def standings(self) -> tuple:
        protocol_by_phase = {"decode": "P-BENCH-1", "prefill": "P-BENCH-PREFILL-1"}
        standing_by_phase = {"decode": t3.STANDING_IMPROVED,
                             "prefill": t3.STANDING_NON_INFERIOR}
        return tuple(
            t3.PhaseStanding(
                backend="llama_cpu", workload_phase=phase,
                protocol_id=protocol_by_phase[phase],
                standing=standing_by_phase[phase],
                cell_ids=tuple(c.cell_id for c in self.performance_cells(phase)),
                evidence_ref=f"journal://{CAMPAIGN}/standing/llama_cpu.{phase}")
            for phase in ("decode", "prefill"))

    def sealed(self, **overrides) -> t3.SealedCandidate:
        fields = {"candidate_commit": CANDIDATE_COMMIT,
                  "production_base_commit": V8_HEAD,
                  "candidate_branch": self.compiled.target.candidate_branch,
                  "build_dirs": {b: FPL.EXPERIMENTAL_BUILD for b in self.view.backends}}
        fields.update(overrides)
        return FPK.sealed_candidate(**fields)

    def seal(self, **overrides) -> packager.SealedRelease:
        """The AK6 seal over this chain's candidate, through the real door."""
        fields = {
            "champion_id": "akch-llama-v9",
            "candidate": self.sealed(),
            "build_receipt_sha256": FPK.digest("build-receipt"),
            "seal_inputs_ref": "data/ak-v9/seal-inputs.json",
            "sealed_at": NOW,
            "pinned_evaluator_bundle_sha256": FPK.EVALUATOR_BUNDLE,
            "incumbent_branch": self.view.incumbent_branch,
            "incumbent_commit": self.view.incumbent_commit,
        }
        fields.update(overrides)
        return packager.seal_champion(**fields)

    def request(self, **overrides) -> t3.T3Request:
        failing = overrides.pop("_failing_cell_id", "")
        fields = {
            "plan": self.view,
            "sealed": self.sealed(),
            "_results": overrides.pop("_results", None) or self.results(failing),
            "standings": self.standings(),
            "backend_unchanged": self.unchanged,
            "transfer_receipts": self.receipts,
            "now": NOW,
        }
        fields.update(overrides)
        return FPK.t3_request(**fields)

    def run(self, **overrides) -> t3.T3Result:
        return t3.T3Runner().evaluate_release(self.request(**overrides))


def waiver_document(cell_id: str, **overrides) -> dict:
    """An operator waiver whose OWN scope names the cell, per §10.4."""
    document = {
        "schema": schemas.SCHEMA_OPERATOR_WAIVER,
        "waiver_id": "WAIVE-CPU-PREFILL-V9",
        "campaign_id": CAMPAIGN,
        "decision": "WAIVE",
        "protocol": "P-BENCH-PREFILL-1",
        "protocol_changed": False,
        "candidate_head": CANDIDATE_COMMIT,
        "production_head": V8_HEAD,
        "scope": {"excluded_cell_ids": [cell_id], "excluded_models": ["qwen36_q8"],
                  "excluded_pairs": ["qwen36_q8-pp2048-iqk1"],
                  "remaining_matched_pairs": 7},
        "reason": ("the Q8 prefill pair cannot satisfy the ratified core-equivalent "
                   "eligibility floor on this lineup"),
        "consequences": ["No v9 Q8 prefill non-regression claim may be made."],
        "authorized_by": "daniele",
        "expiry": {"expires_at": None, "reopen_predicate": "the floor is re-derived"},
        "created_at": "2026-08-02T00:00:00Z",
        "narrative": None,
    }
    document.update(overrides)
    return document


def waiver_binding(cell_id: str, document=None) -> t3.WaiverBinding:
    document = document if document is not None else waiver_document(cell_id)
    pinned = schemas.content_hash(json.loads(json.dumps(document)))
    return t3.WaiverBinding(
        waiver_id=document["waiver_id"], pinned_sha256=pinned, observed_sha256=pinned,
        document=document, document_path="artifacts/operator/waive-cpu-prefill-v9.json",
        covers_cell_ids=(cell_id,))


# =============================================================================
# 1 — the plan -> T3 seam
# =============================================================================

class TestPlanToT3Seam(unittest.TestCase):
    """§3.2 / §10.2 phase 1: one backend changed, one confirmed unchanged and dropped."""

    def setUp(self):
        self.chain = Chain()

    def test_the_compiled_plan_carries_both_backends_into_the_graded_view(self):
        # The dropped backend stays in the view. A backend that vanished between the
        # compiler and the gate is a backend nobody had to justify dropping.
        self.assertEqual(self.chain.view.backends, ("llama_cpu", "llama_gpu"))
        self.assertEqual(self.chain.view.source_tree, "llama.cpp")
        self.assertTrue(self.chain.view.cells)

    def test_every_compiled_cell_reaches_the_gate_with_its_owning_protocol(self):
        for cell in self.chain.performance_cells("decode"):
            self.assertEqual(cell.protocol_id, "P-BENCH-1")
        for cell in self.chain.performance_cells("prefill"):
            self.assertEqual(cell.protocol_id, "P-BENCH-PREFILL-1")

    def test_the_unchanged_backend_drops_only_with_a_transfer_receipt(self):
        self.assertTrue(self.chain.unchanged["llama_gpu"].may_drop_cells)
        self.assertFalse(self.chain.unchanged["llama_cpu"].may_drop_cells)
        receipt = self.chain.receipts["llama_gpu"]
        self.assertTrue(receipt.incumbent_artifacts)
        self.assertEqual(receipt.incumbent_commit, V8_HEAD)
        self.assertNotIn("llama_cpu", self.chain.receipts)

    def test_the_gate_drops_the_unchanged_backend_and_charges_the_other(self):
        result = self.chain.run()
        self.assertEqual(result.products.dropped_backends, ("llama_gpu",))
        self.assertEqual(result.products.evidence_owed_backends, ("llama_cpu",))

    def test_dropping_without_the_receipt_leaves_the_backend_owing_evidence(self):
        result = t3.run_t3(self.chain.request(transfer_receipts={}))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("llama_gpu", result.products.evidence_owed_backends)
        self.assertIn("unaudited hole in the matrix",
                      " | ".join(result.verdict_computation.blocking_reasons))


# =============================================================================
# 2 — operator freeze request -> seal -> T3 -> bundle -> package
# =============================================================================

class TestFreezeRequestToPackage(unittest.TestCase):
    """The whole AK5+AK6 chain, PASS, with no production write anywhere in it."""

    def setUp(self):
        self.chain = Chain()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = journal_mod.Journal(os.path.join(self.tmp.name, "journal"),
                                           campaign_id=CAMPAIGN)
        self.journal.initialize()

    def machine(self, **kwargs) -> sm.ControllerStateMachine:
        kwargs.setdefault("journal_", self.journal)
        kwargs.setdefault("root", os.path.join(self.tmp.name, "controller"))
        kwargs.setdefault("campaign_id", CAMPAIGN)
        machine = sm.ControllerStateMachine(**kwargs)
        for step in (sm.DISCOVER, sm.SELECT_TARGET, sm.PROPOSE, sm.PRE_RUN_CRITIC,
                     sm.MUTATE, sm.BUILD, sm.T0_GATE, sm.T1_SEARCH_EVAL,
                     sm.POST_RUN_CRITIC, sm.BANK_EVENT, sm.UPDATE_SEARCH_STATE,
                     sm.CHAMPION_GUARD):
            machine.transition(step, trigger="integration", reason="walk to the guard")
        return machine

    # -- AK7: the request is the operator's, and only the operator's ----------

    def test_the_loop_cannot_reach_the_release_branch_without_an_operator(self):
        machine = self.machine()
        with self.assertRaises(TypeError):
            machine.request_freeze(reason="the champion looks ready")  # no requested_by

    def test_the_release_branch_opens_on_an_operator_freeze_request(self):
        machine = self.machine()
        transition = machine.request_freeze(
            requested_by="daniele", reason="four banked candidates since v8")
        self.assertEqual(machine.state, sm.SEAL)
        self.assertEqual(transition.trigger, "operator_freeze_request")
        self.assertEqual(transition.detail["requested_by"], "daniele")

    def test_an_unwired_release_gate_refuses_the_tier_rather_than_guessing(self):
        machine = self.machine()
        machine.request_freeze(requested_by="daniele", reason="ready")
        with self.assertRaises(evaluator_api.TierNotOwned) as ctx:
            machine.run_release_gate(self.chain.request())
        self.assertIn(evaluator_api.RELEASE_TIER_OWNER, str(ctx.exception))

    def test_the_wired_t3_runner_satisfies_the_ak4_release_gate_seam(self):
        machine = self.machine(release_gate=t3.T3Runner())
        machine.request_freeze(requested_by="daniele", reason="ready")
        transition, result = machine.run_release_gate(self.chain.request())
        self.assertEqual(machine.state, sm.T3_RELEASE_GATE)
        self.assertEqual(transition.detail["tier"], "T3")
        self.assertEqual(transition.detail["owner"], evaluator_api.RELEASE_TIER_OWNER)
        self.assertIsInstance(result, t3.T3Result)
        self.assertEqual(result.verdict, "PASS")

    # -- AK6: seal, evaluate, package ----------------------------------------

    def evaluation(self, **overrides) -> packager.TrustedEvaluation:
        return packager.run_release_evaluation(
            self.chain.request(**overrides), evaluator=t3.T3Runner())

    def package(self, **overrides) -> packager.ReleasePackage:
        evaluation = overrides.pop("evaluation", None) or self.evaluation()
        fields = {
            "evaluation": evaluation,
            "sealed": self.chain.seal(),
            "release_plan": self.chain.view.to_dict(),
        }
        fields.update(overrides)
        return FPK.release_package(**fields)

    def test_the_champion_seals_and_the_evaluator_seam_verifies_this_request(self):
        evaluation = self.evaluation()
        self.assertEqual(evaluation.check.outcome, schemas.PASS, evaluation.check.reasons)
        self.assertEqual(evaluation.evaluator_tier, t3.TIER)
        self.assertEqual(evaluation.evaluator_class, "T3Runner")
        self.assertEqual(evaluation.verdict, "PASS")
        self.assertIsNotNone(evaluation.bundle_sha256)

    def test_the_bundle_rehashes_to_its_own_payload_and_names_this_candidate(self):
        bundle = self.evaluation().result.bundle
        self.assertEqual(schemas.content_hash(bundle.payload), bundle.bundle_sha256)
        self.assertEqual(bundle.payload["sealed_candidate"]["candidate_id"], "akc-v9")
        self.assertEqual(bundle.payload["campaign_id"], CAMPAIGN)
        self.assertEqual(bundle.payload["tier"], "T3")
        self.assertEqual(bundle.payload["schema"], t3.BUNDLE_SCHEMA)
        self.assertEqual(bundle.payload["release_plan"], self.chain.view.to_dict())
        self.assertEqual(schemas.find_authority_flavoured_keys(bundle.payload), [])

    def test_the_package_is_ready_and_validates_against_the_schema(self):
        package = self.package()
        self.assertEqual(package.state, packager.STATE_READY,
                         [f.to_dict() for f in package.blocking_findings])
        self.assertEqual(package.schema_violations(), [])
        self.assertEqual(schemas.find_authority_flavoured_keys(package.to_dict()), [])

    def test_the_package_carries_its_drafted_era_rows_and_rebaseline_note(self):
        record = self.package().to_dict()
        draft = record["draft_era_registry_row"]
        rows = draft["rows"]
        # All three era kinds: a kernel era, the AutoPilot speed era it moves, and
        # the umbrella era every later number is labelled with.
        self.assertEqual({row["kind"] for row in rows}, set(packager.ERA_ROW_KINDS))
        for row in rows:
            self.assertTrue(row["draft"], row)
            self.assertEqual(row["written_by"], packager.OPERATOR_AUTHORITY, row)
            self.assertEqual(row["effective_from"], packager.ERA_EFFECTIVE_FROM, row)
        self.assertEqual(draft["registry_path"], FPK.ERA_REGISTRY)
        self.assertTrue(draft["human_only_path"])
        self.assertIn(FPK.AUTOPILOT_BASELINE, record["draft_autopilot_rebaseline_note"])

    def test_the_cutover_is_a_routed_ask_and_not_a_scheduled_action(self):
        message = FPK.cutover_request().to_bus_message()
        self.assertEqual(message["schema_version"], "session_bus.msg.v1")
        # Structural routing intent, never payload prose.
        self.assertEqual(message["needs_routing_to"], ["inference"])
        self.assertTrue(message["action_required"])
        self.assertEqual(message["kind"], "request")
        self.assertIn(packager.CUTOVER_ASK, str(message["payload"]))
        # No time field: AutoKernel does not schedule an inference owner's reload,
        # it asks for one at that owner's own boundary.
        for forbidden in ("scheduled_at", "execute_at", "deadline", "reload_at",
                          "when", "at"):
            self.assertNotIn(forbidden, message)
            self.assertNotIn(forbidden, message["payload"])
        # And the transport is refused outright: the ask is routed by a human.
        with self.assertRaises(packager.PackagerError):
            packager.send_cutover_request(message)

    def test_the_first_page_says_a_human_executes_this(self):
        page = packager.render_first_page(self.package())
        self.assertIn(packager.EXECUTED_BY, page)
        self.assertIn(packager.PACKAGE_NOTICE, page)


# =============================================================================
# 3 — one chain, three verdicts
# =============================================================================

class TestVerdictSpectrum(unittest.TestCase):
    """PASS / FAIL / PASS_WITH_WAIVER off ONE chain, differing only in the evidence."""

    def setUp(self):
        self.chain = Chain()
        self.failing = self.chain.performance_cells("prefill")[0].cell_id

    def test_pass(self):
        result = self.chain.run()
        self.assertEqual(result.verdict, "PASS")
        self.assertEqual(result.verdict_computation.failed_cells, ())
        self.assertTrue(result.receipt.claims)
        self.assertEqual(result.receipt.withheld_claims, ())

    def test_fail_names_the_cell_and_licenses_nothing(self):
        result = self.chain.run(_failing_cell_id=self.failing)
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.verdict_computation.failed_cells, (self.failing,))
        # The cells that DID pass are kept, but a failing release licenses nothing.
        self.assertEqual(result.receipt.claims, ())
        self.assertTrue(result.receipt.withheld_claims)

    def test_pass_with_waiver_suppresses_exactly_the_waived_claim(self):
        binding = waiver_binding(self.failing)
        result = self.chain.run(_failing_cell_id=self.failing, waivers=(binding,))
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER",
                         " | ".join(result.verdict_computation.blocking_reasons))
        suppressed = {s["claim"] for s in result.receipt.suppressed_claims}
        self.assertEqual(len(suppressed), 1)
        self.assertTrue(suppressed.isdisjoint(set(result.receipt.claims)))
        self.assertIn("No v9 Q8 prefill non-regression claim may be made.",
                      result.receipt.forfeited_claims)

    def test_a_waiver_covering_a_cell_outside_its_own_scope_suppresses_nothing(self):
        other = self.chain.performance_cells("decode")[0].cell_id
        # The document's scope still names the prefill cell; the BINDING claims the
        # decode one. The scope that counts is the operator's.
        binding = waiver_binding(other, document=waiver_document(self.failing))
        result = self.chain.run(_failing_cell_id=self.failing, waivers=(binding,))
        self.assertEqual(result.verdict, "FAIL")

    def test_a_failing_run_produces_a_blocked_package_not_a_ready_one(self):
        evaluation = packager.run_release_evaluation(
            self.chain.request(_failing_cell_id=self.failing), evaluator=t3.T3Runner())
        package = FPK.release_package(
            evaluation=evaluation,
            sealed=self.chain.seal(),
            release_plan=self.chain.view.to_dict())
        self.assertEqual(package.state, packager.STATE_BLOCKED)
        self.assertIn("T3_VERDICT_FAIL",
                      {f.code for f in package.blocking_findings})


# =============================================================================
# 4 — the negative paths
# =============================================================================

class TestPackagerCannotExecute(unittest.TestCase):
    """§11.2: every "may not" is a door that raises, not a flag that is checked."""

    def test_every_refused_capability_raises(self):
        self.assertTrue(packager.REFUSED_CAPABILITIES)
        for capability, function_name in packager.REFUSED_CAPABILITIES.items():
            door = getattr(packager, function_name)
            with self.assertRaises(packager.PackagerError, msg=capability):
                door("anything", keyword="anything")

    def test_the_packager_cannot_execute_a_command_it_drafted(self):
        commands = FPK.operator_commands()
        self.assertTrue(commands)
        for command in commands:
            # The drafted command is text plus a rollback plus an expected effect. It
            # carries no executor, and the module's only execution door raises.
            self.assertFalse(hasattr(command, "execute"))
            self.assertFalse(hasattr(command, "run"))
            with self.assertRaises(packager.PackagerError):
                packager.execute_operator_command(command)

    def test_the_human_only_steps_are_derived_from_the_text_not_declared(self):
        review = packager.validate_command_sequence(
            FPK.operator_commands(), transaction=FPK.transaction_plan(),
            rollback=FPK.rollback_plan(), era_row=FPK.era_draft(),
            autopilot_baseline_path=FPK.AUTOPILOT_BASELINE)
        human_only = [c for c in review.validated_commands if c.human_only]
        self.assertTrue(human_only)
        for command in human_only:
            # `human_only` is a derived property over the command TEXT, so it cannot
            # be turned off by a flag beside the command.
            self.assertTrue(command.human_only_reasons, command.command)
            self.assertEqual(command.human_only,
                             bool(packager._human_only_reasons(command.scanned_text)))
        self.assertEqual(review.findings, (), review.findings)

    def test_the_transaction_plan_cannot_be_constructed_as_executed(self):
        with self.assertRaises(t3.T3Error):
            FPK.transaction_plan(executed=True)


class TestSelfGrantedWaiverIsRefused(unittest.TestCase):
    """§10.4: a waiver is human-authored by definition."""

    def setUp(self):
        self.chain = Chain()
        self.failing = self.chain.performance_cells("prefill")[0].cell_id

    def _evaluation(self, document):
        binding = waiver_binding(self.failing, document=document)
        return binding, packager.run_release_evaluation(
            self.chain.request(_failing_cell_id=self.failing, waivers=(binding,)),
            evaluator=t3.T3Runner())

    def test_the_packager_has_no_door_that_grants_one(self):
        with self.assertRaises(packager.PackagerError):
            packager.waive_failed_evidence(cell_id=self.failing, reason="it is fine")

    def test_an_unattributed_waiver_suppresses_nothing(self):
        document = waiver_document(self.failing)
        document.pop("authorized_by")
        _, evaluation = self._evaluation(document)
        self.assertEqual(evaluation.verdict, "FAIL")

    def test_a_waiver_attributed_to_the_loop_cannot_reach_a_ready_package(self):
        document = waiver_document(self.failing, authorized_by="autokernel")
        binding, evaluation = self._evaluation(document)
        package = FPK.release_package(
            evaluation=evaluation, waivers=(binding,),
            sealed=self.chain.seal(),
            release_plan=self.chain.view.to_dict())
        codes = {f.code for f in package.blocking_findings}
        self.assertIn("WAIVER_SELF_GRANTED", codes)
        self.assertEqual(package.state, packager.STATE_BLOCKED)

    def test_a_human_attributed_waiver_still_reaches_a_ready_package(self):
        # The guard must not forbid its own compliant idiom.
        binding, evaluation = self._evaluation(waiver_document(self.failing))
        self.assertEqual(evaluation.verdict, "PASS_WITH_WAIVER")
        package = FPK.release_package(
            evaluation=evaluation, waivers=(binding,),
            sealed=self.chain.seal(),
            release_plan=self.chain.view.to_dict())
        self.assertNotIn("WAIVER_SELF_GRANTED", {f.code for f in package.findings})
        self.assertEqual(package.state, packager.STATE_READY,
                         [f.to_dict() for f in package.blocking_findings])


class TestRerunOnAnUnchangedFingerprint(unittest.TestCase):
    """§9.1/§12: a failed gate does not get re-rolled until something changes."""

    def setUp(self):
        self.chain = Chain()
        self.failing = self.chain.performance_cells("prefill")[0].cell_id
        self.request = self.chain.request(_failing_cell_id=self.failing)
        self.first = t3.run_t3(self.request)
        self.assertEqual(self.first.verdict, "FAIL")
        self.attempt = t3.T3Attempt(
            fingerprint=self.first.fingerprint, verdict="FAIL", completed_at=NOW,
            bundle_sha256=(self.first.bundle.bundle_sha256 if self.first.bundle
                           else FPK.digest("no-bundle")),
            failed_phases=(t3.PHASE_PERFORMANCE_MATRIX,))

    def test_the_same_fingerprint_is_refused_rather_than_returned_as_a_verdict(self):
        with self.assertRaises(t3.RerunRefused) as ctx:
            t3.run_t3(self.chain.request(_failing_cell_id=self.failing,
                                         attempt_ledger=(self.attempt,), now=LATER))
        self.assertIn(t3.RERUN_REFUSED_UNCHANGED_FINGERPRINT, str(ctx.exception))

    def test_a_refusal_is_not_a_fail_verdict(self):
        # Recording "we declined to re-measure" as FAIL would put it in the same
        # column as "the kernel regressed".
        try:
            t3.run_t3(self.chain.request(_failing_cell_id=self.failing,
                                         attempt_ledger=(self.attempt,), now=LATER))
        except t3.RerunRefused as exc:
            self.assertNotIn("regressed", str(exc))
        else:  # pragma: no cover - the assertion above owns the failure
            self.fail("the rerun was admitted")

    def test_a_changed_fingerprint_is_admitted(self):
        # New evidence-affecting input -> new fingerprint -> a rerun is a new question.
        changed = self.chain.request(
            _failing_cell_id=self.failing, attempt_ledger=(self.attempt,), now=LATER,
            sealed=self.chain.sealed(seal_sha256=FPK.digest("seal-take-2")),
            host_escalation_deadline="2026-08-05T12:00:00Z")
        self.assertNotEqual(changed.fingerprint(), self.first.fingerprint)
        result = t3.run_t3(changed)
        self.assertTrue(result.rerun.admissible)
        self.assertEqual(result.rerun.code, t3.RERUN_ADMITTED_NEW_FINGERPRINT)

    def test_a_sealed_pass_is_not_re_run_either(self):
        passing = self.chain.request()
        sealed_attempt = t3.T3Attempt(
            fingerprint=passing.fingerprint(), verdict="PASS", completed_at=NOW,
            bundle_sha256=FPK.digest("sealed-bundle"))
        with self.assertRaises(t3.RerunRefused) as ctx:
            t3.run_t3(self.chain.request(attempt_ledger=(sealed_attempt,), now=LATER))
        self.assertIn(t3.RERUN_REFUSED_ALREADY_SEALED, str(ctx.exception))


class TestServingRuntimeRefusesTheKernelFreezePath(unittest.TestCase):
    """§11.6 / §13.5: a serving change may not impersonate a kernel era.

    Four doors, and now all four refuse. `readiness.py` was the one that did not:
    `serving_runtime` is in `schemas.BACKENDS` but absent from both
    `SOURCE_TREE_BY_BACKEND` and `PHASES_BY_BACKEND`, so the champion-lineage check
    and the §1.6 conjunction check each degraded to a no-op on it while the signal
    still rendered as a kernel backend's.
    """

    def test_the_adapter_answers_for_its_own_backend_only(self):
        self.assertEqual(serving_runtime.release_path_for("serving_runtime"),
                         serving_runtime.RELEASE_PATH)
        for other in ("llama_cpu", "llama_gpu", "whisper_stt", "qwentts_tts"):
            with self.assertRaises(serving_runtime.ServingAdapterError):
                serving_runtime.release_path_for(other)

    def test_the_adapter_refuses_the_freeze_path_rather_than_degrading_to_it(self):
        with self.assertRaises(serving_runtime.KernelFreezePathRefused) as ctx:
            serving_runtime.refuse_kernel_freeze("seal a kernel release candidate")
        self.assertIn(serving_runtime.RELEASE_PATH, str(ctx.exception))

    def test_the_kernel_adapters_refuse_the_stack_change_path_symmetrically(self):
        for adapter in (whisper_stt, qwentts_tts):
            with self.assertRaises(Exception):
                adapter.refuse_stack_change_path()

    def test_the_plan_compiler_refuses_it(self):
        with self.assertRaises(plan.KernelFreezePathRefused):
            FPL.target(backends=("llama_cpu", "serving_runtime"))

    def test_the_release_gate_refuses_it(self):
        with self.assertRaises(t3.StackChangePathRequired):
            t3.ReleasePlanView(
                plan_id="p", plan_sha256=FPK.digest("p"), source_tree="llama.cpp",
                backends=("llama_cpu", "serving_runtime"), cells=(),
                incumbent_branch="production-consolidated-v8",
                incumbent_commit=V8_HEAD, incumbent_version_number=8)

    def test_the_packager_refuses_it(self):
        with self.assertRaises(packager.PackagerInputError):
            FPK.freeze_request(source_tree="serving_runtime")

    def test_the_readiness_signal_refuses_it(self):
        with self.assertRaises(readiness.CellInadmissible) as ctx:
            readiness.ObjectiveSpec(
                backend="serving_runtime", phases=("throughput",),
                protocol_by_phase={"throughput": "P-SERVING-1"},
                improvement_quantifier=readiness.QUANTIFIER_BACKEND_WIDE)
        self.assertIn("§11.6", str(ctx.exception))

    def test_a_stack_change_package_declares_no_kernel_freeze_action(self):
        check = serving_runtime.scan_for_kernel_freeze_actions(
            {"actions": ["freeze", "move_stable_kernel_symlink"]})
        self.assertEqual(check.outcome, schemas.FAIL)


# =============================================================================
# 6b — the 2026-08-03 hardening pass, driven through the CHAIN
#
# Each guarantee below was closed inside one module and red-teamed there. What no
# module's own suite can show is that the guarantee survives composition: the
# waiver vocabulary is enforced in `schemas`, consumed by `t3.verify_waiver` AND by
# `packager`, and a verdict is what a caller actually sees. These drive the real
# chain and assert on the VERDICT, not on the helper.
# =============================================================================

class TestWaiverAuthorityHoldsThroughTheChain(unittest.TestCase):
    """One vocabulary, two planes. Before the pass a waiver attributed to
    `autokernel` verified as human-attested in AK5 and was refused only in AK6, so
    T3's own verdict read PASS_WITH_WAIVER and only the packager said no.
    """

    def setUp(self):
        self.chain = Chain()
        self.cell_id = self.chain.performance_cells("prefill")[0].cell_id

    def _verdict(self, *, author=None, path=None) -> t3.T3Result:
        overrides = {} if author is None else {"authorized_by": author}
        document = waiver_document(self.cell_id, **overrides)
        binding = waiver_binding(self.cell_id, document)
        if path is not None:
            binding = t3.WaiverBinding(
                waiver_id=binding.waiver_id, pinned_sha256=binding.pinned_sha256,
                observed_sha256=binding.observed_sha256, document=binding.document,
                document_path=path, covers_cell_ids=binding.covers_cell_ids)
        return self.chain.run(_failing_cell_id=self.cell_id, waivers=(binding,))

    def test_a_human_attributed_waiver_still_suppresses_its_cell(self):
        """COMPLIANT-PATH CONTROL. The guard must not forbid the thing §10.4
        exists to permit: a real operator waiving a real cell.
        """
        result = self._verdict()
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER",
                         " | ".join(result.verdict_computation.blocking_reasons))

    def test_no_spelling_of_the_loop_can_waive_its_own_cell(self):
        """The separator walk-around, end to end. `[a-z0-9]+` tokenisation refused
        `autokernel` and admitted `auto-kernel`, and a waiver is the loop's only
        route to licensing a claim it did not earn.
        """
        for author in ("autokernel", "auto-kernel", "auto_kernel", "auto.kernel",
                       "AutoKernel", "autokernel-controller", "ak-controller"):
            with self.subTest(author=author):
                result = self._verdict(author=author)
                self.assertNotEqual(result.verdict, "PASS_WITH_WAIVER")

    def test_a_human_name_that_merely_contains_a_token_is_still_a_human(self):
        """COMPLIANT-PATH CONTROL for the vocabulary. Substring matching would
        refuse these, and refusing a real operator is the same defect pointed the
        other way.
        """
        for author in ("Daniele Pinna", "d.pinna", "ops-daniele", "Jean-Luc Picard"):
            with self.subTest(author=author):
                self.assertEqual(self._verdict(author=author).verdict,
                                 "PASS_WITH_WAIVER")

    def test_the_operator_attestation_root_cannot_be_manufactured(self):
        """`repo_relative_forms` stripped ANY leading segment, so the loop's own
        scratch root spelled the operator's attestation root. `/mnt/raid0/llm/tmp/`
        is where `resource/device_claim.py` puts its lock files — a path the loop
        can write.
        """
        result = self._verdict(path="/mnt/raid0/llm/tmp/artifacts/operator/w.json")
        self.assertNotEqual(result.verdict, "PASS_WITH_WAIVER")

    def test_a_real_operator_checkout_path_still_resolves(self):
        """COMPLIANT-PATH CONTROL for containment: the reduction exists so that a
        genuine absolute citation under a real checkout root still matches.
        """
        for path in ("artifacts/operator/waive-cpu-prefill-v9.json",
                     "/workspace/artifacts/operator/waive-cpu-prefill-v9.json"):
            with self.subTest(path=path):
                self.assertEqual(self._verdict(path=path).verdict,
                                 "PASS_WITH_WAIVER")


class TestOnePerPhaseStandingThroughTheChain(unittest.TestCase):
    """§1.6 is adjudicated over caller-supplied standings, and the caller is the
    party being gated. Every dict in the gate resolved a duplicate last-wins, so a
    regressed standing plus a later `non_inferior` one over the same cell PASSed.
    """

    def setUp(self):
        self.chain = Chain()

    def test_one_standing_per_phase_still_passes(self):
        """COMPLIANT-PATH CONTROL: the ordinary shape is unaffected."""
        self.assertEqual(self.chain.run().verdict, "PASS")

    def test_a_duplicated_phase_standing_is_a_blocking_contradiction(self):
        standings = self.chain.standings()
        result = self.chain.run(standings=standings + (standings[0],))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("more than one standing for the phase",
                      " | ".join(result.verdict_computation.blocking_reasons))

    def test_a_regression_cannot_be_overwritten_by_a_later_standing(self):
        """The exploit direction, exactly: the regressed record must survive."""
        standings = self.chain.standings()
        regressed = dataclasses.replace(
            standings[0], standing=t3.STANDING_REGRESSED,
            evidence_ref=standings[0].evidence_ref + "-earlier")
        result = self.chain.run(standings=(regressed,) + standings)
        self.assertEqual(result.verdict, "FAIL")
        reasons = " | ".join(result.verdict_computation.blocking_reasons)
        self.assertIn("more than one standing for the phase", reasons)
        self.assertIn("regressed", reasons)


class TestStableKernelPathTraversalThroughTheChain(unittest.TestCase):
    """The stable kernel symlink is the trust boundary, and `..` is a plain path
    character. Depth alone said "well below the link" about a command naming
    `/usr/bin/install` with the production binary as its operand.
    """

    LINK = "/mnt/raid0/llm/kernels/production/cpu"

    def test_a_traversal_out_of_the_stable_path_is_a_positive_finding(self):
        """`assertEqual(FAIL)`, deliberately, NOT `assertNotEqual(PASS)`: a
        "not PASS" assertion is satisfied by COULD_NOT_CHECK, which is how the
        doubled-slash defect below hid in the first place. These two commands are
        refused by the traversal rule AND by the below-link operand rule, so this is
        a defence-in-depth assertion rather than an isolating one.
        """
        for command in (
                f"{self.LINK}/../../../../usr/bin/install -m755 evil {self.LINK}/bin",
                f"{self.LINK}/../../../../bin/rm -rf {self.LINK}/bin",
                f"{self.LINK}/../../../../usr/bin/install -m755 evil /var/tmp/x",
        ):
            with self.subTest(command=command):
                self.assertEqual(
                    serving_runtime.classify_stable_kernel_path_use(command).outcome,
                    schemas.FAIL)

    def test_a_repeated_slash_does_not_hide_the_stable_path(self):
        """`a//b` is `a/b` to every kernel on this host. Without slash collapsing
        this returned COULD_NOT_CHECK — no finding at all — and packaged.
        """
        doubled = "rm -rf /mnt/raid0/llm/kernels//production/cpu/bin/llama-server"
        self.assertEqual(
            serving_runtime.classify_stable_kernel_path_use(doubled).outcome,
            schemas.FAIL)

    def test_an_ordinary_launcher_command_is_still_admitted(self):
        """COMPLIANT-PATH CONTROL. A guard that refused every `..`, or every
        doubled slash, would forbid the adapter's own launcher idiom.
        """
        for command in (
                f"{self.LINK}/bin/llama-server -m ../models/qwen.gguf",
                "taskset -c 0-95 /mnt/raid0/llm/llama.cpp/build/bin/llama-server",
                "curl https://localhost:8000/health",
        ):
            with self.subTest(command=command):
                self.assertNotEqual(
                    serving_runtime.classify_stable_kernel_path_use(command).outcome,
                    schemas.FAIL)


# =============================================================================
# 5 — the seam regressions this integration pass found
# =============================================================================

class TestSeamRegressions(unittest.TestCase):
    """One test per cross-module disagreement fixed in this pass.

    Each of these passed both modules' own suites before the fix, because each
    module was right about itself.
    """

    def setUp(self):
        self.chain = Chain()

    # -- M1: plan.py re-derives the drop verdict; t3.py used to read it -------

    def test_a_forged_drop_verdict_is_refused_at_both_doors(self):
        forged = dict(
            backend="llama_gpu", may_drop_cells=True,
            unchanged_outcome=schemas.PASS, agreement_outcome=schemas.FAIL,
            stage2_ran=False)
        with self.assertRaises(t3.T3InputError) as ctx:
            t3.UnchangedView(**forged)
        self.assertIn("plan.drop_verdict_contradictions", str(ctx.exception))
        # And the same shape at the compiler's door, so the two agree.
        result = FPL.gpu_unchanged()
        self.assertEqual(plan.drop_verdict_contradictions(result), ())

    def test_a_genuine_unchanged_result_still_drops_its_cells(self):
        # The guard must not forbid the idiom it exists to admit.
        view = t3.unchanged_view(FPL.gpu_unchanged())
        self.assertTrue(view.may_drop_cells)
        self.assertEqual(view.drop_contradictions(), ())

    def test_stage_two_alone_is_not_optional_for_a_drop(self):
        for facet in ({"agreement_outcome": schemas.COULD_NOT_CHECK},
                      {"stage2_ran": False},
                      {"blocking_reasons": ("the trace observed this backend running",)},
                      {"findings": ({"code": "BUILD_IDENTITY_STAGE_DISAGREEMENT"},)}):
            fields = dict(backend="llama_gpu", may_drop_cells=True,
                          unchanged_outcome=schemas.PASS,
                          agreement_outcome=schemas.PASS, stage2_ran=True)
            fields.update(facet)
            with self.assertRaises(t3.T3InputError, msg=str(facet)):
                t3.UnchangedView(**fields)

    # -- M2: §1.6 adjudicated over standings with nothing joining the matrix --

    def test_a_standing_must_name_cells_this_run_measured(self):
        empty = tuple(
            t3.PhaseStanding(backend=s.backend, workload_phase=s.workload_phase,
                             protocol_id=s.protocol_id, standing=s.standing,
                             cell_ids=(), evidence_ref=s.evidence_ref)
            for s in self.chain.standings())
        result = t3.run_t3(self.chain.request(standings=empty))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("the standing names no cells",
                      " | ".join(result.verdict_computation.blocking_reasons))

    def test_deleting_a_phases_cells_while_keeping_its_standing_fails(self):
        # THE defect: the matrix loses every prefill cell, the standing still says
        # prefill was non-inferior, and the gate used to return PASS.
        kept = tuple(r for r in self.chain.results()
                     if not (r.cell.release_phase == t3.PHASE_PERFORMANCE_MATRIX
                             and r.cell.workload_phase == "prefill"))
        view = t3.release_plan_view_from_compiled(
            self.chain.compiled, incumbent_branch="production-consolidated-v8",
            incumbent_commit=V8_HEAD, incumbent_version_number=8,
            extra_cells=extra_cells())
        result = t3.run_t3(self.chain.request(plan=view, _results=kept))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("recorded no result",
                      " | ".join(result.verdict_computation.blocking_reasons))

    def test_a_standing_may_not_be_assembled_across_a_phase_boundary(self):
        decode_ids = tuple(c.cell_id for c in self.chain.performance_cells("decode"))
        crossed = tuple(
            t3.PhaseStanding(backend=s.backend, workload_phase=s.workload_phase,
                             protocol_id=s.protocol_id, standing=s.standing,
                             cell_ids=decode_ids, evidence_ref=s.evidence_ref)
            for s in self.chain.standings())
        result = t3.run_t3(self.chain.request(standings=crossed))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is not that phase's standing",
                      " | ".join(result.verdict_computation.blocking_reasons))

    def test_the_conjunction_cannot_be_satisfied_by_dropping_a_conjunct(self):
        # A protocol map naming only decode does not fail prefill — it deletes it.
        # `schemas.PHASES_BY_BACKEND` is the SSOT `plan.py` and `readiness.py` both
        # already hold themselves to.
        decode_only = {b: {"decode": "P-BENCH-1"} for b in self.chain.view.backends}
        kept = tuple(r for r in self.chain.results()
                     if not (r.cell.release_phase == t3.PHASE_PERFORMANCE_MATRIX
                             and r.cell.workload_phase == "prefill"))
        standings = tuple(s for s in self.chain.standings()
                          if s.workload_phase != "prefill")
        result = t3.run_t3(self.chain.request(
            _results=kept, standings=standings, phase_protocols=decode_only))
        self.assertEqual(result.verdict, "FAIL")
        self.assertIn("is never asked about",
                      " | ".join(result.verdict_computation.blocking_reasons))

    def test_both_declared_phases_still_pass_together(self):
        self.assertEqual(self.chain.run().verdict, "PASS")

    # -- M4: a FAILing run licensed claims AK6 then rendered -----------------

    def test_a_failing_run_licenses_nothing_but_loses_nothing(self):
        failing = self.chain.performance_cells("decode")[0].cell_id
        result = self.chain.run(_failing_cell_id=failing)
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.receipt.claims, ())
        self.assertTrue(result.receipt.withheld_claims)
        self.assertEqual(result.receipt.to_dict()["claims"], [])
        self.assertTrue(result.receipt.to_dict()["withheld_claims"])

    # -- M5: a result could relabel the plan's cell --------------------------

    def test_a_result_may_not_relabel_the_planned_cell(self):
        results = list(self.chain.results())
        original = results[0].cell
        fields = {f.name: getattr(original, f.name)
                  for f in original.__dataclass_fields__.values()}
        fields["co_resident"] = not original.co_resident
        results[0] = t3.CellResult(
            cell=t3.Cell(**fields), check=schemas.Check(schemas.PASS),
            raw_samples_ref="data/x.jsonl", reducer_id="median_mad/v1")
        with self.assertRaises(t3.T3InputError) as ctx:
            self.chain.request(_results=tuple(results))
        self.assertIn("co_resident", str(ctx.exception))

    def test_reps_are_a_measurement_fact_and_may_differ_from_the_plan(self):
        # The compiler runs before anything is measured, so `reps` is deliberately
        # NOT a scope facet. Without this the plan->T3 adapter would be unusable.
        planned = self.chain.view.cells[0]
        self.assertIsNone(planned.reps)
        self.assertEqual(self.chain.run().verdict, "PASS")


# =============================================================================
# 6 — the cardinal rule, proved from the AST of both planes
# =============================================================================

#: Bare-name calls that write, execute or import dynamically. `compile` is here as a
#: NAME only: `re.compile` is an attribute call and is legitimate in every module.
_DENIED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "execfile", "reload",
})

#: Attribute calls that mutate a path, spawn a process, or signal one. `replace` is
#: included deliberately: `Path.replace` IS the move-a-stable-kernel-symlink
#: primitive, and no non-test module in either directory uses `.replace` for
#: anything, so including it costs nothing and closes the door.
_DENIED_ATTRS = frozenset({
    # writes
    "write", "writelines", "write_text", "write_bytes", "writestr", "truncate",
    "flush", "dump",
    # path mutation, including the symlink verbs pathlib spells differently
    "mkdir", "makedirs", "rmdir", "removedirs", "remove", "unlink", "rename",
    "renames", "replace", "symlink", "symlink_to", "hardlink_to", "link",
    "chmod", "lchmod", "chown", "lchown", "mknod", "touch",
    "rmtree", "copy", "copy2", "copyfile", "copytree", "copyfileobj", "move",
    # process
    "system", "popen", "fork", "forkpty", "spawnl", "spawnv", "spawnvp",
    "posix_spawn", "execv", "execve", "execvp", "execvpe", "execl", "execlp",
    "run", "call", "check_call", "check_output", "Popen", "communicate",
    "kill", "killpg", "terminate", "send_signal", "startfile",
    # temp files and sockets
    "mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryFile",
    "TemporaryDirectory", "connect", "sendall", "urlopen",
})

#: The denied primitives that are NEVER exempt, whatever a module's exemption
#: says. An exemption is granted for a capability ("this module writes one JSON
#: file"), and until this existed it was applied to a BUCKET — `write_or_process`
#: — so the exemption for the one writer in the plane also bought it
#: `subprocess.run`, `os.kill`, `os.fork`, sockets and `exec`. An adversarial pass
#: appended `subprocess.run(['systemctl', 'restart', 'dashboard'])` and
#: `os.kill(pid, SIGKILL)` to the exempt module and the audit reported ZERO
#: offenders: path-narrow, capability-wide. On a shared host where every dashboard
#: and inference process belongs to somebody else, spawning and signalling is the
#: capability that must not be reachable — so it is split out of the bucket and
#: the exemption cannot reach it.
_NEVER_EXEMPT_ATTRS = frozenset({
    "system", "popen", "fork", "forkpty", "spawnl", "spawnv", "spawnvp",
    "posix_spawn", "execv", "execve", "execvp", "execvpe", "execl", "execlp",
    "run", "call", "check_call", "check_output", "Popen", "communicate",
    "kill", "killpg", "terminate", "send_signal", "startfile",
    "connect", "sendall", "urlopen",
})
_NEVER_EXEMPT_NAMES = frozenset({"exec", "eval", "execfile", "__import__", "reload"})
_NEVER_EXEMPT_IMPORTS = frozenset({
    "subprocess", "signal", "socket", "multiprocessing", "threading", "pty",
    "asyncio", "http", "urllib", "requests", "ftplib", "smtplib", "webbrowser",
    "ctypes", "runpy", "importlib",
})

#: Modules a non-test module in either plane may not import at all.
_DENIED_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "threading", "tempfile", "sqlite3", "urllib", "http", "requests", "pty",
    "fcntl", "resource", "shlex", "asyncio", "importlib", "posix", "pickle",
    "runpy", "io", "glob", "atexit", "webbrowser", "ftplib", "smtplib",
})

#: RULE 1's scope — the planes, and ONLY the planes. Rule 1 is "this module cannot
#: write or spawn AT ALL", which is true of `release/`, `adapters/` and `surface/`
#: by design and FALSE of the rest of the package by design: `journal.py` appends to
#: a journal, `storage.py` expires artifacts, `resource/device_claim.py` takes lock
#: files, `controller/state_machine.py` persists its own state. Those modules write
#: because writing is their job, and widening rule 1 to them would be a guard that
#: forbids the package's own legitimate idiom.
_PLANE_DIRS = ("release", "adapters", "surface")

#: The directories rule 1 deliberately does NOT reach, each because writing is the
#: module's job. Listed rather than implied: `_PLANE_DIRS` is a hardcoded tuple, so
#: before this existed a new top-level subpackage fell outside rule 1 SILENTLY while
#: looking exactly like a module inside it. `test_every_package_directory_is_classified`
#: makes a new directory a test failure until someone classifies it, which is the
#: same defect class the audit itself exists to catch — a guarantee that quietly
#: stops covering things.
_RULE_ONE_UNSCOPED_DIRS = (
    ".",            # journal.py, storage.py, schemas.py — journal/artifact writers
    "controller",   # state_machine.py latches control state to disk
    "evaluator",    # evaluator/surface.py and friends persist evaluator state
    "resource",     # device_claim.py takes lock files
)

#: RULE 1's ONE exemption, keyed by EXACT package-relative module path.
#:
#: `surface/dashboard_contract.py` is the AK6 operator-surface producer, and it is
#: the only module in a plane that writes. It has to: the `/kernel` panel it feeds
#: was reading `KERNEL_DASHBOARD_JSON` — a gitignored scratch path that does not
#: exist, credited to an exporter that is not part of AutoKernel — and rendering
#: clean over the hole for as long as anyone had been looking. A contract nothing
#: produces is the absence-tolerance scar itself, so "this plane writes nothing"
#: cannot be the property that holds here.
#:
#: What DOES hold, unchanged and unweakened:
#:   * RULE 2 binds over it exactly as over every other module — no call may name a
#:     production branch, a `kernels/production` path, an era registry or an
#:     AutoPilot baseline. The tests below prove rule 2 still bites on this module
#:     specifically, not merely on the corpus containing it.
#:   * the module refuses those targets at RUNTIME too, via
#:     `assert_exportable_destination`, which checks the literal path, the absolute
#:     path AND the symlink-resolved path against `packager.HUMAN_ONLY_TARGET_PATTERNS`
#:     — the same SSOT this audit uses — plus the frozen production trees and the
#:     scratch roots. A static exemption paired with a runtime guard is a different
#:     thing from an exemption alone.
#:
#: The key is an exact path, never a prefix or a directory. A directory-shaped
#: exemption would silently cover the next module someone adds beside it, and
#: `test_the_exemption_does_not_reach_a_sibling_in_the_same_directory` is what stops
#: that from being discovered later.
_RULE_ONE_EXEMPT_MODULES = {
    "surface/dashboard_contract.py":
        "the AK6 /kernel contract producer — the operator surface's only writer, "
        "emitting exactly one derived JSON export to a durable path. Rule 1 cannot "
        "hold for a producer; rule 2 holds over it unchanged, its WRITES are "
        "exempt but nothing in _NEVER_EXEMPT_* is, and its own "
        "assert_exportable_destination refuses every human-only target, every "
        "frozen production tree, every scratch root and every git working tree at "
        "runtime.",
}


def package_relative(path: pathlib.Path, root: pathlib.Path) -> str:
    """The audit's key for a module: its path relative to the package root."""
    return path.relative_to(root).as_posix()


def rule_one_offenders(findings_by_path, *, exempt=None) -> list:
    """Rule-1 offenders, with the narrow per-module exemption applied.

    Takes an already-computed `{package-relative path: findings}` mapping rather
    than reading the filesystem, so a test can hand it a corpus containing a module
    that does not exist — which is the only way to prove the exemption does NOT
    widen to a sibling that has not been written yet.

    The exemption is bounded on BOTH axes. By path: exact keys, no prefixes. By
    CAPABILITY: it suppresses write findings only, and `process_or_exec` findings
    are added back for an exempt module, so the one module allowed to write a file
    still may not spawn, signal, open a socket or execute a string.
    """
    exempt = _RULE_ONE_EXEMPT_MODULES if exempt is None else exempt
    offenders = []
    for rel in sorted(findings_by_path):
        findings = findings_by_path[rel]
        if rel in exempt:
            offenders.extend(findings.get("process_or_exec", []))
            continue
        offenders.extend(findings["write_or_process"])
    return offenders


def discover_plane_modules(root: pathlib.Path) -> list:
    """Every `.py` under the two planes — RECURSIVELY.

    `rglob`, not `glob`. The non-recursive form audited only files sitting directly
    in `release/` and `adapters/`, so a module added at `release/anything/x.py` was
    outside the corpus while looking exactly like a module inside it — and the
    completeness test would still have passed, because it asserts named modules and a
    floor, neither of which a new subpackage changes.
    """
    found = []
    for name in _PLANE_DIRS:
        for path in sorted((root / name).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.append(path)
    return found


def discover_package_modules(root: pathlib.Path) -> list:
    """Every `.py` in the WHOLE autokernel package — rule 2's scope.

    Rule 2 is *"no call anywhere names a human-only target"*, and unlike rule 1 that
    obligation is not a property of a plane. It binds hardest exactly where this
    corpus used to stop: `storage.py` imports `shutil` and `subprocess` and calls
    `rmtree`/`makedirs`, `journal.py` and `resource/device_claim.py` call `mkdir` and
    `replace`, `controller/state_machine.py` calls `write`. Those are the modules
    that HAVE the capability to repoint a stable kernel symlink or delete a
    production tree, and they were the twenty-four this audit never looked at — the
    audit finds such a call the instant it is handed the source, so the defect was
    never the detector, only the corpus it was pointed at.
    """
    return [path for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts]


#: The ONE qualified name exempted from `_DENIED_ATTRS`, and only in this exact
#: spelling. `Path.replace` is the move-a-stable-kernel-symlink primitive and
#: `str.replace`/`datetime.replace`/`dataclasses.replace` are indistinguishable from
#: it to an AST. The resolution is not to drop `replace` from the denylist — that
#: would open the primitive — but to exempt the one call that provably cannot touch a
#: filesystem: `dataclasses.replace` returns a new dataclass instance and has no I/O
#: at all. An unqualified `x.replace(...)` or a `Path(...).replace(...)` is still
#: caught, and a test below proves both.
_EXEMPT_QUALIFIED_CALLS = frozenset({"dataclasses.replace"})


def _qualified_call_name(func: ast.Attribute) -> str:
    return (f"{func.value.id}.{func.attr}"
            if isinstance(func.value, ast.Name) else "")


def _denied_call_name(node: ast.Call) -> str:
    """The denied name this call invokes, or ''. Covers the `getattr` bypass."""
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in _DENIED_NAMES:
            return func.id
        # `getattr(x, "write_text")()` routes around attribute matching entirely.
        if func.id == "getattr" and len(node.args) >= 2:
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str) \
                    and second.value in (_DENIED_ATTRS | _DENIED_NAMES):
                return f"getattr(..., {second.value!r})"
        return ""
    if isinstance(func, ast.Attribute) and func.attr in _DENIED_ATTRS:
        if _qualified_call_name(func) in _EXEMPT_QUALIFIED_CALLS:
            return ""
        return func.attr
    return ""


def _never_exempt_call_name(node: ast.Call) -> str:
    """The never-exempt primitive this call invokes, or ''.

    Mirrors `_denied_call_name`, including the `getattr` bypass, over the
    spawn/signal/socket/dynamic-exec subset. Kept as its own function rather than
    a filter over the finding STRINGS: parsing a message back into a primitive is
    how a guard stops matching the day someone rewords the message.
    """
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in _NEVER_EXEMPT_NAMES:
            return func.id
        if func.id == "getattr" and len(node.args) >= 2:
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str) \
                    and second.value in (_NEVER_EXEMPT_ATTRS | _NEVER_EXEMPT_NAMES):
                return f"getattr(..., {second.value!r})"
        return ""
    if isinstance(func, ast.Attribute) and func.attr in _NEVER_EXEMPT_ATTRS:
        return func.attr
    return ""


def _string_constants(node: ast.AST) -> list:
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _human_only_reasons(text: str) -> list:
    return [reason for pattern, reason in packager.HUMAN_ONLY_TARGET_PATTERNS
            if pattern.search(text)]


def audit_module_source(source: str, *, label: str, is_test: bool) -> dict:
    """Every write/process call in `source`, and which of them name a human-only target.

    Two findings lists, because the two rules have different scopes:

      * `write_or_process` — any denied call or import. For a NON-test module this is
        the whole rule: the release plane and the adapter plane both claim, in their
        package docstrings, that nothing in them writes a file or touches a process.
      * `human_only_targets` — a denied call one of whose argument strings names a
        production branch, a stable kernel path, the era registry, the AutoPilot
        baseline or the human-only path list. This is the rule that has teeth in a
        TEST module, which legitimately writes into a temporary directory.

    `packager.HUMAN_ONLY_TARGET_PATTERNS` is the SSOT for what a human-only target
    is; this does not restate it.
    """
    findings = {"write_or_process": [], "process_or_exec": [],
                "human_only_targets": [], "unparsed": None}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        findings["unparsed"] = f"{label}: {exc}"
        return findings
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            denied = _denied_call_name(node)
            if not denied:
                continue
            findings["write_or_process"].append(f"{label}:{node.lineno} calls {denied}")
            if _never_exempt_call_name(node):
                findings["process_or_exec"].append(
                    f"{label}:{node.lineno} calls {denied}")
            for text in _string_constants(node):
                for reason in _human_only_reasons(text):
                    findings["human_only_targets"].append(
                        f"{label}:{node.lineno} {denied}({text!r}) — {reason}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            else:
                roots = [(node.module or "").split(".")[0]]
            for root in roots:
                if root in _DENIED_IMPORTS:
                    findings["write_or_process"].append(
                        f"{label}:{node.lineno} imports {root!r}")
                if root in _NEVER_EXEMPT_IMPORTS:
                    findings["process_or_exec"].append(
                        f"{label}:{node.lineno} imports {root!r}")
    if is_test:
        findings["write_or_process"] = []
        findings["process_or_exec"] = []
    return findings


class TestNoProductionWritePathsAnywhere(unittest.TestCase):
    """No module in either plane can write a production branch, a stable kernel
    symlink, an era-registry row or an AutoPilot baseline.

    Parsed from the AST, never grepped. Both directories mention every one of those
    targets in nearly every docstring — `packager.HUMAN_ONLY_TARGET_PATTERNS` exists
    to CLASSIFY the operator's own commands, and the drafted era row names
    `instrument_eras.yaml` on purpose. A grep would fail on correct code and a
    grep tuned until it passed would prove nothing.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = pathlib.Path(__file__).resolve().parent.parent
        cls.modules = discover_plane_modules(cls.root)
        cls.package_modules = discover_package_modules(cls.root)

    def _audit(self, path: pathlib.Path) -> dict:
        return audit_module_source(
            path.read_text(encoding="utf-8"),
            label=str(path.relative_to(self.root)),
            is_test=path.name.startswith("test_"))

    # -- the audit is over the real corpus, computed from disk ---------------

    def test_the_corpus_is_discovered_and_complete(self):
        names = {f"{p.parent.name}/{p.name}" for p in self.modules}
        for expected in ("release/plan.py", "release/readiness.py", "release/t3.py",
                         "release/packager.py", "adapters/serving_runtime.py",
                         "adapters/whisper_stt.py", "adapters/qwentts_tts.py"):
            self.assertIn(expected, names)
        # A glob that returned nothing would pass every assertion below vacuously.
        self.assertGreaterEqual(len(self.modules), 10)

    def test_every_module_parses(self):
        for path in self.modules:
            self.assertIsNone(self._audit(path)["unparsed"], path)

    # -- the corpus cannot be escaped by nesting ------------------------------

    def test_a_module_nested_under_a_plane_is_still_discovered(self):
        """The recursion, proved on a tree this test builds rather than on the
        package's current shape — `release/` and `adapters/` are flat TODAY, so a
        non-recursive glob and a recursive one agree on the real corpus and the
        defect would be invisible if this asserted against the package.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for name in _PLANE_DIRS:
                (root / name / "nested").mkdir(parents=True)
                (root / name / "flat.py").write_text("x = 1\n", encoding="utf-8")
                (root / name / "nested" / "deep.py").write_text("y = 2\n",
                                                                encoding="utf-8")
            found = {p.name for p in discover_plane_modules(root)}
            self.assertIn("flat.py", found)
            self.assertIn("deep.py", found)   # the bite: `glob` never returns this

    def test_the_discovery_never_returns_compiled_bytecode_caches(self):
        """The compliant-path control for the recursion: `rglob` now reaches into
        `__pycache__`, whose `.pyc` files are not source and would be an unparsable
        finding in every module. Excluding them is not a hole — a `.pyc` has no
        author and cannot be reviewed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for name in _PLANE_DIRS:
                (root / name / "__pycache__").mkdir(parents=True)
                (root / name / "real.py").write_text("x = 1\n", encoding="utf-8")
                (root / name / "__pycache__" / "real.py").write_text(
                    "import os\n", encoding="utf-8")
            found = discover_plane_modules(root)
            self.assertEqual({p.name for p in found}, {"real.py"})
            self.assertTrue(all("__pycache__" not in p.parts for p in found))

    # -- rule 2 binds over the WHOLE package, not just the two planes ---------

    def test_rule_two_covers_every_module_in_the_package(self):
        """§ the cardinal rule. Rule 1's scope is the two planes because only they
        can honestly claim "no writes at all". Rule 2 has no such excuse, and the
        modules it was NOT applied to are precisely the ones holding the primitives:
        `storage.py` (`shutil`, `subprocess`, `rmtree`), `journal.py` (`mkdir`,
        `replace`), `resource/device_claim.py`, `controller/state_machine.py`.
        """
        offenders = []
        for path in self.package_modules:
            offenders.extend(self._audit(path)["human_only_targets"])
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_the_package_corpus_contains_the_write_capable_modules(self):
        """Anti-vacuity, and the whole point of widening the corpus. If this list
        ever shrinks to nothing the test above passes by inspecting nothing.
        """
        names = {str(p.relative_to(self.root)) for p in self.package_modules}
        for expected in ("storage.py", "journal.py", "schemas.py",
                         "resource/device_claim.py", "controller/state_machine.py",
                         "evaluator/surface.py"):
            self.assertIn(expected, names)
        self.assertGreater(len(self.package_modules), len(self.modules))

    def test_the_write_capable_modules_really_do_hold_the_primitives(self):
        """The corpus above is worth auditing only if those modules CAN do the
        forbidden thing. Asserted, not assumed: each names a write/process primitive
        rule 1 would refuse — which is exactly why rule 1 does not reach them and
        rule 2 must.
        """
        for name in ("storage.py", "journal.py", "resource/device_claim.py",
                     "controller/state_machine.py"):
            path = self.root / name
            findings = audit_module_source(
                path.read_text(encoding="utf-8"), label=name, is_test=False)
            self.assertTrue(findings["write_or_process"],
                            f"{name} holds no write primitive, so rule 2 over it "
                            "proves nothing")

    def test_rule_two_bites_on_a_write_capable_module_outside_the_planes(self):
        """The bite, on the real defect: a production-tree delete appended to the
        real `storage.py`. Before the corpus was widened this returned a finding the
        suite never asked for, so the module could have shipped it.
        """
        source = (self.root / "storage.py").read_text(encoding="utf-8")
        doctored = source + (
            "\n\ndef cutover_production():\n"
            "    import shutil\n"
            "    shutil.rmtree('/mnt/raid0/llm/kernels/production/cpu')\n")
        findings = audit_module_source(doctored, label="storage.py", is_test=False)
        self.assertTrue(findings["human_only_targets"])

    def test_rule_two_does_not_forbid_the_write_capable_modules_own_idiom(self):
        """The compliant-path control for widening rule 2: `storage.py` and friends
        keep writing where they legitimately write. Rule 2 forbids naming a
        human-only TARGET, never writing as such — a guard that banned `mkdir` here
        would have banned the package's own storage plane.
        """
        for name in ("storage.py", "journal.py", "resource/device_claim.py"):
            path = self.root / name
            findings = audit_module_source(
                path.read_text(encoding="utf-8"), label=name, is_test=False)
            self.assertEqual(findings["human_only_targets"], [], name)

    def test_an_empty_package_corpus_is_not_a_pass(self):
        """A guarantee obtainable by deleting what it inspects is not one: an empty
        root yields no offenders, so emptiness must be refused by the corpus test
        rather than read as compliance.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(discover_package_modules(pathlib.Path(tmp)), [])
        # ...and the real corpus is what stops that from being the state we are in.
        self.assertGreaterEqual(len(self.package_modules), 20)

    # -- rule 1: no write or process path in any non-test module -------------

    def _rule_one_corpus(self) -> dict:
        """`{package-relative path: findings}` for every non-test plane module."""
        return {package_relative(path, self.root): self._audit(path)
                for path in self.modules if not path.name.startswith("test_")}

    def test_no_non_test_module_writes_or_spawns_at_all(self):
        offenders = rule_one_offenders(self._rule_one_corpus())
        self.assertEqual(offenders, [], "\n".join(offenders))

    # -- rule 1's scope is TOTAL: a new directory cannot escape it silently ---

    def test_every_package_directory_is_classified(self):
        """Every directory holding a non-test module is in exactly one bucket.

        `_PLANE_DIRS` is a hardcoded tuple, so a top-level subpackage added
        tomorrow used to fall outside rule 1 while looking exactly like a module
        inside it — the same shape as the `glob`/`rglob` defect one layer up. This
        makes the omission a failure instead of a silence: a new directory is
        either a plane (rule 1 applies) or explicitly listed as a writer.
        """
        classified = set(_PLANE_DIRS) | set(_RULE_ONE_UNSCOPED_DIRS)
        found = set()
        for path in discover_package_modules(self.root):
            if path.name.startswith("test_"):
                continue
            found.add(path.relative_to(self.root).parent.as_posix())
        unclassified = sorted(found - classified)
        self.assertEqual(
            unclassified, [],
            f"unclassified package directories {unclassified}: each must be added "
            f"to _PLANE_DIRS (rule 1 applies: the modules write nothing) or to "
            f"_RULE_ONE_UNSCOPED_DIRS (writing is their job). Leaving one out is "
            f"not neutral — it removes rule 1 from those modules without saying so.")
        # Anti-vacuity: both buckets must actually be populated by the real tree.
        self.assertTrue(found & set(_PLANE_DIRS))
        self.assertTrue(found & set(_RULE_ONE_UNSCOPED_DIRS))

    def test_an_unclassified_directory_is_detected(self):
        """The bite for the test above, on a tree it builds: an unlisted directory
        holding a module must show up as unclassified.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "web").mkdir()
            (root / "web" / "panel.py").write_text("x = 1\n", encoding="utf-8")
            found = {p.relative_to(root).parent.as_posix()
                     for p in discover_package_modules(root)
                     if not p.name.startswith("test_")}
            classified = set(_PLANE_DIRS) | set(_RULE_ONE_UNSCOPED_DIRS)
            self.assertEqual(sorted(found - classified), ["web"])

    # -- rule 1's single exemption, and the fence around it ------------------

    def test_the_rule_one_exemption_names_exactly_one_module_by_exact_path(self):
        """One entry, one exact path, and a justification long enough to be one."""
        self.assertEqual(set(_RULE_ONE_EXEMPT_MODULES),
                         {"surface/dashboard_contract.py"})
        for rel, reason in _RULE_ONE_EXEMPT_MODULES.items():
            self.assertFalse(rel.endswith("/"), rel)      # not a directory
            self.assertNotIn("*", rel)                    # not a glob
            self.assertTrue(rel.endswith(".py"), rel)     # a module, not a tree
            self.assertGreater(len(reason), 120, rel)

    def test_the_exempt_module_really_does_write(self):
        """Anti-vacuity. An exemption for a module that writes nothing is dead
        weight that will one day be inherited by something that does; if this
        fails, DELETE the exemption rather than keeping it warm.
        """
        path = self.root / "surface" / "dashboard_contract.py"
        findings = audit_module_source(path.read_text(encoding="utf-8"),
                                       label="surface/dashboard_contract.py",
                                       is_test=False)
        self.assertTrue(findings["write_or_process"])

    def test_the_exemption_does_not_reach_a_sibling_in_the_same_directory(self):
        """THE fence: the exemption is keyed by exact path, so a second writer in
        `surface/` is still an offender. A directory- or prefix-shaped exemption
        would swallow it, and nobody would find out until the second writer had
        been in the tree for a while.
        """
        writer = audit_module_source(
            "import pathlib\n"
            "def go():\n"
            "    pathlib.Path('/mnt/raid0/llm/x.json').write_text('{}')\n",
            label="surface/other_writer.py", is_test=False)
        corpus = {"surface/dashboard_contract.py": writer,
                  "surface/other_writer.py": writer}
        offenders = rule_one_offenders(corpus)
        self.assertTrue(offenders)
        self.assertTrue(all("other_writer" in o for o in offenders), offenders)

    def test_the_exemption_suppresses_rule_one_only_for_its_own_module(self):
        """The compliant-path control: the exempted module's writes are allowed
        through, so the exemption is not a no-op that the test above would pass
        vacuously.
        """
        writer = audit_module_source(
            "import os\n"
            "def go(dst):\n"
            "    os.replace('a', dst)\n",
            label="surface/dashboard_contract.py", is_test=False)
        self.assertTrue(writer["write_or_process"])
        self.assertEqual(
            rule_one_offenders({"surface/dashboard_contract.py": writer}), [])

    def test_the_exemption_does_not_reach_spawning_or_signalling(self):
        """THE OTHER fence, and the one the exemption did not have.

        The exemption was keyed by exact path and reviewed on that axis alone,
        while it suppressed a whole findings BUCKET — so the module allowed to
        write one JSON file was equally allowed to `subprocess.run` and
        `os.kill`. Proved on the REAL module's source with a hub restart and a
        SIGKILL appended: on a host whose dashboards and inference servers belong
        to other sessions, that is the capability that must not be reachable.
        """
        source = (self.root / "surface" / "dashboard_contract.py").read_text(
            encoding="utf-8")
        doctored = source + (
            "\n\ndef restart_the_hub():\n"
            "    import subprocess\n"
            "    subprocess.run(['systemctl', 'restart', 'dashboard'])\n"
            "    import signal\n"
            "    os.kill(4242, signal.SIGKILL)\n")
        findings = audit_module_source(doctored, label="surface/dashboard_contract.py",
                                       is_test=False)
        offenders = rule_one_offenders({"surface/dashboard_contract.py": findings})
        self.assertTrue(offenders, "a spawn and a kill in the exempt module "
                                   "produced no offender")
        self.assertTrue(any("run" in o for o in offenders), offenders)
        self.assertTrue(any("kill" in o for o in offenders), offenders)

    def test_the_exempt_module_spawns_and_signals_nothing_as_it_ships(self):
        """The compliant-path control for the fence above: the real module has an
        empty never-exempt bucket, so the check is a fence and not a ban on the
        module existing. Its `os.write`/`os.replace`/`os.makedirs` writes — the
        thing it IS exempt for — still pass.
        """
        path = self.root / "surface" / "dashboard_contract.py"
        findings = audit_module_source(path.read_text(encoding="utf-8"),
                                       label="surface/dashboard_contract.py",
                                       is_test=False)
        self.assertEqual(findings["process_or_exec"], [])
        self.assertTrue(findings["write_or_process"])
        self.assertEqual(
            rule_one_offenders({"surface/dashboard_contract.py": findings}), [])

    def test_the_never_exempt_primitives_are_a_subset_of_the_denied_ones(self):
        """Anti-drift. A never-exempt primitive that rule 1 does not deny at all
        would be a rule nothing computes: the split is a PARTITION of the existing
        denylist, not a second denylist beside it.
        """
        self.assertTrue(_NEVER_EXEMPT_ATTRS <= _DENIED_ATTRS)
        self.assertTrue(_NEVER_EXEMPT_NAMES <= _DENIED_NAMES)
        self.assertTrue(_NEVER_EXEMPT_IMPORTS <= _DENIED_IMPORTS)
        # And the partition is non-trivial in both directions: writing is exemptible,
        # spawning is not.
        self.assertIn("write_text", _DENIED_ATTRS - _NEVER_EXEMPT_ATTRS)
        self.assertIn("Popen", _NEVER_EXEMPT_ATTRS)

    def test_the_getattr_bypass_is_covered_on_the_never_exempt_axis_too(self):
        """`getattr(subprocess, 'Popen')()` routes around attribute matching, and
        the never-exempt check is a second matcher — a bypass closed in one and
        open in the other is a bypass.
        """
        findings = audit_module_source(
            "def go(mod):\n    return getattr(mod, 'Popen')(['x'])\n",
            label="surface/dashboard_contract.py", is_test=False)
        self.assertTrue(findings["process_or_exec"])
        self.assertTrue(rule_one_offenders({"surface/dashboard_contract.py": findings}))

    def test_the_exemption_does_not_suppress_rule_two_on_the_exempt_module(self):
        """Rule 2 is untouched by the exemption. Proved on the REAL module's source
        with a production-symlink write appended — the exemption must not turn the
        one writer in the plane into the one module nobody checks.
        """
        source = (self.root / "surface" / "dashboard_contract.py").read_text(
            encoding="utf-8")
        doctored = source + (
            "\n\ndef cutover(document):\n"
            "    import pathlib\n"
            "    pathlib.Path('/mnt/raid0/llm/kernels/production/cpu')"
            ".write_text('x')\n")
        findings = audit_module_source(doctored, label="surface/dashboard_contract.py",
                                       is_test=False)
        self.assertTrue(findings["human_only_targets"])

    def test_the_exempt_module_names_no_human_only_target(self):
        """The compliant-path control for the check above: the real module, as it
        ships, passes rule 2. It names `HUMAN_ONLY_TARGET_PATTERNS` in order to
        REFUSE those destinations, which is exactly the correct-code case a grep
        would have failed on.
        """
        path = self.root / "surface" / "dashboard_contract.py"
        findings = audit_module_source(path.read_text(encoding="utf-8"),
                                       label="surface/dashboard_contract.py",
                                       is_test=False)
        self.assertEqual(findings["human_only_targets"], [])

    def test_rule_one_still_bites_on_a_non_exempt_plane_module(self):
        """The exemption did not disarm rule 1 for the rest of the plane: the real
        `release/plan.py` with a write appended is still an offender.
        """
        source = (self.root / "release" / "plan.py").read_text(encoding="utf-8")
        doctored = source + (
            "\n\ndef persist():\n"
            "    import pathlib\n"
            "    pathlib.Path('/mnt/raid0/llm/x.json').write_text('{}')\n")
        corpus = {"release/plan.py": audit_module_source(
            doctored, label="release/plan.py", is_test=False)}
        self.assertTrue(rule_one_offenders(corpus))

    # -- rule 2: no call anywhere names a human-only target ------------------

    def test_no_module_passes_a_human_only_target_to_a_write_or_process_call(self):
        offenders = []
        for path in self.modules:
            offenders.extend(self._audit(path)["human_only_targets"])
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_the_human_only_targets_are_present_in_the_corpus_to_be_found(self):
        """Rule 2 must not pass because the corpus never names a production target."""
        named = 0
        for path in self.modules:
            text = path.read_text(encoding="utf-8")
            if _human_only_reasons(text):
                named += 1
        self.assertGreaterEqual(named, 5, "the corpus names no human-only target at "
                                          "all, so rule 2 passed vacuously")

    # -- the audit BITES: doctored sources that must fail --------------------

    def test_the_audit_fails_on_a_symlink_repoint(self):
        doctored = (
            "import pathlib\n"
            "def cutover():\n"
            "    pathlib.Path('/mnt/raid0/llm/llama.cpp-v9/build/bin')"
            ".replace('/mnt/raid0/llm/kernels/production/cpu')\n"
        )
        findings = audit_module_source(doctored, label="doctored", is_test=False)
        self.assertTrue(findings["write_or_process"])
        self.assertTrue(findings["human_only_targets"])

    def test_the_audit_fails_on_an_era_registry_write_in_a_test_module(self):
        doctored = (
            "import pathlib\n"
            "def go():\n"
            "    pathlib.Path('orchestration/instrument_eras.yaml')"
            ".write_text('E9: ...')\n"
        )
        # is_test=True suppresses rule 1 and must NOT suppress rule 2.
        findings = audit_module_source(doctored, label="doctored", is_test=True)
        self.assertEqual(findings["write_or_process"], [])
        self.assertTrue(findings["human_only_targets"])

    def test_the_audit_fails_on_a_production_branch_git_call(self):
        doctored = (
            "import subprocess\n"
            "def freeze():\n"
            "    subprocess.run(['git', 'branch', 'production-consolidated-v9'])\n"
        )
        findings = audit_module_source(doctored, label="doctored", is_test=False)
        self.assertTrue(findings["write_or_process"])
        self.assertTrue(findings["human_only_targets"])

    def test_the_audit_fails_on_an_autopilot_baseline_apply(self):
        doctored = (
            "def apply():\n"
            "    open('orchestration/autopilot_baseline.yaml', 'w').write('x')\n"
        )
        findings = audit_module_source(doctored, label="doctored", is_test=True)
        self.assertTrue(findings["human_only_targets"])

    def test_the_audit_sees_through_the_getattr_bypass(self):
        doctored = (
            "import pathlib\n"
            "def sneak(p):\n"
            "    getattr(pathlib.Path('/mnt/raid0/llm/kernels/production/cpu'),"
            " 'symlink_to')(p)\n"
        )
        findings = audit_module_source(doctored, label="doctored", is_test=False)
        self.assertTrue(findings["write_or_process"])
        self.assertTrue(findings["human_only_targets"])

    def test_an_unparsable_module_is_a_finding_not_a_pass(self):
        findings = audit_module_source("def (:", label="doctored", is_test=False)
        self.assertIsNotNone(findings["unparsed"])

    def test_a_clean_module_still_passes(self):
        # The guard must not forbid its own compliant idiom: reading a file and
        # compiling a regex are what these modules actually do.
        clean = (
            "import re\n"
            "from pathlib import Path\n"
            "PAT = re.compile(r'^x$')\n"
            "def audit():\n"
            "    return Path(__file__).read_text()\n"
        )
        findings = audit_module_source(clean, label="clean", is_test=False)
        self.assertEqual(findings["write_or_process"], [])
        self.assertEqual(findings["human_only_targets"], [])

    # -- the modules' own self-audits agree with this one --------------------

    def test_each_module_self_audit_agrees(self):
        for module in (plan, readiness, t3, packager, serving_runtime,
                       whisper_stt, qwentts_tts):
            audit = getattr(module, "audit_no_write_or_process_paths", None)
            if audit is None:  # plan.py names its audit for what it proves
                audit = module.audit_plan_module_is_read_only
            source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
            # The source is supplied explicitly: the adapters' audits read no file of
            # their own (they hold no `Path(__file__)`), so a no-argument call there
            # is COULD_NOT_CHECK by design rather than PASS.
            self.assertEqual(audit(source).outcome, schemas.PASS, module.__name__)

    def test_a_self_audit_does_not_certify_a_foreign_module(self):
        """Each audit is anchored to its own module, not to whatever string it is given."""
        foreign = pathlib.Path(t3.__file__).read_text(encoding="utf-8")
        for module in (readiness, serving_runtime, whisper_stt, qwentts_tts):
            outcome = module.audit_no_write_or_process_paths(foreign).outcome
            self.assertEqual(outcome, schemas.COULD_NOT_CHECK, module.__name__)


# =============================================================================
# 7 — the §10.4 calibration, replayed through the whole chain
# =============================================================================

class TestPreservedV8Calibration(unittest.TestCase):
    """§10.4: *"the T3 dry-run against preserved v8 artifacts should predict a FAIL
    without the waiver. If it passes, the compiler is wrong."*

    `test_t3.py` calibrates this against an inlined copy of the attestation. This
    reads the REAL ratified artifact off disk, so the calibration is against the
    record rather than against a fixture of it. Read-only.
    """

    RATIFICATION = pathlib.Path(
        "/workspace/artifacts/operator/ratify_v8_final_freeze_20260725.json")
    #: The waiver is read off disk TOO. It used to come from `test_t3.V8_WAIVER`, so
    #: the class docstring's claim — "against the record rather than against a
    #: fixture of it" — was true of the ratification and false of the waiver, which
    #: is the half that decides PASS_WITH_WAIVER. A calibration whose authority
    #: document is a fixture is calibrating the fixture.
    WAIVER = pathlib.Path(
        "/workspace/artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json")

    @classmethod
    def setUpClass(cls):
        # NOT `skipTest`. §10.4's calibration is the one check that says the compiler
        # is right, and a check that evaporates when its subject is removed is a
        # guarantee obtainable by deleting what it inspects. Absence is a FAILURE.
        missing = [str(p) for p in (cls.RATIFICATION, cls.WAIVER) if not p.exists()]
        if missing:
            raise AssertionError(
                "the §10.4 calibration cannot run: " + ", ".join(missing) + " is not "
                "present. This is a FAILURE and never a skip — the preserved v8 "
                "attestation is the only real operator record this suite grades "
                "itself against.")
        cls.ratification = json.loads(cls.RATIFICATION.read_text(encoding="utf-8"))
        cls.waiver = json.loads(cls.WAIVER.read_text(encoding="utf-8"))

    def setUp(self):
        self.freeze = t3.preserved_freeze_from_v8_artifacts(
            self.ratification, self.waiver)

    def test_the_calibration_reads_the_real_documents_not_a_fixture_of_them(self):
        """The inlined fixtures are REDUCTIONS of the real artifacts (`test_t3.py`
        keeps only the fields the gate reads), so the honest seam assertion is not a
        hash of the documents — it is that the READER produces the same freeze from
        either source. If the fixture ever drifts from the record in any field T3
        consumes, the two freezes stop being equal and this fails.
        """
        from_fixture = t3.preserved_freeze_from_v8_artifacts(
            FT3.V8_RATIFICATION, V8_WAIVER)
        self.assertEqual(self.freeze.to_dict(), from_fixture.to_dict())
        self.assertEqual(self.freeze.excluded_pairs,
                         ("qwen36_q8-tg128-iqk1", "qwen36_q8-pp2048-iqk1"))
        self.assertEqual(self.freeze.production_head, FT3.V8_HEAD)
        self.assertEqual(self.freeze.rollback_head, FT3.V7_HEAD)

    def test_the_waiver_on_disk_is_the_document_the_ratification_pins(self):
        """The one authenticity check available here, and it is a real one: the
        attestation carries `evidence_sha256.waive_q8`, so the waiver is not merely
        "a file at the operator path" — it is the document the ratified record
        names. This is the §10.4 binding the red team reported as UNCLOSED for
        caller-supplied waivers generally (U1); for the preserved v8 pair it can be
        closed, and is, because both artifacts exist and one hashes the other.
        """
        digest = hashlib.sha256(self.WAIVER.read_bytes()).hexdigest()
        self.assertEqual(digest, self.ratification["evidence_sha256"]["waive_q8"])
        self.assertEqual(digest, FT3.V8_WAIVER_SHA)

    def test_the_dry_run_fails_without_the_waiver_as_10_4_predicts(self):
        result = t3.run_t3(t3.calibration_request(
            self.freeze, now=NOW, include_waiver=False))
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(set(result.verdict_computation.failed_cells),
                         {"llama_cpu.pair.qwen36_q8-tg128-iqk1",
                          "llama_cpu.pair.qwen36_q8-pp2048-iqk1"})

    def test_the_calibration_run_is_a_dry_run_and_never_a_release(self):
        result = t3.run_t3(t3.calibration_request(self.freeze, now=NOW))
        self.assertEqual(result.mode, t3.MODE_DRY_RUN)
        self.assertIn("NOT a release authorisation",
                      " | ".join(result.phase(t3.PHASE_IDENTITY_PREFLIGHT).notes))

    def test_the_waiver_does_not_clear_the_integrity_spine(self):
        result = t3.run_t3(t3.calibration_request(
            self.freeze, now=NOW, include_waiver=True))
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.verdict_computation.failed_cells, ())
        self.assertIn("no rollback target",
                      " | ".join(result.verdict_computation.blocking_reasons))

    # -- the third outcome, which only `test_t3.py`'s INLINED copy covered ----

    def _with_archive_and_quality(self) -> t3.T3Result:
        """Waiver + a reconstructed N-1 archive + the archived quality baseline."""
        preserved_v7 = t3.IncumbentArchive(entries=(t3.ArchivedBuild(
            generation=t3.ARCHIVE_GENERATION_N1,
            branch="production-consolidated-v7", commit=FT3.V7_HEAD,
            archive_root="/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff",
            binaries=((FT3.V7_BASELINE_BINARY, FT3.digest("v7-llama-server")),),
            libraries=((FT3.LLAMA_BACKENDS,
                        "/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/"
                        "cpu-bin/libggml-base.so.0",
                        FT3.digest("v7-libggml-base")),)),))
        base = t3.calibration_request(self.freeze, now=NOW, include_waiver=True,
                                      archive=preserved_v7)
        quality = tuple(
            t3.QualityEvidence(
                backend=b, mode=t3.QUALITY_MEASURED_PARITY,
                baseline_binary_path=FT3.V7_BASELINE_BINARY,
                baseline_binary_sha256=FT3.digest("v7-llama-server"),
                baseline_kernel="production-consolidated-v7",
                baseline_is_rebuild=False,
                evidence_refs=("data/kernel-v8-candidate/quality-gate/",),
                suites=("mmlu_pro", "gpqa"), shared_question_identity=True)
            for b in self.freeze.backends)
        return t3.run_t3(t3.T3Request(**{
            **{f.name: getattr(base, f.name)
               for f in base.__dataclass_fields__.values()},
            "quality_evidence": quality}))

    def test_the_waiver_plus_an_n_minus_one_archive_is_pass_with_waiver(self):
        result = self._with_archive_and_quality()
        self.assertEqual(result.verdict, "PASS_WITH_WAIVER",
                         " | ".join(result.verdict_computation.blocking_reasons))
        self.assertEqual(result.verdict_computation.blocking_reasons, ())
        self.assertEqual(result.mode, t3.MODE_DRY_RUN)

    def test_exactly_the_two_q8_claims_are_suppressed_by_name(self):
        """"Exactly" is the assertion: an equality on the set, not a membership
        test. A waiver that suppressed a third claim would pass a containment check.
        """
        result = self._with_archive_and_quality()
        self.assertEqual(
            {s["claim"] for s in result.receipt.suppressed_claims},
            {"qwen36_q8-tg128-iqk1 non-regression",
             "qwen36_q8-pp2048-iqk1 non-regression"})
        self.assertEqual(len(result.receipt.suppressed_claims), 2)
        self.assertIn("No v8 Q8 non-regression claim may be made from this campaign.",
                      result.receipt.forfeited_claims)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
