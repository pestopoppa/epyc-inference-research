#!/usr/bin/env python3
"""End-to-end and recovery rehearsals for operator-triggered dry-run closeout.

All records are deterministic architecture fixtures.  No empirical claim is
made, and no inference, build, process, model, production tree or stack is used.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .. import journal as J
from .. import schemas as S
from .. import test_campaign_footprint as footprint
from ..controller import champion as C
from ..controller import sequencer as Q
from ..controller import test_champion as CF
from ..evaluator import api
from . import closeout as O
from . import packager as P
from . import readiness as R
from . import t3
from . import test_packager as PF
from . import test_readiness as RF


class OneCampaignRunner:
    def __init__(self, candidate, events):
        self.candidate = candidate
        self.events = events
        self.calls = 0

    def run_campaign(self, proposal):
        self.calls += 1
        return Q.CampaignRun((self.candidate,), tuple(self.events))


class DynamicOperatorRequest:
    def request(self, *, state, snapshot, champion_event):
        campaign_id = state.candidates[state.composed_champion].campaign["campaign_id"]
        return PF.freeze_request(campaign_id=campaign_id,
                                 source_tree=state.source_tree)


class FixtureCompiler:
    def __init__(self, *, preempt=False, tamper=None, crash_after=None):
        self.preempt = preempt
        self.tamper = tamper
        self.crash_after = crash_after
        self.calls = 0

    def compile(self, *, freeze_request, state, snapshot, champion_event):
        self.calls += 1
        if self.preempt:
            raise O.ResourcePreempted("exclusive release-fixture resource was revoked")
        combined = state.candidates[state.composed_champion]
        evaluator = C.EvaluatorIdentity.from_dict(champion_event.payload["evaluator"])
        cpu_anchor = state.incumbent.artifact("llama_cpu", "llama-bench-cpu")
        signal_anchor = api.AnchorIdentity(
            source_commit=state.incumbent.commit,
            binary_sha256=cpu_anchor.binary_sha256,
            linkage_sha256=cpu_anchor.linkage_sha256,
            measurement_event_ids=("ake-fixture-anchor",),
            tool=cpu_anchor.tool)
        lineage = R.ChampionLineage(
            combined_candidate_id=state.composed_champion,
            source_tree=state.source_tree,
            anchor=signal_anchor,
            entered_lineage_at="2026-08-12T03:00:00+00:00",
            member_candidate_ids=state.champion_members)
        cells = tuple(dataclasses.replace(cell, candidate_id=state.composed_champion)
                      for cell in RF.green_cells())
        mechanisms = tuple(
            R.MechanismConfirmation(
                member_candidate_id=member,
                predicted_mechanism=f"fixture mechanism for {member}",
                confirmed=True,
                event_id=f"ake-mechanism-{index}",
                measured_at="2026-08-12T04:00:00+00:00")
            for index, member in enumerate(state.champion_members))
        signal = R.compute_readiness(
            backend="llama_cpu", campaign_id=freeze_request.campaign_id,
            champion=lineage, objective=RF.objective(), spec=RF.matrix_spec(),
            cells=cells, controls_marker=R.CONTROLS_COMPLETE,
            evaluator_bundle_sha256=evaluator.bundle_sha256,
            computed_at="2026-08-12T05:00:00+00:00",
            capacity_deltas=RF.capacity(), mechanisms=mechanisms)
        report = R.compute_readiness_report(
            campaign_id=freeze_request.campaign_id,
            computed_at="2026-08-12T05:00:00+00:00", signals=(signal,))

        source = combined.record["source_snapshot"]
        worktree = combined.record["worktree"]
        sealed_candidate = PF.sealed_candidate(
            candidate_id=state.composed_champion,
            production_base_commit=state.incumbent.commit,
            candidate_commit=worktree["source_commit"],
            seal_sha256=S.content_hash({"fixture-seal": state.composed_champion}),
            evaluator_bundle_sha256=evaluator.bundle_sha256,
            scope_manifest_sha256=combined.campaign["scope"][
                "derived_role_manifest_sha256"],
            evidence_tree_sha256=S.content_hash({
                "candidate": state.composed_champion,
                "source_snapshot": source["snapshot_sha256"],
                "evaluation_ids": combined.record["evaluation_event_ids"],
            }))
        request = PF.t3_request(
            campaign_id=freeze_request.campaign_id,
            sealed=sealed_candidate)
        if self.tamper == "candidate":
            request = dataclasses.replace(
                request,
                sealed=dataclasses.replace(
                    sealed_candidate, candidate_id="akc-foreign-fixture"))

        sealed_release = PF.sealed_release(
            champion_id=f"akch-{state.composed_champion}",
            candidate=sealed_candidate)
        package = O.PackageAssemblyInputs(
            package_id="akr-v9-001", created_at=PF.NOW,
            freeze_request=freeze_request, sealed=sealed_release,
            version=PF.next_version(), transaction=PF.transaction_plan(),
            rollback=PF.rollback_plan(), era_row_draft=PF.era_draft(),
            rebaseline_note=PF.rebaseline_note(), commands=PF.operator_commands(),
            watch_window=PF.watch_window(), cutover_request=PF.cutover_request(),
            autopilot_baseline_path=PF.AUTOPILOT_BASELINE,
            change_classes=("arithmetic",),
            diff_complexity=dict(PF.DIFF_COMPLEXITY))
        if self.crash_after == "material":
            raise RuntimeError("planted crash after deterministic material compilation")
        return O.CompiledReleaseMaterial(report, request, package)


class _AppendCrashProxy:
    def __init__(self, book, kind):
        self.book = book
        self.kind = kind
        self.crashed = False

    def write_lock(self):
        return self.book.write_lock()

    def read_all(self):
        return self.book.read_all()

    def append(self, kind, payload, **kwargs):
        entry = self.book.append(kind, payload, **kwargs)
        if kind == self.kind and not self.crashed:
            self.crashed = True
            raise RuntimeError(f"planted crash after fsynced {kind}")
        return entry


class CrashOnceCompositionRunner(CF.FakeCompositionRunner):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.did_crash = False

    def _run(self, book, request, request_event):
        if not self.did_crash:
            proxy = _AppendCrashProxy(book, self.kind)
            try:
                return super()._run(proxy, request, request_event)
            finally:
                self.did_crash = proxy.crashed
        return super()._run(book, request, request_event)


class CrashAfterRequestCloseout(O.OperatorCloseout):
    """The production implementation has no hook here; this is a crash rehearsal."""

    def run(self):
        search = self.loop.run()
        snapshot = C.read_validated_snapshot(self.book)
        state = C.project_source_tree(
            snapshot, self.loop.anchor_provider.current_anchor(self.source_tree))
        champion_event = O._latest_champion_event(snapshot, state.source_tree)
        freeze_request = self.request_supplier.request(
            state=state, snapshot=snapshot, champion_event=champion_event)
        payload = O._request_payload(
            freeze_request=freeze_request, champion_event=champion_event,
            state=state, evidence_class=self.evidence_class)
        C.append_idempotent(
            self.book, J.KIND_OPERATOR_RELEASE_DRY_RUN_REQUESTED, payload)
        raise RuntimeError("planted power loss after fsynced operator request")


class CloseoutIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.book = J.Journal(self.temp.name)
        self.book.initialize()
        self.anchor = CF.anchor()
        self.evaluator = CF.evaluator()
        campaign, proposal, candidate, events = CF.records(
            "akc-input", file_name="src/input.cpp", mechanism="fixture:input",
            anchor_id=self.anchor, evaluator_id=self.evaluator)
        self.envelope = Q.ProposalEnvelope(campaign, proposal)
        self.campaign_runner = OneCampaignRunner(candidate, events)
        self.composition_runner = CF.FakeCompositionRunner()

    def tearDown(self):
        self.temp.cleanup()

    def loop(self, proposals=None):
        return Q.Sequencer(
            book=self.book,
            proposal_supplier=Q.ListProposalSupplier(
                (self.envelope,) if proposals is None else proposals),
            campaign_runner=self.campaign_runner,
            composition_runner=self.composition_runner,
            anchor_provider=Q.StaticAnchorProvider({"llama.cpp": self.anchor}),
            evaluators={"llama.cpp": self.evaluator},
            budget=Q.LoopBudget(max_turns=3, max_candidates=3,
                                no_progress_turns=2))

    def closeout(self, compiler=None, *, loop=None):
        return O.OperatorCloseout(
            book=self.book, loop=loop or self.loop(),
            compiler=compiler or FixtureCompiler(),
            request_supplier=DynamicOperatorRequest(), source_tree="llama.cpp",
            evidence_class=O.EVIDENCE_ARCHITECTURE_FIXTURE)

    def test_end_to_end_reaches_validated_ready_package(self):
        result = self.closeout().run()
        self.assertTrue(result.ready, result.detail)
        self.assertEqual(result.state, P.STATE_READY)
        self.assertEqual(S.validate_release_package(result.package), [])
        self.assertFalse(result.package["operator_dry_run"]["empirical_claim"])
        self.assertEqual(result.package["operator_dry_run"]["evidence_class"],
                         O.EVIDENCE_ARCHITECTURE_FIXTURE)
        self.assertEqual(result.package["sealed_candidate"]["candidate_id"],
                         result.package["readiness_signal"]["signals"][0]
                         ["champion_candidate_id"])
        entries = self.book.read_all()
        self.assertLess(
            next(e.seq for e in entries
                 if e.kind == J.KIND_OPERATOR_RELEASE_DRY_RUN_REQUESTED),
            next(e.seq for e in entries
                 if e.kind == J.KIND_RELEASE_PACKAGE_PREPARED))

    def test_restart_after_request_replays_to_one_package(self):
        crashing = CrashAfterRequestCloseout(
            book=self.book, loop=self.loop(), compiler=FixtureCompiler(),
            request_supplier=DynamicOperatorRequest(), source_tree="llama.cpp",
            evidence_class=O.EVIDENCE_ARCHITECTURE_FIXTURE)
        with self.assertRaisesRegex(RuntimeError, "power loss"):
            crashing.run()
        resumed = self.closeout(loop=self.loop(proposals=())).run()
        replayed = self.closeout(loop=self.loop(proposals=())).run()
        self.assertTrue(resumed.ready)
        self.assertEqual(replayed.package_event_id, resumed.package_event_id)
        self.assertEqual(
            len([e for e in self.book.read_all()
                 if e.kind == J.KIND_RELEASE_PACKAGE_PREPARED]), 1)

    def test_restart_after_package_recovers_without_recompiling(self):
        compiler = FixtureCompiler()
        first = self.closeout(compiler).run()
        refusing = FixtureCompiler(preempt=True)
        second = self.closeout(refusing, loop=self.loop(proposals=())).run()
        self.assertEqual(second.package_event_id, first.package_event_id)
        self.assertEqual(refusing.calls, 0)

    def test_resource_preemption_is_terminal_and_idempotent(self):
        compiler = FixtureCompiler(preempt=True)
        first = self.closeout(compiler).run()
        second = self.closeout(
            FixtureCompiler(), loop=self.loop(proposals=())).run()
        self.assertEqual(first.state, O.STATE_RESOURCE_PREEMPTED)
        self.assertEqual(second.state, O.STATE_RESOURCE_PREEMPTED)
        self.assertEqual(first.terminal_event_id, second.terminal_event_id)
        self.assertFalse(any(e.kind == J.KIND_RELEASE_PACKAGE_PREPARED
                             for e in self.book.read_all()))

    def test_tampered_candidate_is_terminal_and_never_packaged(self):
        result = self.closeout(FixtureCompiler(tamper="candidate")).run()
        self.assertEqual(result.state, O.STATE_TAMPER_REFUSED)
        self.assertIn("another candidate", result.detail)
        self.assertFalse(any(e.kind == J.KIND_RELEASE_PACKAGE_PREPARED
                             for e in self.book.read_all()))

    def test_compiler_crash_leaves_durable_terminal_record(self):
        result = self.closeout(
            FixtureCompiler(crash_after="material")).run()
        self.assertEqual(result.state, O.STATE_FAILED)
        terminal = next(e for e in self.book.read_all()
                        if e.event_id == result.terminal_event_id)
        self.assertEqual(terminal.payload["failure_class"], "RuntimeError")

    def test_candidate_and_evidence_append_crashes_resume_idempotently(self):
        for crash_kind in (J.KIND_CANDIDATE_RECORDED, J.KIND_EVALUATION_EVENT):
            with self.subTest(crash_kind=crash_kind):
                with tempfile.TemporaryDirectory() as root:
                    book = J.Journal(root)
                    book.initialize()
                    campaign, proposal, candidate, events = CF.records(
                        "akc-input", file_name="src/input.cpp",
                        mechanism="fixture:input", anchor_id=self.anchor,
                        evaluator_id=self.evaluator)
                    runner = CrashOnceCompositionRunner(crash_kind)
                    first = Q.Sequencer(
                        book=book,
                        proposal_supplier=Q.ListProposalSupplier(
                            (Q.ProposalEnvelope(campaign, proposal),)),
                        campaign_runner=OneCampaignRunner(candidate, events),
                        composition_runner=runner,
                        anchor_provider=Q.StaticAnchorProvider(
                            {"llama.cpp": self.anchor}),
                        evaluators={"llama.cpp": self.evaluator}).run()
                    self.assertEqual(first.stop_reason, Q.StopReason.NO_PROGRESS)
                    resumed = Q.Sequencer(
                        book=book, proposal_supplier=Q.ListProposalSupplier(()),
                        campaign_runner=OneCampaignRunner(candidate, events),
                        composition_runner=runner,
                        anchor_provider=Q.StaticAnchorProvider(
                            {"llama.cpp": self.anchor}),
                        evaluators={"llama.cpp": self.evaluator}).run()
                    self.assertEqual(resumed.champions_updated, 1)
                    state = C.project_source_tree(
                        C.read_validated_snapshot(book), self.anchor)
                    self.assertIsNotNone(state.active_champion)

    def test_crash_after_composed_evidence_before_champion_resumes(self):
        original = C.append_idempotent
        crashed = False

        def crash_before_champion(book, kind, payload):
            nonlocal crashed
            if kind == J.KIND_CHAMPION_UPDATED and not crashed \
                    and payload.get("status") == "active":
                crashed = True
                raise RuntimeError("planted crash before champion visibility")
            return original(book, kind, payload)

        with mock.patch.object(C, "append_idempotent", crash_before_champion):
            first = self.loop().run()
        self.assertEqual(first.stop_reason, Q.StopReason.NO_PROGRESS)
        snapshot = C.read_validated_snapshot(self.book)
        composed = [record for record in snapshot.views.candidates.values()
                    if "composition_lineage" in record]
        self.assertEqual(len(composed), 1)
        resumed = self.loop(proposals=()).run()
        self.assertEqual(resumed.champions_updated, 1)
        state = C.project_source_tree(
            C.read_validated_snapshot(self.book), self.anchor)
        self.assertIsNotNone(state.active_champion)


class CloseoutBoundaryTests(unittest.TestCase):
    def test_closeout_is_outside_campaign_import_closure(self):
        graph = footprint.ImportGraph(footprint.PKG_DIR, footprint.ROOT_PKG)
        modules = set(graph.closure(footprint.campaign_roots()))
        self.assertNotIn(f"{footprint.ROOT_PKG}.release.closeout", modules)

    def test_release_initializers_do_not_import_closeout(self):
        source = Path(O.__file__).with_name("__init__.py").read_text()
        self.assertNotIn("closeout", source)

    def test_module_has_no_process_clock_build_inference_or_production_write(self):
        tree = ast.parse(Path(O.__file__).read_text())
        imports = set()
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        self.assertTrue(imports.isdisjoint({
            "os", "subprocess", "multiprocessing", "signal", "socket", "time"}))
        self.assertTrue(calls.isdisjoint({
            "open", "write", "write_text", "write_bytes", "unlink", "remove",
            "rename", "replace", "symlink_to", "Popen", "system", "kill",
            "send_signal", "now", "utcnow", "build", "benchmark", "inference"}))

    def test_module_cannot_mint_an_operator_request(self):
        tree = ast.parse(Path(O.__file__).read_text())
        constructors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and ((isinstance(node.func, ast.Name)
                              and node.func.id == "OperatorFreezeRequest")
                             or (isinstance(node.func, ast.Attribute)
                                 and node.func.attr == "OperatorFreezeRequest"))]
        self.assertEqual(constructors, [])


if __name__ == "__main__":
    unittest.main()
