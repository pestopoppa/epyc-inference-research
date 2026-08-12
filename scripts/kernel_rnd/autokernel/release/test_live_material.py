"""No-inference tests for the actual-journal release material compiler."""

from __future__ import annotations

import dataclasses
import tempfile
import unittest

from .. import journal as J
from .. import schemas as S
from ..controller import champion as C
from ..controller import sequencer as Q
from ..controller import test_champion as CF
from . import live_material as L
from . import test_closeout as OF
from . import test_packager as PF


class LiveMaterialCompilerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.book = J.Journal(self.temp.name)
        self.book.initialize()
        self.anchor = CF.anchor()
        self.evaluator = CF.evaluator()
        campaign, proposal, candidate, events = CF.records(
            "akc-input", file_name="src/input.cpp", mechanism="fixture:input",
            anchor_id=self.anchor, evaluator_id=self.evaluator)
        self.loop = Q.Sequencer(
            book=self.book,
            proposal_supplier=Q.ListProposalSupplier(
                (Q.ProposalEnvelope(campaign, proposal),)),
            campaign_runner=OF.OneCampaignRunner(candidate, events),
            composition_runner=CF.FakeCompositionRunner(),
            anchor_provider=Q.StaticAnchorProvider({"llama.cpp": self.anchor}),
            evaluators={"llama.cpp": self.evaluator},
            budget=Q.LoopBudget(max_turns=3, max_candidates=3, no_progress_turns=2))
        self.loop.run()
        self.snapshot = C.read_validated_snapshot(self.book)
        self.state = C.project_source_tree(self.snapshot, self.anchor)
        self.champion_event = max(
            (entry for entry in self.snapshot.entries
             if entry.kind == J.KIND_CHAMPION_UPDATED
             and entry.payload.get("status") == "active"),
            key=lambda entry: entry.seq)
        self.freeze = PF.freeze_request(
            campaign_id=self.state.candidates[
                self.state.composed_champion].campaign["campaign_id"])
        self.fixture_material = OF.FixtureCompiler().compile(
            freeze_request=self.freeze, state=self.state, snapshot=self.snapshot,
            champion_event=self.champion_event)
        self.candidate = self.state.candidates[self.state.composed_champion]
        overlay_receipt = S.content_hash({"overlay": "reviewed-present"})
        record = self.candidate.record
        backend = self.candidate.campaign["backend"]
        base_seal = self.fixture_material.t3_request.sealed
        preliminary = dataclasses.replace(
            base_seal,
            candidate_id=self.state.composed_champion,
            source_tree=self.state.source_tree,
            candidate_branch=record["worktree"]["branch"],
            production_base_commit=self.state.incumbent.commit,
            candidate_commit=record["worktree"]["source_commit"],
            evaluator_bundle_sha256=self.champion_event.payload["evaluator"][
                "bundle_sha256"],
            scope_manifest_sha256=self.candidate.campaign["scope"][
                "derived_role_manifest_sha256"],
            binary_sha256={**base_seal.binary_sha256,
                           backend: record["artifacts"]["binary_sha256"]},
            linkage_sha256={**base_seal.linkage_sha256,
                            backend: record["artifacts"]["linkage_sha256"]},
            build_dirs={**base_seal.build_dirs,
                        backend: record["build"]["build_dir"]})
        sealed = L.bind_sealed_candidate(
            template=preliminary, state=self.state, candidate=self.candidate,
            champion_event=self.champion_event,
            overlay_receipt_sha256=overlay_receipt)
        build_dir = record["build"]["build_dir"]
        linkage_receipts = tuple(
            dataclasses.replace(
                receipt,
                binary_path=f"{build_dir}/bin/llama-server",
                expected_tree_root=build_dir,
                stdout=(f"binary : {build_dir}/bin/llama-server\n"
                        f"  OK   libggml-base.so.0 -> "
                        f"{build_dir}/bin/libggml-base.so.0\n"
                        f"PASS: all linked ggml libraries resolve inside "
                        f"{build_dir}\n"),
                ld_library_path=(f"{build_dir}/bin", "/opt/rocm/lib"))
            if receipt.backend == backend else receipt
            for receipt in self.fixture_material.t3_request.linkage_receipts)
        t3_template = dataclasses.replace(
            self.fixture_material.t3_request, sealed=sealed,
            linkage_receipts=linkage_receipts)
        package_template = dataclasses.replace(
            self.fixture_material.package,
            sealed=dataclasses.replace(
                self.fixture_material.package.sealed, candidate=sealed))
        self.fixture_material = dataclasses.replace(
            self.fixture_material, t3_request=t3_template,
            package=package_template)
        self.receipt = L.make_receipt(
            champion_event=self.champion_event, candidate=self.candidate,
            readiness_report=self.fixture_material.readiness_report,
            t3_template=self.fixture_material.t3_request,
            package_template=self.fixture_material.package,
            overlay_receipt_sha256=overlay_receipt,
            overlay_present=True, sealed_at=PF.NOW)

    def tearDown(self):
        self.temp.cleanup()

    def compiler(self, **overrides):
        values = {
            "readiness_report": self.fixture_material.readiness_report,
            "t3_template": self.fixture_material.t3_request,
            "package_template": self.fixture_material.package,
            "receipt": self.receipt,
        }
        values.update(overrides)
        return L.JournalReleaseMaterialCompiler(**values)

    def test_compiles_material_bound_to_actual_journal_champion(self):
        material = self.compiler().compile(
            freeze_request=self.freeze, state=self.state, snapshot=self.snapshot,
            champion_event=self.champion_event)
        self.assertEqual(material.t3_request.sealed.candidate_id,
                         self.state.composed_champion)
        self.assertEqual(material.t3_request.mode, "dry_run")
        self.assertEqual(material.package.freeze_request, self.freeze)
        self.assertEqual(material.package.sealed.candidate,
                         material.t3_request.sealed)

    def test_operator_closeout_with_live_compiler_reaches_package(self):
        empty_loop = Q.Sequencer(
            book=self.book, proposal_supplier=Q.ListProposalSupplier(()),
            campaign_runner=OF.OneCampaignRunner((), ()),
            composition_runner=CF.FakeCompositionRunner(),
            anchor_provider=Q.StaticAnchorProvider({"llama.cpp": self.anchor}),
            evaluators={"llama.cpp": self.evaluator})
        result = OF.O.OperatorCloseout(
            book=self.book, loop=empty_loop, compiler=self.compiler(),
            request_supplier=OF.DynamicOperatorRequest(), source_tree="llama.cpp",
            evidence_class=OF.O.EVIDENCE_OPERATOR_SUPPLIED).run()
        self.assertTrue(result.ready, result.detail)
        self.assertEqual(result.package["operator_dry_run"]["empirical_claim"], None)

    def test_tampered_readiness_receipt_fails_closed(self):
        bad = dataclasses.replace(
            self.receipt, readiness_report_sha256=S.content_hash({"other": True}))
        body = bad.body()
        bad = dataclasses.replace(bad, receipt_sha256=S.content_hash(body))
        with self.assertRaisesRegex(L.LiveMaterialError, "readiness bytes"):
            self.compiler(receipt=bad).compile(
                freeze_request=self.freeze, state=self.state,
                snapshot=self.snapshot, champion_event=self.champion_event)

    def test_t3_observations_for_another_seal_fail_closed(self):
        other = dataclasses.replace(
            self.fixture_material.t3_request.sealed,
            seal_sha256=S.content_hash({"seal": "other"}))
        request = dataclasses.replace(
            self.fixture_material.t3_request, sealed=other)
        receipt = L.make_receipt(
            champion_event=self.champion_event, candidate=self.candidate,
            readiness_report=self.fixture_material.readiness_report,
            t3_template=request, package_template=self.fixture_material.package,
            overlay_receipt_sha256=self.receipt.overlay_receipt_sha256,
            overlay_present=True, sealed_at=PF.NOW)
        with self.assertRaisesRegex(L.LiveMaterialError, "different evidence seal"):
            self.compiler(t3_template=request, receipt=receipt).compile(
                freeze_request=self.freeze, state=self.state,
                snapshot=self.snapshot, champion_event=self.champion_event)

    def test_absent_overlay_attestation_fails_before_t3(self):
        bad = dataclasses.replace(self.receipt, overlay_present=False)
        bad = dataclasses.replace(bad, receipt_sha256=S.content_hash(bad.body()))
        with self.assertRaisesRegex(L.LiveMaterialError, "overlay"):
            self.compiler(receipt=bad).compile(
                freeze_request=self.freeze, state=self.state,
                snapshot=self.snapshot, champion_event=self.champion_event)


if __name__ == "__main__":
    unittest.main()
