#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
import contextlib
import hashlib
import io
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from .. import journal as J
from .. import schemas as S
from .. import test_campaign_footprint as footprint
from .. import test_schemas as fixtures
from . import champion as C
from . import sequencer as Q


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _concurrent_append(args: tuple[str, str, dict]) -> str:
    root, kind, payload = args
    book = J.Journal(root)
    entry = C.append_idempotent(book, kind, payload)
    return entry.event_id


ANCHOR_COMMIT = fixtures.V8_COMMIT
NEW_ANCHOR_COMMIT = sha("new-production")[:40]
RUNTIME_LABEL = "scripts/kernel_rnd/autokernel/evaluator/runtime-source-v1.json"


def anchor(commit: str = ANCHOR_COMMIT, suffix: str = "v8") -> C.AnchorIdentity:
    return C.AnchorIdentity(
        "llama.cpp", f"production-consolidated-{suffix}", commit,
        tuple(sorted((
            C.AnchorArtifact("llama_cpu", "llama-bench-cpu",
                             sha(f"cpu-binary-{suffix}"), sha(f"cpu-linkage-{suffix}")),
            C.AnchorArtifact("llama_gpu", "llama-bench-hip",
                             sha(f"gpu-binary-{suffix}"), sha(f"gpu-linkage-{suffix}")),
        ))), True)


def evaluator(suffix: str = "one") -> C.EvaluatorIdentity:
    scope = fixtures._campaign()["scope"]["derived_role_manifest_sha256"]
    cells = tuple(sorted((
        C.T2Cell("llama_cpu", "decode", scope, "P-BENCH-1",
                 "decode_tokens_per_s", "higher_better"),
        C.T2Cell("llama_cpu", "prefill", scope, "P-BENCH-PREFILL-1",
                 "prefill_tokens_per_s", "higher_better"),
    )))
    return C.EvaluatorIdentity(
        "P-AK-SEARCH-1/v1", sha(f"evaluator-{suffix}"), RUNTIME_LABEL,
        tuple(sorted(("P-AK-SEARCH-1/v1", "P-BENCH-1", "P-BENCH-PREFILL-1"))),
        cells)


def _banking(events: tuple[dict, ...], *, disposition: str = "banked") -> dict:
    t0 = next(event for event in events if event["tier"] == "T0")
    t1 = next(event for event in events if event["tier"] == "T1")
    return {
        "disposition": disposition,
        "t0": {"all_pass_event_id": t0["event_id"]},
        "sentinels": {"required_all_pass_event_ids": [t0["event_id"]]},
        "real_path_dispatch": {
            "resolution": "confirmed", "event_id": t1["event_id"],
            "gate_id": "real-path-v1"},
        "mechanism": {
            "resolution": "explained", "event_id": t1["event_id"],
            "gate_id": "mechanism-v1"},
        "qualifying_axis": {
            "axis": "throughput", "evaluation_event_id": t1["event_id"],
            "resolution": "above_floor", "observed_effect": 0.04,
            "calibrated_floor": 0.03, "minimum_detectable_effect": 0.025,
            "non_dominated": None, "non_dominated_check_ref": None,
        },
    }


def _evaluation(candidate: dict, campaign: dict, anchor_id: C.AnchorIdentity,
                evaluator_id: C.EvaluatorIdentity, *, tier: str, event_id: str,
                phase: str = "decode", status: str = "pass",
                cell: C.T2Cell | None = None) -> dict:
    artifact = anchor_id.artifact("llama_cpu", "llama-bench-cpu")
    event = copy.deepcopy(fixtures._event_v5())
    protocol = evaluator_id.protocol_ids[0]
    metric = "decode_tokens_per_s"
    direction = "higher_better"
    scope = campaign["scope"]["derived_role_manifest_sha256"]
    if cell is not None:
        phase, protocol, metric, direction = (
            cell.phase, cell.protocol_id, cell.metric, cell.metric_direction)
        scope = cell.scope_manifest_sha256
    event.update({
        "event_id": event_id, "campaign_id": campaign["campaign_id"],
        "candidate_id": candidate["candidate_id"], "tier": tier,
        "backend": "llama_cpu", "device_state": None, "phase": phase,
        "status": status, "integrity_flags": [],
        "scope_manifest_sha256": scope,
        "created_at": f"2026-08-12T03:00:{len(event_id) % 50:02d}Z",
    })
    event["claim_grammar"].update({
        "protocol_id": protocol, "metric": metric,
        "metric_direction": direction})
    event["evaluator"] = {
        "id": evaluator_id.evaluator_id,
        "bundle_sha256": evaluator_id.bundle_sha256,
        "runtime_source_label_ref": evaluator_id.runtime_source_label_ref,
    }
    event["artifact"] = {
        "source_sha256": candidate["source_snapshot"]["snapshot_sha256"],
        "binary_sha256": candidate["artifacts"]["binary_sha256"],
        "linkage_sha256": candidate["artifacts"]["linkage_sha256"],
    }
    event["anchor"] = {
        "source_commit": anchor_id.commit, "tool": artifact.tool,
        "binary_sha256": artifact.binary_sha256,
        "linkage_sha256": artifact.linkage_sha256,
        "measurement_event_ids": ["ake-sealed-anchor"],
    }
    event["performance"] = {
        "raw_samples": [1.0, 1.04], "paired_blocks": 2,
        "estimate": 0.04 if tier == "T2" else 1.02,
        "uncertainty": {"method": "fixture", "value": 0.01},
    }
    if status != "pass":
        event["performance"] = {
            "raw_samples": [], "paired_blocks": 0,
            "estimate": None, "uncertainty": None}
    return event


def records(candidate_id: str, *, file_name: str, mechanism: str,
            anchor_id: C.AnchorIdentity, evaluator_id: C.EvaluatorIdentity,
            status: str = "banked", flag: tuple[str, object] | None = None,
            predicates: tuple[str, ...] = (), event_status: str = "pass"
            ) -> tuple[dict, dict, dict, tuple[dict, ...]]:
    suffix = candidate_id.removeprefix("akc-")
    campaign = copy.deepcopy(fixtures._campaign())
    campaign.update({"campaign_id": f"ak-campaign-{suffix}",
                     "backend": "llama_cpu", "source_tree": "llama.cpp"})
    campaign["production_anchor"].update({
        "repo": "/mnt/raid0/llm/llama.cpp", "branch": anchor_id.branch,
        "commit": anchor_id.commit})
    proposal = copy.deepcopy(fixtures._proposal())
    proposal.update({"proposal_id": f"akp-{suffix}",
                     "campaign_id": campaign["campaign_id"],
                     "change_class": "dispatcher"})
    candidate = copy.deepcopy(fixtures._candidate())
    candidate.update({
        "candidate_id": candidate_id, "campaign_id": campaign["campaign_id"],
        "proposal_id": proposal["proposal_id"], "status": status})
    candidate["worktree"].update({
        "path": f"/tmp/{candidate_id}", "branch": f"ak/{candidate_id}",
        "source_commit": sha(f"source-{candidate_id}")[:40], "clean": True})
    candidate["source_snapshot"].update({
        "snapshot_sha256": sha(f"snapshot-{candidate_id}"),
        "patch_bundle_sha256": sha(f"patch-{candidate_id}")})
    candidate["ancestry"].update({
        "production_base_commit": anchor_id.commit,
        "is_descendant_of_production_base": True})
    candidate["artifacts"].update({
        "binary_sha256": sha(f"binary-{candidate_id}"),
        "linkage_sha256": sha(f"linkage-{candidate_id}")})
    candidate["affected_surface"].update({
        "derived_sha256": sha(f"derived-{candidate_id}"),
        "traced_sha256": sha(f"traced-{candidate_id}"), "reconciled": True})
    candidate["evaluator"] = {
        "id": evaluator_id.evaluator_id,
        "bundle_sha256": evaluator_id.bundle_sha256,
        "runtime_source_label_ref": evaluator_id.runtime_source_label_ref,
    }
    flags = {} if flag is None else {flag[0]: flag[1]}
    candidate["composition_evidence"] = {
        "source_tree": "llama.cpp", "production_base_commit": anchor_id.commit,
        "candidate_source_commit": candidate["worktree"]["source_commit"],
        "patch_bundle_sha256": candidate["source_snapshot"]["patch_bundle_sha256"],
        "actual_files": [file_name],
        "actual_hunk_ids": [sha(f"hunk-{file_name}-{candidate_id}")],
        "actual_symbols": [f"{file_name}:symbol_{suffix}"],
        "derived_surface_tokens": [f"surface_{suffix}"],
        "traced_surface_tokens": [f"surface_{suffix}"],
        "feature_flag_assignments": flags,
        "dispatch_predicates": list(predicates), "mechanism_id": mechanism,
        "change_class": "dispatcher", "evaluator_id": evaluator_id.evaluator_id,
        "evaluator_bundle_sha256": evaluator_id.bundle_sha256,
        "evaluator_runtime_source_label_ref": evaluator_id.runtime_source_label_ref,
        "protocol_ids": list(evaluator_id.protocol_ids),
    }
    events = (
        _evaluation(candidate, campaign, anchor_id, evaluator_id,
                    tier="T0", event_id=f"ake-{suffix}-t0"),
        _evaluation(candidate, campaign, anchor_id, evaluator_id,
                    tier="T1", event_id=f"ake-{suffix}-t1", status=event_status),
    )
    candidate["evaluation_event_ids"] = [event["event_id"] for event in events]
    candidate["banking_verdict"] = _banking(events)
    return campaign, proposal, candidate, events


def materialize(book: J.Journal, candidate_id: str, **kwargs) -> C.CandidateSnapshot:
    campaign, proposal, candidate, events = records(candidate_id, **kwargs)
    book.append(J.KIND_CAMPAIGN_OPENED, campaign)
    book.append(J.KIND_PROPOSAL_RECORDED, proposal)
    book.append(J.KIND_CANDIDATE_RECORDED, candidate)
    for event in events:
        book.append(J.KIND_EVALUATION_EVENT, event)
    state = C.project_source_tree(C.read_validated_snapshot(book), kwargs["anchor_id"])
    return state.candidates[candidate_id]


class FakeCompositionRunner:
    def __init__(self, *, fail: bool = False, drop_last_t2: bool = False):
        self.calls: list[C.CompositionRequest] = []
        self.fail = fail
        self.drop_last_t2 = drop_last_t2

    def _run(self, book: J.Journal, request: C.CompositionRequest,
             request_event: J.JournalEntry) -> C.CompositionReceipt:
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("planted composition failure")
        campaign, proposal, candidate, base_events = records(
            request.combined_candidate_id,
            file_name=f"composed/{request.request_sha256[:12]}.cpp",
            mechanism=f"composition:{request.request_sha256}",
            anchor_id=request.anchor, evaluator_id=request.evaluator)
        candidate["composition_lineage"] = {
            "request_sha256": request.request_sha256,
            "member_candidates": list(request.member_candidates),
            "absorbed_member_candidates": list(request.absorbed_member_candidates),
            "release_package_event_id": request.release_package_event_id,
            "parent_champion_event_id": request.parent_champion_event_id,
            "mode": request.mode,
        }
        cells = request.required_t2_cells[:-1] if self.drop_last_t2 \
            else request.required_t2_cells
        t2_events = tuple(
            _evaluation(
                candidate, campaign, request.anchor, request.evaluator, tier="T2",
                event_id=(f"ake-{request.request_sha256[:12]}-t2-"
                          f"{cell.backend}-{cell.phase}"), cell=cell)
            for cell in cells)
        events = tuple(base_events) + t2_events
        candidate["evaluation_event_ids"] = [event["event_id"] for event in events]
        candidate["banking_verdict"] = _banking(events)
        campaign_entry = C.append_idempotent(book, J.KIND_CAMPAIGN_OPENED, campaign)
        proposal_entry = C.append_idempotent(book, J.KIND_PROPOSAL_RECORDED, proposal)
        candidate_entry = C.append_idempotent(book, J.KIND_CANDIDATE_RECORDED, candidate)
        event_entries = [C.append_idempotent(book, J.KIND_EVALUATION_EVENT, event)
                         for event in events]
        return C.CompositionReceipt(
            request_event.event_id, campaign_entry.event_id, proposal_entry.event_id,
            candidate_entry.event_id, tuple(entry.event_id for entry in event_entries),
            {"build_seconds": 7.0, "evaluation_seconds": 13.0})

    def run_composition(self, book: J.Journal, request: C.CompositionRequest,
                        request_event: J.JournalEntry) -> C.CompositionReceipt:
        return self._run(book, request, request_event)

    def run_reanchor(self, book: J.Journal, request: C.CompositionRequest,
                     request_event: J.JournalEntry) -> C.CompositionReceipt:
        return self._run(book, request, request_event)


class AutoKernelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.book = J.Journal(self.temp.name)
        self.book.initialize()
        self.anchor = anchor()
        self.evaluator = evaluator()

    def tearDown(self):
        self.temp.cleanup()

    def candidate(self, cid: str = "akc-one", **overrides) -> C.CandidateSnapshot:
        args = dict(file_name=f"src/{cid}.cpp", mechanism=f"mechanism:{cid}",
                    anchor_id=self.anchor, evaluator_id=self.evaluator)
        args.update(overrides)
        return materialize(self.book, cid, **args)

    def promote(self, candidates: tuple[C.CandidateSnapshot, ...] | None = None,
                runner: FakeCompositionRunner | None = None):
        candidates = candidates or (self.candidate(),)
        runner = runner or FakeCompositionRunner()
        request = C.composition_request(
            candidates, anchor=self.anchor, evaluator=self.evaluator)
        return request, C.promote_composition(self.book, request, runner), runner

    def test_frontier_requires_validated_write_side_banking(self):
        candidate = self.candidate()
        self.assertIsNotNone(candidate.banking)
        state = C.project_source_tree(C.read_validated_snapshot(self.book), self.anchor)
        self.assertEqual(state.frontier, ("akc-one",))
        raw = dict(candidate.record)
        raw.pop("banking_verdict")
        broken = C.CandidateSnapshot(
            raw, candidate.record_event_id, candidate.campaign,
            candidate.evaluations, candidate.evidence, None)
        self.assertFalse(broken.frontier_eligible(self.anchor))

    def test_full_anchor_artifact_drift_invalidates_active_champion(self):
        self.promote()
        drifted = C.AnchorIdentity(
            self.anchor.source_tree, self.anchor.branch, self.anchor.commit,
            (self.anchor.artifacts[0],
             C.AnchorArtifact("llama_gpu", "llama-bench-hip",
                              sha("drifted-gpu"), sha("drifted-linkage"))))
        state = C.project_source_tree(C.read_validated_snapshot(self.book), drifted)
        self.assertIsNone(state.active_champion)

    def test_file_symbol_and_dispatch_conflicts_fail_closed(self):
        one = self.candidate("akc-one", predicates=("AK_PATH=true",))
        two = self.candidate("akc-two", predicates=("AK_PATH=false",))
        report = C.compatibility((one, two), anchor=self.anchor,
                                 evaluator=self.evaluator)
        self.assertFalse(report.compatible)
        self.assertTrue(any(item.startswith("mutually_exclusive_dispatch:")
                            for item in report.conflicts))
        raw = copy.deepcopy(two.record)
        raw["composition_evidence"]["actual_symbols"] = ["unqualified_symbol"]
        with self.assertRaises(C.EvidenceRefused):
            C.CompositionEvidence.from_candidate(raw, campaign=two.campaign)

    def test_opaque_and_internally_contradictory_predicates_refuse(self):
        opaque = self.candidate("akc-opaque", predicates=("K >= 4096",))
        clean = self.candidate("akc-clean")
        report = C.compatibility((opaque, clean), anchor=self.anchor,
                                 evaluator=self.evaluator)
        self.assertTrue(any(item.startswith("opaque_dispatch_uncomposable:")
                            for item in report.conflicts))
        campaign, _, raw, events = records(
            "akc-contradictory", file_name="src/contradictory.cpp",
            mechanism="mechanism:contradictory", anchor_id=self.anchor,
            evaluator_id=self.evaluator,
            predicates=("AK_PATH=false", "AK_PATH=true"))
        contradictory = C.CandidateSnapshot(
            raw, "journal-candidate-contradictory", campaign, events,
            C.CompositionEvidence.from_candidate(raw, campaign=campaign),
            C.BankingVerdict.from_candidate(raw, events))
        with self.assertRaises(C.CompatibilityRefused):
            C.compatibility((contradictory,), anchor=self.anchor,
                            evaluator=self.evaluator)

    def test_composition_is_requested_before_durable_evidence_and_covers_t2(self):
        request, entry, _ = self.promote()
        entries = self.book.read_all()
        requested = next(item for item in entries
                         if item.kind == J.KIND_COMPOSITION_REQUESTED)
        candidate_entry = next(item for item in entries
                               if item.kind == J.KIND_CANDIDATE_RECORDED
                               and item.record_id == request.combined_candidate_id)
        self.assertLess(requested.seq, candidate_entry.seq)
        self.assertEqual(entry.payload["status"], "active")
        self.assertEqual(len(entry.payload["t2_coverage"]), 2)
        self.assertEqual(set(entry.payload["readiness"]["by_backend"]["llama_cpu"]),
                         {"prefill", "decode"})

    def test_missing_predeclared_t2_cell_fails_and_never_replaces_champion(self):
        _, active, _ = self.promote()
        two = self.candidate("akc-two")
        request = C.composition_request(
            (self.candidate("akc-three"), two), anchor=self.anchor,
            evaluator=self.evaluator, parent_champion_event_id=active.event_id)
        with self.assertRaises(C.CompositionRefused):
            C.promote_composition(
                self.book, request, FakeCompositionRunner(drop_last_t2=True))
        view = C.read_validated_snapshot(self.book).views.champions["llama.cpp"]
        self.assertEqual(view["combined_candidate_id"], active.payload["combined_candidate_id"])
        self.assertTrue(any(item.kind == J.KIND_COMPOSITION_FAILED
                            for item in self.book.read_all()))

    def test_runner_failure_has_separate_attempt_record_and_preserves_champion(self):
        _, active, _ = self.promote()
        two = self.candidate("akc-two")
        request = C.composition_request(
            (two,), anchor=self.anchor, evaluator=self.evaluator,
            parent_champion_event_id=active.event_id)
        with self.assertRaises(RuntimeError):
            C.promote_composition(self.book, request, FakeCompositionRunner(fail=True))
        view = C.read_validated_snapshot(self.book).views.champions["llama.cpp"]
        self.assertEqual(view, active.payload)
        self.assertTrue(any(item.kind == J.KIND_COMPOSITION_FAILED
                            for item in self.book.read_all()))

    def test_incompatible_attempt_record_does_not_mutate_champion(self):
        _, active, _ = self.promote()
        one = self.candidate("akc-overlap-one", file_name="src/shared.cpp")
        two = self.candidate("akc-overlap-two", file_name="src/shared.cpp")
        report = C.compatibility((one, two), anchor=self.anchor,
                                 evaluator=self.evaluator)
        rejected = C.record_rejected_composition(self.book, self.anchor, report)
        self.assertEqual(rejected.kind, J.KIND_COMPOSITION_REJECTED)
        view = C.read_validated_snapshot(self.book).views.champions["llama.cpp"]
        self.assertEqual(view, active.payload)

    def test_exact_replay_is_idempotent(self):
        candidate = self.candidate()
        request = C.composition_request(
            (candidate,), anchor=self.anchor, evaluator=self.evaluator)
        runner = FakeCompositionRunner()
        first = C.promote_composition(self.book, request, runner)
        second = C.promote_composition(self.book, request, runner)
        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(sum(item.kind == J.KIND_COMPOSITION_REQUESTED
                             for item in self.book.read_all()), 1)

    def test_concurrent_idempotent_append_has_one_envelope(self):
        payload = {"state": "NO_PROGRESS", "turns": 2,
                   "candidates_run": 1, "champions_updated": 0,
                   "detail": "same deterministic stop"}
        context = multiprocessing.get_context("fork")
        with context.Pool(6) as pool:
            event_ids = pool.map(
                _concurrent_append,
                [(self.temp.name, J.KIND_STOP_STATE, payload)] * 12)
        self.assertEqual(len(set(event_ids)), 1)
        self.assertEqual(sum(item.kind == J.KIND_STOP_STATE
                             for item in self.book.read_all()), 1)

    def _release_receipt(self, new: C.AnchorIdentity,
                         members: tuple[str, ...]) -> J.JournalEntry:
        package = copy.deepcopy(fixtures._release_package())
        package["package_id"] = f"akr-{new.commit[:12]}"
        package["campaign_id"] = next(iter(
            C.read_validated_snapshot(self.book).views.campaigns))
        package["production_anchor"] = new.to_dict()
        package["sealed_candidate"]["member_candidates"] = list(members)
        return self.book.append(J.KIND_RELEASE_PACKAGE_PREPARED, package)

    def test_reanchor_drops_members_already_present_in_new_production(self):
        one = self.candidate("akc-one")
        two = self.candidate("akc-two")
        _, active, _ = self.promote((one, two))
        new = anchor(NEW_ANCHOR_COMMIT, "v9")
        receipt = self._release_receipt(new, ("akc-one",))
        runner = FakeCompositionRunner()
        entry = C.reanchor_champion(
            self.book, prior_champion=active.payload, old_anchor=self.anchor,
            new_anchor=new, evaluator=self.evaluator, runner=runner)
        self.assertEqual(runner.calls[-1].member_candidates, ("akc-two",))
        self.assertEqual(runner.calls[-1].absorbed_member_candidates, ("akc-one",))
        self.assertEqual(runner.calls[-1].release_package_event_id, receipt.event_id)
        self.assertEqual(entry.payload["status"], "reanchored")

    def test_reanchor_without_matching_sealed_receipt_fails_closed(self):
        _, active, _ = self.promote()
        new = anchor(NEW_ANCHOR_COMMIT, "v9")
        with self.assertRaises(C.AnchorMoved):
            C.reanchor_champion(
                self.book, prior_champion=active.payload, old_anchor=self.anchor,
                new_anchor=new, evaluator=self.evaluator,
                runner=FakeCompositionRunner())
        latest = C.read_validated_snapshot(self.book).views.champions["llama.cpp"]
        self.assertEqual(latest["status"], "anchor_moved")

    def test_sequencer_composes_existing_frontier_without_importing_campaign(self):
        self.candidate()

        class UnusedCampaignRunner:
            def run_campaign(self, proposal):  # pragma: no cover
                raise AssertionError(proposal)

        result = Q.Sequencer(
            book=self.book, proposal_supplier=Q.ListProposalSupplier(()),
            campaign_runner=UnusedCampaignRunner(),
            composition_runner=FakeCompositionRunner(),
            anchor_provider=Q.StaticAnchorProvider({"llama.cpp": self.anchor}),
            evaluators={"llama.cpp": self.evaluator},
            budget=Q.LoopBudget(max_turns=2, max_candidates=2,
                                no_progress_turns=1)).run()
        self.assertEqual(result.stop_reason, Q.StopReason.NO_PROPOSAL)
        self.assertEqual(result.champions_updated, 1)


class StaticBoundaryTests(unittest.TestCase):
    def test_cli_identity_manifest_loads_runtime_and_t2_cells(self):
        with tempfile.TemporaryDirectory() as root:
            book = J.Journal(root)
            book.initialize()
            manifest = Path(root) / "identities.json"
            manifest.write_text(json.dumps({
                "anchors": {"llama.cpp": anchor().to_dict()},
                "evaluators": {"llama.cpp": evaluator().to_dict()},
            }))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = Q.main([
                    "--journal-root", root,
                    "--identity-manifest", str(manifest),
                ])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["mode"], "inspect_only")

    def test_controller_has_no_process_mutation_or_release_imports(self):
        root = Path(__file__).parent
        for name in ("champion.py", "sequencer.py"):
            tree = ast.parse((root / name).read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
            for forbidden in ("subprocess", "multiprocessing", "os", "signal",
                              "execution", "worktree", "release", "adapters"):
                self.assertFalse(any(forbidden in item for item in imported),
                                 (name, forbidden, imported))

    def test_campaign_import_closure_cannot_reach_new_controller(self):
        graph = footprint.ImportGraph(footprint.PKG_DIR, footprint.ROOT_PKG)
        modules = set(graph.closure(footprint.campaign_roots()))
        self.assertNotIn(f"{footprint.ROOT_PKG}.controller.champion", modules)
        self.assertNotIn(f"{footprint.ROOT_PKG}.controller.sequencer", modules)


if __name__ == "__main__":
    unittest.main()
