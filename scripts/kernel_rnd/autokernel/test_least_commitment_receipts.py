#!/usr/bin/env python3
"""Tests for the governed AK-WM-2a journal-to-receipt projection.

Pure local JSON/journal fixtures only: no model, inference, build, process, or
device is opened.  The integration case deliberately exercises the unchanged
archive builder after the producer writes hash-bound receipt files.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from . import journal as J
from . import least_commitment_archive_builder as B
from . import least_commitment_capture as C
from . import least_commitment_receipts as P
from . import offline_least_commitment as L
from . import schemas as S
from .controller import hypotheses as H
from .test_journal import _candidate, _event
from .test_least_commitment_capture import plan as _capture_plan
from .test_least_commitment_capture import proposal as _capture_proposal
from .test_schemas import _proposal


DIAGNOSTIC_DIRECTIONS = {
    name: ("lower" if name in {
        "unsupported_scope_width", "k_rho", "raw_impurity",
    } else "higher")
    for name in L.DIAGNOSTICS
}


class _EmptyDoNotRepeat:
    def matches_for(self, regime, statement):
        return ()


def _diagnostics(score: float) -> dict[str, float]:
    return {
        "unsupported_scope_width": 2.0 - score,
        "compatible_future_mass": score,
        "k_rho": 2.0 - score,
        "information_gain": score,
        "novelty": score,
        "raw_impurity": 2.0 - score,
        "weighted_minority": score,
    }


def _binding(pointer: str, record_sha256: str) -> dict[str, str]:
    return {
        "record": "candidate", "pointer": pointer,
        "record_sha256": record_sha256,
    }


def _bindings(prefix: str, names, record_sha256: str) -> dict[str, dict[str, str]]:
    return {
        name: _binding(f"{prefix}/{name}", record_sha256) for name in names
    }


class ReceiptProjectionTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.rows = [
            self._completed_campaign(
                ordinal=1, campaign_suffix="control", arm="0", score=0.5,
                outcome=0.0),
            self._completed_campaign(
                ordinal=2, campaign_suffix="intervention", arm="1", score=1.0,
                outcome=0.2),
        ]
        self.rows[1]["matched_control_id"] = self.rows[0]["proposal_id"]
        first_candidate = self.rows[0]["candidate_frame_id_binding"]
        self.plan = {
            "schema": P.PLAN_SCHEMA,
            "archive_id": "ak-real-archive-projection-test",
            "created_at": "2026-08-12T00:00:00+00:00",
            "candidate_frame_id_binding": copy.deepcopy(first_candidate),
            "diagnostic_directions": DIAGNOSTIC_DIRECTIONS,
            "outcome_weights": {
                "heldout_regime_transfer": 0.5,
                "falsifier_resolution": 0.5,
            },
            "rows": self.rows,
        }
        self.plan["plan_sha256"] = P._plan_hash(self.plan)

    def tearDown(self):
        self.temp.cleanup()

    def _completed_campaign(self, *, ordinal: int, campaign_suffix: str,
                            arm: str, score: float, outcome: float,
                            semantic_label: str | None = None) -> dict:
        campaign_id = f"ak-llama_cpu-prefill-20260812-{campaign_suffix}"
        proposal_id = f"akp-20260812-{ordinal:04d}"
        candidate_id = f"akc-20260812-{ordinal:04d}"
        evaluation_id = f"ake-20260812-{ordinal:04d}"
        journal_root = self.root / campaign_suffix

        proposal = _proposal()
        proposal.update({
            "proposal_id": proposal_id,
            "campaign_id": campaign_id,
            "campaign_kind": "config",
            "change_class": "parameter",
        })
        proposal["change"]["parameter_surface"] = {
            "candidate": {"ggml_iqk": arm},
            "anchor": {"ggml_iqk": "0"},
        }

        values = _diagnostics(score)
        fixture_ids = proposal["representation_contract"][
            "semantics_preserving_recoding_fixture_ids"]
        least_commitment = {
            "schema": C.BLOCK_SCHEMA,
            "capture_mode": "measured",
            "role": "control" if arm == "0" else "intervention",
            "matched_control_proposal_id": (
                None if arm == "0" else f"akp-20260812-{ordinal - 1:04d}"),
            "matched_experiment_id": "akm-iqk-20260812-0001",
            "candidate_frame_id": "candidate-frame-real-v1",
            "regime": "prefill",
            "surface": "mul_mat",
            "intervention_id": f"ggml-iqk-{arm}",
            "changed_factor": "ggml_iqk",
            "factors": {"ggml_iqk": arm, "threads": 96},
            "diagnostics": values,
            "diagnostic_semantics_sha256": S.content_hash({
                "role": semantic_label or campaign_suffix,
                "diagnostics": values}),
            "recodings": {
                fixture_id: copy.deepcopy(values) for fixture_id in fixture_ids
            },
            "outcome": {
                "heldout_regime_transfer": outcome,
                "falsifier_resolution": outcome,
                "noise_floor": 0.01,
            },
        }
        candidate = _candidate(f"{ordinal:04d}", status="banked")
        candidate.update({
            "candidate_id": candidate_id,
            "campaign_id": campaign_id,
            "proposal_id": proposal_id,
            "evaluation_event_ids": [evaluation_id],
        })
        candidate["derived_verdicts"] = {"least_commitment": least_commitment}

        evaluation = _event(f"{ordinal:04d}")
        evaluation.update({
            "event_id": evaluation_id,
            "campaign_id": campaign_id,
            "candidate_id": candidate_id,
        })
        evaluation["device_state"]["source"] = "rocm-smi"
        evaluation["device_state"]["receipt_ref"] = "rcpt-device-state-journal"

        book = J.Journal(str(journal_root), campaign_id=campaign_id)
        book.initialize()
        book.append(J.KIND_PROPOSAL_RECORDED, proposal)
        hypothesis_id = f"akh-receipt-projection-{ordinal:04d}"
        tracker = H.HypothesisTracker(
            journal_=book, root=str(journal_root), campaign_id=campaign_id)
        tracker.open_hypothesis(H.Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=proposal["hypothesis"],
            falsifier="The accepted paired run fails its predeclared effect floor.",
            origin=H.ORIGIN_CONTROLLER,
            author="least-commitment-receipt-test",
            regime={"recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1"},
        ))
        authorization = tracker.authorize_claim(
            hypothesis_id,
            purpose="exercise the completed-proposal archive admission path",
            authorized_by="least-commitment-receipt-test",
            ledger=_EmptyDoNotRepeat(),
        )
        book.append(J.KIND_EVALUATION_EVENT, evaluation)
        book.append(J.KIND_CANDIDATE_RECORDED, candidate)
        terminal = book.append(J.KIND_STOP_STATE, {
            "state": "decided",
            "result": {
                "state": "decided", "campaign_id": campaign_id,
                "candidate_id": candidate_id, "executed": True, "ok": True,
                "spec": {
                    "recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1",
                    "hypothesis": {
                        "bound": True,
                        "hypothesis_id": hypothesis_id,
                        "authorization": authorization.to_dict(),
                    },
                    "proposal": {
                        "schema": proposal["schema"],
                        "proposal_id": proposal_id,
                        "representation_frame_sha256": proposal[
                            "representation_contract"]["frame_sha256"],
                    },
                },
                "decision": {"keep": arm == "1"},
                "production_unchanged": {"outcome": S.PASS},
                "releases": [{"claim": "cpu", "released": True}],
                "pairs": [{"block_index": 0, "candidate": 1.0, "anchor": 1.0}],
            },
        })

        prefix = "/derived_verdicts/least_commitment"
        candidate_sha256 = S.content_hash(candidate)
        return {
            "journal_root": str(journal_root),
            "campaign_id": campaign_id,
            "proposal_id": proposal_id,
            "completion_event_id": terminal.event_id,
            "candidate_frame_id_binding": _binding(
                f"{prefix}/candidate_frame_id", candidate_sha256),
            "regime_binding": _binding(f"{prefix}/regime", candidate_sha256),
            "surface_binding": _binding(f"{prefix}/surface", candidate_sha256),
            "intervention_id_binding": _binding(
                f"{prefix}/intervention_id", candidate_sha256),
            "changed_factor_binding": _binding(
                f"{prefix}/changed_factor", candidate_sha256),
            "matched_experiment_id_binding": _binding(
                f"{prefix}/matched_experiment_id", candidate_sha256),
            "factor_bindings": _bindings(
                f"{prefix}/factors", ("ggml_iqk", "threads"), candidate_sha256),
            "diagnostic_bindings": _bindings(
                f"{prefix}/diagnostics", L.DIAGNOSTICS, candidate_sha256),
            "recoding_bindings": {
                fixture_id: _bindings(
                    f"{prefix}/recodings/{fixture_id}", L.DIAGNOSTICS,
                    candidate_sha256)
                for fixture_id in fixture_ids
            },
            "outcome_bindings": _bindings(
                f"{prefix}/outcome", P._OUTCOMES, candidate_sha256),
            "matched_control_id": None,
        }

    def _rehash(self):
        self.plan["plan_sha256"] = P._plan_hash(self.plan)

    def _retry_control(self, *, ordinal: int, suffix: str,
                       proposal_mutation=None) -> tuple[dict, dict]:
        """Create one valid bootstrap control plan and clean completed journal."""
        intervention = _capture_proposal()
        intervention.update({
            "campaign_id": f"ak-retry-intervention-{suffix}",
            "proposal_id": f"akp-retry-intervention-{ordinal:04d}",
        })
        control = C.make_iqk_control_proposal(
            intervention, campaign_id=f"ak-retry-control-{suffix}",
            proposal_id=f"akp-retry-control-{ordinal:04d}")
        if proposal_mutation is not None:
            proposal_mutation(control)
        candidate_id = f"akc-retry-control-{ordinal:04d}"
        raw = _capture_plan(control, role="control", matched_control=None)
        raw.update({
            "capture_id": f"aklc-retry-control-{ordinal:04d}",
            "campaign_id": control["campaign_id"],
            "proposal_id": control["proposal_id"],
            "candidate_id": candidate_id,
            "evidence_stage": "bootstrap",
            "heldout_outcome_receipt": None,
            "outcome_reducers": dict(C.BOOTSTRAP_OUTCOME_REDUCERS),
        })
        claim_path = self.root / "shared-retry-claims.jsonl"
        claim_path.touch()
        raw["factors"]["claim_journal_path"] = str(claim_path)
        raw["factors"]["fresh_pairs_per_block"] = 1
        raw["plan_sha256"] = C.plan_sha256(raw)
        parsed = C.from_mapping(
            raw, proposal=control, campaign_id=control["campaign_id"],
            candidate_id=candidate_id)
        block = C.materialize(
            parsed,
            decision=SimpleNamespace(
                keep=False, median_relative=0.001, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=parsed.raw["factors"])
        # The receipt projector requires a held-out-bound campaign later in the
        # full workflow.  These tests exercise only retry identity admission.
        root = self.root / suffix
        root.mkdir(parents=True)
        (root / "least-commitment-capture-plan.json").write_text(
            json.dumps(raw), encoding="utf-8")
        book = J.Journal(str(root), campaign_id=control["campaign_id"])
        book.initialize()
        book.append(J.KIND_PROPOSAL_RECORDED, control)
        hypothesis_id = f"akh-retry-control-{ordinal:04d}"
        tracker = H.HypothesisTracker(
            journal_=book, root=str(root), campaign_id=control["campaign_id"])
        tracker.open_hypothesis(H.Hypothesis(
            hypothesis_id=hypothesis_id, statement=control["hypothesis"],
            falsifier="The A/A retry exceeds its predeclared noise floor.",
            origin=H.ORIGIN_CONTROLLER, author="retry-lineage-test",
            regime={"recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1"},
        ))
        authorization = tracker.authorize_claim(
            hypothesis_id, purpose="exercise preflight retry lineage",
            authorized_by="retry-lineage-test", ledger=_EmptyDoNotRepeat())
        evaluation_id = f"ake-retry-control-{ordinal:04d}"
        evaluation = _event(f"retry-{ordinal:04d}")
        evaluation.update({
            "event_id": evaluation_id, "campaign_id": control["campaign_id"],
            "candidate_id": candidate_id,
        })
        evaluation["device_state"]["source"] = "rocm-smi"
        evaluation["device_state"]["receipt_ref"] = "retry-device-state"
        candidate = _candidate(f"retry-{ordinal:04d}", status="rejected")
        candidate.update({
            "candidate_id": candidate_id, "campaign_id": control["campaign_id"],
            "proposal_id": control["proposal_id"],
            "evaluation_event_ids": [evaluation_id],
            "derived_verdicts": {"least_commitment": block},
        })
        book.append(J.KIND_EVALUATION_EVENT, evaluation)
        book.append(J.KIND_CANDIDATE_RECORDED, candidate)
        terminal = book.append(J.KIND_STOP_STATE, {
            "state": "decided", "result": {
                "state": "decided", "campaign_id": control["campaign_id"],
                "candidate_id": candidate_id, "executed": True, "ok": True,
                "spec": {
                    "recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1",
                    "least_commitment_capture_plan_sha256": raw["plan_sha256"],
                    "hypothesis": {
                        "bound": True, "hypothesis_id": hypothesis_id,
                        "authorization": authorization.to_dict(),
                    },
                    "proposal": {
                        "schema": control["schema"],
                        "proposal_id": control["proposal_id"],
                        "representation_frame_sha256": control[
                            "representation_contract"]["frame_sha256"],
                    },
                },
                "decision": {"keep": False, "median_relative": 0.001},
                "production_unchanged": {"outcome": S.PASS},
                "releases": [{"claim": "cpu", "released": True}],
                "pairs": [{"block_index": 0, "candidate": 1.0, "anchor": 1.0}],
            },
        })
        return ({
            "journal_root": str(root), "campaign_id": control["campaign_id"],
            "proposal_id": control["proposal_id"],
            "completion_event_id": terminal.event_id, "matched_control_id": None,
        }, {"proposal": control, "plan": raw})

    def _preflight_source(self, replacement_material: dict, *, suffix: str,
                          compute_started: bool = False,
                          claim_started: bool = False) -> dict:
        replacement_proposal = replacement_material["proposal"]
        replacement_plan = replacement_material["plan"]
        intervention = _capture_proposal()
        intervention.update({
            "campaign_id": f"ak-retry-source-intervention-{suffix}",
            "proposal_id": f"akp-retry-source-intervention-{suffix}",
        })
        proposal = C.make_iqk_control_proposal(
            intervention, campaign_id=f"ak-retry-source-{suffix}",
            proposal_id=f"akp-retry-source-{suffix}")
        # The generated proposals differ only in their fresh retry identities.
        # Preserve every substantive field from the replacement.
        proposal = copy.deepcopy(replacement_proposal)
        proposal["campaign_id"] = f"ak-retry-source-{suffix}"
        proposal["proposal_id"] = f"akp-retry-source-{suffix}"
        proposal["hypothesis"] = (
            f"Matched A/A control for akp-retry-source-intervention-{suffix}")
        candidate_id = f"akc-retry-source-{suffix}"
        raw = copy.deepcopy(replacement_plan)
        raw.update({
            "capture_id": f"aklc-retry-source-{suffix}",
            "campaign_id": proposal["campaign_id"],
            "proposal_id": proposal["proposal_id"],
            "candidate_id": candidate_id,
        })
        raw["factors"].pop("fresh_pairs_per_block", None)
        # Rebind prospective sources to the source proposal; their quotient
        # semantics remain byte-for-byte identical after identity stripping.
        source_path = self.root / f"{suffix}-diagnostic-source.json"
        sample = json.loads(Path(next(iter(
            raw["diagnostic_source_receipts"].values()))["path"]).read_text())
        sample["receipt_id"] = f"aklc-source-retry-{suffix}"
        sample["proposal_sha256"] = S.content_hash(proposal)
        source_path.write_text(json.dumps(sample), encoding="utf-8")
        binding = C.source_binding(source_path)
        raw["diagnostic_source_receipts"] = {
            name: dict(binding) for name in C.DIAGNOSTICS}
        raw["plan_sha256"] = C.plan_sha256(raw)
        C.from_mapping(
            raw, proposal=proposal, campaign_id=proposal["campaign_id"],
            candidate_id=candidate_id)

        root = self.root / f"source-{suffix}"
        root.mkdir()
        (root / "least-commitment-capture-plan.json").write_text(
            json.dumps(raw), encoding="utf-8")
        claim_path = Path(raw["factors"]["claim_journal_path"])
        if claim_started:
            claim_path.write_text(json.dumps({
                "record_id": "akclaim-started",
                "detail": {"receipt": {"campaign_id": proposal["campaign_id"]}},
            }) + "\n", encoding="utf-8")
        book = J.Journal(str(root), campaign_id=proposal["campaign_id"])
        book.initialize()
        book.append(J.KIND_PROPOSAL_RECORDED, proposal)
        if compute_started:
            evaluation = _event(f"retry-started-{suffix}")
            evaluation.update({
                "event_id": f"ake-retry-started-{suffix}",
                "campaign_id": proposal["campaign_id"],
                "candidate_id": candidate_id,
            })
            evaluation["device_state"]["source"] = "rocm-smi"
            evaluation["device_state"]["receipt_ref"] = "retry-started-device-state"
            book.append(J.KIND_EVALUATION_EVENT, evaluation)
        terminal = book.append(J.KIND_STOP_STATE, {
            "state": "preflight_refused", "result": {
                "state": "preflight_refused", "campaign_id": proposal["campaign_id"],
                "candidate_id": candidate_id, "executed": True, "ok": True,
                "spec": {
                    "least_commitment_capture_plan_sha256": raw["plan_sha256"],
                    "claim_journal_path": str(claim_path),
                },
                "decision": None, "pairs": [], "steps": [], "releases": [],
                "t0": None, "preflight": {"outcome": S.FAIL},
                "production_unchanged": {"outcome": S.PASS},
            },
        })
        return {
            "schema": P.CONTROL_RETRY_SCHEMA, "journal_root": str(root),
            "campaign_id": proposal["campaign_id"],
            "proposal_id": proposal["proposal_id"],
            "completion_event_id": terminal.event_id,
        }

    def test_projects_three_real_receipt_families_and_feeds_builder(self):
        output = self.root / "projected"
        result = P.project(self.plan, output)
        manifest_ref = result["archive_build_manifest"]
        manifest_path = Path(manifest_ref["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive = B.build_archive(manifest)
        report = L.evaluate_archive(archive)
        real_report_path = self.root / "ap-wm-real-report.json"
        self.assertEqual(L.main([
            str(output / "archive.json"),
            "--projection-result", str(output / "projection-result.json"),
            "--output", str(real_report_path),
        ]), 0)
        real_report = json.loads(real_report_path.read_text(encoding="utf-8"))

        self.assertEqual(result["authority"], P.AUTHORITY)
        self.assertEqual(len(result["emitted_receipts"]), 5)
        self.assertEqual(L.validate_archive(archive), [])
        self.assertEqual(result["archive_sha256"], P._content_hash(archive))
        self.assertEqual(report["authority"], L.AUTHORITY)
        self.assertEqual(real_report["evidence_label"], "real")
        self.assertEqual(real_report["matched_validation"]["status"], "PASS")
        self.assertEqual(archive["rows"][1]["matched_control_id"],
                         self.rows[0]["proposal_id"])
        receipt = json.loads((
            output / self.rows[1]["proposal_id"] / "diagnostics.json"
        ).read_text(encoding="utf-8"))
        source = receipt["source_provenance"]["diagnostics"]["information_gain"]
        self.assertEqual(source["record"], "candidate")
        self.assertEqual(len(source["record_sha256"]), 64)
        self.assertEqual(len(source["value_sha256"]), 64)

    def test_assembles_all_bindings_from_completed_campaigns(self):
        completed = [{
            key: row[key] for key in (
                "journal_root", "campaign_id", "proposal_id",
                "completion_event_id", "matched_control_id")
        } for row in self.rows]
        compiled = P.assemble_plan(
            archive_id="ak-live-binding-dress-rehearsal",
            created_at="2026-08-12T00:00:00+00:00",
            diagnostic_directions=DIAGNOSTIC_DIRECTIONS,
            outcome_weights={
                "heldout_regime_transfer": 0.5,
                "falsifier_resolution": 0.5,
            },
            completed_rows=completed,
        )
        output = self.root / "assembled-projection"
        P.project(compiled, output)
        self.assertTrue((output / "archive.json").is_file())
        archive = json.loads((output / "archive.json").read_text(encoding="utf-8"))
        self.assertEqual(archive["archive_id"], "ak-live-binding-dress-rehearsal")

    def test_cli_derives_projection_plan_without_empirical_literals(self):
        completed = [{
            key: row[key] for key in (
                "journal_root", "campaign_id", "proposal_id",
                "completion_event_id", "matched_control_id")
        } for row in self.rows]
        source = self.root / "completed.json"
        output = self.root / "projection-plan.json"
        source.write_text(json.dumps({
            "archive_id": "ak-cli-plan-test",
            "created_at": "2026-08-12T00:00:00+00:00",
            "diagnostic_directions": DIAGNOSTIC_DIRECTIONS,
            "outcome_weights": {
                "heldout_regime_transfer": 0.5,
                "falsifier_resolution": 0.5,
            },
            "rows": completed,
        }), encoding="utf-8")
        self.assertEqual(P.main([
            "--assemble-completed", str(source),
            "--plan-output", str(output),
        ]), 0)
        compiled = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(compiled["plan_sha256"], P._plan_hash(compiled))
        self.assertTrue(all(
            set(row["diagnostic_bindings"]) == set(L.DIAGNOSTICS)
            for row in compiled["rows"]))

    def test_preflight_control_retry_compiles_typed_lineage(self):
        _, material = self._retry_control(ordinal=19, suffix="retry-template")
        original = self._preflight_source(material, suffix="retry-original")
        replacement_row, _ = self._retry_control(
            ordinal=20, suffix="retry-replacement")
        evidence = P._load_completed_evidence(replacement_row)
        receipt = P._compile_control_retry_lineage(original, evidence)
        self.assertEqual(receipt["schema"], P.CONTROL_RETRY_SCHEMA)
        self.assertEqual(receipt["original"]["proposal_id"], original["proposal_id"])
        self.assertEqual(
            receipt["replacement"]["proposal_id"], replacement_row["proposal_id"])
        self.assertEqual(receipt["claim_journal_observation"][
            "matching_claim_record_ids"], [])
        self.assertEqual(P._validate_control_retry_lineage(receipt, evidence), receipt)

    def test_retry_lineage_refuses_semantic_mismatch(self):
        source_row, source_material = self._retry_control(
            ordinal=21, suffix="retry-source-material")
        del source_row  # only its valid prospective material is needed
        original = self._preflight_source(
            source_material, suffix="retry-mismatch-original")
        replacement_row, _ = self._retry_control(
            ordinal=22, suffix="retry-mismatch-replacement",
            proposal_mutation=lambda value: value["target"]["shapes"].append("pp1024"))
        evidence = P._load_completed_evidence(replacement_row)
        with self.assertRaisesRegex(P.ReceiptProjectionError, "changes source/frame/factor"):
            P._compile_control_retry_lineage(original, evidence)

    def test_retry_lineage_refuses_tampered_receipt(self):
        _, material = self._retry_control(
            ordinal=22, suffix="retry-tamper-template")
        original = self._preflight_source(material, suffix="retry-tamper-original")
        replacement_row, _ = self._retry_control(
            ordinal=23, suffix="retry-tamper-replacement")
        evidence = P._load_completed_evidence(replacement_row)
        receipt = P._compile_control_retry_lineage(original, evidence)
        receipt["semantic_contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(P.ReceiptProjectionError, "stale or tampered"):
            P._validate_control_retry_lineage(receipt, evidence)

    def test_retry_lineage_refuses_compute_started_original(self):
        _, material = self._retry_control(
            ordinal=23, suffix="retry-compute-template")
        original = self._preflight_source(
            material, suffix="retry-compute-original", compute_started=True)
        replacement_row, _ = self._retry_control(
            ordinal=24, suffix="retry-compute-replacement")
        evidence = P._load_completed_evidence(replacement_row)
        with self.assertRaisesRegex(P.ReceiptProjectionError, "compute or claim-boundary"):
            P._compile_control_retry_lineage(original, evidence)

    def test_retry_lineage_refuses_claim_started_original(self):
        _, material = self._retry_control(
            ordinal=24, suffix="retry-claim-template")
        original = self._preflight_source(
            material, suffix="retry-claim-original", claim_started=True)
        replacement_row, _ = self._retry_control(
            ordinal=25, suffix="retry-claim-replacement")
        evidence = P._load_completed_evidence(replacement_row)
        with self.assertRaisesRegex(P.ReceiptProjectionError, "acquired a resource claim"):
            P._compile_control_retry_lineage(original, evidence)

    def test_retry_join_resolves_predeclared_original_to_completed_replacement(self):
        lineage = {"original": {"proposal_id": "akp-control-original"}}
        P._require_planned_control_join(
            intervention_block={
                "role": "intervention",
                "matched_control_proposal_id": "akp-control-original"},
            control_block={
                "role": "control", "matched_control_proposal_id": None},
            intervention_proposal_id="akp-intervention",
            control_proposal_id="akp-control-replacement",
            retry_lineage=lineage)
        with self.assertRaisesRegex(P.ReceiptProjectionError, "planned control"):
            P._require_planned_control_join(
                intervention_block={
                    "role": "intervention",
                    "matched_control_proposal_id": "akp-unrelated"},
                control_block={
                    "role": "control", "matched_control_proposal_id": None},
                intervention_proposal_id="akp-intervention",
                control_proposal_id="akp-control-replacement",
                retry_lineage=lineage)

    def test_missing_journaled_diagnostic_fails_closed(self):
        self.plan["rows"][1]["diagnostic_bindings"]["novelty"]["pointer"] += "-absent"
        self._rehash()
        with self.assertRaisesRegex(P.ReceiptProjectionError, "absent"):
            P.project(self.plan, self.root / "missing")

    def test_binding_record_hash_mismatch_fails_closed(self):
        self.plan["rows"][1]["diagnostic_bindings"]["novelty"][
            "record_sha256"] = "0" * 64
        self._rehash()
        with self.assertRaisesRegex(P.ReceiptProjectionError, "record_sha256"):
            P.project(self.plan, self.root / "record-hash")

    def test_two_rows_from_one_campaign_are_not_two_clean_campaigns(self):
        self.plan["rows"][1] = copy.deepcopy(self.plan["rows"][0])
        self.plan["rows"][1]["proposal_id"] = self.plan["rows"][0]["proposal_id"]
        self._rehash()
        with self.assertRaisesRegex(P.ReceiptProjectionError, "distinct clean"):
            P.project(self.plan, self.root / "one-campaign")

    def test_two_changed_factors_fail_closed(self):
        source_sha = self.plan["rows"][1]["factor_bindings"]["threads"][
            "record_sha256"]
        self.plan["rows"][1]["factor_bindings"]["threads"] = _binding(
            "/derived_verdicts/least_commitment/outcome/heldout_regime_transfer",
            source_sha)
        self._rehash()
        with self.assertRaisesRegex(P.ReceiptProjectionError, "changes 2 factors"):
            P.project(self.plan, self.root / "two-factor")

    def test_copied_control_diagnostic_semantics_fail_closed(self):
        control = self._completed_campaign(
            ordinal=3, campaign_suffix="copied-control", arm="0", score=0.5,
            outcome=0.0, semantic_label="copied")
        intervention = self._completed_campaign(
            ordinal=4, campaign_suffix="copied-intervention", arm="1", score=0.5,
            outcome=0.2, semantic_label="copied")
        intervention["matched_control_id"] = control["proposal_id"]
        self.plan["rows"] = [control, intervention]
        self.plan["candidate_frame_id_binding"] = copy.deepcopy(
            control["candidate_frame_id_binding"])
        self._rehash()
        with self.assertRaisesRegex(P.ReceiptProjectionError, "semantics equal"):
            P.project(self.plan, self.root / "copied-semantics-output")

    def test_non_real_marker_anywhere_in_source_record_fails_closed(self):
        # Add an explicit marker in the terminal result without modifying any
        # requested binding.  Provenance eligibility is record-wide.
        row = self.plan["rows"][1]
        book = J.Journal(row["journal_root"], campaign_id=row["campaign_id"])
        entries = book.read_all()
        terminal = next(entry for entry in entries
                        if entry.event_id == row["completion_event_id"])
        # Journal bytes are immutable, so create a separate campaign whose
        # terminal event is explicitly dry-run and assert clean admission stops.
        payload = copy.deepcopy(terminal.payload)
        payload["result"]["dry_run"] = True
        synthetic_root = self.root / "synthetic"
        synthetic = J.Journal(str(synthetic_root), campaign_id=row["campaign_id"])
        synthetic.initialize()
        shutil.copy2(
            Path(row["journal_root"]) / H.LEDGER_FILENAME,
            synthetic_root / H.LEDGER_FILENAME,
        )
        for entry in entries[:-1]:
            synthetic.append(entry.kind, entry.payload, record_id=entry.record_id)
        replacement = synthetic.append(J.KIND_STOP_STATE, payload)
        row["journal_root"] = str(synthetic_root)
        row["completion_event_id"] = replacement.event_id
        self._rehash()
        with self.assertRaisesRegex(P.ReceiptProjectionError, "non-real marker"):
            P.project(self.plan, self.root / "non-real")

    def test_completed_journal_with_torn_tail_fails_closed(self):
        journal_path = Path(self.plan["rows"][1]["journal_root"]) / "events.jsonl"
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write('{"unacknowledged":')
        with self.assertRaisesRegex(P.ReceiptProjectionError, "torn tail"):
            P.project(self.plan, self.root / "torn-tail")

    def test_plan_cannot_supply_empirical_literals(self):
        self.plan["rows"][0]["diagnostic_bindings"]["novelty"] = {
            "record": "candidate", "pointer": "/does-not-matter",
            "record_sha256": "a" * 64, "value": 99,
        }
        self._rehash()
        with self.assertRaisesRegex(P.ReceiptProjectionError, "binding fields"):
            P.project(self.plan, self.root / "literal")

    def test_builder_refusal_publishes_no_partial_directory(self):
        output = self.root / "refused"
        with mock.patch.object(B, "build_archive", side_effect=ValueError("refused")):
            with self.assertRaisesRegex(ValueError, "refused"):
                P.project(self.plan, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".refused.staging-*")), [])

    def test_relative_output_directory_is_refused_before_staging(self):
        with self.assertRaisesRegex(P.ReceiptProjectionError, "must be absolute"):
            P.project(self.plan, Path("relative-output"))


if __name__ == "__main__":
    unittest.main()
