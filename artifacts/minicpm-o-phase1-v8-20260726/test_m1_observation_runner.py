import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).with_name("m1_observation_runner.py")
SPEC = importlib.util.spec_from_file_location("m1", MODULE)
assert SPEC and SPEC.loader
m1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m1)


class M1ObservationRunnerTests(unittest.TestCase):
    @staticmethod
    def provenance(case_id, content):
        return {
            "case_id": case_id,
            "raw_content": content,
            "model_sha256": "a" * 64,
            "mmproj_sha256": "b" * 64,
            "binary_sha256": "c" * 64,
            "endpoint_or_sidecar": "http://127.0.0.1:19001",
            "started_at": "2026-07-26T12:00:00Z",
            "finished_at": "2026-07-26T12:00:01Z",
            "request_parameters": {"temperature": 0, "seed": 35, "max_tokens": 32},
        }

    def test_exact_score_and_accepted_alternative(self):
        self.assertTrue(m1.score_response(" Sofia  Vergara ", ["Sofia Vergara"])["pass"])
        self.assertTrue(m1.score_response("63,086", ["63086", "63,086"])["pass"])

    def test_score_rejects_explanation_and_substring_false_positive(self):
        self.assertFalse(m1.score_response("The answer is 4", ["4"])["pass"])
        self.assertFalse(m1.score_response("14", ["4"])["pass"])

    def test_manifest_is_complete_and_roles_are_disjoint(self):
        worker = m1.manifest_for_role("worker_vision")
        escalation = m1.manifest_for_role("vision_escalation")
        worker_ids = {item["case_id"] for item in worker["fixtures"]}
        escalation_ids = {item["case_id"] for item in escalation["fixtures"]}
        self.assertEqual(len(worker_ids), 8)
        self.assertEqual(len(escalation_ids), 10)
        self.assertFalse(worker_ids & escalation_ids)
        self.assertTrue(all(len(item["image_sha256"]) == 64 for item in worker["fixtures"] + escalation["fixtures"]))

    def test_source_suite_parity(self):
        self.assertIsNone(m1.assert_source_parity())

    def test_score_saved_responses_requires_exact_fixture_set(self):
        manifest = m1.manifest_for_role("worker_vision")
        rows = [self.provenance(item["case_id"], item["accepted_answers"][0]) for item in manifest["fixtures"]]
        self.assertEqual(m1.score_saved_responses(manifest, rows, "d" * 64, "qwen25-worker")["passed"], 8)
        with self.assertRaises(ValueError):
            m1.score_saved_responses(manifest, rows[:-1], "d" * 64, "qwen25-worker")

    def test_provenance_and_request_contract_are_required(self):
        manifest = m1.manifest_for_role("worker_vision")
        fixture = manifest["fixtures"][0]
        row = self.provenance(fixture["case_id"], fixture["accepted_answers"][0])
        del row["binary_sha256"]
        with self.assertRaises(ValueError):
            m1.score_saved_responses(manifest, [row] * len(manifest["fixtures"]), "d" * 64, "qwen25-worker")
        rows = [self.provenance(item["case_id"], item["accepted_answers"][0]) for item in manifest["fixtures"]]
        rows[0]["request_parameters"]["seed"] = 99
        with self.assertRaises(ValueError):
            m1.score_saved_responses(manifest, rows, "d" * 64, "qwen25-worker")
        rows = [self.provenance(item["case_id"], item["accepted_answers"][0]) for item in manifest["fixtures"]]
        rows[0]["model_sha256"] = "not-a-sha"
        with self.assertRaises(ValueError):
            m1.score_saved_responses(manifest, rows, "d" * 64, "qwen25-worker")
        rows = [self.provenance(item["case_id"], item["accepted_answers"][0]) for item in manifest["fixtures"]]
        rows[0]["started_at"] = "2026-07-26T12:00:00"
        with self.assertRaises(ValueError):
            m1.score_saved_responses(manifest, rows, "d" * 64, "qwen25-worker")
        rows = [self.provenance(item["case_id"], item["accepted_answers"][0]) for item in manifest["fixtures"]]
        rows[1]["model_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            m1.score_saved_responses(manifest, rows, "d" * 64, "qwen25-worker")

    def test_exact_mcnemar_and_pairing(self):
        self.assertEqual(m1.mcnemar_exact(0, 0), 1.0)
        provenance = {"model_sha256": "a" * 64, "mmproj_sha256": "b" * 64, "binary_sha256": "c" * 64, "endpoint_or_sidecar": "sidecar"}
        def scored(arm_id, outcomes):
            rows = [{"case_id": key, "score": {"pass": outcome}, "provenance": {"case_id": key, **provenance}} for key, outcome in outcomes.items()]
            return {"schema": m1.SCHEMA + ".scored-responses.v1", "role": "worker_vision", "protocol_status": "observation_only_unratified", "manifest_sha256": "d" * 64, "arm_id": arm_id, "arm_provenance": provenance, "total": len(rows), "passed": sum(outcomes.values()), "rows": rows}
        baseline = scored("qwen", {"a": True, "b": True, "c": False})
        candidate = scored("minicpm", {"a": True, "b": False, "c": True})
        result = m1.paired_analysis(baseline, candidate)
        self.assertEqual(result["paired_2x2"], {"both_pass": 1, "baseline_only": 1, "candidate_only": 1, "neither": 0})
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 1.0)

    def test_pairing_rejects_different_manifest_or_same_arm(self):
        provenance = {"model_sha256": "a" * 64, "mmproj_sha256": "b" * 64, "binary_sha256": "c" * 64, "endpoint_or_sidecar": "sidecar"}
        scored = {"schema": m1.SCHEMA + ".scored-responses.v1", "role": "worker_vision", "protocol_status": "observation_only_unratified", "manifest_sha256": "d" * 64, "arm_id": "same", "arm_provenance": provenance, "total": 1, "passed": 1, "rows": [{"case_id": "a", "score": {"pass": True}, "provenance": {"case_id": "a", **provenance}}]}
        with self.assertRaises(ValueError):
            m1.paired_analysis(scored, dict(scored))
        other = dict(scored, arm_id="other", manifest_sha256="e" * 64)
        with self.assertRaises(ValueError):
            m1.paired_analysis(scored, other)

    def test_pairing_rejects_duplicate_rows_and_inconsistent_totals(self):
        provenance = {"model_sha256": "a" * 64, "mmproj_sha256": "b" * 64, "binary_sha256": "c" * 64, "endpoint_or_sidecar": "sidecar"}
        def scored(arm_id):
            rows = [{"case_id": "a", "score": {"pass": True}, "provenance": {"case_id": "a", **provenance}}]
            return {"schema": m1.SCHEMA + ".scored-responses.v1", "role": "worker_vision", "protocol_status": "observation_only_unratified", "manifest_sha256": "d" * 64, "arm_id": arm_id, "arm_provenance": provenance, "total": 1, "passed": 1, "rows": rows}
        baseline, candidate = scored("qwen"), scored("minicpm")
        duplicate = dict(candidate, rows=candidate["rows"] * 2, total=2, passed=2)
        with self.assertRaises(ValueError):
            m1.paired_analysis(baseline, duplicate)
        inconsistent = dict(candidate, total=2)
        with self.assertRaises(ValueError):
            m1.paired_analysis(baseline, inconsistent)

    def test_atomic_verify_and_cli_rejects_mixed_operations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.json"
            m1.atomic_or_verify_json(path, {"a": 1})
            m1.atomic_or_verify_json(path, {"a": 1})
            with self.assertRaises(RuntimeError):
                m1.atomic_or_verify_json(path, {"a": 2})
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    m1.main(["--write-manifests", temp, "--manifest", str(path)])


if __name__ == "__main__":
    unittest.main()
