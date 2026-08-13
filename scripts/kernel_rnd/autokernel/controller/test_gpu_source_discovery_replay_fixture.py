"""Hardware-free acceptance replay for GPU source discovery decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "gpu_source_discovery_replay_v1.json"
)
PROOF_CONTRACT = FIXTURE.with_name("gpu_source_proof_contract_v1.json")
LAUNCH_GATE = FIXTURE.with_name("discovery_controller_launch_gate_v1.json")


def _screen_rows(trace: dict) -> list[dict]:
    return [row for row in trace["evidence"] if row["role"].startswith("screen_s")]


def replay_classification(trace: dict) -> str:
    """Apply only the discovery policy encoded by the replay contract."""
    screens = _screen_rows(trace)
    if not screens:
        raise ValueError("a replay trace must contain a screen")
    effects = [row["median_relative"] for row in screens]
    if len(effects) == 1:
        return "screened_out" if effects[0] < 0 else "inconclusive"
    if effects[0] * effects[1] <= 0:
        return "inconclusive"
    if all(effect > 0 for effect in effects):
        if "component_pooled_median_relative" in trace:
            component_floor = max(trace["component_pooled_median_relative"].values())
            if trace["pooled"]["median_relative"] < component_floor:
                return "replicated_but_subadditive"
        return "top_k_replicated_candidate"
    return "screened_out"


class ReplayFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_discovery_authority_and_stage_order_are_frozen(self) -> None:
        value = self.fixture
        self.assertEqual(value["schema"], "epyc.autokernel.discovery_replay_fixture.v1")
        self.assertEqual(
            value["authority"],
            "offline_replay_only_no_inference_or_promotion_authority",
        )
        self.assertEqual(
            value["invariants"]["stage_order"],
            [
                "planned",
                "critic_scope",
                "patched",
                "built",
                "correctness",
                "attribution",
                "screen_s1",
                "optional_screen_s2",
                "classified",
                "feedback",
            ],
        )
        self.assertTrue(value["invariants"]["candidate_only"])
        self.assertTrue(value["invariants"]["non_promotable"])
        self.assertFalse(value["invariants"]["promotion_claim"])

    def test_six_manual_transitions_replay_to_recorded_decisions(self) -> None:
        traces = self.fixture["traces"]
        self.assertEqual(len(traces), 6)
        for trace in traces:
            with self.subTest(trace=trace["trace_id"]):
                self.assertEqual(
                    replay_classification(trace),
                    trace["expected_classification"],
                )

    def test_screen_evidence_is_hash_bound_to_preserved_receipts_when_present(self) -> None:
        for trace in self.fixture["traces"]:
            for row in _screen_rows(trace):
                with self.subTest(trace=trace["trace_id"], role=row["role"]):
                    self.assertRegex(row["file_sha256"], r"^[0-9a-f]{64}$")
                    self.assertRegex(row["result_sha256"], r"^[0-9a-f]{64}$")
                    path = Path(row["path"])
                    if not path.is_file():
                        continue
                    self.assertFalse(path.is_symlink())
                    self.assertEqual(
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        row["file_sha256"],
                    )
                    receipt = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(receipt["result_sha256"], row["result_sha256"])
                    self.assertEqual(receipt["median_relative"], row["median_relative"])
                    self.assertFalse(receipt["promotion_claim"])

    def test_invalid_receipts_are_preserved_but_never_authoritative(self) -> None:
        invalid = []
        for trace in self.fixture["traces"]:
            for row in trace["evidence"]:
                if row.get("usable") is False or row.get("usable_for_exact_gate") is False:
                    invalid.append((trace["trace_id"], row))
        self.assertGreaterEqual(len(invalid), 4)
        for trace_id, row in invalid:
            with self.subTest(trace=trace_id, role=row["role"]):
                self.assertNotIn(row["role"], {"screen_s1", "screen_s2"})
                self.assertIn(
                    False,
                    (row.get("usable"), row.get("usable_for_exact_gate")),
                )

    def test_proof_contract_requires_two_arm_provenance_and_native_hashes(self) -> None:
        contract = json.loads(PROOF_CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(
            contract["schema"],
            "epyc.autokernel.gpu_source_proof_contract.v1",
        )
        self.assertTrue(contract["native_hash_rule"]["file_sha256_is_distinct"])
        self.assertIn(
            "anchor_build_identity",
            contract["proof_bundle"]["required"],
        )
        self.assertIn(
            "candidate_build_identity",
            contract["proof_bundle"]["required"],
        )
        self.assertIn(
            "candidate_attribution_receipt_sha256",
            contract["proof_bundle"]["required"],
        )
        self.assertIn(
            "anchor_attribution_receipt_sha256",
            contract["proof_bundle"]["required"],
        )
        self.assertIn(
            "caller-created dictionaries as proof",
            contract["forbidden_shortcuts"],
        )

    def test_launch_gate_names_every_required_black_box_failure_mode(self) -> None:
        gate = json.loads(LAUNCH_GATE.read_text(encoding="utf-8"))
        self.assertEqual(
            gate["schema"],
            "epyc.autokernel.discovery_controller_launch_gate.v1",
        )
        self.assertEqual(
            {case["id"] for case in gate["required_cases"]},
            {
                "pending_exact_resume",
                "crash_before_runner",
                "crash_after_runner",
                "nomination_exactly_once",
                "dnr_measured_outcomes",
                "pooled_classifier",
                "producer_tamper_matrix",
                "concrete_adapter_factory",
            },
        )
        self.assertEqual(
            gate["gate"],
            "all_required_cases_pass_before_any_live_adapter_is_accepted",
        )


if __name__ == "__main__":
    unittest.main()
