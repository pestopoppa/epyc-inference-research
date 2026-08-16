#!/usr/bin/env python3
"""Acceptance gate for the 2026-08-16 planner-conditioning preload.

This test is intentionally independent of the portfolio loader.  It reads the
sealed JSON product directly so a loader/schema regression cannot hide a
conditioning regression.  The required tests cover portfolio content only.
The final class records cross-cutting controller enforcement as expected
missing; if that product scope lands, its unexpected successes force the
expectation to be promoted into the required gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PORTFOLIO_PATH = HERE / "discovery_hypothesis_portfolio_v2.json"

EXPECTED_ELIGIBLE_IDS = {
    "akh-v2-fa-gqa7-pair-tail",
    "akh-v2-q5-type-specific-dequant",
    "akh-v2-q8-quantizer-new-mechanism",
    "akh-v2-rms-direct-load-reduction",
}

LADDER_EVIDENCE = {
    "ev-quant-ladder-occupancy-knee-20260816": (
        "scripts/kernel_rnd/autokernel/evidence/quant-ladder-20260816/"
        "a10_quant_ladder_occupancy_knee_20260816.md",
        "eec0b1733d4bf5c684f537cf7c28c442f938b17bf3ea6ebfeee85a86c5ccab78",
    ),
    "ev-quant-ladder-tg128-raw-20260816": (
        "scripts/kernel_rnd/autokernel/evidence/quant-ladder-20260816/"
        "ladder-results-20260816.jsonl",
        "7dbdf4dc10b9f2cd4f90c7a2d00bd778d9c67cfca1e2238771b4685744aec3a7",
    ),
    "ev-quant-ladder-np-raw-20260816": (
        "scripts/kernel_rnd/autokernel/evidence/quant-ladder-20260816/"
        "np_sweep-20260816.log",
        "8f19b1c54ce1ebb7c8d95e5948488411f9c1f11468b68a8c2c07618fa6edf3bb",
    ),
    "ev-quant-ladder-manifest-20260815": (
        "scripts/kernel_rnd/autokernel/evidence/quant-ladder-20260816/"
        "a10_quant_ladder_MANIFEST_20260815.md",
        "bcefb69acf85ec73e6b3a0300be38fd6055671e325338f6d6c1658ba37193722",
    ),
    "ev-iq2-vgpr-static-20260812": (
        "scripts/kernel_rnd/autokernel/evidence/quant-ladder-20260816/"
        "a10_iq2_vgpr_lever_20260812.md",
        "7ec3e4340b243c6f7afdb0f9c5cea6b95227351c88cab2a03ecefbcc8183cd46",
    ),
}

Q4K_MMQ_RECEIPTS = {
    "ev-q4k-mmq-stock-correctness-v9":
        "6e7870558c6317a7ef12f7fca8a267466cc0734295d0c33562103a0c022a3502",
    "ev-q4k-mmq-dp4a-negative-v9":
        "c955f5df4cd584586b0e59746b8dc1e2542fbde57a558de0257ba616dee78021",
    "ev-q4k-mmq-ls-scale-negative-v9":
        "476e6fb3fc004b30067ea9de4826a094a90db529c610670673c159e803342d89",
    "ev-q4k-mmq-qsum-fix-diagnostic-v9":
        "355bdcf169cb8682d2f56e1754b321f770a0fe3c0bbc5f6e1dc58eaffb443fb2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _carrier(path: str) -> Path:
    if path.startswith("repo://"):
        return REPO / path.removeprefix("repo://")
    return Path(path)


def _blob(value: object) -> str:
    return json.dumps(value, sort_keys=True).lower()


class PlannerConditioningAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.body = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
        cls.evidence = {row["evidence_id"]: row for row in cls.body["evidence"]}
        cls.hypotheses = {
            row["hypothesis_id"]: row for row in cls.body["hypotheses"]
        }
        cls.dnr = {row["dnr_id"]: row for row in cls.body["do_not_repeat"]}

    def _hypothesis(self, hypothesis_id: str) -> dict:
        if hypothesis_id not in self.hypotheses:
            self.fail(f"missing required hypothesis: {hypothesis_id}")
        return self.hypotheses[hypothesis_id]

    def _evidence(self, evidence_id: str) -> dict:
        if evidence_id not in self.evidence:
            self.fail(f"missing required evidence: {evidence_id}")
        return self.evidence[evidence_id]

    def _assert_words(self, value: object, *words: str) -> None:
        text = _blob(value)
        for word in words:
            self.assertIn(word.lower(), text)

    def _assert_immutable_carrier(self, row: dict, expected_sha256: str) -> None:
        self.assertEqual(expected_sha256, row["sha256"])
        path = _carrier(row["path"])
        self.assertTrue(path.is_file(), f"missing evidence carrier: {path}")
        self.assertFalse(path.is_symlink(), f"symlink evidence carrier: {path}")
        self.assertEqual(expected_sha256, _sha256(path))

    def test_exact_four_eligible_hypotheses_are_unchanged(self) -> None:
        eligible = {
            row["hypothesis_id"] for row in self.body["hypotheses"]
            if row["current_bundle_eligibility"]["eligible"] is True
        }
        self.assertEqual(EXPECTED_ELIGIBLE_IDS, eligible)

    def test_ladder_evidence_is_immutable_and_design_prior_only(self) -> None:
        for evidence_id, (relative, digest) in LADDER_EVIDENCE.items():
            row = self._evidence(evidence_id)
            self.assertEqual(f"repo://{relative}", row["path"])
            self.assertEqual("non_governed_design_prior", row["authority"])
            self.assertEqual("current_v9", row["temporal_status"])
            self._assert_immutable_carrier(row, digest)

        for hypothesis_id in (
            "akh-v2-lowbit-type-specialized-mmvq",
            "akh-v2-quant-ladder-batched-wave-slot-residual",
            "akh-v2-iq1s-occupancy-discriminator",
            "akh-v2-batching-closes-all-lowbit-gaps",
        ):
            row = self._hypothesis(hypothesis_id)
            self.assertEqual("design_prior", row["epistemic"]["grade"])
            self.assertFalse(row["current_bundle_eligibility"]["eligible"])
            self._assert_words(
                row["epistemic"]["limitations"],
                "no a/a band" if hypothesis_id !=
                "akh-v2-iq1s-occupancy-discriminator" else "no iq1 throughput",
            )

        claims = self._evidence(
            "ev-quant-ladder-occupancy-knee-20260816")["claims"]
        self._assert_words(
            claims, "design-prior", "not acceptance thresholds", "non-governed",
            "no promotion authority", "suspected outlier",
        )

    def test_iq2_requires_exact_zero_spill_64_vgpr_source_mechanism(self) -> None:
        row = self._hypothesis("akh-v2-lowbit-type-specialized-mmvq")
        self.assertEqual(
            "iq2_xxs_vperm_zero_spill_occupancy_threshold",
            row["mechanism"]["facets"]["mechanism"],
        )
        self.assertEqual(
            ["vec_dot_iq2_xxs_q8_1"], row["target"]["source_symbols"])
        self._assert_words(
            row,
            "at most 64", "scratch=0", "vgpr_spill_count=0", "eight waves",
            "v_perm_b32", "65 to 70 vgpr", "no predicted occupancy payoff",
        )
        self.assertIn("ev-iq2-vgpr-static-20260812", row["evidence_refs"])
        self._assert_words(
            self._evidence("ev-iq2-vgpr-static-20260812")["claims"],
            "iq4_xs at 64", "iq2_xxs at 78", "scratch zero",
            "vgpr_spill_count zero", "437-instruction", "sign expansion",
        )

    def test_iq1_discriminator_is_inactive_and_cannot_spend(self) -> None:
        row = self._hypothesis("akh-v2-iq1s-occupancy-discriminator")
        self.assertEqual("needs-template", row["status"])
        self.assertEqual([], row["current_bundle_eligibility"]["template_ids"])
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        self.assertEqual([0.0, 0.0], row["expected_value"][
            "expected_relative_gain_pct_range"])
        self.assertEqual(0.0, row["expected_value"]["device_time_ceiling_pct"])
        self._assert_words(
            row,
            "memory-only", "explicit operator", "practical iq1 serving relevance",
            "do not spend gpu or source-authoring budget", "none; retain as inactive",
        )

    def test_blanket_batching_warning_is_scoped_and_not_a_hard_dnr(self) -> None:
        hypothesis_id = "akh-v2-batching-closes-all-lowbit-gaps"
        row = self._hypothesis(hypothesis_id)
        self.assertEqual("retired", row["status"])
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        self.assertNotIn(hypothesis_id, self.dnr)
        self.assertFalse(any(
            item["mechanism"]["facets"].get("mechanism") ==
            "blanket_batching_closes_all_lowbit_dequant_gaps"
            for item in self.body["do_not_repeat"]
        ))
        self._assert_words(
            row,
            "mi210", "goedel-8b", "b1-through-b32", "design-prior",
            "not a hard dnr", "does not forbid measuring batching",
            "narrower per-format", "permit reentry",
        )

    def test_q4k_branchless_remains_control_flow_not_occupancy_reframing(self) -> None:
        row = self._hypothesis("akh-v2-q4k-branchless-scale-min")
        self.assertEqual("branchless_scale_min_control_flow",
                         row["mechanism"]["facets"]["mechanism"])
        self.assertEqual("control_flow", row["mechanism"]["facets"]["change_class"])
        self.assertEqual(
            "3657b770d508453ee9dfda3bbd9bb4f6535ab11196b6c1759c3771af259aac5b",
            row["mechanism"]["fingerprint_sha256"],
        )
        self.assertEqual("dirty_diagnostic", row["epistemic"]["grade"])
        self.assertEqual(0.4115, row["expected_value"][
            "current_bundle_plausible_gain_ceiling_pct"])
        conditioning = _blob({
            "statement": row["statement"],
            "implementation": row["implementation"],
            "stop_rule": row["stop_rule"],
            "portability": row["portability"],
        })
        for forbidden in ("64-vgpr", "vgpr_spill_count", "eight waves"):
            self.assertNotIn(forbidden, conditioning)

    def test_q4k_mmq_is_blocked_on_exact_correctness_receipts(self) -> None:
        for evidence_id, digest in Q4K_MMQ_RECEIPTS.items():
            row = self._evidence(evidence_id)
            self.assertEqual("candidate_only", row["authority"])
            self._assert_immutable_carrier(row, digest)

        row = self._hypothesis("akh-v2-q4k-mmq-dequant-gemv")
        self.assertTrue(set(Q4K_MMQ_RECEIPTS).issubset(row["evidence_refs"]))
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        self._assert_words(
            row["current_bundle_eligibility"]["blocking_conditions"],
            "25/43", "frozen v9", "clean", "source identity",
        )
        self._assert_words(
            row["portability"]["required_validation"],
            "172/172", "1.5", "clean", "source identity",
        )
        self._assert_words(
            row["stop_rule"], "refuse", "performance ranking", "correctness",
        )
        self._assert_words(
            self._evidence("ev-q4k-mmq-qsum-fix-diagnostic-v9"),
            "uncommitted", "diagnostic", "172/172",
        )

    def test_q8_integer_native_fact_has_exact_dnr(self) -> None:
        evidence_id = "ev-q8-mmvq-int8-native-source-v9"
        evidence = self._evidence(evidence_id)
        self.assertEqual("governance_snapshot", evidence["authority"])
        self._assert_immutable_carrier(
            evidence,
            "1e8768a89815cc6c8cf5277ddc437ac9d2a5353597478c68d23bd79646dd0d91",
        )
        self._assert_words(
            evidence["claims"],
            "integer-native", "ggml_cuda_dp4a", "one", "scale", "32",
            "no per-element", "fp dequant",
        )

        dnr_id = "dnr-q8-fp-dequant-port-into-mmvq"
        self.assertIn(dnr_id, self.dnr)
        row = self.dnr[dnr_id]
        self.assertEqual("hard_refusal_exact_mechanism_and_regime", row["enforcement"])
        self.assertEqual("prior_art_already_present", row["classification"])
        self.assertEqual(
            "q8_0_per_element_fp_dequant_fusion_port",
            row["mechanism"]["facets"]["mechanism"],
        )
        self.assertEqual(
            "cdfa933eb91641e6397aef957a98d47fd6ce5c9d72f26964385a48f867c83935",
            row["mechanism"]["fingerprint_sha256"],
        )
        self.assertEqual([evidence_id], row["evidence_refs"])
        self._assert_words(row["reentry_conditions"], "integer-native", "distinct")

    def test_live_iq_residency_is_a_latent_blocker_and_reprice_tripwire(self) -> None:
        evidence_id = "ev-iq-live-residency-tripwire-20260815"
        evidence = self._evidence(evidence_id)
        self.assertEqual("governance_snapshot", evidence["authority"])
        self._assert_immutable_carrier(
            evidence,
            "1e8768a89815cc6c8cf5277ddc437ac9d2a5353597478c68d23bd79646dd0d91",
        )
        self._assert_words(
            evidence["claims"], "no iq-format role", "rocm0", "serving-resident",
            "reopen", "reprice", "registry",
        )
        row = self._hypothesis("akh-v2-lowbit-type-specialized-mmvq")
        self.assertIn(evidence_id, row["evidence_refs"])
        self._assert_words(
            row["current_bundle_eligibility"]["blocking_conditions"],
            "no iq-format", "serving", "rocm0", "resident",
        )
        self._assert_words(
            row, "latent", "registry", "reprice", "live", "diagnostic",
        )

    def test_method_traps_are_explicit_evidence_limitations(self) -> None:
        required = {
            "ev-iq2-exact-instantiation-confound-v9": (
                "63", "8 waves", "78", "6 waves", "synthetic", "production",
            ),
            "ev-iq2-decode-phase-confound-v9": (
                "-n 0", "prefill-only", "decode", "non-governed",
            ),
            "ev-static-tool-silence-confound-v9": (
                "llvm-nm", "does not exist", "stderr", "llvm-readelf -s",
                "negative",
            ),
        }
        for evidence_id, words in required.items():
            row = self._evidence(evidence_id)
            self.assertEqual("governance_snapshot", row["authority"])
            self._assert_immutable_carrier(row, row["sha256"])
            self._assert_words(row["claims"], *words)

        lowbit = self._hypothesis("akh-v2-lowbit-type-specialized-mmvq")
        self.assertTrue(set(required).issubset(lowbit["evidence_refs"]))
        self._assert_words(
            lowbit["epistemic"]["limitations"],
            "exact", "instantiation", "phase", "command", "tool", "execution",
        )


class CrossCuttingMachineEnforcementExpectedMissing(unittest.TestCase):
    """Aspirational machine gates intentionally outside the required preload."""

    POLICY_PATH = HERE / "discovery_admissibility_policy_v1.json"
    REQUIRED_POLICY_IDS = {
        "ak-admit-command-phase-consistency-v1",
        "ak-admit-exact-kernel-instantiation-v1",
        "ak-admit-q4k-mmq-baseline-correctness-v1",
        "ak-admit-q8-native-source-premise-v1",
        "ak-admit-static-tool-execution-v1",
        "ak-trigger-iq-serving-residency-v1",
    }

    @unittest.expectedFailure
    def test_typed_policy_manifest_is_present_and_complete(self) -> None:
        body = json.loads(self.POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual("epyc.autokernel.admissibility_policy.v1", body["schema"])
        self.assertEqual(
            self.REQUIRED_POLICY_IDS,
            {row["policy_id"] for row in body["admissibility_policies"]},
        )

    @unittest.expectedFailure
    def test_policy_manifest_is_bound_into_planner_and_critic_context(self) -> None:
        factory = (HERE / "controller" / "discovery_deployment_factory.py").read_text(
            encoding="utf-8")
        deployment = (HERE / "controller" / "discovery_deployment.py").read_text(
            encoding="utf-8")
        self.assertIn("admissibility_policies", factory)
        self.assertIn("admissibility_policies", deployment)
        self.assertIn("admissibility_policy_sha256", factory)
        self.assertIn("admissibility_policy_sha256", deployment)

    @unittest.expectedFailure
    def test_controller_and_proofs_enforce_policy_before_model_judgment(self) -> None:
        controller = (HERE / "controller" / "discovery_controller.py").read_text(
            encoding="utf-8")
        producer = (HERE / "controller" / "gpu_source_evidence.py").read_text(
            encoding="utf-8")
        proofs = (HERE / "controller" / "gpu_source_proofs.py").read_text(
            encoding="utf-8")
        combined = controller + producer + proofs
        for policy_id in self.REQUIRED_POLICY_IDS:
            self.assertIn(policy_id, combined)
        self.assertIn("refuse", combined.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
