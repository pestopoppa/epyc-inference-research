"""Adversarial acceptance for the 2026-08-16 quant-ladder design prior."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

from . import hypothesis_portfolio as P


IQ2 = "akh-v2-lowbit-type-specialized-mmvq"
BATCHED = "akh-v2-quant-ladder-batched-wave-slot-residual"
IQ1 = "akh-v2-iq1s-occupancy-discriminator"
BLANKET = "akh-v2-batching-closes-all-lowbit-gaps"

LADDER_EVIDENCE = {
    "ev-quant-ladder-occupancy-knee-20260816": (
        "a10_quant_ladder_occupancy_knee_20260816.md",
        "eec0b1733d4bf5c684f537cf7c28c442f938b17bf3ea6ebfeee85a86c5ccab78",
    ),
    "ev-quant-ladder-tg128-raw-20260816": (
        "ladder-results-20260816.jsonl",
        "7dbdf4dc10b9f2cd4f90c7a2d00bd778d9c67cfca1e2238771b4685744aec3a7",
    ),
    "ev-quant-ladder-np-raw-20260816": (
        "np_sweep-20260816.log",
        "8f19b1c54ce1ebb7c8d95e5948488411f9c1f11468b68a8c2c07618fa6edf3bb",
    ),
    "ev-quant-ladder-manifest-20260815": (
        "a10_quant_ladder_MANIFEST_20260815.md",
        "bcefb69acf85ec73e6b3a0300be38fd6055671e325338f6d6c1658ba37193722",
    ),
    "ev-iq2-vgpr-static-20260812": (
        "a10_iq2_vgpr_lever_20260812.md",
        "7ec3e4340b243c6f7afdb0f9c5cea6b95227351c88cab2a03ecefbcc8183cd46",
    ),
}


def text(value: object) -> str:
    def jsonable(item: object):
        if isinstance(item, Mapping):
            return {key: jsonable(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(child) for child in item]
        return item
    return json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False).casefold()


class QuantLadderPortfolioAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.portfolio = P.load(P.DEFAULT_PORTFOLIO)
        cls.evidence = {
            row["evidence_id"]: row for row in cls.portfolio.body["evidence"]
        }

    def test_all_root_evidence_is_immutable_checked_in_and_byte_exact(self):
        for evidence_id, (basename, expected_sha) in LADDER_EVIDENCE.items():
            with self.subTest(evidence_id=evidence_id):
                row = self.evidence[evidence_id]
                self.assertEqual(row["authority"], "non_governed_design_prior")
                self.assertTrue(row["path"].startswith(
                    "repo://scripts/kernel_rnd/autokernel/evidence/"))
                self.assertTrue(row["path"].endswith(basename))
                self.assertEqual(row["sha256"], expected_sha)
                raw = P.read_evidence_bytes(row["path"], evidence_id)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha)
                self.assertIn("9e21451c5680d10eae7b577979a9e78b39d27eed",
                              text(row["claims"]))
        P.verify_evidence_files(self.portfolio, LADDER_EVIDENCE)

    def test_repo_carrier_traversal_and_non_evidence_paths_are_refused(self):
        body = json.loads(P.DEFAULT_PORTFOLIO.read_bytes())
        body["evidence"][0]["path"] = "repo://scripts/kernel_rnd/autokernel/evidence/../../secret"
        with self.assertRaisesRegex(P.PortfolioError, "must stay under"):
            P.validate(body)
        body = json.loads(P.DEFAULT_PORTFOLIO.read_bytes())
        body["evidence"][0]["path"] = "repo://docs/not-evidence.md"
        with self.assertRaisesRegex(P.PortfolioError, "must stay under"):
            P.validate(body)
        with self.assertRaisesRegex(P.PortfolioError, "must stay under"):
            P.read_evidence_bytes(
                "repo://scripts/kernel_rnd/autokernel/evidence/../../secret",
                "direct-traversal")

    @staticmethod
    def temporary_carrier(temporary: str) -> tuple[Path, Path, str]:
        root = Path(temporary) / "research"
        carrier = (root / "scripts" / "kernel_rnd" / "autokernel" /
                   "evidence" / "test" / "carrier.bin")
        carrier.parent.mkdir(parents=True)
        carrier.write_bytes(b"sealed-carrier")
        value = "repo://scripts/kernel_rnd/autokernel/evidence/test/carrier.bin"
        return root, carrier, value

    def test_repo_carrier_leaf_symlink_is_refused_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, carrier, value = self.temporary_carrier(temporary)
            target = carrier.with_name("target.bin")
            target.write_bytes(carrier.read_bytes())
            carrier.unlink()
            carrier.symlink_to(target.name)
            with mock.patch.object(P, "REPOSITORY_ROOT", root), \
                    self.assertRaisesRegex(P.PortfolioError, "without following links"):
                P.read_evidence_bytes(value, "leaf-symlink")

    def test_repo_carrier_intermediate_symlink_inside_or_outside_is_refused(self):
        for outside in (False, True):
            with self.subTest(outside=outside), tempfile.TemporaryDirectory() as temporary:
                root, carrier, _ = self.temporary_carrier(temporary)
                evidence = carrier.parents[1]
                real = ((Path(temporary) / "outside") if outside
                        else (evidence / "real"))
                real.mkdir(parents=True)
                (real / "carrier.bin").write_bytes(b"sealed-carrier")
                link = evidence / "linked"
                link.symlink_to(real, target_is_directory=True)
                value = ("repo://scripts/kernel_rnd/autokernel/evidence/"
                         "linked/carrier.bin")
                with mock.patch.object(P, "REPOSITORY_ROOT", root), \
                        self.assertRaisesRegex(P.PortfolioError,
                                               "without following links"):
                    P.read_evidence_bytes(value, "intermediate-symlink")

    def test_repo_evidence_root_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "research"
            autokernel = root / "scripts" / "kernel_rnd" / "autokernel"
            autokernel.mkdir(parents=True)
            outside = Path(temporary) / "outside"
            (outside / "test").mkdir(parents=True)
            (outside / "test" / "carrier.bin").write_bytes(b"sealed-carrier")
            (autokernel / "evidence").symlink_to(outside, target_is_directory=True)
            value = "repo://scripts/kernel_rnd/autokernel/evidence/test/carrier.bin"
            with mock.patch.object(P, "REPOSITORY_ROOT", root), \
                    self.assertRaisesRegex(P.PortfolioError, "without following links"):
                P.read_evidence_bytes(value, "root-symlink")

    def test_repo_carrier_hardlink_is_refused_by_single_link_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, carrier, value = self.temporary_carrier(temporary)
            os.link(carrier, carrier.with_name("second-link.bin"))
            with mock.patch.object(P, "REPOSITORY_ROOT", root), \
                    self.assertRaisesRegex(P.PortfolioError, "single-link"):
                P.read_evidence_bytes(value, "hardlink")

    def test_repo_carrier_replacement_race_is_detected_on_pinned_fd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, carrier, value = self.temporary_carrier(temporary)
            real_read = os.read
            replaced = False

            def racing_read(fd: int, amount: int) -> bytes:
                nonlocal replaced
                data = real_read(fd, amount)
                if data and not replaced:
                    replaced = True
                    moved = carrier.with_name("moved.bin")
                    carrier.rename(moved)
                    carrier.symlink_to(moved.name)
                return data

            with mock.patch.object(P, "REPOSITORY_ROOT", root), \
                    mock.patch.object(P.os, "read", side_effect=racing_read), \
                    self.assertRaisesRegex(P.PortfolioError, "changed while it was read"):
                P.read_evidence_bytes(value, "replacement-race")

    def test_hardlink_creation_race_is_detected_for_repo_and_absolute_readers(self):
        for repository_relative in (True, False):
            with self.subTest(repository_relative=repository_relative), \
                    tempfile.TemporaryDirectory() as temporary:
                root, carrier, repo_value = self.temporary_carrier(temporary)
                value = repo_value if repository_relative else str(carrier)
                real_read = os.read
                linked = False

                def racing_read(fd: int, amount: int) -> bytes:
                    nonlocal linked
                    data = real_read(fd, amount)
                    if data and not linked:
                        linked = True
                        os.link(carrier, carrier.with_name("concurrent-link.bin"))
                    return data

                root_patch = (mock.patch.object(P, "REPOSITORY_ROOT", root)
                              if repository_relative else mock.patch.object(
                                  P, "REPOSITORY_ROOT", P.REPOSITORY_ROOT))
                with root_patch, mock.patch.object(
                        P.os, "read", side_effect=racing_read), \
                        self.assertRaisesRegex(P.PortfolioError,
                                               "changed while it was read"):
                    P.read_evidence_bytes(value, "hardlink-race")

    def test_repository_carrier_is_independent_of_process_cwd(self):
        evidence_id = "ev-iq2-vgpr-static-20260812"
        row = self.evidence[evidence_id]
        expected = P.read_evidence_bytes(row["path"], evidence_id)
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                actual = P.read_evidence_bytes(row["path"], evidence_id)
            finally:
                os.chdir(previous)
        self.assertEqual(actual, expected)

    def test_iq2_authoring_is_a_zero_spill_threshold_not_linear_instruction_value(self):
        row = self.portfolio.hypothesis(IQ2)
        row_text = text(row)
        self.assertEqual(row["epistemic"]["grade"], "design_prior")
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        self.assertEqual(row["target"]["source_symbols"],
                         ("vec_dot_iq2_xxs_q8_1",))
        self.assertIn("v_perm_b32", row_text)
        self.assertIn("at most 64", row_text)
        self.assertRegex(row_text, r"65 to 70.{0,80}no predicted occupancy payoff")
        self.assertIn("scratch=0", row_text)
        self.assertIn("vgpr_spill_count=0", row_text)
        self.assertIn("iq4_xs", row_text)
        self.assertIn("quality-valid", row_text)
        self.assertNotIn("vec_dot_iq3_xxs_q8_1", row["target"]["source_symbols"])
        self.assertIn("iq3 remains a separate characterization question", row_text)
        self.assertEqual(row["primary_falsifier"],
                         "A forced register reduction reaches at most 64 true allocated "
                         "VGPR but introduces scratch or a nonzero vgpr_spill_count")
        policy_text = text(row["decision_policy"])
        self.assertNotIn("90.05", policy_text)
        self.assertNotIn("82.89", policy_text)

    def test_batched_residual_is_conditional_t1b_memory_not_a_run_order(self):
        row = self.portfolio.hypothesis(BATCHED)
        row_text = text(row)
        self.assertEqual(row["status"], "needs-template")
        self.assertEqual(row["lifecycle"]["maturity"], "design_prior")
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        self.assertIn("no autonomous action", row["lifecycle"]["next_action"].casefold())
        self.assertIn("if explicitly activated", row_text)
        self.assertIn("t1b real-graph", row_text)
        self.assertNotIn("t1a", row_text)
        self.assertIn("declared comparator semantics", row_text)
        self.assertIn("min_measurable_us", row_text)
        self.assertIn("cache_state", row_text)
        self.assertIn("a/a band", row_text)
        self.assertIn("b8", row_text)
        self.assertIn("suspected outlier", row_text)
        self.assertIn("graphs-on/real-graph", row_text)
        self.assertIn("graphs-off target-profile", row_text)
        self.assertEqual(row["decision_policy"]["nomination_floor_pct"], 0.0)

    def test_iq1_discriminator_is_inactive_and_never_automatic_spend(self):
        row = self.portfolio.hypothesis(IQ1)
        row_text = text(row)
        self.assertEqual(row["status"], "needs-template")
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        self.assertEqual(row["dispatch_anchors"][0]["aggregation"], "not_applicable")
        self.assertIn("none; retain as inactive", row["lifecycle"]["next_action"].casefold())
        self.assertIn("explicit operator", row_text)
        self.assertIn("practical iq1 serving relevance", row_text)
        self.assertIn("t1b real-graph", row_text)
        self.assertIn("do not spend gpu or source-authoring budget", row_text)
        self.assertEqual(row["decision_policy"]["nomination_floor_pct"], 0.0)

    def test_blanket_batching_premise_is_warn_only_not_a_hard_dnr(self):
        row = self.portfolio.hypothesis(BLANKET)
        row_text = text(row)
        self.assertEqual(row["status"], "retired")
        self.assertEqual(row["lifecycle"]["maturity"], "retired")
        self.assertEqual(row["epistemic"]["grade"], "design_prior")
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        self.assertNotIn(BLANKET,
                         {item["dnr_id"] for item in self.portfolio.do_not_repeat})
        self.assertIn("planning assumption only", row_text)
        self.assertIn("not a hard dnr", row_text)
        self.assertIn("does not block batching measurement", row_text)
        self.assertIn("narrower per-format", row_text)
        self.assertIn("governed t1b real-graph", row_text)

    def test_correctness_and_promotion_prohibitions_survive_every_new_record(self):
        iq2 = text(self.portfolio.hypothesis(IQ2))
        batched = text(self.portfolio.hypothesis(BATCHED))
        blanket = text(self.portfolio.hypothesis(BLANKET))
        for record in (iq2, batched, blanket):
            self.assertIn("correctness", record)
            self.assertRegex(record, r"promotion|production")
        self.assertIn("thirty-chunk", iq2)
        self.assertIn("speed probes", batched)

    def test_current_eligible_projection_is_unchanged(self):
        self.assertEqual(
            {row["hypothesis_id"] for row in self.portfolio.eligible_projection()},
            {"akh-v2-q5-type-specific-dequant",
             "akh-v2-q8-quantizer-new-mechanism",
             "akh-v2-fa-gqa7-pair-tail",
             "akh-v2-rms-direct-load-reduction"},
        )
        for hypothesis_id in (IQ2, BATCHED, IQ1, BLANKET):
            with self.assertRaisesRegex(P.PortfolioError,
                                        "not current-bundle eligible"):
                self.portfolio.eligible_record(hypothesis_id)


if __name__ == "__main__":
    unittest.main()
