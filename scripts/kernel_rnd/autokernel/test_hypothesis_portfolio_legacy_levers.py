"""Acceptance gate for preserving the L1-MoE, L6, and L21 research memory.

These assertions intentionally fail on portfolio v2 at commit 2153ccac: the three
lever families are present in the source handoffs but are not represented by exact,
regime-scoped portfolio records.  The gate is deliberately semantic.  A neighboring
MoE/layout or Q4_K/MMVQ record cannot satisfy it by merely mentioning the same op.
"""
from __future__ import annotations

import json
import unittest
from collections.abc import Mapping

from . import hypothesis_portfolio as P


LEVER_MATRIX_EVIDENCE = "ev-fable5-mi210-lever-matrix"
ROOFLINE_EVIDENCE = "ev-fable5-mi210-roofline"

EXPECTED_EVIDENCE = {
    LEVER_MATRIX_EVIDENCE: {
        "path": (
            "/workspace/handoffs/active/"
            "fable5-window2-findings-05c-mi210-lever-category-matrix.md"
        ),
        "sha256": "2f6cb30655b4cf01998249fc57619a9e080ae45ba3f22e95eda29bd8bbc179bb",
    },
    ROOFLINE_EVIDENCE: {
        "path": (
            "/workspace/handoffs/active/"
            "fable5-window2-findings-05-intake-sweep-and-roofline.md"
        ),
        "sha256": "08de87ef44a14de4420432bd04aad5f7e3c3f41639578e9e7f4a7f35dea64357",
    },
}

L1_DNR_ID = "dnr-l1-moe-mmid-a3b-a4b-low-batch"
L1_ULTRA_SPARSE_ID = "akh-v2-ultra-sparse-moe-mmid"
L6_SOA_ID = "akh-v2-q8-soa-repack-conditional"
L21_Q4K_ID = "akh-v2-q4k-mmq-dequant-gemv"


def _text(value: object) -> str:
    """Return stable case-folded prose without losing Unicode result signs."""

    def jsonable(item: object):
        if isinstance(item, Mapping):
            return {key: jsonable(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(child) for child in item]
        return item

    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True).casefold()


class LegacyLeverPortfolioAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.portfolio = P.load(P.DEFAULT_PORTFOLIO)
        cls.hypotheses = {
            row["hypothesis_id"]: row for row in cls.portfolio.hypotheses
        }
        cls.dnrs = {row["dnr_id"]: row for row in cls.portfolio.do_not_repeat}
        cls.evidence = {
            row["evidence_id"]: row for row in cls.portfolio.body["evidence"]
        }

    def hypothesis(self, hypothesis_id: str):
        if hypothesis_id not in self.hypotheses:
            self.fail(f"missing exact legacy-lever hypothesis {hypothesis_id}")
        return self.hypotheses[hypothesis_id]

    def dnr(self, dnr_id: str):
        if dnr_id not in self.dnrs:
            self.fail(f"missing exact legacy-lever DNR {dnr_id}")
        return self.dnrs[dnr_id]

    def assert_handoff_provenance(self, row, *evidence_ids: str) -> None:
        self.assertTrue(set(evidence_ids).issubset(row["evidence_refs"]))
        provenance = _text(row["provenance"])
        self.assertIn("fable5-window2-findings", provenance)

    def assert_ineligible(self, row) -> None:
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        eligible_ids = {
            item["hypothesis_id"] for item in self.portfolio.eligible_projection()
        }
        self.assertNotIn(row["hypothesis_id"], eligible_ids)

    def test_exact_handoff_evidence_identities_are_bound(self):
        missing = sorted(set(EXPECTED_EVIDENCE) - set(self.evidence))
        self.assertFalse(missing, f"missing exact handoff evidence identities: {missing}")
        for evidence_id, expected in EXPECTED_EVIDENCE.items():
            with self.subTest(evidence_id=evidence_id):
                actual = self.evidence[evidence_id]
                self.assertEqual(actual["path"], expected["path"])
                self.assertEqual(actual["sha256"], expected["sha256"])
                self.assertTrue(
                    any("hypoth" in claim.casefold() or "lever" in claim.casefold()
                        for claim in actual["claims"]),
                    f"{evidence_id} must declare its research-memory authority",
                )

    def test_l1_moe_scoped_negative_and_ultra_sparse_residual(self):
        dnr = self.dnr(L1_DNR_ID)
        residual = self.hypothesis(L1_ULTRA_SPARSE_ID)

        self.assertEqual(dnr["classification"], "measured_negative")
        self.assertEqual(
            dnr["enforcement"], "hard_refusal_exact_mechanism_and_regime"
        )
        self.assertEqual(dnr["regime"]["backend"], "hip")
        self.assertEqual(dnr["regime"]["phase"], "decode")
        self.assertEqual(dnr["regime"]["architecture"], "gfx90a")
        self.assertRegex(str(dnr["regime"]["batch"]).casefold(), r"(?:low.*batch|b?2.*b?8)")

        dnr_text = _text(dnr)
        self.assertIn("a3b", dnr_text)
        self.assertIn("a4b", dnr_text)
        self.assertIn("get_mmvq_mmid_max_batch_cdna", dnr_text)
        self.assertRegex(dnr_text, r"default(?:\s+threshold)?\s*(?:=|is)\s*8")
        for batch, loss in ((2, "30"), (4, "21"), (8, "10\\.5")):
            self.assertRegex(
                dnr_text,
                rf"\bb{batch}\b.{{0,48}}(?:-|−)\s*{loss}\s*%",
            )
        self.assertRegex(dnr_text, r"b\s*[≥>]=?\s*16|b>=16|b≥16")
        self.assertRegex(dnr_text, r"(?:±|\+/-)\s*0\.4\s*%")
        self.assertIn(L1_ULTRA_SPARSE_ID, _text(dnr["reentry_conditions"]))
        self.assert_handoff_provenance(dnr, LEVER_MATRIX_EVIDENCE)

        self.assert_ineligible(residual)
        self.assertIn(residual["status"], {"queued", "needs-template"})
        residual_text = _text(residual)
        self.assertIn("ultra-sparse", residual_text)
        self.assertIn("256", residual_text)
        self.assertRegex(residual_text, r"(?:256\s*[-/ ]\s*(?:of\s*)?8|8\s*(?:active|selected))")
        self.assertIn("mmid", residual_text)
        self.assertRegex(
            _text(residual["current_bundle_eligibility"]),
            r"(?:routing|expert).*(?:evidence|histogram)"
            r"|(?:evidence|histogram).*(?:routing|expert)",
        )
        self.assertNotEqual(
            residual["mechanism"]["fingerprint_sha256"],
            dnr["mechanism"]["fingerprint_sha256"],
        )
        self.assert_handoff_provenance(residual, LEVER_MATRIX_EVIDENCE)

    def test_l6_soa_repack_is_a_profile_admitted_conditional_hold(self):
        row = self.hypothesis(L6_SOA_ID)
        self.assert_ineligible(row)
        self.assertEqual(row["status"], "needs-template")
        self.assertEqual(row["regime"]["backend"], "hip")
        self.assertEqual(row["regime"]["phase"], "decode")
        self.assertEqual(row["regime"]["architecture"], "gfx90a")

        row_text = _text(row)
        self.assertIn("q8_0", row_text)
        self.assertIn("soa", row_text)
        self.assertIn("repack", row_text)
        self.assertIn("tcc_ea_rdreq_32b", row_text)
        self.assertIn("cache-line", row_text)
        self.assertIn("amplification", row_text)
        self.assertRegex(row_text, r"healthy.{0,48}coalesc|coalesc.{0,48}healthy")
        self.assertRegex(
            _text(row["current_bundle_eligibility"]),
            r"(?:refus|reject|not admit|remain.*hold).{0,96}(?:healthy|coalesc)"
            r"|(?:healthy|coalesc).{0,96}(?:refus|reject|not admit|remain.*hold)",
        )
        self.assertRegex(
            _text([row["falsifiers"], row["stop_rule"]]),
            r"(?:no|without|absent|healthy).{0,64}(?:sub-line|cache-line|coalesc)",
        )
        self.assert_handoff_provenance(row, LEVER_MATRIX_EVIDENCE)

    def test_l21_q4k_mmq_parent_preserves_regime_falsifiers_and_children(self):
        row = self.hypothesis(L21_Q4K_ID)
        self.assert_ineligible(row)
        self.assertEqual(row["status"], "needs-template")
        self.assertEqual(row["regime"]["backend"], "hip")
        self.assertEqual(row["regime"]["phase"], "decode")
        self.assertEqual(row["regime"]["architecture"], "gfx90a")
        self.assertEqual(row["regime"]["batch"], 1)
        self.assertIn("q4_k", row["regime"]["quant"].casefold())

        row_text = _text(row)
        for token in ("mmq", "dequant", "gemv", "latency"):
            self.assertIn(token, row_text)
        self.assertRegex(row_text, r"(?:gpu drafter|interactive)")
        self.assertRegex(row_text, r"valu.{0,32}issue|issue.{0,32}valu")
        self.assertRegex(row_text, r"hbm.{0,24}saturat|saturat.{0,24}hbm")
        self.assertRegex(row_text, r"28\s*(?:pp|percentage point)")
        self.assertRegex(row_text, r"45\s*(?:-|–|to)\s*50\s*%")
        self.assertIn("mmq.cu", _text(row["target"]["source_files"]))

        linked = {item["with"] for item in row["interactions"]}
        self.assertTrue(
            {
                "akh-v2-q4k-branchless-scale-min",
                "akh-v2-q4k-onewave-incumbent",
            }.issubset(linked),
            "L21 must remain the parent of the narrower branchless and one-wave mechanisms",
        )
        self.assert_handoff_provenance(
            row, LEVER_MATRIX_EVIDENCE, ROOFLINE_EVIDENCE
        )


if __name__ == "__main__":
    unittest.main()
