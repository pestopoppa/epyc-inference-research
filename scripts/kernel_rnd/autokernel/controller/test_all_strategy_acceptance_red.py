"""Hardware-free acceptance gate for every currently eligible GPU strategy.

The non-FA assertions describe seams that are already usable.  The two FA
assertions are intentionally red until the paired-head bulk/tail dispatch and
odd-GQA7 correctness contracts are sealed.  This file is test-only audit work;
it does not widen runtime authority.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from .. import hypothesis_portfolio
from . import discovery_controller as C
from . import discovery_deployment_factory as F


ELIGIBLE = (
    ("akh-v2-q5-type-specific-dequant", "cuda-vecdotq-v1", "MUL_MAT", 1139),
    ("akh-v2-q8-quantizer-new-mechanism", "cuda-quantize-q8-v1", "MUL_MAT", 1139),
    ("akh-v2-fa-gqa7-pair-tail", "cuda-fattn-tile-v1", "FLASH_ATTN_EXT", 2868),
    ("akh-v2-rms-direct-load-reduction", "cuda-norm-v2", "RMS_NORM", 21),
)


class AllStrategyAcceptanceRedGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.portfolio = hypothesis_portfolio.load(
            hypothesis_portfolio.DEFAULT_PORTFOLIO)
        cls.registry = F._template_registry()
        cls.dispatch = F._portfolio_dispatch_authority(
            cls.registry, cls.portfolio)
        cls.surfaces = F._normalized_template_surfaces(
            cls.registry, cls.portfolio)

    def test_all_four_planner_source_and_current_dispatch_bindings_exist(self):
        records = sorted(
            self.portfolio.eligible_hypotheses(),
            key=lambda row: row["priority"]["rank"])
        self.assertEqual(
            [row["hypothesis_id"] for row in records],
            [row[0] for row in ELIGIBLE])
        for hypothesis_id, template_id, op, cases in ELIGIBLE:
            with self.subTest(hypothesis=hypothesis_id):
                record = next(row for row in records
                              if row["hypothesis_id"] == hypothesis_id)
                template = self.registry.templates[template_id]
                self.assertEqual(
                    tuple(record["target"]["source_files"]),
                    tuple(sorted(template.allowed_files)))
                self.assertTrue(set(record["target"]["source_symbols"]).issubset(
                    template.allowed_symbols[record["target"]["source_files"][0]]))
                self.assertIn(
                    record["mechanism"]["facets"]["change_class"],
                    self.surfaces[template_id]["change_classes"])
                self.assertEqual(template.semantics["correctness_op"], op)
                self.assertEqual(
                    template.semantics["expected_correctness_cases"], cases)
                self.assertGreaterEqual(len(self.dispatch[hypothesis_id]), 1)

    def test_portfolio_accounting_selects_each_next_hypothesis_after_budget(self):
        config = SimpleNamespace(
            hypothesis_portfolio=self.portfolio,
            hypothesis_portfolio_sha256=self.portfolio.sha256,
            planner_context={"portfolio_dispatch_authority": self.dispatch})
        state = {"iterations": [], "portfolio_terminals": {}}
        for hypothesis_id, _template_id, _op, _cases in ELIGIBLE:
            selected = C._select_portfolio_binding(state, config)
            self.assertIsNotNone(selected)
            self.assertEqual(selected["hypothesis_id"], hypothesis_id)
            policy = selected["decision_policy"]
            # A measured terminal is authoritative regardless of whether its
            # budget was 2 or 3 distinct candidates.
            state["portfolio_terminals"][hypothesis_id] = {
                "disposition": policy["terminal_rule"], "policy": policy}
        self.assertIsNone(C._select_portfolio_binding(state, config))

    def test_fa_has_sealed_distinct_bulk_and_tail_candidate_routes(self):
        """RED: a pair+tail mutation cannot be forced through the anchor route."""
        template = self.registry.templates["cuda-fattn-tile-v1"]
        variants = template.semantics.get("candidate_dispatch_variants")
        self.assertIsInstance(variants, dict)
        self.assertEqual(set(variants), {"gqa7_bulk_pairs", "gqa7_scalar_tail"})
        for name, row in variants.items():
            with self.subTest(route=name):
                self.assertEqual(row["gqa_ratio"], 7)
                self.assertEqual(row["head_size"], 64)
                self.assertIn(row["ncols2"], {1, 2})
                self.assertIsInstance(row["kernel_name"], str)
                self.assertGreater(row["calls"], 0)
                self.assertGreater(row["grid"], 0)
                self.assertGreater(row["workgroup"], 0)
                self.assertGreaterEqual(row["lds_bytes"], 0)
        self.assertEqual(variants["gqa7_bulk_pairs"]["ncols2"], 2)
        self.assertEqual(variants["gqa7_scalar_tail"]["ncols2"], 1)

    def test_fa_correctness_requires_and_receipts_exact_odd_gqa7_cases(self):
        """RED: the generic FLASH_ATTN_EXT total has no GQA7 shape proof."""
        template = self.registry.templates["cuda-fattn-tile-v1"]
        required = template.semantics.get("required_correctness_cases")
        self.assertIsInstance(required, list)
        self.assertGreaterEqual(len(required), 1)
        for row in required:
            self.assertEqual(row["op"], "FLASH_ATTN_EXT")
            self.assertEqual(row["hsk"], 64)
            self.assertEqual(row["hsv"], 64)
            self.assertEqual(row["gqa_ratio"], 7)
            self.assertEqual(row["query_tokens"], 1)
            self.assertGreaterEqual(row["expected_matches"], 1)

    def test_each_strategy_has_a_separate_graphs_on_target_runtime_screen(self):
        """RED: serialized graphs-off attribution cannot discharge this gate."""
        records = {row["hypothesis_id"]: row
                   for row in self.portfolio.eligible_hypotheses()}
        for hypothesis_id, template_id, _op, _cases in ELIGIBLE:
            with self.subTest(hypothesis=hypothesis_id):
                record = records[hypothesis_id]
                self.assertIs(record["regime"]["target_runtime_graphs"], True)
                screen = self.registry.templates[template_id].semantics.get(
                    "target_runtime_screen")
                self.assertIsInstance(screen, dict)
                self.assertEqual(screen["workload"], "decode_tg128")
                self.assertIs(screen["hip_graphs"], True)
                self.assertIs(screen["paired"], True)
                self.assertIs(screen["decision_required"], True)


if __name__ == "__main__":
    unittest.main()
