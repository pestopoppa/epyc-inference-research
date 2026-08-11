from __future__ import annotations

import unittest
import hashlib

from .. import schemas
from . import authoring_contract as A


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def context(*items: A.ContextItem) -> A.PricedContext:
    return A.price_context(
        round_id="round-1",
        budget=A.ContextBudget(max_total_tokens=200, max_item_tokens=100, max_items=4),
        items=items,
    )


class TestPromptLeakGuard(unittest.TestCase):
    def test_compliant_fully_rendered_prompt_passes(self):
        rendered = A.assemble_authoring_prompt(
            role="planner", task="Improve decode without changing semantics.",
            context=context(A.ContextItem(
                source_ref="kernel.cpp@abc:L10-L30",
                purpose="inspect dispatch", content="choose the existing path")),
        )
        self.assertIn("AUTOKERNEL AUTHORING ROLE", rendered)

    def test_each_sealed_marker_is_rejected_after_assembly(self):
        for marker in A.FORBIDDEN_PROMPT_MARKERS:
            with self.subTest(marker=marker), self.assertRaises(A.PromptLeakError):
                A.assemble_authoring_prompt(
                    role="actor", task="Implement the candidate.",
                    context=context(A.ContextItem(
                        source_ref="diagnostic", purpose="repair", content=marker)),
                )

    def test_exact_error_threshold_pair_is_rejected(self):
        with self.assertRaises(A.PromptLeakError):
            A.assemble_authoring_prompt(
                role="actor", task="repair ERR = 0.12 > 0.10",
                context=context())


class TestPublicSealedDiagnostics(unittest.TestCase):
    def population(self) -> A.CasePopulation:
        return A.CasePopulation(
            population_id="akcases-v1",
            selection_seed_sha256=hashlib.sha256(b"case split").hexdigest(),
            cases=(
                A.EvaluationCase("public-1", "boundary", "PUBLIC"),
                A.EvaluationCase("sealed-1", "hostile", "SEALED"),
            ))

    def test_only_public_safe_summaries_reach_the_prompt(self):
        disclosure = A.filter_refine_diagnostics(
            population=self.population(), diagnostics=(
                A.DiagnosticRecord(
                    "public-1", "ERR = 0.12 > 0.10 in test-backend-ops",
                    "max_nmse_err=0.10"),
                A.DiagnosticRecord(
                    "sealed-1", "secret shape failed", "sealed exact values"),
            ))
        prompt = A.assemble_authoring_prompt(
            role="actor", task="repair the public failure", context=context(),
            diagnostics=disclosure)
        self.assertIn("numerical mismatch in a public case", prompt)
        self.assertNotIn("sealed-1", prompt)
        self.assertNotIn("secret shape", prompt)
        self.assertEqual(len(disclosure.sealed), 1)

    def test_population_requires_both_disjoint_visibility_sets(self):
        with self.assertRaises(ValueError):
            A.CasePopulation(
                population_id="bad",
                selection_seed_sha256=hashlib.sha256(b"bad").hexdigest(),
                cases=(A.EvaluationCase("only", "one", "PUBLIC"),))


class TestPricedContext(unittest.TestCase):
    def test_table_prices_each_selected_excerpt(self):
        priced = context(A.ContextItem(
            source_ref="a.cpp@deadbeef:L1-L4", purpose="one symbol", content="abcd"))
        table = priced.render_budget_table()
        self.assertIn("Estimated tokens", table)
        self.assertIn("1 / 200", table)
        self.assertIn("NEVER", table)

    def test_bulk_read_is_never_admitted(self):
        with self.assertRaisesRegex(A.ContextBudgetError, "bulk reads are never"):
            context(A.ContextItem(
                source_ref="whole-repo", purpose="maybe useful", content="all files",
                bulk_read=True))

    def test_item_and_total_caps_bite(self):
        budget = A.ContextBudget(max_total_tokens=2, max_item_tokens=2, max_items=2)
        with self.assertRaisesRegex(A.ContextBudgetError, "ceiling"):
            A.price_context(
                round_id="r", budget=budget,
                items=(A.ContextItem("a", "a", "12345678"),
                       A.ContextItem("b", "b", "12345678")))


class TestReversibleCompaction(unittest.TestCase):
    def test_header_names_kept_dropped_and_exact_git_recovery(self):
        record = A.CompactionRecord(
            kept=("champion and current falsifier",),
            dropped=("superseded exploration transcript",),
            recovery=A.GitRecoveryRecipe(
                repo="/workspace/repos/research", commit=COMMIT,
                path="logs/autokernel.md"),
            compacted_body="# Current state\nKeep working.",
        )
        rendered = record.render()
        self.assertLess(rendered.index("WHAT WAS KEPT"), rendered.index("# Current state"))
        self.assertIn("WHAT WAS DROPPED", rendered)
        self.assertIn(f"git -C /workspace/repos/research show {COMMIT}:logs/autokernel.md",
                      rendered)

    def test_unsafe_or_ambiguous_recovery_is_refused(self):
        for path in ("../log.md", "/absolute/log.md", "-option"):
            with self.subTest(path=path), self.assertRaises(A.CompactionError):
                A.GitRecoveryRecipe(repo="/workspace", commit=COMMIT, path=path)


class TestExternalNumbers(unittest.TestCase):
    def number(self, **overrides) -> A.ExternalNumber:
        values = dict(
            external_number_id="akxn-h100-bf16-001",
            label="published single-stream throughput",
            observed_value=800.0, unit="GB/s",
            source_ref="https://example.invalid/paper",
            retrieved_at="2026-08-11T00:00:00+00:00",
            quant="BF16", basis="spec", denominator_value=1000.0,
            denominator_source_ref="https://example.invalid/vendor-spec",
        )
        values.update(overrides)
        return A.ExternalNumber(**values)

    def test_structured_number_is_normalized_and_rendered(self):
        number = self.number()
        self.assertEqual(number.normalized_roofline_utilization, 0.8)
        prompt = A.assemble_authoring_prompt(
            role="planner", task="Consider the cited cross-vendor prior.",
            context=context(), external_numbers=(number,))
        self.assertIn("roofline_utilization=0.800000", prompt)
        self.assertIn("source_revision=2026-08-11", prompt)

    def test_pooled_quant_missing_revision_and_mixed_basis_are_refused(self):
        with self.assertRaises(ValueError):
            self.number(quant="pooled")
        with self.assertRaises(ValueError):
            self.number(retrieved_at=None, source_commit=None)
        payload = self.number().to_dict()
        payload["roofline_denominator"]["basis"] = "measured_achievable"
        self.assertTrue(any("does not match" in error
                            for error in schemas.validate_external_number(payload)))


if __name__ == "__main__":
    unittest.main()
