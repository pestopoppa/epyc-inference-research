#!/usr/bin/env python3
"""test_selection.py — the regression barrier for AK4 target selection (§8.3, §8.4, §8.4.1).

WHY THIS FILE EXISTS
--------------------
Each property below is one this project has paid for once already, and each is
ENFORCED rather than documented:

  * **a cheaper layer is not skipped on a hunch** — the §8.3 receipt is
    arithmetic, and a receipt whose measured ceiling could explain the gap, whose
    events do not resolve, or whose binding is to a commit that has moved, does
    not license the skip;
  * **no filtered proposal is discarded** — every rejection is journaled with its
    fingerprint and reason codes, the second occurrence auto-blacklists, and a run
    trips PLANNER_DEGRADED with evidence the state machine accepts;
  * **cheap checks run before metered drafting** — a fake drafter is asserted
    NEVER to have been called when the prescreen refuses, and a drafter that
    wanders off the screened mechanism is refused after the fact;
  * **the phase floor is derived, never supplied** — a hand-built calibration
    carrying a convenient floor raises at construction;
  * **falling yield with rising PROPOSAL_SKIPPED is PLANNER_DEGRADED, not
    EXPLORE** — the §8.10 conflation that cost this project months of paid no-ops;
  * **deep work is not starved by arithmetic** — an architectural proposal ranked
    LAST on EIG is still selected out of its reserved arm, and an incremental
    proposal may not spend that arm even when the general one is empty;
  * **AK-D36 constrains the metric, never the batch regime** — a batch-128 decode
    proposal is admitted; a whole-stack cross-engine ratio objective is not.

NO inference, NO benchmark, NO build, NO model call, NO process. The only
"provider" here is `_FakeDrafter`, a counter. Every file this suite writes lives
under a per-test temporary directory.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_selection.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_selection.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE so `selection.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.controller import planner as PL  # noqa: E402
from autokernel.controller import selection as SEL  # noqa: E402
from autokernel.controller import state_machine as SM  # noqa: E402

V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
CAMPAIGN = "ak-llama_gpu-decode-20260803"
SHAPE_A = {"op": "mul_mat_vec_q", "m": 4096, "n": 1, "k": 4096}
SHAPE_B = {"op": "mul_mat_vec_q", "m": 8192, "n": 1, "k": 8192}
SHAPE_CONF = {"op": "mul_mat_vec_q", "m": 5120, "n": 1, "k": 5120}
DIGEST_A = S.content_hash(SHAPE_A)
DIGEST_B = S.content_hash(SHAPE_B)
DIGEST_CONF = S.content_hash(SHAPE_CONF)


def _sha(tag: str) -> str:
    import hashlib
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _receipt(layer: str = "placement_and_launch_config", *, ceiling: float = 0.01,
             gap: float = 0.20, commit: str = V8_COMMIT, events=("ake-profile-1",)) -> dict:
    return {
        "layer": layer,
        "measured_gap": gap,
        "layer_ceiling": ceiling,
        # An id that RESOLVES: `check_layer_skip` treats the gap the arithmetic is
        # against as evidence, so a fabricated gap receipt does not license a skip.
        "gap_receipt_id": "ake-profile-1",
        "evidence_event_ids": list(events),
        "anchor_commit": commit,
        "basis": "measured launch-config sweep moves decode by at most 1% in this regime",
    }


def _proposal(**over) -> dict:
    """A §7.2-valid proposal manifest that also satisfies the §8.3/§8.4 screen.

    Deliberately JSON-only: this manifest is journaled, so every nested block —
    including the layer-skip receipts — travels as data.
    """
    proposal = {
        "schema": S.SCHEMA_PROPOSAL,
        "proposal_id": "akp-0001",
        "campaign_id": CAMPAIGN,
        "parent_candidate_id": None,
        "controller": {
            "provider": "fake", "model_id": "fake-planner", "effort": "medium",
            "prompt_bundle_sha256": _sha("prompt"),
            "context_manifest_sha256": _sha("context"),
            "sampling_params": {"temperature": 0.0},
        },
        "realized_cost": {
            "controller_tokens": 0, "build_seconds": 0, "evaluator_wall_seconds": 0,
            "gpu_seconds": 0, "cpu_region_seconds": 0, "storage_gb": 0,
        },
        "hypothesis": "widening the mmvq dispatch threshold keeps the quantized path at n=1",
        "narrative": "planner prose that must never be retrievable",
        "narrative_retrievable": False,
        "change_class": "dispatcher",
        "declared_symbol_deltas": {"added": [], "removed": [], "arity_changed": []},
        "campaign_kind": "dispatch",
        "oracle_reference": {"oracle": None, "commit": None, "license_check": None},
        "novelty_basis": {
            "prior_event_ids": [], "source_receipts": [], "do_not_repeat_matches": [],
        },
        "expected_information_gain": 0.40,
        "target": {
            "regimes": ["decode_b1"], "shapes": [SHAPE_A], "ops": ["mul_mat_vec_q"],
            "models": ["gemma4-26b-a4b-q4km"],
        },
        "non_target": {"regimes": ["prefill_b1"], "shapes": []},
        "mechanism_prediction": {
            "bottleneck_before": "memory_bandwidth",
            "expected_counter_changes": {"MemUnitStalled": "down"},
            "expected_wall_share_ceiling": 0.30,
            "wall_share_receipt_id": "wsr-1",
        },
        "change": {
            "predicted_affected_surface": ["ggml-cuda/mmvq.cu"],
            "files_and_symbols": ["ggml-cuda/mmvq.cu:mul_mat_vec_q"],
            "conceptual_change": "raise the mmvq n-threshold from 1 to 2",
            "parameter_surface": {"threshold": [1, 2, 4]},
            "estimated_diff_size": 12,
        },
        "risks": {
            "correctness": [], "numerical": [], "state_or_rollback": [], "resource": [],
            "integrity": [],
        },
        "fallback": {"dispatch_guard": "GGML_CUDA_MMVQ_N", "kill_switch": "env off"},
        "evaluation_plan": {
            "required_t0": ["t0.ops"], "required_t1": ["t1a.target_op"],
            "conditional_t2": [], "profiler_questions": [],
        },
        "resource_request": {"lane": "gpu", "expected_minutes": 20, "expected_storage_gb": 2},
        "stop_condition": "abandon when the paired-block effect cannot clear the floor",
        "critic_verdict": {"status": "pending", "reasons": []},
        SEL.SELECTION_BLOCK_KEY: {
            "mechanism": "mmvq-dispatch-threshold",
            "hierarchy_layer": "dispatcher",
            "conceptual_change_count": 1,
            "expected_end_to_end_gain": 0.05,
            "domains": ["llama.cpp/ggml-cuda"],
            "regime_identity": {
                "backend": ["llama_gpu"], "phase": ["decode"], "quant": ["Q4_K"],
                "batch": [1],
            },
            "layer_skip_receipts": [_receipt()],
        },
    }
    for key, value in over.items():
        if key == SEL.SELECTION_BLOCK_KEY and isinstance(value, dict):
            proposal[key] = {**proposal[key], **value}
        else:
            proposal[key] = value
    return proposal


def _context(**over) -> SEL.SelectionContext:
    base = dict(
        campaign_id=CAMPAIGN,
        backend="llama_gpu",
        source_tree="llama.cpp",
        anchor_commit=V8_COMMIT,
        phase=SEL.PHASE_HARVEST,
        owned_domains=frozenset({"llama.cpp/ggml-cuda"}),
        correctness_oracles={"mul_mat_vec_q": "oracle.ops.mmvq"},
        real_graph_shape_digests=frozenset({DIGEST_A, DIGEST_B}),
        confirmation_shape_digests=frozenset({DIGEST_CONF}),
        wall_share_receipts={"wsr-1": 0.30},
        measured_profile={"gemm": 0.55, "elementwise_norm": 0.30, "attention": 0.15},
        evaluator_steps=frozenset({"t0.ops", "t1a.target_op", "t1b.tiny_graph", "t2.lineage"}),
        budget_remaining={
            "wall_minutes": 600.0, "gpu_minutes": 300.0, "cpu_region_minutes": 300.0,
            "storage_gb": 50.0, "candidates": 20.0,
        },
        known_event_ids=frozenset({"ake-profile-1", "ake-profile-2"}),
    )
    base.update(over)
    return SEL.SelectionContext(**base)


def _screen(proposal=None, context=None, blacklist=frozenset()) -> SEL.ScreenResult:
    return SEL.screen_proposal(
        proposal if proposal is not None else _proposal(),
        context if context is not None else _context(),
        blacklisted_fingerprints=blacklist,
    )


class _JournalCase(unittest.TestCase):
    """Every journal-backed test gets its own tmpdir and closes it deterministically."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = str(Path(self._tmp.name) / "journal")
        self.journal = J.Journal(self.root, campaign_id=CAMPAIGN)
        self.journal.initialize()

    def screener(self) -> SEL.ProposalScreener:
        return SEL.ProposalScreener(self.journal, campaign_id=CAMPAIGN)


# =============================================================================
# §8.3 — the hierarchy and its skip receipts
# =============================================================================

class TestHierarchy(unittest.TestCase):

    def test_hierarchy_is_the_designs_order(self):
        self.assertEqual(SEL.HIERARCHY[0], "placement_and_launch_config")
        self.assertEqual(SEL.HIERARCHY[-1], "alternate_engine")
        self.assertEqual(len(SEL.HIERARCHY), 9)
        self.assertEqual(
            [SEL.HIERARCHY_RANK[n] for n in SEL.HIERARCHY], list(range(9))
        )

    def test_cheapest_layer_needs_no_receipt(self):
        check = SEL.check_layer_skip(
            "placement_and_launch_config", (), anchor_commit=V8_COMMIT,
            known_event_ids=frozenset(),
        )
        self.assertEqual(check.outcome, S.PASS)

    def test_skipping_without_a_receipt_fails_and_names_every_layer(self):
        check = SEL.check_layer_skip(
            "operator_fusion", (), anchor_commit=V8_COMMIT, known_event_ids=frozenset(),
        )
        self.assertEqual(check.outcome, S.FAIL)
        joined = " ".join(check.reasons)
        for layer in SEL.HIERARCHY[:4]:
            self.assertIn(layer, joined)

    def test_receipt_whose_ceiling_could_explain_the_gap_fails(self):
        receipt = SEL.LayerSkipReceipt.from_dict(_receipt(ceiling=0.25, gap=0.20))
        check = SEL.check_layer_skip(
            "dispatcher", (receipt,), anchor_commit=V8_COMMIT,
            known_event_ids=frozenset({"ake-profile-1"}),
        )
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("CAN explain the gap", " ".join(check.reasons))

    def test_receipt_bound_to_a_moved_anchor_does_not_license_the_skip(self):
        receipt = SEL.LayerSkipReceipt.from_dict(_receipt(commit=V7_COMMIT))
        check = SEL.check_layer_skip(
            "dispatcher", (receipt,), anchor_commit=V8_COMMIT,
            known_event_ids=frozenset({"ake-profile-1"}),
        )
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("no longer resolves", " ".join(check.reasons))

    def test_receipt_citing_an_unresolvable_event_fails(self):
        receipt = SEL.LayerSkipReceipt.from_dict(_receipt(events=("ake-does-not-exist",)))
        check = SEL.check_layer_skip(
            "dispatcher", (receipt,), anchor_commit=V8_COMMIT,
            known_event_ids=frozenset({"ake-profile-1"}),
        )
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("do not resolve", " ".join(check.reasons))

    def test_unmeasured_ceiling_is_could_not_check_not_a_pass(self):
        receipt = SEL.LayerSkipReceipt.from_dict(_receipt(ceiling=None))
        check = SEL.check_layer_skip(
            "dispatcher", (receipt,), anchor_commit=V8_COMMIT,
            known_event_ids=frozenset({"ake-profile-1"}),
        )
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(check.passed)

    def test_full_receipt_set_passes(self):
        receipts = tuple(
            SEL.LayerSkipReceipt.from_dict(_receipt(layer=layer))
            for layer in SEL.HIERARCHY[:4]
        )
        check = SEL.check_layer_skip(
            "operator_fusion", receipts, anchor_commit=V8_COMMIT,
            known_event_ids=frozenset({"ake-profile-1"}),
        )
        self.assertEqual(check.outcome, S.PASS)

    def test_receipt_refuses_a_non_positive_gap(self):
        with self.assertRaises(ValueError):
            SEL.LayerSkipReceipt.from_dict(_receipt(gap=0.0))

    def test_receipt_refuses_empty_evidence(self):
        with self.assertRaises(ValueError):
            SEL.LayerSkipReceipt.from_dict(_receipt(events=()))

    def test_known_event_ids_is_required(self):
        with self.assertRaises(TypeError):
            SEL.check_layer_skip("dispatcher", (), anchor_commit=V8_COMMIT,
                                 known_event_ids=None)


# =============================================================================
# §19.2 / §19.3 — the do-not-repeat ledger
# =============================================================================

class TestLedger(unittest.TestCase):

    def _entry(self, **over) -> SEL.LedgerEntry:
        base = dict(
            entry_id="dnr-mmvq-1",
            entry_class="MATCHED_NEGATIVE",
            mechanism="mmvq-dispatch-threshold",
            match_dimensions={"backend": ("llama_gpu",), "quant": ("Q4_K",)},
            reopen_when="mmvq.cu changes",
            receipt="67a433bf:ggml-cuda/mmvq.cu:538",
            anchor_commit=V8_COMMIT,
        )
        base.update(over)
        return SEL.LedgerEntry(**base)

    def test_suppression_without_a_receipt_is_inadmissible(self):
        with self.assertRaises(SEL.LedgerEntryInadmissible):
            self._entry(receipt=None)
        with self.assertRaises(SEL.LedgerEntryInadmissible):
            self._entry(entry_class="HARD_CONSTRAINT", anchor_commit=None)

    def test_suppression_without_regime_identity_is_inadmissible(self):
        with self.assertRaises(SEL.LedgerEntryInadmissible):
            self._entry(match_dimensions={})
        with self.assertRaises(SEL.LedgerEntryInadmissible):
            self._entry(match_dimensions={"not_a_dimension": ("x",)})

    def test_low_value_needs_no_receipt(self):
        entry = self._entry(entry_class="LOW_VALUE", receipt=None, anchor_commit=None)
        self.assertTrue(entry.authoritative_against(V8_COMMIT))

    def test_matching_entry_rejects(self):
        facets = SEL.mechanism_facets(_proposal())
        matches = SEL.match_ledger(
            facets, (self._entry(),), anchor_commit=V8_COMMIT,
            satisfied_reopen_predicates=frozenset(),
        )
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].rejects)

    def test_entry_bound_to_a_moved_anchor_stops_blocking(self):
        facets = SEL.mechanism_facets(_proposal())
        matches = SEL.match_ledger(
            facets, (self._entry(anchor_commit=V7_COMMIT),), anchor_commit=V8_COMMIT,
            satisfied_reopen_predicates=frozenset(),
        )
        self.assertFalse(matches[0].rejects)
        self.assertIn("no longer resolves", matches[0].reason)

    def test_conflicted_entry_never_blocks(self):
        facets = SEL.mechanism_facets(_proposal())
        matches = SEL.match_ledger(
            facets, (self._entry(conflicted=True),), anchor_commit=V8_COMMIT,
            satisfied_reopen_predicates=frozenset(),
        )
        self.assertFalse(matches[0].rejects)

    def test_satisfied_reopen_predicate_unblocks_a_matched_negative(self):
        facets = SEL.mechanism_facets(_proposal())
        matches = SEL.match_ledger(
            facets, (self._entry(),), anchor_commit=V8_COMMIT,
            satisfied_reopen_predicates=frozenset({"mmvq.cu changes"}),
        )
        self.assertFalse(matches[0].rejects)
        self.assertIn("newly satisfied", matches[0].reason)

    def test_hard_constraint_ignores_a_reopen_predicate(self):
        facets = SEL.mechanism_facets(_proposal())
        matches = SEL.match_ledger(
            facets, (self._entry(entry_class="HARD_CONSTRAINT"),), anchor_commit=V8_COMMIT,
            satisfied_reopen_predicates=frozenset({"mmvq.cu changes"}),
        )
        self.assertTrue(matches[0].rejects)

    def test_non_matching_regime_does_not_match(self):
        facets = SEL.mechanism_facets(_proposal())
        matches = SEL.match_ledger(
            facets, (self._entry(match_dimensions={"quant": ("IQ2_XXS",)}),),
            anchor_commit=V8_COMMIT, satisfied_reopen_predicates=frozenset(),
        )
        self.assertEqual(matches, ())


# =============================================================================
# Fingerprinting — structural, never prose
# =============================================================================

class TestFingerprint(unittest.TestCase):

    def test_reworded_prose_does_not_change_the_fingerprint(self):
        first = _proposal()
        second = _proposal(
            proposal_id="akp-0099",
            hypothesis="a completely different sentence about the same change",
            narrative="entirely new prose",
        )
        second["change"] = {**second["change"], "conceptual_change": "reworded identically"}
        self.assertEqual(
            SEL.proposal_fingerprint(first), SEL.proposal_fingerprint(second),
            "a fingerprint that includes prose is one a reworder defeats",
        )

    def test_a_structural_change_changes_the_fingerprint(self):
        first = _proposal()
        second = _proposal(target={**first["target"], "ops": ["mul_mat_q"]})
        self.assertNotEqual(SEL.proposal_fingerprint(first), SEL.proposal_fingerprint(second))

    def test_fingerprint_is_stable_across_key_order(self):
        first = _proposal()
        shuffled = {k: first[k] for k in sorted(first, reverse=True)}
        self.assertEqual(SEL.proposal_fingerprint(first), SEL.proposal_fingerprint(shuffled))


# =============================================================================
# §8.4 — the rejection conditions, before mutation
# =============================================================================

class TestScreenRejections(unittest.TestCase):

    def test_the_baseline_fixture_is_admitted(self):
        result = _screen()
        self.assertTrue(result.admitted, result.codes)
        self.assertEqual(result.rejections, ())
        self.assertEqual(result.checks["wall_share"].outcome, S.PASS)
        self.assertEqual(result.checks["correctness_oracle"].outcome, S.PASS)
        self.assertAlmostEqual(result.performance_value, 0.05)

    def test_gain_above_the_measured_ceiling_is_rejected(self):
        result = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "expected_end_to_end_gain": 0.55,
        }}))
        self.assertIn(SEL.REJECT_WALL_SHARE_CEILING, result.codes)
        self.assertEqual(result.performance_value, 0.0)

    def test_a_fusion_explanation_only_helps_an_actual_fusion(self):
        over = {SEL.SELECTION_BLOCK_KEY: {
            "expected_end_to_end_gain": 0.55,
            "fusion_explanation": "removes an intermediate materialization between two ops",
        }}
        still_rejected = _screen(_proposal(**over))
        self.assertIn(SEL.REJECT_WALL_SHARE_CEILING, still_rejected.codes)

        fusion = _proposal(change_class="fusion", **over)
        admitted = _screen(fusion)
        self.assertNotIn(SEL.REJECT_WALL_SHARE_CEILING, admitted.codes)

    def test_declared_ceiling_above_the_receipt_is_rejected(self):
        proposal = _proposal()
        proposal["mechanism_prediction"] = {
            **proposal["mechanism_prediction"], "expected_wall_share_ceiling": 0.90,
        }
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_WALL_SHARE_CEILING, result.codes)
        self.assertIn("the receipt is the ceiling", " ".join(
            r.reason for r in result.rejections))

    def test_unresolvable_wall_share_receipt_is_unverifiable_not_a_pass(self):
        proposal = _proposal()
        proposal["mechanism_prediction"] = {
            **proposal["mechanism_prediction"], "wall_share_receipt_id": "wsr-missing",
        }
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_UNVERIFIABLE, result.codes)
        self.assertEqual(result.checks["wall_share"].outcome, S.COULD_NOT_CHECK)

    def test_undeclared_expected_gain_is_rejected(self):
        proposal = _proposal()
        del proposal[SEL.SELECTION_BLOCK_KEY]["expected_end_to_end_gain"]
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_GAIN_UNDECLARED, result.codes)

    def test_uncovered_correctness_oracle_is_rejected(self):
        result = _screen(context=_context(correctness_oracles={}))
        self.assertIn(SEL.REJECT_NO_CORRECTNESS_ORACLE, result.codes)

    def test_shapes_absent_from_a_real_graph_are_rejected(self):
        result = _screen(context=_context(real_graph_shape_digests=frozenset()))
        self.assertIn(SEL.REJECT_SHAPES_NOT_IN_REAL_GRAPH, result.codes)

    def test_microkernel_only_campaign_admits_unseen_shapes(self):
        result = _screen(context=_context(
            real_graph_shape_digests=frozenset(), microkernel_only=True))
        self.assertNotIn(SEL.REJECT_SHAPES_NOT_IN_REAL_GRAPH, result.codes)

    def test_receipted_negative_is_rejected(self):
        entry = SEL.LedgerEntry(
            entry_id="dnr-1", entry_class="MATCHED_NEGATIVE",
            mechanism="mmvq-dispatch-threshold",
            match_dimensions={"quant": ("Q4_K",)}, reopen_when="never",
            receipt="67a433bf:mmvq.cu:538", anchor_commit=V8_COMMIT,
        )
        result = _screen(context=_context(ledger=(entry,)))
        self.assertIn(SEL.REJECT_REPEATS_RECEIPTED_NEGATIVE, result.codes)

    def test_a_proposal_cannot_escape_the_ledger_by_omitting_its_regime(self):
        entry = SEL.LedgerEntry(
            entry_id="dnr-1", entry_class="MATCHED_NEGATIVE",
            mechanism="mmvq-dispatch-threshold",
            match_dimensions={"context": ("32k",)}, reopen_when="never",
            receipt="67a433bf:mmvq.cu:538", anchor_commit=V8_COMMIT,
        )
        result = _screen(context=_context(ledger=(entry,)))
        self.assertIn(SEL.REJECT_REGIME_IDENTITY_INCOMPLETE, result.codes)

    def test_conditional_negative_excludes_cells_without_rejecting(self):
        entry = SEL.LedgerEntry(
            entry_id="dnr-2", entry_class="CONDITIONAL_NEGATIVE",
            mechanism="mmvq-dispatch-threshold",
            match_dimensions={"quant": ("Q4_K",)}, reopen_when="new quant",
        )
        result = _screen(context=_context(ledger=(entry,)))
        self.assertTrue(result.admitted, result.codes)
        self.assertIn("quant", result.excluded_cells)

    def test_budget_exhaustion_is_rejected(self):
        result = _screen(context=_context(budget_remaining={
            "wall_minutes": 600.0, "gpu_minutes": 5.0, "cpu_region_minutes": 300.0,
            "storage_gb": 50.0, "candidates": 20.0,
        }))
        self.assertIn(SEL.REJECT_BUDGET_EXCEEDED, result.codes)

    def test_candidate_budget_exhaustion_is_rejected(self):
        result = _screen(context=_context(budget_remaining={
            "wall_minutes": 600.0, "gpu_minutes": 300.0, "cpu_region_minutes": 300.0,
            "storage_gb": 50.0, "candidates": 0.0,
        }))
        self.assertIn(SEL.REJECT_BUDGET_EXCEEDED, result.codes)

    def test_unowned_domain_is_rejected(self):
        result = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "domains": ["epyc-orchestrator/api"],
        }}))
        self.assertIn(SEL.REJECT_CROSSES_UNOWNED_DOMAIN, result.codes)

    def test_evaluator_change_is_rejected(self):
        proposal = _proposal()
        proposal["evaluation_plan"] = {
            **proposal["evaluation_plan"], "required_t1": ["t1z.new_reducer"],
        }
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_REQUIRES_EVALUATOR_CHANGE, result.codes)

    def test_more_than_one_conceptual_change_is_rejected(self):
        result = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "conceptual_change_count": 3,
        }}))
        self.assertIn(SEL.REJECT_MULTIPLE_CONCEPTUAL_CHANGES, result.codes)

    def test_confirmation_stratum_shape_is_rejected_before_a_window(self):
        proposal = _proposal()
        proposal["target"] = {**proposal["target"], "shapes": [SHAPE_CONF]}
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_TARGETS_CONFIRMATION_SHAPE, result.codes)

    def test_context_holds_confirmation_digests_not_shapes(self):
        context = _context()
        for digest in context.confirmation_shape_digests:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_schema_violation_is_rejected(self):
        proposal = _proposal()
        del proposal["mechanism_prediction"]
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_SCHEMA_INVALID, result.codes)

    def test_another_campaigns_proposal_is_rejected(self):
        result = _screen(_proposal(campaign_id="ak-llama_cpu-prefill-20260801"))
        self.assertIn(SEL.REJECT_CAMPAIGN_MISMATCH, result.codes)

    def test_unnamed_mechanism_is_rejected(self):
        proposal = _proposal()
        del proposal[SEL.SELECTION_BLOCK_KEY]["mechanism"]
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_MECHANISM_UNNAMED, result.codes)

    def test_eig_outside_the_unit_interval_is_rejected(self):
        result = _screen(_proposal(expected_information_gain=42.0))
        self.assertIn(SEL.REJECT_EIG_OUT_OF_RANGE, result.codes)

    def test_hierarchy_skip_without_receipts_is_rejected_by_the_screen(self):
        result = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "hierarchy_layer": "new_kernel", "layer_skip_receipts": [],
        }}))
        self.assertIn(SEL.REJECT_HIERARCHY_SKIP_UNRECEIPTED, result.codes)

    def test_malformed_receipt_is_a_rejection_not_an_exception(self):
        result = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "layer_skip_receipts": [{"layer": "placement_and_launch_config"}],
        }}))
        self.assertIn(SEL.REJECT_HIERARCHY_SKIP_UNRECEIPTED, result.codes)

    def test_every_reason_is_reported_not_only_the_first(self):
        proposal = _proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "domains": ["epyc-orchestrator/api"], "expected_end_to_end_gain": 0.99,
        }})
        result = _screen(proposal, context=_context(correctness_oracles={}))
        for code in (SEL.REJECT_CROSSES_UNOWNED_DOMAIN, SEL.REJECT_WALL_SHARE_CEILING,
                     SEL.REJECT_NO_CORRECTNESS_ORACLE):
            self.assertIn(code, result.codes)

    def test_the_proposal_manifest_stays_json_serialisable(self):
        json.dumps(_proposal())


# =============================================================================
# AK-D36 / AK-D37 — the constraint is on the metric, never the batch regime
# =============================================================================

class TestObjectiveScope(unittest.TestCase):

    def test_whole_stack_cross_engine_ratio_is_refused(self):
        result = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "objective": {"kind": "cross_engine_whole_stack_ratio",
                          "comparison_engine": "vllm"},
        }}))
        self.assertIn(SEL.REJECT_FORBIDDEN_OBJECTIVE, result.codes)

    def test_comparison_against_another_engine_is_refused(self):
        result = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "objective": {"kind": "per_phase_improvement", "comparison_engine": "sglang"},
        }}))
        self.assertIn(SEL.REJECT_FORBIDDEN_OBJECTIVE, result.codes)

    def test_batched_decode_is_in_scope(self):
        """AK-D37: read as 'batch-1 only', AK-D36 would retire G15."""
        proposal = _proposal()
        proposal["target"] = {**proposal["target"], "regimes": ["decode_b128"]}
        proposal[SEL.SELECTION_BLOCK_KEY] = {
            **proposal[SEL.SELECTION_BLOCK_KEY],
            "regime_identity": {"backend": ["llama_gpu"], "phase": ["decode"],
                                "quant": ["Q4_K"], "batch": [128]},
            "objective": {"kind": "per_phase_improvement", "comparison_engine": "anchor"},
        }
        result = _screen(proposal)
        self.assertTrue(result.admitted, result.codes)

    def test_batched_prefill_is_in_scope(self):
        proposal = _proposal()
        proposal["target"] = {**proposal["target"], "regimes": ["prefill_b64"]}
        proposal["non_target"] = {"regimes": ["decode_b1"], "shapes": []}
        result = _screen(proposal)
        self.assertTrue(result.admitted, result.codes)


# =============================================================================
# §8.4.1 — architectural campaigns replace three conditions, waive none
# =============================================================================

def _campaign(steps: int = 2, fraction: float = 0.25) -> SEL.ArchitecturalCampaign:
    return SEL.ArchitecturalCampaign(
        campaign_id="akl-persistent-mmvq",
        end_state="a persistent decode team with a resident KV tile",
        steps=tuple(
            SEL.LineageStep(index=i, conceptual_change=f"step {i}",
                            end_state_contribution=f"brings the end state {i}")
            for i in range(steps)
        ),
        reserved_budget_fraction=fraction,
    )


def _architectural_proposal(**over) -> dict:
    proposal = _proposal(**{SEL.SELECTION_BLOCK_KEY: {
        "lineage_step": 0,
        "predicted_post_change_profile": {
            "gemm": 0.62, "elementwise_norm": 0.20, "attention": 0.16,
        },
        "expected_end_to_end_gain": 0.45,
    }})
    for key, value in over.items():
        if key == SEL.SELECTION_BLOCK_KEY and isinstance(value, dict):
            proposal[key] = {**proposal[key], **value}
        else:
            proposal[key] = value
    return proposal


class TestArchitecturalCampaigns(unittest.TestCase):

    def test_campaign_declaration_requires_an_end_state_and_a_reserve(self):
        with self.assertRaises(ValueError):
            SEL.ArchitecturalCampaign(campaign_id="x", end_state="", steps=(),
                                      reserved_budget_fraction=0.2)
        with self.assertRaises(ValueError):
            SEL.ArchitecturalCampaign(
                campaign_id="x", end_state="an end state",
                steps=(SEL.LineageStep(0, "a", "b"),), reserved_budget_fraction=0.0)

    def test_predicted_profile_replaces_the_ceiling(self):
        context = _context(architectural=_campaign(), open_lineage_step=0)
        result = _screen(_architectural_proposal(), context=context)
        self.assertTrue(result.admitted, result.codes)
        self.assertEqual(result.arm, SEL.ARM_ARCHITECTURAL)
        # value is derived from the predicted reduction, never from the planner
        self.assertAlmostEqual(result.performance_value, 0.10, places=6)

    def test_an_identical_predicted_profile_predicts_nothing(self):
        context = _context(architectural=_campaign(), open_lineage_step=0)
        result = _screen(_architectural_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "predicted_post_change_profile": dict(context.measured_profile),
        }}), context=context)
        self.assertIn(SEL.REJECT_WALL_SHARE_CEILING, result.codes)
        self.assertIn("predicts", " ".join(r.reason for r in result.rejections))

    def test_a_profile_that_omits_a_measured_family_is_unfalsifiable_there(self):
        context = _context(architectural=_campaign(), open_lineage_step=0)
        result = _screen(_architectural_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "predicted_post_change_profile": {"gemm": 0.40},
        }}), context=context)
        self.assertIn(SEL.REJECT_WALL_SHARE_CEILING, result.codes)
        self.assertIn("omits", " ".join(r.reason for r in result.rejections))

    def test_prospective_shapes_are_admissible_with_mechanism_and_observation(self):
        context = _context(architectural=_campaign(), open_lineage_step=0,
                           real_graph_shape_digests=frozenset())
        result = _screen(_architectural_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "prospective_shapes": [{
                "shape_digest": DIGEST_A,
                "mechanism": "the new layout emits n=2 tiles where the old one emitted n=1",
                "observation": "dispatch trace records the n=2 kernel executing",
            }],
        }}), context=context)
        self.assertTrue(result.admitted, result.codes)

    def test_prospective_shape_without_an_observation_is_refused(self):
        context = _context(architectural=_campaign(), open_lineage_step=0,
                           real_graph_shape_digests=frozenset())
        result = _screen(_architectural_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "prospective_shapes": [{"shape_digest": DIGEST_A, "mechanism": "the layout"}],
        }}), context=context)
        self.assertIn(SEL.REJECT_SHAPES_NOT_IN_REAL_GRAPH, result.codes)

    def test_the_replacements_are_unavailable_outside_a_declared_campaign(self):
        result = _screen(_architectural_proposal())
        self.assertIn(SEL.REJECT_ARCHITECTURAL_ESCAPE_UNDECLARED, result.codes)

    def test_a_lineage_step_outside_a_declared_campaign_is_refused(self):
        proposal = _proposal(**{SEL.SELECTION_BLOCK_KEY: {"lineage_step": 0}})
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_ARCHITECTURAL_ESCAPE_UNDECLARED, result.codes)

    def test_steps_are_taken_in_the_declared_order(self):
        context = _context(architectural=_campaign(), open_lineage_step=0)
        result = _screen(_architectural_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "lineage_step": 1,
        }}), context=context)
        self.assertIn(SEL.REJECT_LINEAGE_STEP_OUT_OF_ORDER, result.codes)

    def test_a_step_that_does_not_exist_is_refused(self):
        context = _context(architectural=_campaign(steps=2), open_lineage_step=None)
        result = _screen(_architectural_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "lineage_step": 7,
        }}), context=context)
        self.assertIn(SEL.REJECT_LINEAGE_STEP_OUT_OF_ORDER, result.codes)

    def test_correctness_and_budget_gates_are_not_waived_inside_a_campaign(self):
        context = _context(architectural=_campaign(), open_lineage_step=0,
                           correctness_oracles={})
        result = _screen(_architectural_proposal(), context=context)
        self.assertIn(SEL.REJECT_NO_CORRECTNESS_ORACLE, result.codes)


# =============================================================================
# §8.4.1 / AK-D33 — spikes are cheap by construction, and still claimed
# =============================================================================

def _spike(**over) -> dict:
    base = {
        "spike_id": "aks-0001",
        "mechanism_question": "does a persistent team remove the launch tail at all?",
        "resource_lane": "gpu",
        "claim_receipt": "claim-gpu-0007",
        "preflight_ref": "ake-preflight-3",
        "expected_minutes": 4.0,
    }
    base.update(over)
    return base


class TestSpikes(unittest.TestCase):

    def test_a_spike_must_carry_a_claim_and_a_preflight(self):
        with self.assertRaises(ValueError):
            SEL.SpikeDeclaration.from_dict(_spike(claim_receipt=""))
        with self.assertRaises(ValueError):
            SEL.SpikeDeclaration.from_dict(_spike(preflight_ref=""))

    def _spike_proposal(self, **over) -> dict:
        proposal = _proposal(**{SEL.SELECTION_BLOCK_KEY: {"spike": _spike()}})
        proposal["evaluation_plan"] = {
            **proposal["evaluation_plan"], "required_t1": [], "conditional_t2": [],
        }
        for key, value in over.items():
            proposal[key] = value
        return proposal

    def test_a_well_formed_spike_is_admitted_and_carries_no_rate_value(self):
        result = _screen(self._spike_proposal())
        self.assertTrue(result.admitted, result.codes)
        self.assertEqual(result.performance_value, 0.0)
        self.assertGreater(result.information_gain, 0.0)

    def test_a_spike_that_buys_a_rate_cell_is_refused(self):
        proposal = self._spike_proposal()
        proposal["evaluation_plan"] = {
            **proposal["evaluation_plan"], "conditional_t2": ["t2.lineage"],
        }
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_SPIKE_MALFORMED, result.codes)

    def test_a_spike_that_buys_paired_blocks_is_refused(self):
        proposal = self._spike_proposal()
        proposal["evaluation_plan"] = {
            **proposal["evaluation_plan"], "required_t1": ["t1a.target_op"],
        }
        result = _screen(proposal)
        self.assertIn(SEL.REJECT_SPIKE_MALFORMED, result.codes)

    def test_a_malformed_spike_is_a_rejection_not_an_exception(self):
        result = _screen(self._spike_proposal(**{SEL.SELECTION_BLOCK_KEY: dict(
            _proposal()[SEL.SELECTION_BLOCK_KEY], spike={"spike_id": "aks-0002"})}))
        self.assertIn(SEL.REJECT_SPIKE_MALFORMED, result.codes)

    def test_cost_regression_monitor_has_three_outcomes(self):
        self.assertEqual(
            SEL.spike_cost_regression([], [30.0]).outcome, S.COULD_NOT_CHECK)
        self.assertEqual(
            SEL.spike_cost_regression([4.0], []).outcome, S.COULD_NOT_CHECK)
        self.assertEqual(
            SEL.spike_cost_regression([4.0, 5.0], [30.0, 45.0]).outcome, S.PASS)
        self.assertEqual(
            SEL.spike_cost_regression([31.0, 40.0], [30.0, 45.0]).outcome, S.FAIL)


# =============================================================================
# §8.4 — PROPOSAL_SKIPPED, the blacklist, and PLANNER_DEGRADED
# =============================================================================

class TestSkipJournalling(_JournalCase):

    def test_a_rejection_is_journaled_with_its_fingerprint_and_codes(self):
        screener = self.screener()
        result = screener.screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "domains": ["epyc-orchestrator/api"]}}), _context())
        self.assertFalse(result.admitted)
        self.assertIsNotNone(result.journal_event_id)
        entries = [e for e in self.journal.read_all()
                   if e.kind == J.KIND_PROPOSAL_SKIPPED]
        self.assertEqual(len(entries), 1)
        payload = entries[0].payload
        self.assertEqual(payload["fingerprint"], result.fingerprint)
        self.assertIn(SEL.REJECT_CROSSES_UNOWNED_DOMAIN, payload["detail"]["reason_codes"])
        self.assertIn("epyc-orchestrator/api", payload["reason"])

    def test_an_admitted_proposal_writes_nothing(self):
        screener = self.screener()
        result = screener.screen(_proposal(), _context())
        self.assertTrue(result.admitted, result.codes)
        self.assertEqual(
            [e for e in self.journal.read_all() if e.kind == J.KIND_PROPOSAL_SKIPPED], []
        )

    def test_a_repeated_fingerprint_auto_blacklists(self):
        screener = self.screener()
        bad = _proposal(**{SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}})
        screener.screen(bad, _context())
        self.assertEqual(screener.blacklist(), frozenset())
        screener.screen(_proposal(proposal_id="akp-0002", **{
            SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}}), _context())
        fingerprint = SEL.proposal_fingerprint(bad)
        self.assertIn(fingerprint, screener.blacklist())

    def test_a_blacklisted_fingerprint_is_rejected_even_when_otherwise_clean(self):
        screener = self.screener()
        bad = _proposal(**{SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}})
        screener.screen(bad, _context())
        screener.screen(_proposal(proposal_id="akp-0002", **{
            SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}}), _context())
        # Third attempt: the domain is now owned, so ONLY the blacklist can refuse it.
        third = _proposal(proposal_id="akp-0003", **{
            SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}})
        context = _context(owned_domains=frozenset({"nope/nope", "llama.cpp/ggml-cuda"}))
        result = screener.screen(third, context)
        self.assertIn(SEL.REJECT_FINGERPRINT_BLACKLISTED, result.codes)

    def test_the_blacklist_is_re_read_from_the_record_every_call(self):
        screener = self.screener()
        bad = _proposal(**{SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}})
        first = SEL.ProposalScreener(self.journal, campaign_id=CAMPAIGN)
        first.screen(bad, _context())
        first.screen(_proposal(proposal_id="akp-0002", **{
            SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}}), _context())
        # `screener` was constructed BEFORE those appends and holds no cache.
        self.assertIn(SEL.proposal_fingerprint(bad), screener.blacklist())

    def test_feedback_is_deterministically_ordered_and_carries_the_codes(self):
        screener = self.screener()
        for i in range(3):
            screener.screen(_proposal(proposal_id=f"akp-{i:04d}", **{
                SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}}), _context())
        screener.screen(_proposal(proposal_id="akp-9999", **{
            SEL.SELECTION_BLOCK_KEY: {"domains": ["other/other"],
                                      "mechanism": "some-other-mechanism"}}), _context())
        feedback = screener.history().feedback()
        self.assertEqual(feedback[0].count, 3)
        self.assertTrue(feedback[0].blacklisted)
        self.assertFalse(feedback[-1].blacklisted)
        self.assertIn(SEL.REJECT_CROSSES_UNOWNED_DOMAIN, feedback[0].codes)
        self.assertEqual(feedback, screener.history().feedback())

    def test_a_failed_append_is_a_failure_not_a_silent_discard(self):
        class _RefusingJournal(J.Journal):
            def append(self, *args, **kwargs):
                raise OSError("disk went away")

        journal_ = _RefusingJournal(self.root, campaign_id=CAMPAIGN)
        screener = SEL.ProposalScreener(journal_, campaign_id=CAMPAIGN)
        with self.assertRaises(SEL.SkipNotRecorded):
            screener.screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
                "domains": ["nope/nope"]}}), _context())

    def test_a_screener_refuses_another_campaigns_context(self):
        screener = self.screener()
        with self.assertRaises(ValueError):
            screener.screen(_proposal(), _context(campaign_id="ak-other-20260803"))


class TestPlannerHealth(_JournalCase):

    STOP_POLICY = {"max_consecutive_proposal_skips": 3}

    def _skip(self, screener, index: int) -> None:
        screener.screen(_proposal(proposal_id=f"akp-{index:04d}", **{
            SEL.SELECTION_BLOCK_KEY: {
                "domains": ["nope/nope"], "mechanism": f"mech-{index}"}}), _context())

    def test_a_run_of_skips_trips_planner_degraded(self):
        screener = self.screener()
        for i in range(2):
            self._skip(screener, i)
        self.assertIsNone(SEL.planner_health_stop_request(
            screener.history(), stop_policy=self.STOP_POLICY))
        self._skip(screener, 2)
        request = SEL.planner_health_stop_request(
            screener.history(), stop_policy=self.STOP_POLICY)
        self.assertIsNotNone(request)
        self.assertEqual(request.state, SM.PLANNER_DEGRADED)

    def test_the_stop_request_carries_evidence_the_machine_accepts(self):
        screener = self.screener()
        for i in range(3):
            self._skip(screener, i)
        request = SEL.planner_health_stop_request(
            screener.history(), stop_policy=self.STOP_POLICY)
        check = SM.check_stop_evidence(request.state, request.reason, request.detail)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_the_run_length_must_be_declared_not_invented(self):
        screener = self.screener()
        self._skip(screener, 0)
        with self.assertRaises(ValueError):
            SEL.planner_health_stop_request(screener.history(), stop_policy={})
        with self.assertRaises(ValueError):
            SEL.planner_health_stop_request(
                screener.history(), stop_policy={"max_consecutive_proposal_skips": 0})

    def test_an_admitted_proposal_breaks_the_run(self):
        screener = self.screener()
        for i in range(2):
            self._skip(screener, i)
        self.journal.append(J.KIND_PROPOSAL_RECORDED, _proposal(proposal_id="akp-5000"))
        self._skip(screener, 3)
        self.assertIsNone(SEL.planner_health_stop_request(
            screener.history(), stop_policy=self.STOP_POLICY))

    def test_the_machine_disposes_the_request(self):
        machine_root = str(Path(self._tmp.name) / "controller")
        machine = SM.ControllerStateMachine(
            journal_=self.journal, root=machine_root, campaign_id=CAMPAIGN)
        machine.transition(SM.DISCOVER, trigger="test", reason="bootstrap done")
        machine.transition(SM.SELECT_TARGET, trigger="test", reason="discovery done")
        screener = self.screener()
        for i in range(3):
            self._skip(screener, i)
        request = SEL.planner_health_stop_request(
            screener.history(), stop_policy=self.STOP_POLICY)
        machine.dispose_stop_request(request)
        self.assertEqual(machine.state, SM.PLANNER_DEGRADED)
        self.assertTrue(machine.is_stopped())


# =============================================================================
# §8.4 — cheap deterministic checks BEFORE metered drafting
# =============================================================================

class _FakeDrafter:
    """The only 'provider' in this suite: a counter that returns a fixed manifest.

    A test that would reach a real model fails the task, so the guard is proven
    against a fake and the assertion is on `calls`, not on any output.
    """

    def __init__(self, produce=None) -> None:
        self.calls = 0
        self._produce = produce

    def __call__(self, brief: SEL.DraftBrief) -> dict:
        self.calls += 1
        if self._produce is not None:
            return self._produce(brief)
        return _proposal()


def _brief(**over) -> SEL.DraftBrief:
    base = dict(
        seed_id="aks-seed-1",
        mechanism="mmvq-dispatch-threshold",
        hierarchy_layer="dispatcher",
        change_class="dispatcher",
        campaign_kind="dispatch",
        regime_identity={"backend": ("llama_gpu",), "quant": ("Q4_K",)},
        target_ops=("mul_mat_vec_q",),
        target_regimes=("decode_b1",),
        target_shape_digests=(DIGEST_A,),
        domains=("llama.cpp/ggml-cuda",),
        layer_skip_receipts=(SEL.LayerSkipReceipt.from_dict(_receipt()),),
        estimated_minutes=20.0,
        estimated_storage_gb=2.0,
        lane="gpu",
    )
    base.update(over)
    return SEL.DraftBrief(**base)


class TestPrescreenOrdering(unittest.TestCase):

    def test_a_clean_seed_gets_a_ticket_and_the_drafter_runs_once(self):
        outcome = SEL.prescreen(_brief(), _context(), blacklisted_fingerprints=frozenset())
        self.assertTrue(outcome.admitted, outcome.rejections)
        drafter = _FakeDrafter()
        guard = SEL.MeteredDraftGuard(drafter)
        guard.draft(_brief(), outcome.ticket)
        self.assertEqual(drafter.calls, 1)

    def test_a_blacklisted_seed_never_reaches_the_drafter(self):
        brief = _brief()
        fingerprint = SEL.mechanism_fingerprint(brief.facets())
        outcome = SEL.prescreen(
            brief, _context(), blacklisted_fingerprints=frozenset({fingerprint}))
        self.assertFalse(outcome.admitted)
        drafter = _FakeDrafter()
        guard = SEL.MeteredDraftGuard(drafter)
        with self.assertRaises(SEL.DraftingRefused):
            guard.draft(brief, outcome.ticket)
        self.assertEqual(drafter.calls, 0, "metered drafting ran before the cheap check")

    def test_a_receipted_negative_never_reaches_the_drafter(self):
        entry = SEL.LedgerEntry(
            entry_id="dnr-1", entry_class="HARD_CONSTRAINT",
            mechanism="mmvq-dispatch-threshold",
            match_dimensions={"quant": ("Q4_K",)}, reopen_when="never",
            receipt="67a433bf:mmvq.cu:538", anchor_commit=V8_COMMIT,
        )
        outcome = SEL.prescreen(_brief(), _context(ledger=(entry,)),
                                blacklisted_fingerprints=frozenset())
        self.assertFalse(outcome.admitted)
        self.assertIn(SEL.REJECT_REPEATS_RECEIPTED_NEGATIVE,
                      [r.code for r in outcome.rejections])
        drafter = _FakeDrafter()
        with self.assertRaises(SEL.DraftingRefused):
            SEL.MeteredDraftGuard(drafter).draft(_brief(), outcome.ticket)
        self.assertEqual(drafter.calls, 0)

    def test_an_unreceipted_layer_skip_never_reaches_the_drafter(self):
        outcome = SEL.prescreen(
            _brief(hierarchy_layer="new_kernel", layer_skip_receipts=()),
            _context(), blacklisted_fingerprints=frozenset())
        self.assertFalse(outcome.admitted)
        self.assertIn(SEL.REJECT_HIERARCHY_SKIP_UNRECEIPTED,
                      [r.code for r in outcome.rejections])

    def test_an_over_budget_seed_never_reaches_the_drafter(self):
        outcome = SEL.prescreen(
            _brief(estimated_minutes=10_000.0), _context(),
            blacklisted_fingerprints=frozenset())
        self.assertFalse(outcome.admitted)
        self.assertIn(SEL.REJECT_BUDGET_EXCEEDED, [r.code for r in outcome.rejections])

    def test_a_confirmation_shape_seed_never_reaches_the_drafter(self):
        outcome = SEL.prescreen(
            _brief(target_shape_digests=(DIGEST_CONF,)), _context(),
            blacklisted_fingerprints=frozenset())
        self.assertFalse(outcome.admitted)
        self.assertIn(SEL.REJECT_TARGETS_CONFIRMATION_SHAPE,
                      [r.code for r in outcome.rejections])

    def test_drafting_without_a_ticket_is_refused(self):
        drafter = _FakeDrafter()
        with self.assertRaises(SEL.DraftingRefused):
            SEL.MeteredDraftGuard(drafter).draft(_brief(), None)
        self.assertEqual(drafter.calls, 0)

    def test_a_ticket_for_another_seed_is_refused(self):
        outcome = SEL.prescreen(_brief(), _context(), blacklisted_fingerprints=frozenset())
        drafter = _FakeDrafter()
        with self.assertRaises(SEL.DraftingRefused):
            SEL.MeteredDraftGuard(drafter).draft(_brief(seed_id="aks-seed-2"), outcome.ticket)
        self.assertEqual(drafter.calls, 0)

    def test_a_brief_edited_after_screening_is_refused(self):
        outcome = SEL.prescreen(_brief(), _context(), blacklisted_fingerprints=frozenset())
        drafter = _FakeDrafter()
        with self.assertRaises(SEL.DraftingRefused):
            SEL.MeteredDraftGuard(drafter).draft(
                _brief(change_class="core_header"), outcome.ticket)
        self.assertEqual(drafter.calls, 0)

    def test_a_drafter_that_wanders_off_the_screened_mechanism_is_refused(self):
        def wander(brief):
            return _proposal(**{SEL.SELECTION_BLOCK_KEY: {
                "mechanism": "rewrite-the-whole-attention-stack",
                "hierarchy_layer": "alternate_engine"}})

        drafter = _FakeDrafter(produce=wander)
        outcome = SEL.prescreen(_brief(), _context(), blacklisted_fingerprints=frozenset())
        with self.assertRaises(SEL.DraftedProposalDiverged):
            SEL.MeteredDraftGuard(drafter).draft(_brief(), outcome.ticket)

    def test_a_prescreen_rejection_is_journaled_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_ = J.Journal(str(Path(tmp) / "j"), campaign_id=CAMPAIGN)
            journal_.initialize()
            screener = SEL.ProposalScreener(journal_, campaign_id=CAMPAIGN)
            brief = _brief()
            outcome = SEL.prescreen(
                brief, _context(),
                blacklisted_fingerprints=frozenset({
                    SEL.mechanism_fingerprint(brief.facets())}))
            entry = screener.record_prescreen_rejection(brief, outcome)
            self.assertEqual(entry.payload["detail"]["stage"], "prescreen")
            self.assertEqual(entry.payload["proposal_ref"], brief.seed_id)

    def test_recording_an_admitted_prescreen_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_ = J.Journal(str(Path(tmp) / "j"), campaign_id=CAMPAIGN)
            journal_.initialize()
            screener = SEL.ProposalScreener(journal_, campaign_id=CAMPAIGN)
            outcome = SEL.prescreen(_brief(), _context(),
                                    blacklisted_fingerprints=frozenset())
            with self.assertRaises(ValueError):
                screener.record_prescreen_rejection(_brief(), outcome)


# =============================================================================
# §8.4.1 — HARVEST / EXPLORE phases on marginal yield
# =============================================================================

def _observation(index: int, *, gain: float, spent: float = 1.0, skips: int = 0,
                 repeats: int = 0, deep: bool = False, anchor: bool = False):
    return SEL.YieldObservation(
        round_index=index, banked_gain=gain, budget_spent=spent,
        proposal_skipped_count=skips, repeated_fingerprint_count=repeats,
        receipt=f"ake-round-{index}", deep_lever_landed=deep, anchor_moved=anchor,
    )


class TestPhases(unittest.TestCase):

    def _calibration(self) -> SEL.YieldCalibration:
        return SEL.derive_yield_calibration(
            (0.10, 0.08, 0.06), derivation_id="cal-harvest-1")

    def test_the_floor_window_and_dwell_are_derived_from_the_samples(self):
        calibration = self._calibration()
        self.assertAlmostEqual(calibration.floor, 0.06)
        self.assertEqual(calibration.window_rounds, 3)
        self.assertEqual(calibration.min_dwell_rounds, 3)
        self.assertEqual(calibration.verify().outcome, S.PASS)

    def test_a_supplied_floor_cannot_survive_construction(self):
        with self.assertRaises(SEL.CalibrationTampered):
            SEL.YieldCalibration(
                floor=0.001, window_rounds=3, min_dwell_rounds=3,
                derivation_samples=(0.10, 0.08, 0.06), derivation_id="hand-built")

    def test_a_supplied_window_cannot_survive_construction(self):
        with self.assertRaises(SEL.CalibrationTampered):
            SEL.YieldCalibration(
                floor=0.06, window_rounds=1, min_dwell_rounds=1,
                derivation_samples=(0.10, 0.08, 0.06), derivation_id="hand-built")

    def test_a_floor_needs_material(self):
        with self.assertRaises(SEL.InsufficientYieldMaterial):
            SEL.derive_yield_calibration((0.10,), derivation_id="too-thin")
        with self.assertRaises(SEL.InsufficientYieldMaterial):
            SEL.derive_yield_calibration((0.10, 0.0), derivation_id="zero-yield")

    def test_a_deep_lever_landing_enters_harvest_immediately(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_EXPLORE, phase_started_round=0,
            observations=[_observation(0, gain=0.0), _observation(1, gain=0.5, deep=True)],
            calibration=self._calibration(),
        )
        self.assertEqual(decision.phase, SEL.PHASE_HARVEST)
        self.assertTrue(decision.changed)
        self.assertEqual(decision.trigger, SEL.TRIGGER_DEEP_LEVER)

    def test_an_anchor_move_enters_harvest_immediately(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_EXPLORE, phase_started_round=0,
            observations=[_observation(0, gain=0.01, anchor=True)],
            calibration=self._calibration(),
        )
        self.assertEqual(decision.phase, SEL.PHASE_HARVEST)
        self.assertEqual(decision.trigger, SEL.TRIGGER_ANCHOR_MOVE)

    def test_the_minimum_dwell_blocks_a_switch(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=2,
            observations=[_observation(i, gain=0.001) for i in range(4)],
            calibration=self._calibration(),
        )
        self.assertEqual(decision.phase, SEL.PHASE_HARVEST)
        self.assertEqual(decision.trigger, SEL.TRIGGER_DWELL)

    def test_an_incomplete_window_blocks_a_switch(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=[_observation(i, gain=0.001) for i in range(2)],
            calibration=SEL.derive_yield_calibration(
                (0.10, 0.08, 0.06, 0.05), derivation_id="cal-4"),
        )
        self.assertEqual(decision.trigger, SEL.TRIGGER_WINDOW)
        self.assertFalse(decision.changed)

    def test_decay_across_the_full_window_switches_to_explore(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=[_observation(i, gain=0.001) for i in range(4)],
            calibration=self._calibration(),
        )
        self.assertEqual(decision.phase, SEL.PHASE_EXPLORE)
        self.assertTrue(decision.changed)
        self.assertEqual(decision.trigger, SEL.TRIGGER_YIELD_DECAY)

    def test_one_round_above_the_floor_holds_the_phase(self):
        observations = [_observation(i, gain=0.001) for i in range(3)]
        observations[1] = _observation(1, gain=0.5)
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=observations, calibration=self._calibration(),
        )
        self.assertEqual(decision.phase, SEL.PHASE_HARVEST)
        self.assertEqual(decision.trigger, SEL.TRIGGER_YIELD_HOLDING)

    def test_falling_yield_with_rising_skips_is_degraded_not_explore(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=[
                _observation(0, gain=0.005, skips=1),
                _observation(1, gain=0.003, skips=4),
                _observation(2, gain=0.001, skips=9),
            ],
            calibration=self._calibration(),
        )
        self.assertEqual(decision.phase, SEL.PHASE_HARVEST)
        self.assertEqual(decision.trigger, SEL.TRIGGER_DEGRADED)
        self.assertIsNotNone(decision.stop_request)
        self.assertEqual(decision.stop_request.state, SM.PLANNER_DEGRADED)

    def test_a_repeated_fingerprint_alone_is_degraded_not_explore(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=[
                _observation(0, gain=0.005, repeats=1),
                _observation(1, gain=0.003),
                _observation(2, gain=0.001),
            ],
            calibration=self._calibration(),
        )
        self.assertEqual(decision.trigger, SEL.TRIGGER_DEGRADED)

    def test_the_degraded_stop_request_is_accepted_by_the_machine(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=[
                _observation(0, gain=0.005, skips=1),
                _observation(1, gain=0.003, skips=4),
                _observation(2, gain=0.001, skips=9),
            ],
            calibration=self._calibration(),
        )
        request = decision.stop_request
        check = SM.check_stop_evidence(request.state, request.reason, request.detail)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_explore_does_not_flip_back_on_yield_alone(self):
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_EXPLORE, phase_started_round=0,
            observations=[_observation(i, gain=0.5) for i in range(4)],
            calibration=self._calibration(),
        )
        self.assertEqual(decision.phase, SEL.PHASE_EXPLORE)
        self.assertFalse(decision.changed)

    def test_a_round_that_spent_nothing_has_no_yield(self):
        with self.assertRaises(ValueError):
            _observation(0, gain=0.1, spent=0.0)

    def test_observations_must_be_ordered(self):
        with self.assertRaises(ValueError):
            SEL.decide_phase(
                current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
                observations=[_observation(1, gain=0.1), _observation(0, gain=0.1)],
                calibration=self._calibration(),
            )

    def test_an_empty_observation_set_is_a_guess(self):
        with self.assertRaises(ValueError):
            SEL.decide_phase(
                current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
                observations=[], calibration=self._calibration(),
            )


# =============================================================================
# §8.4 ranking, §8.4.1 reserved arm
# =============================================================================

def _result(pid: str, *, eig: float, value: float, arm: str = SEL.ARM_INCREMENTAL,
            tier: int = 0, admitted: bool = True) -> SEL.ScreenResult:
    return SEL.ScreenResult(
        proposal_id=pid, fingerprint=S.content_hash(pid), admitted=admitted,
        rejections=() if admitted else (SEL.Rejection(
            code=SEL.REJECT_BUDGET_EXCEEDED, reason="over budget"),),
        checks={}, ledger_matches=(), excluded_cells=(), information_gain=eig,
        performance_value=value, arm=arm, tier_cost_rank=tier,
        oracle_coverage_basis="declared_target_ops",
    )


class TestRankingAndArms(unittest.TestCase):

    def test_information_gain_ranks_before_performance_value(self):
        ranked = SEL.rank_proposals(
            [_result("low-eig-high-value", eig=0.1, value=0.9),
             _result("high-eig-low-value", eig=0.9, value=0.01)],
            phase=SEL.PHASE_HARVEST,
        )
        self.assertEqual(ranked[0].proposal_id, "high-eig-low-value")

    def test_value_breaks_an_eig_tie(self):
        ranked = SEL.rank_proposals(
            [_result("a", eig=0.5, value=0.01), _result("b", eig=0.5, value=0.20)],
            phase=SEL.PHASE_HARVEST,
        )
        self.assertEqual(ranked[0].proposal_id, "b")

    def test_the_order_is_deterministic_under_a_full_tie(self):
        first = SEL.rank_proposals(
            [_result("a", eig=0.5, value=0.1), _result("b", eig=0.5, value=0.1)],
            phase=SEL.PHASE_HARVEST)
        second = SEL.rank_proposals(
            [_result("b", eig=0.5, value=0.1), _result("a", eig=0.5, value=0.1)],
            phase=SEL.PHASE_HARVEST)
        self.assertEqual([r.proposal_id for r in first], [r.proposal_id for r in second])

    def test_a_rejected_proposal_never_ranks_however_high_its_eig(self):
        ranked = SEL.rank_proposals(
            [_result("rejected", eig=1.0, value=1.0, admitted=False),
             _result("admitted", eig=0.01, value=0.0)],
            phase=SEL.PHASE_HARVEST,
        )
        self.assertEqual([r.proposal_id for r in ranked], ["admitted"])

    def test_harvest_prioritises_incremental_work_and_cheap_tiers(self):
        ranked = SEL.rank_proposals(
            [_result("arch", eig=0.99, value=0.9, arm=SEL.ARM_ARCHITECTURAL),
             _result("incr-t2", eig=0.90, value=0.5, tier=1),
             _result("incr-t1", eig=0.10, value=0.1)],
            phase=SEL.PHASE_HARVEST,
        )
        self.assertEqual([r.proposal_id for r in ranked], ["incr-t1", "incr-t2", "arch"])

    def test_explore_prioritises_architectural_work(self):
        ranked = SEL.rank_proposals(
            [_result("incr", eig=0.99, value=0.9),
             _result("arch", eig=0.10, value=0.1, arm=SEL.ARM_ARCHITECTURAL)],
            phase=SEL.PHASE_EXPLORE,
        )
        self.assertEqual([r.proposal_id for r in ranked], ["arch", "incr"])

    def test_budget_partition_reserves_only_for_a_declared_campaign(self):
        self.assertEqual(SEL.partition_budget(100.0, None).architectural_minutes, 0.0)
        arms = SEL.partition_budget(100.0, _campaign(fraction=0.25))
        self.assertAlmostEqual(arms.architectural_minutes, 25.0)
        self.assertAlmostEqual(arms.incremental_minutes, 75.0)

    def test_the_reserved_arm_is_not_available_to_incremental_work(self):
        arms = SEL.ArmBudget(incremental_minutes=1.0, architectural_minutes=100.0)
        decision = SEL.select_next(
            [_result("incr", eig=0.99, value=0.9),
             _result("arch", eig=0.01, value=0.01, arm=SEL.ARM_ARCHITECTURAL)],
            phase=SEL.PHASE_HARVEST, arm_budget=arms,
            cost_minutes_by_proposal={"incr": 40.0, "arch": 40.0},
        )
        self.assertIsNotNone(decision.chosen)
        self.assertEqual(decision.chosen.proposal_id, "arch")
        self.assertEqual(decision.arm, SEL.ARM_ARCHITECTURAL)

    def test_the_lowest_ranked_architectural_proposal_still_gets_selected(self):
        """AK-D31: EIG-first ranking starves high-variance work by arithmetic."""
        arms = SEL.ArmBudget(incremental_minutes=0.0, architectural_minutes=100.0)
        decision = SEL.select_next(
            [_result(f"incr-{i}", eig=0.99, value=0.9) for i in range(5)]
            + [_result("arch", eig=0.001, value=0.0, arm=SEL.ARM_ARCHITECTURAL)],
            phase=SEL.PHASE_HARVEST, arm_budget=arms,
            cost_minutes_by_proposal={
                **{f"incr-{i}": 10.0 for i in range(5)}, "arch": 10.0},
        )
        self.assertEqual(decision.chosen.proposal_id, "arch")
        self.assertEqual(decision.ranked[-1].proposal_id, "arch")

    def test_nothing_is_selected_when_no_arm_can_pay(self):
        arms = SEL.ArmBudget(incremental_minutes=1.0, architectural_minutes=1.0)
        decision = SEL.select_next(
            [_result("incr", eig=0.9, value=0.9)], phase=SEL.PHASE_HARVEST,
            arm_budget=arms, cost_minutes_by_proposal={"incr": 40.0},
        )
        self.assertIsNone(decision.chosen)
        self.assertIn("reserved", decision.reason)

    def test_an_unpriced_candidate_is_refused(self):
        with self.assertRaises(ValueError):
            SEL.select_next(
                [_result("incr", eig=0.9, value=0.9)], phase=SEL.PHASE_HARVEST,
                arm_budget=SEL.ArmBudget(10.0, 0.0), cost_minutes_by_proposal={},
            )

    def test_the_decision_detail_drives_a_real_machine_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_ = J.Journal(str(Path(tmp) / "j"), campaign_id=CAMPAIGN)
            journal_.initialize()
            machine = SM.ControllerStateMachine(
                journal_=journal_, root=str(Path(tmp) / "c"), campaign_id=CAMPAIGN)
            machine.transition(SM.DISCOVER, trigger="test", reason="bootstrap done")
            machine.transition(SM.SELECT_TARGET, trigger="test", reason="discovery done")
            decision = SEL.select_next(
                [_result("incr", eig=0.9, value=0.9)], phase=SEL.PHASE_HARVEST,
                arm_budget=SEL.ArmBudget(100.0, 0.0),
                cost_minutes_by_proposal={"incr": 40.0},
            )
            machine.transition(
                SM.PROPOSE, trigger="selection", reason=decision.reason,
                detail=decision.transition_detail(),
            )
            self.assertEqual(machine.state, SM.PROPOSE)
            ledger = machine.ledger.read().transitions
            self.assertEqual(ledger[-1].detail["chosen_proposal_id"], "incr")


# =============================================================================
# Cross-cutting: the module decides nothing an LLM said, and mutates no input
# =============================================================================

class TestDeterminismAndPurity(unittest.TestCase):

    def test_screening_does_not_mutate_the_proposal(self):
        proposal = _proposal()
        before = copy.deepcopy(proposal)
        _screen(proposal)
        self.assertEqual(proposal, before)

    def test_screening_is_repeatable(self):
        proposal = _proposal()
        context = _context()
        first = SEL.screen_proposal(proposal, context, blacklisted_fingerprints=frozenset())
        second = SEL.screen_proposal(proposal, context, blacklisted_fingerprints=frozenset())
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_the_planners_declared_value_is_never_the_ranked_value(self):
        """The planner may claim any worth; the controller recomputes from the receipt."""
        proposal = _proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "expected_end_to_end_gain": 0.29, "claimed_value": 0.99}})
        result = _screen(proposal)
        self.assertTrue(result.admitted, result.codes)
        self.assertAlmostEqual(result.performance_value, 0.29)

    def test_value_is_capped_by_the_measured_ceiling(self):
        proposal = _proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "expected_end_to_end_gain": 0.30}})
        result = _screen(proposal, context=_context(wall_share_receipts={"wsr-1": 0.30}))
        self.assertAlmostEqual(result.performance_value, 0.30)

    def test_every_rejection_code_is_declared(self):
        for code in SEL.REJECTION_CODES:
            SEL.Rejection(code=code, reason="declared")
        with self.assertRaises(ValueError):
            SEL.Rejection(code="INVENTED_CODE", reason="nope")

    def test_a_rejection_must_carry_a_reason(self):
        with self.assertRaises(ValueError):
            SEL.Rejection(code=SEL.REJECT_BUDGET_EXCEEDED, reason="   ")

    def test_context_refuses_an_undeclared_budget(self):
        with self.assertRaises(ValueError):
            _context(budget_remaining={"wall_minutes": 1.0})

    def test_context_refuses_an_unknown_phase(self):
        with self.assertRaises(ValueError):
            _context(phase="COAST")


class TestCrossModuleAgreement(_JournalCase):
    """One meaning of "repeated", across two AK4 modules.

    `planner.assess_repetition` is the pure in-memory fold; `read_skip_history`
    is the journal-backed one. They must never disagree about what auto-blacklists,
    because a blacklist that depends on which module you asked is not a blacklist.
    """

    def test_the_blacklist_agrees_with_the_pure_sibling_fold(self):
        screener = self.screener()
        for index, mechanism in enumerate(("mech-a", "mech-a", "mech-b")):
            screener.screen(_proposal(proposal_id=f"akp-{index:04d}", **{
                SEL.SELECTION_BLOCK_KEY: {
                    "domains": ["nope/nope"], "mechanism": mechanism}}), _context())
        history = screener.history()
        pure = PL.assess_repetition(
            [r.fingerprint for r in history.records], degraded_run=99)
        self.assertEqual(history.blacklisted, pure.blacklisted)
        self.assertEqual(history.counts, pure.counts)

    def test_the_skipped_payload_is_the_shared_planner_shape(self):
        screener = self.screener()
        screener.screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "domains": ["nope/nope"]}}), _context())
        payload = [e for e in self.journal.read_all()
                   if e.kind == J.KIND_PROPOSAL_SKIPPED][0].payload
        for key in ("proposal_ref", "reason", "fingerprint", "detail"):
            self.assertIn(key, payload)
        self.assertIn("reason_codes", payload["detail"])


class TestRedTeamRegressions(unittest.TestCase):
    """Defects found by adversarial review on 2026-08-03, each closed here.

    Every one of these passed the original suite. They are grouped because they
    share one shape: a check that LOOKED like a check but could be satisfied by
    choosing an encoding, deleting a field, or handing it a number it could not
    compare.
    """

    # ---- the ledger compared two encodings of the same value ----------------

    def test_a_suppression_keyed_on_a_scalar_facet_still_matches(self):
        """`change_class` and `hierarchy_layer` are SCALARS in the facet map and
        tuples in the entry. Canonicalising only one side made every suppression
        keyed on them a silent no-op — the worst possible failure for a ledger
        whose entries are never re-tested."""
        entry = SEL.LedgerEntry(
            entry_id="dnr-change-class", entry_class="HARD_CONSTRAINT",
            mechanism="mmvq-dispatch-threshold",
            match_dimensions={"change_class": ("dispatcher",)},
            reopen_when="never", receipt="commit:path:line", anchor_commit=V8_COMMIT)
        result = _screen(_proposal(), _context(ledger=(entry,)))
        self.assertFalse(result.admitted)
        self.assertIn(SEL.REJECT_REPEATS_RECEIPTED_NEGATIVE, result.codes)

    def test_a_scalar_regime_identity_does_not_escape_a_receipted_negative(self):
        """The actor chose the encoding: declaring `{"quant": "Q4_K"}` instead of
        `{"quant": ["Q4_K"]}` walked past a MATCHED_NEGATIVE that the list form
        was rejected by."""
        entry = SEL.LedgerEntry(
            entry_id="dnr-1", entry_class="MATCHED_NEGATIVE",
            mechanism="mmvq-dispatch-threshold",
            match_dimensions={"backend": ("llama_gpu",), "quant": ("Q4_K",)},
            reopen_when="mmvq.cu changes", receipt="commit:path:line",
            anchor_commit=V8_COMMIT)
        context = _context(ledger=(entry,))
        as_lists = _screen(_proposal(), context)
        as_scalars = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {"regime_identity": {
            "backend": "llama_gpu", "phase": "decode", "quant": "Q4_K", "batch": 1,
        }}}), context)
        self.assertFalse(as_lists.admitted)
        self.assertFalse(
            as_scalars.admitted,
            "a receipted negative that a re-encoding escapes is not a suppression")
        self.assertEqual(as_lists.codes, as_scalars.codes)

    def test_the_int_one_does_not_match_the_string_one(self):
        """The documented property (`1 must not match "1"`) held for collections
        and failed for scalars, because scalars skipped `canonical_json`."""
        facets = SEL.mechanism_facets(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "regime_identity": {"batch": 1}}}))
        common = dict(
            entry_id="dnr-batch", entry_class="HARD_CONSTRAINT",
            mechanism="mmvq-dispatch-threshold", reopen_when="never",
            receipt="commit:path:line", anchor_commit=V8_COMMIT)
        as_text = SEL.LedgerEntry(match_dimensions={"batch": ("1",)}, **common)
        as_int = SEL.LedgerEntry(match_dimensions={"batch": (1,)}, **common)
        kwargs = dict(anchor_commit=V8_COMMIT, satisfied_reopen_predicates=frozenset())
        self.assertEqual(SEL.match_ledger(facets, (as_text,), **kwargs), ())
        self.assertTrue(SEL.match_ledger(facets, (as_int,), **kwargs)[0].rejects)

    # ---- checks that passed by deleting what they inspect -------------------

    def test_an_undeclared_conceptual_change_count_is_not_a_licence(self):
        """`conceptual_change_count: 3` was rejected and OMITTING it passed, so
        invariant 13 was enforced only against a planner that confessed."""
        confessed = _screen(_proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "conceptual_change_count": 3}}))
        self.assertIn(SEL.REJECT_MULTIPLE_CONCEPTUAL_CHANGES, confessed.codes)
        silent = _proposal()
        silent[SEL.SELECTION_BLOCK_KEY].pop("conceptual_change_count")
        result = _screen(silent)
        self.assertFalse(result.admitted, "a check passed by deleting its input")
        self.assertIn(SEL.REJECT_UNVERIFIABLE, result.codes)
        self.assertEqual(result.checks["conceptual_scope"].outcome, S.COULD_NOT_CHECK)

    def test_a_fabricated_gap_receipt_does_not_license_a_skip(self):
        """`layer_ceiling < measured_gap` is arithmetic only if its operands are
        measured. `gap_receipt_id` was validated as a non-empty string and never
        resolved, so a receipt could name a gap that exists nowhere."""
        receipt = _receipt()
        receipt["gap_receipt_id"] = "wsr-i-made-this-up"
        check = SEL.check_layer_skip(
            "dispatcher", (SEL.LayerSkipReceipt.from_dict(receipt),),
            anchor_commit=V8_COMMIT, known_event_ids=frozenset({"ake-profile-1"}))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("do not resolve", " ".join(check.reasons))

    def test_a_jump_to_the_top_of_the_hierarchy_needs_receipts_that_resolve(self):
        proposal = _proposal(**{SEL.SELECTION_BLOCK_KEY: {
            "hierarchy_layer": "alternate_engine",
            "layer_skip_receipts": [
                {"layer": layer, "measured_gap": 1e6, "layer_ceiling": 0.0,
                 "gap_receipt_id": "wsr-does-not-exist",
                 "evidence_event_ids": ["ake-profile-1"], "anchor_commit": V8_COMMIT,
                 "basis": "trust me"}
                for layer in SEL.HIERARCHY[:-1]]}})
        result = _screen(proposal)
        self.assertFalse(result.admitted)
        self.assertIn(SEL.REJECT_HIERARCHY_SKIP_UNRECEIPTED, result.codes)

    def test_contradicting_receipts_for_one_layer_license_nothing(self):
        """Two receipts for one layer used to be a last-wins map, so a submission
        could overwrite its own failing ceiling by appending a better one."""
        failing = SEL.LayerSkipReceipt.from_dict(
            _receipt(layer="placement_and_launch_config", ceiling=0.99, gap=0.20))
        passing = SEL.LayerSkipReceipt.from_dict(
            _receipt(layer="placement_and_launch_config", ceiling=0.001, gap=0.20))
        known = frozenset({"ake-profile-1"})
        for order in ((failing, passing), (passing, failing)):
            check = SEL.check_layer_skip(
                "dispatcher", order, anchor_commit=V8_COMMIT, known_event_ids=known)
            self.assertEqual(check.outcome, S.FAIL)
            self.assertIn("contradicts itself", " ".join(check.reasons))

    # ---- a number the gate could not compare --------------------------------

    def test_a_seed_cost_the_budget_gate_cannot_compare_is_refused(self):
        """`prescreen` compares with `>`; NaN loses every comparison, so a NaN
        seed cost was admitted against a budget of zero."""
        for value in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                _brief(estimated_minutes=value)
            with self.assertRaises(ValueError):
                _brief(estimated_storage_gb=value)

    def test_an_arm_budget_must_be_a_budget(self):
        for arms in ((float("nan"), 1.0), (-1.0, 1.0), (1.0, float("nan"))):
            with self.assertRaises(ValueError):
                SEL.ArmBudget(*arms)

    # ---- the ticket bound only the labels -----------------------------------

    def test_a_drafter_that_moves_the_target_is_refused(self):
        """The ticket bound the brief's fingerprint but only four facets were
        compared, so a drafter could screen `mul_mat_vec_q` at decode and draft
        `flash_attn_ext` at prefill under the same ticket."""
        def wander(brief):
            return _proposal(target={
                "regimes": ["prefill_b64"], "shapes": [SHAPE_B],
                "ops": ["mul_mat_q", "flash_attn_ext"], "models": ["glm-5.2"]})

        drafter = _FakeDrafter(produce=wander)
        outcome = SEL.prescreen(_brief(), _context(), blacklisted_fingerprints=frozenset())
        with self.assertRaises(SEL.DraftedProposalDiverged):
            SEL.MeteredDraftGuard(drafter).draft(_brief(), outcome.ticket)

    def test_a_drafter_that_moves_the_regime_is_refused(self):
        def wander(brief):
            return _proposal(**{SEL.SELECTION_BLOCK_KEY: {"regime_identity": {
                "backend": ["llama_gpu"], "phase": ["decode"], "quant": ["IQ2_XXS"],
                "batch": [1]}}})

        drafter = _FakeDrafter(produce=wander)
        outcome = SEL.prescreen(_brief(), _context(), blacklisted_fingerprints=frozenset())
        with self.assertRaises(SEL.DraftedProposalDiverged):
            SEL.MeteredDraftGuard(drafter).draft(_brief(), outcome.ticket)

    def test_a_draft_may_say_more_about_its_regime_than_the_seed_did(self):
        """The guard must not forbid the compliant path: the fixture brief names
        two regime dimensions and the drafted manifest names four."""
        drafter = _FakeDrafter()
        outcome = SEL.prescreen(_brief(), _context(), blacklisted_fingerprints=frozenset())
        SEL.MeteredDraftGuard(drafter).draft(_brief(), outcome.ticket)
        self.assertEqual(drafter.calls, 1)

    # ---- planner health --------------------------------------------------

    def test_another_campaigns_progress_does_not_clear_this_run(self):
        """`read_skip_history` reset the trailing run on ANY admitted proposal,
        campaign or not, so a neighbour's progress on a shared journal silenced
        this campaign's PLANNER_DEGRADED evidence."""
        with tempfile.TemporaryDirectory() as tmp:
            journal_ = J.Journal(str(Path(tmp) / "j"), campaign_id=None)
            journal_.initialize()
            screener = SEL.ProposalScreener(journal_, campaign_id=CAMPAIGN)
            for index in range(2):
                screener.screen(_proposal(proposal_id=f"akp-bad{index}", **{
                    SEL.SELECTION_BLOCK_KEY: {"domains": ["nope/nope"]}}), _context())
            self.assertEqual(screener.history().trailing_run, 2)

            other = "ak-neighbour-20260101"
            journal_.append(
                J.KIND_PROPOSAL_RECORDED,
                _proposal(proposal_id="akp-other", campaign_id=other),
                campaign_id=other, record_id="akp-other")
            self.assertEqual(
                screener.history().trailing_run, 2,
                "a neighbouring campaign's admitted proposal is not evidence that "
                "THIS planner recovered")

            journal_.append(
                J.KIND_PROPOSAL_RECORDED,
                _proposal(proposal_id="akp-mine"),
                campaign_id=CAMPAIGN, record_id="akp-mine")
            self.assertEqual(screener.history().trailing_run, 0)

    def test_a_flat_zero_yield_with_rising_skips_is_degraded_not_explore(self):
        """The commonest broken searcher banks NOTHING, so its marginal yield is
        flat zero and never strictly falls. The strict test routed exactly that
        case to EXPLORE — the §8.10 conflation this module exists to prevent."""
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=[
                _observation(0, gain=0.0, skips=1),
                _observation(1, gain=0.0, skips=5),
                _observation(2, gain=0.0, skips=9),
            ],
            calibration=SEL.derive_yield_calibration(
                (0.10, 0.08, 0.06), derivation_id="cal-harvest-1"),
        )
        self.assertEqual(decision.trigger, SEL.TRIGGER_DEGRADED)
        self.assertEqual(decision.phase, SEL.PHASE_HARVEST)
        self.assertIsNotNone(decision.stop_request)
        self.assertEqual(decision.stop_request.state, SM.PLANNER_DEGRADED)
        check = SM.check_stop_evidence(
            decision.stop_request.state, decision.stop_request.reason,
            decision.stop_request.detail)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_a_decaying_region_with_a_quiet_planner_still_explores(self):
        """The counterpart: widening the degradation test must not swallow the
        plateau case it is supposed to be distinguished from."""
        decision = SEL.decide_phase(
            current_phase=SEL.PHASE_HARVEST, phase_started_round=0,
            observations=[_observation(i, gain=0.001) for i in range(4)],
            calibration=SEL.derive_yield_calibration(
                (0.10, 0.08, 0.06), derivation_id="cal-harvest-1"),
        )
        self.assertEqual(decision.phase, SEL.PHASE_EXPLORE)
        self.assertIsNone(decision.stop_request)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
