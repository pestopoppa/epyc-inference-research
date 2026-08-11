#!/usr/bin/env python3
"""test_do_not_repeat.py — the regression barrier for the memory-update plane (§19.2).

WHY THIS FILE EXISTS
--------------------
`hypotheses.py:261` said the do-not-repeat ledger belonged to a plane that did not
exist, so `check_do_not_repeat()` was a correct guard wired to nothing. This suite
holds the plane that now feeds it to the properties that make it safe to feed:

  * **the six classes fold from the record**, each from a realistic journal, and the
    two that reject reject while the four that are advisory leave the question OPEN —
    asserted through `hypotheses.check_do_not_repeat()` itself, so the two modules
    cannot drift into two opinions about what closes a question.
  * **matching is structural.** A reworded restatement of a tried idea still matches;
    two genuinely different ideas about one function in one regime do not. The statement
    is never read, and that is proved from the objects (`MatchQuery` has no field for
    it) and behaviourally (identical matches for two statements with no words in
    common), with an anti-vacuous control so "matches nothing either way" cannot pass.
  * **a moved anchor supersedes rather than rejects.** An idea that failed on v7 may
    win on v8; a stale anchor must not close it forever. All three outcomes of
    `AnchorIdentity.identity_matches` are exercised, COULD_NOT_CHECK included.
  * **a contended run is no run.** The 2026-08-04 A/A destroyed mid-flight by seven
    llama-servers folds to CONFOUNDED_RESULT and leaves the question open — it does
    NOT become a negative result.
  * **an empty ledger rejects nothing**, and a ledger with no current anchor rejects
    nothing on measurement grounds.

Every guard here also has a COMPLIANT-PATH CONTROL: a matcher that matched nothing, or
a fold that emitted nothing, would satisfy most of the assertions above vacuously, so
each one is paired with a case that must still get through.

NO inference, NO benchmark, NO build, NO model call, NO process, NO file I/O — the fold
is a pure function of records held in memory.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_do_not_repeat.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_do_not_repeat.py
"""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.controller import do_not_repeat as D  # noqa: E402
from autokernel.controller import shared as FP  # noqa: E402  (was fingerprint;
# only selection_block was ever used and it now lives in shared.py)
from autokernel.controller import hypotheses as H  # noqa: E402
from autokernel.controller import shared as SEL  # noqa: E402  (was selection;
# LEDGER_DIMENSIONS describes what the LEDGER keys on and now lives with it)
from autokernel.evaluator import api as EV  # noqa: E402

CAMPAIGN = "ak-llama_gpu-decode-20260803"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"


def _sha(seed: str) -> str:
    return S.content_hash({"seed": seed})


def _representation_contract() -> dict:
    contract = {
        "vocabulary": {
            "regimes": ["decode", "prefill"],
            "surfaces": ["rms_norm"],
            "outcomes": ["throughput_gain", "non_inferiority"],
            "contradictions": ["counter_did_not_move"],
        },
        "vocabulary_source_receipts": ["rcpt-vocabulary-g15"],
        "considered_alternatives": ["fuse_norm", "dispatcher_only"],
        "excluded_alternatives": [],
        "empirical_demand": {
            "receipt_id": "rcpt-demand-g15", "weights_sha256": _sha("demand-g15"),
        },
        "abstraction_construction_cost": {
            "value": 2, "unit": "typed_facts", "receipt_id": "rcpt-cost-g15",
        },
        "canonical_encoding": {
            "encoding_id": "ak-representation-json/v1",
            "schema_sha256": _sha("representation-schema-v1"),
        },
        "semantics_preserving_recoding_fixture_ids": ["ak-recode-g15-renamed"],
    }
    contract["frame_sha256"] = S.representation_frame_sha256(contract)
    return contract


ANCHOR_V8 = EV.AnchorIdentity(
    source_commit=V8_COMMIT,
    binary_sha256=_sha("anchor-binary-v8"),
    linkage_sha256=_sha("anchor-linkage-v8"),
    measurement_event_ids=("ake-20260801-0009",),
)
ANCHOR_V7 = EV.AnchorIdentity(
    source_commit=V7_COMMIT,
    binary_sha256=_sha("anchor-binary-v7"),
    linkage_sha256=_sha("anchor-linkage-v7"),
    measurement_event_ids=("ake-20260721-0003",),
)

#: The operator's worked example from §8.4.0, plus the two keys that make it
#: matchable: a mechanism (what change is being made) and the op it is made to.
G15_REGIME = {
    "backend": "llama_gpu",
    "phase": "decode",
    "batch_band": "b128",
    "mechanism": "elementwise_norm_fusion",
    "ops": ["ggml_cuda_op_rms_norm"],
}
G15_STATEMENT = (
    "G15's elementwise/norm cluster is where the B=128 decode time is, and fusing it "
    "lands >= 15%"
)
G15_FALSIFIER = "a current wall-share map shows the cluster under 20%"


# =============================================================================
# Fixtures — records shaped exactly like the ones the loop writes
# =============================================================================

def _entry(kind: str, payload: dict, *, seq: int, event_id: str,
           record_id=None) -> J.JournalEntry:
    return J.JournalEntry(
        event_id=event_id, seq=seq, kind=kind, campaign_id=CAMPAIGN,
        record_id=record_id, written_at="2026-08-04T09:00:00.000000Z",
        payload=payload,
    )


def _campaign_entry(seq: int = 1, backend: str = "llama_gpu") -> J.JournalEntry:
    return _entry(
        J.KIND_CAMPAIGN_OPENED,
        {"schema": S.SCHEMA_CAMPAIGN, "campaign_id": CAMPAIGN, "backend": backend},
        seq=seq, event_id="akj-000000000001-campaign", record_id=CAMPAIGN,
    )


def _proposal_payload(
    proposal_id: str = "akp-20260803-0001",
    *,
    mechanism: str = "elementwise_norm_fusion",
    ops=("ggml_cuda_op_rms_norm",),
    regime_identity=None,
    change_class: str = "fusion",
    conceptual_change: str = "fuse the elementwise/norm cluster into one kernel",
) -> dict:
    return {
        "schema": S.SCHEMA_PROPOSAL,
        "proposal_id": proposal_id,
        "campaign_id": CAMPAIGN,
        # §7.1's planner-authored block: where `fingerprint.mechanism_facets` and
        # `selection.match_ledger` both read structural identity from.
        FP.SELECTION_BLOCK_KEY: {
            "mechanism": mechanism,
            "hierarchy_layer": "kernel",
            "regime_identity": dict(regime_identity or {"batch": ["b128"]}),
        },
        "change_class": change_class,
        "hypothesis": "prose the matcher must never read",
        "representation_contract": _representation_contract(),
        "narrative": "more prose the matcher must never read",
        "target": {"regimes": ["decode"], "ops": list(ops), "shapes": [], "models": []},
        "change": {
            "conceptual_change": conceptual_change,
            "files_and_symbols": ["ggml-cuda/norm.cu:rms_norm_f32"],
        },
    }


def _proposal_entry(seq: int = 2, **over) -> J.JournalEntry:
    payload = _proposal_payload(**over)
    return _entry(
        J.KIND_PROPOSAL_RECORDED, payload, seq=seq,
        event_id=f"akj-{seq:012d}-proposal", record_id=payload["proposal_id"],
    )


def _candidate_entry(seq: int = 3, *, candidate_id="akc-20260803-0001",
                     proposal_id="akp-20260803-0001") -> J.JournalEntry:
    return _entry(
        J.KIND_CANDIDATE_RECORDED,
        {"schema": S.SCHEMA_CANDIDATE, "candidate_id": candidate_id,
         "campaign_id": CAMPAIGN, "proposal_id": proposal_id},
        seq=seq, event_id=f"akj-{seq:012d}-candidate", record_id=candidate_id,
    )


def _event_payload(
    event_id: str = "ake-20260803-0001",
    *,
    status: str = "fail",
    anchor: EV.AnchorIdentity = ANCHOR_V8,
    integrity_flags=(),
    machine_subset: str = "full",
    co_residency: str = "single",
    candidate_id: str = "akc-20260803-0001",
    tier: str = "T1",
) -> dict:
    payload = {
        "schema": S.SCHEMA_EVALUATION_EVENT_V3,
        "event_id": event_id,
        "campaign_id": CAMPAIGN,
        "candidate_id": candidate_id,
        "tier": tier,
        "status": status,
        "co_residency": co_residency,
        "integrity_flags": list(integrity_flags),
        "scope_denominator": {
            "machine_subset": machine_subset,
            "numa_nodes": [0, 1] if machine_subset == "partial" else [],
            "devices": ["gfx90a:0"] if machine_subset == "partial" else [],
            "cores": 96,
        },
    }
    if anchor is not None:
        payload["anchor"] = anchor.to_dict()
    return payload


def _event_entry(seq: int = 4, **over) -> J.JournalEntry:
    payload = _event_payload(**over)
    return _entry(
        J.KIND_EVALUATION_EVENT, payload, seq=seq,
        event_id=f"akj-{seq:012d}-event", record_id=payload["event_id"],
    )


def _full_proposal() -> dict:
    """A proposal the SHIPPED validator accepts with zero violations.

    Fields this module never reads are present because a fixture that satisfies only
    the reader proves nothing about the writer. `selection` is the §7.1 planner block
    `selection.screen_proposal` requires and `fingerprint.mechanism_facets` reads.
    """
    return {
        "schema": S.SCHEMA_PROPOSAL,
        "proposal_id": "akp-20260803-0001",
        "campaign_id": CAMPAIGN,
        "parent_candidate_id": None,
        "controller": {
            "provider": "local", "model_id": "architect-a4", "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
            "sampling_params": {"temperature": 0.0, "seed": 42},
            "context_manifest_sha256": _sha("context-manifest"),
        },
        "realized_cost": {
            "controller_tokens": 18_500, "build_seconds": 412.0,
            "evaluator_wall_seconds": 900.0, "gpu_seconds": 240.0,
            "cpu_region_seconds": 0.0, "storage_gb": 1.5,
        },
        "hypothesis": "prose the matcher must never read",
        "narrative": "planner prose that must never be retrieved as fact",
        "narrative_retrievable": False,
        "change_class": "fusion",
        FP.SELECTION_BLOCK_KEY: {
            "mechanism": "elementwise_norm_fusion",
            "hierarchy_layer": "kernel",
            "regime_identity": {"batch": ["b128"]},
        },
        "declared_symbol_deltas": {"added": [], "removed": [], "arity_changed": []},
        "campaign_kind": "fusion",
        "oracle_reference": {"oracle": None, "commit": None, "license_check": None},
        "novelty_basis": {
            "prior_event_ids": [], "source_receipts": [], "do_not_repeat_matches": [],
        },
        "expected_information_gain": 0.4,
        "representation_contract": _representation_contract(),
        "external_numbers": [],
        "target": {"regimes": ["decode"], "ops": ["ggml_cuda_op_rms_norm"],
                   "shapes": [], "models": []},
        "non_target": {"regimes": ["prefill"], "shapes": []},
        "mechanism_prediction": {
            "bottleneck_before": "memory_bandwidth",
            "expected_counter_changes": {"L2CacheHit": "increase"},
            "expected_wall_share_ceiling": 0.35,
            "wall_share_receipt_id": "rcpt-wall-share-0007",
        },
        "change": {
            "predicted_affected_surface": ["rms_norm"],
            "files_and_symbols": ["ggml-cuda/norm.cu:rms_norm_f32"],
            "conceptual_change": "fuse the elementwise/norm cluster into one kernel",
            "parameter_surface": {}, "estimated_diff_size": 40,
        },
        "risks": {"correctness": [], "numerical": [], "state_or_rollback": [],
                  "resource": [], "integrity": []},
        "fallback": {"dispatch_guard": "GGML_AK_FUSE_NORM=0",
                     "kill_switch": "env GGML_AK_FUSE_NORM=0"},
        "evaluation_plan": {
            "required_t0": ["symbol_preservation", "clean_snapshot_build"],
            "required_t1": ["t1a_target_operator_discriminator"],
            "conditional_t2": [], "profiler_questions": [],
        },
        "resource_request": {"lane": "gpu", "expected_minutes": 25,
                             "expected_storage_gb": 2.0},
        "stop_condition": "reject if the discriminator shows no path change",
        "critic_verdict": {"status": "pending", "reasons": []},
    }


def _full_event() -> dict:
    """A v3 evaluation event the SHIPPED validator accepts with zero violations."""
    return {
        "schema": S.SCHEMA_EVALUATION_EVENT_V3,
        "event_id": "ake-20260803-0001",
        "campaign_id": CAMPAIGN,
        "candidate_id": "akc-20260803-0001",
        "tier": "T1",
        "claim_grammar": {
            "category": "CANDIDATE", "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": "decode_tokens_per_s", "metric_direction": "higher_better",
            "reps": 5, "attestation_ref": "rcpt-host-20260804T090000Z",
        },
        "evaluator": {"id": "P-AK-SEARCH-1/v1",
                      "bundle_sha256": _sha("evaluator-bundle")},
        "artifact": {"source_sha256": _sha("snapshot"),
                     "binary_sha256": _sha("candidate-binary"),
                     "linkage_sha256": _sha("candidate-linkage")},
        "anchor": ANCHOR_V8.to_dict(),
        "scope_manifest_sha256": _sha("scope-manifest"),
        "host_receipt": "rcpt-host-20260804T090000Z",
        "resource_claim_receipt": "rcpt-gpu-claim-0042",
        "co_residency": "single",
        "correctness": {"test_backend_ops": "pass"},
        "quality": {}, "stability": {}, "mechanism": {},
        "scope_denominator": {"machine_subset": "full", "numa_nodes": [],
                              "devices": [], "cores": 96},
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "performance": {"raw_samples": [48.1, 48.0, 48.3], "paired_blocks": 3,
                        "estimate": 48.13, "uncertainty": {"e_process_value": 12.4}},
        "integrity_flags": [],
        "status": "fail",
        "supersedes": [],
        "created_at": "2026-08-04T09:45:00+00:00",
    }


def _hypothesis(hid: str = "akh-g15-fusion", *, regime=None,
                statement: str = G15_STATEMENT) -> H.Hypothesis:
    return H.Hypothesis(
        hypothesis_id=hid, statement=statement, falsifier=G15_FALSIFIER,
        origin=H.ORIGIN_OPERATOR, author="operator",
        regime=dict(G15_REGIME if regime is None else regime),
    )


def _ledger_events(
    *,
    hypothesis: H.Hypothesis = None,
    proposal_id: str = "akp-20260803-0001",
    bears: bool = True,
    outcome=None,
    evidence_refs=("ake-20260803-0001",),
) -> tuple:
    """OPENED [+ ATTEMPTED] [+ RESOLVED] — the hypothesis ledger for one question."""
    hypothesis = hypothesis if hypothesis is not None else _hypothesis()
    events = [H.LedgerEvent(
        seq=1, kind=H.EVENT_OPENED, hypothesis_id=hypothesis.hypothesis_id,
        at="2026-08-04T09:00:00.000000Z",
        payload={"hypothesis": hypothesis.to_dict()},
    )]
    if proposal_id is not None:
        events.append(H.LedgerEvent(
            seq=2, kind=H.EVENT_ATTEMPTED, hypothesis_id=hypothesis.hypothesis_id,
            at="2026-08-04T09:10:00.000000Z",
            payload={"attempt": H.Attempt(
                hypothesis_id=hypothesis.hypothesis_id, proposal_id=proposal_id,
                disposition="evaluated", bears_on_falsifier=bears,
                note="the fused kernel was built and measured",
            ).to_dict()},
        ))
    if outcome is not None:
        events.append(H.LedgerEvent(
            seq=3, kind=H.EVENT_RESOLVED, hypothesis_id=hypothesis.hypothesis_id,
            at="2026-08-04T10:00:00.000000Z",
            payload={"resolution": H.ResolutionEvidence(
                outcome=outcome, evidence_grade=H.GRADE_PROTOCOL_BOUND,
                evidence_refs=tuple(evidence_refs),
                falsifier_observed="the cluster measured 31% and fusing it lost 4%",
                bears_on_falsifier=True, resolved_by="controller",
            ).to_dict()},
        ))
    return tuple(events)


def _fold(journal_entries=(), hypothesis_events=(), *, anchor=ANCHOR_V8,
          satisfied=frozenset()) -> D.CompiledLedger:
    return D.fold_journal(
        journal_entries=tuple(journal_entries),
        hypothesis_events=tuple(hypothesis_events),
        current_anchor=anchor,
        satisfied_reopen_predicates=satisfied,
    )


def _refuted_ledger(**event_over) -> D.CompiledLedger:
    """The canonical MATCHED_NEGATIVE corpus: tried, measured, refuted, at v8."""
    return _fold(
        (_campaign_entry(), _proposal_entry(), _candidate_entry(),
         _event_entry(**event_over)),
        _ledger_events(outcome=H.RESOLUTION_REFUTED),
    )


def _classes(ledger: D.CompiledLedger) -> list:
    return sorted(a.entry_class for a in ledger.attempts)


def _check(ledger: D.CompiledLedger, regime=None, statement: str = G15_STATEMENT):
    """The disposition the CONSUMER makes of this ledger's matches."""
    regime = dict(G15_REGIME if regime is None else regime)
    return H.check_do_not_repeat(
        regime=regime, matches=ledger.matches_for(regime, statement)
    )


# =============================================================================
# The six classes, each folded from a realistic journal
# =============================================================================

class TestTheSixClassesFoldFromTheRecord(unittest.TestCase):
    """§19.2's six classes are not an enum this module declares; they are
    dispositions it DERIVES. Each one gets a corpus that produces it."""

    def test_matched_negative(self):
        """Tried, measured on the full machine, refuted, at the CURRENT anchor."""
        ledger = _refuted_ledger()
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        attempt = ledger.attempts[0]
        self.assertEqual(attempt.outcome, H.RESOLUTION_REFUTED)
        self.assertIsNotNone(attempt.receipt)
        self.assertIn("ake-20260803-0001", attempt.receipt)
        self.assertTrue(attempt.rejects())

    def test_hard_constraint(self):
        """A compiled §19.4 constraint: a prohibition, not a measurement."""
        ledger = _fold((_campaign_entry(), _entry(
            "CONSTRAINT_COMPILED",
            {
                "constraint_id": "akx-frozen-tree",
                "mechanism": "elementwise_norm_fusion",
                "regime": {"backend": "llama_gpu", "phase": "decode",
                           "batch_band": "b128"},
                "receipt": "measurement/protocols/kernel-research.md:L212",
                "reopen_when": "the operator authorises a v9 kernel line",
            },
            seq=2, event_id="akj-000000000002-constraint",
        ),))
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_HARD_CONSTRAINT])
        self.assertTrue(ledger.attempts[0].rejects())

    def test_conditional_negative_from_a_partial_machine_subset(self):
        """A full-machine claim from a partial-machine cell is a category error.

        The negative is real, but it can only exclude the cells it measured — so it
        excludes, and does not close.
        """
        ledger = _refuted_ledger(machine_subset="partial")
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_CONDITIONAL_NEGATIVE])
        self.assertTrue(any("PARTIAL" in w for w in ledger.attempts[0].why))

    def test_confounded_result_from_a_voided_window(self):
        """2026-08-04: an A/A destroyed by seven llama-servers coming up mid-flight."""
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(), _event_entry(
                status="invalid",
                integrity_flags=[
                    f"{S.VOID_FLAG_PREFIX}{EV.VOID_CONCURRENT_INFERENCE}:FAIL"
                ],
            )),
            # NOT resolved, and that is the realistic part: a voided window cannot
            # resolve a falsifier, so the question is still open.
            _ledger_events(outcome=None),
        )
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_CONFOUNDED_RESULT])
        self.assertTrue(any(
            EV.VOID_CONCURRENT_INFERENCE in w for w in ledger.attempts[0].why
        ))

    def test_superseded_fact_from_a_moved_anchor(self):
        """Refuted on v7. The anchor is v8 now, so the premise is stale."""
        ledger = _refuted_ledger(anchor=ANCHOR_V7)
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertEqual(ledger.attempts[0].anchor_outcome, S.FAIL)

    def test_low_value_from_a_skipped_proposal(self):
        """Skipped is not a result: nothing was learned, so nothing is closed."""
        ledger = _fold((
            _campaign_entry(), _proposal_entry(),
            _entry(J.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "akp-20260803-0001",
                    "reason": "below the wall-share threshold for this round",
                    "fingerprint": _sha("fp")},
                   seq=4, event_id="akj-000000000004-skip"),
        ))
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_LOW_VALUE])

    def test_every_declared_class_is_reachable(self):
        """The control on the five above: no class is decoration.

        A class this module can name but never produce is a row in a table nobody
        enforces — the exact shape `hypotheses.audit_no_origin_grade_promotion` was
        rewritten to stop trusting.
        """
        produced = set()
        for ledger in (
            _refuted_ledger(),
            _refuted_ledger(machine_subset="partial"),
            _refuted_ledger(anchor=ANCHOR_V7),
        ):
            produced.update(a.entry_class for a in ledger.attempts)
        produced.add(H.MATCH_CLASS_HARD_CONSTRAINT)
        produced.add(H.MATCH_CLASS_CONFOUNDED_RESULT)
        produced.add(H.MATCH_CLASS_LOW_VALUE)
        self.assertEqual(produced, set(H.MATCH_CLASSES))
        self.assertEqual(set(D.CLASS_PRECEDENCE), set(H.MATCH_CLASSES))


# =============================================================================
# What rejects, and what only advises
# =============================================================================

class TestRejectingVersusAdvisory(unittest.TestCase):
    """Disposition is asserted through `hypotheses.check_do_not_repeat()` itself.

    Asserting it against a local copy of the rule would let this module and its
    consumer drift into two opinions about what closes a question — which is the
    disagreement `fingerprint.py` exists because of.
    """

    def test_the_two_rejecting_classes_reject(self):
        for label, ledger in (
            ("MATCHED_NEGATIVE", _refuted_ledger()),
            ("HARD_CONSTRAINT", _fold((_campaign_entry(), _entry(
                "CONSTRAINT_COMPILED",
                {"mechanism": "elementwise_norm_fusion",
                 "regime": {"backend": "llama_gpu", "phase": "decode",
                            "batch_band": "b128"},
                 "receipt": "measurement/protocols/kernel-research.md:L212"},
                seq=2, event_id="akj-000000000002-constraint"),))),
        ):
            with self.subTest(entry_class=label):
                verdict = _check(ledger)
                self.assertEqual(verdict.outcome, S.FAIL, verdict.reasons)

    def test_the_four_advisory_classes_leave_the_question_open(self):
        skipped = _fold((
            _campaign_entry(), _proposal_entry(),
            _entry(J.KIND_PROPOSAL_SKIPPED,
                   {"proposal_ref": "akp-20260803-0001", "reason": "low yield"},
                   seq=4, event_id="akj-000000000004-skip"),
        ))
        confounded = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(), _event_entry(
                status="invalid",
                integrity_flags=[
                    f"{S.VOID_FLAG_PREFIX}{EV.VOID_CONCURRENT_INFERENCE}:FAIL"],
            )),
            _ledger_events(outcome=None),
        )
        for label, ledger in (
            ("CONDITIONAL_NEGATIVE", _refuted_ledger(machine_subset="partial")),
            ("SUPERSEDED_FACT", _refuted_ledger(anchor=ANCHOR_V7)),
            ("CONFOUNDED_RESULT", confounded),
            ("LOW_VALUE", skipped),
        ):
            with self.subTest(entry_class=label):
                verdict = _check(ledger)
                # The control that stops this passing vacuously: the entry MATCHED.
                self.assertEqual(
                    [m.entry_class for m in ledger.matches_for(G15_REGIME, "x")],
                    [label], f"{label} did not match its own regime",
                )
                self.assertEqual(verdict.outcome, S.PASS, verdict.reasons)

    def test_an_empty_ledger_rejects_nothing(self):
        empty = _fold()
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.matches_for(G15_REGIME, G15_STATEMENT), ())
        self.assertEqual(_check(empty).outcome, S.PASS)

    def test_a_receipt_is_never_synthesised_to_make_a_match_bite(self):
        """§19.3's price of closing a family, at the unit that computes it.

        `None` here is what turns a rejecting class into COULD_NOT_CHECK downstream. A
        placeholder string would read as a resolvable receipt to every later reader —
        the same defect `schemas.is_placeholder_digest` refuses in an anchor block.
        """
        self.assertIsNone(D._receipt((), (), None))
        self.assertIsNone(D._receipt((), (), ANCHOR_V8))
        self.assertIsNone(D._receipt(("", "   "), (), ANCHOR_V8))
        receipted = D._receipt(("ake-1",), ("akj-2",), ANCHOR_V8)
        self.assertIn("ake-1", receipted)
        self.assertIn("akj-2", receipted)
        self.assertIn(ANCHOR_V8.short(), receipted)

    def test_an_unreceipted_negative_is_could_not_check_never_a_rejection(self):
        """§8.4 rejects only a negative *carrying a receipt*; §19.3 says why.

        A receipt is never synthesised to make a match bite, so a negative with no
        resolvable id leaves the caller at COULD_NOT_CHECK — which is a third outcome,
        not a soft pass and not a rejection.
        """
        ledger = _fold((_campaign_entry(), _entry(
            "CONSTRAINT_COMPILED",
            {"mechanism": "elementwise_norm_fusion",
             "regime": {"backend": "llama_gpu", "phase": "decode",
                        "batch_band": "b128"}},
            seq=2, event_id="akj-000000000002-constraint",
        ),))
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_HARD_CONSTRAINT])
        self.assertIsNone(ledger.attempts[0].receipt)
        self.assertEqual(_check(ledger).outcome, S.COULD_NOT_CHECK)

    def test_a_satisfied_reopen_predicate_admits_a_matched_negative(self):
        """§19.2's reopen predicate — and it excuses ONLY `MATCHED_NEGATIVE`.

        The consumer has this branch; before the constraint ledger could compile a
        class other than HARD_CONSTRAINT it was unreachable from its only producer,
        which is a guard wired to nothing one layer further in.
        """
        predicate = "the operator authorises a v9 kernel line"

        def _compiled(entry_class, **over):
            payload = {
                "entry_class": entry_class,
                "mechanism": "elementwise_norm_fusion",
                "regime": {"backend": "llama_gpu", "phase": "decode",
                           "batch_band": "b128"},
                "receipt": "protocols/kernel-research.md:L212",
                "anchor_commit": V8_COMMIT,
                "reopen_when": predicate,
            }
            payload.update(over)
            return _entry("CONSTRAINT_COMPILED", payload, seq=9,
                          event_id="akj-000000000009-constraint")

        negative = (_campaign_entry(),
                    _compiled(H.MATCH_CLASS_MATCHED_NEGATIVE))
        blocked = _fold(negative)
        self.assertEqual(_check(blocked).outcome, S.FAIL)
        self.assertTrue(blocked.lookup(G15_REGIME).rejecting)

        reopened = _fold(negative, satisfied=frozenset({predicate}))
        self.assertEqual(_check(reopened).outcome, S.PASS)
        # …and the EXPLANATION says so too. Asserting only the consumer's verdict
        # would leave `PriorAttempt.rejects` free to disagree with it, and the
        # explanation is the surface an agent inspects when deciding whether to argue.
        lookup = reopened.lookup(G15_REGIME)
        self.assertEqual(len(lookup.matches), 1)
        self.assertEqual(lookup.rejecting, ())
        self.assertTrue(lookup.explanations[0].reopen_predicate_satisfied)

        # A hard constraint is a PROHIBITION, not a negative: a fact about the world
        # does not reopen it, and it still rejects.
        hard = _fold((_campaign_entry(), _compiled(H.MATCH_CLASS_HARD_CONSTRAINT)),
                     satisfied=frozenset({predicate}))
        self.assertEqual(_check(hard).outcome, S.FAIL)
        self.assertTrue(hard.lookup(G15_REGIME).rejecting)

    def test_a_compiled_negative_bound_to_a_stale_anchor_supersedes(self):
        """§19.3, verbatim: *a suppression whose receipt no longer resolves reverts.*"""
        def _compiled(anchor_commit):
            return _entry("CONSTRAINT_COMPILED", {
                "entry_class": H.MATCH_CLASS_MATCHED_NEGATIVE,
                "mechanism": "elementwise_norm_fusion",
                "regime": {"backend": "llama_gpu", "phase": "decode",
                           "batch_band": "b128"},
                "receipt": "protocols/kernel-research.md:L212",
                "anchor_commit": anchor_commit,
            }, seq=9, event_id="akj-000000000009-constraint")

        stale = _fold((_campaign_entry(), _compiled(V7_COMMIT)))
        self.assertEqual(_classes(stale), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertEqual(_check(stale).outcome, S.PASS)
        current = _fold((_campaign_entry(), _compiled(V8_COMMIT)))
        self.assertEqual(_classes(current), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        self.assertEqual(_check(current).outcome, S.FAIL)

    def test_a_conflicted_compiled_entry_never_rejects(self):
        ledger = _fold((_campaign_entry(), _entry("CONSTRAINT_COMPILED", {
            "entry_class": H.MATCH_CLASS_HARD_CONSTRAINT,
            "mechanism": "elementwise_norm_fusion",
            "regime": {"backend": "llama_gpu", "phase": "decode",
                       "batch_band": "b128"},
            "receipt": "protocols/kernel-research.md:L212",
            "conflicted": True,
        }, seq=9, event_id="akj-000000000009-constraint")))
        self.assertTrue(ledger.attempts[0].conflicted)
        self.assertEqual(len(ledger.matches_for(G15_REGIME, "x")), 1)
        self.assertEqual(_check(ledger).outcome, S.PASS)

    def test_a_compiled_entry_outside_the_closed_vocabulary_is_refused(self):
        ledger = _fold((_campaign_entry(), _entry("CONSTRAINT_COMPILED", {
            "entry_class": "PROBABLY_BAD_IDEA",
            "mechanism": "elementwise_norm_fusion",
            "receipt": "protocols:L1",
        }, seq=9, event_id="akj-000000000009-constraint")))
        self.assertEqual(ledger.attempts, ())
        self.assertEqual(len(ledger.unusable), 1)
        self.assertIn("PROBABLY_BAD_IDEA", ledger.unusable[0]["reason"])


# =============================================================================
# Matching — the whole value of the module
# =============================================================================

class TestMatchingIsStructuralNotProse(unittest.TestCase):

    def test_a_reworded_restatement_of_a_tried_idea_still_matches(self):
        """The failure §8.4 names by name: attempt 119 looking novel.

        Same mechanism, same regime, completely different words. It matches, and it
        matches with the SAME disposition — otherwise an agent walks around the
        ledger with a thesaurus.
        """
        ledger = _refuted_ledger()
        reworded = dict(G15_REGIME)
        original = "fusing G15's elementwise/norm cluster lands >= 15% on decode"
        rewording = (
            "collapsing the normalisation and pointwise stages into a single launch "
            "should buy at least fifteen percent"
        )
        self.assertEqual(
            [m.to_dict() for m in ledger.matches_for(reworded, original)],
            [m.to_dict() for m in ledger.matches_for(reworded, rewording)],
        )
        self.assertEqual(_check(ledger, reworded, rewording).outcome, S.FAIL)

    def test_a_differently_spelled_mechanism_still_matches(self):
        """`Elementwise/Norm Fusion` and `elementwise_norm_fusion` are one identifier.

        Case-folding and separator-collapsing, and NOTHING else — no stemming, no edit
        distance, no substring test. The next test is the control that this is not
        fuzzy matching in disguise.
        """
        ledger = _refuted_ledger()
        regime = dict(G15_REGIME, mechanism="Elementwise/Norm Fusion")
        self.assertEqual(len(ledger.matches_for(regime, "x")), 1)

    def test_two_different_ideas_about_one_function_do_not_match(self):
        """The SILENT failure this module is most dangerous for.

        Same op, same regime, same everything except what is actually being done —
        and the second idea is genuinely new. If this ever matches, the loop goes
        sterile while looking productive and nothing tests the family again.
        """
        ledger = _refuted_ledger()
        other_idea = dict(G15_REGIME, mechanism="vectorize_norm_reduction_loop")
        self.assertEqual(ledger.matches_for(other_idea, G15_STATEMENT), ())
        self.assertEqual(_check(ledger, other_idea).outcome, S.PASS)
        # …and it says WHY it did not match, rather than being silently absent.
        near = ledger.lookup(other_idea).near_misses
        self.assertEqual(len(near), 1)
        self.assertTrue(any("different mechanism" in r for r in near[0].reasons))

    def test_the_same_idea_in_a_different_regime_does_not_match(self):
        """§19.2: this project repeatedly observes SIGN CHANGES across regimes.

        A GPU-decode negative says nothing about CPU prefill, and a ledger that
        suppressed it would close a family it never measured.
        """
        ledger = _refuted_ledger()
        for dimension, value in (("backend", "llama_cpu"), ("phase", "prefill"),
                                 ("batch_band", "b1")):
            with self.subTest(dimension=dimension):
                elsewhere = dict(G15_REGIME, **{dimension: value})
                self.assertEqual(ledger.matches_for(elsewhere, G15_STATEMENT), ())
                self.assertEqual(_check(ledger, elsewhere).outcome, S.PASS)

    def test_a_question_that_declares_less_than_the_entry_does_not_match(self):
        """`selection.match_ledger`'s rule: an unobserved dimension is not agreement.

        The bias is deliberate and is stated in the module docstring — failing to
        reject costs a claim and shows up in the journal; rejecting wrongly costs a
        research family and shows up nowhere.
        """
        ledger = _refuted_ledger()
        thin = {"mechanism": "elementwise_norm_fusion", "backend": "llama_gpu"}
        self.assertEqual(ledger.matches_for(thin, G15_STATEMENT), ())
        self.assertTrue(any(
            "does not declare" in r for r in ledger.lookup(thin).near_misses[0].reasons
        ))

    def test_under_specified_is_reported_and_is_not_genuinely_new(self):
        """"Nobody has tried this" and "you did not say enough" must not be one answer.

        The proposal was measured at a named quant; the question does not say which
        quant it is about. That is not a new idea, it is an unanswerable comparison,
        and `selection` rejects a PROPOSAL outright for the same gap
        (`REJECT_REGIME_IDENTITY_INCOMPLETE`).
        """
        ledger = _fold(
            (_campaign_entry(),
             _proposal_entry(regime_identity={"batch": ["b128"], "quant": ["Q4_K_M"]}),
             _candidate_entry(), _event_entry()),
            _ledger_events(outcome=H.RESOLUTION_REFUTED),
        )
        lookup = ledger.lookup(G15_REGIME)
        self.assertEqual(lookup.matches, ())
        self.assertEqual(lookup.undeclared_dimensions, ("quant",))
        self.assertEqual(len(lookup.incomplete_comparisons), 1)

        # …and declaring it resolves the comparison in BOTH directions: the same quant
        # matches and rejects, a different quant is genuinely new.
        same = ledger.lookup(dict(G15_REGIME, quant="Q4_K_M"))
        self.assertEqual(len(same.matches), 1)
        self.assertEqual(same.undeclared_dimensions, ())
        other = ledger.lookup(dict(G15_REGIME, quant="UD-IQ2_M"))
        self.assertEqual(other.matches, ())
        self.assertEqual(other.undeclared_dimensions, ())

    def test_a_negative_with_no_regime_identity_cannot_close_a_question(self):
        """§19.2: *"'do not repeat' without regime identity is dangerous, because this
        project repeatedly observes sign changes across architecture, substrate, batch,
        context and quant."*

        An entry with NO regime dimensions matches every question about its mechanism —
        which is precisely why it must not be allowed to be the class that rejects. It
        excludes the cells it names and nothing more.
        """
        ledger = _fold(
            (_campaign_entry(), _event_entry()),
            _ledger_events(
                hypothesis=_hypothesis(regime={"mechanism": "elementwise_norm_fusion"}),
                proposal_id=None, outcome=H.RESOLUTION_REFUTED),
        )
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_CONDITIONAL_NEGATIVE])
        # It matches everywhere — the reason it must never reject.
        for regime in (G15_REGIME,
                       dict(G15_REGIME, backend="llama_cpu", phase="prefill")):
            with self.subTest(regime=regime.get("backend")):
                self.assertEqual(len(ledger.matches_for(regime, "x")), 1)
                self.assertEqual(_check(ledger, regime).outcome, S.PASS)
        self.assertTrue(any("no regime identity" in w
                            for w in ledger.attempts[0].why))

    def test_hierarchy_layer_is_a_target_key_not_a_regime_barrier(self):
        """A proposal declares its layer; a one-line operator hypothesis does not.

        Filed as a regime dimension it would block every match on a question that
        never mentioned it — a barrier on the wrong axis, and a ledger that suppresses
        nothing looks exactly like a ledger with nothing in it.
        """
        self.assertIn("hierarchy_layer", D.TARGET_SET_DIMENSIONS)
        self.assertNotIn("hierarchy_layer", D.REGIME_DIMENSIONS)
        ledger = _refuted_ledger()
        self.assertIn("hierarchy_layer", ledger.attempts[0].target.sets)
        self.assertEqual(len(ledger.matches_for(G15_REGIME, G15_STATEMENT)), 1)

    def test_a_question_with_no_mechanism_matches_nothing_and_SAYS_SO(self):
        """Regime-only matching would suppress every idea in a regime.

        This test used to assert `matches_for(...) == ()` and stop there, which encoded
        the fail-open: an empty match set is, in the CONSUMER's vocabulary, "consulted
        and matched nothing" — PASS. `matches_for` now refuses rather than making that
        statement, and `disposition()` is COULD_NOT_CHECK. See
        `TestAnUnanswerableQuestionIsNotAClearOne`.
        """
        ledger = _refuted_ledger()
        no_mechanism = {k: v for k, v in G15_REGIME.items() if k != "mechanism"}
        with self.assertRaises(D.DoNotRepeatError):
            ledger.matches_for(no_mechanism, G15_STATEMENT)
        self.assertEqual(ledger.lookup(no_mechanism).matches, ())
        self.assertTrue(any(
            "names no mechanism" in r
            for r in ledger.lookup(no_mechanism).near_misses[0].reasons
        ))

    def test_a_different_op_under_one_mechanism_does_not_match(self):
        """The structural sets constrain when BOTH sides declare them."""
        ledger = _refuted_ledger()
        other_op = dict(G15_REGIME, ops=["ggml_cuda_op_softmax"])
        self.assertEqual(ledger.matches_for(other_op, G15_STATEMENT), ())

    def test_the_statement_cannot_change_a_match(self):
        """Proved from the objects and behaviourally, with the vacuity control."""
        self.assertEqual(
            [f.name for f in dataclasses.fields(D.MatchQuery)
             if "statement" in f.name or "prose" in f.name], [])
        ledger = _refuted_ledger()
        baseline = ledger.matches_for(G15_REGIME, G15_STATEMENT)
        self.assertTrue(baseline, "control: the probe must actually match")
        for statement in ("", "a" * 400, "wholly unrelated words about nothing"):
            with self.subTest(statement=statement[:20]):
                self.assertEqual(
                    [m.to_dict() for m in ledger.matches_for(G15_REGIME, statement)],
                    [m.to_dict() for m in baseline],
                )

    def test_the_prose_audit_passes_and_is_not_vacuous(self):
        audit = D.audit_matching_ignores_prose()
        self.assertEqual(audit.outcome, S.PASS, audit.reasons)
        probe = D._audit_probe_ledger()
        self.assertTrue(probe.matches_for(D._AUDIT_REGIME, D._AUDIT_STATEMENTS[0]))

    def test_the_prose_audit_fails_when_the_matcher_reads_prose(self):
        """BITE-VERIFICATION: break the property, watch the audit catch it.

        An audit that has never been observed to fail is an audit nobody has checked.
        """
        original = D.CompiledLedger.matches_for

        def prose_sensitive(self, regime, statement):
            return () if "unrelated" in statement or "differently" in statement \
                else original(self, regime, statement)

        D.CompiledLedger.matches_for = prose_sensitive
        try:
            self.assertEqual(D.audit_matching_ignores_prose().outcome, S.FAIL)
        finally:
            D.CompiledLedger.matches_for = original
        self.assertEqual(D.audit_matching_ignores_prose().outcome, S.PASS)

    def test_the_prose_audit_is_could_not_check_when_the_probe_matches_nothing(self):
        """The anti-vacuous control, verified by biting it.

        A matcher that returns nothing for everything satisfies "the same answer for
        two statements" perfectly, so that alone must not read as PASS.
        """
        original = D.CompiledLedger.matches_for
        D.CompiledLedger.matches_for = lambda self, regime, statement: ()
        try:
            self.assertEqual(
                D.audit_matching_ignores_prose().outcome, S.COULD_NOT_CHECK)
        finally:
            D.CompiledLedger.matches_for = original


# =============================================================================
# Anchor sensitivity
# =============================================================================

class TestAnchorSensitivity(unittest.TestCase):
    """*"A kernel idea that failed on v7 may well win on v8."*"""

    def test_a_negative_against_a_moved_anchor_is_superseded_not_matched(self):
        stale = _refuted_ledger(anchor=ANCHOR_V7)
        self.assertEqual(_classes(stale), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertNotIn(H.MATCH_CLASS_MATCHED_NEGATIVE, _classes(stale))
        self.assertEqual(_check(stale).outcome, S.PASS)
        # CONTROL: the identical corpus at the CURRENT anchor still rejects, so the
        # PASS above is the anchor move and not a matcher that stopped matching.
        current = _refuted_ledger()
        self.assertEqual(_classes(current), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        self.assertEqual(_check(current).outcome, S.FAIL)
        self.assertEqual(len(stale.matches_for(G15_REGIME, "x")), 1)

    def test_an_unobservable_anchor_comparison_is_never_a_pass(self):
        """COULD_NOT_CHECK supersedes too: a tool named on one side only.

        `AnchorIdentity.identity_matches` returns COULD_NOT_CHECK when one side names
        its tool and the other does not — *"not naming a tool is not evidence that it
        is the same tool"* — and an unobserved component must never be the thing that
        closes a question.
        """
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(), _event_entry()),
            _ledger_events(outcome=H.RESOLUTION_REFUTED),
            anchor=ANCHOR_V8.for_tool("llama-bench"),
        )
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertEqual(ledger.attempts[0].anchor_outcome, S.COULD_NOT_CHECK)

    def test_one_stale_measurement_supersedes_a_multi_measurement_negative(self):
        """EVERY measurement behind a negative must still bind, not just one of them.

        Taking "the anchor" off whichever event sorted first would let a v8 run vouch
        for the v7 run sitting beside it, and the negative would keep rejecting on
        evidence half of which is stale.
        """
        entries = (
            _campaign_entry(), _proposal_entry(), _candidate_entry(),
            _event_entry(seq=4, event_id="ake-20260803-0001", anchor=ANCHOR_V8),
            _event_entry(seq=5, event_id="ake-20260803-0002", anchor=ANCHOR_V7),
        )
        events = _ledger_events(
            outcome=H.RESOLUTION_REFUTED,
            evidence_refs=("ake-20260803-0001", "ake-20260803-0002"),
        )
        self.assertEqual(_classes(_fold(entries, events)),
                         [H.MATCH_CLASS_SUPERSEDED_FACT])
        # CONTROL: both at the current anchor and it is a negative again.
        current = (entries[:4] + (_event_entry(seq=5, event_id="ake-20260803-0002",
                                               anchor=ANCHOR_V8),))
        self.assertEqual(_classes(_fold(current, events)),
                         [H.MATCH_CLASS_MATCHED_NEGATIVE])

    def test_a_measurement_that_names_no_anchor_blocks_the_comparison(self):
        """An unobserved component is never a PASS, even beside an observed one."""
        entries = (
            _campaign_entry(), _proposal_entry(), _candidate_entry(),
            _event_entry(seq=4, event_id="ake-20260803-0001", anchor=ANCHOR_V8),
            _event_entry(seq=5, event_id="ake-20260803-0002", anchor=None,
                         status="fail"),
        )
        ledger = _fold(entries, _ledger_events(
            outcome=H.RESOLUTION_REFUTED,
            evidence_refs=("ake-20260803-0001", "ake-20260803-0002"),
        ))
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertEqual(ledger.attempts[0].anchor_outcome, S.COULD_NOT_CHECK)

    def test_a_negative_with_no_anchor_at_all_is_superseded(self):
        ledger = _refuted_ledger(anchor=None)
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertIsNone(ledger.attempts[0].anchor)

    def test_a_ledger_compiled_without_a_current_anchor_rejects_nothing_measured(self):
        """`current_anchor=None` is a position, not a default to paper over."""
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(), _event_entry()),
            _ledger_events(outcome=H.RESOLUTION_REFUTED),
            anchor=None,
        )
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertEqual(_check(ledger).outcome, S.PASS)

    def test_a_commit_string_is_refused_as_a_current_anchor(self):
        """A bare string compares unequal to every anchor and would silently
        supersede the whole ledger — a fail-open that looks like a fail-closed."""
        with self.assertRaises(D.LedgerFoldError):
            D.fold_journal(current_anchor=V8_COMMIT)


# =============================================================================
# Contention
# =============================================================================

class TestContentionIsNotAResult(unittest.TestCase):

    def test_a_contended_measurement_is_confounded_and_leaves_the_question_open(self):
        confounded = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(), _event_entry(
                status="invalid",
                integrity_flags=[
                    f"{S.VOID_FLAG_PREFIX}{EV.VOID_CONCURRENT_INFERENCE}:FAIL"],
            )),
            _ledger_events(outcome=None),
        )
        self.assertEqual(_classes(confounded), [H.MATCH_CLASS_CONFOUNDED_RESULT])
        matches = confounded.matches_for(G15_REGIME, G15_STATEMENT)
        self.assertEqual(len(matches), 1, "the attempt must still be VISIBLE")
        self.assertEqual(_check(confounded).outcome, S.PASS)

    def test_a_contended_refutation_does_not_become_a_negative(self):
        """The dangerous case: the window was voided AND somebody resolved anyway.

        Precedence, not narrative: `CONFOUNDED_RESULT` outranks `MATCHED_NEGATIVE`,
        so a result taken under a voided window cannot close the question no matter
        what the resolution says.
        """
        ledger = _refuted_ledger(
            status="invalid",
            integrity_flags=[
                f"{S.VOID_FLAG_PREFIX}{EV.VOID_CONCURRENT_INFERENCE}:FAIL"],
        )
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_CONFOUNDED_RESULT])
        self.assertEqual(_check(ledger).outcome, S.PASS)
        self.assertLess(
            D.CLASS_PRECEDENCE.index(H.MATCH_CLASS_CONFOUNDED_RESULT),
            D.CLASS_PRECEDENCE.index(H.MATCH_CLASS_MATCHED_NEGATIVE),
        )

    def test_co_residency_alone_is_not_a_confounder(self):
        """Co-residency is scheduling DATA, not a trust gate (2026-07-27).

        Some lineups are concurrent BY DESIGN. The trust signal is the void finding
        the evaluator wrote, and treating co-residency as one would void every
        co-resident cell the project deliberately runs.
        """
        ledger = _refuted_ledger(co_residency="co_resident:big-plus-quarters")
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        self.assertEqual(_check(ledger).outcome, S.FAIL)

    def test_every_void_flag_confounds_not_only_the_contention_one(self):
        for reason in EV.VOID_REASONS:
            with self.subTest(void_reason=reason):
                ledger = _refuted_ledger(
                    integrity_flags=[f"{S.VOID_FLAG_PREFIX}{reason}:FAIL"])
                self.assertEqual(_classes(ledger), [H.MATCH_CLASS_CONFOUNDED_RESULT])

    def test_a_clean_status_with_no_flags_is_not_confounded(self):
        """The compliant-path control for the guard above."""
        self.assertEqual(
            _classes(_refuted_ledger(integrity_flags=[])),
            [H.MATCH_CLASS_MATCHED_NEGATIVE],
        )


# =============================================================================
# What the fold refuses to turn into a negative
# =============================================================================

class TestWhatIsDeliberatelyNotANegative(unittest.TestCase):

    def test_an_attempt_that_did_not_bear_on_the_falsifier_produces_no_entry(self):
        """§8.4.0's founding defect, one layer up.

        A hypothesis used to evaporate when its proposal was dispositioned *including
        when that proposal failed for an unrelated reason*. Folding a build break into
        a do-not-repeat entry would rebuild exactly that.
        """
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(),
             _event_entry(status="fail")),
            _ledger_events(bears=False, outcome=None),
        )
        self.assertEqual(ledger.attempts, ())
        self.assertEqual(_check(ledger).outcome, S.PASS)

    def test_a_non_bearing_attempt_does_not_contaminate_a_real_negative(self):
        """The sharper half of the same rule: WHOSE evidence the negative is built on.

        The question was refuted by a clean full-machine run (E1, cited by the
        resolution). Separately, a proposal that did NOT bear on the falsifier ran and
        its window was voided (E2). Joining E2 in would drag its confounder onto a
        negative it had nothing to do with — the failure §8.4.0 names, arriving through
        the evidence join instead of through the status field.
        """
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(),
             _event_entry(seq=4, event_id="ake-20260803-0001", status="fail",
                          candidate_id="akc-unrelated"),
             _event_entry(seq=5, event_id="ake-20260803-0002", status="invalid",
                          candidate_id="akc-20260803-0001",
                          integrity_flags=[
                              f"{S.VOID_FLAG_PREFIX}{EV.VOID_CONCURRENT_INFERENCE}:FAIL"
                          ])),
            _ledger_events(bears=False, outcome=H.RESOLUTION_REFUTED,
                           evidence_refs=("ake-20260803-0001",)),
        )
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        self.assertEqual(ledger.attempts[0].event_ids, ("ake-20260803-0001",))
        self.assertEqual(ledger.attempts[0].proposal_ids, ())

    def test_an_inconclusive_resolution_produces_no_entry(self):
        """*"The experiment ran and did not resolve"* is not a negative."""
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(),
             _event_entry(status="inconclusive")),
            _ledger_events(outcome=H.RESOLUTION_INCONCLUSIVE),
        )
        self.assertEqual(ledger.attempts, ())

    def test_an_open_question_with_a_clean_attempt_produces_no_entry(self):
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(),
             _event_entry(status="pass")),
            _ledger_events(outcome=None),
        )
        self.assertEqual(ledger.attempts, ())

    def test_a_question_with_no_mechanism_is_reported_not_dropped(self):
        """It cannot be matched on, so it is not compiled — and it is REPORTED.

        Silently dropping it is how a ledger looks populated while suppressing
        nothing.
        """
        regime = {k: v for k, v in G15_REGIME.items() if k != "mechanism"}
        ledger = _fold(
            (_campaign_entry(), _candidate_entry(), _event_entry()),
            _ledger_events(hypothesis=_hypothesis(regime=regime), proposal_id=None,
                           outcome=H.RESOLUTION_REFUTED),
        )
        self.assertEqual(ledger.attempts, ())
        self.assertEqual(len(ledger.unusable), 1)
        self.assertIn("mechanism", ledger.unusable[0]["reason"])

    def test_a_constraint_with_no_mechanism_is_reported_not_dropped(self):
        ledger = _fold((_campaign_entry(), _entry(
            "CONSTRAINT_COMPILED",
            {"regime": {"backend": "llama_gpu"}, "receipt": "protocols:L1"},
            seq=2, event_id="akj-000000000002-constraint",
        ),))
        self.assertEqual(ledger.attempts, ())
        self.assertEqual(len(ledger.unusable), 1)

    def test_contradictory_outcomes_conflict_and_stop_rejecting(self):
        """§19.3: a suppression whose evidence disagrees with itself does not block."""
        entries = (
            _campaign_entry(),
            _proposal_entry(seq=2),
            _proposal_entry(seq=5, proposal_id="akp-20260803-0002"),
            _candidate_entry(seq=3),
            _candidate_entry(seq=6, candidate_id="akc-20260803-0002",
                             proposal_id="akp-20260803-0002"),
            _event_entry(seq=4, status="fail"),
            _event_entry(seq=7, event_id="ake-20260803-0002", status="pass",
                         candidate_id="akc-20260803-0002"),
        )
        ledger = _fold(entries)
        self.assertEqual(len(ledger), 2)
        self.assertTrue(all(a.conflicted for a in ledger.attempts))
        self.assertFalse(any(a.rejects() for a in ledger.attempts))
        verdict = _check(ledger)
        self.assertNotEqual(verdict.outcome, S.FAIL)
        self.assertTrue(any("CONFLICTED" in w
                            for a in ledger.attempts for w in a.why))


# =============================================================================
# Conformance to the consumer, and to the rest of the plane
# =============================================================================

class TestConformsToItsConsumer(unittest.TestCase):

    def test_the_compiled_ledger_satisfies_the_declared_protocol(self):
        ledger = _refuted_ledger()
        self.assertTrue(hasattr(H.DoNotRepeatLedger, "matches_for"))
        matches = ledger.matches_for(G15_REGIME, G15_STATEMENT)
        for match in matches:
            self.assertIsInstance(match, H.LedgerMatch)
            self.assertIn(match.entry_class, H.MATCH_CLASSES)

    def test_it_plugs_into_the_planner_round_block_unchanged(self):
        """End to end through the surface the planner actually reads.

        `planner_round_block` is where a match reaches a planning round, and it calls
        `check_do_not_repeat` per hypothesis. If the shapes did not line up this is
        where it would surface, so it is asserted rather than assumed.
        """
        import tempfile

        ledger = _refuted_ledger()
        with tempfile.TemporaryDirectory() as tmp:
            jr = J.Journal(root=tmp + "/journal")
            jr.initialize()
            tracker = H.HypothesisTracker(journal_=jr, root=tmp + "/hyp",
                                          campaign_id=CAMPAIGN)
            hypothesis = _hypothesis()
            tracker.open_hypothesis(hypothesis)
            block = tracker.planner_round_block(
                round_id="akr-0001",
                matches_by_hypothesis={
                    hypothesis.hypothesis_id: ledger.matches_for(
                        hypothesis.regime, hypothesis.statement)
                },
            )
        entry = block["still_open"][0]
        self.assertEqual(entry["do_not_repeat"]["outcome"], S.FAIL)
        self.assertTrue(any("MATCHED_NEGATIVE" in r
                            for r in entry["do_not_repeat"]["reasons"]))

    def test_the_regime_vocabulary_agrees_with_selection(self):
        """One §19.2 dimension vocabulary, split across two axes — not two vocabularies.

        Every dimension `selection.LedgerEntry` admits is classified here as either a
        regime dimension or a structural-target key (`regimes` being `phase` under
        §7.2's spelling). A dimension in one module and not the other is a suppression
        one plane enforces and the other cannot see.
        """
        self.assertTrue(D.REGIME_DIMENSIONS <= SEL.LEDGER_DIMENSIONS)
        classified = (
            set(D.REGIME_DIMENSIONS) | set(D.TARGET_SET_DIMENSIONS)
            | set(D.DIMENSION_ALIASES)
        )
        self.assertEqual(SEL.LEDGER_DIMENSIONS - classified, set())
        self.assertEqual(
            sorted(set(D.TARGET_SET_DIMENSIONS) - SEL.LEDGER_DIMENSIONS),
            ["files", "symbols"],
        )

    def test_matched_negative_is_last_in_precedence_and_hard_constraint_first(self):
        """The two properties that carry the module's safety."""
        self.assertEqual(D.CLASS_PRECEDENCE[-1], H.MATCH_CLASS_MATCHED_NEGATIVE)
        self.assertEqual(D.CLASS_PRECEDENCE[0], H.MATCH_CLASS_HARD_CONSTRAINT)
        for demotion in (H.MATCH_CLASS_CONFOUNDED_RESULT,
                         H.MATCH_CLASS_SUPERSEDED_FACT,
                         H.MATCH_CLASS_CONDITIONAL_NEGATIVE):
            self.assertLess(D.CLASS_PRECEDENCE.index(demotion),
                            D.CLASS_PRECEDENCE.index(H.MATCH_CLASS_MATCHED_NEGATIVE))

    def test_the_fold_reads_the_SHIPPED_record_shapes(self):
        """ANTI-DRIFT: the keys this module reads are keys the records actually have.

        A fold that reads `payload["mechanism"]` when the record writes
        `payload["selection"]["mechanism"]` produces an EMPTY ledger from a FULL
        journal — and every other test in this file still passes, because they all run
        on the same wrong fixture. So the identity is extracted here out of records
        that the shipped validators accept with ZERO violations, and out of
        `fingerprint.mechanism_facets`, which is the package's existing answer to
        "where does a proposal's structural identity live".
        """
        proposal = _full_proposal()
        self.assertEqual(S.validate_proposal(proposal), [])
        self.assertEqual(FP.mechanism_facets(proposal)["mechanism"],
                         "elementwise_norm_fusion")
        event = _full_event()
        self.assertEqual(S.validate_evaluation_event_v3(event), [])

        shipped = _fold(
            (_campaign_entry(),
             _entry(J.KIND_PROPOSAL_RECORDED, proposal, seq=2,
                    event_id="akj-000000000002-proposal",
                    record_id=proposal["proposal_id"]),
             _candidate_entry(seq=3, candidate_id=event["candidate_id"],
                              proposal_id=proposal["proposal_id"]),
             _entry(J.KIND_EVALUATION_EVENT, event, seq=4,
                    event_id="akj-000000000004-event",
                    record_id=event["event_id"])),
        )
        self.assertEqual(_classes(shipped), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        from_shipped = shipped.attempts[0]
        self.assertEqual(from_shipped.target.mechanism_raw, "elementwise_norm_fusion")
        self.assertEqual(from_shipped.anchor.source_commit, V8_COMMIT)
        self.assertIn("backend", from_shipped.regime)
        self.assertIn("phase", from_shipped.regime)
        self.assertEqual(len(shipped.matches_for(G15_REGIME, G15_STATEMENT)), 1)

        ledger = _refuted_ledger()
        attempt = ledger.attempts[0]
        self.assertEqual(attempt.target.mechanism_raw, "elementwise_norm_fusion")
        self.assertEqual(sorted(attempt.regime), ["backend", "batch", "phase"])
        self.assertEqual(attempt.proposal_ids, ("akp-20260803-0001",))
        self.assertEqual(attempt.candidate_ids, ("akc-20260803-0001",))
        self.assertIn("ake-20260803-0001", attempt.event_ids)
        self.assertEqual(attempt.anchor.source_commit, V8_COMMIT)


# =============================================================================
# Explainability
# =============================================================================

class TestEveryMatchIsExplainable(unittest.TestCase):
    """*"A rejection an agent cannot inspect is one it will route around."*"""

    def test_a_match_names_its_prior_attempt_its_events_and_its_class(self):
        ledger = _refuted_ledger()
        lookup = ledger.lookup(G15_REGIME)
        self.assertEqual(len(lookup.explanations), 1)
        row = lookup.explanations[0]
        self.assertTrue(row.matched)
        self.assertTrue(row.rejects)
        self.assertEqual(row.entry_class, H.MATCH_CLASS_MATCHED_NEGATIVE)
        self.assertIn("ake-20260803-0001", row.event_ids)
        self.assertEqual(row.hypothesis_ids, ("akh-g15-fusion",))
        self.assertEqual(row.proposal_ids, ("akp-20260803-0001",))
        self.assertTrue(row.reasons)
        self.assertTrue(any("same mechanism" in r for r in row.reasons))
        self.assertTrue(any("backend agrees" in r for r in row.reasons))

    def test_a_non_match_says_why_it_did_not_match(self):
        ledger = _refuted_ledger()
        lookup = ledger.lookup(dict(G15_REGIME, backend="llama_cpu"))
        self.assertEqual(lookup.matches, ())
        self.assertEqual(len(lookup.near_misses), 1)
        self.assertTrue(any("different backend" in r
                            for r in lookup.near_misses[0].reasons))

    def test_a_regime_key_that_did_nothing_is_reported(self):
        """The store refuses an unknown key because the operator believes it had an
        effect; the fold cannot refuse a durable record, so it REPORTS instead."""
        ledger = _refuted_ledger()
        lookup = ledger.lookup(dict(G15_REGIME, weather="rain"))
        self.assertTrue(any("'weather'" in i for i in lookup.ignored_dimensions))
        # …and a key that DID something is not reported as ignored. Without this the
        # channel cries wolf on every real dimension and stops being read.
        for key in ("backend", "phase", "mechanism", "ops"):
            with self.subTest(key=key):
                self.assertFalse(any(f"'{key}'" in i
                                     for i in lookup.ignored_dimensions))

    def test_a_declared_but_empty_dimension_is_reported_once_and_accurately(self):
        """`target.models: []` is a DECLARED dimension carrying no value.

        Both normalisation passes see every key, so the naive report says `models` is
        "not a declared match dimension" (the structural-target pass's view) as well as
        "declares no value" (the regime pass's). Two answers to one question, and the
        wrong one is the memorable one — an author reads it and adds `models` to the
        wrong axis.
        """
        entry = _refuted_ledger().attempts[0]
        models = [i for i in entry.ignored if i.startswith("'models'")]
        self.assertEqual(len(models), 1, entry.ignored)
        self.assertIn("declares no value", models[0])
        self.assertNotIn("not a declared match dimension", models[0])
        # CONTROL: a key that really is outside the vocabulary still says so.
        query = D.MatchQuery.from_regime(dict(G15_REGIME, weather="rain"))
        self.assertTrue(any("not a declared match dimension" in i
                            for i in query.ignored))

    def test_explain_is_canonical_json_safe(self):
        block = _refuted_ledger().explain(G15_REGIME)
        S.canonical_json(block)
        self.assertEqual(block["current_anchor"], ANCHOR_V8.short())
        self.assertEqual(len(block["matches"]), 1)


# =============================================================================
# Input discipline — an unreadable input is never an empty ledger
# =============================================================================

class TestTheFoldRefusesRatherThanDegrades(unittest.TestCase):

    def test_a_bare_string_is_refused_not_exploded_into_records(self):
        for kwargs in ({"journal_entries": "akj-1"},
                       {"hypothesis_events": "akh-1"},
                       # The EMPTY string is the dangerous one: `tuple("")` is `()`, so
                       # a per-element type check alone lets it through and the fold
                       # returns an empty ledger — which is the statement "nothing has
                       # ever been tried", from an unreadable input.
                       {"journal_entries": ""},
                       {"hypothesis_events": ""}):
            with self.subTest(**kwargs):
                with self.assertRaises(D.LedgerFoldError):
                    D.fold_journal(**kwargs)

    def test_a_wrongly_typed_element_is_refused(self):
        with self.assertRaises(D.LedgerFoldError):
            D.fold_journal(journal_entries=[{"kind": "PROPOSAL_RECORDED"}])

    def test_every_refusal_is_catchable_as_the_controller_plane(self):
        """A driver catches the plane, not a module of it."""
        from autokernel.controller import shared as SM
        self.assertTrue(issubclass(D.DoNotRepeatError, SM.ControllerError))
        self.assertTrue(issubclass(D.LedgerFoldError, D.DoNotRepeatError))

    def test_an_incoherent_hypothesis_ledger_raises_rather_than_folding_to_empty(self):
        """The fold delegates history legality to `hypotheses.fold_ledger`.

        A second opinion about one record is how two planes disagree without either
        being able to see it.
        """
        dangling = (H.LedgerEvent(
            seq=1, kind=H.EVENT_ATTEMPTED, hypothesis_id="akh-never-opened",
            at="2026-08-04T09:00:00.000000Z",
            payload={"attempt": H.Attempt(
                hypothesis_id="akh-never-opened", proposal_id="akp-1",
                disposition="evaluated", bears_on_falsifier=True, note="n",
            ).to_dict()},
        ),)
        with self.assertRaises(H.HypothesisLedgerCorruption):
            D.fold_journal(hypothesis_events=dangling)

    def test_a_non_scalar_dimension_value_is_reported_not_silently_dropped(self):
        """A dimension the matcher cannot read makes the match LOOSER.

        That is the invisible failure direction, so it is reported rather than
        skipped.
        """
        query = D.MatchQuery.from_regime(
            dict(G15_REGIME, backend={"nested": "mapping"}))
        self.assertNotIn("backend", query.regime)
        self.assertTrue(any("'backend'" in i for i in query.ignored))

    def test_canonical_token_does_not_conflate_a_number_with_its_spelling(self):
        self.assertNotEqual(D.canonical_token(128), D.canonical_token("128"))
        self.assertEqual(D.canonical_token("Elementwise/Norm  Fusion"),
                         D.canonical_token("elementwise_norm_fusion"))
        self.assertIsNone(D.canonical_token("   "))
        self.assertIsNone(D.canonical_token(None))

    def test_two_receipts_for_one_proposal_produce_one_entry(self):
        """Two ATTEMPT events against one proposal are two receipts for one thing
        tried; an entry per receipt would put two rows with one content-addressed
        `entry_id` in front of a reader."""
        events = list(_ledger_events(outcome=H.RESOLUTION_REFUTED))
        second = H.LedgerEvent(
            seq=4, kind=H.EVENT_ATTEMPTED, hypothesis_id="akh-g15-fusion",
            at="2026-08-04T09:20:00.000000Z",
            payload={"attempt": H.Attempt(
                hypothesis_id="akh-g15-fusion", proposal_id="akp-20260803-0001",
                disposition="re-evaluated", bears_on_falsifier=True,
                note="the same proposal, measured again",
            ).to_dict()},
        )
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(), _event_entry()),
            tuple(events[:2]) + (second,) + tuple(events[2:]),
        )
        self.assertEqual(len(ledger), 1)

    def test_the_fold_is_pure(self):
        """Same inputs, same ledger; and nothing is written anywhere."""
        entries = (_campaign_entry(), _proposal_entry(), _candidate_entry(),
                   _event_entry())
        events = _ledger_events(outcome=H.RESOLUTION_REFUTED)
        first = _fold(entries, events)
        second = _fold(entries, events)
        self.assertEqual([a.to_dict() for a in first.attempts],
                         [a.to_dict() for a in second.attempts])
        self.assertEqual(first.attempts[0].entry_id, second.attempts[0].entry_id)


# =============================================================================
# RED TEAM 2026-08-04 — five defects found by attacking the built module
#
# Each class below is the regression barrier for one of them. Every one is
# bite-verified: reverting the fix makes the named test fail, and each carries a
# COMPLIANT-PATH CONTROL so a fix that simply suppressed everything would be caught.
# =============================================================================

class TestAnUnanswerableQuestionIsNotAClearOne(unittest.TestCase):
    """RED TEAM 1 — the fail-open, and the one that mattered most.

    `matches_for()` returned `()` for a question it could not compare at all. An empty
    sequence is not a neutral value here: `check_do_not_repeat()` DEFINES it as "it WAS
    consulted and matched nothing" and returns PASS. So a question the matcher could
    not answer came back as *answered and clear*.

    And it was not a corner case, it was THE case. The operator drops in a one-line
    idea with a regime like `{"backend": "llama_gpu", "phase": "decode"}` and does not
    write `mechanism` — the key every match is made on. Every operator hypothesis
    therefore cleared a ledger holding a receipted negative about the same idea in the
    same regime, silently, on exactly the path the hypothesis work exists to serve.

    The module already knew the difference — `LedgerLookup.incomplete_comparisons` was
    written for it — and the knowledge did not reach the verdict.
    """

    def setUp(self):
        self.ledger = _refuted_ledger()
        self.answerable = dict(G15_REGIME)
        self.operator = {"backend": "llama_gpu", "phase": "decode",
                         "batch_band": "b128"}

    def test_the_operators_one_line_regime_is_not_a_pass(self):
        check = D.disposition(self.operator, self.ledger)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(any("names no 'mechanism'" in r for r in check.reasons))

    def test_control_a_comparable_question_still_passes_and_still_fails(self):
        """The anti-vacuous control: demoting EVERYTHING would satisfy the test above.

        A fully specified question that repeats the negative must still FAIL, and the
        same question against an empty ledger must still PASS — otherwise "the ledger
        answers" has been traded for "the ledger never answers", which is the
        toothless-ledger failure wearing the fix's clothes.
        """
        self.assertEqual(
            D.disposition(self.answerable, self.ledger).outcome, S.FAIL)
        self.assertEqual(
            D.disposition(self.answerable, D.CompiledLedger()).outcome, S.PASS)

    def test_matches_for_refuses_rather_than_saying_nothing_matched(self):
        with self.assertRaises(D.DoNotRepeatError) as ctx:
            self.ledger.matches_for(self.operator, G15_STATEMENT)
        self.assertIn("empty match set would say it did", str(ctx.exception))
        # ...and it still ANSWERS the question it can answer.
        self.assertEqual(len(self.ledger.matches_for(self.answerable, G15_STATEMENT)), 1)

    def test_a_dimension_the_matcher_could_not_read_is_not_silently_dropped(self):
        """A named dimension whose value cannot be read makes the query LOOSER.

        Dropping it and answering anyway is the same fail-open by another route: the
        question meant to constrain `backend` and the comparison proceeded without it.
        """
        unreadable = dict(G15_REGIME, backend={"nested": "mapping"})
        self.assertIn("backend", D.MatchQuery.from_regime(unreadable).unusable_dimensions)
        self.assertEqual(D.disposition(unreadable, self.ledger).outcome,
                         S.COULD_NOT_CHECK)
        # A mechanism naming two changes is the same shape: nothing was taken.
        two = dict(G15_REGIME, mechanism=["a", "b"])
        self.assertEqual(D.disposition(two, self.ledger).outcome, S.COULD_NOT_CHECK)

    def test_an_incomplete_comparison_demotes_a_pass(self):
        """"Genuinely new" and "you did not say enough" must not share an outcome."""
        # The entry was measured in `phase`; the question does not state it.
        partial = {k: v for k, v in G15_REGIME.items() if k != "phase"}
        lookup = self.ledger.lookup(partial)
        self.assertTrue(lookup.incomplete_comparisons)
        self.assertEqual(D.disposition(partial, self.ledger).outcome, S.COULD_NOT_CHECK)

    def test_an_incomplete_comparison_is_NEVER_a_route_around_a_receipted_negative(self):
        """A FAIL survives an incompleteness sitting beside it.

        This needs a question that is BOTH matched and incomplete — one entry it
        matches outright and another, about the same mechanism, that breaks only on a
        dimension the question does not state. Asserting the FAIL on a fully specified
        question proves nothing: there is no incompleteness there for the rule to
        outrank, so the guard could be deleted and the assertion would still hold.
        `check_do_not_repeat` applies the same precedence internally — a concrete
        receipted match is a FACT and outranks an incomplete comparison.
        """
        matched = _refuted_ledger().attempts[0]
        narrower = _fold(
            (_campaign_entry(), _proposal_entry(proposal_id="akp-20260803-0002"),
             _candidate_entry(candidate_id="akc-0002",
                              proposal_id="akp-20260803-0002"),
             _event_entry(event_id="ake-0002", status="fail",
                          candidate_id="akc-0002")),
            _ledger_events(
                hypothesis=_hypothesis(hid="akh-narrower",
                                       regime=dict(G15_REGIME, quant="iq2_m")),
                proposal_id="akp-20260803-0002", outcome=H.RESOLUTION_REFUTED),
        ).attempts[0]
        both = D.CompiledLedger((matched, narrower), current_anchor=ANCHOR_V8)
        lookup = both.lookup(G15_REGIME)
        self.assertTrue(lookup.matches, "control: one entry must MATCH")
        self.assertTrue(lookup.incomplete_comparisons,
                        "control: the other must be an incomplete comparison")
        self.assertEqual(D.disposition(G15_REGIME, both).outcome, S.FAIL)

    def test_disposition_does_not_restate_the_class_rules(self):
        """It joins; it does not re-decide. Advisory classes still leave a PASS."""
        advisory = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(),
             _event_entry(status="fail", anchor=ANCHOR_V7)),
            _ledger_events(outcome=H.RESOLUTION_REFUTED),
        )
        self.assertEqual(_classes(advisory), [H.MATCH_CLASS_SUPERSEDED_FACT])
        self.assertEqual(D.disposition(G15_REGIME, advisory).outcome, S.PASS)
        with self.assertRaises(TypeError):
            D.disposition(G15_REGIME, "not a ledger")


class TestTheMeasuredRegimeWinsOverTheClaimedOne(unittest.TestCase):
    """RED TEAM 2 — the fold unioned two regimes that disagreed.

    A hypothesis the operator filed under `backend: llama_cpu`, attempted by a proposal
    that ran in a GPU campaign, compiled to `backend: [llama_gpu, llama_cpu]` — and a
    CPU question was then rejected by a negative that was never measured on a CPU. The
    builder's own report names this exact hazard ("without it a GPU negative would match
    a CPU question") as the reason `backend` is joined off the campaign record; the
    merge put it back.
    """

    def _entry_for(self, claimed_backend):
        hypothesis = _hypothesis(regime=dict(G15_REGIME, backend=claimed_backend))
        return _fold(
            (_campaign_entry(backend="llama_gpu"), _proposal_entry(),
             _candidate_entry(), _event_entry(status="fail")),
            _ledger_events(hypothesis=hypothesis, outcome=H.RESOLUTION_REFUTED),
        )

    def test_a_gpu_negative_does_not_reject_a_cpu_question(self):
        ledger = self._entry_for("llama_cpu")
        entry = ledger.attempts[0]
        self.assertEqual(list(entry.regime_raw["backend"]), ["llama_gpu"])
        self.assertEqual(
            D.disposition(dict(G15_REGIME, backend="llama_cpu"), ledger).outcome,
            S.PASS)
        self.assertTrue(any("MEASURED at" in r for r in entry.why))

    def test_control_the_measured_regime_still_rejects_its_own_question(self):
        ledger = self._entry_for("llama_cpu")
        self.assertEqual(
            D.disposition(dict(G15_REGIME, backend="llama_gpu"), ledger).outcome,
            S.FAIL)

    def test_a_dimension_only_the_hypothesis_declares_still_constrains(self):
        """Filling in is not the defect; overriding is. A dimension the measured side
        never declared still comes from the hypothesis, and it NARROWS the entry."""
        hypothesis = _hypothesis(regime=dict(G15_REGIME, quant="iq2_m"))
        ledger = _fold(
            (_campaign_entry(), _proposal_entry(), _candidate_entry(),
             _event_entry(status="fail")),
            _ledger_events(hypothesis=hypothesis, outcome=H.RESOLUTION_REFUTED),
        )
        self.assertIn("quant", ledger.attempts[0].regime)
        self.assertEqual(
            D.disposition(dict(G15_REGIME, quant="iq2_m"), ledger).outcome, S.FAIL)
        self.assertEqual(
            D.disposition(dict(G15_REGIME, quant="q4_k_m"), ledger).outcome, S.PASS)


class TestAGateFailureIsNotANegativeAboutTheMechanism(unittest.TestCase):
    """RED TEAM 3 — §8.4.0's original defect, rebuilt on the orphan path.

    The fold excluded non-bearing ATTEMPTS from a hypothesis's entry precisely because
    "a hypothesis used to evaporate when its proposal failed for an unrelated reason —
    a build break". An ORPHAN proposal has no falsifier and so no `bears_on_falsifier`,
    and the rule there was `"fail" in statuses`: ANY failing event anywhere in the
    proposal's history was the negative.

    So a proposal that failed its T0 gate, was repaired, and then PASSED T1 compiled to
    a receipted MATCHED_NEGATIVE and rejected the very idea it had just been measured
    to support. The tier was in the record the whole time — `schemas` says outright
    that "only T0 compares artifacts rather than rates".
    """

    def _orphan(self, *events):
        return _fold((_campaign_entry(), _proposal_entry(), _candidate_entry())
                     + tuple(events))

    def test_a_t0_gate_failure_alone_is_not_a_negative(self):
        ledger = self._orphan(
            _event_entry(seq=4, event_id="ake-0001", status="fail", tier="T0"))
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_LOW_VALUE])
        self.assertEqual(D.disposition(G15_REGIME, ledger).outcome, S.PASS)
        self.assertTrue(any("T0" in r for r in ledger.attempts[0].why))

    def test_a_t0_failure_then_a_measured_pass_is_not_a_negative(self):
        ledger = self._orphan(
            _event_entry(seq=4, event_id="ake-0001", status="fail", tier="T0"),
            _event_entry(seq=5, event_id="ake-0002", status="pass"),
        )
        self.assertEqual(D.disposition(G15_REGIME, ledger).outcome, S.PASS)

    def test_a_history_that_both_passed_and_failed_is_conflicted(self):
        """§19.3's existing rule, reused rather than a seventh disposition invented."""
        ledger = self._orphan(
            _event_entry(seq=4, event_id="ake-0001", status="pass"),
            _event_entry(seq=5, event_id="ake-0002", status="fail"),
        )
        self.assertTrue(ledger.attempts[0].conflicted)
        self.assertEqual(D.disposition(G15_REGIME, ledger).outcome, S.PASS)

    def test_control_a_measured_failure_is_still_the_negative(self):
        """The anti-vacuous control: a T1 `fail` must still close the question."""
        ledger = self._orphan(_event_entry(status="fail"))
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        self.assertEqual(D.disposition(G15_REGIME, ledger).outcome, S.FAIL)

    def test_control_a_t0_pass_beside_a_measured_fail_is_still_the_negative(self):
        ledger = self._orphan(
            _event_entry(seq=4, event_id="ake-0001", status="pass", tier="T0"),
            _event_entry(seq=5, event_id="ake-0002", status="fail"),
        )
        self.assertEqual(D.disposition(G15_REGIME, ledger).outcome, S.FAIL)
        self.assertFalse(ledger.attempts[0].conflicted)


class TestAnImportedNegativeWithNoAnchorDoesNotBlockForever(unittest.TestCase):
    """RED TEAM 4 — a stale anchor permanently closing a question.

    The measurement path treats "names no anchor" as COULD_NOT_CHECK and demotes to
    SUPERSEDED_FACT, because an unobserved component is never a PASS. The constraint
    path checked only that a DECLARED `anchor_commit` still matched — so an imported
    `MATCHED_NEGATIVE` that named no commit at all rejected at v8, rejected at v7, and
    rejected with NO current anchor, which is the position the fold's own docstring
    says rejects nothing on measurement grounds.
    """

    QUESTION = {"backend": "llama_gpu", "phase": "decode",
                "mechanism": "elementwise_norm_fusion"}

    def _constraint(self, **over):
        payload = {
            "entry_class": H.MATCH_CLASS_MATCHED_NEGATIVE,
            "mechanism": "elementwise_norm_fusion",
            "regime": {"backend": "llama_gpu", "phase": "decode"},
            "receipt": "ake-20260101-0001 @ an-old-commit",
        }
        payload.update(over)
        return _entry(D.CONSTRAINT_EVENT_KIND, payload, seq=9,
                      event_id="akj-000000000009-constraint")

    def test_no_anchor_commit_supersedes_rather_than_rejects(self):
        for anchor in (ANCHOR_V8, ANCHOR_V7, None):
            with self.subTest(anchor=anchor):
                ledger = _fold((self._constraint(),), anchor=anchor)
                self.assertEqual(_classes(ledger), [H.MATCH_CLASS_SUPERSEDED_FACT])
                self.assertEqual(
                    D.disposition(self.QUESTION, ledger).outcome, S.PASS)

    def test_a_current_anchor_commit_still_rejects(self):
        """The anti-vacuous control: an entry that DOES bind must still close it."""
        ledger = _fold((self._constraint(anchor_commit=V8_COMMIT),), anchor=ANCHOR_V8)
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_MATCHED_NEGATIVE])
        self.assertEqual(D.disposition(self.QUESTION, ledger).outcome, S.FAIL)

    def test_a_hard_constraint_needs_no_anchor(self):
        """A hardware/policy prohibition is not measurement-derived and no anchor move
        supersedes it — the one rejecting class the anchor rule must NOT touch."""
        ledger = _fold(
            (self._constraint(entry_class=H.MATCH_CLASS_HARD_CONSTRAINT),),
            anchor=ANCHOR_V8,
        )
        self.assertEqual(_classes(ledger), [H.MATCH_CLASS_HARD_CONSTRAINT])
        self.assertEqual(D.disposition(self.QUESTION, ledger).outcome, S.FAIL)


class TestNoneIsNotAnEmptyRecord(unittest.TestCase):
    """RED TEAM 5 — `journal_entries=None` folded to "nothing has been tried".

    `_require_sequence` refused `""` with the words "would fold into an empty ledger,
    which is the statement 'nothing has been tried'", and then returned `()` for `None`
    — the value a failed read leaves behind.
    """

    def test_none_is_refused_on_both_record_streams(self):
        with self.assertRaises(D.LedgerFoldError):
            D.fold_journal(journal_entries=None)
        with self.assertRaises(D.LedgerFoldError):
            D.fold_journal(hypothesis_events=None)

    def test_control_omitting_them_is_still_a_legal_empty_ledger(self):
        """A fresh campaign has no records, and that is a real position."""
        self.assertEqual(len(D.fold_journal()), 0)
        self.assertEqual(
            D.disposition(G15_REGIME, D.fold_journal()).outcome, S.PASS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
