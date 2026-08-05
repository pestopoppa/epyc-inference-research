#!/usr/bin/env python3
"""test_hypothesis_path_end_to_end.py — the operator's idea, from JSON file to memory.

WHY THIS FILE EXISTS
--------------------
`hypotheses.py` and `do_not_repeat.py` each have a unit suite, both green, and for one
commit the two modules were not joined at all: `check_do_not_repeat()` was a correct
guard nothing called with a real ledger, `fold_journal()` was a correct fold nothing
compiled, and `authorize_claim()` spent claims without asking either. Each suite tested
its own half against a stub of the other, which is exactly the shape in which a seam
defect survives two green suites.

So this file tests neither module. It tests THE PATH, once, in one process, over real
files:

    the operator writes one JSON line with NO falsifier
      -> intake() tracks it and says a claim may not be spent on it
      -> authorize_claim() REFUSES, and writes nothing
      -> an agent proposes a falsifier
      -> the agent ADOPTS it: the entry leaves the store, and every other byte of the
         operator's file is unchanged
      -> a claim is authorized, with the memory plane consulted for the first time
      -> the proposal is attempted, evaluated, and the question resolves REFUTED
      -> the SAME IDEA, reworded by another agent, is now refused by the ledger
      -> the anchor MOVES, and the same idea is allowed again

That last hop is the one worth the file. A ledger that rejects forever is a blacklist;
a ledger that reopens when the thing it was measured against has moved is MEMORY. iqk
is the standing example — an idea that lost on v7 and won on v8 by +33-43% prefill —
and a loop that cannot make that distinction stops being able to learn.

NO inference, NO benchmark, NO build, NO model call, NO process spawn. Every file this
suite writes is under a `tempfile.TemporaryDirectory` it removes.

Run:
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_hypothesis_path_end_to_end.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.controller import do_not_repeat as D  # noqa: E402
from autokernel.controller import shared as FP  # was fingerprint  # noqa: E402
from autokernel.controller import hypotheses as H  # noqa: E402
from autokernel.evaluator import api as EV  # noqa: E402

CAMPAIGN = "ak-llama_gpu-decode-20260803"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
V9_COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
PROPOSAL_ID = "akp-20260803-0001"
CANDIDATE_ID = "akc-20260803-0001"
EVENT_ID = "ake-20260803-0001"
MECHANISM = "elementwise_norm_fusion"


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


def _anchor(commit: str, tag: str) -> EV.AnchorIdentity:
    return EV.AnchorIdentity(
        source_commit=commit,
        binary_sha256=_sha(f"anchor-binary-{tag}"),
        linkage_sha256=_sha(f"anchor-linkage-{tag}"),
        measurement_event_ids=(f"ake-anchor-{tag}",),
    )


ANCHOR_V8 = _anchor(V8_COMMIT, "v8")
ANCHOR_V9 = _anchor(V9_COMMIT, "v9")

#: What the operator actually types. One line, no falsifier, no mechanism token — the
#: regime is the three words they would think to write and nothing else.
OPERATOR_STATEMENT = (
    "the elementwise/norm cluster is where the B=128 decode time goes; fusing it should "
    "be worth 15%"
)
OPERATOR_REGIME = {"backend": "llama_gpu", "phase": "decode", "batch_band": "b128"}

#: The agent's own question about the same idea, later. An agent writes a `mechanism`
#: because §7.1's selection block makes it write one; the operator does not.
AGENT_REGIME = dict(OPERATOR_REGIME, mechanism=MECHANISM)


def _store_document(*entries) -> str:
    """The operator's file, with a human's ragged formatting deliberately preserved.

    Uneven indentation and a mix of one-line and multi-line entries, because the whole
    claim being tested is that the file comes back as the operator typed it.
    """
    rendered = []
    for index, entry in enumerate(entries):
        if index == 1:
            rendered.append(
                "    " + json.dumps(entry, indent=2).replace("\n", "\n    "))
        else:
            rendered.append("      " + json.dumps(entry))
    return (
        "{\n"
        '  "schema": "' + H.STORE_SCHEMA + '",\n'
        '  "hypotheses": [\n'
        + ",\n\n".join(rendered) +
        "\n  ]\n"
        "}\n"
    )


def _operator_entry() -> dict:
    """The MINIMUM the operator has to type. Note the absence of `falsifier`."""
    return {
        "hypothesis_id": "akh-g15-fusion",
        "statement": OPERATOR_STATEMENT,
        "regime": dict(OPERATOR_REGIME),
    }


def _neighbour(hid: str, statement: str) -> dict:
    """Another entry in the same file, so 'byte-identical apart from that entry' has
    something to be identical about."""
    return {
        "hypothesis_id": hid,
        "statement": statement,
        "falsifier": "the measured share stays inside the noise band",
        "regime": {"backend": "llama_cpu", "phase": "prefill"},
    }


def _proposal_payload() -> dict:
    """A §7 proposal the SHIPPED validator accepts with zero violations."""
    return {
        "schema": S.SCHEMA_PROPOSAL,
        "proposal_id": PROPOSAL_ID,
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
            "mechanism": MECHANISM,
            "hierarchy_layer": "kernel",
            "regime_identity": {},
        },
        "declared_symbol_deltas": {"added": [], "removed": [], "arity_changed": []},
        "campaign_kind": "fusion",
        "oracle_reference": {"oracle": None, "commit": None, "license_check": None},
        "novelty_basis": {
            "prior_event_ids": [], "source_receipts": [], "do_not_repeat_matches": [],
        },
        "expected_information_gain": 0.4,
        "representation_contract": _representation_contract(),
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


def _candidate_payload() -> dict:
    return {
        "schema": S.SCHEMA_CANDIDATE,
        "candidate_id": CANDIDATE_ID,
        "campaign_id": CAMPAIGN,
        "proposal_id": PROPOSAL_ID,
        "parent_candidate_id": None,
        "worktree": {
            "path": "/mnt/raid0/llm/llama.cpp-" + CAMPAIGN,
            "branch": f"ak/{CAMPAIGN}/akp-0001",
            "source_commit": V8_COMMIT,
            "clean": True,
        },
        "source_snapshot": {
            "snapshot_sha256": _sha("snapshot"),
            "patch_bundle_sha256": _sha("patch"),
        },
        "ancestry": {
            "production_base_commit": V8_COMMIT,
            "is_descendant_of_production_base": True,
            "proof": "git merge-base --is-ancestor 67a433bf.. HEAD -> 0",
        },
        "build": {
            "toolchain": "rocm-6.2", "compiler": "hipcc 6.2.0",
            "command": "cmake --build build -j 96",
            "build_dir": "/mnt/raid0/llm/tmp/ak-build/akc-0001",
            "log_path": f"data/{CAMPAIGN}/build/akc-0001.log",
            "log_sha256": _sha("build-log"),
        },
        "artifacts": {
            "binary_sha256": _sha("candidate-binary"),
            "linkage_sha256": _sha("candidate-linkage"),
            "library_sha256s": {"libggml.so": _sha("libggml")},
        },
        "dispatch": {"feature_flags": ["GGML_AK_FUSE_NORM"],
                     "dispatch_predicate": "K >= 4096"},
        "affected_surface": {
            "derived_sha256": _sha("derived-surface"),
            "traced_sha256": None, "reconciled": False,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "evaluator": {"id": "P-AK-SEARCH-1/v1",
                      "bundle_sha256": _sha("evaluator-bundle")},
        "receipts": {
            "host_receipt": "rcpt-host-20260803T101500Z",
            "resource_claim_receipt": "rcpt-gpu-claim-0042",
        },
        "storage": {"footprint_gb": 3.4,
                    "durability_class": "hash_and_provenance_only"},
        "evaluation_event_ids": [],
        "derived_verdicts": {},
        "controller": {
            "provider": "local", "model_id": "architect-a4", "effort": "high",
            "prompt_bundle_sha256": _sha("prompt-bundle"),
        },
        "champion_status": "none",
        "status": "built",
        "supersession_reason": None,
        "created_at": "2026-08-03T10:15:00+00:00",
    }


def _evaluation_payload() -> dict:
    """The measurement that refutes it: a T1 FAIL, anchored on v8."""
    return {
        "schema": S.SCHEMA_EVALUATION_EVENT_V3,
        "event_id": EVENT_ID,
        "campaign_id": CAMPAIGN,
        "candidate_id": CANDIDATE_ID,
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


class _PathCase(unittest.TestCase):
    """One temp root holding a real journal, a real ledger and a real operator file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.journal = J.Journal(os.path.join(self.root, "journal"),
                                 campaign_id=CAMPAIGN)
        self.journal.initialize()
        self.tracker = H.HypothesisTracker(
            journal_=self.journal,
            root=os.path.join(self.root, "controller"),
            campaign_id=CAMPAIGN,
        )
        self.store_path = os.path.join(self.root, "operator_hypotheses.json")

    def write_store(self, text: str) -> H.OperatorHypothesisStore:
        with open(self.store_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return H.OperatorHypothesisStore(self.store_path)

    def store_text(self) -> str:
        with open(self.store_path, encoding="utf-8") as handle:
            return handle.read()

    def ledger(self, *, anchor=ANCHOR_V8) -> D.CompiledLedger:
        return D.compile_for_tracker(self.tracker, current_anchor=anchor)


# =============================================================================
# THE PATH, in order, as one test
# =============================================================================

class TheOperatorsIdeaTravelsTheWholePath(_PathCase):
    """Every hop, in sequence, with the state asserted between each pair.

    Deliberately ONE test rather than nine. Each hop's precondition is the previous
    hop's postcondition on real files, and nine tests with nine setUps would each
    re-create the state they were meant to inherit — which is how a suite passes while
    the path between its steps is broken.
    """

    def test_the_whole_path(self):
        # ---- 0. the operator types one line, with no falsifier ---------------
        entry = _operator_entry()
        self.assertNotIn("falsifier", entry,
                         "the point of the amendment is that this key is optional")
        store = self.write_store(_store_document(
            _neighbour("akh-prefill-tail", "the prefill tail is bound by the K cache"),
            entry,
            _neighbour("akh-quarters", "quarter-fleet mode is worth it under 30B"),
        ))
        before = self.store_text()
        _scanned, spans_before = store.entry_spans()
        self.assertEqual(_scanned, before)

        # ---- 1. intake tracks it, and names the work it owes the operator ----
        report = self.tracker.intake(store)
        self.assertIn("akh-g15-fusion", report.opened)
        self.assertIn("akh-g15-fusion", report.awaiting_falsifier)
        tracked = self.tracker.get("akh-g15-fusion")
        self.assertEqual(tracked.falsifier_state, H.FALSIFIER_ABSENT)
        self.assertIsNone(tracked.falsifier_source)
        self.assertFalse(tracked.may_spend_a_claim)
        self.assertEqual(tracked.owner, H.OWNER_OPERATOR)

        # ---- 2. NO claim may be spent on it, and the refusal writes nothing --
        kinds_before = [e.kind for e in self.tracker.read().events]
        with self.assertRaises(H.FalsifierRequiredBeforeCompute) as ctx:
            self.tracker.authorize_claim(
                "akh-g15-fusion", purpose="one B=128 decode sweep",
                authorized_by="mainA", ledger=self.ledger(),
            )
        self.assertIn("'absent'", str(ctx.exception))
        self.assertIn("propose_falsifier", str(ctx.exception))
        self.assertEqual([e.kind for e in self.tracker.read().events], kinds_before)

        # ---- 3. an agent writes the predicate the operator did not have to ---
        self.tracker.propose_falsifier(
            "akh-g15-fusion",
            falsifier="a current wall-share map shows the cluster under 20%",
            proposed_by="mainA",
            rationale=("the claim is about where the decode time IS, so a wall-share "
                       "map under the threshold refutes it without a build"),
        )
        tracked = self.tracker.get("akh-g15-fusion")
        self.assertEqual(tracked.falsifier_state, H.FALSIFIER_STATED)
        self.assertEqual(tracked.falsifier_source, H.FALSIFIER_SOURCE_PROPOSED)
        # The QUESTION did not move. The operator's own words are still their words.
        self.assertEqual(tracked.hypothesis.statement, OPERATOR_STATEMENT)
        self.assertIsNone(tracked.hypothesis.falsifier)

        # ---- 4. the agent adopts it: it LEAVES the store ---------------------
        adoption = self.tracker.adopt(
            "akh-g15-fusion", store,
            adopted_by="mainA",
            reason="taking this into the fusion campaign as our own",
        )
        self.assertEqual(adoption.adopted_by, "mainA")
        self.assertEqual(self.tracker.get("akh-g15-fusion").owner, H.OWNER_AGENTS)
        self.assertEqual([h.hypothesis_id for h in store.load()],
                         ["akh-prefill-tail", "akh-quarters"])

        # …and the file is BYTE-IDENTICAL apart from that entry. Not "parses to the
        # same thing" — the operator opens this file in an editor.
        after = self.store_text()
        self.assertNotIn("akh-g15-fusion", after)
        self.assertIn(adoption.entry_text, before)
        # 1. every surviving entry, character for character, exactly as typed
        for span in spans_before:
            if span.hypothesis_id == "akh-g15-fusion":
                continue
            self.assertIn(span.text_of(before), after,
                          f"{span.hypothesis_id} was reformatted by the removal")
        # 2. the bytes BEFORE the first entry and AFTER the last one — the preamble and
        #    the tail are where a rewrite-and-dump would show up first
        first, last = spans_before[0], spans_before[-1]
        self.assertEqual(before[:first.start], after[:first.start])
        self.assertEqual(before[last.end:], after[len(after) - len(before[last.end:]):])
        # 3. the deletion is ONE CONTIGUOUS REGION, and everything in it — once the
        #    separating comma and its whitespace are set aside — came out of that one
        #    entry. This is the assertion that would catch a rewrite-and-dump: a
        #    re-serialized file diverges early and converges late, so its removed
        #    region is the whole array, not one entry's characters.
        prefix = os.path.commonprefix([before, after])
        suffix = len(os.path.commonprefix([before[::-1], after[::-1]]))
        removed_region = before[len(prefix):len(before) - suffix]
        self.assertIn(
            removed_region.strip(" \t\r\n,"), adoption.entry_text,
            "the characters that left the file are not all from the adopted entry",
        )
        # anti-vacuity: a splice that removed nothing would satisfy the line above
        self.assertGreater(len(removed_region), len(adoption.entry_text) * 0.9)

        # An adopted hypothesis is absent from the store BY DESIGN, and intake must
        # not report a designed outcome as an anomaly.
        report = self.tracker.intake(store)
        self.assertEqual(report.open_but_absent_from_store, ())
        self.assertEqual(report.adopted_but_still_in_store, ())

        # ---- 5. now a claim may be spent, and memory is consulted ------------
        token = self.tracker.authorize_claim(
            "akh-g15-fusion", purpose="one B=128 decode sweep",
            authorized_by="mainA", ledger=self.ledger(),
        )
        # COULD_NOT_CHECK, not PASS, and this is the honest answer: the operator's
        # regime names no `mechanism`, so the ledger cannot compare the question
        # against anything and REFUSES rather than returning "nothing matched".
        self.assertEqual(token.do_not_repeat_outcome, S.COULD_NOT_CHECK)
        self.assertTrue(token.do_not_repeat_reasons)
        # It still opens the door: a wrong suppression is silent and permanent, a
        # wasted re-run is loud and costs one claim (§19.3).
        acquired = []
        H.claim_for_hypothesis(token, lambda **kw: acquired.append(kw),
                               device_id="mi210_0")
        self.assertEqual(len(acquired), 1)
        self.assertIn("wall-share map", acquired[0]["purpose"])

        # ---- 6. the proposal runs, and the measurement refutes it ------------
        self.journal.append(J.KIND_PROPOSAL_RECORDED, _proposal_payload())
        self.journal.append(J.KIND_CANDIDATE_RECORDED, _candidate_payload())
        self.journal.append(J.KIND_EVALUATION_EVENT, _evaluation_payload())
        self.tracker.note_attempt(
            "akh-g15-fusion", proposal_id=PROPOSAL_ID, disposition="evaluated",
            bears_on_falsifier=True,
            note="paired alternating blocks, five reps, full machine",
            refs=(EVENT_ID,),
        )
        self.tracker.resolve("akh-g15-fusion", H.ResolutionEvidence(
            outcome=H.RESOLUTION_REFUTED,
            evidence_grade=H.GRADE_PROTOCOL_BOUND,
            evidence_refs=(EVENT_ID,),
            falsifier_observed=(
                "the wall-share map put the cluster at 11%, under the 20% line"
            ),
            bears_on_falsifier=True,
            resolved_by="mainA",
        ))
        self.assertFalse(self.tracker.get("akh-g15-fusion").is_open)

        # ---- 7. the SAME IDEA, reworded, is now refused ----------------------
        ledger_v8 = self.ledger(anchor=ANCHOR_V8)
        self.assertTrue(len(ledger_v8), "the resolution must have compiled to an entry")
        reworded = H.Hypothesis(
            hypothesis_id="akh-norm-fusion-again",
            statement=(
                "collapsing the RMS-norm and elementwise ops into a single launch is "
                "the win at batch 128"
            ),
            falsifier="the fused path is not faster than the two-launch path",
            origin=H.ORIGIN_PLANNER,
            author="mainB",
            regime=dict(AGENT_REGIME),
        )
        self.tracker.open_hypothesis(reworded)
        with self.assertRaises(H.RepeatsAReceiptedNegative) as ctx:
            self.tracker.authorize_claim(
                "akh-norm-fusion-again", purpose="fusion sweep",
                authorized_by="mainB", ledger=ledger_v8,
            )
        self.assertIn("receipt", str(ctx.exception).lower())
        # And nothing was recorded: a refused spend is not a spend.
        self.assertNotIn(
            H.EVENT_CLAIM_AUTHORIZED,
            [e.kind for e in self.tracker.read().events
             if e.hypothesis_id == "akh-norm-fusion-again"],
        )
        # The round block says the same thing, in the same words, to the planner.
        block = D.planner_round_block(self.tracker, ledger_v8, round_id="r-1")
        rendered = {e["hypothesis_id"]: e for e in block["still_open"]}
        self.assertEqual(
            rendered["akh-norm-fusion-again"]["do_not_repeat"]["outcome"], S.FAIL)

        # ---- 8. the anchor MOVES, and the same idea is allowed again ---------
        ledger_v9 = self.ledger(anchor=ANCHOR_V9)
        token = self.tracker.authorize_claim(
            "akh-norm-fusion-again", purpose="fusion sweep on the new anchor",
            authorized_by="mainB", ledger=ledger_v9,
        )
        self.assertIn(token.do_not_repeat_outcome, (S.PASS, S.COULD_NOT_CHECK))
        # It is not silence — the ledger still SAYS the prior result, as advice.
        self.assertTrue(
            any(H.MATCH_CLASS_SUPERSEDED_FACT in reason
                for reason in token.do_not_repeat_reasons),
            f"expected a SUPERSEDED_FACT advisory, got {token.do_not_repeat_reasons}",
        )
        # The refuted question is still refuted. Moving the anchor reopens the FAMILY,
        # never the individual resolution — that costs new evidence.
        self.assertFalse(self.tracker.get("akh-g15-fusion").is_open)

        # ---- 9. the operator can find out what happened to their idea --------
        trace = self.tracker.trace("akh-g15-fusion")
        self.assertEqual(trace.hypothesis_id, "akh-g15-fusion")
        self.assertIsNotNone(trace.adoption)
        self.assertIn("refuted", trace.answer)


# =============================================================================
# The seam functions, each with the failure they exist to make impossible
# =============================================================================

class TheClaimGateCannotBeReachedWithoutMemory(_PathCase):
    """`authorize_claim` has no ledger default, and the door re-derives the verdict."""

    def _open(self) -> None:
        self.tracker.open_hypothesis(H.Hypothesis(
            hypothesis_id="akh-g15-fusion", statement=OPERATOR_STATEMENT,
            falsifier="a current wall-share map shows the cluster under 20%",
            origin=H.ORIGIN_OPERATOR, author="operator", regime=dict(AGENT_REGIME),
        ))

    def test_the_ledger_argument_has_no_default(self):
        """A default would rebuild the original defect: every caller that forgot the
        argument would silently get the unconsulted behaviour."""
        import inspect
        parameter = inspect.signature(
            H.HypothesisTracker.authorize_claim).parameters["ledger"]
        self.assertIs(parameter.default, inspect.Parameter.empty)
        self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_passing_no_ledger_is_refused_rather_than_treated_as_empty(self):
        self._open()
        with self.assertRaises(H.LedgerNotConsulted) as ctx:
            self.tracker.authorize_claim(
                "akh-g15-fusion", purpose="p", authorized_by="a", ledger=None)
        self.assertIn("CompiledLedger()", str(ctx.exception))

    def test_the_control_an_empty_compiled_ledger_is_a_real_answer(self):
        """The compliant path: 'nothing has been tried' must still open the door."""
        self._open()
        token = self.tracker.authorize_claim(
            "akh-g15-fusion", purpose="p", authorized_by="a",
            ledger=D.CompiledLedger())
        self.assertEqual(token.do_not_repeat_outcome, S.PASS)
        self.assertEqual(token.do_not_repeat_reasons, ())

    def test_a_ledger_that_is_not_one_is_refused(self):
        self._open()
        with self.assertRaises(TypeError):
            self.tracker.authorize_claim(
                "akh-g15-fusion", purpose="p", authorized_by="a", ledger=object())

    def test_a_token_with_no_verdict_cannot_reach_an_acquirer(self):
        """The second gate, at the door, re-derived rather than trusted."""
        self._open()
        token = self.tracker.authorize_claim(
            "akh-g15-fusion", purpose="p", authorized_by="a",
            ledger=D.CompiledLedger())
        object.__setattr__(token, "do_not_repeat_outcome", None)
        calls = []
        with self.assertRaises(H.LedgerNotConsulted):
            H.claim_for_hypothesis(token, lambda **kw: calls.append(kw))
        self.assertEqual(calls, [])

    def test_a_token_edited_to_carry_a_fail_cannot_reach_an_acquirer(self):
        self._open()
        token = self.tracker.authorize_claim(
            "akh-g15-fusion", purpose="p", authorized_by="a",
            ledger=D.CompiledLedger())
        object.__setattr__(token, "do_not_repeat_outcome", S.FAIL)
        calls = []
        with self.assertRaises(H.RepeatsAReceiptedNegative):
            H.claim_for_hypothesis(token, lambda **kw: calls.append(kw))
        self.assertEqual(calls, [])

    def test_a_fail_verdict_cannot_be_put_in_a_token_at_all(self):
        with self.assertRaises(H.RepeatsAReceiptedNegative):
            H.ClaimAuthorization(
                hypothesis_id="akh-x", falsifier="a real predicate",
                falsifier_source=H.FALSIFIER_SOURCE_STATED,
                origin=H.ORIGIN_OPERATOR, purpose="p", authorized_by="a",
                authorized_at="2026-08-04T00:00:00.000000Z", ledger_seq=1,
                do_not_repeat_outcome=S.FAIL,
                do_not_repeat_reasons=("dnr-1 already records this",),
            )

    def test_the_verdict_survives_the_record_round_trip(self):
        self._open()
        token = self.tracker.authorize_claim(
            "akh-g15-fusion", purpose="p", authorized_by="a",
            ledger=D.CompiledLedger())
        stored = self.tracker.get("akh-g15-fusion").claim_authorizations[-1]
        self.assertEqual(stored, token)
        self.assertEqual(stored.do_not_repeat_outcome, S.PASS)

    def test_a_record_written_before_this_seam_existed_reads_as_not_consulted(self):
        """An old record did not consult the ledger; `None` is the true statement."""
        record = {
            "hypothesis_id": "akh-x", "falsifier": "a real predicate",
            "falsifier_source": H.FALSIFIER_SOURCE_STATED,
            "origin": H.ORIGIN_OPERATOR, "purpose": "p", "authorized_by": "a",
            "authorized_at": "2026-08-04T00:00:00.000000Z", "ledger_seq": 1,
        }
        token = H.ClaimAuthorization.from_dict(record)
        self.assertIsNone(token.do_not_repeat_outcome)
        with self.assertRaises(H.LedgerNotConsulted):
            H.claim_for_hypothesis(token, lambda **kw: kw)


class CompileForTrackerReadsOneRecord(_PathCase):

    def test_it_refuses_anything_that_is_not_a_tracker(self):
        with self.assertRaises(TypeError):
            D.compile_for_tracker(object())

    def test_an_empty_campaign_compiles_to_an_empty_ledger(self):
        ledger = D.compile_for_tracker(self.tracker, current_anchor=ANCHOR_V8)
        self.assertEqual(len(ledger), 0)
        self.assertEqual(ledger.current_anchor, ANCHOR_V8)

    def test_a_torn_tail_is_a_refusal_not_a_shorter_ledger(self):
        """A suppression that comes and goes depending on where a process died is
        worse than either answer."""
        self.tracker.open_hypothesis(H.Hypothesis(
            hypothesis_id="akh-g15-fusion", statement=OPERATOR_STATEMENT,
            falsifier="a wall-share map shows the cluster under 20%",
            origin=H.ORIGIN_OPERATOR, author="operator", regime=dict(AGENT_REGIME),
        ))
        with open(self.tracker.ledger.path, "a", encoding="utf-8") as handle:
            handle.write('{"seq": 2, "kind": "HYPOTH')
        with self.assertRaises(D.LedgerFoldError) as ctx:
            D.compile_for_tracker(self.tracker)
        self.assertIn("repair_torn_tail", str(ctx.exception))
        # The control: repaired, it compiles.
        self.tracker.repair_torn_tail()
        self.assertEqual(len(D.compile_for_tracker(self.tracker)), 0)


class MatchesByHypothesisNeverInventsAClearAnswer(_PathCase):

    def setUp(self) -> None:
        super().setUp()
        self.tracker.open_hypothesis(H.Hypothesis(
            hypothesis_id="akh-vague", statement=OPERATOR_STATEMENT, falsifier=None,
            origin=H.ORIGIN_OPERATOR, author="operator", regime=dict(OPERATOR_REGIME),
        ))
        self.tracker.open_hypothesis(H.Hypothesis(
            hypothesis_id="akh-specific", statement="a different idea entirely",
            falsifier="the fused path is not faster", origin=H.ORIGIN_PLANNER,
            author="mainB", regime=dict(AGENT_REGIME),
        ))

    def test_an_uncomparable_question_maps_to_none_not_to_an_empty_tuple(self):
        """`()` means 'consulted, matched nothing' and comes back PASS. A question the
        ledger could not compare must not say that."""
        mapping = D.matches_by_hypothesis(self.tracker, D.CompiledLedger())
        self.assertIsNone(mapping["akh-vague"])
        self.assertEqual(mapping["akh-specific"], ())

    def test_the_round_block_carries_the_difference_through(self):
        block = D.planner_round_block(
            self.tracker, D.CompiledLedger(), round_id="r-1")
        rendered = {e["hypothesis_id"]: e for e in block["still_open"]}
        self.assertEqual(rendered["akh-vague"]["do_not_repeat"]["outcome"],
                         S.COULD_NOT_CHECK)
        self.assertEqual(rendered["akh-specific"]["do_not_repeat"]["outcome"], S.PASS)

    def test_every_question_is_keyed_open_or_resolved(self):
        self.tracker.resolve("akh-specific", H.ResolutionEvidence(
            outcome=H.RESOLUTION_INCONCLUSIVE,
            evidence_grade=H.GRADE_OBSERVATION,
            evidence_refs=("ake-20260803-0009",),
            falsifier_observed="the window was voided; nothing was observed",
            bears_on_falsifier=True, resolved_by="mainB",
        ))
        mapping = D.matches_by_hypothesis(self.tracker, D.CompiledLedger())
        self.assertEqual(sorted(mapping), ["akh-specific", "akh-vague"])


class TheOperatorDocumentIsCheckedAgainstTheCode(_PathCase):
    """`HYPOTHESES.md` is the file that makes the drop-in usable. It must be TRUE.

    A document telling the operator which keys are refused is a table, and a table is
    not an enforcement — it is a second copy of one, which drifts. Every factual claim
    the document makes about the store is asserted here against the store itself.
    """

    DOC = Path(__file__).resolve().parent.parent / "HYPOTHESES.md"

    def setUp(self) -> None:
        super().setUp()
        self.text = self.DOC.read_text(encoding="utf-8")

    def _json_blocks(self) -> list:
        blocks, rest = [], self.text
        while "```json" in rest:
            _head, rest = rest.split("```json", 1)
            body, rest = rest.split("```", 1)
            blocks.append(body)
        return blocks

    def test_the_document_exists_and_names_the_schema(self):
        self.assertTrue(self.DOC.is_file(), f"{self.DOC} is missing")
        self.assertIn(H.STORE_SCHEMA, self.text)

    def test_the_example_file_is_a_file_the_store_actually_loads(self):
        """The one assertion that matters: what the operator is told to type, works."""
        blocks = self._json_blocks()
        self.assertEqual(len(blocks), 1, "expected exactly one example store document")
        with open(self.store_path, "w", encoding="utf-8") as handle:
            handle.write(blocks[0])
        store = H.OperatorHypothesisStore(self.store_path)
        stated = store.load()
        self.assertEqual(len(stated), 1)
        entry = stated[0]
        # …and it really is the minimum the document claims it is.
        self.assertIsNone(entry.falsifier)
        self.assertEqual(entry.origin, H.ORIGIN_OPERATOR)
        self.assertEqual(entry.evidence_grade, H.GRADE_DESIGN_PRIOR)
        # …and it travels: tracked, and named as owed a falsifier.
        report = self.tracker.intake(store)
        self.assertEqual(report.opened, (entry.hypothesis_id,))
        self.assertEqual(report.awaiting_falsifier, (entry.hypothesis_id,))

    def test_every_key_the_document_says_is_refused_is_refused(self):
        listed = {
            key for key in set(H._REFUSED_ENTRY_KEYS) | set(H._ALLOWED_ENTRY_KEYS)
            if f"`{key}`" in self.text
        }
        missing = sorted(set(H._REFUSED_ENTRY_KEYS) - listed)
        self.assertEqual(
            missing, [],
            f"the store refuses {missing} and HYPOTHESES.md does not say so; the "
            "operator meets the refusal at load instead of reading it",
        )

    def test_it_does_not_promise_a_key_the_store_would_refuse(self):
        """The other direction, and the one that would actively mislead."""
        table = self.text.split("### The four other fields you may use", 1)[1]
        table = table.split("### What the file will not accept", 1)[0]
        promised = {key for key in set(H._REFUSED_ENTRY_KEYS) if f"`{key}`" in table}
        self.assertEqual(promised, set(),
                         f"HYPOTHESES.md offers refused key(s) {sorted(promised)}")
        offered = {key for key in set(H._ALLOWED_ENTRY_KEYS) if f"`{key}`" in table}
        self.assertEqual(
            offered, set(H._ALLOWED_ENTRY_KEYS) - {"hypothesis_id", "statement"},
            "the optional-field table and the store's allowed keys disagree",
        )

    def test_the_placeholder_it_warns_about_is_really_refused(self):
        self.assertIn('"tbd"', self.text)
        with self.assertRaises(H.HypothesisError):
            H.Hypothesis(
                hypothesis_id="akh-x", statement="an idea", falsifier="tbd",
                origin=H.ORIGIN_OPERATOR, author="operator",
            )

    def test_the_files_it_points_at_exist(self):
        for name in ("controller/hypotheses.py", "controller/do_not_repeat.py",
                     "controller/test_hypothesis_path_end_to_end.py"):
            self.assertIn(name, self.text)
            self.assertTrue((self.DOC.parent / name).is_file(), name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
