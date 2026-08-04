#!/usr/bin/env python3
"""test_hypotheses.py — the regression barrier for the operator channel (§8.4.0, AK-D38).

WHY THIS FILE EXISTS
--------------------
Each property below is one that the predecessor loop did not have, verified against
its source rather than assumed (§8.4.0's 2026-08-03 correction), and each one is
asserted here rather than described in a docstring:

  * **the falsifier is mandatory.** AutoPilot's defaulted to `""`, its rationale
    sidecar was observability-only, and a missing block did not abort a trial. Here a
    hypothesis without one — absent, empty, a placeholder, a paragraph, or a restatement
    of itself — is REFUSED at load.
  * **origin cannot raise grade.** There is no field to set, no store key that states
    one, no branch in `entry_grade`, and a resolution carrying `protocol_bound`
    evidence still leaves the hypothesis at `design_prior`. Tested per origin, not once.
  * **an attempt never resolves.** A proposal that failed for an unrelated reason
    leaves the question OPEN with a receipt saying what was tried — the failure mode
    that otherwise leaves a question feeling "already tried" with nothing to show.
  * **a malformed store raises.** Never an empty list, because an empty list is the
    statement *"the operator has no hypotheses"* and the planner acts on it.
  * **deleting the line does not close the question**, and rewriting a falsifier under
    a tracked id is refused — the two edits that would otherwise let a file answer a
    question that evidence never answered.

NO inference, NO benchmark, NO build, NO model call, NO process. Every file this suite
writes lives under a per-test temporary directory.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_hypotheses.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/controller/test_hypotheses.py
"""
from __future__ import annotations

import dataclasses
import fcntl
import inspect
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

# Import through the PACKAGE so `hypotheses.schemas` is the same module object the
# journal validates with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import journal as J  # noqa: E402
from autokernel import schemas as S  # noqa: E402
from autokernel.controller import hypotheses as H  # noqa: E402
from autokernel.controller import state_machine as SM  # noqa: E402
from autokernel.resource import device_claim as _dc  # noqa: E402

CAMPAIGN = "ak-llama_gpu-decode-20260803"

# The operator's own worked example from §8.4.0.
G15_STATEMENT = (
    "G15's elementwise/norm cluster is where the B=128 decode time is, and fusing it "
    "lands >= 15%"
)
G15_FALSIFIER = "a current wall-share map shows the cluster under 20%"
G15_REGIME = {"backend": "llama_gpu", "phase": "decode", "batch_band": "b128"}


def _hypothesis(
    hid: str = "akh-g15-fusion",
    *,
    origin: str = H.ORIGIN_OPERATOR,
    statement: str = G15_STATEMENT,
    falsifier: str = G15_FALSIFIER,
    regime=None,
    author: str = "operator",
) -> H.Hypothesis:
    return H.Hypothesis(
        hypothesis_id=hid,
        statement=statement,
        falsifier=falsifier,
        origin=origin,
        author=author,
        regime=dict(G15_REGIME if regime is None else regime),
    )


def _evidence(
    outcome: str = H.RESOLUTION_REFUTED,
    *,
    grade: str = H.GRADE_PROTOCOL_BOUND,
    refs=("akj-000000000042-abcdef123456",),
    observed: str = "wall-share map puts the cluster at 14.2%, under the 20% line",
    bears: bool = True,
    by: str = "controller",
) -> H.ResolutionEvidence:
    return H.ResolutionEvidence(
        outcome=outcome,
        evidence_grade=grade,
        evidence_refs=tuple(refs),
        falsifier_observed=observed,
        bears_on_falsifier=bears,
        resolved_by=by,
    )


def _store_doc(*entries) -> dict:
    return {"schema": H.STORE_SCHEMA, "hypotheses": list(entries)}


def _g15_entry(**overrides) -> dict:
    entry = {
        "hypothesis_id": "akh-g15-fusion",
        "statement": G15_STATEMENT,
        "falsifier": G15_FALSIFIER,
        "author": "operator",
        "regime": dict(G15_REGIME),
    }
    entry.update(overrides)
    return entry


class _TempCase(unittest.TestCase):
    """A journal, a controller root and an operator store, per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = self._tmp.name
        self.journal_root = os.path.join(self.base, "journal")
        os.makedirs(self.journal_root, exist_ok=True)
        self.journal = J.Journal(self.journal_root, campaign_id=CAMPAIGN)
        self.journal.initialize()
        self.root = os.path.join(self.base, "controller")
        self.store_path = os.path.join(self.base, "operator_hypotheses.json")

    def tracker(self, **kwargs) -> H.HypothesisTracker:
        kwargs.setdefault("journal_", self.journal)
        kwargs.setdefault("root", self.root)
        kwargs.setdefault("campaign_id", CAMPAIGN)
        return H.HypothesisTracker(**kwargs)

    def write_store(self, doc) -> H.OperatorHypothesisStore:
        with open(self.store_path, "w", encoding="utf-8") as handle:
            if isinstance(doc, (str, bytes)):
                handle.write(doc if isinstance(doc, str) else doc.decode("utf-8"))
            else:
                json.dump(doc, handle)
        return H.OperatorHypothesisStore(self.store_path)


# =============================================================================
# 1. The falsifier is what makes a hypothesis a hypothesis (§8.4.0)
# =============================================================================

class FalsifierIsMandatoryTest(_TempCase):

    def test_falsifier_is_a_required_constructor_field(self):
        fields = {f.name: f for f in dataclasses.fields(H.Hypothesis)}
        self.assertIn("falsifier", fields)
        self.assertIs(fields["falsifier"].default, dataclasses.MISSING)
        self.assertIs(fields["falsifier"].default_factory, dataclasses.MISSING)
        with self.assertRaises(TypeError):
            H.Hypothesis(  # type: ignore[call-arg]
                hypothesis_id="akh-x", statement="s", origin=H.ORIGIN_PLANNER,
                author="planner",
            )

    def test_store_entry_without_a_falsifier_loads_in_the_absent_state(self):
        """2026-08-04: optional on OPERATOR entry, mandatory before compute.

        The refusal did not disappear — it MOVED (see `ClaimGateTest`). What must not
        happen here is the entry being rejected, because the one person whose barrier
        to entry should be zero is the operator.
        """
        entry = _g15_entry()
        del entry["falsifier"]
        store = self.write_store(_store_doc(entry))
        (loaded,) = store.load()
        self.assertIsNone(loaded.falsifier)
        self.assertEqual(loaded.falsifier_state, H.FALSIFIER_ABSENT)
        self.assertEqual(loaded.statement, G15_STATEMENT)
        # And it is still graded exactly as every other hypothesis is.
        self.assertEqual(loaded.evidence_grade, H.GRADE_DESIGN_PRIOR)

    def test_a_non_operator_hypothesis_still_needs_its_falsifier_at_entry(self):
        for origin in sorted(H.ORIGINS - {H.ORIGIN_OPERATOR}):
            with self.subTest(origin=origin):
                with self.assertRaises(H.FalsifierMissing) as ctx:
                    H.Hypothesis(hypothesis_id="akh-x", statement="s", falsifier=None,
                                 origin=origin, author="a")
                self.assertIn("must be stated WITH its falsifier", str(ctx.exception))

    def test_empty_falsifier_is_refused(self):
        store = self.write_store(_store_doc(_g15_entry(falsifier="   ")))
        with self.assertRaises(H.FalsifierMissing):
            store.load()

    def test_placeholder_falsifiers_are_refused(self):
        for placeholder in ("n/a", "N/A", "none", "TBD", "-", "?", "pending",
                            "unstated", "not applicable"):
            with self.subTest(placeholder=placeholder):
                store = self.write_store(_store_doc(_g15_entry(falsifier=placeholder)))
                with self.assertRaises(H.FalsifierMissing) as ctx:
                    store.load()
                self.assertIn("placeholder", str(ctx.exception))

    def test_multiline_falsifier_is_refused(self):
        store = self.write_store(_store_doc(
            _g15_entry(falsifier="the map shows under 20%\nand also the counters move")
        ))
        with self.assertRaises(H.FalsifierMissing) as ctx:
            store.load()
        self.assertIn("ONE LINE", str(ctx.exception))

    def test_falsifier_that_restates_the_hypothesis_is_refused(self):
        store = self.write_store(_store_doc(
            _g15_entry(falsifier=G15_STATEMENT.upper())
        ))
        with self.assertRaises(H.FalsifierMissing) as ctx:
            store.load()
        self.assertIn("restates", str(ctx.exception))

    def test_falsifier_missing_is_a_store_error_and_a_controller_error(self):
        # A driver catching the controller plane must catch this one too.
        self.assertTrue(issubclass(H.FalsifierMissing, H.HypothesisStoreError))
        self.assertTrue(issubclass(H.HypothesisStoreError, H.HypothesisError))
        self.assertTrue(issubclass(H.HypothesisError, SM.ControllerError))

    def test_every_tracked_hypothesis_carries_a_falsifier_into_the_round_block(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.open_hypothesis(_hypothesis("akh-planner-tile", origin=H.ORIGIN_PLANNER,
                                            statement="tile 32 beats tile 16 at B=1",
                                            falsifier="no measurable delta at B=1",
                                            author="planner"))
        block = tracker.planner_round_block(round_id="r1")
        self.assertEqual(block["open_count"], 2)
        for entry in block["still_open"]:
            self.assertTrue(entry["falsifier"].strip())


# =============================================================================
# 2. A malformed store RAISES — it never degrades to an empty list
# =============================================================================

class MalformedStoreRaisesTest(_TempCase):

    def test_empty_hypotheses_list_is_the_only_way_to_say_none(self):
        store = self.write_store(_store_doc())
        self.assertEqual(store.load(), ())

    def test_absent_store_raises_rather_than_reporting_none(self):
        store = H.OperatorHypothesisStore(os.path.join(self.base, "not-there.json"))
        self.assertFalse(store.exists())
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("An absent store is not an empty one", str(ctx.exception))

    def test_empty_file_raises(self):
        store = self.write_store("")
        with self.assertRaises(H.HypothesisStoreError):
            store.load()

    def test_truncated_json_raises(self):
        store = self.write_store('{"schema": "epyc.autokernel.operator_hypo')
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("unparseable", str(ctx.exception))

    def test_json_array_at_the_top_level_raises(self):
        store = self.write_store("[]")
        with self.assertRaises(H.HypothesisStoreError):
            store.load()

    def test_wrong_schema_string_raises(self):
        store = self.write_store({"schema": "something.else.v1", "hypotheses": []})
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("schema", str(ctx.exception))

    def test_absent_hypotheses_key_raises(self):
        store = self.write_store({"schema": H.STORE_SCHEMA})
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("Write [] to say the operator has none", str(ctx.exception))

    def test_hypotheses_not_a_list_raises(self):
        store = self.write_store({"schema": H.STORE_SCHEMA, "hypotheses": {}})
        with self.assertRaises(H.HypothesisStoreError):
            store.load()

    def test_entry_that_is_not_an_object_raises(self):
        store = self.write_store(_store_doc("just a sentence"))
        with self.assertRaises(H.HypothesisStoreError):
            store.load()

    def test_unknown_top_level_key_raises(self):
        doc = _store_doc(_g15_entry())
        doc["notes"] = "reminder to myself"
        store = self.write_store(doc)
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("unknown top-level keys", str(ctx.exception))

    def test_unknown_entry_key_raises(self):
        store = self.write_store(_store_doc(_g15_entry(confidence=0.9)))
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("unknown keys", str(ctx.exception))

    def test_duplicate_hypothesis_id_raises(self):
        store = self.write_store(_store_doc(_g15_entry(), _g15_entry()))
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("one id is one question", str(ctx.exception))

    def test_bad_id_prefix_raises(self):
        store = self.write_store(_store_doc(_g15_entry(hypothesis_id="g15-fusion")))
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("akh-", str(ctx.exception))

    def test_regime_that_is_not_a_mapping_raises(self):
        store = self.write_store(_store_doc(_g15_entry(regime=["llama_gpu"])))
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("regime", str(ctx.exception))

    def test_no_malformed_input_ever_returns_a_list(self):
        # The property stated as a property: every malformed shape RAISES, and the one
        # well-formed empty shape returns (). Nothing in between.
        malformed = [
            "", "{", "[]", '{"schema": "x", "hypotheses": []}',
            json.dumps({"schema": H.STORE_SCHEMA}),
            json.dumps({"schema": H.STORE_SCHEMA, "hypotheses": "none"}),
            json.dumps(_store_doc({"statement": "s", "falsifier": "f"})),
        ]
        for raw in malformed:
            with self.subTest(raw=raw[:40]):
                store = self.write_store(raw)
                with self.assertRaises(H.HypothesisStoreError):
                    store.load()
        store = self.write_store(json.dumps(_store_doc()))
        self.assertEqual(store.load(), ())

    def test_load_with_digest_reads_the_file_once(self):
        store = self.write_store(_store_doc(_g15_entry()))
        loaded, digest = store.load_with_digest()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(digest, store.content_sha256())
        self.assertEqual(loaded[0].source["store_sha256"], digest)


# =============================================================================
# 3. THE SAFETY PROPERTY: origin cannot raise grade (§8.4.0, AK-D38, §19.0 rule 4)
# =============================================================================

class OriginCannotRaiseGradeTest(_TempCase):

    def test_entry_grade_is_constant_over_every_declared_origin(self):
        grades = {origin: H.entry_grade(origin) for origin in H.ORIGINS}
        self.assertEqual(set(grades.values()), {H.GRADE_DESIGN_PRIOR})
        self.assertEqual(H.ENTRY_GRADE, H.GRADE_DESIGN_PRIOR)

    def test_entry_grade_refuses_an_undeclared_origin(self):
        with self.assertRaises(ValueError):
            H.entry_grade("chief_architect")

    def test_hypothesis_has_no_grade_field_to_set(self):
        names = {f.name for f in dataclasses.fields(H.Hypothesis)}
        self.assertEqual([n for n in names if "grade" in n.lower()], [])
        self.assertIsInstance(H.Hypothesis.evidence_grade, property)
        self.assertIsNone(H.Hypothesis.evidence_grade.fset)

    def test_hypothesis_is_frozen_so_nothing_can_be_regraded_after_the_fact(self):
        hypothesis = _hypothesis()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            hypothesis.statement = "something else"  # type: ignore[misc]

    def test_operator_origin_grades_exactly_like_planner_origin(self):
        for origin in sorted(H.ORIGINS):
            with self.subTest(origin=origin):
                hypothesis = _hypothesis(f"akh-{origin}", origin=origin, author=origin)
                self.assertEqual(hypothesis.evidence_grade, H.GRADE_DESIGN_PRIOR)

    def test_store_refuses_a_stated_evidence_grade(self):
        store = self.write_store(_store_doc(
            _g15_entry(evidence_grade=H.GRADE_PROTOCOL_BOUND)
        ))
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("evidence grade is DERIVED", str(ctx.exception))

    def test_store_refuses_every_laundering_key(self):
        for key in sorted(H._REFUSED_ENTRY_KEYS):
            with self.subTest(key=key):
                store = self.write_store(_store_doc(_g15_entry(**{key: "whatever"})))
                with self.assertRaises(H.HypothesisStoreError) as ctx:
                    store.load()
                self.assertIn(repr(key), str(ctx.exception))

    def test_store_refuses_an_origin_relabel(self):
        store = self.write_store(_store_doc(_g15_entry(origin=H.ORIGIN_PLANNER)))
        with self.assertRaises(H.HypothesisStoreError):
            store.load()

    def test_protocol_bound_resolution_does_not_promote_the_hypothesis(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.resolve("akh-g15-fusion", _evidence(
            H.RESOLUTION_CONFIRMED, grade=H.GRADE_PROTOCOL_BOUND
        ))
        tracked = tracker.get("akh-g15-fusion")
        self.assertEqual(tracked.status, H.RESOLUTION_CONFIRMED)
        self.assertEqual(tracked.resolution.evidence_grade, H.GRADE_PROTOCOL_BOUND)
        # The hypothesis's OWN grade is untouched: the evidence is protocol-bound, the
        # hunch that prompted it never becomes one.
        self.assertEqual(tracked.evidence_grade, H.GRADE_DESIGN_PRIOR)

    def test_round_block_names_the_two_grades_differently(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.resolve("akh-g15-fusion", _evidence(grade=H.GRADE_SOURCE_VERIFIED))
        block = tracker.planner_round_block(round_id="r1")
        entry = block["resolved"][0]
        self.assertEqual(entry["entry_evidence_grade"], H.GRADE_DESIGN_PRIOR)
        self.assertEqual(
            entry["resolution"]["resolution_evidence_grade"], H.GRADE_SOURCE_VERIFIED
        )
        self.assertNotIn("evidence_grade", entry)

    def test_audit_passes_on_the_shipped_contract(self):
        check = H.audit_no_origin_grade_promotion()
        self.assertEqual(check.outcome, S.PASS, check.reasons)
        self.assertTrue(check.passed)

    def test_audit_fails_if_the_store_stops_refusing_a_stated_grade(self):
        # The audit must be able to FAIL, or it is decoration. Removing the one refusal
        # that keeps the operator-editable input from stating its own grade trips it.
        original = H._REFUSED_ENTRY_KEYS
        H._REFUSED_ENTRY_KEYS = {
            k: v for k, v in original.items() if k != "evidence_grade"
        }
        try:
            check = H.audit_no_origin_grade_promotion()
        finally:
            H._REFUSED_ENTRY_KEYS = original
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("evidence_grade" in r for r in check.reasons))
        self.assertEqual(H.audit_no_origin_grade_promotion().outcome, S.PASS)


# =============================================================================
# 4. An attempt is not a resolution (the half AutoKernel lacked)
# =============================================================================

class AttemptNeverResolvesTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.tracker_ = self.tracker()
        self.tracker_.open_hypothesis(_hypothesis())

    def test_a_proposal_that_failed_for_an_unrelated_reason_leaves_it_open(self):
        self.tracker_.note_attempt(
            "akh-g15-fusion",
            proposal_id="akp-20260803-0001",
            disposition="build_failed",
            bears_on_falsifier=False,
            note="compile break in the repack path; the fusion was never executed",
        )
        tracked = self.tracker_.get("akh-g15-fusion")
        self.assertTrue(tracked.is_open)
        self.assertEqual(tracked.status, H.STATUS_OPEN)
        self.assertEqual(len(tracked.attempts), 1)
        self.assertFalse(tracked.attempts[0].bears_on_falsifier)

    def test_even_an_attempt_that_bears_on_the_falsifier_leaves_it_open(self):
        self.tracker_.note_attempt(
            "akh-g15-fusion",
            proposal_id="akp-20260803-0002",
            disposition="banked",
            bears_on_falsifier=True,
            note="T1 landed; the wall-share re-measure has not been read yet",
        )
        self.assertTrue(self.tracker_.get("akh-g15-fusion").is_open)
        self.assertEqual(
            [t.hypothesis_id for t in self.tracker_.still_open()], ["akh-g15-fusion"]
        )

    def test_note_attempt_cannot_write_a_status(self):
        # Structural, not behavioural: there is no parameter through which a caller
        # could set one.
        params = set(inspect.signature(H.HypothesisTracker.note_attempt).parameters)
        self.assertEqual(params & {"status", "outcome", "resolution", "resolve"}, set())

    def test_many_attempts_still_leave_it_open_with_every_receipt(self):
        for index in range(4):
            self.tracker_.note_attempt(
                "akh-g15-fusion",
                proposal_id=f"akp-20260803-{index:04d}",
                disposition="PROPOSAL_SKIPPED",
                bears_on_falsifier=False,
                note="filtered on the wall-share ceiling before drafting",
            )
        tracked = self.tracker_.get("akh-g15-fusion")
        self.assertTrue(tracked.is_open)
        self.assertEqual(len(tracked.attempts), 4)
        entry = self.tracker_.planner_round_block(round_id="r1")["still_open"][0]
        self.assertEqual(entry["attempt_count"], 4)
        self.assertEqual(
            {a["disposition"] for a in entry["attempts"]}, {"PROPOSAL_SKIPPED"}
        )

    def test_attempt_on_an_untracked_question_is_refused(self):
        with self.assertRaises(H.UnknownHypothesis):
            self.tracker_.note_attempt(
                "akh-never-opened", proposal_id="akp-1", disposition="banked",
                bears_on_falsifier=True, note="n",
            )

    def test_attempt_after_resolution_is_recorded_and_changes_nothing(self):
        self.tracker_.resolve("akh-g15-fusion", _evidence())
        self.tracker_.note_attempt(
            "akh-g15-fusion", proposal_id="akp-late", disposition="invalid",
            bears_on_falsifier=False, note="window voided by the anchor gate",
        )
        tracked = self.tracker_.get("akh-g15-fusion")
        self.assertEqual(tracked.status, H.RESOLUTION_REFUTED)
        self.assertEqual(len(tracked.attempts), 1)

    def test_bears_on_falsifier_must_be_an_explicit_bool(self):
        with self.assertRaises(TypeError):
            H.Attempt(
                hypothesis_id="akh-x", proposal_id="akp-1", disposition="banked",
                bears_on_falsifier=1, note="n",
            )


# =============================================================================
# 5. Resolution costs evidence
# =============================================================================

class ResolutionRequiresEvidenceTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.tracker_ = self.tracker()
        self.tracker_.open_hypothesis(_hypothesis())

    def test_resolution_without_evidence_refs_is_refused(self):
        with self.assertRaises(H.ResolutionEvidenceMissing) as ctx:
            _evidence(refs=())
        self.assertIn("at least one reference", str(ctx.exception))

    def test_resolution_that_does_not_bear_on_the_falsifier_is_refused(self):
        with self.assertRaises(H.ResolutionEvidenceMissing) as ctx:
            _evidence(bears=False)
        self.assertIn("note_attempt", str(ctx.exception))

    def test_resolution_without_an_observation_against_the_falsifier_is_refused(self):
        with self.assertRaises(H.ResolutionEvidenceMissing):
            _evidence(observed="   ")

    def test_unknown_outcome_and_grade_are_refused(self):
        with self.assertRaises(ValueError):
            _evidence("probably_true")
        with self.assertRaises(ValueError):
            _evidence(grade="operator_says_so")

    def test_resolving_an_untracked_question_is_refused(self):
        with self.assertRaises(H.UnknownHypothesis):
            self.tracker_.resolve("akh-nope", _evidence())

    def test_resolving_twice_is_refused(self):
        self.tracker_.resolve("akh-g15-fusion", _evidence())
        with self.assertRaises(H.HypothesisNotOpen) as ctx:
            self.tracker_.resolve("akh-g15-fusion", _evidence(H.RESOLUTION_CONFIRMED))
        self.assertIn("reopen()", str(ctx.exception))

    def test_an_operator_hypothesis_the_loop_refutes_is_refuted_on_the_record(self):
        # §8.4.0: "An operator hypothesis that the loop refutes is refuted, and the
        # record says so — that is the mechanism working."
        self.tracker_.resolve("akh-g15-fusion", _evidence(H.RESOLUTION_REFUTED))
        tracked = self.tracker_.get("akh-g15-fusion")
        self.assertEqual(tracked.status, H.RESOLUTION_REFUTED)
        self.assertEqual(tracked.hypothesis.origin, H.ORIGIN_OPERATOR)
        self.assertIn("14.2%", tracked.resolution.falsifier_observed)
        self.assertEqual(self.tracker_.still_open(), ())

    def test_all_three_resolutions_are_reachable(self):
        for index, outcome in enumerate(sorted(H.RESOLUTIONS)):
            hid = f"akh-outcome-{index}"
            self.tracker_.open_hypothesis(_hypothesis(hid, origin=H.ORIGIN_PLANNER,
                                                      author="planner"))
            self.tracker_.resolve(hid, _evidence(outcome))
            self.assertEqual(self.tracker_.get(hid).status, outcome)

    def test_inconclusive_is_a_resolution_not_a_synonym_for_open(self):
        self.tracker_.resolve("akh-g15-fusion", _evidence(H.RESOLUTION_INCONCLUSIVE))
        self.assertFalse(self.tracker_.get("akh-g15-fusion").is_open)
        self.assertEqual(self.tracker_.still_open(), ())


# =============================================================================
# 6. Reopening costs new evidence, and never destroys the old receipt
# =============================================================================

class ReopenTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.tracker_ = self.tracker()
        self.tracker_.open_hypothesis(_hypothesis())
        self.tracker_.resolve("akh-g15-fusion", _evidence(H.RESOLUTION_INCONCLUSIVE))

    def test_reopen_requires_new_evidence(self):
        with self.assertRaises(H.ResolutionEvidenceMissing):
            self.tracker_.reopen(
                "akh-g15-fusion", reason="asking again", new_evidence_refs=(),
                reopened_by="operator",
            )

    def test_reopen_preserves_the_resolution_it_supersedes(self):
        self.tracker_.reopen(
            "akh-g15-fusion",
            reason="the anchor moved and the wall-share map was re-measured",
            new_evidence_refs=("akj-000000000099-fedcba987654",),
            reopened_by="controller",
        )
        tracked = self.tracker_.get("akh-g15-fusion")
        self.assertTrue(tracked.is_open)
        self.assertIsNone(tracked.resolution)
        self.assertEqual(tracked.reopen_count, 1)
        self.assertEqual(len(tracked.superseded_resolutions), 1)
        self.assertEqual(
            tracked.superseded_resolutions[0].outcome, H.RESOLUTION_INCONCLUSIVE
        )

    def test_reopening_an_open_question_is_refused(self):
        self.tracker_.reopen(
            "akh-g15-fusion", reason="new map", new_evidence_refs=("akj-1",),
            reopened_by="controller",
        )
        with self.assertRaises(H.HypothesisNotOpen):
            self.tracker_.reopen(
                "akh-g15-fusion", reason="again", new_evidence_refs=("akj-2",),
                reopened_by="controller",
            )

    def test_reopening_an_untracked_question_is_refused(self):
        with self.assertRaises(H.UnknownHypothesis):
            self.tracker_.reopen(
                "akh-nope", reason="r", new_evidence_refs=("akj-1",),
                reopened_by="controller",
            )


# =============================================================================
# 7. Editing the file cannot answer a question (intake semantics)
# =============================================================================

class StoreEditsDoNotCloseQuestionsTest(_TempCase):

    def test_intake_opens_stated_hypotheses_and_is_idempotent(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        first = tracker.intake(store)
        self.assertEqual(first.opened, ("akh-g15-fusion",))
        events_after_first = len(tracker.read().events)
        second = tracker.intake(store)
        self.assertEqual(second.opened, ())
        self.assertEqual(second.already_tracked, ("akh-g15-fusion",))
        self.assertEqual(len(tracker.read().events), events_after_first)

    def test_deleting_the_line_does_not_close_the_question(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        store = self.write_store(_store_doc())
        report = tracker.intake(store)
        self.assertEqual(report.open_but_absent_from_store, ("akh-g15-fusion",))
        self.assertEqual(
            [t.hypothesis_id for t in tracker.still_open()], ["akh-g15-fusion"]
        )

    def test_no_store_at_all_does_not_close_anything(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        report = tracker.intake(None)
        self.assertIsNone(report.store_path)
        self.assertEqual(report.open_but_absent_from_store, ("akh-g15-fusion",))
        self.assertEqual(len(tracker.still_open()), 1)

    def test_a_configured_store_that_vanished_raises_rather_than_reporting_none(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        os.unlink(self.store_path)
        with self.assertRaises(H.HypothesisStoreError):
            tracker.intake(store)
        self.assertEqual(len(tracker.still_open()), 1)

    def test_leaving_a_resolved_hypothesis_in_the_store_does_not_reopen_it(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        tracker.resolve("akh-g15-fusion", _evidence(H.RESOLUTION_REFUTED))
        report = tracker.intake(store)
        self.assertEqual(report.resolved_but_still_in_store, ("akh-g15-fusion",))
        self.assertEqual(report.opened, ())
        self.assertEqual(
            tracker.get("akh-g15-fusion").status, H.RESOLUTION_REFUTED
        )

    def test_rewriting_the_falsifier_under_a_tracked_id_is_refused(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        store = self.write_store(_store_doc(
            _g15_entry(falsifier="the cluster is under 2% of wall share")
        ))
        with self.assertRaises(H.QuestionRewritten) as ctx:
            tracker.intake(store)
        self.assertIn("new id", str(ctx.exception))
        self.assertEqual(
            tracker.get("akh-g15-fusion").hypothesis.falsifier, G15_FALSIFIER
        )

    def test_rewriting_the_statement_or_regime_is_refused(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        for override in ({"statement": "a different claim entirely"},
                         {"regime": {"backend": "llama_cpu"}}):
            with self.subTest(override=sorted(override)):
                store = self.write_store(_store_doc(_g15_entry(**override)))
                with self.assertRaises(H.QuestionRewritten):
                    tracker.intake(store)

    def test_open_hypothesis_refuses_a_rewrite_and_a_duplicate(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        with self.assertRaises(H.HypothesisAlreadyTracked):
            tracker.open_hypothesis(_hypothesis())
        with self.assertRaises(H.QuestionRewritten):
            tracker.open_hypothesis(_hypothesis(falsifier="something else entirely"))

    def test_intake_reports_the_store_digest_it_acted_on(self):
        store = self.write_store(_store_doc(_g15_entry()))
        report = self.tracker().intake(store)
        self.assertEqual(report.store_sha256, store.content_sha256())
        self.assertEqual(report.store_path, store.path)


# =============================================================================
# 8. The ledger is durable, ordered and refuses an incoherent history
# =============================================================================

class LedgerDurabilityTest(_TempCase):

    def test_state_survives_a_restart(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.note_attempt("akh-g15-fusion", proposal_id="akp-1",
                             disposition="build_failed", bears_on_falsifier=False,
                             note="unrelated")
        revived = self.tracker()
        tracked = revived.get("akh-g15-fusion")
        self.assertTrue(tracked.is_open)
        self.assertEqual(len(tracked.attempts), 1)

    def test_torn_tail_is_discarded_and_reported_not_swallowed(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        with open(tracker.ledger.path, "ab") as handle:
            handle.write(b'{"seq": 2, "kind": "HYPOTHESIS_RESOL')
        read = tracker.read()
        self.assertEqual(len(read.events), 1)
        self.assertGreater(read.discarded_tail_bytes, 0)
        self.assertTrue(tracker.get("akh-g15-fusion").is_open)

    def test_blank_and_unparseable_lines_are_corruption(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        with open(tracker.ledger.path, "ab") as handle:
            handle.write(b"   \n")
        with self.assertRaises(H.HypothesisLedgerCorruption):
            tracker.read()

    def test_non_increasing_seq_is_corruption(self):
        ledger = H.HypothesisLedger(os.path.join(self.root, "l.jsonl"))
        ledger.initialize()
        for seq in (2, 1):
            ledger.append(H.LedgerEvent(
                seq=seq, kind=H.EVENT_OPENED, hypothesis_id="akh-a",
                at="2026-08-03T00:00:00Z",
                payload={"hypothesis": _hypothesis("akh-a").to_dict()},
            ))
        with self.assertRaises(H.HypothesisLedgerCorruption):
            ledger.read()

    def test_dangling_attempt_is_corruption(self):
        events = (H.LedgerEvent(
            seq=1, kind=H.EVENT_ATTEMPTED, hypothesis_id="akh-ghost",
            at="2026-08-03T00:00:00Z",
            payload={"attempt": H.Attempt(
                hypothesis_id="akh-ghost", proposal_id="akp-1", disposition="banked",
                bears_on_falsifier=True, note="n",
            ).to_dict()},
        ),)
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            H.fold_ledger(events)
        self.assertIn("never opened", str(ctx.exception))

    def test_double_open_is_corruption(self):
        opened = H.LedgerEvent(
            seq=1, kind=H.EVENT_OPENED, hypothesis_id="akh-a",
            at="2026-08-03T00:00:00Z",
            payload={"hypothesis": _hypothesis("akh-a").to_dict()},
        )
        again = dataclasses.replace(opened, seq=2)
        with self.assertRaises(H.HypothesisLedgerCorruption):
            H.fold_ledger((opened, again))

    def test_envelope_and_payload_must_name_the_same_hypothesis(self):
        event = H.LedgerEvent(
            seq=1, kind=H.EVENT_OPENED, hypothesis_id="akh-a",
            at="2026-08-03T00:00:00Z",
            payload={"hypothesis": _hypothesis("akh-b").to_dict()},
        )
        with self.assertRaises(H.HypothesisLedgerCorruption):
            H.fold_ledger((event,))

    def test_unknown_ledger_kind_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            H.LedgerEvent(seq=1, kind="HYPOTHESIS_WITHDRAWN", hypothesis_id="akh-a",
                          at="2026-08-03T00:00:00Z")

    def test_a_recorder_that_raises_leaves_the_ledger_and_the_state_untouched(self):
        class Refusing:
            def record(self, event):
                raise RuntimeError("disk went away")

        tracker = self.tracker(recorder=Refusing())
        with self.assertRaises(RuntimeError):
            tracker.open_hypothesis(_hypothesis())
        self.assertEqual(tracker.read().events, ())
        self.assertEqual(tracker.still_open(), ())

    def test_a_recorder_returning_a_different_event_is_refused(self):
        class Substituting:
            def record(self, event):
                return dataclasses.replace(event, seq=event.seq + 7)

        tracker = self.tracker(recorder=Substituting())
        with self.assertRaises(H.HypothesisError) as ctx:
            tracker.open_hypothesis(_hypothesis())
        self.assertIn("not the one it was asked to record", str(ctx.exception))

    def test_this_module_writes_nothing_into_the_journal(self):
        # The substrate note in the module docstring, asserted: `journal.KINDS` has no
        # hypothesis kind, so events land in the ledger and the journal stays a
        # journal. When a kind is added, this test is the one that must change.
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.resolve("akh-g15-fusion", _evidence())
        self.assertEqual(self.journal.read_all(), [])
        self.assertEqual(
            {k for k in J.KINDS if "HYPOTHES" in k.upper()}, set()
        )

    def test_the_ledger_orders_under_the_journal_write_lock(self):
        # The recorder takes the JOURNAL lock, so hypothesis events and journal appends
        # share one order. Asserted by observing that a journal append inside the
        # recorder's window is possible at all (the lock is re-entrant per instance).
        recorded = []

        class Observing:
            def __init__(self, inner):
                self._inner = inner

            def record(self, event):
                recorded.append(event.kind)
                return self._inner.record(event)

        ledger = H.HypothesisLedger(os.path.join(self.root, H.LEDGER_FILENAME))
        ledger.initialize()
        tracker = self.tracker(
            recorder=Observing(H.JournalOrderedRecorder(self.journal, ledger))
        )
        tracker.open_hypothesis(_hypothesis())
        self.assertEqual(recorded, [H.EVENT_OPENED])
        self.assertEqual(len(tracker.read().events), 1)


# =============================================================================
# 9. Do-not-repeat: authorship is not new evidence (§8.4, §19.2, §19.3)
# =============================================================================

class DoNotRepeatTest(unittest.TestCase):

    def test_the_check_cannot_see_who_stated_the_hypothesis(self):
        params = set(inspect.signature(H.check_do_not_repeat).parameters)
        self.assertEqual(params, {"regime", "matches"})
        self.assertNotIn("origin", params)
        self.assertNotIn("author", params)

    def test_ledger_not_consulted_is_could_not_check_never_pass(self):
        check = H.check_do_not_repeat(regime=G15_REGIME, matches=None)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(check.passed)

    def test_consulted_and_clean_is_pass(self):
        self.assertEqual(
            H.check_do_not_repeat(regime=G15_REGIME, matches=()).outcome, S.PASS
        )

    def test_receipted_hard_constraint_rejects(self):
        match = H.LedgerMatch(
            entry_id="mfma-decode-kernels-are-worth-zero",
            entry_class=H.MATCH_CLASS_HARD_CONSTRAINT,
            match_dimensions=dict(G15_REGIME),
            receipt="artifact:sha256:abc",
        )
        check = H.check_do_not_repeat(regime=G15_REGIME, matches=(match,))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("mfma-decode-kernels-are-worth-zero", check.reasons[0])

    def test_receipted_matched_negative_rejects(self):
        match = H.LedgerMatch(
            entry_id="ngram-drafter", entry_class=H.MATCH_CLASS_MATCHED_NEGATIVE,
            match_dimensions=dict(G15_REGIME), receipt="commit:deadbeef:path:12",
        )
        self.assertEqual(
            H.check_do_not_repeat(regime=G15_REGIME, matches=(match,)).outcome, S.FAIL
        )

    def test_matched_negative_with_a_satisfied_reopen_predicate_does_not_reject(self):
        match = H.LedgerMatch(
            entry_id="ngram-drafter", entry_class=H.MATCH_CLASS_MATCHED_NEGATIVE,
            match_dimensions=dict(G15_REGIME), receipt="commit:deadbeef:path:12",
            reopen_predicate_satisfied=True,
        )
        check = H.check_do_not_repeat(regime=G15_REGIME, matches=(match,))
        self.assertEqual(check.outcome, S.PASS)
        self.assertTrue(any("reopen predicate" in r for r in check.reasons))

    def test_an_unreceipted_negative_neither_rejects_nor_clears(self):
        # §19.3: a wrong suppression is invisible because nothing ever tests it again.
        match = H.LedgerMatch(
            entry_id="confident-sentence", entry_class=H.MATCH_CLASS_MATCHED_NEGATIVE,
            match_dimensions=dict(G15_REGIME),
        )
        check = H.check_do_not_repeat(regime=G15_REGIME, matches=(match,))
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(any("NO receipt" in r for r in check.reasons))

    def test_a_conflicted_entry_is_never_authoritative(self):
        match = H.LedgerMatch(
            entry_id="contradicts-a-live-decision",
            entry_class=H.MATCH_CLASS_HARD_CONSTRAINT,
            match_dimensions=dict(G15_REGIME), receipt="artifact:sha256:abc",
            conflicted=True,
        )
        check = H.check_do_not_repeat(regime=G15_REGIME, matches=(match,))
        self.assertEqual(check.outcome, S.PASS)
        self.assertTrue(any("CONFLICTED" in r for r in check.reasons))

    def test_the_four_advisory_classes_do_not_close_the_question(self):
        for entry_class in sorted(H.MATCH_CLASSES - H.REJECTING_MATCH_CLASSES):
            with self.subTest(entry_class=entry_class):
                match = H.LedgerMatch(
                    entry_id="e", entry_class=entry_class,
                    match_dimensions=dict(G15_REGIME), receipt="r",
                )
                check = H.check_do_not_repeat(regime=G15_REGIME, matches=(match,))
                self.assertEqual(check.outcome, S.PASS)
                self.assertTrue(any("advisory" in r for r in check.reasons))

    def test_a_concrete_receipted_match_outranks_an_absent_regime(self):
        match = H.LedgerMatch(
            entry_id="hard", entry_class=H.MATCH_CLASS_HARD_CONSTRAINT, receipt="r",
        )
        self.assertEqual(
            H.check_do_not_repeat(regime={}, matches=(match,)).outcome, S.FAIL
        )

    def test_an_absent_regime_alone_is_could_not_check(self):
        check = H.check_do_not_repeat(regime={}, matches=())
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(any("no regime" in r for r in check.reasons))

    def test_the_verdict_is_identical_for_every_origin(self):
        # AK-D38 in its testable form: being the operator's idea is not new evidence.
        match = H.LedgerMatch(
            entry_id="cdna2-abandoned-by-vendor-and-quant-schools",
            entry_class=H.MATCH_CLASS_HARD_CONSTRAINT,
            match_dimensions=dict(G15_REGIME), receipt="handoff:19.2",
        )
        verdicts = set()
        for origin in sorted(H.ORIGINS):
            hypothesis = _hypothesis(f"akh-{origin}", origin=origin, author=origin)
            check = H.check_do_not_repeat(
                regime=hypothesis.regime, matches=(match,)
            )
            verdicts.add((check.outcome, check.reasons))
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(next(iter(verdicts))[0], S.FAIL)

    def test_bad_match_types_raise_rather_than_degrade(self):
        with self.assertRaises(TypeError):
            H.check_do_not_repeat(regime=G15_REGIME, matches=[{"entry_id": "e"}])
        with self.assertRaises(TypeError):
            H.check_do_not_repeat(regime=G15_REGIME, matches="HARD_CONSTRAINT")
        with self.assertRaises(ValueError):
            H.LedgerMatch(entry_id="e", entry_class="PROBABLY_BAD")


# =============================================================================
# 10. What it must not become (§8.4.0)
# =============================================================================

def _walk_keys(obj, path="$"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield f"{path}.{key}", key
            yield from _walk_keys(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _walk_keys(value, f"{path}[{index}]")


class NotAQueueJumpingMechanismTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.tracker_ = self.tracker()
        # Planner question first, operator question second, deliberately.
        self.tracker_.open_hypothesis(_hypothesis(
            "akh-planner-tile", origin=H.ORIGIN_PLANNER, author="planner",
            statement="tile 32 beats tile 16 at B=1",
            falsifier="no measurable delta at B=1 after five paired blocks",
        ))
        self.tracker_.open_hypothesis(_hypothesis())

    def test_the_operator_hypothesis_does_not_jump_the_queue(self):
        order = [t.hypothesis_id for t in self.tracker_.still_open()]
        self.assertEqual(order, ["akh-planner-tile", "akh-g15-fusion"])

    def test_no_ranking_field_exists_anywhere_in_the_rendered_block(self):
        block = self.tracker_.planner_round_block(round_id="r1")
        banned = ("priority", "rank", "weight", "boost", "score", "order")
        offenders = [
            path for path, key in _walk_keys(block)
            if any(term in key.lower() for term in banned)
        ]
        self.assertEqual(offenders, [])

    def test_no_authority_flavoured_key_exists_in_the_rendered_block(self):
        block = self.tracker_.planner_round_block(round_id="r1")
        self.assertEqual(S.find_authority_flavoured_keys(block), [])
        self.assertEqual(block["authority"], "proposal_source_not_authority")

    def test_the_store_refuses_an_authority_flavoured_key_in_a_regime(self):
        store = self.write_store(_store_doc(_g15_entry(
            regime={"backend": "llama_gpu", "auto_promote": True}
        )))
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("authority-flavoured", str(ctx.exception))

    def test_a_hypothesis_cannot_be_constructed_with_an_authority_key(self):
        with self.assertRaises(ValueError):
            H.Hypothesis(
                hypothesis_id="akh-x", statement="s", falsifier="f",
                origin=H.ORIGIN_OPERATOR, author="operator",
                regime={"freeze_authorized": True},
            )


# =============================================================================
# 11. The planner round block, and the absence of a cache
# =============================================================================

class RoundBlockAndNoCacheTest(_TempCase):

    def test_round_block_carries_grade_authority_and_every_falsifier(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        block = tracker.planner_round_block(round_id="round-7")
        self.assertEqual(block["schema"], H.ROUND_BLOCK_SCHEMA)
        self.assertEqual(block["round_id"], "round-7")
        self.assertEqual(block["campaign_id"], CAMPAIGN)
        self.assertEqual(block["entry_evidence_grade"], H.GRADE_DESIGN_PRIOR)
        self.assertEqual(block["authority"], "proposal_source_not_authority")
        entry = block["still_open"][0]
        self.assertEqual(entry["falsifier"], G15_FALSIFIER)
        self.assertEqual(entry["provenance"], {})
        self.assertEqual(entry["do_not_repeat"]["outcome"], S.COULD_NOT_CHECK)

    def test_round_block_carries_the_store_provenance_of_an_operator_hypothesis(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        entry = tracker.planner_round_block(round_id="r1")["still_open"][0]
        self.assertEqual(entry["provenance"]["kind"], "operator_store")
        self.assertEqual(entry["provenance"]["path"], store.path)
        self.assertEqual(
            entry["provenance"]["store_sha256"], store.content_sha256()
        )

    def test_a_hypothesis_absent_from_the_match_map_is_could_not_check(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.open_hypothesis(_hypothesis("akh-other", origin=H.ORIGIN_PLANNER,
                                            author="planner",
                                            statement="another question",
                                            falsifier="another predicted outcome"))
        block = tracker.planner_round_block(
            round_id="r1", matches_by_hypothesis={"akh-g15-fusion": ()}
        )
        by_id = {e["hypothesis_id"]: e for e in block["still_open"]}
        self.assertEqual(by_id["akh-g15-fusion"]["do_not_repeat"]["outcome"], S.PASS)
        self.assertEqual(by_id["akh-other"]["do_not_repeat"]["outcome"],
                         S.COULD_NOT_CHECK)

    def test_round_block_is_canonicalizable_and_hashable(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        block = tracker.planner_round_block(round_id="r1")
        self.assertIsInstance(S.canonical_json(block), str)
        self.assertEqual(len(S.content_hash(block)), 64)

    def test_round_block_requires_a_round_id(self):
        tracker = self.tracker()
        with self.assertRaises(ValueError):
            tracker.planner_round_block(round_id="  ")

    def test_resolved_entries_are_rendered_separately_from_open_ones(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.resolve("akh-g15-fusion", _evidence(H.RESOLUTION_REFUTED))
        block = tracker.planner_round_block(round_id="r1")
        self.assertEqual(block["open_count"], 0)
        self.assertEqual(block["still_open"], [])
        self.assertEqual(len(block["resolved"]), 1)
        self.assertEqual(block["resolved"][0]["status"], H.RESOLUTION_REFUTED)
        lean = tracker.planner_round_block(round_id="r1", include_resolved=False)
        self.assertNotIn("resolved", lean)

    def test_the_tracker_holds_no_folded_state(self):
        slots = set(H.HypothesisTracker.__slots__)
        self.assertEqual(
            slots & {"_state", "_tracked", "_open", "_cache", "_fold"}, set()
        )

    def test_an_external_append_is_visible_on_the_next_query(self):
        # Proof there is no cache: a second tracker over the same root writes, and the
        # first one sees it without being told.
        first = self.tracker()
        second = self.tracker()
        first.open_hypothesis(_hypothesis())
        self.assertEqual(
            [t.hypothesis_id for t in second.still_open()], ["akh-g15-fusion"]
        )
        second.resolve("akh-g15-fusion", _evidence())
        self.assertFalse(first.get("akh-g15-fusion").is_open)

    def test_get_on_an_unknown_id_raises(self):
        with self.assertRaises(H.UnknownHypothesis):
            self.tracker().get("akh-nope")

    def test_tracker_refuses_an_uninitialized_journal(self):
        root = os.path.join(self.base, "empty-journal")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "events_007.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("")
        with self.assertRaises(H.HypothesisError):
            H.HypothesisTracker(
                journal_=J.Journal(root), root=os.path.join(self.base, "c2")
            )


# =============================================================================
# 12. Adversarial regressions (red-team pass, 2026-08-03)
#
# Every test below reproduces a defect that was live in the shipped module and is
# named by the behaviour it must keep, not by the fix. Each one was observed FAILING
# against the pre-fix module.
# =============================================================================

class TornTailIsNotAppendedOntoTest(_TempCase):
    """A torn tail plus one more append used to destroy the whole ledger.

    `read()` discards an unterminated trailing fragment because the event it describes
    never took effect — but the next O_APPEND write landed straight after those bytes
    and fused them into a single unparseable line, which is NOT a torn tail (it ends in
    a newline). From that point on every read raised and every question in the ledger
    was unreachable: still-open tracking, resolutions, receipts, all of it, lost to one
    write that returned successfully.
    """

    def test_append_onto_a_torn_tail_is_refused_not_fused(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        with open(tracker.ledger.path, "ab") as handle:
            handle.write(b'{"seq": 2, "kind": "HYPOTHESIS_RESOL')
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            tracker.note_attempt("akh-g15-fusion", proposal_id="akp-1",
                                 disposition="build_failed", bears_on_falsifier=False,
                                 note="unrelated")
        self.assertIn("torn tail", str(ctx.exception))
        # …and the refusal costs nothing: the ledger is exactly as readable as before,
        # the torn bytes are still there to be inspected, and the question is open.
        read = tracker.read()
        self.assertEqual(len(read.events), 1)
        self.assertGreater(read.discarded_tail_bytes, 0)
        self.assertTrue(tracker.get("akh-g15-fusion").is_open)

    def test_the_ledger_reads_again_once_the_torn_tail_is_repaired(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        with open(tracker.ledger.path, "ab") as handle:
            handle.write(b'{"seq": 2, "kind": "HYPOTHESIS_RESOL')
        torn = tracker.read().discarded_tail_bytes
        size = os.path.getsize(tracker.ledger.path)
        with open(tracker.ledger.path, "r+b") as handle:
            handle.truncate(size - torn)
        tracker.note_attempt("akh-g15-fusion", proposal_id="akp-1",
                             disposition="build_failed", bears_on_falsifier=False,
                             note="unrelated")
        self.assertEqual(len(tracker.read().events), 2)
        self.assertTrue(tracker.get("akh-g15-fusion").is_open)


class LedgerReadsEveryLineOrRefusesTest(_TempCase):

    def test_a_truly_empty_line_is_corruption_not_a_silent_skip(self):
        # The blank-line refusal only ever fired for a WHITESPACE line: a genuinely
        # empty one was skipped by the same branch that skips the split artefact after
        # the final newline, so a ledger that had lost a record read back as intact
        # with discarded_tail_bytes == 0.
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        with open(tracker.ledger.path, "rb") as handle:
            data = handle.read()
        with open(tracker.ledger.path, "wb") as handle:
            handle.write(b"\n" + data)
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            tracker.read()
        self.assertIn("blank line", str(ctx.exception))

    def test_an_empty_line_between_records_is_corruption(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.note_attempt("akh-g15-fusion", proposal_id="akp-1",
                             disposition="banked", bears_on_falsifier=False, note="n")
        with open(tracker.ledger.path, "rb") as handle:
            first, second = handle.read().split(b"\n")[:2]
        with open(tracker.ledger.path, "wb") as handle:
            handle.write(first + b"\n\n" + second + b"\n")
        with self.assertRaises(H.HypothesisLedgerCorruption):
            tracker.read()

    def test_a_well_formed_ledger_still_reads(self):
        # The compliant path, so the refusal above cannot be satisfied by refusing
        # everything.
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        tracker.note_attempt("akh-g15-fusion", proposal_id="akp-1",
                             disposition="banked", bears_on_falsifier=True, note="n")
        self.assertEqual(len(tracker.read().events), 2)
        self.assertEqual(tracker.read().discarded_tail_bytes, 0)


class RefsReadBackFromARecordTest(unittest.TestCase):
    """`tuple("akj-1")` is five references that pass every check `_require_refs` makes."""

    def test_a_resolution_record_cannot_state_its_refs_as_a_string(self):
        record = {
            "outcome": H.RESOLUTION_CONFIRMED,
            "evidence_grade": H.GRADE_PROTOCOL_BOUND,
            "evidence_refs": "trustme",          # no spaces: every char is "non-empty"
            "falsifier_observed": "observed",
            "bears_on_falsifier": True,
            "resolved_by": "planner",
        }
        with self.assertRaises(H.ResolutionEvidenceMissing) as ctx:
            H.ResolutionEvidence.from_dict(record)
        self.assertIn("LIST of reference strings", str(ctx.exception))

    def test_a_mapping_of_refs_is_refused_rather_than_reduced_to_its_keys(self):
        record = {
            "outcome": H.RESOLUTION_CONFIRMED,
            "evidence_grade": H.GRADE_OBSERVATION,
            "evidence_refs": {"akj-1": "…"},
            "falsifier_observed": "observed",
            "bears_on_falsifier": True,
            "resolved_by": "controller",
        }
        with self.assertRaises(H.ResolutionEvidenceMissing):
            H.ResolutionEvidence.from_dict(record)

    def test_an_attempt_record_cannot_state_its_refs_as_a_string(self):
        record = {"hypothesis_id": "akh-a", "proposal_id": "akp-1",
                  "disposition": "build_failed", "bears_on_falsifier": False,
                  "note": "n", "refs": "akj-1"}
        with self.assertRaises(TypeError):
            H.Attempt.from_dict(record)

    def test_a_real_ref_list_still_round_trips(self):
        evidence = _evidence(refs=("akj-000000000042-abcdef123456", "akj-7"))
        self.assertEqual(
            H.ResolutionEvidence.from_dict(evidence.to_dict()), evidence
        )
        attempt = H.Attempt(hypothesis_id="akh-a", proposal_id="akp-1",
                            disposition="banked", bears_on_falsifier=True, note="n",
                            refs=("akj-1",))
        self.assertEqual(H.Attempt.from_dict(attempt.to_dict()), attempt)


class LedgerCorruptionIsTypedTest(_TempCase):
    """A corrupt ledger must not surface as a bare ValueError or as a STORE error."""

    def _write_line(self, tracker, obj) -> None:
        with open(tracker.ledger.path, "ab") as handle:
            handle.write((S.canonical_json(obj) + "\n").encode("utf-8"))

    def test_a_malformed_resolution_payload_is_ledger_corruption(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis())
        self._write_line(tracker, {
            "seq": 2, "kind": H.EVENT_RESOLVED, "hypothesis_id": "akh-g15-fusion",
            "at": "2026-08-03T00:00:00Z",
            "payload": {"resolution": {"outcome": "confirmed"}},
        })
        with self.assertRaises(H.HypothesisLedgerCorruption):
            tracker.still_open()

    def test_a_falsifier_less_hypothesis_in_the_ledger_is_ledger_corruption(self):
        # Not FalsifierMissing: that is a HypothesisStoreError, and it would send a
        # reader hunting through the operator's file for a defect that is in the ledger.
        tracker = self.tracker()
        self._write_line(tracker, {
            "seq": 1, "kind": H.EVENT_OPENED, "hypothesis_id": "akh-x",
            "at": "2026-08-03T00:00:00Z",
            "payload": {"hypothesis": {"hypothesis_id": "akh-x", "statement": "s",
                                       "falsifier": "   ", "origin": "planner",
                                       "author": "a"}},
        })
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            tracker.still_open()
        self.assertNotIsInstance(ctx.exception, H.HypothesisStoreError)

    def test_every_ledger_refusal_is_catchable_as_a_controller_error(self):
        tracker = self.tracker()
        self._write_line(tracker, {
            "seq": 1, "kind": H.EVENT_OPENED, "hypothesis_id": "akh-x",
            "at": "2026-08-03T00:00:00Z", "payload": {"hypothesis": {}},
        })
        with self.assertRaises(SM.ControllerError):
            tracker.state()

    def test_an_attempt_cannot_be_filed_under_another_question(self):
        # The OPENED branch checked envelope-vs-payload identity; the ATTEMPTED branch
        # did not, so a receipt for what was tried could be filed against a question it
        # was never about.
        events = (
            H.LedgerEvent(seq=1, kind=H.EVENT_OPENED, hypothesis_id="akh-a",
                          at="2026-08-03T00:00:00Z",
                          payload={"hypothesis": _hypothesis("akh-a").to_dict()}),
            H.LedgerEvent(seq=2, kind=H.EVENT_ATTEMPTED, hypothesis_id="akh-a",
                          at="2026-08-03T00:00:01Z",
                          payload={"attempt": H.Attempt(
                              hypothesis_id="akh-b", proposal_id="akp-1",
                              disposition="banked", bears_on_falsifier=True,
                              note="n").to_dict()}),
        )
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            H.fold_ledger(events)
        self.assertIn("akh-b", str(ctx.exception))

    def test_fold_refuses_a_non_event_before_it_sorts_them(self):
        # The documented TypeError was unreachable for anything without a `.seq`:
        # `sorted(..., key=lambda e: e.seq)` raised AttributeError first.
        with self.assertRaises(TypeError):
            H.fold_ledger([{"seq": 1, "kind": H.EVENT_OPENED}])


class RecorderCannotSubstituteTheRecordTest(_TempCase):
    """The recorder is a seam, not an authority.

    It is the documented swap point for the day `journal.py` grows hypothesis kinds —
    i.e. the one place an adapter sits between "what the controller decided" and "what
    the record says". Only `seq` and `kind` were compared, so an adapter could record a
    different OUTCOME on different EVIDENCE and the tracker would report it.
    """

    def _swapping_recorder(self, ledger):
        class Swapping:
            def record(self, event):
                if event.kind == H.EVENT_RESOLVED:
                    payload = dict(event.payload)
                    resolution = dict(payload["resolution"])
                    resolution["outcome"] = H.RESOLUTION_CONFIRMED
                    resolution["evidence_refs"] = ["fabricated"]
                    payload["resolution"] = resolution
                    event = dataclasses.replace(event, payload=payload)
                return ledger.append(event)

        return Swapping()

    def test_a_rewritten_payload_is_refused(self):
        root = os.path.join(self.base, "controller")
        os.makedirs(root, exist_ok=True)
        ledger = H.HypothesisLedger(os.path.join(root, H.LEDGER_FILENAME))
        ledger.initialize()
        tracker = self.tracker(root=root, recorder=self._swapping_recorder(ledger))
        tracker.open_hypothesis(_hypothesis())
        with self.assertRaises(H.HypothesisError) as ctx:
            tracker.resolve("akh-g15-fusion", _evidence(H.RESOLUTION_REFUTED))
        self.assertIn("rewritten", str(ctx.exception))

    def test_a_recorder_that_records_nothing_is_refused(self):
        class Silent:
            def record(self, event):
                return event

        tracker = self.tracker(recorder=Silent())
        with self.assertRaises(H.HypothesisError) as ctx:
            tracker.open_hypothesis(_hypothesis())
        self.assertIn("does not end with the event just recorded", str(ctx.exception))
        self.assertEqual(tracker.still_open(), ())

    def test_a_recorder_may_still_bind_extra_payload_keys(self):
        # The compliant path: `state_machine`'s recorder fills in a journal binding the
        # same way, so the check must forbid substitution WITHOUT forbidding its own
        # documented idiom.
        root = os.path.join(self.base, "controller")
        os.makedirs(root, exist_ok=True)
        ledger = H.HypothesisLedger(os.path.join(root, H.LEDGER_FILENAME))
        ledger.initialize()

        class Binding:
            def record(self, event):
                bound = dataclasses.replace(
                    event, payload=dict(event.payload) | {"journal_event_id": "akj-1"}
                )
                return ledger.append(bound)

        tracker = self.tracker(root=root, recorder=Binding())
        tracker.open_hypothesis(_hypothesis())
        self.assertEqual(
            tracker.read().events[0].payload["journal_event_id"], "akj-1"
        )
        self.assertTrue(tracker.get("akh-g15-fusion").is_open)


class IntakeAppliesAllOrNothingTest(_TempCase):
    """Intake used to open entries as it walked the file.

    A rewritten entry anywhere in the store therefore raised AFTER earlier entries had
    been committed, and the `IntakeReport` — the only account of what the intake did,
    including which operator lines have vanished — was destroyed by the exception that
    travelled past it.
    """

    def test_a_rewritten_entry_opens_nothing_at_all(self):
        tracker = self.tracker()
        tracker.open_hypothesis(_hypothesis("akh-tracked"))
        store = self.write_store(_store_doc(
            _g15_entry(hypothesis_id="akh-brand-new"),
            _g15_entry(hypothesis_id="akh-tracked", statement="a DIFFERENT question"),
        ))
        with self.assertRaises(H.QuestionRewritten):
            tracker.intake(store)
        self.assertEqual(sorted(tracker.state()), ["akh-tracked"])

    def test_a_clean_store_still_opens_every_entry(self):
        tracker = self.tracker()
        report = tracker.intake(self.write_store(_store_doc(
            _g15_entry(hypothesis_id="akh-one"),
            _g15_entry(hypothesis_id="akh-two", statement="a second question"),
        )))
        self.assertEqual(sorted(report.opened), ["akh-one", "akh-two"])
        self.assertEqual(sorted(tracker.state()), ["akh-one", "akh-two"])


class TheGradeAuditExercisesTheRefusalTest(_TempCase):
    """The audit read a TABLE where it claimed to prove a BEHAVIOUR.

    Point 4 was `"evidence_grade" in _REFUSED_ENTRY_KEYS`, so deleting the loop in
    `_load_entry` that consults that table — leaving the table itself untouched — left
    the audit reporting PASS while the store loaded `evidence_grade: protocol_bound`.
    """

    @staticmethod
    def _unenforcing_load_entry(self, entry, index, store_sha):
        stated = dict(entry)
        stated.pop("evidence_grade", None)
        return H.Hypothesis(
            hypothesis_id=stated["hypothesis_id"], statement=stated["statement"],
            falsifier=stated["falsifier"], origin=H.ORIGIN_OPERATOR,
            author=stated.get("author", "operator"),
            regime=dict(stated.get("regime") or {}),
            source={"kind": "operator_store", "entry_index": index},
        )

    def test_deleting_the_enforcement_fails_the_audit(self):
        original = H.OperatorHypothesisStore._load_entry
        H.OperatorHypothesisStore._load_entry = self._unenforcing_load_entry
        try:
            check = H.audit_no_origin_grade_promotion()
        finally:
            H.OperatorHypothesisStore._load_entry = original
        self.assertEqual(check.outcome, S.FAIL)
        self.assertTrue(any("evidence_grade" in reason for reason in check.reasons))
        self.assertEqual(H.audit_no_origin_grade_promotion().outcome, S.PASS)

    def test_a_loader_that_refuses_everything_cannot_pass_the_audit(self):
        # The control: refusing every entry, including every legitimate operator
        # hypothesis, must not read as the strongest possible enforcement.
        def refuse_everything(self, entry, index, store_sha):
            raise H.HypothesisStoreError("no")

        original = H.OperatorHypothesisStore._load_entry
        H.OperatorHypothesisStore._load_entry = refuse_everything
        try:
            check = H.audit_no_origin_grade_promotion()
        finally:
            H.OperatorHypothesisStore._load_entry = original
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_the_audit_probe_touches_no_filesystem(self):
        # It runs on every audit call, so it must not depend on a path existing.
        self.assertFalse(os.path.exists("<audit-probe>"))
        self.assertEqual(H.audit_no_origin_grade_promotion().outcome, S.PASS)


class UnreadableStoreIsATypedRefusalTest(_TempCase):

    def test_a_store_that_is_not_utf8_refuses_instead_of_raising_unicode_errors(self):
        # `surrogateescape` made this the one public store method whose failure was
        # neither a HypothesisStoreError nor a ControllerError.
        with open(self.store_path, "wb") as handle:
            handle.write(b'{"schema": "x", "hypotheses": []}\xff')
        store = H.OperatorHypothesisStore(self.store_path)
        with self.assertRaises(H.HypothesisStoreError):
            store.content_sha256()
        with self.assertRaises(SM.ControllerError):
            store.load()

    def test_a_utf8_store_still_digests_and_loads(self):
        store = self.write_store(_store_doc(_g15_entry(
            statement="the ≥15% claim for the elementwise cluster",
        )))
        loaded, digest = store.load_with_digest()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(digest, store.content_sha256())


# =============================================================================
# 20. THREE falsifier states, and collapsing any two of them is the defect
#
# (i)   absent      — legal on an operator entry, illegal to spend a claim on
# (ii)  placeholder — illegal ALWAYS, at entry and after
# (iii) stated      — legal to spend a claim on
#
# (i) and (ii) are refused in different places for different reasons, and the
# suite asserts they never become one state: (i) is an honest "nobody has written
# a predicate yet", (ii) is an empty string wearing a hat.
# =============================================================================

class ThreeFalsifierStatesTest(_TempCase):

    def test_the_three_states_are_three(self):
        self.assertEqual(
            H.FALSIFIER_STATES,
            {H.FALSIFIER_ABSENT, H.FALSIFIER_PLACEHOLDER, H.FALSIFIER_STATED},
        )
        self.assertNotEqual(H.FALSIFIER_ABSENT, H.FALSIFIER_PLACEHOLDER)
        self.assertEqual(
            H.FALSIFIER_STATES_REFUSING_COMPUTE,
            {H.FALSIFIER_ABSENT, H.FALSIFIER_PLACEHOLDER},
        )
        self.assertNotIn(H.FALSIFIER_STATED, H.FALSIFIER_STATES_REFUSING_COMPUTE)

    def test_classify_is_total_over_the_three(self):
        self.assertEqual(H.classify_falsifier(None), H.FALSIFIER_ABSENT)
        self.assertEqual(H.classify_falsifier(G15_FALSIFIER), H.FALSIFIER_STATED)
        for placeholder in sorted(H._PLACEHOLDER_FALSIFIERS):
            with self.subTest(placeholder=placeholder):
                self.assertEqual(
                    H.classify_falsifier(placeholder), H.FALSIFIER_PLACEHOLDER
                )
        for spelling in ("", "   ", "\t\n ", "TBD", "  N/A  ", "?"):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    H.classify_falsifier(spelling), H.FALSIFIER_PLACEHOLDER
                )

    def test_a_falsifier_that_changed_type_raises_rather_than_classifying(self):
        # The YAML hazard the store is JSON to avoid: `no` becoming False, `1.5-3`
        # becoming something else. A bool is not a third kind of falsifier.
        for value in (False, True, 0, 1.5, [], {}, ()):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    H.classify_falsifier(value)

    def test_absent_and_placeholder_are_refused_in_different_places(self):
        # (i) constructs for the operator …
        absent = H.Hypothesis(hypothesis_id="akh-a", statement=G15_STATEMENT,
                              falsifier=None, origin=H.ORIGIN_OPERATOR,
                              author="operator")
        self.assertEqual(absent.falsifier_state, H.FALSIFIER_ABSENT)
        # … (ii) never does, for the operator or anyone else.
        for origin in sorted(H.ORIGINS):
            with self.subTest(origin=origin):
                with self.assertRaises(H.FalsifierMissing) as ctx:
                    H.Hypothesis(hypothesis_id="akh-b", statement=G15_STATEMENT,
                                 falsifier="tbd", origin=origin, author="a")
                self.assertIn("placeholder", str(ctx.exception))
                # And the refusal says so, so a reader can tell the two apart.
                self.assertIn("may be ABSENT on an operator entry", str(ctx.exception))

    def test_optionality_is_a_function_of_origin_and_of_nothing_else(self):
        for origin in sorted(H.ORIGINS):
            with self.subTest(origin=origin):
                self.assertEqual(
                    H.falsifier_optional_on_entry(origin),
                    origin == H.ORIGIN_OPERATOR,
                )
        with self.assertRaises(ValueError):
            H.falsifier_optional_on_entry("archangel")

    def test_a_lower_barrier_is_not_a_higher_grade(self):
        """The operator's origin buys a lower barrier to ENTRY and nothing else.

        `entry_grade` does not branch on origin at all, so the two facts point in
        opposite directions on purpose — which is the whole safety property of §8.4.0.
        """
        for origin in sorted(H.ORIGINS):
            with self.subTest(origin=origin):
                self.assertEqual(H.entry_grade(origin), H.GRADE_DESIGN_PRIOR)
        absent = H.Hypothesis(hypothesis_id="akh-a", statement="s", falsifier=None,
                              origin=H.ORIGIN_OPERATOR, author="operator")
        self.assertEqual(absent.evidence_grade, H.GRADE_DESIGN_PRIOR)

    def test_the_falsifier_field_still_has_no_default(self):
        """`None` must be WRITTEN OUT. A defaulted falsifier is the AutoPilot shape."""
        field = {f.name: f for f in dataclasses.fields(H.Hypothesis)}["falsifier"]
        self.assertIs(field.default, dataclasses.MISSING)
        self.assertIs(field.default_factory, dataclasses.MISSING)

    def test_a_json_null_falsifier_is_absent_not_a_placeholder(self):
        store = self.write_store(_store_doc(_g15_entry(falsifier=None)))
        (loaded,) = store.load()
        self.assertEqual(loaded.falsifier_state, H.FALSIFIER_ABSENT)

    def test_the_store_still_refuses_a_placeholder_after_the_amendment(self):
        # The trap the amendment must not open: "optional" is a route to supplying
        # NOTHING, never a route to supplying "tbd".
        for placeholder in ("tbd", "TBD", "n/a", "?", "", "   ", "todo"):
            with self.subTest(placeholder=placeholder):
                store = self.write_store(_store_doc(_g15_entry(falsifier=placeholder)))
                with self.assertRaises(H.FalsifierMissing):
                    store.load()


# =============================================================================
# 21. The gate: a claim is refused on (i) and (ii) and granted on (iii)
# =============================================================================

class ClaimGateTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.tr = self.tracker()
        self.absent = H.Hypothesis(
            hypothesis_id="akh-operator-hunch", statement=G15_STATEMENT,
            falsifier=None, origin=H.ORIGIN_OPERATOR, author="operator",
            regime=dict(G15_REGIME),
        )

    def _authorized_kinds(self):
        return [e.kind for e in self.tr.read().events]

    # ---- (i) absent --------------------------------------------------------

    def test_a_claim_is_refused_on_an_absent_falsifier(self):
        self.tr.open_hypothesis(self.absent)
        with self.assertRaises(H.FalsifierRequiredBeforeCompute) as ctx:
            self.tr.authorize_claim("akh-operator-hunch", purpose="B=128 decode sweep",
                                    authorized_by="mainA")
        self.assertIn("'absent'", str(ctx.exception))
        self.assertIn("propose_falsifier", str(ctx.exception))
        # And nothing was recorded: a refused spend is not a spend.
        self.assertEqual(self._authorized_kinds(), [H.EVENT_OPENED])

    def test_a_token_cannot_be_constructed_for_an_absent_falsifier(self):
        with self.assertRaises(H.FalsifierRequiredBeforeCompute):
            H.ClaimAuthorization(
                hypothesis_id="akh-x", falsifier=None,
                falsifier_source=H.FALSIFIER_SOURCE_STATED,
                origin=H.ORIGIN_OPERATOR, purpose="p", authorized_by="a",
                authorized_at="2026-08-04T00:00:00.000000Z", ledger_seq=1,
            )

    # ---- (ii) placeholder --------------------------------------------------

    def test_a_token_cannot_be_constructed_for_a_placeholder_falsifier(self):
        for placeholder in ("tbd", "", "  ", "n/a", "?"):
            with self.subTest(placeholder=placeholder):
                with self.assertRaises(H.FalsifierRequiredBeforeCompute) as ctx:
                    H.ClaimAuthorization(
                        hypothesis_id="akh-x", falsifier=placeholder,
                        falsifier_source=H.FALSIFIER_SOURCE_STATED,
                        origin=H.ORIGIN_OPERATOR, purpose="p", authorized_by="a",
                        authorized_at="2026-08-04T00:00:00.000000Z", ledger_seq=1,
                    )
                self.assertIn("'placeholder'", str(ctx.exception))
                self.assertIn("wearing a hat", str(ctx.exception))

    def test_the_two_refusals_are_not_the_same_refusal(self):
        """(i) and (ii) must stay distinguishable at the gate, not just at entry."""
        def message(falsifier):
            try:
                H.ClaimAuthorization(
                    hypothesis_id="akh-x", falsifier=falsifier,
                    falsifier_source=H.FALSIFIER_SOURCE_STATED,
                    origin=H.ORIGIN_OPERATOR, purpose="p", authorized_by="a",
                    authorized_at="2026-08-04T00:00:00.000000Z", ledger_seq=1,
                )
            except H.FalsifierRequiredBeforeCompute as exc:
                return str(exc)
            raise AssertionError("expected a refusal")  # pragma: no cover

        absent, placeholder = message(None), message("tbd")
        self.assertNotEqual(absent, placeholder)
        self.assertIn("propose_falsifier", absent)
        self.assertNotIn("propose_falsifier", placeholder)

    # ---- (iii) stated: the COMPLIANT-PATH CONTROL --------------------------

    def test_a_claim_is_granted_on_a_stated_falsifier(self):
        self.tr.open_hypothesis(_hypothesis())
        token = self.tr.authorize_claim(
            "akh-g15-fusion", purpose="B=128 decode sweep", authorized_by="mainA",
        )
        self.assertIsInstance(token, H.ClaimAuthorization)
        self.assertEqual(token.falsifier, G15_FALSIFIER)
        self.assertEqual(token.falsifier_source, H.FALSIFIER_SOURCE_STATED)
        self.assertEqual(token.campaign_id, CAMPAIGN)
        self.assertEqual(token.evidence_grade, H.GRADE_DESIGN_PRIOR)
        self.assertEqual(self._authorized_kinds(),
                         [H.EVENT_OPENED, H.EVENT_CLAIM_AUTHORIZED])
        # The record's seq is the token's seq: an authorization with nothing behind it
        # is not an authorization.
        self.assertEqual(token.ledger_seq, self.tr.read().events[-1].seq)
        tracked = self.tr.get("akh-g15-fusion")
        self.assertEqual(tracked.claim_authorizations, (token,))
        self.assertTrue(tracked.may_spend_a_claim)

    def test_a_proposed_falsifier_opens_the_gate_the_operator_left_shut(self):
        self.tr.open_hypothesis(self.absent)
        self.assertFalse(self.tr.get("akh-operator-hunch").may_spend_a_claim)
        self.tr.propose_falsifier(
            "akh-operator-hunch",
            falsifier="a current wall-share map shows the cluster under 20%",
            proposed_by="mainA",
            rationale=("the operator's claim is about where the decode time IS, so a "
                       "wall-share map under the threshold refutes it directly"),
        )
        tracked = self.tr.get("akh-operator-hunch")
        self.assertEqual(tracked.falsifier_state, H.FALSIFIER_STATED)
        self.assertEqual(tracked.falsifier_source, H.FALSIFIER_SOURCE_PROPOSED)
        self.assertTrue(tracked.may_spend_a_claim)
        token = self.tr.authorize_claim("akh-operator-hunch", purpose="sweep",
                                        authorized_by="mainA")
        self.assertEqual(token.falsifier_source, H.FALSIFIER_SOURCE_PROPOSED)
        self.assertIn(token.falsifier, token.claim_purpose)

    def test_a_resolved_question_cannot_take_another_claim(self):
        self.tr.open_hypothesis(_hypothesis())
        self.tr.resolve("akh-g15-fusion", _evidence())
        with self.assertRaises(H.HypothesisNotOpen):
            self.tr.authorize_claim("akh-g15-fusion", purpose="p", authorized_by="a")

    def test_an_untracked_question_cannot_take_a_claim(self):
        with self.assertRaises(H.UnknownHypothesis):
            self.tr.authorize_claim("akh-never-opened", purpose="p", authorized_by="a")

    # ---- the door ----------------------------------------------------------

    def test_claim_for_hypothesis_refuses_everything_that_is_not_a_token(self):
        class Lookalike:
            hypothesis_id = "akh-x"
            falsifier = G15_FALSIFIER
            claim_purpose = "looks exactly like a token"

        for impostor in (None, "akh-x", 1, {"falsifier": G15_FALSIFIER}, Lookalike()):
            with self.subTest(impostor=type(impostor).__name__):
                with self.assertRaises(H.FalsifierRequiredBeforeCompute):
                    H.claim_for_hypothesis(impostor, lambda **kw: kw)

    def test_the_door_passes_the_falsifier_into_the_claims_own_receipt(self):
        self.tr.open_hypothesis(_hypothesis())
        token = self.tr.authorize_claim("akh-g15-fusion", purpose="B=128 decode sweep",
                                        authorized_by="mainA")
        seen = {}

        def fake_acquire(**kwargs):
            seen.update(kwargs)
            return "claim-handle"

        handle = H.claim_for_hypothesis(
            token, fake_acquire, device_id="mi210_0", campaign_id=CAMPAIGN,
        )
        self.assertEqual(handle, "claim-handle")
        self.assertEqual(seen["device_id"], "mi210_0")
        self.assertIn("akh-g15-fusion", seen["purpose"])
        self.assertIn(G15_FALSIFIER, seen["purpose"])
        self.assertIn("B=128 decode sweep", seen["purpose"])

    def test_the_caller_cannot_supply_its_own_purpose(self):
        self.tr.open_hypothesis(_hypothesis())
        token = self.tr.authorize_claim("akh-g15-fusion", purpose="p",
                                        authorized_by="a")
        with self.assertRaises(ValueError) as ctx:
            H.claim_for_hypothesis(token, lambda **kw: kw, purpose="something else")
        self.assertIn("taken from the authorization", str(ctx.exception))

    def test_the_door_refuses_a_non_callable_acquirer(self):
        self.tr.open_hypothesis(_hypothesis())
        token = self.tr.authorize_claim("akh-g15-fusion", purpose="p",
                                        authorized_by="a")
        with self.assertRaises(TypeError):
            H.claim_for_hypothesis(token, "not-callable")


# =============================================================================
# 22. Proposing a falsifier ADDS; it never rewrites
# =============================================================================

class FalsifierProposalTest(_TempCase):

    REAL = "a current wall-share map shows the cluster under 20%"
    WHY = "the statement is about where the time is; the map is the direct measurement"

    def setUp(self) -> None:
        super().setUp()
        self.tr = self.tracker()
        self.entry = _g15_entry()
        del self.entry["falsifier"]

    def test_a_question_that_already_has_one_refuses_a_proposal(self):
        self.tr.open_hypothesis(_hypothesis())
        with self.assertRaises(H.FalsifierAlreadyStated) as ctx:
            self.tr.propose_falsifier("akh-g15-fusion", falsifier=self.REAL,
                                      proposed_by="mainA", rationale=self.WHY)
        self.assertIn("new question and gets a new id", str(ctx.exception))

    def test_a_second_proposal_is_a_rewrite_by_another_name(self):
        store = self.write_store(_store_doc(self.entry))
        self.tr.intake(store)
        self.tr.propose_falsifier("akh-g15-fusion", falsifier=self.REAL,
                                  proposed_by="mainA", rationale=self.WHY)
        with self.assertRaises(H.FalsifierAlreadyStated):
            self.tr.propose_falsifier("akh-g15-fusion", falsifier="something else",
                                      proposed_by="mainB", rationale=self.WHY)

    def test_a_placeholder_proposal_is_refused(self):
        store = self.write_store(_store_doc(self.entry))
        self.tr.intake(store)
        for placeholder in ("tbd", "", "n/a", "?"):
            with self.subTest(placeholder=placeholder):
                with self.assertRaises(H.FalsifierMissing) as ctx:
                    self.tr.propose_falsifier("akh-g15-fusion",
                                              falsifier=placeholder,
                                              proposed_by="mainA", rationale=self.WHY)
                self.assertIn("never so an agent can satisfy", str(ctx.exception))

    def test_a_proposal_must_say_why(self):
        store = self.write_store(_store_doc(self.entry))
        self.tr.intake(store)
        with self.assertRaises(H.FalsifierMissing):
            self.tr.propose_falsifier("akh-g15-fusion", falsifier=self.REAL,
                                      proposed_by="mainA", rationale="  ")

    def test_a_proposal_that_restates_the_hypothesis_is_refused(self):
        store = self.write_store(_store_doc(self.entry))
        self.tr.intake(store)
        with self.assertRaises(H.FalsifierMissing) as ctx:
            self.tr.propose_falsifier("akh-g15-fusion", falsifier=G15_STATEMENT,
                                      proposed_by="mainA", rationale=self.WHY)
        self.assertIn("restates", str(ctx.exception))

    def test_a_multiline_proposal_is_refused(self):
        store = self.write_store(_store_doc(self.entry))
        self.tr.intake(store)
        with self.assertRaises(H.FalsifierMissing) as ctx:
            self.tr.propose_falsifier("akh-g15-fusion",
                                      falsifier="under 20%\nand the counters move",
                                      proposed_by="mainA", rationale=self.WHY)
        self.assertIn("ONE LINE", str(ctx.exception))

    def test_a_proposal_leaves_the_operators_own_record_untouched(self):
        """The structural reason a proposal is a SEPARATE record.

        Folding it into the hypothesis would move `fingerprint`, and the operator's
        next intake of their own UNEDITED file would then be told it states a different
        question. That is `QuestionRewritten` firing on a file nobody edited.
        """
        store = self.write_store(_store_doc(self.entry))
        self.tr.intake(store)
        before = self.tr.get("akh-g15-fusion").hypothesis.fingerprint
        self.tr.propose_falsifier("akh-g15-fusion", falsifier=self.REAL,
                                  proposed_by="mainA", rationale=self.WHY)
        tracked = self.tr.get("akh-g15-fusion")
        self.assertEqual(tracked.hypothesis.fingerprint, before)
        self.assertIsNone(tracked.hypothesis.falsifier)
        self.assertEqual(tracked.falsifier, self.REAL)
        # …and the operator's untouched file still intakes cleanly.
        report = self.tr.intake(store)
        self.assertEqual(report.already_tracked, ("akh-g15-fusion",))
        self.assertEqual(report.opened, ())

    def test_a_proposal_on_an_untracked_question_is_refused(self):
        with self.assertRaises(H.UnknownHypothesis):
            self.tr.propose_falsifier("akh-nope", falsifier=self.REAL,
                                      proposed_by="mainA", rationale=self.WHY)

    def test_intake_names_the_questions_awaiting_a_falsifier(self):
        store = self.write_store(_store_doc(self.entry, _g15_entry(
            hypothesis_id="akh-with-one",
        )))
        report = self.tr.intake(store)
        self.assertEqual(report.awaiting_falsifier, ("akh-g15-fusion",))
        block = self.tr.planner_round_block(round_id="r1")
        self.assertEqual(block["awaiting_falsifier"], ["akh-g15-fusion"])
        rendered = {e["hypothesis_id"]: e for e in block["still_open"]}
        self.assertEqual(rendered["akh-g15-fusion"]["falsifier_state"],
                         H.FALSIFIER_ABSENT)
        self.assertIsNone(rendered["akh-g15-fusion"]["falsifier"])
        self.assertFalse(rendered["akh-g15-fusion"]["may_spend_a_claim"])
        self.assertEqual(rendered["akh-with-one"]["falsifier_state"],
                         H.FALSIFIER_STATED)
        self.assertTrue(rendered["akh-with-one"]["may_spend_a_claim"])
        S.canonical_json(block)

    def test_the_fold_refuses_a_claim_recorded_against_no_falsifier(self):
        """The ledger cannot hold a history in which the gate did not hold."""
        store = self.write_store(_store_doc(self.entry))
        self.tr.intake(store)
        ledger_path = os.path.join(self.root, H.LEDGER_FILENAME)
        forged = {
            "seq": 2, "kind": H.EVENT_CLAIM_AUTHORIZED,
            "hypothesis_id": "akh-g15-fusion", "at": "2026-08-04T00:00:00Z",
            "payload": {"authorization": {
                "hypothesis_id": "akh-g15-fusion", "falsifier": G15_FALSIFIER,
                "falsifier_source": H.FALSIFIER_SOURCE_STATED,
                "origin": H.ORIGIN_OPERATOR, "purpose": "p", "authorized_by": "a",
                "authorized_at": "2026-08-04T00:00:00Z", "ledger_seq": 2,
            }},
        }
        with open(ledger_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(forged) + "\n")
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            self.tr.state()
        self.assertIn("mandatory before compute", str(ctx.exception))


# =============================================================================
# 23. The claim-gate audit, and its compliant-path control
# =============================================================================

class ClaimGateAuditTest(unittest.TestCase):

    def test_the_module_passes_its_own_audit(self):
        self.assertEqual(
            H.audit_falsifier_required_before_claim().outcome, S.PASS
        )

    def test_a_second_route_to_a_claim_fails_the_audit(self):
        doctored = (
            "def something_else(device):\n"
            "    return acquire_device_claim(device, purpose='whatever')\n"
        )
        check = H.audit_falsifier_required_before_claim(source=doctored)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("outside claim_for_hypothesis()", " ".join(check.reasons))

    def test_the_compliant_source_shape_still_passes(self):
        """The control: a call INSIDE the door must not be reported as a bypass."""
        compliant = (
            "def claim_for_hypothesis(authorization, acquire):\n"
            "    return acquire_device_claim(purpose=authorization.claim_purpose)\n"
        )
        self.assertEqual(
            H.audit_falsifier_required_before_claim(source=compliant).outcome, S.PASS
        )

    def test_the_audit_would_notice_a_token_type_that_refused_everything(self):
        # Vacuity control, exercised through the audit's own probe: if a REAL
        # falsifier stopped constructing, the two refusals above would prove nothing,
        # and the audit reports COULD_NOT_CHECK rather than PASS.
        self.assertIsNone(H._authorization_probe(G15_FALSIFIER))
        self.assertIsInstance(H._authorization_probe(None),
                              H.FalsifierRequiredBeforeCompute)
        self.assertIsInstance(H._authorization_probe("tbd"),
                              H.FalsifierRequiredBeforeCompute)


# =============================================================================
# 24. Adoption transfers ownership — journal FIRST, then remove
#
# The operator: "if the agents choose to pickup one of my hypotheses, it should
# be removed from OperatorHypothesisStore since it becomes owned by the agents."
#
# A move between two durable stores has exactly three failure modes. This suite
# asserts which one it fails toward:
#   LOST      — removed, never recorded. UNACCEPTABLE, and made impossible.
#   DUPLICATE — recorded, not removed. Detectable by id, repairable by id.
#   ORPHANED  — both, with nothing linking them. Prevented by inlining content.
# =============================================================================

#: An operator-written store with DELIBERATELY awkward formatting: entries in
#: three different layouts, keys in three different orders, ragged indentation,
#: extra spaces inside an object, and a `≥` escape that a re-serializer
#: would silently turn into a literal `≥`. Every one of those is a way to catch a
#: rewrite pretending to be a removal.
ODD_STORE_TEXT = '''{
  "schema": "epyc.autokernel.operator_hypotheses.v1",
  "hypotheses": [
    {
        "statement": "G15's elementwise/norm cluster is where the B=128 decode time is",
        "hypothesis_id": "akh-first",
      "falsifier": "a current wall-share map shows the cluster under 20%",
          "author":    "operator",
        "regime": {"backend": "llama_gpu",   "phase": "decode", "batch_band": "b128"}
    },
    {"hypothesis_id": "akh-middle", "statement": "the \\u2265 15% claim also holds at B=1", "falsifier": "no measurable delta at B=1 after five paired blocks"},
    {
      "hypothesis_id": "akh-last",
      "statement": "the dispatcher is the layer that explains the gap, not the kernel"
    }
  ]
}
'''


class _SimulatedCrash(RuntimeError):
    """A power cut between the adoption record and the store rewrite."""


class _CrashingStore(H.OperatorHypothesisStore):
    """A store whose rewrite never lands. Everything before it already has."""

    __slots__ = ()

    def remove_entry(self, hypothesis_id, *, expected_sha256=None):
        raise _SimulatedCrash("the machine went away between the record and the write")


class AdoptionTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.tr = self.tracker()
        self.store = self.write_store(ODD_STORE_TEXT)

    def _kinds(self):
        return [e.kind for e in self.tr.read().events]

    def _text(self) -> str:
        with open(self.store_path, "r", encoding="utf-8") as handle:
            return handle.read()

    # ---- the removal itself ------------------------------------------------

    def test_adoption_removes_exactly_one_entry(self):
        self.tr.intake(self.store)
        adoption = self.tr.adopt("akh-middle", self.store, adopted_by="mainA",
                                 reason="picking this up for the B=1 sweep")
        self.assertEqual(adoption.hypothesis_id, "akh-middle")
        self.assertEqual(
            [h.hypothesis_id for h in self.store.load()], ["akh-first", "akh-last"]
        )
        self.assertNotIn("akh-middle", self._text())

    def test_every_surviving_entry_comes_through_byte_for_byte(self):
        before_text, before_spans = self.store.entry_spans()
        kept = {s.hypothesis_id: s.text_of(before_text)
                for s in before_spans if s.hypothesis_id != "akh-middle"}
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")

        after_text, after_spans = self.store.entry_spans()
        self.assertEqual([s.hypothesis_id for s in after_spans],
                         ["akh-first", "akh-last"])
        for span in after_spans:
            with self.subTest(entry=span.hypothesis_id):
                self.assertEqual(span.text_of(after_text), kept[span.hypothesis_id])
                # and the operator's own characters are literally still there
                self.assertIn(kept[span.hypothesis_id], after_text)
        # The header the operator wrote, and their trailing newline, are untouched.
        self.assertTrue(after_text.startswith(
            '{\n  "schema": "epyc.autokernel.operator_hypotheses.v1",\n'
        ))
        self.assertTrue(after_text.endswith("}\n"))
        self.assertIn('"regime": {"backend": "llama_gpu",   "phase": "decode"',
                      after_text)

    def test_a_unicode_escape_is_not_re_encoded_by_a_removal(self):
        """`\\u2265` must still be `\\u2265`. A re-serializer would write `≥`.

        This is the difference between a removal and a rewrite-that-removes, and it is
        invisible in a `json.loads` comparison — which is why it is asserted on bytes.
        """
        self.tr.intake(self.store)
        self.tr.adopt("akh-last", self.store, adopted_by="mainA", reason="picked up")
        self.assertIn("\\u2265", self._text())
        self.assertNotIn("≥", self._text())

    def test_removing_the_first_middle_last_and_only_entry_all_work(self):
        for target, remaining in (("akh-first", ["akh-middle", "akh-last"]),
                                  ("akh-middle", ["akh-first", "akh-last"]),
                                  ("akh-last", ["akh-first", "akh-middle"])):
            with self.subTest(target=target):
                store = self.write_store(ODD_STORE_TEXT)
                tracker = self.tracker(root=os.path.join(self.base, "c-" + target))
                tracker.intake(store)
                tracker.adopt(target, store, adopted_by="mainA", reason="picked up")
                self.assertEqual([h.hypothesis_id for h in store.load()], remaining)
                # No trailing comma, no orphaned separator: it parses.
                json.loads(self._text())

    def test_removing_the_only_entry_leaves_a_valid_empty_store(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker(root=os.path.join(self.base, "solo"))
        tracker.intake(store)
        tracker.adopt("akh-g15-fusion", store, adopted_by="mainA", reason="picked up")
        self.assertEqual(store.load(), ())

    def test_the_store_is_replaced_atomically_and_leaves_no_temp_files(self):
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        leftovers = [n for n in os.listdir(self.base) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_a_short_write_refuses_instead_of_installing_a_truncated_store(self):
        """RED TEAM 2026-08-04. Atomicity is not integrity.

        `os.write` is `write(2)`: on a filesystem that fills mid-call it writes what it
        can and RETURNS the short count rather than raising. Unchecked, `_atomic_write`
        fsynced the truncated file, `os.replace`d it over the operator's store and
        `adopt()` RETURNED SUCCESS — a store that no longer parses, with every
        hypothesis after the cut permanently gone. `HypothesisLedger.append` had
        checked exactly this all along; the operator's own file had not.
        """
        real_write, real_atomic = os.write, H.OperatorHypothesisStore._atomic_write
        depth = []

        def short_write(fd, data):
            return real_write(fd, data[:len(data) // 2]) if depth else real_write(fd, data)

        def instrumented(inner_self, raw):
            depth.append(1)
            try:
                return real_atomic(inner_self, raw)
            finally:
                depth.pop()

        os.write = short_write
        H.OperatorHypothesisStore._atomic_write = instrumented
        self.addCleanup(setattr, os, "write", real_write)
        self.addCleanup(setattr, H.OperatorHypothesisStore, "_atomic_write", real_atomic)
        self.tr.intake(self.store)
        with self.assertRaises(H.StoreRewriteRefused) as ctx:
            self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="r")
        self.assertIn("short write", str(ctx.exception))
        os.write = real_write
        # The operator's file is EXACTLY as it was, and nothing is left behind.
        self.assertEqual(self._text(), ODD_STORE_TEXT)
        self.assertEqual(
            sorted(h.hypothesis_id for h in self.store.load()),
            sorted(h.hypothesis_id for h in H.OperatorHypothesisStore(
                self.store_path).load()),
        )
        self.assertEqual([n for n in os.listdir(self.base) if n.endswith(".tmp")], [])

    def test_control_a_whole_write_still_removes_the_entry(self):
        """The compliant-path control for the guard above: a normal write must still
        go through, or the fix would be a store that can never be rewritten."""
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="r")
        self.assertNotIn("akh-middle", [h.hypothesis_id for h in self.store.load()])

    def test_a_recoverable_short_write_completes_rather_than_refusing(self):
        """A short write is not by itself an error, and refusing one would be a fault.

        `write(2)` can return early on a signal with the file perfectly writable, so
        the guard above must not turn an interrupted write into a failed adoption. The
        rule is: retry from the short offset, and refuse only a write that cannot make
        progress. Without the retry loop this store is unwritable the first time a
        signal lands mid-`os.write`.
        """
        real_write, real_atomic = os.write, H.OperatorHypothesisStore._atomic_write
        depth, budget = [], [1]

        def interrupted_once(fd, data):
            if depth and budget[0]:
                budget[0] -= 1
                return real_write(fd, data[:1])
            return real_write(fd, data)

        def instrumented(inner_self, raw):
            depth.append(1)
            try:
                return real_atomic(inner_self, raw)
            finally:
                depth.pop()

        os.write = interrupted_once
        H.OperatorHypothesisStore._atomic_write = instrumented
        self.addCleanup(setattr, os, "write", real_write)
        self.addCleanup(setattr, H.OperatorHypothesisStore, "_atomic_write", real_atomic)
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="r")
        os.write = real_write
        self.assertEqual(budget[0], 0, "control: the short write must have happened")
        self.assertNotIn("akh-middle", [h.hypothesis_id for h in self.store.load()])
        self.assertEqual([n for n in os.listdir(self.base) if n.endswith(".tmp")], [])

    def test_the_store_is_never_written_in_place(self):
        """`os.replace`, proved from the INODE rather than from the final bytes.

        A truncate-then-write produces the same final content and a window in which
        the operator's file is empty or half a document. Holding a descriptor on the
        original inode across the removal is what tells the two apart: after a
        replace the old inode still has every original byte, because it was never the
        file that got written.
        """
        before = os.stat(self.store_path)
        original = open(self.store_path, "rb")
        self.addCleanup(original.close)
        self.store.remove_entry("akh-middle")
        self.assertEqual(original.read().decode("utf-8"), ODD_STORE_TEXT)
        self.assertNotEqual(os.stat(self.store_path).st_ino, before.st_ino)
        self.assertEqual(os.stat(self.store_path).st_mode, before.st_mode)

    def test_the_splice_is_verified_before_a_byte_is_written(self):
        """The verification must be ON the removal path, not merely present.

        Asserted by making the check fail from a subclass: if `remove_entry` still
        succeeds, the guard is a guard nothing consults — the shape this whole package
        keeps producing.
        """
        class _Unverifiable(H.OperatorHypothesisStore):
            __slots__ = ()

            def _verify_splice(self, new_text, spans, old_text, removed):
                raise H.StoreRewriteRefused("survivors would not come through unchanged")

        store = _Unverifiable(self.store_path)
        with self.assertRaises(H.StoreRewriteRefused):
            store.remove_entry("akh-middle")
        self.assertEqual(self._text(), ODD_STORE_TEXT)

    def test_a_store_that_changed_underneath_wins(self):
        text, spans = self.store.entry_spans()
        stale = "0" * 64
        with self.assertRaises(H.StoreRewriteRefused) as ctx:
            self.store.remove_entry("akh-middle", expected_sha256=stale)
        self.assertIn("the operator's edit wins", str(ctx.exception))
        self.assertEqual(self._text(), text)

    def test_removing_an_id_that_is_not_there_is_typed(self):
        with self.assertRaises(H.HypothesisNotInStore):
            self.store.remove_entry("akh-never-written")

    # ---- journal first -----------------------------------------------------

    def test_the_adoption_record_carries_the_content_inline(self):
        self.tr.intake(self.store)
        adoption = self.tr.adopt("akh-middle", self.store, adopted_by="mainA",
                                 reason="picked up for the B=1 sweep")
        event = [e for e in self.tr.read().events if e.kind == H.EVENT_ADOPTED][0]
        record = event.payload["adoption"]
        # ORPHANED is what this prevents: the record stands alone, with the operator's
        # own bytes and the full structured content, so it survives the file it names.
        self.assertEqual(record["hypothesis"]["statement"],
                         "the ≥ 15% claim also holds at B=1")
        self.assertIn("akh-middle", record["entry_text"])
        self.assertEqual(json.loads(record["entry_text"])["hypothesis_id"],
                         "akh-middle")
        self.assertEqual(record["store_path"], self.store.path)
        self.assertEqual(record["entry_index"], 1)
        self.assertEqual(adoption.owner, H.OWNER_AGENTS)

    def test_a_hypothesis_never_intaken_is_opened_before_the_file_is_touched(self):
        """LOST is structurally impossible: the content is durable first."""
        adoption = self.tr.adopt("akh-last", self.store, adopted_by="mainA",
                                 reason="picked up")
        self.assertEqual(self._kinds(), [H.EVENT_OPENED, H.EVENT_ADOPTED])
        self.assertNotIn("akh-last", self._text())
        # …and the question is answerable from the ledger alone.
        trace = self.tr.trace("akh-last")
        self.assertEqual(trace.adoption.adopted_by, "mainA")
        self.assertEqual(adoption.hypothesis["statement"],
                         "the dispatcher is the layer that explains the gap, "
                         "not the kernel")

    def test_a_crash_between_the_record_and_the_removal_leaves_a_duplicate(self):
        crashing = _CrashingStore(self.store_path)
        self.tr.intake(crashing)
        with self.assertRaises(_SimulatedCrash):
            self.tr.adopt("akh-middle", crashing, adopted_by="mainA", reason="picked up")
        # DUPLICATE, not LOST: recorded AND still in the file.
        self.assertIn(H.EVENT_ADOPTED, self._kinds())
        self.assertIn("akh-middle", [h.hypothesis_id for h in self.store.load()])
        # …and it is DETECTABLE by id.
        self.assertEqual(self.tr.adoption_duplicates(self.store), ("akh-middle",))
        report = self.tr.intake(self.store)
        self.assertEqual(report.adopted_but_still_in_store, ("akh-middle",))
        self.assertEqual(report.opened, ())

    def test_a_duplicate_is_repaired_without_a_second_record(self):
        crashing = _CrashingStore(self.store_path)
        self.tr.intake(crashing)
        with self.assertRaises(_SimulatedCrash):
            self.tr.adopt("akh-middle", crashing, adopted_by="mainA", reason="picked up")
        repaired = self.tr.reconcile_adoptions(self.store)
        self.assertEqual(repaired, ("akh-middle",))
        self.assertEqual(self.tr.adoption_duplicates(self.store), ())
        self.assertEqual(self._kinds().count(H.EVENT_ADOPTED), 1)
        self.assertEqual([h.hypothesis_id for h in self.store.load()],
                         ["akh-first", "akh-last"])
        # Idempotent: a second reconcile has nothing to do.
        self.assertEqual(self.tr.reconcile_adoptions(self.store), ())

    def test_the_same_adopter_retrying_completes_the_removal(self):
        crashing = _CrashingStore(self.store_path)
        self.tr.intake(crashing)
        with self.assertRaises(_SimulatedCrash):
            self.tr.adopt("akh-middle", crashing, adopted_by="mainA", reason="picked up")
        again = self.tr.adopt("akh-middle", self.store, adopted_by="mainA",
                              reason="picked up")
        self.assertEqual(again.adopted_by, "mainA")
        self.assertEqual(self._kinds().count(H.EVENT_ADOPTED), 1)
        self.assertNotIn("akh-middle", self._text())

    def test_a_different_adopter_is_refused_and_told_about_the_duplicate(self):
        crashing = _CrashingStore(self.store_path)
        self.tr.intake(crashing)
        with self.assertRaises(_SimulatedCrash):
            self.tr.adopt("akh-middle", crashing, adopted_by="mainA", reason="picked up")
        with self.assertRaises(H.HypothesisAlreadyAdopted) as ctx:
            self.tr.adopt("akh-middle", self.store, adopted_by="mainB", reason="mine")
        self.assertIn("mainA", str(ctx.exception))
        self.assertIn("DUPLICATE", str(ctx.exception))

    def test_adopting_twice_after_a_clean_adoption_is_refused(self):
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        with self.assertRaises(H.HypothesisAlreadyAdopted) as ctx:
            self.tr.adopt("akh-middle", self.store, adopted_by="mainB", reason="mine")
        self.assertIn("already out of the operator store", str(ctx.exception))
        self.assertEqual(self._kinds().count(H.EVENT_ADOPTED), 1)

    def test_adopting_an_id_the_store_does_not_carry_is_refused(self):
        with self.assertRaises(H.HypothesisNotInStore):
            self.tr.adopt("akh-not-there", self.store, adopted_by="mainA",
                          reason="picked up")
        self.assertEqual(self._kinds(), [])

    def test_the_fold_refuses_a_second_adoption_record(self):
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        event = [e for e in self.tr.read().events if e.kind == H.EVENT_ADOPTED][0]
        forged = event.to_dict()
        forged["seq"] = self.tr.read().events[-1].seq + 1
        with open(os.path.join(self.root, H.LEDGER_FILENAME), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(forged) + "\n")
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            self.tr.state()
        self.assertIn("ownership transfers once", str(ctx.exception))

    # ---- what adoption does and does NOT change ---------------------------

    def test_adoption_moves_ownership_and_never_relabels_provenance(self):
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        tracked = self.tr.get("akh-middle")
        # Origin is who HAD the idea, and it is frozen forever.
        self.assertEqual(tracked.hypothesis.origin, H.ORIGIN_OPERATOR)
        self.assertEqual(tracked.hypothesis.author, "operator")
        # Ownership is who is working it, and adoption is the one thing that moves it.
        self.assertEqual(tracked.owner, H.OWNER_AGENTS)
        self.assertTrue(tracked.is_agent_owned)
        # And adoption buys no evidence: still design_prior.
        self.assertEqual(tracked.evidence_grade, H.GRADE_DESIGN_PRIOR)

    def test_an_adopted_question_stays_open_and_keeps_its_history(self):
        self.tr.intake(self.store)
        self.tr.note_attempt("akh-middle", proposal_id="akp-1",
                             disposition="build_failed", bears_on_falsifier=False,
                             note="toolchain, unrelated to the question")
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        tracked = self.tr.get("akh-middle")
        self.assertTrue(tracked.is_open)
        self.assertEqual(len(tracked.attempts), 1)
        self.assertIn("akh-middle", [t.hypothesis_id for t in self.tr.still_open()])

    def test_an_adopted_question_is_not_reported_as_deleted_from_the_store(self):
        """Absent BY DESIGN is not the same as the operator deleting a line."""
        self.tr.intake(self.store)
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        report = self.tr.intake(self.store)
        self.assertNotIn("akh-middle", report.open_but_absent_from_store)
        self.assertEqual(report.opened, ())
        self.assertNotIn("akh-middle", report.already_tracked)
        # …and a genuine deletion IS still reported.
        remaining = json.loads(self._text())
        remaining["hypotheses"] = [e for e in remaining["hypotheses"]
                                   if e["hypothesis_id"] != "akh-first"]
        store = self.write_store(remaining)
        report = self.tr.intake(store)
        self.assertEqual(report.open_but_absent_from_store, ("akh-first",))

    def test_adopting_a_rewritten_entry_is_refused(self):
        self.tr.intake(self.store)
        doc = json.loads(self._text())
        for entry in doc["hypotheses"]:
            if entry["hypothesis_id"] == "akh-middle":
                entry["falsifier"] = "no delta at all, ever"
        store = self.write_store(doc)
        with self.assertRaises(H.QuestionRewritten):
            self.tr.adopt("akh-middle", store, adopted_by="mainA", reason="picked up")

    def test_adoption_refuses_a_store_that_is_not_a_store(self):
        with self.assertRaises(TypeError):
            self.tr.adopt("akh-middle", self.store_path, adopted_by="a", reason="r")
        for missing in ("adopted_by", "reason"):
            with self.subTest(missing=missing):
                kwargs = {"adopted_by": "mainA", "reason": "picked up"}
                kwargs[missing] = "   "
                with self.assertRaises(ValueError):
                    self.tr.adopt("akh-middle", self.store, **kwargs)


# =============================================================================
# 25. Two agents must not both adopt one hypothesis
# =============================================================================

class AdoptionConcurrencyTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.store = self.write_store(ODD_STORE_TEXT)
        self.tracker().intake(self.store)

    def _agent(self) -> H.HypothesisTracker:
        """A SECOND process's view: its own Journal object, so its own flock fd."""
        journal_ = J.Journal(self.journal_root, campaign_id=CAMPAIGN)
        return H.HypothesisTracker(journal_=journal_, root=self.root,
                                   campaign_id=CAMPAIGN)

    def test_exactly_one_of_two_concurrent_adopters_wins(self):
        barrier = threading.Barrier(2)
        results: dict = {}

        def attempt(name):
            tracker = self._agent()
            store = H.OperatorHypothesisStore(self.store_path)
            barrier.wait(timeout=30)
            try:
                results[name] = tracker.adopt(
                    "akh-middle", store, adopted_by=name, reason="picked up",
                    lock_timeout_s=30.0,
                )
            except Exception as exc:  # noqa: BLE001 - the loser's refusal is the result
                results[name] = exc

        threads = [threading.Thread(target=attempt, args=(name,))
                   for name in ("mainA", "mainB")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            self.assertFalse(thread.is_alive(), "an adopter never returned")

        winners = [n for n, r in results.items() if isinstance(r, H.Adoption)]
        losers = [r for r in results.values() if isinstance(r, Exception)]
        self.assertEqual(len(winners), 1, f"expected one winner, got {results}")
        self.assertEqual(len(losers), 1)
        self.assertIsInstance(losers[0], H.HypothesisAlreadyAdopted)

        tracker = self._agent()
        events = [e for e in tracker.read().events if e.kind == H.EVENT_ADOPTED]
        self.assertEqual(len(events), 1)
        self.assertEqual(tracker.get("akh-middle").adoption.adopted_by, winners[0])
        self.assertNotIn(
            "akh-middle", [h.hypothesis_id for h in self.store.load()]
        )

    def test_a_held_lock_times_out_rather_than_racing(self):
        fd = os.open(self.store.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.addCleanup(fcntl.flock, fd, fcntl.LOCK_UN)
        tracker = self._agent()
        with self.assertRaises(H.AdoptionLockUnavailable) as ctx:
            tracker.adopt("akh-middle", self.store, adopted_by="mainB",
                          reason="picked up", lock_timeout_s=0)
        self.assertIn("adoption lock", str(ctx.exception))
        # Nothing was recorded and nothing was removed.
        self.assertEqual(
            [e.kind for e in tracker.read().events].count(H.EVENT_ADOPTED), 0
        )
        self.assertIn("akh-middle", [h.hypothesis_id for h in self.store.load()])

    def test_a_free_lock_naming_a_live_holder_is_refused_not_taken(self):
        """`device_claim`'s rule, not a second one: a live holder is never stolen from.

        pid 1 is used because it is the one process guaranteed to be running and NOT
        this one; only its `/proc/1/stat` start-tick field is read.
        """
        stat = _dc._read_proc_stat(1)
        self.assertIsNotNone(stat)
        planted = {
            "store_path": self.store.path,
            "holder": {"pid": 1, "start_ticks": stat[1],
                       "boot_id": _dc._read_boot_id(),
                       "host": socket.gethostname(), "label": "planted"},
            "acquired_at": "2026-08-04T00:00:00.000000Z",
        }
        with open(self.store.lock_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(planted))
        with self.assertRaises(H.AdoptionLockInconsistent) as ctx:
            self.tracker().adopt("akh-middle", self.store, adopted_by="mainA",
                                 reason="picked up")
        self.assertIn("live", str(ctx.exception))
        self.assertIn("akh-middle", [h.hypothesis_id for h in self.store.load()])

    def test_a_free_lock_left_by_a_dead_holder_is_taken(self):
        """The compliant-path control: a leftover payload must not lock us out."""
        planted = {
            "store_path": self.store.path,
            "holder": {"pid": 424242, "start_ticks": 999,
                       "boot_id": "00000000-0000-0000-0000-000000000000",
                       "host": socket.gethostname(), "label": "crashed"},
            "acquired_at": "2026-08-04T00:00:00.000000Z",
        }
        with open(self.store.lock_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(planted))
        adoption = self.tracker().adopt("akh-middle", self.store, adopted_by="mainA",
                                        reason="picked up")
        self.assertEqual(adoption.adopted_by, "mainA")

    def test_an_unreadable_lock_payload_is_unknown_and_never_a_soft_dead(self):
        with open(self.store.lock_path, "w", encoding="utf-8") as handle:
            handle.write("not json at all")
        with self.assertRaises(H.AdoptionLockInconsistent):
            self.tracker().adopt("akh-middle", self.store, adopted_by="mainA",
                                 reason="picked up")


# =============================================================================
# 26. "What happened to the hypothesis I wrote?" — the read-back that makes
#     removal something other than deletion
# =============================================================================

class OperatorTraceabilityTest(_TempCase):

    def setUp(self) -> None:
        super().setUp()
        self.tr = self.tracker()
        self.store = self.write_store(ODD_STORE_TEXT)
        self.tr.intake(self.store)

    def test_the_trace_answers_for_a_hypothesis_that_left_the_store(self):
        self.tr.adopt("akh-last", self.store, adopted_by="mainA",
                      reason="taking this into the dispatcher campaign")
        self.tr.propose_falsifier(
            "akh-last",
            falsifier="a dispatcher-only change moves decode by under 2%",
            proposed_by="mainA",
            rationale="the claim is that the LAYER is the dispatcher; this isolates it",
        )
        self.tr.authorize_claim("akh-last", purpose="dispatcher A/B",
                                authorized_by="mainA")
        self.tr.note_attempt("akh-last", proposal_id="akp-7", disposition="evaluated",
                             bears_on_falsifier=True, note="paired blocks, five reps",
                             refs=("akj-000000000009-abcdef123456",))
        self.tr.resolve("akh-last", _evidence(
            outcome=H.RESOLUTION_REFUTED,
            observed="dispatcher-only change moved decode by 0.4%, under the 2% line",
        ))

        trace = self.tr.trace("akh-last", self.store)
        self.assertEqual(trace.hypothesis_id, "akh-last")
        self.assertEqual(trace.adoption.adopted_by, "mainA")
        self.assertFalse(trace.in_store)
        self.assertEqual(trace.tracked.status, H.RESOLUTION_REFUTED)
        self.assertEqual(len(trace.tracked.attempts), 1)
        self.assertEqual(len(trace.tracked.claim_authorizations), 1)

        answer = trace.answer
        for fragment in ("akh-last", "adopted by mainA", "dispatcher campaign",
                         self.store.path, "refuted", "no longer in the operator store"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, answer)

        payload = trace.to_dict()
        self.assertEqual(payload["owner"], H.OWNER_AGENTS)
        self.assertEqual(payload["origin"], H.ORIGIN_OPERATOR)
        self.assertEqual(payload["falsifier_source"], H.FALSIFIER_SOURCE_PROPOSED)
        self.assertEqual(payload["entry_evidence_grade"], H.GRADE_DESIGN_PRIOR)
        S.canonical_json(payload)

    def test_the_trace_does_not_need_the_store_to_answer(self):
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        trace = self.tr.trace("akh-middle")
        self.assertIsNone(trace.in_store)
        self.assertIn("adopted by mainA", trace.answer)
        self.assertIn("no falsifier yet", self.tr.trace("akh-last").answer)

    def test_the_trace_of_an_unadopted_question_says_who_owns_it(self):
        trace = self.tr.trace("akh-first", self.store)
        self.assertTrue(trace.in_store)
        self.assertIsNone(trace.adoption)
        self.assertIn("still owned by operator_store", trace.answer)

    def test_an_unknown_id_is_a_typed_refusal_not_a_blank_answer(self):
        with self.assertRaises(H.UnknownHypothesis):
            self.tr.trace("akh-never")

    def test_the_round_block_names_the_new_owner(self):
        self.tr.adopt("akh-middle", self.store, adopted_by="mainA", reason="picked up")
        block = self.tr.planner_round_block(round_id="r1")
        rendered = {e["hypothesis_id"]: e for e in block["still_open"]}
        self.assertEqual(rendered["akh-middle"]["owner"], H.OWNER_AGENTS)
        self.assertTrue(rendered["akh-middle"]["adopted"])
        self.assertEqual(rendered["akh-middle"]["origin"], H.ORIGIN_OPERATOR)
        self.assertEqual(rendered["akh-first"]["owner"], H.OWNER_OPERATOR)
        self.assertFalse(rendered["akh-first"]["adopted"])
        S.canonical_json(block)


# =============================================================================
# 27. The store scanner: a splice that cannot read the file the same way twice
#     must refuse rather than approximate
# =============================================================================

class StoreSpanScannerTest(_TempCase):

    def test_spans_are_verified_against_the_parse(self):
        store = self.write_store(ODD_STORE_TEXT)
        text, spans = store.entry_spans()
        self.assertEqual([s.hypothesis_id for s in spans],
                         ["akh-first", "akh-middle", "akh-last"])
        parsed = json.loads(text)["hypotheses"]
        for span, entry in zip(spans, parsed):
            with self.subTest(entry=span.hypothesis_id):
                self.assertEqual(json.loads(span.text_of(text)), entry)

    def test_the_hypotheses_key_is_found_structurally_not_by_search(self):
        """An operator statement containing the literal `"hypotheses":` must not
        capture the scanner — the statement is the one field guaranteed to contain
        whatever the operator likes."""
        doc = _store_doc(_g15_entry(
            statement='my note about "hypotheses": [ and other punctuation'
        ))
        store = self.write_store(doc)
        text, spans = store.entry_spans()
        self.assertEqual([s.hypothesis_id for s in spans], ["akh-g15-fusion"])
        self.assertEqual(json.loads(spans[0].text_of(text))["hypothesis_id"],
                         "akh-g15-fusion")

    def test_an_empty_store_scans_to_no_spans(self):
        store = self.write_store(_store_doc())
        _, spans = store.entry_spans()
        self.assertEqual(spans, ())

    def test_a_compact_store_splices_correctly(self):
        store = self.write_store(
            '{"schema":"epyc.autokernel.operator_hypotheses.v1","hypotheses":'
            '[{"hypothesis_id":"akh-a","statement":"a","falsifier":"fa"},'
            '{"hypothesis_id":"akh-b","statement":"b","falsifier":"fb"}]}'
        )
        removal = store.remove_entry("akh-a")
        self.assertEqual(removal.remaining_ids, ("akh-b",))
        with open(self.store_path, "r", encoding="utf-8") as handle:
            after = handle.read()
        self.assertEqual(
            after,
            '{"schema":"epyc.autokernel.operator_hypotheses.v1","hypotheses":'
            '[{"hypothesis_id":"akh-b","statement":"b","falsifier":"fb"}]}'
        )

    def test_a_removal_that_would_rewrite_a_survivor_refuses(self):
        """The guarantee, verified from the OUTSIDE of the splice.

        `_verify_splice` is what stands between "removal" and "rewrite that happens to
        remove"; handing it a spliced text that changed a survivor must refuse.
        """
        store = self.write_store(ODD_STORE_TEXT)
        text, spans = store.entry_spans()
        tampered = json.dumps(
            {"schema": H.STORE_SCHEMA,
             "hypotheses": [e for e in json.loads(text)["hypotheses"]
                            if e["hypothesis_id"] != "akh-middle"]},
            indent=2,
        )
        with self.assertRaises(H.StoreRewriteRefused) as ctx:
            store._verify_splice(tampered, spans, text, spans[1])
        self.assertIn("would be REWRITTEN", str(ctx.exception))

    def test_the_control_a_real_splice_verifies(self):
        store = self.write_store(ODD_STORE_TEXT)
        text, spans = store.entry_spans()
        spliced = text[:spans[1].start] + text[spans[2].start:]
        self.assertEqual(
            store._verify_splice(spliced, spans, text, spans[1]),
            ("akh-first", "akh-last"),
        )


# =============================================================================
# 22. The 2026-08-04 red team. Seven defects, found by execution, each with the
#     mutation that makes its test bite and the compliant path that stops the test
#     from passing vacuously.
# =============================================================================

class UnreadableLedgerIsNotAnEmptyLedgerTest(_TempCase):
    """An absent or unreadable ledger read back as *"nothing has ever been tried"*.

    `read()` began `if not os.path.exists(self.path): return LedgerRead((), 0)` — the
    exact inversion of the rule the store side is built around ("an absent store is not
    an empty one"). A ledger deleted, renamed or unmounted under a live tracker made
    `state()` `{}`, and the next `intake()` re-opened every operator hypothesis as brand
    new, discarding its attempts, its adoption and its resolution while reporting a
    clean run: the loop can no longer tell "tried and failed" from "never tried", which
    is the whole reason this plane exists.
    """

    def _tracked(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        tracker.note_attempt(
            "akh-g15-fusion", proposal_id="akp-1", disposition="applied",
            bears_on_falsifier=True, note="fused the cluster",
        )
        return store, tracker

    def test_a_vanished_ledger_refuses_rather_than_reading_as_empty(self):
        store, tracker = self._tracked()
        self.assertEqual(len(tracker.still_open()), 1)
        os.rename(tracker.ledger.path, tracker.ledger.path + ".moved")
        for name, call in (
            ("read", tracker.read),
            ("state", tracker.state),
            ("still_open", tracker.still_open),
            ("intake", lambda: tracker.intake(store)),
        ):
            with self.subTest(call=name):
                with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
                    call()
                self.assertIn("absent ledger is not an empty one", str(ctx.exception))

    def test_a_vanished_ledger_cannot_reopen_the_questions_it_forgot(self):
        """The consequence, asserted directly: the receipts must not be re-openable."""
        store, tracker = self._tracked()
        os.rename(tracker.ledger.path, tracker.ledger.path + ".moved")
        with self.assertRaises(H.HypothesisLedgerCorruption):
            tracker.intake(store)
        os.rename(tracker.ledger.path + ".moved", tracker.ledger.path)
        report = tracker.intake(store)
        self.assertEqual(report.opened, ())
        self.assertEqual(report.already_tracked, ("akh-g15-fusion",))
        self.assertEqual(len(tracker.get("akh-g15-fusion").attempts), 1)

    def test_an_unreadable_ledger_is_a_TYPED_refusal_not_a_bare_OSError(self):
        _, tracker = self._tracked()
        os.chmod(tracker.ledger.path, 0o000)
        self.addCleanup(os.chmod, tracker.ledger.path, 0o644)
        if os.access(tracker.ledger.path, os.R_OK):  # pragma: no cover - running as root
            self.skipTest("mode 000 is readable here; the refusal cannot be exercised")
        with self.assertRaises(SM.ControllerError):
            tracker.state()

    def test_the_control_an_initialized_but_empty_ledger_still_reads_as_empty(self):
        """A ledger somebody ESTABLISHED with nothing in it is a real statement, and
        the refusal above must not swallow it — otherwise every fresh campaign fails."""
        ledger = H.HypothesisLedger(os.path.join(self.base, "fresh", "l.jsonl"))
        ledger.initialize()
        self.assertEqual(ledger.read(), H.LedgerRead((), 0))
        self.assertEqual(self.tracker().state(), {})


class ClosureCostsAFalsifierTest(_TempCase):
    """The amendment gated COMPUTE on a falsifier and left CLOSURE ungated.

    Closing is the cheaper and more damaging move: an agent could mark the operator's
    one-line idea `confirmed` having written no predicate, spent no claim and run
    nothing — and the store entry is gone by then, so the operator's only view is a
    trace that says "confirmed". Every field of `ResolutionEvidence` is a claim ABOUT
    the falsifier, so a resolution in state `absent` describes an observation against a
    predicate that does not exist.
    """

    def _absent(self):
        store = self.write_store(_store_doc(_g15_entry(falsifier=None)))
        tracker = self.tracker()
        tracker.intake(store)
        self.assertEqual(tracker.get("akh-g15-fusion").falsifier_state,
                         H.FALSIFIER_ABSENT)
        return tracker

    def test_a_question_with_no_falsifier_cannot_be_resolved(self):
        tracker = self._absent()
        for outcome in sorted(H.RESOLUTIONS):
            with self.subTest(outcome=outcome):
                with self.assertRaises(H.ResolutionEvidenceMissing) as ctx:
                    tracker.resolve("akh-g15-fusion", _evidence(outcome))
                self.assertIn("nothing for this evidence", str(ctx.exception))

    def test_the_fold_refuses_a_hand_written_resolution_of_an_unfalsified_question(self):
        """The gate is re-derived from the history, not trusted from the record —
        the same treatment `HYPOTHESIS_CLAIM_AUTHORIZED` already gets."""
        hypothesis = _hypothesis(falsifier=None)
        events = (
            H.LedgerEvent(seq=1, kind=H.EVENT_OPENED,
                          hypothesis_id=hypothesis.hypothesis_id,
                          at="2026-08-04T00:00:00Z",
                          payload={"hypothesis": hypothesis.to_dict()}),
            H.LedgerEvent(seq=2, kind=H.EVENT_RESOLVED,
                          hypothesis_id=hypothesis.hypothesis_id,
                          at="2026-08-04T00:00:01Z",
                          payload={"resolution": _evidence().to_dict()}),
        )
        with self.assertRaises(H.HypothesisLedgerCorruption) as ctx:
            H.fold_ledger(events)
        self.assertIn("no evidence could have been observed", str(ctx.exception))

    def test_a_proposed_falsifier_is_what_makes_it_closable(self):
        """The compliant path: the amendment's own route must still reach a resolution,
        or this guard would just be the old mandatory-on-entry rule with a new message."""
        tracker = self._absent()
        tracker.propose_falsifier(
            "akh-g15-fusion", falsifier=G15_FALSIFIER,
            proposed_by="agent-a", rationale="the only cheap probe of the claim",
        )
        tracker.resolve("akh-g15-fusion", _evidence())
        tracked = tracker.get("akh-g15-fusion")
        self.assertEqual(tracked.status, H.RESOLUTION_REFUTED)
        self.assertEqual(tracked.falsifier_source, H.FALSIFIER_SOURCE_PROPOSED)

    def test_the_control_a_stated_falsifier_resolves_untouched(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        tracker.resolve("akh-g15-fusion", _evidence())
        self.assertEqual(tracker.get("akh-g15-fusion").status, H.RESOLUTION_REFUTED)


class ADesignPriorIsNotEvidenceTest(_TempCase):
    """`design_prior` closed questions. It is the grade every hypothesis ENTERS at.

    §19.1: it means "worth considering", not "probably true". Accepting it as the grade
    of the evidence that CLOSES a question let one prior resolve another —
    `confirmed ... (design_prior): looks right to me` — which is the promotion §8.4.0 /
    AK-D38 / §19.0 rule 4 forbid, reached through the resolution instead of through the
    hypothesis.
    """

    def test_a_prior_cannot_close_a_question(self):
        for outcome in sorted(H.RESOLUTIONS):
            with self.subTest(outcome=outcome):
                with self.assertRaises(H.ResolutionEvidenceMissing) as ctx:
                    H.ResolutionEvidence(
                        outcome=outcome, evidence_grade=H.GRADE_DESIGN_PRIOR,
                        evidence_refs=("a-note-i-wrote",),
                        falsifier_observed="looks right to me",
                        bears_on_falsifier=True, resolved_by="agent-a",
                    )
                self.assertIn("grade a hypothesis ENTERS at", str(ctx.exception))

    def test_it_is_refused_on_the_RECORD_so_resolve_cannot_be_routed_around(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        with self.assertRaises(H.ResolutionEvidenceMissing):
            tracker.resolve("akh-g15-fusion", _evidence(grade=H.GRADE_DESIGN_PRIOR))
        self.assertTrue(tracker.get("akh-g15-fusion").is_open)

    def test_the_control_every_other_grade_still_closes(self):
        for grade in sorted(H.EVIDENCE_GRADES - {H.GRADE_DESIGN_PRIOR}):
            with self.subTest(grade=grade):
                self.assertEqual(
                    H.ResolutionEvidence(
                        outcome=H.RESOLUTION_REFUTED, evidence_grade=grade,
                        evidence_refs=("akj-1",), falsifier_observed="no gain",
                        bears_on_falsifier=True, resolved_by="agent-a",
                    ).evidence_grade,
                    grade,
                )

    def test_the_hypothesis_still_ENTERS_at_design_prior(self):
        """The two uses of the same token are different facts, and only one moved."""
        self.assertEqual(H.ENTRY_GRADE, H.GRADE_DESIGN_PRIOR)
        self.assertEqual(_hypothesis().evidence_grade, H.GRADE_DESIGN_PRIOR)
        self.assertIn(H.GRADE_DESIGN_PRIOR, H.EVIDENCE_GRADES)


class AMatchAboutAnotherRegimeDoesNotRejectTest(unittest.TestCase):
    """`check_do_not_repeat` held the question's regime and the entry's own dimensions
    and compared them not at all — `regime` was read only to ask whether it was empty.

    A receipted `MATCHED_NEGATIVE` recorded at llama_cpu/prefill/b1 therefore rejected a
    llama_gpu/decode/b128 question, and 56 of the 56 contradicting cells of that grid
    rejected: a 100% false-reject rate on ideas nobody has tried. It is the §19.3 failure
    in its worst form, because a wrong suppression is invisible — nothing tests that
    family again, and the loop looks productive while being sterile.
    """

    CPU = {"backend": "llama_cpu", "phase": "prefill", "batch_band": "b1"}

    def _match(self, dimensions, **kw):
        kw.setdefault("receipt", "commit:deadbeef:path:12")
        return H.LedgerMatch(
            entry_id="dnr-elsewhere", entry_class=H.MATCH_CLASS_MATCHED_NEGATIVE,
            match_dimensions=dimensions, **kw)

    def test_a_contradicting_match_neither_rejects_nor_clears(self):
        match = self._match({"regime": {k: [v] for k, v in self.CPU.items()}})
        check = H.check_do_not_repeat(regime=G15_REGIME, matches=(match,))
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertTrue(any("CONTRADICT" in r for r in check.reasons))
        self.assertTrue(any("backend" in r for r in check.reasons))

    def test_no_contradicting_cell_of_the_grid_rejects(self):
        """The measured rate, kept as a test so it stays at zero."""
        values = {"backend": ("llama_gpu", "llama_cpu"),
                  "phase": ("decode", "prefill"),
                  "batch_band": ("b1", "b128")}
        rejected_when_different = 0
        same = different = 0
        for qb in values["backend"]:
            for qp in values["phase"]:
                for qd in values["batch_band"]:
                    for eb in values["backend"]:
                        for ep in values["phase"]:
                            for ed in values["batch_band"]:
                                question = {"backend": qb, "phase": qp,
                                            "batch_band": qd}
                                entry = {"backend": [eb], "phase": [ep],
                                         "batch_band": [ed]}
                                check = H.check_do_not_repeat(
                                    regime=question,
                                    matches=(self._match({"regime": entry}),))
                                if (qb, qp, qd) == (eb, ep, ed):
                                    same += 1
                                    self.assertEqual(check.outcome, S.FAIL)
                                else:
                                    different += 1
                                    rejected_when_different += (
                                        check.outcome == S.FAIL)
        self.assertEqual((same, different), (8, 56))
        self.assertEqual(rejected_when_different, 0)

    def test_the_flat_dimension_shape_is_compared_too(self):
        """Both shapes that actually occur: the map under a `regime` key (what the
        compiled §19.2 ledger emits) and the map itself (a caller passing it through)."""
        check = H.check_do_not_repeat(
            regime=G15_REGIME, matches=(self._match(dict(self.CPU)),))
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_an_undeclared_dimension_is_not_a_contradiction(self):
        """A dimension one side does not state is an incomplete comparison, not a
        disagreement — that call belongs to the producer's matching rule, and this
        consumer must not start second-guessing it."""
        partial = {"backend": ["llama_gpu"]}
        self.assertEqual(
            H.check_do_not_repeat(regime=G15_REGIME,
                                  matches=(self._match({"regime": partial}),)).outcome,
            S.FAIL,
        )

    def test_the_control_an_agreeing_match_still_rejects(self):
        agreeing = {k: [v] for k, v in G15_REGIME.items()}
        check = H.check_do_not_repeat(
            regime=G15_REGIME, matches=(self._match({"regime": agreeing}),))
        self.assertEqual(check.outcome, S.FAIL)

    def test_a_mis_keyed_planner_mapping_cannot_silently_close_a_question(self):
        """The likeliest wiring mistake, and the one whose damage nothing surfaces."""
        block_matches = {"akh-g15-fusion": (
            self._match({"regime": {k: [v] for k, v in self.CPU.items()}}),
        )}
        with tempfile.TemporaryDirectory() as base:
            journal_root = os.path.join(base, "journal")
            os.makedirs(journal_root)
            jrnl = J.Journal(journal_root, campaign_id=CAMPAIGN)
            jrnl.initialize()
            tracker = H.HypothesisTracker(
                journal_=jrnl, root=os.path.join(base, "c"), campaign_id=CAMPAIGN)
            tracker.open_hypothesis(_hypothesis())
            block = tracker.planner_round_block(
                round_id="akr-1", matches_by_hypothesis=block_matches)
        entry = block["still_open"][0]
        self.assertEqual(entry["do_not_repeat"]["outcome"], S.COULD_NOT_CHECK)


class AdoptionLockIdentityIncludesTheHostTest(_TempCase):
    """The lock's "that payload is MINE" shortcut compared pid, start_ticks and boot_id
    and NOT host — the one field that says "another machine" was the one field not
    consulted.

    Containers sharing a kernel share `/proc/sys/kernel/random/boot_id` while each has
    its own PID namespace, so a (pid, start_ticks, boot_id) collision between two
    sessions on one store is ordinary rather than exotic. The shortcut returns BEFORE
    `assess_holder_liveness`, so the effect was the worst available: a LIVE holder's
    lock taken silently, and two adopters running at once.
    """

    def _store_with_payload(self, holder):
        store = self.write_store(_store_doc(_g15_entry()))
        with open(store.lock_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "store_path": store.path, "holder": holder,
                "acquired_at": "2026-08-04T00:00:00.000000Z",
            }) + "\n")
        return store

    def test_a_payload_from_another_host_is_never_taken_for_this_process(self):
        alien = dict(_dc.current_holder_identity("autokernel.hypotheses.adopt"))
        alien["host"] = alien["host"] + "-some-other-box"
        store = self._store_with_payload(alien)
        with self.assertRaises(H.AdoptionLockInconsistent) as ctx:
            with store.adoption_lock(timeout_s=0.0):
                pass  # pragma: no cover - the lock must not be granted
        self.assertIn("unknown", str(ctx.exception))

    def test_the_adoption_itself_is_refused_not_just_the_lock(self):
        alien = dict(_dc.current_holder_identity("autokernel.hypotheses.adopt"))
        alien["host"] = alien["host"] + "-elsewhere"
        store = self._store_with_payload(alien)
        tracker = self.tracker()
        with self.assertRaises(H.AdoptionLockInconsistent):
            tracker.adopt("akh-g15-fusion", store, adopted_by="agent-a", reason="r")
        self.assertEqual([h.hypothesis_id for h in store.load()], ["akh-g15-fusion"])

    def test_the_control_this_process_still_recognises_its_own_payload(self):
        """Without the shortcut a process interrupted between taking the lock and
        releasing it would assess its own LIVE pid and lock itself out forever."""
        mine = dict(_dc.current_holder_identity("autokernel.hypotheses.adopt"))
        store = self._store_with_payload(mine)
        with store.adoption_lock(timeout_s=0.0) as holder:
            self.assertEqual(holder["pid"], os.getpid())
        tracker = self.tracker()
        tracker.adopt("akh-g15-fusion", store, adopted_by="agent-a", reason="r")
        self.assertEqual([h.hypothesis_id for h in store.load()], [])

    def test_the_control_a_dead_holder_still_does_not_block(self):
        dead = dict(_dc.current_holder_identity("autokernel.hypotheses.adopt"))
        dead["start_ticks"] = int(dead["start_ticks"]) + 10_000_000  # PID recycled
        store = self._store_with_payload(dead)
        self.assertEqual(_dc.assess_holder_liveness(dead).state, _dc.DEAD)
        self.tracker().adopt(
            "akh-g15-fusion", store, adopted_by="agent-a", reason="r")
        self.assertEqual([h.hypothesis_id for h in store.load()], [])


class AStoreThisModuleReadsTwoWaysIsRefusedTest(_TempCase):
    """`json.loads` keeps the LAST value for a duplicated key; the span scanner walks to
    the FIRST.

    A store with two top-level `"hypotheses"` arrays was therefore readable two ways:
    adoption recorded the transfer and spliced an entry out of the array nobody parses,
    leaving the operator's entry in the file permanently — a duplicate
    `reconcile_adoptions()` cannot clear. The same hook catches the sharper case inside
    an entry, where a repeated `falsifier` means the predicate you can see is not
    necessarily the one that took effect.
    """

    DUPLICATE_ARRAY = (
        '{"schema": "epyc.autokernel.operator_hypotheses.v1",\n'
        ' "hypotheses": [{"hypothesis_id": "akh-x", "statement": "s",'
        ' "falsifier": "the map shows it under 20%"}],\n'
        ' "hypotheses": [{"hypothesis_id": "akh-x", "statement": "s",'
        ' "falsifier": "the map shows it under 20%"}]}\n'
    )
    DUPLICATE_FALSIFIER = (
        '{"schema": "epyc.autokernel.operator_hypotheses.v1", "hypotheses": ['
        '{"hypothesis_id": "akh-x", "statement": "s",'
        ' "falsifier": "the map shows it under 20%", "falsifier": "tbd"}]}\n'
    )

    def test_a_duplicated_hypotheses_key_is_refused_at_load(self):
        store = self.write_store(self.DUPLICATE_ARRAY)
        for name, call in (("load", store.load),
                           ("entry_spans", store.entry_spans)):
            with self.subTest(call=name):
                with self.assertRaises(H.HypothesisStoreError) as ctx:
                    call()
                self.assertIn("duplicate key", str(ctx.exception))

    def test_adoption_cannot_record_a_transfer_it_cannot_perform(self):
        store = self.write_store(self.DUPLICATE_ARRAY)
        tracker = self.tracker()
        with self.assertRaises(H.HypothesisStoreError):
            tracker.adopt("akh-x", store, adopted_by="agent-a", reason="r")
        self.assertEqual(tracker.state(), {})
        with open(store.path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), self.DUPLICATE_ARRAY)

    def test_a_duplicated_falsifier_key_is_refused(self):
        store = self.write_store(self.DUPLICATE_FALSIFIER)
        with self.assertRaises(H.HypothesisStoreError) as ctx:
            store.load()
        self.assertIn("duplicate key", str(ctx.exception))

    def test_the_control_a_store_with_no_repeated_key_loads(self):
        store = self.write_store(_store_doc(_g15_entry()))
        self.assertEqual([h.hypothesis_id for h in store.load()], ["akh-g15-fusion"])
        _, spans = store.entry_spans()
        self.assertEqual([s.hypothesis_id for s in spans], ["akh-g15-fusion"])


class TheTokenIsReDerivedAtTheDoorTest(_TempCase):
    """`ClaimAuthorization`'s invariant was checked at construction only.

    `claim_for_hypothesis`'s docstring claimed it refused a token "that came back from a
    seam"; a frozen dataclass gives no such guarantee, because `copy.copy` and every
    `__reduce__` path build an instance without calling `__init__`, after which
    `object.__setattr__` puts `"tbd"` in a field whose type says it cannot hold one.
    """

    def _token(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        return tracker.authorize_claim(
            "akh-g15-fusion", purpose="one decode sweep", authorized_by="agent-a")

    def test_a_token_whose_falsifier_was_edited_after_construction_is_refused(self):
        import copy as _copy
        import pickle as _pickle
        for name, rebuild in (
            ("copy.copy", _copy.copy),
            ("pickle round-trip", lambda t: _pickle.loads(_pickle.dumps(t))),
        ):
            for value, state in ((None, H.FALSIFIER_ABSENT),
                                 ("tbd", H.FALSIFIER_PLACEHOLDER)):
                with self.subTest(route=name, falsifier=value):
                    forged = rebuild(self._token())
                    object.__setattr__(forged, "falsifier", value)
                    calls = []
                    with self.assertRaises(H.FalsifierRequiredBeforeCompute) as ctx:
                        H.claim_for_hypothesis(
                            forged, lambda **kw: calls.append(kw))
                    self.assertIn(state, str(ctx.exception))
                    self.assertEqual(calls, [])

    def test_the_control_an_unedited_token_still_opens_the_door(self):
        acquired = []
        token = self._token()
        H.claim_for_hypothesis(token, lambda **kw: acquired.append(kw))
        self.assertEqual(len(acquired), 1)
        self.assertIn(G15_FALSIFIER, acquired[0]["purpose"])


class TornTailIsRepairableTest(_TempCase):
    """`append` refuses to write onto a torn tail — correctly — but nothing could clear
    one, so a process killed mid-append froze EVERY question in the campaign until a
    human edited an append-only durable record by hand. That is the absence of a
    recovery procedure, not one.
    """

    FRAGMENT = b'{"seq": 2, "kind": "HYPOTHESIS_RESOL'

    def _torn(self):
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        with open(tracker.ledger.path, "ab") as handle:
            handle.write(self.FRAGMENT)
        return store, tracker

    def test_a_torn_tail_freezes_every_writer_until_it_is_repaired(self):
        store, tracker = self._torn()
        self.assertEqual(tracker.read().discarded_tail_bytes, len(self.FRAGMENT))
        with self.assertRaises(H.HypothesisLedgerCorruption):
            tracker.note_attempt(
                "akh-g15-fusion", proposal_id="akp-1", disposition="applied",
                bears_on_falsifier=False, note="n")
        self.assertEqual(tracker.repair_torn_tail(), len(self.FRAGMENT))
        self.assertEqual(tracker.read().discarded_tail_bytes, 0)
        tracker.note_attempt(
            "akh-g15-fusion", proposal_id="akp-1", disposition="applied",
            bears_on_falsifier=False, note="n")
        self.assertEqual(len(tracker.get("akh-g15-fusion").attempts), 1)

    def test_the_repair_removes_the_fragment_and_nothing_else(self):
        store, tracker = self._torn()
        before = tracker.read().events
        tracker.repair_torn_tail()
        self.assertEqual(tracker.read().events, before)
        self.assertEqual(tracker.get("akh-g15-fusion").hypothesis.falsifier,
                         G15_FALSIFIER)

    def test_the_control_repairing_an_intact_ledger_is_a_no_op(self):
        """A repair that is a way to drop a record that DID land is not a repair."""
        store = self.write_store(_store_doc(_g15_entry()))
        tracker = self.tracker()
        tracker.intake(store)
        with open(tracker.ledger.path, "rb") as handle:
            durable = handle.read()
        self.assertEqual(tracker.repair_torn_tail(), 0)
        with open(tracker.ledger.path, "rb") as handle:
            self.assertEqual(handle.read(), durable)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
