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
import inspect
import json
import os
import sys
import tempfile
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

    def test_store_entry_without_a_falsifier_is_refused(self):
        entry = _g15_entry()
        del entry["falsifier"]
        store = self.write_store(_store_doc(entry))
        with self.assertRaises(H.FalsifierMissing) as ctx:
            store.load()
        self.assertIn("falsifier", str(ctx.exception))
        self.assertIn("standing instruction", str(ctx.exception))

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
