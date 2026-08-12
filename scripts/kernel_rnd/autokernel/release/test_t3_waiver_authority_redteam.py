#!/usr/bin/env python3
"""test_t3_waiver_authority_redteam.py — adversarial pass over what a §10.4 waiver is
allowed to DO once it has been read.

Run standalone:

    python3 -W error::ResourceWarning -m unittest \
        scripts.kernel_rnd.autokernel.release.test_t3_waiver_authority_redteam

WHY THIS IS A SEPARATE FILE
---------------------------
`waiver_binding_from_path()` closed the question *"where did this document come
from"*. This file attacks the question it does not answer: given a document that
genuinely was read, WHAT MAY IT SUPPRESS, and WHO may it say wrote it. Three of the
findings below are reachable with the genuine, ratified, hash-verified v8 WAIVE-Q8
attestation — no forgery at all — because they are defects in how its scope and its
authorship are resolved, not in how its bytes are obtained.

Each class states the composition that was wrong, and each carries a COMPLIANT-PATH
control beside it. The control that matters most is the same one throughout: the real
`artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json`, in its legacy
`epyc.cpu_prefill_v8.operator_waiver.v1` schema with none of the five attribution
fields, must still read, still verify, and still suppress exactly its two Q8 arms. A
reader that rejects the genuine ratified record is a worse outcome than the defect it
fixes.

Zero inference, zero builds, zero process management, no production tree touched, and
`/workspace/artifacts/operator/` is READ and never written.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from .. import schemas, storage
from . import t3
from .test_t3 import (
    _LIVE_HUMAN_ONLY_BOUNDARY, _TEST_ATTESTATION_ROOT, _test_human_only_boundary,
    NOW, matrix_cells, read_waiver, reasons_of, request,
)

#: The genuine ratified records. Both are READ; neither is ever written.
V8_WAIVER_PATH = "/workspace/artifacts/operator/waive_q8_cpu_prefill_v8_20260725.json"
V8_RATIFICATION_PATH = (
    "/workspace/artifacts/operator/ratify_v8_final_freeze_20260725.json")
#: `sha256(<the file's bytes>)`, which is what
#: `ratify_v8_final_freeze_20260725.json → evidence_sha256.waive_q8` pins. NOT
#: `schemas.content_hash`, which digests a canonical re-encoding and matches nothing
#: anybody ratified.
V8_WAIVER_SHA = "fcd52b61610fcc2782e11f41ffac359343233924805f83d872eeceffbb7522d7"
V8_WAIVER = json.loads(Path(V8_WAIVER_PATH).read_bytes())
V8_CANDIDATE = V8_WAIVER["candidate_head"]
V8_PRODUCTION = V8_WAIVER["production_head"]
#: The two arms the operator actually excluded, as this gate spells their cells.
V8_ARM_CELLS = tuple(f"llama_cpu.pair.{pair}"
                     for pair in V8_WAIVER["scope"]["excluded_pairs"])

_FORGERY_DIR = None


def setUpModule() -> None:  # noqa: N802 - unittest's spelling
    t3.human_only_boundary = _test_human_only_boundary


def _forgery_dir() -> Path:
    """A directory inside THIS CHECKOUT spelled `artifacts/operator/…`.

    This disposable worktree is deliberately admitted only by the suite's
    synthetic boundary. The live boundary refuses it. That keeps the hostile
    read-root experiment possible without manufacturing operator ownership from
    an arbitrary checkout name.
    """
    global _FORGERY_DIR
    if _FORGERY_DIR is None:
        _TEST_ATTESTATION_ROOT.mkdir(parents=True, exist_ok=True)
        _FORGERY_DIR = Path(tempfile.mkdtemp(prefix="_ak_authority_",
                                             dir=_TEST_ATTESTATION_ROOT))
    return _FORGERY_DIR


def tearDownModule() -> None:  # noqa: N802 - unittest's spelling
    global _FORGERY_DIR
    t3.human_only_boundary = _LIVE_HUMAN_ONLY_BOUNDARY
    if _FORGERY_DIR is not None:
        shutil.rmtree(_FORGERY_DIR, ignore_errors=True)
        _FORGERY_DIR = None


def _write(document, *, name="waive.json") -> tuple:
    payload = json.dumps(document, indent=1, sort_keys=True).encode("utf-8")
    target = _forgery_dir() / name
    target.write_bytes(payload)
    return (str(target), schemas.raw_bytes_digest(payload))


def _v8_binding(cells) -> t3.ReadWaiver:
    """The GENUINE v8 attestation, read from the operator's own tree, bound to
    `cells`. No overrides: default boundary, default attestation roots, and the
    ratification cross-check the freeze record supplies."""
    return t3.waiver_binding_from_path(
        V8_WAIVER_PATH, pinned_sha256=V8_WAIVER_SHA, waiver_id="WAIVE-Q8",
        covers_cell_ids=tuple(cells),
        ratification_pin=(V8_RATIFICATION_PATH, "waive_q8"))


def _v8_verify(binding, *, failing=None, **kwargs) -> t3.WaiverVerification:
    cells = tuple(binding.covers_cell_ids)
    return t3.verify_waiver(
        binding, candidate_commit=V8_CANDIDATE, production_base_commit=V8_PRODUCTION,
        campaign_id="ak-calibration", known_cell_ids=cells,
        failing_cell_ids=cells if failing is None else failing, now=NOW, **kwargs)


# =============================================================================
# Defect 1 — the reader's narrowing was an argument the reader offered
# =============================================================================

class TestTheGateHoldsItsOwnAttestationRoots(unittest.TestCase):
    """`waiver_binding_from_path` narrows to `DEFAULT_ATTESTATION_ROOTS` because
    `operator_owned_path_check` answers a question about the SPELLING of a citation —
    a citation explicitly admitted by a boundary can still resolve under a root the
    gated party controls. The narrowing was a KEYWORD ARGUMENT of the reader, so the
    party being gated supplied it.

    MEASURED, end to end, before the fix: a document the loop wrote into its own
    checkout, naming `autokernel` as its author, read back with
    `attestation_roots=(<that directory>,)`, produced

        read=PASS  human_attested=PASS  attribution_source='operator_owned_path'
        verdict=PASS  covered_cell_ids=('llama_gpu.backend_correctness',)

    — the entire §10.4 defect restored, by an argument the reader offers. A reader
    cannot defend a boundary its caller hands it. So the party that decides a waiver
    SUPPRESSES a failing gate now states, itself, where operator attestations live,
    and defaults to the real root.
    """

    def _loop_authored(self, **overrides):
        document = {
            "schema": t3.WAIVER_SCHEMA_V8_CPU_PREFILL,
            "decision": "WAIVE-EVERYTHING",
            "protocol": "P-BENCH-PREFILL-1",
            "protocol_changed": False,
            "candidate_head": V8_CANDIDATE,
            "production_head": V8_PRODUCTION,
            "scope": {"excluded_pairs": ["llama_gpu.backend_correctness"]},
            "reason": "the loop wrote this itself",
            "consequences": ["nothing at all"],
            "ratified_at": "2026-08-03T00:00:00Z",
        }
        document.update(overrides)
        path, sha = _write(document, name="waive_loop_authored.json")
        return t3.waiver_binding_from_path(
            path, pinned_sha256=sha, waiver_id="FORGED",
            covers_cell_ids=("llama_gpu.backend_correctness",),
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))

    def test_the_reader_still_accepts_the_root_the_caller_hands_it(self):
        """The premise, asserted rather than assumed: the read itself succeeds. The
        defect is not that the bytes cannot be obtained — they can — it is that
        obtaining them conferred authority."""
        binding = self._loop_authored()
        self.assertTrue(binding.was_read)
        self.assertEqual(binding.read_receipt.attestation_root,
                         str(_TEST_ATTESTATION_ROOT))

    def test_the_gate_refuses_a_read_from_a_root_it_does_not_hold(self):
        """THE BITE. Delete the `under_any_root` guard in `verify_waiver`'s read
        predicate and this returns PASS with the failing cell covered."""
        verification = _v8_verify(self._loop_authored())
        self.assertFalse(verification.verified)
        self.assertEqual(verification.predicate_results["read"],
                         schemas.COULD_NOT_CHECK)
        self.assertEqual(verification.covered_cell_ids, ())
        self.assertIn("attestation roots this gate holds", " ".join(
            verification.check.reasons))

    def test_a_checkout_the_loop_can_write_is_not_the_operators_tree(self):
        """Naming the hazard rather than the symptom: the refused root is an
        `artifacts/operator/` directory that PASSES the citation check."""
        located = schemas.operator_owned_path_check(
            str(_TEST_ATTESTATION_ROOT / "w.json"),
            boundary=t3.human_only_boundary())
        self.assertEqual(located.outcome, schemas.PASS)
        self.assertFalse(schemas.under_any_root(
            str(_TEST_ATTESTATION_ROOT), t3.DEFAULT_ATTESTATION_ROOTS))

    def test_the_whole_run_blocks_rather_than_merely_not_suppressing(self):
        """End to end. A run that does not declare its roots gets the real one, so a
        waiver read from anywhere else stops the run instead of silently doing
        nothing."""
        result = t3.T3Runner().evaluate_release(request(
            waivers=(read_waiver(),), attestation_roots=()))
        self.assertNotEqual(result.verdict, "PASS_WITH_WAIVER")
        self.assertIn("attestation roots this gate holds", reasons_of(result))

    def test_an_empty_root_set_is_refused_rather_than_read_as_anywhere(self):
        with self.assertRaises(t3.T3InputError) as caught:
            _v8_verify(_v8_binding(V8_ARM_CELLS[:1]), attestation_roots=())
        self.assertIn("must not be empty", str(caught.exception))

    # -- compliant-path controls ------------------------------------------------

    def test_the_genuine_v8_record_verifies_with_no_roots_argument_at_all(self):
        """THE control. The real ratified attestation, at the operator's real path,
        read and verified with every default in place — no boundary override, no
        attestation-roots override, no fixture. If this ever fails, the guard has
        started refusing the record it exists to honour."""
        binding = _v8_binding(V8_ARM_CELLS)
        verification = _v8_verify(binding)
        self.assertTrue(verification.verified, verification.check.reasons)
        self.assertEqual(verification.predicate_results["read"], schemas.PASS)
        self.assertEqual(verification.predicate_results["attribution_source"],
                         "operator_owned_path")
        self.assertEqual(set(verification.covered_cell_ids), set(V8_ARM_CELLS))
        # ...and it is genuinely the legacy shape, carrying none of the five
        # attribution fields, which is why the provenance fallback is load-bearing.
        self.assertEqual(
            [f for f in schemas.ACTOR_ATTRIBUTION_FIELDS if f in V8_WAIVER], [])

    def test_a_run_that_declares_its_root_still_verifies_through_the_seam(self):
        """The seam must not forbid its own legitimate idiom: a run that SAYS where
        its attestations live is honoured, which is what makes this a declaration
        rather than a hardcoded path."""
        verification = _v8_verify(
            self._loop_authored(),
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertEqual(verification.predicate_results["read"], schemas.PASS)


# =============================================================================
# Defect 2 — scope tokens matched as raw substrings, and the model name outranked
#            the arms the operator actually listed
# =============================================================================

class TestAWaiverCoversOnlyTheCellsItsScopeNames(unittest.TestCase):
    """The scope predicate resolves the operator's scope tokens against this run's
    cell ids, and it did so with `token in cell_id` — a raw substring test — over a
    token list that FLATTENED `excluded_model` together with `excluded_pairs`.

    Both halves are reachable with the genuine v8 attestation, unmodified and
    hash-verified. Its scope names two CPU prefill/decode arms and, beside them, the
    model `qwen36_q8`; its consequences forfeit exactly *"No v8 Q8 non-regression
    claim may be made from this campaign."* MEASURED before the fix, that document
    covered `llama_gpu.qwen36_q8.backend_correctness` and
    `llama_gpu.qwen36_q8.quality` — a CORRECTNESS failure and a QUALITY regression,
    suppressed by a prefill eligibility-floor waiver, forfeiting a claim nobody made
    — and `x_qwen36_q8_y`, which the operator's token merely appears inside.

    Note what the fix must NOT be. Binding the waiver to the protocol it names would
    invalidate the ratified record: v8's `protocol` is `P-BENCH-PREFILL-1` and one of
    the two arms it excludes, `qwen36_q8-tg128-iqk1`, is a DECODE cell graded under
    `P-BENCH-1`. Making a new rule tidy at the cost of the real attestation is the
    failure mode here, not the fix.
    """

    def _scope(self, cell_id):
        return _v8_verify(_v8_binding((cell_id,))).predicate_results["scope"]

    def test_the_two_arms_the_operator_listed_are_covered(self):
        """COMPLIANT-PATH CONTROL, and it comes first because it is the constraint."""
        verification = _v8_verify(_v8_binding(V8_ARM_CELLS))
        self.assertEqual(verification.predicate_results["scope"], schemas.PASS)
        self.assertEqual(set(verification.covered_cell_ids), set(V8_ARM_CELLS))

    def test_a_prefill_waiver_does_not_cover_the_models_correctness_cell(self):
        """THE BITE. Restore `tokens = pairs + models` in `_waiver_scope` — or
        `token in cell_id` in the predicate — and this becomes PASS."""
        self.assertEqual(self._scope("llama_gpu.qwen36_q8.backend_correctness"),
                         schemas.FAIL)

    def test_a_prefill_waiver_does_not_cover_the_models_quality_cell(self):
        self.assertEqual(self._scope("llama_gpu.qwen36_q8.quality"), schemas.FAIL)

    def _model_only_scope(self, cell_id, *, model="qwen36_q8"):
        """The v8 document with its arm list removed, so the MODEL token is the
        operative scope. Deliberately isolated from the demotion fix: with the arms
        present the model token is demoted and a substring bug is unreachable
        through this document, which would leave the substring guard untested."""
        document = dict(V8_WAIVER)
        document["scope"] = {"excluded_model": model}
        path, sha = _write(document, name=f"waive_model_only_{model}.json")
        binding = t3.waiver_binding_from_path(
            path, pinned_sha256=sha, waiver_id="WAIVE-MODEL",
            covers_cell_ids=(cell_id,),
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        return _v8_verify(binding,
                          attestation_roots=(str(_TEST_ATTESTATION_ROOT),))

    def test_a_token_is_not_matched_as_a_fragment_of_a_component(self):
        """THE BITE for the substring half. `qwen36_q8` appears inside
        `x_qwen36_q8_y`, and appearing inside a name is not being named."""
        self.assertEqual(self._scope("x_qwen36_q8_y"), schemas.FAIL)
        self.assertEqual(
            self._model_only_scope("x_qwen36_q8_y").predicate_results["scope"],
            schemas.FAIL)
        # ...and a token is not matched across a component boundary either.
        self.assertEqual(
            self._model_only_scope("llama_cpu.qwen36_q8_prefill")
            .predicate_results["scope"], schemas.FAIL)

    def test_a_cell_the_scope_never_mentions_is_still_refused(self):
        """Anti-vacuity: the plainly-out-of-scope case must not have started passing
        for some unrelated reason."""
        self.assertEqual(self._scope("llama_gpu.backend_correctness"), schemas.FAIL)

    def test_a_waiver_that_names_only_a_model_still_covers_that_models_cells(self):
        """CONTROL for the demotion. Most-specific-wins must not delete the broad
        grant — it demotes it only when the operator ALSO listed arms or cells. A
        document that names a model and nothing else keeps the model as its scope."""
        document = dict(V8_WAIVER)
        document["scope"] = {"excluded_model": "qwen36_q8"}
        path, sha = _write(document, name="waive_model_only.json")
        binding = t3.waiver_binding_from_path(
            path, pinned_sha256=sha, waiver_id="WAIVE-MODEL",
            covers_cell_ids=("llama_cpu.qwen36_q8.prefill",),
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        verification = _v8_verify(
            binding, attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertEqual(verification.predicate_results["scope"], schemas.PASS)

    def test_a_document_with_no_resolvable_scope_still_covers_nothing(self):
        """The pre-existing floor, re-asserted: demotion must not have created a path
        where an empty token set reads as 'everything'."""
        document = dict(V8_WAIVER)
        document["scope"] = {"remaining_matched_pairs": 14}
        path, sha = _write(document, name="waive_no_scope.json")
        binding = t3.waiver_binding_from_path(
            path, pinned_sha256=sha, waiver_id="WAIVE-NONE",
            covers_cell_ids=("llama_cpu.prefill",),
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        verification = _v8_verify(
            binding, attestation_roots=(str(_TEST_ATTESTATION_ROOT),))
        self.assertEqual(verification.predicate_results["scope"], schemas.FAIL)
        self.assertEqual(verification.covered_cell_ids, ())


# =============================================================================
# Defect 3 — the machine-actor refusal scanned a closed list of five key names
# =============================================================================

class TestAnAttributionIsOneWhateverItsKeyIsCalled(unittest.TestCase):
    """`ACTOR_ATTRIBUTION_FIELDS`' own docstring says *"a guard that scans
    `authorized_by` and not `approved_by` is a guard with a rename-shaped hole"* — and
    the guard was still a closed list of five.

    So a §10.4 waiver carrying `waived_by: "autokernel"` was seen by NOTHING, and then
    took the *no attribution at all* branch of `verify_waiver`, which reads an
    unattributed document as human-attested on the strength of where it lives —
    because the preserved v8 record has no author field and that fallback exists for
    it. An EXPLICIT machine attribution in an unenumerated key was therefore strictly
    safer for a forger than no attribution at all.

    The refusal is widened; the NAMED-ACTOR set is not. Widening a refusal cannot
    admit anything that was refused before.
    """

    def _authored(self, **overrides):
        document = dict(V8_WAIVER)
        document.update(overrides)
        path, sha = _write(document, name="waive_attributed.json")
        return t3.waiver_binding_from_path(
            path, pinned_sha256=sha, waiver_id="WAIVE-Q8",
            covers_cell_ids=V8_ARM_CELLS,
            attestation_roots=(str(_TEST_ATTESTATION_ROOT),))

    def _verify(self, **overrides):
        return _v8_verify(self._authored(**overrides),
                          attestation_roots=(str(_TEST_ATTESTATION_ROOT),))

    def test_no_key_spelling_lets_the_loop_attribute_a_waiver_to_itself(self):
        """THE BITE. Revert `machine_attributions` to iterate
        `ACTOR_ATTRIBUTION_FIELDS` alone and every one of these verifies."""
        for key in ("waived_by", "signed_by", "issued_by", "created_by",
                    "requested_by", "granted_by", "author", "actor"):
            with self.subTest(key=key):
                verification = self._verify(**{key: "autokernel"})
                self.assertFalse(verification.verified)
                self.assertEqual(verification.predicate_results["human_attested"],
                                 schemas.FAIL)
                self.assertEqual(verification.covered_cell_ids, ())

    def test_no_digit_infix_spelling_of_the_loop_gets_through_either(self):
        """THE BITE for `identity_candidates`. `[a-z0-9]+` swallows a digit into the
        run it separates, so `autokernel`, `auto-kernel`, `auto_kernel` and
        `auto kernel` were all refused while `auto2kernel` sailed through — the same
        separator-shaped hole, with a digit as the separator. `strip(_DIGITS)` never
        reached it because it only strips the ends."""
        for identity in ("auto2kernel", "auto1kernel", "auto0pilot", "sub9agent"):
            with self.subTest(identity=identity):
                self.assertTrue(schemas.machine_actor_tokens(identity),
                                f"{identity!r} read as a human")

    # -- compliant-path controls ------------------------------------------------

    def test_the_genuine_v8_record_carries_no_attribution_key_at_all(self):
        """THE control for widening the scan: the ratified record has none of these
        keys, so nothing about it can have changed."""
        self.assertEqual(
            [k for k in schemas.attribution_keys(V8_WAIVER) if k in V8_WAIVER], [])
        self.assertEqual(schemas.machine_attributions(V8_WAIVER), ())
        self.assertTrue(_v8_verify(_v8_binding(V8_ARM_CELLS)).verified)

    def test_a_human_named_in_an_unenumerated_key_is_still_a_human(self):
        """The guard must not forbid its own legitimate idiom: widening WHICH keys
        are scanned must not widen WHO is refused."""
        for author in ("Daniele Pinna", "d.pinna", "ops-daniele", "Jean-Luc Picard",
                       "scriptor", "d4niele"):
            with self.subTest(author=author):
                self.assertEqual(schemas.machine_actor_tokens(author), ())
                self.assertTrue(self._verify(waived_by=author).verified)

    def test_the_named_actor_set_is_unchanged(self):
        """Only the refusal was widened. `attribution_source` still reads
        `named_actor` from the five enumerated fields and from nothing else, so a
        `*_by` key does not silently become an authority."""
        self.assertEqual(
            self._verify(authorized_by="daniele")
            .predicate_results["attribution_source"], "named_actor")
        self.assertEqual(
            self._verify(waived_by="daniele")
            .predicate_results["attribution_source"], "operator_owned_path")


# =============================================================================
# Controls that must keep holding — the two claims this pass did NOT refute
# =============================================================================

class TestTheClaimsThatSurvivedTheAttack(unittest.TestCase):
    """Recorded as tests rather than as prose, because a claim that survived one
    adversarial pass is exactly the kind of thing a later edit quietly removes."""

    def test_the_fingerprint_separates_waivers_covering_different_cells(self):
        """§9.1: two runs whose waivers cover different cells must not share an
        idempotence key, or a failed gate is rerunnable into a pass on a fingerprint
        that already succeeded. `active_waiver_coverage` holds this."""
        import dataclasses
        binding = read_waiver()
        other = dataclasses.replace(binding, covers_cell_ids=("llama_cpu.decode",))
        self.assertNotEqual(request(waivers=(binding,)).fingerprint(),
                            request(waivers=(other,)).fingerprint())

    def test_re_pointing_a_read_waiver_at_other_cells_is_caught_by_the_scope(self):
        """`dataclasses.replace` keeps the receipt and re-points the coverage, and
        `__post_init__` does not constrain `covers_cell_ids` — deliberately, since the
        binding is where the RUN says which cells it is invoking the waiver for. The
        containment is the scope predicate, so it is asserted here rather than
        assumed."""
        import dataclasses
        re_pointed = dataclasses.replace(
            _v8_binding(V8_ARM_CELLS),
            covers_cell_ids=("llama_gpu.backend_correctness",))
        verification = _v8_verify(re_pointed)
        self.assertEqual(verification.predicate_results["scope"], schemas.FAIL)
        self.assertEqual(verification.covered_cell_ids, ())

    def test_the_reader_never_reads_a_production_kernel_tree(self):
        """A frozen production tree is a record, never a waiver source, and the
        refusal is asserted against the real tree names rather than a fixture."""
        self.assertTrue(storage.production_tree_forms())
        for tree in storage.production_tree_forms():
            with self.subTest(tree=tree):
                self.assertFalse(schemas.under_any_root(
                    tree, t3.DEFAULT_ATTESTATION_ROOTS))

    def test_the_matrix_this_suite_reasons_about_is_the_real_one(self):
        """Anti-vacuity for every cell id above: the ones this file asserts are
        REFUSED must not simply be absent from the vocabulary."""
        self.assertIn("llama_cpu.prefill", [c.cell_id for c in matrix_cells()])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
