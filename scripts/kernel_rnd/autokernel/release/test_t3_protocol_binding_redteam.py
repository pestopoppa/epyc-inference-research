#!/usr/bin/env python3
"""test_t3_protocol_binding_redteam.py — adversarial pass over the protocol-binding
closure (per-phase ratification, the adapter readiness seam, the packager verb
vocabulary, and backend attribution on archived libraries).

Run standalone:

    python3 -W error::ResourceWarning -m unittest \
        scripts.kernel_rnd.autokernel.release.test_t3_protocol_binding_redteam

WHY THIS IS A SEPARATE FILE
---------------------------
It is not a second opinion on the closure's own suite; it is the three defects that
survived it. Each class below states the composition the original guard got wrong —
a correct check wired to an input that does not carry the fact, or a conjunct that
disappears with its subject — and each carries the COMPLIANT-PATH control beside it,
because two of the three defects this package has shipped were guards that forbade
their own legitimate idiom.

Zero inference, zero builds, zero process management, no read of any production
tree. Every fixture is hand-built or borrowed from the two suites next door.
"""
from __future__ import annotations

import unittest

from .. import schemas
from . import packager, t3
from .test_packager import (
    AUTOPILOT_BASELINE, ARCHIVE_ROOT, CPU_LINK, ERA_REGISTRY, GPU_LINK,
    INCUMBENT_BINARIES, LLAMA_BACKENDS, NOW, V8_HEAD, digest, era_draft,
    incumbent_archive, operator_commands, rollback_plan, transaction_plan,
)
from .test_t3 import draft_protocol, ratified_protocol, request

# The two protocols the speech readiness seam is actually satisfied through. Neither
# is a per-phase grading instrument: `P-AK-SEARCH-1` is the search authority and
# `P-STT-REL-1` is a family-level release protocol, so neither has a workload phase
# to be filed under and both arrive in `T3Request.protocol_registry`.
SEAM_PROTOCOLS = ("P-AK-SEARCH-1", "P-STT-REL-1")


# =============================================================================
# Defect 1 — the ratification facet covered the per-phase half only
# =============================================================================

class TestTheFingerprintCoversTheRegistryHalfOfRatification(unittest.TestCase):
    """`declared_ratified_protocol_ids()` reads THREE sources — `protocol`,
    `phase_protocols`, `protocol_registry` — and `phase_identity_preflight` BLOCKS on
    what they produce. The `phase_protocol_standing` facet covered one of the three.

    The omitted one is the one that moves: a speech backend is unblocked by declaring
    `P-AK-SEARCH-1` and the family's release protocols as ratified `ProtocolBinding`s,
    and those are registry entries, not phase entries. So the exact rerun the facet
    was added to admit — the operator ratifies, we run again — computed an UNCHANGED
    fingerprint and was refused. Fail-closed, and still the wrong answer, which is
    what the facet's own comment says it exists to prevent.
    """

    def _draft(self, **overrides) -> t3.T3Request:
        return request(protocol_registry=tuple(draft_protocol(p)
                                               for p in SEAM_PROTOCOLS), **overrides)

    def _ratified(self, **overrides) -> t3.T3Request:
        return request(protocol_registry=tuple(ratified_protocol(p, annex="K")
                                               for p in SEAM_PROTOCOLS), **overrides)

    def test_the_facet_set_names_the_registry_half(self):
        self.assertIn("protocol_registry_standing", t3.FINGERPRINT_FACETS)
        self.assertIn("phase_protocol_standing", t3.FINGERPRINT_FACETS)

    def test_ratifying_a_registry_protocol_moves_the_fingerprint(self):
        """The bite. Before the fix these two fingerprints are byte-identical."""
        self.assertNotEqual(self._draft().fingerprint(), self._ratified().fingerprint())

    def test_adding_a_registry_binding_at_all_moves_the_fingerprint(self):
        """Absent and present-but-draft are different evidence states, and the
        adapters are asked a different question in each."""
        self.assertNotEqual(request().fingerprint(), self._draft().fingerprint())

    def test_the_post_ratification_rerun_is_admitted_rather_than_refused(self):
        """End to end through §9.1, which is where the defect was actually felt.

        Run 1 FAILs because the speech family is a draft. The operator ratifies it.
        Run 2 must be a NEW fingerprint — otherwise `check_rerun` reads it as the same
        evidence and returns `REFUSED_UNCHANGED_FINGERPRINT`, and the only way past
        that is a repair receipt for a stage that never ran wrong.
        """
        blocked = self._draft()
        prior = t3.T3Attempt(
            fingerprint=blocked.fingerprint(), verdict="FAIL",
            completed_at="2026-08-03T09:00:00Z",
            bundle_sha256=digest("bundle:blocked-on-draft-protocols"),
            failed_phases=(t3.PHASE_IDENTITY_PREFLIGHT,))
        after = self._ratified(attempt_ledger=(prior,))
        disposition = t3.check_rerun(
            after.fingerprint(), after.attempt_ledger, now=NOW, cooldown_seconds=3600)
        self.assertTrue(disposition.admissible, disposition.reason)
        self.assertEqual(disposition.code, t3.RERUN_ADMITTED_NEW_FINGERPRINT)

    # -- compliant-path controls ------------------------------------------------

    def test_an_unchanged_registry_keeps_the_fingerprint_stable(self):
        """Control: the facet must not make the idempotence key perturbable for free.

        Passes with and without the fix — a fingerprint that moved on its own would
        turn §9.1 off rather than sharpen it.
        """
        self.assertEqual(self._ratified().fingerprint(), self._ratified().fingerprint())

    def test_registry_order_does_not_change_the_fingerprint(self):
        """Control: the same declaration written in the other order is the same
        declaration. A facet sensitive to tuple order would refuse a rerun on a
        cosmetic edit."""
        forward = request(protocol_registry=tuple(
            ratified_protocol(p, annex="K") for p in SEAM_PROTOCOLS))
        reverse = request(protocol_registry=tuple(
            ratified_protocol(p, annex="K") for p in reversed(SEAM_PROTOCOLS)))
        self.assertEqual(forward.fingerprint(), reverse.fingerprint())

    def test_a_foreign_shape_in_the_registry_facet_is_refused_not_hashed(self):
        """The facet may not silently digest something that is not a binding."""
        with self.assertRaises(t3.T3InputError):
            t3.sealed_fingerprint(
                sealed=request().sealed, plan=request().plan,
                protocol=request().protocol, protocol_registry=("P-AK-SEARCH-1",))


# =============================================================================
# Defect 2 — the library attribution held at the compiler door only
# =============================================================================

def _hand_built_plan(**overrides) -> packager.RollbackPlan:
    """A `RollbackPlan` constructed DIRECTLY, the way `assemble_release_package`
    accepts one. `build_rollback_plan()` is not the only door and never was."""
    fields = {
        "rollback_branch": "production-consolidated-v8",
        "rollback_head": V8_HEAD,
        "incumbent_archive_path": ARCHIVE_ROOT,
        "incumbent_binaries": tuple(
            (backend, f"{ARCHIVE_ROOT}/{backend}/llama-server", INCUMBENT_BINARIES[backend])
            for backend in LLAMA_BACKENDS),
        "incumbent_libraries": ((LLAMA_BACKENDS, f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0",
                                 digest("v8-libggml-base")),),
        "stable_path_restore": ((CPU_LINK, "/mnt/raid0/llm/llama.cpp/build/bin"),
                                (GPU_LINK, "/mnt/raid0/llm/llama.cpp/build-hip/bin")),
        # Supplied by the caller, exactly as the dataclass allows. This is the point:
        # the verdict is a FIELD, so the party being gated can hand in a PASS.
        "archive_check": schemas.Check(schemas.PASS),
        "verified_at": NOW,
    }
    fields.update(overrides)
    return packager.RollbackPlan(**fields)


class TestRollbackAttributionHoldsAtBothDoors(unittest.TestCase):
    """`verify_archive_target()` FAILs a rolled-back backend with no attributed
    library — and `assemble_release_package()` never calls it. It takes a
    `RollbackPlan` OBJECT whose `archive_check` is a plain field, so a plan built by
    hand with `incumbent_libraries=()` and `archive_check=PASS` carried the new
    requirement past the only place that asked. Same shape as the `unchanged_view()`
    seam already recorded in this package's README: the compiler is not the only door.

    The conjunct is not satisfiable by deletion in the other direction either —
    dropping the BINARY to escape the requirement drops the backend out of
    `RollbackPlan.backends`, which `assemble_release_package` already reports against
    the sealed backend set.
    """

    def test_a_hand_built_plan_may_not_drop_the_attribution_entirely(self):
        with self.assertRaises(packager.RollbackIncomplete) as caught:
            _hand_built_plan(incumbent_libraries=())
        self.assertIn("no attributed library", str(caught.exception))
        self.assertIn("llama_cpu", str(caught.exception))
        self.assertIn("llama_gpu", str(caught.exception))

    def test_a_plan_attributing_only_one_of_two_rolled_back_backends_is_refused(self):
        """The half-attributed archive is the realistic shape, not the empty one:
        one `libggml-base.so.0` preserved and filed against the CPU build alone."""
        with self.assertRaises(packager.RollbackIncomplete) as caught:
            _hand_built_plan(incumbent_libraries=(
                (("llama_cpu",), f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0",
                 digest("v8-libggml-base")),))
        self.assertIn("llama_gpu", str(caught.exception))
        self.assertNotIn("'llama_cpu'", str(caught.exception).split("attributes")[0])

    def test_a_passing_archive_check_does_not_buy_the_attribution(self):
        """The verdict is read from the party supplying it; the requirement must not
        be reachable through that field."""
        with self.assertRaises(packager.RollbackIncomplete):
            _hand_built_plan(incumbent_libraries=(),
                             archive_check=schemas.Check(schemas.PASS))

    # -- compliant-path controls ------------------------------------------------

    def test_a_shared_library_may_name_both_backends(self):
        """Control: one `libggml-base.so.0` legitimately serves both llama backends
        of one tree, and the guard must not demand a library per backend."""
        plan = _hand_built_plan()
        self.assertEqual(plan.backends, tuple(sorted(LLAMA_BACKENDS)))
        self.assertEqual(len(plan.incumbent_libraries), 1)

    def test_one_library_per_backend_is_equally_accepted(self):
        """Control: and the guard must not demand a SHARED library either."""
        plan = _hand_built_plan(incumbent_libraries=(
            (("llama_cpu",), f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0", digest("cpu-ggml")),
            (("llama_gpu",), f"{ARCHIVE_ROOT}/gpu/libggml-base.so.0", digest("gpu-ggml")),
        ))
        self.assertEqual(len(plan.incumbent_libraries), 2)

    def test_the_compiled_plan_is_unaffected(self):
        """Control: `build_rollback_plan()` over a well-formed archive still builds,
        and still carries the attribution through verbatim."""
        plan = rollback_plan()
        self.assertEqual(plan.archive_check.outcome, schemas.PASS,
                         plan.archive_check.reasons)
        self.assertEqual(plan.incumbent_libraries,
                         incumbent_archive().entry(t3.ARCHIVE_GENERATION_N1).libraries)

    def test_an_extra_backend_in_the_attribution_is_not_penalised(self):
        """Control: the archive may attribute a library to a backend this rollback
        does not restore. The rule is coverage of the rolled-back set, not equality."""
        plan = _hand_built_plan(incumbent_libraries=(
            (tuple(sorted(LLAMA_BACKENDS + ("whisper_stt",))),
             f"{ARCHIVE_ROOT}/cpu/libggml-base.so.0", digest("v8-libggml-base")),))
        self.assertIn("whisper_stt", plan.incumbent_libraries[0][0])


# =============================================================================
# Defect 3 — the era-registry conjunct disappeared with its subject
# =============================================================================

def _sequence_without_the_era_step() -> tuple:
    """The operator sequence with the `$EDITOR instrument_eras.yaml` step removed and
    the remaining steps renumbered — i.e. a package that writes no era row at all."""
    kept = [c for c in operator_commands() if ERA_REGISTRY not in c.scanned_text]
    return tuple(
        packager.OperatorCommand(
            step=index + 1, command=c.command, purpose=c.purpose,
            expected_effect=c.expected_effect, target_paths=c.target_paths,
            validation_receipt=c.validation_receipt,
            validation_method=c.validation_method, validated=c.validated,
            rollback_command=c.rollback_command)
        for index, c in enumerate(kept))


def _review(era_row, commands=None) -> packager.CommandSequenceReview:
    return packager.validate_command_sequence(
        commands if commands is not None else operator_commands(),
        transaction=transaction_plan(), rollback=rollback_plan(),
        era_row=era_row, autopilot_baseline_path=AUTOPILOT_BASELINE)


class TestTheEraRegistryConjunctCannotBeDeleted(unittest.TestCase):
    """`_transaction_elements()` appended the `era_registry` element only when the
    era row happened to carry a `registry_path`. Every other element of the coverage
    denominator comes off the transaction; this one came off a caller-supplied
    mapping and was CONDITIONAL on a key being present in it.

    So the strengthened verb vocabulary — the whole point of the closure's item 2 —
    was reachable around: omit the key, omit the step, and the human-only era-registry
    write left the package with the sequence reviewed as complete. That is the
    `readiness.py` `objective_met` defect ("an objective naming only `decode` reached
    objective_met") in the packager, and the closure's own README names it as the
    class it was closing.
    """

    def test_an_era_row_naming_no_registry_is_a_finding(self):
        """The bite, in its exploitable form: no key AND no step, which before the
        fix reviewed as a fully covered sequence."""
        row = {k: v for k, v in era_draft().items() if k != "registry_path"}
        review = _review(row, _sequence_without_the_era_step())
        self.assertEqual(review.check.outcome, schemas.FAIL)
        self.assertIn("ERA_ROW_NAMES_NO_REGISTRY_PATH", " ".join(review.findings))

    def test_a_blank_registry_path_is_the_same_finding(self):
        """`""` is the other way to have the key without having the fact."""
        row = dict(era_draft(), registry_path="")
        review = _review(row, _sequence_without_the_era_step())
        self.assertIn("ERA_ROW_NAMES_NO_REGISTRY_PATH", " ".join(review.findings))

    def test_a_non_string_registry_path_is_the_same_finding(self):
        row = dict(era_draft(), registry_path=None)
        review = _review(row, _sequence_without_the_era_step())
        self.assertIn("ERA_ROW_NAMES_NO_REGISTRY_PATH", " ".join(review.findings))

    def test_the_element_is_still_in_the_denominator_when_it_is_named(self):
        """And the coverage requirement it gates is genuinely enforced: naming the
        registry while removing the step that writes it is UNCOMMANDED, not covered."""
        review = _review(era_draft(), _sequence_without_the_era_step())
        self.assertIn("TRANSACTION_STEP_UNCOMMANDED", " ".join(review.findings))
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.uncovered_elements)

    # -- compliant-path controls ------------------------------------------------

    def test_the_ordinary_drafted_sequence_still_passes(self):
        """Control: the real package — a drafted era row plus the sanctioned
        `$EDITOR instrument_eras.yaml` step — must be unaffected."""
        review = _review(era_draft())
        self.assertEqual(review.check.outcome, schemas.PASS, review.findings)
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.covered_elements)

    def test_the_helper_reads_the_drafted_row(self):
        """Control: the reader the guard and the enumerator share must agree with
        what `draft_era_registry_row()` actually writes."""
        self.assertEqual(packager.era_row_registry_path(era_draft()), ERA_REGISTRY)
        self.assertEqual(packager.era_row_registry_path({}), "")


# =============================================================================
# Defect 4 — the comment hole survived under a different quoting character
# =============================================================================

def _command(step: int, text: str, targets=()) -> packager.OperatorCommand:
    return packager.OperatorCommand(
        step=step, command=text, purpose="red-team probe",
        expected_effect="whatever this line actually does", target_paths=tuple(targets),
        validation_receipt=digest(f"prevalidation:probe:{step}"),
        validation_method="static pre-validation", validated=True,
        rollback_command="true")


class TestAVerbMayNotBeReadOutOfAQuotedString(unittest.TestCase):
    """The closure replaced *"the element's value appears anywhere in the pooled
    text"* with *"named and acted on in ONE comment-stripped command"*, citing the
    case it fixed: `$EDITOR something_else  # remember to update instrument_eras.yaml`.

    `#` is not the only way to write prose on a command line. `echo "reminder:
    $EDITOR orchestration/instrument_eras.yaml"` puts the element's name in the
    executable text and `$EDITOR` in the token set while running `echo` — and the
    whole sequence reviewed as fully covered, PASS, no findings. Same hole, one
    quoting character over.
    """

    def test_a_quoted_reminder_does_not_act_on_the_era_registry(self):
        """The bite. Before the fix this returns True."""
        probe = _command(1, f'echo "reminder: $EDITOR {ERA_REGISTRY}"')
        self.assertFalse(packager._acts_on(probe, "era_registry", ERA_REGISTRY))

    def test_the_whole_sequence_is_not_covered_by_a_quoted_reminder(self):
        """And end to end: a package whose only mention of the human-only registry
        write is inside an `echo` must not review as complete."""
        commands = tuple(
            _command(c.step, f'echo "reminder: $EDITOR {ERA_REGISTRY}"')
            if (ERA_REGISTRY in c.scanned_text and c.command.startswith("$EDITOR"))
            else c
            for c in operator_commands())
        review = _review(era_draft(), commands)
        self.assertEqual(review.check.outcome, schemas.FAIL)
        self.assertIn(f"era_registry:{ERA_REGISTRY}", review.uncovered_elements)

    def test_a_quoted_reminder_cannot_cover_the_autopilot_baseline_either(self):
        probe = _command(1, f"echo 'later: $EDITOR {AUTOPILOT_BASELINE}'")
        self.assertFalse(packager._acts_on(probe, "autopilot_baseline",
                                           AUTOPILOT_BASELINE))

    # -- compliant-path controls ------------------------------------------------

    def test_the_editor_idiom_is_still_a_verb(self):
        """Control: `$EDITOR <path>` is the sanctioned way to write a human-only
        registry row. Narrowing where a verb may be read must not touch it."""
        probe = _command(1, f"$EDITOR {ERA_REGISTRY}", (ERA_REGISTRY,))
        self.assertTrue(packager._acts_on(probe, "era_registry", ERA_REGISTRY))

    def test_a_trailing_comment_on_a_real_command_is_still_coverage(self):
        """Control: the comment is prose, the command in front of it is not."""
        probe = _command(1, f"$EDITOR {ERA_REGISTRY}  # write the three E9 rows",
                         (ERA_REGISTRY,))
        self.assertTrue(packager._acts_on(probe, "era_registry", ERA_REGISTRY))

    def test_a_quoted_argument_may_still_NAME_the_element(self):
        """Control, and the reason only the VERB half was narrowed: `python3 -c
        "…"` edits the registry with the path inside the quotes and the verb outside.
        Narrowing both halves would forbid it — the defect that once left the serving
        adapter unable to name its own launch command."""
        probe = _command(
            1, f"python3 -c \"import pathlib; pathlib.Path('{ERA_REGISTRY}').touch()\"")
        self.assertTrue(packager._acts_on(probe, "era_registry", ERA_REGISTRY))

    def test_a_quoted_message_beside_a_real_verb_is_still_coverage(self):
        """Control: `git tag -m "…"` carries prose AND does the thing."""
        probe = _command(
            1, 'git -C /mnt/raid0/llm/llama.cpp tag -a production-consolidated-v9 '
               '-m "cut over from v8; see the freeze receipts"')
        self.assertTrue(packager._acts_on(probe, "tag", "production-consolidated-v9"))

    def test_an_unterminated_quote_does_not_swallow_a_real_verb(self):
        """Control: the span pattern needs a closing quote, so a stray `\"` cannot
        turn the rest of a genuine command into prose and manufacture a finding."""
        probe = _command(1, f'$EDITOR {ERA_REGISTRY} " oops', (ERA_REGISTRY,))
        self.assertTrue(packager._acts_on(probe, "era_registry", ERA_REGISTRY))

    def test_the_real_operator_sequence_is_unaffected(self):
        """Control: the drafted sequence this package actually ships still passes."""
        review = _review(era_draft())
        self.assertEqual(review.check.outcome, schemas.PASS, review.findings)


# =============================================================================
# Defect 5 — the readiness audit proved the call, not the consequence
# =============================================================================

_AUDIT_PREAMBLE = (
    "RELEASE_READINESS_BY_BACKEND = {}\n"
    "def declared_ratified_protocol_ids(request):\n    return ()\n"
)


class TestTheReadinessAuditProvesTheVerdictIsAdjudicated(unittest.TestCase):
    """`audit_backend_readiness_is_consulted()` proved two things — phase 1 READS the
    registry, and phase 1 CALLS what it read. Neither implies the third: that the
    answer is used.

    A `phase_identity_preflight` that computes `readiness = readiness_of(ids)` and
    drops it PASSED the audit. That is the pre-AK5 defect one step further in, and it
    is strictly less visible: the registry is populated, the call is in the trace, the
    bundle carries a `backend_release_readiness` detail block, and a whisper release
    graded entirely under draft protocols emerges as though the adapters had approved
    it. An audit that cannot see the difference is a parser with a docstring.
    """

    def test_the_live_module_still_passes(self):
        self.assertEqual(t3.audit_backend_readiness_is_consulted().outcome, schemas.PASS)

    def test_a_module_that_computes_the_verdict_and_discards_it_fails(self):
        """The bite. Before the fix this is PASS."""
        source = _AUDIT_PREAMBLE + (
            "def phase_identity_preflight(request):\n"
            "    blocking = []\n"
            "    ids = declared_ratified_protocol_ids(request)\n"
            "    for backend in request.plan.backends:\n"
            "        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)\n"
            "        if readiness_of is None:\n"
            "            continue\n"
            "        readiness = readiness_of(ids)\n"
            "    if not request.resource_claims:\n"
            "        blocking.append('a real blocker, just not this one')\n"
            "    return blocking\n"
        )
        check = t3.audit_backend_readiness_is_consulted(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("reaches `blocking`", " ".join(check.reasons))

    def test_a_module_that_never_binds_the_result_fails(self):
        """The cruder shape: call it as a bare statement."""
        source = _AUDIT_PREAMBLE + (
            "def phase_identity_preflight(request):\n"
            "    blocking = []\n"
            "    for backend in request.plan.backends:\n"
            "        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)\n"
            "        readiness_of(declared_ratified_protocol_ids(request))\n"
            "    return blocking\n"
        )
        check = t3.audit_backend_readiness_is_consulted(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("without binding its result", " ".join(check.reasons))

    def test_appending_a_DIFFERENT_verdict_does_not_satisfy_it(self):
        """The guard must join the readiness verdict specifically, not observe that
        `blocking` is appended to somewhere in a 200-line function."""
        source = _AUDIT_PREAMBLE + (
            "def phase_identity_preflight(request):\n"
            "    blocking = []\n"
            "    ids = declared_ratified_protocol_ids(request)\n"
            "    host = request.host\n"
            "    for backend in request.plan.backends:\n"
            "        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)\n"
            "        readiness = readiness_of(ids)\n"
            "    if host is None:\n"
            "        blocking.append('host health')\n"
            "    return blocking\n"
        )
        self.assertEqual(t3.audit_backend_readiness_is_consulted(source).outcome,
                         schemas.FAIL)

    # -- compliant-path controls ------------------------------------------------

    def test_the_if_then_append_idiom_passes(self):
        """Control: the shape the live module uses."""
        source = _AUDIT_PREAMBLE + (
            "def phase_identity_preflight(request):\n"
            "    blocking = []\n"
            "    ids = declared_ratified_protocol_ids(request)\n"
            "    for backend in request.plan.backends:\n"
            "        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)\n"
            "        if readiness_of is None:\n"
            "            continue\n"
            "        readiness = readiness_of(ids)\n"
            "        if readiness.outcome != 'PASS':\n"
            "            blocking.append(str(readiness))\n"
            "    return blocking\n"
        )
        self.assertEqual(t3.audit_backend_readiness_is_consulted(source).outcome,
                         schemas.PASS)

    def test_the_factored_out_helper_idiom_also_passes(self):
        """Control, and the reason the audit accepts two routes. Recognising only
        `if`-then-append would FAIL the moment somebody factored the reasons into a
        helper — a guard forbidding a legitimate rewrite of what it guards. This is
        the defect that once left the serving adapter unable to name its own launch
        command, and it is cheaper to avoid than to find."""
        source = _AUDIT_PREAMBLE + (
            "def phase_identity_preflight(request):\n"
            "    blocking = []\n"
            "    ids = declared_ratified_protocol_ids(request)\n"
            "    for backend in request.plan.backends:\n"
            "        readiness_of = RELEASE_READINESS_BY_BACKEND.get(backend)\n"
            "        if readiness_of is None:\n"
            "            continue\n"
            "        verdict = readiness_of(ids)\n"
            "        blocking.extend(_readiness_reasons(backend, verdict))\n"
            "    return blocking\n"
        )
        self.assertEqual(t3.audit_backend_readiness_is_consulted(source).outcome,
                         schemas.PASS)

    def test_empty_and_foreign_source_are_still_could_not_check(self):
        """Control: the identity binding must not have become a FAIL. A guarantee
        obtainable by handing the auditor a different string is not one, and an audit
        that FAILs on absence is as wrong as one that PASSes on it — it reports a
        defect in a module it never read."""
        for source in ("", "import os\nx = 1\n", "def phase_identity_preflight(r):\n"
                                                 "    return None\n"):
            self.assertEqual(t3.audit_backend_readiness_is_consulted(source).outcome,
                             schemas.COULD_NOT_CHECK, source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
