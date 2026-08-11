#!/usr/bin/env python3
"""test_correctness.py — the regression barrier for the T0 correctness surfaces.

WHY THIS FILE EXISTS
--------------------
`scripts/kernel_rnd/kernel_eval.sh` shipped after review with three defects that
were all visible in its source and none of which was ASSERTED anywhere:

  * it ran `test-backend-ops -o MUL_MAT` and nothing else, so every MoE expert
    path went unexercised on a host whose production worker is a MoE model;
  * it set `COH="coherent"` for any non-empty generation, with the baseline
    comparison optional; and
  * it emitted `"status":"OK"` regardless.

The tests below are the assertions those defects never had. The headline one is
`test_mul_mat_only_suite_fails_the_way_kernel_eval_sh_did_not`, which replays the
replaced script's exact op coverage and requires a FAIL.

NO inference, NO benchmark, NO build, NO sanitizer run, NO process, NO file
written. The suite also asserts that last property of the module under test by
running its AST self-audit.

Run:
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_correctness.py
    python3 -W error::ResourceWarning -m unittest \\
        scripts/kernel_rnd/autokernel/evaluator/test_correctness.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
import unittest
from pathlib import Path

# Import through the PACKAGE so `correctness.schemas` is the same module object
# `api` and `journal` validate with (README, "Import convention").
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel.evaluator import api  # noqa: E402
from autokernel.evaluator import correctness as C  # noqa: E402
from autokernel.evaluator import devices as D  # noqa: E402

PASS = S.Check(S.PASS)
NOW = "2026-08-03T12:00:00+00:00"
V8_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
#: v7, the rollback anchor. A REAL other commit, so an anchor-drift test compares
#: two commits that both exist rather than one commit and one malformed string.
V7_COMMIT = "6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
#: The three anchor-naming component names, as they appear on both evidence types.
ANCHOR_TRIPLE = ("anchor_source_commit", "anchor_binary_sha256", "anchor_linkage_sha256")


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures. Built field-by-field on purpose: `correctness` has no all_clear()
# helper, because a fixture that fabricates PASS is the fixture that removes the
# signal under test. Every builder below takes **overrides so a test can break
# exactly one thing and leave the other sixteen surfaces honest.
# ---------------------------------------------------------------------------

WORKTREE = "/mnt/raid0/llm/tmp/ak-campaigns/ak-llama_gpu-decode-20260803/wt-0001"
LIBROOT = f"{WORKTREE}/build/bin"


def anchor(**overrides) -> api.AnchorIdentity:
    kwargs = dict(source_commit=V8_COMMIT, binary_sha256=sha("anchor-binary"),
                  linkage_sha256=sha("anchor-linkage"),
                  measurement_event_ids=("ake-anchor-0001",))
    kwargs.update(overrides)
    return api.AnchorIdentity(**kwargs)


def request(**overrides) -> api.EvaluationRequest:
    kwargs = dict(
        event_id="ake-t0-0001",
        campaign_id="ak-llama_gpu-decode-20260803",
        candidate_id="akc-0001",
        tier="T0",
        backend="llama_gpu",
        phase="decode",
        cell_class="instrument_tokens_per_s",
        protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(
            source_sha256=sha("cand-source"),
            binary_sha256=sha("cand-binary"),
            linkage_sha256=sha("cand-linkage"),
        ),
        anchor=anchor(),
        evaluator=api.EvaluatorIdentity(
            id="P-AK-SEARCH-1/v1",
            bundle_sha256=sha("evaluator-bundle"),
            runtime_source_label_ref="ake-srclabel-0003",
        ),
        scope_denominator=api.ScopeDenominator(
            machine_subset="partial", numa_nodes=(), devices=("mi210_0",), cores=8),
        scope_manifest_sha256=sha("scope-manifest"),
        co_residency="single",
        determinism=api.DeterminismReport(
            determinism_class="bitwise_stable", same_seed_repeat_runs=3),
        metric="decode_tokens_per_s",
        metric_direction="higher_better",
        reps=10,
        change_class="parameter", anchor_tier="T0", transfer_ratio_to=(),
        created_at=NOW,
        # Precondition 8 is required of EVERY run, not only rate comparisons.
        campaign_controls=api.CampaignControls(
            calibration_block_count=30, contribution_floor=0.02, max_candidates=100,
            confirmation_admission_count=5, max_blocks_per_candidate=40,
            storage_floor_bytes_free=200 * 1024 ** 3),
        # A T0 correctness record is not a rate comparison, so it legitimately
        # carries no calibration block: `VOID_INCOMPLETE_CALIBRATION` and the
        # statistical conjuncts are rate-only.
        calibration=None,
        device_state=D.DeviceState(
            device_id="mi210_0", source="fixture/rocm-smi",
            nominal_sclk_mhz=1700, min_sclk_ratio=0.9,
            samples=(D.DeviceStateSample(1700, 1600, 180, 55, True),),
            receipt_ref="fixture://device-state/correctness"),
    )
    kwargs.update(overrides)
    return api.EvaluationRequest(**kwargs)


def policy(**overrides) -> C.T0Policy:
    kwargs = dict(
        required_backend_ops=("MUL_MAT", "MUL_MAT_ID"),
        symbol_shrinkage_reject_ratio=0.02,
        diff_ceiling=C.DiffComplexityCeiling(
            backend="llama_gpu", max_changed_lines=400, max_files_touched=8,
            shared_core_forces_review=True),
        determinism_min_runs=3,
        coherence_tolerance_floor=0.995,
        policy_ref="evaluator-bundle://t0/policy/llama_gpu/v1",
    )
    kwargs.update(overrides)
    return C.T0Policy(**kwargs)


def surface(**overrides) -> C.ChangeSurface:
    kwargs = dict(
        derived_touches_memory=True,
        derived_touches_threading=True,
        derived_touches_dispatch=True,
        derived_touches_persistent_state=True,
        derived_ops=("MUL_MAT_ID",),
        derived_files=(f"{WORKTREE}/ggml/src/ggml-cuda/mmq.cu",),
        declared_touches_memory=True,
        declared_touches_threading=True,
        declared_ops=("MUL_MAT_ID",),
        touches_shared_core_header=False,
        derivation_ref="ake-derivation-0001",
    )
    kwargs.update(overrides)
    return C.ChangeSurface(**kwargs)


def symbols(**overrides) -> C.SymbolTableDiff:
    kwargs = dict(
        removed_symbols=(), arity_changed_symbols=(), added_symbols=("_Z9mmq_id_v2Pf",),
        removed_op_registrations=(), removed_dispatch_predicates=(), declared_removals=(),
        anchor_symbol_count=18422, candidate_symbol_count=18423,
        tool_id="nm -D --defined-only", receipt_ref="data/ak/akc-0001/symbols.json",
        produced_by="evaluator")
    kwargs.update(overrides)
    return C.SymbolTableDiff(**kwargs)


def build(**overrides) -> C.BuildProvenance:
    kwargs = dict(
        built_from_snapshot_sha256=sha("cand-source"),
        build_dir=f"{WORKTREE}/build",
        build_dir_was_fresh=True,
        incremental_objects_present=False,
        compiler_id="hipcc", compiler_version="6.2.0",
        build_log_ref="data/ak/akc-0001/build.log",
        production_tree_paths_touched=(),
        output_binary_sha256=sha("cand-binary"),
        produced_by="evaluator")
    kwargs.update(overrides)
    return C.BuildProvenance(**kwargs)


def envelope(**overrides) -> C.ChangeClassEnvelope:
    kwargs = dict(change_class="dispatcher", max_changed_lines=300, max_files_touched=4)
    kwargs.update(overrides)
    return C.ChangeClassEnvelope(**kwargs)


def diff(**overrides) -> C.DiffPolicyEvidence:
    files = (f"{WORKTREE}/ggml/src/ggml-cuda/mmq.cu",)
    kwargs = dict(
        files_touched=files, declared_surface_files=files, unrelated_deletions=(),
        changed_lines=118, change_class="dispatcher", envelope=envelope(),
        branch_name="llama.cpp-experimental/ak-mmq-id-tile", commit_was_pathspec_limited=True,
        production_tree_paths=(), record_schema_violations=(),
        diff_ref="data/ak/akc-0001/diff.patch", produced_by="evaluator")
    kwargs.update(overrides)
    return C.DiffPolicyEvidence(**kwargs)


def static_analysis(**overrides) -> C.StaticAnalysisEvidence:
    kwargs = dict(
        compiler_id="hipcc", compiler_version="6.2.0",
        anchor_compiler_id="hipcc", anchor_compiler_version="6.2.0",
        error_count=0, warning_count=0, anchor_warning_count=0,
        anchor_source_commit=V8_COMMIT, anchor_binary_sha256=sha("anchor-binary"),
        anchor_linkage_sha256=sha("anchor-linkage"),
        warnings_as_errors=True,
        analyzer_id="clang-tidy-18", analyzer_error_findings=(),
        receipt_ref="data/ak/akc-0001/static.json", produced_by="evaluator")
    kwargs.update(overrides)
    return C.StaticAnalysisEvidence(**kwargs)


def invocation(**overrides) -> C.SanitizerInvocation:
    kwargs = dict(
        source_dir=WORKTREE, build_dir=f"{WORKTREE}/build-asan", target="test-backend-ops",
        run_argv=(f"{WORKTREE}/build-asan/bin/test-backend-ops", "-o", "MUL_MAT_ID"),
        jobs=8, backend="llama_gpu")
    kwargs.update(overrides)
    return C.build_sanitizer_invocation(**kwargs)


def sanitizers(**overrides) -> C.SanitizerEvidence:
    kwargs = dict(
        invocation=invocation(), executed=True, exit_code=0,
        asan_findings=(), ubsan_findings=(),
        sanitizer_build_binary_sha256=sha("cand-binary-asan"),
        log_ref="data/ak/akc-0001/sanitizer.log", produced_by="evaluator")
    kwargs.update(overrides)
    return C.SanitizerEvidence(**kwargs)


def op_suite(**overrides) -> C.OpSuiteEvidence:
    kwargs = dict(
        suite_id="test-backend-ops", suite_source_sha256=sha("cand-source"),
        ops_exercised=("MUL_MAT", "MUL_MAT_ID"), ops_failed=(),
        cases_by_op=(("MUL_MAT", 4231, 4231), ("MUL_MAT_ID", 1188, 1188)),
        shapes_ref="data/ak/akc-0001/shapes.json",
        receipt_ref="data/ak/akc-0001/tbo.json", produced_by="evaluator")
    kwargs.update(overrides)
    return C.OpSuiteEvidence(**kwargs)


def reference(**overrides) -> C.ReferenceEvidence:
    kwargs = dict(
        comparisons=(
            C.ReferenceComparison(
                shape_id="m4096n1k4096-q4_K", op="MUL_MAT", mode="exact_bitwise",
                mismatch_count=0, max_ulp_observed=None, tolerance_ulp=None,
                oracle_id="ik_llama.cpp@iqk-ref", oracle_is_candidate_derived=False),
            C.ReferenceComparison(
                shape_id="e128t1k4096-q4_K", op="MUL_MAT_ID", mode="ulp_bounded",
                mismatch_count=0, max_ulp_observed=1.0, tolerance_ulp=2.0,
                oracle_id="ik_llama.cpp@iqk-ref", oracle_is_candidate_derived=False),
        ),
        undefined_for=(), oracle_registry_ref="evaluator-bundle://oracles/v1",
        produced_by="evaluator")
    kwargs.update(overrides)
    return C.ReferenceEvidence(**kwargs)


def boundary(**overrides) -> C.BoundaryShapeEvidence:
    kwargs = dict(
        unseen_shapes=("m1n1k4096", "m8191n7k4096"), boundary_shapes=("m0n0k0", "m1n2048k1"),
        failures=(), selection_rule_id="ak.holdout.shape_partition/v1",
        selection_seed="campaign-seed-4711", held_out_from_planner=True,
        receipt_ref="data/ak/akc-0001/boundary.json", produced_by="evaluator")
    kwargs.update(overrides)
    return C.BoundaryShapeEvidence(**kwargs)


def trace(**overrides) -> C.DispatchTraceEvidence:
    kwargs = dict(
        derived_surface=("MUL_MAT_ID", "mul_mat_vec_q", "mmq_id_tile"),
        traced_kernels=("MUL_MAT_ID", "mmq_id_tile"),
        fallback_events=(), fallback_instrumentation_active=True,
        trace_ref="data/ak/akc-0001/dispatch.jsonl", produced_by="evaluator")
    kwargs.update(overrides)
    return C.DispatchTraceEvidence(**kwargs)


def state_safety(**overrides) -> C.StateSafetyEvidence:
    kwargs = dict(
        rollback_tested=True, teardown_tested=True, race_detector_id="tsan",
        race_findings=(), leaked_resources=(), orphan_processes=(),
        receipt_ref="data/ak/akc-0001/state.json", produced_by="evaluator")
    kwargs.update(overrides)
    return C.StateSafetyEvidence(**kwargs)


def coherence(**overrides) -> C.CoherenceEvidence:
    kwargs = dict(
        candidate_output_sha256=sha("gen-out"), candidate_output_len=160,
        anchor_output_sha256=sha("gen-out"), anchor_output_len=160,
        sampler_id="greedy-topk1-temp0", sampler_is_greedy=True, seed=42,
        tokens_requested=160, token_agreement_ratio=1.0, divergence_first_index=None,
        anchor_determinism_class="bitwise_stable",
        anchor_source_commit=V8_COMMIT, anchor_binary_sha256=sha("anchor-binary"),
        anchor_linkage_sha256=sha("anchor-linkage"),
        prompt_ref="evaluator-bundle://prompts/coherence/v1",
        receipt_ref="data/ak/akc-0001/coherence.json", produced_by="evaluator")
    kwargs.update(overrides)
    return C.CoherenceEvidence(**kwargs)


def determinism(**overrides) -> C.DeterminismEvidence:
    kwargs = dict(
        seed=42, runs=3,
        candidate_output_digests=(sha("gen-out"),) * 3,
        anchor_output_digests=(sha("gen-out"),) * 3,
        anchor_determinism_class="bitwise_stable",
        anchor_source_commit=V8_COMMIT, anchor_binary_sha256=sha("anchor-binary"),
        anchor_linkage_sha256=sha("anchor-linkage"),
        declared_class_change=False, declared_class_change_ref=None,
        receipt_ref="data/ak/akc-0001/determinism.json", produced_by="evaluator")
    kwargs.update(overrides)
    return C.DeterminismEvidence(**kwargs)


def linkage(**overrides) -> C.LinkageEvidence:
    kwargs = dict(
        binary_sha256=sha("cand-binary"), linkage_sha256=sha("cand-linkage"),
        anchor_source_commit=V8_COMMIT,
        anchor_binary_sha256=sha("anchor-binary"), anchor_linkage_sha256=sha("anchor-linkage"),
        resolved_libraries=(("libggml-base.so", f"{LIBROOT}/libggml-base.so", sha("ggml-base")),
                            ("libggml-hip.so", f"{LIBROOT}/libggml-hip.so", sha("ggml-hip"))),
        expected_library_root=LIBROOT, verifier_id="verify_ggml_linkage.sh",
        receipt_ref="data/ak/akc-0001/linkage.json", produced_by="evaluator")
    kwargs.update(overrides)
    return C.LinkageEvidence(**kwargs)


def anti_hack(**overrides) -> C.AntiRewardHackingEvidence:
    kwargs = dict(
        cache_state="cold", correctness_verdict_source="evaluator",
        candidate_output_used_as_oracle=False, oracle_ids=("ik_llama.cpp@iqk-ref",),
        delivered_unit_name="generated_tokens",
        delivered_units_candidate=160, delivered_units_anchor=160,
        anchor_source_commit=V8_COMMIT, anchor_binary_sha256=sha("anchor-binary"),
        anchor_linkage_sha256=sha("anchor-linkage"),
        environment_probe_findings=(), timing_dependent_branch_findings=(),
        receipt_ref="data/ak/akc-0001/integrity.json",
        environment_probe_detector_id="environment-probe/v1",
        timing_dependent_branch_detector_id="timing-branch/v1")
    kwargs.update(overrides)
    return C.AntiRewardHackingEvidence(**kwargs)


def evidence(**overrides) -> C.T0Evidence:
    kwargs = dict(
        control_role=None,
        change_surface=surface(),
        symbols=symbols(),
        build=build(),
        diff=diff(),
        static_analysis=static_analysis(),
        sanitizers=sanitizers(),
        op_suite=op_suite(),
        reference=reference(),
        boundary_shapes=boundary(),
        dispatch_trace=trace(),
        state_safety=state_safety(),
        coherence=coherence(),
        determinism=determinism(),
        linkage=linkage(),
        anti_reward_hacking=anti_hack(),
    )
    kwargs.update(overrides)
    return C.T0Evidence(**kwargs)


def run(req=None, ev=None, pol=None) -> C.T0Report:
    return C.evaluate_t0(req or request(), ev or evidence(), pol or policy())


# ---------------------------------------------------------------------------
# The clean baseline. Every failure test below differs from this by ONE thing.
# ---------------------------------------------------------------------------

class TestCleanCandidatePasses(unittest.TestCase):

    def test_every_declared_gate_is_present_exactly_once(self):
        report = run()
        self.assertEqual(tuple(g.gate_id for g in report.gates), C.T0_GATE_IDS)
        self.assertEqual(len(report.gates), 17)

    def test_all_seventeen_gates_pass(self):
        report = run()
        self.assertEqual(report.failed, ())
        self.assertEqual(report.unevaluated, ())
        for gate in report.gates:
            with self.subTest(gate=gate.gate_id):
                self.assertEqual(gate.check.outcome, S.PASS, gate.check.reasons)

    def test_coherence_is_byte_identical_against_the_named_anchor(self):
        report = run()
        self.assertEqual(report.coherence.label, C.COHERENCE_BYTE_IDENTICAL)
        self.assertTrue(report.coherence.anchor_bound)
        self.assertTrue(report.coherence.asserts_equivalence)

    def test_no_human_review_marker_and_no_release_relevant_property(self):
        report = run()
        self.assertFalse(report.requires_human_code_review)
        self.assertEqual(report.human_review_reasons, ())
        self.assertEqual(report.release_relevant_properties, ())

    def test_every_gate_class_is_speed_blocking(self):
        # If this ever fails, a T0 gate has been filed under a class that does not
        # block ranking, and a failing candidate could be ranked anyway.
        for gate in run().gates:
            with self.subTest(gate=gate.gate_id):
                self.assertIn(gate.gate_class, api.SPEED_BLOCKING_GATE_CLASSES)

    def test_report_is_canonical_json_serialisable(self):
        # The report has to survive into the journal; a payload that cannot be
        # canonicalised cannot be content-hashed and cannot be replayed.
        payload = run().to_dict()
        self.assertTrue(S.content_hash(payload))


# ---------------------------------------------------------------------------
# THE HEADLINE: the op coverage kernel_eval.sh actually had.
# ---------------------------------------------------------------------------

class TestBackendOpUnits(unittest.TestCase):

    def test_mul_mat_only_suite_fails_the_way_kernel_eval_sh_did_not(self):
        """`test-backend-ops -o MUL_MAT`, all cases passing, MUL_MAT_ID never run."""
        report = run(ev=evidence(op_suite=op_suite(
            ops_exercised=("MUL_MAT",),
            cases_by_op=(("MUL_MAT", 4231, 4231),))))
        gate = report.gate(C.GID_OP_UNITS)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("MUL_MAT_ID", " ".join(gate.check.reasons))
        self.assertIn("An untested op is not a passing op", " ".join(gate.check.reasons))

    def test_policy_cannot_be_configured_to_drop_mul_mat_id(self):
        with self.assertRaises(ValueError) as ctx:
            policy(required_backend_ops=("MUL_MAT",))
        self.assertIn("MUL_MAT_ID", str(ctx.exception))

    def test_policy_cannot_drop_mul_mat_either(self):
        with self.assertRaises(ValueError):
            policy(required_backend_ops=("MUL_MAT_ID",))

    def test_derived_surface_ops_join_the_required_set(self):
        report = run(ev=evidence(
            change_surface=surface(derived_ops=("MUL_MAT_ID", "FLASH_ATTN_EXT")),
            reference=reference(undefined_for=(("FLASH_ATTN_EXT", "no bitwise oracle"),))))
        gate = report.gate(C.GID_OP_UNITS)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("FLASH_ATTN_EXT", " ".join(gate.check.reasons))

    def test_exercised_with_zero_cases_is_not_tested(self):
        report = run(ev=evidence(op_suite=op_suite(
            cases_by_op=(("MUL_MAT", 4231, 4231), ("MUL_MAT_ID", 0, 0)))))
        gate = report.gate(C.GID_OP_UNITS)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("zero cases", " ".join(gate.check.reasons))

    def test_exercised_with_no_case_counts_at_all_is_a_name_in_a_list(self):
        report = run(ev=evidence(op_suite=op_suite(
            cases_by_op=(("MUL_MAT", 4231, 4231),))))
        self.assertEqual(report.outcome(C.GID_OP_UNITS), S.FAIL)

    def test_partial_case_pass_fails(self):
        report = run(ev=evidence(op_suite=op_suite(
            cases_by_op=(("MUL_MAT", 4231, 4231), ("MUL_MAT_ID", 1188, 1187)))))
        self.assertEqual(report.outcome(C.GID_OP_UNITS), S.FAIL)

    def test_suite_built_from_a_different_tree_says_nothing(self):
        report = run(ev=evidence(op_suite=op_suite(
            suite_source_sha256=sha("some-other-source"))))
        gate = report.gate(C.GID_OP_UNITS)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("another tree", " ".join(gate.check.reasons))

    def test_candidate_self_reported_suite_is_refused(self):
        report = run(ev=evidence(op_suite=op_suite(produced_by="candidate")))
        gate = report.gate(C.GID_OP_UNITS)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("NEVER self-reported", " ".join(gate.check.reasons))

    def test_missing_suite_is_could_not_check_not_pass(self):
        report = run(ev=evidence(op_suite=None))
        self.assertEqual(report.outcome(C.GID_OP_UNITS), S.COULD_NOT_CHECK)


# ---------------------------------------------------------------------------
# Coherence: a computed verdict, or nothing at all.
# ---------------------------------------------------------------------------

class TestCoherenceIsComputed(unittest.TestCase):

    def test_verdict_cannot_be_constructed_directly(self):
        with self.assertRaises(C.CoherenceTampering):
            C.CoherenceVerdict(
                label=C.COHERENCE_BYTE_IDENTICAL, anchor_bound=True,
                anchor_identity_recorded=True,
                candidate_output_sha256=sha("x"), candidate_output_len=160,
                anchor_output_sha256=sha("x"), sampler_is_greedy=True,
                anchor_determinism_class="bitwise_stable", token_agreement_ratio=1.0,
                tolerance_floor=None, reasons=())

    def test_taking_the_mint_token_buys_nothing(self):
        """The second lock: the label is re-derived from the object's own evidence."""
        token = C._COHERENCE_MINT
        with self.assertRaises(C.CoherenceTampering) as ctx:
            C.CoherenceVerdict(
                label=C.COHERENCE_BYTE_IDENTICAL, anchor_bound=True,
                anchor_identity_recorded=True,
                candidate_output_sha256=sha("a"), candidate_output_len=160,
                anchor_output_sha256=sha("b"), sampler_is_greedy=True,
                anchor_determinism_class="bitwise_stable", token_agreement_ratio=None,
                tolerance_floor=None, reasons=(), mint=token)
        self.assertIn("does not follow from its own evidence", str(ctx.exception))

    def test_equivalence_label_without_an_anchor_raises(self):
        with self.assertRaises(C.CoherenceWithoutAnchor):
            C.CoherenceVerdict(
                label=C.COHERENCE_BYTE_IDENTICAL, anchor_bound=False,
                anchor_identity_recorded=True,
                candidate_output_sha256=sha("x"), candidate_output_len=160,
                anchor_output_sha256=sha("x"), sampler_is_greedy=True,
                anchor_determinism_class="bitwise_stable", token_agreement_ratio=1.0,
                tolerance_floor=None, reasons=(), mint=C._COHERENCE_MINT)

    def test_dataclasses_replace_cannot_relabel_a_verdict(self):
        verdict = C.compute_coherence(anchor=anchor(), evidence=coherence(),
                                      tolerance_floor=None)
        with self.assertRaises(C.CoherenceTampering):
            dataclasses.replace(verdict, label=C.COHERENCE_DIVERGENT)

    def test_no_anchor_yields_not_compared_never_coherent(self):
        verdict = C.compute_coherence(anchor=None, evidence=coherence(), tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_NOT_COMPARED)
        self.assertFalse(verdict.asserts_equivalence)
        self.assertEqual(verdict.to_check().outcome, S.COULD_NOT_CHECK)

    def test_empty_generation_is_a_failure_not_an_ok_status(self):
        """kernel_eval.sh recorded COH="empty-generation" and status OK anyway."""
        verdict = C.compute_coherence(
            anchor=anchor(), evidence=coherence(candidate_output_len=0), tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_EMPTY)
        self.assertEqual(verdict.to_check().outcome, S.FAIL)

    def test_empty_generation_is_detected_even_with_no_anchor(self):
        verdict = C.compute_coherence(
            anchor=None, evidence=coherence(candidate_output_len=0), tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_EMPTY)

    def test_divergent_under_greedy_sampling_fails(self):
        report = run(ev=evidence(coherence=coherence(
            candidate_output_sha256=sha("different"), token_agreement_ratio=0.6)))
        self.assertEqual(report.coherence.label, C.COHERENCE_DIVERGENT)
        self.assertEqual(report.outcome(C.GID_COHERENCE), S.FAIL)

    def test_difference_under_a_sampling_sampler_is_undecidable_not_divergent(self):
        verdict = C.compute_coherence(
            anchor=anchor(),
            evidence=coherence(candidate_output_sha256=sha("different"),
                               sampler_is_greedy=False, sampler_id="temp0.7-topp0.9"),
            tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_UNDECIDABLE)
        self.assertEqual(verdict.to_check().outcome, S.COULD_NOT_CHECK)

    def test_byte_identity_under_a_sampling_sampler_is_also_undecidable(self):
        verdict = C.compute_coherence(
            anchor=anchor(), evidence=coherence(sampler_is_greedy=False),
            tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_UNDECIDABLE)

    def test_unrecorded_sampler_determinism_is_undecidable(self):
        verdict = C.compute_coherence(
            anchor=anchor(), evidence=coherence(sampler_is_greedy=None),
            tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_UNDECIDABLE)

    def test_tolerance_applies_only_when_the_anchor_itself_is_unstable(self):
        stable = C.compute_coherence(
            anchor=anchor(),
            evidence=coherence(candidate_output_sha256=sha("different"),
                               token_agreement_ratio=0.999),
            tolerance_floor=0.995)
        self.assertEqual(stable.label, C.COHERENCE_DIVERGENT)
        unstable = C.compute_coherence(
            anchor=anchor(),
            evidence=coherence(candidate_output_sha256=sha("different"),
                               token_agreement_ratio=0.999,
                               anchor_determinism_class="bitwise_unstable"),
            tolerance_floor=0.995)
        self.assertEqual(unstable.label, C.COHERENCE_WITHIN_TOLERANCE)
        self.assertEqual(unstable.to_check().outcome, S.PASS)

    def test_agreement_below_the_declared_floor_is_divergent(self):
        verdict = C.compute_coherence(
            anchor=anchor(),
            evidence=coherence(candidate_output_sha256=sha("different"),
                               token_agreement_ratio=0.900,
                               anchor_determinism_class="bitwise_unstable"),
            tolerance_floor=0.995)
        self.assertEqual(verdict.label, C.COHERENCE_DIVERGENT)

    def test_no_coherence_evidence_is_could_not_check_with_the_reason_on_the_gate(self):
        report = run(ev=evidence(coherence=None))
        gate = report.gate(C.GID_COHERENCE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("no coherence evidence was captured for this candidate", gate.notes)


# ---------------------------------------------------------------------------
# WHICH anchor produced the anchor output. `CoherenceEvidence` recorded what the
# anchor produced and not whose output it was, so a capture taken against anchor
# A could be re-scored against anchor B and return `byte_identical`. Invariant 11
# makes that replay the DESIGNED path — *"deterministic replay before
# regeneration"* — so it is the path a mismatch would travel on.
# ---------------------------------------------------------------------------

UNRECORDED = {name: None for name in ANCHOR_TRIPLE}


class TestCoherenceRecordsWhichAnchorProducedItsAnchorOutput(unittest.TestCase):

    def test_two_of_three_components_is_rejected(self):
        for omitted in ANCHOR_TRIPLE:
            with self.subTest(omitted=omitted):
                with self.assertRaises(ValueError) as ctx:
                    coherence(**{omitted: None})
                self.assertIn("A partially named anchor is the defect", str(ctx.exception))
                self.assertIn(f"coherence.{omitted}", str(ctx.exception))

    def test_a_placeholder_capture_anchor_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            coherence(anchor_binary_sha256="0" * 64)
        self.assertIn("placeholder digest", str(ctx.exception))

    def test_a_replay_against_a_different_anchor_is_refused_by_name(self):
        with self.assertRaises(C.CoherenceAnchorMismatch) as ctx:
            C.compute_coherence(
                anchor=anchor(binary_sha256=sha("some-other-anchor-binary")),
                evidence=coherence(), tolerance_floor=None)
        message = str(ctx.exception)
        self.assertIn("was taken against anchor", message)
        self.assertIn("anchor.binary_sha256 moved", message)

    def test_a_commit_only_mismatch_is_refused(self):
        """Both digests still agree. This is the mismatch two-of-three cannot see."""
        with self.assertRaises(C.CoherenceAnchorMismatch):
            C.compute_coherence(anchor=anchor(source_commit=V7_COMMIT),
                                evidence=coherence(), tolerance_floor=None)

    def test_a_mismatched_replay_is_never_silently_downgraded(self):
        """The refusal must not be expressible as a quiet `not_compared`."""
        mismatched = coherence(anchor_linkage_sha256=sha("another-anchor-linkage"))
        try:
            verdict = C.compute_coherence(anchor=anchor(), evidence=mismatched,
                                          tolerance_floor=None)
        except C.CoherenceAnchorMismatch:
            return
        self.fail(f"a mismatched replay returned {verdict.label!r} instead of refusing")

    def test_a_mismatched_replay_refuses_the_whole_report(self):
        """The consumer, not only the function: `evaluate_t0` does not absorb it."""
        with self.assertRaises(C.CoherenceAnchorMismatch):
            run(ev=evidence(coherence=coherence(
                anchor_binary_sha256=sha("some-other-anchor-binary"))))

    def test_an_empty_generation_from_a_mismatched_capture_is_still_refused(self):
        """Even the one label that needs no anchor is not minted from another
        anchor's material: the mismatch is a defect in the replay, not a finding
        about this candidate."""
        with self.assertRaises(C.CoherenceAnchorMismatch):
            C.compute_coherence(
                anchor=anchor(),
                evidence=coherence(candidate_output_len=0,
                                   anchor_binary_sha256=sha("some-other-anchor-binary")),
                tolerance_floor=None)

    def test_measurement_event_ids_are_not_part_of_the_identity(self):
        """Identity is the three components — the same rule
        `api.AnchorIdentity.identity_matches` applies to the record."""
        verdict = C.compute_coherence(
            anchor=anchor(measurement_event_ids=("ake-anchor-0002", "ake-anchor-0003")),
            evidence=coherence(), tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_BYTE_IDENTICAL)

    def test_an_unrecorded_capture_anchor_never_reads_as_a_match(self):
        """Evidence that predates the field: identical digests, and still not a
        comparison. Absence of a recorded identity is not agreement."""
        verdict = C.compute_coherence(anchor=anchor(), evidence=coherence(**UNRECORDED),
                                      tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_NOT_COMPARED)
        self.assertFalse(verdict.asserts_equivalence)
        self.assertFalse(verdict.anchor_identity_recorded)
        self.assertEqual(verdict.to_check().outcome, S.COULD_NOT_CHECK)
        self.assertIn("records no anchor identity", " ".join(verdict.reasons))

    def test_an_unrecorded_capture_anchor_is_could_not_check_at_the_gate(self):
        report = run(ev=evidence(coherence=coherence(**UNRECORDED)))
        gate = report.gate(C.GID_COHERENCE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("capture_anchor=unrecorded", gate.notes)

    def test_an_unrecorded_capture_anchor_cannot_be_labelled_equivalent(self):
        """The re-derivation lock covers the new field: minting a verdict that
        claims byte identity while its capture named no anchor is tampering."""
        verdict = C.compute_coherence(anchor=anchor(), evidence=coherence(),
                                      tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_BYTE_IDENTICAL)
        with self.assertRaises(C.CoherenceTampering):
            dataclasses.replace(verdict, anchor_identity_recorded=False)

    def test_an_unrecorded_capture_anchor_raises_nothing(self):
        """COULD_NOT_CHECK-shaped, not a refusal: nothing disagrees, so there is
        no replay bug to surface — only a comparison that cannot be made."""
        C.compute_coherence(anchor=anchor(), evidence=coherence(**UNRECORDED),
                            tolerance_floor=None)

    def test_a_correctly_matched_replay_still_passes(self):
        """The compliant-path counterpart, at both altitudes: the new rule must
        not be satisfiable by refusing everything."""
        verdict = C.compute_coherence(anchor=anchor(), evidence=coherence(),
                                      tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_BYTE_IDENTICAL)
        self.assertTrue(verdict.anchor_identity_recorded)
        self.assertEqual(verdict.to_check().outcome, S.PASS)
        report = run()
        self.assertEqual(report.outcome(C.GID_COHERENCE), S.PASS)
        self.assertIn(f"capture_anchor={anchor().short()}", report.gate(C.GID_COHERENCE).notes)

    def test_an_anchor_less_run_still_needs_no_recorded_identity(self):
        """`anchor=None` is `not_compared` for the ORIGINAL reason, and a capture
        that does name an anchor does not smuggle a comparison in."""
        verdict = C.compute_coherence(anchor=None, evidence=coherence(), tolerance_floor=None)
        self.assertEqual(verdict.label, C.COHERENCE_NOT_COMPARED)
        self.assertIn("no anchor is bound", " ".join(verdict.reasons))

    def test_candidate_self_reported_coherence_cannot_pass(self):
        report = run(ev=evidence(coherence=coherence(produced_by="candidate")))
        self.assertEqual(report.outcome(C.GID_COHERENCE), S.FAIL)


# ---------------------------------------------------------------------------
# The anchor-less case is structurally INVALID.
# ---------------------------------------------------------------------------

class TestAnchorlessIsStructurallyInvalid(unittest.TestCase):

    def setUp(self):
        self.report = run(req=request(anchor=None))

    def test_every_anchor_requiring_gate_is_could_not_check(self):
        for gate in self.report.gates:
            if gate.requires_anchor:
                with self.subTest(gate=gate.gate_id):
                    self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)

    def test_the_five_anchor_requiring_gates_are_the_expected_ones(self):
        got = tuple(g.gate_id for g in self.report.gates if g.requires_anchor)
        self.assertEqual(got, (C.GID_SYMBOLS, C.GID_EXACT_REFERENCE, C.GID_COHERENCE,
                               C.GID_DETERMINISM, C.GID_LINKAGE))

    def test_coherence_is_not_compared_not_coherent(self):
        self.assertEqual(self.report.coherence.label, C.COHERENCE_NOT_COMPARED)
        self.assertFalse(self.report.coherence.asserts_equivalence)

    def test_the_computed_verdict_is_invalid(self):
        verdict = api.compute_verdict(
            tier="T0", gates=self.report.gates,
            void_scan=api.VoidScan(findings=(), evaluated=(), not_applicable=()),
            search_grade=api.SearchGradeResult(True, (), (), (), ()),
            anchor=None, effect=None)
        self.assertEqual(verdict.status, api.STATUS_INVALID)
        with self.assertRaises(api.SpeedRankUnavailable):
            verdict.rank_key()

    def test_the_checks_refuse_at_source_so_nothing_needs_demoting(self):
        # Each anchor-requiring check short-circuits before it can answer PASS, so
        # the second enforcement finds nothing to do. That the list is EMPTY is the
        # assertion: it proves the refusal happened at the check, not downstream.
        self.assertEqual(self.report.demoted_gates, ())
        for gate in self.report.gates:
            if gate.requires_anchor:
                self.assertTrue(any("no anchor is bound" in r for r in gate.check.reasons),
                                gate.gate_id)

    def test_the_second_enforcement_catches_a_check_that_forgot_its_guard(self):
        # The shape a future anchor-requiring check would produce if it omitted its
        # `request.anchor is None` guard.
        forgetful = api.GateResult("t0.output_coherence_vs_anchor", api.GATE_CORRECTNESS,
                                   PASS, requires_anchor=True)
        gates, demoted = C.demote_anchor_requiring_passes((forgetful,), anchor_bound=False)
        self.assertEqual(demoted, ("t0.output_coherence_vs_anchor",))
        self.assertEqual(gates[0].check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("is not a verdict", " ".join(gates[0].check.reasons))
        self.assertIn("PASS demoted to COULD_NOT_CHECK at T0: no anchor bound",
                      gates[0].notes)

    def test_the_second_enforcement_is_a_no_op_when_an_anchor_is_bound(self):
        gate = api.GateResult("t0.output_coherence_vs_anchor", api.GATE_CORRECTNESS,
                              PASS, requires_anchor=True)
        gates, demoted = C.demote_anchor_requiring_passes((gate,), anchor_bound=True)
        self.assertEqual(demoted, ())
        self.assertIs(gates[0], gate)

    def test_a_failing_gate_is_not_demoted_its_signal_is_kept(self):
        failing = api.GateResult("t0.output_coherence_vs_anchor", api.GATE_CORRECTNESS,
                                 S.Check(S.FAIL, ("empty generation",)), requires_anchor=True)
        gates, demoted = C.demote_anchor_requiring_passes((failing,), anchor_bound=False)
        self.assertEqual(demoted, ())
        self.assertEqual(gates[0].check.outcome, S.FAIL)

    def test_anchor_bound_is_recorded_on_the_report(self):
        self.assertFalse(self.report.anchor_bound)
        self.assertTrue(run().anchor_bound)


# ---------------------------------------------------------------------------
# §8.5.1 source-integrity gates
# ---------------------------------------------------------------------------

class TestSourceIntegrityGates(unittest.TestCase):

    def test_undeclared_symbol_removal_is_a_hard_failure(self):
        report = run(ev=evidence(symbols=symbols(
            removed_symbols=("_Z10mmq_case_1Pf", "_Z10mmq_case_2Pf"))))
        gate = report.gate(C.GID_SYMBOLS)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("_Z10mmq_case_1Pf", " ".join(gate.check.reasons))

    def test_declared_symbol_removal_passes(self):
        report = run(ev=evidence(symbols=symbols(
            removed_symbols=("_Z10mmq_case_1Pf",),
            declared_removals=("_Z10mmq_case_1Pf",))))
        self.assertEqual(report.outcome(C.GID_SYMBOLS), S.PASS)

    def test_undeclared_op_registration_removal_fails(self):
        report = run(ev=evidence(symbols=symbols(
            removed_op_registrations=("GGML_OP_MUL_MAT_ID",))))
        self.assertEqual(report.outcome(C.GID_SYMBOLS), S.FAIL)

    def test_undeclared_dispatch_predicate_removal_fails(self):
        report = run(ev=evidence(symbols=symbols(
            removed_dispatch_predicates=("ggml_cuda_should_use_mmq",))))
        self.assertEqual(report.outcome(C.GID_SYMBOLS), S.FAIL)

    def test_undeclared_arity_change_fails(self):
        report = run(ev=evidence(symbols=symbols(
            arity_changed_symbols=("ggml_cuda_mul_mat_id",))))
        self.assertEqual(report.outcome(C.GID_SYMBOLS), S.FAIL)

    def test_symbol_table_shrinkage_above_the_policy_ratio_fails(self):
        report = run(ev=evidence(symbols=symbols(
            anchor_symbol_count=10000, candidate_symbol_count=9000)))
        gate = report.gate(C.GID_SYMBOLS)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("shrank", " ".join(gate.check.reasons))

    def test_incremental_build_fails(self):
        report = run(ev=evidence(build=build(incremental_objects_present=True)))
        gate = report.gate(C.GID_CLEAN_BUILD)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("stale object", " ".join(gate.check.reasons))

    def test_dirty_build_dir_fails(self):
        report = run(ev=evidence(build=build(build_dir_was_fresh=False)))
        self.assertEqual(report.outcome(C.GID_CLEAN_BUILD), S.FAIL)

    def test_snapshot_mismatch_means_the_binary_is_not_the_source(self):
        report = run(ev=evidence(build=build(built_from_snapshot_sha256=sha("other"))))
        gate = report.gate(C.GID_CLEAN_BUILD)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("not the source under test", " ".join(gate.check.reasons))

    def test_a_build_inside_a_production_tree_fails(self):
        report = run(ev=evidence(build=build(
            build_dir="/mnt/raid0/llm/llama.cpp/build",
            production_tree_paths_touched=("/mnt/raid0/llm/llama.cpp/build",))))
        gate = report.gate(C.GID_CLEAN_BUILD)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("FROZEN", " ".join(gate.check.reasons))

    def test_diff_outside_the_declared_surface_fails(self):
        report = run(ev=evidence(diff=diff(
            files_touched=(f"{WORKTREE}/ggml/src/ggml-cuda/mmq.cu",
                           f"{WORKTREE}/src/llama-model.cpp"))))
        gate = report.gate(C.GID_SEMANTIC_DIFF)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("llama-model.cpp", " ".join(gate.check.reasons))

    def test_unrelated_deletions_fail(self):
        report = run(ev=evidence(diff=diff(unrelated_deletions=("ggml_cuda_op_gelu",))))
        self.assertEqual(report.outcome(C.GID_SEMANTIC_DIFF), S.FAIL)

    def test_diff_outside_its_change_class_envelope_fails(self):
        report = run(ev=evidence(diff=diff(changed_lines=900)))
        gate = report.gate(C.GID_SEMANTIC_DIFF)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("one conceptual mutation", " ".join(gate.check.reasons))

    def test_an_envelope_for_another_change_class_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            diff(change_class="fusion")


# ---------------------------------------------------------------------------
# Schema and diff policy, and the §10.6 ceiling
# ---------------------------------------------------------------------------

class TestSchemaAndDiffPolicy(unittest.TestCase):

    def test_production_named_branch_fails(self):
        report = run(ev=evidence(diff=diff(branch_name="production-consolidated-v8")))
        gate = report.gate(C.GID_SCHEMA_DIFF_POLICY)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("experimental", " ".join(gate.check.reasons))

    def test_production_path_in_the_diff_fails(self):
        report = run(ev=evidence(diff=diff(
            production_tree_paths=("/mnt/raid0/llm/whisper.cpp/src/whisper.cpp",))))
        self.assertEqual(report.outcome(C.GID_SCHEMA_DIFF_POLICY), S.FAIL)

    def test_non_pathspec_limited_commit_fails(self):
        report = run(ev=evidence(diff=diff(commit_was_pathspec_limited=False)))
        gate = report.gate(C.GID_SCHEMA_DIFF_POLICY)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("staged files", " ".join(gate.check.reasons))

    def test_record_schema_violations_fail(self):
        report = run(ev=evidence(diff=diff(
            record_schema_violations=("candidate.branch: names a production branch",))))
        self.assertEqual(report.outcome(C.GID_SCHEMA_DIFF_POLICY), S.FAIL)

    def test_ceiling_breach_marks_for_review_without_failing_the_gate(self):
        report = run(ev=evidence(diff=diff(
            changed_lines=900, envelope=envelope(max_changed_lines=1000))))
        self.assertEqual(report.outcome(C.GID_SCHEMA_DIFF_POLICY), S.PASS)
        self.assertTrue(report.requires_human_code_review)
        self.assertIn("§10.6", " ".join(report.human_review_reasons))

    def test_core_header_change_class_always_marks_for_review(self):
        report = run(ev=evidence(diff=diff(
            change_class="core_header", envelope=envelope(change_class="core_header"))))
        self.assertTrue(report.requires_human_code_review)

    def test_shared_core_header_surface_marks_for_review(self):
        report = run(ev=evidence(change_surface=surface(touches_shared_core_header=True)))
        self.assertTrue(report.requires_human_code_review)

    def test_the_review_marker_survives_onto_the_gate_notes(self):
        # api.TierGateRunner hands back only gate results, so a marker that lives
        # only on the report never reaches the record. See correctness.SEAMS[0].
        report = run(ev=evidence(change_surface=surface(touches_shared_core_header=True)))
        notes = " ".join(report.gate(C.GID_SCHEMA_DIFF_POLICY).notes)
        self.assertIn(C.REQUIRES_HUMAN_CODE_REVIEW, notes)

    def test_marker_and_reasons_cannot_disagree(self):
        with self.assertRaises(ValueError):
            C.T0Report(
                event_id="ake-t0-0001", candidate_id="akc-0001", tier="T0",
                gates=run().gates, coherence=run().coherence,
                requires_human_code_review=True, human_review_reasons=(),
                release_relevant_properties=(), actor_prediction_score=(),
                anchor_bound=True, demoted_gates=(), policy_ref="x")


# ---------------------------------------------------------------------------
# Static / compile
# ---------------------------------------------------------------------------

class TestStaticAndCompile(unittest.TestCase):

    def test_compiler_errors_fail(self):
        report = run(ev=evidence(static_analysis=static_analysis(error_count=3)))
        self.assertEqual(report.outcome(C.GID_STATIC_COMPILE), S.FAIL)

    def test_analyzer_error_findings_fail(self):
        report = run(ev=evidence(static_analysis=static_analysis(
            analyzer_error_findings=("mmq.cu:412: use of uninitialised value",))))
        self.assertEqual(report.outcome(C.GID_STATIC_COMPILE), S.FAIL)

    def test_a_different_compiler_than_the_anchor_is_a_confound(self):
        report = run(ev=evidence(static_analysis=static_analysis(
            compiler_version="6.3.0")))
        gate = report.gate(C.GID_STATIC_COMPILE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("toolchain comparison", " ".join(gate.check.reasons))

    def test_new_warnings_versus_the_anchor_fail(self):
        report = run(ev=evidence(static_analysis=static_analysis(
            warnings_as_errors=False, warning_count=7, anchor_warning_count=2)))
        gate = report.gate(C.GID_STATIC_COMPILE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("5 new", " ".join(gate.check.reasons))

    def test_no_werror_and_no_anchor_baseline_is_could_not_check(self):
        report = run(ev=evidence(static_analysis=static_analysis(
            warnings_as_errors=False, anchor_warning_count=None)))
        self.assertEqual(report.outcome(C.GID_STATIC_COMPILE), S.COULD_NOT_CHECK)

    def test_a_real_failure_is_not_downgraded_by_an_unknown(self):
        report = run(ev=evidence(static_analysis=static_analysis(
            error_count=1, warnings_as_errors=False, anchor_warning_count=None)))
        self.assertEqual(report.outcome(C.GID_STATIC_COMPILE), S.FAIL)


class TestStaticAnalysisNamesTheAnchorItsToolchainCameFrom(unittest.TestCase):
    """`anchor_compiler_id`, `anchor_compiler_version` and `anchor_warning_count`
    are the ANCHOR's build, and the capture named no anchor — so the confound this
    gate exists to catch could arrive through the gate itself, as a toolchain
    comparison against a toolchain belonging to some other anchor."""

    def test_two_of_three_components_is_rejected(self):
        for omitted in ANCHOR_TRIPLE:
            with self.subTest(omitted=omitted):
                with self.assertRaises(ValueError) as ctx:
                    static_analysis(**{omitted: None})
                self.assertIn("A partially named anchor is the defect", str(ctx.exception))
                self.assertIn(f"static.{omitted}", str(ctx.exception))

    def test_a_placeholder_capture_anchor_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            static_analysis(anchor_binary_sha256="0" * 64)
        self.assertIn("placeholder digest", str(ctx.exception))

    def test_a_capture_from_another_anchor_is_refused_by_name(self):
        with self.assertRaises(C.StaticAnalysisAnchorMismatch) as ctx:
            run(ev=evidence(static_analysis=static_analysis(
                anchor_binary_sha256=sha("some-other-anchor-binary"))))
        self.assertIn("this static-analysis capture was taken against anchor",
                      str(ctx.exception))

    def test_a_commit_only_mismatch_is_refused(self):
        """Both anchor digests still agree; only the commit moved."""
        with self.assertRaises(C.StaticAnalysisAnchorMismatch):
            run(ev=evidence(static_analysis=static_analysis(
                anchor_source_commit=V7_COMMIT)))

    def test_an_identical_toolchain_does_not_excuse_the_mismatch(self):
        """The compilers agree, so every reason text this gate can emit is silent.
        The refusal is about WHOSE build they were read from, not what they say."""
        with self.assertRaises(C.StaticAnalysisAnchorMismatch):
            run(ev=evidence(static_analysis=static_analysis(
                compiler_id="hipcc", compiler_version="6.2.0",
                anchor_compiler_id="hipcc", anchor_compiler_version="6.2.0",
                anchor_linkage_sha256=sha("another-anchor-linkage"))))

    def test_an_unrecorded_capture_anchor_is_could_not_check(self):
        report = run(ev=evidence(static_analysis=static_analysis(**UNRECORDED)))
        gate = report.gate(C.GID_STATIC_COMPILE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("records no anchor identity", " ".join(gate.check.reasons))
        self.assertIn("capture_anchor=unrecorded", gate.notes)

    def test_an_unrecorded_capture_anchor_raises_nothing(self):
        """COULD_NOT_CHECK-shaped, not a refusal: nothing disagrees."""
        run(ev=evidence(static_analysis=static_analysis(**UNRECORDED)))

    def test_an_unrecorded_capture_anchor_does_not_downgrade_a_real_failure(self):
        report = run(ev=evidence(static_analysis=static_analysis(
            error_count=1, **UNRECORDED)))
        self.assertEqual(report.outcome(C.GID_STATIC_COMPILE), S.FAIL)

    def test_the_compliant_capture_still_passes(self):
        """The counterpart: the new rule must not be satisfiable by always failing."""
        report = run()
        gate = report.gate(C.GID_STATIC_COMPILE)
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertIn(f"capture_anchor={anchor().short()}", gate.notes)


# ---------------------------------------------------------------------------
# ASAN / UBSAN — the invocation is built, never run
# ---------------------------------------------------------------------------

class TestSanitizers(unittest.TestCase):

    def test_the_constructed_invocation_passes_its_own_check(self):
        self.assertEqual(C.check_sanitizer_invocation(invocation()).outcome, S.PASS)

    def test_the_invocation_carries_a_recipe_receipt_with_real_hashes(self):
        receipt = invocation().receipt
        self.assertIsInstance(receipt, api.RecipeReceipt)
        self.assertRegex(receipt.constructor_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(receipt.argv_sha256, r"^[0-9a-f]{64}$")
        self.assertIn("@", receipt.render())

    def test_the_invocation_is_deterministic(self):
        self.assertEqual(invocation().receipt.argv_sha256, invocation().receipt.argv_sha256)

    def test_argv_is_never_a_shell_string(self):
        inv = invocation()
        for argv in (inv.configure_argv, inv.build_argv, inv.run_argv):
            self.assertIsInstance(argv, tuple)
            for item in argv:
                self.assertIsInstance(item, str)

    def test_core_dumps_are_disabled(self):
        self.assertIn("disable_coredump=1", invocation().env_value("ASAN_OPTIONS"))

    def test_an_invocation_that_would_dump_core_fails_the_check(self):
        inv = invocation()
        broken = C.SanitizerInvocation(
            constructor_id=inv.constructor_id, configure_argv=inv.configure_argv,
            build_argv=inv.build_argv, run_argv=inv.run_argv,
            env=(("ASAN_OPTIONS", "abort_on_error=1:disable_coredump=0:detect_leaks=1:"
                                  "print_stacktrace=1"),
                 ("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")),
            receipt=inv.receipt, notes=())
        check = C.check_sanitizer_invocation(broken)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("NEVER core dumps", " ".join(check.reasons))

    def test_ulimit_in_argv_fails_the_check(self):
        inv = invocation()
        broken = C.SanitizerInvocation(
            constructor_id=inv.constructor_id, configure_argv=inv.configure_argv,
            build_argv=inv.build_argv,
            run_argv=("sh", "-c", "ulimit -c unlimited; ./test-backend-ops"),
            env=inv.env, receipt=inv.receipt, notes=())
        self.assertEqual(C.check_sanitizer_invocation(broken).outcome, S.FAIL)

    def test_a_recovering_sanitizer_is_fail_open_and_is_refused(self):
        inv = invocation()
        broken = C.SanitizerInvocation(
            constructor_id=inv.constructor_id,
            configure_argv=("cmake", "-S", "/x", "-B", "/y",
                            "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined "
                            "-fno-omit-frame-pointer -g"),
            build_argv=inv.build_argv, run_argv=inv.run_argv, env=inv.env,
            receipt=inv.receipt, notes=())
        check = C.check_sanitizer_invocation(broken)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("-fno-sanitize-recover=all", " ".join(check.reasons))

    def test_a_compliant_guard_does_not_forbid_its_own_idiom(self):
        # `disable_coredump=1` contains the substring `coredump=1`; a naive token
        # list would fail the compliant invocation. This is that regression test.
        self.assertNotIn("coredump=1", C._CORE_DUMP_TOKENS)
        self.assertEqual(C.check_sanitizer_invocation(invocation()).outcome, S.PASS)

    def test_a_sanitizer_build_inside_a_production_tree_is_refused(self):
        with self.assertRaises(C.CorrectnessError):
            invocation(build_dir="/mnt/raid0/llm/llama.cpp/build-asan")

    def test_a_build_with_no_targeted_run_is_refused(self):
        with self.assertRaises(ValueError):
            invocation(run_argv=())

    def test_memory_change_without_sanitizers_fails_both_gates(self):
        report = run(ev=evidence(sanitizers=None))
        for gate_id in (C.GID_ASAN, C.GID_UBSAN):
            with self.subTest(gate=gate_id):
                gate = report.gate(gate_id)
                self.assertEqual(gate.check.outcome, S.FAIL)
                self.assertIn("MANDATORY", " ".join(gate.check.reasons))

    def test_undetermined_memory_surface_is_could_not_check_not_pass(self):
        report = run(ev=evidence(
            sanitizers=None,
            change_surface=surface(derived_touches_memory=None,
                                   derived_touches_threading=False)))
        self.assertEqual(report.outcome(C.GID_ASAN), S.COULD_NOT_CHECK)
        self.assertEqual(report.outcome(C.GID_UBSAN), S.COULD_NOT_CHECK)

    def test_not_mandatory_when_the_derivation_says_neither_surface_is_touched(self):
        report = run(ev=evidence(
            sanitizers=None,
            change_surface=surface(derived_touches_memory=False,
                                   derived_touches_threading=False,
                                   declared_touches_memory=False,
                                   declared_touches_threading=False)))
        self.assertEqual(report.outcome(C.GID_ASAN), S.PASS)
        self.assertEqual(report.outcome(C.GID_UBSAN), S.PASS)

    def test_asan_findings_fail_the_asan_gate_only(self):
        report = run(ev=evidence(sanitizers=sanitizers(
            asan_findings=("heap-buffer-overflow in mmq_id_tile",))))
        self.assertEqual(report.outcome(C.GID_ASAN), S.FAIL)
        self.assertEqual(report.outcome(C.GID_UBSAN), S.PASS)

    def test_ubsan_findings_fail_the_ubsan_gate_only(self):
        report = run(ev=evidence(sanitizers=sanitizers(
            ubsan_findings=("signed integer overflow in row_stride",))))
        self.assertEqual(report.outcome(C.GID_UBSAN), S.FAIL)
        self.assertEqual(report.outcome(C.GID_ASAN), S.PASS)

    def test_a_built_but_never_run_sanitizer_proves_nothing(self):
        report = run(ev=evidence(sanitizers=sanitizers(executed=False)))
        self.assertEqual(report.outcome(C.GID_ASAN), S.FAIL)

    def test_the_measured_binary_may_not_be_the_sanitizer_build(self):
        report = run(ev=evidence(sanitizers=sanitizers(
            sanitizer_build_binary_sha256=sha("cand-binary"))))
        gate = report.gate(C.GID_ASAN)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("not a performance artifact", " ".join(gate.check.reasons))

    def test_unexplained_nonzero_exit_is_not_a_pass(self):
        report = run(ev=evidence(sanitizers=sanitizers(exit_code=1)))
        self.assertEqual(report.outcome(C.GID_ASAN), S.FAIL)


# ---------------------------------------------------------------------------
# Reference comparisons, boundary shapes, dispatch trace
# ---------------------------------------------------------------------------

class TestReferenceAndDispatchSurfaces(unittest.TestCase):

    def test_exact_reference_mismatch_fails(self):
        broken = C.ReferenceComparison(
            shape_id="m4096n1k4096-q4_K", op="MUL_MAT", mode="exact_bitwise",
            mismatch_count=17, max_ulp_observed=None, tolerance_ulp=None,
            oracle_id="ik_llama.cpp@iqk-ref", oracle_is_candidate_derived=False)
        report = run(ev=evidence(reference=reference(
            comparisons=(broken,) + reference().comparisons[1:])))
        self.assertEqual(report.outcome(C.GID_EXACT_REFERENCE), S.FAIL)

    def test_ulp_beyond_the_declared_tolerance_fails(self):
        loose = C.ReferenceComparison(
            shape_id="e128t1k4096-q4_K", op="MUL_MAT_ID", mode="ulp_bounded",
            mismatch_count=0, max_ulp_observed=9.0, tolerance_ulp=2.0,
            oracle_id="ik_llama.cpp@iqk-ref", oracle_is_candidate_derived=False)
        report = run(ev=evidence(reference=reference(
            comparisons=reference().comparisons[:1] + (loose,))))
        self.assertEqual(report.outcome(C.GID_EXACT_REFERENCE), S.FAIL)

    def test_a_candidate_derived_oracle_is_refused(self):
        cheating = C.ReferenceComparison(
            shape_id="m4096n1k4096-q4_K", op="MUL_MAT", mode="exact_bitwise",
            mismatch_count=0, max_ulp_observed=None, tolerance_ulp=None,
            oracle_id="akc-0001-cached-output", oracle_is_candidate_derived=True)
        report = run(ev=evidence(reference=reference(
            comparisons=(cheating,) + reference().comparisons[1:])))
        gate = report.gate(C.GID_EXACT_REFERENCE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("MUST NEVER be cached or reused as a correctness oracle",
                      " ".join(gate.check.reasons))

    def test_an_op_neither_compared_nor_declared_undefined_fails(self):
        report = run(ev=evidence(reference=reference(
            comparisons=reference().comparisons[:1])))
        gate = report.gate(C.GID_EXACT_REFERENCE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("silence is not a reference", " ".join(gate.check.reasons))

    def test_declaring_an_op_undefined_is_accepted(self):
        report = run(ev=evidence(reference=reference(
            comparisons=reference().comparisons[:1],
            undefined_for=(("MUL_MAT_ID", "no bitwise oracle for expert routing"),))))
        self.assertEqual(report.outcome(C.GID_EXACT_REFERENCE), S.PASS)

    def test_a_dispatch_change_with_no_unseen_shapes_fails(self):
        report = run(ev=evidence(boundary_shapes=None))
        gate = report.gate(C.GID_BOUNDARY_SHAPES)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("overfit", " ".join(gate.check.reasons))

    def test_a_holdout_the_planner_saw_is_not_a_holdout(self):
        report = run(ev=evidence(boundary_shapes=boundary(held_out_from_planner=False)))
        gate = report.gate(C.GID_BOUNDARY_SHAPES)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("special-cases it", " ".join(gate.check.reasons))

    def test_boundary_shape_failures_fail(self):
        report = run(ev=evidence(boundary_shapes=boundary(failures=("m0n0k0: assert",))))
        self.assertEqual(report.outcome(C.GID_BOUNDARY_SHAPES), S.FAIL)

    def test_undetermined_dispatch_surface_is_could_not_check(self):
        report = run(ev=evidence(
            boundary_shapes=None,
            change_surface=surface(derived_touches_dispatch=None)))
        self.assertEqual(report.outcome(C.GID_BOUNDARY_SHAPES), S.COULD_NOT_CHECK)

    def test_traced_outside_derived_is_a_hard_failure(self):
        report = run(ev=evidence(dispatch_trace=trace(
            traced_kernels=("MUL_MAT_ID", "mmq_id_tile", "ggml_cuda_op_soft_max"))))
        gate = report.gate(C.GID_SURFACE_RECONCILIATION)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("traced ⊄ derived", " ".join(gate.check.reasons))

    def test_an_empty_trace_reconciles_with_nothing(self):
        report = run(ev=evidence(dispatch_trace=trace(traced_kernels=())))
        self.assertEqual(report.outcome(C.GID_SURFACE_RECONCILIATION), S.FAIL)

    def test_a_fallback_event_fails_and_blocks_the_speed_rank(self):
        report = run(ev=evidence(dispatch_trace=trace(
            fallback_events=("mmq_id_tile -> generic mul_mat_id at n=1",))))
        gate = report.gate(C.GID_NO_FALLBACK)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("no speed rank at all", " ".join(gate.check.reasons))
        self.assertIn(gate.gate_class, api.SPEED_BLOCKING_GATE_CLASSES)

    def test_no_fallback_gate_is_integrity_not_mechanism(self):
        # `mechanism` is NOT speed-blocking; filing it there would let a
        # silently-falling-back candidate be ranked.
        self.assertEqual(run().gate(C.GID_NO_FALLBACK).gate_class, api.GATE_INTEGRITY)
        self.assertNotIn(api.GATE_MECHANISM, api.SPEED_BLOCKING_GATE_CLASSES)

    def test_an_uninstrumented_trace_cannot_prove_no_fallback(self):
        report = run(ev=evidence(dispatch_trace=trace(
            fallback_instrumentation_active=False)))
        gate = report.gate(C.GID_NO_FALLBACK)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("evidence of no instrumentation", " ".join(gate.check.reasons))

    def test_the_actor_prediction_is_scored_and_never_gates(self):
        report = run(ev=evidence(change_surface=surface(
            declared_ops=(), declared_touches_threading=False)))
        # The declaration was wrong on two counts and the surface gate still passes:
        # the derivation is the scope input, not the declaration.
        self.assertEqual(report.outcome(C.GID_SURFACE_RECONCILIATION), S.PASS)
        rows = dict((row[0], row[3]) for row in report.actor_prediction_score)
        self.assertFalse(rows["touches_threading"])
        self.assertFalse(rows["ops_missed_by_actor"])


# ---------------------------------------------------------------------------
# State / rollback / teardown / race
# ---------------------------------------------------------------------------

class TestStateSafety(unittest.TestCase):

    def test_race_findings_fail(self):
        report = run(ev=evidence(state_safety=state_safety(
            race_findings=("data race on ggml_backend_sched.splits",))))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.FAIL)

    def test_leaked_resources_fail(self):
        report = run(ev=evidence(state_safety=state_safety(
            leaked_resources=("hipStream_t 0x7f2a",))))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.FAIL)

    def test_orphan_processes_fail(self):
        report = run(ev=evidence(state_safety=state_safety(
            orphan_processes=("pid 88213 test-backend-ops",))))
        gate = report.gate(C.GID_STATE_SAFETY)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("invariant 10", " ".join(gate.check.reasons))

    def test_threading_change_without_a_race_detector_fails(self):
        report = run(ev=evidence(state_safety=state_safety(race_detector_id=None)))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.FAIL)

    def test_missing_evidence_on_a_relevant_surface_fails(self):
        report = run(ev=evidence(state_safety=None))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.FAIL)

    def test_derivation_declared_not_applicable_passes(self):
        # The n/a note and the derivation must agree. This test used to attach
        # "no persistent state and no threading in the diff" to the DEFAULT
        # surface, whose derived_touches_persistent_state and
        # derived_touches_threading are both True — it asserted that a
        # `static_derivation` waiver beats the same derivation's own flags.
        na = C.NotApplicable(reason="no persistent state and no threading in the diff",
                             source="static_derivation", ref="ake-derivation-0001")
        report = run(ev=evidence(
            state_safety=na,
            change_surface=surface(derived_touches_persistent_state=False,
                                   derived_touches_threading=False,
                                   declared_touches_threading=False)))
        gate = report.gate(C.GID_STATE_SAFETY)
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertIn("static_derivation", " ".join(gate.notes))

    def test_the_actor_may_not_declare_a_surface_not_applicable(self):
        with self.assertRaises(C.ActorDeclaredScope) as ctx:
            C.NotApplicable(reason="I looked and it is fine", source="actor", ref="x")
        self.assertIn("invariant 18", str(ctx.exception))


# ---------------------------------------------------------------------------
# Determinism class — invariant 12
# ---------------------------------------------------------------------------

class TestDeterminismClass(unittest.TestCase):

    def test_an_undeclared_class_change_fails(self):
        digests = (sha("a"), sha("b"), sha("c"))
        report = run(req=request(determinism=api.DeterminismReport(
                         determinism_class="bitwise_unstable", same_seed_repeat_runs=3)),
                     ev=evidence(determinism=determinism(
                         candidate_output_digests=digests)))
        gate = report.gate(C.GID_DETERMINISM)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("invariant 12", " ".join(gate.check.reasons))

    def test_a_declared_class_change_passes_and_becomes_a_release_relevant_property(self):
        digests = (sha("a"), sha("b"), sha("c"))
        report = run(req=request(determinism=api.DeterminismReport(
                         determinism_class="bitwise_unstable", same_seed_repeat_runs=3)),
                     ev=evidence(determinism=determinism(
                         candidate_output_digests=digests,
                         declared_class_change=True,
                         declared_class_change_ref="akp-0001#determinism")))
        self.assertEqual(report.outcome(C.GID_DETERMINISM), S.PASS)
        self.assertEqual(len(report.release_relevant_properties), 1)
        self.assertIn("RELEASE_RELEVANT_PROPERTY",
                      " ".join(report.gate(C.GID_DETERMINISM).notes))

    def test_a_declared_change_must_name_where_it_was_declared(self):
        with self.assertRaises(ValueError):
            determinism(declared_class_change=True, declared_class_change_ref=None)

    def test_the_evaluator_overrides_the_records_self_report(self):
        digests = (sha("a"), sha("b"), sha("c"))
        report = run(ev=evidence(determinism=determinism(
            candidate_output_digests=digests, declared_class_change=True,
            declared_class_change_ref="akp-0001#determinism")))
        gate = report.gate(C.GID_DETERMINISM)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("never self-reported", " ".join(gate.check.reasons))

    def test_too_few_repeats_is_could_not_check(self):
        report = run(req=request(determinism=api.DeterminismReport(
                         determinism_class="bitwise_stable", same_seed_repeat_runs=2)),
                     ev=evidence(determinism=determinism(
                         runs=2, candidate_output_digests=(sha("gen-out"),) * 2,
                         anchor_output_digests=(sha("gen-out"),) * 2)))
        self.assertEqual(report.outcome(C.GID_DETERMINISM), S.COULD_NOT_CHECK)

    def test_digest_count_must_match_the_declared_run_count(self):
        report = run(ev=evidence(determinism=determinism(
            candidate_output_digests=(sha("gen-out"),) * 2)))
        self.assertEqual(report.outcome(C.GID_DETERMINISM), S.FAIL)

    def test_a_self_contradicting_anchor_invalidates_the_comparison(self):
        report = run(ev=evidence(determinism=determinism(
            anchor_output_digests=(sha("a"), sha("b"), sha("c")))))
        gate = report.gate(C.GID_DETERMINISM)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("measured behaviour disagree", " ".join(gate.check.reasons))

    def test_not_measured_is_sayable(self):
        ev = determinism(runs=1, candidate_output_digests=(sha("gen-out"),),
                         anchor_output_digests=(sha("gen-out"),))
        self.assertEqual(ev.measured_class(), "not_measured")


class TestDeterminismNamesTheAnchorItWasCapturedAgainst(unittest.TestCase):
    """`anchor_output_digests` and `anchor_determinism_class` are what SOME anchor
    did. Invariant 12 makes the class an interface, so a class comparison against
    another anchor's interface can report a change that never happened or miss one
    that did — and the capture recorded which anchor it ran against nowhere."""

    def test_two_of_three_components_is_rejected(self):
        for omitted in ANCHOR_TRIPLE:
            with self.subTest(omitted=omitted):
                with self.assertRaises(ValueError) as ctx:
                    determinism(**{omitted: None})
                self.assertIn("A partially named anchor is the defect", str(ctx.exception))
                self.assertIn(f"determinism.{omitted}", str(ctx.exception))

    def test_a_placeholder_capture_anchor_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            determinism(anchor_linkage_sha256="f" * 64)
        self.assertIn("placeholder digest", str(ctx.exception))

    def test_a_capture_from_another_anchor_is_refused_by_name(self):
        with self.assertRaises(C.DeterminismAnchorMismatch) as ctx:
            run(ev=evidence(determinism=determinism(
                anchor_binary_sha256=sha("some-other-anchor-binary"))))
        message = str(ctx.exception)
        self.assertIn("this determinism capture was taken against anchor", message)
        self.assertIn("anchor.binary_sha256 moved", message)

    def test_a_commit_only_mismatch_is_refused(self):
        """Both anchor digests still agree. This is the mismatch two-of-three
        components could not see."""
        with self.assertRaises(C.DeterminismAnchorMismatch):
            run(ev=evidence(determinism=determinism(anchor_source_commit=V7_COMMIT)))

    def test_an_agreeing_class_does_not_excuse_the_mismatch(self):
        """The anchor class matches the candidate's measured one, so this gate has
        nothing to report — and still refuses. Agreement with another anchor's
        class is not agreement with this anchor's."""
        with self.assertRaises(C.DeterminismAnchorMismatch):
            run(ev=evidence(determinism=determinism(
                anchor_determinism_class="bitwise_stable",
                anchor_linkage_sha256=sha("another-anchor-linkage"))))

    def test_a_mismatched_capture_is_never_silently_downgraded(self):
        try:
            report = run(ev=evidence(determinism=determinism(
                anchor_binary_sha256=sha("some-other-anchor-binary"))))
        except C.DeterminismAnchorMismatch:
            return
        self.fail(f"a mismatched determinism capture produced "
                  f"{report.outcome(C.GID_DETERMINISM)!r} instead of refusing")

    def test_an_unrecorded_capture_anchor_is_could_not_check(self):
        report = run(ev=evidence(determinism=determinism(**UNRECORDED)))
        gate = report.gate(C.GID_DETERMINISM)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("records no anchor identity", " ".join(gate.check.reasons))
        self.assertIn("capture_anchor=unrecorded", gate.notes)

    def test_an_unrecorded_capture_anchor_raises_nothing(self):
        run(ev=evidence(determinism=determinism(**UNRECORDED)))

    def test_an_unrecorded_capture_anchor_does_not_downgrade_a_real_failure(self):
        report = run(ev=evidence(determinism=determinism(
            candidate_output_digests=(sha("gen-out"),) * 2, **UNRECORDED)))
        self.assertEqual(report.outcome(C.GID_DETERMINISM), S.FAIL)

    def test_an_unrecorded_capture_anchor_still_reports_an_undeclared_change(self):
        """The invariant-12 FAIL survives an unattributable anchor: an unknown
        never buys silence about a class change the evaluator measured."""
        report = run(req=request(determinism=api.DeterminismReport(
                         determinism_class="bitwise_unstable", same_seed_repeat_runs=3)),
                     ev=evidence(determinism=determinism(
                         candidate_output_digests=(sha("a"), sha("b"), sha("c")),
                         **UNRECORDED)))
        gate = report.gate(C.GID_DETERMINISM)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("invariant 12", " ".join(gate.check.reasons))

    def test_the_compliant_matched_capture_still_passes(self):
        """The counterpart: the new rule must not be satisfiable by always failing."""
        report = run()
        gate = report.gate(C.GID_DETERMINISM)
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertIn(f"capture_anchor={anchor().short()}", gate.notes)

    def test_measurement_event_ids_are_not_part_of_the_identity(self):
        report = run(req=request(anchor=anchor(
            measurement_event_ids=("ake-anchor-0002", "ake-anchor-0003"))))
        self.assertEqual(report.outcome(C.GID_DETERMINISM), S.PASS)


# ---------------------------------------------------------------------------
# Binary / linkage identity
# ---------------------------------------------------------------------------

class TestBinaryAndLinkageIdentity(unittest.TestCase):

    def test_a_candidate_byte_identical_to_the_anchor_has_nothing_to_rank(self):
        report = run(ev=evidence(linkage=linkage(
            binary_sha256=sha("anchor-binary")),
            build=build(output_binary_sha256=sha("anchor-binary"))),
            req=request(artifact=api.ArtifactIdentity(
                source_sha256=sha("cand-source"),
                binary_sha256=sha("anchor-binary"),
                linkage_sha256=sha("cand-linkage"))))
        gate = report.gate(C.GID_LINKAGE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("nothing to rank", " ".join(gate.check.reasons))

    def test_the_aa_control_REQUIRES_the_identical_binary(self):
        report = run(ev=evidence(
            control_role="aa",
            linkage=linkage(binary_sha256=sha("anchor-binary")),
            build=build(output_binary_sha256=sha("anchor-binary"))),
            req=request(artifact=api.ArtifactIdentity(
                source_sha256=sha("cand-source"),
                binary_sha256=sha("anchor-binary"),
                linkage_sha256=sha("cand-linkage"))))
        self.assertEqual(report.outcome(C.GID_LINKAGE), S.PASS)

    def test_the_aa_control_with_a_different_binary_calibrates_nothing(self):
        report = run(ev=evidence(control_role="aa"))
        gate = report.gate(C.GID_LINKAGE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("calibrates nothing", " ".join(gate.check.reasons))

    def test_a_library_from_another_ggml_tree_fails(self):
        report = run(ev=evidence(linkage=linkage(resolved_libraries=(
            ("libggml-base.so", "/mnt/raid0/llm/whisper.cpp/build/bin/libggml-base.so",
             sha("wrong-ggml")),))))
        gate = report.gate(C.GID_LINKAGE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("SILENTLY wrong", " ".join(gate.check.reasons))

    def test_a_drifted_anchor_binary_fails(self):
        report = run(ev=evidence(linkage=linkage(anchor_binary_sha256=sha("rebuilt-anchor"))))
        gate = report.gate(C.GID_LINKAGE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("a rebuilt anchor is a different anchor", " ".join(gate.check.reasons))

    def test_verifying_a_different_binary_than_the_record_names_fails(self):
        report = run(ev=evidence(linkage=linkage(binary_sha256=sha("some-other-binary"))))
        self.assertEqual(report.outcome(C.GID_LINKAGE), S.FAIL)

    def test_uncaptured_anchor_digests_are_could_not_check(self):
        report = run(ev=evidence(linkage=linkage(
            anchor_source_commit=None, anchor_binary_sha256=None,
            anchor_linkage_sha256=None)))
        self.assertEqual(report.outcome(C.GID_LINKAGE), S.COULD_NOT_CHECK)


class TestLinkageNamesTheAnchorByAllThreeComponents(unittest.TestCase):
    """Precondition 4: *"names its anchor by source commit, binary SHA-256, and
    linkage SHA-256"*. `LinkageEvidence` carried two of the three, so the gate
    whose subject is *"a rebuilt anchor is a different anchor"* was missing the
    component that says what the anchor was rebuilt FROM — while the
    `evaluation_event.v3` record it feeds requires all three."""

    def test_two_of_three_components_is_rejected(self):
        for omitted in ANCHOR_TRIPLE:
            with self.subTest(omitted=omitted):
                with self.assertRaises(ValueError) as ctx:
                    linkage(**{omitted: None})
                self.assertIn("A partially named anchor is the defect", str(ctx.exception))
                self.assertIn(f"linkage.{omitted}", str(ctx.exception))

    def test_one_of_three_components_is_rejected(self):
        for kept in ANCHOR_TRIPLE:
            with self.subTest(kept=kept):
                dropped = {name: None for name in ANCHOR_TRIPLE if name != kept}
                with self.assertRaises(ValueError):
                    linkage(**dropped)

    def test_all_three_absent_is_constructible_and_reads_as_unverified(self):
        """Absent is sayable; partially named is not. The two are different states."""
        ev = linkage(**{name: None for name in ANCHOR_TRIPLE})
        self.assertIsNone(ev.anchor_source_commit)
        report = run(ev=evidence(linkage=ev))
        gate = report.gate(C.GID_LINKAGE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("source commit", " ".join(gate.check.reasons))

    def test_a_drifted_anchor_commit_fails(self):
        """Both digests agree; only the commit moved. Two-of-three could not see it."""
        report = run(ev=evidence(linkage=linkage(anchor_source_commit=V7_COMMIT)))
        gate = report.gate(C.GID_LINKAGE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("anchor source commit verified", " ".join(gate.check.reasons))

    def test_a_placeholder_anchor_commit_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            linkage(anchor_source_commit="0" * 40)
        self.assertIn("placeholder digest", str(ctx.exception))

    def test_a_placeholder_anchor_digest_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            linkage(anchor_binary_sha256="f" * 64)
        self.assertIn("placeholder digest", str(ctx.exception))

    def test_a_short_commit_is_not_a_commit(self):
        with self.assertRaises(ValueError) as ctx:
            linkage(anchor_source_commit=V8_COMMIT[:12])
        self.assertIn("40-hex git commit", str(ctx.exception))

    def test_the_compliant_three_component_anchor_still_passes(self):
        """The counterpart: the new rule must not pass by always failing."""
        report = run()
        self.assertEqual(report.outcome(C.GID_LINKAGE), S.PASS)


# ---------------------------------------------------------------------------
# Anti-reward-hacking — control 3's detector
# ---------------------------------------------------------------------------

class TestAntiRewardHacking(unittest.TestCase):

    def test_a_cached_result_fails(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            cache_state="served_from_cache")))
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("control 3", " ".join(gate.check.reasons))

    def test_an_undeclared_cache_state_is_could_not_check(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(cache_state="unknown")))
        self.assertEqual(report.outcome(C.GID_ANTI_REWARD_HACKING), S.COULD_NOT_CHECK)

    def test_an_out_of_vocabulary_cache_state_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            anti_hack(cache_state="probably fine")

    def test_reducing_delivered_work_fails(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            delivered_units_candidate=96)))
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("reducing work", " ".join(gate.check.reasons))

    def test_the_candidate_may_not_be_its_own_oracle(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            candidate_output_used_as_oracle=True)))
        self.assertEqual(report.outcome(C.GID_ANTI_REWARD_HACKING), S.FAIL)

    def test_a_self_reported_correctness_verdict_fails(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            correctness_verdict_source="candidate")))
        self.assertEqual(report.outcome(C.GID_ANTI_REWARD_HACKING), S.FAIL)

    def test_no_declared_oracle_fails(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(oracle_ids=())))
        self.assertEqual(report.outcome(C.GID_ANTI_REWARD_HACKING), S.FAIL)

    def test_environment_probing_fails(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            environment_probe_findings=("getenv(\"ASAN_OPTIONS\") in the hot path",))))
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("under test", " ".join(gate.check.reasons))

    def test_empty_findings_from_detectors_that_did_not_run_are_unknown(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            environment_probe_detector_id=None,
            timing_dependent_branch_detector_id=None)))
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("did not run", " ".join(gate.check.reasons))

    def test_timing_dependent_branch_fails(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            timing_dependent_branch_findings=("kernel.hip:7:rdtsc",))))
        self.assertEqual(report.outcome(C.GID_ANTI_REWARD_HACKING), S.FAIL)

    def test_a_fail_is_not_downgraded_by_an_unknown(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            cache_state="unknown", candidate_output_used_as_oracle=True)))
        self.assertEqual(report.outcome(C.GID_ANTI_REWARD_HACKING), S.FAIL)


class TestTheDeliveredWorkFloorNamesTheAnchorThatSetIt(unittest.TestCase):
    """`delivered_units_anchor` is control 3's floor and the comparison against it
    is exact, with no tolerance knob — which is exactly why both counts must come
    from ONE anchor. A floor lifted from another anchor's run is a number, not a
    floor, and it can clear a candidate that really did reduce work."""

    def test_two_of_three_components_is_rejected(self):
        for omitted in ANCHOR_TRIPLE:
            with self.subTest(omitted=omitted):
                with self.assertRaises(ValueError) as ctx:
                    anti_hack(**{omitted: None})
                self.assertIn("A partially named anchor is the defect", str(ctx.exception))
                self.assertIn(f"anti_reward_hacking.{omitted}", str(ctx.exception))

    def test_a_placeholder_capture_anchor_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            anti_hack(anchor_source_commit="0" * 40)
        self.assertIn("placeholder digest", str(ctx.exception))

    def test_a_floor_from_another_anchor_is_refused_by_name(self):
        with self.assertRaises(C.AntiRewardHackingAnchorMismatch) as ctx:
            run(ev=evidence(anti_reward_hacking=anti_hack(
                anchor_binary_sha256=sha("some-other-anchor-binary"))))
        self.assertIn("this anti-reward-hacking capture was taken against anchor",
                      str(ctx.exception))

    def test_a_commit_only_mismatch_is_refused(self):
        with self.assertRaises(C.AntiRewardHackingAnchorMismatch):
            run(ev=evidence(anti_reward_hacking=anti_hack(
                anchor_source_commit=V7_COMMIT)))

    def test_a_floor_the_candidate_clears_does_not_excuse_the_mismatch(self):
        """The candidate delivers MORE than the floor, so the gate has nothing to
        report — and the floor still belongs to another anchor."""
        with self.assertRaises(C.AntiRewardHackingAnchorMismatch):
            run(ev=evidence(anti_reward_hacking=anti_hack(
                delivered_units_candidate=4096,
                anchor_linkage_sha256=sha("another-anchor-linkage"))))

    def test_an_unrecorded_capture_anchor_is_could_not_check(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(**UNRECORDED)))
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("records no anchor identity", " ".join(gate.check.reasons))
        self.assertIn("capture_anchor=unrecorded", gate.notes)

    def test_an_unrecorded_capture_anchor_raises_nothing(self):
        run(ev=evidence(anti_reward_hacking=anti_hack(**UNRECORDED)))

    def test_an_unrecorded_capture_anchor_does_not_downgrade_a_real_failure(self):
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            delivered_units_candidate=96, **UNRECORDED)))
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("reducing work", " ".join(gate.check.reasons))

    def test_an_absent_floor_keeps_its_own_reason_and_gains_no_second_one(self):
        """With no anchor count there is no anchor-derived material to attribute,
        so the pre-existing COULD_NOT_CHECK says what it always said."""
        report = run(ev=evidence(anti_reward_hacking=anti_hack(
            delivered_units_anchor=None, **UNRECORDED)))
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertEqual(gate.check.reasons,
                         ("the anchor's delivered-work count was not recorded, so work "
                          "reduction could not be checked",))

    def test_the_compliant_matched_floor_still_passes(self):
        """The counterpart: the new rule must not be satisfiable by always failing."""
        report = run()
        gate = report.gate(C.GID_ANTI_REWARD_HACKING)
        self.assertEqual(gate.check.outcome, S.PASS)
        self.assertIn(f"capture_anchor={anchor().short()}", gate.notes)


# ---------------------------------------------------------------------------
# §15.2's five controls, end to end through T0
# ---------------------------------------------------------------------------

class TestControlAcceptance(unittest.TestCase):

    def _verdict(self, report):
        return api.compute_verdict(
            tier="T0", gates=report.gates,
            void_scan=api.VoidScan(findings=(), evaluated=(), not_applicable=()),
            search_grade=api.SearchGradeResult(True, (), (), (), ()),
            anchor=anchor(), effect=None)

    def test_positive_control_passes_t0(self):
        report = run(ev=evidence(control_role="positive"))
        self.assertEqual(report.failed, ())
        self.assertEqual(self._verdict(report).status, api.STATUS_PASS)

    def test_neutral_control_passes_t0(self):
        report = run(ev=evidence(control_role="neutral"))
        self.assertEqual(self._verdict(report).status, api.STATUS_PASS)

    def test_degraded_negative_that_falls_back_gets_no_speed_rank_at_all(self):
        report = run(ev=evidence(
            control_role="degraded_negative",
            dispatch_trace=trace(fallback_events=("mmq_id_tile -> generic path",))))
        verdict = self._verdict(report)
        self.assertEqual(verdict.status, api.STATUS_FAIL)
        with self.assertRaises(api.SpeedRankUnavailable):
            verdict.rank_key()
        self.assertIn("not a penalised one", verdict.speed_rank_withheld_reason())

    def test_degraded_negative_that_serves_a_cached_result_gets_no_rank(self):
        report = run(ev=evidence(
            control_role="degraded_negative",
            anti_reward_hacking=anti_hack(cache_state="served_from_cache")))
        with self.assertRaises(api.SpeedRankUnavailable):
            self._verdict(report).rank_key()

    def test_degraded_negative_that_reduces_work_gets_no_rank(self):
        report = run(ev=evidence(
            control_role="degraded_negative",
            anti_reward_hacking=anti_hack(delivered_units_candidate=40)))
        with self.assertRaises(api.SpeedRankUnavailable):
            self._verdict(report).rank_key()

    def test_degraded_negative_that_is_numerically_wrong_gets_no_rank(self):
        wrong = C.ReferenceComparison(
            shape_id="m4096n1k4096-q4_K", op="MUL_MAT", mode="exact_bitwise",
            mismatch_count=4096, max_ulp_observed=None, tolerance_ulp=None,
            oracle_id="ik_llama.cpp@iqk-ref", oracle_is_candidate_derived=False)
        report = run(ev=evidence(
            control_role="degraded_negative",
            reference=reference(comparisons=(wrong,) + reference().comparisons[1:])))
        with self.assertRaises(api.SpeedRankUnavailable):
            self._verdict(report).rank_key()

    def test_aa_control_passes_t0_with_the_anchor_binary(self):
        report = run(ev=evidence(
            control_role="aa",
            linkage=linkage(binary_sha256=sha("anchor-binary")),
            build=build(output_binary_sha256=sha("anchor-binary"))),
            req=request(artifact=api.ArtifactIdentity(
                source_sha256=sha("cand-source"),
                binary_sha256=sha("anchor-binary"),
                linkage_sha256=sha("cand-linkage"))))
        self.assertEqual(report.failed, ())

    def test_historical_replay_control_passes_t0(self):
        report = run(ev=evidence(control_role="historical_replay"))
        self.assertEqual(self._verdict(report).status, api.STATUS_PASS)

    def test_an_unknown_control_role_is_refused(self):
        with self.assertRaises(ValueError):
            evidence(control_role="probably_fine")


# ---------------------------------------------------------------------------
# Coverage: a report cannot omit a gate
# ---------------------------------------------------------------------------

class TestGateCoverageCannotShrink(unittest.TestCase):

    def test_a_missing_gate_raises(self):
        report = run()
        with self.assertRaises(C.GateCoverageGap) as ctx:
            C.T0Report(
                event_id="ake-t0-0001", candidate_id="akc-0001", tier="T0",
                gates=report.gates[:-1], coherence=report.coherence,
                requires_human_code_review=False, human_review_reasons=(),
                release_relevant_properties=(), actor_prediction_score=(),
                anchor_bound=True, demoted_gates=(), policy_ref="x")
        self.assertIn("missing", str(ctx.exception))
        self.assertIn(C.GID_ANTI_REWARD_HACKING, str(ctx.exception))

    def test_a_duplicated_gate_raises(self):
        report = run()
        with self.assertRaises(C.GateCoverageGap):
            C.T0Report(
                event_id="ake-t0-0001", candidate_id="akc-0001", tier="T0",
                gates=report.gates + (report.gates[0],), coherence=report.coherence,
                requires_human_code_review=False, human_review_reasons=(),
                release_relevant_properties=(), actor_prediction_score=(),
                anchor_bound=True, demoted_gates=(), policy_ref="x")

    def test_an_invented_gate_raises(self):
        report = run()
        stray = api.GateResult("t0.invented", api.GATE_INTEGRITY, PASS)
        with self.assertRaises(C.GateCoverageGap) as ctx:
            C.T0Report(
                event_id="ake-t0-0001", candidate_id="akc-0001", tier="T0",
                gates=report.gates[:-1] + (stray,), coherence=report.coherence,
                requires_human_code_review=False, human_review_reasons=(),
                release_relevant_properties=(), actor_prediction_score=(),
                anchor_bound=True, demoted_gates=(), policy_ref="x")
        self.assertIn("unknown", str(ctx.exception))

    def test_evaluate_t0_always_returns_the_full_set_even_with_no_evidence_at_all(self):
        blank = C.T0Evidence(
            control_role=None, change_surface=surface(
                derived_touches_memory=None, derived_touches_threading=None,
                derived_touches_dispatch=None, derived_touches_persistent_state=None,
                derived_ops=(), declared_ops=()),
            symbols=None, build=None, diff=None, static_analysis=None, sanitizers=None,
            op_suite=None, reference=None, boundary_shapes=None, dispatch_trace=None,
            state_safety=None, coherence=None, determinism=None, linkage=None,
            anti_reward_hacking=None)
        report = C.evaluate_t0(request(), blank, policy())
        self.assertEqual(tuple(g.gate_id for g in report.gates), C.T0_GATE_IDS)
        self.assertEqual(report.failed, ())
        self.assertEqual(len(report.unevaluated), 17)


# ---------------------------------------------------------------------------
# The runner seam
# ---------------------------------------------------------------------------

class TestRunnerSeam(unittest.TestCase):

    def runner(self, **kwargs):
        provider = C.StaticEvidenceProvider({"akc-0001": evidence(**kwargs)})
        return C.T0CorrectnessRunner(provider=provider, policy=policy())

    def test_run_gates_returns_exactly_the_declared_gate_set(self):
        gates = self.runner().run_gates(request())
        self.assertEqual(tuple(g.gate_id for g in gates), C.T0_GATE_IDS)

    def test_the_runner_satisfies_the_dispatcher_protocol(self):
        dispatcher = api.TierDispatcher(gate_runners={"T0": self.runner()})
        self.assertEqual(dispatcher.tiers, ("T0",))

    def test_missing_evidence_raises_rather_than_synthesising_a_report(self):
        runner = self.runner()
        with self.assertRaises(C.T0EvidenceUnavailable) as ctx:
            runner.evaluate(request(candidate_id="akc-9999"))
        self.assertIn("must not be synthesisable from nothing", str(ctx.exception))

    def test_a_non_t0_request_is_refused(self):
        with self.assertRaises(api.EvaluatorNotWired):
            self.runner().evaluate(request(tier="T1"))

    def test_a_release_tier_is_refused_by_name(self):
        with self.assertRaises(api.TierNotOwned):
            self.runner().evaluate(request(tier="T3"))

    def test_the_provider_refuses_a_non_evidence_value(self):
        with self.assertRaises(TypeError):
            C.StaticEvidenceProvider({"akc-0001": {"looks": "like evidence"}})

    def test_the_runner_refuses_a_provider_without_the_method(self):
        with self.assertRaises(TypeError):
            C.T0CorrectnessRunner(provider=object(), policy=policy())

    def test_the_runner_refuses_a_non_policy(self):
        with self.assertRaises(TypeError):
            C.T0CorrectnessRunner(provider=self.runner()._provider, policy={"ops": ()})


# ---------------------------------------------------------------------------
# End to end through api.TierDispatcher
# ---------------------------------------------------------------------------

def window(**overrides) -> api.WindowAttestations:
    kwargs = dict(
        resource_claim_receipt="gpu_device.mi210_0:claim-20260803T1200Z-8801",
        resource_claim_open=PASS, resource_claim_close=PASS,
        resource_claim_same_holder=PASS, no_concurrent_inference=PASS,
        preflight_attestation_ref="ake-preflight-0007",
        host_receipt="host-health-20260803T1159Z", host_health=PASS,
        anchor_at_open=anchor(), anchor_at_close=anchor(), anchor_gate=PASS,
        evaluator_bundle=PASS, runtime_source_label=PASS,
        recipe=api.RecipeReceipt(constructor_id="ak.t0.opsuite.llama_gpu/v1",
                                 constructor_sha256=sha("recipe-constructor"),
                                 argv_sha256=sha("argv")),
        storage_open=PASS, storage_close=PASS, strata=PASS,
        stopping_rule_id="ak.stopping.bounded_extension/v1", rule_immutability=PASS,
        order_randomized=PASS, order_seed="campaign-seed-4711", aa_cadence=PASS,
        controls=api.ControlPanel(positive=PASS, neutral=PASS, degraded_negative=PASS,
                                  aa=PASS, historical_replay=PASS),
        calibration=PASS, control_definitions_immutable=PASS,
        raw_evidence_ref="data/ak/akc-0001/")
    kwargs.update(overrides)
    return api.WindowAttestations(**kwargs)


class TestEndToEndDispatch(unittest.TestCase):

    def dispatch(self, req=None, **ev_kwargs):
        provider = C.StaticEvidenceProvider({"akc-0001": evidence(**ev_kwargs)})
        runner = C.T0CorrectnessRunner(provider=provider, policy=policy())
        dispatcher = api.TierDispatcher(gate_runners={"T0": runner})
        return dispatcher.dispatch(req or request(), window())

    def test_a_clean_t0_candidate_yields_a_pass_verdict_and_an_emitted_event(self):
        outcome = self.dispatch()
        self.assertEqual(outcome.verdict.status, api.STATUS_PASS)
        self.assertTrue(outcome.emitted)
        self.assertEqual(outcome.event_violations, ())

    def test_the_mul_mat_only_candidate_is_not_ranked_end_to_end(self):
        outcome = self.dispatch(op_suite=op_suite(
            ops_exercised=("MUL_MAT",), cases_by_op=(("MUL_MAT", 4231, 4231),)))
        self.assertEqual(outcome.verdict.status, api.STATUS_FAIL)
        with self.assertRaises(api.SpeedRankUnavailable):
            outcome.verdict.rank_key()

    def test_the_grammar_line_says_search_record_not_a_claim(self):
        outcome = self.dispatch()
        self.assertIn("SEARCH RECORD, NOT A CLAIM", outcome.grammar_line)
        self.assertIn("category=CANDIDATE", outcome.grammar_line)

    def test_the_durable_payload_carries_every_gate(self):
        outcome = self.dispatch()
        gates = outcome.durable_payload["verdict"]["gates"]
        self.assertEqual(tuple(g["gate_id"] for g in gates), C.T0_GATE_IDS)

    def test_an_anchorless_run_is_invalid_and_is_still_journaled_as_a_record(self):
        outcome = self.dispatch(req=request(anchor=None))
        self.assertEqual(outcome.verdict.status, api.STATUS_INVALID)
        self.assertTrue(outcome.durable_payload)
        self.assertIn(api.VOID_ANCHOR_MISSING_OR_MUTATED,
                      outcome.void_scan.reasons())
        # v3: the void is a RECORD, not only a durable payload the caller has to
        # invent a home for. Emitted, valid, and carrying no anchor block.
        self.assertTrue(outcome.emitted)
        self.assertIsNone(outcome.event_blocked_reason)
        self.assertEqual(outcome.event_violations, ())
        self.assertNotIn("anchor", outcome.event)
        self.assertEqual(S.validate_evaluation_event(outcome.event), [])


# ---------------------------------------------------------------------------
# The module's own no-write / no-process property
# ---------------------------------------------------------------------------

class TestModuleHasNoWriteOrProcessPath(unittest.TestCase):

    def test_the_self_audit_passes(self):
        check = C.audit_no_write_or_process_paths()
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_the_audit_detects_a_write_path(self):
        check = api.audit_no_write_or_process_paths(
            "from pathlib import Path\n"
            "def f(p):\n    Path(p).write_text('x')\n")
        self.assertEqual(check.outcome, S.FAIL)

    def test_the_audit_detects_a_process_launch(self):
        check = api.audit_no_write_or_process_paths(
            "import subprocess\ndef f():\n    subprocess.check_call(['ls'])\n")
        self.assertEqual(check.outcome, S.FAIL)

    def test_the_recipe_constructor_hash_is_this_modules_own_source(self):
        expected = S.content_hash({
            "module": "autokernel.evaluator.correctness",
            "source": Path(C.__file__).read_text(encoding="utf-8"),
        })
        self.assertEqual(invocation().receipt.constructor_sha256, expected)

    def test_seams_are_recorded_not_silently_carried(self):
        self.assertTrue(C.SEAMS)
        for seam in C.SEAMS:
            self.assertIsInstance(seam, str)
            self.assertGreater(len(seam), 80)


# ---------------------------------------------------------------------------
# Policy and evidence construction refuse the unsayable
# ---------------------------------------------------------------------------

class TestConstructionRefusals(unittest.TestCase):

    def test_policy_has_no_defaults(self):
        with self.assertRaises(TypeError):
            C.T0Policy()

    def test_evidence_has_no_defaults(self):
        with self.assertRaises(TypeError):
            C.T0Evidence()

    def test_a_bool_cannot_stand_in_for_a_three_outcome_check(self):
        with self.assertRaises(TypeError):
            api.GateResult("t0.x", api.GATE_CORRECTNESS, True)

    def test_an_undetermined_derived_flag_is_none_not_false(self):
        self.assertIsNone(surface(derived_touches_memory=None,
                                  derived_touches_threading=None).sanitizers_mandatory)
        self.assertIs(surface(derived_touches_memory=False,
                              derived_touches_threading=False).sanitizers_mandatory, False)
        self.assertIs(surface().sanitizers_mandatory, True)

    def test_a_non_boolean_derived_flag_is_refused(self):
        with self.assertRaises(TypeError):
            surface(derived_touches_memory="probably")

    def test_a_ulp_comparison_with_no_tolerance_is_not_a_comparison(self):
        with self.assertRaises(ValueError):
            C.ReferenceComparison(
                shape_id="s", op="MUL_MAT", mode="ulp_bounded", mismatch_count=0,
                max_ulp_observed=1.0, tolerance_ulp=None, oracle_id="o",
                oracle_is_candidate_derived=False)

    def test_an_operator_waiver_must_name_its_reference(self):
        with self.assertRaises(ValueError):
            C.NotApplicable(reason="operator waived", source="operator_waiver", ref="")

    def test_a_determinism_class_cannot_be_claimed_from_zero_repeats(self):
        with self.assertRaises(ValueError):
            api.DeterminismReport(determinism_class="bitwise_stable", same_seed_repeat_runs=0)

    def test_every_t0_gate_id_is_unique(self):
        self.assertEqual(len(set(C.T0_GATE_IDS)), len(C.T0_GATE_IDS))


# ---------------------------------------------------------------------------
# Adversarial review 2026-08-03. Each test below is a defect that shipped: a
# surface that could be PASSed by deleting, mislabelling, or simply not
# recording the thing it inspects. Each one FAILED before its fix.
# ---------------------------------------------------------------------------

class TestUndeterminedFlagsNeverReadAsFalse(unittest.TestCase):
    """`ChangeSurface` documents that a derived `None` fails closed. One consumer
    did the two-valued thing and turned "we could not tell" into "it does not"."""

    def test_undetermined_threading_is_could_not_check_not_pass(self):
        report = run(ev=evidence(
            state_safety=None,
            change_surface=surface(derived_touches_persistent_state=False,
                                   derived_touches_threading=None,
                                   derived_touches_memory=False)))
        gate = report.gate(C.GID_STATE_SAFETY)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("did not determine", " ".join(gate.check.reasons))

    def test_undetermined_persistent_state_is_could_not_check_not_pass(self):
        report = run(ev=evidence(
            state_safety=None,
            change_surface=surface(derived_touches_persistent_state=None,
                                   derived_touches_threading=False,
                                   derived_touches_memory=False)))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.COULD_NOT_CHECK)

    def test_both_determined_false_still_passes(self):
        # The compliant path must survive the fix, or the guard forbids its own idiom.
        report = run(ev=evidence(
            state_safety=None,
            change_surface=surface(derived_touches_persistent_state=False,
                                   derived_touches_threading=False,
                                   declared_touches_threading=False)))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.PASS)

    def test_three_valued_or_is_none_when_any_input_is_none(self):
        self.assertIs(C._any_true(False, None), None)
        self.assertIs(C._any_true(True, None), True)
        self.assertIs(C._any_true(False, False), False)


class TestNotApplicableCannotOverrideItsOwnDerivation(unittest.TestCase):
    """`NotApplicable` refuses `source="actor"`, which closes the front door.
    The back door was a `static_derivation` waiver contradicting the same
    derivation's `derived_touches_*` flags."""

    def test_a_derivation_may_not_waive_a_dispatch_surface_it_says_is_touched(self):
        na = C.NotApplicable(reason="nothing to hold out", source="static_derivation",
                             ref="ake-derivation-0001")
        report = run(ev=evidence(boundary_shapes=na,
                                 change_surface=surface(derived_touches_dispatch=True)))
        gate = report.gate(C.GID_BOUNDARY_SHAPES)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("invariant 18", " ".join(gate.check.reasons))

    def test_a_derivation_may_not_waive_a_threading_surface_it_says_is_touched(self):
        na = C.NotApplicable(reason="no state in this diff", source="static_derivation",
                             ref="ake-derivation-0001")
        report = run(ev=evidence(state_safety=na,
                                 change_surface=surface(derived_touches_threading=True)))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.FAIL)

    def test_an_operator_waiver_is_a_human_on_the_record_and_still_stands(self):
        na = C.NotApplicable(reason="accepted for this campaign", source="operator_waiver",
                             ref="akw-0007")
        report = run(ev=evidence(state_safety=na,
                                 change_surface=surface(derived_touches_threading=True)))
        self.assertEqual(report.outcome(C.GID_STATE_SAFETY), S.PASS)

    def test_a_waiver_consistent_with_the_derivation_still_passes(self):
        na = C.NotApplicable(reason="not a dispatch change", source="static_derivation",
                             ref="ake-derivation-0001")
        report = run(ev=evidence(boundary_shapes=na,
                                 change_surface=surface(derived_touches_dispatch=False)))
        self.assertEqual(report.outcome(C.GID_BOUNDARY_SHAPES), S.PASS)


class TestSanitizerFlagCheckIsTokenNotSubstring(unittest.TestCase):
    """`-g` is a substring of a build directory named `build-gpu`. The flag check
    joined argv with spaces and asked `in`, so an invocation carrying no
    debug-info flag at all satisfied it."""

    def _inv(self, configure_argv):
        base = invocation()
        return C.SanitizerInvocation(
            constructor_id=base.constructor_id, configure_argv=configure_argv,
            build_argv=base.build_argv, run_argv=base.run_argv, env=base.env,
            receipt=base.receipt, notes=())

    def test_a_path_containing_the_flag_does_not_satisfy_the_flag(self):
        broken = self._inv((
            "cmake", "-S", "/tmp/ak-gpu/src", "-B", "/tmp/ak-gpu/build-asan",
            "-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all "
            "-fno-omit-frame-pointer",
        ))
        check = C.check_sanitizer_invocation(broken)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("'-g'", " ".join(check.reasons))

    def test_the_sanitize_flag_must_reach_the_compiler_not_only_the_linker(self):
        broken = self._inv((
            "cmake", "-S", "/tmp/src", "-B", "/tmp/build",
            "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined "
            "-fno-sanitize-recover=all -fno-omit-frame-pointer -g",
            "-DCMAKE_C_FLAGS=-O2",
        ))
        # Token-level: the linker line carries them, so this specific shape still
        # resolves. What must NOT resolve is a flag nobody passed at all.
        self.assertEqual(C.check_sanitizer_invocation(broken).outcome, S.PASS)
        gone = self._inv(("cmake", "-S", "/tmp/src", "-B", "/tmp/build",
                          "-DCMAKE_C_FLAGS=-O2 -g"))
        check = C.check_sanitizer_invocation(gone)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("-fsanitize=address,undefined", " ".join(check.reasons))

    def test_the_constructed_invocations_own_idiom_still_passes(self):
        self.assertEqual(C.check_sanitizer_invocation(invocation()).outcome, S.PASS)
        tokens = C._compile_flag_tokens(invocation().configure_argv)
        for flag in C._SANITIZER_COMPILE_FLAGS:
            with self.subTest(flag=flag):
                self.assertIn(flag, tokens)


class TestAnUnrecordedSanitizerExitIsNotAPass(unittest.TestCase):

    def test_executed_with_no_exit_code_is_could_not_check_on_both_gates(self):
        report = run(ev=evidence(sanitizers=sanitizers(exit_code=None)))
        for gate_id in (C.GID_ASAN, C.GID_UBSAN):
            with self.subTest(gate=gate_id):
                gate = report.gate(gate_id)
                self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
                self.assertIn("exit code was not recorded", " ".join(gate.check.reasons))

    def test_an_unrecorded_exit_never_downgrades_a_real_failure(self):
        report = run(ev=evidence(sanitizers=sanitizers(
            exit_code=None, sanitizer_build_binary_sha256=sha("cand-binary"))))
        self.assertEqual(report.outcome(C.GID_ASAN), S.FAIL)


class TestTheNoFallbackProofRefusesSelfReport(unittest.TestCase):
    """Control 3's *"silently falling back"* detector read a `produced_by` field
    on the very object it inspects and did not look at it, while the sibling gate
    reading the SAME object refused a candidate-produced trace."""

    def test_a_candidate_produced_trace_cannot_prove_no_fallback(self):
        report = run(ev=evidence(dispatch_trace=trace(produced_by="candidate")))
        gate = report.gate(C.GID_NO_FALLBACK)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("NEVER self-reported", " ".join(gate.check.reasons))

    def test_both_dispatch_gates_agree_about_who_may_produce_the_trace(self):
        for producer in ("candidate", "actor", "unknown"):
            report = run(ev=evidence(dispatch_trace=trace(produced_by=producer)))
            with self.subTest(producer=producer):
                self.assertEqual(report.outcome(C.GID_NO_FALLBACK), S.FAIL)
                self.assertEqual(report.outcome(C.GID_SURFACE_RECONCILIATION), S.FAIL)


class TestToleranceRestsOnAReconciledAnchorClass(unittest.TestCase):
    """The tolerance branch is the only place a self-declared field turns a byte
    DIFFERENCE into an equivalence label. It read the anchor's determinism class
    from the coherence capture and never compared it with the determinism
    evidence sitting in the same T0Evidence."""

    DIVERGENT = dict(candidate_output_sha256=sha("different"), token_agreement_ratio=0.999)

    def test_a_mislabelled_anchor_cannot_buy_an_equivalence_pass(self):
        report = run(ev=evidence(
            coherence=coherence(anchor_determinism_class="bitwise_unstable",
                                **self.DIVERGENT),
            determinism=determinism(anchor_determinism_class="bitwise_stable")))
        gate = report.gate(C.GID_COHERENCE)
        self.assertEqual(report.coherence.label, C.COHERENCE_WITHIN_TOLERANCE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("two records of one anchor property disagreeing",
                      " ".join(gate.check.reasons))

    def test_an_unreconcilable_tolerance_is_could_not_check(self):
        report = run(ev=evidence(
            coherence=coherence(anchor_determinism_class="bitwise_unstable",
                                **self.DIVERGENT),
            determinism=None))
        self.assertEqual(report.outcome(C.GID_COHERENCE), S.COULD_NOT_CHECK)

    def test_a_reconciled_tolerance_still_passes(self):
        report = run(ev=evidence(
            coherence=coherence(anchor_determinism_class="bitwise_unstable",
                                **self.DIVERGENT),
            determinism=determinism(
                anchor_determinism_class="bitwise_unstable",
                anchor_output_digests=(sha("a"), sha("b"), sha("c")))))
        self.assertEqual(report.outcome(C.GID_COHERENCE), S.PASS)

    def test_byte_identity_needs_no_reconciliation(self):
        report = run(ev=evidence(determinism=None))
        self.assertEqual(report.coherence.label, C.COHERENCE_BYTE_IDENTICAL)
        self.assertEqual(report.outcome(C.GID_COHERENCE), S.PASS)


class TestReconciliationRequiresBothRecordsToNameOneAnchor(unittest.TestCase):
    """The reconciliation above compared what two records SAY and never which
    anchor each says it about. Two captures taken against different anchors that
    happen to agree read exactly like corroboration, and bought the tolerance
    branch's PASS — the one place a self-declared field turns a byte difference
    into an equivalence label."""

    DIVERGENT = dict(candidate_output_sha256=sha("different"), token_agreement_ratio=0.999)

    def _unstable(self, **determinism_overrides):
        kwargs = dict(anchor_determinism_class="bitwise_unstable",
                      anchor_output_digests=(sha("a"), sha("b"), sha("c")))
        kwargs.update(determinism_overrides)
        return evidence(
            coherence=coherence(anchor_determinism_class="bitwise_unstable",
                                **self.DIVERGENT),
            determinism=determinism(**kwargs))

    def test_two_records_of_different_anchors_do_not_corroborate(self):
        with self.assertRaises(C.DeterminismAnchorMismatch) as ctx:
            run(ev=self._unstable(anchor_binary_sha256=sha("some-other-anchor-binary")))
        message = str(ctx.exception)
        self.assertIn("this determinism capture was taken against anchor", message)
        self.assertIn("two independent records corroborating", message)

    def test_a_commit_only_disagreement_between_the_two_records_is_refused(self):
        """Both digests agree, both records declare `bitwise_unstable`, and the
        agreement is between two different anchors."""
        with self.assertRaises(C.DeterminismAnchorMismatch):
            run(ev=self._unstable(anchor_source_commit=V7_COMMIT))

    def test_the_refusal_precedes_the_class_comparison(self):
        """A mismatch is refused even when the classes ALSO disagree: the FAIL
        would report a disagreement between two anchors as a disagreement about
        one, which is a different — and wrong — finding."""
        with self.assertRaises(C.DeterminismAnchorMismatch):
            run(ev=self._unstable(anchor_determinism_class="bitwise_stable",
                                  anchor_output_digests=(sha("gen-out"),) * 3,
                                  anchor_binary_sha256=sha("some-other-anchor-binary")))

    def test_an_unrecorded_determinism_identity_is_not_corroboration(self):
        report = run(ev=self._unstable(**UNRECORDED))
        gate = report.gate(C.GID_COHERENCE)
        self.assertEqual(report.coherence.label, C.COHERENCE_WITHIN_TOLERANCE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("does not establish that they describe the SAME anchor",
                      " ".join(gate.check.reasons))

    def test_an_unrecorded_determinism_identity_raises_nothing(self):
        run(ev=self._unstable(**UNRECORDED))

    def test_the_reconciliation_refuses_on_its_own(self):
        """Called directly — `check_output_coherence` is public — the reconciliation
        refuses without help from the determinism gate downstream of it. The two
        bindings are independent: this one is record-to-record, and the gate's is
        record-to-request."""
        ev = self._unstable(anchor_binary_sha256=sha("some-other-anchor-binary"))
        with self.assertRaises(C.DeterminismAnchorMismatch):
            C.check_output_coherence(request(), ev.coherence, policy(), ev.determinism)

    def test_the_reconciliation_withholds_corroboration_on_its_own(self):
        """Same altitude for the absent case: COULD_NOT_CHECK from this function,
        not from a later gate."""
        ev = self._unstable(**UNRECORDED)
        gate, verdict = C.check_output_coherence(request(), ev.coherence, policy(),
                                                 ev.determinism)
        self.assertEqual(verdict.label, C.COHERENCE_WITHIN_TOLERANCE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("does not establish that they describe the SAME anchor",
                      " ".join(gate.check.reasons))

    def test_a_class_disagreement_keeps_its_original_reason(self):
        """Same anchor, disagreeing classes: the pre-existing FAIL, untouched."""
        report = run(ev=self._unstable(anchor_determinism_class="bitwise_stable",
                                       anchor_output_digests=(sha("gen-out"),) * 3))
        gate = report.gate(C.GID_COHERENCE)
        self.assertEqual(gate.check.outcome, S.FAIL)
        self.assertIn("two records of one anchor property disagreeing",
                      " ".join(gate.check.reasons))

    def test_two_records_of_the_SAME_anchor_still_corroborate(self):
        """The required counterpart: a reconciliation between two captures taken
        against one anchor corroborates exactly as it did before."""
        ev = self._unstable()
        self.assertEqual(ev.coherence.recorded_anchor(), ev.determinism.recorded_anchor())
        report = run(ev=ev)
        self.assertEqual(report.coherence.label, C.COHERENCE_WITHIN_TOLERANCE)
        self.assertEqual(report.outcome(C.GID_COHERENCE), S.PASS)

    def test_the_absent_determinism_record_keeps_its_original_reason(self):
        report = run(ev=evidence(
            coherence=coherence(anchor_determinism_class="bitwise_unstable",
                                **self.DIVERGENT),
            determinism=None))
        gate = report.gate(C.GID_COHERENCE)
        self.assertEqual(gate.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("no determinism evidence was captured to reconcile that against",
                      " ".join(gate.check.reasons))


class TestEveryAnchorDerivedSurfaceNamesItsAnchorTheSameWay(unittest.TestCase):
    """One defect class, one rule. Five evidence types carry anchor-DERIVED
    material; each names the anchor by the same triple, under the same validator,
    and each refuses a replay against a different one by a named error."""

    #: `(builder, label)` for every evidence type carrying the triple.
    CARRIERS = (
        (coherence, "coherence"), (linkage, "linkage"), (determinism, "determinism"),
        (static_analysis, "static"), (anti_hack, "anti_reward_hacking"),
    )

    def test_every_carrier_names_the_anchor_by_all_three_components(self):
        for builder, label in self.CARRIERS:
            for omitted in ANCHOR_TRIPLE:
                with self.subTest(evidence=label, omitted=omitted):
                    with self.assertRaises(ValueError) as ctx:
                        builder(**{omitted: None})
                    # One validator, one sentence — not five hand-copied rules.
                    self.assertIn("A partially named anchor is the defect", str(ctx.exception))
                    self.assertIn(f"{label}.{omitted}", str(ctx.exception))

    def test_every_carrier_refuses_a_placeholder_component(self):
        for builder, label in self.CARRIERS:
            with self.subTest(evidence=label):
                with self.assertRaises(ValueError) as ctx:
                    builder(anchor_binary_sha256="0" * 64)
                self.assertIn("placeholder digest", str(ctx.exception))

    def test_every_mismatch_error_is_one_catchable_family(self):
        for error in (C.CoherenceAnchorMismatch, C.DeterminismAnchorMismatch,
                      C.StaticAnalysisAnchorMismatch, C.AntiRewardHackingAnchorMismatch):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, C.EvidenceAnchorMismatch))
                self.assertTrue(issubclass(error, C.CorrectnessError))
                self.assertTrue(issubclass(error, api.EvaluatorError))
                self.assertIn(error.__name__, C.__all__)

    def test_a_report_where_nothing_recorded_its_anchor_raises_nothing_and_passes_nothing(self):
        """Absence is COULD_NOT_CHECK-shaped on every surface at once. It must not
        raise — nothing disagrees — and it must not pass either."""
        report = run(ev=evidence(
            coherence=coherence(**UNRECORDED), determinism=determinism(**UNRECORDED),
            static_analysis=static_analysis(**UNRECORDED), linkage=linkage(**UNRECORDED),
            anti_reward_hacking=anti_hack(**UNRECORDED)))
        for gate_id in (C.GID_COHERENCE, C.GID_DETERMINISM, C.GID_STATIC_COMPILE,
                        C.GID_LINKAGE, C.GID_ANTI_REWARD_HACKING):
            with self.subTest(gate=gate_id):
                self.assertEqual(report.outcome(gate_id), S.COULD_NOT_CHECK)

    def test_the_fully_recorded_report_still_passes_every_one_of_them(self):
        """The counterpart at report altitude: five new refusals, and the clean
        candidate is still clean. A rule satisfiable by always refusing would show
        up here."""
        report = run()
        for gate_id in (C.GID_COHERENCE, C.GID_DETERMINISM, C.GID_STATIC_COMPILE,
                        C.GID_LINKAGE, C.GID_ANTI_REWARD_HACKING):
            with self.subTest(gate=gate_id):
                self.assertEqual(report.outcome(gate_id), S.PASS)


class TestATotalDisagreementIsMeasurable(unittest.TestCase):
    """An observed agreement of exactly 0.0 was validated against a POLICY floor's
    `(0, 1]` domain and raised, forcing the provider to send `None` — which the
    derivation reports as *"no token agreement ratio was measured"*. A measured
    extreme rewritten as a missing measurement."""

    def test_zero_agreement_is_recordable(self):
        ev = coherence(candidate_output_sha256=sha("different"), token_agreement_ratio=0.0,
                       anchor_determinism_class="bitwise_unstable")
        self.assertEqual(ev.token_agreement_ratio, 0.0)

    def test_zero_agreement_is_divergent_and_says_why(self):
        verdict = C.compute_coherence(
            anchor=anchor(),
            evidence=coherence(candidate_output_sha256=sha("different"),
                               token_agreement_ratio=0.0,
                               anchor_determinism_class="bitwise_unstable"),
            tolerance_floor=0.995)
        self.assertEqual(verdict.label, C.COHERENCE_DIVERGENT)
        self.assertIn("is below the declared floor", " ".join(verdict.reasons))
        self.assertNotIn("no token agreement ratio was measured", " ".join(verdict.reasons))

    def test_a_policy_floor_of_zero_is_still_refused(self):
        # A floor of zero would admit everything; it is not an observation.
        with self.assertRaises(ValueError):
            policy(coherence_tolerance_floor=0.0)

    def test_an_out_of_range_observation_is_still_refused(self):
        with self.assertRaises(ValueError):
            coherence(token_agreement_ratio=1.5)
        with self.assertRaises(ValueError):
            coherence(token_agreement_ratio=-0.1)


class TestAReportCannotMisfileAGate(unittest.TestCase):
    """A report could carry all seventeen ids and file one of them under a class
    `api.SPEED_BLOCKING_GATE_CLASSES` does not contain — seventeen lines, one of
    which does not block ranking. The import-time assertion pins the SPEC; nothing
    pinned the RESULTS, and `T0Report` is public."""

    def _report_with(self, gate_id, **overrides):
        good = run()
        gates = tuple(
            api.GateResult(
                gate_id=g.gate_id,
                gate_class=overrides.get("gate_class", g.gate_class),
                check=g.check,
                requires_anchor=overrides.get("requires_anchor", g.requires_anchor),
                evidence_ref=g.evidence_ref, notes=g.notes)
            if g.gate_id == gate_id else g
            for g in good.gates)
        return C.T0Report(
            event_id="ake-t0-0001", candidate_id="akc-0001", tier="T0", gates=gates,
            coherence=good.coherence, requires_human_code_review=False,
            human_review_reasons=(), release_relevant_properties=(),
            actor_prediction_score=(), anchor_bound=True, demoted_gates=(), policy_ref="x")

    def test_a_t0_gate_refiled_under_a_non_blocking_class_raises(self):
        with self.assertRaises(C.GateCoverageGap) as ctx:
            self._report_with(C.GID_NO_FALLBACK, gate_class=api.GATE_MECHANISM)
        self.assertIn("lexicographically prior", str(ctx.exception))

    def test_clearing_requires_anchor_raises(self):
        with self.assertRaises(C.GateCoverageGap) as ctx:
            self._report_with(C.GID_COHERENCE, requires_anchor=False)
        self.assertIn("precondition 4", str(ctx.exception))

    def test_a_non_gate_result_is_a_type_error_not_an_attribute_error(self):
        good = run()
        with self.assertRaises(TypeError):
            C.T0Report(
                event_id="ake-t0-0001", candidate_id="akc-0001", tier="T0",
                gates=("not a gate",) * 17, coherence=good.coherence,
                requires_human_code_review=False, human_review_reasons=(),
                release_relevant_properties=(), actor_prediction_score=(),
                anchor_bound=True, demoted_gates=(), policy_ref="x")

    def test_the_honest_report_still_builds(self):
        self.assertEqual(len(self._report_with(C.GID_NO_FALLBACK).gates), 17)


# ---------------------------------------------------------------------------
# The build and diff records name their producer
# ---------------------------------------------------------------------------

class TestBuildAndDiffEvidenceCannotBeSelfReported(unittest.TestCase):
    """RED TEAM: `BuildProvenance` and `DiffPolicyEvidence` had no `produced_by`.

    Eleven of this file's other evidence types have carried it since they were
    written, and every one of their gates FAILs a record the evaluator did not
    produce. These two did not, so three §8.5.1 gates believed whoever handed
    them a record:

      * `build_dir_was_fresh` / `incremental_objects_present` — the clean-build
        claim, taken on the producer's word.
      * `commit_was_pathspec_limited` — in a shared clone an unrestricted commit
        sweeps another session's staged files into the artifact, and the
        CANDIDATE was believed about whether it had done that.

    `t0_provider.SCHEMA_FOLLOWUPS` recorded both gaps and named this exact
    remedy. Break it by deleting either field, or either `produced_by !=
    "evaluator"` branch.
    """

    PRODUCERS = ("candidate", "actor", "unknown")

    def test_the_records_carry_the_field_and_validate_it(self):
        for factory, label in ((build, "build"), (diff, "diff")):
            with self.subTest(record=label):
                self.assertIn("produced_by", dataclasses.asdict(factory()))
                with self.assertRaises(ValueError) as ctx:
                    factory(produced_by="the actor, honest")
                self.assertIn(f"{label}.produced_by", str(ctx.exception))

    def test_the_clean_build_gate_fails_a_self_report(self):
        for producer in self.PRODUCERS:
            with self.subTest(produced_by=producer):
                gate = C.check_clean_build_from_snapshot(
                    request(), build(produced_by=producer))
                self.assertEqual(gate.check.outcome, S.FAIL)
                self.assertIn("NEVER self-reported", gate.check.reasons[0])

    def test_the_semantic_diff_gate_fails_a_self_report(self):
        for producer in self.PRODUCERS:
            with self.subTest(produced_by=producer):
                gate = C.check_semantic_diff_conformance(diff(produced_by=producer))
                self.assertEqual(gate.check.outcome, S.FAIL)
                self.assertIn("NEVER self-reported", gate.check.reasons[0])

    def test_the_schema_and_diff_policy_gate_fails_a_self_report(self):
        for producer in self.PRODUCERS:
            with self.subTest(produced_by=producer):
                gate, review = C.check_schema_and_diff_policy(
                    diff(produced_by=producer), surface(), policy())
                self.assertEqual(gate.check.outcome, S.FAIL)
                self.assertIn("NEVER self-reported", gate.check.reasons[0])
                self.assertEqual(review, ())

    def test_a_self_report_is_not_rescued_by_being_otherwise_clean(self):
        """The evaluator-produced twin of each record PASSes, so the FAIL above is
        attributable to the producer and to nothing else."""
        self.assertEqual(
            C.check_clean_build_from_snapshot(request(), build()).check.outcome, S.PASS)
        self.assertEqual(
            C.check_semantic_diff_conformance(diff()).check.outcome, S.PASS)
        gate, _ = C.check_schema_and_diff_policy(diff(), surface(), policy())
        self.assertEqual(gate.check.outcome, S.PASS)


# ---------------------------------------------------------------------------
# A fabricated digest is not a measured identity
# ---------------------------------------------------------------------------

class TestAPlaceholderDigestIsNotAnIdentity(unittest.TestCase):
    """RED TEAM: `_req_sha256` matched `^[0-9a-f]{64}$` and stopped there.

    PROBE: `BuildProvenance(output_binary_sha256="0" * 64)` constructed cleanly,
    so the IDENTITY OF THE BUILT CANDIDATE could be a hand-typed filler. Every
    downstream reader — the champion view, the release package, a human reading
    the journal — takes a well-formed digest for a measured one, and an ABSENT
    identity is loud where a fabricated one is silent and wrong.

    `execution/t0_provider._req_sha256` was byte-identical to this one but for the
    five lines that reject it; this file did not have them.

    Break it by deleting the `schemas.is_placeholder_digest` branch from
    `correctness._req_sha256`.
    """

    FILLERS = ("0" * 64, "f" * 64, "a" * 64,
               hashlib.sha256(b"").hexdigest())

    def test_the_build_output_binary_digest_refuses_a_filler(self):
        for value in self.FILLERS:
            with self.subTest(digest=value[:8]):
                with self.assertRaises(ValueError) as ctx:
                    build(output_binary_sha256=value)
                self.assertIn("placeholder digest", str(ctx.exception))
                self.assertIn("build.output_binary_sha256", str(ctx.exception))

    def test_the_snapshot_digest_refuses_a_filler_too(self):
        with self.assertRaises(ValueError) as ctx:
            build(built_from_snapshot_sha256="0" * 64)
        self.assertIn("placeholder digest", str(ctx.exception))

    def test_the_optional_form_refuses_it_as_well(self):
        with self.assertRaises(ValueError):
            C._opt_sha256("0" * 64, "probe")
        self.assertIsNone(C._opt_sha256(None, "probe"))

    def test_COMPLIANT_a_measured_digest_is_still_accepted(self):
        measured = hashlib.sha256(b"a real build artifact").hexdigest()
        self.assertEqual(build(output_binary_sha256=measured).output_binary_sha256,
                         measured)

    def test_a_malformed_digest_still_fails_for_its_own_reason(self):
        with self.assertRaises(ValueError) as ctx:
            build(output_binary_sha256="not-a-digest")
        self.assertIn("lowercase sha256 hex digest", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
