"""Unit tests for c6_reward_integrity.py — the C6 reward-integrity + provenance
harness for the MI210 auto-kernel loop.

NO GPU, NO llama.cpp, NO inference, NO model. Every "kernel" here is a DUMMY
Python script that writes a JSON artifact; every "evaluator" is a DUMMY trusted
scorer that prints one JSON verdict line. The candidate sandbox layer is
exercised with allow_unsandboxed=True because these tests run inside an
already-isolated devcontainer (the module fails closed by default — that path
is covered by test_sandbox_fails_closed).

Run standalone (no pytest needed):
    python3 scripts/kernel_rnd/test_c6_reward_integrity.py
Or via pytest:
    python3 -m pytest scripts/kernel_rnd/test_c6_reward_integrity.py -v
"""
from __future__ import annotations

import json
import math
import os
import stat
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c6_reward_integrity as c6


def _write(path, text, mode=0o755):
    with open(path, "w") as f:
        f.write(textwrap.dedent(text))
    os.chmod(path, mode)
    return path


# A trusted evaluator that RECOMPUTES the verdict from the artifact content and
# deliberately IGNORES any self_reported_* fields the candidate wrote.
_EVALUATOR = """\
    import json, sys
    data = json.load(open(sys.argv[1]))
    answer = data.get("answer")
    # self_reported_score / self_reported_latency_ms are present but NEVER read.
    if answer == "hello":
        print(json.dumps({"is_correct": True, "sol_score": 42.0,
                          "latency_ms": 12.5, "metrics": {"recomputed": True}}))
    else:
        # Naughty on purpose: emit a blazing score for a WRONG answer. The
        # harness correctness-gate must strip it so it can never rank.
        print(json.dumps({"is_correct": False, "sol_score": 5000.0,
                          "latency_ms": 0.01}))
"""

_CANDIDATE_LYING = """\
    import json
    # Reports an absurd score + near-zero latency for itself.
    json.dump({"answer": "hello", "self_reported_score": 999999.0,
               "self_reported_latency_ms": 0.0001}, open("solution.json", "w"))
"""

_CANDIDATE_FAST_BUT_WRONG = """\
    import json
    json.dump({"answer": "WRONG", "self_reported_latency_ms": 0.01},
              open("solution.json", "w"))
"""


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="c6_test_")
        self.evaluator = _write(
            os.path.join(self.tmp, "evaluator.py"), _EVALUATOR)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _task(self, **over):
        base = dict(
            entry_point="kernel.py::run",
            target_hardware="MI210/gfx90a",
            dependencies=("numpy",),
            evaluator=self.evaluator,
            timeout_s=30,
            max_memory_mb=2048,          # generous so RLIMIT_AS never flakes
            evaluator_max_memory_mb=1024,
            evaluator_timeout_s=30,
        )
        base.update(over)
        return c6.KernelTaskSpec(**base)


# ---------------------------------------------------------------------------
# 1. Trusted evaluator ignores a lying self-reported score
# ---------------------------------------------------------------------------
class TestTrustedEvaluator(_Base):
    def test_ignores_lying_self_reported_score(self):
        work = os.path.join(self.tmp, "work")
        _write(os.path.join(self.tmp, "cand.py"), _CANDIDATE_LYING)
        result = c6.evaluate_candidate(
            [sys.executable, os.path.join(self.tmp, "cand.py")],
            work, self._task(), allow_unsandboxed=True,
        )
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.is_correct)
        # The verdict comes from the trusted evaluator, NOT the candidate's
        # self_reported_score (999999) or self_reported_latency_ms (0.0001).
        self.assertEqual(result.sol_score, 42.0)
        self.assertEqual(result.latency_ms, 12.5)
        self.assertNotEqual(result.sol_score, 999999.0)
        self.assertIsNotNone(result.candidate_artifact_sha256)


# ---------------------------------------------------------------------------
# 2. Anti-TOCTOU snapshot rejects symlink / FIFO / hard-linked artifacts
# ---------------------------------------------------------------------------
class TestAntiTOCTOU(_Base):
    def _trusted_dir(self, name):
        return os.path.join(self.tmp, ".trusted", name)

    def test_clean_regular_file_snapshots_immutably(self):
        art = os.path.join(self.tmp, "solution.json")
        with open(art, "w") as f:
            f.write('{"answer": "hello"}')
        snap, digest, data = c6.snapshot_candidate_artifact(
            art, self._trusted_dir("clean"), proc=None)
        self.assertTrue(os.path.exists(snap))
        # snapshot is read-only (0444)
        mode = stat.S_IMODE(os.stat(snap).st_mode)
        self.assertEqual(mode, 0o444)
        self.assertEqual(len(digest), 64)

    def test_rejects_symlink_swap(self):
        target = os.path.join(self.tmp, "real.json")
        with open(target, "w") as f:
            f.write('{"answer": "hello"}')
        link = os.path.join(self.tmp, "solution.json")
        os.symlink(target, link)
        with self.assertRaises(c6.ArtifactRejected) as ctx:
            c6.snapshot_candidate_artifact(link, self._trusted_dir("sym"))
        self.assertIn("symbolic link", str(ctx.exception))

    def test_rejects_fifo(self):
        fifo = os.path.join(self.tmp, "solution.json")
        os.mkfifo(fifo)
        with self.assertRaises(c6.ArtifactRejected) as ctx:
            c6.snapshot_candidate_artifact(fifo, self._trusted_dir("fifo"))
        self.assertIn("regular file", str(ctx.exception))

    def test_rejects_hard_linked_file(self):
        art = os.path.join(self.tmp, "solution.json")
        with open(art, "w") as f:
            f.write('{"answer": "hello"}')
        os.link(art, os.path.join(self.tmp, "second_link.json"))  # st_nlink -> 2
        with self.assertRaises(c6.ArtifactRejected) as ctx:
            c6.snapshot_candidate_artifact(art, self._trusted_dir("hard"))
        self.assertIn("hard link", str(ctx.exception))

    def test_rejects_oversize_file(self):
        art = os.path.join(self.tmp, "solution.json")
        with open(art, "w") as f:
            f.write("x" * 4096)
        with self.assertRaises(c6.ArtifactRejected):
            c6.snapshot_candidate_artifact(
                art, self._trusted_dir("big"), max_bytes=1024)


# ---------------------------------------------------------------------------
# 3. Correctness-gate blocks a fast-but-wrong kernel from scoring
# ---------------------------------------------------------------------------
class TestCorrectnessGate(_Base):
    def test_fast_but_wrong_never_scores(self):
        work = os.path.join(self.tmp, "work")
        _write(os.path.join(self.tmp, "cand.py"), _CANDIDATE_FAST_BUT_WRONG)
        result = c6.evaluate_candidate(
            [sys.executable, os.path.join(self.tmp, "cand.py")],
            work, self._task(), allow_unsandboxed=True,
        )
        # Evaluator emitted a fast sol_score for the wrong answer; the gate
        # strips both score and latency because is_correct is False.
        self.assertFalse(result.is_correct)
        self.assertIsNone(result.sol_score)
        self.assertIsNone(result.latency_ms)

    def test_gate_drops_score_at_type_boundary(self):
        # Even if a score is passed explicitly, gated() drops it when not correct.
        ev = c6.KernelEvaluation.gated(
            is_correct=False, sol_score=123.0, latency_ms=0.5, status="ok")
        self.assertIsNone(ev.sol_score)
        self.assertIsNone(ev.latency_ms)

    def test_rank_excludes_incorrect_even_if_faster(self):
        wrong = c6.KernelEvaluation.gated(
            is_correct=False, sol_score=9999.0, latency_ms=0.001, status="ok")
        right = c6.KernelEvaluation.gated(
            is_correct=True, sol_score=10.0, latency_ms=100.0, status="ok")
        ranked = c6.rank_correct_first([wrong, right])
        self.assertEqual(ranked, [right])       # wrong is not present at all

    def test_rank_agrees_with_kernel_store_is_correct(self):
        # A kernel_eval.sh-shaped record flows through the same gate.
        good = {"status": "OK",
                "correctness": {"test_backend_ops": "42/42 tests passed",
                                "coherence": "byte-identical"},
                "sol_score": 50.0}
        bad = {"status": "OK",
               "correctness": {"test_backend_ops": "41/42 tests passed",
                               "coherence": "byte-identical"},
               "sol_score": 99.0}
        self.assertTrue(c6.is_correct(good))
        self.assertFalse(c6.is_correct(bad))
        self.assertEqual(c6.rank_correct_first([bad, good]), [good])


# ---------------------------------------------------------------------------
# 4. Provenance drift rejects a resume after an evaluator/source edit
# ---------------------------------------------------------------------------
class TestProvenance(_Base):
    def _sources(self):
        src = os.path.join(self.tmp, "proposal_agent.py")
        _write(src, "def propose():\n    return 1\n", mode=0o644)
        return {"proposal_agent.py": src}

    def test_clean_resume_passes(self):
        task = self._task()
        sources = self._sources()
        m1 = c6.build_run_manifest(task, run_id="r1", sources=sources,
                                   config={"trial_seed": 42})
        path = os.path.join(self.tmp, "run_manifest.json")
        c6.write_run_manifest(path, m1)
        recorded = c6.load_run_manifest(path)     # checksum verified on load
        current = c6.build_run_manifest(task, run_id="r1", sources=sources,
                                        config={"trial_seed": 42})
        self.assertIs(c6.validate_run_manifest(recorded, current), recorded)
        self.assertEqual(
            recorded["evaluator_policy"]["flashinfer_bench_source_commit"],
            c6.FLASHINFER_BENCH_SOURCE_COMMIT)
        self.assertEqual(recorded["evaluator_policy"]["deterministic_runs"], 3)

    def test_evaluator_edit_rejects_resume(self):
        task = self._task()
        sources = self._sources()
        recorded = c6.build_run_manifest(task, run_id="r1", sources=sources,
                                         config={"trial_seed": 42})
        # Operator edits the trusted evaluator between runs -> result-affecting.
        with open(self.evaluator, "a") as f:
            f.write("\n# tweak scoring\n")
        current = c6.build_run_manifest(task, run_id="r1", sources=sources,
                                        config={"trial_seed": 42})
        with self.assertRaises(c6.ProvenanceError) as ctx:
            c6.validate_run_manifest(recorded, current)
        self.assertIn("task", str(ctx.exception))   # evaluator sha lives in task

    def test_source_edit_rejects_resume(self):
        task = self._task()
        sources = self._sources()
        recorded = c6.build_run_manifest(task, run_id="r1", sources=sources)
        with open(next(iter(sources.values())), "a") as f:
            f.write("\n# refactor\n")
        current = c6.build_run_manifest(task, run_id="r1", sources=sources)
        with self.assertRaises(c6.ProvenanceError) as ctx:
            c6.validate_run_manifest(recorded, current)
        self.assertIn("source_sha256", str(ctx.exception))

    def test_config_drift_rejects_resume(self):
        task = self._task()
        sources = self._sources()
        recorded = c6.build_run_manifest(task, run_id="r1", sources=sources,
                                         config={"trial_seed": 42})
        current = c6.build_run_manifest(task, run_id="r1", sources=sources,
                                        config={"trial_seed": 7})
        with self.assertRaises(c6.ProvenanceError) as ctx:
            c6.validate_run_manifest(recorded, current)
        self.assertIn("config", str(ctx.exception))

    def test_evaluator_policy_drift_rejects_resume(self):
        task = self._task()
        recorded = c6.build_run_manifest(
            task, run_id="r1", sources=self._sources())
        current = json.loads(json.dumps(recorded))
        current["evaluator_policy"]["maximum_atol"] = 0.02
        with self.assertRaises(c6.ProvenanceError) as ctx:
            c6.validate_run_manifest(recorded, current)
        self.assertIn("evaluator_policy", str(ctx.exception))

    def test_manifest_checksum_tamper_detected(self):
        task = self._task()
        m = c6.build_run_manifest(task, run_id="r1", sources=self._sources())
        path = os.path.join(self.tmp, "run_manifest.json")
        c6.write_run_manifest(path, m)
        # Tamper with a signed field without fixing manifest_sha256.
        with open(path) as stream:
            blob = json.load(stream)
        blob["config"] = {"trial_seed": 999}
        with open(path, "w") as stream:
            json.dump(blob, stream)
        with self.assertRaises(c6.ProvenanceError):
            c6.load_run_manifest(path)


# ---------------------------------------------------------------------------
# 5. RunLock is mutually exclusive
# ---------------------------------------------------------------------------
class TestRunLock(_Base):
    def test_single_writer(self):
        lock_path = os.path.join(self.tmp, "run", ".lock")
        first = c6.RunLock(lock_path)
        first.acquire()
        second = c6.RunLock(lock_path)
        with self.assertRaises(c6.RunLockError):
            second.acquire()
        # Release the first; the second can then take it.
        first.release()
        third = c6.RunLock(lock_path)
        third.acquire()          # succeeds now
        third.release()

    def test_context_manager(self):
        lock_path = os.path.join(self.tmp, "run", ".lock")
        with c6.RunLock(lock_path):
            other = c6.RunLock(lock_path)
            with self.assertRaises(c6.RunLockError):
                other.acquire()
        # lock released on exit -> re-acquirable
        with c6.RunLock(lock_path):
            pass


# ---------------------------------------------------------------------------
# 6. (bonus) Sandbox fails closed; evidence-gated stop
# ---------------------------------------------------------------------------
class TestSandboxFailsClosed(_Base):
    def test_no_backend_no_override_raises(self):
        saved = (c6.SANDBOX_BACKEND, c6.SANDBOX_TOOL)
        c6.SANDBOX_BACKEND, c6.SANDBOX_TOOL = None, None
        env_saved = os.environ.pop(c6._ALLOW_ENV, None)
        try:
            with self.assertRaises(c6.SandboxUnavailable):
                c6.build_sandboxed_command(
                    ["true"], writable_dir=self.tmp, allow_unsandboxed=False)
            # explicit override returns the command unwrapped
            self.assertEqual(
                c6.build_sandboxed_command(
                    ["true"], writable_dir=self.tmp, allow_unsandboxed=True),
                ["true"])
        finally:
            c6.SANDBOX_BACKEND, c6.SANDBOX_TOOL = saved
            if env_saved is not None:
                os.environ[c6._ALLOW_ENV] = env_saved


class TestEvidenceGatedStop(_Base):
    def _correct_rec(self, score):
        return {"is_correct": True, "sol_score": score, "latency_ms": 10.0}

    def test_empty_records_never_stop(self):
        ctrl = c6.KernelStopController(c6.KernelStopPolicy(enabled=True))
        review = ctrl.review(c6.StopRequest("stop"), [])
        self.assertFalse(review.accepted)
        self.assertIn("no_records", review.reasons)

    def test_malformed_records_never_stop(self):
        pol = c6.KernelStopPolicy(enabled=True, min_records=1, min_correct=1,
                                  stop_patience=0)
        ctrl = c6.KernelStopController(pol)
        review = ctrl.review(c6.StopRequest("stop"),
                             ["not-a-dict", {"garbage": 1}])
        self.assertFalse(review.accepted)
        self.assertIn("malformed_records", review.reasons)

    def test_continue_request_never_accepted_as_stop(self):
        pol = c6.KernelStopPolicy(enabled=True, min_records=1, min_correct=1,
                                  stop_patience=0)
        ctrl = c6.KernelStopController(pol)
        recs = [self._correct_rec(10.0), self._correct_rec(10.0)]
        review = ctrl.review(c6.StopRequest("continue"), recs)
        self.assertFalse(review.accepted)
        self.assertIn("not_a_stop_request", review.reasons)

    def test_evidence_agrees_accepts_stop(self):
        pol = c6.KernelStopPolicy(enabled=True, min_records=3, min_correct=3,
                                  stop_patience=2)
        ctrl = c6.KernelStopController(pol)
        # 4 correct evals; best set on the first, no improvement after ->
        # evals_since_improvement = 3 >= patience 2.
        recs = [self._correct_rec(s) for s in (50.0, 40.0, 45.0, 30.0)]
        review = ctrl.review(c6.StopRequest("stop"), recs)
        self.assertTrue(review.accepted, review.reasons)
        self.assertEqual(review.evidence["correct_records"], 4)

    def test_disabled_policy_blocks_stop(self):
        ctrl = c6.KernelStopController(c6.KernelStopPolicy(enabled=False))
        recs = [self._correct_rec(10.0)] * 5
        review = ctrl.review(c6.StopRequest("stop"), recs)
        self.assertFalse(review.accepted)
        self.assertIn("stops_disabled", review.reasons)


class TestTaskSpec(_Base):
    def test_entry_point_parsing_and_gate(self):
        t = self._task()
        self.assertEqual(t.entry_module(), "kernel.py")
        self.assertEqual(t.entry_callable(), "run")
        # with_result gates: a non-correct result carries no score/latency
        gated = t.with_result(is_correct=False, sol_score=5.0, latency_ms=1.0)
        self.assertIsNone(gated.sol_score)
        self.assertIsNone(gated.latency_ms)
        core = t.scoring_core()
        self.assertEqual(
            set(core),
            {"entry_point", "target_hardware", "dependencies",
             "is_correct", "sol_score", "latency_ms"})


class TestRatifiedNumericalPolicy(_Base):
    def evidence(self, output="float32", accumulator="float32"):
        return c6.StructuralPrecisionEvidence(
            output_dtype=output, accumulator_dtype=accumulator,
            evidence_sha256="a" * 64)

    def test_flashinfer_source_pin_and_exact_defaults(self):
        self.assertEqual(
            c6.FLASHINFER_BENCH_SOURCE_COMMIT,
            "40e6ca7844b514eb4b1c7edba6d6a7377df57870")
        self.assertEqual(c6.FLASHINFER_DEFAULT_ATOL, 1e-2)
        self.assertEqual(c6.FLASHINFER_DEFAULT_RTOL, 1e-2)
        self.assertEqual(c6.FLASHINFER_LOWBITS_MATCHED_RATIO, 0.95)

    def test_structural_dtype_and_accumulator_precede_tolerance(self):
        policy = c6.PrecisionContract("float32", "float32")
        bad_output = c6.evaluate_numerics(
            [float("nan")], [float("nan")],
            structural=self.evidence(output="float16"), policy=policy)
        self.assertEqual(bad_output.stage, "structural")
        self.assertEqual(bad_output.reason, "incorrect_output_dtype")
        self.assertEqual(bad_output.total_elements, 0)
        self.assertEqual(bad_output.structural_evidence_sha256, "a" * 64)
        self.assertEqual(bad_output.required_output_dtype, "float32")
        self.assertEqual(bad_output.observed_output_dtype, "float16")
        bad_acc = c6.evaluate_numerics(
            [0.0], [0.0], structural=self.evidence(accumulator="float16"),
            policy=policy)
        self.assertEqual(bad_acc.reason, "incorrect_accumulator_dtype")
        bad_shape = c6.evaluate_numerics(
            [[0.0, 1.0]], [0.0, 1.0], structural=self.evidence(),
            policy=policy)
        self.assertEqual(bad_shape.reason, "incorrect_shape")

    def test_policy_can_tighten_but_never_loosen_source_bounds(self):
        c6.PrecisionContract("float32", "float32", atol=1e-3, rtol=1e-4)
        with self.assertRaisesRegex(c6.EvaluatorPolicyError, "equal to or tighter"):
            c6.PrecisionContract("float32", "float32", atol=2e-2)
        with self.assertRaisesRegex(c6.EvaluatorPolicyError, "equal to or tighter"):
            c6.PrecisionContract("float32", "float32", rtol=2e-2)

    def test_elementwise_predicate_is_abs_and_rel(self):
        policy = c6.PrecisionContract("float32", "float32")
        # abs error exceeds atol, but relative error does not: source AND
        # predicate says this element is matched.
        verdict = c6.evaluate_numerics(
            [1000.0], [1000.02], structural=self.evidence(), policy=policy)
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.outlier_elements, 0)
        # Both exceed their bounds: it is an outlier.
        verdict = c6.evaluate_numerics(
            [1.0], [1.02], structural=self.evidence(), policy=policy)
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.outlier_elements, 1)

    def test_lowbit_matched_ratio_has_explicit_outlier_budget(self):
        policy = c6.PrecisionContract(
            "float16", "float32", required_matched_ratio=0.95, lowbit=True)
        reference = [1.0] * 100
        passing = reference.copy()
        passing[:5] = [2.0] * 5
        verdict = c6.evaluate_numerics(
            reference, passing,
            structural=self.evidence("float16", "float32"), policy=policy)
        self.assertTrue(verdict.correct)
        self.assertEqual(verdict.allowed_outliers, 5)
        self.assertEqual(verdict.outlier_elements, 5)
        failing = passing.copy()
        failing[5] = 2.0
        verdict = c6.evaluate_numerics(
            reference, failing,
            structural=self.evidence("float16", "float32"), policy=policy)
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.matched_ratio, 0.94)
        with self.assertRaisesRegex(c6.EvaluatorPolicyError, r"\[0.95, 1.0\]"):
            c6.PrecisionContract(
                "float16", "float32", required_matched_ratio=0.94,
                lowbit=True)

    def test_nonfinite_refusal_and_max_errors_are_recorded(self):
        policy = c6.PrecisionContract("float32", "float32")
        verdict = c6.evaluate_numerics(
            [0.0, 1.0], [0.0, float("inf")],
            structural=self.evidence(), policy=policy)
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.reason, "nonfinite_output")
        self.assertEqual(verdict.nonfinite_count, 1)
        self.assertEqual(verdict.max_absolute_error, "inf")
        finite = c6.evaluate_numerics(
            [1.0, 2.0], [1.005, 2.03],
            structural=self.evidence(), policy=policy)
        self.assertTrue(math.isclose(finite.max_absolute_error, 0.03))
        self.assertIsInstance(finite.max_relative_error, float)


class TestDeterminismAndFallback(_Base):
    def test_determinism_runs_exactly_three_times(self):
        calls = []

        def run():
            calls.append(len(calls))
            return [1.0, -0.0, b"same"]

        verdict, output = c6.run_three_bitwise(run)
        self.assertEqual(len(calls), 3)
        self.assertEqual(verdict.run_count, 3)
        self.assertTrue(verdict.correct)
        self.assertEqual(output, [1.0, -0.0, b"same"])

    def test_bitwise_difference_is_rejected_after_all_three_runs(self):
        calls = []

        def run():
            calls.append(len(calls))
            return 0.0 if len(calls) < 3 else -0.0

        verdict, _ = c6.run_three_bitwise(run)
        self.assertEqual(len(calls), 3)
        self.assertFalse(verdict.correct)

    def test_fallback_return_is_replaced_and_rerun(self):
        source = """
def wrapper(kernel, reference, x):
    try:
        return kernel(x)
    except Exception:
        return reference(x)
"""
        seen = []

        def rerun(mutated):
            seen.append(mutated)
            namespace = {}
            exec(mutated, namespace)
            try:
                namespace["wrapper"](
                    lambda _x: (_ for _ in ()).throw(RuntimeError("kernel")),
                    lambda _x: "laundered", 1)
            except RuntimeError as exc:
                return str(exc) == "C6 fallback return disabled for re-run"
            return False

        probe = c6.probe_fallback_laundering(source, rerun)
        self.assertTrue(probe.correct)
        self.assertEqual(probe.mutated_returns, 1)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("return reference(x)", seen[0])

    def test_laundering_is_detected_when_mutated_rerun_does_not_pass(self):
        source = "try:\n    result = kernel()\nexcept Exception:\n    return_value = 1\n"
        probe = c6.probe_fallback_laundering(source, lambda _source: True)
        self.assertEqual(probe.reason, "no_fallback_return")
        laundering = "def f():\n try:\n  return kernel()\n except:\n  return reference()\n"
        probe = c6.probe_fallback_laundering(laundering, lambda _source: False)
        self.assertFalse(probe.correct)
        self.assertEqual(probe.reason, "fallback_laundering_detected")


class TestSemanticJudgeAndHardware(_Base):
    def test_tiers_preserve_l1_l2_judge_and_drop_l3(self):
        self.assertEqual(
            c6.C6_GATE_TIERS,
            ("L1_static", "L2_ghost_replay", "semantic_judge"))
        self.assertEqual(c6.C6_DROPPED_TIERS, ("L3",))

    def test_semantic_judge_is_non_gating_until_all_three_rejected(self):
        partial = c6.calibrate_semantic_judge({
            "layernorm_no_affine": "REJECT",
            "softmax_no_maxsub": "REJECT",
            "matmul_transpose_no_t": "ACCEPT",
        })
        self.assertFalse(partial.gating)
        self.assertEqual(partial.missing_mutants, ("matmul_transpose_no_t",))
        complete = c6.calibrate_semantic_judge({
            name: "REJECT" for name in c6.C6_SEMANTIC_CALIBRATION_MUTANTS})
        self.assertTrue(complete.gating)

    def test_unknown_gpu_refuses_without_fallback(self):
        self.assertEqual(c6.require_supported_gpu("gfx90a")["part"], "gfx90a")
        with self.assertRaisesRegex(c6.UnknownHardwareError, "refusing"):
            c6.require_supported_gpu("gfx942")


class TestAdmissionReceipts(_Base):
    def receipt(self, **overrides):
        values = dict(
            task_id="task-1", candidate_commit="a" * 40,
            anchor_commit="b" * 40, evaluator_commit="c" * 40,
            first_turn_anchor_latency_ms=120.0,
            first_turn_candidate_latency_ms=100.0,
            verification_anchor_latency_ms=156.0,
            verification_candidate_latency_ms=100.0,
            first_turn_correct=True, verification_correct=True,
            reopen_when="candidate or evaluator commit changes",
            policy=c6.AdmissionPolicy(implausible_speedup_cap=32.0),
        )
        values.update(overrides)
        return c6.build_admission_receipt(**values)

    def test_admits_only_execution_verified_rerun(self):
        receipt = self.receipt()
        self.assertAlmostEqual(receipt["first_turn_speedup"], 1.2)
        self.assertAlmostEqual(receipt["required_speedup"], 1.44)
        self.assertAlmostEqual(receipt["verification_speedup"], 1.56)
        self.assertTrue(receipt["admitted"])
        self.assertEqual(c6.validate_admission_receipt(receipt), receipt)

    def test_floor_failure_is_recorded_not_admitted(self):
        receipt = self.receipt(
            verification_anchor_latency_ms=130.0,
            verification_candidate_latency_ms=100.0)
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["reason"], "verification_threshold_not_met")

    def test_wrong_rerun_cannot_enter_the_library(self):
        receipt = self.receipt(verification_correct=False)
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["reason"], "correctness_refused")

    def test_implausible_speedup_is_refused(self):
        receipt = self.receipt(
            first_turn_anchor_latency_ms=4000.0,
            verification_anchor_latency_ms=5000.0)
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["reason"], "implausible_speedup_refused")

    def test_commit_reopen_and_policy_are_mandatory(self):
        with self.assertRaisesRegex(c6.EvaluatorPolicyError, "40-hex"):
            self.receipt(candidate_commit="main")
        with self.assertRaisesRegex(c6.EvaluatorPolicyError, "reopen_when"):
            self.receipt(reopen_when="")
        with self.assertRaisesRegex(c6.EvaluatorPolicyError, ">= 1.2"):
            c6.AdmissionPolicy(implausible_speedup_cap=32.0, alpha=1.1)

    def test_receipt_tamper_and_extra_fields_refuse(self):
        receipt = self.receipt()
        receipt["verification_speedup"] = 99.0
        with self.assertRaisesRegex(c6.AdmissionReceiptError, "self-hash"):
            c6.validate_admission_receipt(receipt)
        receipt = self.receipt()
        receipt["extra"] = True
        with self.assertRaisesRegex(c6.AdmissionReceiptError, "missing or extra"):
            c6.validate_admission_receipt(receipt)
        receipt = self.receipt()
        receipt["verification_speedup"] = 99.0
        unsigned = {key: value for key, value in receipt.items()
                    if key != "receipt_sha256"}
        receipt["receipt_sha256"] = c6.sha256_json(unsigned)
        with self.assertRaisesRegex(c6.AdmissionReceiptError, "recomputed"):
            c6.validate_admission_receipt(receipt)

    def test_write_side_capture_and_store_are_bound_and_tamper_evident(self):
        receipt = self.receipt()
        capture = c6.build_admission_claim_capture(
            receipt, producer_sha256="d" * 64)
        self.assertEqual(
            c6.validate_admission_claim_capture(capture, receipt), capture)
        path = os.path.join(self.tmp, "admission.jsonl")
        store = c6.AdmissionReceiptStore(path)
        envelope = store.append(receipt, producer_sha256="d" * 64)
        self.assertEqual(store.records(), [envelope])
        with open(path, "r+") as stream:
            payload = json.loads(stream.readline())
            payload["belief_capture"]["value"] = 999.0
            stream.seek(0)
            stream.write(json.dumps(payload) + "\n")
            stream.truncate()
        with self.assertRaisesRegex(c6.AdmissionReceiptError, "invalid admission"):
            store.records()


class TestSeparableRecords(_Base):
    def test_g15_retrodiction_selects_gather_scatter(self):
        record = c6.retrodict_g15_selector()
        self.assertEqual(record["selected_family"], "gather_scatter")
        self.assertTrue(record["selector_validated"])

    def test_round_reflexion_carries_estimate_and_actual(self):
        record = c6.RoundReflexionRecord(
            round_id="r1", candidate_commit="a" * 40,
            was_diagnosis_correct=True, was_fix_effective=False,
            expected_outcome="1.3x from reduced memory traffic",
            actual_outcome="1.0x; traffic unchanged", estimated_speedup=1.3,
            achieved_speedup=1.0, lessons=("counter premise was wrong",),
            avoid_patterns=("unverified traffic assumption",),
            try_patterns=("measure transaction count first",),
        ).to_dict()
        self.assertAlmostEqual(record["estimate_error_fraction"], -0.3 / 1.3)
        self.assertEqual(record["lessons"], ["counter premise was wrong"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
