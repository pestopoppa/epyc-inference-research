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

    def test_manifest_checksum_tamper_detected(self):
        task = self._task()
        m = c6.build_run_manifest(task, run_id="r1", sources=self._sources())
        path = os.path.join(self.tmp, "run_manifest.json")
        c6.write_run_manifest(path, m)
        # Tamper with a signed field without fixing manifest_sha256.
        blob = json.load(open(path))
        blob["config"] = {"trial_seed": 999}
        json.dump(blob, open(path, "w"))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
