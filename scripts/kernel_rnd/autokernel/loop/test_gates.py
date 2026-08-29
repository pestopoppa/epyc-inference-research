"""The gates, and the one property that makes them gates: order."""
from pathlib import Path
import unittest
from unittest import mock

from autokernel.loop import gates


class TheShortCircuitMustBeReal(unittest.TestCase):
    """`run_all` documented a short-circuit it could not perform.

    It took `*verdicts: Verdict`, and Python evaluates every argument before the call,
    so `run_all(compiles(...), op_correctness(...))` ran the correctness suite even when
    the build had just failed -- against whatever stale binary sat in the candidate build
    directory. The reported verdicts stayed correct -- the first failure short-circuits the
    RETURN, so the extra verdict was discarded -- but every failed build in run 9 still paid
    for a full test-backend-ops run against a stale artifact. A gate that runs after the
    previous gate refused is not a gate, even when nobody reads its answer.
    """

    def test_a_later_check_is_never_called_after_a_refusal(self):
        ran = []

        def failing():
            ran.append("build")
            return gates.Verdict("compile", False, "build failed")

        def must_not_run():
            ran.append("correctness")
            return gates.Verdict("correctness", True)

        passed, verdicts = gates.run_all(failing, must_not_run)
        self.assertFalse(passed)
        self.assertEqual(ran, ["build"],
                         "the correctness suite ran against a binary the build never made")
        self.assertEqual(len(verdicts), 1)

    def test_all_checks_run_when_each_passes(self):
        ran = []

        def ok(name):
            def check():
                ran.append(name)
                return gates.Verdict(name, True)
            return check

        passed, verdicts = gates.run_all(ok("compile"), ok("correctness"))
        self.assertTrue(passed)
        self.assertEqual(ran, ["compile", "correctness"])
        self.assertEqual(len(verdicts), 2)

    def test_the_runner_passes_callables_not_evaluated_verdicts(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        block = source.split("gates.run_all(", 1)[1][:320]
        self.assertIn("lambda: gates.compiles", block)
        self.assertIn("lambda: gates.op_correctness", block)


class ARefusedPatchMustSurviveTheReset(unittest.TestCase):
    """Run 9 lost all ten candidate patches.

    `reset_tree` returns the worktree to the champion before each iteration, so a
    refused patch exists nowhere afterwards. Seven of those ten died on `MUL_MAT failed
    on ROCm0` and not one can be reproduced, re-read or diagnosed. A negative written up
    without its diff is not evidence anyone can act on.
    """

    def test_the_runner_saves_the_diff_before_gating(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        self.assertIn("def keep_the_diff(", source)
        # It must run BEFORE the gate, because a failed build still leaves a patch
        # worth reading and that is the last moment it exists on disk.
        gate_body = source.split("def gate(hypothesis, paths):", 1)[1][:900]
        # `keep_the_diff` takes the LANE as well as the hypothesis since the gate
        # became per-worker: with concurrent lanes a bare `<mechanism>.patch` is two
        # lanes overwriting one file, which loses diffs the same way run 9 did. The
        # property under test is the ORDER, so match the call, not one spelling of
        # its argument list.
        self.assertIn("keep_the_diff(", gate_body)
        before = gate_body.index("keep_the_diff(")
        self.assertLess(before, gate_body.index("gates.run_all"))

    def test_an_empty_diff_writes_nothing(self):
        """An actor that changed nothing must not leave an empty patch file that
        later reads as a real attempt."""
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        body = source.split("def keep_the_diff(", 1)[1][:1200]
        self.assertIn("if not diff.strip():", body)
        self.assertIn("return None", body)


class AnOracleThatCannotRunIsNotAFailedPatch(unittest.TestCase):
    """The worst defect in this rebuild: the correctness gate never ran, and said the
    patch was wrong every time.

    `op_correctness` passed `--suite-seed <n>`, which test-backend-ops does not accept
    in this tree. It printed usage and exited 1, so EVERY candidate was refused with
    "MUL_MAT failed on ROCm0". Proven against the anchor, which passes 1139/1139: the
    exact gate command exits 1, and exits 0 with the flag removed. Seven of ten run-9
    iterations died on this and were written into durable memory as measured negatives.
    """

    def _fake_run(self, stdout, code):
        from unittest import mock
        return mock.patch.object(gates.subprocess, "run",
                                 return_value=mock.Mock(stdout=stdout, stderr="",
                                                        returncode=code))

    def test_usage_text_is_a_harness_fault_not_a_correctness_verdict(self):
        usage = "Usage: test-backend-ops [mode] [-o <op,..>]\n    valid modes:\n"
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run(usage, 1):
            verdict = gates.op_correctness(Path("/nonexistent"))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate, "oracle_unavailable",
                         "an argument error must never read as a failed patch")
        self.assertNotIn("MUL_MAT failed", verdict.reason)

    def test_a_real_failure_is_still_a_correctness_verdict(self):
        ran = "  1100/1139 tests passed\n  Backend ROCm0: FAIL\n1/2 backends passed\n"
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run(ran, 1):
            verdict = gates.op_correctness(Path("/nonexistent"))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate, "correctness")
        self.assertIn("MUL_MAT failed", verdict.reason)

    def test_a_pass_requires_proof_the_suite_executed(self):
        ran = "  1139/1139 tests passed\n2/2 backends passed\nOK\n"
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run(ran, 0):
            self.assertTrue(gates.op_correctness(Path("/nonexistent")).passed)

    def test_a_silent_zero_exit_does_not_pass(self):
        """Exit 0 with no evidence the suite ran is not a pass."""
        with mock.patch.object(Path, "is_file", return_value=True), \
                self._fake_run("", 0):
            verdict = gates.op_correctness(Path("/nonexistent"))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate, "oracle_unavailable")

    def test_the_unsupported_flag_is_gone_from_the_INVOCATION(self):
        """The docstring still names the flag on purpose -- it records the defect.
        What must not contain it is the argv actually handed to the binary."""
        import inspect
        body = inspect.getsource(gates.op_correctness)
        body = body.split('"""', 2)[-1]          # drop the docstring
        self.assertIn("argv = [", body)
        self.assertNotIn("--suite-seed", body)

    def test_the_invocation_is_the_one_proven_to_work_on_the_anchor(self):
        """`test-backend-ops test -o MUL_MAT -b ROCm0 -j 1` exits 0 on the anchor;
        adding --suite-seed makes it exit 1. Pin the proven form."""
        import inspect
        body = inspect.getsource(gates.op_correctness).split('"""', 2)[-1]
        for token in ('"test"', '"-o", op', '"-b", backend', '"-j", "1"'):
            self.assertIn(token, body, token)
