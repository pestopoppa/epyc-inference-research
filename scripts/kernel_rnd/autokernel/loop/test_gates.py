"""The gates, and the one property that makes them gates: order."""
from pathlib import Path
import unittest

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
