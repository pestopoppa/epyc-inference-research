"""The gates, and the one property that makes them gates: order."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import gates


def _champion_build(dest, targets=gates.DEFAULT_TARGETS):
    """Stand in for `gates.compiles`: produce the binary the anchor is measured with."""
    (Path(dest) / "bin").mkdir(parents=True, exist_ok=True)
    (Path(dest) / "bin" / "llama-bench").write_text("elf", encoding="utf-8")
    return gates.Verdict("compile", True)


def _broken_build(dest, targets=gates.DEFAULT_TARGETS):
    return gates.Verdict("compile", False, "build failed")


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


class TheAnchorMustAdvanceWithTheChampion(unittest.TestCase):
    """Run 13 kept four patches whose MARGINAL effects were +5.574%, -0.209%, -0.478%
    and -2.864%. Only the first improved anything.

    The anchor was a fixed binary while the candidate worktree accumulated every kept
    patch, so each reported effect was cumulative against original v9. A patch that
    made the champion WORSE still cleared the floor, because the accumulated total
    did. The champion ended at +1.846% having been +5.574% after one patch.
    """

    def _source(self):
        return (Path(__file__).resolve().parent / "run.py").read_text()

    def test_the_anchor_arm_is_not_the_immutable_cli_argument(self):
        source = self._source()
        block = source.split("def measure_for(", 1)[1][:600]
        self.assertIn('bench.Arm("anchor", anchor_build[0]', block)
        self.assertNotIn('bench.Arm("anchor", args.anchor_build', block,
                         "a static anchor makes every effect cumulative, not marginal")

    def test_it_advances_only_after_the_commit_succeeds(self):
        """An anchor advanced for a patch that did not land would silently raise the
        bar for everything after it. Since the sequential path's deletion the one
        commit is `commit_pooled`: the champion ref moves, THEN the anchor builds."""
        source = self._source()
        block = source.split(
            "def commit_pooled(worker, hypothesis, paths, comparison)", 1)[1][:2900]
        self.assertIn("advance_champion", block)
        self.assertIn("promote_anchor", block)
        self.assertLess(block.index("advance_champion"),
                        block.index("promote_anchor("),
                        "promotion must follow the commit, never precede it")

    def test_the_guard_runs_before_the_headline_is_published(self):
        """The guard (`verify_anchor`) must precede any headline publish.

        A headline refreshed BEFORE `verify_anchor` would publish a number measured
        against a slot nobody has yet proven holds the champion -- run 18's void
        number, on the panel the operator reads. Found as a live mutation hole at
        the R21-7 port: swapping the two calls survived every suite, because the
        end-to-end tests compose the modules themselves and cannot see this wiring.

        R23-44 relocated the headline: `promote_anchor` advances the ACCUMULATOR and
        runs the guard but NO LONGER publishes; `publish_headline` moved into
        `accumulate_after_keep`, fired only when a bundle's SERVING gate promotes the
        champion of record. The ordering guarantee is preserved structurally: in
        `commit_pooled`, `promote_anchor()` (which contains `verify_anchor()`) is
        called before `accumulate_after_keep()` (which contains `publish_headline()`),
        and inside accumulate_after_keep the publish sits on the PROMOTE branch after
        the champion-of-record snapshot. Pinned here the same way the order is."""
        source = self._source()
        # promote_anchor advances the accumulator + guards, and must NOT publish.
        promote = source.split("def promote_anchor()", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("verify_anchor()", promote)
        self.assertNotIn("publish_headline()", promote)
        # the headline lives in accumulate_after_keep, after the cor snapshot.
        accum = source.split("def accumulate_after_keep(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("publish_headline()", accum)
        self.assertLess(accum.index("snapshot_cor("), accum.index("publish_headline()"))
        # commit_pooled calls promote_anchor (guard) BEFORE accumulate_after_keep (headline).
        commit = source.split("def commit_pooled(", 1)[1].split("return pool.drive(", 1)[0]
        self.assertLess(commit.index("promote_anchor()"),
                        commit.index("accumulate_after_keep("))

    def test_the_guard_is_wired_with_the_real_code_digest(self):
        """R22-3: the hash pre-check only exists if `run.py` actually injects it.

        `anchor.verify(digest=None)` is the A/A-only fallback, so dropping this one
        kwarg silently reverts the whole triad -- run 21's healthy run aborts again
        and run 18's mismatch costs 20 pairs again -- while every injected-double
        test stays green. Same wiring-only blind spot as the ordering test above."""
        source = self._source()
        # `verify_anchor` nests `keep_verdict`, so cut at the NEXT top-level def.
        block = source.split("def verify_anchor()", 1)[1]
        block = block.split("def promote_anchor", 1)[0]
        self.assertIn("digest=anchor_integrity.build_digest", block)

    def test_an_excursion_note_reaches_the_headline_refresh(self):
        """The excursion-flagged promotion still publishes -- the anchor is
        hash-proven -- but the bundle must say the session's A/A read above the
        floor. `publish_headline` is the only caller that can carry that note."""
        source = self._source()
        block = source.split("def publish_headline()", 1)[1].split("def ", 1)[0]
        self.assertIn("anchor_guard_seen", block)
        self.assertIn("excursion", block)
        self.assertIn("note=", block)

    def test_the_promoted_build_is_BUILT_in_the_anchor_slot(self):
        """EXECUTED, not grepped. Its predecessor asserted that the string
        "shutil.move" appeared in the source; it passed while `shutil` was never
        imported, so the first real keep raised NameError and the anchor silently
        never advanced.

        The contract itself changed on 2026-08-30: promotion BUILDS the champion into
        the new slot instead of renaming a build directory into it. A CMake build
        directory is not relocatable, and that is the leading root cause of run 18.
        """
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            promoted = pool.promote_anchor(store, build=_champion_build,
                                           champion_commit="5ad3e36d")
            self.assertTrue((promoted / "bin" / "llama-bench").is_file())
            self.assertEqual(promoted.parent, store)

    def test_generations_do_not_collide(self):
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            seen = [pool.promote_anchor(store, build=_champion_build,
                                        champion_commit="5ad3e36d")
                    for _ in range(3)]
            self.assertEqual(len(set(seen)), 3, "each keep needs its own anchor")

    def test_a_champion_that_will_not_build_is_refused(self):
        """A promotion that produced no binary would make every later comparison
        measure nothing at all."""
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                pool.promote_anchor(Path(tmp), build=_broken_build,
                                    champion_commit="5ad3e36d")


class ThePooledPathMustAdvanceTheAnchorToo(unittest.TestCase):
    """The sequential path promotes the anchor inside its own commit. The pooled path
    has a separate commit, and not doing it there would leave the anchor static across
    every lane -- reproducing run 13's defect (cumulative effects reported as marginal,
    a -2.864% regression committed as a keep) at seven times the rate."""

    def _pooled_block(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        return source.split("def commit_pooled(", 1)[1][:2900]  # widened for the R23-43 serving gate

    def test_it_advances_the_champion_then_the_anchor(self):
        block = self._pooled_block()
        self.assertIn("pool.advance_champion", block)
        self.assertIn("promote_anchor", block)
        self.assertLess(block.index("pool.advance_champion"),
                        block.index("promote_anchor("),
                        "the anchor must follow the commit, never precede it")

    def test_it_never_promotes_a_lane_build_DIRECTORY(self):
        """A lane's build directory is not relocatable, so it can no longer be handed
        to the promotion at all: the champion is rebuilt into the anchor slot."""
        block = self._pooled_block()
        self.assertIn("promote_anchor()", block)
        self.assertNotIn("promote_anchor(worker.build_dir)", block)
        self.assertNotIn("promote_anchor(args.candidate_build)", block)

    def test_the_pooled_commit_is_actually_wired_in(self):
        """A commit_pooled that nothing calls is the defect it was written to fix."""
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        drive = source.split("return pool.drive(", 1)[1][:400]
        self.assertIn("commit=commit_pooled", drive)


class PromoteAnchorMustACTUALLYRun(unittest.TestCase):
    """`test_the_promoted_build_is_moved_out_of_the_candidate_slot` asserted that the
    string "shutil.move" appears in run.py. It passed. `shutil` was never imported.

    Run 14 kept a real +6.723% patch, advanced the champion, and then raised
    NameError inside promote_anchor -- so the keep was recorded as a lane_error and
    the anchor never advanced. A test that greps source for a spelling proves the
    spelling. This one imports the module and executes the function."""

    def test_the_module_imports_everything_promote_anchor_uses(self):
        from autokernel.loop import pool
        self.assertTrue(hasattr(pool, "json"),
                        "promote_anchor writes provenance.json; pool must import json")

    def test_the_promotion_actually_executes(self):
        """Exercise the real function, not the source text."""
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            promoted = pool.promote_anchor(Path(tmp), build=_champion_build,
                                           champion_commit="5ad3e36d")
            self.assertTrue((promoted / "bin" / "llama-bench").is_file())
            self.assertTrue((promoted / "provenance.json").is_file())


class PromotionMustSurvivePruning(unittest.TestCase):
    """Run 17 lost 23 of its 30 champion advances to this, and the errors were the
    lesser harm: the anchor stopped advancing after the FIRST keep, so every later
    effect was cumulative against a stale champion -- the defect the advancing anchor
    exists to prevent, reintroduced by the pruning meant to save disk.

    Numbering by COUNT collides the moment pruning holds the population steady. With
    keep=1 the count is always 1, so every promotion targeted anchor-gen-002 forever.
    """

    def test_repeated_promote_and_prune_never_collides(self):
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            seen = []
            for _ in range(5):
                promoted = pool.promote_anchor(store, build=_champion_build,
                                               champion_commit="5ad3e36d")
                pool.prune_anchor_generations(store, current=promoted)
                seen.append(promoted.name)
            self.assertEqual(len(set(seen)), 5, f"generations collided: {seen}")

    def test_it_refuses_to_reuse_an_existing_anchor(self):
        """Configuring into a slot that already holds a generation would build over
        another champion's CMakeCache -- the relocation hazard by another route."""
        from autokernel.loop import pool
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "anchor-gen-001").mkdir()
            (store / "anchor-gen-002").mkdir()
            # numbering must step past BOTH, not reuse 002
            promoted = pool.promote_anchor(store, build=_champion_build,
                                           champion_commit="5ad3e36d")
            self.assertEqual(promoted.name, "anchor-gen-003")
            self.assertTrue((promoted / "bin").is_dir())


class TwoTierChampionWiring(unittest.TestCase):
    """R23-44: the accumulator advances on bench keeps; the champion of record advances
    only when a bundle's serving gate promotes it. These pin the run.py wiring the unit
    tests on accumulate.py cannot see (same wiring-only blind spot as the guard-order test)."""

    def _source(self):
        return (Path(__file__).resolve().parent / "run.py").read_text()

    def test_keep_gate_is_the_bench_confirm_rung_not_a_per_keep_serving_gate(self):
        src = self._source()
        # the removed per-keep serving gate must be gone; the keep gate is confirm.gate
        self.assertNotIn("def serving_confirm", src)
        commit = src.split("def commit_pooled(", 1)[1].split("return pool.drive(", 1)[0]
        self.assertIn("confirm.gate(", commit)
        self.assertIn("accumulate_after_keep(hypothesis.mechanism_id)", commit)

    def test_serving_gate_compares_champion_of_record_against_accumulator(self):
        src = self._source()
        accum = src.split("def accumulate_after_keep(", 1)[1].split("\n    def ", 1)[0]
        # the serving A-arm is the champion-of-record build, B-arm the accumulator anchor
        self.assertIn("serving.compare(serving_recipe, cor_build[0], anchor_build[0]", accum)
        # it only fires when the bundle clears the threshold
        self.assertIn("accumulate.Decision.FIRE_SERVING", accum)
        # promote advances cor + snapshots + headline; divergence journals evidence
        self.assertIn("accumulate.Outcome.PROMOTE", accum)
        self.assertIn("planner_evidence", accum)
        self.assertIn("measured_divergence", accum)

    def test_cor_build_is_snapshotted_before_the_loop_and_protected_from_pruning(self):
        src = self._source()
        # cor lives in a slot NOT matching anchor-gen-*, so prune_anchor_generations cannot hit it
        self.assertIn('cor_slot = args.store / "cor-build"', src)
        # snapshot happens at startup under the claim, before run_pooled
        held = src.split("with claim.hold()", 1)[1].split("pooled = run_pooled()", 1)[0]
        self.assertIn("snapshot_cor(args.anchor_build)", held)

    def test_fire_multiple_arg_defaults_to_operator_range(self):
        src = self._source()
        arg = src.split('"--fire-multiple"', 1)[1][:120]
        self.assertIn("default=2.5", arg)
