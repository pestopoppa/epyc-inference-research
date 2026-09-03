"""R22-7: the promoted anchor is production-complete, and only the promoted anchor.

THE DEFECT. Every `anchor-gen-NNN` the loop ever produced was bench-only --
`gates.compiles`' default targets are llama-bench and the op oracle -- so the
artifact whose whole point is being trivially promotable could not serve a
request. Operator ruling (2026-09-01, verbatim): "the autokernel agent should
have built one alongside each champion. The whole point of a champion is that it
needs to be extremely easy to promote into production… If we're not compiling
llama-servers that's a problem" -- and the fix must live INSIDE the loop's own
champion-advancement step, not in any post-hoc script.

THE OTHER HALF of the contract is negative space: candidate lane builds run
hundreds of times per run and the anchor guard's fresh build runs once per keep
purely to answer "is the anchor slot the champion" -- neither may pay server link
time, so their target set must stay exactly `gates.DEFAULT_TARGETS`.

`TheKeepBuildsAProductionCompleteAnchor` is the acceptance test: a simulated keep
driven end-to-end through `run.main` -- real argparse, real startup gate, real
`pipeline.run_pool`/`pool.drive`, real `advance_champion` + `promote_anchor` +
`anchor.verify` on a real temp git repo -- with only the device-shaped edges
doubled (compiles, op oracle, bench, actors, claim, profile, headline refresh).
The property is proven on the AUTOMATIC path a real unattended keep takes, not on
a directly-called helper. It doubles as R22-6's end-to-end: the store's inbox
carries one good seed and two unreadable files, and the run must complete anyway
with both noted in the log.

Mutation notes (both directions):
  * drop "llama-server" (or "llama-cli") from PROMOTION_TARGETS -> the unit and
    e2e target assertions fail (server-target-dropped-from-promotion);
  * widen DEFAULT_TARGETS or hand PROMOTION_TARGETS to the lane/guard builds ->
    the narrowness assertions fail (server-target-LEAKED);
  * stop writing "targets" to provenance.json, or record a list other than the
    one handed to `build` -> the provenance assertions fail;
  * revert `run.py` to an inline bare inbox read -> the e2e dies (breaker) and
    the seam test loses the `read_inbox` reference.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

from autokernel.loop import (anchor, bench, champion, gates, hotspots, loop,
                             pipeline, pool)
from autokernel.loop import run as run_mod

CANONICAL = champion.CANONICAL_BRANCH


def _sh(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {done.stderr}")
    return done.stdout.strip()


class TheTargetSets(unittest.TestCase):
    """The two tuples, and the relation between them."""

    def test_promotion_is_a_superset_of_the_iteration_set(self):
        """A promoted artifact must never lack a binary the loop measured with."""
        self.assertTrue(set(gates.PROMOTION_TARGETS) >= set(gates.DEFAULT_TARGETS))

    def test_promotion_carries_the_production_binaries(self):
        self.assertIn("llama-server", gates.PROMOTION_TARGETS)
        self.assertIn("llama-cli", gates.PROMOTION_TARGETS)

    def test_the_iteration_default_stays_narrow(self):
        """Per-iteration cost protection: `gates.compiles` called without `targets`
        -- every candidate lane build, the guard's fresh build and its heal --
        must not link a server hundreds of times per run."""
        self.assertEqual(gates.DEFAULT_TARGETS, ("llama-bench", "test-backend-ops"))
        self.assertNotIn("llama-server",
                         gates.compiles.__kwdefaults__["targets"])
        self.assertEqual(gates.compiles.__kwdefaults__["targets"],
                         gates.DEFAULT_TARGETS)


class PromoteAnchorOwnsTheWideSet(unittest.TestCase):
    """The default lives INSIDE the champion-advancement step, so no caller can
    produce a bench-only anchor by forgetting a kwarg."""

    def _promote(self, store: Path, **kwargs):
        calls = []

        def build(dest, targets):
            calls.append((Path(dest), tuple(targets)))
            (Path(dest) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(dest) / "bin" / "llama-bench").write_text("elf", encoding="utf-8")
            return gates.Verdict("compile", True)

        promoted = pool.promote_anchor(store, build=build, champion_commit="5ad3e36d",
                                       recipe={"name": "house-gpu"}, **kwargs)
        return promoted, calls

    def test_the_default_build_invocation_carries_the_full_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            promoted, calls = self._promote(Path(tmp))
            self.assertEqual(calls, [(promoted, gates.PROMOTION_TARGETS)])

    def test_provenance_records_the_targets_the_build_was_handed(self):
        """One fact, one source: the recorded list IS the list passed to `build`,
        so an artifact whose record omits llama-server is detectably incomplete."""
        with tempfile.TemporaryDirectory() as tmp:
            promoted, calls = self._promote(Path(tmp))
            body = json.loads((promoted / "provenance.json").read_text())
            self.assertEqual(body["targets"], list(calls[0][1]))
            self.assertIn("llama-server", body["targets"])

    def test_an_explicit_override_is_built_AND_recorded_identically(self):
        """The one-source property must hold off the default path too, or the
        record lies exactly when someone narrows a build by hand."""
        with tempfile.TemporaryDirectory() as tmp:
            promoted, calls = self._promote(Path(tmp), targets=("llama-bench",))
            body = json.loads((promoted / "provenance.json").read_text())
            self.assertEqual(calls[0][1], ("llama-bench",))
            self.assertEqual(body["targets"], ["llama-bench"])


class TheKeepBuildsAProductionCompleteAnchor(unittest.TestCase):
    """One simulated keep, end-to-end through `run.main`'s automatic path."""

    GOOD_SEED = "USE THE MEASURED MFMA LEVER FROM THE BACKLOG"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # `run.main` installs SIGTERM/SIGINT handlers; put the suite's back after.
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))

        # The champion tree: a real repo with THE canonical branch checked out.
        self.repo = self.root / "tree"
        self.repo.mkdir()
        subprocess.run(["git", "-C", str(self.repo), "init", "-q", "-b", CANONICAL],
                       capture_output=True, text=True, timeout=60)
        for key, value in (("user.email", "t@t"), ("user.name", "t")):
            subprocess.run(["git", "-C", str(self.repo), "config", key, value],
                           capture_output=True, text=True, timeout=60)
        (self.repo / "kernel.c").write_text("base\n", encoding="utf-8")
        _sh(self.repo, "add", "kernel.c")
        _sh(self.repo, "commit", "-q", "-m", "champion tip")
        self.tip = _sh(self.repo, "rev-parse", "HEAD")

        self.store = self.root / "store"
        startup_anchor = self.store / "anchor-gen-001"
        startup_anchor.mkdir(parents=True)
        (startup_anchor / "provenance.json").write_text(json.dumps(
            {"champion_commit": self.tip, "build_recipe": {"name": "house-gpu"}}),
            encoding="utf-8")
        self.startup_anchor = startup_anchor
        (self.root / f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf").write_text("", encoding="utf-8")

        # R22-6's live shape, riding along: one good seed, two unreadable files.
        inbox_dir = self.store / "inbox"
        inbox_dir.mkdir()
        (inbox_dir / "seed.md").write_text(f"  {self.GOOD_SEED}\n", encoding="utf-8")
        (inbox_dir / "bad-utf8.md").write_bytes(b"\xff\xfe\xfa")
        (inbox_dir / "dangling.md").symlink_to(inbox_dir / "no-such-target")

    def _run_one_keep(self):
        """Drive `run.main` through exactly one kept iteration; return the record."""
        compile_calls: list[dict] = []
        planners: list = []
        scratch = self.root / "scratch-verify"
        repo, root, tip = self.repo, self.root, self.tip

        def fake_compiles(source_root, build_dir, *, cmake_defines, jobs, cpu_list,
                          targets=gates.DEFAULT_TARGETS, cmake="cmake"):
            compile_calls.append({"source": Path(source_root),
                                  "dest": Path(build_dir),
                                  "targets": tuple(targets)})
            (Path(build_dir) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(build_dir) / "bin" / "llama-bench").write_text("elf",
                                                                 encoding="utf-8")
            return gates.Verdict("compile", True)

        class _Planner:
            def __init__(self, workspace):
                self.workspace = Path(workspace)
                self.contexts: list[dict] = []
                planners.append(self)

            def propose(self, context):
                self.contexts.append(dict(context))
                return loop.Hypothesis(mechanism_id="akm-e2e-keep", statement="s",
                                       falsifier="f", target_surface="kernel.c",
                                       target_symbol="sym")

            def author(self, hypothesis, context):
                (self.workspace / "kernel.c").write_text("patched\n",
                                                         encoding="utf-8")
                return ("kernel.c",)

        class _Critic:
            def __init__(self, workspace):
                pass

            def review_hypothesis(self, hypothesis, context):
                return loop.Review(True)

            def review_patch(self, hypothesis, paths, context):
                return loop.Review(True)

        def fake_compare(anchor_arm, candidate_arm, model, **kw):
            # The lane's A/B keeps (decisively over the pp512 floor); the anchor
            # guard's A/A of two same-commit builds reads exactly zero.
            effect = 0.05 if anchor_arm.name == "anchor" else 0.0
            return bench.Comparison(
                surface=kw["surface"], anchor_samples=[100.0] * 4,
                candidate_samples=[100.0 * (1 + effect)] * 4, effect=effect,
                estimator="median_over_median", pairs=kw["pairs"],
                noise_floor_pct=kw["noise_floor_pct"],
                residency={"invocations": 10, "resident": 10},
                calibrated=kw["calibrated"])

        def fake_provision(count, **kwargs):
            lane, build = root / "lane0", root / "lane0-build"
            if not (lane / ".git").exists():
                _sh(repo, "worktree", "add", "--detach", str(lane), tip)
            build.mkdir(exist_ok=True)
            return [pipeline.Worker("lane0", lane, build)]

        @contextmanager
        def fake_hold(*args, **kwargs):
            yield {"device_id": "e2e-double"}

        real_verify = anchor.verify

        def verify_in_scratch(**kwargs):
            kwargs["scratch_build"] = scratch
            return real_verify(**kwargs)

        argv = ["--worktree", str(self.repo),
                "--anchor-build", str(self.startup_anchor),
                "--model", str(self.root / f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf"), "--store", str(self.store),
                "--iterations", "1", "--workers", "1",
                "--worker-root", str(self.root / "lanes"),
                "--worker-build-root", str(self.root / "builds")]
        out = io.StringIO()
        with mock.patch.object(gates, "compiles", fake_compiles), \
             mock.patch.object(gates, "op_correctness",
                               lambda _b: gates.Verdict("op_correctness", True)), \
             mock.patch.object(run_mod.bench, "compare", fake_compare), \
             mock.patch.object(run_mod.actors, "AgentPlanner",
                               lambda workspace, **_: _Planner(workspace)), \
             mock.patch.object(run_mod.actors, "AgentCritic",
                               lambda workspace, **_: _Critic(workspace)), \
             mock.patch.object(run_mod.pool, "provision", fake_provision), \
             mock.patch.object(run_mod.claim, "hold", fake_hold), \
             mock.patch.object(run_mod.hotspots, "profile",
                               mock.Mock(side_effect=hotspots.ProfileFailed("e2e"))), \
             mock.patch.object(run_mod.workload_contract, "verify_workload",
                               lambda _m: mock.Mock(n_embd=1536,
                                                    dominant_quant="Q4_K")), \
             mock.patch.object(run_mod.production, "refresh",
                               lambda **kw: run_mod.production.Refresh(
                                   published=False, reason="stubbed for the e2e")), \
             mock.patch.object(run_mod.anchor, "verify", verify_in_scratch), \
             redirect_stdout(out):
            rc = run_mod.main(argv)
        return rc, compile_calls, planners, scratch, out.getvalue()

    def test_the_automatic_keep_path_promotes_production_complete(self):
        rc, calls, planners, scratch, log = self._run_one_keep()
        self.assertEqual(rc, 0, log)

        promoted = self.store / "anchor-gen-002"
        promotion = [c for c in calls if c["dest"] == promoted]
        guard = [c for c in calls if c["dest"] == scratch]
        candidate = [c for c in calls if c["dest"] == self.root / "lane0-build"]

        # Non-vacuity: the keep really flowed through all three build classes.
        self.assertEqual((len(candidate), len(promotion), len(guard)), (1, 1, 1),
                         [str(c["dest"]) for c in calls])
        self.assertIn("kept", log)

        # THE ruling: the promoted-anchor build, on the per-keep automatic path,
        # carries the full production target set...
        self.assertEqual(promotion[0]["targets"], gates.PROMOTION_TARGETS)
        self.assertIn("llama-server", promotion[0]["targets"])
        # ...and provenance records it, so an incomplete artifact is detectable.
        body = json.loads((promoted / "provenance.json").read_text())
        self.assertEqual(body["targets"], list(gates.PROMOTION_TARGETS))
        self.assertEqual(body["champion_commit"],
                         _sh(self.repo, "rev-parse", CANONICAL))

        # Negative space: candidate and guard builds pay NO server link time.
        self.assertEqual(candidate[0]["targets"], gates.DEFAULT_TARGETS)
        self.assertEqual(guard[0]["targets"], gates.DEFAULT_TARGETS)
        # Every build was compiled AT the directory it serves (never relocated).
        self.assertEqual(promotion[0]["source"], self.repo)

    def test_the_unreadable_inbox_files_could_not_kill_the_keep(self):
        """R22-6 end-to-end: same run, poisoned live-shaped inbox. BROKEN READS
        (bare reader): zero keeps, three lane_errors, breaker abort, rc != 0."""
        rc, _calls, planners, _scratch, log = self._run_one_keep()
        self.assertEqual(rc, 0, log)
        self.assertEqual(len(planners), 1)
        context = planners[0].contexts[0]
        self.assertEqual(context["inbox"], [self.GOOD_SEED])
        self.assertEqual(log.count("inbox_file_unreadable"), 2, log)
        self.assertIn("bad-utf8.md", log)
        self.assertIn("dangling.md", log)


class TheRunWiringSeams(unittest.TestCase):
    """Structural guards on the closures inside `run.main` no unit test can call."""

    def _referenced(self, code) -> set:
        names = set(code.co_names)
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                names |= self._referenced(const)
        return names

    def test_build_context_reaches_the_hardened_reader(self):
        """A `build_context` quietly reverted to an inline `read_text` loop drops
        the `read_inbox` attribute reference; `inbox` alone would survive a rename,
        so both are pinned. Non-vacuity both directions, as in test_ranking."""
        referenced = self._referenced(run_mod.main.__code__)
        self.assertIn("inbox", referenced)
        self.assertIn("read_inbox", referenced)
        self.assertNotIn("no_such_global_anywhere", referenced)

    def test_run_py_does_not_reimplement_the_bare_reader(self):
        """The bare per-file read must not creep back in beside the helper."""
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        self.assertNotIn("inbox_dir.glob", source)
        self.assertIn("inbox.read_inbox(args.store", source)

    def test_build_champion_forwards_its_targets(self):
        """`build_champion(dest, targets)` that ignores `targets` and calls
        `gates.compiles` bare would give every promotion a bench-only anchor while
        the e2e's fake still records what the closure was HANDED... it does not:
        the e2e records what `gates.compiles` RECEIVED, so this seam test is the
        redundant second lock, kept because it is free and names the line."""
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        block = source.split("def build_champion(", 1)[1]
        block = block.split("def ", 1)[0]
        self.assertIn("targets=targets", block)
        self.assertIn("gates.DEFAULT_TARGETS", block.split(")", 1)[0])


if __name__ == "__main__":
    unittest.main()
