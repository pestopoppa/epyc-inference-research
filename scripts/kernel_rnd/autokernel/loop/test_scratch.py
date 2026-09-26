#!/usr/bin/env python3
"""The scratch-resource registry: release on every exit path, sweep scope, worktree
safety, the disk guard, the prune/gc ban, stats."""
from __future__ import annotations

import ast
import io
import json
import os
import re
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import scratch
from autokernel.loop.scratch import ScratchRefused, ScratchRegistry

PKG = Path(__file__).resolve().parent


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *argv], check=True,
                          capture_output=True, text=True).stdout.strip()


def _make_repo(base: Path) -> tuple[Path, str]:
    repo = base / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo, _git(repo, "rev-parse", "HEAD")


def _worktrees(repo: Path) -> list[str]:
    out = _git(repo, "worktree", "list", "--porcelain")
    return [line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("worktree ")]


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-scratch-test-"))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.tmp)]))
        self.root = self.tmp / "state" / "scratch"

    def reg(self, run_id: str = "run-A", pid: int | None = None, **kw) -> ScratchRegistry:
        owner = {"campaign": "c", "state_dir": str(self.tmp / "state"), "run_id": run_id}
        if pid is not None:
            owner["pid"] = pid
            owner["pid_start"] = None
        return ScratchRegistry(self.root, owner, kw.pop("min_free_bytes", 0), **kw)

    def journal(self) -> list[dict]:
        return [json.loads(line) for line in
                (self.root / scratch.JOURNAL).read_text().splitlines()]


class ReleasePaths(Base):
    def test_normal_exit_releases_all_allocators(self) -> None:
        reg = self.reg()
        with reg.scope("run", name="r") as run:
            d = run.dir("actor-context", "bundle-1")
            f = run.file("opencode-config", "actor-opencode-author.json")
            f.write_text("{}")
            self.assertTrue((d / scratch.MARKER).is_file())
            self.assertTrue(Path(str(f) + scratch.SIDECAR_SUFFIX).is_file())
            marker = json.loads((d / scratch.MARKER).read_text())
            for key in ("owner", "scope", "kind", "created_at", "pid"):
                self.assertIn(key, marker)
        self.assertFalse(d.exists())
        self.assertFalse(f.exists())
        self.assertFalse(Path(str(f) + scratch.SIDECAR_SUFFIX).exists())
        events = [r["event"] for r in self.journal()]
        self.assertEqual(events.count("allocate"), 2)
        self.assertEqual(events.count("release"), 2)

    def test_exception_releases_and_propagates(self) -> None:
        reg = self.reg()
        with self.assertRaises(ZeroDivisionError):
            with reg.scope("iteration", name="it") as it:
                d = it.dir("tmp", "x")
                1 / 0
        self.assertFalse(d.exists())

    def test_keyboard_interrupt_releases(self) -> None:
        reg = self.reg()
        with self.assertRaises(KeyboardInterrupt):
            with reg.scope("call", name="c") as c:
                d = c.dir("tmp", "x")
                raise KeyboardInterrupt
        self.assertFalse(d.exists())

    def test_sigterm_through_the_stop_path_releases(self) -> None:
        """run.py's handler only sets a flag; the loop unwinds at its next boundary
        (here: raising the stop, as `ActorStopped` does) through the `with` blocks."""
        stopping = {"asked": False}

        def _ask_stop(signum, _frame):
            stopping["asked"] = True

        old = signal.signal(signal.SIGTERM, _ask_stop)
        self.addCleanup(signal.signal, signal.SIGTERM, old)
        reg = self.reg()
        made = []

        class Stopped(Exception):
            pass

        with self.assertRaises(Stopped):
            with reg.scope("run", name="r") as run:
                made.append(run.dir("run-scratch", "r"))
                for n in range(100):
                    with reg.scope("iteration", name=f"it-{n}") as it:
                        made.append(it.dir("tmp", f"it-{n}"))
                        if n == 2:
                            os.kill(os.getpid(), signal.SIGTERM)
                        if stopping["asked"]:
                            raise Stopped
        self.assertTrue(stopping["asked"])
        self.assertEqual(len(made), 4)
        self.assertFalse(any(p.exists() for p in made))

    def test_nested_scopes_release_innermost_first(self) -> None:
        reg = self.reg()
        with reg.scope("run", name="r") as run:
            outer = run.dir("k", "outer")
            with reg.scope("batch", name="b") as batch:
                self.assertIs(batch.parent, run)
                mid = batch.dir("k", "mid")
                with reg.scope("iteration", name="i") as it:
                    inner = it.dir("k", "inner")
                self.assertFalse(inner.exists())
                self.assertTrue(mid.exists() and outer.exists())
            self.assertFalse(mid.exists())
            self.assertTrue(outer.exists())
        released = [Path(r["path"]).name for r in self.journal() if r["event"] == "release"]
        self.assertEqual(released, ["inner", "mid", "outer"])

    def test_leaked_child_closed_before_parent(self) -> None:
        reg = self.reg()
        with reg.scope("run", name="r") as run:
            cm = run.scope("iteration", name="leak")
            it = cm.__enter__()  # never exited: a thread that died mid-iteration
            leaked = it.dir("k", "leaked")
        self.assertFalse(leaked.exists())

    def test_keep_failed_retains_only_failed_and_next_sweep_collects(self) -> None:
        reg = self.reg(keep="failed")
        with reg.scope("iteration", name="ok") as ok:
            good = ok.dir("k", "good")
        with self.assertRaises(RuntimeError):
            with reg.scope("iteration", name="bad") as bad:
                kept = bad.dir("k", "kept")
                raise RuntimeError("boom")
        with reg.scope("iteration", name="soft") as soft:
            soft_kept = soft.dir("k", "soft")
            soft.mark_failed()
        self.assertFalse(good.exists())
        self.assertTrue(kept.exists() and soft_kept.exists())
        self.assertTrue(json.loads((kept / scratch.MARKER).read_text())["retained"])
        self.assertEqual(reg.sweep()["removed"], [])  # knob still failed: kept
        later = self.reg(run_id="run-B", keep="none")
        removed = {r["path"] for r in later.sweep()["removed"]}
        self.assertEqual(removed, {str(kept), str(soft_kept)})
        self.assertFalse(kept.exists())

    def test_name_collision_with_live_resource_refused(self) -> None:
        reg = self.reg()
        with reg.scope("run", name="r") as run:
            run.dir("k", "same")
            with self.assertRaises(FileExistsError):
                run.dir("k", "same")

    def test_unmarked_collision_refused(self) -> None:
        (self.root / "k" / "foreign").mkdir(parents=True)
        reg = self.reg()
        with reg.scope("run", name="r") as run, self.assertRaises(FileExistsError):
            run.dir("k", "foreign")
        self.assertTrue((self.root / "k" / "foreign").is_dir())


class TmpEnv(Base):
    def test_tmp_env_is_scoped_and_released(self) -> None:
        reg = self.reg()
        with reg.scope("call", name="actor") as call:
            env = call.tmp_env({"PATH": "/usr/bin"})
            tmp = Path(env["TMPDIR"])
            self.assertEqual(env["TMP"], env["TEMP"])
            self.assertEqual(env["PATH"], "/usr/bin")
            self.assertEqual(call.tmpdir(), tmp)  # once per scope
            out = subprocess.run(["python3", "-c", "import tempfile;print(tempfile.mkstemp()[1])"],
                                 env=env, capture_output=True, text=True, check=True).stdout
            self.assertEqual(Path(out.strip()).parent, tmp)
        self.assertFalse(tmp.exists())


class EarlyReleaseAndTempfile(Base):
    def test_scope_release_frees_one_resource_early(self) -> None:
        reg = self.reg()
        with reg.scope("iteration", name="i") as it:
            loser = it.dir("bestof", "author-b")
            keeper = it.dir("bestof", "author-a")
            self.assertTrue(it.release(loser))
            self.assertFalse(loser.exists())
            self.assertTrue(keeper.exists())
            with self.assertRaises(ScratchRefused):
                it.release(self.tmp / "not-mine")
        self.assertFalse(keeper.exists())
        self.assertEqual(reg.stats()["released"], 2)

    def test_adopt_tempfile_scopes_in_process_temp(self) -> None:
        import tempfile as tf
        reg = self.reg()
        before = tf.tempdir
        with reg.scope("run", name="r") as run:
            with scratch.adopt_tempfile(run) as tmp:
                with tf.TemporaryDirectory() as inner:
                    self.assertEqual(Path(inner).parent, tmp)
                fd, stray = tf.mkstemp()
                os.close(fd)
                self.assertEqual(Path(stray).parent, tmp)
            self.assertEqual(tf.tempdir, before)
        self.assertFalse(Path(stray).exists(), "a stray tempfile goes with the run scope")

    def test_ambient_registry_and_fallback(self) -> None:
        reg = self.reg()
        scratch.install(reg)
        try:
            self.assertIs(scratch.ambient(), reg)
            self.assertIs(scratch.active_scope(), reg.standing())
            with reg.scope("run", name="r") as run:
                self.assertIs(scratch.active_scope(), run)
                with reg.scope("iteration", name="i") as it, reg.scope("call", name="c"):
                    self.assertIs(scratch.current("iteration"), it)
        finally:
            scratch.uninstall(reg)
            reg.close()
        self.assertIsNone(scratch.ambient())
        fallback = scratch.registry_for(self.tmp / "fb")
        self.assertEqual(fallback.root, (self.tmp / "fb").resolve())
        self.assertIs(scratch.registry_for(self.tmp / "fb"), fallback)
        # An unwritable hint (a workspace at `/x`) falls back to system temp, not a crash.
        self.assertTrue(scratch.registry_for(Path("/proc/ak-no-such-dir")).root.is_dir())


class PipelineWiring(Base):
    """run_pool opens a batch scope and one iteration scope per draw on the lane
    thread, sweeps at each start, and hands the scope to iterate()."""

    def _drive(self, reg, gate_passes: bool, iterations: int = 3):
        from autokernel.loop import gates, test_pipeline as tp
        made, handed = [], []

        def make_gate(worker):
            def gate(hypothesis, paths):
                it = scratch.current("iteration")
                made.append(it.dir("ak-check-build", f"{worker.name}-{len(made)}"))
                return gate_passes, [gates.Verdict("compile", gate_passes, "x")]
            return gate

        real = tp.pipeline.loop_mod.iterate

        def spy(**kw):
            handed.append(kw.get("iteration_scope"))
            return real(**kw)

        scratch.install(reg)
        try:
            with mock.patch.object(tp.pipeline.loop_mod, "iterate", side_effect=spy):
                outcomes, _ = tp._drive(workers=tp._workers(2), iterations=iterations,
                                        gate=make_gate)
        finally:
            scratch.uninstall(reg)
        return outcomes, made, handed

    def test_iteration_scratch_released_and_sweeps_run(self) -> None:
        reg = self.reg()
        outcomes, made, handed = self._drive(reg, gate_passes=True)
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(len(made), 3)
        self.assertFalse(any(p.exists() for p in made))
        self.assertTrue(all(s is not None and s.level == "iteration" for s in handed))
        self.assertTrue(all(s.parent.level == "batch" for s in handed))
        self.assertEqual(reg.stats()["sweeps"], 1 + 3)  # batch start + each iteration

    def test_keep_failed_retains_failed_iteration_scratch(self) -> None:
        reg = self.reg(keep="failed")
        outcomes, made, _ = self._drive(reg, gate_passes=False, iterations=2)
        self.assertTrue(all(o.status not in {"kept", "measured_null"} for o in outcomes))
        self.assertTrue(made and all(p.exists() for p in made))
        self.assertTrue(all(json.loads((p / scratch.MARKER).read_text())["retained"]
                            for p in made))
        self.reg(run_id="run-next").sweep()
        self.assertFalse(any(p.exists() for p in made))


class Sweep(Base):
    def _crashed(self, run_id: str, pid: int, name: str) -> Path:
        reg = self.reg(run_id=run_id, pid=pid)
        it = reg.scope("iteration", name="crashed").__enter__()  # never exited
        return it.dir("k", name)

    def test_sweep_removes_only_marked_dead_owner(self) -> None:
        dead = self._crashed("run-dead", _dead_pid(), "dead")
        live_other = self._crashed("run-live", os.getppid(), "live")
        unmarked = self.root / "k" / "unmarked"
        unmarked.mkdir(parents=True)
        (unmarked / "keep.txt").write_text("x")
        outside = self.tmp / "elsewhere"
        outside.mkdir()
        (outside / scratch.MARKER).write_text("{}")

        reg = self.reg(run_id="run-now")
        result = reg.sweep()
        self.assertEqual([r["path"] for r in result["removed"]], [str(dead)])
        self.assertEqual(result["removed"][0]["reason"], "owner-dead")
        self.assertFalse(dead.exists())
        self.assertTrue(live_other.exists())
        self.assertTrue((unmarked / "keep.txt").exists())
        self.assertTrue(outside.exists())
        self.assertIn("sweep", [r["event"] for r in self.journal()])
        self.assertIn("sweep_remove", [r["event"] for r in self.journal()])

    def test_recycled_pid_is_dead(self) -> None:
        reg = self.reg(run_id="run-old")
        reg.owner["pid_start"] = "1"  # our pid, but a different process start
        it = reg.scope("iteration", name="x").__enter__()
        d = it.dir("k", "recycled")
        now = self.reg(run_id="run-now")
        self.assertEqual([r["path"] for r in now.sweep()["removed"]], [str(d)])

    def test_same_process_older_run_is_collected(self) -> None:
        old = self._crashed("run-old", os.getpid(), "old")
        now = self.reg(run_id="run-now")
        self.assertEqual(now.sweep()["removed"][0]["reason"], "stale-run-same-process")
        self.assertFalse(old.exists())

    def test_own_active_scope_not_swept_but_orphans_are(self) -> None:
        reg = self.reg()
        with reg.scope("run", name="r") as run:
            live = run.dir("k", "live")
            cm = reg.scope("iteration", name="orphan", parent=run)
            orphan_scope = cm.__enter__()
            orphan = orphan_scope.dir("k", "orphan")
            # simulate the scope vanishing from the active table without release
            reg._active.pop(orphan_scope.id)
            removed = [r["path"] for r in reg.sweep()["removed"]]
            self.assertEqual(removed, [str(orphan)])
            self.assertTrue(live.exists())

    def test_at_path_found_through_journal(self) -> None:
        reg = self.reg(run_id="run-dead", pid=_dead_pid())
        it = reg.scope("iteration", name="x").__enter__()
        elsewhere = it.dir("actor-context", "b", at=self.tmp / "ws" / "actor-context" / "b")
        now = self.reg(run_id="run-now")
        self.assertEqual([r["path"] for r in now.sweep()["removed"]], [str(elsewhere)])
        self.assertFalse(elsewhere.exists())

    def test_stale_collision_is_reclaimed(self) -> None:
        stale = self._crashed("run-dead", _dead_pid(), "same")
        (stale / "junk").write_text("x")
        reg = self.reg(run_id="run-now")
        with reg.scope("iteration", name="i") as it:
            fresh = it.dir("k", "same")
            self.assertEqual(fresh, stale)
            self.assertFalse((fresh / "junk").exists())


class Worktrees(Base):
    def setUp(self) -> None:
        super().setUp()
        self.repo, self.base = _make_repo(self.tmp)

    def test_dirty_marked_worktree_removed(self) -> None:
        reg = self.reg()
        with reg.scope("iteration", name="i") as it:
            wt = it.worktree(self.repo, self.base, "author-a")
            self.assertEqual(_git(wt, "rev-parse", "HEAD"), self.base)
            self.assertIn(str(wt), _worktrees(self.repo))
            (wt / "a.txt").write_text("modified\n")
            (wt / "untracked.bin").write_bytes(b"\0" * 1024)
            (wt / "build").mkdir()
            (wt / "build" / "o.o").write_bytes(b"\0" * 4096)
            self.assertFalse((wt / scratch.MARKER).exists())  # sidecar, not in-tree
            self.assertTrue(Path(str(wt) + scratch.SIDECAR_SUFFIX).is_file())
        self.assertFalse(wt.exists())
        self.assertNotIn(str(wt), _worktrees(self.repo))
        self.assertGreaterEqual(reg.stats()["bytes_freed"], 5120)

    def test_unmarked_worktree_refused(self) -> None:
        other = self.tmp / "unmarked-wt"
        _git(self.repo, "worktree", "add", "--detach", str(other), self.base)
        reg = self.reg()
        with self.assertRaises(ScratchRefused):
            reg.remove_worktree(self.repo, other)
        self.assertTrue(other.is_dir())
        self.assertIn(str(other), _worktrees(self.repo))
        reg.sweep()
        self.assertTrue(other.is_dir())

    def test_foreign_registry_marker_refused(self) -> None:
        mine = self.reg()
        theirs = ScratchRegistry(self.tmp / "other-root", {"run_id": "x"}, 0)
        with theirs.scope("iteration", name="i") as it:
            wt = it.worktree(self.repo, self.base, "theirs")
            with self.assertRaises(ScratchRefused):
                mine.remove_worktree(self.repo, wt)
            self.assertTrue(wt.exists())

    def test_missing_dir_admin_entry_removed_after_verification(self) -> None:
        reg = self.reg()
        with reg.scope("iteration", name="i") as it:
            wt = it.worktree(self.repo, self.base, "vanished")
            admin = Path(json.loads(Path(str(wt) + scratch.SIDECAR_SUFFIX)
                                    .read_text())["admin_dir"])
            self.assertTrue(admin.is_dir())
            subprocess.run(["rm", "-rf", str(wt)], check=True)
        self.assertFalse(admin.exists())
        self.assertNotIn(str(wt), _worktrees(self.repo))
        self.assertIn("admin_entry_removed", [r["event"] for r in self.journal()])

    def test_admin_entry_pointing_elsewhere_refused(self) -> None:
        reg = self.reg()
        it = reg.scope("iteration", name="i").__enter__()
        wt = it.worktree(self.repo, self.base, "tampered")
        admin = Path(json.loads(Path(str(wt) + scratch.SIDECAR_SUFFIX).read_text())["admin_dir"])
        subprocess.run(["rm", "-rf", str(wt)], check=True)
        (admin / "gitdir").write_text(str(self.tmp / "somewhere-else" / ".git") + "\n")
        with self.assertRaises(ScratchRefused):
            reg.remove_worktree(self.repo, wt)
        self.assertTrue(admin.is_dir())
        it.close()  # release fails, journals, never raises
        self.assertTrue(admin.is_dir())
        self.assertEqual(reg.stats()["release_failures"], 1)

    def test_dead_owner_worktree_swept(self) -> None:
        dead = self.reg(run_id="run-dead", pid=_dead_pid())
        wt = dead.scope("iteration", name="i").__enter__().worktree(
            self.repo, self.base, "crashed")
        (wt / "dirty").write_text("x")
        now = self.reg(run_id="run-now")
        self.assertEqual([r["path"] for r in now.sweep()["removed"]], [str(wt)])
        self.assertNotIn(str(wt), _worktrees(self.repo))

    def test_git_wrapper_refuses_prune_and_gc(self) -> None:
        with self.assertRaises(ScratchRefused):
            scratch._git("-C", str(self.repo), "worktree", "pr" + "une")
        with self.assertRaises(ScratchRefused):
            scratch._git("-C", str(self.repo), "g" + "c")


class Guard(Base):
    def test_ensure_free_degrade_path(self) -> None:
        reg = self.reg(min_free_bytes=100 * scratch.GB)
        with mock.patch.object(ScratchRegistry, "free_bytes", return_value=120 * scratch.GB):
            self.assertTrue(reg.ensure_free(10 * scratch.GB))
            self.assertFalse(reg.ensure_free(30 * scratch.GB))

        def best_of(n: int) -> int:  # the caller contract: False -> degrade
            return n if reg.ensure_free(n * 10 * scratch.GB) else 1

        with mock.patch.object(ScratchRegistry, "free_bytes", return_value=115 * scratch.GB):
            self.assertEqual(best_of(2), 1)
        stats = reg.stats()
        self.assertEqual(stats["guard_checks"], 3)
        self.assertEqual(stats["guard_refusals"], 2)
        self.assertIn("guard_refused", [r["event"] for r in self.journal()])

    def test_default_floor_and_cli_knobs(self) -> None:
        import argparse
        parser = argparse.ArgumentParser()
        scratch.add_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.scratch_min_free_gb, 50)
        self.assertEqual(args.scratch_keep, "none")
        reg = scratch.from_args(parser.parse_args(["--scratch-min-free-gb", "7",
                                                   "--scratch-keep", "failed"]),
                                root=self.root, owner={"run_id": "r"})
        self.assertEqual(reg.min_free_bytes, 7 * scratch.GB)
        self.assertEqual(reg.keep, "failed")
        self.assertEqual(scratch.DEFAULT_MIN_FREE_BYTES, 50 * scratch.GB)


class RunWiring(unittest.TestCase):
    def test_run_exposes_the_knobs(self) -> None:
        import contextlib
        from autokernel.loop import run
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            run.main(["--help"])
        self.assertIn("--scratch-min-free-gb", out.getvalue())
        self.assertIn("--scratch-keep", out.getvalue())

    def test_run_opens_the_run_scope_installs_and_sweeps(self) -> None:
        import inspect
        from autokernel.loop import run
        src = inspect.getsource(run.main)
        for needle in ('scratch.from_args(args, root=args.store / "scratch"',
                       'scope("run", name=scratch_run_id)', "scratch.install(",
                       "scratch.uninstall", ".sweep()", "scratch.adopt_tempfile(run_scope)"):
            self.assertIn(needle, src)


class Stats(Base):
    def test_stats_counts_and_bytes(self) -> None:
        reg = self.reg()
        with reg.scope("run", name="r") as run:
            d = run.dir("k", "a")
            (d / "blob").write_bytes(b"x" * 10_000)
            s = reg.stats(measure_live=True)
            self.assertEqual(s["allocated"], 1)
            self.assertEqual(s["live"], 1)
            self.assertGreaterEqual(s["bytes_live"], 10_000)
        s = reg.stats(measure_live=True)
        self.assertEqual((s["released"], s["live"]), (1, 0))
        self.assertGreaterEqual(s["bytes_freed"], 10_000)
        self.assertEqual(s["bytes_allocated"], s["bytes_freed"])
        dead = self.reg(run_id="d", pid=_dead_pid())
        dead.scope("iteration", name="x").__enter__().dir("k", "z")
        reg.sweep()
        s = reg.stats()
        self.assertEqual((s["sweeps"], s["sweep_removed"]), (1, 1))


class PruneGcBan(unittest.TestCase):
    """No `git worktree prune` / `git gc` anywhere in the loop package. Comments and
    docstrings explaining the ban are exempt (they run nothing), as is the ban table
    `scratch._FORBIDDEN_GIT` itself. Patterns are built so this file cannot match."""

    PRUNE = "pr" + "une"
    GC = "g" + "c"
    TEXT = re.compile(r"\bworktree\s+" + "pr" + r"une\b|\bgit\b.*\s" + "g" + r"c\b")

    def _offenders(self, path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exempt: set[int] = set()
        for node in ast.walk(tree):
            # docstrings / prose string statements
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                exempt.add(id(node.value))
            # the ban table itself
            if (isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "_FORBIDDEN_GIT" for t in node.targets)):
                exempt.update(id(n) for n in ast.walk(node.value))
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in exempt and self.TEXT.search(node.value):
                out.append(f"{path.name}:{node.lineno}: {node.value[:60]!r}")
            seq = None
            if isinstance(node, (ast.List, ast.Tuple)):
                seq = node.elts
            elif isinstance(node, ast.Call):
                seq = node.args
            if not seq:
                continue
            vals = [e.value if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    and id(e) not in exempt else None for e in seq]
            for a, b in zip(vals, vals[1:]):
                if a == "worktree" and b == self.PRUNE:
                    out.append(f"{path.name}:{node.lineno}: worktree {self.PRUNE} argv")
            if self.GC in vals:
                out.append(f"{path.name}:{node.lineno}: {self.GC} argv")
        return out

    def test_no_prune_or_gc_in_loop_package(self) -> None:
        offenders = [o for path in sorted(PKG.glob("*.py")) for o in self._offenders(path)]
        self.assertEqual(offenders, [])

    def test_scanner_catches_a_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.py"
            bad.write_text('"""Never run git worktree ' + self.PRUNE + '."""\n'
                           'import subprocess\n'
                           '# a comment about worktree ' + self.PRUNE + ' is fine\n'
                           'subprocess.run(["git", "-C", "r", "' + self.GC + '"])\n'
                           'subprocess.run(["git", "worktree", "' + self.PRUNE + '"])\n'
                           'subprocess.run("git worktree ' + self.PRUNE + '", shell=True)\n')
            found = self._offenders(bad)
            self.assertEqual(len(found), 3, found)
            self.assertTrue(all(":1:" not in f and ":3:" not in f for f in found))


# ---------------------------------------------------------------------------------------
# THE ENFORCEMENT: every file/dir-creating call in the loop package is either scratch
# allocated through `scratch.py`, or on this reviewed allowlist with a one-line reason.
# A new feature that creates scratch any other way fails `ScratchInventory` -- "the next
# feature forgets it" is caught here, not in a disk-full incident.
# ---------------------------------------------------------------------------------------

_CREATORS = frozenset({"mkdtemp", "mkstemp", "mktemp", "NamedTemporaryFile", "TemporaryDirectory",
                       "SpooledTemporaryFile", "TemporaryFile", "makedirs", "mkdir", "copytree",
                       "write_text", "write_bytes", "touch", "copyfile", "copy2",
                       "symlink_to", "hardlink_to"})
_OS_SHUTIL = frozenset({"copy", "move", "link", "symlink"})


def _creation_kind(call: ast.Call) -> str | None:
    """The file/dir-creating kind of one call, or None."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else (
        func.id if isinstance(func, ast.Name) else None)
    owner = func.value.id if (isinstance(func, ast.Attribute)
                              and isinstance(func.value, ast.Name)) else None
    if name == "open":
        if owner == "os":
            flags = ast.unparse(call.args[1]) if len(call.args) > 1 else ""
            return "os-open-create" if "O_CREAT" in flags else None
        args = call.args[1:] if isinstance(func, ast.Name) or owner == "io" else call.args
        mode = next((a.value for a in args[:1] if isinstance(a, ast.Constant)), None)
        for kw in call.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        return "open-write" if isinstance(mode, str) and set(mode) & set("wax+") else None
    if name in _CREATORS:
        return name
    if name in _OS_SHUTIL and owner in ("os", "shutil"):
        return f"{owner}.{name}"
    return None


def _argv_worktree_add(call: ast.Call) -> bool:
    for seq in [call.args] + [a.elts for a in call.args if isinstance(a, (ast.List, ast.Tuple))]:
        vals = [e.value if isinstance(e, ast.Constant) else None for e in seq]
        if any(a == "worktree" and b == "add" for a, b in zip(vals, vals[1:])):
            return True
    return False


def creation_sites(path: Path) -> list[tuple[str, int]]:
    """(`module:qualname:kind`, line) for every creating call in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[str, int]] = []

    def walk(node: ast.AST, qual: str) -> None:
        for child in ast.iter_child_nodes(node):
            q = qual
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{qual}.{child.name}" if qual else child.name
            if isinstance(child, ast.Call):
                kind = _creation_kind(child)
                if kind is None and _argv_worktree_add(child):
                    kind = "worktree-add"
                if kind is not None:
                    out.append((f"{path.name}:{q or '<module>'}:{kind}", child.lineno))
            walk(child, q)

    walk(tree, "")
    return out


#: Writes INTO a registry-allocated resource (the path was handed out by a Scope).
_IN_SCOPE = "writes inside a registry-allocated scratch resource"
#: `with tempfile...` blocks that delete themselves; in a run they land in the run
#: scope's tmp (`scratch.adopt_tempfile`), so even a hard kill leaves them swept.
_SELF_CLEANING = "self-cleaning with-block temp; lands in the run scope's tmp (adopt_tempfile)"

#: `module:qualname:kind` -> (count, reason). Counts are exact: a NEW creating call in an
#: allowlisted function fails too, so each one is reviewed.
ALLOWLIST: dict[str, tuple[int, str]] = {
    # -- actor call scratch (migrated): bundle, seat config, orchestrator inputs --------
    "actor_context.py:Bundle.seal:write_text": (1, _IN_SCOPE + " (bundle manifest)"),
    "actor_context.py:explode:mkdir": (1, _IN_SCOPE + " (bundle json tree)"),
    "actor_context.py:explode:write_text": (2, _IN_SCOPE + " (bundle json tree)"),
    "actor_context.py:materialize:mkdir": (1, _IN_SCOPE + " (bundle sections/)"),
    "actor_context.py:materialize:write_text": (2, _IN_SCOPE + " (bundle sections, INDEX)"),
    "actor_opencode_config.py:_atomic_write:mkstemp": (1, "atomic temp+rename beside a caller-given target (the call-scope seat dir)"),
    "actor_opencode_config.py:write_actor_config:mkdir": (2, "writer at a caller-given path; actors pass a call-scope seat dir"),
    "actor_opencode_config.py:write_actor_config:mkstemp": (1, "atomic temp+rename beside a caller-given target (the call-scope seat dir)"),
    "actor_opencode_config.py:write_plain_config:mkdir": (1, "writer at a caller-given path; actors pass a call-scope seat dir"),
    "actor_orchestrator.py:OrchestratorBackend.argv:mkdir": (1, "evidence: provenance sidecar dir, bound by digest into the metrics row"),
    "actor_orchestrator.py:OrchestratorBackend.argv:write_text": (3, _IN_SCOPE + " (schema/bundle/scouts request inputs)"),
    "actors.py:_run_agent_in:TemporaryFile": (2, "anonymous stdout/stderr capture inside the attempt scope's tmp"),
    "actor_metrics.py:_run_cli:TemporaryFile": (1, "anonymous (unlinked) capture file, deleted on close"),
    # -- actor evidence (kept) -----------------------------------------------------------
    "actor_metrics.py:export_session:mkdir": (1, "evidence: opencode session export, bound by digest on actor_call_metrics"),
    "actor_metrics.py:export_session:open-write": (1, "evidence: opencode session export, bound by digest on actor_call_metrics"),
    "actor_metrics.py:record_report_source:mkdir": (1, "evidence: actor call log"),
    "actor_metrics.py:record_report_source:open-write": (1, "evidence: actor call log (append)"),
    "actors.py:_persist_reply:mkdir": (1, "evidence: raw replies, bound by digest in the call record"),
    "actors.py:_persist_reply:write_bytes": (1, "evidence: raw replies, bound by digest in the call record"),
    "actors.py:_record_call:mkdir": (1, "evidence: VB-AK-SEAT call record log"),
    "actors.py:_record_call:open-write": (1, "evidence: VB-AK-SEAT call record log (append)"),
    "actors.py:_record_metrics:mkdir": (2, "evidence: actor_call_metrics log and session exports"),
    "actors.py:_record_metrics:open-write": (1, "evidence: actor_call_metrics log (append)"),
    "belief_context.py:write_receipt:mkdir": (1, "evidence: belief receipt log"),
    "belief_context.py:write_receipt:open-write": (1, "evidence: belief receipt log (append)"),
    # -- self-cleaning temp --------------------------------------------------------------
    "archive.py:keep:TemporaryDirectory": (1, _SELF_CLEANING + " (private git index)"),
    "archive.py:keep:write_text": (1, "commit message inside the self-cleaning private-index dir"),
    "cpu_quant_reference.py:check_cpu_quant_suite:TemporaryDirectory": (1, _SELF_CLEANING + " (probe build)"),
    "gdn_reference.py:check_cpu_gdn:TemporaryDirectory": (1, _SELF_CLEANING + " (probe build)"),
    "hotspots.py:profile:TemporaryDirectory": (1, _SELF_CLEANING + " (rocprof trace)"),
    "integrity.py:candidate_tree:mkstemp": (1, "temp git index unlinked in finally; lands in the run scope's tmp"),
    "surface_fold.py:validate_original:TemporaryDirectory": (1, _SELF_CLEANING + " (private git index)"),
    "surface_fold.py:validate_original:write_bytes": (1, "patch copy inside the self-cleaning private-index dir"),
    # -- resume apply check (migrated) ---------------------------------------------------
    "resume.py:_scratch_tree:mkdir": (1, _IN_SCOPE + " (resume-apply pre-image)"),
    "resume.py:_scratch_tree:write_bytes": (1, _IN_SCOPE + " (resume-apply pre-image)"),
    "resume.py:preview_op_scope:write_text": (1, _IN_SCOPE + " (resume-apply pre-image)"),
    "resume.py:ClaimLedger.__init__:mkdir": (1, "state: resume claim ledger (sqlite)"),
    # -- store evidence / state ----------------------------------------------------------
    "archive.py:_retain_bytes:mkstemp": (1, "evidence: immutable retention (temp + link in the destination dir)"),
    "archive.py:_retain_bytes:os.link": (1, "evidence: immutable retention (temp + link in the destination dir)"),
    "archive.py:retain_patch_bytes:mkdir": (1, "evidence: retained candidate patches"),
    "census.py:store_census:mkdir": (1, "evidence: GGUF census in the store"),
    "census.py:store_census:write_text": (1, "evidence: GGUF census in the store (atomic)"),
    "claim.py:hold:mkdir": (1, "state: device claim lock file"),
    "claim.py:hold:open-write": (1, "state: device claim lock file"),
    "cpu_profile.py:CpuProfileCapture._initialize:mkdir": (1, "evidence: cpu-raw capture dir, retained and referenced"),
    "cpu_profile.py:CpuProfileCapture._open:os-open-create": (1, "evidence: files of the cpu-raw capture dir"),
    "cpu_screen.py:prepare_batch:mkdir": (1, "evidence: retained CPU screen selection"),
    "direct_gpu_control.py:run_or_reopen:mkdir": (2, "evidence: GPU control declaration roots in the store"),
    "direct_historical_control.py:_collect:mkdir": (1, "evidence: historical control collection root"),
    "direct_historical_control.py:_collect_t0:mkdir": (1, "evidence: historical control T0 root"),
    "direct_historical_control.py:run_or_reopen:mkdir": (1, "evidence: historical control declaration root"),
    "dispatch_guard.py:Registry.__init__:mkdir": (1, "state: dispatch identity ledger (sqlite)"),
    "evidence_feed.py:EvidenceFeed._initialize:mkdir": (1, "evidence: evidence feed store"),
    "evidence_feed.py:EvidenceFeed._initialize:os-open-create": (1, "state: evidence feed lock"),
    "evidence_feed.py:_atomic_json:mkdir": (1, "evidence: evidence feed documents (atomic)"),
    "evidence_feed.py:_atomic_json:mkstemp": (1, "evidence: atomic temp+rename of a feed document"),
    "evidence_feed.py:_journal_snapshot_lock:os-open-create": (1, "state: journal snapshot lock"),
    "fold2_gates.py:main:mkdir": (1, "evidence: CLI --out report"),
    "fold2_gates.py:main:write_text": (2, "evidence: CLI --out report"),
    "hip_routes.py:main:mkdir": (1, "evidence: CLI --out report"),
    "hip_routes.py:main:write_text": (1, "evidence: CLI --out report"),
    "historical_trajectory.py:write:mkdir": (1, "evidence: trajectory output (atomic)"),
    "historical_trajectory.py:write:mkstemp": (1, "evidence: atomic temp+rename of the trajectory"),
    "journal_feed_owner.py:JournalFeedOwner.__init__:os-open-create": (1, "state: journal feed owner lock"),
    "lineage_beliefs.py:publish:mkdir": (1, "evidence: lineage belief sidecars"),
    "lineage_capture.py:_immutable:mkdir": (1, "evidence: immutable lineage capture"),
    "lineage_capture.py:_immutable:os-open-create": (1, "evidence: immutable lineage capture (O_EXCL)"),
    "measurement_capture.py:ArtifactStore._recover_stage:os.link": (1, "evidence: measurement artifact store"),
    "measurement_capture.py:ArtifactStore._write_locked:os.link": (1, "evidence: measurement artifact store"),
    "node_profile.py:profile_loop:mkdir": (1, "evidence: node-profile run dir in the store"),
    "node_profile.py:retain_observation:mkdir": (1, "evidence: retained node-profile observation"),
    "node_profile.py:retain_observation:write_text": (1, "evidence: retained node-profile observation"),
    "perf_cache.py:_atomic_write_json:makedirs": (1, "state: persistent perf cache (atomic)"),
    "perf_cache.py:_atomic_write_json:mkstemp": (1, "state: atomic temp+rename of the perf cache"),
    "recal_serving_floor.py:main:copy2": (1, "evidence: operator CLI backup of the replaced floor"),
    "run.py:_publish_preclaim_failure:mkdir": (1, "evidence: batch --out dir (pre-claim failure marker)"),
    "run.py:_q3_cpu_gpu_quiet_window:mkdir": (1, "state: CPU/GPU quiet-window lock"),
    "run.py:_q3_cpu_gpu_quiet_window:open-write": (1, "state: CPU/GPU quiet-window lock"),
    "run.py:main.publish_held_claims:mkdir": (1, "evidence: batch --out dir (held-claim evidence)"),
    "run.py:main:mkdir": (1, "evidence: batch --out dir (loop-run.json)"),
    "runtime_calibration.py:neutral_material:mkdir": (1, "evidence: direct-neutral executable keyed by launch snapshot"),
    "runtime_calibration.py:neutral_material:os-open-create": (1, "evidence: direct-neutral executable (O_EXCL)"),
    "seed.py:install:copyfile": (1, "state: operator seed placed in the inbox"),
    "seed.py:install:mkdir": (1, "state: operator inbox"),
    "serial_build_retention.py:_write_retry:open-write": (1, "state: retained-build retry marker (atomic)"),
    "serial_run.py:_BoundedChildOutput.finish:open-write": (1, "evidence: bounded child log tails in the batch dir"),
    "serial_run.py:_drive.request_stop:touch": (1, "state: STOP request file"),
    "serial_run.py:_drive:mkdir": (1, "evidence: serial batch directory"),
    "serial_run.py:_load_completed_or_recover:mkstemp": (1, "evidence: atomic recovery-validation temp beside its target"),
    "serial_run.py:_publish_current_run:mkdir": (1, "state: current-run pointer"),
    "serial_run.py:_publish_current_run:open-write": (1, "state: current-run pointer lock"),
    "serial_run.py:main:mkdir": (1, "evidence: serial run root"),
    "serial_run.py:main:open-write": (1, "state: serial run lock"),
    "serving_beliefs.py:_write_exact:mkdir": (1, "evidence: serving belief exports"),
    "source_build_execution.py:SourceBuildStageExecutor.__call__:mkdir": (1, "evidence: source build logs"),
    "startup_factory.py:_write_new:os-open-create": (1, "evidence: standalone startup package files (O_EXCL)"),
    "startup_factory.py:build_startup:mkdir": (1, "evidence: standalone startup package dir"),
    "status.py:write_json:mkdir": (1, "evidence: loop status / loop-run documents"),
    "status.py:write_json:mkstemp": (1, "evidence: atomic temp+rename of a status document"),
    "surface_fold.py:retain_receipt:mkdir": (1, "evidence: retained keep receipts"),
    # -- deliberately NOT migrated (lifecycle owned elsewhere, documented) ---------------
    "pool.py:provision:mkdir": (2, "persistent lane worktree/build parents: REUSED across runs, never deleted (pool.py)"),
    "pool.py:provision:worktree-add": (1, "persistent lane worktrees: reused across runs by design; deleting them lost five lanes"),
    "pool.py:promote_anchor:mkdir": (1, "state: anchor generation (the champion build), pruned by prune_anchor_generations"),
    "pool.py:promote_anchor:write_text": (1, "state: anchor generation provenance.json"),
    "pool.py:prune_anchor_generations:mkdir": (1, "private quarantine dir of the anchor pruner's own delete protocol"),
    "source_loo.py:execute_surface:mkdir": (1, "evidence: LOO result dir; omission trees retained with deletion_authorized=False"),
    "source_loo.py:execute_surface:worktree-add": (1, "evidence: LOO omission worktree retained with its result (deletion_authorized=False)"),
}


class ScratchInventory(unittest.TestCase):
    """Every creating call is scratch.py's or reviewed here; nothing else may make disk."""

    def _sites(self) -> dict[str, list[int]]:
        sites: dict[str, list[int]] = {}
        for path in sorted(PKG.glob("*.py")):
            if path.name.startswith("test_") or path.name == "scratch.py":
                continue
            for key, line in creation_sites(path):
                sites.setdefault(key, []).append(line)
        return sites

    def test_every_creating_call_is_the_registry_or_reviewed_evidence(self) -> None:
        sites = self._sites()
        problems = []
        for key, lines in sorted(sites.items()):
            allowed = ALLOWLIST.get(key)
            if allowed is None:
                problems.append(f"{key} (lines {lines}): NOT ALLOWLISTED -- allocate scratch "
                                f"through scratch.Scope (dir/file/worktree/tmp_env), or add a "
                                f"reviewed evidence entry with a one-line reason")
            elif allowed[0] != len(lines):
                problems.append(f"{key}: {len(lines)} call(s) at lines {lines}, allowlist "
                                f"reviewed {allowed[0]} -- review the new one")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_allowlist_has_no_stale_entries_and_every_reason_is_one_line(self) -> None:
        sites = self._sites()
        stale = sorted(set(ALLOWLIST) - set(sites))
        self.assertEqual(stale, [], "allowlist entries with no call left: delete them")
        for key, (count, reason) in ALLOWLIST.items():
            self.assertGreater(count, 0, key)
            self.assertTrue(reason.strip() and "\n" not in reason and len(reason) <= 140, key)

    def test_scanner_sees_every_creator_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.py"
            bad.write_text(
                "import os, shutil, subprocess, tempfile\n"
                "from pathlib import Path\n"
                "def f(p):\n"
                "    tempfile.mkdtemp()\n"
                "    Path(p).mkdir()\n"
                "    open(p, 'w')\n"
                "    Path(p).open('a')\n"
                "    open(p)\n"                               # read: not a creator
                "    os.open(p, os.O_WRONLY | os.O_CREAT)\n"
                "    shutil.copytree(p, p)\n"
                "    subprocess.run(['git', 'worktree', 'add', p])\n")
            kinds = sorted(key.rsplit(":", 1)[1] for key, _ in creation_sites(bad))
        self.assertEqual(kinds, ["copytree", "mkdir", "mkdtemp", "open-write", "open-write",
                                 "os-open-create", "worktree-add"])


if __name__ == "__main__":
    unittest.main()
