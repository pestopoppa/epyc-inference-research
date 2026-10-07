#!/usr/bin/env python3
"""The scratch-resource registry: release on every exit path, sweep scope, worktree
safety, the disk guard, the prune/gc ban, stats."""
from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
import io
import json
import os
import re
import signal
import subprocess
import tempfile
import unittest
import pytest
from pathlib import Path
from unittest import mock

from autokernel.loop import scratch
from autokernel.loop.scratch import ScratchRefused, ScratchRegistry
from .test_cpu_quiet_region_backoff import native

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
    "actor_metrics.py:record_salvage_turn:mkdir": (1, "evidence: actor call log (planner salvage-turn row)"),
    "actor_metrics.py:record_salvage_turn:open-write": (1, "evidence: actor call log (append)"),
    "actor_metrics.py:record_answer_protocol:mkdir": (1, "evidence: actor call log (UFH14-B1 F2 answer-protocol row)"),
    "actor_metrics.py:record_answer_protocol:open-write": (1, "evidence: actor call log (append)"),
    "actors.py:_persist_reply:mkdir": (1, "evidence: raw replies, bound by digest in the call record"),
    "actors.py:_persist_reply:write_bytes": (1, "evidence: raw replies, bound by digest in the call record"),
    "actors.py:_record_call:mkdir": (1, "evidence: VB-AK-SEAT call record log"),
    "actors.py:_record_call:open-write": (1, "evidence: VB-AK-SEAT call record log (append)"),
    "actors.py:_record_metrics:mkdir": (2, "evidence: actor_call_metrics log and session exports"),
    "actors.py:_record_metrics:open-write": (1, "evidence: actor_call_metrics log (append)"),
    "belief_context.py:write_receipt:mkdir": (1, "evidence: belief receipt log"),
    "belief_context.py:write_receipt:open-write": (1, "evidence: belief receipt log (append)"),
    "cpu_window.py:CpuWindow._event:mkdir": (1, "evidence: CPU window event log beside the published window file"),
    "cpu_window.py:CpuWindow._event:open-write": (1, "evidence: CPU window event log (append) peers tail"),
    # -- ak-check sandbox (lane/ak-sandbox): everything it builds lives in the check dir
    #    the loop allocates per iteration (ak_check.scratch_provider -> Scope.dir) ------
    "ak_check.py:author_env:mkdir": (1, _IN_SCOPE + " (ak-check shim; no check dir: one per-lane shim beside the lane)"),
    "ak_check.py:author_env:write_text": (1, _IN_SCOPE + " (ak-check shim, temp + os.replace)"),
    "ak_check.py:compile_units.one:mkdir": (1, _IN_SCOPE + " (ak-check object cache)"),
    "ak_check.py:compile_units.one:write_text": (1, _IN_SCOPE + " (ak-check object cache key stamp)"),
    "ak_check.py:relink:mkdir": (1, _IN_SCOPE + " (relinked libraries under <check dir>/bin)"),
    "ak_check.py:mirror_sonames:symlink_to": (1, _IN_SCOPE + " (soname links under <check dir>/bin)"),
    "ak_check.py:op_test:mkdir": (1, _IN_SCOPE + " (<check dir>/bin for test-backend-ops)"),
    "ak_check.py:_open_lock:mkdir": (1, "state: ak-check cross-process fence dir beside the lane / lane lock in the check dir"),
    "ak_check.py:_open_lock:open-write": (1, "state: ak-check flock files (fence gate/slot, per-check-dir lane lock)"),
    "ak_check.py:record_call:mkdir": (1, "evidence: ak-check calls log, read back into the actor_call_metrics row"),
    "ak_check.py:record_call:open-write": (1, "evidence: ak-check calls log (append)"),
    "ak_check.py:_flock_currently_held:open-write": (1, "state: non-blocking flock probe of an EXISTING lock file (a+b after an exists check); creates nothing"),
    # -- best-of panel (lane/ak-bestof): member trees are Scope.worktree/dir -------------
    "bestof.py:AuthorPanel._harvest_metrics:mkdir": (1, "evidence: member actor-call rows moved into the LANE's actor-calls.jsonl"),
    "bestof.py:AuthorPanel._harvest_metrics:open-write": (2, "evidence: the lane's actor-calls.jsonl (append) and the member log truncation"),
    "bestof.py:AuthorPanel._write_row:mkdir": (1, "evidence: author-panel.jsonl in the lane's reply dir"),
    "bestof.py:AuthorPanel._write_row:open-write": (1, "evidence: author-panel.jsonl (append)"),
    "bestof.py:command_validator.validate:TemporaryFile": (2, "anonymous (unlinked) stdout/stderr capture of the winner check, deleted on close"),
    "archive.py:keep:TemporaryDirectory": (1, _SELF_CLEANING + " (private git index)"),
    "archive.py:keep:write_text": (1, "commit message inside the self-cleaning private-index dir"),
    "cpu_quant_reference.py:check_cpu_quant_suite:TemporaryDirectory": (1, _SELF_CLEANING + " (probe build)"),
    "cpu_fusion_reference.py:check_norm_mulmat_suite:TemporaryDirectory": (1, _SELF_CLEANING + " (probe build)"),
    "cross_target.py:append:open-write": (1, "evidence: cross-target lineage ledger (flock append beside the lane binding)"),
    "cross_target.py:enqueue_gate:mkdir": (1, "input: <lane store>/inbox, the planner's existing hypothesis channel"),
    "cross_target.py:enqueue_gate:write_text": (1, "input: one queued gate-this-keep hypothesis in the lane inbox (retired when the champion carries the keep)"),
    "kernel_coverage.py:LaunchSink.__init__:open-write": (1, "per-launch stderr capture under <store>/kernel-coverage, deleted at close"),
    "kernel_coverage.py:close_launch_sink:write_text": (1, "evidence: compacted launch kernel markers (capped per shape, pruned per build)"),
    "kernel_coverage.py:enable_capture:mkdir": (1, "evidence: <store>/kernel-coverage capture root"),
    "kernel_coverage.py:main:write_text": (1, "evidence: CLI --json verdict"),
    "kernel_coverage.py:open_launch_sink:mkdir": (1, "evidence: <store>/kernel-coverage/<build>/<shape>"),
    "kernel_coverage.py:record:mkdir": (1, "evidence: <store>/kernel-preservation verdicts"),
    "kernel_coverage.py:record:write_text": (1, "evidence: kernel-preservation verdict per keep"),
    "lane_targets.py:cross_check:mkdir": (1, "evidence: <store>/cross-target-checks"),
    "surface_validation.py:retain_dimensions:mkdir": (1, "evidence: <store>/keep-dimensions (G5)"),
    "surface_validation.py:retain_dimensions:write_text": (1, "evidence: keep-dimension record per keep (G5)"),
    "gpu_serving_profile.py:run:mkdir": (1, "evidence: GPU serving profile trace dir in the store (G2)"),
    "gpu_serving_profile.py:run:open-write": (1, "evidence: profiled server log beside its trace (G2)"),
    "gpu_serving_profile.py:retain:mkdir": (1, "evidence: retained GPU serving profile (G2)"),
    "gpu_serving_profile.py:retain:write_text": (1, "evidence: retained GPU serving profile, atomic tmp+replace (G2)"),
    "cpu_fa_reference.py:check_anchor_identity:TemporaryDirectory": (1, _SELF_CLEANING + " (FA probe build per arm)"),
    "longctx.py:Surface.ensure_slot:mkdir": (1, "evidence: long-context slot cache under the store (C1)"),
    "longctx.py:SurfaceLaunch._generate:write_text": (1, "evidence: restore-vs-fresh identity receipt, atomic tmp+replace (C1)"),
    "longctx_tools.py:_build_manifest:mkdir": (1, "evidence: operator tool writes the long-context spec where asked (C1)"),
    "longctx_tools.py:_build_manifest:write_text": (2, "evidence: operator tool writes the spec and its manifest (C1)"),
    "longctx_tools.py:_histogram:write_text": (1, "evidence: operator tool writes the production context histogram (C4)"),
    "lane_targets.py:cross_check:write_text": (1, "evidence: cross-target check record per keep"),
    "lane_targets.py:resolve:open-write": (1, "per-lane flock file beside the lane binding"),
    "runtime_identity.py:Ledger.__init__:mkdir": (1, "evidence: <store> for the runtime-treatment identity ledger"),
    "cpu_norm_reference.py:check_rms_norm_suite:TemporaryDirectory": (1, _SELF_CLEANING + " (probe build)"),
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
    # -- prospective original FA capture inputs (OP80 source integration only) -----------
    'cpu_fa_mask_capture.py:prepare:mkdir': (1, 'evidence: explicit capture-dir for prospective original FA masks; refuses nonempty prior captures and retains inputs/manifests'),
    'cpu_fa_mask_capture.py:prepare:open-write': (2, 'evidence: exclusive recipe/prompt snapshots and capture manifest bind run/source/model/tool/input hashes before governed FA capture'),
    # -- reviewed quality/calibration writers: scratch owners remain native -------------
    'calibrate_served_shapes.py:build_calibration:write_text': (1, 'writes inside registered staged calibration scratch (provenance for the separate CLI execute phase)'),
    'calibrate_served_shapes.py:execute:mkdir': (1, 'evidence: served_shape calibration record directory, read back and recipe/binary/seed validated before apply'),
    'calibrate_served_shapes.py:execute:write_text': (1, 'evidence: complete seeded/sharded NMSE record with launch, binary and region-claim provenance'),
    'calibrate_served_shapes.py:apply:mkdir': (1, 'evidence: retained served-shape manifests and generated test patch, used by the source gate'),
    'calibrate_served_shapes.py:apply:write_text': (1, 'evidence: served_shape/patch.cpp generated from the validated calibration corpus'),
    'gates.py:_cache_put:mkdir': (1, 'state: reusable PPL/token cache keyed by tool/DSO/model/prompt/launch/environment content identities'),
    'gates.py:_cache_put:write_text': (1, 'state: atomic PPL/token cache value and metadata; readers require matching content key'),
    'gates.py:_completion:write_text': (1, 'writes inside a registry-allocated completion input directory; prompt and session released together'),
    'gates.py:_run_tool:mkdir': (1, 'evidence: retained PPL/quality tool transcript directory in the caller-provided gate log root'),
    'gates.py:_run_tool:write_text': (1, 'evidence: bounded PPL/quality tool argv, return code and stdout/stderr transcript, retained on failure too'),
    'gates.py:bind_bit_exact_record:mkdir': (1, 'evidence: commit-to-bit-exact-record binding directory in the campaign store'),
    'gates.py:bind_bit_exact_record:write_text': (1, 'evidence: atomic commit binding to the retained oracle record digest'),
    'gates.py:pinned_production_reference:mkdir': (1, 'state: campaign production-reference pin directory; no production kernel write'),
    'gates.py:pinned_production_reference:write_text': (1, 'state: atomic fixed production tool/DSO/environment identity pin, revalidated on reuse'),
    'gates.py:ppl_contract_ledger_add:mkdir': (1, 'state: admitted PPL mechanism ledger directory used to keep bundle quality obligations'),
    'gates.py:ppl_contract_ledger_add:write_text': (1, 'state: atomic admitted-mechanism ledger, read before anchor target selection and bundle gating'),
    'gates.py:write_bit_exact_record:mkdir': (1, 'evidence: digest-addressed bit-exact oracle records in the campaign store'),
    'gates.py:write_bit_exact_record:write_bytes': (1, 'evidence: atomic retained bit-exact source/tree/oracle/verdict/mechanism record'),
    'new_epoch.py:start_new_anchor_epoch:mkdir': (1, 'evidence: new-epoch archive root retaining the previous bundle and journal under the existing owner lock'),
    'new_epoch.py:start_new_anchor_epoch:shutil.move': (2, 'evidence: previous bundle and journal archived under the owner lock before starting a fresh epoch'),
    'roofline_coverage.py:write_scope_gap:mkdir': (1, 'evidence: campaign scope_gap.json parent; durable stagnation report for route review'),
    'roofline_coverage.py:write_scope_gap:write_text': (1, 'evidence: durable schema/timestamp/coverage-gap report, emitted only by the stagnation hook'),
    'run.py:main.ppl_contract_bisect_build:write_text': (1, 'writes inside a run-scope registered bisect build (commit marker); build and source released with the run'),
    'run.py:main.ppl_contract_tools_build:write_text': (1, 'writes inside a run-scope registered tools build; object/artifact identity marker revalidated on reuse'),
    'run.py:fail_remeasure_request:os.link': (1, "state: atomic no-clobber restore of this process's claimed REMEASURE_REQUEST, preserving a newer request"),
    'served_shape_cases.py:apply_patch_block:write_text': (1, 'source: explicit calibration tool stages the generated case block in experimental tests/test-backend-ops.cpp'),
    'served_shape_cases.py:write_manifest:mkdir': (1, 'evidence: served-shape manifest directory, consumed and checked by the candidate source gate'),
    'served_shape_cases.py:write_manifest:write_text': (1, 'evidence: atomic retained served-shape case/bound/profile/seed manifest, used by the source gate'),
    # -- store evidence / state ----------------------------------------------------------
    "archive.py:_retain_bytes:mkstemp": (1, "evidence: immutable retention (temp + link in the destination dir)"),
    "archive.py:_retain_bytes:os.link": (1, "evidence: immutable retention (temp + link in the destination dir)"),
    "archive.py:retain_patch_bytes:mkdir": (1, "evidence: retained candidate patches"),
    "census.py:store_census:mkdir": (1, "evidence: GGUF census in the store"),
    "census.py:store_census:write_text": (1, "evidence: GGUF census in the store (atomic)"),
    "model_identity.py:_persist_divergence:mkdir": (1, "evidence: caller-selected retained original per-repeat model-identity divergence records"),
    "model_identity.py:_persist_divergence:write_text": (1, "evidence: content-named full original anchor/candidate observation arrays behind an identity refusal; no grading projection"),
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
    "fold2_gates.py:main:mkdir": (2, "evidence: CLI --out report (G0 fail and full runs)"),
    "fold2_gates.py:main:write_text": (3, "evidence: CLI --out report (G0 fail and full runs)"),
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
    "run.py:main.publish_held_claims:mkdir": (1, "evidence: batch --out dir (held-claim evidence)"),
    "run.py:main:mkdir": (2, "evidence: batch --out and original GPU child phase event directory; pending markers retain failed release refusal"),
    "gpu_phases.py:child_build:write_text": (2, "evidence: pending native child-owner marker and completed original CPU build interval; release errors retain pending refusal"),
    "runtime_calibration.py:neutral_material:mkdir": (1, "evidence: direct-neutral executable keyed by launch snapshot"),
    "runtime_calibration.py:neutral_material:os-open-create": (1, "evidence: direct-neutral executable (O_EXCL)"),
    "seed.py:install:copyfile": (1, "state: operator seed placed in the inbox"),
    "seed.py:install:mkdir": (1, "state: operator inbox"),
    "serial_build_retention.py:append_decision:open-write": (1, "evidence: append-only retention decision log in the state dir"),
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


class NewWriterScratchBoundaries(Base):
    def test_checked_cookie_excludes_only_proven_older_unreadable_processes(self):
        from . import procguard
        from .test_procguard import FakeProc
        fake = FakeProc(self.tmp / "proc")
        fake.add(100, start=5000)
        fake.add(200, start=4999)
        guard = procguard.Guard(proc_root=fake.root, identity=(100, 5000))
        token = guard.new_scope()
        real_open = open
        def denied(path, *args, **kwargs):
            if Path(path) == fake.root / "200" / "environ":
                raise PermissionError("synthetic same-uid daemon provenance refusal")
            return real_open(path, *args, **kwargs)
        with mock.patch("builtins.open", side_effect=denied):
            state = guard.scope_state(token)
            self.assertTrue(state["census_verified"], state)
            self.assertEqual(state["survivors"], [])
            # Chronology is valid only for this guard's own minted cookie.
            foreign = guard.scope_state("none.999.5000.foreign-cookie")
            self.assertFalse(foreign["census_verified"], foreign)
            for start in (5000, 5001):
                fake.add(200, start=start)
                state = guard.scope_state(token)
                self.assertFalse(state["census_verified"], state)
                self.assertTrue(any("unreadable scope provenance: 200" in error
                                    for error in state["census_errors"]))
            # A captured child remains a survivor even after its cookie is hidden.
            fake.add(200, start=5001, scope=token)
            original = procguard.read_proc(fake.root, 200)
            guard._scope_captured[token] = {original.identity: original}
            state = guard.scope_state(token)
            self.assertFalse(state["census_verified"], state)
            self.assertEqual([row["pid"] for row in state["survivors"]], [200])

    def test_no_force_worktree_removal_refuses_dirty_tree_and_releases_clean_tree(self):
        repo, head = _make_repo(self.tmp)
        reg = self.reg()
        with reg.scope("run", name="bisect") as owner:
            tree = owner.worktree(repo, head, "clean-bisect", force_remove=False)
            (tree / "a.txt").write_text("dirty fixture\n")
        self.assertTrue(tree.exists())
        self.assertIn(str(tree), _worktrees(repo))
        self.assertEqual(reg.stats()["release_failures"], 1)
        self.assertFalse(scratch.read_marker("worktree", tree)["force_remove"])
        _git(tree, "restore", "a.txt")
        reg.remove_worktree(repo, tree)
        self.assertFalse(tree.exists())
        self.assertNotIn(str(tree), _worktrees(repo))

    def test_cookie_owned_descendant_is_dead_before_scratch_release_after_timeout(self):
        import sys
        from . import procguard
        reg = self.reg()
        pid_file = self.tmp / "captured-child.pid"
        script = ("import os,subprocess,sys,time; "
                  "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                  "open(sys.argv[1],'w').write(str(child.pid)); time.sleep(60)")
        with self.assertRaises(subprocess.TimeoutExpired):
            with reg.scope("call", name="timeout") as scope:
                directory = scope.dir("owned-inputs", "timeout")
                with scratch.owned_child_env(scope) as env:
                    subprocess.run([sys.executable, "-c", script, str(pid_file)],
                                   env=env, cwd=directory, timeout=0.5,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child_pid = int(pid_file.read_text())  # exact child PID captured by this fixture
        proc = procguard.read_proc(Path("/proc"), child_pid)
        self.assertTrue(proc is None or proc.state == "Z", proc)
        self.assertFalse(directory.exists())
        self.assertTrue(scope._tmp is None or not scope._tmp.exists())
        self.assertEqual(reg.stats()["release_failures"], 0)

    def test_uncertain_child_cleanup_holds_marked_paths_even_under_later_keep_none_sweep(self):
        from . import procguard
        reg = self.reg()
        class Guard:
            @staticmethod
            def call_scope(env, *, checked=False):
                from contextlib import contextmanager
                @contextmanager
                def scope():
                    yield {**env, procguard.ENV_SCOPE: "only-this-fixture"}
                return scope()
            sweeps = 0
            proofs = 0
            @classmethod
            def sweep_scope_checked(cls, token):
                assert token == "only-this-fixture"
                cls.sweeps += 1
                return {"survivors": [], "census_verified": cls.sweeps > 1,
                        "census_errors": [] if cls.sweeps > 1 else ["temporary census refusal"]}
            @classmethod
            def scope_state(cls, token):
                assert token == "only-this-fixture"
                cls.proofs += 1
                return {"survivors": [], "census_verified": cls.proofs > 1}
            @staticmethod
            def forget_checked_scope(token):
                pass
        with mock.patch.object(procguard, "current", return_value=Guard()):
            with self.assertRaises(ScratchRefused):
                with reg.scope("call", name="uncertain") as scope:
                    directory = scope.dir("owned-inputs", "uncertain")
                    with scratch.owned_child_env(scope):
                        pass
        self.assertGreaterEqual(Guard.sweeps, 2)
        marker = scratch.read_marker("dir", directory)
        self.assertIn("release_blocked", marker)
        self.assertTrue(directory.exists())
        self.assertEqual(self.reg(run_id="next").sweep()["removed"], [])
        self.assertTrue(directory.exists())

    def test_registry_instance_names_prevent_same_root_same_scope_name_collisions(self):
        first, second = self.reg(), self.reg()
        with first.scope("call", name="ppl-completion") as a, \
                second.scope("call", name="ppl-completion") as b:
            self.assertEqual(a.id, b.id)  # the sequence alone is only registry-local
            path_a = a.dir("inputs", f"{a.registry.instance}-{a.id}")
            path_b = b.dir("inputs", f"{b.registry.instance}-{b.id}")
            self.assertNotEqual(path_a, path_b)
            self.assertNotEqual(a.tmpdir(), b.tmpdir())
            self.assertTrue(path_a.exists() and path_b.exists())
        self.assertFalse(path_a.exists() or path_b.exists())

    def test_real_compile_forwarder_preserves_owned_env_and_build_context(self):
        from autokernel.loop import gates
        tree = ast.parse((PKG / "run.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        forwarder = next(node for node in main.body if isinstance(node, ast.FunctionDef)
                         and node.name == "local_compiles")
        events = []
        @contextmanager
        def build():
            events.append("acquired")
            try:
                yield
            finally:
                events.append("released")
        from types import SimpleNamespace
        ns = {"gates": gates, "gpu_local": SimpleNamespace(build=build,
              compile_env=lambda original, **kwargs: nullcontext(original)),
              "nullcontext": nullcontext}
        exec(compile(ast.Module(body=[forwarder], type_ignores=[]),
                     "real-compile-forwarder", "exec"), ns)
        env = {"TMPDIR": "/owned/synthetic-scratch", "AK_PROC_SCOPE": "original-cookie"}
        def compiler(*args, **kwargs):
            self.assertEqual(events, ["acquired"])
            self.assertEqual(args, (Path("/source"), Path("/owned/build")))
            self.assertIs(kwargs["env"], env)
            self.assertEqual(kwargs["targets"], ("llama-ppl-contract",))
            raise ValueError("compile failure")
        with mock.patch.object(gates, "compiles", side_effect=compiler):
            with self.assertRaisesRegex(ValueError, "compile failure"):
                ns["local_compiles"](Path("/source"), Path("/owned/build"),
                    env=env, targets=("llama-ppl-contract",))
        self.assertEqual(events, ["acquired", "released"])

    def test_run_ppl_helpers_register_builds_and_clean_bisect_source_before_release(self):
        import hashlib
        from types import SimpleNamespace
        from . import gates, anchor_integrity
        repo, earlier = _make_repo(self.tmp)
        (repo / "a.txt").write_text("second\n")
        _git(repo, "commit", "-qam", "second")
        head = _git(repo, "rev-parse", "HEAD")
        store, slot = self.tmp / "state", self.tmp / "slot"
        slot.mkdir()
        reg = self.reg()
        source = ast.parse((PKG / "run.py").read_text())
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        names = ("local_compiles", "ppl_contract_tools_build", "ppl_contract_anchor_for_gate", "ppl_contract_bisect_build")
        functions = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in names]
        ns = {"Path": Path, "scratch": scratch, "gates": gates, "anchor_integrity": anchor_integrity,
              "json": json, "hashlib": hashlib, "_git": _git,
              "args": SimpleNamespace(store=store, worktree=repo),
              "recipe": SimpleNamespace(cmake_defines=lambda: ()),
              "anchor_build_jobs": lambda _recipe, _jobs: 1,
              "build_jobs": 1, "build_cpu_list": "0", "anchor_build": [slot],
              "gpu_local": None, "nullcontext": nullcontext,
              "current_anchor_commit": [head], "cor_build": [slot], "cor_commit": [head]}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "real-run-ppl-helpers", "exec"), ns)
        calls = []
        def synthetic_compile(src, dest, **kwargs):
            self.assertIsNotNone(scratch.read_marker("dir", dest))
            self.assertIn("release_blocked", scratch.read_marker("dir", dest))
            self.assertIn("AK_PROC_SCOPE", kwargs["env"])
            self.assertIsNotNone(scratch.read_marker("dir", Path(kwargs["env"]["TMPDIR"])))
            (dest / "bin").mkdir()
            for tool in gates.PPL_CONTRACT_TOOL_TARGETS:
                (dest / "bin" / tool).write_bytes(b"synthetic compiled fixture tool")
            calls.append((src, dest))
            return gates.Verdict("compile", True)
        def synthetic_artifact(build, tool):
            return hashlib.sha256((build / "bin" / tool).read_bytes()).hexdigest()
        scratch.install(reg)
        try:
            with mock.patch.object(gates, "compiles", side_effect=synthetic_compile), \
                    mock.patch.object(gates, "_build_identity", side_effect=synthetic_artifact), \
                    mock.patch.object(anchor_integrity, "object_digest", return_value="synthetic-object-identity"):
                with reg.scope("run", name="ppl") as owner:
                    tools = ns["ppl_contract_tools_build"](slot, head)
                    self.assertEqual(ns["ppl_contract_tools_build"](slot, head), tools)
                    bisect = ns["ppl_contract_bisect_build"](earlier)
                    self.assertEqual(ns["ppl_contract_bisect_build"](earlier), bisect)
                    source_tree = calls[-1][0]
                    self.assertIn(str(source_tree), _worktrees(repo))
                    self.assertFalse(scratch.read_marker("worktree", source_tree)["force_remove"])
                    self.assertEqual(len(calls), 2)
                    (tools / "bin" / gates.PPL_CONTRACT_TOOL_TARGETS[0]).write_bytes(b"changed tool")
                    with self.assertRaisesRegex(ValueError, "no longer matches"):
                        ns["ppl_contract_tools_build"](slot, head)
                self.assertFalse(tools.exists() or bisect.exists() or source_tree.exists())
                self.assertNotIn(str(source_tree), _worktrees(repo))
                self.assertTrue(slot.exists())
                self.assertEqual(reg.stats()["release_failures"], 0)
                import sys
                from . import procguard
                pid_file = self.tmp / "compile-child.pid"
                timed_paths = []
                def timed_compile(src, dest, **kwargs):
                    timed_paths.extend([src, dest])
                    self.assertIn("release_blocked", scratch.read_marker("worktree", src))
                    script = ("import subprocess,sys,time; "
                              "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                              "open(sys.argv[1],'w').write(str(child.pid));time.sleep(60)")
                    subprocess.run([sys.executable, "-c", script, str(pid_file)],
                                   env=kwargs["env"], cwd=self.tmp, timeout=0.5,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.fail("synthetic compiler should have timed out")
                with mock.patch.object(gates, "compiles", side_effect=timed_compile):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        with reg.scope("run", name="ppl-compile-timeout"):
                            ns["ppl_contract_bisect_build"](earlier)
                child = procguard.read_proc(Path("/proc"), int(pid_file.read_text()))
                self.assertTrue(child is None or child.state == "Z", child)
                self.assertTrue(timed_paths and all(not path.exists() for path in timed_paths))
                self.assertNotIn(str(timed_paths[0]), _worktrees(repo))
                class UncertainGuard:
                    @staticmethod
                    def call_scope(env, *, checked=False):
                        from contextlib import contextmanager
                        @contextmanager
                        def scope():
                            yield {**env, procguard.ENV_SCOPE: "synthetic-compile-cookie"}
                        return scope()
                    sweeps = 0
                    proofs = 0
                    @staticmethod
                    def scope_is_open(token):
                        return False
                    @classmethod
                    def sweep_scope_checked(cls, token):
                        assert token == "synthetic-compile-cookie"
                        cls.sweeps += 1
                        return {"survivors": [], "census_verified": cls.sweeps > 1}
                    @classmethod
                    def scope_state(cls, token):
                        assert token == "synthetic-compile-cookie"
                        cls.proofs += 1
                        return {"survivors": [], "census_verified": cls.proofs > 1}
                    @staticmethod
                    def forget_checked_scope(token):
                        pass
                held_paths = []
                def uncertain_compile(src, dest, **kwargs):
                    held_paths.extend([src, dest])
                    return synthetic_compile(src, dest, **kwargs)
                with mock.patch.object(procguard, "current", return_value=UncertainGuard()), \
                        mock.patch.object(gates, "compiles", side_effect=uncertain_compile):
                    with self.assertRaises(ScratchRefused):
                        with reg.scope("run", name="ppl-compile-uncertain"):
                            ns["ppl_contract_bisect_build"](earlier)
                self.assertTrue(held_paths and all(path.exists() for path in held_paths))
                self.assertIn("release_blocked", scratch.read_marker("worktree", held_paths[0]))
                self.assertIn("release_blocked", scratch.read_marker("dir", held_paths[1]))
                self.assertEqual(self.reg(run_id="later").sweep()["removed"], [])
        finally:
            scratch.uninstall(reg)
            reg.close()


    def test_a_new_cookie_cannot_clear_an_existing_scope_or_disk_safety_hold(self):
        reg = self.reg()
        with reg.scope("run", name="already-held") as owner:
            directory = owner.dir("held", "existing")
            owner.block_release("older unresolved child", "older-cookie")
            original = scratch.read_marker("dir", directory)
            with reg.scope("call", name="unrelated") as current:
                with self.assertRaisesRegex(ScratchRefused, "already has"):
                    with scratch.owned_child_env(current, protect=(owner,)):
                        self.fail("a child cannot launch against an existing hold")
            self.assertEqual(scratch.read_marker("dir", directory), original)
            self.assertFalse(owner.release(directory))
        self.assertTrue(directory.exists())
        self.assertEqual(self.reg(run_id="later").sweep()["removed"], [])

    def test_failed_preflight_marker_write_never_launches_a_child(self):
        reg = self.reg()
        writes = reg._write_marker
        entered = []
        def fail_hold(resource, path, marker):
            if marker.get("release_blocked"):
                raise OSError("synthetic durable-marker write failure")
            return writes(resource, path, marker)
        with reg.scope("call", name="failed-preflight") as scope:
            directory = scope.dir("owned-inputs", "preflight")
            with mock.patch.object(reg, "_write_marker", side_effect=fail_hold):
                with self.assertRaisesRegex(OSError, "durable-marker"):
                    with scratch.owned_child_env(scope):
                        entered.append("launched")
            self.assertEqual(entered, [])
        self.assertFalse(directory.exists())

    def test_failed_post_run_marker_write_keeps_the_pre_spawn_durable_hold(self):
        reg = self.reg()
        writes = reg._write_marker
        with self.assertRaises(ScratchRefused):
            with reg.scope("call", name="failed-clear") as scope:
                directory = scope.dir("owned-inputs", "clear")
                scope.tmpdir()  # complete allocation before injecting the post-run write fault
                def fail_clear(resource, path, marker):
                    if not marker.get("release_blocked"):
                        raise OSError("synthetic durable-fence clear failure")
                    # Also simulate the pre-existing best-effort retention write failing.
                    if marker.get("retained"):
                        raise OSError("synthetic retention marker failure")
                    return writes(resource, path, marker)
                with mock.patch.object(reg, "_write_marker", side_effect=fail_clear):
                    with scratch.owned_child_env(scope):
                        self.assertIn("release_blocked", scratch.read_marker("dir", directory))
        self.assertTrue(directory.exists())
        self.assertIn("release_blocked", scratch.read_marker("dir", directory))
        self.assertEqual(self.reg(run_id="later").sweep()["removed"], [])


@pytest.mark.parametrize("helper", ["ppl_contract_tools_build", "ppl_contract_bisect_build"])
@pytest.mark.parametrize("cleanup", ["timeout", "uncertain"])
def test_original_cpu_owner_outlives_ppl_cookie_cleanup(native, tmp_path, monkeypatch, helper, cleanup):
    """The actual helpers keep native CPU ownership through the original cookie sweep."""
    import hashlib
    import sys
    from types import SimpleNamespace
    from . import claim, gates, anchor_integrity, gpu_phases, procguard
    repo, earlier = _make_repo(tmp_path)
    (repo / "a.txt").write_text("second\n")
    _git(repo, "commit", "-qam", "second")
    head = _git(repo, "rev-parse", "HEAD")
    store, slot = tmp_path / "state", tmp_path / "slot"
    slot.mkdir()
    registry = ScratchRegistry(store / "scratch",
        {"campaign": "private-ppl-owner", "state_dir": str(store), "run_id": "original-owner"}, 0)
    local = gpu_phases.LocalPhases(cpu_list="0", quiet=False, should_stop=lambda: False,
                                   on_wait=lambda kind: pytest.fail("unexpected native wait"))
    source = ast.parse((PKG / "run.py").read_text())
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    names = {"local_compiles", "ppl_contract_tools_build", "ppl_contract_anchor_for_gate",
             "ppl_contract_bisect_build"}
    functions = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"Path": Path, "scratch": scratch, "gates": gates, "anchor_integrity": anchor_integrity,
        "json": json, "hashlib": hashlib, "_git": _git,
        "args": SimpleNamespace(store=store, worktree=repo),
        "recipe": SimpleNamespace(cmake_defines=lambda: ()), "anchor_build_jobs": lambda *_: 1,
        "build_jobs": 1, "build_cpu_list": "0", "anchor_build": [slot],
        "gpu_local": local, "nullcontext": nullcontext,
        "current_anchor_commit": [head], "cor_build": [slot], "cor_commit": [head]}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "original-ppl-owner-helper", "exec"), namespace)
    guard = procguard.Guard(store=store)
    original_sweep = guard.sweep_scope_checked
    original_cpu_owner = native[1].cpu_region_lock
    original_close = claim.HeldCpuClaim._closing
    events, paths, tokens = [], [], []
    pid_file = tmp_path / "captured-compiler-descendant.pid"
    child_pid = [None]

    def assert_native_held():
        owners = claim.observe_gpu_quiet(native[1].global_region_lock_path("q0"))["owners"]
        assert [row["pid"] for row in owners] == [os.getpid()]
        assert events.count("native_acquired") == 1 and "native_released" not in events

    def assert_child_dead():
        if child_pid[0] is not None:
            child = procguard.read_proc(Path("/proc"), child_pid[0])
            assert child is None or child.state == "Z", child

    @contextmanager
    def cpu_owner(*args, **kwargs):
        with original_cpu_owner(*args, **kwargs) as receipt:
            events.append("native_acquired")
            yield receipt
        assert_child_dead()
        events.append("native_released")

    def closing(receipt):
        assert_native_held()
        assert "sweep_done" in events
        assert_child_dead()
        events.append("close_observed")
        return original_close(receipt)

    def sweep(token):
        assert_native_held()
        assert tokens == [token]  # Only the original compiler's exact cookie.
        for kind, path in paths:
            assert path.exists()
            marker = scratch.read_marker(kind, path)
            assert "release_blocked" in marker
        events.append("sweep_held")
        if cleanup == "uncertain":
            result = {"survivors": [{"pid": -1}]}
        else:
            result = original_sweep(token)
            assert not result["survivors"]
            assert_child_dead()
        events.append("sweep_done")
        return result

    def compile_timeout(src, dest, **kwargs):
        assert_native_held()
        events.append("compile_held")
        assert kwargs["cpu_list"] == "0" and kwargs["jobs"] == 1
        assert kwargs["targets"] == gates.PROMOTION_TARGETS + gates.PPL_CONTRACT_TOOL_TARGETS
        env = kwargs["env"]
        tokens.append(env[procguard.ENV_SCOPE])
        paths.append(("dir", dest))
        if helper == "ppl_contract_bisect_build":
            paths.append(("worktree", src))
            assert scratch.read_marker("worktree", src)["force_remove"] is False
        assert scratch.read_marker("dir", Path(env["TMPDIR"])) is not None
        if cleanup == "uncertain":
            return gates.Verdict("compile", True)
        # The cookie selects the descendant even though its cwd is outside scratch.
        script = ("import subprocess,sys,time; "
                  "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                  "open(sys.argv[1],'w').write(str(child.pid));time.sleep(60)")
        try:
            subprocess.run([sys.executable, "-c", script, str(pid_file)], env=env,
                           cwd=tmp_path, timeout=.5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            child_pid[0] = int(pid_file.read_text())
            child = procguard.read_proc(Path("/proc"), child_pid[0])
            assert child is not None and child.state != "Z"
            assert child.cwd == str(tmp_path) and child.scope == tokens[0]
            raise
        pytest.fail("compiler should time out")

    monkeypatch.setattr(native[1], "cpu_region_lock", cpu_owner)
    monkeypatch.setattr(claim.HeldCpuClaim, "_closing", closing)
    monkeypatch.setattr(guard, "sweep_scope_checked", sweep)
    monkeypatch.setattr(procguard, "current", lambda: guard)
    monkeypatch.setattr(gates, "compiles", compile_timeout)
    monkeypatch.setattr(anchor_integrity, "object_digest", lambda _: "synthetic-slot-object-identity")
    scratch.install(registry)
    try:
        with pytest.raises(subprocess.TimeoutExpired if cleanup == "timeout" else ScratchRefused):
            with registry.scope("run", name="ppl-owner-order"):
                namespace[helper](slot, head) if helper == "ppl_contract_tools_build" else namespace[helper](earlier)
        assert events[0:2] == ["native_acquired", "compile_held"]
        assert events[-2:] == ["close_observed", "native_released"]
        assert_child_dead()
        if cleanup == "timeout":
            assert len(local.closed_phases()) == 1
        else:
            with pytest.raises(claim.ClaimRefused, match="capture failed"):
                local.closed_phases()
        assert native[0].lock_owners() == {}
        with original_cpu_owner("private-release-probe", {"q0"}, timeout_s=.1):
            pass
        if cleanup == "timeout":
            assert all(not path.exists() for _, path in paths)
        else:
            assert all(path.exists() and "release_blocked" in scratch.read_marker(kind, path)
                       for kind, path in paths)
            later = ScratchRegistry(store / "scratch",
                {"campaign": "private-ppl-owner", "state_dir": str(store), "run_id": "later"}, 0)
            try:
                assert later.sweep()["removed"] == []
            finally:
                later.close()
            assert all(path.exists() for _, path in paths)
        assert slot.exists() and _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    finally:
        scratch.uninstall(registry)
        registry.close()
