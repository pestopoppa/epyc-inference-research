#!/usr/bin/env python3
"""The scratch-resource registry: release on every exit path, sweep scope, worktree
safety, the disk guard, the prune/gc ban, stats."""
from __future__ import annotations

import ast
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


if __name__ == "__main__":
    unittest.main()
