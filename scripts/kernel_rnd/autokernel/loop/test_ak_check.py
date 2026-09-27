"""`ak-check`, the author's scratch sanity-check sandbox (operator 2026-09-26), offline.

What must hold:
* TU selection: the changed .c/.cpp of the lane diff (untracked included), and for a
  changed header the TUs whose anchor depfile lists it; a TU the anchor never built is
  reported, not silently skipped;
* the compile command is the anchor's, re-pointed at the lane (source AND include dirs),
  generated-header dirs kept, dependency outputs dropped, `-o` in the scratch dir only;
* the fence: a check refuses while a tail session holds it, a tail waits for a running
  check, and a waiting tail refuses new checks (no starvation); the process-tree belt
  finds a measuring binary under the launching loop and ignores a check's own children;
* ak-check builds only in a loop-allocated (marked) scratch dir, reused per iteration
  scope, and `ensure_free` failing degrades the op test to the compile check;
* permissions: the author may run exactly `ak-check` / `ak-check --op-test`; `gcc`, `make`,
  `cmake` (alone or chained after ak-check) stay denied; planner and critic cannot run it;
* knobs off (the library default, or on without an allocator): prompt, env and config
  byte-identical; on: the one rule line, the shim on PATH, the metrics row's block.
"""
import errno
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
import unittest
from unittest import mock

from autokernel.loop import ak_check, actor_opencode_config as aoc, actors, pipeline
from autokernel.loop import test_actor_context as fx
from autokernel.loop.test_actor_seat_trim_guard import _Opencode


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout


class _Tree(unittest.TestCase):
    """A fake anchor (git root + build dir with a compile db and depfiles) and a lane."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "llama.cpp-experimental-x"
        self.build = self.root / "build-cpu"
        (self.root / ".git").mkdir(parents=True)
        (self.build / "ggml/src").mkdir(parents=True)
        self.lane = self.tmp / "workers" / "lane0"
        (self.lane / "ggml/src/ggml-cpu/iqk").mkdir(parents=True)
        (self.lane / "ggml/include").mkdir(parents=True)
        for rel, text in {"ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp": "int a;\n",
                          "ggml/src/ggml-cpu/ops.cpp": "int b;\n",
                          "ggml/src/ggml-cpu/iqk/iqk_common.h": "#pragma once\n",
                          "ggml/include/ggml.h": "#pragma once\n"}.items():
            (self.lane / rel).write_text(text)
        _git(self.lane, "init", "-q")
        _git(self.lane, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
        _git(self.lane, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
        self.db = self.build / "compile_commands.json"
        entries = []
        for rel, deps in (("ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp",
                           ["ggml/src/ggml-cpu/iqk/iqk_common.h", "ggml/include/ggml.h"]),
                          ("ggml/src/ggml-cpu/ops.cpp", ["ggml/include/ggml.h"])):
            out = f"CMakeFiles/ggml-cpu.dir/{rel.split('ggml/src/', 1)[1]}.o"
            entries.append({"directory": str(self.build / "ggml/src"), "file": str(self.root / rel),
                            "output": f"ggml/src/{out}",
                            "command": (f"/usr/bin/g++-15 -DX -I{self.root}/ggml/src/.. "
                                        f"-I{self.root}/ggml/src/ggml-cpu/iqk "
                                        f"-I{self.build}/generated -O3 -march=native "
                                        f"-MD -MT {out} -MF {out}.d "
                                        f"-o {out} -c {self.root}/{rel}")})
            dep = self.build / "ggml/src" / f"{out}.d"
            dep.parent.mkdir(parents=True, exist_ok=True)
            dep.write_text(f"{out}: {self.root}/{rel} \\\n " +
                           " \\\n ".join(f"{self.root}/{d}" for d in deps) + "\n")
        self.db.write_text(json.dumps(entries))

    def tearDown(self):
        self._tmp.cleanup()

    def units(self, changed, max_header_tus=3):
        return ak_check.select_units(changed, ak_check.load_compile_db(self.db),
                                     anchor_root=self.root, build_dir=self.build,
                                     max_header_tus=max_header_tus)


class TranslationUnitSelection(_Tree):

    def test_changed_files_include_untracked_and_ignore_clean(self):
        (self.lane / "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp").write_text("int a2;\n")
        (self.lane / "ggml/src/ggml-cpu/new.cpp").write_text("int n;\n")
        self.assertEqual(ak_check.changed_files(self.lane, "HEAD"),
                         ["ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "ggml/src/ggml-cpu/new.cpp"])

    def test_a_changed_source_selects_its_own_entry(self):
        units, notes = self.units(["ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"])
        self.assertEqual([rel for rel, _ in units], ["ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"])
        self.assertEqual(notes, [])

    def test_an_unbuilt_tu_is_reported_not_skipped_silently(self):
        units, notes = self.units(["ggml/src/ggml-cpu/new.cpp"])
        self.assertEqual(units, [])
        self.assertIn("no compile command", notes[0])

    def test_a_header_selects_the_tus_whose_depfile_lists_it(self):
        units, _ = self.units(["ggml/src/ggml-cpu/iqk/iqk_common.h"])
        self.assertEqual([rel for rel, _ in units], ["ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"])
        units, notes = self.units(["ggml/include/ggml.h"], max_header_tus=1)
        self.assertEqual(len(units), 1)
        self.assertIn("included by 2 TUs; checked 1", notes[0])

    def test_anchor_root_comes_from_cmakecache_for_builds_outside_the_tree(self):
        # DS41 2026-09-26: a keep's promoted anchor build lives in store/anchor-gen-NNN,
        # outside the source tree; the .git-parent walk returned the store and every
        # changed TU read as "no compile command" (a vacuous NOTHING/pass).
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "store" / "anchor-gen-001"
            outside.mkdir(parents=True)
            (outside / "CMakeCache.txt").write_text(
                f"FOO:STRING=bar\nCMAKE_HOME_DIRECTORY:INTERNAL={self.root}\n")
            self.assertEqual(ak_check.anchor_root_of(outside), Path(self.root))
            (outside / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/nonexistent\n")
            self.assertEqual(ak_check.anchor_root_of(outside), outside.parent)

    def test_an_unmapped_changed_source_is_an_error_never_nothing(self):
        self.assertTrue(ak_check._unchecked_sources(["ggml/src/ggml-cpu/new.cpp"]))
        self.assertFalse(ak_check._unchecked_sources(["README.md", "ggml/include/ggml.h"]))
        with mock.patch.object(ak_check, "changed_files",
                               return_value=["ggml/src/ggml-cpu/new.cpp"]):
            result = ak_check.compile_check(lane=self.lane, build_dir=self.build, db_path=self.db,
                                            scratch=Path(tempfile.mkdtemp()), base="HEAD",
                                            cpus=(0,))
        self.assertEqual(result["status"], "error")
        self.assertIn("could NOT check", result["reason"])

    def test_non_code_changes_select_nothing(self):
        self.assertEqual(self.units(["README.md"])[0], [])

    def test_touched_types_come_from_the_diff_else_the_file_family(self):
        path = self.lane / "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"
        path.write_text("int a; // DequantizerQ5K and block_q4_K\n")
        self.assertEqual(ak_check.touched_types(self.lane, "HEAD", [str(path.relative_to(self.lane))]),
                         ("q4_K", "q5_K"))
        path.write_text("int a; // layout fix\n")
        self.assertEqual(ak_check.touched_types(self.lane, "HEAD", [str(path.relative_to(self.lane))]),
                         ak_check.FILE_TYPES["iqk_gemm_kquants"])


class CompileCommandRewrite(_Tree):

    def entry(self):
        db = ak_check.load_compile_db(self.db)
        return db[f"{self.root}/ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"]

    def test_source_and_includes_move_to_the_lane_generated_dirs_stay(self):
        argv = ak_check.rewrite_command(self.entry(), anchor_root=self.root, lane=self.lane,
                                        output=self.tmp / "s/x.o")
        self.assertIn(f"{self.lane}/ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", argv)
        self.assertIn(f"-I{self.lane}/ggml/src/..", argv)
        self.assertIn(f"-I{self.lane}/ggml/src/ggml-cpu/iqk", argv)
        self.assertIn(f"-I{self.build}/generated", argv, "generated headers stay in the build")
        self.assertFalse([a for a in argv if str(self.root) + "/ggml" in a],
                         "no anchor SOURCE path survives")
        self.assertEqual(argv[0], "/usr/bin/g++-15")
        self.assertIn("-march=native", argv)

    def test_output_goes_to_scratch_and_dependency_outputs_are_dropped(self):
        out = self.tmp / "scratch/obj/x.o"
        argv = ak_check.rewrite_command(self.entry(), anchor_root=self.root, lane=self.lane,
                                        output=out)
        self.assertEqual(argv.count("-o"), 1)
        self.assertEqual(argv[argv.index("-o") + 1], str(out))
        for flag in ("-MD", "-MT", "-MF"):
            self.assertNotIn(flag, argv)
        self.assertFalse([a for a in argv if a.endswith(".o.d")])

    def test_syntax_only_has_no_output(self):
        argv = ak_check.rewrite_command(self.entry(), anchor_root=self.root, lane=self.lane,
                                        output=None)
        self.assertIn("-fsyntax-only", argv)
        self.assertNotIn("-o", argv)

    def test_relink_swaps_objects_and_writes_only_under_scratch(self):
        target_dir = self.build / "ggml/src"
        link = target_dir / "CMakeFiles/ggml-cpu.dir/link.txt"
        link.parent.mkdir(parents=True, exist_ok=True)
        (self.build / "bin").mkdir()
        (self.build / "bin/libggml-base.so.0.1").write_text("")
        link.write_text('/usr/bin/g++-15 -fPIC -Wl,--dependency-file=CMakeFiles/ggml-cpu.dir/link.d '
                        '-shared -o ../../bin/libggml-cpu.so.0.1 "CMakeFiles/ggml-cpu.dir/a.o" '
                        '"CMakeFiles/ggml-cpu.dir/b.o" ../../bin/libggml-base.so.0.1 -fopenmp\n')
        scratch = self.tmp / "scratch"
        swapped = scratch / "obj/b.o"
        seen = {}

        def fake(argv, **kw):
            seen["argv"], seen["cwd"] = argv, kw["cwd"]
            Path(argv[argv.index("-o") + 1]).write_text("")
            return 0, "", False
        with mock.patch.object(ak_check, "run_bounded", side_effect=fake):
            row = ak_check.relink(target_dir, "ggml-cpu", scratch=scratch, cpus=[0], timeout_s=5,
                                  objects={str((target_dir / "CMakeFiles/ggml-cpu.dir/b.o").resolve()):
                                           swapped})
        self.assertTrue(row["ok"])
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("-o") + 1], str(scratch / "bin/libggml-cpu.so.0.1"))
        self.assertIn(str(swapped), argv)
        self.assertIn(str((target_dir / "CMakeFiles/ggml-cpu.dir/a.o").resolve()), argv)
        self.assertIn(str((self.build / "bin/libggml-base.so.0.1").resolve()), argv)
        self.assertFalse([a for a in argv if "dependency-file" in a])
        self.assertEqual(seen["cwd"], scratch)

    def test_diagnostics_are_lane_relative_and_deduplicated(self):
        block = (f"{self.lane}/k.cpp: In instantiation of 'f<int>':\n"
                 f"{self.lane}/k.cpp:9:1:   required from here\n"
                 f"{self.lane}/k.cpp:834:40: error: '_mm512_set_m128' was not declared\n"
                 "  834 |   x = _mm512_set_m128(a, b);\n      |       ^~~\n")
        text = ak_check.diagnostics(block * 8, self.lane)
        self.assertEqual(text.count("error: '_mm512_set_m128'"), 1)
        self.assertIn("7 repeat(s)", text)
        self.assertNotIn(str(self.lane), text)


class Fence(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fence = Path(self._tmp.name) / ak_check.FENCE_DIR_NAME

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_check_refuses_while_a_tail_holds_the_fence(self):
        with ak_check.tail_fence([self.fence]):
            with self.assertRaises(ak_check.Refused):
                with ak_check.sandbox_slot(self.fence):
                    pass
        with ak_check.sandbox_slot(self.fence):
            pass   # free again after the session

    def test_a_tail_waits_for_a_running_check(self):
        entered = threading.Event()
        release = threading.Event()

        def check():
            with ak_check.sandbox_slot(self.fence):
                entered.set()
                release.wait(5)
        worker = threading.Thread(target=check)
        worker.start()
        entered.wait(5)
        timer = threading.Timer(0.3, release.set)
        timer.start()
        started = time.monotonic()
        with ak_check.tail_fence([self.fence], poll_s=0.02):
            waited = time.monotonic() - started
        worker.join()
        self.assertGreaterEqual(waited, 0.25)

    def test_a_waiting_tail_refuses_new_checks_and_times_out_bounded(self):
        slot = ak_check._open_lock(self.fence / ak_check.SLOT_LOCK)
        fcntl.flock(slot, fcntl.LOCK_SH)          # a running check
        try:
            state = {}

            def tail():
                try:
                    with ak_check.tail_fence([self.fence], wait_s=0.6, poll_s=0.02):
                        state["entered"] = True
                except ak_check.TailFenceTimeout:
                    state["timeout"] = True
            thread = threading.Thread(target=tail)
            thread.start()
            time.sleep(0.2)                         # the tail now holds the gate, waiting
            with self.assertRaises(ak_check.Refused):
                with ak_check.sandbox_slot(self.fence):
                    pass
            thread.join()
            self.assertEqual(state, {"timeout": True})
        finally:
            slot.close()

    def test_every_lane_shares_one_fence(self):
        self.assertEqual(ak_check.fence_dir(Path("/w/lane0")), ak_check.fence_dir(Path("/w/lane6")))

    def test_the_belt_finds_measurements_under_the_launching_loop_only(self):
        table = {
            10: (1, ["python", "-m", "scripts.kernel_rnd.autokernel.loop.serial_run"]),
            11: (10, ["python", "-m", "scripts.kernel_rnd.autokernel.loop.run"]),
            12: (11, ["/b/bin/llama-server", "-m", "x"]),
            13: (11, ["opencode", "run"]),
            14: (13, ["bash", "-c", "ak-check"]),
            15: (14, ["python3", "/r/ak_check.py", "--op-test"]),
            16: (15, ["/s/bin/test-backend-ops", "test"]),
            20: (1, ["/other/llama-bench"]),
        }
        found = ak_check.measuring_processes(pid=15, table=table)
        self.assertEqual(len(found), 1)
        self.assertIn("pid 12", found[0])
        del table[12]
        self.assertEqual(ak_check.measuring_processes(pid=15, table=table), [],
                         "a check's own test-backend-ops and another session's bench are not ours")
        self.assertEqual(ak_check.measuring_processes(pid=20, table=table), [], "no loop: manual run")

    def test_the_serialized_tail_enters_the_fence_per_session(self):
        entered = []

        class Fence:
            def __enter__(self):
                entered.append("in")

            def __exit__(self, *exc):
                entered.append("out")
                return False
        tail = pipeline.SerializedTail(lambda: "abc", fence=Fence)
        with tail.session("abc"):
            self.assertEqual(entered, ["in"])
        self.assertEqual(entered, ["in", "out"])
        with self.assertRaises(pipeline.Superseded):
            with tail.session("stale"):
                pass
        self.assertEqual(entered, ["in", "out"], "a superseded attempt never waits on the fence")

    def test_the_sandbox_dir_lives_in_the_run_registry_iteration_scope(self):
        """ONE registry: the provider allocates in the iteration scope `run_pool` opens
        on the lane thread of the run's installed registry (actor `call` scopes nested
        under it), the dir carries the registry's marker, and closing the iteration
        releases it."""
        from autokernel.loop import run as run_mod, scratch
        with tempfile.TemporaryDirectory() as tmp:
            reg = scratch.ScratchRegistry(Path(tmp) / "scratch", owner={"test": "sandbox"})
            args = mock.Mock(actor_author_sandbox="on")
            self.assertIs(run_mod._sandbox_scratch(args, reg), reg)
            self.assertIsNone(run_mod._sandbox_scratch(mock.Mock(actor_author_sandbox="off"),
                                                       reg))
            provide = ak_check.scratch_provider(reg, "lane0")
            batch = reg.scope("batch", name="pool-1")
            iteration = reg.scope("iteration", name="lane0-1", parent=batch).__enter__()
            with reg.scope("call", name="author"):
                path, degrade = provide()
            self.assertIsNone(degrade)
            self.assertTrue((path / ak_check.SCRATCH_MARKER).is_file())
            self.assertEqual(path.parent.parent, reg.root)
            iteration.__exit__(None, None, None)
            self.assertFalse(path.exists(), "the iteration's close releases the build dir")
            batch.close()
            reg.close()


class _Scope:
    def __init__(self, reg, sid, level="iteration", parent=None):
        self.reg, self.id, self.level, self.parent, self.closed = reg, sid, level, parent, False

    def dir(self, kind, name):
        path = self.reg.root / kind / name / self.id
        path.mkdir(parents=True)
        (path / ak_check.SCRATCH_MARKER).write_text("{}")
        self.reg.allocations.append(path)
        return path


class _Registry:
    def __init__(self, root):
        self.root, self.allocations, self.stack, self.free = Path(root), [], [], True
        self.min_free_bytes = 50 * 10 ** 9

    def current(self):
        return self.stack[-1] if self.stack else None

    def ensure_free(self, needed):
        return self.free


class ScratchContract(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reg = _Registry(self.tmp / "scratch")

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_dir_per_iteration_scope_reused_across_calls(self):
        provide = ak_check.scratch_provider(self.reg, "lane0")
        self.assertEqual(provide(), (None, None), "no iteration scope: no dir")
        self.reg.stack.append(_Scope(self.reg, "iteration-1"))
        self.reg.stack.append(_Scope(self.reg, "call-2", level="call", parent=self.reg.stack[-1]))
        first, _ = provide()
        second, _ = provide()
        self.assertEqual(first, second)
        self.assertEqual(len(self.reg.allocations), 1)
        self.assertIn(ak_check.SCRATCH_KIND, str(first))
        self.reg.stack[:] = [_Scope(self.reg, "iteration-3")]
        third, _ = provide()
        self.assertNotEqual(third, first)

    def test_low_disk_degrades_the_op_test(self):
        provide = ak_check.scratch_provider(self.reg, "lane0")
        self.reg.stack.append(_Scope(self.reg, "iteration-1"))
        self.reg.free = False
        path, degrade = provide()
        self.assertIsNotNone(path)
        self.assertIn("50 GB", degrade)

    def _main(self, *extra, env=None):
        build = self.tmp / "build"
        build.mkdir(exist_ok=True)
        (build / "compile_commands.json").write_text("[]")
        lane = self.tmp / "workers" / "lane0"
        if not (lane / ".git").exists():
            lane.mkdir(parents=True, exist_ok=True)
            _git(lane, "init", "-q")
            _git(lane, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base",
                "--allow-empty")
        out = []
        with mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch("builtins.print", side_effect=lambda text: out.append(text)):
            os.environ.pop(ak_check.ENV_SCRATCH, None) if not env or ak_check.ENV_SCRATCH not in env \
                else None
            code = ak_check.main(["--lane", str(lane), "--build-dir", str(build), *extra])
        return code, "\n".join(out)

    def test_no_scratch_or_an_unmarked_one_is_refused(self):
        code, text = self._main()
        self.assertEqual(code, ak_check.EXIT_REFUSED)
        self.assertIn("REFUSED: no scratch dir", text)
        bare = self.tmp / "bare"
        bare.mkdir()
        code, text = self._main(env={ak_check.ENV_SCRATCH: str(bare)})
        self.assertEqual(code, ak_check.EXIT_REFUSED)
        self.assertIn("builds nowhere else", text)

    def test_the_command_refuses_while_a_tail_is_active_and_logs_it(self):
        scratch = self.tmp / "s"
        scratch.mkdir()
        (scratch / ak_check.SCRATCH_MARKER).write_text("{}")
        log = self.tmp / "calls.jsonl"
        env = {ak_check.ENV_SCRATCH: str(scratch), ak_check.ENV_LOG: str(log),
               ak_check.ENV_CALL_ID: "c1"}
        with ak_check.tail_fence([ak_check.fence_dir(self.tmp / "workers" / "lane0")]):
            code, text = self._main("--op-test", env=env)
        self.assertEqual(code, ak_check.EXIT_REFUSED)
        self.assertIn("tail measurement/calibration of this campaign is active", text)
        row = json.loads(log.read_text().splitlines()[-1])
        self.assertEqual((row["status"], row["mode"], row["call_id"]), ("refused", "op-test", "c1"))

    def test_usage_summary_counts_one_calls_checks(self):
        log = self.tmp / "calls.jsonl"
        rows = [{"call_id": "a", "mode": "compile", "status": "fail", "seconds": 4.0,
                 "scratch_bytes_delta": 1000, "scratch_bytes_after": 1000},
                {"call_id": "a", "mode": "compile", "status": "pass", "seconds": 5.0,
                 "scratch_bytes_delta": 500, "scratch_bytes_after": 1500},
                {"call_id": "a", "mode": "op-test", "status": "pass", "seconds": 12.0,
                 "scratch_bytes_delta": 2000, "scratch_bytes_after": 3500},
                {"call_id": "b", "mode": "compile", "status": "pass", "seconds": 1.0}]
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = ak_check.usage_summary(log, "a")
        self.assertEqual((out["calls"], out["pass"], out["fail"]), (3, 2, 1))
        self.assertEqual(out["seconds"], 21.0)
        self.assertEqual(out["bytes_created"], 3500)
        self.assertEqual(out["scratch_bytes"], 3500)
        self.assertEqual(out["by_mode"]["op-test"], {"calls": 1, "pass": 1, "fail": 0,
                                                     "seconds": 12.0})
        self.assertEqual(ak_check.usage_summary(log, "zzz"), {"calls": 0})
        self.assertIsNone(ak_check.usage_summary(None, "a"))

    def test_no_peer_proceeds_and_logs_zero_wait(self):
        scratch = self.tmp / "s-nopeer"
        scratch.mkdir()
        (scratch / ak_check.SCRATCH_MARKER).write_text("{}")
        log = self.tmp / "calls-nopeer.jsonl"
        env = {ak_check.ENV_SCRATCH: str(scratch), ak_check.ENV_LOG: str(log),
               ak_check.ENV_CALL_ID: "c-nopeer"}
        with mock.patch.object(ak_check, "default_peer_status",
                              lambda cpus, exclude_role=ak_check.LOOP_REGION_ROLE: None):
            code, _text = self._main(env=env)
        self.assertEqual(code, ak_check.EXIT_PASS)
        row = json.loads(log.read_text().splitlines()[-1])
        self.assertEqual((row["peer_wait_s"], row["refused_peer"]), (0.0, False))

    def test_a_peer_past_the_bound_refuses_exit_2_and_records_metrics(self):
        # AK_CHECK_PEER_WAIT_S=0 makes any reported peer exceed the bound on the very
        # first check -- no real waiting needed to exercise the refusal path.
        scratch = self.tmp / "s-peer"
        scratch.mkdir()
        (scratch / ak_check.SCRATCH_MARKER).write_text("{}")
        log = self.tmp / "calls-peer.jsonl"
        env = {ak_check.ENV_SCRATCH: str(scratch), ak_check.ENV_LOG: str(log),
               ak_check.ENV_CALL_ID: "c-peer", ak_check.ENV_PEER_WAIT_S: "0"}
        with mock.patch.object(
                ak_check, "default_peer_status",
                lambda cpus, exclude_role=ak_check.LOOP_REGION_ROLE: {"q3": ["bench"]}):
            code, text = self._main(env=env)
        self.assertEqual(code, ak_check.EXIT_REFUSED)
        self.assertIn("a peer measurement holds the CPU region covering ak-check's cores", text)
        self.assertIn("retry later", text)
        row = json.loads(log.read_text().splitlines()[-1])
        self.assertEqual(row["status"], "refused")
        self.assertTrue(row["refused_peer"])
        self.assertGreaterEqual(row["peer_wait_s"], 0.0)


class Permissions(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.lane = self.tmp / "workers" / "lane0"
        self.lane.mkdir(parents=True)
        self.build = self.tmp / "anchor" / "build-cpu"
        (self.tmp / "anchor" / ".git").mkdir(parents=True)
        self.build.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def plain(self, role, **kw):
        return aoc.build_plain_config(role=role, lane=self.lane, build_dir=self.build,
                                      lane_guard=True, trim_tools=True, **kw)

    def test_the_author_runs_exactly_ak_check_and_still_no_build(self):
        for cfg, agent in ((self.plain("author", author_sandbox=True), None),
                           (lambda c: (c, c["agent"][aoc.AGENT_NAMES["author"]]["permission"]))(
                               aoc.build_actor_config(role="author", lane=self.lane,
                                                      build_dir=self.build, lane_guard=True,
                                                      replace_system_prompt=True,
                                                      author_sandbox=True))):
            oc = _Opencode(cfg, agent)
            self.assertEqual(oc.bash("ak-check"), "allow")
            self.assertEqual(oc.bash("ak-check --op-test"), "allow")
            for denied in (("gcc -c x.c",), ("make -j8",), ("cmake -B b",), ("g++ x.cpp",),
                           ("ak-check", "gcc -c x.c"), ("ak-check --op-test", "make"),
                           ("timeout 60 cmake ..",), ("ninja -C build",)):
                self.assertEqual(oc.bash(*denied), "deny", denied)
            rules = list(cfg["permission"]["bash"].items())
            self.assertEqual(rules[-2:], [("ak-check", "allow"), ("ak-check --op-test", "allow")],
                             "the allows are the LAST rules: last match wins")

    def test_planner_and_critic_cannot_run_ak_check(self):
        for role in ("planner", "critic"):
            oc = _Opencode(self.plain(role, author_sandbox=True))
            for command in ("ak-check", "ak-check --op-test", "/w/ak-check-bin/lane0/ak-check",
                            "python3 /r/ak_check.py --op-test", "timeout 9 ak-check"):
                self.assertEqual(oc.bash(command), "deny", (role, command))
        cfg = aoc.build_actor_config(role="planner", lane=self.lane, build_dir=self.build,
                                     replace_system_prompt=True, author_sandbox=True)
        self.assertEqual(_Opencode(cfg).bash("ak-check"), "deny")

    def test_off_is_byte_identical_config(self):
        for role in ("planner", "author", "critic"):
            self.assertEqual(json.dumps(self.plain(role)),
                             json.dumps(self.plain(role, author_sandbox=False)))
        self.assertEqual(aoc.seat_label("plain", lane_guard=True),
                         aoc.seat_label("plain", lane_guard=True, author_sandbox=False))
        self.assertTrue(aoc.seat_label("plain", author_sandbox=True).endswith("+ak-check"))


class AuthorCall(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "anchor"
        self.build = self.root / "build-cpu"
        (self.root / ".git").mkdir(parents=True)
        self.build.mkdir()
        self.lane = self.tmp / "workers" / "lane0"
        self.lane.mkdir(parents=True)
        self.scratch = self.tmp / "scratch" / "ak-check-build" / "lane0"
        self.scratch.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def author(self, seat, provider=None):
        seen = {}

        def run(prompt, **kw):
            seen["prompt"], seen["env"] = prompt, dict(kw.get("env") or {})
            config = (kw.get("env") or {}).get("OPENCODE_CONFIG")
            seen["config"] = Path(config).read_text() if config else None
            return '{"paths": ["ggml/src/x.c"]}'
        planner = actors.AgentPlanner(workspace=self.lane, seat=seat, sandbox_scratch=provider,
                                      backend=actors.backend_for("q/m", "high"))
        context = {"target": {"recipe": {"backend": "cpu", "build_dir": str(self.build)}}}
        with mock.patch.object(actors, "render_context", return_value=fx._real_context_text()), \
                mock.patch.object(actors, "_run_agent", side_effect=run), \
                mock.patch.object(actors.subprocess, "run",
                                  return_value=mock.Mock(stdout=" M ggml/src/x.c\n")):
            planner.author(actors.Hypothesis("akm-x", "s", "f", "ggml/src/x.c", "g"), context)
        return seen

    def test_knobs_off_is_byte_identical(self):
        base = self.author(actors.ActorSeat(bounded=False, lane_guard=True))
        off_with_allocator = self.author(actors.ActorSeat(bounded=False, lane_guard=True),
                                         provider=lambda: (self.scratch, None))
        explicit_off = self.author(actors.ActorSeat(bounded=False, lane_guard=True,
                                                    author_sandbox=False),
                                   provider=lambda: (self.scratch, None))
        def env_of(seen):
            # The per-call OPENCODE_CONFIG lives in a fresh per-call scratch dir (its
            # path is unique by construction); its NAME and bytes are what must match.
            env = dict(seen["env"])
            if "OPENCODE_CONFIG" in env:
                env["OPENCODE_CONFIG"] = Path(env["OPENCODE_CONFIG"]).name
            return env
        for other in (off_with_allocator, explicit_off):
            self.assertEqual(other["prompt"], base["prompt"])
            self.assertEqual(env_of(other), env_of(base))
            self.assertEqual(other["config"], base["config"])
        self.assertNotIn("ak-check", base["prompt"] + base["config"])

    def test_on_without_an_allocator_never_tells_the_author_to_run_it(self):
        base = self.author(actors.ActorSeat(bounded=False, lane_guard=True))
        seen = self.author(actors.ActorSeat(bounded=False, lane_guard=True, author_sandbox=True))
        self.assertEqual(seen["prompt"], base["prompt"])
        self.assertNotIn(ak_check.ENV_SCRATCH, seen["env"])
        self.assertNotIn(ak_check.ENV_CALL_ID, seen["env"])

    def test_on_adds_one_rule_the_shim_and_the_allocated_scratch(self):
        seen = self.author(actors.ActorSeat(bounded=False, lane_guard=True, author_sandbox=True),
                           provider=lambda: (self.scratch, None))
        prompt = seen["prompt"]
        self.assertEqual(prompt.count(actors.AUTHOR_SANDBOX_RULE), 1)
        self.assertIn(f"{actors._SANDBOX_ANCHOR}\n\n{actors.AUTHOR_SANDBOX_RULE}", prompt)
        env = seen["env"]
        # The shim lives in the loop-allocated check dir (released with its iteration).
        shim_dir = self.scratch / ak_check.SHIM_DIR_NAME
        self.assertTrue(env["PATH"].startswith(str(shim_dir) + os.pathsep))
        self.assertEqual(env[ak_check.ENV_SCRATCH], str(self.scratch))
        self.assertNotIn(ak_check.ENV_OP_TEST, env)
        shim = (shim_dir / "ak-check").read_text()
        self.assertIn(f"--lane {self.lane}", shim)
        self.assertIn(f"--build-dir {self.build}", shim)
        self.assertTrue(os.access(shim_dir / "ak-check", os.X_OK))
        self.assertFalse((self.lane / "ak-check").exists(), "never inside the lane")
        config = json.loads(seen["config"])
        self.assertEqual(config["permission"]["bash"]["ak-check"], "allow")
        self.assertTrue(env[actors.SEAT_ENV_ARM].endswith("+ak-check"))

    def test_low_disk_reaches_the_command_as_a_degrade(self):
        seen = self.author(actors.ActorSeat(bounded=False, author_sandbox=True),
                           provider=lambda: (self.scratch, "free disk is below the floor"))
        self.assertEqual(seen["env"][ak_check.ENV_OP_TEST], "off:free disk is below the floor")

    def test_the_metrics_row_carries_the_calls_of_this_actor_call(self):
        log = self.tmp / "actor-replies" / "ak-check-lane0.jsonl"
        log.parent.mkdir()
        log.write_text(json.dumps({"call_id": "k1", "mode": "compile", "status": "pass",
                                   "seconds": 3.0, "scratch_bytes_delta": 10,
                                   "scratch_bytes_after": 10}) + "\n")
        backend = actors.backend_for("q/m", "high")
        env = {ak_check.ENV_LOG: str(log), ak_check.ENV_CALL_ID: "k1"}
        actors._record_metrics(self.lane, backend, role="author", returncode=0, wall_s=1.0,
                               timed_out=False, before_ids=set(), collect_metrics=False,
                               schema=None, final_text=None, salvaged=False, env=env)
        row = json.loads((self.lane.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG)
                         .read_text().splitlines()[-1])
        self.assertEqual((row["ak_check"]["calls"], row["ak_check"]["pass"],
                          row["ak_check"]["bytes_created"]), (1, 1, 10))
        actors._record_metrics(self.lane, backend, role="author", returncode=0, wall_s=1.0,
                               timed_out=False, before_ids=set(), collect_metrics=False,
                               schema=None, final_text=None, salvaged=False, env={})
        row = json.loads((self.lane.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG)
                         .read_text().splitlines()[-1])
        self.assertNotIn("ak_check", row, "rows without the sandbox are unchanged")


class PeerCpuRegionStatus(unittest.TestCase):
    """`default_peer_status`: read-only region occupancy against a FAKE orchestrator
    module tree in `sys.modules` (the `test_cpu_window.py` `provider` fixture's own
    seam) -- no real orchestrator checkout, no lock ever taken by this test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lock_dir = Path(self._tmp.name)
        # cores 0-3 -> q0, cores 4-7 -> q1: enough to test overlap and disjointness.
        region_of = {0: "q0", 1: "q0", 2: "q0", 3: "q0", 4: "q1", 5: "q1", 6: "q1", 7: "q1"}
        crl = ModuleType("src.runtime.cpu_region_lock")
        crl.region_lock_path = lambda role, region: self.lock_dir / f"cpu_region.{role}.{region}.lock"
        topo = ModuleType("src.runtime.instance_topology")
        topo.ATOMIC_REGIONS = ("q0", "q1")
        topo.cores_to_regions = lambda cpus: frozenset(region_of[c] for c in cpus if c in region_of)
        patcher = mock.patch.dict(sys.modules, {"src.runtime.cpu_region_lock": crl,
                                                "src.runtime.instance_topology": topo})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _hold(self, role: str, region: str):
        path = self.lock_dir / f"cpu_region.{role}.{region}.lock"
        handle = open(path, "a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(handle.close)
        return handle

    def test_no_lock_file_at_all_is_no_peer(self):
        self.assertIsNone(ak_check.default_peer_status([0, 1]))

    def test_a_peer_role_holding_the_overlapping_region_is_reported(self):
        self._hold("bench", "q0")
        self.assertEqual(ak_check.default_peer_status([0, 1]), {"q0": ["bench"]})

    def test_a_peer_on_disjoint_cores_is_not_reported(self):
        self._hold("bench", "q1")
        self.assertIsNone(ak_check.default_peer_status([0, 1, 2, 3]))

    def test_the_loops_own_claim_is_ignored(self):
        self._hold(ak_check.LOOP_REGION_ROLE, "q0")
        self.assertIsNone(ak_check.default_peer_status([0, 1]))

    def test_the_global_pseudo_role_is_ignored(self):
        self._hold("GLOBAL", "q0")
        self.assertIsNone(ak_check.default_peer_status([0]))

    def test_a_lock_file_present_but_not_currently_flocked_is_not_a_peer(self):
        # A stale/never-acquired lock file (created, never flocked) is not a live holder.
        (self.lock_dir / "cpu_region.bench.q0.lock").touch()
        self.assertIsNone(ak_check.default_peer_status([0]))

    def test_an_explicit_exclude_role_overrides_the_default(self):
        self._hold("bench", "q0")
        self.assertIsNone(ak_check.default_peer_status([0], exclude_role="bench"))


class WaitForPeerRegion(unittest.TestCase):
    """`wait_for_peer_region`'s bounded-wait loop, against a fake status provider (a
    fake clock/sleep so "waiting" never actually sleeps): no peer proceeds at once, a
    peer that clears in time proceeds, a peer past the bound refuses with the reason."""

    class _Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

        def advance(self, seconds):
            self.t += seconds

    def test_no_peer_proceeds_without_waiting(self):
        result = ak_check.wait_for_peer_region(
            [0], wait_s=100, status_provider=lambda cpus, exclude_role: None)
        self.assertEqual(result, {"waited_s": 0.0, "refused": False, "peer": None})

    def test_a_peer_that_clears_in_time_lets_it_proceed(self):
        clock = self._Clock()
        calls = {"n": 0}

        def provider(cpus, exclude_role):
            calls["n"] += 1
            return {"q3": ["bench"]} if calls["n"] < 3 else None

        logged = []
        result = ak_check.wait_for_peer_region(
            [88], wait_s=100, poll_s=10, status_provider=provider,
            clock=clock, sleep=clock.advance, log=logged.append)
        self.assertEqual(result, {"waited_s": 20.0, "refused": False, "peer": None})
        self.assertTrue(any("waiting for a peer measurement" in line for line in logged))

    def test_a_peer_past_the_bound_refuses_with_the_reason(self):
        clock = self._Clock()
        result = ak_check.wait_for_peer_region(
            [88], wait_s=10, poll_s=3,
            status_provider=lambda cpus, exclude_role: {"q3": ["bench"]},
            clock=clock, sleep=clock.advance)
        self.assertTrue(result["refused"])
        self.assertGreaterEqual(result["waited_s"], 10)
        self.assertEqual(result["peer"], {"q3": ["bench"]})

    def test_a_peer_on_disjoint_cores_never_enters_the_wait_loop(self):
        # The region-overlap filtering itself is `default_peer_status`'s job (see
        # PeerCpuRegionStatus); here a provider that already reflects "no overlap"
        # (falsy) must make wait_for_peer_region proceed without ever sleeping.
        sleeps = []
        result = ak_check.wait_for_peer_region(
            [0], wait_s=100, status_provider=lambda cpus, exclude_role: {},
            sleep=sleeps.append)
        self.assertFalse(result["refused"])
        self.assertEqual(sleeps, [])

    def test_the_loops_own_role_is_excluded_by_default(self):
        seen = {}

        def provider(cpus, exclude_role):
            seen["exclude_role"] = exclude_role
            return None

        ak_check.wait_for_peer_region([0], wait_s=1, status_provider=provider)
        self.assertEqual(seen["exclude_role"], ak_check.LOOP_REGION_ROLE)


if __name__ == "__main__":
    unittest.main()
