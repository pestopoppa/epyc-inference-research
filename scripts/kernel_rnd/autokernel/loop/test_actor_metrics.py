"""actor_metrics.py: opencode-export parsing, session collection, and the summarizer.

Fixtures under `testdata/actor_metrics/` are REAL opencode artifacts, not synthesized:

* `ds41_c20c_plain_export.json` -- the actual `opencode export` of the DS41-C20c
  plain-seat session (`/mnt/raid0/llm/tmp/ak-seat-ab/plain-ses_f2cdb6f1effeKFxMQILnVhjMxl.json`).
  It reproduces the numbers the seat A/B reported by hand: 23 steps, 25 tool calls
  (bash:11, glob:1, read:13), 1 compaction, 57,702 decoded tokens.
* `truncated_pipe_export.json` -- the actual 98,304-byte capture of the SAME session
  read through a PIPE (opencode/Bun exits without draining it): invalid JSON,
  "Unterminated string ...". This is the real pitfall #2 evidence, used here to
  exercise the metrics-collection-failure path.
* `salvaged_rc1_run7_0955.stdout` -- the real DS41 run-7 planner stdout from
  2026-09-24 09:55 (`(res.stderr || "").trim is not a function`): `opencode run`
  exited 1, but stdout ends with a complete HYPOTHESIS_SCHEMA object. Not an
  export -- raw CLI stdout, used to exercise the salvage-detection integration.

No inference: every test here is a pure function over a file already on disk, or a
mocked subprocess. Nothing calls the real `opencode` binary.
"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actor_metrics

FIXTURES = Path(__file__).parent / "testdata" / "actor_metrics"
PLAIN_EXPORT = FIXTURES / "ds41_c20c_plain_export.json"
TRUNCATED_EXPORT = FIXTURES / "truncated_pipe_export.json"


class ParseExportReproducesC20c(unittest.TestCase):
    """DS41-C20c (2026-09-24): plain seat, 31.9 min, 23 steps, 25 tool calls,
    57.7k decoded tokens, 1 compaction, schema-valid -- reproduced here from the
    real export still on disk."""

    def test_c20c_plain_seat_numbers(self):
        stats = actor_metrics.parse_export(PLAIN_EXPORT)
        self.assertEqual(stats["steps"], 23)
        self.assertEqual(stats["tool_calls"], 25)
        self.assertEqual(stats["tools"], {"bash": 11, "glob": 1, "read": 13})
        self.assertEqual(stats["compactions"], 1)
        self.assertEqual(stats["decoded_tokens"], 57702)
        self.assertEqual(stats["prompt_tokens"], 175656)
        self.assertEqual(stats["cache_read_tokens"], 1228302)
        self.assertEqual(stats["context_first_tokens"], 39192)
        self.assertEqual(stats["context_max_tokens"], 93443)
        self.assertEqual(stats["session_id"], "ses_f2cdb6f1effeKFxMQILnVhjMxl")

    def test_a_call_with_a_compaction_is_counted(self):
        """The same real export carries exactly one auto-compaction (an
        overflow mid-session) -- this is the "call with a compaction" case."""
        stats = actor_metrics.parse_export(PLAIN_EXPORT)
        self.assertEqual(stats["compactions"], 1)
        # The steps/tokens totals above are POST-compaction totals over the whole
        # session -- a compaction does not zero or exclude anything already counted.
        self.assertGreater(stats["steps"], 0)

    def test_truncated_pipe_capture_raises(self):
        """The real 98,304-byte pipe-truncated capture of this same session
        (pitfall #2): invalid JSON, must raise, never silently return zeros."""
        with self.assertRaises(json.JSONDecodeError):
            actor_metrics.parse_export(TRUNCATED_EXPORT)


class Collect(unittest.TestCase):
    """`collect()` orchestrates list -> export -> parse -> total, and must never
    raise regardless of what opencode does."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name) / "lane"
        self.workspace.mkdir()
        self.replies = Path(self._tmp.name) / "actor-replies"

    def tearDown(self):
        self._tmp.cleanup()

    def _patch_export(self, source: Path):
        """Stand in for `opencode export`: copy a fixture to the requested path
        instead of shelling out."""
        def fake_export(workspace, session_id, out_path, *, timeout_s=60.0):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(source.read_bytes())
        return mock.patch.object(actor_metrics, "export_session", side_effect=fake_export)

    def test_happy_path_totals_match_c20c(self):
        with mock.patch.object(actor_metrics, "list_session_ids",
                               return_value={"ses_new"}), \
             self._patch_export(PLAIN_EXPORT):
            result = actor_metrics.collect(self.workspace, set(), self.replies, stamp="20260924T000000")
        self.assertIsNone(result["metrics_error"])
        self.assertEqual(result["session_ids"], ["ses_new"])
        self.assertEqual(result["primary_session_id"], "ses_f2cdb6f1effeKFxMQILnVhjMxl")
        self.assertEqual(result["totals"]["steps"], 23)
        self.assertEqual(result["totals"]["tool_calls"], 25)
        self.assertEqual(result["totals"]["decoded_tokens"], 57702)
        self.assertEqual(result["totals"]["compactions"], 1)
        self.assertEqual(result["context_first_tokens"], 39192)
        self.assertEqual(result["context_max_tokens"], 93443)
        # The export was written into the replies dir, referenced by a file_ref.
        exported = self.replies / "20260924T000000-export-ses_new.json"
        self.assertTrue(exported.is_file())
        self.assertEqual(result["sessions"][0]["export"]["path"], str(exported))

    def test_no_new_session_is_reported_not_raised(self):
        with mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            result = actor_metrics.collect(self.workspace, set(), self.replies, stamp="s")
        self.assertIn("no new opencode session", result["metrics_error"])
        self.assertEqual(result["sessions"], [])

    def test_a_truncated_export_is_a_metrics_error_never_a_raise(self):
        """A metrics-collection failure (here: the real pipe-truncation pitfall)
        must never propagate -- it is `metrics_error`, not an exception."""
        with mock.patch.object(actor_metrics, "list_session_ids",
                               return_value={"ses_truncated"}), \
             self._patch_export(TRUNCATED_EXPORT):
            result = actor_metrics.collect(self.workspace, set(), self.replies, stamp="s")
        self.assertIsNotNone(result["metrics_error"])
        self.assertIsNone(result["totals"])
        self.assertEqual(result["sessions"], [])

    def test_opencode_binary_missing_is_a_metrics_error(self):
        with mock.patch.object(actor_metrics, "list_session_ids",
                               return_value={"ses_new"}), \
             mock.patch.object(actor_metrics, "export_session",
                               side_effect=FileNotFoundError("opencode not found")):
            result = actor_metrics.collect(self.workspace, set(), self.replies, stamp="s")
        self.assertIn("FileNotFoundError", result["metrics_error"])

    def test_list_session_ids_never_raises_on_a_broken_subprocess(self):
        with mock.patch.object(actor_metrics.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired(cmd=["opencode"], timeout=1)):
            self.assertEqual(actor_metrics.list_session_ids(self.workspace), set())
        with mock.patch.object(actor_metrics.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0, stdout="not json")):
            self.assertEqual(actor_metrics.list_session_ids(self.workspace), set())


    def test_list_session_ids_keeps_only_this_workspace_sessions(self):
        """`opencode session list` is project-scoped: a DS41 lane's listing carried
        run-5/7/8 lanes and the seat A/B lane together (2026-09-24)."""
        rows = [{"id": "ses_here", "directory": str(self.workspace)},
                {"id": "ses_other_lane", "directory": "/elsewhere/workers/lane1"},
                {"id": "ses_old_opencode"}]
        with mock.patch.object(actor_metrics.subprocess, "run",
                               return_value=subprocess.CompletedProcess(
                                   [], 0, stdout=json.dumps(rows))):
            self.assertEqual(actor_metrics.list_session_ids(self.workspace),
                             {"ses_here", "ses_old_opencode"})

    def test_list_session_ids_matches_a_symlinked_workspace_by_its_real_path(self):
        real = self.workspace.parent / "real-lane"
        real.mkdir()
        link = self.workspace.parent / "link-lane"
        link.symlink_to(real, target_is_directory=True)
        rows = [{"id": "ses_resolved", "directory": str(real)}]
        with mock.patch.object(actor_metrics.subprocess, "run",
                               return_value=subprocess.CompletedProcess(
                                   [], 0, stdout=json.dumps(rows))):
            self.assertEqual(actor_metrics.list_session_ids(link), {"ses_resolved"})


STORE_STDERR = (FIXTURES / "store_error_run9c_0759.stderr").read_text()


def _cli(script):
    """A `subprocess.run` stand-in for opencode's CLI: `script` is a list of
    (returncode, stdout, stderr) consumed one per call. stderr is written to the
    handle the caller passed (actor_metrics captures it to a FILE, never a pipe)."""
    calls = []

    def run(argv, **kw):
        rc, out, err = script[len(calls)]
        calls.append(list(argv))
        if err and hasattr(kw.get("stderr"), "write"):
            kw["stderr"].write(err)
        handle = kw.get("stdout")
        if out and hasattr(handle, "write"):
            handle.write(out)
            out = None
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr=None)
    return run, calls


class SessionScopingByCreationTime(unittest.TestCase):
    """DS41 run 9c, 2026-09-25 07:59:47Z: the author call died on a locked store, its
    BEFORE listing had died the same way (empty set), and the AFTER listing returned
    the planner's session from 07:21 -- so the failed call was credited 29 steps. A
    call's sessions are the ones CREATED at or after the call started."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name) / "lane"
        self.workspace.mkdir()
        self.replies = Path(self._tmp.name) / "actor-replies"
        self.start_ms = 1790323175000   # 2026-09-25T07:59:35Z, the author call
        self.rows = [
            {"id": "ses_planner", "directory": str(self.workspace), "created": 1790320875000},
            {"id": "ses_author", "directory": str(self.workspace), "created": self.start_ms + 900},
            {"id": "ses_no_created", "directory": str(self.workspace)},
            {"id": "ses_other_lane", "directory": "/elsewhere", "created": self.start_ms + 5},
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_sessions_created_at_or_after_the_start_are_listed(self):
        run, _ = _cli([(0, json.dumps(self.rows), "")] * 2)
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            self.assertEqual(actor_metrics.list_session_ids(
                self.workspace, created_since_ms=self.start_ms), {"ses_author"})
            self.assertEqual(actor_metrics.list_session_ids(self.workspace),
                             {"ses_planner", "ses_author", "ses_no_created"},
                             "without a start time the historical filter is unchanged")

    def test_an_exactly_simultaneous_creation_counts(self):
        rows = [{"id": "ses_edge", "directory": str(self.workspace), "created": self.start_ms}]
        run, _ = _cli([(0, json.dumps(rows), "")])
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            self.assertEqual(actor_metrics.list_session_ids(
                self.workspace, created_since_ms=self.start_ms), {"ses_edge"})

    def test_collect_never_credits_an_older_session_when_the_before_listing_failed(self):
        """The run 9c shape: before_ids is EMPTY (that listing failed), the after
        listing holds only the planner's session."""
        rows = [self.rows[0]]
        run, calls = _cli([(0, json.dumps(rows), "")])
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run), \
             mock.patch.object(actor_metrics, "export_session",
                               side_effect=AssertionError("nothing to export")):
            result = actor_metrics.collect(self.workspace, set(), self.replies, stamp="s",
                                           started_at=self.start_ms / 1000)
        self.assertEqual(result["session_ids"], [])
        self.assertIsNone(result["totals"])
        self.assertIn("none created at or after", result["metrics_error"])
        self.assertEqual(len(calls), 1)

    def test_collect_exports_the_session_the_call_created(self):
        run, _ = _cli([(0, json.dumps(self.rows), "")])

        def fake_export(workspace, session_id, out_path, *, timeout_s=60.0):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(PLAIN_EXPORT.read_bytes())

        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run), \
             mock.patch.object(actor_metrics, "export_session", side_effect=fake_export):
            result = actor_metrics.collect(self.workspace, set(), self.replies, stamp="s",
                                           started_at=self.start_ms / 1000)
        self.assertIsNone(result["metrics_error"])
        self.assertEqual(result["session_ids"], ["ses_author"])

    def test_a_failed_after_listing_is_reported_as_such_not_as_no_session(self):
        run, _ = _cli([(1, "", "Error: boom")])
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            result = actor_metrics.collect(self.workspace, set(), self.replies, stamp="s",
                                           started_at=self.start_ms / 1000)
        self.assertIn("session list failed", result["metrics_error"])
        self.assertIn("boom", result["metrics_error"])


class StoreBusyRetry(unittest.TestCase):
    """The collector's own opencode processes hit the same locked store (run 9c): a
    short, bounded busy-wait on a store error only."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name) / "lane"
        self.workspace.mkdir()
        self.slept = []
        self._p = mock.patch.object(actor_metrics, "_sleep", side_effect=self.slept.append)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def test_the_real_run9c_stderr_is_a_store_error(self):
        self.assertTrue(actor_metrics.is_store_error(1, STORE_STDERR))
        self.assertFalse(actor_metrics.is_store_error(0, STORE_STDERR), "rc 0 is never one")
        self.assertFalse(actor_metrics.is_store_error(1, "Error: model not found"))
        for text in ("SQLITE_BUSY: database is locked", "database is locked"):
            self.assertTrue(actor_metrics.is_store_error(1, text))

    def test_session_list_retries_a_locked_store_then_succeeds(self):
        rows = [{"id": "ses_a", "directory": str(self.workspace)}]
        run, calls = _cli([(1, "", STORE_STDERR), (1, "", STORE_STDERR),
                           (0, json.dumps(rows), "")])
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            self.assertEqual(actor_metrics.list_session_ids(self.workspace, strict=True),
                             {"ses_a"})
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.slept, list(actor_metrics.STORE_RETRY_SLEEPS_S[:2]))

    def test_session_list_gives_up_after_a_bounded_wait(self):
        n = len(actor_metrics.STORE_RETRY_SLEEPS_S) + 1
        run, calls = _cli([(1, "", STORE_STDERR)] * n)
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            with self.assertRaises(RuntimeError):
                actor_metrics.list_session_ids(self.workspace, strict=True)
            self.assertEqual(len(calls), n)
        self.assertEqual(self.slept, list(actor_metrics.STORE_RETRY_SLEEPS_S))

    def test_a_non_store_failure_is_not_retried(self):
        run, calls = _cli([(1, "", "Error: unknown flag")])
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            self.assertEqual(actor_metrics.list_session_ids(self.workspace), set())
        self.assertEqual((len(calls), self.slept), (1, []))

    def test_export_retries_a_locked_store_and_keeps_only_the_good_attempt(self):
        out = self.workspace.parent / "replies" / "x-export.json"
        run, calls = _cli([(1, "partial", STORE_STDERR), (0, '{"messages": []}', "")])
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            actor_metrics.export_session(self.workspace, "ses_a", out)
        self.assertEqual(out.read_text(), '{"messages": []}')
        self.assertEqual(len(calls), 2)

    def test_export_failure_still_raises(self):
        out = self.workspace.parent / "replies" / "x-export.json"
        run, _ = _cli([(2, "", "Error: session not found")])
        with mock.patch.object(actor_metrics.subprocess, "run", side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                actor_metrics.export_session(self.workspace, "ses_a", out)


class BundleAccess(unittest.TestCase):
    """Variable-mode context: which bundle files the actor's tools named."""

    def _export(self, tools):
        parts = [{"type": "tool", "tool": name, "state": {"input": args, "output": "x"}}
                 for name, args in tools]
        return {"messages": [{"info": {"role": "assistant", "sessionID": "ses_1",
                                       "tokens": {"input": 10, "output": 5,
                                                  "cache": {"read": 0, "write": 0}}},
                              "parts": parts}]}

    def test_reads_greps_and_bash_on_bundle_files_are_counted_by_tool_and_path(self):
        bundle = "/w/actor-context/20260925T010203-planner-abc123"
        data = self._export([
            ("read", {"filePath": f"{bundle}/INDEX.md"}),
            ("read", {"filePath": f"{bundle}/sections/05-node_profile.md", "offset": 1}),
            ("grep", {"pattern": "MUL_MAT", "path": f"{bundle}/sections/09-inbox.md"}),
            ("bash", {"command": f"sed -n 1,40p {bundle}/json/target/recipe.json | head"}),
            ("read", {"filePath": "/w/lane/ggml/src/ggml-cpu/ops.cpp"}),
            ("glob", {"pattern": "**/*.md", "path": bundle}),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.json"
            path.write_text(json.dumps(data))
            stats = actor_metrics.parse_export(path)
        self.assertEqual(stats["bundle_tool_calls"], 4)
        self.assertEqual(stats["bundle_tools"], {"read": 2, "grep": 1, "bash": 1})
        self.assertEqual(stats["bundle_paths"], ["INDEX.md", "json/target/recipe.json",
                                                 "sections/05-node_profile.md",
                                                 "sections/09-inbox.md"])
        self.assertEqual(stats["tool_calls"], 6)

    def test_an_inline_call_touches_no_bundle(self):
        stats = actor_metrics.parse_export(PLAIN_EXPORT)
        self.assertEqual((stats["bundle_tool_calls"], stats["bundle_tools"],
                          stats["bundle_paths"]), (0, {}, []))


class Summarize(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _row(self, role, wall_s, steps, tool_calls, decoded, compactions, **extra):
        row = {"schema": actor_metrics.METRICS_SCHEMA, "role": role, "wall_s": wall_s,
              "opencode": {"totals": {"steps": steps, "tool_calls": tool_calls,
                                      "decoded_tokens": decoded, "compactions": compactions}}}
        row.update(extra)
        return row

    def _write(self, path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_medians_and_totals_per_role(self):
        worker_a = self.state_dir / "targets/t1/workers/actor-replies/actor-calls.jsonl"
        worker_b = self.state_dir / "targets/t2/workers/actor-replies/actor-calls.jsonl"
        self._write(worker_a, [
            self._row("planner", 100.0, 10, 5, 1000, 0),
            self._row("planner", 200.0, 20, 15, 3000, 1),
            {"schema": "epyc.autokernel.actor_call.v1", "role": "planner"},  # v1 row: ignored
            "not json",
        ])
        self._write(worker_b, [
            self._row("critic", 50.0, 2, 1, 100, 0, schema_valid=False, repair_ran=True),
            self._row("planner", 300.0, 30, 25, 5000, 0, salvaged=True, metrics_error="x"),
        ])
        report = actor_metrics.summarize(self.state_dir)
        planner = report["roles"]["planner"]
        self.assertEqual(planner["calls"], 3)
        self.assertEqual(planner["wall_s_total"], 600.0)
        self.assertEqual(planner["wall_s_median"], 200.0)
        self.assertEqual(planner["steps_total"], 60.0)
        self.assertEqual(planner["tool_calls_total"], 45.0)
        self.assertEqual(planner["decoded_tokens_total"], 9000.0)
        self.assertEqual(planner["compactions_total"], 1.0)
        self.assertEqual(planner["salvaged"], 1)
        self.assertEqual(planner["metrics_errors"], 1)
        critic = report["roles"]["critic"]
        self.assertEqual(critic["calls"], 1)
        self.assertEqual(critic["schema_invalid"], 1)
        self.assertEqual(critic["repair_ran"], 1)
        self.assertEqual(len(report["files"]), 2)

    def test_rows_with_a_seat_arm_are_also_grouped_per_role_and_arm(self):
        worker = self.state_dir / "actor-replies/actor-calls.jsonl"
        self._write(worker, [
            self._row("planner", 100.0, 10, 5, 1000, 1, seat_arm="plain"),
            self._row("planner", 50.0, 6, 4, 800, 0, seat_arm="plain+ctx-variable"),
            self._row("planner", 70.0, 8, 4, 900, 0, seat_arm="plain+ctx-variable"),
            self._row("critic", 10.0, 1, 0, 10, 0),
        ])
        report = actor_metrics.summarize(self.state_dir)
        self.assertEqual(sorted(report["role_arms"]),
                         ["planner|plain", "planner|plain+ctx-variable"])
        self.assertEqual(report["role_arms"]["planner|plain+ctx-variable"]["wall_s_median"], 60.0)
        self.assertEqual(report["roles"]["planner"]["calls"], 3)

    def test_empty_state_dir_reports_no_roles(self):
        report = actor_metrics.summarize(self.state_dir)
        self.assertEqual(report["roles"], {})
        self.assertEqual(report["files"], [])

    def test_cli_main_prints_json(self):
        worker = self.state_dir / "actor-replies/actor-calls.jsonl"
        self._write(worker, [self._row("author", 42.0, 3, 2, 500, 0)])
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = actor_metrics.main([str(self.state_dir)])
        self.assertEqual(rc, 0)
        printed = json.loads(buf.getvalue())
        self.assertEqual(printed["roles"]["author"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
