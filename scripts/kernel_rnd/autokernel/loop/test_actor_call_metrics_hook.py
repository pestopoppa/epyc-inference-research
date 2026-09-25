"""actors.py's turn/efficiency metrics hook: `_run_agent` -> `_record_metrics`.

Ports the DS41-C20c seat A/B driver's `opencode export` parsing
(`/mnt/raid0/llm/tmp/ak-seat-ab/driver.py`) into the seat, so every campaign call
(not only an A/B arm) appends a metrics line to `actor-calls.jsonl`, alongside the
existing VB-AK-SEAT `epyc.autokernel.actor_call.v1` line `_record_call` already
writes. The two lines are correlated by role/timestamp, and the metrics line is
written FIRST so a v1 consumer reading "the last line" (as the existing
`test_actor_call_record.py` suite and any VB-AK-SEAT reader do) is unaffected.

No inference: `actors.subprocess.run` is always mocked (never a real `opencode`
call), and opencode session listing/export is mocked at the `actor_metrics`
function boundary using REAL fixture data (see `test_actor_metrics.py`'s fixture
docstring) copied in by the mock instead of shelled out.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actor_metrics, actors

FIXTURES = Path(__file__).parent / "testdata" / "actor_metrics"
PLAIN_EXPORT = FIXTURES / "ds41_c20c_plain_export.json"
TRUNCATED_EXPORT = FIXTURES / "truncated_pipe_export.json"
SALVAGE_RC1_STDOUT = (FIXTURES / "salvaged_rc1_run7_0955.stdout").read_text()

HYP = json.dumps({"mechanism_id": "akm-x", "statement": "s", "falsifier": "f",
                  "target_surface": "a.cpp", "target_symbol": "f"})


def _copy_fixture_export(source: Path):
    def fake_export(workspace, session_id, out_path, *, timeout_s=60.0):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(source.read_bytes())
    return fake_export


class _Lane(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = self.root / "lane"
        self.ws.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _rows(self) -> list[dict]:
        log = self.ws.parent / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG
        if not log.is_file():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def _metrics_rows(self) -> list[dict]:
        return [r for r in self._rows() if r.get("schema") == actor_metrics.METRICS_SCHEMA]

    def _done(self, stdout: str, stderr: str = "", rc: int = 0):
        return subprocess.CompletedProcess(args=["x"], returncode=rc, stdout=stdout, stderr=stderr)


class NormalCall(_Lane):
    """A plain, unbounded opencode planner call with a genuine new session."""

    def test_records_c20c_totals_and_a_counted_compaction(self):
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        self.assertEqual(backend.binary, actors.OPENCODE, "sanity: this IS the real-binary path")
        with mock.patch.object(actors.subprocess, "run", return_value=self._done(HYP)), \
             mock.patch.object(actor_metrics, "list_session_ids",
                               side_effect=[set(), {"ses_new"}]), \
             mock.patch.object(actor_metrics, "export_session",
                               side_effect=_copy_fixture_export(PLAIN_EXPORT)):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                    schema=actors.HYPOTHESIS_SCHEMA)
        self.assertEqual(raw, HYP)
        rows = self._metrics_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["role"], "planner")
        self.assertEqual(row["backend_kind"], "opencode")
        self.assertEqual(row["returncode"], 0)
        self.assertFalse(row["timed_out"])
        self.assertFalse(row["salvaged"])
        self.assertTrue(row["schema_valid"])
        self.assertFalse(row["repair_ran"])
        self.assertIsNone(row["metrics_error"])
        totals = row["opencode"]["totals"]
        self.assertEqual(totals["steps"], 23)
        self.assertEqual(totals["tool_calls"], 25)
        self.assertEqual(totals["decoded_tokens"], 57702)
        # The "call with a compaction" case: this real export carries exactly one.
        self.assertEqual(totals["compactions"], 1)
        self.assertEqual(row["opencode"]["context_first_tokens"], 39192)
        self.assertEqual(row["opencode"]["context_max_tokens"], 93443)
        self.assertEqual(row["opencode"]["primary_session_id"], "ses_f2cdb6f1effeKFxMQILnVhjMxl")

    def test_metrics_row_precedes_the_v1_call_record_row(self):
        """A reader taking "the last line" for the v1 record (the existing
        `test_actor_call_record.py` suite, and any VB-AK-SEAT consumer) must be
        unaffected by this addition."""
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", return_value=self._done(HYP)), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                              schema=actors.HYPOTHESIS_SCHEMA)
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[-2]["seat_arm"], "no arm set, none invented")
        # The metrics row is written first, whatever `_record_call` produced (v1
        # or its pre-hook fallback) is last -- unaffected by this addition.
        self.assertEqual(rows[-2]["schema"], actor_metrics.METRICS_SCHEMA)
        self.assertNotEqual(rows[-1].get("schema"), actor_metrics.METRICS_SCHEMA)


    def test_the_seat_arm_label_rides_on_the_metrics_row(self):
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", return_value=self._done(HYP)), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                              schema=actors.HYPOTHESIS_SCHEMA,
                              env={actors.SEAT_ENV_ARM: "plain+ctx-variable"})
        self.assertEqual(self._metrics_rows()[-1]["seat_arm"], "plain+ctx-variable")


class SalvagedCall(_Lane):
    """DS41 run 7, 2026-09-24 09:55: opencode's own bash tool threw
    `(res.stderr || "").trim is not a function`, `run` exited 1, but stdout ends
    with a complete hypothesis. This real capture is the fixture."""

    def test_salvaged_rc1_reply_is_recorded_as_salvaged_and_schema_valid(self):
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run",
                               return_value=self._done(SALVAGE_RC1_STDOUT, "internal error", rc=1)), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                    schema=actors.HYPOTHESIS_SCHEMA)
        self.assertIn("mechanism_id", raw)
        row = self._metrics_rows()[0]
        self.assertEqual(row["returncode"], 1)
        self.assertTrue(row["salvaged"])
        self.assertTrue(row["schema_valid"])
        self.assertFalse(row["repair_ran"])

    def test_a_genuine_failure_rc1_with_no_answer_is_not_salvaged(self):
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run",
                               return_value=self._done("no json here", "boom", rc=1)), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            with self.assertRaises(actors.ProviderTransient):
                actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                  schema=actors.HYPOTHESIS_SCHEMA)
        row = self._metrics_rows()[0]
        self.assertEqual(row["returncode"], 1)
        self.assertFalse(row["salvaged"])
        # No `final_text` was ever handed to `_parse_reply` (the call is about to
        # raise), so schema_valid/repair_ran are unknowable, not guessed False.
        self.assertIsNone(row["schema_valid"])
        self.assertIsNone(row["repair_ran"])


class MetricsCollectionFailureNeverFailsTheCall(_Lane):

    def test_a_truncated_export_is_metrics_error_and_the_call_still_succeeds(self):
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", return_value=self._done(HYP)), \
             mock.patch.object(actor_metrics, "list_session_ids",
                               side_effect=[set(), {"ses_bad"}]), \
             mock.patch.object(actor_metrics, "export_session",
                               side_effect=_copy_fixture_export(TRUNCATED_EXPORT)):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                    schema=actors.HYPOTHESIS_SCHEMA)
        self.assertEqual(raw, HYP, "the actor call itself must succeed regardless")
        row = self._metrics_rows()[0]
        self.assertIsNotNone(row["metrics_error"])
        self.assertIsNone(row["opencode"]["totals"])

    def test_session_listing_itself_raising_is_metrics_error_not_a_call_failure(self):
        """`list_session_ids` promises never to raise, but the BEFORE-call
        snapshot site in `_run_agent` (and `collect`'s own AFTER-call snapshot)
        each carry their own defensive catch too -- belt and braces, so a mock
        (or a future refactor) that breaks that internal promise still cannot
        fail the actor call."""
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", return_value=self._done(HYP)), \
             mock.patch.object(actor_metrics, "list_session_ids",
                               side_effect=RuntimeError("opencode db locked")):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                    schema=actors.HYPOTHESIS_SCHEMA)
        self.assertEqual(raw, HYP)
        row = self._metrics_rows()[0]
        self.assertIsNotNone(row["metrics_error"])
        self.assertIn("opencode db locked", row["opencode"]["metrics_error"])
        self.assertIsNone(row["opencode"]["totals"])

    def test_an_unexpected_exception_inside_record_metrics_still_never_raises(self):
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", return_value=self._done(HYP)), \
             mock.patch.object(actors, "_precheck_reply",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                    schema=actors.HYPOTHESIS_SCHEMA)
        self.assertEqual(raw, HYP)
        row = self._metrics_rows()[0]
        self.assertIn("boom", row["metrics_error"])


STORE_STDERR = (FIXTURES / "store_error_run9c_0759.stderr").read_text()
PATHS = json.dumps({"paths": ["ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"]})


class OpencodeStoreErrorCall(_Lane):
    """DS41 run 9c 07:59:47Z (the real stderr is the fixture): the author exited 1
    in ~12 s, empty stdout, "Unexpected error / Failed query: insert into project".
    The reaper's VACUUM held the write lock; no model turn ever ran."""

    def _call(self, done, backend=None, schema=None):
        backend = backend or actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", return_value=done), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            return actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                     schema=schema or actors.PATHS_SCHEMA)

    def test_is_classified_recorded_and_raised_as_a_store_error(self):
        with self.assertRaises(actors.OpencodeStoreError) as caught:
            self._call(self._done("", STORE_STDERR, rc=1))
        self.assertIsInstance(caught.exception, actors.ProviderTransient, "retryable")
        self.assertIn("opencode_store_error", str(caught.exception))
        row = self._metrics_rows()[0]
        self.assertEqual(row["failure_class"], "opencode_store_error")
        self.assertEqual(row["returncode"], 1)
        self.assertFalse(row["salvaged"])
        self.assertIsNone(row["schema_valid"], "no reply exists to judge")

    def test_a_reply_shaped_object_in_the_failed_statement_is_never_salvaged(self):
        """A `part` insert's params ARE model output (2026-09-01 "Failed query: insert
        into part ..."): a complete paths object quoted there is not a reply."""
        stderr = ('Error: Unexpected error\n\nFailed query: insert into "part" (...) '
                  f'values (?, ?)\nparams: prt_1,msg_1,{PATHS}')
        with self.assertRaises(actors.OpencodeStoreError):
            self._call(self._done("", stderr, rc=1))
        self.assertFalse(self._metrics_rows()[0]["salvaged"])

    def test_stdout_present_is_not_a_store_error(self):
        """Non-empty stdout means the session ran: the ordinary salvage path owns it."""
        raw = self._call(self._done(PATHS, "Failed query: insert into part", rc=1))
        self.assertEqual(raw, PATHS)
        row = self._metrics_rows()[0]
        self.assertTrue(row["salvaged"])
        self.assertIsNone(row["failure_class"])

    def test_other_rc1_failures_and_other_backends_keep_the_plain_transient(self):
        with self.assertRaises(actors.ProviderTransient) as caught:
            self._call(self._done("", "Error: model not found", rc=1))
        self.assertNotIsInstance(caught.exception, actors.OpencodeStoreError)
        with self.assertRaises(actors.ProviderTransient) as caught:
            self._call(self._done("", STORE_STDERR, rc=1), backend=actors.CRITIC_DEFAULT,
                       schema=actors.REVIEW_SCHEMA)
        self.assertNotIsInstance(caught.exception, actors.OpencodeStoreError)
        self.assertEqual([r["failure_class"] for r in self._metrics_rows()], [None, None])

    def test_the_author_retries_after_a_store_error_and_succeeds(self):
        """The run 9c sequence end to end: store error, backoff, retried call, reply."""
        seq = iter([self._done("", STORE_STDERR, rc=1), self._done(PATHS)])
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        slept = []
        with mock.patch.object(actors.subprocess, "run", side_effect=lambda *a, **k: next(seq)), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            raw, streak = actors._with_backoff(
                lambda: actors._run_agent("p", workspace=self.ws, backend=backend,
                                          schema=actors.PATHS_SCHEMA),
                sleep=slept.append)
        self.assertEqual((raw, streak, slept), (PATHS, 1, [actors.STORE_ERROR_BACKOFF_S[0]]))
        self.assertEqual([r["failure_class"] for r in self._metrics_rows()],
                         ["opencode_store_error", None])


class Run9cSessionMisattribution(_Lane):
    """The failed author row carried the PLANNER's session (29 steps): the before
    listing died on the locked store (empty), the after listing returned the lane's
    older planner session. Real `list_session_ids` + `collect`, subprocess mocked."""

    def test_a_failed_call_is_never_credited_an_older_session(self):
        planner_session = {"id": "ses_f288f639cffer5HW8JogPQjCYW",
                           "directory": str(self.ws), "created": 1000}  # long before
        seen = []

        def run(argv, **kw):
            seen.append(argv[:3])
            if argv[1:3] == ["session", "list"]:
                if len([a for a in seen if a[1:3] == ["session", "list"]]) <= 4:
                    kw["stderr"].write(STORE_STDERR)          # before: locked, x4
                    return subprocess.CompletedProcess(argv, 1, stdout="", stderr=None)
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps([planner_session]),
                                                   stderr=None)
            if argv[1] == "export":
                raise AssertionError("the planner's session must not be exported")
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=STORE_STDERR)

        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", side_effect=run), \
             mock.patch.object(actor_metrics, "_sleep"):
            with self.assertRaises(actors.OpencodeStoreError):
                actors._run_agent("p", workspace=self.ws, backend=backend,
                                  schema=actors.PATHS_SCHEMA)
        row = self._metrics_rows()[0]
        self.assertEqual(row["opencode"]["session_ids"], [])
        self.assertIsNone(row["opencode"]["totals"])
        self.assertIn("none created at or after", row["metrics_error"])
        self.assertEqual(row["failure_class"], "opencode_store_error")

    def test_the_session_the_call_created_is_still_found(self):
        import time as _time
        future = int((_time.time() + 3600) * 1000)
        rows = [{"id": "ses_old", "directory": str(self.ws), "created": 1000},
                {"id": "ses_new", "directory": str(self.ws), "created": future}]

        listings = iter([rows[:1], rows])   # before the call: only the old session

        def run(argv, **kw):
            if argv[1:3] == ["session", "list"]:
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(next(listings)),
                                                   stderr=None)
            return self._done(HYP)

        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", side_effect=run), \
             mock.patch.object(actor_metrics, "export_session",
                               side_effect=_copy_fixture_export(PLAIN_EXPORT)):
            actors._run_agent("p", workspace=self.ws, backend=backend,
                              schema=actors.HYPOTHESIS_SCHEMA)
        row = self._metrics_rows()[0]
        self.assertEqual(row["opencode"]["session_ids"], ["ses_new"])
        self.assertEqual(row["opencode"]["totals"]["steps"], 23)


class NonOpencodeBackend(_Lane):
    """Universal fields (wall/returncode/schema_valid) still record for a hosted
    (non-opencode) backend; there is no session to export."""

    def test_a_hosted_claude_critic_call_has_no_opencode_block(self):
        with mock.patch.object(actors.subprocess, "run",
                               return_value=self._done('{"accepted": true, "reason": "ok"}')):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=actors.CRITIC_DEFAULT,
                                    schema=actors.REVIEW_SCHEMA)
        self.assertIn("accepted", raw)
        row = self._metrics_rows()[0]
        self.assertEqual(row["role"], "critic")
        self.assertEqual(row["backend_kind"], "claude")
        self.assertIsNone(row["opencode"])
        self.assertIsNone(row["metrics_error"])
        self.assertTrue(row["schema_valid"])

    def test_run9c_codex_critic_accept_without_reason_is_schema_valid(self):
        """DS41 run 9c 07:59:35Z: codex's stdout was exactly `{"accepted":true}\\n`
        and the row recorded schema_valid=false."""
        codex = actors.backend_for("gpt-6-sol", "high")
        with mock.patch.object(actors.subprocess, "run",
                               return_value=self._done('{"accepted":true}\n')):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=codex,
                                    schema=actors.REVIEW_SCHEMA)
        self.assertEqual(raw.strip(), '{"accepted":true}')
        row = self._metrics_rows()[0]
        self.assertEqual((row["role"], row["backend_kind"]), ("critic", "codex"))
        self.assertTrue(row["schema_valid"])
        self.assertFalse(row["repair_ran"])


class ScriptBackendDoubleNeverTouchesOpencode(_Lane):
    """`test_actor_stop.py`'s script-backend double sets `kind="opencode"` but
    `binary=sys.executable` to run a real child process for signal-tree testing.
    This must NEVER trigger a real `opencode session list`/`export` call -- the
    hook is gated on `binary == OPENCODE`, not merely `kind == "opencode"`."""

    def test_a_script_double_skips_session_collection_entirely(self):
        backend = actors.Backend(kind="opencode", model="test/double", effort="high",
                                 binary=sys.executable)
        with mock.patch.object(actors.subprocess, "run", return_value=self._done(HYP)), \
             mock.patch.object(actor_metrics, "list_session_ids",
                               side_effect=AssertionError("must not be called")), \
             mock.patch.object(actor_metrics, "export_session",
                               side_effect=AssertionError("must not be called")):
            raw = actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                    schema=actors.HYPOTHESIS_SCHEMA)
        self.assertEqual(raw, HYP)
        row = self._metrics_rows()[0]
        self.assertIsNone(row["opencode"])
        self.assertIsNone(row["metrics_error"])


class StopAndTimeoutPaths(_Lane):

    def test_timeout_still_records_a_metrics_row(self):
        exc = subprocess.TimeoutExpired(cmd=["x"], timeout=1, output=b"partial", stderr=b"")
        backend = actors.backend_for("qwen-gpu/qwen3.8-27b", "high")
        with mock.patch.object(actors.subprocess, "run", side_effect=exc), \
             mock.patch.object(actor_metrics, "list_session_ids", return_value=set()):
            with self.assertRaises(actors.ProviderTransient):
                actors._run_agent("the prompt", workspace=self.ws, backend=backend,
                                  schema=actors.HYPOTHESIS_SCHEMA)
        row = self._metrics_rows()[0]
        self.assertEqual(row["returncode"], -1)
        self.assertTrue(row["timed_out"])
        self.assertIsNone(row["schema_valid"])


if __name__ == "__main__":
    unittest.main()
