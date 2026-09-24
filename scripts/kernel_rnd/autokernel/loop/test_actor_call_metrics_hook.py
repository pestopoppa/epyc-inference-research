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
