"""Mutation tests for the three EffiBench-X upstream blockers.

Run against the UNPATCHED tree these tests FAIL (they detect the defects).
Run against the PATCHED tree they PASS. Both runs are recorded in REPORT.md.

Blockers covered:
  B1  deprecated openjdk:21-jdk-bookworm image        -> test_no_deprecated_openjdk_image
  B2  generate_solution.py time.sleep AttributeError  -> test_generate_solution_time_sleep_resolves
  B3  fail-open-to-0.0 metrics                        -> test_docker_stats_parse_failure_raises
                                                         test_done_without_metrics_fails_closed
                                                         test_local_backend_shape_fails_closed
                                                         test_compute_model_stats_handles_none_runtime

Target tree: /workspace/tmp/effibench-x-upstream (installed editable in the venv).
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path("/workspace/tmp/effibench-x-upstream")
sys.path.insert(0, str(REPO))


# ----------------------------------------------------------------------------
# B1: no deprecated Java image anywhere in the registries
# ----------------------------------------------------------------------------

def test_no_deprecated_openjdk_image():
    from effibench.utils import EFFIBENCH_REGISTRY
    from llm_sandbox.const import DefaultImage

    images = [cfg.get("image") for cfg in EFFIBENCH_REGISTRY.values()]
    images += [v for k, v in DefaultImage.__dict__.items() if not k.startswith("__")]
    offenders = [img for img in images if img and "openjdk" in img]
    assert offenders == [], (
        f"Deprecated openjdk image(s) still referenced: {offenders} "
        "(openjdk:21-jdk-bookworm is gone from Docker Hub; manifest inspect 404s)"
    )


# ----------------------------------------------------------------------------
# B2: `from time import time` + `time.sleep(...)` AttributeError
# ----------------------------------------------------------------------------

def test_generate_solution_time_sleep_resolves():
    import importlib
    gs = importlib.import_module("generate_solution")
    # Pre-patch: gs.time is the builtin function `time` -> no .sleep attribute.
    assert hasattr(gs.time, "sleep") and callable(gs.time.sleep), (
        "generate_solution.time has no .sleep — `from time import time` shadows "
        "the module and line ~123 `time.sleep(submit_gap)` raises AttributeError "
        "on the first submitted task"
    )


# ----------------------------------------------------------------------------
# B3a: llm_sandbox docker session — statistics parse failure must raise,
#      never silently coerce runtime/memory/integral to 0.0
# ----------------------------------------------------------------------------

def _make_stub_docker_session(stats_text: str):
    """Build a SandboxDockerSession without touching the docker daemon."""
    from llm_sandbox.docker import SandboxDockerSession
    from llm_sandbox.base import ConsoleOutput

    sess = SandboxDockerSession.__new__(SandboxDockerSession)
    sess.lang = "python"
    sess.verbose = False
    sess.container = object()  # truthy: run() only checks presence
    sess.session_dir = Path("/workspace")
    sess._file_content_cache = None
    sess._dir_cache = {"", "/"}
    sess.installed_libraries = set()
    sess.logger = __import__("logging").getLogger("stub")

    sess.create_file = lambda *a, **k: None
    sess.install_libraries = lambda *a, **k: None

    def fake_execute(command, workdir=None, use_tty=None):
        if "statistics" in str(command):
            return ConsoleOutput(text=stats_text, exit_code=0)
        return ConsoleOutput(text="42\n", exit_code=0)

    sess.execute_command = fake_execute
    return sess


def test_docker_stats_parse_failure_raises():
    sess = _make_stub_docker_session(stats_text="garbage not three floats")
    # Post-patch: unparseable statistics raise RuntimeError (loud, propagates to
    # the manager's error path). Pre-patch: run() returns fabricated 0.0 metrics
    # and the AssertionError below (NOT a RuntimeError) fails the test.
    with pytest.raises(RuntimeError, match="[Ss]tatistic"):
        out = sess.run("print(42)", return_statistics=True, use_tty=False)
        raise AssertionError(
            f"FAIL-OPEN: unparseable statistics fabricated runtime={out.runtime}, "
            f"memory={out.memory}, integral={out.integral} instead of raising"
        )


def test_docker_stats_parse_success_still_works():
    sess = _make_stub_docker_session(stats_text="123456789 2048 3.14")
    out = sess.run("print(42)", return_statistics=True, use_tty=False)
    assert out.runtime == 123456789.0
    assert out.memory == 2048.0
    assert out.integral == 3.14


# ----------------------------------------------------------------------------
# B3b: backend_utils — a "done" execution with missing metrics must become an
#      error record, never a scored record with 0.0 metrics
# ----------------------------------------------------------------------------

def _run_through_manager(fake_result):
    from effibench.backends.backend_utils import (
        BaseExecutionManager,
        CodeExecutionRequest,
        SubmissionRecord,
    )

    class FakeSession:
        def run(self, **kwargs):
            return fake_result

    class FakeManager(BaseExecutionManager):
        def _create_initial_session(self, lang, config):
            pass

        def get_session(self, worker_id, language):
            return FakeSession()

    mgr = FakeManager(session_class=FakeSession, num_workers=0, skip_setup=True)
    record = SubmissionRecord(
        id=0, request=CodeExecutionRequest(code="print(42)", language="python")
    )
    mgr.execute_in_sandbox(0, record)
    return record


def test_done_without_metrics_fails_closed():
    # Successful execution (exit 0) whose result carries runtime=None:
    # exactly what a statistics failure produces.
    res = SimpleNamespace(text="42\n", exit_code=0, runtime=None, memory=None, integral=None)
    record = _run_through_manager(res)
    assert record.status != "done", (
        f"FAIL-OPEN: unmeasured execution scored as status={record.status!r} "
        f"runtime={record.runtime!r}"
    )
    assert record.runtime != 0.0 and record.memory != 0.0 and record.integral != 0.0, (
        "FAIL-OPEN: fabricated 0.0 metrics for an unmeasured execution"
    )


def test_local_backend_shape_fails_closed():
    # The bundled "local" backend returns objects with NO metric attributes at
    # all (local_sandbox.run never sets them) — upstream issue #4's shape.
    class BareResult:
        text = "42\n"
        exit_code = 0

    record = _run_through_manager(BareResult())
    assert record.status != "done", (
        f"FAIL-OPEN: local-backend-shaped result scored as status={record.status!r} "
        f"runtime={record.runtime!r}"
    )
    assert record.runtime != 0.0, "FAIL-OPEN: fabricated runtime=0.0"


# ----------------------------------------------------------------------------
# B3c: evaluate_solution.compute_model_stats must tolerate None metrics on
#      failed records (fail-closed carrier) without crashing or passing them
# ----------------------------------------------------------------------------

def test_compute_model_stats_handles_none_runtime(tmp_path):
    import importlib
    es = importlib.import_module("evaluate_solution")

    eval_dir = tmp_path / "model_x"
    eval_dir.mkdir()
    records = [
        {"status": "done", "exit_code": 0, "text": "42", "input": "", "output": "42",
         "passed": True, "runtime": 1_000_000, "memory": 1024, "integral": 0.5},
        {"status": "error", "exit_code": 1,
         "text": "MeasurementError: no statistics", "input": "", "output": "43",
         "passed": False, "runtime": None, "memory": None, "integral": None},
    ]
    (eval_dir / "leetcode_1_two-sum_python3.json").write_text(json.dumps(records))

    stats = es.compute_model_stats(
        ["leetcode_1_two-sum"], ["python3"], eval_dir, tmp_path / "stats.json"
    )
    s = stats["leetcode_1_two-sum"]["python3"]
    assert s is not None, "stats vanished for a problem with a failed record"
    assert s["passed"] is False, (
        "a problem with an unmeasured/failed record must not count as passed"
    )


# ----------------------------------------------------------------------------
# B3d (found at runtime, first-class finding): is_passed must not rescue a
# "done" record with empty output and exit 0 that the evaluator judged failed.
# Observed live: leetcode_3264 — evaluator 0/100 passed, harness is_passed True.
# ----------------------------------------------------------------------------

def test_is_passed_empty_output_not_rescued():
    import importlib
    es = importlib.import_module("evaluate_solution")
    records = [{
        "status": "done", "exit_code": 0, "text": "",
        "input": "511930105,71", "output": "1252193124",
        "passed": False, "runtime": 22751972.0, "memory": 9284.0, "integral": 0.1,
    }] * 3
    assert es.is_passed(records) is False, (
        "FAIL-OPEN: empty-output records with exit 0 counted as a pass despite "
        "the evaluator judging them failed"
    )
