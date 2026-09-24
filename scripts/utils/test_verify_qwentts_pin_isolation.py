"""INF-41 S-13: qwentts.cpp stays a pinned, isolated dependency.

Runs the read-only guard script against the actual host trees. Skips (rather
than fails) when the trees this guard inspects are not present on the host
running the suite — this test proves the guard is CORRECT on a host that has
the trees, not that the trees exist everywhere pytest runs.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("verify_qwentts_pin_isolation.sh")
QWENTTS_TREE = Path("/mnt/raid0/llm/qwentts.cpp")
LLAMA_TREE = Path("/mnt/raid0/llm/llama.cpp")
RATIFICATION = Path(
    "/workspace/artifacts/operator/ratify_speech_kernel_freeze_20260731.json"
)

pytestmark = pytest.mark.skipif(
    not (QWENTTS_TREE / ".git").is_dir() or not RATIFICATION.is_file(),
    reason="host trees for the speech-kernel freeze are not present",
)


def _run(env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_guard_passes_on_the_real_pinned_tree():
    """The actual on-disk qwentts.cpp tree is exactly what the ratification pins."""
    result = _run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: qwentts.cpp remains a pinned, isolated, versioned dependency." in result.stdout


def test_guard_reports_no_isolation_violation_against_real_llama_cpp():
    """The frozen production llama.cpp tree carries no qwentts-derived artifacts."""
    if not (LLAMA_TREE / ".git").is_dir():
        pytest.skip("production llama.cpp tree not present on this host")
    result = _run()
    assert "no qwentts.cpp-unique source files found" in result.stdout
    assert "no commit in llama.cpp's reachable history mentions 'qwentts'" in result.stdout


def test_guard_fails_closed_on_a_missing_ratification_artifact(tmp_path):
    """A guard that cannot read its own pin source must FAIL, never PASS-by-skip."""
    missing = tmp_path / "no-such-ratification.json"
    result = _run({"RATIFICATION": str(missing)})
    assert result.returncode != 0
    assert "not readable" in result.stdout + result.stderr


def test_ratification_declares_the_fields_this_guard_reads():
    """Non-vacuity: the JSON keys this script parses actually exist and are non-empty."""
    doc = json.loads(RATIFICATION.read_text())
    qwentts = doc["kernels"]["qwentts_cpp"]
    for key in ("branch", "commit", "ggml_submodule_commit"):
        assert qwentts.get(key), f"ratification is missing/empty for {key!r}"
