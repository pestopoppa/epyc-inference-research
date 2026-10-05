"""Regression coverage for executable code-oracle scoring."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

import pytest

import debug_scorer
from debug_scorer import ScoringUnavailableError, score_answer  # noqa: E402


@pytest.fixture(autouse=True)
def _bind_scorer_scratch_to_private_test_root(tmp_path: Path, monkeypatch) -> None:
    scratch_root = tmp_path / "debug-scorer-scratch"
    scratch_root.mkdir(mode=0o700)
    assert not scratch_root.is_symlink()
    assert scratch_root.stat().st_uid == os.getuid()
    assert scratch_root.stat().st_mode & 0o777 == 0o700
    monkeypatch.setattr(debug_scorer, "_SCORER_TMP_ROOT", scratch_root)


_BCB190_ANSWER = """```python
import sqlite3
import pandas as pd
import csv
from io import StringIO

DATABASE_NAME = 'test.db'
TABLE_NAME = 'test_table'

def task_func(csv_input):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    if isinstance(csv_input, str):
        with open(csv_input, 'r') as file:
            csv_data = file.read()
    else:
        csv_data = csv_input.getvalue()
    reader = csv.reader(StringIO(csv_data))
    headers = next(reader)
    cursor.execute(f\"DROP TABLE IF EXISTS {TABLE_NAME}\")
    cursor.execute(
        f\"CREATE TABLE {TABLE_NAME} ({', '.join([f'{header} TEXT' for header in headers])})\"
    )
    placeholders = ', '.join(['?' for _ in headers])
    for row in reader:
        cursor.execute(f\"INSERT INTO {TABLE_NAME} VALUES ({placeholders})\", row)
    conn.commit()
    result = pd.read_sql_query(f\"SELECT * FROM {TABLE_NAME}\", conn)
    conn.close()
    return result
```"""


def test_code_execution_without_oracle_does_not_pass_on_importable_code() -> None:
    answer = "```python\nprint('hello')\n```"

    assert not score_answer(
        answer=answer,
        expected="print",
        scoring_method="code_execution",
        scoring_config={"language": "python", "timeout": 5},
    )


def test_code_execution_comment_only_assertions_are_not_an_oracle() -> None:
    answer = "```python\ndef add(a, b):\n    return a + b\n```"

    assert not score_answer(
        answer=answer,
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "test_code": "# assert add(1, 2) == 3\n",
        },
    )


def test_code_execution_vacuous_assert_true_is_not_an_oracle() -> None:
    answer = "```python\ndef add(a, b):\n    return a - b\n```"

    assert not score_answer(
        answer=answer,
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "test_code": "assert True\n",
        },
    )


def test_code_execution_entry_point_requires_real_cases() -> None:
    answer = "```python\ndef add(a, b):\n    return a + b\n```"

    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer=answer,
            expected="add",
            scoring_method="code_execution",
            scoring_config={
                "language": "python",
                "timeout": 5,
                "entry_point": "add",
            },
        )


def test_code_execution_entry_point_cases_execute() -> None:
    assert score_answer(
        answer="```python\ndef add(a, b):\n    return a + b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "entry_point": "add",
            "entry_point_cases": [
                {"args": [1, 2], "expected": 3},
                {"args": [-1, 5], "expected": 4},
            ],
        },
    )
    assert not score_answer(
        answer="```python\ndef add(a, b):\n    return a - b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 5,
            "entry_point": "add",
            "entry_point_cases": [{"args": [1, 2], "expected": 3}],
        },
    )


def test_code_execution_refuses_missing_scorer_scratch(monkeypatch, tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-scratch-root"
    assert not missing_root.exists()

    monkeypatch.setattr(debug_scorer, "_SCORER_TMP_ROOT", missing_root)

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("scorer must refuse before spawning when its scratch root is missing")

    monkeypatch.setattr(debug_scorer.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(ScoringUnavailableError, match="temporary harness"):
        score_answer(
            answer="```python\ndef add(a, b):\n    return a + b\n```",
            expected="",
            scoring_method="code_execution",
            scoring_config={
                "language": "python",
                "timeout": 5,
                "test_code": "assert add(1, 2) == 3",
            },
        )
    assert not missing_root.exists()


def test_code_execution_rejects_unsafe_entry_point_name() -> None:
    with pytest.raises(ScoringUnavailableError):
        score_answer(
            answer="```python\ndef add(a, b):\n    return a + b\n```",
            expected="",
            scoring_method="code_execution",
            scoring_config={
                "language": "python",
                "timeout": 5,
                "entry_point": "add(); import os",
                "entry_point_cases": [{"args": [1, 2], "expected": 3}],
            },
        )


def test_code_execution_runs_unittest_cases_without_embedded_runner() -> None:
    test_code = """
import unittest

class TestCases(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""

    assert score_answer(
        answer="```python\ndef add(a, b):\n    return a + b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={"language": "python", "timeout": 5, "test_code": test_code},
    )
    assert not score_answer(
        answer="```python\ndef add(a, b):\n    return a - b\n```",
        expected="",
        scoring_method="code_execution",
        scoring_config={"language": "python", "timeout": 5, "test_code": test_code},
    )


def test_bcb190_sqlite_answer_still_scores_true(monkeypatch) -> None:
    """BCB190 writes ``test.db`` relative to the scorer CWD."""
    # The scorer spawns bare ``python3`` resolved via PATH. Without an activated
    # venv that is the system interpreter, which lacks pandas, so the harness
    # fails on import rather than on the CWD behaviour under test. Pin PATH to
    # the interpreter running this suite (which must itself provide pandas).
    pytest.importorskip("pandas")
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    assert score_answer(
        answer=_BCB190_ANSWER,
        expected="task_func",
        scoring_method="code_execution",
        scoring_config={
            "language": "python",
            "timeout": 10,
            "test_code": """
from io import StringIO

result = task_func(StringIO('Name,Age\\nAlice,25\\nBob,30\\n'))
assert result.to_dict(orient='list') == {'Name': ['Alice', 'Bob'], 'Age': ['25', '30']}
""",
        },
    )


def test_code_execution_timeout_kills_descendant_processes() -> None:
    """A timed-out harness must not wait for a child holding its output pipes."""
    child = "import time; time.sleep(5)"
    test_code = f"""
import subprocess
import sys
import time

subprocess.Popen([sys.executable, '-c', {child!r}])
time.sleep(5)
assert time.monotonic() >= 0
"""
    started = time.monotonic()
    with pytest.raises(ScoringUnavailableError, match="configured timeout"):
        score_answer(
            answer="```python\ndef task_func():\n    return None\n```",
            expected="",
            scoring_method="code_execution",
            scoring_config={"language": "python", "timeout": 0.2, "test_code": test_code},
        )
    assert time.monotonic() - started < 2


def test_code_execution_child_dies_when_its_scorer_parent_is_killed(tmp_path: Path) -> None:
    """An externally killed scorer must not strand its isolated child on CPU."""
    pid_file = tmp_path / "solution.pid"
    child_scratch = tmp_path / "child-scorer-scratch"
    child_scratch.mkdir(mode=0o700)
    wrapper = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(REPO_ROOT / 'scripts' / 'benchmark')!r})
import debug_scorer
debug_scorer._SCORER_TMP_ROOT = Path({str(child_scratch)!r})
from debug_scorer import score_answer
answer = '''```python
from pathlib import Path
import os
import time
def task_func():
    Path({str(pid_file)!r}).write_text(str(os.getpid()))
    while True:
        pass
```'''
score_answer(answer, '', 'code_execution', {{'language': 'python', 'timeout': 30, 'test_code': 'assert task_func() is None'}})
"""
    parent = subprocess.Popen([sys.executable, "-c", wrapper])
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.is_file(), "scorer child did not start"
        child_pid = int(pid_file.read_text())
        os.kill(parent.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("scorer child survived its killed parent")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_code_execution_concurrent_relative_files_are_isolated(tmp_path: Path) -> None:
    """Concurrent scorers cannot cross-contaminate a BCB190-style ``test.db``."""
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()

    def score_with_token(token: str) -> bool:
        test_code = f"""
from pathlib import Path
import time

barrier_dir = Path({str(barrier_dir)!r})
(barrier_dir / {token!r}).write_text('ready', encoding='utf-8')
deadline = time.monotonic() + 5
while len(list(barrier_dir.glob('*'))) < 2:
    assert time.monotonic() < deadline
    time.sleep(0.01)

Path('test.db').write_text({token!r}, encoding='utf-8')
time.sleep(0.25)
(barrier_dir / {token + ".cwd"!r}).write_text(str(Path.cwd()), encoding='utf-8')
assert Path('test.db').read_text(encoding='utf-8') == {token!r}
"""
        return score_answer(
            answer="```python\ndef task_func():\n    return None\n```",
            expected="",
            scoring_method="code_execution",
            scoring_config={"language": "python", "timeout": 10, "test_code": test_code},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(score_with_token, ("left", "right")))

    assert results == [True, True]
    left_cwd = Path((barrier_dir / "left.cwd").read_text(encoding="utf-8"))
    right_cwd = Path((barrier_dir / "right.cwd").read_text(encoding="utf-8"))
    assert left_cwd != right_cwd
    assert not left_cwd.exists()
    assert not right_cwd.exists()
