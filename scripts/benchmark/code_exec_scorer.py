"""Sandboxed executable-scoring for code-generation suites (LiveCodeBench/BigCodeBench).

The standardized code-correctness scorer: extract the model's code, run it against
the problem's test cases in an isolated subprocess, return pass/fail. This is Track 2
of handoffs/active/scoring-infra-standardization.md — the missing piece that lets a
code-gen suite measure real capability instead of the adapter's placeholder
`substring "def "` check.

Isolation (scaffold level): fresh temp cwd, RLIMIT_CPU / RLIMIT_AS (memory) /
RLIMIT_CORE=0, wall-clock timeout, minimal env, and a dedicated process group so a
timeout kills the child's whole descendant tree rather than just the child.

⚠ NOT bounded: process/thread COUNT. RLIMIT_NPROC is deliberately NOT set. It is
enforced per real UID, not per process tree, and this host runs ~9.5k threads under
the same uid (llama-server alone accounts for most of them). Any per-scorer cap
would therefore fail the child's *first* fork under normal fleet load, and would
fail NONDETERMINISTICALLY as that load varies — turning a scorer into an instrument
whose results track how busy the box is. That is a worse failure than the one it
would fix. The correct mechanism is cgroup v2 `pids.max`, which counts only the
scorer's own subtree and is unaffected by co-tenants; verified available on this
host (needs `+pids` in cgroup.subtree_control). Tracked in
handoffs/active/scoring-infra-standardization.md. Corrected 2026-07-29 — this
docstring previously claimed RLIMIT_NPROC was set; it never was.

⚠ HARDENING TODO (before untrusted / at-scale runs, Phase 2b): this does NOT yet
provide network isolation or a real filesystem jail. Run only trusted algorithmic
benchmarks until wrapped in unshare/nsjail/container. Never point this at
adversarial code.
"""
from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def extract_code(response: str, language: str = "python") -> str:
    """Pull the model's code: last fenced block for the language, else last fenced
    block, else the raw text (a bare code answer)."""
    if not response:
        return ""
    fences = re.findall(r"```(?:\s*(\w+))?\s*\n(.*?)```", response, re.DOTALL)
    if fences:
        typed = [body for lang, body in fences if lang and lang.lower() in
                 (language, "py" if language == "python" else language)]
        return (typed[-1] if typed else fences[-1][1]).strip()
    return response.strip()


def _limits(cpu_s: int, mem_mb: int):
    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        soft = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


def _kill_group(proc: subprocess.Popen | None) -> None:
    """SIGKILL the child's entire process group.

    subprocess's own timeout path kills ONLY the direct child, so anything the
    scored code spawned survives the timeout as an orphan and keeps consuming
    cores reserved for inference. `start_new_session=True` puts the child in its
    own process group precisely so the group can be killed as a unit here.
    """
    if proc is None or proc.pid is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # already reaped, or the group is gone
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001 — best-effort reap; never mask the real error
        pass


def _run_once(code: str, stdin: str, timeout: int, cpu_s: int, mem_mb: int,
              python_exe: str | None = None) -> tuple[bool, str]:
    """Run code with stdin, return (ok, stdout-or-error)."""
    # absolutize BEFORE the child runs with cwd=tempdir — a relative interpreter
    # path would otherwise resolve inside the tempdir and fail to spawn.
    # absolute(), NOT resolve(): resolve() dereferences the venv's bin/python
    # symlink to the BASE interpreter, silently dropping the venv site-packages.
    exe = str(Path(python_exe).absolute()) if python_exe else sys.executable
    with tempfile.TemporaryDirectory(prefix="codeexec_") as d:
        src = Path(d) / "sol.py"
        src.write_text(code)
        proc: subprocess.Popen | None = None
        try:
            # Popen (not run) so the pid is available for a GROUP kill on timeout;
            # start_new_session=True gives the child its own process group so the
            # kill reaches descendants the scored code spawned.
            proc = subprocess.Popen(
                [exe, str(src)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
                cwd=d, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                            "MPLBACKEND": "Agg", "HOME": d,
                            # single-threaded math libs: OpenBLAS sizes buffers by
                            # nproc (192 here) and blows RLIMIT_AS otherwise
                            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
                preexec_fn=_limits(cpu_s, mem_mb),
                start_new_session=True,
            )
            out, err = proc.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            # drain the pipes the killed group left behind, else the fds leak
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            return False, "__timeout__"
        except Exception as e:  # noqa: BLE001
            _kill_group(proc)
            return False, f"__spawn_error__:{e}"
        if proc.returncode != 0:
            return False, f"__exit_{proc.returncode}__:{err[-200:]}"
        return True, out


def score_code(
    response: str,
    test_cases: list[dict[str, Any]],
    language: str = "python",
    timeout: int = 10,
    cpu_s: int = 8,
    mem_mb: int = 1024,
) -> dict[str, Any]:
    """Score generated code against test_cases.

    Each test case is either stdin/stdout: {"input": str, "output": str}, or a
    self-checking snippet: {"assert": "<python that raises on failure>"}.
    Returns {passed, total, pass_rate, correct(bool: all passed), detail[]}.
    """
    code = extract_code(response, language)
    if not code or language != "python":
        return {"passed": 0, "total": len(test_cases), "pass_rate": 0.0,
                "correct": False, "detail": ["__no_code__" if not code else "__unsupported_lang__"]}
    passed, detail = 0, []
    for tc in test_cases:
        if "assert" in tc:
            ok, out = _run_once(code + "\n\n" + tc["assert"], "", timeout, cpu_s, mem_mb)
        else:
            ok, out = _run_once(code, tc.get("input", ""), timeout, cpu_s, mem_mb)
            if ok:
                ok = out.strip() == str(tc.get("output", "")).strip()
        passed += int(ok)
        detail.append("pass" if ok else (out[:60] if out.startswith("__") else "wrong_output"))
    n = len(test_cases)
    return {"passed": passed, "total": n, "pass_rate": passed / n if n else 0.0,
            "correct": n > 0 and passed == n, "detail": detail}


def score_functional(
    response: str,
    test_code: str,
    entry_point: str,
    prompt: str = "",
    timeout: int = 10,
    cpu_s: int = 8,
    mem_mb: int = 1024,
) -> bool:
    """HumanEval/MBPP-style: the model completes a function; `test_code` is a
    `check(candidate)` body of asserts. Pass iff the assembled program runs clean.

    Robust to instruct models that either reproduce the full function or emit only
    the body: if the extracted code does not define `entry_point`, we prepend the
    prompt (signature+docstring) so the completion attaches to it.
    """
    code = extract_code(response, "python")
    if not code:
        return False
    if not re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        code = (prompt or "") + "\n" + code
    program = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    ok, _ = _run_once(program, "", timeout, cpu_s, mem_mb)
    return ok


def score_unittest(
    response: str,
    test_code: str,
    entry_point: str,
    code_prompt: str = "",
    python_exe: str | None = None,
    timeout: int = 30,
    cpu_s: int = 25,
    mem_mb: int = 4096,
) -> bool:
    """BigCodeBench-style: the model implements `entry_point` (task_func); the
    suite ships a unittest.TestCase in `test_code`. Pass iff the whole unittest
    run exits 0. Requires the dep-rich interpreter via `python_exe`
    (matplotlib/pandas/sklearn tasks won't import under the default venv);
    MPLBACKEND=Agg + HOME=tmpdir are set by the runner env."""
    code = extract_code(response, "python")
    if not code:
        return False
    if not re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        code = (code_prompt or "") + "\n" + code
    program = (f"{code}\n\n{test_code}\n\n"
               "import unittest as _ut\n_ut.main(argv=['x'], exit=True)\n")
    ok, _ = _run_once(program, "", timeout, cpu_s, mem_mb, python_exe=python_exe)
    return ok


if __name__ == "__main__":
    # smoke test on TRUSTED code (no model involved) — proves the harness scores
    # correct vs wrong solutions and enforces timeout.
    good = "```python\nn=int(input())\nprint(n*n)\n```"
    bad = "```python\nn=int(input())\nprint(n+1)\n```"
    loop = "```python\nwhile True:\n    pass\n```"
    tests = [{"input": "3", "output": "9"}, {"input": "5", "output": "25"}]
    r_good = score_code(good, tests)
    r_bad = score_code(bad, tests)
    r_loop = score_code(loop, [{"input": "3", "output": "9"}], timeout=3, cpu_s=2)
    assert r_good["correct"] and r_good["passed"] == 2, r_good
    assert not r_bad["correct"] and r_bad["passed"] == 0, r_bad
    # runaway caught either by wall-timeout or the CPU rlimit (SIGKILL exit -9)
    assert not r_loop["correct"] and r_loop["detail"][0].startswith("__"), r_loop
    # functional/assert-style
    fn = "```python\ndef sq(x):\n    return x*x\n```"
    r_fn = score_code(fn, [{"assert": "assert sq(4)==16"}, {"assert": "assert sq(0)==0"}])
    assert r_fn["correct"], r_fn
    # HumanEval/MBPP functional path: model completes a function, check() asserts
    hprompt = "def add(a, b):\n    "
    hgood = "```python\ndef add(a, b):\n    return a + b\n```"
    hbad = "```python\ndef add(a, b):\n    return a - b\n```"
    htest = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(0, 0) == 0"
    assert score_functional(hgood, htest, "add", hprompt)
    assert not score_functional(hbad, htest, "add", hprompt)
    print("PASS code_exec_scorer smoke:",
          {"good": r_good, "bad": r_bad["passed"], "loop": r_loop["detail"][0], "fn": r_fn["correct"]})
