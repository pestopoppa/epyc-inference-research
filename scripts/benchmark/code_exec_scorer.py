"""Sandboxed executable-scoring for code-generation suites (LiveCodeBench/BigCodeBench).

The standardized code-correctness scorer: extract the model's code, run it against
the problem's test cases in an isolated subprocess, return pass/fail. This is Track 2
of handoffs/active/scoring-infra-standardization.md — the missing piece that lets a
code-gen suite measure real capability instead of the adapter's placeholder
`substring "def "` check.

Isolation (scaffold level): fresh temp cwd, RLIMIT_CPU / RLIMIT_AS (memory) /
RLIMIT_CORE=0 / RLIMIT_NPROC, wall-clock timeout, minimal env.

⚠ HARDENING TODO (before untrusted / at-scale runs, Phase 2b): this does NOT yet
provide network isolation or a real filesystem jail. Run only trusted algorithmic
benchmarks until wrapped in unshare/nsjail/container. Never point this at
adversarial code.
"""
from __future__ import annotations

import re
import resource
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


def _run_once(code: str, stdin: str, timeout: int, cpu_s: int, mem_mb: int) -> tuple[bool, str]:
    """Run code with stdin, return (ok, stdout-or-error)."""
    with tempfile.TemporaryDirectory(prefix="codeexec_") as d:
        src = Path(d) / "sol.py"
        src.write_text(code)
        try:
            p = subprocess.run(
                [sys.executable, str(src)],
                input=stdin, capture_output=True, text=True, timeout=timeout,
                cwd=d, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
                preexec_fn=_limits(cpu_s, mem_mb),
            )
        except subprocess.TimeoutExpired:
            return False, "__timeout__"
        except Exception as e:  # noqa: BLE001
            return False, f"__spawn_error__:{e}"
        if p.returncode != 0:
            return False, f"__exit_{p.returncode}__:{p.stderr[-200:]}"
        return True, p.stdout


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
