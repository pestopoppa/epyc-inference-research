"""New-fail-vs-baseline correctness gate for GPU test-backend-ops runs (INF-48 RVP-EQV-1 / RVP-RACE-1).

Operator decision 2026-11-10: on ROCm the CPU-anchored NMSE bounds (served_shape_lowbit_v1, anchor x3
floored at 1e-8) are below normal ROCm numeric error (~2.5e-5), so the unmutated BASELINE build fails
some cases on every run.  The gate therefore compares each mutant/candidate run against the BASELINE
run of the SAME suite, SAME seed and SAME binary family: a candidate fails only on cases the baseline
passed.  Baseline numeric error cancels out.

Parsing strips ANSI colour codes first: test-backend-ops prints ``\\x1b[1;31mFAIL\\x1b[0m`` and the
``\\bFAIL\\b`` regex of the r3 job matched none of the 312 real failures (r3 recorded 0 fail lines).

Pure functions, no I/O beyond the caller's text.  Stdlib only so a row dir can copy this file verbatim.
"""
from __future__ import annotations

import re
from collections import Counter

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PASSED = re.compile(r"(\d+)/(\d+) tests passed")
# "[OP] ERR = a > b   OP(params): test failed FAIL" -- debug output may be interleaved after the colon.
_ERR = re.compile(r"ERR = \S+ > \S+\s+([A-Z][A-Z0-9_]*\([^)]*\))")
# Any other per-case failure line: "  OP(params): ... FAIL"
_CASE_FAIL = re.compile(r"^\s*([A-Z][A-Z0-9_]*\([^)]*\)):.*\bFAIL\s*$", re.M)
_MAXNMSE = re.compile(r",max_nmse=[^,)]*")

# Pre-registered futility floors (RVP-EQV-1 box, 2026-11-10).
MIN_CASES = 50          # a baseline that parses fewer cases did not run the suite
MIN_PASS_FRACTION = 0.5  # a baseline failing most cases is broken, not numeric noise


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def case_key(raw: str) -> str:
    """Case identity without the per-case bound, which is a threshold, not part of the case."""
    return _MAXNMSE.sub("", raw.strip())


def parse_log(text: str) -> dict:
    """Parse one test-backend-ops log into a multiset of failed case keys plus the pass count."""
    t = strip_ansi(text)
    counts = [(int(a), int(b)) for a, b in _PASSED.findall(t)]
    passed, total = sum(a for a, _ in counts), sum(b for _, b in counts)
    err_lines = {}
    for m in _ERR.finditer(t):
        line_start = t.rfind("\n", 0, m.start()) + 1
        err_lines[line_start] = case_key(m.group(1))
    for m in _CASE_FAIL.finditer(t):
        line_start = t.rfind("\n", 0, m.start(1)) + 1
        err_lines.setdefault(line_start, case_key(m.group(1)))   # one key per line, ERR form wins
    fails = Counter(err_lines.values())
    n_fail = sum(fails.values())
    return {
        "complete": bool(counts),
        "passed": passed,
        "total": total,
        "fail_counts": dict(sorted(fails.items())),
        "n_fail": n_fail,
        "consistent": bool(counts) and n_fail == total - passed,
    }


def baseline_usable(p: dict, *, min_cases: int = MIN_CASES, min_pass_fraction: float = MIN_PASS_FRACTION) -> tuple[bool, str]:
    """Stop-on-futility predicate for one BASELINE run."""
    if not p["complete"]:
        return False, "crash_or_truncated"
    if p["total"] < min_cases:
        return False, f"fewer_than_{min_cases}_cases"
    if not p["consistent"]:
        return False, "fail_count_unparsed"
    if p["passed"] < min_pass_fraction * p["total"]:
        return False, "baseline_pass_fraction_below_floor"
    return True, "ok"


def new_fails(base: dict, cand: dict) -> dict:
    """Failures of ``cand`` on cases ``base`` passed (multiset difference)."""
    b, c = Counter(base["fail_counts"]), Counter(cand["fail_counts"])
    return dict(sorted((c - b).items()))


def compare(base: dict, cand: dict) -> dict:
    """Gate verdict of one candidate run against the paired BASELINE run.

    PASS      no failure on a case the baseline passed, same case count, complete log
    NEW_FAIL  >=1 failure on a baseline-passing case (gate caught it)
    CRASH     candidate log has no pass summary (crash/abort/timeout counts as caught)
    CASES_MISSING candidate ran a different number of cases (counts as caught)
    UNPARSED  candidate fail count does not reconcile; harness defect, never scored
    """
    if not cand["complete"]:
        v = "CRASH"
    elif not cand["consistent"]:
        v = "UNPARSED"
    elif cand["total"] != base["total"]:
        v = "CASES_MISSING"
    else:
        v = "NEW_FAIL" if new_fails(base, cand) else "PASS"
    nf = new_fails(base, cand) if cand["complete"] else {}
    return {"verdict": v, "gate_pass": v == "PASS", "n_new_fail": sum(nf.values()), "new_fail": list(nf)[:200]}


def determinism(base: dict, repeat: dict) -> dict:
    """Same binary, same seed, twice: fail sets must match in both directions."""
    a, b = Counter(base["fail_counts"]), Counter(repeat["fail_counts"])
    d = (a - b) + (b - a)
    return {"deterministic": not d and base["total"] == repeat["total"], "n_diff": sum(d.values()), "diff": list(d)[:50]}
