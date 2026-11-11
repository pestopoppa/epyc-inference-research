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
# r8: authoritative list printed by test-backend-ops at the end of a run ("Failing tests:" then one indented case per line).  Needed because
# value-transform failures print the case on one line and "AK_VALUE_V1 ... completed=N | test failed FAIL" on another, with backend debug
# output between them, so neither the ERR nor the line-anchored FAIL form sees them (r7: 59 GATED_DELTA_NET failures, 0 parsed).
_FAILING = re.compile(r"^Failing tests:\s*\n((?:[ \t]+\S[^\n]*\n?)+)", re.M)
_VT_FAIL = re.compile(r"AK_VALUE_V1 [^\n]*?completed=(\d+)[^\n]*\|\s*(?:test|compare) failed")
GDN_OP = "GATED_DELTA_NET("
N_TRANSFORMS = 4        # identity, x3, x0p01, negate; the loop stops at the first failing transform and completed=N names it
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
    tail = Counter()
    fl = _FAILING.search(t)
    if fl:
        listed = [case_key(x) for x in fl.group(1).splitlines() if re.match(r"\s*[A-Z][A-Z0-9_]*\(", x)]   # drops the trailing "Backend ROCm0: FAIL"
        if len(listed) == total - passed:                       # the list reconciles: it is the authority
            vt = [int(c) for c in _VT_FAIL.findall(t)]
            if vt and len(vt) == len(listed):                   # same execution order: tag each failure with the transform it died at
                listed = [f"{k}@t{c}" for k, c in zip(listed, vt)]
                tail = Counter(k for k in listed if k.endswith(f"@t{N_TRANSFORMS}") and k.startswith(GDN_OP))
            fails = Counter(listed)
    n_fail = sum(fails.values())
    return {
        "complete": bool(counts),
        "passed": passed,
        "total": total,
        "fail_counts": dict(sorted(fails.items())),
        "n_fail": n_fail,
        "tail_transform_fails": sum(tail.values()),
        "consistent": bool(counts) and n_fail == total - passed,
    }


def domain_fails(p: dict, vals: dict | None = None) -> int:
    """Distinct GATED_DELTA_NET baseline failures that are the op leaving its numeric domain, not a kernel defect.

    (a) failed at the 4th (negate) transform after identity/x3/x0p01 passed (``@t4`` tag), in ANY suite;
    (b) hostile-distribution cases (alternating / sparse_outlier gate+state patterns grow exp(+|g|) state): the raw nmse of the case is
        non-finite (ROCm overflows to NaN where the CPU reference is finite-but-huge, or both overflow).
    Exempt keys stay in fail_counts, so a candidate that fails a case the baseline PASSED is still a new fail; only the pass-fraction floor
    ignores them.  Non-GDN ops are never exempted (they are KNOWN failures instead, see known_failures).
    """
    keys = {k for k in p["fail_counts"] if k.startswith(GDN_OP)}
    out = {k for k in keys if k.endswith(f"@t{N_TRANSFORMS}")}
    if vals:
        bad = set()
        for rk, v in vals.items():
            if v[0] != v[0] or v[0] == float("inf") or v[1] != v[1] or v[1] == float("inf"):
                bad.add(rk.split("|", 1)[0])
        out |= {k for k in keys if re.sub(r"@t\d+$", "", k)[len(GDN_OP):-1] in bad}
    return len(out)


def baseline_usable(p: dict, *, min_cases: int = MIN_CASES, min_pass_fraction: float = MIN_PASS_FRACTION, domain: int | None = None) -> tuple[bool, str]:
    """Stop-on-futility predicate for one BASELINE run.  ``domain`` = domain_fails(p, raw); default: the @t4 count carried by the parse."""
    if not p["complete"]:
        return False, "crash_or_truncated"
    if p["total"] < min_cases:
        return False, f"fewer_than_{min_cases}_cases"
    if not p["consistent"]:
        return False, "fail_count_unparsed"
    if domain is None:
        domain = p.get("tail_transform_fails", 0)
    valid = p["total"] - domain
    if p["passed"] < min_pass_fraction * valid:
        return False, "baseline_pass_fraction_below_floor"
    return True, "ok"


def known_failures(parses: list[dict]) -> dict:
    """KNOWN baseline failures: keys failing with identical multiplicity in EVERY supplied complete baseline run (>=2 runs needed to claim
    determinism).  Excluded from the new-fail comparison (new_fails already does that); recorded so the verdict says what was tolerated."""
    runs = [Counter(q["fail_counts"]) for q in parses if q["complete"]]
    if len(runs) < 2:
        return {"deterministic": False, "n_runs": len(runs), "known": []}
    same = all(r == runs[0] for r in runs[1:])
    return {"deterministic": same, "n_runs": len(runs), "known": sorted(runs[0]) if same else sorted(set.intersection(*[set(r) for r in runs]))}


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
