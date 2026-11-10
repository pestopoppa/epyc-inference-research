"""Re-score r3 (race-eqv-gpu-r3-20261109, BASELINE|Q|0) under the 2026-11-10 new-fail-vs-baseline gate.

Fixture: the r3 run's raw log, gzip of out/logs/current.log, SHA-256 of the uncompressed bytes
67e4760557e509a3dcb5aa4ba7e237f28df0029a4b52c042e3d64679880b0801 (matches r3 results.jsonl log_sha256).
"""
from __future__ import annotations

import gzip
import hashlib
import re
from pathlib import Path

import pytest

from . import gpu_newfail_gate as g

FIX = Path(__file__).with_name("fixtures") / "race_eqv_r3_baseline_q0.log.gz"
R3_SHA = "67e4760557e509a3dcb5aa4ba7e237f28df0029a4b52c042e3d64679880b0801"


@pytest.fixture(scope="module")
def r3_text() -> str:
    raw = gzip.decompress(FIX.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == R3_SHA
    return raw.decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def r3(r3_text):
    return g.parse_log(r3_text)


def test_r3_old_regex_was_blind(r3_text):
    # the r3 job's parser: 0 FAIL lines over a log with 312 real failures
    assert re.findall(r"^.*\bFAIL\b.*$", r3_text, re.M) == []
    assert len(re.findall(r"^.*\bFAIL\b.*$", g.strip_ansi(r3_text), re.M)) >= 312


def test_r3_parse_reconciles(r3):
    assert (r3["passed"], r3["total"]) == (2295, 2607)
    assert r3["n_fail"] == 312 and r3["consistent"]
    assert all("max_nmse" not in k for k in r3["fail_counts"])
    assert all(k.startswith(("MUL_MAT(", "MUL_MAT_ID(")) for k in r3["fail_counts"])


def test_r3_baseline_is_usable_not_invalid(r3):
    # r3 filed INVALID_BASELINE_FAILS; under the new gate the same run is a valid baseline
    assert g.baseline_usable(r3) == (True, "ok")


def test_r3_baseline_vs_itself_passes(r3):
    c = g.compare(r3, r3)
    assert c["verdict"] == "PASS" and c["gate_pass"] and c["n_new_fail"] == 0
    assert g.determinism(r3, r3)["deterministic"]


def _flip_first_ok(text: str) -> str:
    # turn one baseline-passing case into a failure, as a mutant would
    m = re.search(r"^(\s*)(MUL_MAT\([^)]*\)): OK$", g.strip_ansi(text), re.M)
    assert m
    case = m.group(2)
    t = g.strip_ansi(text).replace(m.group(0), f"[MUL_MAT] ERR = 0.1 > 0.00003   {case}: test failed FAIL", 1)
    return t.replace("2295/2607 tests passed", "2294/2607 tests passed")


def test_mutant_new_fail_is_caught(r3, r3_text):
    mut = g.parse_log(_flip_first_ok(r3_text))
    assert mut["consistent"]
    c = g.compare(r3, mut)
    assert c["verdict"] == "NEW_FAIL" and c["n_new_fail"] == 1 and not c["gate_pass"]


def test_mutant_fixing_a_baseline_fail_still_passes(r3, r3_text):
    t = g.strip_ansi(r3_text)
    line = next(l for l in t.splitlines() if l.endswith("test failed FAIL"))
    t = t.replace(line, "", 1).replace("2295/2607 tests passed", "2296/2607 tests passed")
    assert g.compare(r3, g.parse_log(t))["verdict"] == "PASS"


def test_crash_and_missing_cases(r3, r3_text):
    trunc = g.parse_log(r3_text[: len(r3_text) // 2])
    assert g.compare(r3, trunc)["verdict"] == "CRASH"
    assert g.baseline_usable(trunc)[1] == "crash_or_truncated"
    fewer = dict(r3, total=2600, passed=2288)
    assert g.compare(r3, fewer)["verdict"] == "CASES_MISSING"


def test_futility_floors():
    tiny = {"complete": True, "passed": 10, "total": 10, "fail_counts": {}, "n_fail": 0, "consistent": True}
    assert g.baseline_usable(tiny)[1] == "fewer_than_50_cases"
    bad = {"complete": True, "passed": 40, "total": 100, "fail_counts": {"X()": 60}, "n_fail": 60, "consistent": True}
    assert g.baseline_usable(bad)[1] == "baseline_pass_fraction_below_floor"
    unp = {"complete": True, "passed": 90, "total": 100, "fail_counts": {}, "n_fail": 0, "consistent": False}
    assert g.baseline_usable(unp)[1] == "fail_count_unparsed"
