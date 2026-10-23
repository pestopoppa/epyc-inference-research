#!/usr/bin/env python3
"""INF-42 G1 probe: grid identity, resume, abort, claim, window guard, analysis, verdict (no server, no httpx)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

g1 = pytest.importorskip("g1_amnesia_probe")
verdict = pytest.importorskip("g1_amnesia_verdict")

ESSAYS = Path("/mnt/raid0/llm/data/eval/needle/repo/needlehaystack/PaulGrahamEssays")
needs_corpus = pytest.mark.skipif(not ESSAYS.exists(), reason="essay corpus not on this host")


@pytest.fixture(scope="module")
def grid():
    return g1.build_grid()


def fake_trials(n=4, length=4096):
    return [{"trial_id": f"l{length}_d10_s{i:02d}", "context_length": length, "depth": 0.1, "sample": i,
             "prompt": "p", "expected": "123", "prompt_chars": 1} for i in range(n)]


@needs_corpus
def test_base_grid_unchanged_by_extra_length(grid):
    base = [t for t in grid if t["context_length"] in g1.CONTEXT_LENGTHS]
    assert len(base) == 200
    assert [t["context_length"] for t in grid[200:]] == [262144] * 50
    # the rng stream of the base trials is the pre-256K stream: base trials come first in draw order
    ids = [t["trial_id"] for t in base]
    assert ids[0] == "l4096_d10_s00" and ids[-1] == "l131072_d90_s09"
    # the 2026-08-26 persisted trial l131072_d10_s00 had needle_key 4533779 and expected 6752576
    t = next(x for x in base if x["trial_id"] == "l131072_d10_s00")
    assert (t["needle_key"], t["expected"]) == ("4533779", "6752576")


@needs_corpus
def test_256k_prompt_fits_context_and_repeats_corpus(grid):
    t = next(x for x in grid if x["trial_id"] == "l262144_d50_s00")
    assert t["prompt_chars"] > 1_000_000          # 262144*4 chars + needle + template
    assert t["expected"] in t["prompt"] and t["expected"] != t["needle_key"]


@needs_corpus
def test_select_trials_diagnostic_is_prefix_of_full_cell(grid):
    sel = g1.select_trials(grid, [262144], [0.25, 0.5, 0.75], 3)
    assert len(sel) == 9
    assert {t["sample"] for t in sel} == {0, 1, 2} and {t["depth"] for t in sel} == {0.25, 0.5, 0.75}
    default = g1.select_trials(grid, g1.CONTEXT_LENGTHS)
    assert len(default) == 200


def test_resume_skips_scored_and_reruns_null(tmp_path):
    ts = fake_trials(3)
    d = tmp_path / "trials"
    d.mkdir()
    (d / f"trial_{ts[0]['trial_id']}.json").write_text(json.dumps({"score": True}))
    (d / f"trial_{ts[1]['trial_id']}.json").write_text(json.dumps({"score": None}))   # the 36 server-died trials
    calls = []

    def call(prompt):
        calls.append(prompt)
        return "the answer is 123", {"prompt_tokens": 5, "elapsed_s": 1.0}

    rc = g1.run_trials(ts, d, call, backend="gpu", log=lambda *_: None)
    assert rc == g1.RC_DONE and len(calls) == 2
    rec = json.loads((d / f"trial_{ts[1]['trial_id']}.json").read_text())
    assert rec["score"] is True and rec["backend"] == "gpu"
    assert not list(d.glob(".claim_*"))            # claims are released


def test_abort_after_consecutive_errors_not_whole_grid(tmp_path):
    ts = fake_trials(6)

    def dead(prompt):
        raise ConnectionError("refused")

    rc = g1.run_trials(ts, tmp_path / "t", dead, abort_after=2, log=lambda *_: None)
    assert rc == g1.RC_SERVER_ERRORS
    assert len(list((tmp_path / "t").glob("trial_*.json"))) == 2     # not 6 null files
    assert g1.is_pending(tmp_path / "t", ts[0])                       # null score stays pending


def test_max_requests_returns_pending_code(tmp_path):
    ts = fake_trials(5)
    rc = g1.run_trials(ts, tmp_path / "t", lambda p: ("123", {"elapsed_s": 0.1}), max_requests=2, log=lambda *_: None)
    assert rc == g1.RC_PENDING_MAX
    assert sum(not g1.is_pending(tmp_path / "t", t) for t in ts) == 2


def test_window_guard_uses_slowest_observed_trial(tmp_path):
    ts = fake_trials(4)
    clock = {"t": 1000.0}

    def call(prompt):
        clock["t"] += 100.0
        return "123", {"elapsed_s": 100.0}

    # deadline leaves room for 2 trials plus the 1.25x margin: after one 100 s trial the next needs 125 s
    rc = g1.run_trials(ts, tmp_path / "t", call, deadline=1000.0 + 100.0 + 124.0, est_trial_s=50.0,
                       now=lambda: clock["t"], log=lambda *_: None)
    assert rc == g1.RC_PENDING_WINDOW
    assert sum(not g1.is_pending(tmp_path / "t", t) for t in ts) == 1


def test_stopfile_between_trials(tmp_path):
    ts = fake_trials(3)
    stop = tmp_path / "STOP"

    def call(prompt):
        stop.write_text("x")
        return "123", {"elapsed_s": 0.1}

    rc = g1.run_trials(ts, tmp_path / "t", call, stopfile=str(stop), log=lambda *_: None)
    assert rc == g1.RC_PENDING_WINDOW
    assert sum(not g1.is_pending(tmp_path / "t", t) for t in ts) == 1


def test_claim_is_exclusive_and_stale_claims_are_taken_over(tmp_path):
    assert g1.claim_trial(tmp_path, "x", 3600, "a")
    assert not g1.claim_trial(tmp_path, "x", 3600, "b")
    old = time.time() - 7200
    os.utime(tmp_path / ".claim_x", (old, old))
    assert g1.claim_trial(tmp_path, "x", 3600, "b")


def test_claimed_trial_is_skipped_and_reported_pending(tmp_path):
    ts = fake_trials(2)
    d = tmp_path / "t"
    d.mkdir()
    assert g1.claim_trial(d, ts[0]["trial_id"], 3600, "other")
    rc = g1.run_trials(ts, d, lambda p: ("123", {"elapsed_s": 0.1}), log=lambda *_: None)
    assert rc == g1.RC_PENDING_MAX
    assert g1.is_pending(d, ts[0]) and not g1.is_pending(d, ts[1])


def _rec(length, depth, sample, score, backend="cpu"):
    return {"context_length": length, "depth": depth, "sample": sample, "score": score, "backend": backend}


def test_analyze_excludes_unscored_from_accuracy():
    trials = [_rec(131072, 0.1, i, True) for i in range(14)] + [_rec(131072, 0.25, i, None) for i in range(36)]
    rep = g1.analyze(trials)
    pl = rep["per_length"]["131072"]
    assert (pl["trials"], pl["hits"], pl["unscored"], pl["accuracy_pct"]) == (14, 14, 36, 100.0)


def test_analyze_bootstrap_vs_4k_and_diagnostic_n():
    trials = []
    for d in g1.DEPTHS:
        trials += [_rec(4096, d, s, True) for s in range(10)]
    for d in (0.25, 0.5, 0.75):
        trials += [_rec(262144, d, s, s == 0, "gpu") for s in range(3)]   # 1/3 hit per depth
    rep = g1.analyze(trials)
    b = rep["bootstrap"]["262144"]
    assert b["n_placements"] == 3 and b["n_trials_long"] == 9
    assert b["mean_drop_pp"] == pytest.approx(66.67, abs=0.1)
    assert rep["per_length"]["262144"]["backends"] == ["gpu"]


def test_diagnostic_verdict_and_shortest_drop():
    trials = []
    for d in g1.DEPTHS:
        trials += [_rec(4096, d, s, True) for s in range(10)]
        trials += [_rec(131072, d, s, True) for s in range(10)]
    for d in (0.25, 0.5, 0.75):
        trials += [_rec(262144, d, s, s == 0) for s in range(3)]
    rep = g1.analyze(trials)
    dv = verdict.diagnostic_verdict(rep)
    assert dv["checked"] and dv["escalate"] and dv["hits"] == 3
    assert verdict.shortest_drop_length(rep) == "262144"
    clean = g1.analyze([t for t in trials if t["context_length"] != 262144])
    assert verdict.shortest_drop_length(clean) == "131072"            # run_g1.sh rule: nothing drops -> 128K


def test_summary_written_with_checksums(tmp_path):
    d = tmp_path / "q8" / "trials"
    d.mkdir(parents=True)
    (d / "trial_a.json").write_text(json.dumps(_rec(4096, 0.1, 0, True)))
    rep = g1.write_summary(tmp_path / "q8")
    assert rep["per_length"]["4096"]["hits"] == 1
    assert (tmp_path / "q8" / "SHA256SUMS").read_text().strip().endswith("trial_a.json")
