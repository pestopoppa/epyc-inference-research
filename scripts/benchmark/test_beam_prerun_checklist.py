#!/usr/bin/env python3
"""M-12c: BEAM pre-run checklist, fixtures only."""

from __future__ import annotations

import json
from pathlib import Path

import beam_prerun_checklist as bp

OK_SHA = bp.PROVENANCE_SHA


def _summary(headline=0.62, judge="gemma-judge", role="qwen-reader", fold="beam_macro", trunc=0):
    return {"fold": fold, "headline": headline, "judge_model": judge, "model_role": role,
            "truncated_rows": trunc}


def _receipt(**over):
    r = {"reader": {"model": "qwen-reader", "max_tokens": 2048, "prompt_sha256": "p1",
                    "closed_book_instruction": False},
         "ingest": {"roles": ["user", "assistant"], "chunking": "pair", "retrieval_budget": 8000},
         "dataset": {"path": "/x/ds.parquet", "sha256": OK_SHA}}
    for k, v in over.items():
        r[k] = {**r[k], **v} if isinstance(v, dict) else v
    return r


def _arms(**per):
    return [{"arm": n, "summary": _summary(), "receipt": _receipt()} for n in ("full", "rag", "trace")]


def _check(arms, **kw):
    kw.setdefault("hash_file", lambda p: OK_SHA)
    rep = bp.check_arms(arms, **kw)
    return rep, {i["item"]: i["status"] for i in rep["items"]}


def test_all_pass_clears(tmp_path):
    f = tmp_path / "ds.parquet"; f.write_bytes(b"x")
    arms = _arms()
    for a in arms:
        a["receipt"]["dataset"]["path"] = str(f)
    rep, st = _check(arms)
    assert set(st.values()) == {bp.PASS} and rep["cleared"]


def test_historical_runs_without_receipts_are_unknown_not_pass():
    arms = [{"arm": n, "summary": _summary(trunc=2), "receipt": None} for n in ("full", "rag", "trace")]
    rep, st = _check(arms)
    assert st[1] == bp.PASS and st[4] == bp.PASS and st[8] == bp.PASS
    assert {st[i] for i in (2, 3, 5, 6, 7)} == {bp.UNKNOWN} and not rep["cleared"]


def test_item1_binarised_headline_fails():
    arms = _arms(); arms[0]["summary"]["fold"] = "rubric_micro"
    assert _check(arms)[1][1] == bp.FAIL


def test_item2_user_only_fails():
    arms = _arms(); arms[1]["receipt"]["ingest"]["roles"] = ["user"]
    assert _check(arms)[1][2] == bp.FAIL


def test_item3_low_max_tokens_and_truncation_fail():
    arms = _arms(); arms[0]["receipt"]["reader"]["max_tokens"] = 512
    assert _check(arms)[1][3] == bp.FAIL
    arms = _arms(); arms[2]["summary"]["truncated_rows"] = 3
    assert _check(arms)[1][3] == bp.FAIL


def test_item4_judge_equals_reader_or_varies():
    arms = _arms(); arms[0]["summary"]["judge_model"] = "qwen-reader.gguf"
    assert _check(arms)[1][4] == bp.FAIL
    arms = _arms(); arms[1]["summary"]["judge_model"] = "other-judge"
    assert _check(arms)[1][4] == bp.FAIL


def test_item5_digest_mismatch_and_missing_file(tmp_path):
    arms = _arms()
    arms[0]["receipt"]["dataset"]["path"] = str(tmp_path / "absent")
    assert _check(arms)[1][5] == bp.FAIL
    f = tmp_path / "ds"; f.write_bytes(b"x")
    arms = _arms()
    for a in arms:
        a["receipt"]["dataset"]["path"] = str(f)
    _, st = _check(arms, hash_file=lambda p: "0" * 64)
    assert st[5] == bp.FAIL


def test_item6_budget_mismatch_fails_and_single_arm_unknown():
    arms = _arms(); arms[2]["receipt"]["ingest"]["retrieval_budget"] = 4000
    assert _check(arms)[1][6] == bp.FAIL
    assert _check([a for a in _arms() if a["arm"] != "trace"])[1][6] == bp.UNKNOWN


def test_item7_closed_book_mismatch_fails():
    arms = _arms(); arms[2]["receipt"]["reader"]["closed_book_instruction"] = True
    assert _check(arms)[1][7] == bp.FAIL


def test_item8_far_below_floor_is_defect_but_full_arm_exempt():
    arms = _arms(); arms[1]["summary"]["headline"] = 0.30
    assert _check(arms)[1][8] == bp.FAIL
    arms = _arms(); arms[0]["summary"]["headline"] = 0.30  # full-context arm: floor not comparable
    assert _check(arms)[1][8] == bp.PASS


def test_load_arm_reads_summary_and_receipt(tmp_path):
    d = tmp_path / "run"; d.mkdir()
    (d / "scored-beam-full.json").write_text(json.dumps({"summary": _summary(), "per_question": []}))
    assert bp.load_arm(d, "full")["receipt"] is None
    (d / "beam_run_receipt.json").write_text(json.dumps(_receipt()))
    a = bp.load_arm(d, "full")
    assert a["summary"]["headline"] == 0.62 and a["receipt"]["reader"]["max_tokens"] == 2048
