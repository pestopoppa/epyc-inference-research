#!/usr/bin/env python3
"""Inference-free tests for glm52_ccrab_accept_control_filter.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "glm52_ccrab_accept_control_filter.py"
_SPEC = importlib.util.spec_from_file_location("glm52_ccrab_accept_control_filter", _MODULE_PATH)
filter_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_ccrab_accept_control_filter"] = filter_mod
_SPEC.loader.exec_module(filter_mod)


def _row(
    row_id: str,
    *,
    candidate: str,
    confidence: str = "observation",
    executable_oracle: dict | None = None,
) -> dict:
    return {
        "row_id": row_id,
        "source_benchmark": "c-crab",
        "source_suite": "python",
        "gold_label": "accept",
        "gold_source": "merged_pr_accepted",
        "gold_confidence": confidence,
        "executable_oracle": executable_oracle,
        "defect_origin": "natural",
        "ambiguous_tail": False,
        "provenance": {"clean_control": True, "instance_id": "owner__repo-1@abc"},
        "candidate": candidate,
    }


GOOD_PATCH = """diff --git a/pkg/tests/test_bug.py b/pkg/tests/test_bug.py
--- a/pkg/tests/test_bug.py
+++ b/pkg/tests/test_bug.py
@@ -0,0 +1,3 @@
+def test_fixed():
+    assert run() == 1
"""


def test_select_rows_requires_clean_c_crab_accept_with_test_evidence():
    oracle = {
        "oracle_type": "testgen_fail_then_pass",
        "verdict": "pass",
        "source": "c-crab/stage4_agent_resolved",
        "resolution": "agent_resolved",
    }
    rows = [
        _row("b", candidate=GOOD_PATCH),
        _row("a", candidate=GOOD_PATCH, confidence="multi_oracle", executable_oracle=oracle),
        {**_row("no-test", candidate="diff --git a/pkg/mod.py b/pkg/mod.py\n+value = 1\n")},
        {**_row("reject", candidate=GOOD_PATCH), "gold_label": "reject"},
        {**_row("dirty", candidate=GOOD_PATCH), "provenance": {"clean_control": False}},
    ]

    selected = filter_mod.select_rows(rows, n=24, max_chars=15000)

    assert [row.row_id for row in selected] == ["a", "b"]
    assert selected[0].hard_accept_control is True
    assert selected[1].hard_accept_control is False


def test_select_rows_prefers_hard_accept_controls_before_observations():
    oracle = {
        "oracle_type": "testgen_fail_then_pass",
        "verdict": "pass",
        "source": "c-crab/stage3_testgen_verified",
        "resolution": "testgen_verified",
    }
    rows = [
        _row("a-observation", candidate=GOOD_PATCH),
        _row("b-hard", candidate=GOOD_PATCH, confidence="multi_oracle", executable_oracle=oracle),
        _row("c-hard", candidate=GOOD_PATCH, confidence="multi_oracle", executable_oracle=oracle),
    ]

    selected = filter_mod.select_rows(rows, n=2, max_chars=15000)

    assert [row.row_id for row in selected] == ["b-hard", "c-hard"]
    assert all(row.hard_accept_control for row in selected)


def test_cli_writes_json_and_row_ids(tmp_path):
    corpus = tmp_path / "rows.jsonl"
    corpus.write_text(json.dumps(_row("a", candidate=GOOD_PATCH)) + "\n", encoding="utf-8")
    json_out = tmp_path / "report.json"
    row_ids_out = tmp_path / "rows.txt"

    rc = filter_mod.main([
        "--corpus",
        str(corpus),
        "--json-out",
        str(json_out),
        "--row-ids-out",
        str(row_ids_out),
    ])

    assert rc == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["matching_pool_n"] == 1
    assert report["hard_accept_control_pool_n"] == 0
    assert report["selected_row_ids"] == ["a"]
    assert report["decision_grade"] is False
    assert row_ids_out.read_text(encoding="utf-8") == "a\n"


def test_cli_writes_audit_packet_with_full_candidate_context(tmp_path):
    corpus = tmp_path / "rows.jsonl"
    row = _row("a", candidate=GOOD_PATCH, confidence="multi_oracle")
    row["task"] = "Fix the regression and keep the public API stable."
    row["decontamination"] = {
        "repo": "owner/repo",
        "pull_number": 123,
        "base_commit": "abc123",
    }
    corpus.write_text(json.dumps(row) + "\n", encoding="utf-8")
    audit_out = tmp_path / "audit.json"

    rc = filter_mod.main([
        "--corpus",
        str(corpus),
        "--audit-packet-out",
        str(audit_out),
    ])

    assert rc == 0
    packet = json.loads(audit_out.read_text(encoding="utf-8"))
    assert packet["schema"] == "glm52_ccrab_accept_control_audit_packet.v1"
    assert packet["decision_grade"] is False
    assert packet["selected_n"] == 1
    packet_row = packet["rows"][0]
    assert packet_row["row_id"] == "a"
    assert packet_row["hard_accept_control"] is True
    assert packet_row["task"] == row["task"]
    assert packet_row["candidate"] == GOOD_PATCH
    assert packet_row["candidate_redacted_long_digit_runs"] is False
    assert packet_row["repo"] == "owner/repo"
    assert packet_row["signoff"]["status"] == "unreviewed"


def test_audit_packet_can_truncate_large_fields():
    row = _row("a", candidate=GOOD_PATCH + "x" * 50)
    row["task"] = "y" * 50
    selected = filter_mod.select_rows([row], n=1, max_chars=15000)

    packet = filter_mod.build_audit_packet(
        [row],
        selected,
        corpus=Path("rows.jsonl"),
        max_row_chars=10,
    )

    packet_row = packet["rows"][0]
    assert packet_row["task"] == "y" * 10
    assert packet_row["task_truncated"] is True
    assert packet_row["candidate"] == (GOOD_PATCH + "x" * 50)[:10]
    assert packet_row["candidate_truncated"] is True


def test_audit_packet_redacts_account_number_shaped_digit_runs():
    long_digit_run = "123456" + "789012"
    row = _row("a", candidate=GOOD_PATCH + f"\n+value = {long_digit_run}\n")
    selected = filter_mod.select_rows([row], n=1, max_chars=15000)

    packet = filter_mod.build_audit_packet([row], selected, corpus=Path("rows.jsonl"))

    packet_row = packet["rows"][0]
    assert long_digit_run not in packet_row["candidate"]
    assert "[redacted-long-digit-run]" in packet_row["candidate"]
    assert packet_row["candidate_redacted_long_digit_runs"] is True
