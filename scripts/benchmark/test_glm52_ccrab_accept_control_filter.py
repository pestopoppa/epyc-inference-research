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


def _row(row_id: str, *, candidate: str, confidence: str = "observation") -> dict:
    return {
        "row_id": row_id,
        "source_benchmark": "c-crab",
        "source_suite": "python",
        "gold_label": "accept",
        "gold_source": "merged_pr_accepted",
        "gold_confidence": confidence,
        "executable_oracle": None,
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
    rows = [
        _row("b", candidate=GOOD_PATCH),
        _row("a", candidate=GOOD_PATCH, confidence="multi_oracle"),
        {**_row("no-test", candidate="diff --git a/pkg/mod.py b/pkg/mod.py\n+value = 1\n")},
        {**_row("reject", candidate=GOOD_PATCH), "gold_label": "reject"},
        {**_row("dirty", candidate=GOOD_PATCH), "provenance": {"clean_control": False}},
    ]

    selected = filter_mod.select_rows(rows, n=24, max_chars=15000)

    assert [row.row_id for row in selected] == ["a", "b"]
    assert selected[0].hard_accept_control is True
    assert selected[1].hard_accept_control is False


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
    assert report["selected_row_ids"] == ["a"]
    assert report["decision_grade"] is False
    assert row_ids_out.read_text(encoding="utf-8") == "a\n"
