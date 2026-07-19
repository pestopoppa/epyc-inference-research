#!/usr/bin/env python3
"""Inference-free tests for glm52_ccrab_hard_negative_filter.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "glm52_ccrab_hard_negative_filter.py"
_SPEC = importlib.util.spec_from_file_location("glm52_ccrab_hard_negative_filter", _MODULE_PATH)
filter_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_ccrab_hard_negative_filter"] = filter_mod
_SPEC.loader.exec_module(filter_mod)


def _row(row_id: str, *, label: str = "reject", candidate: str = "diff --git a/x b/x\n+bad\n") -> dict:
    return {
        "row_id": row_id,
        "domain": "code",
        "source_benchmark": "c-crab",
        "source_suite": "python",
        "gold_label": label,
        "gold_confidence": "multi_oracle",
        "gold_source": "human_review_comment+testgen_oracle",
        "provenance": {"instance_id": "owner__repo-1@abc"},
        "candidate": candidate,
    }


def test_select_rows_requires_c_crab_python_multioracle_reject_full_candidate():
    rows = [
        _row("reject-a"),
        _row("reject-b"),
        _row("accept", label="accept"),
        {**_row("single"), "gold_confidence": "single_oracle"},
        {**_row("fragment"), "provenance": {"scoring_method": "substring"}},
        {**_row("wrong-suite"), "source_suite": "javascript"},
        {**_row("no-candidate"), "candidate": ""},
    ]

    selected = filter_mod.select_rows(rows, n=24, max_chars=15000, seed=42)

    assert {row.row_id for row in selected} == {"reject-a", "reject-b"}
    assert all(row.gold_source == "human_review_comment+testgen_oracle" for row in selected)


def test_cli_writes_row_ids_combined_and_markdown(tmp_path):
    corpus = tmp_path / "rows.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(_row("reject-a")),
                "{bad json",
                json.dumps(_row("reject-b")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    accept_ids = tmp_path / "accept.txt"
    accept_ids.write_text("# accept controls\naccept-a\naccept-b # keep\n", encoding="utf-8")
    json_out = tmp_path / "report.json"
    row_ids_out = tmp_path / "reject.txt"
    combined_out = tmp_path / "combined.txt"
    md_out = tmp_path / "report.md"

    rc = filter_mod.main(
        [
            "--corpus",
            str(corpus),
            "--n",
            "2",
            "--accept-row-ids",
            str(accept_ids),
            "--json-out",
            str(json_out),
            "--row-ids-out",
            str(row_ids_out),
            "--combined-row-ids-out",
            str(combined_out),
            "--md-out",
            str(md_out),
        ]
    )

    assert rc == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["matching_pool_n"] == 2
    assert report["selected_n"] == 2
    assert report["invalid_json_lines"] == 1
    assert report["decision_grade"] is False
    assert report["reject_side_decision_grade"] is True
    assert set(report["selected_row_ids"]) == {"reject-a", "reject-b"}
    assert set(row_ids_out.read_text(encoding="utf-8").splitlines()) == {"reject-a", "reject-b"}
    combined_lines = combined_out.read_text(encoding="utf-8").splitlines()
    assert "do_not_execute_live_until" in "\n".join(combined_lines[:4])
    assert "accept-a" in combined_lines
    assert "accept-b" in combined_lines
    assert "GC-shadow-repair4b.2b" in md_out.read_text(encoding="utf-8")


def test_cli_requires_combined_inputs_as_pair(tmp_path):
    rc = filter_mod.main(["--combined-row-ids-out", str(tmp_path / "combined.txt")])

    assert rc == 2
