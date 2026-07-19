#!/usr/bin/env python3
"""Inference-free tests for glm52_ccrab_accept_control_review_sheet.py."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_MODULE_PATH = _SCRIPT_DIR / "glm52_ccrab_accept_control_review_sheet.py"
_SPEC = importlib.util.spec_from_file_location("glm52_ccrab_accept_control_review_sheet", _MODULE_PATH)
review_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_ccrab_accept_control_review_sheet"] = review_mod
_SPEC.loader.exec_module(review_mod)


def _packet(*rows: dict) -> dict:
    return {
        "schema": "glm52_ccrab_accept_control_audit_packet.v1",
        "selected_n": len(rows),
        "rows": list(rows),
    }


def _row(row_id: str) -> dict:
    return {
        "row_id": row_id,
        "instance_id": f"repo__project-{row_id}",
        "repo": "repo/project",
        "pull_number": 123,
        "candidate_chars": 42,
        "candidate_redacted_long_digit_runs": False,
        "task": "Fix the bug.",
        "candidate": "diff --git a/a.py b/a.py\n+fixed = True\n",
        "signoff": {
            "status": "unreviewed",
            "reviewer": None,
            "reviewed_at": None,
            "decision": None,
            "notes": None,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_mod.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_review_csv_generation_includes_recommendations(tmp_path):
    packet_path = tmp_path / "packet.json"
    rec_path = tmp_path / "recommendations.json"
    csv_out = tmp_path / "review.csv"
    summary_out = tmp_path / "summary.json"
    packet_path.write_text(json.dumps(_packet(_row("a"))) + "\n", encoding="utf-8")
    rec_path.write_text(
        json.dumps(
            {
                "recommendations": [
                    {
                        "row_id": "a",
                        "recommendation": "hard_accept_candidate",
                        "reason": "Patch includes direct fix and regression test.",
                        "format_concerns": ["check fixture literal"],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = review_mod.main(
        [
            str(packet_path),
            "--machine-recommendations",
            str(rec_path),
            "--review-csv-out",
            str(csv_out),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert rc == 0
    rows = list(csv.DictReader(csv_out.open(encoding="utf-8", newline="")))
    assert rows[0]["row_id"] == "a"
    assert rows[0]["machine_recommendation"] == "hard_accept_candidate"
    assert rows[0]["format_concerns"] == "check fixture literal"
    assert rows[0]["decision"] == ""
    assert json.loads(summary_out.read_text(encoding="utf-8"))["review_row_n"] == 1


def test_review_markdown_generation_includes_bounded_excerpts(tmp_path):
    packet_path = tmp_path / "packet.json"
    rec_path = tmp_path / "recommendations.json"
    md_out = tmp_path / "review.md"
    summary_out = tmp_path / "summary.json"
    packet_path.write_text(json.dumps(_packet(_row("a"))) + "\n", encoding="utf-8")
    rec_path.write_text(
        json.dumps(
            {
                "recommendations": [
                    {
                        "row_id": "a",
                        "recommendation": "hard_accept_candidate",
                        "reason": "Patch includes direct fix and regression test.",
                        "format_concerns": ["check fixture literal"],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = review_mod.main(
        [
            str(packet_path),
            "--machine-recommendations",
            str(rec_path),
            "--review-md-out",
            str(md_out),
            "--excerpt-chars",
            "20",
            "--summary-out",
            str(summary_out),
        ]
    )

    assert rc == 0
    text = md_out.read_text(encoding="utf-8")
    assert "GLM-5.2 C-CRAB Accept-Control Review Packet" in text
    assert "`a`" in text
    assert "check fixture literal" in text
    assert "...[truncated]" in text
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["review_md_written"] == str(md_out)


def test_apply_review_csv_writes_signed_packet(tmp_path):
    packet_path = tmp_path / "packet.json"
    csv_path = tmp_path / "review.csv"
    signed_out = tmp_path / "signed.json"
    summary_out = tmp_path / "summary.json"
    packet_path.write_text(json.dumps(_packet(_row("a"), _row("b"))) + "\n", encoding="utf-8")
    _write_csv(
        csv_path,
        [
            {
                **{field: "" for field in review_mod.CSV_FIELDS},
                "row_id": "a",
                "decision": "hard_accept",
                "reviewer": "operator",
                "reviewed_at": "2026-07-19T00:00:00Z",
                "notes": "Complete patch with regression test.",
            },
            {
                **{field: "" for field in review_mod.CSV_FIELDS},
                "row_id": "b",
                "decision": "reject_or_ambiguous",
                "reviewer": "operator",
                "reviewed_at": "2026-07-19T00:00:00Z",
                "notes": "Requires execution to know.",
            },
        ],
    )

    rc = review_mod.main(
        [
            str(packet_path),
            "--apply-review-csv",
            str(csv_path),
            "--signed-packet-out",
            str(signed_out),
            "--summary-out",
            str(summary_out),
        ]
    )

    assert rc == 0
    signed = json.loads(signed_out.read_text(encoding="utf-8"))
    assert signed["rows"][0]["signoff"]["decision"] == "hard_accept"
    assert signed["rows"][1]["signoff"]["decision"] == "reject_or_ambiguous"
    report = json.loads(summary_out.read_text(encoding="utf-8"))["signed_packet_report"]
    assert report["hard_accept_n"] == 1
    assert report["rejected_or_ambiguous_n"] == 1
    assert report["decision_grade"] is False


def test_apply_review_csv_rejects_row_order_drift(tmp_path):
    packet = _packet(_row("a"), _row("b"))
    csv_rows = [
        {**{field: "" for field in review_mod.CSV_FIELDS}, "row_id": "b"},
        {**{field: "" for field in review_mod.CSV_FIELDS}, "row_id": "a"},
    ]

    try:
        review_mod.apply_review_csv(packet, csv_rows, default_reviewer=None, default_reviewed_at=None)
    except review_mod.ReviewSheetError as exc:
        assert "exactly match packet order" in str(exc)
    else:
        raise AssertionError("expected row-order drift to fail")


def test_apply_review_csv_requires_notes_for_reviewed_decisions(tmp_path):
    packet = _packet(_row("a"))
    csv_rows = [
        {
            **{field: "" for field in review_mod.CSV_FIELDS},
            "row_id": "a",
            "decision": "hard_accept",
            "reviewer": "operator",
            "reviewed_at": "2026-07-19T00:00:00Z",
            "notes": "",
        }
    ]

    try:
        review_mod.apply_review_csv(packet, csv_rows, default_reviewer=None, default_reviewed_at=None)
    except review_mod.ReviewSheetError as exc:
        assert "notes are required" in str(exc)
    else:
        raise AssertionError("expected empty notes to fail")


def test_apply_review_csv_requires_reviewed_at_for_reviewed_decisions(tmp_path):
    packet = _packet(_row("a"))
    csv_rows = [
        {
            **{field: "" for field in review_mod.CSV_FIELDS},
            "row_id": "a",
            "decision": "hard_accept",
            "reviewer": "operator",
            "reviewed_at": "",
            "notes": "Complete patch with regression test.",
        }
    ]

    try:
        review_mod.apply_review_csv(packet, csv_rows, default_reviewer=None, default_reviewed_at=None)
    except review_mod.ReviewSheetError as exc:
        assert "reviewed_at is required" in str(exc)
    else:
        raise AssertionError("expected empty reviewed_at to fail")


def test_apply_review_csv_allows_explicit_default_reviewed_at(tmp_path):
    packet = _packet(_row("a"))
    csv_rows = [
        {
            **{field: "" for field in review_mod.CSV_FIELDS},
            "row_id": "a",
            "decision": "hard_accept",
            "reviewer": "operator",
            "reviewed_at": "",
            "notes": "Complete patch with regression test.",
        }
    ]

    signed = review_mod.apply_review_csv(
        packet,
        csv_rows,
        default_reviewer=None,
        default_reviewed_at="2026-07-19T00:00:00Z",
    )

    assert signed["rows"][0]["signoff"]["reviewed_at"] == "2026-07-19T00:00:00Z"
