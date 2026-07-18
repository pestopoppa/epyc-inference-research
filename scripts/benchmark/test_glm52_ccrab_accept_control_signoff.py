#!/usr/bin/env python3
"""Inference-free tests for glm52_ccrab_accept_control_signoff.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "glm52_ccrab_accept_control_signoff.py"
_SPEC = importlib.util.spec_from_file_location("glm52_ccrab_accept_control_signoff", _MODULE_PATH)
signoff_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_ccrab_accept_control_signoff"] = signoff_mod
_SPEC.loader.exec_module(signoff_mod)


def _signoff(
    *,
    status: str = "reviewed",
    decision: str | None = "hard_accept",
    reviewer: str | None = "manual-reviewer",
    reviewed_at: str | None = "2026-07-18T00:00:00Z",
    notes: str | None = "Complete executable/manual accept-control signoff.",
) -> dict:
    return {
        "status": status,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "decision": decision,
        "notes": notes,
    }


def _packet(*rows: dict) -> dict:
    return {
        "schema": "glm52_ccrab_accept_control_audit_packet.v1",
        "selected_n": len(rows),
        "rows": list(rows),
    }


def _row(row_id: str, *, signoff: dict | None = None) -> dict:
    return {
        "row_id": row_id,
        "task": "Fix a Python regression.",
        "candidate": "diff --git a/pkg/tests/test_bug.py b/pkg/tests/test_bug.py\n+def test_fixed():\n+    assert True\n",
        "signoff": signoff or _signoff(),
    }


def test_all_reviewed_hard_accepts_are_decision_grade():
    report = signoff_mod.build_report(
        _packet(_row("a"), _row("b")),
        min_hard_accepts=2,
        allow_unreviewed=False,
    )

    assert report["selected_n"] == 2
    assert report["hard_accept_n"] == 2
    assert report["rejected_or_ambiguous_n"] == 0
    assert report["unreviewed_n"] == 0
    assert report["decision_grade"] is True
    assert report["accepted_row_ids"] == ["a", "b"]


def test_unreviewed_rows_block_decision_grade_unless_explicitly_allowed():
    unreviewed = _row(
        "b",
        signoff=_signoff(status="unreviewed", decision=None, reviewer=None, reviewed_at=None, notes=None),
    )
    packet = _packet(_row("a"), unreviewed)

    blocked = signoff_mod.build_report(packet, min_hard_accepts=1, allow_unreviewed=False)
    allowed = signoff_mod.build_report(packet, min_hard_accepts=1, allow_unreviewed=True)

    assert blocked["hard_accept_n"] == 1
    assert blocked["unreviewed_n"] == 1
    assert blocked["unreviewed_row_ids"] == ["b"]
    assert blocked["decision_grade"] is False
    assert allowed["decision_grade"] is True


def test_rejected_or_ambiguous_rows_are_dropped_from_accepts():
    rejected = _row("b", signoff=_signoff(decision="reject_or_ambiguous", notes="Incomplete fix."))

    report = signoff_mod.build_report(
        _packet(_row("a"), rejected),
        min_hard_accepts=1,
        allow_unreviewed=False,
    )

    assert report["hard_accept_n"] == 1
    assert report["rejected_or_ambiguous_n"] == 1
    assert report["accepted_row_ids"] == ["a"]
    assert report["rejected_row_ids"] == ["b"]
    assert report["decision_grade"] is True


def test_invalid_reviewed_decision_fails_validation():
    packet = _packet(_row("a", signoff=_signoff(decision="maybe_accept")))

    try:
        signoff_mod.build_report(packet, min_hard_accepts=1, allow_unreviewed=False)
    except signoff_mod.ValidationError as exc:
        assert "unknown signoff.decision" in str(exc)
    else:
        raise AssertionError("expected invalid decision to fail")


def test_cli_writes_report_row_ids_and_oracle_notes(tmp_path):
    packet_path = tmp_path / "audit.json"
    packet_path.write_text(
        json.dumps(_packet(_row("a", signoff=_signoff(notes="Strong manual signoff.")))) + "\n",
        encoding="utf-8",
    )
    json_out = tmp_path / "report.json"
    row_ids_out = tmp_path / "row_ids.txt"
    oracle_notes_out = tmp_path / "oracle_notes.json"

    rc = signoff_mod.main([
        str(packet_path),
        "--min-hard-accepts",
        "1",
        "--json-out",
        str(json_out),
        "--row-ids-out",
        str(row_ids_out),
        "--oracle-notes-out",
        str(oracle_notes_out),
    ])

    assert rc == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    oracle_notes = json.loads(oracle_notes_out.read_text(encoding="utf-8"))
    assert report["decision_grade"] is True
    assert report["accepted_row_ids"] == ["a"]
    assert row_ids_out.read_text(encoding="utf-8") == "a\n"
    assert oracle_notes == {
        "a": {
            "notes": "Strong manual signoff.",
            "reviewer": "manual-reviewer",
            "reviewed_at": "2026-07-18T00:00:00Z",
        },
    }


def test_cli_writes_empty_row_ids_file_when_no_accepts(tmp_path):
    packet_path = tmp_path / "audit.json"
    unreviewed = _row(
        "a",
        signoff=_signoff(status="unreviewed", decision=None, reviewer=None, reviewed_at=None, notes=None),
    )
    packet_path.write_text(json.dumps(_packet(unreviewed)) + "\n", encoding="utf-8")
    row_ids_out = tmp_path / "row_ids.txt"

    rc = signoff_mod.main([
        str(packet_path),
        "--row-ids-out",
        str(row_ids_out),
    ])

    assert rc == 0
    assert row_ids_out.read_text(encoding="utf-8") == ""
