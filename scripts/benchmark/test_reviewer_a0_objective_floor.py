#!/usr/bin/env python3
"""Tests for reviewer_a0_objective_floor.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "reviewer_a0_objective_floor.py"
_SPEC = importlib.util.spec_from_file_location("reviewer_a0_objective_floor", _MODULE_PATH)
a0 = importlib.util.module_from_spec(_SPEC)
sys.modules["reviewer_a0_objective_floor"] = a0
_SPEC.loader.exec_module(a0)


def test_materialize_writes_gold_label_decisions(tmp_path):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        {
            "row_id": "row-a",
            "corpus_id": "nearmiss-v1",
            "domain": "code",
            "gold_label": "accept",
            "gold_source": "merged_pr_accepted",
            "gold_instrument_version": "v1",
        },
        {
            "row_id": "row-r",
            "corpus_id": "nearmiss-v1",
            "domain": "code",
            "gold_label": "reject",
            "gold_source": "multi_oracle",
            "gold_instrument_version": "v1",
        },
    ]
    corpus.write_text(
        "\n".join([json.dumps(rows[0]), '{"broken": ', json.dumps(rows[1])]) + "\n",
        encoding="utf-8",
    )
    row_ids = tmp_path / "row_ids.txt"
    row_ids.write_text("row-a\nrow-r\n", encoding="utf-8")
    out = tmp_path / "out"

    summary = a0.materialize(
        a0.parse_args(
            [
                "--corpus",
                str(corpus),
                "--row-ids-file",
                str(row_ids),
                "--output-dir",
                str(out),
                "--reviewer-id",
                "a0_test",
                "--protocol-attestation",
                "attested",
            ]
        )
    )

    decisions = [
        json.loads(line)
        for line in (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["n"] == 2
    assert [row["decision"] for row in decisions] == ["approve", "reject"]
    assert [row["confidence"] for row in decisions] == [1.0, 1.0]
    assert decisions[1]["tripwire"] is True
    assert manifest["measurement_protocol"] == "p_rev1"
    assert manifest["observation_only"] is False


def test_p_rev1_requires_attestation(tmp_path):
    corpus = tmp_path / "rows.jsonl"
    corpus.write_text(
        json.dumps({"row_id": "row-a", "gold_label": "accept"}) + "\n",
        encoding="utf-8",
    )
    row_ids = tmp_path / "row_ids.txt"
    row_ids.write_text("row-a\n", encoding="utf-8")
    args = a0.parse_args(
        [
            "--corpus",
            str(corpus),
            "--row-ids-file",
            str(row_ids),
            "--output-dir",
            str(tmp_path / "out"),
            "--protocol-attestation",
            "",
        ]
    )

    try:
        a0.materialize(args)
    except ValueError as exc:
        assert "--protocol-attestation" in str(exc)
    else:
        raise AssertionError("expected p_rev1 without attestation to fail")
