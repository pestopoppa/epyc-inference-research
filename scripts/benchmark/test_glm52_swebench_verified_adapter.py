#!/usr/bin/env python3
"""No-inference tests for glm52_swebench_verified_adapter.py."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "glm52_swebench_verified_adapter.py"
_SPEC = importlib.util.spec_from_file_location("glm52_swebench_verified_adapter", _MODULE_PATH)
adapter = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_swebench_verified_adapter"] = adapter
_SPEC.loader.exec_module(adapter)


def _record(**overrides):
    row = {
        "repo": "astropy/astropy",
        "instance_id": "astropy__astropy-12907",
        "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n",
        "problem_statement": "Fix the separability matrix for nested compound models.",
        "FAIL_TO_PASS": json.dumps(["astropy/modeling/tests/test_separable.py::test_nested"]),
        "PASS_TO_PASS": json.dumps(["astropy/modeling/tests/test_separable.py::test_existing"]),
        "test_patch": "diff --git a/test.py b/test.py\n",
        "difficulty": "15 min - 1 hour",
        "version": "4.3",
        "environment_setup_commit": "298ccb478e6bf092953bca67a3d29dc6c35f6752",
    }
    row.update(overrides)
    return row


def test_normalizes_record_to_patch_review_oracle_contract():
    row = adapter.normalize_swebench_verified_record(
        _record(),
        row_index=7,
        gold_instrument_version="file-sha256:test",
    )
    payload = row.to_dict()

    assert payload["row_id"].startswith("glm52-swebench-verified:")
    assert payload["repo"] == "astropy/astropy"
    assert payload["instance_id"] == "astropy__astropy-12907"
    assert payload["problem_statement"] == payload["task"]
    assert payload["patch"] == payload["candidate"]
    assert payload["base_commit"] == "d16bfe05a744909de4b27f5875fe0d4ed41ce607"
    assert payload["FAIL_TO_PASS"] == ["astropy/modeling/tests/test_separable.py::test_nested"]
    assert payload["PASS_TO_PASS"] == ["astropy/modeling/tests/test_separable.py::test_existing"]
    assert payload["gold_label"] == "accept"
    assert payload["gold_source"] == "swe-bench-verified"
    assert payload["gold_confidence"] == "test_oracle"
    assert payload["gold_instrument_version"] == "file-sha256:test"
    assert payload["task_kind"] == "patch_review_oracle"
    assert payload["provenance"]["source_row_index"] == 7
    assert payload["provenance"]["test_patch_present"] is True


def test_normalizes_predecoded_test_lists():
    row = adapter.normalize_swebench_verified_record(
        _record(FAIL_TO_PASS=["a::test"], PASS_TO_PASS=["b::test"]),
        row_index=0,
        gold_instrument_version="file-sha256:test",
    )

    assert row.FAIL_TO_PASS == ["a::test"]
    assert row.PASS_TO_PASS == ["b::test"]


def test_rejects_missing_required_mechanical_fields():
    with pytest.raises(ValueError, match="missing non-empty string field 'patch'"):
        adapter.normalize_swebench_verified_record(
            _record(patch=""),
            row_index=0,
            gold_instrument_version="file-sha256:test",
        )


def test_rejects_malformed_test_oracle_fields():
    with pytest.raises(ValueError, match="FAIL_TO_PASS.*JSON list string"):
        adapter.normalize_swebench_verified_record(
            _record(FAIL_TO_PASS="not-json"),
            row_index=0,
            gold_instrument_version="file-sha256:test",
        )


def test_selection_is_deterministic():
    rows = adapter.normalize_swebench_verified_records(
        [
            _record(instance_id=f"repo__project-{idx}", patch=f"diff {idx}", FAIL_TO_PASS=[f"fail::{idx}"])
            for idx in range(8)
        ],
        gold_instrument_version="file-sha256:test",
    )

    selected1 = adapter.select_rows(rows, n=4, seed=52)
    selected2 = adapter.select_rows(rows, n=4, seed=52)
    selected3 = adapter.select_rows(rows, n=4, seed=53)

    assert [row.row_id for row in selected1] == [row.row_id for row in selected2]
    assert len(selected1) == 4
    assert [row.row_id for row in selected1] != [row.row_id for row in selected3]


def test_parquet_loading_missing_pyarrow_has_clear_message(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pyarrow.parquet" or name.startswith("pyarrow."):
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="requires pyarrow.*normalize_swebench_verified_records"):
        adapter.read_parquet_records(tmp_path / "missing.parquet")


def test_cli_dry_run_writes_plan_and_rows(monkeypatch, tmp_path):
    data_path = tmp_path / "test.parquet"
    data_path.write_bytes(b"fake parquet bytes")
    rows = [_record(instance_id=f"repo__project-{idx}", patch=f"diff {idx}", FAIL_TO_PASS=[f"fail::{idx}"]) for idx in range(3)]

    monkeypatch.setattr(adapter, "read_parquet_records", lambda path: rows)

    plan = tmp_path / "plan.json"
    rows_out = tmp_path / "rows.jsonl"
    rc = adapter.main(
        [
            "--path",
            str(data_path),
            "--n",
            "2",
            "--seed",
            "52",
            "--out-plan",
            str(plan),
            "--out-rows-jsonl",
            str(rows_out),
        ]
    )

    assert rc == 0
    plan_data = json.loads(plan.read_text())
    assert plan_data["schema"] == adapter.SCHEMA
    assert plan_data["mode"] == "dry-run"
    assert plan_data["execution"]["inference_allowed"] is False
    assert plan_data["dataset"]["file_sha256"] == adapter.sha256_file(data_path)
    assert plan_data["dataset"]["gold_instrument_version"].startswith("file-sha256:")
    assert plan_data["dataset"]["selected"]["n"] == 2
    written = [json.loads(line) for line in rows_out.read_text().splitlines()]
    assert len(written) == 2
    assert {row["gold_label"] for row in written} == {"accept"}
    assert {row["task_kind"] for row in written} == {"patch_review_oracle"}
