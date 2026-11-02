"""Model-free integrity report tests."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.benchmark import v11_fa4b_integrity_report as report


def test_flags_nonfinite_and_established_garbage():
    assert report.classify_response('{"value": NaN}', "!!!!!!!!") == {
        "nonfinite_json": True, "garbage_text": True}
    assert report.classify_response('{"value": 1}', "a normal answer") == {
        "nonfinite_json": False, "garbage_text": False}
    with pytest.raises(ValueError):
        report.classify_response("[]", "answer")


def test_complete_report_counts_and_missing_records_refuse(tmp_path):
    manifest = {"scored": [{"id": "item"}], "greedy_ids": ["item"]}
    row = {"id": "item", "content": "good answer", "raw_response_text": '{"x": 2}'}
    for arm in ("ON", "OFF"):
        for phase in ("score", "greedy"):
            folder = tmp_path / arm / "q8-off" / phase
            folder.mkdir(parents=True)
            (folder / "000-item.json").write_text(json.dumps(row))
    base = {"cells": {"q8:off": {"agreement": False}}}
    with (patch.object(report.m2, "KVS", ("q8",)),
          patch.object(report.m2, "DRAFTS", ("off",)),
          patch.object(report.m2, "analyze", return_value=base),
          patch.object(report.m2, "validate_record"),
          patch.object(report.m2, "historical_row", return_value=None)):
        result = report.analyze(manifest, tmp_path)
        assert result["cells"]["q8:off"]["agreement"] is False
        assert result["cells"]["q8:off"]["response_integrity"]["ON"]["score"] == {
            "responses": 1, "nonfinite_json": 0, "garbage_text": 0}
        (tmp_path / "ON/q8-off/score/000-item.json").unlink()
        with pytest.raises(ValueError, match="incomplete"):
            report.analyze(manifest, tmp_path)


def test_original_analysis_failure_is_not_turned_into_zero_counts(tmp_path):
    with patch.object(report.m2, "analyze", side_effect=ValueError("failure record")):
        with pytest.raises(ValueError, match="failure record"):
            report.analyze({}, Path(tmp_path))
