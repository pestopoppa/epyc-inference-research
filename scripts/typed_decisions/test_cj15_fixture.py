from __future__ import annotations

import shutil
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("cj15_fixture.py")
MODULE_SPEC = importlib.util.spec_from_file_location("cj15_fixture_test_target", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)
FixtureError = MODULE.FixtureError
validate_fixture = MODULE.validate_fixture


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "typed_decisions"
    / "cj15_judge_cascade_v1"
)


def test_frozen_fixture_validates_all_rows_and_keeps_failures_in_denominator() -> None:
    result = validate_fixture(FIXTURE)

    assert result["source_rows"] == 6
    assert result["stress"]["rows"] == 14
    assert result["stress"]["denominator"] == 14
    assert result["stress"]["correct"] == 12
    assert result["stress"]["abstentions"] == 2
    assert result["stress"]["invalid_primary"] == 4
    assert result["stress"]["fallbacks"] == 6
    assert result["stress"]["accuracy_over_all_rows"] == pytest.approx(12 / 14)


def test_fixture_rejects_changed_data_even_when_json_still_parses(tmp_path: Path) -> None:
    copied = tmp_path / "cj15"
    shutil.copytree(FIXTURE, copied)
    rows = copied / "stress_rows.jsonl"
    rows.write_bytes(rows.read_bytes().replace(b"confidence", b"confidenceX", 1))

    with pytest.raises(FixtureError, match="digest mismatch for stress_rows.jsonl"):
        validate_fixture(copied)


def test_fixture_manifest_freezes_both_orders_for_every_scenario() -> None:
    result = validate_fixture(FIXTURE)
    assert result["stress"]["rows"] == 7 * 2
