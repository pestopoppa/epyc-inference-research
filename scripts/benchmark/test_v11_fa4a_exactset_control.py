import copy
import hashlib
import json

import pytest

from scripts.benchmark.v11_fa4a_exactset_control import validate


def fixture(tmp_path):
    plan = {"schema": "fa4a-exact-control.v1", "case_ids": ["shape"], "repeats": 1}
    records = []
    for bias in (256, 1024):
        data = f"identical input {bias}".encode()
        (tmp_path / f"input-{bias}").write_bytes(data)
        for arm in ("ON", "OFF"):
            native = {"case_id": "shape", "bias": bias, "arm": arm, "repeat": 1,
                      "input_sha256": hashlib.sha256(data).hexdigest(),
                      "nonfinite_elements": int(bias == 1024 and arm == "OFF")}
            raw = json.dumps(native).encode()
            filename = f"{bias}-{arm}.json"
            (tmp_path / filename).write_bytes(raw)
            records.append({**native, "raw_file": filename, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                            "input_file": f"input-{bias}"})
    return plan, records


def test_reports_finite_control_and_ungraded_stress(tmp_path):
    plan, rows = fixture(tmp_path)
    result = validate(plan, rows, tmp_path)
    assert result["bias256_finite_control"] == "PASS"
    assert result["bias1024_nonfinite_elements_by_case"]["OFF"]["shape"] == [1]
    assert result["numeric_accuracy"] == result["promotion"] == "NOT_EVALUATED"


def test_missing_duplicate_and_bad_custody_refuse(tmp_path):
    plan, rows = fixture(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        validate(plan, rows[:-1], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        validate(plan, rows + rows[:1], tmp_path)
    altered = copy.deepcopy(rows)
    altered[0]["nonfinite_elements"] = 5
    with pytest.raises(ValueError, match="disagrees"):
        validate(plan, altered, tmp_path)
    (tmp_path / "input-256").write_bytes(b"different")
    with pytest.raises(ValueError, match="input bytes"):
        validate(plan, rows, tmp_path)
