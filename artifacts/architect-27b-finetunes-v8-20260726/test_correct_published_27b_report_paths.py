from __future__ import annotations

import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("path_correction", HERE / "correct_published_27b_report_paths.py")
assert SPEC and SPEC.loader
correction = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(correction)


def test_preflight_binds_original_package_and_corrects_only_appended_paths() -> None:
    table = correction.verify_original_finalization()
    rows = correction.corrected_rows(table)
    original = {row["arm"]: row for row in table["rows"]}
    corrected = {row["arm"]: row for row in rows}
    for arm in original:
        before, after = dict(original[arm]), dict(corrected[arm])
        before.pop("report"); after.pop("report")
        assert before == after
    assert all(not corrected[arm]["report"].startswith("/") for arm in correction.APPENDED)
    assert correction.preflight()["status"] == "PRECHECK_OK"
