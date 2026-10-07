"""Synthetic controls for the read-only benchmark pin scanner."""
from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.benchmark import check_pin_staleness as scanner


def _fixture(tmp_path: Path, payload: bytes, source: str) -> tuple[Path, str]:
    root = tmp_path / "research"
    benchmark = root / "scripts/benchmark"
    benchmark.mkdir(parents=True)
    asset = benchmark / "asset.bin"
    asset.write_bytes(payload)
    module = benchmark / "runner.py"
    module.write_text(source, encoding="utf-8")
    return root, str(module.relative_to(root))


def test_literal_pin_reports_current_then_stale_content(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"original").hexdigest()
    root, relative = _fixture(
        tmp_path,
        b"original",
        f'from pathlib import Path\nDATA = Path("scripts/benchmark/asset.bin")\n'
        f'EXPECTED_DATA_SHA256 = "{expected}"\nfile_identity(DATA, EXPECTED_DATA_SHA256)\n',
    )

    first = scanner.scan_source(root, relative)
    assert len(first) == 1
    assert first[0]["status"] == "current"

    (root / "scripts/benchmark/asset.bin").write_bytes(b"changed")
    second = scanner.scan_source(root, relative)
    assert second[0]["status"] == "stale"
    assert second[0]["expected_sha256"] == expected
    assert second[0]["actual_sha256"] == hashlib.sha256(b"changed").hexdigest()


def test_missing_pin_target_is_distinct_from_mismatch(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"expected").hexdigest()
    root, relative = _fixture(
        tmp_path,
        b"unused",
        f'from pathlib import Path\nDATA = Path("scripts/benchmark/missing.bin")\n'
        f'EXPECTED_DATA_SHA256 = "{expected}"\nfile_identity(DATA, EXPECTED_DATA_SHA256)\n',
    )
    assert scanner.scan_source(root, relative)[0]["status"] == "missing"


def test_dynamic_call_and_unpaired_pin_remain_visible_as_unresolved(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"expected").hexdigest()
    root, relative = _fixture(
        tmp_path,
        b"unused",
        f'EXPECTED_DYNAMIC_SHA256 = "{expected}"\n'
        'def verify(path):\n    file_identity(path, EXPECTED_DYNAMIC_SHA256)\n'
        'EXPECTED_UNPAIRED_SHA256 = "' + expected + '"\n',
    )
    rows = scanner.scan_source(root, relative)
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"unresolved"}
    assert {row["pin"] for row in rows} == {
        "EXPECTED_DYNAMIC_SHA256", "EXPECTED_UNPAIRED_SHA256"}
