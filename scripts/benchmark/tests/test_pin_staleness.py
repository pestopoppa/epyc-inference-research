"""Synthetic controls for the read-only benchmark pin scanner."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.benchmark import check_pin_staleness as scanner

ZERO_SHA256 = "0" * 64


def _fixture(tmp_path: Path, payload: bytes, source: str) -> tuple[Path, str]:
    root = tmp_path / "research"
    benchmark = root / "scripts/benchmark"
    benchmark.mkdir(parents=True)
    asset = benchmark / "asset.bin"
    asset.write_bytes(payload)
    module = benchmark / "runner.py"
    module.write_text(source, encoding="utf-8")
    return root, str(module.relative_to(root))


def _pin_source(path_expr: str, digest_expr: str) -> str:
    return (
        "from pathlib import Path\n"
        f"DATA = {path_expr}\n"
        f"EXPECTED_DATA_SHA256 = {digest_expr}\n"
        "file_identity(DATA, EXPECTED_DATA_SHA256)\n"
    )


def _git_track(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "scripts/benchmark"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=NI08 fixture",
         "-c", "user.email=ni08-fixture@example.invalid", "commit", "-q", "-m",
         "synthetic scanner fixture"],
        check=True,
    )


def test_literal_pin_reports_current_then_stale_content(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"original").hexdigest()
    root, relative = _fixture(
        tmp_path, b"original",
        _pin_source('Path("scripts/benchmark/asset.bin")', f'"{expected}"'),
    )

    first = scanner.scan_source(root, relative)
    assert len(first) == 1
    assert first[0]["status"] == "current"

    (root / "scripts/benchmark/asset.bin").write_bytes(b"changed")
    second = scanner.scan_source(root, relative)
    assert second[0]["status"] == "stale"
    assert second[0]["expected_sha256"] == expected
    assert second[0]["actual_sha256"] == hashlib.sha256(b"changed").hexdigest()


def test_keyword_file_identity_arguments_are_checked(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"original").hexdigest()
    source = (
        "from pathlib import Path\n"
        'DATA = Path("scripts/benchmark/asset.bin")\n'
        f'EXPECTED_DATA_SHA256 = "{expected}"\n'
        "file_identity(path=DATA, expected_sha=EXPECTED_DATA_SHA256)\n"
    )
    root, relative = _fixture(tmp_path, b"original", source)
    rows = scanner.scan_source(root, relative)
    assert len(rows) == 1 and rows[0]["status"] == "current"


def test_missing_pin_target_is_distinct_from_mismatch(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"expected").hexdigest()
    root, relative = _fixture(
        tmp_path, b"unused",
        _pin_source('Path("scripts/benchmark/missing.bin")', f'"{expected}"'),
    )
    row = scanner.scan_source(root, relative)[0]
    assert row["status"] == "missing"
    assert row["expected_sha256"] == expected


def test_dynamic_missing_argument_and_unpaired_pin_remain_unresolved(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"expected").hexdigest()
    source = (
        f'EXPECTED_DYNAMIC_SHA256 = "{expected}"\n'
        "def verify(path):\n    file_identity(path)\n"
        f'EXPECTED_UNPAIRED_SHA256 = "{expected}"\n'
    )
    root, relative = _fixture(tmp_path, b"unused", source)
    rows = scanner.scan_source(root, relative)
    assert len(rows) == 3
    assert sum("file_identity requires" in row.get("reason", "") for row in rows) == 1
    assert sum("no statically paired" in row.get("reason", "") for row in rows) == 2
    assert all(row["status"] == "unresolved" for row in rows)


def test_symlink_parent_outside_and_fifo_are_refused_without_following(tmp_path: Path) -> None:
    root, relative = _fixture(
        tmp_path, b"payload",
        _pin_source('Path("scripts/benchmark/link.bin")', repr(ZERO_SHA256)),
    )
    benchmark = root / "scripts/benchmark"
    (benchmark / "link.bin").symlink_to(benchmark / "asset.bin")
    row = scanner.scan_source(root, relative)[0]
    assert row["status"] == "unresolved" and "symlink" in row["reason"]

    (benchmark / "link.bin").unlink()
    (benchmark / "linked-dir").symlink_to(benchmark, target_is_directory=True)
    (benchmark / "runner.py").write_text(
        _pin_source('Path("scripts/benchmark/linked-dir/asset.bin")', repr(ZERO_SHA256)),
        encoding="utf-8",
    )
    row = scanner.scan_source(root, relative)[0]
    assert row["status"] == "unresolved"

    (benchmark / "runner.py").write_text(
        _pin_source('Path("scripts/benchmark/cycle.bin")', repr(ZERO_SHA256)),
        encoding="utf-8",
    )
    (benchmark / "cycle.bin").symlink_to("cycle.bin")
    row = scanner.scan_source(root, relative)[0]
    assert row["status"] == "unresolved"

    (benchmark / "runner.py").write_text(
        _pin_source('Path("/tmp/outside-checkout.bin")', repr(ZERO_SHA256)),
        encoding="utf-8",
    )
    row = scanner.scan_source(root, relative)[0]
    assert row["status"] == "unresolved" and "outside" in row["reason"]

    (benchmark / "fifo.bin").unlink(missing_ok=True)
    os.mkfifo(benchmark / "fifo.bin")
    (benchmark / "runner.py").write_text(
        _pin_source('Path("scripts/benchmark/fifo.bin")', repr(ZERO_SHA256)),
        encoding="utf-8",
    )
    row = scanner.scan_source(root, relative)[0]
    assert row["status"] == "unresolved" and "regular file" in row["reason"]

    external = tmp_path / "external-synthetic.bin"
    external.write_bytes(b"synthetic only")
    os.link(external, benchmark / "hardlink.bin")
    (benchmark / "runner.py").write_text(
        _pin_source('Path("scripts/benchmark/hardlink.bin")', repr(ZERO_SHA256)),
        encoding="utf-8",
    )
    row = scanner.scan_source(root, relative)[0]
    assert row["status"] == "unresolved" and "multiply linked" in row["reason"]


def test_replacement_during_read_is_reported_unresolved(tmp_path: Path, monkeypatch) -> None:
    expected = hashlib.sha256(b"original").hexdigest()
    root, relative = _fixture(
        tmp_path, b"original",
        _pin_source('Path("scripts/benchmark/asset.bin")', f'"{expected}"'),
    )
    target = root / "scripts/benchmark/asset.bin"
    target_inode = target.stat().st_ino
    original_read = scanner.os.read
    replaced = False

    def replace_after_first_target_read(fd: int, size: int) -> bytes:
        nonlocal replaced
        data = original_read(fd, size)
        if not replaced and os.fstat(fd).st_ino == target_inode and data:
            replacement = target.with_suffix(".replacement")
            replacement.write_bytes(b"replacement")
            replacement.replace(target)
            replaced = True
        return data

    monkeypatch.setattr(scanner.os, "read", replace_after_first_target_read)
    row = scanner.scan_source(root, relative)[0]
    assert replaced
    assert row["status"] == "unresolved" and "changed while" in row["reason"]


def test_inspect_tree_and_cli_exit_contract_on_tracked_synthetic_checkout(
    tmp_path: Path, capsys,
) -> None:
    expected = hashlib.sha256(b"original").hexdigest()
    root, relative = _fixture(
        tmp_path, b"original",
        _pin_source('Path("scripts/benchmark/asset.bin")', f'"{expected}"'),
    )
    _git_track(root)

    report = scanner.inspect_tree(root)
    assert report["tracked_python_files"] == 1
    assert report["counts"] == {"current": 1, "stale": 0, "missing": 0, "unresolved": 0}
    assert scanner.main(["--root", str(root), "--require-resolved"]) == 0
    capsys.readouterr()

    (root / "scripts/benchmark/asset.bin").write_bytes(b"changed")
    assert scanner.main(["--root", str(root)]) == 1
    capsys.readouterr()

    (root / "scripts/benchmark/asset.bin").write_bytes(b"original")
    (root / "scripts/benchmark/runner.py").write_text(
        _pin_source('Path("scripts/benchmark/missing.bin")', f'"{expected}"'),
        encoding="utf-8",
    )
    assert scanner.main(["--root", str(root)]) == 1
    capsys.readouterr()

    (root / "scripts/benchmark/runner.py").write_text(
        f'EXPECTED_DYNAMIC_SHA256 = "{expected}"\n'
        "def verify(path):\n    file_identity(path, EXPECTED_DYNAMIC_SHA256)\n",
        encoding="utf-8",
    )
    assert scanner.main(["--root", str(root)]) == 0
    capsys.readouterr()
    assert scanner.main(["--root", str(root), "--require-resolved"]) == 2
    capsys.readouterr()


def test_json_binds_checker_and_scanned_source_to_git_snapshot(tmp_path: Path, monkeypatch) -> None:
    expected = hashlib.sha256(b"original").hexdigest()
    root, relative = _fixture(
        tmp_path, b"original",
        _pin_source('Path("scripts/benchmark/asset.bin")', f'"{expected}"'),
    )
    checker_copy = root / "scripts/benchmark/check_pin_staleness.py"
    checker_copy.write_bytes(Path(scanner.__file__).read_bytes())
    _git_track(root)

    first = scanner.inspect_tree(root)
    encoded_first = json.dumps(first, sort_keys=True)
    repeated = scanner.inspect_tree(root)
    assert json.dumps(repeated, sort_keys=True) == encoded_first
    assert first["root_git"]["stable"] is True
    assert first["root_git"]["before"]["available"] is True
    assert first["root_git"]["before"]["head"]
    assert first["root_git"]["before"]["tracked_dirty"] is False
    assert first["root_git"]["before"]["tracked_status_sha256"]
    assert first["root_git"]["stable"] is True
    assert first["checker_source"]["stable"] is True
    assert first["checker_source"]["before"]["status"] == "read"
    assert first["checker_source"]["after"]["status"] == "read"
    assert first["checker_source"]["matches_scanned_copy"] is True
    assert "not an atomic whole-checkout snapshot" in first["stability_scope"]
    source_identity = next(item for item in first["source_file_identities"]
                           if item["path"] == relative)
    assert source_identity["sha256"] == hashlib.sha256(
        (root / relative).read_bytes()).hexdigest()

    original_list = scanner.tracked_python_files

    def mutate_checker_during_scan(scan_root: Path) -> list[str]:
        checker_copy.write_bytes(checker_copy.read_bytes() + b"\n# synthetic source change\n")
        return original_list(scan_root)

    monkeypatch.setattr(scanner, "tracked_python_files", mutate_checker_during_scan)
    changed = scanner.inspect_tree(root)
    changed_checker = next(item for item in changed["source_file_identities"]
                           if item["path"] == "scripts/benchmark/check_pin_staleness.py")
    assert changed["root_git"]["stable"] is False
    assert changed["root_git"]["after"]["tracked_dirty"] is True
    assert changed["checker_source"]["stable"] is False
    assert changed["checker_source"]["before"]["sha256"] != \
        changed["checker_source"]["after"]["sha256"]
    assert changed_checker["sha256"] != first["checker_source"]["before"]["sha256"]
    assert changed["checker_source"]["matches_scanned_copy"] is False
    assert changed["complete"] is False
    assert any(row.get("reason") ==
               "executing checker differs from the scanned tracked checker source"
               for row in changed["rows"])
    snapshots = iter([
        {"available": True, "head": "a" * 40, "tracked_dirty": True,
         "tracked_change_count": 1, "tracked_status_sha256": "1" * 64},
        {"available": True, "head": "a" * 40, "tracked_dirty": True,
         "tracked_change_count": 1, "tracked_status_sha256": "2" * 64},
    ])
    monkeypatch.setattr(scanner, "tracked_python_files", original_list)
    monkeypatch.setattr(scanner, "_git_snapshot", lambda _root: next(snapshots))
    status_changed = scanner.inspect_tree(root)
    assert status_changed["root_git"]["stable"] is False
    assert status_changed["root_git"]["before"]["tracked_change_count"] == 1
    assert status_changed["root_git"]["after"]["tracked_change_count"] == 1
    assert status_changed["root_git"]["before"]["tracked_status_sha256"] != \
        status_changed["root_git"]["after"]["tracked_status_sha256"]
    assert status_changed["complete"] is False
