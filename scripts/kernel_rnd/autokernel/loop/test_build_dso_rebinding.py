"""Actual ELF loader-name rebinding across build version suffix changes."""
from pathlib import Path
import shutil
import subprocess

import pytest

from .run import _rebind_build_dso


@pytest.fixture
def libraries(tmp_path):
    # A tiny library exercises actual readelf and filesystem resolution; no model.
    source = tmp_path / "library.c"
    source.write_text("int test_value(void) { return 1; }\n")
    original = tmp_path / "libtest.so.0.0.10125"
    subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-soname,libtest.so.0",
                    str(source), "-o", str(original)], check=True)
    candidate_dir = tmp_path / "candidate" / "bin"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "libtest.so.0.0.10301"
    shutil.copyfile(original, candidate)
    (candidate_dir / "libtest.so.0").symlink_to(candidate.name)
    return original, candidate_dir, candidate


def test_actual_changed_version_uses_contained_soname(libraries):
    original, binary_dir, candidate = libraries
    assert _rebind_build_dso(original, binary_dir) == candidate


def test_existing_exact_filename_does_not_need_readelf(tmp_path):
    candidate = tmp_path / "libsame.so"
    candidate.write_bytes(b"unchanged existing fixture")
    assert _rebind_build_dso(Path("/old/bin/libsame.so"), tmp_path) == candidate


def test_missing_or_escaping_candidate_refuses(libraries):
    original, binary_dir, _ = libraries
    alias = binary_dir / "libtest.so.0"
    alias.unlink()
    with pytest.raises(FileNotFoundError):
        _rebind_build_dso(original, binary_dir)
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="escapes build"):
        _rebind_build_dso(original, binary_dir)
