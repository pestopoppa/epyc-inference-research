import os

import pytest

from scripts.benchmark.kvu19a2_immutable_build import inventory


def test_inventory_changes_with_bytes_and_refuses_unsafe_paths(tmp_path):
    source = tmp_path / "tree"
    source.mkdir(mode=0o700)
    path = source / "tile.cuh"
    path.write_bytes(b"before")
    before = inventory(source, expected_uid=os.geteuid())
    path.write_bytes(b"after")
    assert inventory(source, expected_uid=os.geteuid()) != before
    path.chmod(0o666)
    with pytest.raises(ValueError, match="permissions"):
        inventory(source, expected_uid=os.geteuid())
    path.chmod(0o600)
    (source / "escape").symlink_to("/tmp")
    with pytest.raises(ValueError, match="symlink"):
        inventory(source, expected_uid=os.geteuid())
