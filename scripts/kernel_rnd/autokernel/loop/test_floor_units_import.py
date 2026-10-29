"""Bench floor records must not load serving through the unit vocabulary."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("serving_first", [False, True])
def test_record_unit_import_order(serving_first):
    package_root = Path(__file__).resolve().parents[2]
    code = """
import sys
sys.path[:0] = sys.argv[2:]
serving_first = sys.argv[1] == "True"
if serving_first:
    from autokernel.loop import serving
from autokernel.loop import bench, floor_units
assert ('autokernel.loop.serving' in sys.modules) is serving_first
for unit in floor_units.FLOOR_UNITS:
    assert bench.record_unit({'unit': unit}) == unit
assert bench.record_unit({'unit': 'unknown'}) is None
assert bench.record_unit({'schema': bench.CALIBRATION_SCHEMA}) == floor_units.UNIT_PROCESS
assert ('autokernel.loop.serving' in sys.modules) is serving_first
from autokernel.loop import serving
assert serving.FLOOR_UNITS is floor_units.FLOOR_UNITS
assert serving.UNIT_ARM == floor_units.UNIT_ARM
assert serving.UNIT_SESSION == floor_units.UNIT_SESSION
assert serving.UNIT_PROCESS == floor_units.UNIT_PROCESS == bench.FLOOR_UNIT
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(serving_first),
            str(package_root),
            str(package_root.parents[1]),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
