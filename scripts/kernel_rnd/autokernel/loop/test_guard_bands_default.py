"""AKX-ALL-29.5a: the guard-band check runs by default on the GPU leg of `op_correctness`."""
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from autokernel.loop import gates, guard_bands

RAN = ("Backend 1/2: ROCm0\n  12/12 tests passed\n  Backend ROCm0: OK\n2/2 backends passed\nOK\n")
RAN_CPU = RAN.replace("ROCm0", "CPU")


def _receipt(violations=()):
    return guard_bands.RECEIPT_PREFIX + json.dumps({
        "schema": guard_bands.SCHEMA, "op": "MUL_MAT", "vars": "type_a=q4_K",
        "output_dtype": "f32", "output_rms": 0.5,
        "views": [{"name": "dst", "extent": 64, "offset": 4096}],
        "violations": list(violations)}) + "\n"


def _run(stdout, backend="ROCm0"):
    seen = {}

    def fake(argv, **kw):
        seen["env"] = kw.get("env") or {}
        return subprocess.CompletedProcess(argv, 0, stdout, "")
    with mock.patch.object(Path, "is_file", return_value=True), \
            mock.patch.object(gates.subprocess, "run", side_effect=fake):
        v = gates.op_correctness(Path("/nonexistent"), backend=backend)
    return v, seen["env"]


@pytest.fixture
def gate_default(monkeypatch):
    monkeypatch.delenv("AK_GUARD_BANDS_GATE", raising=False)


def test_default_is_on_for_gpu_only(gate_default):
    assert guard_bands.required_for("ROCm0")
    assert not guard_bands.required_for("CPU")
    assert not guard_bands.required_for("ROCm0", {"AK_GUARD_BANDS_GATE": "0"})


def test_gpu_run_sets_producer_env_and_passes_with_intact_receipt(gate_default):
    v, env = _run(_receipt() + RAN)
    assert env.get("AKX_GUARD_BANDS") == "1"
    assert v.passed, v
    rec = [r for r in v.case_records or () if r.get("output_dtype")]
    assert rec == [] or rec[0]["output_dtype"] == "f32"


def test_gpu_keep_refused_when_receipt_missing(gate_default):
    v, _ = _run(RAN)
    assert not v.passed and v.gate == "oracle_unavailable"
    assert "guard_band_receipt_missing" in v.reason


def test_gpu_touched_band_is_a_correctness_failure(gate_default):
    bad = {"view": "dst", "band": "tail", "first_offset": 0, "n_bytes": 4,
           "expected": 91, "observed": 0}
    v, _ = _run(_receipt([bad]) + RAN)
    assert not v.passed and v.gate == "correctness"
    assert "guard_band_violation" in v.reason


def test_cpu_leg_unchanged(gate_default):
    v, env = _run(RAN_CPU, backend="CPU")
    assert "AKX_GUARD_BANDS" not in env
    # CPU with a GPU-only gate: no receipt needed; the CPU independence rule may still refuse.
    assert "guard_band" not in v.reason


def test_opt_out_restores_old_behaviour(monkeypatch):
    monkeypatch.setenv("AK_GUARD_BANDS_GATE", "0")
    v, env = _run(RAN)
    assert "AKX_GUARD_BANDS" not in env and v.passed
