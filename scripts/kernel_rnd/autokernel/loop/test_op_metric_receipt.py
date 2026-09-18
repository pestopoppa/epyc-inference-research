"""Fixture-only checks for the source-selected GPU reference-metric gate."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from autokernel.loop import gates


HELP = ("Usage: test-backend-ops test -o <op> -b <backend> --output <format> "
        "--suite-seed <u64> --autokernel-properties\n")
REFERENCE = (
    "Testing 1 devices\n\nBackend 1/1: ROCm0\n"
    "  MUL_MAT(type_a=q4_K,m=16,n=1,k=256): "
    "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
    "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 OK\n"
    "  1/1 tests passed\n  Backend ROCm0: OK\n1/1 backends passed\nOK\n")


def _run(help_text=HELP, suite=REFERENCE, *, rc=0):
    calls = []

    def invoke(argv, **_kwargs):
        calls.append(tuple(argv))
        if "--help" in argv:
            return SimpleNamespace(returncode=1, stdout=help_text, stderr="")
        return SimpleNamespace(returncode=rc, stdout=suite, stderr="")

    return calls, invoke


def test_selected_gpu_op_emits_existing_per_case_metric_receipt():
    calls, invoke = _run()
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=invoke):
        verdict = gates.op_correctness(Path("/candidate"), op="MUL_MAT",
                                       require_reference=True)
    assert verdict.passed
    assert len(calls) == 2
    assert calls[1][-3:] == ("--suite-seed", "71", "--autokernel-properties")
    receipt = json.loads(verdict.detail)
    assert receipt["worst_metric"] == "test_backend_ops_error/v1"
    assert receipt["metrics"] == ["test_backend_ops_error/v1"]
    assert receipt["cases"] == 1
    assert receipt["observed"] == 2.5e-09
    assert receipt["tolerance"] == 1e-07
    assert receipt["worst_fraction_of_tolerance"] == 0.025


def test_old_instrument_does_not_run_or_misclassify_suite():
    calls, invoke = _run(help_text=HELP.replace("--suite-seed <u64> ", ""))
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=invoke):
        verdict = gates.op_correctness(Path("/candidate"), require_reference=True)
    assert not verdict.passed and verdict.gate == "oracle_unavailable"
    assert len(calls) == 1


def test_missing_reference_on_passing_suite_is_unavailable():
    missing = REFERENCE.replace(
        "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
        "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 ", "")
    calls, invoke = _run(suite=missing)
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=invoke):
        verdict = gates.op_correctness(Path("/candidate"), require_reference=True)
    assert not verdict.passed and verdict.gate == "oracle_unavailable"
    assert len(calls) == 2


def test_cpu_candidate_reference_cannot_be_accepted_as_independent():
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.subprocess, "run") as invoke:
        verdict = gates.op_correctness(Path("/candidate"), backend="CPU",
                                       require_reference=True)
    assert not verdict.passed and verdict.gate == "oracle_unavailable"
    invoke.assert_not_called()


def test_live_source_gate_requests_metric_only_for_gpu():
    source = Path(gates.__file__).with_name("run.py").read_text()
    assert "require_reference=not cpu_launch" in source
