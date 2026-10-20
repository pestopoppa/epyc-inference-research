"""AKX-ALL-29.4 write side: per-case gate detail is additive, nullable and inert."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from autokernel.loop import gates

HELP = ("Usage: test-backend-ops test -o <op> -b <backend> --output <format> "
        "--suite-seed <u64> --autokernel-properties\n")
REF = ("AK_REF_V1 metric=test_backend_ops_error/v1 observed={obs} tolerance=1e-07 "
       "comparisons=3 oracle=ggml_cpu_reference/v1")
PASSING = (
    "Testing 1 devices\n\nBackend 1/1: ROCm0\n"
    f"  MUL_MAT(type_a=q4_K,type_b=f32,m=16,n=1,k=256): {REF.format(obs='2.5e-09')} OK\n"
    f"  MUL_MAT(type_a=f16,type_b=f32,m=16,n=1,k=256): {REF.format(obs='5e-08')} OK\n"
    "  2/2 tests passed\n  Backend ROCm0: OK\n1/1 backends passed\nOK\n")
FAILING = (
    "Testing 1 devices\n\nBackend 1/1: ROCm0\n"
    "  MUL_MAT(type_a=q4_K,type_b=f32,m=16,n=1,k=256): FAIL\n"
    f"  MUL_MAT(type_a=f16,type_b=f32,m=16,n=1,k=256): {REF.format(obs='5e-08')} OK\n"
    "  1/2 tests passed\n  Backend ROCm0: FAIL\n0/1 backends passed\nFAIL\n")


def _gate(suite, *, reference=True):
    def invoke(argv, **_kw):
        if "--help" in argv:
            return SimpleNamespace(returncode=1, stdout=HELP, stderr="")
        return SimpleNamespace(returncode=0 if "FAIL" not in suite else 1,
                               stdout=suite, stderr="")
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=invoke):
        return gates.op_correctness(Path("/c"), op="MUL_MAT", require_reference=reference)


def _core(v):
    return (v.gate, v.passed, v.reason, v.detail)


def test_fields_written_on_passing_gate():
    v = _gate(PASSING)
    assert v.passed and len(v.case_records) == 2
    top = v.case_records[0]          # worst fraction of tolerance first
    assert top["op"] == "MUL_MAT" and top["types"] == {"type_a": "f16", "type_b": "f32"}
    assert top["observed"] == 5e-08 and top["tolerance"] == 1e-07
    assert top["fraction_of_tolerance"] == 0.5 and top["passed"] is True
    assert top["metric_id"] == "test_backend_ops_error/v1"
    assert top["oracle_id"] == "ggml_cpu_reference/v1"
    assert v.to_dict()["case_records"] == list(v.case_records)
    json.dumps(v.to_dict())


def test_unknown_fields_are_null_not_invented():
    v = _gate(FAILING)
    assert not v.passed
    failed = v.case_records[0]       # failing cases sort first
    assert failed["passed"] is False and failed["status"] == "fail"
    for key in ("observed", "tolerance", "fraction_of_tolerance", "metric_id", "oracle_id"):
        assert failed[key] is None
    for rec in v.case_records:       # the harness computes neither
        assert rec["output_dtype"] is None and rec["output_rms"] is None


def test_old_rows_and_plain_verdicts_unchanged():
    v = gates.Verdict("correctness", True, "r", "d")
    assert v.case_records is None
    assert v.to_dict() == {"gate": "correctness", "passed": True, "reason": "r", "detail": "d"}
    assert v == gates.Verdict("correctness", True, "r", "d", case_records=({"x": 1},))
    old_row = {"gate": "correctness", "passed": False, "reason": "x", "detail": None}
    assert "case_records" not in old_row and old_row["passed"] is False


def test_gate_verdict_identical_with_and_without_records():
    for suite, reference in ((PASSING, True), (FAILING, True), (PASSING, False),
                             (FAILING, False)):
        with_records = _gate(suite, reference=reference)
        with mock.patch.object(gates, "gate_case_records", return_value=None):
            without = _gate(suite, reference=reference)
        assert _core(with_records) == _core(without)
        assert without.case_records is None


def test_unparseable_output_yields_none_and_never_raises():
    assert gates.gate_case_records("usage text only", "ROCm0", "MUL_MAT") is None
    assert gates.gate_case_records("", "ROCm0", "MUL_MAT") is None


def test_sharded_merge_carries_records_and_same_verdict():
    shards = [("a", 2), ("b", 2)]
    seen = []

    def fake(build_dir, **kw):
        seen.append(kw["params_filter"])
        return gates.Verdict("correctness", True, case_records=({
            "op": "MUL_MAT", "passed": True, "fraction_of_tolerance": 0.1},))

    v = gates.op_correctness_sharded(Path("/c"), op="MUL_MAT", backend="ROCm0",
                                     resolved_recipe=None, case_shards=shards,
                                     _op_correctness=fake)
    assert v.passed and len(v.case_records) == 2 and sorted(seen) == ["a", "b"]
