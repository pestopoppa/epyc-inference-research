"""R22-5 boundary corpus must actually run and refuse an old instrument."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from autokernel.loop import gates


def _suite(n, status="OK"):
    return (f"Backend 1/1: ROCm0\n  {n}/{n} tests passed\n"
            f"  Backend ROCm0: {status}\n1/1 backends passed\n")


def _run(marker=True, boundary_n=36, op="MUL_MAT"):
    seen = []
    def command(argv, **kwargs):
        boundary = kwargs["env"].get("AUTOKERNEL_BOUNDARY_CASE_SET")
        seen.append(boundary)
        return SimpleNamespace(stdout=_suite(boundary_n if boundary else 12),
                               stderr=("AUTOKERNEL_BOUNDARY_CASE_SET=r22-5-v1 cases=108\n"
                                       if boundary and marker else ""), returncode=0)
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=command):
        verdict = gates.boundary_matrix_correctness(Path("/mock"), op=op, backend="ROCm0")
    return verdict, seen


def test_explicit_boundary_executes_selected_suite():
    result, seen = _run()
    assert result.passed
    assert seen == ["r22-5-v1"]


def test_old_binary_refuses():
    result, _ = _run(marker=False)
    assert not result.passed and result.gate == "oracle_unavailable"


def test_missing_cases_refuse():
    result, _ = _run(boundary_n=35)
    assert not result.passed and result.gate == "oracle_unavailable"


def test_seeded_receipt_includes_boundary():
    import json
    from autokernel.execution import t0_provider

    ref = SimpleNamespace(oracle_id="ggml_cpu_reference/v1", observed=0.1,
                          tolerance=1.0, metric_id="nmse")
    case = SimpleNamespace(op="MUL_MAT", status="ok", passed=True,
                           reference=ref, params="m=33")
    lengths = [36]
    def parsed(_):
        n = lengths.pop(0)
        return SimpleNamespace(reconcile=lambda: None,
                               backends=[SimpleNamespace(name="ROCm0", skipped=False,
                                                         cases=[case] * n)])
    def command(argv, **kwargs):
        if "--help" in argv:
            return SimpleNamespace(stdout="--suite-seed --autokernel-properties", stderr="", returncode=0)
        boundary = kwargs["env"].get("AUTOKERNEL_BOUNDARY_CASE_SET")
        return SimpleNamespace(stdout=_suite(36 if boundary else 12),
                               stderr=("AUTOKERNEL_BOUNDARY_CASE_SET=r22-5-v1 cases=108\n"
                                       if boundary else ""), returncode=0)
    caps = SimpleNamespace(require=lambda _: None)
    with mock.patch.object(Path, "is_file", return_value=True), \
         mock.patch.object(gates.residency, "loader_env", return_value={}), \
         mock.patch.object(gates.subprocess, "run", side_effect=command), \
         mock.patch.object(t0_provider, "parse_backend_ops_help", return_value=caps), \
         mock.patch.object(t0_provider, "parse_backend_ops_console", side_effect=parsed):
        result = gates.boundary_matrix_correctness(Path("/mock"), op="MUL_MAT", backend="ROCm0", require_reference=True)
    assert result.passed
    assert json.loads(result.detail)["cases"] == 36


def test_other_admitted_gpu_ops_require_their_complete_boundary_matrix():
    for op, count in (("MUL_MAT_ID", 36), ("GATED_DELTA_NET", 9),
                      ("RMS_NORM", 9), ("RMS_NORM_MUL_ADD", 9),
                      ("FLASH_ATTN_EXT", 9)):
        result, seen = _run(op=op, boundary_n=count)
        assert result.passed and seen == ["r22-5-v1"]
        result, _ = _run(op=op, boundary_n=count - 1)
        assert not result.passed and result.gate == "oracle_unavailable"


def test_unmapped_cpu_graph_op_refuses_instead_of_claiming_coverage():
    with mock.patch.object(gates.subprocess, "run") as run:
        result = gates.boundary_matrix_correctness(Path("/mock"), op="ROPE", backend="CPU")
    assert result.gate == "oracle_unavailable" and not result.passed
    assert "no reviewed boundary-shape corpus" in result.reason
    run.assert_not_called()
