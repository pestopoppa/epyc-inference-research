from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "run_inf37_iq2_fancy_simd_ab.py"
SPEC = importlib.util.spec_from_file_location("inf37_iq2_ab", PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def sql(rows: list[tuple[str, float, int]]) -> str:
    head = """CREATE TABLE test_backend_ops (
op_params TEXT, supported INTEGER, passed INTEGER, time_us REAL, n_runs INTEGER);
"""
    return head + "".join(
        "INSERT INTO test_backend_ops VALUES "
        f"('type_a=iq2_xxs,type_b=f32,m=4096,n={n},k=14336',1,1,{time_us},{runs});\n"
        for n, time_us, runs in rows)


def test_balanced_orders_are_exact() -> None:
    orders = runner.balanced_orders(10)
    assert orders.count(("baseline", "candidate")) == 5
    assert orders.count(("candidate", "baseline")) == 5
    with pytest.raises(ValueError):
        runner.balanced_orders(3)


def test_git_status_preserves_porcelain_index_column(tmp_path, monkeypatch) -> None:
    class Result:
        stdout = " M ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp\n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Result())
    assert runner.git_status(tmp_path).startswith(" M ")


def test_linkage_prepends_build_local_library_path(tmp_path, monkeypatch) -> None:
    binary = tmp_path / "bin" / "test-backend-ops"
    binary.parent.mkdir()
    binary.write_bytes(b"executable")
    binary.chmod(0o755)
    seen = {}

    class Result:
        stdout = f"libggml-cpu.so.0 => {binary.parent}/libggml-cpu.so.0\n"

    def fake_run(*args, **kwargs):
        seen.update(kwargs["env"])
        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    row = runner.linkage(binary)
    assert seen["LD_LIBRARY_PATH"].split(":")[0] == str(binary.parent)
    assert str(binary.parent) in row["ggml_cpu_row"]


def test_parse_sql_rows_requires_exact_cells() -> None:
    rows = runner.parse_sql_rows(sql([(1, 100.0, 10), (512, 200.0, 5)]))
    assert [row["n"] for row in rows] == [1, 512]
    with pytest.raises(RuntimeError, match="expected exact cells"):
        runner.parse_sql_rows(sql([(1, 100.0, 10)]))


def test_summary_uses_lower_time_as_better() -> None:
    invocations = []
    for block in range(4):
        for arm, scale in (("baseline", 1.0), ("candidate", 0.8)):
            invocations.append({
                "block": block, "arm": arm,
                "rows": [
                    {"n": 1, "time_us": 100 * scale},
                    {"n": 512, "time_us": 200 * scale},
                ],
            })
    summary = runner.summarize(invocations, 4)
    assert summary["direction"] == "lower_is_better"
    assert all(row["all_candidate_faster"] for row in summary["cells"])
    assert all(row["median_candidate_speedup_fraction"] == pytest.approx(0.25)
               for row in summary["cells"])


def test_belief_measurements_are_arm_specific_and_identity_bound() -> None:
    invocations = []
    for block in range(4):
        for arm, scale in (("baseline", 1.0), ("candidate", 0.8)):
            invocations.append({
                "block": block, "arm": arm,
                "rows": [
                    {"n": 1, "time_us": 100 * scale},
                    {"n": 512, "time_us": 200 * scale},
                ],
            })
    summary = runner.summarize(invocations, 4)
    source = {
        "commit": "a" * 40,
        "candidate_diff_sha256": "b" * 64,
        "baseline_source_sha256": "c" * 64,
        "candidate_source_sha256": "d" * 64,
    }
    binaries = {
        arm: {
            "path": f"/build/{arm}/test-backend-ops",
            "sha256": digest * 64,
            "ggml_cpu_row": f"libggml-cpu => /build/{arm}/libggml-cpu.so",
        }
        for arm, digest in (("baseline", "e"), ("candidate", "f"))
    }
    rows = runner.belief_measurements(
        summary, blocks=4, source_identity=source,
        binary_identity=binaries, claim_id="akclaim-test")
    assert len(rows) == 4
    assert {(row["extra"]["shape"]["n"], row["extra"]["arm"])
            for row in rows} == {
                (1, "baseline"), (1, "candidate"),
                (512, "baseline"), (512, "candidate"),
            }
    assert all(row["metric_direction"] == "lower_better" for row in rows)
    assert all(row["reps"] == 4 for row in rows)
    assert all(row["extra"]["resource_claim_receipt"] == "akclaim-test"
               for row in rows)
    assert next(row for row in rows if row["measurement_id"] ==
                "iq2_xxs_n1_candidate_median_time_us")["value"] == 80.0
