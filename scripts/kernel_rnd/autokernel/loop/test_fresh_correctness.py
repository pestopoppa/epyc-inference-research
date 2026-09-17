"""Hermetic receipts: no test launches an op suite or claims hardware."""

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import hashlib
from unittest import mock

from autokernel.loop import bench
from autokernel.loop import fresh_correctness as fresh
from autokernel.loop.integrity import CandidateIntegrity


HELP = ("Usage: test-backend-ops test -o <op> -b <backend> --output <format> "
        "--suite-seed <seed> --autokernel-properties\n")
REFERENCE = (
    "Testing 1 devices\n\nBackend 1/1: ROCm0\n"
    "  MUL_MAT(type_a=q4_K,m=16,n=1,k=256): "
    "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
    "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 OK\n"
    "  1/1 tests passed\n  Backend ROCm0: OK\n1/1 backends passed\nOK\n")
PROPERTY = (
    "Testing 1 devices\n\nBackend 1/1: CPU\n"
    "  MUL_MAT(type_a=q4_K,m=16,n=1,k=256): "
    "AK_PROP_V1 metric=raw_buffer/v1 residual=2.5e-09 "
    "tolerance=1e-07 passed=1 suite_seed=71 OK\n"
    "  1/1 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n")


class FakeRunner:
    def __init__(self, suite: str, *, help_text: str = HELP, exit_code: int = 0):
        self.suite, self.help_text, self.exit_code = suite, help_text, exit_code
        self.calls = []

    def __call__(self, argv, env, timeout_s):
        self.calls.append((tuple(argv), dict(env), timeout_s))
        if "--help" in argv:
            return SimpleNamespace(returncode=1, stdout=self.help_text, stderr="")
        return SimpleNamespace(returncode=self.exit_code, stdout=self.suite, stderr="")


def plan(tmp_path: Path, *, backend="ROCm0", paths=("ggml/src/ggml-cuda/mmvq.cu",)):
    binary = tmp_path / "build" / "bin" / "test-backend-ops"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"mock binary identity")
    candidate = tmp_path / "candidate"
    anchor = tmp_path / "anchor"
    for root in (candidate, anchor):
        instrument = root / "tests" / "test-backend-ops.cpp"
        instrument.parent.mkdir(parents=True)
        instrument.write_text("reviewed test instrument\n")
    integrity = CandidateIntegrity(paths=paths, tree="a" * 40, findings=())
    return fresh.Plan(binary=binary,
                      expected_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      candidate_root=candidate, anchor_root=anchor,
                      arm_id="attempt-1/A/0", recipe_hash="b" * 64,
                      backend=backend, op="MUL_MAT", suite_seed=71,
                      candidate_integrity=integrity)


def test_gpu_reviewed_reference_is_report_only(tmp_path):
    selected = plan(tmp_path)
    runner = FakeRunner(REFERENCE)
    row = fresh.collect(selected, invoke=runner, env={"PATH": "/usr/bin"})
    assert row["status"] == "reference_valid"
    assert row["verdict"] == "passed"
    assert row["ranked_sample"] is False
    assert row["authority"] == "report_only"
    assert row["reference_cases"] == row["compared_cases"] == 1
    assert row["raw_stdout"] == REFERENCE
    assert len(runner.calls) == 2
    assert "--suite-seed" in runner.calls[1][0]
    assert "71" in runner.calls[1][0]


def test_cpu_candidate_reference_is_not_independent(tmp_path):
    selected = plan(tmp_path, backend="CPU", paths=("ggml/src/ggml-cpu/ggml-cpu.c",))
    row = fresh.collect(selected, invoke=FakeRunner(PROPERTY), env={})
    assert row["status"] == "property_only"
    assert row["verdict"] == "passed"
    assert row["reference_cases"] == 0
    assert row["property_checks"] == 1


def test_missing_seed_flag_refuses_before_suite(tmp_path):
    selected = plan(tmp_path)
    runner = FakeRunner(REFERENCE, help_text=HELP.replace("--suite-seed <seed> ", ""))
    row = fresh.collect(selected, invoke=runner, env={})
    assert row["status"] == "oracle_unavailable"
    assert row["verdict"] == "unavailable"
    assert len(runner.calls) == 1


def test_binary_swap_refuses_before_any_invocation(tmp_path):
    selected = plan(tmp_path)
    selected.binary.write_bytes(b"different binary")
    runner = FakeRunner(REFERENCE)
    row = fresh.collect(selected, invoke=runner, env={})
    assert row["status"] == "oracle_unavailable"
    assert runner.calls == []


def test_missing_structured_reference_is_unavailable(tmp_path):
    selected = plan(tmp_path)
    plain = REFERENCE.replace(
        "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
        "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 ", "")
    row = fresh.collect(selected, invoke=FakeRunner(plain), env={})
    assert row["status"] == "oracle_unavailable"
    assert row["verdict"] == "unavailable"


def test_failed_unverified_suite_is_not_a_correctness_failure(tmp_path):
    selected = plan(tmp_path)
    plain = REFERENCE.replace(
        "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
        "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 ", "")
    row = fresh.collect(selected, invoke=FakeRunner(plain, exit_code=1), env={})
    assert row["status"] == "oracle_unavailable"
    assert row["verdict"] == "unavailable"
    assert row["suite_exit_code"] == 1


def test_instrument_source_mismatch_cannot_claim_reference(tmp_path):
    selected = plan(tmp_path)
    (selected.candidate_root / "tests" / "test-backend-ops.cpp").write_text("modified\n")
    runner = FakeRunner(REFERENCE)
    row = fresh.collect(selected, invoke=runner, env={})
    assert row["status"] == "oracle_unavailable"
    assert row["instrument_source_outcome"] == "FAIL"
    assert runner.calls == []


def test_instrument_source_mismatch_cannot_claim_host_property(tmp_path):
    selected = plan(tmp_path, backend="CPU", paths=("ggml/src/ggml-cpu/ggml-cpu.c",))
    (selected.candidate_root / "tests" / "test-backend-ops.cpp").write_text("modified\n")
    row = fresh.collect(selected, invoke=FakeRunner(PROPERTY), env={})
    assert row["status"] == "oracle_unavailable"


def test_cpu_candidate_local_reference_alone_is_not_independent(tmp_path):
    selected = plan(tmp_path, backend="CPU", paths=("ggml/src/ggml-cpu/ggml-cpu.c",))
    row = fresh.collect(selected, invoke=FakeRunner(
        REFERENCE.replace("ROCm0", "CPU")), env={})
    assert row["status"] == "oracle_unavailable"
    assert row["reference_cases"] == 1
    assert row["property_checks"] == 0


def test_suite_failure_is_not_a_passing_receipt(tmp_path):
    selected = plan(tmp_path)
    row = fresh.collect(selected, invoke=FakeRunner(REFERENCE, exit_code=1), env={})
    assert row["status"] == "reference_valid"
    assert row["verdict"] == "failed"


def test_skipped_backend_cannot_satisfy_reference_vacuously(tmp_path):
    selected = plan(tmp_path)
    skipped = ("Testing 1 devices\n\nBackend 1/1: ROCm0\nSkipping\n"
               "  0/0 tests passed\n  Backend ROCm0: OK\n"
               "1/1 backends passed\nOK\n")
    row = fresh.collect(selected, invoke=FakeRunner(skipped), env={})
    assert row["status"] == "oracle_unavailable"
    assert row["verdict"] == "unavailable"
    assert row["compared_cases"] == 0


def test_gpu_compare_opt_in_runs_fresh_suite_after_each_ranked_arm(tmp_path):
    selected = plan(tmp_path)
    runner = FakeRunner(REFERENCE)
    timed = []

    def fake_run_once(binary, model, **kwargs):
        timed.append(str(binary))
        return (100.0 if str(binary) == "/anchor" else 110.0,
                {"resident": True, "peak_vram_bytes": 1 << 31,
                 "peak_kfd_processes": 1})

    def check(arm, pair, hardening_seed):
        # New, declared suite seed per measured arm. The benchmark hardening
        # seed is a join identity, not the independent op-suite seed.
        assert isinstance(hardening_seed, int)
        arm_plan = replace(selected, arm_id=f"{arm.name}/{pair}",
                           suite_seed=selected.suite_seed + len(runner.calls))
        return fresh.collect(arm_plan, invoke=runner, env={})

    with mock.patch.object(bench, "run_once", side_effect=fake_run_once):
        row = bench.compare(bench.Arm("anchor", Path("/anchor")),
                            bench.Arm("candidate", Path("/candidate")),
                            Path("/model.gguf"), pp=0, tg=128, pairs=2,
                            warmup_pairs=1, fresh_check=check, calibrated=False)
    assert len(timed) == 6  # Two warmups, then four ranked invocations.
    assert len(runner.calls) == 8  # Help + seeded suite for each ranked arm.
    assert len(row.fresh_correctness) == 4
    assert [receipt["arm"] for receipt in row.fresh_correctness] == [
        "anchor", "candidate", "anchor", "candidate"]
    assert [receipt["pair"] for receipt in row.fresh_correctness] == [1, 1, 2, 2]
    assert all(receipt["status"] == "reference_valid"
               for receipt in row.fresh_correctness)
    assert all("raw_stdout" not in receipt for receipt in row.fresh_correctness)
    assert abs(row.effect - 0.1) < 1e-12
    assert row.decisive is None


def test_gpu_compare_default_off_and_observer_failure_cannot_change_effect():
    def fake_run_once(binary, model, **kwargs):
        return (100.0 if str(binary) == "/anchor" else 110.0,
                {"resident": True, "peak_vram_bytes": 1 << 31,
                 "peak_kfd_processes": 1})

    args = (bench.Arm("anchor", Path("/anchor")),
            bench.Arm("candidate", Path("/candidate")), Path("/model.gguf"))
    with mock.patch.object(bench, "run_once", side_effect=fake_run_once):
        plain = bench.compare(*args, pp=0, tg=128, pairs=1, warmup_pairs=0,
                              calibrated=False)
        failed = bench.compare(*args, pp=0, tg=128, pairs=1, warmup_pairs=0,
                               calibrated=False, fresh_check=lambda *_: 1 / 0)
    assert plain.to_dict().get("fresh_correctness") is None
    assert len(failed.fresh_correctness) == 2
    assert all(receipt["status"] == "oracle_unavailable"
               for receipt in failed.fresh_correctness)
    assert failed.effect == plain.effect
    assert failed.decisive == plain.decisive


def test_gpu_fresh_check_refuses_promotable_comparison_before_running():
    with mock.patch.object(bench, "run_once") as timed:
        try:
            bench.compare(bench.Arm("anchor", Path("/anchor")),
                          bench.Arm("candidate", Path("/candidate")),
                          Path("/model.gguf"), pp=0, tg=128, pairs=1,
                          fresh_check=lambda *_: {})
        except ValueError as exc:
            assert "report-only" in str(exc)
        else:
            raise AssertionError("fresh check admitted a promotable comparison")
        timed.assert_not_called()


def test_gpu_fresh_summary_caps_caller_text():
    def fake_run_once(binary, model, **kwargs):
        return (100.0, {"resident": True, "peak_vram_bytes": 1 << 31,
                        "peak_kfd_processes": 1})

    def oversized(*_):
        return {"schema": fresh.SCHEMA, "authority": "report_only",
                "ranked_sample": False, "arm_id": "x" * 10000,
                "recipe_hash": "b" * 64, "suite_seed": 71,
                "status": "oracle_unavailable", "verdict": "unavailable",
                "reason": "y" * 10000, "raw_stdout": "z" * 10000}

    with mock.patch.object(bench, "run_once", side_effect=fake_run_once):
        row = bench.compare(bench.Arm("anchor", Path("/anchor")),
                            bench.Arm("candidate", Path("/candidate")),
                            Path("/model.gguf"), pp=0, tg=128, pairs=1,
                            warmup_pairs=0, calibrated=False, fresh_check=oversized)
    assert all(len(receipt["arm_id"]) == len(receipt["reason"]) == 512
               and "raw_stdout" not in receipt for receipt in row.fresh_correctness)
