"""Hermetic receipts: no test launches an op suite or claims hardware."""

from pathlib import Path
from types import SimpleNamespace
import hashlib

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
