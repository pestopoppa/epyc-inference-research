#!/usr/bin/env python3
"""Tests for `execution/t0_provider.py`.

HOW THESE TESTS ARE BUILT
-------------------------
Every parser test runs against **real recorded tool output** in `testdata/`,
captured on this host on 2026-08-03 by actually invoking the tools. Each file
carries its argv, its exit status and a `---8<---` marker; everything below the
marker is verbatim. Nothing in this file invents a tool's output format.

Four of those captures are findings in their own right and each one has a test
that FAILS without the guard it motivated:

  * `recorded_t0_backend_ops_console_skip.txt` — `-b ROCm0` on a CPU-only build:
    `Skipping`, `1/1 backends passed`, `OK`, **exit 0**, zero cases.
  * `recorded_t0_backend_ops_console_cpu_skip.txt` — no `-b` at all: `Skipping
    CPU backend`, same clean exit, zero cases.
  * `recorded_t0_backend_ops_mandatory_ops.txt` — `-o MUL_MAT,MUL_MAT_ID` with a
    `-p` filter: 286 case lines, **all of them MUL_MAT**. `MUL_MAT_ID` — the MoE
    expert-routing op that `correctness.MANDATORY_BACKEND_OPS` exists to force —
    was requested by name and never ran, and the tool said `OK`. This is
    `kernel_eval.sh`'s defect reproduced with a modern flag.
  * `recorded_t0_linkage_fail.txt` — the experimental binary under this
    container's ambient `LD_LIBRARY_PATH`: five libraries, all of ggml,
    resolving out of the FROZEN PRODUCTION TREE.

The process tests spawn real processes (`/bin/sleep`, `/usr/bin/env`) and kill
only the pids they captured. They take under two seconds and touch no model.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[2]))

from autokernel import schemas                                    # noqa: E402
from autokernel.evaluator import api, correctness, integrity, recipes  # noqa: E402
from autokernel.execution import t0_provider as t0                # noqa: E402
from autokernel.execution import cpu_region_claim               # noqa: E402

TESTDATA = _HERE.parent / "testdata"
MARKER = "---8<--- verbatim below this line ---8<---"


def recorded(name: str) -> str:
    """Load a recorded capture, dropping only the provenance header."""
    text = (TESTDATA / name).read_text(encoding="utf-8")
    if MARKER not in text:
        raise AssertionError(f"{name} has no provenance marker; it may not be a real capture")
    return text.split(MARKER, 1)[1].lstrip("\n")


# =============================================================================
# Fixtures that are NOT recorded output are built here and labelled as such
# =============================================================================

# Real SHA-256 values, not `"a" * 64`. `schemas.is_placeholder_digest` refuses a
# single repeated hex character precisely because no hash ever emits one, and
# this module's `_req_sha256` calls it — so a lazy fixture is refused by the
# same rule that refuses a fabricated anchor. These are
# sha256(b"ak-t0-test-fixture-<X>").
SHA_A = "e97e8ca8f0f142ac245595e63b98303aa1f9971827ef474693c21ad8b1013896"
SHA_B = "533b46c31ced63c1b4631c7a1569231018ed56f90875931be1b6657f3b7e5e4d"
SHA_C = "cf4047c088463b8eda4e4ff4a935a84968d77207f25c046d1d375550faf506ad"
SHA_D = "345a34511f86b38c32c99a6581188df0cd17a31570b11b8e957005fda70b8765"
SHA_E = "6ece250d9e12238aff48865301134000a545e0d21fd6d331c33584932c60a5f2"
SHA_F = "b8d88ca61c6c566bf04116fc7b695eb0d2127551c44b088d24e186e1709db7c3"
COMMIT_ANCHOR = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
COMMIT_CANDIDATE = "9f3c1a77b2d4e6058c19ab3e7d5f04b8c6172e9d"
PLACEHOLDER = "0" * 64

WORKTREE = "/mnt/raid0/llm/llama.cpp-experimental"
BUILD_DIR = "/mnt/raid0/llm/llama.cpp-experimental/build-t0"
BIN_DIR = "/mnt/raid0/llm/llama.cpp-experimental/build-t0/bin"
PROD_TREE = "/mnt/raid0/llm/llama.cpp"


class FakeClaim:
    """Minimal `HeldClaim`. Structural duck-typing is the seam, by design."""

    def __init__(self, held: bool = True, claim_id: str = "akclaim-cpu-0-95") -> None:
        self._held = held
        self._id = claim_id

    @property
    def claim_id(self) -> str:
        return self._id

    def is_held(self) -> bool:
        return self._held

    def describe(self) -> str:
        return f"cpu region 0-95, held={self._held}"


def capture(argv, stdout="", stderr="", exit_code=0, env=(), orphans=()):
    return t0.CompletedProcess(
        argv=tuple(argv), env=tuple(env), cwd=WORKTREE, exit_code=exit_code,
        stdout=stdout, stderr=stderr, duration_s=0.5, timed_out=False, signalled=False,
        orphans=tuple(orphans))


def candidate_build(**overrides):
    kwargs = dict(
        worktree=WORKTREE, build_dir=BUILD_DIR, source_commit=COMMIT_CANDIDATE,
        source_sha256=SHA_A, binary=f"{BIN_DIR}/llama-cli", library_path=BIN_DIR,
        test_backend_ops=f"{BIN_DIR}/test-backend-ops")
    kwargs.update(overrides)
    return t0.CandidateBuild(**kwargs)


def tools():
    return t0.ToolPaths(bash="/bin/bash",
                        verify_ggml_linkage_sh=str(_HERE.parents[4] / "scripts" / "utils"
                                                   / "verify_ggml_linkage.sh"),
                        cmake="/usr/bin/cmake")


def op_suite_plan(**overrides):
    kwargs = dict(backend_filter="CPU", ops=("MUL_MAT", "MUL_MAT_ID"),
                  suite_id="test-backend-ops/v1", suite_source_sha256=SHA_A)
    kwargs.update(overrides)
    return t0.OpSuitePlan(**kwargs)


def execution_plan(**overrides):
    kwargs = dict(
        candidate=candidate_build(), tools=tools(), op_suite=op_suite_plan(),
        dispatch=t0.DispatchTracePlan(derived_surface=("MUL_MAT", "MUL_MAT_ID")),
        candidate_diff_text="")
    kwargs.update(overrides)
    return t0.T0ExecutionPlan(**kwargs)


def generation_plan(**overrides):
    kwargs = dict(prompt="The capital of France is", prompt_ref="ak-prompt-001",
                  n_predict=32, seed=42)
    kwargs.update(overrides)
    return t0.GenerationPlan(**kwargs)


def anchor_capture(**overrides):
    kwargs = dict(source_commit=COMMIT_ANCHOR, binary_sha256=SHA_B, linkage_sha256=SHA_C)
    kwargs.update(overrides)
    return t0.AnchorCapture(**kwargs)


def evaluation_request(*, anchor=None, source_sha256=SHA_A, binary_sha256=SHA_D,
                       linkage_sha256=SHA_E, determinism_class="not_measured",
                       repeats=0):
    return api.EvaluationRequest(
        event_id="ake-0001", campaign_id="ak-0001", candidate_id="akc-0001", tier="T0",
        backend="llama_cpu", phase="decode", cell_class="operator_microbench",
        protocol_id=api.PROTOCOL_ID,
        artifact=api.ArtifactIdentity(source_sha256=source_sha256,
                                      binary_sha256=binary_sha256,
                                      linkage_sha256=linkage_sha256),
        anchor=anchor,
        evaluator=api.EvaluatorIdentity(id="ak-eval/v1", bundle_sha256=SHA_A,
                                        runtime_source_label_ref="ref://bundle"),
        scope_denominator=api.ScopeDenominator(machine_subset="full", numa_nodes=(),
                                               devices=(), cores=96),
        scope_manifest_sha256=SHA_A, co_residency="single",
        determinism=api.DeterminismReport(determinism_class=determinism_class,
                                          same_seed_repeat_runs=repeats),
        metric="op_throughput_gflops", metric_direction="higher_better", reps=1,
        change_class="parameter", anchor_tier="T0", transfer_ratio_to=(),
        created_at="2026-08-03T22:00:00Z", campaign_controls=None, calibration=None)


def t0_policy():
    return correctness.T0Policy(
        required_backend_ops=correctness.MANDATORY_BACKEND_OPS,
        symbol_shrinkage_reject_ratio=0.6,
        diff_ceiling=correctness.DiffComplexityCeiling(
            backend="llama_cpu", max_changed_lines=400, max_files_touched=10,
            shared_core_forces_review=True),
        determinism_min_runs=3, coherence_tolerance_floor=0.98,
        policy_ref="ak-policy/v1")


# =============================================================================
# A. `test-backend-ops` console parsing, against recorded output
# =============================================================================

class BackendOpsConsoleParsing(unittest.TestCase):

    def test_stateful_receipt_preserves_the_complete_triad(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SSM_SCAN(type=f32): AK_STATE_V1 inputs=1 initial_equal=1 "
                "input_immutable=1 final_outputs=1 suite_seed=4711 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        case = t0.parse_backend_ops_console(text).cases[0]
        self.assertEqual(case.stateful.input_count, 1)
        self.assertTrue(case.stateful.initial_equal)
        self.assertTrue(case.stateful.input_immutable)
        self.assertEqual(case.stateful.final_output_count, 1)

    def test_passing_case_cannot_claim_a_mutated_state_input(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  GATED_DELTA_NET(type=f32): AK_STATE_V1 inputs=1 initial_equal=1 "
                "input_immutable=0 final_outputs=1 suite_seed=4711 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_console(text)

    def test_structured_reference_receipt_is_parsed_separately_from_diagnostics(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(type_a=q4_K,type_b=f32,m=16,n=1,k=256): "
                "AK_REF_V1 metric=test_backend_ops_error/v1 observed=2.5e-09 "
                "tolerance=1e-07 comparisons=3 oracle=ggml_cpu_reference/v1 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        run = t0.parse_backend_ops_console(text)
        case = run.cases[0]
        self.assertEqual(case.interleaved, "")
        self.assertEqual(case.reference.metric_id, "test_backend_ops_error/v1")
        self.assertEqual(case.reference.observed, 2.5e-09)
        self.assertEqual(case.reference.tolerance, 1e-07)
        self.assertEqual(case.reference.comparisons, 3)
        self.assertEqual(case.reference.oracle_id, "ggml_cpu_reference/v1")
        run.reconcile()

    def test_malformed_reference_receipt_refuses(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(type_a=q4_K): AK_REF_V1 observed=0 tolerance=1e-7 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_console(text)

    def test_property_residual_is_parsed_as_candidate_only_measurement(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SOFT_MAX(type=f32,ne=[83,2,1,1]): "
                "AK_PROP_V1 metric=softmax_invariants/v1 residual=2.5e-08 "
                "tolerance=0.0001 passed=1 suite_seed=4711 | "
                "AK_REF_V1 metric=test_backend_ops_error/v1 observed=1e-09 "
                "tolerance=1e-07 comparisons=1 oracle=ggml_cpu_reference/v1 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        case = t0.parse_backend_ops_console(text).cases[0]
        self.assertEqual(len(case.properties), 1)
        measurement = case.properties[0]
        self.assertEqual(measurement.metric_id, "softmax_invariants/v1")
        self.assertEqual(measurement.residual, 2.5e-08)
        self.assertEqual(measurement.tolerance, 0.0001)
        self.assertTrue(measurement.passed)
        self.assertEqual(measurement.suite_seed, 4711)

    def test_property_receipt_cannot_lie_about_its_derived_verdict(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SOFT_MAX(type=f32): AK_PROP_V1 metric=softmax_invariants/v1 "
                "residual=0.25 tolerance=0.0001 passed=1 suite_seed=4711 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_console(text)

    def test_layout_receipt_preserves_all_three_explicit_families(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(type_a=f32): AK_LAYOUT_V1 "
                "families=offset,transpose,stride_gap suite_seed=4711 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        run = t0.parse_backend_ops_console(text)
        self.assertEqual(
            run.cases[0].layout.families,
            ("offset", "transpose", "stride_gap"))
        self.assertEqual(run.cases[0].layout.suite_seed, 4711)
        self.assertEqual(run.layout_families(),
                         ("offset", "stride_gap", "transpose"))

    def test_unsupported_layout_is_a_failure_not_a_declined_case(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(type_a=f32): AK_LAYOUT_V1 families=stride_gap "
                "suite_seed=4711 | non-contiguous layout is not supported FAIL\n"
                "  0/1 tests passed\n  Backend CPU: FAIL\n"
                "0/1 backends passed\nFAIL\n")
        run = t0.parse_backend_ops_console(text)
        self.assertEqual(run.cases[0].status, "fail")
        self.assertEqual(run.failed_ops(), ("MUL_MAT",))
        self.assertEqual(run.unsupported_by_op(), ())
        run.reconcile()

    def test_value_transform_receipt_and_v2_property_transform_are_structured(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SOFT_MAX(type=f32): AK_VALUE_V1 "
                "transforms=identity,x3,x0p01,negate completed=4 suite_seed=4711 | "
                "AK_PROP_V2 metric=softmax_invariants/v1 residual=2e-08 "
                "tolerance=0.0001 passed=1 suite_seed=4711 transform=x3 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        case = t0.parse_backend_ops_console(text).cases[0]
        self.assertEqual(case.value_transforms.transforms,
                         ("identity", "x3", "x0p01", "negate"))
        self.assertEqual(case.value_transforms.completed, 4)
        self.assertEqual(case.properties[0].transform, "x3")

    def test_passing_case_cannot_claim_an_incomplete_transform_pass(self):
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(type=f32): AK_VALUE_V1 "
                "transforms=identity,x3,x0p01,negate completed=2 suite_seed=4711 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_console(text)

    def test_recorded_passing_run_parses_cases_and_ops(self):
        run = t0.parse_backend_ops_console(recorded("recorded_t0_backend_ops_console_ok.txt"))
        self.assertEqual(len(run.backends), 1)
        self.assertEqual(run.backends[0].name, "CPU")
        self.assertFalse(run.backends[0].skipped)
        self.assertEqual(len(run.cases), 2)
        self.assertEqual(run.exercised_ops(), ("ARANGE",))
        self.assertEqual(run.cases_by_op(), (("ARANGE", 2, 2),))
        self.assertEqual(run.failed_ops(), ())
        self.assertEqual((run.backends_passed, run.backends_total), (1, 1))
        self.assertEqual(run.overall, "OK")
        run.reconcile()

    def test_skipped_backend_yields_zero_coverage_despite_ok_and_exit_zero(self):
        """THE GUARD. Recorded: `-b ROCm0` on a CPU-only build.

        The tool prints `Skipping`, `1/1 backends passed`, `OK` and exits 0. The
        skip path increments `n_ok` (test-backend-ops.cpp:10366-10370), so a
        mistyped backend name buys a clean pass over an op that never ran.

        Without the guard `exercised_ops()` would read the summary line, or the
        exit status, and report coverage. It reads CASES, so it reports none.
        """
        run = t0.parse_backend_ops_console(recorded("recorded_t0_backend_ops_console_skip.txt"))
        self.assertEqual(run.skipped_backends, ("CPU",))
        self.assertEqual(run.backends[0].skip_reason, "Skipping")
        self.assertEqual(run.cases, ())
        self.assertEqual(run.exercised_ops(), ())
        self.assertEqual(run.cases_by_op(), ())
        # The clean-looking summary really is in the capture — the point is that
        # the parser does not consult it for coverage.
        self.assertEqual((run.backends_passed, run.backends_total), (1, 1))
        self.assertEqual(run.overall, "OK")

    def test_cpu_backend_is_skipped_without_an_explicit_filter(self):
        """Recorded: `test -o ARANGE` with no `-b`. Test mode skips CPU outright."""
        run = t0.parse_backend_ops_console(
            recorded("recorded_t0_backend_ops_console_cpu_skip.txt"))
        self.assertEqual(run.backends[0].skip_reason, "Skipping CPU backend")
        self.assertEqual(run.exercised_ops(), ())

    def test_requested_op_that_never_ran_is_not_reported_as_exercised(self):
        """THE HEADLINE. Recorded `-o MUL_MAT,MUL_MAT_ID`: only MUL_MAT ran.

        286 case lines, every one of them `MUL_MAT`. `MUL_MAT_ID` was named on
        the command line and produced no case at all, because the `-p` shape
        filter matched none of its parameter spellings — and the run still
        printed `178/178 tests passed`, `Backend CPU: OK`, `1/1 backends
        passed`, `OK`, exit 0.

        That is exactly `kernel_eval.sh`: *"it reported MUL_MAT 4231/4231 OK and
        that sentence was true — it was just not a statement about MUL_MAT_ID,
        which it never ran."*
        """
        run = t0.parse_backend_ops_console(
            recorded("recorded_t0_backend_ops_mandatory_ops.txt"))
        self.assertEqual(run.exercised_ops(), ("MUL_MAT",))
        self.assertNotIn("MUL_MAT_ID", run.exercised_ops())
        self.assertEqual(run.overall, "OK")

    def test_unsupported_shapes_are_excluded_from_the_comparison_denominator(self):
        """108 of 286 recorded `MUL_MAT` shapes are `not supported [CPU]`.

        They are not passes and they are not failures: the backend declined
        them, so nothing was compared. Counting them as failures would FAIL
        every honest MUL_MAT run on this backend, and a gate that fails on
        correct input gets switched off. `exercised_ops()` is where the coverage
        question is answered instead.
        """
        run = t0.parse_backend_ops_console(
            recorded("recorded_t0_backend_ops_mandatory_ops.txt"))
        by_op = dict((op, (total, passed)) for op, total, passed in run.cases_by_op())
        self.assertEqual(by_op["MUL_MAT"], (178, 178))
        self.assertEqual(len(run.cases), 286)
        self.assertEqual(dict(run.unsupported_by_op())["MUL_MAT"], 108)
        self.assertEqual(run.failed_ops(), ())

    def test_sanitizer_diagnostics_interleaved_into_case_lines_are_retained(self):
        """The recorded capture came from a UBSAN-instrumented build.

        Four real `runtime error: load of misaligned address … for type 'const
        uint32_t'` reports from `ggml/src/ggml-cpu/arch/x86/quants.c` arrive
        BETWEEN the two halves of a case line, and the case's `OK` lands several
        lines later. The parser holds the case open, attributes the verdict, and
        keeps the diagnostic — a parser that dropped the tail would have deleted
        four genuine UBSAN findings in the production-lineage quant kernels
        while reporting the run clean.
        """
        run = t0.parse_backend_ops_console(
            recorded("recorded_t0_backend_ops_mandatory_ops.txt"))
        diagnostics = run.interleaved_diagnostics()
        self.assertTrue(diagnostics)
        joined = " ".join(diagnostics)
        self.assertIn("runtime error", joined)
        self.assertIn("quants.c", joined)
        # The cases they belong to still carry a verdict, and it is a pass.
        carriers = [c for c in run.cases if c.interleaved]
        self.assertTrue(carriers)
        self.assertTrue(all(c.status == "ok" for c in carriers))

    def test_a_case_left_open_refuses(self):
        """Truncated mid-diagnostic: an unknown result is not a pass."""
        text = recorded("recorded_t0_backend_ops_mandatory_ops.txt")
        head = text.split("0x7cb4a064a3d6: note")[0]
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_console(head)

    def test_an_op_the_backend_declines_entirely_is_not_exercised(self):
        """THE GUARD, isolated. Built from the recorded `not supported` lines.

        `exercised_ops()` asks for a SUPPORTED case, not merely a case. Drop
        that word and an op the backend declined on every single shape reads as
        covered, and `check_backend_op_units` ticks off a required op that ran
        nothing — the same clean-looking nothing the skipped-backend capture
        produces, arriving through a different door.
        """
        declined = [line for line in
                    recorded("recorded_t0_backend_ops_mandatory_ops.txt").splitlines()
                    if "not supported" in line][:5]
        self.assertEqual(len(declined), 5, "recorded capture should carry declined shapes")
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  Device description: AMD EPYC 9655 96-Core Processor\n\n"
                + "\n".join(declined)
                + "\n  0/0 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n")
        run = t0.parse_backend_ops_console(text)
        self.assertEqual(len(run.cases), 5)
        self.assertTrue(all(c.status == "not_supported" for c in run.cases))
        self.assertEqual(run.exercised_ops(), ())
        self.assertEqual(run.cases_by_op(), ())
        run.reconcile()

    def test_reconcile_agrees_with_the_tools_own_summary(self):
        """The parser's compared-case count must equal `178/178 tests passed`."""
        run = t0.parse_backend_ops_console(
            recorded("recorded_t0_backend_ops_mandatory_ops.txt"))
        self.assertEqual(run.backends[0].reported_total, 178)
        run.reconcile()

    def test_reconcile_refuses_when_parser_and_tool_disagree(self):
        """The bite: drop one OK line and the tool's summary no longer matches."""
        text = recorded("recorded_t0_backend_ops_console_ok.txt")
        mangled = text.replace(
            "  ARANGE(type=f32,start=0.000000,stop=10.000000,step=1.000000): \x1b[1;32mOK\x1b[0m\n",
            "", 1)
        run = t0.parse_backend_ops_console(mangled)
        self.assertEqual(len(run.cases), 1)
        with self.assertRaises(t0.OutputParseError):
            run.reconcile()

    def test_unknown_verdict_refuses_rather_than_reading_clean(self):
        text = recorded("recorded_t0_backend_ops_console_ok.txt").replace("OK\x1b[0m", "WOBBLE")
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_console(text)

    def test_output_with_no_frame_refuses(self):
        """'no log' is not 'no findings'."""
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_console("cmake: command not found\n")

    def test_failing_case_is_reported_as_a_failed_op(self):
        text = recorded("recorded_t0_backend_ops_console_ok.txt").replace(
            "\x1b[1;32mOK\x1b[0m", "\x1b[1;31mFAIL\x1b[0m", 1).replace(
            "  2/2 tests passed", "  1/2 tests passed")
        run = t0.parse_backend_ops_console(text)
        self.assertEqual(run.failed_ops(), ("ARANGE",))
        self.assertEqual(run.cases_by_op(), (("ARANGE", 2, 1),))
        run.reconcile()


class BackendOpsCsvParsing(unittest.TestCase):

    def test_recorded_csv_parses(self):
        run = t0.parse_backend_ops_csv(recorded("recorded_t0_backend_ops_csv.txt"))
        self.assertEqual(run.exercised_ops(), ("ARANGE",))
        self.assertEqual(run.cases_by_op(), (("ARANGE", 2, 2),))

    def test_current_csv_keeps_reference_receipt_out_of_error_message(self):
        receipt = ("AK_REF_V1 metric=test_backend_ops_error/v1 observed=0 "
                   "tolerance=1e-07 comparisons=2 oracle=ggml_cpu_reference/v1")
        text = (
            '"backend_name","op_name","op_params","test_mode","supported",'
            '"error_message","backend_reg_name","reference_receipt"\n'
            f'"CPU","MUL_MAT","type_a=q4_K","test","1","","CPU","{receipt}"\n')
        run = t0.parse_backend_ops_csv(text)
        self.assertEqual(run.cases[0].status, "ok")
        self.assertEqual(run.cases[0].interleaved, "")
        self.assertEqual(run.cases[0].reference.oracle_id, "ggml_cpu_reference/v1")

    def test_current_csv_keeps_property_receipt_structured(self):
        property_receipt = (
            "AK_PROP_V1 metric=argsort_permutation_violations/v1 residual=0 "
            "tolerance=0 passed=1 suite_seed=99")
        text = (
            '"backend_name","op_name","op_params","test_mode","supported",'
            '"error_message","backend_reg_name","property_receipt","reference_receipt"\n'
            f'"CPU","ARGSORT","type=i32","test","1","","CPU",'
            f'"{property_receipt}",""\n')
        case = t0.parse_backend_ops_csv(text).cases[0]
        self.assertEqual(case.properties[0].suite_seed, 99)
        self.assertEqual(case.properties[0].residual, 0.0)

    def test_csv_hard_layout_failure_is_not_downgraded_to_not_supported(self):
        text = (
            '"backend_name","op_name","op_params","test_mode","supported",'
            '"hard_failure","error_message","layout_receipt"\n'
            '"CPU","MUL_MAT","v=1","test","0","1",'
            '"non-contiguous layout is not supported",'
            '"AK_LAYOUT_V1 families=stride_gap suite_seed=4711"\n')
        case = t0.parse_backend_ops_csv(text).cases[0]
        self.assertEqual(case.status, "fail")
        self.assertEqual(case.layout.families, ("stride_gap",))

    def test_csv_preserves_value_transform_receipt(self):
        text = (
            '"backend_name","op_name","op_params","test_mode","supported",'
            '"error_message","value_receipt"\n'
            '"CPU","MUL_MAT","type=f32","test","1","",'
            '"AK_VALUE_V1 transforms=identity,x3,x0p01,negate completed=4 '
            'suite_seed=4711"\n')
        case = t0.parse_backend_ops_csv(text).cases[0]
        self.assertEqual(case.value_transforms.completed, 4)

    def test_csv_preserves_stateful_receipt(self):
        text = (
            '"backend_name","op_name","op_params","test_mode","supported",'
            '"error_message","state_receipt"\n'
            '"CPU","FLASH_ATTN_EXT","type=f32","test","1","",'
            '"AK_STATE_V1 inputs=2 initial_equal=1 input_immutable=1 '
            'final_outputs=2 suite_seed=4711"\n')
        case = t0.parse_backend_ops_csv(text).cases[0]
        self.assertEqual(case.stateful.input_count, 2)
        self.assertEqual(case.stateful.final_output_count, 2)

    def test_csv_cannot_express_a_skipped_backend(self):
        """Why console is the T0 default, stated as a test rather than a comment.

        `csv_printer` emits a header and case rows and nothing else — no backend
        line, no skip reason. A CSV run whose `-b` filter matched no device is a
        lone header, byte-indistinguishable from a run that simply selected no
        case. The recorded console capture of the same failure carries
        `Skipping`; the CSV shape could not.
        """
        header_only = recorded("recorded_t0_backend_ops_csv.txt").splitlines()[3]
        run = t0.parse_backend_ops_csv(header_only + "\n")
        self.assertEqual(run.cases, ())
        self.assertEqual(run.skipped_backends, ())   # nothing to report: the format has none

    def test_header_change_refuses(self):
        text = recorded("recorded_t0_backend_ops_csv.txt").replace("op_name", "operation")
        with self.assertRaises(t0.OutputParseError):
            t0.parse_backend_ops_csv(text)


# =============================================================================
# B. Linkage, against both recorded reports
# =============================================================================

class LinkageParsing(unittest.TestCase):

    def test_recorded_pass_report(self):
        report = t0.parse_linkage_report(recorded("recorded_t0_linkage_pass.txt"))
        self.assertEqual(report.verdict, schemas.PASS)
        self.assertEqual(len(report.rows), 5)
        self.assertEqual(report.stray, ())
        self.assertEqual(report.loader_path,
                         ("/mnt/raid0/llm/llama.cpp-experimental/build-v8-sanitize/bin",))

    def test_recorded_fail_report_is_the_production_tree_leak(self):
        """INC-20260731, live on this host today.

        The SAME experimental binary, run under the container's ambient
        `LD_LIBRARY_PATH`, resolves all five of its ggml/llama libraries out of
        `/mnt/raid0/llm/llama.cpp/build/bin` — the frozen production tree. The
        run completes, the output is well-formed, and only the thing being
        measured is wrong.
        """
        report = t0.parse_linkage_report(recorded("recorded_t0_linkage_fail.txt"))
        self.assertEqual(report.verdict, schemas.FAIL)
        self.assertEqual(len(report.stray), 5)
        for row in report.stray:
            self.assertTrue(t0.under_production_tree(row.path), row.path)
        self.assertIn("/mnt/raid0/llm/llama.cpp/build/bin", report.loader_path)

    def test_truncated_report_refuses(self):
        text = recorded("recorded_t0_linkage_pass.txt").split("PASS:")[0]
        with self.assertRaises(t0.OutputParseError):
            t0.parse_linkage_report(text)

    def test_linkage_digest_distinguishes_resolved_tables(self):
        good = t0.parse_linkage_report(recorded("recorded_t0_linkage_pass.txt"))
        bad = t0.parse_linkage_report(recorded("recorded_t0_linkage_fail.txt"))
        self.assertNotEqual(t0.ExecutedT0EvidenceProvider.linkage_digest(good),
                            t0.ExecutedT0EvidenceProvider.linkage_digest(bad))

    def test_linkage_digest_names_the_ggml_generation_not_tool_specific_libs(self):
        # The live linker table is deliberately asymmetric: llama-cli has a
        # direct libggml.so edge while inspecting libggml.so does not.  Both
        # nevertheless name exactly the same complete ggml closure.
        cli = t0.LinkageReport("/anchor/llama-cli", "/anchor", (
            t0.LinkageRow("libggml.so.0", "/anchor/libggml.so.0", True),
            t0.LinkageRow("libggml-base.so.0", "/anchor/libggml-base.so.0", True),
            t0.LinkageRow("libggml-cpu.so.0", "/anchor/libggml-cpu.so.0", True),
            t0.LinkageRow("libllama.so.0", "/anchor/libllama.so.0", True),
        ), schemas.PASS, ())
        direct_child = t0.LinkageReport("/anchor/libggml.so.0", "/anchor", (
            t0.LinkageRow("libggml-base.so.0", "/anchor/libggml-base.so.0", True),
            t0.LinkageRow("libggml-cpu.so.0", "/anchor/libggml-cpu.so.0", True),
        ), schemas.PASS, ())
        self.assertEqual(t0.ExecutedT0EvidenceProvider.linkage_digest(cli),
                         t0.ExecutedT0EvidenceProvider.linkage_digest(direct_child))

    def test_linkage_digest_direct_libggml_still_detects_a_changed_child(self):
        base = t0.LinkageRow("libggml-base.so.0", "/anchor/libggml-base.so.0", True)
        direct = t0.LinkageReport("/anchor/libggml.so.0", "/anchor", (
            base, t0.LinkageRow("libggml-cpu.so.0", "/anchor/libggml-cpu.so.0", True),
        ), schemas.PASS, ())
        changed = t0.LinkageReport("/anchor/libggml.so.0", "/anchor", (
            base, t0.LinkageRow("libggml-cpu.so.0", "/other/libggml-cpu.so.0", False),
        ), schemas.PASS, ())
        self.assertNotEqual(t0.ExecutedT0EvidenceProvider.linkage_digest(direct),
                            t0.ExecutedT0EvidenceProvider.linkage_digest(changed))

    def test_linkage_digest_refuses_missing_ggml_generation(self):
        report = t0.LinkageReport("bad", "/anchor", (
            t0.LinkageRow("libggml.so.0", "/anchor/libggml.so.0", True),
        ), schemas.PASS, ())
        with self.assertRaisesRegex(t0.OutputParseError, "exactly one libggml-base"):
            t0.ExecutedT0EvidenceProvider.linkage_digest(report)


# =============================================================================
# C. Sanitizer, diagnostic, trace and token parsing
# =============================================================================

#: NOT recorded output. Assembled from ASAN/UBSAN's documented line shapes
#: because producing a real sanitizer log needs a sanitizer BUILD, which is
#: tomorrow's work under a claim. Labelled so nobody mistakes it for a capture.
SYNTHETIC_SANITIZER_LOG = """\
=================================================================
==31337==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6110000001f4
    #0 0x55f in ggml_compute_forward_mul_mat ggml/src/ggml-cpu/ggml-cpu.c:7412
ggml/src/ggml-cpu/ops.cpp:221:17: runtime error: signed integer overflow: 2147483647 + 1
ggml/src/ggml-cpu/vec.h:88:9: runtime error: load of misaligned address 0x7ffd
=================================================================
==31337==ERROR: LeakSanitizer: detected memory leaks
"""


class SanitizerAndDiagnosticParsing(unittest.TestCase):

    def test_asan_and_ubsan_findings_land_in_separate_surfaces(self):
        asan, ubsan = t0.parse_sanitizer_findings(SYNTHETIC_SANITIZER_LOG)
        self.assertEqual(len(asan), 2)                       # heap overflow + leak
        self.assertTrue(any("heap-buffer-overflow" in f for f in asan))
        self.assertTrue(any("LeakSanitizer" in f for f in asan))
        self.assertEqual(len(ubsan), 2)
        self.assertTrue(all("runtime error" in f for f in ubsan))

    def test_clean_log_yields_no_findings(self):
        self.assertEqual(t0.parse_sanitizer_findings("all tests passed\n"), ((), ()))

    def test_compiler_diagnostics_are_deduplicated(self):
        log = ("ggml/src/x.c:10:5: warning: unused variable 'a'\n"
               "ggml/src/x.c:10:5: warning: unused variable 'a'\n"
               "ggml/src/y.c:3:1: error: expected ';'\n")
        errors, warnings, findings = t0.parse_compiler_diagnostics(log)
        self.assertEqual((errors, warnings), (1, 1))
        self.assertEqual(len(findings), 1)

    def test_sched_trace_marker_is_what_proves_instrumentation(self):
        trace = ("## SPLIT #0: CPU # 0 inputs\n"
                 "node #  0 (   MUL_MAT):            ffn_up-0 (  1MB) [  CPU 1.dst  ] use=1,c=1:\n"
                 "node #  1 (MUL_MAT_ID):           ffn_moe-0 (  2MB) [ ROCm0 1.src ] use=1,c=1:\n")
        emitted, splits, nodes = t0.parse_sched_trace(trace)
        self.assertTrue(emitted)
        self.assertEqual(splits, ((0, "CPU", 0),))
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[1][1], "MUL_MAT_ID")
        self.assertEqual(nodes[1][3], "ROCm0")

    def test_sched_trace_without_the_marker_reports_no_instrumentation(self):
        emitted, splits, nodes = t0.parse_sched_trace("llama_perf_context_print: load time\n")
        self.assertFalse(emitted)
        self.assertEqual((splits, nodes), ((), ()))

    def test_delivered_tokens_absent_is_none_not_zero(self):
        self.assertIsNone(t0.parse_delivered_tokens("no perf print here"))
        self.assertEqual(t0.parse_delivered_tokens(
            "llama_perf_context_print:        eval time =    1234.56 ms /   128 runs   "
            "(    9.65 ms per token,   103.68 tokens per second)\n"), 128)


# =============================================================================
# D. The anchor triple — all three, or none, and always the MEASURED one
# =============================================================================

class AnchorTriple(unittest.TestCase):

    def test_full_triple_constructs(self):
        cap = anchor_capture()
        self.assertEqual(t0._anchor_triple(cap), (COMMIT_ANCHOR, SHA_B, SHA_C))
        self.assertIsInstance(cap.identity(), api.AnchorIdentity)

    def test_no_capture_yields_three_nones(self):
        self.assertEqual(t0._anchor_triple(None), (None, None, None))

    def test_each_missing_component_refuses(self):
        """A partially named anchor is unconstructible, not merely discouraged."""
        for field_name in ("source_commit", "binary_sha256", "linkage_sha256"):
            with self.subTest(field=field_name):
                with self.assertRaises(t0.AnchorCaptureIncomplete):
                    anchor_capture(**{field_name: None})

    def test_placeholder_digest_refuses(self):
        with self.assertRaises(t0.AnchorCaptureIncomplete):
            anchor_capture(binary_sha256=PLACEHOLDER)
        with self.assertRaises(t0.AnchorCaptureIncomplete):
            anchor_capture(linkage_sha256=PLACEHOLDER)

    def test_short_commit_refuses(self):
        with self.assertRaises(t0.AnchorCaptureIncomplete):
            anchor_capture(source_commit=COMMIT_ANCHOR[:12])


# =============================================================================
# E. Claim enforcement — denial 8
# =============================================================================

class ClaimEnforcement(unittest.TestCase):

    def test_require_claim_refuses_none(self):
        with self.assertRaises(t0.ClaimNotHeld):
            t0.require_claim(None, what="a generation")

    def test_require_claim_refuses_a_released_claim(self):
        with self.assertRaises(t0.ClaimNotHeld):
            t0.require_claim(FakeClaim(held=False), what="a generation")

    def test_require_claim_refuses_an_object_that_is_not_a_claim(self):
        with self.assertRaises(TypeError):
            t0.require_claim(object(), what="a generation")

    def test_require_claim_returns_the_id_on_the_compliant_path(self):
        self.assertEqual(t0.require_claim(FakeClaim(), what="a generation"),
                         "akclaim-cpu-0-95")

    def test_op_suite_refuses_to_run_unclaimed(self):
        provider = t0.ExecutedT0EvidenceProvider(
            plan=execution_plan(), runner=t0.RecordedProcessRunner([]), claim=None)
        with self.assertRaises(t0.ClaimNotHeld):
            provider.collect_op_suite(t0._Collected())

    def test_coherence_refuses_to_run_unclaimed(self):
        plan = execution_plan(generation=generation_plan())
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), claim=None)
        with self.assertRaises(t0.ClaimNotHeld):
            provider.collect_coherence(t0._Collected())

    def test_determinism_refuses_to_run_unclaimed(self):
        plan = execution_plan(generation=generation_plan(), determinism_runs=3)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), claim=None)
        with self.assertRaises(t0.ClaimNotHeld):
            provider.collect_determinism(t0._Collected())


# =============================================================================
# F. Production-tree refusals
# =============================================================================

class ProductionTreeRefusals(unittest.TestCase):

    def test_candidate_paths_inside_production_are_refused(self):
        for field_name, value in (
                ("worktree", PROD_TREE),
                ("build_dir", f"{PROD_TREE}/build-t0"),
                ("binary", f"{PROD_TREE}/build/bin/llama-cli"),
                ("test_backend_ops", f"{PROD_TREE}/build/bin/test-backend-ops")):
            with self.subTest(field=field_name):
                overrides = {field_name: value}
                if field_name == "binary":
                    overrides["library_path"] = f"{PROD_TREE}/build/bin"
                with self.assertRaises(t0.ProductionTreeRefusal):
                    candidate_build(**overrides)

    def test_experimental_paths_are_accepted(self):
        self.assertIsInstance(candidate_build(), t0.CandidateBuild)

    def test_anchor_inside_production_is_allowed_because_it_is_read_only(self):
        """The anchor IS the frozen production binary; executing it is not a write."""
        anchor = t0.AnchorBuild(worktree=PROD_TREE, source_commit=COMMIT_ANCHOR,
                                binary=f"{PROD_TREE}/build/bin/llama-cli",
                                library_path=f"{PROD_TREE}/build/bin")
        self.assertEqual(anchor.worktree, PROD_TREE)

    def test_sanitizer_build_dir_inside_production_is_refused(self):
        with self.assertRaises(t0.ProductionTreeRefusal):
            execution_plan(sanitizer_build_dir=f"{PROD_TREE}/build-asan",
                           sanitizer_target="test-backend-ops")

    def test_library_path_must_be_the_binarys_own_directory(self):
        with self.assertRaises(ValueError):
            candidate_build(library_path="/mnt/raid0/llm/llama.cpp/build/bin")


# =============================================================================
# G. Argv construction and the launch environment
# =============================================================================

class ArgvConstruction(unittest.TestCase):

    def test_stateful_probe_has_a_separate_explicit_flag(self):
        inv = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("SSM_SCAN",), base_env=(),
            suite_seed=4711, stateful_probe=True)
        self.assertIn("--autokernel-stateful", inv.argv)
        self.assertNotIn("--autokernel-layouts", inv.argv)
        self.assertNotIn("--autokernel-value-transforms", inv.argv)

    def test_stateful_flag_cannot_merge_with_another_axis(self):
        with self.assertRaises(ValueError):
            t0.build_backend_ops_invocation(
                binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
                backend_filter="CPU", ops=("SSM_SCAN",), base_env=(),
                suite_seed=4711, stateful_probe=True, layout_probe=True)

    def test_value_transforms_have_a_separate_explicit_flag(self):
        inv = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=(),
            suite_seed=4711, value_transform_probe=True)
        self.assertIn("--autokernel-value-transforms", inv.argv)
        self.assertNotIn("--autokernel-layouts", inv.argv)

    def test_layout_and_value_flags_cannot_merge(self):
        with self.assertRaises(ValueError):
            t0.build_backend_ops_invocation(
                binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
                backend_filter="CPU", ops=("MUL_MAT",), base_env=(),
                suite_seed=4711, layout_probe=True, value_transform_probe=True)

    def test_layout_probe_has_its_own_explicit_flag_and_receipt(self):
        inv = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=(),
            suite_seed=4711, layout_probe=True)
        self.assertIn("--autokernel-layouts", inv.argv)
        self.assertIn("--autokernel-layouts", " ".join(inv.notes))
        self.assertEqual(inv.constructor_id, "ak.t0.backend_ops_test/v3")

    def test_backend_ops_argv_uses_the_ratified_canonical_prefix(self):
        inv = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT", "MUL_MAT_ID"), base_env=())
        self.assertEqual(list(inv.argv[:len(recipes.CANONICAL_PREFIX)]),
                         list(recipes.CANONICAL_PREFIX))
        self.assertIn("test", inv.argv)
        self.assertNotIn("perf", inv.argv)
        self.assertEqual(inv.argv[inv.argv.index("-b") + 1], "CPU")
        self.assertEqual(inv.argv[inv.argv.index("-o") + 1], "MUL_MAT,MUL_MAT_ID")
        self.assertEqual(inv.argv[inv.argv.index("--suite-seed") + 1], "0")
        self.assertIn("--autokernel-properties", inv.argv)

    def test_backend_ops_seed_is_explicit_and_receipted(self):
        zero = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=(), suite_seed=0)
        seeded = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=(), suite_seed=4711)
        self.assertEqual(seeded.argv[seeded.argv.index("--suite-seed") + 1], "4711")
        self.assertNotEqual(zero.receipt.argv_sha256, seeded.receipt.argv_sha256)

    def test_backend_ops_env_carries_the_full_canonical_omp_stack(self):
        inv = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=())
        env = inv.env_dict()
        for key, value in recipes.CANONICAL_OMP_ENV.items():
            self.assertEqual(env[key], value, key)

    def test_registered_parameter_env_is_the_only_canonical_override(self):
        inv = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=(),
            parameter_env=(("GGML_IQK", "0"),))
        env = inv.env_dict()
        self.assertEqual(env["GGML_IQK"], "0")
        for key, value in recipes.CANONICAL_OMP_ENV.items():
            if key != "GGML_IQK":
                self.assertEqual(env[key], value, key)

    def test_unregistered_parameter_env_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a registered arm-local variant"):
            t0.build_backend_ops_invocation(
                binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
                backend_filter="CPU", ops=("MUL_MAT",), base_env=(),
                parameter_env=(("UNSAFE", "1"),))

    def test_launch_env_overrides_an_ambient_production_library_path(self):
        """THE GUARD, with its compliant control.

        `base_env` here carries exactly what this container's ambient
        environment carries — the production build dir on `LD_LIBRARY_PATH`.
        The launcher must overwrite it with the binary's own directory, not
        append to it, or the recorded linkage FAILURE is what happens at run
        time.
        """
        poisoned = (("LD_LIBRARY_PATH",
                     "/opt/AMD/aocc-compiler-5.0.0/lib:/mnt/raid0/llm/llama.cpp/build/bin"),)
        inv = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=poisoned)
        self.assertEqual(inv.env_dict()["LD_LIBRARY_PATH"], BIN_DIR)
        self.assertNotIn("/mnt/raid0/llm/llama.cpp/build/bin",
                         inv.env_dict()["LD_LIBRARY_PATH"])

    def test_receipt_binds_argv_and_env(self):
        one = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT",), base_env=())
        two = t0.build_backend_ops_invocation(
            binary=f"{BIN_DIR}/test-backend-ops", library_path=BIN_DIR,
            backend_filter="CPU", ops=("MUL_MAT", "MUL_MAT_ID"), base_env=())
        self.assertNotEqual(one.receipt.argv_sha256, two.receipt.argv_sha256)
        self.assertEqual(one.receipt.constructor_sha256, two.receipt.constructor_sha256)
        self.assertIsInstance(one.receipt, api.RecipeReceipt)

    def test_generation_argv_states_its_sampling_explicitly(self):
        plan = generation_plan()
        inv = t0.build_generation_invocation(
            binary=f"{BIN_DIR}/llama-cli", library_path=BIN_DIR, plan=plan, base_env=())
        for flag in ("--temp", "--top-k", "--seed", "--no-warmup", "-no-cnv"):
            self.assertIn(flag, inv.argv, flag)
        self.assertEqual(inv.argv[inv.argv.index("--seed") + 1], "42")

    def test_greedy_is_derived_from_the_sampling_parameters(self):
        self.assertTrue(generation_plan().is_greedy())
        self.assertFalse(generation_plan(temperature=0.7).is_greedy())
        self.assertFalse(generation_plan(top_k=40).is_greedy())

    def test_linkage_invocation_runs_under_the_measurement_environment(self):
        inv = t0.build_linkage_invocation(
            bash="/bin/bash", script="/x/verify_ggml_linkage.sh",
            binary=f"{BIN_DIR}/llama-cli", expected_root=BIN_DIR, library_path=BIN_DIR,
            base_env=(("LD_LIBRARY_PATH", "/mnt/raid0/llm/llama.cpp/build/bin"),))
        self.assertEqual(inv.env_dict()["LD_LIBRARY_PATH"], BIN_DIR)
        self.assertEqual(inv.argv, ("/bin/bash", "/x/verify_ggml_linkage.sh",
                                    f"{BIN_DIR}/llama-cli", BIN_DIR))


class PlanRefusals(unittest.TestCase):

    def test_op_suite_without_ops_refuses(self):
        with self.assertRaises(ValueError):
            op_suite_plan(ops=())

    def test_op_suite_without_a_backend_filter_refuses(self):
        with self.assertRaises(ValueError):
            op_suite_plan(backend_filter="")

    def test_cache_state_must_be_in_the_vocabulary(self):
        with self.assertRaises(ValueError):
            execution_plan(cache_state="probably_cold")

    def test_unknown_cache_state_is_sayable(self):
        self.assertEqual(execution_plan(cache_state="unknown").cache_state, "unknown")

    def test_determinism_runs_without_a_generation_refuses(self):
        with self.assertRaises(ValueError):
            execution_plan(determinism_runs=3)

    def test_holdout_visibility_must_be_stated(self):
        with self.assertRaises(TypeError):
            t0.HoldoutPlan(unseen_case_filter="a", boundary_case_filter="b",
                           selection_rule_id="r", selection_seed="s",
                           visible_to_planner="no")


# =============================================================================
# H. The process runner — real processes, killed by captured pid
# =============================================================================

class SubprocessRunnerDiscipline(unittest.TestCase):

    def test_environment_is_declared_and_nothing_is_inherited(self):
        """The bite: this container's ambient env carries LD_LIBRARY_PATH.

        If `SubprocessRunner` inherited the parent environment, that variable
        would appear in the child's `env` output — and an experimental binary
        would resolve production ggml (recorded fixture
        `recorded_t0_linkage_fail.txt`). It is passed, never inherited, so the
        child sees exactly two variables.
        """
        self.assertIn("LD_LIBRARY_PATH", os.environ,
                      "precondition for this test: the ambient env sets LD_LIBRARY_PATH")
        runner = t0.SubprocessRunner()
        result = runner.run(["/usr/bin/env"], env={"FOO": "bar", "PATH": "/usr/bin:/bin"},
                            cwd="/tmp", timeout_s=30)
        lines = {line.split("=", 1)[0] for line in result.stdout.splitlines() if "=" in line}
        self.assertEqual(result.exit_code, 0)
        self.assertIn("FOO", lines)
        self.assertNotIn("LD_LIBRARY_PATH", lines)

    def test_timeout_kills_the_child_it_launched_and_verifies_death(self):
        runner = t0.SubprocessRunner(term_grace_s=2.0, kill_grace_s=2.0)
        result = runner.run(["/bin/sleep", "30"], env={"PATH": "/usr/bin:/bin"},
                            cwd="/tmp", timeout_s=1.0)
        self.assertTrue(result.timed_out)
        self.assertTrue(result.signalled)
        self.assertEqual(result.orphans, (),
                         "the child survived SIGTERM and SIGKILL and must be reported")

    def test_capture_ref_is_content_addressed(self):
        one = capture(["/bin/true"], stdout="a")
        two = capture(["/bin/true"], stdout="b")
        self.assertNotEqual(t0.capture_ref(one), t0.capture_ref(two))
        self.assertTrue(t0.capture_ref(one).startswith("akcap:"))


class RecordedRunnerRefusal(unittest.TestCase):

    def test_unknown_argv_refuses_rather_than_returning_empty_output(self):
        runner = t0.RecordedProcessRunner([capture(["/bin/true"], stdout="x")])
        with self.assertRaises(t0.CaptureUnavailable):
            runner.run(["/bin/false"], env={}, cwd="/tmp", timeout_s=1)

    def test_known_argv_replays(self):
        runner = t0.RecordedProcessRunner([capture(["/bin/true"], stdout="x")])
        self.assertEqual(runner.run(["/bin/true"], env={}, cwd="/tmp", timeout_s=1).stdout, "x")
        self.assertEqual(len(runner.calls), 1)


# =============================================================================
# I. Self-audit
# =============================================================================

class ProcessDisciplineAudit(unittest.TestCase):

    def test_this_module_passes_its_own_audit(self):
        check = t0.audit_process_discipline()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_audit_catches_a_name_pattern_kill(self):
        source = "import subprocess\ndef f():\n    subprocess.call(['pkill', '-f', 'llama'])\n"
        check = t0.audit_process_discipline(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("call" in r for r in check.reasons))

    def test_audit_catches_shell_true(self):
        source = "import subprocess\ndef f():\n    subprocess.Popen('x | y', shell=True)\n"
        check = t0.audit_process_discipline(source)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertTrue(any("shell=True" in r for r in check.reasons))

    def test_audit_catches_os_system(self):
        check = t0.audit_process_discipline("import os\ndef f():\n    os.system('ls')\n")
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_audit_passes_a_compliant_source(self):
        source = ("import os\ndef f(pid):\n    os.kill(pid, 15)\n")
        check = t0.audit_process_discipline(source)
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_module_source_contains_no_name_pattern_tool(self):
        text = Path(t0.__file__).read_text(encoding="utf-8")
        for token in ("pkill ", "pgrep ", "killall"):
            self.assertNotIn(f"{token}(", text)


# =============================================================================
# J. Collection — the provider driven entirely from recorded output
# =============================================================================

def _console_capture(provider_plan, text, *, ops=None, params_filter=None):
    """Build the recorded capture for the argv the provider will actually emit."""
    inv = t0.build_backend_ops_invocation(
        binary=provider_plan.candidate.test_backend_ops,
        library_path=provider_plan.candidate.library_path,
        backend_filter=provider_plan.op_suite.backend_filter,
        ops=ops or provider_plan.op_suite.ops,
        base_env=provider_plan.base_env,
        suite_seed=provider_plan.op_suite.suite_seed,
        layout_probe=provider_plan.op_suite.layout_probe,
        value_transform_probe=provider_plan.op_suite.value_transform_probe,
        stateful_probe=provider_plan.op_suite.stateful_probe,
        params_filter=params_filter)
    return capture(inv.argv, stdout=text, exit_code=0)


def _linkage_capture(provider_plan, text, *, binary=None, library_path=None, exit_code=0):
    binary = binary or provider_plan.candidate.binary
    library_path = library_path or provider_plan.candidate.library_path
    inv = t0.build_linkage_invocation(
        bash=provider_plan.tools.bash, script=provider_plan.tools.verify_ggml_linkage_sh,
        binary=binary, expected_root=library_path, library_path=library_path,
        base_env=provider_plan.base_env)
    return capture(inv.argv, stdout=text, exit_code=exit_code)


class OpSuiteCollection(unittest.TestCase):

    def test_stateful_pass_binds_all_four_ops_and_the_suite_seed(self):
        ops = ("SSM_SCAN", "SSM_CONV", "FLASH_ATTN_EXT", "GATED_DELTA_NET")
        plan = execution_plan(op_suite=op_suite_plan(
            ops=ops, suite_seed=4711, stateful_probe=True))
        rows = "".join(
            f"  {op}(type=f32): AK_STATE_V1 inputs=1 initial_equal=1 "
            f"input_immutable=1 final_outputs=1 suite_seed=4711 OK\n"
            for op in ops)
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n" + rows +
                "  4/4 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        evidence = provider.collect_op_suite(t0._Collected())
        self.assertTrue(evidence.stateful_probe)
        self.assertEqual(evidence.stateful_case_count, 4)
        self.assertEqual(evidence.stateful_ops,
                         ("FLASH_ATTN_EXT", "GATED_DELTA_NET", "SSM_CONV", "SSM_SCAN"))

    def test_stateful_pass_refuses_a_case_without_its_receipt(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("SSM_SCAN",), suite_seed=4711, stateful_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SSM_SCAN(type=f32): OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        with self.assertRaises(t0.OutputParseError):
            provider.collect_op_suite(t0._Collected())

    def test_stateful_receipt_with_wrong_seed_refuses(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("SSM_SCAN",), suite_seed=4711, stateful_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SSM_SCAN(type=f32): AK_STATE_V1 inputs=1 initial_equal=1 "
                "input_immutable=1 final_outputs=1 suite_seed=99 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        with self.assertRaises(t0.OutputParseError):
            provider.collect_op_suite(t0._Collected())

    def test_value_transform_pass_binds_all_four_transforms_and_seed(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("SOFT_MAX",), suite_seed=4711, value_transform_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SOFT_MAX(type=f32): AK_VALUE_V1 "
                "transforms=identity,x3,x0p01,negate completed=4 suite_seed=4711 | "
                "AK_PROP_V2 metric=softmax_invariants/v1 residual=2e-08 "
                "tolerance=0.0001 passed=1 suite_seed=4711 transform=negate OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        evidence = provider.collect_op_suite(t0._Collected())
        self.assertTrue(evidence.value_transform_probe)
        self.assertEqual(evidence.value_transform_case_count, 1)
        self.assertEqual(evidence.value_transforms,
                         ("identity", "negate", "x0p01", "x3"))
        self.assertEqual(evidence.property_measurements[0].input_transform, "negate")

    def test_value_transform_receipt_with_wrong_seed_refuses(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("MUL_MAT",), suite_seed=4711, value_transform_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(type=f32): AK_VALUE_V1 "
                "transforms=identity,x3,x0p01,negate completed=4 suite_seed=99 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        with self.assertRaises(t0.OutputParseError):
            provider.collect_op_suite(t0._Collected())

    def test_value_transform_pass_refuses_a_case_without_its_receipt(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("MUL_MAT",), suite_seed=4711, value_transform_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(type=f32): OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        with self.assertRaises(t0.OutputParseError):
            provider.collect_op_suite(t0._Collected())

    def test_value_transform_pass_refuses_legacy_property_receipt(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("SOFT_MAX",), suite_seed=4711, value_transform_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SOFT_MAX(type=f32): AK_VALUE_V1 "
                "transforms=identity,x3,x0p01,negate completed=4 suite_seed=4711 | "
                "AK_PROP_V1 metric=softmax_invariants/v1 residual=2e-08 "
                "tolerance=0.0001 passed=1 suite_seed=4711 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        with self.assertRaises(t0.OutputParseError):
            provider.collect_op_suite(t0._Collected())

    def test_layout_pass_requires_all_families_and_binds_the_suite_seed(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("MUL_MAT",), suite_seed=4711, layout_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(case=offset): AK_LAYOUT_V1 families=offset "
                "suite_seed=4711 OK\n"
                "  MUL_MAT(case=transpose): AK_LAYOUT_V1 families=transpose "
                "suite_seed=4711 OK\n"
                "  MUL_MAT(case=stride): AK_LAYOUT_V1 families=stride_gap "
                "suite_seed=4711 OK\n"
                "  3/3 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        evidence = provider.collect_op_suite(t0._Collected())
        self.assertTrue(evidence.layout_probe)
        self.assertEqual(evidence.layout_case_count, 3)
        self.assertEqual(evidence.layout_families,
                         ("offset", "stride_gap", "transpose"))

    def test_layout_receipt_with_the_wrong_seed_refuses(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("MUL_MAT",), suite_seed=4711, layout_probe=True))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  MUL_MAT(case=offset): AK_LAYOUT_V1 families=offset "
                "suite_seed=99 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        with self.assertRaises(t0.OutputParseError):
            provider.collect_op_suite(t0._Collected())

    def test_property_residuals_bind_backend_shape_and_suite_seed(self):
        plan = execution_plan(op_suite=op_suite_plan(
            ops=("SOFT_MAX",), suite_seed=4711))
        text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                "  SOFT_MAX(type=f32,ne=[83,2,1,1]): "
                "AK_PROP_V1 metric=softmax_invariants/v1 residual=2.5e-08 "
                "tolerance=0.0001 passed=1 suite_seed=4711 OK\n"
                "  1/1 tests passed\n  Backend CPU: OK\n"
                "1/1 backends passed\nOK\n")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([_console_capture(plan, text)]),
            claim=FakeClaim())
        evidence = provider.collect_op_suite(t0._Collected())
        self.assertEqual(len(evidence.property_measurements), 1)
        measurement = evidence.property_measurements[0]
        self.assertEqual(measurement.backend, "CPU")
        self.assertEqual(measurement.op, "SOFT_MAX")
        self.assertEqual(measurement.shape_id,
                         "SOFT_MAX(type=f32,ne=[83,2,1,1])#0")
        self.assertEqual(measurement.suite_seed, 4711)

    def test_op_suite_evidence_reports_only_what_ran(self):
        plan = execution_plan()
        runner = t0.RecordedProcessRunner([
            _console_capture(plan, recorded("recorded_t0_backend_ops_mandatory_ops.txt"))])
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        collected = t0._Collected()
        evidence = provider.collect_op_suite(collected)
        self.assertEqual(evidence.ops_exercised, ("MUL_MAT",))
        self.assertEqual(evidence.produced_by, "evaluator")
        self.assertEqual(evidence.cases_for("MUL_MAT"), (178, 178))
        self.assertIsNone(evidence.cases_for("MUL_MAT_ID"))
        self.assertTrue(any("MUL_MAT_ID" in note for note in collected.notes))
        self.assertIsNone(provider._op_suite_reference)

    def test_the_gate_fails_the_unexercised_mandatory_op(self):
        """End to end through the REAL gate, on REAL recorded output.

        This is the whole point of the module: `check_backend_op_units` FAILs,
        naming `MUL_MAT_ID`, from a run that exited 0 and printed OK.
        """
        plan = execution_plan()
        runner = t0.RecordedProcessRunner([
            _console_capture(plan, recorded("recorded_t0_backend_ops_mandatory_ops.txt"))])
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        evidence = provider.collect_op_suite(t0._Collected())
        request = evaluation_request()
        surface = correctness.ChangeSurface(
            derived_touches_memory=False, derived_touches_threading=False,
            derived_touches_dispatch=False, derived_touches_persistent_state=False,
            derived_ops=(), derived_files=(), declared_touches_memory=False,
            declared_touches_threading=False, declared_ops=(),
            touches_shared_core_header=False, derivation_ref="ref://derivation")
        gate = correctness.check_backend_op_units(request, evidence, surface, t0_policy())
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("MUL_MAT_ID" in reason for reason in gate.check.reasons))

    def test_skipped_backend_also_fails_the_gate(self):
        plan = execution_plan(op_suite=op_suite_plan(ops=("MUL_MAT", "MUL_MAT_ID")))
        runner = t0.RecordedProcessRunner([
            _console_capture(plan, recorded("recorded_t0_backend_ops_console_skip.txt"))])
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        evidence = provider.collect_op_suite(t0._Collected())
        self.assertEqual(evidence.ops_exercised, ())


class LinkageCollection(unittest.TestCase):

    def _provider(self, *, anchor=None, linkage_text=None):
        plan = execution_plan()
        runner = t0.RecordedProcessRunner([
            _linkage_capture(plan, linkage_text
                             or recorded("recorded_t0_linkage_pass.txt"))])
        return plan, t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner,
                                                   claim=FakeClaim(),
                                                   anchor_capture=anchor)

    def test_linkage_without_an_anchor_capture_records_no_anchor_component(self):
        _, provider = self._provider()
        # The recorded PASS report names libraries in a build dir that does not
        # exist here, so hashing them is what fails; the anchor rule is what is
        # under test, so exercise it on the evidence type directly.
        commit, binary_sha, linkage_sha = t0._anchor_triple(provider.anchor_capture)
        self.assertEqual((commit, binary_sha, linkage_sha), (None, None, None))
        evidence = correctness.LinkageEvidence(
            binary_sha256=SHA_D, linkage_sha256=SHA_E,
            anchor_source_commit=commit, anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            resolved_libraries=(("libggml.so.0", f"{BIN_DIR}/libggml.so.0", SHA_A),),
            expected_library_root=BIN_DIR, verifier_id="verify_ggml_linkage.sh@abc",
            receipt_ref="akcap:x", produced_by="evaluator")
        gate = correctness.check_binary_and_linkage_identity(
            evaluation_request(anchor=api.AnchorIdentity(source_commit=COMMIT_ANCHOR,
                                                         binary_sha256=SHA_B,
                                                         linkage_sha256=SHA_C)),
            evidence, None)
        self.assertEqual(gate.check.outcome, schemas.COULD_NOT_CHECK)

    def test_measured_anchor_that_differs_from_the_request_fails_the_gate(self):
        """The honesty rule, proved through the consumer.

        The provider records the anchor it MEASURED. When the request names a
        different one, `check_binary_and_linkage_identity` FAILs — which it can
        only do because the evidence was not copied from the request.
        """
        commit, binary_sha, linkage_sha = t0._anchor_triple(anchor_capture())
        evidence = correctness.LinkageEvidence(
            binary_sha256=SHA_D, linkage_sha256=SHA_E,
            anchor_source_commit=commit, anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            resolved_libraries=(("libggml.so.0", f"{BIN_DIR}/libggml.so.0", SHA_A),),
            expected_library_root=BIN_DIR, verifier_id="verify_ggml_linkage.sh@abc",
            receipt_ref="akcap:x", produced_by="evaluator")
        rebuilt = api.AnchorIdentity(source_commit=COMMIT_ANCHOR, binary_sha256=SHA_D,
                                     linkage_sha256=SHA_C)
        gate = correctness.check_binary_and_linkage_identity(
            evaluation_request(anchor=rebuilt), evidence, None)
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("rebuilt anchor is a different anchor" in r
                            for r in gate.check.reasons))

    def test_matching_anchor_passes(self):
        commit, binary_sha, linkage_sha = t0._anchor_triple(anchor_capture())
        evidence = correctness.LinkageEvidence(
            binary_sha256=SHA_D, linkage_sha256=SHA_E,
            anchor_source_commit=commit, anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            resolved_libraries=(("libggml.so.0", f"{BIN_DIR}/libggml.so.0", SHA_A),),
            expected_library_root=BIN_DIR, verifier_id="verify_ggml_linkage.sh@abc",
            receipt_ref="akcap:x", produced_by="evaluator")
        matching = api.AnchorIdentity(source_commit=COMMIT_ANCHOR, binary_sha256=SHA_B,
                                      linkage_sha256=SHA_C)
        gate = correctness.check_binary_and_linkage_identity(
            evaluation_request(anchor=matching), evidence, None)
        self.assertEqual(gate.check.outcome, schemas.PASS, gate.check.reasons)

    def test_stray_library_fails_the_gate(self):
        report = t0.parse_linkage_report(recorded("recorded_t0_linkage_fail.txt"))
        rows = tuple((row.soname, row.path, SHA_A) for row in report.rows)
        commit, binary_sha, linkage_sha = t0._anchor_triple(anchor_capture())
        evidence = correctness.LinkageEvidence(
            binary_sha256=SHA_D, linkage_sha256=SHA_E,
            anchor_source_commit=commit, anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha, resolved_libraries=rows,
            expected_library_root=report.expected_root,
            verifier_id="verify_ggml_linkage.sh@abc", receipt_ref="akcap:x",
            produced_by="evaluator")
        matching = api.AnchorIdentity(source_commit=COMMIT_ANCHOR, binary_sha256=SHA_B,
                                      linkage_sha256=SHA_C)
        gate = correctness.check_binary_and_linkage_identity(
            evaluation_request(anchor=matching), evidence, None)
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("SILENTLY wrong" in r for r in gate.check.reasons))


class DispatchTraceCollection(unittest.TestCase):

    TRACE = ("## SPLIT #0: CPU # 0 inputs\n"
             "node #  0 (   MUL_MAT):            ffn_up-0 (  1MB) [  CPU 1.dst  ] use=1,c=1:\n")

    def _provider(self, trace, *, fallback_scope="inter_backend"):
        plan = execution_plan(generation=generation_plan(),
                              dispatch=t0.DispatchTracePlan(
                                  derived_surface=("MUL_MAT",),
                                  fallback_scope=fallback_scope))
        inv = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env,
            extra_env={"GGML_SCHED_DEBUG": "2"})
        runner = t0.RecordedProcessRunner([capture(inv.argv, stdout=trace)])
        return t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())

    def test_instrumented_trace_is_active_and_the_gate_passes(self):
        provider = self._provider(self.TRACE)
        evidence = provider.collect_dispatch_trace(t0._Collected())
        self.assertTrue(evidence.fallback_instrumentation_active)
        self.assertEqual(evidence.traced_kernels, ("MUL_MAT",))
        gate = correctness.check_no_fallback_dispatch_proof(evidence)
        self.assertEqual(gate.check.outcome, schemas.PASS, gate.check.reasons)

    def test_uninstrumented_trace_is_could_not_check_not_pass(self):
        """An empty fallback list from an uninstrumented trace is a fact about the
        instrument. Without the guard this PASSes."""
        provider = self._provider("llama_perf_context_print: load time = 1 ms\n")
        evidence = provider.collect_dispatch_trace(t0._Collected())
        self.assertFalse(evidence.fallback_instrumentation_active)
        self.assertEqual(evidence.fallback_events, ())
        gate = correctness.check_no_fallback_dispatch_proof(evidence)
        self.assertEqual(gate.check.outcome, schemas.COULD_NOT_CHECK)

    def test_out_of_scope_fallback_class_reports_no_instrumentation(self):
        """`GGML_SCHED_DEBUG` cannot see intra-backend kernel selection."""
        provider = self._provider(self.TRACE, fallback_scope="intra_backend_kernel_selection")
        collected = t0._Collected()
        evidence = provider.collect_dispatch_trace(collected)
        self.assertFalse(evidence.fallback_instrumentation_active)
        self.assertTrue(any("outside what GGML_SCHED_DEBUG can observe" in n
                            for n in collected.notes))

    def test_node_on_another_backend_is_a_fallback_event(self):
        trace = self.TRACE + (
            "node #  1 (   MUL_MAT):           ffn_moe-0 (  2MB) [ ROCm0 1.src ] use=1,c=1:\n")
        provider = self._provider(trace)
        evidence = provider.collect_dispatch_trace(t0._Collected())
        self.assertEqual(len(evidence.fallback_events), 1)
        gate = correctness.check_no_fallback_dispatch_proof(evidence)
        self.assertEqual(gate.check.outcome, schemas.FAIL)

    def test_unaffected_graph_nodes_do_not_expand_the_change_surface(self):
        trace = self.TRACE + (
            "node #  1 (       ADD):             residual (  2MB) [  CPU 1.dst  ] use=1,c=1:\n")
        provider = self._provider(trace)
        collected = t0._Collected()
        evidence = provider.collect_dispatch_trace(collected)
        self.assertEqual(evidence.traced_kernels, ("MUL_MAT",))
        self.assertTrue(any("outside the mechanically derived affected surface" in note
                            for note in collected.notes))


class CoherenceAndDeterminismCollection(unittest.TestCase):

    def _runner(self, plan, outputs):
        captures = []
        for text in outputs:
            inv = t0.build_generation_invocation(
                binary=plan.candidate.binary, library_path=plan.candidate.library_path,
                plan=plan.generation, base_env=plan.base_env, seed=plan.generation.seed)
            captures.append(capture(inv.argv, stdout=text))
        return t0.RecordedProcessRunner(captures)

    def test_coherence_without_an_anchor_capture_is_not_compared(self):
        plan = execution_plan(generation=generation_plan())
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._runner(plan, ["Paris."]), claim=FakeClaim())
        evidence = provider.collect_coherence(t0._Collected())
        self.assertIsNone(evidence.anchor_source_commit)
        self.assertIsNone(evidence.anchor_output_sha256)
        verdict = correctness.compute_coherence(anchor=None, evidence=evidence,
                                                tolerance_floor=None)
        self.assertEqual(verdict.label, correctness.COHERENCE_NOT_COMPARED)

    def test_byte_identical_output_against_a_named_anchor(self):
        text = "Paris."
        anchor = anchor_capture(output_digests=(t0.sha256_text(text),),
                                output_lengths=(len(text),),
                                determinism_class="bitwise_stable")
        plan = execution_plan(generation=generation_plan())
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._runner(plan, [text]), claim=FakeClaim(),
            anchor_capture=anchor)
        evidence = provider.collect_coherence(t0._Collected())
        self.assertEqual(evidence.anchor_source_commit, COMMIT_ANCHOR)
        self.assertTrue(evidence.sampler_is_greedy)
        verdict = correctness.compute_coherence(anchor=anchor.identity(), evidence=evidence,
                                                tolerance_floor=None)
        self.assertEqual(verdict.label, correctness.COHERENCE_BYTE_IDENTICAL)

    def test_replaying_against_a_different_anchor_raises(self):
        """Invariant 11's replay path, refused rather than mislabelled."""
        text = "Paris."
        anchor = anchor_capture(output_digests=(t0.sha256_text(text),),
                                output_lengths=(len(text),),
                                determinism_class="bitwise_stable")
        plan = execution_plan(generation=generation_plan())
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._runner(plan, [text]), claim=FakeClaim(),
            anchor_capture=anchor)
        evidence = provider.collect_coherence(t0._Collected())
        other = api.AnchorIdentity(source_commit="9" * 40, binary_sha256=SHA_B,
                                   linkage_sha256=SHA_C)
        with self.assertRaises(correctness.CoherenceAnchorMismatch):
            correctness.compute_coherence(anchor=other, evidence=evidence,
                                          tolerance_floor=None)

    def test_non_greedy_sampler_makes_the_comparison_undecidable(self):
        text = "Paris."
        anchor = anchor_capture(output_digests=(t0.sha256_text(text),),
                                output_lengths=(len(text),),
                                determinism_class="bitwise_stable")
        plan = execution_plan(generation=generation_plan(temperature=0.7, top_k=40))
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._runner(plan, [text]), claim=FakeClaim(),
            anchor_capture=anchor)
        evidence = provider.collect_coherence(t0._Collected())
        self.assertFalse(evidence.sampler_is_greedy)
        verdict = correctness.compute_coherence(anchor=anchor.identity(), evidence=evidence,
                                                tolerance_floor=None)
        self.assertEqual(verdict.label, correctness.COHERENCE_UNDECIDABLE)

    def test_determinism_class_is_measured_from_the_repeats(self):
        plan = execution_plan(generation=generation_plan(), determinism_runs=3)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._runner(plan, ["Paris.", "Paris.", "Paris."]),
            claim=FakeClaim(), anchor_capture=anchor_capture(
                determinism_class="bitwise_stable"))
        evidence = provider.collect_determinism(t0._Collected())
        self.assertEqual(evidence.runs, 3)
        self.assertEqual(evidence.measured_class(), "bitwise_stable")
        self.assertEqual(evidence.anchor_source_commit, COMMIT_ANCHOR)

    def test_unstable_repeats_measure_unstable(self):
        plan = execution_plan(generation=generation_plan(), determinism_runs=3)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._runner(plan, ["Paris.", "Paris!", "Paris."]),
            claim=FakeClaim(), anchor_capture=anchor_capture(
                determinism_class="bitwise_stable"))
        evidence = provider.collect_determinism(t0._Collected())
        self.assertEqual(evidence.measured_class(), "bitwise_unstable")

    def test_empty_generation_is_a_candidate_defect(self):
        anchor = anchor_capture(output_digests=(SHA_B,), output_lengths=(6,))
        plan = execution_plan(generation=generation_plan())
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=self._runner(plan, [""]), claim=FakeClaim(),
            anchor_capture=anchor)
        evidence = provider.collect_coherence(t0._Collected())
        self.assertEqual(evidence.candidate_output_len, 0)
        verdict = correctness.compute_coherence(anchor=anchor.identity(), evidence=evidence,
                                                tolerance_floor=None)
        self.assertEqual(verdict.label, correctness.COHERENCE_EMPTY)


class AntiRewardHackingCollection(unittest.TestCase):

    def test_unknown_cache_state_is_could_not_check_not_a_pass(self):
        plan = execution_plan(cache_state="unknown", oracle_ids=("oracle://anchor",))
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), claim=FakeClaim())
        evidence = provider.collect_anti_reward_hacking(128, t0._Collected())
        self.assertEqual(evidence.correctness_verdict_source, "evaluator")
        self.assertFalse(evidence.candidate_output_used_as_oracle)
        gate = correctness.check_anti_reward_hacking(evidence, None, None)
        self.assertEqual(gate.check.outcome, schemas.COULD_NOT_CHECK)

    def test_declared_cold_cache_with_an_anchor_floor_passes(self):
        anchor = anchor_capture(delivered_units=128, oracle_ids=("oracle://anchor",))
        plan = execution_plan(cache_state="cold")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), claim=FakeClaim(),
            anchor_capture=anchor)
        evidence = provider.collect_anti_reward_hacking(128, t0._Collected())
        gate = correctness.check_anti_reward_hacking(evidence, None, anchor.identity())
        self.assertEqual(gate.check.outcome, schemas.PASS, gate.check.reasons)

    def test_reduced_delivered_work_fails(self):
        anchor = anchor_capture(delivered_units=128, oracle_ids=("oracle://anchor",))
        plan = execution_plan(cache_state="cold")
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), claim=FakeClaim(),
            anchor_capture=anchor)
        evidence = provider.collect_anti_reward_hacking(64, t0._Collected())
        gate = correctness.check_anti_reward_hacking(evidence, None, anchor.identity())
        self.assertEqual(gate.check.outcome, schemas.FAIL)


class ProducedByDiscipline(unittest.TestCase):

    def test_producer_constant_is_the_evaluator(self):
        self.assertEqual(t0.PRODUCER, "evaluator")
        self.assertIn(t0.PRODUCER, correctness.EVIDENCE_PRODUCERS)

    def test_every_produced_evidence_object_is_stamped(self):
        """Sweep, not a spot check: any new producer that forgets fails here."""
        text = "Paris."
        anchor = anchor_capture(output_digests=(t0.sha256_text(text),),
                                output_lengths=(len(text),),
                                determinism_class="bitwise_stable",
                                delivered_units=8, oracle_ids=("oracle://anchor",))
        plan = execution_plan(generation=generation_plan(), determinism_runs=2,
                              cache_state="cold", state_safety_probe=True)
        gen = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env, seed=plan.generation.seed)
        trace = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env,
            extra_env={"GGML_SCHED_DEBUG": "2"})
        runner = t0.RecordedProcessRunner([
            _console_capture(plan, recorded("recorded_t0_backend_ops_console_ok.txt")),
            capture(trace.argv, stdout="## SPLIT #0: CPU # 0 inputs\n"),
            capture(gen.argv, stdout=text),
        ])
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim(),
                                                 anchor_capture=anchor)
        collected = t0._Collected()
        produced = [
            provider.collect_op_suite(collected),
            provider.collect_dispatch_trace(collected),
            provider.collect_coherence(collected),
            provider.collect_determinism(collected),
            provider.collect_state_safety(collected),
        ]
        stamped = 0
        for evidence in produced:
            self.assertIsNotNone(evidence)
            if hasattr(evidence, "produced_by"):
                self.assertEqual(evidence.produced_by, "evaluator",
                                 type(evidence).__name__)
                stamped += 1
        self.assertGreaterEqual(stamped, 5)

    def test_schema_followups_hold_only_what_is_still_open(self):
        """The follow-up list is pinned to the code, in BOTH directions.

        It used to pin three gaps and assert `BuildProvenance` and
        `DiffPolicyEvidence` had NO `produced_by` — reported rather than patched,
        because those dataclasses belonged to another agent that hour. Both were
        closed on 2026-08-04, so the assertions invert: the field must now be
        present, and the entries must be GONE from the list. A follow-up list that
        keeps closed items stops being read.

        The third is still open, and is pinned exactly as before.
        """
        self.assertEqual(len(t0.SCHEMA_FOLLOWUPS), 1)
        self.assertNotIn(
            "BuildProvenance", " ".join(t0.SCHEMA_FOLLOWUPS),
            "BuildProvenance.produced_by exists; its follow-up must be deleted")
        self.assertNotIn(
            "DiffPolicyEvidence", " ".join(t0.SCHEMA_FOLLOWUPS),
            "DiffPolicyEvidence.produced_by exists; its follow-up must be deleted")
        # Still open: `delivered_units_candidate` is `int`, so "not read" and
        # "delivered nothing" are the same value. Pinned the same way.
        self.assertEqual(
            correctness.AntiRewardHackingEvidence
            .__dataclass_fields__["delivered_units_candidate"].type, "int",
            "the remaining SCHEMA_FOLLOWUP is closed; delete it")
        for evidence_type in (correctness.BuildProvenance, correctness.DiffPolicyEvidence,
                              correctness.OpSuiteEvidence, correctness.CoherenceEvidence,
                              correctness.DeterminismEvidence, correctness.LinkageEvidence,
                              correctness.StateSafetyEvidence,
                              correctness.BoundaryShapeEvidence,
                              correctness.StaticAnalysisEvidence,
                              correctness.DispatchTraceEvidence,
                              correctness.SanitizerEvidence):
            self.assertIn("produced_by", evidence_type.__dataclass_fields__,
                          evidence_type.__name__)


class BoundaryShapeCollection(unittest.TestCase):

    def _plan(self, visible):
        return execution_plan(holdout=t0.HoldoutPlan(
            unseen_case_filter="k=1024", boundary_case_filter="k=1",
            selection_rule_id="ak.holdout.seeded/v1", selection_seed="seed-7",
            visible_to_planner=visible))

    def _runner(self, plan):
        text = recorded("recorded_t0_backend_ops_console_ok.txt")
        return t0.RecordedProcessRunner([
            _console_capture(plan, text, params_filter="k=1024"),
            _console_capture(plan, text, params_filter="k=1"),
        ])

    def test_holdout_visible_to_the_planner_fails_the_gate(self):
        plan = self._plan(visible=True)
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=self._runner(plan),
                                                 claim=FakeClaim())
        evidence = provider.collect_boundary_shapes(t0._Collected())
        self.assertFalse(evidence.held_out_from_planner)
        surface = correctness.ChangeSurface(
            derived_touches_memory=None, derived_touches_threading=None,
            derived_touches_dispatch=True, derived_touches_persistent_state=None,
            derived_ops=(), derived_files=(), declared_touches_memory=None,
            declared_touches_threading=None, declared_ops=(),
            touches_shared_core_header=False, derivation_ref="ref://d")
        gate = correctness.check_unseen_boundary_shapes(evidence, surface)
        self.assertEqual(gate.check.outcome, schemas.FAIL)

    def test_genuine_holdout_passes(self):
        plan = self._plan(visible=False)
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=self._runner(plan),
                                                 claim=FakeClaim())
        evidence = provider.collect_boundary_shapes(t0._Collected())
        self.assertTrue(evidence.held_out_from_planner)
        self.assertTrue(evidence.unseen_shapes)
        self.assertTrue(evidence.boundary_shapes)
        surface = correctness.ChangeSurface(
            derived_touches_memory=None, derived_touches_threading=None,
            derived_touches_dispatch=True, derived_touches_persistent_state=None,
            derived_ops=(), derived_files=(), declared_touches_memory=None,
            declared_touches_threading=None, declared_ops=(),
            touches_shared_core_header=False, derivation_ref="ref://d")
        gate = correctness.check_unseen_boundary_shapes(evidence, surface)
        self.assertEqual(gate.check.outcome, schemas.PASS, gate.check.reasons)


class SanitizerCollection(unittest.TestCase):

    def _plan(self):
        return execution_plan(
            sanitizer_target="test-backend-ops",
            sanitizer_build_dir="/mnt/raid0/llm/llama.cpp-experimental/build-asan",
            op_suite=op_suite_plan(ops=("MUL_MAT",)),
            # The instrumented binary does not exist on this filesystem, so its
            # digest is supplied rather than measured. There is no third option:
            # substituting the LOG's digest is what made
            # `_sanitizer_preamble`'s distinct-builds guard unfalsifiable.
            sanitizer_binary_sha256=SHA_F)

    def _captures(self, plan, log):
        invocation = correctness.build_sanitizer_invocation(
            source_dir=plan.candidate.worktree, build_dir=plan.sanitizer_build_dir,
            target=plan.sanitizer_target,
            run_argv=(f"{plan.sanitizer_build_dir}/bin/{plan.sanitizer_target}",
                      "test", "-o", "MUL_MAT", "-b", "CPU"),
            jobs=plan.sanitizer_jobs, backend=plan.backend)
        out = []
        for stage in (invocation.configure_argv, invocation.build_argv,
                      invocation.run_argv):
            argv = list(stage)
            if argv[0] == "cmake":
                argv[0] = plan.tools.cmake
            out.append(capture(argv, stdout="",
                               stderr=log if stage is invocation.run_argv else ""))
        return out, invocation

    def test_the_constructed_invocation_would_gate(self):
        """Compliant-path control for the guard below: the real constructor's
        argv passes `check_sanitizer_invocation`."""
        plan = self._plan()
        _, invocation = self._captures(plan, "")
        self.assertEqual(correctness.check_sanitizer_invocation(invocation).outcome,
                         schemas.PASS)

    def test_findings_are_parsed_and_split_by_surface(self):
        plan = self._plan()
        captures, _ = self._captures(plan, SYNTHETIC_SANITIZER_LOG)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner(captures), claim=FakeClaim())
        evidence = provider.collect_sanitizers(t0._Collected())
        self.assertTrue(evidence.executed)
        self.assertEqual(evidence.exit_code, 0)
        self.assertEqual(len(evidence.asan_findings), 2)
        self.assertEqual(len(evidence.ubsan_findings), 2)
        self.assertEqual(evidence.produced_by, "evaluator")

    def test_no_sanitizer_plan_yields_no_evidence(self):
        provider = t0.ExecutedT0EvidenceProvider(
            plan=execution_plan(), runner=t0.RecordedProcessRunner([]), claim=FakeClaim())
        self.assertIsNone(provider.collect_sanitizers(t0._Collected()))


# =============================================================================
# K. The whole provider, through the real T0 runner
# =============================================================================

class EndToEndT0Report(unittest.TestCase):
    """`evidence_for` -> `evaluate_t0` -> a seventeen-gate `T0Report`.

    Every input is either a recorded capture or a value derived from one. The
    report this produces is the first T0 report in this package whose op-suite
    surface is a real measurement rather than a fixture.
    """

    def _build(self, *, op_suite_text, anchor=None, generation_text="Paris."):
        plan = execution_plan(generation=generation_plan(), determinism_runs=2,
                              cache_state="cold", state_safety_probe=True,
                              oracle_ids=("oracle://anchor-v8",))
        gen = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env, seed=plan.generation.seed)
        trace = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env,
            extra_env={"GGML_SCHED_DEBUG": "2"})
        perf = ("llama_perf_context_print:        eval time =   1234.56 ms /    32 runs   "
                "(   38.58 ms per token,    25.92 tokens per second)\n")
        runner = t0.RecordedProcessRunner([
            _console_capture(plan, op_suite_text),
            capture(trace.argv, stdout="## SPLIT #0: CPU # 0 inputs\n" + generation_text,
                    stderr=perf),
            _linkage_capture(plan, recorded("recorded_t0_linkage_pass.txt")),
            capture(gen.argv, stdout=generation_text, stderr=perf),
        ])
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim(),
                                                 anchor_capture=anchor)
        return plan, provider

    def _evidence(self, provider, request):
        # `collect_linkage` hashes every resolved library path; the recorded
        # report names a build dir that is not present here, so linkage is
        # exercised in its own test class and substituted with a measured-shape
        # object built from the SAME recorded report.
        report = t0.parse_linkage_report(recorded("recorded_t0_linkage_pass.txt"))
        commit, binary_sha, linkage_sha = t0._anchor_triple(provider.anchor_capture)
        linkage = correctness.LinkageEvidence(
            binary_sha256=request.artifact.binary_sha256,
            linkage_sha256=request.artifact.linkage_sha256,
            anchor_source_commit=commit, anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            resolved_libraries=tuple((r.soname, r.path, SHA_A) for r in report.rows),
            expected_library_root=report.expected_root,
            verifier_id="verify_ggml_linkage.sh@recorded", receipt_ref="akcap:linkage",
            produced_by="evaluator")
        provider.collect_linkage = lambda collected: linkage    # noqa: E731
        return provider.evidence_for(request)

    def test_report_covers_every_gate_and_fails_the_unexercised_mandatory_op(self):
        anchor = anchor_capture(output_digests=(t0.sha256_text("Paris."),),
                                output_lengths=(6,), determinism_class="bitwise_stable",
                                delivered_units=32, oracle_ids=("oracle://anchor-v8",))
        _, provider = self._build(
            op_suite_text=recorded("recorded_t0_backend_ops_mandatory_ops.txt"),
            anchor=anchor)
        request = evaluation_request(anchor=anchor.identity(),
                                     determinism_class="bitwise_stable", repeats=2)
        evidence = self._evidence(provider, request)
        self.assertIsInstance(evidence, correctness.T0Evidence)
        report = correctness.evaluate_t0(request, evidence, t0_policy())
        self.assertEqual(len(report.gates), len(correctness.T0_GATE_IDS))
        self.assertIn(correctness.GID_OP_UNITS, report.failed)
        self.assertEqual(report.outcome(correctness.GID_COHERENCE), schemas.PASS)
        self.assertEqual(report.coherence.label, correctness.COHERENCE_BYTE_IDENTICAL)
        self.assertEqual(report.outcome(correctness.GID_NO_FALLBACK), schemas.PASS)
        self.assertEqual(report.outcome(correctness.GID_ANTI_REWARD_HACKING), schemas.PASS)

    def test_passing_op_suite_passes_its_gate(self):
        """Compliant-path control: the same pipeline with both ops exercised.

        Built by substituting op names into the RECORDED grammar, which is
        labelled here rather than passed off as a capture: no CPU-only build can
        produce a real MUL_MAT_ID case list under a shape filter, which is the
        finding the other test records.
        """
        receipt = ("AK_REF_V1 metric=test_backend_ops_error/v1 observed=0 "
                   "tolerance=1e-07 comparisons=2 oracle=ggml_cpu_reference/v1")
        synthetic = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                     "  Device description: AMD EPYC 9655 96-Core Processor\n\n"
                     f"  MUL_MAT(type_a=f32,type_b=f32,m=16,n=1,k=256): {receipt} OK\n"
                     f"  MUL_MAT_ID(type_a=f32,type_b=f32,n_mats=4,n_used=2): {receipt} OK\n"
                     "  2/2 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n")
        anchor = anchor_capture(output_digests=(t0.sha256_text("Paris."),),
                                output_lengths=(6,), determinism_class="bitwise_stable",
                                delivered_units=32, oracle_ids=("oracle://anchor-v8",))
        _, provider = self._build(op_suite_text=synthetic, anchor=anchor)
        request = evaluation_request(anchor=anchor.identity(),
                                     determinism_class="bitwise_stable", repeats=2)
        evidence = self._evidence(provider, request)
        report = correctness.evaluate_t0(request, evidence, t0_policy())
        self.assertNotIn(correctness.GID_OP_UNITS, report.failed)
        self.assertEqual(report.outcome(correctness.GID_OP_UNITS), schemas.PASS,
                         report.gate(correctness.GID_OP_UNITS).check.reasons)
        self.assertEqual(report.outcome(correctness.GID_EXACT_REFERENCE), schemas.PASS,
                         report.gate(correctness.GID_EXACT_REFERENCE).check.reasons)
        self.assertEqual(len(evidence.reference.comparisons), 2)
        self.assertTrue(all(c.mode == "metric_bounded"
                            for c in evidence.reference.comparisons))

    def test_provider_satisfies_the_t0_correctness_runner_seam(self):
        anchor = anchor_capture(output_digests=(t0.sha256_text("Paris."),),
                                output_lengths=(6,), determinism_class="bitwise_stable",
                                delivered_units=32, oracle_ids=("oracle://anchor-v8",))
        _, provider = self._build(
            op_suite_text=recorded("recorded_t0_backend_ops_console_ok.txt"), anchor=anchor)
        request = evaluation_request(anchor=anchor.identity(),
                                     determinism_class="bitwise_stable", repeats=2)
        report = t0.parse_linkage_report(recorded("recorded_t0_linkage_pass.txt"))
        commit, binary_sha, linkage_sha = t0._anchor_triple(anchor)
        provider.collect_linkage = lambda collected: correctness.LinkageEvidence(  # noqa: E731
            binary_sha256=request.artifact.binary_sha256,
            linkage_sha256=request.artifact.linkage_sha256,
            anchor_source_commit=commit, anchor_binary_sha256=binary_sha,
            anchor_linkage_sha256=linkage_sha,
            resolved_libraries=tuple((r.soname, r.path, SHA_A) for r in report.rows),
            expected_library_root=report.expected_root,
            verifier_id="verify_ggml_linkage.sh@recorded", receipt_ref="akcap:linkage",
            produced_by="evaluator")
        runner = correctness.T0CorrectnessRunner(provider=provider, policy=t0_policy())
        gates = runner.run_gates(request)
        self.assertEqual(len(gates), len(correctness.T0_GATE_IDS))


class SeamsAreRecorded(unittest.TestCase):

    def test_reference_evidence_is_instrument_derived_not_plan_asserted(self):
        plan = execution_plan()
        self.assertIsNone(plan.reference)
        self.assertFalse(any("ReferenceEvidence" in seam for seam in t0.SEAMS))

    def test_seams_name_the_dispatch_instrumentation_limit(self):
        self.assertTrue(any("INTER-backend" in seam for seam in t0.SEAMS))


# =============================================================================
# R. RED-TEAM REGRESSIONS
#
# Every test below FAILED against the module as delivered. Each names the
# attack it closes and, where the fix could be satisfied by weakening the
# guard, is paired with a compliant-path control that keeps passing.
# =============================================================================

class ProductionTreeContainmentIsStructural(unittest.TestCase):
    """A. The one boundary that must be structural, not checked.

    `under_production_tree` compared `str(Path(path))` against the root list.
    `str(Path(...))` folds `//` and `.` and leaves `..` exactly where it was,
    and it resolves no symlinks at all — so two spellings of the frozen tree
    read as "not production" and were accepted as CANDIDATE BUILD PATHS:

        /mnt/raid0/llm/exp/../llama.cpp/build      -> a cmake build into v8
        <any symlink>/build                        -> the same, via one symlink

    Both were verified accepted before the fix: `CandidateBuild` constructed
    without complaint and `_refuse_production_write` returned the path.
    """

    DOTDOT = "/mnt/raid0/llm/llama.cpp-experimental/../llama.cpp"

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self.symlink = os.path.join(self._tmp, "candidate-worktree")
        os.symlink(correctness.PRODUCTION_TREE_ROOTS[0], self.symlink)

    def _candidate(self, base):
        return t0.CandidateBuild(
            worktree=base, build_dir=f"{base}/build", source_commit=COMMIT_CANDIDATE,
            source_sha256=SHA_A, binary=f"{base}/build/bin/llama-cli",
            library_path=f"{base}/build/bin",
            test_backend_ops=f"{base}/build/bin/test-backend-ops")

    def test_a_dotdot_path_into_the_frozen_tree_is_refused(self):
        self.assertTrue(t0.under_production_tree(f"{self.DOTDOT}/build"))
        with self.assertRaises(t0.ProductionTreeRefusal):
            self._candidate(self.DOTDOT)

    def test_a_symlink_into_the_frozen_tree_is_refused(self):
        self.assertEqual(os.path.realpath(self.symlink),
                         os.path.realpath(correctness.PRODUCTION_TREE_ROOTS[0]),
                         "precondition: the symlink really does reach the frozen tree")
        self.assertTrue(t0.under_production_tree(f"{self.symlink}/build"))
        with self.assertRaises(t0.ProductionTreeRefusal):
            self._candidate(self.symlink)

    def test_a_dotdot_sanitizer_build_dir_is_refused(self):
        with self.assertRaises(t0.ProductionTreeRefusal):
            execution_plan(sanitizer_target="test-backend-ops",
                           sanitizer_build_dir=f"{self.DOTDOT}/build-asan")

    def test_a_symlinked_sanitizer_build_dir_is_refused(self):
        with self.assertRaises(t0.ProductionTreeRefusal):
            execution_plan(sanitizer_target="test-backend-ops",
                           sanitizer_build_dir=f"{self.symlink}/build-asan")

    def test_the_compliant_experimental_path_still_builds(self):
        """The control. A guard that refuses everything is not a guard.

        `llama.cpp-experimental` shares the frozen tree's PREFIX up to a
        hyphen, which is exactly the case a sloppy `startswith` gets wrong in
        the other direction.
        """
        candidate = self._candidate("/mnt/raid0/llm/llama.cpp-experimental")
        self.assertFalse(t0.under_production_tree(candidate.build_dir))
        self.assertEqual(t0.under_production_tree("/mnt/raid0/llm/llama.cpp-experimental"),
                         False)
        plan = execution_plan(sanitizer_target="test-backend-ops",
                              sanitizer_build_dir="/mnt/raid0/llm/llama.cpp-exp/build-asan")
        self.assertEqual(plan.sanitizer_build_dir, "/mnt/raid0/llm/llama.cpp-exp/build-asan")

    def test_the_anchor_may_still_name_the_frozen_tree(self):
        """The second control: anchoring READS production and must keep working."""
        anchor = t0.AnchorBuild(
            worktree=correctness.PRODUCTION_TREE_ROOTS[0], source_commit=COMMIT_ANCHOR,
            binary=f"{correctness.PRODUCTION_TREE_ROOTS[0]}/build/bin/llama-cli",
            library_path=f"{correctness.PRODUCTION_TREE_ROOTS[0]}/build/bin")
        self.assertTrue(t0.under_production_tree(anchor.binary))


class SanitizerRunsTheInstrumentedBinary(unittest.TestCase):
    """D/F. The mandatory ASAN/UBSAN surface, wired to the wrong binary.

    As delivered, `collect_sanitizers` built the instrumented target into
    `sanitizer_build_dir`, then ran `candidate.test_backend_ops` — the ORDINARY
    build — under `candidate.library_path` — the ordinary ggml. Verified live:
    the run stage's argv[0] was `.../build-t0/bin/test-backend-ops` and its
    `LD_LIBRARY_PATH` was `.../build-t0/bin`, while the digest recorded as
    `sanitizer_build_binary_sha256` came off `.../build-asan/bin/...`.

    Consequence: zero findings, exit 0, `executed=True` — `check_asan` and
    `check_ubsan` both PASS — and `_sanitizer_preamble`'s "the sanitizer build's
    binary is not the record's binary" guard is satisfied by the hash of a file
    nothing executed. The delivered test could not see it because it built its
    replay captures from the same wrong argv.
    """

    def _plan(self, **overrides):
        kwargs = dict(sanitizer_target="test-backend-ops",
                      sanitizer_build_dir="/mnt/raid0/llm/llama.cpp-experimental/build-asan",
                      op_suite=op_suite_plan(ops=("MUL_MAT",)),
                      sanitizer_binary_sha256=SHA_F)
        kwargs.update(overrides)
        return execution_plan(**kwargs)

    def _stages(self, plan, *, build_exit=0, log=""):
        invocation = correctness.build_sanitizer_invocation(
            source_dir=plan.candidate.worktree, build_dir=plan.sanitizer_build_dir,
            target=plan.sanitizer_target,
            run_argv=(f"{plan.sanitizer_build_dir}/bin/{plan.sanitizer_target}",
                      "test", "-o", "MUL_MAT", "-b", "CPU"),
            jobs=plan.sanitizer_jobs, backend=plan.backend)
        out = []
        for stage in (invocation.configure_argv, invocation.build_argv, invocation.run_argv):
            argv = list(stage)
            if argv[0] == "cmake":
                argv[0] = plan.tools.cmake
            is_build = stage is invocation.build_argv
            out.append(capture(argv, stderr=log if stage is invocation.run_argv else "",
                               exit_code=build_exit if is_build else 0))
        return out

    def test_the_run_stage_executes_the_binary_that_was_instrumented(self):
        plan = self._plan()
        runner = t0.RecordedProcessRunner(self._stages(plan))
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        provider.collect_sanitizers(t0._Collected())
        run_argv, run_env = runner.calls[2]
        instrumented = f"{plan.sanitizer_build_dir}/bin/{plan.sanitizer_target}"
        self.assertEqual(run_argv[0], instrumented)
        self.assertNotEqual(run_argv[0], plan.candidate.test_backend_ops)
        self.assertEqual(dict(run_env)["LD_LIBRARY_PATH"],
                         f"{plan.sanitizer_build_dir}/bin",
                         "the instrumented binary must load the instrumented ggml")

    def test_the_recorded_digest_names_the_binary_that_ran(self):
        plan = self._plan()
        runner = t0.RecordedProcessRunner(self._stages(plan))
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        evidence = provider.collect_sanitizers(t0._Collected())
        self.assertEqual(evidence.invocation.run_argv[0],
                         f"{plan.sanitizer_build_dir}/bin/{plan.sanitizer_target}")
        self.assertEqual(evidence.sanitizer_build_binary_sha256, SHA_F)

    def test_an_unhashable_binary_with_no_supplied_digest_refuses(self):
        """The old fallback hashed the LOG — a well-formed sha256 naming nothing.

        That value is what `_sanitizer_preamble` compares against
        `request.artifact.binary_sha256`, so the distinct-builds guard could
        never fail whatever was built.
        """
        plan = self._plan(sanitizer_binary_sha256=None)
        runner = t0.RecordedProcessRunner(self._stages(plan, log=SYNTHETIC_SANITIZER_LOG))
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        with self.assertRaises(t0.ExecutionError) as ctx:
            provider.collect_sanitizers(t0._Collected())
        self.assertIn("names the binary that RAN", str(ctx.exception))

    def test_a_failed_build_stage_refuses_instead_of_reporting_a_clean_surface(self):
        plan = self._plan()
        runner = t0.RecordedProcessRunner(self._stages(plan, build_exit=2))
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        with self.assertRaises(t0.ExecutionError) as ctx:
            provider.collect_sanitizers(t0._Collected())
        self.assertIn("did not complete", str(ctx.exception))

    def test_findings_still_parse_on_the_compliant_path(self):
        """The control: a real sanitizer log still produces both finding lists."""
        plan = self._plan()
        runner = t0.RecordedProcessRunner(self._stages(plan, log=SYNTHETIC_SANITIZER_LOG))
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim())
        evidence = provider.collect_sanitizers(t0._Collected())
        self.assertTrue(evidence.executed)
        self.assertEqual(len(evidence.asan_findings), 2)
        self.assertEqual(len(evidence.ubsan_findings), 2)


class LibraryPathPinCannotBeLifted(unittest.TestCase):
    """A/E. The pin was applied BEFORE `extra`, so `extra` won last-wins."""

    def test_an_extra_env_naming_the_variable_is_refused(self):
        with self.assertRaises(t0.ExecutionError):
            t0._launch_env(BIN_DIR, (), {"LD_LIBRARY_PATH": f"{PROD_TREE}/build/bin"})

    def test_a_poisoned_base_env_is_still_merely_defeated(self):
        """The control. `base_env` may carry it; the pin wins, as before."""
        env = dict(t0._launch_env(BIN_DIR, (("LD_LIBRARY_PATH", f"{PROD_TREE}/build/bin"),),
                                  {"OMP_NUM_THREADS": "96"}))
        self.assertEqual(env["LD_LIBRARY_PATH"], BIN_DIR)
        self.assertEqual(env["OMP_NUM_THREADS"], "96")


class AFailedGenerationIsNotAMeasurement(unittest.TestCase):
    """E/F. Three segfaults read as `bitwise_stable`.

    Nothing consulted a generation's exit status, so the digest of a run that
    produced nothing was recorded as an output digest. Verified live against
    the module as delivered: three llama-cli captures with `exit_code=139` and
    empty stdout produced three identical digests and
    `DeterminismEvidence.measured_class() == 'bitwise_stable'` — the strongest
    class there is, for a candidate that cannot run.
    """

    def _crashing(self, plan, seeded, count):
        invocation = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env,
            seed=plan.generation.seed if seeded else None)
        return [capture(list(invocation.argv), stdout="", stderr="Segmentation fault",
                        exit_code=139) for _ in range(count)]

    def test_crashed_repeats_contribute_no_digest(self):
        plan = execution_plan(generation=generation_plan(), determinism_runs=3)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner(self._crashing(plan, True, 3)),
            claim=FakeClaim())
        collected = t0._Collected()
        evidence = provider.collect_determinism(collected)
        self.assertEqual(evidence.candidate_output_digests, ())
        self.assertNotEqual(evidence.measured_class(), "bitwise_stable")
        self.assertEqual(len(collected.notes), 3)

    def test_the_real_determinism_gate_fails_on_it(self):
        """The bite through the CONSUMER, not through the field."""
        plan = execution_plan(generation=generation_plan(), determinism_runs=3)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner(self._crashing(plan, True, 3)),
            claim=FakeClaim())
        evidence = provider.collect_determinism(t0._Collected())
        request = evaluation_request(anchor=api.AnchorIdentity(
            source_commit=COMMIT_ANCHOR, binary_sha256=SHA_B, linkage_sha256=SHA_C),
            determinism_class="bitwise_stable", repeats=3)
        gate, _ = correctness.check_determinism_class(request, evidence, t0_policy())
        self.assertEqual(gate.check.outcome, schemas.FAIL, gate.check.reasons)

    def test_a_crashed_coherence_generation_records_no_output(self):
        plan = execution_plan(generation=generation_plan())
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner(self._crashing(plan, False, 1)),
            claim=FakeClaim(),
            anchor_capture=anchor_capture(output_digests=(SHA_D,), output_lengths=(64,),
                                          determinism_class="bitwise_stable"))
        collected = t0._Collected()
        evidence = provider.collect_coherence(collected)
        self.assertIsNone(evidence.candidate_output_sha256)
        self.assertIsNone(evidence.token_agreement_ratio,
                          "a ratio computed from a crashed run compares nothing")
        self.assertTrue(any("recorded as ABSENT" in note for note in collected.notes))

    def test_a_partial_output_from_a_nonzero_exit_is_not_compared(self):
        """The subtler half: a run that emitted SOMETHING and then died."""
        plan = execution_plan(generation=generation_plan())
        invocation = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env)
        truncated = capture(list(invocation.argv), stdout="The capital of Fra", exit_code=1)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([truncated]), claim=FakeClaim())
        evidence = provider.collect_coherence(t0._Collected())
        self.assertIsNone(evidence.candidate_output_sha256)

    def test_a_healthy_generation_is_still_measured(self):
        """The control."""
        plan = execution_plan(generation=generation_plan(), determinism_runs=2)
        invocation = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env, seed=plan.generation.seed)
        good = [capture(list(invocation.argv), stdout="Paris.\n") for _ in range(2)]
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner(good), claim=FakeClaim())
        evidence = provider.collect_determinism(t0._Collected())
        self.assertEqual(len(evidence.candidate_output_digests), 2)
        self.assertEqual(evidence.measured_class(), "bitwise_stable")

    def test_a_crashed_anchor_does_not_certify_itself_bitwise_stable(self):
        plan = execution_plan(
            generation=generation_plan(),
            anchor=t0.AnchorBuild(worktree=PROD_TREE, source_commit=COMMIT_ANCHOR,
                                  binary=f"{PROD_TREE}/build/bin/llama-cli",
                                  library_path=f"{PROD_TREE}/build/bin"))
        captures = []
        for seed in (1, 2):
            invocation = t0.build_generation_invocation(
                binary=plan.anchor.binary, library_path=plan.anchor.library_path,
                plan=plan.generation, base_env=plan.base_env, seed=seed)
            captures.append(capture(list(invocation.argv), stdout="", exit_code=139))
        link = t0.build_linkage_invocation(
            bash=plan.tools.bash, script=plan.tools.verify_ggml_linkage_sh,
            binary=plan.anchor.binary, expected_root=plan.anchor.library_path,
            library_path=plan.anchor.library_path, base_env=plan.base_env)
        captures.append(capture(list(link.argv), stdout=recorded("recorded_t0_linkage_pass.txt")))
        with unittest.mock.patch.object(t0, "sha256_file", return_value=SHA_B):
            result = t0.capture_anchor(plan=plan, runner=t0.RecordedProcessRunner(captures),
                                       claim=FakeClaim(), generation_seeds=(1, 2))
        self.assertEqual(result.determinism_class, "not_measured")
        self.assertEqual(result.output_digests, ())
        self.assertEqual(len(result.notes), 2)

    def test_identity_only_capture_hashes_the_named_tool_without_generation(self):
        plan = execution_plan()
        anchor = t0.AnchorBuild(
            worktree=PROD_TREE, source_commit=COMMIT_ANCHOR,
            binary=f"{PROD_TREE}/build/bin/llama-bench",
            library_path=f"{PROD_TREE}/build/bin")
        invocation = t0.build_linkage_invocation(
            bash=plan.tools.bash, script=plan.tools.verify_ggml_linkage_sh,
            binary=anchor.binary, expected_root=anchor.library_path,
            library_path=anchor.library_path, base_env=())
        runner = t0.RecordedProcessRunner([
            capture(list(invocation.argv), stdout=recorded("recorded_t0_linkage_pass.txt"))
        ])
        with unittest.mock.patch.object(t0, "sha256_file", return_value=SHA_B):
            result = t0.capture_anchor_identity(
                anchor=anchor, tools=plan.tools, runner=runner)
        self.assertEqual(result.binary_sha256, SHA_B)
        self.assertEqual(result.source_commit, COMMIT_ANCHOR)
        self.assertEqual(result.output_digests, ())
        self.assertEqual(result.determinism_class, "not_measured")
        self.assertEqual(len(result.capture_refs), 1)

    def test_anchor_toolchain_is_measured_from_its_own_cmake_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory, "build")
            binary = build / "bin" / "llama-cli"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"anchor")
            cmake = build / "CMakeFiles" / "3.31.6" / "CMakeCXXCompiler.cmake"
            cmake.parent.mkdir(parents=True)
            cmake.write_text(
                'set(CMAKE_CXX_COMPILER_ID "GNU")\n'
                'set(CMAKE_CXX_COMPILER_VERSION "15.2.0")\n',
                encoding="utf-8")
            self.assertEqual(
                t0._measure_cmake_toolchain(str(binary)),
                ("CXX GNU", "15.2.0"))

    def test_half_an_anchor_toolchain_attestation_is_refused(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            t0._complete_anchor_toolchain(
                f"{PROD_TREE}/build/bin/llama-cli", "CXX GNU", None)


class ClaimsAreCheckedAgainstTheRealImplementation(unittest.TestCase):
    """C. The documented wiring raised `TypeError` before any guard could run.

    `require_claim` demanded `claim_id`/`is_held`/`describe` by name.
    `execution/cpu_region_claim.CpuRegionClaim` — the only CPU region claim in
    this tree, and the one the module's own docstring tells tomorrow's session
    to pass — exposes `claim_id`, `held`, `verify_held()` and `covers()`, and
    neither `is_held` nor `describe`.
    """

    class RegionClaimShaped:
        """`CpuRegionClaim`'s members, no more. Not a HeldClaim by name."""

        def __init__(self, held=True, covered="0-95"):
            self._held, self._covered = held, covered

        @property
        def claim_id(self):
            return "akclaim-cpu-region-1"

        @property
        def held(self):
            return self._held

        def verify_held(self):
            return schemas.Check(schemas.PASS if self._held else schemas.FAIL, ())

        def covers(self, cpu_list, sibling_map=None):
            return cpu_list == self._covered

    def test_the_real_cpu_region_claim_shape_is_accepted(self):
        self.assertFalse(hasattr(cpu_region_claim.CpuRegionClaim, "is_held"),
                         "precondition: the real claim has no is_held()")
        self.assertFalse(hasattr(cpu_region_claim.CpuRegionClaim, "describe"))
        for member in ("claim_id", "held", "verify_held", "covers"):
            self.assertTrue(hasattr(cpu_region_claim.CpuRegionClaim, member))
        self.assertEqual(
            t0.require_claim(self.RegionClaimShaped(), what="the op suite"),
            "akclaim-cpu-region-1")

    def test_a_region_claim_that_is_not_held_still_refuses(self):
        with self.assertRaises(t0.ClaimNotHeld):
            t0.require_claim(self.RegionClaimShaped(held=False), what="the op suite")

    def test_a_claim_that_does_not_cover_the_pinned_cpus_refuses(self):
        """A held claim over a smaller region does not authorise a wider run.

        The argv carries `recipes.CANONICAL_PREFIX`, which pins the whole
        machine; a claim over cores 0-7 answered `is_held()` identically.
        """
        with self.assertRaises(t0.ClaimNotHeld) as ctx:
            t0.require_claim(self.RegionClaimShaped(covered="0-7"),
                             what="the op suite", cpu_list=t0._canonical_cpu_list())
        self.assertIn("does not cover", str(ctx.exception))

    def test_the_covering_claim_is_accepted(self):
        """The control."""
        self.assertEqual(
            t0.require_claim(self.RegionClaimShaped(covered=t0._canonical_cpu_list()),
                             what="the op suite", cpu_list=t0._canonical_cpu_list()),
            "akclaim-cpu-region-1")

    def test_the_op_suite_passes_the_pinned_footprint_through(self):
        plan = execution_plan()
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]),
            claim=self.RegionClaimShaped(covered="0-7"))
        with self.assertRaises(t0.ClaimNotHeld):
            provider.collect_op_suite(t0._Collected())

    def test_the_pinned_footprint_is_read_off_the_ratified_prefix(self):
        prefix = list(recipes.CANONICAL_PREFIX)
        self.assertEqual(t0._canonical_cpu_list(), prefix[prefix.index("-c") + 1])


class NoOrphanOnAnUnexpectedFailure(unittest.TestCase):
    """B. Only `TimeoutExpired` was handled; every other exit left the child.

    Verified live against the module as delivered: a `MemoryError` out of
    `communicate` propagated with `/bin/sleep 60` still running, and `run()` is
    the only frame that ever held that pid. On this host the escapee would be a
    `taskset -c 0-95` llama-cli with the full OMP stack, findable afterwards
    only by the name-pattern operation INC-20260731 forbids.
    """

    def test_a_non_timeout_failure_still_kills_the_child_it_launched(self):
        spawned = []
        real = subprocess.Popen.communicate

        def explode(self_, *args, **kwargs):
            spawned.append(self_.pid)
            raise MemoryError("simulated failure inside communicate")

        subprocess.Popen.communicate = explode
        try:
            with self.assertRaises(MemoryError):
                t0.SubprocessRunner(term_grace_s=2.0, kill_grace_s=2.0).run(
                    ["/bin/sleep", "60"], env={"PATH": "/usr/bin:/bin"}, cwd="/tmp",
                    timeout_s=30)
        finally:
            subprocess.Popen.communicate = real
        self.assertEqual(len(spawned), 1)
        pid = spawned[0]
        deadline = time.monotonic() + 10.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.05)
        if alive:                                             # pragma: no cover
            os.kill(pid, signal.SIGKILL)
        self.assertFalse(alive, f"pid {pid} escaped run()'s failure path")


class BoundaryShapesAreReadTheSameWayAsTheOpSuite(unittest.TestCase):
    """D. One parser, one tool, two opposite readings.

    `collect_op_suite` parses stdout and calls `reconcile()`; the boundary path
    parsed the merged stream and skipped the cross-check — on the ONE run a
    `-p` shape filter can silently empty. It also counted a `not supported`
    case as a FAILURE, which `collect_op_suite` explicitly does not, so an
    honest CPU run that legitimately declines a shape FAILed the gate.
    """

    DECLINED = (
        "Testing 1 devices\n\nBackend 1/1: CPU\n"
        "  MUL_MAT(type_a=f32,type_b=f16,m=16,n=1,k=256): not supported [CPU]\n"
        "  MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=256): OK\n"
        "  1/1 tests passed\n  Backend CPU: OK\n\n1/1 backends passed\nOK\n")

    def _provider(self, text):
        holdout = t0.HoldoutPlan(unseen_case_filter="m=16,n=1,k=256",
                                 boundary_case_filter="m=1,n=1,k=1",
                                 selection_rule_id="rule-1", selection_seed="seed-1",
                                 visible_to_planner=False)
        plan = execution_plan(holdout=holdout)
        captures = []
        for case_filter in (holdout.unseen_case_filter, holdout.boundary_case_filter):
            invocation = t0.build_backend_ops_invocation(
                binary=plan.candidate.test_backend_ops,
                library_path=plan.candidate.library_path,
                backend_filter=plan.op_suite.backend_filter, ops=plan.op_suite.ops,
                base_env=plan.base_env, params_filter=case_filter)
            captures.append(capture(list(invocation.argv), stdout=text))
        return t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner(captures), claim=FakeClaim())

    def test_a_declined_shape_is_not_reported_as_a_failure(self):
        collected = t0._Collected()
        evidence = self._provider(self.DECLINED).collect_boundary_shapes(collected)
        self.assertEqual(evidence.failures, ())
        self.assertEqual(len(evidence.unseen_shapes), 1)
        self.assertTrue(any("not supported" in note for note in collected.notes))

    def test_a_parser_tool_disagreement_is_refused_here_too(self):
        text = self.DECLINED.replace("1/1 tests passed", "9/9 tests passed")
        with self.assertRaises(t0.OutputParseError):
            self._provider(text).collect_boundary_shapes(t0._Collected())

    def test_a_real_failing_shape_is_still_a_failure(self):
        """The control: the guard must not be satisfiable by dropping cases."""
        text = self.DECLINED.replace(
            "MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=256): OK",
            "MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=256): FAIL"
        ).replace("1/1 tests passed", "0/1 tests passed")
        evidence = self._provider(text).collect_boundary_shapes(t0._Collected())
        self.assertEqual(len(evidence.failures), 2)


class DeliveredUnitsDoNotDependOnTheSink(unittest.TestCase):
    """D. A storage choice turned a control-3 detector off.

    `_delivered_units` returned `None` unless the sink was a
    `MemoryCaptureSink` — the sink whose own docstring says it "writes
    nothing". Every durable sink, which is what a real campaign installs,
    therefore reported `delivered_units_candidate=0`: a manufactured
    "the candidate delivered less work than the anchor" FAIL.
    """

    PERF = "llama_perf_context_print:        eval time =  1000.00 ms /   256 runs\n"

    class DurableSink:
        def __init__(self):
            self.written = []

        def store(self, capture_):
            ref = t0.capture_ref(capture_)
            self.written.append(ref)
            return ref

    def _collect(self, sink):
        plan = execution_plan(generation=generation_plan(), oracle_ids=("oracle-1",))
        invocation = t0.build_generation_invocation(
            binary=plan.candidate.binary, library_path=plan.candidate.library_path,
            plan=plan.generation, base_env=plan.base_env)
        runner = t0.RecordedProcessRunner(
            [capture(list(invocation.argv), stdout="Paris.\n" + self.PERF)])
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=runner, claim=FakeClaim(),
                                                 sink=sink)
        collected = t0._Collected()
        provider.collect_coherence(collected)
        return provider, collected

    def test_a_durable_sink_still_reads_the_delivered_count(self):
        sink = self.DurableSink()
        provider, collected = self._collect(sink)
        self.assertTrue(sink.written, "precondition: the durable sink was used")
        self.assertEqual(provider._delivered_units(collected), 256)
        evidence = provider.collect_anti_reward_hacking(
            provider._delivered_units(collected), collected)
        self.assertEqual(evidence.delivered_units_candidate, 256)

    def test_the_memory_sink_reads_the_same_number(self):
        """The control: the path that already worked keeps working."""
        provider, collected = self._collect(None)
        self.assertEqual(provider._delivered_units(collected), 256)

    def test_an_unread_count_says_so_in_the_record(self):
        """0 is the only sayable value; the note is what makes it honest."""
        plan = execution_plan(oracle_ids=("oracle-1",))
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), claim=FakeClaim())
        collected = t0._Collected()
        evidence = provider.collect_anti_reward_hacking(None, collected)
        self.assertEqual(evidence.delivered_units_candidate, 0)
        self.assertTrue(any("UNREAD" in note for note in collected.notes))
        self.assertTrue(any("delivered_units_candidate" in item
                            for item in t0.SCHEMA_FOLLOWUPS))


class AssertedSurfacesAreRecordedAsGaps(unittest.TestCase):
    """E/denial 6. Remaining clean-shaped fields nothing measured.

    `race_findings`, `leaked_resources`, `rollback_tested`,
    `candidate_output_used_as_oracle` remain constants. Environment and timing
    findings were removed from this list when RVP-C6-9 installed real versioned
    source-diff detectors; their absence now reads UNKNOWN.
    """

    def test_the_seam_names_every_asserted_field(self):
        blob = "\n".join(t0.SEAMS)
        for field_name in ("race_findings", "leaked_resources", "rollback_tested",
                           "candidate_output_used_as_oracle"):
            self.assertIn(field_name, blob, f"{field_name} is asserted, not measured")

    def test_the_fields_really_are_constants_here(self):
        """If any of them becomes a measurement, the SEAM must be rewritten."""
        source = Path(t0.__file__).read_text(encoding="utf-8")
        for literal in ("race_findings=()", "leaked_resources=()",
                        "rollback_tested=False", "candidate_output_used_as_oracle=False"):
            self.assertIn(literal, source)

    def test_the_two_source_detectors_populate_versioned_receipts(self):
        provider = t0.ExecutedT0EvidenceProvider(
            plan=execution_plan(candidate_diff_text=(
                "diff --git a/k.hip b/k.hip\n--- a/k.hip\n+++ b/k.hip\n"
                "@@ -1 +1 @@\n-int x;\n+if (getenv(\"BENCH\")) fast();\n")),
            runner=t0.RecordedProcessRunner([]))
        evidence = provider.collect_anti_reward_hacking(1, t0._Collected())
        self.assertTrue(evidence.environment_probe_findings)
        self.assertRegex(evidence.environment_probe_detector_id, r"/v\d+$")
        self.assertRegex(evidence.timing_dependent_branch_detector_id, r"/v\d+$")


class ThePlanCarriesTheDerivationAndTypeChecksIt(unittest.TestCase):
    """§6.1 item 4. `_change_surface`'s pass-through was UNREACHABLE.

    The method reads `getattr(self._plan, "change_surface", None)` and
    `T0ExecutionPlan` had no such field, so no caller could supply a derivation
    and every candidate got the all-`None` surface — which four gates read as
    COULD_NOT_CHECK. These tests fail on the pre-fix module: the first with
    `TypeError: unexpected keyword argument 'change_surface'`.
    """

    @staticmethod
    def _surface(**overrides):
        kwargs = dict(
            derived_touches_memory=True, derived_touches_threading=None,
            derived_touches_dispatch=True, derived_touches_persistent_state=None,
            derived_ops=("MUL_MAT_ID",), derived_files=("ggml/src/ggml-cpu/ggml-cpu.c",),
            declared_touches_memory=None, declared_touches_threading=None,
            declared_ops=(), touches_shared_core_header=False,
            derivation_ref="ref://test-derivation")
        kwargs.update(overrides)
        return correctness.ChangeSurface(**kwargs)

    def test_a_supplied_derivation_reaches_the_provider(self):
        supplied = self._surface()
        plan = execution_plan(change_surface=supplied)
        provider = t0.ExecutedT0EvidenceProvider(plan=plan, runner=t0.RecordedProcessRunner([]))
        self.assertIs(provider._change_surface(), supplied)

    def test_without_a_derivation_every_behavioural_flag_is_undetermined(self):
        """The COULD_NOT_CHECK default is BY DESIGN and must not change."""
        provider = t0.ExecutedT0EvidenceProvider(plan=execution_plan(),
                                                 runner=t0.RecordedProcessRunner([]))
        derived = provider._change_surface()
        for name in ("derived_touches_memory", "derived_touches_threading",
                     "derived_touches_dispatch", "derived_touches_persistent_state"):
            self.assertIsNone(getattr(derived, name), name)
        self.assertIsNone(derived.sanitizers_mandatory)

    def test_a_surface_derivation_object_is_not_a_change_surface(self):
        """The projection is `chain.change_surface_from`, not an assignment."""
        with self.assertRaises(TypeError) as ctx:
            execution_plan(change_surface={"derived_touches_memory": True})
        self.assertIn("chain.py", str(ctx.exception))

    def test_the_integrity_build_provenance_is_refused_where_correctness_is_required(self):
        """Seam 1, at the door. The wrong record used to fail SILENTLY.

        `integrity.BuildProvenance` has no `build_log_ref` and no `compiler_id`;
        `collect_static_analysis` read both with `getattr` defaults, so the whole
        static-analysis surface disappeared into a COULD_NOT_CHECK whose stated
        reason named the anchor.
        """
        wrong = integrity.BuildProvenance(
            candidate_id="akc-0001", snapshot_sha256=SHA_A,
            source_root=WORKTREE, actor_worktree=WORKTREE, build_dir=BUILD_DIR,
            build_dir_created_for_this_build=True,
            build_dir_pre_build_digest=integrity.EMPTY_TREE_SHA256,
            toolchain="cmake + GNU make", compiler="CXX GNU 15.2.0",
            command=f"cmake --build {BUILD_DIR}",
            build_log_path=f"{BUILD_DIR}/build.log",
            build_log_sha256=SHA_B, output_binary_sha256=SHA_D,
            incremental_output_binary_sha256=None,
            production_tree_paths=tuple(correctness.PRODUCTION_TREE_ROOTS))
        self.assertFalse(hasattr(wrong, "build_log_ref"))
        self.assertFalse(hasattr(wrong, "compiler_id"))
        with self.assertRaises(TypeError) as ctx:
            execution_plan(build=wrong)
        self.assertIn("BuildProvenance", str(ctx.exception))
        self.assertIn("fails silently", str(ctx.exception))

    def test_the_correctness_build_provenance_is_the_compliant_control(self):
        """The guard must not forbid the idiom it exists to require."""
        right = correctness.BuildProvenance(
            built_from_snapshot_sha256=SHA_A, build_dir=BUILD_DIR,
            build_dir_was_fresh=True, incremental_objects_present=False,
            compiler_id="CXX GNU", compiler_version="15.2.0",
            build_log_ref=f"file://{BUILD_DIR}/build.log#sha256={SHA_B}",
            production_tree_paths_touched=(), output_binary_sha256=SHA_D,
            produced_by="evaluator")
        plan = execution_plan(build=right, change_surface=self._surface())
        self.assertIs(plan.build, right)


class TheStaticAnalysisSurfaceNamesTheCandidatesOwnToolchain(unittest.TestCase):
    """§6.1 item 3. `compiler_id` was `getattr(build, "compiler_id", "unknown")`.

    `"unknown"` is not an absent value: it is a compiler identity no anchor can
    equal, so the gate FAILed with "the candidate was built with unknown unknown
    but the anchor with CXX GNU 15.2.0" — a toolchain confound reported about a
    candidate whose toolchain was never read.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = Path(self._tmp.name) / "build.log"
        self.log.write_text(
            "ggml/src/ggml-cpu/ggml-cpu.c:12:5: warning: unused variable 'x'\n",
            encoding="utf-8")
        self.build = correctness.BuildProvenance(
            built_from_snapshot_sha256=SHA_A, build_dir=BUILD_DIR,
            build_dir_was_fresh=True, incremental_objects_present=False,
            compiler_id="CXX GNU", compiler_version="15.2.0",
            build_log_ref=f"file://{self.log}", production_tree_paths_touched=(),
            output_binary_sha256=SHA_D, produced_by="evaluator")

    def _evidence(self, anchor):
        plan = execution_plan(build=self.build)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), anchor_capture=anchor)
        return provider.collect_static_analysis(t0._Collected())

    def test_the_anchor_toolchain_is_what_makes_the_surface_produced_at_all(self):
        """No compiler_id/version on the capture -> no evidence -> COULD_NOT_CHECK."""
        self.assertIsNone(self._evidence(anchor_capture()))

    def test_with_the_anchor_toolchain_the_gate_is_real(self):
        anchor = anchor_capture(compiler_id="CXX GNU", compiler_version="15.2.0",
                                warning_count=1)
        evidence = self._evidence(anchor)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.compiler_id, "CXX GNU")
        self.assertEqual(evidence.compiler_version, "15.2.0")
        self.assertNotIn("unknown", (evidence.compiler_id, evidence.compiler_version))
        gate = correctness.check_static_and_compile(evidence, anchor.identity())
        self.assertEqual(gate.check.outcome, schemas.PASS, gate.check.reasons)

    def test_a_real_toolchain_difference_still_fails(self):
        """The compliant path passes and the confound it exists to catch still bites."""
        anchor = anchor_capture(compiler_id="CXX Clang", compiler_version="19.1.0",
                                warning_count=1)
        gate = correctness.check_static_and_compile(self._evidence(anchor),
                                                    anchor.identity())
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(any("toolchain comparison" in r for r in gate.check.reasons))

    def test_an_absent_anchor_warning_count_leaves_the_delta_unknowable(self):
        anchor = anchor_capture(compiler_id="CXX GNU", compiler_version="15.2.0")
        gate = correctness.check_static_and_compile(self._evidence(anchor),
                                                    anchor.identity())
        self.assertEqual(gate.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertTrue(any("anchor warning count" in r for r in gate.check.reasons))

    def test_a_receipted_fatal_warnings_build_needs_no_anchor_warning_count(self):
        self.build = correctness.BuildProvenance(
            built_from_snapshot_sha256=SHA_A, build_dir=BUILD_DIR,
            build_dir_was_fresh=True, incremental_objects_present=False,
            compiler_id="CXX GNU", compiler_version="15.2.0",
            build_log_ref=f"file://{self.log}", production_tree_paths_touched=(),
            output_binary_sha256=SHA_D, produced_by="evaluator",
            warnings_as_errors=True)
        anchor = anchor_capture(compiler_id="CXX GNU", compiler_version="15.2.0")
        evidence = self._evidence(anchor)
        self.assertTrue(evidence.warnings_as_errors)
        gate = correctness.check_static_and_compile(evidence, anchor.identity())
        self.assertEqual(gate.check.outcome, schemas.PASS, gate.check.reasons)


class TheStateSafetyGateCannotPass(unittest.TestCase):
    """The impossibility, proved by exhaustion rather than described in a runbook.

    `collect_state_safety` hardcodes `rollback_tested=False` and
    `check_state_rollback_teardown_race` FAILs on it unconditionally, so **no
    state-safety MEASUREMENT can PASS**. Every PASS this surface can produce
    comes from the probe being OFF and the derivation declaring the surface not
    applicable — a PASS granted by the change surface, not by anything observed.

    This test is the tripwire: it fails the day a real rollback probe exists,
    which is when the constant, the note and the README paragraph all have to go.
    """

    @staticmethod
    def _surface(**overrides):
        kwargs = dict(
            derived_touches_memory=None, derived_touches_threading=None,
            derived_touches_dispatch=None, derived_touches_persistent_state=None,
            derived_ops=(), derived_files=(), declared_touches_memory=None,
            declared_touches_threading=None, declared_ops=(),
            touches_shared_core_header=False, derivation_ref="ref://test-derivation")
        kwargs.update(overrides)
        return correctness.ChangeSurface(**kwargs)

    def _outcomes(self, probe):
        plan = execution_plan(state_safety_probe=probe)
        provider = t0.ExecutedT0EvidenceProvider(
            plan=plan, runner=t0.RecordedProcessRunner([]), claim=FakeClaim())
        collected = t0._Collected()
        evidence = provider.collect_state_safety(collected)
        outcomes = set()
        for threading in (True, False, None):
            for persistent in (True, False, None):
                gate = correctness.check_state_rollback_teardown_race(
                    evidence, self._surface(derived_touches_threading=threading,
                                            derived_touches_persistent_state=persistent))
                outcomes.add(gate.check.outcome)
        return outcomes, collected.notes

    def test_the_probe_on_is_a_guaranteed_fail_for_every_surface(self):
        """Nine derived surfaces, one outcome. The gate says nothing about any of them."""
        outcomes, _notes = self._outcomes(True)
        self.assertEqual(outcomes, {schemas.FAIL})

    def test_every_pass_this_surface_can_produce_comes_from_the_probe_being_off(self):
        """A PASS by non-applicability is not a measurement that passed.

        With the probe off there is no evidence at all, so the gate answers from
        `ChangeSurface` alone — and it is the derivation, not this collector,
        that granted the PASS. That is the honest reading; the point of the test
        is that it is the ONLY reading under which this surface ever passes.
        """
        outcomes, _notes = self._outcomes(False)
        self.assertEqual(outcomes, {schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK})
        provider = t0.ExecutedT0EvidenceProvider(
            plan=execution_plan(state_safety_probe=False),
            runner=t0.RecordedProcessRunner([]), claim=FakeClaim())
        self.assertIsNone(provider.collect_state_safety(t0._Collected()))

    def test_the_impossibility_is_recorded_on_the_collection_either_way(self):
        for probe in (True, False):
            _outcomes, notes = self._outcomes(probe)
            self.assertTrue(any(t0.STATE_SAFETY_CANNOT_PASS in note for note in notes),
                            f"state_safety_probe={probe} recorded no note")

    def test_the_constant_says_why_and_names_the_one_real_observation(self):
        self.assertIn("rollback probe", t0.STATE_SAFETY_CANNOT_PASS)
        self.assertIn("orphan_processes", t0.STATE_SAFETY_CANNOT_PASS)


if __name__ == "__main__":                                        # pragma: no cover
    unittest.main()
