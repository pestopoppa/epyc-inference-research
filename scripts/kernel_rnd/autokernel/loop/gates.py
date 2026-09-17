#!/usr/bin/env python3
"""Correctness gates. Cheap, in order, and every failure returns a reason.

Ordering is the design. The build is the most expensive step, so anything that can
refuse a patch before it runs, does. What survives to the benchmark has compiled and
passed the op oracle, so GPU time is spent only on candidates that could plausibly
be kept.

Every gate returns a `Verdict` carrying the toolchain's own message. That message
goes back to the planner verbatim: the defect this loop replaces filtered refusal
reasons on a status string the controller never wrote, so 22 of 23 authoring failures
returned nothing and the planner re-derived rejected work blind.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Callable
import subprocess

from .. import schemas
from ..evaluator import correctness
from . import bench, census, residency

#: One op suite, on the backend under test. 53 seconds measured, and it is the gate
#: that decides whether a candidate is CORRECT -- everything downstream assumes it.
CORRECTNESS_TIMEOUT_S = 1800
BUILD_TIMEOUT_S = 7200

#: Per-ITERATION builds: candidate lanes and the anchor guard's fresh build. The
#: bench binary and the op oracle are all a measurement needs, and at hundreds of
#: iterations per run every extra link is paid for by nobody.
DEFAULT_TARGETS = ("llama-bench", "test-backend-ops")
#: Per-KEEP promotion builds (`pool.promote_anchor`). Every `anchor-gen-NNN` before
#: R22-7 was bench-only; the operator's ruling (2026-09-01, verbatim): "The whole
#: point of a champion is that it needs to be extremely easy to promote into
#: production… If we're not compiling llama-servers that's a problem." A superset of
#: DEFAULT_TARGETS by construction, so the promoted artifact can never lack a binary
#: the loop itself measured with.
PROMOTION_TARGETS = (*DEFAULT_TARGETS, "llama-cli", "llama-server")


@dataclass(frozen=True)
class Verdict:
    """Passed, or refused with the reason the actor needs to fix it."""
    gate: str
    passed: bool
    reason: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {"gate": self.gate, "passed": self.passed,
                "reason": self.reason or None, "detail": self.detail[:2000] or None}


def compiles(source_root: Path, build_dir: Path, *, cmake_defines: tuple,
             jobs: int, cpu_list: str | None, targets: tuple = DEFAULT_TARGETS,
             cmake: str = "cmake") -> Verdict:
    """Configure and build. A compile failure is cheap, automatic planner feedback."""
    prefix = ("taskset", "-c", cpu_list) if cpu_list else ()
    configure = [*prefix, cmake, "-S", str(source_root), "-B", str(build_dir),
                 "-DCMAKE_BUILD_TYPE=Release",
                 *[f"-D{name}={value}" for name, value in cmake_defines]]
    done = subprocess.run(configure, capture_output=True, text=True,
                          timeout=BUILD_TIMEOUT_S)
    if done.returncode != 0:
        return Verdict("configure", False, "cmake configure failed", done.stderr[-2000:])

    build = [*prefix, cmake, "--build", str(build_dir), "-j", str(jobs)]
    for target in targets:
        build += ["--target", target]
    done = subprocess.run(build, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    if done.returncode != 0:
        return Verdict("compile", False, "build failed", done.stderr[-2000:])
    # Exit code alone is not enough: a pipe can lose the compiler's status, and a
    # build that printed `Error` while exiting 0 is the case that hides.
    haystack = (done.stdout + done.stderr).lower()
    if "error 2" in haystack or "*** error" in haystack:
        return Verdict("compile", False,
                       "build log reports an error despite exit 0",
                       (done.stdout + done.stderr)[-2000:])
    return Verdict("compile", True)


#: Proof the suite actually EXECUTED. `test-backend-ops` prints this summary whether
#: it passes or fails, so its ABSENCE means the run never happened.
RAN_MARKER = "backends passed"
TEST_COUNT = re.compile(r"(\d+)/(\d+) tests passed")
REFERENCE_SUITE_SEED = 71  # fixed case population; not a numerical acceptance threshold
MAX_REFERENCE_OUTPUT_BYTES = 2 * 1024 * 1024


def _gdn_hunks_confined(source_text: str | None, patch_text: str | None) -> bool:
    """The shared ops.cpp file is GDN-only only when every new hunk is in its block."""
    if not source_text or not patch_text:
        return False
    lines = source_text.splitlines()
    markers = [i + 1 for i, line in enumerate(lines)
               if line.startswith("// ggml_compute_forward_")]
    starts = [i for i in markers if lines[i - 1] == "// ggml_compute_forward_gated_delta_net"]
    if len(starts) != 1:
        return False
    start = starts[0]
    end = next((i - 1 for i in markers if i > start), None)
    if end is None:
        return False
    hunks = re.findall(r"(?m)^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", patch_text)
    return bool(hunks) and all(start <= int(line) and
                               int(line) + max(int(count or 1), 1) - 1 <= end
                               for line, count in hunks)


def _iqk_moe_rows_hunks_confined(source_text: str | None,
                                  patch_text: str | None,
                                  target_symbol: str = "iqk_mul_mat_moe_rows") -> bool:
    """Only the named real exported helper body, never siblings or stubs."""
    if not source_text or not patch_text:
        return False
    lines = source_text.splitlines()
    starts = [i + 1 for i, line in enumerate(lines)
              if line.startswith(f'extern "C" IQK_API bool {target_symbol}(')]
    next_symbol = ("iqk_moe_fused_up_gate" if target_symbol == "iqk_mul_mat_moe_rows"
                   else "#if defined __x86_64__")
    ends = [i + 1 for i, line in enumerate(lines)
            if line.startswith('extern "C" IQK_API bool iqk_moe_fused_up_gate(')
            or (target_symbol == "iqk_moe_fused_up_gate" and
                line.startswith(next_symbol))]
    if not starts:
        return False
    ends = [end for end in ends if end > starts[0]]
    if not ends:
        return False
    body_start = next((i + 1 for i in range(starts[0] - 1, ends[0] - 1)
                       if lines[i].rstrip().endswith('{')), None)
    if body_start is None:
        return False
    hunks = re.findall(r"(?m)^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", patch_text)
    return bool(hunks) and all(body_start < int(line) and
                               int(line) + max(int(count or 1), 1) - 1 < ends[0]
                               for line, count in hunks)


def affected_op_scope(paths: tuple[str, ...], *, target_surface: str,
                      target_symbol: str, source_text: str | None = None,
                      patch_text: str | None = None) -> tuple[str, ...] | Verdict:
    """Resolve known changed-source routes; never inherit MUL_MAT by default.

    Paths are read from Git by the owner, not taken from the actor's response.
    Unknown/shared edits must acquire a native op map and reference before timing.
    """
    changed = set(paths)
    if not changed or len(changed) != len(paths) or target_surface not in changed:
        return Verdict("op_scope", False,
                       "actual changed paths are empty, repeated or omit the target surface")
    if changed <= {"ggml/src/ggml-cuda/gated_delta_net.cu",
                   "ggml/src/ggml-cuda/gated_delta_net.cuh"} and \
            "gated_delta_net" in target_symbol.lower():
        return ("GATED_DELTA_NET",)
    if changed == {"ggml/src/ggml-cpu/ops.cpp"} and \
            "gated_delta_net" in target_symbol.lower() and \
            _gdn_hunks_confined(source_text, patch_text):
        return ("GATED_DELTA_NET",)
    if changed == {"ggml/src/ggml-cuda/vecdotq.cuh"} and \
            target_symbol.startswith("vec_dot_"):
        return ("MUL_MAT", "MUL_MAT_ID")
    if changed == {"ggml/src/ggml-cuda/mmvq.cu"} and \
            (target_symbol.startswith("vec_dot_") or "mul_mat_vec" in target_symbol):
        return ("MUL_MAT", "MUL_MAT_ID")
    if changed == {"ggml/src/ggml-cuda/mmq.cu"} and \
            ("mul_mat_q" in target_symbol or "should_use_mmq" in target_symbol):
        return ("MUL_MAT", "MUL_MAT_ID")
    if changed == {"ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp"} and \
            target_symbol == "iqk_mul_mat_moe_rows" and \
            _iqk_moe_rows_hunks_confined(source_text, patch_text):
        return ("MUL_MAT_ID",)
    if changed == {"ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp"} and \
            target_symbol == "iqk_moe_fused_up_gate" and \
            _iqk_moe_rows_hunks_confined(source_text, patch_text, target_symbol):
        # The native GLU selector currently reports 0/0 CPU cases. The
        # independent fused graph fixture below exercises the actual GLU and
        # exact edited helper, after the nonempty MUL_MAT_ID host/op suite.
        return ("MUL_MAT_ID",)
    if any(path.startswith("ggml/src/ggml-cpu/iqk/") for path in changed):
        return Verdict("op_scope", False,
                       "CPU IQK source refused before build: the selected MUL_MAT/MUL_MAT_ID "
                       "case must prove the edited quant/function path executed with use_ref=false "
                       "and passed against the independent use_ref=true reference; the generic "
                       "per-type [iqk] ACTIVE marker does not identify the edited path or case")
    return Verdict("op_scope", False,
                   "affected native op/reference is unresolved for actual changed source; "
                   "MUL_MAT is not a universal correctness oracle")


def check_cpu_gdn_reference(build_dir: Path, source_root: Path, *,
                            resolved_recipe=None) -> Verdict:
    """Independent scalar fixture, after the native CPU suite's host/unit check."""
    from . import gdn_reference

    options = ({"launch_env": resolved_recipe.launch_env,
                "topology_prefix": tuple(resolved_recipe.topology_prefix)}
               if resolved_recipe is not None else {})
    result = gdn_reference.check_cpu_gdn(build_dir, source_root, **options)
    return Verdict("reference_comparison" if result.status == "wrong" else
                   "oracle_unavailable" if result.status == "unavailable" else
                   "reference_comparison", result.status == "pass",
                   result.reason, result.detail)


def check_cpu_iqk_reference(build_dir: Path, source_root: Path, *,
                            resolved_recipe, target_symbol: str) -> Verdict:
    """Run the exact helper's independent numerical and engagement witness."""
    from . import iqk_witness

    if target_symbol == "iqk_mul_mat_moe_rows":
        result = iqk_witness.check(build_dir, resolved_recipe=resolved_recipe,
                                   source_root=source_root)
    elif target_symbol == "iqk_moe_fused_up_gate":
        result = iqk_witness.check_fused(build_dir, resolved_recipe=resolved_recipe,
                                         source_root=source_root)
    else:
        return Verdict("oracle_unavailable", False,
                       "unsupported CPU IQK helper has no independent reference")
    return Verdict("reference_comparison" if result.status == "wrong" else
                   "oracle_unavailable" if result.status == "unavailable" else
                   "reference_comparison", result.status == "pass",
                   result.reason, result.detail)


def op_correctness(build_dir: Path, *, op: str = "MUL_MAT",
                   backend: str = "ROCm0", resolved_recipe=None,
                   require_reference: bool = False) -> Verdict:
    """`test-backend-ops` on the op the patch touches. The real correctness gate.

    THE DEFECT THIS SHAPE EXISTS TO PREVENT. An older binary did not accept
    `--suite-seed <n>`; blindly passing it produced usage text and fabricated
    "MUL_MAT failed on ROCm0" refusals. Seven of ten run-9 iterations died on
    that harness fault. The optional metric route now checks the selected
    binary's capability before passing the flags. The original route remains
    the default for older instruments and CPU checks.

    A non-zero exit is NOT sufficient evidence that a test failed: it is equally
    consistent with the tool refusing to run at all. So the pass/fail decision is made
    on POSITIVE evidence that the suite executed, and an oracle that could not run
    returns a distinct verdict that must never be read as "the patch is wrong".
    """
    binary = build_dir / "bin" / "test-backend-ops"
    if not binary.is_file():
        return Verdict("oracle_unavailable", False,
                       f"no test-backend-ops at {binary}")
    if require_reference and not re.fullmatch(r"ROCm[0-9]+", backend):
        return Verdict("oracle_unavailable", False,
                       "candidate-local CPU reference is not independent for a CPU source edit")
    argv = [str(binary), "test", "-o", op, "-b", backend, "-j", "1"]
    environment = residency.loader_env(binary)
    if resolved_recipe is not None:
        resolved_recipe.validate_launch(resolved_recipe.template, build_dir, resolved_recipe.port)
        expected = "CPU" if resolved_recipe.backend == "cpu" else resolved_recipe.template.device
        if backend != expected:
            raise ValueError("runtime oracle backend differs from the original recipe")
        # This is still the existing op oracle, not an exact-token serving proof.
        # Run it under the actual treatment's loader/env and CPU/NUMA prefix.
        argv = [*resolved_recipe.topology_prefix, *argv]
        environment = dict(resolved_recipe.launch_env)
    if require_reference:
        # An older test-backend-ops rejected --suite-seed and printed usage. The
        # selected binary, not the source tree or an anchor, must prove support.
        from ..execution import t0_provider
        try:
            help_run = subprocess.run([str(binary), "--help"], capture_output=True,
                                      text=True, timeout=30, env=environment)
            help_text = help_run.stdout + help_run.stderr
            if len(help_text.encode()) > 256 * 1024:
                raise ValueError("help output exceeds bound")
            capabilities = t0_provider.parse_backend_ops_help(help_text)
            capabilities.require(("--suite-seed", "--autokernel-properties"))
        except (OSError, ValueError, subprocess.TimeoutExpired,
                t0_provider.InstrumentCapabilityError) as exc:
            return Verdict("oracle_unavailable", False,
                           f"selected test-backend-ops lacks a reviewed metric receipt: {exc}")
        argv.extend(("--suite-seed", str(REFERENCE_SUITE_SEED),
                     "--autokernel-properties"))
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=CORRECTNESS_TIMEOUT_S,
                          env=environment)
    output = done.stdout + done.stderr
    if require_reference and len(output.encode()) > MAX_REFERENCE_OUTPUT_BYTES:
        return Verdict("oracle_unavailable", False,
                       "seeded reference suite output exceeds inspection bound")
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    block = re.search(
        rf"(?ms)^Backend \d+/\d+: {re.escape(backend)}\b(.*?)"
        rf"^  Backend {re.escape(backend)}: (OK|FAIL)\b", plain)
    counts = TEST_COUNT.findall(block.group(1)) if block else []
    if RAN_MARKER not in plain or not counts or not any(int(total) > 0 for _, total in counts):
        # Usage text, a missing backend, a loader failure -- anything that means the
        # suite did not execute. Blaming the patch for this is how a harness fault
        # becomes a fabricated scientific result.
        return Verdict("oracle_unavailable", False,
                       f"test-backend-ops did not prove a nonempty {backend} op suite; "
                       "this is a harness fault, NOT evidence about the patch",
                       output[-2000:])
    if block.group(2) == "FAIL":
        return Verdict("correctness", False, f"{op} failed on {backend}",
                       done.stdout[-2000:] + done.stderr[-1000:])
    if done.returncode != 0 or any(int(passed) != int(total) for passed, total in counts):
        return Verdict("oracle_unavailable", False,
                       f"test-backend-ops gave contradictory {backend} status and exit/tally; "
                       "this is a harness fault, NOT evidence about the patch",
                       output[-2000:])
    if require_reference:
        from ..execution import t0_provider
        try:
            parsed = t0_provider.parse_backend_ops_console(output)
            parsed.reconcile()
        except (ValueError, t0_provider.OutputParseError) as exc:
            return Verdict("oracle_unavailable", False,
                           f"seeded reference suite is unreadable: {exc}")
        selected = [case for frame in parsed.backends
                    if frame.name == backend and not frame.skipped
                    for case in frame.cases
                    if case.op == op and case.status != "not_supported"]
        if (not selected or any(not case.passed or case.reference is None
                                for case in selected)):
            return Verdict("oracle_unavailable", False,
                           f"{op} did not emit a reference metric for every selected {backend} case")
        if any(case.reference.oracle_id != "ggml_cpu_reference/v1" for case in selected):
            return Verdict("oracle_unavailable", False,
                           f"{op} emitted an unrecognized reference oracle")
        if any(case.reference.observed > case.reference.tolerance for case in selected):
            return Verdict("correctness", False,
                           f"{op} exceeds a declared native reference tolerance")
        ratios = [(case.reference.observed / case.reference.tolerance
                   if case.reference.tolerance else 0.0, case) for case in selected]
        worst_ratio, worst = max(ratios, key=lambda row: row[0])
        ref = worst.reference
        receipt = {
            "schema": "epyc.autokernel.native_op_metric.v1", "op": op,
            "backend": backend, "suite_seed": REFERENCE_SUITE_SEED,
            "cases": len(selected), "oracle": ref.oracle_id,
            "metrics": sorted({case.reference.metric_id for case in selected}),
            "worst_metric": ref.metric_id,
            "worst_fraction_of_tolerance": worst_ratio,
            "worst_case": worst.params[:256],
            "worst_case_sha256": hashlib.sha256(worst.params.encode()).hexdigest(),
            "observed": ref.observed,
            "tolerance": ref.tolerance,
            "raw_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }
        return Verdict("correctness", True, detail=json.dumps(receipt, sort_keys=True))
    return Verdict("correctness", True, detail=done.stdout[-500:])


def deterministic(build_dir: Path, model: Path, *, runs: int = 3) -> Verdict:
    """The same input must give the same output three times.

    Cheap, and it catches a class the op oracle does not: a kernel that is correct on
    average and racy in practice. Run on the candidate only -- the anchor's
    determinism is not what is in question.
    """
    binary = build_dir / "bin" / "llama-bench"
    if not binary.is_file():
        return Verdict("determinism", False, f"no llama-bench at {binary}")
    seed = bench._candidate_seed()
    seen: set[str] = set()
    for _ in range(runs):
        done = subprocess.run(
            [str(binary), "-m", str(model), "-p", "0", "-n", "8", "-r", "1",
             "-ngl", "99", "-fa", "1", "-o", "json",
             "--autokernel-harden", str(seed)],
            capture_output=True, text=True, timeout=600,
            env=residency.loader_env(binary))
        if done.returncode != 0:
            return Verdict("determinism", False, "candidate failed to run",
                           done.stderr[-1000:])
        try:
            row = bench.hardened_row(done.stdout, pp=0, tg=8, reps=1)
        except bench.BenchFailed as exc:
            return Verdict("determinism", False, str(exc), done.stdout[-1000:])
        seen.add(row.autokernel_output_hashes)
    if len(seen) != 1:
        return Verdict("determinism", False,
                       f"candidate outputs changed across {runs} identical hardened runs",
                       "\n".join(sorted(seen)))
    return Verdict("determinism", True)


def no_fallback_dispatch(build_dir: Path, model: Path, *, pp: int, tg: int,
                         ubatch: int | None = None, op: str = "MUL_MAT") -> Verdict:
    """Observe the affected op's scheduler placement and apply the T0 no-fallback gate."""
    binary = build_dir / "bin" / "llama-bench"
    if not binary.is_file():
        return Verdict("no_fallback_dispatch", False, f"no llama-bench at {binary}")
    shape = census.Shape("prefill" if pp else "decode", pp if pp else tg)
    recipe = ["-ngl", "99", "-fa", "1"]
    if ubatch:
        recipe.extend(("-b", str(ubatch), "-ub", str(ubatch)))
    row = census.run_dispatch_probe(
        binary, model, shape, recipe_argv=recipe,
        env=residency.loader_env(binary), expected_ops=(op,), require_device=True)
    assignments = row.get("op_backend", {}).get(op, {})
    fallback = tuple(
        f"{count} {op} node(s) assigned to {backend}, not ROCm0"
        for backend, count in sorted(assignments.items())
        if backend not in {"ROCm0", "NULL"}
    )
    evidence = correctness.DispatchTraceEvidence(
        derived_surface=(op,),
        traced_kernels=((op,) if op in row.get("op_backend", {}) else ()),
        fallback_events=fallback,
        fallback_instrumentation_active=row.get("state") == census.OBSERVED,
        trace_ref="inline:autokernel-loop-scheduler-trace",
        produced_by="evaluator",
    )
    result = correctness.check_no_fallback_dispatch_proof(evidence)
    passed = result.check.outcome == schemas.PASS
    reason = "" if passed else "; ".join(result.check.reasons)
    return Verdict("no_fallback_dispatch", passed, reason,
                   str({"state": row.get("state"), "assignments": assignments,
                        "nodes_total": row.get("nodes_total")}))


def run_all(*checks: "Callable[[], Verdict]") -> tuple[bool, list[Verdict]]:
    """Short-circuit at the first refusal; return every verdict for the record.

    Takes CALLABLES, not verdicts. It used to take `*verdicts: Verdict`, which made the
    documented short-circuit impossible: Python evaluates every argument before the call,
    so `run_all(compiles(...), op_correctness(...))` ran the correctness suite even when
    the build had just FAILED -- against whatever binary happened to be left in the
    candidate build directory from a previous iteration.

    The recorded verdicts stayed correct -- the loop returns at the first failure, so the
    eagerly computed correctness verdict was discarded rather than reported. What was lost
    was time and meaning: every failed build in run 9 still paid for a full
    `test-backend-ops` run, executed against whatever stale binary the previous iteration
    left behind. A gate that runs after the gate before it refused is not a gate, even
    when nobody reads its answer.
    """
    collected: list[Verdict] = []
    for check in checks:
        verdict = check()
        collected.append(verdict)
        if not verdict.passed:
            return False, collected
    return True, collected


__all__ = ["BUILD_TIMEOUT_S", "CORRECTNESS_TIMEOUT_S", "DEFAULT_TARGETS",
           "PROMOTION_TARGETS", "Verdict", "compiles", "deterministic",
           "affected_op_scope", "check_cpu_gdn_reference", "check_cpu_iqk_reference",
           "no_fallback_dispatch",
           "op_correctness", "run_all"]
