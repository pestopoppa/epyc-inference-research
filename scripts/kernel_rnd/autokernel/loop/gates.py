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

import dataclasses
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


#: The Q4_K/Q5_K dot route's edit boundary, in source order: (label, line prefix,
#: closing line, required).  The two scale-unpack helpers exist only in trees that
#: carry the `akm-q4k-pairrow-metadata-zmm` keep (the v10 keeps tree); production
#: v10 and every tree cut from it (the DS41 port included) have neither.  Requiring
#: them refused EVERY edit of this kernel on those anchors -- DS41 run 9c spent two
#: critic-accepted authoring rounds (~47 actor-minutes) on a hoist the gate could
#: never admit, under a reason that named neither the missing marker nor the rule.
#: An optional helper is admitted only when HEAD and candidate agree on its presence.
_Q45_DOT_REGIONS = (
    ("DequantizerQ4K_AVX2", "struct DequantizerQ4K_AVX2 final :", "};", True),
    ("DequantizerQ5K_AVX2", "struct DequantizerQ5K_AVX2 final :", "};", True),
    ("unpack_q4_scales", "inline __m128i unpack_q4_scales(", "}", False),
    ("unpack_q4_scales_2", "inline __m256i unpack_q4_scales_2(", "}", False),
    ("mul_mat_qX_K_q8_2_X4_T", "static void mul_mat_qX_K_q8_2_X4_T(", "}", True),
)
_Q45_DOT_FENCE = "struct DequantizerQ6K_AVX2 final :"


def _iqk_q45_dot_scope_refusal(source_text: str | None,
                               pre_source_text: str | None,
                               patch_text: str | None) -> str | None:
    """Admit only the Q4_K/Q5_K private dot implementation, never siblings.

    Returns None when every hunk lies inside one admitted body in both HEAD and
    the candidate, else the exact rule that refused it (logged verbatim in the
    gate verdict, the step line and the author's next prompt).

    The x86 dispatch instantiates this template for Q4_K and Q5_K only.  The
    Q4Bits_AVX2 helper immediately before it is also used by Q6_K and is
    deliberately outside the edit boundary.  A missing/duplicated REQUIRED marker
    refuses rather than silently expanding the boundary after source drift.
    """
    if not source_text or not pre_source_text or not patch_text:
        return "HEAD source, candidate source or the -U0 patch is empty"

    def bounds(text: str, side: str):
        lines = text.splitlines()
        present = []
        for label, prefix, token, required in _Q45_DOT_REGIONS:
            hits = [i + 1 for i, line in enumerate(lines) if line.startswith(prefix)]
            if len(hits) > 1 or (required and len(hits) != 1):
                return (f"{side}: marker `{prefix}` occurs {len(hits)} times "
                        f"(the {label} boundary needs exactly one)")
            if hits:
                present.append((label, hits[0], token))
        fence = [i + 1 for i, line in enumerate(lines) if line.startswith(_Q45_DOT_FENCE)]
        if len(fence) != 1:
            return (f"{side}: fence `{_Q45_DOT_FENCE}` occurs {len(fence)} times "
                    "(needs exactly one)")
        positions = [pos for _label, pos, _token in present] + fence
        if positions != sorted(set(positions)):
            return f"{side}: admitted bodies are not in source order before the Q6_K fence"
        # A duplicated or rewritten selector no longer warrants Q4/Q5-only
        # scope.  Q6 must remain on the different qY template.
        for quant, dequant in (("Q4_K", "DequantizerQ4K_AVX2"),
                               ("Q5_K", "DequantizerQ5K_AVX2")):
            dispatch = f"IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_X4_T, {dequant}, kernels)"
            if text.count(dispatch) != 1 or text.count(f"case GGML_TYPE_{quant}:") < 1:
                return f"{side}: the {quant} dispatch `{dispatch}` is missing or duplicated"

        def closing(start: int, stop: int, token: str, *, last=False):
            hits = [i for i in range(start + 1, stop) if lines[i - 1].strip() == token]
            return (hits[-1] if last else hits[0]) if hits else None

        regions = []
        for index, (label, start, token) in enumerate(present):
            stop = positions[index + 1]
            end = closing(start, stop, token, last=index == len(present) - 1)
            if end is None:
                return f"{side}: {label} body (line {start}) has no closing `{token}`"
            # A candidate must not close the named body early and smuggle a new
            # sibling definition between its original markers.  This allowlist
            # contains no brace-bearing strings/comments in its reviewed base.
            depth = 0
            for pos in range(start, end + 1):
                depth += lines[pos - 1].count("{") - lines[pos - 1].count("}")
                if depth <= 0 and pos < end:
                    return (f"{side}: {label} body closes early at line {pos} "
                            "(braces must stay balanced inside the admitted body)")
            if depth != 0:
                return f"{side}: {label} body has unbalanced braces"
            regions.append((label, start + 1, end - 1))
        return tuple(regions), tuple(lines[pos - 1] for pos in positions)

    old = bounds(pre_source_text, "HEAD")
    if isinstance(old, str):
        return old
    new = bounds(source_text, "candidate")
    if isinstance(new, str):
        return new
    if old[1] != new[1]:
        return ("an admitted signature/marker line or the helper set differs between HEAD "
                "and the candidate; only the bodies may change")
    hunks = re.findall(r"(?m)^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", patch_text)
    if not hunks:
        return "the -U0 patch has no hunks"

    def within(line: str, count: str, region) -> bool:
        first, size = int(line), int(count) if count else 1
        return region[1] <= first and first + max(size, 1) - 1 <= region[2]

    for old_line, old_count, new_line, new_count in hunks:
        if not any(within(old_line, old_count, old[0][i]) and
                   within(new_line, new_count, new[0][i]) for i in range(len(old[0]))):
            admitted = ", ".join(f"{label} {first}-{last}" for label, first, last in old[0])
            return (f"hunk @@ -{old_line},{old_count or 1} +{new_line},{new_count or 1} @@ "
                    f"lies outside every admitted body (HEAD lines: {admitted})")
    return None


def _iqk_q45_dot_hunks_confined(source_text: str | None,
                                pre_source_text: str | None,
                                patch_text: str | None) -> bool:
    return _iqk_q45_dot_scope_refusal(source_text, pre_source_text, patch_text) is None


# --- Widened CPU source routes (DS41 decode scope, operator decision 2026-09-26) ------
#
# Each route is ONE file, a closed set of target symbols, and hunks confined to named
# function/member BODIES in both HEAD and the candidate, with every header line (the
# marker through its opening brace) byte-identical.  Markers are searched inside an
# optional container (a class) and before an optional fence (a stub section), so a
# marker that recurs elsewhere in the file (sgemm.cpp has nine `mnpack`s) or in a
# disabled-build stub (iqk_dispatch.cpp) never widens the boundary.
#
# route -> (path, container prefix, fence prefix, admitted bodies, native ops,
#           forbidden added-line pattern, what the author is told)
_DS41_GRAPH_OPS = ("MUL_MAT", "MUL_MAT_ID", "ADD", "MUL", "RMS_NORM", "SCALE", "CLAMP",
                   "CONT", "CPY", "CONCAT", "GLU", "UNARY", "SUM_ROWS", "GET_ROWS",
                   "SET_ROWS", "ROPE", "SOFT_MAX", "ARGSORT", "FLASH_ATTN_EXT")
# cpu_graph_sync edits decide WHICH nodes run solo on thread 0 and which barriers remain,
# so every op `ggml_cpu_node_is_solo` admits is affected whether or not the DS41 list above
# named it (DIV closes DS41's own MoE weight normalization).  TOPK_MOE is the one native
# fixture that runs that normalization chain (GET_ROWS, SUM_ROWS, CLAMP, DIV; with norm and
# SQRT_SOFTPLUS gating among its cases) as consecutive nodes of ONE graph through the
# candidate's graph walk: a single-op case never forms the multi-node solo run a
# solo-eligibility change collapses (DS41-C96).
_SOLO_ELIGIBLE_OPS = ("SUB", "DIV", "SQR", "SQRT", "LOG", "SIN", "COS", "FILL", "DUP")
_DS41_SYNC_OPS = (*_DS41_GRAPH_OPS, *_SOLO_ELIGIBLE_OPS, "TOPK_MOE")

#: `test-backend-ops -o` compares its list against `ggml_op_desc()`, which names a UNARY
#: or GLU node by its SUB-op (SILU, SWIGLU, ...) and never returns "UNARY" or "GLU". So
#: `-o GLU` selected 0/0 CPU cases and every cpu_graph_sync candidate was refused
#: `oracle_unavailable` at its GLU suite after ten passing ones: 10 gate refusals, 5
#: scope_blocked hypotheses and the run10w abstention storm (DS41-C96). The scope keeps
#: the ggml op TYPE; the oracle passes every sub-op name (ggml.c GGML_UNARY_OP_NAME /
#: GGML_GLU_OP_NAME, in table order) except those in UNSELECTABLE_ON_ANCHOR.
#:
#: EXP and EXPM1 are left out because they FAIL ON THE ANCHOR ITSELF. Their f32 cases draw
#: inputs from [-150, 150], exp(150) overflows to +inf in both arms, the error is
#: non-finite and test-backend-ops prints "ERR is invalid ... FAIL". That is 8 of 196
#: cases on anchor-gen-011 (2026-09-30). Selecting them would turn every cpu_graph_sync
#: candidate into a "UNARY failed on CPU" correctness verdict that says nothing about the
#: patch, so they are named here rather than silently absent. The other 20 unary sub-ops
#: pass 180/180 on that anchor, and GLU passes 128/128.
UNSELECTABLE_ON_ANCHOR = {"UNARY": ("EXP", "EXPM1")}
BACKEND_OPS_SELECTORS = {
    "UNARY": ("ABS", "SGN", "NEG", "STEP", "TANH", "ELU", "RELU", "SIGMOID", "GELU",
              "GELU_QUICK", "SILU", "HARDSWISH", "HARDSIGMOID", "SOFTPLUS",
              "GELU_ERF", "XIELU", "FLOOR", "CEIL", "ROUND", "TRUNC"),
    "GLU": ("REGLU", "GEGLU", "SWIGLU", "SWIGLU_OAI", "GEGLU_ERF", "GEGLU_QUICK"),
}


def backend_ops_selector(op: str) -> str:
    """The `test-backend-ops -o` list that selects the cases of ggml op `op`."""
    return ",".join(BACKEND_OPS_SELECTORS.get(op, (op,)))

# cpu_norm_rowsplit: HEAD's rms_norm numerics, spelled as HEAD spells them. A within-row
# split may re-add these lines (moved or re-indented), but no other added code line may
# name the accumulator, mean, scale or eps, an accumulation type, a square of x, a
# SIMD/vector reduction, a pragma, or an in-op barrier (RMS_NORM is tiny-solo eligible: a
# solo node runs on thread 0 alone, so an in-op ggml_barrier would deadlock). The loop
# index name is free (backreferenced); order, types and association are not. Comment-only
# lines (`//`, one-line `/* */`) are exempt.
_RMS_NORM_CANONICAL = (
    r"ggml_float sum = 0\.0;",
    r"sum \+= \(ggml_float\)\(x\[(?P<si>\w+)\] \* x\[(?P=si)\]\);",
    r"const float mean\s+= sum/ne00;",
    r"const float scale = 1\.0f/sqrtf\(mean \+ eps\);",
    r"assert\(scale > 0\.0f\);",
    r"y\[(?P<yi>\w+)\] = x\[(?P=yi)\] \* scale \* w\[(?P=yi)\];",
    r"ggml_vec_scale_f32\((?:(?!\bscale\b)[^;])*,\s*scale\);",
    r"float eps;",
    r"memcpy\(&eps, dst_rms_norm->op_params, sizeof\(float\)\);",
    r"GGML_ASSERT\(eps >= 0\.0f\);",
)
_RMS_NORM_FORBIDDEN = (
    r"^\+(?!\s*(?://.*|/\*(?:(?!\*/).)*\*/\s*|(?:" + "|".join(_RMS_NORM_CANONICAL) +
    r")\s*(?://.*)?)$).*?(?:"
    r"\b(?:sum|mean|scale|eps|ggml_float|double|sqrtf?|rsqrtf?|fmaf?|ggml_barrier|"
    r"accumulate|inner_product|reduce|transform_reduce)\b"
    r"|_mm\w*|__m\d+\w*|\bGGML_F\d+\w*|\bggml_vec_(?!(?:scale|cpy)_f32\b)\w+"
    r"|#\s*pragma|\bx\s*\[[^\]]*\]\s*\*\s*x\s*\[)")


@dataclass(frozen=True)
class CpuSourceRoute:
    route: str
    path: str
    symbols: tuple[str, ...]
    bodies: tuple[tuple[str, str], ...]
    ops: tuple[str, ...]
    container: str | None = None
    fence: str | None = None
    forbidden_added: str | None = None
    admitted_text: str = ""
    #: 2026-10-03 (operator: "the loop must be able to author the kernel mutations the
    #: seeds need"): also admit pure-insertion hunks at file scope that add NEW `static`
    #: helpers (functions, constants) or `#include <system>` lines -- see
    #: `_new_helper_refusal`. Bodies stay the only place existing code may change, and
    #: `forbidden_added` still applies to every added line.
    new_helpers: bool = False
    #: The route's correctness gate includes the whole-model output identity check
    #: (`model_identity`): a placement/loader edit that no op suite exercises, or (with
    #: a witness as well) a scheduler/model-op edit whose failure only a whole graph shows.
    model_identity: bool = False
    # ---- 2026-10-04 structural routes (cpu_graph_sched / cpu_model_fused_op /
    # cpu_graph_optimize; spec: tmp/cpu-structural-seeds-20261004 §R1-R3) ----
    #: Bodies (by label) whose ONLY admitted change is a pure-insertion APPEND at the
    #: tail: (label, tail regex | None, entry regex). Every HEAD line between the
    #: insertion point and the first line matching the tail regex (else the body's last
    #: line) must be blank/comment, and every added code line must match the entry regex
    #: -- so existing fields/enumerators/table rows are never removed, reordered or edited.
    append_bodies: tuple[tuple[str, str | None, str], ...] = ()
    #: Switch-bearing bodies (by label) whose ONLY admitted change is a NEW `case` block
    #: inserted at a case boundary, labelled only with enumerators this same patch
    #: appends (`append_bodies` of `enum ggml_op`), ending in `break;`/`return`, so no
    #: existing case gains or loses code or a fallthrough.
    case_bodies: tuple[str, ...] = ()
    #: Regex for the first line of a NON-static NEW file-scope definition/declaration
    #: (an exported `ggml_<op>()` constructor, a `ggml_compute_forward_<op>` kernel, a
    #: `GGML_API` prototype). The defined name must not occur anywhere in HEAD's file.
    new_definitions: str | None = None
    #: `new_definitions` are prototypes (headers): no braces in the inserted statement.
    declarations_only: bool = False
    #: Whole-line rewrites admitted outside the bodies: (old regex, new regex), each
    #: removed line and its replacement fully matching one pair (GGML_OP_COUNT asserts,
    #: the `graph_optimize = NULL` slot).
    line_rewrites: tuple[tuple[str, str], ...] = ()
    #: `forbidden_added` does not apply to hunks inside these bodies.
    forbidden_exempt_bodies: tuple[str, ...] = ()
    #: (regex, refusal) -- some added line of the patch must match the regex.
    required_added: tuple[str, str] | None = None
    #: Whose model the identity gate serves: "own" (this lane's target), "peers" (also
    #: every peer target of the lane binding, when one is configured) or "shared_peers"
    #: (peers only when a changed path is outside this lane's exclusive paths).
    identity_targets: str = "own"
    #: Serve each frozen request this many times from the candidate (cache_prompt off):
    #: all repetitions must be byte-identical -- the race detector a scheduler needs.
    identity_repeats: int = 1
    #: At least one identity target must serve a model of one of these GGUF
    #: architectures (a model-specific builder is only judged by its own model).
    identity_arch: tuple[str, ...] = ()


CPU_SOURCE_ROUTES = (
    # Dense Q8_0 small-N GEMM: the type-GENERIC tiling/compute members of
    # tinyBLAS_Q0_AVX (verify-batch N=2..3 lands in gemm4xN<2|3>). The class is also
    # instantiated for Q4_0/Q5_0/IQ4_NL, and llamafile is NOT bypassed by use_ref, so
    # the native suite is not independent here: type-specific load*/updot/denibble/
    # bittobyte stay outside the boundary and added lines may not touch block quant
    # fields directly, which keeps the Q8_0 scalar reference representative.
    CpuSourceRoute(
        route="dense_q8_tinyblas",
        path="ggml/src/ggml-cpu/llamafile/sgemm.cpp",
        symbols=("tinyBLAS_Q0_AVX", "mnpack", "gemm4xN", "gemmMx4", "gemm"),
        container="class tinyBLAS_Q0_AVX {",
        bodies=(("mnpack", "    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {"),
                ("gemm4xN", "    NOINLINE void gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {"),
                ("gemmMx4", "    NOINLINE void gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {"),
                ("gemm", "    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {")),
        ops=("MUL_MAT",),
        forbidden_added=r"(\.|->)(qs|qh)\b",
        admitted_text=("hunks inside the tinyBLAS_Q0_AVX mnpack/gemm4xN/gemmMx4/gemm bodies; "
                       "load*/updot/denibble/bittobyte, the class header, other classes and "
                       "direct .qs/.qh access unchanged")),
    # MoE expert dispatch: single-token B3-k slab AND the N>1 verify-batch path
    # (thread-0 activation quantization + mapping barrier + 1/nth per-expert stripes).
    # Also the landing zone for the rowexact partition keeps, which touch only this body.
    CpuSourceRoute(
        route="iqk_mmid_dispatch",
        path="ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp",
        symbols=("ggml_iqk_try_mul_mat_id",),
        fence="#else  // iqk not implemented",
        bodies=(("ggml_iqk_try_mul_mat_id",
                 'extern "C" bool ggml_iqk_try_mul_mat_id(const struct ggml_compute_params * params, '
                 'struct ggml_tensor * dst) {'),),
        ops=("MUL_MAT_ID",),
        admitted_text=("hunks inside the implemented ggml_iqk_try_mul_mat_id body; its "
                       "signature, the disabled-build stub, helpers and kernels unchanged")),
    # Dense iqk dispatch, including the Q8_0 opt-in (`iqk_q8_0_enabled`, default off
    # since aebb556b1); that predicate also gates Q8_0 MUL_MAT_ID, hence both ops.
    CpuSourceRoute(
        route="iqk_dense_dispatch",
        path="ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp",
        symbols=("ggml_iqk_try_mul_mat", "iqk_q8_0_enabled"),
        fence="#else  // iqk not implemented",
        bodies=(("iqk_q8_0_enabled", "inline bool iqk_q8_0_enabled() {"),
                ("ggml_iqk_try_mul_mat",
                 'extern "C" bool ggml_iqk_try_mul_mat(const struct ggml_compute_params * params, '
                 'struct ggml_tensor * dst) {')),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        admitted_text=("hunks inside the iqk_q8_0_enabled or implemented ggml_iqk_try_mul_mat "
                       "bodies; signatures, the stub and every other helper unchanged")),
    # Per-node synchronisation, graph walk, tiny-solo selection and in-backend fusion.
    # Numerics-free by construction, so the independent reference is the full scalar
    # quant suite (it runs through the candidate's barriers) plus every DS41 and
    # solo-eligible op suite and the whole-graph TOPK_MOE chain. Those native suites are
    # NOT independent of the walk: use_ref disables only fusion, so the reference arm runs
    # the same edited scheduler (at 4 threads against the candidate's full team). They
    # catch per-op numerics and publish-before-consume races; a deterministic walk defect
    # shared by both arms is caught only where the scalar quant reference reaches it.
    CpuSourceRoute(
        route="cpu_graph_sync",
        path="ggml/src/ggml-cpu/ggml-cpu.c",
        symbols=("ggml_barrier", "ggml_cpu_node_is_solo", "ggml_cpu_try_fuse_ops",
                 "ggml_graph_compute_thread"),
        bodies=(("ggml_barrier", "void ggml_barrier(struct ggml_threadpool * tp) {"),
                ("ggml_cpu_node_is_solo",
                 "static bool ggml_cpu_node_is_solo(const struct ggml_tensor * node) {"),
                ("ggml_cpu_try_fuse_ops", "static int ggml_cpu_try_fuse_ops("),
                ("ggml_graph_compute_thread",
                 "static thread_ret_t ggml_graph_compute_thread(void * data) {")),
        ops=_DS41_SYNC_OPS,
        new_helpers=True,
        admitted_text=("hunks inside the ggml_barrier, ggml_cpu_node_is_solo, "
                       "ggml_cpu_try_fuse_ops or ggml_graph_compute_thread bodies, plus NEW "
                       "file-scope static helpers (e.g. a fused kernel called from "
                       "ggml_cpu_try_fuse_ops) and #include <...> lines; existing headers, "
                       "globals, op kernels and every other function unchanged")),
    # GRAPH SCHEDULING (2026-10-04, structural seeds 3 "co-group independent nodes" and
    # 4(B) "expert groups"): the cpu_graph_sync bodies PLUS ggml_graph_plan (wdata sizing
    # for concurrent nodes) and ggml_graph_compute (the one `#pragma omp parallel`), and
    # field APPENDS to struct ggml_threadpool / struct ggml_compute_state (per-group
    # chunk counters, group barriers). Tried AFTER cpu_graph_sync on the shared bodies,
    # so a patch that fits the narrower route keeps its gate. SCHED: per-element
    # arithmetic is unchanged, but the thread->work mapping moves, so the gate is the
    # sync oracle + scalar quant suite + fusion reference (witness `cpu_graph_sched`),
    # PLUS whole-model greedy identity on this lane's target AND every peer target, each
    # frozen request served 3x from the candidate (a race between two concurrent nodes
    # is invisible to a per-op suite). `use_ref` must force group size 1 (the reference
    # arm of every native suite then walks sequentially): an added line must name it.
    # `#pragma omp` stays forbidden outside ggml_barrier (OpenMP build: a group that
    # reaches a team-wide barrier another group does not deadlocks).
    CpuSourceRoute(
        route="cpu_graph_sched",
        path="ggml/src/ggml-cpu/ggml-cpu.c",
        symbols=("ggml_barrier", "ggml_cpu_node_is_solo", "ggml_cpu_try_fuse_ops",
                 "ggml_graph_compute_thread", "ggml_graph_plan", "ggml_graph_compute",
                 "ggml_threadpool", "ggml_compute_state"),
        bodies=(("ggml_threadpool", "struct ggml_threadpool {"),
                ("ggml_compute_state", "struct ggml_compute_state {"),
                ("ggml_barrier", "void ggml_barrier(struct ggml_threadpool * tp) {"),
                ("ggml_cpu_node_is_solo",
                 "static bool ggml_cpu_node_is_solo(const struct ggml_tensor * node) {"),
                ("ggml_graph_plan", "struct ggml_cplan ggml_graph_plan("),
                ("ggml_cpu_try_fuse_ops", "static int ggml_cpu_try_fuse_ops("),
                ("ggml_graph_compute_thread",
                 "static thread_ret_t ggml_graph_compute_thread(void * data) {"),
                ("ggml_graph_compute",
                 "enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, "
                 "struct ggml_cplan * cplan) {")),
        ops=_DS41_SYNC_OPS,
        new_helpers=True,
        model_identity=True,
        identity_targets="peers",
        identity_repeats=3,
        append_bodies=(
            ("ggml_threadpool", None, r"^[^{}#=]*;\s*(//.*)?$"),
            ("ggml_compute_state", None, r"^[^{}#=]*;\s*(//.*)?$")),
        forbidden_added=r"#\s*pragma\s+omp",
        forbidden_exempt_bodies=("ggml_barrier",),
        required_added=(r"\buse_ref\b",
                        "a scheduling change must fall back to group size 1 (HEAD's "
                        "sequential walk) when cplan->use_ref / params->use_ref is set, so "
                        "the native suites' reference arm is the sequential schedule; no "
                        "added line names use_ref"),
        admitted_text=("hunks inside the ggml_barrier / ggml_cpu_node_is_solo / "
                       "ggml_cpu_try_fuse_ops / ggml_graph_compute_thread / ggml_graph_plan "
                       "/ ggml_graph_compute bodies, field APPENDS at the tail of struct "
                       "ggml_threadpool and struct ggml_compute_state (no removal, reorder "
                       "or edit of an existing field), plus NEW file-scope static helpers "
                       "and #include <...> lines; the change must honour use_ref (group size "
                       "1); no `#pragma omp` outside ggml_barrier; op kernels, headers and "
                       "every other function unchanged. Gate: sync oracle + scalar quant "
                       "suite + fusion reference, and greedy identity on this lane's target "
                       "and every peer target with each request served 3x byte-identical")),
    # Float (F32/F16/BF16) GEMM tile PLAN: the type-generic `tinyBLAS::matmul` picks
    # RM*BM = 8- or 16-row y-tiles, so a narrow-M matrix gets fewer jobs than threads.
    # DS41 hc_mixes (F16 [20480, 24], 80 nodes per verify at N=3) yields 24/8 = 3 jobs
    # for 48 threads (DS41 inbox 52). gemm_bloc accumulates every output element over k
    # in the same order whatever RM/RN/BM or thread it lands on, so an edit confined to
    # the plan is bit-exact; gemm_bloc/load/madd/hsum and the barrier-bearing gemm body
    # stay outside the boundary, and added lines may not touch the numerics or operands.
    CpuSourceRoute(
        route="float_tinyblas_plan",
        path="ggml/src/ggml-cpu/llamafile/sgemm.cpp",
        symbols=("tinyBLAS", "matmul"),
        container="class tinyBLAS {",
        bodies=(("matmul", "    bool matmul(int64_t m, int64_t n) {"),),
        ops=("MUL_MAT",),
        forbidden_added=r"\b(madd|hsum|load|gemm_bloc)\b|_mm\w*|\b[ABC]\s*\[",
        admitted_text=("hunks inside the class tinyBLAS matmul body (tile plan: which "
                       "mnpack<RM, RN, BM> and SIZE_N/BN it calls); mnpack, gemm_bloc, gemm, "
                       "load/madd/hsum, other classes and direct A/B/C access unchanged")),
    # Narrow-row F32 RMS_NORM split: HEAD deals whole rows to threads, so DS41's hc_mixes
    # input norm ([20480, nt], nt = 2..3 in serving verify, 80 per graph, unfused: its
    # consumer is the hc_mixes MUL_MAT) keeps 3 of 48 threads busy. The admitted mechanism
    # is a bit-exact within-row split: every thread recomputes the FULL row sum in HEAD's
    # order, then scales only its own column segment (`get_rowcol_split` in common.h is
    # the existing helper). The one template body serves both RMS_NORM and the fused
    # RMS_NORM+MUL (`ggml_compute_forward_rms_norm_mul_fused`), so both ops are in scope.
    # NORM/GROUP_NORM/RMS_NORM_BACK and the dispatchers stay outside the boundary.
    # Measured (2026-09-27, see program.md): the node is the serial double-add chain of
    # the row sum, which every split task recomputes, so the split's bound is the scale
    # pass (<= 0.4 us of a ~12 us node), not the node's 3.1-3.5% cycle share.
    CpuSourceRoute(
        route="cpu_norm_rowsplit",
        path="ggml/src/ggml-cpu/ops.cpp",
        symbols=("ggml_compute_forward_rms_norm_f32", "ggml_compute_forward_rms_norm",
                 "ggml_compute_forward_rms_norm_mul_fused"),
        bodies=(("ggml_compute_forward_rms_norm_f32",
                 "static void ggml_compute_forward_rms_norm_f32("),),
        ops=("RMS_NORM", "RMS_NORM_MUL_ADD"),
        forbidden_added=_RMS_NORM_FORBIDDEN,
        admitted_text=("hunks inside the ggml_compute_forward_rms_norm_f32 template body "
                       "(work split only: every thread recomputes the full row sum in HEAD's "
                       "order, then scales its own column segment; in place, i.e. dst->data == "
                       "src0->data, keep the row split). HEAD's sum/mean/scale/eps and fused "
                       "product lines may be re-added verbatim (any loop index); no other "
                       "added code line may name sum/mean/scale/eps, ggml_float/double, "
                       "sqrt/fma, x[i]*x[j], _mm*/GGML_F*/ggml_vec_* other than "
                       "ggml_vec_scale_f32/ggml_vec_cpy_f32, #pragma or ggml_barrier. The "
                       "header, the dispatchers, NORM/GROUP_NORM/RMS_NORM_BACK unchanged")),
    # 2026-10-03 widening (Fable CPU seeds 1, 3, 4, 6, 9; operator: "unacceptable" that
    # the loop could not author them). Each new route keeps a reviewed, independent gate.
    #
    # RMS_NORM NUMERICS (seed 4, vector sum of squares): the same body as
    # cpu_norm_rowsplit, tried AFTER it, so a bit-exact split still gets the bit-exact
    # gate and only a patch that changes the arithmetic lands here. Reference:
    # `cpu_norm_reference` in tolerance mode -- the 16 fixed cases against a float64
    # reference at 2^-16 relative (vs HEAD's own 5.5 u), repetitions bit-identical to
    # each other (races), no bit identity with HEAD. An in-op barrier still deadlocks a
    # solo node, so ggml_barrier stays forbidden.
    CpuSourceRoute(
        route="cpu_norm_numerics",
        path="ggml/src/ggml-cpu/ops.cpp",
        symbols=("ggml_compute_forward_rms_norm_f32", "ggml_compute_forward_rms_norm",
                 "ggml_compute_forward_rms_norm_mul_fused"),
        bodies=(("ggml_compute_forward_rms_norm_f32",
                 "static void ggml_compute_forward_rms_norm_f32("),),
        ops=("RMS_NORM", "RMS_NORM_MUL_ADD"),
        forbidden_added=r"\bggml_barrier\b|#\s*pragma\s+omp",
        new_helpers=True,
        admitted_text=("hunks inside the ggml_compute_forward_rms_norm_f32 template body "
                       "plus NEW file-scope static helpers; the arithmetic may change "
                       "(vector/multi-accumulator sum of squares), judged against float64 "
                       "at 2^-16 relative with repetitions bit-identical; no ggml_barrier "
                       "(solo nodes run on one thread); header, dispatchers and every "
                       "other function unchanged")),
    # MUL_MAT BODY (seed 6 small-node prelude / src1 quantisation; seed 9's prefetch may
    # also live here): `ggml_compute_forward_mul_mat` (src1 conversion, mm_batch1, the
    # current_chunk barrier, the llamafile/iqk dispatch, the chunk loop) and its
    # `_one_chunk` worker. Reference: the full scalar Q8_0/Q4_K/Q5_K/F16 MUL_MAT
    # fixture at widths 1-8 (it decodes stored bytes without ggml, so it is independent
    # of this body, which use_ref also runs) plus a GDB entry hit in the candidate DSO.
    CpuSourceRoute(
        route="cpu_mul_mat_body",
        path="ggml/src/ggml-cpu/ggml-cpu.c",
        symbols=("ggml_compute_forward_mul_mat", "ggml_compute_forward_mul_mat_one_chunk"),
        bodies=(("ggml_compute_forward_mul_mat_one_chunk",
                 "static void ggml_compute_forward_mul_mat_one_chunk("),
                ("ggml_compute_forward_mul_mat", "void ggml_compute_forward_mul_mat(")),
        ops=("MUL_MAT",),
        new_helpers=True,
        admitted_text=("hunks inside the ggml_compute_forward_mul_mat or "
                       "ggml_compute_forward_mul_mat_one_chunk bodies plus NEW file-scope "
                       "static helpers and #include <...> lines; headers, MUL_MAT_ID, the "
                       "graph walk and every other function unchanged")),
    # WEIGHT PLACEMENT AT LOAD (seed 1, NUMA-quartered weights): the loader's data-load
    # bodies may place (mbind/madvise) the bytes they read. Placement does not change a
    # single value, and no op suite runs the loader, so the gate is the whole model:
    # `model_identity` serves the frozen requests greedily from the anchor and the
    # candidate under the campaign launch and requires identical completions (an anchor
    # that disagrees with itself makes the gate unavailable, never a verdict).
    CpuSourceRoute(
        route="cpu_weight_placement",
        path="src/llama-model-loader.cpp",
        symbols=("load_all_data", "load_all_data_parallel", "llama_model_loader"),
        bodies=(("load_all_data_parallel",
                 "bool llama_model_loader::load_all_data_parallel("),
                ("load_all_data", "bool llama_model_loader::load_all_data(")),
        ops=(),
        new_helpers=True,
        model_identity=True,
        admitted_text=("hunks inside the llama_model_loader::load_all_data or "
                       "load_all_data_parallel bodies plus NEW file-scope static helpers "
                       "and #include <...> lines (placement only: mbind/madvise/first "
                       "touch of the bytes being loaded); headers and every other "
                       "function unchanged. Gate: identical greedy completions vs the "
                       "anchor on the frozen requests")),
)
CPU_SOURCE_ROUTE_PATHS = tuple(sorted({route.path for route in CPU_SOURCE_ROUTES}))


def _route_symbol_names(target_symbol: str) -> list[str]:
    """Bare names a planner-written target symbol can mean, most specific first.

    Planners write symbols the way they read them, e.g. ``tinyBLAS_Q0_AVX<block_q8_0,
    block_q8_0, float>::gemm4xN (template body, RN=1..4)`` (DS41 run 10h), which the exact
    lookup never matched, so an admitted route refused the patch as "unresolved". Only the
    LOOKUP is lenient: hunk confinement to the route's named bodies is still enforced.
    """
    text = str(target_symbol or "").split(" (", 1)[0].strip()
    while True:
        stripped = re.sub(r"<[^<>]*>", "", text)
        if stripped == text:
            break
        text = stripped
    # `gemm(long, long, long)` as a demangler prints it: drop the parameter list.
    parts = [re.sub(r"\(.*$", "", part).strip() for part in text.split("::")]
    return list(dict.fromkeys(reversed([part for part in parts if part])))


def cpu_source_routes(path: str, target_symbol: str) -> tuple[CpuSourceRoute, ...]:
    """Every widened route a (single path, target symbol) pair names, most specific first.

    Several routes may name one body (cpu_norm_rowsplit, then cpu_norm_numerics): they
    are tried in this order and the FIRST that admits the patch governs it, so the
    stricter gate always wins when the patch fits it."""
    candidates = [route for route in CPU_SOURCE_ROUTES if route.path == path]
    exact = tuple(route for route in candidates if target_symbol in route.symbols)
    if exact:
        return exact
    names = _route_symbol_names(target_symbol)
    # A class-qualified symbol picks the route that admits the member AND its class:
    # sgemm.cpp has two tinyBLAS routes, and bare-name order must not hand
    # `tinyBLAS<...>::matmul` or a Q0 member to the other class's route.
    if len(names) > 1:
        qualified = tuple(route for route in candidates
                          if names[0] in route.symbols and names[1] in route.symbols)
        if qualified:
            return qualified
    for name in names:
        routes = tuple(route for route in candidates if name in route.symbols)
        if routes:
            return routes
    return ()


def cpu_source_route(path: str, target_symbol: str) -> CpuSourceRoute | None:
    """The first widened route a (single path, target symbol) pair names, if any."""
    routes = cpu_source_routes(path, target_symbol)
    return routes[0] if routes else None


def cpu_route_named(name: str):
    """The single-file or multi-file route called `name`, if any."""
    return next((route for route in (*CPU_SOURCE_ROUTES, *CPU_MULTI_FILE_ROUTES)
                 if route.route == name), None)


def admit_cpu_route(path: str, target_symbol: str, source_text: str | None,
                    pre_source_text: str | None, patch_text: str | None
                    ) -> tuple[CpuSourceRoute | None, str | None]:
    """(route that admits the patch, None) or (first named route, every refusal).

    (None, None) when no route names the pair at all."""
    routes = cpu_source_routes(path, target_symbol)
    refusals = []
    for route in routes:
        refusal = _cpu_route_scope_refusal(route, source_text, pre_source_text, patch_text)
        if refusal is None:
            return route, None
        refusals.append((route, refusal))
    if not refusals:
        return None, None
    if len(refusals) == 1:
        return refusals[0][0], refusals[0][1]
    return refusals[0][0], " | ".join(
        f"{route.route}: {refusal}. Admitted: {route.admitted_text}"
        for route, refusal in refusals)


def _strip_code_line(line: str, in_block: bool) -> tuple[str, bool]:
    """Code text with comments and string/char literals removed, for brace counting."""
    out, i, n = [], 0, len(line)
    while i < n:
        if in_block:
            end = line.find("*/", i)
            if end < 0:
                return "".join(out), True
            i, in_block = end + 2, False
            continue
        ch = line[i]
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            in_block, i = True, i + 2
            continue
        if ch in "\"'":
            j = i + 1
            while j < n and line[j] != ch:
                j += 2 if line[j] == "\\" else 1
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), in_block


def _brace_deltas(lines: list[str]) -> list[int]:
    deltas, in_block = [], False
    for line in lines:
        code, in_block = _strip_code_line(line, in_block)
        deltas.append(code.count("{") - code.count("}"))
    return deltas


def _cpu_route_bounds(text: str, side: str, route: CpuSourceRoute):
    """Admitted body regions (label, first, last) and header text, or a refusal."""
    lines = text.splitlines()
    deltas = _brace_deltas(lines)

    def block_end(start: int) -> int | None:
        """1-based line closing the block whose first `{` is at/after `start`."""
        depth, opened = 0, False
        for pos in range(start, len(lines) + 1):
            depth += deltas[pos - 1]
            opened = opened or depth > 0
            if opened and depth <= 0:
                return pos
        return None

    lo, hi = 1, len(lines)
    if route.fence is not None:
        fence = [i + 1 for i, line in enumerate(lines) if line.startswith(route.fence)]
        if len(fence) != 1:
            return f"{side}: fence `{route.fence}` occurs {len(fence)} times (needs exactly one)"
        hi = fence[0] - 1
    if route.container is not None:
        starts = [i + 1 for i, line in enumerate(lines[:hi]) if line.startswith(route.container)]
        if len(starts) != 1:
            return (f"{side}: container `{route.container}` occurs {len(starts)} times "
                    "(needs exactly one)")
        end = block_end(starts[0])
        if end is None:
            return f"{side}: container `{route.container}` never closes"
        lo, hi = starts[0], end
    regions, headers = [], []
    for label, prefix in route.bodies:
        hits = [i + 1 for i in range(lo - 1, hi) if lines[i].startswith(prefix)]
        if len(hits) != 1:
            return (f"{side}: marker `{prefix.strip()}` occurs {len(hits)} times inside the "
                    f"admitted window (the {label} boundary needs exactly one)")
        start = hits[0]
        opening = next((pos for pos in range(start, hi + 1) if "{" in
                        _strip_code_line(lines[pos - 1], False)[0]), None)
        end = block_end(start)
        if opening is None or end is None or end > hi:
            return f"{side}: {label} body (line {start}) has no balanced closing brace"
        regions.append((label, opening + 1, end - 1))
        headers.append("\n".join(lines[start - 1:opening]))
    spans = sorted((first, last) for _label, first, last in regions)
    if any(a_last >= b_first for (_a, a_last), (b_first, _b) in zip(spans, spans[1:])):
        return f"{side}: admitted bodies overlap"
    return tuple(regions), tuple(headers)


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunks(patch_text: str) -> list[tuple[str, str, str, str, list[str], list[str]]]:
    """[(old_line, old_count, new_line, new_count, removed, added)] of a -U0 patch.

    Counts are the header's text ("" when omitted, i.e. 1); removed/added lines carry no
    leading -/+."""
    hunks: list = []
    current = None
    for line in patch_text.splitlines():
        header = _HUNK_HEADER.match(line)
        if header:
            current = (*header.groups(""), [], [])
            hunks.append(current)
            continue
        if line.startswith("diff --git "):
            current = None
            continue
        if current is None or line.startswith("\\"):
            continue
        if line.startswith("+"):
            current[5].append(line[1:])
        elif line.startswith("-"):
            current[4].append(line[1:])
    return hunks


def _code_lines(lines: list[str]) -> list[str]:
    """Each line's code text (comments and literals stripped), carrying /* */ state."""
    out, in_block = [], False
    for line in lines:
        code, in_block = _strip_code_line(line, in_block)
        out.append(code)
    return out


def _placements(pre_lines: list[str], old_line: int, block: list[str]):
    """Every equivalent anchoring of a pure insertion of `block` after HEAD `old_line`.

    A diff may anchor the same insertion a few lines up or down where the inserted text
    and its neighbours repeat (a case block ending in `} break;` placed after a case that
    also ends in `} break;`); a rule admits when ANY equivalent placement satisfies it."""
    placements = [(old_line, list(block))]
    at, moved = old_line, list(block)
    while at > 0 and moved and pre_lines[at - 1] == moved[-1] and len(placements) < 64:
        at, moved = at - 1, [pre_lines[at - 1]] + moved[:-1]
        placements.append((at, moved))
    at, moved = old_line, list(block)
    while at < len(pre_lines) and moved and pre_lines[at] == moved[0] and len(placements) < 128:
        at, moved = at + 1, moved[1:] + [pre_lines[at]]
        placements.append((at, moved))
    return placements


def _first_admitting(rule, pre_lines: list[str], old_line: str, added: list[str], region):
    """None if `rule(after, lines)` admits some equivalent placement inside `region`."""
    first_refusal = None
    for after, lines in _placements(pre_lines, int(old_line), added):
        if not region[1] - 1 <= after <= region[2]:
            continue
        refusal = rule(after, lines)
        if refusal is None:
            return None
        first_refusal = first_refusal or refusal
    return first_refusal or "no placement of the insertion lies inside the body"


def _append_refusal(rule: tuple[str | None, str], region, pre_lines: list[str],
                    old_line: str, old_count: str, added: list[str]) -> str | None:
    """None when a hunk is a pure-insertion APPEND at the tail of an append-only body."""
    if old_count != "0":
        return ("only a pure-insertion append is admitted here (an existing line is "
                "changed or removed)")
    return _first_admitting(lambda after, lines: _append_placement_refusal(
        rule, region, pre_lines, after, lines), pre_lines, old_line, added, region)


def _append_placement_refusal(rule, region, pre_lines: list[str], after: int,
                              added: list[str]) -> str | None:
    tail, entry = rule
    _label, first, last = region
    stop = last + 1
    if tail is not None:
        stop = next((n for n in range(first, last + 1) if re.search(tail, pre_lines[n - 1])),
                    None)
        if stop is None:
            return f"tail marker `{tail}` is not in the body"
    if after >= stop:
        return "the insertion is past the body's tail marker"
    if any(code.strip() for code in _code_lines(pre_lines[after:stop - 1])):
        return ("not an append at the tail: existing entries follow the insertion point "
                "(existing entries may not be removed, reordered or edited)")
    for line, code in zip(added, _code_lines(added)):
        if code.strip() and not re.match(entry, line):
            return f"appended line is not an admitted entry ({line.strip()[:80]!r})"
    return None


_CASE_LABEL = re.compile(r"^\s*(?:case\b[^:]*:|default\s*:)")


def _case_insert_refusal(region, pre_lines: list[str], old_line: str, old_count: str,
                         added: list[str], case_labels: frozenset[str]) -> str | None:
    """None when a hunk inserts only NEW case blocks, at a case boundary, labelled with
    enumerators this patch appends: no existing case gains code, loses code or falls
    through differently."""
    if old_count != "0":
        return "only a NEW case block may be inserted here (an existing line is changed or removed)"
    head_codes = _code_lines(pre_lines)
    return _first_admitting(lambda after, lines: _case_placement_refusal(
        region, head_codes, after, lines, case_labels), pre_lines, old_line, added, region)


def _case_placement_refusal(region, head_codes: list[str], after: int, added: list[str],
                            case_labels: frozenset[str]) -> str | None:
    _label, first, last = region
    following = next((head_codes[n - 1] for n in range(after + 1, last + 2)
                      if head_codes[n - 1].strip()), "")
    preceding = next((head_codes[n - 1] for n in range(after, first - 2, -1)
                      if n >= 1 and head_codes[n - 1].strip()), "")
    if not _CASE_LABEL.match(following):
        return "a new case block must be inserted directly before an existing case/default label"
    if not re.search(r"\bbreak\s*;|\breturn\b[^;]*;|\{\s*$", preceding):
        return ("the line before the insertion neither ends a case (`break;`/`return`) nor "
                "opens the switch")
    codes = [code for code in _code_lines(added) if code.strip()]
    if not codes or not re.match(r"^\s*case\s+\w+\s*:", codes[0]):
        return "the inserted text must start with a `case` label"
    joined = "\n".join(codes)
    if re.search(r"\bdefault\s*:", joined):
        return "no `default` label may be added"
    labels = re.findall(r"\bcase\s+(\w+)\s*:", joined)
    unknown = sorted(set(labels) - set(case_labels))
    if unknown:
        return (f"case label(s) {unknown} are not enumerators this patch appends to enum "
                "ggml_op (an existing op's dispatch must not change)")
    if joined.count("{") != joined.count("}"):
        return "the inserted case block is not brace-balanced"
    if not re.search(r"\bbreak\s*;|\breturn\b", codes[-1]):
        return ("the inserted case block must end in `break;` or `return` (no fallthrough "
                "into the next case)")
    return None


def _line_rewrite_ok(route: CpuSourceRoute, removed: list[str], added: list[str]) -> bool:
    """A hunk that only rewrites whole lines `route.line_rewrites` admits, one for one."""
    if not removed or len(removed) != len(added):
        return False
    return all(any(re.fullmatch(old, gone.strip()) and re.fullmatch(new, came.strip())
                   for old, new in route.line_rewrites)
               for gone, came in zip(removed, added))


def _cpu_route_scope_refusal(route: CpuSourceRoute, source_text: str | None,
                             pre_source_text: str | None,
                             patch_text: str | None, *,
                             case_labels: frozenset[str] = frozenset()) -> str | None:
    """None when every -U0 hunk lies inside one admitted body on both sides (or is an
    admitted new helper/definition or line rewrite), obeying the body's own rule."""
    if not source_text or not pre_source_text or not patch_text:
        return "HEAD source, candidate source or the -U0 patch is empty"
    old = _cpu_route_bounds(pre_source_text, "HEAD", route)
    if isinstance(old, str):
        return old
    new = _cpu_route_bounds(source_text, "candidate", route)
    if isinstance(new, str):
        return new
    if old[1] != new[1]:
        return "an admitted header (marker through opening brace) differs; only bodies may change"
    hunks = _parse_hunks(patch_text)
    if not hunks:
        return "the -U0 patch has no hunks"
    pre_lines = pre_source_text.splitlines()
    appends = {label: (tail, entry) for label, tail, entry in route.append_bodies}

    def within(line: str, count: str, region) -> bool:
        first, size = int(line), int(count) if count else 1
        if count == "0":
            # -U0 pure insertion AFTER line `first`: inside the body when it follows the
            # opening-brace line (region[1] - 1) up to the last body line.
            return region[1] - 1 <= first <= region[2]
        return region[1] <= first and first + max(size, 1) - 1 <= region[2]

    labels: list[str | None] = []
    for old_line, old_count, new_line, new_count, removed, added in hunks:
        where = f"@@ -{old_line},{old_count or 1} +{new_line},{new_count or 1} @@"
        index = next((i for i in range(len(old[0]))
                      if within(old_line, old_count, old[0][i]) and
                      within(new_line, new_count, new[0][i])), None)
        if index is not None:
            region = old[0][index]
            label = region[0]
            refusal = (_append_refusal(appends[label], region, pre_lines, old_line,
                                       old_count, added) if label in appends else
                       _case_insert_refusal(region, pre_lines, old_line, old_count, added,
                                            case_labels) if label in route.case_bodies else
                       None)
            if refusal is not None:
                return f"hunk {where} in {label}: {refusal}"
            labels.append(label)
            continue
        if (route.new_helpers or route.new_definitions) and old_count == "0":
            helper = _new_helper_refusal(pre_source_text, source_text, int(old_line),
                                         int(new_line), int(new_count or 1), route)
            if helper is None:
                labels.append(None)
                continue
            what = ("a new file-scope helper" if route.new_helpers and not route.new_definitions
                    else "an admitted new file-scope helper/definition")
            return (f"hunk @@ -{old_line},0 +{new_line},{new_count or 1} @@ lies outside "
                    f"every admitted body and is not {what}: {helper}")
        if route.line_rewrites and _line_rewrite_ok(route, removed, added):
            labels.append(None)
            continue
        admitted = ", ".join(f"{label} {first}-{last}" for label, first, last in old[0])
        return (f"hunk {where} lies outside every admitted body (HEAD lines: "
                f"{admitted or 'none'})")
    if route.forbidden_added is not None:
        pattern = re.compile(route.forbidden_added)
        for hunk, label in zip(hunks, labels):
            if label is not None and label in route.forbidden_exempt_bodies:
                continue
            for line in hunk[5]:
                if pattern.search("+" + line):
                    return (f"an added line matches the forbidden pattern "
                            f"`{route.forbidden_added}` ({line.strip()[:120]!r})")
    if route.required_added is not None:
        regex, why = route.required_added
        if not any(re.search(regex, code) for hunk in hunks for code in _code_lines(hunk[5])):
            return why
    return None


_HELPER_START = re.compile(
    r"^(?:static\b|inline\s+static\b|template\s*<[^>]*>\s*static\b|"
    r"\[\[[\w:, ]+\]\]\s*static\b)")
_HELPER_INCLUDE = re.compile(r"^#\s*include\s*<[\w./+-]+>\s*$")
_HELPER_CONDITIONAL = re.compile(r"^#\s*(if|ifdef|ifndef|elif|else|endif)\b")
_EXTERN_C_OPEN = re.compile(r'^\s*extern\s+"C"\s*\{')


def _new_helper_refusal(pre_text: str, post_text: str, old_line: int, new_line: int,
                        count: int, route: CpuSourceRoute | None = None) -> str | None:
    """None when a pure-insertion hunk adds only NEW file-scope static helpers.

    The insertion point must be at file scope in HEAD (brace depth 0 after HEAD line
    `old_line`, outside any comment; an enclosing `extern "C" {` counts as file scope).
    Every top-level statement of the inserted text must be an `#include <...>` line or
    start with `static` (functions, constants, tables) -- or, where the route names
    `new_definitions`, be a NON-static definition/prototype of a name HEAD's file never
    mentions --, the text must be brace-balanced at file scope, and nothing else may
    appear: no `#define`/`#undef`/`#pragma` (a macro placed above existing code would
    change that code without touching it), no non-static global or definition of an
    existing symbol (those would change ABI or shadow HEAD code).

    A diff may anchor the same insertion a few lines up or down where the inserted text
    and its neighbours repeat (a helper ending in `}` placed after a function that also
    ends in `}`), so every equivalent placement is tried and any valid one admits."""
    pre_lines, post_lines = pre_text.splitlines(), post_text.splitlines()
    block = post_lines[new_line - 1:new_line - 1 + count]
    if len(block) != count:
        return "inserted lines are not in the candidate"
    first_refusal = None
    for at, lines in _placements(pre_lines, old_line, block):
        refusal = _helper_block_refusal(pre_lines, at, lines, route, pre_text)
        if refusal is None:
            return None
        first_refusal = first_refusal or refusal
    return first_refusal


def _helper_block_refusal(pre_lines: list[str], after: int, added: list[str],
                          route: CpuSourceRoute | None = None,
                          pre_text: str = "") -> str | None:
    static_ok = route is None or route.new_helpers
    definitions = re.compile(route.new_definitions) if route and route.new_definitions else None
    stack, in_block = [], False
    for line in pre_lines[:after]:
        code, in_block = _strip_code_line(line, in_block)
        extern_c = bool(_EXTERN_C_OPEN.match(line))
        for ch in code:
            if ch == "{":
                stack.append(extern_c)
                extern_c = False
            elif ch == "}" and stack:
                stack.pop()
    if in_block or not all(stack):
        return f"HEAD line {after} is not at file scope (depth {len(stack)})"
    depth, in_block, in_statement, conditional = 0, False, False, 0
    prototype = False
    for line in added:
        code, in_block = _strip_code_line(line, in_block)
        text = code.strip()
        directive = _HELPER_CONDITIONAL.match(text)
        if directive:
            # #if/#ifdef/#else/#endif (e.g. an __AVX512F__ variant) must balance inside
            # the inserted text, so it can never swallow or reshape HEAD code after it.
            conditional += {"if": 1, "ifdef": 1, "ifndef": 1, "endif": -1}.get(
                directive.group(1), 0)
            if conditional < 0:
                return "inserted #endif closes a conditional it did not open"
            continue
        if depth == 0 and not in_statement and text:
            if text.startswith("#"):
                if not (static_ok and _HELPER_INCLUDE.match(text)):
                    return (f"only `#include <...>` and balanced #if/#endif lines may be "
                            f"added at file scope ({text[:80]!r})")
                continue
            if static_ok and _HELPER_START.match(text):
                prototype = False
            elif definitions is not None and definitions.match(text):
                name = re.search(r"\b([A-Za-z_]\w*)\s*\(", text)
                if name is None:
                    return f"cannot name the new definition ({text[:80]!r})"
                if re.search(rf"\b{re.escape(name.group(1))}\b", pre_text):
                    return (f"`{name.group(1)}` already occurs in HEAD's file; only NEW "
                            "names may be defined or declared")
                prototype = bool(route.declarations_only)
            elif definitions is not None and not static_ok:
                return (f"a file-scope statement here must match `{route.new_definitions}` "
                        f"({text[:80]!r})")
            else:
                return ("a file-scope statement must start with `static` (a new helper, "
                        f"constant or table) ({text[:80]!r})")
            in_statement = True
        elif text.startswith("#"):
            return f"preprocessor line inside a new helper ({text[:80]!r})"
        if prototype and ("{" in code or "}" in code):
            return f"only prototypes may be added to this header ({text[:80]!r})"
        depth += code.count("{") - code.count("}")
        if depth < 0:
            return "inserted text closes a scope it did not open"
        if depth == 0 and in_statement and (text.endswith(";") or text.endswith("}")):
            in_statement = False
    if depth != 0 or in_statement or in_block:
        return "inserted text is not a complete, brace-balanced file-scope declaration"
    if conditional:
        return "inserted #if/#ifdef is not closed inside the inserted text"
    return None


# ---------------------------------------------------------------- multi-file routes
#
# 2026-10-04 (structural seeds 1 and 6: model-specific fused ops and load-time weight
# packing; R3: the plan-time graph_optimize hook). A model-level mutation spans files
# no single-file route can name: the op enum, its constructor and name tables, the CPU
# kernel and its dispatch, the model's builders. Each file keeps a body-scoped
# `CpuSourceRoute` component; the route admits a patch only when every changed path is
# one of its components and every component admits its own hunks.

@dataclass(frozen=True)
class CpuMultiFileRoute:
    route: str
    symbols: tuple[str, ...]
    files: tuple[CpuSourceRoute, ...]
    ops: tuple[str, ...]
    admitted_text: str = ""
    #: Paths the patch must change (the model-specific file a model route exists for).
    required_paths: tuple[str, ...] = ()
    #: Paths named with the reason they are refused (a shared builder).
    refused_paths: tuple[tuple[str, str], ...] = ()
    #: The header whose appended `GGML_OP_*` enumerators may label new dispatch cases.
    enum_source: str | None = None
    #: (regex, refusal) over the added code of every file.
    required_added: tuple[str, str] | None = None
    model_identity: bool = True
    identity_targets: str = "own"
    identity_repeats: int = 1
    identity_arch: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        """The route's primary file (the witness checks it exists in the candidate)."""
        return self.files[0].path

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(component.path for component in self.files)


_GRAPH_SCHED = next(route for route in CPU_SOURCE_ROUTES if route.route == "cpu_graph_sched")
_FIELD_ENTRY = r"^[^{}#]*;\s*(//.*)?$"
_OP_NAME_ENTRY = r'^\s*"[^"\\]*",\s*(//.*)?$'
_OP_COUNT_ASSERT = (r'static_assert\(GGML_OP_COUNT == \d+, "GGML_OP_COUNT != \d+"\);',
                    r'static_assert\(GGML_OP_COUNT == (\d+), "GGML_OP_COUNT != \1"\);')

CPU_MULTI_FILE_ROUTES = (
    # MODEL-SPECIFIC FUSED OPS (R2; structural seed 1 "port DS41's HC_PRE/COMB/POST
    # fused-op pattern to qwen4exp", seed 6 "pack same-input projections at load").
    # Operator 2026-10-04: model-specific kernels are acceptable. Template: the DSV4_HC
    # ops (enum append, ggml_dsv4_hc_*() constructors + name tables, ops.cpp forward,
    # ggml-cpu.c dispatch/n_tasks cases). Every existing op, field and enumerator is
    # untouched by construction (appends, new names and new-op cases only), so the gate
    # is the whole model: greedy identity vs the anchor on a target that serves a
    # qwen4exp GGUF, plus every peer target when a shared path (ggml/, llama-model.h) is
    # touched -- the lane binding's cross-target serving A/B then also runs at keep time.
    CpuMultiFileRoute(
        route="cpu_model_fused_op",
        symbols=("load_arch_tensors", "build_hc_mix", "build_hc_combine", "graph",
                 "build_qkvz", "build_layer_attn_linear", "build_layer_ffn",
                 "llama_model_qwen4exp", "qwen4exp"),
        files=(
            CpuSourceRoute(
                route="cpu_model_fused_op", path="src/models/qwen4exp.cpp", symbols=(),
                bodies=(("load_arch_tensors",
                         "void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {"),
                        ("build_hc_mix", "ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix("),
                        ("build_hc_combine",
                         "ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine("),
                        ("graph", "llama_model_qwen4exp::graph::graph("),
                        ("build_qkvz", "std::pair<ggml_tensor *, ggml_tensor *> "
                                       "llama_model_qwen4exp::graph::build_qkvz("),
                        ("build_layer_attn_linear",
                         "ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear("),
                        ("build_layer_ffn",
                         "ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(")),
                ops=(), new_helpers=True),
            CpuSourceRoute(
                route="cpu_model_fused_op", path="src/llama-model.h", symbols=(),
                bodies=(("llama_layer", "struct llama_layer {"),), ops=(),
                append_bodies=(("llama_layer", None, _FIELD_ENTRY),)),
            CpuSourceRoute(
                route="cpu_model_fused_op", path="ggml/include/ggml.h", symbols=(),
                bodies=(("ggml_op", "    enum ggml_op {"),), ops=(),
                append_bodies=(("ggml_op", r"^\s*GGML_OP_COUNT\s*,",
                                r"^\s*GGML_OP_[A-Z0-9_]+\s*,\s*(//.*)?$"),),
                new_definitions=r"^GGML_API\s+struct\s+ggml_tensor\s*\*\s*ggml_\w+\s*\(",
                declarations_only=True),
            CpuSourceRoute(
                route="cpu_model_fused_op", path="ggml/src/ggml.c", symbols=(),
                bodies=(("GGML_OP_NAME", "static const char * GGML_OP_NAME[GGML_OP_COUNT] = {"),
                        ("GGML_OP_SYMBOL",
                         "static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {")),
                ops=(), new_helpers=True,
                append_bodies=(("GGML_OP_NAME", None, _OP_NAME_ENTRY),
                               ("GGML_OP_SYMBOL", None, _OP_NAME_ENTRY)),
                new_definitions=r"^struct\s+ggml_tensor\s*\*\s*ggml_\w+\s*\(",
                line_rewrites=(_OP_COUNT_ASSERT,)),
            CpuSourceRoute(
                route="cpu_model_fused_op", path="ggml/src/ggml-cpu/ops.h", symbols=(),
                bodies=(), ops=(),
                new_definitions=r"^void\s+ggml_compute_forward_\w+\s*\(",
                declarations_only=True),
            CpuSourceRoute(
                route="cpu_model_fused_op", path="ggml/src/ggml-cpu/ops.cpp", symbols=(),
                bodies=(), ops=(), new_helpers=True,
                new_definitions=r"^void\s+ggml_compute_forward_\w+\s*\("),
            CpuSourceRoute(
                route="cpu_model_fused_op", path="ggml/src/ggml-cpu/ggml-cpu.c", symbols=(),
                bodies=(("ggml_compute_forward",
                         "static void ggml_compute_forward(struct ggml_compute_params * params, "
                         "struct ggml_tensor * tensor) {"),
                        ("ggml_get_n_tasks",
                         "static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {"),
                        ("ggml_graph_plan", "struct ggml_cplan ggml_graph_plan(")),
                ops=(),
                case_bodies=("ggml_compute_forward", "ggml_get_n_tasks", "ggml_graph_plan"))),
        ops=(),
        required_paths=("src/models/qwen4exp.cpp",),
        refused_paths=(("src/llama-graph.cpp",
                        "llama-graph.cpp (build_moe_ffn and every shared builder) serves "
                        "every model; a model-specific fused op is built in "
                        "src/models/qwen4exp.cpp"),),
        enum_source="ggml/include/ggml.h",
        model_identity=True,
        identity_targets="shared_peers",
        identity_arch=("qwen4exp",),
        admitted_text=(
            "src/models/qwen4exp.cpp hunks inside load_arch_tensors / build_hc_mix / "
            "build_hc_combine / graph::graph / build_qkvz / build_layer_attn_linear / "
            "build_layer_ffn plus NEW static helpers (this file must change); field appends "
            "at the tail of struct llama_layer (src/llama-model.h); GGML_OP_* appends "
            "directly before GGML_OP_COUNT and NEW `GGML_API struct ggml_tensor * "
            "ggml_<op>(...);` prototypes (ggml.h); NEW `struct ggml_tensor * ggml_<op>(` "
            "constructors, static helpers, GGML_OP_NAME/GGML_OP_SYMBOL tail appends and "
            "the two GGML_OP_COUNT static_assert lines (ggml.c); NEW "
            "`ggml_compute_forward_<op>` prototypes (ops.h) and definitions plus static "
            "helpers (ops.cpp); NEW `case GGML_OP_<appended>:` blocks ending in break; at a "
            "case boundary of ggml_compute_forward / ggml_get_n_tasks / ggml_graph_plan "
            "(ggml-cpu.c). Nothing else: no existing op, field, enumerator or function "
            "changes, and src/llama-graph.cpp (build_moe_ffn) is refused. Gate: greedy "
            "identity vs the anchor on a qwen4exp target, plus every peer target when a "
            "shared path is touched")),
    # PLAN-TIME GRAPH REWRITE HOME (R3, optional alternative for seeds 3/4B): replace the
    # CPU backend's NULL `graph_optimize` slot with a NEW static hook that annotates
    # co-group runs once per graph shape (a side table or op_params tail); the walk that
    # consumes the annotation is the cpu_graph_sched component. Same gate as R1.
    CpuMultiFileRoute(
        route="cpu_graph_optimize",
        symbols=("graph_optimize", "ggml_backend_cpu_graph_optimize"),
        files=(
            CpuSourceRoute(
                route="cpu_graph_optimize", path="ggml/src/ggml-cpu/ggml-cpu.cpp", symbols=(),
                bodies=(), ops=(), new_helpers=True,
                line_rewrites=((r"/\* \.graph_optimize\s+= \*/ NULL,",
                                r"/\* \.graph_optimize\s+= \*/ [A-Za-z_]\w*,"),)),
            dataclasses.replace(_GRAPH_SCHED, route="cpu_graph_optimize", required_added=None)),
        ops=_GRAPH_SCHED.ops,
        required_paths=("ggml/src/ggml-cpu/ggml-cpu.cpp",),
        required_added=_GRAPH_SCHED.required_added,
        model_identity=True,
        identity_targets=_GRAPH_SCHED.identity_targets,
        identity_repeats=_GRAPH_SCHED.identity_repeats,
        admitted_text=(
            "ggml/src/ggml-cpu/ggml-cpu.cpp: NEW static helpers (the hook) and the one "
            "`/* .graph_optimize = */ NULL,` slot rewritten to name one (this file must "
            "change); ggml/src/ggml-cpu/ggml-cpu.c: everything cpu_graph_sched admits. "
            "The change must honour use_ref. Gate: as cpu_graph_sched")),
)
CPU_ROUTE_PATHS_ALL = tuple(sorted({*CPU_SOURCE_ROUTE_PATHS,
                                    *(path for route in CPU_MULTI_FILE_ROUTES
                                      for path in route.paths),
                                    *(path for route in CPU_MULTI_FILE_ROUTES
                                      for path, _why in route.refused_paths)}))


def cpu_multi_file_routes(target_symbol: str) -> tuple[CpuMultiFileRoute, ...]:
    """Every multi-file route the target symbol names, in table order."""
    names = [target_symbol, *_route_symbol_names(target_symbol)]
    return tuple(route for route in CPU_MULTI_FILE_ROUTES
                 if any(name in route.symbols for name in names))


def _multi_route_refusal(route: CpuMultiFileRoute, paths, file_texts) -> str | None:
    components = {component.path: component for component in route.files}
    refused = dict(route.refused_paths)
    for path in paths:
        if path in refused:
            return f"{path} is refused: {refused[path]}"
        if path not in components:
            return f"{path} is not one of this route's files ({', '.join(components)})"
    missing = [path for path in route.required_paths if path not in paths]
    if missing:
        return f"the patch must change {', '.join(missing)}"
    texts = file_texts or {}
    labels: set[str] = set()
    # Files without case bodies first: the enum header's appended enumerators are the
    # only labels a new dispatch case may carry.
    for path in sorted(paths, key=lambda p: (bool(components[p].case_bodies), p)):
        source, pre, patch = texts.get(path, (None, None, None))
        refusal = _cpu_route_scope_refusal(components[path], source, pre, patch,
                                           case_labels=frozenset(labels))
        if refusal is not None:
            return f"{path}: {refusal}"
        if path == route.enum_source:
            labels |= {match for hunk in _parse_hunks(patch)
                       for line in hunk[5]
                       for match in re.findall(r"^\s*(GGML_OP_[A-Z0-9_]+)\s*,", line)}
    if route.required_added is not None:
        regex, why = route.required_added
        if not any(re.search(regex, code)
                   for path in paths
                   for hunk in _parse_hunks(texts.get(path, (None, None, ""))[2] or "")
                   for code in _code_lines(hunk[5])):
            return why
    return None


def admit_cpu_multi_route(paths, target_symbol: str, file_texts
                          ) -> tuple[CpuMultiFileRoute | None, str | None]:
    """(multi-file route that admits the patch, None) or (first named route, refusals).

    (None, None) when no multi-file route names the target symbol."""
    routes = cpu_multi_file_routes(target_symbol)
    refusals = []
    for route in routes:
        refusal = _multi_route_refusal(route, tuple(paths), file_texts)
        if refusal is None:
            return route, None
        refusals.append((route, refusal))
    if not refusals:
        return None, None
    if len(refusals) == 1:
        return refusals[0][0], refusals[0][1]
    return refusals[0][0], " | ".join(
        f"{route.route}: {refusal}. Admitted: {route.admitted_text}"
        for route, refusal in refusals)


def affected_op_scope(paths: tuple[str, ...], *, target_surface: str,
                      target_symbol: str, source_text: str | None = None,
                      patch_text: str | None = None,
                      pre_source_text: str | None = None,
                      file_texts: dict | None = None) -> tuple[str, ...] | Verdict:
    """Resolve known changed-source routes; never inherit MUL_MAT by default.

    Paths are read from Git by the owner, not taken from the actor's response.
    Unknown/shared edits must acquire a native op map and reference before timing.
    `file_texts` maps each changed path to (candidate text, HEAD text, -U0 patch) for
    the multi-file routes (`CPU_MULTI_FILE_ROUTES`); a single changed file falls back
    to `source_text`/`pre_source_text`/`patch_text`.
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
    if changed == {"ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"} and \
            target_symbol == "mul_mat_qX_K_q8_2_X4_T":
        refusal = _iqk_q45_dot_scope_refusal(source_text, pre_source_text, patch_text)
        if refusal is None:
            # Both quants and every nrc_y specialization are checked by the
            # independent scalar suite plus exact candidate-DSO dot hits below.
            return ("MUL_MAT", "MUL_MAT_ID")
        # The route exists; name the rule that refused THIS patch. The generic
        # IQK text below told the author nothing it could act on (run 9c).
        return Verdict("op_scope", False,
                       "CPU IQK Q4_K/Q5_K dot route refused before build: " + refusal +
                       ". Admitted: hunks inside the DequantizerQ4K_AVX2/DequantizerQ5K_AVX2 "
                       "bodies, the unpack_q4_scales helpers where HEAD has them, or the "
                       "mul_mat_qX_K_q8_2_X4_T body; signatures, Q4Bits_AVX2 and Q6_K unchanged")
    if len(changed) == 1:
        route, refusal = admit_cpu_route(next(iter(changed)), target_symbol, source_text,
                                         pre_source_text, patch_text)
        if route is not None:
            if refusal is None:
                return route.ops
            if len(cpu_source_routes(next(iter(changed)), target_symbol)) > 1:
                return Verdict("op_scope", False,
                               f"CPU {route.route} route refused before build (and every "
                               f"other route on this body): {refusal}")
            return Verdict("op_scope", False,
                           f"CPU {route.route} route refused before build: {refusal}. "
                           f"Admitted: {route.admitted_text}")
    # Multi-file routes (2026-10-04): a model-level mutation (new op + kernel + dispatch +
    # builder) or the plan-time hook. Named by the target symbol; the target surface must
    # be one of the route's files.
    multi = [route for route in cpu_multi_file_routes(target_symbol)
             if target_surface in route.paths]
    if multi:
        if file_texts is None and len(changed) == 1:
            file_texts = {next(iter(changed)): (source_text, pre_source_text, patch_text)}
        route, refusal = admit_cpu_multi_route(tuple(sorted(changed)), target_symbol,
                                               file_texts)
        if route is not None:
            if refusal is None:
                return route.ops
            return Verdict("op_scope", False,
                           f"CPU {route.route} route refused before build: {refusal}. "
                           f"Admitted: {route.admitted_text}")
    if any(path.startswith("ggml/src/ggml-cpu/iqk/") for path in changed):
        return Verdict("op_scope", False,
                       f"CPU IQK source refused before build (paths {sorted(changed)}, "
                       f"target_symbol {target_symbol!r} has no admitted route): "
                       "the selected MUL_MAT/MUL_MAT_ID "
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
    elif target_symbol == "mul_mat_qX_K_q8_2_X4_T":
        result = iqk_witness.check_q45_dot(build_dir, resolved_recipe=resolved_recipe,
                                            source_root=source_root)
    else:
        return Verdict("oracle_unavailable", False,
                       "unsupported CPU IQK helper has no independent reference")
    return Verdict("reference_comparison" if result.status == "wrong" else
                   "oracle_unavailable" if result.status == "unavailable" else
                   "reference_comparison", result.status == "pass",
                   result.reason, result.detail)


def check_cpu_route_reference(build_dir: Path, source_root: Path, *, resolved_recipe,
                              path: str, target_symbol: str,
                              route_name: str | None = None) -> Verdict:
    """Independent reference for a widened CPU route (see `CPU_SOURCE_ROUTES`).

    `route_name` is the route `admit_cpu_route` admitted the patch under; it must be
    one the (path, target symbol) pair names. Without it the first named route is used
    (single-route bodies)."""
    from . import cpu_route_witness

    routes = cpu_source_routes(path, target_symbol)
    route = (next((r for r in routes if r.route == route_name), None)
             if route_name is not None else (routes[0] if routes else None))
    if route is None and route_name is not None:
        # A multi-file route (2026-10-04) is named by the target symbol, not the path.
        route = next((r for r in cpu_multi_file_routes(target_symbol)
                      if r.route == route_name), None)
    if route is None:
        return Verdict("oracle_unavailable", False,
                       "CPU source route has no reviewed independent reference")
    result = cpu_route_witness.check(build_dir, resolved_recipe=resolved_recipe,
                                     source_root=source_root, route=route.route,
                                     source_path=route.path)
    return Verdict("reference_comparison" if result.status == "wrong" else
                   "oracle_unavailable" if result.status == "unavailable" else
                   "reference_comparison", result.status == "pass",
                   result.reason, result.detail)


def check_model_output_identity(*, anchor_recipe, candidate_recipe, requests,
                                window=None, repeats: int = 1) -> Verdict:
    """Whole-model gate for a `model_identity` route (`cpu_weight_placement`)."""
    from . import model_identity

    result = model_identity.check(anchor_recipe=anchor_recipe,
                                  candidate_recipe=candidate_recipe,
                                  requests=tuple(requests or ()), window=window,
                                  repeats=repeats)
    return Verdict("reference_comparison" if result.status != "unavailable" else
                   "oracle_unavailable", result.status == "pass",
                   result.reason, result.detail)


def check_model_identity_targets(targets, *, window=None, repeats: int = 1,
                                 required_arch: tuple[str, ...] = (),
                                 architecture=None) -> Verdict:
    """Whole-model identity on every target of a route (2026-10-04 structural routes).

    `targets` is [(label, anchor recipe, candidate recipe, frozen requests)]: this lane's
    own target first, then the lane binding's peer targets when the route asks for them
    (`CpuSourceRoute.identity_targets`). Every target must pass; any `wrong` is a
    verdict, otherwise any `unavailable` makes the gate unavailable. `required_arch`: at
    least one target must serve a GGUF of one of these architectures, or the edited
    model-specific code would never run under the gate (unavailable, not a pass)."""
    from . import model_identity

    targets = tuple(targets)
    if not targets:
        return Verdict("oracle_unavailable", False, "model identity gate has no target")
    if required_arch:
        read = architecture or model_identity.model_architecture
        archs = {label: read(anchor) for label, anchor, _candidate, _requests in targets}
        if not any(arch in required_arch for arch in archs.values()):
            return Verdict("oracle_unavailable", False,
                           f"no identity target serves a {'/'.join(required_arch)} model "
                           f"(targets: {archs}); the edited model code would never run "
                           "under the gate -- run this route on a lane bound to that model "
                           "or with it as a peer")
    rows, wrong, unavailable = [], [], []
    for label, anchor, candidate, requests in targets:
        result = model_identity.check(anchor_recipe=anchor, candidate_recipe=candidate,
                                      requests=tuple(requests or ()), window=window,
                                      repeats=repeats)
        rows.append({"target": label, "status": result.status, "reason": result.reason,
                     "detail": result.detail[:600]})
        if result.status == "wrong":
            wrong.append(f"{label}: {result.reason}")
        elif result.status != "pass":
            unavailable.append(f"{label}: {result.reason}")
    detail = json.dumps(rows)
    if wrong:
        return Verdict("reference_comparison", False, "; ".join(wrong), detail)
    if unavailable:
        return Verdict("oracle_unavailable", False, "; ".join(unavailable), detail)
    return Verdict("reference_comparison", True,
                   f"greedy identity on {len(targets)} target(s) "
                   f"({', '.join(label for label, *_ in targets)})"
                   + (f", each request served {repeats}x byte-identical" if repeats > 1
                      else ""), detail)


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
    # `-o` takes op_desc names: UNARY/GLU expand to their sub-ops (DS41-C96).
    argv = [str(binary), "test", "-o", backend_ops_selector(op), "-b", backend, "-j", "1"]
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
                       f"test-backend-ops did not prove a nonempty {backend} op suite "
                       f"for {op}; this is a harness fault, NOT evidence about the patch",
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
                    if case.op in backend_ops_selector(op).split(",")
                    and case.status != "not_supported"]
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


__all__ = ["BACKEND_OPS_SELECTORS", "BUILD_TIMEOUT_S", "CORRECTNESS_TIMEOUT_S",
           "CPU_SOURCE_ROUTES",
           "CPU_SOURCE_ROUTE_PATHS", "DEFAULT_TARGETS",
           "PROMOTION_TARGETS", "UNSELECTABLE_ON_ANCHOR", "Verdict",
           "backend_ops_selector", "compiles",
           "cpu_source_route", "deterministic",
           "affected_op_scope", "check_cpu_gdn_reference", "check_cpu_iqk_reference",
           "check_cpu_route_reference", "no_fallback_dispatch",
           "op_correctness", "run_all"]
