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
_DS41_SYNC_OPS = ("MUL_MAT", "MUL_MAT_ID", "ADD", "MUL", "RMS_NORM", "SCALE", "CLAMP",
                  "CONT", "CPY", "CONCAT", "GLU", "UNARY", "SUM_ROWS", "GET_ROWS",
                  "SET_ROWS", "ROPE", "SOFT_MAX", "ARGSORT", "FLASH_ATTN_EXT")


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
    # quant suite (it runs through the candidate's barriers) plus every DS41 op suite.
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
        admitted_text=("hunks inside the ggml_barrier, ggml_cpu_node_is_solo, "
                       "ggml_cpu_try_fuse_ops or ggml_graph_compute_thread bodies; headers, "
                       "globals, op kernels and every other function unchanged")),
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
    parts = [part.strip() for part in text.split("::") if part.strip()]
    return list(dict.fromkeys(reversed(parts)))


def cpu_source_route(path: str, target_symbol: str) -> CpuSourceRoute | None:
    """The widened route a (single path, target symbol) pair names, if any."""
    candidates = [route for route in CPU_SOURCE_ROUTES if route.path == path]
    exact = next((route for route in candidates if target_symbol in route.symbols), None)
    if exact is not None:
        return exact
    for name in _route_symbol_names(target_symbol):
        route = next((route for route in candidates if name in route.symbols), None)
        if route is not None:
            return route
    return None


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


def _cpu_route_scope_refusal(route: CpuSourceRoute, source_text: str | None,
                             pre_source_text: str | None,
                             patch_text: str | None) -> str | None:
    """None when every -U0 hunk lies inside one admitted body on both sides."""
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
    if route.forbidden_added is not None:
        pattern = re.compile(route.forbidden_added)
        for line in patch_text.splitlines():
            if line.startswith("+") and not line.startswith("+++") and pattern.search(line):
                return (f"an added line matches the forbidden pattern `{route.forbidden_added}` "
                        f"({line[1:].strip()[:120]!r})")
    return None


def affected_op_scope(paths: tuple[str, ...], *, target_surface: str,
                      target_symbol: str, source_text: str | None = None,
                      patch_text: str | None = None,
                      pre_source_text: str | None = None) -> tuple[str, ...] | Verdict:
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
        route = cpu_source_route(next(iter(changed)), target_symbol)
        if route is not None:
            refusal = _cpu_route_scope_refusal(route, source_text, pre_source_text, patch_text)
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
                              path: str, target_symbol: str) -> Verdict:
    """Independent reference for a widened CPU route (see `CPU_SOURCE_ROUTES`)."""
    from . import cpu_route_witness

    route = cpu_source_route(path, target_symbol)
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


__all__ = ["BUILD_TIMEOUT_S", "CORRECTNESS_TIMEOUT_S", "CPU_SOURCE_ROUTES",
           "CPU_SOURCE_ROUTE_PATHS", "DEFAULT_TARGETS",
           "PROMOTION_TARGETS", "Verdict", "compiles", "cpu_source_route", "deterministic",
           "affected_op_scope", "check_cpu_gdn_reference", "check_cpu_iqk_reference",
           "check_cpu_route_reference", "no_fallback_dispatch",
           "op_correctness", "run_all"]
