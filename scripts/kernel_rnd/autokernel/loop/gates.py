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
import os
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


def _correctness_detail(backend: str, counts, stdout: str, stderr: str,
                        limit: int = 2000) -> str:
    """A failed-suite detail that survives `Verdict.to_dict`'s 2000-character cut.

    The tally and the stderr FAIL lines come FIRST: `--autokernel-properties` prints
    its HOST PROPERTY / FP64 RATIO / CHECKER ISOLATION failures on stderr, and the
    old `stdout[-2000:] + stderr[-1000:]` detail lost every one of them to the cut
    (ak-27b-gpu-verify-20261004 batch 1: only q4_K case names survived, no reason).
    """
    passed = sum(int(item) for item, _total in counts)
    total = sum(int(item) for _passed, item in counts)
    head = [f"{backend} tally {passed}/{total} tests passed"]
    flagged = [line.strip()[:240] for line in stderr.splitlines() if "FAIL" in line]
    if flagged:
        head.append(f"stderr FAIL lines ({len(flagged)}, last 8):")
        head.extend(flagged[-8:])
    text = "\n".join(head)[:limit // 2]
    remaining = limit - len(text) - 2
    tail = stdout[-max(0, remaining):] if remaining > 0 else ""
    return (text + "\n" + "\n" + tail)[:limit]


def anchor_relative_correctness(candidate: "Verdict", anchor_check) -> "Verdict":
    """A seeded-reference `correctness` refusal counts against the patch only when the
    ANCHOR passes the same gate.

    `anchor_check` runs the identical `op_correctness` invocation on the anchor build
    (callers cache it per anchor/op). If the anchor fails it too, the gate or one of its
    `--autokernel-properties` checks is broken for this build, and the refusal is the
    harness's `oracle_unavailable` (no authoring attempt spent; resumable once the gate
    changes), never wrong-kernel evidence. ak-27b-gpu-verify-20261004 batch 1: both
    Q8_0-guarded stream-k candidates were refused on q4_K MUL_MAT cases their diff
    cannot reach, and nothing had ever run the seeded gate on the anchor.
    """
    if candidate.passed or candidate.gate != "correctness":
        return candidate
    anchor = anchor_check()
    if anchor.passed or anchor.gate != "correctness":
        return candidate
    return Verdict("oracle_unavailable", False,
                   f"the anchor fails the same seeded gate ({candidate.reason}); a gate or "
                   "property-check fault for this build, NOT evidence about the patch",
                   ("anchor: " + anchor.detail)[:990] + "\ncandidate: " + candidate.detail[:990])


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
    #: Also judge greedy identity on the lane's LONG-context manifest when the long
    #: surface provides one (run.py `long_identity_targets`; audit 2026-10-04 C1/C2). An
    #: attention change whose defect only shows at depth is invisible at ~300 tokens.
    long_identity: bool = False
    #: New file-scope `static` OBJECTS (not functions) must be `const`/`constexpr` with a
    #: call-free initializer: a static whose initializer calls anything runs at library
    #: load, outside every gate (audit 2026-10-04: `static int x = (setenv(..), 0);`).
    helper_objects_const: bool = False
    #: "bit_exact" (default: the route's existing reference decides) or "ppl_contract"
    #: (2026-10-06, operator layered-acceptance amendment): the kernel may change
    #: floating-point accumulation order. Whole-model judgement is NEVER byte identity
    #: for these routes; it is the four independently-gated layers in
    #: `ppl_contract_gate` (op-level NMSE on served shapes, wikitext2 |delta ppl|,
    #: production-length coherence/agreement, long-generation canary), ALL required,
    #: fail-closed on any missing or errored layer. A ppl_contract route must never
    #: also be model_identity=True (the two numerics contracts are mutually exclusive).
    numerics: str = "bit_exact"
    #: Labels (from `bodies`) whose marker is legitimately DUPLICATED in HEAD, e.g. a
    #: `#ifdef HAVE_FANCY_SIMD` / `#else` pair defining the same struct/function twice
    #: with different bodies for two mutually-exclusive compiled variants. Normally a
    #: marker occurring more than once REFUSES (ambiguous widening). For a label named
    #: here, every occurrence inside the admitted window is a candidate and the LAST
    #: one (source order) is admitted -- by convention in this tree the later `#else`
    #: arm is the one actually compiled on this host (the `#if` arm uses a disabled
    #: sentinel macro, e.g. `z_HAVE_FANCY_SIMD`, or is the non-fancy-SIMD default).
    #: `route.fence` should still exclude an unrelated far-away duplicate (e.g. the
    #: `__aarch64__` branch of the same file) -- this field only disambiguates
    #: occurrences the fence cannot tell apart.
    body_duplicates: tuple[str, ...] = ()
    #: As `body_duplicates`, but the FIRST occurrence is admitted. Needed where the
    #: duplicate is a `#ifdef HAVE_FANCY_SIMD` / `#else` pair: iqk_config.h defines
    #: HAVE_FANCY_SIMD whenever AVX512F/VNNI/VL/BW/DQ are on, which GGML_NATIVE=ON gives
    #: on this EPYC 9655 host -- the `#if` arm is the compiled one and the `#else` arm is
    #: dead code here (review 2026-10-06; `z_HAVE_FANCY_SIMD` arms are the opposite).
    body_duplicates_first: tuple[str, ...] = ()


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
    # 2026-10-06 low-bit kernel routes (operator: "WE MUST improve lower quant
    # performance"). Numerics: ppl_contract (see `ppl_contract_gate` below) -- the
    # independent per-op reference is the native `use_ref=true` path plus
    # test-backend-ops on the exact served shapes (decode N=1, verify widths 2-5);
    # whole-model judgement is wikitext2 |delta ppl| <= 0.5%, a production-length
    # coherence/agreement floor and a long-generation canary, never byte identity.
    #
    # Each of these four files carries the x86_64 implementation ABOVE a top-level
    # `#else` / `// ------- __aarch64__ -------` arm that redefines several of the
    # same symbols verbatim for NEON (confirmed against the champion tree
    # llama.cpp-experimental-cor-b0ba1d427-20261005, 2026-10-06): every route below
    # fences the window to end before that arm, or the generic single-occurrence
    # marker check refuses EVERY patch to the admitted symbol as "occurs 2 times".
    CpuSourceRoute(
        route="iqk_iquants_dequant",
        path="ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp",
        symbols=("IndexHelperIQ3S", "DequantizerIQ3S", "EvenSignHelper",
                 "mul_mat_qX_K_q8_K_IQ_1", "mul_mat_qX_K_q8_K_IQ_N", "iqk_set_kernels_iquants"),
        fence="// --------------------------------------- __aarch64__ "
              "---------------------------------------------",
        bodies=(("IndexHelperIQ3S", "struct IndexHelperIQ3S {"),
                ("DequantizerIQ3S", "struct DequantizerIQ3S final : public BaseDequantizer<block_iq3_s> {"),
                ("EvenSignHelper", "struct EvenSignHelper {"),
                ("mul_mat_qX_K_q8_K_IQ_1",
                 "static void mul_mat_qX_K_q8_K_IQ_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {"),
                ("mul_mat_qX_K_q8_K_IQ_N",
                 "static void mul_mat_qX_K_q8_K_IQ_N(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {"),
                ("iqk_set_kernels_iquants",
                 "bool iqk_set_kernels_iquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {")),
        # IndexHelperIQ3S is defined TWICE in the x86 window, once under `#ifdef
        # z_HAVE_FANCY_SIMD` (a never-defined sentinel name -- this tree's convention
        # for a disabled variant, see the comment above it: "makes PP ~6% slower")
        # and once in the paired `#else`. The `#else` (plain AVX2) arm is the one
        # actually compiled; it is also the LAST occurrence in source order.
        body_duplicates=("IndexHelperIQ3S",),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        new_helpers=True,
        forbidden_added=r"#pragma omp|_Pragma|#define|#undef|#include \"",
        numerics="ppl_contract",
        admitted_text=("hunks inside the IQ3_S index/sign helpers, DequantizerIQ3S, the IQ_1/IQ_N "
                       "kernel bodies or the iquants kernel-selection switch (x86_64 arm only), "
                       "plus new static helpers; block layouts, iq3s_grid, every other quant's "
                       "dequantizer, the aarch64 arm and the x86 native path unchanged")),
    CpuSourceRoute(
        route="iqk_legacy_iq4nl",
        path="ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp",
        symbols=("IQ4_NL_DequantizerU", "IQ4_NL_DequantizerS", "IQ4_NL_UnpackerU", "IQ4_NL_UnpackerS",
                 "iqk_set_kernels_legacy_quants"),
        fence="// ---------------------------- __aarch64__ "
              "----------------------------------------------",
        bodies=(("IQ4_NL_DequantizerU", "struct IQ4_NL_DequantizerU {"),
                ("IQ4_NL_DequantizerS", "struct IQ4_NL_DequantizerS {"),
                ("IQ4_NL_UnpackerU",
                 "struct IQ4_NL_UnpackerU final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0_1<128>, IQ4_NL_DequantizerU> {"),
                ("IQ4_NL_UnpackerS",
                 "struct IQ4_NL_UnpackerS final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0, IQ4_NL_DequantizerS> {"),
                ("iqk_set_kernels_legacy_quants",
                 "bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {")),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        new_helpers=True,
        numerics="ppl_contract",
        admitted_text=("hunks inside the IQ4_NL dequantizers/unpackers or the legacy "
                       "kernel-selection switch (x86_64 arm only), plus new static helpers; "
                       "Q8_0/Q4_0/Q5_x unpackers, the repack path and the aarch64 arm unchanged")),
    # The constexpr whitelists sit ABOVE the two admitted dispatch bodies (iqk_mmid_dispatch,
    # iqk_dense_dispatch above); a type can only be routed to iqk by editing them. Repack
    # must release the same type (cpu_repack_mmid below) or the type still falls back to
    # the un-repacked generic path despite being iqk-admitted.
    CpuSourceRoute(
        route="iqk_type_whitelist",
        path="ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp",
        symbols=("iqk_typeA_supported", "iqk_weight_uses_q8_k", "iqk_mmid_shape_supported"),
        fence="#else  // iqk not implemented",
        bodies=(("iqk_typeA_supported", "constexpr bool iqk_typeA_supported(int t) {"),
                ("iqk_weight_uses_q8_k", "constexpr bool iqk_weight_uses_q8_k(int t) {"),
                ("iqk_mmid_shape_supported", "constexpr bool iqk_mmid_shape_supported(int weight_type, int64_t n_tokens) {")),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        required_added=(r"static_assert\(",
                        "a whitelist change must carry a matching static_assert (positive or "
                        "negative) proving the new admission/refusal, as every existing case does"),
        # Re-review 2026-10-06: the served-shape oracle covers exactly
        # served_shape_cases.WITNESS_TYPES; a whitelist edit naming any OTHER type could
        # route it into iqk with no served-shape evidence, so it is refused outright.
        # Round-3 review: a NAME check cannot stop `return true;` / `default: return true;`.
        # Every added line must be one of the closed grammar below -- a `case` label of
        # a served-corpus type, a `static_assert`, `return false;`, a comment, or blank --
        # so the admitted set can only change by naming served-corpus types. The MMID
        # shape gate is exempt: it is an AND-ed refusal after iqk_typeA_supported
        # (iqk_dispatch.cpp: `if (!iqk_typeA_supported(tA)) return false;` precedes
        # `if (!iqk_mmid_shape_supported(...)) return false;`), so it cannot admit a type.
        # Round-5 resolution B: fully anchored, one statement per line, no greedy
        # `.*` (round 4 showed `static_assert(.*);` swallowing
        # `static_assert(true); if (...) return true; static_assert(true);`).
        forbidden_added=(r"^\+(?!(?:\s*case\s+GGML_TYPE_(?:Q4_0|Q4_1|Q5_0|Q5_1|Q8_0|IQ4_NL|Q2_K|Q3_K|Q4_K|Q5_K|Q6_K|IQ4_XS|IQ2_XXS|IQ2_XS|IQ2_S|IQ3_XXS|IQ3_S)\s*:\s*(?://(?:.*[^\\\s])?\s*)?"
                         r"|\s*static_assert\([^;]*\);\s*(?://(?:.*[^\\\s])?\s*)?"
                         r"|\s*return\s+false\s*;\s*(?://(?:.*[^\\\s])?\s*)?"
                         r"|\s*(?://(?:.*[^\\\s])?\s*)?)$)"),
        forbidden_exempt_bodies=("iqk_mmid_shape_supported",),
        numerics="ppl_contract",
        admitted_text=("case additions/removals inside the three constexpr whitelists, each with a "
                       "matching static_assert; the dispatch bodies, kernels and the disabled-build "
                       "stub unchanged")),
    CpuSourceRoute(
        route="cpu_repack_mmid",
        path="ggml/src/ggml-cpu/repack.cpp",
        symbols=("ggml_repack_get_optimal_repack_type", "forward_mul_mat_id"),
        bodies=(("ggml_repack_get_optimal_repack_type",
                 "static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {"),
                ("forward_mul_mat_id", "    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {")),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        forbidden_added=r"getenv\(",
        numerics="ppl_contract",
        admitted_text=("hunks inside the repack type-claim (which types repack takes when "
                       "GGML_IQK is on) and the repack MUL_MAT_ID walk; the packed layouts, "
                       "gemv/gemm kernels and the GGML_IQK env read unchanged")),
    # Q6_K/IQ4_XS dequantizers (seed 6 widening): DequantizerQ6K_AVX2's own kernel body
    # (mul_mat_qY_K_q8_2_X4_T, a DIFFERENT template than Q4_K/Q5_K's mul_mat_qX_K_q8_2_X4_T,
    # so this route cannot collide with the existing Q4_K/Q5_K ad-hoc admission) and
    # DequantizerIQ4XS (also defined under `#ifdef HAVE_FANCY_SIMD`/`#else`; UNLIKE
    # IndexHelperIQ3S's `z_` sentinel this is the REAL macro, defined on this AVX512 host,
    # so the FIRST (fancy) occurrence is the compiled one -- review 2026-10-06).
    CpuSourceRoute(
        route="iqk_kquants_q6_iq4xs_dequant",
        path="ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp",
        symbols=("DequantizerQ6K_AVX2", "mul_mat_qY_K_q8_2_X4_T", "DequantizerIQ4XS"),
        fence="// --------------------------------- __aarch64__ --------------------------------------",
        bodies=(("DequantizerQ6K_AVX2",
                 "struct DequantizerQ6K_AVX2 final : public BaseDequantizer<block_q6_K> {"),
                ("mul_mat_qY_K_q8_2_X4_T",
                 "static void mul_mat_qY_K_q8_2_X4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {"),
                ("DequantizerIQ4XS",
                 "struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {")),
        body_duplicates_first=("DequantizerIQ4XS",),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        new_helpers=True,
        numerics="ppl_contract",
        admitted_text=("hunks inside the DequantizerQ6K_AVX2 struct, its mul_mat_qY_K_q8_2_X4_T "
                       "kernel body, or the DequantizerIQ4XS struct (x86_64 HAVE_FANCY_SIMD arm only), "
                       "plus new static helpers; Q4Bits_AVX2, the Q4_K/Q5_K dot route, every other "
                       "dequantizer and the aarch64 arm unchanged")),
    # iqk_set_kernels_kquants: the type-dispatch switch Q6_K/IQ4_XS admission actually
    # needs to route through. Its x86_64 signature differs textually from the aarch64
    # one (`[[maybe_unused]]` on func16), so it is already unambiguous without
    # `body_duplicates`; the fence is kept anyway as defence in depth.
    CpuSourceRoute(
        route="iqk_kquants_set_kernels",
        path="ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp",
        symbols=("iqk_set_kernels_kquants",),
        fence="// --------------------------------- __aarch64__ --------------------------------------",
        bodies=(("iqk_set_kernels_kquants",
                 "bool iqk_set_kernels_kquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {"),),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        numerics="ppl_contract",
        admitted_text=("hunks inside the iqk_set_kernels_kquants (x86_64) switch body; its "
                       "signature, every kernel it dispatches to and the aarch64 arm unchanged")),
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
    # FLASH-ATTENTION SCHEDULING (2026-10-04, ak-longctx audit C2). Decode/verify steps
    # have N <= 5 query rows, below the tiled path's 64, so each row is one serial
    # `_one_chunk` walk over the whole KV, threads split rows only, and the G query heads
    # that share a KV head (Q38FN: 12) each re-read that KV. Admitted: work split and
    # traversal ONLY, every output row's reduction order unchanged -- e.g. the G heads of
    # one KV head in a single KV pass, or a KV split whose merge reproduces the anchor's
    # order. The tiled body, K/V conversion and wdata sizing (ggml_graph_plan, ggml-cpu.c)
    # stay outside. Bit-exact by construction, so the anchor is the reference: the
    # generic FLASH_ATTN_EXT native suite, the `cpu_fa_longctx_v1` test-backend-ops case
    # set (when the binary carries it), `cpu_fa_reference` bit identity with the anchor
    # (both GGML_FA_SPLIT_KV settings, odd and recipe teams, 3 repetitions each), the
    # paired FA perf screen, then greedy identity on the frozen requests served 3x on
    # this lane's target and every peer (and on the long manifest once provided).
    CpuSourceRoute(
        route="cpu_fa_schedule",
        path="ggml/src/ggml-cpu/ops.cpp",
        symbols=("ggml_compute_forward_flash_attn_ext_f16",
                 "ggml_compute_forward_flash_attn_ext_f16_one_chunk",
                 "ggml_flash_attn_ext_reduce_partials",
                 "ggml_compute_forward_flash_attn_ext"),
        bodies=(("ggml_compute_forward_flash_attn_ext_f16_one_chunk",
                 "static void ggml_compute_forward_flash_attn_ext_f16_one_chunk("),
                ("ggml_flash_attn_ext_reduce_partials",
                 "static void ggml_flash_attn_ext_reduce_partials("),
                ("ggml_compute_forward_flash_attn_ext_f16",
                 "static void ggml_compute_forward_flash_attn_ext_f16("),
                ("ggml_compute_forward_flash_attn_ext",
                 "void ggml_compute_forward_flash_attn_ext(")),
        ops=("FLASH_ATTN_EXT",),
        new_helpers=True,
        model_identity=True,
        identity_targets="peers",
        identity_repeats=3,
        long_identity=True,
        helper_objects_const=True,
        # No OpenMP in any spelling, and no preprocessor line that could reach past the
        # hunk: a `#define`/`#undef` inside an FA body stays in force to the end of
        # ops.cpp (the tiled body, ssm_conv and gated_delta_net follow it); `#include
        # "..."` pulls in-tree code. `#include <...>` (system headers) and balanced
        # `#if`/`#endif` remain admitted (audit 2026-10-04 integration).
        forbidden_added=(r"#\s*pragma\s+omp|\b_Pragma\s*\(|\bomp_\w+\s*\(|"
                         r'^\+\s*#\s*(define|undef|pragma|line|error|warning)\b|^\+\s*#\s*include\s*"'),
        admitted_text=("hunks inside the ggml_compute_forward_flash_attn_ext_f16 (dispatch), "
                       "ggml_compute_forward_flash_attn_ext_f16_one_chunk, "
                       "ggml_flash_attn_ext_reduce_partials or "
                       "ggml_compute_forward_flash_attn_ext bodies plus NEW file-scope "
                       "static helpers (functions, or const objects with call-free "
                       "initializers) and #include <...> lines; work split and traversal "
                       "only, every output row's reduction order unchanged (bit-identical "
                       "to the anchor); no `#pragma omp`/`_Pragma`/omp_* calls, no "
                       "`#define`/`#undef`/`#include \"...\"` anywhere in the patch; the "
                       "tiled body, headers, ggml-cpu.c (wdata sizing) and every other "
                       "function unchanged. Gate: FLASH_ATTN_EXT native suite, bit identity "
                       "with the anchor on the FA probe cases (both GGML_FA_SPLIT_KV "
                       "settings, 3 repetitions), greedy identity on the frozen requests "
                       "served 3x (and on the long-context manifest when the lane opts in). "
                       "The cpu_fa_longctx_v1 case set and the paired FA perf screen run "
                       "only once test-backend-ops carries the case set; until then both "
                       "record SKIP and are not gates")),
)
# Numerics contracts are a closed set and mutually exclusive (review 2026-10-06): a
# ppl_contract route is never judged by greedy model identity and vice versa. Checked
# at import so a mis-declared route can never reach a running loop.
for _route in CPU_SOURCE_ROUTES:
    if _route.numerics not in ("bit_exact", "ppl_contract"):
        raise AssertionError(f"route {_route.route}: unknown numerics {_route.numerics!r}")
    if _route.numerics == "ppl_contract" and _route.model_identity:
        raise AssertionError(f"route {_route.route}: ppl_contract and model_identity are "
                             "mutually exclusive")
    if set(_route.body_duplicates) & set(_route.body_duplicates_first):
        raise AssertionError(f"route {_route.route}: a label cannot pick both occurrences")
del _route
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
        if label in route.body_duplicates_first:
            if not hits:
                return (f"{side}: marker `{prefix.strip()}` occurs 0 times inside the "
                        f"admitted window (the {label} boundary needs at least one)")
            start = hits[0]
        elif label in route.body_duplicates:
            if not hits:
                return (f"{side}: marker `{prefix.strip()}` occurs 0 times inside the "
                        f"admitted window (the {label} boundary needs at least one)")
            start = hits[-1]
        elif len(hits) != 1:
            return (f"{side}: marker `{prefix.strip()}` occurs {len(hits)} times inside the "
                    f"admitted window (the {label} boundary needs exactly one)")
        else:
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
    # Round-7 resolution C: lexical refusal FIRST, on every added line of every route,
    # so no grammar ever sees a line whose meaning depends on another line: no block
    # comment delimiter (`/*`, `*/`) anywhere, no line continuation, no raw string, no
    # unterminated literal, and no `#` outside a string/char literal (comments
    # included) except the documented file-scope new-helper directive lines.
    for hunk, label in zip(hunks, labels):
        for line in hunk[5]:
            # A whole-line rewrite the route admits by a closed regex (e.g. the
            # `/* .graph_optimize = */ <fn>,` slot) is fully determined by that regex.
            if label is None and route.line_rewrites and any(
                    re.fullmatch(new, line.strip()) for _old, new in route.line_rewrites):
                # Round-9: even the closed-regex slot may not name a dangerous API
                # (`/* .graph_optimize = */ system,` would install libc system()).
                if _DANGEROUS_API.search(line):
                    return (f"an added line rewrite names a process/filesystem/loader "
                            f"API: {line.strip()[:120]!r}")
                continue
            why = _lexical_refusal(line, file_scope_helper=(label is None),
                                   new_helpers=route.new_helpers)
            if why is not None:
                return f"an added line is refused lexically ({why}): {line.strip()[:120]!r}"
    if route.forbidden_added is not None:
        pattern = re.compile(route.forbidden_added)
        for hunk, label in zip(hunks, labels):
            if label is not None and label in route.forbidden_exempt_bodies:
                continue
            for line in hunk[5]:
                if pattern.search("+" + line):
                    return (f"an added line matches the forbidden pattern "
                            f"`{route.forbidden_added}` ({line.strip()[:120]!r})")
    # Round-5 resolution C: no route may add a preprocessor directive (or `_Pragma`)
    # anywhere it admits code -- a `#define` in an exempt body could otherwise rename a
    # checked symbol (`#define iqk_typeA_supported(t) true`). The single exception is
    # the documented `new_helpers` file-scope `#include <system>` insertion.
    #   * `#define` / `#undef` (and `_Pragma`) are refused EVERYWHERE, file scope too;
    #   * inside an admitted body every directive is refused (`#pragma omp` included);
    #   * a file-scope new-helper hunk may carry only conditional-compilation lines
    #     (balanced; see `_new_helper_refusal`) and `#include <system>`.
    #   * operator 2026-10-06 (friction): ISA/feature guards from the closed
    #     `_ISA_GUARD_LINE` grammar are admitted anywhere, balanced within the hunk.
    # Round-12: balance is judged on the RESULTING structure -- each edited body's
    # post-image must have the same conditional-compilation profile as its HEAD image
    # (so replacing an existing `#if` while keeping its `#endif` is fine); file-scope
    # helper lines (no body) must balance among themselves.
    post_lines = source_text.splitlines()
    for label in dict.fromkeys(label for label in labels if label is not None):
        head_region = next(r for r in old[0] if r[0] == label)
        post_region = next(r for r in new[0] if r[0] == label)
        head_profile = _directive_profile(pre_lines[head_region[1] - 1:head_region[2]])
        post_profile = _directive_profile(post_lines[post_region[1] - 1:post_region[2]])
        if post_profile != head_profile:
            return (f"conditional compilation in {label} no longer balances like HEAD "
                    f"(net/min depth {post_profile} vs {head_profile})")
    helper_lines = [line for hunk, label in zip(hunks, labels) if label is None
                    for line in hunk[5]]
    balance = _directive_balance_refusal(helper_lines)
    if balance is not None:
        return f"unbalanced conditional compilation in added file-scope lines: {balance}"
    for hunk, label in zip(hunks, labels):
        for line in hunk[5]:
            directive = _PREPROCESSOR_LINE.match(line)
            if directive is None and "_Pragma" not in line:
                continue
            word = directive.group(1) if directive is not None else "_Pragma"
            if _ISA_GUARD_LINE.fullmatch(_directive_code(line)):
                continue
            if label is None and word in ("if", "ifdef", "ifndef", "elif", "else", "endif"):
                continue
            if label is None and route.new_helpers and _SYSTEM_INCLUDE.fullmatch(line):
                continue
            return (f"an added line is a preprocessor directive or _Pragma "
                    f"({line.strip()[:120]!r}); no route may add #define/#undef/_Pragma, "
                    f"and no admitted body may add any directive")
    if route.required_added is not None:
        regex, why = route.required_added
        if not any(re.search(regex, code) for hunk in hunks for code in _code_lines(hunk[5])):
            return why
    return None


_PREPROCESSOR_LINE = re.compile(r"^\s*#\s*([A-Za-z_]+)")
_FILE_SCOPE_DIRECTIVES = ("if", "ifdef", "ifndef", "elif", "else", "endif")
#: Operator 2026-10-06 ("don't introduce excessive friction"): the CLOSED set of ISA /
#: feature guards a SIMD kernel edit legitimately needs inside a body. Terms are only
#: `defined(X)` / `!defined(X)` over this allowlist, joined by && / ||; plus
#: `#ifdef X`, `#ifndef X`, `#else`, `#endif`. Nothing else (no `#if 0`, no
#: arithmetic, no unlisted macro), and each file's added guards must balance.
_ISA_MACRO = (r"(?:__AVX__|__AVX2__|__AVX512F__|__AVX512BW__|__AVX512VL__|__AVX512VNNI__"
              r"|__AVX512BF16__|__AVXVNNI__|__F16C__|__FMA__|__ARM_NEON|__ARM_FEATURE_[A-Z0-9_]+"
              r"|HAVE_FANCY_SIMD|GGML_USE_[A-Z0-9_]+)")
_ISA_TERM = rf"!?\s*defined\s*\(\s*{_ISA_MACRO}\s*\)"
_ISA_EXPR = rf"{_ISA_TERM}(?:\s*(?:&&|\|\|)\s*{_ISA_TERM})*"
_ISA_GUARD_LINE = re.compile(
    rf"\s*#\s*(?:(?:if|elif)\s+{_ISA_EXPR}|ifdef\s+{_ISA_MACRO}|ifndef\s+{_ISA_MACRO}"
    rf"|else|endif)\s*")


def _directive_code(line: str) -> str:
    """The code part of a line (trailing `//` comment and literal contents removed)."""
    split = _split_line_lexically(line.rstrip("\n"))
    return line.rstrip("\n") if split is None else split[0]


def _directive_profile(lines) -> tuple[int, int]:
    """(net depth, minimum depth) of the conditional-compilation lines in `lines`."""
    depth = low = 0
    for line in lines:
        match = _PREPROCESSOR_LINE.match(_directive_code(line))
        if match is None:
            continue
        word = match.group(1)
        if word in ("if", "ifdef", "ifndef"):
            depth += 1
        elif word == "endif":
            depth -= 1
            low = min(low, depth)
        elif word in ("elif", "else"):
            low = min(low, depth - 1)
    return depth, low


def _directive_balance_refusal(lines) -> "str | None":
    """None when the conditional-compilation lines among `lines` open and close in
    order (`#else`/`#elif`/`#endif` never at depth 0, depth 0 at the end)."""
    depth = 0
    for line in lines:
        match = _PREPROCESSOR_LINE.match(_directive_code(line))
        if match is None:
            continue
        word = match.group(1)
        if word in ("if", "ifdef", "ifndef"):
            depth += 1
        elif word in ("elif", "else"):
            if depth == 0:
                return f"`#{word}` without an opening guard"
        elif word == "endif":
            if depth == 0:
                return "`#endif` without an opening guard"
            depth -= 1
    return None if depth == 0 else f"{depth} guard(s) left open"


def _split_line_lexically(line: str) -> "tuple[str, str] | None":
    """(code outside literals and comments, comment text) for one source line, or None
    when a string/char literal is unterminated. Literal CONTENTS are dropped.

    Round-14: comments are recognised IN the lexer -- a `//` or `/*` outside a literal
    suspends quote parsing (so `/* don't reorder */` or `// it's` is not an
    unterminated char literal). A `/* ... */` closed on this line becomes one space in
    the code and its text joins the comment; an UNCLOSED `/*` is left in the code (as
    `/*`), and a stray `*/` stays in the code, so the caller refuses both."""
    code, comments, i, n = [], [], 0, len(line)
    while i < n:
        ch = line[i]
        if line.startswith("//", i):
            comments.append(line[i + 2:])
            break
        if line.startswith("/*", i):
            end = line.find("*/", i + 2)
            if end < 0:
                code.append("/*")
                comments.append(line[i + 2:])
                break
            comments.append(line[i + 2:end])
            code.append(" ")
            i = end + 2
            continue
        if ch in "\"'":
            quote, i = ch, i + 1
            while i < n and line[i] != quote:
                i += 2 if line[i] == "\\" else 1
            if i >= n:
                return None
            code.append(quote + quote)
            i += 1
            continue
        code.append(ch)
        i += 1
    return "".join(code), "\n".join(comments)


#: Round-9 resolution B: process, filesystem, dynamic-loading, environment and
#: inline-asm APIs refused in every added line of every route (code part, outside
#: literals): candidate code shares the loop's account, so it must not be able to
#: touch the store, spawn processes, load code or read/alter the environment.
_DANGEROUS_API = re.compile(
    r"\b(?:system|popen|execl|execlp|execle|execv|execvp|execvpe|execve|fexecve|fork|vfork"
    r"|posix_spawnp?|fopen|fopen64|freopen|open|open64|openat|creat|write|pwrite|pwrite64"
    r"|pwritev2?|fwrite|ofstream|ifstream|fstream|filesystem|rename|renameat2?|unlink"
    r"|unlinkat|remove|mkdir|mkdirat|syscall|dlopen|dlmopen|dlsym|dlvsym"
    r"|getenv|secure_getenv|setenv|putenv|unsetenv|clearenv|mmap|mprotect|ptrace|kill"
    r"|raise|signal|sigaction|chmod|chown|truncate|ftruncate|symlink|link|socket|connect)\b")
_DANGEROUS_INCLUDE = re.compile(
    r"#\s*include\s*<\s*(?:cstdlib|stdlib\.h|fstream|filesystem|unistd\.h|dlfcn\.h|"
    r"spawn\.h)\s*>")
#: Round-9 resolution C: digraphs and trigraphs refused anywhere in an added line.
_DIGRAPH_TRIGRAPH = re.compile(r"%:|<%|%>|<:|:>|\?\?")


def _lexical_refusal(line: str, *, file_scope_helper: bool, new_helpers: bool) -> "str | None":
    if _DIGRAPH_TRIGRAPH.search(line):
        return "digraph or trigraph"
    if _DANGEROUS_INCLUDE.search(line):
        return "process/filesystem/loader header include"
    if line.rstrip("\n").rstrip().endswith("\\"):
        return "line continuation"
    split = _split_line_lexically(line.rstrip("\n"))
    if split is None:
        return "unterminated string/char literal"
    code, comment = split
    # Round-12 (honest-author model): a block comment that opens AND closes on this
    # line is ordinary kernel practice (the lexer already removed it from `code`);
    # an unclosed `/*`, a stray `*/`, or a delimiter inside a comment stays refused.
    for text in (code, comment):
        if "/*" in text or "*/" in text:
            return "multi-line block comment delimiter"
    if re.search(r"\b(?:u8|u|U|L)?R\"\"", code):
        return "raw string literal"
    # (an admitted file-scope `#include <...>` line names a header, not a call; the
    # header itself is screened by `_DANGEROUS_INCLUDE` above)
    api = None if code.lstrip().startswith("#") else _DANGEROUS_API.search(code)
    if api is not None:
        return f"process/filesystem/loader/asm API `{api.group(0)}`"
    if "#" in comment:
        return "`#` in a comment"
    if "#" in code:
        directive = _PREPROCESSOR_LINE.match(code)
        stripped = code.strip()
        if _ISA_GUARD_LINE.fullmatch(code):
            return None
        if not (file_scope_helper and directive is not None and stripped.startswith("#")
                and (directive.group(1) in _FILE_SCOPE_DIRECTIVES
                     or (new_helpers and _SYSTEM_INCLUDE.fullmatch(line.rstrip("\n"))))):
            return "`#` outside a string literal"
    return None
_SYSTEM_INCLUDE = re.compile(r"\s*#\s*include\s*<[^>\s]+>\s*")


_HELPER_START = re.compile(
    r"^(?:static\b|inline\s+static\b|template\s*<[^>]*>\s*static\b|"
    r"\[\[[\w:, ]+\]\]\s*static\b)")
_STATIC_FUNCTION = re.compile(r"^(?:template\s*<[^>]*>\s*)?(?:\[\[[\w:, ]+\]\]\s*)?"
                              r"(?:inline\s+)?static\b[^=;{]*\(")


def _static_object_refusal(statement: str) -> str | None:
    """`CpuSourceRoute.helper_objects_const`: a new file-scope `static` that is not a
    function must be `const`/`constexpr` and its initializer must not call anything."""
    if not _HELPER_START.match(statement) or _STATIC_FUNCTION.match(statement):
        return None
    head, _eq, init = statement.partition("=")
    if not re.search(r"\b(const|constexpr)\b", head):
        return ("a new file-scope static object must be const/constexpr "
                f"({statement[:80]!r})")
    if "(" in init or "(" in head:
        return ("a new file-scope static object's initializer must not call anything "
                f"({statement[:80]!r})")
    return None
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
    statement: list[str] = []
    const_objects = bool(route and route.helper_objects_const)
    for line in added:
        code, in_block = _strip_code_line(line, in_block)
        text = code.strip()
        if in_statement or (depth == 0 and text):
            statement.append(text)
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
            if const_objects:
                refusal = _static_object_refusal(" ".join(statement))
                if refusal is not None:
                    return refusal
            statement = []
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


# ------------------------------------------------------------------ static GPU keep rules
#
# GPU-POOL-1 (stack owner, 2026-10-04). v10's AutoKernel keep `mmvq_q8_1_graph_cache`
# (dd161d519, c0d42d81c, 8a3049beb) held `ggml_cuda_pool_alloc` buffers from the SHARED
# legacy pool (`ctx.pool()`, GGML_HIP_NO_VMM: best-fit, whole-buffer, 256 slots, never
# shrinks) from one HIP-graph capture until the next. Capacity cost ~+0.29 GiB per
# n_max:0 alternation (:8083 51.69 -> 59.77 GiB, KVU-16h), and a latent correctness
# hazard: capturing graph B returned graph A's baked-in buffers to the pool, which could
# then cudaFree memory a live graph instance still references. Reference fix: a private
# per-context arena rewound at capture, freed at context destruction
# (experimental/mmvq-graph-cache-pool-20261004, 656c9a66b). A stack-local
# `ggml_cuda_pool_alloc<T> x(ctx.pool(), n)` released at the end of the op stays legal.
_GPU_PERSISTENT_POOL_HOLD = (
    (re.compile(r"\b(?:unique_ptr|shared_ptr|optional)\s*<\s*ggml_cuda_pool_alloc\b"),
     "an owning holder of a ggml_cuda_pool_alloc"),
    (re.compile(r"\bnew\s+ggml_cuda_pool_alloc\b"), "a heap ggml_cuda_pool_alloc"),
    (re.compile(r"\bstatic\b[^;]*\bggml_cuda_pool_alloc\b"), "a static ggml_cuda_pool_alloc"),
    (re.compile(r"\bggml_cuda_pool\s*\*\s*\w+\s*(?:=|;)"), "a stored ggml_cuda_pool pointer"),
    (re.compile(r"=\s*&\s*\w+(?:->|\.)pool\s*\("), "a stored &ctx.pool() address"),
)


def _added_code_lines(patch_text: str) -> list[str]:
    """Added lines of a unified diff with // and /* */ comments removed."""
    out, in_block = [], False
    for raw in patch_text.splitlines():
        if not raw.startswith("+") or raw.startswith("+++"):
            continue
        line, in_block = _strip_code_line(raw[1:], in_block)
        if line.strip():
            out.append(line)
    return out


_GPU_STATIC_OBJECT = re.compile(
    r"\b(?:static\s+(?:thread_local\s+)?|thread_local\s+(?:static\s+)?)"
    r"[A-Za-z_][\w:<>,\s*&]*?\s+([A-Za-z_]\w*)\s*(?:;|=|\{|\[|$)")
_GPU_POOL_ADDRESS = re.compile(r"\b\w+\s*(?:\.|->)\s*(?:get\s*\(\s*\)|ptr\b)")
_GPU_LOCAL_FROM_POOL = re.compile(
    r"(?:^|[;{(\s])(?:[\w:<>*&\s]+\s)?\*?\s*([A-Za-z_]\w*)\s*=(?!=)\s*\w+\s*(?:\.|->)\s*"
    r"(?:get\s*\(\s*\)|ptr\b)")
_GPU_MEMBER_STORE = re.compile(
    r"\b(\w+)\s*(?:(?:\.|->)\s*\w+|\[[^\]]*\])\s*(?:\[[^\]]*\]\s*)?=(?!=)\s*([^;]*)")
_GPU_POOL_REFERENCE = re.compile(r"&\s*\w+\s*=(?!=)\s*\w+\s*(?:->|\.)\s*pool\s*\(")


def _gpu_statements(lines: list[str]) -> list[str]:
    """Added code joined and split on `;`/`{`/`}` so a declaration split across lines
    (`static\n thread_local\n T x;`) is one statement."""
    text = " ".join(line.strip() for line in lines)
    return [part.strip() for part in re.split(r"[;{}]", text) if part.strip()]


def _gpu_persistent_holds(lines: list[str]):
    """(statement, what) for every persistent hold of pool memory in the added lines.

    2026-10-04 (integration audit): the rule is syntactic and judged on the whole added
    text, not line by line, and it no longer waits for a `graph`/`capture` token --
    with GGML_HIP_GRAPHS=ON every op is captured, so every CUDA/HIP patch is in scope.
    """
    statements = _gpu_statements(lines)
    for statement in statements:
        for pattern, what in _GPU_PERSISTENT_POOL_HOLD:
            if pattern.search(statement):
                yield statement, what
        if _GPU_POOL_REFERENCE.search(statement):
            yield statement, "a stored reference to ctx.pool()"
    # A pool allocation's ADDRESS (`.get()` / public `.ptr`) retained anywhere that is not
    # an automatic local -- a member, an element, a static/thread_local object, or a local
    # that then flows into one -- outlives the RAII scope that returns the buffer to the
    # pool (dd161d519 shape: `cache.q8_1 = src1_q8_1.get();` into a thread_local cache).
    statics = {m.group(1) for st in statements for m in [_GPU_STATIC_OBJECT.search(st)] if m}
    carriers = set()
    for statement in statements:
        for m in _GPU_LOCAL_FROM_POOL.finditer(statement):
            carriers.add(m.group(1))
    for statement in statements:
        store = _GPU_MEMBER_STORE.search(statement)
        if store is None:
            continue
        target, value = store.group(1), store.group(2)
        if _GPU_POOL_ADDRESS.search(value) or any(
                re.search(rf"\b{re.escape(name)}\b", value) for name in carriers):
            yield statement, ("a pool-allocation address (.get()/.ptr) retained in "
                              + ("static storage" if target in statics
                                 else "a member or element that outlives the op"))
    for statement in statements:
        m = re.match(r"(?:static\s+)?(?:thread_local\s+)?[\w:<>*&\s]*?\b(\w+)\s*=(?!=)\s*(.*)$",
                     statement)
        if m and m.group(1) in statics and (_GPU_POOL_ADDRESS.search(m.group(2)) or any(
                re.search(rf"\b{re.escape(name)}\b", m.group(2)) for name in carriers)):
            yield statement, "a pool-allocation address (.get()/.ptr) retained in static storage"


def gpu_graph_pool_hold_refusal(patch_text: str | None) -> str | None:
    """GPU-POOL-1: refuse device memory kept alive across graph captures from ctx.pool()."""
    if not patch_text:
        return None
    lines = _added_code_lines(patch_text)
    for line, what in _gpu_persistent_holds(lines):
        return ("GPU-POOL-1: the patch keeps device memory alive across HIP-graph "
                f"captures with {what} (`{line.strip()[:120]}`). Memory a captured "
                "graph keeps must not come from ctx.pool()/ggml_cuda_pool_alloc (the "
                "shared NO_VMM legacy pool: +0.29 GiB per n_max:0 alternation in "
                "v10's mmvq_q8_1_graph_cache, and a released buffer a live graph still "
                "references can be freed). Use a private per-context arena or an "
                "explicit owned allocation freed at context destruction (reference: "
                "656c9a66b on experimental/mmvq-graph-cache-pool-20261004)")
    return None


#: Symbol markers admitted on `mmvq.cu` beyond `vec_dot_*` / `*mul_mat_vec*`: the
#: MMVQ/MMQ crossover (`ggml_cuda_should_use_mmvq`) and the per-cell geometry
#: (`calc_nwarps`, `calc_rows_per_block`) a multi-row GEMV changes.
GPU_MMVQ_SYMBOL_MARKERS = ("mmvq", "calc_nwarps", "calc_rows_per_block")
_GPU_MMQ_MARKERS = ("mul_mat_q", "mmq", "stream_k", "load_tiles", "vec_dot_")
GPU_MATMUL_HEADER_ROUTES = {
    "ggml/src/ggml-cuda/mmvq.cuh": ("mmvq", "mul_mat_vec"),
    "ggml/src/ggml-cuda/mmq.cuh": _GPU_MMQ_MARKERS,
    "ggml/src/ggml-cuda/mmq-config-cdna.cuh": _GPU_MMQ_MARKERS + ("config", "nwarps", "occupancy"),
    "ggml/src/ggml-cuda/mmq-load-tiles.cuh": _GPU_MMQ_MARKERS,
    "ggml/src/ggml-cuda/mmq-vec-dot.cuh": _GPU_MMQ_MARKERS,
}


def _gpu_matmul_symbol(symbol: str, markers: tuple[str, ...]) -> bool:
    lowered = (symbol or "").lower()
    return any(marker in lowered for marker in markers)


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
    if any(path.startswith("ggml/src/ggml-cuda/") for path in changed):
        patches = [patch_text, *((texts[2] if texts and len(texts) > 2 else None)
                                 for texts in (file_texts or {}).values())]
        pool_refusal = gpu_graph_pool_hold_refusal("\n".join(p for p in patches if p))
        if pool_refusal is not None:
            return Verdict("op_scope", False, pool_refusal)
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
            (target_symbol.startswith("vec_dot_") or "mul_mat_vec" in target_symbol
             or _gpu_matmul_symbol(target_symbol, GPU_MMVQ_SYMBOL_MARKERS)):
        return ("MUL_MAT", "MUL_MAT_ID")
    if changed == {"ggml/src/ggml-cuda/mmq.cu"} and \
            ("mul_mat_q" in target_symbol or "should_use_mmq" in target_symbol):
        return ("MUL_MAT", "MUL_MAT_ID")
    # Multi-row dequant-GEMV / verify-GEMM headers (27B GPU verify campaign, 2026-10-04):
    # the MMVQ cell table, the MMQ kernel/stream-k launch, the CDNA tile config and the
    # MMQ tile loaders/dot products. One header per patch; MUL_MAT + MUL_MAT_ID suites
    # against the CPU reference, exactly as the .cu routes above.
    if len(changed) == 1 and next(iter(changed)) in GPU_MATMUL_HEADER_ROUTES and \
            _gpu_matmul_symbol(target_symbol,
                               GPU_MATMUL_HEADER_ROUTES[next(iter(changed))]):
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


#: 2026-10-06 follow-up (seed 6 kquants widening, `iqk_kquants_q6_iq4xs_dequant` /
#: `iqk_kquants_set_kernels`): symbols `check_cpu_iqk_reference` now NAMES explicitly
#: instead of lumping them into the generic "unsupported" bucket, and the specific,
#: honest reason each has no PASSING independent reference today. `cpu_quant_reference`
#: (the scalar-decode independent oracle `check_q45_dot` uses) only implements and
#: EMPIRICALLY CALIBRATES Q4_K/Q5_K/Q8_0 -- its tolerances are fit against measured
#: anchor-gen scalar-vs-graph error, something this review could not do (no build, no
#: inference). Q6_K's block layout (`ggml-quants.c::dequantize_row_q6_K`) is simple
#: enough to transcribe correctly from source alone, but a transcribed decode with an
#: UNCALIBRATED tolerance is not a safer "pass" than no reference at all -- a bound
#: guessed instead of measured could pass a wrong kernel or fail a correct one just as
#: easily as having no bound. IQ4_XS's codebook-based layout is materially more
#: complex (non-uniform scale packing, shared LUT) and was not attempted blind.
#: `iqk_set_kernels_kquants` is different: it is the SAME dispatch switch
#: `mul_mat_qX_K_q8_2_X4_T` already reviews for Q4_K/Q5_K (`check_q45_dot`), so a
#: Q4_K/Q5_K-slice reference is real, immediately-usable coverage for it today, even
#: though the switch also routes Q6_K/IQ4_XS this reference cannot independently check.
_CPU_IQK_UNCALIBRATED_SYMBOLS = {
    "DequantizerQ6K_AVX2": (
        "Q6_K has a transcribable scalar decode (ggml-quants.c dequantize_row_q6_K) "
        "but NO calibrated tolerance: cpu_quant_reference's bounds are fit against "
        "measured anchor-gen scalar-vs-graph error, which requires a build+run this "
        "review could not do. Refusing rather than guessing a bound."),
    "mul_mat_qY_K_q8_2_X4_T": (
        "Q6_K kernel-selection body: same gap as DequantizerQ6K_AVX2 -- a "
        "transcribable decode with no calibrated tolerance is refused, not guessed."),
    "DequantizerIQ4XS": (
        "IQ4_XS uses a codebook-based (non-uniform scale) layout materially more "
        "complex than the K-quant family; its decode was not transcribed blind, "
        "and has no independent reference."),
}


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
    elif target_symbol in ("mul_mat_qX_K_q8_2_X4_T", "iqk_set_kernels_kquants"):
        # iqk_set_kernels_kquants is the type-dispatch switch Q4_K/Q5_K already share
        # with mul_mat_qX_K_q8_2_X4_T's own route; the Q4_K/Q5_K slice of an edit to
        # either body is real, reviewable evidence even though the switch also
        # dispatches Q6_K/IQ4_XS, which this reference cannot independently check.
        result = iqk_witness.check_q45_dot(build_dir, resolved_recipe=resolved_recipe,
                                            source_root=source_root)
    elif target_symbol in _CPU_IQK_UNCALIBRATED_SYMBOLS:
        return Verdict("oracle_unavailable", False, _CPU_IQK_UNCALIBRATED_SYMBOLS[target_symbol])
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


def check_cpu_fa_case_set(build_dir: Path, *, resolved_recipe) -> Verdict:
    """`cpu_fa_schedule`: the FLASH_ATTN_EXT long-context case set as a correctness corpus.

    Runs only when the build's test-backend-ops carries the `cpu_fa_longctx_v1`
    selector (the llama-tree patch); otherwise a recorded SKIP, because a 0/0 suite is
    not evidence and the anchor-identity probe covers the same shapes."""
    from . import cpu_fa_reference as fa

    if not fa.binary_has_case_set(build_dir):
        return Verdict("correctness", True,
                       f"SKIPPED: test-backend-ops does not carry the {fa.CASE_SET_ID} case "
                       "set (llama-tree patch not applied); the anchor-identity probe runs "
                       "the same shapes")
    return op_correctness(build_dir, op="FLASH_ATTN_EXT", backend="CPU",
                          resolved_recipe=resolved_recipe, params_filter=fa.CASE_SET_REGEX,
                          environment_overrides=((fa.CASE_SET_ENV, fa.CASE_SET_ID),),
                          expected_cases=len(fa.CASE_SET))


def check_cpu_fa_reference(anchor_build: Path, candidate_build: Path, source_root: Path, *,
                           anchor_recipe, candidate_recipe, window=None) -> Verdict:
    """`cpu_fa_schedule`: bit identity with the ANCHOR on the FA probe cases."""
    from . import cpu_fa_reference

    result = cpu_fa_reference.check_anchor_identity(
        anchor_build, candidate_build, source_root, anchor_recipe=anchor_recipe,
        candidate_recipe=candidate_recipe, window=window)
    return Verdict("reference_comparison" if result.status != "unavailable" else
                   "oracle_unavailable", result.status == "pass", result.reason, result.detail)


def check_cpu_fa_perf_screen(anchor_build: Path, candidate_build: Path, *, anchor_recipe,
                             candidate_recipe, window=None) -> Verdict:
    """`cpu_fa_schedule`: paired anchor/candidate `test-backend-ops perf` screen on the
    case set, before the serving A/B is paid for. A binary without the case set is a
    recorded SKIP; a harness fault is `oracle_unavailable`; too little speedup refuses."""
    from . import cpu_fa_reference as fa

    if not (fa.binary_has_case_set(anchor_build) and fa.binary_has_case_set(candidate_build)):
        return Verdict("cpu_fa_perf_screen", True,
                       f"SKIPPED: the anchor or candidate test-backend-ops does not carry the "
                       f"{fa.CASE_SET_ID} case set (llama-tree patch not applied)")
    result = fa.perf_screen(anchor_build, candidate_build, anchor_recipe=anchor_recipe,
                            candidate_recipe=candidate_recipe, window=window)
    if result.status == "unavailable":
        return Verdict("oracle_unavailable", False, result.reason, result.detail)
    return Verdict("cpu_fa_perf_screen", result.status == "pass", result.reason,
                   result.detail)


def check_served_shape_case_set(build_dir: Path, *, resolved_recipe,
                                manifest_path: Path, routed: bool = False,
                                lane: "str | None" = None) -> Verdict:
    """ppl_contract layer (a) addendum (review 2026-10-06): the model's OWN served
    (k, m[, n_mats, n_used]) matmul shapes, at the served widths, for every witness
    quant -- `served_shape_cases.py`'s HONEST LIMITATIONS fix for the generic
    `k=256, m=16/512` sweep layer (a) otherwise runs alone.

    UNLIKE `check_cpu_fa_case_set`, this FAILS CLOSED, never skips, when the manifest
    is missing/malformed or the binary lacks the case-set literal: there is no
    redundant independent probe covering these shapes, so an absent corpus here is a
    gap in the correctness oracle, not a harmlessly-skipped extra."""
    from . import served_shape_cases as ssc

    # Round-12: unavailable evidence is gate "oracle_unavailable" (never a numerical
    # FAIL), so a bundle bisect can tell "could not measure" from "measured wrong".
    set_id = ssc.ROUTED_CASE_SET_ID if routed else ssc.CASE_SET_ID
    if lane not in ssc.LANE_PROFILES:
        # Round-15: both corpora are lane-partitioned (the candidate corpus measures the
        # lane's own served shapes; the routed one its GGUF expert profile).
        return Verdict("oracle_unavailable", False,
                       f"no served-shape profile for this model (lane {lane!r}); the "
                       "corpora derive from the lane's served GGUF")
    try:
        cases = ssc.load_manifest(manifest_path, routed=routed, lane=lane)
    except ssc.ManifestRefused as exc:
        return Verdict("oracle_unavailable", False,
                       f"served-shape case set is not available: {exc}")
    has = ssc.binary_has_routed_case_set if routed else ssc.binary_has_case_set
    if not has(build_dir):
        return Verdict("oracle_unavailable", False,
                       f"test-backend-ops at {build_dir} does not carry the {set_id} case "
                       "set with the backend-thread control (llama-tree patch not "
                       "applied); layer (a) has no served-shape evidence for this build")
    try:
        threads = served_launch(resolved_recipe)["threads"]
    except (ValueError, TypeError) as exc:
        return Verdict("oracle_unavailable", False,
                       f"served backend thread count is unknown ({exc}); refusing to run "
                       "the served-shape cases at an uncontrolled thread count")
    ops_present = sorted({c.shape.op for c in cases})
    verdict = op_correctness(build_dir, op=",".join(ops_present), backend="CPU",
        resolved_recipe=resolved_recipe, params_filter=ssc.case_set_regex(cases),
        environment_overrides=((ssc.CASE_SET_ENV, set_id),
                               (ssc.BACKEND_THREADS_ENV, str(threads))),
        expected_cases=len(cases))
    gate = ("served_shape_case_set" if verdict.passed or verdict.gate != "oracle_unavailable"
            else "oracle_unavailable")
    return Verdict(gate, verdict.passed, verdict.reason, verdict.detail)


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

    `targets` is [(label, anchor recipe, candidate recipe, frozen requests[, prepare])]:
    this lane's own target first, then the lane binding's peer targets when the route
    asks for them (`CpuSourceRoute.identity_targets`); the optional fifth element is a
    per-request `prepare(port)` hook (the long-context surface's slot restore). Every target must pass; any `wrong` is a
    verdict, otherwise any `unavailable` makes the gate unavailable. `required_arch`: at
    least one target must serve a GGUF of one of these architectures, or the edited
    model-specific code would never run under the gate (unavailable, not a pass)."""
    from . import model_identity

    targets = tuple(targets)
    if not targets:
        return Verdict("oracle_unavailable", False, "model identity gate has no target")
    if required_arch:
        read = architecture or model_identity.model_architecture
        archs = {label: read(anchor) for label, anchor, *_rest in targets}
        if not any(arch in required_arch for arch in archs.values()):
            return Verdict("oracle_unavailable", False,
                           f"no identity target serves a {'/'.join(required_arch)} model "
                           f"(targets: {archs}); the edited model code would never run "
                           "under the gate -- run this route on a lane bound to that model "
                           "or with it as a peer")
    rows, wrong, unavailable = [], [], []
    for label, anchor, candidate, requests, *extra in targets:
        result = model_identity.check(anchor_recipe=anchor, candidate_recipe=candidate,
                                      requests=tuple(requests or ()), window=window,
                                      repeats=repeats,
                                      **({"prepare": extra[0]} if extra and extra[0] else {}))
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
                   require_reference: bool = False,
                   params_filter: str | None = None,
                   environment_overrides: tuple[tuple[str, str], ...] = (),
                   expected_cases: int | None = None) -> Verdict:
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

    `params_filter` / `environment_overrides` / `expected_cases` select a reviewed case
    set (`-p <regex>` plus its `AUTOKERNEL_CORRECTNESS_CASE_SET` selector, e.g.
    `cpu_fa_reference.CASE_SET_ID`); a run that selects any other number of cases is a
    harness fault, not a verdict.
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
    if params_filter is not None:
        argv.extend(("-p", params_filter))
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
    if environment_overrides:
        environment = {**environment, **dict(environment_overrides)}
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
    if expected_cases is not None and \
            sum(int(total) for _, total in counts) != expected_cases:
        return Verdict("oracle_unavailable", False,
                       f"test-backend-ops selected {sum(int(t) for _, t in counts)} {op} "
                       f"case(s) on {backend}, not the case set's {expected_cases}; this is a "
                       "harness fault, NOT evidence about the patch", output[-2000:])
    if block.group(2) == "FAIL":
        return Verdict("correctness", False, f"{op} failed on {backend}",
                       _correctness_detail(backend, counts, done.stdout, done.stderr))
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


# --- ppl_contract numerics (2026-10-06, operator layered-acceptance amendment) -------
#
# A `CpuSourceRoute` with `numerics="ppl_contract"` trades bit-exactness for performance.
# The operator's instruction, verbatim: design this "with utmost care to not introduce
# garbage-generating changes"; acceptance is LAYERED, every layer REQUIRED, none alone
# sufficient, FAIL CLOSED on any layer that is missing or errors:
#   (a) op-level NMSE vs the independent reference, on the served widths (decode N=1,
#       verify widths 2-5), over EVERY quant type test-backend-ops carries;
#   (b) wikitext2 |delta ppl| <= 0.5% vs the anchor (32x512 chunks), AND no cumulative
#       drift beyond max(0.5%, the anchor's own drift) vs a FIXED reference build;
#   (c) greedy token agreement >= 0.98 on >= 2 PRODUCTION-length prompts (>= 4096
#       prompt tokens each, enforced from the tool's own token count), vs the anchor
#       and -- non-regressing -- vs the fixed reference;
#   (d) a >= 1024-token canary: repetition (distinct-4gram ratio, longest repeat run)
#       no worse than the anchor's or the fixed reference's by more than a set margin.
# Layer (e) -- "never folds on bench evidence alone" -- is enforced by WIRING: the gate
# is a BLOCKING `route_references` entry in run.py (pre-keep), and the serving-gate fold
# re-runs layers (a)-(d) on the whole bundle (`ppl_contract_bundle_gate`) before the
# champion of record may advance. Bit-exact routes never call anything in this section.
#
# REVIEW 2026-10-06 (fixes over the first cut): every tool run checks its exit status
# (a usage/crash text used to be compared as "output" and agreed with itself); the
# generator is `llama-completion -no-cnv` (this tree's `llama-cli` is the chat REPL and
# ignores -no-cnv); prompt echo/logs are excluded (stdout only, --no-display-prompt);
# agreement divides by the LONGER output (an early-EOS candidate no longer passes); the
# anchor-only comparison is backed by a fixed reference because the anchor ADVANCES on
# every keep (0.5% per keep would otherwise compound without bound); caches are keyed by
# content (every DSO + tool + model identity + corpus + every run parameter + env) and
# live in the loop store, never in a build dir (the fixed reference is the frozen
# production kernel store).
#
# HONEST LIMITATIONS (left for the lane owner; each fails CLOSED, never open):
#  * layer (a) is test-backend-ops' own per-case bound (MUL_MAT/MUL_MAT_ID max NMSE 5e-4,
#    the generic quantized-matmul tolerance) on its generic shapes (k=256, m=16/512) at
#    the served widths -- not a literal NMSE float on the model's served (m, k), which
#    needs a served-shape case set (`test-backend-ops --test-file`) and a tighter bound;
#  * test-backend-ops never allocates the CPU_REPACK extra buffer, so no op oracle
#    reaches `repack.cpp`: routes in PPL_CONTRACT_NO_OP_ORACLE are refused at layer (a);
#  * the anchor generations and candidate builds must carry PPL_CONTRACT_TOOL_TARGETS;
#    a build without them fails layers (b)-(d) with a named reason.
PPL_CORPUS = "/mnt/raid0/llm/data/wikitext2_test.txt"
#: The corpus is not in git: pin it, and refuse (not silently re-baseline) on a change.
PPL_CORPUS_SHA256 = "aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc"
PPL_CHUNKS, PPL_CTX, PPL_BATCH, PPL_REL_BAR = 32, 512, 512, 0.005
#: Layer (a): the activation-width ("n") values a decode step (1) and a speculative
#: verify step (2-5) actually serve; test-backend-ops case strings carry `,n=<int>,`.
PPL_CONTRACT_SERVED_WIDTHS = (1, 2, 3, 4, 5)
#: Layer (a): routes whose window no test-backend-ops case can reach (see above).
PPL_CONTRACT_NO_OP_ORACLE = frozenset({"cpu_repack_mmid"})
#: Layer (c): the existing T0 coherence floor (campaign.py:3054), reused verbatim.
PPL_CONTRACT_AGREEMENT_FLOOR = 0.98
#: Layer (c): "production-length" -- the tool's own prompt token count must reach this.
PPL_CONTRACT_PROD_PROMPT_TOKENS_MIN = 4096
PPL_CONTRACT_PROD_PROMPT_COUNT_MIN = 2
#: Layer (c): (byte offset, max chars) slices of PPL_CORPUS. Every slice lies far past
#: the ppl window (32x512 tokens ~ the first 75 KB), so (b) and (c) never share text;
#: ~26 KB of wikitext is ~5.5-6.5k tokens on the served tokenizers (enforced >= 4096).
PPL_CONTRACT_PROD_PROMPT_SLICES = ((400_000, 26_000), (800_000, 26_000))
#: Layer (c): greedy tokens generated per prompt; the floor admits a divergence only in
#: the last 2% of them (~10 tokens).
PPL_CONTRACT_PROD_GEN_TOKENS = 512
PPL_CONTRACT_GEN_CTX = 12288
#: Layer (d): canary length, prompt slice, context, and the repetition bars.
PPL_CONTRACT_CANARY_TOKENS = 1024
PPL_CONTRACT_CANARY_SLICE = (1_100_000, 1_500)
PPL_CONTRACT_CANARY_CTX = 4096
PPL_CONTRACT_CANARY_MAX_REPEAT_RUN = 32
PPL_CONTRACT_CANARY_DISTINCT4_DROP = 0.10
#: Tools layers (b)-(d) run; a candidate build of a ppl_contract route adds them.
PPL_CONTRACT_TOOL_TARGETS = ("llama-perplexity", "llama-completion")
#: The FIXED reference for cumulative drift: the frozen production CPU kernel store
#: (a symlink to `<store>/builds/cpu-<date>-<sha>/bin`; its parent is the "build").
PRODUCTION_CPU_KERNEL = "/mnt/raid0/llm/kernels/production/cpu"


def production_cpu_reference_build() -> Path:
    """The build directory (parent of `bin/`) behind the production CPU kernel store.

    Resolves the store symlink only -- it does NOT prove the resulting directory is a
    real build or that it can load any particular lane's model (a newer GGUF
    architecture than the frozen production tree supports, or a broken/missing store
    symlink, both land here as a directory `_build_identity`/`check_production_
    reference_loads` must still prove is usable). Never caches: the store symlink can
    move across a production promotion, and a cached stale target would silently
    compare a candidate against last freeze's kernel forever.
    """
    return Path(os.path.realpath(PRODUCTION_CPU_KERNEL)).parent


#: Store-relative pin of the fixed reference (re-review 2026-10-06): the production
#: symlink is movable, so the first ppl_contract use records the RESOLVED build path
#: and its content identity; every later gate/fold refuses on any change. Resetting the
#: contract is an explicit act: delete this file (an operator decision, visible).
PPL_CONTRACT_REFERENCE_PIN = Path("ppl_contract") / "reference_pin.json"


def _reference_identity(build: Path, env: "dict | None" = None,
                        candidate_build: "Path | None" = None) -> str:
    """Both tools + every DSO + the resolved closure + the EFFECTIVE environment
    (LD_PRELOAD contents and closure included; round-3 review)."""
    return hashlib.sha256("\n".join(
        [*(_build_identity(build, tool, _rebind_ld(env or {}, build, candidate_build))
           for tool in PPL_CONTRACT_TOOL_TARGETS),
         _env_identity(env or {})]).encode()).hexdigest()


def pinned_production_reference(store: Path, env: "dict | None" = None,
                                candidate_build: "Path | None" = None) -> Path:
    """The campaign's FIXED reference build: pinned on first use (resolved path +
    content identity of both tools and every DSO), verified on every later use.
    Raises ValueError on a moved symlink, a changed build, or an unreadable pin."""
    pin = Path(store) / PPL_CONTRACT_REFERENCE_PIN
    current = production_cpu_reference_build()
    if pin.exists():
        try:
            body = json.loads(pin.read_text(encoding="utf-8"))
            pinned, digest = Path(body["path"]), body["identity"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{pin} is unreadable ({exc}); refusing to re-pin silently")
        if pinned != current:
            raise ValueError(f"production reference moved: pinned {pinned}, symlink now "
                             f"resolves to {current}; delete {pin} to re-baseline explicitly")
        if _reference_identity(pinned, env, candidate_build) != digest:
            raise ValueError(f"pinned production reference {pinned} changed content or "
                             "effective environment (LD_PRELOAD / launch env)")
        return pinned
    identity = _reference_identity(current, env, candidate_build)
    pin.parent.mkdir(parents=True, exist_ok=True)
    tmp = pin.with_name(f".{pin.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"path": str(current), "identity": identity}), encoding="utf-8")
    os.replace(tmp, pin)
    return current


def check_production_reference_loads(reference_build: Path, *, model: Path, cpu_list: str,
                                     env: dict, log_dir: Path, threads: int = 1,
                                     candidate_build: "Path | None" = None,
                                     launch: "dict | None" = None) -> Verdict:
    """Fail closed when the frozen production CPU kernel reference cannot load THIS
    lane's model at all (review 2026-10-06).

    `ppl_contract_gate`'s layers (b)-(d) treat `reference_build` as a FIXED, trusted
    baseline every candidate (and every advancing anchor) is compared against. A
    reference that cannot load the model at all is not "no evidence" -- without this
    preflight, `ppl_wikitext2` reads it as `ppl_r=None` and SKIPS the fixed-reference
    comparison entirely (silently falling back to anchor-only bars), which is
    indistinguishable from "the reference agrees" to anything reading the verdict
    reason alone. A model built against an architecture newer than the frozen
    production tree (this lane's own DS41/Q38FN GGUFs, both well past the v10
    freeze date) is exactly the failure this preflight exists to catch, named and
    fail-closed, before any layer silently degrades.
    """
    binary = Path(reference_build) / "bin" / "llama-completion"
    if not binary.is_file():
        return Verdict("production_reference_load", False,
                       f"no llama-completion at the frozen production reference "
                       f"{reference_build} (PPL_CONTRACT_TOOL_TARGETS missing from the "
                       "production kernel store build)")
    # 8 tokens, not 1: a single generated token can legitimately be whitespace, which
    # `_completion` (correctly) reads as "no output" -- a spurious fail-closed.
    result = _completion(reference_build, "The quick brown fox jumps.", 8, 64, model=model,
                         threads=threads, env=env, cpu_list=cpu_list, log_dir=log_dir,
                         cache_dir=None, label="production_reference_load",
                         candidate_build=candidate_build, launch=launch)
    if result is None:
        return Verdict("production_reference_load", False,
                       f"the frozen production CPU kernel reference at {reference_build} "
                       f"could not load {model} (llama-completion refused or produced no "
                       "output) -- this lane's model may use an architecture the frozen "
                       "production kernel predates; the reference is not independent "
                       "evidence if it cannot run the model at all")
    return Verdict("production_reference_load", True,
                   f"production reference at {reference_build} loads {model}")


def ppl_contract_paths() -> frozenset[str]:
    """Every source path a ppl_contract route can edit (for the fold-time bundle check)."""
    return frozenset(route.path for route in CPU_SOURCE_ROUTES
                     if route.numerics == "ppl_contract")


#: Ledger of mechanism ids a ppl_contract route admitted (written BEFORE its layers run,
#: so a crash between keep and fold can never lose the mark). The fold reads it against
#: `Bundle.keeps` (mechanism ids) to decide whether the bundle needs layers (a)-(d).
PPL_CONTRACT_LEDGER = Path("ppl_contract") / "admitted_mechanisms.json"


def ppl_contract_only_paths() -> frozenset[str]:
    """Paths that ONLY a ppl_contract route can edit (no bit-exact route, no ad-hoc rule):
    any change to them in a bundle is ppl_contract by construction."""
    others = {route.path for route in CPU_SOURCE_ROUTES if route.numerics != "ppl_contract"}
    others |= {"ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp",
               "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp"}
    return ppl_contract_paths() - others


def ppl_contract_ledger_read(store: Path) -> "set[str] | None":
    """Admitted mechanism ids; empty when no ledger exists; None when it is unreadable."""
    path = Path(store) / PPL_CONTRACT_LEDGER
    if not path.exists():
        return set()
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(body, list) or not all(isinstance(x, str) for x in body):
            return None
        return set(body)
    except (OSError, ValueError):
        return None


def ppl_contract_ledger_add(store: Path, mechanism_id: str) -> None:
    """Record a ppl_contract admission. Raises on failure: an unrecorded admission could
    later fold without its bundle check, so the gate must not proceed without it."""
    path = Path(store) / PPL_CONTRACT_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    current = ppl_contract_ledger_read(store)
    if current is None:
        raise ValueError(f"{path} is unreadable; refusing to admit a ppl_contract candidate")
    if mechanism_id in current:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(sorted(current | {mechanism_id})), encoding="utf-8")
    os.replace(tmp, path)


def ppl_contract_anchor_obligated(store: Path) -> bool:
    """True when the next champion-advancement anchor build must widen its targets to
    `PROMOTION_TARGETS + PPL_CONTRACT_TOOL_TARGETS`, so a ppl_contract candidate's
    layers (b)-(d) (which compare the CANDIDATE build against the CURRENT anchor)
    have a reference anchor that carries `llama-perplexity`/`llama-completion` at all.

    Review 2026-10-06: `pool.promote_anchor` built every anchor with
    `gates.PROMOTION_TARGETS` alone, so even a run with an admitted ppl_contract route
    could never clear layer (b) -- the shared anchor never had the tool to run it with
    (`_build_identity` raises 'the build lacks llama-perplexity'). This is the signal
    `run.py`'s keep path reads before calling `pool.promote_anchor`.

    Fails WIDE (True), never narrow, when the ledger cannot be read: a too-narrow
    anchor is a missing-tool refusal discovered later (named, at layer (a)); a
    too-wide one costs one extra link. An EMPTY, readable ledger (no ppl_contract
    keep admitted yet in this store) is the only case that stays narrow -- the
    existing-run/resume case this must not regress (2026-10-06 follow-up): a store
    with no ppl_contract history keeps promoting bench-only-plus-server anchors
    exactly as before.
    """
    ledger = ppl_contract_ledger_read(store)
    return ledger is None or bool(ledger)


#: Git trailer the keep commit carries when its patch was admitted under a ppl_contract
#: route (re-review 2026-10-06): git history is the durable record, so a lost or reset
#: ledger can no longer let a shared-path ppl_contract keep fold unexamined.
PPL_CONTRACT_TRAILER = "AK-Numerics: ppl_contract"


def ppl_contract_commits_in_range(worktree: Path, base: str, tip: str) -> "bool | None":
    """True when any commit in base..tip carries PPL_CONTRACT_TRAILER; None when git
    cannot answer (callers treat None as True)."""
    done = subprocess.run(["git", "-C", str(worktree), "log", "--format=%B", f"{base}..{tip}"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        return None
    return any(line.strip() == PPL_CONTRACT_TRAILER for line in done.stdout.splitlines())


#: Round-5 resolution A: a commit touching a ppl_contract-RELEVANT file folds without
#: the quality gate only when the loop itself stamped it bit-exact AFTER its bit-exact
#: oracle passed: both lines below, the oracle line carrying the sha256 of the passing
#: verdicts. Absent, legacy, merge, or any other provenance means the gate is required.
PPL_CONTRACT_BIT_EXACT_TRAILER = "AK-Numerics: bit_exact"
PPL_CONTRACT_ORACLE_PREFIX = "AK-Oracle: sha256:"


def ppl_contract_relevant(path: str) -> bool:
    """Any iqk file, repack.cpp, or any path a ppl_contract route can edit."""
    return (path.startswith("ggml/src/ggml-cpu/iqk/") or path == "ggml/src/ggml-cpu/repack.cpp"
            or path in ppl_contract_paths())


#: Round-9 resolution A: DIAGNOSTIC ONLY. Candidate code runs unprivileged in the same
#: account as the loop (an admitted body executing `system()` could write these files),
#: so nothing reads these records as proof: no stamp or record exempts a range from the
#: fold-time quality gate. They are kept as an audit trail of bit-exact admissions.
BIT_EXACT_RECORD_DIR = Path("ppl_contract") / "bit_exact_records"
BIT_EXACT_RECORD_SCHEMA = "epyc.autokernel.bit_exact_record.v1"
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def bit_exact_record_bytes(*, tree: str, parent: str, route: str, oracle: str,
                           verdict_digests, changed_files, mechanism_id: str) -> bytes:
    body = {"schema": BIT_EXACT_RECORD_SCHEMA, "route_class": "bit_exact", "tree": tree,
            "parent": parent, "route": route, "oracle": oracle,
            "verdict_digests": list(verdict_digests),
            "changed_files": sorted(set(changed_files)), "mechanism_id": mechanism_id}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def write_bit_exact_record(store: Path, record: bytes) -> str:
    digest = hashlib.sha256(record).hexdigest()
    folder = Path(store) / BIT_EXACT_RECORD_DIR
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{digest}.{os.getpid()}.tmp"
    tmp.write_bytes(record)
    os.replace(tmp, folder / f"{digest}.json")
    return digest


def bind_bit_exact_record(store: Path, commit: str, digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("bind_bit_exact_record needs a commit sha and a sha256 digest")
    folder = Path(store) / BIT_EXACT_RECORD_DIR / "by-commit"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{commit}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"commit": commit, "record": digest}), encoding="utf-8")
    os.replace(tmp, folder / f"{commit}.json")


def ppl_contract_range_requires_gate(worktree: Path, base: str, tip: str,
                                     store: "Path | None" = None) -> bool:
    """Round-9 resolution A: must base..tip pass the ppl_contract quality layers before
    it reaches a champion? True when git cannot answer; when any commit message carries
    a control character or mentions ppl_contract; or when ANY commit in the range (or
    the range as a whole) touches a ppl_contract-relevant file -- any iqk file,
    repack.cpp, the dispatch whitelists, any ppl_contract path. There is NO exemption:
    stamps and store records are not proof (`store` is accepted for call compatibility
    and deliberately unused). History is read framing-safe: SHAs from rev-list, raw
    messages via cat-file, file lists via diff-tree -z, bound by SHA."""
    del store  # never an exemption source (round-9 resolution A)

    def run_git(*args) -> bytes:
        done = subprocess.run(["git", "-C", str(worktree), *args], capture_output=True)
        if done.returncode != 0:
            raise OSError(done.stderr.decode(errors="replace").strip()[:200])
        return done.stdout
    try:
        whole = [n for n in run_git("diff", "--name-only", "--no-renames", "-z", base,
                                    tip).decode().split("\x00") if n]
        if any(ppl_contract_relevant(n) for n in whole):
            return True
        shas = run_git("rev-list", f"{base}..{tip}").decode().split()
        for sha in shas:
            if not re.fullmatch(r"[0-9a-f]{40}", sha):
                return True
            raw = run_git("cat-file", "commit", sha)
            _header, _, message = raw.partition(b"\n\n")
            text = message.decode("utf-8", errors="replace")
            if _CONTROL.search(text) or "\ufffd" in text or "ppl_contract" in text:
                return True
            files = [n for n in run_git("diff-tree", "-r", "-m", "--root", "--no-commit-id",
                                        "--name-only", "-z", sha).decode().split("\x00") if n]
            if any(ppl_contract_relevant(n) for n in files):
                return True
    except (OSError, UnicodeDecodeError, IndexError):
        return True
    return False


def ppl_contract_fold_required(keeps, changed_paths, ledger: "set[str] | None",
                               trailer: "bool | None" = False) -> bool:
    """Does a champion-of-record fold of this bundle need `ppl_contract_bundle_gate`?

    Yes when a bundled keep was admitted under a ppl_contract route, when the bundle
    touches a path only ppl_contract routes can edit, or -- fail closed -- when the
    ledger is unreadable and the bundle touches ANY ppl_contract path."""
    changed = set(changed_paths)
    if trailer is None or trailer:
        return True
    if ledger is None:
        return bool(changed & ppl_contract_paths())
    return bool(set(keeps) & ledger) or bool(changed & ppl_contract_only_paths())


def _run_tool(argv: list[str], *, env: dict, log_dir: Path, label: str,
              timeout: int = 3600) -> tuple["int | None", str, str]:
    """Run `argv` with stdin closed; persist stdout/stderr; return (rc, stdout, stderr).

    rc is None when the tool could not run or timed out. Callers MUST treat any rc != 0
    as a failed layer: a usage text or a crash message is never an observation."""
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env=env, stdin=subprocess.DEVNULL)
        rc, out, err = done.returncode, done.stdout, done.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        rc, out, err = None, "", f"{type(exc).__name__}: {exc}"
    try:
        (log_dir / f"{label}.log").write_text(
            f"argv: {argv!r}\nrc: {rc!r}\n--- stdout ---\n{out[-100_000:]}\n"
            f"--- stderr ---\n{err[-100_000:]}", encoding="utf-8")
    except OSError:
        pass
    return rc, out, err


_SHA_MEMO: dict = {}


def _file_sha256_memo(path: Path) -> str:
    """`_file_sha256` memoized on (realpath, inode, size, mtime_ns, ctime_ns)."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = (real, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if key not in _SHA_MEMO:
        _SHA_MEMO[key] = _file_sha256(Path(real))
    return _SHA_MEMO[key]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SERVED_FLAGS = {"-t": "-t", "--threads": "-t", "-tb": "-tb", "--threads-batch": "-tb",
                 "-ub": "-ub", "--ubatch-size": "-ub", "-b": "-b", "--batch-size": "-b",
                 "--numa": "--numa"}


def served_launch(resolved_recipe) -> dict:
    """Round-10 N1: the SERVED recipe's execution shape for layers (b)-(d) -- the same
    topology prefix (taskset/numactl) layer (a) and serving use, the served thread
    counts, batch/ubatch and NUMA mode, read from the recipe's own command. Raises
    ValueError when the recipe has no thread count (never a silent default)."""
    prefix = tuple(getattr(resolved_recipe, "topology_prefix", ()) or ())
    argv = list(getattr(resolved_recipe, "command_argv", ()) or ())
    found: dict = {}
    for i, token in enumerate(argv[:-1]):
        if token in _SERVED_FLAGS:
            found[_SERVED_FLAGS[token]] = argv[i + 1]
    if "-t" not in found or not str(found["-t"]).isdigit() or int(found["-t"]) <= 0:
        raise ValueError("the served recipe carries no thread count (-t); refusing to run "
                         "the quality layers at a different partition")
    return {"prefix": prefix, "threads": int(found["-t"]),
            "args": tuple(item for flag in ("-tb", "-b", "-ub", "--numa") if flag in found
                          for item in (flag, str(found[flag])))}


def _launch_argv(launch: "dict | None", cpu_list: str, threads: int, *, batch_ok: bool):
    """(prefix, threads, extra args) for a tool run: the served launch when bound, else
    the unit-seam default. `batch_ok=False` drops `-b` (llama-perplexity sizes its own
    batch to the context; the served ubatch, threads and placement still apply)."""
    if launch is None:
        return ["taskset", "-c", cpu_list, "numactl", "--interleave=all"], threads, []
    args = list(launch["args"])
    if not batch_ok and "-b" in args:
        i = args.index("-b")
        del args[i:i + 2]
    return list(launch["prefix"]), int(launch["threads"]), args


def _rebind_ld(env: dict, build: Path, candidate_build: "Path | None") -> str:
    """Round-9 resolution D: the LD_LIBRARY_PATH a tool on `build` runs under -- the
    candidate arm's RESOLVED RECIPE value (the one layer (a) and serving use) with the
    candidate's own bin entry rebound to `build`'s bin, every other entry unchanged
    (exactly `run._cpu_arm`'s rebinding). Raises OSError when the recipe value is
    missing, carries an empty entry (an empty entry means the cwd), or does not name
    the candidate's bin. Without a recipe value AND without a candidate (unit seams
    only) the build's own bin is used."""
    value = (env or {}).get("LD_LIBRARY_PATH")
    if not value:
        if candidate_build is None:
            return str(Path(build) / "bin")
        raise OSError("the resolved recipe environment carries no LD_LIBRARY_PATH")
    if candidate_build is None:
        raise OSError("LD_LIBRARY_PATH is set but no candidate build names its bin entry")
    parts = value.split(":")
    if any(not part for part in parts):
        raise OSError("LD_LIBRARY_PATH carries an empty entry (the cwd); refusing")
    candidate_bin = os.path.realpath(Path(candidate_build) / "bin")
    hits = [i for i, part in enumerate(parts) if os.path.realpath(part) == candidate_bin]
    if not hits:
        raise OSError(f"LD_LIBRARY_PATH {value!r} does not name the candidate's bin "
                      f"{candidate_bin}; not the candidate's resolved recipe")
    for i in hits:
        parts[i] = str(Path(build) / "bin")
    return ":".join(parts)


def _build_identity(build: Path, tool: str, ld_path: "str | None" = None) -> str:
    """Content identity of what a `tool` run on `build` executes: the tool executable
    and EVERY shared object in `build/bin` (the kernel lives in libggml-cpu.so; the tool
    executable is often byte-identical across builds). Raises OSError when the tool is
    missing -- a cache key must never be computed for a run that cannot happen."""
    bin_dir = Path(build) / "bin"
    exe = bin_dir / tool
    if not exe.is_file():
        raise OSError(f"{exe} does not exist (the build lacks {tool!r}; "
                      f"add PPL_CONTRACT_TOOL_TARGETS to its build targets)")
    ld_path = ld_path or str(bin_dir)
    parts = [f"{tool}:{_file_sha256(exe)}", f"LD_LIBRARY_PATH={ld_path}"]
    libs = [lib for lib in sorted(bin_dir.glob("lib*.so*")) if lib.is_file()]
    # Round-9 resolution D: EVERY regular file in EVERY LD_LIBRARY_PATH directory the
    # tool actually runs under (the recipe's own, rebound), plus the build's bin.
    for directory in dict.fromkeys([str(bin_dir), *ld_path.split(":")]):
        for entry in sorted(Path(directory).iterdir()):
            if entry.is_file():
                parts.append(f"{directory}/{entry.name}:{_file_sha256_memo(entry)}")
    # The RESOLVED dependency closure under that exact loader path.
    parts.extend(_resolved_closure([exe, *libs], lib_dir=ld_path))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _resolved_closure(objects, *, lib_dir: "Path | None") -> list[str]:
    """`name:resolved-path:sha256` for every shared object the dynamic loader resolves
    for `objects` (ldd, with LD_LIBRARY_PATH as the gate runs set it). Raises OSError
    when the closure cannot be established -- an unresolved ("not found") dependency,
    a failed ldd -- so no cache key or pin is ever computed over an unknown closure."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if lib_dir is not None:
        env["LD_LIBRARY_PATH"] = str(lib_dir)
    resolved: dict[str, str] = {}
    for obj in objects:
        try:
            done = subprocess.run(["ldd", str(obj)], capture_output=True, text=True,
                                  timeout=60, env=env, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OSError(f"ldd {obj} could not run: {exc}") from exc
        if done.returncode != 0:
            raise OSError(f"ldd {obj} failed: {done.stderr.strip()[:200]}")
        for line in done.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # Round-5 resolution D: EVERY non-empty line must parse; paths may contain
            # spaces (the address is always the rightmost parenthesised group).
            if re.fullmatch(r"linux-(?:vdso|gate)\.so\.\d+(?:\s+=>\s*)?\s*\(0x[0-9a-f]+\)",
                            line):
                continue
            if re.fullmatch(r".+?\s+=>\s+not found", line):
                raise OSError(f"{obj}: unresolved dependency ({line})")
            arrow = re.fullmatch(r"(.+?)\s+=>\s+(/.+)\s+\(0x[0-9a-f]+\)", line)
            bare = re.fullmatch(r"(/.+)\s+\(0x[0-9a-f]+\)", line)
            if arrow is not None:
                name, path = arrow.group(1), arrow.group(2)
            elif bare is not None:
                name = path = bare.group(1)
            else:
                raise OSError(f"ldd {obj}: unparsed line {line!r}; refusing an unknown closure")
            resolved[os.path.realpath(path)] = name
    return [f"{name}:{path}:{_file_sha256(Path(path))}"
            for path, name in sorted(resolved.items())]


def _model_identity(model: Path) -> str:
    """Path + size + mtime of the (immutable, often >100 GB) GGUF; re-hashing it per
    candidate would cost minutes for no added safety."""
    real = Path(os.path.realpath(model))
    stat = real.stat()
    return f"{real}:{stat.st_size}:{stat.st_mtime_ns}"


def _env_identity(env: dict) -> str:
    """Every launch variable except the loader path (which is set per build), plus the
    CONTENT of every file `LD_PRELOAD` names (re-review 2026-10-06: an unchanged
    preload path over a replaced library changed execution without changing the key).
    A preload entry that cannot be hashed raises: never a key for an unknown binary."""
    loader = sorted(k for k in (env or {}) if k.startswith("LD_")
                    and k != "LD_LIBRARY_PATH"
                    and not (k == "LD_PRELOAD" and not str(env[k]).strip()))
    if loader:
        # Round-5 resolution E: LD_AUDIT and every other loader knob change what runs
        # without changing any hashed file; the gate refuses to run under them.
        raise OSError(f"launch env sets dynamic-loader variables {loader}; the ppl_contract "
                      "gate refuses to run under an unbound loader configuration")
    # Round-7 resolution D: LD_PRELOAD is refused outright (above); the gate runs every
    # tool with LD_LIBRARY_PATH = <build>/bin exactly, so the loader environment the
    # closure is resolved under (`_resolved_closure`) IS the execution environment.
    items = sorted((k, v) for k, v in (env or {}).items() if k != "LD_LIBRARY_PATH")
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def _cache_get(cache_dir: "Path | None", key: str):
    if cache_dir is None:
        return None
    try:
        body = json.loads((Path(cache_dir) / f"{key}.json").read_text(encoding="utf-8"))
        return body.get("value") if body.get("key") == key else None
    except (OSError, ValueError):
        return None


def _cache_put(cache_dir: "Path | None", key: str, value, meta: dict) -> None:
    if cache_dir is None:
        return
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tmp = Path(cache_dir) / f".{key}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"key": key, "value": value, "meta": meta}),
                       encoding="utf-8")
        os.replace(tmp, Path(cache_dir) / f"{key}.json")
    except OSError:
        pass


def _key(**fields) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def _corpus_text() -> str:
    """The pinned corpus, or ValueError (a moved corpus must not re-baseline silently)."""
    data = Path(PPL_CORPUS).read_bytes()
    if hashlib.sha256(data).hexdigest() != PPL_CORPUS_SHA256:
        raise ValueError(f"{PPL_CORPUS} sha256 differs from the pinned PPL_CORPUS_SHA256")
    return data.decode("utf-8", errors="replace")


def _corpus_slice(text: str, offset: int, length: int) -> str:
    """`length` chars from the first line start at/after `offset`, cut at a line end."""
    start = text.find("\n", offset)
    start = offset if start < 0 else start + 1
    chunk = text[start:start + length]
    cut = chunk.rfind("\n")
    return chunk[:cut + 1] if cut > 0 else chunk


def ppl_contract_prod_prompts() -> tuple[str, ...]:
    """Layer (c)'s production-length prompts: deterministic, pinned corpus slices."""
    text = _corpus_text()
    return tuple(_corpus_slice(text, off, n) for off, n in PPL_CONTRACT_PROD_PROMPT_SLICES)


def ppl_contract_canary_prompt() -> str:
    text = _corpus_text()
    return _corpus_slice(text, *PPL_CONTRACT_CANARY_SLICE)


def ppl_wikitext2(anchor_build: Path, candidate_build: Path, *, model: Path, threads: int,
                  env: dict, cpu_list: str, log_dir: Path,
                  reference_build: "Path | None" = None,
                  cache_dir: "Path | None" = None,
                  launch: "dict | None" = None,
                  _run_binary: "Callable[[Path], float | None] | None" = None) -> Verdict:
    # Round-9 resolution D: every arm runs under the candidate's resolved recipe env;
    # round-10 N1: and under the served launch shape (`launch`).
    """Layer (b): wikitext2 perplexity under the lane's CPU claim.

    Pass iff |ppl_c - ppl_a| / ppl_a <= PPL_REL_BAR (marginal) AND, when a fixed
    `reference_build` is given, |ppl_c - ppl_r| / ppl_r <= max(PPL_REL_BAR,
    |ppl_a - ppl_r| / ppl_r) (no cumulative growth). `ppl_contract_gate` always passes
    one. `_run_binary(build) -> ppl|None` is a test seam that REPLACES the subprocess."""
    def run(build: Path) -> float | None:
        if _run_binary is not None:
            return _run_binary(build)
        ld_path = _rebind_ld(env, build, candidate_build)
        prefix, run_threads, extra = _launch_argv(launch, cpu_list, threads, batch_ok=False)
        key = _key(kind="ppl-v2", build=_build_identity(build, "llama-perplexity", ld_path),
                   model=_model_identity(model), corpus=PPL_CORPUS_SHA256,
                   chunks=PPL_CHUNKS, ctx=PPL_CTX, batch=PPL_BATCH, threads=run_threads,
                   prefix=prefix, extra=extra, env=_env_identity(env))
        cached = _cache_get(cache_dir, key)
        if isinstance(cached, (int, float)):
            return float(cached)
        if hashlib.sha256(Path(PPL_CORPUS).read_bytes()).hexdigest() != PPL_CORPUS_SHA256:
            raise ValueError(f"{PPL_CORPUS} sha256 differs from PPL_CORPUS_SHA256")
        argv = [*prefix,
                str(Path(build) / "bin" / "llama-perplexity"), "-m", str(model),
                "-f", PPL_CORPUS, "-c", str(PPL_CTX), "--chunks", str(PPL_CHUNKS),
                "-t", str(run_threads), *extra, "-b", str(PPL_BATCH), "--no-mmap"]
        rc, out, err = _run_tool(argv, env={**env, "LD_LIBRARY_PATH": ld_path},
                                 log_dir=log_dir, label=f"ppl_{Path(build).name}")
        # Round-10 N3: the FINAL-estimate line only, anchored, exactly one value. This
        # tree prints it with LOG_INF (tools/perplexity/perplexity.cpp:654), i.e. on
        # STDERR -- a stdout-only parse would refuse every run. Absent or conflicting
        # estimates refuse.
        finals = set(re.findall(r"^Final estimate: PPL = ([0-9]+\.[0-9]+) \+/- [0-9.]+\s*$",
                                err, flags=re.M))
        if rc != 0 or len(finals) != 1:
            return None
        ppl = float(next(iter(finals)))
        if not ppl > 0.0:
            return None
        _cache_put(cache_dir, key, ppl, {"build": str(build), "model": str(model)})
        return ppl

    ppl_a = run(anchor_build)
    ppl_c = run(candidate_build) if ppl_a is not None else None
    ppl_r = (run(reference_build) if reference_build is not None and ppl_c is not None
             else None)
    if ppl_a is None or ppl_c is None or (reference_build is not None and ppl_r is None):
        return Verdict("ppl_wikitext2", False,
                       "perplexity run failed or reported no final estimate "
                       f"(anchor={ppl_a!r} candidate={ppl_c!r} reference={ppl_r!r})")
    rel = abs(ppl_c - ppl_a) / ppl_a
    reason = f"ppl anchor {ppl_a:.4f} candidate {ppl_c:.4f} rel {rel:.5f} (bar {PPL_REL_BAR})"
    passed = rel <= PPL_REL_BAR
    if ppl_r is not None:
        rel_r = abs(ppl_c - ppl_r) / ppl_r
        allowed = max(PPL_REL_BAR, abs(ppl_a - ppl_r) / ppl_r)
        passed = passed and rel_r <= allowed
        reason += (f"; vs fixed reference {ppl_r:.4f} rel {rel_r:.5f} "
                   f"(allowed {allowed:.5f})")
    return Verdict("ppl_wikitext2", passed, reason)


def ppl_contract_op_nmse(candidate_build: Path, *, resolved_recipe,
                         route_name: "str | None" = None,
                         ops: tuple[str, ...] = ("MUL_MAT", "MUL_MAT_ID"),
                         served_shape_manifest: "Path | None" = None,
                         routed_manifest: "Path | None" = None,
                         routed_lane: "str | None" = None,
                         _op_correctness=None,
                         _check_served_shape_case_set=None) -> Verdict:
    """Layer (a): test-backend-ops vs the independent `use_ref=true` reference at the
    served widths (decode N=1, verify 2-5), for EVERY quant type the suite carries --
    never narrowed to the route's witness quants: a whitelist / kernel-selection edit
    can reroute any type, so the oracle must cover any type. `_op_correctness` is a test
    seam with `op_correctness`'s signature.

    `served_shape_manifest` (review 2026-10-06): when given, this layer ALSO requires
    `check_served_shape_case_set` on the model's own served (k, m[, n_mats, n_used])
    shapes -- see `served_shape_cases.py` -- and FAILS CLOSED (never skips) when that
    manifest or the binary's case-set literal is missing: the generic sweep above has
    no served-shape evidence of its own. `None` (the default, every caller before this
    review) leaves this layer's pre-existing behaviour untouched -- the real pipeline
    call site now always passes a manifest path; a caller that does not is making an
    explicit, reviewable choice to run layer (a) without served-shape evidence.
    `_check_served_shape_case_set` is a test seam."""
    if route_name in PPL_CONTRACT_NO_OP_ORACLE:
        return Verdict("ppl_contract_nmse_unavailable", False,
                       f"route {route_name} has no op-level oracle: test-backend-ops never "
                       "allocates the CPU_REPACK extra buffer, so no case reaches its window")
    run = _op_correctness or op_correctness
    width_pattern = "|".join(str(w) for w in PPL_CONTRACT_SERVED_WIDTHS)
    params_filter = rf"(^|,)n=({width_pattern})(,|$)"
    for op in ops:
        verdict = run(candidate_build, op=op, backend="CPU", resolved_recipe=resolved_recipe,
                      params_filter=params_filter)
        if verdict is None or not verdict.passed:
            unavailable = verdict is None or verdict.gate == "oracle_unavailable"
            return Verdict("ppl_contract_nmse_unavailable" if unavailable
                           else "ppl_contract_nmse", False,
                           f"{op} served-width suite ({params_filter}) refused: "
                           f"{getattr(verdict, 'reason', 'no verdict')}",
                           getattr(verdict, "detail", ""))
    check = _check_served_shape_case_set or check_served_shape_case_set
    corpora = []
    if served_shape_manifest is not None:
        corpora.append(("served-shape", served_shape_manifest, {"lane": routed_lane}))
    if routed_manifest is not None:
        corpora.append((f"{routed_lane} served-expert routed", routed_manifest,
                        {"routed": True, "lane": routed_lane}))
    for label, manifest, extra in corpora:
        served_verdict = check(candidate_build, resolved_recipe=resolved_recipe,
                               manifest_path=manifest, **extra)
        if not served_verdict.passed:
            unavailable = served_verdict.gate == "oracle_unavailable"
            return Verdict("ppl_contract_nmse_unavailable" if unavailable
                           else "ppl_contract_nmse", False,
                           f"{label} suite refused: {served_verdict.reason}",
                           served_verdict.detail)
    return Verdict("ppl_contract_nmse", True,
                   f"served-width suite ({params_filter}), all types, passed for "
                   f"{', '.join(ops)}" + "".join(f"; {label} suite passed"
                                                 for label, _m, _e in corpora))


def _prefix_agreement(candidate, anchor) -> tuple[float, "int | None"]:
    """Prefix agreement between two sequences: matching prefix length / the LONGER
    length, so an early stop is a divergence, never a pass; two empty sequences are 0.0.
    Returns (ratio, first divergence index or None when identical)."""
    candidate, anchor = list(candidate), list(anchor)
    denom = max(len(candidate), len(anchor))
    if denom == 0:
        return 0.0, 0
    first_divergence = next(
        (i for i, (a, b) in enumerate(zip(candidate, anchor)) if a != b), None)
    if first_divergence is None and len(candidate) != len(anchor):
        first_divergence = min(len(candidate), len(anchor))
    agree = denom if first_divergence is None else first_divergence
    return agree / denom, first_divergence


def prefix_token_agreement(candidate_text: str, anchor_text: str, *,
                           tokenize: "Callable[[str], list[str]] | None" = None
                           ) -> tuple[float, "int | None"]:
    """Text-level prefix agreement (diagnostic only since the 2026-10-06 round-3 review:
    no character or whitespace metric is conservative w.r.t. token IDs, because two
    different token sequences can render identical bytes). The GATES compare generated
    token IDs (`prefix_id_agreement`)."""
    tok = tokenize or list
    return _prefix_agreement(tok(candidate_text), tok(anchor_text))


def prefix_id_agreement(candidate_ids, anchor_ids) -> tuple[float, "int | None"]:
    """Greedy agreement over GENERATED TOKEN IDS -- the layer (c)/(d) metric."""
    return _prefix_agreement(tuple(candidate_ids), tuple(anchor_ids))


LLAMA_SESSION_MAGIC, LLAMA_SESSION_VERSION = 0x6767736E, 10


def _session_tokens(path: Path) -> "list[int] | None":
    """The token list `llama_state_save_file` writes at the head of a session file
    (u32 magic 'ggsn', u32 version, u32 n, n x int32), or None on any mismatch."""
    import struct
    try:
        with open(path, "rb") as handle:
            head = handle.read(12)
            if len(head) != 12:
                return None
            magic, version, count = struct.unpack("<III", head)
            if magic != LLAMA_SESSION_MAGIC or version != LLAMA_SESSION_VERSION:
                return None
            raw = handle.read(4 * count)
            if len(raw) != 4 * count:
                return None
            return list(struct.unpack(f"<{count}i", raw))
    except OSError:
        return None


_SESSION_SAVE_LINE = re.compile(r"\n?[\w:]*: saving final output to session file '[^'\n]*'\n?")


def _completion(build: Path, prompt: str, n_predict: int, ctx: int, *, model: Path,
                threads: int, env: dict, cpu_list: str, log_dir: Path,
                cache_dir: "Path | None", label: str,
                candidate_build: "Path | None" = None, launch: "dict | None" = None
                ) -> "tuple[str, int, tuple[int, ...], tuple[int, ...]] | None":
    """Greedy raw completion on `build`: (generated text, prompt token count, GENERATED
    TOKEN IDS) or None.

    `llama-completion -no-cnv` (raw text, no chat template, no REPL), stdin closed, the
    prompt from a file and NOT echoed, EOS ignored. Token IDs come from the session file
    `--prompt-cache --prompt-cache-all` writes (the tool's own decoded token list; a
    fresh, unique path that cannot pre-exist, so no stale state is ever LOADED). The
    run is refused unless the session holds exactly prompt + (n_predict-1 or n_predict)
    tokens, the prompt count matches --verbose-prompt, and --perf reports the decodes:
    a prompt-only or short session can never be read as agreement (round-3 review)."""
    import uuid
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    ld_path = _rebind_ld(env, build, candidate_build)
    prefix, run_threads, extra = _launch_argv(launch, cpu_list, threads, batch_ok=True)
    key = _key(kind="completion-v6", build=_build_identity(build, "llama-completion", ld_path),
               model=_model_identity(model), prompt=prompt_sha, n_predict=n_predict,
               ctx=ctx, threads=run_threads, prefix=prefix, extra=extra,
               env=_env_identity(env))
    cached = _cache_get(cache_dir, key)
    if isinstance(cached, list) and len(cached) == 4 and isinstance(cached[2], list) \
            and isinstance(cached[3], list):
        return (str(cached[0]), int(cached[1]), tuple(int(t) for t in cached[2]),
                tuple(int(t) for t in cached[3]))
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = log_dir / f"prompt-{prompt_sha[:16]}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    session = log_dir / f"session-{uuid.uuid4().hex}.bin"
    if session.exists():
        return None
    argv = [*prefix,
            str(Path(build) / "bin" / "llama-completion"), "-m", str(model),
            "-f", str(prompt_file), "-n", str(n_predict), "-c", str(ctx),
            "-t", str(run_threads), *extra, "--temp", "0", "--top-k", "1", "--seed", "0",
            "-no-cnv", "--no-display-prompt", "--ignore-eos", "--verbose-prompt",
            "--perf", "--prompt-cache", str(session), "--prompt-cache-all", "--no-mmap"]
    try:
        rc, out, err = _run_tool(argv, env={**env, "LD_LIBRARY_PATH": ld_path},
                                 log_dir=log_dir, label=f"{label}_{Path(build).name}")
        tokens = _session_tokens(session) if rc == 0 else None
    finally:
        try:
            session.unlink()
        except OSError:
            pass
    found = re.search(r"number of tokens in prompt = (\d+)", err + out)
    decoded = re.search(r"(?<!prompt )eval time =\s*[0-9.]+ ms /\s*(\d+) runs", err + out)
    if rc != 0 or found is None or decoded is None or tokens is None:
        return None
    n_prompt = int(found.group(1))
    if n_prompt <= 0 or len(tokens) <= n_prompt:
        return None
    prompt_ids, generated = tokens[:n_prompt], tokens[n_prompt:]
    if (int(decoded.group(1)) < max(1, n_predict - 1)
            or len(generated) not in (n_predict - 1, n_predict)):
        return None
    text = _SESSION_SAVE_LINE.sub("", out)
    if not text.strip():
        return None
    result = (text, n_prompt, tuple(generated), tuple(prompt_ids))
    _cache_put(cache_dir, key, [text, n_prompt, list(generated), list(prompt_ids)],
               {"build": str(build), "label": label})
    return result


def _valid_generation(result) -> bool:
    """(text, n_prompt, generated ids, prompt ids) with the prompt ids exactly n_prompt
    long -- round-5 resolution F: prompt identity is checked on IDs, not counts."""
    return (isinstance(result, tuple) and len(result) == 4 and isinstance(result[0], str)
            and result[0].strip() != "" and isinstance(result[1], int)
            and isinstance(result[2], tuple) and len(result[2]) > 0
            and isinstance(result[3], tuple) and len(result[3]) == result[1])


def ppl_contract_coherence(anchor_build: Path, candidate_build: Path, *, model: Path,
                           prompts: tuple[str, ...], n_predict: int, env: dict,
                           cpu_list: str, log_dir: Path, threads: int = 1,
                           reference_build: "Path | None" = None,
                           cache_dir: "Path | None" = None,
                           launch: "dict | None" = None,
                           _generate: "Callable[[Path, str], tuple | None] | None" = None
                           ) -> Verdict:
    """Layer (c): greedy agreement over GENERATED TOKEN IDS on >=
    PPL_CONTRACT_PROD_PROMPT_COUNT_MIN prompts of >= PPL_CONTRACT_PROD_PROMPT_TOKENS_MIN
    prompt tokens each (the tool's own count).

    Per prompt: agreement(candidate, anchor) >= floor, and with a fixed reference,
    agreement(candidate, reference) >= min(floor, agreement(anchor, reference)).
    Missing token evidence refuses. `_generate(build, prompt) -> (text, prompt_tokens,
    generated_ids) | None` is a test seam."""
    def generate(build: Path, prompt: str):
        if _generate is not None:
            return _generate(build, prompt)
        return _completion(build, prompt, n_predict, PPL_CONTRACT_GEN_CTX, model=model,
                           threads=threads, env=env, cpu_list=cpu_list, log_dir=log_dir,
                           cache_dir=cache_dir, label="coherence",
                           candidate_build=candidate_build, launch=launch)
    if len(prompts) < PPL_CONTRACT_PROD_PROMPT_COUNT_MIN:
        return Verdict("ppl_contract_coherence", False,
                       f"{len(prompts)} production-length prompt(s) supplied; "
                       f"{PPL_CONTRACT_PROD_PROMPT_COUNT_MIN} required")
    worst_ratio, details = None, []
    for index, prompt in enumerate(prompts):
        builds = [("anchor", anchor_build), ("candidate", candidate_build)]
        if reference_build is not None:
            builds.append(("reference", reference_build))
        outputs = {}
        for name, build in builds:
            result = generate(build, prompt)
            if not _valid_generation(result):
                return Verdict("ppl_contract_coherence", False,
                               f"prompt {index}: {name} build produced no token-level "
                               "generation evidence (tool failed, missing, short, or no "
                               "generated token IDs)")
            outputs[name] = result
        tokens = {name: result[1] for name, result in outputs.items()}
        short = {name: n for name, n in tokens.items()
                 if n < PPL_CONTRACT_PROD_PROMPT_TOKENS_MIN}
        if short:
            return Verdict("ppl_contract_coherence", False,
                           f"prompt {index} is not production-length: prompt tokens {short} "
                           f"< {PPL_CONTRACT_PROD_PROMPT_TOKENS_MIN}")
        if len({result[3] for result in outputs.values()}) != 1:
            return Verdict("ppl_contract_coherence", False,
                           f"prompt {index}: prompt token IDs differ across builds "
                           f"(counts {tokens}); not the same input")
        ratio, first = prefix_id_agreement(outputs["candidate"][2], outputs["anchor"][2])
        line = (f"prompt {index} ({tokens['candidate']} tok): vs anchor {ratio:.4f} "
                f"(first differing generated token {first!r})")
        passed = ratio >= PPL_CONTRACT_AGREEMENT_FLOOR
        if reference_build is not None:
            ratio_r, _ = prefix_id_agreement(outputs["candidate"][2], outputs["reference"][2])
            base_r, _ = prefix_id_agreement(outputs["anchor"][2], outputs["reference"][2])
            allowed = min(PPL_CONTRACT_AGREEMENT_FLOOR, base_r)
            passed = passed and ratio_r >= allowed
            line += f", vs reference {ratio_r:.4f} (allowed >= {allowed:.4f})"
        details.append(line)
        if not passed:
            return Verdict("ppl_contract_coherence", False,
                           f"agreement below floor {PPL_CONTRACT_AGREEMENT_FLOOR}: {line}",
                           "\n".join(details))
        worst_ratio = ratio if worst_ratio is None else min(worst_ratio, ratio)
    return Verdict("ppl_contract_coherence", True,
                   f"worst token-ID agreement vs anchor {worst_ratio:.4f} over {len(prompts)} "
                   f"production-length prompts; " + "; ".join(details))


def _longest_exact_repeat_run(tokens: list) -> int:
    best = run = 1 if tokens else 0
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        best = max(best, run)
    return best


def _distinct_ngram_ratio(tokens: list, n: int = 4) -> float:
    """Distinct n-grams / total n-grams: ~1.0 for prose, collapses in a degenerate loop
    (a repeated PHRASE, which a single-token run length never sees)."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def ppl_contract_long_canary(candidate_build: Path, *, model: Path, prompt: str,
                             env: dict, cpu_list: str, log_dir: Path,
                             anchor_build: "Path | None" = None,
                             reference_build: "Path | None" = None, threads: int = 1,
                             cache_dir: "Path | None" = None,
                             n_predict: int = PPL_CONTRACT_CANARY_TOKENS,
                             launch: "dict | None" = None,
                             _generate: "Callable[[Path, str, int], tuple | None] | None" = None
                             ) -> Verdict:
    """Layer (d): a >= 1024-token greedy free generation (EOS ignored) on the candidate
    AND both baselines, compared by GENERATED TOKEN IDS over the whole length (round-3
    review: statistics alone accept an unrelated non-repetitive tail). Refused unless:
    an anchor AND a reference are given (none is "no evidence"); every arm produced
    the full token count; agreement(candidate, anchor) >= the agreement floor;
    agreement(candidate, reference) >= min(floor, agreement(anchor, reference)); and
    the token-level repetition statistics (longest repeat run, whole and last-quarter
    distinct-4) are no worse than every baseline's by more than the set margins.
    `_generate(build, prompt, count) -> (text, prompt_tokens, ids) | None` is a test seam."""
    if n_predict < PPL_CONTRACT_CANARY_TOKENS:
        raise ValueError(
            f"ppl_contract_long_canary requires >= {PPL_CONTRACT_CANARY_TOKENS} tokens")
    if anchor_build is None or reference_build is None:
        return Verdict("ppl_contract_long_canary", False,
                       "the canary needs both an anchor and a fixed reference to compare "
                       "against; a baseline-free canary is not evidence")

    def generate(build: Path):
        if _generate is not None:
            return _generate(build, prompt, n_predict)
        return _completion(build, prompt, n_predict, PPL_CONTRACT_CANARY_CTX, model=model,
                           threads=threads, env=env, cpu_list=cpu_list, log_dir=log_dir,
                           cache_dir=cache_dir, label="canary",
                           candidate_build=candidate_build, launch=launch)

    def stats(ids) -> tuple[float, float, int]:
        ids = list(ids)
        tail = ids[-max(16, len(ids) // 4):]
        return (_distinct_ngram_ratio(ids), _distinct_ngram_ratio(tail),
                _longest_exact_repeat_run(ids))

    outputs = {}
    for name, build in (("candidate", candidate_build), ("anchor", anchor_build),
                        ("reference", reference_build)):
        result = generate(build)
        if not _valid_generation(result) or len(result[2]) < n_predict - 1:
            return Verdict("ppl_contract_long_canary", False,
                           f"{name} build did not complete a {n_predict}-token generation "
                           "with token-level evidence")
        outputs[name] = result
    if len({result[3] for result in outputs.values()}) != 1:
        return Verdict("ppl_contract_long_canary", False,
                       "canary prompt token IDs differ across builds; not the same input")
    outputs = {name: result[2] for name, result in outputs.items()}
    cand = outputs["candidate"]
    ratio_a, first_a = prefix_id_agreement(cand, outputs["anchor"])
    ratio_r, _ = prefix_id_agreement(cand, outputs["reference"])
    base_r, _ = prefix_id_agreement(outputs["anchor"], outputs["reference"])
    allowed_r = min(PPL_CONTRACT_AGREEMENT_FLOOR, base_r)
    passed = ratio_a >= PPL_CONTRACT_AGREEMENT_FLOOR and ratio_r >= allowed_r
    distinct, tail, run = stats(cand)
    reason = (f"agreement vs anchor {ratio_a:.4f} (first differing token {first_a!r}), "
              f"vs reference {ratio_r:.4f} (allowed >= {allowed_r:.4f}); candidate "
              f"distinct-4 {distinct:.3f} tail {tail:.3f} longest run {run}")
    for name in ("anchor", "reference"):
        b_distinct, b_tail, b_run = stats(outputs[name])
        passed = (passed and run <= max(PPL_CONTRACT_CANARY_MAX_REPEAT_RUN, b_run)
                  and distinct >= b_distinct - PPL_CONTRACT_CANARY_DISTINCT4_DROP
                  and tail >= b_tail - PPL_CONTRACT_CANARY_DISTINCT4_DROP)
        reason += f"; {name} distinct-4 {b_distinct:.3f} tail {b_tail:.3f} run {b_run}"
    return Verdict("ppl_contract_long_canary", passed, reason)


#: Operator tiering (2026-10-06, "don't introduce excessive friction"): a CANDIDATE
#: keep pays only layer (a) (plus its route oracles); layers (b)-(d) -- with the fixed
#: reference preflight -- run once per BUNDLE, at the fold / serving-gate decision that
#: actually changes the champion of record.
PPL_CONTRACT_CANDIDATE_LAYERS = ("nmse",)
PPL_CONTRACT_BUNDLE_LAYERS = ("production_reference_load", "nmse", "ppl", "coherence",
                              "long_canary")


def _ppl_contract_layers(anchor_build: Path, candidate_build: Path, *, route_name, resolved_recipe,
                         model: Path, threads: int, cpu_list: str, env: dict, log_dir: Path,
                         reference_build: "Path | None", cache_dir: "Path | None",
                         prod_prompts, canary_prompt, _layers: dict | None,
                         gate_name: str,
                         served_shape_manifest: "Path | None" = None,
                         layer_names: tuple = PPL_CONTRACT_BUNDLE_LAYERS) -> Verdict:
    overrides = _layers or {}

    def layer(name: str, default):
        try:
            verdict = overrides.get(name, default)()
        except Exception as exc:  # noqa: BLE001 -- fail CLOSED, never silently pass
            return Verdict(f"ppl_contract_{name}_error", False,
                           f"layer {name!r} errored: {type(exc).__name__}: {exc}")
        if not isinstance(verdict, Verdict):
            return Verdict(f"ppl_contract_{name}_error", False,
                           f"layer {name!r} produced no verdict")
        return verdict

    if "nmse" in layer_names and served_shape_manifest is None and "nmse" not in overrides:
        return Verdict(gate_name, False, "no served-shape manifest: layer (a) would run "
                       "without served-shape evidence, refusing (re-review 2026-10-06)",
                       "failure_class=unavailable")
    needs_reference = {"production_reference_load", "ppl", "coherence",
                       "long_canary"} & set(layer_names)
    if needs_reference and reference_build is None and \
            not ({"ppl", "coherence", "long_canary"} & set(layer_names)) <= set(overrides):
        return Verdict(gate_name, False, "no fixed reference build: cumulative drift across "
                       "advancing anchors cannot be bounded, refusing",
                       "failure_class=unavailable")
    prompts = prod_prompts
    canary = canary_prompt
    launch = None
    if resolved_recipe is not None and getattr(resolved_recipe, "command_argv", None) and \
            ({"production_reference_load", "ppl", "coherence", "long_canary"}
             & set(layer_names)):
        try:
            launch = served_launch(resolved_recipe)
        except ValueError as exc:
            return Verdict(gate_name, False, f"{gate_name} refused: {exc}",
                           "failure_class=unavailable")
    layers = (
        ("production_reference_load", lambda: (
            check_production_reference_loads(
                reference_build, model=model, cpu_list=cpu_list, env=env,
                log_dir=log_dir, threads=threads, candidate_build=candidate_build,
                launch=launch)
            if reference_build is not None else
            Verdict("production_reference_load", True,
                   "no fixed reference build supplied (test seam)"))),
        ("nmse", lambda: ppl_contract_op_nmse(
            candidate_build, resolved_recipe=resolved_recipe, route_name=route_name,
            served_shape_manifest=served_shape_manifest,
            # Round-12: the 128-expert routed corpus runs at the BUNDLE tier only.
            routed_manifest=(Path(served_shape_manifest).with_name("manifest-routed.json")
                             if served_shape_manifest is not None
                             and "ppl" in layer_names else None),
            # the lane's own MoE profile (GGUF expert_count / expert_used_count)
            routed_lane=_lane_for_model(model))),
        ("ppl", lambda: ppl_wikitext2(
            anchor_build, candidate_build, model=model, threads=threads, env=env,
            cpu_list=cpu_list, log_dir=log_dir, reference_build=reference_build,
            cache_dir=cache_dir, launch=launch)),
        ("coherence", lambda: ppl_contract_coherence(
            anchor_build, candidate_build, model=model,
            prompts=prompts if prompts is not None else ppl_contract_prod_prompts(),
            n_predict=PPL_CONTRACT_PROD_GEN_TOKENS, env=env, cpu_list=cpu_list,
            log_dir=log_dir, threads=threads, reference_build=reference_build,
            cache_dir=cache_dir, launch=launch)),
        ("long_canary", lambda: ppl_contract_long_canary(
            candidate_build, model=model,
            prompt=canary if canary is not None else ppl_contract_canary_prompt(),
            env=env, cpu_list=cpu_list, log_dir=log_dir, anchor_build=anchor_build,
            reference_build=reference_build, threads=threads, cache_dir=cache_dir,
            launch=launch)),
    )
    layers = tuple(item for item in layers if item[0] in layer_names)
    verdicts = []
    for name, default in layers:   # cheapest first; the first refusal stops the spend
        verdict = layer(name, default)
        verdicts.append(verdict)
        if not verdict.passed:
            return Verdict(gate_name, False,
                           f"{gate_name} refused at layer {name}: {verdict.gate}: {verdict.reason}",
                           f"failure_class={_layer_failure_class(name, verdict)}\n" +
                           "\n".join(f"{v.gate}: passed={v.passed} {v.reason}" for v in verdicts))
    return Verdict(gate_name, True,
                   f"ppl_contract layers {', '.join(layer_names)} passed: " +
                   "; ".join(f"{v.gate}={v.reason}" for v in verdicts))


def _lane_for_model(model) -> "str | None":
    from . import served_shape_cases as ssc
    return ssc.lane_for_model(model)


def _layer_failure_class(name: str, verdict: Verdict) -> str:
    """Round-12: "numerical" only for a layer that MEASURED and found the build wrong;
    every tool/reference/parse/evidence failure is "unavailable" (a bisect must never
    evict a keep on missing evidence)."""
    reason = verdict.reason or ""
    numerical = {
        "nmse": verdict.gate == "ppl_contract_nmse",
        "ppl": verdict.gate == "ppl_wikitext2" and reason.startswith("ppl anchor"),
        "coherence": (verdict.gate == "ppl_contract_coherence"
                      and reason.startswith("agreement below floor")),
        "long_canary": (verdict.gate == "ppl_contract_long_canary"
                        and reason.startswith("agreement vs anchor")),
    }.get(name, False)
    return "numerical" if numerical else "unavailable"


def ppl_contract_failure_class(verdict: Verdict) -> str:
    """"numerical" / "unavailable" for a refused ppl_contract gate verdict (anything
    unrecognised is "unavailable")."""
    first = (verdict.detail or "").split("\n", 1)[0]
    return "numerical" if first == "failure_class=numerical" else "unavailable"


def ppl_contract_gate(anchor_build: Path, candidate_build: Path, *, route: "CpuSourceRoute",
                      resolved_recipe, model: Path, threads: int, cpu_list: str, env: dict,
                      log_dir: Path, reference_build: "Path | None" = None,
                      cache_dir: "Path | None" = None,
                      prod_prompts: "tuple[str, ...] | None" = None,
                      canary_prompt: "str | None" = None,
                      served_shape_manifest: "Path | None" = None,
                      _layers: dict | None = None) -> Verdict:
    """The blocking pre-keep check for a `numerics="ppl_contract"` CANDIDATE: layer (a)
    only (`PPL_CONTRACT_CANDIDATE_LAYERS`), fail-closed on a missing/errored layer.
    Layers (b)-(d) run once per bundle in `ppl_contract_bundle_gate` (operator tiering).
    `anchor_build`, `reference_build`, the prompts and the cache are accepted for call
    compatibility and unused at this tier.
    `served_shape_manifest` is layer (a)'s served-shape addendum (`ppl_contract_op_nmse`);
    `None` leaves layer (a) exactly as it ran before the 2026-10-06 served-shape review.
    `_layers` is a test seam (layer name -> callable returning `Verdict`)."""
    if route.numerics != "ppl_contract":
        raise ValueError(f"ppl_contract_gate called for a {route.numerics!r} route "
                         f"({route.route}); this gate is for ppl_contract routes only")
    if route.model_identity:
        raise ValueError(f"route {route.route} declares both model_identity and "
                         "ppl_contract numerics; the two contracts are mutually exclusive")
    return _ppl_contract_layers(
        anchor_build, candidate_build, route_name=route.route,
        resolved_recipe=resolved_recipe, model=model, threads=threads, cpu_list=cpu_list,
        env=env, log_dir=log_dir, reference_build=reference_build, cache_dir=cache_dir,
        prod_prompts=prod_prompts, canary_prompt=canary_prompt, _layers=_layers,
        gate_name="ppl_contract", served_shape_manifest=served_shape_manifest,
        layer_names=PPL_CONTRACT_CANDIDATE_LAYERS)


def ppl_contract_bundle_gate(cor_build: Path, tip_build: Path, *, resolved_recipe,
                             model: Path, threads: int, cpu_list: str, env: dict,
                             log_dir: Path, reference_build: "Path | None",
                             cache_dir: "Path | None" = None,
                             prod_prompts: "tuple[str, ...] | None" = None,
                             canary_prompt: "str | None" = None,
                             served_shape_manifest: "Path | None" = None,
                             _layers: dict | None = None) -> Verdict:
    """Layer (e) at FOLD time: the whole accumulated bundle (champion of record -> tip)
    re-judged by layers (a)-(d) before the champion of record may advance. Individually
    admitted ppl_contract keeps compound; only the bundle is what a fold ships."""
    return _ppl_contract_layers(
        cor_build, tip_build, route_name=None, resolved_recipe=resolved_recipe, model=model,
        threads=threads, cpu_list=cpu_list, env=env, log_dir=log_dir,
        reference_build=reference_build, cache_dir=cache_dir, prod_prompts=prod_prompts,
        canary_prompt=canary_prompt, _layers=_layers, gate_name="ppl_contract_bundle",
        served_shape_manifest=served_shape_manifest, layer_names=PPL_CONTRACT_BUNDLE_LAYERS)


def cpu_route_witness_quants(route: "CpuSourceRoute") -> tuple[str, ...]:
    """The route's reviewed witness quants (documentation/engagement only: layer (a) is
    deliberately NOT narrowed to them -- see `ppl_contract_op_nmse`)."""
    from . import cpu_route_witness
    witness = cpu_route_witness.WITNESSES.get(route.route)
    return witness.quants if witness is not None else ()


__all__ = ["BACKEND_OPS_SELECTORS", "BUILD_TIMEOUT_S", "CORRECTNESS_TIMEOUT_S",
           "CPU_SOURCE_ROUTES",
           "CPU_SOURCE_ROUTE_PATHS", "DEFAULT_TARGETS",
           "PROMOTION_TARGETS", "UNSELECTABLE_ON_ANCHOR", "Verdict",
           "backend_ops_selector", "compiles",
           "cpu_source_route", "deterministic",
           "affected_op_scope", "check_cpu_fa_case_set", "check_cpu_fa_perf_screen",
           "check_served_shape_case_set",
           "check_cpu_fa_reference", "check_cpu_gdn_reference", "check_cpu_iqk_reference",
           "check_cpu_route_reference", "gpu_graph_pool_hold_refusal",
           "no_fallback_dispatch", "op_correctness", "run_all",
           "PPL_CORPUS", "PPL_CORPUS_SHA256", "PPL_CHUNKS", "PPL_CTX", "PPL_BATCH",
           "PPL_REL_BAR", "PPL_CONTRACT_NO_OP_ORACLE", "PPL_CONTRACT_PROD_PROMPT_COUNT_MIN",
           "PPL_CONTRACT_PROD_PROMPT_SLICES", "PPL_CONTRACT_PROD_GEN_TOKENS",
           "PPL_CONTRACT_GEN_CTX", "PPL_CONTRACT_CANARY_SLICE", "PPL_CONTRACT_CANARY_CTX",
           "PPL_CONTRACT_CANARY_DISTINCT4_DROP", "PPL_CONTRACT_TOOL_TARGETS",
           "PRODUCTION_CPU_KERNEL", "production_cpu_reference_build",
           "check_production_reference_loads", "ppl_contract_paths",
           "ppl_contract_prod_prompts", "ppl_contract_canary_prompt",
           "ppl_contract_bundle_gate", "PPL_CONTRACT_LEDGER", "ppl_contract_only_paths",
           "ppl_contract_ledger_read", "ppl_contract_ledger_add", "ppl_contract_fold_required",
           "ppl_contract_anchor_obligated", "PPL_CONTRACT_TRAILER",
           "ppl_contract_commits_in_range", "PPL_CONTRACT_REFERENCE_PIN",
           "ppl_contract_range_requires_gate", "prefix_id_agreement",
           "PPL_CONTRACT_BIT_EXACT_TRAILER", "PPL_CONTRACT_ORACLE_PREFIX",
           "ppl_contract_relevant", "PPL_CONTRACT_CANDIDATE_LAYERS",
           "PPL_CONTRACT_BUNDLE_LAYERS", "served_launch", "ppl_contract_failure_class",
           "pinned_production_reference",
           "PPL_CONTRACT_SERVED_WIDTHS", "PPL_CONTRACT_AGREEMENT_FLOOR",
           "PPL_CONTRACT_PROD_PROMPT_TOKENS_MIN", "PPL_CONTRACT_CANARY_TOKENS",
           "PPL_CONTRACT_CANARY_MAX_REPEAT_RUN",
           "ppl_wikitext2", "ppl_contract_op_nmse", "prefix_token_agreement",
           "ppl_contract_coherence", "ppl_contract_long_canary", "ppl_contract_gate",
           "cpu_route_witness_quants"]
