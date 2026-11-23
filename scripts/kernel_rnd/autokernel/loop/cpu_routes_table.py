"""CPU source-route data (AK-TRIM-OR-SPLIT-GATES): the route table and its dataclasses,
moved verbatim out of gates.py, which re-exports every name."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

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
    #: AKX-AUTO-9: a route GENERATED from `route_defaults` for a compiled CPU file no
    #: hand-listed route names. It has no body confinement; its safety is the lexical
    #: refusal of every added line plus the oracle class's own gate (ppl_contract
    #: layers, or whole-model identity), with the thresholds those gates already carry.
    generated: bool = False


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
        # served_shape_cases.ALL_TYPES (WITNESS_TYPES + LANE_SERVED_TYPES; MXFP4 since
        # 2026-10-21, covered by the DS41-1017 lane corpus, the only lane serving it);
        # a whitelist edit naming any OTHER type could
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
        forbidden_added=(r"^\+(?!(?:\s*case\s+GGML_TYPE_(?:Q4_0|Q4_1|Q5_0|Q5_1|Q8_0|IQ4_NL|Q2_K|Q3_K|Q4_K|Q5_K|Q6_K|IQ4_XS|IQ2_XXS|IQ2_XS|IQ2_S|IQ3_XXS|IQ3_S|MXFP4)\s*:\s*(?://(?:.*[^\\\s])?\s*)?"
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
    # 2026-10-21 (DS41-1017 seeds S2-S4, store inbox 30): the x86 repack GEMV that serves
    # every DS41-1017 MXFP4 expert (`gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>`, reached
    # from `ggml_gemv_mxfp4_8x8_q8_0`). The template is SHARED with block_q4_0x8 and
    # block_iq4_nlx8 (Q38FN serves IQ4_NL through it), so a keep here runs the shared-type
    # cases of the CPU_REPACK oracle too (`cpu_repack_cases.touches_repack`). No
    # test-backend-ops case reaches it (PPL_CONTRACT_NO_OP_ORACLE); its op oracle is
    # `check_cpu_repack_case_set`, and a lane without a calibrated CPU_REPACK manifest
    # refuses the keep.
    CpuSourceRoute(
        route="cpu_repack_gemv",
        path="ggml/src/ggml-cpu/arch/x86/repack.cpp",
        symbols=("gemv_q4_b32_8x8_q8_0_lut_avx", "ggml_gemv_mxfp4_8x8_q8_0"),
        bodies=(("gemv_q4_b32_8x8_q8_0_lut_avx",
                 "static void gemv_q4_b32_8x8_q8_0_lut_avx(int n, float * GGML_RESTRICT s, size_t bs, "
                 "const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, "
                 "__m256i signextendlut) {"),
                ("ggml_gemv_mxfp4_8x8_q8_0",
                 "void ggml_gemv_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, "
                 "const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {")),
        ops=("MUL_MAT", "MUL_MAT_ID"),
        new_helpers=True,
        forbidden_added=r"getenv\(|#pragma omp|_Pragma|#define|#undef|#include \"",
        numerics="ppl_contract",
        admitted_text=("hunks inside the shared 8x8 q4-b32 GEMV template body or the MXFP4 GEMV "
                       "entry, plus new static helpers; the GEMM template, the other entries, the "
                       "block layouts and the repack itself unchanged")),
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
