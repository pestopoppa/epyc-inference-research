"""Served-shape `test-backend-ops` case set for the ppl_contract layer (a) gate
(review 2026-10-06, `gates.ppl_contract_op_nmse` HONEST LIMITATIONS).

WHY THIS EXISTS. Layer (a) ran the generic MUL_MAT/MUL_MAT_ID sweep (`k=256,
m=16/512`), test-backend-ops' own shapes, at the served widths -- never the model's
ACTUAL served (m, k, n_expert). A kernel that is correct at k=256 and wrong at
k=5120 (a stride/alignment/remainder-handling bug, the class of defect a generic
narrow-k sweep is least likely to exercise) would still pass layer (a) outright.
This module is the served-shape corpus that closes that gap: real (k, m, n_mats,
n_used) dims read directly off the lanes' own GGUF headers (never invented), at the
served widths (decode n=1, verify n=2-5), for the witness quant types the admitted
low-bit CPU routes edit.

PROVENANCE (2026-10-06, read-only header parse, no build/inference):
  * Q38FN dense: `/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf` ("qwen35" arch) --
    `embedding_length=5120`, `feed_forward_length=17408`. No `.expert_count` key and
    no `ffn_*_exps` tensors: Q38FN is DENSE-only (hybrid attention+SSM, no MoE),
    matching the task's own "Q38FN ... dense shapes" framing.
  * DS41 expert + dense: `/mnt/raid0/llm/models/deepseek-ai/
    DeepSeek-V4.1-Flash-DSpark.gguf` ("deepseek41-dspark" arch, the DSpark drafter
    extraction that shares the served backbone's FFN dims per its own
    `dspark.target_arch=deepseek41` field) -- `embedding_length=5120`,
    `expert_feed_forward_length=2304`, `expert_count=128`, `expert_used_count=3`,
    plus the dense shared-expert tensors `ffn_{gate,up,down}_shexp` at the same
    (5120, 2304) dims. `ffn_{gate,up}_exps.weight` is `[5120, 2304, 128]` and
    `ffn_down_exps.weight` is `[2304, 5120, 128]` in GGUF's `ne[]` (ggml-major,
    reversed from the C `(rows, cols)` convention) -- `k` (test-backend-ops'
    reduction dim) is `ne[0]`, `m` is `ne[1]`, `n_mats` is `ne[2]`.

SCOPE. Covers MUL_MAT (dense) and MUL_MAT_ID (expert), at `SERVED_WIDTHS` (must stay
identical to `gates.PPL_CONTRACT_SERVED_WIDTHS`; cross-checked by
`test_served_shape_cases.py`), for `WITNESS_TYPES` -- the six witness quants the
admitted low-bit CPU routes edit (`iqk_iquants_dequant`, `iqk_legacy_iq4nl`, the
ad-hoc dense Q4_K/Q5_K admission, `iqk_kquants_q6_iq4xs_dequant`). Registered via a
llama-tree patch (`backend_ops_patch_block`, the same mechanism as
`cpu_fa_reference.py`'s `cpu_fa_longctx_v1` set) behind a reviewed env selector, so
the generic test-backend-ops corpus is unaffected when the patch is not applied --
and `gates.check_served_shape_case_set` (below) FAILS CLOSED, never skips, when the
selected binary does not carry it: unlike `cpu_fa_schedule`'s case set there is no
redundant independent probe covering these shapes, so a 0/0 suite here would be read
as a pass with no evidence behind it.

THE THRESHOLD (`tightened_nmse_bound`). `max_nmse_err()` in `test-backend-ops.cpp` is
a per-subclass VIRTUAL override returning a flat `5e-4` for every MUL_MAT/MUL_MAT_ID
case regardless of shape -- the generic, type-and-shape-blind bound layer (a) already
had. This module's served-shape cases instantiate `test_mul_mat_served_shape`/
`test_mul_mat_id_served_shape` (the two subclasses `backend_ops_patch_block` emits),
which take an explicit per-case `max_nmse` and override `max_nmse_err()` to return
it, instead of the flat constant. The bound is never invented here: callers MUST
supply the anchor's OWN measured NMSE on that exact shape/type (the independent
`use_ref=true` native-CPU reference run on the anchor build), and
`tightened_nmse_bound` is only the FORMULA -- anchor NMSE times a small safety
factor, hard-capped at `SERVED_SHAPE_NMSE_CAP` (well below the generic 5e-4) and
floored at `SERVED_SHAPE_NMSE_FLOOR` so a near-zero anchor measurement (quantization
noise can read as 0.0 in a tiny sample) never produces a bound so tight it refuses
numerically-legitimate rounding. `backend_ops_patch_block` REFUSES (raises) to emit a
patch for any shape/type pair missing from the `anchor_nmse_by_shape` mapping it is
given -- a served-shape case this module cannot justify a bound for is never baked in
with a guessed one.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

MANIFEST_SCHEMA = "epyc.autokernel.served_shape_manifest.v1"

#: Must stay identical to `gates.PPL_CONTRACT_SERVED_WIDTHS` (decode n=1, verify
#: n=2-5); duplicated here rather than imported to keep this module import-cycle-free
#: of `gates` (which lazily imports modules like this one, never the reverse at
#: module scope). `test_served_shape_cases.py` cross-checks the two stay equal.
SERVED_WIDTHS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: The witness quants the admitted 2026-10-06 low-bit CPU routes edit
#: (`gates.CPU_SOURCE_ROUTES`: `iqk_iquants_dequant`, `iqk_legacy_iq4nl`, the ad-hoc
#: dense Q4_K/Q5_K admission, `iqk_kquants_q6_iq4xs_dequant`).
#:
#: Re-review 2026-10-06: the whitelist (`iqk_type_whitelist`) and kernel-selection
#: (`iqk_set_kernels_*`) routes can reroute ANY type their switches name, so the
#: served-shape oracle covers every type an iqk selection switch can dispatch (the
#: legacy, K-quant and i-quant families); `gates`' whitelist route refuses an edit
#: naming any type outside this set.
WITNESS_TYPES: tuple[str, ...] = (
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S")

#: `ggml_type_name()` spellings, which test-backend-ops prints in `vars()`.
GGML_TYPE_NAMES: dict[str, str] = {
    "Q4_0": "q4_0", "Q4_1": "q4_1", "Q5_0": "q5_0", "Q5_1": "q5_1", "Q8_0": "q8_0",
    "IQ4_NL": "iq4_nl", "Q2_K": "q2_K", "Q3_K": "q3_K", "Q4_K": "q4_K", "Q5_K": "q5_K",
    "Q6_K": "q6_K", "IQ4_XS": "iq4_xs", "IQ2_XXS": "iq2_xxs", "IQ2_XS": "iq2_xs",
    "IQ2_S": "iq2_s", "IQ3_XXS": "iq3_xxs", "IQ3_S": "iq3_s"}

CASE_SET_ID = "served_shape_lowbit_v1"
#: Round-12 (bundle tier): 128-expert MUL_MAT_ID cases with deterministic, dispersed,
#: high expert IDs at the verify widths, run only at bundle promotion.
ROUTED_CASE_SET_ID = "served_shape_lowbit_routed128_v1"
ROUTED_MANIFEST_SCHEMA = "epyc.autokernel.served_shape_routed_manifest.v1"
#: The env var the patched test-backend-ops reads for its CPU backend thread count
#: (the stock tool always uses hardware_concurrency()).
BACKEND_THREADS_ENV = "AUTOKERNEL_BACKEND_THREADS"
CASE_SET_ENV = "AUTOKERNEL_CORRECTNESS_CASE_SET"

#: Well below the generic sweep's flat 5e-4 bound: a "tightened" bound that could
#: reach 5e-4 would not be a tightening of the existing correctness floor at all.
SERVED_SHAPE_NMSE_CAP = 1e-4
#: A measured-zero anchor NMSE (quantization noise reading as exactly 0.0 in a small
#: sample) must not produce a bound of 0.0, which would refuse every candidate
#: including a bit-identical one to floating-point noise.
SERVED_SHAPE_NMSE_FLOOR = 1e-8


@dataclass(frozen=True)
class ServedShape:
    """One (op, k, m[, n_mats, n_used]) shape read off a lane's own GGUF header."""
    name: str
    op: str                 # "MUL_MAT" | "MUL_MAT_ID"
    k: int
    m: int
    n_mats: int = 1          # MUL_MAT_ID only: total experts (ne[2] of ffn_*_exps)
    n_used: int = 1          # MUL_MAT_ID only: experts routed per token
    routed: bool = False     # bundle-tier: deterministic dispersed high-ID routing

    def __post_init__(self):
        if self.routed and self.op != "MUL_MAT_ID":
            raise ValueError(f"{self.name}: only MUL_MAT_ID shapes can be routed")
        if self.op not in ("MUL_MAT", "MUL_MAT_ID"):
            raise ValueError(f"{self.name}: op must be MUL_MAT or MUL_MAT_ID, got {self.op!r}")
        if self.op == "MUL_MAT" and (self.n_mats != 1 or self.n_used != 1):
            raise ValueError(f"{self.name}: MUL_MAT carries no n_mats/n_used")
        if self.op == "MUL_MAT_ID" and not (1 <= self.n_used <= self.n_mats):
            raise ValueError(f"{self.name}: n_used must be in [1, n_mats]")


#: Real (k, m[, n_mats, n_used]) dims, 2026-10-06 GGUF header read (see module
#: docstring PROVENANCE). `k` is test-backend-ops' reduction dim == GGUF `ne[0]`;
#: `m` is the per-expert/per-row output dim == GGUF `ne[1]`.
SERVED_SHAPES: tuple[ServedShape, ...] = (
    # DS41 (deepseek41) expert matmuls: ffn_{gate,up}_exps [5120, 2304, 128],
    # ffn_down_exps [2304, 5120, 128]; expert_used_count=3.
    #
    # TEST EXPERT COUNT (operator 2026-10-06, layer (a) cost): the served model has 128
    # experts, the test cases carry 8. Per-expert arithmetic is expert-count-
    # independent: MUL_MAT_ID computes each routed (token, expert) pair as an ordinary
    # (m x k) . (k x 1) product over that ONE expert's weight slab -- the kernel, its
    # k-loop, row partitioning and activation quantization see only k, m, the width
    # and n_used; n_mats only sizes the id range and the weight allocation. 8 >= n_used
    # (3) keeps distinct routed experts per token and the same dispatch path, at 1/16
    # of the weights to quantize (128 x 2304 x 5120 = 1.5e9 -> 9.4e7 per case).
    ServedShape("ds41_expert_gate_up", "MUL_MAT_ID", k=5120, m=2304, n_mats=8, n_used=3),
    ServedShape("ds41_expert_down", "MUL_MAT_ID", k=2304, m=5120, n_mats=8, n_used=3),
    # DS41 dense shared-expert path: ffn_{gate,up}_shexp [5120, 2304],
    # ffn_down_shexp [2304, 5120] -- same dims as the expert path, MUL_MAT not _ID.
    ServedShape("ds41_dense_shexp_gate_up", "MUL_MAT", k=5120, m=2304),
    ServedShape("ds41_dense_shexp_down", "MUL_MAT", k=2304, m=5120),
    # Q38FN (qwen35) dense FFN: ffn_{gate,up} [5120, 17408], ffn_down [17408, 5120].
    ServedShape("q38fn_dense_ffn_gate_up", "MUL_MAT", k=5120, m=17408),
    ServedShape("q38fn_dense_ffn_down", "MUL_MAT", k=17408, m=5120),
)


#: Round-12 (bundle tier, operator honest-author model): the served 128 experts, with
#: ids chosen deterministically so a verify call (widths 2-5, n_used 3) routes to up to
#: 15 DISTINCT experts spread over 0-127 incl. 64-127 -- the dispatch/mapping logic
#: (expert-id masks, per-expert counts, row partitions) that 8-expert cases cannot reach.
#: Row r uses ids 127 - (29 r + 43 j) mod 128 for j < n_used (r=0: 127, 84, 41; r=1: 98,
#: 55, 12; ...). One type per iqk activation path plus the common served types: the
#: per-type kernels are already covered by the 8-expert corpus at every type.
ROUTED_SHAPES: tuple[ServedShape, ...] = (
    ServedShape("ds41_expert_gate_up_routed128", "MUL_MAT_ID", k=5120, m=2304, n_mats=128,
                n_used=3, routed=True),
    ServedShape("ds41_expert_down_routed128", "MUL_MAT_ID", k=2304, m=5120, n_mats=128,
                n_used=3, routed=True),
)
ROUTED_TYPES: tuple[str, ...] = ("Q4_0", "Q8_0", "Q4_K", "Q6_K", "IQ4_XS", "IQ3_S",
                                 "IQ2_XXS", "IQ4_NL")
ROUTED_WIDTHS: tuple[int, ...] = (2, 3, 4, 5)


def tightened_nmse_bound(anchor_nmse: float, *, factor: float = 3.0) -> float:
    """anchor NMSE * `factor`, capped at `SERVED_SHAPE_NMSE_CAP`, floored at
    `SERVED_SHAPE_NMSE_FLOOR`. Raises on a negative/non-finite input: a bound must
    never be derived from a measurement that cannot itself be trusted."""
    if not isinstance(anchor_nmse, (int, float)) or isinstance(anchor_nmse, bool) \
            or anchor_nmse != anchor_nmse or anchor_nmse in (float("inf"), float("-inf")) \
            or anchor_nmse < 0:
        raise ValueError(f"anchor_nmse must be a finite non-negative number, got {anchor_nmse!r}")
    if anchor_nmse >= SERVED_SHAPE_NMSE_CAP:
        # Re-review 2026-10-06: clipping would yield a bound BELOW the anchor's own
        # error, refusing an unchanged correct kernel. Incompatible calibration is
        # refused explicitly instead.
        raise ValueError(f"anchor_nmse {anchor_nmse!r} is at or above the cap "
                         f"{SERVED_SHAPE_NMSE_CAP}; this shape/type cannot be bound "
                         "tighter than the generic sweep -- investigate, do not clip")
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor <= 0:
        raise ValueError(f"factor must be a positive number, got {factor!r}")
    return min(SERVED_SHAPE_NMSE_CAP, max(SERVED_SHAPE_NMSE_FLOOR, anchor_nmse * factor))


@dataclass(frozen=True)
class ServedShapeCase:
    """One test-backend-ops case: a `ServedShape` at one quant type, one served width,
    with the max-NMSE bound this specific (shape, type) pair is admitted under."""
    shape: ServedShape
    type_a: str
    n: int
    max_nmse: float

    @property
    def name(self) -> str:
        return f"{self.shape.name}_{self.type_a.lower()}_n{self.n}"

    def vars(self) -> str:
        """`test_mul_mat[_id]_served_shape::vars()`, byte for byte (see `.cpp()`).

        Review 2026-10-06: test-backend-ops prints types with `ggml_type_name`
        (`iq3_s`, `q4_K`, `f32`), and the base `vars()` carries no `max_nmse` -- the
        first cut spelled `IQ3_S`/`F32` and relied on a `max_nmse` field the subclasses
        never printed, so the anchored selector matched 0 cases and the layer could
        never pass. The subclasses now override `vars()` to append the bound (binding
        the manifest to what the binary was compiled with)."""
        type_a = GGML_TYPE_NAMES[self.type_a]
        if self.shape.op == "MUL_MAT":
            return (f"type_a={type_a},type_b=f32,m={self.shape.m},n={self.n},"
                    f"k={self.shape.k},bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,"
                    f"max_nmse={self.max_nmse:g}")
        routed = ",routed=1" if self.shape.routed else ""
        return (f"type_a={type_a},type_b=f32,n_mats={self.shape.n_mats},"
                f"n_used={self.shape.n_used},b=0,m={self.shape.m},n={self.n},"
                f"k={self.shape.k}{routed},max_nmse={self.max_nmse:g}")

    def cpp(self) -> str:
        if self.shape.op == "MUL_MAT":
            return (f"        test_cases.emplace_back(new test_mul_mat_served_shape("
                    f"GGML_TYPE_{self.type_a}, GGML_TYPE_F32, {self.shape.m}, {self.n}, "
                    f"{self.shape.k}, {{1, 1}}, {{1, 1}}, {{0, 1, 2, 3}}, 0, 1, "
                    f"{self.max_nmse:g}));")
        cls = ("test_mul_mat_id_served_routed" if self.shape.routed
               else "test_mul_mat_id_served_shape")
        return (f"        test_cases.emplace_back(new {cls}("
                f"GGML_TYPE_{self.type_a}, GGML_TYPE_F32, {self.shape.n_mats}, "
                f"{self.shape.n_used}, false, {self.shape.m}, {self.n}, {self.shape.k}, "
                f"{self.max_nmse:g}));")


def corpus(routed: bool = False):
    """(shapes, types, widths) of the candidate corpus or the bundle-tier routed one."""
    if routed:
        return ROUTED_SHAPES, ROUTED_TYPES, ROUTED_WIDTHS
    return SERVED_SHAPES, WITNESS_TYPES, SERVED_WIDTHS


def case_set(anchor_nmse_by_shape: Mapping[tuple[str, str, int], float], *,
            factor: float = 3.0, routed: bool = False) -> tuple[ServedShapeCase, ...]:
    """Every `(ServedShape, type, width)` case, bound from `anchor_nmse_by_shape`
    keyed `(shape.name, type_a, n)` -- the independent reference run's measured NMSE on
    the anchor build at that exact shape, type AND width (re-review 2026-10-06: one
    calibration reused across widths 1-5 hid width-specific kernels). Raises
    `KeyError` naming the missing triple -- a served-shape case this caller cannot
    justify a bound for is never silently dropped OR silently given a guessed one."""
    cases = []
    shapes, types, widths = corpus(routed)
    for shape in shapes:
        for type_a in types:
            for width in widths:
                key = (shape.name, type_a, width)
                if key not in anchor_nmse_by_shape:
                    raise KeyError(f"no anchor NMSE measurement for {key}; refusing to "
                                   "bake a served-shape case with a guessed bound")
                bound = tightened_nmse_bound(anchor_nmse_by_shape[key], factor=factor)
                cases.append(ServedShapeCase(shape, type_a, width, bound))
    return tuple(cases)


def case_set_regex(cases: tuple[ServedShapeCase, ...]) -> str:
    """`test-backend-ops -p` selector: anchored exact alternation over `vars()`, so no
    other MUL_MAT/MUL_MAT_ID case (the generic sweep included) can ride along."""
    import re
    return "^(" + "|".join(re.escape(case.vars()) for case in cases) + ")$"


_SUBCLASSES_CPP = """\
// AutoKernel served-shape case subclasses (epyc-inference-research
// scripts/kernel_rnd/autokernel/loop/served_shape_cases.py, generated). Identical to
// test_mul_mat[_id] except max_nmse_err() returns a per-instance bound instead of the
// flat 5e-4 every generic case carries, so a served-shape case can be held to a
// tighter, shape-justified floor without touching the generic corpus's bound.
static std::string autokernel_served_shape_nmse_str(double v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", v);   // == Python f"{v:g}"
    return buf;
}
// Round-12: the CPU backend thread count the tool ACTUALLY uses -- the served -t,
// passed by the gate/calibration in AUTOKERNEL_BACKEND_THREADS (stock: hardware
// concurrency, whatever the affinity). Called from main's backend init.
static int autokernel_backend_threads() {
    const char * v = std::getenv("AUTOKERNEL_BACKEND_THREADS");
    if (v != nullptr && std::atoi(v) > 0) {
        return std::atoi(v);
    }
    return (int) N_THREADS;
}
// Round-12: deterministic, dispersed, high expert ids for the 128-expert routed cases:
// row r routes to 127 - (29 r + 43 j) mod n_mats for j < n_used; the rest of the row
// keeps a full permutation so the ids tensor stays well-formed.
static void autokernel_route_ids(ggml_context * ctx, int n_mats, int n_used) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != GGML_TYPE_I32 || ggml_is_view_op(t->op)) { continue; }
        for (int64_t r = 0; r < ggml_nrows(t); r++) {
            std::vector<int32_t> data(t->ne[0]);
            std::vector<char> used(n_mats, 0);
            int pos = 0;
            for (int j = 0; j < n_used && pos < t->ne[0]; j++) {
                const int id = n_mats - 1 - (int) ((r * 29 + j * 43) % n_mats);
                data[pos++] = id;
                used[id] = 1;
            }
            for (int i = 0; i < n_mats && pos < t->ne[0]; i++) {
                if (!used[i]) { data[pos++] = i; }
            }
            ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
        }
    }
}
struct test_mul_mat_served_shape : public test_mul_mat {
    const double max_nmse;
    test_mul_mat_served_shape(ggml_type type_a, ggml_type type_b, int64_t m, int64_t n,
            int64_t k, std::array<int64_t, 2> bs, std::array<int64_t, 2> nr,
            std::array<int64_t, 4> per, int64_t k_v, uint32_t o, double max_nmse)
        : test_mul_mat(type_a, type_b, m, n, k, bs, nr, per, k_v, o), max_nmse(max_nmse) {}
    double max_nmse_err() override { return max_nmse; }
    std::string vars() override {
        return test_mul_mat::vars() + ",max_nmse=" + autokernel_served_shape_nmse_str(max_nmse);
    }
};
struct test_mul_mat_id_served_shape : public test_mul_mat_id {
    const double max_nmse;
    test_mul_mat_id_served_shape(ggml_type type_a, ggml_type type_b, int n_mats,
            int n_used, bool b, int64_t m, int64_t n, int64_t k, double max_nmse)
        : test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k), max_nmse(max_nmse) {}
    double max_nmse_err() override { return max_nmse; }
    std::string vars() override {
        return test_mul_mat_id::vars() + ",max_nmse=" + autokernel_served_shape_nmse_str(max_nmse);
    }
};
struct test_mul_mat_id_served_routed : public test_mul_mat_id {
    const double max_nmse;
    test_mul_mat_id_served_routed(ggml_type type_a, ggml_type type_b, int n_mats,
            int n_used, bool b, int64_t m, int64_t n, int64_t k, double max_nmse)
        : test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k), max_nmse(max_nmse) {}
    double max_nmse_err() override { return max_nmse; }
    std::string vars() override {
        return test_mul_mat_id::vars() + ",routed=1,max_nmse=" + autokernel_served_shape_nmse_str(max_nmse);
    }
    void initialize_tensors(ggml_context * ctx) override {
        test_mul_mat_id::initialize_tensors(ctx);
        autokernel_route_ids(ctx, n_mats, n_used);
    }
};
"""


def backend_ops_patch_block(cases: tuple[ServedShapeCase, ...],
                            routed_cases: tuple[ServedShapeCase, ...] = ()) -> str:
    """The C++ the llama-tree patch adds: the subclasses above, plus a static helper
    (precedent: `cpu_fa_reference.backend_ops_patch_block`) registering the candidate
    corpus under CASE_SET_ID and the bundle-tier routed corpus under ROUTED_CASE_SET_ID,
    each ONLY when the reviewed selector names it."""
    lines = "\n".join(case.cpp() for case in cases)
    routed = "\n".join(case.cpp() for case in routed_cases)
    return (
        _SUBCLASSES_CPP +
        "static void autokernel_add_served_shape_lowbit_cases("
        "std::vector<std::unique_ptr<test_case>> & test_cases) {\n"
        f"    const char * case_set = std::getenv(\"{CASE_SET_ENV}\");\n"
        f"    if (case_set != nullptr && std::strcmp(case_set, \"{CASE_SET_ID}\") == 0) {{\n"
        f"{lines}\n"
        "    }\n"
        f"    if (case_set != nullptr && std::strcmp(case_set, \"{ROUTED_CASE_SET_ID}\") == 0) {{\n"
        f"{routed}\n"
        "    }\n"
        "}\n")


# --- calibration (operator 2026-10-06): measure the anchor's own per-case NMSE ------
#
# The calibration build carries the SAME shapes/types/widths under a separate selector,
# with a non-failing bound, and prints `AK_SERVED_NMSE\t<vars>\t<nmse>` for every case
# (the case's own `err()` -- test-backend-ops' NMSE against the use_ref=true reference).
# `calibrate_served_shapes` parses those lines into `case_set`'s per-(shape, type,
# width) input. Both blocks live between the same markers; staging one replaces the
# other, so a tree never carries both.
CALIBRATION_CASE_SET_ID = "served_shape_lowbit_calibrate_v1"
CALIBRATION_MARKER = "AK_SERVED_NMSE"
PATCH_BEGIN = "// AK-SERVED-SHAPE-BEGIN (generated by autokernel served_shape_cases.py)"
PATCH_END = "// AK-SERVED-SHAPE-END"
PATCH_CALL = "    autokernel_add_served_shape_lowbit_cases(test_cases);  // AK-SERVED-SHAPE-CALL"
_THREADS_STOCK = "            ggml_backend_set_n_threads_fn(backend.get(), N_THREADS);"
THREADS_PATCHED = ("            ggml_backend_set_n_threads_fn(backend.get(), "
                   "autokernel_backend_threads());  // AK-SERVED-SHAPE-THREADS")
_EVAL_SIGNATURE = "static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {"
_EVAL_FIRST_LINE = "    std::vector<std::unique_ptr<test_case>> test_cases;"

_COMMON_CPP = (
    "static std::string autokernel_served_shape_nmse_str(double v) {\n"
    "    char buf[32];\n"
    "    snprintf(buf, sizeof(buf), \"%g\", v);\n"
    "    return buf;\n"
    "}\n") + """// Round-12: the CPU backend thread count the tool ACTUALLY uses -- the served -t,
// passed by the gate/calibration in AUTOKERNEL_BACKEND_THREADS (stock: hardware
// concurrency, whatever the affinity). Called from main's backend init.
static int autokernel_backend_threads() {
    const char * v = std::getenv("AUTOKERNEL_BACKEND_THREADS");
    if (v != nullptr && std::atoi(v) > 0) {
        return std::atoi(v);
    }
    return (int) N_THREADS;
}
// Round-12: deterministic, dispersed, high expert ids for the 128-expert routed cases:
// row r routes to 127 - (29 r + 43 j) mod n_mats for j < n_used; the rest of the row
// keeps a full permutation so the ids tensor stays well-formed.
static void autokernel_route_ids(ggml_context * ctx, int n_mats, int n_used) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != GGML_TYPE_I32 || ggml_is_view_op(t->op)) { continue; }
        for (int64_t r = 0; r < ggml_nrows(t); r++) {
            std::vector<int32_t> data(t->ne[0]);
            std::vector<char> used(n_mats, 0);
            int pos = 0;
            for (int j = 0; j < n_used && pos < t->ne[0]; j++) {
                const int id = n_mats - 1 - (int) ((r * 29 + j * 43) % n_mats);
                data[pos++] = id;
                used[id] = 1;
            }
            for (int i = 0; i < n_mats && pos < t->ne[0]; i++) {
                if (!used[i]) { data[pos++] = i; }
            }
            ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
        }
    }
}
"""
_CALIBRATION_CPP = """\
// AutoKernel served-shape CALIBRATION subclasses: never fail (bound 1.0), print the
// case's NMSE vs the use_ref reference so the anchor's own error can be measured.
struct test_mul_mat_served_calib : public test_mul_mat {
    test_mul_mat_served_calib(ggml_type type_a, ggml_type type_b, int64_t m, int64_t n,
            int64_t k, std::array<int64_t, 2> bs, std::array<int64_t, 2> nr,
            std::array<int64_t, 4> per, int64_t k_v, uint32_t o)
        : test_mul_mat(type_a, type_b, m, n, k, bs, nr, per, k_v, o) {}
    double max_nmse_err() override { return 1.0; }
    std::string vars() override { return test_mul_mat::vars() + ",calibrate=1"; }
    double err(const float * a, const float * b, size_t n) override {
        const double e = test_case::err(a, b, n);
        fprintf(stdout, "AK_SERVED_NMSE\\t%s\\t%.17g\\n", vars().c_str(), e);
        fflush(stdout);
        return e;
    }
};
struct test_mul_mat_id_served_calib : public test_mul_mat_id {
    test_mul_mat_id_served_calib(ggml_type type_a, ggml_type type_b, int n_mats,
            int n_used, bool b, int64_t m, int64_t n, int64_t k)
        : test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k) {}
    double max_nmse_err() override { return 1.0; }
    std::string vars() override { return test_mul_mat_id::vars() + ",calibrate=1"; }
    double err(const float * a, const float * b, size_t n) override {
        const double e = test_case::err(a, b, n);
        fprintf(stdout, "AK_SERVED_NMSE\\t%s\\t%.17g\\n", vars().c_str(), e);
        fflush(stdout);
        return e;
    }
};
struct test_mul_mat_id_served_routed_calib : public test_mul_mat_id {
    test_mul_mat_id_served_routed_calib(ggml_type type_a, ggml_type type_b, int n_mats,
            int n_used, bool b, int64_t m, int64_t n, int64_t k)
        : test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k) {}
    double max_nmse_err() override { return 1.0; }
    std::string vars() override { return test_mul_mat_id::vars() + ",routed=1,calibrate=1"; }
    void initialize_tensors(ggml_context * ctx) override {
        test_mul_mat_id::initialize_tensors(ctx);
        autokernel_route_ids(ctx, n_mats, n_used);
    }
    double err(const float * a, const float * b, size_t n) override {
        const double e = test_case::err(a, b, n);
        fprintf(stdout, "AK_SERVED_NMSE\\t%s\\t%.17g\\n", vars().c_str(), e);
        fflush(stdout);
        return e;
    }
};
"""


def calibration_vars(shape: ServedShape, type_a: str, n: int) -> str:
    """`test_mul_mat[_id]_served_calib::vars()`, byte for byte."""
    name = GGML_TYPE_NAMES[type_a]
    if shape.op == "MUL_MAT":
        return (f"type_a={name},type_b=f32,m={shape.m},n={n},k={shape.k},bs=[1,1],"
                f"nr=[1,1],per=[0,1,2,3],k_v=0,o=1,calibrate=1")
    routed = ",routed=1" if shape.routed else ""
    return (f"type_a={name},type_b=f32,n_mats={shape.n_mats},n_used={shape.n_used},b=0,"
            f"m={shape.m},n={n},k={shape.k}{routed},calibrate=1")


def canonical_triples(routed: bool = False) -> tuple[tuple[ServedShape, str, int], ...]:
    shapes, types, widths = corpus(routed)
    return tuple((shape, type_a, width) for shape in shapes
                 for type_a in types for width in widths)


def calibration_triples() -> tuple[tuple[ServedShape, str, int], ...]:
    """Both corpora: the candidate corpus and the bundle-tier routed one."""
    return canonical_triples() + canonical_triples(routed=True)


def calibration_regex() -> str:
    import re
    return "^(" + "|".join(re.escape(calibration_vars(*t))
                           for t in calibration_triples()) + ")$"


def calibration_patch_block() -> str:
    lines = []
    for shape, type_a, n in calibration_triples():
        if shape.routed:
            lines.append(f"        test_cases.emplace_back(new test_mul_mat_id_served_routed_calib("
                         f"GGML_TYPE_{type_a}, GGML_TYPE_F32, {shape.n_mats}, {shape.n_used}, "
                         f"false, {shape.m}, {n}, {shape.k}));")
            continue
        if shape.op == "MUL_MAT":
            lines.append(f"        test_cases.emplace_back(new test_mul_mat_served_calib("
                         f"GGML_TYPE_{type_a}, GGML_TYPE_F32, {shape.m}, {n}, {shape.k}, "
                         f"{{1, 1}}, {{1, 1}}, {{0, 1, 2, 3}}, 0, 1));")
        else:
            lines.append(f"        test_cases.emplace_back(new test_mul_mat_id_served_calib("
                         f"GGML_TYPE_{type_a}, GGML_TYPE_F32, {shape.n_mats}, {shape.n_used}, "
                         f"false, {shape.m}, {n}, {shape.k}));")
    return (_COMMON_CPP + _CALIBRATION_CPP +
            "static void autokernel_add_served_shape_lowbit_cases("
            "std::vector<std::unique_ptr<test_case>> & test_cases) {\n"
            f"    const char * case_set = std::getenv(\"{CASE_SET_ENV}\");\n"
            f"    if (case_set != nullptr && std::strcmp(case_set, \"{CALIBRATION_CASE_SET_ID}\") == 0) {{\n"
            + "\n".join(lines) + "\n    }\n}\n")


def parse_calibration(output: str) -> dict:
    """{(shape name, type, width): NMSE} from calibration output -- the max over any
    repeated print of one case. Raises ValueError unless EVERY canonical triple has a
    finite, non-negative value and no unknown case was printed."""
    import math
    by_vars = {calibration_vars(*t): (t[0].name, t[1], t[2]) for t in calibration_triples()}
    found: dict = {}
    for line in output.splitlines():
        if not line.startswith(CALIBRATION_MARKER + "\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or parts[1] not in by_vars:
            raise ValueError(f"unexpected calibration line {line[:160]!r}")
        value = float(parts[2])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"non-finite/negative NMSE for {parts[1]}: {parts[2]}")
        key = by_vars[parts[1]]
        found[key] = max(value, found.get(key, 0.0))
    missing = [k for k in by_vars.values() if k not in found]
    if missing:
        raise ValueError(f"calibration measured {len(found)}/{len(by_vars)} cases; "
                         f"missing e.g. {missing[:3]}")
    return found


def apply_patch_block(test_backend_ops: Path, block: str) -> None:
    """Stage `block` into a llama tree's tests/test-backend-ops.cpp: replace any earlier
    served-shape block between the markers (else insert before make_test_cases_eval)
    and make sure make_test_cases_eval calls it once. Idempotent."""
    text = Path(test_backend_ops).read_text(encoding="utf-8")
    wrapped = f"{PATCH_BEGIN}\n{block.rstrip()}\n{PATCH_END}\n"
    if PATCH_BEGIN in text:
        start = text.index(PATCH_BEGIN)
        end = text.index(PATCH_END, start) + len(PATCH_END) + 1
        text = text[:start] + wrapped + text[end:]
    else:
        if text.count(_EVAL_SIGNATURE) != 1:
            raise ValueError("make_test_cases_eval() signature not found exactly once")
        at = text.index(_EVAL_SIGNATURE)
        text = text[:at] + wrapped + "\n" + text[at:]
    if PATCH_CALL not in text:
        at = text.index(_EVAL_SIGNATURE)
        first = text.index(_EVAL_FIRST_LINE, at)
        if text[at:first].count("\n") > 1:
            raise ValueError("unexpected make_test_cases_eval() prologue")
        cut = first + len(_EVAL_FIRST_LINE)
        text = text[:cut] + "\n" + PATCH_CALL + text[cut:]
    if THREADS_PATCHED not in text:
        if text.count(_THREADS_STOCK) != 1:
            raise ValueError("main()'s backend thread-count line not found exactly once")
        text = text.replace(_THREADS_STOCK, THREADS_PATCHED, 1)
    Path(test_backend_ops).write_text(text, encoding="utf-8")


def binary_has_calibration(build_dir: Path) -> bool:
    binary = Path(build_dir) / "bin" / "test-backend-ops"
    try:
        data = binary.read_bytes()
    except OSError:
        return False
    return (CALIBRATION_CASE_SET_ID.encode() in data and CALIBRATION_MARKER.encode() in data
            and BACKEND_THREADS_ENV.encode() in data)


def binary_has_case_set(build_dir: Path) -> bool:
    """True when the build's test-backend-ops carries the case-set selector literal
    (a 0/0 suite from a binary that lacks it is not evidence about anything -- and
    UNLIKE `cpu_fa_reference`'s FLASH_ATTN_EXT set, there is no redundant independent
    probe covering these shapes, so the caller must FAIL CLOSED, never skip, on
    `False`)."""
    binary = Path(build_dir) / "bin" / "test-backend-ops"
    try:
        data = binary.read_bytes()
    except OSError:
        return False
    return CASE_SET_ID.encode() in data and BACKEND_THREADS_ENV.encode() in data


def binary_has_routed_case_set(build_dir: Path) -> bool:
    binary = Path(build_dir) / "bin" / "test-backend-ops"
    try:
        data = binary.read_bytes()
    except OSError:
        return False
    return ROUTED_CASE_SET_ID.encode() in data and BACKEND_THREADS_ENV.encode() in data


_SHAPES_BY_NAME: dict[str, ServedShape] = {shape.name: shape for shape in SERVED_SHAPES}
_ROUTED_BY_NAME: dict[str, ServedShape] = {shape.name: shape for shape in ROUTED_SHAPES}


def write_manifest(path: Path, cases: tuple[ServedShapeCase, ...], *,
                   routed: bool = False) -> None:
    """Persist the EXACT baked case set (with its anchor-derived `max_nmse` bounds)
    beside the applied llama-tree patch, so a later gate run reads what was actually
    compiled in rather than recomputing (and potentially drifting from) it."""
    body = {"schema": ROUTED_MANIFEST_SCHEMA if routed else MANIFEST_SCHEMA,
            "case_set_id": ROUTED_CASE_SET_ID if routed else CASE_SET_ID,
           "cases": [{"shape_name": c.shape.name, "type_a": c.type_a, "n": c.n,
                      "max_nmse": c.max_nmse} for c in cases]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class ManifestRefused(ValueError):
    """A served-shape manifest is absent, malformed, or names an unknown shape."""


def load_manifest(path: Path, *, routed: bool = False) -> tuple[ServedShapeCase, ...]:
    """Reload a `write_manifest` file, re-resolving each row's `ServedShape` from
    `SERVED_SHAPES` by name -- never trusting the file's own k/m/n_mats/n_used, so a
    manifest that drifted from this module's canonical shapes is refused, not
    silently honoured. Raises `ManifestRefused` on anything short of a perfectly
    well-formed, fully resolvable manifest: this is the fail-closed boundary for the
    served-shape gate, so a missing or corrupt file must never read as "no cases"."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestRefused(f"{path}: cannot read served-shape manifest: {exc}") from exc
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise ManifestRefused(f"{path}: served-shape manifest is not valid JSON: {exc}") from exc
    schema = ROUTED_MANIFEST_SCHEMA if routed else MANIFEST_SCHEMA
    set_id = ROUTED_CASE_SET_ID if routed else CASE_SET_ID
    by_name = _ROUTED_BY_NAME if routed else _SHAPES_BY_NAME
    _shapes, types, widths = corpus(routed)
    if not isinstance(body, dict) or body.get("schema") != schema \
            or body.get("case_set_id") != set_id or not isinstance(body.get("cases"), list) \
            or not body["cases"]:
        raise ManifestRefused(f"{path}: served-shape manifest has the wrong shape or schema")
    cases: list[ServedShapeCase] = []
    for row in body["cases"]:
        if not isinstance(row, dict) or set(row) != {"shape_name", "type_a", "n", "max_nmse"}:
            raise ManifestRefused(f"{path}: malformed served-shape manifest row: {row!r}")
        shape = by_name.get(row["shape_name"])
        if shape is None:
            raise ManifestRefused(f"{path}: unknown shape {row['shape_name']!r}; "
                                  "this module's SERVED_SHAPES moved on without the manifest")
        if row["type_a"] not in types:
            raise ManifestRefused(f"{path}: {row['type_a']!r} is not a witness type")
        if row["n"] not in widths:
            raise ManifestRefused(f"{path}: n={row['n']!r} is not a served width")
        max_nmse = row["max_nmse"]
        if not isinstance(max_nmse, (int, float)) or isinstance(max_nmse, bool) \
                or not (0 < max_nmse <= SERVED_SHAPE_NMSE_CAP):
            raise ManifestRefused(f"{path}: max_nmse={max_nmse!r} is out of bounds "
                                  f"(0, {SERVED_SHAPE_NMSE_CAP}]")
        cases.append(ServedShapeCase(shape, row["type_a"], row["n"], float(max_nmse)))
    # Re-review 2026-10-06: exact canonical SET equality, duplicates refused -- a row
    # count alone accepted 180 copies of one case.
    keys = [(c.shape.name, c.type_a, c.n) for c in cases]
    canonical = {(t[0].name, t[1], t[2]) for t in canonical_triples(routed)}
    if len(keys) != len(set(keys)):
        raise ManifestRefused(f"{path}: manifest repeats a (shape, type, width) case")
    if set(keys) != canonical:
        missing = sorted(canonical - set(keys))[:5]
        raise ManifestRefused(f"{path}: manifest is not the full served-shape corpus "
                              f"({len(set(keys))}/{len(canonical)}; missing e.g. {missing})")
    return tuple(cases)


__all__ = ["CASE_SET_ENV", "CASE_SET_ID", "MANIFEST_SCHEMA", "ManifestRefused",
           "SERVED_SHAPE_NMSE_CAP", "SERVED_SHAPE_NMSE_FLOOR", "SERVED_SHAPES",
           "SERVED_WIDTHS", "WITNESS_TYPES", "GGML_TYPE_NAMES", "ServedShape", "ServedShapeCase",
           "backend_ops_patch_block", "binary_has_case_set", "case_set", "case_set_regex",
           "load_manifest", "tightened_nmse_bound", "write_manifest",
           "CALIBRATION_CASE_SET_ID", "CALIBRATION_MARKER", "PATCH_BEGIN", "PATCH_END",
           "PATCH_CALL", "calibration_vars", "canonical_triples", "calibration_regex",
           "calibration_patch_block", "parse_calibration", "apply_patch_block",
           "binary_has_calibration", "ROUTED_CASE_SET_ID", "ROUTED_SHAPES",
           "ROUTED_TYPES", "ROUTED_WIDTHS", "BACKEND_THREADS_ENV", "corpus",
           "calibration_triples", "binary_has_routed_case_set", "THREADS_PATCHED"]
