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

PROVENANCE (2026-10-06 round 13, read-only header parse of the SERVED models -- the
models in each lane's resolved launch record, not a drafter or a sibling model):
  * DS41: `/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/
    DeepSeek-V4.1-Flash-Q4.gguf` ("deepseek41"): `n_routed_experts=384`,
    `num_experts_per_tok=6`, `n_shared_experts=1`; `ffn_{gate,up}_exps [5120, 2304, 384]`
    and `ffn_down_exps [2304, 5120, 384]` (Q4_K); `ffn_*_shexp` at (5120, 2304) (Q8_0).
  * Q38FN: `/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/
    Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf` ("qwen4exp"):
    `expert_count=512`, `expert_used_count=10`, `embedding_length=2560`,
    `expert_feed_forward_length=640`; `ffn_{gate,up}_exps [2560, 640, 512]` (IQ3_S /
    IQ4_XS), `ffn_down_exps [640, 2560, 512]` (IQ4_NL / Q8_0); `ffn_*_shexp` at
    (2560, 640) (Q8_0).
  (The first cut read the DSpark drafter -- 128 experts / 3 used -- for DS41 and a dense
  Qwen3.8-27B GGUF for Q38FN; both were the wrong models.) GGUF `ne[]` is ggml-major:
  `k` (test-backend-ops' reduction dim) is `ne[0]`, `m` is `ne[1]`, `n_mats` is
  `ne[2]`. `LANE_PROFILES` records these values; `moe_profile_from_gguf` re-derives
  them from the files, and calibration refuses a lane whose GGUF disagrees.

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
factor, capped at `SERVED_SHAPE_NMSE_CAP` (the generic test-backend-ops 5e-4) and
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
#: Round-15: the input-seeding scheme the generated subclasses use (recorded in every
#: manifest; a manifest from another scheme -- e.g. the random-input calibrations --
#: is refused).
SEED_SCHEME = "ak-served-shape-fnv1a64-casekey-v1"
AK_SERVED_SHAPE_SEED = 0x414B53455256
#: Round-17: compiled into every generated block (and printed once per run on stderr);
#: a test-backend-ops without it was built before seeding and draws random inputs.
SEED_MARKER = "AK_SEED_SCHEME=" + SEED_SCHEME
SEED_REBUILD_HINT = ("was built before deterministic per-case seeding (no "
                     f"{SEED_MARKER} literal) and would measure random inputs")
#: Round-17 debug mode: the env var / output marker of the per-case input hash.
INPUT_HASH_ENV = "AUTOKERNEL_INPUT_HASH"
INPUT_HASH_MARKER = "AK_INPUT_HASH"


def case_key(shape: "ServedShape", type_a: str, n: int) -> str:
    """The bound-free case key the generated C++ hashes: the base `vars()` (plus
    `,routed=1` for a routed case) -- identical for the bound and calibrate variants."""
    return calibration_vars(shape, type_a, n)[:-len(",calibrate=1")]


def case_seed_index(key: str) -> int:
    """FNV-1a-style 64-bit hash of `key`, exactly as `autokernel_seed_case` computes it.
    The offset basis is the tree's own `suite_seed_hash_string` constant
    (0x14650FB0739D0383, decimal one digit short of canonical FNV's) -- kept identical so
    both sides agree; only determinism matters here, not FNV conformance."""
    h = 0x14650FB0739D0383
    for byte in key.encode():
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
CASE_SET_ENV = "AUTOKERNEL_CORRECTNESS_CASE_SET"

#: Operator 2026-10-06 (after the first real calibration): the cap is the generic
#: test-backend-ops MUL_MAT/MUL_MAT_ID bound, 5e-4. The per-case bound stays TIGHT
#: wherever the anchor is accurate (anchor x factor, floored); the cap only stops it
#: from ever being looser than the generic check. (1e-4 refused 12 IQ3_XXS shared-
#: expert cases whose anchor itself measured 2e-4 .. 5.25e-4.)
SERVED_SHAPE_NMSE_CAP = 5e-4
#: The previous cap. Review of bb169f4b: raising the cap must not LOOSEN any case the
#: old policy accepted, so every anchor below 1e-4 keeps exactly its old bound
#: min(max(anchor x factor, floor), 1e-4); only anchors in [1e-4, 5e-4) use the new
#: min(anchor x factor, 5e-4) path.
SERVED_SHAPE_NMSE_LEGACY_CAP = 1e-4
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


@dataclass(frozen=True)
class MoeProfile:
    """One lane's served MoE FFN, as its GGUF header states it."""
    lane: str
    model: str               # first GGUF file of the served model
    expert_count: int
    expert_used: int
    hidden: int              # embedding_length == ffn_{gate,up}_exps ne[0]
    expert_ff: int           # expert_feed_forward_length == ffn_{gate,up}_exps ne[1]
    shexp_ff: int            # shared-expert ffn length (ffn_{gate,up}_shexp ne[1])
    gate_up_types: tuple[str, ...]  # quant types of ffn_{gate,up}_exps
    down_types: tuple[str, ...]     # quant types of ffn_down_exps
    extra_types: tuple[str, ...] = ()   # operator-listed served types beyond the experts

    @property
    def expert_types(self) -> tuple[str, ...]:
        return tuple(t for t in WITNESS_TYPES
                     if t in set(self.gate_up_types) | set(self.down_types))


#: Derived from each lane's served GGUF (see PROVENANCE; `moe_profile_from_gguf`).
LANE_PROFILES: dict[str, MoeProfile] = {
    "ds41": MoeProfile(
        "ds41", "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/"
        "DeepSeek-V4.1-Flash-Q4.gguf", expert_count=384, expert_used=6, hidden=5120,
        expert_ff=2304, shexp_ff=2304, gate_up_types=("Q4_K",), down_types=("Q4_K",)),
    "q38fn": MoeProfile(
        "q38fn", "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", expert_count=512, expert_used=10,
        hidden=2560, expert_ff=640, shexp_ff=640,
        gate_up_types=("IQ3_S", "IQ4_XS"), down_types=("IQ4_NL", "Q8_0"),
        extra_types=("Q6_K",)),
}


def candidate_experts(profile: MoeProfile) -> int:
    """The CANDIDATE corpus' test expert count: 8, or the next power of two that still
    routes `expert_used` DISTINCT experts per token (Q38FN: 10 used -> 16).

    Per-expert arithmetic is expert-count-independent: MUL_MAT_ID computes each routed
    (token, expert) pair as an ordinary (m x k) . (k x 1) product over that ONE
    expert's weight slab -- the kernel, its k-loop, row partitioning and activation
    quantization see only k, m, the width and n_used; n_mats only sizes the id range
    and the weight allocation. The served expert count (and its high ids) is exercised
    at the bundle tier by the routed corpus."""
    n = 8
    while n < profile.expert_used:
        n *= 2
    return n


def _lane_served_shapes(profile: MoeProfile) -> tuple[ServedShape, ...]:
    lane, n = profile.lane, candidate_experts(profile)
    return (
        ServedShape(f"{lane}_expert_gate_up", "MUL_MAT_ID", k=profile.hidden,
                    m=profile.expert_ff, n_mats=n, n_used=profile.expert_used),
        ServedShape(f"{lane}_expert_down", "MUL_MAT_ID", k=profile.expert_ff,
                    m=profile.hidden, n_mats=n, n_used=profile.expert_used),
        ServedShape(f"{lane}_shexp_gate_up", "MUL_MAT", k=profile.hidden, m=profile.shexp_ff),
        ServedShape(f"{lane}_shexp_down", "MUL_MAT", k=profile.shexp_ff, m=profile.hidden),
    )


#: The CANDIDATE corpus: every lane's real per-expert and shared-expert dims (cheap
#: expert counts, see `candidate_experts`), every witness type, widths 1-5.
SERVED_SHAPES: tuple[ServedShape, ...] = tuple(
    shape for profile in LANE_PROFILES.values() for shape in _lane_served_shapes(profile))


#: Round-12/13 (bundle tier, honest-author model): each lane's SERVED expert count and
#: expert_used, with ids chosen deterministically so a verify call (widths 2-5) routes
#: to up to 5 * expert_used DISTINCT experts spread over the whole id range -- the
#: dispatch/mapping logic (expert-id masks, per-expert counts, row partitions) that
#: small-expert cases cannot reach. Row r uses ids
#: n_mats - 1 - (29 r + step j) mod n_mats, step = (n_mats / n_used) | 1, j < n_used
#: (Q38FN r=0: 511, 460, 409, 358, 307, 256, 205, 154, 103, 52; DS41 r=0: 383, 318, ...).
ROUTED_WIDTHS: tuple[int, ...] = (2, 3, 4, 5)


def route_ids(n_mats: int, n_used: int, row: int) -> tuple[int, ...]:
    """The routed ids of row `row` -- byte-for-byte what `autokernel_route_ids` sets."""
    step = (n_mats // n_used) | 1
    return tuple(n_mats - 1 - (row * 29 + j * step) % n_mats for j in range(n_used))


def lane_served_shapes(lane: str) -> tuple[ServedShape, ...]:
    """Round-15 partition: the candidate corpus a lane measures -- its OWN served shapes
    plus any other lane's shape with the identical (op, k, m), i.e. a shape this lane's
    selection routes reach with the same kernels (none today: DS41 5120x2304 and Q38FN
    2560x640 share no dims)."""
    own = _lane_served_shapes(LANE_PROFILES[lane])
    dims = {(shape.op, shape.k, shape.m) for shape in own}
    shared = tuple(shape for other, profile in LANE_PROFILES.items() if other != lane
                   for shape in _lane_served_shapes(profile)
                   if (shape.op, shape.k, shape.m) in dims)
    return own + shared


def routed_shapes(lane: str) -> tuple[ServedShape, ...]:
    profile = LANE_PROFILES[lane]
    return (
        ServedShape(f"{lane}_expert_gate_up_routed{profile.expert_count}", "MUL_MAT_ID",
                    k=profile.hidden, m=profile.expert_ff, n_mats=profile.expert_count,
                    n_used=profile.expert_used, routed=True),
        ServedShape(f"{lane}_expert_down_routed{profile.expert_count}", "MUL_MAT_ID",
                    k=profile.expert_ff, m=profile.hidden, n_mats=profile.expert_count,
                    n_used=profile.expert_used, routed=True),
    )


#: ggml block size (elements) per witness type: a MUL_MAT over a type needs
#: k % block == 0 (Q38FN's down shape has k=640, invalid for every 256-block type).
BLOCK_SIZE: dict[str, int] = {
    "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32, "IQ4_NL": 32,
    "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256, "IQ4_XS": 256,
    "IQ2_XXS": 256, "IQ2_XS": 256, "IQ2_S": 256, "IQ3_XXS": 256, "IQ3_S": 256}


def type_fits(shape: ServedShape, type_a: str) -> bool:
    return shape.k % BLOCK_SIZE[type_a] == 0


def _direction(shape: ServedShape) -> str:
    return "down" if "_down" in shape.name else "gate_up"


def routed_types(lane: str, shape: "ServedShape | None" = None) -> tuple[str, ...]:
    """The lane's served expert types for `shape`'s direction (gate/up vs down, as the
    served GGUF's tensors carry them) plus its operator-listed extras, valid for the
    shape's k, in witness order. Without a shape: the union over both directions."""
    profile = LANE_PROFILES[lane]
    if shape is None:
        wanted = set(profile.gate_up_types) | set(profile.down_types) | set(profile.extra_types)
        return tuple(t for t in WITNESS_TYPES if t in wanted)
    direct = profile.down_types if _direction(shape) == "down" else profile.gate_up_types
    wanted = set(direct) | set(profile.extra_types)
    return tuple(t for t in WITNESS_TYPES if t in wanted and type_fits(shape, t))


def lane_for_model(model) -> "str | None":
    """The lane whose served GGUF `model` is (by resolved path, shard-tolerant)."""
    import os
    if model is None:
        return None
    real = os.path.realpath(str(model))
    for lane, profile in LANE_PROFILES.items():
        if real == os.path.realpath(profile.model) or \
                os.path.dirname(real) == os.path.dirname(os.path.realpath(profile.model)):
            return lane
    return None


def moe_profile_from_gguf(paths, lane: str, *,
                          gguf_py: str = "/mnt/raid0/llm/llama.cpp/gguf-py") -> MoeProfile:
    """Re-derive a lane's `MoeProfile` from its GGUF shard(s) (read-only header parse).
    Expert counts come from `<arch>.expert_count` / `<arch>.n_routed_experts`, used
    counts from `<arch>.expert_used_count` / `<arch>.num_experts_per_tok`; dims and
    types from the `ffn_*_exps` / `ffn_*_shexp` tensors. Raises ValueError when any of
    these is missing."""
    import sys as _sys
    if gguf_py not in _sys.path:
        _sys.path.insert(0, gguf_py)
    from gguf import GGUFReader  # noqa: PLC0415
    values, types, dims = {}, set(), {}
    for path in paths:
        reader = GGUFReader(str(path))
        for field in reader.fields.values():
            name = field.name.split(".", 1)[-1]
            if name in ("expert_count", "n_routed_experts", "expert_used_count",
                        "num_experts_per_tok"):
                values[name] = int(field.parts[field.data[0]][0])
        for tensor in reader.tensors:
            base = tensor.name.split(".", 2)[-1]
            if base in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
                types.add(("gate_up", tensor.tensor_type.name))
                dims.setdefault(base, [int(x) for x in tensor.shape])
            elif base == "ffn_down_exps.weight":
                types.add(("down", tensor.tensor_type.name))
                dims.setdefault(base, [int(x) for x in tensor.shape])
            elif base == "ffn_gate_shexp.weight":
                dims.setdefault(base, [int(x) for x in tensor.shape])
    count = values.get("expert_count", values.get("n_routed_experts"))
    used = values.get("expert_used_count", values.get("num_experts_per_tok"))
    if not count or not used or "ffn_gate_exps.weight" not in dims:
        raise ValueError(f"{lane}: GGUF carries no MoE expert count/used/expert tensors")
    gate = dims["ffn_gate_exps.weight"]
    shexp = dims.get("ffn_gate_shexp.weight", [gate[0], gate[1]])
    known = LANE_PROFILES.get(lane)
    return MoeProfile(lane, str(paths[0]), expert_count=count, expert_used=used,
                      hidden=gate[0], expert_ff=gate[1], shexp_ff=shexp[1],
                      gate_up_types=tuple(t for t in WITNESS_TYPES if ("gate_up", t) in types),
                      down_types=tuple(t for t in WITNESS_TYPES if ("down", t) in types),
                      extra_types=known.extra_types if known else ())


def tightened_nmse_bound(anchor_nmse: float, *, factor: float = 3.0,
                         anchor_relative: bool = False) -> float:
    """anchor NMSE * `factor`, capped at `SERVED_SHAPE_NMSE_CAP`, floored at
    `SERVED_SHAPE_NMSE_FLOOR`. Raises on a negative/non-finite input: a bound must
    never be derived from a measurement that cannot itself be trusted."""
    if not isinstance(anchor_nmse, (int, float)) or isinstance(anchor_nmse, bool) \
            or anchor_nmse != anchor_nmse or anchor_nmse in (float("inf"), float("-inf")) \
            or anchor_nmse < 0:
        raise ValueError(f"anchor_nmse must be a finite non-negative number, got {anchor_nmse!r}")
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor <= 0:
        raise ValueError(f"factor must be a positive number, got {factor!r}")
    if anchor_nmse >= SERVED_SHAPE_NMSE_CAP:
        # The anchor itself fails the generic bound on this case. Clipping would set a
        # bound BELOW the anchor's own error (refusing an unchanged correct kernel), so
        # by default this refuses. `anchor_relative=True` (an explicit, recorded
        # operator choice) instead holds the candidate to "no worse than factor x the
        # anchor" on this case -- looser than the generic bound, but never unchecked.
        if not anchor_relative:
            raise ValueError(f"anchor_nmse {anchor_nmse!r} is at or above the generic "
                             f"bound {SERVED_SHAPE_NMSE_CAP}: the anchor itself fails "
                             "the generic check here -- investigate, do not clip")
        return anchor_nmse * factor
    if anchor_nmse < SERVED_SHAPE_NMSE_LEGACY_CAP:
        # Every case the old policy accepted keeps its OLD bound exactly (no loosening).
        return min(SERVED_SHAPE_NMSE_LEGACY_CAP,
                   max(SERVED_SHAPE_NMSE_FLOOR, anchor_nmse * factor))
    # 1e-4 <= anchor < 5e-4: the old policy refused; now factor x anchor, never looser
    # than the generic bound.
    return min(SERVED_SHAPE_NMSE_CAP, anchor_nmse * factor)


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


def anchor_exceeds_generic(anchor_nmse_by_shape: Mapping[tuple[str, str, int], float]
                           ) -> frozenset:
    """The measured (shape, type, width) keys whose ANCHOR NMSE is at/above the generic
    bound -- the cases `--anchor-exceeds-generic anchor-relative` holds anchor-relative."""
    return frozenset(key for key, value in anchor_nmse_by_shape.items()
                     if value >= SERVED_SHAPE_NMSE_CAP)


def corpus(routed: bool = False, lane: "str | None" = None):
    """(shapes, types, widths) of the candidate corpus or a lane's bundle-tier routed one."""
    if routed:
        if lane not in LANE_PROFILES:
            raise ValueError(f"the routed corpus needs a known lane, got {lane!r}")
        return routed_shapes(lane), routed_types(lane), ROUTED_WIDTHS
    if lane is None:   # unpartitioned (unit tests / tooling only); gates always pass a lane
        return SERVED_SHAPES, WITNESS_TYPES, SERVED_WIDTHS
    if lane not in LANE_PROFILES:
        raise ValueError(f"unknown lane {lane!r}")
    return lane_served_shapes(lane), WITNESS_TYPES, SERVED_WIDTHS


def case_set(anchor_nmse_by_shape: Mapping[tuple[str, str, int], float], *,
            factor: float = 3.0, routed: bool = False,
            lane: "str | None" = None,
            anchor_relative_keys=frozenset()) -> tuple[ServedShapeCase, ...]:
    """Every `(ServedShape, type, width)` case, bound from `anchor_nmse_by_shape`
    keyed `(shape.name, type_a, n)` -- the independent reference run's measured NMSE on
    the anchor build at that exact shape, type AND width (re-review 2026-10-06: one
    calibration reused across widths 1-5 hid width-specific kernels). Raises
    `KeyError` naming the missing triple -- a served-shape case this caller cannot
    justify a bound for is never silently dropped OR silently given a guessed one."""
    cases = []
    for shape, type_a, width in canonical_triples(routed, lane):
        key = (shape.name, type_a, width)
        if key not in anchor_nmse_by_shape:
            raise KeyError(f"no anchor NMSE measurement for {key}; refusing to "
                           "bake a served-shape case with a guessed bound")
        bound = tightened_nmse_bound(anchor_nmse_by_shape[key], factor=factor,
                                     anchor_relative=key in anchor_relative_keys)
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
// Round-15: deterministic per-case inputs. Every served / routed / calibration case
// seeds from AK_SERVED_SHAPE_SEED and the FNV-1a 64 hash of its bound-free case key, so
// the calibration run and every candidate see byte-identical inputs -- list position
// and --suite-seed play no part. The MUL_MAT_ID ids draw from the same seeded stream.
// Round-17: the seed scheme is COMPILED IN -- the literal below is what
// `binary_has_seed_scheme` looks for (a test-backend-ops built before seeding draws
// random inputs, and the gate / calibration refuse it) -- and announced once per run
// on stderr, which calibration also requires at run time.
static void autokernel_seed_case(const std::string & key) {
    static bool announced = false;
    if (!announced) {
        fprintf(stderr, "%s\\n", "AK_SEED_SCHEME=ak-served-shape-fnv1a64-casekey-v1");
        announced = true;
    }
    uint64_t h = 0x14650FB0739D0383ULL;
    for (unsigned char c : key) { h ^= c; h *= 0x100000001B3ULL; }
    suite_seed_begin(0x414b53455256ULL, (size_t) h, "AK_SERVED_SHAPE");
}
// Round-17 debug mode: AUTOKERNEL_INPUT_HASH=1 prints
// `AK_INPUT_HASH\\t<case key>\\t<fnv1a64 of every leaf tensor's bytes>` after the inputs
// are initialised, so a calibration run and a gate run (any case order) can be compared
// case by case -- `served_shape_cases.input_hash_mismatches`.
static void autokernel_report_inputs(ggml_context * ctx, const std::string & key) {
    const char * v = std::getenv("AUTOKERNEL_INPUT_HASH");
    if (v == nullptr || std::strcmp(v, "1") != 0) {
        return;
    }
    uint64_t h = 0x14650FB0739D0383ULL;
    std::vector<uint8_t> buf;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->buffer == NULL || t->op != GGML_OP_NONE) { continue; }
        const size_t nbytes = ggml_nbytes(t);
        for (size_t off = 0; off < nbytes; off += ((size_t) 1 << 20)) {
            const size_t len = std::min(((size_t) 1 << 20), nbytes - off);
            buf.resize(len);
            ggml_backend_tensor_get(t, buf.data(), off, len);
            for (size_t i = 0; i < len; i++) { h ^= buf[i]; h *= 0x100000001B3ULL; }
        }
    }
    fprintf(stdout, "AK_INPUT_HASH\\t%s\\t%016llx\\n", key.c_str(), (unsigned long long) h);
    fflush(stdout);
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
// Round-12/13: deterministic, dispersed, high expert ids for the routed cases: row r
// routes to n_mats - 1 - (29 r + step j) mod n_mats, step = (n_mats / n_used) | 1, for
// j < n_used; the rest of the row keeps a full permutation (well-formed ids tensor).
static void autokernel_route_ids(ggml_context * ctx, int n_mats, int n_used) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != GGML_TYPE_I32 || ggml_is_view_op(t->op)) { continue; }
        for (int64_t r = 0; r < ggml_nrows(t); r++) {
            std::vector<int32_t> data(t->ne[0]);
            std::vector<char> used(n_mats, 0);
            int pos = 0;
            const int64_t step = (n_mats / n_used) | 1;
            for (int j = 0; j < n_used && pos < t->ne[0]; j++) {
                const int id = n_mats - 1 - (int) ((r * 29 + j * step) % n_mats);
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
    void initialize_tensors(ggml_context * ctx) override {
        autokernel_seed_case(test_mul_mat::vars());
        test_mul_mat::initialize_tensors(ctx);
        autokernel_report_inputs(ctx, test_mul_mat::vars());
    }
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
    void initialize_tensors(ggml_context * ctx) override {
        autokernel_seed_case(test_mul_mat_id::vars());
        test_mul_mat_id::initialize_tensors(ctx);
        autokernel_report_inputs(ctx, test_mul_mat_id::vars());
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
        autokernel_seed_case(test_mul_mat_id::vars() + ",routed=1");
        test_mul_mat_id::initialize_tensors(ctx);
        autokernel_route_ids(ctx, n_mats, n_used);
        autokernel_report_inputs(ctx, test_mul_mat_id::vars() + ",routed=1");
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
    "}\n") + """// Round-15: deterministic per-case inputs. Every served / routed / calibration case
// seeds from AK_SERVED_SHAPE_SEED and the FNV-1a 64 hash of its bound-free case key, so
// the calibration run and every candidate see byte-identical inputs -- list position
// and --suite-seed play no part. The MUL_MAT_ID ids draw from the same seeded stream.
// Round-17: the seed scheme is COMPILED IN -- the literal below is what
// `binary_has_seed_scheme` looks for (a test-backend-ops built before seeding draws
// random inputs, and the gate / calibration refuse it) -- and announced once per run
// on stderr, which calibration also requires at run time.
static void autokernel_seed_case(const std::string & key) {
    static bool announced = false;
    if (!announced) {
        fprintf(stderr, "%s\\n", "AK_SEED_SCHEME=ak-served-shape-fnv1a64-casekey-v1");
        announced = true;
    }
    uint64_t h = 0x14650FB0739D0383ULL;
    for (unsigned char c : key) { h ^= c; h *= 0x100000001B3ULL; }
    suite_seed_begin(0x414b53455256ULL, (size_t) h, "AK_SERVED_SHAPE");
}
// Round-17 debug mode: AUTOKERNEL_INPUT_HASH=1 prints
// `AK_INPUT_HASH\\t<case key>\\t<fnv1a64 of every leaf tensor's bytes>` after the inputs
// are initialised, so a calibration run and a gate run (any case order) can be compared
// case by case -- `served_shape_cases.input_hash_mismatches`.
static void autokernel_report_inputs(ggml_context * ctx, const std::string & key) {
    const char * v = std::getenv("AUTOKERNEL_INPUT_HASH");
    if (v == nullptr || std::strcmp(v, "1") != 0) {
        return;
    }
    uint64_t h = 0x14650FB0739D0383ULL;
    std::vector<uint8_t> buf;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->buffer == NULL || t->op != GGML_OP_NONE) { continue; }
        const size_t nbytes = ggml_nbytes(t);
        for (size_t off = 0; off < nbytes; off += ((size_t) 1 << 20)) {
            const size_t len = std::min(((size_t) 1 << 20), nbytes - off);
            buf.resize(len);
            ggml_backend_tensor_get(t, buf.data(), off, len);
            for (size_t i = 0; i < len; i++) { h ^= buf[i]; h *= 0x100000001B3ULL; }
        }
    }
    fprintf(stdout, "AK_INPUT_HASH\\t%s\\t%016llx\\n", key.c_str(), (unsigned long long) h);
    fflush(stdout);
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
// Round-12/13: deterministic, dispersed, high expert ids for the routed cases: row r
// routes to n_mats - 1 - (29 r + step j) mod n_mats, step = (n_mats / n_used) | 1, for
// j < n_used; the rest of the row keeps a full permutation (well-formed ids tensor).
static void autokernel_route_ids(ggml_context * ctx, int n_mats, int n_used) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != GGML_TYPE_I32 || ggml_is_view_op(t->op)) { continue; }
        for (int64_t r = 0; r < ggml_nrows(t); r++) {
            std::vector<int32_t> data(t->ne[0]);
            std::vector<char> used(n_mats, 0);
            int pos = 0;
            const int64_t step = (n_mats / n_used) | 1;
            for (int j = 0; j < n_used && pos < t->ne[0]; j++) {
                const int id = n_mats - 1 - (int) ((r * 29 + j * step) % n_mats);
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
    void initialize_tensors(ggml_context * ctx) override {
        autokernel_seed_case(test_mul_mat::vars());
        test_mul_mat::initialize_tensors(ctx);
        autokernel_report_inputs(ctx, test_mul_mat::vars());
    }
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
    void initialize_tensors(ggml_context * ctx) override {
        autokernel_seed_case(test_mul_mat_id::vars());
        test_mul_mat_id::initialize_tensors(ctx);
        autokernel_report_inputs(ctx, test_mul_mat_id::vars());
    }
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
        autokernel_seed_case(test_mul_mat_id::vars() + ",routed=1");
        test_mul_mat_id::initialize_tensors(ctx);
        autokernel_route_ids(ctx, n_mats, n_used);
        autokernel_report_inputs(ctx, test_mul_mat_id::vars() + ",routed=1");
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


def canonical_triples(routed: bool = False,
                      lane: "str | None" = None) -> tuple[tuple[ServedShape, str, int], ...]:
    """Every VALID (shape, type, width): the type fits the shape's k (block size) and,
    for a routed shape, is one its direction actually serves."""
    shapes, types, widths = corpus(routed, lane)
    return tuple((shape, type_a, width) for shape in shapes
                 for type_a in (routed_types(lane, shape) if routed else types)
                 if type_fits(shape, type_a) for width in widths)


def calibration_triples(lane: str) -> tuple[tuple[ServedShape, str, int], ...]:
    """Both of `lane`'s corpora: its partitioned candidate corpus and its routed one."""
    return canonical_triples(lane=lane) + canonical_triples(routed=True, lane=lane)


def shard_sequence(items, n_shards: int) -> tuple:
    """Split `items` into `n_shards` disjoint, stably-ordered shards whose union is
    `items` with every item exactly once (round-robin assignment by position, so no
    shard's case set depends on another shard's size). `n_shards` is clamped to
    `[1, len(items)]`: fewer, non-empty shards are produced when there are not enough
    items to fill every requested shard. Raises ValueError for `n_shards < 1`."""
    items = tuple(items)
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if not items:
        return ()
    n = max(1, min(n_shards, len(items)))
    buckets = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return tuple(tuple(bucket) for bucket in buckets)


def calibration_regex_for(triples) -> str:
    """`calibration_regex`'s body, generalized to an arbitrary (shard's) subset of
    calibration triples instead of a whole lane's corpus."""
    import re
    return "^(" + "|".join(re.escape(calibration_vars(*t)) for t in triples) + ")$"


def calibration_regex(lane: str) -> str:
    return calibration_regex_for(calibration_triples(lane))


def calibration_patch_block(lane: str) -> str:
    lines = []
    for shape, type_a, n in calibration_triples(lane):
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


def parse_calibration(output: str, lane: str, *, triples=None) -> dict:
    """{(shape name, type, width): NMSE} from calibration output -- the max over any
    repeated print of one case. `triples` restricts the EXPECTED set to a shard's own
    assignment (default: the lane's whole corpus, `calibration_triples(lane)`). Raises
    ValueError unless every expected triple has a finite, non-negative value and no
    case outside the expected set was printed -- a shard whose `-p` filter leaked
    another shard's case is a harness fault, not a silent extra measurement."""
    import math
    expected = triples if triples is not None else calibration_triples(lane)
    by_vars = {calibration_vars(*t): (t[0].name, t[1], t[2]) for t in expected}
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


def binary_has_seed_scheme(build_dir: Path) -> bool:
    """Round-17: True when the build's test-backend-ops carries the CURRENT seed-scheme
    marker -- i.e. was built from a generated block that seeds every case from its key."""
    binary = Path(build_dir) / "bin" / "test-backend-ops"
    try:
        data = binary.read_bytes()
    except OSError:
        return False
    return SEED_MARKER.encode() in data


def parse_input_hashes(output: str) -> dict:
    """`{case key: input hash}` from an AUTOKERNEL_INPUT_HASH=1 run; a key printed twice
    with different hashes is itself a nondeterminism finding (ValueError)."""
    hashes: dict = {}
    for line in output.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3 or parts[0] != INPUT_HASH_MARKER:
            continue
        if hashes.setdefault(parts[1], parts[2]) != parts[2]:
            raise ValueError(f"case {parts[1]!r} printed two different input hashes")
    return hashes


def input_hash_mismatches(first: dict, second: dict) -> list:
    """Case keys whose inputs differ between two runs (or that only one run measured).
    Order-free: the comparison is by case key, never by position."""
    return sorted(key for key in set(first) | set(second)
                  if first.get(key) != second.get(key))


def binary_has_calibration(build_dir: Path) -> bool:
    binary = Path(build_dir) / "bin" / "test-backend-ops"
    try:
        data = binary.read_bytes()
    except OSError:
        return False
    return (CALIBRATION_CASE_SET_ID.encode() in data and CALIBRATION_MARKER.encode() in data
            and BACKEND_THREADS_ENV.encode() in data and SEED_MARKER.encode() in data)


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
    return (CASE_SET_ID.encode() in data and BACKEND_THREADS_ENV.encode() in data
            and SEED_MARKER.encode() in data)


def binary_has_routed_case_set(build_dir: Path) -> bool:
    binary = Path(build_dir) / "bin" / "test-backend-ops"
    try:
        data = binary.read_bytes()
    except OSError:
        return False
    return (ROUTED_CASE_SET_ID.encode() in data and BACKEND_THREADS_ENV.encode() in data
            and SEED_MARKER.encode() in data)


_SHAPES_BY_NAME: dict[str, ServedShape] = {shape.name: shape for shape in SERVED_SHAPES}


def write_manifest(path: Path, cases: tuple[ServedShapeCase, ...], *,
                   routed: bool = False, lane: "str | None" = None,
                   anchor_relative_keys=frozenset()) -> None:
    """Persist the EXACT baked case set (with its anchor-derived `max_nmse` bounds)
    beside the applied llama-tree patch, so a later gate run reads what was actually
    compiled in rather than recomputing (and potentially drifting from) it."""
    body = {"schema": ROUTED_MANIFEST_SCHEMA if routed else MANIFEST_SCHEMA,
            "case_set_id": ROUTED_CASE_SET_ID if routed else CASE_SET_ID,
           "cases": [{"shape_name": c.shape.name, "type_a": c.type_a, "n": c.n,
                      "max_nmse": c.max_nmse} for c in cases]}
    if routed and lane not in LANE_PROFILES:
        raise ValueError(f"a routed manifest needs a known lane, got {lane!r}")
    body["lane"] = lane
    body["seed_scheme"] = SEED_SCHEME
    body["partition"] = [shape.name for shape in corpus(routed, lane)[0]]
    own = {(c.shape.name, c.type_a, c.n) for c in cases}
    body["anchor_relative"] = sorted(list(k) for k in anchor_relative_keys if tuple(k) in own)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class ManifestRefused(ValueError):
    """A served-shape manifest is absent, malformed, or names an unknown shape."""


def load_manifest(path: Path, *, routed: bool = False,
                  lane: "str | None" = None) -> tuple[ServedShapeCase, ...]:
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
    if not isinstance(body, dict) or body.get("schema") != schema \
            or body.get("case_set_id") != set_id or not isinstance(body.get("cases"), list) \
            or not body["cases"]:
        raise ManifestRefused(f"{path}: served-shape manifest has the wrong shape or schema")
    recorded = body.get("lane") if isinstance(body, dict) else None
    if (routed and recorded not in LANE_PROFILES) or (lane is not None and recorded != lane) \
            or (recorded is not None and recorded not in LANE_PROFILES):
        raise ManifestRefused(f"{path}: manifest is for lane {recorded!r}, not {lane!r}")
    lane = recorded
    if not isinstance(body, dict) or body.get("seed_scheme") != SEED_SCHEME:
        raise ManifestRefused(f"{path}: manifest input-seed scheme "
                              f"{body.get('seed_scheme') if isinstance(body, dict) else None!r}"
                              f" is not {SEED_SCHEME!r} (re-calibrate)")
    try:
        partition = [shape.name for shape in corpus(routed, lane)[0]]
    except ValueError as exc:
        raise ManifestRefused(f"{path}: {exc}") from exc
    if body.get("partition") != partition:
        raise ManifestRefused(f"{path}: manifest partition does not match lane {lane!r}")
    shapes, types, widths = corpus(routed, lane)
    by_name = {shape.name: shape for shape in shapes}
    if not isinstance(body, dict) or body.get("schema") != schema \
            or body.get("case_set_id") != set_id or not isinstance(body.get("cases"), list) \
            or not body["cases"]:
        raise ManifestRefused(f"{path}: served-shape manifest has the wrong shape or schema")
    recorded = body.get("anchor_relative", [])
    if not isinstance(recorded, list) or not all(
            isinstance(k, list) and len(k) == 3 for k in recorded):
        raise ManifestRefused(f"{path}: malformed anchor_relative list")
    anchor_relative = {tuple(k) for k in recorded}
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
        key = (row["shape_name"], row["type_a"], row["n"])
        # Above the generic bound only for a case the calibration RECORDED as
        # anchor-relative (the anchor itself measured >= the generic bound there).
        limit = (SERVED_SHAPE_NMSE_CAP * 1000 if key in anchor_relative
                 else SERVED_SHAPE_NMSE_CAP)
        if not isinstance(max_nmse, (int, float)) or isinstance(max_nmse, bool) \
                or not (0 < max_nmse <= limit) \
                or (key in anchor_relative and max_nmse <= SERVED_SHAPE_NMSE_CAP):
            raise ManifestRefused(f"{path}: max_nmse={max_nmse!r} is out of bounds "
                                  f"(0, {SERVED_SHAPE_NMSE_CAP}] for {key}")
        cases.append(ServedShapeCase(shape, row["type_a"], row["n"], float(max_nmse)))
    # Re-review 2026-10-06: exact canonical SET equality, duplicates refused -- a row
    # count alone accepted 180 copies of one case.
    keys = [(c.shape.name, c.type_a, c.n) for c in cases]
    canonical = {(t[0].name, t[1], t[2]) for t in canonical_triples(routed, lane)}
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
           "binary_has_calibration", "ROUTED_CASE_SET_ID", "routed_shapes",
           "routed_types", "ROUTED_WIDTHS", "LANE_PROFILES", "MoeProfile", "route_ids",
           "candidate_experts", "lane_for_model", "moe_profile_from_gguf", "BLOCK_SIZE",
           "type_fits", "anchor_exceeds_generic", "SEED_SCHEME", "AK_SERVED_SHAPE_SEED",
           "case_key", "case_seed_index", "lane_served_shapes", "BACKEND_THREADS_ENV", "corpus",
           "calibration_triples", "binary_has_routed_case_set", "THREADS_PATCHED",
           "SEED_MARKER", "binary_has_seed_scheme", "INPUT_HASH_ENV", "INPUT_HASH_MARKER",
           "parse_input_hashes", "input_hash_mismatches", "shard_sequence",
           "calibration_regex_for"]
