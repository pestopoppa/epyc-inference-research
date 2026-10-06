"""Per-node CPU roofline + admitted-route coverage: the loop's own autonomy self-check.

WHY THIS EXISTS
---------------
A lane ran for hours on 2026-10-06 making no progress, because its biggest measured
roofline gaps -- Q38FN IQ3_S/IQ4_NL MoE kernels running at 10-40% of the host's
~394-470 GB/s memory-bandwidth ceiling, in `iqk_gemm_iquants.cpp`,
`iqk_gemm_legacy_quants.cpp` and `repack.cpp` -- had NO admitted route in `gates.py`
(the planner can write a hypothesis, the actor can author a patch, and `gates.py`
refuses it at review with "no admitted route" every single time). Nobody noticed for
hours because the planner was reading PATH-LEVEL bandwidth figures (`paths[*]
.GBs_on_wall` in the raw `ggml-cpu-prof/1` dump) that pool every node of every shape
that hits a path across the whole replay, prefill included -- so the number the
planner saw was never the number the gap sat at.

This module is the fix: it reads the dump's PER-NODE rows (never the `paths`
aggregate), computes an achieved-bandwidth-vs-ceiling roofline per node, resolves
which source file/symbol a gap node's op+type maps to, and checks that resolution
against the SAME admitted-route tables `gates.py` uses at patch-review time -- so
"the planner's biggest target" and "a target `gates.py` can ever admit a patch for"
are the same question asked the same way, instead of two views that can silently
diverge for an entire run.

It is detection only. Widening an admitted route is an explicit human/operator
decision (see CLAUDE.md "Experimental Kernel Workflow"); this module never proposes,
authors or applies one -- it names the files and symbols a human would need to widen.

SCOPE
-----
Per-node bytes are computed from `(src0_type, src0_ne)` using the ggml block-quant
layouts transcribed from `ggml/src/ggml-common.h` (the FROZEN production tree is read
but never built or linked against). A node whose type is not in
`GGML_TYPE_BLOCK_BYTES`, or whose dump row lacks `wall_us`/`evals`, produces NO
bandwidth reading -- never a zero that would silently read as "no gap here".

Route resolution is a small, explicit `(op, src0_type) -> (file, symbol)` table
transcribed from the 2026-10-06 lowbit-scope audit
(`/mnt/raid0/llm/tmp/lowbit-lanes-20261006/scope/README.md`). An (op, type) pair this
table does not name resolves to `None` ("unresolved"), which `uncovered_gap_summary`
reports distinctly from `covered=False` ("resolved, but no admitted route").
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from . import gates

SCHEMA = "epyc.autokernel.loop_roofline_coverage.v1"
SCOPE_GAP_SCHEMA = "epyc.autokernel.loop_roofline_scope_gap.v1"
SCOPE_GAP_FILENAME = "scope_gap.json"

#: (block_size, bytes_per_block), transcribed from the `block_<type>` struct layouts
#: in `ggml/src/ggml-common.h` (QK_K=256, QK4_0=QK8_0=QK4_NL=32). This loop never
#: builds or links ggml, so these are read off the source, not computed at runtime --
#: a type missing here is "unresolved", never silently "free" (0 bytes).
GGML_TYPE_BLOCK_BYTES: dict[str, tuple[int, int]] = {
    "f32": (1, 4), "f16": (1, 2),
    "q4_0": (32, 18), "q5_0": (32, 22), "q8_0": (32, 34),
    "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210), "q8_k": (256, 292),
    "iq3_s": (256, 110), "iq4_nl": (32, 18), "iq4_xs": (256, 136),
}

#: Observed on this host 2026-10-06 (scope/README.md roofline table): the production
#: decode floor sat at 394-470 GB/s depending on shape. This is a DEFAULT, not a
#: measured constant -- a caller with a fresh STREAM-triad/roofline calibration for
#: the live host should pass its own `ceiling_gb_s` instead of relying on it.
DEFAULT_HOST_CEILING_GB_S = 430.0

#: A gap node's share of the top-K gap total that must sit in UNCOVERED nodes before
#: this is worth a loud warning rather than a quiet table entry.
UNCOVERED_GAP_WARN_THRESHOLD = 0.30

#: Stagnation hook thresholds (see `is_stagnant`): either reading alone is enough.
STAGNATION_MIN_MEASURED_POINTS = 5
STAGNATION_NULL_STREAK = 5


#: (op marker, src0_type) -> (source file, representative symbol) the admitted-route
#: tables in `gates.py` key on. Transcribed from the 2026-10-06 lowbit-scope audit.
#: Extend this as new quant types/ops get profiled -- an (op, type) pair this table
#: does not name resolves to `None` ("unresolved"), never a false "covered".
ROUTE_FILE_SYMBOL: dict[tuple[str, str], tuple[str, str]] = {
    ("MUL_MAT", "iq3_s"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp", "DequantizerIQ3S"),
    ("MUL_MAT_ID", "iq3_s"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp", "DequantizerIQ3S"),
    ("MUL_MAT", "iq4_nl"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp", "IQ4_NL_UnpackerU"),
    ("MUL_MAT_ID", "iq4_nl"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp", "IQ4_NL_UnpackerU"),
    # Ad-hoc admitted (gates.py ~1725-1736, `_iqk_q45_dot_scope_refusal`): the DENSE
    # (MUL_MAT) Q4_K/Q5_K dot route's dequantizer bodies.
    ("MUL_MAT", "q4_k"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "DequantizerQ4K_AVX2"),
    ("MUL_MAT", "q5_k"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "DequantizerQ5K_AVX2"),
    # 2026-10-06 seed-6 widening (`iqk_kquants_q6_iq4xs_dequant`, gates.py ~635-653):
    # the DENSE (MUL_MAT) Q6_K/IQ4_XS dot route's dequantizer bodies.
    ("MUL_MAT", "q6_k"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "DequantizerQ6K_AVX2"),
    ("MUL_MAT", "iq4_xs"): (
        "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "DequantizerIQ4XS"),
    # The ad-hoc/table-driven dense admissions above are DENSE-only; the MoE
    # (MUL_MAT_ID) path for every K-quant/IQ4_XS expert matmul falls through to
    # `repack.cpp` (scope/README.md 2026-10-06). `cpu_repack_mmid` DOES name that
    # (file, symbol) pair structurally (gates.py ~616-628), but is refused at
    # ppl_contract layer (a) for every patch unconditionally
    # (`gates.PPL_CONTRACT_NO_OP_ORACLE`, gates.py ~2436) -- `route_is_admitted`
    # treats that as uncovered, not "named".
    ("MUL_MAT_ID", "q4_k"): (
        "ggml/src/ggml-cpu/repack.cpp", "forward_mul_mat_id"),
    ("MUL_MAT_ID", "q5_k"): (
        "ggml/src/ggml-cpu/repack.cpp", "forward_mul_mat_id"),
    ("MUL_MAT_ID", "q6_k"): (
        "ggml/src/ggml-cpu/repack.cpp", "forward_mul_mat_id"),
    ("MUL_MAT_ID", "iq4_xs"): (
        "ggml/src/ggml-cpu/repack.cpp", "forward_mul_mat_id"),
}

#: (path, symbol) pairs `gates.affected_op_scope` admits through INLINE structural
#: rules that never populate `CPU_SOURCE_ROUTES`/`CPU_MULTI_FILE_ROUTES` -- so
#: `gates.cpu_source_routes`/`cpu_multi_file_routes` alone under-report them as
#: uncovered. Transcribed 2026-10-06 from `affected_op_scope` (gates.py ~1714-1736):
#: the MoE-rows/fused-up-gate dispatch on `iqk_mul_mat.cpp`, and the dense Q4_K/Q5_K
#: dot route on `iqk_gemm_kquants.cpp`. Naming a pair here says the PAIR is
#: structurally admitted, not that every patch against it passes -- each is still
#: gated further by hunk confinement at author time
#: (`_iqk_moe_rows_hunks_confined`, `_iqk_q45_dot_scope_refusal`). Re-check this set
#: against `affected_op_scope` whenever gates.py's ad-hoc rules change: there is no
#: single callable that reports structural (patch-independent) admission for these
#: the way `cpu_source_routes` does for the table-driven routes.
AD_HOC_ADMITTED_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", "iqk_mul_mat_moe_rows"),
    ("ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", "iqk_moe_fused_up_gate"),
    ("ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "mul_mat_qX_K_q8_2_X4_T"),
    ("ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "DequantizerQ4K_AVX2"),
    ("ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp", "DequantizerQ5K_AVX2"),
})


class RooflineRefused(ValueError):
    """Input to a roofline/coverage computation does not carry what it needs."""


def tensor_bytes(type_name: Any, ne: Sequence[Any]) -> int | None:
    """Bytes a tensor of this ggml type and shape occupies; `None` if unresolved."""
    sizing = GGML_TYPE_BLOCK_BYTES.get(str(type_name).lower())
    if sizing is None:
        return None
    block, bytes_per_block = sizing
    elements = 1
    for dim in ne:
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            return None
        if dim < 0:
            return None
        elements *= dim
    if elements == 0:
        return 0
    blocks = -(-elements // block)  # ceil division
    return blocks * bytes_per_block


def node_achieved_bandwidth(node: Mapping[str, Any]) -> dict[str, float] | None:
    """Achieved GB/s for one raw per-node row, or `None` if it cannot be computed.

    `node` is one row of the `nodes` array in a `ggml-cpu-prof/1` dump (per-node, NOT
    the `paths`/`weight_paths` aggregate -- see the module docstring). Bytes moved are
    src0 (the weight/activation this op reads once per call) plus dst (written once,
    always f32 in this instrument); wall is per accumulated graph eval, so dividing by
    `evals` gives wall PER APPEARANCE, matching the bytes-per-appearance numerator.
    """
    evals = node.get("evals")
    wall_us = node.get("wall_us")
    if not isinstance(evals, (int, float)) or isinstance(evals, bool) or evals <= 0:
        return None
    if not isinstance(wall_us, (int, float)) or isinstance(wall_us, bool) or wall_us <= 0:
        return None
    src0_bytes = tensor_bytes(node.get("src0_type"), node.get("src0_ne") or ())
    if src0_bytes is None:
        return None
    dst_bytes = tensor_bytes("f32", node.get("dst_ne") or ()) or 0
    total_bytes = src0_bytes + dst_bytes
    wall_s_per_eval = (float(wall_us) / float(evals)) / 1e6
    if wall_s_per_eval <= 0:
        return None
    return {"bytes_per_eval": float(total_bytes), "wall_s_per_eval": wall_s_per_eval,
            "achieved_gb_s": (total_bytes / wall_s_per_eval) / 1e9}


def per_node_roofline(nodes: Sequence[Mapping[str, Any]], *, total_wall_us: float,
                      ceiling_gb_s: float = DEFAULT_HOST_CEILING_GB_S,
                      limit: int = 12) -> list[dict[str, Any]]:
    """Rank per-node gap share: time share x (1 - achieved/ceiling), clamped at 0.

    Never reads `paths`/`weight_path_shares` -- callers must pass the `nodes` rows
    only. Rows this module cannot compute a bandwidth reading for are skipped, not
    zeroed, so a silent "no gap" never masks an unresolved type.
    """
    if not isinstance(total_wall_us, (int, float)) or total_wall_us <= 0:
        raise RooflineRefused("total_wall_us must be positive")
    if not isinstance(ceiling_gb_s, (int, float)) or ceiling_gb_s <= 0:
        raise RooflineRefused("ceiling_gb_s must be positive")
    rows: list[dict[str, Any]] = []
    for node in nodes:
        bandwidth = node_achieved_bandwidth(node)
        if bandwidth is None:
            continue
        wall_us = float(node["wall_us"])
        time_share = wall_us / float(total_wall_us)
        achieved = bandwidth["achieved_gb_s"]
        headroom_fraction = max(0.0, 1.0 - min(achieved, ceiling_gb_s) / ceiling_gb_s)
        rows.append({
            "idx": node.get("idx"), "op": node.get("op"), "name": node.get("name"),
            "src0_type": node.get("src0_type"),
            "time_share": time_share, "achieved_gb_s": achieved,
            "ceiling_gb_s": float(ceiling_gb_s),
            "headroom_fraction": headroom_fraction,
            "gap_share": time_share * headroom_fraction,
        })
    rows.sort(key=lambda row: (-row["gap_share"], str(row.get("name"))))
    return rows[:max(0, int(limit))]


def resolve_route_file_symbol(op: Any, src0_type: Any) -> tuple[str, str] | None:
    """`(file, symbol)` the gate set would need to admit a route for, or `None`."""
    return ROUTE_FILE_SYMBOL.get((str(op).upper(), str(src0_type).lower()))


def route_is_admitted(file_path: str, symbol: str) -> bool:
    """True only when `gates.py`'s OWN admitted-route tables name this (file, symbol).

    Reuses `gates.cpu_source_routes`/`gates.cpu_multi_file_routes` directly -- the
    same lookup the actor's patch-admission path runs -- so this can never disagree
    with `gates.py` about what is covered today. It answers NAMED, not
    ADMITS-THIS-PATCH: a route naming the pair can still refuse a specific patch on
    hunk-confinement grounds; that gate runs at author time, not here.

    A route whose `route` name sits in `gates.PPL_CONTRACT_NO_OP_ORACLE` is named in
    the structural table but is refused at ppl_contract layer (a) for EVERY patch,
    unconditionally (no op oracle on this host can ever reach it -- gates.py
    ~2423-2425) -- not a hunk-confinement refusal that depends on the patch's shape.
    That makes it a standing, not patch-dependent, refusal, so this reports it
    uncovered rather than "named, patch TBD" (`cpu_repack_mmid`/`repack.cpp`, 2026-10-06).
    """
    if (file_path, symbol) in AD_HOC_ADMITTED_PAIRS:
        return True
    routes = gates.cpu_source_routes(file_path, symbol)
    if routes and any(route.route not in gates.PPL_CONTRACT_NO_OP_ORACLE
                      for route in routes):
        return True
    multi = gates.cpu_multi_file_routes(symbol)
    if multi:
        for route in multi:
            if file_path in {component.path for component in route.files}:
                return True
    return False


def route_coverage_map(gap_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each gap row plus its resolved (file, symbol) and admitted-route coverage."""
    out = []
    for row in gap_rows:
        resolved = resolve_route_file_symbol(row.get("op"), row.get("src0_type"))
        if resolved is None:
            out.append({**row, "file": None, "symbol": None, "covered": None,
                       "coverage_basis": "unresolved: no (op, src0_type) route map entry"})
            continue
        file_path, symbol = resolved
        out.append({**row, "file": file_path, "symbol": symbol,
                   "covered": route_is_admitted(file_path, symbol),
                   "coverage_basis": "gates.cpu_source_routes/cpu_multi_file_routes"})
    return out


def uncovered_gap_summary(covered_rows: Sequence[Mapping[str, Any]], *,
                          threshold: float = UNCOVERED_GAP_WARN_THRESHOLD
                          ) -> dict[str, Any]:
    """Roll the per-row coverage up into one triggered/not-triggered verdict.

    A row with `covered is None` (unresolved) counts as uncovered: "we don't even
    know which route would cover this" is never safer than "no route covers this".
    """
    total_gap_share = sum(float(row.get("gap_share", 0.0)) for row in covered_rows)
    uncovered_rows = [row for row in covered_rows if row.get("covered") is not True]
    uncovered_gap_share = sum(float(row.get("gap_share", 0.0)) for row in uncovered_rows)
    fraction = (uncovered_gap_share / total_gap_share) if total_gap_share > 0 else 0.0
    triggered = uncovered_gap_share > 0 and fraction >= threshold
    return {
        "schema": SCHEMA, "threshold": threshold,
        "total_gap_share": total_gap_share,
        "uncovered_gap_share": uncovered_gap_share,
        "uncovered_fraction": fraction,
        "triggered": triggered,
        "uncovered_nodes": [
            {"op": row.get("op"), "name": row.get("name"), "src0_type": row.get("src0_type"),
             "file": row.get("file"), "symbol": row.get("symbol"),
             "gap_share": row.get("gap_share"), "achieved_gb_s": row.get("achieved_gb_s"),
             "ceiling_gb_s": row.get("ceiling_gb_s"),
             "coverage_basis": row.get("coverage_basis")}
            for row in sorted(uncovered_rows, key=lambda r: -float(r.get("gap_share", 0.0)))],
    }


def from_node_dump(dump: Mapping[str, Any], *,
                   ceiling_gb_s: float = DEFAULT_HOST_CEILING_GB_S,
                   threshold: float = UNCOVERED_GAP_WARN_THRESHOLD,
                   limit: int = 12) -> dict[str, Any]:
    """Entry point: a `ggml-cpu-prof/1`-shaped dump in, the full gap report out.

    Reads ONLY `dump["nodes"]`. `dump["paths"]` (the raw dump's weight-path aggregate,
    `weight_path_shares` in `node_profile`'s parsed view) is the path-level
    `GBs_on_wall` the 2026-10-06 roofline audit found contaminated by prefill -- a
    path pools every node of every shape and every phase that hits it -- and is never
    read here, by construction.
    """
    total_wall_us = dump.get("total_wall_us")
    if not isinstance(total_wall_us, (int, float)) or total_wall_us <= 0:
        raise RooflineRefused("dump.total_wall_us must be positive")
    nodes = dump.get("nodes") or []
    gap_table = route_coverage_map(
        per_node_roofline(nodes, total_wall_us=total_wall_us,
                          ceiling_gb_s=ceiling_gb_s, limit=limit))
    return {"schema": SCHEMA, "ceiling_gb_s": float(ceiling_gb_s),
            "gap_table": gap_table,
            "uncovered_gap": uncovered_gap_summary(gap_table, threshold=threshold)}


def is_stagnant(stagnation: Mapping[str, Any] | None, *,
                min_points: int = STAGNATION_MIN_MEASURED_POINTS,
                null_streak: int = STAGNATION_NULL_STREAK) -> bool:
    """`status.stagnation_signal`'s view, read as a yes/no for the scope-gap hook.

    Either a non-positive best-effect slope over enough trailing points, or a long
    enough run of consecutive measured-null outcomes, counts: a lane can stagnate by
    flatlining OR by abstaining/nulling every attempt, and this hook must catch both.
    """
    if not isinstance(stagnation, Mapping):
        return False
    slope = stagnation.get("best_effect_slope_per_measurement")
    points = stagnation.get("measured_points") or 0
    if isinstance(slope, (int, float)) and not isinstance(slope, bool) \
            and slope <= 0 and points >= min_points:
        return True
    nulls = stagnation.get("consecutive_measured_nulls") or 0
    return isinstance(nulls, (int, float)) and nulls >= null_streak


def write_scope_gap(store_root: Path | str, uncovered_gap: Mapping[str, Any], *,
                    now: str | None = None) -> Path:
    """Write the machine-readable `scope_gap.json` naming the uncovered files/symbols.

    Called from the stagnation hook, never on every iteration: this is a durable
    artifact for a human to act on (widen a route, or decide not to), not a live
    status field -- that is `loop-status.json`'s `uncovered_gap`, written every beat.
    """
    body = {**dict(uncovered_gap), "schema": SCOPE_GAP_SCHEMA,
            "generated_at": now or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    target = Path(store_root) / SCOPE_GAP_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return target


def stagnation_scope_gap_hook(uncovered_gap: Mapping[str, Any] | None,
                              stagnation: Mapping[str, Any] | None, *,
                              store_root: Path | str) -> Path | None:
    """The stagnation hook: write `scope_gap.json` iff BOTH the gap and the stall hold.

    Pure decision plus one write; never raises on a bad `stagnation` shape (observe-
    only instrumentation must never cost the loop its own publish step) -- a
    malformed `uncovered_gap` still raises, because that is this module's own output
    and a malformed one is this module's own bug to surface, not swallow.
    """
    if not uncovered_gap or not uncovered_gap.get("triggered"):
        return None
    if not is_stagnant(stagnation):
        return None
    return write_scope_gap(store_root, uncovered_gap)


__all__ = ["DEFAULT_HOST_CEILING_GB_S", "GGML_TYPE_BLOCK_BYTES", "ROUTE_FILE_SYMBOL",
           "RooflineRefused", "SCHEMA", "SCOPE_GAP_FILENAME", "SCOPE_GAP_SCHEMA",
           "STAGNATION_MIN_MEASURED_POINTS", "STAGNATION_NULL_STREAK",
           "UNCOVERED_GAP_WARN_THRESHOLD", "from_node_dump", "is_stagnant",
           "node_achieved_bandwidth", "per_node_roofline", "resolve_route_file_symbol",
           "route_coverage_map", "route_is_admitted", "stagnation_scope_gap_hook",
           "tensor_bytes", "uncovered_gap_summary", "write_scope_gap"]
