"""Noise-floor-relative model-level tolerance for non-bit-exact multi-row kernels (AK-PORT-LB1).

WHY THIS EXISTS.  A multi-row (batched-verify) kernel is not row-invariant: the incumbent's own
N-row pass already differs from its 1-row pass by floating-point reassociation, and the 27B GPU
anchor is not row-invariant by construction (N=1 is MMVQ dp4a, N>=5 is MMQ int8-MFMA stream-k).
A fixed absolute logit tolerance is therefore the wrong yardstick twice: too tight for a
legitimately reassociated candidate, too loose for a model whose own batching noise is tiny.
The yardstick is the incumbent's own batched-vs-single divergence -- its NOISE FLOOR (Locals'
``test_q35_kernels`` tolerance idea, intake-2386).  Two conditions, both required:

  (A) divergence.  ``cand`` = divergence(candidate batched, incumbent single) must be at most
      ``ratio`` (1.5) x ``floor`` = divergence(incumbent batched, incumbent single).
  (B) no greedy flips where it matters.  At every position whose incumbent top-2 logit margin
      is at least ``max(margin_floor (0.25), margin_mult (2) x floor)`` the candidate's argmax
      must equal the incumbent's.  Below that margin the two top tokens are a near-tie the
      incumbent's own noise can already flip, so a flip there is not evidence about the kernel.

Divergence is the per-position maximum absolute logit difference over the vocabulary,
aggregated over positions by ``stat`` (``max`` default; ``p99`` and ``mean`` available) and the
SAME aggregate is used for the floor and the candidate, so the ratio compares like with like.

THE VACUOUS-PASS GUARDS ARE THE POINT.  Empty input, mismatched shapes, a non-finite incumbent
value, or fewer than ``min_positions`` positions is ``unavailable``, never ``pass`` (a tolerance
computed from nothing admits everything).  A non-finite candidate value is ``wrong``.  When the
floor is exactly zero the incumbent IS row-invariant on these inputs, the 1.5x band collapses to
zero, and the candidate must match bit-for-bit -- reported as such, not silently loosened.

This module is a pure function over logit rows.  Producing the rows (a ``/completion`` with
``n_probs`` or a ``llama-perplexity``-style logits dump at N=1 and N>1 on both builds) is the
caller's job; ``gates.check_noise_relative_tolerance`` wraps the result in a ``Verdict``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

#: Locals-derived constants (intake-2386); see the module docstring.
DEFAULT_RATIO = 1.5
DEFAULT_MARGIN_FLOOR = 0.25
DEFAULT_MARGIN_MULT = 2.0
#: A floor from fewer positions is a point estimate, not a floor.
DEFAULT_MIN_POSITIONS = 8
STATS = ("max", "p99", "mean")

PASS, WRONG, UNAVAILABLE = "pass", "wrong", "unavailable"


@dataclass(frozen=True)
class Config:
    ratio: float = DEFAULT_RATIO
    margin_floor: float = DEFAULT_MARGIN_FLOOR
    margin_mult: float = DEFAULT_MARGIN_MULT
    stat: str = "max"
    min_positions: int = DEFAULT_MIN_POSITIONS

    def __post_init__(self) -> None:
        if self.stat not in STATS:
            raise ValueError(f"stat must be one of {STATS}, got {self.stat!r}")
        if not (self.ratio >= 1.0 and self.margin_floor >= 0 and self.margin_mult >= 0
                and self.min_positions >= 1):
            raise ValueError("ratio >= 1, margin_floor >= 0, margin_mult >= 0, min_positions >= 1")


@dataclass(frozen=True)
class Result:
    status: str                    # "pass" | "wrong" | "unavailable"
    reason: str
    detail: dict = field(default_factory=dict, compare=False)

    @property
    def passed(self) -> bool:
        return self.status == PASS


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    """Largest ``|a[i] - b[i]|``.  Rows must be equal length (the caller checks)."""
    return max(abs(x - y) for x, y in zip(a, b))


def top2(row: Sequence[float]) -> tuple[int, float]:
    """(argmax index, top1 - top2 margin).  Ties resolve to the first index; margin 0."""
    best = second = -math.inf
    arg = 0
    for i, v in enumerate(row):
        if v > best:
            second, best, arg = best, v, i
        elif v > second:
            second = v
    return arg, (best - second) if second > -math.inf else math.inf


def aggregate(values: Sequence[float], stat: str) -> float:
    if stat == "max":
        return max(values)
    if stat == "mean":
        return sum(values) / len(values)
    ordered = sorted(values)  # p99, nearest-rank
    return ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]


def _finite(row: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in row)


def _shape_problem(inc_single, inc_batched, cand_batched) -> str | None:
    n = len(inc_single)
    if n == 0:
        return "no positions supplied"
    if len(inc_batched) != n or len(cand_batched) != n:
        return (f"position counts differ: incumbent single {n}, incumbent batched "
                f"{len(inc_batched)}, candidate batched {len(cand_batched)}")
    for i in range(n):
        width = len(inc_single[i])
        if width < 2:
            return f"position {i}: fewer than 2 logits, no top-2 margin exists"
        if len(inc_batched[i]) != width or len(cand_batched[i]) != width:
            return f"position {i}: logit row widths differ"
    return None


def evaluate(incumbent_single: Sequence[Sequence[float]],
             incumbent_batched: Sequence[Sequence[float]],
             candidate_batched: Sequence[Sequence[float]],
             config: Config = Config()) -> Result:
    """Judge a candidate's batched logits against the incumbent's own batched-vs-single noise.

    All three arguments are lists of per-position logit rows for the SAME prompts and positions:
    the incumbent evaluated one row at a time (the reference), the incumbent evaluated in its
    multi-row pass (its noise floor), and the candidate in its multi-row pass.
    """
    problem = _shape_problem(incumbent_single, incumbent_batched, candidate_batched)
    if problem:
        return Result(UNAVAILABLE, problem)
    n = len(incumbent_single)
    if n < config.min_positions:
        return Result(UNAVAILABLE, f"{n} positions < min_positions {config.min_positions}: a floor "
                                   "from so few positions is a point estimate, not a floor")
    for name, rows in (("incumbent single", incumbent_single),
                       ("incumbent batched", incumbent_batched)):
        bad = next((i for i, r in enumerate(rows) if not _finite(r)), None)
        if bad is not None:
            return Result(UNAVAILABLE, f"{name} has a non-finite logit at position {bad}: the "
                                       "reference itself is broken")
    bad = next((i for i, r in enumerate(candidate_batched) if not _finite(r)), None)
    if bad is not None:
        return Result(WRONG, f"candidate has a non-finite logit at position {bad}")

    floor_rows = [max_abs_diff(b, s) for b, s in zip(incumbent_batched, incumbent_single)]
    cand_rows = [max_abs_diff(c, s) for c, s in zip(candidate_batched, incumbent_single)]
    floor = aggregate(floor_rows, config.stat)
    cand = aggregate(cand_rows, config.stat)
    bound = config.ratio * floor
    margin_gate = max(config.margin_floor, config.margin_mult * floor)

    flips, gated, exempt = [], 0, 0
    for i in range(n):
        arg_s, margin = top2(incumbent_single[i])
        if margin < margin_gate:
            exempt += 1
            continue
        gated += 1
        arg_c, _ = top2(candidate_batched[i])
        if arg_c != arg_s:
            flips.append({"position": i, "incumbent_argmax": arg_s, "candidate_argmax": arg_c,
                          "incumbent_margin": margin})
    detail = {"positions": n, "stat": config.stat, "floor": floor, "candidate_divergence": cand,
              "ratio_bound": bound, "ratio_observed": (cand / floor) if floor > 0 else None,
              "margin_gate": margin_gate, "gated_positions": gated, "near_tie_exempt": exempt,
              "flips": flips[:20], "n_flips": len(flips), "floor_is_zero": floor == 0.0}
    if floor == 0.0:
        # The incumbent is row-invariant on these inputs: the band is zero wide.
        if cand > 0.0:
            return Result(WRONG, f"incumbent is row-invariant here (floor 0) but the candidate "
                                 f"diverges by {cand:.6g}: bit-exactness is the only admissible "
                                 "tolerance", detail)
    elif cand > bound:
        return Result(WRONG, f"candidate divergence {cand:.6g} > {config.ratio}x the incumbent's own "
                             f"batched-vs-single floor {floor:.6g} (bound {bound:.6g})", detail)
    if flips:
        return Result(WRONG, f"{len(flips)} greedy flip(s) at positions with incumbent top-2 margin "
                             f">= {margin_gate:.4g} (first at position {flips[0]['position']})",
                      detail)
    if gated == 0:
        return Result(UNAVAILABLE, f"no position has an incumbent top-2 margin >= {margin_gate:.4g}: "
                                   "the greedy-flip half of the gate observed nothing", detail)
    return Result(PASS, f"candidate divergence {cand:.6g} <= {config.ratio}x floor {floor:.6g}; "
                        f"0 greedy flips over {gated} margin-gated positions "
                        f"({exempt} near-tie positions exempt)", detail)


__all__ = ["Config", "Result", "evaluate", "top2", "max_abs_diff", "aggregate", "PASS", "WRONG",
           "UNAVAILABLE", "DEFAULT_RATIO", "DEFAULT_MARGIN_FLOOR", "DEFAULT_MARGIN_MULT"]
