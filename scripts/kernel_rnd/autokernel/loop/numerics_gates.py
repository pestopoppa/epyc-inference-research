"""AKX-ALL-31: planted-control numerics gates for AK candidates (TensorFold kernel-contract gates 1 and 4).

Two gates, each with a planted control that MUST go red:

  non_finite   NaN, +Inf and -Inf are planted at known input rows; the class of every
               output (nan / +inf / -inf / finite) must equal a float64 reference.
               Tolerance-free: class equality, never a threshold.
  fp32_accum   integer inputs whose partial sums stay below 2^24, so an fp32 accumulator
               is exact in ANY order; the result must equal the exact integer. Before the
               gate may judge anything the harness proves that fp16 and bf16 replays miss
               by >= 100x the fp32 rounding bound (otherwise the inputs have no power and
               the gate refuses -- a gate that cannot tell the controls apart is not a gate).

A candidate is an `Op`: `max_rows(rows) -> list[float]` (row-wise max) and
`dot_rows(a, b) -> list[float]` (row-wise dot product). A native leg feeds the same
inputs to the kernel and returns outputs through `AK_NUMERICS_V1 {json}` receipts
(`parse_receipts`), judged by `verdict_from_receipt`; absence of a receipt is a failure.

OPT-IN. `NUMERICS_GATES_DEFAULT` is False: the gates are NOT in the default gate chain
until the operator ratifies it. `enabled()` honours an explicit flag or the
AK_NUMERICS_GATES env var; `chain_verdicts()` returns [] when disabled. Pure
Python/numpy over host values; reuses guard_bands' receipt convention, not new machinery.
"""
from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from . import gates

SCHEMA = "epyc.autokernel.numerics_gates.v1"
GATE_NON_FINITE = "numerics_non_finite"
GATE_FP32_ACCUM = "numerics_fp32_accum"
GATE_CONTROLS = "numerics_controls"
RECEIPT_PREFIX = "AK_NUMERICS_V1 "
ENV_FLAG = "AK_NUMERICS_GATES"
#: Ratification flips THIS line (see the AKX-ALL-31 report); nothing else enables the gates by default.
NUMERICS_GATES_DEFAULT = False

POWER_FACTOR = 100.0        # fp16/bf16 replays must miss by >= this x the fp32 bound
FP32_EXACT_LIMIT = 1 << 24
NAN, PINF, NINF = float("nan"), float("inf"), float("-inf")


def enabled(flag: bool | None = None, environ: Mapping[str, str] | None = None) -> bool:
    """Explicit `flag` wins; else AK_NUMERICS_GATES in {1,true,on}/{0,false,off}; else the default."""
    if flag is not None:
        return bool(flag)
    raw = (environ if environ is not None else os.environ).get(ENV_FLAG, "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return NUMERICS_GATES_DEFAULT


# --- classes ------------------------------------------------------------------------------

def value_class(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return "finite"


# --- low-precision replays (round-to-nearest-even) ----------------------------------------

def to_fp32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def to_fp16(x: float) -> float:
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def to_bf16(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return x
    bits = struct.unpack("<I", struct.pack("<f", to_fp32(x)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF0000) & 0xFFFFFFFF))[0]


def _replay_dot(a: Sequence[float], b: Sequence[float], rnd: Callable[[float], float]) -> float:
    acc = 0.0
    for x, y in zip(a, b):
        acc = rnd(acc + rnd(x * y))
    return acc


# --- candidates (the clean ones are the reference-quality implementations) ----------------

@dataclass(frozen=True)
class Op:
    name: str
    max_rows: Callable[[Sequence[Sequence[float]]], list[float]]
    dot_rows: Callable[[Sequence[Sequence[float]], Sequence[Sequence[float]]], list[float]]


def _ieee_max(row: Sequence[float]) -> float:
    out = row[0]
    for v in row[1:]:
        if math.isnan(v) or math.isnan(out):
            out = NAN
        elif v > out:
            out = v
    return out


def _fp32_dot(a, b):
    import numpy as np
    acc = np.float32(0)
    for x, y in zip(a, b):
        acc = np.float32(acc + np.float32(np.float32(x) * np.float32(y)))
    return float(acc)


CLEAN = Op("clean_fp32", lambda rows: [_ieee_max(r) for r in rows],
           lambda A, B: [_fp32_dot(a, b) for a, b in zip(A, B)])


def _nan_dropping_max(row: Sequence[float]) -> float:
    # the classic `a > b ? a : b` / fmax fold: a NaN operand silently loses
    out = row[0]
    for v in row[1:]:
        if math.isnan(out) or v > out:
            out = v
    return out


#: planted control for gate 1: NaN-dropping max (and a dot that is otherwise correct)
CONTROL_NAN_DROP = Op("control_nan_dropping_fmax", lambda rows: [_nan_dropping_max(r) for r in rows],
                      CLEAN.dot_rows)
#: planted control for gate 2: half-precision dot (max stays correct)
CONTROL_HALF_DOT = Op("control_half_precision_dot", CLEAN.max_rows,
                      lambda A, B: [_replay_dot(a, b, to_fp16) for a, b in zip(A, B)])


# --- gate 1: non-finite -------------------------------------------------------------------

def non_finite_inputs(n_cols: int = 64) -> list[list[float]]:
    """Known planted rows: clean, NaN at head/middle/tail, +Inf, -Inf, +Inf & -Inf together,
    NaN+Inf, all -Inf. Position varies so a kernel that drops a tail or a lane is caught."""
    base = [float((i * 7) % 13) - 6.0 for i in range(n_cols)]

    def planted(**at):
        r = list(base)
        for idx, v in at.items():
            r[int(idx[1:])] = v
        return r

    return [list(base),
            planted(p0=NAN), planted(p17=NAN), planted(**{f"p{n_cols - 1}": NAN}),
            planted(p3=PINF), planted(p40=NINF),
            planted(p5=PINF, p9=NINF),
            planted(p2=NAN, p11=PINF),
            [NINF] * n_cols]


def non_finite_reference(rows) -> list[str]:
    import numpy as np
    return [value_class(float(np.max(np.asarray(r, dtype=np.float64))) if not any(math.isnan(v) for v in r)
                        else NAN) for r in rows]


def non_finite_gate(op: Op, rows=None, want: Sequence[str] | None = None) -> gates.Verdict:
    """`want` overrides the float64/IEEE class reference (a native leg whose op has DOCUMENTED non-IEEE
    max semantics passes its own op's reference; default is the IEEE reference)."""
    rows = non_finite_inputs() if rows is None else rows
    want = non_finite_reference(rows) if want is None else list(want)
    got = [value_class(float(v)) for v in op.max_rows(rows)]
    if len(got) != len(want):
        return gates.Verdict(GATE_NON_FINITE, False,
                             f"non_finite_outputs_missing: {len(got)} outputs for {len(want)} rows")
    bad = [(i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
    if bad:
        i, w, g = bad[0]
        return gates.Verdict(GATE_NON_FINITE, False,
                             f"non_finite_class_mismatch: row {i} expected {w} got {g}"
                             + (f" (+{len(bad) - 1} more row(s))" if len(bad) > 1 else ""),
                             "\n".join(f"row {i}: expected {w} got {g}" for i, w, g in bad[:8]))
    return gates.Verdict(GATE_NON_FINITE, True, f"{len(rows)} planted row(s), every output class equals float64")


# --- gate 2: fp32 accumulator -------------------------------------------------------------

def accum_inputs(n_rows: int = 8, n: int = 4096, seed: int = 31):
    """Integer vectors in [-3,3]; running |partial sums| stay far below 2^24 (asserted)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    A = rng.integers(-3, 4, size=(n_rows, n)).astype(np.float64)
    B = rng.integers(1, 4, size=(n_rows, n)).astype(np.float64)
    # bias the sign so partial sums grow large enough that fp16/bf16 must round
    A = np.abs(A) + 1.0
    assert float(np.max(np.cumsum(np.abs(A * B), axis=1))) < FP32_EXACT_LIMIT
    return A.tolist(), B.tolist()


def _fp32_bound(a, b) -> float:
    """Worst-case fp32 accumulation rounding bound: n * 2^-24 * sum|a*b|."""
    s = sum(abs(x * y) for x, y in zip(a, b))
    return len(a) * 2.0 ** -24 * s


def accumulator_power_proof(A, B) -> dict[str, Any]:
    """fp16 and bf16 replays must each miss the exact sum by >= POWER_FACTOR x the fp32 bound."""
    out = {"factor_required": POWER_FACTOR}
    ok = True
    for tag, rnd in (("fp16", to_fp16), ("bf16", to_bf16)):
        ratios = []
        for a, b in zip(A, B):
            exact = sum(x * y for x, y in zip(a, b))
            err = abs(_replay_dot(a, b, rnd) - exact)
            ratios.append(err / _fp32_bound(a, b) if math.isfinite(err) else math.inf)
        out[f"{tag}_min_ratio"] = min(ratios)
        ok = ok and min(ratios) >= POWER_FACTOR
    out["ok"] = ok
    return out


def fp32_accum_gate(op: Op, A=None, B=None) -> gates.Verdict:
    if A is None:
        A, B = accum_inputs()
    proof = accumulator_power_proof(A, B)
    if not proof["ok"]:
        return gates.Verdict(GATE_FP32_ACCUM, False,
                             f"accumulator_inputs_lack_power: fp16/bf16 replays miss by <{POWER_FACTOR:g}x the "
                             f"fp32 bound ({proof}); the gate refuses to judge", json.dumps(proof))
    exact = [sum(x * y for x, y in zip(a, b)) for a, b in zip(A, B)]
    got = [float(v) for v in op.dot_rows(A, B)]
    if len(got) != len(exact):
        return gates.Verdict(GATE_FP32_ACCUM, False,
                             f"accumulator_outputs_missing: {len(got)} outputs for {len(exact)} rows")
    bad = [(i, e, g) for i, (e, g) in enumerate(zip(exact, got)) if not (g == e)]
    if bad:
        i, e, g = bad[0]
        return gates.Verdict(GATE_FP32_ACCUM, False,
                             f"accumulator_not_exact: row {i} expected {e:.0f} got {g!r} (integer inputs, "
                             "partial sums < 2^24: any fp32 order is exact)"
                             + (f" (+{len(bad) - 1} more row(s))" if len(bad) > 1 else ""))
    return gates.Verdict(GATE_FP32_ACCUM, True,
                         f"{len(exact)} row(s) exact; fp16 miss {proof['fp16_min_ratio']:.0f}x, "
                         f"bf16 miss {proof['bf16_min_ratio']:.0f}x the fp32 bound")


# --- controls and the chain ---------------------------------------------------------------

def controls_gate() -> gates.Verdict:
    """Self-test: each planted control must go red on ITS gate, and the clean candidate must
    pass both. A control that passes (or a clean that fails) fails this gate."""
    problems = []
    if non_finite_gate(CONTROL_NAN_DROP).passed:
        problems.append(f"control {CONTROL_NAN_DROP.name} passed the non-finite gate")
    if fp32_accum_gate(CONTROL_HALF_DOT).passed:
        problems.append(f"control {CONTROL_HALF_DOT.name} passed the fp32-accumulator gate")
    for v in (non_finite_gate(CLEAN), fp32_accum_gate(CLEAN)):
        if not v.passed:
            problems.append(f"clean reference failed {v.gate}: {v.reason}")
    if problems:
        return gates.Verdict(GATE_CONTROLS, False, "planted_control_not_red: " + "; ".join(problems))
    return gates.Verdict(GATE_CONTROLS, True, "both planted controls red, clean reference green")


def chain_verdicts(op: Op | None = None, *, flag: bool | None = None,
                   environ: Mapping[str, str] | None = None) -> list[gates.Verdict]:
    """The chain entry. Disabled (the default) -> []. Enabled -> controls first (a gate whose
    control passes cannot vouch for a candidate), then both gates on `op`; no `op` is a failure."""
    if not enabled(flag, environ):
        return []
    controls = controls_gate()
    if op is None:
        return [controls, gates.Verdict(GATE_NON_FINITE, False, "numerics_candidate_missing: no candidate op")]
    if not controls.passed:
        return [controls]
    return [controls, non_finite_gate(op), fp32_accum_gate(op)]


# --- native leg receipts ------------------------------------------------------------------

def make_receipt(op: Op) -> dict[str, Any]:
    """What a native harness emits (here: computed from an Op) so the receipt path is testable."""
    rows = non_finite_inputs()
    A, B = accum_inputs()
    enc = lambda v: "nan" if math.isnan(v) else ("inf" if v == PINF else ("-inf" if v == NINF else v))
    return {"schema": SCHEMA, "max_rows": [enc(float(v)) for v in op.max_rows(rows)],
            "dot_rows": [float(v) for v in op.dot_rows(A, B)]}


def parse_receipts(text: str) -> list[dict[str, Any]]:
    dec, out, i = json.JSONDecoder(), [], 0
    while (j := text.find(RECEIPT_PREFIX, i)) >= 0:
        start = j + len(RECEIPT_PREFIX)
        try:
            obj, i = dec.raw_decode(text, start)
        except ValueError:
            i = start
            continue
        out.append(obj)
    return out


def verdict_from_receipt(receipt: Mapping[str, Any] | None, *,
                         want_max: Sequence[str] | None = None) -> list[gates.Verdict]:
    if not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA:
        return [gates.Verdict(GATE_NON_FINITE, False,
                              f"numerics_receipt_missing: no {SCHEMA} receipt; absence is not a pass")]
    dec = lambda v: float(v) if not isinstance(v, str) else {"nan": NAN, "inf": PINF, "-inf": NINF}[v]
    try:
        mx = [dec(v) for v in receipt["max_rows"]]
        dt = [dec(v) for v in receipt["dot_rows"]]
    except (KeyError, TypeError, ValueError):
        return [gates.Verdict(GATE_NON_FINITE, False, "numerics_receipt_malformed")]
    op = Op("native", lambda rows: mx, lambda A, B: dt)
    return [non_finite_gate(op, want=want_max), fp32_accum_gate(op)]
