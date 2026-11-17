"""AKX-ALL-31.EDGE: edge-value input families and a per-element worst-error bound beside NMSE.

The AKX-ALL-31 chain judged two planted gates (non-finite classes, exact fp32 accumulation). This
module extends it, on both native legs (CPU and HIP, producer `test-akx-numerics --edge`), with four
STRUCTURED input families on the served matmul shapes (`served_shape_cases.canonical_triples`):

  near_f16_max   activations up to 61440 (just under the f16 max 65504): every partial sum of a
                 row overflows an f16 accumulator or an f16 block-sum field, fp32 does not.
  f16_subnormal  weights at 2^-12 and activations at 2^-10 scale, so every per-block f16 scale
                 (weight d, activation q8_1 d) lands in the f16 SUBNORMAL range: a kernel that
                 flushes subnormal half scales to zero disagrees with the dequantizer.
  cancel_rows    every row is [w, -w] against activations [x, x + 2^-12 u]: the dot is
                 -sum(w * 2^-12 u), a tiny result built from large terms (catastrophic
                 cancellation; an error proportional to the TERMS, not the result, shows here).
  tile_const     weights constant per 16-row x 32-column tile, activations constant per 32-block:
                 degenerate quant blocks (amax == every element) and constant block sums.

The metric is the aggregate NMSE AND the per-element maximum absolute error against a float64
reference built from the backend's own dequantized weights (ggml `to_float`), so activation
quantization and accumulation are what is measured. Bounds are calibrated per
(op, type, shape, width, family) from the ANCHOR (v10) over `CALIBRATION_SEEDS` (5 seeds):
nmse bound = `served_shape_cases.tightened_nmse_bound` (the served-shape formula), max-abs bound =
`factor` x the anchor's worst max-abs over the seeds (floored at 2^-20 x the reference rms). A fault
confined to one tile or one row moves the NMSE by ~1/m of its size and hides in it; the max-abs
bound sees it.

SEEDED-DEFECT TWINS. For every case the producer also runs two mutants through the REAL kernel:
`twin_row` negates one 32-element weight block of one seeded row, `twin_tile` negates the same block
on a seeded 16-row tile (the block chosen as the one contributing most to token 0, so the defect is
never accidentally invisible). They are judged against the CLEAN reference and MUST go red; a twin
that passes voids the chain (`numerics_edge_twin_not_red`, an instrument fault). At calibration a
key whose twins are not red on the anchor itself is excluded with that reason, and a family left
with no discriminating key refuses the calibration.

Inputs are generated identically on host and native (counter-based splitmix64, scheme
`GEN_SCHEME`), so each receipt's `probe` (first and last W and X element) is checked bit-for-bit
against this generator: a native leg that judged different data is refused, not believed.

Receipts: `AK_NUMERICS_EDGE_V1 {json}` per (case, width, variant); bounds:
`AK_NUMERICS_EDGE_BOUNDS_V1 {json}`. Both parsed with `guard_bands.parse_receipts` (reused, not a
new framework). OPT-IN like the rest of AKX-ALL-31: only `gates._numerics_chain` calls this, and only
when `AK_NUMERICS_GATES` is on; `AK_NUMERICS_EDGE_BOUNDS` names the bounds file.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import gates, guard_bands

SCHEMA = "epyc.autokernel.numerics_edge.v1"
RECEIPT_PREFIX = "AK_NUMERICS_EDGE_V1 "
BOUNDS_SCHEMA = "epyc.autokernel.numerics_edge_bounds.v1"
BOUNDS_PREFIX = "AK_NUMERICS_EDGE_BOUNDS_V1 "
GEN_SCHEME = "akx-edge-splitmix64-v1"
BOUNDS_ENV = "AK_NUMERICS_EDGE_BOUNDS"
GATE_EDGE = "numerics_edge"
GATE_TWINS = "numerics_edge_twins"
GATE_CONTROLS = "numerics_edge_controls"
#: reasons that are the instrument's fault (gates maps them to oracle_unavailable)
ORACLE_PREFIXES = ("numerics_edge_receipt_", "numerics_edge_bounds_", "numerics_edge_twin_not_red",
                   "numerics_edge_controls_")

FAMILIES = ("near_f16_max", "f16_subnormal", "cancel_rows", "tile_const")
VARIANTS = ("incumbent", "twin_row", "twin_tile")
TWINS = ("twin_row", "twin_tile")
CALIBRATION_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_FACTOR = 3.0
MAXABS_FLOOR_REL = 2.0 ** -20
TILE_ROWS = 16
BLOCK = 32
NEAR_F16_MAX_SCALE = 61440.0
SUBNORMAL_W_SCALE = 2.0 ** -12
SUBNORMAL_X_SCALE = 2.0 ** -10
CANCEL_DELTA_SCALE = 2.0 ** -12
STREAM_W, STREAM_X, STREAM_DELTA, STREAM_TWIN = 0, 1, 2, 3
_M64 = (1 << 64) - 1
_FIELDS = ("op", "type", "k", "m", "n_mats", "n_used", "n", "family", "seed", "variant")


# --- deterministic generator (bit-identical to test-akx-numerics.cpp akx_sm64 / akx_u) -------------

def splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def stream_base(seed: int, stream: int) -> int:
    return splitmix64(((int(seed) << 8) | int(stream)) & _M64)


def _uniform(base: int, idx):
    """u in [-1, 1) with 2^-23 granularity (exact in float32) at counter `base + idx`."""
    import numpy as np
    z = np.asarray(idx, dtype=np.uint64) + np.uint64(base)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(40)).astype(np.float64) * 2.0 ** -23 - 1.0).astype(np.float32)


@dataclass(frozen=True)
class EdgeCase:
    """One (op, ggml type, shape, widths, family, seed) the native producer evaluates."""
    op: str                     # MUL_MAT | MUL_MAT_ID
    type: str                   # ggml_type_name spelling (q4_K, iq2_xxs, mxfp4, f32 ...)
    k: int
    m: int
    n_mats: int
    n_used: int
    widths: tuple[int, ...]
    family: str
    seed: int

    def __post_init__(self):
        if self.op not in ("MUL_MAT", "MUL_MAT_ID"):
            raise ValueError(f"op must be MUL_MAT or MUL_MAT_ID, got {self.op!r}")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if self.k <= 0 or self.k % BLOCK or self.m <= 0:
            raise ValueError(f"k must be a positive multiple of {BLOCK} and m positive")
        if self.family == "cancel_rows" and self.k % (2 * BLOCK):
            raise ValueError("cancel_rows needs k divisible by 64")
        if self.op == "MUL_MAT" and (self.n_mats, self.n_used) != (1, 1):
            raise ValueError("MUL_MAT carries no n_mats/n_used")
        if self.op == "MUL_MAT_ID" and not (1 <= self.n_used <= self.n_mats):
            raise ValueError("n_used must be in [1, n_mats]")
        if not self.widths or any(int(w) <= 0 for w in self.widths):
            raise ValueError("widths must be positive")

    @property
    def slots(self) -> int:
        return self.n_used if self.op == "MUL_MAT_ID" else 1

    @property
    def nmax(self) -> int:
        return max(self.widths)

    def line(self) -> str:
        return (f"EDGE {self.op} {self.type} {self.k} {self.m} {self.n_mats} {self.n_used} "
                f"{','.join(str(w) for w in self.widths)} {self.family} {self.seed}")


def parse_case_line(line: str) -> EdgeCase:
    tag, op, ty, k, m, e, u, widths, fam, seed = line.split()
    if tag != "EDGE":
        raise ValueError(f"not an EDGE line: {line!r}")
    return EdgeCase(op, ty, int(k), int(m), int(e), int(u),
                    tuple(int(w) for w in widths.split(",")), fam, int(seed))


def key_of(rec: Mapping[str, Any]) -> str:
    """The calibration key: (op, type, k, m, n_mats, n_used, width, family)."""
    return "|".join(str(rec[f]) for f in ("op", "type", "k", "m", "n_mats", "n_used", "n", "family"))


def w_at(case: EdgeCase, idx):
    """Weight elements at flat indices of the [n_mats][m][k] array (float32)."""
    import numpy as np
    idx = np.asarray(idx, dtype=np.int64)
    k, m = case.k, case.m
    er, col = np.divmod(idx, k)
    e, r = np.divmod(er, m)
    bw = stream_base(case.seed, STREAM_W)
    if case.family == "tile_const":
        tr, kb = -(-m // TILE_ROWS), k // BLOCK
        return _uniform(bw, (e * tr + r // TILE_ROWS) * kb + col // BLOCK)
    if case.family == "cancel_rows":
        h = k // 2
        w = _uniform(bw, er * k + np.where(col < h, col, col - h))
        return np.where(col < h, w, -w).astype(np.float32)
    w = _uniform(bw, idx)
    if case.family == "f16_subnormal":
        w = (w * np.float32(SUBNORMAL_W_SCALE)).astype(np.float32)
    return w


def x_at(case: EdgeCase, idx):
    """Activation elements at flat indices of the [nmax][slots][k] array (float32)."""
    import numpy as np
    idx = np.asarray(idx, dtype=np.int64)
    k = case.k
    ts, col = np.divmod(idx, k)
    bx = stream_base(case.seed, STREAM_X)
    if case.family == "tile_const":
        return _uniform(bx, ts * (k // BLOCK) + col // BLOCK)
    if case.family == "cancel_rows":
        h = k // 2
        c0 = np.where(col < h, col, col - h)
        x = _uniform(bx, ts * k + c0)
        d = (_uniform(stream_base(case.seed, STREAM_DELTA), ts * k + c0)
             * np.float32(CANCEL_DELTA_SCALE)).astype(np.float32)
        return np.where(col < h, x, (x + d).astype(np.float32)).astype(np.float32)
    x = _uniform(bx, idx)
    if case.family == "near_f16_max":
        return (x * np.float32(NEAR_F16_MAX_SCALE)).astype(np.float32)
    if case.family == "f16_subnormal":
        return (x * np.float32(SUBNORMAL_X_SCALE)).astype(np.float32)
    return x


def weights(case: EdgeCase):
    import numpy as np
    n = case.n_mats * case.m * case.k
    return w_at(case, np.arange(n)).reshape(case.n_mats, case.m, case.k)


def activations(case: EdgeCase):
    import numpy as np
    n = case.nmax * case.slots * case.k
    return x_at(case, np.arange(n)).reshape(case.nmax, case.slots, case.k)


def route_ids(case: EdgeCase):
    """[nmax][slots] expert ids: `served_shape_cases.route_ids` per token (0 for MUL_MAT)."""
    import numpy as np
    from . import served_shape_cases as ssc
    if case.op == "MUL_MAT":
        return np.zeros((case.nmax, 1), dtype=np.int64)
    return np.array([ssc.route_ids(case.n_mats, case.n_used, t) for t in range(case.nmax)], dtype=np.int64)


def expected_probe(case: EdgeCase) -> list[float]:
    """[W[0], W[last], X[0], X[last]] -- what the native receipt's `probe` must equal."""
    nw = case.n_mats * case.m * case.k
    nx = case.nmax * case.slots * case.k
    return [float(w_at(case, [0])[0]), float(w_at(case, [nw - 1])[0]),
            float(x_at(case, [0])[0]), float(x_at(case, [nx - 1])[0])]


def reference(case: EdgeCase, wd, x, ids=None):
    """float64 [nmax][slots][m] from the (dequantized) weights."""
    import numpy as np
    ids = route_ids(case) if ids is None else ids
    out = np.empty((case.nmax, case.slots, case.m), dtype=np.float64)
    for t in range(case.nmax):
        for s in range(case.slots):
            out[t, s] = wd[ids[t, s]].astype(np.float64) @ x[t, s].astype(np.float64)
    return out


def twin_target(case: EdgeCase, w, x, variant: str, ids=None) -> dict[str, int]:
    """(expert, row0, rows, block) a twin corrupts: seeded row / 16-row tile of the expert token 0
    slot 0 routes to; the 32-block contributing most (|sum|) to that token over those rows."""
    import numpy as np
    ids = route_ids(case) if ids is None else ids
    e0 = int(ids[0, 0])
    bt = stream_base(case.seed, STREAM_TWIN)
    if variant == "twin_row":
        row0, rows = splitmix64(bt) % case.m, 1
    elif variant == "twin_tile":
        tiles = -(-case.m // TILE_ROWS)
        row0 = (splitmix64((bt + 1) & _M64) % tiles) * TILE_ROWS
        rows = min(TILE_ROWS, case.m - row0)
    else:
        raise ValueError(f"not a twin: {variant!r}")
    sel = w[e0, row0:row0 + rows, :].astype(np.float64) * x[0, 0, :].astype(np.float64)
    contrib = np.abs(sel.reshape(rows, case.k // BLOCK, BLOCK).sum(axis=(0, 2)))
    return {"expert": e0, "row0": int(row0), "rows": int(rows), "block": int(np.argmax(contrib))}


def apply_twin(w, target: Mapping[str, int]):
    out = w.copy()
    e, r0, nr, b = target["expert"], target["row0"], target["rows"], target["block"]
    out[e, r0:r0 + nr, b * BLOCK:(b + 1) * BLOCK] *= -1
    return out


def metrics(out, ref, m: int) -> dict[str, Any]:
    """NMSE and per-element max |err| over one width's outputs (layout [n][slots][m] flattened);
    a non-finite output where the reference is finite is counted, never folded into a number."""
    import numpy as np
    o = np.asarray(out, dtype=np.float64).ravel()
    d = np.asarray(ref, dtype=np.float64).ravel()
    fin = np.isfinite(o)
    err = np.abs(np.where(fin, o, 0.0) - d)
    err[~fin] = 0.0
    se, sr = float(np.sum(err ** 2)), float(np.sum(d ** 2))
    i = int(np.argmax(err)) if err.size else 0
    return {"nmse": (se / sr) if sr > 0 else (0.0 if se == 0 else math.inf),
            "max_abs": float(err[i]) if err.size else 0.0, "arg_row": i % m, "arg_col": i // m,
            "rms_ref": math.sqrt(sr / max(1, d.size)), "max_ref": float(np.max(np.abs(d))) if d.size else 0.0,
            "nonfinite": int(np.sum(~fin))}


# --- host leg (the module's own self-test, and the unit tests' anchor/candidates) ----------------

Kernel = Callable[[Any, Any, Any], Any]   # (wd [E][m][k], x [n][slots][k], ids [n][slots]) -> [n][slots][m]


def host_kernel_fp32(wd, x, ids):
    import numpy as np
    out = np.empty((x.shape[0], x.shape[1], wd.shape[1]), dtype=np.float32)
    for t in range(x.shape[0]):
        for s in range(x.shape[1]):
            out[t, s] = (wd[ids[t, s]].astype(np.float32) @ x[t, s].astype(np.float32)).astype(np.float32)
    return out


def host_kernel_q8(wd, x, ids):
    """Emulates a q8_0-activation kernel: x quantized per 32-block (fp32 scale amax/127)."""
    import numpy as np
    n, slots, k = x.shape
    xb = x.astype(np.float32).reshape(n, slots, k // BLOCK, BLOCK)
    d = (np.max(np.abs(xb), axis=-1, keepdims=True) / np.float32(127)).astype(np.float32)
    q = np.where(d > 0, np.rint(xb / np.where(d > 0, d, 1)), 0).clip(-127, 127)
    xq = (q * d).astype(np.float32).reshape(n, slots, k)
    out = np.empty((n, slots, wd.shape[1]), dtype=np.float32)
    for t in range(n):
        for s in range(slots):
            out[t, s] = (wd[ids[t, s]].astype(np.float64) @ xq[t, s].astype(np.float64)).astype(np.float32)
    return out


def host_kernel_f16_accum(wd, x, ids):
    """Planted family control: products and the running sum in f16 (overflows on near_f16_max)."""
    import numpy as np
    n, slots, k = x.shape
    out = np.empty((n, slots, wd.shape[1]), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(n):
            for s in range(slots):
                w16 = wd[ids[t, s]].astype(np.float16)
                x16 = x[t, s].astype(np.float16)
                acc = np.zeros(wd.shape[1], dtype=np.float16)
                for c in range(k):
                    acc = (acc + (w16[:, c] * x16[c]).astype(np.float16)).astype(np.float16)
                out[t, s] = acc.astype(np.float32)
    return out


def host_receipts(case: EdgeCase, kernel: Kernel, *, backend: str = "HOST",
                  mutate: Callable[[str, int, Any], Any] | None = None) -> list[dict[str, Any]]:
    """The receipts a native leg would print for `case`, computed on the host with `kernel` (no
    weight quantization: the 'dequantized' weights are the generated ones). `mutate(variant, n,
    out)` may corrupt the output (tests plant post-kernel faults with it)."""
    w, x, ids = weights(case), activations(case), route_ids(case)
    ref = reference(case, w, x, ids)
    probe = expected_probe(case)
    out_recs = []
    for variant in VARIANTS:
        target = None if variant == "incumbent" else twin_target(case, w, x, variant, ids)
        wv = w if target is None else apply_twin(w, target)
        full = kernel(wv, x, ids)
        for n in case.widths:
            out = full[:n] if mutate is None else mutate(variant, n, full[:n].copy())
            rec = {"schema": SCHEMA, "gen": GEN_SCHEME, "backend": backend, "op": case.op,
                   "type": case.type, "k": case.k, "m": case.m, "n_mats": case.n_mats,
                   "n_used": case.n_used, "n": n, "nmax": case.nmax, "family": case.family, "seed": case.seed,
                   "variant": variant, "supported": True, "probe": probe, "twin": target}
            rec.update(metrics(out, ref[:n], case.m))
            out_recs.append(rec)
    return out_recs


def receipt_text(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(RECEIPT_PREFIX + json.dumps(_jsonable(r)) + "\n" for r in records)


def _jsonable(rec: Mapping[str, Any]) -> dict[str, Any]:
    enc = lambda v: ("inf" if v > 0 else "-inf") if isinstance(v, float) and math.isinf(v) else (
        "nan" if isinstance(v, float) and math.isnan(v) else v)
    return {k: enc(v) for k, v in rec.items()}


def _num(v) -> float:
    if isinstance(v, str):
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[v]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"not a number: {v!r}")
    return float(v)


# --- receipts ---------------------------------------------------------------------------------

def parse_records(text: str) -> list[dict[str, Any]]:
    return [r for r in guard_bands.parse_receipts(text, prefix=RECEIPT_PREFIX)
            if isinstance(r, dict) and r.get("schema") == SCHEMA]


def _case_of(rec: Mapping[str, Any]) -> EdgeCase:
    return EdgeCase(rec["op"], rec["type"], int(rec["k"]), int(rec["m"]), int(rec["n_mats"]),
                    int(rec["n_used"]), (int(rec.get("nmax", rec["n"])),), rec["family"], int(rec["seed"]))


def probe_ok(rec: Mapping[str, Any], _cache: dict | None = None) -> bool:
    """The receipt's inputs are this module's inputs, bit for bit (float32 compare). The native
    receipt carries `nmax` (the widest width the case generated X for); the X probe depends on it."""
    import numpy as np
    try:
        case = _case_of(rec)
        key = (case.op, case.k, case.m, case.n_mats, case.n_used, case.nmax, case.family, case.seed)
        if _cache is not None and key in _cache:
            want = _cache[key]
        else:
            want = expected_probe(case)
            if _cache is not None:
                _cache[key] = want
        got = [float(v) for v in rec["probe"]]
    except (KeyError, TypeError, ValueError):
        return False
    return len(got) == 4 and all(np.float32(a) == np.float32(b) for a, b in zip(got, want))


def _red(rec: Mapping[str, Any], bound: float) -> bool:
    return int(rec.get("nonfinite", 0)) > 0 or _num(rec["max_abs"]) > bound


def _index(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    out: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for r in records:
        try:
            out.setdefault((key_of(r), int(r["seed"]), str(r["variant"])), r)
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --- calibration ------------------------------------------------------------------------------

def calibrate(records: Sequence[Mapping[str, Any]], *, factor: float = DEFAULT_FACTOR,
              seeds: Sequence[int] = CALIBRATION_SEEDS, anchor: str = "") -> dict[str, Any]:
    """Per-key bounds from the ANCHOR's receipts over `seeds`. Raises ValueError on input that
    cannot be calibrated at all (mixed backends, an incomplete seed set, a probe mismatch);
    returns a bounds receipt whose `ok` is False when a family is left with no discriminating
    key. Excluded keys (unsupported on the backend, anchor non-finite -- a FINDING about the
    anchor --, twin not red on the anchor) are listed with their reason, never silently dropped."""
    from . import served_shape_cases as ssc
    if not isinstance(factor, (int, float)) or factor <= 0:
        raise ValueError(f"factor must be positive, got {factor!r}")
    recs = [r for r in records if r.get("schema") == SCHEMA]
    if not recs:
        raise ValueError("no edge receipts to calibrate from")
    if any(r.get("gen") != GEN_SCHEME for r in recs):
        raise ValueError(f"receipt generator scheme is not {GEN_SCHEME}")
    backends = sorted({str(r.get("backend")) for r in recs})
    if len(backends) != 1:
        raise ValueError(f"calibration receipts span several backends {backends}; calibrate per leg")
    cache: dict = {}
    bad_probe = sorted({key_of(r) for r in recs if not probe_ok(r, cache)})
    if bad_probe:
        raise ValueError(f"input probe mismatch on {len(bad_probe)} key(s), e.g. {bad_probe[0]}: the "
                         "producer did not evaluate this generator's inputs")
    idx = _index(recs)
    keys = sorted({key_of(r) for r in recs})
    want_seeds = sorted(int(s) for s in seeds)
    bounds: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    # AKX-ALL-31.EDGE-HIP-NONFINITE: an anchor that goes non-finite on finite inputs cannot give a
    # bound, but the key is NOT dropped -- the candidate is still judged there on "no new non-finite"
    # (per seed, its non-finite count may not exceed the anchor's).
    nonfinite_watch: dict[str, dict] = {}
    for key in keys:
        missing = [(s, v) for s in want_seeds for v in VARIANTS if (key, s, v) not in idx]
        if missing:
            raise ValueError(f"{key}: calibration needs every variant on seeds {want_seeds}; missing "
                             f"{missing[:4]}")
        inc = [idx[(key, s, "incumbent")] for s in want_seeds]
        supported = [bool(r.get("supported", True)) for r in inc]
        if not any(supported):
            excluded[key] = "unsupported: the anchor backend does not support this op/type"
            continue
        if not all(supported):
            raise ValueError(f"{key}: support differs across seeds {supported}")
        nonfin = [s for s, r in zip(want_seeds, inc) if int(r.get("nonfinite", 0)) > 0]
        if nonfin:
            nonfinite_watch[key] = {
                "anchor_nonfinite_seeds": nonfin,
                "anchor_nonfinite": {str(s): int(r.get("nonfinite", 0)) for s, r in zip(want_seeds, inc)},
                "finding": "the ANCHOR produces non-finite outputs on finite inputs here (a finding to "
                           "file); judged on the candidate as no-new-non-finite, not bounded"}
            continue
        nmse_max = max(_num(r["nmse"]) for r in inc)
        maxabs_max = max(_num(r["max_abs"]) for r in inc)
        rms_min = min(_num(r["rms_ref"]) for r in inc)
        if not all(math.isfinite(v) for v in (nmse_max, maxabs_max, rms_min)):
            excluded[key] = "anchor_metric_nonfinite"
            continue
        anchor_relative = False
        try:
            nmse_bound = ssc.tightened_nmse_bound(nmse_max, factor=factor)
        except ValueError:
            nmse_bound = ssc.tightened_nmse_bound(nmse_max, factor=factor, anchor_relative=True)
            anchor_relative = True
        maxabs_bound = max(factor * maxabs_max, MAXABS_FLOOR_REL * rms_min)
        not_red = [(v, s, _num(idx[(key, s, v)]["max_abs"])) for s in want_seeds for v in TWINS
                   if not _red(idx[(key, s, v)], maxabs_bound)]
        if not_red:
            v, s, x = not_red[0]
            excluded[key] = (f"twin_not_red_on_anchor: {v} seed {s} max_abs {x:.3e} <= bound "
                             f"{maxabs_bound:.3e} (+{len(not_red) - 1} more)")
            continue
        bounds[key] = {"nmse_bound": nmse_bound, "maxabs_bound": maxabs_bound,
                       "anchor_nmse_max": nmse_max, "anchor_maxabs_max": maxabs_max,
                       "anchor_rms_ref_min": rms_min, "nmse_anchor_relative": anchor_relative}
    problems = []
    for fam in sorted({k.rsplit("|", 1)[1] for k in keys}):
        if not any(k.rsplit("|", 1)[1] == fam for k in bounds):
            problems.append(f"family {fam} has no discriminating key (every key excluded)")
    return {"schema": BOUNDS_SCHEMA, "gen": GEN_SCHEME, "backend": backends[0], "anchor": anchor,
            "factor": factor, "seeds": want_seeds, "maxabs_floor_rel": MAXABS_FLOOR_REL,
            "ok": not problems, "problems": problems, "bounds": bounds, "excluded": excluded,
            "nonfinite_watch": nonfinite_watch}


def bounds_text(bounds: Mapping[str, Any]) -> str:
    return BOUNDS_PREFIX + json.dumps(bounds, sort_keys=True) + "\n"


def load_bounds(text: str) -> dict[str, Any] | None:
    found = [b for b in guard_bands.parse_receipts(text, prefix=BOUNDS_PREFIX)
             if isinstance(b, dict) and b.get("schema") == BOUNDS_SCHEMA]
    return found[-1] if found else None


# --- the gate ---------------------------------------------------------------------------------

def edge_verdicts(records: Sequence[Mapping[str, Any]], bounds: Mapping[str, Any] | None) -> list[gates.Verdict]:
    """Twins first (a gate whose twin passes cannot vouch for a candidate), then the incumbent
    against every calibrated key on every calibrated seed. Absence is never a pass."""
    if not isinstance(bounds, Mapping) or bounds.get("schema") != BOUNDS_SCHEMA:
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_bounds_missing: no {BOUNDS_SCHEMA} receipt")]
    if bounds.get("gen") != GEN_SCHEME:
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_bounds_stale: generator {bounds.get('gen')!r} "
                              f"is not {GEN_SCHEME}; recalibrate")]
    if not bounds.get("ok"):
        return [gates.Verdict(GATE_EDGE, False, "numerics_edge_bounds_refused: "
                              + "; ".join(bounds.get("problems") or ["calibration not ok"]))]
    recs = [r for r in records if isinstance(r, Mapping) and r.get("schema") == SCHEMA]
    table: Mapping[str, Mapping[str, float]] = bounds.get("bounds") or {}
    if not table:
        return [gates.Verdict(GATE_EDGE, False, "numerics_edge_bounds_refused: empty bound table")]
    backends = sorted({str(r.get("backend")) for r in recs})
    if recs and backends != [str(bounds.get("backend"))]:
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_receipt_backend_mismatch: receipts from "
                              f"{backends}, bounds calibrated on {bounds.get('backend')!r}")]
    idx = _index(recs)
    seeds = [int(s) for s in bounds.get("seeds") or ()]
    need = [(k, s, v) for k in sorted(table) for s in seeds for v in VARIANTS]
    watch: Mapping[str, Mapping[str, Any]] = bounds.get("nonfinite_watch") or {}
    watch_need = [(k, s, "incumbent") for k in sorted(watch) for s in seeds]
    missing = [n for n in need + watch_need if n not in idx]
    if missing:
        k, s, v = missing[0]
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_receipt_missing: {len(missing)} of {len(need)} "
                              f"receipt(s), e.g. {k} seed {s} {v}; absence is not a pass")]
    cache: dict = {}
    bad = [n for n in need + watch_need if not probe_ok(idx[n], cache)]
    if bad:
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_receipt_input_mismatch: {len(bad)} receipt(s), "
                              f"e.g. {bad[0][0]} seed {bad[0][1]}: the inputs are not {GEN_SCHEME}'s")]
    unsupported = [n for n in need + watch_need if not idx[n].get("supported", True)]
    if unsupported:
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_receipt_unsupported: {unsupported[0][0]} is "
                              "calibrated but the candidate backend reports it unsupported")]
    try:
        not_red = [(k, s, v, _num(idx[(k, s, v)]["max_abs"]), table[k]["maxabs_bound"])
                   for k, s, v in need if v in TWINS and not _red(idx[(k, s, v)], table[k]["maxabs_bound"])]
    except (KeyError, TypeError):
        return [gates.Verdict(GATE_EDGE, False, "numerics_edge_receipt_malformed")]
    if not_red:
        k, s, v, x, b = not_red[0]
        return [gates.Verdict(GATE_TWINS, False, f"numerics_edge_twin_not_red: {v} on {k} seed {s} max_abs "
                              f"{x:.3e} <= bound {b:.3e} (+{len(not_red) - 1} more); the bound cannot see a "
                              "one-row/one-tile fault here")]
    twins = gates.Verdict(GATE_TWINS, True, f"{sum(1 for n in need if n[2] in TWINS)} seeded-defect twin(s) red")
    viol = []
    for k, s, v in need:
        if v != "incumbent":
            continue
        r, b = idx[(k, s, v)], table[k]
        try:
            nmse, mx = _num(r["nmse"]), _num(r["max_abs"])
        except (KeyError, TypeError):
            return [twins, gates.Verdict(GATE_EDGE, False, "numerics_edge_receipt_malformed")]
        where = f"{k} seed {s}"
        if int(r.get("nonfinite", 0)) > 0:
            viol.append(f"numerics_edge_nonfinite: {where}: {r['nonfinite']} non-finite output(s) on finite inputs")
        if mx > b["maxabs_bound"]:
            viol.append(f"numerics_edge_maxabs: {where}: max|err| {mx:.3e} > bound {b['maxabs_bound']:.3e} at row "
                        f"{r.get('arg_row')} col {r.get('arg_col')} (nmse {nmse:.3e}, bound {b['nmse_bound']:.3e})")
        if not nmse <= b["nmse_bound"]:
            viol.append(f"numerics_edge_nmse: {where}: nmse {nmse:.3e} > bound {b['nmse_bound']:.3e}")
    for k, s, _v in watch_need:
        try:
            got = int(idx[(k, s, "incumbent")].get("nonfinite", 0))
            allowed = int((watch[k].get("anchor_nonfinite") or {}).get(str(s), 0))
        except (TypeError, ValueError, AttributeError):
            return [twins, gates.Verdict(GATE_EDGE, False, "numerics_edge_receipt_malformed")]
        if got > allowed:
            viol.append(f"numerics_edge_new_nonfinite: {k} seed {s}: {got} non-finite output(s) > the anchor's "
                        f"{allowed} on finite inputs (anchor-non-finite key, judged on no new non-finite)")
    if viol:
        return [twins, gates.Verdict(GATE_EDGE, False, viol[0] + (f" (+{len(viol) - 1} more)" if len(viol) > 1 else ""),
                                     "\n".join(viol[:16]))]
    excl = bounds.get("excluded") or {}
    return [twins, gates.Verdict(GATE_EDGE, True, f"{len(table)} key(s) x {len(seeds)} seed(s) within the anchor's "
                                 f"NMSE and per-element max-abs bounds; {len(watch)} anchor-non-finite key(s) with "
                                 f"no new non-finite; {len(excl)} key(s) excluded at calibration")]


# --- host self-test (the chain's instrument check) ---------------------------------------------

def control_cases(seeds: Sequence[int] = CALIBRATION_SEEDS) -> list[EdgeCase]:
    shapes = (("MUL_MAT", 128, 256, 1, 1, (1, 4)), ("MUL_MAT_ID", 128, 64, 4, 2, (1, 4)))
    return [EdgeCase(op, "f32", k, m, e, u, w, fam, s)
            for op, k, m, e, u, w in shapes for fam in FAMILIES for s in seeds]


def controls_gate() -> gates.Verdict:
    """On the host: calibrate a q8-activation anchor over the 5 seeds, then (1) the same kernel
    passes, (2) every twin is red, (3) an f16-accumulating kernel goes red on near_f16_max and
    (4) a one-row output fault that the NMSE bound misses is caught by the max-abs bound."""
    import numpy as np
    problems = []
    try:
        cases = control_cases()
        anchor = [r for c in cases for r in host_receipts(c, host_kernel_q8)]
        bounds = calibrate(anchor, anchor="host_q8")
        if not bounds["ok"] or bounds["excluded"] or bounds["nonfinite_watch"]:
            problems.append(f"host calibration not clean: {bounds['problems']} {list(bounds['excluded'].items())[:2]} "
                            f"{sorted(bounds['nonfinite_watch'])[:2]}")
        vs = edge_verdicts(anchor, bounds)
        if not all(v.passed for v in vs):
            problems.append("clean host anchor failed: " + "; ".join(v.reason for v in vs if not v.passed))
        f16 = [r for c in cases if c.family == "near_f16_max" and c.op == "MUL_MAT"
               for r in host_receipts(c, host_kernel_f16_accum)]
        rest = [r for r in anchor if not (r["family"] == "near_f16_max" and r["op"] == "MUL_MAT")]
        if all(v.passed for v in edge_verdicts(rest + f16, bounds)):
            problems.append("f16-accumulating kernel passed near_f16_max")
        row = one_row_fault_records(anchor, bounds)
        vs = edge_verdicts(row, bounds)
        if all(v.passed for v in vs) or "numerics_edge_maxabs" not in vs[-1].reason:
            problems.append("one-row fault under the NMSE bound was not caught by max-abs")
    except (ValueError, KeyError, ImportError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    if problems:
        return gates.Verdict(GATE_CONTROLS, False, "numerics_edge_controls_failed: " + "; ".join(problems))
    return gates.Verdict(GATE_CONTROLS, True, "host edge controls: clean green, twins red, f16 accumulation "
                         "red, NMSE-hidden row fault red on max-abs")


def one_row_fault_records(records: Sequence[Mapping[str, Any]], bounds: Mapping[str, Any]) -> list[dict]:
    """Copies of `records` where ONE incumbent receipt carries a single-row fault sized to sit
    between the max-abs bound and the NMSE bound (the case aggregate NMSE cannot see)."""
    out = [dict(r) for r in records]
    for r in out:
        if r["variant"] != "incumbent" or r["n"] < 2:
            continue
        b = bounds["bounds"].get(key_of(r))
        if not b:
            continue
        # one element off by 2x the max-abs bound: nmse rises by (2B)^2 / (N rms^2)
        n_el = r["n"] * r["n_used"] * r["m"] if r["op"] == "MUL_MAT_ID" else r["n"] * r["m"]
        extra = (2 * b["maxabs_bound"]) ** 2 / (n_el * r["rms_ref"] ** 2)
        if r["nmse"] + extra <= b["nmse_bound"]:
            r["max_abs"], r["nmse"] = 2 * b["maxabs_bound"], r["nmse"] + extra
            return out
    raise ValueError("no receipt admits an NMSE-hidden one-row fault")


def chain_verdicts(output: str, environ: Mapping[str, str] | None = None) -> list[gates.Verdict]:
    """`gates._numerics_chain`'s entry. No bounds file configured and no edge receipt in the output:
    [] (edge not in play for this run). Otherwise bounds are required, the host controls run
    first, then `edge_verdicts`."""
    env = os.environ if environ is None else environ
    path = env.get(BOUNDS_ENV, "").strip()
    records = parse_records(output)
    if not path and not records:
        return []
    if not path:
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_bounds_missing: edge receipts present but "
                              f"{BOUNDS_ENV} is unset")]
    try:
        with open(path, encoding="utf-8") as f:
            bounds = load_bounds(f.read())
    except OSError as exc:
        return [gates.Verdict(GATE_EDGE, False, f"numerics_edge_bounds_missing: {exc}")]
    controls = controls_gate()
    if not controls.passed:
        return [controls]
    return [controls, *edge_verdicts(records, bounds)]


# --- case set ---------------------------------------------------------------------------------

def edge_cases(lane: str, *, families: Sequence[str] = FAMILIES, seeds: Sequence[int] = CALIBRATION_SEEDS,
               widths: Sequence[int] | None = None) -> list[EdgeCase]:
    """The served-shape case set (`served_shape_cases.canonical_triples(lane=lane)`) grouped per
    (shape, type) with its widths, crossed with `families` and `seeds`."""
    from . import served_shape_cases as ssc
    grouped: dict[tuple, list[int]] = {}
    for shape, type_a, width in ssc.canonical_triples(lane=lane):
        if widths is None or width in widths:
            grouped.setdefault((shape, type_a), []).append(width)
    out = []
    for (shape, type_a), ws in grouped.items():
        for fam in families:
            if fam == "cancel_rows" and shape.k % (2 * BLOCK):
                continue
            for s in seeds:
                out.append(EdgeCase(shape.op, ssc.GGML_TYPE_NAMES[type_a], shape.k, shape.m, shape.n_mats,
                                    shape.n_used, tuple(sorted(ws)), fam, int(s)))
    return out


def cases_text(cases: Iterable[EdgeCase]) -> str:
    return "".join(c.line() + "\n" for c in cases)
