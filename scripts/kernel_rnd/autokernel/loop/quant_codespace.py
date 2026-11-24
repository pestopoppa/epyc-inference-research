"""AKX-ALL-31.QDEC: exhaustive code-space dequant check for the served quant types.

Host side of the native producer `tests/test-akx-qdec.cpp` (llama.cpp experimental branch
`llama.cpp-experimental-c12-akx-qdec-20261117`, forked from C12 457d5fc52). This module owns:

  (a) independent numpy reference decoders for Q4_K, Q6_K, Q8_0, IQ4_XS, MXFP4, IQ2_XXS, IQ2_XS,
      IQ2_S, IQ3_XXS and IQ3_S. Block layouts follow ggml/src/ggml-common.h at C12. The decoders are
      written from the layout, not transcribed from ggml-quants.c, and compute every element as ONE
      correctly rounded float32 (see EXACTNESS below), so the native output must match bit for bit;
  (b) an enumerator that writes block sets covering every per-element code at every element position
      (the ADDRESSING axes; position-digit blocks also make every pair of positions differ, so a
      permuted decode cannot pass) plus every sub-scale x element-value code under every given
      super-block scale (the ARITHMETIC axes), sharded over large harvested scale sets;
  (c) a read-only, memory-mapped GGUF scanner (own header parser) that harvests the distinct fp16
      d / dmin bit patterns (and MXFP4 E8M0 bytes) per type from the served GGUFs;
  (d) a coverage function over those axes, and a model of the test-backend-ops random suite: an exact
      emulation of ggml's reference quantizer for Q8_0 and MXFP4, a documented random-code proxy for
      the rest (it OVERSTATES reach), and coverage of real suite blocks emitted by the native
      `random` mode;
  (e) a judge that compares native float32 output bytes with the reference and emits a verdict JSON.

GRID TABLES. The IQ grids, kvalues_iq4nl and kvalues_fp4 are embedded in quant_codespace_tables.json,
parsed from `git show 457d5fc52:ggml/src/ggml-common.h` (blob 75aaf934d, sha256 below). The JSON
records that provenance and its own content hash; `tables-check --repo` re-parses the source at C12 and
compares. ksigns_iq2xs is NOT embedded: the reference derives it (7 sign bits + an even-parity 8th
bit) and the table check proves that equals the source table.

EXACTNESS. Every product the served dequant formulas form fits in a float32 significand: fp16 d has
11 significant bits, sub-scales at most 8, element codes at most 7 (d*sc*q <= 24 bits for Q6_K and
IQ4_XS; IQ db = d*(0.5+s)*k has 16 bits, times a grid byte <= 22). MXFP4 is a power of two times a
small integer (exact, subnormals included; e=255 overflows to inf for |kv|>=2 exactly as in C). The
only rounding anywhere is Q4_K's final `d*sc*q - dmin*m`, whose two operands are exact; it is rounded
once here with a TwoSum + midpoint correction. Therefore FMA contraction, operand order and float
vs double evaluation cannot legitimately change a single bit, and any bit difference is a defect (or
FTZ/DAZ on a subnormal, which the judge classifies separately). This is not true of the
MUL_MAT-identity leg (activations are re-quantized to q8), which is judged against a bound instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MARKER = "AK_QDEC_V1"
SCHEMA = "epyc.autokernel.quant_codespace.v1"
C12 = "457d5fc523ba5cc687a333a588c1dc15db9346f4"
COMMON_H = "ggml/src/ggml-common.h"
COMMON_H_BLOB = "75aaf934d2c39236df50a32cb14b0ca0f66adc4a"
COMMON_H_SHA256 = "3e7a03c2deb30a27c04aaad631ccd38a7c0403b923fdda3dc34356673f1a7a5f"
TABLES_PATH = Path(__file__).with_name("quant_codespace_tables.json")
TABLE_NAMES = ("ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid", "iq3s_grid",
               "kvalues_iq4nl", "kvalues_fp4")
EMBEDDED = tuple(n for n in TABLE_NAMES if n != "ksigns_iq2xs")
ROW = 256            # elements per native row; QK=32 types are padded to whole rows of 8 blocks
F16_ONE, F16_HALF = 0x3C00, 0x3800
#: default super-block scales when no harvested set is given: 1.0, 0.5, the smallest fp16
#: subnormal, the largest finite fp16 and a few mid-range values (both signs)
NOMINAL_D = (0x3C00, 0x3800, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0x2E66, 0x1A00, 0xBC00)
#: MUL_MAT-identity leg: one-hot activation 127 makes the q8 activation scale exactly 1.0
ONEHOT = 127.0
IDENTITY_ULPS = 8
#: enumerate measures coverage block by block up to this size, structurally above it
FULL_COVERAGE_MAX_BLOCKS = 1 << 16


def parse_tables(text: str) -> dict[str, list[int]]:
    """Parse the GGML_TABLE_BEGIN(type, name, size) ... GGML_TABLE_END() tables we need."""
    out: dict[str, list[int]] = {}
    pat = r"GGML_TABLE_BEGIN\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)(.*?)GGML_TABLE_END\(\)"
    for m in re.finditer(pat, text, re.S):
        _, name, size, body = m.groups()
        if name not in TABLE_NAMES:
            continue
        vals = [int(t, 0) for t in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", body)]
        if size.isdigit() and len(vals) != int(size):
            raise ValueError(f"{name}: {len(vals)} values, declared {size}")
        out[name] = vals
    missing = sorted(set(TABLE_NAMES) - set(out))
    if missing:
        raise ValueError(f"tables missing from source: {missing}")
    return out


def _tables_digest(tables: dict[str, list[int]]) -> str:
    return hashlib.sha256(json.dumps({k: tables[k] for k in EMBEDDED}, sort_keys=True)
                          .encode()).hexdigest()


def load_tables(path: Path = TABLES_PATH) -> dict[str, list[int]]:
    raw = json.loads(Path(path).read_text())
    tables = {k: [int(v, 16) if isinstance(v, str) else v for v in vals] for k, vals in raw["tables"].items()}
    if raw["source"]["commit"] != C12 or raw["source"]["blob"] != COMMON_H_BLOB:
        raise ValueError("embedded tables are not bound to C12 ggml-common.h")
    if _tables_digest(tables) != raw["tables_sha256"]:
        raise ValueError("embedded tables do not match their recorded sha256")
    return tables


def source_tables(repo: str | Path, commit: str = C12) -> dict[str, list[int]]:
    """Read ggml-common.h from git objects at `commit` (read-only; never touches a working tree)."""
    text = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{COMMON_H}"],
                          check=True, capture_output=True).stdout
    if commit == C12 and hashlib.sha256(text).hexdigest() != COMMON_H_SHA256:
        raise ValueError("ggml-common.h at C12 does not match the recorded sha256")
    return parse_tables(text.decode())


def write_tables_json(tables: dict[str, list[int]], path: Path = TABLES_PATH) -> None:
    doc = {"schema": SCHEMA + ".tables", "source": {"commit": C12, "path": COMMON_H,
                                                    "blob": COMMON_H_BLOB, "sha256": COMMON_H_SHA256},
           "tables_sha256": _tables_digest(tables),
           # grids as fixed-width hex strings (as in the C source); decimal uint64s read as long digit runs
           "tables": {k: ([f"0x{v:0{16 if k.startswith('iq2') else 8}x}" for v in tables[k]]
                          if k.endswith("_grid") else tables[k]) for k in EMBEDDED}}
    Path(path).write_text(json.dumps(doc, separators=(",", ":")) + "\n")


def ksigns() -> np.ndarray:
    """ksigns_iq2xs derived, not copied: low 7 bits are the index, bit 7 makes the popcount even."""
    i = np.arange(128)
    parity = np.array([bin(int(v)).count("1") & 1 for v in i])
    return (i | (parity << 7)).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# helpers

def f16(bits: np.ndarray) -> np.ndarray:
    """fp16 bit patterns -> exact float64 (subnormals, inf and NaN included)."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float64)


def round_sub_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correctly rounded float32(a - b) for float64 a, b that are exact. TwoSum gives the exact
    difference as hi + lo; hi rounds correctly to float32 unless it sits exactly on a float32
    midpoint, in which case the sign of lo decides."""
    with np.errstate(invalid="ignore", over="ignore"):
        nb = -b
        hi = a + nb
        bb = hi - a
        lo = (a - (hi - bb)) + (nb - bb)
        r = hi.astype(np.float32)
        r64 = r.astype(np.float64)
        toward = np.where(hi > r64, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
        other = np.nextafter(r, toward)
        o64 = other.astype(np.float64)
        mid = (r64 + o64) / 2
        fix = np.isfinite(hi) & (hi != r64) & (hi == mid) & (lo != 0) & \
            (np.sign(lo) == np.sign(o64 - r64))
        return np.where(fix, other, r)


def _u16(b: np.ndarray, off: int) -> np.ndarray:
    return b[:, off].astype(np.uint16) | (b[:, off + 1].astype(np.uint16) << 8)


def _u32(b: np.ndarray, off: int) -> np.ndarray:
    return (b[:, off].astype(np.uint32) | (b[:, off + 1].astype(np.uint32) << 8) |
            (b[:, off + 2].astype(np.uint32) << 16) | (b[:, off + 3].astype(np.uint32) << 24))


def _put16(b: np.ndarray, off: int, v: np.ndarray) -> None:
    v = np.asarray(v).astype(np.uint32).reshape(-1)
    b[:, off], b[:, off + 1] = v & 0xFF, (v >> 8) & 0xFF


def _put32(b: np.ndarray, off: int, v: np.ndarray) -> None:
    v = np.asarray(v).astype(np.uint64).reshape(-1)
    for k in range(4):
        b[:, off + k] = (v >> np.uint64(8 * k)) & np.uint64(0xFF)


def _bits8(byte: np.ndarray) -> np.ndarray:
    """(n, g) bytes -> (n, g*8) bits, bit j of byte g at column 8g+j."""
    return ((byte[..., None].astype(np.uint16) >> np.arange(8)) & 1).reshape(byte.shape[0], -1)


# ----------------------------------------------------------------------------------------------
# quant types. `fields` are the ADDRESSING axes: every bit of a block belongs to exactly one field
# position (fp16 scales are listed in `scales` instead), so unpack/pack are mutual inverses.
# `elements` maps fields to per-element arithmetic codes; `arith` lists the ARITHMETIC axes.

class QType:
    name = ""; ggml_id = -1; bsize = 0; qk = 256
    scales: tuple[str, ...] = ("d",)
    fields: dict[str, tuple[int, int]] = {}
    elem_codes: dict[str, int] = {}
    arith: tuple[tuple[str, ...], ...] = ()

    def unpack(self, b: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def pack(self, f: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def elements(self, f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def value(self, f: dict[str, np.ndarray], e: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def magnitude(self, f, e) -> np.ndarray:
        return np.abs(self.value(f, e).astype(np.float64))

    def blocks(self, data) -> np.ndarray:
        b = np.frombuffer(bytes(data), dtype=np.uint8) if not isinstance(data, np.ndarray) else data
        if b.size % self.bsize:
            raise ValueError(f"{self.name}: {b.size} bytes is not a whole number of blocks")
        return b.reshape(-1, self.bsize)

    def decode(self, data) -> np.ndarray:
        """Reference dequant: (n_blocks, qk) float32."""
        f = self.unpack(self.blocks(data))
        with np.errstate(over="ignore", invalid="ignore"):   # inf d / MXFP4 e=255 overflow as in C
            return self.value(f, self.elements(f)).astype(np.float32)

    def empty(self, n: int) -> np.ndarray:
        return np.zeros((n, self.bsize), dtype=np.uint8)


class Q8_0(QType):
    name, ggml_id, bsize, qk = "Q8_0", 8, 34, 32
    fields = {"q": (32, 256)}
    elem_codes = {"q": 256}
    arith = (("d", "q"),)

    def unpack(self, b):
        return {"d": _u16(b, 0)[:, None], "q": b[:, 2:34].astype(np.int64)}

    def pack(self, f):
        b = self.empty(len(f["q"]))
        _put16(b, 0, f["d"]); b[:, 2:34] = f["q"]
        return b

    def elements(self, f):
        return {"q": f["q"]}

    def value(self, f, e):
        return f16(f["d"]) * e["q"].astype(np.uint8).view(np.int8)


class MXFP4(QType):
    name, ggml_id, bsize, qk = "MXFP4", 39, 17, 32
    scales = ()
    fields = {"e": (1, 256), "q": (32, 16)}
    elem_codes = {"q": 16}
    arith = (("e", "q"),)

    def __init__(self, kvalues=None):
        self.kv = np.array(kvalues if kvalues is not None else load_tables()["kvalues_fp4"],
                           dtype=np.float64)

    def unpack(self, b):
        qs = b[:, 1:17].astype(np.int64)
        return {"e": b[:, 0:1].astype(np.int64), "q": np.concatenate([qs & 0xF, qs >> 4], axis=1)}

    def pack(self, f):
        b = self.empty(len(f["q"]))
        b[:, 0] = f["e"][:, 0]; b[:, 1:17] = f["q"][:, :16] | (f["q"][:, 16:] << 4)
        return b

    def elements(self, f):
        return {"q": f["q"]}

    def value(self, f, e):
        # GGML_E8M0_TO_FP32_HALF(e) == 2^(e-128) for every e (e<2 are the float32 subnormals)
        return np.ldexp(self.kv[e["q"]], (f["e"] - 128).astype(np.int64))


def _scale_min_k4(s: np.ndarray):
    sc = np.empty((len(s), 8), np.int64); m = np.empty((len(s), 8), np.int64)
    s = s.astype(np.int64)
    for j in range(4):
        sc[:, j], m[:, j] = s[:, j] & 63, s[:, j + 4] & 63
        sc[:, j + 4] = (s[:, j + 8] & 0xF) | ((s[:, j] >> 6) << 4)
        m[:, j + 4] = (s[:, j + 8] >> 4) | ((s[:, j + 4] >> 6) << 4)
    return sc, m


class Q4_K(QType):
    name, ggml_id, bsize = "Q4_K", 12, 144
    scales = ("d", "dmin")
    fields = {"sc": (8, 64), "m": (8, 64), "q": (256, 16)}
    elem_codes = {"sc": 64, "m": 64, "q": 16}
    arith = (("d", "sc", "q"), ("dmin", "m"))
    # element e: 64-group g, nibble h (low = first 32), lane l -> qs[32g+l]
    _e = np.arange(256)
    QS_IDX, QS_SHIFT = 32 * (_e // 64) + _e % 32, 4 * ((_e % 64) // 32)

    def unpack(self, b):
        sc, m = _scale_min_k4(b[:, 4:16])
        qs = b[:, 16:144].astype(np.int64)
        return {"d": _u16(b, 0)[:, None], "dmin": _u16(b, 2)[:, None], "sc": sc, "m": m,
                "q": (qs[:, self.QS_IDX] >> self.QS_SHIFT) & 0xF}

    def pack(self, f):
        b = self.empty(len(f["q"]))
        _put16(b, 0, f["d"]); _put16(b, 2, f["dmin"])
        sc, m = f["sc"], f["m"]
        for j in range(4):
            b[:, 4 + j] = (sc[:, j] & 63) | ((sc[:, j + 4] >> 4) << 6)
            b[:, 8 + j] = (m[:, j] & 63) | ((m[:, j + 4] >> 4) << 6)
            b[:, 12 + j] = (sc[:, j + 4] & 0xF) | ((m[:, j + 4] & 0xF) << 4)
        q = f["q"]
        for g in range(4):
            b[:, 16 + 32 * g:48 + 32 * g] = q[:, 64 * g:64 * g + 32] | (q[:, 64 * g + 32:64 * g + 64] << 4)
        return b

    def elements(self, f):
        sub = np.arange(256) // 32
        return {"sc": f["sc"][:, sub], "m": f["m"][:, sub], "q": f["q"]}

    def _terms(self, f, e):
        return f16(f["d"]) * e["sc"] * e["q"], f16(f["dmin"]) * e["m"]

    def value(self, f, e):
        return round_sub_f32(*self._terms(f, e))

    def magnitude(self, f, e):
        a, b = self._terms(f, e)
        return np.abs(a) + np.abs(b)


class Q6_K(QType):
    name, ggml_id, bsize = "Q6_K", 14, 210
    fields = {"sc": (16, 256), "q": (256, 64)}
    elem_codes = {"sc": 256, "q": 64}
    arith = (("d", "sc", "q"),)
    _e = np.arange(256)
    _n, _r = _e // 128, _e % 128
    _quarter, _l = _r // 32, _r % 32
    QL_IDX = 64 * _n + _l + 32 * (_quarter % 2)
    QL_SHIFT = 4 * (_quarter // 2)
    QH_IDX, QH_SHIFT = 32 * _n + _l, 2 * _quarter
    SC_IDX = 8 * _n + _l // 16 + 2 * _quarter

    def unpack(self, b):
        ql, qh = b[:, 0:128].astype(np.int64), b[:, 128:192].astype(np.int64)
        q = ((ql[:, self.QL_IDX] >> self.QL_SHIFT) & 0xF) | \
            (((qh[:, self.QH_IDX] >> self.QH_SHIFT) & 3) << 4)
        return {"d": _u16(b, 208)[:, None], "sc": b[:, 192:208].astype(np.int64), "q": q}

    def pack(self, f):
        n = len(f["q"]); b = self.empty(n)
        ql = np.zeros((n, 128), np.int64); qh = np.zeros((n, 64), np.int64)
        for e in range(256):
            ql[:, self.QL_IDX[e]] |= (f["q"][:, e] & 0xF) << self.QL_SHIFT[e]
            qh[:, self.QH_IDX[e]] |= (f["q"][:, e] >> 4) << self.QH_SHIFT[e]
        b[:, 0:128], b[:, 128:192], b[:, 192:208] = ql, qh, f["sc"]
        _put16(b, 208, f["d"])
        return b

    def elements(self, f):
        return {"sc": f["sc"][:, self.SC_IDX], "q": f["q"]}

    def value(self, f, e):
        return f16(f["d"]) * e["sc"].astype(np.uint8).view(np.int8) * (e["q"] - 32)


class IQ4_XS(QType):
    name, ggml_id, bsize = "IQ4_XS", 23, 136
    fields = {"s": (8, 64), "q": (256, 16)}
    elem_codes = {"s": 64, "q": 16}
    arith = (("d", "s", "q"),)
    _e = np.arange(256)
    QS_IDX, QS_SHIFT = 16 * (_e // 32) + _e % 16, 4 * ((_e % 32) // 16)

    def __init__(self, kvalues=None):
        self.kv = np.array(kvalues if kvalues is not None else load_tables()["kvalues_iq4nl"],
                           dtype=np.float64)

    def unpack(self, b):
        sh, sl = _u16(b, 2).astype(np.int64), b[:, 4:8].astype(np.int64)
        ib = np.arange(8)
        s = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
        qs = b[:, 8:136].astype(np.int64)
        return {"d": _u16(b, 0)[:, None], "s": s, "q": (qs[:, self.QS_IDX] >> self.QS_SHIFT) & 0xF}

    def pack(self, f):
        n = len(f["q"]); b = self.empty(n); s = f["s"]
        _put16(b, 0, f["d"])
        _put16(b, 2, sum((s[:, ib] >> 4) << (2 * ib) for ib in range(8)))
        for i in range(4):
            b[:, 4 + i] = (s[:, 2 * i] & 0xF) | ((s[:, 2 * i + 1] & 0xF) << 4)
        q = f["q"]
        for ib in range(8):
            b[:, 8 + 16 * ib:24 + 16 * ib] = q[:, 32 * ib:32 * ib + 16] | (q[:, 32 * ib + 16:32 * ib + 32] << 4)
        return b

    def elements(self, f):
        return {"s": f["s"][:, np.arange(256) // 32], "q": f["q"]}

    def value(self, f, e):
        return f16(f["d"]) * (e["s"] - 32) * self.kv[e["q"]]


class _IQGrid(QType):
    """IQ2/IQ3 grid types. A sign group is 8 elements; `gw` is the grid entry width (8 for IQ2,
    4 for IQ3), `gps` the sign groups per sub-scale, `sign_bits` 7 (parity-completed via ksigns)
    or 8 (explicit). value = d * scale(s) * grid_byte * (+-1)."""
    table = ""; gw = 8; gps = 4; sign_bits = 7

    def __init__(self, grid=None):
        g = np.array(grid if grid is not None else load_tables()[self.table], dtype=np.uint64)
        self.grid = ((g[:, None] >> (np.uint64(8) * np.arange(self.gw, dtype=np.uint64)))
                     & np.uint64(0xFF)).astype(np.int64)
        self.mags = np.unique(self.grid)
        self.ngrid = len(g)
        self.nscale = 32 // self.gps
        self.fields = {"s": (self.nscale, 16), "grid": (256 // self.gw, self.ngrid),
                       "sign": (32, 1 << self.sign_bits)}
        self.elem_codes = {"s": 16, "mag": len(self.mags), "neg": 2}
        self.arith = (("d", "s", "mag", "neg"),)
        self.ks = ksigns()

    def scale(self, d, s):
        raise NotImplementedError

    def elements(self, f):
        n = len(f["grid"])
        mag = self.grid[f["grid"]].reshape(n, 256)
        sign8 = self.ks[f["sign"]] if self.sign_bits == 7 else f["sign"]
        return {"s": f["s"][:, np.arange(256) // (8 * self.gps)],
                "mag": np.searchsorted(self.mags, mag), "magv": mag, "neg": _bits8(sign8)}

    def value(self, f, e):
        return self.scale(f16(f["d"]), e["s"]) * e["magv"] * (1 - 2 * e["neg"])


class IQ2_XXS(_IQGrid):
    name, ggml_id, bsize, table, gw, gps, sign_bits = "IQ2_XXS", 16, 66, "iq2xxs_grid", 8, 4, 7

    def unpack(self, b):
        ib = np.arange(8)
        aux = np.stack([_u32(b, 2 + 8 * i + 4) for i in ib], axis=1).astype(np.int64)
        grid = np.stack([b[:, 2 + 8 * i + l] for i in ib for l in range(4)], axis=1).astype(np.int64)
        sign = np.stack([(aux[:, i] >> (7 * l)) & 127 for i in ib for l in range(4)], axis=1)
        return {"d": _u16(b, 0)[:, None], "s": aux >> 28, "grid": grid, "sign": sign}

    def pack(self, f):
        b = self.empty(len(f["grid"])); _put16(b, 0, f["d"])
        for i in range(8):
            for l in range(4):
                b[:, 2 + 8 * i + l] = f["grid"][:, 4 * i + l]
            _put32(b, 2 + 8 * i + 4, sum(f["sign"][:, 4 * i + l] << (7 * l) for l in range(4))
                   | (f["s"][:, i] << 28))
        return b

    def scale(self, d, s):
        return d * (0.5 + s) * 0.25


class IQ2_XS(_IQGrid):
    name, ggml_id, bsize, table, gw, gps, sign_bits = "IQ2_XS", 17, 74, "iq2xs_grid", 8, 2, 7

    def unpack(self, b):
        qs = np.stack([_u16(b, 2 + 2 * g) for g in range(32)], axis=1).astype(np.int64)
        sc = b[:, 66:74].astype(np.int64)
        s = np.stack([(sc[:, k // 2] >> (4 * (k % 2))) & 0xF for k in range(16)], axis=1)
        return {"d": _u16(b, 0)[:, None], "s": s, "grid": qs & 511, "sign": qs >> 9}

    def pack(self, f):
        b = self.empty(len(f["grid"])); _put16(b, 0, f["d"])
        for g in range(32):
            _put16(b, 2 + 2 * g, f["grid"][:, g] | (f["sign"][:, g] << 9))
        for i in range(8):
            b[:, 66 + i] = f["s"][:, 2 * i] | (f["s"][:, 2 * i + 1] << 4)
        return b

    def scale(self, d, s):
        return d * (0.5 + s) * 0.25


class IQ2_S(_IQGrid):
    name, ggml_id, bsize, table, gw, gps, sign_bits = "IQ2_S", 22, 82, "iq2s_grid", 8, 2, 8

    def unpack(self, b):
        g = np.arange(32)
        qh = b[:, 66:74].astype(np.int64)
        grid = b[:, 2:34].astype(np.int64) | (((qh[:, g // 4] >> (2 * (g % 4))) & 3) << 8)
        sc = b[:, 74:82].astype(np.int64)
        s = np.stack([(sc[:, k // 2] >> (4 * (k % 2))) & 0xF for k in range(16)], axis=1)
        return {"d": _u16(b, 0)[:, None], "s": s, "grid": grid, "sign": b[:, 34:66].astype(np.int64)}

    def pack(self, f):
        b = self.empty(len(f["grid"])); _put16(b, 0, f["d"])
        b[:, 2:34] = f["grid"] & 0xFF
        b[:, 34:66] = f["sign"]
        for i in range(8):
            b[:, 66 + i] = sum((f["grid"][:, 4 * i + l] >> 8) << (2 * l) for l in range(4))
            b[:, 74 + i] = f["s"][:, 2 * i] | (f["s"][:, 2 * i + 1] << 4)
        return b

    def scale(self, d, s):
        return d * (0.5 + s) * 0.25


class IQ3_XXS(_IQGrid):
    name, ggml_id, bsize, table, gw, gps, sign_bits = "IQ3_XXS", 18, 98, "iq3xxs_grid", 4, 4, 7

    def unpack(self, b):
        aux = np.stack([_u32(b, 66 + 4 * i) for i in range(8)], axis=1).astype(np.int64)
        sign = np.stack([(aux[:, i] >> (7 * l)) & 127 for i in range(8) for l in range(4)], axis=1)
        return {"d": _u16(b, 0)[:, None], "s": aux >> 28, "grid": b[:, 2:66].astype(np.int64),
                "sign": sign}

    def pack(self, f):
        b = self.empty(len(f["grid"])); _put16(b, 0, f["d"])
        b[:, 2:66] = f["grid"]
        for i in range(8):
            _put32(b, 66 + 4 * i, sum(f["sign"][:, 4 * i + l] << (7 * l) for l in range(4))
                   | (f["s"][:, i] << 28))
        return b

    def scale(self, d, s):
        return d * (0.5 + s) * 0.5


class IQ3_S(_IQGrid):
    name, ggml_id, bsize, table, gw, gps, sign_bits = "IQ3_S", 21, 110, "iq3s_grid", 4, 4, 8

    def unpack(self, b):
        t = np.arange(64)
        qh = b[:, 66:74].astype(np.int64)
        grid = b[:, 2:66].astype(np.int64) | (((qh[:, t // 8] >> (t % 8)) & 1) << 8)
        sc = b[:, 106:110].astype(np.int64)
        s = np.stack([(sc[:, k // 2] >> (4 * (k % 2))) & 0xF for k in range(8)], axis=1)
        return {"d": _u16(b, 0)[:, None], "s": s, "grid": grid, "sign": b[:, 74:106].astype(np.int64)}

    def pack(self, f):
        b = self.empty(len(f["grid"])); _put16(b, 0, f["d"])
        b[:, 2:66] = f["grid"] & 0xFF
        for k in range(8):
            b[:, 66 + k] = sum((f["grid"][:, 8 * k + i] >> 8) << i for i in range(8))
        b[:, 74:106] = f["sign"]
        for i in range(4):
            b[:, 106 + i] = f["s"][:, 2 * i] | (f["s"][:, 2 * i + 1] << 4)
        return b

    def scale(self, d, s):
        return d * (1 + 2 * s)


TYPE_CLASSES = {c.name: c for c in (Q4_K, Q6_K, Q8_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
                                    IQ4_XS, MXFP4)}
TYPES = tuple(TYPE_CLASSES)
GGML_IDS = {c.ggml_id: c.name for c in TYPE_CLASSES.values()}
#: byte offset of each fp16 super-block scale field, for the GGUF harvest
SCALE_OFFSETS = {"Q8_0": {"d": 0}, "Q4_K": {"d": 0, "dmin": 2}, "Q6_K": {"d": 208},
                 "IQ2_XXS": {"d": 0}, "IQ2_XS": {"d": 0}, "IQ2_S": {"d": 0}, "IQ3_XXS": {"d": 0},
                 "IQ3_S": {"d": 0}, "IQ4_XS": {"d": 0}, "MXFP4": {"e": 0}}


def qtype(name: str) -> QType:
    key = {n.upper(): n for n in TYPES}.get(name.upper())
    if key is None:
        raise ValueError(f"unsupported type {name}; supported: {', '.join(TYPES)}")
    return TYPE_CLASSES[key]()


# ----------------------------------------------------------------------------------------------
# enumerator

def _digits(npos: int, nc: int) -> int:
    d = 1
    while nc ** d < npos:
        d += 1
    return d


def _addressing_fields(qt: QType, n: int, d_bits: int = F16_ONE) -> dict[str, np.ndarray]:
    """Block k < n sets position p of every field to (k + p) % ncodes: across n = max(ncodes) blocks
    every (position, code) pair occurs. That shift pattern gives positions p and p + ncodes the same
    code in every block (Q4_K nibble l and l+32 would be indistinguishable), so `position digit`
    blocks follow: block i writes base-ncodes digit i of p, and any two positions of a field differ
    in at least one block -- a permutation of positions cannot decode identically."""
    k = np.arange(n)[:, None]
    f = {name: (k + np.arange(npos)[None, :]) % nc for name, (npos, nc) in qt.fields.items()}
    nd = max(_digits(npos, nc) for npos, nc in qt.fields.values())
    for name, (npos, nc) in qt.fields.items():
        p = np.arange(npos)[None, :]
        i = np.arange(nd)[:, None]
        digit = (p // (nc ** np.minimum(i, _digits(npos, nc) - 1))) % nc
        f[name] = np.concatenate([f[name], np.where(i < _digits(npos, nc), digit, (i + p) % nc)])
    n += nd
    for s in qt.scales:
        f[s] = np.full((n, 1), d_bits if s == "d" else F16_HALF, np.int64)
    return f


def _cover_grid(qt: _IQGrid) -> list[int]:
    """Greedy small set of grid indices whose bytes include every distinct magnitude."""
    need, chosen = set(qt.mags.tolist()), []
    while need:
        best = max(range(qt.ngrid), key=lambda i: len(need & set(qt.grid[i].tolist())))
        chosen.append(best); need -= set(qt.grid[best].tolist())
    return chosen


def _arith_template(qt: QType) -> dict[str, np.ndarray]:
    """One d's worth of blocks realising every (sub-scale code, element code) combination of the
    type's ARITHMETIC axes (fp16 scales left at the nominal 1.0 / 0.5; the tiler patches them)."""
    if isinstance(qt, Q8_0):
        f = {"q": (np.arange(8)[:, None] * 32 + np.arange(32)[None, :])}
    elif isinstance(qt, Q4_K):
        u = np.arange(8)[:, None] * 8 + np.arange(8)[None, :]
        f = {"sc": u, "m": 63 - u, "q": np.tile(np.arange(256) % 16, (8, 1))}
    elif isinstance(qt, IQ4_XS):
        f = {"s": np.arange(8)[:, None] * 8 + np.arange(8)[None, :],
             "q": np.tile(np.arange(256) % 16, (8, 1))}
    elif isinstance(qt, Q6_K):
        # global sub-block u = 0..1023: sc = u // 4, its 16 elements carry 16*(u%4) + rank
        q = np.zeros((64, 256), np.int64); sc = np.zeros((64, 16), np.int64)
        for u in range(1024):
            blk, sub = divmod(u, 16)
            sc[blk, sub] = u // 4
            idx = np.nonzero(qt.SC_IDX == sub)[0]
            q[blk, idx] = 16 * (u % 4) + np.arange(len(idx))
        f = {"sc": sc, "q": q}
    elif isinstance(qt, _IQGrid):
        cover = _cover_grid(qt)
        neg = (1 << qt.sign_bits) - 1     # all-negative sign code (ksigns[127] == 255)
        combos = [(g, sg) for g in cover for sg in (0, neg)]
        slots_per_s = -(-len(combos) // qt.gps)
        nb = -(-(16 * slots_per_s) // qt.nscale)
        s = np.zeros((nb, qt.nscale), np.int64)
        grid = np.zeros((nb, 256 // qt.gw), np.int64); sign = np.zeros((nb, 32), np.int64)
        per_group = 8 // qt.gw
        for slot in range(nb * qt.nscale):
            blk, k = divmod(slot, qt.nscale)
            sval, part = divmod(slot, slots_per_s)
            s[blk, k] = sval % 16
            for j in range(qt.gps):
                g = k * qt.gps + j
                gi, sg = combos[(part * qt.gps + j) % len(combos)]
                grid[blk, g * per_group:(g + 1) * per_group] = gi
                sign[blk, g] = sg
        f = {"s": s, "grid": grid, "sign": sign}
    else:
        raise ValueError(qt.name)
    n = len(next(iter(f.values())))
    for name in qt.fields:
        f.setdefault(name, np.zeros((n, qt.fields[name][0]), np.int64))
    for sc_name in qt.scales:
        f[sc_name] = np.full((n, 1), F16_ONE if sc_name == "d" else F16_HALF, np.int64)
    return f


def scale_rows(qt: QType, d_set: Iterable[int] | None, dmin_set: Iterable[int] | None):
    """The (d, dmin) pairs the arithmetic part walks: every d, every dmin, the shorter list cycled."""
    d_list = sorted(set(d_set)) if d_set else list(NOMINAL_D)
    if "dmin" not in qt.scales:
        return [(d, None) for d in d_list]
    m_list = sorted(set(dmin_set)) if dmin_set else list(NOMINAL_D)
    return [(d_list[r % len(d_list)], m_list[r % len(m_list)])
            for r in range(max(len(d_list), len(m_list)))]


def enumerate_blocks(qt: QType, d_set: Iterable[int] | None = None,
                     dmin_set: Iterable[int] | None = None, shard: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Addressing blocks (nominal d) + the arithmetic template tiled once per (d, dmin) row, with the
    fp16 scale bytes patched in place. `shard` = (i, n) keeps scale rows i, i+n, ... so a large
    harvested set can be split across files; every shard repeats the (small) addressing part and is
    complete on its own. MXFP4's E8M0 byte has only 256 values, so all of them are enumerated
    regardless of `d_set`."""
    i, n = shard
    n_addr = max(nc for _, nc in qt.fields.values())
    parts = [qt.pack(_addressing_fields(qt, n_addr))]
    if isinstance(qt, MXFP4):
        parts.append(qt.pack({"e": np.arange(256)[:, None], "q": np.tile(np.arange(32) % 16, (256, 1))}))
    else:
        rows = scale_rows(qt, d_set, dmin_set)[i::n]
        tmpl = qt.pack(_arith_template(qt))
        if rows:
            b = np.tile(tmpl, (len(rows), 1))
            offs = SCALE_OFFSETS[qt.name]
            for col, name in enumerate(("d", "dmin")[:len(qt.scales)]):
                v = np.repeat(np.array([r[col] for r in rows], np.int64), len(tmpl))
                b[:, offs[name]], b[:, offs[name] + 1] = v & 0xFF, (v >> 8) & 0xFF
            parts.append(b)
    return np.concatenate(parts)


def pad_to_rows(qt: QType, blocks: np.ndarray) -> np.ndarray:
    per_row = ROW // qt.qk
    extra = (-len(blocks)) % per_row
    return np.concatenate([blocks, np.repeat(blocks[:1], extra, axis=0)]) if extra else blocks


def estimate_bytes(qt: QType, n_d: int, n_dmin: int = 0) -> int:
    probe = _arith_template(qt) if not isinstance(qt, MXFP4) else None
    per = 0 if probe is None else len(probe["d"])
    rows = max(n_d, n_dmin) if "dmin" in qt.scales else n_d
    return (max(nc for _, nc in qt.fields.values()) + per * rows + (256 if probe is None else 0)) * qt.bsize


# ----------------------------------------------------------------------------------------------
# coverage

def _frac(reached: int, total: int) -> dict:
    return {"reached": int(reached), "total": int(total),
            "fraction": (reached / total) if total else 1.0}


def coverage(qt: QType, blocks: np.ndarray, scale_sets: dict[str, Iterable[int]] | None = None) -> dict:
    """Fraction of the code space `blocks` reach.

    addressing[field]: (position, code) pairs over npos x ncodes.
    arith[axis]: element-level code tuples. A block-level fp16 scale in the axis counts against
    `scale_sets[name]` when given (tuples whose scale is outside the set are ignored); without a set
    the scale is dropped from the axis and only its distinct-value count is reported."""
    blocks = qt.blocks(blocks)
    f = qt.unpack(blocks)
    out: dict = {"type": qt.name, "n_blocks": int(len(blocks)), "addressing": {}, "arith": {},
                 "scales_distinct": {s: int(len(np.unique(f[s]))) for s in (*qt.scales, *(
                     ("e",) if isinstance(qt, MXFP4) else ()))}}
    for name, (npos, nc) in qt.fields.items():
        pairs = np.unique((np.arange(npos)[None, :] * nc + f[name]).ravel())
        out["addressing"][name] = _frac(len(pairs), npos * nc)
    e = qt.elements(f)
    for axis in qt.arith:
        key = np.zeros((len(blocks), qt.qk), np.int64); total = 1; mask = np.ones_like(key, bool)
        used = []
        for comp in axis:
            if comp in qt.scales:
                if scale_sets is None or scale_sets.get(comp) is None:
                    continue
                vals = np.array(sorted(set(int(v) for v in scale_sets[comp])), np.int64)
                pos = np.searchsorted(vals, f[comp][:, 0])
                ok = (pos < len(vals)) & (vals[np.minimum(pos, len(vals) - 1)] == f[comp][:, 0])
                mask &= ok[:, None]
                key = key * len(vals) + np.minimum(pos, len(vals) - 1)[:, None]
                total *= len(vals)
            elif comp == "e":
                key = key * 256 + f["e"][:, 0:1]; total *= 256
            else:
                nc = qt.elem_codes[comp]
                key = key * nc + e[comp]; total *= nc
            used.append(comp)
        out["arith"]["x".join(used)] = _frac(len(np.unique(key[mask])), total)
    tot = sum(v["total"] for v in out["addressing"].values())
    out["addressing_fraction"] = sum(v["reached"] for v in out["addressing"].values()) / tot
    out["arith_fraction"] = min(v["fraction"] for v in out["arith"].values())
    return out


# ----------------------------------------------------------------------------------------------
# test-backend-ops random suite model

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def suite_seed_mix(v):
    """splitmix64 finaliser, identical to tests/test-backend-ops.cpp suite_seed_mix at C12."""
    v = np.asarray(v, dtype=np.uint64)
    with np.errstate(over="ignore"):
        v = v + np.uint64(0x9E3779B97F4A7C15)
        v = (v ^ (v >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        v = (v ^ (v >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return v ^ (v >> np.uint64(31))


def suite_uniform(seed: int, n: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """init_tensor_uniform's deterministic path: f32(lo + (hi-lo) * (mix(seed ^ mix(i)) >> 11) 2^-53)."""
    bits = suite_seed_mix(np.uint64(seed) ^ suite_seed_mix(np.arange(n, dtype=np.uint64)))
    unit = (bits >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
    return (lo + (hi - lo) * unit).astype(np.float32)


def _roundf(x: np.ndarray) -> np.ndarray:
    t = np.trunc(x.astype(np.float64))
    return t + np.where(np.abs(x - t) >= 0.5, np.sign(x), 0)


def quantize_ref(qt: QType, x: np.ndarray) -> np.ndarray:
    """Emulation of ggml's reference quantizer (Q8_0, MXFP4 only), float32 throughout."""
    x = np.asarray(x, np.float32).reshape(-1, qt.qk)
    amax = np.abs(x).max(axis=1)
    if isinstance(qt, Q8_0):
        d = (amax / np.float32(127)).astype(np.float32)
        with np.errstate(divide="ignore"):
            idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
        q = _roundf((x * idv[:, None]).astype(np.float32)).astype(np.int64) & 0xFF
        return qt.pack({"d": d.astype(np.float16).view(np.uint16).astype(np.int64)[:, None], "q": q})
    if isinstance(qt, MXFP4):
        with np.errstate(divide="ignore"):
            lg = np.floor(np.log2(amax.astype(np.float32)))
        e = np.where(amax > 0, lg - 2 + 127, 0).astype(np.int64) & 0xFF
        d = np.ldexp(np.float32(1), (e - 128).astype(np.int32)).astype(np.float32)
        cand = (qt.kv.astype(np.float32)[None, None, :] * d[:, None, None]).astype(np.float32)
        err = np.abs(cand - x[:, :, None]).astype(np.float32)
        return qt.pack({"e": e[:, None], "q": np.argmin(err, axis=2)})
    raise ValueError(f"no reference quantizer emulation for {qt.name}")


def random_suite_blocks(qt: QType, n_blocks: int, seed: int = 0) -> tuple[np.ndarray, str]:
    """Blocks the random suite would quantize. Exact for Q8_0/MXFP4 (uniform [-1,1] through the
    emulated reference quantizer); for every other type a RANDOM-CODE PROXY (uniform random bytes
    with d = 1/127-ish), which overstates reach because ggml's quantizers concentrate codes."""
    if isinstance(qt, (Q8_0, MXFP4)):
        return quantize_ref(qt, suite_uniform(seed, n_blocks * qt.qk)), "emulated_ref_quantizer"
    rng = np.random.default_rng(seed)
    f = qt.unpack(rng.integers(0, 256, size=(n_blocks, qt.bsize), dtype=np.uint8))
    for s in qt.scales:
        f[s] = np.full((n_blocks, 1), 0x2008, np.int64)   # fp16(1/127)
    return qt.pack(f), "random_code_proxy"


def random_suite_report(n_blocks: Sequence[int] = (1, 16, 256, 4096), seed: int = 0,
                        types: Sequence[str] = TYPES) -> dict:
    out = {}
    for name in types:
        qt = qtype(name)
        rows = []
        for n in n_blocks:
            b, model = random_suite_blocks(qt, n, seed)
            c = coverage(qt, b)
            rows.append({"n_blocks": n, "model": model, "addressing_fraction": c["addressing_fraction"],
                         "arith_fraction_without_d": c["arith_fraction"],
                         "fields": {k: v["fraction"] for k, v in c["addressing"].items()}})
        out[name] = rows
    return out


# ----------------------------------------------------------------------------------------------
# GGUF scanner (read-only, memory-mapped)

_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def take(self, fmt: str):
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def string(self) -> str:
        n = self.take("<Q")
        s = bytes(self.buf[self.pos:self.pos + n]); self.pos += n
        return s.decode("utf-8", "replace")

    def value(self, vt: int, keep_arrays: bool = False):
        if vt in _GGUF_SCALAR:
            return self.take(_GGUF_SCALAR[vt])
        if vt == 8:
            return self.string()
        if vt == 9:
            et, n = self.take("<I"), self.take("<Q")
            if et in _GGUF_SCALAR and not keep_arrays:
                self.pos += n * struct.calcsize(_GGUF_SCALAR[et])
                return {"array_of": et, "len": n}
            items = [self.value(et, keep_arrays) for _ in range(n)]
            return items if keep_arrays else {"array_of": et, "len": n}
        raise ValueError(f"unknown GGUF value type {vt}")


def read_gguf_header(path: str | Path) -> dict:
    """Parse a GGUF (v2/v3) header: metadata scalars, tensor infos and the data offset."""
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        r = _Reader(mm)
        if bytes(mm[0:4]) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        r.pos = 4
        version = r.take("<I")
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} unsupported")
        n_tensors, n_kv = r.take("<Q"), r.take("<Q")
        kv = {}
        for _ in range(n_kv):
            key = r.string(); kv[key] = r.value(r.take("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string(); nd = r.take("<I")
            dims = [r.take("<Q") for _ in range(nd)]
            tensors.append({"name": name, "dims": dims, "type": r.take("<I"), "offset": r.take("<Q")})
        align = int(kv.get("general.alignment", 32))
        data = r.pos + (-r.pos) % align
        return {"path": str(path), "version": version, "alignment": align, "data_offset": data,
                "kv": kv, "tensors": tensors, "file_size": len(mm)}


def harvest_scales(paths: Sequence[str | Path], chunk_blocks: int = 1 << 20,
                   max_blocks_per_tensor: int | None = None) -> dict:
    """Distinct fp16 d/dmin bit patterns (and MXFP4 E8M0 bytes) per served type. Reads only the
    scale bytes of each block through a memory map; whole-file page-in is the cost."""
    seen: dict[str, dict[str, np.ndarray]] = {}
    sources, counts = [], {}
    for p in paths:
        hdr = read_gguf_header(p)
        sources.append({"path": hdr["path"], "file_size": hdr["file_size"], "tensors": len(hdr["tensors"])})
        with open(p, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for t in hdr["tensors"]:
                name = GGML_IDS.get(t["type"])
                if name is None:
                    continue
                qt = TYPE_CLASSES[name]
                nel = int(np.prod(t["dims"], dtype=np.int64))
                nb = nel // qt.qk
                if max_blocks_per_tensor is not None:
                    nb = min(nb, max_blocks_per_tensor)
                base = hdr["data_offset"] + t["offset"]
                if base + nb * qt.bsize > len(mm):
                    raise ValueError(f"{p}: tensor {t['name']} runs past end of file")
                view = np.frombuffer(mm, dtype=np.uint8, count=nb * qt.bsize, offset=base).reshape(nb, qt.bsize)
                bm = seen.setdefault(name, {})
                for field, off in SCALE_OFFSETS[name].items():
                    width = 1 if field == "e" else 2
                    bitmap = bm.setdefault(field, np.zeros(256 if width == 1 else 65536, bool))
                    for i in range(0, nb, chunk_blocks):
                        raw = np.ascontiguousarray(view[i:i + chunk_blocks, off:off + width])
                        bitmap[raw.view("<u2").ravel() if width == 2 else raw.ravel()] = True
                c = counts.setdefault(name, {"tensors": 0, "blocks": 0})
                c["tensors"] += 1; c["blocks"] += nb
                del view
    return {"schema": SCHEMA + ".scales", "sources": sources,
            "types": {n: {**counts[n], **{f: np.nonzero(b)[0].tolist() for f, b in bm.items()}}
                      for n, bm in seen.items()}}


# ----------------------------------------------------------------------------------------------
# judge

def _hex32(v: float) -> str:
    return f"0x{np.float32(v).view(np.uint32):08x}"


#: MUL_MAT-identity leg error model (2026-11-17, from the first C12 row akx31-qdec-c12-gpu-20261117).
#: The leg is NOT a decode leg: the activation is re-quantized to q8 and the kernels scale integer dot
#: products by fp16 values, so bit-identity with onehot*ref is not a property any served kernel has.
#: A mismatch is BENIGN only when it falls in one of these classes; anything else stays a failure:
#:   fp16_scale_rounding  |got - onehot*ref| <= FP16_SCALE_REL * |onehot*ref| (the d*d8 / dmin*bsum products
#:                        are formed or stored in half precision: 2077 -> 2076 at ROCm0, 4.8e-4 rel for Q4_K dmin)
#:   ftz_subnormal_scale  got == 0 and the block scale is subnormal (fp16 d/dmin exponent field 0, MXFP4 e == 0):
#:                        the GPU kernels flush subnormal scales to zero
#:   int8_min_code        Q8_0 code -128: ggml's quantizer never emits it and the CPU int8 dot (abs/sign trick)
#:                        negates it; a served GGUF containing -128 would hit this, so it is counted separately
#:   onehot_overflow      onehot*ref exceeds float32 (MXFP4 e >= 246 times 127): an artefact of this probe
#: r3 amendments (2026-11-18, akx31-qdec-c12-gpu-20261117-r3):
#:   fp16_scale_rounding  is now relative to the element's TERM magnitude (onehot * magnitude, e.g.
#:                        |d*sc*q| + |dmin*m| for Q4_K), not to the result: cancellation between the two
#:                        Q4_K terms made a 2^-10 rounding of a term look like a large relative error
#:   ftz_output_subnormal got == 0 and the decoded value is float32-subnormal (0 < |ref| < FLT_MIN): the GPU
#:                        flushes the subnormal dequantized operand (MXFP4 e = 1) before the onehot scale
#:   fp16_ftz_operand     GPU leg only: got == 0 and |ref| < 2^-14 (fp16 min normal): the GPU forms the
#:                        dequantized operand in half precision with fp16 denormals flushed (MXFP4 e <= 113
#:                        at small codes, r3); bounded absolute error < 2^-14 * onehot per element
#: Declared OUT-OF-DOMAIN (not benign; the verdict says "out_of_domain", never "pass"):
#:   fp16_intermediate_overflow  GPU leg only: got is NaN/inf where the fp16 scale is 0x7BFF (65504, the
#:                        fp16 maximum) or |ref| or |onehot*ref| > 65504, i.e. a half-precision intermediate
#:                        overflowed; no served GGUF carries such a scale (harvested d / e are far below).
#:                        A non-finite element in the same 256-element MUL_MAT row as such an element is in
#:                        this class too (its inf partial poisons the row's other outputs via 0 * inf)
#: iqk_q6k_q8_0_requant (2026-11-24, ak-q6k-cpu-mulmat-20261124/FINDINGS.md; CPU leg "mul_mat:CPU", Q6_K only):
#:                        iqk requantises Q6_K weights to Q8_0_R8 when the activation has >= 64 columns (the probe's
#:                        B = onehot * I has 256), under ONE fp16 scale per 32-element block. A 32-block spans two Q6_K
#:                        sub-blocks (sc) so the low-sc half loses precision: upstream design, not a decode defect. With
#:                        M = max |sc*(q-32)| over the element's 32-block the int8 step is M/127, so an element is benign
#:                        iff |got - onehot*ref| <= onehot*|d|*M * (0.5 + 2^-10 * 32)/127 (half-step rounding plus the
#:                        fp16 rounding of the block scale). Worst case M = 4064, d = 1: 2032 + fp16 slack. The class is
#:                        offered only to the wide default-path leg: the <= 32-column leg, GGML_IQK_DEQUANT=0 and
#:                        GGML_IQK=0 legs are judged WITHOUT it (the tight model, zero tolerance beyond the other classes).
FP16_SCALE_REL = 2.0 ** -10
REQUANT_REL = (0.5 + FP16_SCALE_REL * 32) / 127.0
REQUANT_LEG = "mul_mat:CPU"
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14
IDENTITY_CLASSES = ("fp16_scale_rounding", "ftz_subnormal_scale", "ftz_output_subnormal", "int8_min_code",
                    "onehot_overflow", "fp16_ftz_operand", "iqk_q6k_q8_0_requant")
OUT_OF_DOMAIN_CLASSES = ("fp16_intermediate_overflow",)


QDEC_ROW = 256   # test-akx-qdec: elements per MUL_MAT row (QK=32 types carry 8 blocks per row)


def classify_identity(ref, got, magnitude, onehot, *, d=None, dmin=None, e=None, q8=None, gpu=False, row=None,
                      requant_scale=None):
    """Return (bad_mask, class_counts) for the MUL_MAT-identity leg. bad = mismatches in NO benign or
    out-of-domain class. class_counts also carries max_abs_err / max_rel_err (|got - onehot*ref|, absolute
    and relative to onehot*magnitude) over the finite mismatches and over the unexplained ones.
    requant_scale (CPU Q6_K wide leg only, else None): per element |d|*M, M = max |sc*(q-32)| over its 32-block;
    enables the iqk_q6k_q8_0_requant class (see above) and the max_requant_err_over_bound figure."""
    ref = np.asarray(ref, dtype=np.float32); got = np.asarray(got, dtype=np.float32)
    fin = np.isfinite(ref)
    exp = ref.astype(np.float64) * onehot
    g = got.astype(np.float64)
    mag = np.asarray(magnitude, dtype=np.float64) * onehot
    bound = IDENTITY_ULPS * 2.0 ** -24 * mag
    zeros = np.zeros(ref.shape, dtype=bool)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        err = np.abs(g - exp)
        mis = fin & ~(err <= bound)
        overflow = mis & (np.abs(exp) > np.finfo(np.float32).max) & ~np.isfinite(g)
        sub = zeros.copy()
        for c in (d, dmin):
            if c is not None:
                sub |= (np.asarray(c, dtype=np.int64) & 0x7C00) == 0
        if e is not None:
            sub |= np.asarray(e, dtype=np.int64) == 0
        ftz = mis & ~overflow & (g == 0) & (ref != 0) & sub
        ftz_out = mis & ~overflow & ~ftz & (g == 0) & (ref != 0) & (np.abs(ref) < np.finfo(np.float32).tiny)
        i8 = mis & ~overflow & ~ftz & ~ftz_out & (np.asarray(q8) == 128) if q8 is not None else zeros
        rnd = mis & ~overflow & ~ftz & ~ftz_out & ~i8 & np.isfinite(g) & (err <= FP16_SCALE_REL * mag)
        if gpu:
            big = (np.abs(ref.astype(np.float64)) > FP16_MAX) | (np.abs(exp) > FP16_MAX)
            for c in (d, dmin):
                if c is not None:
                    big |= (np.asarray(c, dtype=np.int64) & 0xFFFF) == 0x7BFF
            if row and ref.size % row == 0:
                # a NaN/inf partial of an overflowed element poisons the rest of its MUL_MAT row (0 * inf)
                big |= np.repeat((big & fin).reshape(-1, row).any(1), row)
            ood = mis & ~overflow & ~np.isfinite(g) & big
            ftz16 = mis & ~overflow & ~ftz & ~ftz_out & ~i8 & ~rnd & ~ood & (g == 0) & \
                (np.abs(ref.astype(np.float64)) < FP16_MIN_NORMAL)
        else:
            ood = ftz16 = zeros
        if requant_scale is not None and not gpu:
            rq_bound = onehot * np.asarray(requant_scale, dtype=np.float64) * REQUANT_REL
            rq = mis & ~overflow & ~ftz & ~ftz_out & ~i8 & ~rnd & np.isfinite(g) & (err <= rq_bound)
            rq_ratio = err / np.where(rq_bound > 0, rq_bound, np.nan)
        else:
            rq = zeros; rq_ratio = np.full(ref.shape, np.nan)
        bad = mis & ~(overflow | ftz | ftz_out | i8 | rnd | ood | ftz16 | rq)
        rel = err / np.where(mag > 0, mag, np.nan)

    def mx(a, m):
        v = a[m & np.isfinite(a)]
        return float(v.max()) if v.size else 0.0
    return bad, {"fp16_scale_rounding": int(rnd.sum()), "ftz_subnormal_scale": int(ftz.sum()),
                 "ftz_output_subnormal": int(ftz_out.sum()), "int8_min_code": int(i8.sum()),
                 "onehot_overflow": int(overflow.sum()), "fp16_ftz_operand": int(ftz16.sum()),
                 "fp16_intermediate_overflow": int(ood.sum()),
                 "iqk_q6k_q8_0_requant": int(rq.sum()),
                 "identity_mismatch_total": int(mis.sum()), "unexplained": int(bad.sum()),
                 "max_abs_err": mx(err, mis), "max_rel_err": mx(rel, mis),
                 "max_abs_err_unexplained": mx(err, bad), "max_rel_err_unexplained": mx(rel, bad),
                 "max_abs_err_requant": mx(err, rq), "max_requant_err_over_bound": mx(rq_ratio, mis)}


def q6k_block32_scale(qt: QType, f, e) -> np.ndarray:
    """Per element |d| * M with M = max |sc*(q-32)| over the element's 32-element block (natural element order; the unit
    iqk_convert_q6_k_q8_0_r8 requantises under one fp16 scale). Shape (n_blocks * 256,)."""
    n = len(f["q"])
    v = np.abs(e["sc"].astype(np.uint8).view(np.int8).astype(np.int64) * (e["q"] - 32)).reshape(n, 8, 32)
    m = np.repeat(v.max(axis=2), 32, axis=1)
    return (np.abs(f16(f["d"])).astype(np.float64) * m).ravel()


def _judge_chunk(qt: QType, blocks: np.ndarray, got: np.ndarray, mode: str, onehot: float, gpu: bool = False,
                 requant: bool = False):
    f = qt.unpack(blocks); e = qt.elements(f)
    with np.errstate(over="ignore", invalid="ignore"):
        ref = qt.value(f, e).astype(np.float32).ravel()
    rnan, gnan = np.isnan(ref), np.isnan(got)
    if mode == "exact":
        eq = ref.view(np.uint32) == got.view(np.uint32)
        bad = ~eq & ~(rnan & gnan)
        tiny = np.finfo(np.float32).tiny
        classes = {"nan_payload_differs": int((~eq & rnan & gnan).sum()),
                   "zero_sign": int((~eq & (ref == 0) & (got == 0)).sum()),
                   "ftz_subnormal": int((bad & (np.abs(ref) < tiny) & (ref != 0) & (got == 0)).sum())}
    elif mode == "identity":
        fin = np.isfinite(ref)
        n = len(blocks)
        bc = lambda k: np.broadcast_to(f[k], (n, qt.qk)).ravel() if k in f else None
        q8 = np.asarray(f["q"]).reshape(n, qt.qk).ravel() if qt.name == "Q8_0" else None   # raw byte; 128 == -128
        rqs = q6k_block32_scale(qt, f, e) if requant and qt.name == "Q6_K" else None
        bad, classes = classify_identity(ref, got, qt.magnitude(f, e).ravel(), onehot,
                                         d=bc("d"), dmin=bc("dmin"), e=bc("e"), q8=q8, gpu=gpu, row=QDEC_ROW,
                                         requant_scale=rqs)
        classes["nonfinite_reference_skipped"] = int((~fin).sum())
    else:
        raise ValueError(f"unknown judge mode {mode}")
    return f, e, ref, bad, classes


def judge(qt: QType, blocks, native: np.ndarray, leg: str, mode: str = "exact",
          onehot: float = ONEHOT, max_report: int = 16, chunk_blocks: int = 1 << 15) -> dict:
    """Compare native float32 output with the reference, chunk by chunk. mode="exact": bit-for-bit
    (to_float and GET_ROWS legs; NaN matches NaN, counted if the payload differs). mode="identity":
    MUL_MAT(W, onehot*I) leg: |native - onehot*ref| within IDENTITY_ULPS float32 ulps of the element's
    term magnitude, or a mismatch in one of the declared benign classes (classify_identity); `mismatches`
    counts only the unexplained ones, and every class is reported (activations are re-quantized to q8,
    so this leg is a path check, not a bit-exact one)."""
    blocks = qt.blocks(blocks)
    got = np.asarray(native, dtype=np.float32).ravel()
    n_el = len(blocks) * qt.qk
    res = {"schema": SCHEMA + ".verdict", "marker": MARKER, "c12": C12, "type": qt.name, "leg": leg,
           "mode": mode, "n_blocks": int(len(blocks)), "n_elements": int(n_el),
           "blocks_sha256": hashlib.sha256(blocks.tobytes()).hexdigest()}
    # the requant class belongs to the wide default-path CPU leg only (REQUANT_LEG); every other leg label (n32,
    # DEQUANT=0, IQK=0, ...) is judged under the tight model
    requant = mode == "identity" and leg == REQUANT_LEG and qt.name == "Q6_K"
    if mode == "identity":
        res.update(ulp_bound=IDENTITY_ULPS, onehot=onehot, requant_class=requant)
    if got.size != n_el:
        return {**res, "verdict": "invalid", "reason": f"native has {got.size} floats, expected {n_el}"}
    gpu = mode == "identity" and not leg.endswith(":CPU")
    total, classes, first, max_err = 0, {}, [], {}
    for c0 in range(0, len(blocks), chunk_blocks):
        cb = blocks[c0:c0 + chunk_blocks]
        f, e, ref, bad, cls = _judge_chunk(qt, cb, got[c0 * qt.qk:(c0 + len(cb)) * qt.qk], mode, onehot, gpu, requant)
        for k, v in cls.items():
            if k.startswith("max_"):
                max_err[k[4:]] = max(max_err.get(k[4:], 0.0), v)
            else:
                classes[k] = classes.get(k, 0) + v
        idx = np.nonzero(bad)[0]
        total += int(idx.size)
        for i in idx[:max(0, max_report - len(first))]:
            blk, el = divmod(int(i), qt.qk)
            codes = {k: int(v[blk, el]) for k, v in e.items() if v.ndim == 2 and v.shape[1] == qt.qk}
            codes.update({s: int(f[s][blk, 0]) for s in (*qt.scales, *(("e",) if "e" in f else ()))})
            g = got[c0 * qt.qk + int(i)]
            first.append({"block": c0 + blk, "element": el, "ref": _hex32(ref[i]), "got": _hex32(g),
                          "ref_value": float(ref[i]), "got_value": float(g), "codes": codes})
    ood = sum(classes.get(k, 0) for k in OUT_OF_DOMAIN_CLASSES)
    if mode == "identity":
        res["max_err"] = max_err
        res["out_of_domain"] = ood
    return {**res, "mismatches": total, "classes": classes, "first": first,
            "verdict": "fail" if total else ("out_of_domain" if ood else "pass")}


def receipts(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith(MARKER + " "):
            try:
                out.append(json.loads(line[len(MARKER) + 1:]))
            except json.JSONDecodeError:
                out.append({"status": "malformed", "line": line[:200]})
    return out


# ----------------------------------------------------------------------------------------------
# CLI

def _load_scale_sets(path: str | None, name: str) -> dict[str, list[int]]:
    if not path:
        return {}
    doc = json.loads(Path(path).read_text())
    t = doc.get("types", {}).get(name, {})
    return {k: v for k, v in t.items() if k in ("d", "dmin", "e")}


def _finite_f16(bits: Iterable[int]) -> list[int]:
    return [b for b in bits if (b & 0x7C00) != 0x7C00]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AKX-ALL-31.QDEC code-space dequant check")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tc = sub.add_parser("tables-check"); tc.add_argument("--repo", default="/mnt/raid0/llm/llama.cpp")
    hv = sub.add_parser("harvest"); hv.add_argument("--out", required=True); hv.add_argument("gguf", nargs="+")
    hv.add_argument("--max-blocks-per-tensor", type=int)
    en = sub.add_parser("enumerate"); en.add_argument("--type", required=True)
    en.add_argument("--scales"); en.add_argument("--all-finite-fp16", action="store_true")
    en.add_argument("--include-nonfinite", action="store_true"); en.add_argument("--out-dir")
    en.add_argument("--estimate-only", action="store_true"); en.add_argument("--shard", default="0/1")
    cv = sub.add_parser("coverage"); cv.add_argument("--type", required=True)
    cv.add_argument("--blocks", required=True); cv.add_argument("--scales")
    rs = sub.add_parser("random-suite"); rs.add_argument("--types", default=",".join(TYPES))
    rs.add_argument("--n-blocks", default="1,16,256,4096"); rs.add_argument("--seed", type=int, default=0)
    jd = sub.add_parser("judge"); jd.add_argument("--type", required=True)
    jd.add_argument("--blocks", required=True); jd.add_argument("--native", required=True)
    jd.add_argument("--leg", required=True); jd.add_argument("--mode", default="exact",
                                                           choices=("exact", "identity"))
    jd.add_argument("--onehot", type=float, default=ONEHOT); jd.add_argument("--out")
    a = ap.parse_args(argv)

    if a.cmd == "tables-check":
        src, emb = source_tables(a.repo), load_tables()
        ok = all(src[k] == emb[k] for k in EMBEDDED) and src["ksigns_iq2xs"] == ksigns().tolist()
        print(json.dumps({"tables_match_c12": ok, "commit": C12, "blob": COMMON_H_BLOB}))
        return 0 if ok else 1
    if a.cmd == "harvest":
        doc = harvest_scales(a.gguf, max_blocks_per_tensor=a.max_blocks_per_tensor)
        Path(a.out).write_text(json.dumps(doc) + "\n")
        print(json.dumps({t: {k: (len(v) if isinstance(v, list) else v) for k, v in d.items()}
                          for t, d in doc["types"].items()}))
        return 0
    if a.cmd == "enumerate":
        qt = qtype(a.type)
        sets = _load_scale_sets(a.scales, qt.name)
        if a.all_finite_fp16:
            sets = {s: list(range(65536)) for s in qt.scales}
        if not a.include_nonfinite:
            sets = {k: (_finite_f16(v) if k != "e" else v) for k, v in sets.items()}
        est = estimate_bytes(qt, len(sets.get("d", NOMINAL_D)), len(sets.get("dmin", NOMINAL_D)))
        if a.estimate_only:
            print(json.dumps({"type": qt.name, "estimated_bytes": est}))
            return 0
        if not a.out_dir:
            raise SystemExit("enumerate: --out-dir is required unless --estimate-only")
        shard = tuple(int(x) for x in a.shard.split("/"))
        if len(shard) != 2 or not 0 <= shard[0] < shard[1]:
            raise SystemExit("--shard must be i/n with 0 <= i < n")
        blocks = pad_to_rows(qt, enumerate_blocks(qt, sets.get("d"), sets.get("dmin"), shard))
        rows = scale_rows(qt, sets.get("d"), sets.get("dmin"))[shard[0]::shard[1]]
        own = {"d": sorted({r[0] for r in rows}), "dmin": sorted({r[1] for r in rows if r[1] is not None})}
        if len(blocks) <= FULL_COVERAGE_MAX_BLOCKS:
            cov = coverage(qt, blocks, {k: own[k] for k in qt.scales})
            scale_ok = True
        else:
            # structurally complete: the addressing part plus one template tile is measured, and
            # every scale row's value is checked to be present in the written scale bytes
            r0 = rows[0] if rows else (F16_ONE, F16_HALF)
            probe = enumerate_blocks(qt, [r0[0]], [r0[1] or F16_HALF])
            cov = coverage(qt, probe, {"d": [r0[0]], "dmin": [r0[1] or F16_HALF]})
            cov["structural"] = True
            written = {k: set(np.unique(_u16(blocks, off)).tolist()) for k, off in SCALE_OFFSETS[qt.name].items()}
            scale_ok = all(set(own[k]) <= written[k] for k in qt.scales)
        suffix = "" if shard == (0, 1) else f".shard{shard[0]}of{shard[1]}"
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        bpath = out / f"{qt.name}{suffix}.blocks"
        bpath.write_bytes(blocks.tobytes())
        man = {"schema": SCHEMA + ".manifest", "type": qt.name, "ggml_type_name": qt.name.lower()
               if not qt.name.endswith("_K") else qt.name[:-2].lower() + "_K",
               "c12": C12, "shard": list(shard), "n_blocks": int(len(blocks)), "block_bytes": qt.bsize,
               "blocks": str(bpath), "blocks_sha256": hashlib.sha256(blocks.tobytes()).hexdigest(),
               "scales_file": a.scales, "scale_rows": len(rows),
               "scale_counts": {k: len(own[k]) for k in qt.scales}, "scale_bytes_ok": scale_ok,
               "coverage": cov}
        (out / f"{qt.name}{suffix}.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        print(json.dumps({k: man[k] for k in ("type", "shard", "n_blocks", "blocks_sha256")}))
        return 0 if scale_ok and cov["addressing_fraction"] == 1.0 and cov["arith_fraction"] == 1.0 else 1
    if a.cmd == "coverage":
        qt = qtype(a.type)
        sets = _load_scale_sets(a.scales, qt.name) or None
        print(json.dumps(coverage(qt, Path(a.blocks).read_bytes(), sets), indent=1))
        return 0
    if a.cmd == "random-suite":
        print(json.dumps(random_suite_report([int(x) for x in a.n_blocks.split(",")], a.seed,
                                             [t for t in a.types.split(",") if t]), indent=1))
        return 0
    if a.cmd == "judge":
        qt = qtype(a.type)
        native = np.fromfile(a.native, dtype="<f4")
        res = judge(qt, Path(a.blocks).read_bytes(), native, a.leg, a.mode, a.onehot)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({k: res.get(k) for k in ("type", "leg", "mode", "verdict", "mismatches", "out_of_domain",
                                                  "max_err")}))
        return {"pass": 0, "out_of_domain": 3}.get(res["verdict"], 1)   # out-of-domain is never a silent pass
    return 2


if __name__ == "__main__":
    sys.exit(main())
