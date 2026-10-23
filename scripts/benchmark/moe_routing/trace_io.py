"""Numpy reader for epyc.moe.routing_trace.v1 shards (INF-72 RT-1 output) plus shared grouping helpers.

Layout (written by llama-moe-trace, mirrored by scripts/kernel_rnd/moe_routing_tap/trace_shape.py):
  routing_trace.json       manifest
  routing_trace.rows.bin   16-byte rows: u32 sample_id, u32 pos, u32 step, u16 spec_block, u8 phase, u8 spec_accepted
  routing_trace.ids.i16    int16 [rows, n_layers, max_top_k], -1 pad
  routing_trace.provenance.json  carries first_window (bulk shards number their samples from 0 again)

CPU analysis only: nothing here loads a model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_ID = "epyc.moe.routing_trace.v1"
ROW_DTYPE = np.dtype([("sample_id", "<u4"), ("pos", "<u4"), ("step", "<u4"),
                      ("spec_block", "<u2"), ("phase", "u1"), ("acc", "u1")])
assert ROW_DTYPE.itemsize == 16
PREFILL, DECODE = 0, 1


@dataclass
class Trace:
    domain: str
    arch: str
    n_experts: np.ndarray      # [L]
    top_k: np.ndarray          # [L]
    win: np.ndarray            # [R] global window id (unique per independent sequence)
    pos: np.ndarray            # [R]
    step: np.ndarray           # [R]
    phase: np.ndarray          # [R] 0 prefill, 1 decode
    ids: np.ndarray            # [R, L, K] int16, -1 pad
    shards: list

    @property
    def R(self) -> int:
        return int(self.ids.shape[0])

    @property
    def L(self) -> int:
        return int(self.ids.shape[1])

    def select(self, mask: np.ndarray) -> "Trace":
        return Trace(self.domain, self.arch, self.n_experts, self.top_k, self.win[mask], self.pos[mask],
                     self.step[mask], self.phase[mask], self.ids[mask], self.shards)

    def layer_ids(self, l: int) -> np.ndarray:
        """Contiguous [R, K_l] ids of layer l (pads dropped by width)."""
        return np.ascontiguousarray(self.ids[:, l, : int(self.top_k[l])])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_shard(run_dir, verify: bool = False) -> Trace:
    d = Path(run_dir)
    m = json.loads((d / "routing_trace.json").read_text())
    if m.get("schema") != SCHEMA_ID:
        raise ValueError(f"{d}: schema {m.get('schema')!r}")
    prov_p = d / "routing_trace.provenance.json"
    prov = json.loads(prov_p.read_text()) if prov_p.exists() else {}
    if verify:
        if sha256_file(d / "routing_trace.rows.bin") != m["rows_sha256"]:
            raise ValueError(f"{d}: rows sha256 mismatch")
        if sha256_file(d / "routing_trace.ids.i16") != m["ids"]["sha256"]:
            raise ValueError(f"{d}: ids sha256 mismatch")
    rows = np.fromfile(d / "routing_trace.rows.bin", dtype=ROW_DTYPE)
    layers = m["layers"]
    nl, mk = len(layers), max(l["top_k"] for l in layers)
    flat = np.fromfile(d / "routing_trace.ids.i16", dtype="<i2")
    if flat.size != rows.size * nl * mk:
        raise ValueError(f"{d}: ids hold {flat.size} int16, expected {rows.size * nl * mk}")
    ids = flat.reshape(rows.size, nl, mk)
    first = int(prov.get("first_window", 0))
    dom = prov.get("domain") or (m["samples"][0].get("domain") if m.get("samples") else "?")
    return Trace(dom, m["arch"], np.array([l["n_experts"] for l in layers]), np.array([l["top_k"] for l in layers]),
                 rows["sample_id"].astype(np.int64) + first, rows["pos"].astype(np.int64), rows["step"].astype(np.int64),
                 rows["phase"].astype(np.int8), ids, [str(d)])


def concat(traces: list) -> Trace:
    t0 = traces[0]
    for t in traces[1:]:
        if t.arch != t0.arch or not np.array_equal(t.n_experts, t0.n_experts) or not np.array_equal(t.top_k, t0.top_k):
            raise ValueError("cannot concatenate traces of different geometry")
    return Trace(t0.domain, t0.arch, t0.n_experts, t0.top_k, np.concatenate([t.win for t in traces]),
                 np.concatenate([t.pos for t in traces]), np.concatenate([t.step for t in traces]),
                 np.concatenate([t.phase for t in traces]), np.concatenate([t.ids for t in traces]),
                 sum((t.shards for t in traces), []))


def discover(results_dir) -> dict:
    """{domain: {"decode": [dir...], "bulk": [dir...]}} for every shard dir that holds a manifest and is not quarantined."""
    root, out = Path(results_dir), {}
    for man in sorted(root.glob("*/*/shard-*/routing_trace.json")):
        shard = man.parent
        if (shard / "QUARANTINED").exists():
            continue
        out.setdefault(shard.parent.parent.name, {}).setdefault(shard.parent.name, []).append(str(shard))
    return out


def load_domain(results_dir, domain: str, parts=("decode", "bulk"), verify: bool = False) -> Trace:
    found = discover(results_dir).get(domain, {})
    dirs = [d for p in parts for d in found.get(p, [])]
    if not dirs:
        raise FileNotFoundError(f"no shards for domain {domain!r} parts {parts} under {results_dir}")
    t = concat([load_shard(d, verify) for d in dirs])
    t.domain = domain
    return t


def blocks(t: Trace, T: int):
    """Group rows into steps of T consecutive tokens of one window (a speculative block / decode step / prefill chunk).
    Rows must already be filtered to one phase. Returns (block_id [R], n_blocks, window_of_block [n_blocks]); rows in an
    incomplete tail block get block id -1. Block ids are dense over complete blocks, in row order."""
    R = t.R
    if R == 0:
        return np.zeros(0, np.int64), 0, np.zeros(0, np.int64)
    new_win = np.empty(R, bool)
    new_win[0] = True
    new_win[1:] = t.win[1:] != t.win[:-1]
    start_idx = np.maximum.accumulate(np.where(new_win, np.arange(R), 0))
    cc = np.arange(R) - start_idx
    new_blk = new_win | (cc % T == 0)
    bid = np.cumsum(new_blk) - 1
    size = np.bincount(bid)
    ok = size[bid] == T
    dense = np.cumsum(size == T) - 1
    out = np.where(ok, dense[bid], -1)
    nb = int((size == T).sum())
    win_of = np.zeros(nb, np.int64)
    if nb:
        win_of[out[ok]] = t.win[ok]
    return out, nb, win_of


def layer_loads(ids_l: np.ndarray, block: np.ndarray, nb: int, E: int) -> np.ndarray:
    """[nb, E] selection counts per block for one layer (rows with block<0 dropped)."""
    m = block >= 0
    b = np.broadcast_to(block[m][:, None], (int(m.sum()), ids_l.shape[1]))
    e = ids_l[m].astype(np.int64)
    ok = e >= 0
    flat = b[ok] * E + e[ok]
    return np.bincount(flat, minlength=nb * E).reshape(nb, E)


def layer_counts(ids_l: np.ndarray, E: int) -> np.ndarray:
    e = ids_l.ravel().astype(np.int64)
    return np.bincount(e[e >= 0], minlength=E)
