"""RT-2.1 trace-shape validator and [steps, layers, experts] count derivation (pure stdlib, no inference)."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from array import array
from collections import defaultdict
from pathlib import Path

SCHEMA_ID = "epyc.moe.routing_trace.v1"
COLUMNS = ["sample_id", "pos", "phase", "step", "spec_block", "spec_accepted"]
INT16_MAX = 32767
SCHEMA_PATH = Path(__file__).with_name("routing_trace.schema.json")
# RT-1 on-disk layout written by llama-moe-trace (llama.cpp tools/moe-trace/moe-trace-core.h), little endian:
#   routing_trace.rows.bin  16-byte rows: u32 sample_id, u32 pos, u32 step, u16 spec_block, u8 phase (0 prefill, 1 decode),
#                           u8 spec_accepted (0 false, 1 true, 255 null)
#   routing_trace.ids.i16   int16 [rows][n_layers][max(top_k)], layers with a smaller top_k pad with -1
ROW_STRUCT = struct.Struct("<IIIHBB")
PHASES = ("prefill", "decode")
ACC_NULL = 255


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate_manifest(m: dict) -> list[str]:
    """Structural checks that do not need the row data. Returns problems (empty = ok)."""
    p: list[str] = []
    if m.get("schema") != SCHEMA_ID:
        p.append("schema id")
    layers = m.get("layers") or []
    if not layers:
        p.append("no layers")
    if [l.get("layer") for l in layers] != list(range(len(layers))):
        p.append("layers must be listed in order 0..n-1")
    for l in layers:
        if l.get("id_dtype") != "int16":
            p.append(f"layer {l.get('layer')}: id_dtype must be int16, not {l.get('id_dtype')!r}")
        if not 1 <= l.get("top_k", 0) <= l.get("n_experts", 0):
            p.append(f"layer {l.get('layer')}: top_k outside 1..n_experts")
        if l.get("n_experts", 0) > INT16_MAX:
            p.append(f"layer {l.get('layer')}: n_experts exceeds int16")
    if m.get("columns") != COLUMNS:
        p.append("columns must be exactly " + ",".join(COLUMNS))
    if (m.get("ids") or {}).get("dtype") != "int16":
        p.append("ids.dtype must be int16")
    s = m.get("samples") or []
    if not s:
        p.append("no samples")
    nxt = 0
    for x in sorted(s, key=lambda y: y.get("start_row", 0)):
        if x.get("start_row") != nxt:
            p.append(f"sample {x.get('sample_id')}: rows not contiguous at {nxt}")
        nxt = x.get("start_row", 0) + x.get("n_rows", 0)
    return p


def validate_rows(m: dict, rows: list[list], ids: list[list[list[int]]]) -> list[str]:
    """rows: one list per token in COLUMNS order. ids: [rows][n_layers][top_k], -1 pads short top_k."""
    p = validate_manifest(m)
    layers = m["layers"]
    total = sum(x["n_rows"] for x in m["samples"])
    if len(rows) != total or len(ids) != total:
        return p + [f"row count {len(rows)}/{len(ids)} != samples total {total}"]
    for s in m["samples"]:
        seg = rows[s["start_row"]:s["start_row"] + s["n_rows"]]
        last_pos, last_phase, forwards = -1, "prefill", defaultdict(list)
        for r in seg:
            sid, pos, phase, step, blk, acc = r
            if sid != s["sample_id"]:
                p.append(f"sample {s['sample_id']}: foreign row {sid} inside its boundary")
            if pos <= last_pos:
                p.append(f"sample {sid}: token order broken at pos {pos}")
            if phase not in ("prefill", "decode"):
                p.append(f"sample {sid} pos {pos}: phase {phase!r}")
            if last_phase == "decode" and phase == "prefill":
                p.append(f"sample {sid} pos {pos}: prefill after decode")
            if blk < 1 or (phase == "prefill" and acc is not None) or (acc is not None and blk == 1):
                p.append(f"sample {sid} pos {pos}: bad spec_block/spec_accepted")
            last_pos, last_phase = pos, phase
            forwards[step].append(r)
        for step, grp in forwards.items():
            if {g[2] for g in grp} != {grp[0][2]}:
                p.append(f"sample {s['sample_id']} step {step}: mixed phases in one forward")
            if any(g[4] != len(grp) for g in grp):
                p.append(f"sample {s['sample_id']} step {step}: spec_block != rows in the forward")
    for i, rowids in enumerate(ids):
        if len(rowids) != len(layers):
            p.append(f"row {i}: {len(rowids)} layers != {len(layers)}")
            continue
        for l, sel in zip(layers, rowids):
            real = [e for e in sel if e != -1]
            if len(real) != l["top_k"] or len(set(real)) != len(real):
                p.append(f"row {i} layer {l['layer']}: need {l['top_k']} distinct experts")
            if any(not 0 <= e < l["n_experts"] for e in real):
                p.append(f"row {i} layer {l['layer']}: expert id outside 0..{l['n_experts'] - 1}")
    return p


def derive_counts(m: dict, rows: list[list], ids, phase: str | None = None) -> dict:
    """[steps, layers, experts] selection counts: {(sample_id, step): [layer][expert] -> count}.
    A step is one forward (a prefill chunk, a decode token, or a speculative verify block); `phase` filters."""
    out: dict = {}
    for r, rowids in zip(rows, ids):
        if phase and r[2] != phase:
            continue
        cell = out.setdefault((r[0], r[3]), [[0] * l["n_experts"] for l in m["layers"]])
        for li, sel in enumerate(rowids):
            for e in sel:
                if e != -1:
                    cell[li][e] += 1
    return out


def load_run(run_dir, verify: bool = True) -> tuple[dict, list[list], list[list[list[int]]]]:
    """Read one llama-moe-trace shard directory -> (manifest, rows, ids) in the shapes validate_rows / derive_counts take.

    Raises ValueError on a structurally invalid manifest, a sha256 that does not match the manifest (verify=True),
    or file sizes that disagree with the sample boundaries. A directory without routing_trace.json is not a trace
    (an interrupted or failed capture has none): FileNotFoundError. Stdlib only; nested lists are for validation and
    modest shards - bulk analysis should use the numpy reader in the tool (tools/moe-trace/moe_trace_convert.py load_run).
    """
    d = Path(run_dir)
    m = json.loads((d / "routing_trace.json").read_text())
    probs = validate_manifest(m)
    if probs:
        raise ValueError("invalid manifest: " + "; ".join(probs))
    rows_b = (d / "routing_trace.rows.bin").read_bytes()
    ids_b = (d / "routing_trace.ids.i16").read_bytes()
    if verify:
        if hashlib.sha256(rows_b).hexdigest() != m["rows_sha256"]:
            raise ValueError("rows.bin sha256 does not match the manifest")
        if hashlib.sha256(ids_b).hexdigest() != m["ids"]["sha256"]:
            raise ValueError("ids.i16 sha256 does not match the manifest")
    if len(rows_b) % ROW_STRUCT.size:
        raise ValueError(f"rows.bin size {len(rows_b)} is not a multiple of {ROW_STRUCT.size}")
    n = len(rows_b) // ROW_STRUCT.size
    total = sum(x["n_rows"] for x in m["samples"])
    if n != total:
        raise ValueError(f"rows.bin holds {n} rows, the samples declare {total}")
    nl = len(m["layers"])
    mk = max(l["top_k"] for l in m["layers"])
    if len(ids_b) != n * nl * mk * 2:
        raise ValueError(f"ids.i16 size {len(ids_b)} != rows {n} x layers {nl} x max_top_k {mk} x 2")
    rows: list[list] = []
    for sid, pos, step, blk, ph, acc in ROW_STRUCT.iter_unpack(rows_b):
        if ph > 1 or acc not in (0, 1, ACC_NULL):
            raise ValueError(f"bad phase/spec_accepted byte at row {len(rows)}")
        rows.append([sid, pos, PHASES[ph], step, blk, None if acc == ACC_NULL else bool(acc)])
    flat = array("h")
    flat.frombytes(ids_b)
    if sys.byteorder == "big":
        flat.byteswap()
    ids = [[list(flat[(r * nl + l) * mk:(r * nl + l) * mk + mk]) for l in range(nl)] for r in range(n)]
    return m, rows, ids
