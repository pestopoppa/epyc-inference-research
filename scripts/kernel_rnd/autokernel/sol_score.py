#!/usr/bin/env python3
"""AKX-SOLSCORE-1: descriptive, bounded gap-closed column for keep-gate reports.

    S = 1 / (1 + (Tk - Tsol) / (Tb - Tsol))        0 < S <= 1, S = 1/2 at the baseline, 1 at SOL

* ``Tb``   per-token time of the champion ANCHOR arm (1 / anchor tok/s).
* ``Tk``   per-token time of the keep candidate (1 / candidate tok/s).
* ``Tsol`` speed-of-light per-token time: weight bytes read per token (from GGUF per-tensor
           bytes; routed-expert tensors count only ``n_expert_used / n_expert`` of their
           bytes; ``token_embd`` counts one row) over a MEASURED achievable bandwidth.

DESCRIPTIVE ONLY. The column never enters a promotion number, a keep decision or a gate; it is
computed from receipts already recorded. Weights-only bytes make Tsol a lower bound on the
true floor (no KV/activation traffic) and, for speculative serving, the per-token Tsol ignores
accepted-draft amortisation, so ``Tk < Tsol`` is possible: that is reported as ``BOUND-AUDIT``
(the bound, the bandwidth or the byte count is wrong - never a "better than SOL" claim) and S
is withheld. The Tb <= Tsol case (baseline already at/below the bound) is ``BOUND-AUDIT`` too.

Usage::

    sol_score.py retro [--campaigns /mnt/raid0/llm/autokernel/campaigns] [--census CENSUS.json]
                       [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.sol_score.v1"
OK, BOUND_AUDIT, INDETERMINATE = "OK", "BOUND-AUDIT", "INDETERMINATE"
DEFAULT_CENSUS = Path(__file__).resolve().parent / "loop" / "census" / "mi210_bandwidth_census.json"

# ggml type id -> (elements per block, bytes per block)
GGML_BLOCK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
    21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
    27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
    39: (32, 17), 40: (64, 36),
}


class GgufError(ValueError):
    pass


# ------------------------------------------------------------------ GGUF header reader
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
           11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def take(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GgufError("truncated GGUF header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > 1 << 24:
            raise GgufError("implausible string length")
        return self.take(n).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype in _SCALAR:
            fmt = _SCALAR[vtype]
            return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, n = self.u32(), self.u64()
            if n > 1 << 28:
                raise GgufError("implausible array length")
            if etype in _SCALAR:  # skip scalar arrays in one read
                self.take(struct.calcsize(_SCALAR[etype]) * n)
                return None
            return [self.value(etype) for _ in range(n)] if etype == 8 else None
        raise GgufError(f"unknown GGUF value type {vtype}")


def read_gguf_header(path: str | Path) -> dict[str, Any]:
    """``{"metadata": {...scalar/str...}, "tensors": [(name, dims, ggml_type)]}``; header only."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if r.take(4) != b"GGUF":
            raise GgufError("not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise GgufError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.u64(), r.u64()
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            val = r.value(r.u32())
            if val is not None:
                meta[key] = val
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = tuple(r.u64() for _ in range(nd))
            ttype = r.u32()
            r.u64()  # data offset
            tensors.append((name, dims, ttype))
    return {"metadata": meta, "tensors": tensors}


def read_gguf_header_sharded(path: str | Path) -> dict[str, Any]:
    """Like ``read_gguf_header`` but merges ``-0000K-of-0000N.gguf`` shards (metadata from the
    first shard that has any, tensors from all)."""
    import re
    path = Path(path)
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if not m:
        return read_gguf_header(path)
    total = int(m.group(3))
    merged: dict[str, Any] = {"metadata": {}, "tensors": []}
    for i in range(1, total + 1):
        shard = path.with_name(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf")
        h = read_gguf_header(shard)
        for k, v in h["metadata"].items():
            merged["metadata"].setdefault(k, v)
        merged["tensors"].extend(h["tensors"])
    return merged


def tensor_bytes(dims: Sequence[int], ggml_type: int) -> int:
    if ggml_type not in GGML_BLOCK:
        raise GgufError(f"unknown ggml type {ggml_type}")
    elems, size = GGML_BLOCK[ggml_type]
    n = math.prod(dims) if dims else 1
    if n % elems:
        raise GgufError("tensor element count not a multiple of the block size")
    return n // elems * size


def _meta_suffix(meta: Mapping[str, Any], *suffixes: str) -> int | None:
    vals = [v for k, v in meta.items() if k.endswith(suffixes) and isinstance(v, int)]
    return vals[0] if len(set(vals)) == 1 else None


def active_bytes_per_token(header: Mapping[str, Any]) -> dict[str, Any]:
    """Weight bytes read per decoded token. Routed-expert tensors (``*_exps``) are scaled by
    ``expert_used_count / expert_count``; lookup tables (``token_embd``, ``engram_embd``) count one row per token (a lower bound); everything else full."""
    meta = header["metadata"]
    n_exp = _meta_suffix(meta, ".expert_count", ".n_routed_experts", ".num_local_experts")
    n_used = _meta_suffix(meta, ".expert_used_count", ".num_experts_per_tok", ".n_expert_used")
    total = active = 0.0
    moe_scaled = 0
    for name, dims, ttype in header["tensors"]:
        b = tensor_bytes(dims, ttype)
        total += b
        if "_exps" in name and n_exp and n_used and len(dims) >= 3 and dims[-1] == n_exp:
            active += b * n_used / n_exp
            moe_scaled += 1
        elif ("embd" in name.split(".")[0] or "embd" in name or "engram" in name) \
                and len(dims) >= 2 and name.endswith(".weight") and "norm" not in name:
            active += b / dims[-1]
        else:
            active += b
    return {"total_bytes": int(total), "active_bytes": active, "expert_count": n_exp,
            "expert_used_count": n_used, "moe_tensors_scaled": moe_scaled}


# ------------------------------------------------------------------------------ S column
def tsol_seconds(active_bytes: float, bandwidth_gbps: float) -> float:
    if not (active_bytes > 0 and bandwidth_gbps > 0):
        raise ValueError("active_bytes and bandwidth_gbps must be positive")
    return active_bytes / (bandwidth_gbps * 1e9)


def gap_closed(tk: float, tb: float, tsol: float) -> dict[str, Any]:
    """Descriptive S for one (Tk, Tb, Tsol); never raises on physical nonsense, it audits."""
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (tk, tb, tsol)):
        return {"status": INDETERMINATE, "S": None, "reason": "non-positive or non-finite time"}
    if tb <= tsol:
        return {"status": BOUND_AUDIT, "S": None,
                "reason": "baseline anchor at or below the speed-of-light bound (Tb <= Tsol)"}
    if tk < tsol:
        return {"status": BOUND_AUDIT, "S": None,
                "reason": "candidate faster than the bound (Tk < Tsol): bound, bandwidth or bytes wrong"}
    return {"status": OK, "S": 1.0 / (1.0 + (tk - tsol) / (tb - tsol)), "reason": None}


def residency_tag(*, ngl: int | None, device_none: bool, vram_bytes: int | None,
                  resident_floor_bytes: int = 1 << 30) -> str:
    """Per-graph residency tag: GPU-RESIDENT, CPU-RESIDENT, HYBRID or UNKNOWN."""
    if ngl is None:
        return "UNKNOWN"
    if device_none or ngl == 0:
        return "HYBRID" if (vram_bytes or 0) > resident_floor_bytes else "CPU-RESIDENT"
    return "GPU-RESIDENT" if (vram_bytes or 0) > resident_floor_bytes else "HYBRID"


def with_sol_column(report_row: Mapping[str, Any], *, tsol: float | None, residency: str,
                    ) -> dict[str, Any]:
    """Add the descriptive column to a keep-gate report row carrying ``anchor_tok_s`` and
    ``candidate_tok_s``. The input row is not modified and no existing field is overwritten."""
    row = dict(report_row)
    a, c = row.get("anchor_tok_s"), row.get("candidate_tok_s")
    if tsol is None or not (a and c):
        res = {"status": INDETERMINATE, "S": None, "reason": "missing tok/s or Tsol"}
    else:
        res = gap_closed(1.0 / c, 1.0 / a, tsol)
    row["sol_gap_closed"] = {"schema": SCHEMA, "authority": "descriptive_only",
                             "promotion_input": False, "residency": residency,
                             "tsol_s": tsol, **res}
    return row


# ------------------------------------------------------------------------ retro ledger
def rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def load_bandwidths(census_path: str | Path = DEFAULT_CENSUS) -> dict[str, float]:
    """Measured achievable GB/s from the (read-only) bandwidth census."""
    c = json.loads(Path(census_path).read_text())
    return {"gpu": float(c["ceiling_gbps"]), "cpu-48t": float(c["cpu"]["ceiling_gbps"]["half-48t"]),
            "cpu-24t": float(c["cpu"]["ceiling_gbps"]["quarter-24t"])}


def _flag(argv: Sequence[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def kept_rows(db_path: str | Path) -> Iterable[dict[str, Any]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for attempt_id, eff, payload in con.execute(
                "select attempt_id, effect_fraction, payload from experiments where status='kept'"):
            yield {"attempt_id": attempt_id, "effect_fraction": eff, "payload": json.loads(payload)}
    finally:
        con.close()


def score_kept_row(row: Mapping[str, Any], bw: Mapping[str, float],
                   header_cache: dict[str, Any],
                   proxy_models: Mapping[str, str] | None = None) -> dict[str, Any]:
    p = row["payload"]
    cmp = p.get("comparison") or {}
    arm = ((p.get("research_scope") or {}).get("recipe") or {}).get("original_serving_arms", {}).get("anchor", {})
    argv = arm.get("argv") or []
    model, backend = _flag(argv, "-m"), arm.get("backend")
    threads = int(_flag(argv, "-t") or 0)
    ngl = _flag(argv, "-ngl")
    vram = (cmp.get("residency") or {}).get("median_vram_bytes")
    out = {"attempt_id": row["attempt_id"], "backend": backend, "model": model,
           "effect_pct": (row["effect_fraction"] or 0) * 100.0,
           "noise_floor_pct": cmp.get("noise_floor_pct"), "anchor_tok_s": cmp.get("anchor_tok_s"),
           "candidate_tok_s": cmp.get("candidate_tok_s"),
           "residency": residency_tag(ngl=None if ngl is None else int(ngl),
                                      device_none=_flag(argv, "--device") == "none", vram_bytes=vram)}
    try:
        source = model
        if proxy_models and model in proxy_models and not Path(model).exists():
            source = proxy_models[model]
            out["bytes_source"] = f"proxy_gguf:{source}"  # the anchor GGUF no longer exists
        else:
            out["bytes_source"] = "anchor_gguf"
        if source not in header_cache:
            header_cache[source] = active_bytes_per_token(read_gguf_header_sharded(source))
        hb = header_cache[source]
        key = "gpu" if backend and backend != "cpu" else ("cpu-48t" if threads >= 48 else "cpu-24t")
        tsol = tsol_seconds(hb["active_bytes"], bw[key])
        out.update(active_bytes=hb["active_bytes"], bandwidth_gbps=bw[key], bandwidth_key=key)
    except (OSError, GgufError, KeyError, TypeError, ValueError) as exc:
        tsol = None
        out["tsol_error"] = f"{type(exc).__name__}: {exc}"
    return with_sol_column(out, tsol=tsol, residency=out["residency"])


def retro(campaigns: str | Path = "/mnt/raid0/llm/autokernel/campaigns",
          census: str | Path = DEFAULT_CENSUS,
          proxy_models: Mapping[str, str] | None = None) -> dict[str, Any]:
    bw = load_bandwidths(census)
    cache: dict[str, Any] = {}
    rows, devices = [], {}
    for db in sorted(Path(campaigns).glob("*/store/experiments.db")):
        try:
            for r in kept_rows(db):
                s = score_kept_row(r, bw, cache, proxy_models)
                s["campaign"] = db.parent.parent.name
                rows.append(s)
        except sqlite3.Error:
            continue
    for r in rows:
        dev = "gpu" if r["backend"] and r["backend"] != "cpu" else "cpu"
        devices[dev] = devices.get(dev, 0) + 1
    scored = [r for r in rows if r["sol_gap_closed"]["status"] == OK]
    corr = {}
    if len(scored) >= 3:
        s_vals = [r["sol_gap_closed"]["S"] for r in scored]
        corr["spearman_S_vs_effect_pct"] = spearman(s_vals, [r["effect_pct"] for r in scored])
        nf = [(r["sol_gap_closed"]["S"], r["effect_pct"] / r["noise_floor_pct"]) for r in scored
              if r.get("noise_floor_pct")]
        if len(nf) >= 3:
            corr["spearman_S_vs_effect_over_noise_floor"] = spearman(*zip(*nf))
    return {"schema": SCHEMA, "authority": "descriptive_only", "promotion_input": False,
            "keep_rows": len(rows), "by_device": devices,
            "status_counts": {k: sum(r["sol_gap_closed"]["status"] == k for r in rows)
                              for k in (OK, BOUND_AUDIT, INDETERMINATE)},
            "n_scored": len(scored), "rank_correlation": corr, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retro")
    r.add_argument("--campaigns", default="/mnt/raid0/llm/autokernel/campaigns")
    r.add_argument("--census", default=str(DEFAULT_CENSUS))
    r.add_argument("--json", type=Path)
    r.add_argument("--proxy-model", action="append", default=[], metavar="OLD=NEW",
                   help="use NEW GGUF bytes when the anchor GGUF OLD no longer exists (tagged)")
    a = ap.parse_args(argv)
    doc = retro(a.campaigns, a.census, dict(x.split("=", 1) for x in a.proxy_model))
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
