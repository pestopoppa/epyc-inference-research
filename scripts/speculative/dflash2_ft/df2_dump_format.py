#!/usr/bin/env python3
"""df2-feat-v1: the on-disk contract between df2_feature_dump (C++) and the DFlash2 trainer.

Layout per shard directory (all little-endian, row-major):
  tokens.i32     int32 [N]
  feats.bf16     bf16  [N, n_taps * n_embd]   taps = layer-INPUT ids (HF target_layer_ids + 1)
  final.bf16     bf16  [N, n_embd]            optional, post-norm final hidden (KL arm teacher)
  records.jsonl  {rec, id, tok_off, n_tok, loss_start, n_prompt, n_completion, mode, sampler}
  manifest.json  {format, n_embd, taps, has_final, n_tokens, n_records, prefix_window, feat_dtype, model[, sha256]}

Record r covers rows [tok_off, tok_off + n_tok). Rows [tok_off, tok_off + loss_start) are the context
window prefix (features only, no loss); rows from loss_start on are the completion (anchors + labels).

CLI:
  prep-requests <in.jsonl> <out.tsv>   JSONL {"id", "prompt_tokens": [...], ["completion_tokens": [...]]} -> dump input
  seal <shard_dir>                     add sha256 of every array to manifest.json
  check <shard_dir>                    validate sizes, offsets and sha256 (exit 1 on any defect)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

FORMAT = "df2-feat-v1"
ARRAYS = ("tokens.i32", "feats.bf16", "final.bf16")


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even, identical to f32_to_bf16() in df2_feature_dump.cpp."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = 0x7FC0
    return r


class Shard:
    def __init__(self, d: str):
        self.dir = d
        self.manifest = json.load(open(os.path.join(d, "manifest.json")))
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{d}: format {self.manifest.get('format')!r} != {FORMAT}")
        self.n_embd = int(self.manifest["n_embd"])
        self.taps = list(self.manifest["taps"])
        n = int(self.manifest["n_tokens"])
        w = len(self.taps) * self.n_embd
        self.tokens = np.memmap(os.path.join(d, "tokens.i32"), dtype=np.int32, mode="r", shape=(n,)) if n else np.zeros(0, np.int32)
        self.feats = np.memmap(os.path.join(d, "feats.bf16"), dtype=np.uint16, mode="r", shape=(n, w)) if n else np.zeros((0, w), np.uint16)
        self.final = None
        if self.manifest.get("has_final") and n:
            self.final = np.memmap(os.path.join(d, "final.bf16"), dtype=np.uint16, mode="r", shape=(n, self.n_embd))
        self.records = [json.loads(l) for l in open(os.path.join(d, "records.jsonl")) if l.strip()]

    def record(self, i: int) -> dict:
        r = self.records[i]
        s, e = r["tok_off"], r["tok_off"] + r["n_tok"]
        out = {"tokens": np.asarray(self.tokens[s:e]), "feats": bf16_to_f32(np.asarray(self.feats[s:e])),
               "loss_start": r["loss_start"], "id": r["id"]}
        if self.final is not None:
            out["final"] = bf16_to_f32(np.asarray(self.final[s:e]))
        return out


class ShardWriter:
    """Pure-Python writer with the same layout as the C++ tool (used by the smoke test and for re-sharding)."""

    def __init__(self, d: str, n_embd: int, taps: list[int], has_final: bool, prefix_window: int, model: str = "synthetic"):
        os.makedirs(d, exist_ok=True)
        self.d, self.n_embd, self.taps, self.has_final = d, n_embd, list(taps), has_final
        self.prefix_window, self.model = prefix_window, model
        self.ft = open(os.path.join(d, "tokens.i32"), "wb")
        self.ff = open(os.path.join(d, "feats.bf16"), "wb")
        self.fh = open(os.path.join(d, "final.bf16"), "wb") if has_final else None
        self.fr = open(os.path.join(d, "records.jsonl"), "w")
        self.n_tok = self.n_rec = 0

    def add(self, rid: str, tokens: np.ndarray, feats: np.ndarray, loss_start: int, final: np.ndarray | None = None):
        n = len(tokens)
        assert feats.shape == (n, len(self.taps) * self.n_embd), feats.shape
        np.asarray(tokens, dtype=np.int32).tofile(self.ft)
        f32_to_bf16(feats).tofile(self.ff)
        if self.has_final:
            assert final is not None and final.shape == (n, self.n_embd)
            f32_to_bf16(final).tofile(self.fh)
        self.fr.write(json.dumps({"rec": self.n_rec, "id": rid, "tok_off": self.n_tok, "n_tok": n, "loss_start": loss_start,
                                  "n_prompt": loss_start, "n_completion": n - loss_start, "mode": "synthetic", "sampler": {}}) + "\n")
        self.n_tok += n
        self.n_rec += 1

    def close(self):
        for f in (self.ft, self.ff, self.fh, self.fr):
            if f:
                f.close()
        json.dump({"format": FORMAT, "n_embd": self.n_embd, "taps": self.taps, "has_final": self.has_final,
                   "n_tokens": self.n_tok, "n_records": self.n_rec, "prefix_window": self.prefix_window,
                   "feat_dtype": "bf16", "model": self.model}, open(os.path.join(self.d, "manifest.json"), "w"))


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def seal(d: str) -> None:
    mp = os.path.join(d, "manifest.json")
    m = json.load(open(mp))
    m["sha256"] = {a: _sha(os.path.join(d, a)) for a in ARRAYS + ("records.jsonl",) if os.path.exists(os.path.join(d, a))}
    json.dump(m, open(mp, "w"), indent=1)


def check(d: str) -> list[str]:
    errs = []
    try:
        s = Shard(d)
    except Exception as e:  # noqa: BLE001
        return [f"open: {e}"]
    m, n = s.manifest, int(s.manifest["n_tokens"])
    w = len(s.taps) * s.n_embd
    sizes = {"tokens.i32": n * 4, "feats.bf16": n * w * 2}
    if m.get("has_final"):
        sizes["final.bf16"] = n * s.n_embd * 2
    for a, want in sizes.items():
        got = os.path.getsize(os.path.join(d, a))
        if got != want:
            errs.append(f"{a}: {got} B != {want} B")
    off = 0
    for r in s.records:
        if r["tok_off"] != off:
            errs.append(f"record {r['rec']}: tok_off {r['tok_off']} != {off}")
        if not 0 <= r["loss_start"] < r["n_tok"]:
            errs.append(f"record {r['rec']}: loss_start {r['loss_start']} outside [0, {r['n_tok']})")
        off = r["tok_off"] + r["n_tok"]
    if off != n:
        errs.append(f"records cover {off} rows, manifest says {n}")
    if len(s.records) != int(m["n_records"]):
        errs.append("n_records mismatch")
    for a, h in (m.get("sha256") or {}).items():
        if _sha(os.path.join(d, a)) != h:
            errs.append(f"{a}: sha256 mismatch")
    return errs


def prep_requests(src: str, dst: str) -> int:
    n = 0
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r["id"]).replace("\t", " ").replace('"', "'")
            cols = [rid, ",".join(map(str, r["prompt_tokens"]))]
            if r.get("completion_tokens"):
                cols.append(",".join(map(str, r["completion_tokens"])))
            fo.write("\t".join(cols) + "\n")
            n += 1
    return n


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "seal":
        seal(argv[1]); return 0
    if len(argv) >= 2 and argv[0] == "check":
        e = check(argv[1]); print("\n".join(e) or "ok"); return 1 if e else 0
    if len(argv) >= 3 and argv[0] == "prep-requests":
        print(prep_requests(argv[1], argv[2]), "requests"); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
