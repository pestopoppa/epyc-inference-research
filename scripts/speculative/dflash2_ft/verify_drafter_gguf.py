#!/usr/bin/env python3
"""Gate a DFlash2 drafter GGUF before it is served: metadata the C14 runtime reads, and tensor-set parity
with the incumbent drafter (same names, shapes; only dflash.block_size may differ).

  verify_drafter_gguf.py <cand.gguf> --block-size 16 [--reference incumbent.gguf] [--json]

C14 reads dflash.block_size (common/speculative.cpp:1410-1419, default 16 if ABSENT -- so absence is a
defect here, not a default) and clamps n_max to block_size-1 (:1457-1462).
"""
from __future__ import annotations

import argparse
import json
import sys

REQUIRED = ["dflash.block_size", "dflash.selector_top_k", "dflash.selector_rank", "dflash.conv_kernel_size",
            "dflash.conv_group_size", "dflash.target_layers", "tokenizer.ggml.mask_token_id"]


def meta(r) -> dict:
    out = {}
    for k, f in r.fields.items():
        if k.startswith("tokenizer.ggml.") and k != "tokenizer.ggml.mask_token_id":
            continue
        try:
            out[k] = f.contents()
        except Exception:  # noqa: BLE001
            pass
    return out


def check(cand: str, block_size: int, reference: str | None) -> list[str]:
    import gguf
    r = gguf.GGUFReader(cand)
    m = meta(r)
    errs = []
    if m.get("general.architecture") != "dflash":
        errs.append(f"architecture {m.get('general.architecture')!r} != 'dflash'")
    for k in REQUIRED:
        if k not in m:
            errs.append(f"missing {k}")
    if int(m.get("dflash.block_size", -1)) != block_size:
        errs.append(f"dflash.block_size {m.get('dflash.block_size')} != {block_size}")
    if reference:
        rr = gguf.GGUFReader(reference)
        rm = meta(rr)
        for k in REQUIRED[1:]:
            if rm.get(k) != m.get(k):
                errs.append(f"{k}: cand {m.get(k)} != reference {rm.get(k)}")
        cs = {t.name: list(t.shape) for t in r.tensors}
        rs = {t.name: list(t.shape) for t in rr.tensors}
        if set(cs) != set(rs):
            errs.append(f"tensor set differs: +{sorted(set(cs) - set(rs))} -{sorted(set(rs) - set(cs))}")
        for n in set(cs) & set(rs):
            if cs[n] != rs[n]:
                errs.append(f"{n}: shape {cs[n]} != {rs[n]}")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--reference")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    errs = check(a.gguf, a.block_size, a.reference)
    print(json.dumps({"gguf": a.gguf, "ok": not errs, "errors": errs}) if a.json else ("\n".join(errs) or "ok"))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
