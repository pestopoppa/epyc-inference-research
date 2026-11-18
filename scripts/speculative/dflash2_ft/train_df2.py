#!/usr/bin/env python3
"""Offline-feature DFlash2 trainer (reference implementation and fallback for NeMo Automodel train_dflash2).

Reads df2-feat-v1 shards (df2_dump_format.py), warm-starts the drafter from an HF dir (gguf_to_hf.py output or
the z-lab bf16 repo), trains with the block-size curriculum and position-decayed loss from the YAML config, and
writes an HF dir (model.safetensors + config.json with dflash_config.block_size = train.block_size) that
export_gguf.sh converts with convert_hf_to_gguf.py.

The primary trainer named by DF2-FT-1 is NeMo Automodel train_dflash2 @1667095; its offline-feature dataset
adapter is what this script's `iter_batches` defines. This script is the CPU smoke path and the GPU fallback if
the Automodel recipe cannot read precomputed features on ROCm.

  train_df2.py --config train_block16.yaml [--override key=value ...] [--max-steps N] [--device cpu]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from df2_dump_format import Shard  # noqa: E402
from df2_model import DF2Config, DF2Drafter  # noqa: E402


def load_cfg(path: str, overrides: list[str]) -> dict:
    import yaml
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return cfg


def load_frozen(cfg: dict, c: DF2Config, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Target token_embd and output, frozen. Source: the target GGUF (dequantized) or synthetic for the smoke."""
    fz = cfg["frozen"]
    if fz.get("synthetic"):
        g = torch.Generator().manual_seed(int(fz.get("seed", 0)))
        e = torch.randn(c.vocab_size, c.hidden_size, generator=g) * 0.02
        o = torch.randn(c.vocab_size, c.hidden_size, generator=g) * 0.02
        return e.to(dtype), o.to(dtype)
    import gguf
    from gguf_to_hf import tensor_f32
    r = gguf.GGUFReader(fz["target_gguf"])
    by = {t.name: t for t in r.tensors}
    e = torch.from_numpy(tensor_f32(by["token_embd.weight"]).copy())
    o = torch.from_numpy(tensor_f32(by.get("output.weight", by["token_embd.weight"])).copy())
    return e.to(dtype), o.to(dtype)


def build_model(cfg: dict, device, dtype) -> DF2Drafter:
    m = cfg["model"]
    if m.get("synthetic_config"):
        c = DF2Config(**m["synthetic_config"])
    else:
        c = DF2Config.from_hf(json.load(open(os.path.join(m["init_dir"], "config.json"))))
    c.block_size = int(cfg["train"]["block_size"])
    emb, out = load_frozen(cfg, c, dtype)
    model = DF2Drafter(c, emb, out)
    if m.get("init_dir") and os.path.exists(os.path.join(m["init_dir"], "model.safetensors")):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(m["init_dir"], "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected or missing:
            raise SystemExit(f"warm-start mismatch: missing={missing} unexpected={unexpected}")
    elif m.get("random_init_seed") is not None:
        torch.manual_seed(int(m["random_init_seed"]))
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.02)
            elif "base_kernel" in n:
                torch.nn.init.zeros_(p)
                with torch.no_grad():
                    p.view(2, -1, p.shape[-1])[:, 0] = 1.0   # identity tap 0 so the conv starts as a pass-through
    return model.to(device=device, dtype=dtype)


def block_size_for_step(cfg: dict, step: int, total: int) -> int:
    cur = cfg["train"].get("block_curriculum") or []
    frac = step / max(1, total)
    for stage in cur:
        if frac < float(stage["until_frac"]):
            sizes, probs = stage["sizes"], stage["probs"]
            return int(random.choices(sizes, weights=probs, k=1)[0])
    return int(cfg["train"]["block_size"])


def iter_batches(shard_dirs: list[str], anchors_per_record: int, max_rows: int, seed: int):
    """Yield (feats, tokens, final|None, anchor_rows) per record; anchors sampled from the completion region.
    max_rows crops long records to the last max_rows rows (keeps loss_start semantics)."""
    rng = random.Random(seed)
    shards = [Shard(d) for d in shard_dirs]
    order = [(si, ri) for si, s in enumerate(shards) for ri in range(len(s.records))]
    while True:
        rng.shuffle(order)
        for si, ri in order:
            r = shards[si].record(ri)
            n = len(r["tokens"])
            s0 = max(0, n - max_rows)
            ls = max(r["loss_start"] - s0, 0)
            cand = list(range(max(ls, 1), n - s0 - 1))     # row indices in the CROPPED record
            if not cand:
                continue
            anchors = sorted(rng.sample(cand, min(anchors_per_record, len(cand))))
            yield (torch.from_numpy(r["feats"][s0:]), torch.from_numpy(r["tokens"][s0:].astype(np.int64)),
                   torch.from_numpy(r["final"][s0:]) if "final" in r else None, torch.tensor(anchors))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config, a.override)
    tr = cfg["train"]
    device = torch.device(a.device or tr.get("device", "cuda"))
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[tr.get("dtype", "bf16")]
    random.seed(tr["seed"]); torch.manual_seed(tr["seed"])

    model = build_model(cfg, device, dtype)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tr["lr"]), betas=tuple(tr.get("betas", [0.9, 0.95])),
                            weight_decay=float(tr.get("weight_decay", 0.0)))
    steps = int(a.max_steps or tr["steps"])
    total = int(tr.get("total_steps") or steps)      # global schedule length across rolling shards
    warm = int(tr.get("warmup_steps", 0))
    start = 0
    res = tr.get("resume_from")                      # rolling shards: previous run's output dir
    if res:
        st0 = json.load(open(os.path.join(res, "state.json")))
        opt.load_state_dict(torch.load(os.path.join(res, "optimizer.pt"), map_location=device))
        start = int(st0["global_step"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + start + 1) / max(1, warm)) *
                                              (0.5 * (1 + math.cos(math.pi * min(1.0, (s + start) / max(1, total))))
                                               if tr.get("cosine", True) else 1.0))
    shard_dirs = sorted(glob.glob(os.path.join(cfg["data"]["root"], cfg["data"].get("glob", "shard-*"))))
    if not shard_dirs:
        raise SystemExit(f"no shards under {cfg['data']['root']}")
    it = iter_batches(shard_dirs, int(tr["anchors_per_record"]), int(tr["max_rows"]), int(tr["seed"]))
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "a")
    accum = int(tr.get("grad_accum", 1))
    model.train()
    t0 = time.time()
    for step in range(steps):
        B = block_size_for_step(cfg, start + step, total)
        opt.zero_grad(set_to_none=True)
        stats_acc = {}
        for _ in range(accum):
            feats, toks, final, anchors = next(it)
            gamma = float(tr["loss_gamma"]) * (B / int(tr["block_size"]) if tr.get("gamma_scales_with_block") else 1.0)
            loss, st = model.loss(feats.to(device, dtype), toks.to(device), anchors.to(device), B, gamma=gamma,
                                  final=None if final is None else final.to(device, dtype),
                                  objective=tr["objective"], selector_weight=float(tr["selector_weight"]))
            (loss / accum).backward()
            for k, v in st.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v / accum
        gn = torch.nn.utils.clip_grad_norm_(params, float(tr.get("grad_clip", 1.0)))
        opt.step(); sched.step()
        rec = {"step": start + step, "B": B, "loss": float(loss), "grad_norm": float(gn), "lr": sched.get_last_lr()[0],
               "elapsed_s": round(time.time() - t0, 2), **stats_acc}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if not math.isfinite(rec["loss"]) or not math.isfinite(rec["grad_norm"]):
            raise SystemExit(f"non-finite at step {step}: {rec}")
        if step % int(tr.get("log_every", 50)) == 0 or step == steps - 1:
            print(json.dumps(rec), flush=True)
    save_hf(model, cfg, out_dir)
    torch.save(opt.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    json.dump({"global_step": start + steps, "shards": shard_dirs}, open(os.path.join(out_dir, "state.json"), "w"))
    return 0


def save_hf(model: DF2Drafter, cfg: dict, out_dir: str) -> None:
    from safetensors.torch import save_file
    c = model.config
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()
          if k not in ("tok_embd", "lm_head")}
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    init_cfg = os.path.join(cfg["model"].get("init_dir") or "", "config.json")
    hf = json.load(open(init_cfg)) if os.path.exists(init_cfg) else {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size, "num_hidden_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads, "num_key_value_heads": c.num_key_value_heads,
        "head_dim": c.head_dim, "rms_norm_eps": c.rms_norm_eps, "rope_theta": c.rope_theta,
        "vocab_size": c.vocab_size, "target_hidden_size": c.target_hidden_size, "has_embed_tokens": False,
        "dflash_config": {}}
    hf["dflash_config"].update(block_size=c.block_size, mask_token_id=c.mask_token_id,
                               target_layer_ids=c.target_layer_ids, conv_kernel_size=c.conv_kernel_size,
                               conv_group_size=c.conv_group_size, selector_rank=c.selector_rank,
                               selector_top_k=c.selector_top_k)
    json.dump(hf, open(os.path.join(out_dir, "config.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
