#!/usr/bin/env python3
"""CPU-only smoke test for the DF2 block-16 pipeline: dump format -> trainer step(s) -> HF save -> (optional)
GGUF export round-trip. Tiny synthetic model; no GPU, no network, no installs, no servers.

  <venv>/bin/python smoke_test.py --work <scratch dir> [--roundtrip]

Needs torch, numpy, safetensors, pyyaml (present in /mnt/raid0/llm/venvs/llama-gguf-convert). --roundtrip also
needs gguf + transformers and a `git archive` of the C14 converter (made under --work from the read-only
experimental repo) and the target tokenizer dir; it does NOT need NeMo Automodel.
Exit 0 only if every check passes; prints one JSON line per check.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import df2_dump_format as fmt  # noqa: E402
from df2_model import DF2Config, DF2Drafter  # noqa: E402

TINY = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-6, rope_theta=1e7, sliding_window=32, target_hidden_size=64,  # = hidden: drafter shares the target lm_head
            target_layer_ids=[1, 3, 5], block_size=16, mask_token_id=511, conv_kernel_size=2, conv_group_size=16,
            selector_rank=8, selector_top_k=4, vocab_size=512)
RESULTS = []


def ok(name, cond, **kw):
    RESULTS.append(bool(cond))
    print(json.dumps({"check": name, "pass": bool(cond), **kw}), flush=True)


def make_dump(root: str, n_records=6, seed=0) -> str:
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "dump", "shard-0000")
    taps = [t + 1 for t in TINY["target_layer_ids"]]
    w = fmt.ShardWriter(d, TINY["target_hidden_size"], taps, has_final=True, prefix_window=24)
    for i in range(n_records):
        n_prefix, n_comp = int(rng.integers(8, 24)), int(rng.integers(20, 40))
        n = n_prefix + n_comp
        toks = rng.integers(0, 500, n)
        feats = rng.standard_normal((n, len(taps) * TINY["target_hidden_size"])).astype(np.float32)
        final = rng.standard_normal((n, TINY["target_hidden_size"])).astype(np.float32)
        w.add(f"req{i}", toks, feats, n_prefix, final)
    w.close()
    return d


def check_conv_against_ggml_loop():
    """build_dflash2_conv re-derived with explicit loops (ggml ne order) vs the vectorised DF2Conv."""
    from df2_model import DF2Conv
    c = DF2Config(**TINY)
    torch.manual_seed(1)
    conv = DF2Conv(c)
    torch.nn.init.normal_(conv.base_kernel, std=0.5)
    A, B, H = 2, 5, c.hidden_size
    x = torch.randn(A, B, H)
    co = conv.coeffs(torch.randn(A, B, H))
    got = conv.conv(x, co, 1)
    ref = torch.zeros_like(x)
    G, K = H // c.conv_group_size, c.conv_kernel_size
    for a in range(A):
        for b in range(B):
            for tap in range(K):
                if b - tap < 0:
                    continue
                for ch in range(H):
                    wgt = co[a, b, 1, tap, ch // c.conv_group_size] + conv.base_kernel[1, tap, ch]
                    ref[a, b, ch] += wgt * x[a, b - tap, ch]
    ok("conv_matches_ggml_loop", torch.allclose(got, ref, atol=1e-5), max_abs=float((got - ref).abs().max()), groups=G)


def check_block_agnostic():
    c = DF2Config(**TINY)
    torch.manual_seed(0)
    m = DF2Drafter(c, torch.randn(c.vocab_size, c.hidden_size) * 0.02, torch.randn(c.vocab_size, c.hidden_size) * 0.02)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    n_params = sum(p.numel() for p in m.parameters())
    N = 40
    feats = torch.randn(N, len(c.target_layer_ids) * c.target_hidden_size)
    toks = torch.randint(0, 500, (N,))
    anchors = torch.tensor([10, 20])
    h8 = m(feats, toks, anchors, 8)
    h16 = m(feats, toks, anchors, 16)
    ok("block_size_agnostic_shapes", h8.shape == (2, 8, c.hidden_size) and h16.shape == (2, 16, c.hidden_size),
       n_params_same_for_8_and_16=n_params)
    # slot 0..7 outputs change when 8 more MASK rows join the non-causal block: the reason for the curriculum
    ok("noncausal_coupling_visible", not torch.allclose(h8, h16[:, :8], atol=1e-6),
       mean_abs_diff=float((h8 - h16[:, :8]).abs().mean()))
    # context mask: a block never sees context rows at or after its anchor, nor beyond the sliding window
    mask = DF2Drafter.build_mask(N, anchors, 4, c.sliding_window)
    ok("mask_causal_context", (not mask[0, 10:N].any()) and bool(mask[0, :10].all()) and (not mask[4, :20 - 32 if 20 - 32 > 0 else 0].any()))


def run_train(work: str, steps: int) -> str:
    import yaml
    out = os.path.join(work, "train-out")
    cfg = {"model": {"init_dir": None, "synthetic_config": TINY, "random_init_seed": 7},
           "frozen": {"synthetic": True, "seed": 3},
           "data": {"root": os.path.join(work, "dump"), "glob": "shard-*"},
           "train": {"device": "cpu", "dtype": "f32", "seed": 1, "block_size": 16,
                     "block_curriculum": [{"until_frac": 0.5, "sizes": [8, 16], "probs": [0.5, 0.5]}],
                     "loss_gamma": 7.0, "gamma_scales_with_block": True, "objective": "ce", "selector_weight": 0.1,
                     "lr": 1e-3, "warmup_steps": 1, "cosine": True, "betas": [0.9, 0.95], "weight_decay": 0.0,
                     "grad_clip": 1.0, "anchors_per_record": 6, "max_rows": 48, "grad_accum": 2, "steps": steps,
                     "log_every": 1},
           "output": {"dir": out}}
    p = os.path.join(work, "smoke.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    import train_df2
    rc = train_df2.main(["--config", p])
    log = [json.loads(l) for l in open(os.path.join(out, "train_log.jsonl"))]
    ok("train_steps_finite", rc == 0 and len(log) == steps and all(np.isfinite(r["loss"]) for r in log),
       first=log[0], last=log[-1])
    ok("curriculum_used_both_sizes", {r["B"] for r in log} >= {8, 16} or steps < 4, sizes=sorted({r["B"] for r in log}))
    hf = json.load(open(os.path.join(out, "config.json")))
    ok("saved_hf_block16", hf["dflash_config"]["block_size"] == 16 and os.path.getsize(os.path.join(out, "model.safetensors")) > 0)
    # KL arm wiring on the same dump (final.bf16 present)
    rc2 = train_df2.main(["--config", p, "--override", "train.objective=kl", "--override", f"output.dir={out}-kl",
                          "--max-steps", "1"])
    ok("kl_arm_one_step", rc2 == 0)
    # rolling-shard resume: warm-start weights + optimizer state + global step from the previous run
    rc3 = train_df2.main(["--config", p, "--override", f"model.init_dir={out}", "--override", f"train.resume_from={out}",
                          "--override", f"train.total_steps={steps + 2}", "--override", f"output.dir={out}-resume",
                          "--max-steps", "2"])
    st = json.load(open(os.path.join(out + "-resume", "state.json")))
    ok("rolling_resume", rc3 == 0 and st["global_step"] == steps + 2, global_step=st["global_step"])
    return out


def roundtrip(work: str, hf_out: str) -> None:
    """HF (trained, block 16) -> C14 convert_hf_to_gguf.py -> GGUF -> gguf_to_hf -> tensors equal; verify_drafter_gguf."""
    conv = os.path.join(work, "c14-convert")
    if not os.path.exists(os.path.join(conv, "convert_hf_to_gguf.py")):
        os.makedirs(conv, exist_ok=True)
        a = subprocess.run(["git", "-C", "/mnt/raid0/llm/llama.cpp-experimental", "archive", "eee3da838",
                            "convert_hf_to_gguf.py", "conversion", "gguf-py"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", conv], input=a.stdout, check=True)
    tgt_src = "/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw"
    tgt = os.path.join(work, "fake-target")
    os.makedirs(tgt, exist_ok=True)
    for f in os.listdir(tgt_src):
        if f.endswith((".json", ".txt", ".jinja", ".model")) and f != "config.json":
            shutil.copy(os.path.join(tgt_src, f), tgt)
    tc = json.load(open(os.path.join(tgt_src, "config.json")))
    tc["hidden_size"] = TINY["target_hidden_size"]
    if "text_config" in tc:
        tc["text_config"]["hidden_size"] = TINY["target_hidden_size"]
    json.dump(tc, open(os.path.join(tgt, "config.json"), "w"))
    hf = json.load(open(os.path.join(hf_out, "config.json")))
    hf.update(architectures=["DFlash2DraftModel"], model_type="qwen3", max_position_embeddings=4096,
              use_sliding_window=True, sliding_window=TINY["sliding_window"],
              layer_types=["sliding_attention"] * TINY["num_hidden_layers"], tie_word_embeddings=False)
    # the converter writes the TARGET tokenizer and asserts vocab_size covers it; the tiny codebooks stay 512 rows
    hf["vocab_size"] = (tc.get("text_config") or tc).get("vocab_size", 248320)
    json.dump(hf, open(os.path.join(hf_out, "config.json"), "w"))
    g1 = os.path.join(work, "rt-1.gguf")
    env = dict(os.environ, PYTHONPATH=os.path.join(conv, "gguf-py"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run([sys.executable, os.path.join(conv, "convert_hf_to_gguf.py"), hf_out, "--target-model-dir", tgt,
                        "--outtype", "f32", "--outfile", g1], capture_output=True, text=True, env=env)
    ok("c14_converter_accepts_block16", r.returncode == 0, tail=r.stderr[-600:] if r.returncode else "")
    if r.returncode:
        return
    sys.path.insert(0, os.path.join(conv, "gguf-py"))
    import gguf_to_hf
    import verify_drafter_gguf
    hf2 = os.path.join(work, "rt-hf")
    gguf_to_hf.main([g1, hf2])
    from safetensors.torch import load_file
    a, b = load_file(os.path.join(hf_out, "model.safetensors")), load_file(os.path.join(hf2, "model.safetensors"))
    same = set(a) == set(b) and all(torch.equal(a[k].float(), b[k].float()) for k in a)
    ok("hf_gguf_hf_roundtrip_exact", same, n_tensors=len(a), missing=sorted(set(a) ^ set(b))[:6])
    c2 = json.load(open(os.path.join(hf2, "config.json")))
    ok("roundtrip_config", c2["dflash_config"]["block_size"] == 16 and
       c2["dflash_config"]["target_layer_ids"] == TINY["target_layer_ids"], dflash_config=c2["dflash_config"])
    errs = verify_drafter_gguf.check(g1, 16, None)
    ok("verify_drafter_gguf_block16", not errs, errors=errs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--roundtrip", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.work):
        shutil.rmtree(a.work)
    os.makedirs(a.work)
    torch.set_num_threads(4)
    d = make_dump(a.work)
    ok("dump_check_unsealed", fmt.check(d) == [])
    fmt.seal(d)
    ok("dump_check_sealed", fmt.check(d) == [])
    s = fmt.Shard(d)
    r0 = s.record(0)
    ok("dump_record_shapes", r0["feats"].shape[1] == 3 * TINY["target_hidden_size"] and r0["final"].shape[1] == TINY["hidden_size"])
    x = np.array([1.0, -2.5, 3.14159, 1e-8, 65504.0, 1e30], np.float32)
    ok("bf16_roundtrip_rel_err", np.allclose(fmt.bf16_to_f32(fmt.f32_to_bf16(x)), x, rtol=8e-3))
    with open(os.path.join(d, "tokens.i32"), "r+b") as f:
        f.write(b"\xff\xff\xff\xff")
    ok("dump_tamper_detected", any("sha256" in e for e in fmt.check(d)))
    fmt.seal(d)
    check_conv_against_ggml_loop()
    check_block_agnostic()
    out = run_train(a.work, a.steps)
    if a.roundtrip:
        roundtrip(a.work, out)
    print(json.dumps({"smoke": "PASS" if all(RESULTS) else "FAIL", "checks": len(RESULTS), "failed": RESULTS.count(False)}))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
