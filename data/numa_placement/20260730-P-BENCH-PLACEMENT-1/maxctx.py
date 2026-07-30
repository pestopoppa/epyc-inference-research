"""Maximum context tokens a single instance can hold, per NUMA shape.

Pure arithmetic from GGUF geometry + measured per-node RAM. No inference.

  max_ctx = min( trained_context,
                 (shape_RAM - weights - overhead) / KV_bytes_per_token )

Shape RAM is what ONE instance can reach:
  quarter  --membind=N        -> that node only. Hard: it does not spill, it fails.
  half     --interleave=0,1   -> two nodes
  full     --interleave=all   -> all four
Weights are counted per instance because --no-mmap gives each its own private copy —
a quarter instance holds the WHOLE model, not a quarter of it.
"""

NODE_FREE_GIB = 263.0          # measured: numactl --hardware, ~269 GB free per node
OVERHEAD_GIB = 8.0             # compute buffers, draft model, allocator slack
SHAPES = {"quarter": 1, "half": 2, "full": 4}
KVQ = {"f16": 2.0, "q8_0": 34/32, "q4_0": 18/32}

# name -> (layers_with_kv, n_head_kv, k_len, v_len, weights_GiB, ctx_train, k_quant, v_quant)
MODELS = {
    "gemma4-26B-A4B Q4_K_M\n(worker_general)":      (30,  8, 512, 512,  15.6,  262144, "q8_0", "q8_0"),
    "Qwen3.6-35B-A3B Q8_0\n(frontdoor)":            (41,  2, 256, 256,  35.2,  262144, "q8_0", "q8_0"),
    "Qwen3-Next-80B-A3B Q4_K_M\n(ingest_long_context)": (48, 2, 256, 256, 45.1, 262144, "q4_0", "q4_0"),
    "Qwen3.5-122B-A10B Q4_K_M\n(architect_general)": (49,  2, 256, 256,  72.9,  262144, "q4_0", "f16"),
    "GLM-5.2 UD-IQ2_M\n(not deployed)":              (79,  1, 576, 512, 222.2, 1048576, "q8_0", "q8_0"),
}

def kv_bytes_per_token(layers, hkv, klen, vlen, kq, vq):
    return layers * hkv * (klen * KVQ[kq] + vlen * KVQ[vq])

rows = []
for name, (L, H, K, V, W, CT, KQ, VQ) in MODELS.items():
    per_tok = kv_bytes_per_token(L, H, K, V, KQ, VQ)
    r = {"model": name.replace("\n", " "), "kv_kib": per_tok/1024, "w": W,
         "ctx_train": CT, "kq": KQ, "vq": VQ}
    for shape, nodes in SHAPES.items():
        budget = NODE_FREE_GIB * nodes - W - OVERHEAD_GIB
        if budget <= 0:
            r[shape] = 0; r[shape + "_cap"] = "RAM"; continue
        toks = int(budget * (1024**3) / per_tok)
        r[shape] = min(toks, CT)
        r[shape + "_cap"] = "trained-ctx" if toks >= CT else "RAM"
    rows.append(r)

w = 44
print(f"{'model':{w}} {'KV/tok':>9} {'weights':>8} | {'quarter':>12} {'half':>12} {'full':>12}")
print("-" * (w + 62))
for r in rows:
    def f(s):
        v = r[s]
        return f"{v/1000:,.0f}k" + ("*" if r[s+"_cap"] == "trained-ctx" else "")
    print(f"{r['model']:{w}} {r['kv_kib']:8.1f}K {r['w']:7.1f}G | "
          f"{f('quarter'):>12} {f('half'):>12} {f('full'):>12}")
print()
print(f"per-node free {NODE_FREE_GIB:.0f} GiB, overhead allowance {OVERHEAD_GIB:.0f} GiB/instance")
print("* = capped by the model's TRAINED context, not by RAM — headroom to spare")
print("no * = capped by RAM on that shape")
print()
print("KV quant per role is as configured today:")
for r in rows:
    print(f"  {r['model'][:44]:44} k={r['kq']:5} v={r['vq']:5}  {r['kv_kib']:7.1f} KiB/token")
