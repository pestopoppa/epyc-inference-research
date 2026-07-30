"""Per-node RAM for the proposed lineup: 1 full + 2 halves per quarterable role.

KV is allocated UPFRONT at load for the whole -c, so raising context is a real
resident-memory commitment, not a ceiling that fills lazily.

Placement of each instance's memory:
  full   --interleave=all  -> spreads evenly over all 4 nodes
  half A --interleave=0,1  -> spreads over nodes 0,1
  half B --interleave=2,3  -> spreads over nodes 2,3
Under --no-mmap every instance carries a PRIVATE copy of the whole model.
"""

NODE_GIB = 283.0          # per-node total
NODE_FREE_GIB = 263.0     # measured free
OVERHEAD = 8.0            # compute buffers + draft, per instance

# name -> (weights_GiB, kv_KiB_per_token, trained_ctx, quarterable)
MODELS = {
    "gemma4-26B-A4B Q4_K_M":      (15.6, 255.0, 262144, True),
    "Qwen3.6-35B-A3B Q8_0":       (35.2,  43.6, 262144, True),
    "Qwen3-Next-80B-A3B Q4_K_M":  (45.1,  27.0, 262144, True),
    "Qwen3.5-122B-A10B Q4_K_M":   (72.9,  62.8, 262144, False),   # full only today
}
CTXS = [32768, 65536, 131072, 262144]

def kv_gib(kib_per_tok, ctx): return kib_per_tok * ctx / (1024**2)

print("KV cost per instance, by context length (GiB)")
print(f"{'model':30}" + "".join(f"{c//1024:>10}k" for c in CTXS))
print("-"*72)
for n,(w,k,ct,_) in MODELS.items():
    print(f"{n:30}" + "".join(f"{kv_gib(k,c):10.1f} " for c in CTXS))

print()
print("PROPOSED LINEUP: 1 full + 2 halves for the three quarterable roles,")
print("122B full only. Per-NODE resident memory:")
print()
print(f"{'context':>10} {'total GiB':>11} {'per-node GiB':>14} {'% of 283':>10} {'verdict':>10}")
print("-"*62)
for ctx in CTXS:
    per_node = 0.0
    total = 0.0
    for n,(w,k,ct,q) in MODELS.items():
        c = min(ctx, ct)
        inst = w + kv_gib(k, c) + OVERHEAD
        if q:
            total += 3*inst
            # full spreads /4 across nodes; each half spreads /2 across its own 2 nodes
            per_node += inst/4 + inst/2
        else:
            total += inst
            per_node += inst/4
    ok = "OK" if per_node < NODE_FREE_GIB else "OVER"
    print(f"{ctx//1024:>9}k {total:11.1f} {per_node:14.1f} {100*per_node/NODE_GIB:9.0f}% {ok:>10}")

print()
print(f"per-node free measured {NODE_FREE_GIB:.0f} GiB of {NODE_GIB:.0f} GiB total")
print("per-node = (full instance / 4 nodes) + (its own half / 2 nodes), summed over roles")
