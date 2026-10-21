import sys; sys.path.insert(0,"/mnt/raid0/llm/llama.cpp/gguf-py"); import gguf, numpy as np
a={t.name:t for t in gguf.GGUFReader(sys.argv[1]).tensors}; b={t.name:t for t in gguf.GGUFReader(sys.argv[2]).tensors}
names=["output_norm.weight","blk.0.dspark_main_norm.weight","dspark_markov_embd.weight","dspark_markov_head.weight","dspark_confidence_proj.weight","blk.0.attn_norm.weight","blk.2.ffn_norm.weight"]
def v(t): return np.asarray(t.data,dtype=np.float32).ravel()
for x in names:
    row=[]
    for y in names:
        if a[x].data.size!=b[y].data.size: row.append("  -  "); continue
        row.append(f"{np.corrcoef(v(a[x]),v(b[y]))[0,1]:.3f}")
    print(x[:28].ljust(28), " ".join(row))
