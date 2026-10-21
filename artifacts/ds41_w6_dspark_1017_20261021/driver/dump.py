import sys
sys.path.insert(0, "/mnt/raid0/llm/llama.cpp/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    if k.startswith("GGUF.") : continue
    t = f.types
    try:
        v = f.contents()
    except Exception as e:
        v = "?"
    if isinstance(v, list) and len(v) > 12: v = f"[list len={len(v)} first={v[:4]}]"
    print("KV", k, t, v)
for t in r.tensors:
    print("T", t.name, t.tensor_type.name, list(t.shape), t.n_bytes)
