import sys, hashlib; sys.path.insert(0,"/mnt/raid0/llm/llama.cpp/gguf-py"); import gguf
a={t.name:t for t in gguf.GGUFReader(sys.argv[1]).tensors}; b={t.name:t for t in gguf.GGUFReader(sys.argv[2]).tensors}
for n in ("token_embd.weight","output.weight"):
    ha=hashlib.sha256(a[n].data.tobytes()).hexdigest(); hb=hashlib.sha256(b[n].data.tobytes()).hexdigest()
    print(n, ha==hb, ha[:16])
