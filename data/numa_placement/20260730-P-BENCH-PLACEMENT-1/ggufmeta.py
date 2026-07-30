import struct, sys

# Minimal GGUF key-value header reader. Reads only the metadata block at the head
# of the file — no tensors, no model load. Enough to get the KV-cache geometry.
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
FIX = {T_U8:("<B",1),T_I8:("<b",1),T_U16:("<H",2),T_I16:("<h",2),T_U32:("<I",4),
       T_I32:("<i",4),T_F32:("<f",4),T_BOOL:("<?",1),T_U64:("<Q",8),T_I64:("<q",8),T_F64:("<d",8)}

def read(f):
    def u32(): return struct.unpack("<I", f.read(4))[0]
    def u64(): return struct.unpack("<Q", f.read(8))[0]
    def s():
        n = u64(); return f.read(n).decode("utf-8", "replace")
    def val(t):
        if t in FIX:
            fmt, n = FIX[t]; return struct.unpack(fmt, f.read(n))[0]
        if t == T_STR: return s()
        if t == T_ARR:
            et = u32(); n = u64()
            return [val(et) for _ in range(n)]
        raise ValueError(f"unknown gguf type {t}")
    magic = f.read(4)
    if magic != b"GGUF": raise ValueError(f"not GGUF: {magic!r}")
    u32(); u64()                    # version, tensor_count
    nkv = u64()
    kv = {}
    for _ in range(nkv):
        k = s(); t = u32(); kv[k] = val(t)
    return kv

for path in sys.argv[1:]:
    with open(path, "rb") as f:
        kv = read(f)
    arch = kv.get("general.architecture", "?")
    g = lambda k, d=None: kv.get(f"{arch}.{k}", d)
    layers = g("block_count")
    hkv    = g("attention.head_count_kv")
    h      = g("attention.head_count")
    embd   = g("embedding_length")
    klen   = g("attention.key_length")
    vlen   = g("attention.value_length")
    ctx    = g("context_length")
    if klen is None and h and embd: klen = embd // h
    if vlen is None: vlen = klen
    # head_count_kv may be a per-layer array (hybrid/SSM models): 0 = no KV on that layer
    if isinstance(hkv, list):
        kv_layers = sum(1 for x in hkv if x)
        hkv_eff = max(hkv) if hkv else 0
    else:
        kv_layers, hkv_eff = layers, hkv
    print(f"{path.split('/')[-1]}")
    print(f"  arch={arch} layers={layers} kv_layers={kv_layers} n_head={h} "
          f"n_head_kv={hkv_eff} k_len={klen} v_len={vlen} ctx_train={ctx}")
    if kv_layers and hkv_eff and klen:
        # bytes/token = kv_layers * n_head_kv * (k_len + v_len) * bytes_per_element
        for name, bpe in (("f16", 2.0), ("q8_0", 34/32), ("q4_0", 18/32)):
            per_tok = kv_layers * hkv_eff * (klen + vlen) * bpe
            print(f"    KV @ {name:4}: {per_tok/1024:8.2f} KiB/token  "
                  f"-> {per_tok*32768/1024**3:7.2f} GiB @ 32k ctx")
