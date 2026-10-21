"""DS41-W5a part 1: weight fidelity of the antirez 1017 GGUF against the HF 1017 source tensors.

Reads a handful of source tensors from HF by HTTP range (no full-shard download), dequantizes both
sides to float64 and reports NMSE = sum((gguf - src)^2) / sum(src^2) per tensor, plus a bit-level
check for the MXFP4 experts (codes + E8M0 scales identical). Single-threaded numpy on a few MB.

usage: weight_fidelity.py <index.json> <gguf_tensors.json> <out.json>
"""
import json, struct, sys
import numpy as np
sys.path.insert(0, "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/gguf-py")
sys.path.insert(0, "/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021/w5a")
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize
from sthdr import header, tensor_bytes

GD = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/"
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def e4m3_table():
    t = np.zeros(256)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        if e == 15 and m == 7:
            v = np.nan
        t[b] = -v if s else v
    return t


E4M3 = e4m3_table()


def e8m0(u8):
    return np.ldexp(1.0, u8.astype(np.int64) - 127)


def gguf_raw(meta, name, row0=None, nrows=None):
    """Raw bytes of a GGUF tensor (optionally rows [row0, row0+nrows) of the flattened 2D view)."""
    m = meta[name]
    mm = np.memmap(GD + m["file"], dtype=np.uint8, mode="r")
    a = mm[m["offset"]: m["offset"] + m["nbytes"]]
    if row0 is not None:
        rowbytes = m["nbytes"] // int(np.prod(m["shape"][1:]))
        a = a[row0 * rowbytes:(row0 + nrows) * rowbytes]
    return np.array(a)


def nmse(a, ref):
    return float(np.sum((a - ref) ** 2) / np.sum(ref ** 2))


def main():
    idx = json.load(open(sys.argv[1]))["weight_map"]
    meta = json.load(open(sys.argv[2]))
    hdrs = {}

    def src(name):
        fn = idx[name]
        if fn not in hdrs:
            hdrs[fn] = header(fn)
        n, h = hdrs[fn]
        info = h[name]
        return info, tensor_bytes(fn, n, info)

    res = []
    # --- MXFP4 experts: (gguf name, expert id, hf name) ---
    for L, e, gname, hname in [(3, 0, "ffn_gate_exps", "w1"), (3, 0, "ffn_up_exps", "w3"),
                               (3, 7, "ffn_down_exps", "w2"), (30, 200, "ffn_gate_exps", "w1")]:
        gn = f"blk.{L}.{gname}.weight"
        shp = meta[gn]["shape"]                     # [k, m, n_exp]
        k, mrows = shp[0], shp[1]
        raw = gguf_raw(meta, gn, row0=e * mrows, nrows=mrows)
        g = dequantize(raw, Q.MXFP4).astype(np.float64).reshape(mrows, k)
        blocks = raw.reshape(-1, 17)
        g_e = blocks[:, 0].reshape(mrows, k // 32)
        g_q = blocks[:, 1:]                          # ggml layout: low nibble = j, high = j+16
        g_codes = np.concatenate([g_q & 15, g_q >> 4], axis=1).reshape(mrows, k)
        winfo, wb = src(f"layers.{L}.ffn.experts.{e}.{hname}.weight")
        sinfo, sb = src(f"layers.{L}.ffn.experts.{e}.{hname}.scale")
        w = np.frombuffer(wb, dtype=np.uint8).reshape(winfo["shape"])
        s = np.frombuffer(sb, dtype=np.uint8).reshape(sinfo["shape"])
        best = None
        for order in ("lo_first", "hi_first"):
            lo, hi = w & 15, w >> 4
            codes = np.stack([lo, hi] if order == "lo_first" else [hi, lo], axis=-1).reshape(w.shape[0], -1)
            ref = E2M1[codes] * np.repeat(e8m0(s), 32, axis=1)
            r = {"tensor": gn, "expert": e, "src": f"layers.{L}.ffn.experts.{e}.{hname}", "gguf_type": "MXFP4",
                 "src_dtype": f"{winfo['dtype']}(packed fp4)+{sinfo['dtype']}", "nibble_order": order,
                 "nmse": nmse(g, ref), "max_abs_diff": float(np.max(np.abs(g - ref))),
                 "codes_identical": bool(np.array_equal(g_codes, codes)),
                 "scales_identical": bool(np.array_equal(g_e, s)), "n": int(g.size)}
            if best is None or r["nmse"] < best["nmse"]:
                best = r
        res.append(best)
        print(json.dumps(best), flush=True)
    # --- FP8 e4m3 + E8M0 32x32 block scale -> Q8_0 ---
    for L, gname, hname in [(3, "attn_q_a", "attn.wq_a"), (3, "attn_kv", "attn.wkv"),
                            (3, "ffn_gate_shexp", "ffn.shared_experts.w1"), (30, "attn_q_a", "attn.wq_a")]:
        gn = f"blk.{L}.{gname}.weight"
        k, mrows = meta[gn]["shape"][:2]
        g = dequantize(gguf_raw(meta, gn), Q.Q8_0).astype(np.float64).reshape(mrows, k)
        winfo, wb = src(f"layers.{L}.{hname}.weight")
        sinfo, sb = src(f"layers.{L}.{hname}.scale")
        w = E4M3[np.frombuffer(wb, dtype=np.uint8)].reshape(winfo["shape"])
        sc = e8m0(np.frombuffer(sb, dtype=np.uint8).reshape(sinfo["shape"]))
        bm, bk = w.shape[0] // sc.shape[0], w.shape[1] // sc.shape[1]
        ref = w * np.repeat(np.repeat(sc, bm, axis=0), bk, axis=1)
        r = {"tensor": gn, "src": f"layers.{L}.{hname}", "gguf_type": "Q8_0",
             "src_dtype": f"{winfo['dtype']}+{sinfo['dtype']}[{bm}x{bk}]", "nmse": nmse(g, ref),
             "max_abs_diff": float(np.max(np.abs(g - ref))), "n": int(g.size)}
        res.append(r)
        print(json.dumps(r), flush=True)
    # --- F16 (source dtype read from the header) ---
    for L, gname, hname in [(3, "hc_attn_fn", "hc_attn_fn"), (3, "hc_ffn_fn", "hc_ffn_fn")]:
        gn = f"blk.{L}.{gname}.weight"
        g = np.frombuffer(gguf_raw(meta, gn).tobytes(), dtype=np.float16).astype(np.float64)
        winfo, wb = src(f"layers.{L}.{hname}")
        dt = winfo["dtype"]
        if dt == "BF16":
            ref = (np.frombuffer(wb, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        elif dt == "F32":
            ref = np.frombuffer(wb, dtype=np.float32).astype(np.float64)
        else:
            ref = np.frombuffer(wb, dtype=np.float16).astype(np.float64)
        r = {"tensor": gn, "src": f"layers.{L}.{hname}", "gguf_type": "F16", "src_dtype": dt,
             "src_shape": winfo["shape"], "gguf_shape": meta[gn]["shape"],
             "nmse": nmse(g, ref.reshape(-1)), "max_abs_diff": float(np.max(np.abs(g - ref.reshape(-1)))),
             "n": int(g.size)}
        res.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"schema": "ds41.w5a.weight_fidelity.v1", "hf_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
               "gguf_dir": GD, "rows": res}, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
