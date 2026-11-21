#!/usr/bin/env python3
"""Scales-only census: raw int8 codes + coded fp16 block scales. Per tensor: H(scale), H(zigzag delta vs previous
block in the same row), best-k Rice code on zigzag delta (static, no table), plus per-row 32-bit offset for random access."""
import sys, json, math
sys.path.insert(0, "/mnt/raid0/llm/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader

MAXROWS = 4096

def ent(c):
    c = c[c > 0].astype(np.float64); n = c.sum(); p = c / n
    return float(-(p * np.log2(p)).sum() + (len(c) - 1) / (2 * n * math.log(2)))

def kv(r, k):
    f = r.fields.get(k)
    return None if f is None else int(f.parts[f.data[0]][0])

def main(path, out):
    r = GGUFReader(path, "r")
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    ne_, nu = kv(r, f"{arch}.expert_count"), kv(r, f"{arch}.expert_used_count")
    rows, tot = [], 0.0
    for t in r.tensors:
        nb = int(t.n_bytes); n = t.name
        frac = 0.0 if n.startswith("token_embd") or ".nextn." in n else (nu / ne_ if "_exps" in n and ne_ else 1.0)
        st = nb * frac; tot += st; row = dict(name=n, stream=st)
        if int(t.tensor_type) == 8 and st > 0:
            bpr = int(t.shape[0]) // 32
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            nrow = raw.size // (34 * bpr)
            blk = raw[: nrow * bpr * 34].reshape(nrow, bpr, 34)
            if nrow > MAXROWS:
                blk = blk[np.linspace(0, nrow - 1, MAXROWS).astype(np.int64)]
            sc = np.ascontiguousarray(blk[:, :, :2]).view(np.uint16)[..., 0].astype(np.int32)  # rows x bpr
            d = np.diff(sc, axis=1, prepend=0)  # first block of a row coded against 0
            zz = np.where(d >= 0, 2 * d, -2 * d - 1).reshape(-1)
            hs = ent(np.bincount(sc.reshape(-1), minlength=65536))
            hd = ent(np.bincount(zz, minlength=131072))
            cnt = np.bincount(sc.reshape(-1), minlength=65536); srt = np.sort(cnt)[::-1]
            cov255 = float(srt[:255].sum() / cnt.sum()); ndist = int((cnt > 0).sum())
            idx8 = 8 + (1 - cov255) * 16  # fixed 8-bit index into per-tensor 255-entry table, escape -> 16 raw bits side stream
            rice = min(float(np.mean((zz >> k) + 1 + k)) for k in range(0, 16))  # bits per scale, static
            off = 32.0 / bpr  # per-row 32-bit offset, bits per block
            def sv(bits):  # tensor saving fraction given scale bits per block
                return (16 - bits) / 272.0
            row.update(bpr=bpr, h_scale=hs, h_delta=hd, rice_bits=rice, cov255=cov255, ndistinct=ndist, idx8_bits=idx8, save_idx8=sv(idx8 + (0 if cov255 == 1 else off)) - 255*16/ (nb*8) * 0,
                       save_ideal=sv(hs), save_entropy_net=sv(hs * 1.021 + off), save_delta_net=sv(hd * 1.021 + off),
                       save_rice=sv(rice + off))
        rows.append(row)
    w = lambda k: sum(x["stream"] * x.get(k, 0.0) for x in rows) / tot
    s = dict(model=path, ideal=w("save_ideal"), entropy_net=w("save_entropy_net"), delta_entropy_net=w("save_delta_net"),
             rice_static=w("save_rice"), idx8_fixed=w("save_idx8"),
             mean_rice_bits=w("rice_bits") / max(1e-9, w("save_ideal") and 1) if False else None)
    q8 = [x for x in rows if "h_scale" in x]
    sw = sum(x["stream"] for x in q8)
    for k in ("h_scale", "h_delta", "rice_bits", "cov255", "idx8_bits"):
        s["w_" + k] = sum(x["stream"] * x[k] for x in q8) / sw
    s.pop("mean_rice_bits")
    json.dump(dict(summary=s, tensors=rows), open(out, "w"), indent=1)
    print(json.dumps(s, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
