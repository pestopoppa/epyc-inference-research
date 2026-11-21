#!/usr/bin/env python3
"""WSTREAM-ENTROPY-CENSUS: zeroth-order entropy of Q8_0 int8 codes and fp16 scales,
weighted by bytes streamed per decode token. Reads GGUF via mmap; samples up to
MAXB contiguous-chunked blocks per tensor (strided chunks across the tensor)."""
import sys, json, math
sys.path.insert(0, "/mnt/raid0/llm/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader

MAXB = 1 << 20          # blocks sampled per tensor (34 MiB)
CHUNKS = 64
OVH = (0.012, 0.021)    # per-tensor coder overhead band from the box

def ent(counts):
    c = counts[counts > 0].astype(np.float64); n = c.sum()
    p = c / n
    h = -(p * np.log2(p)).sum()
    return h + (len(c) - 1) / (2 * n * math.log(2))  # Miller-Madow

def kv(r, key):
    f = r.fields.get(key)
    if f is None: return None
    return int(f.parts[f.data[0]][0])

def main(path, out):
    r = GGUFReader(path, "r")
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    n_exp = kv(r, f"{arch}.expert_count"); n_used = kv(r, f"{arch}.expert_used_count")
    rows = []; tot_stream = 0.0
    for t in r.tensors:
        name = t.name; nb = int(t.n_bytes); ty = int(t.tensor_type)
        if name.startswith("token_embd"):
            frac = 0.0  # get_rows: one row per token, negligible
        elif "_exps" in name and n_exp:
            frac = n_used / n_exp
        elif ".nextn." in name or name.startswith("mtp"):
            frac = 0.0  # MTP head: not on the main decode stream
        else:
            frac = 1.0
        stream = nb * frac; tot_stream += stream
        row = dict(name=name, type=ty, bytes=nb, frac=frac, stream=stream)
        if ty == 8 and stream > 0:  # Q8_0
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            nblk = raw.size // 34
            blk = raw[: nblk * 34].reshape(nblk, 34)
            if nblk > MAXB:
                per = MAXB // CHUNKS; starts = np.linspace(0, nblk - per, CHUNKS).astype(np.int64)
                blk = np.concatenate([blk[s:s + per] for s in starts])
            sc = np.ascontiguousarray(blk[:, :2]).view(np.uint16).reshape(-1)
            qs = blk[:, 2:].reshape(-1)
            hq = ent(np.bincount(qs, minlength=256))
            hs = ent(np.bincount(sc, minlength=65536))
            hs_hi = ent(np.bincount(sc >> 8, minlength=256))
            # split stream: raw low 4 bits + entropy-coded high nibble
            hq_hi = ent(np.bincount(qs >> 4, minlength=16))
            bits = 32 * hq + hs
            ideal = 1 - bits / 272.0
            row.update(hq=hq, hs=hs, hs_hi=hs_hi, hq_hi=hq_hi, ideal=ideal,
                       net_lo=1 - (bits / 272.0) * (1 + OVH[1]),
                       net_hi=1 - (bits / 272.0) * (1 + OVH[0]),
                       split=1 - (32 * (4 + hq_hi) + 16) / 272.0,
                       sampled_blocks=int(blk.shape[0]))
        rows.append(row)
        print(name, ty, row.get("hq"), row.get("ideal"), file=sys.stderr, flush=True)
    def wsum(k):
        return sum(r_["stream"] * r_.get(k, 0.0) for r_ in rows) / tot_stream
    summ = dict(model=path, arch=arch, n_expert=n_exp, n_used=n_used,
                stream_bytes_per_token=tot_stream,
                q8_stream_share=sum(r_["stream"] for r_ in rows if "hq" in r_) / tot_stream,
                w_ideal=wsum("ideal"), w_net_ovh2p1=wsum("net_lo"), w_net_ovh1p2=wsum("net_hi"),
                w_split=wsum("split"))
    json.dump(dict(summary=summ, tensors=rows), open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
