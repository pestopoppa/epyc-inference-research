#!/usr/bin/env python3
"""Header-only GGUF review (AMSB-C390-0): arch, KV summary, tokenizer/template digests,
tensor-name families, and a ggml-type histogram.  Reads at most the file prefix given
(default: whole file passed, expected to be an HTTP-range prefix).  Never touches tensor data.
Usage: python3 -I gguf_header_review.py PREFIX.bin [OUT.json]
"""
import hashlib, json, struct, sys
from collections import Counter

SZ = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
TN = {0: "F32", 1: "F16", 2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16", 42: "Q2_0"}


def main(path):
    buf = open(path, "rb").read()
    p = 4
    assert buf[:4] == b"GGUF"

    def rd(fmt):
        nonlocal p
        v = struct.unpack_from(fmt, buf, p)[0]
        p += struct.calcsize(fmt)
        return v

    def rs():
        nonlocal p
        n = rd("<Q")
        s = buf[p:p + n]
        p += n
        return s

    def val(t):
        nonlocal p
        if t in SZ:
            return rd(SZ[t])
        if t == 8:
            return rs()
        if t == 9:
            et = rd("<I")
            n = rd("<Q")
            if et in SZ:
                sz = struct.calcsize(SZ[et])
                raw = buf[p:p + sz * n]
                p += sz * n
                return ("array", et, n, hashlib.sha256(raw).hexdigest())
            items = [val(et) for _ in range(n)]
            h = hashlib.sha256(b"\x00".join(items)).hexdigest() if et == 8 else None
            return ("array", et, n, h)
        raise ValueError(t)

    ver = rd("<I")
    nt = rd("<Q")
    nkv = rd("<Q")
    kv = {}
    for _ in range(nkv):
        k = rs().decode()
        t = rd("<I")
        kv[k] = val(t)
    out = {"gguf_version": ver, "n_tensors": nt, "n_kv": nkv, "kv_scalar": {}, "arrays": {}, "strings": {}}
    for k, v in kv.items():
        if isinstance(v, tuple):
            out["arrays"][k] = {"elem_type": v[1], "len": v[2], "sha256": v[3]}
        elif isinstance(v, bytes):
            d = {"len": len(v), "sha256": hashlib.sha256(v).hexdigest()}
            if len(v) < 200:
                d["text"] = v.decode("utf-8", "replace")
            out["strings"][k] = d
        else:
            out["kv_scalar"][k] = v
    ct = kv.get("tokenizer.chat_template")
    if isinstance(ct, bytes):
        open(path + ".chat_template.txt", "wb").write(ct)
    names, types = [], Counter()
    for _ in range(nt):
        n = rs().decode()
        nd = rd("<I")
        dims = [rd("<Q") for _ in range(nd)]
        ty = rd("<I")
        rd("<Q")
        names.append((n, dims, TN.get(ty, f"type{ty}")))
        types[TN.get(ty, f"type{ty}")] += 1
    out["type_hist"] = dict(types)
    out["blk_max"] = max(int(n.split(".")[1]) for n, _, _ in names if n.startswith("blk."))
    out["blk_max_tensors"] = [(n, d, t) for n, d, t in names if n.startswith(f"blk.{out['blk_max']}.")]
    out["non_blk"] = [(n, d, t) for n, d, t in names if not n.startswith("blk.")]
    out["header_end_offset"] = p
    return out


if __name__ == "__main__":
    r = main(sys.argv[1])
    s = json.dumps(r, indent=1, default=str)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(s)
    print(s[:6000])
