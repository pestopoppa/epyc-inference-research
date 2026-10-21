"""Read safetensors headers (and optional raw tensor bytes) from the HF 1017 release by HTTP range.

usage: sthdr.py <index.json> <out.json> <tensor names...>
"""
import json, struct, sys, urllib.request

BASE = ("https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/"
        "1bae0e1088993b9b82de08f59de5be5a34e42170/")


def rng(fn, a, b):
    req = urllib.request.Request(BASE + fn, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=300).read()


def header(fn):
    n = struct.unpack("<Q", rng(fn, 0, 7))[0]
    return n, json.loads(rng(fn, 8, 8 + n - 1))


def tensor_bytes(fn, hdr_len, info):
    a, b = info["data_offsets"]
    base = 8 + hdr_len
    return rng(fn, base + a, base + b - 1)


if __name__ == "__main__":
    idx = json.load(open(sys.argv[1]))
    wm = idx["weight_map"]
    names = sys.argv[3:]
    out = {}
    for fn in sorted({wm[n] for n in names}):
        n, h = header(fn)
        out[fn] = {"hdr_len": n, "tensors": {k: v for k, v in h.items() if k in names}}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))
