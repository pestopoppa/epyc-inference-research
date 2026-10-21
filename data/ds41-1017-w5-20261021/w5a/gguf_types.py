"""List GGUF tensor names/types/shapes across the 1017 shards (header only; tensor data is mmapped, not read).

usage: gguf_types.py <out.json>
"""
import glob, json, sys
sys.path.insert(0, "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/gguf-py")
from gguf import GGUFReader

D = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf"
out = {}
for f in sorted(glob.glob(D + "/DeepSeek-V4.1-Flash-1017-MXFP4-experts-*-of-00009.gguf")):
    r = GGUFReader(f)
    for t in r.tensors:
        out[t.name] = {"type": t.tensor_type.name, "shape": [int(x) for x in t.shape],
                       "file": f.rsplit("/", 1)[1], "offset": int(t.data_offset), "nbytes": int(t.n_bytes)}
json.dump(out, open(sys.argv[1], "w"), indent=0)
import collections
c = collections.Counter(v["type"] for v in out.values())
print(len(out), c)
for k, v in out.items():
    if k.startswith("blk.3.") or v["type"] in ("F16", "BF16") and not k.startswith("blk.") or (v["type"] == "F16" and k.startswith("blk.3")):
        print(k, v["type"], v["shape"], v["file"][-14:])
print("F16:", sorted({k.split(".", 2)[-1] if k.startswith("blk.") else k for k, v in out.items() if v["type"] == "F16"}))
print("Q8_0:", sorted({k.split(".", 2)[-1] if k.startswith("blk.") else k for k, v in out.items() if v["type"] == "Q8_0"}))
print("MXFP4:", sorted({k.split(".", 2)[-1] if k.startswith("blk.") else k for k, v in out.items() if v["type"] == "MXFP4"}))
