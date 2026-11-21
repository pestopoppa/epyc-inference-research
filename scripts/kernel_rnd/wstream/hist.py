import sys,json
sys.path.insert(0,"/mnt/raid0/llm/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader
out={}
for tag,path,pat in [("q27b","/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf","ffn_"),("a3b","/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf","_exps")]:
    r=GGUFReader(path,"r"); h=np.zeros(256,np.int64); hs=np.zeros(65536,np.int64)
    for t in r.tensors:
        if int(t.tensor_type)!=8 or pat not in t.name or not any(f"blk.{i}." in t.name for i in (3,17,31)): continue
        raw=np.asarray(t.data).reshape(-1).view(np.uint8); n=raw.size//34; b=raw[:n*34].reshape(n,34)[: 1<<19]
        h+=np.bincount(b[:,2:].reshape(-1),minlength=256); hs+=np.bincount(np.ascontiguousarray(b[:,:2]).view(np.uint16).reshape(-1),minlength=65536)
    open(f"hist_{tag}.txt","w").write("\n".join(map(str,h))+"\n")
    p=h[h>0]/h.sum(); print(tag,-(p*np.log2(p)).sum())
