#!/usr/bin/env python3
"""r3 smoke: ONE process keeps the GPU resident >= --hold_s (default 50 s) while repeatedly running the retargeted gfx90a
topksoftmax asm kernel (aiter.topk_softmax_asm) against a torch softmax+topk reference via checkAllclose (ASMT1_CHECK lines)."""
import argparse, time, torch, aiter
from aiter.test_common import checkAllclose
p = argparse.ArgumentParser()
p.add_argument('--hold_s', type=float, default=50.0)
p.add_argument('-e', '--expert', type=int, default=128)
a = p.parse_args()
torch.manual_seed(0)
t0 = time.time(); it = 0
ballast = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')  # visible VRAM residency for the whole hold
while True:
    M, E, k = 64, a.expert, 8
    g = torch.randn(M, E, dtype=torch.bfloat16, device='cuda')
    w_ref, id_ref = torch.softmax(g.float(), -1).topk(k, -1)
    w_ref = w_ref / w_ref.sum(-1, keepdim=True)
    w = torch.empty(M, k, dtype=torch.float32, device='cuda')
    ids = torch.empty(M, k, dtype=torch.int32, device='cuda')
    tei = torch.empty(M, k, dtype=torch.int32, device='cuda')
    aiter.topk_softmax_asm(w, ids, tei, g, True)
    torch.cuda.synchronize()
    if it % 20 == 0:
        o = ids.long().sort(-1); r = id_ref.sort(-1)
        checkAllclose(r.values.float(), o.values.float(), msg=f'asm topk_ids iter {it}')
        checkAllclose(w_ref.gather(1, r.indices), w.gather(1, o.indices), msg=f'asm topk_weights iter {it}')
    it += 1
    if time.time() - t0 >= a.hold_s:
        break
print(f'hold done: {time.time() - t0:.1f} s, {it} launches, ballast {ballast.numel() >> 20} MiB', flush=True)
