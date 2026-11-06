#!/usr/bin/env python3
"""Forward-only fmha check for the AK-ASM-T1 harness (test_mha.py imports mha_bwd_* modules that are not built).
aiter.flash_attn_func (bf16, hdim 128, no bias/alibi/dropout -> fmha_v3_fwd asm when gated in) vs a float32 torch
reference; checkAllclose (instrumented: ASMT1_CHECK) plus an `Output max diff` line per case. No autograd."""
import argparse, torch, aiter
from aiter.test_common import checkAllclose
p = argparse.ArgumentParser()
p.add_argument('-b', '--batch_size', type=int, nargs='*', default=[2])
p.add_argument('-n', '--nheads', type=int, nargs='*', default=[16])
p.add_argument('-q', '--seqlen_q', type=int, nargs='*', default=[512])
p.add_argument('-k', '--seqlen_k', type=int, nargs='*', default=[512])
p.add_argument('--causal', type=int, nargs='*', default=[0, 1])
a = p.parse_args()
torch.manual_seed(0)
for B in a.batch_size:
    for H in a.nheads:
        for sq in a.seqlen_q:
            for sk in a.seqlen_k:
                for c in a.causal:
                    q, k, v = (torch.randn(B, s, H, 128, dtype=torch.bfloat16, device='cuda') for s in (sq, sk, sk))
                    with torch.no_grad():
                        out = aiter.flash_attn_func(q, k, v, causal=bool(c))
                        out = out[0] if isinstance(out, tuple) else out
                        qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
                        s = qf @ kf.transpose(-1, -2) / 128 ** 0.5
                        if c:
                            m = torch.ones(sq, sk, dtype=torch.bool, device='cuda').tril(sk - sq)
                            s = s.masked_fill(~m, float('-inf'))
                        ref = (s.softmax(-1) @ vf).transpose(1, 2).to(torch.bfloat16)
                    d = (out.float() - ref.float()).abs().max().item()
                    print(f'Output max diff B={B} H={H} sq={sq} sk={sk} causal={c}: {d}', flush=True)
                    checkAllclose(ref, out, msg=f'fmha fwd B={B} H={H} sq={sq} sk={sk} causal={c}', atol=0.03, rtol=0.03)
