#!/usr/bin/env python3
"""Reference PyTorch DFlash2 drafter with llama.cpp C14 semantics, plus the block-diffusion training losses.

Mirrors src/models/dflash.cpp @ eee3da838 (graph<true> encoder, graph<false> decoder, build_dflash2_conv,
build_post_sampling selector) so that a checkpoint trained here converts with convert_hf_to_gguf.py
(DFlash2DraftModel) and decodes the same way in llama-server. Parameter names are the HF names the converter
maps (gguf-py tensor_mapping.py: model.fc, model.hidden_norm, attention_conv.base_kernel, ...).

Why block 16 needs no new parameters: every weight here is position-agnostic. The block input is the anchor
token plus (B-1) copies of the shared MASK embedding (target token_embd row mask_token_id), position enters
only through NeoX RoPE, the in-block conv is a kernel-2 shift, and the selector codebooks are token-indexed.
So the block-8 weights load unchanged into a block-16 model; only dflash_config.block_size changes. What IS
new is behaviour: positions 8-15 were never trained, and positions 0-7 now attend (non-causally) to 8 extra
MASK rows. The trainer handles that with a block-size curriculum and a position-decayed loss (see
train_block16.yaml), not with new initialisation.

Shapes (z-lab Qwen3.8-27B-DFlash2): hidden 5120, 5 layers, 32 q heads / 8 kv heads x 128, ffn 17408,
conv kernel 2 / group 16 (320 groups -> kernel_projection 5120->1280), selector rank 256, top-k 16,
fc 25600->5120 (5 taps x target hidden 5120), sliding window 2048, rope base 1e7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DF2Config:
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    sliding_window: int = 2048
    target_hidden_size: int = 5120
    target_layer_ids: list = field(default_factory=lambda: [5, 19, 33, 47, 61])  # HF ids; GGUF taps are +1
    block_size: int = 8
    mask_token_id: int = 248070
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    vocab_size: int = 248320

    @classmethod
    def from_hf(cls, cfg: dict) -> "DF2Config":
        d = cfg.get("dflash_config", {})
        return cls(hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
                   num_hidden_layers=cfg["num_hidden_layers"], num_attention_heads=cfg["num_attention_heads"],
                   num_key_value_heads=cfg["num_key_value_heads"], head_dim=cfg.get("head_dim", 128),
                   rms_norm_eps=cfg.get("rms_norm_eps", 1e-6), rope_theta=cfg.get("rope_theta", 1e7),
                   sliding_window=cfg.get("sliding_window") or 0, target_hidden_size=cfg["target_hidden_size"],
                   target_layer_ids=list(d["target_layer_ids"]), block_size=int(d["block_size"]),
                   mask_token_id=int(d["mask_token_id"]), conv_kernel_size=int(d["conv_kernel_size"]),
                   conv_group_size=int(d["conv_group_size"]), selector_rank=int(d["selector_rank"]),
                   selector_top_k=int(d["selector_top_k"]), vocab_size=cfg["vocab_size"])


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope_neox(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """x [..., T, H, D], pos [..., T] -> NeoX rotation (first half / second half pairs), as ggml_rope_ext NEOX."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.float()[..., None] * inv                                # [..., T, D/2]
    cos, sin = ang.cos()[..., None, :], ang.sin()[..., None, :]       # broadcast over heads
    x1, x2 = x[..., : d // 2].float(), x[..., d // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


class DF2Conv(nn.Module):
    """build_dflash2_conv: per-token dynamic grouped kernel + static base kernel, causal shift inside each block.
    base_kernel [2, K, H] (GGUF ne [H, K, 2]); kernel_projection H -> G*K*2 laid out [side, tap, group]."""

    def __init__(self, c: DF2Config):
        super().__init__()
        self.K, self.G, self.H = c.conv_kernel_size, c.hidden_size // c.conv_group_size, c.hidden_size
        self.gs = c.conv_group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, self.K, self.H))
        self.kernel_projection = nn.Linear(self.H, self.G * self.K * 2, bias=False)

    def coeffs(self, h_norm: torch.Tensor) -> torch.Tensor:          # [A, B, H] -> [A, B, 2, K, G]
        A, B, _ = h_norm.shape
        return self.kernel_projection(h_norm).view(A, B, 2, self.K, self.G)

    def conv(self, x: torch.Tensor, coeffs: torch.Tensor, side: int) -> torch.Tensor:
        A, B, H = x.shape
        out = None
        for tap in range(self.K):
            vals = x if tap == 0 else F.pad(x, (0, 0, tap, 0))[:, :B]   # shift by tap within the block, zero fill
            c = coeffs[:, :, side, tap, :].repeat_interleave(self.gs, dim=-1)   # [A, B, H]
            term = (c + self.base_kernel[side, tap]) * vals
            out = term if out is None else out + term
        return out


class DF2Layer(nn.Module):
    def __init__(self, c: DF2Config):
        super().__init__()
        self.c = c
        H, nh, nkv, hd = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.input_layernorm = RMSNorm(H, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(H, c.rms_norm_eps)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(H, nh * hd, bias=False)
        self.self_attn.k_proj = nn.Linear(H, nkv * hd, bias=False)
        self.self_attn.v_proj = nn.Linear(H, nkv * hd, bias=False)
        self.self_attn.o_proj = nn.Linear(nh * hd, H, bias=False)
        self.self_attn.q_norm = RMSNorm(hd, c.rms_norm_eps)
        self.self_attn.k_norm = RMSNorm(hd, c.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(H, c.intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(H, c.intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(c.intermediate_size, H, bias=False)
        self.attention_conv = DF2Conv(c)
        self.mlp_conv = DF2Conv(c)

    def ctx_kv(self, g: torch.Tensor, pos: torch.Tensor):
        """graph<false> embd batch: K/V injection from the encoded target features (no attn_norm, k_norm + rope)."""
        c = self.c
        k = self.self_attn.k_norm(self.self_attn.k_proj(g).view(*g.shape[:-1], c.num_key_value_heads, c.head_dim))
        v = self.self_attn.v_proj(g).view(*g.shape[:-1], c.num_key_value_heads, c.head_dim)
        return rope_neox(k, pos, c.rope_theta), v

    def forward(self, x, pos, ctx_k, ctx_v, mask):
        """x [A, B, H] block rows, pos [A, B]; ctx_k/v [N, kvh, d]; mask [A*B, N + A*B] bool (True = attend)."""
        c = self.c
        A, B, H = x.shape
        h = self.input_layernorm(x)
        co = self.attention_conv.coeffs(h)
        h = self.attention_conv.conv(h, co, 0)
        q = self.self_attn.q_norm(self.self_attn.q_proj(h).view(A, B, c.num_attention_heads, c.head_dim))
        k = self.self_attn.k_norm(self.self_attn.k_proj(h).view(A, B, c.num_key_value_heads, c.head_dim))
        v = self.self_attn.v_proj(h).view(A, B, c.num_key_value_heads, c.head_dim)
        q, k = rope_neox(q, pos, c.rope_theta), rope_neox(k, pos, c.rope_theta)
        K = torch.cat([ctx_k, k.reshape(A * B, *k.shape[2:])], 0)           # [N + A*B, kvh, d]
        V = torch.cat([ctx_v, v.reshape(A * B, *v.shape[2:])], 0)
        rep = c.num_attention_heads // c.num_key_value_heads
        Kh = K.repeat_interleave(rep, dim=1).transpose(0, 1)                # [nh, S, d]
        Vh = V.repeat_interleave(rep, dim=1).transpose(0, 1)
        Qh = q.reshape(A * B, c.num_attention_heads, c.head_dim).transpose(0, 1)
        o = F.scaled_dot_product_attention(Qh, Kh, Vh, attn_mask=mask, scale=1.0 / math.sqrt(c.head_dim))
        o = self.self_attn.o_proj(o.transpose(0, 1).reshape(A, B, -1))
        o = self.attention_conv.conv(o, co, 1)
        ffn_inp = o + x
        f = self.post_attention_layernorm(ffn_inp)
        cf = self.mlp_conv.coeffs(f)
        f = self.mlp_conv.conv(f, cf, 0)
        f = self.mlp.down_proj(F.silu(self.mlp.gate_proj(f)) * self.mlp.up_proj(f))
        f = self.mlp_conv.conv(f, cf, 1)
        return f + ffn_inp


class DF2Drafter(nn.Module):
    """Trainable drafter. tok_embd / lm_head are the TARGET's frozen tensors, passed in as buffers."""

    def __init__(self, c: DF2Config, tok_embd: torch.Tensor, lm_head: torch.Tensor):
        super().__init__()
        self.config = c
        self.model = nn.Module()
        n_in = len(c.target_layer_ids) * c.target_hidden_size
        self.model.fc = nn.Linear(n_in, c.hidden_size, bias=False)
        self.model.hidden_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.model.layers = nn.ModuleList([DF2Layer(c) for _ in range(c.num_hidden_layers)])
        self.model.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.model.candidate_selector = nn.Module()
        cs = self.model.candidate_selector
        cs.predecessor_codebook = nn.Parameter(torch.zeros(c.vocab_size, c.selector_rank))
        cs.successor_codebook = nn.Parameter(torch.zeros(c.vocab_size, c.selector_rank))
        cs.hidden_projection = nn.Linear(c.hidden_size, c.selector_rank, bias=False)
        self.register_buffer("tok_embd", tok_embd, persistent=False)   # [V, H]  frozen, target's
        self.register_buffer("lm_head", lm_head, persistent=False)     # [V, H]  frozen, target's

    # ---- attention mask for packed anchors -------------------------------------------------------------
    @staticmethod
    def build_mask(N: int, anchors: torch.Tensor, B: int, window: int) -> torch.Tensor:
        A = anchors.numel()
        dev = anchors.device
        n = torch.arange(N, device=dev)
        p = anchors[:, None]                                              # [A, 1]
        ctx_ok = (n[None] < p) & ((window <= 0) | (n[None] >= p - window))   # [A, N]
        ctx_ok = ctx_ok.repeat_interleave(B, 0)                           # [A*B, N]
        blk = torch.arange(A, device=dev).repeat_interleave(B)
        blk_ok = blk[:, None] == blk[None, :]                             # non-causal inside the block
        return torch.cat([ctx_ok, blk_ok], 1)

    def forward(self, feats: torch.Tensor, tokens: torch.Tensor, anchors: torch.Tensor, B: int):
        """feats [N, n_taps*E_t] (one record), tokens [N], anchors [A] row indices of id_last.
        Returns block hidden [A, B, H] (post output_norm)."""
        c = self.config
        N = feats.shape[0]
        g = self.model.hidden_norm(self.model.fc(feats))                  # encoder: fc -> hidden_norm
        cpos = torch.arange(N, device=feats.device)
        blk_tok = torch.full((anchors.numel(), B), c.mask_token_id, dtype=torch.long, device=feats.device)
        blk_tok[:, 0] = tokens[anchors]
        x = self.tok_embd[blk_tok].to(g.dtype)
        pos = anchors[:, None] + torch.arange(B, device=feats.device)[None]
        mask = self.build_mask(N, anchors, B, c.sliding_window)
        for layer in self.model.layers:
            ck, cv = layer.ctx_kv(g, cpos)
            x = layer(x, pos, ck, cv, mask)
        return self.model.norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.lm_head.to(h.dtype).T

    # ---- losses ---------------------------------------------------------------------------------------
    def loss(self, feats, tokens, anchors, B, gamma: float, final: torch.Tensor | None = None,
             objective: str = "ce", selector_weight: float = 0.1):
        """Block-diffusion loss on positions 1..B-1 with weight exp(-(j-1)/gamma) (gamma<=0: uniform).
        objective 'ce' = hard labels (arm H); 'kl' = forward KL to the target's own next-token distribution,
        teacher logits = lm_head(final[p+j-1]) from the dump's final.bf16 (arm K)."""
        c = self.config
        N = tokens.shape[0]
        h = self.forward(feats, tokens, anchors, B)                       # [A, B, H]
        j = torch.arange(1, B, device=tokens.device)
        lab_idx = anchors[:, None] + j[None]                              # [A, B-1]
        valid = lab_idx < N
        lab = tokens[lab_idx.clamp(max=N - 1)]
        w = torch.exp(-(j - 1).float() / gamma) if gamma > 0 else torch.ones_like(j, dtype=torch.float32)
        W = (w[None] * valid).float()
        lg = self.logits(h[:, 1:]).float()                                # [A, B-1, V]
        if objective == "ce":
            tok_loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), reduction="none").view_as(W)
        elif objective == "kl":
            assert final is not None, "KL arm needs the dump's final hidden (--final)"
            t_lg = (final[(lab_idx - 1).clamp(max=N - 1)].to(self.lm_head.dtype) @ self.lm_head.T).float()
            tok_loss = F.kl_div(F.log_softmax(lg, -1), F.log_softmax(t_lg, -1), log_target=True, reduction="none").sum(-1)
        else:
            raise ValueError(objective)
        main = (tok_loss * W).sum() / W.sum().clamp(min=1e-6)

        # selector: teacher-forced predecessor (anchor at j=1, true token after), top-k candidates of slot j
        cs = self.model.candidate_selector
        hid = cs.hidden_projection(h[:, 1:]).float()                      # [A, B-1, R]
        prev_tok = torch.cat([tokens[anchors][:, None], lab[:, :-1]], 1)
        pred = cs.predecessor_codebook[prev_tok].float()                  # [A, B-1, R]
        unary, cand = lg.topk(c.selector_top_k, -1)                       # [A, B-1, K]
        succ = cs.successor_codebook[cand].float()                        # [A, B-1, K, R]
        scores = unary + torch.einsum("abr,abkr->abk", pred * hid, succ)
        hit = cand == lab[..., None]
        sel_ok = hit.any(-1) & valid
        sel_lab = hit.float().argmax(-1)
        sel = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), sel_lab.reshape(-1), reduction="none").view_as(W)
        Ws = W * sel_ok
        sel_loss = (sel * Ws).sum() / Ws.sum().clamp(min=1e-6)

        with torch.no_grad():
            top1 = (lg.argmax(-1) == lab) & valid
            # teacher-forced expected accept length proxy: sum_k prod_{j<=k} 1[top1_j]
            acc_len = torch.cumprod(top1.float(), 1).sum(1).mean()
        return main + selector_weight * sel_loss, {"main": main.item(), "selector": sel_loss.item(),
                                                    "top1": (top1.sum() / valid.sum().clamp(min=1)).item(),
                                                    "accept_len_proxy": acc_len.item()}
