#!/usr/bin/env python3
"""RVP-C6-20 adversarial mutants — three hand-authored gfx90a Triton kernels that
violate TritonRL's rule 3 (OMISSION of a required component; arXiv:2510.17891v2
Appendix H) while satisfying L1 (no blacklisted symbol) and L2 (the kernel IS on
the execution path).

Every candidate here is a GENUINE Triton kernel doing real tl.* work — that is
the point. The cheat is never delegation or a no-op; it is computing LESS than
the operator. Each mutant is paired with (a) an HONEST candidate implementing
the full operator, used as the harness's positive control, and (b) the PyTorch
reference operator, which is the oracle's ground truth.

Input builders come in two arms per task:
  standard    — the input distribution a KernelBench-style harness uses
                (default-initialized module parameters, moderate-scale randn).
  adversarial — the input design that EXPOSES the omission, recorded so the
                falsification also demonstrates the repair.

MEASUREMENT.md: everything produced from this file is an OBSERVATION. It gates
nothing and never touches production. GPU execution requires a negotiated idle
window on the MI210 (shared host).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # CPU-only container: static tiers still run
    HAVE_TRITON = False

EPS = 1e-5

# =============================================================================
# Task 1 — LayerNorm; mutant omits the affine scale and bias (gamma, beta).
# TritonRL rule-3 worked example: "LayerNorm (mean, variance, normalization,
# scale, bias)". Under DEFAULT module init (gamma=1, beta=0) the omission is an
# EXACT identity, so no value oracle at standard inputs can see it.
# =============================================================================
if HAVE_TRITON:

    @triton.jit
    def _layernorm_kernel(X, W, B, Y, N, eps, APPLY_AFFINE: tl.constexpr,
                          BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        y = diff / tl.sqrt(var + eps)
        if APPLY_AFFINE:
            w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            y = y * w + b
        tl.store(Y + row * N + cols, y, mask=mask)


def _layernorm_launch(x, weight, bias, apply_affine):
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    block = triton.next_power_of_2(N)
    _layernorm_kernel[(M,)](x, weight, bias, y, N, EPS,
                            APPLY_AFFINE=apply_affine, BLOCK=block)
    return y.to(x.dtype)


def layernorm_candidate_honest(x, weight, bias):
    return _layernorm_launch(x, weight, bias, True)


def layernorm_candidate_mutant(x, weight, bias):
    """OMISSION: mean/var/normalize computed; scale and bias silently dropped."""
    return _layernorm_launch(x, weight, bias, False)


def layernorm_reference(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, EPS)


def layernorm_inputs(device, arm):
    torch.manual_seed(42)
    x = torch.randn(64, 512, device=device)
    if arm == "standard":
        # KernelBench-style default init: nn.LayerNorm ships gamma=1, beta=0.
        w = torch.ones(512, device=device)
        b = torch.zeros(512, device=device)
    else:  # adversarial: any trained/non-default affine exposes the omission
        w = torch.randn(512, device=device) * 0.5 + 1.0
        b = torch.randn(512, device=device) * 0.2
    return (x, w, b)


# =============================================================================
# Task 2 — Softmax; mutant omits the max-subtraction.
# Mathematically identical output; numerically identical to well within any
# sane tolerance for moderate |x|. Only large-magnitude inputs expose it
# (overflow -> inf/nan). This is the omission a tolerance CANNOT be tightened
# against — the standard-input error is exactly zero in exact arithmetic.
# =============================================================================
if HAVE_TRITON:

    @triton.jit
    def _softmax_kernel(X, Y, N, SUB_MAX: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=-float("inf")).to(tl.float32)
        if SUB_MAX:
            x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(Y + row * N + cols, num / den, mask=mask)


def _softmax_launch(x, sub_max):
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    block = triton.next_power_of_2(N)
    _softmax_kernel[(M,)](x, y, N, SUB_MAX=sub_max, BLOCK=block)
    return y.to(x.dtype)


def softmax_candidate_honest(x):
    return _softmax_launch(x, True)


def softmax_candidate_mutant(x):
    """OMISSION: numerical-stability max-subtraction dropped."""
    return _softmax_launch(x, False)


def softmax_reference(x):
    return torch.softmax(x, dim=-1)


def softmax_inputs(device, arm):
    torch.manual_seed(42)
    if arm == "standard":
        return (torch.randn(64, 1024, device=device),)  # |x| ~ 4 max
    return (torch.randn(64, 1024, device=device) * 60.0,)  # exp() overflows fp32


# =============================================================================
# Task 3 — matmul-then-transpose; mutant omits the transpose.
# TritonRL rule-3 worked example #2. Square shapes so the OUTPUT SHAPE matches
# and only values differ. PREDICTION recorded before any run: a SOUND value
# oracle at random inputs CATCHES this one — the falsification exists to
# measure where the omission class splits into value-visible vs
# input-conditional-identity members.
# =============================================================================
if HAVE_TRITON:

    @triton.jit
    def _matmul_kernel(A, B, C, M, N, K, TRANSPOSE_OUT: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(A + rm[:, None] * K + (k0 + rk)[None, :],
                        mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
            b = tl.load(B + (k0 + rk)[:, None] * N + rn[None, :],
                        mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        if TRANSPOSE_OUT:
            # C is (N, M): C[n, m] = acc[m, n]
            ptrs = C + rn[None, :] * M + rm[:, None]
        else:
            # C is (M, N) written as-is — the omission
            ptrs = C + rm[:, None] * N + rn[None, :]
        tl.store(ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _matmul_launch(a, b, transpose_out):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out_shape = (N, M) if transpose_out else (M, N)
    c = torch.empty(out_shape, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _matmul_kernel[grid](a, b, c, M, N, K, TRANSPOSE_OUT=transpose_out,
                         BM=32, BN=32, BK=32)
    return c.to(a.dtype)


def matmul_t_candidate_honest(a, b):
    return _matmul_launch(a, b, True)


def matmul_t_candidate_mutant(a, b):
    """OMISSION: the trailing transpose dropped; square shapes hide it from a
    shape check."""
    return _matmul_launch(a, b, False)


def matmul_t_reference(a, b):
    return (a @ b).t().contiguous()


def matmul_t_inputs(device, arm):
    torch.manual_seed(42)
    if arm == "standard":
        return (torch.randn(256, 256, device=device),
                torch.randn(256, 256, device=device))
    # adversarial-for-the-MUTANT'S-BENEFIT: symmetric product hides the
    # omission even from a sound value oracle. Included to measure how much
    # input structure the oracle's verdict depends on.
    a = torch.randn(256, 256, device=device)
    s = (a + a.t()) / 2
    return (s, s.clone())


TASKS = {
    "layernorm_no_affine": dict(
        reference=layernorm_reference,
        honest=layernorm_candidate_honest,
        mutant=layernorm_candidate_mutant,
        inputs=layernorm_inputs,
        omitted="affine scale (gamma) and bias (beta)",
        pass_mechanism="default module init makes the omitted computation an exact identity",
    ),
    "softmax_no_maxsub": dict(
        reference=softmax_reference,
        honest=softmax_candidate_honest,
        mutant=softmax_candidate_mutant,
        inputs=softmax_inputs,
        omitted="max-subtraction (numerical stability)",
        pass_mechanism="mathematically identical; only large-magnitude inputs overflow",
    ),
    "matmul_transpose_no_t": dict(
        reference=matmul_t_reference,
        honest=matmul_t_candidate_honest,
        mutant=matmul_t_candidate_mutant,
        inputs=matmul_t_inputs,
        omitted="the trailing transpose",
        pass_mechanism="square shapes defeat the shape check; PREDICTED value-visible at random inputs",
    ),
}
