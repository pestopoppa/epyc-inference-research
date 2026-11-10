"""Backend-matched reference for the C6 native MUL_MAT harness on ROCm/CUDA.

Why this exists (c6-native-residency-20261109-r2, 2026-11-10): the harness oracle
(`c6_mul_mat_harness.cpp`) builds both of its references on the CPU path, which
quantizes the activations to Q8_K. The ROCm/CUDA MMVQ path quantizes them to Q8_1
instead and, for K-quants with a min term (q4_K), uses the *float* block sum
`s8 = half(sum x)` for the min term while the scale term uses the quantized values.
For one 256-wide q4_K row that arithmetic differs from the CPU's by ~0.03-0.09 per
output, so an elementwise atol=rtol=0.01 gate against the CPU reference reports
matched_ratio 0.625 on a bit-exact, upstream-identical kernel (NMSE 1.2e-4, well
inside test-backend-ops' 5e-4 MUL_MAT bound). Emulating the GPU arithmetic here
reproduces the production v10 output to 1.6e-6.

Kernel generations differ: v10 (upstream-identical) uses the float block sum for
the q4_K min term (`min_sum="float"`); the champion line from C9 on carries
C6-Q4-MMVQ-FIX (0bccaa85d, 9e9d7f291), which uses the exact quantized block sum
(`min_sum="quantized"`). Pick the mode of the binary under test. Even a
consistent kernel scores only ~0.69 against the CPU reference at atol=rtol=0.01
on the r2 inputs (NMSE 5e-5), so the elementwise CPU-reference gate is
mis-specified for any cross-quantization comparison; use this emulation or the
test-backend-ops NMSE bound instead.

Stdlib only: row jobs run under /usr/bin/python3 without numpy.
"""
from __future__ import annotations

import math
import struct
from typing import Sequence

QK_K = 256
QK8_1 = 32
Q4_K_BLOCK_BYTES = 144  # half d, half dmin, 12 scale bytes, 128 nibble bytes


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _f16(x: float) -> float:
    return struct.unpack("<e", struct.pack("<e", x))[0]


def _roundf(x: float) -> float:
    """C roundf: half away from zero."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


def _scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def quantize_q8_1_gpu(x: Sequence[float]) -> list[tuple[float, float, list[int]]]:
    """quantize_q8_1 as the HIP/CUDA kernel computes it: per 32 values
    d = amax/127 (f32), q = roundf(x/d), ds = (half(d), half(float sum))."""
    if len(x) % QK8_1:
        raise ValueError("activation length is not a multiple of 32")
    blocks = []
    for start in range(0, len(x), QK8_1):
        xs = [_f32(v) for v in x[start:start + QK8_1]]
        amax = max(abs(v) for v in xs)
        d = _f32(amax / 127.0)
        qs = [0 if amax == 0.0 else int(_roundf(_f32(v / d))) for v in xs]
        blocks.append((_f16(d), _f16(_f32(math.fsum(xs))), qs))
    return blocks


def q4_k_mmvq_reference(weights: bytes, activations: Sequence[float],
                        m: int, k: int, n: int, *, min_sum: str = "float") -> list[float]:
    """f64 emulation of ROCm/CUDA vec_dot_q4_K_q8_1 for an (m x k) q4_K matrix
    times n f32 activation columns; output column-major [col*m + row]."""
    if min_sum not in ("float", "quantized"):
        raise ValueError("min_sum must be 'float' (v10) or 'quantized' (C9+)")
    if k % QK_K:
        raise ValueError("q4_K k must be a multiple of 256")
    row_bytes = (k // QK_K) * Q4_K_BLOCK_BYTES
    if len(weights) != row_bytes * m or len(activations) != k * n:
        raise ValueError("q4_K weights/activations do not match the shape")
    out = [0.0] * (m * n)
    for col in range(n):
        q8 = quantize_q8_1_gpu(activations[col * k:(col + 1) * k])
        for row in range(m):
            total = 0.0
            for blk in range(k // QK_K):
                b = weights[row * row_bytes + blk * Q4_K_BLOCK_BYTES:
                            row * row_bytes + (blk + 1) * Q4_K_BLOCK_BYTES]
                d, dmin = struct.unpack("<ee", b[:4])
                scales, qs = b[4:16], b[16:]
                sum_d = sum_m = 0.0
                for sub in range(8):
                    sc, mn = _scale_min_k4(sub, scales)
                    chunk = qs[(sub // 2) * 32:(sub // 2) * 32 + 32]
                    nib = [(v & 0xF) if sub % 2 == 0 else (v >> 4) for v in chunk]
                    d8, s8, q = q8[blk * 8 + sub]
                    sum_d += d8 * sc * sum(a * c for a, c in zip(nib, q))
                    if min_sum == "quantized":
                        s8 = d8 * sum(q)
                    sum_m += s8 * mn
                total += d * sum_d - dmin * sum_m
            out[col * m + row] = total
    return out


def matched_ratio(reference: Sequence[float], candidate: Sequence[float],
                  atol: float, rtol: float) -> float:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("reference/candidate length mismatch")
    ok = sum(math.isfinite(c) and abs(c - r) <= atol + rtol * abs(r)
             for r, c in zip(reference, candidate))
    return ok / len(reference)


def nmse(reference: Sequence[float], candidate: Sequence[float]) -> float:
    """test-backend-ops normalized MSE: sum (c-r)^2 / sum r^2."""
    num = math.fsum((c - r) ** 2 for r, c in zip(reference, candidate))
    den = math.fsum(r * r for r in reference)
    return math.inf if den == 0.0 else num / den
