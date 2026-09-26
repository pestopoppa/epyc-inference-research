"""Independent scalar interpretation of native EXL3; stdlib only.

No donor source is imported. Codebook results are IEEE binary16 RNE. Operator
reference is materialized-weight, ordered FP32 accumulation, not fused runtime.
"""
from __future__ import annotations
import math
import struct
from .contract import Refusal, validate, sha


def half(value):
    try:
        return struct.unpack('<e', struct.pack('<e', value))[0]
    except OverflowError as exc:
        raise Refusal('binary16 reference overflow') from exc


def half_bits(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 65535))[0]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def decode(window, codebook):
    if type(window) is not int or not 0 <= window < 65536:
        raise Refusal('window is not uint16')
    if codebook == 'mcg':
        product = (window * 0xCBAC1FED) & 0xFFFFFFFF
        # LUT 0x6a is c XOR (a AND b), derived bitwise from its truth table.
        mixed = (product & 0x8FFF8FFF) ^ 0x3B603B60
        return half(half_bits(mixed) + half_bits(mixed >> 16))
    if codebook == 'mul1':
        product = (window * 0x83DCD12D) & 0xFFFFFFFF
        byte_sum = sum(product.to_bytes(4, 'little'))
        # One half FMA, with a single final RNE rounding. An affine Q8
        # approximation (byte_sum - 510)*constant is NOT this codebook.
        return half(half_bits(0x6400 + byte_sum) * half_bits(0x1EEE) + half_bits(0xC931))
    raise Refusal('unknown codebook')


def state_index(row, column):
    if not 0 <= row < 16 or not 0 <= column < 16:
        raise Refusal('tile coordinate out of bounds')
    return ((column % 8) * 4 + (row % 8) // 2) * 8 + row % 2 + (row // 8) * 2 + (column // 8) * 4


def windows(packed, k):
    if type(k) is not int or not 1 <= k <= 8 or len(packed) != 32 * k:
        raise Refusal('packed tile must contain 16*K little-endian uint16s')
    # File bytes are little endian uint32; the circular stream walks each
    # uint32 from most to least significant bit, then the next uint32.
    stream = ''.join(f'{word:032b}' for (word,) in struct.iter_unpack('<I', packed))
    return [
        int((stream + stream)[start:start + 16], 2)
        for start in (((i + 257) * k - 16) % len(stream) for i in range(256))]


def tile(packed, k, codebook):
    values = [decode(w, codebook) for w in windows(packed, k)]
    return [[values[state_index(r, c)] for c in range(16)] for r in range(16)]


def hadamard(values):
    """Normalized H128 matrix, ascending-index FP32 FMA, final half RNE.

    Normalization is in each matrix coefficient, not after a butterfly. This
    pins rounding order for a materialized reference independent of fused paths.
    """
    if len(values) != 128:
        raise Refusal('only H128 is admitted')
    scale = f32(1 / math.sqrt(128))
    out = []
    for row in range(128):
        acc = 0.0
        for column, value in enumerate(values):
            coefficient = -scale if (row & column).bit_count() % 2 else scale
            acc = f32(acc + coefficient * value)
        out.append(half(acc))
    return out


def reconstruct(manifest, payload, matrix_id):
    validate(manifest)
    matches = [m for m in manifest['matrices'] if m['id'] == matrix_id]
    if len(matches) != 1:
        raise Refusal('unknown logical matrix')
    m = matches[0]
    # Validate *all* bytes before allocating any reconstructed output.
    for mat in manifest['matrices']:
        for d in mat['tensors'].values():
            if d is not None and (d['path'] not in payload or len(payload[d['path']]) != d['nbytes'] or sha(payload[d['path']]) != d['sha256']):
                raise Refusal('payload length/digest mismatch')
    vectors = {}
    for name in ('suh', 'svh', 'bias'):
        d = m['tensors'][name]
        vectors[name] = None if d is None else [x[0] for x in struct.iter_unpack('<e', payload[d['path']])]
        if vectors[name] is not None and not all(math.isfinite(x) for x in vectors[name]):
            raise Refusal('nonfinite transform or bias')
    ki, no = m['padded_shape']
    raw = [[0.0] * no for _ in range(ki)]
    packed = payload[m['tensors']['trellis']['path']]
    size = 32 * m['K']
    for kb in range(ki // 16):
        for nb in range(no // 16):
            offset = (kb * (no // 16) + nb) * size
            block = tile(packed[offset:offset+size], m['K'], m['codebook'])
            for r in range(16):
                raw[kb*16+r][nb*16:nb*16+16] = block[r]
    weight = [row[:] for row in raw]
    for base in range(0, ki, 128):
        for c in range(no):
            h = hadamard([raw[base+r][c] for r in range(128)])
            for r in range(128):
                weight[base+r][c] = half(h[r] * vectors['suh'][base+r])
    for r in range(ki):
        for base in range(0, no, 128):
            h = hadamard(weight[r][base:base+128])
            weight[r][base:base+128] = [half(x * vectors['svh'][base+c]) for c, x in enumerate(h)]
    return raw, weight, vectors


def gemm(manifest, payload, matrix_id, activations):
    """y = FP32(x @ materialized_W) + bias, fixed ascending input order.

    The contract names the exact reference arithmetic. Fused activation-domain
    transforms require an explicit tolerance and a different path identity.
    """
    validate(manifest)
    m = next((m for m in manifest['matrices'] if m['id'] == matrix_id), None)
    if m is None or not isinstance(activations, (list, tuple)) or not activations:
        raise Refusal('invalid activation matrix')
    if len(activations) > 256 or len(activations) * m['padded_shape'][1] > 1_048_576:
        raise Refusal('activation/output capacity exceeded')
    for x in activations:
        if not isinstance(x, (list, tuple)) or len(x) != m['shape'][0] or not all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 65504 for v in x):
            raise Refusal('invalid or nonfinite activation matrix')
    _, weight, vectors = reconstruct(manifest, payload, matrix_id)
    out = []
    for x in activations:
        row = []
        for c in range(m['shape'][1]):
            acc = 0.0
            for r, value in enumerate(x):
                acc = f32(acc + f32(value) * weight[r][c])
            row.append(f32(acc + (vectors['bias'][c] if vectors['bias'] else 0.0)))
        out.append(row)
    return out


def gemv(manifest, payload, matrix_id, activation):
    return gemm(manifest, payload, matrix_id, [activation])[0]
