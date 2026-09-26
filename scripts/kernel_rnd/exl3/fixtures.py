"""Deterministic project-owned fixtures and fail-closed real-fixture admission.

Golden vectors use a separate bit-at-a-time decoder and explicit Hadamard sums.
The production oracle is never called while constructing golden data.
"""
from __future__ import annotations
import argparse
import json
import math
import struct
from pathlib import Path
from .contract import PACKING, SCHEMA, Refusal, canonical, digest, sha, load

FIXTURE_SCHEMA = 'epyc.exl3.fixture.v1'
REVISION = '6b84a21b6f1e5da3f291b9e1019061f0de788279'


def _h(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def _f(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _frombits(value):
    sign = -1 if value & 32768 else 1
    e, m = (value >> 10) & 31, value & 1023
    if e == 0:
        return sign * m * 2 ** -24
    if e == 31:
        raise Refusal('nonfinite fixture half')
    return sign * (1024 + m) * 2 ** (e - 25)


def golden_decode(window, cb):
    mult = 0xCBAC1FED if cb == 'mcg' else 0x83DCD12D
    p = window * mult % 2**32
    if cb == 'mcg':
        mixed = 0
        for bit in range(32):
            index = ((p >> bit & 1) * 4 + (0x8FFF8FFF >> bit & 1) * 2 + (0x3B603B60 >> bit & 1))
            mixed += ((0x6A >> index) & 1) << bit
        return _h(_frombits(mixed & 65535) + _frombits(mixed >> 16))
    if cb != 'mul1':
        raise Refusal('unknown codebook')
    total = sum((p // 256**i) % 256 for i in range(4))
    return _h(_frombits(0x6400 + total) * _frombits(0x1EEE) + _frombits(0xC931))


def golden_windows(data, k):
    words = struct.unpack('<' + 'I' * (len(data)//4), data)
    result = []
    for index in range(256):
        start = ((index+1)*k - 16) % (256*k)
        value = 0
        for offset in range(16):
            pos = (start + offset) % (256*k)
            value = value*2 + ((words[pos//32] >> (31-pos%32)) & 1)
        result.append(value)
    return result


def golden_tile(data, k, cb):
    states = golden_windows(data, k)
    out = [[0.0]*16 for _ in range(16)]
    # Forward scatter from warp lane and fragment, independent of inverse map.
    for lane in range(32):
        for fragment in range(8):
            row = 2*(lane%4) + fragment%2 + 8*((fragment//2)%2)
            col = 2*(lane//8) + (lane//4)%2 + 8*(fragment//4)
            out[row][col] = golden_decode(states[8*lane+fragment], cb)
    return states, out


def golden_h(values):
    # A separate interpreter for the normalized matrix and its ordered dot.
    import ctypes
    scale = ctypes.c_float(1/math.sqrt(128)).value
    accumulators = [0.0] * 128
    for j, value in enumerate(values):
        for i in range(128):
            sign = (-1)**((i & j).bit_count())
            accumulators[i] = ctypes.c_float(accumulators[i] + sign * scale * value).value
    return [_h(v) for v in accumulators]


def _pack(matrix, fmt):
    return b''.join(struct.pack(fmt, x) for row in matrix for x in row)


def golden(manifest, payload, activations):
    m = manifest['matrices'][0]
    ki, no = m['padded_shape']
    data = payload[m['tensors']['trellis']['path']]
    raw = [[0.0]*no for _ in range(ki)]
    first_states, first_tile = None, None
    for kb in range(ki//16):
        for nb in range(no//16):
            off = (kb*(no//16)+nb)*32*m['K']
            states, tile = golden_tile(data[off:off+32*m['K']], m['K'], m['codebook'])
            if first_states is None:
                first_states, first_tile = states, tile
            for r in range(16):
                raw[kb*16+r][nb*16:nb*16+16] = tile[r]
    vectors = {name: [_frombits(x[0]) for x in struct.iter_unpack('<H', payload[m['tensors'][name]['path']])] for name in ('suh', 'svh')}
    transformed = [row[:] for row in raw]
    for base in range(0, ki, 128):
        for c in range(no):
            h = golden_h([raw[base+r][c] for r in range(128)])
            for r in range(128):
                transformed[base+r][c] = _h(h[r] * vectors['suh'][base+r])
    for row in transformed:
        for base in range(0, no, 128):
            h = golden_h(row[base:base+128])
            row[base:base+128] = [_h(x*vectors['svh'][base+c]) for c,x in enumerate(h)]
    outputs = []
    for x in activations:
        row = []
        for c in range(m['shape'][1]):
            value = 0.0
            for r in range(m['shape'][0]):
                value = _f(value + _f(x[r])*transformed[r][c])
            bias = m['tensors']['bias']
            if bias is not None:
                value = _f(value + struct.unpack_from('<e',payload[bias['path']],c*2)[0])
            row.append(value)
        outputs.append(row)
    return {'packed_states': first_states, 'reconstructed_tile': first_tile,
            'raw_fp16_sha256': sha(_pack(raw, '<e')),
            'transformed_fp16_sha256': sha(_pack(transformed, '<e')),
            'suh': vectors['suh'], 'svh': vectors['svh'], 'operator_outputs': outputs}


def generate(directory, k, cb, *, make_golden=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Deterministic non-random integer pattern, includes varied boundaries and
    # distinct tiles. No downloaded/model-derived tensors are relabeled here.
    packed = bytes(((i*73 + (i//7)*29 + k*17) & 255) for i in range(128*128*k//8))
    suh = b''.join(struct.pack('<e', (-1 if i%3 else 1)*2**(-i%3-2)) for i in range(128))
    svh = b''.join(struct.pack('<e', (-1 if i%5 else 1)*2**(-i%2-1)) for i in range(128))
    payload = {'trellis.bin': packed, 'suh.bin': suh, 'svh.bin': svh}
    tensors = {name: {'path': name+'.bin', 'sha256': sha(payload[name+'.bin']), 'nbytes': len(payload[name+'.bin']),
                      'dtype': 'uint16_le' if name == 'trellis' else 'float16_le',
                      'shape': [8,8,16*k] if name == 'trellis' else [128]} for name in ('trellis', 'suh', 'svh')}
    bias = b''.join(struct.pack('<e', (i%7-3)/64) for i in range(128))
    payload['bias.bin'] = bias
    tensors['bias'] = dict(path='bias.bin',sha256=sha(bias),nbytes=256,dtype='float16_le',shape=[128])
    matrix = dict(id='projection', tensor_role='q', shape=[125,123], padded_shape=[128,128], order='input_output', K=k, rate_x2=2*k,
                  codebook=cb, packing=PACKING, hadamard={'input':128,'output':128,'normalization':'orthonormal','rounding':'fp16_rne_each_stage','algorithm':'normalized_matrix_ordered_fp32_fma'},
                  scaling='folded_in_suh_svh', padding='zero_input_crop_output_after_transform',
                  expert_domain={'kind':'dense','local':None,'global':None,'count':0}, tensors=tensors, source_sha256=sha(packed))
    manifest = dict(schema=SCHEMA, source={'repository':'https://github.com/turboderp-org/exllamav3','revision':REVISION,'kind':'synthetic'}, matrices=[matrix])
    manifest['artifact_sha256'] = digest(manifest)
    activations = [[((i*7+r*3)%17-8)/16 for i in range(125)] for r in range(3)]
    reference = golden(manifest, payload, activations) if make_golden else {}
    fixture = dict(schema=FIXTURE_SCHEMA, kind='synthetic', artifact_sha256=manifest['artifact_sha256'],
                   source_revision=REVISION, matrix_id='projection', activations=activations, reference=reference,
                   operator_path='materialized_weight_fp32_fma_v1', oracle='project-independent-direct-v1',
                   generator={'path':'scripts/kernel_rnd/exl3/fixtures.py','sha256':sha(Path(__file__).read_bytes())},
                   provenance_note='source pins upstream codec semantics; payloads are project-owned synthetic formulas')
    for name, data in payload.items():
        (directory/name).write_bytes(data)
    (directory/'manifest.json').write_bytes(canonical(manifest))
    (directory/'fixture.json').write_bytes(canonical(fixture))
    return directory


def admit_real(manifest_path, fixture_path):
    """Require real provenance and complete independently captured references.

    Importers must extract native tensor bytes from a revision-pinned checkpoint,
    preserve checkpoint/source digests, and supply the actual read-set receipt.
    This validator never generates expected output from the implementation.
    """
    from .contract import read_json
    manifest, payload = load(manifest_path)
    fixture = read_json(fixture_path)
    if manifest['source']['kind'] != 'real_weight' or fixture.get('kind') != 'real_weight' or fixture.get('schema') != FIXTURE_SCHEMA:
        raise Refusal('mandatory real-weight fixture is absent')
    if fixture.get('artifact_sha256') != manifest['artifact_sha256'] or fixture.get('source_revision') != manifest['source']['revision']:
        raise Refusal('real fixture source/artifact revision mismatch')
    if fixture.get('oracle') in {None, '', 'project-independent-direct-v1'}:
        raise Refusal('real fixture needs independently captured oracle identity')
    reference = fixture.get('reference', {})
    if not {'packed_states','reconstructed_tile','raw_fp16_sha256','transformed_fp16_sha256','suh','svh','operator_outputs'} <= set(reference):
        raise Refusal('real fixture is missing a required parity stage')
    if not fixture.get('activations') or not fixture.get('provenance'):
        raise Refusal('real fixture activation/provenance read set required')
    for desc in fixture['provenance']:
        path = Path(fixture_path).parent / desc['path']
        if not path.is_file() or sha(path.read_bytes()) != desc['sha256']:
            raise Refusal('real source provenance unavailable or drifted')
    receipt_paths = [Path(fixture_path).parent / d['path'] for d in fixture['provenance']]
    receipts = [read_json(p) for p in receipt_paths]
    bound = False
    for receipt in receipts:
        remote = receipt.get('model_revision_verification') or {}
        comparisons = remote.get('comparisons', [])
        if (remote.get('revision') == manifest['source']['revision'] and
                manifest['source']['repository'] == 'https://huggingface.co/' + remote.get('repo', '') and
                len(comparisons) >= 3 and all(c.get('equal_retained_capture') is True for c in comparisons) and
                receipt.get('canonical_native_sha256') == {name: sha(data) for name, data in payload.items()}):
            bound = True
    if not bound:
        raise Refusal('real fixture requires byte-verified immutable model revision provenance')
    return manifest, payload, fixture


def require_real_suite(paths):
    seen = set()
    for manifest, fixture in paths:
        m, _, _ = admit_real(manifest, fixture)
        for matrix in m['matrices']:
            if matrix['codebook'] == 'mcg' and matrix['K'] != 4:
                raise Refusal('first real MCG fixture must be uniform K4')
            seen.add(matrix['codebook'])
    if seen != {'mul1','mcg'}:
        raise Refusal('G1 requires revision-pinned real MUL1 and uniform-K4 MCG fixtures; no skip permitted')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generate-synthetic', type=Path)
    args = p.parse_args()
    if args.generate_synthetic:
        for cb in ('mul1', 'mcg'):
            for k in range(1,9):
                generate(args.generate_synthetic/f'{cb}-k{k}', k, cb)


if __name__ == '__main__':
    main()
