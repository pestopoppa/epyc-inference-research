"""Bounded GGUF metadata/tensor-header preflight; never reads tensor payloads."""
import re
import struct

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
LIMIT = 64 * 1024 * 1024


def inspect(path):
    with open(path, 'rb') as f:
        def read(n):
            if n < 0 or f.tell() + n > LIMIT:
                raise ValueError('GGUF header exceeds bound')
            b = f.read(n)
            if len(b) != n:
                raise ValueError('truncated GGUF header')
            return b
        def u32(): return struct.unpack('<I', read(4))[0]
        def u64(): return struct.unpack('<Q', read(8))[0]
        def string(): return read(u64()).decode('utf-8')
        def value(t, capture=False):
            if t in SIZES:
                raw = read(SIZES[t])
                return int.from_bytes(raw, 'little') if capture and t in (0, 2, 4, 10) else None
            if t == 8:
                return string()
            if t == 9:
                element, count = u32(), u64()
                if count > LIMIT:
                    raise ValueError('oversize array')
                for _ in range(count): value(element)
                return None
            raise ValueError('unknown GGUF type')
        if read(4) != b'GGUF' or u32() not in (2, 3):
            raise ValueError('unsupported GGUF')
        tensors, count = u64(), u64()
        if tensors > 100000 or count > 100000:
            raise ValueError('oversize header')
        meta = {}
        for _ in range(count):
            key, t = string(), u32()
            meta[key] = value(t, key.endswith(('.embedding_length', '.block_count')))
        shapes = []
        for _ in range(tensors):
            name, n = string(), u32()
            if n > 4: raise ValueError('invalid tensor dimensions')
            dims = [u64() for _ in range(n)]
            typ, offset = u32(), u64()
            shapes.append((name, dims, typ, offset))
        return meta, shapes


def verify(model, vector):
    meta, _ = inspect(model)
    vm, shapes = inspect(vector)
    arch = meta.get('general.architecture')
    if arch != 'qwen2': raise ValueError(f'unexpected architecture: {arch}')
    width = meta.get(f'{arch}.embedding_length')
    layers = meta.get(f'{arch}.block_count')
    if not width or not layers or not shapes:
        raise ValueError('missing model dimensions or vector tensors')
    seen = set()
    for name, dims, typ, _ in shapes:
        match = re.fullmatch(r'direction\.([1-9][0-9]*)', name)
        if not match or int(match[1]) > layers or int(match[1]) in seen or dims != [width] or typ != 0:
            raise ValueError(f'incompatible vector tensor: {name} {dims} type={typ}')
        seen.add(int(match[1]))
    return {'architecture': arch, 'embedding_length': width, 'block_count': layers,
            'vector_layers': sorted(seen), 'vector_architecture': vm.get('general.architecture')}
