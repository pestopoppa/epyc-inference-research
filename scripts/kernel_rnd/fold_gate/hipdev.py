#!/usr/bin/env python3
"""hipdev.py - deterministic comparison of the gfx device code objects embedded in a .hip_fatbin.

Why not "exact .text equality": device code embeds __FILE__ (assert / no_device_code messages).  The string
lives in .rodata, is reached pc-relatively, and its LENGTH is constant-folded into the code (strlen unrolled or looped
depending on the path length).  Two builds from different worktree roots therefore differ in .text even when the source
is identical.  Making the bytes equal needs `-ffile-prefix-map` / `-fmacro-prefix-map` at build time (README).

What this module does instead (no opcode allow-lists, no pattern matching on instruction classes):

  1. Parse every `__CLANG_OFFLOAD_BUNDLE__` and take the gfx code object of each.
  2. Per code object, disassemble (llvm-objdump) and split into functions at the symbol labels.
  3. Resolve every pc-relative address formed by `s_getpc_b64` + `s_add_u32 imm` (target = address of the s_add_u32 + imm: the relocation addend +4 cancels the literal's own offset) to its target in
     the code object's own .rodata, and replace the raw immediate by WHAT IT POINTS AT (C string content with the
     build root replaced by <ROOT>, or a hash of the 16 bytes).  Immediates then compare equal iff the referenced
     data is equal, independent of layout.
  4. Function classes (exact, structural):
       IDENTICAL       resolved bodies equal line for line, branch offsets included
       PATH-DEPENDENT  bodies differ AND both sides reference >= 1 build-root path string AND those path strings form the same SET
                       modulo root (strlen folding changes how many times the pointer is formed).  Their instruction bytes differ only because strlen/offset codegen depends on the
                       path length; they are COUNTED and reported, not instruction-verified.
       OTHER           anything else -> FAIL.  A kernel (function with a `<name>.kd` descriptor) is never path-dependent:
                       a __FILE__ assert lives in a noinline helper (`no_device_code`), kernels must be IDENTICAL.
  5. .rodata: split into NUL-separated entries; path strings compared modulo root, all other entries exactly.
     AMDGPU kernel descriptors (symbol `*.kd`, 64 bytes) compared with the entry-offset field (bytes 16..24) masked.
  6. Function set (names) must be equal on both sides.

PATH NORMALISATION (v4): a candidate built with the CH-28d prefix-map recipe carries `ggml/src/x.cu` where the baseline carries
`<root>/ggml/src/x.cu`.  Both forms are one "build-root path" (REL_RX / ROOT_RX) and normalise to `<ROOT>/ggml/src/x.cu`; only a
string that is equal after that normalisation is explained, so a real path/name change still fails.  Such a pair reports
`path_touched` and the lib is classified EXPLAINED-PATH by libcmp.

When both trees were built with a path-independent __FILE__ (prefix-map recipe) every function is IDENTICAL and the
result is `exact`.  Equal-length roots likewise produce zero PATH-DEPENDENT functions.
"""
import hashlib
import os
import re
import struct
import subprocess
import tempfile

BUNDLE_MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
ROCM_LLVM = os.environ.get('ROCM_LLVM_BIN', '/opt/rocm/llvm/bin')
ROOT_RX = re.compile(rb'^(/.*?)/(ggml|src|common|tools|include|vendor|examples)/')
# a source path from a build with `-ffile-prefix-map=<root>=.`: no root left, `ggml/src/x.cu` (clang) or `./ggml/src/x.cu` (gcc)
REL_RX = re.compile(rb'^(?:\./)?((?:ggml|src|common|tools|include|vendor|examples)/[\w./+\-]*\.(?:cu|cuh|hip|c|cc|cpp|h|hpp))$')
LABEL = re.compile(r'^([0-9a-f]{16}) <(.*)>:$')
INSN_ADDR = re.compile(r'//\s*([0-9A-F]{12}):')


def device_objects(blob):
    """-> list (one per bundle) of the gfx code object bytes (None if the bundle has no gfx entry)."""
    out = []
    for m in re.finditer(re.escape(BUNDLE_MAGIC), blob):
        p = m.start()
        n, = struct.unpack_from('<Q', blob, p + 24)
        q = p + 32
        co = None
        for _ in range(n):
            off, size, idl = struct.unpack_from('<QQQ', blob, q)
            ident = blob[q + 24:q + 24 + idl].decode()
            q += 24 + idl
            if 'gfx' in ident:
                co = blob[p + off:p + off + size]
        out.append(co)
    return out


def _run(tool, *args):
    return subprocess.run([os.path.join(ROCM_LLVM, tool), *args], capture_output=True, text=True, check=True).stdout


def _sections(path):
    r = {}
    for l in _run('llvm-readelf', '-SW', path).splitlines():
        m = re.match(r'\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', l)
        if m:
            r[m.group(1)] = (int(m.group(3), 16), int(m.group(4), 16), int(m.group(5), 16))  # addr, off, size
    return r


def _symbols(path):
    """{name: (addr, size)} for object symbols (kernel descriptors etc.)"""
    out = {}
    for l in _run('llvm-readelf', '-sW', path).splitlines():
        f = l.split()
        if len(f) >= 8 and f[3] == 'OBJECT':
            try:
                out[f[7]] = (int(f[1], 16), int(f[2], 0))
            except ValueError:
                pass
    return out


def build_root(strings):
    """root prefix common to the path strings of one code object (None if there are none)"""
    roots = {m.group(1) for s in strings if (m := ROOT_RX.match(s))}
    if len(roots) > 1:   # keep the shortest consistent root; paths of one tree share it
        return sorted(roots, key=len)[0]
    return next(iter(roots), None)


class CodeObject:
    def __init__(self, data, tmpdir, tag):
        self.path = os.path.join(tmpdir, tag + '.co')
        with open(self.path, 'wb') as f:
            f.write(data)
        self.data = data
        sec = _sections(self.path)
        a, o, s = sec['.rodata']
        self.ro_addr, self.ro_off, self.ro = a, o, data[o:o + s]
        self.syms = _symbols(self.path)
        ents = [e for e in self.ro.split(b'\0') if len(e) >= 4 and all(32 <= c < 127 for c in e)]
        self.paths = [e for e in ents if ROOT_RX.match(e) or REL_RX.match(e)]
        self.root = build_root(self.paths)
        self.funcs = self._funcs()

    def _norm_str(self, s):
        m = REL_RX.match(s)
        if m:
            return b'<ROOT>/' + m.group(1)
        return s.replace(self.root, b'<ROOT>') if self.root else s

    def _target_token(self, addr):
        i = addr - self.ro_addr
        if not 0 <= i < len(self.ro):
            return None
        e = self.ro.find(b'\0', i)
        s = self.ro[i:e if e >= 0 else len(self.ro)]
        if len(s) >= 4 and all(32 <= c < 127 for c in s):
            return 'S:' + self._norm_str(s).decode(), ('P' if (ROOT_RX.match(s) or REL_RX.match(s)) else 'S')
        return 'D:' + hashlib.sha1(self.ro[i:i + 16]).hexdigest()[:12], 'D'

    def _funcs(self):
        txt = _run('llvm-objdump', '-d', self.path).splitlines()
        starts = sorted((int(m.group(1), 16), m.group(2)) for m in (LABEL.match(l) for l in txt) if m)
        funcs, cur, order = {}, None, []
        pend = None   # (reg, getpc_result_addr) awaiting its s_add_u32
        for l in txt:
            m = LABEL.match(l)
            if m:
                cur = m.group(2)
                funcs[cur] = dict(lines=[], paths=[], raw=[])
                order.append(cur)
                pend = None
                continue
            if cur is None or not l.startswith('\t'):
                continue
            am = INSN_ADDR.search(l)
            body = re.sub(r'\s*//.*$', '', l).strip()
            addr = int(am.group(1), 16) if am else None
            # branch targets of the form <sym+off> are layout; plain relative offsets are kept exact
            f = funcs[cur]
            if body.startswith('s_getpc_b64') and addr is not None:
                pend = (re.search(r's\[(\d+):\d+\]', body).group(1), addr + 4)
                f['lines'].append(body)
                continue
            mm = re.match(r's_add_u32 s(\d+), s(\d+), (0x[0-9a-f]+)$', body)
            if mm and pend and mm.group(1) == mm.group(2) == pend[0] and addr == pend[1]:
                imm = int(mm.group(3), 16)
                imm = imm - (1 << 32) if imm >= 1 << 31 else imm
                tok = self._target_token(addr + imm)
                if tok:
                    f['lines'].append(f's_add_u32 s{mm.group(1)}, s{mm.group(1)}, @{tok[0]}')
                    if tok[1] == 'P':
                        f['paths'].append(tok[0])
                else:   # call/jump target inside .text: name the callee so calling a different function is visible
                    t = addr + imm
                    cand = [(a_, n_) for a_, n_ in starts if a_ <= t]
                    ref = f'F:{cand[-1][1]}+{t - cand[-1][0]}' if cand else f'X:{imm}'
                    f['lines'].append(f's_add_u32 s{mm.group(1)}, s{mm.group(1)}, @{ref}')
                pend = None
                continue
            f['lines'].append(body)
        return funcs

    def descriptors(self):
        out = {}
        for n, (a, sz) in self.syms.items():
            if n.endswith('.kd') and sz == 64:
                i = a - self.ro_addr
                b = bytearray(self.ro[i:i + 64])
                b[16:24] = b'\0' * 8
                out[n] = bytes(b)
        return out

    def rodata_entries(self):
        kd_ranges = [(a - self.ro_addr, a - self.ro_addr + sz) for n, (a, sz) in self.syms.items() if n.endswith('.kd') and sz == 64]
        buf = bytearray(self.ro)
        for s, e in kd_ranges:
            buf[s:e] = b'\0' * (e - s)
        return sorted(self._norm_str(e) for e in bytes(buf).split(b'\0') if e)


def compare_object(a: CodeObject, b: CodeObject):
    """-> dict(identical, path_dependent, other, rodata_ok, notes[list])"""
    res = dict(identical=0, path_dependent=0, other=[], rodata_ok=True, notes=[], path_touched=False)
    if set(a.funcs) != set(b.funcs):
        res['other'].append('function set differs: ' + str(sorted(set(a.funcs) ^ set(b.funcs))[:4]))
        return res
    for n in a.funcs:
        fa, fb = a.funcs[n], b.funcs[n]
        if fa['lines'] == fb['lines']:
            res['identical'] += 1
        elif n + '.kd' in a.syms or n + '.kd' in b.syms:
            res['other'].append(f'{n}: KERNEL body differs (kernels must be identical; only non-kernel helpers may be path-dependent)')
        elif fa['paths'] and set(fa['paths']) == set(fb['paths']):
            res['path_dependent'] += 1
        else:
            res['other'].append(f'{n}: body differs and is not explained by a build-root path reference '
                                f'(paths a={len(fa["paths"])} b={len(fb["paths"])})')
    if a.descriptors() != b.descriptors():
        res['rodata_ok'] = False
        res['notes'].append('kernel descriptors differ beyond the entry offset')
    if a.rodata_entries() != b.rodata_entries():
        res['rodata_ok'] = False
        res['notes'].append('.rodata strings differ modulo build root')
    elif a.paths != b.paths:
        res['path_touched'] = True   # raw path strings differ (other root / prefix-mapped) but are equal modulo root
    return res


def _pair(args):
    k, x, y, d = args
    if x is None or y is None:
        return k, 'nogfx', None
    if x == y:
        return k, 'exact', None
    wd = tempfile.mkdtemp(prefix='p%d_' % k, dir=d)
    try:
        return k, 'cmp', compare_object(CodeObject(x, wd, 'a'), CodeObject(y, wd, 'b'))
    finally:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)


def compare_fatbins(blob_a, blob_b, jobs=None):
    """-> (ok, summary_line, detail_lines).  Bundles are compared in parallel (llvm-objdump dominates)."""
    return compare_fatbins_ex(blob_a, blob_b, jobs)[:3]


def compare_fatbins_ex(blob_a, blob_b, jobs=None):
    """-> (ok, summary_line, detail_lines, path_touched)  path_touched: some pair differed only in build-root path spelling."""
    from concurrent.futures import ThreadPoolExecutor
    oa, ob = device_objects(blob_a), device_objects(blob_b)
    if len(oa) != len(ob):
        return False, f'.hip_fatbin: {len(oa)} vs {len(ob)} device bundles', [], False
    d = tempfile.mkdtemp(prefix='fold_gate_hip_', dir=os.environ.get('FOLD_GATE_TMP'))
    n_exact = n_pd_obj = pd_funcs = 0
    bad = []
    touched = False
    try:
        with ThreadPoolExecutor(jobs or min(8, os.cpu_count() or 2)) as ex:
            results = list(ex.map(_pair, [(k, x, y, d) for k, (x, y) in enumerate(zip(oa, ob))]))
        for k, kind, r in results:
            if kind == 'nogfx':
                bad.append(f'bundle {k}: no gfx object')
            elif kind == 'exact':
                n_exact += 1
            elif r['other'] or not r['rodata_ok']:
                bad.append(f'bundle {k}: ' + '; '.join(r['other'][:2] + r['notes']))
            elif r['path_dependent']:
                n_pd_obj += 1
                pd_funcs += r['path_dependent']
                touched = True
            else:
                n_exact += 1
                touched = touched or r['path_touched']
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    mode = 'exact' if not n_pd_obj and not bad else 'path-dependent-codegen'
    line = (f'.hip_fatbin: {len(oa)} gfx code objects: {n_exact} identical (resolved-operand disassembly, rodata, kernel descriptors), '
            f'{n_pd_obj} differ only in {pd_funcs} path-length-dependent function(s) (mode={mode}), {len(bad)} other')
    return (not bad), line, bad[:6], touched or bool(n_pd_obj)
