#!/usr/bin/env python3
"""libcmp.py BASE_LIB NEW_LIB -> one verdict line on stdout + detail lines (prefixed '    ').
       [--ui-source-identical]
Verdicts: SAME | EXPLAINED | EXPLAINED-UI | UNEXPLAINED | CODE-DIFF.   Read-only on both inputs.

--ui-source-identical : the caller PROVED `git diff <base> <cand> -- tools/ui tools/server` is empty (fold_gate.py does).
  Only then may a CODE-DIFF confined to the web-UI static initialiser be classified EXPLAINED-UI (class 4 below).

Classification (fold gate v3; v2 = CH-28 2026-10-23):
  1. sha256 identical                                   -> SAME
  2. executable sections (.init .plt* .text .fini ...): normalised disassembly must be identical
     (addresses, rip displacements, branch/call targets stripped; opcodes and immediates kept).
     If not, the differing functions are listed                                -> CODE-DIFF
  3. every non-code section must be one of:
       layout-derived (relocs, symbol tables, eh_frame, got, dynamic, init arrays, data.rel.ro, ...)
         -> allowed only if its SIZE is unchanged (content shifts with layout);
       .note.gnu.build-id, .comment, .debug_*                  -> allowed
       .rodata -> printable strings present on one side only must all be build-info
         (commit, build number, worktree/build-dir path); after removing those strings and collapsing
         NUL padding the remaining bytes must be IDENTICAL.
     .hip_fatbin -> hipdev.py: deterministic device-code check (no opcode allow-list; see that module)
     Anything else                                                             -> UNEXPLAINED
  4. EXPLAINED-UI (needs --ui-source-identical): the ONLY differing function(s) are the `_GLOBAL__sub_I_ui.cpp`
     static initialiser, every differing instruction is a `movq $imm,<rip-rel>` immediate (the gzip sizes of the embedded
     assets), and the rest of the lib passes classes 2-3 except that .rodata may differ (it holds the gzip bytes of the
     UI bundle, regenerated non-deterministically by `npm run build`).  Deterministic fix: pin tools/ui/dist (README).
"""
import sys, os, re, subprocess, hashlib, collections, array, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hipdev

TOKENS = re.compile(rb'49550eda5|141e4bcf5|1041[0-9]|worktrees|ch28|lbh|nemotron|2026102[0-9]|kernels/builds|/mnt/raid0')
LAYOUT = re.compile(r'^\.(rela\..*|rel\..*|dynsym|dynstr|symtab|strtab|shstrtab|eh_frame|eh_frame_hdr|gcc_except_table|sframe|'
                    r'got|got\.plt|dynamic|init_array|fini_array|data\.rel\.ro|data|bss|tbss|tdata|gnu\.hash|gnu\.version|gnu\.version_r|hash)$')
FREE = re.compile(r'^(\.note\.gnu\.build-id|\.comment|\.debug_.*|\.gnu_debuglink)$')
EXEC_FLAG = 'X'


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def sections(p):
    r = {}
    for l in sh(['readelf', '-SW', p]).splitlines():
        m = re.match(r'\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+\S+\s+(\S*)\s', l)
        if m:
            r[m.group(2)] = dict(type=m.group(3), addr=int(m.group(4), 16), off=int(m.group(5), 16), size=int(m.group(6), 16), flags=m.group(7))
    return r


def sec_bytes(p, s):
    if s['type'] == 'NOBITS':
        return b''
    with open(p, 'rb') as f:
        f.seek(s['off'])
        return f.read(s['size'])


NORM = [
    (re.compile(r'^ *[0-9a-f]+:\t?'), ''),
    (re.compile(r'0x[0-9a-f]+\(%rip\)'), 'RIP'),
    (re.compile(r'# [0-9a-f]+ <[^>]*>'), '# A'),
    (re.compile(r'<[^>]*>'), '<S>'),
    (re.compile(r'^((?:call|jmp|j[a-z]+)\w* +)[0-9a-f]{3,}( <S>)?\s*$'), r'\1T'),
    (re.compile(r'^([a-z0-9]+ +)[0-9a-f]{3,} <S>'), r'\1T <S>'),
]
HDR = re.compile(r'^[0-9a-f]+ <(.*)>:$')


def funcs(p):
    """normalised disassembly of all executable sections -> {func_name: [lines]} (order kept as list of names too)"""
    out = sh(['objdump', '-d', '-w', '--no-show-raw-insn', p])
    d = collections.OrderedDict()
    cur = '<none>'
    d[cur] = []
    for l in out.splitlines():
        if not l or 'file format' in l or l.startswith('Disassembly of section'):
            continue
        m = HDR.match(l)
        if m:
            cur = m.group(1)
            d.setdefault(cur, [])
            continue
        for rx, rep in NORM:
            l = rx.sub(rep, l)
        d[cur].append(l.rstrip())
    return d


def code_verdict(b, n):
    fb, fn = funcs(b), funcs(n)
    hb = {k: hashlib.sha256('\n'.join(v).encode()).hexdigest() for k, v in fb.items()}
    hn = {k: hashlib.sha256('\n'.join(v).encode()).hexdigest() for k, v in fn.items()}
    # whole-section hash first (function names are stable across these builds, but do not rely on it)
    if [l for v in fb.values() for l in v] == [l for v in fn.values() for l in v]:
        return True, [], {}
    bad = sorted(k for k in set(hb) | set(hn) if hb.get(k) != hn.get(k))
    diffmap = {k: (fb.get(k, []), fn.get(k, [])) for k in bad}
    det = []
    for k in bad[:12]:
        a, c = fb.get(k, []), fn.get(k, [])
        dl = [(x, y) for x, y in zip(a, c) if x != y]
        det.append(f"function {k!r}: {len(dl)} differing insn(s) of {len(a)}/{len(c)}" +
                   ''.join(f"\n        - {x}\n        + {y}" for x, y in dl[:4]))
    if len(bad) > 12:
        det.append(f"... {len(bad) - 12} more differing functions")
    return False, [f"{len(bad)} differing function(s)"] + det, diffmap


PR = re.compile(rb'[\x20-\x7e]{4,}')

def mask_jt(x, plo, phi):
    """Overwrite 4-aligned (absolute .rodata offset) signed 32-bit words that look like switch-table entries with 0xffffffff.
    Entries are (target - table_address) and the table moves when .rodata strings change length -> layout-derived.
    Masked: every NEGATIVE word >= -8 MiB, and, when .text lies >= 1 MiB after .rodata (hip libs: a 55 MB fat binary sits between),
    positive words inside the exact reachable interval [plo, phi]."""
    n4 = len(x) // 4 * 4
    a = array.array('i'); a.frombytes(x[:n4])
    if sys.byteorder != 'little':
        a.byteswap()
    for i, w in enumerate(a):
        if -(8 << 20) <= w < 0 or (plo is not None and plo <= w <= phi):
            a[i] = -1
    if sys.byteorder != 'little':
        a.byteswap()
    return a.tobytes() + x[n4:]


def rodata_verdict(xb, xn, sb):
    """-> (unexplained_strings, residual_equal, n_dropped_strings, (len_b, len_n))
    mask table words, then: printable runs (>=4) present on one side only must all be build-info (TOKENS); delete them, drop NULs
    (alignment padding), residual must be identical.
    LIMITATION (named): a real edit of an int32 constant inside the masked ranges (large negative ints) is masked too; the primary
    evidence for 'no semantic change' is the identical normalised disassembly of all executable sections + source-diff scope."""
    ro = sb['.rodata']
    ex = [v for v in sb.values() if 'X' in v['flags'] and v['type'] != 'NOBITS']
    plo = min(v['addr'] for v in ex) - (ro['addr'] + ro['size'])
    phi = max(v['addr'] + v['size'] for v in ex) - ro['addr']
    if plo < (1 << 20):
        plo = phi = None
    xb, xn = mask_jt(xb, plo, phi), mask_jt(xn, plo, phi)
    cb, cn = collections.Counter(PR.findall(xb)), collections.Counter(PR.findall(xn))
    only_b, only_n = cb - cn, cn - cb
    unexplained = [s for s in list((only_b + only_n).keys()) if not TOKENS.search(s)]
    if unexplained:
        return unexplained, False, len(only_b) + len(only_n), (0, 0)
    drop = set(only_b) | set(only_n)
    if drop:
        rx = re.compile(b'|'.join(re.escape(s) for s in sorted(drop, key=len, reverse=True)))
        rb, rn = rx.sub(b'', xb).replace(b'\x00', b''), rx.sub(b'', xn).replace(b'\x00', b'')
    else:
        rb, rn = xb.replace(b'\x00', b''), xn.replace(b'\x00', b'')
    return unexplained, rb == rn, len(drop), (len(rb), len(rn))


# ---------------------------------------------------------------- HIP device code (.hip_fatbin) -> hipdev.py
def hip_verdict(b, n, sb, sn):
    ok, line, bad = hipdev.compare_fatbins(sec_bytes(b, sb['.hip_fatbin']), sec_bytes(n, sn['.hip_fatbin']))
    return ok, [line] + bad


UI_FUNC = re.compile(r'^_GLOBAL__sub_I_ui\.cpp$')
UI_INSN = re.compile(r'^movq +\$0x[0-9a-f]+,(.+)$')   # store of an immediate; destination must be identical on both sides


def ui_only(cdet_funcs):
    """cdet_funcs: {name: (a_lines, c_lines)} of differing functions -> True iff only the ui.cpp initialiser differs, via movq immediates"""
    if not cdet_funcs or not all(UI_FUNC.match(k) for k in cdet_funcs):
        return False
    for a, c in cdet_funcs.values():
        if len(a) != len(c):
            return False
        for x, y in zip(a, c):
            if x != y and not (UI_INSN.match(x) and UI_INSN.match(y) and UI_INSN.match(x).group(1) == UI_INSN.match(y).group(1)):
                return False
    return True


def names_modulo_cuid(x):
    return sorted(re.sub(rb'__hip_cuid_[0-9a-f]+', b'__hip_cuid_X', t) for t in x.split(b'\0'))


def classify(b, n, ui_source_identical=False):
    """-> (verdict, [detail lines])"""
    if hashlib.sha256(open(b, 'rb').read()).digest() == hashlib.sha256(open(n, 'rb').read()).digest():
        return 'SAME', []
    sb, sn = sections(b), sections(n)
    if set(sb) != set(sn):
        return 'UNEXPLAINED', ['section set differs: ' + str(sorted(set(sb) ^ set(sn)))]
    ok_code, cdet, diffmap = code_verdict(b, n)
    ui = False
    if not ok_code and ui_source_identical and ui_only(diffmap):
        ui = True
        cdet = ['only the web-UI static initialiser differs (movq immediates = gzip sizes of embedded assets); '
                'tools/ui + tools/server source diff is empty']
    bad, notes = [], []
    for k in sorted(sb):
        if EXEC_FLAG in sb[k]['flags']:
            continue
        xb, xn = sec_bytes(b, sb[k]), sec_bytes(n, sn[k])
        if xb == xn:
            continue
        if FREE.match(k):
            notes.append(f"{k}: differs (free)")
        elif k == '.rodata':
            if ui:
                notes.append(".rodata: differs (embedded UI asset bytes; accepted under EXPLAINED-UI)")
                continue
            un, resid_eq, nstr, rl = rodata_verdict(xb, xn, sb)
            if un:
                bad.append(f".rodata: {len(un)} non-build-info string(s) differ, e.g. {[s[:60] for s in un[:3]]}")
            elif not resid_eq:
                bad.append(f".rodata: non-string residual differs after removing {nstr} build-info string(s) (residual len {rl[0]} vs {rl[1]})")
            else:
                notes.append(f".rodata: only {nstr} build-info string(s) differ; residual identical")
        elif k == '.hip_fatbin':
            ok, det = hip_verdict(b, n, sb, sn)
            (notes if ok else bad).extend(det)
        elif k in ('.dynstr', '.strtab') and len(xb) != len(xn) and names_modulo_cuid(xb) == names_modulo_cuid(xn):
            notes.append(f"{k}: size {len(xb)}->{len(xn)} only through __hip_cuid_<hash> symbol names (hash printed without leading zeros)")
        elif LAYOUT.match(k):
            if len(xb) != len(xn):
                bad.append(f"{k}: size {len(xb)}->{len(xn)} (layout section changed size)")
            else:
                notes.append(f"{k}: layout-derived content shift")
        else:
            bad.append(f"{k}: non-code section differs, not in the allow-list")
    if not ok_code and not ui:
        return 'CODE-DIFF', cdet + bad + ['note ' + x for x in notes]
    if bad:
        return 'UNEXPLAINED', cdet + bad + ['note ' + x for x in notes] if ui else bad + ['note ' + x for x in notes]
    return ('EXPLAINED-UI' if ui else 'EXPLAINED'), (cdet if ui else []) + ['note ' + x for x in notes]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('base'); ap.add_argument('new')
    ap.add_argument('--ui-source-identical', action='store_true')
    a = ap.parse_args(argv)
    v, lines = classify(a.base, a.new, a.ui_source_identical)
    print(v)
    for l in lines:
        print('    ' + l)


if __name__ == '__main__':
    main()
