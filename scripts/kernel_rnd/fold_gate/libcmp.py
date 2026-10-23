#!/usr/bin/env python3
"""libcmp.py BASE_LIB NEW_LIB -> one verdict line on stdout + detail lines (prefixed '    ').
       [--ui-source-identical]
Verdicts: SAME | EXPLAINED | EXPLAINED-PATH | EXPLAINED-UI | UNEXPLAINED | CODE-DIFF.   Read-only on both inputs.

--ui-source-identical : the caller PROVED `git diff <base> <cand> -- tools/ui tools/server` is empty (fold_gate.py does).
  Only then may a CODE-DIFF confined to the web-UI static initialiser be classified EXPLAINED-UI (class 4 below).

Classification (fold gate v4; v3 = CH-28 2026-10-23, v4 = C3 path-prefix normalisation):
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
  3b. PATH NORMALISATION (v4).  The two images may have been built from different source roots, and one of them may have
     used `-ffile-prefix-map=<src>=.` / `-fmacro-prefix-map` (CH-28d recipe) while the other did not.  Each side's roots are
     read from ITS OWN image (CMakeCache.txt CMAKE_HOME_DIRECTORY / CMAKE_CACHEFILE_DIR and the prefix-map flags in
     CMAKE_{C,CXX,HIP}_FLAGS; BUILD.log is the fallback) and every string is canonicalised before comparison:
     `<abs src root>/x` becomes `./x` (the spelling `-ffile-prefix-map=<src>=.` gives gcc; clang's `x` is rewritten to `./x` on a
     prefix-mapped image, for source-file paths only); `<abs build root>` and the mapped `.build` both become `<BLD>`.
     Only strings that are EQUAL after canonicalisation are explained; a real string change stays UNEXPLAINED.  Build-info is the
     short sha / build number each image reports (VERSION.txt, common/build-info.cpp) plus the --base/--cand commits.
     A lib whose only differences needed this normalisation is EXPLAINED-PATH (a pass, counted separately).
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
PBOUND = rb'(?<![\w.\-/])'          # a path starts here: not preceded by a path character


class ImageInfo:
    """Build roots and build-info of ONE image.  known=False -> nothing could be read (legacy static TOKENS apply)."""

    def __init__(self, src_root=None, build_root=None, maps=(), tokens=(), known=False):
        self.src_root, self.build_root, self.maps, self.tokens, self.known = src_root, build_root, list(maps), set(tokens), known


def _find_up(path, name, depth=3):
    d = os.path.dirname(os.path.abspath(path)) if not os.path.isdir(path) else os.path.abspath(path)
    for _ in range(depth + 1):
        c = os.path.join(d, name)
        if os.path.isfile(c):
            return c
        d = os.path.dirname(d)
    return None


def _read(p, limit=4 << 20):
    try:
        with open(p, 'r', errors='replace') as f:
            return f.read(limit)
    except OSError:
        return ''


def image_info(path, commits=()):
    """path = a lib or the image's bin/ (or image) dir.  -> ImageInfo.  Read-only."""
    src = bld = None
    maps, toks = [], set()
    cache = _find_up(path, 'CMakeCache.txt')
    flags = ''
    if cache:
        txt = _read(cache)
        m = re.search(r'^CMAKE_HOME_DIRECTORY:\w+=(.+)$', txt, re.M)
        src = m.group(1).strip() if m else None
        m = re.search(r'^CMAKE_CACHEFILE_DIR:\w+=(.+)$', txt, re.M)
        bld = m.group(1).strip() if m else os.path.dirname(cache)
        flags = '\n'.join(re.findall(r'^CMAKE_\w*_FLAGS:\w+=(.*)$', txt, re.M))
        root = os.path.dirname(cache)
    else:
        root = None
    log = _find_up(path, 'BUILD.log')
    if log:
        blog = _read(log, 1 << 20)
        flags += '\n' + blog
        if not src:   # fallback: a prefix-map whose source side is not the build dir names the source root
            for a, _b in re.findall(r'-f(?:file|macro)-prefix-map=(\S+?)=(\S*)', blog):
                if '/builds/' not in a and a != bld:
                    src = a
                    break
    for a, b in re.findall(r'-f(?:file|macro)-prefix-map=(\S+?)=(\S*)', flags):
        if (a, b) not in maps:
            maps.append((a, b))
    for fn in ('VERSION.txt', os.path.join('common', 'build-info.cpp')):
        t = _read(os.path.join(root, fn)) if root else ''
        for num, sha in re.findall(r'version:\s*(\d+)\s*\(([0-9a-f]{7,40})\)', t):
            toks.update((num, sha))
        for sha in re.findall(r'LLAMA_COMMIT\s*=\s*"([0-9a-f]{7,40})"', t):
            toks.add(sha)
        for num in re.findall(r'LLAMA_BUILD_NUMBER\s*=\s*(\d+)', t):
            toks.add(num)
    for c in commits:
        if c:
            toks.update({c, c[:7], c[:9]})
    return ImageInfo(src, bld, maps, toks, known=bool(src or bld or maps))


SRC_REL = re.compile(PBOUND + rb'(?:ggml|src|common|tools|include|vendor|examples)/[\w./+\-]*\.(?:cu|cuh|hip|c|cc|cpp|h|hpp)(?![\w])')


def make_canon(info):
    """-> canon(bytes)->bytes for one image's strings.  Canonical form: source root -> `./` (what a `-ffile-prefix-map=<src>=.`
    build emits, so strings that are the same on both sides are never touched), build root -> `<BLD>`.  Rules are derived
    from THIS image's own roots only, so an unmapped legitimate `./x` string is left alone on both sides."""
    rules = []   # (compiled regex, replacement)
    abs_ = []
    if info.src_root:
        abs_.append((info.src_root, 'S'))
    if info.build_root:
        abs_.append((info.build_root, 'B'))
    for root, role in sorted(abs_, key=lambda t: -len(t[0])):
        r = root.rstrip('/').encode()
        if role == 'S':
            rules.append((re.compile(PBOUND + re.escape(r) + rb'/'), b'./'))
            rules.append((re.compile(PBOUND + re.escape(r) + rb'(?![\w\-.])'), b'.'))
        else:
            rules.append((re.compile(PBOUND + re.escape(r) + rb'(?![\w\-.])'), b'<BLD>'))
    src_mapped = False
    for a, b in info.maps:
        role = 'S' if a.rstrip('/') == (info.src_root or '').rstrip('/') else 'B' if a.rstrip('/') == (info.build_root or '').rstrip('/') else None
        if role is None or not b:
            continue
        if role == 'S':
            src_mapped = True      # gcc: `./x` already canonical.  clang drops the `./`: restore it for source-file paths
        else:
            rules.append((re.compile(PBOUND + re.escape(b.rstrip('/').encode()) + rb'(?![\w\-.])'), b'<BLD>'))
    if src_mapped:
        rules.append((SRC_REL, lambda m: b'./' + m.group()))

    def canon(x):
        for rx, rep in rules:
            x = rx.sub(rep, x)
        return x
    return canon


def _token_rx(infos):
    toks = set().union(*(i.tokens for i in infos)) if infos else set()
    if not toks:
        return None
    return re.compile(b'|'.join(re.escape(t.encode()) for t in sorted(toks, key=len, reverse=True)))


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


def rodata_verdict(xb, xn, sb, ctx=None):
    """-> (unexplained_strings, residual_equal, n_dropped_strings, (len_b, len_n), n_path_normalised, pool_reordered)
    mask table words, then: printable runs (>=4) are canonicalised per side (path roots, see make_canon); canonical strings present
    on one side only must all be build-info (tokens); delete them and every run whose canonical form was touched by path
    normalisation, drop NULs (alignment padding), residual must be identical.
    ctx = (ImageInfo base, ImageInfo new) or None.  With no readable image info the legacy static TOKENS apply.
    LIMITATION (named): a real edit of an int32 constant inside the masked ranges (large negative ints) is masked too; the primary
    evidence for 'no semantic change' is the identical normalised disassembly of all executable sections + source-diff scope."""
    ro = sb['.rodata']
    ex = [v for v in sb.values() if 'X' in v['flags'] and v['type'] != 'NOBITS']
    plo = min(v['addr'] for v in ex) - (ro['addr'] + ro['size'])
    phi = max(v['addr'] + v['size'] for v in ex) - ro['addr']
    if plo < (1 << 20):
        plo = phi = None
    xb, xn = mask_jt(xb, plo, phi), mask_jt(xn, plo, phi)
    ib, inn = ctx if ctx else (ImageInfo(), ImageInfo())
    canon_b, canon_n = make_canon(ib), make_canon(inn)
    tokrx = _token_rx([ib, inn]) if (ib.known or inn.known) else TOKENS
    rb_runs, rn_runs = PR.findall(xb), PR.findall(xn)
    cb, cn = collections.Counter(canon_b(r) for r in rb_runs), collections.Counter(canon_n(r) for r in rn_runs)
    touched = {canon_b(r) for r in rb_runs if canon_b(r) != r} | {canon_n(r) for r in rn_runs if canon_n(r) != r}
    only_b, only_n = cb - cn, cn - cb
    if ib.known or inn.known:
        # strict: a one-sided string is build-info only if it matches a one-sided string of the other image once the tokens of
        # BOTH images' own build-info are blanked ("built dcc853b55 #10413" ~ "built 141e4bcf5 #10412", never "built deadbeef0")
        bi = lambda t: tokrx.sub(b'<BI>', t) if tokrx else t
        mb, mn = collections.Counter(), collections.Counter()
        for t, c in only_b.items():
            mb[bi(t)] += c
        for t, c in only_n.items():
            mn[bi(t)] += c
        bad_forms = set(((mb - mn) + (mn - mb)).keys())
        unexplained = [t for t in list((only_b + only_n).keys()) if bi(t) in bad_forms]
    else:
        unexplained = [t for t in list((only_b + only_n).keys()) if not (tokrx and tokrx.search(t))]
    path_norm = sum(1 for t in touched if t in (cb & cn))   # path-differing strings that normalisation proved equal
    if unexplained:
        return unexplained, False, len(only_b) + len(only_n), (0, 0), path_norm, False
    drop = set(only_b) | set(only_n) | touched

    def strip(x, canon):
        return PR.sub(lambda m: b'' if canon(m.group()) in drop else m.group(), x).replace(b'\x00', b'')
    rb, rn = strip(xb, canon_b), strip(xn, canon_n)
    if rb == rn:
        return unexplained, True, len(drop), (len(rb), len(rn)), path_norm, False

    def chunks(x):   # non-string bytes between printable runs, as a multiset: invariant under a re-ordered merged string pool
        return collections.Counter(c for c in PR.sub(b'\x00', x).split(b'\x00') if c)
    # ld merges .rodata.str* through a hash table: when string contents change (shorter paths) the pool is emitted in a different
    # ORDER.  Equal strings (checked above) in another order are layout, so fall back to an order-free comparison of the rest.
    return unexplained, chunks(xb) == chunks(xn), len(drop), (len(rb), len(rn)), path_norm, True


# ---------------------------------------------------------------- HIP device code (.hip_fatbin) -> hipdev.py
def hip_verdict(b, n, sb, sn):
    ok, line, bad, ptouched = hipdev.compare_fatbins_ex(sec_bytes(b, sb['.hip_fatbin']), sec_bytes(n, sn['.hip_fatbin']))
    return ok, [line] + bad, ptouched


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


def classify(b, n, ui_source_identical=False, ctx=None):
    """-> (verdict, [detail lines]).  ctx = (ImageInfo base, ImageInfo new); None -> read from the lib paths' own image dirs."""
    if ctx is None:
        ctx = (image_info(b), image_info(n))
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
    path_used = False
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
            un, resid_eq, nstr, rl, npath, reord = rodata_verdict(xb, xn, sb, ctx)
            if un:
                bad.append(f".rodata: {len(un)} non-build-info string(s) differ, e.g. {[s[:60] for s in un[:3]]}")
            elif not resid_eq:
                bad.append(f".rodata: non-string residual differs after removing {nstr} build-info string(s) (residual len {rl[0]} vs {rl[1]})")
            else:
                notes.append(f".rodata: only {nstr} build-info/path string(s) differ; residual identical"
                             + (f" ({npath} equal after build-root normalisation)" if npath else "")
                             + ("; string pool re-ordered by the linker, non-string bytes equal as a multiset" if reord else ""))
                path_used = path_used or npath > 0
        elif k == '.hip_fatbin':
            ok, det, ptouched = hip_verdict(b, n, sb, sn)
            (notes if ok else bad).extend(det)
            path_used = path_used or (ok and ptouched)
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
    return ('EXPLAINED-UI' if ui else 'EXPLAINED-PATH' if path_used else 'EXPLAINED'), (cdet if ui else []) + ['note ' + x for x in notes]


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
