#!/usr/bin/env python3
"""AK-ASM-T1 harness, install step (CPU only, offline; run once BEFORE the GPU packet is published).

Turns the COPY /mnt/raid0/llm/tmp/ak-asm-t1-20261105/aiter-src-90a into an AITER tree whose hsa/gfx90a/ holds the retargeted
gfx942 -> gfx90a code objects, so AITER's own loader (aiter_hip_common.h load_hsaco_file: $AITER_ASM_DIR/<gcnArchName>/<path>,
AITER_ASM_DIR = <aiter>/hsa/ set by aiter/jit/core.py) and its codegen (hsa/codegen.py: CSV manifests per hsa/<arch>/<family>/,
rows kept only for arch in AITER_GPU_ARCHS, map key = arch + knl_name) pick them on the MI210.  Never touches src-gfx90a.

  1. .co: every `yes` row of the T1 pass table (TABLE) -> batch_out/<rel with / -> __>.gfx90a.co copied to hsa/gfx90a/<rel>.
  2. CSV manifests: each hsa/gfx942/**/*.csv copied to hsa/gfx90a/ with ONLY the rows whose co_name is installed (a row pointing at
     a missing .co would let a heuristic pick an absent kernel). Manifests left with 0 rows are not written.
  3. Arch allow-lists: gates_90a.json, exact-count regex patches (python gates + csrc/cpp_itfs/mha_fwd.cu); idempotent.
  4. Instrumentation: aiter/test_common.py gets a checkAllclose wrapper printing `ASMT1_CHECK {json}` (msg, max_abs, ret).
  5. Invalidation: prebuilt JIT modules that embed asm config maps or patched C++ are renamed *.pre90a (no deletion) so the GPU
     packet's first use rebuilds them with the gfx90a manifests.
  6. INSTALL_MANIFEST.json (sha256 of every installed .co / CSV / patched file) in the packet dir; job.py --precond verifies it.

usage: install_90a.py [--dry-run]"""
import csv, glob, hashlib, io, json, os, re, shutil, sys

T1 = '/mnt/raid0/llm/tmp/ak-asm-t1-20261105'
DST = T1 + '/aiter-src-90a-r3'
SRC942 = DST + '/hsa/gfx942'
OUT90A = DST + '/hsa/gfx90a'
TABLE = T1 + '/table_nonfp8.tsv'
BATCH = T1 + '/batch_out'
PK = '/mnt/raid0/llm/tmp/ak-asm-t1-harness-r3-20261106'
GATES = PK + '/gates_90a.json'
MANIFEST = PK + '/INSTALL_MANIFEST.json'
DRY = '--dry-run' in sys.argv
INSTR_MARK = '# [ak-asm-t1] ASMT1_CHECK wrapper'
INSTR = '''

# [ak-asm-t1] ASMT1_CHECK wrapper: one machine-readable line per comparison (max |a-b| over all elements, return value)
_asmt1_checkAllclose_orig = checkAllclose


def checkAllclose(a, b, *args, **kw):  # noqa: F811
    import json as _j
    r = _asmt1_checkAllclose_orig(a, b, *args, **kw)
    rec = {"msg": str(kw.get("msg", args[3] if len(args) > 3 else ""))[:200], "ret": float(r) if r is not None else None,
           "tol_err_ratio": float(kw.get("tol_err_ratio", args[2] if len(args) > 2 else 0.05))}
    try:
        d = (a.detach().float() - b.detach().float()).abs()
        m = kw.get("mask")
        if m is not None:
            d = d[m.to(device=d.device, dtype=torch.bool).broadcast_to(d.shape)]
        rec["numel"] = int(d.numel())
        rec["nan"] = int(torch.isnan(d).sum().item())
        rec["max_abs"] = float(torch.nan_to_num(d, nan=0.0).max().item()) if d.numel() else 0.0
        rec["ref_absmax"] = float(a.detach().float().abs().max().item()) if a.numel() else 0.0
    except Exception as e:  # noqa: BLE001
        rec["err"] = repr(e)[:160]
    print("ASMT1_CHECK " + _j.dumps(rec), flush=True)
    return r
'''


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def put(src, dst):
    if DRY:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def main():
    if not os.path.isdir(SRC942) or os.path.realpath(DST).startswith('/mnt/raid0/llm/tmp/rock-aiter-gfx90a-20261105'):
        raise SystemExit('refusing: DST is not the T1 copy')
    man = {'dst': DST, 'table': TABLE, 'table_sha256': sha(TABLE), 'co': {}, 'csv': {}, 'patches': [], 'invalidated': [],
           'instrumentation': None, 'refused_rows_by_family': {}}
    rows = list(csv.reader(open(TABLE), delimiter='\t'))[1:]
    yes = []
    for r in rows:
        rel = r[0]
        fam = rel.split('/')[0] if '/' in rel else '(top)'
        if r[1] != 'yes':
            man['refused_rows_by_family'][fam] = man['refused_rows_by_family'].get(fam, 0) + 1
            continue
        src = os.path.join(BATCH, rel.replace('/', '__')[:-3] + '.gfx90a.co')
        if not os.path.exists(src):
            raise SystemExit(f'missing retargeted object for a yes row: {src}')
        yes.append(rel)
        put(src, os.path.join(OUT90A, rel))
        man['co'][rel] = {'from': src, 'sha256': sha(src)}
    installed = set(yes)
    # 2. manifests
    for c in sorted(glob.glob(SRC942 + '/**/*.csv', recursive=True)):
        relcsv = os.path.relpath(c, SRC942)
        d = os.path.dirname(relcsv)
        raw = open(c, encoding='utf-8-sig').read()
        lines = raw.splitlines(keepends=True)
        hdr_i = next(i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith(('//', '#', ';')))
        hdr = next(csv.reader([lines[hdr_i]]))
        if 'co_name' not in hdr:
            continue
        ci = hdr.index('co_name')
        keep, total = [lines[hdr_i]], 0
        for l in lines[hdr_i + 1:]:
            if not l.strip() or l.lstrip().startswith(('//', '#', ';', '/*')):
                continue
            total += 1
            f = next(csv.reader([l]))
            co = f[ci].strip() if len(f) > ci else ''
            if os.path.normpath(os.path.join(d, co)) in installed:
                keep.append(l if l.endswith('\n') else l + '\n')
        man['csv'][relcsv] = {'rows_942': total, 'rows_90a': len(keep) - 1}
        if len(keep) > 1 and not DRY:
            os.makedirs(os.path.join(OUT90A, d), exist_ok=True)
            open(os.path.join(OUT90A, relcsv), 'w').write(''.join(keep))
            man['csv'][relcsv]['sha256'] = sha(os.path.join(OUT90A, relcsv))
    # 3. gates
    g = json.load(open(GATES))
    for p in g['patches']:
        fp = os.path.join(DST, p['file'])
        s = open(fp).read()
        done_n = len(re.findall(p['done'], s, re.M))
        n = len(re.findall(p['regex'], s, re.M))
        want = p['count']
        rec = {'family': p['family'], 'file': p['file'], 'matches': n, 'already_done': done_n}
        if want == 'ALL>=1':
            if n == 0 and done_n == 0:
                raise SystemExit(f"gate patch matched nothing: {p['file']} {p['regex']}")
            if n:
                s2 = re.sub(p['regex'], p['repl'], s, flags=re.M)
                rec['applied'] = n
            else:
                s2 = s
        elif done_n == want and n == 0:
            s2, rec['applied'] = s, 0
        elif n == want:
            s2 = re.sub(p['regex'], p['repl'], s, flags=re.M)
            rec['applied'] = n
            if len(re.findall(p['done'], s2, re.M)) < want:
                raise SystemExit(f"gate patch 'done' check failed after apply: {p['file']}")
        else:
            raise SystemExit(f"gate patch count mismatch {p['file']}: regex {n} done {done_n} want {want}")
        if s2 != s and not DRY:
            open(fp, 'w').write(s2)
        rec['sha256_after'] = sha(fp) if not DRY else None
        man['patches'].append(rec)
    # 4. instrumentation
    tc = os.path.join(DST, 'aiter/test_common.py')
    s = open(tc).read()
    if INSTR_MARK not in s:
        if not re.search(r'^def checkAllclose\(', s, re.M) or not re.search(r'^import torch', s, re.M):
            raise SystemExit('test_common.py: checkAllclose/torch import not found')
        if not DRY:
            open(tc, 'a').write(INSTR)
    man['instrumentation'] = {'file': 'aiter/test_common.py', 'sha256': sha(tc)}
    # 5. invalidate prebuilt modules (rename, never delete)
    jit = os.path.join(DST, 'aiter/jit')
    cands = [os.path.join(jit, m + '.so') for m in g['invalidate_modules']]
    for gl in g['invalidate_globs']:
        cands += glob.glob(os.path.join(jit, gl))
    cands += [os.path.join(jit, 'build', m) for m in g['invalidate_modules']]
    for c in sorted(set(cands)):
        if os.path.exists(c):
            man['invalidated'].append(os.path.relpath(c, DST))
            if not DRY:
                os.rename(c, c + '.pre90a')
    man['n_co'] = len(man['co'])
    man['n_csv_written'] = sum(1 for v in man['csv'].values() if v['rows_90a'] > 0)
    if not DRY:
        json.dump(man, open(MANIFEST, 'w'), indent=1)
    print(json.dumps({k: man[k] for k in ('n_co', 'n_csv_written', 'invalidated', 'refused_rows_by_family')}, indent=1))
    print(json.dumps([{k: p[k] for k in ('family', 'file', 'matches', 'already_done', 'applied') if k in p} for p in man['patches']]))


if __name__ == '__main__':
    main()
