#!/usr/bin/env python3
"""fold_gate.py - gate (b) of a champion fold: shared libs of a candidate store image vs the previous champion image.

usage: fold_gate.py --base-image DIR --new-image DIR [--src-repo DIR --base-commit SHA --cand-commit SHA]
                    [--expected-changed libllama-bench-impl.so ...] [--ui-path tools/ui --ui-path tools/server]
exit 0 = PASS, 1 = FAIL.  Read-only on both image dirs.

Per library (matched by soname stem, so 10411 -> 10412 maps): SAME | EXPLAINED | EXPLAINED-UI | EXPECTED | CODE-DIFF |
UNEXPLAINED | MISSING | EXTRA.  EXPECTED = a lib named by --expected-changed whose code differs (the fold's own change).
EXPLAINED-UI is allowed only when `git diff <base> <cand> -- <ui paths>` in --src-repo is EMPTY (proved here, not assumed).
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import libcmp

LIBGLOB = re.compile(r'^lib(ggml|llama|mtmd).*\.so(\..*)?$')


def stem(name):
    return re.sub(r'\.so(\..*)?$', '.so', name)


def libs(d):
    return sorted(f for f in os.listdir(d) if LIBGLOB.match(f) and os.path.isfile(os.path.join(d, f)) and not os.path.islink(os.path.join(d, f)))


def links(d):
    return sorted(f for f in os.listdir(d) if LIBGLOB.match(f) and os.path.islink(os.path.join(d, f)))


def ui_source_identical(repo, base, cand, paths):
    if not (repo and base and cand):
        return False, 'no --src-repo/--base-commit/--cand-commit: EXPLAINED-UI disabled'
    r = subprocess.run(['git', '-C', repo, 'diff', '--stat', base, cand, '--', *paths], capture_output=True, text=True)
    if r.returncode != 0:
        return False, f'git diff failed ({r.stderr.strip()[:80]}): EXPLAINED-UI disabled'
    if r.stdout.strip():
        return False, 'tools/ui or tools/server changed between base and candidate: EXPLAINED-UI disabled'
    return True, f'git diff {base[:9]} {cand[:9]} -- {" ".join(paths)} is EMPTY'


def run(a, out=print):
    ui_ok, ui_msg = ui_source_identical(a.src_repo, a.base_commit, a.cand_commit, a.ui_path)
    out(f'ui-source: {ui_msg}')
    cnt = dict(same=0, explained=0, ui=0, expected=0, bad=0)
    nstem = {stem(f): f for f in libs(a.new_image)}
    for f in libs(a.base_image):
        st = stem(f)
        nf = nstem.pop(st, None)
        if nf is None:
            out(f'  MISSING     {f} (no lib with stem {st} in new image)'); cnt['bad'] += 1; continue
        if f != nf:
            out(f'  (map) {f} -> {nf}')
        v, det = libcmp.classify(os.path.join(a.base_image, f), os.path.join(a.new_image, nf), ui_ok)
        if v == 'SAME':
            cnt['same'] += 1; continue
        if v == 'CODE-DIFF' and st in a.expected_changed:
            out(f'  EXPECTED    {nf}  code differs (this fold changes it)'); cnt['expected'] += 1; continue
        if v in ('EXPLAINED', 'EXPLAINED-UI'):
            out(f'  {v:<11} {nf}')
            cnt['ui' if v == 'EXPLAINED-UI' else 'explained'] += 1
            for l in det:
                out('      ' + l)
        else:
            out(f'  {v:<11} {nf}')
            for l in det:
                out('      ' + l)
            cnt['bad'] += 1
    for st, nf in sorted(nstem.items()):
        out(f'  EXTRA       {nf} (not in base image)'); cnt['bad'] += 1
    bn = lambda t: re.sub(r'\.\d{4,6}$', '.BN', t)
    for ln in links(a.base_image):
        bt = os.readlink(os.path.join(a.base_image, ln))
        p = os.path.join(a.new_image, ln)
        if not os.path.islink(p):
            out(f'  SYMLINK-MISSING {ln}'); cnt['bad'] += 1; continue
        nt = os.readlink(p)
        if bn(bt) != bn(nt):
            out(f'  SYMLINK-DIFF {ln}: {bt} vs {nt}'); cnt['bad'] += 1
        elif not os.path.exists(os.path.join(a.new_image, nt)):
            out(f'  SYMLINK-DANGLING {ln} -> {nt}'); cnt['bad'] += 1
    out(f"(b) byte-identical={cnt['same']} explained={cnt['explained']} explained-ui={cnt['ui']} expected={cnt['expected']} "
        f"unexplained/code-diff/missing={cnt['bad']}")
    out('(b) PASS' if cnt['bad'] == 0 else '(b) FAIL')
    return cnt['bad'] == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base-image', required=True)
    ap.add_argument('--new-image', required=True)
    ap.add_argument('--src-repo'); ap.add_argument('--base-commit'); ap.add_argument('--cand-commit')
    ap.add_argument('--expected-changed', action='append', default=[], help='lib stem(s) whose code is supposed to change, e.g. libllama-bench-impl.so')
    ap.add_argument('--ui-path', action='append', default=None)
    a = ap.parse_args(argv)
    a.ui_path = a.ui_path or ['tools/ui', 'tools/server']
    for d in (a.base_image, a.new_image):
        d2 = os.path.join(d, 'bin')
        if os.path.isdir(d2) and not any(LIBGLOB.match(f) for f in os.listdir(d)):
            if d is a.base_image: a.base_image = d2
            else: a.new_image = d2
    return 0 if run(a) else 1


if __name__ == '__main__':
    sys.exit(main())
