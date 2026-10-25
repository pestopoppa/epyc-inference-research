#!/usr/bin/env python3
"""fold_gate.py - gate (b) of a champion fold: shared libs of a candidate store image vs the previous champion image.

usage: fold_gate.py --base-image DIR --new-image DIR [--src-repo DIR --base-commit SHA --cand-commit SHA]
                    [--expected-changed libllama-bench-impl.so ...] [--ui-path tools/ui --ui-path tools/server]
exit 0 = PASS, 1 = FAIL.  Read-only on both image dirs.

Per library (matched by soname stem, so 10411 -> 10412 maps): SAME | EXPLAINED | EXPLAINED-PATH | EXPLAINED-UI | EXPECTED |
CODE-DIFF | UNEXPLAINED | MISSING | EXTRA.  EXPLAINED-PATH = identical code, differences only in strings that are equal once each
image's own build/source root (absolute or `-ffile-prefix-map`ped, read from its CMakeCache.txt) is normalised, build-info and layout.
EXPECTED = a lib named by --expected-changed whose code differs (the fold's own change).
EXPLAINED-UI is allowed only when `git diff <base> <cand> -- <ui paths>` in --src-repo is EMPTY (proved here, not assumed).
"""
import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import libcmp

LIBGLOB = re.compile(r'^lib(ggml|llama|mtmd).*\.so(\..*)?$')
RECORD_SCHEMA = 'epyc.champion_fold_gate.v1'
GATE_VERSION = 'fold_gate.v4'
IDENTICAL_CLASSES = frozenset(('SAME', 'EXPLAINED', 'EXPLAINED-PATH', 'UNEXPLAINED'))
PASS_CLASSES = (IDENTICAL_CLASSES - frozenset(('UNEXPLAINED',))) | frozenset(('EXPLAINED-UI', 'EXPECTED'))


def stem(name):
    return re.sub(r'\.so(\..*)?$', '.so', name)


def libs(d):
    return sorted(f for f in os.listdir(d) if LIBGLOB.match(f) and os.path.isfile(os.path.join(d, f)) and not os.path.islink(os.path.join(d, f)))


def links(d):
    return sorted(f for f in os.listdir(d) if LIBGLOB.match(f) and os.path.islink(os.path.join(d, f)))


def library_proposition(row, base_store, candidate_store):
    identity = 'is identical' if row['code_identical'] else 'is not established as identical'
    pair = f"{row['base_soname']} -> {row['candidate_soname']}" if row['base_soname'] and row['candidate_soname'] else row['soname_stem']
    return (f"Fold gate v4 classified {row['verdict_class']} for {pair}: executable code {identity} "
            f"between base store {base_store} and candidate store {candidate_store}.")


def exception_proposition(row):
    return (f"Fold gate v4 accepted the EXPLAINED-UI exception for {row['soname_stem']} under "
            "its deterministic UI-initializer rule; no other code-difference exception is implied.")


def _store_attestation(image_dir):
    image = Path(image_dir).resolve()
    store = image.parent if image.name == 'bin' else image
    sums = store / 'SHA256SUMS'
    fd = os.open(sums, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f'not a regular SHA256SUMS file: {sums}')
        with os.fdopen(fd, 'rb') as stream:
            raw = stream.read((64 << 20) + 1)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    if len(raw) > 64 << 20:
        raise OSError(f'SHA256SUMS file exceeds 64 MiB: {sums}')
    return {'store_path': str(store), 'sha256s_path': str(sums),
            'sha256s_sha256': hashlib.sha256(raw).hexdigest()}


def _write_record(path, record):
    """Publish one complete machine record without replacing an earlier receipt."""
    target = Path(path).resolve(strict=False)
    for side in ('base', 'candidate'):
        store = Path(record[side]['store_path']).resolve()
        if target == store or store in target.parents:
            raise OSError(f'record output must be outside the {side} store: {target}')
    payload = (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link() is atomic and refuses to replace an existing receipt.
        os.link(temporary, target)
        os.unlink(temporary)
        dfd = os.open(target.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def ui_source_identical(repo, base, cand, paths):
    if not (repo and base and cand):
        return False, 'no --src-repo/--base-commit/--cand-commit: EXPLAINED-UI disabled'
    r = subprocess.run(['git', '-C', repo, 'diff', '--stat', base, cand, '--', *paths], capture_output=True, text=True)
    if r.returncode != 0:
        return False, f'git diff failed ({r.stderr.strip()[:80]}): EXPLAINED-UI disabled'
    if r.stdout.strip():
        return False, 'tools/ui or tools/server changed between base and candidate: EXPLAINED-UI disabled'
    return True, f'git diff {base[:9]} {cand[:9]} -- {" ".join(paths)} is EMPTY'


def run(a, out=print, *, record_json=None):
    record_base = getattr(a, 'record_base', None)
    record_candidate = getattr(a, 'record_candidate', None)
    if record_json and (record_base is None or record_candidate is None):
        record_base = _store_attestation(a.base_image)
        record_candidate = _store_attestation(a.new_image)
    ui_ok, ui_msg = ui_source_identical(a.src_repo, a.base_commit, a.cand_commit, a.ui_path)
    out(f'ui-source: {ui_msg}')
    cnt = dict(same=0, explained=0, path=0, ui=0, expected=0, bad=0)
    commits = [c for c in (a.base_commit, a.cand_commit) if c]
    ib, inn = libcmp.image_info(a.base_image, commits), libcmp.image_info(a.new_image, commits)
    for tag, i in (('base', ib), ('new', inn)):
        out(f'image-roots {tag}: src={i.src_root} build={i.build_root} prefix-maps={len(i.maps)} build-info={sorted(i.tokens)[:4]}'
            + ('' if i.known else '  (UNREADABLE: legacy static tokens, no path normalisation)'))
    nstem = {stem(f): f for f in libs(a.new_image)}
    library_rows = []
    for f in libs(a.base_image):
        st = stem(f)
        nf = nstem.pop(st, None)
        if nf is None:
            v, det = 'MISSING', [f'no candidate library with stem {st}']
            out(f'  MISSING     {f} (no lib with stem {st} in new image)')
            cnt['bad'] += 1
        else:
            if f != nf:
                out(f'  (map) {f} -> {nf}')
            v, det = libcmp.classify(os.path.join(a.base_image, f), os.path.join(a.new_image, nf), ui_ok, (ib, inn))
            if v == 'SAME':
                cnt['same'] += 1
            elif v == 'CODE-DIFF' and st in a.expected_changed:
                out(f'  EXPECTED    {nf}  code differs (this fold changes it)')
                cnt['expected'] += 1
                v = 'EXPECTED'
            elif v in ('EXPLAINED', 'EXPLAINED-PATH', 'EXPLAINED-UI'):
                out(f'  {v:<11} {nf}')
                cnt[{'EXPLAINED-UI': 'ui', 'EXPLAINED-PATH': 'path'}.get(v, 'explained')] += 1
                for l in det:
                    out('      ' + l)
            else:
                out(f'  {v:<11} {nf}')
                for l in det:
                    out('      ' + l)
                cnt['bad'] += 1
        row = {'record_type': 'library', 'soname_stem': st, 'base_soname': f,
               'candidate_soname': nf, 'verdict_class': v,
               'code_identical': v in IDENTICAL_CLASSES,
               'differing_sections': list(det)}
        row['decided_proposition'] = library_proposition(row, record_base['store_path'],
                                                         record_candidate['store_path']) if record_json else ''
        library_rows.append(row)
    candidate_only = []
    for st, nf in sorted(nstem.items()):
        out(f'  EXTRA       {nf} (not in base image)')
        cnt['bad'] += 1
        row = {'record_type': 'library', 'soname_stem': st, 'base_soname': None,
               'candidate_soname': nf, 'verdict_class': 'EXTRA', 'code_identical': False,
               'differing_sections': [f'no base library with stem {st}']}
        if record_json:
            row['decided_proposition'] = library_proposition(row, record_base['store_path'],
                                                             record_candidate['store_path'])
        candidate_only.append(row)
    library_rows.extend(candidate_only)
    bn = lambda t: re.sub(r'\.\d{4,6}$', '.BN', t)
    symlink_rows = []
    for ln in links(a.base_image):
        bt = os.readlink(os.path.join(a.base_image, ln))
        p = os.path.join(a.new_image, ln)
        if not os.path.islink(p):
            status, nt, target_exists = 'SYMLINK-MISSING', None, False
            out(f'  SYMLINK-MISSING {ln}')
            cnt['bad'] += 1
        else:
            nt = os.readlink(p)
            target_exists = os.path.exists(os.path.join(a.new_image, nt))
            if bn(bt) != bn(nt):
                status = 'SYMLINK-DIFF'
                out(f'  SYMLINK-DIFF {ln}: {bt} vs {nt}')
                cnt['bad'] += 1
            elif not target_exists:
                status = 'SYMLINK-DANGLING'
                out(f'  SYMLINK-DANGLING {ln} -> {nt}')
                cnt['bad'] += 1
            else:
                status = 'MATCHED'
        symlink_rows.append({'name': ln, 'base_target': bt, 'candidate_target': nt,
                             'candidate_target_exists': target_exists, 'verdict_class': status})
    out(f"(b) byte-identical={cnt['same']} explained={cnt['explained']} explained-path={cnt['path']} explained-ui={cnt['ui']} expected={cnt['expected']} "
        f"unexplained/code-diff/missing={cnt['bad']}")
    out('(b) PASS' if cnt['bad'] == 0 else '(b) FAIL')
    passed = cnt['bad'] == 0
    if record_json:
        exceptions = []
        for row in library_rows:
            if row['verdict_class'] == 'EXPLAINED-UI':
                exceptions.append({'record_type': 'accepted_exception',
                                   'exception_class': 'EXPLAINED-UI',
                                   'library_stem': row['soname_stem'],
                                   'accepted_by': 'fold_gate.v4:EXPLAINED-UI',
                                   'decided_proposition': exception_proposition(row),
                                   'differing_sections': row['differing_sections']})
        _write_record(record_json, {'schema': RECORD_SCHEMA, 'gate_version': GATE_VERSION,
                                    'recorded_at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                                    'base': record_base, 'candidate': record_candidate,
                                    'libraries': library_rows, 'symlinks': symlink_rows,
                                    'exceptions': exceptions,
                                    'gate_verdict': 'PASS' if passed else 'FAIL'})
    return passed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base-image', required=True)
    ap.add_argument('--new-image', required=True)
    ap.add_argument('--src-repo'); ap.add_argument('--base-commit'); ap.add_argument('--cand-commit')
    ap.add_argument('--expected-changed', action='append', default=[], help='lib stem(s) whose code is supposed to change, e.g. libllama-bench-impl.so')
    ap.add_argument('--ui-path', action='append', default=None)
    ap.add_argument('--record-json', help='write an immutable structured Vidya source record; existing stdout/exit output is unchanged')
    a = ap.parse_args(argv)
    a.ui_path = a.ui_path or ['tools/ui', 'tools/server']
    for d in (a.base_image, a.new_image):
        d2 = os.path.join(d, 'bin')
        if os.path.isdir(d2) and not any(LIBGLOB.match(f) for f in os.listdir(d)):
            if d is a.base_image: a.base_image = d2
            else: a.new_image = d2
    if a.record_json:
        try:
            a.record_base = _store_attestation(a.base_image)
            a.record_candidate = _store_attestation(a.new_image)
        except OSError as exc:
            print(f'fold-gate record refused: cannot read store SHA256SUMS attestation: {exc}', file=sys.stderr)
            return 2
    try:
        return 0 if run(a, record_json=a.record_json) else 1
    except (OSError, FileExistsError) as exc:
        print(f'fold-gate record refused: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
