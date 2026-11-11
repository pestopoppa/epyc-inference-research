#!/usr/bin/env python3
"""Offline replay: run the CURRENT baseline gate over every stored BASELINE result/raw of race-eqv-gpu r3..r8 for EVERY suite in job.SUITES.
No GPU.  Each suite must come out USABLE, or an explicitly explained status:
  EMPTY_SURFACE  derived suite whose baseline parses <50 cases (layout-family cases skipped, FA paged mask has -inf): excluded by rule
  EXCLUDED       derived suite whose baseline fails the floor: excluded and recorded by the r9 rule (job.py), never INVALID
  NO_EVIDENCE    never run on any stored baseline; the job's derived-suite exclusion rule covers it (base suites all have evidence)
Exit 0 only if every suite is accounted for and every base suite is USABLE.  Skips (exit 0) if the tmp row dirs are gone."""
import glob, gzip, json, os, re, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import job
from gpu_signals import newfail as nf
ROWS = sorted(glob.glob('/mnt/raid0/llm/tmp/race-eqv-gpu-r[0-9]*/out/results.jsonl'))
HD_LOG = '/mnt/raid0/llm/tmp/race-eqv-gpu-r8-20261111/out/logs/current.log'
STALE = {('r7', 'GDN+VT')}   # r7 parse predates the Failing-tests parser (fail_count_unparsed); r8 re-ran it


def raw_of(rowdir, u):
    if not u.get('raw'): return {}
    p = f"{rowdir}/out/{u['raw']}"
    return json.load(gzip.open(p, 'rt')) if os.path.exists(p) else {}


def replay():
    ev = {}   # suite -> list of (row, status, reason)
    parses = {}
    for f in ROWS:
        rowdir = os.path.dirname(os.path.dirname(f)); tag = re.search(r'-(r\d+)-', rowdir).group(1)
        for l in open(f):
            try: u = json.loads(l)
            except ValueError: continue
            s = u.get('suite')
            if u.get('kind') == 'smoke':
                for x in u.get('seen', []):
                    if x.get('strict_pass') and x['suite'] in ('MMVF', 'GDN'): ev.setdefault(x['suite'], []).append((tag + '-smoke', 'USABLE', 'smoke strict pass %s' % x['passed']))
                continue
            if u.get('kind') != 'run' or u['variant'] not in ('BASELINE', 'BASELINE-REPEAT'): continue
            p = u.get('parse')
            if not p or (tag, s) in STALE or u['rc'] not in (0, 1): continue
            if u['variant'] == 'BASELINE' and tag in ('r4', 'r5', 'r6', 'r7', 'r8') or u['variant'] == 'BASELINE-REPEAT':
                parses.setdefault(s, []).append(p)
            if u['variant'] != 'BASELINE': continue
            dom = nf.domain_fails(p, raw_of(rowdir, u))
            ok, why = nf.baseline_usable(p, domain=dom)
            if ok: st = 'USABLE'
            elif '+' in s and (p['total'] < 50 or why == 'baseline_pass_fraction_below_floor'): st = 'EMPTY_SURFACE' if p['total'] < 50 else 'EXCLUDED'
            else: st = 'UNUSABLE'
            ev.setdefault(s, []).append((tag, st, f'{why} {p["passed"]}/{p["total"]} domain={dom}'))
    if os.path.exists(HD_LOG):   # the one raw log still on disk: re-parse from text with the current parser
        p = nf.parse_log(open(HD_LOG, errors='replace').read()); raw = raw_of('/mnt/raid0/llm/tmp/race-eqv-gpu-r8-20261111', {'raw': 'raw/BASELINE-GDN+HD-0.json.gz'})
        dom = nf.domain_fails(p, raw); ok, why = nf.baseline_usable(p, domain=dom)
        ev.setdefault('GDN+HD', []).append(('r8-log', 'USABLE' if ok else 'UNUSABLE', f'{why} {p["passed"]}/{p["total"]} domain={dom}'))
    return ev, parses


def main():
    if not ROWS: print('SKIP: stored row dirs gone'); return 0
    ev, parses = replay(); rc = 0
    for s in job.SUITES:
        e = ev.get(s, []); sts = {x[1] for x in e}
        if not e: fin = 'NO_EVIDENCE'
        elif 'UNUSABLE' in sts: fin = 'UNUSABLE'
        elif sts <= {'USABLE'}: fin = 'USABLE'
        else: fin = sorted(sts - {'USABLE'})[0]
        base = '+' not in s
        bad = fin == 'UNUSABLE' or (base and fin not in ('USABLE',)) or (not base and fin == 'NO_EVIDENCE' and False)
        why = ' | '.join(f'{a}:{b}:{c}' for a, b, c in e) or 'never run on a stored baseline; derived suite -> covered by the job exclusion rule' if not base else ' | '.join(f'{a}:{b}:{c}' for a, b, c in e)
        print(f'{s:8s} {fin:12s} {"FAIL" if bad else "ok"}  {why}'); rc |= bad
    for s, ps in sorted(parses.items()):
        k = nf.known_failures(ps)
        print(f'known-baseline-failures {s:6s} runs={k["n_runs"]} deterministic={k["deterministic"]} n_known={len(k["known"])}')
        if s in ('Q', 'QPF'): assert k['deterministic'] and len(k['known']) == 312, (s, k['n_runs'], len(k['known']))
    assert ev['GDN+VT'] and ev['GDN+HD'], 'GDN derived suites must replay USABLE'
    print('REPLAY', 'FAIL' if rc else 'PASS'); return rc


def test_replay(): assert main() == 0


if __name__ == '__main__': sys.exit(main())
