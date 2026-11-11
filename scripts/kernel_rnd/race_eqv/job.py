#!/usr/bin/env python3
"""race-eqv-gpu-r8-20261111 (slimmed r4; 1 seed per suite, BASELINE|Q|0 reused from r4): gate each seeded-defect build vs BASELINE on the MI210 (RVP-RACE-1, RVP-EQV-1).
Order: BASELINE, BASELINE-REPEAT (determinism), per-suite kernel-instance coverage, controls, mutants.
Gate (pre-registered 2026-11-10, operator decision, RVP-EQV-1 box): new-fail vs the BASELINE run of the same suite and seed;
ANSI stripped before FAIL parsing (gpu_signals/newfail.py = research gpu_newfail_gate.py).
rc: 0 verdict | 75 stop/deadline/builds absent | 3 smoke failure."""
import gzip, hashlib, json, os, re, subprocess, sys, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import gpuproof as gp
from gpu_signals import race_eqv as re1
from gpu_signals import newfail as nf
ROW_ID = 'race-eqv-gpu-r8-20261111'
ST = '/mnt/raid0/llm/kernels/builds/MUTANT-race-eqv-r3-20261109-36f024341'; SHA = '36f024341ce2a48106bbeabd19943384849f378c'
BUILDRES = '/mnt/raid0/llm/tmp/race-eqv-mutants-r3-20261109/out/RESULT.json'; C9 = '/mnt/raid0/llm/kernels/builds/gpu-20261109-d210f85c2-c9fold/bin'
OUT = D + '/out'; LOGS = OUT + '/logs'; RAW = OUT + '/raw'; RES = OUT + '/results.jsonl'; RESULT = OUT + '/RESULT.json'
STOPF = os.environ.get('FILLER_STOPFILE', D + '/STOP'); DEADLINE = int(os.environ.get('FILLER_DEADLINE', str(int(time.time()) + 6 * 3600)))
SMOKE = os.environ.get('FILLER_DRYRUN') == '1' or os.environ.get('FILLER_SMOKE') == '1'
CPUS = os.environ.get('FILLER_CPUS') or '184-191'; R, E, COV = 1, 1, 1;   # r5 seed-count amendment (RVP-EQV-1 box, root 95d6eff52): one seed (1000) per suite
RAW_CAP_B = 60 << 20; MAXABS_BAND = 4.0
FA_P = 'hsk=(64|128|256)'; CS = {'AUTOKERNEL_CORRECTNESS_CASE_SET': 'served_shape_lowbit_v1'}; VT = ['--autokernel-value-transforms']; HD = ['--autokernel-hostile-distributions']
ops = lambda o, p=None: (o, p)
SUITES = {'Q': (ops('MUL_MAT,MUL_MAT_ID'), CS, []), 'QPF': (ops('MUL_MAT,MUL_MAT_ID'), dict(CS, GGML_CUDA_Q8_PREFETCH='1'), []), 'FA': (ops('FLASH_ATTN_EXT', FA_P), {}, []), 'GDN': (ops('GATED_DELTA_NET'), {}, []), 'MMVF': (ops('MUL_MAT'), {}, [])}
for b in ('FA', 'GDN', 'MMVF'):
    o, e, _ = SUITES[b]; SUITES[b + '+VT'] = (o, e, VT); SUITES[b + '+HD'] = (o, e, HD); SUITES[b + '+ED'] = (o, dict(e, RACE_EQV_EDGE='1'), HD)
RACE = {'R1-MMQ-TILE-LOAD': 'Q', 'R1-MMVQ-TMP-NW4': 'Q', 'R1-MMVQ-TMP-NW8': 'Q', 'R1-MMVQ-LDSDMA': 'QPF', 'R1-FA-MMA-SOFTMAX': 'FA'}
EQV = {'E1-FA-RESCALE-SKIP': 'FA', 'E1-FA-DROP-LAST-KV': 'FA', 'E1-FATILE-DROP-TAIL': 'FA', 'E1-GDN-DECAY-DROP': 'GDN', 'E1-GDN-BETA-DROP': 'GDN', 'E1-MMVF-PARTIAL-K': 'MMVF', 'E1-CTRL-A': 'FA', 'E1-CTRL-B': 'GDN'}
GUARD = {'R1-MMQ-TILE-LOAD': ('COV-MMQ', 'Q'), 'R1-MMVQ-TMP-NW4': ('COV-MMVQ', 'Q'), 'R1-MMVQ-TMP-NW8': ('COV-MMVQ', 'Q'), 'R1-MMVQ-LDSDMA': ('COV-MMVQ-LDS', 'QPF'), 'R1-FA-MMA-SOFTMAX': ('E1-CTRL-A', 'FA'),
         'E1-FA-RESCALE-SKIP': ('E1-CTRL-A', 'FA'), 'E1-FA-DROP-LAST-KV': ('E1-CTRL-A', 'FA'), 'E1-FATILE-DROP-TAIL': ('COV-FATILE', 'FA'), 'E1-GDN-DECAY-DROP': ('E1-CTRL-B', 'GDN'), 'E1-GDN-BETA-DROP': ('E1-CTRL-B', 'GDN'), 'E1-MMVF-PARTIAL-K': ('COV-MMVF', 'MMVF')}
CTRLS = [('COV-MMQ', 'Q'), ('COV-MMVQ', 'Q'), ('COV-MMVQ-LDS', 'QPF'), ('COV-FATILE', 'FA'), ('COV-MMVF', 'MMVF')]
BASE_RACE = ('Q', 'QPF', 'FA'); BASE_EQV = ('FA', 'GDN', 'MMVF')
EMPTY = {}   # r6: derived suites (+VT/+HD/+ED) whose BASELINE parsed <50 cases: layout-family cases are skipped in those modes and the 2 surviving FA cases carry a -inf F16 mask that x3 cannot represent (r5 finding)
def eqv_suites(b): return [b] + [x for x in (b + '+VT', b + '+HD', b + '+ED') if x not in EMPTY]
PLAN = [('BASELINE', s, R) for s in BASE_RACE] + [('BASELINE', s, E) for b in BASE_EQV for s in eqv_suites(b) if s not in BASE_RACE]
REPEAT_SUITES = ('Q', 'QPF', 'FA', 'GDN', 'MMVF')
PLAN += [('BASELINE-REPEAT', s, 1) for s in REPEAT_SUITES]
PLAN += [(c, s, COV) for c, s in CTRLS] + [(c, s, E) for c, b in (('E1-CTRL-A', 'FA'), ('E1-CTRL-B', 'GDN')) for s in eqv_suites(b)]
PLAN += [(m, s, R) for m, s in RACE.items()] + [(m, s, E) for m, b in EQV.items() if 'CTRL' not in m for s in eqv_suites(b)]
SHN = re.compile(r'ShaderName\s*:\s*(\S+)')
def now(): return time.strftime('%FT%TZ', time.gmtime())
def append(u):
    with open(RES, 'a') as f: f.write(json.dumps(u) + '\n'); f.flush(); os.fsync(f.fileno())
def jsave(p, o):
    with open(p + '.tmp', 'w') as f: json.dump(o, f, indent=1)
    os.replace(p + '.tmp', p)
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def stop(): return os.path.exists(STOPF) or time.time() > DEADLINE - 300
def raw_size(): return sum(os.path.getsize(os.path.join(RAW, f)) for f in os.listdir(RAW))
def tbo_env(binroot, senv):
    e = dict(os.environ, FILLER_ID=os.environ.get('FILLER_ID', ROW_ID), LD_LIBRARY_PATH=binroot + ':/opt/rocm/lib', HIP_VISIBLE_DEVICES='0', ROCM_PATH='/opt/rocm', AUTOKERNEL_BACKEND_THREADS='16', RACE_EQV_RAW='1', **senv)
    for k in ('GGML_CUDA_Q8_PREFETCH', 'RACE_EQV_EDGE', 'AMD_LOG_LEVEL'):
        if k not in senv: e.pop(k, None)
    return e
def argv_of(binroot, suite, seed):
    (o, p), senv, fl = SUITES[suite]
    a = ['taskset', '-c', CPUS, binroot + '/test-backend-ops', 'test', '-o', o, '-b', 'ROCm0', '-j', '1', '--suite-seed', str(seed)] + fl
    return a + (['-p', p] if p else []), senv
def run_tbo(binroot, suite, seed, tag, timeout):
    a, senv = argv_of(binroot, suite, seed); log = f'{LOGS}/current.log'; t0 = time.time()   # one rolling log: per-run raw lives in out/raw
    rc, err, p = gp.run_sampled(a, log, env=tbo_env(binroot, senv), timeout_at=min(DEADLINE - 240, time.time() + timeout), stopfile=STOPF, cwd=D)
    t = open(log, errors='replace').read(); pl = nf.parse_log(t); passed = [[str(pl['passed']), str(pl['total'])]] if pl['complete'] else []
    vals = {}
    for m in re.finditer(r'REQ_RAW case=(\S+) nmse=(\S+) max_abs=(\S+)', t):
        k = m.group(1); n, x = float(m.group(2)), float(m.group(3)); o = vals.get(k)
        vals[k] = [n, x] if o is None else [max(o[0], n), max(o[1], x)]
    if err: pl = dict(pl, complete=False)   # stop/deadline/timeout: never a complete log
    r = {'rc': rc, 'err': err, 'n_fail_lines': pl['n_fail'], 'fail_keys': sorted(pl['fail_counts']), 'parse': pl, 'passed': passed, 'vram_peak': max(p.get('kfd_pids_ours', {}).values(), default=0), 'seconds': round(time.time() - t0, 1), 'n_raw': len(vals), 'log_sha256': hashlib.sha256(t.encode()).hexdigest()}
    r['hip_loaded'] = next((l for m in p.get('maps', {}).values() for l in m.get('libs', []) if 'libggml-hip' in l), None)
    r['strict_pass'] = rc == 0 and not err and pl['complete'] and pl['n_fail'] == 0   # informational only; the gate is set by the caller
    r['gate_pass'] = None
    if raw_size() < RAW_CAP_B:
        with gzip.open(f'{RAW}/{tag}.json.gz', 'wt') as f: json.dump(vals, f)
        r['raw'] = f'raw/{tag}.json.gz'
    else: r['raw'] = None
    return r
def coverage(binroot, suite):
    a, senv = argv_of(binroot, suite, 1000); e = dict(tbo_env(binroot, senv), AMD_LOG_LEVEL='4'); names = set(); t0 = time.time()
    p = subprocess.Popen(a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e, cwd=D, text=True, errors='replace', start_new_session=True)
    for l in p.stdout:
        m = SHN.search(l)
        if m: names.add(m.group(1).replace('.kd', ''))
        if stop() or time.time() - t0 > 2400: os.killpg(p.pid, 9); break
    p.wait(); return {'suite': suite, 'n_names': len(names), 'names': sorted(names), 'seconds': round(time.time() - t0, 1)}
def load_done():
    done = {}
    if os.path.exists(RES):
        for l in open(RES):
            try: u = json.loads(l)
            except ValueError: continue
            if u.get('kind') == 'run': done[(u['variant'], u['suite'], u['run'])] = u
    return done
def vals_of(u):
    if not u.get('raw'): return {}
    with gzip.open(f"{OUT}/{u['raw']}", 'rt') as f: return json.load(f)
def verdict(done, builds, cov):
    def rs(v, s, n): return [done[(v, s, i)] for i in range(n) if (v, s, i) in done]
    invalid = [f'BASELINE {s} not {n}/{n}' for s, n in [(s, R) for s in BASE_RACE] + [(b, E) for b in BASE_EQV if b not in BASE_RACE] if len(rs('BASELINE', s, n)) != n or not all(r['gate_pass'] for r in rs('BASELINE', s, n))]
    ctl = {}
    for c, s in CTRLS: x = rs(c, s, COV); ctl[c] = {'runs': len(x), 'passes': sum(r['gate_pass'] for r in x)}
    for c, b in (('E1-CTRL-A', 'FA'), ('E1-CTRL-B', 'GDN')): x = rs(c, b, E); ctl[c] = {'runs': len(x), 'passes': sum(r['gate_pass'] for r in x)}
    inst = {}
    def exec_k(m, s):
        k = builds[m]['isa_proof']['kernels']; names = set(cov.get(s, {}).get('names', []))
        return {n: v for n, v in k.items() if n in names} if names else None
    excl = {}
    for m, (g, s) in GUARD.items():
        if ctl[g]['runs'] == 0 or ctl[g]['passes'] > 0: excl[m] = 'coverage_control_passed'; continue
        ek = exec_k(m, s); inst[m] = None if ek is None else len(ek)
        if ek is not None and not ek: excl[m] = 'mutated_instance_not_executed'
    race_in = {}
    for m, s in RACE.items():
        if m in excl: continue
        ek = exec_k(m, s); k = ek if ek else builds[m]['isa_proof']['kernels']
        race_in[m] = {'gate_pass': [r['gate_pass'] for r in rs(m, s, R)], 'barriers_base': sum(v[0] for v in k.values()), 'barriers_mut': sum(v[1] for v in k.values()), 'waves_per_wg': min([v[2] for v in k.values()] or [0])}
    bfull = [all(('BASELINE', s, i) in done and done[('BASELINE', s, i)]['gate_pass'] for s in BASE_RACE) for i in range(R)]
    race = re1.race1_decide(race_in, bfull, runs=R) if not invalid else {'decision': 'INVALID'}
    band, bfail, nonfin = {}, {}, set()
    for b in BASE_EQV:
        for s in eqv_suites(b):
            for r in rs('BASELINE', s, E):
                for k, v in vals_of(r).items():
                    if v[1] != v[1] or v[1] == float('inf'): nonfin.add((s, k))   # r8: baseline itself non-finite (GDN negate): no band
                    else: band[(s, k)] = max(band.get((s, k), 0.0), v[1])
    surv, caught = {}, {}
    def one(m, b, i):
        ok = True; why = []
        for s in eqv_suites(b):
            u = done.get((m, s, i))
            if u is None: return False, ['missing']
            if not u['gate_pass']: ok = False; why.append(s + ':' + u['gate']['verdict'])
            for k, v in vals_of(u).items():
                if (s, k) in band and (s, k) not in nonfin and not (v[1] <= MAXABS_BAND * band[(s, k)] + 1e-9): ok = False; why.append(s + ':maxabs'); break
        return ok, why
    for m, b in EQV.items():
        if m in excl and 'CTRL' not in m: continue
        o = [one(m, b, i) for i in range(E)]; surv[m] = [x[0] for x in o]; caught[m] = [x[1] for x in o]
    ebase = [all(('BASELINE', s, i) in done and done[('BASELINE', s, i)]['gate_pass'] for b in BASE_EQV for s in eqv_suites(b)) for i in range(E)]
    eqv = re1.eqv1_decide(surv, ['E1-CTRL-A', 'E1-CTRL-B'], ebase, seeds=E) if not invalid else {'decision': 'INVALID'}
    if ctl['E1-CTRL-A']['passes'] or ctl['E1-CTRL-B']['passes']: invalid.append('EQV positive control passed the gate')
    return {'at': now(), 'complete': True, 'verdict': 'INVALID_HARNESS' if invalid else 'DONE', 'invalid': invalid, 'excluded': excl, 'controls': ctl, 'instances_executed': inst, 'race1': race, 'eqv1': eqv, 'eqv_caught_by': caught,
            'empty_surfaces': dict(EMPTY), 'instance_coverage': {s: c['n_names'] for s, c in cov.items()}, 'maxabs_band': MAXABS_BAND, 'source_sha': SHA,
            'determinism': {s: done[('BASELINE-REPEAT', s, 0)]['gate'] for s in REPEAT_SUITES if ('BASELINE-REPEAT', s, 0) in done},
            'gate_scope': 'new-fail vs BASELINE same suite+seed (operator 2026-11-10): every suite (strict, value transforms, hostile, edge) fails a run only on cases its paired BASELINE run passed, or on crash / case-count change; EQV adds per-element max-abs within 4x the BASELINE 5-seed max'}
def smoke():
    t0 = time.time(); seen = []
    for s in ('MMVF', 'GDN'):   # r5: Q/FA dropped from the smoke (Q always hit the 240 s smoke cap); residency+parse proven on the two short suites
        r = run_tbo(C9, s, 1000, f'smoke-{os.getpid()}-{s}', 240); seen.append({'suite': s, 'consistent': r['parse']['consistent'], 'n_fail': r['parse']['n_fail'], **{k: r[k] for k in ('rc', 'err', 'vram_peak', 'hip_loaded', 'seconds', 'strict_pass', 'passed')}})
    cv = coverage(C9, 'MMVF')
    ok = all(x['vram_peak'] > 0 and x['consistent'] for x in seen if not x['err']) and any(not x['err'] for x in seen) and time.time() - t0 >= 30
    u = {'unit_id': f'smoke-{now()}', 'mode': 'smoke', 'kind': 'smoke', 'ok': ok, 'seen': seen, 'coverage_names': cv['n_names'], 'cov_s': cv['seconds'], 'total_s': round(time.time() - t0, 1), 'planned_runs': sum(n for _, _, n in PLAN)}
    append(u); print(json.dumps(u)[:2000], flush=True); return 0 if ok else 3
def main():
    os.makedirs(LOGS, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    if SMOKE: return smoke()
    try: br = json.load(open(BUILDRES)); built = br.get('verdict') == 'BUILD_OK' and br.get('source_sha') == SHA
    except (OSError, ValueError): built = False
    if not built: print('mutant builds not ready; rc 75, no GPU work', flush=True); return 75
    builds = {}
    for v, _, _ in PLAN:
        if v in builds or v == 'BASELINE-REPEAT': continue
        b = json.load(open(f'{ST}/{v}/BUILD-OK.json'))
        if b['source_sha'] != SHA or sha256(os.path.realpath(f'{ST}/{v}/bin/libggml-hip.so')) != b['libggml_hip_sha256']: print('identity FAILED', v, flush=True); return 3
        builds[v] = b
    done = load_done()
    for (v, s, _), u in done.items():
        if v == 'BASELINE' and '+' in s and u.get('gate', {}).get('verdict') == 'EMPTY_SURFACE': EMPTY[s] = u['gate']['reason']
    cov = {}; cf = OUT + '/coverage.json'
    if os.path.exists(cf): cov = json.load(open(cf))
    for v, s, n in PLAN:
        if v != 'BASELINE' and s in EMPTY: continue
        if v != 'BASELINE':
            for su in ('Q', 'QPF', 'FA', 'GDN', 'MMVF'):
                if su in cov: continue
                if stop(): return 75
                cov[su] = coverage(f'{ST}/BASELINE/bin', su); jsave(cf, cov); append({'unit_id': f'coverage-{su}', 'mode': 'full', 'n_names': cov[su]['n_names'], 'seconds': cov[su]['seconds']})
        for r_i in range(n):
            if (v, s, r_i) in done: continue
            if stop(): return 75
            bv = 'BASELINE' if v == 'BASELINE-REPEAT' else v
            r = run_tbo(f'{ST}/{bv}/bin', s, 1000 + r_i, f'{v}-{s}-{r_i}', 2400)
            if r['err'] in ('stop', 'deadline'): return 75
            want = os.path.realpath(f'{ST}/{bv}/bin/libggml-hip.so')
            if r['vram_peak'] <= 0 or r['hip_loaded'] != want:
                jsave(RESULT, {'at': now(), 'complete': True, 'verdict': 'INVALID_NO_RESIDENCY', 'variant': v, 'suite': s, 'run': r_i, 'detail': r}); return 0
            if v == 'BASELINE':
                ok, why = nf.baseline_usable(r['parse'], tail_ok=(s == 'GDN+VT'))
                if not ok and '+' in s and not r['err'] and r['rc'] in (0, 1) and r['parse']['total'] < 50:
                    EMPTY[s] = why; r['gate'] = {'verdict': 'EMPTY_SURFACE', 'reason': why, 'cases': r['parse']['total']}; r['gate_pass'] = None
                    append({'unit_id': f'{v}|{s}|{r_i}', 'mode': 'full', 'kind': 'run', 'variant': v, 'suite': s, 'run': r_i, **r}); continue
                r['gate'] = {'verdict': 'USABLE' if ok else 'UNUSABLE', 'reason': why}; r['gate_pass'] = ok
            elif v == 'BASELINE-REPEAT':
                d = nf.determinism(done[('BASELINE', s, 0)]['parse'], r['parse']); r['gate'] = dict(d, verdict='DETERMINISTIC' if d['deterministic'] else 'NONDETERMINISTIC'); r['gate_pass'] = d['deterministic']
            else:
                r['gate'] = nf.compare(done[('BASELINE', s, r_i)]['parse'], r['parse']); r['gate_pass'] = r['gate']['gate_pass']
            u = {'unit_id': f'{v}|{s}|{r_i}', 'mode': 'full', 'kind': 'run', 'variant': v, 'suite': s, 'run': r_i, **r}; append(u); done[(v, s, r_i)] = u
            fut = {'BASELINE': 'INVALID_BASELINE_UNUSABLE', 'BASELINE-REPEAT': 'INVALID_BASELINE_NONDETERMINISTIC'}.get(v)
            if fut and not r['gate_pass']:
                jsave(RESULT, {'at': now(), 'complete': True, 'verdict': fut, 'suite': s, 'run': r_i, 'gate': r['gate'], 'detail': {k: x for k, x in r.items() if k != 'parse'}}); return 0
            if r['gate']['verdict'] == 'UNPARSED':
                jsave(RESULT, {'at': now(), 'complete': True, 'verdict': 'INVALID_HARNESS', 'reason': 'candidate fail count does not reconcile', 'variant': v, 'suite': s, 'run': r_i}); return 0
    res = verdict(done, builds, cov); jsave(RESULT, res); append({'unit_id': 'verdict', 'mode': 'full', 'verdict': res['verdict']}); print('RESULT', res['verdict'], flush=True); return 0
if __name__ == '__main__':
    sys.exit(main())
