#!/usr/bin/env python3
"""AK-ASM-T1 GPU correctness harness: retargeted gfx942 -> gfx90a AITER asm code objects, judged by AITER's own op_tests against
AITER's torch references, on the MI210.

Tree: /mnt/raid0/llm/tmp/ak-asm-t1-20261105/aiter-src-90a (cp -a of rock-aiter-prebuild src-gfx90a), prepared OFFLINE by
install_90a.py: hsa/gfx90a/ = retargeted .co + gfx942 CSV manifests filtered to installed kernels, arch allow-lists opened for
gfx90a (gates_90a.json), checkAllclose instrumented (ASMT1_CHECK lines), asm-embedding JIT modules renamed *.pre90a so the
first use rebuilds them with the gfx90a manifests (JIT build happens inside the unit, covered by its timeout).

PRECONDITION (job.py --precond; launcher --precond-cmd): INSTALL_MANIFEST.json present and every installed .co still matches its
sha256, and the unit's family has >= 1 installed .co. Otherwise exit 5 `install-missing: <reason>`.

One UNIT = one kernel family = one fresh container (cpuset 184-191, /dev/kfd + /dev/dri) = one fresh test process (op_runner.py:
--help logged, args adapted, optional test-copy patch). Verdict per unit, decided only from that family's evidence:
  PASS  >= 1 LoadKernel of hsa/gfx90a/<family> in the log (the retargeted object really ran), >= 1 family check, all of them pass
        (checkAllclose ret <= tol_err_ratio, no NaN; mha: every Output max diff <= tol), max_abs reported.
  FAIL  any family check fails (real numerics), max_abs reported.
  HANG  runner hang watchdog fired (no output for idle_s after a gfx90a LoadKernel, no JIT build running) or the hard timeout hit
        after a gfx90a LoadKernel -> HANG.json written, the packet STOPS (rc 3); remaining units are not started.
  ERROR infra (import, JIT build failure, timeout during build, missing kernel / no heuristic, invalid device function) or
        not-exercised (no gfx90a LoadKernel, or no family check: vacuous).
Stop rule: single pass, no repeats; STOP file / deadline -> rc 75 resumable (finished units are not re-run).
Smoke (FILLER_SMOKE / FILLER_DRYRUN = 1): smoke_hold.py only: one process, >= 50 s GPU residency, retargeted topksoftmax vs torch."""
import json, os, re, subprocess, sys, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
T1 = '/mnt/raid0/llm/tmp/ak-asm-t1-20261105'
SRC = T1 + '/aiter-src-90a-r3'
PYDEPS = '/mnt/raid0/llm/tmp/rock-aiter-gfx90a-20261105/rock-aiter-prebuild-20261105/pydeps'
IMG = 'vllm/vllm-openai-rocm@sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba'
OUT = D + '/out'; RES = OUT + '/results.jsonl'; RESULT = OUT + '/RESULT.json'; HANGF = OUT + '/HANG.json'
MANIFEST = D + '/INSTALL_MANIFEST.json'
STOPF = os.environ.get('FILLER_STOPFILE', D + '/STOP')
DEADLINE = int(os.environ.get('FILLER_DEADLINE', str(int(time.time()) + 6 * 3600)))
PLUMB = os.environ.get('FILLER_DRYRUN') == '1' or os.environ.get('FILLER_SMOKE') == '1'
CPUSET = os.environ.get('FILLER_CPUS') or '184-191'


def G(flags, *values, tup=None): return {'flags': flags if isinstance(flags, list) else [flags], 'values': list(values), 'tuple': tup}


# test-copy patch: test_pa.py's quant sweep minus the fp8 KV entries (no fp8 .co is installed; int8 KV and bf16/fp16 KV stay)
PA_PATCH = {'dst': 'op_tests/test_pa_asmt1.py', 'ops': [{'do': 'comment', 'regex': r'^\s+\(2, dtypes\.fp8\),\s*$'},
                                                        {'do': 'comment', 'regex': r'^\s+\(4, dtypes\.fp8\),\s*$'}]}
MOE = [G(['-d', '--dtype'], 'bf16'), G(['-m', '--token'], 32, 128), G(['-hd', '--hidden_dim'], 4096), G(['-id', '--inter_dim'], 1024),
       G(['-e', '--expert'], 32), G(['-k', '--topk'], 5)]


def U(unit, family, script, groups, load_re, msg_re=None, timeout=3600, patch=None, tol=None, idle_s=600):
    return dict(unit=unit, family=family, script=script, groups=groups, load_re=load_re, msg_re=msg_re, timeout=timeout, patch=patch,
                tol=tol, idle_s=idle_s)


# r3 (ak-asm-t1-harness-r3-20261106-r2): retarget now 782/782 non-fp8 (R9 scratch spill lifts the 53 no-dead-register
# refusals). New: fmoe-16bit (fmoe_b16.co) and fmoe-g1u0 at -id 1024 (subGU_512), both spill-path kernels. Regression: every
# r1/r2-passing family. bf16gemm bf16out/fp32out pair again (the 1-ulp question). splitk-diag dropped (r2 logged no checks).
def moe(idim): return [G(['-d', '--dtype'], 'bf16'), G(['-m', '--token'], 32, 128), G(['-hd', '--hidden_dim'], 4096),
                       G(['-id', '--inter_dim'], *idim), G(['-e', '--expert'], 32), G(['-k', '--topk'], 5)]


GEMM16 = lambda ot: [G(['-t', '--test'], 'normal'), G(['-d', '--dtype'], 'bf16'), G(['-o', '--otype'], ot),
                     G('-mnk', '16,1280,8192', '128,1280,8192', '256,1024,1024', tup='comma')]
SMOKE = U('smoke-hold-topksoftmax', 'topksoftmax', '/pk/smoke_hold.py', [G('--hold_s', 50)], r'/gfx90a/topksoftmax/', msg_re=r'asm',
          timeout=1800)
UNITS = [
    U('fmoe-16bit', 'fmoe', 'op_tests/test_moe.py', [G(['-t', '--test'], 'test_fmoe_16_bit')] + moe([1024]), r'/gfx90a/fmoe_b16', timeout=3600),
    U('fmoe-g1u0-int8smoothquant-id1024', 'fmoe', 'op_tests/test_moe.py', [G(['-t', '--test'], 'g1u0_int8smoothquant')] + moe([1024]),
      r'/gfx90a/fmoe/(silu|gelu)/fmoe_int8_g1u0_subGU_512', timeout=3000),
    U('fmoe-g1u0-int8smoothquant-id384', 'fmoe', 'op_tests/test_moe.py', [G(['-t', '--test'], 'g1u0_int8smoothquant')] + moe([384, 768]),
      r'/gfx90a/fmoe', timeout=3000),
    U('fmoe-g1u1-int8smoothquant', 'fmoe', 'op_tests/test_moe.py', [G(['-t', '--test'], 'g1u1_int8smoothquant')] + moe([1024]),
      r'/gfx90a/fmoe', timeout=3000),
    U('fmoe-g1u1-int8quant', 'fmoe', 'op_tests/test_moe.py', [G(['-t', '--test'], 'g1u1_int8quant')] + moe([1024]), r'/gfx90a/fmoe', timeout=3000),
    U('topksoftmax', 'topksoftmax', 'op_tests/test_moeTopkSoftmax.py',
      [G(['-d', '--dtype'], 'bf16'), G(['-e', '--expert'], 128, 256), G(['-t', '--token'], 1, 64, 1024), G('-k', 8)],
      r'/gfx90a/topksoftmax/', msg_re=r'asm', timeout=2400),
    U('pa-bf16', 'pa', 'op_tests/test_pa.py', [G(['-d', '--dtype'], 'bf16'), G(['-n', '--num_heads'], '8,1', '16,1', tup='comma'),
      G(['-c', '--ctx_len'], 7, 128, 1024)], r'/gfx90a/(pa/|pa_)', msg_re=r'asm', timeout=3000, patch=PA_PATCH),
    U('pa-fp16', 'pa', 'op_tests/test_pa.py', [G(['-d', '--dtype'], 'fp16'), G(['-n', '--num_heads'], '8,1', tup='comma'),
      G(['-c', '--ctx_len'], 7, 1024)], r'/gfx90a/(pa/|pa_)', msg_re=r'asm', timeout=2400, patch=PA_PATCH),
    U('mhc_seam', 'mhc_seam', 'op_tests/test_mhc_seam.py', [G('-m', 16, 64, 1024, 4096)], r'/gfx90a/mhc_seam/', timeout=2400),
    U('i8gemm', 'i8gemm', 'op_tests/test_gemm_a8w8.py',
      [G(['-d', '--dtype'], 'bf16'), G(['-q', '--quantDtype'], 'i8'), G('--quant-mode', 'per-token'), G('--no-flydsl-csv'),
       G('-mnk', '32,1280,8192', '128,1280,8192', '1024,7168,2048', tup='comma')], r'/gfx90a/i8gemm/', msg_re=r'^asm', timeout=2400),
    U('fmha_v3_fwd', 'fmha_v3_fwd', '/pk/test_mha_fwd_asmt1.py',
      [G(['-b', '--batch_size'], 2), G(['-n', '--nheads'], 16), G(['-q', '--seqlen_q'], 512, 1024), G(['-k', '--seqlen_k'], 512, 1024)],
      r'/gfx90a/fmha_v3_fwd/', timeout=3600, tol=0.03),
    # the 1-ulp question: bf16 out accumulates split-K through R7; fp32 out uses global_atomic_add_f32 (no R7) as the control
    U('bf16gemm-bf16out', 'bf16gemm', 'op_tests/test_gemm_a16w16.py', GEMM16('bf16'), r'/gfx90a/bf16gemm/', msg_re=r'asm-', timeout=2400),
    U('bf16gemm-fp32out', 'bf16gemm', 'op_tests/test_gemm_a16w16.py', GEMM16('fp32'), r'/gfx90a/bf16gemm/', msg_re=r'asm-', timeout=2400),
]
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
INFRA_RE = re.compile(r'ModuleNotFoundError|ImportError|OutOfMemoryError|out of memory|cannot get heuristic kernel|hasco_obj not found|'
                      r'failed to open|hipErrorNoBinaryForGpu|invalid device function|core dumped|Aborted|Segmentation fault|'
                      r'failed jit build|FileNotFoundError|No kernel|not found in|NotImplementedError', re.I)


def now(): return time.strftime('%FT%TZ', time.gmtime())


def manifest():
    try: return json.load(open(MANIFEST))
    except (OSError, ValueError): return None


def precond(units):
    m = manifest()
    if m is None: return False, f'{MANIFEST} missing (run install_90a.py)'
    import hashlib
    bad = []
    for rel, v in m['co'].items():
        p = f'{SRC}/hsa/gfx90a/{rel}'
        try: h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        except OSError: bad.append(rel); continue
        if h != v['sha256']: bad.append(rel)
    if bad: return False, f'{len(bad)} installed .co missing/changed, e.g. {bad[:3]}'
    fams = {u['family'] for u in units}
    have = {(rel.split('/')[0] if '/' in rel else rel) for rel in m['co']}
    tops = {rel for rel in m['co'] if '/' not in rel}
    miss = [f for f in fams if f not in have and not any(t.startswith(f) for t in tops)]
    if miss: return False, f'no installed .co for families {miss}'
    return True, ''


def append(u):
    with open(RES, 'a') as f: f.write(json.dumps(u) + '\n'); f.flush(); os.fsync(f.fileno())


def done():
    s = set()
    try:
        for l in open(RES):
            try: u = json.loads(l)
            except ValueError: continue
            if u.get('mode') == 'full' and u.get('final'): s.add(u['unit_id'])
    except OSError: pass
    return s


def judge(unit, txt, runner, rc, err):
    t = ANSI.sub('', txt).split('==RUN==', 1)[-1]
    loads = sorted(set(re.findall(r'LoadKernel: (\S+) hsaco: (\S+)', t)))
    fam_loads = [x for x in loads if re.search(unit['load_re'], x[1])]
    checks = []
    for l in t.splitlines():
        if l.startswith('ASMT1_CHECK '):
            try: checks.append(json.loads(l[len('ASMT1_CHECK '):]))
            except ValueError: pass
    fam = [c for c in checks if unit['msg_re'] is None or re.search(unit['msg_re'], c.get('msg', ''))]
    diffs = [float(x) for x in re.findall(r'Output\s+max diff[^:]*:\s*([0-9.eE+-]+)', t) if re.match(r'^[0-9.eE+-]+$', x)]
    bad = [c for c in fam if c.get('ret') is None or c['ret'] > c.get('tol_err_ratio', 0.05) or c.get('nan')]
    over = [d for d in diffs if unit['tol'] is not None and d > unit['tol']]
    mx = max([c.get('max_abs', 0.0) for c in fam] + diffs, default=None)
    info = {'gfx90a_loads': [f'{k} <- {p}' for k, p in fam_loads][:40], 'n_gfx90a_loads': len(fam_loads),
            'other_loads': [p for k, p in loads if (k, p) not in fam_loads][:10], 'n_family_checks': len(fam), 'n_all_checks': len(checks),
            'n_family_fail': len(bad), 'max_abs_err': mx, 'mha_max_diffs': diffs[:20], 'worst': sorted(fam, key=lambda c: -c.get('max_abs', 0))[:3],
            'non_family_fail': [c['msg'][:120] for c in checks if c not in fam and (c.get('ret') or 0) > c.get('tol_err_ratio', 0.05)][:5]}
    infra = [l.strip()[:300] for l in t.splitlines() if INFRA_RE.search(l) and not l.startswith(('==', 'usage', 'ASMT1_CHECK'))][:6]
    info['infra_lines'] = infra
    if err in ('stop', 'deadline'): return 'interrupted', err, info
    if runner and runner.get('hang'): return 'hang', f"watchdog: no output {unit['idle_s']} s after gfx90a LoadKernel", info
    if (runner and runner.get('timed_out')) or err == 'op_timeout':
        if fam_loads: return 'hang', f"hard timeout {unit['timeout']} s after gfx90a LoadKernel", info
        return 'error', f"timeout {unit['timeout']} s before any gfx90a kernel ran (JIT build?)", info
    if runner is None or runner.get('runner_error'): return 'error', (runner or {}).get('runner_error', 'no RUNNER_RESULT in log'), info
    if bad or over:
        return 'fail', f"{len(bad)}/{len(fam)} family checks failed" + (f", {len(over)} mha max-diff > {unit['tol']}" if over else '') + f"; max_abs={mx}", info
    if not fam_loads: return 'error', 'not exercised: no LoadKernel from hsa/gfx90a for this family' + (f' ({infra[0]})' if infra else ''), info
    if not fam and not diffs: return 'error', 'vacuous: gfx90a kernel loaded but no family check in log' + (f' ({infra[0]})' if infra else ''), info
    if rc != 0 and infra: return 'error', f'rc={rc} after {len(fam)} passing checks: {infra[0]}', info
    return 'pass', f"{len(fam)} family checks passed" + (f", {len(diffs)} mha diffs <= {unit['tol']}" if diffs else '') + f"; max_abs={mx}", info


def run_one(unit, mode):
    import gpuproof as gp
    cname = f'ak-asm-t1-h3-{re.sub("[^a-z0-9]", "-", unit["unit"].lower())[:30]}-{os.getpid()}'
    log = f'{OUT}/units/{unit["unit"]}-{mode}-{int(time.time())}.log'; os.makedirs(os.path.dirname(log), exist_ok=True)
    spec = {'run': unit['unit'], 'op': unit['family'], 'script': unit['script'], 'groups': unit['groups'], 'timeout': unit['timeout'],
            'patch': unit['patch'], 'idle_s': unit['idle_s']}
    inner = f"cd /aiter && pip uninstall -y amd-aiter aiter >/dev/null 2>&1 || true; python3 /pk/op_runner.py '{json.dumps(spec)}'"
    env = {'GPU_ARCHS': 'gfx90a', 'PYTORCH_ROCM_ARCH': 'gfx90a', 'MAX_JOBS': '8', 'AITER_JIT_DIR': '/aiter/aiter/jit',
           'PYTHONPATH': '/aiter:/pydeps', 'HIP_VISIBLE_DEVICES': '0', 'AITER_LOG_LEVEL': 'INFO'}
    argv = ['docker', 'run', '--rm', '--name', cname, '--entrypoint', '/bin/bash', '--device', '/dev/kfd', '--device', '/dev/dri',
            '--group-add', 'video', '--ipc', 'host', '--security-opt', 'seccomp=unconfined', '--cpuset-cpus', CPUSET, '--ulimit', 'core=0']
    for k, v in env.items(): argv += ['-e', f'{k}={v}']
    argv += ['-v', f'{SRC}:/aiter', '-v', f'{PYDEPS}:/pydeps', '-v', f'{D}:/pk:ro', IMG, '-c', inner]
    t0 = time.time()
    rc, err, proof = gp.run_sampled(argv, log, timeout_at=min(DEADLINE - 30, time.time() + unit['timeout'] + 900), stopfile=STOPF,
                                    container_name=cname)
    # keeper VRAM sampler polls every 5 s and stops at the first unit: in smoke mode keep a VRAM-resident GPU process alive
    # >= SMOKE_HOLD_S by re-running the same unit container BEFORE the smoke unit is written (rowflow residency gate).
    # The first run's rc/err/proof/log are what gets judged and recorded; hold runs log to *-hold-N.log.
    SMOKE_HOLD_S = 45; _n = 0
    while mode == 'smoke' and err is None and time.time() - t0 < SMOKE_HOLD_S and not os.path.exists(STOPF) and time.time() < DEADLINE - 120:
        _n += 1; hname = f'{cname}-hold{_n}'
        hargv = [hname if a == cname else a for a in argv]
        gp.run_sampled(hargv, log[:-4] + f'-hold-{_n}.log', timeout_at=min(DEADLINE - 30, time.time() + unit['timeout'] + 900),
                       stopfile=STOPF, container_name=hname)
    txt = open(log, errors='replace').read(); runner = None
    for l in txt.splitlines():
        if l.startswith('RUNNER_RESULT '):
            try: runner = json.loads(l[len('RUNNER_RESULT '):])
            except ValueError: pass
    if err == 'deadline' and time.time() < DEADLINE - 90: err = 'op_timeout'
    status, reason, info = judge(unit, txt, runner, rc, err)
    u = {'unit_id': unit['unit'] if mode == 'full' else f'smoke-{unit["unit"]}-{now()}', 'unit': unit['unit'], 'family': unit['family'],
         'mode': mode, 'at': now(), 'script': unit['script'], 'rc': rc, 'run_error': err, 'status': status, 'reason': reason, **info,
         'adaptations': (runner or {}).get('adaptations'), 'argv': (runner or {}).get('argv'), 'patch': (runner or {}).get('patch'),
         'jit_during_test': (runner or {}).get('jit_during_test'), 'wall_s': round(time.time() - t0, 1), 'log': log,
         'vram_peak_gib': round(max(proof['kfd_pids_ours'].values(), default=0) / 1024**3, 3),
         'kfd_foreign_pid_count': len(proof['kfd_foreign']), 'proof': proof, 'final': status != 'interrupted'}
    append(u); print(f"{u['unit_id']} {status} {reason[:200]}", flush=True)
    return u


def write_result():
    last = {}
    for l in open(RES):
        try: u = json.loads(l)
        except ValueError: continue
        if u.get('mode') == 'full' and u.get('final'): last[u['unit_id']] = u
    fams = {}
    for un in UNITS:
        fams.setdefault(un['family'], [])
        if un['unit'] in last: fams[un['family']].append(last[un['unit']])
    out = []
    for f, rs in fams.items():
        st = [r['status'] for r in rs]
        top = 'hang' if 'hang' in st else 'fail' if 'fail' in st else 'error' if 'error' in st else 'pass' if rs and all(s == 'pass' for s in st) else 'pending'
        mx = max([r['max_abs_err'] for r in rs if r.get('max_abs_err') is not None], default=None)
        out.append({'family': f, 'status': top, 'max_abs_err': mx,
                    'units': [{k: r.get(k) for k in ('unit', 'status', 'reason', 'n_gfx90a_loads', 'n_family_checks', 'max_abs_err', 'wall_s')} for r in rs]})
    hang = os.path.exists(HANGF)
    json.dump({'at': now(), 'aiter_src': SRC, 'image': IMG, 'gpu_arch': 'gfx90a', 'families': out,
               'passed': [x['family'] for x in out if x['status'] == 'pass'], 'failed': [x['family'] for x in out if x['status'] == 'fail'],
               'errored': [x['family'] for x in out if x['status'] == 'error'], 'hung': [x['family'] for x in out if x['status'] == 'hang'],
               'stopped_on_hang': hang, 'complete': hang or all(u['unit'] in last for u in UNITS)}, open(RESULT, 'w'), indent=1)


def clear_stale_locks():
    """r1 lost 35 min to a stale JIT lock (lock_mha_bwd_*) inherited from src-gfx90a by cp -a. When no container of this
    harness (name ak-asm-t1-h*) is running, move each aiter/jit/build/lock_* FILE of the harness copy aside to *.stale-<ts>."""
    r = subprocess.run(['docker', 'ps', '-q', '--filter', 'name=ak-asm-t1-h'], capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip():
        print('lock cleanup skipped: ' + ('docker ps failed' if r.returncode else 'a harness container is running'), flush=True); return
    import glob
    for f in glob.glob(SRC + '/aiter/jit/build/lock_*'):
        if os.path.isfile(f) and '.stale-' not in f:
            os.rename(f, f'{f}.stale-{int(time.time())}'); print(f'moved stale JIT lock aside: {f}', flush=True)


def main():
    os.makedirs(OUT + '/units', exist_ok=True)
    units = [SMOKE] if PLUMB else UNITS
    if '--precond' in sys.argv:
        ok, why = precond(units); print('' if ok else why); return 0 if ok else 1
    ok, why = precond(units)
    if not ok:
        print(f'install-missing: {why}', flush=True)
        json.dump({'at': now(), 'reason': 'install-missing', 'detail': why}, open(OUT + '/INSTALL_MISSING.json', 'w'), indent=1)
        return 5
    clear_stale_locks()
    if os.path.exists(HANGF):
        print(f'previous run stopped on a GPU hang ({HANGF}); operator must clear it before re-running', flush=True); return 3
    if subprocess.run(['docker', 'image', 'inspect', IMG], capture_output=True).returncode != 0:
        print('pulling pinned image by digest', flush=True)
        if subprocess.run(['docker', 'pull', IMG]).returncode != 0: print('image pull failed', flush=True); return 4
    mode = 'smoke' if PLUMB else 'full'
    have = done() if mode == 'full' else set()
    for un in units:
        if un['unit'] in have: continue
        if os.path.exists(STOPF) or time.time() > DEADLINE - 120:
            print('stop/deadline: resumable exit', flush=True); return 75
        u = run_one(un, mode)
        if u['status'] == 'hang':
            json.dump({'at': now(), 'unit': un['unit'], 'family': un['family'], 'reason': u['reason'], 'log': u['log']}, open(HANGF, 'w'), indent=1)
            if mode == 'full': write_result()
            print('HARD HANG: packet stopped; remaining units not started', flush=True); return 3
        if not u['final']: print('interrupted: resumable exit', flush=True); return 75
    if mode == 'full': write_result(); print('RESULT written', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
