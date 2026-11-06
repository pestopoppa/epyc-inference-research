#!/usr/bin/env python3
"""ak-setup-2-proof-20261106 (autokernel-loop.md AK-SETUP-2): ONE serial (-j1) and ONE parallel (-j<quarter threads>) CPU build of the SAME anchor,
then build_parallel_proof.prove(). Anchor: llama.cpp-experimental worktree llama-c7-asm-ports-20261106 @ c420b9d1a (read-only; builds go to
scratch). ccache OFF (a cache hit would fake the wall), llguidance OFF (offline rust build, identical for both arms), targets llama-bench +
test-backend-ops. Thread env comes from the shared job_fence (affinity count). rc: 0 done | 75 stop | 3 preflight.
Final record carries wall_drop_pct, digests_identical, smoke_clean, accepted (1/0) for file_result.py."""
import json, os, shutil, subprocess, sys, time
sys.path.insert(0, '/mnt/raid0/llm/tmp/compute-flags/bin/job_templates')
import job_fence  # noqa: E402
job_fence.set_thread_defaults()
PRUNE = '/mnt/raid0/llm/tmp/wt-intake-close-research/scripts/kernel_rnd/autokernel/controller'
if not os.path.isdir(PRUNE):
    PRUNE = '/workspace/repos/epyc-inference-research/scripts/kernel_rnd/autokernel/controller'
sys.path.insert(0, PRUNE)
import build_parallel_proof as bpp  # noqa: E402
from pathlib import Path  # noqa: E402
D = os.path.dirname(os.path.abspath(__file__))
WT = '/mnt/raid0/llm/worktrees/llama-c7-asm-ports-20261106'
SHA = 'c420b9d1aff85555e966b9633cc841b3c48416a3'
UIDIST = '/mnt/raid0/llm/kernels/builds/cpu-20261103-181483cfc-lbt/tools/ui/dist'
TARGETS = ['llama-bench', 'test-backend-ops']
OUT = D + '/out'; RES = OUT + '/results.jsonl'
STOPF = os.environ.get('FILLER_STOPFILE', D + '/STOP')
DEADLINE = int(os.environ.get('FILLER_DEADLINE', str(int(time.time()) + 6 * 3600)))
PLUMB = os.environ.get('FILLER_DRYRUN') == '1' or os.environ.get('FILLER_SMOKE') == '1'
JOBS_PAR = job_fence.affinity_threads()
def now(): return time.strftime('%FT%TZ', time.gmtime())
def append(u):
    with open(RES, 'a') as f: f.write(json.dumps(u) + '\n'); f.flush(); os.fsync(f.fileno())
def git(*a): return subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
def stop(): return os.path.exists(STOPF) or time.time() > DEADLINE - 120
def env():
    e = dict(os.environ); e.update(CCACHE_DISABLE='1', PATH='/usr/bin:/bin'); return e
def configure(bdir, log):
    a = json.load(open(D + '/configure_argv-cpu.json')); a[a.index('-B') + 1] = bdir
    mp = f'-ffile-prefix-map={WT}=. -fmacro-prefix-map={WT}=. -ffile-prefix-map={bdir}=.build -fmacro-prefix-map={bdir}=.build'
    a = [f'-DCMAKE_CXX_FLAGS:STRING={mp}' if x.startswith('-DCMAKE_CXX_FLAGS:') else f'-DCMAKE_C_FLAGS:STRING={mp}' if x.startswith('-DCMAKE_C_FLAGS:') else x for x in a]
    os.makedirs(bdir + '/tools/ui', exist_ok=True)
    if not os.path.isdir(bdir + '/tools/ui/dist'): shutil.copytree(UIDIST, bdir + '/tools/ui/dist')
    with open(log, 'wb') as f: return subprocess.run(a, stdout=f, stderr=subprocess.STDOUT, cwd=D, env=env()).returncode
def build(arm, jobs):
    B = f'{D}/build-{arm}'; done = f'{OUT}/{arm}.json'
    if os.path.exists(done): return json.load(open(done))
    shutil.rmtree(B, ignore_errors=True)
    rc = configure(B, f'{OUT}/logs/configure-{arm}.log')
    if rc != 0: return {'arm': arm, 'ok': False, 'stage': 'configure', 'rc': rc}
    t0 = time.time()
    with open(f'{OUT}/logs/build-{arm}.log', 'wb') as f:
        p = subprocess.Popen(['nice', '-n', '10', '/usr/bin/cmake', '--build', B, '--parallel', str(jobs), '--target', *TARGETS],
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env())
        while p.poll() is None:
            if stop():
                os.killpg(p.pid, 15)
                try: p.wait(60)
                except subprocess.TimeoutExpired: os.killpg(p.pid, 9); p.wait()
                return None
            time.sleep(5)
    r = {'arm': arm, 'ok': p.returncode == 0, 'jobs': jobs, 'wall_s': round(time.time() - t0, 1), 'build_dir': B, 'rc': p.returncode, 'at': now()}
    json.dump(r, open(done, 'w'))
    append({'unit_id': f'{arm}-build', 'mode': 'full', **r})
    return r
def main():
    os.makedirs(OUT + '/logs', exist_ok=True)
    head = git('-C', WT, 'rev-parse', 'HEAD')
    if head != SHA or git('-C', WT, 'status', '--porcelain', '--untracked-files=no') or not os.path.isdir(UIDIST):
        append({'unit_id': f'preflight-fail-{now()}', 'mode': 'smoke' if PLUMB else 'full', 'head': head}); return 3
    if PLUMB:
        q = f'{D}/qual-{os.getpid()}'; rc = configure(q, f'{OUT}/logs/configure-smoke-{os.getpid()}.log')
        append({'unit_id': f'smoke-configure-{now()}', 'mode': 'smoke', 'configure_rc': rc, 'ok': rc == 0, 'par_jobs': JOBS_PAR})
        shutil.rmtree(q, ignore_errors=True); return 0 if rc == 0 else 3
    arms = {}
    for arm, jobs in (('serial', 1), ('parallel', JOBS_PAR)):
        if stop(): return 75
        r = build(arm, jobs)
        if r is None: return 75
        arms[arm] = r
        if not r['ok']:
            append({'unit_id': 'verdict', 'mode': 'full', 'accepted': 0, 'wall_drop_pct': 0, 'digests_identical': 0, 'smoke_clean': 0, 'reason': f'{arm} build failed', 'arm': r}); return 0
    pr = bpp.prove(serial_dir=Path(arms['serial']['build_dir']), serial_walls=[arms['serial']['wall_s']], serial_jobs=1,
                   parallel_dir=Path(arms['parallel']['build_dir']), parallel_walls=[arms['parallel']['wall_s']], parallel_jobs=arms['parallel']['jobs'])
    rec = {'unit_id': 'verdict', 'mode': 'full', 'accepted': int(pr['accepted']), 'wall_drop_pct': round((pr['median_wall_drop'] or 0) * 100, 2),
           'digests_identical': int(pr['digests_identical']), 'smoke_clean': int(pr['smoke_clean']), 'reasons': pr['reasons'],
           'serial_wall_s': arms['serial']['wall_s'], 'parallel_wall_s': arms['parallel']['wall_s'], 'serial_jobs': 1,
           'parallel_jobs': arms['parallel']['jobs'], 'anchor': SHA, 'proof': pr}
    append(rec); json.dump(rec, open(OUT + '/RESULT.json', 'w'), indent=1); print('RESULT accepted=', rec['accepted'], flush=True); return 0
if __name__ == '__main__': sys.exit(main())
