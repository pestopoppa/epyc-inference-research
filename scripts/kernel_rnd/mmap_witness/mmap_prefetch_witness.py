"""LLAMA_NO_MMAP_PREFETCH default-off witness for a kernel store (CPU, correctness only, no timing).

Why this exists (c6-features-cpu-r3-20261105, INCONCLUSIVE): the r3 positive control read the model file's page-cache
residency AFTER /health. That observable saturates: the CPU loader touches every mapped page whether or not it
prefetches, so every arm (including LLAMA_NO_MMAP_PREFETCH=1) ends at residency 1.0. The eviction itself worked
(mincore 0.0 before each arm); the instrument looked at the end state instead of the mmap call that differs.
The witness here reads the call: an LD_PRELOAD shim logs mmap flags of the .gguf mapping; MAP_POPULATE present =
prefetch path, absent = no-prefetch. Polarity: the knob value 1 DISABLES prefetch; unset and 0 are the prefetch path.

Verdicts (a positive-control failure is FUTILITY, never a verdict about the build):
  FUTILITY_CONTROL   shim saw no .gguf mapping in some arm, or the three arms do not differ in MAP_POPULATE at all
  PASS               control separates, unset and =0 populate, =1 does not, unset tokens == =0 tokens,
                     no-prefetch log line only in =1
  FAIL               control separates but the pattern is wrong (e.g. unset does not populate)
"""
import json
import os
import re
import resource
import subprocess
import time
import urllib.request
from pathlib import Path

LINE = 'mmap eager prefetch disabled by LLAMA_NO_MMAP_PREFETCH=1'
ARMS = [('unset', None), ('zero', '0'), ('one-noprefetch', '1')]


def parse_shim_log(text):
    """Largest successful .gguf mapping from the shim's JSONL, or None."""
    maps = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            m = json.loads(ln)
        except ValueError:
            continue
        if m.get('ok') and str(m.get('target', '')).endswith('.gguf'):
            maps.append(m)
    return max(maps, key=lambda m: m['len']) if maps else None


def classify(arms):
    """arms: label -> {'top': mapping|None, 'line': bool, 'tokens': list|None}. Returns (verdict, checks)."""
    have = all(arms[k]['top'] for k, _ in ARMS)
    pop = {k: bool(arms[k]['top'] and arms[k]['top']['populate']) for k, _ in ARMS}
    separates = have and len(set(pop.values())) > 1
    checks = {'shim_saw_gguf_every_arm': have, 'control_separates': separates}
    if not (have and separates):
        return 'FUTILITY_CONTROL', checks
    checks.update(unset_populate=pop['unset'], zero_populate=pop['zero'], one_no_populate=not pop['one-noprefetch'],
                  line_only_in_one=(not arms['unset']['line'] and not arms['zero']['line'] and arms['one-noprefetch']['line']),
                  unset_tokens_eq_zero=arms['unset']['tokens'] is not None and arms['unset']['tokens'] == arms['zero']['tokens'])
    return ('PASS' if all(checks.values()) else 'FAIL'), checks


def no_core():
    # rlimit only; NO setsid: children stay in the row's process group so the engine's stop reaches them
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_arm(store, model, shim_so, run_dir, label, val, port, threads, prompts, n_predict, stop_check, with_tokens=True):
    stop_check()
    maplog, log = run_dir / f'{label}.mmaps.jsonl', run_dir / f'{label}.server.log'
    env = {k: v for k, v in os.environ.items() if k not in ('LLAMA_NO_MMAP_PREFETCH', 'LD_PRELOAD')}
    env.update(LD_LIBRARY_PATH=f'{store}/bin:/opt/rocm/lib', GGML_IQK='1', OMP_NUM_THREADS=str(threads),
               HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='',
               LD_PRELOAD=str(shim_so), MMAP_SHIM_LOG=str(maplog))
    if val is not None:
        env['LLAMA_NO_MMAP_PREFETCH'] = val
    argv = [str(store / 'bin/llama-server'), '-m', str(model), '--host', '127.0.0.1', '--port', str(port), '-c', '2048',
            '-np', '1', '--no-warmup', '-t', str(threads), '-ngl', '0', '--device', 'none', '--no-repack', '-v',
            '--seed', '1']
    tokens = []
    with log.open('wb') as out:
        p = subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT, preexec_fn=no_core)
        try:
            t0 = time.time()
            while True:
                stop_check()
                if p.poll() is not None:
                    raise RuntimeError(f'{label}: server exited rc={p.returncode} before health')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                if time.time() - t0 > 300:
                    raise RuntimeError(f'{label}: no health in 300 s')
                time.sleep(1)
            if with_tokens:
                for pr in prompts:
                    req = urllib.request.Request(
                        f'http://127.0.0.1:{port}/completion', headers={'Content-Type': 'application/json'},
                        data=json.dumps({'prompt': pr, 'n_predict': n_predict, 'temperature': 0, 'top_k': 1, 'seed': 1,
                                         'cache_prompt': False, 'return_tokens': True, 'ignore_eos': True}).encode())
                    with urllib.request.urlopen(req, timeout=300) as r:
                        tokens.append(json.loads(r.read()).get('tokens'))
        finally:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(30)
    if Path(f'/proc/{p.pid}').exists():
        raise RuntimeError(f'{label}: server pid {p.pid} still present after kill')
    top = parse_shim_log(maplog.read_text()) if maplog.exists() else None
    return {'top': top, 'line': LINE in log.read_text(errors='replace'), 'tokens': tokens or None, 'rc': p.returncode,
            'log': str(log)}
