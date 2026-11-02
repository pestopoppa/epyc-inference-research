#!/usr/bin/env python3
"""Predeclare and collect an architect_critic load-only, sampled VRAM record.

Run from the pinned orchestrator checkout, with that checkout's Python path.
No completion or generation request is ever sent. A sample maximum is not a peak.
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import re
from pathlib import Path
import signal
import sys
import time

import device_identity
import sampler
import server_lifecycle

ENV_PREFIXES = ('LLAMA_', 'GGML_', 'OMP_', 'KMP_', 'HIP_', 'HSA_', 'ROCR_', 'CUDA_', 'LD_')
ENV_NAMES = {'PATH', 'ROCM_PATH'}
ASSET_FLAGS = ('-m', '-md', '--chat-template-file')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_env(env):
    return {k: v for k, v in env.items() if k in ENV_NAMES or k.startswith(ENV_PREFIXES)}


def replace_single(argv, flag, value):
    if argv.count(flag) != 1:
        raise ValueError('expected exactly one ' + flag)
    result = list(argv)
    result[result.index(flag) + 1] = str(value)
    return result


def render(port, slots):
    from scripts.server import orchestrator_stack as s
    import subprocess
    checkout = Path.cwd().resolve()
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    if status:
        raise RuntimeError('orchestrator checkout has tracked changes')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    registry = checkout / 'orchestration/model_registry.yaml'
    role = s.RegistryLoader(registry_path=registry, validate_paths=False).get_role('architect_critic')
    # Use a fixed render port and permit exactly two isolation substitutions.
    raw = s.build_server_command(role, 8083, False, numa_instance=0, prepare_runtime_dirs=False)
    argv = replace_single(replace_single(raw, '--port', port), '--slot-save-path', slots)
    prefix = s._numa_prefix('architect_critic', 0)
    env = s.build_launch_env('architect_critic', os.environ.copy())
    binary, paths = s._stack_prior_runtime_overrides('architect_critic')
    s._apply_runtime_requirements_env(env, binary_override=binary, ld_paths=paths,
                                      preserve={**s.recipe_env_for('architect_critic'), **s._role_env_overrides('architect_critic')})
    s.apply_pinned_launch_env(env, prefix)
    if '--no-cache-idle-slots' not in argv:
        raise RuntimeError('expected compiled cache-idle behavior missing')
    files = [argv[0]] + [argv[argv.index(flag) + 1] for flag in ASSET_FLAGS]
    identities = {str(Path(f).resolve()): digest(f) for f in files}
    return {'checkout': str(checkout), 'head': head, 'registry_sha256': digest(registry), 'stack_priors_path': str(s.STACK_PRIORS_PATH.resolve()),
            'stack_priors_sha256': digest(s.STACK_PRIORS_PATH),
            'rendered_argv': raw, 'argv': argv, 'prefix': prefix,
            'effective_env': selected_env(env), 'identities': identities}, env


def proc_identity(pid, birth, plan):
    proc = Path('/proc') / str(pid)
    if sampler.start_ticks(proc) != birth:
        raise RuntimeError('process birth changed')
    actual = (proc / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    actual = [os.fsdecode(x) for x in actual]
    if actual != plan['argv']:
        raise RuntimeError('executed argv differs from rendered argv')
    observed_env = dict(os.fsdecode(x).split('=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
    if selected_env(observed_env) != plan['effective_env']:
        raise RuntimeError('effective environment differs from rendered environment')
    executable = str((proc / 'exe').resolve())
    if executable != str(Path(plan['argv'][0]).resolve()):
        raise RuntimeError('executed binary differs from rendered binary')
    maps = (proc / 'maps').read_text().splitlines()
    libs = sorted({line.split()[-1] for line in maps if line.split() and line.split()[-1].startswith('/')
                   and any(x in Path(line.split()[-1]).name for x in ('libggml', 'libllama', 'libmtmd'))})
    if not any('libggml-hip' in p for p in libs):
        raise RuntimeError('no mapped HIP backend')
    lib_ids = {}
    for path in libs:
        stat = os.stat(path)
        rows = [line.split() for line in maps if line.split() and line.split()[-1] == path]
        if not rows or any(int(row[4]) != stat.st_ino or row[3] != f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}' for row in rows):
            raise RuntimeError('mapped library identity changed: ' + path)
        lib_ids[path] = {'sha256': digest(path), 'device': stat.st_dev, 'inode': stat.st_ino}
    for path, sha in plan['identities'].items():
        if digest(path) != sha:
            raise RuntimeError('binary or model identity changed: ' + path)
    if sampler.start_ticks(proc) != birth:
        raise RuntimeError('process birth changed during identity capture')
    return {'exe': executable, 'argv': actual, 'effective_env': selected_env(observed_env), 'mapped_libraries_sha256': lib_ids}


def health(port):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
    try:
        conn.request('GET', '/health')
        response = conn.getresponse(); response.read()
        return response.status == 200
    except OSError:
        return False
    finally:
        conn.close()


def write_json(path, record):
    with Path(path).open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())


def fake_dry_run(port, dry_dir, result_path, pidfile):
    """Exercise the keeper entrypoint with an owned fake HTTP server, never a model."""
    dry_dir = Path(dry_dir)
    dry_dir.mkdir(parents=True, exist_ok=True)
    code = ("from http.server import BaseHTTPRequestHandler,HTTPServer; "
            "exec('class H(BaseHTTPRequestHandler):\\n def do_GET(self):\\n  self.send_response(200 if self.path == \"/health\" else 404); self.end_headers()'); "
            "HTTPServer(('127.0.0.1',int(__import__('sys').argv[1])),H).serve_forever()")
    server = server_lifecycle.Server([sys.executable, '-c', code, str(port)], dict(os.environ),
                                     dry_dir / 'fake-server.log', pidfile)
    cleanup = None; error = None
    try:
        server.start(); deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not server.alive(): raise RuntimeError('fake server exited')
            if health(port): break
            time.sleep(0.05)
        else: raise RuntimeError('fake health timeout')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        try: cleanup = server.close(term_grace=2, kill_grace=2)
        except BaseException as exc: error = error or f'cleanup: {exc}'
    passed = not error and bool(cleanup and cleanup.get('verified_dead'))
    write_json(result_path, {'schema': 'kvu16a.fake-dry-run.v1', 'fake': True, 'passed': passed,
                             'error': error, 'cleanup': cleanup,
                             'interpretation': 'entrypoint control only; no GPU, model, sample or belief'})
    return 0 if passed else 75


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--belief-adapter', required=True, type=Path, help='pinned epyc-root scripts/vidya/adapters/kvu16a_native.py')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--max-gap-seconds', required=True, type=float, help='predeclared maximum sampling gap')
    p.add_argument('--healthy-samples', required=True, type=int, help='predeclared number of post-health samples')
    p.add_argument('--run-plan-sha256', help='execute previously created plan with this SHA-256 under an existing GPU grant')
    args = p.parse_args()
    if not 1024 <= args.port <= 65535 or args.max_gap_seconds <= 0 or args.healthy_samples < 2:
        p.error('invalid port, maximum gap, or healthy sample count')
    if not args.run_plan_sha256:
        args.output_dir.mkdir(mode=0o700)
        (args.output_dir / 'slots').mkdir(mode=0o700)
    slots = args.output_dir.resolve() / 'slots'
    if not slots.is_dir() or any(slots.iterdir()):
        raise RuntimeError('isolated slot directory missing or not empty')
    adapter_path = args.belief_adapter.resolve(strict=True)
    if adapter_path.name != 'kvu16a_native.py': raise RuntimeError('expected KVU-16a belief adapter')
    spec = importlib.util.spec_from_file_location('kvu16a_native', adapter_path)
    belief_adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(belief_adapter)
    plan, env = render(args.port, slots)
    plan.update({'belief_adapter_sha256': digest(adapter_path), 'producer_path': str(Path(__file__).resolve()),
                 'producer_sha256': digest(Path(__file__)),
                 'producer_files_sha256': {name: digest(Path(__file__).resolve().parent / name) for name in
                                           ('producer.py', 'sampler.py', 'device_identity.py', 'server_lifecycle.py')}})
    plan.update({'max_gap_seconds': args.max_gap_seconds, 'healthy_samples': args.healthy_samples,
                 'meaning': 'sampled own-PID VRAM; observed maximum is not a measured peak'})
    plan_path = args.output_dir / 'plan.json'
    if not args.run_plan_sha256:
        write_json(plan_path, plan)
        return 0
    if digest(plan_path) != args.run_plan_sha256 or json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('predeclared plan hash or rendered launch changed')
    mode = os.environ.get('FILLER_DRYRUN')
    if mode not in ('0', '1'):
        raise RuntimeError('explicit FILLER_DRYRUN mode required')
    if mode == '1':
        return fake_dry_run(args.port, os.environ['FILLER_DRYRUN_DIR'],
                            os.environ['FILLER_DRYRUN_RESULT_PATH'], os.environ['FILLER_PIDFILE'])
    if os.environ.get('FILLER_NEEDGPU') != '1':
        raise RuntimeError('GPU grant missing')
    deadline = int(os.environ['FILLER_DEADLINE']) - 30
    stopfile = Path(os.environ['FILLER_STOPFILE'])
    device = device_identity.discover()
    server = server_lifecycle.Server(plan['prefix'] + plan['argv'], env, args.output_dir / 'server.log', os.environ['FILLER_PIDFILE'])
    state = {'phase': 'startup', 'healthy_at': None, 'started_at': time.monotonic(), 'error': None}
    monitor = None; cleanup = None; identity = None; stop = False
    def halt(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, halt); signal.signal(signal.SIGINT, halt)
    try:
        pid, birth = server.start()
        hip = str((Path(plan['argv'][0]).parent / 'libggml-hip.so.0').resolve())
        monitor = sampler.Sampler(pid, birth, hip, args.output_dir / 'samples.jsonl', startup=True,
                                  phase=lambda: state['phase'], device=device).start()
        while time.time() < deadline and not stop and not stopfile.exists():
            if not server.alive() or monitor.abort.is_set():
                raise RuntimeError(monitor.reason or 'server exited')
            last = monitor.last
            if last and last['hip_resident'] and (last['own_vram_b'] or 0) > 0 and health(args.port):
                if state['healthy_at'] is None:
                    identity = proc_identity(pid, birth, plan)
                    state['healthy_at'] = time.monotonic(); state['phase'] = 'healthy'
                lines = (args.output_dir / 'samples.jsonl').read_text().splitlines()
                healthy_samples = [json.loads(x) for x in lines if json.loads(x).get('phase') == 'healthy']
                if len(healthy_samples) >= args.healthy_samples:
                    end_identity = proc_identity(pid, birth, plan)
                    if identity != end_identity:
                        raise RuntimeError('runtime identity changed during healthy window')
                    break
            time.sleep(0.2)
        else:
            raise RuntimeError('deadline or stop before healthy window completed')
    except BaseException as exc:
        state['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        state['ended_at'] = time.monotonic()
        if monitor and not monitor.stop(): state['error'] = state['error'] or 'sampler did not stop'
        try: cleanup = server.close()
        except BaseException as exc: state['error'] = state['error'] or f'cleanup: {exc}'
    samples_path = args.output_dir / 'samples.jsonl'
    records = [json.loads(x) for x in samples_path.read_text().splitlines()] if samples_path.exists() else []
    observed = [x for x in records if x.get('status') == 'observed']
    begins = [x['sample_start_monotonic_s'] for x in observed] + [state['ended_at']]
    ends = [state['started_at']] + [x['sample_end_monotonic_s'] for x in observed]
    gaps = [max(0, b-a) for a,b in zip(ends, begins)]
    missing = [x for x in records if x.get('status') == 'missing']
    complete = not state['error'] and bool(cleanup and cleanup.get('verified_dead')) and not missing and bool(observed) and all(x['own_vram_b'] is not None for x in observed) and max(gaps) <= args.max_gap_seconds
    healthy_vram = [x['own_vram_b'] for x in observed if x['phase'] == 'healthy' and x['own_vram_b'] is not None]
    log = (args.output_dir / 'server.log').read_text(errors='replace') if (args.output_dir / 'server.log').exists() else ''
    kv_matches = re.findall(r'llama_kv_cache: size =\s*([0-9.]+) MiB \(393216 cells,\s*16 layers,\s*4/1 seqs\), K \(q8_0\)', log)
    kv_b = float(kv_matches[0]) * 1024**2 if len(kv_matches) == 1 else None
    if kv_b is None or not healthy_vram:
        complete = False
    grant = {k: os.environ.get(k) for k in ('FILLER_NEEDGPU', 'FILLER_DEADLINE', 'FILLER_STOPFILE', 'FILLER_PIDFILE')}
    result = {'schema': 'kvu16a.load.sampled.v1', 'grant': grant, 'state': state, 'device': device, 'runtime_identity': identity,
              'cleanup': cleanup, 'sample_count': len(observed), 'missing': missing, 'sampling_gaps_s': gaps,
              'coverage_qualified': complete, 'sampled_max_own_vram_b': max((x['own_vram_b'] for x in observed if x['own_vram_b'] is not None), default=None),
              'target_kv_b_from_log': kv_b,
              'healthy_sampled_max_non_kv_b': max(healthy_vram) - kv_b if healthy_vram and kv_b is not None else None,
              'interpretation': 'sampled maximum only, not a peak; no registry ratification'}
    write_json(args.output_dir / 'result.json', result)
    belief_adapter.record_run(args.output_dir)
    return 0 if complete else 75


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print('load producer refused:', exc, file=sys.stderr); sys.exit(75)
