#!/usr/bin/env python3
"""Owned A/A/B pre-sampling log-probability probe, to run inside filler_run12.

A changed map indicates an observable effect for this fixture only. It is not
an assertion about raw logits, utility, or other prompts and vectors.
"""
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error

from cvec_effect import compare
from cvec_metadata import verify
import cvec_descendants as owned
import cvec_process_identity as ident

SERVER = Path('/mnt/raid0/llm/kernels/production/cpu/llama-server')
LIBDIR = SERVER.parent
MODEL = Path('/mnt/raid0/llm/tmp/refactor-01a16fc3/cvec-hooks-ready/KEEP/model/Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf')
VECTOR = Path('/mnt/raid0/llm/models/qwen2.5-7b-seal-concise.gguf')
EXPECTED_MODEL_SHA256 = 'fafe6e14252a94d92dbf56408568f79bde5dd7f3d8e7bc0eb6275195c6214d31'
PROMPT = 'Solve for x: 4x - 7 = 21. Answer:'
FLAGS = Path('/mnt/raid0/llm/tmp/compute-flags')
REQUEST_TIMEOUT = 30
GRANT_BYTES = None
SIGNAL_HOLD = False
PENDING_SIGNAL = None


def interrupted(signum, _frame):
    global PENDING_SIGNAL
    if SIGNAL_HOLD:
        PENDING_SIGNAL = signum
    else:
        raise InterruptedError(f'signal {signum}')
REQUEST = {'prompt': PROMPT, 'n_predict': 1, 'n_probs': 64, 'temperature': -1,
           'cache_prompt': False, 'post_sampling_probs': False, 'stream': False}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
            check()
    return h.hexdigest()


def check():
    if Path(os.environ['FILLER_STOPFILE']).exists() or time.time() >= int(os.environ['FILLER_DEADLINE']):
        raise RuntimeError('STOP or deadline')
    if list(FLAGS.glob('HOLD_*')):
        raise RuntimeError('HOLD')
    grant = FLAGS / 'grants' / (os.environ['FILLER_ID'] + '.grant')
    if grant.read_bytes() != GRANT_BYTES:
        raise RuntimeError('grant changed or missing')


def start_watch():
    def watch():
        while True:
            time.sleep(0.2)
            try:
                check()
            except Exception:
                os.kill(os.getpid(), signal.SIGUSR1)
                return
    threading.Thread(target=watch, daemon=True).start()


def response(port, output, label, argv):
    check()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion',
        data=json.dumps(REQUEST).encode(), headers={'Content-Type':'application/json'})
    meta = {'label': label, 'status': None, 'outcome': 'INCOMPLETE', 'argv': argv,
            'request': REQUEST, 'bytes': 0}
    meta_path = output / f'{label}.http.json'
    body_path = output / f'{label}.body'
    def save():
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    save()
    try:
        with body_path.open('wb', buffering=0) as body:
            try:
                r = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
            except urllib.error.HTTPError as e:
                r = e
            with r:
                meta['status'] = r.status
                meta['headers'] = dict(r.headers.items())
                save()
                while True:
                    chunk = r.read1(4096)
                    if not chunk: break
                    body.write(chunk)
                    meta['bytes'] += len(chunk)
                    save()
            length = meta['headers'].get('Content-Length')
            if length is not None and meta['bytes'] != int(length):
                raise ValueError('truncated HTTP body')
            if meta['status'] != 200:
                raise ValueError(f'HTTP status {meta["status"]}')
            check()
            data = json.loads(body_path.read_bytes())
            meta['outcome'] = 'COMPLETE'
            save()
            return data
    except BaseException as exc:
        meta['outcome'] = 'INCONCLUSIVE'
        meta['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


def verify_mappings(proc, output, vector):
    mappings = set()
    for line in (Path('/proc') / str(proc.pid) / 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        name = Path(fields[5]).name
        relevant = name.startswith(('libggml', 'libllama', 'libcommon')) or Path(fields[5]).parent.resolve() == LIBDIR.resolve()
        if relevant and '.so' in name:
            if fields[5].endswith(' (deleted)'):
                raise RuntimeError('deleted native library mapping')
            path = Path(fields[5]).resolve()
            if path.parent != LIBDIR.resolve():
                raise RuntimeError('in-tree native library resolved outside pinned store')
            mappings.add(path)
    if not any(p.name.startswith('libggml') for p in mappings) or not any(p.name.startswith('libllama') for p in mappings):
        raise RuntimeError('required native library mapping absent')
    (output / ('vector.maps.json' if vector else 'baseline.maps.json')).write_text(json.dumps({str(p): digest(p) for p in sorted(mappings)}, indent=2) + '\n')

def run_arm(port, vector, output):
    check()
    args = [str(SERVER), '-m', str(MODEL), '--host', '127.0.0.1', '--port', str(port),
            '-t', os.environ['FILLER_THREADS'], '-c', '2048', '-ngl', '0', '--no-warmup', '-np', '1', '--no-kv-unified']
    if vector:
        args += ['--control-vector-scaled', f'{VECTOR}:0.5']
    env = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': str(LIBDIR), 'OMP_NUM_THREADS': '1'}
    if owned.children():
        raise RuntimeError('unexpected pre-existing children')
    with (output / ('vector.log' if vector else 'baseline.log')).open('w') as log:
        global SIGNAL_HOLD, PENDING_SIGNAL
        SIGNAL_HOLD = True
        try:
            proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
        except BaseException:
            SIGNAL_HOLD = False
            raise
        birth = None
        try:
            birth = ident.identity(proc.pid)
            if birth is None or birth[0] != os.getuid() or ident.parent(proc.pid) != os.getpid():
                raise RuntimeError('cannot capture owned server identity')
            (output / ('vector.identity.json' if vector else 'baseline.identity.json')).write_text(json.dumps({'pid': proc.pid, 'uid': birth[0], 'birth': birth[1], 'argv': args}) + '\n')
            with open(os.environ['FILLER_PIDFILE'], 'a') as pids:
                print(proc.pid, file=pids)
            SIGNAL_HOLD = False
            if PENDING_SIGNAL is not None:
                raise InterruptedError(f'signal {PENDING_SIGNAL}')
            while True:
                check()
                if proc.poll() is not None:
                    raise RuntimeError('owned server exited')
                ss = subprocess.run(['ss', '-ltnp', f'sport = :{port}'], capture_output=True, text=True, check=True)
                if f'pid={proc.pid},' in ss.stdout:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                            if r.status == 200:
                                break
                    except OSError:
                        pass
                time.sleep(1)
            verify_mappings(proc, output, vector)
            labels = ['B'] if vector else ['A1', 'A2']
            return [response(port, output, label, args) for label in labels]
        finally:
            SIGNAL_HOLD = True
            if proc.poll() is None:
                if birth is None:
                    proc.kill()
                else:
                    ident.stop(proc.pid, birth, grace=2)
            proc.wait(timeout=5)
            owned.drain()
            if birth is not None and ident.identity(proc.pid) == birth:
                raise RuntimeError('owned server cleanup failed')
            SIGNAL_HOLD = False
            if PENDING_SIGNAL is not None:
                raise InterruptedError(f'signal {PENDING_SIGNAL}')



def main():
    for name in ('FILLER_ID', 'FILLER_STOPFILE', 'FILLER_DEADLINE', 'FILLER_PIDFILE', 'FILLER_THREADS', 'FILLER_PORTS', 'CVEC_GRANT_SHA256'):
        if not os.environ.get(name):
            raise RuntimeError(f'missing {name}; must run inside an admitted filler')
    global GRANT_BYTES
    grant = FLAGS / 'grants' / (os.environ['FILLER_ID'] + '.grant')
    GRANT_BYTES = grant.read_bytes()
    if hashlib.sha256(GRANT_BYTES).hexdigest() != os.environ['CVEC_GRANT_SHA256']:
        raise RuntimeError('grant changed before probe start')
    if owned.children():
        raise RuntimeError("pre-existing children")
    owned.subreaper()
    start_watch()
    port = int(os.environ['FILLER_PORTS'].split(',')[0])
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps({'result': 'INCONCLUSIVE', 'denominator': 3, 'cells': {name: 'UNKNOWN' for name in ('A1', 'A2', 'B')}}, indent=2) + '\n')
    predeclare = {'schema': 'cvec-hook-fixed-fixture-v1',
        'declared_at_utc': datetime.now(timezone.utc).isoformat(),
        'decided_proposition': 'A1 and A2 token-ID to reported pre-sampling log-probability maps are exactly equal and B differs from A1',
        'rule': {'unstable_or_malformed': 'INCONCLUSIVE', 'equal_B': 'NO_EFFECT_OBSERVED', 'different_B': 'hook_effect_observed_on_fixed_fixture'},
        'cells': ['A1', 'A2', 'B'], 'denominator': 3, 'request': REQUEST,
        'model': str(MODEL), 'vector': str(VECTOR), 'vector_scale': 0.5,
        'server': str(SERVER), 'protocol_id': '', 'cotenant': 'UNKNOWN',
        'kv_mode': 'non-unified', 'server_environment': {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': str(LIBDIR), 'OMP_NUM_THREADS': '1'},
        'grant_path': str(grant), 'grant_bytes_utf8': GRANT_BYTES.decode('utf-8'),
        'grant_sha256': os.environ['CVEC_GRANT_SHA256'], 'cpus': os.environ.get('FILLER_CPUS', 'UNKNOWN'),
        'threads': os.environ['FILLER_THREADS'], 'deadline': os.environ['FILLER_DEADLINE']}
    (output / 'predeclare.json').write_text(json.dumps(predeclare, indent=2) + '\n')
    check()
    compatibility = verify(MODEL, VECTOR)
    env = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': str(LIBDIR), 'OMP_NUM_THREADS': '1'}
    version = subprocess.run([str(SERVER), '--version'], env=env, capture_output=True, text=True, check=True).stdout
    if '10303' not in version:
        raise RuntimeError('unexpected production kernel version')
    linkage = subprocess.run([str(Path(__file__).parents[1] / 'utils/verify_ggml_linkage.sh'), str(SERVER), str(LIBDIR.resolve())], env=env, capture_output=True, text=True, check=True)
    (output / 'linkage.txt').write_text(linkage.stdout + linkage.stderr)
    paths = [SERVER, MODEL, VECTOR, Path(__file__), Path(__file__).with_name('cvec_effect.py'), Path(__file__).with_name('cvec_metadata.py'), Path(__file__).with_name('cvec_descendants.py'), Path(__file__).with_name('cvec_process_identity.py'), Path(__file__).parents[1] / 'utils/verify_ggml_linkage.sh']
    paths += sorted(p for p in LIBDIR.glob('*.so*') if p.is_file() and not p.is_symlink())
    hashes = {str(p): digest(p) for p in paths}
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise RuntimeError('model digest does not match pinned acquisition')
    manifest = {'files': hashes, 'request': REQUEST, 'vector_scale': 0.5,
                'library_directory': str(LIBDIR.resolve()), 'server_version': version, 'port': port, 'metadata_compatibility': compatibility}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    cells = {name: 'UNKNOWN' for name in ('A1', 'A2', 'B')}
    result = {'result': 'INCONCLUSIVE', 'denominator': 3, 'cells': cells}
    def save_result():
        for name in cells:
            meta_path = output / f'{name}.http.json'
            if meta_path.exists():
                cells[name] = json.loads(meta_path.read_text()).get('outcome', 'UNKNOWN')
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    save_result()
    try:
        a = run_arm(port, False, output)
        b = run_arm(port, True, output)
        result.update(compare(*a, *b))
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_result()
    if result['result'] != 'hook_effect_observed_on_fixed_fixture':
        print('NO_EFFECT_OBSERVED', file=sys.stderr)
        sys.exit(2)



if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except Exception as e:
        print(f'INCONCLUSIVE: {e}', file=sys.stderr)
        sys.exit(1)
