"""Finite native server probe injected into the existing census producer."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import resource
import subprocess
import time
import urllib.request
import urllib.error

from autokernel.loop import census, census_capture, residency
from autokernel.loop.serving_server_owner import ServerPortOwner, require_free_port
from .server_census_evidence import graph_evidence, shape_evidence, tensor_shape_observed
from .server_census_affinity import within


def request(port, endpoint, body=None, timeout=2):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/{endpoint}', data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read((16 << 20) + 1)
            if len(raw) > 16 << 20:
                raise ValueError('HTTP response exceeds bound')
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        exc.close()
        raise


def own_sample(pid, *, gpu, phase, port):
    root = Path('/proc') / str(pid)
    all_maps = (root / 'maps').read_text().splitlines()
    maps = sorted({line.split()[-1] for line in all_maps
                   if '/libggml' in line and line.split()[-1].startswith('/')})
    instrument = sorted({line.split()[-1] for line in all_maps
                          if '/libcensus-native.so' in line and line.split()[-1].startswith('/')})
    affinity = {}
    for task in (root / 'task').iterdir():
        try:
            affinity[task.name] = next(line.split(':', 1)[1].strip()
                for line in (task / 'status').read_text().splitlines()
                if line.startswith('Cpus_allowed_list:'))
        except FileNotFoundError:
            continue
    slots = []
    if phase == 'requests':
        slots = request(port, 'slots', timeout=0.5)
    return {'utc_ns': time.time_ns(), 'pid': pid, 'phase': phase,
            'loaded_ggml': maps, 'loaded_instrument':instrument, 'task_affinity': affinity, 'slots': slots,
            'own_vram_bytes': residency.kfd_pid_vram(pid) if gpu else None,
            'global_vram_bytes': residency.vram_bytes() if gpu else None,
            'kfd_processes': residency.kfd_processes() if gpu else None}


class ServerProbe:
    """One owned, bounded server lifecycle per requested shape; no default recipes."""
    def __init__(self, packet, out):
        self.packet, self.out, self.journals = packet, Path(out), []

    def __call__(self, binary, model, shape, *, expected_ops=(), require_device=False, **unused):
        p = self.packet
        tag = f'{p["id"]}-{shape.phase}'
        log_path, journal = self.out / f'{tag}.log', self.out / f'{tag}.capture.jsonl'
        argv, env = list(p['argv']), dict(p['env'])
        env.update(GGML_SCHED_DEBUG='2', LLAMA_TRACE='1', FILLER_CPUS=os.environ['FILLER_CPUS'])
        if str(binary) not in argv or str(model) != p['model']:
            raise ValueError('probe binary/model differs from frozen packet')
        requests = json.loads(Path(p['request_files'][shape.phase]).read_text())
        if len(requests) != shape.n_seq:
            raise ValueError('request width differs from shape')
        grant_path = Path(os.environ.get('OFFICIAL_FILLER_GRANT',
            '/mnt/raid0/llm/tmp/compute-flags/grants/'+os.environ['FILLER_ID']+'.grant'))
        inputs = {**p['capture_inputs'], 'requests': p['request_files'][shape.phase]}
        inputs['grant'] = str(grant_path)
        census_capture.begin(journal, attempt_id=tag, started_ns=time.time_ns(), argv=argv,
            env=env, shape=shape.to_dict(), grant=grant_path.read_text(),
            deadline=os.environ['FILLER_DEADLINE'], kernel_commit=p['kernel_commit'], inputs=inputs,
            large_refs=tuple(p.get('large_refs', ())))
        self.journals.append(journal)
        deadline = min(time.time() + p['timeout_s'], int(os.environ['FILLER_DEADLINE']) - 10)
        proc, spawned_ns, error, responses, samples, loaded_before = None, None, '', [], [], {}
        offset, pool, owner, foreign = None, None, None, None
        def sample(phase):
            if phase != 'load':
                owner.require_listener()
            row = own_sample(proc.pid, gpu=require_device, phase=phase, port=p['port'])
            samples.append(row)  # Preserve the exact offending sample on every refusal.
            for path in row['loaded_ggml']:
                if not Path(path).resolve().is_relative_to(Path(p['build_dir']).resolve() / 'bin'):
                    raise ValueError(f'foreign loaded ggml: {path}')
                loaded_before.setdefault(path, census_capture.file_identity(Path(path)))
            for path in row['loaded_instrument']:
                if Path(path).resolve() != Path(p['instrument_library']).resolve():
                    raise ValueError('unexpected native trace instrument mapping')
                loaded_before.setdefault(path, census_capture.file_identity(Path(path)))
            if not within(row['task_affinity'],os.environ['FILLER_CPUS']):
                raise ValueError('server task mask outside claim: claimed='+os.environ['FILLER_CPUS']+
                                 ' actual='+repr(row['task_affinity']))
        def alive():
            if time.time() >= deadline or proc.poll() is not None:
                raise ValueError('server exited or bounded attempt deadline reached')
        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        with log_path.open('w+b') as log:
            try:
                if time.time() + 30 >= deadline:
                    raise ValueError('insufficient window after prelaunch capture')
                # Refuse an existing listener; never send traffic to a foreign process.
                require_free_port(p['port'])
                spawned_ns = time.time_ns()
                proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, close_fds=True, preexec_fn=limits)
                with Path(os.environ['FILLER_PIDFILE']).open('a') as pidfile:
                    pidfile.write(str(proc.pid)+'\n')
                owner = ServerPortOwner(proc, p['port'])
                # Same owned server tree is excluded by the canonical monitor.
                foreign = subprocess.Popen(['/usr/bin/python3', '-B', p['foreign_load'],
                    '--label', tag, '--server-pid', str(proc.pid), '--interval', '1',
                    '--bench-cpus', p['physical_cpus'], '--out',
                    str(self.out / f'{tag}.foreign-load.jsonl')], env={
                        'PATH':'/usr/bin:/bin', 'LANG':'C.UTF-8'},
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log)
                while True:
                    alive()
                    sample('load')
                    if not owner.listener_ready():
                        time.sleep(0.1)
                        continue
                    try:
                        request(p['port'], 'health')
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.1)
                offset = log.seek(0, 2)
                pool = ThreadPoolExecutor(shape.n_seq)
                futures = [pool.submit(request, p['port'], 'completion', body,
                                       max(1, deadline - time.time())) for body in requests]
                try:
                    while not all(future.done() for future in futures):
                        alive()
                        sample('requests')
                        time.sleep(0.05)
                    responses = [future.result() for future in futures]
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
                sample('complete')
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
            finally:
                if foreign is not None:
                    if foreign.poll() is not None:
                        error = error or 'foreign-load observer exited before native server'
                    else:
                        foreign.terminate()
                        try:
                            foreign.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            foreign.kill()
                            foreign.wait(timeout=5)
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                rc = None if proc is None else proc.returncode
                dsos = {path: path for row in samples for path in row['loaded_ggml']+row['loaded_instrument']}
                census_capture.finish(journal, attempt_id=tag, ended_ns=time.time_ns(), rc=rc,
                    error=error, samples=samples, log=str(log_path), loaded_dsos=dsos,
                    spawned_ns=spawned_ns, loaded_before=loaded_before)
            log.seek(0)
            raw = log.read().decode(errors='replace')
        # Offset is in bytes; decode each byte slice independently before parsing.
        blob = log_path.read_bytes()
        startup = blob[:offset].decode(errors='replace') if offset is not None else raw
        phase_log = blob[offset:].decode(errors='replace') if offset is not None else ''
        graph = graph_evidence(raw, phase_log, str(model))
        graph['device_seen'] = any((row.get('own_vram_bytes') or 0) > (1 << 30) for row in samples)
        evidence = shape_evidence(shape, startup, phase_log, requests, responses, samples)
        evidence['guards']['native_tensor_width_and_api_sequences'] = tensor_shape_observed(shape, graph)
        evidence['guards']['native_instrument_mapping'] = any(row['loaded_instrument'] for row in samples)
        evidence['observed'] = all(evidence['guards'].values())
        (self.out / f'{tag}.requests.json').write_text(json.dumps({
            'requests': requests, 'responses': responses, 'samples': samples,
            'phase_log_byte_offset': offset, 'native_shape_evidence': evidence}, sort_keys=True))
        device = not require_device or sum(row.get('phase') == 'requests'
            and (row.get('own_vram_bytes') or 0) > (1 << 30)
            and (row.get('global_vram_bytes') or 0) > (1 << 30)
            and (row.get('kfd_processes') or 0) > 0 for row in samples) >= 2
        observed = (not error and rc == 0 and evidence['observed'] and device
                    and graph['nodes_total'] > census.MIN_GRAPH_NODES
                    and set(expected_ops) <= graph['op_backend'].keys())
        foreign_path = self.out / f'{tag}.foreign-load.jsonl'
        foreign_rows = foreign_path.read_text().splitlines() if foreign_path.exists() else []
        if len(foreign_rows) < 2:
            error = error or 'foreign-load observer lacks a during-run sample'
            observed = False
        return {**graph, 'shape': shape.to_dict(), 'argv': argv, 'rc': rc, 'error': error,
            'state': census.OBSERVED if observed else census.UNKNOWN, 'instrument': 'server',
            'native_shape_evidence': evidence, 'phase_log_byte_offset': offset,
            'request_log': str(self.out / f'{tag}.requests.json'),
            'request_log_sha256': census_capture.file_sha256(self.out / f'{tag}.requests.json'),
            'foreign_load':str(foreign_path),
            'foreign_load_sha256':census_capture.file_sha256(foreign_path) if foreign_path.exists() else None}
