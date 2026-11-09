#!/usr/bin/python3
"""AKX-P0d W2 GPU structural census on C9: runs prefill512 and decode1, one owned server each.

Run under resource_watch by job.py. A smoke runs both phases with a 45 s repeat of the
prefill request so the server holds VRAM long enough for the keeper's residency check.
It writes only its own smoke root and never the live RESULT.json.
"""
import hashlib, json, os, signal, sys, time
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path[:0] = [str(D), str(D / 'producer')]
P = json.loads((D / 'recipe.json').read_text())
from autokernel.loop import census  # noqa: E402
import gpu_probe  # noqa: E402
from gpu_probe import ServerProbe  # noqa: E402

STOP = [False]
for s in (signal.SIGTERM, signal.SIGINT):
    signal.signal(s, lambda *_: STOP.__setitem__(0, True))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def packet(root, phase):
    p = json.loads(json.dumps(P))
    n = 0
    for i, v in enumerate(p['argv']):
        if isinstance(v, str) and '__RUN_ROOT__' in v:
            p['argv'][i] = v.replace('__RUN_ROOT__', str(root))
            n += 1
    if n != 1:
        raise ValueError('expected exactly one __RUN_ROOT__ token')
    p['capture_inputs']['request_contract'] = str(D / f'requests-{phase}.json')
    return p


def run_phase(root, phase, burst_s):
    p = packet(root, phase)
    original = gpu_probe.timed_request
    if burst_s:
        def burst(fn, port, endpoint, body, timeout, windows):
            first = original(fn, port, endpoint, body, timeout, windows)
            start = time.monotonic()
            while time.monotonic() - start < burst_s and not STOP[0]:
                original(fn, port, endpoint, body, min(30, max(1, timeout)), windows)
            return first
        gpu_probe.timed_request = burst
    out = root / 'native' / phase
    out.mkdir(parents=True)
    server = ServerProbe(p, out)
    shape = census.Shape(phase, 512 if phase == 'prefill' else 1, 1)
    try:
        row = server(P['binary'], Path(P['model']), shape, expected_ops=P['expected_ops'],
                     require_device=True, timeout=P['timeout_s'])
        return row, [str(j) for j in server.journals], p
    finally:
        gpu_probe.timed_request = original


def main():
    smoke = os.environ.get('FILLER_SMOKE') == '1'
    root = Path(os.environ['W2_RUN_ROOT'])
    state = {'schema': 'inf75.w2.gpu.c9.work.v1', 'id': P['id'], 'scope': P['scope'], 'smoke': smoke,
             'first_request_ok': False, 'phases': [], 'rc': 1}
    rc = 1
    try:
        pins = json.loads((D / 'pins.json').read_text())
        rows = {}
        for phase in ('prefill', 'decode'):
            if STOP[0]:
                raise InterruptedError('STOP requested')
            row, journals, p = run_phase(root, phase, 45 if smoke and phase == 'prefill' else 0)
            rows[phase] = row
            state['phases'].append({'phase': phase, 'state': row['state'], 'rc': row['rc'], 'error': row['error'],
                                    'guards': row['native_shape_evidence'].get('guards'),
                                    'image_encoder': row['native_shape_evidence'].get('image_encoder'),
                                    'journals': journals})
            state['first_request_ok'] = True
            write(root / 'WORK-RESULT.json', state)
        observed = all(r['state'] == census.OBSERVED for r in rows.values())
        state['all_phases_observed'] = observed
        if not smoke:
            identity = {
                'workload_id': P['id'], 'workload_signature': P['scope'], 'phase': 'both',
                'kernel': {'commit': P['kernel_commit'], 'binary_sha256': pins['binary_sha256'],
                           'build_manifest_sha256': pins['build_manifest_sha256']},
                'model': {'path': P['model'], 'sha256': pins['model_sha256']},
                'projector': {'path': P['projector'], 'sha256': pins['projector_sha256']},
                'backend': 'ROCm0', 'claim_cpus': os.environ['FILLER_CPUS'],
                'task_affinity': sorted(os.sched_getaffinity(0)), 'argv': p['argv'], 'effective_argv': p['argv'],
                'env': p['env'], 'request_sha256': sha(P['request_files']['prefill']),
                'source_manifest_sha256': sha(D / 'SOURCES.sha256')}
            record = {'schema': census.SCHEMA, 'identity': identity,
                      'weights': census.workload_contract.read_census(Path(P['model'])).to_dict(),
                      'graph': {'shape_envelope': [{'phase': 'prefill', 'n_tokens': 512, 'n_seq': 1},
                                                   {'phase': 'decode', 'n_tokens': 1, 'n_seq': 1}],
                                'per_shape': [rows['prefill'], rows['decode']]},
                      'validity': {'freshness_key': census.freshness_key(identity)}}
            record['validity']['state'] = census.census_state(record)
            record['validity']['observed'] = record['validity']['state'] == census.OBSERVED
            state['native_census'] = record
            observed = observed and record['validity']['observed']
        rc = 0 if observed else 1
    except BaseException as exc:  # recorded verbatim, never converted to a pass
        state['error'] = f'{type(exc).__name__}: {exc}'
        rc = 75 if STOP[0] else 1
    state['rc'] = rc
    write(root / 'WORK-RESULT.json', state)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
