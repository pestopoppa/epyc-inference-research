#!/usr/bin/python3
"""Entry for the C9 W2 GPU census row: seal checks, grant contract, resource watch, result writing.

  job.py              granted run (or FILLER_SMOKE=1 smoke) under filler_run15
  job.py --preflight  model-free: seal, tracer ABI pins, imports and census_capture.begin on the
                      real capture_inputs with a synthetic grant; no server, no GPU.
"""
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

D = Path(__file__).resolve().parent
P = json.loads((D / 'recipe.json').read_text())
PIN = json.loads((D / 'pins.json').read_text())
CPUS = {184, 185}


def parse(text):
    out = set()
    for t in text.split(','):
        a, _, b = t.partition('-')
        out |= set(range(int(a), int(b or a) + 1))
    return out


def check(cond, msg):
    if not cond:
        raise ValueError(msg)


def verify_seal():
    subprocess.run(['/usr/bin/sha256sum', '--check', '--status', str(D / 'SOURCES.sha256')], check=True, timeout=120)
    # SHA256SUMS paths are relative to the store root (bin/...).
    subprocess.run(['/usr/bin/sha256sum', '--check', '--status', str(Path(P['build_dir']) / 'SHA256SUMS')],
                   cwd=P['build_dir'], check=True, timeout=120)
    for key, size in (('model', PIN['model_size']), ('projector', PIN['projector_size'])):
        check(Path(P[key]).stat().st_size == size, key + ' size changed')  # full digest verified at prepare time
    check(shutil.disk_usage('/mnt/raid0/llm').free >= 20 << 30, 'disk floor 20 GiB')


def grant_path():
    return Path(os.environ.get('FILLER_GRANTS_DIR', '/mnt/raid0/llm/tmp/compute-flags/grants')) / (P['id'] + '.grant')


def validate_entry():
    verify_seal()
    check(not (D / 'STOP').exists(), 'STOP present')
    check(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'ambient loader injection')
    check(set(os.sched_getaffinity(0)) == CPUS, 'task affinity must be exactly 184-185')
    check(parse(os.environ['FILLER_CPUS']) == CPUS, 'FILLER_CPUS must be exactly 184-185')
    check(os.environ.get('FILLER_ID') == P['id'], 'FILLER_ID mismatch')
    grant = grant_path()
    check(grant.is_file() and not grant.is_symlink() and grant.stat().st_size <= 4096, 'bounded regular grant required')
    check('mi210_0' in grant.read_text(), 'grant lacks mi210_0')
    check(int(os.environ['FILLER_DEADLINE']) - time.time() >= 120, 'insufficient deadline reserve')


def preflight():
    verify_seal()
    sys.path[:0] = [str(D), str(D / 'producer')]
    from autokernel.loop import census_capture
    import gpu_probe, work  # noqa: F401
    with tempfile.TemporaryDirectory(prefix='w2-preflight-', dir=D) as td:
        grant = Path(td) / 'grant'
        grant.write_text('regions=mi210_0,184-185 synthetic-preflight=1\n')
        for phase in ('prefill', 'decode'):
            p = work.packet(Path(td), phase)
            inputs = {**p['capture_inputs'], 'requests': p['request_files'][phase], 'grant': str(grant)}
            census_capture.begin(Path(td) / f'{phase}.jsonl', attempt_id='preflight-' + phase,
                                 started_ns=time.time_ns(), argv=p['argv'], env=p['env'], shape={'phase': phase},
                                 grant=grant.read_text(), deadline='0', kernel_commit=p['kernel_commit'], inputs=inputs)
    print('PREFLIGHT OK')
    return 0


def main():
    if sys.argv[1:] == ['--preflight']:
        return preflight()
    if sys.argv[1:]:
        return 64
    smoke = os.environ.get('FILLER_SMOKE') == '1'
    validate_entry()
    os.umask(0o077)
    started = time.time()
    if not smoke:
        for name in ('RESULT.json', 'LAST-RESULT.json'):
            (D / name).unlink(missing_ok=True)
    root = D / (('smoke-' if smoke else 'run-') + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'-{os.getpid()}')
    root.mkdir()
    env = dict(os.environ, W2_RUN_ROOT=str(root), WATCH_RESULT=str(root / 'watch-result.json'),
               FILLER_STOPFILE=str(D / 'STOP'), OFFICIAL_FILLER_GRANT=str(grant_path()))
    if smoke:
        env['FILLER_DEADLINE'] = str(min(int(env['FILLER_DEADLINE']), int(started + 270)))
    with (root / 'work.log').open('wb') as log:
        rc = subprocess.call(['/usr/bin/python3', '-B', str(D / 'resource_watch.py'), '--limit-kib', str(64 << 20),
                              '/usr/bin/python3', '-B', str(D / 'work.py')],
                             env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    watch = json.loads((root / 'watch-result.json').read_text()) if (root / 'watch-result.json').exists() else {}
    work_result = root / 'WORK-RESULT.json'
    result = json.loads(work_result.read_text()) if work_result.exists() else {'first_request_ok': False}
    if rc == 0 and result.get('rc') != 0:
        rc = 1
    result.update(rc=rc, partial=rc == 75 and bool(watch.get('partial')), cleanup_verified=watch.get('cleanup_verified'),
                  wall_s=time.time() - started, watch=watch, path=str(root))
    tmp = root / 'result.tmp'
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    tmp.replace(root / 'result.json')
    if smoke:
        # the smoke predicate reads this; it can never satisfy the live --complete-cmd
        tmp = root / 'SMOKE-RESULT.tmp'
        tmp.write_text(json.dumps({'all_phases_observed': result.get('all_phases_observed') is True and rc == 0,
                                   'cleanup_verified': watch.get('cleanup_verified') is True, 'path': str(root)}) + '\n')
        tmp.replace(root / 'SMOKE-RESULT.json')
    elif rc == 0 and watch.get('cleanup_verified') is True:
        for name in ('LAST-RESULT', 'RESULT'):
            tmp = D / (name + '.tmp')
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
            tmp.replace(D / (name + '.json'))
    print(json.dumps({'path': str(root), 'rc': rc}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
