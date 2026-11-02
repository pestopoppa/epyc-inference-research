"""NEW admitted fixed-family six-cell run; original UNVERIFIED output is untouched."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'driver')]
import kvu_ckpt_arms as K
import holdout_lifecycle as lifecycle
from affinity_guard import record, cpu_set
from corpus_guard import verify as verify_corpus
from hpr_no_redraw_guard import reserve as reserve_exposure

# A successor packet must include the exact ledger, guard module and one
# shared claims directory. The original packet and observations are untouched.
EXPOSURE_LEDGER_SHA256 = '2dd66aee9b3aefca1637c2ea341bc685bc3941d0bff8912ae09cbaecbcf18005'

DISCLAIMER = 'UNVERIFIED native contract; golden parity/control review and new predeclaration required'
CPU_SCOPE = 'keeper-assigned 24-primary-CPU quarter; runtime list sealed in grant evidence'

def raid_path(value):
    value = str(value)
    if not value.startswith('raid/'):
        raise ValueError('pinned path must be relative to canonical RAID root')
    return Path('/mnt/raid0/llm') / value[len('raid/'):]

def validate_cpu_scope(pins, cpu_list, actual):
    assigned = cpu_set(cpu_list)
    actual = set(actual)
    if pins.get('cpu_scope') != CPU_SCOPE:
        raise RuntimeError('CPU scope differs from the relocatable predeclaration')
    if (len(assigned) != 24 or max(assigned) >= 96 or min(assigned)//24 != max(assigned)//24
            or actual != assigned):
        raise RuntimeError('run requires the exact keeper-assigned 24-primary-CPU quarter')
    return assigned

def cell_problems(row, requests):
    problems = []
    if row.get('result') != 'OK':
        problems.append('cell not a complete admitted native capture')
    if row.get('unit') != '64k' or row.get('arm') not in ('dense', 'coarse') or row.get('wave') not in (0, 1, 2):
        problems.append('unexpected cell identity')
    checks = K.smoke_check(row, requests)
    problems.extend(checks)
    expected = ['cold_build'] + [x[0] for x in K.probe_plan(K.LS_ARM['64k'], 0)]
    if [r.get('probe') for r in requests] != expected:
        problems.append('incomplete or unexpected probe sequence')
    if any((r.get('prefix_capture') or {}).get('state') != 'complete' for r in requests):
        problems.append('incomplete native capture')
    mismatches = (row.get('sim_check') or {}).get('mismatches')
    if mismatches != []:
        problems.append('simulator mismatch or absent comparison: ' + repr(mismatches))
    if len((row.get('sim_check') or {}).get('per_probe') or {}) != len(requests):
        problems.append('simulator did not compare every probe')
    return problems

def witness_problems(windows, requests, cpus, libdir):
    problems, counts = [], []
    if len(windows) != requests:
        return ['active request witness count mismatch'], counts
    for window in windows:
        start, end = window.get('request_start'), window.get('request_end')
        if start is None or end is None or end < start:
            problems.append('missing request interval')
            counts.append(0)
            continue
        samples = [s for s in window.get('samples', []) if start <= s.get('monotonic', -1) <= end]
        counts.append(len(samples))
        for sample in samples:
            status = sample.get('status', '')
            matches = [line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('Cpus_allowed_list:')]
            libs = sample.get('libs', [])
            if sample.get('error') or len(matches) != 1 or not cpu_set(matches[0]) <= cpus:
                problems.append('missing or escaped CPU witness')
            if not libs or any(Path(path).resolve().parent != libdir for path in libs):
                problems.append('missing or foreign mapped library witness')
    return problems, counts

def observed_request(fn, sample_path, *args, **kwargs):
    server = lifecycle.ACTIVE_SERVER
    if server is None or server.proc is None:
        raise RuntimeError('no owned server for active request witness')
    pid, samples, done = server.proc.pid, [], threading.Event()
    def sample():
        while not done.is_set():
            try:
                status = Path(f'/proc/{pid}/status').read_text()
                maps = Path(f'/proc/{pid}/maps').read_text()
                samples.append(dict(monotonic=time.monotonic(), status=status,
                                    libs=sorted({line.split()[-1] for line in maps.splitlines()
                                                 if '/libggml' in line or '/libllama' in line})))
            except OSError as exc:
                samples.append(dict(monotonic=time.monotonic(), error=repr(exc)))
            done.wait(0.25)
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    request_start = time.monotonic()
    try:
        return fn(*args, **kwargs)
    finally:
        request_end = time.monotonic()
        done.set(); t.join()
        with sample_path.open('a') as f:
            f.write(json.dumps(dict(pid=pid, request_start=request_start, request_end=request_end, samples=samples), sort_keys=True) + '\n')

def full_window_admissible(deadline, now, required):
    return deadline - now >= required

def main():
    contract=K.HA.create_contract(K.HPR_NATIVE_CONTRACT)
    K.HPR_NATIVE_CONTRACT=contract
    K.HPR_NATIVE_CONTRACT_SHA256=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    pins = json.loads((HERE / 'instrument-pins.json').read_text())
    record(HERE / 'holdout-grant-affinity.json')
    validate_cpu_scope(pins, os.environ["FILLER_CPUS"], os.sched_getaffinity(0))
    if not full_window_admissible(int(os.environ["FILLER_DEADLINE"]), time.time(), pins["holdout_window_s"]):
        raise RuntimeError("grant deadline lacks declared full-run window")
    if K.M.MODEL != str(raid_path(pins["model"])) or K.M.TEMPLATE != str(raid_path(pins["template"])):
        raise RuntimeError("driver model/template differs from sealed pins")
    if hashlib.sha256((HERE / "selection.json").read_bytes()).hexdigest() != "84ea5daeff6fcf69932f98faa2a1fba7b2f0c99252bf3f1b8c59174ef29d994d":
        raise RuntimeError("selection differs from original predeclared selection")
    corpus_reference=json.loads((HERE / "CORPUS-REFERENCE.json").read_text())
    corpus_manifest = raid_path(corpus_reference["manifest_path"])
    corpus_root = raid_path(corpus_reference["path"])
    if hashlib.sha256(corpus_manifest.read_bytes()).hexdigest()!=corpus_reference["manifest_sha256"]:
        raise ValueError("immutable referenced corpus manifest changed")
    verify_corpus(corpus_root.parent)
    out = Path(os.environ['HPR_OUT'])
    out.mkdir(exist_ok=False)
    lifecycle.install(K)
    orig_request, orig_launch = K.hpr_completion_request, K.run_launch
    samples = out / 'active-request-samples.jsonl'
    witness_counts = []
    K.hpr_completion_request = lambda *a, **kw: observed_request(orig_request, samples, *a, **kw)
    def checked_launch(a, launch, todo, budget, calib, ctx, output):
        if len(todo) != 1:
            raise RuntimeError('no-redraw requires one cell per launch')
        cell = todo[0]
        reserve_exposure(Path(os.environ['HPR_EXPOSURE_LEDGER']), EXPOSURE_LEDGER_SHA256,
                         Path(os.environ['HPR_EXPOSURE_CLAIMS']),
                         corpus_sha256=corpus_reference['manifest_sha256'],
                         selection_sha256=hashlib.sha256((HERE / 'selection.json').read_bytes()).hexdigest(),
                         model=cell.model, unit=cell.unit, arm=cell.arm, wave=cell.wave,
                         attempt_id=output.name)
        record(out / ('grant-' + launch.id + '.json'))
        subprocess.run([str(raid_path(pins['ram_admit_script'])), str(pins['ram_gib'])], check=True)
        before = len(K.read_rows(output / K.CELLS_FILE))
        before_samples = len(K.read_rows(samples)) if samples.exists() else 0
        result = orig_launch(a, launch, todo, budget, calib, ctx, output)
        rows = K.read_rows(output / K.CELLS_FILE)[before:]
        if result != 'done' or len(todo) != 1 or len(rows) != 1:
            raise RuntimeError('cell incomplete, stopped, or blocked; preserved raw rows')
        reqs = [r for r in K.read_rows(output / K.REQS_FILE) if r.get('cell_key') == todo[0].key]
        problems = cell_problems(rows[0], reqs)
        windows = K.read_rows(samples)[before_samples:] if samples.exists() else []
        witness_errors, counts = witness_problems(windows, len(reqs), cpu_set(os.environ["FILLER_CPUS"]),
                                                  raid_path(pins["binary"]).resolve().parent)
        problems.extend(witness_errors)
        witness_counts.extend(counts)
        if any(n < 2 for n in counts):
            problems.append('fewer than2 during-request placement/library samples')
        if problems:
            raise RuntimeError('cell control refused: ' + '; '.join(problems))
        return result
    K.run_launch = checked_launch
    args = ['run', '--out', str(out), '--fork-binary', str(raid_path(pins['binary'])), '--models', 'q36_35b',
            '--cpu-list', os.environ['FILLER_CPUS'], '--threads', '24', '--port', str(pins['port']),
            '--stop-file', os.environ['FILLER_STOPFILE'], '--deadline-epoch', os.environ['FILLER_DEADLINE'],
            '--stages', 'holdout', '--lens-holdout', 'q36_35b=64k', '--arms', 'dense,coarse',
            '--corpus-roots', str(corpus_root / 'handoffs') + ',' + str(corpus_root / 'docs'),
            '--waves', '3', '--selection', str(HERE / 'selection.json'), '--hpr-native-prefix-capture']
    rc = K.main(args)
    rows = K.read_rows(out / K.CELLS_FILE)
    identities = {(r.get('arm'), r.get('wave')) for r in rows}
    if rc != 0 or len(rows) != 6 or identities != {(a, w) for a in ('dense', 'coarse') for w in range(3)}:
        raise RuntimeError('six-cell run incomplete')
    (out / 'NATIVE-HOLDOUT.json').write_text(json.dumps(dict(status='PROSPECTIVE_FIXED_FAMILY_COUNTS',
        cells=6, native_contract=K.HPR_NATIVE_CONTRACT, policy_verdict=None,
        request_sample_counts=witness_counts, requests_without_samples=sum(n == 0 for n in witness_counts),
        residency_coverage='INCOMPLETE' if any(n == 0 for n in witness_counts) else 'OBSERVED_ONLY',
        source_manifest_sha256=hashlib.sha256((HERE / 'source-pins.json').read_bytes()).hexdigest()), indent=2) + '\n')
    import policy_holdout
    verdict=policy_holdout.decide(rows)
    (out/'POLICY-VERDICT.json').write_text(json.dumps(verdict,indent=2)+'\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())
