"""Prospective first-turn lesson verification; no library retrieval or model launch policy."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .controller.loop_experiment_runner import _run_process
from .lesson_admission import _evaluate_and_record, _number, violations


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, data):
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _source_proof(source_repo, commit, code_artifacts):
    repo = Path(source_repo)
    if not repo.is_absolute() or not repo.is_dir():
        raise ValueError('source_repo must be an absolute existing directory')
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True,
                              capture_output=True, text=True).stdout
    head = git('rev-parse', '--verify', 'HEAD').strip()
    status = git('status', '--porcelain=v1', '--untracked-files=no')
    artifacts = {}
    for name, expected in code_artifacts.items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError('evaluator code path must be an absolute regular file')
        actual = _sha(path.read_bytes())
        if actual != expected:
            raise ValueError('evaluator code digest mismatch')
        artifacts[name] = actual
    if head != commit or status:
        raise ValueError(f'source mismatch: observed_head={head}, tracked_clean={not bool(status)}')
    return {'source_repo': str(repo), 'head': head, 'tracked_clean': True,
            'code_artifacts': artifacts}


def _capture(root, name, argv, prompt, timeout, env, source_repo, commit, code_artifacts):
    """Capture the existing process executor and the exact first-turn input."""
    folder = root / name
    folder.mkdir()
    prompt_bytes = prompt.encode('utf-8')
    _write(folder / 'prompt.txt', prompt_bytes)
    exe = Path(argv[0])
    exe_sha = _sha(exe.read_bytes())
    try:
        source_before = _source_proof(source_repo, commit, code_artifacts)
    except Exception as exc:
        _write(folder / 'source-failure.txt', str(exc).encode())
        raise
    result = _run_process(argv, folder, env, prompt, timeout, None)
    _write(folder / 'stdout.txt', result.stdout.encode('utf-8'))
    _write(folder / 'stderr.txt', result.stderr.encode('utf-8'))
    try:
        source_after = _source_proof(source_repo, commit, code_artifacts)
    except Exception as exc:
        _write(folder / 'source-failure.txt', str(exc).encode())
        raise
    if exe_sha != _sha(exe.read_bytes()):
        raise ValueError('executable changed during execution')
    meta = {'argv': list(result.argv), 'executable_sha256': exe_sha,
            'prompt_sha256': _sha(prompt_bytes),
            'stdout_sha256': _sha(result.stdout.encode('utf-8')),
            'stderr_sha256': _sha(result.stderr.encode('utf-8')),
            'started_at': result.started_at, 'finished_at': result.finished_at,
            'elapsed_wall_seconds': result.elapsed_wall_seconds,
            'source_before': source_before, 'source_after': source_after,
            'returncode': result.returncode, 'timed_out': result.timed_out}
    _write(folder / 'capture.json', json.dumps(meta, sort_keys=True).encode())
    if result.returncode != 0 or result.timed_out:
        raise ValueError('native first-turn execution failed')
    return folder


def _read_capture(folder, expected_prompt, expected_argv, expected_exe_sha, protocol_id, source_repo, commit, code_artifacts):
    meta = json.loads((folder / 'capture.json').read_text())
    prompt = (folder / 'prompt.txt').read_bytes()
    stdout = (folder / 'stdout.txt').read_bytes()
    stderr = (folder / 'stderr.txt').read_bytes()
    if (meta['argv'] != list(expected_argv) or meta['executable_sha256'] != expected_exe_sha
            or meta['returncode'] != 0 or meta['timed_out'] is not False
            or meta['source_before'] != {'source_repo': source_repo, 'head': commit, 'tracked_clean': True, 'code_artifacts': code_artifacts}
            or meta['source_after'] != meta['source_before']
            or prompt != expected_prompt.encode('utf-8')
            or meta['prompt_sha256'] != _sha(prompt) or meta['stdout_sha256'] != _sha(stdout)
            or meta['stderr_sha256'] != _sha(stderr)):
        raise ValueError('native capture binding mismatch')
    result = json.loads(stdout)
    if (not isinstance(result, dict)
            or set(result) != {'baseline_ns', 'candidate_ns', 'correctness', 'protocol_id'}
            or result['protocol_id'] != protocol_id
            or result['correctness'] is not True
            or not _number(result['baseline_ns']) or not _number(result['candidate_ns'])):
        raise ValueError('native output must contain positive raw durations and correctness=true')
    speedup = result['baseline_ns'] / result['candidate_ns']
    if not _number(speedup):
        raise ValueError('native ratio is non-finite or non-positive')
    return speedup


def _cap_source_valid(provenance, task_id):
    if not isinstance(task_id, str) or not isinstance(provenance, dict) or set(provenance) != {'kind', 'source'}:
        return False
    kind, source = provenance['kind'], provenance['source']
    if not isinstance(source, str) or not source.strip():
        return False
    # A workload_roofline label and a source string are only assertions.
    # Until there is an evidence verifier and a selected workload they cannot
    # authorize an operational cap. Controls are explicitly isolated.
    return kind == 'control' and task_id.startswith('CONTROL-')


def produce(journal, *, artifact_root, task_id, task_prompt, candidate_id, commit,
            lesson, reopen_when, upper_cap, argv, timeout, evaluator_sha256, protocol_id, source_repo=None, code_artifacts=None, environment=None, cap_provenance=None):
    """Execute baseline and injected same-task first turns and append the receipt.

    argv names the native task execution entry. Its stdout contract is exactly
    {"baseline_ns": number, "candidate_ns": number, "correctness": true,
     "protocol_id": selected protocol}.
    The caller must bind a trustworthy native evaluator; this adapter derives
    the ratio from its captured raw timings and does not validate its internals.
    Caller/scheduler owns resource admission.
    No subprocess is started if cap is absent. The task and lesson are captured
    before execution; no earlier, unbound result is promoted into this receipt.
    """
    if not isinstance(task_id, str) or not _cap_source_valid(cap_provenance, task_id):
        upper_cap = None
        cap_provenance = None
    preliminary = {'schema': 'autokernel.lesson_admission.v1', 'task_id': task_id,
        'candidate_id': candidate_id, 'commit': commit, 'lesson': lesson,
        'reopen_when': reopen_when, 'r1': None, 'upper_cap': upper_cap,
        'verification': None, 'attempt_ref': None, 'decision': 'UNKNOWN',
        'reason': 'cap_missing' if upper_cap is None else 'verification_unavailable'}
    issues = violations(preliminary)
    if issues:
        raise ValueError('; '.join(issues))
    if upper_cap is None:
        return _evaluate_and_record(journal, task_id=task_id, candidate_id=candidate_id,
            commit=commit, lesson=lesson, reopen_when=reopen_when, r1=None,
            upper_cap=None, rerun=lambda **_: None)
    if not _number(upper_cap):
        raise ValueError('explicit finite positive upper cap required')
    if not task_prompt or not lesson or not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError('task prompt, lesson, and execution argv are required')
    if not Path(argv[0]).is_absolute():
        raise ValueError('execution entry must be an absolute path')
    if not isinstance(source_repo, str) or not isinstance(code_artifacts, dict):
        raise TypeError('source_repo and evaluator code artifact digest mapping are required')
    executable_name = Path(argv[0]).name.lower()
    interpreters = ('python', 'pypy', 'bash', 'sh', 'node', 'perl', 'ruby')
    if executable_name in {'env', 'timeout', 'stdbuf', 'setsid'}:
        raise ValueError('ambiguous execution wrapper is not supported')
    if executable_name.startswith(interpreters) and (len(argv) < 2 or argv[1] not in code_artifacts):
        raise ValueError('interpreter commands require an explicitly pinned entry script as argv[1]')
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ValueError('artifact root must be absolute')
    root.mkdir(parents=False, exist_ok=False)
    baseline_prompt = task_prompt
    injected_prompt = 'LESSON (injected before the first turn):\n' + lesson + '\n\nTASK:\n' + task_prompt
    env = dict(os.environ if environment is None else environment)
    executable_sha = _sha(Path(argv[0]).read_bytes())
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError('non-empty protocol is required')
    manifest = {'task_id': task_id, 'task_prompt': task_prompt, 'candidate_id': candidate_id,
                'commit': commit, 'lesson': lesson, 'reopen_when': reopen_when,
                'upper_cap': upper_cap, 'cap_provenance': cap_provenance, 'argv': list(argv), 'executable_sha256': executable_sha, 'expected_evaluator_sha256': evaluator_sha256, 'protocol_id': protocol_id,
                'environment_sha256': _sha(json.dumps(env, sort_keys=True).encode()),
                'source_repo': source_repo, 'code_artifacts': code_artifacts}
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    _write(root / 'manifest.json', manifest_bytes)
    def seal_attempt():
        entries = {}
        for part in ('manifest.json', 'baseline/prompt.txt', 'baseline/stdout.txt',
                     'baseline/stderr.txt', 'baseline/capture.json', 'baseline/source-failure.txt', 'verification/prompt.txt',
                     'verification/stdout.txt', 'verification/stderr.txt', 'verification/capture.json', 'verification/source-failure.txt'):
            path = root / part
            if path.is_file() and not path.is_symlink():
                entries[part] = _sha(path.read_bytes())
        raw = json.dumps(entries, sort_keys=True).encode()
        target = root / 'attempt.json'
        _write(target, raw)
        return f'{target}#{_sha(raw)}'
    try:
        if executable_sha != evaluator_sha256:
            raise ValueError('pinned evaluator SHA mismatch')
        baseline = _capture(root, 'baseline', argv, baseline_prompt, timeout, env, source_repo, commit, code_artifacts)
        r1 = _read_capture(baseline, baseline_prompt, argv, executable_sha, protocol_id, source_repo, commit, code_artifacts)
    except Exception:  # noqa: BLE001 - a failed native execution must remain UNKNOWN
        return _evaluate_and_record(journal, task_id=task_id, candidate_id=candidate_id,
            commit=commit, lesson=lesson, reopen_when=reopen_when, r1=None,
            upper_cap=upper_cap, rerun=lambda **_: None, attempt_ref=seal_attempt)
    def rerun(**kw):
        if kw != {'task_id': task_id, 'commit': commit, 'lesson': lesson}:
            raise ValueError('requested binding differs')
        verify = _capture(root, 'verification', argv, injected_prompt, timeout, env, source_repo, commit, code_artifacts)
        result = _read_capture(verify, injected_prompt, argv, executable_sha, protocol_id, source_repo, commit, code_artifacts)
        if _read_capture(baseline, baseline_prompt, argv, executable_sha, protocol_id, source_repo, commit, code_artifacts) != r1:
            raise ValueError('baseline changed during verification')
        if (root / 'manifest.json').read_bytes() != manifest_bytes:
            raise ValueError('manifest changed during verification')
        baseline_hash = _sha((baseline / 'capture.json').read_bytes())
        verify_hash = _sha((verify / 'capture.json').read_bytes())
        return {'task_id': task_id, 'commit': commit, 'r_verify': result,
                'run_ref': f'{root}#manifest={_sha(manifest_bytes)};baseline={baseline_hash};verification={verify_hash}'}
    return _evaluate_and_record(journal, task_id=task_id, candidate_id=candidate_id,
        commit=commit, lesson=lesson, reopen_when=reopen_when, r1=r1,
        upper_cap=upper_cap, rerun=rerun, attempt_ref=seal_attempt)


def verify_artifacts(payload):
    """Reopen bound native bytes and rederive the recorded first-turn values."""
    verification = payload['verification']
    if not isinstance(verification, dict):
        raise TypeError('no native verification to check')
    path, fragment = verification['run_ref'].rsplit('#', 1)
    pairs = dict(part.split('=', 1) for part in fragment.split(';'))
    if set(pairs) != {'manifest', 'baseline', 'verification'}:
        raise ValueError('invalid native reference')
    root = Path(path)
    manifest_raw = (root / 'manifest.json').read_bytes()
    baseline_raw = (root / 'baseline/capture.json').read_bytes()
    verify_raw = (root / 'verification/capture.json').read_bytes()
    if any(_sha(data) != pairs[key] for key, data in (
            ('manifest', manifest_raw), ('baseline', baseline_raw), ('verification', verify_raw))):
        raise ValueError('native capture digest mismatch')
    m = json.loads(manifest_raw)
    if not _cap_source_valid(m.get('cap_provenance'), m.get('task_id', '')):
        raise ValueError('native cap provenance absent or invalid; legacy receipt is not qualified')
    for key in ('task_id', 'candidate_id', 'commit', 'lesson', 'reopen_when', 'upper_cap'):
        if m[key] != payload[key]:
            raise ValueError('native capture and journal binding mismatch')
    if m['executable_sha256'] != m['expected_evaluator_sha256']:
        raise ValueError('captured evaluator differs from configured pin')
    argv = m['argv']
    baseline = _read_capture(root / 'baseline', m['task_prompt'], argv, m['executable_sha256'], m['protocol_id'], m['source_repo'], m['commit'], m['code_artifacts'])
    prompt = 'LESSON (injected before the first turn):\n' + m['lesson'] + '\n\nTASK:\n' + m['task_prompt']
    actual = _read_capture(root / 'verification', prompt, argv, m['executable_sha256'], m['protocol_id'], m['source_repo'], m['commit'], m['code_artifacts'])
    if baseline != payload['r1'] or actual != verification['r_verify']:
        raise ValueError('native values differ from journal')
    return True


def main():
    import argparse

    from .journal import Journal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='JSON with producer arguments')
    parser.add_argument('--journal', required=True, help='existing initialized journal root')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    event = produce(Journal(args.journal), **config)
    print(json.dumps({'event_id': event.event_id, 'decision': event.payload['decision'],
                      'reason': event.payload['reason']}))


if __name__ == '__main__':
    main()
