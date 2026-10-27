"""Model-free transaction tests against the actual script and full identity helper."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SCRIPT = Path(__file__).with_name('ablate.sh')
IDENT = Path(__file__).with_name('ablate_identity.py')
TRANSACTION = Path(__file__).with_name('ablate_transaction.sh')


def test_transaction(tmp_path):
    w = tmp_path / 'work'; w.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(w), *args], text=True)
    git('init', '-q'); git('config', 'user.name', 'fixture'); git('config', 'user.email', 'fixture@example.invalid')
    (w / 'lever').write_text('champion\n'); (w / 'obsolete').write_text('kept\n'); git('add', 'lever', 'obsolete'); git('commit', '-qm', 'champion'); git('checkout', '--detach', '-q')
    bd = w / 'build-cpu'; bd.mkdir(); (bd / 'lever.o').write_text('champion\n')
    h = tmp_path / 'h'; h.mkdir()
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps({'lever.o': hashlib.sha256(b'champion\n').hexdigest()}))
    exp = tmp_path / 'expected.json'; exp.write_text('["lever.o"]')
    runner = tmp_path / 'ablate.sh'; text = SCRIPT.read_text()
    # Retain all transaction logic; only replace the host-specific lock executor.
    original = '/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list "$FILLER_CPUS" --role build --tag "abl-$NAME" -- '
    assert text.count(original) == 1
    runner.write_text(text.replace(original, 'env ')); (tmp_path / 'ablate_identity.py').write_bytes(IDENT.read_bytes()); (tmp_path / 'ablate_transaction.sh').write_bytes(TRANSACTION.read_bytes())
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    cmake = bin_dir / 'cmake'; cmake.write_text('#!/bin/bash\nset -e\ncp "$ABL_WORKTREE/lever" "$ABL_WORKTREE/build-cpu/lever.o"\n'); cmake.chmod(0o755)
    env = dict(os.environ, ABL_WORKTREE=str(w), ABL_HARNESS=str(h), ABL_MANIFEST=str(manifest), FILLER_CPUS=str(min(os.sched_getaffinity(0))), FILLER_ID='fixture', FILLER_DEADLINE=str(int(time.time())+3600), FILLER_END=str(int(time.time())+3690), FILLER_STOPFILE=str(tmp_path/'STOP'), PATH=str(bin_dir) + os.pathsep + os.environ['PATH'])
    def run(name, mutation='printf arm > lever'):
        return subprocess.run(['bash', str(runner), name, str(exp), mutation], env=env, capture_output=True, text=True)
    # Expired and stopped grants must never invoke the mutation.
    env['FILLER_DEADLINE'] = str(int(time.time()) - 1)
    assert run('expired', 'touch never-expired').returncode != 0
    assert not (w/'never-expired').exists()
    env['FILLER_DEADLINE'] = str(int(time.time()) + 3600)
    Path(env['FILLER_STOPFILE']).touch()
    assert run('stopped', 'touch never-stopped').returncode != 0
    assert not (w/'never-stopped').exists()
    Path(env['FILLER_STOPFILE']).unlink()
    # A stop arriving during mutation refuses the arm build but still restores.
    result = run('during-stop', 'printf arm > lever; touch "$FILLER_STOPFILE"')
    assert result.returncode != 0
    assert (w/'lever').read_text() == 'champion\n'
    assert (bd/'lever.o').read_text() == 'champion\n'
    assert (h/'ablate/during-stop/restored.json').exists()
    assert not (h/'ablate/during-stop/arm.build.log').exists()
    assert not (bd/'.ablate-quarantined').exists()
    Path(env['FILLER_STOPFILE']).unlink()
    # A staged-only edit must not be cleared by the cleanup handler.
    (w / 'lever').write_text('staged')
    git('add', 'lever')
    (w / 'lever').write_text('champion\n')
    before = git('diff', '--cached')
    assert run('dirty').returncode != 0
    assert git('diff', '--cached') == before
    git('restore', '--staged', '--worktree', 'lever')
    assert run('good').returncode == 0
    assert (w/'lever').read_text() == 'champion\n'
    assert json.loads((h/'ablate/good/arm.json').read_text())['changed'] == ['lever.o']
    assert json.loads((h/'ablate/good/restored.json').read_text())['changed'] == []
    assert run('staged-arm', 'printf arm > lever; git add lever; git rm obsolete; printf added > new; git add new').returncode == 0
    assert (w/'obsolete').read_text() == 'kept\n' and not (w/'new').exists()
    assert git('diff', 'HEAD') == ''
    exp.write_text('["wrong.o"]')
    assert run('wrong').returncode != 0
    assert (w/'lever').read_text() == 'champion\n' and (bd/'lever.o').read_text() == 'champion\n'
    (bd/'lever.o').write_text('stale')
    assert run('stale').returncode != 0
    assert (w/'lever').read_text() == 'champion\n'
    assert not (bd/'.ablate-quarantined').exists()
    (bd/'lever.o').write_text('champion\n')
    cmake.write_text('#!/bin/bash\nexit 1\n'); cmake.chmod(0o755)
    assert run('buildfail').returncode != 0
    assert (w/'lever').read_text() == 'champion\n'
    assert (bd/'.ablate-quarantined').exists()


def test_identity_reports_all_changed_objects(tmp_path):
    build = tmp_path / 'build'; build.mkdir()
    original = {}
    for i in range(25):
        name = f'{i:02}.o'
        original[name] = hashlib.sha256(b'before').hexdigest()
        (build / name).write_bytes(b'after')
    ref = tmp_path / 'ref.json'; ref.write_text(json.dumps(original))
    expected = tmp_path / 'expected.json'; expected.write_text(json.dumps(sorted(original)))
    output = tmp_path / 'result.json'
    result = subprocess.run(['python3', str(IDENT), str(ref), str(build), str(expected), str(output)])
    assert result.returncode == 0
    assert len(json.loads(output.read_text())['changed']) == 25
