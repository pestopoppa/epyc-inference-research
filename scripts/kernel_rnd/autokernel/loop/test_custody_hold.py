"""Failure and restart checks for the abandoned-candidate custody fence."""
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
from . import loop, pipeline, pool
from .test_candidate_dispositions import _Planner, _Critic, _hypothesis, _refusing_gate


def test_recorder_failure_stops_before_next_author(tmp_path):
    path = tmp_path / 'authored'
    class Planner(_Planner):
        calls = 0
        def author(self, hypothesis, context):
            self.calls += 1
            path.write_text(str(self.calls))
            return (path,)
    planner = Planner([_hypothesis()])
    def fail(_candidate):
        raise OSError('archive unavailable')
    with pytest.raises(loop.RunAborted, match='custody failed'):
        loop.iterate(planner=planner, critic=_Critic(), context={},
                     measure=lambda *a: None, gate=_refusing_gate, commit=lambda *a: None,
                     hypothesis_rounds=1, patch_rounds=2, record_abandoned=fail)
    assert planner.calls == 1 and path.read_text() == '1'


def test_error_only_recorder_stops_before_next_author(tmp_path):
    path = tmp_path / 'authored'
    class Planner(_Planner):
        calls = 0
        def author(self, hypothesis, context):
            self.calls += 1
            path.write_text(str(self.calls))
            return (path,)
    planner = Planner([_hypothesis()])
    def fail(candidate):
        candidate.retained_patch = {'error': 'retention failed'}
    with pytest.raises(loop.RunAborted, match='retention failed'):
        loop.iterate(planner=planner, critic=_Critic(), context={},
                     measure=lambda *a: None, gate=_refusing_gate, commit=lambda *a: None,
                     hypothesis_rounds=1, patch_rounds=2, record_abandoned=fail)
    assert planner.calls == 1 and path.read_text() == '1'


def test_hold_survives_process_exit_and_blocks_reset(tmp_path):
    lane = tmp_path / 'lane'; lane.mkdir()
    patch = lane / 'candidate.c'; patch.write_text('keep original')
    code = ('from pathlib import Path; from autokernel.loop import pool,pipeline; '
            'import os; w=pipeline.Worker(name="lane0",worktree=Path(os.environ["LANE"]),'
            'build_dir=Path(os.environ["LANE"])); pool.begin_custody_hold(w,"m"); os._exit(17)')
    env = dict(os.environ, LANE=str(lane))
    child = subprocess.run([sys.executable, '-c', code], env=env)
    assert child.returncode == 17
    worker = pipeline.Worker(name='lane0', worktree=lane, build_dir=lane)
    with mock.patch.object(pool, 'champion_head') as champion:
        with pytest.raises(RuntimeError, match='custody hold'):
            pool.reset_to_champion(worker)
        champion.assert_not_called()
    assert patch.read_text() == 'keep original'
    with pytest.raises(FileExistsError):
        pool.begin_custody_hold(worker, 'other')
    pool.finish_custody_hold(pool.custody_hold_path(worker))
    assert not (pool.custody_hold_path(worker)).exists()


def test_pool_failure_stops_run_and_restart_cannot_reset_lane(tmp_path):
    lane = tmp_path / 'lane'; lane.mkdir()
    patch = lane / 'candidate.c'
    class Planner(_Planner):
        calls = 0
        def author(self, hypothesis, context):
            self.calls += 1
            patch.write_text('valuable patch')
            return (patch,)
    planner = Planner([_hypothesis()])
    worker = pipeline.Worker('lane0', lane, lane)
    recorded = []
    def failed_record(w, candidate):
        pool.begin_custody_hold(w, candidate.hypothesis.mechanism_id)
        raise OSError('retention failed')
    with pytest.raises(loop.RunAborted, match='retention failed'):
        pipeline.run_pool(
            workers=[worker], make_planner=lambda w: planner,
            make_critic=lambda w: _Critic(), build_context=dict,
            make_gate=lambda w: _refusing_gate, make_measure=lambda w: None,
            commit=lambda *a: None, champion_head=lambda: 'c0',
            reset_to_champion=lambda w: 'c0', record=recorded.append,
            iterations=3, record_abandoned=failed_record)
    assert planner.calls == 1 and patch.read_text() == 'valuable patch'
    assert [item.status for item in recorded] == ['run_aborted']
    assert recorded[0].resumed_from is None
    with pytest.raises(RuntimeError, match='custody hold'):
        pool.reset_to_champion(worker)


@pytest.mark.parametrize('marker', ['partial', 'dangling_symlink'])
def test_incomplete_or_symlink_hold_fences_reset(tmp_path, marker):
    lane = tmp_path / 'lane'; lane.mkdir()
    worker = pipeline.Worker('lane0', lane, lane)
    hold = pool.custody_hold_path(worker)
    assert hold.parent == lane.parent and hold.parent != lane
    if marker == 'partial':
        hold.write_bytes(b'{')  # crash during marker creation, before fsync
    else:
        hold.symlink_to(tmp_path / 'missing')
    with mock.patch.object(pool, 'champion_head') as champion:
        with pytest.raises(RuntimeError, match='custody hold'):
            pool.reset_to_champion(worker)
        champion.assert_not_called()
    with pytest.raises(FileExistsError):
        pool.begin_custody_hold(worker, 'm')


def test_real_git_clean_cannot_remove_sibling_hold(tmp_path):
    lane = tmp_path / 'lane'; lane.mkdir()
    subprocess.run(['git', 'init', '-q', str(lane)], check=True)
    worker = pipeline.Worker('lane0', lane, lane)
    hold = pool.begin_custody_hold(worker, 'm')
    (lane / 'untracked.c').write_text('candidate')
    subprocess.run(['git', '-C', str(lane), 'clean', '-fdx'], check=True,
                   stdout=subprocess.DEVNULL)
    assert hold.is_file()
    with pytest.raises(RuntimeError, match='custody hold'):
        pool.reset_to_champion(worker)


def test_empty_pre_author_proof_refuses_dirty_ignored_and_changed_head(tmp_path):
    lane = tmp_path / 'lane'; lane.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(lane), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    git('config', 'user.email', 'fixture@example.invalid')
    git('config', 'user.name', 'fixture')
    (lane / '.gitignore').write_text('ignored\n')
    (lane / 'tracked').write_text('base')
    git('add', '.gitignore', 'tracked'); git('commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD')
    worker = pipeline.Worker('lane0', lane, lane)
    assert pool.prove_empty_pre_author(worker, base)
    assert not pool.prove_empty_pre_author(worker, None)
    (lane / 'untracked').write_text('x')
    assert not pool.prove_empty_pre_author(worker, base)
    (lane / 'untracked').unlink(); (lane / 'ignored').write_text('x')
    assert not pool.prove_empty_pre_author(worker, base)
    (lane / 'ignored').unlink(); (lane / 'tracked').write_text('changed')
    assert not pool.prove_empty_pre_author(worker, base)
    git('add', 'tracked'); git('commit', '-qm', 'changed')
    assert not pool.prove_empty_pre_author(worker, base)


@pytest.mark.parametrize('phase', ['before_archive', 'after_archive', 'after_ledger'])
def test_real_archive_ledger_crash_boundaries_leave_claim_and_lane_held(tmp_path, phase):
    from . import resume
    lane = tmp_path / 'lane'; lane.mkdir()
    store = tmp_path / 'store'; store.mkdir()
    worker = pipeline.Worker('lane0', lane, lane)
    (lane / 'candidate.c').write_text('original authored bytes')
    with resume.ClaimLedger(store) as ledger:
        assert ledger.claim('checkpoint', 'base', state='resumed')
    code = r'''
import os
from pathlib import Path
from autokernel.loop import archive, pool, pipeline, resume
lane = Path(os.environ['LANE']); store = Path(os.environ['STORE'])
worker = pipeline.Worker('lane0', lane, lane)
pool.begin_custody_hold(worker, 'mechanism', identity={'resumed_from':'checkpoint', 'spawn_parent':'base'})
if os.environ['PHASE'] == 'before_archive': os._exit(17)
a = {'status':'gate_refused', 'mechanism_id':'mechanism', 'reason':'refused', 'proposal_sha256':'fixture-attempt'}
archive.record(store, a, epoch='epoch', recorded_at='2026-01-01T00:00:00Z', campaign_id='ak-loop')
if os.environ['PHASE'] == 'after_archive': os._exit(17)
with resume.ClaimLedger(store) as ledger:
    assert ledger.settle_outcome('checkpoint', 'base', result_status='gate_refused') == 'pending'
os._exit(17)
'''
    child = subprocess.run([sys.executable, '-c', code], env=dict(os.environ,
                           LANE=str(lane), STORE=str(store), PHASE=phase))
    assert child.returncode == 17
    with resume.ClaimLedger(store) as ledger:
        row = ledger.rows()[0]
        assert row['state'] == 'resumed' and row['result_status'] is None
        assert not ledger.claim('checkpoint', 'base', state='resumed')
    assert (lane / 'candidate.c').read_text() == 'original authored bytes'
    with pytest.raises(RuntimeError, match='custody hold'):
        pool.reset_to_champion(worker)
    if phase != 'before_archive':
        from . import archive
        attempt = {'status':'gate_refused', 'mechanism_id':'mechanism',
                   'reason':'refused', 'proposal_sha256':'fixture-attempt'}
        assert not archive.record(store, attempt, epoch='epoch',
                                  recorded_at='2026-01-01T00:00:00Z', campaign_id='ak-loop')
