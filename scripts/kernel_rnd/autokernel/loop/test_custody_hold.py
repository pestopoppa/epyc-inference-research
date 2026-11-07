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
    # Bind the child to THIS tree: the research venv's .pth puts the shared clone on
    # sys.path, so from a lane worktree the child would import the shared clone's
    # scripts.lib.canonical_recipe and recipes.py rightly refuses it as foreign.
    here = Path(__file__).resolve()
    tree_path = os.pathsep.join([str(here.parents[4]), str(here.parents[2])])
    env = dict(os.environ, LANE=str(lane),
               PYTHONPATH=tree_path + (os.pathsep + os.environ['PYTHONPATH']
                                       if os.environ.get('PYTHONPATH') else ''))
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


def test_final_recorder_failure_stops_pool_without_another_draw(tmp_path):
    worker = pipeline.Worker("lane0", tmp_path, tmp_path)
    planner = _Planner([_hypothesis()])
    calls = []
    def reset(_worker):
        calls.append("reset")
        return "base"
    def fail(_outcome):
        raise OSError("journal unavailable")
    with pytest.raises(loop.RunAborted, match="journal unavailable"):
        pipeline.run_pool(
            workers=[worker], make_planner=lambda w: planner,
            make_critic=lambda w: _Critic(), build_context=dict,
            make_gate=lambda w: _refusing_gate, make_measure=lambda w: None,
            commit=lambda *a: None, champion_head=lambda: "base",
            reset_to_champion=reset, record=fail, iterations=3,
            record_abandoned=lambda w, c: None)  # CONTROL_ONLY: fake gate, no source tree
    assert calls == ["reset"]



def _fold_lane(tmp_path):
    from . import run
    lane = tmp_path / "lane"; lane.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(lane), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "fixture")
    path = lane / "tracked.c"
    path.write_text("base\n")
    git("add", "tracked.c"); git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    path.write_text("committed fold\n")
    git("add", "tracked.c"); git("commit", "-qm", "fold")
    path.write_text("committed fold\ndirty fold\n")
    return run, pipeline.Worker("lane0", lane, lane), base, git


def test_fold_reset_disposition_retains_committed_and_dirty_bytes(tmp_path):
    import json
    run, worker, base, git = _fold_lane(tmp_path)
    manifest = run._retain_fold_before_reset(worker, store=tmp_path / "store",
                                             base=base, mechanism="foldcheck")
    record = json.loads(manifest.read_text())
    assert record["reason"] and record["base"] == base
    assert b"committed fold" in Path(record["committed_patch"]).read_bytes()
    assert b"dirty fold" in Path(record["dirty_patch"]).read_bytes()
    assert not pool.custody_hold_path(worker).exists()


def test_fold_retention_failure_leaves_committed_tree_held(tmp_path):
    run, worker, base, git = _fold_lane(tmp_path)
    head = git("rev-parse", "HEAD")
    with mock.patch.object(run.archive, "retain_patch_bytes", side_effect=OSError("store unavailable")):
        with pytest.raises(loop.RunAborted, match="lane remains held"):
            run._retain_fold_before_reset(worker, store=tmp_path / "store",
                                          base=base, mechanism="foldcheck")
    assert pool.custody_hold_path(worker).exists()
    assert git("rev-parse", "HEAD") == head
    assert (worker.worktree / "tracked.c").read_text() == "committed fold\ndirty fold\n"
    with pytest.raises(RuntimeError, match="custody hold"):
        pool.reset_to_champion(worker)



def test_author_checkpoint_precedes_call_and_patch_disposition_precedes_release(tmp_path):
    import json
    run, worker, base, git = _fold_lane(tmp_path)
    hypothesis = _hypothesis()
    ticket = run._begin_author_custody(worker, hypothesis, {"stage": "author"},
                                       store=tmp_path / "store")
    assert pool.custody_hold_path(worker).exists()
    hold = json.loads(pool.custody_hold_path(worker).read_text())
    assert hold["identity"]["hypothesis"]["mechanism_id"] == hypothesis.mechanism_id
    inflight = json.loads(ticket["record"].read_text())
    assert inflight["hypothesis"]["mechanism_id"] == hypothesis.mechanism_id
    retained = run._finish_author_custody(worker, ticket, hypothesis,
                                          "ActorTransient: transport failed",
                                          store=tmp_path / "store")
    assert b"dirty fold" in Path(retained["patch_file"]).read_bytes()
    row = json.loads(Path(retained["disposition_record"]).read_text())
    assert row["reason"] == "ActorTransient: transport failed"
    assert row["hypothesis_record"] == str(ticket["record"].resolve())
    assert not pool.custody_hold_path(worker).exists()


def test_author_capture_failure_keeps_lane_and_accepted_hypothesis(tmp_path):
    run, worker, base, git = _fold_lane(tmp_path)
    hypothesis = _hypothesis()
    ticket = run._begin_author_custody(worker, hypothesis, {"stage": "author"},
                                       store=tmp_path / "store")
    with mock.patch.object(run.archive, "retain_patch_bytes", side_effect=OSError("store unavailable")):
        with pytest.raises(loop.RunAborted, match="lane remains held"):
            run._finish_author_custody(worker, ticket, hypothesis, None,
                                       store=tmp_path / "store")
    assert ticket["record"].is_file()
    assert pool.custody_hold_path(worker).exists()
    assert (worker.worktree / "tracked.c").read_text() == "committed fold\ndirty fold\n"



def test_transient_author_keeps_exact_written_patch_and_failure_reason(tmp_path):
    import json
    run, worker, base, git = _fold_lane(tmp_path)
    git("restore", "tracked.c")
    hypothesis = _hypothesis()
    class FailingPlanner(_Planner):
        def author(self, hypothesis, context):
            (worker.worktree / "tracked.c").write_text("partial authored edit\n")
            raise loop.ActorTransient("provider lost after editing")
    outcome = loop.iterate(
        planner=FailingPlanner([hypothesis]), critic=_Critic(), context={},
        gate=lambda *a: (True, []), measure=lambda *a: None, commit=lambda *a: None,
        author_lane=(worker.worktree, git("rev-parse", "HEAD")),
        begin_author_custody=lambda h, ck: run._begin_author_custody(
            worker, h, ck, store=tmp_path / "store"),
        finish_author_custody=lambda ticket, h, failure: run._finish_author_custody(
            worker, ticket, h, failure, store=tmp_path / "store"))
    retained = outcome.retained_patch
    assert b"partial authored edit" in Path(retained["patch_file"]).read_bytes()
    row = json.loads(Path(retained["disposition_record"]).read_text())
    assert "provider lost after editing" in row["reason"]
    assert not pool.custody_hold_path(worker).exists()



def test_pooled_author_transient_records_hypothesis_and_actual_patch(tmp_path):
    import json
    run, worker, base, git = _fold_lane(tmp_path)
    git("restore", "tracked.c")
    hypothesis = _hypothesis()
    class FailingPlanner(_Planner):
        def author(self, hypothesis, context):
            (worker.worktree / "tracked.c").write_text("pooled source edit\n")
            raise loop.ActorTransient("provider failed after edit")
    records = []
    outcomes = pipeline.run_pool(
        workers=[worker], make_planner=lambda w: FailingPlanner([hypothesis]),
        make_critic=lambda w: _Critic(), build_context=dict,
        make_gate=lambda w: lambda *a: (True, []),
        make_measure=lambda w: lambda *a: None, commit=lambda *a: None,
        champion_head=lambda: git("rev-parse", "HEAD"),
        reset_to_champion=lambda w: git("rev-parse", "HEAD"),
        record=records.append, iterations=1,
        begin_author_custody=lambda w, h, ck: run._begin_author_custody(
            w, h, ck, store=tmp_path / "store"),
        finish_author_custody=lambda w, ticket, h, failure: run._finish_author_custody(
            w, ticket, h, failure, store=tmp_path / "store"))
    assert len(outcomes) == len(records) == 1
    retained = records[0].retained_patch
    assert b"pooled source edit" in Path(retained["patch_file"]).read_bytes()
    disposition = json.loads(Path(retained["disposition_record"]).read_text())
    assert "provider failed after edit" in disposition["reason"]
    hypothesis_record = json.loads(Path(disposition["hypothesis_record"]).read_text())
    assert hypothesis_record["hypothesis"]["mechanism_id"] == hypothesis.mechanism_id



def test_missing_disposition_recorder_stops_before_another_author(tmp_path):
    run, worker, base, git = _fold_lane(tmp_path)
    git("restore", "tracked.c")
    hypothesis = _hypothesis()
    calls = []
    class Planner(_Planner):
        def author(self, hypothesis, context):
            calls.append("author")
            (worker.worktree / "tracked.c").write_text("the first patch\n")
            return ("tracked.c",)
    with pytest.raises(loop.RunAborted, match="no recorder"):
        pipeline.run_pool(
            workers=[worker], make_planner=lambda w: Planner([hypothesis]),
            make_critic=lambda w: _Critic(), build_context=dict,
            make_gate=lambda w: _refusing_gate, make_measure=lambda w: lambda *a: None,
            commit=lambda *a: None, champion_head=lambda: git("rev-parse", "HEAD"),
            reset_to_champion=lambda w: git("rev-parse", "HEAD"),
            record=lambda o: None, iterations=2,
            begin_author_custody=lambda w, h, ck: run._begin_author_custody(
                w, h, ck, store=tmp_path / "store"),
            finish_author_custody=lambda w, ticket, h, failure: run._finish_author_custody(
                w, ticket, h, failure, store=tmp_path / "store"))
    assert calls == ["author"]
    assert (worker.worktree / "tracked.c").read_text() == "the first patch\n"
    assert list((tmp_path / "store" / "patches").glob("*.patch"))



def test_author_commit_and_dirty_source_survive_transient_and_reset(tmp_path):
    import json
    run, worker, base, git = _fold_lane(tmp_path)
    git("restore", "tracked.c")
    hypothesis = _hypothesis()
    author_base = git("rev-parse", "HEAD")
    class CommitPlanner(_Planner):
        def author(self, hypothesis, context):
            (worker.worktree / "tracked.c").write_text("authored commit\n")
            git("add", "tracked.c")
            git("commit", "-qm", "authored")
            (worker.worktree / "tracked.c").write_text("authored commit\ndirty edit\n")
            (worker.worktree / "src").mkdir()
            (worker.worktree / "src" / "new.c").write_text("new source\n")
            raise loop.ActorTransient("failure after commit and edits")
    outcome = loop.iterate(
        planner=CommitPlanner([hypothesis]), critic=_Critic(), context={},
        gate=lambda *a: (True, []), measure=lambda *a: None, commit=lambda *a: None,
        author_lane=(worker.worktree, author_base),
        begin_author_custody=lambda h, ck: run._begin_author_custody(
            worker, h, ck, store=tmp_path / "store"),
        finish_author_custody=lambda ticket, h, failure: run._finish_author_custody(
            worker, ticket, h, failure, store=tmp_path / "store"))
    pointer = outcome.retained_patch
    patch = Path(pointer["patch_file"]).read_bytes()
    sidecar = json.loads(Path(pointer["metadata_file"]).read_text())
    assert sidecar["original_head"] == author_base
    assert b"authored commit" in patch and b"dirty edit" in patch and b"new source" in patch
    assert "failure after commit and edits" in json.loads(
        Path(pointer["disposition_record"]).read_text())["reason"]
    git("reset", "--hard", author_base)
    git("clean", "-fd", "--", "src")
    subprocess.run(["git", "-C", str(worker.worktree), "apply", "--check", "--binary", "-"],
                   input=patch, check=True)
    subprocess.run(["git", "-C", str(worker.worktree), "apply", "--binary", "-"],
                   input=patch, check=True)
    assert (worker.worktree / "tracked.c").read_text() == "authored commit\ndirty edit\n"
    assert (worker.worktree / "src" / "new.c").read_text() == "new source\n"
