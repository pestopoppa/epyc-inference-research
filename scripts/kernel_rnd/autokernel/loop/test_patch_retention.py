"""Original source survives interrupted formation, without execution authority."""
import hashlib
import json
import os
from unittest import mock

import pytest

from . import archive, pool, run
from .test_archive_commit import _git, _index_bytes, _repo
from .test_promotion_targets import TheKeepBuildsAProductionCompleteAnchor


def test_tracked_recipes_docs_and_new_kernel_text_remain_exact_and_immutable(tmp_path):
    repo = _repo(tmp_path)
    (repo / "CMakeLists.txt").write_text("# original build recipe\n")
    (repo / "README.md").write_text("original docs\n")
    _git(repo, "add", "CMakeLists.txt", "README.md")
    _git(repo, "commit", "-qm", "recipe and docs")
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "CMakeLists.txt").write_text("# changed build recipe\n")
    (repo / "README.md").write_text("changed docs\n")
    _git(repo, "add", "README.md")
    source = repo / "ggml/src/ggml-cpu/new.cpp"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first\r\nsecond\x0bthird")
    (source.parent / "empty.h").touch()
    (source.parent / "credentials.key").write_text("DO NOT ARCHIVE")
    (source.parent / "build.so").write_bytes(b"\x00DO NOT ARCHIVE")
    (repo / "scratch.txt").write_text("DO NOT ARCHIVE")
    original_index = _index_bytes(repo)
    path = archive.retain_patch(tmp_path / "store", repo, lane="lane0", mechanism_id="same")
    body = json.loads(path.with_suffix(".json").read_text())
    raw = path.read_bytes()
    assert body["original_head"] == head
    assert body["patch_sha256"] == hashlib.sha256(raw).hexdigest()
    assert body["scope"] == "source_only_not_execution_evidence"
    assert b"CMakeLists.txt" in raw and b"README.md" in raw
    assert b"+first\r\n+second\x0bthird\n\\ No newline at end of file\n" in raw
    assert b"empty.h" in raw and b"DO NOT ARCHIVE" not in raw
    assert _index_bytes(repo) == original_index
    assert archive.retain_patch(tmp_path / "store", repo, lane="lane0", mechanism_id="same") == path
    clean = tmp_path / "clean"
    _git(repo, "worktree", "add", "--detach", str(clean), head)
    _git(clean, "apply", "--check", str(path))
    _git(clean, "apply", str(path))
    assert (clean / "CMakeLists.txt").read_bytes() == (repo / "CMakeLists.txt").read_bytes()
    assert (clean / "README.md").read_bytes() == (repo / "README.md").read_bytes()
    assert (clean / "ggml/src/ggml-cpu/new.cpp").read_bytes() == source.read_bytes()
    assert (clean / "ggml/src/ggml-cpu/empty.h").read_bytes() == b""
    source.write_text("different attempt\n")
    changed = archive.retain_patch(tmp_path / "store", repo, lane="lane0", mechanism_id="same")
    assert changed != path and path.read_bytes() == raw
    assert len(list(path.parent.glob("same.lane0.*.patch"))) == 2


@pytest.mark.parametrize("kind", ["symlink", "binary", "fifo"])
def test_untracked_non_source_content_cannot_be_archived(tmp_path, monkeypatch, kind):
    repo = _repo(tmp_path)
    path = repo / "src/new.cpp"
    path.parent.mkdir()
    if kind == "symlink":
        secret = tmp_path / "secret"
        secret.write_text("do not follow")
        path.symlink_to(secret)
    elif kind == "binary":
        path.write_bytes(b"\0binary masquerading as source")
    else:
        path.write_text("source before enumeration\n")
        original = archive.subprocess.run
        def substitute(argv, **kwargs):
            result = original(argv, **kwargs)
            if "ls-files" in argv:
                path.unlink()
                os.mkfifo(path)
            return result
        monkeypatch.setattr(archive.subprocess, "run", substitute)
    with pytest.raises(archive.RatchetRefused):
        archive.retain_patch(tmp_path / "store", repo, lane="lane0")
    assert not (tmp_path / "store/patches").exists()


def test_clean_lane_has_no_archive_and_changed_immutable_file_refuses(tmp_path):
    repo = _repo(tmp_path)
    store = tmp_path / "store"
    assert archive.retain_patch(store, repo, lane="lane0") is None
    assert not store.exists()
    (repo / "selected.txt").write_text("changed\n")
    path = archive.retain_patch(store, repo, lane="lane0")
    path.write_text("tampered retained artifact")
    with pytest.raises(archive.RatchetRefused, match="immutable"):
        archive.retain_patch(store, repo, lane="lane0")


def test_actual_run_stop_before_gate_preserves_patch_before_next_owned_reset():
    fixture = TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    original_main, original_reset = run.main, pool.reset_to_champion
    stopped = []
    protected = []
    try:
        def selected_main(argv):
            installed_planner = run.actors.AgentPlanner
            def planner(*args, **kwargs):
                actor = installed_planner(*args, **kwargs)
                author = actor.author
                def stop_after_author(*args):
                    paths = author(*args)
                    if not stopped:
                        stopped.append(True)
                        (fixture.store / "STOP").touch()
                    return paths
                actor.author = stop_after_author
                return actor
            with mock.patch.object(run.actors, "AgentPlanner", planner):
                return original_main(argv)

        def checked_reset(worker, **kwargs):
            if (worker.worktree / "kernel.c").read_text() == "patched\n":
                paths = list((fixture.store / "patches").glob("interrupted.lane0.*.patch"))
                assert paths, "original dirty source must be retained before reset"
                metadata = json.loads(paths[0].with_suffix(".json").read_text())
                assert metadata["original_head"] == fixture.tip
                assert b"+patched\n" in paths[0].read_bytes()
                protected.append(paths[0])
            return original_reset(worker, **kwargs)

        with mock.patch.object(run, "main", selected_main), \
                mock.patch.object(pool, "reset_to_champion", checked_reset):
            rc, builds, _, _, log = fixture._run_one_keep()
            assert rc == 0 and not builds and "stopped_mid_formation" in log
            assert (fixture.root / "lane0/kernel.c").read_text() == "patched\n"
            (fixture.store / "STOP").unlink()
            rc, builds, _, _, log = fixture._run_one_keep()
            assert rc == 0 and builds and "kept" in log
        assert len(protected) == 1
    finally:
        fixture.doCleanups()


def test_actual_run_archive_failure_prevents_dirty_lane_reset():
    fixture = TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        lane = fixture.root / "lane0"
        _git(fixture.repo, "worktree", "add", "--detach", str(lane), fixture.tip)
        (lane / "kernel.c").write_text("interrupted original source\n")
        with mock.patch.object(archive, "_retain_bytes", side_effect=OSError("archive unavailable")), \
                mock.patch.object(pool, "reset_to_champion", side_effect=AssertionError("must not reset")):
            rc, builds, _, _, log = fixture._run_one_keep()
        assert rc == 0 and not builds and "lane_error" in log
        assert (lane / "kernel.c").read_text() == "interrupted original source\n"
        assert _git(lane, "rev-parse", "HEAD") == fixture.tip
    finally:
        fixture.doCleanups()
