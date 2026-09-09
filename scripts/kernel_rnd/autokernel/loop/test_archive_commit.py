"""Hermetic Git tests for accepted-patch commits with peer staging present."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from autokernel.loop import archive, kernel_mutation_guard


def _git(repo: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=False)
    if check and done.returncode:
        raise AssertionError(done.stderr)
    return done.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Archive Test")
    _git(repo, "config", "user.email", "archive@example.invalid")
    (repo / "selected.txt").write_text("base selected\n", encoding="utf-8")
    (repo / "peer.txt").write_text("base peer\n", encoding="utf-8")
    (repo / "nested").mkdir()
    (repo / "nested" / "file.txt").write_text("base nested\n", encoding="utf-8")
    _git(repo, "add", "selected.txt", "peer.txt", "nested/file.txt")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _index_bytes(repo: Path) -> bytes:
    path = Path(_git(repo, "rev-parse", "--git-path", "index"))
    if not path.is_absolute():
        path = repo / path
    return path.read_bytes()


def _install_hook(repo: Path, name: str, source: str) -> None:
    hook = repo / ".git" / "hooks" / name
    hook.write_text("#!/bin/sh\nset -eu\n" + source, encoding="utf-8")
    hook.chmod(0o755)


def test_peer_staging_is_absent_from_commit_and_index_bytes_are_unchanged(tmp_path):
    repo = _repo(tmp_path)
    (repo / "peer.txt").write_text("peer staged\n", encoding="utf-8")
    _git(repo, "add", "peer.txt")
    (repo / "selected.txt").write_text("accepted candidate\n", encoding="utf-8")
    before = _index_bytes(repo)
    peer_working = (repo / "peer.txt").read_bytes()

    head = archive.keep(repo, branch="main", message="accepted",
                        paths=("selected.txt",))

    assert _index_bytes(repo) == before
    assert (repo / "peer.txt").read_bytes() == peer_working
    assert _git(repo, "show", "--format=", "--name-only", head) == "selected.txt"
    assert _git(repo, "show", f"{head}:selected.txt") == "accepted candidate"
    assert _git(repo, "show", f"{head}:peer.txt") == "base peer"
    assert _git(repo, "show", "-s", "--format=%an <%ae>", head) == (
        "Archive Test <archive@example.invalid>")


def test_selected_peer_stage_is_not_substituted_for_working_candidate(tmp_path):
    repo = _repo(tmp_path)
    (repo / "selected.txt").write_text("peer staged version\n", encoding="utf-8")
    _git(repo, "add", "selected.txt")
    before = _index_bytes(repo)
    (repo / "selected.txt").write_text("accepted working candidate\n", encoding="utf-8")

    head = archive.keep(repo, branch="main", message="accepted working bytes",
                        paths=("selected.txt",))

    assert _git(repo, "show", f"{head}:selected.txt") == "accepted working candidate"
    assert _index_bytes(repo) == before
    assert _git(repo, "show", ":selected.txt") == "peer staged version"


def test_no_own_diff_does_not_commit_peer_staging(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "peer.txt").write_text("peer staged only\n", encoding="utf-8")
    _git(repo, "add", "peer.txt")
    before = _index_bytes(repo)

    with pytest.raises(archive.RatchetRefused, match="no committable change"):
        archive.keep(repo, branch="main", message="must refuse",
                     paths=("selected.txt",))

    assert _git(repo, "rev-parse", "HEAD") == expected
    assert _index_bytes(repo) == before


def test_tracked_deletion_is_committed_without_touching_shared_index(tmp_path):
    repo = _repo(tmp_path)
    (repo / "peer.txt").write_text("peer staged\n", encoding="utf-8")
    _git(repo, "add", "peer.txt")
    before = _index_bytes(repo)
    (repo / "selected.txt").unlink()

    head = archive.keep(repo, branch="main", message="delete selected",
                        paths=("selected.txt",))

    assert _index_bytes(repo) == before
    assert _git(repo, "show", "--format=", "--name-status", head) == "D\tselected.txt"


def test_detached_head_is_advanced_when_branch_is_HEAD(tmp_path):
    repo = _repo(tmp_path)
    parent = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", parent)
    (repo / "selected.txt").write_text("detached candidate\n", encoding="utf-8")
    before = _index_bytes(repo)

    head = archive.keep(repo, branch="HEAD", message="detached keep",
                        paths=("selected.txt",))

    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "rev-parse", f"{head}^") == parent
    assert not _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    assert _index_bytes(repo) == before


def test_attached_head_is_advanced_when_branch_is_HEAD(tmp_path):
    repo = _repo(tmp_path)
    parent = _git(repo, "rev-parse", "HEAD")
    (repo / "selected.txt").write_text("attached candidate\n", encoding="utf-8")
    before = _index_bytes(repo)

    head = archive.keep(repo, branch="HEAD", message="attached keep",
                        paths=("selected.txt",))

    assert _git(repo, "rev-parse", "HEAD") == head
    assert _git(repo, "rev-parse", "refs/heads/main") == head
    assert _git(repo, "rev-parse", f"{head}^") == parent
    assert _index_bytes(repo) == before


def test_named_branch_must_be_checked_out(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, "branch", "other")
    before = _index_bytes(repo)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "selected.txt").write_text("candidate\n", encoding="utf-8")

    with pytest.raises(archive.RatchetRefused, match="is not checked out"):
        archive.keep(repo, branch="other", message="wrong branch",
                     paths=("selected.txt",))
    assert _git(repo, "rev-parse", "HEAD") == expected
    assert _index_bytes(repo) == before


@pytest.mark.parametrize("bad", [
    "../selected.txt", ".", "nested", "*.txt", ":(glob)*", "nested/../peer.txt",
    " selected.txt", "selected.txt ",
])
def test_unsafe_path_or_directory_is_refused(tmp_path, bad):
    repo = _repo(tmp_path)
    before = _index_bytes(repo)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "selected.txt").write_text("candidate\n", encoding="utf-8")

    with pytest.raises(archive.RatchetRefused):
        archive.keep(repo, branch="main", message="unsafe", paths=(bad,))
    assert _git(repo, "rev-parse", "HEAD") == expected
    assert _index_bytes(repo) == before


def test_absolute_path_is_refused(tmp_path):
    repo = _repo(tmp_path)
    before = _index_bytes(repo)
    with pytest.raises(archive.RatchetRefused, match="literal relative"):
        archive.keep(repo, branch="main", message="unsafe",
                     paths=(str(repo / "selected.txt"),))
    assert _index_bytes(repo) == before


def test_nul_path_output_preserves_leading_space_exactly(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / " leading.txt").write_text("private candidate\n", encoding="utf-8")
    env = {
        "GIT_INDEX_FILE": str(tmp_path / "private-index"),
        "GIT_LITERAL_PATHSPECS": "1",
    }
    archive._git(repo, "read-tree", expected, env=env)
    archive._git(repo, "add", "--", " leading.txt", env=env)

    assert archive._changed_paths(repo, expected, env) == {" leading.txt"}


def test_repository_subdirectory_is_not_accepted_as_repo_root(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(archive.RatchetRefused, match="actual root"):
        archive.keep(repo / "nested", branch="main", message="wrong root",
                     paths=("file.txt",))


def test_frozen_root_is_refused_before_hooks_or_git_mutation(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    marker = tmp_path / "hook-ran"
    _install_hook(repo, "pre-commit", f"touch {marker}\n")
    parent = _git(repo, "rev-parse", "HEAD")
    (repo / "selected.txt").write_text("candidate\n", encoding="utf-8")
    monkeypatch.setattr(
        kernel_mutation_guard, "FROZEN_PRODUCTION_ROOTS", frozenset({repo}))

    with pytest.raises(archive.RatchetRefused,
                       match="canonical frozen production checkout"):
        archive.keep(repo, branch="main", message="must refuse",
                     paths=("selected.txt",))

    assert not marker.exists()
    assert _git(repo, "rev-parse", "HEAD") == parent


def test_branch_becoming_production_is_refused_before_ref_cas(tmp_path):
    repo = _repo(tmp_path)
    parent = _git(repo, "rev-parse", "HEAD")
    frozen_branch = "production-consolidated-v999"
    _git(repo, "branch", frozen_branch)
    (repo / "selected.txt").write_text("candidate\n", encoding="utf-8")
    _install_hook(
        repo, "pre-commit",
        f"git symbolic-ref HEAD refs/heads/{frozen_branch}\n")

    with pytest.raises(archive.RatchetRefused,
                       match="frozen production branch"):
        archive.keep(repo, branch="main", message="must refuse branch swap",
                     paths=("selected.txt",))

    assert _git(repo, "rev-parse", f"refs/heads/{frozen_branch}") == parent


def test_ambient_repository_redirection_is_refused_without_mutating_environment(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "wanted")
    other = _repo(tmp_path / "redirected")
    wanted_head = _git(repo, "rev-parse", "HEAD")
    other_head = _git(other, "rev-parse", "HEAD")
    git_dir = str(other / ".git")
    work_tree = str(other)
    monkeypatch.setenv("GIT_DIR", git_dir)
    monkeypatch.setenv("GIT_WORK_TREE", work_tree)

    with pytest.raises(archive.RatchetRefused, match="repository redirection"):
        archive.keep(repo, branch="main", message="must not redirect",
                     paths=("selected.txt",))

    assert os.environ["GIT_DIR"] == git_dir
    assert os.environ["GIT_WORK_TREE"] == work_tree
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_WORK_TREE")
    assert _git(repo, "rev-parse", "HEAD") == wanted_head
    assert _git(other, "rev-parse", "HEAD") == other_head


def test_changed_head_binding_at_same_commit_is_refused(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "other", expected)
    (repo / "selected.txt").write_text("accepted candidate\n", encoding="utf-8")
    _install_hook(repo, "commit-msg",
                  "git symbolic-ref HEAD refs/heads/other\n")

    with pytest.raises(archive.RatchetRefused, match="HEAD binding changed"):
        archive.keep(repo, branch="main", message="stale binding",
                     paths=("selected.txt",))

    assert _git(repo, "symbolic-ref", "HEAD") == "refs/heads/other"
    assert _git(repo, "rev-parse", "refs/heads/main") == expected
    assert _git(repo, "rev-parse", "refs/heads/other") == expected


def test_concurrent_parent_advance_wins_and_CAS_refuses_overwrite(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    before = _index_bytes(repo)
    (repo / "selected.txt").write_text("accepted candidate\n", encoding="utf-8")
    _install_hook(repo, "pre-commit", """
parent=$(git rev-parse HEAD)
tree=$(git rev-parse "${parent}^{tree}")
new=$(printf 'concurrent advance\\n' | git commit-tree "$tree" -p "$parent")
git update-ref refs/heads/main "$new" "$parent"
""")

    with pytest.raises(archive.RatchetRefused, match="HEAD moved"):
        archive.keep(repo, branch="main", message="losing candidate",
                     paths=("selected.txt",))

    winner = _git(repo, "rev-parse", "HEAD")
    assert winner != expected
    assert _git(repo, "show", "-s", "--format=%s", winner) == "concurrent advance"
    assert _git(repo, "rev-parse", f"{winner}^") == expected
    assert _index_bytes(repo) == before


@pytest.mark.parametrize("fault", [
    OSError("post hook unavailable"),
    subprocess.TimeoutExpired("post-commit", 600),
])
def test_post_commit_fault_warns_but_returns_landed_commit(tmp_path, monkeypatch,
                                                           capsys, fault):
    repo = _repo(tmp_path)
    parent = _git(repo, "rev-parse", "HEAD")
    (repo / "selected.txt").write_text("accepted candidate\n", encoding="utf-8")
    _install_hook(repo, "post-commit", "exit 0\n")
    real_run = subprocess.run

    def fail_post_hook(command, *args, **kwargs):
        if Path(command[0]).name == "post-commit":
            raise fault
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(archive.subprocess, "run", fail_post_hook)
    head = archive.keep(repo, branch="main", message="land despite notification",
                        paths=("selected.txt",))

    assert head == _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "rev-parse", f"{head}^") == parent
    assert "warning: post-commit hook could not report" in capsys.readouterr().err


def test_commit_hook_failure_preserves_ref_index_and_peer_working_file(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "peer.txt").write_text("peer staged and working\n", encoding="utf-8")
    _git(repo, "add", "peer.txt")
    before = _index_bytes(repo)
    peer_working = (repo / "peer.txt").read_bytes()
    (repo / "selected.txt").write_text("accepted candidate\n", encoding="utf-8")
    _install_hook(repo, "commit-msg", "echo 'policy rejected message' >&2\nexit 1\n")

    with pytest.raises(archive.RatchetRefused, match="commit-msg hook refused"):
        archive.keep(repo, branch="main", message="rejected",
                     paths=("selected.txt",))

    assert _git(repo, "rev-parse", "HEAD") == expected
    assert _index_bytes(repo) == before
    assert (repo / "peer.txt").read_bytes() == peer_working


def test_commit_creation_failure_preserves_ref_and_shared_index(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "peer.txt").write_text("peer staged\n", encoding="utf-8")
    _git(repo, "add", "peer.txt")
    before = _index_bytes(repo)
    (repo / "selected.txt").write_text("accepted candidate\n", encoding="utf-8")
    signer = repo / "reject-signing"
    signer.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    signer.chmod(0o755)
    _git(repo, "config", "commit.gpgSign", "true")
    _git(repo, "config", "user.signingKey", "missing-test-key")
    _git(repo, "config", "gpg.program", str(signer))

    with pytest.raises(archive.RatchetRefused, match="commit-tree"):
        archive.keep(repo, branch="main", message="cannot sign",
                     paths=("selected.txt",))

    assert _git(repo, "rev-parse", "HEAD") == expected
    assert _index_bytes(repo) == before


def test_index_stage_failure_preserves_ref_and_shared_index(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "peer.txt").write_text("peer staged\n", encoding="utf-8")
    _git(repo, "add", "peer.txt")
    before = _index_bytes(repo)

    with pytest.raises(archive.RatchetRefused, match="git add"):
        archive.keep(repo, branch="main", message="missing candidate",
                     paths=("missing.txt",))

    assert _git(repo, "rev-parse", "HEAD") == expected
    assert _index_bytes(repo) == before
