from pathlib import Path
import subprocess

import pytest

from . import integrity


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True)
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    for name in ("ggml/src/kernel.cpp", "tests/test-backend-ops.cpp"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("int base = 1;\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def test_undeclared_backend_oracle_edit_is_refused_before_build(repo):
    (repo / "ggml/src/kernel.cpp").write_text("int base = 2;\n")
    # Recorded exploit shape: the candidate edits the self-built oracle while
    # declaring only its apparent kernel edit.
    (repo / "tests/test-backend-ops.cpp").write_text("int base = 999;\n")
    with pytest.raises(integrity.IntegrityRefused) as caught:
        integrity.validate_candidate(repo, ("ggml/src/kernel.cpp",))
    assert caught.value.refusal_class == "dirty_set_mismatch"
    assert "tests/test-backend-ops.cpp" in str(caught.value)


def test_declared_oracle_edit_has_durable_protected_refusal(repo):
    (repo / "tests/test-backend-ops.cpp").write_text("int base = 999;\n")
    with pytest.raises(integrity.IntegrityRefused) as caught:
        integrity.validate_candidate(repo, ("tests/test-backend-ops.cpp",))
    assert caught.value.refusal_class == "oracle_bench_source_modified"


def test_literal_shape_predicate_and_mutable_state_need_confirmation(repo):
    (repo / "ggml/src/kernel.cpp").write_text(
        "int base = 1;\n"
        "+static int calls = 0;\n".replace("+", "")
        + "if (src->ne[0] == 4096 && src->type == GGML_TYPE_Q4_K) calls++;\n")
    checked = integrity.validate_candidate(repo, ("ggml/src/kernel.cpp",))
    assert checked.needs_confirm
    assert {row.kind for row in checked.findings} == {
        "literal_shape_predicate", "hot_path_mutable_state"}


def test_measured_tree_and_kept_commit_are_exact(repo):
    path = repo / "ggml/src/kernel.cpp"
    path.write_text("int base = 2;\n")
    checked = integrity.validate_candidate(repo, ("ggml/src/kernel.cpp",))
    integrity.assert_measured_tree(repo, checked.tree)
    _git(repo, "add", "ggml/src/kernel.cpp")
    _git(repo, "commit", "-qm", "keep")
    head = _git(repo, "rev-parse", "HEAD")
    integrity.assert_kept_commit(repo, head, checked.tree)

    path.write_text("int base = 3;\n")
    with pytest.raises(integrity.IntegrityRefused) as caught:
        integrity.assert_measured_tree(repo, checked.tree)
    assert caught.value.refusal_class == "measured_tree_changed"
