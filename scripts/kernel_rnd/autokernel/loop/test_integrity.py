from pathlib import Path
import subprocess

import pytest

from . import integrity, loop, pipeline


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
    assert checked.to_dict()["needs_confirm"] is True
    assert checked.to_dict()["measured_tree"] == checked.tree
    with pytest.raises(integrity.IntegrityRefused) as caught:
        integrity.require_unseen_confirmation(
            checked, screen_surface="tg128", screen_model="public.gguf",
            confirm_surfaces=("tg128",), confirm_model="public.gguf")
    assert caught.value.refusal_class == "held_out_confirmation_missing"
    proof = integrity.require_unseen_confirmation(
        checked, screen_surface="tg128", screen_model="public.gguf",
        confirm_surfaces=("tg192",), confirm_model="held-out.gguf")
    assert proof["held_out_identities"] == [["tg192", "held-out.gguf"]]


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


def test_pipeline_returns_journalable_refusal_before_critic_or_build():
    hypothesis = loop.Hypothesis("akm-exploit", "edit oracle", "must refuse",
                                 "ggml/src/kernel.cpp", "kernel")

    class Planner:
        def propose(self, context):
            return hypothesis

        def author(self, proposed, context):
            return ("ggml/src/kernel.cpp",)

    calls = {"patch_critic": 0, "gate": 0}

    class Critic:
        def review_hypothesis(self, proposed, context):
            return loop.Review(True)

        def review_patch(self, proposed, paths, context):
            calls["patch_critic"] += 1
            return loop.Review(True)

    worker = pipeline.Worker("lane0", Path("/unused"), Path("/unused-build"))

    def gate(_worker):
        def invoke(*_args):
            calls["gate"] += 1
            return True, []
        return invoke

    outcomes = pipeline.run_pool(
        workers=[worker], make_planner=lambda _worker: Planner(),
        make_critic=lambda _worker: Critic(), build_context=dict, make_gate=gate,
        make_measure=lambda _worker: lambda *_args: None,
        commit=lambda *_args: "unused", champion_head=lambda: "a" * 40,
        reset_to_champion=lambda _worker: "a" * 40, record=lambda _outcome: None,
        iterations=1,
        validate_candidate=lambda *_args: (_ for _ in ()).throw(
            integrity.IntegrityRefused("dirty_set_mismatch", "undeclared oracle")))
    assert [outcome.status for outcome in outcomes] == ["integrity_refused"]
    assert calls == {"patch_critic": 0, "gate": 0}
    row = outcomes[0].to_attempt()
    assert row["integrity_screen"]["refusal_class"] == "dirty_set_mismatch"
    assert "dirty_set_mismatch" in row["reason"]
