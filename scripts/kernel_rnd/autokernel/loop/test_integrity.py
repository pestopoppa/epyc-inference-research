from pathlib import Path
import subprocess

import pytest

from . import bench, integrity, loop, pipeline, pool, run


def test_missing_census_quant_is_not_a_candidate_validation_error():
    assert run._candidate_quant_tokens(None) == []
    assert run._candidate_quant_tokens("Q4_K") == ["Q4_K", "GGML_TYPE_Q4_K"]


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
    checked = integrity.validate_candidate(
        repo, ("ggml/src/kernel.cpp",),
        oracle_shape={"dims": [4096], "types": ["GGML_TYPE_Q4_K"]},
        bench_shape={"dims": [4096], "types": ["GGML_TYPE_Q4_K"]})
    assert checked.needs_confirm
    assert {row.kind for row in checked.findings} == {
        "literal_shape_predicate", "hot_path_mutable_state"}
    assert checked.to_dict()["needs_confirm"] is True
    assert checked.to_dict()["measured_tree"] == checked.tree
    predicate = next(row for row in checked.findings
                     if row.kind == "literal_shape_predicate")
    assert predicate.oracle_matches == ("4096", "GGML_TYPE_Q4_K")
    assert predicate.bench_matches == ("4096", "GGML_TYPE_Q4_K")
    with pytest.raises(integrity.IntegrityRefused) as caught:
        integrity.require_unseen_confirmation(
            checked, screen_surface="tg128", screen_model="public.gguf",
            confirm_surfaces=("tg128",), confirm_model="public.gguf")
    assert caught.value.refusal_class == "held_out_confirmation_missing"
    proof = integrity.require_unseen_confirmation(
        checked, screen_surface="tg128", screen_model="public.gguf",
        confirm_surfaces=("tg192",), confirm_model="held-out.gguf")
    assert proof["held_out_identities"] == [["tg192", "held-out.gguf"]]


def test_different_model_same_bench_shape_is_not_held_out(repo):
    (repo / "ggml/src/kernel.cpp").write_text(
        "int base = 1;\nif (src->ne[0] == 128) return;\n")
    checked = integrity.validate_candidate(repo, ("ggml/src/kernel.cpp",))
    with pytest.raises(integrity.IntegrityRefused) as caught:
        integrity.require_unseen_confirmation(
            checked, screen_surface="tg128", screen_model="public.gguf",
            confirm_surfaces=("tg128",), confirm_model="different.gguf")
    assert caught.value.refusal_class == "held_out_confirmation_missing"
    proof = integrity.require_unseen_confirmation(
        checked, screen_surface="tg128", screen_model="public.gguf",
        confirm_surfaces=("serving:decode",), confirm_model="public.gguf")
    assert proof["held_out_identities"] == [["serving:decode", "public.gguf"]]


def test_multiline_predicate_crosses_declared_shapes(repo):
    (repo / "ggml/src/kernel.cpp").write_text(
        "int base = 1;\nif (src->ne[0] ==\n    4096 && src->type ==\n    GGML_TYPE_Q4_K) return;\n")
    checked = integrity.validate_candidate(
        repo, ("ggml/src/kernel.cpp",),
        oracle_shape={"dims": [2048], "types": ["GGML_TYPE_Q4_K"]},
        bench_shape={"dims": [4096], "types": ["GGML_TYPE_Q4_K"]})
    finding = next(row for row in checked.findings
                   if row.kind == "literal_shape_predicate")
    assert finding.oracle_matches == ("GGML_TYPE_Q4_K",)
    assert finding.bench_matches == ("4096", "GGML_TYPE_Q4_K")


def test_added_read_of_existing_mutable_global_is_flagged(repo):
    kernel = repo / "ggml/src/kernel.cpp"
    kernel.write_text("int call_count = 0;\nint base = 1;\n")
    _git(repo, "add", "ggml/src/kernel.cpp")
    _git(repo, "commit", "-qm", "global state")
    kernel.write_text("int call_count = 0;\nint base = 1;\nif (ready) call_count++;\n")
    checked = integrity.validate_candidate(repo, ("ggml/src/kernel.cpp",))
    reads = [row for row in checked.findings
             if row.kind == "hot_path_mutable_state_read"]
    assert len(reads) == 1 and reads[0].oracle_matches == ("call_count",)


@pytest.mark.parametrize("protected", [
    "tests/other.cpp", "tools/llama-bench/hack.cpp", "tools/server/hack.cpp",
    "examples/hack.cpp", "scripts/hack.py", "ggml/src/nested/CMakeLists.txt",
])
def test_every_protected_path_class_is_refused(repo, protected):
    path = repo / protected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("added\n")
    with pytest.raises(integrity.IntegrityRefused) as caught:
        integrity.validate_candidate(repo, (protected,))
    assert caught.value.refusal_class == "oracle_bench_source_modified"


def test_evidence_identity_is_lane_and_attempt_specific():
    common = dict(base_commit="a" * 40, paths=("ggml/src/kernel.cpp",))
    first = integrity.evidence_key(lane="lane0", attempt_id="attempt-1", **common)
    assert first != integrity.evidence_key(
        lane="lane1", attempt_id="attempt-1", **common)
    assert first != integrity.evidence_key(
        lane="lane0", attempt_id="attempt-2", **common)


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


def test_kept_tree_mismatch_never_advances_champion_ref(tmp_path):
    repo = tmp_path / "repo"
    lane = tmp_path / "lane"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "champion")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    kernel = repo / "ggml/src/kernel.cpp"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("int base = 1;\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "refs/heads/champion")
    _git(repo, "worktree", "add", "--detach", str(lane), base)
    (lane / "ggml/src/kernel.cpp").write_text("int base = 2;\n")
    worker = pipeline.Worker("lane0", lane, tmp_path / "build")
    hypothesis = loop.Hypothesis("akm-tree", "change", "f", "ggml/src/kernel.cpp", "k")
    comparison = bench.Comparison(
        "tg128", [100.0], [102.0], 0.02, "median_over_median", 5, 1.0,
        {"invocations": 10, "resident": 10})
    with pytest.raises(ValueError, match="differs from measured"):
        pool.advance_champion(
            worker, hypothesis, ("ggml/src/kernel.cpp",), comparison,
            champion_tree=repo, branch="champion", expected_tree="0" * 40)
    assert _git(repo, "rev-parse", "refs/heads/champion") == base


def test_held_out_gap_is_persisted_on_actual_outcome():
    hypothesis = loop.Hypothesis("akm-held", "change", "f", "ggml/src/kernel.cpp", "k")

    class Planner:
        def propose(self, _context): return hypothesis
        def author(self, _hypothesis, _context): return ("ggml/src/kernel.cpp",)

    class Critic:
        def review_hypothesis(self, *_args): return loop.Review(True)
        def review_patch(self, *_args): return loop.Review(True)

    evidence = {"needs_confirm": True, "findings": [{"kind": "literal_shape_predicate"}]}
    comparison = bench.Comparison(
        "tg128", [100.0], [102.0], 0.02, "median_over_median", 5, 1.0,
        {"invocations": 10, "resident": 10})

    def commit(*_args):
        evidence["public_to_held_out_speedup_gap"] = [0.015]
        evidence["held_out_identities"] = [["tg192", "model.gguf"]]
        return "f" * 40

    outcome = loop.iterate(
        planner=Planner(), critic=Critic(), context={},
        validate_candidate=lambda *_args: evidence,
        gate=lambda *_args: (True, []), measure=lambda *_args: comparison,
        commit=commit)
    assert outcome.status == "kept"
    assert outcome.integrity_screen["public_to_held_out_speedup_gap"] == [0.015]
    assert outcome.to_attempt()["integrity_screen"]["held_out_identities"][0][0] == "tg192"
