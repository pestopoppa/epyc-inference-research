from pathlib import Path

from . import resolved_recipe as rr
from . import serving
from . import surface_validation as sv


def _launch(tmp_path: Path, recipe: serving.Recipe, build: Path):
    artifact = lambda role, path, digit: {
        "schema": rr.ARTIFACT_SCHEMA, "role": role, "path": str(path), "sha256": digit * 64}
    return rr.resolve_recipe(recipe, build_dir=build, backend="cpu", port=18311,
        artifact_identities={"model": artifact("model", recipe.model, "1"), "drafter": None,
            "executable": artifact("executable", build / "bin" / "llama-server", "2"),
            "dsos": [artifact("dso", build / "bin" / "libggml.so", "3")]},
        environment_policy={"schema": rr.ENVIRONMENT_POLICY_SCHEMA, "version": "test-v1",
            "measurement_keys": [], "allowed_inherit_keys": [], "witnesses": {}},
        inherited_environment={})


def test_actual_serving_compare_row_preserves_existing_nonregression_policy(tmp_path, monkeypatch):
    recipe = serving.Recipe(name="test", model=str(tmp_path / "model.gguf"), device="none",
                            ngl=0, np=1, ctx=512, threads=1, batch=512, ubatch=512,
                            cpu_list="0")
    anchor_build, candidate_build = tmp_path / "a", tmp_path / "b"
    anchor, candidate = _launch(tmp_path, recipe, anchor_build), _launch(tmp_path, recipe, candidate_build)
    frozen = (("prompt-0", b'{"prompt":"hello"}'),)
    samples = iter([100.0, 100.5])
    def measured(*_args, **kwargs):
        kwargs["evidence"].append({"backend": "cpu", "status": "not_applicable",
            "window_start": 1.0, "window_end": 2.0})
        return next(samples)
    monkeypatch.setattr(serving, "_measure_once", measured)
    comparison = serving.compare(
        recipe, anchor_build, candidate_build, pairs=1, floor_pct=1.0,
        anchor_resolved_recipe=anchor, candidate_resolved_recipe=candidate,
        frozen_requests=frozen, floor_request_digest=serving.request_digest(recipe, frozen))
    body = sv.row(source_commit="a" * 40, source_tree="b" * 40,
        source_keep_ids=["keep-1"], target={"selected_id": "target"},
        original_anchor={"path": str(anchor_build.resolve()), "commit": "0" * 40},
        candidate_anchor={"path": str(candidate_build.resolve()), "commit": "a" * 40},
        request_digest=serving.request_digest(recipe, frozen),
        recipe_execution_digest=candidate.execution_digest,
        comparison=comparison, intended_target=False)
    assert body["disposition"] == "passed"


def test_missing_original_anchor_identity_is_retained_as_pending_debt(tmp_path):
    body = sv.debt(
        source_commit="a" * 40, source_tree="b" * 40, source_keep_ids=["keep"],
        target={"selected_id": "gpu"},
        original_anchor={"path": str((tmp_path / "handbuilt").resolve()), "commit": None},
        candidate_anchor={"path": str((tmp_path / "candidate").resolve()),
                          "commit": "a" * 40},
        request_digest="c" * 64, recipe_execution_digest="d" * 64,
        reason="original anchor source identity is unavailable",
        failure={"type": "original_anchor_identity_unavailable"})
    reference = sv.retain(tmp_path, body)
    reopened = sv.reopen_reference(reference)
    assert reopened["disposition"] == "pending"
    assert reopened["original_anchor"]["commit"] is None
