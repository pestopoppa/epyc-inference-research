from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from . import archive, claim, gates, resolved_recipe as rr, serving, serving_beliefs
from . import source_loo, surface_fold
from .loop import MeasurementInvalid, RunAborted


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@contextmanager
def _held(path, cpu, device="cpu"):
    with path.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = claim.HeldCpuClaim({"device_id": device}, [path], affinity=(cpu,))
        assert owner.observe()["status"] == "held"
        try:
            yield owner
        finally:
            owner._closing()
            fcntl.flock(stream, fcntl.LOCK_UN)
            owner._released_now()


def _files(build):
    (build / "bin").mkdir(parents=True)
    for name in ("llama-server", "libggml.so", "test-backend-ops"):
        (build / "bin" / name).write_text("tiny nonexecuted build fixture\n")


def _launch(recipe, build, backend):
    def artifact(role, path):
        return rr.ArtifactDigest(role, str(path), hashlib.sha256(path.read_bytes()).hexdigest()).to_dict()
    command = recipe.server_argv(build, 18311)
    return rr.resolve_canonical_launch(recipe, build_dir=build,
        command_argv=command[3:], topology_prefix=command[:3],
        launch_environment=recipe.server_env(build, base={}),
        artifact_identities={"model": artifact("model", Path(recipe.model)), "drafter": None,
            "executable": artifact("executable", build / "bin" / "llama-server"),
            "dsos": [artifact("dso", build / "bin" / "libggml.so")]},
        backend=backend, environment_policy={"schema": rr.ENVIRONMENT_POLICY_SCHEMA,
            "version": "tiny-fixture", "measurement_keys": [], "allowed_inherit_keys": [],
            "witnesses": {}}, port=18311, runtime_binary_dir=str(build / "bin"),
        runtime_ld_paths=(str(build / "bin"),), provenance={"export_sha256": "0" * 64,
            "instance_mode": "full", "source:fixture": "tiny-not-production"})


def _inputs(tmp_path, backend="cpu"):
    cpu = min(os.sched_getaffinity(0))
    repo, store = tmp_path / "repo", tmp_path / "store"
    repo.mkdir()
    _git(repo, "init", "-b", "experimental")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "kernel.c").write_text("int keep = 0;\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "original")
    parent = _git(repo, "rev-parse", "HEAD")
    (repo / "kernel.c").write_text("int keep = 1;\n")
    patch = archive.retain_patch(store, repo, lane="fixture", mechanism_id="first-keep")
    _git(repo, "commit", "-qam", "keep")
    kept = _git(repo, "rev-parse", "HEAD")
    original = {"fixture": "original retained source receipt, not hardware qualification"}
    receipt = surface_fold.ExperimentalKeepReceipt(
        "serving", "first-keep", "request", str(repo), "experimental", parent, kept,
        str(patch), hashlib.sha256(patch.read_bytes()).hexdigest(), str(patch.with_suffix(".json")),
        hashlib.sha256(patch.with_suffix(".json").read_bytes()).hexdigest(),
        {"selected_id": "fixture"}, "1" * 64, "2" * 64, "process", original,
        surface_fold._digest(original))
    reference = surface_fold.receipt_reference(surface_fold.retain_receipt(store, receipt))
    (repo / "later.c").write_text("int later = 2;\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "later independent keep remains in the full stack")
    tip = _git(repo, "rev-parse", "HEAD")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"tiny model identity fixture, never launched")
    recipe = serving.Recipe(name="tiny", model=str(model), cpu_list=str(cpu), np=1,
        ctx=512, threads=1, batch=512, ubatch=512, device="none" if backend == "cpu" else "ROCm0",
        ngl=0 if backend == "cpu" else 99)
    full, old = tmp_path / "full", tmp_path / "old"
    for build, commit in ((full, tip), (old, parent)):
        _files(build)
        (build / "provenance.json").write_text(json.dumps({"champion_commit": commit}))
    frozen = (("prompt-0", b'{"prompt":"original exact request","n_predict":4}'),)
    serving.write_floor(store, recipe, {"floor_pct": 1.0, "recipe_hash": recipe.recipe_hash,
        "request_digest": serving.request_digest(recipe, frozen)}, frozen_requests=frozen)
    return cpu, dict(directory=tmp_path / "operation", store_root=store, repo=repo,
        assembled_commit=tip, assembled_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
        keep_references=[reference], target={"selected_id": backend, "model": str(model)},
        full_launch=_launch(recipe, full, backend), baseline=(parent, _launch(recipe, old, backend)),
        frozen_requests=frozen, instrument=serving.LEGACY_INSTRUMENT, pairs=1,
        cmake_defines=("GGML_NATIVE=ON",), jobs=1, build_cpu_list=str(cpu),
        epoch="fixture-epoch", campaign_id="fixture-loo", should_stop=lambda: False)


def test_keep_receipt_allows_embedded_lifecycle_telemetry_over_metadata_bound(tmp_path):
    _, inputs = _inputs(tmp_path)
    original = surface_fold.reopen_reference(inputs["keep_references"][0]).to_dict()
    original["comparison"] = {"lifecycle_telemetry": "x" * (surface_fold.MAX_RECEIPT_BYTES + 1)}
    original["comparison_digest"] = surface_fold._digest(original["comparison"])
    receipt = surface_fold.ExperimentalKeepReceipt.from_dict(original)

    retained = surface_fold.retain_receipt(inputs["store_root"], receipt)
    assert retained.stat().st_size > surface_fold.MAX_RECEIPT_BYTES
    reference = surface_fold.receipt_reference(retained)
    assert surface_fold.reopen_reference(reference) == receipt


def _hardware_edges(monkeypatch, inputs, *, rates=(100.0, 90.0, 100.0, 110.0), failure=None):
    calls = []
    def compile(source, build, **kwargs):
        calls.append(("build", source, kwargs))
        assert (source / "kernel.c").read_text() == "int keep = 0;\n"
        assert (source / "later.c").read_text() == "int later = 2;\n"
        assert _git(source, "rev-parse", "HEAD") == inputs["assembled_commit"]
        assert kwargs["targets"] == gates.PROMOTION_TARGETS
        if failure == "build":
            return gates.Verdict("compiles", False, "fixture build failure")
        _files(build)
        return gates.Verdict("compiles", True, "fixture compiler")
    def oracle(build, **kwargs):
        calls.append(("oracle", build, kwargs))
        expected = "CPU" if inputs["full_launch"].backend == "cpu" else "ROCm0"
        assert kwargs["backend"] == expected
        assert kwargs["resolved_recipe"].backend == inputs["full_launch"].backend
        return gates.Verdict("op_correctness", True, "fixture oracle")
    values = iter(rates)
    def measure(recipe, build, port, **kwargs):
        calls.append(("measure", build, kwargs))
        assert kwargs["frozen_requests"] == inputs["frozen_requests"]
        if failure == "invalid":
            raise MeasurementInvalid("fixture settled invalid arm", {"original": True})
        if failure == "cleanup":
            raise RunAborted("fixture unresolved server cleanup")
        kwargs["evidence"].append({"backend": inputs["full_launch"].backend,
            "status": "not_applicable" if inputs["full_launch"].backend == "cpu" else "proven",
            "window_start": 1.0, "window_end": 2.0})
        return next(values)
    monkeypatch.setattr(gates, "compiles", compile)
    monkeypatch.setattr(gates, "op_correctness", oracle)
    monkeypatch.setattr(serving, "_measure_once", measure)
    return calls


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_detached_omission_rebaseline_native_archive_and_export(tmp_path, monkeypatch, backend):
    cpu, inputs = _inputs(tmp_path, backend)
    calls = _hardware_edges(monkeypatch, inputs)
    exports = []
    before = os.sched_getaffinity(0)
    with _held(tmp_path / "cpu.lock", cpu) as held_cpu, _held(
            tmp_path / "gpu.lock", cpu, claim.DEVICE_ID) as held_gpu:
        result = source_loo.execute_surface(**inputs, held_cpu=held_cpu, held_gpu=held_gpu,
                                            on_serving_export=exports.append)
        assert held_cpu.observe()["status"] == held_gpu.observe()["status"] == "held"
    assert os.sched_getaffinity(0) == before
    assert result["loo"][0]["disposition"] == "supports_keep"
    assert result["rebaseline"]["disposition"] == "passed"
    assert len(exports) == 2
    for reference, receipt_path in zip([*result["loo"], result["rebaseline"]], exports):
        raw = Path(reference["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
        retained = json.loads(raw)
        assert retained["deletion_authorized"] is False
        receipt = json.loads(receipt_path.read_bytes())
        native_path = receipt_path.parent / receipt["native_reference"]["path"]
        native = json.loads(native_path.read_bytes())
        assert native["comparison"] == retained["comparison"]
        capture = native["comparison"]["belief_capture"]
        assert serving_beliefs.finish(native["comparison"], capture["inputs"]) == capture
    assert [call[0] for call in calls] == ["build", "oracle", "measure", "measure",
                                          "oracle", "measure", "measure"]
    assert _git(inputs["repo"], "rev-parse", "HEAD") == inputs["assembled_commit"]
    assert _git(inputs["repo"], "status", "--porcelain") == ""
    assert (inputs["repo"] / "kernel.c").read_text() == "int keep = 1;\n"
    assert {row["status"] for row in archive.recall(inputs["store_root"], epoch=inputs["epoch"])} == {
        "measured_loo", "measured_rebaseline"}


@pytest.mark.parametrize("candidate,expected", [(110.0, "supports_removal"), (100.5, "neutral")])
def test_original_serving_dispositions(tmp_path, monkeypatch, candidate, expected):
    cpu, inputs = _inputs(tmp_path)
    inputs["baseline"] = None
    _hardware_edges(monkeypatch, inputs, rates=(100.0, candidate))
    with _held(tmp_path / "cpu.lock", cpu) as owner:
        result = source_loo.execute_surface(**inputs, held_cpu=owner)
    assert result["loo"][0]["disposition"] == expected


@pytest.mark.parametrize("failure", ["build", "invalid", "cleanup", "stop"])
def test_failure_never_becomes_neutral_or_launches_successor(tmp_path, monkeypatch, failure):
    cpu, inputs = _inputs(tmp_path)
    inputs["baseline"] = None
    calls = _hardware_edges(monkeypatch, inputs, failure=failure)
    if failure == "stop":
        inputs["should_stop"] = lambda: True
    with _held(tmp_path / "cpu.lock", cpu) as owner:
        if failure in {"cleanup", "stop"}:
            with pytest.raises(RunAborted):
                source_loo.execute_surface(**inputs, held_cpu=owner)
        else:
            result = source_loo.execute_surface(**inputs, held_cpu=owner)
            assert result["loo"][0]["disposition"] == "inconclusive"
    if failure == "stop":
        assert calls == []
    if failure == "build":
        assert [call[0] for call in calls] == ["build"]


def test_foreign_or_expired_claim_is_not_reacquired(tmp_path, monkeypatch):
    cpu, inputs = _inputs(tmp_path)
    calls = _hardware_edges(monkeypatch, inputs)
    with _held(tmp_path / "cpu.lock", cpu) as owner:
        pass
    with pytest.raises(claim.ClaimRefused):
        source_loo.execute_surface(**inputs, held_cpu=owner)
    assert not calls and not inputs["directory"].exists()
