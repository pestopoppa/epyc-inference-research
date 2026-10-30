"""Actual temporary Git/build closure checks; no native workloads or claims."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from . import historical_instrument as hi
from ..execution import t0_provider as tp


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()


def seal(body):
    result = copy.deepcopy(body)
    result.pop("binding_sha256", None)
    result["binding_sha256"] = hi.schemas.content_hash(result)
    return result


@pytest.fixture
def material(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "experimental/historical-test")
    git(source, "config", "user.name", "fixture")
    git(source, "config", "user.email", "fixture@example.invalid")
    (source / "source.txt").write_text("original source")
    git(source, "add", "source.txt")
    git(source, "commit", "-m", "fixture only")
    commit = git(source, "rev-parse", "HEAD")
    monkeypatch.setattr(hi.production_pins, "MEASUREMENT_COMMIT", commit)
    build = tmp_path / "build-instrument"
    (build / "bin").mkdir(parents=True)
    compiler = tmp_path / "compiler"
    compiler.write_text("#!/bin/sh\necho 'fixture compiler 1'\n")
    compiler.chmod(0o755)
    cache = build / "CMakeCache.txt"
    cache.write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(source) + "\nCMAKE_CXX_COMPILER:FILEPATH=" + str(compiler) + "\n")
    for name in ["llama-bench", "llama-completion", "test-backend-ops", "libggml.so"]:
        (build / "bin" / name).write_text("non-executable fixture:" + name)
    manifest = seal({"schema": hi.SCHEMA,
        "source": {"root": str(source), "commit": commit,
            "branch": "experimental/historical-test", "tree_oid": git(source, "rev-parse", "HEAD^{tree}"),
            "ls_tree_sha256": hashlib.sha256(hi._git(source, "ls-tree", "-r", "-z", commit)).hexdigest()},
        "build": {"root": str(build), "compiler": {"path": str(compiler),
            "sha256": hi._sha(compiler), "version": "fixture compiler 1"},
            "files": {str(p.relative_to(build)): hi._sha(p) for p in [cache, *(build / "bin").iterdir()]}}})
    return source, build, manifest


def test_actual_private_source_and_external_build_closure(material):
    _source, _build, manifest = material
    binding = hi.resolve(manifest)
    assert binding.validate_live() is binding
    assert binding.to_dict() == manifest


@pytest.mark.parametrize("change", ["tracked", "untracked", "branch", "head"])
def test_actual_git_source_drift_refuses_before_native(material, change):
    source, _build, manifest = material
    if change in ("tracked", "head"):
        (source / "source.txt").write_text("changed")
        if change == "head":
            git(source, "add", "source.txt")
            git(source, "commit", "-m", "changed fixture")
    elif change == "untracked":
        (source / "extra.txt").write_text("untracked")
    else:
        git(source, "checkout", "-b", "experimental/other")
    with pytest.raises(hi.InstrumentBindingRefused, match="Git source"):
        hi.resolve(manifest).validate_live()


@pytest.mark.parametrize("change", ["library", "extra", "cache", "home"])
def test_actual_build_library_and_cmake_drift_refuses(material, change):
    _source, build, manifest = material
    if change == "library":
        (build / "bin/libggml.so").write_text("changed library")
    elif change == "extra":
        (build / "bin/libforeign.so").write_text("extra library")
    else:
        (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/other/source\n")
        if change == "home":
            # A copied/relabelled cache cannot pass merely by resealing its bytes.
            manifest["build"]["files"]["CMakeCache.txt"] = hi._sha(build / "CMakeCache.txt")
            manifest = seal(manifest)
    with pytest.raises(hi.InstrumentBindingRefused):
        hi.resolve(manifest).validate_live()


@pytest.mark.parametrize("symlink", [False, True])
def test_original_frozen_tree_guard_still_runs_first(material, tmp_path, monkeypatch, symlink):
    _source, _build, manifest = material
    frozen = Path(hi.production_pins.PRODUCTION_REPO)
    if symlink:
        alias = tmp_path / "frozen-alias"
        alias.symlink_to(frozen, target_is_directory=True)
        frozen = alias
    manifest["source"]["root"] = str(frozen)
    monkeypatch.setattr(hi, "_git", lambda *args: pytest.fail("frozen guard must precede Git/native work"))
    with pytest.raises(tp.ProductionTreeRefusal):
        hi.resolve(seal(manifest)).validate_live()


def test_build_alias_is_not_private_materialization(material, tmp_path):
    _source, build, manifest = material
    alias = tmp_path / "build-alias"
    alias.symlink_to(build, target_is_directory=True)
    manifest["build"]["root"] = str(alias)
    with pytest.raises(hi.InstrumentBindingRefused, match="alias"):
        hi.resolve(seal(manifest)).validate_live()


def test_replay_identity_load_does_not_probe_or_rebuild(material, monkeypatch):
    _source, _build, manifest = material
    monkeypatch.setattr(hi, "_git", lambda *args: pytest.fail("replay identity must not probe Git"))
    binding = hi.resolve(manifest)
    assert hi.resolve(binding).to_dict() == manifest
    changed = copy.deepcopy(manifest)
    changed["source"]["commit"] = "0" * 40
    forged = hi.InstrumentBinding(json.dumps(seal(changed)))
    with pytest.raises(hi.InstrumentBindingRefused, match="original exact commit"):
        hi.resolve(forged)


def test_mutable_manifest_cannot_relabel_sealed_binding(material):
    _source, _build, manifest = material
    binding = hi.resolve(manifest)
    manifest["build"]["compiler"]["version"] = "later compiler"
    assert binding.to_dict()["build"]["compiler"]["version"] == "fixture compiler 1"
    with pytest.raises(hi.InstrumentBindingRefused, match="digest"):
        hi.resolve(manifest)


@pytest.mark.parametrize("change", ["digest", "version", "cache_compiler"])
def test_rehashed_compiler_descriptor_cannot_relabel_live_toolchain(material, change):
    source, build, manifest = material
    if change == "digest":
        manifest["build"]["compiler"]["sha256"] = "0" * 64
    elif change == "version":
        manifest["build"]["compiler"]["version"] = "a different compiler"
    else:
        cache = build / "CMakeCache.txt"
        cache.write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(source) + "\nCMAKE_CXX_COMPILER:FILEPATH=/other/compiler\n")
        manifest["build"]["files"]["CMakeCache.txt"] = hi._sha(cache)
    with pytest.raises(hi.InstrumentBindingRefused, match="compiler/cache"):
        hi.resolve(seal(manifest)).validate_live()
