"""Synthetic serialization/refusal controls; no model, C++ build or kernel execution."""
from __future__ import annotations

import importlib.util
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile

import pytest

from . import cpu_fa_reference as fa
from .cpu_fa_verifier_record import canonical, digest
from .test_cpu_fa_route import _builds, _fake_runner, _recipe, _write_capture_fixture


@lru_cache(maxsize=1)
def reader():
    root = Path(os.environ["EPYC_REALMASK_ROOT"]).resolve(strict=True)
    sys.path.insert(0, str(root / "scripts/vidya"))
    spec = importlib.util.spec_from_file_location("real_mask_reader_under_test",
        root / "scripts/vidya/adapters/autokernel_real_mask.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def private_native_test_root():
    # The strict reader already accepts root-owned /tmp with exact sticky 1777
    # permissions. A custom pytest basetemp may have writable RAID ancestors;
    # keep these synthetic archives and every mutated clone in private custody.
    with tempfile.TemporaryDirectory(prefix="epyc-real-mask-fixtures-", dir="/tmp") as directory:
        yield Path(directory)


@pytest.fixture
def tmp_path(private_native_test_root):
    with tempfile.TemporaryDirectory(prefix="case-", dir=private_native_test_root) as directory:
        yield Path(directory)


@pytest.fixture(scope="module")
def native(private_native_test_root):
    base = private_native_test_root / "prospective-real-mask-source-controls"
    base.mkdir(mode=0o700)
    builds, source = _builds(base)
    capture = base / "masks"
    cases = fa.ds41_real_mask_cases()
    _write_capture_fixture(capture, cases)
    calls = []
    def run(*args, **kwargs):
        return fa.check_anchor_identity(*args, **kwargs,
            runner=_fake_runner(builds, calls, cases=cases, capture_dir=capture))
    result = fa.check_real_mask_identity(builds["anchor"], builds["candidate"], source,
        capture_dir=capture, anchor_recipe=_recipe(), candidate_recipe=_recipe(),
        check_anchor_identity_fn=run)
    assert result.status == "pass", result
    return Path(json.loads(result.detail)["native_record"])


def test_original_native_roundtrip_uses_shared_verifier_ladder(native):
    adapter = reader()
    rows = adapter.native_rows(native)
    assert len(rows) == 1
    projected = adapter.project_real_mask(rows[0])
    assert projected.value is True and projected.reps == 3
    assert projected.claim.startswith("Synthetic source-conformance fixture only:")
    assert projected.source_class == "verifier" and projected.protocol_id == ""
    assert adapter.verify_and_grade(native)[0][:2] == ("Judged", "Located")
    assert len(rows[0]["record"]["observations"]) == 2 + 16 * 3 * 2
    assert len(rows[0]["request"]["cases"]) == 16


def test_native_archive_under_writable_ancestor_is_refused(native, tmp_path):
    unsafe = tmp_path / "writable-ancestor"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    path = clone(native, unsafe)
    with pytest.raises(ValueError, match="native custody ancestor is symlink/nonowned/writable"):
        reader().native_rows(path)


def clone(native, tmp_path):
    import shutil
    destination = tmp_path / native.parent.name
    shutil.copytree(native.parent, destination)
    destination.chmod(0o700)
    return destination / "record.json"


def reseal(path, record):
    record.pop("record_sha256", None)
    record["record_sha256"] = digest(canonical(record))
    path.write_bytes(canonical(record))


def test_resealed_forged_verdict_is_refused(native, tmp_path):
    path = clone(native, tmp_path)
    record = json.loads(path.read_bytes())
    record["value"], record["verdict"] = False, "wrong"
    reseal(path, record)
    with pytest.raises(ValueError, match="forged verdict"):
        reader().native_rows(path)


def test_native_mask_digest_change_is_refused(native, tmp_path):
    path = clone(native, tmp_path)
    mask = path.parent / "ds41_realmask_kv4k_nb2.mask.f16"
    data = mask.read_bytes()
    mask.write_bytes(b"\x01" + data[1:])
    with pytest.raises(ValueError, match="digest/size changed"):
        reader().native_rows(path)


@pytest.mark.parametrize("defect", ["missing_repeat", "wrong_mask", "forged_env"])
def test_resealed_native_probe_defects_are_refused(native, tmp_path, defect):
    path = clone(native, tmp_path)
    record = json.loads(path.read_bytes())
    observation = record["observations"][2]
    original = path.parent / observation["name"]
    row = json.loads(original.read_bytes())
    if defect == "missing_repeat":
        row["stdout"] = "\n".join(line for line in row["stdout"].splitlines()
                                    if not line.startswith("D 2 ")) + "\n"
        output = row["original_outputs"]["stdout"]
        data = row["stdout"].encode()
        (path.parent / output["name"]).write_bytes(data)
        output.update(sha256=digest(data), bytes=len(data))
        next(item for item in record["pins"] if item["name"] == output["name"]).update(output)
    elif defect == "wrong_mask":
        row["argv"][-1] = row["argv"][-1].replace("nb2", "nb3")
        launch = path.parent / "launch-0002.json"
        launch_row = json.loads(launch.read_bytes())
        launch_row["argv"] = row["argv"]
        launch.write_bytes(canonical(launch_row))
    else:
        row["env"]["GGML_FA_SPLIT_KV"] = "1"
        launch = path.parent / "launch-0002.json"
        launch_row = json.loads(launch.read_bytes())
        launch_row["env"] = row["env"]
        launch.write_bytes(canonical(launch_row))
    original.write_bytes(canonical(row))
    observation["sha256"] = digest(canonical(row))
    reseal(path, record)
    with pytest.raises(ValueError, match="missing original repetitions|argv/environment/exit/mask binding"):
        reader().native_rows(path)


def test_projection_reopens_native_record_instead_of_trusting_row(native):
    adapter = reader()
    row = adapter.native_rows(native)[0]
    row["record"]["value"] = False
    with pytest.raises(ValueError, match="changed since reopening"):
        adapter.project_real_mask(row)


def test_pre_hook_manifest_is_not_backfilled(tmp_path):
    builds, source = _builds(tmp_path)
    capture = tmp_path / "masks"
    _write_capture_fixture(capture, fa.ds41_real_mask_cases())
    path = capture / "capture-manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest.pop("decided_proposition")
    path.write_text(json.dumps(manifest))
    result = fa.check_real_mask_identity(builds["anchor"], builds["candidate"], source,
        capture_dir=capture, anchor_recipe=_recipe(), candidate_recipe=_recipe())
    assert result.status == "unavailable" and "predates the prospective verifier hook" in result.reason


@pytest.mark.parametrize("fault", ["spawn", "timeout", "runtime", "mask_mutation"])
def test_native_execution_fault_retains_null_originals(tmp_path, fault):
    builds, source = _builds(tmp_path)
    capture = tmp_path / "masks"
    cases = fa.ds41_real_mask_cases()
    _write_capture_fixture(capture, cases)
    ordinary = _fake_runner(builds, [], cases=cases, capture_dir=capture)
    def failed(argv, **kwargs):
        if fault == "mask_mutation" and argv[0] != "c++":
            done = ordinary(argv, **kwargs)
            mask = Path(argv[-1])
            data = mask.read_bytes()
            mask.write_bytes(b"\x01" + data[1:])
            return done
        if argv[0] == "c++":
            return ordinary(argv, **kwargs)
        if fault == "spawn":
            raise OSError("injected spawn failure")
        if fault == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"native partial output", stderr=b"native stderr")
        raise RuntimeError("injected unexpected runner fault")
    def run(*args, **kwargs):
        return fa.check_anchor_identity(*args, **kwargs, runner=failed)
    result = fa.check_real_mask_identity(builds["anchor"], builds["candidate"], source,
        capture_dir=capture, anchor_recipe=_recipe(), candidate_recipe=_recipe(),
        check_anchor_identity_fn=run)
    assert result.status == "unavailable", result
    path = Path(json.loads(result.detail)["native_record"])
    record = json.loads(path.read_bytes())
    assert record["value"] is None and record["verdict"] == "unavailable"
    terminal = json.loads((path.parent / record["observations"][-1]["name"]).read_bytes())
    assert terminal["sequence"] == 2
    if fault != "mask_mutation":
        assert terminal["returncode"] is None and terminal["error"] in ("OSError", "TimeoutExpired", "RuntimeError")
    else:
        assert terminal["mask_sha256_before"] != terminal["mask_sha256_after"]
    assert reader().native_rows(path) == ()
    if fault == "timeout":
        assert terminal["stdout"] == "native partial output" and terminal["stderr"] == "native stderr"


def test_incomplete_native_archive_cannot_grade(native, tmp_path):
    path = clone(native, tmp_path)
    (path.parent / "stdout-0002.bin").unlink()
    with pytest.raises(ValueError):
        reader().native_rows(path)
