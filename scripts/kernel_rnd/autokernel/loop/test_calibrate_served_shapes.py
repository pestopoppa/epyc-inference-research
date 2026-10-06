"""calibrate_served_shapes: dry by default, region-locked execute, fail-closed parsing,
manifest/patch staging, loop-alive and frozen-tree refusals -- with a stubbed tool."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import textwrap

import pytest

from . import calibrate_served_shapes as cal
from . import served_shape_cases as ssc
from . import status


def _fake_build(tmp: Path, *, calibrated=True, values=None, skip=0, rc=0) -> Path:
    """A build whose bin/test-backend-ops is a python stub printing calibration lines."""
    build = tmp / "calib-build"
    (build / "bin").mkdir(parents=True)
    lines = [f"{ssc.CALIBRATION_MARKER}\t{ssc.calibration_vars(*t)}\t"
             f"{(values or {}).get(i, 1e-6)}"
             for i, t in enumerate(ssc.calibration_triples())][skip:]
    payload = "\n".join(lines)
    literal = (f"# {ssc.CALIBRATION_CASE_SET_ID} {ssc.CALIBRATION_MARKER} "
               f"{ssc.BACKEND_THREADS_ENV}" if calibrated else "# nothing")
    tool = build / "bin" / "test-backend-ops"
    check = (f'assert os.environ["AUTOKERNEL_CORRECTNESS_CASE_SET"] == '
             f'"{ssc.CALIBRATION_CASE_SET_ID}" and os.environ["AUTOKERNEL_BACKEND_THREADS"] '
             f'== "48" and os.environ["GGML_IQK"] == "1" and os.environ["LD_LIBRARY_PATH"]'
             f'.startswith("{build}/bin")') if calibrated else "pass"
    if not calibrated:
        payload = payload.replace(ssc.CALIBRATION_MARKER, "XX")
    tool.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        {literal}
        import os, sys
        {check}
        print({payload!r})
        sys.exit({rc})
        """))
    tool.chmod(0o755)
    (build / "provenance.json").write_text(json.dumps({"champion_commit": "c" * 40}))
    return build


def _fake_region_lock(tmp: Path) -> Path:
    """Records its argv and runs the command after `--`."""
    script = tmp / "region-lock"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        echo "$@" > {tmp}/region-lock.argv
        while [ "$1" != "--" ]; do shift; done; shift
        shift 3   # drop taskset -c <list>
        exec "$@"
        """))
    script.chmod(0o755)
    return script


def _tree(tmp: Path) -> Path:
    tree = tmp / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests" / "test-backend-ops.cpp").write_text(
        "int x;\n"
        "static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {\n"
        "    std::vector<std::unique_ptr<test_case>> test_cases;\n"
        "    return test_cases;\n}\n"
        "int main() {\n"
        "            ggml_backend_set_n_threads_fn(backend.get(), N_THREADS);\n}\n")
    subprocess.run(["git", "init", "-q", "-b", "experimental/x", str(tree)], check=True)
    return tree


def _launch(tmp: Path) -> Path:
    """A resolved served launch record like inputs-*/<target>.launch.json."""
    served = tmp / "served-build"
    (served / "bin").mkdir(parents=True, exist_ok=True)
    path = tmp / "target.launch.json"
    path.write_text(json.dumps({
        "launch_env": {"GGML_IQK": "1", "OMP_PROC_BIND": "close",
                       "LD_LIBRARY_PATH": f"{served}/bin"},
        "topology_prefix": ["taskset", "-c", "0-95"],
        "command_argv": ["llama-server", "-m", "m.gguf", "-t", "48", "-tb", "48"],
        "build_dir": str(served), "template": {"cpu_list": "0-95"}}))
    return path


def _run(*argv):
    out = io.StringIO()
    rc = cal.main(list(map(str, argv)), out=out)
    return rc, out.getvalue()


def test_dry_by_default_changes_nothing(tmp_path):
    build = _fake_build(tmp_path)
    store = tmp_path / "store"
    rc, out = _run("--store", store, "--anchor-build", build)
    assert rc == 0 and "DRY RUN" in out and "carries" in out
    assert not (store / "served_shape").exists()


def test_execute_takes_the_region_lock_and_records_provenance(tmp_path):
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute")
    assert rc == 0, out
    argv = (tmp_path / "region-lock.argv").read_text()
    assert argv.startswith("run --cpu-list 0-95 --role bench --")
    record = json.loads(next((store / "served_shape").glob("calibration-*.json")).read_text())
    assert len(record["measurements"]) == len(ssc.calibration_triples())
    prov = record["provenance"]
    assert prov["anchor_commit"] == "c" * 40 and prov["cpu_list"] == "0-95"
    assert prov["threads"] == 48 and "test-backend-ops" in prov["binary_digests"]
    assert prov["served_env"]["GGML_IQK"] == "1" and prov["launch_sha256"]
    assert prov["served_env"]["AUTOKERNEL_BACKEND_THREADS"] == "48"


def test_execute_then_apply_writes_manifest_and_stages_the_final_block(tmp_path):
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    tree = _tree(tmp_path)
    rc, _ = _run("--store", store, "--tree", tree, "--stage-calibration-patch")
    assert rc == 0
    staged = (tree / "tests" / "test-backend-ops.cpp").read_text()
    assert ssc.CALIBRATION_CASE_SET_ID in staged and staged.count(ssc.PATCH_CALL) == 1
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--apply", "--tree", tree)
    assert rc == 0, out
    cases = ssc.load_manifest(store / "served_shape" / "manifest.json")
    assert len(cases) == len(ssc.canonical_triples())
    final = (tree / "tests" / "test-backend-ops.cpp").read_text()
    assert ssc.CASE_SET_ID in final and ssc.CALIBRATION_CASE_SET_ID not in final
    assert final.count(ssc.PATCH_BEGIN) == 1 and final.count(ssc.PATCH_CALL) == 1
    assert "commit it on the experimental champion branch" in out


def test_incomplete_or_failed_calibration_refuses(tmp_path):
    lock = _fake_region_lock(tmp_path)
    for kwargs in ({"skip": 1}, {"rc": 3}, {"calibrated": False}, {"values": {0: "nan"}}):
        sub = tmp_path / str(len(list(tmp_path.iterdir())))
        sub.mkdir()
        build = _fake_build(sub, **kwargs)
        with pytest.raises((cal.Refused, ValueError)):
            recipe = cal.served_recipe(_launch(sub), build, cpu_list=None, threads=None)
            cal.execute(build, sub / "store", recipe, str(lock), out=io.StringIO())
        assert not (sub / "store" / "served_shape" / "manifest.json").exists()


def test_an_anchor_at_or_above_the_cap_refuses_to_bake(tmp_path):
    lock = _fake_region_lock(tmp_path)
    build = _fake_build(tmp_path, values={5: ssc.SERVED_SHAPE_NMSE_CAP})
    rc, _ = _run("--store", tmp_path / "s", "--anchor-build", build,
                 "--launch", _launch(tmp_path), "--region-lock", lock, "--execute", "--apply")
    assert rc == 2
    assert not (tmp_path / "s" / "served_shape" / "manifest.json").exists()


def test_refuses_while_the_loop_owning_the_store_is_alive(tmp_path):
    store = tmp_path / "s"
    status.write_json(store, status.STATUS_FILENAME, {
        "state": "running", "generated_at": status.datetime.now(
            status.timezone.utc).isoformat(), "stale_after_s": 180})
    rc, _ = _run("--store", store, "--tree", _tree(tmp_path), "--stage-calibration-patch")
    assert rc == 2


def test_refuses_the_frozen_production_tree(tmp_path, monkeypatch):
    tree = _tree(tmp_path)
    monkeypatch.setattr(cal, "FROZEN_TREE", str(tree))
    rc, _ = _run("--store", tmp_path / "s", "--tree", tree, "--stage-calibration-patch")
    assert rc == 2
    subprocess.run(["git", "-C", str(tmp_path / "t2"), "init", "-q"], capture_output=True)
    other = tmp_path / "prod"
    (other / "tests").mkdir(parents=True)
    (other / "tests" / "test-backend-ops.cpp").write_text("x")
    subprocess.run(["git", "init", "-q", "-b", "production-consolidated-v10", str(other)],
                   check=True)
    assert cal.frozen_tree_refusal(other) is not None


def test_the_test_expert_count_is_eight_with_real_per_expert_dims():
    experts = [s for s in ssc.SERVED_SHAPES if s.op == "MUL_MAT_ID"]
    assert experts and all(s.n_mats == 8 and s.n_used <= 8 for s in experts)
    assert {(s.k, s.m) for s in experts} == {(5120, 2304), (2304, 5120)}


def test_build_calibration_uses_the_anchor_recipe_under_the_build_lock(tmp_path):
    tree = _tree(tmp_path)
    store = tmp_path / "s"
    rc, _ = _run("--store", store, "--tree", tree, "--build-calibration", "--cpu-list", "0-95")
    assert rc == 2   # no calibration block staged yet
    assert _run("--store", store, "--tree", tree, "--stage-calibration-patch")[0] == 0
    lock = tmp_path / "region-lock"
    lock.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        echo "$@" > {tmp_path}/build.argv
        mkdir -p {tree}/build-ak-calib/bin
        printf '{ssc.CALIBRATION_CASE_SET_ID} {ssc.CALIBRATION_MARKER} {ssc.BACKEND_THREADS_ENV}' \\
            > {tree}/build-ak-calib/bin/test-backend-ops
        """))
    lock.chmod(0o755)
    rc, out = _run("--store", store, "--tree", tree, "--build-calibration",
                   "--cpu-list", "0-95", "--region-lock", lock)
    assert rc == 0, out
    argv = (tmp_path / "build.argv").read_text()
    assert argv.startswith("run --cpu-list 0-95 --role build")
    for define in cal.ANCHOR_RECIPE_DEFINES:
        assert define in argv
    assert "--target test-backend-ops" in argv
    assert json.loads((tree / "build-ak-calib" / "provenance.json").read_text())["recipe"]


def test_build_calibration_refuses_other_tree_changes_and_a_live_loop(tmp_path):
    tree = _tree(tmp_path)
    store = tmp_path / "s"
    (tree / "other.c").write_text("x")
    subprocess.run(["git", "-C", str(tree), "add", "other.c"], check=True)
    assert _run("--store", store, "--tree", tree, "--stage-calibration-patch")[0] == 0
    rc, _ = _run("--store", store, "--tree", tree, "--build-calibration", "--cpu-list", "0-95")
    assert rc == 2
    status.write_json(store, status.STATUS_FILENAME, {
        "state": "running", "generated_at": status.datetime.now(
            status.timezone.utc).isoformat(), "stale_after_s": 180})
    rc, _ = _run("--store", store, "--tree", tree, "--build-calibration", "--cpu-list", "0-95")
    assert rc == 2



def test_execute_reproduces_the_served_recipe_or_refuses(tmp_path):
    """Round-12: the served env (GGML_IQK, OMP_*, rebound LD_LIBRARY_PATH), prefix and -t
    are applied -- the stub asserts them -- and any mismatch refuses."""
    build, lock = _fake_build(tmp_path), _fake_region_lock(tmp_path)
    launch = _launch(tmp_path)
    recipe = cal.served_recipe(launch, build, cpu_list="0-95", threads=48)
    assert recipe["env"]["GGML_IQK"] == "1" and recipe["threads"] == 48
    assert recipe["env"]["LD_LIBRARY_PATH"] == f"{build}/bin"
    assert recipe["env"]["AUTOKERNEL_BACKEND_THREADS"] == "48"
    with pytest.raises(cal.Refused):
        cal.served_recipe(launch, build, cpu_list=None, threads=24)
    with pytest.raises(cal.Refused):
        cal.served_recipe(launch, build, cpu_list="0-47", threads=None)
    body = json.loads(launch.read_text())
    body["launch_env"]["LD_PRELOAD"] = "/x.so"
    launch.write_text(json.dumps(body))
    with pytest.raises(cal.Refused):
        cal.served_recipe(launch, build, cpu_list=None, threads=None)
    rc, _ = _run("--store", tmp_path / "s", "--anchor-build", build, "--region-lock", lock,
                 "--execute")
    assert rc == 2   # no --launch: the served env cannot be reproduced
