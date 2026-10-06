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


def _fake_build(tmp: Path, *, calibrated=True, values=None, skip=0, rc=0,
                seeded=True, announce=True, fail_vars=None) -> Path:
    """A build whose bin/test-backend-ops is a python stub printing calibration lines,
    honoring its own `-p <regex>` argument (so a sharded, filtered invocation only sees
    its own assigned cases -- the same contract the real test-backend-ops carries).
    `fail_vars`, given, exits 9 with no output whenever the invocation's `-p` regex
    would select that one case's vars string -- i.e. exactly the shard carrying that
    case fails, the others succeed."""
    build = tmp / "calib-build"
    (build / "bin").mkdir(parents=True)
    rows = [(ssc.calibration_vars(*t), (values or {}).get(i, 1e-6))
            for i, t in enumerate(ssc.calibration_triples("q38fn"))][skip:]
    literal = (f"# {ssc.CALIBRATION_CASE_SET_ID} {ssc.CALIBRATION_MARKER} "
               f"{ssc.BACKEND_THREADS_ENV}" if calibrated else "# nothing")
    if seeded:
        literal += f" {ssc.SEED_MARKER}"
    announcement = (f"print({ssc.SEED_MARKER!r}, file=sys.stderr)" if announce
                    else "pass")
    tool = build / "bin" / "test-backend-ops"
    check = (f'assert os.environ["AUTOKERNEL_CORRECTNESS_CASE_SET"] == '
             f'"{ssc.CALIBRATION_CASE_SET_ID}" and os.environ["AUTOKERNEL_BACKEND_THREADS"] '
             f'== "48" and os.environ["GGML_IQK"] == "1" and os.environ["LD_LIBRARY_PATH"]'
             f'.startswith("{build}/bin")') if calibrated else "pass"
    marker = ssc.CALIBRATION_MARKER if calibrated else "XX"
    script = "\n".join([
        "#!/usr/bin/env python3",
        literal,
        "import os, re, sys",
        check,
        announcement,
        f"MARKER = {marker!r}",
        f"ROWS = {rows!r}",
        f"FAIL_VARS = {fail_vars!r}",
        "pattern = None",
        "if '-p' in sys.argv:",
        "    pattern = re.compile(sys.argv[sys.argv.index('-p') + 1])",
        "if FAIL_VARS is not None and pattern is not None and pattern.search(FAIL_VARS):",
        "    sys.exit(9)",
        "for vars_str, value in ROWS:",
        "    if pattern is not None and not pattern.search(vars_str):",
        "        continue",
        "    print(MARKER + '\\t' + vars_str + '\\t' + str(value))",
        f"sys.exit({rc})",
    ])
    tool.write_text(script + "\n")
    tool.chmod(0o755)
    (build / "provenance.json").write_text(json.dumps({"champion_commit": "c" * 40}))
    return build


def _fake_region_lock(tmp: Path) -> Path:
    """Records its argv (one APPENDED line per invocation -- concurrent shards each call
    this once) and runs the command after `--`."""
    script = tmp / "region-lock"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        echo "$@" >> {tmp}/region-lock.argv
        while [ "$1" != "--" ]; do shift; done; shift
        shift 3   # drop taskset -c <list>
        exec "$@"
        """))
    script.chmod(0o755)
    return script


def _region_lock_invocations(tmp: Path) -> list:
    """Every `region-lock` invocation recorded by `_fake_region_lock`, one per line."""
    path = tmp / "region-lock.argv"
    return path.read_text().splitlines() if path.is_file() else []


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
        "build_dir": str(served), "template": {"cpu_list": "0-95"},
        "model": {"path": cal.ssc.LANE_PROFILES["q38fn"].model}}))
    return path


def _run(*argv):
    out = io.StringIO()
    from unittest import mock
    with mock.patch.object(cal, "lane_profile_refusal", return_value=None):
        rc = cal.main(list(map(str, argv)), out=out)
    return rc, out.getvalue()


def test_dry_by_default_changes_nothing(tmp_path):
    build = _fake_build(tmp_path)
    store = tmp_path / "store"
    rc, out = _run("--store", store, "--anchor-build", build)
    assert rc == 0 and "DRY RUN" in out and "carries" in out
    assert not (store / "served_shape").exists()


def test_execute_takes_the_correctness_mode_build_lock_and_records_provenance(tmp_path):
    """2026-10-06: the region-lock claim is `--role build` (correctness, not timing),
    and with --shards 1 there is exactly one test-backend-ops invocation."""
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--lane", "q38fn", "--shards", "1")
    assert rc == 0, out
    invocations = _region_lock_invocations(tmp_path)
    assert len(invocations) == 1
    assert invocations[0].startswith("run --cpu-list 0-95 --role build --")
    record = json.loads(next((store / "served_shape").glob("calibration-*.json")).read_text())
    assert record["schema"] == "epyc.autokernel.served_shape_calibration.v2"
    assert len(record["measurements"]) == len(ssc.calibration_triples("q38fn"))
    prov = record["provenance"]
    assert prov["anchor_commit"] == "c" * 40 and prov["cpu_list"] == "0-95"
    assert prov["lock_cpu_list"] == "0-95" and prov["shards"] == 1
    assert prov["threads"] == 48 and "test-backend-ops" in prov["binary_digests"]
    assert prov["served_env"]["GGML_IQK"] == "1" and prov["launch_sha256"]
    assert prov["served_env"]["AUTOKERNEL_BACKEND_THREADS"] == "48"
    assignment = prov["shard_assignment"]
    assert set(assignment) == {"0"}
    assert len(assignment["0"]) == len(ssc.calibration_triples("q38fn"))


def test_execute_shards_the_corpus_across_concurrent_processes(tmp_path):
    """The default (no --shards) shards the corpus into min(DEFAULT_SHARDS, cases)
    disjoint groups, one test-backend-ops process per shard, merged into one record
    covering every case exactly once."""
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--lane", "q38fn")
    assert rc == 0, out
    triples = ssc.calibration_triples("q38fn")
    expected_shards = min(cal.DEFAULT_SHARDS, len(triples))
    invocations = _region_lock_invocations(tmp_path)
    assert len(invocations) == expected_shards
    assert all(line.startswith("run --cpu-list 0-95 --role build --") for line in invocations)
    record = json.loads(next((store / "served_shape").glob("calibration-*.json")).read_text())
    prov = record["provenance"]
    assert prov["shards"] == expected_shards
    assignment = prov["shard_assignment"]
    assert set(assignment) == {str(i) for i in range(expected_shards)}
    all_keys = [key for keys in assignment.values() for key in keys]
    assert len(all_keys) == len(set(all_keys)) == len(triples)   # every case exactly once
    assert len(record["measurements"]) == len(triples)


def test_an_explicit_shards_count_is_honored_and_clamped(tmp_path):
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--lane", "q38fn", "--shards", "4")
    assert rc == 0, out
    assert len(_region_lock_invocations(tmp_path)) == 4
    record = json.loads(next((store / "served_shape").glob("calibration-*.json")).read_text())
    assert record["provenance"]["shards"] == 4
    with pytest.raises(cal.Refused):
        cal.resolve_shard_count(0, 10)
    # clamped: more shards requested than cases
    assert cal.resolve_shard_count(1000, 10) == 10


def test_a_failing_shard_fails_the_whole_execute_and_records_nothing(tmp_path):
    """Exactly the shard carrying one particular case exits non-zero; the other 7
    shards succeed. The whole --execute still refuses and writes nothing."""
    third_vars = ssc.calibration_vars(*ssc.calibration_triples("q38fn")[2])
    build, lock = _fake_build(tmp_path, fail_vars=third_vars), _fake_region_lock(tmp_path)
    store = tmp_path / "s"
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--lane", "q38fn", "--shards", "8")
    assert rc == 2
    assert not (store / "served_shape").exists() or not list(
        (store / "served_shape").glob("calibration-*.json"))


def test_execute_then_apply_writes_manifest_and_stages_the_final_block(tmp_path):
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    tree = _tree(tmp_path)
    rc, _ = _run("--store", store, "--tree", tree, "--stage-calibration-patch", "--lane", "q38fn")
    assert rc == 0
    staged = (tree / "tests" / "test-backend-ops.cpp").read_text()
    assert ssc.CALIBRATION_CASE_SET_ID in staged and staged.count(ssc.PATCH_CALL) == 1
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--apply", "--tree", tree, "--lane", "q38fn")
    assert rc == 0, out
    cases = ssc.load_manifest(store / "served_shape" / "manifest.json")
    assert len(cases) == len(ssc.canonical_triples(lane="q38fn"))
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
            cal.execute(build, sub / "store", recipe, str(lock), "q38fn",
                        out=io.StringIO())
        assert not (sub / "store" / "served_shape" / "manifest.json").exists()


def test_an_anchor_at_or_above_the_cap_refuses_to_bake(tmp_path):
    lock = _fake_region_lock(tmp_path)
    build = _fake_build(tmp_path, values={5: ssc.SERVED_SHAPE_NMSE_CAP})
    rc, _ = _run("--store", tmp_path / "s", "--anchor-build", build,
                 "--launch", _launch(tmp_path), "--region-lock", lock, "--execute", "--apply",
                 "--lane", "q38fn")
    assert rc == 2
    assert not (tmp_path / "s" / "served_shape" / "manifest.json").exists()


def test_an_existing_record_applies_under_the_new_policy_and_anchor_relative_opt_in(tmp_path):
    """Operator 2026-10-06: a record measured before the policy change applies via
    --measurements (lane/launch/binary validated); cases whose anchor is >= the generic
    bound refuse by default and are held anchor-relative only when asked."""
    lock, launch, store = _fake_region_lock(tmp_path), _launch(tmp_path), tmp_path / "s"
    build = _fake_build(tmp_path, values={3: 2e-4, 5: 5.25e-4})
    assert _run("--store", store, "--anchor-build", build, "--launch", launch,
                "--region-lock", lock, "--execute", "--lane", "q38fn")[0] == 0
    record = next((store / "served_shape").glob("calibration-*.json"))
    base = ("--store", store, "--measurements", record, "--launch", launch, "--lane", "q38fn",
            "--region-lock", lock, "--apply")
    rc, out = _run(*base)
    assert rc == 2 and "anchor-relative" in out
    assert not (store / "served_shape" / "manifest.json").exists()
    rc, out = _run(*base, "--anchor-exceeds-generic", "anchor-relative")
    assert rc == 0, out
    cases = ssc.load_manifest(store / "served_shape" / "manifest.json")
    triples = ssc.calibration_triples("q38fn")
    by_key = {(c.shape.name, c.type_a, c.n): c.max_nmse for c in cases}
    over = (triples[5][0].name, triples[5][1], triples[5][2])
    tight = (triples[3][0].name, triples[3][1], triples[3][2])
    assert by_key[over] > ssc.SERVED_SHAPE_NMSE_CAP     # anchor-relative, recorded
    assert by_key[tight] == ssc.SERVED_SHAPE_NMSE_CAP   # min(3 x 2e-4, 5e-4)


def test_refuses_while_the_loop_owning_the_store_is_alive(tmp_path):
    store = tmp_path / "s"
    status.write_json(store, status.STATUS_FILENAME, {
        "state": "running", "generated_at": status.datetime.now(
            status.timezone.utc).isoformat(), "stale_after_s": 180})
    rc, _ = _run("--store", store, "--tree", _tree(tmp_path), "--stage-calibration-patch", "--lane", "q38fn")
    assert rc == 2


def test_refuses_the_frozen_production_tree(tmp_path, monkeypatch):
    tree = _tree(tmp_path)
    monkeypatch.setattr(cal, "FROZEN_TREE", str(tree))
    rc, _ = _run("--store", tmp_path / "s", "--tree", tree, "--stage-calibration-patch", "--lane", "q38fn")
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
    assert experts and all(s.n_mats in (8, 16) and s.n_used <= s.n_mats for s in experts)
    assert {(s.k, s.m) for s in experts} == {(5120, 2304), (2304, 5120), (2560, 640),
                                             (640, 2560)}


def test_shard_argv_uses_role_build_with_the_lock_cpu_list_and_served_threads(tmp_path):
    """Requirement 2/5: the correctness-mode lock's argv claims `--role build` with
    WHATEVER cpu list the recipe's lock_cpu_list carries (narrowed or not), while the
    thread count baked into the served env stays the served -t regardless."""
    build = tmp_path / "build"
    recipe = {"prefix": ["taskset", "-c", "0-47"], "lock_cpu_list": "0-47",
             "served_cpu_list": "0-95", "threads": 48,
             "env": {"AUTOKERNEL_BACKEND_THREADS": "48"}}
    argv = cal.shard_argv(build, recipe, "region-lock", "^(rx)$", shard_index=3,
                          timeout_s=120)
    assert argv[:2] == ["region-lock", "run"]
    assert argv[argv.index("--cpu-list") + 1] == "0-47"   # the LOCK's list, narrowed
    assert argv[argv.index("--role") + 1] == "build"      # correctness mode, not bench
    assert "ak-served-shape-calibration-shard3" in argv
    assert recipe["env"]["AUTOKERNEL_BACKEND_THREADS"] == "48"   # served -t, unaffected
    assert argv[-1] == "^(rx)$"


def test_shard_argvs_covers_every_case_with_disjoint_regexes(tmp_path):
    build = tmp_path / "build"
    recipe = {"prefix": [], "lock_cpu_list": "0-95", "served_cpu_list": "0-95",
             "threads": 48, "env": {}}
    triples = ssc.calibration_triples("q38fn")
    argvs = cal.shard_argvs(build, recipe, "region-lock", "q38fn", 7)
    assert len(argvs) == 7
    regexes = [a[-1] for a in argvs]
    assert len(set(regexes)) == 7   # disjoint shards never share a filter
    # every triple's vars string matches exactly one shard's regex
    import re
    for shape, type_a, n in triples:
        var_string = ssc.calibration_vars(shape, type_a, n)
        matches = [i for i, rx in enumerate(regexes) if re.search(rx, var_string)]
        assert len(matches) == 1, (shape, type_a, n, matches)
    hit_counts = [sum(1 for shape, type_a, n in triples
                      if re.search(rx, ssc.calibration_vars(shape, type_a, n)))
                 for rx in regexes]
    assert sum(hit_counts) == len(triples)


def test_build_calibration_uses_the_anchor_recipe_under_the_build_lock(tmp_path):
    tree = _tree(tmp_path)
    store = tmp_path / "s"
    rc, _ = _run("--store", store, "--tree", tree, "--build-calibration", "--cpu-list", "0-95")
    assert rc == 2   # no calibration block staged yet
    assert _run("--store", store, "--tree", tree, "--stage-calibration-patch", "--lane", "q38fn")[0] == 0
    lock = tmp_path / "region-lock"
    lock.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        echo "$@" > {tmp_path}/build.argv
        mkdir -p {tree}/build-ak-calib/bin
        printf '{ssc.CALIBRATION_CASE_SET_ID} {ssc.CALIBRATION_MARKER} {ssc.BACKEND_THREADS_ENV} {ssc.SEED_MARKER}' \\
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
    assert _run("--store", store, "--tree", tree, "--stage-calibration-patch", "--lane", "q38fn")[0] == 0
    rc, _ = _run("--store", store, "--tree", tree, "--build-calibration", "--cpu-list", "0-95")
    assert rc == 2
    status.write_json(store, status.STATUS_FILENAME, {
        "state": "running", "generated_at": status.datetime.now(
            status.timezone.utc).isoformat(), "stale_after_s": 180})
    rc, _ = _run("--store", store, "--tree", tree, "--build-calibration", "--cpu-list", "0-95")
    assert rc == 2



def test_execute_reproduces_the_served_recipe_or_refuses(tmp_path):
    """Round-12: the served env (GGML_IQK, OMP_*, rebound LD_LIBRARY_PATH), prefix and -t
    are applied -- the stub asserts them -- and a THREADS mismatch refuses.

    2026-10-06 correctness-mode lock: --cpu-list is now the LOCK's cpu list and may
    differ from the served topology (narrowed for correctness mode); only --threads
    must still equal the served -t."""
    build, lock = _fake_build(tmp_path), _fake_region_lock(tmp_path)
    launch = _launch(tmp_path)
    recipe = cal.served_recipe(launch, build, cpu_list="0-95", threads=48)
    assert recipe["env"]["GGML_IQK"] == "1" and recipe["threads"] == 48
    assert recipe["env"]["LD_LIBRARY_PATH"] == f"{build}/bin"
    assert recipe["env"]["AUTOKERNEL_BACKEND_THREADS"] == "48"
    assert recipe["served_cpu_list"] == "0-95" and recipe["lock_cpu_list"] == "0-95"
    with pytest.raises(cal.Refused):
        cal.served_recipe(launch, build, cpu_list=None, threads=24)
    # A narrowed lock cpu list no longer disagrees with the served topology: it is the
    # correctness-mode region-lock's OWN claim, decoupled from where the model serves.
    narrowed = cal.served_recipe(launch, build, cpu_list="0-47", threads=None)
    assert narrowed["lock_cpu_list"] == "0-47" and narrowed["served_cpu_list"] == "0-95"
    assert narrowed["threads"] == 48
    body = json.loads(launch.read_text())
    body["launch_env"]["LD_PRELOAD"] = "/x.so"
    launch.write_text(json.dumps(body))
    with pytest.raises(cal.Refused):
        cal.served_recipe(launch, build, cpu_list=None, threads=None)
    rc, _ = _run("--store", tmp_path / "s", "--anchor-build", build, "--region-lock", lock,
                 "--execute")
    assert rc == 2   # no --launch: the served env cannot be reproduced



def test_the_lane_is_required_and_checked_against_its_gguf(tmp_path):
    tree = _tree(tmp_path)
    out = io.StringIO()
    rc = cal.main(["--store", str(tmp_path / "s"), "--tree", str(tree),
                   "--stage-calibration-patch"], out=out)
    assert rc == 2   # no --lane
    launch = _launch(tmp_path)
    assert "not lane ds41" in cal.lane_profile_refusal(launch, "ds41")



def test_imported_measurements_must_match_the_intended_recipe(tmp_path):
    """Round-14: --apply --measurements validates lane, launch, env, cpu list, threads,
    topology/argv and the calibration binary before baking bounds."""
    from unittest import mock
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    launch = _launch(tmp_path)
    assert _run("--store", store, "--anchor-build", build, "--launch", launch,
                "--region-lock", lock, "--execute", "--lane", "q38fn")[0] == 0
    record = next((store / "served_shape").glob("calibration-*.json"))
    with mock.patch.object(cal, "lane_profile_refusal", return_value=None):
        assert cal.measurement_record_refusal(record, launch, "q38fn", str(lock)) is None
        assert "lane" in cal.measurement_record_refusal(record, launch, "ds41", str(lock))
        body = json.loads(launch.read_text())
        body["launch_env"]["GGML_IQK"] = "0"
        other = tmp_path / "other.launch.json"
        other.write_text(json.dumps(body))
        assert "served_env" in cal.measurement_record_refusal(record, other, "q38fn", str(lock))
        body["launch_env"]["GGML_IQK"] = "1"
        body["command_argv"][-3] = "24"   # -t 24
        other.write_text(json.dumps(body))
        why = cal.measurement_record_refusal(record, other, "q38fn", str(lock))
        assert why and "threads" in why
        (build / "bin" / "test-backend-ops").write_text("#!/bin/sh\nexit 0\n")
        assert "calibration binary" in cal.measurement_record_refusal(
            record, launch, "q38fn", str(lock))
    rc, _ = _run("--store", store, "--measurements", record, "--apply", "--lane", "q38fn")
    assert rc == 2   # no --launch
    rc, _ = _run("--store", store, "--measurements", record, "--launch", launch, "--apply",
                 "--lane", "q38fn", "--region-lock", lock)
    assert rc == 2   # binary changed above
    assert not (store / "served_shape" / "manifest.json").exists()


def test_timeout_s_reaches_subprocess_call(tmp_path):
    """--timeout-s is passed to region-lock and recorded in provenance."""
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--timeout-s", "7200", "--execute", "--lane", "q38fn")
    assert rc == 0, out
    argv = (tmp_path / "region-lock.argv").read_text()
    assert "--timeout-s 7200" in argv
    record = json.loads(next((store / "served_shape").glob("calibration-*.json")).read_text())
    prov = record["provenance"]
    assert prov["timeout_s"] == 7200


def test_v1_records_are_refused_for_baking(tmp_path):
    """2026-10-06: the sharded --execute moved the schema to v2 (shards/shard_assignment
    provenance, per-shard argv). A pre-sharding v1 record -- which also predates
    per-case input seeding -- is refused by schema, not silently misread as v2."""
    record = tmp_path / "calibration-old.json"
    record.write_text(json.dumps({
        "schema": "epyc.autokernel.served_shape_calibration.v1", "lane": "q38fn",
        "provenance": {"anchor_build": str(tmp_path)}, "measurements": []}))
    why = cal.measurement_record_refusal(record, _launch(tmp_path), "q38fn", "region-lock")
    assert why and "v1" in why and "pre-sharding" in why
    real = Path("/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/"
                "store-b0ba1d427/served_shape/calibration-20261006T104207Z.json")
    if real.is_file():
        why = cal.measurement_record_refusal(real, _launch(tmp_path), "ds41", "region-lock")
        assert why and ("v1" in why or "seed scheme" in why)


def test_a_binary_built_before_seeding_refuses_everywhere(tmp_path, capsys):
    """Round-17: a stale build-ak-calib (built before per-case seeding) would draw random
    inputs and, without this check, get the new SEED_SCHEME label -- every path refuses."""
    lock, store = _fake_region_lock(tmp_path), tmp_path / "s"
    stale = _fake_build(tmp_path, seeded=False, announce=False)
    assert ssc.binary_has_calibration(stale) is False
    rc, out = _run("--store", store, "--anchor-build", stale, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--lane", "q38fn")
    err = capsys.readouterr().err
    assert rc == 2 and "--build-calibration" in err and "before deterministic" in err
    assert not list((store / "served_shape").glob("calibration-*.json")) \
        if (store / "served_shape").exists() else True
    with pytest.raises(cal.Refused, match="--build-calibration"):
        cal.execute(stale, store, {"cpu_list": "0-95", "threads": 48, "env": {}},
                    str(lock), "q38fn")


def test_a_run_that_does_not_announce_the_seed_scheme_refuses(tmp_path, capsys):
    lock, store = _fake_region_lock(tmp_path), tmp_path / "s"
    build = _fake_build(tmp_path, announce=False)
    rc, out = _run("--store", store, "--anchor-build", build, "--launch", _launch(tmp_path),
                   "--region-lock", lock, "--execute", "--lane", "q38fn")
    assert rc == 2 and f"did not announce {ssc.SEED_MARKER}" in capsys.readouterr().err
    assert not (store / "served_shape").exists() or not list(
        (store / "served_shape").glob("calibration-*.json"))


def test_a_record_whose_binary_predates_seeding_refuses_to_bake(tmp_path):
    build, lock, store = _fake_build(tmp_path), _fake_region_lock(tmp_path), tmp_path / "s"
    launch = _launch(tmp_path)
    assert _run("--store", store, "--anchor-build", build, "--launch", launch,
                "--region-lock", lock, "--execute", "--lane", "q38fn")[0] == 0
    record = next((store / "served_shape").glob("calibration-*.json"))
    body = json.loads(record.read_text())
    tool = build / "bin" / "test-backend-ops"
    tool.write_text(tool.read_text().replace(ssc.SEED_MARKER, "AK_NO_SEED"))
    body["provenance"]["binary_digests"]["test-backend-ops"] = cal._sha256(tool)
    record.write_text(json.dumps(body))
    from unittest import mock
    with mock.patch.object(cal, "lane_profile_refusal", return_value=None):
        why = cal.measurement_record_refusal(record, launch, "q38fn", str(lock))
    assert why and "--build-calibration" in why
