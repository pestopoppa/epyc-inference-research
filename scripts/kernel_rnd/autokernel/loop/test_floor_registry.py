"""Execution-equivalence floor registry: key, era rules, grade, claim, and the AK flag gate.

WRITTEN UNDER A HOST CONSTRAINT (code + py_compile only on 2026-10-21): NOT YET RUN. Pure stdlib
plus a throwaway git repo; no server, no GPU, no CPU region. Run:
    cd scripts/kernel_rnd && python3 -m pytest autokernel/loop/test_floor_registry.py -q
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autokernel.loop import floor_registry as fr

BRANCH = "ak/champion/test"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(repo, files: dict, msg):
    for name, body in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "llama"
    r.mkdir()
    git(r, "init", "-q", "-b", BRANCH)
    commit(r, {"ggml/a.c": "A", "CMakeLists.txt": "cmake", "tests/t.cpp": "T", "docs/x.md": "D"}, "base")
    return r


def build(tmp_path, repo, name, sha, flags="-O3"):
    b = tmp_path / name
    b.mkdir()
    (b / "VERSION.txt").write_text(f"version: 1 ({sha[:9]})\nbuilt with GNU 15.2.0 for Linux x86_64\n")
    (b / "SHA256SUMS").write_text(f"{'a' * 64}  ./libggml-cpu.so.0\n")
    (b / "PROVENANCE.md").write_text("x")
    (b / "CMakeCache.txt").write_text(
        f"CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CXX_FLAGS_RELEASE:STRING={flags}\n"
        "GGML_NATIVE:BOOL=ON\nLLAMA_BUILD_TESTS:BOOL=ON\nCMAKE_HOME_DIRECTORY:INTERNAL=/x\n")
    return b


def floor(*, recipe_hash="r" * 64, request="q" * 64, model="m" * 64, threads=48, pairs=5):
    frame = {"recipe_hash": recipe_hash, "request_digest": request, "backend": "cpu",
             "model": {"sha256": model, "path": "/m"}, "drafter": None, "environment": {},
             "topology_prefix": [], "applicability": "same_workload_source_build_treatment_only"}
    row = {"schema": fr.MATCHED_SCHEMA, "unit": "process", "n": 24, "recipe": "t", "recipe_hash": recipe_hash,
           "request_digest": request, "floor_pct": 1.5, "comparison_pairs": pairs, "frame": frame,
           "baseline_resolved_recipe": {"execution_digest": "e" * 64,
                                        "argv": ["taskset", "-c", "0", "/x/llama-server", "-t", str(threads)],
                                        "dsos": [{"path": "/b/libggml-cpu.so.0", "sha256": "d" * 64}],
                                        "executable": {"sha256": "f" * 64}},
           "residency": {"window_end": 1000.0}}
    row["content_sha256"] = fr.digest({k: v for k, v in row.items() if k != "content_sha256"})
    return row


QUIET = {"threads": 48, "co_tenancy": "quiet", "claim_class": "decision"}


def lookup_kwargs(eq, champ, **over):
    kw = dict(equivalence_key=eq["equivalence_key"], recipe_hash="r" * 64, request_digest="q" * 64,
              model_sha="m" * 64, numerics_epoch="n1", pairs=5, champion=champ)
    kw.update(over)
    return kw


def test_tests_only_commit_is_same_equivalence_key(tmp_path, repo):
    c1 = commit(repo, {"x.txt": "x"}, "src")
    c2 = commit(repo, {"tests/t.cpp": "T2", "docs/y.md": "more", "tools/server/tests/z.py": "z"}, "tests only")
    e1 = fr.equivalence_for_build(build(tmp_path, repo, "b1", c1), source_repo=repo)
    e2 = fr.equivalence_for_build(build(tmp_path, repo, "b2", c2), source_repo=repo)
    assert e1["commit"] != e2["commit"]
    assert e1["equivalence_key"] == e2["equivalence_key"]


def test_source_flag_and_compiler_changes_change_the_key(tmp_path, repo):
    c1 = commit(repo, {"x.txt": "x"}, "src")
    base = fr.equivalence_for_build(build(tmp_path, repo, "b1", c1), source_repo=repo)["equivalence_key"]
    flagged = fr.equivalence_for_build(build(tmp_path, repo, "b2", c1, flags="-O2"), source_repo=repo)
    assert flagged["equivalence_key"] != base
    c2 = commit(repo, {"ggml/a.c": "A2"}, "real source change")
    assert fr.equivalence_for_build(build(tmp_path, repo, "b3", c2), source_repo=repo)["equivalence_key"] != base


def test_register_lookup_roundtrip_and_idempotence(tmp_path, repo):
    c = git(repo, "rev-parse", "HEAD")
    eq = fr.equivalence_for_build(build(tmp_path, repo, "b", c), source_repo=repo)
    reg = fr.FloorRegistry(tmp_path / "reg")
    p1 = reg.register(floor(), equivalence=eq, grade=QUIET, numerics_epoch="n1")
    assert reg.register(floor(), equivalence=eq, grade=QUIET, numerics_epoch="n1") == p1
    champ = fr.champion_state(repo, BRANCH)
    got = reg.lookup(**lookup_kwargs(eq, champ))
    assert got["floor_pct"] == 1.5 and got["content_sha256"] == floor()["content_sha256"]
    assert len((tmp_path / "reg" / "index.jsonl").read_text().splitlines()) == 1


def test_tests_only_advance_is_not_a_champion_advance_but_a_source_advance_is(tmp_path, repo):
    c1 = git(repo, "rev-parse", "HEAD")
    eq = fr.equivalence_for_build(build(tmp_path, repo, "b", c1), source_repo=repo)
    reg = fr.FloorRegistry(tmp_path / "reg")
    reg.register(floor(), equivalence=eq, grade=QUIET, numerics_epoch="n1")
    commit(repo, {"tests/t.cpp": "T9"}, "tests only")
    assert reg.lookup(**lookup_kwargs(eq, fr.champion_state(repo, BRANCH)))
    commit(repo, {"ggml/a.c": "A9"}, "real advance")
    res = reg.lookup_detail(**lookup_kwargs(eq, fr.champion_state(repo, BRANCH)))
    assert res.status == fr.STATUS_STALE_CHAMPION and res.floor is None
    rows = reg.status(fr.champion_state(repo, BRANCH))
    assert rows[0]["state"] == "STALE_CHAMPION_ADVANCED"


def test_numerics_change_and_unresolvable_champion_refuse(tmp_path, repo):
    eq = fr.equivalence_for_build(build(tmp_path, repo, "b", git(repo, "rev-parse", "HEAD")), source_repo=repo)
    reg = fr.FloorRegistry(tmp_path / "reg")
    reg.register(floor(), equivalence=eq, grade=QUIET, numerics_epoch="n1")
    champ = fr.champion_state(repo, BRANCH)
    assert reg.lookup_detail(**lookup_kwargs(eq, champ, numerics_epoch="n2")).status == fr.STATUS_STALE_NUMERICS
    assert reg.lookup_detail(**lookup_kwargs(eq, None)).status == fr.STATUS_UNRESOLVABLE


def test_cotenant_floor_is_indicative_and_never_satisfies_decision(tmp_path, repo):
    eq = fr.equivalence_for_build(build(tmp_path, repo, "b", git(repo, "rev-parse", "HEAD")), source_repo=repo)
    reg = fr.FloorRegistry(tmp_path / "reg")
    with pytest.raises(fr.FloorRegistryError):
        reg.register(floor(), equivalence=eq, numerics_epoch="n1",
                     grade={"threads": 48, "co_tenancy": "co-tenant", "claim_class": "decision"})
    reg.register(floor(), equivalence=eq, numerics_epoch="n1",
                 grade={"threads": 48, "co_tenancy": "co-tenant", "claim_class": "indicative"})
    champ = fr.champion_state(repo, BRANCH)
    assert reg.lookup_detail(**lookup_kwargs(eq, champ)).status == fr.STATUS_GRADE
    assert reg.lookup(**lookup_kwargs(eq, champ, require_grade="indicative")) is not None


def test_grade_threads_must_match_the_floors_own_launch(tmp_path, repo):
    eq = fr.equivalence_for_build(build(tmp_path, repo, "b", git(repo, "rev-parse", "HEAD")), source_repo=repo)
    with pytest.raises(fr.FloorRegistryError):
        fr.FloorRegistry(tmp_path / "reg").register(
            floor(threads=24), equivalence=eq, numerics_epoch="n1", grade=QUIET)


def test_tampered_floor_and_unsealed_v2_are_refused(tmp_path, repo):
    eq = fr.equivalence_for_build(build(tmp_path, repo, "b", git(repo, "rev-parse", "HEAD")), source_repo=repo)
    reg = fr.FloorRegistry(tmp_path / "reg")
    bad = floor()
    bad["floor_pct"] = 0.1  # seal no longer re-derives
    with pytest.raises(fr.FloorRegistryError):
        reg.register(bad, equivalence=eq, grade=QUIET, numerics_epoch="n1")
    path = reg.register(floor(), equivalence=eq, grade=QUIET, numerics_epoch="n1")
    doc = json.loads(path.read_text())
    doc["floor"]["floor_pct"] = 9.9
    path.write_text(json.dumps(doc))
    champ = fr.champion_state(repo, BRANCH)
    assert reg.lookup_detail(**lookup_kwargs(eq, champ)).status == fr.STATUS_CORRUPT


def test_check_launch_reuse_then_one_claim_per_condition(tmp_path, repo):
    eq = fr.equivalence_for_build(build(tmp_path, repo, "b", git(repo, "rev-parse", "HEAD")), source_repo=repo)
    reg = fr.FloorRegistry(tmp_path / "reg")
    champ = fr.champion_state(repo, BRANCH)
    kw = dict(equivalence=eq, recipe_hash="r" * 64, request_digest="q" * 64, model_sha="m" * 64, pairs=5,
              numerics_epoch="n1", require_grade="decision", champion=champ)
    first = fr.check_launch(reg, owner="one", **kw)
    second = fr.check_launch(reg, owner="two", **kw)
    assert (first["action"], second["action"]) == ("calibrate", "claimed")
    assert second["claim"]["owner"] == "one"
    reg.register(floor(), equivalence=eq, grade=QUIET, numerics_epoch="n1")  # releases the claim
    assert fr.check_launch(reg, owner="three", **kw)["action"] == "reuse"


def test_check_launch_refuses_to_calibrate_a_stale_anchor(tmp_path, repo):
    """The Q38FN case: the campaign anchors a pre-advance build; calibrating it files a floor
    that is stale at birth."""
    old = fr.equivalence_for_build(build(tmp_path, repo, "old", git(repo, "rev-parse", "HEAD")), source_repo=repo)
    commit(repo, {"ggml/a.c": "ADVANCED"}, "champion advance")
    out = fr.check_launch(fr.FloorRegistry(tmp_path / "reg"), equivalence=old, recipe_hash="r" * 64,
                          request_digest="q" * 64, model_sha="m" * 64, pairs=5, numerics_epoch="n1",
                          require_grade="decision", champion=fr.champion_state(repo, BRANCH), owner="o")
    assert out["action"] == "error" and out["status"] == fr.STATUS_STALE_CHAMPION


def test_is_store_build(tmp_path, repo):
    b = build(tmp_path, repo, "b", git(repo, "rev-parse", "HEAD"))
    assert fr.is_store_build(b)
    (b / "provenance.json").write_text("{}")
    assert not fr.is_store_build(b)


def test_ak_flag_is_off_by_default():
    """No behaviour change unless --floor-registry is passed: the module global stays None and
    `_select_source_floor` / `_write_new_source_floor` only branch on it."""
    from autokernel.loop import run
    assert run._FLOOR_REGISTRY is None
    parser_help = subprocess.run(["python3", "-m", "autokernel.loop.run", "--help"], capture_output=True,
                                 text=True, cwd=Path(__file__).resolve().parents[2]).stdout
    assert "--floor-registry" in parser_help
