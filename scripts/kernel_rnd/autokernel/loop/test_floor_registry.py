"""Floor registry over the MASTER model registry: key, era rules, grade, claims, applier, validator.

Pure stdlib + yaml + a throwaway git repo; no server, no GPU. Run (CPU build claim):
    cd scripts/kernel_rnd && python3 -m pytest autokernel/loop/test_floor_registry.py -q
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autokernel.loop import apply_registry_serving_floors as ap
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
           "request_digest": request, "floor_pct": 1.5,
           "interval": {"level": 0.95, "low_pct": 1.0, "high_pct": 3.0}, "comparison_pairs": pairs, "frame": frame,
           "baseline_resolved_recipe": {"execution_digest": "e" * 64,
                                        "argv": ["taskset", "-c", "0", "/x/llama-server", "-t", str(threads)],
                                        "dsos": [{"path": "/b/libggml-cpu.so.0", "sha256": "d" * 64}],
                                        "executable": {"sha256": "f" * 64}},
           "residency": {"window_end": 1000.0}}
    row["content_sha256"] = fr.digest({k: v for k, v in row.items() if k != "content_sha256"})
    return row


QUIET = {"threads": 48, "co_tenancy": "quiet", "claim_class": "decision"}
ROLE = "ds41_test_role"
MASTER_TEXT = """# Master registry (test copy): the comments below must survive every edit.
schema_version: "1"

roles:
  ds41_test_role:
    tier: test   # inline comment
    model:
      name: x

# trailing comment block
observations:
  a: 1
"""


def master_in(tmp_path):
    root = tmp_path / "researchrepo"
    (root / "orchestration").mkdir(parents=True)
    path = root / "orchestration" / "model_registry.yaml"
    path.write_text(MASTER_TEXT)
    return path


def register(tmp_path, master, repo_build_eq, *, flr=None, grade=QUIET, epoch="n1", role=ROLE, note="n"):
    """Calibrated floor -> plan -> applier, exactly as the owning session would."""
    flr = flr or floor()
    f = tmp_path / "campaign" / f"floor-{flr['content_sha256'][:8]}-{grade['claim_class']}.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(flr))
    plan = fr.plan_registration(flr, floor_file=f, role=role, equivalence=repo_build_eq, grade=grade,
                                numerics_epoch=epoch, annotation=note)
    return ap.apply_plan(master, plan, allow_dirty=True), plan


def reg_of(tmp_path, master):
    return fr.FloorRegistry(master, tmp_path / "claims")


def lookup_kwargs(eq, champ, **over):
    kw = dict(equivalence_key=eq["equivalence_key"], recipe_hash="r" * 64, request_digest="q" * 64,
              model_sha="m" * 64, numerics_epoch="n1", pairs=5, champion=champ)
    kw.update(over)
    return kw


def eq_of(tmp_path, repo, name="b"):
    return fr.equivalence_for_build(build(tmp_path, repo, name, git(repo, "rev-parse", "HEAD")),
                                    source_repo=repo)


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


def test_applier_preserves_every_comment_and_is_idempotent(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    res, plan = register(tmp_path, master, eq)
    assert res["status"] == "applied" and (master.parent.parent / plan["record_dest"]).is_file()
    text = master.read_text()
    for line in MASTER_TEXT.split("\n"):
        assert line in text.split("\n")          # pure insertion: no original line lost
    assert "# inline comment" in text and "# trailing comment block" in text
    assert text.count("serving_floors:") == 1
    again, _ = register(tmp_path, master, eq)
    assert again["status"] == "already_present"
    assert master.read_text() == text


def test_second_entry_and_second_role_land_in_the_right_place(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    register(tmp_path, master, eq)
    register(tmp_path, master, eq, flr=floor(pairs=7), epoch="n2")
    register(tmp_path, master, eq, role="other_role", flr=floor(model="z" * 64))
    import yaml
    data = yaml.safe_load(master.read_text())
    assert len(data["serving_floors"][ROLE]) == 2 and len(data["serving_floors"]["other_role"]) == 1
    assert data["observations"] == {"a": 1} and data["roles"][ROLE]["tier"] == "test"
    errors, _ = fr.validate_section(data, master.parent.parent)
    assert errors == []


def test_register_lookup_roundtrip_reads_only_the_master(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    register(tmp_path, master, eq)
    reg = reg_of(tmp_path, master)
    got = reg.lookup(**lookup_kwargs(eq, fr.champion_state(repo, BRANCH)))
    assert got["floor_pct"] == 1.5 and got["content_sha256"] == floor()["content_sha256"]
    assert reg.lookup(**lookup_kwargs(eq, fr.champion_state(repo, BRANCH), role="other_role")) is None


def test_tests_only_advance_is_not_a_champion_advance_but_a_source_advance_is(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    register(tmp_path, master, eq)
    reg = reg_of(tmp_path, master)
    commit(repo, {"tests/t.cpp": "T9"}, "tests only")
    assert reg.lookup(**lookup_kwargs(eq, fr.champion_state(repo, BRANCH)))
    commit(repo, {"ggml/a.c": "A9"}, "real advance")
    res = reg.lookup_detail(**lookup_kwargs(eq, fr.champion_state(repo, BRANCH)))
    assert res.status == fr.STATUS_STALE_CHAMPION and res.floor is None
    assert reg.status(fr.champion_state(repo, BRANCH))[0]["state"] == "STALE_CHAMPION_ADVANCED"


def test_numerics_change_and_unresolvable_champion_refuse(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    register(tmp_path, master, eq)
    reg = reg_of(tmp_path, master)
    champ = fr.champion_state(repo, BRANCH)
    assert reg.lookup_detail(**lookup_kwargs(eq, champ, numerics_epoch="n2")).status == fr.STATUS_STALE_NUMERICS
    assert reg.lookup_detail(**lookup_kwargs(eq, None)).status == fr.STATUS_UNRESOLVABLE


def test_cotenant_floor_is_indicative_and_never_satisfies_decision(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    with pytest.raises(fr.FloorRegistryError):
        register(tmp_path, master, eq, grade={"threads": 48, "co_tenancy": "co-tenant", "claim_class": "decision"})
    register(tmp_path, master, eq, grade={"threads": 48, "co_tenancy": "co-tenant", "claim_class": "indicative"})
    reg = reg_of(tmp_path, master)
    champ = fr.champion_state(repo, BRANCH)
    assert reg.lookup_detail(**lookup_kwargs(eq, champ)).status == fr.STATUS_GRADE
    assert reg.lookup(**lookup_kwargs(eq, champ, require_grade="indicative")) is not None


def test_grade_threads_must_match_the_floors_own_launch(tmp_path, repo):
    with pytest.raises(fr.FloorRegistryError):
        register(tmp_path, master_in(tmp_path), eq_of(tmp_path, repo), flr=floor(threads=24))


def test_unsealed_floor_refused_and_tampered_record_is_corrupt(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    bad = floor()
    bad["floor_pct"] = 0.1  # seal no longer re-derives
    with pytest.raises(fr.FloorRegistryError):
        register(tmp_path, master, eq, flr=bad)
    _, plan = register(tmp_path, master, eq)
    rec = master.parent.parent / plan["record_dest"]
    doc = json.loads(rec.read_text())
    doc["floor_pct"] = 9.9
    rec.write_text(json.dumps(doc))
    res = reg_of(tmp_path, master).lookup_detail(**lookup_kwargs(eq, fr.champion_state(repo, BRANCH)))
    assert res.status == fr.STATUS_CORRUPT


def test_applier_refuses_a_record_that_differs_from_its_pin(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    f = tmp_path / "f.json"
    f.write_text(json.dumps(floor()))
    plan = fr.plan_registration(floor(), floor_file=f, role=ROLE, equivalence=eq, grade=QUIET, numerics_epoch="n1")
    f.write_text(json.dumps(floor(request="x" * 64)))
    with pytest.raises(ap.ApplyError):
        ap.apply_plan(master, plan, allow_dirty=True)
    assert master.read_text() == MASTER_TEXT


def test_validator_rules(tmp_path, repo):
    import yaml
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    register(tmp_path, master, eq)
    data = yaml.safe_load(master.read_text())
    root = master.parent.parent
    assert fr.validate_section(data, root)[0] == []
    entry = data["serving_floors"][ROLE][0]
    entry["grade"] = {"threads": 48, "co_tenancy": "co-tenant", "claim_class": "decision"}
    assert any("quiet" in m for m in fr.validate_section(data, root)[0])
    entry["grade"] = dict(QUIET)
    entry["unit"] = "arm"
    assert any("unit" in m for m in fr.validate_section(data, root)[0])
    entry["unit"] = "process"
    data["serving_floors"][ROLE].append(dict(entry))
    assert any("duplicate" in m for m in fr.validate_section(data, root)[0])
    data["serving_floors"]["not_a_role"] = [dict(entry, measured_at="2020-01-01T00:00:00Z")]
    assert any("not a roles" in m for m in fr.validate_section(data, root)[1])


def test_check_launch_reuse_then_one_claim_per_condition(tmp_path, repo):
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    reg = reg_of(tmp_path, master)
    champ = fr.champion_state(repo, BRANCH)
    kw = dict(equivalence=eq, recipe_hash="r" * 64, request_digest="q" * 64, model_sha="m" * 64, pairs=5,
              numerics_epoch="n1", require_grade="decision", champion=champ)
    first = fr.check_launch(reg, owner="one", **kw)
    second = fr.check_launch(reg, owner="two", **kw)
    assert (first["action"], second["action"]) == ("calibrate", "claimed")
    assert second["claim"]["owner"] == "one" and second["claim"]["state"] == "calibrating"
    assert reg.mark_awaiting_registration(first["condition"], "plan.json")
    assert fr.check_launch(reg, owner="three", **kw)["claim"]["state"] == "awaiting_registration"
    register(tmp_path, master, eq)  # the owner commits the registry entry
    out = fr.check_launch(reg, owner="three", **kw)
    assert out["action"] == "reuse" and reg.claim_holder(first["condition"]) is None


def test_check_launch_refuses_to_calibrate_a_stale_anchor(tmp_path, repo):
    """The Q38FN case: the campaign anchors a pre-advance build; calibrating it files a floor
    that is stale at birth."""
    old = eq_of(tmp_path, repo, "old")
    commit(repo, {"ggml/a.c": "ADVANCED"}, "champion advance")
    out = fr.check_launch(reg_of(tmp_path, master_in(tmp_path)), equivalence=old, recipe_hash="r" * 64,
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
