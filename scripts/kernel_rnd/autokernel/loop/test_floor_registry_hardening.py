"""The floor registry cannot silently read a stale registry (operator 2026-10-22).

Written under the code+py_compile-only constraint: NOT YET RUN. Run (CPU build claim) with
    cd scripts/kernel_rnd && python3 -m pytest autokernel/loop/test_floor_registry_hardening.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autokernel.loop import floor_registry as fr
from autokernel.loop.test_floor_registry import (BRANCH, MASTER_TEXT, QUIET, ROLE, eq_of, git, master_in,
                                                 register, repo)  # noqa: F401  (repo is a fixture)

def ok_fetch(repo):
    return {"ok": True, "error": None}


def failed_fetch(repo):
    return {"ok": False, "error": "could not resolve host"}


DS41_RECORD = ("/mnt/raid0/llm/autokernel/campaigns/ak-ds41-1017-cpu-decode-20261021/baseline-floor-20261022/"
               "serving-floor.ds41-1017-mxfp4-cpu-t48-targetonly.requests-e50732d9ce365259d5bb95fe31e36995f98bcd0"
               "986dc0e48d489f99390e42f58.matched_process_v2.pairs-5.json")


# ---------------------------------------------------------------- 1. the default is the CODE repo's registry
def test_default_master_resolves_inside_the_code_repo_not_a_hardcoded_shared_path(monkeypatch):
    monkeypatch.delenv("FLOOR_REGISTRY_MASTER", raising=False)
    code_repo = Path(fr.__file__).resolve().parents[4]
    assert fr.DEFAULT_MASTER == code_repo / "orchestration" / "model_registry.yaml"
    path, source = fr.resolve_master(None)
    assert (path, source) == (fr.DEFAULT_MASTER, "code-repo")
    assert fr.resolve_master(fr.CODE_REPO_TOKEN) == (fr.DEFAULT_MASTER, "code-repo")
    src = Path(fr.__file__).read_text()
    assert 'DEFAULT_MASTER = Path(' not in src and 'DEFAULT_MASTER = CODE_REPO' in src


def test_lib_registry_default_is_also_repo_relative():
    lib = Path(fr.__file__).resolve().parents[3] / "lib" / "registry.py"
    assert 'Path(__file__).resolve().parents[2] / "orchestration"' in lib.read_text()


# ---------------------------------------------------------------- 2. a registry behind origin/main refuses
def _behind_repo(tmp_path):
    """A git repo whose origin/main carries a DS41 entry the working-tree registry lacks."""
    root = tmp_path / "researchrepo"
    master = master_in(tmp_path)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return root, master


def test_behind_registry_makes_ak_refuse(tmp_path, repo):
    from autokernel.loop import run
    root, master = _behind_repo(tmp_path)
    eq = eq_of(tmp_path, repo)
    register(tmp_path, master, eq)                                    # local file now has the entry
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "floor")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")      # origin/main has it
    master.write_text(MASTER_TEXT)                                    # ... the local file does not
    with pytest.raises(fr.RegistryBehindUpstream, match="behind origin/main"):
        fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, fetch=ok_fetch)
    ctx = run._FloorRegistryContext(master, role=ROLE, numerics_epoch="n1", require_grade="decision",
                                    co_tenancy="quiet", source_repo=repo, branch=BRANCH, fetch=ok_fetch)
    with pytest.raises(fr.RegistryBehindUpstream):
        ctx.ensure_preflight("m" * 64)
    # the lookup path must propagate it, never swallow it into "registry unavailable"
    class Launch:                                                      # only what the preflight touches
        build_dir = "/nonexistent"
        class model:
            sha256 = "m" * 64
    run._FLOOR_REGISTRY = ctx
    try:
        with pytest.raises(fr.RegistryBehindUpstream):
            run._registry_floor_reading(object(), Launch, frozen_requests=None, pairs=5)
    finally:
        run._FLOOR_REGISTRY = None


def test_up_to_date_registry_passes_and_unreadable_upstream_only_warns(tmp_path, repo):
    root, master = _behind_repo(tmp_path)
    lines: list[str] = []
    res = fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, log=lines.append,
                                 fetch=ok_fetch)   # no origin/main ref at all
    assert res["status"] == "skipped" and any("SKIPPED" in l for l in lines)
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, log=lines.append,
                                  fetch=ok_fetch)["status"] == "ok"


def test_model_sha_alone_is_enough_to_detect_a_behind_registry(tmp_path, repo):
    root, master = _behind_repo(tmp_path)
    register(tmp_path, master, eq_of(tmp_path, repo))
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "floor")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    master.write_text(MASTER_TEXT)
    with pytest.raises(fr.RegistryBehindUpstream):
        fr.preflight_or_refuse(master, role="a_different_role", model_sha="m" * 64, fetch=ok_fetch)


# ---------------------------------------------------------------- 3. hit/miss logging reaches the receipt
def test_every_lookup_is_logged_and_lands_in_the_receipt(tmp_path, repo):
    from autokernel.loop import run
    master = master_in(tmp_path)
    eq = eq_of(tmp_path, repo)
    register(tmp_path, master, eq)
    lines: list[str] = []
    reg = fr.FloorRegistry(master, tmp_path / "claims", log=lines.append)
    champ = fr.champion_state(repo, BRANCH)
    kw = dict(equivalence_key=eq["equivalence_key"], recipe_hash="r" * 64, request_digest="q" * 64,
              model_sha="m" * 64, numerics_epoch="n1", pairs=5, champion=champ)
    assert reg.lookup_detail(**kw)
    assert not reg.lookup_detail(**{**kw, "numerics_epoch": "n2"})
    rows = reg.receipt()["lookups"]
    assert [r["status"] for r in rows] == [fr.STATUS_HIT, fr.STATUS_STALE_NUMERICS]
    assert rows[0]["entry_id"] and rows[0]["floor_pct"] == 1.5
    assert all(r["registry"] == str(master) and "registry_commit" in r for r in rows)
    assert rows[1]["reasons"]
    assert len(lines) == 2 and "floor registry lookup: hit" in lines[0] and str(master) in lines[0]
    # ... and the AK record gets the block only when the registry is enabled
    assert run._floor_registry_receipt() == {}
    ctx = run._FloorRegistryContext(master, role=ROLE, numerics_epoch="n1", require_grade="decision",
                                    co_tenancy="quiet", source_repo=repo, branch=BRANCH, store=tmp_path)
    ctx.registry.lookups.extend(rows)
    run._FLOOR_REGISTRY = ctx
    try:
        block = run._floor_registry_receipt()["floor_registry"]
        assert block["lookup_count"] == 2 and block["path"] == str(master)
        ctx.persist()
        persisted = [json.loads(l) for l in (tmp_path / "floor-registry-lookups.jsonl").read_text().splitlines()]
        assert [r["status"] for r in persisted] == [fr.STATUS_HIT, fr.STATUS_STALE_NUMERICS]
    finally:
        run._FLOOR_REGISTRY = None


# ---------------------------------------------------------------- 4. DS41-1017 end to end, DEFAULT path
def test_ds41_1017_lookup_through_the_default_registry(monkeypatch):
    if not Path(DS41_RECORD).is_file():
        pytest.skip("DS41-1017 sealed record is not on this host")
    monkeypatch.delenv("FLOOR_REGISTRY_MASTER", raising=False)
    lines: list[str] = []
    reg = fr.FloorRegistry(log=lines.append)                       # NO path: the code repo's registry
    assert reg.master == fr.DEFAULT_MASTER and reg.master_source == "code-repo"
    champion = {"branch": "ak/champion/llama-cpp-ffc1bac82eec",
                "commit": "dd3d9721ca37321ad9159520bbd9616e3d0d6228",
                "source_tree_hash": "fa33ccecb9e201780cb2f71be0327a76866e7342154113d2ff9de65a2c9e44ec"}
    result = reg.lookup_detail(
        equivalence_key="c28bd41baf245eebe54e05167735250477245b970e342b1b950604c2b9a63c3a",
        recipe_hash="65f1c793e12f37a2055e2f4b84c345663a09c4491b29553f7321f40461274e4b",
        request_digest="e50732d9ce365259d5bb95fe31e36995f98bcd0986dc0e48d489f99390e42f58",
        model_sha="ba9f129c15f823503d00b088ff07e42aa3953d58cf287d019229af052b2e12b2",
        numerics_epoch="pre-W5.1a", pairs=5, require_grade="decision", champion=champion,
        role="deepseek_v41_flash_1017_mxfp4")
    assert result, result.reasons
    assert result.floor["floor_pct"] == 2.862 and result.floor["n"] == 24
    assert result.entry["grade"] == {"threads": 48, "co_tenancy": "quiet", "claim_class": "decision"}
    receipt = reg.receipt()
    assert receipt["path"] == str(fr.DEFAULT_MASTER) and receipt["source"] == "code-repo"
    assert receipt["lookups"][0]["entry_id"] == "8f2e058d7686b9f7"
    stale = reg.lookup_detail(
        equivalence_key="c28bd41baf245eebe54e05167735250477245b970e342b1b950604c2b9a63c3a",
        recipe_hash="65f1c793e12f37a2055e2f4b84c345663a09c4491b29553f7321f40461274e4b",
        request_digest="e50732d9ce365259d5bb95fe31e36995f98bcd0986dc0e48d489f99390e42f58",
        model_sha="ba9f129c15f823503d00b088ff07e42aa3953d58cf287d019229af052b2e12b2",
        numerics_epoch="post-W5.1a", pairs=5, champion=champion)
    assert stale.status == fr.STATUS_STALE_NUMERICS


# ---------------------------------------------------------------- 5. origin/main is REFRESHED before comparing
def _up_to_date_repo(tmp_path):
    root, master = _behind_repo(tmp_path)
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return master


def test_fetch_runs_before_the_comparison(tmp_path):
    master = _up_to_date_repo(tmp_path)
    order: list[str] = []
    fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, log=lambda m: None,
                           fetch=lambda r: order.append("fetch") or {"ok": True, "error": None},
                           show=lambda r, rel: order.append("show") or master.read_text())
    assert order == ["fetch", "show"]


def test_failed_fetch_with_a_recent_fetch_only_warns(tmp_path):
    master = _up_to_date_repo(tmp_path)
    lines: list[str] = []
    res = fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, log=lines.append,
                                 fetch=failed_fetch, age=lambda r: 3600)
    assert res["status"] == "ok" and res["last_fetch_age_s"] == 3600
    assert any("FAILED" in l and "3600s ago" in l for l in lines)


@pytest.mark.parametrize("age", [25 * 3600, None])
def test_failed_fetch_with_an_old_or_unknown_fetch_refuses_unless_allowed(tmp_path, age):
    master = _up_to_date_repo(tmp_path)
    with pytest.raises(fr.RegistryUpstreamUnknown, match="--floor-allow-stale-upstream"):
        fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, fetch=failed_fetch, age=lambda r: age)
    res = fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, log=lambda m: None,
                                 fetch=failed_fetch, age=lambda r: age, allow_stale=True)
    assert res["status"] == "ok"


def test_allow_stale_never_hides_a_behind_registry(tmp_path, repo):
    root, master = _behind_repo(tmp_path)
    register(tmp_path, master, eq_of(tmp_path, repo))
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "floor")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    master.write_text(MASTER_TEXT)
    with pytest.raises(fr.RegistryBehindUpstream):
        fr.preflight_or_refuse(master, role=ROLE, model_sha="m" * 64, fetch=failed_fetch,
                               age=lambda r: 60, allow_stale=True)


def test_real_fetch_failure_is_reported_not_raised(tmp_path):
    root, _ = _behind_repo(tmp_path)               # a repo with no `origin` remote
    got = fr.fetch_origin(root, timeout=5)
    assert got["ok"] is False and got["error"]


def test_ak_flag_and_serial_switch_are_wired():
    from autokernel.loop import serial_run as sr
    assert "--floor-allow-stale-upstream" in sr.FLOOR_REGISTRY_SWITCHES
    assert "--floor-allow-stale-upstream" in sr._BOOLEAN_FLAGS and "--floor-allow-stale-upstream" in sr._CHANGING_FLAGS
    argv = ["--worktree", "/w", "--floor-allow-stale-upstream", "--store", "/s"]
    assert sr._without(argv, sr.FLOOR_REGISTRY_SWITCHES) == ["--worktree", "/w", "--store", "/s"]


# ---------------------------------------------------------------- 6. scripts/lib/registry.py from any cwd
def test_lib_registry_default_is_cwd_independent(tmp_path):
    repo_root = Path(fr.__file__).resolve().parents[4]
    lib = repo_root / "scripts" / "lib" / "registry.py"
    code = ("import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('lib_registry_probe', sys.argv[1])\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(m.DEFAULT_REGISTRY_PATH)\n")
    expected = str(repo_root / "orchestration" / "model_registry.yaml")
    store_like = tmp_path / "autokernel-store"
    store_like.mkdir()
    (tmp_path / "linked").symlink_to(lib.parent)                    # the /workspace/repos/... symlink shape
    for cwd, path in ((tmp_path, lib), (store_like, lib), (store_like, tmp_path / "linked" / "registry.py")):
        out = subprocess.run([sys.executable, "-I", "-c", code, str(path)], cwd=cwd, capture_output=True,
                             text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == expected
    assert Path(expected).is_file()
