"""AKX-AUTO-6: champion-advance detection, re-anchor plan, one recalibration per advance.

NOT YET RUN (host constraint until 2026-10-22 06:30Z: code + py_compile only). Stdlib + a throwaway
git repo; `accumulate.would_refuse_anchor` is stubbed (it needs a real journal store).
    cd scripts/kernel_rnd && python3 -m pytest autokernel/loop/test_champion_reanchor.py -q
"""
from __future__ import annotations

import json
import time

import pytest

from autokernel.loop import champion_reanchor as cr
from autokernel.loop import floor_registry as fr
from autokernel.loop.test_floor_registry import BRANCH, MASTER_TEXT, QUIET, ROLE, build, commit, floor, git, master_in, register

PREFLIGHT = {"refuses": True, "reason": "ancestry", "kind": "ancestry",
             "needs_new_anchor_epoch": True, "blocking": False}


@pytest.fixture()
def world(tmp_path, monkeypatch):
    repo = tmp_path / "llama"
    repo.mkdir()
    git(repo, "init", "-q", "-b", BRANCH)
    old = commit(repo, {"ggml/a.c": "A", "tests/t.cpp": "T", "CMakeLists.txt": "c"}, "old champion")
    store = tmp_path / "store"
    store.mkdir()
    (store / "accumulator-bundle.json").write_text(json.dumps({"champion_of_record": old[:8], "tip": old}))
    (store / "loop-status.json").write_text(json.dumps(
        {"state": "complete", "anchor_commit": old, "generated_at": "2026-10-11T06:28:25.5Z",
         "stale_after_s": 1800, "accumulator": {"champion_of_record": old}}))
    builds = tmp_path / "builds"
    builds.mkdir()
    monkeypatch.setattr(cr, "_would_refuse", lambda *a, **k: dict(PREFLIGHT))
    return {"repo": repo, "store": store, "builds": builds, "old": old, "tmp": tmp_path}


def advance(world, tests_only=False):
    files = {"tests/t.cpp": "T2"} if tests_only else {"ggml/a.c": "A2"}
    new = commit(world["repo"], files, "advance")
    return new


def test_tests_only_advance_is_current(world):
    advance(world, tests_only=True)
    out = cr.detect(world["store"], world["repo"], BRANCH)
    assert out["state"] == "current"


def test_real_advance_is_behind_and_names_both_commits(world):
    new = advance(world)
    out = cr.detect(world["store"], world["repo"], BRANCH)
    assert out["state"] == "behind"
    assert out["champion"]["commit"] == new and out["campaign"]["cor_full"] == world["old"]


def test_unreadable_and_diverged_are_never_actionable(world, tmp_path):
    empty = tmp_path / "nostore"
    empty.mkdir()
    assert cr.detect(empty, world["repo"], BRANCH)["state"] == "unreadable"
    git(world["repo"], "checkout", "-q", "-b", "side", world["old"])
    side = commit(world["repo"], {"ggml/a.c": "SIDE"}, "side")
    git(world["repo"], "checkout", "-q", BRANCH)
    advance(world)
    (world["store"] / "accumulator-bundle.json").write_text(json.dumps({"champion_of_record": side}))
    assert cr.detect(world["store"], world["repo"], BRANCH)["state"] == "diverged"


def store_build_for_champion(world):
    return build(world["builds"].parent, world["repo"], "builds/cpu-new",
                 git(world["repo"], "rev-parse", "HEAD"))


def seed_floor(world, registry_root):
    """The campaign's existing floor condition, on disk, so plan can name the identity."""
    d = world["store"] / "runtime-source-floors" / ("r" * 64) / ("e" * 64)
    d.mkdir(parents=True)
    (d / "serving-floor.t.requests-x.matched_process_v2.pairs-5.json").write_text(json.dumps(floor()))


def test_plan_calibrates_once_then_second_launcher_is_claimed_then_reuse(world):
    advance(world)
    store_build_for_champion(world)
    seed_floor(world, None)
    master = master_in(world["tmp"])
    kw = dict(master=master, role=ROLE, claims_root=world["tmp"] / "claims", branch=BRANCH,
              builds_root=world["builds"], numerics_epoch="n1")
    p1 = cr.plan(world["store"], world["repo"], **kw)
    assert p1["actionable"] and p1["floor"]["would"] == "calibrate"
    assert p1["run_argv_additions"][:2] == ["--anchor-build", p1["anchor_build"]]
    assert "--new-anchor-epoch" in p1["run_argv_additions"]
    # apply claims; a second apply by another owner is refused (claimed), not a second A/A
    done = cr.apply(world["store"], world["repo"], owner="f8", execute=True, relaunch_cmd=["true"], **kw)
    assert done["executed"] and done["action"] == "calibrate"
    spec = json.loads(open(done["spec"]).read())
    assert spec["floor_action"] == "calibrate" and spec["calibrate_serving_pairs"] == 24
    (world["store"] / "loop-status.json").write_text(json.dumps({"state": "complete"}))
    # relaunch rc 0 but nothing registered: the claim is still held by the first owner
    with pytest.raises(cr.ReanchorRefused):
        cr.apply(world["store"], world["repo"], owner="other", execute=True, relaunch_cmd=["true"], **kw)
    # once a decision floor is registered the next advance-check reuses it: no A/A
    eq = fr.equivalence_for_build(p1["anchor_build"], source_repo=world["repo"])
    register(world["tmp"], master, eq)  # the owning session commits the registry entry
    p2 = cr.plan(world["store"], world["repo"], **kw)
    assert p2["floor"]["would"] == "reuse"
    again = cr.apply(world["store"], world["repo"], owner="other", execute=True, relaunch_cmd=["true"], **kw)
    assert again["action"] == "reuse"


def test_failed_relaunch_releases_the_claim(world):
    advance(world)
    store_build_for_champion(world)
    seed_floor(world, None)
    master = master_in(world["tmp"])
    kw = dict(master=master, role=ROLE, claims_root=world["tmp"] / "claims", branch=BRANCH,
              builds_root=world["builds"], numerics_epoch="n1")
    out = cr.apply(world["store"], world["repo"], owner="f8", execute=True, relaunch_cmd=["false"], **kw)
    assert out["relaunch_rc"] != 0 and out["claim_released"] is True
    assert cr.plan(world["store"], world["repo"], **kw)["floor"]["would"] == "calibrate"


def test_live_campaign_is_never_touched(world):
    advance(world)
    store_build_for_champion(world)
    seed_floor(world, None)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (world["store"] / "loop-status.json").write_text(json.dumps(
        {"state": "running", "generated_at": now, "stale_after_s": 1800,
         "accumulator": {"champion_of_record": world["old"]}}))
    with pytest.raises(cr.ReanchorRefused):
        cr.apply(world["store"], world["repo"], owner="f8", execute=True, relaunch_cmd=["true"],
                 master=master_in(world["tmp"]), role=ROLE, claims_root=world["tmp"] / "claims",
                 branch=BRANCH, builds_root=world["builds"])


def test_dry_run_changes_nothing(world):
    advance(world)
    store_build_for_champion(world)
    seed_floor(world, None)
    claims = world["tmp"] / "claims"
    out = cr.apply(world["store"], world["repo"], owner="f8", relaunch_cmd=["false"],
                   master=master_in(world["tmp"]), role=ROLE, claims_root=claims,
                   branch=BRANCH, builds_root=world["builds"])
    assert out["executed"] is False and not claims.exists()
    assert not (world["store"] / "reanchor").exists()


def test_no_store_build_for_the_champion_blocks(world):
    advance(world)
    out = cr.plan(world["store"], world["repo"], master=master_in(world["tmp"]), role=ROLE, branch=BRANCH,
                  builds_root=world["builds"])
    assert not out["actionable"] and "store" in out["blockers"][0]


def test_rejected_tree_matching_image_is_named_with_its_reason(world):
    advance(world)
    b = store_build_for_champion(world)
    (b / "PROVENANCE.md").unlink()
    out = cr.plan(world["store"], world["repo"], master=master_in(world["tmp"]), role=ROLE, branch=BRANCH,
                  builds_root=world["builds"])
    assert not out["actionable"]
    msg = out["blockers"][0]
    assert "cpu-new" in msg and "missing PROVENANCE.md" in msg and "REJECTED" in msg
    assert "build and store it first" not in msg
    assert out["rejected_builds"][0]["build"] == str(b)


def test_in_tree_provenance_json_image_is_rejected_by_name(world):
    advance(world)
    b = store_build_for_champion(world)
    (b / "provenance.json").write_text("{}")
    diag = cr.diagnose_store_builds(
        fr.champion_state(world["repo"], BRANCH), world["repo"], world["builds"])
    assert diag["match"] is None and "provenance.json" in diag["rejected"][0]["reason"]


def test_other_tree_image_is_not_blamed_and_message_stays_build_it(world):
    old_build = build(world["tmp"], world["repo"], "builds/cpu-old", world["old"])
    (old_build / "PROVENANCE.md").unlink()
    advance(world)
    out = cr.plan(world["store"], world["repo"], master=master_in(world["tmp"]), role=ROLE, branch=BRANCH,
                  builds_root=world["builds"])
    assert "build and store it first" in out["blockers"][0] and "rejected_builds" not in out


def test_good_image_still_found_and_find_store_build_unchanged(world):
    advance(world)
    b = store_build_for_champion(world)
    eq = cr.find_store_build(fr.champion_state(world["repo"], BRANCH), world["repo"], world["builds"])
    assert eq is not None and eq["build_dir"] == str(b)


def test_loop_toplevel_invocation_is_refused_with_supported_forms(capsys):
    assert cr._invocation_ok("loop") is False and cr._invocation_ok(None) is False
    assert cr._invocation_ok("autokernel.loop") and cr._invocation_ok("kernel_rnd.autokernel.loop")
    assert "autokernel.loop.champion_reanchor" in cr.SUPPORTED_INVOCATION
    assert "SUPPORTED INVOCATION" in cr.__doc__


PLAIN = {"refuses": False, "reason": "tip advanced to the anchor", "kind": "other",
         "needs_new_anchor_epoch": False, "blocking": False}


def test_plain_advance_names_the_exact_store_image_of_the_kept_champion_of_record(world, monkeypatch):
    """CH-27 (2026-10-23, Q38FN dd3d9721c -> 141e4bcf5): a plain advance keeps the store's
    champion_of_record, and run.py refuses without that commit's EXACT build as --cor-build. The
    plan names its store image (or blocks), and the relaunch paths carry it as owner cor_build."""
    monkeypatch.setattr(cr, "_would_refuse", lambda *a, **k: dict(PLAIN))
    advance(world)
    store_build_for_champion(world)
    seed_floor(world, None)
    kw = dict(master=master_in(world["tmp"]), role=ROLE, claims_root=world["tmp"] / "claims",
              branch=BRANCH, builds_root=world["builds"], numerics_epoch="n1")
    blocked = cr.plan(world["store"], world["repo"], **kw)
    assert not blocked["actionable"] and "--cor-build" in blocked["blockers"][0]
    old_image = build(world["builds"].parent, world["repo"], "builds/cpu-old", world["old"])
    p = cr.plan(world["store"], world["repo"], **kw)
    assert p["actionable"] and p["cor_build"] == str(old_image)
    add = p["run_argv_additions"]
    assert "--new-anchor-epoch" not in add
    assert add[add.index("--cor-build") + 1] == str(old_image)
    assert cr.find_exact_store_build(world["old"], world["repo"], world["builds"]) == str(old_image)
    # ancestry refusal (new epoch) => the COR is re-based, no --cor-build
    monkeypatch.setattr(cr, "_would_refuse", lambda *a, **k: dict(PREFLIGHT))
    assert "--cor-build" not in cr.plan(world["store"], world["repo"], **kw)["run_argv_additions"]
