"""Cross-lane quality gate wired into the global-champion fold (cross_target.fold_onto_champion)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from . import cross_lane_gate as clg
from . import cross_target as ct
from .test_cross_lane_gate import ev

GUARD = "autokernel.loop.kernel_mutation_guard.ensure_kernel_mutation_allowed"


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def world(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.c").write_text("int x = 0;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "champ")
    (repo / "a.c").write_text("int x = 1;\n")
    _git(repo, "commit", "-qam", "keep\n\nAK-Numerics: ppl_contract")
    return repo, _git(repo, "rev-parse", "HEAD")


def _fold(repo, keep, **kw):
    with mock.patch(GUARD, return_value=None):
        return ct.fold_onto_champion(repo, "champ", [keep], note="t",
                                     fold_check=lambda **k: {"passed": True}, **kw)


def test_hold_is_unchanged_without_lane_or_evidence(world):
    repo, keep = world
    old = _git(repo, "rev-parse", "champ")
    for kw in ({}, {"lane": "q38fn"}, {"lane_evidence": {"q38fn": ev()}}):
        out = _fold(repo, keep, **kw)
        assert out["result"] == "ppl_contract_held" and out["commit"] is None
    assert _git(repo, "rev-parse", "champ") == old


def test_admissible_lane_evidence_lifts_the_hold_and_folds(world):
    repo, keep = world
    good = ev("q38fn")
    good["candidate_commit"] = keep
    out = _fold(repo, keep, lane="q38fn", lane_evidence={"q38fn": good})
    assert out["result"] == "folded" and out["cross_lane"]["admissible"] is True
    assert _git(repo, "rev-parse", "champ") == out["commit"]


def test_refused_evidence_keeps_the_hold_and_reports_reasons(world):
    repo, keep = world
    old = _git(repo, "rev-parse", "champ")
    bad = ev("q38fn")                     # evidence is for another commit
    out = _fold(repo, keep, lane="q38fn", lane_evidence={"q38fn": bad})
    assert out["result"] == "ppl_contract_held" and out["cross_lane"]["admissible"] is False
    assert any("different build" in r
               for r in out["cross_lane"]["lanes"]["q38fn"]["reasons"])
    worse = ev("q38fn")
    worse["candidate_commit"] = keep
    worse["ppl"]["candidate_ppl"] = 12.0
    out = _fold(repo, keep, lane="q38fn", lane_evidence={"q38fn": worse})
    assert out["result"] == "ppl_contract_held"
    assert _git(repo, "rev-parse", "champ") == old


def test_evidence_is_loaded_from_the_store_and_damaged_files_only_refuse(world, tmp_path):
    repo, keep = world
    store = tmp_path / "store"
    (store / clg.EVIDENCE_DIR).mkdir(parents=True)
    (store / clg.EVIDENCE_DIR / "q38fn.json").write_text("{not json")
    assert _fold(repo, keep, lane="q38fn", store=store)["result"] == "ppl_contract_held"
    good = ev("q38fn")
    good["candidate_commit"] = keep
    (store / clg.EVIDENCE_DIR / "q38fn.json").write_text(json.dumps(good))
    assert clg.load_lane_evidence(store)["q38fn"]["candidate_commit"] == keep
    assert _fold(repo, keep, lane="q38fn", store=store)["result"] == "folded"


def test_non_ppl_commit_never_consults_the_gate(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "docs.txt").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "branch", "champ")
    (repo / "docs.txt").write_text("b\n")
    _git(repo, "commit", "-qam", "plain")
    out = _fold(repo, _git(repo, "rev-parse", "HEAD"), lane="q38fn")
    assert out["result"] == "folded" and "cross_lane" not in out
