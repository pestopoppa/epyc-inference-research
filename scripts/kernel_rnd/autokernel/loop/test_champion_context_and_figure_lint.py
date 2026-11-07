"""AK-TOOL-CHAMPION-CONTEXT and AK-TOOL-UNSOURCED-LINT, each with a negative control."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from autokernel.loop import actor_context, belief_context, champion, figure_lint

MARK = figure_lint.UNSOURCED_MARK


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "tree"
    r.mkdir()
    _git(r, "init", "-q", "-b", champion.CANONICAL_BRANCH)
    for t in ("a", "b"):
        (r / "f").write_text(t)
        _git(r, "add", "f")
        _git(r, "commit", "-q", "-m", t)
    return r


def _anchor(tmp_path, commit):
    d = tmp_path / "anchor-gen-001"
    d.mkdir()
    (d / "provenance.json").write_text(json.dumps({"champion_commit": commit}))
    return d


def test_current_anchor_passes(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    assert champion.verify_startup(worktree=repo, branch=champion.CANONICAL_BRANCH,
                                   anchor_build=_anchor(tmp_path, head)) == head


def test_stale_anchor_is_refused_with_typed_reason(repo, tmp_path):
    """NEGATIVE CONTROL: an ancestor anchor passes the legacy ancestry check yet is refused."""
    head = _git(repo, "rev-parse", "HEAD")
    old = _git(repo, "rev-parse", "HEAD~1")
    anchor = _anchor(tmp_path, old)
    champion.verify_anchor(anchor, repo, head)  # legacy check alone accepts it: non-vacuous
    with pytest.raises(champion.StartupRefused) as err:
        champion.verify_startup(worktree=repo, branch=champion.CANONICAL_BRANCH, anchor_build=anchor)
    assert err.value.reason == champion.REASON_STALE_ANCHOR
    # the escape hatch for the legacy check does not waive currency
    assert champion.verify_startup(worktree=repo, branch=champion.CANONICAL_BRANCH,
                                   anchor_build=anchor, require_current_anchor=False) == head


def test_unattested_anchor_cannot_be_proven_current(repo, tmp_path):
    d = tmp_path / "hand"
    d.mkdir()
    with pytest.raises(champion.StartupRefused) as err:
        champion.verify_anchor_current(d, repo, _git(repo, "rev-parse", "HEAD"))
    assert err.value.reason == champion.REASON_NO_ATTESTATION


# ------------------------------------------------------------------ unsourced lint

RESULT = "/raw/RESULT.json"
_exists = lambda t: t == RESULT  # noqa: E731


def test_record_figure_sourced_by_ancestor_key_is_clean():
    rec = {"h6": {"evidence": {"result_path": RESULT}, "cells": [{"ratio": 0.9997}]}}
    assert figure_lint.find_unsourced(rec, exists=_exists) == []


def test_h6_style_figure_without_raw_data_is_flagged():
    rec = {"h6": {"ratio": 1.017}, "h10": {"result_path": RESULT, "ratio": 1.05}}
    assert figure_lint.find_unsourced(rec, exists=_exists) == ["h6.ratio"]
    # a source key whose path does not resolve is no source either
    assert figure_lint.find_unsourced({"r": 1.0, "result_path": "/nope.json"}, exists=_exists) == ["r"]


def test_text_marking_and_negative_control():
    text, flagged = figure_lint.mark_text("H6 gave 1.017x\nH10 +4.8% see /raw/RESULT.json\nno figures",
                                          exists=_exists)
    assert flagged == ["1.017x"] and f"1.017x {MARK}" in text
    assert text.count(MARK) == 1 and "+4.8% see" in text


def test_strict_refuses():
    with pytest.raises(figure_lint.UnsourcedFigureRefused):
        figure_lint.mark_text("1.017x", exists=_exists, strict=True)
    with pytest.raises(figure_lint.UnsourcedFigureRefused):
        belief_context.unsourced_figures({"x": 3.2}, strict=True, exists=_exists)




def test_index_names_unsourced_figures_without_mutating_sections(tmp_path):
    from autokernel.loop import scratch
    reg = scratch.ScratchRegistry(tmp_path / "scratch", {"run_id": "t"}, 0)
    ctx = "## evidence\nH6 measured 1.017x\n"
    with reg.scope("call", "planner") as scope:
        bundle = actor_context.materialize(ctx, tmp_path, role="planner", scope=scope)
    assert MARK in bundle.index and "1.017x" in bundle.index
    assert "H6 measured 1.017x\n" in bundle.index  # inline section stays verbatim
    # negative control: no figures -> no block
    with reg.scope("call2", "planner") as scope:
        clean = actor_context.materialize("## evidence\nnothing numeric\n", tmp_path,
                                          role="planner", scope=scope)
    assert MARK not in clean.index


def test_section_level_resolution_clears_a_cited_table():
    text = "Record: /raw/RESULT.json\n| 50.02% | a |\n"
    assert figure_lint.mark_text(text, exists=_exists, section_level=True)[1] == []
    assert figure_lint.mark_text(text, exists=lambda _t: False, section_level=True)[1] == ["50.02%"]
