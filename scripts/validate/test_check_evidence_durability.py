#!/usr/bin/env python3
"""Tests for the evidence-durability validator.

Two things these tests care about beyond the obvious.

FIRST, the guard must not forbid its own idiom. A validator that only ever gets
exercised on broken input can be quietly wrong about correct input, and the failure
mode -- flagging every properly-migrated citation -- is the one that gets the check
switched off. `test_compliant_registry_is_silent` and the real-registry integration
test cover the COMPLIANT path deliberately.

SECOND, a waiver must never be a silent pass. `ARTIFACT LOST` suppresses an error, so
it is exactly the mechanism someone reaches for to make a red tree green. It is
therefore asserted to stay VISIBLE (severity "warn", never "ok") and to be caught by
`--warnings-as-errors`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_evidence_durability import (  # noqa: E402
    ARTIFACT_TREES,
    Citation,
    check,
    classify,
    expand_braces,
    extract_citations,
    main,
)

SCRIPT = Path(__file__).resolve().parent / "check_evidence_durability.py"
REAL_REPO = Path(__file__).resolve().parents[2]
REAL_REGISTRY = REAL_REPO / "orchestration" / "model_registry.yaml"


# --------------------------------------------------------------------------- fixture

@pytest.fixture
def repo(tmp_path):
    """A miniature repository with real artifacts on disk.

    The fixture deliberately CREATES the files a compliant citation points at. A
    fixture that omitted them would make every citation MISSING and let a validator
    that had lost the ability to recognise a good path still pass the suite.
    """
    r = tmp_path / "epyc-inference-research"
    (r / "data" / "good_campaign_20260801").mkdir(parents=True)
    (r / "data" / "good_campaign_20260801" / "summary.json").write_text("{}")
    (r / "data" / "good_campaign_20260801" / "README.md").write_text("# what")
    (r / "data" / "good_campaign_20260801" / "SHA256SUMS").write_text("x  y\n")
    (r / "data" / "undocumented_campaign").mkdir(parents=True)
    (r / "data" / "undocumented_campaign" / "result.json").write_text("{}")
    (r / "artifacts").mkdir()
    (r / "artifacts" / "run-a-scored.json").write_text("{}")
    (r / "artifacts" / "run-b-scored.json").write_text("{}")
    return r


@pytest.fixture
def no_ephemeral_tmp(monkeypatch):
    """Drop `/tmp` from the ephemeral roots for the two tests that need a path OUTSIDE
    the fixture repo but still under pytest's `/tmp` basetemp. `/mnt/raid0/llm/tmp` stays
    ephemeral, so nothing that this suite actually asserts about scratch is weakened."""
    import check_evidence_durability as m
    monkeypatch.setattr(m, "EPHEMERAL_ROOTS", ("/mnt/raid0/llm/tmp", "/dev/shm"))


def write_registry(repo: Path, body: str) -> Path:
    p = repo / "registry.yaml"
    p.write_text(body)
    return p


def verdicts(res):
    return {c.path: c.verdict for c in res.citations}


# ------------------------------------------------------------------------ extraction

def test_finds_scratch_paths_in_comments_and_prose(repo):
    reg = write_registry(repo, """
roles:
  a:
    # baseline held (/mnt/raid0/llm/tmp/gpu_coresidency/qwen27b.log:8). Never a headline.
    note: "measured at /mnt/raid0/llm/tmp/some-run-20260716/summary.json under load"
    evidence: /mnt/raid0/llm/tmp/other-run/plan.json
""")
    res = check(reg, repo)
    paths = {c.path for c in res.citations}
    assert "/mnt/raid0/llm/tmp/gpu_coresidency/qwen27b.log" in paths
    assert "/mnt/raid0/llm/tmp/some-run-20260716/summary.json" in paths
    assert "/mnt/raid0/llm/tmp/other-run/plan.json" in paths
    # comments and quoted prose are where most real citations lived; a key-driven
    # scan would have found only the third one
    assert len(res.errors) == 3


def test_line_reference_is_preserved_and_not_part_of_the_path(repo):
    reg = write_registry(repo, "x: data/good_campaign_20260801/summary.json:8\n")
    (c,) = check(reg, repo).citations
    assert c.path == "data/good_campaign_20260801/summary.json"
    assert c.lineref == ":8"
    assert c.verdict == "OK"


def test_trailing_prose_punctuation_is_trimmed(repo):
    reg = write_registry(repo, "# see data/good_campaign_20260801/summary.json.\n")
    (c,) = check(reg, repo).citations
    assert c.path == "data/good_campaign_20260801/summary.json"
    assert c.verdict == "OK"


def test_brace_expansion():
    assert expand_braces("a{x,y}b") == ["axb", "ayb"]
    assert expand_braces("plain") == ["plain"]


def test_brace_group_cites_two_real_artifacts(repo):
    reg = write_registry(repo, "# rows carry score.pass: artifacts/run-{a,b}-scored.json\n")
    res = check(reg, repo)
    assert sorted(c.path for c in res.citations) == [
        "artifacts/run-a-scored.json", "artifacts/run-b-scored.json"]
    assert not res.errors


def test_provenance_notes_are_not_live_citations(repo):
    """A line recording where an artifact USED to live is history. Re-flagging it would
    punish the very act of documenting a migration."""
    reg = write_registry(repo, """
      evidence: data/good_campaign_20260801/summary.json
      # 2026-08-02: REPOINTED from /mnt/raid0/llm/tmp/vision_final_results.json. Copied
""")
    res = check(reg, repo)
    assert [c.path for c in res.citations] == ["data/good_campaign_20260801/summary.json"]
    assert not res.errors


def test_bare_scratch_root_is_prose_not_a_citation(repo):
    """Policy text names the scratch root; that is documentation, not evidence."""
    reg = write_registry(repo, "# Scratch paths (/mnt/raid0/llm/tmp/) must not be cited.\n")
    assert check(reg, repo).citations == []


def test_anything_under_the_scratch_root_still_counts(repo):
    """The complement of the rule above -- the exemption must not generalise."""
    reg = write_registry(repo, "# see /mnt/raid0/llm/tmp/x.json\n")
    (c,) = check(reg, repo).citations
    assert c.verdict == "EPHEMERAL"


# ----------------------------------------------------------------------------- scope

@pytest.mark.parametrize("line", [
    "    script: scripts/validate/stack_change_guard.py",
    "    plan: handoffs/active/some-handoff.md",
    "    doc: docs/reference/whatever.md",
    "    topology: orchestration/stack_topology.yaml",
    "    # supergemma4_31b_q4km is a research/catalogue model.",
    "    model_path: /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
    "    lib: /opt/rocm/lib",
    "    probe: /sys/class/kfd/topology",
    "    mount: --index-dir /data/indices",
])
def test_non_evidence_paths_are_out_of_scope(repo, line):
    """Code, prose, weights and system paths are not measurement evidence. Folding them
    in buries the evidence signal under dozens of stale doc links -- and `research/`
    collides with English."""
    assert check(write_registry(repo, line + "\n"), repo).citations == []


def test_artifact_trees_are_the_scope_definition():
    assert ARTIFACT_TREES == ("data/", "artifacts/", "benchmarks/", "measurement/")


def test_scope_test_is_identical_for_relative_and_absolute(repo):
    """A citation must not change category by being spelled differently."""
    rel = check(write_registry(repo, "a: data/good_campaign_20260801/summary.json\n"), repo)
    ab = check(write_registry(
        repo, f"a: {repo}/data/good_campaign_20260801/summary.json\n"), repo)
    assert len(rel.citations) == len(ab.citations) == 1
    assert not rel.errors and not ab.errors


# -------------------------------------------------------------------- classification

def test_ephemeral_is_an_error_even_when_the_file_exists(repo, tmp_path):
    live = tmp_path / "live.json"
    live.write_text("{}")
    reg = write_registry(repo, f"evidence: /tmp/{live.name}\n")
    # point at a real scratch-rooted file
    reg.write_text(f"evidence: /tmp/{live.name}\n")
    res = check(reg, repo)
    (c,) = res.citations
    assert c.verdict == "EPHEMERAL"
    assert c.severity == "error"


def test_ephemeral_hint_distinguishes_still_there_from_already_gone(repo):
    reg = write_registry(repo, "evidence: /mnt/raid0/llm/tmp/definitely-not-here/x.json\n")
    (c,) = check(reg, repo).citations
    assert c.verdict == "EPHEMERAL"
    assert "already GONE" in c.hint
    assert "re-measured or demoted" in c.hint


def test_missing_is_an_error(repo):
    reg = write_registry(repo, "evidence: data/good_campaign_20260801/nope.json\n")
    (c,) = check(reg, repo).citations
    assert (c.verdict, c.severity) == ("MISSING", "error")


def test_outside_repo_is_an_error(repo, tmp_path, no_ephemeral_tmp):
    stray = tmp_path / "elsewhere"
    (stray / "data").mkdir(parents=True)
    (stray / "data" / "x.json").write_text("{}")
    reg = write_registry(repo, f"evidence: {stray}/data/x.json\n")
    (c,) = check(reg, repo).citations
    assert (c.verdict, c.severity) == ("OUTSIDE_REPO", "error")


def test_repo_membership_beats_the_scratch_heuristic(repo):
    """The ephemeral roots are a PROXY for "not in a repository". pytest builds its
    fixtures under /tmp, so this repo genuinely lives under a scratch root -- and its own
    contents must still validate. Precedence is deliberate, not incidental: if the proxy
    outranked demonstrated repo membership, a relocated checkout would fail everything.
    """
    assert str(repo).startswith("/tmp/"), "fixture no longer exercises the precedence"
    reg = write_registry(repo, "evidence: data/good_campaign_20260801/summary.json\n")
    (c,) = check(reg, repo).citations
    assert c.verdict == "OK"


def test_absolute_in_repo_warns_and_names_the_relative_form(repo):
    reg = write_registry(repo, f"evidence: {repo}/data/good_campaign_20260801/summary.json\n")
    (c,) = check(reg, repo).citations
    assert (c.verdict, c.severity) == ("ABSOLUTE_IN_REPO", "warn")
    assert "data/good_campaign_20260801/summary.json" in c.hint


def test_sibling_repo_warns_but_does_not_fail(repo, monkeypatch, no_ephemeral_tmp):
    import check_evidence_durability as m
    sib = repo.parent / "epyc-root"
    (sib / "data" / "compliance").mkdir(parents=True)
    (sib / "data" / "compliance" / "SUMMARY.md").write_text("x")
    monkeypatch.setattr(m, "SIBLING_REPO_ROOTS", (str(sib),))
    reg = write_registry(repo, f"src: {sib}/data/compliance/SUMMARY.md\n")
    res = check(reg, repo)
    (c,) = res.citations
    assert (c.verdict, c.severity) == ("SIBLING_REPO", "warn")
    assert not res.errors


# ---------------------------------------------------------------------- lost artifacts

def test_artifact_lost_marker_waives_the_error(repo):
    reg = write_registry(
        repo,
        "  source: /mnt/raid0/llm/tmp/buun-llama-cpp-src  "
        "# ARTIFACT LOST (build tree, not a measurement result) — recorded 2026-08-02\n")
    (c,) = check(reg, repo).citations
    assert c.verdict == "WAIVED_LOST"
    assert c.path == "/mnt/raid0/llm/tmp/buun-llama-cpp-src"   # provenance kept verbatim


def test_a_waiver_is_never_silent(repo):
    """The waiver is the obvious lever for making a red tree green, so it must stay
    visible: reported as a warning, never as OK, and caught by -W."""
    body = ("  source: /mnt/raid0/llm/tmp/gone  "
            "# ARTIFACT LOST (build tree, not a measurement result) — recorded 2026-08-02\n")
    reg = write_registry(repo, body)
    res = check(reg, repo)
    (c,) = res.citations
    assert c.severity == "warn"
    assert c.severity != "ok"
    assert len(res.warnings) == 1
    assert main([str(reg), "--repo", str(repo)]) == 0
    assert main([str(reg), "--repo", str(repo), "-W"]) == 1


def test_waiver_does_not_leak_to_other_citations_on_other_lines(repo):
    reg = write_registry(repo, (
        "  a: /mnt/raid0/llm/tmp/gone  # ARTIFACT LOST — recorded 2026-08-02\n"
        "  b: /mnt/raid0/llm/tmp/also-gone/x.json\n"))
    res = check(reg, repo)
    assert verdicts(res)["/mnt/raid0/llm/tmp/gone"] == "WAIVED_LOST"
    assert verdicts(res)["/mnt/raid0/llm/tmp/also-gone/x.json"] == "EPHEMERAL"
    assert len(res.errors) == 1


# --------------------------------------------------------------- the compliant path

def test_compliant_registry_is_silent(repo):
    """The guard must not forbid its own idiom: a fully migrated registry passes clean,
    with no errors AND no warnings."""
    reg = write_registry(repo, """
roles:
  a:
    evidence: data/good_campaign_20260801/summary.json
    # cross-checked against data/good_campaign_20260801/summary.json:8
    note: "scored rows in artifacts/run-{a,b}-scored.json"
""")
    res = check(reg, repo)
    assert [c.verdict for c in res.citations] == ["OK"] * 4
    assert not res.errors and not res.warnings
    assert main([str(reg), "--repo", str(repo)]) == 0


# --------------------------------------------------------------------- campaign docs

def test_campaign_missing_docs_is_reported(repo):
    reg = write_registry(repo, "evidence: data/undocumented_campaign/result.json\n")
    res = check(reg, repo)
    assert not res.errors
    assert [i["campaign"] for i in res.campaign_issues] == ["undocumented_campaign"]
    assert sorted(res.campaign_issues[0]["missing"]) == ["README.md", "SHA256SUMS"]


def test_campaign_docs_can_be_made_fatal(repo):
    reg = write_registry(repo, "evidence: data/undocumented_campaign/result.json\n")
    assert main([str(reg), "--repo", str(repo)]) == 0
    assert main([str(reg), "--repo", str(repo), "--require-campaign-docs"]) == 1


def test_documented_campaign_raises_no_issue(repo):
    reg = write_registry(repo, "evidence: data/good_campaign_20260801/summary.json\n")
    assert check(reg, repo).campaign_issues == []


# ------------------------------------------------------------------------------- cli

def test_exit_code_and_fix_hint(repo, capsys):
    reg = write_registry(repo, "evidence: /mnt/raid0/llm/tmp/run/summary.json\n")
    rc = main([str(reg), "--repo", str(repo), "--fix-hint"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "EPHEMERAL" in out
    assert "FIX HINTS" in out
    assert "data/<campaign>" in out
    assert "sha256sum" in out.lower()


def test_json_output_is_machine_readable(repo, capsys):
    reg = write_registry(repo, "evidence: /mnt/raid0/llm/tmp/run/summary.json\n")
    main([str(reg), "--repo", str(repo), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] == 1
    assert payload["citations"][0]["verdict"] == "EPHEMERAL"


def test_missing_registry_exits_2(tmp_path, capsys):
    assert main([str(tmp_path / "nope.yaml")]) == 2


def test_runs_standalone_as_a_script(repo):
    reg = write_registry(repo, "evidence: data/good_campaign_20260801/summary.json\n")
    r = subprocess.run([sys.executable, str(SCRIPT), str(reg), "--repo", str(repo)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "errors: 0" in r.stdout


# ----------------------------------------------------------------------- integration

@pytest.mark.skipif(not REAL_REGISTRY.exists(), reason="real registry not present")
def test_the_real_registry_has_no_durability_errors():
    """Regression guard for the 2026-08-02 migration: the master registry must never
    again cite evidence from scratch or from a path that does not exist."""
    res = check(REAL_REGISTRY, REAL_REPO)
    assert res.errors == [], "\n".join(
        f"L{c.line} {c.verdict} {c.path} -> {c.hint}" for c in res.errors)


@pytest.mark.skipif(not REAL_REGISTRY.exists(), reason="real registry not present")
def test_the_only_scratch_paths_left_are_recorded_losses():
    res = check(REAL_REGISTRY, REAL_REPO)
    scratch = [c for c in res.citations if c.path.startswith("/mnt/raid0/llm/tmp/")]
    assert scratch, "extraction is broken if it finds no scratch paths at all"
    assert all(c.verdict == "WAIVED_LOST" for c in scratch), [
        (c.line, c.verdict, c.path) for c in scratch if c.verdict != "WAIVED_LOST"]
