"""Custody fixtures for CS-4.STT2. Every seal root is a pytest tmp_path — never the real
default. Synthetic strings only."""
from __future__ import annotations

import hashlib
import json
import os
import stat

import pytest

from scripts.speech.stt_bakeoff import custody as C
from scripts.speech.stt_bakeoff import scorer as S


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rows():
    out = []
    for i, (ref, hyp) in enumerate([("one two three", "one two three"),
                                    ("four five six", "four five sex"),
                                    ("seven eight", "seven eight")]):
        out.append({"utterance_id": f"conf-utt-{i}", "pcm_sha256": sha(f"pcm{i}"),
                    "reference_raw": ref, "hypothesis_raw": hyp, "language": "en",
                    "set_id": "private-set", "reference_track": "standard",
                    "content_stratum": "hesitation_conversational"})
    return out


def result(stratum="confirmation", arm_id="armA"):
    m = {"schema_version": S.RAW_MANIFEST_SCHEMA_VERSION, "corpus_id": "cs4-human",
         "stratum": stratum, "language_mode": "forced", "segmentation_mode": "none",
         "arm": {"arm_id": arm_id, "arm_kind": S.ARM_KIND_OFFLINE, "runner_id": "r"},
         "rows": rows()}
    return S.score_arm(m, column="v1", bootstrap_resamples=100, bootstrap_seed=7)


@pytest.fixture
def seal_root(tmp_path):
    assert C.git_tree_containing(tmp_path) is None, "tmp_path must be outside git"
    return tmp_path / "sealed"


def test_default_seal_root_is_real_custody_location_not_used_in_tests():
    assert C.DEFAULT_SEAL_ROOT == "/mnt/raid0/llm/data/stt-cs4-sealed"


# --- two-output split -------------------------------------------------------------

def test_two_output_split(seal_root):
    res = result()
    agg, receipt = C.seal_scoring_run(res, seal_root=seal_root)
    assert set(agg) == set(C.AGGREGATE_FIELDS)
    assert agg["n"] == 3 and agg["stratum"] == "confirmation"
    assert agg["wer"] == pytest.approx(100.0 / 8)
    assert agg["corpus_key"].startswith("cs4-human@") and len(agg["ci"]) == 2
    blob = json.dumps(agg)
    for u in res["per_utterance"]:
        assert u["utterance_id"] not in blob and u["pcm_sha256"] not in blob
    sealed = C.read_sealed(receipt["sealed_path"], reader="operator", seal_root=seal_root)
    assert sealed[0]["kind"] == "stt_bakeoff.sealed_per_utterance/v1"
    assert [r["utterance_id"] for r in sealed[1:]] == [f"conf-utt-{i}" for i in range(3)]


def test_sealed_permissions(seal_root):
    _, receipt = C.seal_scoring_run(result(), seal_root=seal_root)
    p = receipt["sealed_path"]
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(seal_root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode) == 0o700
    for name in (C.LOG_FILE, C.SUMS_FILE, C.COUNTERS_FILE):
        assert stat.S_IMODE(os.stat(seal_root / name).st_mode) == 0o600


def test_sha256sums_lists_sealed_file(seal_root):
    _, receipt = C.seal_scoring_run(result(), seal_root=seal_root)
    sums = (seal_root / C.SUMS_FILE).read_text().splitlines()
    rel = os.path.relpath(receipt["sealed_path"], seal_root)
    assert f"{receipt['sealed_sha256']}  {rel}" in sums
    assert C.file_sha256(receipt["sealed_path"]) == receipt["sealed_sha256"]


def test_tampered_sealed_file_refused_on_read(seal_root):
    _, receipt = C.seal_scoring_run(result(), seal_root=seal_root)
    os.chmod(receipt["sealed_path"], 0o600)
    with open(receipt["sealed_path"], "a") as fh:
        fh.write("{}\n")
    with pytest.raises(C.CustodyRefused):
        C.read_sealed(receipt["sealed_path"], reader="audit", seal_root=seal_root)


# --- counter and log --------------------------------------------------------------

def test_reuse_counter_increments_per_scoring_and_logs(seal_root):
    a1, r1 = C.seal_scoring_run(result(arm_id="armA"), seal_root=seal_root)
    a2, r2 = C.seal_scoring_run(result(arm_id="armB"), seal_root=seal_root)
    a3, _ = C.seal_scoring_run(result(arm_id="armA"), seal_root=seal_root)
    assert [a1["confirmation_reuse_n"], a2["confirmation_reuse_n"],
            a3["confirmation_reuse_n"]] == [1, 2, 3]
    assert r1["sealed_path"] != r2["sealed_path"]
    assert C.read_reuse_counter(a1["corpus_key"], seal_root=seal_root) == 3
    log = [json.loads(l) for l in (seal_root / C.LOG_FILE).read_text().splitlines()]
    assert [e["confirmation_reuse_n"] for e in log] == [1, 2, 3]
    assert [e["arm_id"] for e in log] == ["armA", "armB", "armA"]
    assert all(e["event"] == "confirmation_scoring" for e in log)
    assert len((seal_root / C.SUMS_FILE).read_text().splitlines()) == 3


def test_counter_is_per_corpus(seal_root):
    C.seal_scoring_run(result(), seal_root=seal_root)
    res = result()
    res["records"][0]["corpus_key"] = "other-corpus@abcdefabcdef"
    agg, _ = C.seal_scoring_run(res, seal_root=seal_root)
    assert agg["confirmation_reuse_n"] == 1


# --- three refusal cases (+ variants) ---------------------------------------------

def test_refusal_1_seal_root_inside_git_tree(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(C.SealPathInGitTree):
        C.seal_scoring_run(result(), seal_root=repo / "deep" / "sealed")
    assert not (repo / "deep").exists()


def test_refusal_1b_git_worktree_file_marker(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: /elsewhere\n")
    with pytest.raises(C.SealPathInGitTree):
        C.seal_scoring_run(result(), seal_root=wt / "sealed")


def test_refusal_1c_symlink_into_git_tree(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "data").mkdir()
    link = tmp_path / "innocent"
    link.symlink_to(repo / "data")
    with pytest.raises(C.SealPathInGitTree):
        C.seal_scoring_run(result(), seal_root=link / "sealed")


@pytest.mark.parametrize("consumer", ["planner", "autokernel", "AutoPilot", " Planner "])
def test_refusal_2_forbidden_consumer(seal_root, consumer):
    with pytest.raises(C.ForbiddenConsumer):
        C.seal_scoring_run(result(), seal_root=seal_root, consumer=consumer)
    assert not seal_root.exists()  # refused before anything was written or counted


def test_refusal_2b_forbidden_reader_of_sealed_file(seal_root):
    _, receipt = C.seal_scoring_run(result(), seal_root=seal_root)
    with pytest.raises(C.ForbiddenConsumer):
        C.read_sealed(receipt["sealed_path"], reader="autokernel", seal_root=seal_root)
    with pytest.raises(C.CustodyRefused):
        C.read_sealed(receipt["sealed_path"], reader="dashboard", seal_root=seal_root)


@pytest.mark.parametrize("field", ["utterance_id", "per_utterance", "errors",
                                   "hypothesis_normalised"])
def test_refusal_3_per_utterance_in_aggregate(field):
    with pytest.raises(C.PerUtteranceLeak):
        C.loop_visible_aggregate(result(), extra_fields=[field])


def test_refusal_3b_leak_detected_by_value():
    res = result()
    agg = C.loop_visible_aggregate(res)
    agg["note"] = res["per_utterance"][0]["utterance_id"]
    with pytest.raises(C.PerUtteranceLeak):
        C.assert_no_per_utterance(agg, res["per_utterance"])
    agg2 = C.loop_visible_aggregate(res)
    agg2["detail"] = [{"reference_normalised": "x"}]
    with pytest.raises(C.PerUtteranceLeak):
        C.assert_no_per_utterance(agg2, res["per_utterance"])


def test_refusal_3c_confirmation_written_unsealed(tmp_path):
    with pytest.raises(C.UnsealedConfirmationWrite):
        C.write_per_utterance(result("confirmation"), tmp_path / "out.jsonl")
    with pytest.raises(C.UnsealedConfirmationWrite):
        C.write_per_utterance(result("private/2026q4"), tmp_path / "out.jsonl")
    with pytest.raises(C.ForbiddenConsumer):
        C.write_per_utterance(result("confirmation"), tmp_path / "o.jsonl",
                              consumer="planner")
    assert not (tmp_path / "out.jsonl").exists()


def test_public_stratum_per_utterance_write_allowed(tmp_path):
    p = C.write_per_utterance(result("selection"), tmp_path / "sel.jsonl")
    assert len(p.read_text().splitlines()) == 3


def test_non_sealed_stratum_not_sealed(seal_root):
    with pytest.raises(C.NotASealedStratum):
        C.seal_scoring_run(result("selection"), seal_root=seal_root)


def test_is_sealed_stratum():
    assert C.is_sealed_stratum("confirmation") and C.is_sealed_stratum("Private:x")
    assert not C.is_sealed_stratum("selection") and not C.is_sealed_stratum("public")


# --- public manifest (OP-PUBLICREPOS) ---------------------------------------------

def test_public_manifest_carries_hashes_only():
    pm = C.public_manifest(rows())
    assert all(set(e) == {"utterance_id", "pcm_sha256", "reference_text_sha256"}
               for e in pm)
    assert pm[1]["reference_text_sha256"] == sha("four five six")
    blob = json.dumps(pm)
    assert "four five" not in blob and "sex" not in blob


def test_stratum_sha256sums(tmp_path):
    d = tmp_path / "stratum"
    (d / "audio").mkdir(parents=True)
    (d / "audio" / "a.pcm").write_bytes(b"\x00\x01")
    (d / "refs.jsonl").write_text("{}\n")
    sums = C.write_stratum_sha256sums(d).read_text().splitlines()
    h_audio = hashlib.sha256(b"\x00\x01").hexdigest()
    h_refs = hashlib.sha256(b"{}\n").hexdigest()
    assert sums == [f"{h_audio}  audio/a.pcm", f"{h_refs}  refs.jsonl"]
