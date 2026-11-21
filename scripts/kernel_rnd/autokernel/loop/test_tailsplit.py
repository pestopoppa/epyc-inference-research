"""AKPL-P3: lanes emit candidate records through the real run_pool/iterate instead of entering the serialized tail."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autokernel.loop import loop, pipeline, tailsplit
from autokernel.loop.test_pipeline import _Champion, _Critic, _Planner, _control_disposition, _workers


def _pool(workers, iterations, handoff, snapshot, *, gate_calls):
    champion = _Champion()

    def gate_for(worker):
        def gate(h, p):
            gate_calls.append(worker.name)
            raise AssertionError("the in-process tail must not run when lanes hand off")
        return gate

    return pipeline.run_pool(
        workers=workers, make_planner=lambda w: _Planner(), make_critic=lambda w: _Critic(),
        build_context=dict, make_gate=gate_for, make_measure=gate_for,
        commit=lambda *a: champion.advance(), champion_head=champion.head,
        reset_to_champion=lambda w: champion.head(), record=lambda o: None,
        record_abandoned=_control_disposition, iterations=iterations,
        make_emitting_tail=lambda w, base: tailsplit.EmittingTail(worker=w, base=base, handoff=handoff,
                                                                   snapshot=snapshot)), champion


def test_lanes_hand_off_and_never_enter_the_tail(tmp_path):
    handed, gate_calls = [], []
    snaps = iter(f"{n:040x}" for n in range(1, 100))

    def snapshot(wt, base, paths, msg):
        assert paths == ["ggml/src/ggml-cuda/vecdotq.cuh"]
        return next(snaps)

    outcomes, champion = _pool(_workers(3), 6, lambda r: handed.append(r) or dict(r), snapshot,
                               gate_calls=gate_calls)
    assert len(outcomes) == 6 and {o.status for o in outcomes} == {loop.HANDED_OFF}
    assert gate_calls == [] and champion.advances == 0
    assert len(handed) == 6
    for o in outcomes:
        rec = o.handoff_record
        assert rec["schema"] == tailsplit.CANDIDATE_SCHEMA
        assert rec["base_sha"] == champion.head() and rec["hypothesis_id"] == "akm-q5-bit-deposit"
        assert rec["paths"] == ["ggml/src/ggml-cuda/vecdotq.cuh"]
        assert o.to_attempt()["handoff_record"]["patch_sha"] == rec["patch_sha"]


def test_without_factory_the_legacy_tail_runs():
    from autokernel.loop.test_pipeline import _drive
    outcomes, champion = _drive(workers=_workers(2), iterations=2)
    assert loop.HANDED_OFF not in {o.status for o in outcomes}


def test_run_flags_select_the_path():
    from autokernel.loop import run
    from types import SimpleNamespace as NS
    assert run._tail_split_enabled(NS(workers=4, legacy_tail=False))
    assert not run._tail_split_enabled(NS(workers=4, legacy_tail=True))
    assert not run._tail_split_enabled(NS(workers=1, legacy_tail=False))


def _repo(tmp_path):
    repo = tmp_path / "llama"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    g = lambda *a, **k: subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                                       capture_output=True, text=True, check=True, **k).stdout.strip()
    (repo / "ggml/src/ggml-cuda").mkdir(parents=True)
    (repo / "ggml/src/ggml-cuda/k.cu").write_text("k1\n")
    (repo / "docs.md").write_text("d1\n")
    g("add", "-A"); g("commit", "-qm", "base")
    return repo, g


def test_snapshot_commit_leaves_lane_index_and_head_alone(tmp_path):
    repo, g = _repo(tmp_path)
    base = g("rev-parse", "HEAD")
    (repo / "ggml/src/ggml-cuda/k.cu").write_text("k2\n")
    (repo / "stray.txt").write_text("not declared\n")
    status_before = g("status", "--porcelain")
    patch = tailsplit.snapshot_commit(repo, base, ["ggml/src/ggml-cuda/k.cu"], "cand")
    assert g("rev-parse", f"{patch}^") == base
    assert g("diff", "--name-only", base, patch).splitlines() == ["ggml/src/ggml-cuda/k.cu"]
    assert g("rev-parse", "HEAD") == base and g("status", "--porcelain") == status_before


def test_rebase_recheck_oracle_only_unless_measured_path_moves(tmp_path):
    repo, g = _repo(tmp_path)
    base = g("rev-parse", "HEAD")
    (repo / "ggml/src/ggml-cuda/k.cu").write_text("k1\nk-cand\n")
    patch = tailsplit.snapshot_commit(repo, base, ["ggml/src/ggml-cuda/k.cu"], "cand")
    rec = tailsplit.candidate_record(base=base, patch=patch, hypothesis={"mechanism_id": "AK-R"},
                                     paths=["ggml/src/ggml-cuda/k.cu"], lane="lane0")
    # champion advances on a doc file only -> oracle only, no re-timing
    (repo / "ggml/src/ggml-cuda/k.cu").write_text("k1\n")          # the lane's diff lives only in `patch`
    g("checkout", "-q", "--detach", base)
    (repo / "docs.md").write_text("d2\n")
    g("commit", "-qam", "doc keep")
    head1 = g("rev-parse", "HEAD")
    rb, ch = tailsplit.git_rebase_fns(repo)
    r1 = tailsplit.rebase_recheck(rec, head1, rebase=rb, changed_since_base=ch)
    assert r1["stages"] == ["build", "oracle-kld", "fold-check"] and r1["skip_stages"] == ["ab-timing"]
    assert r1["hypothesis_id"] == "AK-R" and r1["base_sha"] == head1
    assert g("rev-parse", f"{r1['patch_sha']}^") == head1           # built on the base it will land on
    assert r1["rebase_of"]["candidate"] == rec["id"]
    # champion advances on a measured kernel file -> timing re-runs
    (repo / "ggml/src/ggml-cuda/other.cu").write_text("o\n")
    g("add", "-A"); g("commit", "-qm", "kernel keep")
    head2 = g("rev-parse", "HEAD")
    r2 = tailsplit.rebase_recheck(rec, head2, rebase=rb, changed_since_base=ch)
    assert r2["stages"] == ["build", "oracle-kld", "ab-timing", "fold-check"]


def test_chain_handoff_spools_and_dedups(tmp_path):
    calls = []
    h = tailsplit.ChainHandoff(spool=tmp_path, scaffold=lambda p: calls.append(p) or {"rc": 0})
    rec = tailsplit.candidate_record(base="a" * 40, patch="b" * 40, hypothesis={"mechanism_id": "AK-D"},
                                     paths=["x"], lane="l0")
    out = h(rec)
    assert Path(out["record_path"]).exists() and out["scaffold"] == {"rc": 0}
    again = h(dict(rec, emitted_at="later"))
    assert again["duplicate_of"] == rec["id"] and len(calls) == 1
    bad = tailsplit.ChainHandoff(spool=tmp_path / "b", scaffold=lambda p: (_ for _ in ()).throw(RuntimeError("x")))
    assert "scaffold_error" in bad(rec)
    assert json.loads(Path(out["record_path"]).read_text())["patch_sha"] == "b" * 40


# ---- custody: failure injection on the hand-off path (test_discard_invariant review, AKPL-P3) -----------------

def test_snapshot_failure_is_a_contained_lane_error_and_hands_nothing_off():
    handed, gate_calls = [], []

    def snapshot(*a):
        raise OSError("disk full")

    with pytest.raises(loop.RunAborted):        # 3 consecutive pool-wide errors trip the breaker, as for any lane_error
        _pool(_workers(1), 3, lambda r: handed.append(r) or dict(r), snapshot, gate_calls=gate_calls)
    assert handed == [] and gate_calls == []


def test_handoff_failure_is_a_contained_lane_error():
    gate_calls, seen = [], []

    def handoff(rec):
        seen.append(rec["patch_sha"])
        raise RuntimeError("spool unwritable")

    outcomes, _ = None, None
    with pytest.raises(loop.RunAborted):
        _pool(_workers(1), 3, handoff, lambda *a: "e" * 40, gate_calls=gate_calls)
    assert len(seen) == 3 and gate_calls == []           # the record was formed; nothing measured in-process


def test_one_failed_handoff_does_not_lose_the_others():
    calls = {"n": 0}
    handed = []

    def handoff(rec):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient")
        handed.append(rec)
        return dict(rec)

    snaps = iter(f"{n:040x}" for n in range(1, 100))
    outcomes, _ = _pool(_workers(1), 4, handoff, lambda *a: next(snaps), gate_calls=[])
    assert [o.status for o in outcomes].count(loop.HANDED_OFF) == 3
    assert [o.status for o in outcomes].count("lane_error") == 1 and len(handed) == 3
