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


def test_tests_never_spool_into_the_live_candidate_spool():
    # The live spool is counted by epyc-root flow_status.py as real candidates; the conftest
    # host-isolation fixture must redirect it (in-process and for subprocesses).
    import os
    live = Path("/mnt/raid0/llm/tmp/akchain-spool")
    assert tailsplit.SPOOL != live
    assert os.environ.get("AKCHAIN_SPOOL") and Path(os.environ["AKCHAIN_SPOOL"]) != live


def _seed_inbox(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / "BWS-LM.md").write_text('## BWS-LM\n<!-- journal-key: {"op_family": "lm-head-gemv", "device": "gpu", '
                                     '"shape_class": {"m": 248320, "k": 5120}} -->\n')
    return inbox


def _rec(tmp_path, hid, **kw):
    h = loop.Hypothesis(hid, "statement", "f", "ggml/src/ggml-cuda/a.cu", "sym")
    return tailsplit.candidate_record(base="b" * 40, patch="p" * 40, hypothesis=h, paths=["a"], lane="l",
                                      inbox=_seed_inbox(tmp_path), **kw)


def test_candidate_citing_a_bws_seed_gets_its_shape_class(tmp_path):
    assert _rec(tmp_path, "BWS-LM")["shape_class"] == {"op_family": "MUL_MAT", "dims": {"k": [5120], "m": [248320]}}


def test_candidate_carries_the_journal_key_and_its_op_perf_class_derives_from_it(tmp_path):
    rec = _rec(tmp_path, "BWS-LM")
    assert rec["journal_key"] == {"op_family": "lm-head-gemv", "device": "gpu", "shape_class": {"m": 248320, "k": 5120}}
    assert rec["shape_class"] == tailsplit.journal_keys.op_perf_class(rec["journal_key"])


def test_candidate_own_journal_key_beats_statement_and_seed(tmp_path):
    own = {"op_family": "mmid-moe", "shape_class": {"k": 2048}}
    h = {"mechanism_id": "BWS-LM", "statement": '<!-- journal-key: {"op_family": "dense-q8-gemv"} -->', "journal_key": own}
    assert tailsplit.candidate_record(base="b", patch="p" * 9, hypothesis=h, paths=[], lane="l",
                                      inbox=_seed_inbox(tmp_path))["journal_key"] == own
    h.pop("journal_key")   # statement comment is next
    assert tailsplit.candidate_record(base="b", patch="p" * 9, hypothesis=h, paths=[], lane="l")["journal_key"]["op_family"] == "dense-q8-gemv"


def test_candidate_overlapping_a_measured_negative_is_marked_and_refused_unless_it_reopens(tmp_path):
    inbox = _seed_inbox(tmp_path)   # <store>/inbox; the negatives sidecar lives in <store>
    tailsplit.journal_keys.key_entry(tmp_path, "akm-lm-null", {"op_family": "lm-head-gemv", "shape_class": {"m": [200000, 300000]}})
    assert _rec(tmp_path, "BWS-LM")["suppressed_by"] == "akm-lm-null"
    h = {"mechanism_id": "BWS-LM", "statement": "s", "reopens": ["akm-lm-null"]}
    assert "suppressed_by" not in tailsplit.candidate_record(base="b", patch="p" * 9, hypothesis=h, paths=[], lane="l", inbox=inbox)
    handed = []
    tail = tailsplit.EmittingTail(worker=type("W", (), {"worktree": ".", "name": "l"})(), base="b" * 40, inbox=inbox,
                                  handoff=handed.append, snapshot=lambda *a: "p" * 40)
    h2 = loop.Hypothesis("BWS-LM", "statement", "f", "ggml/src/ggml-cuda/a.cu", "sym")
    with pytest.raises(tailsplit.SuppressedByNegative, match="akm-lm-null"):
        with tail(h2, ["a"]):
            pass
    assert handed == []   # nothing reached the chain scaffold


def test_candidate_with_its_own_shape_class_keeps_it(tmp_path):
    own = {"op_family": "MUL_MAT", "dims": {"k": [1]}}
    assert _rec(tmp_path, "BWS-LM", shape_class=own)["shape_class"] == own
    assert "shape_class" not in _rec(tmp_path, "unrelated-idea")


# --- AK-SURFACE-SERVING-1: planner choice -> serving-knob candidate, end to end ---
def _knob_tail(tmp_path, snapshot=None, recipe=True):
    import copy
    from autokernel.loop import seed, serving
    from autokernel.loop.test_serving_knobs import MODEL, REGISTRY
    reg = copy.deepcopy(REGISTRY)
    reg["server_mode"]["architect_critic"] = reg["server_mode"].pop("srv")
    seed.install(tmp_path)   # the shipped *.serving-knob.json lands in <store>/inbox beside its planner-visible .md
    rcp = serving.Recipe(name="r", model=MODEL, np=4, batch=2048, ubatch=2048, kv_unified=False, temperature=1.0,
                         spec_decode={"type": "draft-dflash", "drafter": "/m/dflash.gguf", "ngld": 99, "draft_n_max": 8})
    worker = type("W", (), {"worktree": ".", "name": "lane0", "build_dir": "/k/gpu"})()
    handed, champion = [], "c" * 40
    return (tailsplit.EmittingTail(worker=worker, base=champion, inbox=tmp_path / "inbox", registry=reg,
                                   recipe=rcp if recipe else None, handoff=lambda r: handed.append(dict(r)) or dict(r),
                                   snapshot=snapshot or (lambda *a: pytest.fail("a knob screen has no patch to snapshot"))),
            handed, champion, rcp)


def test_inbox_serving_knob_seed_becomes_a_serving_knob_candidate_record(tmp_path):
    tail, handed, champion, rcp = _knob_tail(tmp_path)
    assert list((tmp_path / "inbox").glob("*.serving-knob.md")) and list((tmp_path / "inbox").glob("*.serving-knob.json"))
    h = loop.Hypothesis("serving-knob:architect-critic-serving-20261124:batch-ubatch-1024", "bigger batches",
                        "no aggregate tok/s gain", "serving", "batch")
    with pytest.raises(tailsplit.HandedOff) as hand:
        with tail(h, []):
            pass
    rec = hand.value.record
    assert handed == [rec] or handed[0]["id"] == rec["id"]
    assert rec["schema"] == tailsplit.CANDIDATE_SCHEMA and rec["kind"] == "serving-knob" and rec["paths"] == []
    assert rec["base_sha"] == rec["patch_sha"] == champion          # a knob screen has no patch: both are the champion
    assert {f for f, _, _ in map(tuple, rec["argv_delta"])} == {"-b", "-ub"} and rec["base_recipe_hash"] == rcp.recipe_hash
    assert rec["server"] == "architect_critic" and rec["screening_only"] and rec["delta"]["knobs"] == {"batch": 1024, "ubatch": 1024}
    assert rec["arm_map"] == {"control": "control", "batch-ubatch-1024": "batch-ubatch-1024"}
    assert rec["steady_results"].startswith("@screening-row:") and rec["floor_result"].startswith("@screening-row:")
    assert set(rec["burst_results"]) == {"control", "batch-ubatch-1024"}   # named placeholders the screening row fills


def test_serving_knob_route_refuses_what_it_cannot_render_and_leaves_other_hypotheses_alone(tmp_path):
    tail, _, _, _ = _knob_tail(tmp_path)
    mk = lambda mid: loop.Hypothesis(mid, "s", "f", "serving", "x")   # noqa: E731
    with pytest.raises(loop.TailRefused, match="not in"):
        tail.knob(mk("serving-knob:no-such-seed:a"))
    with pytest.raises(loop.TailRefused, match="not an offered option"):
        tail.knob(mk("serving-knob:architect-critic-serving-20261124:no-such-arm"))
    assert tail.knob(mk("akm-some-kernel-idea")) is None
    tail2, _, _, _ = _knob_tail(tmp_path / "again", recipe=False)
    with pytest.raises(loop.TailRefused, match="serving recipe"):
        tail2.knob(mk("serving-knob:architect-critic-serving-20261124:batch-ubatch-1024"))


def test_negatives_are_always_consulted_sidecar_overlaid_on_the_seeded_table(tmp_path):
    key = {"op_family": "lm-head-gemv", "shape_class": {"m": 248320}}
    h = {"mechanism_id": "x", "statement": "s", "journal_key": key}
    kw = dict(base="b", patch="p" * 9, hypothesis=h, paths=[], lane="l")
    seeded = [m for m, k in tailsplit.journal_keys.negatives().items() if tailsplit.journal_keys.overlaps(key, k)]
    # no inbox, no negatives argument: the seeded negatives are still asked (no conditional skip)
    assert tailsplit.candidate_record(**kw).get("suppressed_by") == (sorted(seeded)[0] if seeded else None)
    assert tailsplit.candidate_record(**kw, negatives={}).get("suppressed_by") is None   # an explicit empty table is honoured
    tailsplit.journal_keys.key_entry(tmp_path, "akm-sidecar-null", {"op_family": "lm-head-gemv", "shape_class": {"m": [1, 999999]}})
    (tmp_path / "inbox").mkdir()
    assert tailsplit.candidate_record(**kw, inbox=tmp_path / "inbox").get("suppressed_by") in {"akm-sidecar-null", *seeded}
    assert tailsplit.candidate_record(**kw, inbox=tmp_path / "inbox", negatives={}).get("suppressed_by") is None
