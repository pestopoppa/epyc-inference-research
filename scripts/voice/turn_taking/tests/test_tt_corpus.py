"""CS-45 conformance suite: build/verify on a synthetic corpus, then refusal (mutation) cases.

Run file-scoped:  /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -I -m pytest -q test_tt_corpus.py
Set TT_REAL_CORPUS=<dir> to run the same refusal cases against the committed corpus (copied to tmp).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TT = HERE.parent
sys.path.insert(0, str(TT))
sys.path.insert(0, str(HERE))
import fake_renders  # noqa: E402
import tt_audio as A  # noqa: E402
import tt_corpus as C  # noqa: E402
import tt_silero as TS  # noqa: E402
import tt_verify as V  # noqa: E402

REPO = TT.parents[2]


@pytest.fixture(scope="session")
def fake_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake")
    D, _ = fake_renders.build_fake_corpus(root)
    return D


def real_corpus():
    p = os.environ.get("TT_REAL_CORPUS")
    return Path(p) if p else None


@pytest.fixture(params=["fake", "real"])
def corpus(request, fake_corpus, tmp_path):
    if request.param == "real":
        rc = real_corpus()
        if rc is None:
            pytest.skip("TT_REAL_CORPUS not set")
        src = rc
    else:
        src = fake_corpus
    dst = tmp_path / "c"
    # hard-link audio (cheap), copy everything small; mutations replace files, never write through links
    def _lnk(a, b):
        # link ONLY the (never-rewritten) audio; everything else is copied because seal()/mutations rewrite
        # corpus.json / manifests in place and a shared inode would corrupt the source corpus
        if str(a).endswith((".wav", ".tar")):
            try:
                os.link(a, b)
                return
            except OSError:
                pass
        shutil.copy2(a, b)

    shutil.copytree(src, dst, copy_function=_lnk)
    return dst


def failing(D: Path):
    return {c["check"] for c in V.verify(D).checks if not c["ok"]}


def reseal(D: Path):
    V.seal(D)


def mutate_recording(D: Path, rec: dict, fn):
    """Rewrite one WAV member of its archive with fn(Pcm)->Pcm (archive rebuilt; hash-linked originals untouched)."""
    import tarfile

    arc, member = rec["path"].split("#")
    p = D / arc
    with tarfile.open(p) as t:
        members = {m.name: t.extractfile(m).read() for m in t.getmembers()}
    members[member] = A.wav_bytes(fn(A.read_wav_bytes(members[member])))
    p.unlink()
    A.write_tar(p, members)


def unlink_write(p: Path, data: bytes):
    p.unlink()
    p.write_bytes(data)


def test_baseline_passes(corpus):
    assert failing(corpus) == set()


def test_pair_count_floor(corpus):
    n = sum(1 for _ in open(corpus / "manifest" / "pairs.jsonl"))
    assert n >= 100


def test_altered_clip_source_refused_with_stale_seal(corpus):
    rec = next(r for r in map(json.loads, open(corpus / "manifest" / "recordings.jsonl")) if r["role"] == "user")

    def flip(pc):
        s = pc.samples.copy()
        s[(rec["trim"][0] + rec["trim"][1]) // 2 - 3, 0] += 1000
        return A.Pcm(s, pc.sample_rate)

    mutate_recording(corpus, rec, flip)
    f = failing(corpus)
    assert "file_digests" in f and "recordings_pcm_bound" in f


def test_altered_clip_source_refused_even_if_resealed(corpus):
    rec = next(r for r in map(json.loads, open(corpus / "manifest" / "recordings.jsonl")) if r["role"] == "user")

    def flip(pc):
        s = pc.samples.copy()
        s[(rec["trim"][0] + rec["trim"][1]) // 2 - 3, 0] += 1000
        return A.Pcm(s, pc.sample_rate)

    mutate_recording(corpus, rec, flip)
    reseal(corpus)
    f = failing(corpus)
    assert "recordings_pcm_bound" in f and "clips_reproduce_pcm_sha256" in f


def test_cut_point_moved_refused(corpus):
    rows = [json.loads(x) for x in open(corpus / "manifest" / "clips.jsonl")]
    for r in rows:
        if r["user"]["cut"]:
            r["user"]["cut"] -= 800
            break
    (corpus / "manifest" / "clips.jsonl").unlink()
    (corpus / "manifest" / "clips.jsonl").write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n")
    reseal(corpus)
    assert "clips_reproduce_pcm_sha256" in failing(corpus)


def test_unmatched_trailing_silence_refused(corpus, monkeypatch):
    monkeypatch.setattr(C, "TRAIL", C.TRAIL + 1600)
    assert failing(corpus) & {"clips_reproduce_pcm_sha256", "pairs_are_splices_of_one_recording"}


def test_speaker_crossing_splits_refused(corpus):
    rows = [json.loads(x) for x in open(corpus / "manifest" / "clips.jsonl")]
    sel = [r for r in rows if r["split"] == "selection"]
    conf = [r for r in rows if r["split"] == "confirmation"]
    victim = conf[0]["speaker_id"]
    for r in rows:
        if r["pair_id"] == sel[0]["pair_id"]:
            r["speaker_id"] = victim
    p = corpus / "manifest" / "clips.jsonl"
    p.unlink()
    p.write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n")
    reseal(corpus)
    assert "selection_confirmation_disjoint" in failing(corpus) or "pairs_are_splices_of_one_recording" in failing(corpus)


def test_dialogue_crossing_splits_refused(corpus):
    rows = [json.loads(x) for x in open(corpus / "manifest" / "clips.jsonl")]
    d0 = next(r for r in rows if r["split"] == "selection")["dialogue_id"]
    for r in rows:
        if r["split"] == "confirmation":
            r["dialogue_id"] = d0
            break
    p = corpus / "manifest" / "clips.jsonl"
    p.unlink()
    p.write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n")
    reseal(corpus)
    assert "selection_confirmation_disjoint" in failing(corpus)


def test_label_swap_refused(corpus):
    rows = [json.loads(x) for x in open(corpus / "manifest" / "clips.jsonl")]
    for r in rows:
        if r["label"] == "incomplete":
            r["label"] = "complete"
            break
    p = corpus / "manifest" / "clips.jsonl"
    p.unlink()
    p.write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n")
    reseal(corpus)
    assert failing(corpus) & {"label_contract_conformance", "pairs_are_splices_of_one_recording"}


def test_dropping_pairs_below_100_refused(corpus):
    rows = [json.loads(x) for x in open(corpus / "manifest" / "pairs.jsonl")]
    keep = rows[:99]
    ids = {c for r in keep for c in r["clips"]}
    clips = [json.loads(x) for x in open(corpus / "manifest" / "clips.jsonl")]
    for fn, data in (("pairs.jsonl", keep), ("clips.jsonl", [c for c in clips if c["clip_id"] in ids])):
        p = corpus / "manifest" / fn
        p.unlink()
        p.write_text("\n".join(json.dumps(x, sort_keys=True) for x in data) + "\n")
    reseal(corpus)
    assert "min_100_english_pairs" in failing(corpus)


def test_mixed_stratum_refused(corpus):
    rows = [json.loads(x) for x in open(corpus / "manifest" / "clips.jsonl")]
    rows[0]["stratum"] = "human_recorded"
    p = corpus / "manifest" / "clips.jsonl"
    p.unlink()
    p.write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n")
    reseal(corpus)
    assert "strata_separate_and_advance_policy" in failing(corpus)


def test_advance_policy_widened_refused(corpus):
    m = json.loads((corpus / "corpus.json").read_text())
    m["advance_eligible_strata"] = ["human_recorded", "synthesized"]
    (corpus / "corpus.json").unlink()
    (corpus / "corpus.json").write_text(json.dumps(m))
    assert "strata_separate_and_advance_policy" in failing(corpus) or "file_digests" in failing(corpus)


def test_wrong_silero_hash_in_pins_refused(corpus):
    p = json.loads((corpus / "pins.json").read_text())
    p["silero_vad"]["sha256"] = "0" * 64
    (corpus / "pins.json").unlink()
    (corpus / "pins.json").write_text(json.dumps(p))
    reseal(corpus)
    assert "instrument_pins_recorded_and_hash_checked" in failing(corpus)


def test_swapped_silero_file_refused_by_loader_and_gate(corpus, tmp_path):
    f = corpus / "pins" / "silero_vad_v6.onnx"
    b = bytearray(f.read_bytes())
    b[1000] ^= 0xFF
    bad = tmp_path / "silero_bad.onnx"
    bad.write_bytes(bytes(b))
    with pytest.raises(TS.PinMismatch):
        TS.PinnedSilero(bad)
    unlink_write(f, bytes(b))
    reseal(corpus)
    assert "instrument_pins_recorded_and_hash_checked" in failing(corpus)


def test_tampered_scorer_patch_refused(corpus):
    pf = next((corpus / "scorer_patches").rglob("*.patch"))
    unlink_write(pf, pf.read_bytes() + b"\n# tamper\n")
    reseal(corpus)
    assert "instrument_pins_recorded_and_hash_checked" in failing(corpus)


def test_external_audio_copied_in_refused(corpus):
    ext = json.loads((corpus / "external_crosscheck.json").read_text())
    ext["audio_copied_into_corpus"] = True
    (corpus / "external_crosscheck.json").unlink()
    (corpus / "external_crosscheck.json").write_text(json.dumps(ext))
    reseal(corpus)
    assert "external_material_hashes_and_ids_only" in failing(corpus)
    (corpus / "audio" / "turnbench").mkdir(parents=True)
    (corpus / "audio" / "turnbench" / "x.wav").write_bytes(b"RIFF")
    reseal(corpus)
    assert "external_material_hashes_and_ids_only" in failing(corpus)


def test_pair_invariants_catch_non_splice():
    rng = np.random.default_rng(1)
    rec = (rng.standard_normal(40000) * 3000).astype(np.int16)
    asst = (rng.standard_normal(20000) * 3000).astype(np.int16)
    a, la = C.assemble_clip(rec, (0, 40000), None, asst, "ended")
    b, lb = C.assemble_clip(rec, (0, 40000), 20000, asst, "ended")
    assert C.check_pair_invariants("CUT_IC", a, la, None, b, lb, 20000) == []
    b2 = b.copy()
    b2[lb.user_onset + 100, 0] ^= 1
    assert C.check_pair_invariants("CUT_IC", a, la, None, b2, lb, 20000)
    b3 = b.copy()
    b3[lb.user_end + 5, 0] = 7
    assert C.check_pair_invariants("CUT_IC", a, la, None, b3, lb, 20000)


def test_silero_streaming_equals_offline():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(16000 * 2) * 0.1).astype(np.float32)
    m = TS.PinnedSilero()
    off = m.speech_probs(x)
    st = m.stream()
    xs = np.pad(x, (0, 512 - len(x) % 512))
    on = np.array([st.feed(xs[i : i + 512]) for i in range(0, len(xs), 512)])
    assert np.abs(off - on).max() < 1e-6


# ------------------------------------------------------------------ reserve stratum
def jl(p: Path):
    return [json.loads(x) for x in open(p)]


def wjl(p: Path, rows):
    p.unlink()
    p.write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n")


def wjson(p: Path, obj):
    p.unlink()
    p.write_text(json.dumps(obj, indent=1))


def test_reserve_present_and_sealed(corpus):
    assert len(jl(corpus / "manifest" / "reserve" / "pairs.jsonl")) >= 20
    assert json.loads((corpus / "corpus.json").read_text())["reserve"]["policy_id"] == "cs45-reserve-v1"


def _reject(corpus, n):
    pairs = jl(corpus / "manifest" / "pairs.jsonl")
    return pairs[:n]


def test_reserve_not_counted_without_backfill(corpus):
    # six review rejections leave 99 effective pairs; 20+ reserve pairs exist but are not counted
    rej = _reject(corpus, len(jl(corpus / "manifest" / "pairs.jsonl")) - 99)
    bf = json.loads((corpus / "manifest" / "backfill.json").read_text())
    bf["rejections"] = [{"pair_id": p["pair_id"], "reviewer": "test", "date": "2026-10-20", "reason": "t"} for p in rej]
    wjson(corpus / "manifest" / "backfill.json", bf)
    reseal(corpus)
    assert "min_100_english_pairs" in failing(corpus)


def test_recorded_backfill_restores_count(corpus):
    pairs = jl(corpus / "manifest" / "pairs.jsonl")
    res = jl(corpus / "manifest" / "reserve" / "pairs.jsonl")
    n_rej = len(pairs) - 99
    rej, bfs, used = [], [], set()
    for p in pairs:
        if len(rej) == n_rej:
            break
        cand = next((r for r in res if (r["family"], r["split"]) == (p["family"], p["split"]) and r["pair_id"] not in used), None)
        if cand is None:
            continue
        used.add(cand["pair_id"])
        rej.append({"pair_id": p["pair_id"], "reviewer": "test", "date": "2026-10-20", "reason": "t"})
        if len(bfs) < 1:
            bfs.append({"reserve_pair_id": cand["pair_id"], "replaces_pair_id": p["pair_id"]})
    bf = json.loads((corpus / "manifest" / "backfill.json").read_text())
    bf["rejections"], bf["backfills"] = rej, bfs
    wjson(corpus / "manifest" / "backfill.json", bf)
    reseal(corpus)
    assert failing(corpus) == set()  # 99 + 1 backfill = 100 effective


def _with_backfill(corpus, reserve_pick, main_pick):
    bf = json.loads((corpus / "manifest" / "backfill.json").read_text())
    bf["rejections"] = [{"pair_id": main_pick["pair_id"], "reviewer": "test", "date": "2026-10-20", "reason": "t"}]
    bf["backfills"] = [{"reserve_pair_id": reserve_pick["pair_id"], "replaces_pair_id": main_pick["pair_id"]}]
    wjson(corpus / "manifest" / "backfill.json", bf)
    reseal(corpus)


def test_backfill_without_rejection_refused(corpus):
    pairs, res = jl(corpus / "manifest" / "pairs.jsonl"), jl(corpus / "manifest" / "reserve" / "pairs.jsonl")
    m = next(p for p in pairs if p["family"] == res[0]["family"] and p["split"] == res[0]["split"])
    bf = json.loads((corpus / "manifest" / "backfill.json").read_text())
    bf["backfills"] = [{"reserve_pair_id": res[0]["pair_id"], "replaces_pair_id": m["pair_id"]}]
    wjson(corpus / "manifest" / "backfill.json", bf)
    reseal(corpus)
    assert "reserve_stratum_policy" in failing(corpus)


def test_backfill_family_or_split_mismatch_refused(corpus):
    pairs, res = jl(corpus / "manifest" / "pairs.jsonl"), jl(corpus / "manifest" / "reserve" / "pairs.jsonl")
    r = res[0]
    m = next(p for p in pairs if (p["family"], p["split"]) != (r["family"], r["split"]))
    _with_backfill(corpus, r, m)
    assert "reserve_stratum_policy" in failing(corpus)


def test_reserve_speaker_crossing_splits_refused(corpus):
    rc = jl(corpus / "manifest" / "reserve" / "clips.jsonl")
    rp = jl(corpus / "manifest" / "reserve" / "pairs.jsonl")
    main = jl(corpus / "manifest" / "clips.jsonl")
    pid = rc[0]["pair_id"]
    other = next(c["speaker_id"] for c in main if c["split"] != rc[0]["split"])
    for r in rc:
        if r["pair_id"] == pid:
            r["speaker_id"] = other
    for p in rp:
        if p["pair_id"] == pid:
            p["speaker_id"] = other
    wjl(corpus / "manifest" / "reserve" / "clips.jsonl", rc)
    wjl(corpus / "manifest" / "reserve" / "pairs.jsonl", rp)
    reseal(corpus)
    f = failing(corpus)
    assert "reserve_stratum_policy" in f and "selection_confirmation_disjoint" in f


def test_reserve_assistant_context_from_other_split_refused(corpus):
    rc = jl(corpus / "manifest" / "reserve" / "clips.jsonl")
    main = jl(corpus / "manifest" / "clips.jsonl")
    foreign = next(c["assistant"]["recording_id"] for c in main if c["split"] != rc[0]["split"] and c["assistant"]["mode"] == rc[0]["assistant"]["mode"])
    rc[0]["assistant"]["recording_id"] = foreign
    wjl(corpus / "manifest" / "reserve" / "clips.jsonl", rc)
    reseal(corpus)
    f = failing(corpus)
    assert "reserve_stratum_policy" in f or "selection_confirmation_disjoint" in f


def test_reserve_labelled_synthesized_refused(corpus):
    rc = jl(corpus / "manifest" / "reserve" / "clips.jsonl")
    rc[0]["stratum"] = "synthesized"
    wjl(corpus / "manifest" / "reserve" / "clips.jsonl", rc)
    reseal(corpus)
    assert "strata_separate_and_advance_policy" in failing(corpus)


def test_reserve_dialogue_colliding_with_main_refused(corpus):
    rc = jl(corpus / "manifest" / "reserve" / "clips.jsonl")
    rp = jl(corpus / "manifest" / "reserve" / "pairs.jsonl")
    main = jl(corpus / "manifest" / "clips.jsonl")
    pid = rc[0]["pair_id"]
    for p in rp:
        if p["pair_id"] == pid:
            p["dialogue_id"] = main[0]["dialogue_id"]
    wjl(corpus / "manifest" / "reserve" / "pairs.jsonl", rp)
    reseal(corpus)
    assert "reserve_stratum_policy" in failing(corpus)


def test_reserve_audio_altered_refused(corpus):
    rec = jl(corpus / "manifest" / "reserve" / "recordings.jsonl")[0]

    def flip(pc):
        s = pc.samples.copy()
        s[(rec["trim"][0] + rec["trim"][1]) // 2 - 3, 0] += 1000
        return A.Pcm(s, pc.sample_rate)

    mutate_recording(corpus, rec, flip)
    reseal(corpus)
    assert "recordings_pcm_bound" in failing(corpus)


# ------------------------------------------------------------------ independent QA gates (tt_qa_gates.py)
def test_qa_report_missing_refused(corpus):
    shutil.rmtree(corpus / "qa")
    reseal(corpus)
    assert "independent_qa_gates_passed" in failing(corpus)


def test_qa_report_failed_clip_refused(corpus):
    p = corpus / "qa" / "qa_gates_report.json"
    q = json.loads(p.read_text())
    cid = next(iter(q["clips"]))
    q["clips"][cid]["gates"]["U1_words"] = False
    q["clips"][cid]["pass"] = False
    q["all_pass"] = False
    wjson(p, q)
    reseal(corpus)
    assert "independent_qa_gates_passed" in failing(corpus)


def test_qa_report_for_other_audio_refused(corpus):
    p = corpus / "qa" / "qa_gates_report.json"
    q = json.loads(p.read_text())
    cid = next(iter(q["clips"]))
    q["clips"][cid]["pcm_sha256"] = "0" * 64
    wjson(p, q)
    reseal(corpus)
    assert "independent_qa_gates_passed" in failing(corpus)


def test_qa_asr_must_differ_from_build_and_scoring_asr(corpus):
    p = corpus / "qa" / "qa_gates_report.json"
    q = json.loads(p.read_text())
    q["asr_b"] = q["built_with_asr_a"]
    wjson(p, q)
    reseal(corpus)
    assert "independent_qa_gates_passed" in failing(corpus)


def _gate_on_tampered(tmp_path, tamper):
    import stub_asr
    import tt_qa_gates as Q

    root = tmp_path / "t"
    root.mkdir()
    D, rep = fake_renders.build_fake_corpus(root, tamper=tamper)
    return D, rep["main"]


def _edit_clips(cand: Path, fn):
    p = cand / "manifest" / "clips.jsonl"
    rows = jl(p)
    fn(rows)
    wjl(p, rows)


def _first_b(rows, fam, variant=".v0"):
    return next(r for r in rows if r["pair_role"] == "b" and r["family"] == fam and r["clip_id"].endswith("-b") and r["stratum"] == "synthesized")


def test_gate_catches_cut_one_word_early(tmp_path):
    victim = {}

    def tamper(cand):
        def fn(rows):
            r = _first_b(rows, "CUT_IC")
            r["user"]["cut"] -= int(0.40 * 16000)  # one 0.28 s word + gap earlier
            victim["pair"] = r["pair_id"]
        _edit_clips(cand, fn)

    D, rep = _gate_on_tampered(tmp_path, tamper)
    sel = json.loads((D / "qa" / "selection_report.json").read_text())
    assert victim["pair"].split(".v")[0] in sel["unfilled_pairs"]
    assert rep["clips"][victim["pair"] + "-b"]["gates"]["U1_words"]["ok"] is False


def test_gate_catches_cut_into_next_word(tmp_path):
    victim = {}

    def tamper(cand):
        def fn(rows):
            r = _first_b(rows, "CUT_IC")
            r["user"]["cut"] += int(0.15 * 16000)  # 60 ms into the next word
            victim["pair"] = r["pair_id"]
        _edit_clips(cand, fn)

    D, rep = _gate_on_tampered(tmp_path, tamper)
    g = rep["clips"][victim["pair"] + "-b"]["gates"]
    assert not (g["U1_words"]["ok"] and g["U2_boundary"]["ok"])


def test_gate_catches_cut_inside_word(tmp_path):
    victim = {}

    def tamper(cand):
        def fn(rows):
            r = _first_b(rows, "CUT_IC")
            r["user"]["cut"] -= int(0.14 * 16000)  # inside the last prefix word
            victim["pair"] = r["pair_id"]
        _edit_clips(cand, fn)

    D, rep = _gate_on_tampered(tmp_path, tamper)
    assert not rep["clips"][victim["pair"] + "-b"]["pass"]


def test_gate_catches_silent_user_channel(tmp_path):
    victim = {}

    def tamper(cand):
        def fn(rows):
            r = _first_b(rows, "CUT_WAIT")
            r["user"]["cut"] = int(0.03 * 16000)
            victim["pair"] = r["pair_id"]
        _edit_clips(cand, fn)

    D, rep = _gate_on_tampered(tmp_path, tamper)
    assert rep["clips"][victim["pair"] + "-b"]["gates"]["U3_nonempty"]["ok"] is False


def test_gate_catches_assistant_truncated_at_clip_end(tmp_path, monkeypatch):
    """Emulates the v1 layout: the continuing assistant is cut at user_end + TRAIL."""
    import stub_asr
    import tt_qa_gates as Q

    orig = C.assemble_clip

    def v1_like(user_rec, trim, cut, assistant_rec, mode, tail=None):
        if mode == "continuing":
            seg = C.user_segment(user_rec, trim[0], trim[1], cut)
            keep = C.BC_ONSET_IN_ASSIST + len(seg) + C.TRAIL
            arr, lay = orig(user_rec, trim, cut, assistant_rec, mode, tail)
            arr = arr[: C.LEAD_IN + keep].copy()
            return arr, C.Layout(lay.user_onset, lay.user_len, lay.user_end, len(arr), lay.assistant_start, len(arr), C.TRAIL)
        return orig(user_rec, trim, cut, assistant_rec, mode, tail)

    root = tmp_path / "t"
    root.mkdir()
    D, rep = fake_renders.build_fake_corpus(root)
    monkeypatch.setattr(C, "assemble_clip", v1_like)
    r = Q.run(D, "main", 1, None, asr=stub_asr.StubAsr(), progress=lambda *_: None)
    assert r["defect_counts_by_gate"]["A1_assistant_continues"] >= 30


def test_ctx_pair_assistant_continues_past_backchannel(fake_corpus):
    clips = [json.loads(x) for x in open(fake_corpus / "manifest" / "clips.jsonl")]
    bs = [c for c in clips if c["family"] == "CTX_BC" and c["pair_role"] == "b"]
    assert bs
    for c in bs:
        lay = c["layout"]
        assert lay["assistant_end"] - lay["user_end"] >= C.N_CONT_MIN
        assert lay["total"] - lay["assistant_end"] >= C.END_PAD
        assert lay["tail"] >= C.TRAIL


def test_gates_reuse_skips_identical_audio(tmp_path):
    import stub_asr
    import tt_qa_gates as Q

    root = tmp_path / "t"
    root.mkdir()
    D, rep = fake_renders.build_fake_corpus(root)
    # the gates were run on the candidate set; re-run on the final corpus with the first report as reuse -> nothing to gate
    reuse = {e["pcm_sha256"]: e for e in rep["main"]["clips"].values()}

    class Count(stub_asr.StubAsr):
        n = 0

        def words(self, x, ctx=None):
            Count.n += 1
            return super().words(x, ctx)

    r = Q.run(D, "main", 1, None, asr=Count(), progress=lambda *_: None, reuse=reuse)
    assert Count.n == 0 and r["defective_pairs"] == 0 and r["n_clips"] == len([1 for _ in open(D / "manifest" / "clips.jsonl")])


def test_variant_fallback_uses_next_passing_variant(tmp_path):
    victim = {}

    def tamper(cand):
        def fn(rows):
            r = next(x for x in rows if x["pair_role"] == "b" and x["family"] == "CUT_IC" and x["clip_id"].endswith(".v0-b") and x["stratum"] == "synthesized")
            r["user"]["cut"] -= int(0.40 * 16000)
            victim["pair"] = r["pair_id"].split(".v")[0]
        _edit_clips(cand, fn)

    root = tmp_path / "t"
    root.mkdir()
    D, rep = fake_renders.build_fake_corpus(root, tamper=tamper, variants=2)
    pairs = {p["pair_id"]: p for p in jl(D / "manifest" / "pairs.jsonl")}
    assert pairs[victim["pair"]]["chosen_variant"] == 1
    others = [p for k, p in pairs.items() if k != victim["pair"]]
    assert all(p["chosen_variant"] == 0 for p in others)
    assert failing(D) == set()


# ------------------------------------------------------------------ consumer path: tt_materialize must agree with tt_verify
def _materialize_all(corpus: Path, out: Path):
    return subprocess.run([sys.executable, "-I", str(TT / "tt_materialize.py"), "--corpus", str(corpus), "--out", str(out), "--include-reserve"],
                          capture_output=True, text=True)


def test_materialize_every_clip_like_a_consumer(corpus, tmp_path):
    """The consumer path (tt_materialize, which CS-47/CS-48 call) must reproduce EVERY committed clip, main and reserve, and
    agree with tt_verify. v2's first push shipped a tt_materialize that omitted layout.tail: tt_verify passed while CTX_BC
    clips failed here."""
    n = len(jl(corpus / "manifest" / "clips.jsonl")) + len(jl(corpus / "manifest" / "reserve" / "clips.jsonl"))
    assert failing(corpus) == set()
    r = _materialize_all(corpus, tmp_path / "m")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(list((tmp_path / "m").glob("*.wav"))) == n
    assert any(c["layout"]["tail"] > C.TRAIL for c in jl(corpus / "manifest" / "clips.jsonl")), "no shared-tail (CTX_BC) clip to exercise"


def test_materialize_and_verify_agree_when_the_tar_diverges(corpus, tmp_path):
    rec = next(r for r in map(json.loads, open(corpus / "manifest" / "recordings.jsonl")) if r["role"] == "user")

    def flip(pc):
        s = pc.samples.copy()
        s[(rec["trim"][0] + rec["trim"][1]) // 2 - 3, 0] += 1000
        return A.Pcm(s, pc.sample_rate)

    mutate_recording(corpus, rec, flip)
    reseal(corpus)  # digests consistent, manifest PCM hashes now stale versus the tar
    assert failing(corpus), "tt_verify accepted a tar that no longer matches the manifest"
    assert _materialize_all(corpus, tmp_path / "m").returncode != 0, "tt_materialize accepted a tar that no longer matches the manifest"


def test_materialize_and_verify_agree_when_a_layout_tail_diverges(corpus, tmp_path):
    rows = jl(corpus / "manifest" / "clips.jsonl")
    for r in rows:
        if r["layout"]["tail"] > C.TRAIL:
            r["layout"]["tail"] -= 800
            break
    wjl(corpus / "manifest" / "clips.jsonl", rows)
    reseal(corpus)
    assert failing(corpus)
    assert _materialize_all(corpus, tmp_path / "m").returncode != 0
