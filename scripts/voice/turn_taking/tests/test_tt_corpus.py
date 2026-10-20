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
STATIC = ["label_contract.json", "pins.json", "external_crosscheck.json", "measurement_era_changes.json"]


def _copy_static(dst: Path):
    src = REPO / "data" / "turn-taking-cs45-v1"
    for f in STATIC:
        shutil.copyfile(src / f, dst / f)
    shutil.copytree(src / "pins", dst / "pins")
    shutil.copytree(src / "scorer_patches", dst / "scorer_patches")
    (dst / "manifest").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "manifest" / "HUMAN_SLOT.md", dst / "manifest" / "HUMAN_SLOT.md")
    (dst / "manifest" / "human_recorded.jsonl").write_text("")


@pytest.fixture(scope="session")
def fake_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake")
    S, D = root / "scratch", root / "corpus"
    fake_renders.make_scratch(S)
    D.mkdir()
    _copy_static(D)
    r = subprocess.run([sys.executable, "-I", str(TT / "tt_build.py"), "--scratch", str(S), "--out", str(D)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    V.seal(D)
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
