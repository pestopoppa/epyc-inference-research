"""CS-45.1 tail stratum builder tests. Run: python3 -m pytest -q tests/test_tt_tail_stratum.py (from scripts/voice/turn_taking)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TT = HERE.parent
sys.path.insert(0, str(TT))
import tt_audio as A  # noqa: E402
import tt_corpus as C  # noqa: E402
import tt_tail_stratum as S  # noqa: E402

REPO = TT.parents[2]
V2 = REPO / "data" / "turn-taking-cs45-v1"
needs_corpus = pytest.mark.skipif(not (V2 / "qa" / "qa_gates_report.json").exists(), reason="v2 corpus absent")


def synth_parent(tail=C.TRAIL):
    rng = np.random.default_rng(1)
    user = (rng.normal(0, 3000, 16000)).astype(np.int16)
    asst = (rng.normal(0, 2000, 12000)).astype(np.int16)
    return C.assemble_clip(user, (0, 16000), 8000, asst, "ended", tail)


def test_room_tone_deterministic_level_and_not_digital_silence():
    a, b = S.room_tone(32000, 7), S.room_tone(32000, 7)
    assert np.array_equal(a, b) and not np.array_equal(a, S.room_tone(32000, 8))
    assert abs(S.level_dbfs(a) - S.ROOM_TONE_DBFS) < 0.2
    assert (a == 0).mean() < 0.01


@pytest.mark.parametrize("tail_s", S.TAILS_S)
def test_recut_exact_tail_and_bound_to_parent(tail_s):
    parent, lay = synth_parent()
    new, nl = S.recut(parent, lay, "tt1-px", tail_s)
    assert len(new) == lay.user_end + int(round(tail_s * 16000))
    g = S.tail_gates(new, nl, parent, lay, tail_s)
    assert all(v["ok"] for v in g.values()), g
    assert np.array_equal(new[lay.user_onset : lay.user_end, 0], parent[lay.user_onset : lay.user_end, 0])
    assert not np.any(new[nl["user_end"] :, 0] == 0) or (new[nl["user_end"] :, 0] == 0).mean() < 0.01


def test_pair_members_share_bed_on_common_clock():
    parent, lay = synth_parent()
    n1, _ = S.recut(parent, lay, "tt1-px", 1.0)
    n2, _ = S.recut(parent, lay, "tt1-px", 1.0)
    assert np.array_equal(n1, n2)
    n3, _ = S.recut(parent, lay, "tt1-py", 1.0)
    assert not np.array_equal(n1[lay.user_end :, 0], n3[lay.user_end :, 0])


def test_gates_refuse_digital_silence_tail_wrong_length_and_altered_segment():
    parent, lay = synth_parent()
    new, nl = S.recut(parent, lay, "tt1-px", 0.6)
    z = new.copy(); z[nl["user_end"] :, 0] = 0
    assert not S.tail_gates(z, nl, parent, lay, 0.6)["T2_room_tone"]["ok"]
    assert not S.tail_gates(new[:-1], dict(nl, total=len(new) - 1), parent, lay, 0.6)["T1_tail_length"]["ok"]
    t = new.copy(); t[nl["user_onset"] + 5, 0] ^= 1
    assert not S.tail_gates(t, nl, parent, lay, 0.6)["T3_bound_to_parent"]["ok"]
    loud = new.copy(); loud[nl["user_end"] : nl["user_end"] + 1600, 0] = 8000
    assert not S.tail_gates(loud, nl, parent, lay, 0.6)["T2_room_tone"]["ok"]


def test_tail_sample_rounding_refused():
    with pytest.raises(ValueError):
        S.tail_samples(0.30001)


@needs_corpus
def test_build_verify_and_mutations(tmp_path):
    d = tmp_path / "t"
    meta = S.build(V2, d, tmp_path / "wav", limit=2)
    assert meta["counts"]["clips"] == 2 * 2 * 5 and meta["counts"]["defective_clips"] == 0
    assert meta["advance_eligible_strata"] == ["human_recorded"] and meta["ranking_only"] and meta["stratum"] == "synthesized"
    assert meta["natural_control"]["status"] == "declared-empty" and meta["natural_control"]["n"] == 0
    assert S.verify(V2, d) == []
    wavs = sorted((tmp_path / "wav").glob("*.wav"))
    assert len(wavs) == 20
    # on-disk WAV decodes to the manifest PCM-SHA-256
    rows = {r["clip_id"]: r for r in map(json.loads, (d / "manifest" / "clips.jsonl").read_text().splitlines())}
    for w in wavs[:4]:
        assert A.read_wav(w).pcm_sha256() == rows[w.stem]["pcm"]["pcm_sha256"]
    # mutation: widen advance policy
    m = json.loads((d / "tail_stratum.json").read_text())
    m["advance_eligible_strata"] = ["human_recorded", "synthesized"]
    (d / "tail_stratum.json").write_text(json.dumps(m, sort_keys=True))
    assert any("advance" in x for x in S.verify(V2, d))
    # mutation: altered clip hash (resealed manifest digest would still fail re-assembly)
    d2 = tmp_path / "t2"; shutil.copytree(d, d2)
    m = json.loads((d2 / "tail_stratum.json").read_text()); m["advance_eligible_strata"] = ["human_recorded"]
    cp = d2 / "manifest" / "clips.jsonl"; ls = cp.read_text().splitlines(); r0 = json.loads(ls[0]); r0["pcm"]["pcm_sha256"] = "0" * 64
    ls[0] = json.dumps(r0, sort_keys=True); cp.write_text("\n".join(ls) + "\n")
    m["files"]["manifest/clips.jsonl"] = S._sha_file(cp); (d2 / "tail_stratum.json").write_text(json.dumps(m, sort_keys=True))
    assert any("PCM-SHA-256 mismatch" in x for x in S.verify(V2, d2))
    # mutation: natural control claimed filled while the human slot is empty
    d3 = tmp_path / "t3"; shutil.copytree(d, d3)
    m = json.loads((d / "tail_stratum.json").read_text()); m["natural_control"]["status"] = "filled"; m["advance_eligible_strata"] = ["human_recorded"]
    (d3 / "tail_stratum.json").write_text(json.dumps(m, sort_keys=True))
    assert any("natural control" in x for x in S.verify(V2, d3))


@needs_corpus
def test_committed_stratum_verifies_if_present():
    d = REPO / "data" / "turn-taking-cs45-1-tail-v1"
    if not d.exists():
        pytest.skip("stratum not built")
    assert S.verify(V2, d) == []
    meta = json.loads((d / "tail_stratum.json").read_text())
    assert meta["counts"]["clips"] == 820 and meta["counts"]["defective_clips"] == 0
