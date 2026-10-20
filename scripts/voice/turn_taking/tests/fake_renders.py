"""Synthetic stand-in for tt_render_pipeline output (tones + gaps), used ONLY by the conformance tests.
Exercises tt_build2 / tt_qa_gates / tt_verify end-to-end without any TTS or ASR binary."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tt_audio as A  # noqa: E402
import tt_plan as P  # noqa: E402

SR = A.SR


def make_scratch(S: Path, reserve: bool = False) -> None:
    (S / "rec16").mkdir(parents=True, exist_ok=True)
    plan = P.build_reserve_plan() if reserve else P.build_plan()
    plan = {"jobs": [j for j in plan["jobs"] if j["role"] == "user"]} if False else plan
    rows = []
    rng = np.random.default_rng(7)
    for j in plan["jobs"]:
        words = P.norm_words(j["text"])
        sig, asr_words, t = [], [], 0
        sig.append(np.zeros(int(0.1 * SR)))
        t += int(0.1 * SR)
        for i, w in enumerate(words):
            n = int(0.28 * SR)
            f = 180 + 17 * (i % 9) + (40 if j["role"] == "assistant" else 0)
            tt = np.arange(n) / SR
            sig.append(0.3 * np.sin(2 * np.pi * f * tt) * (0.6 + 0.4 * np.sin(np.pi * tt / tt[-1])) + 0.01 * rng.standard_normal(n))
            asr_words.append({"w": w, "t0_ms": int(t / SR * 1000), "t1_ms": int((t + n) / SR * 1000), "dtw_ms": int(t / SR * 1000), "p": 0.9})
            t += n
            g = int(0.09 * SR)
            sig.append(np.zeros(g))
            t += g
        x = np.clip(np.concatenate(sig) * 32767, -32768, 32767).astype(np.int16)
        rid = j["render_id"]
        A.write_wav(S / "rec16" / f"{rid}.wav", A.Pcm(x[:, None], SR))
        pc = A.read_wav(S / "rec16" / f"{rid}.wav")
        cands = None
        if j.get("family") in ("CUT_IC", "CUT_WAIT"):
            n = j["prefix_words"]
            gap_mid = (asr_words[n - 1]["t1_ms"] + 90 / 2) / 1000.0  # centre of the 90 ms gap that follows word n-1
            t0_ = A.speech_bounds(x)[0]
            cut_abs = int(gap_mid * SR)
            cands = [{"cut_abs_sample": cut_abs, "cut_rel_sample": cut_abs - t0_, "level_db": -120.0, "p95_db": -10.0, "prefix_ok": True, "suffix_ok": True,
                      "prefix_got": "", "suffix_got": "", "boundary_estimate_s": gap_mid}]
        rows.append({"render_id": rid, "role": j["role"], "text": j["text"], "voice_id": j["voice_id"], "ref_wav": j["ref"], "ref_sha256": "0" * 64,
                     "split": j["split"], "family": j.get("family"), "pair_id": j.get("pair_id"), "prefix_words": j.get("prefix_words"),
                     "seed_final": j["seed"], "history": [{"attempt": 0, "seed": j["seed"], "qa": "ok"}], "qa_ok": True,
                     "raw24_pcm_sha256": "0" * 64, "rec16_pcm_sha256": pc.pcm_sha256(), "rec16_samples": pc.n_samples,
                     "asr": {"text": " ".join(words), "words": asr_words, "cut_candidates": cands}, "base_id": rid, "variant": 0})
    (S / "renders.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (S / "env.json").write_text(json.dumps({"fake": True}))


def build_fake_corpus(root: Path, tamper=None, variants: int = 1):
    """Full v2 pipeline on tone-burst renders with the stub ASR-B: candidates -> gates -> final -> seal.
    tamper(candidate_dir) may corrupt candidates before gating (used to prove the gates refuse defects)."""
    import shutil
    import subprocess

    import stub_asr
    import tt_qa_gates as Q
    import tt_verify as V

    TT = Path(__file__).resolve().parents[1]
    REPO = TT.parents[2]
    S, D, CAND = root / "scratch", root / "corpus", root / "cand"
    make_scratch(S)
    make_scratch(root / "scratch_r", reserve=True)
    D.mkdir()
    src = REPO / "data" / "turn-taking-cs45-v1"
    for f in ("label_contract.json", "pins.json", "external_crosscheck.json", "measurement_era_changes.json"):
        shutil.copyfile(src / f, D / f)
    shutil.copytree(src / "pins", D / "pins")
    shutil.copytree(src / "scorer_patches", D / "scorer_patches")
    (D / "manifest").mkdir()
    shutil.copyfile(src / "manifest" / "HUMAN_SLOT.md", D / "manifest" / "HUMAN_SLOT.md")
    (D / "manifest" / "human_recorded.jsonl").write_text("")
    # assistant recordings = the fake assistant renders, stored the way the real corpus stores them
    members, rows = {}, []
    for j in P.build_plan()["jobs"]:
        if j["role"] != "assistant":
            continue
        data = (S / "rec16" / f"{j['render_id']}.wav").read_bytes()
        members[f"{j['render_id']}.wav"] = data
        pc = A.read_wav_bytes(data)
        rows.append({"recording_id": j["render_id"], "role": "assistant", "stratum": "synthesized", "speaker_id": j["voice_id"], "split": j["split"], "text": j["text"],
                     "path": f"audio/synth/assistant.tar#{j['render_id']}.wav", "sample_rate": SR, "channels": 1, "n_samples": pc.n_samples,
                     "pcm_sha256": pc.pcm_sha256(), "file_sha256": __import__("hashlib").sha256(data).hexdigest(), "trim": None, "provenance": {}})
    A.write_tar(D / "audio" / "synth" / "assistant.tar", members)
    (D / "manifest" / "recordings.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # merge the reserve renders into the same scratch index (ids are disjoint)
    with open(S / "renders.jsonl", "a") as f:
        f.write((root / "scratch_r" / "renders.jsonl").read_text())
    for w in (root / "scratch_r" / "rec16").iterdir():
        shutil.copyfile(w, S / "rec16" / w.name)
    if variants > 1:  # variant k = identical audio under id '<id>.v<k>' (same pcm), enough to exercise the fallback logic
        rows = [json.loads(l) for l in (S / "renders.jsonl").read_text().splitlines()]
        extra = []
        for r in rows:
            if r["role"] != "user":
                continue
            for k in range(1, variants):
                rr = dict(r, render_id=f"{r['render_id']}.v{k}", base_id=r["render_id"], variant=k)
                shutil.copyfile(S / "rec16" / f"{r['render_id']}.wav", S / "rec16" / f"{rr['render_id']}.wav")
                extra.append(rr)
        with open(S / "renders.jsonl", "a") as fh:
            fh.write("\n".join(json.dumps(x) for x in extra) + "\n")
    py = sys.executable
    r = subprocess.run([py, "-I", str(TT / "tt_build2.py"), "candidates", "--scratch", str(S), "--corpus", str(D), "--cand", str(CAND)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    if tamper:
        tamper(CAND)
    rep = {"main": Q.run(CAND, "main", 1, None, asr=stub_asr.StubAsr(), progress=lambda *_: None)}
    rep["main"]["asr_b"] = "faster-whisper large-v3 int8 CPU (stub)"
    (root / "gates.json").write_text(json.dumps(rep))
    r = subprocess.run([py, "-I", str(TT / "tt_build2.py"), "final", "--scratch", str(S), "--corpus", str(D), "--cand", str(CAND), "--gates", str(root / "gates.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    V.seal(D)
    return D, rep
