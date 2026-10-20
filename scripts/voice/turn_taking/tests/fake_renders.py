"""Synthetic stand-in for tt_render_pipeline output (tones + gaps), used ONLY by the conformance tests.
Exercises tt_build / tt_verify end-to-end without any TTS or ASR binary."""
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
        rows.append({"render_id": rid, "role": j["role"], "text": j["text"], "voice_id": j["voice_id"], "ref_wav": j["ref"], "ref_sha256": "0" * 64,
                     "split": j["split"], "family": j.get("family"), "pair_id": j.get("pair_id"), "prefix_words": j.get("prefix_words"),
                     "seed_final": j["seed"], "history": [{"attempt": 0, "seed": j["seed"], "qa": "ok"}], "qa_ok": True,
                     "raw24_pcm_sha256": "0" * 64, "rec16_pcm_sha256": pc.pcm_sha256(), "rec16_samples": pc.n_samples,
                     "asr": {"text": " ".join(words), "words": asr_words}})
    (S / "renders.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (S / "env.json").write_text(json.dumps({"fake": True}))
