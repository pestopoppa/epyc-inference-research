#!/usr/bin/env python3
"""CS-45 ASR-A: word alignment + CUT SEARCH with faster-whisper large-v3-TURBO (CT2, CPU, int8).

This is the BUILD aligner. The QA gates (tt_qa_gates.py) use a different model, faster-whisper large-v3, so the
check is not circular. whisper.cpp's own token timestamps (used in v1) were the root cause of the misaligned cuts and
are no longer used for cut placement (tt_asr_words.py stays the pinned SCORING ASR).

For a cut-family recording the cut is found by GENERATE-AND-TEST, not by trusting word timestamps (cross-attention
timestamps are good to ~0.1 s; a word is ~0.1-0.3 s): candidates are the deepest energy minima within +-200 ms of the
ASR-A word boundary, and a candidate is admitted only if ASR-A transcribes the exact clip segment (prefix, with the
10 ms fade) as the expected prefix AND transcribes the remainder of the recording as the expected suffix.

  /mnt/raid0/llm/pace-env/bin/python -I tt_align_fw.py --list jobs.jsonl --out out.jsonl [--threads 5]
jobs.jsonl rows: {"id","wav","text","n_prefix"(optional, cut recordings)}
out rows: {"id","text","words":[{"w","t0_ms","t1_ms","dtw_ms":null,"p"}],"cut_candidates":[...]}
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_build2 as B  # noqa: E402
import tt_corpus as C  # noqa: E402
import tt_qa_gates as Q  # noqa: E402

TURBO = "/mnt/raid0/llm/cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
N_CAND = 8
SEARCH_S = 0.25


def transcribe(m, x_i16: np.ndarray):
    xx = np.concatenate([np.zeros(int(0.3 * 16000), np.float32), x_i16.astype(np.float32) / 32768.0])
    segs, _ = m.transcribe(xx, language="en", word_timestamps=True, vad_filter=False, beam_size=5, temperature=0.0, condition_on_previous_text=False)
    words, text = [], []
    for s in segs:
        text.append(s.text)
        for w_ in s.words or []:
            words.append({"w": w_.word.strip(), "t0_ms": int(round((w_.start - 0.3) * 1000)), "t1_ms": int(round((w_.end - 0.3) * 1000)), "dtw_ms": None, "p": round(w_.probability, 3)})
    return "".join(text).strip(), words


def candidates(rec: np.ndarray, lo_s: float, hi_s: float):
    """Deepest strict local minima of the 10 ms RMS (5 ms hop) inside [lo_s, hi_s]; returns [(cut_abs, level_db)]."""
    lo, hi = int(max(0, lo_s * A.SR)), int(min(len(rec) - 160, hi_s * A.SR))
    pts = []
    for s in range(lo, max(lo + 1, hi), 80):
        seg = rec[s : s + 160].astype(np.float64) / 32768.0
        pts.append((s + 80, 20 * np.log10(np.sqrt((seg ** 2).mean() + 1e-20) + 1e-12)))
    mins = [pts[i] for i in range(len(pts)) if (i == 0 or pts[i][1] <= pts[i - 1][1]) and (i == len(pts) - 1 or pts[i][1] < pts[i + 1][1])]
    mins.sort(key=lambda p: p[1])
    chosen = []
    for c in mins:
        if all(abs(c[0] - o[0]) >= 320 for o in chosen):
            chosen.append(c)
        if len(chosen) == N_CAND:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    from faster_whisper import WhisperModel

    m = WhisperModel(TURBO, device="cpu", compute_type="int8", cpu_threads=a.threads)
    rows = [json.loads(l) for l in Path(a.list).read_text().splitlines() if l.strip()]
    t0 = time.time()
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            with wave.open(r["wav"], "rb") as w:
                assert w.getframerate() == 16000 and w.getnchannels() == 1
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
            text, words = transcribe(m, x)
            out = {"id": r["id"], "text": text, "words": words}
            if r.get("n_prefix"):
                intended = Q.norm_words(r["text"])
                n_pre = r["n_prefix"]
                mm = B.map_words(intended, words)
                cands_out = []
                if (n_pre - 1) in mm and n_pre in mm:
                    prv, nxt = words[mm[n_pre - 1]], words[mm[n_pre]]
                    b0 = (prv["t1_ms"] + nxt["t0_ms"]) / 2000.0
                    t0_, t1_, vs, ve = A.speech_bounds(x)
                    db = B.frame_db(x)
                    live = db[db > -70]
                    p95 = float(np.percentile(live, 95)) if len(live) else -20.0
                    for cut_abs, lvl in candidates(x, b0 - SEARCH_S, b0 + SEARCH_S):
                        cut_rel = cut_abs - t0_
                        if not (0.2 * A.SR < cut_rel < (t1_ - t0_) - 0.15 * A.SR):
                            continue
                        pre = C.user_segment(x, t0_, t1_, cut_rel)
                        suf = x[cut_abs:t1_].copy()
                        nf = min(80, len(suf))
                        suf[:nf] = (suf[:nf].astype(np.float64) * np.linspace(0.0, 1.0, nf)).astype(np.int16)
                        ptxt, _ = transcribe(m, pre)
                        stxt, _ = transcribe(m, suf)
                        p_ok = Q.norm_words(ptxt) == intended[:n_pre]
                        s_ok = Q.norm_words(stxt) == intended[n_pre:]
                        cands_out.append({"cut_abs_sample": int(cut_abs), "cut_rel_sample": int(cut_rel), "level_db": round(float(lvl), 1), "p95_db": round(p95, 1),
                                          "prefix_ok": bool(p_ok), "suffix_ok": bool(s_ok), "prefix_got": " ".join(Q.norm_words(ptxt)), "suffix_got": " ".join(Q.norm_words(stxt)),
                                          "boundary_estimate_s": round(b0, 3)})
                out["cut_candidates"] = cands_out
            f.write(json.dumps(out) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print("done", len(rows))


if __name__ == "__main__":
    sys.exit(main())
