#!/usr/bin/env python3
"""CS-45 corpus builder: renders (scratch) + plan -> durable in-repo corpus.

  python3 tt_build.py --scratch /mnt/raid0/llm/tmp/cs45/run1 --out data/turn-taking-cs45-v1

Writes, under --out:
  audio/synth/user.tar, audio/synth/assistant.tar   16 kHz mono PCM16 source recordings (WAV members; see tt_audio.write_tar)
  manifest/recordings.jsonl   one row per source recording (PCM-SHA-256, provenance, ASR timing)
  manifest/clips.jsonl        one row per pair member = the recipe + the PCM-SHA-256 of the stereo clip
  manifest/pairs.jsonl        one row per matched pair
  manifest/splits.json        selection / confirmation partition (speaker, dialogue, recording, pair)
  manifest/build_report.json  cut-point decisions, rejects, invariant results

Only SOURCE recordings are stored; clips are deterministic functions of (recording, recipe) and are
re-materialised and hash-checked by tt_verify.py / tt_materialize.py. Nothing here calls a model.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_corpus as C  # noqa: E402
import tt_plan as P  # noqa: E402

PAD_MS = 30
CUT_WINDOW_MS = 160  # search radius around the ASR boundary estimate
VALLEY_MAX_DB_BELOW_P95 = 24.0  # a cut must land >= this many dB below the recording's p95 10-ms RMS
FRAME = 160  # 10 ms


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    fr = x[: n * FRAME].astype(np.float64).reshape(n, FRAME) / 32768.0
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1) + 1e-20) + 1e-12)


def align_intended_to_asr(intended: list[str], asr_words: list[dict]) -> dict[int, int]:
    """Map intended word index -> ASR word index for matched/replaced words (difflib on normalised tokens)."""
    hyp = [" ".join(P.norm_words(w["w"])) for w in asr_words]
    sm = difflib.SequenceMatcher(a=intended, b=hyp, autojunk=False)
    m: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace") and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
        elif tag == "replace" and j2 > j1:  # unequal block (e.g. 'sixty eight' -> '68'): anchor its two ends
            m[i1] = j1
            m[i2 - 1] = j2 - 1
    return m


def choose_cut(rec: np.ndarray, trim: tuple[int, int], intended_words: list[str], n_prefix: int, asr_words: list[dict]) -> dict:
    """Return a cut decision dict; status 'ok' or a reason string."""
    m = align_intended_to_asr(intended_words, asr_words)
    if n_prefix not in m or (n_prefix - 1) not in m:
        return {"status": "no_asr_anchor"}
    nxt, prv = asr_words[m[n_prefix]], asr_words[m[n_prefix - 1]]
    t_prev_end = (prv["t1_ms"] or 0) / 1000.0
    t_next_start = (nxt["t0_ms"] or 0) / 1000.0
    est_gap_mid = 0.5 * (t_prev_end + t_next_start)
    ests = [est_gap_mid]
    if nxt.get("dtw_ms") is not None:
        ests.append(nxt["dtw_ms"] / 1000.0 - 0.04)
    est = float(np.mean(ests))
    db = frame_db(rec)
    p95 = float(np.percentile(db[db > -70], 95)) if np.any(db > -70) else -20.0
    best = None
    lo = int(max(0, (est - CUT_WINDOW_MS / 1000) * A.SR))
    hi = int(min(len(rec) - FRAME, (est + CUT_WINDOW_MS / 1000) * A.SR))
    for s in range(lo, hi, 80):  # 5 ms hop over 10 ms windows
        seg = rec[s : s + FRAME].astype(np.float64) / 32768.0
        d = 20 * np.log10(np.sqrt((seg ** 2).mean() + 1e-20) + 1e-12)
        if best is None or d < best[1]:
            best = (s + FRAME // 2, d)
    if best is None:
        return {"status": "empty_window"}
    cut_abs, cut_db = best
    cut_rel = cut_abs - trim[0]
    out = {
        "status": "ok" if (cut_db <= p95 - VALLEY_MAX_DB_BELOW_P95 and 0 < cut_rel < trim[1] - trim[0] - 4 * FRAME) else "no_valley",
        "asr_prev_word": prv["w"], "asr_next_word": nxt["w"], "asr_prev_end_s": round(t_prev_end, 3),
        "asr_next_start_s": round(t_next_start, 3), "asr_next_dtw_s": None if nxt.get("dtw_ms") is None else round(nxt["dtw_ms"] / 1000, 3),
        "estimate_s": round(est, 3), "cut_abs_sample": int(cut_abs), "cut_rel_sample": int(cut_rel),
        "cut_db": round(cut_db, 1), "recording_p95_db": round(p95, 1),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-rejects", action="store_true", help="write the corpus even if some pairs are rejected (reported)")
    a = ap.parse_args()
    S, D = Path(a.scratch), Path(a.out)
    plan = P.build_plan()
    rows = {}
    for line in (S / "renders.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["render_id"]] = r
    env = json.loads((S / "env.json").read_text())
    asr_layout = json.loads((S / "asr_layout.json").read_text()) if (S / "asr_layout.json").exists() else None

    missing = [j["render_id"] for j in plan["jobs"] if j["render_id"] not in rows or not rows[j["render_id"]].get("qa_ok")]
    if missing:
        print(f"{len(missing)} renders missing or failed QA: {missing[:10]} ...", file=sys.stderr)
        if not a.allow_rejects:
            return 2

    pcm: dict[str, np.ndarray] = {}
    for rid, r in rows.items():
        pcm[rid] = A.read_wav(S / "rec16" / f"{rid}.wav").samples[:, 0]

    report = {"rejected_pairs": [], "cuts": {}, "invariants": {}}
    trims: dict[str, dict] = {}
    for j in plan["jobs"]:
        rid = j["render_id"]
        if rid not in pcm:
            continue
        x = pcm[rid]
        vs, ve = A.trim_to_speech(x, A.SR, 10, -50.0, 0)
        t0 = max(0, vs - PAD_MS * A.SR // 1000)
        t1 = min(len(x), ve + PAD_MS * A.SR // 1000)
        trims[rid] = {"trim": (int(t0), int(t1)), "voiced_start": int(vs - t0), "voiced_end": int(ve - t0)}

    clips, pairs, used_user, used_asst = [], [], set(), set()
    for pr in plan["pairs"]:
        pid, fam = pr["pair_id"], pr["family"]
        urid = pr["user_recording"]
        if urid not in pcm or pr["assistant_a"] not in pcm or pr["assistant_b"] not in pcm:
            report["rejected_pairs"].append({"pair_id": pid, "reason": "missing_render"})
            continue
        tr = trims[urid]
        cut = None
        cut_info = None
        if fam in ("CUT_IC", "CUT_WAIT"):
            intended = P.norm_words(pr["text_full"])
            cut_info = choose_cut(pcm[urid], tr["trim"], intended, pr["prefix_words"], rows[urid].get("asr", {}).get("words", []))
            report["cuts"][pid] = cut_info
            if cut_info["status"] != "ok":
                report["rejected_pairs"].append({"pair_id": pid, "reason": f"cut:{cut_info['status']}", "detail": cut_info})
                continue
            cut = cut_info["cut_rel_sample"]
        members = []
        for role, cutv, asst_id, mode, label in (
            ("a", None, pr["assistant_a"], pr["assistant_mode_a"], pr["label_a"]),
            ("b", cut, pr["assistant_b"], pr["assistant_mode_b"], pr["label_b"]),
        ):
            if fam in ("CUT_IC", "CUT_WAIT") and role == "a":
                cutv = None  # a = natural end (complete)
            arr, lay = C.assemble_clip(pcm[urid], tr["trim"], cutv, pcm[asst_id], mode)
            members.append((role, cutv, asst_id, mode, label, arr, lay))
        # context pairs: b is the cut-less one too
        viol = C.check_pair_invariants(fam, members[0][5], members[0][6], members[0][1], members[1][5], members[1][6], members[1][1])
        report["invariants"][pid] = viol
        if viol:
            report["rejected_pairs"].append({"pair_id": pid, "reason": "invariants", "detail": viol})
            continue
        pair_clips = []
        for role, cutv, asst_id, mode, label, arr, lay in members:
            cid = f"{pid}-{role}"
            voiced_end = tr["voiced_end"] if cutv is None else cutv
            clip = {
                "clip_id": cid, "pair_id": pid, "pair_role": role, "stratum": "synthesized", "split": pr["split"], "family": fam,
                "label": label, "speaker_id": pr["voice_id"], "dialogue_id": f"d-{pid}",
                "user": {"recording_id": urid, "trim": list(tr["trim"]), "cut": cutv, "fade": C.FADE},
                "assistant": {"recording_id": asst_id, "mode": mode},
                "layout": {"lead_in": C.LEAD_IN, "gap_after_assist": C.GAP_AFTER_ASSIST, "bc_onset_in_assist": C.BC_ONSET_IN_ASSIST,
                           "trail": C.TRAIL, "user_onset": lay.user_onset, "user_len": lay.user_len, "user_end": lay.user_end,
                           "total": lay.total, "assistant_start": lay.assistant_start, "assistant_end": lay.assistant_end},
                "gold": gold_for(label, lay, tr, cutv),
                "text": {"user_spoken": pr["text_prefix"] if cutv is not None else pr["text_full"], "user_full": pr["text_full"],
                         "assistant": rows[asst_id]["text"]},
                "pcm": {"sample_rate": A.SR, "channels": 2, "n_samples": int(arr.shape[0]), "pcm_sha256": A.Pcm(arr, A.SR).pcm_sha256(),
                        "ch_sha256": [A.Pcm(arr, A.SR).channel_sha256(0), A.Pcm(arr, A.SR).channel_sha256(1)]},
            }
            if fam in ("CUT_IC", "CUT_WAIT") and cutv is not None:
                clip["cut_decision"] = cut_info
            clips.append(clip)
            pair_clips.append(cid)
        used_user.add(urid)
        used_asst.update([pr["assistant_a"], pr["assistant_b"]])
        pairs.append({
            "pair_id": pid, "family": fam, "split": pr["split"], "stratum": "synthesized", "contrast": pr["contrast"],
            "speaker_id": pr["voice_id"], "dialogue_id": f"d-{pid}", "user_recording": urid, "clips": pair_clips,
            "labels": {"a": pr["label_a"], "b": pr["label_b"]}, "language": "en",
        })

    if report["rejected_pairs"] and not a.allow_rejects:
        (S / "build_report.json").write_text(json.dumps(report, indent=1))
        print(f"{len(report['rejected_pairs'])} pairs rejected; report at {S/'build_report.json'}", file=sys.stderr)
        return 3

    # ---- write durable tree
    (D / "manifest").mkdir(parents=True, exist_ok=True)
    rec_rows = []
    members = {"user": {}, "assistant": {}}
    for rid in sorted(used_user | used_asst):
        r = rows[rid]
        sub = "user" if r["role"] == "user" else "assistant"
        data = (S / "rec16" / f"{rid}.wav").read_bytes()
        members[sub][f"{rid}.wav"] = data
        pc = A.read_wav_bytes(data)
        rec_rows.append({
            "recording_id": rid, "role": r["role"], "stratum": "synthesized", "speaker_id": r["voice_id"], "split": r["split"],
            "text": r["text"], "path": f"audio/synth/{sub}.tar#{rid}.wav", "sample_rate": pc.sample_rate, "channels": 1, "n_samples": pc.n_samples,
            "pcm_sha256": pc.pcm_sha256(), "file_sha256": hashlib.sha256(data).hexdigest(),
            "trim": list(trims[rid]["trim"]) if r["role"] == "user" else None,
            "qa_override": r.get("qa_override"),
            "provenance": {
                "tts": {"binary": "qwen-tts", "tree": "qwentts.cpp@production-speech-v1", "ref_speaker": f"librispeech-test-clean:{r['voice_id']}:{Path(r['ref_wav']).stem}",
                        "ref_wav_sha256": r["ref_sha256"], "seed": r["seed_final"], "attempts": r["history"], "raw24_pcm_sha256": r["raw24_pcm_sha256"]},
                "resample": "scipy.signal.resample_poly up=2 down=3 (24k->16k), float64 -> round -> int16",
                "asr_word_timing": r.get("asr"),
            },
        })
    for sub, m in members.items():
        A.write_tar(D / "audio" / "synth" / f"{sub}.tar", m)
    for fn, rowsx in (("recordings.jsonl", rec_rows), ("clips.jsonl", clips), ("pairs.jsonl", pairs)):
        with open(D / "manifest" / fn, "w") as f:
            for x in rowsx:
                f.write(json.dumps(x, sort_keys=True) + "\n")
    splits = {"rule": "voices are partitioned into selection/confirmation speaker sets; every pair, dialogue, source recording and "
                      "assistant context inherits its split from that partition; nothing crosses",
              "plan_seed": plan["plan_seed"], "selection": {}, "confirmation": {}}
    for sp in P.SPLITS:
        splits[sp] = {
            "speakers": sorted({p["speaker_id"] for p in pairs if p["split"] == sp}),
            "dialogues": sorted(p["dialogue_id"] for p in pairs if p["split"] == sp),
            "pairs": sorted(p["pair_id"] for p in pairs if p["split"] == sp),
            "user_recordings": sorted(p["user_recording"] for p in pairs if p["split"] == sp),
            "assistant_recordings": sorted({c["assistant"]["recording_id"] for c in clips if c["split"] == sp}),
        }
    (D / "manifest" / "splits.json").write_text(json.dumps(splits, indent=1, sort_keys=True) + "\n")
    report["env"] = env
    report["asr_layout"] = asr_layout
    (D / "manifest" / "build_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(rec_rows)} recordings, {len(clips)} clips, {len(pairs)} pairs ({len(report['rejected_pairs'])} rejected)")
    return 0


def gold_for(label: str, lay: C.Layout, tr: dict, cut: int | None) -> dict:
    onset = lay.user_onset
    v_start = (onset + tr["voiced_start"]) / A.SR
    v_end = (onset + (cut if cut is not None else tr["voiced_end"])) / A.SR
    trail_end = lay.total / A.SR
    g = {
        "user_speech_start_s": round(v_start, 4), "user_speech_end_s": round(v_end, 4), "clip_end_s": round(trail_end, 4),
        "assistant_speaking_at_user_onset": lay.assistant_end is not None and lay.assistant_end > onset,
        "assistant_speech_end_s": None if lay.assistant_end is None else round(lay.assistant_end / A.SR, 4),
    }
    if label == "complete":
        g.update({"eot_gold_s": g["user_speech_end_s"], "hold_window_s": None, "floor_holder_during_user": "user"})
    elif label in ("incomplete", "wait"):
        g.update({"eot_gold_s": None, "hold_window_s": [g["user_speech_end_s"], g["clip_end_s"]], "floor_holder_during_user": "user"})
    elif label == "backchannel":
        g.update({"eot_gold_s": None, "hold_window_s": [g["user_speech_start_s"], g["clip_end_s"]], "floor_holder_during_user": "assistant"})
    return g


if __name__ == "__main__":
    sys.exit(main())
