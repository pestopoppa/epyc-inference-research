#!/usr/bin/env python3
"""CS-45 RESERVE stratum builder: backfill pairs for pairs the human review rejects.

  python3 tt_build_reserve.py --scratch /mnt/raid0/llm/tmp/cs45/run_reserve --corpus data/turn-taking-cs45-v1

Same selection rules, voice partition and splice construction as tt_build.py (it reuses its cut rule, trims and gold),
but NEW texts, recordings and dialogues. Assistant contexts are the main set's recordings reused inside the SAME
split. Writes audio/reserve/user.tar and manifest/reserve/{recordings,clips,pairs}.jsonl + build_report.json, and
records the reserve pair ids per split in manifest/reserve/splits.json. Reserve pairs are NEVER counted toward the
100-pair floor unless manifest/backfill.json records a backfill for a review-rejected pair (tt_verify enforces).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_build as B  # noqa: E402
import tt_corpus as C  # noqa: E402
import tt_plan as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    S, D = Path(a.scratch), Path(a.corpus)
    plan = P.build_reserve_plan()
    rows = {json.loads(l)["render_id"]: json.loads(l) for l in (S / "renders.jsonl").read_text().splitlines()}
    main_recs = {r["recording_id"]: r for r in map(json.loads, (D / "manifest" / "recordings.jsonl").read_text().splitlines())}
    main_clips = [json.loads(x) for x in (D / "manifest" / "clips.jsonl").read_text().splitlines()]
    spk_split = {c["speaker_id"]: c["split"] for c in main_clips}
    asst_split = {}
    for c in main_clips:
        asst_split.setdefault(c["assistant"]["recording_id"], c["split"])
    cache: dict[str, np.ndarray] = {}

    def main_pcm(rid):
        if rid not in cache:
            cache[rid] = A.load_audio(D, main_recs[rid]["path"])[0].samples[:, 0]
        return cache[rid]

    pcm, trims = {}, {}
    for j in plan["jobs"]:
        rid = j["render_id"]
        if rid in rows and rows[rid].get("qa_ok"):
            x = A.read_wav(S / "rec16" / f"{rid}.wav").samples[:, 0]
            pcm[rid] = x
            vs, ve = A.trim_to_speech(x, A.SR, 10, -50.0, 0)
            t0 = max(0, vs - B.PAD_MS * A.SR // 1000)
            t1 = min(len(x), ve + B.PAD_MS * A.SR // 1000)
            trims[rid] = {"trim": (int(t0), int(t1)), "voiced_start": int(vs - t0), "voiced_end": int(ve - t0)}
    report = {"rejected_pairs": [], "cuts": {}, "invariants": {}}
    clips, pairs, used = [], [], set()
    for pr in plan["pairs"]:
        pid, fam, urid = pr["pair_id"], pr["family"], pr["user_recording"]
        if urid not in pcm:
            report["rejected_pairs"].append({"pair_id": pid, "reason": "render_missing_or_failed_qa"})
            continue
        problems = [x for x in (pr["assistant_a"], pr["assistant_b"]) if x not in main_recs or asst_split.get(x) != pr["split"]]
        if problems or spk_split.get(pr["voice_id"]) != pr["split"]:
            report["rejected_pairs"].append({"pair_id": pid, "reason": f"split_consistency:{problems}"})
            continue
        tr = trims[urid]
        cut, cut_info = None, None
        if fam in ("CUT_IC", "CUT_WAIT"):
            cut_info = B.choose_cut(pcm[urid], tr["trim"], P.norm_words(pr["text_full"]), pr["prefix_words"], rows[urid].get("asr", {}).get("words", []))
            report["cuts"][pid] = cut_info
            if cut_info["status"] != "ok":
                report["rejected_pairs"].append({"pair_id": pid, "reason": f"cut:{cut_info['status']}", "detail": cut_info})
                continue
            cut = cut_info["cut_rel_sample"]
        members = []
        for role, cutv, aid, mode, label in (("a", None, pr["assistant_a"], pr["assistant_mode_a"], pr["label_a"]),
                                              ("b", cut, pr["assistant_b"], pr["assistant_mode_b"], pr["label_b"])):
            arr, lay = C.assemble_clip(pcm[urid], tr["trim"], cutv, main_pcm(aid), mode)
            members.append((role, cutv, aid, mode, label, arr, lay))
        viol = C.check_pair_invariants(fam, members[0][5], members[0][6], members[0][1], members[1][5], members[1][6], members[1][1])
        report["invariants"][pid] = viol
        if viol:
            report["rejected_pairs"].append({"pair_id": pid, "reason": "invariants", "detail": viol})
            continue
        ids = []
        for role, cutv, aid, mode, label, arr, lay in members:
            cid = f"{pid}-{role}"
            pc = A.Pcm(arr, A.SR)
            clip = {"clip_id": cid, "pair_id": pid, "pair_role": role, "stratum": "reserve", "split": pr["split"], "family": fam, "label": label,
                    "speaker_id": pr["voice_id"], "dialogue_id": f"d-{pid}",
                    "user": {"recording_id": urid, "trim": list(tr["trim"]), "cut": cutv, "fade": C.FADE},
                    "assistant": {"recording_id": aid, "mode": mode},
                    "layout": {"lead_in": C.LEAD_IN, "gap_after_assist": C.GAP_AFTER_ASSIST, "bc_onset_in_assist": C.BC_ONSET_IN_ASSIST, "trail": C.TRAIL,
                               "user_onset": lay.user_onset, "user_len": lay.user_len, "user_end": lay.user_end, "total": lay.total,
                               "assistant_start": lay.assistant_start, "assistant_end": lay.assistant_end},
                    "gold": B.gold_for(label, lay, tr, cutv),
                    "text": {"user_spoken": pr["text_prefix"] if cutv is not None else pr["text_full"], "user_full": pr["text_full"], "assistant": main_recs[aid]["text"]},
                    "pcm": {"sample_rate": A.SR, "channels": 2, "n_samples": int(arr.shape[0]), "pcm_sha256": pc.pcm_sha256(),
                            "ch_sha256": [pc.channel_sha256(0), pc.channel_sha256(1)]}}
            if cut_info and cutv is not None:
                clip["cut_decision"] = cut_info
            clips.append(clip)
            ids.append(cid)
        used.add(urid)
        pairs.append({"pair_id": pid, "family": fam, "split": pr["split"], "stratum": "reserve", "contrast": pr["contrast"], "speaker_id": pr["voice_id"],
                      "dialogue_id": f"d-{pid}", "user_recording": urid, "clips": ids, "labels": {"a": pr["label_a"], "b": pr["label_b"]}, "language": "en"})
    out = D / "manifest" / "reserve"
    out.mkdir(parents=True, exist_ok=True)
    members_tar, rec_rows = {}, []
    for rid in sorted(used):
        r = rows[rid]
        data = (S / "rec16" / f"{rid}.wav").read_bytes()
        members_tar[f"{rid}.wav"] = data
        pc = A.read_wav_bytes(data)
        rec_rows.append({"recording_id": rid, "role": "user", "stratum": "reserve", "speaker_id": r["voice_id"], "split": r["split"], "text": r["text"],
                         "path": f"audio/reserve/user.tar#{rid}.wav", "sample_rate": pc.sample_rate, "channels": 1, "n_samples": pc.n_samples,
                         "pcm_sha256": pc.pcm_sha256(), "file_sha256": hashlib.sha256(data).hexdigest(), "trim": list(trims[rid]["trim"]),
                         "qa_override": r.get("qa_override"),
                         "provenance": {"tts": {"binary": "qwen-tts", "tree": "qwentts.cpp@production-speech-v1", "ref_speaker": f"librispeech-test-clean:{r['voice_id']}:{Path(r['ref_wav']).stem}",
                                                "ref_wav_sha256": r["ref_sha256"], "seed": r["seed_final"], "attempts": r["history"], "raw24_pcm_sha256": r["raw24_pcm_sha256"]},
                                        "resample": "scipy.signal.resample_poly up=2 down=3 (24k->16k), float64 -> round -> int16", "asr_word_timing": r.get("asr")}})
    A.write_tar(D / "audio" / "reserve" / "user.tar", members_tar)
    for fn, rowsx in (("recordings.jsonl", rec_rows), ("clips.jsonl", clips), ("pairs.jsonl", pairs)):
        with open(out / fn, "w") as f:
            for x in rowsx:
                f.write(json.dumps(x, sort_keys=True) + "\n")
    splits = {sp: sorted(p["pair_id"] for p in pairs if p["split"] == sp) for sp in P.SPLITS}
    (out / "splits.json").write_text(json.dumps(splits, indent=1, sort_keys=True) + "\n")
    (out / "build_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    bf = D / "manifest" / "backfill.json"
    if not bf.exists():
        bf.write_text(json.dumps({"policy": "Reserve pairs count toward the 100-pair floor ONLY through a record here: a review rejection of a main pair, and a backfill naming the reserve pair that replaces it (same family and split). Nothing else counts them.",
                                  "rejections": [], "backfills": []}, indent=1) + "\n")
    print(f"reserve: {len(pairs)} pairs ({len(report['rejected_pairs'])} rejected) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
