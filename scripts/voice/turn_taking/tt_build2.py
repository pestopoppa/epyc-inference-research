#!/usr/bin/env python3
"""CS-45 corpus builder v2 (replaces tt_build.py + tt_build_reserve.py after the 2026-10-20 human review).

Two stages, joined by the INDEPENDENT gates (tt_qa_gates.py), so admission never rests on the builder's own checks:

  candidates : every rendered variant of every planned pair (main + reserve) is assembled into a CANDIDATE corpus
               (ids '<pair>.v<k>-a|b'); cut placement uses ASR-A (faster-whisper large-v3-turbo) word times, an energy
               valley, a speech-relative trim and the v2 clip layout (tt_corpus.py).
  final      : reads the gates report (ASR-B = faster-whisper large-v3 on the candidate clips), keeps for each pair the
               FIRST variant whose clips all pass, and writes the durable corpus (audio tars, manifests, qa report).
               Pairs with no passing variant are simply absent (the report lists them); nothing is waived.

  python3 tt_build2.py candidates --scratch DIR[,DIR...] --corpus data/turn-taking-cs45-v1 --cand /mnt/raid0/llm/tmp/cs45/cand
  python3 tt_build2.py final      --scratch DIR[,DIR...] --corpus data/turn-taking-cs45-v1 --cand CAND --gates report.json
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
import tt_qa_gates as Q  # noqa: E402

CUT_LEVEL_DB_BELOW_P95 = 12.0  # same rule as the independent gate (U2); the gate re-checks it
MIN_PREFIX_SPEECH_S = 0.25
FRAME = 160


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    fr = x[: n * FRAME].astype(np.float64).reshape(n, FRAME) / 32768.0
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1) + 1e-20) + 1e-12)


def map_words(intended: list[str], asr_words: list[dict]) -> dict[int, int]:
    """intended word index -> index into asr_words (ASR words may be numerals / split words: expand to tokens first)."""
    flat, idx = [], []
    for k, w in enumerate(asr_words):
        for t in Q.norm_words(w["w"]):
            flat.append(t)
            idx.append(k)
    sm = difflib.SequenceMatcher(a=intended, b=flat, autojunk=False)
    m: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = idx[j1 + k]
        elif tag == "replace" and j2 > j1:  # unequal block ('sixty eight' -> '68'): anchor its two ends
            m[i1] = idx[j1]
            m[i2 - 1] = idx[j2 - 1]
    return m


def cut_options(rec: np.ndarray, trim: tuple[int, int], asr: dict, max_options: int = 3) -> list[dict]:
    """Ordered cut options from the ASR-A generate-and-test candidates (tt_align_fw.py). Eligible: the exact clip segment
    transcribes as the expected prefix AND the cut-point level is >= CUT_LEVEL_DB_BELOW_P95 below the recording's p95.
    Order: candidates whose remainder also transcribes as the expected suffix first (deepest first), then prefix-only
    candidates. Options are >= 60 ms apart; every option is gated independently by ASR-B (prefix AND suffix exact)."""
    cands = asr.get("cut_candidates") or []
    ok = [c for c in cands if c["prefix_ok"] and c["level_db"] <= c["p95_db"] - CUT_LEVEL_DB_BELOW_P95]
    ok.sort(key=lambda c: (not c["suffix_ok"], c["level_db"]))
    out = []
    for c in ok:
        if any(abs(c["cut_abs_sample"] - o["cut_abs_sample"]) < 960 for o in out):
            continue
        if c["cut_abs_sample"] - trim[0] != c["cut_rel_sample"]:
            continue
        out.append(c)
        if len(out) == max_options:
            break
    return [{"status": "ok", "cut_abs_sample": c["cut_abs_sample"], "cut_rel_sample": c["cut_rel_sample"], "cut_db": c["level_db"], "recording_p95_db": c["p95_db"],
             "boundary_estimate_s": c["boundary_estimate_s"], "asrA_suffix_ok": c["suffix_ok"], "n_candidates": len(cands), "aligner": "faster-whisper large-v3-turbo",
             "selection": "generate-and-test (ASR-A prefix exact [+ suffix exact]) then deepest energy minimum"} for c in out]


def gold_for(label: str, lay: C.Layout, tr: dict, cut: int | None) -> dict:
    onset = lay.user_onset
    v_start = (onset + tr["voiced_start"]) / A.SR
    v_end = (onset + (cut if cut is not None else tr["voiced_end"])) / A.SR
    g = {"user_speech_start_s": round(v_start, 4), "user_speech_end_s": round(v_end, 4), "clip_end_s": round(lay.total / A.SR, 4),
         "assistant_speaking_at_user_onset": lay.assistant_end is not None and lay.assistant_end > onset,
         "assistant_speech_end_s": None if lay.assistant_end is None else round(lay.assistant_end / A.SR, 4)}
    if label == "complete":
        g.update({"eot_gold_s": g["user_speech_end_s"], "hold_window_s": None, "floor_holder_during_user": "user"})
    elif label in ("incomplete", "wait"):
        g.update({"eot_gold_s": None, "hold_window_s": [g["user_speech_end_s"], g["clip_end_s"]], "floor_holder_during_user": "user"})
    elif label == "backchannel":
        g.update({"eot_gold_s": None, "hold_window_s": [g["user_speech_start_s"], g["clip_end_s"]], "floor_holder_during_user": "assistant"})
    return g


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_scratch(dirs: list[Path]):
    rows, wavs = {}, {}
    for d in dirs:
        p = d / "renders.jsonl"
        if not p.exists():
            continue
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if r.get("qa_ok") and (d / "rec16" / f"{r['render_id']}.wav").exists():
                rows[r["render_id"]] = r
                wavs[r["render_id"]] = d / "rec16" / f"{r['render_id']}.wav"
    return rows, wavs


def all_plan_pairs():
    pr = P.build_plan()["pairs"] + P.build_reserve_plan()["pairs"]
    return pr


def assistant_lookup(D: Path):
    recs = {r["recording_id"]: r for r in map(json.loads, (D / "manifest" / "recordings.jsonl").read_text().splitlines())}
    recs.update({r["recording_id"]: r for r in map(json.loads, (D / "manifest" / "reserve" / "recordings.jsonl").read_text().splitlines())} if (D / "manifest" / "reserve" / "recordings.jsonl").exists() else {})
    cache = {}

    def get(rid):
        if rid not in cache:
            cache[rid] = A.load_audio(D, recs[rid]["path"])[0].samples[:, 0]
        return cache[rid]

    return recs, get


def build_pair_variant(pr: dict, variant: int, rows: dict, wavs: dict, asst_get, asst_recs, option: int = 0) -> dict:
    """Return {'clips':[...], 'pair':..., 'user_rec':..., 'report':...} or {'reject': reason, ...}."""
    base = pr["user_recording"]
    rid = base + (f".v{variant}" if variant else "")
    if rid not in rows:
        return {"reject": "no_render", "variant": variant}
    x = A.read_wav(wavs[rid]).samples[:, 0]
    t0, t1, vs, ve = A.speech_bounds(x)
    tr = {"trim": (int(t0), int(t1)), "voiced_start": int(vs), "voiced_end": int(ve)}
    fam = pr["family"]
    if fam not in ("CUT_IC", "CUT_WAIT") and option > 0:
        return {"reject": "no_option", "variant": variant}
    cut, cut_info = None, None
    if fam in ("CUT_IC", "CUT_WAIT"):
        opts = cut_options(x, tr["trim"], rows[rid].get("asr", {}))
        if option >= len(opts):
            return {"reject": "no_option" if opts else "cut:no_eligible_candidate", "variant": variant}
        cut_info = opts[option]
        cut = cut_info["cut_rel_sample"]
        pre_speech = float((frame_db(x[tr["trim"][0] : tr["trim"][0] + cut]) >= (cut_info["recording_p95_db"] - 20.0)).sum() * 0.01)
        if pre_speech < MIN_PREFIX_SPEECH_S:
            return {"reject": "prefix_has_no_speech", "variant": variant, "detail": cut_info}
    seg_len = len(C.user_segment(x, tr["trim"][0], tr["trim"][1], None))
    tail = None
    if fam == "CTX_BC":
        tail = C.ctx_tail(len(asst_get(pr["assistant_b"])), seg_len)
    members = []
    try:
        for role, cutv, aid, mode, label in (("a", None, pr["assistant_a"], pr["assistant_mode_a"], pr["label_a"]),
                                              ("b", cut, pr["assistant_b"], pr["assistant_mode_b"], pr["label_b"])):
            arr, lay = C.assemble_clip(x, tr["trim"], cutv, asst_get(aid), mode, tail)
            members.append((role, cutv, aid, mode, label, arr, lay))
    except ValueError as e:
        return {"reject": f"assemble:{e}", "variant": variant}
    viol = C.check_pair_invariants(fam, members[0][5], members[0][6], members[0][1], members[1][5], members[1][6], members[1][1])
    if viol:
        return {"reject": "invariants", "variant": variant, "detail": viol}
    return {"members": members, "tr": tr, "x": x, "cut_info": cut_info, "rid": rid, "variant": variant, "option": option, "tail": tail}


def clip_rows(pr: dict, built: dict, id_suffix: str, stratum: str, asst_recs: dict) -> list[dict]:
    pid, fam, urid = pr["pair_id"], pr["family"], built["rid"]
    out = []
    for role, cutv, aid, mode, label, arr, lay in built["members"]:
        pc = A.Pcm(arr, A.SR)
        c = {"clip_id": f"{pid}{id_suffix}-{role}", "pair_id": pid + id_suffix, "pair_role": role, "stratum": stratum, "split": pr["split"], "family": fam, "label": label,
             "speaker_id": pr["voice_id"], "dialogue_id": f"d-{pid}",
             "user": {"recording_id": urid, "trim": list(built["tr"]["trim"]), "cut": cutv, "fade": C.FADE},
             "assistant": {"recording_id": aid, "mode": mode},
             "layout": {"lead_in": C.LEAD_IN, "gap_after_assist": C.GAP_AFTER_ASSIST, "bc_onset_in_assist": C.BC_ONSET_IN_ASSIST, "trail": C.TRAIL,
                        "tail": lay.tail, "end_pad": C.END_PAD, "user_onset": lay.user_onset, "user_len": lay.user_len, "user_end": lay.user_end,
                        "total": lay.total, "assistant_start": lay.assistant_start, "assistant_end": lay.assistant_end},
             "gold": gold_for(label, lay, built["tr"], cutv),
             "text": {"user_spoken": pr["text_prefix"] if cutv is not None else pr["text_full"], "user_full": pr["text_full"], "assistant": asst_recs[aid]["text"]},
             "pcm": {"sample_rate": A.SR, "channels": 2, "n_samples": int(arr.shape[0]), "pcm_sha256": pc.pcm_sha256(),
                     "ch_sha256": [pc.channel_sha256(0), pc.channel_sha256(1)]}}
        if built["cut_info"] and cutv is not None:
            c["cut_decision"] = built["cut_info"]
        out.append(c)
    return out


def rec_row(rid_final: str, rid_src: str, row: dict, wav: Path, tr: dict, stratum: str, tar: str) -> tuple[dict, bytes]:
    data = wav.read_bytes()
    pc = A.read_wav_bytes(data)
    return ({"recording_id": rid_final, "role": "user", "stratum": stratum, "speaker_id": row["voice_id"], "split": row["split"], "text": row["text"],
             "path": f"{tar}#{rid_final}.wav", "sample_rate": pc.sample_rate, "channels": 1, "n_samples": pc.n_samples, "pcm_sha256": pc.pcm_sha256(),
             "file_sha256": sha_bytes(data), "trim": list(tr["trim"]),
             "provenance": {"render_id": rid_src, "variant": row.get("variant", 0),
                            "tts": {"binary": "qwen-tts", "tree": "qwentts.cpp@production-speech-v1", "ref_speaker": f"librispeech-test-clean:{row['voice_id']}:{Path(row['ref_wav']).stem}",
                                    "ref_wav_sha256": row["ref_sha256"], "seed": row["seed_final"], "attempts": row["history"], "raw24_pcm_sha256": row.get("raw24_pcm_sha256")},
                            "resample": "scipy.signal.resample_poly up=2 down=3 (24k->16k), float64 -> round -> int16",
                            "asr_a_word_timing": row.get("asr")}}, data)


def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for x in rows:
            f.write(json.dumps(x, sort_keys=True) + "\n")


def vtag(v: int, o: int) -> str:
    return f".v{v}" if o == 0 else f".v{v}c{o}"


def parse_vtag(pair_id: str) -> tuple[str, int, int]:
    import re

    m = re.match(r"^(.*)\.v(\d+)(?:c(\d+))?$", pair_id)
    return m.group(1), int(m.group(2)), int(m.group(3) or 0)


def select_pairs(cand: Path, gate_by_hash: dict):
    cand_pairs = [json.loads(x) for x in (cand / "manifest" / "pairs.jsonl").read_text().splitlines()]
    cand_clips = {c["clip_id"]: c for c in map(json.loads, (cand / "manifest" / "clips.jsonl").read_text().splitlines())}
    chosen, unfilled, per_pair = {}, [], {}
    for pr in all_plan_pairs():
        pid = pr["pair_id"]
        opts = [p for p in cand_pairs if parse_vtag(p["pair_id"])[0] == pid]
        per_pair[pid] = []
        for p in sorted(opts, key=lambda p: parse_vtag(p["pair_id"])[1:]):
            res = []
            for c in p["clips"]:
                e = gate_by_hash.get(c)
                # a gate result counts only for the audio it was computed on
                res.append(e if e is not None and e["pcm_sha256"] == cand_clips[c]["pcm"]["pcm_sha256"] else None)
            ok = all(r is not None and r["pass"] for r in res)
            per_pair[pid].append({"variant": p["pair_id"].split(".v")[1], "pass": ok,
                                  "failed_gates": sorted({g for r in res if r for g, v in r["gates"].items() if not v["ok"]})})
            if ok and pid not in chosen:
                chosen[pid] = (pr, p)
        if pid not in chosen:
            unfilled.append(pid)
    return chosen, unfilled, per_pair


def gates_by_hash(path) -> dict:
    """clip_id -> gate entry (the entry carries the PCM-SHA-256 it was computed on; callers must compare it)."""
    rep = json.loads(Path(path).read_text())
    out = {}
    for part in rep.values():
        for cid, e in part["clips"].items():
            out[cid] = e
    return out


def stage_candidates(a) -> int:
    D, cand = Path(a.corpus), Path(a.cand)
    rows, wavs = load_scratch([Path(s) for s in a.scratch.split(",")])
    asst_recs, asst_get = assistant_lookup(D)
    if cand.exists():
        shutil.rmtree(cand)
    clips, pairs, recs, tar_user = [], [], [], {}
    stats = {"pairs": 0, "variants_built": 0, "rejected": []}
    variants = sorted({r.get("variant", 0) for r in rows.values()})
    for pr in all_plan_pairs():
        stats["pairs"] += 1
        stratum = "reserve" if pr["pair_id"].startswith("tt1-r") else "synthesized"
        for v, o in [(v, o) for v in variants for o in range(3)]:
            b = build_pair_variant(pr, v, rows, wavs, asst_get, asst_recs, o)
            if "reject" in b:
                if b["reject"] not in ("no_render", "no_option"):
                    stats["rejected"].append({"pair_id": pr["pair_id"], "variant": v, "option": o, "reason": b["reject"]})
                continue
            stats["variants_built"] += 1
            suffix = vtag(v, o)
            cs = clip_rows(pr, b, suffix, stratum, asst_recs)
            clips += cs
            if f"{b['rid']}.wav" not in tar_user:
                rr, data = rec_row(b["rid"], b["rid"], rows[b["rid"]], wavs[b["rid"]], b["tr"], stratum, "audio/synth/user.tar")
                recs.append(rr)
                tar_user[f"{b['rid']}.wav"] = data
            pairs.append({"pair_id": pr["pair_id"] + suffix, "family": pr["family"], "split": pr["split"], "stratum": stratum, "contrast": pr["contrast"],
                          "speaker_id": pr["voice_id"], "dialogue_id": f"d-{pr['pair_id']}", "user_recording": b["rid"], "clips": [c["clip_id"] for c in cs],
                          "labels": {"a": pr["label_a"], "b": pr["label_b"]}, "language": "en"})
    used_asst = {c["assistant"]["recording_id"] for c in clips}
    amem = {}
    for rid in sorted(used_asst):
        amem[f"{rid}.wav"] = A.load_audio_bytes(D, asst_recs[rid]["path"])
        r = dict(asst_recs[rid])
        r["path"] = f"audio/synth/assistant.tar#{rid}.wav"
        recs.append(r)
    A.write_tar(cand / "audio" / "synth" / "user.tar", tar_user)
    A.write_tar(cand / "audio" / "synth" / "assistant.tar", amem)
    write_jsonl(cand / "manifest" / "recordings.jsonl", recs)
    write_jsonl(cand / "manifest" / "clips.jsonl", clips)
    write_jsonl(cand / "manifest" / "pairs.jsonl", pairs)
    (cand / "manifest" / "candidate_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"candidates: {len(pairs)} pair-variants, {len(clips)} clips; build-stage rejects: {len(stats['rejected'])} -> {cand}")
    return 0


def stage_final(a) -> int:
    D, cand = Path(a.corpus), Path(a.cand)
    rows, wavs = load_scratch([Path(s) for s in a.scratch.split(",")])
    rep = json.loads(Path(a.gates).read_text())
    gate_by_hash = gates_by_hash(a.gates)
    cand_clips = {c["clip_id"]: c for c in map(json.loads, (cand / "manifest" / "clips.jsonl").read_text().splitlines())}
    asst_recs, asst_get = assistant_lookup(cand)  # the live corpus is rewritten below; the candidate set holds the same assistant audio
    chosen, unfilled, per_pair = select_pairs(cand, gate_by_hash)
    # ---- write the durable corpus
    for sub in ("audio/synth", "audio/reserve", "manifest/reserve", "qa"):
        shutil.rmtree(D / sub, ignore_errors=True)
    for f in ("recordings.jsonl", "clips.jsonl", "pairs.jsonl", "splits.json", "build_report.json"):
        (D / "manifest" / f).unlink(missing_ok=True)
    main_c, main_p, main_r, res_c, res_p, res_r = [], [], [], [], [], []
    tar = {"main": {}, "reserve": {}}
    qa_clips = {}
    used_asst = set()
    for pid, (pr, p) in sorted(chosen.items()):
        reserve = pid.startswith("tt1-r")
        k = "reserve" if reserve else "main"
        stratum = "reserve" if reserve else "synthesized"
        _, variant, option = parse_vtag(p["pair_id"])
        b = build_pair_variant(pr, variant, rows, wavs, asst_get, asst_recs, option)
        assert "reject" not in b, (pid, b)
        cs = clip_rows(pr, b, "", stratum, asst_recs)
        for c, old in zip(cs, p["clips"]):
            assert c["pcm"]["pcm_sha256"] == cand_clips[old]["pcm"]["pcm_sha256"], "final clip differs from the gated candidate"
            e = gate_by_hash[old]
            assert e["pcm_sha256"] == c["pcm"]["pcm_sha256"]
            qa_clips[c["clip_id"]] = {"candidate_clip_id": old, "pcm_sha256": c["pcm"]["pcm_sha256"], "pass": e["pass"], "gates": {g: v["ok"] for g, v in e["gates"].items()}}
            used_asst.add(c["assistant"]["recording_id"])
        final_rid = pr["user_recording"]  # variant suffix dropped; the source render id is kept in provenance
        rr, data = rec_row(final_rid, b["rid"], rows[b["rid"]], wavs[b["rid"]], b["tr"], stratum, "audio/reserve/user.tar" if reserve else "audio/synth/user.tar")
        for c in cs:
            c["user"]["recording_id"] = final_rid
        (res_r if reserve else main_r).append(rr)
        tar[k][f"{final_rid}.wav"] = data
        (res_c if reserve else main_c).extend(cs)
        (res_p if reserve else main_p).append({"pair_id": pid, "family": pr["family"], "split": pr["split"], "stratum": stratum, "contrast": pr["contrast"],
                                               "speaker_id": pr["voice_id"], "dialogue_id": f"d-{pid}", "user_recording": final_rid, "clips": [c["clip_id"] for c in cs],
                                               "labels": {"a": pr["label_a"], "b": pr["label_b"]}, "language": "en", "chosen_variant": variant, "chosen_cut_option": option})
    # assistants (shared by main and reserve; they live in the main archive)
    amem = {}
    for rid in sorted(used_asst):
        amem[f"{rid}.wav"] = A.load_audio_bytes(cand, asst_recs[rid]["path"])
        r = dict(asst_recs[rid])
        r["path"] = f"audio/synth/assistant.tar#{rid}.wav"
        main_r.append(r)
    A.write_tar(D / "audio" / "synth" / "user.tar", tar["main"])
    A.write_tar(D / "audio" / "synth" / "assistant.tar", amem)
    if tar["reserve"]:
        A.write_tar(D / "audio" / "reserve" / "user.tar", tar["reserve"])
    write_jsonl(D / "manifest" / "recordings.jsonl", main_r)
    write_jsonl(D / "manifest" / "clips.jsonl", main_c)
    write_jsonl(D / "manifest" / "pairs.jsonl", main_p)
    write_jsonl(D / "manifest" / "reserve" / "recordings.jsonl", res_r)
    write_jsonl(D / "manifest" / "reserve" / "clips.jsonl", res_c)
    write_jsonl(D / "manifest" / "reserve" / "pairs.jsonl", res_p)
    splits = {"rule": "voices are partitioned into selection/confirmation speaker sets; every pair, dialogue, source recording and assistant context "
                      "inherits its split from that partition; nothing crosses", "plan_seed": P.PLAN_SEED}
    for sp in P.SPLITS:
        splits[sp] = {"speakers": sorted({p["speaker_id"] for p in main_p if p["split"] == sp}), "dialogues": sorted(p["dialogue_id"] for p in main_p if p["split"] == sp),
                      "pairs": sorted(p["pair_id"] for p in main_p if p["split"] == sp), "user_recordings": sorted(p["user_recording"] for p in main_p if p["split"] == sp),
                      "assistant_recordings": sorted({c["assistant"]["recording_id"] for c in main_c if c["split"] == sp})}
    (D / "manifest" / "splits.json").write_text(json.dumps(splits, indent=1, sort_keys=True) + "\n")
    (D / "manifest" / "reserve" / "splits.json").write_text(json.dumps({sp: sorted(p["pair_id"] for p in res_p if p["split"] == sp) for sp in P.SPLITS}, indent=1, sort_keys=True) + "\n")
    bf = D / "manifest" / "backfill.json"
    if not bf.exists():
        bf.write_text(json.dumps({"policy": "Reserve pairs count toward the 100-pair floor ONLY through a record here: a review rejection of a main pair, and a backfill naming the reserve pair that replaces it (same family and split). Nothing else counts them.", "rejections": [], "backfills": []}, indent=1) + "\n")
    qa = {"gates_version": rep["main"]["gates_version"] if "main" in rep else next(iter(rep.values()))["gates_version"], "asr_b": next(iter(rep.values()))["asr_b"],
          "built_with_asr_a": "faster-whisper large-v3-turbo int8 CPU (tt_align_fw.py)", "scored_with": "whisper.cpp large-v3-turbo (pinned)",
          "n_clips": len(qa_clips), "all_pass": all(v["pass"] for v in qa_clips.values()), "clips": qa_clips}
    (D / "qa").mkdir(parents=True, exist_ok=True)
    (D / "qa" / "qa_gates_report.json").write_text(json.dumps(qa, indent=1, sort_keys=True) + "\n")
    (D / "qa" / "selection_report.json").write_text(json.dumps({"unfilled_pairs": unfilled, "per_pair_variants": per_pair}, indent=1, sort_keys=True) + "\n")
    print(f"final: main {len(main_p)} pairs ({len(main_c)} clips), reserve {len(res_p)} pairs; unfilled {len(unfilled)}: {unfilled[:12]}")
    return 0


def stage_unfilled(a) -> int:
    chosen, unfilled, per_pair = select_pairs(Path(a.cand), gates_by_hash(a.gates))
    ids = {pr["pair_id"]: pr["user_recording"] for pr in all_plan_pairs()}
    print(",".join(ids[p] for p in unfilled))
    print(f"unfilled {len(unfilled)} of {len(ids)}; chosen {len(chosen)}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "final", "unfilled"])
    ap.add_argument("--scratch", default="")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--gates")
    a = ap.parse_args()
    return {"candidates": stage_candidates, "final": stage_final, "unfilled": stage_unfilled}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
