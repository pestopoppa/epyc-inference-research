#!/usr/bin/env python3
"""CS-45 corpus conformance gate: seal + verify + refusal.

  python3 tt_verify.py seal   --corpus data/turn-taking-cs45-v1      # (re)write corpus.json + SHA256SUMS
  python3 tt_verify.py verify --corpus data/turn-taking-cs45-v1      # exit 0 only if EVERY check passes

verify never trusts a recorded number it can recompute: PCM hashes are recomputed from the WAV
bytes, every clip is re-assembled from (recording, recipe) and re-hashed, every pair is checked as
an exact splice on the assembled arrays, and disjointness is derived from the rows (splits.json is
cross-checked, never believed). No model, server or network is touched.
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
import tt_corpus as C  # noqa: E402

CORPUS_ID = "turn-taking-cs45-v1"
MIN_PAIRS = 100
SEAL_EXCLUDE = {"corpus.json", "SHA256SUMS"}
STRATA = ("synthesized", "human_recorded")
ADVANCE_ELIGIBLE = ["human_recorded"]
RESERVE_POLICY_ID = "cs45-reserve-v1"
SYNTH_FAMILIES = {"CUT_IC": ("complete", "incomplete"), "CUT_WAIT": ("complete", "wait"), "CTX_BC": ("complete", "backchannel")}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def tree_files(D: Path) -> list[str]:
    out = []
    for p in sorted(D.rglob("*")):
        if p.is_file() and p.name not in SEAL_EXCLUDE and ".git" not in p.parts and "__pycache__" not in p.parts:
            out.append(p.relative_to(D).as_posix())
    return out


def seal(D: Path) -> dict:
    files = {rel: sha256_file(D / rel) for rel in tree_files(D)}
    pairs = read_jsonl(D / "manifest" / "pairs.jsonl")
    fam = {}
    for p in pairs:
        fam[p["family"]] = fam.get(p["family"], 0) + 1
    meta = {
        "corpus_id": CORPUS_ID,
        "format": "tt-corpus-1",
        "instrument_version": C.INSTRUMENT_VERSION,
        "constants": {"sample_rate": A.SR, "lead_in": C.LEAD_IN, "gap_after_assist": C.GAP_AFTER_ASSIST, "bc_onset_in_assist": C.BC_ONSET_IN_ASSIST,
                      "trail": C.TRAIL, "fade": C.FADE, "assist_fade": C.ASSIST_FADE, "end_pad": C.END_PAD, "n_cont_min": C.N_CONT_MIN},
        "min_pairs_required": MIN_PAIRS,
        "strata": list(STRATA),
        "advance_eligible_strata": ADVANCE_ELIGIBLE,
        "counts": {"synthesized_pairs": sum(1 for p in pairs if p["stratum"] == "synthesized"), "by_family": fam,
                   "human_recorded_clips": len(read_jsonl(D / "manifest" / "human_recorded.jsonl"))},
        "pcm_binding": "SHA-256 of decoded interleaved little-endian int16 samples; sample rate, channels and sample count recorded with it (Annex S P-STT-1 1.2)",
        "reserve": {"policy_id": RESERVE_POLICY_ID, "pairs": len(read_jsonl(D / "manifest" / "reserve" / "pairs.jsonl")),
                    "policy": "stratum 'reserve': synthesized backfill pairs, same rules and voice partition. Used ONLY to replace main pairs that the human review rejects, through manifest/backfill.json (a recorded rejection plus a backfill of the same family and split). Never counted toward the 100-pair floor otherwise; never evidence for advancing an arm."},
        "files": files,
    }
    (D / "corpus.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    lines = [f"{h}  {rel}" for rel, h in sorted(files.items())]
    lines.append(f"{sha256_file(D / 'corpus.json')}  corpus.json")
    (D / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return meta


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail=None):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail if not ok else None})

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def verify(D: Path, label_contract_path: Path | None = None) -> Report:
    R = Report()
    corpus_p = D / "corpus.json"
    if not corpus_p.exists():
        R.add("corpus_json_present", False, "corpus.json missing")
        return R
    meta = json.loads(corpus_p.read_text())

    # V1 file digests ---------------------------------------------------------------------------
    bad, extra = [], []
    listed = set(meta["files"])
    for rel, h in meta["files"].items():
        p = D / rel
        if not p.exists():
            bad.append(f"missing:{rel}")
        elif sha256_file(p) != h:
            bad.append(f"digest:{rel}")
    for rel in tree_files(D):
        if rel not in listed:
            extra.append(rel)
    sums = D / "SHA256SUMS"
    if not sums.exists():
        bad.append("missing:SHA256SUMS")
    elif f"{sha256_file(corpus_p)}  corpus.json" not in sums.read_text().splitlines():
        bad.append("digest:corpus.json-vs-SHA256SUMS")
    R.add("file_digests", not bad and not extra, {"bad": bad[:20], "unlisted": extra[:20]})

    recs = {r["recording_id"]: r for r in read_jsonl(D / "manifest" / "recordings.jsonl")}
    clips = read_jsonl(D / "manifest" / "clips.jsonl")
    pairs = read_jsonl(D / "manifest" / "pairs.jsonl")
    human = read_jsonl(D / "manifest" / "human_recorded.jsonl")
    rres = {r["recording_id"]: r for r in read_jsonl(D / "manifest" / "reserve" / "recordings.jsonl")}
    rclips = read_jsonl(D / "manifest" / "reserve" / "clips.jsonl")
    rpairs = read_jsonl(D / "manifest" / "reserve" / "pairs.jsonl")
    bfp = D / "manifest" / "backfill.json"
    backfill = json.loads(bfp.read_text()) if bfp.exists() else {"rejections": [], "backfills": []}
    all_recs = {**recs, **rres}

    # V2 recordings: PCM recomputed from the WAV bytes ------------------------------------------
    pcm: dict[str, np.ndarray] = {}
    bad = []
    for rid, r in all_recs.items():
        try:
            w, fsha = A.load_audio(D, r["path"])
        except Exception as e:  # noqa: BLE001
            bad.append(f"{rid}:unreadable:{e}")
            continue
        if (w.sample_rate, w.channels, w.n_samples) != (r["sample_rate"], r["channels"], r["n_samples"]):
            bad.append(f"{rid}:format")
        if w.pcm_sha256() != r["pcm_sha256"]:
            bad.append(f"{rid}:pcm_sha256")
        if fsha != r["file_sha256"]:
            bad.append(f"{rid}:file_sha256")
        pcm[rid] = w.samples[:, 0]
    if len(all_recs) != len(recs) + len(rres):
        bad.append("recording id collision between main and reserve")
    R.add("recordings_pcm_bound", not bad and len(recs) > 0, bad[:20])

    # V3 every clip re-assembles to its recorded PCM-SHA-256 -------------------------------------
    bad = []
    asm: dict[str, tuple[np.ndarray, C.Layout]] = {}
    for c in clips + rclips:
        try:
            u, a_ = pcm[c["user"]["recording_id"]], pcm[c["assistant"]["recording_id"]]
            arr, lay = C.assemble_row(c, u, a_)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{c['clip_id']}:assemble:{e}")
            continue
        asm[c["clip_id"]] = (arr, lay)
        pc = A.Pcm(arr, A.SR)
        want = c["pcm"]
        if pc.pcm_sha256() != want["pcm_sha256"] or [pc.channel_sha256(0), pc.channel_sha256(1)] != want["ch_sha256"] or pc.n_samples != want["n_samples"]:
            bad.append(f"{c['clip_id']}:pcm")
        lw = c["layout"]
        if (lay.user_onset, lay.user_len, lay.user_end, lay.total, lay.tail) != (lw["user_onset"], lw["user_len"], lw["user_end"], lw["total"], lw.get("tail", C.TRAIL)):
            bad.append(f"{c['clip_id']}:layout")
        if lw["trail"] != C.TRAIL or lw["lead_in"] != C.LEAD_IN:
            bad.append(f"{c['clip_id']}:constants")
    R.add("clips_reproduce_pcm_sha256", not bad and len(clips) > 0, bad[:20])

    # V4 pairs are exact splices of ONE recording --------------------------------------------------
    bad = []
    by_id = {c["clip_id"]: c for c in clips + rclips}
    for p in pairs + rpairs:
        try:
            ca, cb = by_id[p["clips"][0]], by_id[p["clips"][1]]
        except (KeyError, IndexError):
            bad.append(f"{p['pair_id']}:members")
            continue
        for k in ("split", "speaker_id", "dialogue_id", "family", "stratum"):
            if ca[k] != cb[k] or ca[k] != p.get(k, ca[k]):
                bad.append(f"{p['pair_id']}:member-mismatch:{k}")
        if ca["user"]["recording_id"] != cb["user"]["recording_id"] or p["user_recording"] != ca["user"]["recording_id"]:
            bad.append(f"{p['pair_id']}:not-one-recording")
        if ca["label"] == cb["label"]:
            bad.append(f"{p['pair_id']}:labels-not-contrastive")
        if {ca["label"], cb["label"]} != set(SYNTH_FAMILIES.get(p["family"], ())):
            bad.append(f"{p['pair_id']}:labels-vs-family")
        if ca["pair_role"] != "a" or cb["pair_role"] != "b":
            bad.append(f"{p['pair_id']}:roles")
        if ca["clip_id"] in asm and cb["clip_id"] in asm:
            viol = C.check_pair_invariants(p["family"], asm[ca["clip_id"]][0], asm[ca["clip_id"]][1], ca["user"]["cut"],
                                           asm[cb["clip_id"]][0], asm[cb["clip_id"]][1], cb["user"]["cut"])
            bad += [f"{p['pair_id']}:{v}" for v in viol]
    ref_ids = {x for p in pairs + rpairs for x in p["clips"]}
    if set(by_id) != ref_ids:
        bad.append("clips-vs-pairs:unreferenced-or-missing-clips")
    R.add("pairs_are_splices_of_one_recording", not bad and len(pairs) > 0, bad[:20])

    # V5 count ---------------------------------------------------------------------------------------
    synth_pairs = [p for p in pairs if p["stratum"] == "synthesized" and p.get("language") == "en"]
    # reserve pairs are NEVER counted unless backfill.json records a review rejection AND the backfill replacing it
    eff = len(synth_pairs) - len(backfill["rejections"]) + len(backfill["backfills"])
    R.add("min_100_english_pairs", eff >= MIN_PAIRS, {"main_pairs": len(synth_pairs), "review_rejections": len(backfill["rejections"]),
                                                        "backfills": len(backfill["backfills"]), "effective": eff, "need": MIN_PAIRS})

    # V6 split disjointness, DERIVED from rows ------------------------------------------------------
    sets = {"selection": {"speaker": set(), "dialogue": set(), "user_rec": set(), "asst_rec": set(), "pair": set()},
            "confirmation": {"speaker": set(), "dialogue": set(), "user_rec": set(), "asst_rec": set(), "pair": set()}}
    bad = []
    for c in clips + rclips:
        s = c["split"]
        if s not in sets:
            bad.append(f"{c['clip_id']}:split={s}")
            continue
        sets[s]["speaker"].add(c["speaker_id"])
        sets[s]["dialogue"].add(c["dialogue_id"])
        sets[s]["user_rec"].add(c["user"]["recording_id"])
        sets[s]["asst_rec"].add(c["assistant"]["recording_id"])
        sets[s]["pair"].add(c["pair_id"])
    for h in human:
        s = h.get("split")
        if s in sets:
            sets[s]["speaker"].add(h["speaker_id"])
            sets[s]["dialogue"].add(h["dialogue_id"])
    for k in ("speaker", "dialogue", "user_rec", "asst_rec", "pair"):
        inter = sets["selection"][k] & sets["confirmation"][k]
        if inter:
            bad.append(f"{k}-overlap:{sorted(inter)[:5]}")
    for sp in ("selection", "confirmation"):
        if not sets[sp]["speaker"]:
            bad.append(f"{sp}:empty")
    sj = D / "manifest" / "splits.json"
    if sj.exists():
        S = json.loads(sj.read_text())
        for sp in ("selection", "confirmation"):
            if set(S[sp]["speakers"]) != {c["speaker_id"] for c in clips if c["split"] == sp}:
                bad.append(f"splits.json:{sp}:speakers-differ")
            if set(S[sp]["pairs"]) != {c["pair_id"] for c in clips if c["split"] == sp}:
                bad.append(f"splits.json:{sp}:pairs-differ")
    else:
        bad.append("splits.json missing")
    R.add("selection_confirmation_disjoint", not bad, bad[:20])

    # V7 strata kept separate --------------------------------------------------------------------------
    bad = []
    for c in rclips:
        if c["stratum"] != "reserve":
            bad.append(f"{c['clip_id']}:stratum={c['stratum']} in reserve/clips.jsonl")
    for c in clips:
        if c["stratum"] != "synthesized":
            bad.append(f"{c['clip_id']}:stratum={c['stratum']} in clips.jsonl")
        if not str(c["speaker_id"]).startswith("synth-"):
            bad.append(f"{c['clip_id']}:synthesized clip with non-synth speaker")
    for h in human:
        if h.get("stratum") != "human_recorded":
            bad.append(f"human:{h.get('clip_id')}:stratum")
        if str(h.get("speaker_id", "")).startswith("synth-"):
            bad.append(f"human:{h.get('clip_id')}:synth speaker id")
    if {c["speaker_id"] for c in clips + rclips} & {h.get("speaker_id") for h in human}:
        bad.append("speaker-in-both-strata")
    if meta.get("advance_eligible_strata") != ADVANCE_ELIGIBLE:
        bad.append(f"advance_eligible_strata={meta.get('advance_eligible_strata')}")
    R.add("strata_separate_and_advance_policy", not bad, bad[:20])

    # V8 label contract ------------------------------------------------------------------------------------
    bad = []
    lc_path = label_contract_path or (D / "label_contract.json")
    if not lc_path.exists():
        bad.append("label_contract.json missing")
    else:
        lc = json.loads(lc_path.read_text())
        states = set(lc["turn_states"])
        if states != {"complete", "incomplete", "backchannel", "wait"}:
            bad.append(f"turn_states={sorted(states)}")
        for c in clips + rclips:
            if c["label"] not in states:
                bad.append(f"{c['clip_id']}:label={c['label']}")
            g = c["gold"]
            if c["label"] == "complete" and (g.get("eot_gold_s") is None or g.get("hold_window_s") is not None):
                bad.append(f"{c['clip_id']}:gold-vs-complete")
            if c["label"] in ("incomplete", "wait", "backchannel") and (g.get("eot_gold_s") is not None or not g.get("hold_window_s")):
                bad.append(f"{c['clip_id']}:gold-vs-{c['label']}")
            if c["label"] == "backchannel" and not g.get("assistant_speaking_at_user_onset"):
                bad.append(f"{c['clip_id']}:backchannel-without-assistant-floor")
        for h in human:
            if h.get("label") not in states:
                bad.append(f"human:{h.get('clip_id')}:label")
    R.add("label_contract_conformance", not bad, bad[:20])

    # V9 external material is hashes and ids only, outside the corpus --------------------------------------
    bad = []
    ext_p = D / "external_crosscheck.json"
    corpus_hashes = {r["pcm_sha256"] for r in all_recs.values()} | {r["file_sha256"] for r in recs.values()} | {c["pcm"]["pcm_sha256"] for c in clips + rclips}
    if not ext_p.exists():
        bad.append("external_crosscheck.json missing")
    else:
        ext = json.loads(ext_p.read_text())
        if ext.get("audio_copied_into_corpus") is not False:
            bad.append("audio_copied_into_corpus must be false")
        for s in ext.get("sets", []):
            for k in s:
                if k in ("path", "paths", "audio", "wav", "bytes_b64", "samples"):
                    bad.append(f"{s.get('set_id')}:forbidden-field:{k}")
            for fh in s.get("file_sha256", {}).values() if isinstance(s.get("file_sha256"), dict) else []:
                if fh in corpus_hashes:
                    bad.append(f"{s.get('set_id')}:hash-collides-with-corpus")
            if s.get("stored_in_corpus") is not False:
                bad.append(f"{s.get('set_id')}:stored_in_corpus")
    audio_dirs = {p.parts[0] + "/" + p.parts[1] for p in (Path(r) for r in tree_files(D)) if p.suffix in (".wav", ".flac", ".mp3", ".ogg", ".opus", ".parquet", ".tar") and len(p.parts) > 1}
    if audio_dirs - {"audio/synth", "audio/reserve"}:
        bad.append(f"audio outside audio/synth|reserve: {sorted(audio_dirs - {'audio/synth', 'audio/reserve'})}")
    R.add("external_material_hashes_and_ids_only", not bad, bad[:20])

    # V10 pins -----------------------------------------------------------------------------------------------
    bad = []
    pins_p = D / "pins.json"
    if not pins_p.exists():
        bad.append("pins.json missing")
    else:
        pins = json.loads(pins_p.read_text())
        sil = pins.get("silero_vad", {})
        file_ = D / sil.get("repo_path", "pins/silero_vad_v6.onnx")
        if not file_.exists() or sha256_file(file_) != sil.get("sha256"):
            bad.append("silero_vad:file-vs-pin")
        try:
            import tt_silero as TS

            if sil.get("sha256") != TS.PINNED_SHA256:
                bad.append("silero_vad:pins.json-vs-tt_silero.PINNED_SHA256")
        except Exception as e:  # noqa: BLE001
            bad.append(f"tt_silero import: {e}")
        for key, commit in (("full_duplex_bench", "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"), ("turnbench", "38a6f87")):
            if not str(pins.get(key, {}).get("commit", "")).startswith(commit[:7]) and pins.get(key, {}).get("commit") != commit:
                bad.append(f"{key}:commit")
        if "asr_word_timestamps" not in pins:
            bad.append("asr pin missing")
        else:
            asr = pins["asr_word_timestamps"]
            for k in ("tool_tree_commit", "binary_sha256", "model_sha256", "model_file"):
                if not asr.get(k):
                    bad.append(f"asr:{k}")
        era = D / "measurement_era_changes.json"
        if not era.exists():
            bad.append("measurement_era_changes.json missing")
        else:
            e = json.loads(era.read_text())
            if not e.get("changes"):
                bad.append("measurement_era_changes: empty")
            for ch in e.get("changes", []):
                pf = ch.get("patch_file")
                if pf and (not (D / pf).exists() or sha256_file(D / pf) != ch.get("patch_sha256")):
                    bad.append(f"{ch.get('id')}:patch-digest")
    R.add("instrument_pins_recorded_and_hash_checked", not bad, bad[:20])

    # V12 reserve stratum policy -------------------------------------------------------------------------------
    bad = []
    main_ids = {"pair": {p["pair_id"] for p in pairs}, "dialogue": {c["dialogue_id"] for c in clips}, "rec": set(recs)}
    spk_main = {c["speaker_id"]: c["split"] for c in clips}
    asst_main = {}
    for c in clips:
        asst_main.setdefault(c["assistant"]["recording_id"], c["split"])
    for p in rpairs:
        if p["stratum"] != "reserve":
            bad.append(f"{p['pair_id']}:stratum")
        if p["pair_id"] in main_ids["pair"] or p["dialogue_id"] in main_ids["dialogue"] or p["user_recording"] in main_ids["rec"]:
            bad.append(f"{p['pair_id']}:id-collides-with-main")
        if spk_main.get(p["speaker_id"]) != p["split"]:
            bad.append(f"{p['pair_id']}:speaker-split-differs-from-main({spk_main.get(p['speaker_id'])} vs {p['split']})")
    for c in rclips:
        if asst_main.get(c["assistant"]["recording_id"]) != c["split"]:
            bad.append(f"{c['clip_id']}:assistant-context-from-other-split")
    rsj = D / "manifest" / "reserve" / "splits.json"
    if rpairs:
        if not rsj.exists():
            bad.append("reserve/splits.json missing")
        else:
            rs = json.loads(rsj.read_text())
            for sp in ("selection", "confirmation"):
                if set(rs.get(sp, [])) != {p["pair_id"] for p in rpairs if p["split"] == sp}:
                    bad.append(f"reserve/splits.json:{sp}:differs")
    rej_ids = [x["pair_id"] for x in backfill["rejections"]]
    main_by = {p["pair_id"]: p for p in pairs}
    res_by = {p["pair_id"]: p for p in rpairs}
    if len(set(rej_ids)) != len(rej_ids):
        bad.append("duplicate review rejection")
    for x in backfill["rejections"]:
        if x["pair_id"] not in main_by:
            bad.append(f"rejection of unknown main pair {x['pair_id']}")
        if not x.get("reviewer") or not x.get("date"):
            bad.append(f"rejection {x['pair_id']} lacks reviewer/date")
    used_res = set()
    for b in backfill["backfills"]:
        rp, mp = res_by.get(b["reserve_pair_id"]), main_by.get(b["replaces_pair_id"])
        if rp is None:
            bad.append(f"backfill names unknown reserve pair {b['reserve_pair_id']}")
            continue
        if b["replaces_pair_id"] not in rej_ids or mp is None:
            bad.append(f"backfill replaces {b['replaces_pair_id']} which has no recorded review rejection")
            continue
        if (rp["family"], rp["split"]) != (mp["family"], mp["split"]):
            bad.append(f"backfill {b['reserve_pair_id']} differs from replaced pair in family/split")
        if b["reserve_pair_id"] in used_res:
            bad.append(f"reserve pair {b['reserve_pair_id']} backfilled twice")
        used_res.add(b["reserve_pair_id"])
    if len({b["replaces_pair_id"] for b in backfill["backfills"]}) != len(backfill["backfills"]):
        bad.append("a rejected pair is backfilled more than once")
    if meta.get("reserve", {}).get("policy_id") != RESERVE_POLICY_ID:
        bad.append("corpus.json lacks the reserve policy")
    if meta.get("reserve", {}).get("pairs") != len(rpairs):
        bad.append("corpus.json reserve pair count differs")
    R.add("reserve_stratum_policy", not bad, bad[:20])

    # V13 independent QA gates (ASR-B + energy) passed on EVERY clip, bound to the clip's PCM-SHA-256 -------------------
    bad = []
    qp = D / "qa" / "qa_gates_report.json"
    if not qp.exists():
        bad.append("qa/qa_gates_report.json missing")
    else:
        qa = json.loads(qp.read_text())
        if qa.get("gates_version") != "tt-qa-gates-2":
            bad.append(f"gates_version={qa.get('gates_version')}")
        asr_b, asr_a, scored = str(qa.get("asr_b", "")), str(qa.get("built_with_asr_a", "")), str(qa.get("scored_with", ""))
        if not asr_b or asr_b in (asr_a, scored) or "turbo" in asr_b:
            bad.append("ASR-B is not independent of the build ASR and the scoring ASR")
        if qa.get("all_pass") is not True:
            bad.append("report all_pass is not true")
        entries = qa.get("clips", {})
        allc = clips + rclips
        if len(entries) != len(allc):
            bad.append(f"report covers {len(entries)} clips, corpus has {len(allc)}")
        for c in allc:
            e = entries.get(c["clip_id"])
            if e is None:
                bad.append(f"{c['clip_id']}:not in qa report")
            elif e.get("pcm_sha256") != c["pcm"]["pcm_sha256"]:
                bad.append(f"{c['clip_id']}:qa report is for different audio")
            elif not e.get("pass") or not all(e.get("gates", {}).values()):
                bad.append(f"{c['clip_id']}:gate failed {[g for g, ok in e.get('gates', {}).items() if not ok]}")
    R.add("independent_qa_gates_passed", not bad, bad[:20])

    # V11 human slot exists and is honest -----------------------------------------------------------------------
    bad = []
    slot = D / "manifest" / "HUMAN_SLOT.md"
    if not slot.exists():
        bad.append("HUMAN_SLOT.md missing")
    for h in human:
        for k in ("clip_id", "speaker_id", "dialogue_id", "split", "label", "pcm"):
            if k not in h:
                bad.append(f"human:{h.get('clip_id')}:missing:{k}")
    R.add("human_recorded_slot", not bad, bad[:20])
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seal", "verify"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    D = Path(a.corpus)
    if a.cmd == "seal":
        m = seal(D)
        print(f"sealed {len(m['files'])} files; counts={m['counts']}")
        return 0
    R = verify(D)
    for c in R.checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"] + ("" if c["ok"] else f"  {json.dumps(c['detail'])[:400]}"))
    if a.json:
        print(json.dumps(R.checks))
    print("VERDICT", "PASS" if R.ok else "REFUSED")
    return 0 if R.ok else 1


if __name__ == "__main__":
    sys.exit(main())
